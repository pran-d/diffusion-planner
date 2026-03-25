#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from config.configure import load_config, get_data_path, get_norm_path, get_save_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset
from utils.eval_metrics import compute_metrics
from inference_mg import extract_initial_condition
from motion_generator import MotionGenerator


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_latest_checkpoint(save_dir: str, ema: bool = True) -> str:
    pattern = "ema_model_" if ema else "model_"
    best_epoch = -1
    best_path = None
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")
    for name in os.listdir(save_dir):
        if not (name.startswith(pattern) and name.endswith(".pth")):
            continue
        try:
            ep = int(name.replace(pattern, "").replace(".pth", ""))
        except ValueError:
            continue
        if ep > best_epoch:
            best_epoch = ep
            best_path = os.path.join(save_dir, name)
    if best_path is None:
        raise FileNotFoundError(f"No checkpoint found in {save_dir} ({pattern}*.pth)")
    return best_path


def _resolve_stitch_steps(cfg_data: dict, dataset: FlexibleWindowDataset, file_idx: int) -> int:
    eff = max(1, cfg_data["num_timesteps"] - cfg_data.get("state_history", 1))
    traj_len = int(dataset.traj_lengths[file_idx])
    return max(1, math.ceil(traj_len / eff))


def _save_quick_plots(out_dir: str, trajectories: np.ndarray, goals_world: np.ndarray, real_lengths: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(trajectories.shape[0]):
        rl = int(np.clip(real_lengths[i], 1, trajectories.shape[1]))
        xy = trajectories[i, :rl, 36:38]
        ax.plot(xy[:, 0], xy[:, 1], alpha=0.7, lw=1.2)
        ax.scatter(goals_world[i, 0], goals_world[i, 1], marker="x", s=30)
    ax.set_xlabel("obj_x")
    ax.set_ylabel("obj_y")
    ax.set_title("Object XY trajectories + goals")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "xy_overlay.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(trajectories.shape[0]):
        rl = int(np.clip(real_lengths[i], 1, trajectories.shape[1]))
        t = np.arange(rl) * 0.01
        ax.plot(t, trajectories[i, :rl, 38], alpha=0.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("obj_z")
    ax.set_title("Object Z profiles")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "z_profiles.png"), dpi=150)
    plt.close(fig)


def _select_fixed_test_indices(dataset: FlexibleWindowDataset, n: int, seed: int, test_indices_path: str) -> List[Tuple[int, int, int]]:
    if os.path.exists(test_indices_path):
        arr = np.load(test_indices_path)
        return [tuple(map(int, row.tolist())) for row in arr]

    rng = random.Random(seed)
    all_idx = list(dataset.indices)
    if n > len(all_idx):
        n = len(all_idx)
    chosen = rng.sample(all_idx, n)
    os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
    np.save(test_indices_path, np.asarray(chosen, dtype=np.int64))
    return chosen


def _build_temp_config(base_config_path: str, merged_ablation_cfg: dict) -> str:
    with open(base_config_path, "r", encoding="utf-8") as f:
        main_cfg = yaml.safe_load(f)

    # map ablation knobs -> model/data/noise cfg
    inf = merged_ablation_cfg.get("inference", {})
    speed = merged_ablation_cfg.get("speedup_tricks", {})

    main_cfg.setdefault("noise_scheduler", {})["inference_timesteps"] = int(inf.get("inference_timesteps", main_cfg["noise_scheduler"].get("inference_timesteps", 10)))
    main_cfg.setdefault("model", {})["use_fp16"] = bool(speed.get("use_fp16", False))
    main_cfg.setdefault("model", {})["compile_model"] = bool(speed.get("torch_compile", False))
    main_cfg.setdefault("model", {})["use_kv_cache"] = bool(speed.get("kv_cache", False))

    fd, temp_path = tempfile.mkstemp(prefix="abl_cfg_", suffix=".yaml")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(main_cfg, f, sort_keys=False)
    return temp_path


def run_single_ablation(
    merged_cfg: dict,
    ablation_name: str,
    output_dir: str,
    test_indices: List[Tuple[int, int, int]],
    base_model_config_path: str = "config/config.yaml",
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    temp_config = _build_temp_config(base_model_config_path, merged_cfg)
    try:
        mg = MotionGenerator(config_path=temp_config, device=merged_cfg.get("runtime", {}).get("device", "cuda"))
        data_cfg = mg.data_cfg

        data_path = get_data_path(data_cfg)
        norm_path = get_norm_path(mg.model_cfg, mg.training_cfg, data_cfg)
        data_buffer = preload_dataset(data_cfg, data_path)
        dataset = FlexibleWindowDataset(
            data_buffer=data_buffer,
            config=data_cfg,
            norm_path=norm_path,
            calculate_stats=False,
            training_cfg={},
        )
        mg.dataset = dataset

        ckpt_cfg = merged_cfg.get("checkpoint", {})
        epoch = ckpt_cfg.get("epoch", "latest")
        ema = bool(ckpt_cfg.get("ema", True))
        if isinstance(epoch, str) and os.path.exists(epoch):
            mg.diffuser.load_weights_from_file(epoch)
            ckpt_path = epoch
        elif str(epoch).lower() == "latest":
            save_dir = get_save_path(mg.model_cfg, mg.data_cfg, mg.training_cfg)
            ckpt_path = _find_latest_checkpoint(save_dir, ema=ema)
            mg.diffuser.load_weights_from_file(ckpt_path)
        else:
            mg.diffuser.loadWeights(int(epoch), ema=ema)
            ckpt_path = str(epoch)

        inf = merged_cfg.get("inference", {})
        speed = merged_cfg.get("speedup_tricks", {})

        traj_list = []
        goal_list = []
        ref_list = []
        rl_list = []
        per_sample_metrics = []

        for i, target in enumerate(test_indices):
            sample_idx = dataset.indices.index(target)
            local_goal_dim = min(3, data_cfg.get("num_task_params", 3))
            initial_condition, goal_local, anchor, file_idx, _, _ = extract_initial_condition(dataset, sample_idx, local_goal_dim)

            stitch_steps = inf.get("stitch_steps", None)
            if stitch_steps is None:
                stitch_steps = _resolve_stitch_steps(data_cfg, dataset, file_idx)

            t0 = time.perf_counter()
            out = mg.generate_trajectory(
                initial_condition=initial_condition,
                goal_condition=goal_local,
                stitch_steps=int(stitch_steps),
                num_samples=1,
                cfg_w=float(inf.get("cfg_w", 1.0)),
                enable_goal_stop=bool(inf.get("enable_goal_stop", True)),
                end_error_threshold=float(inf.get("end_error_threshold", 0.1)),
                enable_physics_stop=bool(inf.get("enable_physics_stop", False)),
                enable_physics_clamp=bool(inf.get("enable_physics_clamp", True)),
                use_last_frame_wp=bool(inf.get("use_last_frame_wp", True)),
                arrival_ratio=float(inf.get("arrival_ratio", 0.85)),
                lift_height=float(inf.get("lift_height", 0.5)),
                no_lower_dist=float(inf.get("no_lower_dist", 0.4)),
                lift_start=float(inf.get("lift_start", 0.0)),
                lift_end=float(inf.get("lift_end", 0.20)),
                walk_start_z=float(inf.get("walk_start_z", 0.80)),
                verbose_timing=True,
                return_timing_stats=True,
                use_fp16=bool(speed.get("use_fp16", False)),
                compile_model=bool(speed.get("torch_compile", False)),
            )
            infer_wall = time.perf_counter() - t0
            full_trajectory, real_lengths, timing = out

            traj = full_trajectory[0]
            rl = int(real_lengths[0])
            goal_world = anchor["final_obj_pos"][:3]
            ref_obj = anchor["ref_obj_pos"][:3]

            mm = compute_metrics(
                traj,
                goal_world=goal_world,
                ref_obj_pos=ref_obj,
                real_length=rl,
                inference_time_s=float(timing.get("total_time_s", infer_wall)),
                gpu_peak_mb=float(timing.get("gpu_peak_mb", 0.0)),
            )
            mm["sample_index"] = int(i)
            mm["dataset_index"] = tuple(map(int, target))
            mm["avg_step_time_s"] = float(timing.get("avg_step_time_s", 0.0))
            mm["avg_forward_time_s"] = float(timing.get("avg_forward_time_s", 0.0))
            mm["avg_reconstruction_time_s"] = float(timing.get("avg_reconstruction_time_s", 0.0))
            per_sample_metrics.append(mm)

            traj_list.append(traj)
            rl_list.append(rl)
            goal_list.append(goal_world)
            ref_list.append(ref_obj)

        trajectories = np.stack(traj_list, axis=0)
        real_lengths = np.asarray(rl_list, dtype=np.int64)
        goals_world = np.asarray(goal_list, dtype=np.float64)
        ref_obj_pos = np.asarray(ref_list, dtype=np.float64)

        np.save(os.path.join(output_dir, "trajectories.npy"), trajectories)
        np.save(os.path.join(output_dir, "goals.npy"), goals_world)
        np.save(os.path.join(output_dir, "real_lengths.npy"), real_lengths)
        np.save(os.path.join(output_dir, "ref_obj_pos.npy"), ref_obj_pos)

        _save_quick_plots(output_dir, trajectories, goals_world, real_lengths)

        arr_err = np.asarray([m["l2_final_error"] for m in per_sample_metrics], dtype=float)
        arr_t = np.asarray([m.get("inference_time_s", 0.0) for m in per_sample_metrics], dtype=float)
        arr_mem = np.asarray([m.get("gpu_peak_mb", 0.0) for m in per_sample_metrics], dtype=float)
        summary = {
            "ablation_name": ablation_name,
            "checkpoint": ckpt_path,
            "num_samples": int(len(per_sample_metrics)),
            "l2_goal_error_mean": float(np.mean(arr_err)),
            "l2_goal_error_std": float(np.std(arr_err)),
            "success_rate_0.05": float(np.mean([m["success_at_0.05m"] for m in per_sample_metrics])),
            "success_rate_0.10": float(np.mean([m["success_at_0.10m"] for m in per_sample_metrics])),
            "success_rate_0.15": float(np.mean([m["success_at_0.15m"] for m in per_sample_metrics])),
            "wall_clock_time_per_traj_s": float(np.mean(arr_t)),
            "gpu_peak_mb": float(np.max(arr_mem) if len(arr_mem) else 0.0),
            "avg_inference_step_time_s": float(np.mean([m.get("avg_step_time_s", 0.0) for m in per_sample_metrics])),
            "planning_horizon_s": float(max(1, data_cfg["num_timesteps"] - data_cfg.get("state_history", 1)) * 0.01),
            "window_size": int(data_cfg["num_timesteps"]),
            "state_history": int(data_cfg.get("state_history", 1)),
            "mujoco_timestep_s": 0.01,
        }

        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(output_dir, "per_sample_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(per_sample_metrics, f, indent=2)

        return summary

    finally:
        if os.path.exists(temp_config):
            os.unlink(temp_config)


def parse_args():
    p = argparse.ArgumentParser("Run config-driven inference ablations")
    p.add_argument("--config_dir", type=str, default="inference_configs")
    p.add_argument("--output_dir", type=str, default="results/ablations")
    p.add_argument("--base_config", type=str, default="inference_configs/base.yaml")
    p.add_argument("--single_config", type=str, default=None, help="Run one ablation YAML only")
    p.add_argument("--test_indices", type=str, default=None, help="Optional fixed indices .npy")
    p.add_argument("--skip_compare", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base_cfg = _load_yaml(args.base_config)

    config_paths: List[str]
    if args.single_config:
        config_paths = [args.single_config]
    else:
        config_paths = sorted(
            [
                str(p) for p in Path(args.config_dir).glob("*.yaml")
                if p.name != Path(args.base_config).name
            ]
        )

    if not config_paths:
        raise RuntimeError("No ablation config YAML files found.")

    # Build one dataset view from base-main config only for index selection
    mg_probe = MotionGenerator(config_path="config/config.yaml", device=base_cfg.get("runtime", {}).get("device", "cuda"))
    data_path = get_data_path(mg_probe.data_cfg)
    norm_path = get_norm_path(mg_probe.model_cfg, mg_probe.training_cfg, mg_probe.data_cfg)
    data_buffer = preload_dataset(mg_probe.data_cfg, data_path)
    ds_probe = FlexibleWindowDataset(
        data_buffer=data_buffer,
        config=mg_probe.data_cfg,
        norm_path=norm_path,
        calculate_stats=False,
        training_cfg={},
    )

    n_test = int(base_cfg.get("num_test_samples", 20))
    seed = int(base_cfg.get("seed", 42))
    test_indices_path = args.test_indices or os.path.join(args.output_dir, "test_indices.npy")
    test_indices = _select_fixed_test_indices(ds_probe, n_test, seed, test_indices_path)

    timing_rows = []
    for cfg_path in config_paths:
        override = _load_yaml(cfg_path)
        merged = _deep_merge(base_cfg, override)
        ablation_name = merged.get("name") or Path(cfg_path).stem
        out_dir = os.path.join(args.output_dir, ablation_name)

        print(f"\n=== Running ablation: {ablation_name} ===")
        t0 = time.perf_counter()
        summary = run_single_ablation(
            merged_cfg=merged,
            ablation_name=ablation_name,
            output_dir=out_dir,
            test_indices=test_indices,
        )
        elapsed = time.perf_counter() - t0
        summary["ablation_wall_time_s"] = float(elapsed)
        timing_rows.append(summary)
        print(f"Done {ablation_name}: error_mean={summary['l2_goal_error_mean']:.4f}, time/trajectory={summary['wall_clock_time_per_traj_s']:.4f}s")

    timing_csv = os.path.join(args.output_dir, "timing_summary.csv")
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ablation_name", "l2_goal_error_mean", "l2_goal_error_std",
                "success_rate_0.10", "wall_clock_time_per_traj_s",
                "avg_inference_step_time_s", "planning_horizon_s", "gpu_peak_mb",
                "ablation_wall_time_s",
            ],
        )
        writer.writeheader()
        for row in timing_rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    print(f"Saved timing summary: {timing_csv}")

    if not args.skip_compare:
        cmd = ["python", "compare_ablations.py", "--output_root", args.output_dir]
        print("Running compare_ablations.py...")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()

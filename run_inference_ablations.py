#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
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

from config.configure import load_config, get_data_path, get_mj_xml_paths, get_norm_path, get_save_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset
from utils.eval_metrics import compute_metrics
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
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
        best_name = "ema_model_best.pth" if ema else "model_best.pth"
        best_candidate = os.path.join(save_dir, best_name)
        if os.path.exists(best_candidate):
            return best_candidate
    if best_path is None:
        raise FileNotFoundError(f"No checkpoint found in {save_dir} ({pattern}*.pth)")
    return best_path


def _resolve_stitch_steps(cfg_data: dict, dataset: FlexibleWindowDataset, file_idx: int) -> int:
    eff = max(1, cfg_data["num_timesteps"] - cfg_data.get("state_history", 1))
    traj_len = int(dataset.traj_lengths[file_idx])
    return max(1, math.ceil(traj_len / eff))


def _pad_trajectories(traj_list: List[np.ndarray]) -> np.ndarray:
    """Pad variable-length (T, D) trajectories to a single (N, T_max, D) array.

    Padding repeats the last valid frame per sample, while true lengths are kept
    separately in ``real_lengths``.
    """
    if not traj_list:
        raise ValueError("traj_list is empty")

    feat_dim = int(traj_list[0].shape[-1])
    max_t = max(int(tr.shape[0]) for tr in traj_list)
    batch = np.empty((len(traj_list), max_t, feat_dim), dtype=traj_list[0].dtype)

    for i, tr in enumerate(traj_list):
        if tr.ndim != 2:
            raise ValueError(f"Expected trajectory with shape (T, D), got {tr.shape} at index {i}")
        if tr.shape[1] != feat_dim:
            raise ValueError(
                f"Feature dimension mismatch at index {i}: expected {feat_dim}, got {tr.shape[1]}"
            )

        t = int(tr.shape[0])
        batch[i, :t, :] = tr
        if t < max_t:
            batch[i, t:, :] = tr[-1]

    return batch


def _save_mujoco_video(
    traj: np.ndarray,
    out_path: str,
    xml_path: str,
    fps: int = 100,
    width: int = 1280,
    height: int = 720,
    track_pelvis: bool = True,
) -> None:
    """Render one trajectory to MP4 using offscreen MuJoCo renderer."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import imageio
    import mujoco

    traj = np.asarray(traj)
    if traj.ndim != 2:
        raise ValueError(f"Expected trajectory shape (T, D), got {traj.shape}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, int(width))
    model.vis.global_.offheight = max(model.vis.global_.offheight, int(height))

    renderer = mujoco.Renderer(model, height=int(height), width=int(width))

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    if track_pelvis:
        try:
            pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = pelvis_id
        except Exception:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    nq = int(model.nq)
    nv = int(model.nv)
    with imageio.get_writer(out_path, fps=int(fps)) as writer:
        for i in range(traj.shape[0]):
            data.qpos[:] = traj[i, :nq]
            if traj.shape[1] >= nq + nv and nv > 0:
                data.qvel[:] = traj[i, nq:nq + nv]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            writer.append_data(frame)

    renderer.close()


def _build_adaptive_xml(base_repeated_xml: str, num_robots: int) -> str:
    """Create a temporary repeated-robot XML with exactly ``num_robots`` copies."""
    with open(base_repeated_xml, "r", encoding="utf-8") as f:
        lines = f.readlines()

    robot_starts = [
        idx for idx, line in enumerate(lines)
        if re.search(r'body name="pelvis_\d+"', line)
    ]
    if not robot_starts:
        raise ValueError(f"No pelvis_N bodies found in {base_repeated_xml}")

    header = "".join(lines[: robot_starts[0]])
    wb_close_idx = next((i for i, l in enumerate(lines) if "</worldbody>" in l), len(lines))
    block_end = robot_starts[1] if len(robot_starts) >= 2 else wb_close_idx
    template = "".join(lines[robot_starts[0]: block_end])
    footer = "".join(lines[wb_close_idx:])

    robot_blocks = [re.sub(r'_0(?=")', f'_{i}', template) for i in range(num_robots)]
    xml_content = header + "".join(robot_blocks) + footer

    source_dir = os.path.dirname(os.path.abspath(base_repeated_xml))
    fd, temp_xml = tempfile.mkstemp(prefix=f"abl_overlay_{num_robots}r_", suffix=".xml", dir=source_dir)
    os.close(fd)
    with open(temp_xml, "w", encoding="utf-8") as f:
        f.write(xml_content)
    return temp_xml


def _save_overlay_video(
    trajectories: np.ndarray,
    real_lengths: np.ndarray,
    out_path: str,
    single_xml_path: str,
    repeated_xml_path: str,
    fps: int = 100,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Save one MP4 with all trajectories overlaid in a repeated-robot scene.

    Rendering runs in an isolated subprocess so GL backend can be selected
    before importing `mujoco` (critical on headless clusters).
    """
    x = np.asarray(trajectories)
    rl = np.asarray(real_lengths, dtype=np.int64)
    if x.ndim != 3:
        raise ValueError(f"Expected trajectories shape (N, T, D), got {x.shape}")

    with tempfile.TemporaryDirectory(prefix="abl_overlay_render_") as td:
        payload = os.path.join(td, "payload.npz")
        np.savez_compressed(payload, trajectories=x, real_lengths=rl)

        render_code = r'''
import os, re, sys, tempfile
import numpy as np
import imageio
import mujoco

payload, out_path, single_xml, repeated_xml, fps, width, height = sys.argv[1:]
fps = int(fps); width = int(width); height = int(height)
arr = np.load(payload)
x = arr["trajectories"]
rl = arr["real_lengths"].astype(np.int64)

single_model = mujoco.MjModel.from_xml_path(single_xml)
nq_single = int(single_model.nq)
num_robots = int(x.shape[0])

with open(repeated_xml, "r", encoding="utf-8") as f:
    lines = f.readlines()
robot_starts = [i for i, line in enumerate(lines) if re.search(r'body name="pelvis_\d+"', line)]
if not robot_starts:
    raise ValueError(f"No pelvis_N bodies found in {repeated_xml}")
header = "".join(lines[: robot_starts[0]])
wb_close_idx = next((i for i, l in enumerate(lines) if "</worldbody>" in l), len(lines))
block_end = robot_starts[1] if len(robot_starts) >= 2 else wb_close_idx
template = "".join(lines[robot_starts[0]: block_end])
footer = "".join(lines[wb_close_idx:])
robot_blocks = [re.sub(r'_0(?=")', f'_{i}', template) for i in range(num_robots)]
xml_content = header + "".join(robot_blocks) + footer

src_dir = os.path.dirname(os.path.abspath(repeated_xml))
fd, adaptive_xml = tempfile.mkstemp(prefix=f"abl_overlay_{num_robots}r_", suffix=".xml", dir=src_dir)
os.close(fd)
with open(adaptive_xml, "w", encoding="utf-8") as f:
    f.write(xml_content)

renderer = None
try:
    model = mujoco.MjModel.from_xml_path(adaptive_xml)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.0, 0.0, 0.9]
    cam.distance = 5.0
    cam.azimuth = 135.0
    cam.elevation = -30.0

    T_max = int(x.shape[1])
    nq_used = num_robots * nq_single
    if nq_used > model.nq:
        raise ValueError(f"Repeated XML nq={model.nq} is smaller than required {nq_used}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for t in range(T_max):
            q_blocks = []
            for i in range(num_robots):
                ti = min(t, int(max(1, rl[i])) - 1)
                q_blocks.append(x[i, ti, :nq_single])
            data.qpos[:nq_used] = np.concatenate(q_blocks, axis=0)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            writer.append_data(frame)
finally:
    try:
        if renderer is not None:
            renderer.close()
    except Exception:
        pass
    try:
        if os.path.exists(adaptive_xml):
            os.unlink(adaptive_xml)
    except Exception:
        pass
'''

        errors = []
        for backend, pyopengl in (("egl", "egl"), ("osmesa", "osmesa")):
            env = os.environ.copy()
            env["MUJOCO_GL"] = backend
            env["PYOPENGL_PLATFORM"] = pyopengl
            env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
            env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
            env.pop("DISPLAY", None)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    render_code,
                    payload,
                    out_path,
                    single_xml_path,
                    repeated_xml_path,
                    str(int(fps)),
                    str(int(width)),
                    str(int(height)),
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                return
            errors.append(
                f"backend={backend}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        raise RuntimeError("Overlay render failed for EGL and OSMesa backends.\n" + "\n---\n".join(errors))


def _generate_goal_offsets_3ring(num_goals: int, goal_spread: float) -> np.ndarray:
    """Generate N displacement vectors on 3 cyclic radii (batch_goal_sweep style)."""
    radii = [goal_spread / 3.0, 2.0 * goal_spread / 3.0, goal_spread]
    thetas = np.linspace(0.0, 2.0 * np.pi, num_goals, endpoint=False)
    goals = []
    for i, theta in enumerate(thetas):
        r = radii[i % len(radii)]
        goals.append(np.array([
            r * np.cos(theta),
            r * np.sin(theta),
            0.0,
        ], dtype=np.float64))
    return np.stack(goals, axis=0)


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
    if "scheduling_matrix" in inf:
        main_cfg.setdefault("noise_scheduler", {})["scheduling_matrix"] = inf.get("scheduling_matrix")
    if "hierarchical_noise" in inf:
        main_cfg.setdefault("noise_scheduler", {})["hierarchical_noise"] = inf.get("hierarchical_noise")
    if "hierarchical_schedule" in inf:
        main_cfg.setdefault("noise_scheduler", {})["hierarchical_schedule"] = inf.get("hierarchical_schedule")
    main_cfg.setdefault("model", {})["use_fp16"] = bool(speed.get("use_fp16", False))
    main_cfg.setdefault("model", {})["compile_model"] = bool(speed.get("torch_compile", False))
    main_cfg.setdefault("model", {})["use_kv_cache"] = bool(speed.get("kv_cache", False))

    base_dir = os.path.dirname(os.path.abspath(base_config_path))
    fd, temp_path = tempfile.mkstemp(prefix="abl_cfg_", suffix=".yaml", dir=base_dir)
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
    save_videos: bool = False,
    videos_per_ablation: int = 1,
    video_fps: int = 100,
    video_width: int = 1280,
    video_height: int = 720,
    overlay_video: bool = True,
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

        local_goal_dim = min(3, data_cfg.get("num_task_params", 3))
        prepared = []
        for i, target in enumerate(test_indices):
            sample_idx = dataset.indices.index(target)
            initial_condition, _goal_local_data, anchor, file_idx, _, _ = extract_initial_condition(
                dataset, sample_idx, local_goal_dim
            )

            base = np.asarray(initial_condition["robot"][0, 3:7], dtype=np.float64)
            ref_obj = np.asarray(initial_condition["obj"][0, :3], dtype=np.float64)
            yaw = yaw_from_quat(base)
            R_world_to_local = yaw_to_rot_matrix(-yaw)

            prepared.append({
                "i": int(i),
                "target": tuple(map(int, target)),
                "ic": initial_condition,
                "anchor": anchor,
                "file_idx": int(file_idx),
                "ref_obj": ref_obj,
                "R_world_to_local": R_world_to_local,
            })

        n_eval = len(prepared)
        goal_spread = float(inf.get("goal_spread", 0.5))
        goal_offsets = _generate_goal_offsets_3ring(n_eval, goal_spread)

        per_item_stitch = []
        for item in prepared:
            st = inf.get("stitch_steps", None)
            if st is None:
                st = _resolve_stitch_steps(data_cfg, dataset, item["file_idx"])
            per_item_stitch.append(int(st))
        stitch_steps = int(max(per_item_stitch)) if per_item_stitch else 1

        runtime_cfg = merged_cfg.get("runtime", {})
        batch_size = int(runtime_cfg.get("batch_size", max(1, n_eval)))

        for b_start in range(0, n_eval, batch_size):
            b_end = min(b_start + batch_size, n_eval)
            chunk = prepared[b_start:b_end]
            bs = len(chunk)
            print(f"[batch] {b_start}:{b_end} / {n_eval} (bs={bs})")

            ic_batch = {
                "robot": np.stack([it["ic"]["robot"] for it in chunk], axis=0),
                "obj": np.stack([it["ic"]["obj"] for it in chunk], axis=0),
            }

            goals_local_batch = np.zeros((bs, local_goal_dim), dtype=np.float64)
            goals_world_batch = np.zeros((bs, 3), dtype=np.float64)
            for j, it in enumerate(chunk):
                gw = it["ref_obj"] + goal_offsets[b_start + j]
                goals_world_batch[j] = gw
                delta_w = gw - it["ref_obj"]
                delta_l = it["R_world_to_local"] @ delta_w
                goals_local_batch[j] = delta_l[:local_goal_dim]

            t0 = time.perf_counter()
            full_trajectory, real_lengths, timing = mg.generate_trajectory(
                initial_condition=ic_batch,
                goal_condition=goals_local_batch,
                stitch_steps=stitch_steps,
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
            per_traj_wall = infer_wall / max(1, bs)

            for j, it in enumerate(chunk):
                traj = full_trajectory[j]
                rl = int(real_lengths[j])
                goal_world = goals_world_batch[j]
                ref_obj = it["ref_obj"]

                mm = compute_metrics(
                    traj,
                    goal_world=goal_world,
                    ref_obj_pos=ref_obj,
                    real_length=rl,
                    inference_time_s=float(per_traj_wall),
                    gpu_peak_mb=float(timing.get("gpu_peak_mb", 0.0)),
                )
                mm["sample_index"] = int(it["i"])
                mm["dataset_index"] = it["target"]
                mm["avg_step_time_s"] = float(timing.get("avg_step_time_s", 0.0))
                mm["avg_forward_time_s"] = float(timing.get("avg_forward_time_s", 0.0))
                mm["avg_reconstruction_time_s"] = float(timing.get("avg_reconstruction_time_s", 0.0))
                per_sample_metrics.append(mm)

                traj_list.append(traj)
                rl_list.append(rl)
                goal_list.append(goal_world)
                ref_list.append(ref_obj)

        trajectories = _pad_trajectories(traj_list)1
        real_lengths = np.asarray(rl_list, dtype=np.int64)
        goals_world = np.asarray(goal_list, dtype=np.float64)
        ref_obj_pos = np.asarray(ref_list, dtype=np.float64)

        np.save(os.path.join(output_dir, "trajectories.npy"), trajectories)
        np.save(os.path.join(output_dir, "goals.npy"), goals_world)
        np.save(os.path.join(output_dir, "real_lengths.npy"), real_lengths)
        np.save(os.path.join(output_dir, "ref_obj_pos.npy"), ref_obj_pos)

        if save_videos:
            # Reduce Torch/CUDA interaction with MuJoCo renderer in the same process.
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            try:
                del dataset
                del data_buffer
                del mg
            except Exception:
                pass
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            xml_path, repeated_xml_path = get_mj_xml_paths()
            videos_dir = os.path.join(output_dir, "videos")
            os.makedirs(videos_dir, exist_ok=True)

            if overlay_video:
                if not os.path.exists(xml_path) or not os.path.exists(repeated_xml_path):
                    print(
                        f"[video] Missing XML for overlay video (single={xml_path}, repeated={repeated_xml_path}); skipping."
                    )
                else:
                    overlaid_path = os.path.join(videos_dir, "overlaid_robots.mp4")
                    try:
                        _save_overlay_video(
                            trajectories=trajectories,
                            real_lengths=real_lengths,
                            out_path=overlaid_path,
                            single_xml_path=xml_path,
                            repeated_xml_path=repeated_xml_path,
                            fps=video_fps,
                            width=video_width,
                            height=video_height,
                        )
                        print(f"[video] Saved {overlaid_path}")
                    except Exception as e:
                        print(f"[video] Overlay export failed: {e}")
            else:
                n_vid = min(int(videos_per_ablation), len(traj_list))
                for vi in range(n_vid):
                    rl = int(np.clip(rl_list[vi], 1, traj_list[vi].shape[0]))
                    traj_vis = traj_list[vi][:rl]
                    vid_path = os.path.join(videos_dir, f"sample_{vi:03d}.mp4")
                    try:
                        _save_mujoco_video(
                            traj=traj_vis,
                            out_path=vid_path,
                            xml_path=xml_path,
                            fps=video_fps,
                            width=video_width,
                            height=video_height,
                        )
                        print(f"[video] Saved {vid_path}")
                    except Exception as e:
                        print(f"[video] Failed for sample {vi}: {e}")

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
    p.add_argument("--include_base", action="store_true", help="Include base_config in multi-config runs")
    p.add_argument("--skip_compare", action="store_true")
    p.add_argument("--save_videos", action="store_true", help="Save MuJoCo MP4 videos in each ablation folder")
    p.add_argument("--videos_per_ablation", type=int, default=1, help="How many sample videos to save per ablation")
    p.add_argument("--video_fps", type=int, default=100)
    p.add_argument("--video_width", type=int, default=1280)
    p.add_argument("--video_height", type=int, default=720)
    p.add_argument("--overlay_video", action="store_true", help="Save a single overlaid-robots MuJoCo video")
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
                if args.include_base or p.name != Path(args.base_config).name
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
            save_videos=bool(args.save_videos),
            videos_per_ablation=int(args.videos_per_ablation),
            video_fps=int(args.video_fps),
            video_width=int(args.video_width),
            video_height=int(args.video_height),
            overlay_video=bool(args.overlay_video or args.save_videos),
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

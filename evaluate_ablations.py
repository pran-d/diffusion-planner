#!/usr/bin/env python3
"""
evaluate_ablations.py

Evaluate every ablation checkpoint using batch_goal_sweep, then produce
per-ablation summary plots:
  1. Error bar chart  (mean ± std across goals for each ablation)
  2. Desired vs achieved displacement scatter
  3. Per-goal polar error plot

Results are saved under:
    <output_root>/<ablation_name>/
        ├── trajectories.npy, goals_world.npy, errors.npy, …
        ├── error_bar.png
        ├── displacement_scatter.png
        └── polar_error.png

Usage:
    python evaluate_ablations.py --epoch 200 --ablations_folder config/ablations/ \
        --num_goals 12 --goal_spread 0.5 --output_root results/ablation_eval
"""

import argparse
import glob
import math
import os
import re
import shutil
import sys
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

from config.configure import (
    load_config,
    get_save_path,
    get_data_path,
    get_norm_path,
    get_mj_xml_paths,
)
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
from motion_generator import MotionGenerator
from utils.inference_config import load_inference_defaults
from utils.inference_utils import DEFAULT_LIFT_HEIGHT
from batch_goal_sweep import run_goal_sweep


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (same suffix logic as run_ablations.py)
# ──────────────────────────────────────────────────────────────────────────────

MAIN_CONFIG_FILE = "config/config.yaml"


def derive_suffix_from_filename(cfg_path: str) -> str | None:
    stem = os.path.splitext(os.path.basename(cfg_path))[0]
    m = re.fullmatch(r"config(_.*)?", stem)
    if m is None:
        return None
    return m.group(1) or ""


def patch_suffix_in_config(cfg_path: str, suffix: str) -> str:
    """Return path to a temp YAML with training.suffix set to *suffix*."""
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return cfg_path
    data.setdefault("training", {})["suffix"] = suffix
    tmp_path = cfg_path + ".eval_patched.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return tmp_path


def find_latest_checkpoint(save_dir: str, ema: bool = False) -> tuple[str | None, int]:
    """
    Scan *save_dir* for model checkpoints and return the path and epoch of
    the latest one.  Returns (None, -1) if nothing is found.
    """
    pattern = "ema_model_*.pth" if ema else "model_*.pth"
    hits = glob.glob(os.path.join(save_dir, pattern))
    if not hits:
        return None, -1
    epoch_map = {}
    for h in hits:
        m = re.search(r"(?:ema_)?model_(\d+)\.pth$", os.path.basename(h))
        if m:
            epoch_map[int(m.group(1))] = h
    if not epoch_map:
        return None, -1
    best_epoch = max(epoch_map)
    return epoch_map[best_epoch], best_epoch


def safe_ablation_name(cfg_path: str, ablations_folder: str) -> str:
    """Derive a filesystem-safe name for the ablation."""
    rel = os.path.relpath(cfg_path, ablations_folder)
    name = os.path.splitext(rel)[0].replace(os.sep, "__")
    return name


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_error_bar(results: dict, out_path: str):
    """Bar chart of mean error ± std for each ablation."""
    names = list(results.keys())
    means = [np.mean(results[n]["errors"]) for n in names]
    stds  = [np.std(results[n]["errors"])  for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Goal Error (m)")
    ax.set_title("Goal-Reaching Error per Ablation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved error bar chart → {out_path}")


def plot_displacement_scatter(results: dict, out_path: str):
    """
    Scatter of desired (x) vs achieved (y) displacement magnitude for
    every ablation on the same axes, colour-coded.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.get_cmap("tab10", max(len(results), 1))

    all_vals = []
    for idx, (name, r) in enumerate(results.items()):
        des = np.linalg.norm(r["desired_displacements"], axis=-1)
        ach = np.linalg.norm(r["achieved_displacements"], axis=-1)
        ax.scatter(des, ach, label=name, s=30, alpha=0.7, color=cmap(idx))
        all_vals.extend(des.tolist())
        all_vals.extend(ach.tolist())

    if all_vals:
        lim = max(all_vals) * 1.15
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="ideal")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    ax.set_xlabel("Desired displacement (m)")
    ax.set_ylabel("Achieved displacement (m)")
    ax.set_title("Desired vs Achieved Object Displacement")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved displacement scatter → {out_path}")


def plot_polar_error(result: dict, name: str, out_path: str):
    """
    Polar plot of per-goal error — angle = direction from ref_obj to goal,
    radius = error.
    """
    goals_world = result["goals_world"]
    ref = result["ref_obj_pos"][:2]
    errors = result["errors"]
    N = len(goals_world)

    thetas = np.arctan2(goals_world[:, 1] - ref[1], goals_world[:, 0] - ref[0])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.scatter(thetas, errors, c="crimson", s=40, zorder=5)
    for i in range(N):
        ax.plot([thetas[i], thetas[i]], [0, errors[i]], c="grey", lw=0.6)
    ax.set_title(f"Per-Goal Error — {name}", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved polar error plot → {out_path}")


def plot_per_ablation_displacement(result: dict, name: str, out_path: str):
    """
    2-D quiver showing desired (blue) and achieved (red) displacement vectors
    for a single ablation.
    """
    desired = result["desired_displacements"]
    achieved = result["achieved_displacements"]

    fig, ax = plt.subplots(figsize=(6, 6))
    N = len(desired)
    for i in range(N):
        ax.annotate("", xy=desired[i], xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2))
        ax.annotate("", xy=achieved[i], xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
    ax.scatter(*desired.T, c="steelblue", s=25, label="desired", zorder=5)
    ax.scatter(*achieved.T, c="crimson",  s=25, label="achieved", zorder=5)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("dy (m)")
    ax.legend(fontsize=8)
    ax.set_title(f"Displacement Vectors — {name}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved displacement vectors → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Physical consistency metrics
# ──────────────────────────────────────────────────────────────────────────────
# Trajectory layout (43-dim, world frame):
#   [0:3]   base xyz          [3:7]   base quat (wxyz)
#   [7:36]  29 joint positions
#   [36:39] object xyz         [39:43] object quat (wxyz)
#
# Mitigation suggestions:
#   1) Robot spinning — Add a **yaw-rate penalty** during training
#      (L_yaw = ||Δyaw_t||² for consecutive frames) or clamp the per-frame
#      yaw change at inference time in the physics-clamp post-processor.
#      Alternatively, inject a soft waypoint constraining yaw at key frames.
#   2) Object floating from hands — Enforce a **hand-object proximity loss**
#      during training that penalises the distance between wrist sites and the
#      object centroid whenever the object is above a rest-z threshold.
#      At inference time, a lightweight FK-based clamp could snap the object
#      back between the hands when it drifts beyond a radius.
#   3) General: Using **classifier-free guidance** (cfg_w > 1) on the object
#      channels only, or history-guidance on the full state, helps keep the
#      generated motions closer to the training distribution and reduces
#      physically implausible artefacts.
# ──────────────────────────────────────────────────────────────────────────────

def _wrap_angle(a):
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_physics_metrics(result: dict, mj_xml_path: str = None):
    """
    Compute per-goal physical consistency metrics from full_traj.

    Returns a dict added to *result* in-place with keys:
        yaw_jump_max      (N,)  — max |Δyaw| across frames per goal
        yaw_jump_mean     (N,)  — mean |Δyaw| across frames per goal
        yaw_jump_series   list of (T-1,) arrays — per-frame |Δyaw|
        hand_obj_dist_max (N,)  — max hand-object distance while lifted  (NaN if no FK)
        hand_obj_dist_mean(N,)  — mean hand-object distance while lifted (NaN if no FK)
        hand_obj_series   list of (T,) arrays — per-frame mean hand-obj dist
        obj_rest_z        float — object z at t=0
    """
    full_traj = result["full_traj"]        # (N, T, 43)
    real_lengths = result["real_lengths"]   # (N,)
    N, T, _ = full_traj.shape

    # ── 1. Robot yaw jumps ────────────────────────────────────────────────
    base_quat = full_traj[:, :, 3:7]           # (N, T, 4) wxyz
    yaw = yaw_from_quat(base_quat)             # (N, T)
    dyaw = np.abs(_wrap_angle(np.diff(yaw, axis=-1)))  # (N, T-1)

    yaw_jump_max = np.zeros(N)
    yaw_jump_mean = np.zeros(N)
    yaw_jump_series = []
    for i in range(N):
        rl = max(real_lengths[i], 2)
        s = dyaw[i, :rl - 1]
        yaw_jump_max[i] = np.max(s) if len(s) > 0 else 0.0
        yaw_jump_mean[i] = np.mean(s) if len(s) > 0 else 0.0
        yaw_jump_series.append(dyaw[i])

    # ── 2. Hand–object distance (requires MuJoCo FK) ─────────────────────
    obj_rest_z = float(full_traj[0, 0, 38])
    _LIFTED_THRESH = 0.02  # metres above rest z to count as "lifted"

    hand_obj_dist_max = np.full(N, np.nan)
    hand_obj_dist_mean = np.full(N, np.nan)
    hand_obj_series = [np.full(T, np.nan) for _ in range(N)]

    if _HAS_MUJOCO and mj_xml_path and os.path.exists(mj_xml_path):
        try:
            mj_model = mujoco.MjModel.from_xml_path(mj_xml_path)
            mj_data = mujoco.MjData(mj_model)

            # Body IDs
            pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            lw_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                       "left_wrist_roll_link")
            rw_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                       "right_wrist_roll_link")

            if pelvis_id >= 0 and lw_id >= 0 and rw_id >= 0:
                for i in range(N):
                    rl = real_lengths[i]
                    lifted_dists = []
                    for t in range(rl):
                        # Set qpos: freejoint (7) + 29 joints
                        mj_data.qpos[:7] = full_traj[i, t, :7]
                        mj_data.qpos[7:36] = full_traj[i, t, 7:36]
                        mujoco.mj_kinematics(mj_model, mj_data)

                        lw_pos = mj_data.xpos[lw_id].copy()
                        rw_pos = mj_data.xpos[rw_id].copy()
                        obj_pos = full_traj[i, t, 36:39]

                        d_left = np.linalg.norm(lw_pos - obj_pos)
                        d_right = np.linalg.norm(rw_pos - obj_pos)
                        d_mean = 0.5 * (d_left + d_right)
                        hand_obj_series[i][t] = d_mean

                        if obj_pos[2] > obj_rest_z + _LIFTED_THRESH:
                            lifted_dists.append(d_mean)

                    if lifted_dists:
                        hand_obj_dist_max[i] = np.max(lifted_dists)
                        hand_obj_dist_mean[i] = np.mean(lifted_dists)
        except Exception as e:
            print(f"  ⚠  FK failed: {e}")

    # ── Store in result ───────────────────────────────────────────────────
    result["yaw_jump_max"] = yaw_jump_max
    result["yaw_jump_mean"] = yaw_jump_mean
    result["yaw_jump_series"] = yaw_jump_series
    result["hand_obj_dist_max"] = hand_obj_dist_max
    result["hand_obj_dist_mean"] = hand_obj_dist_mean
    result["hand_obj_series"] = hand_obj_series
    result["obj_rest_z"] = obj_rest_z
    result["yaw_series"] = yaw  # (N, T) — absolute yaw for trajectory plots
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Physics-metric plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_physics_bar(all_results: dict, out_path_prefix: str):
    """
    Bar chart comparing yaw-jump and hand-obj-dist metrics across ablations.
    Produces two PNGs: <prefix>_yaw_jump.png, <prefix>_hand_obj.png
    """
    names = list(all_results.keys())
    n = len(names)

    # -- Yaw jump --
    means = [np.mean(all_results[k]["yaw_jump_mean"]) for k in names]
    maxes = [np.mean(all_results[k]["yaw_jump_max"]) for k in names]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w / 2, np.degrees(means), w, label="mean |Δyaw| (°)", color="teal")
    ax.bar(x + w / 2, np.degrees(maxes), w, label="max |Δyaw| (°)", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Yaw jump (degrees)")
    ax.set_title("Robot Yaw Jump per Ablation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = f"{out_path_prefix}_yaw_jump.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved yaw-jump bar chart → {p}")

    # -- Hand-object distance --
    has_fk = any(not np.all(np.isnan(all_results[k]["hand_obj_dist_mean"])) for k in names)
    if has_fk:
        means_ho = [np.nanmean(all_results[k]["hand_obj_dist_mean"]) for k in names]
        maxes_ho = [np.nanmean(all_results[k]["hand_obj_dist_max"]) for k in names]

        fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
        ax.bar(x - w / 2, means_ho, w, label="mean dist (m)", color="teal")
        ax.bar(x + w / 2, maxes_ho, w, label="max dist (m)", color="salmon")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Hand–Object Distance (m)")
        ax.set_title("Hand–Object Distance While Lifted")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = f"{out_path_prefix}_hand_obj.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved hand-obj bar chart → {p}")


def plot_per_ablation_physics(result: dict, name: str, abl_dir: str):
    """Per-goal yaw-jump time series and hand-obj distance time series."""
    N = len(result["real_lengths"])

    # -- Yaw jump time series --
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(N):
        rl = result["real_lengths"][i]
        s = result["yaw_jump_series"][i][:max(rl - 1, 1)]
        ax.plot(np.degrees(s), alpha=0.5, lw=0.8, label=f"g{i}" if N <= 15 else None)
    ax.set_xlabel("Frame")
    ax.set_ylabel("|Δyaw| (°)")
    ax.set_title(f"Per-Frame Yaw Jump — {name}")
    if N <= 15:
        ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(abl_dir, "yaw_jump_series.png"), dpi=150)
    plt.close(fig)

    # -- Hand-obj distance time series --
    if not np.all(np.isnan(result["hand_obj_series"][0])):
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(N):
            rl = result["real_lengths"][i]
            s = result["hand_obj_series"][i][:rl]
            ax.plot(s, alpha=0.5, lw=0.8)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean hand–obj dist (m)")
        ax.set_title(f"Hand–Object Distance — {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(abl_dir, "hand_obj_series.png"), dpi=150)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory state plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_ablation_trajectories(result: dict, name: str, abl_dir: str):
    """
    Plot object (x, y, z) and robot (x, y, yaw) over time for every goal
    in a single ablation.  6-subplot figure.
    """
    full_traj = result["full_traj"]  # (N, T, 43)
    real_lengths = result["real_lengths"]
    yaw_all = result.get("yaw_series")  # (N, T) if available
    N = full_traj.shape[0]

    labels = ["obj x", "obj y", "obj z", "robot x", "robot y", "robot yaw (°)"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.ravel()

    for i in range(N):
        rl = result["real_lengths"][i]
        t = np.arange(rl)
        alpha = 0.5 if N > 6 else 0.8
        lw = 0.7 if N > 12 else 1.0

        axes[0].plot(t, full_traj[i, :rl, 36], alpha=alpha, lw=lw)
        axes[1].plot(t, full_traj[i, :rl, 37], alpha=alpha, lw=lw)
        axes[2].plot(t, full_traj[i, :rl, 38], alpha=alpha, lw=lw)
        axes[3].plot(t, full_traj[i, :rl, 0],  alpha=alpha, lw=lw)
        axes[4].plot(t, full_traj[i, :rl, 1],  alpha=alpha, lw=lw)
        if yaw_all is not None:
            axes[5].plot(t, np.degrees(yaw_all[i, :rl]), alpha=alpha, lw=lw)
        else:
            yaw_i = np.degrees(yaw_from_quat(full_traj[i, :rl, 3:7]))
            axes[5].plot(t, yaw_i, alpha=alpha, lw=lw)

    for ax, lbl in zip(axes, labels):
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    for ax in axes[3:]:
        ax.set_xlabel("Frame")

    fig.suptitle(f"State Trajectories — {name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(abl_dir, "state_trajectories.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved state trajectories → {abl_dir}/state_trajectories.png")


def plot_cross_ablation_trajectories(all_results: dict, out_dir: str):
    """
    Overlay mean ± std of object (x,y,z) and robot (x,y,yaw) across goals,
    one line per ablation.  Useful for comparing ablation-level behaviour.
    """
    names = list(all_results.keys())
    if not names:
        return

    cmap = plt.cm.get_cmap("tab10", max(len(names), 1))
    labels = ["obj x", "obj y", "obj z", "robot x", "robot y", "robot yaw (°)"]
    indices = [36, 37, 38, 0, 1, None]  # None = yaw computed separately

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.ravel()

    for ai, name in enumerate(names):
        r = all_results[name]
        ft = r["full_traj"]          # (N, T, 43)
        rl = r["real_lengths"]       # (N,)
        max_rl = int(np.max(rl))
        N = ft.shape[0]

        for si, idx in enumerate(indices):
            if idx is not None:
                vals = ft[:, :max_rl, idx]  # (N, max_rl)
            else:
                yaw_arr = r.get("yaw_series")
                if yaw_arr is not None:
                    vals = np.degrees(yaw_arr[:, :max_rl])
                else:
                    vals = np.degrees(yaw_from_quat(ft[:, :max_rl, 3:7]))

            # Mask frames beyond real_length
            masked = np.ma.array(vals)
            for i in range(N):
                masked[i, rl[i]:] = np.ma.masked
            mean = masked.mean(axis=0)
            std = masked.std(axis=0)
            t = np.arange(max_rl)

            c = cmap(ai)
            axes[si].plot(t, mean, color=c, lw=1.2, label=name if si == 0 else None)
            axes[si].fill_between(t, mean - std, mean + std, color=c, alpha=0.15)

    for ax, lbl in zip(axes, labels):
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    for ax in axes[3:]:
        ax.set_xlabel("Frame")

    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Cross-Ablation State Trajectories (mean ± std)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, "cross_ablation_trajectories.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved cross-ablation trajectories → {p}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser("Evaluate all ablation checkpoints")
    p.add_argument("--inference_config", type=str, default="config/inference.yaml",
                   help="Path to inference defaults YAML")

    # Ablation discovery
    p.add_argument("--ablations_folder", type=str, default=None,
                   help="Root folder containing ablation config YAMLs")
    p.add_argument("--epoch", type=str, default="latest",
                   help="Checkpoint epoch (int) or 'latest' to auto-detect")
    p.add_argument("--ema", action="store_true")

    # IC selection
    p.add_argument("--traj_idx",   type=int, default=0)
    p.add_argument("--batch_idx",  type=int, default=0)
    p.add_argument("--start_time", type=int, default=0)

    # Goal sweep params
    p.add_argument("--num_goals",        type=int,   default=12)
    p.add_argument("--goal_spread",      type=float, default=0.5)
    p.add_argument("--goal_multiplier",  type=float, default=1.0)
    p.add_argument("--include_original", action="store_true")
    p.add_argument("--seed",             type=int,   default=42)

    # Inference
    p.add_argument("--stitch_steps", type=int, default=None)
    p.add_argument("--batch_size",   type=int, default=None,
                   help="Goals per inference batch (default: all at once)")
    p.add_argument("--cfg_w",        type=float, default=1.0)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--enable_goal_stop",   action="store_true")
    p.add_argument("--goal_stop_threshold", type=float, default=0.1)
    p.add_argument("--enable_physics_stop", action="store_true")
    p.add_argument("--enable_phys_clamp",   action="store_true")

    # Waypoint / z-profile
    p.add_argument("--last_frame_waypoint", action="store_true")
    p.add_argument("--arrival_ratio", type=float, default=0.85)
    p.add_argument("--lift_height",   type=float, default=DEFAULT_LIFT_HEIGHT)
    p.add_argument("--lift_start",    type=float, default=0.0)
    p.add_argument("--lift_end",      type=float, default=0.20)
    p.add_argument("--walk_start_z",  type=float, default=0.80)
    p.add_argument("--no_lower_dist", type=float, default=0.4)

    # Output
    p.add_argument("--output_root", type=str, default="results/ablation_eval")
    p.add_argument("--visualize", action="store_true",
                   help="Open MuJoCo viewer for each ablation (blocks)")

    cfg_defaults, _ = load_inference_defaults(
        argv=argv,
        section="evaluate_ablations",
        default_path="config/inference.yaml",
    )
    if cfg_defaults:
        p.set_defaults(**cfg_defaults)

    args = p.parse_args(argv)
    if args.ablations_folder is None:
        p.error("Missing `--ablations_folder`. Set it via CLI or config/inference.yaml::evaluate_ablations.ablations_folder")
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    ablations_folder = args.ablations_folder
    if not os.path.isdir(ablations_folder):
        print(f"ERROR: '{ablations_folder}' does not exist.")
        sys.exit(1)

    # ── Discover ablation configs ────────────────────────────────────────────
    pattern = os.path.join(ablations_folder, "**/*.yaml")
    cfg_files = sorted(glob.glob(pattern, recursive=True))
    cfg_files = [f for f in cfg_files
                 if not f.endswith(".patched.yaml")
                 and not f.endswith(".eval_patched.yaml")]
    if not cfg_files:
        print(f"No YAML files found in {ablations_folder}")
        sys.exit(0)

    print(f"Found {len(cfg_files)} ablation configs.\n")

    # ── Backup main config ───────────────────────────────────────────────────
    backup = MAIN_CONFIG_FILE + ".eval_bak"
    if os.path.exists(MAIN_CONFIG_FILE):
        shutil.copy(MAIN_CONFIG_FILE, backup)

    os.makedirs(args.output_root, exist_ok=True)

    all_results = {}  # name → result dict
    total_t0 = time.time()

    try:
        for ci, cfg_path in enumerate(cfg_files):
            abl_name = safe_ablation_name(cfg_path, ablations_folder)
            print(f"\n{'#' * 72}")
            print(f"  [{ci + 1}/{len(cfg_files)}]  {abl_name}")
            print(f"  Config: {cfg_path}")

            # Derive suffix & patch
            suffix = derive_suffix_from_filename(cfg_path)
            patched_path = None
            if suffix is not None:
                patched_path = patch_suffix_in_config(cfg_path, suffix)
                active_cfg = patched_path
            else:
                active_cfg = cfg_path

            try:
                # ── Install config as the main config ────────────────────────
                shutil.copy(active_cfg, MAIN_CONFIG_FILE)

                # ── Load the ablation config to find checkpoint ──────────────
                model_cfg, data_cfg, training_cfg, noise_cfg = load_config(MAIN_CONFIG_FILE)

                save_dir = get_save_path(model_cfg, data_cfg, training_cfg)
                print(f"  Save dir: {save_dir}")

                # Determine checkpoint path
                if args.epoch == "latest":
                    ckpt_path, epoch_num = find_latest_checkpoint(save_dir, ema=args.ema)
                    if ckpt_path is None:
                        print(f"  ⚠  No checkpoint found in {save_dir} — skipping.\n")
                        continue
                    print(f"  Using latest checkpoint: epoch {epoch_num}")
                else:
                    epoch_num = int(args.epoch)
                    prefix = "ema_model_" if args.ema else "model_"
                    ckpt_path = os.path.join(save_dir, f"{prefix}{epoch_num}.pth")
                    if not os.path.exists(ckpt_path):
                        print(f"  ⚠  Checkpoint not found: {ckpt_path} — skipping.\n")
                        continue

                # ── Build MotionGenerator ────────────────────────────────────
                mg = MotionGenerator(config_path=MAIN_CONFIG_FILE, device=device)
                mg.diffuser.load_weights_from_file(ckpt_path)

                # ── Build full dataset (for initial conditions) ──────────────
                data_path = get_data_path(data_cfg)
                norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
                data_buffer = preload_dataset(data_cfg, data_path)
                dataset = FlexibleWindowDataset(
                    data_buffer=data_buffer,
                    config=data_cfg,
                    norm_path=norm_path,
                    calculate_stats=False,
                    training_cfg={},
                )
                mg.dataset = dataset

                # ── Run sweep ────────────────────────────────────────────────
                result = run_goal_sweep(mg, dataset, args)
                all_results[abl_name] = result

                # ── Save per-ablation arrays ─────────────────────────────────
                abl_dir = os.path.join(args.output_root, abl_name)
                os.makedirs(abl_dir, exist_ok=True)
                for key in ["full_traj", "goals_world", "goals_local", "errors",
                            "real_lengths", "desired_displacements",
                            "achieved_displacements"]:
                    np.save(os.path.join(abl_dir, f"{key}.npy"), result[key])
                np.save(os.path.join(abl_dir, "ref_obj_pos.npy"), result["ref_obj_pos"])
                np.savez_compressed(
                    os.path.join(abl_dir, "goal_sweep_results.npz"),
                    trajectories=result["full_traj"],
                    goals_world=result["goals_world"],
                    goals_local=result["goals_local"],
                    errors=result["errors"],
                    real_lengths=result["real_lengths"],
                    desired_displacements=result["desired_displacements"],
                    achieved_displacements=result["achieved_displacements"],
                    ref_obj_pos=result["ref_obj_pos"],
                    per_goal_time_s=result.get("per_goal_time_s", np.zeros_like(result["errors"])),
                )

                # ── Per-ablation plots ───────────────────────────────────────
                plot_polar_error(result, abl_name,
                                 os.path.join(abl_dir, "polar_error.png"))
                plot_per_ablation_displacement(result, abl_name,
                                               os.path.join(abl_dir, "displacement_vectors.png"))

                # ── Physics consistency metrics ──────────────────────────────
                xml_path, _ = get_mj_xml_paths()
                compute_physics_metrics(result, mj_xml_path=xml_path)
                plot_per_ablation_physics(result, abl_name, abl_dir)
                plot_per_ablation_trajectories(result, abl_name, abl_dir)

                # Save physics metric arrays
                for pk in ["yaw_jump_max", "yaw_jump_mean",
                           "hand_obj_dist_max", "hand_obj_dist_mean"]:
                    if pk in result:
                        np.save(os.path.join(abl_dir, f"{pk}.npy"), result[pk])

                # ── Print summary ────────────────────────────────────────────
                errs = result["errors"]
                print(f"  Mean err: {np.mean(errs):.4f}m  "
                      f"Median: {np.median(errs):.4f}m  "
                      f"Max: {np.max(errs):.4f}m")
                print(f"  Yaw jump — mean: {np.degrees(np.mean(result['yaw_jump_mean'])):.2f}°  "
                      f"max: {np.degrees(np.mean(result['yaw_jump_max'])):.2f}°")
                if not np.all(np.isnan(result["hand_obj_dist_mean"])):
                    print(f"  Hand-obj — mean: {np.nanmean(result['hand_obj_dist_mean']):.4f}m  "
                          f"max: {np.nanmean(result['hand_obj_dist_max']):.4f}m")

                # ── Optional visualisation (blocks) ──────────────────────────
                if args.visualize:
                    from batch_goal_sweep import visualize_results
                    visualize_results(result["full_traj"], result["obj_traj_w"],
                                      result["goals_world"], args)

                # ── Cleanup GPU memory before next ablation ──────────────────
                del mg, dataset, data_buffer
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()

            finally:
                if patched_path and os.path.exists(patched_path):
                    os.remove(patched_path)

    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted by user.")

    finally:
        # Restore original config
        if os.path.exists(backup):
            shutil.copy(backup, MAIN_CONFIG_FILE)
            os.remove(backup)

    # ── Global comparison plots ──────────────────────────────────────────────
    if all_results:
        print(f"\n{'=' * 72}")
        print("Generating comparison plots …")
        plot_error_bar(all_results,
                       os.path.join(args.output_root, "error_comparison.png"))
        plot_displacement_scatter(all_results,
                                  os.path.join(args.output_root, "displacement_scatter.png"))
        plot_physics_bar(all_results,
                         os.path.join(args.output_root, "physics"))
        plot_cross_ablation_trajectories(all_results, args.output_root)

        # Summary CSV
        csv_path = os.path.join(args.output_root, "summary.csv")
        with open(csv_path, "w") as f:
            header = ("ablation,mean_error,median_error,max_error,std_error,"
                      "mean_yaw_jump_deg,max_yaw_jump_deg,"
                      "mean_hand_obj_dist,max_hand_obj_dist,n_goals\n")
            f.write(header)
            for name, r in all_results.items():
                errs = r["errors"]
                yj_m = np.degrees(np.mean(r.get("yaw_jump_mean", [0])))
                yj_x = np.degrees(np.mean(r.get("yaw_jump_max", [0])))
                ho_m = np.nanmean(r.get("hand_obj_dist_mean", [np.nan]))
                ho_x = np.nanmean(r.get("hand_obj_dist_max", [np.nan]))
                f.write(f"{name},{np.mean(errs):.5f},{np.median(errs):.5f},"
                        f"{np.max(errs):.5f},{np.std(errs):.5f},"
                        f"{yj_m:.3f},{yj_x:.3f},"
                        f"{ho_m:.5f},{ho_x:.5f},{len(errs)}\n")
        print(f"  Saved summary CSV → {csv_path}")

    total_time = time.time() - total_t0
    print(f"\nDone. Total time: {total_time / 60:.1f} min.")


if __name__ == "__main__":
    main()

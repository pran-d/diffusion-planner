#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix


def _to_ego_frame_obj_xy(traj: np.ndarray) -> np.ndarray:
    """Convert object XY trajectory to ego frame at t=0 robot yaw."""
    robot_quat0 = traj[0, 3:7]
    obj_xy = traj[:, 36:38]
    obj_z = traj[:, 38]
    obj_xyz = np.concatenate([obj_xy, obj_z[:, None]], axis=-1)

    yaw = yaw_from_quat(robot_quat0)
    R_w2l = yaw_to_rot_matrix(-yaw)

    p0 = obj_xyz[0]
    rel = obj_xyz - p0
    rel_local = (R_w2l @ rel.T).T
    return rel_local[:, :2]


def _load_ablation_dirs(output_root: str) -> List[Path]:
    p = Path(output_root)
    out = []
    for d in p.iterdir():
        if not d.is_dir():
            continue
        if (d / "metrics.json").exists() and (d / "trajectories.npy").exists():
            out.append(d)
    return sorted(out)


def save_ablation_plots(ablation_dir: Path):
    name = ablation_dir.name
    traj = np.load(ablation_dir / "trajectories.npy")
    goals = np.load(ablation_dir / "goals.npy")
    real_lengths = np.load(ablation_dir / "real_lengths.npy")

    # 1) XY ego overlay
    fig_xy, ax_xy = plt.subplots(figsize=(5, 5))
    for i in range(traj.shape[0]):
        rl = int(np.clip(real_lengths[i], 1, traj.shape[1]))
        ego_xy = _to_ego_frame_obj_xy(traj[i, :rl])
        ax_xy.plot(ego_xy[:, 0], ego_xy[:, 1], alpha=0.6, lw=1.0)
        g_rel = goals[i, :2] - traj[i, 0, 36:38]
        ax_xy.scatter(g_rel[0], g_rel[1], marker="x", s=20)
    ax_xy.set_title(f"{name}: object XY (ego)")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_aspect("equal")
    fig_xy.tight_layout()
    xy_path = ablation_dir / "xy_ego_overlay.png"
    fig_xy.savefig(xy_path, dpi=150)
    plt.close(fig_xy)

    # 2) Object Z over time
    fig_z, ax_z = plt.subplots(figsize=(6, 3.5))
    for i in range(traj.shape[0]):
        rl = int(np.clip(real_lengths[i], 1, traj.shape[1]))
        t = np.arange(rl) * 0.01
        ax_z.plot(t, traj[i, :rl, 38], alpha=0.6)
    ax_z.set_title(f"{name}: object z")
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z")
    fig_z.tight_layout()
    z_path = ablation_dir / "obj_z_profiles.png"
    fig_z.savefig(z_path, dpi=150)
    plt.close(fig_z)

    # 3) Representative feature time series
    ridx = 0
    rl = int(np.clip(real_lengths[ridx], 1, traj.shape[1]))
    rep = traj[ridx, :rl]
    t = np.arange(rl) * 0.01
    fig_f, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(t, rep[:, 0], label="base_x")
    axes[0].plot(t, rep[:, 1], label="base_y")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, rep[:, 36], label="obj_x")
    axes[1].plot(t, rep[:, 37], label="obj_y")
    axes[1].legend(fontsize=8)
    axes[2].plot(t, rep[:, 38], label="obj_z")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("time [s]")
    fig_f.tight_layout()
    feat_path = ablation_dir / "feature_timeseries.png"
    fig_f.savefig(feat_path, dpi=150)
    plt.close(fig_f)

    # 4) Summary grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # recreate lightweight versions in grid
    for i in range(traj.shape[0]):
        rli = int(np.clip(real_lengths[i], 1, traj.shape[1]))
        ego_xy = _to_ego_frame_obj_xy(traj[i, :rli])
        axes[0, 0].plot(ego_xy[:, 0], ego_xy[:, 1], alpha=0.5, lw=1.0)
    axes[0, 0].set_title("XY ego")
    axes[0, 0].set_aspect("equal")
    for i in range(traj.shape[0]):
        rli = int(np.clip(real_lengths[i], 1, traj.shape[1]))
        ti = np.arange(rli) * 0.01
        axes[0, 1].plot(ti, traj[i, :rli, 38], alpha=0.5, lw=1.0)
    axes[0, 1].set_title("obj z")
    axes[1, 0].plot(t, rep[:, 36], label="obj_x")
    axes[1, 0].plot(t, rep[:, 37], label="obj_y")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("rep obj xy")
    axes[1, 1].plot(t, rep[:, 0], label="base_x")
    axes[1, 1].plot(t, rep[:, 1], label="base_y")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("rep base xy")
    fig.suptitle(name)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ablation_dir / "summary.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser("Compare ablation outputs")
    p.add_argument("--output_root", type=str, default="results/ablations")
    args = p.parse_args()

    ablation_dirs = _load_ablation_dirs(args.output_root)
    if not ablation_dirs:
        raise RuntimeError(f"No ablation directories with metrics found in {args.output_root}")

    comp_dir = Path(args.output_root) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for d in ablation_dirs:
        with open(d / "metrics.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "ablation": d.name,
            "l2_error_mean": m.get("l2_goal_error_mean", np.nan),
            "l2_error_std": m.get("l2_goal_error_std", np.nan),
            "success_rate_0.10": m.get("success_rate_0.10", np.nan),
            "time_per_traj_s": m.get("wall_clock_time_per_traj_s", np.nan),
            "gpu_peak_mb": m.get("gpu_peak_mb", np.nan),
        })
        save_ablation_plots(d)

    csv_path = comp_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ablation", "l2_error_mean", "l2_error_std", "success_rate_0.10", "time_per_traj_s", "gpu_peak_mb"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nAblation comparison:")
    for r in rows:
        print(
            f"{r['ablation']:20s}  err={r['l2_error_mean']:.4f}±{r['l2_error_std']:.4f}  "
            f"succ@0.10={100.0*r['success_rate_0.10']:.1f}%  "
            f"time={r['time_per_traj_s']:.4f}s  mem={r['gpu_peak_mb']:.1f}MB"
        )

    # Bar chart: mean L2 error
    names = [r["ablation"] for r in rows]
    vals = [r["l2_error_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4.5))
    x = np.arange(len(names))
    ax.bar(x, vals, color="steelblue", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Mean final L2 error [m]")
    ax.set_title("Ablation: final goal error")
    fig.tight_layout()
    fig.savefig(comp_dir / "mean_l2_error_bar.png", dpi=150)
    plt.close(fig)

    # Multi-panel global XY overlay
    fig, axes = plt.subplots(1, len(ablation_dirs), figsize=(5 * len(ablation_dirs), 4), squeeze=False)
    for i, d in enumerate(ablation_dirs):
        traj = np.load(d / "trajectories.npy")
        rl = np.load(d / "real_lengths.npy")
        ax = axes[0, i]
        for j in range(traj.shape[0]):
            rlj = int(np.clip(rl[j], 1, traj.shape[1]))
            ego_xy = _to_ego_frame_obj_xy(traj[j, :rlj])
            ax.plot(ego_xy[:, 0], ego_xy[:, 1], alpha=0.4, lw=1.0)
        ax.set_title(d.name)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(comp_dir / "xy_overlay_all_ablations.png", dpi=150)
    plt.close(fig)

    print(f"Saved comparison outputs to {comp_dir}")


if __name__ == "__main__":
    main()

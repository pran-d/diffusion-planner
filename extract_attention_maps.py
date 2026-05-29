"""Extract and plot attention heatmaps from a trained dit1d_dual_grouped ablation.

Hooks the backbone's forward so that on every call during inference we collect:
  - temporal self-attention   per (branch, block)   shape (B*G, h, T, T)
  - group self-attention      per (branch, block)   shape (B*T, h, G, G)
  - group cross-attention     branch2 blocks only    shape (B*T, h, Gb, Gw)

Maps are accumulated (running mean) over all backbone forward calls across all
trajectories in the evaluation batch, then saved as a .npz plus per-layer PNG
grids.

Usage:
    python extract_attention_maps.py \\
        --ablation_config config/ablations/config_baseline_grouped.yaml \\
        --epoch latest --ema \\
        --num_trajectories 8 \\
        --output_dir results/attention/baseline_grouped/
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports
from config.configure import (
    load_config, get_save_path, get_data_path, get_norm_path,
    load_yaml_with_includes,
)
from motion_generator import MotionGenerator
from utils.data.load_dataset import preload_dataset
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.attention_capture import enable_capture, collect
from evaluate_ablations import build_merged_temp_config, find_latest_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX feature-group symbols (match the trajectory representation in
# docs/example.tex, Eq. (vector_representation) and Table tab:feat_repr). Used to
# label heatmap axes with paper notation instead of internal config group names.
# ──────────────────────────────────────────────────────────────────────────────

GROUP_LATEX: dict[str, str] = {
    "delta_xy":      r"$\Delta p_{rob,xy}$",
    "delta_yaw":     r"$\Delta \psi_{rob}$",
    "body_z":        r"$p_{rob,z}$",
    "joints":        r"$j_{pos}$",
    "joints_lower":  r"$j_{pos}^{\,low}$",
    "joints_upper":  r"$j_{pos}^{\,up}$",
    "body_rot6d":    r"$r_{rob,xy}$",
    "obj_rel_pos":   r"${}^{rob}p_{obj}$",
    "obj_rel_rot6d": r"${}^{rob}r_{obj}$",
    "obj_delta_xy":  r"$\Delta p_{obj,xy}$",
    "obj_z":         r"$p_{obj,z}$",
}


def _latex_labels(names: list[str]) -> list[str]:
    """Map internal config group names to LaTeX symbols for axis labels."""
    return [GROUP_LATEX.get(n, n) for n in names]


# ──────────────────────────────────────────────────────────────────────────────
# Accumulator
# ──────────────────────────────────────────────────────────────────────────────

class AttnAccumulator:
    """Running mean of attention maps, reducing over leading (batch-like) dims.

    For each captured map of shape (N, num_heads, dim1, dim2), we mean over N to
    get (num_heads, dim1, dim2). Then we running-mean over forward calls.
    """

    def __init__(self):
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def update(self, maps: dict[str, torch.Tensor]) -> None:
        for name, t in maps.items():
            reduced = t.float().mean(dim=0).cpu()         # (h, d1, d2)
            if name not in self.sums:
                self.sums[name] = reduced.clone()
                self.counts[name] = 1
            else:
                self.sums[name] += reduced
                self.counts[name] += 1

    def mean(self) -> dict[str, np.ndarray]:
        return {k: (self.sums[k] / self.counts[k]).numpy() for k in self.sums}


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _heatmap_grid(
    maps: np.ndarray,      # (num_heads, d1, d2)
    title: str,
    out_path: str,
    xticklabels: list[str] | None = None,
    yticklabels: list[str] | None = None,
):
    h = maps.shape[0]
    cols = min(h, 4)
    rows = (h + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows), squeeze=False)
    for i in range(h):
        ax = axes[i // cols][i % cols]
        im = ax.imshow(maps[i], aspect="auto", cmap="viridis")
        ax.set_title(f"head {i}", fontsize=9)
        if xticklabels is not None:
            ax.set_xticks(range(len(xticklabels)))
            ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)
        if yticklabels is not None:
            ax.set_yticks(range(len(yticklabels)))
            ax.set_yticklabels(yticklabels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(h, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _single_heatmap(
    m: np.ndarray,           # (d1, d2)
    title: str,
    out_path: str,
    xticklabels: list[str] | None = None,
    yticklabels: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, aspect="auto", cmap="viridis")
    ax.set_title(title, fontsize=12)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=11)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=11)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_maps(
    maps: dict[str, np.ndarray],
    output_dir: str,
    b1_group_names: list[str],
    b2_group_names: list[str],
    per_layer: bool = False,
):
    """Aggregate layers + heads into one temporal and one group-cross heatmap.

    If per_layer=True, also writes one heatmap per (branch, layer) into a
    `per_layer/` subdirectory so you can see which layer develops the structure.
    """
    os.makedirs(output_dir, exist_ok=True)

    temporal_stack, cross_stack = [], []
    temporal_per_layer, cross_per_layer = {}, {}
    for name, m in maps.items():            # m shape: (num_heads, d1, d2)
        if name.endswith("temporal_attn"):
            head_mean = m.mean(axis=0)              # (T, T)
            temporal_stack.append(head_mean)
            temporal_per_layer[name] = head_mean
        elif name.endswith("group_cross_attn"):
            head_mean = m.mean(axis=0)              # (G_b, G_w)
            cross_stack.append(head_mean)
            cross_per_layer[name] = head_mean

    if temporal_stack:
        time_map = np.stack(temporal_stack, axis=0).mean(axis=0)
        out = os.path.join(output_dir, "temporal_attn.png")
        _single_heatmap(
            time_map,
            "Temporal self-attention",
            out,
            xlabel=r"Key time $t'$",
            ylabel=r"Query time $t$",
        )
        print(f"  wrote {out}  shape={time_map.shape}")

    if cross_stack:
        cross_map = np.stack(cross_stack, axis=0).mean(axis=0)
        out = os.path.join(output_dir, "group_cross_attn.png")
        _single_heatmap(
            cross_map,
            "Local-to-global feature-group cross-attention",
            out,
            xticklabels=_latex_labels(b1_group_names),
            yticklabels=_latex_labels(b2_group_names),
            xlabel="Global-stream feature group",
            ylabel="Local-stream feature group",
        )
        print(f"  wrote {out}  shape={cross_map.shape}")

    if per_layer:
        sub = os.path.join(output_dir, "per_layer")
        os.makedirs(sub, exist_ok=True)
        for name, m in temporal_per_layer.items():
            slug = name.replace(".", "_")
            out = os.path.join(sub, f"{slug}.png")
            _single_heatmap(
                m, f"{name}\ntemporal self-attention (head-mean)", out,
                xlabel=r"Key time $t'$", ylabel=r"Query time $t$",
            )
        for name, m in cross_per_layer.items():
            slug = name.replace(".", "_")
            out = os.path.join(sub, f"{slug}.png")
            _single_heatmap(
                m, f"{name}\nlocal-to-global cross-attention (head-mean)", out,
                xticklabels=_latex_labels(b1_group_names),
                yticklabels=_latex_labels(b2_group_names),
                xlabel="Global-stream feature group",
                ylabel="Local-stream feature group",
            )
        print(f"  wrote {len(temporal_per_layer) + len(cross_per_layer)} per-layer plots to {sub}")


# ──────────────────────────────────────────────────────────────────────────────
# Backbone forward hook (accumulate per call)
# ──────────────────────────────────────────────────────────────────────────────

def patched_forward(backbone, captured, accumulator):
    orig = backbone.forward

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        maps = collect(captured)
        if maps:
            accumulator.update(maps)
        return out

    backbone.forward = wrapped
    return orig


# ──────────────────────────────────────────────────────────────────────────────
# Eval-batch construction (mirrors evaluate_ablations: pull a few random
# trajectories from the training dataset and use them as initial conditions).
# ──────────────────────────────────────────────────────────────────────────────

def build_initial_conditions(dataset, num_traj: int, seed: int = 0):
    """Pick `num_traj` windows; return (robot_hist, obj_hist, goal_local) in world frame.

    Mirrors `inference_mg.extract_initial_condition` (pickplace path only).
    """
    from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix

    rng = np.random.default_rng(seed)
    n = len(dataset)
    idxs = rng.choice(n, size=num_traj, replace=False)
    H = dataset.history_size

    robot_hist, obj_hist, goals = [], [], []
    for i in idxs:
        file_idx, batch_idx, start_time, *_ = dataset.indices[int(i)]
        raw_traj = dataset._get_single_traj(file_idx, batch_idx)
        h_start, h_end = start_time, start_time + H

        r = np.concatenate([
            raw_traj['base'][h_start:h_end],     # (H, 7)
            raw_traj['joints'][h_start:h_end],   # (H, 29)
        ], axis=-1)
        o = raw_traj['obj'][h_start:h_end]       # (H, 7)
        robot_hist.append(r)
        obj_hist.append(o)

        # Pickplace goal in robot yaw-aligned local frame, like inference_mg.py.
        init_base_quat = raw_traj['base'][h_end - 1, 3:7]
        init_obj_pos   = raw_traj['obj'][h_end - 1, :3]
        anchor = dataset[int(i)][-1]
        final_obj_pos  = anchor['final_obj_pos']
        yaw = yaw_from_quat(init_base_quat)
        R = yaw_to_rot_matrix(-yaw)
        delta_world = final_obj_pos[:3] - init_obj_pos[:3]
        delta_local = (R @ delta_world[:, None])[:, 0]
        goals.append(delta_local[:3])

    return (
        np.stack(robot_hist, axis=0),
        np.stack(obj_hist, axis=0),
        np.stack(goals, axis=0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_config", required=True,
                    help="Path to ablation YAML, e.g. config/ablations/config_baseline_grouped.yaml")
    ap.add_argument("--base_config", default="config/config.yaml.bak")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--num_trajectories", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    # Waypoint / lift conditioning args (mirror inference_mg.py defaults).
    ap.add_argument("--last_frame_waypoint", action="store_true", default=True)
    ap.add_argument("--no_last_frame_waypoint", dest="last_frame_waypoint",
                    action="store_false")
    ap.add_argument("--arrival_ratio", type=float, default=0.85)
    ap.add_argument("--lift_height", type=float, default=0.5)
    ap.add_argument("--lift_start", type=float, default=0.0)
    ap.add_argument("--lift_end", type=float, default=0.20)
    ap.add_argument("--walk_start_z", type=float, default=0.80)
    ap.add_argument("--no_lower_dist", type=float, default=0.4)
    ap.add_argument("--cfg_w", type=float, default=1.0)
    ap.add_argument("--per_layer", action="store_true",
                    help="Also save per-(branch, layer) temporal + group-cross heatmaps.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build merged config (base + ablation override)
    base_cfg = load_yaml_with_includes(args.base_config) or {}
    tmp_dir = os.path.join(os.path.dirname(args.output_dir) or ".", "_tmp_cfg")
    os.makedirs(tmp_dir, exist_ok=True)
    merged_cfg_path = build_merged_temp_config(base_cfg, args.ablation_config, suffix=None, tmp_dir=tmp_dir)

    # Resolve checkpoint
    model_cfg, data_cfg, training_cfg, _ = load_config(merged_cfg_path)
    save_dir = get_save_path(model_cfg, data_cfg, training_cfg)
    if args.epoch == "latest":
        ckpt_path, epoch_num = find_latest_checkpoint(save_dir, ema=args.ema)
        if ckpt_path is None:
            sys.exit(f"No checkpoint found in {save_dir}")
        print(f"Using latest checkpoint: epoch {epoch_num}  ({ckpt_path})")
    else:
        prefix = "ema_model_" if args.ema else "model_"
        ckpt_path = os.path.join(save_dir, f"{prefix}{args.epoch}.pth")
        if not os.path.exists(ckpt_path):
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        print(f"Using checkpoint: {ckpt_path}")

    # Build MotionGenerator + dataset
    mg = MotionGenerator(config_path=merged_cfg_path, device=device)
    mg.diffuser.load_weights_from_file(ckpt_path)
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    data_buffer = preload_dataset(data_cfg, data_path)
    dataset = FlexibleWindowDataset(
        data_buffer=data_buffer, config=data_cfg, norm_path=norm_path,
        calculate_stats=False, training_cfg={},
    )
    mg.dataset = dataset
    print(f"Dataset has {len(dataset)} windows.")

    backbone = mg.diffuser.model.diffusion_model.model
    bb_name = type(backbone).__name__
    print(f"Backbone class: {bb_name}")
    if bb_name != "DiT1DDualGrouped":
        print(f"⚠  Expected DiT1DDualGrouped, got {bb_name}. Attention shapes "
              f"will fall back to per-layer time x time only.")

    # Group names for axis labels (if available)
    b1_group_names = list(getattr(backbone, "branch1_group_names", []))
    b2_group_names = list(getattr(backbone, "branch2_group_names", []))

    # Enable capture + monkey-patch backbone forward for running-mean collection
    captured = enable_capture(backbone)
    print(f"Captured {len(captured)} attention modules.")
    accumulator = AttnAccumulator()
    patched_forward(backbone, captured, accumulator)

    # Build a small eval batch and run generate_trajectory
    robot_hist, obj_hist, goals = build_initial_conditions(
        dataset, args.num_trajectories, seed=args.seed,
    )
    print(f"Eval batch: {robot_hist.shape=}, {obj_hist.shape=}, {goals.shape=}")

    with torch.no_grad():
        mg.generate_trajectory(
            initial_condition={"robot": robot_hist, "obj": obj_hist},
            goal_condition=goals,
            stitch_steps=1,
            num_samples=1,
            cfg_w=args.cfg_w,
            use_last_frame_wp=args.last_frame_waypoint,
            arrival_ratio=args.arrival_ratio,
            lift_height=args.lift_height,
            lift_start=args.lift_start,
            lift_end=args.lift_end,
            walk_start_z=args.walk_start_z,
            no_lower_dist=args.no_lower_dist,
        )

    maps = accumulator.mean()
    print(f"Collected {len(maps)} layers; counts per layer: "
          f"{set(accumulator.counts.values())}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "attention_maps.npz"),
        **{k.replace(".", "_"): v for k, v in maps.items()},
        b1_group_names=np.array(b1_group_names),
        b2_group_names=np.array(b2_group_names),
    )
    plot_maps(maps, args.output_dir, b1_group_names, b2_group_names,
              per_layer=args.per_layer)
    print(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()

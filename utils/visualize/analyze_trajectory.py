#!/usr/bin/env python3
"""
Trajectory Analysis Tool
========================

Plots feature-level time series for generated (and optionally original) trajectories,
with waypoint markers.  Supports both normalised (model feature-space) and
denormalised (SBTO feature-space) views, as well as reconstructed world-frame plots.

Usage examples
--------------
# Basic: plot normalised windows from an inference run
python analyze_trajectory.py results/inference_analysis.npz

# Compare with a ground-truth dataset sample
python analyze_trajectory.py results/inference_analysis.npz --gt_idx 0

# Plot denormalized feature space
python analyze_trajectory.py results/inference_analysis.npz --mode denormalized

# Plot reconstructed world-frame trajectory
python analyze_trajectory.py results/inference_analysis.npz --mode world

# Select specific sample from batch
python analyze_trajectory.py results/inference_analysis.npz --sample 0

# Only plot specific feature groups
python analyze_trajectory.py results/inference_analysis.npz --groups locomotion object_pos
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature layout (must match training config)
# ---------------------------------------------------------------------------
FEATURE_LAYOUT = {
    "delta_xy": 2,
    "delta_yaw": 1,
    "obj_delta_xy": 2,
    "obj_z": 1,
    "joints": 29,
    "body_z": 1,
    "body_rot6d": 6,
    "obj_rel_pos": 3,
    "obj_rel_rot6d": 6,
}

FEATURE_GROUPS = {
    "locomotion": ["obj_delta_xy"],
    "pick_place": ["obj_z"],
    "robot_pos": ["delta_xy", "delta_yaw", "body_z"],
    "object_pos": ["obj_rel_rot6d", "obj_rel_pos"],
    "pose_summary": ["joints", "body_rot6d"],
}

WORLD_FRAME_LAYOUT = {
    "robot_xyz": (0, 3),
    "robot_quat": (3, 7),
    "robot_joints": (7, 36),
    "obj_xyz": (36, 39),
    "obj_quat": (39, 43),
}


def build_feature_slices(feature_order, feature_dims):
    """Build {name: slice} from ordered feature list and dim array."""
    slices = {}
    idx = 0
    for name, dim in zip(feature_order, feature_dims):
        slices[name] = slice(idx, idx + int(dim))
        idx += int(dim)
    return slices


def collect_group_slices(group_keys, feat_slices):
    """Collect slice objects for all features in a group."""
    parts = []
    for key in group_keys:
        if key in feat_slices:
            parts.append((key, feat_slices[key]))
    return parts


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Distinct color palettes for generated vs ground-truth channels
_CHANNEL_COLORS_GEN = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_CHANNEL_COLORS_GT  = ["#17becf", "#bcbd22", "#7f7f7f", "#e377c2", "#aec7e8", "#ffbb78"]


def _plot_feature_timeseries(
    ax, times, data, label, color, alpha=1.0, linestyle="-",
    palette=None,
):
    """Plot one or more channels vs time on ax. data shape: (T,) or (T, C).

    If *palette* is provided (list of colors), each channel gets its own color
    from the palette.  Otherwise falls back to the single *color* argument.
    """
    if data.ndim == 1:
        ax.plot(times, data, color=color, alpha=alpha, label=label, linestyle=linestyle)
    else:
        for c in range(data.shape[1]):
            lbl = f"{label}[{c}]" if data.shape[1] > 1 else label
            cc = palette[c % len(palette)] if palette else color
            ax.plot(times, data[:, c], color=cc, alpha=alpha, label=lbl, linestyle=linestyle)


def _plot_waypoint_markers(
    ax, times, wp_vals, wp_mask, feat_slice,
    achieved_data=None,
):
    """Overlay desired and achieved markers where the waypoint mask is True.

    Fully-constrained frames (where *all* D dims are True) are treated as
    initial-condition keyframes, not user waypoints, and are **skipped** so
    they don't clutter every subplot with markers.

    Parameters
    ----------
    ax : matplotlib Axes
    times : (total_T,) frame indices
    wp_vals : (total_T, D) desired waypoint values
    wp_mask : (total_T, D) bool mask
    feat_slice : slice into D for this feature
    achieved_data : (total_T, D) actual generated values (optional)
    """
    D = wp_mask.shape[1]

    # Identify frames that are fully constrained (initial condition keyframes)
    fully_constrained = wp_mask.all(axis=1)  # (total_T,) bool

    vals = wp_vals[:, feat_slice]   # (T, feat_dim)
    mask = wp_mask[:, feat_slice].copy()   # (T, feat_dim) bool

    # Zero out fully-constrained frames so they are not plotted as waypoints
    mask[fully_constrained] = False

    if not mask.any():
        return

    num_ch = vals.shape[1]
    added_desired_label = False
    added_achieved_label = False

    for c in range(num_ch):
        t_marked = np.where(mask[:, c])[0]
        if len(t_marked) == 0:
            continue

        ch_color = _CHANNEL_COLORS_GEN[c % len(_CHANNEL_COLORS_GEN)]

        # Desired waypoint value — star marker
        lbl_d = "desired" if not added_desired_label else None
        ax.scatter(
            times[t_marked], vals[t_marked, c],
            marker="*", s=140, c=ch_color, zorder=6,
            edgecolors="black", linewidths=0.6,
            label=lbl_d,
        )
        added_desired_label = True

        # Achieved value at the same positions — diamond marker
        if achieved_data is not None:
            ach = achieved_data[:, feat_slice]
            lbl_a = "achieved" if not added_achieved_label else None
            ax.scatter(
                times[t_marked], ach[t_marked, c],
                marker="D", s=60, c=ch_color, zorder=5,
                edgecolors="white", linewidths=0.8,
                label=lbl_a,
            )
            added_achieved_label = True


# ---------------------------------------------------------------------------
# Main plotting routines
# ---------------------------------------------------------------------------

def plot_feature_space(
    data,            # dict from npz
    mode="normalized",
    sample_idx=0,
    groups=None,
    gt_windows=None, # optional (num_windows, T, D) ground truth in same space
    save_path=None,
):
    """
    Plot per-feature-group time series across all stitched windows.

    mode: 'normalized' or 'denormalized'
    """
    key = "normalized_windows" if mode == "normalized" else "denormalized_windows"
    windows = data[key]            # (num_windows, B, T, D)
    wp_vals = data["waypoint_values"]  # (num_windows, B, T, D)
    wp_masks = data["waypoint_masks"]  # (num_windows, B, T, D)
    feature_order = list(data["feature_order"])
    feature_dims = data["feature_dims"]

    feat_slices = build_feature_slices(feature_order, feature_dims)

    # Select sample from batch
    windows = windows[:, sample_idx]       # (num_windows, T, D)
    wp_vals = wp_vals[:, sample_idx]
    wp_masks = wp_masks[:, sample_idx]

    num_windows, T, D = windows.shape

    # Concatenate windows into a single time axis
    all_data = windows.reshape(-1, D)           # (total_T, D)
    all_wp_vals = wp_vals.reshape(-1, D)
    all_wp_masks = wp_masks.reshape(-1, D)
    total_T = all_data.shape[0]
    times = np.arange(total_T)

    # Select groups to plot
    if groups is None:
        groups = list(FEATURE_GROUPS.keys())
    else:
        groups = [g for g in groups if g in FEATURE_GROUPS]

    # Count total subplots (one per feature in each group, but combine small dims)
    plot_specs = []  # (group_name, feat_name, slice)
    for gname in groups:
        parts = collect_group_slices(FEATURE_GROUPS[gname], feat_slices)
        for fname, fslice in parts:
            dim = fslice.stop - fslice.start
            if dim <= 6:
                plot_specs.append((gname, fname, fslice))
            else:
                # For high-dim features (joints), plot summary statistics
                plot_specs.append((gname, f"{fname} (mean±std)", fslice))

    n_plots = len(plot_specs)
    if n_plots == 0:
        print("No feature groups to plot.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (gname, fname, fslice) in zip(axes, plot_specs):
        dim = fslice.stop - fslice.start

        if dim <= 6:
            feat_data = all_data[:, fslice]
            _plot_feature_timeseries(
                ax, times, feat_data, "generated", "tab:blue",
                palette=_CHANNEL_COLORS_GEN,
            )

            if gt_windows is not None:
                gt_concat = gt_windows[:, sample_idx].reshape(-1, D) if gt_windows.ndim == 4 else gt_windows.reshape(-1, D)
                if gt_concat.shape[0] >= total_T:
                    gt_feat = gt_concat[:total_T, fslice]
                    _plot_feature_timeseries(
                        ax, times, gt_feat, "ground truth", "tab:green",
                        alpha=0.7, linestyle="--",
                        palette=_CHANNEL_COLORS_GT,
                    )
            # Waypoint markers: desired (star) vs achieved (diamond)
            _plot_waypoint_markers(
                ax, times, all_wp_vals, all_wp_masks, fslice,
                achieved_data=all_data,
            )
        else:
            # Summary plot for high-dimensional features
            feat_data = all_data[:, fslice]  # (total_T, dim)
            mean_val = feat_data.mean(axis=1)
            std_val = feat_data.std(axis=1)
            ax.plot(times, mean_val, color="tab:blue", label="generated (mean)")
            ax.fill_between(times, mean_val - std_val, mean_val + std_val,
                            color="tab:blue", alpha=0.2)

            if gt_windows is not None:
                gt_concat = gt_windows[:, sample_idx].reshape(-1, D) if gt_windows.ndim == 4 else gt_windows.reshape(-1, D)
                if gt_concat.shape[0] >= total_T:
                    gt_feat = gt_concat[:total_T, fslice]
                    gt_mean = gt_feat.mean(axis=1)
                    gt_std = gt_feat.std(axis=1)
                    ax.plot(times, gt_mean, color="tab:green", alpha=0.7,
                            linestyle="--", label="ground truth (mean)")
                    ax.fill_between(times, gt_mean - gt_std, gt_mean + gt_std,
                                    color="tab:green", alpha=0.1)

            # Waypoint markers for high-dim: mark any waypointed time step
            for t in range(total_T):
                if all_wp_masks[t, fslice].any():
                    ax.axvline(t, color="red", alpha=0.3, linewidth=0.5)

        ax.set_ylabel(fname, fontsize=9)
        ax.set_title(f"[{gname}] {fname}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Window boundaries
        for w in range(1, num_windows):
            ax.axvline(w * T, color="gray", linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"Feature-Space Trajectory ({mode})", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {mode} plot to {save_path}")
    else:
        plt.show()

    return fig


def plot_world_frame(
    data,
    sample_idx=0,
    save_path=None,
):
    """
    Plot reconstructed world-frame trajectory fields.
    """
    traj = data["trajectory"]  # (B, total_T, 43)
    if traj.ndim == 3:
        traj = traj[sample_idx]  # (total_T, 43)

    total_T = traj.shape[0]
    times = np.arange(total_T)

    plot_specs = [
        ("Robot XY", slice(0, 2)),
        ("Robot Z", slice(2, 3)),
        ("Object XY", slice(36, 38)),
        ("Object Z", slice(38, 39)),
    ]

    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(14, 3 * len(plot_specs)), sharex=True)
    for ax, (name, sl) in zip(axes, plot_specs):
        feat = traj[:, sl]
        _plot_feature_timeseries(ax, times, feat, name, "tab:blue")
        ax.set_ylabel(name)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Also add an XY top-down plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    ax2.plot(traj[:, 0], traj[:, 1], "b-", alpha=0.7, label="Robot path")
    ax2.plot(traj[:, 36], traj[:, 37], "r-", alpha=0.7, label="Object path")
    ax2.plot(traj[0, 0], traj[0, 1], "bo", markersize=10, label="Robot start")
    ax2.plot(traj[-1, 0], traj[-1, 1], "b^", markersize=10, label="Robot end")
    ax2.plot(traj[0, 36], traj[0, 37], "ro", markersize=10, label="Object start")
    ax2.plot(traj[-1, 36], traj[-1, 37], "r^", markersize=10, label="Object end")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top-Down XY Trajectory", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("World-Frame Trajectory", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        xy_path = save_path.replace(".png", "_xy.png")
        fig2.savefig(xy_path, dpi=150, bbox_inches="tight")
        print(f"Saved world-frame plots to {save_path} and {xy_path}")
    else:
        plt.show()

    return fig, fig2


def plot_waypoint_summary(data, sample_idx=0, save_path=None):
    """
    Dedicated waypoint visualization: shows which features are constrained
    at each time step across all windows.
    """
    wp_masks = data["waypoint_masks"]  # (num_windows, B, T, D)
    feature_order = list(data["feature_order"])
    feature_dims = data["feature_dims"]
    feat_slices = build_feature_slices(feature_order, feature_dims)

    masks = wp_masks[:, sample_idx]  # (num_windows, T, D)
    num_windows, T, D = masks.shape
    all_mask = masks.reshape(-1, D)  # (total_T, D)

    # Build per-feature summary: fraction of dims constrained
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # Create a heatmap-like view per feature group
    feature_names = []
    feature_fracs = []
    for fname in feature_order:
        sl = feat_slices[fname]
        dim = sl.stop - sl.start
        frac = all_mask[:, sl].sum(axis=1) / max(dim, 1)  # (total_T,)
        feature_names.append(fname)
        feature_fracs.append(frac)

    heatmap = np.stack(feature_fracs, axis=0)  # (num_features, total_T)
    im = ax.imshow(
        heatmap, aspect="auto", cmap="RdYlGn",
        interpolation="nearest", vmin=0, vmax=1,
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel("Frame")
    ax.set_title("Waypoint Mask Coverage (green = constrained)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Fraction of dims constrained")

    # Window boundaries
    for w in range(1, num_windows):
        ax.axvline(w * T - 0.5, color="white", linewidth=1.5)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved waypoint summary to {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze generated trajectories with feature-level plots and waypoint markers."
    )
    parser.add_argument(
        "analysis_path", type=str,
        help="Path to *_analysis.npz saved by inference_mg.py",
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["normalized", "denormalized", "world", "waypoints", "all"],
        help="Which view(s) to plot",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Batch sample index to plot",
    )
    parser.add_argument(
        "--groups", nargs="+", type=str, default=None,
        help="Feature groups to plot (default: all). Options: "
             + ", ".join(FEATURE_GROUPS.keys()),
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save plots (default: show interactively)",
    )
    parser.add_argument(
        "--gt_path", type=str, default=None,
        help="Path to ground-truth analysis NPZ for comparison",
    )
    args = parser.parse_args()

    print(f"Loading analysis data from {args.analysis_path}...")
    data = dict(np.load(args.analysis_path, allow_pickle=True))

    # Print summary
    if "normalized_windows" in data:
        nw = data["normalized_windows"]
        print(f"  Windows: {nw.shape[0]}, Batch: {nw.shape[1]}, "
              f"T: {nw.shape[2]}, D: {nw.shape[3]}")
    if "trajectory" in data:
        print(f"  World trajectory shape: {data['trajectory'].shape}")
    print(f"  Feature order: {list(data['feature_order'])}")
    print(f"  Feature dims:  {list(data['feature_dims'])}")

    # Load optional GT
    gt_windows = None
    if args.gt_path:
        gt_data = dict(np.load(args.gt_path, allow_pickle=True))
        key = "normalized_windows" if "normalized_windows" in gt_data else "denormalized_windows"
        gt_windows = gt_data.get(key)

    # Determine save paths
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    modes = (
        ["normalized", "denormalized", "world", "waypoints"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in modes:
        sp = str(save_dir / f"{mode}.png") if save_dir else None

        if mode == "normalized":
            plot_feature_space(
                data, mode="normalized", sample_idx=args.sample,
                groups=args.groups, gt_windows=gt_windows, save_path=sp,
            )
        elif mode == "denormalized":
            plot_feature_space(
                data, mode="denormalized", sample_idx=args.sample,
                groups=args.groups, gt_windows=gt_windows, save_path=sp,
            )
        elif mode == "world":
            plot_world_frame(data, sample_idx=args.sample, save_path=sp)
        elif mode == "waypoints":
            plot_waypoint_summary(data, sample_idx=args.sample, save_path=sp)

    if not save_dir:
        plt.show()


if __name__ == "__main__":
    main()

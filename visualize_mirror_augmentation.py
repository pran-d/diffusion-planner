"""
visualize_mirror_augmentation.py
=================================
Visualise a raw data window alongside its mirror-augmented counterpart.

Usage:
    uv run python visualize_mirror_augmentation.py \
        --config config/config.yaml \
        [--window_idx 0] \
        [--out_dir mirror_viz]

Output (saved to --out_dir):
    mirror_joints.png        — joint angles: original vs mirror (left/right swapped)
    mirror_delta_xy_yaw.png  — robot dx/dy/dyaw: original vs mirror (y & yaw negated)
    mirror_body_rot6d.png    — body rotation 6D: original vs mirror
    mirror_obj_rel.png       — object relative pos/rot6d
    mirror_task_params.png   — task params vector
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config.configure import load_config, get_data_path, get_norm_path
from utils.data.load_dataset import preload_dataset
from datasets.flexible_dataset import FlexibleWindowDataset
from verify_mirror import mirror_qpos


# ─── helpers ────────────────────────────────────────────────────────────────

def _get_raw(ds: FlexibleWindowDataset, idx: int):
    """Return the *un-augmented* tensors for a given dataset index."""
    file_idx, batch_idx, t_start = ds.indices[idx]
    raw = ds._get_single_traj(file_idx, batch_idx)
    features, anchor = ds._compute_transform(raw, t_start)

    parts = []
    for key in ds.feature_order:
        part = torch.from_numpy(features[key]).float()
        part = ds._normalize(key, part)
        parts.append(part)
    window_tensor = torch.cat(parts, dim=-1)

    current_state = window_tensor[:ds.history_size, (ds.num_features - ds.num_observations):].clone()
    future_states = window_tensor.clone()

    task_params = torch.from_numpy(features["task_params"]).float()
    task_params = ds._normalize("task_params", task_params)

    return future_states, current_state, task_params, features


def _get_raw_qpos(ds: FlexibleWindowDataset, idx: int) -> np.ndarray:
    """Return the underlying world-frame qpos trajectory for a dataset window.

    Shape: (T, 43) = base(7) + joints(29) + obj(7)
    """
    file_idx, batch_idx, _ = ds.indices[idx]
    raw = ds._get_single_traj(file_idx, batch_idx)
    return np.concatenate([raw['base'], raw['joints'], raw['obj']], axis=-1)


def _denorm_feature(ds: FlexibleWindowDataset, key: str, val: torch.Tensor) -> np.ndarray:
    return ds._denormalize(key, val).numpy()


def _slice_feature(ds: FlexibleWindowDataset, tensor: torch.Tensor, key: str) -> torch.Tensor:
    """Extract a named feature slice from the concatenated window tensor."""
    from utils.math.sbto_utils import build_feature_layout
    layout = build_feature_layout()
    fidx = 0
    for k in ds.feature_order:
        dim = layout.get(k, 0)
        if k == key:
            return tensor[:, fidx: fidx + dim]
        fidx += dim
    raise KeyError(f"Feature '{key}' not in feature_order.")


# ─── plotting helpers ────────────────────────────────────────────────────────

def _plot_comparison(ax, orig: np.ndarray, mirr: np.ndarray,
                     title: str, labels=None, history_size: int = 2):
    T, D = orig.shape
    t = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, D))
    for i in range(D):
        lbl = labels[i] if labels is not None else f"[{i}]"
        ax.plot(t, orig[:, i], color=colors[i], lw=1.5, label=f"orig {lbl}")
        ax.plot(t, mirr[:, i], color=colors[i], lw=1.0, ls="--", alpha=0.7,
                label=f"mirr {lbl}")
    ax.axvline(history_size - 0.5, color="k", lw=0.8, ls=":", alpha=0.5,
               label="history|future")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.set_xlabel("timestep")
    ax.grid(True, alpha=0.3)


def _save_fig(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved → {path}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--window_idx", type=int, default=0,
                    help="Dataset index of the window to visualise.")
    ap.add_argument("--out_dir", default="mirror_viz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── load config & dataset ────────────────────────────────────────────────
    model_cfg, data_cfg, training_cfg, noise_sched_cfg = load_config(args.config)

    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    data_buffer = preload_dataset(data_cfg, data_path)

    # Temporarily force mirror augmentation on so tables are built
    data_cfg_aug = dict(data_cfg)
    data_cfg_aug["augmentation"] = {
        "mirror_symmetry": {"enabled": True, "prob": 0.0}  # prob=0 → never auto-applied
    }

    ds = FlexibleWindowDataset(
        data_buffer=data_buffer,
        config=data_cfg_aug,
        norm_path=norm_path,
        calculate_stats=False,
        training_cfg=training_cfg,
    )

    if args.window_idx >= len(ds):
        print(f"window_idx={args.window_idx} out of range (dataset has {len(ds)} windows). "
              f"Using index 0.")
        args.window_idx = 0

    # ── get original & mirrored tensors ─────────────────────────────────────
    future_orig, state_orig, tp_orig, raw_features = _get_raw(ds, args.window_idx)
    future_mirr, state_mirr, tp_mirr = ds._apply_mirror_symmetry(
        future_orig.clone(), state_orig.clone(), tp_orig.clone()
    )

    # Reference world-space mirror using the repo's qpos implementation.
    qpos_orig = _get_raw_qpos(ds, args.window_idx)
    qpos_mirr = mirror_qpos(qpos_orig)

    # ── denormalize for interpretable units ─────────────────────────────────
    def _dn(key, tensor):
        return ds._denormalize(key, tensor).numpy()

    future_orig_np = ds.denormalize_global(future_orig).numpy()   # (T, D)
    future_mirr_np = ds.denormalize_global(future_mirr).numpy()

    state_orig_np  = state_orig.numpy()  # keep normalised — easier to compare symmetry
    state_mirr_np  = state_mirr.numpy()

    H = ds.history_size

    # ─── Plot 1: joints ──────────────────────────────────────────────────────
    from utils.math.sbto_utils import build_feature_layout
    layout = build_feature_layout()

    def _feat_slice(key):
        fidx = 0
        for k in ds.feature_order:
            d = layout.get(k, 0)
            if k == key:
                return slice(fidx, fidx + d)
            fidx += d
        return None

    js = _feat_slice("joints")
    if js is not None:
        orig_j = future_orig_np[:, js]  # (T, 29)
        mirr_j = future_mirr_np[:, js]

        fig, axes = plt.subplots(5, 1, figsize=(14, 18))
        groups = [
            ("Left Leg  (0-5)",  slice(0, 6),
             ["L_hip_p","L_hip_r","L_hip_y","L_knee","L_ank_p","L_ank_r"]),
            ("Right Leg (6-11)", slice(6, 12),
             ["R_hip_p","R_hip_r","R_hip_y","R_knee","R_ank_p","R_ank_r"]),
            ("Torso (12-14)",    slice(12, 15),
             ["W_yaw","W_roll","W_pitch"]),
            ("Left Arm  (15-21)",slice(15, 22),
             ["Ls_p","Ls_r","Ls_y","Le","Lwr","Lwp","Lwy"]),
            ("Right Arm (22-28)",slice(22, 29),
             ["Rs_p","Rs_r","Rs_y","Re","Rwr","Rwp","Rwy"]),
        ]
        for ax, (title, sl, lbls) in zip(axes, groups):
            _plot_comparison(ax, orig_j[:, sl], mirr_j[:, sl], title, lbls, H)
        fig.suptitle("Joint Angles: Original vs Mirror-Augmented\n"
                     "(dashed = mirrored; left/right blocks should swap)", fontsize=11)
        _save_fig(fig, os.path.join(args.out_dir, "mirror_joints.png"))

    # ─── Plot 2: delta_xy, delta_yaw ─────────────────────────────────────────
    xy_sl  = _feat_slice("delta_xy")
    yaw_sl = _feat_slice("delta_yaw")
    oy_sl  = _feat_slice("obj_delta_xy")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    if xy_sl:
        _plot_comparison(axes[0],
                         future_orig_np[:, xy_sl],
                         future_mirr_np[:, xy_sl],
                         "delta_xy  (mirror should negate y)",
                         ["Δx", "Δy"], H)
    if yaw_sl:
        _plot_comparison(axes[1],
                         future_orig_np[:, yaw_sl],
                         future_mirr_np[:, yaw_sl],
                         "delta_yaw  (mirror should negate)",
                         ["Δyaw"], H)
    if oy_sl:
        _plot_comparison(axes[2],
                         future_orig_np[:, oy_sl],
                         future_mirr_np[:, oy_sl],
                         "obj_delta_xy  (mirror should negate y)",
                         ["obj_Δx", "obj_Δy"], H)
    fig.suptitle("Position / Yaw Features: Original vs Mirror-Augmented", fontsize=11)
    _save_fig(fig, os.path.join(args.out_dir, "mirror_delta_xy_yaw.png"))

    # ─── Plot 3: body_rot6d & obj_rel_pos ───────────────────────────────────
    br_sl  = _feat_slice("body_rot6d")
    op_sl  = _feat_slice("obj_rel_pos")
    or_sl  = _feat_slice("obj_rel_rot6d")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    if br_sl:
        _plot_comparison(axes[0],
                         future_orig_np[:, br_sl],
                         future_mirr_np[:, br_sl],
                         "body_rot6d  [r1x,r1y,r1z,r2x,r2y,r2z] "
                         "(mirror: negate r1y, r2x, r2z)",
                         ["r1x","r1y","r1z","r2x","r2y","r2z"], H)
    if op_sl:
        _plot_comparison(axes[1],
                         future_orig_np[:, op_sl],
                         future_mirr_np[:, op_sl],
                         "obj_rel_pos  (mirror should negate y)",
                         ["x","y","z"], H)
    if or_sl:
        _plot_comparison(axes[2],
                         future_orig_np[:, or_sl],
                         future_mirr_np[:, or_sl],
                         "obj_rel_rot6d  (mirror: negate r1y, r2x, r2z)",
                         ["r1x","r1y","r1z","r2x","r2y","r2z"], H)
    fig.suptitle("Rotation / Object Features: Original vs Mirror-Augmented", fontsize=11)
    _save_fig(fig, os.path.join(args.out_dir, "mirror_body_rot6d.png"))

    # ─── Plot 4: task params ─────────────────────────────────────────────────
    tp_orig_np = ds._denormalize("task_params", tp_orig).numpy()  # (3,) or (T, 3)
    tp_mirr_np = ds._denormalize("task_params", tp_mirr).numpy()
    if tp_orig_np.ndim == 1:
        tp_orig_np = tp_orig_np[None, :]
        tp_mirr_np = tp_mirr_np[None, :]
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(tp_orig_np.shape[-1])
    bar_w = 0.35
    ax.bar(x - bar_w/2, tp_orig_np[0], bar_w, label="original", color="steelblue")
    ax.bar(x + bar_w/2, tp_mirr_np[0], bar_w, label="mirrored", color="tomato", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["goal_x", "goal_y", "dist"][:len(x)])
    ax.set_title("Task Params: Original vs Mirror-Augmented\n"
                 "(goal_y should negate; goal_x & dist should stay the same)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save_fig(fig, os.path.join(args.out_dir, "mirror_task_params.png"))

    # ─── Sanity-check symmetry properties ───────────────────────────────────
    print("\n── Sanity checks ───────────────────────────────────────────────────────")
    print("Feature-space mirror is exact for sign-flip / permutation features, but joint")
    print("values can differ slightly from a world-space mirror because these features are")
    print("relative-frame transformed and normalized. The world-space qpos mirror below is")
    print("the authoritative semantic check.")

    def _check(name, orig, mirr, expect_sign=+1, tol=0.01):
        """Check that mirr ≈ expect_sign * orig."""
        diff = mirr - (expect_sign * orig)
        max_err = np.abs(diff).max()
        status = "✓" if max_err < tol else "✗  <-- MISMATCH"
        print(f"  {name:40s}  max|err|={max_err:.5f}  {status}")

    if js is not None:
        orig_j_np = future_orig_np[:, js]
        mirr_j_np = future_mirr_np[:, js]
     _check("feature-space: left_hip_pitch ≈ mirror right_hip_pitch",
         orig_j_np[:, 0], mirr_j_np[:, 6], expect_sign=+1)
     _check("feature-space: left_hip_roll ≈ -mirror right_hip_roll",
         orig_j_np[:, 1], mirr_j_np[:, 7], expect_sign=-1)
     _check("feature-space: waist_yaw negated",
         orig_j_np[:, 12], mirr_j_np[:, 12], expect_sign=-1)

    if xy_sl:
        _check("delta_x unchanged?",
               future_orig_np[:, xy_sl.start],
               future_mirr_np[:, xy_sl.start], expect_sign=+1)
        _check("delta_y negated?",
               future_orig_np[:, xy_sl.start+1],
               future_mirr_np[:, xy_sl.start+1], expect_sign=-1)
    if yaw_sl:
        _check("delta_yaw negated?",
               future_orig_np[:, yaw_sl.start],
               future_mirr_np[:, yaw_sl.start], expect_sign=-1)

    tp_check_orig = tp_orig_np[0]
    tp_check_mirr = tp_mirr_np[0]
    _check("task goal_x unchanged?", tp_check_orig[0:1], tp_check_mirr[0:1], +1)
    _check("task goal_y negated?",   tp_check_orig[1:2], tp_check_mirr[1:2], -1)
    _check("task dist unchanged?",   tp_check_orig[2:3], tp_check_mirr[2:3], +1)

    print("\n── World-space qpos mirror checks (authoritative) ─────────────────────")
    joints_orig = qpos_orig[:, 7:36]
    joints_mirr = qpos_mirr[:, 7:36]
    _check("qpos: left_hip_pitch == mirrored right_hip_pitch",
        joints_orig[:, 0], joints_mirr[:, 6], +1)
    _check("qpos: left_hip_roll == -mirrored right_hip_roll",
        joints_orig[:, 1], joints_mirr[:, 7], -1)
    _check("qpos: left_hip_yaw == -mirrored right_hip_yaw",
        joints_orig[:, 2], joints_mirr[:, 8], -1)
    _check("qpos: waist_yaw negated",
        joints_orig[:, 12], joints_mirr[:, 12], -1)
    _check("qpos: left_shoulder_pitch == mirrored right_shoulder_pitch",
        joints_orig[:, 15], joints_mirr[:, 22], +1)
    _check("qpos: left_shoulder_roll == -mirrored right_shoulder_roll",
        joints_orig[:, 16], joints_mirr[:, 23], -1)

    print(f"\nAll plots saved to: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()

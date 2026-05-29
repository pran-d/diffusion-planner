"""Compare time-time (within-branch) self-attention between two ablations:

  * single-branch  (dit1d)        — config/ablations/config_singlebranch.yaml
  * dual-branch     (dit1d_dual)  — config/ablations/config_withoutunmasking.yaml

For both models we capture ONLY the temporal self-attention modules
(`Attention`), i.e. the time x time attention *within* a stream. The dual
model's branch2 cross-attention (`CrossAttentionLayer`, branch coupling) is
deliberately excluded because it is not a within-branch self-attention.

We accumulate a running mean of the softmaxed attention over every backbone
forward call (all denoising steps, all trajectories), reduce over heads and
layers, and emit:
  * one (T, T) map for the single-branch model
  * one (T, T) map per stream (branch1 / branch2) for the dual model
  * a side-by-side comparison PNG
  * locality scalars (diagonal mass, mean |t-t'| offset, row entropy)

Usage:
    uv run python compare_branch_self_attn.py \
        --epoch latest --ema --num_trajectories 8 \
        --output_dir results/attention/branch_self_attn_cmp/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from config.configure import (
    load_config, get_save_path, get_data_path, get_norm_path,
    load_yaml_with_includes,
)
from motion_generator import MotionGenerator
from utils.data.load_dataset import preload_dataset
from datasets.flexible_dataset import FlexibleWindowDataset
from diffusion_forcing_transformer.dit_blocks import Attention
from evaluate_ablations import (
    build_merged_temp_config, find_latest_checkpoint, derive_suffix_from_filename,
)
from extract_attention_maps import (
    AttnAccumulator, patched_forward, build_initial_conditions,
)


# ──────────────────────────────────────────────────────────────────────────────
# Capture only temporal self-attention (Attention), not cross-branch coupling.
# ──────────────────────────────────────────────────────────────────────────────

def enable_self_attn_capture(model):
    captured = {}
    for name, mod in model.named_modules():
        if isinstance(mod, Attention):
            mod._capture_attn = True
            mod._last_attn = None
            captured[name] = mod
    return captured


# ──────────────────────────────────────────────────────────────────────────────
# Locality metrics for a row-stochastic (T, T) attention matrix.
# ──────────────────────────────────────────────────────────────────────────────

def locality_metrics(a: np.ndarray) -> dict:
    """a: (T, T), rows = query t, columns = key t'. Assumed row-normalised."""
    T = a.shape[0]
    a = a / np.clip(a.sum(axis=1, keepdims=True), 1e-9, None)
    idx = np.arange(T)
    offset = np.abs(idx[None, :] - idx[:, None])           # |t - t'|
    mean_offset = float((a * offset).sum(axis=1).mean())    # E|t-t'|
    diag_mass = float(np.diag(a).mean())                    # self / near-self
    ent = -(a * np.log(a + 1e-12)).sum(axis=1)              # per-row entropy (nats)
    mean_entropy = float(ent.mean())
    # fraction of mass within a +-2 frame band of the diagonal
    band = (offset <= 2)
    band_mass = float((a * band).sum(axis=1).mean())
    return {
        "mean_offset": mean_offset,
        "diag_mass": diag_mass,
        "band_mass_pm2": band_mass,
        "row_entropy_nats": mean_entropy,
        "T": T,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Run one model: build, load weights, capture self-attn, run generate.
# ──────────────────────────────────────────────────────────────────────────────

def run_model(ablation_config, base_config, epoch, ema, num_traj, seed, device,
              tmp_dir):
    base_cfg = load_yaml_with_includes(base_config) or {}
    # Match evaluate_ablations.py: the checkpoint dir suffix comes from the
    # config *filename* (config_singlebranch -> _singlebranch), not the YAML's
    # training.suffix field.
    suffix = derive_suffix_from_filename(ablation_config)
    merged_cfg_path = build_merged_temp_config(
        base_cfg, ablation_config, suffix=suffix, tmp_dir=tmp_dir)

    model_cfg, data_cfg, training_cfg, _ = load_config(merged_cfg_path)
    save_dir = get_save_path(model_cfg, data_cfg, training_cfg)
    if epoch == "latest":
        ckpt_path, epoch_num = find_latest_checkpoint(save_dir, ema=ema)
        if ckpt_path is None:
            sys.exit(f"No checkpoint found in {save_dir}")
        print(f"  [{ablation_config}] latest checkpoint: epoch {epoch_num}")
    else:
        prefix = "ema_model_" if ema else "model_"
        ckpt_path = os.path.join(save_dir, f"{prefix}{epoch}.pth")
        if not os.path.exists(ckpt_path):
            sys.exit(f"Checkpoint not found: {ckpt_path}")

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

    backbone = mg.diffuser.model.diffusion_model.model
    bb_name = type(backbone).__name__
    print(f"  backbone: {bb_name}")

    captured = enable_self_attn_capture(backbone)
    print(f"  capturing {len(captured)} self-attention modules: "
          f"{sorted(captured.keys())[:4]}{' ...' if len(captured) > 4 else ''}")
    accum = AttnAccumulator()
    patched_forward(backbone, captured, accum)

    robot_hist, obj_hist, goals = build_initial_conditions(dataset, num_traj, seed=seed)
    with torch.no_grad():
        mg.generate_trajectory(
            initial_condition={"robot": robot_hist, "obj": obj_hist},
            goal_condition=goals,
            stitch_steps=1,
            num_samples=1,
            cfg_w=1.0,
            use_last_frame_wp=True,
            arrival_ratio=0.85,
            lift_height=0.5,
            lift_start=0.0,
            lift_end=0.20,
            walk_start_z=0.80,
            no_lower_dist=0.4,
        )

    maps = accum.mean()   # {name: (h, T, T)}
    return bb_name, maps


def _slug(label: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def aggregate(maps: dict, name_filter) -> np.ndarray | None:
    """Mean over heads + matching layers → (T, T)."""
    stack = [m.mean(axis=0) for n, m in maps.items() if name_filter(n)]
    if not stack:
        return None
    return np.stack(stack, axis=0).mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(panels: list[tuple[str, np.ndarray]], out_path: str,
                    ncols: int | None = None):
    n = len(panels)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows),
                             squeeze=False)
    vmax = max(p[1].max() for p in panels)
    for i, (title, m) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        im = ax.imshow(m, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Key time $t'$")
        ax.set_ylabel(r"Query time $t$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Within-branch temporal self-attention (head- & layer-mean)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_config",
                    default="config/ablations/config_singlebranch.yaml")
    ap.add_argument("--dual_nounmask_config",
                    default="config/ablations/config_withoutunmasking.yaml")
    ap.add_argument("--dual_unmask_config",
                    default="config/ablations/config_baseline.yaml")
    ap.add_argument("--base_config", default="config/config.yaml.bak")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--num_trajectories", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default="results/attention/branch_self_attn_cmp")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_tmp_cfg")
    os.makedirs(tmp_dir, exist_ok=True)

    # (label, config, kind) — kind "single" → one panel, "dual" → two streams.
    model_specs = [
        ("Single-branch", args.single_config, "single"),
        ("Dual no-unmask", args.dual_nounmask_config, "dual"),
        ("Dual +unmask (baseline)", args.dual_unmask_config, "dual"),
    ]

    panels: list[tuple[str, np.ndarray]] = []
    saved: dict[str, np.ndarray] = {}
    for label, cfg, kind in model_specs:
        print(f"=== {label} ({cfg}) ===")
        _, maps = run_model(cfg, args.base_config, args.epoch, args.ema,
                            args.num_trajectories, args.seed, device, tmp_dir)
        if kind == "single":
            m = aggregate(maps, lambda n: True)
            panels.append((label, m))
            saved[f"{_slug(label)}"] = m
        else:
            b1 = aggregate(maps, lambda n: "branch1_blocks" in n)
            b2 = aggregate(maps, lambda n: "branch2_blocks" in n)
            if b1 is not None:
                panels.append((f"{label}\nbranch1 (world)", b1))
                saved[f"{_slug(label)}_branch1"] = b1
            if b2 is not None:
                panels.append((f"{label}\nbranch2 (body)", b2))
                saved[f"{_slug(label)}_branch2"] = b2

    plot_comparison(panels, os.path.join(args.output_dir, "self_attn_comparison.png"),
                    ncols=3)
    np.savez(os.path.join(args.output_dir, "self_attn_maps.npz"), **saved)

    # Locality metrics
    rows = []
    for label, m in panels:
        label = label.replace("\n", " — ")
        met = locality_metrics(m)
        met["model"] = label
        rows.append(met)
        print(f"  {label:28s}  E|t-t'|={met['mean_offset']:.2f}  "
              f"diag={met['diag_mass']:.3f}  band±2={met['band_mass_pm2']:.3f}  "
              f"entropy={met['row_entropy_nats']:.2f} nats")

    csv_path = os.path.join(args.output_dir, "self_attn_locality.csv")
    cols = ["model", "mean_offset", "diag_mass", "band_mass_pm2",
            "row_entropy_nats", "T"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {csv_path}")
    print(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()

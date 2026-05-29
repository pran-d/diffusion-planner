"""Plot group cross-attention maps from several ablation runs side by side.

Reads the `attention_maps.npz` produced by `extract_attention_maps.py` for each
run, aggregates the local-to-global group cross-attention (mean over heads and
layers), and renders one panel per run on a shared color scale so the maps are
directly comparable.

Usage:
    # after running extract_attention_maps.py for each ablation:
    python plot_cross_attn_comparison.py \\
        results/attention/baseline_grouped \\
        results/attention/withoutunmasking_grouped \\
        --labels "With Future Unmasking" "Without Future Unmasking" \\
        --output results/attention/cross_attn_comparison.png

Each input may be either an `attention_maps.npz` file or a directory containing
one.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from extract_attention_maps import GROUP_LATEX, _latex_labels


def _resolve_npz(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "attention_maps.npz")
    return path


def load_cross_map(npz_path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (cross_map (Gb, Gw), b1_group_names, b2_group_names).

    Aggregates over all `group_cross_attn` layers: mean over heads, then mean
    over layers — mirroring `extract_attention_maps.plot_maps`.
    """
    data = np.load(npz_path, allow_pickle=True)
    cross_stack = [
        data[k].mean(axis=0)                       # (num_heads, Gb, Gw) -> (Gb, Gw)
        for k in data.files
        if k.endswith("group_cross_attn")
    ]
    if not cross_stack:
        raise ValueError(f"No 'group_cross_attn' arrays found in {npz_path}")
    cross_map = np.stack(cross_stack, axis=0).mean(axis=0)
    b1 = [str(s) for s in data["b1_group_names"]]   # global stream (columns)
    b2 = [str(s) for s in data["b2_group_names"]]   # local stream  (rows)
    return cross_map, b1, b2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="attention_maps.npz files or directories containing one.")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Per-input panel titles (defaults to the input paths).")
    ap.add_argument("--output", default="cross_attn_comparison.png")
    ap.add_argument("--suptitle",
                    default="Local-to-global feature-group cross-attention")
    args = ap.parse_args()

    npz_paths = [_resolve_npz(p) for p in args.inputs]
    labels = args.labels if args.labels else args.inputs
    if len(labels) != len(npz_paths):
        ap.error(f"got {len(npz_paths)} inputs but {len(labels)} labels")

    maps, b1_names, b2_names = [], None, None
    for p in npz_paths:
        m, b1, b2 = load_cross_map(p)
        maps.append(m)
        if b1_names is None:
            b1_names, b2_names = b1, b2
        elif (b1, b2) != (b1_names, b2_names):
            ap.error(
                f"{p} has feature groups {(b1, b2)} which differ from "
                f"{(b1_names, b2_names)}; cannot share axes across panels."
            )

    # Shared color scale across panels for a fair visual comparison.
    vmin = min(m.min() for m in maps)
    vmax = max(m.max() for m in maps)

    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    axes = axes[0]
    xticks = _latex_labels(b1_names)
    yticks = _latex_labels(b2_names)

    im = None
    for ax, m, label in zip(axes, maps, labels):
        im = ax.imshow(m, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=11)
        ax.set_xlabel("Global-stream feature group", fontsize=11)
    axes[0].set_ylabel("Local-stream feature group", fontsize=11)

    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="Attention weight")
    fig.suptitle(args.suptitle, fontsize=13)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}  ({n} panels, shape={maps[0].shape})")


if __name__ == "__main__":
    main()

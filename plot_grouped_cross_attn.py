"""Render the 2x5 feature-group cross-attention heatmap (summed over layers).

Merges joints_lower + joints_upper into a single 'j_pos' row, relabels body_rot6d
as 'r_rob,xy', and uses LaTeX labels for world groups so the baseline and
without-unmasking plots can be compared directly.

Usage:
    python plot_grouped_cross_attn.py \\
        --npz results/attention/baseline_grouped/attention_maps.npz \\
        --out results/attention/baseline_grouped/group_cross_attn_summed.png \\
        --title "Feature-group cross-attention (baseline)"
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


WORLD_LABELS = {
    "delta_xy":     r"$\Delta p_{rob,xy}$",
    "delta_yaw":    r"$\Delta \psi_{rob}$",
    "body_z":       r"$p_{rob,z}$",
    "obj_rel_pos":  r"${}^{rob}p_{obj}$",
    "obj_rel_rot6d": r"${}^{rob}r_{obj}$",
}

BODY_LABEL_JPOS = r"$j_{pos}$"
BODY_LABEL_RROT = r"$r_{rob,xy}$"


def collapse_cross_attn(npz_path: str):
    """Return (matrix 2xG_w, world_labels, body_labels).

    Sums over (layers, heads), then merges joints_lower+joints_upper rows.
    """
    d = np.load(npz_path, allow_pickle=True)
    b1 = [str(x) for x in d["b1_group_names"]]  # world (keys)
    b2 = [str(x) for x in d["b2_group_names"]]  # body  (queries)

    cross_keys = sorted(k for k in d.files if k.endswith("group_cross_attn"))
    if not cross_keys:
        raise RuntimeError(f"No group_cross_attn entries in {npz_path}")

    # Each entry: (heads, G_body, G_world). Per head the softmax sums to 1 along
    # the key (world) axis, so we mean over heads first, then sum over layers.
    stack = np.stack([d[k] for k in cross_keys], axis=0)  # (L, h, G_b, G_w)
    per_layer = stack.mean(axis=1)                        # (L, G_b, G_w)
    summed = per_layer.sum(axis=0)                        # (G_b, G_w), rows sum to L

    # Merge joints_lower + joints_upper → j_pos (mean so the row still sums to L).
    idx_lower = b2.index("joints_lower")
    idx_upper = b2.index("joints_upper")
    idx_rot   = b2.index("body_rot6d")
    j_pos_row = 0.5 * (summed[idx_lower] + summed[idx_upper])
    rot_row   = summed[idx_rot]
    merged = np.stack([j_pos_row, rot_row], axis=0)       # (2, G_w)

    world_labels = [WORLD_LABELS.get(n, n) for n in b1]
    body_labels = [BODY_LABEL_JPOS, BODY_LABEL_RROT]
    return merged, world_labels, body_labels


def plot(matrix, world_labels, body_labels, title, out_path, vmax=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(world_labels)))
    ax.set_xticklabels(world_labels, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(len(body_labels)))
    ax.set_yticklabels(body_labels, fontsize=13)
    ax.set_xlabel("World feature group (key)", fontsize=12)
    ax.set_ylabel("Body feature group (query)", fontsize=12)
    ax.set_title(title, fontsize=13)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight (summed over layers)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}  range=[{matrix.min():.3f}, {matrix.max():.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Feature-group cross-attention")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Color scale max (set equal across plots for comparison).")
    args = ap.parse_args()

    matrix, world_labels, body_labels = collapse_cross_attn(args.npz)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot(matrix, world_labels, body_labels, args.title, args.out, vmax=args.vmax)


if __name__ == "__main__":
    main()

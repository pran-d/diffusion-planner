"""
plot_dataset_stats.py

Loads a top_trajectories.npz file (batched RL rollout format),
converts it to standardized schema, and plots mean ± std of each field
across the batch of trajectories.

Usage:
    python plot_dataset_stats.py --path /path/to/top_trajectories.npz
    python plot_dataset_stats.py --path /path/to/top_trajectories.npz --fields base joints obj
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ---------------------------------------------------------------------------
# Field metadata: human-readable dimension labels for each standardized key
# ---------------------------------------------------------------------------
FIELD_LABELS = {
    "base": (
        ["pelvis_x", "pelvis_y", "pelvis_z",
         "quat_w", "quat_x", "quat_y", "quat_z"]
    ),
    "obj": (
        ["obj_x", "obj_y", "obj_z",
         "obj_qw", "obj_qx", "obj_qy", "obj_qz"]
    ),
    "base_vel": ["lin_vx", "lin_vy", "lin_vz", "ang_vx", "ang_vy", "ang_vz"],
    "obj_vel":  ["obj_lvx", "obj_lvy", "obj_lvz", "obj_avx", "obj_avy", "obj_avz"],
}

def joints_labels(n):
    return [f"j{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Schema conversion (RL rollout → standardized)
# Uses _convert_schema logic from FlexibleWindowDataset.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_field(ax, mean, std, labels, title, fps=None):
    """
    Plot mean ± std over time for all dimensions of a single field.
    mean, std: (T, D)
    """
    T, D = mean.shape
    t = np.arange(T) / fps if fps else np.arange(T)
    xlabel = "Time (s)" if fps else "Frame"
    cmap = plt.get_cmap("tab20")

    for d in range(D):
        color = cmap(d % 20)
        label = labels[d] if d < len(labels) else f"dim_{d}"
        ax.plot(t, mean[:, d], color=color, linewidth=1.2, label=label)
        ax.fill_between(t,
                        mean[:, d] - std[:, d],
                        mean[:, d] + std[:, d],
                        color=color, alpha=0.15)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, ncol=max(1, D // 8), loc="upper right",
              framealpha=0.5, handlelength=1)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_all_fields(data: dict, fields: list, title_prefix: str, fps=None):
    n_fields = len(fields)
    if n_fields == 0:
        print("No fields to plot.")
        return

    fig = plt.figure(figsize=(7 * min(n_fields, 2), 4 * ((n_fields + 1) // 2)))
    fig.suptitle(f"{title_prefix} — Mean ± Std across trajectories",
                 fontsize=13, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec((n_fields + 1) // 2, min(n_fields, 2),
                           figure=fig, hspace=0.55, wspace=0.35)

    for i, field in enumerate(fields):
        arr = data[field]   # (N, T, D)  or  (T, D) if N=1
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # → (1, T, D)

        mean = arr.mean(axis=0)   # (T, D)
        std  = arr.std(axis=0)    # (T, D)

        D = arr.shape[-1]
        labels = (FIELD_LABELS.get(field)
                  or joints_labels(D))

        ax = fig.add_subplot(gs[i // 2, i % 2])
        plot_field(ax, mean, std, labels, title=field, fps=fps)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot mean ± std of trajectory fields.")
    parser.add_argument("--path",   default=None, help="Path to top_trajectories.npz or dataset folder (defaults to config/paths.yaml)")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file for loading dataset")
    parser.add_argument("--fields", nargs="*",     default=None,
                        help="Fields to plot (default: all available). "
                             "E.g. --fields base joints obj")
    parser.add_argument("--save",   default=None,
                        help="Save figure to this path instead of showing it.")
    args = parser.parse_args()

    from utils.data.load_dataset import preload_dataset
    from config.configure import load_config, get_data_path
    from datasets.flexible_dataset import FlexibleWindowDataset
    
    cfg = load_config(args.config)
    data_cfg = cfg[1] if isinstance(cfg, tuple) else cfg.get("data", {})
    
    if args.path:
        target_path = args.path
    else:
        target_path = get_data_path(data_cfg)

    path = Path(target_path)
    if not path.exists():
        print(f"Warning: Target path might not exist or is resolved dynamically: {path}")

    print(f"Loading {target_path} via load_dataset.py (using {args.config} & paths.yaml) ...")
    
    data_buffer = preload_dataset(data_cfg, str(target_path))
    if not data_buffer:
        print("No valid .npz files found or loaded.")
        return

    # Initialize subset of flexible dataset parser
    dataset_helper = FlexibleWindowDataset(data_buffer=[], config=data_cfg, calculate_stats=False)

    # Extract all standardized schemas across files
    all_data_dicts = []
    fps_val = None
    
    for raw in data_buffer:
        converted = dataset_helper._convert_schema(raw)
        if converted:
            all_data_dicts.append(converted)
        if fps_val is None and "fps" in raw:
            fps_val = float(raw["fps"])

    # Aggregate fields
    data = {}
    if all_data_dicts:
        all_keys = set().union(*(d.keys() for d in all_data_dicts))
        for k in all_keys:
            arrays = [d[k] for d in all_data_dicts if k in d]
            if not arrays: continue
            
            # Skip metadata strings/floats
            if k in ["fps", "source_file_path", "source_folder_name"] or not isinstance(arrays[0], np.ndarray):
                continue
                
            expanded = []
            for arr in arrays:
                if arr.ndim == 2:
                    expanded.append(arr[np.newaxis])
                else:
                    expanded.append(arr)
            data[k] = np.concatenate(expanded, axis=0)

    fps = fps_val
    print(f"  Converted and merged keys: {list(data.keys())}")

    # Print shape summary
    print("\nField shapes and Batch Stats (N, T, D):")
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            # Calculate magnitude (absolute values) of the features
            mean_abs = np.mean(np.abs(v))
            max_abs = np.max(np.abs(v))
            mag_str = f" | mag (mean/max): {mean_abs:.4f} / {max_abs:.4f}"
            
            # Calculate variance across batch dimension (axis 0)
            if v.shape[0] > 1:
                batch_var = np.var(v, axis=0) # shape (T, D)
                mean_var = np.mean(batch_var)
                max_var = np.max(batch_var)
                var_str = f" | var (mean/max): {mean_var:.6f} / {max_var:.6f}"
            else:
                var_str = " | var N/A (batch size 1)"
            print(f"  {k:20s}: {str(v.shape):15s}{mag_str}{var_str}")

    # Select fields to plot
    plottable = [k for k, v in data.items()
                 if isinstance(v, np.ndarray) and v.ndim >= 2]

    fields = args.fields if args.fields else plottable
    fields = [f for f in fields if f in data]
    missing = [f for f in (args.fields or []) if f not in data]
    if missing:
        print(f"Warning: requested fields not found and skipped: {missing}")

    fig = plot_all_fields(data, fields,
                          title_prefix=path.stem,
                          fps=fps)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
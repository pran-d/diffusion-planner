import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add current dir to path if needed for module imports
sys.path.append(os.getcwd())

from tqdm import tqdm

from config.configure import load_config, get_data_path, get_norm_path
from datasets import BufferDataset
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
from utils.data.load_dataset import preload_dataset

def load_dataset():
    # Load config file relative to here? Or root. Assuming run from root.
    if not os.path.exists("config/config.yaml"):
        print("Error: config/config.yaml not found. Please run from project root.")
        sys.exit(1)
        
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml")
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    
    print("Loading dataset...")
    # Initialize dataset
    data_buffer = preload_dataset(data_cfg, data_path)
    dataset = BufferDataset(
        data_buffer=data_buffer, config=data_cfg, task_params=None,
        calculate_stats=False, norm_path=norm_path,
        training_cfg=training_cfg,
    )
    return dataset

def get_feature(traj_dict, feature_spec):
    # feature_spec: "key:idx" or "idx" (flattened)
    
    # Flatten strategy matching visualize_dataset.py
    # full_state = np.concatenate([base, joints, obj], axis=-1)
    # base(7), joints(29), obj(7) -> 43 dims (for some datasets).
    # We should handle flexible shapes.
    
    if ":" in feature_spec:
        key, idx_str = feature_spec.split(":")
        idx = int(idx_str)
        if key not in traj_dict:
            raise ValueError(f"Key {key} not found in trajectory. Available: {list(traj_dict.keys())}")
        
        data = traj_dict[key]
        if idx >= data.shape[1]:
             raise ValueError(f"Index {idx} out of bounds for key {key} with dim {data.shape[1]}")
        return data[:, idx]
    else:
        # Flattened index
        try:
            idx = int(feature_spec)
        except:
             # Maybe it's just a key name asking for index 0?
             if feature_spec in traj_dict:
                 return traj_dict[feature_spec][:, 0]
             raise ValueError(f"Invalid feature spec: {feature_spec}")

        base = traj_dict.get('base', np.zeros((1,0)))
        joints = traj_dict.get('joints', np.zeros((1,0)))
        obj = traj_dict.get('obj', np.zeros((1,0)))
        
        # Ensure T
        T = min(len(base), len(joints), len(obj))
        # Handle cases where some keys might be missing or empty?
        # datasets usually guarantee consistency if loaded correctly.
        
        if T == 0:
            # Maybe one key is missing. fallback to concatenation of existing
            parts = []
            if 'base' in traj_dict: parts.append(traj_dict['base'])
            if 'joints' in traj_dict: parts.append(traj_dict['joints'])
            if 'obj' in traj_dict: parts.append(traj_dict['obj'])
            if not parts: raise ValueError("Empty trajectory")
            full_state = np.concatenate(parts, axis=-1)
        else:
            base = base[:T]; joints = joints[:T]; obj = obj[:T]
            full_state = np.concatenate([base, joints, obj], axis=-1)
            
        if idx >= full_state.shape[1]:
             raise ValueError(f"Index {idx} out of bounds for flattened state dim {full_state.shape[1]}")
        return full_state[:, idx]

def main():
    parser = argparse.ArgumentParser(description="Plot trajectory features including --xy mode.")
    parser.add_argument("--indices", nargs="+", type=int, help="List of trajectory indices to plot")
    parser.add_argument("--all", action="store_true", help="Plot all trajectories")
    parser.add_argument("--max_trajs", type=int, default=100, help="Max trajectories if --all is used")
    parser.add_argument("--features", nargs="+", default=["obj:0"], help="Features to plot (e.g. 'obj:0', 'base:2' or '0')")
    parser.add_argument("--xy", action="store_true", help="Plot first two features against each other (XY plot)")
    parser.add_argument("--obj_disp", action="store_true", help="Plot object displacement (obj pos - initial obj pos) yaw-rotated to the initial pelvis frame")
    parser.add_argument("--save_path", default="trajectory_plot.png", help="Path to save plot")
    parser.add_argument("--dataset_paths", nargs="+", type=str, default=None, help="One or more dataset paths (folders or .npz files) to overlay. Legend uses folder names.")
    args = parser.parse_args()
    
    # Build list of (dataset, label) pairs
    if args.dataset_paths:
        datasets_and_labels = []
        for dp in args.dataset_paths:
            label = os.path.basename(os.path.normpath(dp.replace(".npz", "")))
            model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml")
            norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
            # If it's a .npz file, pass its parent dir and let the dataset find it
            # But we need to bypass task_list logic
            data_cfg_copy = dict(data_cfg)
            data_cfg_copy.pop("task_list_path", None)  # Don't use task list for explicit paths
            data_buf = preload_dataset(data_cfg_copy, dp)
            ds = BufferDataset(
                data_buffer=data_buf,
                config=data_cfg_copy,
                norm_path=norm_path,
                calculate_stats=False,
            )
            datasets_and_labels.append((ds, label))
    else:
        dataset = load_dataset()
        # Group file_paths by parent folder name
        # For a single dataset, each file_idx maps to a file_path
        datasets_and_labels = [(dataset, "default")]

    if args.obj_disp:
        _plot_obj_disp(args, datasets_and_labels)
    elif args.xy:
        _plot_xy(args, datasets_and_labels)
    else:
        _plot_timeseries(args, datasets_and_labels)


def _get_unique_trajs_and_indices(dataset, args):
    """Return unique (file_idx, batch_idx) pairs and the indices to plot."""
    unique_trajs = sorted(list(set((f, b) for f, b, t in dataset.indices)))
    
    if args.all:
        idxs = list(range(len(unique_trajs)))
        if args.max_trajs < len(idxs):
            print(f"Limiting to first {args.max_trajs} trajectories.")
            idxs = idxs[:args.max_trajs]
    elif args.indices:
        idxs = args.indices
    else:
        idxs = [0]
        print("No indices specified, plotting index 0.")
    
    return unique_trajs, idxs


def _plot_obj_disp(args, datasets_and_labels):
    """Plot object displacement yaw-rotated to initial pelvis frame."""
    plt.figure(figsize=(10, 10))
    
    # Assign a colour to each (dataset_label, file_idx) group
    color_cycle = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    color_idx = 0
    legend_handles = {}
    
    for dataset, ds_label in datasets_and_labels:
        unique_trajs, idxs_to_plot = _get_unique_trajs_and_indices(dataset, args)
        print(f"[{ds_label}] Plotting {len(idxs_to_plot)} trajectories...")
        
        # Group trajectories by file_idx for colouring
        file_groups = {}
        for idx_i in idxs_to_plot:
            if idx_i >= len(unique_trajs):
                print(f"Index {idx_i} out of bounds ({len(unique_trajs)} total).")
                continue
            f_idx, b_idx = unique_trajs[idx_i]
            file_groups.setdefault(f_idx, []).append((idx_i, f_idx, b_idx))
        
        for f_idx, group in file_groups.items():
            # Determine label: use dataset label if multiple datasets, else folder name
            if len(datasets_and_labels) > 1:
                file_label = ds_label
            else:
                file_label = f"file_{f_idx}"
            
            color = color_cycle[color_idx % len(color_cycle)]
            
            for idx_i, fi, bi in tqdm(group, desc=f"  {file_label}"):
                try:
                    raw = dataset._get_single_traj(fi, bi)
                    
                    # Get object position (T, 3+) and base quaternion
                    if 'obj' not in raw or 'base' not in raw:
                        print(f"  Skipping traj {idx_i}: missing 'obj' or 'base' key.")
                        continue
                    
                    obj_pos = raw['obj'][:, :3]        # (T, 3)
                    base_quat = raw['base'][:, 3:7]    # (T, 4) wxyz
                    
                    # Initial pelvis yaw
                    initial_yaw = yaw_from_quat(base_quat[0])
                    # Inverse rotation: rotate by -yaw
                    R_inv = yaw_to_rot_matrix(-initial_yaw)  # (3, 3)
                    
                    # Object displacement relative to initial object position
                    disp = obj_pos - obj_pos[0:1, :]  # (T, 3)
                    
                    # Rotate into initial pelvis frame
                    disp_rot = (R_inv @ disp.T).T  # (T, 3)
                    
                    label_for_plot = file_label if file_label not in legend_handles else None
                    line, = plt.plot(disp_rot[:, 0], disp_rot[:, 1], color=color, alpha=0.5, label=label_for_plot)
                    if file_label not in legend_handles:
                        legend_handles[file_label] = line
                    
                    # Mark start
                    plt.plot(disp_rot[0, 0], disp_rot[0, 1], 'o', color=color, markersize=4, alpha=0.7)
                    # Mark end
                    plt.plot(disp_rot[-1, 0], disp_rot[-1, 1], 'x', color=color, markersize=5, alpha=0.7)
                    
                except Exception as e:
                    print(f"  Error loading traj {idx_i} ({fi}, {bi}): {e}")
            
            color_idx += 1
    
    plt.xlabel("X displacement (pelvis frame)")
    plt.ylabel("Y displacement (pelvis frame)")
    plt.title("Object Displacement in Initial Pelvis Frame")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_path)
    plt.show()
    print(f"Saved plot to {args.save_path}")


def _plot_xy(args, datasets_and_labels):
    """XY scatter/line plot of first two features."""
    if len(args.features) < 2:
        print("Error: --xy requires at least 2 features.")
        return
    
    feat_x = args.features[0]
    feat_y = args.features[1]
    
    plt.figure(figsize=(10, 10))
    color_cycle = plt.cm.tab10.colors
    color_idx = 0
    legend_handles = {}
    
    for dataset, ds_label in datasets_and_labels:
        unique_trajs, idxs_to_plot = _get_unique_trajs_and_indices(dataset, args)
        
        file_groups = {}
        for idx_i in idxs_to_plot:
            if idx_i >= len(unique_trajs):
                continue
            f_idx, b_idx = unique_trajs[idx_i]
            file_groups.setdefault(f_idx, []).append((idx_i, f_idx, b_idx))
        
        for f_idx, group in file_groups.items():
            if len(datasets_and_labels) > 1:
                file_label = ds_label
            else:
                file_label = _get_file_label(dataset, f_idx) or ds_label
            
            color = color_cycle[color_idx % len(color_cycle)]
            
            for idx_i, fi, bi in group:
                try:
                    raw = dataset._get_single_traj(fi, bi)
                    vx = get_feature(raw, feat_x)
                    vy = get_feature(raw, feat_y)
                    min_len = min(len(vx), len(vy))
                    
                    label_for_plot = file_label if file_label not in legend_handles else None
                    line, = plt.plot(vx[:min_len], vy[:min_len], color=color, alpha=0.5, label=label_for_plot)
                    if file_label not in legend_handles:
                        legend_handles[file_label] = line
                except Exception as e:
                    print(f"Error loading {fi}, {bi}: {e}")
            
            color_idx += 1
    
    plt.xlabel(feat_x)
    plt.ylabel(feat_y)
    plt.title(f"XY Plot: {feat_x} vs {feat_y}")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_path)
    plt.show()
    print(f"Saved plot to {args.save_path}")


def _plot_timeseries(args, datasets_and_labels):
    """Time series subplot for each feature."""
    n_feats = len(args.features)
    fig, axes = plt.subplots(n_feats, 1, figsize=(10, 5*n_feats), squeeze=False)
    
    color_cycle = plt.cm.tab10.colors
    
    for feat_i, feat_spec in enumerate(args.features):
        ax = axes[feat_i, 0]
        color_idx = 0
        legend_handles = {}
        
        for dataset, ds_label in datasets_and_labels:
            unique_trajs, idxs_to_plot = _get_unique_trajs_and_indices(dataset, args)
            
            file_groups = {}
            for idx_i in idxs_to_plot:
                if idx_i >= len(unique_trajs):
                    continue
                f_idx, b_idx = unique_trajs[idx_i]
                file_groups.setdefault(f_idx, []).append((idx_i, f_idx, b_idx))
            
            for f_idx, group in file_groups.items():
                if len(datasets_and_labels) > 1:
                    file_label = ds_label
                else:
                    file_label = _get_file_label(dataset, f_idx) or ds_label
                
                color = color_cycle[color_idx % len(color_cycle)]
                
                for idx_i, fi, bi in group:
                    try:
                        raw = dataset._get_single_traj(fi, bi)
                        seq = get_feature(raw, feat_spec)
                        label_for_plot = file_label if file_label not in legend_handles else None
                        line, = ax.plot(seq, color=color, alpha=0.5, label=label_for_plot)
                        if file_label not in legend_handles:
                            legend_handles[file_label] = line
                    except Exception as e:
                        print(f"Error loading {fi}, {bi}: {e}")
                
                color_idx += 1
        
        ax.set_title(f"Feature: {feat_spec}")
        ax.set_ylabel("Value")
        ax.set_xlabel("Time step")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(args.save_path)
    plt.show()
    print(f"Saved plot to {args.save_path}")

if __name__ == "__main__":
    main()

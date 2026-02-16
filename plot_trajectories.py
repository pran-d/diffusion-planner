import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add current dir to path if needed for module imports
sys.path.append(os.getcwd())

from tqdm import tqdm

from config.configure import load_config, get_data_path, get_norm_path
from datasets.flexible_dataset import FlexibleWindowDataset

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
    dataset = FlexibleWindowDataset(
        data_root=data_path, 
        config=data_cfg, 
        norm_path=norm_path,
        calculate_stats=False
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
    parser.add_argument("--save_path", default="trajectory_plot.png", help="Path to save plot")
    args = parser.parse_args()
    
    dataset = load_dataset()
    unique_trajs = sorted(list(set((f, b) for f, b, t in dataset.indices)))
    
    idxs_to_plot = []
    if args.all:
        idxs_to_plot = range(len(unique_trajs))
        if args.max_trajs < len(idxs_to_plot):
            print(f"Limiting to first {args.max_trajs} trajectories (use --max_trajs to change).")
            idxs_to_plot = idxs_to_plot[:args.max_trajs]
    elif args.indices:
        idxs_to_plot = args.indices
    else:
        idxs_to_plot = [0]
        print("No indices specified, plotting index 0.")
        
    print(f"Plotting {len(idxs_to_plot)} trajectories...")
    
    # Collect data
    data_per_feature = {k: [] for k in args.features}
    
    for idx_i in tqdm(idxs_to_plot):
        if idx_i >= len(unique_trajs):
            print(f"Index {idx_i} out of bounds ({len(unique_trajs)} total).")
            continue
            
        f_idx, b_idx = unique_trajs[idx_i]
        try:
            raw = dataset._get_single_traj(f_idx, b_idx)
            for feat_spec in args.features:
                val = get_feature(raw, feat_spec)
                data_per_feature[feat_spec].append(val)
        except Exception as e:
            print(f"Error loading {f_idx}, {b_idx}: {e}")

    # Plotting
    if args.xy:
        if len(args.features) < 2:
            print("Error: --xy requires at least 2 features.")
            return
        
        feat_x = args.features[0]
        feat_y = args.features[1]
        
        vals_x = data_per_feature[feat_x]
        vals_y = data_per_feature[feat_y]
        
        plt.figure(figsize=(10, 10))
        for vx, vy in zip(vals_x, vals_y):
            # Ensure lengths match
            min_len = min(len(vx), len(vy))
            plt.plot(vx[:min_len], vy[:min_len], alpha=0.5)
            
        plt.xlabel(feat_x)
        plt.ylabel(feat_y)
        plt.title(f"XY Plot: {feat_x} vs {feat_y}")
        plt.axis('equal')
        plt.grid(True)
        
    else:
        # Time series / Subplots
        n_feats = len(args.features)
        fig, axes = plt.subplots(n_feats, 1, figsize=(10, 5*n_feats), squeeze=False)
        
        for i, feat_spec in enumerate(args.features):
            ax = axes[i, 0]
            vals = data_per_feature[feat_spec]
            if not vals: continue
            
            for seq in vals:
                ax.plot(seq, alpha=0.5)
            ax.set_title(f"Feature: {feat_spec}")
            ax.set_ylabel("Value")
            ax.set_xlabel("Time step")
            ax.grid(True)
            
    plt.tight_layout()
    plt.savefig(args.save_path)
    plt.show()
    print(f"Saved plot to {args.save_path}")

if __name__ == "__main__":
    main()

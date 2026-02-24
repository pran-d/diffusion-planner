
import argparse
import time
import numpy as np
import torch
import os

from config.configure import load_config, get_data_path, get_norm_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.visualize.visualize import MjVisualizer
from utils.math.sbto_utils import batch_rotation, quat_to_rot

def load_env_and_data():
    # Use config from file
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
    return data_cfg, dataset

def main():
    parser = argparse.ArgumentParser("Visualize Dataset Trajectories")
    parser.add_argument("--start_idx", type=int, default=0, help="Index of unique trajectory to start from")
    parser.add_argument("--overlay", action="store_true", help="Overlay all object paths and hide robot")
    parser.add_argument("--show_ee", action="store_true", help="Visualize computed EE paths")
    args = parser.parse_args()

    data_cfg, dataset = load_env_and_data()

    # Identify unique (file_idx, batch_idx) pairs to visualize full trajectories
    # dataset.indices is list of (file_idx, batch_idx, t_start)
    # We want unique (file_idx, batch_idx)
    unique_trajs = sorted(list(set((f, b) for f, b, t in dataset.indices)))
    
    print(f"Found {len(unique_trajs)} unique trajectories in dataset.")

    # Pre-load all trajectories for overlay
    all_trajs_data = []
    all_obj_paths = []
    
    if args.overlay:
        print("Pre-loading ALL trajectories for overlay...")
        from tqdm import tqdm
        for f_idx, b_idx in tqdm(unique_trajs):
            try:
                raw = dataset._get_single_traj(f_idx, b_idx)
                all_trajs_data.append(raw)
                if 'obj' in raw:
                    # Ensure numpy array
                    path = raw['obj'][:, :3] if isinstance(raw['obj'], np.ndarray) else np.array(raw['obj'])[:, :3]
                    all_obj_paths.append(path)
                else:
                    all_obj_paths.append(np.zeros((1, 3)))
            except Exception as e:
                print(f"Error loading {f_idx}, {b_idx}: {e}")
                all_trajs_data.append(None)
                all_obj_paths.append(np.zeros((1, 3)))
    else:
        # Just lazy load in loop (or just load standard way - but we already refactored loop to use all_trajs_data, so we should populate it or reverting logic)
        # To minimize code change, let's just populate all_trajs_data but maybe skip obj paths if not needed? 
        # Actually loading 1000 files is fast. Let's just do it consistently but only compute paths if needed.
        # Wait, if !args.overlay, we don't need all_obj_paths.
        print("Pre-loading trajectories...")
        from tqdm import tqdm
        for f_idx, b_idx in tqdm(unique_trajs):
             # We need to load data anyway for the loop structure we committed
             try:
                raw = dataset._get_single_traj(f_idx, b_idx)
                all_trajs_data.append(raw)
             except:
                all_trajs_data.append(None)

    # Initialize Visualizer
    xml_path = "./mj_model.xml"
    if not os.path.exists(xml_path):
         print(f"Warning: {xml_path} not found. Visualization might fail.")
    
    visualizer = MjVisualizer(xml_path, close_on_enter=False) # We handle advancement manually
    
    for i in range(args.start_idx, len(unique_trajs)):
        file_idx, batch_idx = unique_trajs[i]
        
        print(f"\n[{i}/{len(unique_trajs)}] Visualizing File {file_idx}, Batch {batch_idx}")
        
        # Get Full Raw Trajectory
        raw_traj = all_trajs_data[i]
        if raw_traj is None: continue
        
        # Extract components
        # Expected shapes: (T, 7), (T, 29), (T, 7)
        base = raw_traj['base']
        joints = raw_traj['joints']
        obj = raw_traj['obj']
        
        # Ensure length consistency
        T = min(len(base), len(joints), len(obj))
        base = base[:T]
        joints = joints[:T]
        obj = obj[:T]
        
        # Concatenate for visualization: [Base(7) | Joints(29) | Object(7)]
        # Total 43 dims
        full_state = np.concatenate([base, joints, obj], axis=-1)
        
        # Handle Overlay / Hide Robot
        current_overlay = None
        if args.overlay:
            # Hide robot by moving it far away
            # Base position is first 3 indices. Set Z to -100
            full_state[:, 2] = -100.0
            current_overlay = all_obj_paths
        
        if args.show_ee and 'ee_rel_pos' in raw_traj:
            # Visualize EE paths
            # ee_rel_pos: (T, num_ees, 3)
            # base: (T, 7)
            # P_ee = P_base + R_base @ P_rel
            
            ee_rel = raw_traj['ee_rel_pos']
            T_ee = min(len(base), len(ee_rel))
            
            base_pos = base[:T_ee, :3]
            base_quat = base[:T_ee, 3:]
            
            base_rot = quat_to_rot(base_quat) # (T_ee, 3, 3)
            
            ee_global_paths = []
            
            # ee_rel is (T, num_ees * 3). Reshape to (T, num_ees, 3)
            if ee_rel.ndim == 2:
                num_dims = ee_rel.shape[1]
                num_ees = num_dims // 3
                ee_rel_reshaped = ee_rel[:T_ee].reshape(T_ee, num_ees, 3)
            else:
                 # In case it was saved as (T, N, 3)
                 ee_rel_reshaped = ee_rel[:T_ee]
                 num_ees = ee_rel_reshaped.shape[1]

            for j in range(num_ees):
                rel_pos = ee_rel_reshaped[:T_ee, j, :] # (T_ee, 3)
                
                # Transform each point
                # batch_rotation expects R (..., 3, 3) and vectors (..., 3)
                
                ee_pos_delta = batch_rotation(base_rot, rel_pos)
                ee_pos_global = base_pos + ee_pos_delta
                
                ee_global_paths.append(ee_pos_global)
            
            if current_overlay is None:
                current_overlay = list(ee_global_paths)
            else:
                # If current_overlay is all_obj_paths (which is a list of arrays), we should probably not mutate it directly if we loop?
                # But we are inside the loop. 
                # If we extend all_obj_paths, it will grow every iteration!
                if args.overlay:
                    # Create a new list combining both
                    current_overlay = all_obj_paths + ee_global_paths
                else:
                    current_overlay.extend(ee_global_paths)
        
        print(f"  Trajectory length: {T}, Shape: {full_state.shape}")
        if 'fps' in raw_traj:
            print(f"  FPS: {raw_traj['fps']}")
        
        # Create time array
        dt = visualizer.mj_model.opt.timestep if visualizer.mj_model else 0.02
        t = np.linspace(0, T * dt, T)
        
        # Visualize
        # We need a way to keep visualizing until user input.
        # MjVisualizer.visualize_trajectory usually blocks or loops?
        # If repeat=True, it loops.
        # But we need to break that loop on user input.
        # However, standard input (input()) blocks the python process, preventing loop updates if running in same thread.
        # MjVisualizer usually spawns a viewer.
        # If we use `visualizer.visualize_trajectory(..., repeat=True)`, it likely blocks until window closed?
        # Let's assume the user wants to see it loop, then close window or Interrupt?
        # Or we can run standard open_loop visualization.
        
        # Hack: Just run it. The user effectively "Presses Enter" by closing the window or satisfy with terminal input if non-blocking.
        # If visualize_trajectory is NON-blocking, we hit input() immediately.
        # If it IS blocking, we wait for window close.
        # Let's try running it.
        
        visualizer.visualize_trajectory(
            t, 
            full_state, 
            repeat=True, 
            overlay_paths=current_overlay if args.overlay else None,
            markers=ee_global_paths if args.show_ee and 'ee_rel_pos' in raw_traj else None
        )

if __name__ == "__main__":
    main()

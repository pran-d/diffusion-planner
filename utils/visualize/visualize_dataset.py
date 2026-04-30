
import argparse
import time
import numpy as np
import torch
import os

from config.configure import load_config, get_data_path, get_norm_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.visualize.visualize import MjVisualizer
from utils.math.sbto_utils import batch_rotation, quat_to_rot
from utils.data.load_dataset import preload_dataset

def load_env_and_data():
    # Use config from file
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml")

    # For this visualizer, only load canonical per-task trajectory files.
    # data_cfg["file_names"] = ["best_trajectory.npz"]

    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    
    print("Loading dataset...")
    print(f"Filtering dataset files by name: {data_cfg['file_names']}")
    # Initialize dataset
    data_buffer = preload_dataset(data_cfg, data_path)
    dataset = FlexibleWindowDataset(
        data_buffer=data_buffer, 
        config=data_cfg, 
        norm_path=norm_path,
        calculate_stats=False,
        training_cfg={}
    )
    return data_cfg, dataset

def main():
    parser = argparse.ArgumentParser("Visualize Dataset Trajectories")
    parser.add_argument("--start_idx", type=int, default=0, help="Index of unique trajectory to start from")
    parser.add_argument("--overlay", action="store_true", help="Overlay all object paths and hide robot")
    parser.add_argument("--traj_idx", type=int, default=None, help="Trajectory (file) index")
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch index within file")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time in seconds for visualization")
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


    # Initialize Visualizer
    xml_path = "./unitree_g1/mj_model.xml"
    if not os.path.exists(xml_path):
         print(f"Warning: {xml_path} not found. Visualization might fail.")
    
    visualizer = MjVisualizer(xml_path, close_on_enter=False) # We handle advancement manually
    
    if args.traj_idx is not None:
        target = (args.traj_idx, args.batch_idx)
        try:
            args.start_idx = unique_trajs.index(target)
            print(f"Mapped {target} -> Sample {args.start_idx}")
        except ValueError:
            print(f"Error: Target {target} not found in unique trajectories.")
            raise ValueError(f"Target indices {target} (Trajectory {args.traj_idx}, Batch {args.batch_idx}) not present in valid window list.")


    print("Controls: Space=pause/play, Left/Right=step, Esc=next trajectory, N=next file @ batch 0")

    i = args.start_idx
    while i < len(unique_trajs):
        file_idx, batch_idx = unique_trajs[i]
        file_meta = dataset.ram_cache[file_idx] if 0 <= file_idx < len(dataset.ram_cache) else {}
        source_folder = file_meta.get("source_folder_name", "unknown")
        source_file = os.path.basename(file_meta.get("source_file_path", ""))
        
        print(f"\n[{i}/{len(unique_trajs)}] Visualizing File {file_idx}, Batch {batch_idx}")
        print(f"  Source: folder='{source_folder}' file='{source_file or 'unknown'}'")
        
        # Get Full Raw Trajectory
        raw_traj = dataset._get_single_traj(file_idx, batch_idx)
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

        # 'N' requests jump to the next file's batch 0 trajectory.
        if getattr(visualizer, "next_file_batch0_request", {}).get("active", False):
            visualizer.next_file_batch0_request["active"] = False
            next_i = None
            for j in range(i + 1, len(unique_trajs)):
                next_file_idx, next_batch_idx = unique_trajs[j]
                if next_file_idx > file_idx and next_batch_idx == 0:
                    next_i = j
                    break

            if next_i is None:
                print("No later (file_idx, batch_idx=0) found. Exiting viewer.")
                break

            i = next_i
            continue

        i += 1

if __name__ == "__main__":
    main()

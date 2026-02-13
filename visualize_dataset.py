
import argparse
import time
import numpy as np
import torch
import os

from config.configure import load_config, get_data_path, get_norm_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.visualize.visualize import MjVisualizer

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
    args = parser.parse_args()

    data_cfg, dataset = load_env_and_data()

    # Identify unique (file_idx, batch_idx) pairs to visualize full trajectories
    # dataset.indices is list of (file_idx, batch_idx, t_start)
    # We want unique (file_idx, batch_idx)
    unique_trajs = sorted(list(set((f, b) for f, b, t in dataset.indices)))
    
    print(f"Found {len(unique_trajs)} unique trajectories in dataset.")
    
    # Initialize Visualizer
    xml_path = "./mj_model.xml"
    if not os.path.exists(xml_path):
         print(f"Warning: {xml_path} not found. Visualization might fail.")
    
    visualizer = MjVisualizer(xml_path, close_on_enter=False) # We handle advancement manually
    
    for i in range(args.start_idx, len(unique_trajs)):
        file_idx, batch_idx = unique_trajs[i]
        
        print(f"\n[{i}/{len(unique_trajs)}] Visualizing File {file_idx}, Batch {batch_idx}")
        
        # Get Full Raw Trajectory
        # _get_single_traj returns dict with keys like 'base', 'joints', 'obj'
        raw_traj = dataset._get_single_traj(file_idx, batch_idx)
        
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
        
        print(f"  Trajectory length: {T}, Shape: {full_state.shape}")
        if 'fps' in raw_traj:
            print(f"  FPS: {raw_traj['fps']}")
        
        # Create time array
        dt = visualizer.mj_model.opt.timestep if visualizer.mj_model else 0.02
        t = np.linspace(0, T * dt, T)
        
        # Visualize
        print("  Playing (Repeated)... Press Enter in terminal to play next, 'q' to quit.")
        
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
        
        visualizer.visualize_trajectory(t, full_state, repeat=True)
        
        # If non-blocking, we wait here.
        user_in = input(">> Next? (Enter/q): ")
        if user_in.strip().lower() == 'q':
            break

if __name__ == "__main__":
    main()

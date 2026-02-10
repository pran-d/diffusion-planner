import numpy as np
import torch
import os
import sys
from motion_generator import MotionGenerator
from utils.visualize.visualize import MjVisualizer

def test_real_pipeline():
    print("========================================")
    print("Testing Real Data Pipeline")
    print("========================================")

    config_path = "config/config.yaml"
    
    # Force CPU to avoid OOM
    device = "cuda"
    print(f"Using device: {device}")

    # Initialize
    generator = MotionGenerator(config_path=config_path, device=device)
    
    # Modify data config to limit load if possible? 
    # The dataset loader loads everything in the path.
    # Hopefully it's not too big.
    
    print("Calling fit() (1 epoch)...")
    # Save to temp dir
    results_dir = "results/test_real"
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct data path from config
    data_cfg = generator.data_cfg
    full_path = os.path.join(data_cfg['dir_path'], data_cfg['train_path'])
    print(f"Data path: {full_path}")

    generator.fit(data_source=full_path, epochs=10, save_path=results_dir)
    print("Fit completed.")
    
    # Test Generation
    print("Testing Generation...")
    
    # Grab a sample from the loaded dataset to use as initial condition
    if not generator.dataset or not generator.dataset.data_buffer:
        print("Error: Dataset empty after fit.")
        return

    sample_idx = 0
    sample_data = generator.dataset.data_buffer[sample_idx]
    
    # We need history. 
    # history_size is in config, usually 1?
    H = generator.data_cfg.get('state_history', 1)
    
    # Extract raw data. The loader already downsamples in _load_raw_trajectory, 
    # but here we are accessing the RAW buffer before downsampling/windowing?
    # No, FlexibleWindowDataset.__init__ loads files into data_buffer.
    # Then _load_raw_trajectory processes them.
    
    # FlexibleWindowDataset structure:
    # self.data_buffer is a list of raw dicts loaded from files.
    # They are likely full trajectories.
    
    # We need to manually slice the start to simulate history.
    # Let's approximate.
    
    body_pos = sample_data['body_pos_w'] # (T, 1, 3) probably
    body_quat = sample_data['body_quat_w'] # (T, 1, 4)
    obj_pos = sample_data['object_pos_w']
    obj_quat = sample_data['object_quat_w']
    
    # Flatten/Concatenate to match what 'inference' expects
    # Inference expects dictionary:
    # 'robot': (H, 36) -> 7 (base) + 29 (joints)
    # 'obj': (H, 7)
    
    # We need joints too
    joint_pos = sample_data['joint_pos']
    
    # Slice first H
    def get_slice(arr):
        return arr[:H]

    # Robot: base (7) + joints (29) = 36
    # base = pos(3) + quat(4)
    # raw data body_pos is (T, K, 3). K=1 usually.
    b_pos = body_pos[:H, 0, :]
    b_quat = body_quat[:H, 0, :]
    j_pos = joint_pos[:H]
    
    robot_hist = np.concatenate([b_pos, b_quat, j_pos], axis=-1) # (H, 36)
    
    o_pos = obj_pos[:H]
    o_quat = obj_quat[:H]
    obj_hist = np.concatenate([o_pos, o_quat], axis=-1) # (H, 7)
    
    init_cond = {
        'robot': robot_hist,
        'obj': obj_hist
    }
    
    # Goal condition
    # For now, let's take the LAST frame object pose as goal
    goal_pos = obj_pos[-1]
    goal_quat = obj_quat[-1]
    goal_cond = np.concatenate([goal_pos, goal_quat], axis=-1)
    
    print("Generating trajectory...")
    traj = generator.generate_trajectory(
        initial_condition=init_cond,
        goal_condition=goal_cond,
        num_samples=1
    )
    
    print(f"Generated trajectory shape: {traj.shape}")
    print("Test passed successfully.")

    # Visualization
    print("Visualizing trajectory... (Press ENTER to close window, SPACE to pause)")
    try:
        xml_path = "mj_model.xml"
        if not os.path.exists(xml_path):
             print(f"Warning: {xml_path} not found. Skipping visualization.")
             return

        vis = MjVisualizer(xml_path)
        
        # traj is (1, T, 43). Take first sample.
        traj_sample = traj[0] # (T, 43)
        
        # Create time array (assuming 0.01 dt or similar, but generic is fine)
        T_steps = traj_sample.shape[0]
        t = np.arange(T_steps) * 0.1 # dummy time
        
        vis.visualize_trajectory(
            t=t,
            x_traj=traj_sample,
            repeat=True,
            goal_pos=goal_pos,
            goal_quat=goal_quat
        )
        vis.close()
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    test_real_pipeline()

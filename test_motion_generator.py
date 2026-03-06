import numpy as np
import torch
import unittest
import sys
import os
import matplotlib.pyplot as plt
from motion_generator import MotionGenerator
from utils.visualize.visualize import MjVisualizer
from config.configure import get_mj_xml_paths

def run_visualization(stitched_trajs, xml_path):
    # Ensure visualizer can find the XML
    vis = MjVisualizer(xml_path, close_on_enter=False)
    print("Optimization Complete. Visualizing generated sample...")
    print("Controls: SPACE=Pause, ARROWS=Step, ESC=Exit")
    
    # Use first sample (num_samples, T, D) -> (T, D)
    if stitched_trajs.ndim == 3:
        traj = stitched_trajs[0] 
    else:
        traj = stitched_trajs

    T_steps = traj.shape[0]
    t = np.arange(T_steps) * 0.01

    vis.visualize_trajectory(t=t, x_traj=traj, repeat=True)

    vis.close()

import glob
from tqdm import tqdm

def load_real_data(data_root):
    print(f"Loading data from {data_root}...")
    file_paths = sorted(glob.glob(os.path.join(data_root, "**/*.npz"), recursive=True))
    
    if not file_paths:
        raise ValueError(f"No .npz files found in {data_root}")
        
    buffer_lists = {}
    
    # RL Schema keys
    rl_keys = ['body_pos_w', 'body_quat_w', 'joint_pos', 'object_pos_w', 'object_quat_w', 'task_params']
    # SBTO Schema keys
    sbto_keys = ['base_xyz_quat', 'actuator_pos', 'obj_0_xyz_quat']
    
    for fpath in tqdm(file_paths[:10]): # Limit to 10 files for speed in testing? Or load all? 
                                        # Let's load 50 to be safe for batch sizes
        try:
            with np.load(fpath, allow_pickle=True) as data:
                keys_found = [k for k in rl_keys if k in data]
                if not keys_found:
                    keys_found = [k for k in sbto_keys if k in data]
                
                if not keys_found:
                    continue
                
                # Check for batch dim
                is_batched = False
                if 'body_pos_w' in data and data['body_pos_w'].ndim == 4: is_batched = True
                elif 'base_xyz_quat' in data and data['base_xyz_quat'].ndim == 3: is_batched = True
                
                for k in keys_found:
                    arr = data[k]
                    # If not batched, add batch dim
                    if not is_batched:
                        arr = arr[None, ...]
                    
                    if k not in buffer_lists:
                        buffer_lists[k] = []
                    buffer_lists[k].append(arr)
                    
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    # Concatenate
    final_buffer = {}
    for k, v in buffer_lists.items():
        final_buffer[k] = np.concatenate(v, axis=0)

    print("Buffer Loaded:")
    for k, v in final_buffer.items():
        print(f"  {k}: {v.shape}")
        
    return final_buffer

def test_pipeline():
    print("Initializing MotionGenerator...")
    # Initialize with default config, ensure CPU for test stability if no GPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    mg = MotionGenerator(config_path="config/config.yaml", device=device)
    
    # Allow overridng some config parameters for quicker test
    mg.training_cfg['save_every'] = 50
    mg.data_cfg['batch_size'] = 32 # Adjust batch size
    
    # Determine Data Path
    # Combine config paths
    data_cfg = mg.data_cfg
    full_path = os.path.join(data_cfg.get('dir_path', ''), data_cfg.get('train_path', ''))
    
    # Load Real Data
    buffer = load_real_data(full_path)
    
    print("Fitting Model...")
    # Train for 2 epochs
    mg.fit(buffer, epochs=50)
    
    print("Training complete. Testing Generation...")
    

    # Prepare Initial Condition (History)
    # History size is usually small (1 or 2)
    H = mg.dataset.history_size
    
    # Take first sample from buffer as initial condition
    idx = 0
    
    # Construct raw world vectors
    init_cond = {}
    
    if 'body_pos_w' in buffer:
        # RL Schema
        # body: (B, T, 1, 3) 
        bp = buffer['body_pos_w'][idx, :H]
        
        # Consistent with BufferDataset: if multiple bodies (T, N, 3), take index 0 (Base)
        if bp.ndim == 3: 
             if bp.shape[1] == 1: bp = bp[:, 0, :]
             else: bp = bp[:, 0, :] # Assume index 0 is base
        
        bq = buffer['body_quat_w'][idx, :H]
        if bq.ndim == 3:
             if bq.shape[1] == 1: bq = bq[:, 0, :]
             else: bq = bq[:, 0, :]
        
        jp = buffer['joint_pos'][idx, :H]
        # (H, 29)
        
        # Robot = Pos(3) + Quat(4) + Joints(29) = 36
        init_cond['robot'] = np.concatenate([bp, bq, jp], axis=-1)
        
        op = buffer['object_pos_w'][idx, :H]
        oq = buffer['object_quat_w'][idx, :H]
        # Obj = Pos(3) + Quat(4) = 7
        init_cond['obj'] = np.concatenate([op, oq], axis=-1)
        
        # Goal Condition logic
        if 'task_params' in buffer:
            goal_cond = buffer['task_params'][idx]
        else:
            # Fallback goal
            op_traj = buffer['object_pos_w'][idx] # (T, 3)
            goal_cond = (op_traj[-1] - op_traj[0])[:2]
        
    elif 'base_xyz_quat' in buffer:
        # SBTO Schema
        init_cond['robot'] = np.concatenate([
            buffer['base_xyz_quat'][idx, :H],
            buffer['actuator_pos'][idx, :H]
        ], axis=-1)
        
        init_cond['obj'] = buffer['obj_0_xyz_quat'][idx, :H]
        
        # Goal Condition logic
        if 'task_params' in buffer:
            goal_cond = buffer['task_params'][idx]
        else:
            op_traj = buffer['obj_0_xyz_quat'][idx, ..., :3]
            goal_cond = (op_traj[-1] - op_traj[0])[:2]
    
    else:
        raise ValueError("Unknown buffer format for creating initial condition")

    print(f"Goal Condition: {goal_cond}")
    
    # Generate
    traj = mg.generate_trajectory(
        initial_condition=init_cond,
        goal_condition=goal_cond,
        stitch_steps=24,
        num_samples=1
    )
    
    print(f"Generated Trajectory Shape: {traj.shape}")
    # (1, T_total, D) -> D should be 36+7=43
    
    # Visualize in MuJoCo
    model_xml, _ = get_mj_xml_paths()
    if os.path.exists(model_xml):
        print(f"Visualizing with {model_xml}")
        try:
            # We assume the visualizer works in this environment (with display)
            # If standard headless, this might fail or just open a window.
            run_visualization(traj, model_xml)
        except Exception as e:
            print(f"Visualization failed: {e}")
            # Fallback to plot
            print("Falling back to plot...")
            base_pos = traj[0, :, :3]
            obj_pos = traj[0, :, 36:39]
            
            plt.figure()
            plt.plot(base_pos[:, 0], base_pos[:, 1], label='Robot Base')
            plt.plot(obj_pos[:, 0], obj_pos[:, 1], label='Object')
            plt.title(f"Generated Trajectory (Goal: {goal_cond})")
            plt.legend()
            plt.savefig('test_generation.png')
            print("Saved plot to test_generation.png")
    else:
        print(f"XML not found at {model_xml}, skipping viz.")
    
if __name__ == "__main__":
    test_pipeline()

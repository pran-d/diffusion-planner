import argparse
import time
import numpy as np
import torch
import os
import copy
from tqdm import tqdm

from config.configure import load_config, get_data_path, get_norm_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset
from utils.visualize.visualize import MjVisualizer
from utils.math.sbto_utils import rot6d_to_rot, rot_to_quat, reconstruct_sbto_trajectory

# ===============================
# Helper Functions
# ===============================

def get_reconstructed_qpos(dataset, current_idx, force_clean=False):
    # Remember current noise setting
    orig_noise = getattr(dataset, "add_noise", False)
    
    if force_clean:
        dataset.add_noise = False
    else:
        dataset.add_noise = True
        
    sample = dataset[current_idx]
    
    # Restore original setting
    dataset.add_noise = orig_noise

    future_tensor = sample[0]
    anchor = sample[-1]
    
    # Denormalize full window
    if dataset.normalization_type == "min_max":
        g_min = dataset.global_min.to(future_tensor.device)
        g_max = dataset.global_max.to(future_tensor.device)
        denom = g_max - g_min
        denom[denom < 1e-6] = 1.0 
        future_denorm = future_tensor * denom + g_min
    else:
        g_mean = dataset.global_mean.to(future_tensor.device)
        g_std = dataset.global_std.to(future_tensor.device)
        future_denorm = future_tensor * g_std + g_mean
        
    # Reconstruct world state directly from the full feature vector
    ref_pos = anchor["ref_pos"].flatten()
    ref_quat = anchor["ref_quat"].flatten()
    base_pose_world = np.concatenate([ref_pos, ref_quat], axis=-1)[None, ...]

    future_traj = future_denorm.unsqueeze(0).cpu().numpy() # (1, T, D)
    
    robot_w, obj_w, _, _ = reconstruct_sbto_trajectory(base_pose_world, future_traj)
    
    # We want the 'history_size' frame as the current state!
    ref_idx = dataset.history_size - 1
    
    curr_robot = robot_w[0, ref_idx]
    curr_obj = obj_w[0, ref_idx]

    q = np.zeros(43) # Assuming H1 (36) + Object (7) layout from sbto_utils length mappings
    
    # Robot: Base pos (3), Base quat (4), Joints (29)
    q[0:7] = curr_robot[:7]
    q[7:36] = curr_robot[7:36]
    
    # Object: Base pos (3), Base quat (4)
    q[36:43] = curr_obj[:7]
    return q


# ===============================
# Main Loop
# ===============================

def main():
    parser = argparse.ArgumentParser("Visualize State Noise")
    parser.add_argument("--indices", type=int, nargs="+", default=[0], help="Indices of trajectories to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=5, help="Number of timesteps to visualize per trajectory")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    model_cfg, data_cfg, training_cfg, scheduler_cfg = load_config("config/config.yaml")
    
    # Ensure noise config is properly handled
    obs_noise_cfg = training_cfg.get("state_conditioning_noise_level", {})
    if not obs_noise_cfg:
        print("Warning: 'state_conditioning_noise_level' not found in config. Using defaults.")
        # Update training_cfg so dataset initializes it properly
        training_cfg["state_conditioning_noise_level"] = {
            "joints": 0.01, "body_z": 0.01, "body_rot6d": 0.01,
            "obj_rel_pos": 0.01, "obj_rel_rot6d": 0.01, "task_params": 0.01
        }
        
    training_cfg["add_obs_noise"] = True # We force true so FlexibleDataset creates the noise
    data_cfg["augmentation"]["mirror_symmetry"]["enabled"] = False # Disable mirroring for clean comparison

    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)

    print("Loading dataset...")
    data_buff = preload_dataset(data_cfg, data_path)
    dataset = FlexibleWindowDataset(
        data_buffer=data_buff, config=data_cfg, norm_path=norm_path,
        calculate_stats=False, training_cfg=training_cfg,
    )
    
    xml_path = "./unitree_g1/mj_model.xml"
    if not os.path.exists(xml_path):
         print(f"Warning: {xml_path} not found. Visualization might fail.")
    visualizer = MjVisualizer(xml_path)
    
    # Inject the clean_request dict if it wasn't added to the class yet
    if not hasattr(visualizer, "clean_request"):
        visualizer.clean_request = {"active": False}

    print("\n" + "="*40)
    print("CONTROLS:")
    print("  SPACE: Generate NEW noise and apply it to current state")
    print("  'C' Key: Toggle Clean (Original) vs Noisy state")
    print("  RIGHT ARROW: Next Timestep")
    print("  ESC: Quit")
    print("="*40 + "\n")

    for start_idx in args.indices:
        print(f"\n--- Visualizing Trajectory starting at Index {start_idx} for {args.steps} steps ---")
        
        for step_offset in range(args.steps):
            current_idx = start_idx + step_offset
            
            # Check bounds
            if current_idx >= len(dataset):
                print(f"Index {current_idx} out of bounds. Stopping.")
                break

            try:
                future, clean_state, task, anchor = dataset[current_idx] 
                
                # We only visualize the 'history' part which is usually the last observed state
                # If history=1, clean_state is (1, D). If history>1, (H, D).
                # We want to visualize the *current* state at this timestep.
                # Usually the last element of history is the current state.
                
                clean_state_denorm = denormalize_state(clean_state, dataset)
                clean_denorm_comps = decompose_state(clean_state_denorm, dataset, feature_dims)
                
                # Render the LAST frame of the history window as the "current" state
                # If history size is 1, this is just index 0.
                t_idx_to_show = clean_state_denorm.shape[0] - 1
                
                def get_qpos_from_comps(comps, t_idx):
                    """Maps components to mujoco qpos array."""
                    q = np.zeros(43) # Assuming H1 robot (36) + Object (7)
                    
                    # --- ROBOT STATE ---
                    # 1. Base Position
                    base_pos = np.zeros(3)
                    if 'delta_xy' in comps:
                        # Treat delta_xy as absolute XY for visualization
                        base_pos[:2] = comps['delta_xy'][t_idx].cpu().numpy()
                    if 'body_z' in comps:
                        base_pos[2] = comps['body_z'][t_idx].item()
                    q[0:3] = base_pos

                    # 2. Base Rotation
                    base_rot_mat = np.eye(3)
                    if 'body_rot6d' in comps:
                        r6_batch = comps['body_rot6d'][t_idx].unsqueeze(0)
                        # rot6d_to_rot returns (B, 3, 3)
                        base_rot_mat = rot6d_to_rot(r6_batch.cpu().numpy())[0] 
                        q[3:7] = rot_to_quat(base_rot_mat)
                    else:
                        q[3] = 1.0 # Identity quaternion (w=1)

                    # 3. Joints
                    if 'joints' in comps:
                        q[7:36] = comps['joints'][t_idx].cpu().numpy()

                    # --- OBJECT STATE ---
                    # Object state features are typically RELATIVE to the Robot Base
                    # We need to transform them to WORLD frame for MuJoCo visualization.
                    # P_obj_world = P_base_world + R_base_world @ P_obj_rel
                    
                    # 4. Object Position
                    if 'obj_rel_pos' in comps:
                        obj_pos_rel = comps['obj_rel_pos'][t_idx].cpu().numpy() # (3,)
                        # Apply Transform
                        obj_pos_world = base_pos + base_rot_mat @ obj_pos_rel
                        q[36:39] = obj_pos_world
                    else:
                        # Default far away if missing
                        q[36:39] = [10.0, 10.0, 0.0]

                    # 5. Object Rotation
                    # R_obj_world = R_base_world @ R_obj_rel
                    if 'obj_rel_rot6d' in comps:
                        r6_obj_rel = comps['obj_rel_rot6d'][t_idx].unsqueeze(0)
                        obj_rot_rel_mat = rot6d_to_rot(r6_obj_rel.cpu().numpy())[0]
                        
                        obj_rot_world_mat = base_rot_mat @ obj_rot_rel_mat
                        q[39:43] = rot_to_quat(obj_rot_world_mat)
                    else:
                        q[39] = 1.0

                    return q

                # Start with the clean state for this timestep
                q_clean = get_qpos_from_comps(clean_denorm_comps, t_idx_to_show)
                q_show = q_clean.copy()
                
                # Reset visualizer triggers for the new step
                visualizer.paused["active"] = False 
                visualizer.step_request["delta"] = 0
                
                print(f"\nStep {step_offset + 1}/{args.steps}: (Index {current_idx})")
                last_clean_state = None

                while True:
                    # React to Mode toggles for UI prints
                    is_clean = getattr(visualizer.clean_request, "get", lambda k,d: visualizer.clean_request[k])("active", False) if isinstance(visualizer.clean_request, dict) else False

                    if is_clean != last_clean_state:
                         print(f"  -> Mode = {'CLEAN' if is_clean else 'NOISY'}")
                         last_clean_state = is_clean

                    # Regenerate Noisy
                    if getattr(visualizer.paused, "get", lambda k,d: visualizer.paused[k])("active", False) if isinstance(visualizer.paused, dict) else False:
                         if isinstance(visualizer.paused, dict):
                            visualizer.paused["active"] = False
                         if not is_clean:
                              q_noisy = get_reconstructed_qpos(dataset, current_idx, force_clean=False)
                              print(f"  -> Space pressed: Generated NEW noise.")
                         else:
                              print(f"  -> Space pressed: Ignored (In CLEAN mode). Toggle 'C' first.")

                    q_show = q_clean if is_clean else q_noisy
                    
                    # Next Step
                    if getattr(visualizer.step_request, "get", lambda k,d: visualizer.step_request[k])("delta", 0) > 0 if isinstance(visualizer.step_request, dict) else False:
                        if isinstance(visualizer.step_request, dict):
                            visualizer.step_request["delta"] -= 1
                        break 

                    # Quit
                    if getattr(visualizer.exit_request, "get", lambda k,d: visualizer.exit_request[k])("active", False) if isinstance(visualizer.exit_request, dict) else False:
                            return

                    # Sync viewer
                    visualizer.update_data(q_show)
                    time.sleep(0.02)
                    
            except Exception as e:
                print(f"Error visualizing index {current_idx}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()

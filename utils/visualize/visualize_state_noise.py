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

def get_feature_dims(dataset):
    """Returns a dictionary mapping feature name to its dimension size."""
    dims = {}
    for key in dataset.feature_order:
        # Check for min_key or mean_key depending on normalization
        if f"min_{key}" in dataset.stats:
            dims[key] = dataset.stats[f"min_{key}"].shape[0]
        elif f"mean_{key}" in dataset.stats:
             dims[key] = dataset.stats[f"mean_{key}"].shape[0]
    return dims

def denormalize_state(state_tensor, dataset):
    """Denormalizes a partial state vector using the global stats."""
    
    # We are denormalizing the *State*, which is a subset of the *Full Window*.
    # The global stats (mean/std/min/max) are concatenated for the *Full Feature Vector*.
    # We need to slice them to align with the state (last N observations).
    
    # State corresponds to the LAST obs_dim elements of the full feature vector
    obs_dim = dataset.num_observations
    total_dim = dataset.num_features
    # NOTE: In FlexibleDataset, if feature_order is just current observation features, total_dim might equal obs_dim.
    # But usually features = [delta_xy, delta_yaw, ..., joints, ...]
    # The *current state* usually omits the first few elements dependent on deltas from previous frame if they haven't happened yet? 
    # Or in the dataset logic:   current_state = window_tensor[:self.history_size, ...]
    # It takes the full feature vector?
    # No: current_state = window_tensor[:self.history_size, (self.num_features-self.num_observations):]
    # So it takes the LAST obs_dim columns.
    
    global_start_idx = total_dim - obs_dim

    if dataset.normalization_type == "min_max":
        if hasattr(dataset, 'global_min') and hasattr(dataset, 'global_max'):
             # Ensure indices are valid
             if global_start_idx < 0: return state_tensor
                 
             g_min = dataset.global_min[global_start_idx:].to(state_tensor.device)
             g_max = dataset.global_max[global_start_idx:].to(state_tensor.device)
             
             # FlexibleDataset uses: (val - min) / (max - min) -> [0, 1]
             # Denorm = val * (max - min) + min
             denom = g_max - g_min
             denom[denom < 1e-6] = 1.0 
             
             return state_tensor * denom + g_min
             
    else: # mean_std
        if hasattr(dataset, 'global_mean') and hasattr(dataset, 'global_std'):
            if global_start_idx < 0: return state_tensor
            
            g_mean = dataset.global_mean[global_start_idx:].to(state_tensor.device)
            g_std = dataset.global_std[global_start_idx:].to(state_tensor.device)
            
            return state_tensor * g_std + g_mean

    return state_tensor

def decompose_state(state, dataset, feature_dims):
    """Decomposes the flattened state vector back into its components."""
    obs_dim = dataset.num_observations
    total_dim = dataset.num_features
    
    feature_indices = {}
    curr_idx = 0
    ordered_features = []
    
    for key in dataset.feature_order:
        if key not in feature_dims: continue
        dim = feature_dims[key]
        feature_indices[key] = (curr_idx, curr_idx + dim)
        curr_idx += dim
        ordered_features.append(key)
        
    total_indices = curr_idx
    state_start_idx = total_indices - obs_dim
    
    decomposed = {}
    for key in ordered_features:
        f_start, f_end = feature_indices[key]
        overlap_start = max(f_start, state_start_idx)
        overlap_end = min(f_end, total_indices)
        
        if overlap_start < overlap_end:
            rel_start = overlap_start - state_start_idx
            rel_end = overlap_end - state_start_idx
            
            if state.ndim == 1:
                decomposed[key] = state[rel_start:rel_end]
            elif state.ndim == 2:
                 decomposed[key] = state[:, rel_start:rel_end]
            
    return decomposed

def apply_noise_to_components(components, dataset):
    """Applies noise to each component using dataset._add_obs_noise"""
    noisy_components = {}
    noise_mags = {}

    for key, val in components.items():
        val_noisy = dataset._add_obs_noise(val, key)
        noisy_components[key] = val_noisy
        diff = val_noisy - val
        mag = torch.norm(diff)
        noise_mags[key] = mag.item()
        
    return noisy_components, noise_mags

def reconstruct_state(components, dataset, feature_dims, base_tensor):
    """Reassembles the state vector by overwriting the base_tensor with components."""
    reconstructed = base_tensor.clone() # Use the clean state as the canvas!
    
    obs_dim = dataset.num_observations
    curr_idx = 0
    feature_limit_indices = {}
    
    for key in dataset.feature_order:
        if key not in feature_dims: continue
        dim = feature_dims[key]
        feature_limit_indices[key] = (curr_idx, curr_idx + dim)
        curr_idx += dim
        
    state_start_idx = curr_idx - obs_dim
    
    for key, val in components.items():
        f_start, f_end = feature_limit_indices[key]
        overlap_start = max(f_start, state_start_idx)
        overlap_end = min(f_end, curr_idx)
        
        if overlap_start < overlap_end:
            rel_start = overlap_start - state_start_idx
            rel_end = overlap_end - state_start_idx
            
            if reconstructed.ndim == 1:
                reconstructed[rel_start:rel_end] = val
            else:
                reconstructed[:, rel_start:rel_end] = val
                
    return reconstructed

def denormalize_state(state_tensor, dataset):
    """Denormalizes a partial state vector using the global stats."""
    
    # We are denormalizing the *State*, which is a subset of the *Full Window*.
    # The global stats (mean/std/min/max) are concatenated for the *Full Feature Vector*.
    # We need to slice them to align with the state (last N observations).
    
    obs_dim = dataset.num_observations
    total_dim = dataset.num_features
    # State corresponds to the LAST obs_dim elements of the full feature vector
    start_idx = total_dim - obs_dim

    if dataset.normalization_type == "min_max":
        if hasattr(dataset, 'global_min') and hasattr(dataset, 'global_max'):
             g_min = dataset.global_min[start_idx:].to(state_tensor.device)
             g_max = dataset.global_max[start_idx:].to(state_tensor.device)
             
             # FlexibleDataset uses: scalar * (val - min) / (max - min)
             # Wait, usually MinMax is (val - min)/(max - min) -> [0, 1]
             # OR 2*(...) - 1 -> [-1, 1].
             # Let's check FlexibleWindowDataset._normalize:
             # norm = (val - min_v) / denom. Result is [0, 1].
             
             # So Denorm = val * denom + min_v
             denom = g_max - g_min
             denom[denom < 1e-6] = 1.0 # Though global stats probably don't have this issue if computed well
             
             return state_tensor * denom + g_min
             
    else: # mean_std
        if hasattr(dataset, 'global_mean') and hasattr(dataset, 'global_std'):
            g_mean = dataset.global_mean[start_idx:].to(state_tensor.device)
            g_std = dataset.global_std[start_idx:].to(state_tensor.device)
            
            return state_tensor * g_std + g_mean

    return state_tensor


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
    
    obs_noise_cfg = training_cfg.get("state_conditioning_noise_level", {})
    if not obs_noise_cfg:
        print("Warning: 'state_conditioning_noise_level' not found in config. Using defaults.")
        obs_noise_cfg = {
            "joints": 0.01, "body_z": 0.01, "body_rot6d": 0.01,
            "obj_rel_pos": 0.01, "obj_rel_rot6d": 0.01, "task_params": 0.01
        }
    
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)

    print("Loading dataset...")
    # Clean dataset to start
    data_buff = preload_dataset(data_cfg, data_path)
    dataset = FlexibleWindowDataset(
        data_buffer=data_buff, config=data_cfg, norm_path=norm_path,
        calculate_stats=False, training_cfg=training_cfg,
    )
    
    feature_dims = get_feature_dims(dataset)
    
    xml_path = "./unitree_g1/mj_model.xml"
    if not os.path.exists(xml_path):
         print(f"Warning: {xml_path} not found. Visualization might fail.")
    visualizer = MjVisualizer(xml_path)
    
    # Inject the clean_request dict if it wasn't added to the class yet (safety fallback)
    if not hasattr(visualizer, "clean_request"):
        visualizer.clean_request = {"active": False}

    print("\n" + "="*40)
    print("CONTROLS:")
    print("  SPACE: Generate NEW noise and apply it to current state")
    print("  'C' Key: Revert to Clean (Original) state")
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
                visualizer.clean_request["active"] = False
                visualizer.step_request["delta"] = 0
                
                print(f"\nStep {step_offset + 1}/{args.steps}: Mode = CLEAN (Index {current_idx})")

                while True:
                    # --- CHECK SPACEBAR (Generate New Noise) ---
                    if visualizer.paused["active"]:
                        visualizer.paused["active"] = False # Consume toggle
                        
                        # 1. Safely CLONE the state to prevent PyTorch in-place mutation bugs
                        # Take the WHOLE history window to add noise consistently
                        clean_state_t = clean_state.clone()
                        
                        # 2. Decompose and add noise
                        clean_comps_t = decompose_state(clean_state_t, dataset, feature_dims)
                        noisy_comps_t, noise_mags = apply_noise_to_components(clean_comps_t, dataset)
                        
                        # 3. Reconstruct using the cloned clean state as the base tensor
                        noisy_state_t = reconstruct_state(noisy_comps_t, dataset, feature_dims, base_tensor=clean_state_t)
                        
                        # 4. Denormalize and map back to viewer
                        noisy_state_denorm_t = denormalize_state(noisy_state_t, dataset)
                        noisy_denorm_comps_t = decompose_state(noisy_state_denorm_t, dataset, feature_dims)
                        
                        # Update display state (visualize the last frame of the noisy history)
                        q_show = get_qpos_from_comps(noisy_denorm_comps_t, t_idx_to_show)
                        print(f"  -> Space pressed: Generated NEW noise. Mode = NOISY")

                    # --- CHECK 'C' KEY (Revert to Clean) ---
                    if getattr(visualizer, "clean_request", {}).get("active", False):
                        visualizer.clean_request["active"] = False # Consume toggle
                        q_show = q_clean.copy()
                        print(f"  -> 'C' pressed: Reverted. Mode = CLEAN")

                    # --- CHECK RIGHT ARROW (Next Step) ---
                    if visualizer.step_request["delta"] > 0:
                        visualizer.step_request["delta"] -= 1 # Consume toggle
                        # Move to next index in the outer loop
                        break 

                    # Check Exit
                    if getattr(visualizer, "exit_request", {}).get("active", False):
                            return

                    # Push state to mujoco
                    visualizer.update_data(q_show)
                    time.sleep(0.02)
                    
            except Exception as e:
                print(f"Error visualizing index {current_idx}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
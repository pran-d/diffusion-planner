import argparse
import time
import numpy as np
import torch
import os
import copy
from tqdm import tqdm

from config.configure import load_config, get_data_path, get_norm_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.visualize.visualize import MjVisualizer
from utils.math.sbto_utils import rot6d_to_rot, rot_to_quat, reconstruct_sbto_trajectory

# ===============================
# Helper Functions
# ===============================

def get_feature_dims(dataset):
    """Returns a dictionary mapping feature name to its dimension size."""
    dims = {}
    for key in dataset.feature_order:
        if f"min_{key}" in dataset.stats:
            dims[key] = dataset.stats[f"min_{key}"].shape[0]
    return dims

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
    """Denormalizes a partial state vector using the global max/min."""
    if not (hasattr(dataset, 'global_min') and hasattr(dataset, 'global_max')):
        return state_tensor
    obs_dim = dataset.num_observations
    total_dim = dataset.num_features
    start_idx = total_dim - obs_dim
    
    g_min = dataset.global_min[start_idx:].to(state_tensor.device)
    g_max = dataset.global_max[start_idx:].to(state_tensor.device)
    
    return ((state_tensor + 1) / 2) * (g_max - g_min) + g_min


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
    dataset = FlexibleWindowDataset(
        data_root=data_path, config=data_cfg, norm_path=norm_path,
        calculate_stats=False, add_noise=False, noise_cfg=obs_noise_cfg
    )
    
    feature_dims = get_feature_dims(dataset)
    
    xml_path = "./mj_model.xml"
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

    for idx in args.indices:
        print(f"\n--- Visualizing Trajectory Index {idx} ---")
        try:
            future, clean_state, task, anchor = dataset[idx] 
            
            clean_state_denorm = denormalize_state(clean_state, dataset)
            clean_denorm_comps = decompose_state(clean_state_denorm, dataset, feature_dims)
            
            timesteps_to_show = clean_state_denorm.shape[0]

            def get_qpos_from_comps(comps, t_idx):
                """Maps components to mujoco qpos array."""
                q = np.zeros(43)
                if 'body_z' in comps: q[2] = comps['body_z'][t_idx].item()
                if 'body_rot6d' in comps:
                    r6_batch = comps['body_rot6d'][t_idx].unsqueeze(0)
                    quat = rot_to_quat(rot6d_to_rot(r6_batch.cpu().numpy()))
                    q[3:7] = quat[0]
                else:
                    q[3] = 1.0 
                    
                if 'joints' in comps: q[7:36] = comps['joints'][t_idx].cpu().numpy()
                    
                if 'obj_rel_pos' in comps:
                    q[36:39] = comps['obj_rel_pos'][t_idx].cpu().numpy()
                    q[36] += 1.0 # Shift visually to avoid overlap
                    
                if 'obj_rel_rot6d' in comps:
                    r6_batch = comps['obj_rel_rot6d'][t_idx].unsqueeze(0)
                    quat = rot_to_quat(rot6d_to_rot(r6_batch.cpu().numpy()))
                    q[39:43] = quat[0]
                else:
                    q[39] = 1.0
                return q

            for t in range(timesteps_to_show):
                # Start with the clean state for this timestep
                q_clean = get_qpos_from_comps(clean_denorm_comps, t)
                q_show = q_clean.copy()
                
                # Reset visualizer triggers for the new step
                visualizer.paused["active"] = False 
                visualizer.clean_request["active"] = False
                visualizer.step_request["delta"] = 0
                
                print(f"\nStep {t}/{timesteps_to_show - 1}: Mode = CLEAN")

                while True:
                    # --- CHECK SPACEBAR (Generate New Noise) ---
                    if visualizer.paused["active"]:
                        visualizer.paused["active"] = False # Consume toggle
                        
                        # 1. Safely CLONE the state to prevent PyTorch in-place mutation bugs
                        clean_state_t = clean_state[t:t+1].clone()
                        
                        # 2. Decompose and add noise
                        clean_comps_t = decompose_state(clean_state_t, dataset, feature_dims)
                        noisy_comps_t, noise_mags = apply_noise_to_components(clean_comps_t, dataset)
                        
                        # 3. Reconstruct using the cloned clean state as the base tensor
                        noisy_state_t = reconstruct_state(noisy_comps_t, dataset, feature_dims, base_tensor=clean_state_t)
                        
                        # 4. Denormalize and map back to viewer
                        noisy_state_denorm_t = denormalize_state(noisy_state_t, dataset)
                        noisy_denorm_comps_t = decompose_state(noisy_state_denorm_t, dataset, feature_dims)
                        
                        # Update display state
                        q_show = get_qpos_from_comps(noisy_denorm_comps_t, 0)
                        print(f"  -> Space pressed: Generated NEW noise. Mode = NOISY")

                    # --- CHECK 'C' KEY (Revert to Clean) ---
                    if getattr(visualizer, "clean_request", {}).get("active", False):
                        visualizer.clean_request["active"] = False # Consume toggle
                        q_show = q_clean.copy()
                        print(f"  -> 'C' pressed: Reverted. Mode = CLEAN")

                    # --- CHECK RIGHT ARROW (Next Step) ---
                    if visualizer.step_request["delta"] > 0:
                        visualizer.step_request["delta"] -= 1 # Consume toggle
                        break # Exit while loop, move to next t

                    # Check Exit
                    if getattr(visualizer, "exit_request", {}).get("active", False):
                         return

                    # Push state to mujoco
                    visualizer.update_data(q_show)
                    time.sleep(0.02)
                    
        except Exception as e:
            print(f"Error visualizing index {idx}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
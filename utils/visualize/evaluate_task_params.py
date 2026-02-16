
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

from config.configure import load_config, get_data_path, get_norm_path
from models.model import RobotDiffuser
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.math.sbto_utils import reconstruct_sbto_trajectory

def parse_args():
    parser = argparse.ArgumentParser("Evaluate Task Params Accuracy")
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--stitch_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_w", type=float, default=1.0, help="Classifier-Free Guidance weight")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    parser.add_argument("--save_path", type=str, default="task_params_comparison.png", help="Path to save the comparison plot")
    
    # Guidance arguments
    parser.add_argument("--guidance_wt", type=float, default=0.0, help="Test-time gradient guidance strength")
    parser.add_argument("--guidance_goal", nargs="+", type=float, default=None, help="Target values for guidance (normalized)")
    parser.add_argument("--guidance_indices", nargs="+", type=int, default=None, help="Indices of the state vector to apply guidance on")
    
    return parser.parse_args()

def denormalize_task_params(dataset, norm_val):
    """
    Denormalize task parameters manually since they are not part of the global concatenated vector.
    norm_val: (B, D) or (D,) tensor in [-1, 1]
    """
    if "min_task_params" not in dataset.stats:
        return norm_val.cpu().numpy()
        
    min_v = dataset.stats["min_task_params"].to(norm_val.device)
    max_v = dataset.stats["max_task_params"].to(norm_val.device)
    
    # Denorm: x = (norm + 1)/2 * (max - min) + min
    denorm = ((norm_val + 1) / 2) * (max_v - min_v) + min_v
    return denorm.cpu().numpy()

def update_condition(dataset, robot_world_history, obj_world_history):
    """
    Construct next state condition from the end of the previous World-Frame segment.
    """
    B, H, _ = robot_world_history.shape
    next_states = []
    next_anchors = {'ref_pos': [], 'ref_quat': []}
    
    # We are usually batch size 1 here, but keeping it generic
    for b in range(B):
        r_slice = robot_world_history[b] # (H, 36)
        o_slice = obj_world_history[b]   # (H, 7)
        
        raw_chunk = {
            'base': r_slice[:, :7],       
            'joints': r_slice[:, 7:36],
            'obj': o_slice[:, :7]
        }
        
        feats, new_anch = dataset._compute_transform(raw_chunk, t_start=0)
        
        # Helper to normalize and extract obs
        current_parts = []
        obs_start_idx = dataset.num_features - dataset.num_observations
        cumulative_dim = 0
        
        for key in dataset.feature_order:
            if key in feats:
                part = torch.from_numpy(feats[key]).float()
                # Use dataset stats to normalize
                part = dataset._normalize(key, part) 
                
                part_dim = part.shape[-1]
                part_end = cumulative_dim + part_dim
                
                if part_end > obs_start_idx:
                    local_start = max(0, obs_start_idx - cumulative_dim)
                    current_parts.append(part[:, local_start:])
                
                cumulative_dim += part_dim
                
        c_state = torch.cat(current_parts, dim=-1) # (H, F_obs)
        c_state = c_state[:dataset.history_size] # Slice to history size
        next_states.append(c_state)
        # next_anchors uses new_anch which is single dict from _compute_transform
        # but _compute_transform returned dict of arrays, presumably matching window?
        # Check flexible_dataset.py _compute_transform return for anchor
        next_anchors['ref_pos'].append(new_anch['ref_pos'])
        next_anchors['ref_quat'].append(new_anch['ref_quat'])

    return torch.stack(next_states), {
        'ref_pos': np.stack(next_anchors['ref_pos']), 
        'ref_quat': np.stack(next_anchors['ref_quat'])
    }

def main():
    args = parse_args()
    
    # Device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load Config
    config_path = "config/config.yaml"
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config(config_path)
    
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    
    # Load Dataset
    print("Loading dataset...")
    dataset = FlexibleWindowDataset(
        data_root=data_path, 
        config=data_cfg, 
        norm_path=norm_path,
        calculate_stats=False
    )
    
    # Load Model
    diffuser = RobotDiffuser(
        model_config=model_cfg,
        data_config=data_cfg,
        training_config=training_cfg,
        noise_scheduler_config=noise_cfg,
        mode="infer",
        device=device,
    )
    diffuser.loadWeights(args.epoch)
    
    # Identify Trajectory Starts
    # We look for indices where t_start == 0
    start_indices = [i for i, (f, b, t) in enumerate(dataset.indices) if t == 0]
    print(f"Found {len(start_indices)} trajectory start points.")
    
    errors = []
    
    gt_params_list = []
    gen_params_list = []
    
    for idx in tqdm(start_indices, desc="Evaluating"):
        # 1. Get Initial Condition & GT Task
        # dataset[idx] -> future, current, task, anchor
        _, curr_state, gt_task_norm, anchor = dataset[idx]
        
        # print("DEBUG: curr_state shape:", curr_state.shape)

        # Prepare Tensors
        # (1, 1, F_obs) -> need (1, history, F_obs)?
        # dataset returns (history, F_obs) for curr_state
        curr_state_tens = curr_state.unsqueeze(0).to(device) # (1, H, F)
        gt_task_tens = gt_task_norm.unsqueeze(0).to(device)  # (1, 2)
        
        # print("DEBUG: curr_state_tens:", curr_state_tens.shape)
        # print("DEBUG: gt_task_tens:", gt_task_tens.shape)

        # Prepare Anchors
        current_anchors = {
            'ref_pos': anchor['ref_pos'][None, ...],   # (1, 3)
            'ref_quat': anchor['ref_quat'][None, ...],  # (1, 4)
            'ref_obj_pos': anchor['ref_obj_pos'][None, ...] # (1, 3)
        }
        
        generated_segments = []
        
        # 2. Autoregressive Loop
        for step in range(args.stitch_steps):
            # Model Forward
            sample = diffuser.getSample(
                num_trajectories=1,
                state_cond=curr_state_tens,
                goal_cond=gt_task_tens,
                deterministic=True,
                cfg_w=args.cfg_w,
                guidance_wt=args.guidance_wt,
            )
            
            # Denormalize
            denorm = dataset.denormalize_global(sample) # (1, T, C)
            future_traj = denorm.cpu().numpy()
            
            # Reconstruct (Local -> World)
            anchor_arr = np.concatenate([current_anchors['ref_pos'], current_anchors['ref_quat'], current_anchors['ref_obj_pos']], axis=-1)
            robot_world, obj_world, _, _ = reconstruct_sbto_trajectory(
                base_pose_world=anchor_arr,
                future_traj=future_traj,
                inpaint=diffuser.model_cfg.get("inpaint", False)
            )
            
            # Store Segment
            # (1, T, 36) + (1, T, 7) -> (1, T, 43)
            segment = np.concatenate([robot_world[..., :36], obj_world[..., :7]], axis=-1)
            generated_segments.append(segment)
            
            # Update Condition
            # If not last step, prepare next condition
            if step < args.stitch_steps - 1:
                hist_len = dataset.history_size
                # Take last H frames
                r_hist = robot_world[:, -hist_len:, :]
                o_hist = obj_world[:, -hist_len:, :]
                
                curr_state_tens, next_anch = update_condition(dataset, r_hist, o_hist)
                curr_state_tens = curr_state_tens.to(device)
                current_anchors = next_anch
                
        # 3. Analyze Trajectory
        full_traj = np.concatenate(generated_segments, axis=1) # (1, Total_T, 43)
        obj_traj_w = full_traj[0, :, 36:43] # (Total_T, 7)
        
        # Generated Task Params (Delta XY)
        gen_start = obj_traj_w[0, :2]
        gen_end = obj_traj_w[-1, :2]
        gen_param = gen_end - gen_start
        
        # Get GT Task Params (Real Scale)
        gt_param_real = denormalize_task_params(dataset, gt_task_norm)
        
        gt_params_list.append(gt_param_real)
        gen_params_list.append(gen_param)

        # Compute Error
        diff = gen_param - gt_param_real
        error = np.linalg.norm(diff)
        errors.append(error)
        
    # Report
    errors = np.array(errors)
    print("\nEvaluation Complete.")
    print(f"Stats over {len(errors)} trajectories:")
    print(f"Mean Error (L2): {np.mean(errors):.4f}")
    print(f"Std Error: {np.std(errors):.4f}")
    print(f"Min Error: {np.min(errors):.4f}")
    print(f"Max Error: {np.max(errors):.4f}")

    # Plotting
    try:
        gt_params = np.array(gt_params_list)
        gen_params = np.array(gen_params_list)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(gt_params[:, 0], gt_params[:, 1], c='blue', label='Desired (GT)', alpha=0.6)
        plt.scatter(gen_params[:, 0], gen_params[:, 1], c='red', label='Generated', alpha=0.6)
        
        # Draw lines connecting corresponding points
        for i in range(len(gt_params)):
            plt.plot([gt_params[i, 0], gen_params[i, 0]], 
                     [gt_params[i, 1], gen_params[i, 1]], 'gray', alpha=0.3)
            
        plt.xlabel("Task Param X (Delta Object X)")
        plt.ylabel("Task Param Y (Delta Object Y)")
        plt.title("Task Params: Desired vs Generated")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Center plot around 0,0 and fix axes
        all_vals = np.concatenate([gt_params, gen_params], axis=0)
        max_val = np.max(np.abs(all_vals)) * 1.1 # 10% padding
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.axvline(0, color='k', linewidth=0.5)
        
        save_path = args.save_path
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    main()

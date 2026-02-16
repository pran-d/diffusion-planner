
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config.configure import load_config, get_data_path, get_norm_path
from models.model import RobotDiffuser
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.math.sbto_utils import reconstruct_sbto_trajectory

def parse_args():
    parser = argparse.ArgumentParser("Visualize Task Param Sweep")
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--stitch_steps", type=int, default=1)
    parser.add_argument("--sample_idx", type=int, default=0, help="Initial condition index from dataset")
    parser.add_argument("--cfg_w", type=float, default=1.0, help="Classifier-Free Guidance weight")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="task_sweep.png")
    
    # Sweep Config
    parser.add_argument("--x_min", type=float, default=-0.5)
    parser.add_argument("--x_max", type=float, default=0.5)
    parser.add_argument("--y_min", type=float, default=-0.5)
    parser.add_argument("--y_max", type=float, default=0.5)
    parser.add_argument("--grid_size", type=int, default=5, help="Number of points per axis (total grid_size^2)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for inference")
    
    # Dataset Task Params
    parser.add_argument("--use_dataset_tasks", action="store_true", help="Use task parameters sampled from the dataset instead of a grid")
    parser.add_argument("--num_dataset_tasks", type=int, default=25, help="Number of tasks to sample from dataset")

    # Guidance
    parser.add_argument("--guidance_wt", type=float, default=0.0)
    
    return parser.parse_args()

def update_condition(dataset, robot_world_history, obj_world_history):
    """
    Update condition for next autoregressive step.
    Supports batch dimension.
    """
    B, H, _ = robot_world_history.shape
    next_states = []
    next_anchors = {'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': []}
    
    for b in range(B):
        r_slice = robot_world_history[b] # (H, 36)
        o_slice = obj_world_history[b]   # (H, 7)
        
        raw_chunk = {
            'base': r_slice[:, :7],       
            'joints': r_slice[:, 7:36],
            'obj': o_slice[:, :7]
        }
        
        feats, new_anch = dataset._compute_transform(raw_chunk, t_start=0)
        
        # Assemble Feature Vector
        current_parts = []
        obs_start_idx = dataset.num_features - dataset.num_observations
        cumulative_dim = 0
        
        for key in dataset.feature_order:
            if key in feats:
                part = torch.from_numpy(feats[key]).float()
                part = dataset._normalize(key, part) 
                
                part_dim = part.shape[-1]
                part_end = cumulative_dim + part_dim
                
                if part_end > obs_start_idx:
                    local_start = max(0, obs_start_idx - cumulative_dim)
                    current_parts.append(part[:, local_start:])
                
                cumulative_dim += part_dim
        
        # Concatenate and slice history
        c_state = torch.cat(current_parts, dim=-1)
        c_state = c_state[:dataset.history_size] 
        next_states.append(c_state)
        
        next_anchors['ref_pos'].append(new_anch['ref_pos'])
        next_anchors['ref_quat'].append(new_anch['ref_quat'])
        next_anchors['ref_obj_pos'].append(new_anch['ref_obj_pos'])

    return torch.stack(next_states), {
        'ref_pos': np.stack(next_anchors['ref_pos']), 
        'ref_quat': np.stack(next_anchors['ref_quat']),
        'ref_obj_pos': np.stack(next_anchors['ref_obj_pos'])
    }

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # 1. Load Config & Model
    config_path = "config/config.yaml"
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config(config_path)
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    
    dataset = FlexibleWindowDataset(
        data_root=data_path, config=data_cfg, norm_path=norm_path, calculate_stats=False
    )
    
    diffuser = RobotDiffuser(
        model_config=model_cfg, data_config=data_cfg, training_config=training_cfg,
        noise_scheduler_config=noise_cfg, mode="infer", device=device
    )
    diffuser.loadWeights(args.epoch)
    
    # 2. Prepare Task Params (Grid or Dataset)
    if args.use_dataset_tasks:
        print(f"Sampling {args.num_dataset_tasks} tasks from dataset...")
        
        # Find start indices (where time=0)
        start_indices = [i for i, (f, b, t) in enumerate(dataset.indices) if t == 0]
        
        if len(start_indices) < args.num_dataset_tasks:
            print(f"Warning: Only found {len(start_indices)} start points. Using all.")
            selected_indices = start_indices
        else:
            # Random sample
            selected_indices = np.random.choice(start_indices, args.num_dataset_tasks, replace=False)
        
        num_samples = len(selected_indices)
        
        # Extract tasks
        tasks_list = []
        raw_tasks_list = []
        initial_states_list = []
        anchors_ref_pos_list = []
        anchors_ref_quat_list = []
        anchors_ref_obj_pos_list = []
        # Need to iterate to extract
        for idx in tqdm(selected_indices, desc="Extracting Tasks"):
            # dataset[idx] -> _, _, task (norm), _
            _, curr_state, task_norm, anchor = dataset[idx] # task_norm is tensor
            
            tasks_list.append(task_norm)
            initial_states_list.append(curr_state)
            anchors_ref_pos_list.append(anchor['ref_pos'])
            anchors_ref_quat_list.append(anchor['ref_quat'])
            anchors_ref_obj_pos_list.append(anchor['ref_obj_pos'])
            # Let's try to access stats directly
            if "min_task_params" in dataset.stats:
                min_v = dataset.stats["min_task_params"]
                max_v = dataset.stats["max_task_params"]
                # Denorm: x = (norm + 1)/2 * (max - min) + min
                raw = ((task_norm + 1) / 2) * (max_v - min_v) + min_v
                raw_tasks_list.append(raw.numpy())
            else:
                # Fallback if unnormalized (shouldn't happen with valid cfg)
                raw_tasks_list.append(task_norm.numpy())

        norm_task_params = torch.stack(tasks_list).to(device)
        task_params_raw = np.stack(raw_tasks_list)
        
        # Stack Initial Conditions
        all_initial_states = torch.stack(initial_states_list) # (N, H, F)
        all_anchors = {
             'ref_pos': np.stack(anchors_ref_pos_list),
             'ref_quat': np.stack(anchors_ref_quat_list),
             'ref_obj_pos': np.stack(anchors_ref_obj_pos_list)
        }

    else:
        # Grid Mode
        x_vals = np.linspace(args.x_min, args.x_max, args.grid_size)
        y_vals = np.linspace(args.y_min, args.y_max, args.grid_size)
        
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()
        
        num_samples = len(flat_x)
        print(f"Generating {num_samples} trajectories (Grid {args.grid_size}x{args.grid_size})...")
        
        # Create (num_samples, 2) tensor
        task_params_raw = np.stack([flat_x, flat_y], axis=1) # (N, 2)
        task_params_tens = torch.tensor(task_params_raw, dtype=torch.float32)
        
        # Normalize Task Params
        norm_task_params = dataset._normalize("task_params", task_params_tens).to(device)
        
        # Load single initial condition and tile
        print(f"Loading initial condition from sample {args.sample_idx}...")
        _, curr_state, _, anchor = dataset[args.sample_idx]
        
        all_initial_states = curr_state.unsqueeze(0).repeat(num_samples, 1, 1)
        all_anchors = {
            'ref_pos': np.tile(anchor['ref_pos'][None], (num_samples, 1)),
            'ref_quat': np.tile(anchor['ref_quat'][None], (num_samples, 1)),
            'ref_obj_pos': np.tile(anchor['ref_obj_pos'][None], (num_samples, 1))
        }
    
    # 3. Batch Inference Loop
    all_obj_traj_w = []
    
    # 4. Batch Inference Loop
    for b_start in range(0, num_samples, args.batch_size):
        b_end = min(b_start + args.batch_size, num_samples)
        curr_bs = b_end - b_start
        print(f"Processing Batch {b_start}-{b_end} ({curr_bs} samples)...")
        
        # Batch Data
        batch_task_params = norm_task_params[b_start:b_end]
        
        # Use batched initial states
        batch_states = all_initial_states[b_start:b_end].to(device)
        curr_state_tens = batch_states
        
        current_anchors = {
            'ref_pos': all_anchors['ref_pos'][b_start:b_end],
            'ref_quat': all_anchors['ref_quat'][b_start:b_end],
            'ref_obj_pos': all_anchors['ref_obj_pos'][b_start:b_end]
        }
        
        generated_segments = []
        
        # Stitching Loop
        for step in range(args.stitch_steps):
            sample = diffuser.getSample(
                num_trajectories=curr_bs,
                state_cond=curr_state_tens,
                goal_cond=batch_task_params, # Pass the varying task params
                deterministic=True,
                cfg_w=args.cfg_w,
                guidance_wt=args.guidance_wt
            )
            
            # Denorm
            denorm = dataset.denormalize_global(sample)
            future_traj = denorm.cpu().numpy()
            
            # Reconstruct
            anchor_arr = np.concatenate([current_anchors['ref_pos'], current_anchors['ref_quat'], current_anchors['ref_obj_pos']], axis=-1)
            robot_world, obj_world, _, _ = reconstruct_sbto_trajectory(
                base_pose_world=anchor_arr,
                future_traj=future_traj,
                inpaint=model_cfg["inpaint"],
            )
            
            # Store
            segment = np.concatenate([robot_world[..., :36], obj_world[..., :7]], axis=-1)
            generated_segments.append(segment)
            
            if step < args.stitch_steps - 1:
                hist_len = dataset.history_size
                r_hist = robot_world[:, -hist_len:, :]
                o_hist = obj_world[:, -hist_len:, :]
                
                curr_state_tens, next_anch = update_condition(dataset, r_hist, o_hist)
                curr_state_tens = curr_state_tens.to(device)
                current_anchors = next_anch

        # Concatenate segments for this batch -> (B, Total_T, 43)
        full_batch_traj = np.concatenate(generated_segments, axis=1)
        all_obj_traj_w.append(full_batch_traj[:, :, 36:43])

    # 5. Plotting
    obj_traj_w = np.concatenate(all_obj_traj_w, axis=0) # (N, Total_T, 7)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Start (Moved inside loop as it varies)
    # start_pos = obj_traj_w[0, 0, :2] 
    # plt.scatter(start_pos[0], start_pos[1], c='black', marker='*', s=200, label='Start')
    
    # Colormap
    colors = cm.rainbow(np.linspace(0, 1, num_samples))
    
    for i in range(num_samples):
        traj = obj_traj_w[i, :, :2]
        # Calculate intended target
        # For Grid: target = start + task_param
        # For Dataset: target = start + task_param (since task param is delta)
        
        # Start pos is first frame of trajectory
        start_pos = traj[0]
        
        target_pos = start_pos + task_params_raw[i]
        
        # Plot Trajectory
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.5)
        
        # Plot Target
        plt.scatter(target_pos[0], target_pos[1], color=colors[i], marker='x')
        
        # Plot Start (Variable now)
        plt.scatter(start_pos[0], start_pos[1], color=colors[i], marker='*', s=50)

    plt.title(f"Task Param Sweep\n(Mode: {'Dataset' if args.use_dataset_tasks else 'Grid'})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    
    # Adaptive Limits Centered at 0,0
    all_trajs = obj_traj_w[:, :, :2].reshape(-1, 2)
    # Re-calculate targets for limit calculation
    starts = obj_traj_w[:, 0, :2]
    targets = starts + task_params_raw
    
    all_points = np.concatenate([all_trajs, targets], axis=0)
    max_val = np.max(np.abs(all_points)) * 1.1 # 10% buffer
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    
    plt.grid(True)
    # plt.legend()
    
    plt.savefig(args.save_path)
    print(f"Saved plot to {args.save_path}")

if __name__ == "__main__":
    main()

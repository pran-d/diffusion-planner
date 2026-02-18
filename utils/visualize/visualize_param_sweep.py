
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config.configure import load_config, get_data_path, get_norm_path
from models.model import RobotDiffuser
from datasets.flexible_dataset import FlexibleWindowDataset, yaw_to_rot_matrix, yaw_from_quat
from utils.math.sbto_utils import reconstruct_sbto_trajectory

def compute_task_params(current_robot_state, current_obj_state, desired_obj_pos):
    """
    Computes the task parameters (local object displacement) for the diffusion model.

    Args:
        current_robot_state (np.ndarray): Shape (7,) or (3,) [x, y, z, qx, qy, qz, qw]
                                          representing the robot base pose.
        current_obj_state (np.ndarray): Shape (3,) or (7,) [x, y, z, ...]
                                        representing the current object position.
        desired_obj_pos (np.ndarray): Shape (3,) [x, y, z] 
                                      representing the GOAL object position in world frame.

    Returns:
        np.ndarray: Shape (2,) [delta_x_local, delta_y_local] normalized if needed.
    """
    # 1. Extract Positions
    curr_obj_pos = current_obj_state[:3]  # We only care about X, Y for displacement
    goal_obj_pos = desired_obj_pos[:3]
    
    # 2. Calculate World Displacement
    world_delta = goal_obj_pos - curr_obj_pos
    
    # 3. Extract Robot Yaw
    if len(current_robot_state) >= 7:
        quat = current_robot_state[3:7]
    else:
        quat = current_robot_state
    
    # Only use Yaw for rotation
    # If quat is (4,), use yaw_from_quat -> angle -> matrix
    y = yaw_from_quat(quat)
    if isinstance(y, torch.Tensor):
        y = y.numpy()
        
    R_ref_inv = yaw_to_rot_matrix(-y) # (2,2) or (3,3) depending on implementation?
    
    # 4. Rotate into Robot Frame (Global -> Local)
    # R_ref_inv is 3x3. world_delta is 3.
    local_delta = R_ref_inv @ world_delta[..., None]
    
    return local_delta[..., :3, 0] 

def update_condition(dataset, robot_world_history, obj_world_history, final_obj_pos=None):
    """
    Update condition for next autoregressive step.
    Supports batch dimension.
    """
    B, H, _ = robot_world_history.shape
    next_states = []
    next_anchors = {'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': [], 'final_obj_pos': []}
    
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

        if final_obj_pos is not None:
             new_anch['final_obj_pos'] = final_obj_pos[b]
             feats['task_params'] = compute_task_params(
                 current_robot_state=new_anch['ref_quat'], 
                 current_obj_state=new_anch['ref_obj_pos'],
                 desired_obj_pos=final_obj_pos[b]
             )
        
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
        next_anchors['ref_pos'].append(new_anch['ref_pos'])
        next_anchors['ref_quat'].append(new_anch['ref_quat'])
        next_anchors['ref_obj_pos'].append(new_anch['ref_obj_pos'])
        if final_obj_pos is not None:
             next_anchors['final_obj_pos'].append(new_anch['final_obj_pos'])
        
    out_anchors = {
        'ref_pos': np.stack(next_anchors['ref_pos']), 
        'ref_quat': np.stack(next_anchors['ref_quat']),
        'ref_obj_pos': np.stack(next_anchors['ref_obj_pos'])
    }
    if final_obj_pos is not None:
         out_anchors['final_obj_pos'] = np.stack(next_anchors['final_obj_pos'])

    return torch.stack(next_states), out_anchors

def parse_args():
    parser = argparse.ArgumentParser("Visualize Task Param Sweep & Evaluate")
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--stitch_steps", type=int, default=20)
    parser.add_argument("--sample_idx", type=int, default=0, help="Initial condition index from dataset (Grid Mode)")
    parser.add_argument("--cfg_w", type=float, default=1.0, help="Classifier-Free Guidance weight")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="task_viz.png")
    parser.add_argument("--eval_save_path", type=str, default="task_eval.png", help="Path for evaluation scatter plot")
    parser.add_argument("--action_horizon", type=int, default=None, help="Number of future steps to visualize/control (for dataset visualization)")
    
    # Sweep Config
    parser.add_argument("--x_min", type=float, default=-0.5)
    parser.add_argument("--x_max", type=float, default=0.5)
    parser.add_argument("--y_min", type=float, default=-0.5)
    parser.add_argument("--y_max", type=float, default=0.5)
    parser.add_argument("--grid_size", type=int, default=5, help="Number of points per axis (total grid_size^2)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    
    # Dataset Task Params
    parser.add_argument("--use_dataset_tasks", action="store_true", help="Use task parameters sampled from the dataset instead of a grid")
    parser.add_argument("--num_dataset_tasks", type=int, default=25, help="Number of tasks to sample from dataset")
    parser.add_argument("--seed", type=int, default=42)

    # Guidance
    parser.add_argument("--guidance_wt", type=float, default=0.0)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # 1. Load Config & Model
    config_path = "config/config.yaml"
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config(config_path)
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)

    if args.action_horizon is not None:
        args.stitch_steps *= (data_cfg['num_timesteps'] // args.action_horizon) + 1
    
    print("Loading dataset...")
    dataset = FlexibleWindowDataset(
        data_root=data_path, config=data_cfg, norm_path=norm_path, calculate_stats=False
    )
    
    diffuser = RobotDiffuser(
        model_config=model_cfg, data_config=data_cfg, training_config=training_cfg,
        noise_scheduler_config=noise_cfg, mode="infer", device=device
    )
    diffuser.loadWeights(args.epoch)
    
    # Build index map for faster access later
    index_map = {idx_tuple: i for i, idx_tuple in enumerate(dataset.indices)}
    
    tasks_list = []
    task_params_raw = [] # Ground Truth global displacements (World Frame) or Grid Targets
    initial_states_list = []
    
    # Comparison Lists (Only populated in Dataset Mode)
    gt_params_list = [] 
    
    anchors_list = {
        'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': [], 'final_obj_pos': []
    }

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
        
        # Need to iterate to extract
        for idx in tqdm(selected_indices, desc="Extracting Tasks"):
            # dataset[idx] -> _, _, task (norm), _
            _, curr_state, task_norm, anchor = dataset[idx] # task_norm is tensor
            
            initial_states_list.append(curr_state)
            anchors_list['ref_pos'].append(anchor['ref_pos'])
            anchors_list['ref_quat'].append(anchor['ref_quat'])
            anchors_list['ref_obj_pos'].append(anchor['ref_obj_pos'])
            
            # Extract Global Displacement GT
            file_idx, batch_idx, _ = dataset.indices[idx] 
            try:
                # Load full file once
                traj_data = dataset._get_single_traj(file_idx, batch_idx)
                if 'obj' not in traj_data:
                     # Fallback
                     print(f"Warning: 'obj' key missing. Using anchor.")
                     final_pos = anchor['ref_obj_pos']
                else:
                     obj_traj = traj_data['obj'] # (T, 7)
                     final_pos = obj_traj[-1, :3]
            except Exception as e:
                print(f"Error calculating global displacement: {e}")
                final_pos = anchor['ref_obj_pos']

            anchors_list['final_obj_pos'].append(final_pos)
            
            # Store GT Task Param (Delta X, Y)
            start_pos = anchor['ref_obj_pos'][:3]
            delta = (final_pos - start_pos)[:2]
            gt_params_list.append(delta)
            
            # Initial Task Param for Loop (will be overwritten by loop, but needed for shape)
            tasks_list.append(task_norm) 

        # Convert lists to tensors/arrays
        all_initial_states = torch.stack(initial_states_list)
        all_anchors = {k: np.stack(v) for k, v in anchors_list.items()}
        
        # Initial dummy task params (will be recomputed in loop)
        norm_task_params = torch.stack(tasks_list).to(device)
        
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
        grid_deltas_raw = np.stack([flat_x, flat_y], axis=1) # (N, 2) local frame deltas
        
        # Load single initial condition and tile
        print(f"Loading initial condition from sample {args.sample_idx}...")
        _, curr_state, _, anchor = dataset[args.sample_idx]

        # Calculate final_obj_pos for grid points
        # Grid Deltas are in ROBOT FRAME at time 0.
        # We need to transform them to WORLD FRAME to get final_obj_pos.
        
        grid_deltas_3d = np.concatenate([grid_deltas_raw, np.zeros((num_samples, 1))], axis=1) # (N, 3)

        # Robot Rotation
        q_vec = torch.from_numpy(anchor['ref_quat'])
        rot_mat = yaw_to_rot_matrix(yaw_from_quat(q_vec))
        if isinstance(rot_mat, torch.Tensor):
            rot_mat = rot_mat.numpy()

        # Rotate: global = R @ local
        world_deltas = (rot_mat @ grid_deltas_3d.T).T 
        
        start_obj_pos = anchor['ref_obj_pos']
        final_obj_positions = start_obj_pos[None, :] + world_deltas
        
        # Tile states
        all_initial_states = curr_state.unsqueeze(0).repeat(num_samples, 1, 1)
        all_anchors = {
            'ref_pos': np.tile(anchor['ref_pos'][None], (num_samples, 1)),
            'ref_quat': np.tile(anchor['ref_quat'][None], (num_samples, 1)),
            'ref_obj_pos': np.tile(anchor['ref_obj_pos'][None], (num_samples, 1)),
            'final_obj_pos': final_obj_positions
        }
        
        # Prepare Tasks (Normalize the initial grid tasks for first step)
        # However, loop will recompute. We just need valid shape.
        tasks_tens = torch.tensor(grid_deltas_raw, dtype=torch.float32)
        norm_task_params = dataset._normalize("task_params", tasks_tens).to(device)


    # ------------------------------------------------------------------
    # 2. Batch Inference Loop
    # ------------------------------------------------------------------
    all_obj_traj_w = []
    all_gen_params = []
    
    for b_start in range(0, num_samples, args.batch_size):
        b_end = min(b_start + args.batch_size, num_samples)
        curr_bs = b_end - b_start
        print(f"Processing Batch {b_start}-{b_end} / {num_samples}...")
        
        # Batch Data
        curr_state_tens = all_initial_states[b_start:b_end].to(device)
        # Initial Task Params (Dummy, will be updated)
        gt_task_tens = norm_task_params[b_start:b_end].to(device)
        
        current_anchors = {
            k: v[b_start:b_end] for k, v in all_anchors.items()
        }
        
        generated_segments = []
        
        # Autoregressive Loop
        for step in range(args.stitch_steps):
            
            # --------------------------------------------------------
            # Dynamic Task Re-Targeting (Closed Loop Control)
            # --------------------------------------------------------
            # Recompute task params based on current robot state relative to FIXED final_obj_pos
            
            # Using loop over batch for clarity (optimization: vectorize compute_task_params)
            new_task_list = []
            for bi in range(curr_bs):
                c_quat = current_anchors['ref_quat'][bi]
                c_obj = current_anchors['ref_obj_pos'][bi]
                c_goal = current_anchors['final_obj_pos'][bi]
                
                tp = compute_task_params(c_quat, c_obj, c_goal) # (3,)
                new_task_list.append(tp)
            
            new_task_arr = np.stack(new_task_list)
            new_task_tens = torch.from_numpy(new_task_arr).float()
            # Normalize
            gt_task_tens = dataset._normalize("task_params", new_task_tens).to(device)

            # --------------------------------------------------------
            # Inference
            # --------------------------------------------------------
            sample = diffuser.getSample(
                num_trajectories=curr_bs,
                state_cond=curr_state_tens,
                goal_cond=gt_task_tens,
                deterministic=True,
                cfg_w=args.cfg_w,
                guidance_wt=args.guidance_wt,
            )
            
            # Denormalize
            denorm = dataset.denormalize_global(sample) # (B, T, C)
            future_traj = denorm.cpu().numpy()
            
            # Reconstruct (Local -> World)
            # Flatten anchors for broadcasting if needed, but reconstruct handles (B, D)
            # anchors: (B, D)
            anchor_arr = np.concatenate([
                current_anchors['ref_pos'], 
                current_anchors['ref_quat'], 
                current_anchors['ref_obj_pos']
            ], axis=-1)
            
            try:
                res = reconstruct_sbto_trajectory(
                    base_pose_world=anchor_arr,
                    future_traj=future_traj,
                    inpaint=diffuser.model_cfg.get("inpaint", False)
                )
                robot_world, obj_world = res[0], res[1]
            except:
                robot_world, obj_world, _, _ = reconstruct_sbto_trajectory(
                    base_pose_world=anchor_arr,
                    future_traj=future_traj,
                    inpaint=diffuser.model_cfg.get("inpaint", False)
                )
            
            if args.action_horizon is not None:
                robot_world = robot_world[:, :args.action_horizon, :]
                obj_world = obj_world[:, :args.action_horizon, :]

            # Store Segment
            segment = np.concatenate([robot_world[..., :36], obj_world[..., :7]], axis=-1)
            generated_segments.append(segment)

            # Desired Displacement (From Ground Truth Full Trajectory)                    
            err = np.linalg.norm(current_anchors["final_obj_pos"] - segment[:, -1, 36:39])
            if err < 0.25:
                print(f"Segment {step+1} successfully reached the goal (Error: {err:.4f}).")
                break
            
            # Update Condition
            if step < args.stitch_steps - 1:
                hist_len = dataset.history_size
                r_hist = robot_world[:, -hist_len:, :]
                o_hist = obj_world[:, -hist_len:, :]
                
                curr_state_tens, next_anch = update_condition(
                    dataset, r_hist, o_hist, 
                    final_obj_pos=current_anchors['final_obj_pos']
                )
                curr_state_tens = curr_state_tens.to(device)
                current_anchors = next_anch

        # End Step Loop
        full_batch_traj = np.concatenate(generated_segments, axis=1) # (B, Total_T, 43)
        all_obj_traj_w.append(full_batch_traj[:, :, 36:43])
        
        # Calculate Generated Displacement (Global)
        # Start: traj[0, :2], End: traj[-1, :2]
        # Note: Obj traj is indices 36:43.
        for bi in range(curr_bs):
            traj = full_batch_traj[bi, :, 36:43]
            start = traj[0, :3]
            end = traj[-1, :3]
            all_gen_params.append(end - start)

    # ------------------------------------------------------------------
    # 3. Visualization & Analysis
    # ------------------------------------------------------------------
    obj_traj_w = np.concatenate(all_obj_traj_w, axis=0) # (N, Total_T, 7)
    gen_params = np.stack(all_gen_params)
    
    if args.use_dataset_tasks:
        gt_params = np.stack(gt_params_list)
        
        # Calc Stats
        diff = gen_params[..., :2] - gt_params
        errors = np.linalg.norm(diff, axis=1)
        
        print("\n=== Dataset Evaluation Stats ===")
        print(f"Mean Error: {np.mean(errors):.4f}")
        print(f"Max Error: {np.max(errors):.4f}")
        print(f"Std Error: {np.std(errors):.4f}")
        
        # Scatter Plot (GT vs Gen)
        plt.figure(figsize=(8, 8))
        plt.scatter(gt_params[:, 0], gt_params[:, 1], c='blue', label='Desired (GT)', alpha=0.6)
        plt.scatter(gen_params[:, 0], gen_params[:, 1], c='red', label='Generated', alpha=0.6)
        
        # Draw lines
        for i in range(len(gt_params)):
             plt.plot([gt_params[i, 0], gen_params[i, 0]], 
                      [gt_params[i, 1], gen_params[i, 1]], 'gray', alpha=0.3)
             
        plt.xlabel("Delta X")
        plt.ylabel("Delta Y")
        plt.title(f"Task Execution Accuracy (N={num_samples})")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
                
        # Center plot around 0,0 and fix axes
        all_vals = np.concatenate([gt_params, gen_params[..., :2]], axis=0)
        max_val = np.max(np.abs(all_vals)) * 1.1 # 10% padding
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.axvline(0, color='k', linewidth=0.5)
        
        plt.savefig(args.eval_save_path)
        print(f"Scatter plot saved to {args.eval_save_path}")
        
        # Trajectory Plot
        # For dataset tasks, they all start at different places, so a single shared plot is messy.
        # However, we can plot them relative to their start? 
        # Or just plot the trajectories in world frame?
        # If N is small (25), World Frame is fine.
        
        if num_samples <= 50:
            plt.figure(figsize=(10, 8))
            for i in range(num_samples):
                # Plot GT Path if we have it? (We only extracted end point)
                # Plot Gen Path
                path = obj_traj_w[i]
                plt.plot(path[:, 0], path[:, 1], alpha=0.5)
                # Mark Start/End
                plt.scatter(path[0, 0], path[0, 1], marker='o', s=20, c='g')
                plt.scatter(path[-1, 0], path[-1, 1], marker='x', s=20, c='r')
                # Mark GT Goal
                goal = all_anchors['final_obj_pos'][i]
                plt.scatter(goal[0], goal[1], marker='*', s=50, c='gold', edgecolors='k')
            
            # Adaptive Limits Centered at 0,0
            all_trajs = obj_traj_w[:, :, :2].reshape(-1, 2)
            # Re-calculate targets for limit calculation
            starts = obj_traj_w[:, 0, :2]
            targets = all_anchors['final_obj_pos'][:, :2]
            all_points = np.concatenate([all_trajs, targets], axis=0)
            max_val = np.max(np.abs(all_points)) * 1.1 # 10% buffer
            
            plt.xlim(-max_val, max_val)
            plt.ylim(-max_val, max_val)

            plt.title("Generated Trajectories (World Frame)")
            plt.axis('equal')
            plt.grid(True)
            plt.savefig(args.save_path)
            print(f"Trajectory plot saved to {args.save_path}")

    else:
        # Grid Mode Plotting
        # We initialized a grid.
        # Plot Quiver: (Grid Center) -> (Generated Displacement)
        # Wait, Grid Params were INITIAL Local displacements.
        # But robot moves.
        # We can plot: Grid Target (Start + Grid Delta) vs Actual End.
        
        # Reconstruct "Desired" Grid Targets in World Frame
        # final_obj_positions from setup
        
        desired_ends = all_anchors['final_obj_pos'][:, :2] # (N, 2)
        actual_ends = obj_traj_w[:, -1, :2] # (N, 2)
        starts = all_anchors['ref_obj_pos'][:, :2] # (N, 2) (All same for grid usually, unless start changes)
        # Actually in Grid Mode, we tiled the SAME start sample. So all starts are identical.
        
        # Calculate Vectors relative to start
        # Desired Vector
        d_vec = desired_ends - starts
        # Actual Vector
        a_vec = actual_ends - starts
        
        plt.figure(figsize=(10, 8))
        
        # Plot Desired Grid
        plt.scatter(d_vec[:, 0], d_vec[:, 1], c='blue', marker='x', label='Target Grid')
        
        # Plot Actual Endpoints
        # Color by error?
        errors = np.linalg.norm(desired_ends - actual_ends, axis=1)
        
        plt.scatter(a_vec[:, 0], a_vec[:, 1], c=errors, cmap='viridis', label='Actual Reached')
        plt.colorbar(label='Error (L2)')
        
        # Quiver for error direction? Or just trajectories?
        # Trajectories might be too dense if 100 points.
        # Let's plot arrows from Target to Actual (Error Vectors)
        # plt.quiver(d_vec[:, 0], d_vec[:, 1], 
        #            a_vec[:, 0]-d_vec[:, 0], a_vec[:, 1]-d_vec[:, 1],
        #            angles='xy', scale_units='xy', scale=1, alpha=0.3, width=0.003)
        
        plt.title(f"Grid Sweep Execution ({args.grid_size}x{args.grid_size})")
        plt.xlabel("Delta X (m)")
        plt.ylabel("Delta Y (m)")
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(args.save_path)
        print(f"Grid sweep plot saved to {args.save_path}")

if __name__ == "__main__":
    main()

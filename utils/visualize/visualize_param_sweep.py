
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
from utils.math.sbto_utils import reconstruct_sbto_trajectory, compute_task_params

def update_condition(dataset, robot_world_history, obj_world_history, final_obj_pos=None):
    """
    Update condition for next autoregressive step.
    Extracts last history window and re-computes relative SBTO features.
    """
    B, H, _ = robot_world_history.shape
    next_states = []
    next_anchors = {'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': [], 'final_obj_pos': []}

    for b in range(B):
        # Extract robot (base+joints) and object
        r_slice = robot_world_history[b] # (H, 36)
        o_slice = obj_world_history[b]   # (H, 7)
        
        raw_chunk = {
            'base': r_slice[:, :7],       
            'joints': r_slice[:, 7:36],
            'obj': o_slice[:, :7]
        }
        
        # Compute SBTO feats relative to new start (index 0 of chunk)
        feats, new_anch = dataset._compute_transform(raw_chunk, t_start=0)

        if final_obj_pos is not None:
            new_anch['final_obj_pos'] = final_obj_pos[b]
        
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
                
                # Filter for observation features
                if part_end > obs_start_idx:
                    local_start = max(0, obs_start_idx - cumulative_dim)
                    current_parts.append(part[:H, local_start:])
                
                cumulative_dim += part_dim
        
        c_state = torch.cat(current_parts, dim=-1)
        next_states.append(c_state)
        next_anchors['ref_pos'].append(new_anch['ref_pos'])
        next_anchors['ref_quat'].append(new_anch['ref_quat'])
        next_anchors['ref_obj_pos'].append(new_anch['ref_obj_pos'])
        next_anchors['final_obj_pos'].append(new_anch['final_obj_pos'])
        
    next_state_tens = torch.stack(next_states)

    batched_anchor = {
        'ref_pos': np.stack(next_anchors['ref_pos']),
        'ref_quat': np.stack(next_anchors['ref_quat']),
        'ref_obj_pos': np.stack(next_anchors['ref_obj_pos']),
        'final_obj_pos': np.stack(next_anchors['final_obj_pos']),
    }
    return next_state_tens, batched_anchor

def parse_args():
    parser = argparse.ArgumentParser("Visualize Task Param Sweep & Evaluate")
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--stitch_steps", type=int, default=None, help="Number of autoregressive segments to stitch together (default: enough to cover dataset horizon)")
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
    parser.add_argument("--num_dataset_tasks", type=int, default=0, help="Number of tasks to sample from dataset (In-Distribution)")
    parser.add_argument("--num_ood_tasks", type=int, default=0, help="Number of out-of-distribution tasks to generate (Random/Grid)")
    parser.add_argument("--seed", type=int, default=None)

    # Guidance
    parser.add_argument("--guidance_wt", type=float, default=0.0)
    
    return parser.parse_args()

def run_evaluation_batch(
    args, diffuser, dataset, device, 
    initial_states, norm_task_params, anchors_dict, 
    use_state_cond=True, desc="Eval"
):
    """
    Runs inference for a batch of tasks and returns trajectories and displacements.
    """
    num_samples = len(initial_states) if initial_states is not None else len(norm_task_params)
    
    all_obj_traj_w = []
    all_gen_params = []
    
    # Track current anchors which update over time
    current_anchors = {
        k: v.copy() for k, v in anchors_dict.items()
    }
    
    # We maintain current state tensor if using state cond
    curr_state_tens = initial_states.to(device) if initial_states is not None else None
    task_tens = norm_task_params.to(device)

    if args.stitch_steps is None:
        args.stitch_steps = dataset.traj_lengths[args.traj_idx] // diffuser.window_size
        print(f"Auto-setting stitch_steps to {args.stitch_steps} based on dataset length.")

    if args.action_horizon is not None:
        args.stitch_steps *= (diffuser.window_size // args.action_horizon)
        print(f"Adjusting stitch_steps to {args.stitch_steps} based on action horizon")

    # We maintain ground truth (or target) task params
    # Initial tasks (will be updated in loop dynamically)
    gt_task_tens = norm_task_params.to(device)

    generated_segments = []

    for step in range(args.stitch_steps):
        # --------------------------------------------------------
        # Dynamic Task Re-Targeting (Closed Loop Control)
        # --------------------------------------------------------
        new_task_list = []
        for bi in range(num_samples):
            c_quat = current_anchors['ref_quat'][bi]
            c_obj = current_anchors['ref_obj_pos'][bi]
            c_goal = current_anchors['final_obj_pos'][bi]
            
            tp = compute_task_params(
                c_quat, c_obj, c_goal,
                normalize_goal_vec=dataset.normalize_goal_vec,
                num_task_params=dataset.num_task_params 
            ) # (2,) or (3,) depending on impl
            new_task_list.append(tp)
        
        new_task_arr = np.stack(new_task_list)
        new_task_tens = torch.from_numpy(new_task_arr).float()
        
        # Normalize
        gt_task_tens = dataset._normalize("task_params", new_task_tens).to(device)
        
        # --------------------------------------------------------
        # Inference
        # --------------------------------------------------------
        # Wait, if we stitching, we must process all samples for step K before step K+1.
        
        # So inside this step loop, we loop over batches.
        
        step_segments = []
        
        for b_start in range(0, num_samples, args.batch_size):
            b_end = min(b_start + args.batch_size, num_samples)
            bs = b_end - b_start
            
            batch_state = curr_state_tens[b_start:b_end] if curr_state_tens is not None else None
            batch_task = gt_task_tens[b_start:b_end]
            
            sample = diffuser.getSample(
                num_trajectories=bs,
                state_cond=batch_state,
                goal_cond=batch_task,
                deterministic=True,
                cfg_w=args.cfg_w,
                guidance_wt=args.guidance_wt,
                no_state_cond=not use_state_cond,
            )
            
            # Denormalize
            denorm = dataset.denormalize_global(sample)
            future_traj = denorm.cpu().numpy()
            
            # Reconstruct
            b_anchors = np.concatenate([
                current_anchors['ref_pos'][b_start:b_end], 
                current_anchors['ref_quat'][b_start:b_end], 
                current_anchors['ref_obj_pos'][b_start:b_end],
                current_anchors['final_obj_pos'][b_start:b_end]
            ], axis=-1)
            
            try:
                res = reconstruct_sbto_trajectory(
                    base_pose_world=b_anchors,
                    future_traj=future_traj,
                    inpaint=diffuser.model_cfg.get("inpaint", False)
                )
                robot_world, obj_world = res[0], res[1]
            except:
                robot_world, obj_world, _, _ = reconstruct_sbto_trajectory(
                    base_pose_world=b_anchors,
                    future_traj=future_traj,
                    inpaint=diffuser.model_cfg.get("inpaint", False)
                )

            if args.action_horizon is not None:
                robot_world = robot_world[:, :args.action_horizon, :]
                obj_world = obj_world[:, :args.action_horizon, :]

            # Store (B, T, D)
            segment = np.concatenate([robot_world[..., :36], obj_world[..., :7]], axis=-1)
            step_segments.append(segment)
            
        # Concatenate batches for this step
        full_step_segment = np.concatenate(step_segments, axis=0) # (N, T, D)
        generated_segments.append(full_step_segment)
        
        # Update Condition for next step
        if step < args.stitch_steps - 1:
            # We need to update curr_state_tens and current_anchors
            # Robot: 0:36, Obj: 36:43
            r_full = full_step_segment[..., :36]
            o_full = full_step_segment[..., 36:43]
            
            hist_len = dataset.history_size
            r_hist = r_full[:, -hist_len:, :]
            o_hist = o_full[:, -hist_len:, :]
            
            # Update
            # Note: update_condition handles batch correctly
            next_state_tens, next_anch = update_condition(
                dataset, r_hist, o_hist, 
                final_obj_pos=current_anchors['final_obj_pos']
            )
            
            if curr_state_tens is not None:
                curr_state_tens = next_state_tens.to(device)
            # Update anchors
            current_anchors = next_anch

    # End Step Loop
    full_traj = np.concatenate(generated_segments, axis=1) # (N, Total_T, 43)
    
    # Calculate Displacements
    final_displacements = []
    start_displacements = [] # Should be 0 usually but tracking for correctness
    
    # Extract Object Traj
    obj_traj_w = full_traj[..., 36:43]
    
    for i in range(num_samples):
        start_pt = obj_traj_w[i, 0, :3]
        end_pt = obj_traj_w[i, -1, :3]
        final_displacements.append(end_pt - start_pt)

    return obj_traj_w, np.stack(final_displacements)

def plot_trajectories(obj_traj_w, anchors, save_path, title):
    """
    Plots the trajectories of the object in world coordinates.
    """
    num_samples = len(obj_traj_w)
    plt.figure(figsize=(10, 8))
    
    # Plot limit calculation
    all_trajs = obj_traj_w[:, :, :2].reshape(-1, 2)
    targets = anchors['final_obj_pos'][:, :2]
    all_points = np.concatenate([all_trajs, targets], axis=0)
    max_val = np.max(np.abs(all_points)) * 1.1 # 10% buffer
    
    for i in range(num_samples):
        # Plot Gen Path
        path = obj_traj_w[i]
        plt.plot(path[:, 0], path[:, 1], alpha=0.5)
        # Mark Start/End
        plt.scatter(path[0, 0], path[0, 1], marker='o', s=20, c='g')
        plt.scatter(path[-1, 0], path[-1, 1], marker='x', s=20, c='r')
        # Mark GT Goal
        goal = anchors['final_obj_pos'][i]
        plt.scatter(goal[0], goal[1], marker='*', s=50, c='gold', edgecolors='k')
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)

    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Trajectory plot saved to {save_path}")

def maximize_window(plot_title=""):
    mng = plt.get_current_fig_manager()
    try:
        mng.resize(*mng.window.maxsize())
    except:
        pass

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

    if args.stitch_steps is None:
        args.stitch_steps = 20
        if args.action_horizon is not None:
             args.stitch_steps = (data_cfg['num_timesteps'] // args.action_horizon) + 1
    elif args.action_horizon is not None:
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

    # --------------------------------------------------------
    # Dataset Tasks Evaluation (In-Distribution, With State Cond)
    # --------------------------------------------------------
    if args.num_dataset_tasks > 0:
        print(f"\nExample Dataset Tasks: {args.num_dataset_tasks} samples")
        
        start_indices = [i for i, (f, b, t) in enumerate(dataset.indices) if t == 0]
        if len(start_indices) < args.num_dataset_tasks:
            selected = start_indices
        else:
            selected = np.random.choice(start_indices, args.num_dataset_tasks, replace=False)
            
        initial_states = []
        anchors = {'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': [], 'final_obj_pos': []}
        gt_deltas = []
        
        for idx in tqdm(selected, desc="Prep Dataset Tasks"):
            _, curr_state, _, anchor = dataset[idx]
            
            # Find GT Final
            file_idx, batch_idx, _ = dataset.indices[idx]
            traj_data = dataset._get_single_traj(file_idx, batch_idx)
            if 'obj' in traj_data:
                final_pos = traj_data['obj'][-1, :3]
            else:
                final_pos = anchor['ref_obj_pos'][:3] # Fallback
            
            curr_state_tens = curr_state # (C)
            
            initial_states.append(curr_state_tens)
            anchors['ref_pos'].append(anchor['ref_pos'])
            anchors['ref_quat'].append(anchor['ref_quat'])
            anchors['ref_obj_pos'].append(anchor['ref_obj_pos'])
            anchors['final_obj_pos'].append(final_pos)
            
            gt_deltas.append((final_pos - anchor['ref_obj_pos'])[:data_cfg["num_task_params"]])

        # Prepare Batch
        initial_states = torch.stack(initial_states) # (N, C)
        anchors_arr = {k: np.stack(v) for k, v in anchors.items()}
        # Initial dummy task
        dummy_task = torch.zeros(args.num_dataset_tasks, data_cfg["num_task_params"])
        
        # Run Eval
        traj_w, gen_displacements = run_evaluation_batch(
            args, diffuser, dataset, device, 
            initial_states, dummy_task, anchors_arr, 
            use_state_cond=True, desc="Dataset Eval"
        )
        
        # Stats
        gt_deltas = np.array(gt_deltas)
        gen_deltas = gen_displacements[:, :data_cfg["num_task_params"]]
        errors = np.linalg.norm(gt_deltas - gen_deltas, axis=1)
        
        print(f"Dataset Tasks (State Cond ON): Mean Error: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}")
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(gt_deltas[:, 0], gt_deltas[:, 1], c='blue', label='GT Delta')
        plt.scatter(gen_deltas[:, 0], gen_deltas[:, 1], c='red', label='Gen Delta', alpha=0.7)
        for i in range(len(gt_deltas)):
            plt.plot([gt_deltas[i,0], gen_deltas[i,0]], [gt_deltas[i,1], gen_deltas[i,1]], 'gray', alpha=0.3)
        plt.title("In-Distribution Tasks (State Cond)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()

        max_val = np.max(np.concatenate([np.abs(gt_deltas[:, 0]), np.abs(gt_deltas[:, 1])])) * 1.1 # 10% buffer

        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
        
        plt.savefig("eval_dataset_tasks.png")

        # Plot Trajectories
        if args.num_dataset_tasks <= 50:
            plot_trajectories(traj_w, anchors_arr, "eval_dataset_trajs.png", "In-Distribution Trajectories")

    # --------------------------------------------------------
    # OOD Tasks Evaluation (Random Goals, State Cond OFF)
    # --------------------------------------------------------
    if args.num_ood_tasks > 0:
        print(f"\nOOD Tasks: {args.num_ood_tasks} samples (Random Goals, No State Cond)")
        
        # Use random start states from dataset as initial physical configuration
        # But we will nullify state history condition to simulate "No State Cond"
        start_indices = [i for i, (f, b, t) in enumerate(dataset.indices) if t == 0]
        # Allow reuse if we want many samples
        selected = np.random.choice(start_indices, args.num_ood_tasks, replace=True)
        
        initial_states = []
        anchors = {'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': [], 'final_obj_pos': []}
        
        # Generate Random Displacements (OOD)
        # Range: +/- 0.5 meters?
        random_deltas = (np.random.rand(args.num_ood_tasks, data_cfg["num_task_params"]) - 0.5) * 1.0 
        
        target_deltas = []

        for i, idx in enumerate(tqdm(selected, desc="Prep OOD Tasks")):
            _, curr_state, _, anchor = dataset[idx]
            
            # Construct Target
            start_pos = anchor['ref_obj_pos'][:data_cfg["num_task_params"]]
            # Convert random delta (generic XY) to target world pos
            # We treat random delta as WORLD frame delta for simplicity of target generation
            target_pos = start_pos.copy()
            target_pos[:data_cfg["num_task_params"]] += random_deltas[i]
            
            initial_states.append(curr_state)
            anchors['ref_pos'].append(anchor['ref_pos'])
            anchors['ref_quat'].append(anchor['ref_quat'])
            anchors['ref_obj_pos'].append(anchor['ref_obj_pos'])
            anchors['final_obj_pos'].append(target_pos)
            
            target_deltas.append(random_deltas[i])

        initial_states = torch.stack(initial_states)
        anchors_arr = {k: np.stack(v) for k, v in anchors.items()}
        dummy_task = torch.zeros(args.num_ood_tasks, data_cfg["num_task_params"])
        
        # Run Eval with use_state_cond=FALSE
        traj_w, gen_displacements = run_evaluation_batch(
            args, diffuser, dataset, device, 
            initial_states, dummy_task, anchors_arr, 
            use_state_cond=False, desc="OOD Eval"
        )
        
        target_deltas = np.array(target_deltas)
        gen_deltas = gen_displacements[:, :data_cfg["num_task_params"]]
        errors = np.linalg.norm(target_deltas - gen_deltas, axis=1)
        
        print(f"OOD Tasks (State Cond OFF): Mean Error: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}")
        
        plt.figure(figsize=(10, 10))
        plt.scatter(target_deltas[:, 0], target_deltas[:, 1], c='green', label='Target Delta')
        plt.scatter(gen_deltas[:, 0], gen_deltas[:, 1], c='orange', label='Gen Delta', alpha=0.7)
        for i in range(len(target_deltas)):
            plt.plot([target_deltas[i,0], gen_deltas[i,0]], [target_deltas[i,1], gen_deltas[i,1]], 'gray', alpha=0.3)
        plt.title("OOD Tasks (No State Cond)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.savefig("eval_ood_tasks.png")

        # Plot Trajectories
        if args.num_ood_tasks <= 50:
            plot_trajectories(traj_w, anchors_arr, "eval_ood_trajs.png", "OOD Trajectories (No State Cond)")

if __name__ == "__main__":
    main()

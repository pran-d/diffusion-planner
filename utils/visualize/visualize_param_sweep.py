
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
from utils.data.load_dataset import preload_dataset
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

    # Goal multiplier sweep
    parser.add_argument("--goal_multiplier", type=float, default=1.0,
                        help="Single goal multiplier (scales displacement magnitude, keeps direction)")
    parser.add_argument("--goal_multipliers", nargs="+", type=float, default=None,
                        help="Sweep over multiple goal multipliers (e.g. --goal_multipliers 0.5 1.0 1.5 2.0)")
    
    return parser.parse_args()

def run_evaluation_batch(
    args, diffuser, dataset, device, 
    initial_states, norm_task_params, anchors_dict, 
    use_state_cond=True, desc="Eval",
    stitch_steps_list=None
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

    if stitch_steps_list is not None:
        max_stitch_steps = max(stitch_steps_list)
        print(f"Using max stitch_steps {max_stitch_steps} for this batch.")
    elif args.stitch_steps is None:
        max_stitch_steps = dataset.traj_lengths[args.sample_idx] // diffuser.input_size
        print(f"Auto-setting stitch_steps to {max_stitch_steps} based on dataset length.")
    else:
        max_stitch_steps = args.stitch_steps

    if args.action_horizon is not None:
        max_stitch_steps *= (diffuser.input_size // args.action_horizon)
        print(f"Adjusting stitch_steps to {max_stitch_steps} based on action horizon")

    # We maintain ground truth (or target) task params
    # Initial tasks (will be updated in loop dynamically)
    gt_task_tens = norm_task_params.to(device)

    generated_segments = []

    for step in range(max_stitch_steps):
        # --------------------------------------------------------
        # Dynamic Task Re-Targeting (Closed Loop Control)
        # --------------------------------------------------------
        new_task_list = []
        for bi in range(num_samples):
            c_quat = current_anchors['ref_quat'][bi]
            c_obj = current_anchors['ref_obj_pos'][bi]
            c_goal = current_anchors['final_obj_pos'][bi]
            
            tp, _ = compute_task_params(
                c_quat, c_obj, c_goal,
                normalize_goal_vec=dataset.normalize_goal_vec,       
                num_task_params=dataset.num_task_params,
                max_goal_dist=dataset.max_obj_displacement,
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
        if step < max_stitch_steps - 1:
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
    data_buffer = preload_dataset(data_cfg, data_path)
    dataset = FlexibleWindowDataset(
        data_buffer=data_buffer, config=data_cfg, norm_path=norm_path, calculate_stats=False,
        training_cfg=training_cfg,
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
        stitch_steps_list = []
        
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
            # Apply goal multiplier: keep direction, scale magnitude
            if args.goal_multiplier != 1.0:
                delta = final_pos - anchor['ref_obj_pos']
                final_pos = anchor['ref_obj_pos'].copy()
                final_pos[:2] += args.goal_multiplier * delta[:2]

            anchors['final_obj_pos'].append(final_pos)
            
            gt_deltas.append((final_pos - anchor['ref_obj_pos'])[:data_cfg["num_task_params"]])
            
            # Calculate required stitch steps for this trajectory
            traj_len = traj_data['obj'].shape[0] if 'obj' in traj_data else traj_data['joints'].shape[0]
            window_size = diffuser.input_size
            required_steps = int(np.ceil((traj_len - 1) / (window_size - 1)))
            stitch_steps_list.append(required_steps)

        # Prepare Batch
        initial_states = torch.stack(initial_states) # (N, C)
        anchors_arr = {k: np.stack(v) for k, v in anchors.items()}
        # Initial dummy task
        dummy_task = torch.zeros(args.num_dataset_tasks, data_cfg["num_task_params"])
        
        # Run Eval
        traj_w, gen_displacements = run_evaluation_batch(
            args, diffuser, dataset, device, 
            initial_states, dummy_task, anchors_arr, 
            use_state_cond=True, desc="Dataset Eval",
            stitch_steps_list=stitch_steps_list
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
        stitch_steps_list_ood = []

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
            
            # Calculate required stitch steps for this trajectory
            file_idx, batch_idx, _ = dataset.indices[idx]
            traj_data = dataset._get_single_traj(file_idx, batch_idx)
            traj_len = traj_data['obj'].shape[0] if 'obj' in traj_data else traj_data['joints'].shape[0]
            window_size = diffuser.input_size
            required_steps = int(np.ceil((traj_len - 1) / (window_size - 1)))
            stitch_steps_list_ood.append(required_steps)

        initial_states = torch.stack(initial_states)
        anchors_arr = {k: np.stack(v) for k, v in anchors.items()}
        dummy_task = torch.zeros(args.num_ood_tasks, data_cfg["num_task_params"])
        
        # Run Eval with use_state_cond=FALSE
        traj_w, gen_displacements = run_evaluation_batch(
            args, diffuser, dataset, device, 
            initial_states, dummy_task, anchors_arr, 
            use_state_cond=False, desc="OOD Eval",
            stitch_steps_list=stitch_steps_list_ood
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

    # --------------------------------------------------------
    # Goal Multiplier Sweep
    # --------------------------------------------------------
    if args.goal_multipliers is not None and args.num_dataset_tasks > 0:
        multipliers = sorted(set(args.goal_multipliers) | {1.0})  # always include original
        print(f"\nGoal Multiplier Sweep: {multipliers}")
        print(f"Using {args.num_dataset_tasks} dataset tasks as base directions.")

        # Collect base initial conditions (same for every multiplier)
        start_indices = [i for i, (f, b, t) in enumerate(dataset.indices) if t == 0]
        if len(start_indices) < args.num_dataset_tasks:
            selected = start_indices
        else:
            selected = np.random.choice(start_indices, args.num_dataset_tasks, replace=False)

        base_states = []
        base_anchors = {'ref_pos': [], 'ref_quat': [], 'ref_obj_pos': [], 'final_obj_pos': []}
        base_gt_deltas = []  # unit-multiplier delta
        base_stitch_list = []

        for idx in tqdm(selected, desc="Prep Multiplier Base"):
            _, curr_state, _, anchor = dataset[idx]
            file_idx, batch_idx, _ = dataset.indices[idx]
            traj_data = dataset._get_single_traj(file_idx, batch_idx)
            final_pos = traj_data['obj'][-1, :3] if 'obj' in traj_data else anchor['ref_obj_pos'][:3]

            base_states.append(curr_state)
            base_anchors['ref_pos'].append(anchor['ref_pos'])
            base_anchors['ref_quat'].append(anchor['ref_quat'])
            base_anchors['ref_obj_pos'].append(anchor['ref_obj_pos'])
            base_anchors['final_obj_pos'].append(final_pos)  # unscaled
            base_gt_deltas.append((final_pos - anchor['ref_obj_pos'])[:data_cfg["num_task_params"]])

            traj_len = traj_data['obj'].shape[0] if 'obj' in traj_data else traj_data['joints'].shape[0]
            window_size = diffuser.input_size
            base_stitch_list.append(int(np.ceil((traj_len - 1) / (window_size - 1))))

        base_states_t = torch.stack(base_states)
        base_gt_deltas = np.array(base_gt_deltas)  # (N, num_task_params)
        N = len(base_states)

        # Run for each multiplier
        sweep_results = {}  # multiplier -> (gen_deltas, obj_traj_w, errors)
        for mult in multipliers:
            print(f"\n--- Goal Multiplier = {mult} ---")
            # Scale the goal positions
            scaled_anchors = {k: np.stack(v) for k, v in base_anchors.items()}
            for i in range(N):
                ref_obj = scaled_anchors['ref_obj_pos'][i]
                orig_final = scaled_anchors['final_obj_pos'][i].copy()
                delta = orig_final - ref_obj
                scaled_final = ref_obj.copy()
                scaled_final[:2] += mult * delta[:2]
                scaled_anchors['final_obj_pos'][i] = scaled_final

            dummy_task = torch.zeros(N, data_cfg["num_task_params"])

            traj_w, gen_disp = run_evaluation_batch(
                args, diffuser, dataset, device,
                base_states_t.clone(), dummy_task, scaled_anchors,
                use_state_cond=True, desc=f"Multiplier {mult}",
                stitch_steps_list=base_stitch_list * int(mult),
            )

            scaled_gt = base_gt_deltas * mult  # expected delta at this multiplier
            gen_d = gen_disp[:, :data_cfg["num_task_params"]]
            errs = np.linalg.norm(scaled_gt - gen_d, axis=1)
            print(f"  Mean Error: {np.mean(errs):.4f}, Std: {np.std(errs):.4f}")
            sweep_results[mult] = (gen_d, traj_w, errs, scaled_gt)

        # ---- Combined Trajectory + Target Plot (single image) ----
        # Plot relative to robot frame
        n_mult = len(multipliers)
        cmap = cm.get_cmap("tab10", n_mult)
        mult_colors = {m: cmap(i) for i, m in enumerate(multipliers)}

        # Helper to transform world points to ego-relative displacement
        def to_ego_frame(world_pts, start_obj_pos, start_robot_quat):
            # world_pts: (N, T, 2) or (N, 2)
            # start_obj_pos: (N, 2)
            # start_robot_quat: (N, 4)
            if world_pts.ndim == 2:
                world_pts = world_pts[:, None, :] # (N, 1, 2)
            
            # 1. Translation relative to object start (displacement)
            diff = world_pts - start_obj_pos[:, None, :2] # (N, T, 2)

            # 2. Rotation into robot frame
            # Using torch for robust yaw extraction (consistent with model)
            q_t = torch.from_numpy(start_robot_quat).float()
            
            # Check shape of q_t. Ensure it is correct for yaw_from_quat
            # Assuming yaw_from_quat is robust to [x,y,z,w] vs [w,x,y,z] if it's the one from flexible_dataset
            yaw = yaw_from_quat(q_t) # (N,) or (N, 1)
            yaw = yaw.view(-1).numpy()

            # Rotate by -yaw to get into ego frame
            # The rotation matrix from Body to World is R(yaw).
            # We want World to Body (Displacement), so R(-yaw).
            c = np.cos(-yaw)
            s = np.sin(-yaw)
            
            x = diff[..., 0]
            y = diff[..., 1]
            new_x = c[:, None] * x - s[:, None] * y
            new_y = s[:, None] * x + c[:, None] * y
            
            return np.stack([new_x, new_y], axis=-1)

        base_ref_obj = np.stack(base_anchors['ref_obj_pos'])[:, :2] # (N, 2)
        base_ref_quat = np.stack(base_anchors['ref_quat']) # (N, 4)

        # Gather ego-frame points for limits
        all_x, all_y = [], []
        
        sweep_ego_data = {} # Cache transformed data
        
        for mult in multipliers:
            gen_d, traj_w, _, scaled_gt = sweep_results[mult]
            
            # Trajectories (N, T, 2)
            traj_ego = to_ego_frame(traj_w[:, :, :2], base_ref_obj, base_ref_quat) # (N, T, 2)
            
            # Targets (N, 2) -> (N, 1, 2)
            targets_world = base_ref_obj + mult * base_gt_deltas[:, :2]
            target_ego = to_ego_frame(targets_world, base_ref_obj, base_ref_quat) # (N, 1, 2)
            target_ego = target_ego.squeeze(1) # (N, 2)
            
            sweep_ego_data[mult] = (traj_ego, target_ego)
            
            all_x.append(traj_ego[..., 0].flatten())
            all_y.append(traj_ego[..., 1].flatten())
            all_x.append(target_ego[..., 0].flatten())
            all_y.append(target_ego[..., 1].flatten())

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        max_dist = np.max(np.sqrt(all_x**2 + all_y**2)) * 1.15

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Origin is always (0,0) - object start
        ax.scatter([0], [0], marker='o', s=100, c='black', zorder=10, label='obj start')

        for mult in multipliers:
            c = mult_colors[mult]
            gen_d, traj_w, errs, scaled_gt = sweep_results[mult]
            traj_ego, target_ego = sweep_ego_data[mult]
            
            is_orig = (mult == 1.0)
            lw = 2.0 if is_orig else 1.2
            alpha_line = 0.8 if is_orig else 0.5

            # Plot target stars
            ax.scatter(target_ego[:, 0], target_ego[:, 1],
                       marker='*', s=120, c=[c], edgecolors='black', linewidths=0.5,
                       zorder=8, label=f"target ×{mult}")

            # Plot achieved end positions
            end_ego = traj_ego[:, -1, :]
            ax.scatter(end_ego[:, 0], end_ego[:, 1],
                       marker='D', s=40, c=[c], edgecolors='white', linewidths=0.6,
                       zorder=7, alpha=0.85,
                       label=f"achieved ×{mult} (err={np.mean(errs):.3f})")

            # Plot trajectories
            for i in range(N):
                path = traj_ego[i]
                ax.plot(path[:, 0], path[:, 1], color=c, alpha=alpha_line,
                        linewidth=lw)

                # Thin line from achieved end to target
                ax.plot([end_ego[i, 0], target_ego[i, 0]],
                        [end_ego[i, 1], target_ego[i, 1]],
                        color=c, alpha=0.2, linewidth=0.6, linestyle='--')

        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-max_dist, max_dist)
        ax.set_xlabel("Forward (Robot Frame)")
        ax.set_ylabel("Left (Robot Frame)")
        ax.set_title("Goal Multiplier Sweep — Object Displacement (Robot Frame)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        fig.tight_layout()
        fig.savefig("eval_goal_multiplier_all.png", dpi=150)
        print(f"Saved combined trajectory plot to eval_goal_multiplier_all.png")

        # ---- Error vs Multiplier Curve ----
        fig_curve, ax_c = plt.subplots(1, 1, figsize=(8, 5))
        means = [np.mean(sweep_results[m][2]) for m in multipliers]
        stds = [np.std(sweep_results[m][2]) for m in multipliers]
        ax_c.errorbar(multipliers, means, yerr=stds, marker='o', capsize=4)
        ax_c.set_xlabel("Goal Multiplier")
        ax_c.set_ylabel("L2 Displacement Error")
        ax_c.set_title("Goal Multiplier vs Displacement Error")
        ax_c.grid(True, alpha=0.3)
        fig_curve.tight_layout()
        fig_curve.savefig("eval_goal_multiplier_curve.png", dpi=150)
        print(f"Saved error curve to eval_goal_multiplier_curve.png")


if __name__ == "__main__":
    main()

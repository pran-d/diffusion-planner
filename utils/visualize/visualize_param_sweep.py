
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
from utils.visualize.visualize import MjVisualizer
from inference import build_inference_waypoints, DEFAULT_LIFT_HEIGHT, run_visualization

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

    # Waypoint arguments
    parser.add_argument("--last_frame_waypoint", action="store_true",
                        help="Add partial waypoint at last frame (obj_delta_xy + obj_z absolute) when inbetweening is enabled")
    parser.add_argument("--arrival_ratio", type=float, default=0.85,
                        help="Object arrives in this fraction of remaining time (0-1)")
    parser.add_argument("--lift_height", type=float, default=DEFAULT_LIFT_HEIGHT,
                        help=f"Peak lift height in meters for pick-and-place z profile (default: {DEFAULT_LIFT_HEIGHT}m)")
    parser.add_argument("--lift_start", type=float, default=0.0, help="Fraction of trajectory where lift begins (0=immediately)")
    parser.add_argument("--lift_end", type=float, default=0.20, help="Fraction of trajectory where lift reaches peak")
    parser.add_argument("--walk_start_z", type=float, default=0.80,
                        help="Gate XY motion: don't walk until z >= this fraction of lift_height (default: 0.80)")
    parser.add_argument("--no_lower_dist", type=float, default=0.5,
                        help="Lower z when remaining XY distance drops below this value in metres (default: 0.5m)")
    parser.add_argument("--visualize", action="store_true",
                        help="Render generated trajectories in MuJoCo after evaluation")

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
    
    # Track current anchors which update over time
    current_anchors = {
        k: v.copy() for k, v in anchors_dict.items()
    }
    
    # We maintain current state tensor if using state cond
    curr_state_tens = initial_states.to(device) if initial_states is not None else None

    if stitch_steps_list is not None:
        max_stitch_steps = max(stitch_steps_list)
        print(f"Using max stitch_steps {max_stitch_steps} for this batch.")
    elif args.stitch_steps is None:
        _eff = diffuser.input_size - 1  # t=0 always skipped; each window yields (T-1) frames
        max_stitch_steps = (dataset.traj_lengths[args.sample_idx] + _eff - 1) // _eff
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

    # Physical-consistency sanity check (task-agnostic).
    _FLOOR_Z        = -0.1   # robot/object z below this → ground penetration
    _MAX_ROBOT_STEP = 0.1    # max robot pelvis XY step per frame at 100 Hz (m)
    _MAX_OBJ_STEP   = 0.2    # max object XY step per frame at 100 Hz (m)
    _prev_robot_xyz = anchors_dict['ref_pos'][:, :3].copy()    # (N, 3)
    _prev_obj_xyz   = anchors_dict['ref_obj_pos'][:, :3].copy() # (N, 3)

    # Per-trajectory goal-stop state
    enable_goal_stop     = getattr(args, 'enable_goal_stop', True)
    enable_physics_stop  = getattr(args, 'enable_physics_stop', True)
    goal_stop_threshold  = getattr(args, 'goal_stop_threshold', 0.1)
    goal_reached         = np.zeros(num_samples, dtype=bool)
    _goal_last_frame     = np.zeros((num_samples, 43), dtype=np.float64)

    # Waypoint config: always build first-frame keyframe when inbetweening is enabled
    # (matches inference.py which calls build_inference_waypoints unconditionally).
    # use_last_frame_wp controls whether the *last*-frame partial waypoint is also added.
    inbetweening_active = (
        hasattr(diffuser, 'model')
        and getattr(diffuser.model, 'inbetweening_enabled', False)
    )
    use_last_frame_wp = getattr(args, 'last_frame_waypoint', False)
    arrival_ratio = getattr(args, 'arrival_ratio', 0.85)
    lift_height = getattr(args, 'lift_height', DEFAULT_LIFT_HEIGHT)
    no_lower_dist = getattr(args, 'no_lower_dist', 0.5)
    lift_start = getattr(args, 'lift_start', 0.0)
    lift_end = getattr(args, 'lift_end', 0.20)
    walk_start_z = getattr(args, 'walk_start_z', 0.80)
    num_features = dataset.num_features
    window_size = dataset.window_size

    # Episode-start object z — absolute baseline for z-waypoint targeting.
    _rest_obj_z_np = anchors_dict['ref_obj_pos'][:, 2].copy()  # (N,)

    for step in range(max_stitch_steps):
        # --------------------------------------------------------
        # Dynamic Task Re-Targeting (Closed Loop Control)
        # --------------------------------------------------------
        new_task_list = []
        raw_task_list = []  # unclipped task params for waypoint building
        for bi in range(num_samples):
            c_quat = current_anchors['ref_quat'][bi]
            c_obj = current_anchors['ref_obj_pos'][bi]
            c_goal = current_anchors['final_obj_pos'][bi]
            
            tp, actual_dist = compute_task_params(
                c_quat, c_obj, c_goal,
                normalize_goal_vec=dataset.normalize_goal_vec,       
                num_task_params=dataset.num_task_params,
                max_goal_dist=dataset.max_obj_displacement,
            ) # (2,) or (3,) depending on impl
            new_task_list.append(tp)

            # Keep a copy with actual (unclipped) distance for waypoint building
            tp_raw = tp.copy()
            if tp_raw.shape[-1] >= 3:
                tp_raw[..., 2] = actual_dist.item() if hasattr(actual_dist, 'item') else float(actual_dist)
            raw_task_list.append(tp_raw)
        
        new_task_arr = np.stack(new_task_list)
        new_task_tens = torch.from_numpy(new_task_arr).float()
        raw_task_arr = np.stack(raw_task_list)
        
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

            # Build waypoints: always provide t=0 keyframe when inbetweening is active;
            # optionally also add the last-frame partial waypoint.
            wv, wm = None, None
            if inbetweening_active and batch_state is not None:
                remaining = max(max_stitch_steps - step, 1)
                batch_raw_task = torch.from_numpy(raw_task_arr[b_start:b_end]).float()
                _current_obj_z_b = torch.from_numpy(
                    current_anchors['ref_obj_pos'][b_start:b_end, 2].astype(np.float32)
                )
                _rest_obj_z_b = torch.from_numpy(
                    _rest_obj_z_np[b_start:b_end].astype(np.float32)
                )
                wv, wm = build_inference_waypoints(
                    batch_state,
                    batch_raw_task,
                    dataset,
                    num_features,
                    window_size,
                    use_last_frame_wp=use_last_frame_wp,
                    stitch_steps=max_stitch_steps,
                    remaining_steps=remaining,
                    arrival_ratio=arrival_ratio,
                    lift_height=lift_height,
                    no_lower_dist=no_lower_dist,
                    lift_start=lift_start,
                    lift_end=lift_end,
                    walk_start_z=walk_start_z,
                    current_obj_z=_current_obj_z_b,
                    rest_obj_z=_rest_obj_z_b,
                )
            
            sample = diffuser.getSample(
                num_trajectories=bs,
                state_cond=batch_state,
                goal_cond=batch_task,
                deterministic=True,
                cfg_w=args.cfg_w,
                guidance_wt=args.guidance_wt,
                no_state_cond=not use_state_cond,
                waypoint_values=wv,
                waypoint_mask=wm,
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
                robot_world = robot_world[:, 1:, :]  # skip t=0 then apply horizon
                obj_world   = obj_world[:, 1:, :]
                robot_world = robot_world[:, :args.action_horizon, :]
                obj_world = obj_world[:, :args.action_horizon, :]
            else:
                robot_world = robot_world[:, 1:, :]  # skip t=0 (anchored to current state)
                obj_world   = obj_world[:, 1:, :]

            # Store (B, T, D)
            segment = np.concatenate([robot_world[..., :36], obj_world[..., :7]], axis=-1)
            step_segments.append(segment)
            
        # Concatenate batches for this step
        full_step_segment = np.concatenate(step_segments, axis=0) # (N, T, D)

        # ── Physical-consistency check ─────────────────────────────────────────
        _rxy   = full_step_segment[:, :, :2]
        _rz    = full_step_segment[:, :, 2]
        _oxy   = full_step_segment[:, :, 36:38]
        _oz    = full_step_segment[:, :, 38]
        _rxy_p = np.concatenate([_prev_robot_xyz[:, None, :2], _rxy[:, :-1, :]], axis=1)
        _oxy_p = np.concatenate([_prev_obj_xyz[:, None, :2],   _oxy[:, :-1, :]], axis=1)
        _floor   = (_rz < _FLOOR_Z) | (_oz < _FLOOR_Z)
        _rspike  = np.linalg.norm(_rxy - _rxy_p, axis=-1) > _MAX_ROBOT_STEP
        _ospike  = np.linalg.norm(_oxy - _oxy_p, axis=-1) > _MAX_OBJ_STEP
        _bad_t   = np.where((_floor | _rspike | _ospike).any(axis=0))[0]
        _phys_stop = enable_physics_stop and _bad_t.size > 0
        if _phys_stop:
            _t_bad   = int(_bad_t[0])
            _t_clamp = max(_t_bad, 1)
            _last_ok = full_step_segment[:, _t_clamp - 1 : _t_clamp, :]
            full_step_segment = full_step_segment.copy()
            full_step_segment[:, _t_clamp:, :] = _last_ok
            print(f"[physics] step {step+1}: violation at frame {_t_bad} "
                  f"(floor={bool(_floor.any(0)[_t_bad])}, "
                  f"robot_spike={bool(_rspike.any(0)[_t_bad])}, "
                  f"obj_spike={bool(_ospike.any(0)[_t_bad])}). Truncating and halting.")
        # ──────────────────────────────────────────────────────────────────────

        # ── Per-trajectory goal-stop ──────────────────────────────────────────
        if enable_goal_stop:
            # 1. Overwrite segments for trajectories that already reached goal
            if goal_reached.any():
                _seg_T = full_step_segment.shape[1]
                full_step_segment = full_step_segment.copy()
                for _di in np.where(goal_reached)[0]:
                    full_step_segment[_di] = np.tile(_goal_last_frame[_di], (_seg_T, 1))

            # 2. Detect trajectories reaching goal at the end of this segment
            _obj_end  = full_step_segment[:, -1, 36:39]              # (N, 3)
            _goal_3d  = current_anchors['final_obj_pos'][:, :3]      # (N, 3)
            _goal_err = np.linalg.norm(_obj_end - _goal_3d, axis=-1) # (N,)
            _newly_reached = (~goal_reached) & (_goal_err < goal_stop_threshold)
            if _newly_reached.any():
                for _ni in np.where(_newly_reached)[0]:
                    _goal_last_frame[_ni] = full_step_segment[_ni, -1, :]
                goal_reached |= _newly_reached
                print(f"[goal] step {step+1}: traj {np.where(_newly_reached)[0].tolist()} "
                      f"reached goal (err={_goal_err[_newly_reached].round(4).tolist()})")
        # ─────────────────────────────────────────────────────────────────────

        generated_segments.append(full_step_segment)

        # Update per-window position trackers for spike detection at next boundary
        _prev_robot_xyz = full_step_segment[:, -1, :3].copy()
        _prev_obj_xyz   = full_step_segment[:, -1, 36:39].copy()

        if _phys_stop:
            _steps_to_pad = max_stitch_steps - step - 1
            if _steps_to_pad > 0:
                _last_frame = full_step_segment[:, -1:, :]
                _seg_T      = full_step_segment.shape[1]
                _pad_seg    = np.repeat(_last_frame, _seg_T, axis=1)
                for _ in range(_steps_to_pad):
                    generated_segments.append(_pad_seg)
            break

        # ── Early-exit when all trajectories have reached their goal ──────────
        if enable_goal_stop and goal_reached.all():
            _steps_to_pad = max_stitch_steps - step - 1
            if _steps_to_pad > 0:
                _seg_T   = full_step_segment.shape[1]
                _all_pad = np.stack([
                    np.tile(_goal_last_frame[_i], (_seg_T, 1))
                    for _i in range(num_samples)
                ])
                for _ in range(_steps_to_pad):
                    generated_segments.append(_all_pad.copy())
                print(f"[goal] All {num_samples} trajectories reached goal after step "
                      f"{step+1}. Padding {_steps_to_pad} remaining steps.")
            break
        # ─────────────────────────────────────────────────────────────────────

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

    return full_traj, obj_traj_w, np.stack(final_displacements)

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

def to_ego_frame(world_pts, start_obj_pos, start_robot_quat):
    """Transform world XY points into ego-frame displacement (centred on object start,
    rotated into robot heading frame).  Works per-sample: each sample i is independently
    centred and rotated.

    Args:
        world_pts:        (N, T, 2) or (N, 2)
        start_obj_pos:    (N, 2) or (N, 3)  — object position at t=0
        start_robot_quat: (N, 4)            — robot base quaternion at t=0
    Returns:
        (N, T, 2) ego-frame XY
    """
    if world_pts.ndim == 2:
        world_pts = world_pts[:, None, :]  # (N, 1, 2)
    diff = world_pts - start_obj_pos[:, None, :2]  # (N, T, 2)
    q_t  = torch.from_numpy(np.asarray(start_robot_quat)).float()
    yaw  = yaw_from_quat(q_t).view(-1).numpy()     # (N,)
    c, s = np.cos(-yaw), np.sin(-yaw)
    x, y = diff[..., 0], diff[..., 1]
    return np.stack([c[:, None] * x - s[:, None] * y,
                     s[:, None] * x + c[:, None] * y], axis=-1)


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

    # MuJoCo visualizer (created once, reused for all visualizations)
    vis = None
    if args.visualize:
        xml_path = "mj_model.xml"
        if not os.path.exists(xml_path):
            xml_path = os.path.join(data_path, "mj_model.xml")
        if os.path.exists(xml_path):
            vis = MjVisualizer(xml_path, close_on_enter=False)
        else:
            print(f"Warning: mj_model.xml not found at '{xml_path}', --visualize disabled.")
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
        full_traj_ds, traj_w, gen_displacements = run_evaluation_batch(
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

        # MuJoCo visualization
        if vis is not None:
            N_vis = len(full_traj_ds)
            print(f"Visualizing {N_vis} dataset trajectories in MuJoCo (press Enter to advance)...")
            for i in range(N_vis):
                print(f"  Trajectory {i+1}/{N_vis}")
                run_visualization(vis, full_traj_ds[i:i+1],
                                  final_obj_pos=anchors_arr['final_obj_pos'][i:i+1])

        _ds_traj_w_cache  = traj_w
        _ds_anchors_cache = anchors_arr

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
        full_traj_ood, traj_w, gen_displacements = run_evaluation_batch(
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

        # MuJoCo visualization
        if vis is not None:
            N_vis = len(full_traj_ood)
            print(f"Visualizing {N_vis} OOD trajectories in MuJoCo (press Enter to advance)...")
            for i in range(N_vis):
                print(f"  Trajectory {i+1}/{N_vis}")
                run_visualization(vis, full_traj_ood[i:i+1],
                                  final_obj_pos=anchors_arr['final_obj_pos'][i:i+1])

    # --------------------------------------------------------
    # Goal Multiplier Sweep
    # --------------------------------------------------------
    if args.goal_multipliers is not None and args.num_dataset_tasks > 0:
        multipliers = sorted(set(args.goal_multipliers))  # always include original
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

            full_traj_mult, traj_w, gen_disp = run_evaluation_batch(
                args, diffuser, dataset, device,
                base_states_t.clone(), dummy_task, scaled_anchors,
                use_state_cond=True, desc=f"Multiplier {mult}",
                stitch_steps_list=base_stitch_list * int(mult),
            )

            # MuJoCo visualization for this multiplier
            if vis is not None:
                N_vis = len(full_traj_mult)
                print(f"  Visualizing {N_vis} trajectories for multiplier {mult} (press Enter to advance)...")
                for i in range(N_vis):
                    run_visualization(vis, full_traj_mult[i:i+1],
                                      final_obj_pos=scaled_anchors['final_obj_pos'][i:i+1])

            scaled_gt = base_gt_deltas * mult  # expected delta at this multiplier
            gen_d = gen_disp[:, :data_cfg["num_task_params"]]
            errs = np.linalg.norm(scaled_gt - gen_d, axis=1)
            print(f"  Mean Error: {np.mean(errs):.4f}, Std: {np.std(errs):.4f}")
            sweep_results[mult] = (gen_d, traj_w, errs, scaled_gt)

        # ---- Combined Trajectory + Target Plot (single image) ----
        # Plot relative to robot ego frame (see module-level to_ego_frame)
        n_mult = len(multipliers)
        cmap = cm.get_cmap("tab10", n_mult)
        mult_colors = {m: cmap(i) for i, m in enumerate(multipliers)}

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

        _sweep_results      = sweep_results
        _sweep_base_anchors = base_anchors
        _sweep_base_gt      = base_gt_deltas

    # --------------------------------------------------------
    # Combined dataset + goal-multiplier sweep plot
    # --------------------------------------------------------
    if _ds_traj_w_cache is not None and _sweep_results is not None:
        print("\nGenerating combined dataset + sweep plot...")
        base_ref_obj_ds  = _ds_anchors_cache['ref_obj_pos']  # (N_ds, 3)
        base_ref_quat_ds = _ds_anchors_cache['ref_quat']     # (N_ds, 4)
        N_ds = len(_ds_traj_w_cache)

        # Dataset trajectories in ego frame
        ds_traj_ego = to_ego_frame(
            _ds_traj_w_cache[:, :, :2], base_ref_obj_ds, base_ref_quat_ds)
        ds_goal_world = _ds_anchors_cache['final_obj_pos'][:, :2]  # (N_ds, 2)
        ds_goal_ego   = to_ego_frame(ds_goal_world, base_ref_obj_ds, base_ref_quat_ds).squeeze(1)

        # Sweep trajectories in ego frame (use sweep's own base anchors)
        base_ref_obj_sw  = np.stack(_sweep_base_anchors['ref_obj_pos'])[:, :2]
        base_ref_quat_sw = np.stack(_sweep_base_anchors['ref_quat'])

        fig_comb, ax_comb = plt.subplots(1, 1, figsize=(12, 12))
        ax_comb.scatter([0], [0], marker='o', s=120, c='black', zorder=10, label='obj start')

        # Dataset trajectories (grey)
        for i in range(N_ds):
            path = ds_traj_ego[i]        # (T, 2)
            ax_comb.plot(path[:, 0], path[:, 1], color='grey', alpha=0.45,
                         linewidth=1.0, label='dataset' if i == 0 else None)
            ax_comb.scatter(ds_goal_ego[i, 0], ds_goal_ego[i, 1],
                            marker='*', s=80, c='grey', edgecolors='white', linewidths=0.5,
                            zorder=7, label='dataset goal' if i == 0 else None)

        # Sweep trajectories
        n_mult = len(_sweep_results)
        cmap_comb = cm.get_cmap('tab10', max(n_mult, 1))
        for ci, mult in enumerate(sorted(_sweep_results.keys())):
            color = cmap_comb(ci)
            _, traj_w_sw, errs, _ = _sweep_results[mult]
            traj_ego_sw = to_ego_frame(traj_w_sw[:, :, :2],
                                        base_ref_obj_sw, base_ref_quat_sw)
            targets_world = base_ref_obj_sw + mult * _sweep_base_gt[:, :2]
            target_ego_sw = to_ego_frame(targets_world, base_ref_obj_sw,
                                          base_ref_quat_sw).squeeze(1)
            for i in range(len(traj_ego_sw)):
                ax_comb.plot(traj_ego_sw[i, :, 0], traj_ego_sw[i, :, 1],
                             color=color, alpha=0.6, linewidth=1.4,
                             label=f'sweep ×{mult} (err={np.mean(errs):.3f})' if i == 0 else None)
            ax_comb.scatter(target_ego_sw[:, 0], target_ego_sw[:, 1],
                            marker='*', s=100, c=[color], edgecolors='black',
                            linewidths=0.5, zorder=8)

        all_pts = np.concatenate([ds_traj_ego.reshape(-1, 2), ds_goal_ego], axis=0)
        max_d = np.max(np.linalg.norm(all_pts, axis=-1)) * 1.15 + 0.1
        ax_comb.set_xlim(-max_d, max_d)
        ax_comb.set_ylim(-max_d, max_d)
        ax_comb.set_xlabel('Forward (Robot Frame)')
        ax_comb.set_ylabel('Left (Robot Frame)')
        ax_comb.set_title('Dataset Trajectories + Goal-Multiplier Sweep (Ego Frame)')
        ax_comb.set_aspect('equal')
        ax_comb.grid(True, alpha=0.3)
        handles, labels = ax_comb.get_legend_handles_labels()
        seen = {}; dedup_h, dedup_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True; dedup_h.append(h); dedup_l.append(l)
        ax_comb.legend(dedup_h, dedup_l, fontsize=8, ncol=2, loc='upper left')
        fig_comb.tight_layout()
        fig_comb.savefig('eval_combined.png', dpi=150)
        print('Combined plot saved to eval_combined.png')

    if vis is not None:
        vis.close()

if __name__ == "__main__":
    main()

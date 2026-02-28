import torch
import numpy as np
import yaml
import os
import argparse
from config.configure import load_config, get_data_path, get_norm_path
from models.model import RobotDiffuser
from datasets import BufferDataset
from utils.data.load_dataset import preload_dataset
from datasets.flexible_dataset import yaw_to_rot_matrix, yaw_from_quat
from utils.math.sbto_utils import reconstruct_sbto_trajectory, compute_task_params, FEATURE_LAYOUT_NO_VEL


def build_keyframes(normalized_window, keep_first=True, keep_last=True, num_samples=1):
    """
    Build waypoint tensor and feature-level mask from a normalized trajectory window.

    Args:
        normalized_window: (T, D) tensor — a single normalized ground truth window
        keep_first: whether to mark t=0 as a waypoint (all features)
        keep_last:  whether to mark t=-1 as a waypoint (all features)
        num_samples: batch size to tile to

    Returns:
        waypoint_values: (B, T, D) tensor
        waypoint_mask:   (B, T, D) bool tensor — True = known feature
    """
    T, D = normalized_window.shape
    values = normalized_window.unsqueeze(0).repeat(num_samples, 1, 1)  # (B, T, D)
    mask = torch.zeros(num_samples, T, D, dtype=torch.bool)
    if keep_first:
        mask[:, 0, :] = True
    if keep_last:
        mask[:, -1, :] = True
    return values, mask


def build_first_keyframe_from_state(curr_state_tens, dataset, num_features, window_size):
    """
    Build a waypoint tensor where only t=0 is marked (all features known),
    constructed from the current observation state.  The first
    `num_features - num_observations` dims (delta_xy, delta_yaw, obj_delta_xy)
    are zero at t=0 (no motion yet).

    Args:
        curr_state_tens: (B, H, obs_dim)  — current observation (last H frames)
        dataset:         the dataset object (for normalization)
        num_features:    total feature dim (D)
        window_size:     T tokens in a window

    Returns:
        waypoint_values: (B, T, D)
        waypoint_mask:   (B, T, D) bool — True at t=0 for all features
    """
    B = curr_state_tens.shape[0]
    obs_dim = curr_state_tens.shape[-1]
    D = num_features
    T = window_size

    # At t=0 deltas are zero — normalize them
    prefix_dim = D - obs_dim  # typically 5 (delta_xy=2, delta_yaw=1, obj_delta_xy=2)

    # Normalize each delta component
    prefix_parts = []
    cumulative = 0
    for key in dataset.feature_order:
        key_dim_map = {
            'delta_xy': 2, 'delta_yaw': 1, 'obj_delta_xy': 2,
        }
        if key in key_dim_map:
            kd = key_dim_map[key]
            part = dataset._normalize(key, torch.zeros(1, kd))
            prefix_parts.append(part.squeeze(0))
            cumulative += kd
        if cumulative >= prefix_dim:
            break

    if prefix_parts:
        normalized_prefix = torch.cat(prefix_parts)  # (prefix_dim,)
    else:
        normalized_prefix = torch.zeros(prefix_dim)

    # Build first frame: [normalized_deltas=0, observation_features]
    obs_frame = curr_state_tens[:, -1, :]  # (B, obs_dim) — latest observation frame
    first_frame = torch.cat([
        normalized_prefix.unsqueeze(0).expand(B, -1),
        obs_frame
    ], dim=-1)  # (B, D)

    # Construct waypoint tensor (B, T, D) — zeros, then fill t=0
    waypoint_values = torch.zeros(B, T, D, device=curr_state_tens.device)
    waypoint_values[:, 0, :] = first_frame

    # Feature-level mask: all features known at t=0
    waypoint_mask = torch.zeros(B, T, D, dtype=torch.bool, device=curr_state_tens.device)
    waypoint_mask[:, 0, :] = True

    return waypoint_values, waypoint_mask


def build_last_frame_waypoints(
    task_params_raw,     # (B, num_task_params) denormalized task params
    window_size,         # T
    dataset,             # for normalization
    feature_order,       # list of feature keys
    num_features,        # D
    remaining_steps=1,   # how many windows remain to reach goal
    arrival_ratio=0.85,  # finish in this fraction of remaining time
):
    """
    Build a partial waypoint at the last frame (t=T-1) of the window.

    Sets:
    - obj_delta_xy: constant-velocity object displacement to reach goal,
      arriving in `arrival_ratio` fraction of the remaining time.
- delta_yaw: robot faces along the goal direction.

    Returns:
        waypoint_values: (B, T, D) float tensor — values at waypoint positions
        waypoint_mask:   (B, T, D) bool tensor — True = known feature
    """
    if isinstance(task_params_raw, np.ndarray):
        task_params_raw = torch.from_numpy(task_params_raw).float()
    if task_params_raw.ndim == 1:
        task_params_raw = task_params_raw.unsqueeze(0)

    B = task_params_raw.shape[0]
    T = window_size
    D = num_features

    values = torch.zeros(B, T, D)
    mask = torch.zeros(B, T, D, dtype=torch.bool)

    # Parse task params: [dir_x, dir_y, distance] or [delta_x, delta_y]
    if task_params_raw.shape[-1] >= 3:
        direction = task_params_raw[:, :2]   # (B, 2)
        distance = task_params_raw[:, 2:3]   # (B, 1)
        total_remaining_delta = direction * distance  # (B, 2)
    else:
        total_remaining_delta = task_params_raw[:, :2]  # (B, 2)

    # Per-window object displacement (arrive early)
    effective_remaining = max(remaining_steps * arrival_ratio, 1.0)
    per_window_delta = total_remaining_delta / effective_remaining  # (B, 2)

    # Robot yaw: face along goal direction
    goal_yaw = torch.atan2(task_params_raw[:, 1], task_params_raw[:, 0])  # (B,)

    # Build feature index map
    feature_idx = {}
    idx = 0
    for key in feature_order:
        dim = FEATURE_LAYOUT_NO_VEL.get(key, 0)
        if dim > 0:
            feature_idx[key] = slice(idx, idx + dim)
            idx += dim

    t_last = T - 1

    # Set obj_delta_xy at last frame
    if "obj_delta_xy" in feature_idx:
        norm_obj_delta = dataset._normalize("obj_delta_xy", per_window_delta)
        if not isinstance(norm_obj_delta, torch.Tensor):
            norm_obj_delta = torch.tensor(norm_obj_delta, dtype=torch.float32)
        values[:, t_last, feature_idx["obj_delta_xy"]] = norm_obj_delta.reshape(B, -1)
        mask[:, t_last, feature_idx["obj_delta_xy"]] = True

    return values, mask


def build_inference_waypoints(
    curr_state_tens,       # (B, H, obs_dim)
    task_params_raw,       # (B, num_task_params) denormalized task params
    dataset,
    num_features,
    window_size,
    use_last_frame_wp=True,  # whether to add last-frame partial waypoint
    remaining_steps=1,
    arrival_ratio=0.85,
):
    """
    Build complete waypoint specification for inference.
    Combines a full keyframe at t=0 (from current state) with an optional
    analytically-computed partial waypoint at the last frame (obj_delta_xy + delta_yaw).

    Returns:
        waypoint_values: (B, T, D) float tensor
        waypoint_mask:   (B, T, D) bool tensor
    """
    B = curr_state_tens.shape[0]
    D = num_features
    T = window_size

    # 1. Full keyframe at t=0 from current state
    kf_vals, _ = build_first_keyframe_from_state(
        curr_state_tens, dataset, num_features, window_size
    )

    values = kf_vals.clone()
    mask = torch.zeros(B, T, D, dtype=torch.bool, device=curr_state_tens.device)
    mask[:, 0, :] = True  # all features known at t=0

    # 2. Last-frame partial waypoint (obj_delta_xy + delta_yaw)
    if use_last_frame_wp and task_params_raw is not None:
        wp_vals, wp_mask = build_last_frame_waypoints(
            task_params_raw=task_params_raw,
            window_size=window_size,
            dataset=dataset,
            feature_order=dataset.feature_order,
            num_features=num_features,
            remaining_steps=remaining_steps,
            arrival_ratio=arrival_ratio,
        )
        # Merge: last-frame waypoint fills in only its marked positions
        values = torch.where(wp_mask, wp_vals.to(values.device), values)
        mask = mask | wp_mask.to(mask.device)

    return values, mask


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


def run_visualization(vis, stitched_trajs, goal_vectors=None, final_obj_pos=None, repeat=True):    
    # Use first sample (num_samples, T, D) -> (T, D)
    if stitched_trajs.ndim == 3:
        traj = stitched_trajs[0] 
    else:
        traj = stitched_trajs

    T_steps = traj.shape[0]
    t = np.arange(T_steps) * 0.01

    vis.visualize_trajectory(t=t, x_traj=traj, repeat=repeat, guidance_vec=goal_vectors, goal_pos=final_obj_pos)

def main():
    parser = argparse.ArgumentParser(description="Clean Inference & Stitching Pipeline")
    parser.add_argument("--epoch", type=str, required=True, help="Checkpoint epoch or path")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--stitch_steps", type=int, default=None, help="Number of autoregressive segments to generate")
    parser.add_argument("--save_path", type=str, default="results/inference.npy")
    parser.add_argument("--sample_idx", type=int, default=0, help="Initial condition index (Overridden if traj_idx is set)")
    parser.add_argument("--traj_idx", type=int, default=0, help="Trajectory (file) index")
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch index within file")
    parser.add_argument("--start_time", type=int, default=0, help="Window start timestep")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda or cpu)")
    parser.add_argument("--cfg_w", type=float, default=1.0, help="Classifier-free guidance weight")
    parser.add_argument("--task_params", nargs="+", type=float, default=None, help="Custom task parameters (e.g., --task_params 0.5 -0.2)")
    parser.add_argument("--visualize_dataset", action="store_true", help="Whether to visualize the original dataset trajectory instead of the generated one")
    parser.add_argument("--action_horizon", type=int, default=None, help="Number of future steps to visualize/control (for dataset visualization)")
    parser.add_argument("--end_error_threshold", type=float, default=0.1, help="End error threshold for stitching")
    parser.add_argument("--goal_multiplier", type=float, default=1.0, help="Scaling factor for goal (for testing different r for same theta)")
    parser.add_argument("--visualize_windows", action="store_true", help="Render and save each generated window as a video")

    # Guidance arguments
    parser.add_argument("--guidance_wt", type=float, default=0.0, help="Test-time gradient guidance strength")
    parser.add_argument("--guidance_goal", nargs="+", type=float, default=None, help="Target values for guidance (normalized)")
    parser.add_argument("--guidance_indices", nargs="+", type=int, default=None, help="Indices of the state vector to apply guidance on")
    parser.add_argument("--last_frame_waypoint", action="store_true", help="Add partial waypoint at last frame (obj_delta_xy + delta_yaw)")
    parser.add_argument("--arrival_ratio", type=float, default=0.85, help="Object arrives in this fraction of remaining time (0-1)")
    
    args = parser.parse_args()

    # 1. Load Config
    config_path = "config/config.yaml"
    with open(config_path, 'r') as file:
        raw_config = yaml.safe_load(file)
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config(config_path, raw_config.get("auto_conf", False))
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 2. Setup Dataset & Stats
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
    calculate_stats = True
    if norm_path and os.path.exists(norm_path):
        calculate_stats = False

    data_buffer = preload_dataset(data_cfg, data_path)
    dataset = BufferDataset(
        data_buffer=data_buffer, config=data_cfg, task_params=None,
        calculate_stats=calculate_stats, norm_path=norm_path,
        training_cfg={},
    )

    from utils.visualize.visualize import MjVisualizer
    xml_path = "mj_model.xml"
    if not os.path.exists(xml_path):
        xml_path = os.path.join(data_path, "mj_model.xml")
    vis = MjVisualizer(xml_path, close_on_enter=False)

    # 3. Model
    diffuser = RobotDiffuser(
        model_config=model_cfg, data_config=data_cfg,
        training_config=training_cfg, noise_scheduler_config=noise_cfg,
        mode='inference', device=device
    )
    
    if os.path.exists(args.epoch):
        diffuser.load_weights_from_file(args.epoch)
    else:
        diffuser.loadWeights(int(args.epoch))

    # 4. Prepare Initial Condition
    if args.traj_idx is not None:
        target = (args.traj_idx, args.batch_idx, args.start_time)
        try:
            args.sample_idx = dataset.indices.index(target)
            print(f"Mapped {target} -> Sample {args.sample_idx}")
        except ValueError:
            print(f"Error: Target {target} not found in dataset indices.")
            raise ValueError(f"Target indices {target} (Trajectory {args.traj_idx}, Batch {args.batch_idx}, Start {args.start_time}) not present in valid window list.")

    print(f"Loading initial condition (Sample {args.sample_idx})...")

    # Capture start meta for task updates
    current_file_idx, current_batch_idx, current_start_time = dataset.indices[args.sample_idx]

    # Build index map for faster access
    index_map = {idx_tuple: i for i, idx_tuple in enumerate(dataset.indices)}

    fut_traj, curr_state, task_params, anchor = dataset[args.sample_idx]
    
    history_size = dataset.history_size
    stitched_segments = []
    unreconstructed_segments = []
    goal_vectors = []
    # Analysis data: per-window normalized outputs and waypoints
    analysis_normalized_windows = []
    analysis_denormalized_windows = []
    analysis_waypoint_values = []
    analysis_waypoint_masks = []

    # Override task params if provided
    if args.task_params is not None:
        anchor["final_obj_pos"][..., :2] = args.task_params
            
    if args.goal_multiplier != 1.0:
        print(f"Scaling goal by multiplier {args.goal_multiplier}")
        anchor["final_obj_pos"][..., :2] = anchor["ref_obj_pos"][..., :2] + args.goal_multiplier * (anchor["final_obj_pos"] - anchor["ref_obj_pos"])[..., :2]

    print(f"Using final box pos: {anchor['final_obj_pos']} for computing task parameters.")

    current_anchors = {
        'ref_pos': np.tile(anchor['ref_pos'][None], (args.num_samples, 1)),
        'ref_quat': np.tile(anchor['ref_quat'][None], (args.num_samples, 1)),
        'ref_obj_pos': np.tile(anchor['ref_obj_pos'][None], (args.num_samples, 1)),
        'final_obj_pos': np.tile(anchor['final_obj_pos'][None], (args.num_samples, 1)),
    }

    curr_state_tens = curr_state.unsqueeze(0).repeat(args.num_samples, 1, 1)
    task_tens = task_params.repeat(args.num_samples, 1)

    if args.stitch_steps is None:
        args.stitch_steps = dataset.traj_lengths[args.traj_idx] // data_cfg["num_timesteps"] 
        print(f"Auto-setting stitch_steps to {args.stitch_steps} based on dataset length.")

    if args.action_horizon is not None:
        args.stitch_steps *= (data_cfg["num_timesteps"] // args.action_horizon)
        print(f"Adjusting stitch_steps to {args.stitch_steps} based on action horizon")

    # 5. Autoregressive Loop
    for step in range(args.stitch_steps):
        # print(f"Generating segment {step+1}/{args.stitch_steps}...")

        # A. Inference
        if not args.visualize_dataset:
            tp_init, actual_dist = compute_task_params(
                current_robot_state=current_anchors['ref_quat'], 
                current_obj_state=current_anchors['ref_obj_pos'], 
                desired_obj_pos=current_anchors["final_obj_pos"],
                normalize_goal_vec=data_cfg.get("normalize_goal_vec", False),
                num_task_params=data_cfg["num_task_params"],
                max_goal_dist=dataset.max_obj_displacement
            )
            task_actual = tp_init.copy()
            task_actual[..., 2] = actual_dist  
            print(task_actual)

            task_params = dataset._normalize("task_params", tp_init)
            task_tens = task_params if isinstance(task_params, torch.Tensor) else torch.tensor(task_params, dtype=torch.float32)

            # Build waypoints: full keyframe at t=0 + optional last-frame partial waypoint
            wv, wm = None, None
            if args.last_frame_waypoint:
                remaining = max(args.stitch_steps - step, 1)
                wv, wm = build_inference_waypoints(
                    curr_state_tens, task_actual, dataset,
                    data_cfg['num_features'], dataset.window_size,
                    use_last_frame_wp=args.last_frame_waypoint,
                    remaining_steps=remaining,
                    arrival_ratio=args.arrival_ratio,
                )
            normalized_sample = diffuser.getSample(
                num_trajectories=args.num_samples,
                state_cond=curr_state_tens.to(device),
                goal_cond=task_tens.to(device),
                deterministic=True,
                cfg_w=args.cfg_w,
                guidance_wt=args.guidance_wt,
                guidance_goal=args.guidance_goal,
                waypoint_values=wv,
                waypoint_mask=wm,
            )
        else:
            # Bypass generative model for visualization:
            # Use original dataset future trajectory directly
            normalized_sample = fut_traj.unsqueeze(0).repeat(args.num_samples, 1, 1).to(device)
            # Create dummy empty waypoints for analysis compatibility
            wv = torch.zeros_like(normalized_sample)
            wm = torch.zeros_like(normalized_sample, dtype=torch.bool)
        
        # B. Denormalize
        denorm_btc = dataset.denormalize_global(normalized_sample)
        future_traj_np = denorm_btc.cpu().numpy()

        # Store unreconstructed segment for debugging   
        unreconstructed_segments.append(future_traj_np)

        # Collect analysis data
        analysis_normalized_windows.append(normalized_sample.cpu().numpy())
        analysis_denormalized_windows.append(future_traj_np)
        if wv is not None and wm is not None:
            analysis_waypoint_values.append(wv.cpu().numpy())
            analysis_waypoint_masks.append(wm.cpu().numpy())

        # C. Reconstruct World Frame
        anchor_arr = np.concatenate([
            current_anchors['ref_pos'], 
            current_anchors['ref_quat'], 
            current_anchors['ref_obj_pos'],
            current_anchors['final_obj_pos']
        ], axis=-1)

        # Assuming reconstruct_sbto_trajectory returns (robot, object, ...)
        res = reconstruct_sbto_trajectory(anchor_arr, future_traj_np, inpaint=diffuser.model_cfg.get("inpaint", False))
        r_world, o_world = res[0], res[1]

        if args.action_horizon is not None:
            r_world = r_world[:, 1:args.action_horizon+1, :]
            o_world = o_world[:, 1:args.action_horizon+1, :]
        
        # Store Segment
        # Robot(36) + Object(7)
        segment_world = np.concatenate([r_world[..., :36], o_world[..., :7]], axis=-1)
        stitched_segments.append(segment_world)
    
        # Denormalize task params before transforming to global frame
        task_denorm = dataset._denormalize("task_params", task_tens) # (B, 2)
        if task_denorm.shape[1] < 3:
            task_denorm = torch.cat([task_denorm, torch.zeros_like(task_denorm[:, :1])], dim=1)
        task_denorm_3d = task_denorm[..., None] # (B, 3)
        
        goal_vec_global = (yaw_to_rot_matrix(yaw_from_quat(current_anchors['ref_quat'])) @ task_denorm_3d.cpu().numpy())[..., :data_cfg["num_task_params"], 0]
        goal_vectors.append(goal_vec_global.repeat(segment_world.shape[1], 0)[..., :2]) # (B, 3)

        # Optionally visualize this window
        visualize_every = 50
        if args.visualize_windows and step % visualize_every == 0:
            part_traj = np.concatenate(stitched_segments[-visualize_every:], axis=1) # Visualize last 5 segments
            part_goal_vec = np.concatenate(goal_vectors[-visualize_every:], axis=0)
            try:
                # Use the interactive visualizer used at the end of stitching
                run_visualization(vis, part_traj, part_goal_vec, anchor["final_obj_pos"], repeat=False)
            except Exception as e:
                print(f"Warning: interactive visualization failed for window {step}: {e}")

        # Desired Displacement (From Ground Truth Full Trajectory)                    
        err = np.linalg.norm(current_anchors["final_obj_pos"][..., :data_cfg["num_task_params"]] - segment_world[:, -1, 36 : 36 + data_cfg["num_task_params"]])
        if err < args.end_error_threshold and not args.visualize_dataset:
            print(f"Segment {step+1} successfully reached the goal (Error: {err:.4f}).")
            break

        # D. Update Condition
        if step < args.stitch_steps - 1:
            r_hist = r_world[:, -history_size:, :]
            o_hist = o_world[:, -history_size:, :]
            curr_state_tens, current_anchors = update_condition(dataset, r_hist, o_hist, final_obj_pos=current_anchors['final_obj_pos'])

            if args.visualize_dataset: 
                # Update Task Params and anchor for next window from ground truth
                next_start = current_start_time + (step + 1) * dataset.window_size
                target_key = (current_file_idx, current_batch_idx, next_start)
                if target_key in index_map:
                    next_idx = index_map[target_key]
                    fut_traj, _, next_task, next_anchor = dataset[next_idx]
                    task_tens = next_task.unsqueeze(0).repeat(args.num_samples, 1).to(device)
                    # Use ground truth anchor so reconstruction errors don't accumulate
                    current_anchors = {
                        'ref_pos': np.tile(next_anchor['ref_pos'][None], (args.num_samples, 1)),
                        'ref_quat': np.tile(next_anchor['ref_quat'][None], (args.num_samples, 1)),
                        'ref_obj_pos': np.tile(next_anchor['ref_obj_pos'][None], (args.num_samples, 1)),
                        'final_obj_pos': np.tile(next_anchor['final_obj_pos'][None], (args.num_samples, 1)),
                    }
                    print(f"Updated task params for step {step+1} to sample {next_idx} (Start T={next_start})")
                else:
                    print(f"Warning: Could not find next window starting at {next_start}. Reusing previous task params.")


    # 6. Finalize
    full_trajectory = np.concatenate(stitched_segments, axis=1) # (B, T, D)
    unreconstructed_trajectory = np.concatenate(unreconstructed_segments, axis=1) if len(unreconstructed_segments) > 0 else None
    goal_vectors = np.concatenate(goal_vectors, axis=0) # (B, task_dim)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.save(args.save_path, full_trajectory)
    print(f"Stitched trajectory saved to {args.save_path} (Shape: {full_trajectory.shape})")

    # Save analysis data
    analysis_path = args.save_path.replace('.npy', '_analysis.npz')
    analysis_data = {
        'trajectory': full_trajectory,                                          # (B, total_T, 43) world frame
        'normalized_windows': np.array(analysis_normalized_windows),            # (num_windows, B, T, D) normalized
        'denormalized_windows': np.array(analysis_denormalized_windows),        # (num_windows, B, T, D) denormalized features
        'waypoint_values': np.array(analysis_waypoint_values),                  # (num_windows, B, T, D)
        'waypoint_masks': np.array(analysis_waypoint_masks),                    # (num_windows, B, T, D) bool
        'feature_order': np.array(dataset.feature_order),                       # feature names
        'feature_dims': np.array([FEATURE_LAYOUT_NO_VEL[k] for k in dataset.feature_order]),  # dim per feature
    }
    if unreconstructed_trajectory is not None:
        analysis_data['unreconstructed'] = unreconstructed_trajectory
    np.savez(analysis_path, **analysis_data)
    print(f"Analysis data saved to {analysis_path}")
    
    # Object indices: 36:38 (pos), 38:42 (quat)
    if full_trajectory.shape[-1] >= 38:
        # Achieved Displacement
        start_obj = full_trajectory[0, 0, 36 : 36 + data_cfg["num_task_params"]]
        end_obj = full_trajectory[0, -1, 36 : 36 + data_cfg["num_task_params"]]
        achieved_displacement = (end_obj - start_obj)[..., :data_cfg['num_task_params']]
        desired_displacement = (anchor["final_obj_pos"] - anchor["ref_obj_pos"])[..., :data_cfg['num_task_params']]
        try:            
            print("-" * 30)
            print(f"Goal: Full Trajectory Displacement")
            print(f"Desired (GT) Delta XY: {desired_displacement}")
            print(f"Achieved (Gen) Delta XY: {achieved_displacement}")
            err = np.linalg.norm(desired_displacement - achieved_displacement)
            print(f"L2 Error: {err:.4f}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Could not load ground truth for comparison: {e}")
            # Fallback to previous method if needed, but it's likely wrong for windows
            pass

    # 7. Visualize
    if os.path.exists(xml_path):
        run_visualization(vis, full_trajectory, goal_vectors, anchor["final_obj_pos"])
    else:
        print("Could not find mj_model.xml for visualization.")

    vis.close()

if __name__ == "__main__":
    main()

import torch
import numpy as np
import yaml
import os
import argparse
from config.configure import load_config, get_data_path, get_norm_path, get_mj_xml_paths
from utils.data.load_dataset import preload_dataset
from datasets.flexible_dataset import yaw_to_rot_matrix, yaw_from_quat
from utils.math.sbto_utils import reconstruct_sbto_trajectory, compute_task_params, build_feature_layout

# Default lift height derived from dataset statistics:
# median peak z_delta ≈ 0.617 across 231 pick-and-place trajectories.
DEFAULT_LIFT_HEIGHT = 0.62


def compute_z_profile(progress, lift_height=DEFAULT_LIFT_HEIGHT,
                      lift_start=0.10, lift_end=0.30,
                      lower_start=0.60, lower_end=0.80):
    """
    Compute the desired z-offset above rest at a given progress fraction [0, 1].

    Profile phases (calibrated from 231 training trajectories):
        [0, lift_start)             → 0            (object at rest)
        [lift_start, lift_end)      → ramp up      (raised cosine 0 → lift_height)
        [lift_end, lower_start)     → lift_height  (carry at peak)
        [lower_start, lower_end)    → ramp down    (raised cosine lift_height → 0)
        [lower_end, 1]              → 0            (object placed)

    Data calibration:
        lift_start=0.10  — z delta is ~0 at 5% progress
        lift_end=0.40    — median peak z at 40% progress
        lower_start=0.55 — z starts descending around 55%
        lower_end=0.75   — z returns to rest by ~75%

    Args:
        progress: float in [0, 1]
        lift_height: peak z-offset in meters (data median ≈ 0.62)
        lift_start/lift_end: trajectory fraction for the lift ramp
        lower_start/lower_end: trajectory fraction for the descent ramp
    Returns:
        float: desired z-offset above rest at this progress
    """
    import math
    p = max(0.0, min(1.0, progress))

    if p < lift_start:
        return 0.0
    elif p < lift_end:
        # Smooth ramp up (raised cosine: 0 → lift_height)
        t = (p - lift_start) / (lift_end - lift_start)
        return lift_height * 0.5 * (1.0 - math.cos(math.pi * t))
    elif p < lower_start:
        return lift_height  # carry phase — hold at peak
    elif p < lower_end:
        # Smooth ramp down (raised cosine: lift_height → 0)
        t = (p - lower_start) / (lower_end - lower_start)
        return lift_height * 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        return 0.0  # placed


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


def build_first_keyframe_from_state(curr_state_tens, dataset, num_features, window_size,
                                     current_obj_z=None):
    """
    Build a waypoint tensor where only t=0 is marked (all features known),
    constructed from the current observation state.  The first
    `num_features - num_observations` dims (delta_xy, delta_yaw, obj_delta_xy, obj_z)
    are zero at t=0 (no motion yet), except obj_z which uses the actual current height.

    Args:
        curr_state_tens: (B, H, obs_dim)  — current observation (last H frames)
        dataset:         the dataset object (for normalization)
        num_features:    total feature dim (D)
        window_size:     T tokens in a window
        current_obj_z:   (B,) or scalar — absolute object z at start of this window.
                         If None, normalised 0 is used (legacy behaviour).

    Returns:
        waypoint_values: (B, T, D)
        waypoint_mask:   (B, T, D) bool — True at t=0 for all features
    """
    B = curr_state_tens.shape[0]
    obs_dim = curr_state_tens.shape[-1]
    D = num_features
    T = window_size

    # At t=0 deltas are zero; obj_z uses the actual current absolute height.
    prefix_dim = D - obs_dim
    key_dim_map = {
        'delta_xy': 2, 'delta_yaw': 1, 'obj_delta_xy': 2, 'obj_z': 1,
    }

    # Build per-batch prefix: zeros for delta features, actual z for obj_z.
    prefix_B = torch.zeros(B, prefix_dim, device=curr_state_tens.device)
    cumulative = 0
    for key in dataset.feature_order:
        if key in key_dim_map:
            kd = key_dim_map[key]
            if key == 'obj_z' and current_obj_z is not None:
                val = torch.as_tensor(current_obj_z, dtype=torch.float32,
                                      device=curr_state_tens.device).reshape(-1, 1)
                if val.shape[0] == 1:
                    val = val.expand(B, 1)
            else:
                val = torch.zeros(B, kd, device=curr_state_tens.device)
            norm_val = dataset._normalize(key, val)
            if not isinstance(norm_val, torch.Tensor):
                norm_val = torch.tensor(norm_val, dtype=torch.float32,
                                        device=curr_state_tens.device)
            prefix_B[:, cumulative:cumulative + kd] = norm_val.reshape(B, kd)
            cumulative += kd
        if cumulative >= prefix_dim:
            break

    # Build first frame: [prefix (delta + obj_z features), observation_features]
    obs_frame = curr_state_tens[:, -1, :]  # (B, obs_dim) — latest observation frame
    first_frame = torch.cat([prefix_B, obs_frame], dim=-1)  # (B, D)

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
    total_steps,         # total number of windows for the task (used for z-profile progress)
    remaining_steps,     # how many windows remain to reach goal
    arrival_ratio=0.70,  # unused for XY; kept for API compatibility and z-profile
    lift_height=DEFAULT_LIFT_HEIGHT,  # peak z-offset for pick-and-place (data: 0.62m)
    no_lower_dist=0.75,  # lower z when remaining XY dist < this (data: lowering starts ~55%)
    lift_start=0.10,     # data: z delta ~0 at 5%, starts rising by ~10%
    lift_end=0.40,       # data: median peak z at 40% progress
    walk_start_z=0.25,   # data: XY starts (>5%) at ~22% progress, z ≈ 25% of peak there
    rest_obj_z=None,     # (B,) or scalar: absolute object z at episode start (rest height)
):
    """
    Build a partial waypoint at the last frame (t=T-1) of the window.

    XY waypoint: per-window displacement = total_remaining_delta / remaining_steps.
    This is purely local — it only depends on how far the object still needs to travel
    and how many windows are left, not on the global trapezoidal profile.

    Z waypoint: absolute height target derived from the global z-profile (still uses
    total_steps / current_step to place the lift/lower ramps correctly).
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
        direction = task_params_raw[:, :2]   
        distance = task_params_raw[:, 2:3]   
        total_remaining_delta = direction * distance  
    else:
        total_remaining_delta = task_params_raw[:, :2]  

    # --- PER-WINDOW XY DISPLACEMENT ---
    # Divide the remaining displacement evenly across the remaining windows.
    # This keeps the waypoint local: it depends only on the current remaining
    # distance and the number of windows left, not on global trajectory progress.
    current_step = total_steps - remaining_steps
    step_fraction = 1.0 / max(remaining_steps, 1)
    per_window_delta = total_remaining_delta * step_fraction  # (B, 2)

    # --- Z-HEIGHT GATE ON XY MOTION ---
    # Don't let the robot walk until the box is sufficiently lifted.
    if walk_start_z > 0 and lift_height > 0:
        progress_now = current_step / max(total_steps, 1)
        z_now = compute_z_profile(progress_now, lift_height=lift_height,
                                   lift_start=lift_start, lift_end=lift_end,)  # lifting only
        min_walk_z = walk_start_z * lift_height
        if z_now < min_walk_z:
            per_window_delta = torch.zeros_like(per_window_delta)
            # print(f"  [xy-wp] Walk gated: z_now={z_now:.3f}m < threshold={min_walk_z:.3f}m")
    # -----------------------------------
    # ----------------------------------------------

    # Build feature index map
    feature_idx = {}
    idx = 0
    _layout = build_feature_layout()
    for key in feature_order:
        dim = _layout.get(key, 0)
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

    # Set obj_z: ABSOLUTE target height = rest_z + z_profile_offset.
    # Using an absolute constraint eliminates the "delta trap" where tiny
    # unmasked-frame drifts accumulate across windows and cause early dropping.
    if "obj_z" in feature_idx and lift_height > 0:
        import math as _math

        # Time-based lifting profile (offset above rest: 0 → lift_height)
        progress_end = (current_step + 1) / max(total_steps, 1)
        z_lift_end   = compute_z_profile(progress_end, lift_height=lift_height,
                                          lift_start=lift_start, lift_end=lift_end)

        # Distance-based lowering: hold lift_height while far; cosine ramp to 0 near goal.
        remaining_dist_m  = torch.norm(total_remaining_delta, dim=-1)   # (B,)
        per_window_xy_mag = torch.norm(per_window_delta, dim=-1)        # (B,)
        remaining_end_m   = torch.clamp(remaining_dist_m - per_window_xy_mag, min=0.0)

        def _z_from_dist(rem):
            frac = torch.clamp(rem / max(no_lower_dist, 1e-6), 0.0, 1.0)
            return lift_height * 0.5 * (1.0 - torch.cos(_math.pi * frac))

        # Gate: don't enter lowering zone early via XY lookahead
        still_far  = remaining_dist_m >= no_lower_dist
        z_dist_end = torch.where(
            still_far,
            torch.full_like(remaining_dist_m, lift_height),
            _z_from_dist(remaining_end_m),
        )

        # Combined height offset = min(lift profile, distance-based lowering)
        z_offset_end = torch.minimum(torch.full((B,), z_lift_end), z_dist_end)  # (B,)

        # Absolute target z = episode-start rest height + height offset
        if rest_obj_z is not None:
            rest_t = torch.as_tensor(rest_obj_z, dtype=torch.float32)
            if rest_t.ndim == 0:
                rest_t = rest_t.expand(B)
            elif rest_t.shape[0] == 1 and B > 1:
                rest_t = rest_t.expand(B)
        else:
            rest_t = torch.zeros(B)
        target_abs_z = rest_t + z_offset_end  # (B,)

        target_tensor = target_abs_z.unsqueeze(-1)  # (B, 1)
        norm_z = dataset._normalize("obj_z", target_tensor)
        if not isinstance(norm_z, torch.Tensor):
            norm_z = torch.tensor(norm_z, dtype=torch.float32)
        values[:, t_last, feature_idx["obj_z"]] = norm_z.reshape(B, -1)
        mask[:, t_last, feature_idx["obj_z"]] = True

        avg_rem    = remaining_dist_m.mean().item()
        avg_z      = target_abs_z.mean().item()
        n_lowering = (remaining_dist_m < no_lower_dist).sum().item()
        # print(f"  [z-wp] target_z={avg_z:.3f}m  rem={avg_rem:.3f}m  "
        #       f"lowering={n_lowering}/{B}")

    return values, mask


def build_inference_waypoints(
    curr_state_tens,       # (B, H, obs_dim)
    task_params_raw,       # (B, num_task_params) denormalized task params
    dataset,
    num_features,
    window_size,
    use_last_frame_wp=True,  # whether to add last-frame partial waypoint
    stitch_steps=1,
    remaining_steps=1,
    arrival_ratio=0.70,
    lift_height=DEFAULT_LIFT_HEIGHT,  # peak z-offset (data: median 0.62m)
    no_lower_dist=0.75,              # lower z when remaining XY dist < this (metres)
    lift_start=0.10,                 # z profile: lift start (data: ~10%)
    lift_end=0.40,                   # z profile: lift peak (data: ~40%)
    walk_start_z=0.25,               # gate XY walk until z >= this fraction of lift_height
    current_obj_z=None,              # (B,) absolute object z at t=0 of this window (for t=0 keyframe)
    rest_obj_z=None,                 # (B,) or scalar: episode-start rest z (for absolute z target)
):
    """
    Build complete waypoint specification for inference.
    Combines a full keyframe at t=0 (from current state) with an optional
    analytically-computed partial waypoint at the last frame (obj_delta_xy + obj_z).

    Returns:
        waypoint_values: (B, T, D) float tensor
        waypoint_mask:   (B, T, D) bool tensor
    """
    B = curr_state_tens.shape[0]
    D = num_features
    T = window_size

    # 1. Full keyframe at t=0 from current state (anchors obj_z to actual current height)
    kf_vals, _ = build_first_keyframe_from_state(
        curr_state_tens, dataset, num_features, window_size,
        current_obj_z=current_obj_z,
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
            total_steps=stitch_steps,
            remaining_steps=remaining_steps,
            arrival_ratio=arrival_ratio,
            lift_height=lift_height,
            no_lower_dist=no_lower_dist,
            lift_start=lift_start,
            lift_end=lift_end,
            walk_start_z=walk_start_z,
            rest_obj_z=rest_obj_z,
        )
        # Merge: last-frame waypoint fills in only its marked positions
        values = torch.where(wp_mask.to(values.device), wp_vals.to(values.device), values)
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
    parser.add_argument("--analysis_path", type=str, default=None,
                        help="If provided, save a *_analysis.npz with per-window data for analyze_trajectory.py. "
                             "Supports --mode normalized / denormalized / waypoints / world / all.")
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
    parser.add_argument("--last_frame_waypoint", action="store_true", help="Add partial waypoint at last frame (obj_delta_xy + obj_z absolute)")
    parser.add_argument("--arrival_ratio", type=float, default=0.70, help="Object arrives in this fraction of total time (0-1; data: 90%% XY by 65%%)")
    parser.add_argument("--lift_height", type=float, default=DEFAULT_LIFT_HEIGHT,
                        help=f"Peak lift height in meters for pick-and-place z profile (default: {DEFAULT_LIFT_HEIGHT}m from dataset)")
    parser.add_argument("--lift_start", type=float, default=0.10, help="Fraction of trajectory where lift begins (data: ~10%%)")
    parser.add_argument("--lift_end", type=float, default=0.40, help="Fraction of trajectory where lift reaches peak (data: ~40%%)")
    parser.add_argument("--walk_start_z", type=float, default=0.25,
                        help="Gate XY motion: don't walk until z >= this fraction of lift_height (data: ~25%%)")
    parser.add_argument("--no_lower_dist", type=float, default=0.75,
                        help="Lower z when remaining XY distance drops below this value in metres (default: 0.75m)")
    
    args = parser.parse_args()

    # 1. Setup MotionGenerator (handles config, dataset, and model internally)
    from motion_generator import MotionGenerator

    config_path = "config/config.yaml"
    with open(config_path, 'r') as file:
        raw_config = yaml.safe_load(file)
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config(config_path, raw_config.get("auto_conf", False))
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = get_data_path(data_cfg)
    data_buffer = preload_dataset(data_cfg, data_path)

    generator = MotionGenerator(config_path=config_path, device=device)
    ckpt_path = args.epoch if os.path.exists(args.epoch) else None
    generator.fit(
        data_source=data_buffer,
        epochs=0,
        checkpoint=ckpt_path,
    )
    if ckpt_path is None:
        generator.diffuser.loadWeights(int(args.epoch))
    
    dataset = generator.dataset

    from utils.visualize.visualize import MjVisualizer
    xml_path, _ = get_mj_xml_paths()
    if not os.path.exists(xml_path):
        xml_path = os.path.join(data_path, "mj_model.xml")
    vis = MjVisualizer(xml_path, close_on_enter=False)

    # 2. Prepare Initial Condition
    if args.traj_idx is not None:
        target = (args.traj_idx, args.batch_idx, args.start_time)
        try:
            args.sample_idx = dataset.indices.index(target)
            print(f"Mapped {target} -> Sample {args.sample_idx}")
        except ValueError:
            raise ValueError(f"Target indices {target} (Trajectory {args.traj_idx}, Batch {args.batch_idx}, Start {args.start_time}) not present in valid window list.")

    print(f"Loading initial condition (Sample {args.sample_idx})...")
    current_file_idx, current_batch_idx, current_start_time = dataset.indices[args.sample_idx]

    # Extract world-frame robot+object history for generate_trajectory()
    raw_traj = dataset._get_single_traj(current_file_idx, current_batch_idx)
    H = dataset.history_size
    t_start = current_start_time

    base_hist = raw_traj['base'][t_start:t_start + H]       # (H, 7)
    joints_hist = raw_traj['joints'][t_start:t_start + H]   # (H, 29)
    obj_hist = raw_traj['obj'][t_start:t_start + H]         # (H, 7)
    robot_hist = np.concatenate([base_hist, joints_hist], axis=-1)  # (H, 36)

    initial_condition = {
        'robot': robot_hist,  # (H, 36) world frame
        'obj': obj_hist,      # (H, 7)  world frame
    }

    # Get anchor for goal computation and metrics
    _, _, _, anchor = dataset[args.sample_idx]

    # 3. Compute goal as pelvis-local displacement from initial object position
    #    This is the unified goal representation: displacement of the object
    #    from initial to desired final position, expressed in the yaw-rotated
    #    initial robot pelvis frame (matches pick_place_relative_box_pose).
    if args.task_params is not None:
        # User provides goal as pelvis-local displacement directly
        goal_condition = np.array(args.task_params, dtype=np.float64)
    else:
        # Convert dataset's world-frame goal to pelvis-local displacement
        init_quat = anchor['ref_quat']        # (4,) [qw, qx, qy, qz]
        init_obj = anchor['ref_obj_pos'][:3]   # (3,)
        final_obj = anchor['final_obj_pos'][:3] # (3,)

        yaw = yaw_from_quat(init_quat)
        R_inv = yaw_to_rot_matrix(-yaw)  # world -> pelvis-local
        world_delta = final_obj - init_obj
        local_delta = (R_inv @ world_delta[:, None])[:, 0]
        goal_condition = local_delta[:data_cfg["num_task_params"]]

    if args.goal_multiplier != 1.0:
        print(f"Scaling goal by multiplier {args.goal_multiplier}")
        goal_condition = goal_condition * args.goal_multiplier

    print(f"Goal condition (pelvis-local displacement): {goal_condition}")

    # Convert goal_condition (pelvis-local) back to world frame for visualization
    _init_obj = anchor['ref_obj_pos'][:3].copy()
    _yaw = yaw_from_quat(anchor['ref_quat'])
    _R_fwd = yaw_to_rot_matrix(_yaw)  # pelvis-local -> world
    _goal_3d = np.zeros(3)
    _goal_3d[:len(goal_condition)] = goal_condition
    goal_delta_world = (_R_fwd @ _goal_3d[:, None])[:, 0]
    goal_pos_world = _init_obj + goal_delta_world
    print(f"Goal position (world frame): {goal_pos_world}")

    # 4. Auto-compute stitch steps
    if args.stitch_steps is None:
        _eff = dataset.window_size - 1
        args.stitch_steps = (dataset.traj_lengths[current_file_idx] + _eff - 1) // _eff
        print(f"Auto-setting stitch_steps to {args.stitch_steps}")

    if args.action_horizon is not None:
        args.stitch_steps *= (data_cfg["num_timesteps"] // args.action_horizon)
        print(f"Adjusting stitch_steps to {args.stitch_steps} for action horizon")

    if args.goal_multiplier != 1.0:
        args.stitch_steps = max(1, int(args.stitch_steps * abs(args.goal_multiplier)))

    # 5. Generate trajectory
    if args.visualize_dataset:
        # Ground truth mode: reconstruct directly from raw dataset (no diffusion model)
        raw_data = dataset.ram_cache[current_file_idx]
        base_raw = raw_data['base'][current_batch_idx]       # (T_raw, 7)
        joints_raw = raw_data['joints'][current_batch_idx]   # (T_raw, 29)
        obj_raw = raw_data['obj'][current_batch_idx]         # (T_raw, 7)

        robot_raw = np.concatenate([base_raw, joints_raw], axis=-1)  # (T_raw, 36)
        full_trajectory = np.concatenate([robot_raw, obj_raw], axis=-1)  # (T_raw, 43)
        full_trajectory = full_trajectory[None]  # (1, T, 43) add batch dim

        if args.num_samples > 1:
            full_trajectory = np.repeat(full_trajectory, args.num_samples, axis=0)
    else:
        _want_analysis = args.analysis_path is not None
        _gen_result = generator.generate_trajectory(
            initial_condition=initial_condition,
            goal_condition=goal_condition,
            stitch_steps=args.stitch_steps,
            num_samples=args.num_samples,
            cfg_w=args.cfg_w,
            end_error_threshold=args.end_error_threshold,
            enable_goal_stop=True,
            enable_physics_stop=False,
            use_last_frame_wp=args.last_frame_waypoint,
            arrival_ratio=args.arrival_ratio,
            lift_height=args.lift_height,
            no_lower_dist=args.no_lower_dist,
            lift_start=args.lift_start,
            lift_end=args.lift_end,
            walk_start_z=args.walk_start_z,
            return_analysis=_want_analysis,
        )
        if _want_analysis:
            full_trajectory, _analysis_dict = _gen_result
        else:
            full_trajectory = _gen_result

    # 6. Save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.save(args.save_path, full_trajectory)
    print(f"Trajectory saved to {args.save_path} (Shape: {full_trajectory.shape})")

    if args.analysis_path and not args.visualize_dataset:
        os.makedirs(os.path.dirname(args.analysis_path) or ".", exist_ok=True)
        np.savez(
            args.analysis_path,
            normalized_windows=_analysis_dict["normalized_windows"],
            denormalized_windows=_analysis_dict["denormalized_windows"],
            waypoint_values=_analysis_dict["waypoint_values"],
            waypoint_masks=_analysis_dict["waypoint_masks"],
            feature_order=_analysis_dict["feature_order"],
            feature_dims=_analysis_dict["feature_dims"],
            trajectory=full_trajectory,
        )
        print(f"Analysis data saved to {args.analysis_path}")

    # 7. Print metrics
    if full_trajectory.shape[-1] >= 38:
        start_obj = full_trajectory[0, 0, 36:36 + data_cfg["num_task_params"]]
        end_obj = full_trajectory[0, -1, 36:36 + data_cfg["num_task_params"]]
        achieved_displacement = end_obj - start_obj
        target_displacement = goal_delta_world[:data_cfg["num_task_params"]]
        gt_displacement = (anchor["final_obj_pos"] - anchor["ref_obj_pos"])[:data_cfg["num_task_params"]]
        print("-" * 30)
        print(f"Target Delta XY (specified): {target_displacement}")
        print(f"GT Delta XY (dataset):       {gt_displacement}")
        print(f"Achieved Delta XY:           {achieved_displacement}")
        err = np.linalg.norm(target_displacement - achieved_displacement)
        print(f"L2 Error vs target: {err:.4f}")
        print("-" * 30)

    # 8. Visualize
    if os.path.exists(xml_path):
        run_visualization(vis, full_trajectory,
                          goal_vectors=goal_delta_world,
                          final_obj_pos=goal_pos_world)
    else:
        print("Could not find mj_model.xml for visualization.")

    vis.close()

if __name__ == "__main__":
    main()

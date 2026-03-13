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
from utils.math.sbto_utils import reconstruct_sbto_trajectory, compute_task_params, build_feature_layout
from utils.physics_limits import apply_physics_clamp

# Default lift height derived from dataset statistics:
# median peak cumulative obj_delta_z across 1634 pick-and-place trajectories.
DEFAULT_LIFT_HEIGHT = 0.5


def compute_z_profile(progress, lift_height=DEFAULT_LIFT_HEIGHT,
                      lift_start=0.0, lift_end=0.20,):
    """
    Compute the desired cumulative z-offset at a given progress fraction [0, 1]
    using a smooth raised-cosine bell.

    Profile phases:
        [0, lift_start)        → 0            (no lift yet)
        [lift_start, lift_end) → ramp up      (pick up)
        [lift_end, lower_start)→ lift_height  (carry at full height)
        [lower_start, lower_end)→ ramp down   (time-based lower; default disabled)
        [lower_end, 1]         → 0            (settled)

    Lowering is normally controlled by remaining XY distance in
    build_last_frame_waypoints (lower_start=1.0 default disables time-based lower).

    Args:
        progress: float in [0, 1]
        lift_height: peak z-offset in meters
        lift_start/lift_end: trajectory fraction for the lift ramp
        lower_start/lower_end: trajectory fraction for the lower ramp (default disabled)
    Returns:
        float: desired cumulative delta_z at this progress
    """
    import math
    p = max(0.0, min(1.0, progress))
    
    if p < lift_start:
        return 0.0
    elif p < lift_end:
        # Smooth ramp up (raised cosine: 0 → 1)
        t = (p - lift_start) / (lift_end - lift_start)
        return lift_height * 0.5 * (1.0 - math.cos(math.pi * t))
    else:
        return lift_height  # carry phase — hold at peak


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
    total_steps,         # The initial total number of windows/steps for this task
    remaining_steps,     # how many windows remain to reach goal
    arrival_ratio=0.85,  # finish in this fraction of total time
    t_accel=0.35,        # fraction of time spent accelerating (data: peak XY velocity at ~40%)
    t_decel=0.35,        # fraction of time spent decelerating (data: long tail, ~0 at 90%)
    lift_height=DEFAULT_LIFT_HEIGHT,  # peak z-offset for pick-and-place
    no_lower_dist=0.5,   # lower z only when remaining XY dist drops below this (metres)
    lift_start=0.0,      # fraction of trajectory where lift begins (0 = immediately)
    lift_end=0.20,       # fraction of trajectory where lift reaches peak
    walk_start_z=0.80,   # don't move XY until cumulative z >= this fraction of lift_height
    rest_obj_z=None,     # (B,) or scalar: absolute object z at episode start (rest height)
):
    """
    Build a partial waypoint at the last frame (t=T-1) of the window.

    Sequence:
      1. Lift immediately (lift_start=0 → lift_end=0.20).
      2. Walk only after z >= walk_start_z * lift_height (default 80% of peak).
      3. Lower only when remaining XY dist < no_lower_dist (default 0.5 m).
         Lowering uses a smooth raised-cosine ramp: lift_height at no_lower_dist, 0 at goal.

    XY profile (data-calibrated):
        t_accel=0.35  peak velocity at ~40% of walking phase
        t_decel=0.35  long deceleration tail
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

    # --- INLINED TRAPEZOIDAL DISPLACEMENT LOGIC ---
    effective_total_steps = max(total_steps * arrival_ratio, 1.0)
    current_step = total_steps - remaining_steps
    
    tau_start = current_step / effective_total_steps
    tau_end = (current_step + 1) / effective_total_steps

    # Fallback to triangular if accel + decel > 1.0
    if t_accel + t_decel > 1.0: 
        t_accel, t_decel = 0.5, 0.5
        
    # Precompute curve constants
    v_max = 1.0 / (1.0 - 0.5 * t_accel - 0.5 * t_decel)
    p_ta = 0.5 * v_max * t_accel
    p_td_start = p_ta + v_max * (1.0 - t_accel - t_decel)
    
    # Evaluate progress at tau_start and tau_end
    progress = []
    for tau in (tau_start, tau_end):
        tau = max(0.0, min(1.0, tau))
        if tau <= t_accel:
            # Acceleration phase
            s = 0.5 * (v_max / t_accel) * (tau ** 2) if t_accel > 0 else 0.0
        elif tau <= 1.0 - t_decel:
            # Cruise phase
            s = p_ta + v_max * (tau - t_accel)
        else:
            # Deceleration phase
            tau_prime = tau - (1.0 - t_decel)
            if t_decel > 0:
                s = p_td_start + v_max * tau_prime - 0.5 * (v_max / t_decel) * (tau_prime ** 2)
            else:
                s = p_td_start + v_max * tau_prime
        progress.append(s)

    s_start, s_end = progress
    
    # What fraction of the *remaining* distance should we cover in this window?
    if s_start >= 1.0:
        step_fraction = 0.0 # Already arrived
    else:
        step_fraction = (s_end - s_start) / (1.0 - s_start)
        
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
    arrival_ratio=0.85,
    lift_height=DEFAULT_LIFT_HEIGHT,  # peak z-offset for pick-and-place profile
    no_lower_dist=0.5,               # lower z only when remaining XY dist < this (metres)
    lift_start=0.0,                  # z profile: lift start (fraction of total trajectory)
    lift_end=0.20,                   # z profile: lift end
    walk_start_z=0.80,               # gate XY walk until z >= this fraction of lift_height
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
    parser.add_argument("--ema", action="store_true", help="Whether to use EMA weights")
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
    parser.add_argument("--end_error_threshold", type=float, default=0.1,
                        help="XY-plane radius (m) for goal-reached check")
    parser.add_argument("--end_ground_num_frames", type=int, default=5,
                        help="Consecutive on-ground frames required before stopping")
    parser.add_argument("--end_ground_z_tol", type=float, default=0.05,
                        help="Z tolerance (m) for object to be considered on the ground")
    parser.add_argument("--goal_multiplier", type=float, default=1.0, help="Scaling factor for goal (for testing different r for same theta)")
    parser.add_argument("--visualize_windows", action="store_true", help="Render and save each generated window as a video")
    parser.add_argument("--enable_phys_stop", action="store_true", help="Stop trajectory when object reaches goal region (for pick-and-place)")
    parser.add_argument("--enable_phys_clamp", action="store_true", help="Apply data-derived physics clamping (joint pos/vel, body_z, XY vel)")
    parser.add_argument("--verbose_physics", action="store_true", help="Print physics clamping statistics per window")

    # Guidance arguments
    parser.add_argument("--guidance_wt", type=float, default=0.0, help="Test-time gradient guidance strength")
    parser.add_argument("--guidance_goal", nargs="+", type=float, default=None, help="Target values for guidance (normalized)")
    parser.add_argument("--guidance_indices", nargs="+", type=int, default=None, help="Indices of the state vector to apply guidance on")
    parser.add_argument("--last_frame_waypoint", action="store_true", help="Add partial waypoint at last frame (obj_delta_xy + obj_z absolute)")
    parser.add_argument("--arrival_ratio", type=float, default=0.85, help="Object arrives in this fraction of remaining time (0-1)")
    parser.add_argument("--lift_height", type=float, default=DEFAULT_LIFT_HEIGHT,
                        help=f"Peak lift height in meters for pick-and-place z profile (default: {DEFAULT_LIFT_HEIGHT}m from dataset)")
    parser.add_argument("--lift_start", type=float, default=0.0, help="Fraction of trajectory where lift begins (0=immediately)")
    parser.add_argument("--lift_end", type=float, default=0.20, help="Fraction of trajectory where lift reaches peak")
    parser.add_argument("--walk_start_z", type=float, default=0.80,
                        help="Gate XY motion: don't walk until z >= this fraction of lift_height (default: 0.80)")
    parser.add_argument("--no_lower_dist", type=float, default=0.4,
                        help="Lower z when remaining XY distance drops below this value in metres (default: 0.3m)")
    
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
        diffuser.loadWeights(int(args.epoch), ema=args.ema)
        diffuser.loadWeights(int(args.epoch), ema=args.ema)

    # Optional attention explainability capture (DiT backbone only)
    backbone = None
    if hasattr(diffuser.model, "diffusion_model") and hasattr(diffuser.model.diffusion_model, "model"):
        backbone = diffuser.model.diffusion_model.model
  
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

    curr_state_tens = curr_state.unsqueeze(0).repeat(args.num_samples, 1, 1)
    task_tens = task_params.repeat(args.num_samples, 1)

    if args.stitch_steps is None:
        _eff = data_cfg["num_timesteps"] - 1  # t=0 always skipped; each window yields (T-1) frames
        args.stitch_steps = (dataset.traj_lengths[args.traj_idx] + _eff - 1) // _eff
        print(f"Auto-setting stitch_steps to {args.stitch_steps} based on dataset length.")

    if args.action_horizon is not None:
        args.stitch_steps *= (data_cfg["num_timesteps"] // args.action_horizon)
        print(f"Adjusting stitch_steps to {args.stitch_steps} based on action horizon")

    if args.goal_multiplier != 1.0:
        print(f"Scaling goal by multiplier {args.goal_multiplier}")
        anchor["final_obj_pos"][..., :2] = anchor["ref_obj_pos"][..., :2] + args.goal_multiplier * (anchor["final_obj_pos"] - anchor["ref_obj_pos"])[..., :2]
        args.stitch_steps = max(1, int(args.stitch_steps * abs(args.goal_multiplier)))

    print(f"Using final box pos: {anchor['final_obj_pos']} for computing task parameters.")

    current_anchors = {
        'ref_pos': np.tile(anchor['ref_pos'][None], (args.num_samples, 1)),
        'ref_quat': np.tile(anchor['ref_quat'][None], (args.num_samples, 1)),
        'ref_obj_pos': np.tile(anchor['ref_obj_pos'][None], (args.num_samples, 1)),
        'final_obj_pos': np.tile(anchor['final_obj_pos'][None], (args.num_samples, 1)),
    }

    # Episode-start object z — used as absolute baseline for the z-waypoint target.
    _rest_obj_z = float(anchor['ref_obj_pos'][2])

    # Physical-consistency sanity check (task-agnostic).
    # Flags clearly unphysical frames; halts and pads on first violation.
    _FLOOR_Z        = -0.1   # robot/object z below this → ground penetration
    _MAX_ROBOT_STEP = 0.25    # max robot pelvis XY step per frame at 100 Hz (m)
    _MAX_OBJ_STEP   = 0.25    # max object XY step per frame at 100 Hz (m)
    _prev_robot_xyz = current_anchors['ref_pos'][:, :3].copy()    # (B, 3)
    _prev_obj_xyz   = current_anchors['ref_obj_pos'][:, :3].copy() # (B, 3)
    _ground_counter = 0  # consecutive on-ground frames (scalar, single-batch)

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

            task_params = dataset._normalize("task_params", tp_init)
            task_tens = task_params if isinstance(task_params, torch.Tensor) else torch.tensor(task_params, dtype=torch.float32)

            # Build waypoints: full keyframe at t=0 + optional last-frame partial waypoint
            wv, wm = None, None
            if diffuser.model.inbetweening_enabled:
                remaining = max(args.stitch_steps - step, 1)
                _current_obj_z = torch.from_numpy(
                    current_anchors['ref_obj_pos'][:, 2].astype(np.float32)
                )
                wv, wm = build_inference_waypoints(
                    curr_state_tens, task_actual, dataset,
                    data_cfg['num_features'], dataset.window_size,
                    use_last_frame_wp=args.last_frame_waypoint,
                    stitch_steps=args.stitch_steps,
                    remaining_steps=remaining,
                    arrival_ratio=args.arrival_ratio,
                    lift_height=args.lift_height,
                    no_lower_dist=args.no_lower_dist,
                    lift_start=args.lift_start,
                    lift_end=args.lift_end,
                    walk_start_z=args.walk_start_z,
                    current_obj_z=_current_obj_z,
                    rest_obj_z=_rest_obj_z,
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

        # B2. Physics-informed clamping (joint pos/vel, body_z, XY vel)
        if args.enable_phys_clamp:
            dt_eff = data_cfg.get("raw_dt", 0.01) * data_cfg.get("stride", 2)
            future_traj_np = apply_physics_clamp(
                future_traj_np, dt=dt_eff, verbose=args.verbose_physics,
            )

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

        r_world = r_world[:, history_size:, :]
        o_world = o_world[:, history_size:, :]

        if args.action_horizon is not None:
            r_world = r_world[:, :args.action_horizon, :]
            o_world = o_world[:, :args.action_horizon, :]
        
        # Store Segment — Robot(36) + Object(7)
        segment_world = np.concatenate([r_world[..., :36], o_world[..., :7]], axis=-1)

        # ── Physical-consistency check ─────────────────────────────────────────
        # Scan every frame for ground penetration or position spikes.  On the
        # first violating frame: clamp the window there, pad all remaining
        # stitch windows with that static frame, and halt generation.
        _rxy   = segment_world[:, :, :2]      # robot pelvis XY  (B, T, 2)
        _rz    = segment_world[:, :, 2]       # robot pelvis Z   (B, T)
        _oxy   = segment_world[:, :, 36:38]   # object XY        (B, T, 2)
        _oz    = segment_world[:, :, 38]      # object Z         (B, T)
        _rxy_p = np.concatenate([_prev_robot_xyz[:, None, :2], _rxy[:, :-1, :]], axis=1)
        _oxy_p = np.concatenate([_prev_obj_xyz[:, None, :2],   _oxy[:, :-1, :]], axis=1)
        _floor   = (_rz < _FLOOR_Z) | (_oz < _FLOOR_Z)
        _rspike  = np.linalg.norm(_rxy - _rxy_p, axis=-1) > _MAX_ROBOT_STEP
        _ospike  = np.linalg.norm(_oxy - _oxy_p, axis=-1) > _MAX_OBJ_STEP
        _bad_t   = np.where((_floor | _rspike | _ospike).any(axis=0))[0]
        _phys_stop = args.enable_phys_stop and _bad_t.size > 0
        if _phys_stop:
            _t_bad   = int(_bad_t[0])
            _t_clamp = max(_t_bad, 1)
            _last_ok = segment_world[:, _t_clamp - 1 : _t_clamp, :]
            segment_world = segment_world.copy()
            segment_world[:, _t_clamp:, :] = _last_ok
            print(f"[physics] step {step+1}: violation at frame {_t_bad} "
                  f"(floor={bool(_floor.any(0)[_t_bad])}, "
                  f"robot_spike={bool(_rspike.any(0)[_t_bad])}, "
                  f"obj_spike={bool(_ospike.any(0)[_t_bad])}). Truncating and halting.")
        # ──────────────────────────────────────────────────────────────────────

        stitched_segments.append(segment_world)

        # Denormalize task params before transforming to global frame
        task_denorm = dataset._denormalize("task_params", task_tens) # (B, 2)
        if task_denorm.shape[1] < 3:
            task_denorm = torch.cat([task_denorm, torch.zeros_like(task_denorm[:, :1])], dim=1)
        task_denorm_3d = task_denorm[..., None] # (B, 3)
        
        goal_vec_global = (yaw_to_rot_matrix(yaw_from_quat(current_anchors['ref_quat'])) @ task_denorm_3d.cpu().numpy())[..., :data_cfg["num_task_params"], 0]
        goal_vectors.append(goal_vec_global.repeat(segment_world.shape[1], 0)[..., :2]) # (B, 3)

        # Update per-window position trackers for spike detection at next boundary
        _prev_robot_xyz = segment_world[:, -1, :3].copy()
        _prev_obj_xyz   = segment_world[:, -1, 36:39].copy()

        # If physics check fired: pad all remaining windows and stop generation
        if _phys_stop:
            _steps_to_pad = args.stitch_steps - step - 1
            if _steps_to_pad > 0:
                _last_frame = segment_world[:, -1:, :]
                _seg_T      = segment_world.shape[1]
                _pad_seg    = np.repeat(_last_frame, _seg_T, axis=1)
                _last_gv    = goal_vectors[-1]
                for _ in range(_steps_to_pad):
                    stitched_segments.append(_pad_seg)
                    goal_vectors.append(_last_gv)
            break

        # Optionally visualize this window
        visualize_every = 50
        if args.visualize_windows and step % visualize_every == 0:
            part_traj = np.concatenate(stitched_segments[-visualize_every:], axis=1)
            part_goal_vec = np.concatenate(goal_vectors[-visualize_every:], axis=0)
            try:
                run_visualization(vis, part_traj, part_goal_vec, anchor["final_obj_pos"], repeat=False)
            except Exception as e:
                print(f"Warning: interactive visualization failed for window {step}: {e}")

        # Goal-reached check: XY within circle + object on ground for N frames
        if not args.visualize_dataset:
            _seg_T = segment_world.shape[1]
            _goal_triggered = False
            for _t in range(_seg_T):
                _obj_z_t = segment_world[0, _t, 38]  # object z (single batch)
                _on_ground = abs(_obj_z_t - _rest_obj_z) < args.end_ground_z_tol
                _ground_counter = (_ground_counter + 1) if _on_ground else 0
                _xy_err = np.linalg.norm(
                    segment_world[0, _t, 36:38] - current_anchors["final_obj_pos"][0, :2]
                )
                if _xy_err < args.end_error_threshold and _ground_counter >= args.end_ground_num_frames:
                    print(f"Segment {step+1}, frame {_t}: reached goal "
                          f"(xy_err={_xy_err:.4f}, ground_frames={_ground_counter})")
                    _goal_triggered = True
                    break
            if _goal_triggered:
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
        'feature_dims': np.array([build_feature_layout()[k] for k in dataset.feature_order]),  # dim per feature
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

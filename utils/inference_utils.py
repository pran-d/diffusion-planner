import numpy as np
import torch

from utils.math.sbto_utils import build_feature_layout


# Default lift height derived from dataset statistics:
# median peak cumulative obj_delta_z across 1634 pick-and-place trajectories.
DEFAULT_LIFT_HEIGHT = 0.5


def compute_z_profile(progress, lift_height=DEFAULT_LIFT_HEIGHT,
                      lift_start=0.0, lift_end=0.20):
    """
    Compute desired cumulative z-offset at progress fraction [0, 1].
    """
    import math
    p = max(0.0, min(1.0, progress))

    if p < lift_start:
        return 0.0
    elif p < lift_end:
        t = (p - lift_start) / (lift_end - lift_start)
        return lift_height * 0.5 * (1.0 - math.cos(math.pi * t))
    else:
        return lift_height


def build_first_keyframe_from_state(curr_state_tens, dataset, num_features, window_size,
                                    current_obj_z=None):
    """
    Build waypoint tensor with only t=0 marked from current observation state.
    """
    B = curr_state_tens.shape[0]
    obs_dim = curr_state_tens.shape[-1]
    D = num_features
    T = window_size

    prefix_dim = D - obs_dim
    key_dim_map = {
        'delta_xy': 2, 'delta_yaw': 1, 'obj_delta_xy': 2, 'obj_z': 1,
    }

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

    obs_frame = curr_state_tens[:, -1, :]
    first_frame = torch.cat([prefix_B, obs_frame], dim=-1)

    waypoint_values = torch.zeros(B, T, D, device=curr_state_tens.device)
    waypoint_values[:, 0, :] = first_frame

    waypoint_mask = torch.zeros(B, T, D, dtype=torch.bool, device=curr_state_tens.device)
    waypoint_mask[:, 0, :] = True

    return waypoint_values, waypoint_mask


def build_last_frame_waypoints(
    task_params_raw,
    window_size,
    dataset,
    feature_order,
    num_features,
    total_steps,
    remaining_steps,
    arrival_ratio=0.85,
    t_accel=0.35,
    t_decel=0.35,
    lift_height=DEFAULT_LIFT_HEIGHT,
    no_lower_dist=0.5,
    lift_start=0.0,
    lift_end=0.20,
    walk_start_z=0.80,
    rest_obj_z=None,
    goal_obj_z=None,
    normalize_goal_vec=True,
):
    """
    Build a partial waypoint at the last frame (obj_delta_xy + obj_z).
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

    if normalize_goal_vec and task_params_raw.shape[-1] >= 4:
        # New format: [dir_x, dir_y, dir_z, magnitude]
        direction = task_params_raw[:, :3]
        distance = task_params_raw[:, 3:4]
        total_remaining_delta = direction * distance
    elif normalize_goal_vec and task_params_raw.shape[-1] >= 3:
        # Legacy 2D format: [dir_x, dir_y, magnitude]
        direction = task_params_raw[:, :2]
        distance = task_params_raw[:, 2:3]
        total_remaining_delta = direction * distance
    else:
        total_remaining_delta = task_params_raw[:, :2]

    effective_total_steps = max(total_steps * arrival_ratio, 1.0)
    current_step = total_steps - remaining_steps

    tau_start = current_step / effective_total_steps
    tau_end = (current_step + 1) / effective_total_steps

    if t_accel + t_decel > 1.0:
        t_accel, t_decel = 0.5, 0.5

    v_max = 1.0 / (1.0 - 0.5 * t_accel - 0.5 * t_decel)
    p_ta = 0.5 * v_max * t_accel
    p_td_start = p_ta + v_max * (1.0 - t_accel - t_decel)

    progress = []
    for tau in (tau_start, tau_end):
        tau = max(0.0, min(1.0, tau))
        if tau <= t_accel:
            s = 0.5 * (v_max / t_accel) * (tau ** 2) if t_accel > 0 else 0.0
        elif tau <= 1.0 - t_decel:
            s = p_ta + v_max * (tau - t_accel)
        else:
            tau_prime = tau - (1.0 - t_decel)
            if t_decel > 0:
                s = p_td_start + v_max * tau_prime - 0.5 * (v_max / t_decel) * (tau_prime ** 2)
            else:
                s = p_td_start + v_max * tau_prime
        progress.append(s)

    s_start, s_end = progress

    if s_start >= 1.0:
        step_fraction = 0.0
    else:
        step_fraction = (s_end - s_start) / (1.0 - s_start)

    per_window_delta = total_remaining_delta * step_fraction

    if walk_start_z > 0 and lift_height > 0:
        progress_now = current_step / max(total_steps, 1)
        z_now = compute_z_profile(
            progress_now,
            lift_height=lift_height,
            lift_start=lift_start,
            lift_end=lift_end,
        )
        min_walk_z = walk_start_z * lift_height
        if z_now < min_walk_z:
            per_window_delta = torch.zeros_like(per_window_delta)

    feature_idx = {}
    idx = 0
    _layout = build_feature_layout()
    for key in feature_order:
        dim = _layout.get(key, 0)
        if dim > 0:
            feature_idx[key] = slice(idx, idx + dim)
            idx += dim

    t_last = T - 1

    if "obj_delta_xy" in feature_idx:
        per_window_delta_xy = per_window_delta[:, :2]
        norm_obj_delta = dataset._normalize("obj_delta_xy", per_window_delta_xy)
        if not isinstance(norm_obj_delta, torch.Tensor):
            norm_obj_delta = torch.tensor(norm_obj_delta, dtype=torch.float32)
        values[:, t_last, feature_idx["obj_delta_xy"]] = norm_obj_delta.reshape(B, -1)
        mask[:, t_last, feature_idx["obj_delta_xy"]] = True

    if "obj_z" in feature_idx and lift_height > 0:
        import math as _math

        progress_end = (current_step + 1) / max(total_steps, 1)
        z_lift_end = compute_z_profile(
            progress_end,
            lift_height=lift_height,
            lift_start=lift_start,
            lift_end=lift_end,
        )

        remaining_dist_m = torch.norm(total_remaining_delta, dim=-1)
        per_window_xy_mag = torch.norm(per_window_delta, dim=-1)
        remaining_end_m = torch.clamp(remaining_dist_m - per_window_xy_mag, min=0.0)

        def _z_from_dist(rem):
            frac = torch.clamp(rem / max(no_lower_dist, 1e-6), 0.0, 1.0)
            return lift_height * 0.5 * (1.0 - torch.cos(_math.pi * frac))

        still_far = remaining_dist_m >= no_lower_dist
        z_dist_end = torch.where(
            still_far,
            torch.full_like(remaining_dist_m, lift_height),
            _z_from_dist(remaining_end_m),
        )

        z_offset_end = torch.minimum(torch.full((B,), z_lift_end), z_dist_end)

        if goal_obj_z is not None:
            goal_t = torch.as_tensor(goal_obj_z, dtype=torch.float32)
            if goal_t.ndim == 0:
                goal_t = goal_t.expand(B)
            elif goal_t.shape[0] == 1 and B > 1:
                goal_t = goal_t.expand(B)
            base_z = goal_t
        elif rest_obj_z is not None:
            rest_t = torch.as_tensor(rest_obj_z, dtype=torch.float32)
            if rest_t.ndim == 0:
                rest_t = rest_t.expand(B)
            elif rest_t.shape[0] == 1 and B > 1:
                rest_t = rest_t.expand(B)
            base_z = rest_t
        else:
            base_z = torch.zeros(B)
        target_abs_z = base_z + z_offset_end

        target_tensor = target_abs_z.unsqueeze(-1)
        norm_z = dataset._normalize("obj_z", target_tensor)
        if not isinstance(norm_z, torch.Tensor):
            norm_z = torch.tensor(norm_z, dtype=torch.float32)
        values[:, t_last, feature_idx["obj_z"]] = norm_z.reshape(B, -1)
        mask[:, t_last, feature_idx["obj_z"]] = True

    return values, mask


def build_inference_waypoints(
    curr_state_tens,
    task_params_raw,
    dataset,
    num_features,
    window_size,
    use_last_frame_wp=True,
    stitch_steps=1,
    remaining_steps=1,
    arrival_ratio=0.85,
    lift_height=DEFAULT_LIFT_HEIGHT,
    no_lower_dist=0.5,
    lift_start=0.0,
    lift_end=0.20,
    walk_start_z=0.80,
    current_obj_z=None,
    rest_obj_z=None,
    goal_obj_z=None,
    normalize_goal_vec=True,
):
    """
    Build complete waypoint specification for inference.
    """
    B = curr_state_tens.shape[0]
    D = num_features
    T = window_size

    kf_vals, _ = build_first_keyframe_from_state(
        curr_state_tens, dataset, num_features, window_size,
        current_obj_z=current_obj_z,
    )

    values = kf_vals.clone()
    mask = torch.zeros(B, T, D, dtype=torch.bool, device=curr_state_tens.device)
    mask[:, 0, :] = True

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
            goal_obj_z=goal_obj_z,
            normalize_goal_vec=normalize_goal_vec,
        )
        values = torch.where(wp_mask.to(values.device), wp_vals.to(values.device), values)
        mask = mask | wp_mask.to(mask.device)

    return values, mask


def run_visualization(vis, stitched_trajs, goal_vectors=None, final_obj_pos=None,
                      repeat=True, save_video_path=None):
    """
    Utility for quick trajectory playback and optional video export.
    """
    if stitched_trajs.ndim == 3:
        traj = stitched_trajs[0]
    else:
        traj = stitched_trajs

    t_steps = traj.shape[0]
    t = np.arange(t_steps) * 0.01

    if save_video_path:
        vis.render_trajectory_to_video(t=t, x_traj=traj, save_path=save_video_path)
    else:
        vis.visualize_trajectory(
            t=t,
            x_traj=traj,
            repeat=repeat,
            guidance_vec=goal_vectors,
            goal_pos=final_obj_pos,
        )

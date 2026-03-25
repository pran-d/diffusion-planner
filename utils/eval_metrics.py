from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_metrics(
    full_trajectory: np.ndarray,
    goal_world: np.ndarray,
    ref_obj_pos: np.ndarray,
    real_length: Optional[int] = None,
    inference_time_s: Optional[float] = None,
    gpu_peak_mb: Optional[float] = None,
) -> Dict[str, float]:
    """Compute standard rollout metrics for one generated trajectory.

    Args:
        full_trajectory: (T, 43) [robot(36), object(7)] in world frame.
        goal_world: (3,) target object world position.
        ref_obj_pos: (3,) initial object world position.
        real_length: Optional number of valid frames before pad.
        inference_time_s: Optional wall-clock inference time.
        gpu_peak_mb: Optional peak GPU memory in MB.
    """
    traj = np.asarray(full_trajectory)
    goal_world = np.asarray(goal_world)
    ref_obj_pos = np.asarray(ref_obj_pos)

    if traj.ndim != 2 or traj.shape[-1] < 39:
        raise ValueError("full_trajectory must have shape (T, D>=39)")

    T = traj.shape[0]
    rl = int(real_length) if real_length is not None else T
    rl = int(np.clip(rl, 1, T))

    obj_xy = traj[:rl, 36:38]
    obj_z = traj[:rl, 38]
    robot_xy = traj[:rl, 0:2]
    robot_z = traj[:rl, 2]

    final_xy = obj_xy[-1]
    goal_xy = goal_world[:2]
    l2_final_error = float(np.linalg.norm(final_xy - goal_xy))

    dd = robot_xy[2:] - 2.0 * robot_xy[1:-1] + robot_xy[:-2]
    traj_smoothness = float(np.linalg.norm(dd, axis=-1).mean()) if dd.shape[0] > 0 else 0.0

    floor_violations = int(np.sum((robot_z < -0.05) | (obj_z < -0.05)))

    metrics = {
        "l2_final_error": l2_final_error,
        "success_at_0.05m": float(l2_final_error <= 0.05),
        "success_at_0.10m": float(l2_final_error <= 0.10),
        "success_at_0.15m": float(l2_final_error <= 0.15),
        "traj_smoothness": traj_smoothness,
        "obj_lift_peak_z": float(np.max(obj_z)),
        "obj_lift_peak_from_start": float(np.max(obj_z) - ref_obj_pos[2]),
        "floor_violations": floor_violations,
    }

    if inference_time_s is not None:
        metrics["inference_time_s"] = float(inference_time_s)
    if gpu_peak_mb is not None:
        metrics["gpu_peak_mb"] = float(gpu_peak_mb)

    return metrics

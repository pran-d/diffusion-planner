"""
Physics-informed post-generation clamping for diffusion planner outputs.

All limits are derived from training data (231 trajectories, stride=2, dt_eff=0.02s)
and the MuJoCo XML model (mj_model.xml).

Usage:
    from utils.physics_limits import apply_physics_clamp

    # After denormalization, before reconstruction:
    future_traj_np = apply_physics_clamp(future_traj_np, dt=0.02)
"""

import numpy as np
from utils.math.sbto_utils import build_feature_layout, get_feature_indices

# ═══════════════════════════════════════════════════════════════════════════════
# Constants — Joint ordering (matches config/feature_labels.yml indices 7–35)
# ═══════════════════════════════════════════════════════════════════════════════
JOINT_NAMES = [
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",       "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",      "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Joint position limits (from mj_model.xml)
# ═══════════════════════════════════════════════════════════════════════════════
JOINT_POS_LO = np.array([
    -2.5307, -0.5236, -2.7576, -0.0873, -0.8727, -0.2618,  # left leg
    -2.5307, -2.9671, -2.7576, -0.0873, -0.8727, -0.2618,  # right leg
    -2.6180, -0.5200, -0.5200,                              # waist
    -3.0892, -1.5882, -2.6180, -1.0472, -1.9722, -1.6144, -1.6144,  # left arm
    -3.0892, -2.2515, -2.6180, -1.0472, -1.9722, -1.6144, -1.6144,  # right arm
], dtype=np.float64)

JOINT_POS_HI = np.array([
    +2.8798, +2.9671, +2.7576, +2.8798, +0.5236, +0.2618,  # left leg
    +2.8798, +0.5236, +2.7576, +2.8798, +0.5236, +0.2618,  # right leg
    +2.6180, +0.5200, +0.5200,                              # waist
    +2.6704, +2.2515, +2.6180, +2.0944, +1.9722, +1.6144, +1.6144,  # left arm
    +2.6704, +1.5882, +2.6180, +2.0944, +1.9722, +1.6144, +1.6144,  # right arm
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# Per-joint velocity limits (rad/s) — data max at stride=2 (dt_eff=0.02s)
#
# These are the absolute maximum joint velocities observed across 231 training
# trajectories.  Anything the diffusion model produces beyond these values is
# definitively an artifact.
# ═══════════════════════════════════════════════════════════════════════════════
JOINT_VEL_MAX = np.array([
    12.6865,  6.6927,  4.7067, 20.3330,  6.7153,  3.4203,  # left leg
    10.3346,  4.7802,  6.1137, 17.2294,  6.9648,  2.9957,  # right leg
     3.4162,  3.9552,  4.6433,                              # waist
     7.1101,  5.2183,  4.1347,  5.5910,  2.7869,  2.9603,  2.5962,  # left arm
     7.0056,  5.8111,  4.0308,  5.6883,  3.2287,  3.2077,  2.5431,  # right arm
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# Body Z limits (pelvis height, meters) — from training data
# ═══════════════════════════════════════════════════════════════════════════════
BODY_Z_MIN = 0.40    # data min ≈ 0.4154, allow small margin
BODY_Z_MAX = 0.80    # data max ≈ 0.7923, allow small margin

# ═══════════════════════════════════════════════════════════════════════════════
# Robot XY velocity limit (m/s) — from training data at stride=2
# ═══════════════════════════════════════════════════════════════════════════════
ROBOT_XY_VEL_MAX = 1.50  # data max ≈ 1.37 m/s, with ~10% margin

# ═══════════════════════════════════════════════════════════════════════════════
# Yaw velocity limit (rad/s) — from training data at stride=2
# p99.9 = 4.35, true physical max (excluding wrapping artifacts) ~6-9 rad/s
# ═══════════════════════════════════════════════════════════════════════════════
YAW_VEL_MAX = 5.0  # rad/s — generous limit that covers p99.9 (4.35)

# ═══════════════════════════════════════════════════════════════════════════════
# Object relative position bounds (body frame, metres) — from training data
# These define the feasible workspace envelope around the robot pelvis.
# ═══════════════════════════════════════════════════════════════════════════════
OBJ_REL_POS_LO = np.array([0.10, -0.40, -0.55], dtype=np.float64)   # [x, y, z] min
OBJ_REL_POS_HI = np.array([0.75,  0.35,  0.40], dtype=np.float64)   # [x, y, z] max


def apply_physics_clamp(
    future_traj: np.ndarray,
    dt: float = 0.02,
    clamp_joint_pos: bool = True,
    clamp_joint_vel: bool = True,
    clamp_body_z: bool = True,
    clamp_xy_vel: bool = True,
    clamp_yaw_vel: bool = True,
    clamp_obj_rel_pos: bool = True,
    vel_margin: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply physics-informed clamping to denormalized SBTO feature trajectories.

    This function should be called **after** denormalization and **before**
    ``reconstruct_sbto_trajectory()``.

    Clamping order (important — position clamps first, then velocity):
        1. Joint position clamp  (hard XML limits)
        2. Body-Z clamp          (data-derived height range)
        3. Object rel_pos clamp  (data-derived workspace envelope)
        4. Joint velocity clamp   (data-derived per-joint limits, iterative forward pass)
        5. Robot XY velocity clamp (data-derived, via delta_xy increments)
        6. Yaw velocity clamp    (data-derived, via delta_yaw increments)

    Args:
        future_traj: (B, T, D) denormalized SBTO features.
        dt: Effective timestep in seconds (stride × raw_dt).
        clamp_joint_pos: Whether to clamp joint positions to XML limits.
        clamp_joint_vel: Whether to clamp per-joint velocities.
        clamp_body_z: Whether to clamp pelvis height.
        clamp_xy_vel: Whether to clamp robot XY velocity.
        clamp_yaw_vel: Whether to clamp robot yaw velocity.
        clamp_obj_rel_pos: Whether to clamp object relative position to workspace envelope.
        vel_margin: Multiplier on data-max velocity limits (1.0 = exact data max).
        verbose: Print clamping statistics.

    Returns:
        Clamped trajectory, same shape as input.  Modified in-place for
        efficiency but a copy is returned for safety.
    """
    traj = future_traj.copy()
    B, T, D = traj.shape

    # --- Resolve feature indices ---
    has_vel = D > 60
    has_obj_delta_xy = D > 48
    has_obj_z = D > 50
    layout = build_feature_layout(has_vel, has_obj_delta_xy, has_obj_z)
    indices = get_feature_indices(layout)

    IDX_JOINTS = indices["joints"]       # slice for 29 joints
    IDX_BODY_Z = indices["body_z"]       # slice for 1-dim body z
    IDX_DELTA_XY = indices["delta_xy"]   # slice for 2-dim delta_xy
    IDX_DELTA_YAW = indices["delta_yaw"] # slice for 1-dim delta_yaw
    IDX_OBJ_REL_POS = indices["obj_rel_pos"]  # slice for 3-dim obj_rel_pos

    stats = {}

    # ─── 1. Joint position clamp ────────────────────────────────────────────
    if clamp_joint_pos:
        joints = traj[:, :, IDX_JOINTS]  # (B, T, 29)
        lo_viol = (joints < JOINT_POS_LO).sum()
        hi_viol = (joints > JOINT_POS_HI).sum()
        np.clip(joints, JOINT_POS_LO, JOINT_POS_HI, out=joints)
        traj[:, :, IDX_JOINTS] = joints
        stats["joint_pos_clamps"] = int(lo_viol + hi_viol)

    # ─── 2. Body Z clamp ────────────────────────────────────────────────────
    if clamp_body_z:
        bz = traj[:, :, IDX_BODY_Z]  # (B, T, 1)
        bz_viol = ((bz < BODY_Z_MIN) | (bz > BODY_Z_MAX)).sum()
        np.clip(bz, BODY_Z_MIN, BODY_Z_MAX, out=bz)
        traj[:, :, IDX_BODY_Z] = bz
        stats["body_z_clamps"] = int(bz_viol)

    # ─── 3. Object relative position clamp ──────────────────────────────────
    if clamp_obj_rel_pos:
        orp = traj[:, :, IDX_OBJ_REL_POS]  # (B, T, 3)
        orp_viol = ((orp < OBJ_REL_POS_LO) | (orp > OBJ_REL_POS_HI)).sum()
        np.clip(orp, OBJ_REL_POS_LO, OBJ_REL_POS_HI, out=orp)
        traj[:, :, IDX_OBJ_REL_POS] = orp
        stats["obj_rel_pos_clamps"] = int(orp_viol)

    # ─── 4. Joint velocity clamp (iterative forward pass) ───────────────────
    if clamp_joint_vel and T > 1:
        max_delta = JOINT_VEL_MAX * vel_margin * dt  # (29,) max allowed change per step
        joints = traj[:, :, IDX_JOINTS]  # (B, T, 29)
        vel_clamps = 0
        for t in range(1, T):
            delta = joints[:, t, :] - joints[:, t - 1, :]  # (B, 29)
            clamped = np.clip(delta, -max_delta, max_delta)
            n_clamped = (clamped != delta).sum()
            vel_clamps += int(n_clamped)
            joints[:, t, :] = joints[:, t - 1, :] + clamped
        # Re-clamp positions after velocity correction (forward integration may drift)
        if clamp_joint_pos:
            np.clip(joints, JOINT_POS_LO, JOINT_POS_HI, out=joints)
        traj[:, :, IDX_JOINTS] = joints
        stats["joint_vel_clamps"] = vel_clamps

    # ─── 5. Robot XY velocity clamp (via cumulative delta_xy) ───────────────
    if clamp_xy_vel and T > 1:
        max_step = ROBOT_XY_VEL_MAX * vel_margin * dt  # max XY displacement per frame
        dxy = traj[:, :, IDX_DELTA_XY]  # (B, T, 2) — cumulative from anchor
        xy_clamps = 0
        for t in range(1, T):
            inc = dxy[:, t, :] - dxy[:, t - 1, :]  # (B, 2) per-frame increment
            inc_norm = np.linalg.norm(inc, axis=-1, keepdims=True)  # (B, 1)
            exceed = inc_norm > max_step
            if exceed.any():
                scale = np.where(exceed, max_step / np.maximum(inc_norm, 1e-8), 1.0)
                inc = inc * scale
                xy_clamps += int(exceed.sum())
            dxy[:, t, :] = dxy[:, t - 1, :] + inc
        traj[:, :, IDX_DELTA_XY] = dxy
        stats["xy_vel_clamps"] = xy_clamps

    # ─── 6. Yaw velocity clamp (via cumulative delta_yaw) ──────────────────
    if clamp_yaw_vel and T > 1:
        max_yaw_step = YAW_VEL_MAX * vel_margin * dt  # max yaw change per frame
        dyaw = traj[:, :, IDX_DELTA_YAW]  # (B, T, 1) — cumulative from anchor
        yaw_clamps = 0
        for t in range(1, T):
            inc = dyaw[:, t, :] - dyaw[:, t - 1, :]  # (B, 1)
            clamped = np.clip(inc, -max_yaw_step, max_yaw_step)
            n_clamped = int((clamped != inc).sum())
            yaw_clamps += n_clamped
            dyaw[:, t, :] = dyaw[:, t - 1, :] + clamped
        traj[:, :, IDX_DELTA_YAW] = dyaw
        stats["yaw_vel_clamps"] = yaw_clamps

    if verbose and any(v > 0 for v in stats.values()):
        parts = [f"{k}={v}" for k, v in stats.items() if v > 0]
        print(f"[physics_clamp] {', '.join(parts)}")

    return traj

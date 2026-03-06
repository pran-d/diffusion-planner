import mujoco
import numpy as np
from .math_tools import (
    normalize_angle,
    yaw_from_quat,
    yaw_to_rot_matrix,
    compute_relative_se3,
    quat_to_rot,
    rot_to_6d,
    rot6d_to_rot,
    rot_to_quat,
    batch_rotation,
    transpose,
)


def enforce_quaternion_continuity(q_traj, q_start=None):
    """
    Ensure quaternion trajectory is continuous by flipping sign if dot product is negative.
    Supports (B, T, 4) and (T, 4).
    If q_start is provided (B, 4) or (4,), it is used to anchor the first element.
    """
    q_out = q_traj.copy()

    # Check if batched
    is_batched = (q_out.ndim == 3)
    if not is_batched:
        q_out = q_out[None, ...]  # (1, T, 4)

    B, T, _ = q_out.shape
    
    # Handle start condition if provided
    if q_start is not None:
        if not is_batched and q_start.ndim == 1:
            q_start = q_start[None, ...] # (1, 4)
        
        # Check alignment of first frame with q_start
        dot_0 = np.sum(q_start * q_out[:, 0], axis=1) # (B,)
        mask_0 = (dot_0 < 0)
        q_out[mask_0, 0] *= -1.0

    for t in range(1, T):
        # Dot between q_{t-1} and q_t
        # Note: q_out[:, t-1] is already processed/flipped
        dot = np.sum(q_out[:, t - 1] * q_out[:, t], axis=1)  # (B,)

        # If dot < 0, flip q_t
        mask = (dot < 0)
        q_out[mask, t] *= -1.0

    if not is_batched:
        return q_out[0]
    return q_out


def compute_fk_batched(model, 
                       data, 
                       qpos_batch, 
                       base_name="pelvis", 
                       ee_names=("left_wrist_roll_link", "right_wrist_roll_link"),
                       joint_offset=7):
    """
    Computes relative coordinates for batched states of arbitrary dimensions (e.g., B, T, N).
    Returns a NumPy array matching the input batch dimensions: (B, T, num_ees, 3).
    """
    # 1. Standardize input to NumPy
    qpos_batch = np.asarray(qpos_batch)
    
    # Save the original batch dimensions (everything except the last feature dimension)
    batch_dims = qpos_batch.shape[:-1]
    N = qpos_batch.shape[-1]
    num_ees = len(ee_names)
    
    # Flatten all batch/time dimensions into a single list of states: (Total_States, N)
    qpos_flat = qpos_batch.reshape(-1, N)
    total_states = qpos_flat.shape[0]
    
    # 2. Pre-fetch and validate all MuJoCo IDs ONCE for massive speedup
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name)
    if base_id == -1:
        raise ValueError(f"Could not find base '{base_name}' in the model.")
        
    ee_ids = []
    for ee in ee_names:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee)
        if ee_id == -1:
            raise ValueError(f"Could not find end-effector '{ee}' in the model.")
        ee_ids.append(ee_id)

    # 3. Pre-allocate the flat output array
    relative_positions_flat = np.zeros((total_states, num_ees, 3))
    
    # 4. Fast loop over all states
    for i in range(total_states):
        # Inject the state for this step
        if N == model.nq:
            data.qpos[:] = qpos_flat[i]
        else:
            end_idx = joint_offset + N
            if end_idx > model.nq:
                raise ValueError(f"Array length {N} with offset {joint_offset} exceeds model.nq ({model.nq})")
            data.qpos[joint_offset:end_idx] = qpos_flat[i]

        # Compute Forward Kinematics
        mujoco.mj_kinematics(model, data)
        
        # Extract base global position and rotation matrix (3x3)
        base_pos_global = data.xpos[base_id]
        base_rot_global = data.xmat[base_id].reshape(3, 3)
        
        # Calculate and store relative positions for each end-effector
        for j, ee_id in enumerate(ee_ids):
            ee_pos_global = data.xpos[ee_id]
            
            # P_rel = R_base^T * (P_ee_global - P_base_global)
            global_diff = ee_pos_global - base_pos_global
            relative_positions_flat[i, j, :] = base_rot_global.T @ global_diff
            
    # 5. Reshape back to the original batch dimensions + (num_ees, 3)
    final_shape = batch_dims + (num_ees * 3, )
    return relative_positions_flat.reshape(final_shape)

# ============================================================
# Feature Layouts
# ============================================================

def build_feature_layout(has_vel=False, has_obj_delta_xy=True, has_obj_z=True):
    """
    Build the feature layout dict adaptively based on which optional feature
    groups are present in the trajectory.  The insertion order must match the
    assembly order in ``compute_sbto_components``.
    """
    layout = {
        "delta_xy":  2,
        "delta_yaw": 1,
    }
    if has_obj_delta_xy:
        layout["obj_delta_xy"] = 2
    if has_obj_z:
        layout["obj_z"] = 1
    layout.update({
        "joints":       29,
        "body_z":        1,
        "body_rot6d":    6,
        "obj_rel_pos":   3,
        "obj_rel_rot6d": 6,
    })
    if has_vel:
        layout.update({
            "joints_vel":  29,
            "body_lin_vel": 3,
            "body_ang_vel": 3,
            "obj_lin_vel":  3,
            "obj_ang_vel":  3,
        })
    return layout

# ============================================================
# Assembly functions
# ============================================================


def build_robot_frame_current_state(
    joints,
    body_z,
    body_rot6d,
    obj_rel_pos,
    obj_rel_rot6d,
    ref_idx,
    joints_vel=None,
    body_lin_vel=None,
    body_ang_vel=None,
    obj_lin_vel=None,
    obj_ang_vel=None,
    task_scalar=None,
):
    """
    Returns:
        current_state_robot : (B, 86) or (B, 45) or +1 if task_scalar
    """

    components = [
        joints[:, ref_idx],            # (B, 29)
        body_z[:, ref_idx],            # (B, 1)
        body_rot6d[:, ref_idx],        # (B, 6)
        obj_rel_pos[:, ref_idx],       # (B, 3)
        obj_rel_rot6d[:, ref_idx],     # (B, 6)
    ]

    if body_lin_vel is not None and body_ang_vel is not None:
        components.extend([
            joints_vel[:, ref_idx],        # (B, 29)
            body_lin_vel[:, ref_idx],      # (B, 3)
            body_ang_vel[:, ref_idx],      # (B, 3)
            obj_lin_vel[:, ref_idx],       # (B, 3)
            obj_ang_vel[:, ref_idx],       # (B, 3)
        ])
    
    # Append task scalar if provided (B, 1)
    if task_scalar is not None:
         # Ensure shape (B, 1)
         ts = task_scalar if task_scalar.ndim > 1 else task_scalar[:, None]
         # Broadcast if necessary (though usually passed as (B, 1) or (1,))
         if ts.shape[0] != joints.shape[0]:
             ts = np.repeat(ts, joints.shape[0], axis=0)
         components.append(ts)

    return np.concatenate(components, axis=-1).astype(np.float32)



# ============================================================
# SBTO Feature Extraction
# ============================================================

def compute_sbto_components(
    base_w,          # (B, T, 7)
    joints,        # (B, T, 29)
    obj_w,           # (B, T, 7)
    ref_idx,
    base_vel=None,
    joints_vel=None,
    obj_vel=None,
    save_velocities=False,
    ee_rel_pos=None
):
    """
    Core SBTO transformation logic returning a dictionary of components.
    Useful for flexible dataset loaders.
    """
    B, T, _ = base_w.shape

    # --------------------------------------------------------
    # Reference frame (current step)
    # --------------------------------------------------------
    ref_pos = base_w[:, [ref_idx], :3]
    ref_quat = base_w[:, [ref_idx], 3:]
    ref_yaw = yaw_from_quat(ref_quat)
    R_ref_inv = yaw_to_rot_matrix(-ref_yaw)

    # --------------------------------------------------------
    # Robot pose deltas
    # --------------------------------------------------------
    delta_pos_world = base_w[..., :3] - ref_pos
    delta_pos_local = batch_rotation(R_ref_inv, delta_pos_world)

    delta_xy = delta_pos_local[..., :2]
    body_z = base_w[..., 2:3]

    yaw = yaw_from_quat(base_w[..., 3:])
    delta_yaw = normalize_angle(yaw - ref_yaw)[..., None]

    R_body = quat_to_rot(base_w[..., 3:])
    R_yaw_inv = yaw_to_rot_matrix(-yaw)
    R_body_no_yaw = R_yaw_inv @ R_body
     
    body_rot6d = rot_to_6d(R_body_no_yaw)

    # Velocities
    body_lin_vel, body_ang_vel, obj_lin_vel, obj_ang_vel = None, None, None, None
    if save_velocities:
        base_vel = base_vel if base_vel is not None else np.zeros((B, T, 6))
        joints_vel = joints_vel if joints_vel is not None else np.zeros_like(joints)
        obj_vel = obj_vel if obj_vel is not None else np.zeros((B, T, 6))

        R_body_T = transpose(R_body)
        body_lin_vel = batch_rotation(R_body_T, base_vel[..., :3])
        body_ang_vel = batch_rotation(R_body_T, base_vel[..., 3:])
        obj_lin_vel = batch_rotation(R_body_T, obj_vel[..., :3])
        obj_ang_vel = batch_rotation(R_body_T, obj_vel[..., 3:])
    
    # Object relative
    obj_rel_pos, obj_rel_rot = compute_relative_se3(
        base_w[..., :3], base_w[..., 3:],
        obj_w[..., :3], obj_w[..., 3:]
    )
    obj_rel_rot6d = rot_to_6d(obj_rel_rot)

    # Object global
    obj_delta_pos_world = obj_w[..., :3] - obj_w[..., [ref_idx], :3]
    obj_delta_pos_local = batch_rotation(R_ref_inv, obj_delta_pos_world)
    obj_delta_xy = obj_delta_pos_local[..., :2]
    obj_z = obj_w[..., 2:3]

    comps = {
        "joints": joints,
        "delta_xy": delta_xy,
        "delta_yaw": delta_yaw,
        "body_z": body_z,
        "body_rot6d": body_rot6d,
        "obj_rel_pos": obj_rel_pos,
        "obj_rel_rot6d": obj_rel_rot6d,
        "obj_delta_xy": obj_delta_xy,
        "obj_z": obj_z,
        "ee_rel_pos": ee_rel_pos,
    }
    
    if save_velocities:
        comps.update({
             "joints_vel": joints_vel,
             "body_lin_vel": body_lin_vel,
             "body_ang_vel": body_ang_vel,
             "obj_lin_vel": obj_lin_vel,
             "obj_ang_vel": obj_ang_vel
        })

    anchors = {
        "ref_pos": ref_pos,
        "ref_quat": ref_quat,
        "R_ref_inv": R_ref_inv,
        "R_body": R_body,
        "ref_obj_pos": obj_w[:, [ref_idx], :3],
    }

    return comps, anchors

def compute_sbto_features(
    base,          # (B, T, 7)
    joints,        # (B, T, 29)
    obj,           # (B, T, 7)
    ref_idx,
    base_vel=None,     # (B, T, 6)
    joints_vel=None,   # (B, T, 29)
    obj_vel=None,      # (B, T, 6)
    additional_goals=None,
    labels=None,
    save_velocities=False,
    goal_in_curr_state=False,
    task_scalar=None, # (B,) scalar
):
    """
    Compute SBTO-style features.

    Returns:
        obs_history : (B, ref_idx, C)
        obs_future  : (B, T-ref_idx, C)
        guidance    : (B, 10)
        current_state : (B, 1, D)
        base_pose_world : (B, T, 7)
    """

    # 1. Compute components
    comps, anchors = compute_sbto_components(
        base, joints, obj, ref_idx,
        base_vel, joints_vel, obj_vel,
        save_velocities
    )
    
    # Unpack for ease of access
    delta_xy, delta_yaw = comps['delta_xy'], comps['delta_yaw']
    body_z, body_rot6d = comps['body_z'], comps['body_rot6d']
    obj_rel_pos, obj_rel_rot6d = comps['obj_rel_pos'], comps['obj_rel_rot6d']
    
    # Velocities if needed
    joints_vel = comps.get('joints_vel')
    body_lin_vel = comps.get('body_lin_vel')
    body_ang_vel = comps.get('body_ang_vel')
    obj_lin_vel = comps.get('obj_lin_vel')
    obj_ang_vel = comps.get('obj_ang_vel')

    # Anchors
    R_ref_inv, R_body = anchors['R_ref_inv'], anchors['R_body']
    # --------------------------------------------------------
    # Current state (in robot frame)
    # --------------------------------------------------------

    current_state = build_robot_frame_current_state(
        joints,
        body_z,
        body_rot6d,
        obj_rel_pos,
        obj_rel_rot6d,
        ref_idx,
        joints_vel,
        body_lin_vel,
        body_ang_vel,
        obj_lin_vel,
        obj_ang_vel,
        task_scalar=task_scalar,
    )
    # --------------------------------------------------------
    # Guidance (current → final object)
    # --------------------------------------------------------
    R_obj_world = quat_to_rot(obj[:, [ref_idx], 3:])
    R_goal_world = quat_to_rot(obj[:, [-1], 3:])

    # 1. Main goal
    guidance = compute_guidance_vec(
        obj[:, [ref_idx], :3], 
        obj[:, [-1], :3], 
        R_ref_inv,
        current_rot=R_obj_world,
        target_rot=R_goal_world,
    )

    # 2. Additional Goals
    if additional_goals is not None:
        # additional_goals shape: (B, K, 7)
        guidance_list = []
        K = additional_goals.shape[1]
        for k in range(K):
            g_pos = additional_goals[:, [k], :3]
            c_pos = obj[:, [ref_idx], :3]
            guidance_list.append(
                compute_guidance_vec(
                    current_pos=c_pos, 
                    target_pos=g_pos, 
                    R_robot_yaw_inv=R_ref_inv
                )
            )

        # guidance shape: (B, K*4)
        extra_goals = np.concatenate(guidance_list, axis=-1) 

    if goal_in_curr_state:
        # Just use the main goal for the current state vector
        current_state = np.concatenate([current_state, guidance[..., :3]], axis=-1)

    # --------------------------------------------------------
    # Feature assembly
    # --------------------------------------------------------
    if save_velocities:
        features = np.concatenate([
            delta_xy,
            delta_yaw,
            joints,
            body_z,
            body_rot6d,
            obj_rel_pos,
            obj_rel_rot6d,
            joints_vel,
            body_lin_vel,
            body_ang_vel,
            obj_lin_vel,
            obj_ang_vel,
        ], axis=-1)
    else:
        features = np.concatenate([
            delta_xy,
            delta_yaw,
            joints,
            body_z,
            body_rot6d,
            obj_rel_pos,
            obj_rel_rot6d,
        ], axis=-1)

    features = features.astype(np.float32)
    # No tiling needed now
    
    base_pose_world = base[:, ref_idx, :7].astype(np.float32)
    # No tiling needed now

    # --------------------------------------------------------
    # Split history / future
    # --------------------------------------------------------
    obs_history = features[:, :ref_idx]
    obs_future = features[:, ref_idx:]

    return (
        obs_history, 
        obs_future, 
        guidance, 
        current_state,
        base_pose_world,
        extra_goals if additional_goals is not None else None,
        labels if labels is not None else None,
    )


# ============================================================
# World frame trajectory reconstruction
# ============================================================

def reconstruct_sbto_trajectory(
    base_pose_world,
    future_traj,
    inpaint=False,
):
    """
    Reconstruct world-frame robot + object trajectory from robot-frame SBTO features.

    Args:
        current_state: (86,) robot-frame current state
        future_traj: (T, D) robot-frame future trajectory
        base_pose_world: (7,) [x, y, z, qw, qx, qy, qz]

    Returns:
        robot_state_world: (T, 7 + 29 [+ vels])
        object_state_world: (T, 7 [+ vels])
    """

    _, T, D = future_traj.shape
    has_vel = (D > 60) # Heuristic to detect if we have velocity layout (D=89 vs D=48)
    has_obj_delta_xy = (D > 48) # Heuristic to detect if we have obj_delta_xy (D=48 vs D=50)
    has_obj_z = (D > 50) # Heuristic to detect if we have obj_z (D=50 vs D=51)
    # --------------------------------------------------
    # Feature indices
    # --------------------------------------------------
    FEATURE_MAP = build_feature_layout(has_vel, has_obj_delta_xy, has_obj_z)
    indices = get_feature_indices(FEATURE_MAP)

    IDX_DELTA_XY     = indices["delta_xy"]
    IDX_DELTA_YAW    = indices["delta_yaw"]
    IDX_OBJ_DELTA_XY = indices.get("obj_delta_xy")          # None when absent
    IDX_OBJ_Z        = indices.get("obj_z")                 # None when absent
    IDX_JOINTS       = indices["joints"]
    IDX_Z            = indices["body_z"].start               # integer — body_z is 1-D
    IDX_ROT          = indices["body_rot6d"]
    IDX_OBJ_POS      = indices["obj_rel_pos"]
    IDX_OBJ_ROT      = indices["obj_rel_rot6d"]

    # --------------------------------------------------
    # Base world pose (anchor)
    # --------------------------------------------------
    base_pos_world_anchor = base_pose_world[:, :3]
    ref_yaw = yaw_from_quat(base_pose_world[:, 3:7])
    obj_global_anchor = base_pose_world[:, 7:10]
    R_ref_yaw = yaw_to_rot_matrix(ref_yaw)

    # --------------------------------------------------
    # Extract robot-frame quantities
    # --------------------------------------------------
    traj_joints = future_traj[:, :, IDX_JOINTS]
    traj_body_z = future_traj[:, :, IDX_Z]
    traj_body_rot6d = future_traj[:, :, IDX_ROT]

    traj_delta_xy = future_traj[:, :, IDX_DELTA_XY] # (T, 2)
    traj_delta_yaw = future_traj[:, :, IDX_DELTA_YAW] # (T, 1)

    if has_obj_delta_xy:
        traj_delta_obj_xy = future_traj[:, :, IDX_OBJ_DELTA_XY] # (T, 2)
    # --------------------------------------------------
    # Robot pose (world)
    # --------------------------------------------------
    # Position: XY reconstructed from delta_xy and ref frame
    # Z from absolute Z feature
    
    # Delta pos local (T, 3) - but we only have XY
    delta_pos_local = np.zeros(traj_delta_xy.shape[:-1] + (3,))
    delta_pos_local[..., :2] = traj_delta_xy
    
    # Transform to world: pos = ref_pos + R_ref @ delta_local
    # Note: R_ref here is just the yaw rotation, as delta_xy is relative to ref frame yaw
    # Expand dims for broadcasting: (B, 1, 3, 3) @ (B, T, 3, 1) -> squeeze
    delta_pos_world = (R_ref_yaw[:, None, :, :] @ delta_pos_local[..., None]).squeeze(-1)
    
    pos_world = np.zeros(traj_delta_xy.shape[:-1] + (3,))
    pos_world[..., :2] = base_pos_world_anchor[:, None, :2] + delta_pos_world[..., :2]

    pos_world[..., 2] = traj_body_z

    # Orientation
    # yaw = ref_yaw + delta_yaw
    # R = RotZ(yaw) @ R_no_yaw
    
    traj_yaw = normalize_angle(ref_yaw[:, None] + traj_delta_yaw.squeeze(-1))
    R_yaw = yaw_to_rot_matrix(traj_yaw)
    
    R_no_yaw = rot6d_to_rot(traj_body_rot6d)
    
    R_world = R_yaw @ R_no_yaw
    quat_world = rot_to_quat(R_world)
    
    # Enforce quaternion continuity relative to anchor
    # Anchor quat is at base_pose_world[:, 3:7] (B, 4)
    q_anchor = base_pose_world[:, 3:7]
    quat_world = enforce_quaternion_continuity(quat_world, q_start=q_anchor)

    # --------------------------------------------------
    # Object pose
    # --------------------------------------------------
    traj_obj_pos_local = future_traj[:, :, IDX_OBJ_POS]
    traj_obj_rot_local = rot6d_to_rot(future_traj[:, :, IDX_OBJ_ROT])
    
    obj_pos_world = pos_world + (R_world @ traj_obj_pos_local[..., None]).squeeze(-1)
    obj_rot_world = R_world @ traj_obj_rot_local
    obj_quat_world = rot_to_quat(obj_rot_world)
    
    # Enforce quaternion continuity - Object
    # Just continuous within traj for now
    obj_quat_world = enforce_quaternion_continuity(obj_quat_world)

    if inpaint and has_obj_delta_xy:
        # print("Inpainting object_xy...")
        delta_obj_local = np.zeros((_, T, 3))
        delta_obj_local[..., :2] = traj_delta_obj_xy
        delta_obj_global = (R_ref_yaw[:, None, :, :] @ delta_obj_local[..., None]).squeeze(-1)
        obj_pos_world[..., :2] = obj_global_anchor[:, None, :2] + delta_obj_global[..., :2]

    # Use the absolute obj_z feature for the object Z coordinate when available.
    # Without this, the Z comes from obj_rel_pos (relative to pelvis), which is
    # inconsistent with any obj_z waypoint constraints applied during inference.
    if inpaint and IDX_OBJ_Z is not None:
        obj_pos_world[..., 2] = future_traj[:, :, IDX_OBJ_Z.start]

    # --------------------------------------------------
    # Velocities (optional)
    # --------------------------------------------------
    if has_vel:
        IDX_JOINTS_VEL  = indices["joints_vel"]
        IDX_LIN_VEL     = indices["body_lin_vel"]
        IDX_ANG_VEL     = indices["body_ang_vel"]
        IDX_OBJ_LIN_VEL = indices["obj_lin_vel"]
        IDX_OBJ_ANG_VEL = indices["obj_ang_vel"]

        traj_joints_vel = future_traj[:, :, IDX_JOINTS_VEL]

        body_lin_vel_local = future_traj[:, :, IDX_LIN_VEL]

        body_ang_vel_local = future_traj[:, :, IDX_ANG_VEL]

        obj_lin_vel_local = future_traj[:, :, IDX_OBJ_LIN_VEL]
        obj_ang_vel_local = future_traj[:, :, IDX_OBJ_ANG_VEL]
        
        # Transform velocities to world frame
        # v_world = R_world @ v_local
        
        body_lin_vel_world = batch_rotation(R_world, body_lin_vel_local)
        body_ang_vel_world = batch_rotation(R_world, body_ang_vel_local)

        obj_lin_vel_world = batch_rotation(R_world, obj_lin_vel_local)
        obj_ang_vel_world = batch_rotation(R_world, obj_ang_vel_local)

        robot_state_world = np.concatenate([
            pos_world,
            quat_world,
            traj_joints,
        ], axis=-1)

        robot_state_velocities = np.concatenate([
            body_lin_vel_world,
            body_ang_vel_world,
            traj_joints_vel,
        ], axis=-1)

        object_state_world = np.concatenate([
            obj_pos_world,
            obj_quat_world,
        ], axis=-1)

        obj_state_velocities = np.concatenate([
            obj_lin_vel_world,
            obj_ang_vel_world,
        ], axis=-1)

        return robot_state_world, object_state_world, robot_state_velocities, obj_state_velocities

    else:
        robot_state_world = np.concatenate([
            pos_world,
            quat_world,
            traj_joints,
        ], axis=-1)

        object_state_world = np.concatenate([
            obj_pos_world,
            obj_quat_world,
        ], axis=-1)

        return robot_state_world, object_state_world, None, None


def extract_current_object_world_pose(current_state, base_pose_world):
    """
    Reconstruct current object world pose from current_state and base_pose_world
    """
    D = current_state.shape[-1]
    
    # Indices (authoritative)
    idx = 0
    idx += 29      # joint_pos
    idx += 1       # body_z
    idx += 6       # body_rot6d

    obj_rel_pos = current_state[..., idx:idx+3]; idx += 3
    obj_rel_rot6d = current_state[..., idx:idx+6]

    # Base transform
    base_pos = base_pose_world[..., :3]
    R_base = quat_to_rot(base_pose_world[..., 3:7])

    # Object world pose
    obj_pos_world = base_pos + batch_rotation(R_base, obj_rel_pos)
    obj_rot_world = R_base @ rot6d_to_rot(obj_rel_rot6d)

    return obj_pos_world, obj_rot_world


def reconstruct_goal_pose(
    current_state,
    goal_cond,
    base_pose_world,
):
    """
    Reconstruct absolute goal pose from robot-frame guidance.

    Args:
        current_state: (B, 86) or (86,) robot-frame
        goal_cond: (B, 10) or (10,) [dir(3), rot6d(6), dist(1)]
        base_pose_world: (B, 7) or (7,)

    Returns:
        goal_pos_world: (B, 3) or (3,)
        goal_quat_world: (B, 4) or (4,)
    """
    # handle batches
    batched = current_state.ndim > 1
    
    guidance_dir = goal_cond[..., :3]
    guidance_rot6d = goal_cond[..., 3:9]
    distance = goal_cond[..., 9] # (B,) or ()
    
    if batched:
        distance = distance[..., None] # (B, 1)

    # --- Current object pose (world) ---
    obj_pos_world, obj_rot_world = extract_current_object_world_pose(
        current_state, base_pose_world
    )

    # --- Position ---
    delta_pos_robot = guidance_dir * distance
    
    # Batch rotation handling
    q = base_pose_world[..., 3:7]
    yaw = yaw_from_quat(q)
    R_base = yaw_to_rot_matrix(yaw) # (B, 3, 3) or (3, 3)
    
    if batched:
        # (B, 3, 3) @ (B, 3, 1) -> (B, 3, 1)
        delta_pos_world = (R_base @ delta_pos_robot[..., None]).squeeze(-1)
    else:
        delta_pos_world = R_base @ delta_pos_robot

    goal_pos_world = obj_pos_world + delta_pos_world

    # --- Orientation ---
    R_rel = rot6d_to_rot(guidance_rot6d)
    R_goal = obj_rot_world @ R_rel
    goal_quat_world = rot_to_quat(R_goal)

    return goal_pos_world, goal_quat_world


def compute_goal_guidance(
    current_state,
    goal_pos_world,
    goal_quat_world,
    base_pose_world,
):
    """
    Compute robot-frame goal guidance from absolute goal pose.

    Args:
        current_state: (86,) robot-frame
        goal_pos_world: (3,)
        goal_quat_world: (4,)
        base_pose_world: (7,)

    Returns:
        guidance: (10,) [dir(3), rot6d(6), dist(1)]
    """

    # --- Current object pose ---
    obj_pos_world, R_obj_world = extract_current_object_world_pose(
        current_state, base_pose_world
    )

    R_robot_yaw_inv = yaw_to_rot_matrix(-yaw_from_quat(base_pose_world[3:7]))

    return compute_guidance_vec(
        obj_pos_world,
        goal_pos_world,
        R_robot_yaw_inv,
        current_rot=R_obj_world,
        target_rot=quat_to_rot(goal_quat_world),
    )

def compute_guidance_vec(current_pos, target_pos, R_robot_yaw_inv, current_rot=None, target_rot=None):
    delta_obj_world = target_pos - current_pos
    dist = np.linalg.norm(delta_obj_world, axis=-1, keepdims=True)

    delta_obj_body = batch_rotation(R_robot_yaw_inv, delta_obj_world)

    guidance_dir = np.divide(
        delta_obj_body,
        dist,
        out=np.tile(np.array([0.0, 0.0, 1.0]), (delta_obj_body.shape[0], 1, 1)),
        where=dist > 1e-6,
    )

    if current_rot is not None and target_rot is not None:
        R_obj_world = current_rot
        R_goal_world = target_rot
        R_rel = transpose(R_obj_world) @ R_goal_world
        guidance_rot6d = rot_to_6d(R_rel)
        return np.concatenate([guidance_dir, guidance_rot6d, dist], axis=-1).squeeze(1)

    return np.concatenate([guidance_dir, dist], axis=-1).squeeze(1)

def compute_task_params(current_robot_state, current_obj_state, desired_obj_pos, normalize_goal_vec=True, num_task_params=3, max_goal_dist=1.0):
    """
    Computes the task parameters (local object displacement) for the diffusion model.
    Matches the normalization convention of FlexibleWindowDataset._compute_task_params.

    Args:
        current_robot_state (np.ndarray): Shape (7,) or (4,) [x, y, z, qx, qy, qz, qw]
                                          representing the robot base pose (or quat).
        current_obj_state (np.ndarray): Shape (3,) or (7,) [x, y, z, ...]
                                        representing the current object position.
        desired_obj_pos (np.ndarray): Shape (3,) [x, y, z] 
                                      representing the GOAL object position in world frame.
        max_goal_dist (float): Maximum allowed distance for the goal vector.

    Returns:
        np.ndarray: Shape (..., num_task_params).
    """
    current_robot_state = np.asarray(current_robot_state, dtype=np.float64)
    current_obj_state = np.asarray(current_obj_state, dtype=np.float64)
    desired_obj_pos = np.asarray(desired_obj_pos, dtype=np.float64)

    # 1. Extract Positions
    curr_obj_pos = current_obj_state[..., :3]
    goal_obj_pos = desired_obj_pos[..., :3]
    
    # 2. Calculate World Displacement
    world_delta = goal_obj_pos - curr_obj_pos
    
    # 3. Extract Robot Yaw
    if current_robot_state.shape[-1] >= 7:
        quat = current_robot_state[..., 3:7]
    else:
        quat = current_robot_state
    
    # yaw_from_quat returns (N,) or ()
    # yaw_to_rot_matrix returns (N, 3, 3) or (3, 3)
    yaw = -yaw_from_quat(quat)
    R_ref_inv = yaw_to_rot_matrix(yaw)

    # 4. Rotate into Robot Frame (Global -> Local)
    # Ensure correct broadcasting for batch matrix multiplication
    # world_delta is (N, 3) -> (N, 3, 1)
    if world_delta.ndim == 1:
        wd_col = world_delta[:, None]
    else:
        wd_col = world_delta[..., None]
        
    # Standard matrix multiplication (supports broadcasting if shapes align)
    local_delta = (R_ref_inv @ wd_col)[..., 0]

    local_delta_norm = None
    if normalize_goal_vec:    
        # Project to task dimensions (e.g. 2D plane)
        vec_part = local_delta[..., :num_task_params-1]
        
        # Consistent norm calculation
        norm = np.linalg.norm(vec_part, axis=-1, keepdims=True)
        local_delta_norm = norm
        
        # Vectorized normalization with safe division
        mask = (norm > 1e-3)
        normalized_vec = np.zeros_like(vec_part)
        
        # Use np.divide with 'where' to handle zero norms safely and efficiently
        np.divide(vec_part, norm, out=normalized_vec, where=mask)
        
        # Clip distance magnitude
        norm_clipped = np.clip(norm, a_min=None, a_max=max_goal_dist)
        
        # Concatenate: [normalized_dir, clipped_magnitude]
        # Ensure we concatenate along the last axis
        local_delta = np.concatenate([normalized_vec, norm_clipped], axis=-1)
        
    return local_delta, local_delta_norm

def get_feature_indices(feature_layout):
    """
    Utility to get feature indices based on the provided layout dictionary.
    Returns a dict of slices for each feature.
    """
    indices = {}
    idx = 0
    for key, size in feature_layout.items():
        indices[key] = slice(idx, idx + size)
        idx += size
    return indices
import time
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import os
import sys
from scipy.interpolate import interp1d

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# Finite Difference Utilities (Copied from test_rollout_2.py)
# ============================================================

def finite_diff_qpos_traj_high_order(qpos: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocities using 4th-order accurate finite differencing."""
    T, nq = qpos.shape
    qvel = np.zeros_like(qpos)

    if T >= 5:
        qvel[2:-2] = (-qpos[4:] + 8*qpos[3:-1] - 8*qpos[1:-3] + qpos[0:-4]) / (12 * dt)
        qvel[0] = (-25*qpos[0] + 48*qpos[1] - 36*qpos[2] + 16*qpos[3] - 3*qpos[4]) / (12*dt)
        qvel[1] = (-3*qpos[0] - 10*qpos[1] + 18*qpos[2] - 6*qpos[3] + qpos[4]) / (12*dt)
        qvel[-1] = (25*qpos[-1] - 48*qpos[-2] + 36*qpos[-3] - 16*qpos[-4] + 3*qpos[-5]) / (12*dt)
        qvel[-2] = (3*qpos[-1] + 10*qpos[-2] - 18*qpos[-3] + 6*qpos[-4] - qpos[-5]) / (12*dt)
    else:
        qvel[1:-1] = (qpos[2:] - qpos[:-2]) / (2 * dt)
        if T >= 3:
            qvel[0] = (-3*qpos[0] + 4*qpos[1] - qpos[2]) / (2 * dt)
            qvel[-1] = (3*qpos[-1] - 4*qpos[-2] + qpos[-3]) / (2 * dt)
        elif T == 2:
            qvel[0] = (qpos[1] - qpos[0]) / dt
            qvel[-1] = (qpos[-1] - qpos[-2]) / dt

    return qvel


def finite_diff_quat(q1: np.ndarray, q2: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity from two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    wx = w1*x2 - x1*w2 - y1*z2 + z1*y2
    wy = w1*y2 + x1*z2 - y1*w2 - z1*x2
    wz = w1*z2 - x1*y2 + y1*x2 - z1*w2
    return (2.0 / dt) * np.array([wx, wy, wz])


def finite_diff_quat_traj(qpos: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity trajectory from quaternion trajectory."""
    T = qpos.shape[0]
    omega = np.zeros((T, 3))
    q_norm = qpos / np.linalg.norm(qpos, axis=1, keepdims=True)

    for t in range(1, T - 1):
        omega[t] = finite_diff_quat(q_norm[t - 1], q_norm[t + 1], 2 * dt)

    if T >= 3:
        omega[0] = finite_diff_quat(q_norm[0], q_norm[2], 2 * dt)
        omega[-1] = finite_diff_quat(q_norm[-3], q_norm[-1], 2 * dt)
    elif T == 2:
        omega[0] = finite_diff_quat(q_norm[0], q_norm[1], dt)
        omega[-1] = finite_diff_quat(q_norm[-2], q_norm[-1], dt)

    return omega

def load_reference_motion(ref_path: str, dt_model: float) -> np.ndarray:
    """
    Load reference motion and compute full state trajectory.
    """
    data = np.load(ref_path)
    qpos_raw = data["qpos"]
    fps = float(data["fps"].item() if isinstance(data["fps"], np.ndarray) else data["fps"])
    
    # Parse reference layout
    root_rot = qpos_raw[:, :4]
    root_pos = qpos_raw[:, 4:7]
    dof_pos = qpos_raw[:, 7:-7]
    object_rot = qpos_raw[:, -7:-3]
    object_pos = qpos_raw[:, -3:]
    
    dt_ref = 1.0 / fps
    
    # Interpolate to MuJoCo timestep if needed
    if abs(dt_model - dt_ref) > 1e-9:
        t_ref = np.arange(len(qpos_raw)) * dt_ref
        t_mj = np.arange(0, t_ref[-1], dt_model)
        
        def interp(arr):
            f = interp1d(t_ref, arr, axis=0, bounds_error=False, fill_value="extrapolate")
            return f(t_mj)
        
        root_pos = interp(root_pos)
        root_rot = interp(root_rot)
        dof_pos = interp(dof_pos)
        object_pos = interp(object_pos)
        object_rot = interp(object_rot)
        root_rot = root_rot / np.linalg.norm(root_rot, axis=-1, keepdims=True)
        object_rot = object_rot / np.linalg.norm(object_rot, axis=-1, keepdims=True)
        dt = dt_model
    else:
        dt = dt_ref
    
    # Compute velocities
    root_v = finite_diff_qpos_traj_high_order(root_pos, dt)
    root_w = finite_diff_quat_traj(root_rot, dt)
    dof_v = finite_diff_qpos_traj_high_order(dof_pos, dt)
    obj_v = finite_diff_qpos_traj_high_order(object_pos, dt)
    obj_w = finite_diff_quat_traj(object_rot, dt)
    
    # Construct MuJoCo state
    qpos = np.hstack([root_pos, root_rot, dof_pos, object_pos, object_rot])
    qvel = np.hstack([root_v, root_w, dof_v, obj_v, obj_w])
    x = np.hstack([qpos, qvel])
    
    return x

def rollout_trajectory(args):
    # Load data
    if not os.path.exists(args.file):
        print(f"File {args.file} does not exist.")
        return

    data = np.load(args.file)
    print(f"Loaded data from {args.file}")
    
    # 1. Load Control Inputs (u)
    if "u" in data:
        u = data["u"]
    elif "ctrl" in data:
        u = data["ctrl"]
    elif "actuator_force" in data:
        u = data["actuator_force"]
    else:
        print("No control input found.")
        return

    traj_idx = args.traj_num
    if u.ndim == 3:
        u = u[traj_idx]

    # Load Model
    xml_path = "simple_diffusion/test_datasets/mj_model.xml"
    if args.xml:
        xml_path = args.xml
    
    if not os.path.exists(xml_path):
        print(f"XML file {xml_path} not found.")
        return

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # 2. Load Initial State
    qpos0 = None
    qvel0 = None

    # Option A: Load from Reference File (Priority)
    if args.ref:
        if not os.path.exists(args.ref):
            print(f"Reference file {args.ref} not found.")
            return
        print(f"Loading initial state from reference: {args.ref}")
        try:
            ref_traj = load_reference_motion(args.ref, mj_model.opt.timestep)
            # Take the first frame
            x0 = ref_traj[0]
            qpos0 = x0[:mj_model.nq]
            qvel0 = x0[mj_model.nq:mj_model.nq+mj_model.nv]
        except Exception as e:
            print(f"Error loading reference: {e}")

    # Option B: Load from Dataset (Fallback)
    if qpos0 is None:
        if "base_xyz_quat" in data and "actuator_pos" in data:
            try:
                base_pos = data["base_xyz_quat"][traj_idx, 0, :]
                act_pos = data["actuator_pos"][traj_idx, 0, :]
                obj_pos = data["obj_0_xyz_quat"][traj_idx, 0, :]
                
                base_vel = data["base_linvel_angvel"][traj_idx, 0, :]
                act_vel = data["actuator_vel"][traj_idx, 0, :]
                obj_vel = data["obj_0_linvel_angvel"][traj_idx, 0, :]
                
                qpos0 = np.concatenate([base_pos, act_pos, obj_pos])
                qvel0 = np.concatenate([base_vel, act_vel, obj_vel])
            except Exception as e:
                print(f"Error constructing state: {e}")

        if qpos0 is None:
            if "qpos" in data:
                qpos0 = data["qpos"][traj_idx, 0]
                qvel0 = data["qvel"][traj_idx, 0]
            elif "initial" in data:
                init_state = data["initial"][traj_idx, 0]
                qpos0 = init_state[:mj_model.nq]
                qvel0 = init_state[mj_model.nq:mj_model.nq+mj_model.nv]
            elif "x" in data:
                 init_state = data["x"][traj_idx, 0]
                 qpos0 = init_state[:mj_model.nq]
                 qvel0 = init_state[mj_model.nq:mj_model.nq+mj_model.nv]

    if qpos0 is None:
        print("Could not determine initial state.")
        return

    # Use standard mujoco viewer
    print(f"Starting rollout... Steps: {u.shape[0]}, Substeps: {args.substeps}")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Reset
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:] = qpos0
        mj_data.qvel[:] = qvel0
        mujoco.mj_forward(mj_model, mj_data)
        
        viewer.sync()
        time.sleep(1.0)
        
        dt = mj_model.opt.timestep
        
        for t in range(u.shape[0]):
            if not viewer.is_running():
                break
            
            ctrl = u[t]
            if len(ctrl) != mj_model.nu:
                ctrl = ctrl[:mj_model.nu]
            mj_data.ctrl[:] = ctrl
            
            for _ in range(args.substeps):
                mujoco.mj_step(mj_model, mj_data)
                
            viewer.sync()
            time.sleep(dt * args.substeps)

    viewer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--xml", type=str, default="test_datasets/mj_model.xml", help="Path to model XML")
    parser.add_argument("--traj_num", type=int, default=0, help="Trajectory index")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation steps per control input")
    parser.add_argument("--ref", type=str, default=None, help="Path to reference motion npz file for initial state")
    args = parser.parse_args()

    rollout_trajectory(args)
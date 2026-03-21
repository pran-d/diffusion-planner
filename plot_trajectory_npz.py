#!/usr/bin/env python3
"""
Plot trajectory diagnostics from a saved NPZ trajectory bundle.

Outputs:
1) XY in initial robot yaw-rotated frame:
   - object XY path (+ goal marker if present)
   - robot XY path
   - robot yaw vs time (yaw_t - yaw_0)
2) Object Z and robot Z vs time.
3) Robot hand-to-object distance vs time (left/right/mean).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from config.configure import get_mj_xml_paths
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix


def _pick_trajectory(npz_data, index):
    if "trajectory" in npz_data:
        traj = npz_data["trajectory"]
        if traj.ndim == 3:
            traj = traj[index]
        rl = None
        if "real_length" in npz_data:
            rl = int(np.asarray(npz_data["real_length"]).reshape(-1)[0])
        elif "real_lengths" in npz_data:
            rr = np.asarray(npz_data["real_lengths"]).reshape(-1)
            rl = int(rr[min(index, len(rr) - 1)])
        goal_world = None
        if "goal_world" in npz_data:
            goal_world = np.asarray(npz_data["goal_world"]).reshape(-1)
        elif "goals_world" in npz_data:
            gw = np.asarray(npz_data["goals_world"])
            goal_world = gw[index]
        return traj, rl, goal_world

    if "trajectories" in npz_data:
        tr = np.asarray(npz_data["trajectories"])
        traj = tr[index]
        rl = None
        if "real_lengths" in npz_data:
            rl = int(np.asarray(npz_data["real_lengths"])[index])
        goal_world = None
        if "goals_world" in npz_data:
            goal_world = np.asarray(npz_data["goals_world"])[index]
        return traj, rl, goal_world

    raise ValueError("NPZ must contain `trajectory` or `trajectories`.")


def _resolve_xml(xml_path):
    if xml_path:
        return xml_path
    default_xml, _ = get_mj_xml_paths()
    return default_xml


def _compute_hand_distances(traj, xml_path):
    try:
        import mujoco
    except ImportError as e:
        raise RuntimeError("MuJoCo is required for hand-object distance plotting.") from e

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    left_candidates = ["left_wrist_roll_link", "left_wrist_pitch_link", "left_hand"]
    right_candidates = ["right_wrist_roll_link", "right_wrist_pitch_link", "right_hand"]

    def find_body(candidates):
        for name in candidates:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return bid, name
        raise ValueError(f"None of body names found: {candidates}")

    left_id, left_name = find_body(left_candidates)
    right_id, right_name = find_body(right_candidates)

    left_d = np.zeros(traj.shape[0], dtype=np.float64)
    right_d = np.zeros(traj.shape[0], dtype=np.float64)

    for t in range(traj.shape[0]):
        data.qpos[:7] = traj[t, :7]
        data.qpos[7:36] = traj[t, 7:36]
        mujoco.mj_kinematics(model, data)
        obj_pos = traj[t, 36:39]
        lw = data.xpos[left_id].copy()
        rw = data.xpos[right_id].copy()
        left_d[t] = np.linalg.norm(lw - obj_pos)
        right_d[t] = np.linalg.norm(rw - obj_pos)

    return left_d, right_d, left_name, right_name


def main():
    parser = argparse.ArgumentParser(description="Plot trajectory diagnostics from NPZ")
    parser.add_argument("--npz_path", required=True, type=str)
    parser.add_argument("--traj_index", type=int, default=0,
                        help="Trajectory index when NPZ contains a batch")
    parser.add_argument("--xml_path", type=str, default=None,
                        help="MuJoCo XML path for FK hand-distance computation")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--save_path", type=str, default=None,
                        help="Output plot path (default: <npz_stem>_plots.png)")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    traj, real_len, goal_world = _pick_trajectory(data, args.traj_index)

    if traj.ndim != 2 or traj.shape[1] < 43:
        raise ValueError("Trajectory must have shape (T, 43+).")

    T = traj.shape[0]
    if real_len is None:
        real_len = T
    real_len = max(1, min(int(real_len), T))
    traj = traj[:real_len]

    t = np.arange(real_len) * args.dt

    robot_xyz = traj[:, :3]
    robot_quat = traj[:, 3:7]
    obj_xyz = traj[:, 36:39]

    yaw0 = float(yaw_from_quat(robot_quat[0]))
    R_w2l = yaw_to_rot_matrix(-yaw0)
    origin_xy = robot_xyz[0, :2]

    def to_local_xy(xy_world):
        p3 = np.zeros((xy_world.shape[0], 3), dtype=np.float64)
        p3[:, :2] = xy_world - origin_xy[None, :]
        return (R_w2l @ p3.T).T[:, :2]

    robot_xy_l = to_local_xy(robot_xyz[:, :2])
    obj_xy_l = to_local_xy(obj_xyz[:, :2])

    goal_xy_l = None
    if goal_world is not None and len(goal_world) >= 2:
        g = np.zeros((1, 3), dtype=np.float64)
        g[0, :2] = np.asarray(goal_world[:2])
        g[:, :2] -= origin_xy[None, :]
        goal_xy_l = (R_w2l @ g.T).T[0, :2]

    yaw_series = np.asarray(yaw_from_quat(robot_quat))
    yaw_rel = (yaw_series - yaw0 + np.pi) % (2 * np.pi) - np.pi

    xml_path = _resolve_xml(args.xml_path)
    left_d, right_d, left_name, right_name = _compute_hand_distances(traj, xml_path)
    mean_d = 0.5 * (left_d + right_d)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    ax_xy = axs[0, 0]
    ax_xy.plot(obj_xy_l[:, 0], obj_xy_l[:, 1], label="object XY", lw=2)
    ax_xy.plot(robot_xy_l[:, 0], robot_xy_l[:, 1], label="robot XY", lw=2)
    ax_xy.scatter(obj_xy_l[0, 0], obj_xy_l[0, 1], c="green", marker="o", label="object start")
    ax_xy.scatter(robot_xy_l[0, 0], robot_xy_l[0, 1], c="black", marker="x", label="robot start")
    if goal_xy_l is not None:
        ax_xy.scatter(goal_xy_l[0], goal_xy_l[1], c="red", marker="*", s=180, label="goal")
    ax_xy.set_title("XY trajectory in initial yaw-rotated frame")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8)

    ax_yaw = axs[0, 1]
    ax_yaw.plot(t, yaw_rel, lw=2)
    ax_yaw.set_title("Robot yaw vs time (initial-yaw frame)")
    ax_yaw.set_xlabel("time (s)")
    ax_yaw.set_ylabel("yaw (rad)")
    ax_yaw.grid(True, alpha=0.3)

    ax_z = axs[1, 0]
    ax_z.plot(t, obj_xyz[:, 2], label="object z", lw=2)
    ax_z.plot(t, robot_xyz[:, 2], label="robot z", lw=2)
    ax_z.set_title("Object and robot z vs time")
    ax_z.set_xlabel("time (s)")
    ax_z.set_ylabel("z (m)")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(fontsize=8)

    ax_d = axs[1, 1]
    ax_d.plot(t, left_d, label=f"{left_name} → object", lw=1.8)
    ax_d.plot(t, right_d, label=f"{right_name} → object", lw=1.8)
    ax_d.plot(t, mean_d, label="mean hand distance", lw=2.2)
    ax_d.set_title("Robot hands to object distance vs time")
    ax_d.set_xlabel("time (s)")
    ax_d.set_ylabel("distance (m)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=8)

    fig.tight_layout()

    if args.save_path is None:
        stem = os.path.splitext(args.npz_path)[0]
        args.save_path = f"{stem}_plots.png"
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    fig.savefig(args.save_path, dpi=160)
    print(f"Saved plot: {args.save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

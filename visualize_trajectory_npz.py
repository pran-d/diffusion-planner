#!/usr/bin/env python3
"""
Visualize saved trajectory NPZ files in MuJoCo (interactive or video).
"""

import argparse
import os

import numpy as np

from config.configure import get_mj_xml_paths
from utils.visualize.visualize import MjVisualizer


def _pick_trajectory(npz_data, index):
    if "trajectory" in npz_data:
        traj = np.asarray(npz_data["trajectory"])
        if traj.ndim == 3:
            traj = traj[index]
        goal = None
        if "goal_world" in npz_data:
            goal = np.asarray(npz_data["goal_world"]).reshape(-1)
        elif "goals_world" in npz_data:
            goal = np.asarray(npz_data["goals_world"])[index]
        real_len = None
        if "real_length" in npz_data:
            real_len = int(np.asarray(npz_data["real_length"]).reshape(-1)[0])
        elif "real_lengths" in npz_data:
            rr = np.asarray(npz_data["real_lengths"]).reshape(-1)
            real_len = int(rr[min(index, len(rr) - 1)])
        return traj, goal, real_len

    if "trajectories" in npz_data:
        traj = np.asarray(npz_data["trajectories"])[index]
        goal = np.asarray(npz_data["goals_world"])[index] if "goals_world" in npz_data else None
        real_len = int(np.asarray(npz_data["real_lengths"])[index]) if "real_lengths" in npz_data else None
        return traj, goal, real_len

    raise ValueError("NPZ must contain `trajectory` or `trajectories`.")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo visualization for trajectory NPZ")
    parser.add_argument("--npz_path", required=True, type=str)
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--xml_path", type=str, default=None,
                        help="Path to MuJoCo model XML (default from config)")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)
    args = parser.parse_args()

    bundle = np.load(args.npz_path, allow_pickle=True)
    traj, goal_world, real_len = _pick_trajectory(bundle, args.traj_index)

    if real_len is not None:
        real_len = max(1, min(int(real_len), traj.shape[0]))
        traj = traj[:real_len]

    xml_path = args.xml_path
    if xml_path is None:
        xml_path, _ = get_mj_xml_paths()
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    vis = MjVisualizer(xml_path, close_on_enter=False)
    t = np.arange(traj.shape[0]) * args.dt

    if args.video:
        out = args.video_path or os.path.splitext(args.npz_path)[0] + "_viz.mp4"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        vis.render_trajectory_to_video(t=t, x_traj=traj, save_path=out)
        print(f"Saved video: {out}")
    else:
        guidance_vec = None
        if goal_world is not None:
            guidance_vec = np.zeros((traj.shape[0], 3), dtype=np.float64)
            for i in range(traj.shape[0]):
                guidance_vec[i] = goal_world[:3] - traj[i, 36:39]
        vis.visualize_trajectory(
            t=t,
            x_traj=traj,
            repeat=True,
            guidance_vec=guidance_vec,
            goal_pos=goal_world,
        )

    vis.close()


if __name__ == "__main__":
    main()

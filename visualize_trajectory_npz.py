#!/usr/bin/env python3
"""
Visualize saved trajectory NPZ files in MuJoCo (interactive or video).
"""

import argparse
import glob
import os
import re
import tempfile

import numpy as np

from config.configure import get_mj_xml_paths
from utils.visualize.visualize import MjVisualizer, DiffusionOverlayVisualizer


def _load_trajectories_bundle(npz_data):
    """
    Return trajectories/goals/lengths with a unified shape:
      trajectories: (N, T, D)
      goals_world:  (N, 3) or None
      real_lengths: (N,) or None
    """
    if "trajectories" in npz_data:
        trajectories = np.asarray(npz_data["trajectories"])
        goals_world = np.asarray(npz_data["goals_world"]) if "goals_world" in npz_data else None
        real_lengths = np.asarray(npz_data["real_lengths"]).astype(int) if "real_lengths" in npz_data else None
        return trajectories, goals_world, real_lengths

    if "trajectory" in npz_data:
        traj = np.asarray(npz_data["trajectory"])
        trajectories = traj if traj.ndim == 3 else traj[None, ...]

        if "goals_world" in npz_data:
            goals_world = np.asarray(npz_data["goals_world"])
        elif "goal_world" in npz_data:
            g = np.asarray(npz_data["goal_world"]).reshape(-1)
            goals_world = np.repeat(g[None, :], trajectories.shape[0], axis=0)
        else:
            goals_world = None

        if "real_lengths" in npz_data:
            real_lengths = np.asarray(npz_data["real_lengths"]).reshape(-1).astype(int)
        elif "real_length" in npz_data:
            rl = int(np.asarray(npz_data["real_length"]).reshape(-1)[0])
            real_lengths = np.full((trajectories.shape[0],), rl, dtype=int)
        else:
            real_lengths = None

        return trajectories, goals_world, real_lengths

    raise ValueError("NPZ must contain `trajectory` or `trajectories`.")


def _load_trajectories_from_folder(folder_path):
    """
    Load per-trajectory files saved by batch_goal_sweep:
      trajectory_000.npz, trajectory_001.npz, ...
    Returns trajectories/goals/real_lengths with unified shapes.
    """
    pattern = os.path.join(folder_path, "trajectory_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")

    traj_list = []
    goal_list = []
    rl_list = []

    max_t = 0
    for fp in files:
        npz = np.load(fp, allow_pickle=True)
        if "trajectory" not in npz:
            continue
        tr = np.asarray(npz["trajectory"])
        if tr.ndim != 2:
            continue
        traj_list.append(tr)
        max_t = max(max_t, tr.shape[0])

        if "goal_world" in npz:
            goal_list.append(np.asarray(npz["goal_world"]).reshape(-1))
        else:
            goal_list.append(None)

        if "real_length" in npz:
            rl = int(np.asarray(npz["real_length"]).reshape(-1)[0])
        elif "real_lengths" in npz:
            rl = int(np.asarray(npz["real_lengths"]).reshape(-1)[0])
        else:
            rl = tr.shape[0]
        rl_list.append(rl)

    if not traj_list:
        raise ValueError(f"No valid trajectory arrays found in {folder_path}")

    # Pad to common T for overlay viewer.
    d = traj_list[0].shape[1]
    n = len(traj_list)
    trajectories = np.zeros((n, max_t, d), dtype=traj_list[0].dtype)
    for i, tr in enumerate(traj_list):
        t = tr.shape[0]
        trajectories[i, :t, :] = tr
        if t < max_t:
            trajectories[i, t:, :] = tr[t - 1:t, :]

    if all(g is not None for g in goal_list):
        goals_world = np.stack(goal_list, axis=0)
    else:
        goals_world = None

    real_lengths = np.asarray(rl_list, dtype=int)
    return trajectories, goals_world, real_lengths


def _build_guidance_vecs(obj_traj_w, goals_arr):
    """Per-frame unit direction from current object pos to goal. (N, T, 3)"""
    n, t, _ = obj_traj_w.shape
    vecs = np.zeros((n, t, 3), dtype=np.float32)
    for i in range(n):
        for k in range(t):
            d = goals_arr[i, :2] - obj_traj_w[i, k, :2]
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                vecs[i, k, :2] = d / norm
    return vecs


def _build_adaptive_xml(base_repeated_xml, num_robots):
    """Create temp repeated-robot XML with exactly num_robots copies."""
    with open(base_repeated_xml, "r", encoding="utf-8") as f:
        lines = f.readlines()

    robot_starts = [
        idx for idx, line in enumerate(lines)
        if re.search(r'body name="pelvis_\d+"', line)
    ]
    if not robot_starts:
        raise ValueError(f"No pelvis_N bodies found in {base_repeated_xml}")

    header = "".join(lines[:robot_starts[0]])
    wb_close_idx = next((i for i, l in enumerate(lines) if "</worldbody>" in l), len(lines))
    block_end = robot_starts[1] if len(robot_starts) >= 2 else wb_close_idx
    template = "".join(lines[robot_starts[0]:block_end])
    footer = "".join(lines[wb_close_idx:])

    robot_blocks = [re.sub(r'_0(?=")', f'_{i}', template) for i in range(num_robots)]
    xml_content = header + "".join(robot_blocks) + footer

    source_dir = os.path.dirname(os.path.abspath(base_repeated_xml))
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix=f"mj_adaptive_{num_robots}r_",
        dir=source_dir,
        delete=False,
    )
    tmp.write(xml_content)
    tmp.flush()
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser(description="MuJoCo visualization for trajectory NPZ")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Path to .npz file OR folder containing trajectory_*.npz files")
    parser.add_argument("--npz_path", type=str, default=None,
                        help="Deprecated alias for --input_path")
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--single", action="store_true",
                        help="Force single-trajectory view even when NPZ contains multiple trajectories")
    parser.add_argument("--xml_path", type=str, default=None,
                        help="Path to MuJoCo model XML (default from config)")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)
    args = parser.parse_args()

    input_path = args.input_path or args.npz_path
    if input_path is None:
        parser.error("Provide --input_path (or --npz_path)")

    if os.path.isdir(input_path):
        trajectories, goals_world, real_lengths = _load_trajectories_from_folder(input_path)
    else:
        bundle = np.load(input_path, allow_pickle=True)
        trajectories, goals_world, real_lengths = _load_trajectories_bundle(bundle)

    xml_path = args.xml_path
    if xml_path is None:
        xml_path, repeated_xml_path = get_mj_xml_paths()
    else:
        _, repeated_xml_path = get_mj_xml_paths()

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    n_traj = trajectories.shape[0]

    # Multi-trajectory overlay mode (batch_goal_sweep-style)
    if n_traj > 1 and not args.single and not args.video and os.path.exists(repeated_xml_path):
        full_traj = trajectories.copy()
        if real_lengths is not None:
            # Keep padded shape but freeze tail after real end for cleaner playback.
            for i in range(n_traj):
                rl = int(max(1, min(real_lengths[i], full_traj.shape[1])))
                full_traj[i, rl:, :] = full_traj[i, rl - 1:rl, :]

        obj_traj_w = full_traj[:, :, 36:43]
        guidance_vec = _build_guidance_vecs(obj_traj_w, goals_world) if goals_world is not None else None

        adaptive_xml = None
        try:
            adaptive_xml = _build_adaptive_xml(repeated_xml_path, n_traj)
            vis = DiffusionOverlayVisualizer(xml_path)
            vis.visualize_overlay(
                x_trajs=full_traj,
                repeated_xml_path=adaptive_xml,
                timestep_delay=args.dt,
                loop=True,
                guidance_vec=guidance_vec,
                world_goal_pos=goals_world,
                spacing=0.0,
            )
        finally:
            if adaptive_xml and os.path.exists(adaptive_xml):
                os.unlink(adaptive_xml)
        return

    # Single trajectory mode (or fallback when repeated XML is unavailable)
    i = int(max(0, min(args.traj_index, n_traj - 1)))
    traj = trajectories[i]
    goal_world = goals_world[i] if goals_world is not None else None
    real_len = int(real_lengths[i]) if real_lengths is not None else None

    if real_len is not None:
        real_len = max(1, min(int(real_len), traj.shape[0]))
        traj = traj[:real_len]

    vis = MjVisualizer(xml_path, close_on_enter=False)
    t = np.arange(traj.shape[0]) * args.dt

    if args.video:
        if os.path.isdir(input_path):
            default_video = os.path.join(input_path, "folder_viz.mp4")
        else:
            default_video = os.path.splitext(input_path)[0] + "_viz.mp4"
        out = args.video_path or default_video
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

#!/usr/bin/env python3
"""
batch_goal_sweep.py

Takes initial conditions from a single dataset trajectory, generates N goals
arranged on 3 concentric circles (with no two goals sharing the same angle),
and runs inference for all goals via MotionGenerator.  The resulting
trajectories are visualised in a single MuJoCo window with per-trajectory
goal markers and goal-direction arrows.

Example usage:
    python batch_goal_sweep.py --epoch 200 --traj_idx 0 --batch_idx 0 \
        --num_goals 12 --goal_spread 0.5 --last_frame_waypoint --visualize
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch

from config.configure import load_config, get_data_path, get_norm_path, get_mj_xml_paths
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
from utils.visualize.visualize import MjVisualizer, DiffusionOverlayVisualizer
from motion_generator import MotionGenerator
from utils.inference_config import load_inference_defaults
from utils.inference_utils import DEFAULT_LIFT_HEIGHT
from utils.data.style_conditioning import style_name_to_id
from inference_mg import extract_initial_condition


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser("Batch Goal Sweep -- single IC, N goals on 3 rings")
    parser.add_argument("--inference_config", type=str, default="config/inference.yaml",
                        help="Path to inference defaults YAML")

    # Checkpoint
    parser.add_argument("--epoch", type=str, default=None,
                        help="Checkpoint epoch (int) or direct file path")
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA weights for inference")

    # Initial condition selection
    parser.add_argument("--traj_idx",   type=int, default=0,
                        help="Trajectory (file) index")
    parser.add_argument("--batch_idx",  type=int, default=0,
                        help="Batch index within file")
    parser.add_argument("--start_time", type=int, default=0,
                        help="Window start timestep")

    # Goal sweep
    parser.add_argument("--num_goals",       type=int,   default=12,
                        help="Number of goals on the concentric circles (default: 12)")
    parser.add_argument("--goal_spread",     type=float, default=0.5,
                        help="Outer-ring radius in metres (default: 0.5)")
    parser.add_argument("--goal_z_offset",   type=float, default=0.0,
                        help="Constant z offset (m) added to generated goals")
    parser.add_argument("--goal_multiplier", type=float, default=1.0,
                        help="Scale every goal displacement by this factor")
    parser.add_argument("--include_original", action="store_true",
                        help="Prepend the dataset's original goal as goal[0]")
    parser.add_argument("--seed",            type=int,   default=42)

    # Inference
    parser.add_argument("--stitch_steps", type=int, default=None,
                        help="Autoregressive segments (auto-computed if not set)")
    parser.add_argument("--cfg_w",        type=float, default=1.0)
    parser.add_argument("--device",       type=str,   default="cuda")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Goals per inference batch (default: all at once)")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Enable fp16 autocast during inference")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Enable torch.compile() for inference model")
    parser.add_argument("--enable_goal_stop", action="store_true")
    parser.add_argument("--goal_stop_threshold", type=float, default=0.1)
    parser.add_argument("--enable_physics_stop", action="store_true")
    parser.add_argument("--enable_phys_clamp",   action="store_true")
    parser.add_argument("--style", type=str, default=None, choices=["pick", "push", "kick"],
                        help="Use one fixed style for all goals")
    parser.add_argument("--random_styles", action="store_true",
                        help="Randomly allocate style per goal (pick/push/kick)")

    # Waypoint / z-profile
    parser.add_argument("--last_frame_waypoint", action="store_true")
    parser.add_argument("--arrival_ratio", type=float, default=0.85)
    parser.add_argument("--lift_height",   type=float, default=DEFAULT_LIFT_HEIGHT)
    parser.add_argument("--lift_start",    type=float, default=0.0)
    parser.add_argument("--lift_end",      type=float, default=0.20)
    parser.add_argument("--walk_start_z",  type=float, default=0.80)
    parser.add_argument("--no_lower_dist", type=float, default=0.4)

    # Output
    parser.add_argument("--save_dir", type=str, default="results/goal_sweep",
                        help="Directory to save results into")
    parser.add_argument("--metrics_log_path", type=str, default="results/inference_metrics.jsonl",
                        help="Append per-goal metrics and generation timing as JSONL")
    parser.add_argument("--visualize", action="store_true",
                        help="Open MuJoCo viewer after generation")
    parser.add_argument("--video", action="store_true",
                        help="Save visualization as video file instead of opening interactive viewer")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to save video file (default: results/goal_sweep_video.mp4)")

    cfg_defaults, _ = load_inference_defaults(
        argv=argv,
        section="batch_goal_sweep",
        default_path="config/inference.yaml",
    )
    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    args = parser.parse_args(argv)
    if args.epoch is None:
        parser.error("Missing `--epoch`. Set it via CLI or config/inference.yaml::batch_goal_sweep.epoch")
    if args.style is not None and args.random_styles:
        parser.error("Use either --style or --random_styles, not both")
    return args


# --------------------------------------------------------------------------- #
# Goal generation -- 3 concentric rings, alternating radii
# --------------------------------------------------------------------------- #

def generate_goals_3ring(ref_obj_pos, num_goals, goal_spread, z_height=None):
    """
    Distribute *num_goals* evenly in angle around ref_obj_pos, with each
    successive angle placed on an alternating radius so the pattern cycles
    inner → mid → outer → inner → …

    Radii: r_inner = spread/3, r_mid = 2*spread/3, r_outer = spread.
    Angles are uniformly spaced over [0, 2*pi).

    Args:
        ref_obj_pos : (3,) centre point (world-frame object position at t=0)
        num_goals   : int, total goals to place
        goal_spread : float, outer-ring radius in metres
        z_height    : optional fixed z for all goals (default: ref_obj_pos[2])

    Returns:
        goals : (N, 3) world-frame goal positions
    """
    if z_height is None:
        z_height = ref_obj_pos[2]

    radii = [goal_spread / 3.0, 2.0 * goal_spread / 3.0, goal_spread]

    # Uniformly spaced angles
    thetas = np.linspace(0.0, 2.0 * np.pi, num_goals, endpoint=False)

    goals = []
    for i, theta in enumerate(thetas):
        r = radii[i % len(radii)]  # cycle inner → mid → outer → …
        g = np.array([
            ref_obj_pos[0] + r * np.cos(theta),
            ref_obj_pos[1] + r * np.sin(theta),
            z_height,
        ])
        goals.append(g)

    return np.stack(goals)  # (N, 3)


# --------------------------------------------------------------------------- #
# Adaptive XML generation
# --------------------------------------------------------------------------- #

def build_adaptive_xml(base_repeated_xml, num_robots):
    """
    Generate a temporary MuJoCo XML with exactly *num_robots* robots by
    stamping copies of robot-0 from *base_repeated_xml*.
    Written into the same directory so relative meshdir resolves correctly.
    """
    with open(base_repeated_xml) as f:
        lines = f.readlines()

    robot_starts = [
        idx for idx, line in enumerate(lines)
        if re.search(r'body name="pelvis_\d+"', line)
    ]
    if not robot_starts:
        raise ValueError(f"No pelvis_N bodies found in {base_repeated_xml}")

    header = ''.join(lines[: robot_starts[0]])
    wb_close_idx = next(
        (i for i, l in enumerate(lines) if '</worldbody>' in l), len(lines)
    )
    block_end = robot_starts[1] if len(robot_starts) >= 2 else wb_close_idx
    template = ''.join(lines[robot_starts[0]: block_end])
    footer = ''.join(lines[wb_close_idx:])

    robot_blocks = [re.sub(r'_0(?=")', f'_{i}', template) for i in range(num_robots)]
    xml_content = header + ''.join(robot_blocks) + footer

    source_dir = os.path.dirname(os.path.abspath(base_repeated_xml))
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.xml',
        prefix=f'mj_adaptive_{num_robots}r_',
        dir=source_dir,
        delete=False,
    )
    tmp.write(xml_content)
    tmp.flush()
    tmp.close()
    return tmp.name


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #

def _build_guidance_vecs(obj_traj_w, goals_arr):
    """Per-frame unit direction from current object pos to goal. (N, T, 3)"""
    N, T, _ = obj_traj_w.shape
    vecs = np.zeros((N, T, 3), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            d = goals_arr[i, :2] - obj_traj_w[i, t, :2]
            n = np.linalg.norm(d)
            if n > 1e-6:
                vecs[i, t, :2] = d / n
    return vecs


def _save_overlay_video(
    trajectories,
    real_lengths,
    goals_world,
    out_path,
    single_xml_path,
    repeated_xml_path,
    fps=100,
    width=1280,
    height=720,
):
    """Save one MP4 with all trajectories overlaid in a repeated-robot scene.

    Rendering is done in a subprocess so GL backend can be selected before
    importing MuJoCo (important on headless systems).
    """
    x = np.asarray(trajectories)
    rl = np.asarray(real_lengths, dtype=np.int64)
    gw = np.asarray(goals_world, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected trajectories shape (N, T, D), got {x.shape}")
    if gw.ndim != 2 or gw.shape[0] != x.shape[0] or gw.shape[1] < 3:
        raise ValueError(f"Expected goals_world shape (N, 3+), got {gw.shape}")

    with tempfile.TemporaryDirectory(prefix="goal_sweep_overlay_render_") as td:
        payload = os.path.join(td, "payload.npz")
        np.savez_compressed(payload, trajectories=x, real_lengths=rl, goals_world=gw[:, :3])

        render_code = r'''
import os, re, sys, tempfile
import numpy as np
import imageio
import mujoco

payload, out_path, single_xml, repeated_xml, fps, width, height = sys.argv[1:]
fps = int(fps); width = int(width); height = int(height)
arr = np.load(payload)
x = arr["trajectories"]
rl = arr["real_lengths"].astype(np.int64)
gw = arr["goals_world"]

single_model = mujoco.MjModel.from_xml_path(single_xml)
nq_single = int(single_model.nq)
num_robots = int(x.shape[0])

with open(repeated_xml, "r", encoding="utf-8") as f:
    lines = f.readlines()
robot_starts = [i for i, line in enumerate(lines) if re.search(r'body name="pelvis_\d+"', line)]
if not robot_starts:
    raise ValueError(f"No pelvis_N bodies found in {repeated_xml}")
header = "".join(lines[: robot_starts[0]])
wb_close_idx = next((i for i, l in enumerate(lines) if "</worldbody>" in l), len(lines))
block_end = robot_starts[1] if len(robot_starts) >= 2 else wb_close_idx
template = "".join(lines[robot_starts[0]: block_end])
footer = "".join(lines[wb_close_idx:])
robot_blocks = [re.sub(r'_0(?=")', f'_{i}', template) for i in range(num_robots)]
xml_content = header + "".join(robot_blocks) + footer

src_dir = os.path.dirname(os.path.abspath(repeated_xml))
fd, adaptive_xml = tempfile.mkstemp(prefix=f"goal_sweep_overlay_{num_robots}r_", suffix=".xml", dir=src_dir)
os.close(fd)
with open(adaptive_xml, "w", encoding="utf-8") as f:
    f.write(xml_content)

renderer = None
try:
    model = mujoco.MjModel.from_xml_path(adaptive_xml)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    # Remove distance haze/fog for clearer videos.
    try:
        model.vis.map.haze = 0.0
    except Exception:
        pass
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    except Exception:
        pass
    try:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    except Exception:
        pass
    try:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    except Exception:
        pass

    T_max = int(x.shape[1])
    nq_used = num_robots * nq_single
    if nq_used > model.nq:
        raise ValueError(f"Repeated XML nq={model.nq} is smaller than required {nq_used}")

    # Camera framing: center on goal/end-point cloud and zoom out enough so
    # all robots and markers remain visible in the final phase.
    end_obj = []
    for i in range(num_robots):
        ti_end = int(max(1, rl[i])) - 1
        end_obj.append(x[i, ti_end, 36:39])
    end_obj = np.asarray(end_obj, dtype=np.float64)
    pts = np.concatenate([gw[:, :3], end_obj], axis=0)
    center = np.mean(pts, axis=0)
    r_xy = np.linalg.norm(pts[:, :2] - center[:2], axis=1)
    radius = float(np.max(r_xy)) if len(r_xy) else 1.0

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [float(center[0]), float(center[1]), float(center[2] + 0.8)]
    cam.distance = max(1.5, 0.8 + 1.8 * radius)
    cam.azimuth = 135.0
    cam.elevation = -16.0

    # Optional mocap markers/arrows if present in repeated XML.
    goal_mocap_ids = []
    arrow_mocap_ids = []
    for i in range(num_robots):
        bid_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"goal_marker_{i}")
        goal_mocap_ids.append(model.body_mocapid[bid_g] if bid_g >= 0 else -1)

        bid_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"guidance_arrow_{i}")
        arrow_mocap_ids.append(model.body_mocapid[bid_a] if bid_a >= 0 else -1)

    def quat_from_z_to_vec(v):
        # Return wxyz quaternion rotating +Z axis to vector v.
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        t = v / n
        cross = np.cross(z, t)
        dot = float(np.dot(z, t))
        cn = float(np.linalg.norm(cross))
        if cn < 1e-9:
            if dot > 0.0:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        axis = cross / cn
        angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        half = 0.5 * angle
        s = float(np.sin(half))
        return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for t in range(T_max):
            q_blocks = []
            for i in range(num_robots):
                ti = min(t, int(max(1, rl[i])) - 1)
                q_blocks.append(x[i, ti, :nq_single])

                # Goal marker (green box) at fixed goal position.
                gm = goal_mocap_ids[i]
                if gm >= 0:
                    data.mocap_pos[gm] = gw[i, :3]

                # Guidance arrow from current object position towards goal.
                am = arrow_mocap_ids[i]
                if am >= 0:
                    obj = x[i, ti, 36:39]
                    d = gw[i, :3] - obj
                    q = quat_from_z_to_vec(d)
                    data.mocap_pos[am] = obj
                    data.mocap_quat[am] = q

            data.qpos[:nq_used] = np.concatenate(q_blocks, axis=0)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            writer.append_data(frame)
finally:
    try:
        if renderer is not None:
            renderer.close()
    except Exception:
        pass
    try:
        if os.path.exists(adaptive_xml):
            os.unlink(adaptive_xml)
    except Exception:
        pass
'''

        errors = []
        for backend, pyopengl in (("egl", "egl"), ("osmesa", "osmesa")):
            env = os.environ.copy()
            env["MUJOCO_GL"] = backend
            env["PYOPENGL_PLATFORM"] = pyopengl
            env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
            env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
            env.pop("DISPLAY", None)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    render_code,
                    payload,
                    out_path,
                    single_xml_path,
                    repeated_xml_path,
                    str(int(fps)),
                    str(int(width)),
                    str(int(height)),
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                return
            errors.append(
                f"backend={backend}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        raise RuntimeError("Overlay render failed for EGL and OSMesa backends.\n" + "\n---\n".join(errors))


def visualize_results(full_traj, obj_traj_w, goals_arr, real_lengths, args):
    """Visualise all N trajectories side-by-side in MuJoCo."""
    N, T, _ = full_traj.shape
    xml_path, repeated_xml_path = get_mj_xml_paths()
    guidance_vecs = _build_guidance_vecs(obj_traj_w, goals_arr)

    # Video path: mirror run_inference_ablations behavior and avoid creating
    # any GLFW viewer in the parent process (headless-safe).
    if getattr(args, 'video', False):
        video_path = getattr(args, 'video_path', None) or os.path.join(args.save_dir, "videos", "overlaid_robots.mp4")
        if os.path.exists(xml_path) and os.path.exists(repeated_xml_path):
            _save_overlay_video(
                trajectories=full_traj,
                real_lengths=real_lengths,
                goals_world=goals_arr,
                out_path=video_path,
                single_xml_path=xml_path,
                repeated_xml_path=repeated_xml_path,
                fps=100,
                width=1280,
                height=720,
            )
            print(f"[video] Saved {video_path}")
        else:
            print(
                f"[video] Missing XML for overlay video (single={xml_path}, repeated={repeated_xml_path}); skipping."
            )
        return

    if os.path.exists(repeated_xml_path):
        adaptive_xml = None
        try:
            adaptive_xml = build_adaptive_xml(repeated_xml_path, N)
            vis = DiffusionOverlayVisualizer(xml_path)
            vis.visualize_overlay(
                x_trajs=full_traj,
                repeated_xml_path=adaptive_xml,
                timestep_delay=0.01,
                loop=True,
                guidance_vec=guidance_vecs,
                world_goal_pos=goals_arr,
                spacing=0.0,
            )
        finally:
            if adaptive_xml and os.path.exists(adaptive_xml):
                os.unlink(adaptive_xml)
        return

    if not os.path.exists(xml_path):
        print("No mj_model.xml found -- skipping visualisation.")
        return

    overlay_paths = [obj_traj_w[i, :, :3] for i in range(N)]
    markers = [np.tile(goals_arr[i:i+1], (T, 1)) for i in range(N)]
    t_arr = np.arange(T) * 0.01
    vis = MjVisualizer(xml_path, close_on_enter=False)

    vis.visualize_trajectory(
        t=t_arr, x_traj=full_traj[0], repeat=True,
        guidance_vec=guidance_vecs[0], goal_pos=goals_arr[0],
        overlay_paths=overlay_paths, markers=markers,
    )
    vis.close()


# --------------------------------------------------------------------------- #
# Core sweep function (importable by evaluate_ablations.py)
# --------------------------------------------------------------------------- #

def run_goal_sweep(mg, dataset, args):
    """
    Run the full goal sweep using a pre-initialised MotionGenerator & dataset.

    Returns a dict with keys:
        full_traj, goals_world, goals_local, errors, real_lengths, ref_obj_pos,
        desired_displacements, achieved_displacements
    """
    data_cfg = mg.data_cfg

    # -- Extract initial condition -----------------------------------------
    target = (args.traj_idx, args.batch_idx, args.start_time)
    try:
        sample_idx = dataset.indices.index(target)
    except ValueError:
        raise ValueError(f"Index tuple {target} not in dataset.indices")

    num_task_params = data_cfg.get("num_task_params", 3)
    local_goal_dim = min(3, num_task_params)
    initial_condition, goal_local_orig, anchor, file_idx, _, _, _ = \
        extract_initial_condition(dataset, sample_idx, local_goal_dim)

    ref_obj_pos = initial_condition['obj'][0, :3].copy()
    init_base_quat = initial_condition['robot'][0, 3:7]
    yaw = yaw_from_quat(init_base_quat)
    R_local_to_world = yaw_to_rot_matrix(yaw)
    R_world_to_local = yaw_to_rot_matrix(-yaw)

    # -- Generate goals ----------------------------------------------------
    goals_world = generate_goals_3ring(
        ref_obj_pos,
        args.num_goals,
        args.goal_spread,
        z_height=ref_obj_pos[2] + float(getattr(args, 'goal_z_offset', 0.0)),
    )
    if getattr(args, 'include_original', False):
        g0_3d = np.zeros(3, dtype=np.float64)
        g0_3d[:len(goal_local_orig)] = goal_local_orig
        g0_world = (R_local_to_world @ g0_3d[:, None])[:, 0] + ref_obj_pos
        goals_world = np.concatenate([g0_world[None], goals_world], axis=0)

    N = len(goals_world)

    # -- Optional style allocation ----------------------------------------
    style_id_to_name = {0: "pick", 1: "push", 2: "kick"}
    goal_style_ids = None
    goal_style_names = None
    if getattr(args, 'random_styles', False):
        rng = np.random.default_rng(getattr(args, 'seed', 42))
        goal_style_ids = rng.integers(0, 3, size=N, dtype=np.int64)
        goal_style_names = np.array([style_id_to_name[int(s)] for s in goal_style_ids], dtype=object)
    elif getattr(args, 'style', None) is not None:
        sid = int(style_name_to_id(args.style))
        goal_style_ids = np.full((N,), sid, dtype=np.int64)
        goal_style_names = np.array([style_id_to_name[sid]] * N, dtype=object)

    # Convert to local displacements
    goals_local = np.zeros((N, local_goal_dim), dtype=np.float64)
    for i in range(N):
        delta_w = goals_world[i, :3] - ref_obj_pos[:3]
        delta_l = (R_world_to_local @ delta_w[:, None])[:, 0]
        goals_local[i] = delta_l[:local_goal_dim]

    if args.goal_multiplier != 1.0:
        goals_local *= args.goal_multiplier
        for i in range(N):
            g3d = np.zeros(3, dtype=np.float64)
            g3d[:local_goal_dim] = goals_local[i]
            goals_world[i] = (R_local_to_world @ g3d[:, None])[:, 0] + ref_obj_pos

    # -- Auto stitch_steps -------------------------------------------------
    stitch_steps = args.stitch_steps
    if stitch_steps is None:
        _eff = max(1, data_cfg["num_timesteps"] - data_cfg.get("state_history", 1))
        traj_len = dataset.traj_lengths[file_idx]
        base_steps = max(1, math.ceil(traj_len / _eff))
        max_dist = max(np.linalg.norm(goals_local[i, :2]) for i in range(N))
        orig_dist = np.linalg.norm(goal_local_orig[:2])
        ref_dist = orig_dist if orig_dist > 0.1 else args.goal_spread
        m_per_step = max(ref_dist / base_steps, 0.05)
        stitch_steps = max(base_steps, math.ceil(max_dist / m_per_step))

    _eff_horizon = max(1, data_cfg["num_timesteps"] - data_cfg.get("state_history", 1))
    planning_horizon_s = _eff_horizon * data_cfg.get("raw_dt", 0.01) * data_cfg.get("downsample", 1)

    # -- Run inference (batched) -------------------------------------------
    batch_size = getattr(args, 'batch_size', None) or N  # default: all at once
    all_trajs, all_real_lengths = [], []
    per_goal_time_s = []
    max_window_gen_time = 0.0

    for b_start in range(0, N, batch_size):
        b_end = min(b_start + batch_size, N)
        bs = b_end - b_start
        print(f"  Batch [{b_start}:{b_end}] / {N}  (bs={bs})")

        # Tile initial condition to match this sub-batch
        ic_batch = {
            'robot': np.tile(initial_condition['robot'], (bs, 1, 1)),  # (bs, H, 36)
            'obj':   np.tile(initial_condition['obj'],   (bs, 1, 1)),  # (bs, H, 7)
        }

        t0 = time.perf_counter()
        traj, rl, timing_stats = mg.generate_trajectory(
            initial_condition=ic_batch,
            goal_condition=goals_local[b_start:b_end],  # (bs, D)
            style_condition=(goal_style_ids[b_start:b_end] if goal_style_ids is not None else None),
            stitch_steps=stitch_steps,
            num_samples=1,
            cfg_w=getattr(args, 'cfg_w', 1.0),
            enable_goal_stop=getattr(args, 'enable_goal_stop', False),
            end_error_threshold=getattr(args, 'goal_stop_threshold', 0.1),
            enable_physics_stop=getattr(args, 'enable_physics_stop', False),
            enable_physics_clamp=getattr(args, 'enable_phys_clamp', False),
            use_last_frame_wp=getattr(args, 'last_frame_waypoint', True),
            arrival_ratio=getattr(args, 'arrival_ratio', 0.85),
            lift_height=getattr(args, 'lift_height', DEFAULT_LIFT_HEIGHT),
            lift_start=getattr(args, 'lift_start', 0.0),
            lift_end=getattr(args, 'lift_end', 0.20),
            walk_start_z=getattr(args, 'walk_start_z', 0.80),
            no_lower_dist=getattr(args, 'no_lower_dist', 0.4),
            use_fp16=bool(getattr(args, 'use_fp16', False)),
            compile_model=bool(getattr(args, 'torch_compile', False)),
            return_timing_stats=True,
        )
        elapsed = time.perf_counter() - t0
        per_window_gen_time = timing_stats["per_window_gen_time_s"]
        # traj: (bs, T, 43),  rl: (bs,)
        for j in range(bs):
            all_trajs.append(traj[j])
            all_real_lengths.append(int(rl[j]))
            per_goal_time_s.append(np.mean(per_window_gen_time))
            max_window_gen_time = max(max_window_gen_time, np.max(per_window_gen_time))

    full_traj = np.stack(all_trajs)  # (N, T, 43)
    real_lengths = np.array(all_real_lengths)

    # -- Compute errors & displacements ------------------------------------
    errors = []
    desired_displacements = []
    achieved_displacements = []
    for i in range(N):
        rl_i = real_lengths[i]
        final_pos = full_traj[i, rl_i - 1, 36:39]
        goal_xyz = goals_world[i, :3]
        err = np.linalg.norm(goal_xyz - final_pos)
        errors.append(err)
        desired_displacements.append(goals_world[i, :3] - ref_obj_pos[:3])
        achieved_displacements.append(final_pos - ref_obj_pos[:3])

    return {
        "full_traj": full_traj,
        "obj_traj_w": full_traj[:, :, 36:43],
        "goals_world": goals_world,
        "goals_local": goals_local,
        "errors": np.array(errors),
        "real_lengths": real_lengths,
        "per_goal_time_s": np.array(per_goal_time_s, dtype=np.float64),
        "max_window_gen_time_s": max_window_gen_time,
        "ref_obj_pos": ref_obj_pos,
        "desired_displacements": np.array(desired_displacements),
        "achieved_displacements": np.array(achieved_displacements),
        "goal_style_ids": goal_style_ids,
        "goal_style_names": goal_style_names,
        "stitch_steps": stitch_steps,
        "planning_horizon_s": planning_horizon_s,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    # -- 1. Build MotionGenerator & load weights ---------------------------
    mg = MotionGenerator(config_path="config/config.yaml", device=device)
    data_cfg = mg.data_cfg

    if os.path.exists(args.epoch):
        mg.diffuser.load_weights_from_file(args.epoch)
    else:
        mg.diffuser.loadWeights(int(args.epoch), ema=args.ema)

    # -- 2. Full dataset (for initial conditions) --------------------------
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(mg.model_cfg, mg.training_cfg, data_cfg)
    data_buffer = preload_dataset(data_cfg, data_path)
    dataset = FlexibleWindowDataset(
        data_buffer=data_buffer, config=data_cfg,
        norm_path=norm_path, calculate_stats=False, training_cfg={},
    )
    mg.dataset = dataset

    # -- 3. Run sweep ------------------------------------------------------
    result = run_goal_sweep(mg, dataset, args)

    full_traj = result["full_traj"]
    goals_world = result["goals_world"]
    errors = result["errors"]
    real_lengths = result["real_lengths"]
    ref_obj_pos = result["ref_obj_pos"]
    N = len(goals_world)
    per_goal_time_s = np.asarray(result.get("per_goal_time_s", np.zeros(N, dtype=np.float64)), dtype=np.float64)
    max_window_gen_time_s = result.get("max_window_gen_time_s", 0.0)
    goal_style_ids = result.get("goal_style_ids", None)
    goal_style_names = result.get("goal_style_names", None)
    stitch_steps = int(max(1, result.get("stitch_steps", 1)))
    planning_horizon_s = float(result.get("planning_horizon_s", 0.0))
    avg_window_generation_time_s = float(np.mean(per_goal_time_s)) if len(per_goal_time_s) else 0.0

    # -- 4. Print summary --------------------------------------------------
    print(f"\nRef obj pos: {ref_obj_pos}")
    print(f"\n{'=' * 70}")
    print("Results:")
    for i in range(N):
        dd = result["desired_displacements"][i]
        ad = result["achieved_displacements"][i]
        style_suffix = ""
        if goal_style_names is not None:
            style_suffix = f"  style={goal_style_names[i]}"
        print(f"  [{i:2d}]  err={errors[i]:.4f}m  "
              f"desired=({dd[0]:+.3f},{dd[1]:+.3f},{dd[2]:+.3f})  "
              f"achieved=({ad[0]:+.3f},{ad[1]:+.3f},{ad[2]:+.3f})  "
              f"T_real={real_lengths[i]}{style_suffix}")
    print(f"\n  Mean error: {np.mean(errors):.4f}m  "
          f"Median: {np.median(errors):.4f}m  Max: {np.max(errors):.4f}m")
    print(f"  Avg window generation time: {avg_window_generation_time_s:.4f}s/window")
    print(f"  Max window generation time: {max_window_gen_time_s:.4f}s/window")
    print(f"  Planning horizon per window: {planning_horizon_s:.4f}s/window")
    print("=" * 70)

    # -- 5. Save -----------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    for key in ["full_traj", "goals_world", "goals_local", "errors",
                "real_lengths", "desired_displacements", "achieved_displacements"]:
        np.save(os.path.join(args.save_dir, f"{key}.npy"), result[key])
    np.save(os.path.join(args.save_dir, "ref_obj_pos.npy"), ref_obj_pos)

    # Aggregate NPZ for portable downstream plotting/visualisation.
    aggregate_npz = os.path.join(args.save_dir, "goal_sweep_results.npz")
    np.savez_compressed(
        aggregate_npz,
        trajectories=result["full_traj"],
        goals_world=result["goals_world"],
        goals_local=result["goals_local"],
        errors=result["errors"],
        real_lengths=result["real_lengths"],
        desired_displacements=result["desired_displacements"],
        achieved_displacements=result["achieved_displacements"],
        per_goal_time_s=per_goal_time_s,
        avg_window_generation_time_s=np.array([avg_window_generation_time_s], dtype=np.float64),
        planning_horizon_s=np.array([planning_horizon_s], dtype=np.float64),
        ref_obj_pos=ref_obj_pos,
        goal_style_ids=goal_style_ids if goal_style_ids is not None else np.array([], dtype=np.int64),
        goal_style_names=(goal_style_names.astype(str) if goal_style_names is not None else np.array([], dtype=str)),
    )

    summary_json = os.path.join(args.save_dir, "summary_metrics.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "avg_window_generation_time_s": avg_window_generation_time_s,
                "planning_horizon_per_window_s": planning_horizon_s,
                "avg_trajectory_generation_time_s": float(np.mean(per_goal_time_s)) if len(per_goal_time_s) else 0.0,
                "stitch_steps": stitch_steps,
                "num_goals": int(N),
                "random_styles": bool(getattr(args, 'random_styles', False)),
                "fixed_style": getattr(args, 'style', None),
                "mean_error_m": float(np.mean(errors)),
                "median_error_m": float(np.median(errors)),
                "max_error_m": float(np.max(errors)),
            },
            f,
            indent=2,
        )

    # Individual trajectory NPZ files.
    for i in range(N):
        traj_npz = os.path.join(args.save_dir, f"trajectory_{i:03d}.npz")
        np.savez_compressed(
            traj_npz,
            trajectory=result["full_traj"][i],
            real_length=np.array([result["real_lengths"][i]], dtype=np.int64),
            goal_world=result["goals_world"][i],
            goal_local=result["goals_local"][i],
            desired_displacement=result["desired_displacements"][i],
            achieved_displacement=result["achieved_displacements"][i],
            error=np.array([result["errors"][i]], dtype=np.float64),
            generation_time_s=np.array([result.get("per_goal_time_s", np.zeros(N))[i]], dtype=np.float64),
            style_id=np.array([int(goal_style_ids[i])], dtype=np.int64) if goal_style_ids is not None else np.array([], dtype=np.int64),
            style_name=np.array([str(goal_style_names[i])], dtype=str) if goal_style_names is not None else np.array([], dtype=str),
        )

    # Per-goal metrics log (JSONL append).
    os.makedirs(os.path.dirname(args.metrics_log_path) or ".", exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    for i in range(N):
        dxyz = result["desired_displacements"][i]
        mag = float(np.linalg.norm(dxyz))
        direction = (dxyz / mag).tolist() if mag > 1e-9 else [0.0, 0.0, 0.0]
        rec = {
            "timestamp_utc": now_iso,
            "script": "batch_goal_sweep.py",
            "checkpoint": args.epoch,
            "ema": bool(args.ema),
            "goal_index": int(i),
            "traj_idx": int(args.traj_idx),
            "batch_idx": int(args.batch_idx),
            "start_time": int(args.start_time),
            "goal_world": result["goals_world"][i].tolist(),
            "goal_local": result["goals_local"][i].tolist(),
            "goal_direction_xyz": direction,
            "goal_magnitude_xyz": mag,
            "style_id": int(goal_style_ids[i]) if goal_style_ids is not None else None,
            "style_name": str(goal_style_names[i]) if goal_style_names is not None else None,
            "error": float(result["errors"][i]),
            "generation_time_s": float(per_goal_time_s[i]),
            "time_per_inference_step_s": float(per_goal_time_s[i]) / stitch_steps,
            "avg_window_generation_time_s": avg_window_generation_time_s,
            "planning_horizon_s": planning_horizon_s,
            "trajectory_path": os.path.join(args.save_dir, f"trajectory_{i:03d}.npz"),
        }
        with open(args.metrics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved to {args.save_dir}/")

    # -- 6. Visualise ------------------------------------------------------
    if args.visualize or args.video:
        visualize_results(full_traj, result["obj_traj_w"], goals_world, real_lengths, args)


if __name__ == "__main__":
    main()

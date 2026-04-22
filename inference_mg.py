"""
Inference script that uses MotionGenerator as the single entry point.

This is the primary standalone inference entry point and delegates all autoregressive loop logic,
waypoint construction, and SBTO reconstruction to MotionGenerator.generate_trajectory().

Usage:
    python inference_mg.py --epoch 100 --traj_idx 5 --batch_idx 0
    python inference_mg.py --epoch results/model_100.pth --task_params 0.5 -0.3
    python inference_mg.py --epoch 100 --ema --stitch_steps 10 --return_analysis
"""

import argparse
import os
import math
import json
import sys
import time
import subprocess
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

from config.configure import get_data_path, get_norm_path, get_mj_xml_paths
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset
from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
from motion_generator import MotionGenerator
from utils.inference_config import load_inference_defaults
from utils.inference_utils import DEFAULT_LIFT_HEIGHT


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _save_single_video_headless(
    trajectory,
    real_length,
    goal_world,
    out_path,
    xml_path,
    fps=100,
    width=1280,
    height=720,
):
    """Save one trajectory video in a subprocess using offscreen MuJoCo rendering.

    This avoids GLFW initialization in headless environments.
    """
    x = np.asarray(trajectory)
    rl = int(real_length)
    gw = np.asarray(goal_world, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected trajectory shape (T, D), got {x.shape}")
    if gw.shape[0] < 3:
        raise ValueError(f"Expected goal_world with at least 3 dims, got {gw.shape}")

    with tempfile.TemporaryDirectory(prefix="infer_mg_render_") as td:
        payload = os.path.join(td, "payload.npz")
        np.savez_compressed(
            payload,
            trajectory=x,
            real_length=np.array([rl], dtype=np.int64),
            goal_world=gw[:3],
        )

        render_code = r'''
import os, sys
import numpy as np
import imageio
import mujoco

payload, out_path, xml_path, fps, width, height = sys.argv[1:]
fps = int(fps); width = int(width); height = int(height)
arr = np.load(payload)
x = arr["trajectory"]
real_length = int(arr["real_length"][0])
goal_world = arr["goal_world"]

if x.ndim != 2:
    raise ValueError(f"Expected trajectory shape (T, D), got {x.shape}")
if real_length <= 0:
    raise ValueError(f"real_length must be > 0, got {real_length}")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
model.vis.global_.offheight = max(model.vis.global_.offheight, height)

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

renderer = None
try:
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

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # Zoomed out camera framing around goal/final object area.
    obj0 = x[0, 36:39]
    objf = x[min(max(0, real_length - 1), x.shape[0] - 1), 36:39]
    center = (goal_world[:3] + obj0 + objf) / 3.0
    r_xy = max(
        float(np.linalg.norm(goal_world[:2] - center[:2])),
        float(np.linalg.norm(obj0[:2] - center[:2])),
        float(np.linalg.norm(objf[:2] - center[:2])),
    )
    cam.lookat[:] = [float(center[0]), float(center[1]), float(center[2] + 0.8)]
    cam.distance = max(2.2, 1.4 + 2.2 * r_xy)
    cam.azimuth = 135.0
    cam.elevation = -16.0

    # Optional mocap markers/arrows if present in XML.
    goal_mocap_id = -1
    for name in ("goal_marker", "goal_marker_0"):
        bid_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid_g >= 0:
            goal_mocap_id = model.body_mocapid[bid_g]
            break

    arrow_mocap_id = -1
    for name in ("guidance_arrow", "guidance_arrow_0"):
        bid_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid_a >= 0:
            arrow_mocap_id = model.body_mocapid[bid_a]
            break

    T_total = x.shape[0]
    T_render = min(T_total, max(1, real_length))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for t in range(T_render):
            data.qpos[:] = x[t, :model.nq]

            # Goal marker (fixed world position)
            if goal_mocap_id >= 0:
                data.mocap_pos[goal_mocap_id] = goal_world[:3]
                data.mocap_quat[goal_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

            # Guidance arrow from current object toward goal
            if arrow_mocap_id >= 0:
                obj = x[t, 36:39]
                d = goal_world[:3] - obj
                q = quat_from_z_to_vec(d)
                data.mocap_pos[arrow_mocap_id] = obj
                data.mocap_quat[arrow_mocap_id] = q

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
                    xml_path,
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

        raise RuntimeError("Headless video render failed for EGL and OSMesa backends.\n" + "\n---\n".join(errors))


def extract_initial_condition(dataset, sample_idx, local_goal_dim=3):
    """
    Extract the world-frame robot/object history and goal info from a dataset sample.

    Returns:
        initial_condition: dict with 'robot' (H, 36) and 'obj' (H, 7)
        goal_local:        (local_goal_dim,) — object displacement (final − initial)
                           expressed in the yaw-rotated initial robot frame.
                           Axes: (dx_forward, dy_left, dz_up) aligned with the
                           pelvis at t=0 after stripping pitch/roll (yaw-only rotation).
        anchor:            dict with ref_pos, ref_quat, ref_obj_pos, final_obj_pos
        file_idx, batch_idx, start_time: dataset index triple
    """
    file_idx, batch_idx, start_time = dataset.indices[sample_idx]
    raw_traj = dataset._get_single_traj(file_idx, batch_idx)

    # World-frame history window
    history_size = dataset.history_size
    h_start = start_time
    h_end = start_time + history_size

    robot_hist = np.concatenate([
        raw_traj['base'][h_start:h_end],       # (H, 7)
        raw_traj['joints'][h_start:h_end],      # (H, 29)
    ], axis=-1)  # (H, 36)

    obj_hist = raw_traj['obj'][h_start:h_end]   # (H, 7)

    initial_condition = {
        'robot': robot_hist,
        'obj': obj_hist,
    }

    # Task params: dataset already computes the anchor with final_obj_pos
    sample = dataset[sample_idx]
    task_params_norm = sample[2]
    anchor = sample[-1]

    # Recover the object displacement in the yaw-rotated initial robot frame.
    # We rotate (final_obj_pos - init_obj_pos) from world frame into the frame
    # aligned with the pelvis yaw at t=0 (pitch/roll stripped).
    init_base_quat = raw_traj['base'][h_start, 3:7]
    init_obj_pos = raw_traj['obj'][h_start, :3]
    final_obj_pos = anchor['final_obj_pos']

    yaw = yaw_from_quat(init_base_quat)
    R_world_to_local = yaw_to_rot_matrix(-yaw)  # inverse of local-to-world

    delta_world = final_obj_pos[:3] - init_obj_pos[:3]
    delta_local_3d = (R_world_to_local @ delta_world[:, None])[:, 0]
    goal_local = delta_local_3d[:local_goal_dim]

    return initial_condition, goal_local, anchor, file_idx, batch_idx, start_time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Inference via MotionGenerator (unified pipeline)"
    )
    parser.add_argument("--inference_config", type=str, default="config/inference.yaml",
                        help="Path to inference defaults YAML")
    parser.add_argument("--epoch", type=str, default=None,
                        help="Checkpoint epoch number or path to .pth file")
    parser.add_argument("--phase1_checkpoint", type=str, default=None,
                        help="Two-phase mode: path to phase1 checkpoint")
    parser.add_argument("--phase2_checkpoint", type=str, default=None,
                        help="Two-phase mode: path to phase2 checkpoint")
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA weights (only for numeric epoch)")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--stitch_steps", type=int, default=None,
                        help="Number of autoregressive segments (auto-computed if omitted)")
    parser.add_argument("--target_traj_length", type=int, default=None,
                        help="Desired output trajectory length; used to auto-compute stitch_steps")
    parser.add_argument("--save_path", type=str, default="results/inference_mg.npz")
    parser.add_argument("--metrics_log_path", type=str, default="results/inference_metrics.jsonl",
                        help="Append per-run goal/error/timing metrics as JSONL")
    parser.add_argument("--traj_idx", type=int, default=0,
                        help="Trajectory (file) index")
    parser.add_argument("--batch_idx", type=int, default=0,
                        help="Batch index within file")
    parser.add_argument("--start_time", type=int, default=0,
                        help="Window start timestep")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cfg_w", type=float, default=1.0,
                        help="Classifier-free guidance weight")
    parser.add_argument("--task_params", nargs="+", type=float, default=None,
                        help="Override goal as local-frame displacement (e.g. --task_params 0.5 -0.2)")
    parser.add_argument("--style", type=str, default=None, choices=["pick", "push", "kick"],
                        help="Optional style command for style-conditioned model")
    parser.add_argument("--end_error_threshold", type=float, default=0.1,
                        help="XY-plane radius (m) for goal-reached check")
    parser.add_argument("--end_ground_num_frames", type=int, default=5,
                        help="Consecutive on-ground frames required before stopping")
    parser.add_argument("--end_ground_z_tol", type=float, default=0.05,
                        help="Z tolerance (m) for object to be considered on the ground")
    parser.add_argument("--goal_multiplier", type=float, default=1.0,
                        help="Scale goal displacement by this factor")
    parser.add_argument("--enable_goal_stop", action="store_true", default=True,
                        help="Stop when object reaches goal region (default: True)")
    parser.add_argument("--disable_goal_stop", action="store_true",
                        help="Disable early stopping when goal is reached")
    parser.add_argument("--enable_phys_stop", action="store_true",
                        help="Stop on physics violation (floor penetration / spike)")
    parser.add_argument("--enable_phys_clamp", action="store_true",
                        help="Clamp to valid state if physics violation detected (instead of stopping)")

    # Waypoint / z-profile arguments
    parser.add_argument("--last_frame_waypoint", action="store_true",
                        help="Add partial waypoint at last frame (obj_delta_xy + obj_z)")
    parser.add_argument("--arrival_ratio", type=float, default=0.85)
    parser.add_argument("--lift_height", type=float, default=DEFAULT_LIFT_HEIGHT)
    parser.add_argument("--lift_start", type=float, default=0.0)
    parser.add_argument("--lift_end", type=float, default=0.20)
    parser.add_argument("--walk_start_z", type=float, default=0.80)
    parser.add_argument("--no_lower_dist", type=float, default=0.4)

    # Analysis / visualization
    parser.add_argument("--no_visualize", action="store_true",
                        help="Skip MuJoCo visualisation")
    parser.add_argument("--video", action="store_true",
                        help="Save visualization as video file instead of opening interactive viewer")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to save video file (default: results/inference_mg_video.mp4)")

    cfg_defaults, _ = load_inference_defaults(
        argv=argv,
        section="inference_mg",
        default_path="config/inference.yaml",
    )
    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    args = parser.parse_args(argv)
    return args


def main():

    args = parse_args()

    # Handle mutually exclusive goal stop flags
    if args.disable_goal_stop:
        args.enable_goal_stop = False

    # ─── 1. Build MotionGenerator ──────────────────────────────────────────────
    mg = MotionGenerator(config_path="config/config.yaml", device=args.device)
    data_cfg = mg.data_cfg

    # ─── 2. Load weights ──────────────────────────────────────────────────────
    if mg.two_phase_enabled:
        p1 = args.phase1_checkpoint or mg.two_phase_cfg.get("phase1_checkpoint")
        p2 = args.phase2_checkpoint or mg.two_phase_cfg.get("phase2_checkpoint")
        if not p1 or not p2:
            raise ValueError(
                "Two-phase mode is enabled but phase checkpoints are missing. "
                "Provide --phase1_checkpoint and --phase2_checkpoint, or set two_phase.phase1_checkpoint/phase2_checkpoint in config/config.yaml"
            )
        mg.load_two_phase_weights(p1, p2)
    elif args.epoch is None:
        raise ValueError("Missing checkpoint: provide --epoch for single-model inference.")
    elif os.path.exists(args.epoch):
        mg.diffuser.load_weights_from_file(args.epoch)
    else:
        mg.diffuser.loadWeights(int(args.epoch), ema=args.ema)

    # ─── 3. Prepare dataset (needed to extract initial conditions) ─────────────
    # The MotionGenerator already has a lightweight dataset for normalisation.
    # We need a full dataset to look up specific trajectories.
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(mg.model_cfg, mg.training_cfg, data_cfg)
    data_buffer = preload_dataset(data_cfg, data_path)

    dataset = FlexibleWindowDataset(
        data_buffer=data_buffer,
        config=data_cfg,
        norm_path=norm_path,
        calculate_stats=False,
        training_cfg={},
    )
    # Make sure the MotionGenerator uses the same full dataset
    mg.dataset = dataset

    # ─── 4. Extract initial condition ──────────────────────────────────────────
    # Map (traj_idx, batch_idx, start_time) → dataset sample index
    target = (args.traj_idx, args.batch_idx, args.start_time)
    try:
        sample_idx = dataset.indices.index(target)
        print(f"Mapped {target} → sample {sample_idx}")
    except ValueError:
        raise ValueError(
            f"Target {target} not found in dataset indices. "
            f"Check traj_idx={args.traj_idx}, batch_idx={args.batch_idx}, "
            f"start_time={args.start_time}."
        )

    num_task_params = data_cfg.get("num_task_params", 3)
    local_goal_dim = min(3, num_task_params)
    initial_condition, goal_local, anchor, file_idx, batch_idx, start_time = \
        extract_initial_condition(dataset, sample_idx, local_goal_dim)

    # Override goal if provided via CLI
    if args.task_params is not None:
        goal_local = np.array(args.task_params, dtype=np.float64)
        print(f"Using CLI goal (yaw-rotated robot frame): {goal_local}")

    # Auto-compute stitch_steps from dataset trajectory length if not provided.
    # When the user supplied --task_params we calibrate from the goal distance
    # so the trajectory is long enough to actually reach it.
    if args.stitch_steps is None and args.target_traj_length is None:
        _eff = max(1, data_cfg["num_timesteps"] - data_cfg.get("state_history", 1))
        traj_len = dataset.traj_lengths[args.traj_idx]
        base_steps = max(1, math.ceil(traj_len / _eff))

        if args.task_params is not None:
            # Calibrate speed from the *dataset* sample's original goal distance,
            # then use it to size steps for the user-supplied goal.
            orig_dist = np.linalg.norm(anchor['final_obj_pos'][:2] - anchor['ref_obj_pos'][:2])
            goal_dist = np.linalg.norm(goal_local[:2])  # XY distance
            if orig_dist > 0.1 and goal_dist > 0:
                metres_per_step = max(orig_dist / base_steps, 0.05)
                args.stitch_steps = max(base_steps, math.ceil(goal_dist / metres_per_step))
            else:
                args.stitch_steps = base_steps
            print(f"Auto stitch_steps={args.stitch_steps} from goal distance "
                  f"{goal_dist:.2f}m (base={base_steps})")
        else:
            args.stitch_steps = base_steps
            print(f"Auto stitch_steps={args.stitch_steps} from traj length {traj_len}")

    # Scale goal if requested (AFTER auto-computing stitch_steps so we can scale them too)
    if args.goal_multiplier != 1.0:
        goal_local = goal_local * args.goal_multiplier
        # Scale stitch_steps to match the longer/shorter distance.
        if args.stitch_steps is not None:
            args.stitch_steps = max(1, int(args.stitch_steps * abs(args.goal_multiplier)))
        print(f"Scaled goal by {args.goal_multiplier} → {goal_local} "
              f"(stitch_steps={args.stitch_steps})")

    print(f"Goal displacement (yaw-rotated robot frame): {goal_local}")
    print(f"Initial robot pos: {initial_condition['robot'][0, :3]}")
    print(f"Initial obj pos:   {initial_condition['obj'][0, :3]}")
    print(f"Anchor final_obj_pos (world): {anchor['final_obj_pos']}")

    # Ensure stitch_steps is fully determined for logging
    if args.stitch_steps is None:
        _eff_gen = max(1, data_cfg["num_timesteps"] - data_cfg.get("state_history", 1))
        if args.target_traj_length is not None:
            args.stitch_steps = max(1, math.ceil(args.target_traj_length / _eff_gen))
        else:
            args.stitch_steps = 1
            
    _eff_horizon = max(1, data_cfg["num_timesteps"] - data_cfg.get("state_history", 1))
    planning_horizon_s = _eff_horizon * 0.01

    # ─── 5. Generate ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result, real_lengths = mg.generate_trajectory(
        initial_condition=initial_condition,
        goal_condition=goal_local,
        style_condition=args.style,
        target_traj_length=args.target_traj_length,
        stitch_steps=args.stitch_steps,
        num_samples=args.num_samples,
        cfg_w=args.cfg_w,
        end_error_threshold=args.end_error_threshold,
        end_ground_num_frames=args.end_ground_num_frames,
        end_ground_z_tol=args.end_ground_z_tol,
        enable_goal_stop=args.enable_goal_stop,
        enable_physics_stop=args.enable_phys_stop,
        enable_physics_clamp=args.enable_phys_clamp,  # clamp if stopping on physics violation
        use_last_frame_wp=args.last_frame_waypoint,
        arrival_ratio=args.arrival_ratio,
        lift_height=args.lift_height,
        no_lower_dist=args.no_lower_dist,
        lift_start=args.lift_start,
        lift_end=args.lift_end,
        walk_start_z=args.walk_start_z,
    )
    gen_time_s = time.perf_counter() - t0
    full_trajectory = result

    # ─── 6. Save ──────────────────────────────────────────────────────────────
    save_path = args.save_path
    if not save_path.endswith(".npz"):
        save_path = f"{save_path}.npz"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # ─── 7. Compute goal world position from (possibly scaled) goal_local ───
    init_base_quat = initial_condition['robot'][0, 3:7]
    init_obj_pos = initial_condition['obj'][0, :3]
    yaw = yaw_from_quat(init_base_quat)
    R_local_to_world = yaw_to_rot_matrix(yaw)
    goal_3d = np.zeros(3, dtype=np.float64)
    n_goal = min(len(goal_local), 3)
    goal_3d[:n_goal] = goal_local[:n_goal]
    goal_world = (R_local_to_world @ goal_3d[:, None])[:, 0] + init_obj_pos
    print(f"Goal world position: {goal_world}")

    # ─── 8. Report ────────────────────────────────────────────────────────────
    err = float("nan")
    desired = None
    achieved = None
    if full_trajectory.shape[-1] >= 43:
        world_goal_dim = min(3, num_task_params)
        start_obj = full_trajectory[0, 0, 36:36 + world_goal_dim]
        end_idx = int(real_lengths[0] - 1) if len(real_lengths) > 0 else -1
        end_obj = full_trajectory[0, end_idx, 36:36 + world_goal_dim]
        achieved = end_obj - start_obj
        desired = (goal_world - init_obj_pos)[:world_goal_dim]
        err = float(np.linalg.norm(desired - achieved))
        print("-" * 40)
        print(f"Desired displacement  (world): {desired}")
        print(f"Achieved displacement (world): {achieved}")
        print(f"L2 error: {err:.4f}")
        print("-" * 40)

    np.savez_compressed(
        save_path,
        trajectory=full_trajectory,
        real_lengths=np.asarray(real_lengths),
        goal_local=np.asarray(goal_local),
        goal_world=np.asarray(goal_world),
        init_robot=np.asarray(initial_condition['robot']),
        init_obj=np.asarray(initial_condition['obj']),
        traj_idx=np.array([args.traj_idx], dtype=np.int64),
        batch_idx=np.array([args.batch_idx], dtype=np.int64),
        start_time=np.array([args.start_time], dtype=np.int64),
    )
    print(f"Trajectory bundle saved to {save_path}  (shape: {full_trajectory.shape})")

    # ─── 8.1 Metrics logging ─────────────────────────────────────────────────
    goal_xyz = np.asarray(goal_local[:3], dtype=np.float64)
    goal_mag = float(np.linalg.norm(goal_xyz))
    if goal_mag > 1e-9:
        goal_dir = (goal_xyz / goal_mag).tolist()
    else:
        goal_dir = [0.0, 0.0, 0.0]
    metrics_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "inference_mg.py",
        "checkpoint": args.epoch,
        "ema": bool(args.ema),
        "traj_idx": int(args.traj_idx),
        "batch_idx": int(args.batch_idx),
        "start_time": int(args.start_time),
        "goal_local": np.asarray(goal_local, dtype=float).tolist(),
        "goal_direction_xyz": goal_dir,
        "goal_magnitude_xyz": goal_mag,
        "error_l2": err,
        "generation_time_s": float(gen_time_s),
        "time_per_inference_step_s": float(gen_time_s) / max(1, args.stitch_steps),
        "planning_horizon_s": planning_horizon_s,
        "save_path": save_path,
    }
    _append_jsonl(args.metrics_log_path, metrics_record)
    print(f"Metrics appended to {args.metrics_log_path}")

    # ─── 9. Build per-frame guidance vector for the arrow visualisation ───────
    #   guidance_vec: (T_total, 2+) — world-frame direction from current object
    #   to the goal at each frame, so the arrow tracks the remaining displacement.
    traj_0 = full_trajectory[0] if full_trajectory.ndim == 3 else full_trajectory
    T_total = traj_0.shape[0]
    guidance_vec = np.zeros((T_total, 3), dtype=np.float64)
    for t_idx in range(T_total):
        obj_pos_t = traj_0[t_idx, 36:39]
        guidance_vec[t_idx] = goal_world - obj_pos_t

    # ─── 10. Visualize ────────────────────────────────────────────────────────
    if not args.no_visualize:
        try:
            xml_path, _ = get_mj_xml_paths()
            if os.path.exists(xml_path):
                t = np.arange(T_total) * 0.01
                if args.video:
                    video_path = args.video_path or "results/inference_mg_video.mp4"
                    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
                    _save_single_video_headless(
                        trajectory=traj_0,
                        real_length=int(real_lengths[0]) if len(real_lengths) > 0 else int(T_total),
                        goal_world=goal_world,
                        out_path=video_path,
                        xml_path=xml_path,
                        fps=100,
                        width=1280,
                        height=720,
                    )
                    print(f"Saved video to {video_path}")
                else:
                    from utils.visualize.visualize import MjVisualizer
                    vis = MjVisualizer(xml_path, close_on_enter=False)
                    vis.visualize_trajectory(
                        t=t, x_traj=traj_0, repeat=True,
                        guidance_vec=guidance_vec,
                        goal_pos=goal_world,
                    )
                    vis.close()
            else:
                print(f"MuJoCo model not found at {xml_path}; skipping visualisation.")
        except Exception as e:
            print(f"Visualisation failed: {e}")


if __name__ == "__main__":
    main()

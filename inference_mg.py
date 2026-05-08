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
from utils.math.math_tools import (
    yaw_from_quat, yaw_to_rot_matrix, rot_to_quat_wxyz,
    quat_to_rot, roll_yaw_from_rot, roll_pitch_yaw_from_rot, normalize_angle,
)
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
    goal_obj_rot=None,
):
    """Save one trajectory video in a subprocess using offscreen MuJoCo rendering.

    This avoids GLFW initialization in headless environments.
    goal_obj_rot: optional (4,) [qw, qx, qy, qz] — orients the goal marker.
    """
    x = np.asarray(trajectory)
    rl = int(real_length)
    gw = np.asarray(goal_world, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected trajectory shape (T, D), got {x.shape}")
    if gw.shape[0] < 3:
        raise ValueError(f"Expected goal_world with at least 3 dims, got {gw.shape}")
    # Default identity quaternion when no goal rotation is specified
    gr = np.asarray(goal_obj_rot, dtype=np.float64) if goal_obj_rot is not None \
         else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="infer_mg_render_") as td:
        payload = os.path.join(td, "payload.npz")
        np.savez_compressed(
            payload,
            trajectory=x,
            real_length=np.array([rl], dtype=np.int64),
            goal_world=gw[:3],
            goal_obj_rot=gr,
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
goal_obj_rot = arr["goal_obj_rot"] if "goal_obj_rot" in arr.files else np.array([1.0, 0.0, 0.0, 0.0])

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

            # Goal marker: fixed world position, oriented to show desired object yaw
            if goal_mocap_id >= 0:
                data.mocap_pos[goal_mocap_id] = goal_world[:3]
                data.mocap_quat[goal_mocap_id] = goal_obj_rot.astype(np.float64)

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
        goal_local:        task-type dependent goal vector —
          pickplace: (local_goal_dim,) XYZ displacement in yaw-aligned robot frame.
          reorient:  (6,) [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw] in initial robot yaw frame.
        anchor:            dict with ref_pos, ref_quat, ref_obj_pos, final_obj_pos
        file_idx, batch_idx, start_time: dataset index triple
        goal_obj_quat_world: (4,) [qw,qx,qy,qz] actual desired final object quaternion
                             (world frame) for reorient; None otherwise.
    """
    file_idx, batch_idx, start_time, *_ = dataset.indices[sample_idx]
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

    sample = dataset[sample_idx]
    anchor = sample[-1]

    task_type = getattr(dataset, 'task_type', 'pickplace')

    goal_obj_quat_world = None
    if task_type == "reorient":
        # User-facing reorient goal layout (always 6-D, robot yaw frame):
        #   [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
        # The dataset/internal task_params layout is now 7-D when
        # normalize_goal_vec=True, but the CLI/user always works with the
        # raw deltas — convert the world-frame goal pose into the 6-D form.
        init_robot_quat = raw_traj['base'][h_end - 1, 3:7]
        init_obj_pos    = raw_traj['obj'][h_end - 1, :3]
        init_obj_quat   = raw_traj['obj'][h_end - 1, 3:7]

        if 'goal_obj_quat' in raw_traj:
            goal_obj_quat_world = np.asarray(raw_traj['goal_obj_quat'], dtype=np.float64)
        else:
            goal_obj_quat_world = raw_traj['obj'][-1, 3:7].copy()
        goal_obj_pos_world = raw_traj['obj'][-1, :3]

        yaw_r     = float(yaw_from_quat(init_robot_quat))
        R_rob_inv = yaw_to_rot_matrix(-yaw_r)
        world_delta = goal_obj_pos_world - init_obj_pos
        local_delta = (R_rob_inv @ world_delta[:, None])[:, 0]
        delta_x, delta_y, delta_z = float(local_delta[0]), float(local_delta[1]), float(local_delta[2])

        roll_c, pitch_c, yaw_c = roll_pitch_yaw_from_rot(R_rob_inv @ quat_to_rot(init_obj_quat))
        roll_d, pitch_d, yaw_d = roll_pitch_yaw_from_rot(R_rob_inv @ quat_to_rot(goal_obj_quat_world))
        delta_roll  = float(normalize_angle(roll_d  - roll_c))
        delta_pitch = float(normalize_angle(pitch_d - pitch_c))
        delta_yaw   = float(normalize_angle(yaw_d   - yaw_c))
        goal_local = np.array([delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw], dtype=np.float64)
    else:
        init_base_quat = raw_traj['base'][h_end - 1, 3:7]
        init_obj_pos   = raw_traj['obj'][h_end - 1, :3]
        final_obj_pos  = anchor['final_obj_pos']

        yaw = yaw_from_quat(init_base_quat)
        R_world_to_local = yaw_to_rot_matrix(-yaw)

        delta_world    = final_obj_pos[:3] - init_obj_pos[:3]
        delta_local_3d = (R_world_to_local @ delta_world[:, None])[:, 0]
        goal_local     = delta_local_3d[:local_goal_dim]

    return initial_condition, goal_local, anchor, file_idx, batch_idx, start_time, goal_obj_quat_world


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Inference via MotionGenerator (unified pipeline)"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to model/training config YAML (e.g. config/config_reorient.yaml)")
    parser.add_argument("--inference_config", type=str, default="config/inference.yaml",
                        help="Path to inference defaults YAML")
    parser.add_argument("--epoch", type=str, default=None,
                        help="Checkpoint epoch number or path to .pth file")
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
    parser.add_argument("--goal_obj_yaw", type=float, default=None,
                        help="Pickplace: desired world-frame object yaw (rad). "
                             "Reorient: desired delta_yaw in initial robot frame (rad).")
    parser.add_argument("--goal_obj_roll", type=float, default=None,
                        help="Reorient task only: desired delta_roll in initial robot frame (rad).")
    parser.add_argument("--init_obj_offset", nargs="+", type=float, default=None,
                        help="Shift initial box position in robot yaw-frame (dx_fwd, dy_left[, dz_up]). "
                             "Applies on top of the dataset's initial object pose.")
    parser.add_argument("--style", type=str, default=None, choices=["pick", "push", "kick"],
                        help="Optional style command for style-conditioned model")
    parser.add_argument("--end_error_threshold", type=float, default=0.1,
                        help="XY-plane radius (m) for goal-reached check")
    parser.add_argument("--end_yaw_threshold", type=float, default=None,
                        help="Angular tolerance (rad) for object yaw at goal. "
                             "Only active when --goal_obj_yaw is set. Default: no yaw check.")
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
    mg = MotionGenerator(config_path=args.config, device=args.device)
    data_cfg = mg.data_cfg
    task_type = data_cfg.get("task_type", "pickplace")

    # ─── 2. Load weights ──────────────────────────────────────────────────────
    if args.epoch is None:
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
        sample_idx = next(
            i for i, entry in enumerate(dataset.indices)
            if entry[:3] == target
        )
        print(f"Mapped {target} → sample {sample_idx}")
    except StopIteration:
        raise ValueError(
            f"Target {target} not found in dataset indices. "
            f"Check traj_idx={args.traj_idx}, batch_idx={args.batch_idx}, "
            f"start_time={args.start_time}."
        )

    num_task_params = data_cfg.get("num_task_params", 3)
    local_goal_dim = min(3, num_task_params)
    initial_condition, goal_local, anchor, file_idx, batch_idx, start_time, _gt_goal_quat = \
        extract_initial_condition(dataset, sample_idx, local_goal_dim)
    print(f"Task type: {task_type}")

    init_base_quat = initial_condition['robot'][-1, 3:7]

    # Shift initial object position if requested. The offset is in robot yaw frame
    # (forward, left, up). goal_local is adjusted so that goal_world stays at the
    # original (unshifted) destination — this keeps task_params ≈ 0 on the first
    # autoregressive step, matching the approach-augmentation training distribution
    # (object frozen at goal, robot walks toward it, task_params = 0).
    if args.init_obj_offset is not None:
        yaw = yaw_from_quat(init_base_quat)
        R_local_to_world = yaw_to_rot_matrix(yaw)
        offset_local = np.zeros(3, dtype=np.float64)
        offset_local[:len(args.init_obj_offset)] = args.init_obj_offset
        offset_world = (R_local_to_world @ offset_local[:, None])[:, 0]
        initial_condition['obj'][:, :3] += offset_world
        # Subtract offset from goal_local so goal_world stays at the original location.
        # task_params = goal_world - new_obj_pos = (orig_goal_world) - (orig_obj_pos + offset)
        #             = original_goal_local - offset_local
        # For --task_params overrides we skip this (user is specifying the goal explicitly).
        if args.task_params is None:
            goal_local = goal_local.copy()
            goal_local[:len(offset_local)] -= offset_local[:len(goal_local)]
        print(f"Shifted initial obj by (robot frame): {offset_local}  (world): {offset_world}")
        print(f"New initial obj pos: {initial_condition['obj'][-1, :3]}")
        if args.task_params is None:
            print(f"Adjusted goal_local for offset: {goal_local}")

    # Override goal if provided via CLI
    if args.task_params is not None:
        goal_local = np.array(args.task_params, dtype=np.float64)
        print(f"Using CLI goal: {goal_local}")
    elif task_type == "reorient" and (args.goal_obj_roll is not None or args.goal_obj_yaw is not None):
        # Patch roll/yaw components while keeping other components from dataset
        # Layout: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
        if args.goal_obj_roll is not None:
            goal_local[3] = float(args.goal_obj_roll)
        if args.goal_obj_yaw is not None:
            goal_local[5] = float(args.goal_obj_yaw)
        print(f"Using CLI reorient goal [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]: {goal_local}")

    # Auto-compute stitch_steps from dataset trajectory length if not provided.
    if args.stitch_steps is None and args.target_traj_length is None:
        _eff = max(1, data_cfg["num_timesteps"] - data_cfg.get("state_history", 1))
        traj_len = dataset.traj_lengths[args.traj_idx]
        base_steps = max(1, math.ceil(traj_len / _eff))

        if task_type == "reorient":
            # For reorient there is no XY distance metric; use trajectory length directly.
            args.stitch_steps = base_steps
            print(f"Auto stitch_steps={args.stitch_steps} from traj length {traj_len} (reorient)")
        elif args.task_params is not None:
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
    print(f"Initial robot pos (current): {initial_condition['robot'][-1, :3]}")
    print(f"Initial obj pos (current):   {initial_condition['obj'][-1, :3]}")
    print(f"Anchor final_obj_pos (world): {anchor['final_obj_pos']}")

    # Compute goal object rotation for visualization marker
    goal_obj_rot = None
    if task_type == "reorient":
        # Use the ground-truth final object quaternion from the dataset trajectory.
        # The previous approach reconstructed from [delta_roll, delta_yaw] via
        # roll_yaw_from_rot which drops the pitch component, causing the marker
        # to be misaligned whenever the desired orientation has pitch.
        if _gt_goal_quat is not None and not (args.goal_obj_roll is not None or args.goal_obj_yaw is not None or args.task_params is not None):
            goal_obj_rot = _gt_goal_quat.copy()
        else:
            # CLI override: reconstruct full orientation from [delta_roll, delta_pitch, delta_yaw].
            _yaw_r   = float(yaw_from_quat(init_base_quat))
            _R_rob   = yaw_to_rot_matrix(_yaw_r)
            _R_rob_i = yaw_to_rot_matrix(-_yaw_r)
            _init_q  = initial_condition['obj'][0, 3:7]
            _rc, _pc, _yc = roll_pitch_yaw_from_rot(_R_rob_i @ quat_to_rot(_init_q))
            # goal_local layout: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
            _rd = normalize_angle(_rc + goal_local[3])
            _pd = normalize_angle(_pc + goal_local[4])
            _yd = normalize_angle(_yc + goal_local[5])
            _cr, _sr = np.cos(_rd), np.sin(_rd)
            _cp, _sp = np.cos(_pd), np.sin(_pd)
            _Rx = np.array([[1, 0, 0], [0, _cr, -_sr], [0, _sr, _cr]], dtype=np.float64)
            _Ry = np.array([[_cp, 0, _sp], [0, 1, 0], [-_sp, 0, _cp]], dtype=np.float64)
            _R_des_world = _R_rob @ (yaw_to_rot_matrix(_yd) @ _Ry @ _Rx)
            goal_obj_rot = rot_to_quat_wxyz(_R_des_world)
        print(f"Goal object quat (world): {goal_obj_rot}")
    elif args.goal_obj_yaw is not None:
        # Pickplace: construct quaternion from yaw angle
        yaw = float(args.goal_obj_yaw)
        qw = np.cos(yaw / 2.0)
        qx, qy = 0.0, 0.0
        qz = np.sin(yaw / 2.0)
        goal_obj_rot = np.array([qw, qx, qy, qz], dtype=np.float64)
        print(f"Goal object yaw: {yaw:.4f} rad → quat: {goal_obj_rot}")

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
        goal_obj_rot=goal_obj_rot,
        style_condition=args.style,
        target_traj_length=args.target_traj_length,
        stitch_steps=args.stitch_steps,
        num_samples=args.num_samples,
        cfg_w=args.cfg_w,
        end_error_threshold=args.end_error_threshold,
        end_yaw_threshold=args.end_yaw_threshold,
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
    init_base_quat = initial_condition['robot'][-1, 3:7]
    init_obj_pos   = initial_condition['obj'][-1, :3]
    if task_type == "reorient":
        # goal_local = [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
        # XY displacement is in robot yaw frame; rotate back to world frame
        yaw = yaw_from_quat(init_base_quat)
        R_local_to_world = yaw_to_rot_matrix(yaw)
        local_xy = np.array([goal_local[0], goal_local[1], 0.0], dtype=np.float64)
        world_xy = (R_local_to_world @ local_xy[:, None])[:, 0]
        goal_world = init_obj_pos.copy()
        goal_world[0] += world_xy[0]
        goal_world[1] += world_xy[1]
        goal_world[2] += goal_local[2]  # delta_z
        print(f"Goal world position (reorient): {goal_world}")
    else:
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
        end_idx = int(real_lengths[0] - 1) if len(real_lengths) > 0 else -1
        if task_type == "reorient":
            start_obj_pos  = full_trajectory[0, 0, 36:39]
            end_obj_pos    = full_trajectory[0, end_idx, 36:39]
            end_obj_quat   = full_trajectory[0, end_idx, 39:43]
            init_obj_quat  = initial_condition['obj'][0, 3:7]
            robot_yaw      = float(yaw_from_quat(init_base_quat))
            R_rob_inv      = yaw_to_rot_matrix(-robot_yaw)
            # XY displacement in robot frame
            world_delta_pos = end_obj_pos[:3] - start_obj_pos[:3]
            local_delta_pos = (R_rob_inv @ world_delta_pos[:, None])[:, 0]
            delta_x_achieved = float(local_delta_pos[0])
            delta_y_achieved = float(local_delta_pos[1])
            delta_z_achieved = float(local_delta_pos[2])
            # Rotation deltas
            roll_c, pitch_c, yaw_c = roll_pitch_yaw_from_rot(R_rob_inv @ quat_to_rot(init_obj_quat))
            roll_e, pitch_e, yaw_e = roll_pitch_yaw_from_rot(R_rob_inv @ quat_to_rot(end_obj_quat))
            delta_roll_achieved  = float(normalize_angle(roll_e  - roll_c))
            delta_pitch_achieved = float(normalize_angle(pitch_e - pitch_c))
            delta_yaw_achieved   = float(normalize_angle(yaw_e   - yaw_c))
            desired  = goal_local[:6]
            achieved = np.array([delta_x_achieved, delta_y_achieved, delta_z_achieved,
                                  delta_roll_achieved, delta_pitch_achieved, delta_yaw_achieved])
            err = float(np.linalg.norm(desired - achieved))
            print("-" * 40)
            print(f"Desired  [dx, dy, dz, droll, dpitch, dyaw] (robot frame): {np.round(desired, 4)}")
            print(f"Achieved [dx, dy, dz, droll, dpitch, dyaw] (robot frame): {np.round(achieved, 4)}")
            print(f"L2 error: {err:.4f}")
            print("-" * 40)
        else:
            world_goal_dim = min(3, num_task_params)
            start_obj = full_trajectory[0, 0, 36:36 + world_goal_dim]
            end_obj   = full_trajectory[0, end_idx, 36:36 + world_goal_dim]
            achieved  = end_obj - start_obj
            desired   = (goal_world - init_obj_pos)[:world_goal_dim]
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

    # Prepend the history frames so the viewer shows the context given to the model.
    # initial_condition['robot'] is (H, 36), initial_condition['obj'] is (H, 7).
    _r_hist_np = np.asarray(initial_condition['robot'])   # (H, 36)
    _o_hist_np = np.asarray(initial_condition['obj'])     # (H, 7)
    if _r_hist_np.ndim == 3:   # batched (1, H, 36) → (H, 36)
        _r_hist_np = _r_hist_np[0]
        _o_hist_np = _o_hist_np[0]
    hist_frames = np.concatenate([_r_hist_np, _o_hist_np], axis=-1)  # (H, 43)
    # Pad to full traj_0 width if needed (extra columns beyond 43 stay zero)
    if hist_frames.shape[-1] < traj_0.shape[-1]:
        _pad = np.zeros((hist_frames.shape[0], traj_0.shape[-1] - hist_frames.shape[-1]), dtype=np.float64)
        hist_frames = np.concatenate([hist_frames, _pad], axis=-1)
    traj_with_hist = np.concatenate([hist_frames, traj_0], axis=0)
    H_hist = hist_frames.shape[0]

    T_total = traj_with_hist.shape[0]
    guidance_vec = np.zeros((T_total, 3), dtype=np.float64)
    for t_idx in range(T_total):
        obj_pos_t = traj_with_hist[t_idx, 36:39]
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
                        trajectory=traj_with_hist,
                        real_length=(int(real_lengths[0]) + H_hist) if len(real_lengths) > 0 else int(T_total),
                        goal_world=goal_world,
                        out_path=video_path,
                        xml_path=xml_path,
                        fps=100,
                        width=1280,
                        height=720,
                        goal_obj_rot=goal_obj_rot,
                    )
                    print(f"Saved video to {video_path}")
                else:
                    from utils.visualize.visualize import MjVisualizer
                    vis = MjVisualizer(xml_path, close_on_enter=False)
                    vis.visualize_trajectory(
                        t=t, x_traj=traj_with_hist, repeat=True,
                        guidance_vec=guidance_vec,
                        goal_pos=goal_world,
                        goal_quat=goal_obj_rot,
                    )
                    vis.close()
            else:
                print(f"MuJoCo model not found at {xml_path}; skipping visualisation.")
        except Exception as e:
            print(f"Visualisation failed: {e}")


if __name__ == "__main__":
    main()

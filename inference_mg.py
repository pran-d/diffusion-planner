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
import time
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


def extract_initial_condition(dataset, sample_idx, num_task_params):
    """
    Extract the world-frame robot/object history and goal info from a dataset sample.

    Returns:
        initial_condition: dict with 'robot' (H, 36) and 'obj' (H, 7)
        goal_local:        (num_task_params,) — object displacement (final − initial)
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
    _, _, task_params_norm, anchor = dataset[sample_idx]

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
    goal_local = delta_local_3d[:num_task_params]

    return initial_condition, goal_local, anchor, file_idx, batch_idx, start_time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Inference via MotionGenerator (unified pipeline)"
    )
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
    if args.epoch is None:
        parser.error("Missing `--epoch`. Set it via CLI or config/inference.yaml::inference_mg.epoch")
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
    if os.path.exists(args.epoch):
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
    initial_condition, goal_local, anchor, file_idx, batch_idx, start_time = \
        extract_initial_condition(dataset, sample_idx, num_task_params)

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

    # ─── 5. Generate ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result, real_lengths = mg.generate_trajectory(
        initial_condition=initial_condition,
        goal_condition=goal_local,
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
    goal_3d[:len(goal_local)] = goal_local
    goal_world = (R_local_to_world @ goal_3d[:, None])[:, 0] + init_obj_pos
    print(f"Goal world position: {goal_world}")

    # ─── 8. Report ────────────────────────────────────────────────────────────
    err = float("nan")
    desired = None
    achieved = None
    if full_trajectory.shape[-1] >= 43:
        start_obj = full_trajectory[0, 0, 36:36 + num_task_params]
        end_idx = int(real_lengths[0] - 1) if len(real_lengths) > 0 else -1
        end_obj = full_trajectory[0, end_idx, 36:36 + num_task_params]
        achieved = end_obj - start_obj
        desired = (goal_world - init_obj_pos)[:num_task_params]
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
    goal_xy = np.asarray(goal_local[:2], dtype=np.float64)
    goal_mag = float(np.linalg.norm(goal_xy))
    if goal_mag > 1e-9:
        goal_dir = (goal_xy / goal_mag).tolist()
    else:
        goal_dir = [0.0, 0.0]
    metrics_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "inference_mg.py",
        "checkpoint": args.epoch,
        "ema": bool(args.ema),
        "traj_idx": int(args.traj_idx),
        "batch_idx": int(args.batch_idx),
        "start_time": int(args.start_time),
        "goal_local": np.asarray(goal_local, dtype=float).tolist(),
        "goal_direction_xy": goal_dir,
        "goal_magnitude_xy": goal_mag,
        "error_l2": err,
        "generation_time_s": float(gen_time_s),
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
            from utils.visualize.visualize import MjVisualizer
            xml_path, _ = get_mj_xml_paths()
            if os.path.exists(xml_path):
                vis = MjVisualizer(xml_path, close_on_enter=False)
                t = np.arange(T_total) * 0.01
                if args.video:
                    video_path = args.video_path or "results/inference_mg_video.mp4"
                    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
                    vis.render_trajectory_to_video(t=t, x_traj=traj_0, save_path=video_path)
                else:
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

"""
Inference script that uses MotionGenerator as the single entry point.

This mirrors inference.py but delegates all autoregressive loop logic,
waypoint construction, and SBTO reconstruction to MotionGenerator.generate_trajectory().

Usage:
    python inference_mg.py --epoch 100 --traj_idx 5 --batch_idx 0
    python inference_mg.py --epoch results/model_100.pth --task_params 0.5 -0.3
    python inference_mg.py --epoch 100 --ema --stitch_steps 10 --return_analysis
"""

import argparse
import os
import math

import numpy as np
import torch
import yaml

from config.configure import load_config, get_data_path, get_norm_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset
from utils.math.sbto_utils import build_feature_layout
from motion_generator import MotionGenerator
from inference import DEFAULT_LIFT_HEIGHT


def extract_initial_condition(dataset, sample_idx, num_task_params):
    """
    Extract the world-frame robot/object history and goal info from a dataset sample.

    Returns:
        initial_condition: dict with 'robot' (H, 36) and 'obj' (H, 7)
        goal_local:        (num_task_params,) — relative goal displacement in local frame
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

    # Recover the relative goal displacement in the initial pelvis-local frame.
    # anchor gives world-frame final_obj_pos and ref_obj_pos.
    # We need to express (final_obj_pos - ref_obj_pos) in the yaw-rotated local frame.
    from utils.math.math_tools import yaw_from_quat, yaw_to_rot_matrix
    init_base_quat = raw_traj['base'][h_start, 3:7]
    init_obj_pos = raw_traj['obj'][h_start, :3]
    final_obj_pos = anchor['final_obj_pos']

    yaw = yaw_from_quat(init_base_quat)
    R_world_to_local = yaw_to_rot_matrix(-yaw)  # inverse of local-to-world

    delta_world = final_obj_pos[:3] - init_obj_pos[:3]
    delta_local_3d = (R_world_to_local @ delta_world[:, None])[:, 0]
    goal_local = delta_local_3d[:num_task_params]

    return initial_condition, goal_local, anchor, file_idx, batch_idx, start_time


def main():
    parser = argparse.ArgumentParser(
        description="Inference via MotionGenerator (unified pipeline)"
    )
    parser.add_argument("--epoch", type=str, required=True,
                        help="Checkpoint epoch number or path to .pth file")
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA weights (only for numeric epoch)")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--stitch_steps", type=int, default=None,
                        help="Number of autoregressive segments (auto-computed if omitted)")
    parser.add_argument("--target_traj_length", type=int, default=None,
                        help="Desired output trajectory length; used to auto-compute stitch_steps")
    parser.add_argument("--save_path", type=str, default="results/inference_mg.npy")
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
    parser.add_argument("--end_error_threshold", type=float, default=0.1)
    parser.add_argument("--goal_multiplier", type=float, default=1.0,
                        help="Scale goal displacement by this factor")
    parser.add_argument("--enable_goal_stop", action="store_true",
                        help="Stop when object reaches goal region")
    parser.add_argument("--enable_phys_stop", action="store_true",
                        help="Stop on physics violation (floor penetration / spike)")

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
    parser.add_argument("--return_analysis", action="store_true",
                        help="Save per-window analysis data (.npz)")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Skip MuJoCo visualisation")

    args = parser.parse_args()

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
        print(f"Using override goal (local frame): {goal_local}")

    # Scale goal if requested
    if args.goal_multiplier != 1.0:
        goal_local = goal_local * args.goal_multiplier
        print(f"Scaled goal by {args.goal_multiplier} → {goal_local}")

    # Auto-compute stitch_steps from dataset trajectory length if not provided
    if args.stitch_steps is None and args.target_traj_length is None:
        _eff = data_cfg["num_timesteps"] - 1
        traj_len = dataset.traj_lengths[args.traj_idx]
        args.stitch_steps = max(1, math.ceil(traj_len / _eff))
        print(f"Auto stitch_steps={args.stitch_steps} from traj length {traj_len}")

    print(f"Goal (local frame): {goal_local}")
    print(f"Initial robot pos: {initial_condition['robot'][0, :3]}")
    print(f"Initial obj pos:   {initial_condition['obj'][0, :3]}")
    print(f"Anchor final_obj_pos (world): {anchor['final_obj_pos']}")

    # ─── 5. Generate ──────────────────────────────────────────────────────────
    result = mg.generate_trajectory(
        initial_condition=initial_condition,
        goal_condition=goal_local,
        target_traj_length=args.target_traj_length,
        stitch_steps=args.stitch_steps,
        num_samples=args.num_samples,
        cfg_w=args.cfg_w,
        end_error_threshold=args.end_error_threshold,
        enable_goal_stop=args.enable_goal_stop,
        enable_physics_stop=args.enable_phys_stop,
        use_last_frame_wp=args.last_frame_waypoint,
        arrival_ratio=args.arrival_ratio,
        lift_height=args.lift_height,
        no_lower_dist=args.no_lower_dist,
        lift_start=args.lift_start,
        lift_end=args.lift_end,
        walk_start_z=args.walk_start_z,
        return_analysis=args.return_analysis,
    )

    if args.return_analysis:
        full_trajectory, analysis_dict = result
    else:
        full_trajectory = result

    # ─── 6. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    np.save(args.save_path, full_trajectory)
    print(f"Trajectory saved to {args.save_path}  (shape: {full_trajectory.shape})")

    if args.return_analysis:
        analysis_path = args.save_path.replace('.npy', '_analysis.npz')
        # Convert lists to arrays where possible
        save_dict = {}
        for k, v in analysis_dict.items():
            if isinstance(v, list):
                # Filter None entries (waypoints may be None when inbetweening disabled)
                if any(x is None for x in v):
                    save_dict[k] = np.array(v, dtype=object)
                else:
                    save_dict[k] = np.array(v)
            elif isinstance(v, np.ndarray):
                save_dict[k] = v
            else:
                save_dict[k] = np.array(v)
        np.savez(analysis_path, **save_dict)
        print(f"Analysis saved to {analysis_path}")

    # ─── 7. Report ────────────────────────────────────────────────────────────
    if full_trajectory.shape[-1] >= 43:
        start_obj = full_trajectory[0, 0, 36:36 + num_task_params]
        end_obj = full_trajectory[0, -1, 36:36 + num_task_params]
        achieved = end_obj - start_obj
        desired = (anchor['final_obj_pos'] - anchor['ref_obj_pos'])[:num_task_params]
        err = np.linalg.norm(desired - achieved)
        print("-" * 40)
        print(f"Desired displacement  (world): {desired}")
        print(f"Achieved displacement (world): {achieved}")
        print(f"L2 error: {err:.4f}")
        print("-" * 40)

    # ─── 8. Visualize ─────────────────────────────────────────────────────────
    if not args.no_visualize:
        try:
            from utils.visualize.visualize import MjVisualizer
            xml_path = "mj_model.xml"
            if not os.path.exists(xml_path):
                xml_path = os.path.join(data_path, "mj_model.xml")
            if os.path.exists(xml_path):
                vis = MjVisualizer(xml_path, close_on_enter=False)
                traj = full_trajectory[0] if full_trajectory.ndim == 3 else full_trajectory
                T_steps = traj.shape[0]
                t = np.arange(T_steps) * 0.01
                vis.visualize_trajectory(
                    t=t, x_traj=traj, repeat=True,
                    goal_pos=anchor.get('final_obj_pos'),
                )
                vis.close()
            else:
                print(f"MuJoCo model not found at {xml_path}; skipping visualisation.")
        except Exception as e:
            print(f"Visualisation failed: {e}")


if __name__ == "__main__":
    main()

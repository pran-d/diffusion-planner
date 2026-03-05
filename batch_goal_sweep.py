#!/usr/bin/env python3
"""
batch_goal_sweep.py

Takes initial conditions from a single dataset trajectory, generates N goals
(original + random variations) around that trajectory's target, and runs batch
inference for all goals simultaneously.  The resulting trajectories are
visualised in a single MuJoCo window with per-trajectory goal markers and
goal-direction arrows.

Example usage:
    python batch_goal_sweep.py --epoch 5000 --traj_idx 0 --batch_idx 0 \
        --num_goals 6 --goal_spread 0.4 --goal_multiplier 1.2 \
        --last_frame_waypoint --visualize
"""

import argparse
import math
import os
import re
import tempfile

import numpy as np
import torch
import yaml

from config.configure import load_config, get_data_path, get_norm_path
from models.model import RobotDiffuser
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset
from utils.math.sbto_utils import compute_task_params
from utils.visualize.visualize import MjVisualizer, DiffusionOverlayVisualizer
from utils.visualize.visualize_param_sweep import run_evaluation_batch
from inference import DEFAULT_LIFT_HEIGHT


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser("Batch Goal Sweep — single IC, N randomised goals")

    # Checkpoint
    parser.add_argument("--epoch", type=str, required=True,
                        help="Checkpoint epoch (int) or direct file path")

    # Initial condition selection
    parser.add_argument("--traj_idx",   type=int, default=None,
                        help="Trajectory (file) index (takes priority over --sample_idx)")
    parser.add_argument("--batch_idx",  type=int, default=0,
                        help="Batch index within file")
    parser.add_argument("--start_time", type=int, default=0,
                        help="Window start timestep")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Dataset flat sample index (ignored when --traj_idx is set)")

    # Goal sweep
    parser.add_argument("--num_goals",       type=int,   default=5,
                        help="Total number of goals, including the original (default: 5)")
    parser.add_argument("--goal_spread",     type=float, default=0.3,
                        help="Max radius of random goal offsets in XY (metres, default: 0.3)")
    parser.add_argument("--goal_multiplier", type=float, default=1.0,
                        help="Scale every goal's displacement from ref_obj_pos by this factor")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="Random seed for goal generation (default: 42)")

    # Inference
    parser.add_argument("--stitch_steps",   type=int,   default=None,
                        help="Autoregressive segments to generate (auto-computed if not set)")
    parser.add_argument("--cfg_w",          type=float, default=1.0,
                        help="Classifier-free guidance weight")
    parser.add_argument("--guidance_wt",    type=float, default=0.0,
                        help="Test-time gradient guidance strength")
    parser.add_argument("--batch_size",     type=int,   default=64,
                        help="Sub-batch size for GPU inference")
    parser.add_argument("--action_horizon", type=int,   default=None,
                        help="Truncate each segment to this many steps")
    parser.add_argument("--device",         type=str,   default="cuda")
    parser.add_argument("--enable_goal_stop", action="store_true",
                        help="Stop at the end goal (default: False)")
    parser.add_argument("--goal_stop_threshold", type=float, default=0.1,
                        help="Distance threshold for stopping at each goal (default: 0.1)")

    # Waypoint / z-profile args (forwarded to run_evaluation_batch)
    parser.add_argument("--last_frame_waypoint", action="store_true",
                        help="Add last-frame partial waypoint (obj_delta_xy + obj_delta_z)")
    parser.add_argument("--arrival_ratio", type=float, default=0.70,
                        help="Object arrives in this fraction of total time (0-1; data: 90%% XY by 65%%)")
    parser.add_argument("--lift_height",   type=float, default=DEFAULT_LIFT_HEIGHT,
                        help=f"Peak lift height in metres (default: {DEFAULT_LIFT_HEIGHT}m)")
    parser.add_argument("--lift_start",    type=float, default=0.10,
                        help="Fraction of trajectory where lift begins (data: ~10%%)")
    parser.add_argument("--lift_end",      type=float, default=0.40,
                        help="Fraction of trajectory where lift reaches peak (data: ~40%%)")
    parser.add_argument("--walk_start_z",  type=float, default=0.25,
                        help="Gate XY motion until z >= this fraction of lift_height (data: ~25%%)")
    parser.add_argument("--no_lower_dist", type=float, default=0.75,
                        help="Lower z when XY distance to goal drops below this (metres)")

    # Output
    parser.add_argument("--save_path", type=str, default="results/goal_sweep.npy",
                        help="Where to save the generated trajectory array")
    parser.add_argument("--visualize", action="store_true",
                        help="Open MuJoCo viewer after generation")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Goal generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_goals(original_goal, ref_obj_pos, num_goals, goal_spread,
                   goal_multiplier, rng):
    """
    Return an (N, 3) array of goal positions arranged on two concentric circles
    around ref_obj_pos, each goal at a unique, evenly-spaced theta.

    The N goals are distributed as evenly as possible between an inner circle
    (radius = goal_spread / 2) and an outer circle (radius = goal_spread).
    Angles are evenly spaced over [0, 2π) across the full set so every goal
    has a distinct direction from the robot.  Slot 0 is always the original
    goal (scaled by goal_multiplier); the remaining N-1 slots are filled with
    the two-circle layout.

    Args:
        original_goal : (3,) world-frame target object position from dataset
        ref_obj_pos   : (3,) world-frame object position at t=0 (scaling origin)
        num_goals     : total number of goals (int)
        goal_spread   : outer circle radius in metres (float)
        goal_multiplier: scalar to amplify/shrink displacement from ref_obj_pos
        rng           : np.random.Generator (unused, kept for API compatibility)

    Returns:
        goals : (N, 3)  world-frame goal positions
    """
    center = ref_obj_pos.copy()

    # Slot 0: original goal, scaled from ref_obj_pos
    g0 = ref_obj_pos.copy()
    g0[:2] = ref_obj_pos[:2] + goal_multiplier * (original_goal[:2] - ref_obj_pos[:2])
    base_goals = [g0]

    n_extra = num_goals - 1
    if n_extra > 0:
        # Evenly-spaced angles across all extra goals
        thetas = np.linspace(0.0, 2.0 * np.pi, n_extra, endpoint=False)

        # Split into two circles: inner gets ceil(n/2), outer gets floor(n/2)
        n_inner = math.ceil(n_extra / 2)
        n_outer = n_extra - n_inner

        r_outer = goal_spread
        r_inner = goal_spread / 2.0

        for k in range(n_extra):
            theta = thetas[k]
            r = r_inner if k < n_inner else r_outer
            g = ref_obj_pos.copy()
            g[0] = center[0] + r * np.cos(theta)
            g[1] = center[1] + r * np.sin(theta)
            base_goals.append(g)

    return np.stack(base_goals)  # (N, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive XML generation
# ──────────────────────────────────────────────────────────────────────────────

def build_adaptive_xml(base_repeated_xml: str, num_robots: int) -> str:
    """
    Generate a temporary MuJoCo XML that contains exactly `num_robots` robots,
    adapted from `base_repeated_xml` which has an arbitrary (larger) number.

    Strategy:
      1. Extract everything before the first pelvis_0 body as the XML header.
      2. Extract the robot-0 block (pelvis_0 + largebox_0 + goal_marker_0 +
         guidance_arrow_0) as a template.
      3. For each i in 0..num_robots-1, stamp a fresh copy of the template by
         replacing every `_0"` name-attribute suffix with `_i"`.
      4. Append the </worldbody></mujoco> footer.
      5. Write to a temp file and return its path.
    """
    with open(base_repeated_xml) as f:
        lines = f.readlines()  # keeps \n endings

    # ── Locate each robot's pelvis start line (0-indexed) ─────────────────────
    robot_starts = [
        idx for idx, line in enumerate(lines)
        if re.search(r'body name="pelvis_\d+"', line)
    ]
    if not robot_starts:
        raise ValueError(f"No pelvis_N bodies found in {base_repeated_xml}")

    # ── Header: everything before robot 0 ─────────────────────────────────────
    header = ''.join(lines[: robot_starts[0]])

    # ── Template: robot 0's block (up to – not including – robot 1, or </worldbody>) ─
    wb_close_idx = next(
        (i for i, l in enumerate(lines) if '</worldbody>' in l), len(lines)
    )
    block_end = robot_starts[1] if len(robot_starts) >= 2 else wb_close_idx
    template = ''.join(lines[robot_starts[0]: block_end])

    # ── Footer: </worldbody> onward ────────────────────────────────────────────
    footer = ''.join(lines[wb_close_idx:])

    # ── Assemble N robot blocks ───────────────────────────────────────────────
    # Replace every `_0"` (name-attribute suffix) with `_i"`.  The look-ahead
    # `(?=")` ensures we only touch XML name strings, not numeric values.
    robot_blocks = []
    for i in range(num_robots):
        block = re.sub(r'_0(?=")', f'_{i}', template)
        robot_blocks.append(block)

    xml_content = header + ''.join(robot_blocks) + footer

    # ── Write to temp file ────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.xml',
        prefix=f'mj_adaptive_{num_robots}robots_',
        delete=False,
    )
    tmp.write(xml_content)
    tmp.flush()
    tmp.close()
    print(f"Adaptive XML written to {tmp.name}  ({num_robots} robots)")
    return tmp.name


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _build_guidance_vecs(obj_traj_w, goals_arr):
    """
    Build dynamic guidance direction vectors (N, T, 3):
    at each timestep, point from current object position toward the goal.
    """
    N, T, _ = obj_traj_w.shape
    guidance_vecs = np.zeros((N, T, 3), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            curr = obj_traj_w[i, t, :2]
            direction = goals_arr[i, :2] - curr
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                guidance_vecs[i, t, :2] = direction / norm
    return guidance_vecs


def extract_ref_trajectories(dataset, ref_pos, target_T):
    """
    Extract one world-frame ground-truth trajectory per unique (file_idx, batch_idx)
    from the dataset's ram_cache, translate each so its t=0 pelvis XY aligns with
    ref_pos, and pad/trim to target_T frames.

    Returns:
        np.ndarray of shape (K, target_T, 43)  or  None if cache is empty.
    """
    seen = {}   # file_idx → (target_T, 43) array  (batch_idx=0 only)
    for file_idx, _, __ in dataset.indices:
        if file_idx in seen:
            continue
        raw    = dataset.ram_cache[file_idx]
        base   = np.asarray(raw['base'][0],   dtype=np.float64)  # (T, 7)
        joints = np.asarray(raw['joints'][0], dtype=np.float64)  # (T, 29)
        obj    = np.asarray(raw['obj'][0],    dtype=np.float64)  # (T, 7)

        # Translate XY so t=0 pelvis aligns with the current initial condition
        xy_offset    = np.asarray(ref_pos[:2]) - base[0, :2]
        base_a       = base.copy();  base_a[:, :2] += xy_offset
        obj_a        = obj.copy();   obj_a[:, :2]  += xy_offset

        # Guard: base / joints / obj may have different T when downsampling
        # is applied unevenly across keys — trim to the common minimum.
        T_min = min(base_a.shape[0], joints.shape[0], obj_a.shape[0])
        base_a, joints, obj_a = base_a[:T_min], joints[:T_min], obj_a[:T_min]

        traj = np.concatenate([base_a, joints, obj_a], axis=-1).astype(np.float32)  # (T, 43)

        # Pad or trim to target_T
        T = traj.shape[0]
        if T < target_T:
            traj = np.concatenate([traj, np.tile(traj[-1:], (target_T - T, 1))], axis=0)
        else:
            traj = traj[:target_T]
        seen[file_idx] = traj

    if not seen:
        return None
    return np.stack(list(seen.values()))   # (K, target_T, 43)


def visualize_results(full_traj, obj_traj_w, goals_arr, args, dataset=None, ref_pos=None):
    """
    Visualise all N goal trajectories simultaneously, plus ground-truth dataset
    trajectories in a contrasting colour.

    When mj_model_repeated.xml is present, an adaptive copy is generated on the
    fly with exactly N + K robots (generated + reference), so MuJoCo only
    allocates the joints / mocap bodies actually needed.

    Falls back to MjVisualizer with overlay_paths + markers when the repeated
    XML template is unavailable.
    """
    N, T, _ = full_traj.shape
    xml_path          = "mj_model.xml"
    repeated_xml_path = "mj_model_repeated.xml"

    # ── Extract ground-truth reference trajectories ───────────────────────────
    ref_trajs = None
    if dataset is not None and ref_pos is not None:
        ref_trajs = extract_ref_trajectories(dataset, ref_pos, T)
        if ref_trajs is not None:
            K = len(ref_trajs)
            print(f"Loaded {K} reference (ground-truth) trajectories from dataset.")

    # ── Merge generated + reference into one batch ────────────────────────────
    if ref_trajs is not None:
        all_trajs     = np.concatenate([full_traj, ref_trajs], axis=0)   # (N+K, T, 43)
        ref_obj_goals = ref_trajs[:, -1, 36:39]                          # final obj pos
        all_goals     = np.concatenate([goals_arr, ref_obj_goals], axis=0)
        ref_start     = N
        K             = len(ref_trajs)
    else:
        all_trajs = full_traj
        all_goals = goals_arr
        ref_start = -1
        K         = 0

    # Guidance vectors: computed for generated only, padded with zeros for refs
    guidance_vecs = _build_guidance_vecs(obj_traj_w, goals_arr)  # (N, T, 3)
    if K > 0:
        guidance_vecs = np.concatenate(
            [guidance_vecs, np.zeros((K, T, 3), dtype=np.float32)], axis=0
        )

    total_robots = N + K

    # ── Option A: adaptive multi-robot overlay ────────────────────────────────
    if os.path.exists(repeated_xml_path):
        adaptive_xml = None
        try:
            adaptive_xml = build_adaptive_xml(repeated_xml_path, total_robots)
            print(f"\nVisualising {N} generated + {K} reference trajectories "
                  f"with DiffusionOverlayVisualizer…")
            vis = DiffusionOverlayVisualizer(xml_path)
            vis.visualize_overlay(
                x_trajs           = all_trajs,
                repeated_xml_path = adaptive_xml,
                timestep_delay    = 0.01,
                loop              = True,
                guidance_vec      = guidance_vecs,
                world_goal_pos    = all_goals,
                spacing           = 0.0,
                ref_start_idx     = ref_start,
            )
        finally:
            # Remove the temp XML however the visualisation exits
            if adaptive_xml and os.path.exists(adaptive_xml):
                os.unlink(adaptive_xml)
        return

    # ── Option B: single primary robot + overlay paths ────────────────────────
    if not os.path.exists(xml_path):
        print("No mj_model.xml found — skipping visualisation.")
        return

    print(f"\nFallback: MjVisualizer with overlay paths (primary = goal[0])…")

    # All N object trajectories drawn as green path lines
    overlay_paths = [obj_traj_w[i, :, :3] for i in range(N)]

    # All N goal positions as static spheres (tile to match T)
    markers = [np.tile(goals_arr[i:i+1], (T, 1)) for i in range(N)]

    # Guidance vector for primary trajectory
    guidance_vec_primary = guidance_vecs[0]  # (T, 3)

    t_arr = np.arange(T) * 0.01
    vis = MjVisualizer(xml_path, close_on_enter=False)
    vis.visualize_trajectory(
        t            = t_arr,
        x_traj       = full_traj[0],
        repeat       = True,
        guidance_vec = guidance_vec_primary,
        goal_pos     = goals_arr[0],
        overlay_paths= overlay_paths,
        markers      = markers,
    )
    vis.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 1. Config ─────────────────────────────────────────────────────────────
    config_path = "config/config.yaml"
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config(config_path)
    data_path = get_data_path(data_cfg)
    norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    print("Loading dataset…")
    data_buffer = preload_dataset(data_cfg, data_path)
    dataset = FlexibleWindowDataset(
        data_buffer  = data_buffer,
        config       = data_cfg,
        norm_path    = norm_path,
        calculate_stats = False,
        training_cfg = training_cfg,
    )

    # ── 3. Model ──────────────────────────────────────────────────────────────
    diffuser = RobotDiffuser(
        model_config          = model_cfg,
        data_config           = data_cfg,
        training_config       = training_cfg,
        noise_scheduler_config= noise_cfg,
        mode                  = "infer",
        device                = device,
    )
    diffuser.loadWeights(int(args.epoch))

    # ── 4. Initial condition ──────────────────────────────────────────────────
    if args.traj_idx is not None:
        target = (args.traj_idx, args.batch_idx, args.start_time)
        try:
            args.sample_idx = dataset.indices.index(target)
            print(f"Mapped {target} → sample {args.sample_idx}")
        except ValueError:
            raise ValueError(
                f"Target index tuple {target} not found in dataset.indices. "
                "Check --traj_idx / --batch_idx / --start_time."
            )

    _, curr_state, _, anchor = dataset[args.sample_idx]
    file_idx, batch_idx_ds, _ = dataset.indices[args.sample_idx]

    original_goal = anchor['final_obj_pos'].copy()   # (3,)
    ref_obj_pos   = anchor['ref_obj_pos'].copy()     # (3,)
    ref_quat      = anchor['ref_quat'].copy()        # (4,)
    ref_pos       = anchor['ref_pos'].copy()         # (3,) or (7,)

    print(f"\nInitial condition  — sample {args.sample_idx}")
    print(f"  ref_obj_pos  : {ref_obj_pos}")
    print(f"  original_goal: {original_goal}")

    # ── 5. Generate goals ─────────────────────────────────────────────────────
    goals_arr = generate_goals(
        original_goal   = original_goal,
        ref_obj_pos     = ref_obj_pos,
        num_goals       = args.num_goals,
        goal_spread     = args.goal_spread,
        goal_multiplier = args.goal_multiplier,
        rng             = rng,
    )
    N = len(goals_arr)

    print(f"\nGenerated {N} goals  (spread={args.goal_spread}m, "
          f"multiplier={args.goal_multiplier}):")
    for i, g in enumerate(goals_arr):
        tag = "  ← ORIGINAL" if i == 0 else ""
        print(f"  [{i}]  ({g[0]:+.3f}, {g[1]:+.3f}){tag}")

    # ── 6. Build batched initial states & anchors ─────────────────────────────
    curr_state_tens = curr_state.unsqueeze(0).repeat(N, 1, 1)  # (N, H, obs_dim)

    dummy_task_list = []
    for g in goals_arr:
        tp, _ = compute_task_params(
            ref_quat, ref_obj_pos, g,
            normalize_goal_vec = data_cfg.get("normalize_goal_vec", False),
            num_task_params    = data_cfg["num_task_params"],
            max_goal_dist      = dataset.max_obj_displacement,
        )
        dummy_task_list.append(tp)

    dummy_task_arr = np.stack(dummy_task_list)  # (N, task_dim)
    norm_task = dataset._normalize(
        "task_params",
        torch.from_numpy(dummy_task_arr).float()
    )

    anchors_dict = {
        'ref_pos'     : np.tile(ref_pos[None],      (N, 1)),
        'ref_quat'    : np.tile(ref_quat[None],     (N, 1)),
        'ref_obj_pos' : np.tile(ref_obj_pos[None],  (N, 1)),
        'final_obj_pos': goals_arr.copy(),           # (N, 3)
    }

    # ── 7. Stitch steps (auto) ────────────────────────────────────────────────
    if args.stitch_steps is None:
        base_steps = dataset.traj_lengths[file_idx] // data_cfg["num_timesteps"]
        # Scale by the largest goal multiplier / spread distance ratio
        orig_dist  = max(np.linalg.norm(original_goal[:2] - ref_obj_pos[:2]), 1e-6)
        max_dist   = max(np.linalg.norm(g[:2] - ref_obj_pos[:2]) for g in goals_arr)
        scale      = max(1.0, max_dist / orig_dist)
        args.stitch_steps = max(1, int(base_steps * abs(args.goal_multiplier) * scale))
        print(f"\nAuto stitch_steps = {args.stitch_steps}  "
              f"(base={base_steps}, scale={scale:.2f})")

    # ── 8. Batch inference ────────────────────────────────────────────────────
    print(f"\nRunning batch inference for {N} goals…")
    full_traj, obj_traj_w, displacements = run_evaluation_batch(
        args         = args,
        diffuser     = diffuser,
        dataset      = dataset,
        device       = device,
        initial_states = curr_state_tens,
        norm_task_params = norm_task,
        anchors_dict = anchors_dict,
        use_state_cond = True,
        desc         = "Goal Sweep",
    )
    # full_traj  : (N, T, 43)  robot(36) + obj(7)
    # obj_traj_w : (N, T, 7)

    # ── 9. Results summary ────────────────────────────────────────────────────
    print("\nResults:")
    for i in range(N):
        final_pos  = obj_traj_w[i, -1, :2]
        goal_xy    = goals_arr[i, :2]
        goal_err   = np.linalg.norm(goal_xy - final_pos)
        disp_xy    = np.linalg.norm(displacements[i, :2])
        tag = "  ← ORIGINAL" if i == 0 else ""
        print(f"  [{i}]{tag}  goal=({goal_xy[0]:+.3f},{goal_xy[1]:+.3f})  "
              f"err={goal_err:.4f}m  |disp_xy|={disp_xy:.4f}m")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.save(args.save_path, full_traj)
    goals_save = args.save_path.replace('.npy', '_goals.npy')
    np.save(goals_save, goals_arr)
    print(f"\nSaved trajectories → {args.save_path}  {full_traj.shape}")
    print(f"Saved goals        → {goals_save}  {goals_arr.shape}")

    # ── 11. Visualise ─────────────────────────────────────────────────────────
    if args.visualize:
        visualize_results(full_traj, obj_traj_w, goals_arr, args,
                          dataset=dataset, ref_pos=ref_pos)


if __name__ == "__main__":
    main()

"""Generate a trajectory from a diffusion checkpoint and visualise it in MuJoCo.

Loads a trained diffusion model, generates one (or several) trajectories
conditioned on task-space goals (e.g. relative object placement), then
plays them back as kinematic rollouts in MuJoCo's interactive viewer.

The conditioning trajectory for each goal is automatically selected as the
nearest-neighbour from an elite archive (``elites.json``), exactly as the
evolutionary pipeline does.

Usage examples
--------------
# Single goal – nearest elite used as condition
uv run python src/mjlab/scripts/diffusion_planner/generate_and_visualize.py \
    --elites results/gen_002/elites.json \
    --evo_task pick_place_relative_box_pose \
    --checkpoint diffusion_gen.pt \
    --goal 1.0 -0.5

# Multiple distinct goals (each gets its own nearest-neighbour condition)
uv run python src/mjlab/scripts/diffusion_planner/generate_and_visualize.py \
    --elites results/gen_002/elites.json \
    --evo_task pick_place_relative_box_pose \
    --checkpoint diffusion_gen.pt \
    --goal 1.0 -0.5  --goal 0.5 0.2  --goal 2.0 -1.0

# N random samples per goal
uv run python src/mjlab/scripts/diffusion_planner/generate_and_visualize.py \
    --elites results/gen_002/elites.json \
    --evo_task pick_place_relative_box_pose \
    --checkpoint diffusion_gen.pt \
    --goal 1.0 -0.5 \
    --num_samples 5

# Without an archive – use a base motion file directly
uv run python src/mjlab/scripts/diffusion_planner/generate_and_visualize.py \
    --base_motion recorded_positions.npz \
    --checkpoint diffusion_gen.pt \
    --goal 1.0 -0.5

# Record video
... --video --video_out my_gen.mp4

# Save trajectories to disk
... --save_npz generated_trajs.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import mujoco
import mujoco.viewer

# ── Path setup so diffusion planner internals resolve ─────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from mjlab.scripts.diffusion_planner.motion_generator import MotionGenerator

# Re-use the MuJoCo overlay helpers from the FoLM visualiser
from mjlab.scripts.FoLM.utils.visualize_overlay_trajs import (
    _build_multi_robot_model,
    _build_qpos_all,
    _QPOS_PER_COPY,
    _split_multi_env,
)
from mjlab.scripts.FoLM.archive import EliteArchive, Elite
from mjlab.scripts.FoLM.evo_task_config import get_task_config, TaskConfig

# ────────────────────────── defaults ──────────────────────────

_DEFAULT_XML = "src/mjlab/scripts/diffusion_planner/mj_model.xml"
_DEFAULT_CONFIG = str(_SCRIPT_DIR / "config" / "config.yaml")


# ──────────────── diffusion output → traj dict ────────────────


def _raw_traj_to_dict(
    traj: np.ndarray,
    template: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Convert a raw diffusion output ``(T, 43)`` into a keyed traj dict.

    The dict has the same structure as the .npz files the pipeline produces
    (``body_pos_w``, ``body_quat_w``, ``joint_pos``, ``object_pos_w``, …)
    so it can be fed directly into ``_build_qpos_all``.
    """
    T = traj.shape[0]
    num_bodies = template["body_pos_w"].shape[1] if "body_pos_w" in template else 1

    d: Dict[str, np.ndarray] = {}

    # Root body
    body_pos = np.zeros((T, num_bodies, 3), dtype=np.float64)
    body_pos[:, 0, :] = traj[:, :3]
    d["body_pos_w"] = body_pos

    body_quat = np.zeros((T, num_bodies, 4), dtype=np.float64)
    body_quat[:, 0, :] = traj[:, 3:7]
    body_quat[:, 1:, 0] = 1.0  # identity quat for non-root bodies
    d["body_quat_w"] = body_quat

    # Joints (29 DoF)
    d["joint_pos"] = traj[:, 7:36]

    # Object (if present in the output)
    if traj.shape[1] > 36:
        d["object_pos_w"] = traj[:, 36:39]
        d["object_quat_w"] = traj[:, 39:43]

    return d


# ──────────────────── base-motion loading ─────────────────────


def _load_base_motions(path: Path) -> List[Dict[str, np.ndarray]]:
    """Load base motions from a file or directory (same logic as pipeline)."""
    all_base: List[Dict[str, np.ndarray]] = []

    if path.is_dir():
        for f in sorted(path.glob("*.npz")):
            try:
                data = dict(np.load(f))
                if "joint_pos" in data and "body_pos_w" in data:
                    all_base.append(data)
            except Exception as e:
                print(f"[WARN] Skipping {f}: {e}")

    elif path.suffix == ".npz":
        raw = dict(np.load(path, allow_pickle=True))
        if "joint_pos" in raw and raw["joint_pos"].ndim == 3:
            # Batched file — split into individual trajectories
            B = raw["joint_pos"].shape[0]
            for i in range(B):
                single: Dict[str, np.ndarray] = {}
                for k, v in raw.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == B:
                        single[k] = v[i]
                    else:
                        single[k] = v
                all_base.append(single)
        else:
            all_base.append(raw)
    else:
        raise ValueError(f"Unknown base motion format: {path}")

    if not all_base:
        raise ValueError(f"No valid base motions found at {path}")

    print(f"[INFO] Loaded {len(all_base)} base trajectory(ies) from {path}")
    return all_base


# ──────────────── generation + visualisation ──────────────────


def generate(
    diff_generator: MotionGenerator,
    goals: List[np.ndarray],
    num_samples: int,
    *,
    archive: EliteArchive | None = None,
    task_config: TaskConfig | None = None,
    fallback_base: Dict[str, np.ndarray] | None = None,
    target_length: int | None = None,
) -> List[Dict[str, np.ndarray]]:
    """Run diffusion inference and return a list of traj dicts.

    For each goal the nearest-neighbour elite from *archive* is used as the
    conditioning trajectory (exactly as the evolutionary pipeline does).
    If no archive is provided, *fallback_base* is used for every goal.
    """
    if archive is None and fallback_base is None:
        raise ValueError("Either --elites (archive) or --base_motion must be provided.")

    H = diff_generator.data_cfg.get("state_history", 1)

    all_trajs: List[Dict[str, np.ndarray]] = []

    # Expand: each goal × num_samples
    batch_robot: List[np.ndarray] = []
    batch_obj: List[np.ndarray] = []
    batch_goals: List[np.ndarray] = []
    batch_templates: List[Dict[str, np.ndarray]] = []
    tgt_len: int | None = None

    for goal in goals:
        # --- Select conditioning trajectory ---
        if archive is not None and task_config is not None and len(archive) > 0:
            dim_names = [d.name for d in task_config.dimensions]
            target_dict = {dim_names[i]: float(goal[i]) for i in range(len(goal))}
            closest = archive.get_closest(target_dict)
            if closest is not None:
                sel_base = closest.data
                dist = task_config.distance(closest.task_metrics, target_dict)
                print(f"  Goal {goal} → nearest elite (dist={dist:.3f}, "
                      f"fitness={closest.fitness:.3f}, source={closest.source})")
            else:
                sel_base = fallback_base
                print(f"  Goal {goal} → no elite found, using fallback base")
        else:
            sel_base = fallback_base

        if sel_base is None:
            raise RuntimeError("No conditioning trajectory available for goal "
                               f"{goal}. Provide --elites or --base_motion.")

        # --- Build initial condition (same as pipeline) ---
        base_bp = sel_base["body_pos_w"]
        base_bq = sel_base["body_quat_w"]
        base_jp = sel_base["joint_pos"]

        r_hist = np.concatenate(
            [base_bp[:H, 0, :], base_bq[:H, 0, :], base_jp[:H, :]], axis=-1
        )

        for _ in range(num_samples):
            batch_robot.append(r_hist)
            if "object_pos_w" in sel_base:
                o_hist = np.concatenate(
                    [sel_base["object_pos_w"][:H, :],
                     sel_base["object_quat_w"][:H, :]],
                    axis=-1,
                )
                batch_obj.append(o_hist)
            batch_goals.append(goal)
            batch_templates.append(sel_base)

        if tgt_len is None:
            tgt_len = target_length or base_jp.shape[0]

    # --- Run batched diffusion generation ---
    ic: Dict[str, np.ndarray] = {"robot": np.stack(batch_robot)}
    if batch_obj:
        ic["obj"] = np.stack(batch_obj)
    goals_arr = np.array(batch_goals, dtype=np.float64)

    total = len(batch_robot)
    print(f"\n[Diffusion] Generating {total} sample(s) "
          f"({len(goals)} goal(s) × {num_samples} sample(s)), "
          f"target length {tgt_len} …")
    t0 = time.time()
    outs = diff_generator.generate_trajectory(
        initial_condition=ic,
        goal_condition=goals_arr,
        num_samples=1,          # already replicated above
        target_traj_length=tgt_len,
    )
    print(f"[Diffusion] Done in {time.time() - t0:.2f}s  →  shape {outs.shape}")

    # Convert to traj dicts
    for i in range(outs.shape[0]):
        all_trajs.append(_raw_traj_to_dict(outs[i], batch_templates[i]))
    return all_trajs


# ──────────────────── viewer / video ──────────────────────────


def visualize(
    trajs: List[Dict[str, np.ndarray]],
    *,
    xml_path: str = _DEFAULT_XML,
    fps: float = 50.0,
    ghost_alpha: float = 0.5,
    video: bool = False,
    video_out: str | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
    distance: float = 3.5,
    elevation: float = -15.0,
    azimuth: float = 140.0,
    track: bool = True,
) -> None:
    """Play back trajectory dicts in a MuJoCo kinematic viewer."""
    N = len(trajs)
    if N == 0:
        print("No trajectories to visualise.")
        return

    # First trajectory is opaque, rest are ghosts
    copy_tints: list[Tuple[float, float, float] | None] = [None] + [
        (0.20, 0.60, 1.00)  # blue ghost tint
    ] * (N - 1)

    print(f"\nBuilding model with {N} robot(s) …")
    model = _build_multi_robot_model(xml_path, N, ghost_alpha, copy_tints)
    data = mujoco.MjData(model)

    expected_nq = N * _QPOS_PER_COPY
    assert model.nq == expected_nq, (
        f"nq mismatch: model.nq={model.nq}, expected={expected_nq}"
    )

    qpos_all = _build_qpos_all(trajs, model)
    T_max = qpos_all.shape[0]
    print(f"  Samples: {N}  |  Max length: {T_max} frames ({T_max / fps:.1f}s)")

    # Helper: write qpos for timestep t
    def _set_qpos(t: int) -> None:
        for i in range(N):
            start = i * _QPOS_PER_COPY
            data.qpos[start : start + _QPOS_PER_COPY] = qpos_all[t, i]
        mujoco.mj_forward(model, data)

    # ── Camera setup helper ───────────────────────────────────
    def _setup_cam(cam: mujoco.MjvCamera) -> None:
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
        if track:
            pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = pelvis_id
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = qpos_all[0, 0, 0:3]

    # ── VIDEO mode ────────────────────────────────────────────
    if video:
        from moviepy import ImageSequenceClip

        out_path = video_out or "diffusion_gen.mp4"
        print(f"\n  Recording video → {out_path}")

        model.vis.global_.offwidth = max(model.vis.global_.offwidth, video_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, video_height)
        renderer = mujoco.Renderer(model, height=video_height, width=video_width)

        cam = mujoco.MjvCamera()
        _setup_cam(cam)

        frames: list[np.ndarray] = []
        for t in range(T_max):
            _set_qpos(t)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frames.append(frame)
            if (t + 1) % 100 == 0 or t == T_max - 1:
                print(f"  rendered {t + 1}/{T_max}")

        renderer.close()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(out_path)
        print(f"\n  ✓ Saved {out_path}")
        return

    # ── INTERACTIVE viewer ────────────────────────────────────
    print(f"\n  Controls:")
    print(f"    SPACE  – pause / resume")
    print(f"    → ←    – step forward / backward (while paused)")
    print(f"    ENTER  – exit\n")

    paused = {"v": False}
    step_req = {"d": 0}
    quit_req = {"v": False}

    def key_cb(keycode: int) -> None:
        if keycode == 32:       # space
            paused["v"] = not paused["v"]
        elif keycode == 262:    # right
            step_req["d"] = 1
            paused["v"] = True
        elif keycode == 263:    # left
            step_req["d"] = -1
            paused["v"] = True
        elif keycode == 257:    # enter
            quit_req["v"] = True

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_cb)

    step = 0
    try:
        while viewer.is_running() and not quit_req["v"]:
            t0 = time.time()
            t = step % T_max
            _set_qpos(t)
            viewer.sync()
            elapsed = time.time() - t0
            time.sleep(max(0.0, 1.0 / fps - elapsed))
            if paused["v"]:
                if step_req["d"]:
                    step = (step + step_req["d"]) % T_max
                    step_req["d"] = 0
            else:
                step += 1
    finally:
        viewer.close()

    print("Done.")


# ──────────────────────── CLI entry ───────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a diffusion trajectory and visualise in MuJoCo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data sources (at least one required) ──────────────────
    ap.add_argument("--elites", type=str, default=None,
                    help="Path to elites.json — nearest-neighbour conditioning (recommended)")
    ap.add_argument("--evo_task", type=str, default=None,
                    help="Evolutionary task config name (e.g. pick_place_relative_box_pose). "
                         "Required when --elites is used.")
    ap.add_argument("--base_motion", type=str, default=None,
                    help="Path to base motion .npz file or directory (fallback if no elites, "
                         "also used as data_source for loading normalization stats)")

    # ── Required ──────────────────────────────────────────────
    ap.add_argument("--checkpoint", required=True,
                    help="Path to diffusion model checkpoint (.pt/.pth)")
    ap.add_argument("--goal", type=float, nargs="+", action="append", required=True,
                    help="Task-space goal values (e.g. --goal 1.0 -0.5). "
                         "Repeat for multiple goals: --goal 1.0 -0.5 --goal 0.5 0.2")

    # ── Optional — generation ─────────────────────────────────
    ap.add_argument("--num_samples", type=int, default=1,
                    help="Number of trajectories per goal (default: 1)")
    ap.add_argument("--diff_config", type=str, default=None,
                    help="Path to diffusion config YAML (default: auto)")
    ap.add_argument("--device", type=str, default="cuda:0",
                    help="Torch device (default: cuda:0)")
    ap.add_argument("--target_length", type=int, default=None,
                    help="Target trajectory length; defaults to base motion length")

    # ── Optional — visualisation ──────────────────────────────
    ap.add_argument("--xml", type=str, default=_DEFAULT_XML,
                    help="MuJoCo XML for kinematic playback")
    ap.add_argument("--fps", type=float, default=50.0, help="Playback FPS")
    ap.add_argument("--ghost_alpha", type=float, default=0.5,
                    help="Ghost transparency (0=invisible, 1=opaque)")
    ap.add_argument("--distance", type=float, default=3.5, help="Camera distance")
    ap.add_argument("--elevation", type=float, default=-15, help="Camera elevation")
    ap.add_argument("--azimuth", type=float, default=140, help="Camera azimuth")
    ap.add_argument("--track", action="store_true", default=True,
                    help="Camera tracks the first robot's pelvis")

    # ── Optional — output ─────────────────────────────────────
    ap.add_argument("--video", action="store_true",
                    help="Record video instead of launching interactive viewer")
    ap.add_argument("--video_out", type=str, default=None,
                    help="Video output path (default: diffusion_gen.mp4)")
    ap.add_argument("--video_width", type=int, default=1920)
    ap.add_argument("--video_height", type=int, default=1080)
    ap.add_argument("--save_npz", type=str, default=None,
                    help="Save generated trajectories to a batched .npz file")
    ap.add_argument("--include_base", action="store_true",
                    help="Include the nearest-neighbour conditioning trajectory in the visualisation "
                         "(shown first / opaque)")

    args = ap.parse_args()

    # ── Validate arguments ────────────────────────────────────
    if args.elites is None and args.base_motion is None:
        ap.error("At least one of --elites or --base_motion must be provided.")
    if args.elites is not None and args.evo_task is None:
        ap.error("--evo_task is required when using --elites.")

    # ── Load elite archive (optional) ─────────────────────────
    archive: EliteArchive | None = None
    task_config: TaskConfig | None = None

    if args.elites:
        task_config = get_task_config(args.evo_task)
        archive = EliteArchive(task_config=task_config)
        archive.load(args.elites)
        print(f"[INFO] Archive loaded: {len(archive)} elites")

    # ── Load base motions (for stats / fallback) ──────────────
    all_base: List[Dict[str, np.ndarray]] = []
    fallback_base: Dict[str, np.ndarray] | None = None

    if args.base_motion:
        base_path = Path(args.base_motion).resolve()
        all_base = _load_base_motions(base_path)
        fallback_base = all_base[0]

    # If no explicit base_motion but we have elites, collect elite data
    # as the data_source for loading normalization stats
    if not all_base and archive is not None and len(archive) > 0:
        all_base = [e.data for e in archive.values()]
        fallback_base = all_base[0]
        print(f"[INFO] Using {len(all_base)} elite trajectories as data source for norm stats")

    if not all_base:
        ap.error("No trajectory data available. Provide --base_motion or --elites with loaded data.")

    # ── Initialise diffusion model ────────────────────────────
    device = args.device if torch.cuda.is_available() else "cpu"
    cfg_path = args.diff_config or _DEFAULT_CONFIG
    print(f"[INFO] Initialising MotionGenerator with config: {cfg_path}")
    diff_gen = MotionGenerator(config_path=cfg_path, device=device)

    # Load weights via fit(epochs=0)
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    diff_gen.fit(
        data_source=all_base,
        epochs=0,
        checkpoint=args.checkpoint,
    )

    # ── Parse goals ───────────────────────────────────────────
    # args.goal is List[List[float]] due to action="append"
    goals = [np.array(g, dtype=np.float64) for g in args.goal]
    if task_config is not None:
        n_dims = len(task_config.dimensions)
        dim_names = [d.name for d in task_config.dimensions]
        for i, g in enumerate(goals):
            if len(g) != n_dims:
                ap.error(f"Goal {i} has {len(g)} values but task "
                         f"'{args.evo_task}' expects {n_dims}: {dim_names}")
        print(f"[INFO] Task dimensions: {dim_names}")

    print(f"[INFO] Goals: {[g.tolist() for g in goals]}")

    # ── Generate trajectories ─────────────────────────────────
    trajs = generate(
        diff_gen,
        goals,
        num_samples=args.num_samples,
        archive=archive,
        task_config=task_config,
        fallback_base=fallback_base,
        target_length=args.target_length,
    )

    # ── Optionally prepend the conditioning trajectory(ies) ───
    if args.include_base and archive is not None and task_config is not None:
        # Prepend the closest elite for the FIRST goal
        dim_names = [d.name for d in task_config.dimensions]
        target_dict = {dim_names[i]: float(goals[0][i]) for i in range(len(goals[0]))}
        closest = archive.get_closest(target_dict)
        if closest is not None:
            trajs = [closest.data] + trajs
    elif args.include_base and fallback_base is not None:
        trajs = [fallback_base] + trajs

    # ── Save to disk (optional) ───────────────────────────────
    if args.save_npz:
        out_path = Path(args.save_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Stack into batched arrays
        batched: Dict[str, np.ndarray] = {}
        all_keys = set()
        for t in trajs:
            all_keys.update(t.keys())
        for key in sorted(all_keys):
            arrays = [t.get(key) for t in trajs if key in t]
            if not arrays or not isinstance(arrays[0], np.ndarray) or arrays[0].ndim == 0:
                continue
            max_t = max(a.shape[0] for a in arrays)
            trailing = arrays[0].shape[1:]
            padded = np.zeros((len(trajs), max_t, *trailing), dtype=arrays[0].dtype)
            for i, a in enumerate(arrays):
                if a is not None:
                    padded[i, : a.shape[0]] = a
            batched[key] = padded
        np.savez(out_path, **batched)
        print(f"\n[INFO] Saved {len(trajs)} trajectories to {out_path}")

    # ── Visualise ─────────────────────────────────────────────
    visualize(
        trajs,
        xml_path=args.xml,
        fps=args.fps,
        ghost_alpha=args.ghost_alpha,
        video=args.video,
        video_out=args.video_out,
        video_width=args.video_width,
        video_height=args.video_height,
        distance=args.distance,
        elevation=args.elevation,
        azimuth=args.azimuth,
        track=args.track,
    )


if __name__ == "__main__":
    main()
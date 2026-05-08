"""
Visualize the carry-walk approach-phase augmentation side-by-side with the original.

For each carry-while-walking trajectory the script:
  1. Loads the raw base / joints / obj arrays.
  2. Runs _try_carry_approach_augmentation (regardless of the enabled flag in config)
     so you can preview what the augmentation produces.
  3. Shows both trajectories together – original (left, robot carries moving box) vs
     augmented (right, same robot motion but box frozen at final drop location).

Modes
-----
  interactive (default) : opens a passive MuJoCo viewer via GLFW.
                          ESC → next trajectory.
  --video VIDEO_PATH    : headless EGL/OSMesa subprocess render; produces an
                          mp4 with original (left) and augmented (right) frames
                          stacked side-by-side. Suitable for Karolina / Barbora
                          cluster nodes that have no display.

Usage
-----
  python visualize_approach_augmentation.py
  python visualize_approach_augmentation.py --traj_idx 3 --batch_idx 0
  python visualize_approach_augmentation.py --start_idx 0 --num_trajs 5
  python visualize_approach_augmentation.py --traj_idx 3 --video results/aug_vis.mp4
  python visualize_approach_augmentation.py --traj_idx 3 --video results/aug_vis.mp4 --no_video_interactive
"""

import argparse
import os
import sys
import subprocess
import tempfile

import numpy as np

from config.configure import load_config, get_data_path, get_norm_path, get_mj_xml_paths
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Headless two-panel renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_side_by_side_headless(
    orig_state,       # (T, 43) world-frame full state  – original carry traj
    aug_state,        # (T, 43) world-frame full state  – augmented (box frozen)
    goal_world,       # (3,)  world-frame goal position
    goal_obj_quat,    # (4,)  [qw, qx, qy, qz] goal orientation
    out_path,
    xml_path,
    fps=33,
    width=1280,
    height=720,
):
    """Render original (left) and augmented (right) trajectories to a side-by-side mp4."""
    gw = np.asarray(goal_world, dtype=np.float64)
    gr = np.asarray(goal_obj_quat, dtype=np.float64) if goal_obj_quat is not None \
         else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="aug_vis_render_") as td:
        payload = os.path.join(td, "payload.npz")
        np.savez_compressed(
            payload,
            orig_state=orig_state.astype(np.float64),
            aug_state=aug_state.astype(np.float64),
            goal_world=gw[:3],
            goal_obj_quat=gr,
        )

        render_code = r'''
import os, sys, numpy as np, imageio, mujoco

payload, out_path, xml_path, fps, width, height = sys.argv[1:]
fps = int(fps); width = int(width); height = int(height)

arr = np.load(payload)
orig  = arr["orig_state"]   # (T, 43)
aug   = arr["aug_state"]    # (T, 43)
gw    = arr["goal_world"]
gr    = arr["goal_obj_quat"]

def make_cam(model, state, goal):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    obj0 = state[0, 36:39]
    center = (goal + obj0 + state[-1, 36:39]) / 3.0
    r_xy = max(
        float(np.linalg.norm(goal[:2] - center[:2])),
        float(np.linalg.norm(obj0[:2]  - center[:2])),
        2.0,
    )
    cam.lookat[:] = [float(center[0]), float(center[1]), float(center[2]) + 0.8]
    cam.distance  = max(3.0, 1.4 + 2.2 * r_xy)
    cam.azimuth   = 135.0
    cam.elevation = -16.0
    return cam

def find_mocap(model, *names):
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            mid = model.body_mocapid[bid]
            if mid >= 0:
                return mid
    return -1

def quat_z_to_vec(v):
    z = np.array([0.,0.,1.])
    n = np.linalg.norm(v)
    if n < 1e-9: return np.array([1.,0.,0.,0.])
    t = v/n; cross = np.cross(z, t); dot = float(np.dot(z, t)); cn = np.linalg.norm(cross)
    if cn < 1e-9: return np.array([1.,0.,0.,0.]) if dot>0 else np.array([0.,1.,0.,0.])
    axis = cross/cn; ang = np.arccos(np.clip(dot,-1.,1.)); h = 0.5*ang; s = np.sin(h)
    return np.array([np.cos(h), axis[0]*s, axis[1]*s, axis[2]*s])

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)
model.vis.global_.offwidth  = max(model.vis.global_.offwidth,  width)
model.vis.global_.offheight = max(model.vis.global_.offheight, height)

goal_mid   = find_mocap(model, "goal_marker", "goal_marker_0")
arrow_mid  = find_mocap(model, "guidance_arrow", "guidance_arrow_0")

cam_orig = make_cam(model, orig, gw)
cam_aug  = make_cam(model, aug,  gw)

T     = orig.shape[0]
nq    = model.nq

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

half_w   = width // 2
renderer = mujoco.Renderer(model, height=height, width=half_w)
try:
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]     = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG]        = 0
except Exception: pass

with imageio.get_writer(out_path, fps=fps) as writer:
    for t in range(T):
        # ── left panel: original (robot carries box) ──────────────────
        data.qpos[:nq] = orig[t, :nq]
        if goal_mid >= 0:
            data.mocap_pos[goal_mid]  = gw
            data.mocap_quat[goal_mid] = gr
        if arrow_mid >= 0:
            d = gw - orig[t, 36:39]
            data.mocap_pos[arrow_mid]  = orig[t, 36:39]
            data.mocap_quat[arrow_mid] = quat_z_to_vec(d)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        renderer.update_scene(data, camera=cam_orig)
        frame_left = renderer.render()

        # ── right panel: augmented (box frozen at drop location) ──────
        data.qpos[:nq] = aug[t, :nq]
        if goal_mid >= 0:
            data.mocap_pos[goal_mid]  = gw
            data.mocap_quat[goal_mid] = gr
        if arrow_mid >= 0:
            d = gw - aug[t, 36:39]
            data.mocap_pos[arrow_mid]  = aug[t, 36:39]
            data.mocap_quat[arrow_mid] = quat_z_to_vec(d)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        renderer.update_scene(data, camera=cam_aug)
        frame_right = renderer.render()

        if frame_left.dtype != np.uint8:
            frame_left  = (np.clip(frame_left,  0., 1.) * 255).astype(np.uint8)
            frame_right = (np.clip(frame_right, 0., 1.) * 255).astype(np.uint8)

        if frame_left.shape[0] != frame_right.shape[0]:
            h_min = min(frame_left.shape[0], frame_right.shape[0])
            frame_left  = frame_left[:h_min]
            frame_right = frame_right[:h_min]

        writer.append_data(np.concatenate([frame_left, frame_right], axis=1))

renderer.close()
print(f"Saved to {out_path}")
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
                [sys.executable, "-c", render_code,
                 payload, out_path, xml_path,
                 str(int(fps)), str(int(width)), str(int(height))],
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                print(f"[video] Saved side-by-side video → {out_path}")
                return
            errors.append(
                f"backend={backend}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        raise RuntimeError(
            "Headless video render failed for EGL and OSMesa backends.\n"
            + "\n---\n".join(errors)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Interactive viewer (two sequential passes: original → augmented)
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_interactive(orig_state, aug_state, goal_world, goal_obj_quat, xml_path, dt):
    """Show original then augmented in a shared GLFW viewer window.

    Plays the original carry trajectory first, then ESC advances to the
    augmented version where the box is frozen at the drop location.
    """
    from utils.visualize.visualize import MjVisualizer

    goal_pos  = np.asarray(goal_world, dtype=np.float64)
    goal_quat = np.asarray(goal_obj_quat, dtype=np.float64) if goal_obj_quat is not None \
                else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    vis = MjVisualizer(xml_path, close_on_enter=False)

    for label, state in [("ORIGINAL (box moves with robot)", orig_state),
                          ("AUGMENTED (box frozen at drop location)", aug_state)]:
        T = len(state)
        t_arr = np.linspace(0.0, T * dt, T)
        print(f"  [{label}]  T={T}  ESC to advance")
        vis.step = 0
        vis.visualize_trajectory(
            t_arr, state,
            repeat=True,
            goal_pos=goal_pos,
            goal_quat=goal_quat,
        )
        if not vis.viewer or not vis.viewer.is_running():
            break

    vis.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Visualize carry-walk approach-phase augmentation")
    parser.add_argument("--traj_idx",  type=int, default=None,
                        help="File (trajectory) index in the dataset")
    parser.add_argument("--batch_idx", type=int, default=0,
                        help="Batch index within the file")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start from this unique-trajectory index (carry trajs only)")
    parser.add_argument("--num_trajs", type=int, default=1,
                        help="How many carry trajectories to process")
    parser.add_argument("--video", type=str, default=None,
                        help="Save side-by-side mp4 at this path (headless, cluster-safe). "
                             "Use {idx} as placeholder if processing multiple trajectories.")
    parser.add_argument("--fps",   type=int, default=33)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height",type=int, default=720)
    parser.add_argument("--no_video_interactive", action="store_true", default=True,
                        help="When --video is given, skip the interactive viewer as well")
    args = parser.parse_args()

    # ── Load config & dataset ───────────────────────────────────────────────
    model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml")
    data_path = get_data_path(data_cfg)
    norm_path  = get_norm_path(model_cfg, training_cfg, data_cfg)
    xml_path, _ = get_mj_xml_paths()

    print("Loading dataset...")
    data_buffer = preload_dataset(data_cfg, data_path)
    dataset = FlexibleWindowDataset(
        data_buffer=data_buffer,
        config=data_cfg,
        norm_path=norm_path,
        calculate_stats=False,
        training_cfg={},
    )

    dt = float(data_cfg.get("raw_dt", 0.03)) * int(data_cfg.get("downsample", 1))

    # ── Collect unique trajectories ─────────────────────────────────────────
    unique_trajs = sorted(list(set((f, b) for f, b, t in dataset.indices)))
    print(f"Found {len(unique_trajs)} unique trajectories.")

    if args.traj_idx is not None:
        target = (args.traj_idx, args.batch_idx)
        try:
            start = unique_trajs.index(target)
        except ValueError:
            print(f"Error: ({args.traj_idx}, {args.batch_idx}) not found.")
            sys.exit(1)
        # Collect carry-walk trajectories starting from the requested index
        carry_trajs = [
            (f, b) for f, b in unique_trajs[start:]
            if dataset._is_carry_walk_trajectory(dataset.ram_cache[f], b)
        ]
        if not carry_trajs:
            print(f"No carry-walk trajectory found at or after ({args.traj_idx}, {args.batch_idx}).")
            sys.exit(1)
        if carry_trajs[0] != (args.traj_idx, args.batch_idx):
            print(f"  ({args.traj_idx}, {args.batch_idx}) is not carry-walk; "
                  f"advancing to ({carry_trajs[0][0]}, {carry_trajs[0][1]})")
        traj_list = carry_trajs[:args.num_trajs]
    else:
        # Only show carry-walk trajectories when iterating
        carry_trajs = [
            (f, b) for f, b in unique_trajs
            if dataset._is_carry_walk_trajectory(dataset.ram_cache[f], b)
        ]
        print(f"Found {len(carry_trajs)} carry-walk trajectories.")
        end = min(args.start_idx + args.num_trajs, len(carry_trajs))
        traj_list = carry_trajs[args.start_idx:end]

    # ── Process trajectories ────────────────────────────────────────────────
    for vis_idx, (file_idx, batch_idx) in enumerate(traj_list):
        file_meta   = dataset.ram_cache[file_idx] if 0 <= file_idx < len(dataset.ram_cache) else {}
        source_file = os.path.basename(str(file_meta.get("source_file_path", "unknown")))

        print(f"\n[{vis_idx+1}/{len(traj_list)}] file={file_idx}, batch={batch_idx}  ({source_file})")

        raw_traj = dataset._get_single_traj(file_idx, batch_idx)
        if raw_traj is None:
            print("  Skipping — _get_single_traj returned None")
            continue

        base   = raw_traj["base"]
        joints = raw_traj["joints"]
        obj    = raw_traj["obj"]
        T      = min(len(base), len(joints), len(obj))
        base   = base[:T];  joints = joints[:T];  obj = obj[:T]

        # ── Run augmentation ────────────────────────────────────────────────
        aug_result = dataset._try_carry_approach_augmentation(base, obj, joints)
        if aug_result is None:
            print("  Augmentation returned None (trajectory too short). Skipping.")
            continue

        base_aug   = aug_result["base"][0]    # (T, 7)
        joints_aug = aug_result["joints"][0]  # (T, 29)
        obj_aug    = aug_result["obj"][0]     # (T, 7)  — frozen at final position
        goal_world = aug_result["goal_obj_world"][0]  # (3,)
        goal_quat  = aug_result.get("goal_obj_quat")
        if goal_quat is not None:
            goal_quat = goal_quat[0]  # (4,)

        obj_travel = float(np.linalg.norm(obj[-1, :3] - obj[0, :3]))
        print(f"  T={T}")
        print(f"  Box travel in original: {obj_travel:.3f} m")
        print(f"  Goal (frozen drop loc): {goal_world.round(3)}")
        print(f"  Box in augmented: always at {obj_aug[0, :3].round(3)}")

        # ── Build 43-dim full_state arrays ──────────────────────────────────
        orig_state = np.concatenate([base, joints, obj], axis=-1).astype(np.float32)
        aug_state  = np.concatenate([base_aug, joints_aug, obj_aug], axis=-1).astype(np.float32)

        # ── Video (headless) ────────────────────────────────────────────────
        if args.video:
            out = args.video.replace("{idx}", str(vis_idx))
            if len(traj_list) > 1 and "{idx}" not in args.video:
                base_name, ext = os.path.splitext(args.video)
                out = f"{base_name}_{vis_idx:03d}{ext}"
            print(f"  Rendering side-by-side video → {out}")
            _render_side_by_side_headless(
                orig_state, aug_state, goal_world, goal_quat, out, xml_path,
                fps=args.fps, width=args.width, height=args.height,
            )

        # ── Interactive viewer ───────────────────────────────────────────────
        if not args.video or not args.no_video_interactive:
            print("  Interactive viewer: ORIGINAL first, then AUGMENTED.  ESC = next")
            _visualize_interactive(orig_state, aug_state, goal_world, goal_quat, xml_path, dt)

    print("\nDone.")


if __name__ == "__main__":
    main()

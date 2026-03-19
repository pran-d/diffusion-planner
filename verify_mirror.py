"""
Verify mirror symmetry augmentation by visualising original and mirrored
trajectories side-by-side in MuJoCo.

Uses ``DiffusionOverlayVisualizer.visualize_overlay`` from the existing
codebase — the same visualiser used for generated-vs-ground-truth
comparisons — so the look-and-feel is identical.

The mirrored copy is coloured sky-blue (``ref_start_idx=1``) so you can
instantly tell the two apart.

Controls  (inside the viewer):
    Space       — pause / resume
    Right arrow — step forward (while paused)
    Left arrow  — step backward (while paused)
    R           — restart from frame 0

Usage:
    python verify_mirror.py                          # random sample from test data
    python verify_mirror.py --npz path/to/traj.npz   # specific trajectory file
    python verify_mirror.py --idx 5                  # specific dataset window index
    python verify_mirror.py --offset 2.0             # lateral offset between robots
    python verify_mirror.py --diag-only              # print diagnostics, skip viewer
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from config.configure import get_mj_xml_paths

# ────────────────────────── constants ──────────────────────────

# qpos layout for a single G1 + box (43 DOF):
#   [0:7]   base free-joint  (x, y, z, qw, qx, qy, qz)
#   [7:13]  left  leg joints (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   [13:19] right leg joints
#   [19:22] waist joints     (yaw, roll, pitch)
#   [22:29] left  arm joints (sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw)
#   [29:36] right arm joints
#   [36:43] object free-joint (x, y, z, qw, qx, qy, qz)

NQ_SINGLE = 43           # single robot + box
NQ_JOINTS = 29            # hinge joints per robot
BASE_SLICE = slice(0, 7)
JOINTS_SLICE = slice(7, 36)
OBJ_SLICE = slice(36, 43)

# Joint index ranges *within* the 29-joint block
LEFT_LEG  = slice(0, 6)
RIGHT_LEG = slice(6, 12)
WAIST     = slice(12, 15)
LEFT_ARM  = slice(15, 22)
RIGHT_ARM = slice(22, 29)

# Within each 6-DOF leg:  pitch=0, roll=1, yaw=2, knee=3, ankle_pitch=4, ankle_roll=5
# Within each 7-DOF arm:  sh_pitch=0, sh_roll=1, sh_yaw=2, elbow=3, wr_roll=4, wr_pitch=5, wr_yaw=6

# Correct reflections based on user spec:
# Legs: hip_roll, hip_yaw, ankle_roll (1, 2, 5) -> same as before
# Arms: sh_roll, sh_yaw, wr_roll, wr_yaw (1, 2, 4, 6) -> wr_roll is 4, wr_yaw is 6
# Waist: yaw, roll (0, 1) -> pitch is 2

LEG_NEGATE_OFFSETS = [1, 2, 5]         # hip_roll, hip_yaw, ankle_roll
ARM_NEGATE_OFFSETS = [1, 2, 4, 6]      # sh_roll, sh_yaw, wr_roll, wr_yaw
WAIST_NEGATE_OFFSETS = [0, 1]           # waist_yaw, waist_roll


def reflect_quaternion_y(quat: np.ndarray) -> np.ndarray:
    """Reflect a quaternion (w, x, y, z) about the xz-plane (negate y).

    A reflection S = diag(1, -1, 1) acting on rotation R gives R' = S R S.
    In quaternion form: (w, x, y, z) → (w, -1, y, -z).
    """
    q = quat.copy()
    q[..., 1] *= -1   # qx
    q[..., 3] *= -1   # qz
    return q


def mirror_qpos(qpos: np.ndarray, q_off=None) -> np.ndarray:
    """Apply C2 sagittal-plane reflection to a qpos vector (or trajectory).

    Works on shapes (43,) or (T, 43).
    
    q_off: optional (4,) quaternion representing geom offset for the object.
           If None, it tries to load it from the default XML.
    """
    if q_off is None:
        single_xml, _ = get_mj_xml_paths()
        q_off = get_obj_geom_quat(single_xml)

    q = qpos.copy()
    single = q.ndim == 1
    if single:
        q = q[np.newaxis, :]

    T = q.shape[0]

    # ── Base free-joint ──────────────────────────────────────────
    q[:, 1] *= -1                           # base y
    q[:, 3:7] = reflect_quaternion_y(q[:, 3:7])  # base quaternion

    # ── Robot joints (indices 7..35 in qpos → 0..28 in joint block) ──
    joints = q[:, 7:36].copy()              # (T, 29)

    # 1. Swap left ↔ right blocks based on user permutation
    # permutation_Q_js: [[
    #   6, 7, 8, 9, 10, 11,        # right_leg -> left_leg position
    #   0, 1, 2, 3, 4, 5,          # left_leg -> right_leg position
    #   12, 13, 14,                 # waist (yaw, roll, pitch)
    #   22, 23, 24, 25, 26, 27, 28, # right_arm -> left_arm position
    #   15, 16, 17, 18, 19, 20, 21  # left_arm -> right_arm position
    # ]]
    
    # Flattened permutation indices (target_i takes from source_perm[i]):
    perm_indices = [
         6,  7,  8,  9, 10, 11,  # 0-5 (L leg)
         0,  1,  2,  3,  4,  5,  # 6-11 (R leg)
        12, 13, 14,              # 12-14 (Waist)
        22, 23, 24, 25, 26, 27, 28, # 15-21 (L arm)
        15, 16, 17, 18, 19, 20, 21  # 22-28 (R arm)
    ]
    new_joints = joints[:, perm_indices]

    # 2. Negate specified joints based on user feedback
    # reflection_Q_js: [[
    #   1, -1, -1, 1, 1, -1,       # left_leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    #   1, -1, -1, 1, 1, -1,       # right_leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    #   -1, -1, 1,                 # waist: yaw, roll, pitch
    #   1, -1, -1, 1, -1, 1, -1,   # left_arm: sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw
    #   1, -1, -1, 1, -1, 1, -1    # right_arm: sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw
    # ]]
    reflect_sign = np.array([
        1, -1, -1, 1, 1, -1,      # L leg
        1, -1, -1, 1, 1, -1,      # R leg
       -1, -1,  1,                # Waist
        1, -1, -1, 1, -1, 1, -1,  # L arm
        1, -1, -1, 1, -1, 1, -1   # R arm
    ], dtype=np.float32)
    
    new_joints = new_joints * reflect_sign

    q[:, 7:36] = new_joints

    # ── Object free-joint ────────────────────────────────────────
    q[:, 37] *= -1                          # object y
    
    # Correct quaternion mirroring accounting for geom offset:
    #   q_new_body = Refl(q_body * q_off) * q_off_inv
    # This ensures the *Visual* quaternion is mirrored correctly.
    # Note: If the user provided an XML path, we try to parse q_off from it.
    
    # We need XML path for parsing q_off. Ideally we pass it in, but here we can
    # try to use the default one if possible.
    single_xml, _ = get_mj_xml_paths()
    q_off = get_obj_geom_quat(single_xml) # (4,)
    
    # Expand q_off to (T, 4) or (1, 4) depending on q shape
    if q.ndim == 2:
        q_off_T = np.tile(q_off, (T, 1))
    else:
        q_off_T = q_off[None, :]

    q_body_orig = q[:, 39:43]
    
    # 1. Compute CURRENT visual quaternion in world frame
    #    q_vis = q_body * q_off
    q_vis_orig = quat_mul(q_body_orig, q_off_T)
    
    # 2. Reflect visual quaternion
    #    q_vis_mirr = Refl(q_vis)
    q_vis_mirr = reflect_quaternion_y(q_vis_orig)
    
    # 3. Compute NEW body quaternion
    #    q_body_mirr = q_vis_mirr * q_off_inv
    q_off_inv_T = quat_inv(q_off_T)
    q_body_mirr = quat_mul(q_vis_mirr, q_off_inv_T)
    
    q[:, 39:43] = q_body_mirr

    if single:
        q = q[0]
    return q


# ─────────────── build qpos from raw .npz data ───────────────

def npz_to_qpos(data: dict, batch_idx: int | None = None) -> np.ndarray:
    """Convert a raw .npz trajectory dict to (T, 43) qpos array.

    Expects keys: body_pos_w (or root_pos), body_quat_w (or root_rot),
    joint_pos (or dof_pos), object_pos_w (or object_pos), object_quat_w (or object_rot).

    If the arrays have a leading batch dimension (B, T, D) the function
    picks one trajectory:
      * If ``batch_idx`` is given, use that index.
      * Else if a ``cost`` key exists, pick the lowest-cost trajectory.
      * Otherwise default to index 0.
    """
    # Body position & quaternion
    bp = np.asarray(data.get("body_pos_w", data.get("root_pos")))
    bq = np.asarray(data.get("body_quat_w", data.get("root_rot")))
    jp = np.asarray(data.get("joint_pos", data.get("dof_pos")))
    op = np.asarray(data.get("object_pos_w", data.get("object_pos")))
    oq = np.asarray(data.get("object_quat_w", data.get("object_rot")))

    # ── Handle leading batch dimension ───────────────────────────
    # Detect batch: jp should be (T, 29) or (B, T, 29).
    if jp.ndim == 3 and jp.shape[-1] == 29:
        # Batched data — pick one trajectory
        if batch_idx is not None:
            idx = batch_idx
        elif "cost" in data:
            costs = np.asarray(data["cost"])
            idx = int(np.argmin(costs))
            print(f"  Batch dim detected (B={jp.shape[0]}). "
                  f"Picking lowest-cost trajectory idx={idx} "
                  f"(cost={costs[idx]:.4f})")
        else:
            idx = 0
            print(f"  Batch dim detected (B={jp.shape[0]}). "
                  f"Using trajectory idx={idx}")
        bp, bq, jp, op, oq = bp[idx], bq[idx], jp[idx], op[idx], oq[idx]

    # body_pos_w may have a bodies dim: (T, num_bodies, 3) → take pelvis
    if bp.ndim == 3:
        bp = bp[:, 0, :]
    if bq.ndim == 3:
        bq = bq[:, 0, :]

    T = min(len(bp), len(jp), len(op))
    bp, bq, jp, op, oq = bp[:T], bq[:T], jp[:T], op[:T], oq[:T]

    base = np.concatenate([bp, bq], axis=-1)  # (T, 7)
    obj  = np.concatenate([op, oq], axis=-1)  # (T, 7)

    qpos = np.concatenate([base, jp, obj], axis=-1)  # (T, 43)
    return qpos


# ───────────────── diagnostic printout ─────────────────────────

def print_mirror_diagnostics(qpos_orig: np.ndarray, qpos_mirror: np.ndarray):
    """Print key values to sanity-check the mirror transform."""
    t = 0  # check first frame
    o, m = qpos_orig[t], qpos_mirror[t]

    print("\n╔══════════════════════════════════════════════╗")
    print("║       Mirror Symmetry Diagnostics (t=0)      ║")
    print("╠══════════════════════════════════════════════╣")

    print(f"║ Base pos     orig: [{o[0]:+.3f}, {o[1]:+.3f}, {o[2]:+.3f}]")
    print(f"║              mirr: [{m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}]")
    print(f"║   → x,z same, y negated: {'✓' if np.isclose(o[0], m[0]) and np.isclose(o[1], -m[1]) and np.isclose(o[2], m[2]) else '✗'}")

    print(f"║ Base quat    orig: [{o[3]:+.4f}, {o[4]:+.4f}, {o[5]:+.4f}, {o[6]:+.4f}]")
    print(f"║              mirr: [{m[3]:+.4f}, {m[4]:+.4f}, {m[5]:+.4f}, {m[6]:+.4f}]")
    print(f"║   → (w,−x,y,−z):  {'✓' if np.allclose([o[3], -o[4], o[5], -o[6]], m[3:7]) else '✗'}")

    joint_names_left_leg  = ["L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll"]
    joint_names_right_leg = ["R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll"]

    oj = o[7:36]
    mj = m[7:36]
    print("║")
    print("║ Joint swap check (left↔right leg at t=0):")
    for i in range(6):
        ol, or_ = oj[i], oj[6+i]
        ml, mr  = mj[i], mj[6+i]
        # Mirrored left should be original right (with sign flip for roll/yaw)
        negate = i in [1, 2, 5]
        expected_ml = -or_ if negate else or_
        expected_mr = -ol  if negate else ol
        ok_l = np.isclose(ml, expected_ml, atol=1e-6)
        ok_r = np.isclose(mr, expected_mr, atol=1e-6)
        sign_str = " (neg)" if negate else ""
        print(f"║   {joint_names_left_leg[i]:16s}: orig={ol:+.4f}  mirr={ml:+.4f}  expected={expected_ml:+.4f} {'✓' if ok_l else '✗'}{sign_str}")
        print(f"║   {joint_names_right_leg[i]:16s}: orig={or_:+.4f}  mirr={mr:+.4f}  expected={expected_mr:+.4f} {'✓' if ok_r else '✗'}{sign_str}")

    print("║")
    print("║ Object pos   orig:", o[36:39].round(3))
    print("║              mirr:", m[36:39].round(3))
    ok_obj = np.isclose(o[36], m[36]) and np.isclose(o[37], -m[37]) and np.isclose(o[38], m[38])
    print(f"║   → x,z same, y negated: {'✓' if ok_obj else '✗'}")

    print("╚══════════════════════════════════════════════╝\n")


# --------------------------------------------------------------------------- #
# Adaptive XML generation (mirrors batch_goal_sweep.py)
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
# Quaternion helpers
# --------------------------------------------------------------------------- #

def get_obj_geom_quat(xml_path=None):
    """
    Look for a geom named 'obj' in the XML and return its quaternion offset
    (w, x, y, z). If not found or if xml_path is None, defaults to the
    hardcoded value found in unitree_g1/mj_model.xml.
    """
    # Hardcoded fallback from the current XML
    q_default = np.array([0.00991298, 0.849052, -0.523456, 0.0707591], dtype=np.float64)

    if xml_path is None or not os.path.exists(xml_path):
        return q_default

    try:
        tree = ET.parse(xml_path)
        # Search all geoms recursively
        # (ElementTree 1.3+ supports XPath)
        for geom in tree.findall(".//geom"):
            if geom.get("name") == "obj":
                q_str = geom.get("quat")
                if q_str:
                    q_arr = np.fromstring(q_str, sep=' ')
                    if len(q_arr) == 4:
                        return q_arr
    except Exception as e:
        print(f"Warning: Failed to parse geom 'obj' quat from {xml_path}: {e}")

    return q_default


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z). Broadcasts."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of unit quaternion (conjugate): (w, -x, -y, -z)."""
    # q is (..., 4)
    inv = q.copy()
    inv[..., 1:] *= -1
    return inv


# ───────────────────────── main ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify G1 mirror symmetry by visualising original vs "
                    "mirrored trajectory side-by-side in MuJoCo."
    )
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to a .npz trajectory file")
    parser.add_argument("--idx", type=int, default=None,
                        help="Window index in the dataset (uses FlexibleWindowDataset)")
    parser.add_argument("--offset", type=float, default=1.5,
                        help="Spacing between the two robots in the viewer")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Playback timestep (seconds per frame)")
    parser.add_argument("--diag-only", action="store_true",
                        help="Only print diagnostics, don't open viewer")
    args = parser.parse_args()

    # ── Load a trajectory ────────────────────────────────────────
    if args.npz is not None:
        print(f"Loading trajectory from {args.npz}")
        raw = dict(np.load(args.npz, allow_pickle=True))
        qpos = npz_to_qpos(raw)

    elif args.idx is not None:
        # Load from FlexibleWindowDataset
        from config.configure import load_config, get_data_path, get_norm_path
        from datasets.flexible_dataset import FlexibleWindowDataset
        from utils.data.load_dataset import preload_dataset

        model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml")
        data_path = get_data_path(data_cfg)
        norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)

        data_buffer = preload_dataset(data_cfg, data_path)
        dataset = FlexibleWindowDataset(
            data_buffer=data_buffer, config=data_cfg,
            norm_path=norm_path, calculate_stats=False,
            training_cfg=training_cfg,
        )
        file_idx, batch_idx, t_start = dataset.indices[args.idx]
        raw_traj = dataset._get_single_traj(file_idx, batch_idx)
        base   = raw_traj['base']
        joints = raw_traj['joints']
        obj    = raw_traj['obj']
        T = min(len(base), len(joints), len(obj))
        qpos = np.concatenate([base[:T], joints[:T], obj[:T]], axis=-1)

    else:
        # Default: grab a random .npz from test_datasets
        test_dir = "test_datasets/Box_Height_FoLM"
        if not os.path.isdir(test_dir):
            try:
                from config.configure import load_config, get_data_path
                _, data_cfg, _, _ = load_config("config/config.yaml")
                test_dir = get_data_path(data_cfg)
            except Exception:
                pass

        npz_files = sorted([
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir) if f.endswith(".npz")
        ])
        if not npz_files:
            print(f"No .npz files found in {test_dir}. "
                  "Specify --npz or --idx.")
            sys.exit(1)

        chosen = npz_files[0]
        print(f"Using default trajectory: {chosen}")
        raw = dict(np.load(chosen, allow_pickle=True))
        qpos = npz_to_qpos(raw)

    print(f"Trajectory shape: {qpos.shape}  (T={qpos.shape[0]}, nq={qpos.shape[1]})")

    # ── Apply mirror ─────────────────────────────────────────────
    qpos_mirror = mirror_qpos(qpos)

    # ── Diagnostics ──────────────────────────────────────────────
    print_mirror_diagnostics(qpos, qpos_mirror)

    if args.diag_only:
        print("Diagnostics-only mode. Exiting.")
        return

    # ── Visualise using DiffusionOverlayVisualizer ───────────────
    #    Same pattern as generate_and_visualize.py / visualize_dataset.py:
    #    stack [original, mirrored] into (B=2, T, D) and call visualize_overlay.
    #    ref_start_idx=1 colours the mirror copy sky-blue so you can tell them
    #    apart at a glance.
    from utils.visualize.visualize import DiffusionOverlayVisualizer

    single_xml, repeated_xml = get_mj_xml_paths()

    if not os.path.exists(single_xml):
        print(f"ERROR: Single model '{single_xml}' not found. Cannot visualise.")
        sys.exit(1)
    if not os.path.exists(repeated_xml):
        print(f"ERROR: Repeated model '{repeated_xml}' not found. Cannot visualise.")
        sys.exit(1)

    # Shape: (B=2, T, nq_single=43)
    x_trajs = np.stack([qpos, qpos_mirror], axis=0)

    # Use adaptive XML to ensure we have exactly 2 robots (nq=86)
    print("Generating temporary XML for 2 robots...")
    adaptive_xml = None
    try:
        adaptive_xml = build_adaptive_xml(repeated_xml, 2)

        print(f"\nLaunching viewer …")
        print(f"  Robot 0 (default colour)  = ORIGINAL")
        print(f"  Robot 1 (sky-blue)        = MIRRORED")
        print(f"  Spacing = {args.offset}")
        print(f"Controls:  Space=pause  ←/→=step  R=restart\n")

        viz = DiffusionOverlayVisualizer(single_xml)
        viz.visualize_overlay(
            x_trajs,
            repeated_xml_path=adaptive_xml,
            timestep_delay=args.dt,
            loop=True,
            spacing=args.offset,
            ref_start_idx=1,        # mirror copy → sky-blue
        )
    finally:
        if adaptive_xml and os.path.exists(adaptive_xml):
            os.unlink(adaptive_xml)


if __name__ == "__main__":
    main()

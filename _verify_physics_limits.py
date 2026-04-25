"""
Verify that training data does not violate proposed physics thresholds.
Also extracts joint limits from mj_model.xml.

Proposed limits:
- Max robot XY velocity: 2 m/s  (dt=0.01 → max delta_xy per step = 0.02m)
- Max joint velocity: 2 rad/s   (dt=0.01 → max delta_joint per step = 0.02 rad)
- Joint position limits: from XML
- Body Z: from data min/max
"""
import sys, os
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))
from config.configure import load_config, get_data_path
from utils.data.load_dataset import preload_dataset

# ═══════════════════════════════════════════════════════════════════
# 1. Parse joint limits from XML
# ═══════════════════════════════════════════════════════════════════
tree = ET.parse("mj_model.xml")
root = tree.getroot()

# Joint order (must match feature_labels.yml indices 7–35)
JOINT_NAMES = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",      "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",      "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",     "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]
assert len(JOINT_NAMES) == 29

# Build lookup: joint_name → range from XML
# Need to resolve defaults (class-based) and explicit ranges
default_ranges = {}
for default_elem in root.iter("default"):
    cls = default_elem.get("class", "")
    for j in default_elem.findall("joint"):
        r = j.get("range")
        if r:
            lo, hi = map(float, r.split())
            default_ranges[cls] = (lo, hi)

# Find each joint in the worldbody and get its range
joint_limits_lo = np.full(29, -np.pi, dtype=np.float64)
joint_limits_hi = np.full(29,  np.pi, dtype=np.float64)

for i, jname in enumerate(JOINT_NAMES):
    for jelem in root.iter("joint"):
        if jelem.get("name") == jname:
            r = jelem.get("range")
            if r:
                lo, hi = map(float, r.split())
                joint_limits_lo[i] = lo
                joint_limits_hi[i] = hi
            else:
                # Check class-based default
                cls = jelem.get("class", "")
                parent = jelem.getparent() if hasattr(jelem, "getparent") else None
                # Try matching by class hierarchy
                for def_cls, (lo, hi) in default_ranges.items():
                    if def_cls in jname:
                        joint_limits_lo[i] = lo
                        joint_limits_hi[i] = hi
                        break
            break

print("═══════════════════════════════════════════════════════════")
print("JOINT POSITION LIMITS FROM XML")
print("═══════════════════════════════════════════════════════════")
for i, jname in enumerate(JOINT_NAMES):
    print(f"  [{i:2d}] {jname:35s}: [{joint_limits_lo[i]:+.4f}, {joint_limits_hi[i]:+.4f}] rad")

# ═══════════════════════════════════════════════════════════════════
# 2. Load data and profile
# ═══════════════════════════════════════════════════════════════════
model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml", False)
# Override dir_path to match actual workspace location
data_cfg["dir_path"] = os.path.dirname(os.path.abspath(__file__)) + "/"
data_path = get_data_path(data_cfg)
buffer = preload_dataset(data_cfg, data_path)
print(f"\nLoaded {len(buffer)} files")

DT = 0.01  # 100 Hz
STRIDE = data_cfg.get("stride", 2)
DT_STRIDED = DT * STRIDE  # effective dt after stride

# Accumulators
all_body_z = []
all_robot_xy_vel = []     # per-raw-frame XY velocity (m/s)
all_joint_vel = []         # per-raw-frame joint velocity (rad/s)
all_joint_pos_violations = []  # (joint_idx, value, limit_lo, limit_hi)
n_traj = 0

for data in buffer:
    root_pos = data["root_pos"]    # (B, T, 3)
    dof_pos  = data["dof_pos"]     # (B, T, 29)
    
    for b in range(root_pos.shape[0]):
        rp = root_pos[b]   # (T, 3)
        jp = dof_pos[b]    # (T, 29)
        T = rp.shape[0]
        n_traj += 1
        
        # Body Z
        all_body_z.append(rp[:, 2])
        
        # Robot XY velocity (raw frame rate)
        if T > 1:
            dxy = np.diff(rp[:, :2], axis=0)        # (T-1, 2)
            xy_vel = np.linalg.norm(dxy, axis=-1) / DT  # m/s
            all_robot_xy_vel.append(xy_vel)
        
        # Joint velocity (raw frame rate)
        if T > 1:
            djp = np.diff(jp, axis=0)               # (T-1, 29)
            jvel = np.abs(djp) / DT                  # rad/s per joint
            all_joint_vel.append(jvel)
        
        # Joint position limit violations
        lo_viol = jp < joint_limits_lo[None, :]
        hi_viol = jp > joint_limits_hi[None, :]
        if lo_viol.any() or hi_viol.any():
            for j in range(29):
                below = jp[:, j] < joint_limits_lo[j]
                above = jp[:, j] > joint_limits_hi[j]
                if below.any():
                    worst = jp[below, j].min()
                    all_joint_pos_violations.append((j, worst, joint_limits_lo[j], "below"))
                if above.any():
                    worst = jp[above, j].max()
                    all_joint_pos_violations.append((j, worst, joint_limits_hi[j], "above"))

print(f"\nAnalyzed {n_traj} trajectories")

# ═══════════════════════════════════════════════════════════════════
# 3. Report
# ═══════════════════════════════════════════════════════════════════

# --- Body Z ---
body_z_all = np.concatenate(all_body_z)
print("\n═══════════════════════════════════════════════════════════")
print("BODY Z (pelvis height)")
print("═══════════════════════════════════════════════════════════")
print(f"  min={body_z_all.min():.4f}  max={body_z_all.max():.4f}  "
      f"mean={body_z_all.mean():.4f}  std={body_z_all.std():.4f}")
print(f"  p1={np.percentile(body_z_all, 1):.4f}  p5={np.percentile(body_z_all, 5):.4f}  "
      f"p95={np.percentile(body_z_all, 95):.4f}  p99={np.percentile(body_z_all, 99):.4f}")

# --- Robot XY velocity ---
xy_vel_all = np.concatenate(all_robot_xy_vel)
print("\n═══════════════════════════════════════════════════════════")
print(f"ROBOT XY VELOCITY (at {1/DT:.0f} Hz)")
print("═══════════════════════════════════════════════════════════")
print(f"  min={xy_vel_all.min():.4f}  max={xy_vel_all.max():.4f}  "
      f"mean={xy_vel_all.mean():.4f}  std={xy_vel_all.std():.4f}")
print(f"  p95={np.percentile(xy_vel_all, 95):.4f}  p99={np.percentile(xy_vel_all, 99):.4f}  "
      f"p99.9={np.percentile(xy_vel_all, 99.9):.4f}")
n_exceed_2 = (xy_vel_all > 2.0).sum()
print(f"  Frames exceeding 2.0 m/s: {n_exceed_2}/{len(xy_vel_all)} ({100*n_exceed_2/len(xy_vel_all):.4f}%)")
n_exceed_1 = (xy_vel_all > 1.0).sum()
print(f"  Frames exceeding 1.0 m/s: {n_exceed_1}/{len(xy_vel_all)} ({100*n_exceed_1/len(xy_vel_all):.4f}%)")

# --- Robot XY velocity at STRIDE resolution (what the model actually sees) ---
print(f"\n  At stride={STRIDE} (dt_eff={DT_STRIDED}s):")
all_robot_xy_vel_strided = []
for data in buffer:
    root_pos = data["root_pos"]
    for b in range(root_pos.shape[0]):
        rp = root_pos[b, ::STRIDE, :2]  # (T_s, 2)
        if rp.shape[0] > 1:
            dxy = np.diff(rp, axis=0)
            xy_vel_s = np.linalg.norm(dxy, axis=-1) / DT_STRIDED
            all_robot_xy_vel_strided.append(xy_vel_s)
xy_vel_s_all = np.concatenate(all_robot_xy_vel_strided)
print(f"  max={xy_vel_s_all.max():.4f}  p99={np.percentile(xy_vel_s_all, 99):.4f}  "
      f"p99.9={np.percentile(xy_vel_s_all, 99.9):.4f} m/s")

# --- Joint velocity ---
jvel_all = np.concatenate(all_joint_vel)  # (N, 29)
print("\n═══════════════════════════════════════════════════════════")
print(f"JOINT VELOCITY (at {1/DT:.0f} Hz)")
print("═══════════════════════════════════════════════════════════")
for j in range(29):
    jv = jvel_all[:, j]
    mx = jv.max()
    p99 = np.percentile(jv, 99)
    n_exceed = (jv > 2.0).sum()
    flag = " *** EXCEEDS 2 rad/s ***" if mx > 2.0 else ""
    print(f"  [{j:2d}] {JOINT_NAMES[j]:35s}: max={mx:.3f}  p99={p99:.3f}  exceed_2={n_exceed}{flag}")

# At stride resolution
print(f"\n  At stride={STRIDE} (dt_eff={DT_STRIDED}s):")
all_jvel_strided = []
for data in buffer:
    dof_pos = data["dof_pos"]
    for b in range(dof_pos.shape[0]):
        jp = dof_pos[b, ::STRIDE, :]
        if jp.shape[0] > 1:
            djp = np.diff(jp, axis=0)
            jv_s = np.abs(djp) / DT_STRIDED
            all_jvel_strided.append(jv_s)
jvel_s_all = np.concatenate(all_jvel_strided)
print(f"  Global max: {jvel_s_all.max():.4f} rad/s")
for j in range(29):
    jv = jvel_s_all[:, j]
    mx = jv.max()
    if mx > 2.0:
        print(f"    [{j:2d}] {JOINT_NAMES[j]:35s}: max={mx:.3f} *** EXCEEDS 2 rad/s ***")

# --- Joint position violations ---
print("\n═══════════════════════════════════════════════════════════")
print("JOINT POSITION LIMIT VIOLATIONS")
print("═══════════════════════════════════════════════════════════")
if all_joint_pos_violations:
    seen = set()
    for (j, val, lim, direction) in all_joint_pos_violations:
        key = (j, direction)
        if key not in seen:
            seen.add(key)
            print(f"  [{j:2d}] {JOINT_NAMES[j]:35s}: {direction} limit  val={val:.4f}  limit={lim:.4f}  diff={abs(val-lim):.4f}")
else:
    print("  None — all joint positions within XML limits.")

# --- Per-joint position range from data ---
print("\n═══════════════════════════════════════════════════════════")
print("JOINT POSITION RANGE (data vs XML limits)")
print("═══════════════════════════════════════════════════════════")
all_joints = []
for data in buffer:
    dof_pos = data["dof_pos"]
    for b in range(dof_pos.shape[0]):
        all_joints.append(dof_pos[b])
all_joints = np.concatenate(all_joints, axis=0)  # (total_frames, 29)
for j in range(29):
    data_lo = all_joints[:, j].min()
    data_hi = all_joints[:, j].max()
    xml_lo = joint_limits_lo[j]
    xml_hi = joint_limits_hi[j]
    margin_lo = data_lo - xml_lo
    margin_hi = xml_hi - data_hi
    print(f"  [{j:2d}] {JOINT_NAMES[j]:35s}: data=[{data_lo:+.4f}, {data_hi:+.4f}]  xml=[{xml_lo:+.4f}, {xml_hi:+.4f}]  "
          f"margin_lo={margin_lo:+.4f}  margin_hi={margin_hi:+.4f}")

# --- Summary ---
print("\n═══════════════════════════════════════════════════════════")
print("SUMMARY — Proposed thresholds vs data")
print("═══════════════════════════════════════════════════════════")
print(f"  Robot XY vel  ≤ 2.0 m/s @ 100Hz: data max = {xy_vel_all.max():.3f} m/s  ", end="")
print("✓ OK" if xy_vel_all.max() <= 2.0 else "✗ VIOLATED")
print(f"  Robot XY vel  ≤ 2.0 m/s @ stride={STRIDE}: data max = {xy_vel_s_all.max():.3f} m/s  ", end="")
print("✓ OK" if xy_vel_s_all.max() <= 2.0 else "✗ VIOLATED")
print(f"  Joint vel     ≤ 2.0 rad/s @ 100Hz: data max = {jvel_all.max():.3f} rad/s  ", end="")
print("✓ OK" if jvel_all.max() <= 2.0 else "✗ VIOLATED")
print(f"  Joint vel     ≤ 2.0 rad/s @ stride={STRIDE}: data max = {jvel_s_all.max():.3f} rad/s  ", end="")
print("✓ OK" if jvel_s_all.max() <= 2.0 else "✗ VIOLATED")
print(f"  Joint pos limits: ", end="")
print("✓ All within XML" if not all_joint_pos_violations else f"✗ {len(all_joint_pos_violations)} violations")
print(f"  Body Z range:  [{body_z_all.min():.4f}, {body_z_all.max():.4f}]")

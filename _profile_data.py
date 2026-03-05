"""Profile real training data to understand obj_z and obj_delta_xy patterns."""
import sys, os, yaml
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from config.configure import load_config, get_data_path
from utils.data.load_dataset import preload_dataset
from utils.math.sbto_utils import compute_sbto_components

# ---------- load ----------
model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml", False)
data_path = get_data_path(data_cfg)
buffer = preload_dataset(data_cfg, data_path)
print(f"Loaded {len(buffer)} trajectory files")

# ---------- feature-level analysis (SBTO space) ----------
# We need to see what obj_z and obj_delta_xy look like in SBTO feature space
# because those are what the model predicts / waypoints constrain.

feature_labels = yaml.safe_load(open("config/feature_labels.yml"))
if isinstance(feature_labels, dict):
    feat_names = list(feature_labels.keys())
elif isinstance(feature_labels, list):
    feat_names = feature_labels
else:
    feat_names = str(feature_labels)
print("Feature labels:", feat_names)

all_obj_z_feat = []      # obj_z feature over trajectory (SBTO)
all_obj_dxy_feat = []    # obj_delta_xy feature over trajectory (SBTO)
all_raw_obj_z = []       # raw world obj z
all_raw_obj_xy_disp = [] # raw world obj xy displacement
all_traj_len = []
n_skipped = 0

for i, data in enumerate(buffer):
    # Keys: root_pos (B,T,3), root_rot (B,T,4), object_pos (B,T,3), object_rot (B,T,4), dof_pos (B,T,29)
    obj_pos = data.get("object_pos")
    obj_rot = data.get("object_rot")
    root_pos = data.get("root_pos")
    root_rot = data.get("root_rot")
    joints = data.get("dof_pos")
    if obj_pos is None or root_pos is None or joints is None:
        n_skipped += 1
        continue

    # ensure 3D (B, T, D)
    if obj_pos.ndim == 2:
        obj_pos = obj_pos[np.newaxis]
        obj_rot = obj_rot[np.newaxis]
        root_pos = root_pos[np.newaxis]
        root_rot = root_rot[np.newaxis]
        joints = joints[np.newaxis]

    for b in range(obj_pos.shape[0]):
        op = obj_pos[b]   # (T, 3)
        orr = obj_rot[b]  # (T, 4)
        rp = root_pos[b]  # (T, 3)
        rr = root_rot[b]  # (T, 4)
        j = joints[b]     # (T, 29)
        T = op.shape[0]
        all_traj_len.append(T)

        # --- RAW WORLD SPACE ---
        z = op[:, 2].copy()
        z_rest = z[0]
        all_raw_obj_z.append(z - z_rest)  # delta from rest

        xy_disp = np.linalg.norm(op[:, :2] - op[0:1, :2], axis=-1)
        all_raw_obj_xy_disp.append(xy_disp)

        # --- SBTO FEATURE SPACE ---
        # obj_z is just absolute world Z of object
        obj_z_f = op[:, 2]  # (T,)
        
        # obj_delta_xy: XY displacement from t=0 in pelvis-yaw-aligned frame
        # Get reference yaw at t=0
        rr0 = rr[0]  # (4,)  quat wxyz
        # Extract yaw from quaternion
        # quat: (w, x, y, z)
        w, x, y, z = rr0[0], rr0[1], rr0[2], rr0[3]
        yaw0 = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
        cos_y, sin_y = np.cos(-yaw0), np.sin(-yaw0)
        R_inv = np.array([[cos_y, -sin_y], [sin_y, cos_y]])  # 2x2

        # Object XY displacement from initial position in world frame
        obj_dxy_world = op[:, :2] - op[0:1, :2]  # (T, 2)
        # Rotate into pelvis-yaw frame
        obj_dxy_local = (R_inv @ obj_dxy_world.T).T  # (T, 2)

        all_obj_z_feat.append(obj_z_f)
        all_obj_dxy_feat.append(obj_dxy_local)

print(f"Analyzed {len(all_raw_obj_z)} trajectories, skipped {n_skipped}")
print(f"Trajectory lengths: min={min(all_traj_len)}, max={max(all_traj_len)}, median={np.median(all_traj_len):.0f}")

# ---------- SBTO obj_z profile ----------
n_pts = 100

def make_profiles(data_list, n_pts=100):
    profiles = np.zeros((len(data_list), n_pts))
    for i, d in enumerate(data_list):
        T = len(d) if d.ndim == 1 else d.shape[0]
        if T < 2:
            continue
        orig_t = np.linspace(0, 1, T)
        interp_t = np.linspace(0, 1, n_pts)
        if d.ndim == 1:
            profiles[i] = np.interp(interp_t, orig_t, d)
        else:
            for c in range(d.shape[1]):
                profiles[i] = np.interp(interp_t, orig_t, np.linalg.norm(d, axis=-1))
            break  # only norm
    return profiles

# obj_z in SBTO feature space (this is ABSOLUTE world z)
oz_profiles = make_profiles(all_obj_z_feat, n_pts)
oz_rest = np.array([z[0] for z in all_obj_z_feat])
oz_delta_profiles = oz_profiles - oz_profiles[:, 0:1]  # delta from initial

print("\n=== OBJ_Z FEATURE (absolute world Z) ===")
print(f"  Rest z: min={oz_rest.min():.4f}  max={oz_rest.max():.4f}  mean={oz_rest.mean():.4f}  std={oz_rest.std():.4f}")
print()

# Z delta from rest
zd_mean = oz_delta_profiles.mean(axis=0)
zd_med = np.median(oz_delta_profiles, axis=0)
zd_p25 = np.percentile(oz_delta_profiles, 25, axis=0)
zd_p75 = np.percentile(oz_delta_profiles, 75, axis=0)
zd_max = oz_delta_profiles.max(axis=0)

print("=== OBJ_Z DELTA (feat space, above t=0 value) ===")
print("progress | mean   | median | p25    | p75    | max")
for j in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]:
    print(f"  {j:3d}%    | {zd_mean[j]:.4f} | {zd_med[j]:.4f} | {zd_p25[j]:.4f} | {zd_p75[j]:.4f} | {zd_max[j]:.4f}")

# Peak analysis
peak_z_delta = np.array([zd.max() for zd in (oz_profiles - oz_profiles[:, 0:1])])
peak_progress = np.array([np.argmax(oz_profiles[i] - oz_profiles[i, 0]) / (n_pts - 1) for i in range(len(oz_profiles))])
print(f"\nPeak z delta: min={peak_z_delta.min():.4f}  max={peak_z_delta.max():.4f}  median={np.median(peak_z_delta):.4f}  mean={peak_z_delta.mean():.4f}")
print(f"Peak z progress: min={peak_progress.min():.3f}  max={peak_progress.max():.3f}  median={np.median(peak_progress):.3f}  mean={peak_progress.mean():.3f}")

# Z return timing
z_return = []
for i in range(len(oz_delta_profiles)):
    zd = oz_delta_profiles[i]
    pk = zd.max()
    if pk < 0.02:
        continue
    after_peak = np.argmax(zd)
    for t in range(after_peak, n_pts):
        if zd[t] < pk * 0.1:
            z_return.append(t / (n_pts - 1))
            break
    else:
        z_return.append(1.0)
if z_return:
    z_return = np.array(z_return)
    print(f"Z return to <10% peak: min={z_return.min():.3f}  max={z_return.max():.3f}  median={np.median(z_return):.3f}  mean={z_return.mean():.3f}")

# ---------- obj_delta_xy (SBTO) displacement profile ----------
print("\n=== OBJ_DELTA_XY (feat space) displacement ===")
# obj_delta_xy is cumulative displacement in pelvis-yaw frame from t=0
dxy_norms = [np.linalg.norm(d, axis=-1) for d in all_obj_dxy_feat]
dxy_profiles = make_profiles(dxy_norms, n_pts)

# But actually let's recompute properly
dxy_profiles2 = np.zeros((len(all_obj_dxy_feat), n_pts))
for i, d in enumerate(all_obj_dxy_feat):
    T = d.shape[0]
    if T < 2: continue
    norms = np.linalg.norm(d, axis=-1)
    orig_t = np.linspace(0, 1, T)
    interp_t = np.linspace(0, 1, n_pts)
    dxy_profiles2[i] = np.interp(interp_t, orig_t, norms)

dxy_final = dxy_profiles2[:, -1]
print(f"Final xy disp (feat): min={dxy_final.min():.4f}  max={dxy_final.max():.4f}  median={np.median(dxy_final):.4f}  mean={dxy_final.mean():.4f}")

# Normalize each traj to [0,1] of its final to see timing
dxy_normalized = np.zeros_like(dxy_profiles2)
for i in range(len(dxy_profiles2)):
    final = dxy_profiles2[i, -1]
    if final > 0.01:
        dxy_normalized[i] = dxy_profiles2[i] / final
    else:
        dxy_normalized[i] = 0

dxy_n_mean = dxy_normalized.mean(axis=0)
dxy_n_med = np.median(dxy_normalized, axis=0)
print("\n=== NORMALIZED XY DISP TIMING ===")
print("progress | mean   | median")
for j in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]:
    print(f"  {j:3d}%    | {dxy_n_mean[j]:.4f} | {dxy_n_med[j]:.4f}")

# XY start/50%/90% milestones
xy_start_list, xy_50_list, xy_90_list = [], [], []
for i in range(len(dxy_profiles2)):
    final = dxy_profiles2[i, -1]
    if final < 0.01: continue
    for t in range(n_pts):
        if dxy_profiles2[i, t] > 0.05 * final:
            xy_start_list.append(t / (n_pts - 1))
            break
    for t in range(n_pts):
        if dxy_profiles2[i, t] > 0.5 * final:
            xy_50_list.append(t / (n_pts - 1))
            break
    for t in range(n_pts):
        if dxy_profiles2[i, t] > 0.9 * final:
            xy_90_list.append(t / (n_pts - 1))
            break

if xy_start_list:
    xy_start_list = np.array(xy_start_list)
    print(f"\nXY start (>5%): min={xy_start_list.min():.3f} max={xy_start_list.max():.3f} median={np.median(xy_start_list):.3f} mean={xy_start_list.mean():.3f}")
if xy_50_list:
    xy_50_list = np.array(xy_50_list)
    print(f"XY 50%:  min={xy_50_list.min():.3f} max={xy_50_list.max():.3f} median={np.median(xy_50_list):.3f} mean={xy_50_list.mean():.3f}")
if xy_90_list:
    xy_90_list = np.array(xy_90_list)
    print(f"XY 90%:  min={xy_90_list.min():.3f} max={xy_90_list.max():.3f} median={np.median(xy_90_list):.3f} mean={xy_90_list.mean():.3f}")

# ---------- PHASE ANALYSIS (Z vs XY timing) ----------
print("\n=== PHASE ANALYSIS ===")
# For each trajectory: when does Z peak vs when does XY reach 50%
for idx in range(min(15, len(all_obj_z_feat))):
    zd = all_obj_z_feat[idx] - all_obj_z_feat[idx][0]
    dxy = np.linalg.norm(all_obj_dxy_feat[idx], axis=-1)
    T = len(zd)
    pk_idx = np.argmax(zd)
    pk_prog = pk_idx / max(T-1, 1)
    pk_z = zd[pk_idx]
    final_xy = dxy[-1]
    xy_at_peak = dxy[pk_idx] if pk_idx < len(dxy) else 0
    pct = xy_at_peak / max(final_xy, 1e-6) * 100
    print(f"  [{idx:2d}] T={T:3d}: z_peak={pk_z:.3f}m @ prog={pk_prog:.2f}, xy_at_zpeak={xy_at_peak:.3f}/{final_xy:.3f}m ({pct:.0f}%)")

# ---------- NOW COMPARE WITH INFERENCE WAYPOINTS ----------
print("\n\n" + "="*60)
print("=== COMPARISON WITH CURRENT INFERENCE WAYPOINTS ===")
print("="*60)
print()
print("Current inference parameters:")
print("  DEFAULT_LIFT_HEIGHT = 0.5")
print("  lift_start = 0.0, lift_end = 0.20 (cosine ramp 0→20%)")
print("  walk_start_z = 0.80 (XY starts after 80% of lift height)")
print("  t_accel = 0.35, t_decel = 0.35 (trapezoidal XY)")
print("  arrival_ratio = 0.85 (XY arrives at 85%)")
print("  no_lower_dist = 0.5 (lowering starts at <0.5m remaining)")
print()
print("Data shows:")
print(f"  Median peak z_delta: {np.median(peak_z_delta):.4f}  (vs DEFAULT_LIFT_HEIGHT=0.5)")
print(f"  Median peak z progress: {np.median(peak_progress):.3f}  (vs lift_end=0.20)")
if len(z_return) > 0:
    print(f"  Median z return timing: {np.median(z_return):.3f}")
if len(xy_start_list) > 0:
    print(f"  Median XY start (>5%): {np.median(xy_start_list):.3f}  (vs walk_start gated by z)")
if len(xy_50_list) > 0:
    print(f"  Median XY 50%: {np.median(xy_50_list):.3f}")
if len(xy_90_list) > 0:
    print(f"  Median XY 90%: {np.median(xy_90_list):.3f}  (vs arrival_ratio=0.85)")

# ---------- NOW: what does the model see in its window? ----------
# The model processes windows of 20 timesteps at stride 2
# So each window covers 20*2=40 raw timesteps
# Let's show what a window's worth of features looks like
print("\n=== WINDOWED VIEW (20 timesteps, stride 2) ===")
stride = data_cfg.get("stride", 2)
window = data_cfg.get("num_timesteps", 20)
raw_window = window * stride
print(f"Window: {window} steps at stride {stride} = {raw_window} raw frames")

# Show obj_z feature values at each window step for a few trajectories
for idx in range(min(5, len(all_obj_z_feat))):
    z_feat = all_obj_z_feat[idx]
    dxy_feat = all_obj_dxy_feat[idx]
    T = len(z_feat)
    if T < raw_window: continue
    
    print(f"\n  Trajectory [{idx}] (T={T}):")
    # Take a window starting at pick-up (where z starts rising)
    z_delta = z_feat - z_feat[0]
    rise_idx = 0
    for t in range(T):
        if z_delta[t] > 0.02:
            rise_idx = max(0, t - 2*stride)
            break
    
    # Show the window from rise_idx
    end_idx = min(rise_idx + raw_window, T)
    steps = list(range(rise_idx, end_idx, stride))
    print(f"    Window from raw frame {rise_idx} (progress {rise_idx/max(T-1,1):.2f}):")
    print(f"    step | obj_z  | z_delta | dxy_norm")
    for si, raw_i in enumerate(steps[:window]):
        oz = z_feat[raw_i]
        zd = z_delta[raw_i]
        dxy = np.linalg.norm(dxy_feat[raw_i])
        print(f"    {si:4d} | {oz:.4f} | {zd:.4f}  | {dxy:.4f}")

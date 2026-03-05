"""
Plot actual training data profiles vs inference waypoints for the first trajectory.
Shows obj_z (absolute) and obj_delta_xy (displacement norm) over trajectory progress.
"""
import sys, os, yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from config.configure import load_config, get_data_path
from utils.data.load_dataset import preload_dataset

# ── 1. Load data ──────────────────────────────────────────────────
model_cfg, data_cfg, training_cfg, noise_cfg = load_config("config/config.yaml", False)
data_path = get_data_path(data_cfg)
buffer = preload_dataset(data_cfg, data_path)
print(f"Loaded {len(buffer)} files")

# ── 2. Extract first file, first trajectory ───────────────────────
data = buffer[0]
obj_pos = data["object_pos"]  # (B, T, 3)
root_pos = data["root_pos"]    # (B, T, 3)
root_rot = data["root_rot"]    # (B, T, 4) wxyz

# Take trajectory b=0
b = 0
op = obj_pos[b]    # (T, 3)
rp = root_pos[b]   # (T, 3)
rr = root_rot[b]   # (T, 4) wxyz
T_raw = op.shape[0]

# ── 3. Compute ground-truth features ─────────────────────────────
# obj_z: absolute world Z
gt_obj_z = op[:, 2]

# obj_delta_xy: XY displacement from t=0 in pelvis-yaw-aligned frame
w, x, y, z_q = rr[0, 0], rr[0, 1], rr[0, 2], rr[0, 3]
yaw0 = np.arctan2(2.0*(w*z_q + x*y), 1.0 - 2.0*(y*y + z_q*z_q))
cos_y, sin_y = np.cos(-yaw0), np.sin(-yaw0)
R_inv = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
obj_dxy_world = op[:, :2] - op[0:1, :2]
obj_dxy_local = (R_inv @ obj_dxy_world.T).T
gt_dxy_norm = np.linalg.norm(obj_dxy_local, axis=-1)

# Progress axis
gt_progress = np.linspace(0, 1, T_raw)

# ── 4. Compute the inference waypoint profile ────────────────────
# Simulate what the waypoint builder would produce at each progress fraction
import math

DEFAULT_LIFT_HEIGHT = 0.62

def compute_z_profile(progress, lift_height=DEFAULT_LIFT_HEIGHT,
                      lift_start=0.10, lift_end=0.30,
                      lower_start=0.60, lower_end=0.80):
    p = max(0.0, min(1.0, progress))
    if p < lift_start:
        return 0.0
    elif p < lift_end:
        t = (p - lift_start) / (lift_end - lift_start)
        return lift_height * 0.5 * (1.0 - math.cos(math.pi * t))
    elif p < lower_start:
        return lift_height
    elif p < lower_end:
        t = (p - lower_start) / (lower_end - lower_start)
        return lift_height * 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        return 0.0

# Also compute the OLD profile for comparison
def compute_z_profile_old(progress, lift_height=0.5,
                          lift_start=0.0, lift_end=0.20):
    p = max(0.0, min(1.0, progress))
    if p < lift_start:
        return 0.0
    elif p < lift_end:
        t = (p - lift_start) / (lift_end - lift_start)
        return lift_height * 0.5 * (1.0 - math.cos(math.pi * t))
    else:
        return lift_height

# Waypoint Z profiles (offset above rest)
rest_z = gt_obj_z[0]
n_pts = 200
wp_progress = np.linspace(0, 1, n_pts)
wp_z_new = np.array([rest_z + compute_z_profile(p) for p in wp_progress])
wp_z_old = np.array([rest_z + compute_z_profile_old(p) for p in wp_progress])

# XY trapezoidal displacement profile
# New: arrival_ratio=0.70, t_accel=0.35, t_decel=0.25
# Old: arrival_ratio=0.85, t_accel=0.35, t_decel=0.35
def trapezoidal_displacement(tau, t_accel, t_decel):
    """Fractional displacement [0,1] at normalized time tau [0,1]."""
    tau = max(0.0, min(1.0, tau))
    if t_accel + t_decel > 1.0:
        t_accel, t_decel = 0.5, 0.5
    v_max = 1.0 / (1.0 - 0.5 * t_accel - 0.5 * t_decel)
    p_ta = 0.5 * v_max * t_accel
    p_td_start = p_ta + v_max * (1.0 - t_accel - t_decel)
    if tau <= t_accel:
        return 0.5 * (v_max / t_accel) * (tau ** 2) if t_accel > 0 else 0.0
    elif tau <= 1.0 - t_decel:
        return p_ta + v_max * (tau - t_accel)
    else:
        tau_prime = tau - (1.0 - t_decel)
        if t_decel > 0:
            return p_td_start + v_max * tau_prime - 0.5 * (v_max / t_decel) * (tau_prime ** 2)
        else:
            return p_td_start + v_max * tau_prime

final_xy_disp = gt_dxy_norm[-1]  # use actual final displacement as scale

# New params
arrival_new = 0.70
wp_xy_new = np.array([
    final_xy_disp * min(1.0, trapezoidal_displacement(p / arrival_new, 0.35, 0.25))
    for p in wp_progress
])

# Old params
arrival_old = 0.85
wp_xy_old = np.array([
    final_xy_disp * min(1.0, trapezoidal_displacement(p / arrival_old, 0.35, 0.35))
    for p in wp_progress
])

# ── 5. Also compute windowed view ────────────────────────────────
# The model sees 20 steps at stride 2 = 40 raw frames per window
stride = data_cfg.get("stride", 2)
window = data_cfg.get("num_timesteps", 20)
raw_window = window * stride

# Show the data at model's stride resolution
strided_indices = np.arange(0, T_raw, stride)
gt_obj_z_strided = gt_obj_z[strided_indices]
gt_dxy_strided = gt_dxy_norm[strided_indices]
gt_progress_strided = strided_indices / max(T_raw - 1, 1)

# ── 6. Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- Panel 1: obj_z (absolute) ---
ax1 = axes[0]
ax1.plot(gt_progress, gt_obj_z, 'b-', alpha=0.4, linewidth=1, label='GT obj_z (raw)')
ax1.plot(gt_progress_strided, gt_obj_z_strided, 'b.', markersize=3, alpha=0.6, label=f'GT obj_z (stride={stride})')
ax1.plot(wp_progress, wp_z_new, 'r-', linewidth=2.5, label='NEW waypoint z profile')
ax1.plot(wp_progress, wp_z_old, 'orange', linewidth=2, linestyle='--', label='OLD waypoint z profile')
ax1.axhline(y=rest_z, color='gray', linestyle=':', alpha=0.5, label=f'rest z = {rest_z:.4f}')
ax1.axhline(y=rest_z + DEFAULT_LIFT_HEIGHT, color='red', linestyle=':', alpha=0.3, label=f'lift peak = {rest_z + DEFAULT_LIFT_HEIGHT:.3f}')
ax1.axhline(y=rest_z + 0.5, color='orange', linestyle=':', alpha=0.3, label=f'old peak = {rest_z + 0.5:.3f}')

# Mark key progress points
for p_val, label, color in [
    (0.10, 'lift_start=10%', 'green'),
    (0.40, 'lift_end=40%', 'green'),
    (0.55, 'lower_start=55%', 'purple'),
    (0.75, 'lower_end=75%', 'purple'),
]:
    ax1.axvline(x=p_val, color=color, linestyle='--', alpha=0.3, linewidth=0.8)
    ax1.text(p_val, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else rest_z - 0.02,
             label, fontsize=7, color=color, ha='center', va='top', rotation=90)

ax1.set_ylabel('obj_z (absolute world Z) [m]')
ax1.set_title(f'Trajectory [0] — obj_z: Data vs Waypoint Profile (T={T_raw})')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# --- Panel 2: obj_delta_xy displacement norm ---
ax2 = axes[1]
ax2.plot(gt_progress, gt_dxy_norm, 'b-', alpha=0.4, linewidth=1, label='GT ||obj_delta_xy|| (raw)')
ax2.plot(gt_progress_strided, gt_dxy_strided, 'b.', markersize=3, alpha=0.6, label=f'GT (stride={stride})')
ax2.plot(wp_progress, wp_xy_new, 'r-', linewidth=2.5, label='NEW waypoint XY profile')
ax2.plot(wp_progress, wp_xy_old, 'orange', linewidth=2, linestyle='--', label='OLD waypoint XY profile')

for p_val, label, color in [
    (0.70, 'arrival=70%', 'red'),
    (0.85, 'old arrival=85%', 'orange'),
]:
    ax2.axvline(x=p_val, color=color, linestyle='--', alpha=0.4, linewidth=1)
    ax2.text(p_val + 0.01, final_xy_disp * 0.5, label, fontsize=8, color=color, rotation=90, va='center')

ax2.set_xlabel('Trajectory Progress [0 → 1]')
ax2.set_ylabel('||obj_delta_xy|| displacement [m]')
ax2.set_title('obj_delta_xy: Data vs Waypoint Profile')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'waypoint_vs_data.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {save_path}")
plt.show()

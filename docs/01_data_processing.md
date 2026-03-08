# Data Processing Pipeline

## Overview

The data processing pipeline transforms raw absolute-coordinate motion capture trajectories of a G1 humanoid robot (29 DOFs) performing pick-and-place / locomotion tasks into a **robot-relative coordinate system** called **SBTO (Scene-Body-Transform-Object)**. The pipeline handles windowing, on-the-fly coordinate transformation, normalization, and condition extraction.

---

## 1. Raw Data Format

Raw trajectories are stored as `.npz` files with the following fields (per the "standardized" schema):

| Field             | Shape         | Description                                           |
|-------------------|---------------|-------------------------------------------------------|
| `base_xyz_quat`   | `(T, 7)`     | Robot base position (x, y, z) + quaternion (w, x, y, z) in world frame |
| `joint_pos`       | `(T, 29)`    | Joint angles for the G1 humanoid                      |
| `obj_xyz_quat`    | `(T, 7)`     | Object position + quaternion in world frame            |
| `task_type`       | scalar/str    | Task identifier (e.g., "walk_carry_place")             |
| `task_params`     | `(3,)` or dict | Goal object position in world frame                  |

Three input schemas are supported and internally converted:

1. **Standardized** (`base_xyz_quat`, `joint_pos`, `obj_xyz_quat`) — preferred
2. **RL rollout** (`body_pos_w`, `body_quat_w`, `joint_pos`, `obj_pos_w`, `obj_quat_w`) — legacy
3. **SBTO pre-processed** — already in SBTO format, bypasses conversion

Schema detection and conversion happens in `FlexibleWindowDataset._convert_schema()`.

---

## 2. Dataset Class: `FlexibleWindowDataset`

**Location:** `datasets/flexible_dataset.py`

This is the core dataset class. It loads raw trajectories, windows them into fixed-length segments, transforms coordinates on-the-fly, normalizes features, and returns training samples.

### 2.1 Initialization & Loading (`_preload_from_buffer`)

1. **Load trajectories** from the data directory (glob `*.npz`)
2. **Schema conversion** via `_convert_schema()` → produces `base_xyz_quat`, `joint_pos`, `obj_xyz_quat`
3. **Optional downsampling** via `start_timestep` and `downsample_factor` (config: `stride: 2`)
4. **Goal computation**: `goal_obj_world` is extracted from `task_params` — this is the world-frame position of the goal object location
5. Store each trajectory as a dict with all fields, plus `length` (number of timesteps)

### 2.2 Windowing & Indexing (`_index_dataset`)

Trajectories are windowed into segments of length `window_size = num_timesteps + num_history`:

- **`num_timesteps`**: 20 (future prediction horizon)
- **`num_history`**: 1 (state history for conditioning, derived from `state_history`)
- **`stride`**: 2 (window stride for overlapping windows)

For each trajectory of length `L`:
- Calculate `num_batches_per_trajectory = max(1, (L - window_size) // stride + 1)`
- If `L < window_size`, the trajectory is still included once with zero-padding applied during `__getitem__`
- Index entries are tuples of `(file_idx, batch_idx, t_start)`

**Padding**: If a window extends past the trajectory end, padding is applied by repeating the last valid frame (constant padding).

### 2.3 SBTO Coordinate Transform (`_compute_transform`)

This is the core transformation called in `__getitem__`. It converts absolute world-frame coordinates to robot-relative coordinates.

**Implementation:** `utils/math/sbto_utils.py → compute_sbto_components()`

#### Input
- `base_w`: `(B, T, 7)` — robot base xyz + quaternion in world frame
- `joints`: `(B, T, 29)` — joint angles
- `obj_w`: `(B, T, 7)` — object xyz + quaternion in world frame

#### Transformation Steps

1. **Extract yaw** from base quaternion → `yaw_t` for each timestep
2. **Compute anchors** from the first frame:
   - `anchor_xy = base_w[:, 0, :2]` — XY position of robot at t=0
   - `anchor_yaw = yaw_t[:, 0]` — heading yaw at t=0
   - `anchor_obj_xy = obj_w[:, 0, :2]` — object XY at t=0 (for object deltas)
3. **Delta XY**: Frame-to-frame displacement in the robot's heading frame
   ```
   delta_xy[t] = R(-yaw[t]) @ (base_xy[t+1] - base_xy[t])
   ```
   Last frame uses zero delta. Shape: `(B, T, 2)`
4. **Delta Yaw**: Frame-to-frame yaw change, wrapped to `[-π, π]`. Shape: `(B, T, 1)`
5. **Body Z**: Raw z-coordinate of robot base. Shape: `(B, T, 1)`
6. **Body Rot6D**: Body orientation as 6D rotation representation (first two columns of rotation matrix), with the yaw component factored out:
   ```
   R_body = R(-yaw[t]) @ R_full(quat[t])
   ```
   Only the pitch/roll residual is kept. Shape: `(B, T, 6)`
7. **Object Relative Position**: Object XYZ in robot's local frame
   ```
   obj_rel_pos[t] = R(-yaw[t]) @ (obj_xyz[t] - base_xyz[t])
   ```
   Shape: `(B, T, 3)`
8. **Object Relative Rot6D**: Object orientation relative to robot heading, as 6D representation. Shape: `(B, T, 6)`
9. **Object Delta XY**: Object XY displacement relative to its position at t=0, in world frame (anchor-relative)
   ```
   obj_delta_xy[t] = obj_xy[t] - anchor_obj_xy
   ```
   Shape: `(B, T, 2)`
10. **Object Z**: Raw z-coordinate of the object. Shape: `(B, T, 1)`
11. **Joints**: Raw joint angles, unchanged. Shape: `(B, T, 29)`

#### Anchors Returned
The transform also returns anchor values needed for inverse transformation (reconstruction):
- `anchor_xy`, `anchor_yaw`, `anchor_z`, `anchor_obj_xy`
- `anchor_quat` (full quaternion at t=0)

### 2.4 Feature Assembly

After computing SBTO components, features are concatenated following the **feature order** defined in `config.yaml`:

```yaml
feature_order:
  - delta_xy        # dims 0-1   (2)
  - delta_yaw       # dim  2     (1)
  - obj_delta_xy    # dims 3-4   (2)
  - obj_z           # dim  5     (1) — Note: appears at index 4 in code after obj_delta_xy
  - joints          # dims 6-34  (29)
  - body_z          # dim  35    (1) — Note: actual position shifts based on order
  - body_rot6d      # dims 36-41 (6)
  - obj_rel_pos     # dims 42-44 (3)
  - obj_rel_rot6d   # dims 45-50 (6)
```

**Total feature dimension: 51** (`num_features` in config)

The feature layout is built dynamically from `feature_order` by `sbto_utils.build_feature_layout()`, which maps feature names to their dimensions:

| Feature         | Dim |
|-----------------|-----|
| `delta_xy`      | 2   |
| `delta_yaw`     | 1   |
| `obj_delta_xy`  | 2   |
| `obj_z`         | 1   |
| `joints`        | 29  |
| `body_z`        | 1   |
| `body_rot6d`    | 6   |
| `obj_rel_pos`   | 3   |
| `obj_rel_rot6d` | 6   |

### 2.5 Task Parameters (Goal Conditioning)

Task parameters are computed by `compute_task_params()`:

1. Compute goal vector in robot-local frame:
   ```
   goal_local = R(-anchor_yaw) @ (goal_world_xy - anchor_xy)
   ```
2. Compute distance: `d = ||goal_local||`
3. Compute normalized direction: `dir = goal_local / (d + ε)`
4. Clip distance to `max_goal_dist` (default: 3.0)
5. Optionally normalize distance to `[0, 1]` by dividing by `max_goal_dist` (when `normalize_goal_vec: True`)
6. Return: `[dir_x, dir_y, clipped_distance]` — shape `(3,)`

### 2.6 Current State (State Conditioning)

The **current state** (observation) is extracted from the last `num_history` (1) timesteps. It consists of the last `num_observations` (45) dimensions of the feature vector:

```
current_state = features[history_frame, -45:]
```

This corresponds to:
- `joints` (29 dims)
- `body_z` (1 dim)
- `body_rot6d` (6 dims)
- `obj_rel_pos` (3 dims)
- `obj_rel_rot6d` (6 dims)

**Total: 45 dimensions** — note this **excludes** the delta/displacement features (`delta_xy`, `delta_yaw`, `obj_delta_xy`, `obj_z`) which are at the start of the feature vector.

### 2.7 History Frame Noise

To improve robustness, Gaussian noise is optionally added to the history frame(s) of the future trajectory:

```python
if self.add_noise_to_history and self.num_history > 0:
    noise = torch.randn_like(future[:self.num_history]) * self.history_noise_std
    future[:self.num_history] += noise
```

Config: `state_noise_std: 0.01`

---

## 3. Normalization

### 3.1 Stats Computation (`_calculate_stats`)

Normalization statistics are computed over the entire dataset using batched iteration:

- **Min-Max normalization** (default, `normalization.type: min_max`):
  - Computes per-feature `min` and `max` across all timesteps and trajectories
  - Normalizes to `[-1, 1]`: `x_norm = 2 * (x - min) / (max - min + ε) - 1`
  
- **Mean-Std normalization** (alternative):
  - Computes per-feature `mean` and `std`
  - Normalizes: `x_norm = (x - mean) / (std + ε)`

Stats are saved as `norm_stats.npz` at the normalization path and loaded on subsequent runs.

### 3.2 Per-Feature Normalization

Normalization operates on each of the 51 features independently. The `_normalize()` and `_denormalize()` methods handle individual feature tensors (shape `(T, D_feature)`), while `denormalize_global()` handles the full concatenated tensor (shape `(T, 51)`).

---

## 4. `__getitem__` Return Values

Each sample from the dataset is a tuple of 4 tensors:

| Name            | Shape              | Description                                                |
|-----------------|--------------------|------------------------------------------------------------|
| `future_states` | `(T, 51)`          | Normalized SBTO features for the full window (history + prediction). `T = num_history + num_timesteps = 1 + 20 = 21` |
| `current_state` | `(45,)`            | Normalized current state observation (from last history frame) |
| `task_params`   | `(3,)`             | Goal direction + distance in robot-local frame              |
| `anchor`        | `(anchor_dim,)`    | Anchor values for inverse SBTO reconstruction               |

---

## 5. Inverse Transform (Reconstruction)

For inference, SBTO features must be converted back to world-frame trajectories. This is done by `reconstruct_sbto_trajectory()` in `sbto_utils.py`:

1. **Denormalize** all features
2. **Reconstruct base XY** by integrating `delta_xy` displacements through the heading frame:
   ```
   xy[t+1] = xy[t] + R(yaw[t]) @ delta_xy[t]
   ```
3. **Reconstruct yaw** by cumulative summation of `delta_yaw`
4. **Reconstruct base quaternion** from yaw + body_rot6d (pitch/roll residual)
5. **Reconstruct object position** from `obj_rel_pos` + base position + heading rotation
6. **Reconstruct object quaternion** from `obj_rel_rot6d` + heading rotation
7. Return full world-frame `base_xyz_quat`, `joint_pos`, `obj_xyz_quat`

---

## 6. Alternative Dataset Classes

### `BufferDataset` (`datasets/buffer_dataset.py`)
- Thin subclass of `FlexibleWindowDataset`
- Designed for loading in-memory trajectory buffers during RL rollouts
- Same interface but loads data directly from provided buffer rather than disk

### `ConditionalDataset` (`datasets/conditional_dataset.py`)
- Legacy dataset class for pre-processed `.npz` files
- Expects pre-computed keys: `future`, `history`, `goal`, `current_state`
- Uses fixed `[-1, 1]` normalization
- Not used in the current training pipeline

---

## 7. Data Flow Summary

```
Raw .npz files (absolute world-frame coordinates)
        │
        ▼
FlexibleWindowDataset._preload_from_buffer()
   ├── Schema detection & conversion
   ├── Downsampling (stride=2)
   └── Goal extraction from task_params
        │
        ▼
FlexibleWindowDataset._index_dataset()
   └── Create (file_idx, batch_idx, t_start) window indices
        │
        ▼
FlexibleWindowDataset.__getitem__(idx)
   ├── Extract window from trajectory (with padding)
   ├── _compute_transform()
   │     ├── compute_sbto_components()  → SBTO features + anchors
   │     └── compute_task_params()      → [dir_x, dir_y, dist]
   ├── Assemble features per feature_order → (T, 51)
   ├── _normalize() per feature
   ├── Extract current_state from history frame → (45,)
   └── Add noise to history frames (optional)
        │
        ▼
Output: (future_states, current_state, task_params, anchor)
         (T, 51)       (45,)          (3,)         (anchor_dim,)
```

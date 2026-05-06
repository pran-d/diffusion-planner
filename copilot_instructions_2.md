# Data Augmentation Plan: Synthetic Approach Phase Generation

## Background & Motivation

### The Problem

All trajectories in the dataset follow a fixed structure:

```
[bend+pick (box near robot)] → [walk (box in hand)] → [bend+drop (box at goal)]
```

The robot **never starts far from the box**. As a result, the model has never learned what to do when it sees a box at a distance — it has no concept of "walk toward the box." At inference time, if the box is placed anywhere other than directly in front of the robot, the model has no learned behavior to fall back on.

### The Goal

We want the model to learn:

> *"If the box is in a certain direction relative to my pelvis, walk toward it — regardless of what the goal vector says."*

This requires synthetically manufacturing an **approach phase** from existing data, without collecting any new trajectories.

### Key Constraint

The observation space uses **`obj_rel_pos`** — object position relative to pelvis expressed in the pelvis body frame, 3D (x, y, z). This is computed by `compute_relative_se3` inside `compute_sbto_components`. Every manipulation of the object position signal must be done at the **world-frame raw data level** (modifying `base` and `obj` arrays), since `obj_rel_pos` is derived on-the-fly inside `FlexibleWindowDataset._compute_transform` via the SBTO pipeline.

---

## Core Insight

During the **walk phase**, the box is physically in the robot's hand. This means `obj_rel_pos ≈ [~0, ~0, ~0.3]` (small XY, roughly constant hand-height Z offset) throughout the walk. The model currently sees:

```
walk timestep t:  obj_rel_pos = [~0, ~0, ~hand_z],  task_params = goal_direction_to_drop
```

We are going to **lie to the model** at the raw data level by replacing the world-frame `obj` positions during the walk phase, so that after the SBTO transform the model sees:

```
walk timestep t:  obj_rel_pos = [far away, toward drop location, ground_z],  task_params = random
```

Now the walk phase, which contains perfectly good locomotion actions, looks like the robot walking **toward** a distant box on the ground. The joint actions are untouched. Only the `obj` world positions change.

The resulting augmented trajectory:

```
[walk (fake distant obj, random goal)] → [bend+drop, reinterpreted as bend+pick]
```

---

## Codebase Data Structures

### Raw Buffer Format

Each trajectory in `ram_cache` (after `_preload_from_buffer`) has:
- `base`: `(B, T, 7)` — pelvis world pose `[x, y, z, qw, qx, qy, qz]`
- `joints`: `(B, T, 29)` — joint positions
- `obj`: `(B, T, 7)` — object world pose `[x, y, z, qw, qx, qy, qz]`
- `goal_obj_world`: `(B, 3)` — world-frame drop goal position

All augmentation must operate on `base[b, :, :]` and `obj[b, :, :]` for a given batch index `b`.

### `obj_rel_pos` (3D, not 2D)

This is the full 3D relative position, computed in `compute_sbto_components`:

```python
obj_rel_pos, obj_rel_rot = compute_relative_se3(
    base_w[..., :3], base_w[..., 3:],   # pelvis pos + quat
    obj_w[..., :3],  obj_w[..., 3:]     # object pos + quat
)
```

During Phase 3 (bend+drop on the ground), `obj_rel_pos` returns to near-zero XY and a small positive Z (ground-level box slightly below/at pelvis height). This is the signal the model must see at the end of the approach.

### `task_params` (4D)

`task_params` is computed in `compute_task_params` as:

```
[normalized_dir_x, normalized_dir_y, normalized_dir_z, distance]
```

where the direction is the normalized 3D vector from current object position to the desired drop goal, expressed in the pelvis-yaw frame. The config has `num_task_params: 4`.

For **goal randomisation** in Phase 2, we need to override `goal_obj_world` in the augmented trajectory entry so that `compute_task_params` produces a decorrelated output.

---

## Trajectory Segmentation

Each trajectory must be split into three phases before augmentation.

### Phase 1 — Initial Bend + Pick
- Robot starts near the box, bends down, picks it up.
- `obj[:, :3]` (world frame) is approximately stationary while `base[:, :3]` has not yet started translating significantly.
- `obj_rel_pos` XY is small (box is close in pelvis frame).
- **This phase is dropped entirely** from the augmented trajectory.

### Phase 2 — Walk
- Robot walks from start location to goal location with box in hand.
- `obj[:, :3]` tracks `base[:, :3]` closely (box moves with robot).
- `obj_rel_pos` XY ≈ 0 (box is in the hand).
- **This is the augmentation target.** We replace `obj` world positions here.
- Actions (`joints`) are kept completely unchanged.

### Phase 3 — Final Bend + Drop
- Robot arrives at goal, bends down, places the box.
- `obj[:, :3]` decouples from `base[:, :3]` and becomes stationary again.
- `obj_rel_pos` XY returns to near zero.
- **This phase is kept unchanged** — both `base` and `obj` world data and all actions.

### Segmentation Strategy

Detect walk phase (Phase 2) from the **raw world-frame data**:

```python
# For a single trajectory: base shape (T, 7), obj shape (T, 7)
# Box is "in hand" when it moves with the pelvis:
pelvis_xy = base[:, :2]       # (T, 2)
obj_xy    = obj[:, :2]        # (T, 2)
hand_offset = obj_xy - pelvis_xy   # (T, 2) — nearly constant during walk

# Walk phase: hand_offset has small variance over a rolling window
# Alternative simpler fallback:
#   Phase 1: first K timesteps (trim conservatively, e.g. K = 30)
#   Phase 2: middle timesteps
#   Phase 3: last M timesteps (must include full bend+drop motion, e.g. M = 60)
```

Tune `K` and `M` by inspecting a few trajectories via `plot_trajectory_npz.py` or by loading raw `.npz` files directly. Over-trimming Phase 1 slightly is preferable to contaminating Phase 2 with pick-up frames.

---

## Replacement Object Position

### Conceptual Goal

At each walk timestep `t`, replace `obj[t, :3]` (world frame) with a position that, after the SBTO transform, makes `obj_rel_pos` look like:

> *"The box is approximately where I will be at the end of this walk, but further away and at ground level."*

### Computation

```python
import numpy as np
from utils.math.sbto_utils import yaw_from_quat

def augment_approach_phase(base, obj, walk_start, walk_end, scale_range=(1.0, 1.5)):
    """
    base:       (T, 7)  world-frame pelvis [x, y, z, qw, qx, qy, qz]
    obj:        (T, 7)  world-frame object [x, y, z, qw, qx, qy, qz]
    walk_start: int     first timestep of Phase 2
    walk_end:   int     last  timestep of Phase 2 (exclusive)
    """
    obj_aug = obj.copy()
    T_final = walk_end - 1   # last walk timestep → where box will be dropped

    # Drop location ≈ pelvis position at end of walk
    drop_xy_world = base[T_final, :2]           # (2,)

    # Ground-level Z: use the object's Z during Phase 3 (box on floor)
    # A safe proxy is the object Z value at the very start of Phase 3
    ground_z = obj[walk_end, 2]                 # scalar

    for t in range(walk_start, walk_end):
        # Vector from current pelvis to drop location, in world XY
        residual_xy = drop_xy_world - base[t, :2]   # (2,)
        residual_z  = ground_z - base[t, 2]         # scalar (drop Z rel to pelvis Z)

        # Extend distance to add variety (avoids model seeing only one exact distance)
        scale = np.random.uniform(*scale_range)
        obj_aug_xyz = np.array([
            base[t, 0] + residual_xy[0] * scale,
            base[t, 1] + residual_xy[1] * scale,
            base[t, 2] + residual_z,             # keep actual ground level Z
        ])

        obj_aug[t, :3] = obj_aug_xyz
        # Keep object rotation from Phase 3 start (box lying on floor orientation)
        obj_aug[t, 3:] = obj[walk_end, 3:]

    return obj_aug
```

### Why This Works

- At the **start of the walk phase**, `residual_xy` is large (robot far from drop) → after SBTO, `obj_rel_pos` XY is large → model sees box far away → walks forward. ✓
- As the walk **progresses**, `residual_xy` shrinks → `obj_rel_pos` decreases → mirrors a real approach. ✓
- At the **end of the walk phase**, `residual_xy → 0` → matches Phase 3, where box is truly nearby. ✓

The distance profile decreases naturally across the walk phase, exactly mimicking a real approach trajectory.

### Frame Alignment Note

`obj_rel_pos` is expressed in the **full pelvis body frame** (not just yaw-rotated) via `compute_relative_se3`. By computing the fake `obj` world position and letting SBTO handle the frame transform, you automatically get the correct pelvis-frame `obj_rel_pos` — no manual rotation needed. This is why augmentation must be done at the world-frame `obj` level, not by directly writing to `obj_rel_pos`.

---

## Goal Vector Randomisation

### Motivation

During Phase 2, `task_params` is computed from `(current_obj_pos → goal_obj_world)`. With the fake obj position (far away in the drop direction), the computed `task_params` direction would accidentally align with the approach direction — teaching the model to use the goal as an approach proxy.

Instead, override `goal_obj_world` for the augmented trajectory with a **random, decorrelated** position.

### Procedure

The `goal_obj_world` field is stored per-trajectory in `ram_cache`. For the augmented trajectory:

```python
def sample_random_goal(obj_start_world, all_goal_distances, rng=None):
    """
    Sample a random goal position uncorrelated with the approach direction.
    
    obj_start_world: (3,) current object world position
    all_goal_distances: 1D array of real goal distances from the dataset
    """
    rng = rng or np.random.default_rng()
    
    # Random direction in XY, random Z within plausible range
    theta = rng.uniform(0, 2 * np.pi)
    dist  = rng.choice(all_goal_distances)      # empirical magnitude
    
    delta = np.array([
        dist * np.cos(theta),
        dist * np.sin(theta),
        0.0,                                    # goal is at floor level
    ])
    return obj_start_world + delta
```

### Tradeoff: Phase 3 Goal Conditioning

Setting a random `goal_obj_world` for the whole augmented trajectory means Phase 3 windows in that trajectory also receive the wrong goal. This is acceptable because:

1. **Original trajectories** already cover Phase 3 with correct goal conditioning — the augmented data supplements, not replaces, the original.
2. The `task_condition` dropout (`condition_dropout_prob.task: 0.2`) provides additional robustness.

If Phase 3 goal accuracy is critical, create the augmented trajectory as **Phase 2 only** (stopping before Phase 3) and rely on original data for Phase 3. The transition junction (Phase 2 → Phase 3) would then not be explicitly represented in augmented data, but the original Phase 1 → Phase 2 → Phase 3 trajectories cover the interaction.

---

## Full Augmented Trajectory Structure

| Phase | Included? | `obj` world pos | `goal_obj_world` | `joints` (actions) |
|---|---|---|---|---|
| Phase 1: bend+pick | **No — dropped** | — | — | — |
| Phase 2: walk | **Yes** | Replaced: pelvis + residual × scale, Z = ground level | **Randomised** | **Unchanged** |
| Phase 3: bend+drop | **Yes** | **Unchanged** | **Unchanged (random from Phase 2)** | **Unchanged** |

The augmented trajectory starts at Phase 2, frame index `walk_start`.

---

## Implementation: Where to Hook In

The cleanest integration is a **preprocessing script** that produces augmented `.npz` files (same format as the original data), which are then loaded alongside the real data.

### Script Outline

```python
# utils/data/create_approach_augmentation.py
import numpy as np
import os
import glob
from utils.math.sbto_utils import yaw_from_quat

def segment_phases(base, obj, K=30, M=60):
    """Return (walk_start, walk_end) indices for a trajectory."""
    T = base.shape[0]
    walk_start = K
    walk_end   = T - M
    # Validate
    assert walk_end > walk_start + 10, "Trajectory too short for Phase 2"
    return walk_start, walk_end

def augment_trajectory(base, obj, walk_start, walk_end, scale_range=(1.0, 1.5), rng=None):
    """
    Produce augmented (base_aug, obj_aug) starting at walk_start.
    base: (T, 7), obj: (T, 7)
    Returns sliced (T', 7) arrays for the augmented trajectory.
    """
    rng = rng or np.random.default_rng()
    obj_aug  = obj.copy()
    drop_xy  = base[walk_end - 1, :2]
    ground_z = obj[walk_end, 2]

    for t in range(walk_start, walk_end):
        residual_xy = drop_xy - base[t, :2]
        residual_z  = ground_z - base[t, 2]
        scale = rng.uniform(*scale_range)
        obj_aug[t, :3] = np.array([
            base[t, 0] + residual_xy[0] * scale,
            base[t, 1] + residual_xy[1] * scale,
            base[t, 2] + residual_z,
        ])
        obj_aug[t, 3:] = obj[walk_end, 3:]   # floor-level rotation

    # Slice from walk_start onward
    return base[walk_start:], obj_aug[walk_start:]

def run_augmentation(input_dir, output_dir, prob=0.5, K=30, M=60, rng=None):
    rng = rng or np.random.default_rng(0)
    os.makedirs(output_dir, exist_ok=True)

    # Collect all goal distances from the dataset for sampling
    all_goal_dists = []
    for path in glob.glob(os.path.join(input_dir, "**/*.npz"), recursive=True):
        d = np.load(path)
        # Compute goal distances and accumulate...
        pass  # fill in based on your data schema

    for path in glob.glob(os.path.join(input_dir, "**/*.npz"), recursive=True):
        if rng.random() > prob:
            continue                              # skip this trajectory

        d      = np.load(path)
        # Use key_mapping from config: root_pos → base xyz, etc.
        base   = np.concatenate([d['root_pos'], d['root_rot']], axis=-1)  # (T, 7)
        obj    = np.concatenate([d['object_pos'], d['object_rot']], axis=-1)

        try:
            walk_start, walk_end = segment_phases(base, obj, K=K, M=M)
        except AssertionError:
            continue

        base_aug, obj_aug = augment_trajectory(base, obj, walk_start, walk_end, rng=rng)

        # Random goal (decorrelated from approach direction)
        obj_start = obj_aug[0, :3]
        goal_dist  = rng.choice(all_goal_dists)
        theta      = rng.uniform(0, 2 * np.pi)
        goal_world = obj_start + np.array([goal_dist * np.cos(theta), goal_dist * np.sin(theta), 0.0])

        out_path = os.path.join(output_dir, os.path.relpath(path, input_dir))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path,
            root_pos=base_aug[:, :3],
            root_rot=base_aug[:, 3:],
            dof_pos=d['dof_pos'][walk_start:],
            object_pos=obj_aug[:, :3],
            object_rot=obj_aug[:, 3:],
            goal_obj_world=goal_world,
        )
```

Then add the augmented directory to `config/paths.yaml::paths.train_path` alongside the original data, or load both with `utils/data/load_dataset.py`.

### Alternative: In-Training Augmentation

If you prefer not to create additional files, add a `_augment_approach_phase` method to `FlexibleWindowDataset` and call it in `_preload_from_buffer`. The augmented trajectories are added to `ram_cache` as new entries alongside the originals. The `goal_obj_world` field can be set per-entry when appending.

---

## What the Model Learns

| `obj_rel_pos` signal | Expected Learned Behaviour |
|---|---|
| Large XY magnitude, any direction, any `task_params` | Walk toward the box (approach) |
| XY shrinking across consecutive timesteps | Continue approach |
| XY ≈ 0, Z ≈ floor offset, specific `task_params` | Bend down and interact (pick/drop) |

The goal vector (`task_params`) becomes **irrelevant to locomotion** and **relevant only to manipulation** — which is the correct inductive bias.

---

## Inference-Time Stitching

At inference, the full desired behaviour (new scenario: box placed far from robot):

```
1. Box placed far from robot
2. Set current obs: obj_rel_pos = [far_x, far_y, floor_z] (actual measured box position)
3. Model sees large obj_rel_pos → executes approach walk (learned from augmented Phase 2)
4. obj_rel_pos shrinks as robot approaches → walk continues
5. obj_rel_pos ≈ 0 → model transitions to bend+pick (learned from Phase 3)
6. Box in hand, model executes walk to goal → drop (original data)
```

The stitch at step 5→6 works because Phase 3 (bend+drop) is kinematically similar to Phase 1 (bend+pick), so the model generalises via shared joint trajectories.

---

## Assumptions and Validation Checklist

### ✅ Coordinate Frame Check
- Print `obj_rel_pos` values for Phase 3 frames of a real trajectory. They should be `[near_zero_x, near_zero_y, small_positive_z]`.
- Print `obj_rel_pos` for a Phase 2 frame after augmentation (compute SBTO manually). The XY should be in the 0.5–3.0 m range, Z should be negative-ish (box below pelvis height).
- Verify sign convention matches real observations.

### ✅ Distance Profile Decreasing
- `norm(obj_rel_pos_xy[t])` should decrease monotonically (with possible bumps from curved walking paths) across the augmented Phase 2 window.

### ✅ Goal–Object Decorrelation
```python
cos_sims = [
    np.dot(obj_rel_pos_xy[t], task_dir_xy[t]) /
    (np.linalg.norm(obj_rel_pos_xy[t]) * np.linalg.norm(task_dir_xy[t]) + 1e-8)
    for t in walk_indices
]
assert np.mean(np.abs(cos_sims)) < 0.3   # should be near zero if truly random
```

### ✅ Normalisation Bounds
`obj_rel_pos` in augmented data can be much larger than any real training value (box in hand ≈ 0.3 m vs. fake approach ≈ 1–3 m). With `normalization_type: min_max`, the augmented data will extend the normalisation range, which is correct: recompute stats **after** including augmented data, or use a large fixed clip range for `obj_rel_pos`.

### ✅ Action Distribution Unchanged
```python
# Joint stats in augmented trajectories must match originals
assert np.allclose(augmented_joints.mean(0), original_joints.mean(0), atol=0.05)
```

### ✅ Kinematic Similarity of Phase 1 and Phase 3
Plot joint angle trajectories for Phase 1 (pick) and Phase 3 (drop) side-by-side using `plot_trajectory_npz.py`. They should share deep knee bend, torso lean, and arm reach pattern.

---

## Augmentation Hyperparameters

| Parameter | Suggested Value | Notes |
|---|---|---|
| Distance scale factor | `Uniform(1.0, 1.5)` | Multiplied on the residual to drop location |
| Phase 1 trim `K` | ~30 timesteps at raw 30 Hz | Inspect per dataset |
| Phase 3 trim `M` | ~60 timesteps at raw 30 Hz | Must preserve full bend+drop |
| Goal magnitude source | Empirical distribution from `goal_obj_world` across all trajectories | |
| Augmentation probability | 0.5 per trajectory | Mix real and augmented |

---

## Summary

The augmentation repurposes the walk phase of every existing trajectory as synthetic approach training data by:
1. **Dropping Phase 1** (pick-up frames).
2. **Replacing world-frame `obj[:, :3]`** during Phase 2 with a dynamically computed fake position: pelvis position + residual vector to drop location (scaled up), at ground-level Z. The SBTO pipeline then derives the correct `obj_rel_pos` automatically.
3. **Randomising `goal_obj_world`** for the augmented trajectory to decorrelate goal conditioning from approach direction.
4. **Keeping Phase 3 intact** — both observations and actions — so it serves as the pick-up behaviour at inference.
5. **No actions are modified.** The augmentation teaches the model what to observe, not what to do.

Key codebase touchpoints: `datasets/flexible_dataset.py:_preload_from_buffer`, `utils/math/sbto_utils.py:compute_sbto_components`, `utils/data/load_dataset.py`, `config/paths.yaml`.

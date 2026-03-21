# Training & Inference Pipeline

## Part 1: Training

### 1.1 Training Loop Overview

**Location:** `train.py` (~564 lines)

The training loop uses standard PyTorch with the following components:

| Component              | Details                                                   |
|------------------------|-----------------------------------------------------------|
| Optimizer              | AdamW, lr=1e-4, weight_decay=1e-6                         |
| LR Scheduler           | CosineAnnealingLR with T_max = total epochs               |
| Mixed Precision        | GradScaler with autocast (fp16)                            |
| EMA                    | Exponential Moving Average of model weights, decay=0.999   |
| Gradient Clipping      | `max_norm=1.0`                                             |
| Early Stopping         | patience=100, target_loss=0.005                            |
| Sampler                | WeightedRandomSampler for task density balancing           |

### 1.2 Training Step

Each training step:

```python
future_states, current_state, task_params, anchor = batch

# Construct model condition tuple
model_cond = (current_state, task_params)  # (B, 45), (B, 3)

# Forward pass with automatic mixed precision
with autocast():
    loss = model(future_states, model_cond=model_cond)

# Backward + optimize
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()

# EMA update
ema.update()
```

### 1.3 EMA (Exponential Moving Average)

The EMA model maintains a shadow copy of all model parameters:

```python
ema_param = decay * ema_param + (1 - decay) * model_param
```

- Decay: 0.999
- The EMA model is used for:
  - Validation loss computation
  - Physical consistency evaluation
  - Final checkpoint saving for inference
- EMA weights are swapped in during evaluation, then restored

### 1.4 Task Density Balancing

To ensure balanced sampling across different task types (e.g., walk, carry, place), a `WeightedRandomSampler` is used:

1. Count samples per task type
2. Assign weight = 1 / count for each sample
3. WeightedRandomSampler draws samples proportional to these weights

This prevents the model from being biased toward over-represented tasks.

### 1.5 Physical Consistency Evaluation

Periodically during training, generative evaluation runs the EMA model in inference mode:

1. **Generate trajectories** from random validation conditions using `model.sample()`
2. **Reconstruct** world-frame coordinates via SBTO inverse transform
3. **Check physical consistency**:
   - **Floor penetration**: base_z or object_z below threshold
   - **XY spikes**: sudden large displacements between consecutive frames
4. Results logged to wandb

### 1.6 Checkpointing

Saved at each epoch (if improved) and at fixed intervals:

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),
    'ema': ema.state_dict(),
    'epoch': epoch,
    'best_loss': best_loss,
}
```

---

## Part 2: Waypoint Masking System

The waypoint masking system is the key innovation for enabling controlled generation. It operates at two levels: **frame-level** (which frames are keyframes) and **feature-level** (which features within a frame are known).

### 2.1 Frame-Level: In-Betweening Mask

**Method:** `DFoTTrajectory._generate_inbetweening_mask()`

**Configuration:**
```yaml
inbetweening:
  enabled: true
  min_keyframes: 0
  max_keyframes: 3
  always_keep_first: true
  keep_last_partial: true
```

**Algorithm:**
1. For each batch sample, randomly choose `num_keyframes ~ U[min_keyframes, max_keyframes]`
2. Randomly select `num_keyframes` frame indices as **full keyframes**
3. If `always_keep_first: true`, frame 0 is always a full keyframe (but this contributes to the count)
4. If `keep_last_partial: true`, the last frame gets a **partial** mask (only some features are known)

**Output:** A frame-level mask indicating for each frame whether it is:
- **Full keyframe**: All 51 features are known
- **Partial keyframe**: Only selected feature groups are known
- **Unknown**: No features constrained

### 2.2 Feature-Level: Partial Masking

**Method:** `DFoTTrajectory._generate_waypoint_mask()`

**Configuration:**
```yaml
partial_masking:
  enabled: true
  feature_groups:
    locomotion:
      features: [obj_delta_xy]
      keep_prob: 0.8
    pick_place:
      features: [obj_z]
      keep_prob: 0.8
    object_pos:
      features: [obj_rel_pos]
      keep_prob: 0.3
    lower_body:
      features: [joints]  # lower body joints subset
      keep_prob: 0.3
    upper_body:
      features: [joints]  # upper body joints subset
      keep_prob: 0.3
    pose:
      features: [body_rot6d, body_z]
      keep_prob: 0.3
    robot_pos:
      features: [delta_xy, delta_yaw]
      keep_prob: 0.3
```

For **partial keyframes** (typically the last frame):
1. Each feature group independently survives with probability `keep_prob`
2. A Bernoulli draw per group determines which features are revealed
3. The result is a binary mask of shape `(B, T, 51)` where `True` = known

### 2.3 Waypoint Injection

**Method:** `DFoTTrajectory._inject_waypoints()`

During both training and inference, known waypoint values are injected into the noised trajectory:

**For fully-known frames** (k=0):
```python
x_noised[b, t, :] = x_clean[b, t, :]  # Replace with clean values
```

**For partially-known frames** (features partially masked):
```python
# RePaint-style: noise the clean values to match the current noise level
x_noised[b, t, known_features] = q_sample(x_clean[b, t, known_features], k_current)
```

This ensures that partially-known features are noised consistently with the surrounding unknown features, preventing distribution mismatch.

### 2.4 Loss Masking

The training loss is **zeroed out on fully-known waypoint frames** since the model doesn't need to learn to predict values it already has:

```python
loss_mask = ~fully_known_frames  # (B, T)
loss = loss * loss_mask.unsqueeze(-1)
```

Partially-known frames still contribute to the loss (the model must learn the unknown features).

---

## Part 3: Sampling / Inference

### 3.1 Sampling Overview

**Location:** `models/dfot_trajectory.py → _sample_sequence()`

The sampling process iteratively denoises from pure noise to a clean trajectory:

```
x_T ~ N(0, I)    ← pure noise, shape (B, T, 51)
    │
    ▼ (iterate through schedule)
x_{T-1} = denoise_step(x_T, k=T)
x_{T-2} = denoise_step(x_{T-1}, k=T-1)
    ...
x_0 = denoise_step(x_1, k=1)    ← clean trajectory
```

**Inference timesteps:** 10 steps (sub-sampled from 200 training steps)

### 3.2 Scheduling Matrix

For `full_sequence` mode, all tokens share the same noise level at each step:

```
Step:  0    1    2    3    4    5    6    7    8    9
k:    200  180  160  140  120  100   80   60   40   20  →  0
       ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
      All tokens have the same noise level
```

### 3.3 Sampling Step with Waypoint Indicator

**Method:** `_sample_step_with_indicator()`

Each denoising step:

1. **Forward pass** with waypoint indicator:
   ```python
   indicator_emb = waypoint_indicator_proj(waypoint_mask.float())
   # Added to hidden states inside DiT1D
   v_pred = backbone(x_k, k, ext_cond, indicator_emb=indicator_emb)
   ```

2. **Predict x₀** from v-prediction:
   ```python
   x_0_pred = sqrt(alpha_bar) * x_k - sqrt(1 - alpha_bar) * v
   ```

3. **DDIM step** to get x_{k-1}:
   ```python
   x_{k-1} = sqrt(alpha_bar_{k-1}) * x_0_pred + sqrt(1 - alpha_bar_{k-1}) * eps_pred
   ```

4. **Inject waypoints** (overwrite known values in x_{k-1})

5. **RePaint resampling** (if enabled, `resample_steps: 2`):
   - Re-noise x_{k-1} back to x_k
   - Re-denoise x_k → x_{k-1}
   - Repeat `resample_steps` times
   - This improves coherence between waypoint-constrained and free regions

### 3.4 Classifier-Free Guidance at Inference

```python
# Batched CFG: concatenate conditional and unconditional inputs
x_doubled = torch.cat([x_k, x_k], dim=0)
cond_doubled = torch.cat([ext_cond, zeros], dim=0)

v_pred_doubled = backbone(x_doubled, k, cond_doubled)
v_cond, v_uncond = v_pred_doubled.chunk(2)

v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
```

Default guidance scale: configurable (typically 1.0–2.0).

### 3.5 Inpainting (State Condition at t=0)

At each denoising step, frame 0 of the trajectory is overwritten with the **current state observation** features:

```python
# The first frame's observable features (joints, body_z, body_rot6d, obj_rel_pos, obj_rel_rot6d)
# are set to the current_state values, noised to match the current noise level
x_k[:, 0, observable_features] = q_sample(current_state_features, k)
```

This ensures the generated trajectory starts from the actual robot state.

---

## Part 4: Autoregressive Inference with Window Stitching

**Location:** `inference_mg.py` + `motion_generator.py` + `utils/inference_utils.py`

For generating long trajectories beyond the 20-timestep window, an **autoregressive stitching** approach is used.

### 4.1 Stitching Overview

```
Window 1:  [s0, s1, s2, ..., s19, s20]   ← generate 21 frames
                            ↓ overlap
Window 2:        [s15, s16, ..., s34, s35]  ← condition on s15-s20 from Window 1
                                  ↓ overlap
Window 3:              [s30, s31, ..., s49, s50]
                                        ↓ ...
```

**Key parameters:**
- `stitch_steps`: number of autoregressive windows to generate
- Each window generates `num_timesteps` (20) new frames
- Overlap/conditioning from previous window provides continuity

### 4.2 First Window: Building Waypoints

**Method:** `build_inference_waypoints()`

The first window receives two types of waypoints:

#### A. First Frame Keyframe (`build_first_keyframe_from_state`)

Constructs a full keyframe at t=0 from the current observation:
```python
keyframe = zeros(51)
keyframe[delta_xy]     = 0     # No displacement at start
keyframe[delta_yaw]    = 0     # No rotation at start
keyframe[obj_delta_xy] = 0     # No object displacement at start
keyframe[obj_z]        = current_obj_z   # Current object height
keyframe[joints]       = current_joints  # Current joint angles
keyframe[body_z]       = current_body_z  # Current body height
keyframe[body_rot6d]   = current_rot6d   # Current body orientation
keyframe[obj_rel_pos]  = current_obj_rel # Current relative object position
keyframe[obj_rel_rot6d] = current_obj_rel_rot  # Current relative object rotation
```

Mask: all 51 features = True (fully known)

#### B. Last Frame Partial Waypoint (`build_last_frame_waypoints`)

Provides soft guidance for the trajectory's end state. Only object-related features are constrained:

**Features provided:**
- `obj_delta_xy`: Target XY displacement for the object (from trapezoidal profile)
- `obj_z`: Target z-height for the object (from z-profile)

**Mask:** Only `obj_delta_xy` and `obj_z` are True (partial waypoint)

### 4.3 XY Displacement Profile (Trapezoidal)

**Method:** `build_last_frame_waypoints()` → XY profile

The object's XY displacement follows a **trapezoidal velocity profile** for smooth acceleration/deceleration:

```
Velocity
  ▲
  │     ┌─────────────────┐
  │    /│                 │\
  │   / │                 │ \
  │  /  │                 │  \
  │ /   │                 │   \
  └─────┼─────────────────┼─────► Progress (0 to 1)
  0   t_accel           1-t_decel  1
        (0.35)            (0.65)
```

**Parameters:**
- `t_accel = 0.35` — fraction of trajectory spent accelerating
- `t_decel = 0.35` — fraction spent decelerating
- `arrival_ratio = 0.85` — only 85% of the goal distance is targeted per window (to prevent overshooting)

**Walk gating:** The robot doesn't start walking until the object has been lifted (controlled by `walk_gate`):
```python
if object_z < lift_threshold:
    delta_xy = 0   # Don't walk yet
```

### 4.4 Z-Height Profile (Raised Cosine Bell)

The object z-height follows a **smooth bell curve** for pick-up → carry → place:

```
Object Z
  ▲
  │         ┌──────────────┐
  │        /│  carry_height │\
  │       / │   (0.20m)     │ \
  │      /  │               │  \
  │     /   │               │   \
  │    /    │               │    \
  ────┘     │               │     └────
  0     lift_end        lower_start    1
        (0.20)                        → Progress
```

**Profile phases:**
1. **Lift** (`0 → lift_end=0.20`): Raised cosine ramp from ground to carry height
2. **Carry** (`lift_end → lower_start`): Constant carry height (0.20m)
3. **Lower** (`lower_start → 1.0`): Raised cosine ramp from carry height to ground
   - `lower_start` is computed based on remaining distance to goal

**Key formula (raised cosine segments):**
```python
z = carry_height * 0.5 * (1 - cos(π * progress / phase_length))
```

### 4.5 Condition Update Between Windows

**Method:** `update_condition()`

Between autoregressive windows, the robot's state condition must be updated based on the generated trajectory:

1. **Take the last few frames** from the generated window (in world frame)
2. **Re-compute SBTO transform** relative to the new anchor (last frame's position/heading)
3. **Extract new `current_state`** from the re-anchored features
4. **Compute new `task_params`** relative to the new robot position
5. **Build new waypoints** for the next window

This ensures each window starts from the correct robot state and goal direction.

### 4.6 World-Frame Reconstruction

During inference, each window's output is in SBTO (robot-relative) coordinates. To build the full trajectory:

1. **Denormalize** the predicted features
2. **Reconstruct world frame** via `reconstruct_sbto_trajectory()`:
   - Integrate `delta_xy` through heading rotations to get absolute XY
   - Cumulative sum of `delta_yaw` for absolute heading
   - Apply body_rot6d residual to get full orientation
   - Transform `obj_rel_pos/rot` back to world frame
3. **Concatenate** world-frame outputs from all windows (with overlap handling)

### 4.7 Physical Consistency Checks (Inference)

Post-generation checks:

1. **Floor penetration**: `base_z < threshold` or `obj_z < threshold`
2. **XY spikes**: `||delta_xy[t]|| > spike_threshold` (unrealistic jumps)
3. **Joint limits**: Optionally check if joint angles stay within valid ranges
4. **Velocity smoothness**: Check for discontinuities at window boundaries

---

## Part 5: Masking & Waypoint Summary

### Complete Masking Flow (Training)

```
1. _generate_inbetweening_mask()
   ├── Select 0-3 random keyframes (frame indices)
   ├── First frame: always full keyframe
   └── Last frame: partial keyframe (if keep_last_partial)

2. _generate_waypoint_mask()
   ├── Full keyframes → all 51 features = True
   └── Partial keyframes → per-group Bernoulli sampling
       ├── locomotion (obj_delta_xy): 80% chance
       ├── pick_place (obj_z): 80% chance
       ├── object_pos (obj_rel_pos): 30% chance
       ├── lower_body (joint subset): 30% chance
       ├── upper_body (joint subset): 30% chance
       ├── pose (body_rot6d, body_z): 30% chance
       └── robot_pos (delta_xy, delta_yaw): 30% chance

3. Noise level override
   ├── Full keyframes: k = 0 (clean)
   └── Partial keyframes: k ~ U[0, max_wp_k]

4. _inject_waypoints()
   ├── Full keyframes: overwrite with clean x_0
   └── Partial keyframes: RePaint-noise clean values to current k

5. Loss masking
   └── Zero out loss on fully-known frames
```

### Complete Masking Flow (Inference)

```
1. build_inference_waypoints()
   ├── Frame 0: full keyframe from current state
   └── Frame T-1: partial waypoint (obj_delta_xy, obj_z from profiles)

2. At each denoising step:
   ├── Forward pass with waypoint_indicator_proj
   ├── DDIM step
   ├── Inject waypoints (clean/RePaint-noised)
   └── RePaint resampling (2 iterations)

3. Inpainting at frame 0:
   └── Overwrite observable features with current state
```

---

## Part 6: Configuration Reference

### Key Training Config

```yaml
training:
  batch_size: 256
  epochs: 1000
  learning_rate: 1e-4
  weight_decay: 1e-6
  grad_clip: 1.0
  ema_decay: 0.999
  early_stopping_patience: 100
  target_loss: 0.005
```

### Key Inference Config

```yaml
inference:
  guidance_scale: 1.0       # CFG scale
  stitch_steps: 5           # number of autoregressive windows
  resample_steps: 2         # RePaint resampling iterations
  deterministic: true        # DDIM (no stochastic noise)
  inference_timesteps: 10    # denoising steps
```

### Ablation Configs Available

| Config                                     | Description                              |
|--------------------------------------------|------------------------------------------|
| `ablations/geom_loss/config.yaml`          | Geometric loss ablation                   |
| `ablations/masking/config.yaml`            | Masking strategy ablation                 |
| `ablations/masking/config_increased_noise.yaml` | Higher noise variant              |
| `ablations/masking/partial_masking/`       | Partial masking variants                  |
| `ablations/normalized_goal/`               | Goal normalization ablation               |
| `ablations/timesteps/config_{1,5,10,20}.yaml` | Different prediction horizons        |

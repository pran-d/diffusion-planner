# Diffusion Motion Planner

Diffusion-based motion planning for a Unitree G1 humanoid robot performing pick-and-place tasks. The model generates full-body trajectories (29 DoF joints + 6D body pose + object pose) conditioned on task goals using a Diffusion Transformer backbone.

## Setup

### Installation (uv — recommended)

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Activate (or prefix commands with `uv run`)
source .venv/bin/activate
```

### Installation (conda)

```bash
conda env create -f environment.yaml
conda activate diffusers
```

### Data Setup

Place your trajectory dataset (`.npz` files) in a folder and update `config/paths.yaml`:

```yaml
paths:
    dir_path: <absolute/path/to/your/workspace>
    train_path: <relative/path/to/data/folder> # should contain .npz files
```

Additionally, update the path to the Unitree G1 robot assets (.STL mesh files) in your local machine (these are also included in the repository, in the folder `unitree_g1`).

Each `.npz` file should contain arrays keyed by the names in `data.key_mapping` (e.g. `root_pos`, `root_rot`, `dof_pos`, `object_pos`, `object_rot`) as described in `config/config.yaml` with shape `(n_batch, time, feature_dim)`.

## Training

### Standalone script

```bash
python train.py
```

Key flags:
| Flag | Description |
|------|-------------|
| `--resume N` | Resume training from checkpoint epoch N |
| `--save_every N` | Save checkpoint every N epochs (default: 50) |

Checkpoints are saved to the path derived from your config (typically `runs/checkpoints/`). Each checkpoint includes model weights, optimizer state, scheduler state, and EMA state. A separate `ema_model_{epoch}.pth` file with EMA-smoothed weights is also saved alongside.

Hyperparameters are configured in `config/config.yaml` under the `training` section.

### Via MotionGenerator API

```python
from motion_generator import MotionGenerator
from utils.data.load_dataset import preload_dataset

mg = MotionGenerator(config_path="config/config.yaml", device="cuda")

# Load data into memory buffer
data_buffer = preload_dataset(mg.data_cfg, data_path)

# Train (checkpoints saved automatically)
mg.fit(
    data_source=data_buffer,
    epochs=500,
    save_path="runs/checkpoints/",
    checkpoint=None,  # or path to resume from
)
```

## Inference

There are two ways to run inference for the motion-planner module.

### 1. `inference_mg.py` — standalone script (single supported inference entry point)

Full control over the autoregressive loop, waypoints, and visualisation:

```bash
# Basic generation with MuJoCo visualisation
python inference_mg.py --epoch 500 --traj_idx 0 --batch_idx 0 --stitch_steps 10

# Use EMA weights
python inference_mg.py --epoch 500 --ema --stitch_steps 10

# Custom goal in local frame
python inference_mg.py --epoch 500 --task_params 0.5 -0.3 --stitch_steps 8

# With waypoint conditioning and early goal stop
python inference_mg.py --epoch 500 --ema --stitch_steps 15 \
    --last_frame_waypoint --enable_goal_stop --cfg_w 1.5
```

Key flags:
| Flag | Description |
|------|-------------|
| `--epoch N` or `--epoch path.pth` | Checkpoint to load |
| `--ema` | Use EMA weights (loads `ema_model_N.pth`) |
| `--stitch_steps N` | Number of autoregressive segments |
| `--traj_idx / --batch_idx / --start_time` | Select initial condition from dataset |
| `--task_params X Y [Z]` | Override goal as local-frame displacement |
| `--cfg_w W` | Classifier-free guidance weight (default: 1.0) |
| `--last_frame_waypoint` | Enable partial waypoint at last frame |
| `--end_error_threshold R` | XY radius (m) for goal-reached check |
| `--end_ground_num_frames N` | Consecutive on-ground frames before stopping |
| `--goal_multiplier M` | Scale goal displacement from base traj |
| `--save_path results/run_01.npz` | Save full trajectory bundle as `.npz` |
| `--metrics_log_path results/inference_metrics.jsonl` | Append goal direction/magnitude/error and generation time |

#### Centralized inference defaults (no long CLI)

All inference/eval scripts can read defaults from [config/inference.yaml](config/inference.yaml).

```bash
# Run with defaults from config/inference.yaml::inference_mg
python inference_mg.py

# Run batch goal sweep from config/inference.yaml::batch_goal_sweep
python batch_goal_sweep.py

# Override only one field at runtime (CLI overrides YAML)
python inference_mg.py --epoch 700 --save_path results/run_700.npz
```

Use `--inference_config` to point to another YAML file.


### 2. MotionGenerator Python API

For integration into larger pipelines (e.g. RL, MPC, sim-to-real):

```python
import numpy as np
from motion_generator import MotionGenerator

# 1. Initialise
mg = MotionGenerator(config_path="config/config.yaml", device="cuda")

# 2. Load weights (regular or EMA)
mg.diffuser.loadWeights(500, ema=True)

# 3. Prepare initial condition
#    robot: (B, H, 36) — pelvis [x,y,z, qw,qx,qy,qz] + 29 joint angles
#    obj:   (B, H, 7)  — object [x,y,z, qw,qx,qy,qz]
#    H = state_history (typically 1)
initial_condition = {
    "robot": robot_state,   # np.ndarray (1, 1, 36)
    "obj":   object_state,  # np.ndarray (1, 1, 7)
}

# 4. Define goal as local-frame displacement [dx, dy, dz]
goal_local = np.array([0.5, -0.3, 0.0])

# 5. Generate
trajectory, real_lengths = mg.generate_trajectory(
    initial_condition=initial_condition,
    goal_condition=goal_local,
    stitch_steps=10,
    cfg_w=1.5,
    enable_goal_stop=True,
    enable_physics_clamp=True,
    use_last_frame_wp=True,
)
# trajectory: (B, T, 43) — robot(36) + object(7) in world frame
# real_lengths: (B,) — actual trajectory length before padding
```

#### Output format

The output trajectory has shape `(B, T, 43)` where each frame contains:
- **Columns 0–2**: Robot pelvis position (x, y, z) in world frame
- **Columns 3–6**: Robot pelvis quaternion (w, x, y, z)
- **Columns 7–35**: 29 joint angles
- **Columns 36–38**: Object position (x, y, z) in world frame
- **Columns 39–42**: Object quaternion (w, x, y, z)

## Configuration

All settings live in `config/config.yaml`. Key sections:

| Section | Controls |
|---------|----------|
| `data` | Dataset path, feature layout, window size, stride, normalisation |
| `model` | Architecture type, hidden size, depth, heads, conditioning |
| `training` | Learning rate, epochs, noise levels, EMA, waypoint masking |
| `noise_scheduler` | Diffusion timesteps, beta schedule, prediction type |

### Feature Layout (51-dim ego-centric representation)

| Index | Feature | Dim |
|-------|---------|-----|
| 0–1 | `delta_xy` — robot XY displacement from current state | 2 |
| 2 | `delta_yaw` — robot yaw change from current state | 1 |
| 3–4 | `obj_delta_xy` — object XY displacement from current state | 2 |
| 5 | `obj_z` — object height | 1 |
| 6–34 | `joints` — 29 joint angles | 29 |
| 35 | `body_z` — robot pelvis height | 1 |
| 36–41 | `body_rot6d` — robot orientation (6D) | 6 |
| 42–44 | `obj_rel_pos` — object position relative to robot pelvis | 3 |
| 45–50 | `obj_rel_rot6d` — object orientation relative to robot pelvis (6D) | 6 |

## Visualisation

Generated trajectories can be played back in MuJoCo:

```bash
# inference_mg.py includes built-in visualisation
python inference_mg.py --epoch 500 --ema --stitch_steps 10

# Batch goal sweep with multi-trajectory overlay
python batch_goal_sweep.py --epoch 500 --traj_idx 0 --num_goals 6
```

Offline NPZ tools:

```bash
# Plot XY/Z/yaw + hand-object distances
python plot_trajectory_npz.py --npz_path results/inference_mg.npz

# Visualize copied NPZ trajectory later on laptop
python visualize_trajectory_npz.py --npz_path results/inference_mg.npz
```

**MuJoCo viewer controls**: `SPACE` pause/play · `→` step forward · `←` step back · `ESC` exit

Requires `mj_model.xml` in the project root.

## Advanced Features

### 1. Style-Conditioned Motion Planning

The diffusion model can now handle **mixed-task training** by automatically detecting and conditioning on motion style: **pick**, **push**, or **kick**. A single unified model learns to generate appropriate trajectories for all three styles without per-style fine-tuning.

#### Style Detection

Style is automatically inferred from trajectory dynamics during training:
- **Pick**: Object lift > 0.08 m detected → high-confidence pick trajectory
- **Push**: Low lift + high object XY velocity → push trajectory
- **Kick**: High foot-contact speed (lower-body joint velocity > 0.9 rad/s) → kick trajectory
- **Text Fallback**: If motion-based detection is ambiguous, task description from `tasks.yml` is used

#### Configuration

Enable style conditioning in `config/config.yaml`:

```yaml
data:
  style_conditioning:
    enabled: true
    detection_mode: motion_then_text  # text_only | motion_only | text_then_motion | motion_then_text
    num_styles: 3                      # pick=0, push=1, kick=2
    seed: 42
    tasks_file: ./test_datasets/tasks.yml
model:
  style_condition: true
  # ... other model settings
training:
  condition_dropout_prob:
    style: 0.05  # 5% dropout for classifier-free guidance
```

#### Training & Inference with Styles

**Training**: Automatically detects style for each trajectory during data loading. Style one-hot encoding is concatenated with state/task conditions.

```bash
# Train with mixed tasks (pick, push, kick)
python train.py
```

**Inference**: Either auto-detect (defaults to push) or specify style via CLI:

```bash
# Default: push style
python inference_mg.py --epoch 500 --ema

# Override to pick or kick
python inference_mg.py --epoch 500 --ema --style pick
python inference_mg.py --epoch 500 --ema --style kick

# Python API
mg.generate_trajectory(
    initial_condition=ic,
    goal_condition=goal,
    style_condition="pick",  # or np.array([1,0,0]) one-hot
)
```

#### Architecture Integration

Style conditioning flows through the model:
1. **Dataset**: Per-trajectory one-hot style vector (B, 3) computed during pre-load
2. **Model**: Style embedding MLP (3 → hidden_size) concatenated with state/task embeddings
3. **Training**: Style dropout (default 5%) for classifier-free guidance
4. **Inference**: Optional CLI override; defaults to push if disabled

### 2. Hierarchical Two-Phase Trajectory Generation

The model uses **per-feature noise scheduling** to decompose trajectory generation into two phases:

- **Phase 1 (Root & Object Planning)**: Lower noise levels for root position (delta_xy, delta_yaw), object displacement (obj_delta_xy, obj_z, obj_rel_pos, obj_rel_rot6d)
  - Goal: Plan where the robot moves and how the object moves
  - Finishes at ~50% of diffusion steps

- **Phase 2 (Joint & Body Pose)**: Higher noise levels for joints, body pose (body_z, body_rot6d)
  - Goal: Plan body posture conditioned on the fixed root/object trajectory
  - Finishes at 100% of diffusion steps

This two-phase approach improves both sample quality and efficiency:
- **Structured generation**: Root/object moves first, body follows → physically plausible trajectories
- **Interpretability**: Can inspect intermediate outputs at the 50% mark to see high-level plan before pose details
- **Robustness**: Phase 1 errors (e.g., unreachable goals) propagate with lower noise, allowing phase 2 to adapt

#### Configuration

Enable hierarchical scheduling in `config/config.yaml`:

```yaml
noise_scheduler:
  noise_level: random_uniform
  scheduling_matrix: hierarchical_two_phase  # full_sequence | autoregressive | hierarchical_two_phase
  hierarchical_noise:
    enabled: true
    phase1_feature_keys:
      - delta_xy          # Robot XY displacement
      - delta_yaw         # Robot yaw change
      - obj_delta_xy      # Object XY displacement
      - obj_z             # Object height
      - obj_rel_pos       # Object pos relative to robot
      - obj_rel_rot6d     # Object orientation relative to robot
    phase2_feature_keys:
      - joints            # Joint angles
      - body_z            # Robot pelvis height
      - body_rot6d        # Robot orientation (6D)
    phase1_max_ratio: 0.5  # Phase 1 finishes at 50% of steps
    phase2_min_ratio: 0.0  # Phase 2 starts from step 0

  hierarchical_schedule:
    enabled: true
    phase1_end_ratio: 0.5  # Schedule: phase 1 clean by step K/2, phase 2 clean by step K
```

#### Training with Hierarchical Noise

During training, phase 1 features receive lower random noise levels than phase 2:

```python
# Example noise sampling (automatic in forward()):
# Phase 1: noise ~ U(0, 0.5 * T)  (cleaner)
# Phase 2: noise ~ U(0, T)         (noisier, but constrained by phase 1)
```

The loss is weighted per timestep/feature using min-SNR reweighting, treating (B, T, D) shaped noise naturally without reshaping.

#### Inference with Hierarchical Scheduling

During sampling, the noise scheduler generates a 3D matrix `(steps, tokens, features)`:

```
Step 0: Phase 1 @ high noise, Phase 2 @ high noise
Step 1: Phase 1 @ medium noise, Phase 2 @ high noise
...
Step K/2: Phase 1 @ 0 (clean), Phase 2 @ medium noise
...
Step K: Phase 1 @ 0, Phase 2 @ 0 (fully clean)
```

This ensures phase 1 features (root/object) are denoised early and held steady while phase 2 (joints/pose) refines over the remaining steps.

#### Benefits in Practice

- **Pick Tasks**: Root reaches goal early, then joints adjust to grasp configuration
- **Push Tasks**: Robot trajectory stabilizes first, joints then optimize for object control
- **Kick Tasks**: Root/stance established, then legs execute kick motion
- **Generalization**: Single model handles all three styles with coherent two-phase generation

## Combining Style Conditioning + Hierarchical Noise

The most powerful configuration uses both features together:

```bash
# Train on mixed tasks with hierarchical two-phase generation
python train.py

# Inference: specify task style, get phase-aware trajectory
python inference_mg.py --epoch 500 --ema --style kick \
    --save_path results/kick_traj.npz
```

This yields a **unified generalist model** that:
1. Detects or accepts motion style (pick/push/kick)
2. Generates root/object trajectory first (phase 1)
3. Refines body/joint posture second (phase 2)
4. Produces physically feasible, task-appropriate motion

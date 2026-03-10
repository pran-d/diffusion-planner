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

### 1. `inference.py` — standalone script

Full control over the autoregressive loop, waypoints, and visualisation:

```bash
# Basic generation with MuJoCo visualisation
python inference.py --epoch 500 --traj_idx 0 --batch_idx 0 --stitch_steps 10

# Use EMA weights
python inference.py --epoch 500 --ema --stitch_steps 10

# Custom goal in local frame
python inference.py --epoch 500 --task_params 0.5 -0.3 --stitch_steps 8

# With waypoint conditioning and early goal stop
python inference.py --epoch 500 --ema --stitch_steps 15 \
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
# inference.py includes built-in visualisation
python inference.py --epoch 500 --ema --stitch_steps 10

# Batch goal sweep with multi-trajectory overlay
python batch_goal_sweep.py --epoch 500 --traj_idx 0 --num_goals 6
```

**MuJoCo viewer controls**: `SPACE` pause/play · `→` step forward · `←` step back · `ESC` exit

Requires `mj_model.xml` in the project root.

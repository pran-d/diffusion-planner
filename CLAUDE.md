# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diffusion-based motion planner for a Unitree G1 humanoid robot (pick/push/kick tasks). Generates full-body trajectories (51-dim ego-centric features) conditioned on task goals using a Diffusion Transformer backbone.

## Commands

### Setup
```bash
uv sync && source .venv/bin/activate   # recommended
# or: conda env create -f environment.yaml && conda activate diffusers
```

### Training
```bash
python train.py                        # train from scratch
python train.py --resume N             # resume from epoch N
python train.py --save_every 50        # checkpoint every 50 epochs
```

### Inference
```bash
python inference_mg.py                                          # uses config/inference.yaml defaults
python inference_mg.py --epoch 500 --ema --stitch_steps 10
python inference_mg.py --epoch 500 --task_params 0.5 -0.3 --cfg_w 1.5
python inference_mg.py --epoch 500 --ema --style pick          # force motion style
python inference_mg.py --epoch 500 --save_path results/run.npz --metrics_log_path results/metrics.jsonl
```

### Evaluation & Visualisation
```bash
python batch_goal_sweep.py --epoch 500 --traj_idx 0 --num_goals 6
python plot_trajectory_npz.py --npz_path results/inference_mg.npz
python visualize_trajectory_npz.py --npz_path results/inference_mg.npz
```

MuJoCo viewer controls: `SPACE` pause/play · `→/←` step · `ESC` exit

## Architecture

### Layer Stack

| Layer | File | Role |
|---|---|---|
| `RobotDiffuser` | `models/model.py` | Thin interface: load weights, call forward |
| `DFoTTrajectory` | `models/dfot_trajectory.py` | Conditioning, waypoint masking, sampling loop, autoregressive stitching |
| `DiscreteDiffusion` | `diffusion_forcing_transformer/discrete_diffusion.py` | Noise scheduling, v-prediction, DDIM, min-SNR loss |

`MotionGenerator` (`motion_generator.py`) wraps `RobotDiffuser` for external API use.

### 51-dim Feature Layout (ego-centric, defined in `config/config.yaml`)

| Idx | Feature | Dim |
|-----|---------|-----|
| 0–1 | `delta_xy` | 2 |
| 2 | `delta_yaw` | 1 |
| 3–4 | `obj_delta_xy` | 2 |
| 5 | `obj_z` | 1 |
| 6–34 | `joints` (29 DoF) | 29 |
| 35 | `body_z` | 1 |
| 36–41 | `body_rot6d` | 6 |
| 42–44 | `obj_rel_pos` | 3 |
| 45–50 | `obj_rel_rot6d` | 6 |

Feature slices are derived programmatically from `build_feature_layout()` in `utils/math/sbto_utils.py`. **Never hardcode indices.**

### Conditioning Mechanisms

- **Noise-level:** Random Fourier features + MLP → AdaLN-Zero conditioning vector `c`
- **Task (goal):** Goal displacement (3–4 dims) → 256-dim MLP → added to `c`
- **State (optional):** Current obs (45 dims) → 256-dim MLP → branch cross-attention
- **Style (optional):** One-hot (3 styles) → 256-dim MLP → concatenated with state/task embeddings
- **CFG:** 10% state dropout + 20% task dropout during training; dynamic thresholding at inference

### Waypoint-Guided Inpainting (In-Betweening)

During training: 0–3 keyframes per trajectory are randomly selected and partially masked per-feature-group (Bernoulli, ~80% for locomotion/lift, ~30% for body pose). Known features are re-noised to current noise level (RePaint-style); loss is zeroed on fully-known frames. At inference: frame 0 is anchored to current state; final frame has soft XY/Z profile constraints.

### Autoregressive Stitching

Long trajectories are generated in overlapping windows of 20–40 timesteps. Between windows, the conditioning state updates from the generated trajectory. XY displacement uses a trapezoidal velocity profile; Z-height uses a raised-cosine bell.

## Configuration

All settings in `config/config.yaml`. Machine-local paths (data dir, MuJoCo XML, checkpoint dir) in `config/paths.yaml`. Inference defaults in `config/inference.yaml` — CLI flags override YAML values.

Key config sections: `data`, `model` (`backbone_type: dit1d`), `training`, `noise_scheduler`, `style_conditioning`, `inbetweening`.

## Data Format

`.npz` files with keys `root_pos, root_rot, dof_pos, object_pos, object_rot`, shape `(n_batch, time, feature_dim)`. Configure path in `config/paths.yaml::paths.train_path`.

**Output trajectory** from `generate_trajectory()`: `(B, T, 43)` — columns 0–35 robot state (pelvis XYZ + quat + 29 joints), columns 36–42 object state (XYZ + quat).

## Key Files for Common Tasks

- **Change model architecture:** `diffusion_forcing_transformer/dit1d.py`, `diffusion_forcing_transformer/dit_blocks.py`
- **Change diffusion logic:** `diffusion_forcing_transformer/discrete_diffusion.py`
- **Change conditioning/sampling:** `models/dfot_trajectory.py`
- **Change feature layout:** `utils/math/sbto_utils.py` + `config/config.yaml`
- **Add new physics constraints:** `utils/physics_limits.py`, auxiliary loss section in `models/dfot_trajectory.py`

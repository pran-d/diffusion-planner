
---

## Step-by-Step Plan for Your Coding Agent

### Phase 1: Infrastructure — Inference Config System

**Step 1: Create `inference_configs/` directory with base + ablation YAML files**

Create `inference_configs/base.yaml` with all shared defaults (epoch, device, stitch_steps, cfg_w, etc.), then create per-ablation files like `ddim_10steps.yaml`, `ddim_5steps.yaml`, `cfg_w_1.5.yaml`, etc. Each ablation YAML only needs to override the fields it changes — implement simple dict merging (`base → ablation`). Fields to expose:
- `inference_timesteps` (maps to `noise_scheduler.inference_timesteps`)
- `cfg_w`
- `stitch_steps`
- `use_last_frame_wp`
- `arrival_ratio`, `lift_height`, `no_lower_dist`
- `enable_goal_stop`, `end_error_threshold`
- `batch_size` (for the inference batch)
- `speedup_tricks` (dict of flags: `use_fp16`, `torch_compile`, `kv_cache`, `consistency_decoding`)

---

**Step 2: Create `run_inference_ablations.py`**

This is the main driver script. It should:
1. Accept `--config_dir inference_configs/` and `--output_dir results/ablations/`
2. Loop over all YAMLs in the config dir, merging each with base
3. For each config, run inference on a fixed set of `N` test samples (e.g. 20 trajectories sampled deterministically from the dataset with a fixed seed)
4. Save per-ablation outputs to `results/ablations/<ablation_name>/` containing: `trajectories.npy`, `goals.npy`, `metrics.json`, and plots
5. After all ablations, run `compare_ablations.py` automatically
6. Print a timing table to stdout and save as `results/ablations/timing_summary.csv`

The fixed test set is critical — seed the sample selection and save the sample indices to `results/ablations/test_indices.npy` on the first run, then reuse them for all ablations.

---

**Step 3: Create `compare_ablations.py`**

Reads all `metrics.json` files from subdirs and produces:
- A comparison table (CSV + printed): L2 goal error mean/std, success rate (% within threshold), wall-clock time per trajectory, GPU memory peak
- A multi-panel plot: for each ablation, overlay the object XY trajectories in ego frame (reuse your existing `to_ego_frame` logic from `visualize_param_sweep.py`)
- A bar chart of mean L2 error per ablation
- Save everything to `results/ablations/comparison/`

---

### Phase 2: Metrics Collection Inside Inference

**Step 4: Instrument `MotionGenerator.generate_trajectory()` with timing hooks**

Add a `TimingContext` class that records wall-clock time for: (a) model forward pass per step, (b) reconstruction per step, (c) total per trajectory. Store these in a `timing_stats` dict returned alongside the trajectory. This should be off by default (`verbose_timing=False`) and enabled by the ablation runner.

Also record peak GPU memory with `torch.cuda.max_memory_allocated()` before and after.

---

**Step 5: Add a standard `compute_metrics()` function to a new `utils/eval_metrics.py`**

It should take `(full_trajectory, goal_world, ref_obj_pos)` and return:
- `l2_final_error`: distance between final object XY and goal XY
- `success_at_0.05m`, `success_at_0.10m`, `success_at_0.15m`
- `traj_smoothness`: mean second-difference norm of robot pelvis XY (proxy for jerkiness)
- `obj_lift_peak_z`: max object z achieved (pick-and-place quality)
- `floor_violations`: count of frames where robot/object z < -0.05m
- `inference_time_s`
- `gpu_peak_mb`

---

### Phase 3: Inference Speedup Tricks

These are the specific tricks to implement, ordered by expected impact:

**Step 6: Fewer DDIM steps (already configurable, just needs ablation configs)**

Your `noise_scheduler.inference_timesteps` already controls this. Create ablation configs for 10 (current), 7, 5, and 3 steps. From the DreamZero paper, decoupled denoising schedules (their "DreamZero-Flash") show that fewer steps on the action head than the video head works well — the analog for you is testing whether 5 DDIM steps degrades quality.

---

**Step 7: Add `torch.compile` support to `DFoTTrajectory`**

In `RobotDiffuser.__init__`, add an optional `compile_model=False` flag. When enabled, wrap `self.model.diffusion_model.model` (the DiT1D backbone) with `torch.compile(mode="reduce-overhead")`. This is the single highest-impact system trick for transformer inference. Add this as a flag in the inference config. Important: put a warmup step before timing starts.

---

**Step 8: Batch CFG — you already have `_cfg_predict` doing batched CFG, make sure it's always used**

Audit `_sample_step_with_indicator` — it currently does **two separate forward passes** (cond + uncond) instead of the batched CFG path in `_cfg_predict`. Fix it to use the same batched approach: concatenate cond + uncond into a single `2B` batch, single forward, then split. This is a 2× speedup on the model forward when CFG is active, with zero quality change.

---

**Step 9: Add `torch.inference_mode()` context and `float16` mixed precision at inference**

In `getSample()` and `generate_trajectory()`, wrap the entire sampling loop with `torch.inference_mode()` (stronger than `no_grad` — disables view tracking). For fp16: add a `use_fp16` flag that casts the model with `model.half()` before inference and restores after. Your DiT1D with AdaLN should be fp16-safe. Add this as a config option.

---

**Step 10: Implement consistency-model-style one-step distillation config (optional, high effort)**

Based on DreamZero-Flash's insight about decoupled schedules: add an ablation config that uses `inference_timesteps=1` with DDIM eta=0 (your `ddim_sampling_eta` is already 0). This is essentially a direct x0 prediction in one step. Quality will drop but it's useful to measure the floor.

---

**Step 11: Add KV-cache-style optimization for the autoregressive stitching loop**

In your `_sample_sequence`, the DiT1D re-processes all T tokens from scratch each DDIM step. The DreamZero paper calls out KV caching as a major win. For your architecture: since the conditioning `c` (noise level embedding + external cond) changes each DDIM step but the structural pattern is the same, you can cache the `k` and `v` projections from attention for the **first keyframe token** (t=0) which is pinned by the waypoint mask and never changes content. Implement this as an optional `use_kv_cache` flag in `DiT1D.forward` — on the first DDIM step, compute and store the key/value tensors for the t=0 token; on subsequent steps, substitute the cached values. This is moderate effort but meaningful for window_size > 10.

---

### Phase 4: Cluster Setup

**Step 12: Create `scripts/submit_ablations.sh`**

A SLURM script that:
1. Submits one job per ablation config using `--array` or separate `sbatch` calls
2. Each job calls `run_inference_ablations.py --single_config <yaml>` (add this mode)
3. Uses a shared `results/ablations/` output directory on network storage
4. After all jobs finish, a final job calls `compare_ablations.py`

Structure:
```bash
sbatch --job-name=abl_ddim5 --gres=gpu:1 --mem=16G \
  run_single_ablation.sh inference_configs/ddim_5steps.yaml
```

---

**Step 13: Create `run_single_ablation.sh`**

Shell wrapper that activates your venv, sets `CUDA_VISIBLE_DEVICES`, and calls:
```bash
python run_inference_ablations.py \
  --config inference_configs/$1 \
  --test_indices results/ablations/test_indices.npy \
  --output_dir results/ablations/$(basename $1 .yaml)
```

---

### Phase 5: Trajectory Plots

**Step 14: Add `save_ablation_plots()` to `compare_ablations.py`**

For each ablation, generate and save:
1. **XY trajectory plot** (N trajectories, ego frame, with goal markers) — reuse `plot_trajectories()` from `visualize_param_sweep.py`
2. **Object Z over time** — plot the lift profile for each trajectory, useful to verify pick-and-place quality isn't degraded by speedup tricks
3. **Per-feature time series** for a single representative trajectory — reuse `plot_feature_space()` from `analyze_trajectory.py`
4. A **summary grid PNG** combining all three per ablation using matplotlib subplots, saved as `<ablation_name>/summary.png`

---

### Summary of Files to Create/Modify

| File | Action |
|---|---|
| `inference_configs/base.yaml` | Create |
| `inference_configs/ddim_10steps.yaml` | Create |
| `inference_configs/ddim_5steps.yaml` | Create |
| `inference_configs/cfg_w_1.5.yaml` | Create |
| `inference_configs/fp16_compile.yaml` | Create |
| `inference_configs/one_step.yaml` | Create |
| `run_inference_ablations.py` | Create |
| `compare_ablations.py` | Create |
| `utils/eval_metrics.py` | Create |
| `scripts/submit_ablations.sh` | Create |
| `scripts/run_single_ablation.sh` | Create |
| `models/dfot_trajectory.py` | Modify — fix batched CFG in `_sample_step_with_indicator`, add optional kv-cache |
| `diffusion_forcing_transformer/dit1d.py` | Modify — add `torch.compile` hook |
| `models/model.py` | Modify — add `use_fp16`, `compile_model` flags to `getSample` |
| `motion_generator.py` | Modify — add timing instrumentation, `verbose_timing` flag |

---

### Key Implementation Notes for the Agent

- The **batched CFG fix in `_sample_step_with_indicator`** (Step 8) is the most impactful pure-code fix with zero risk — the existing `_cfg_predict` method already has the right logic, `_sample_step_with_indicator` just isn't using it.
- The **fixed test set** (Step 2) is the most important correctness requirement — without it, ablation comparisons are meaningless.
- For `torch.compile`, add a try/except around it since it requires PyTorch 2.0+ and may fail gracefully on older cluster nodes.
- The DreamZero paper's biggest applicable insight for your architecture is: **fewer DDIM steps hurt less than you'd expect** because v-prediction with zero terminal SNR (which you have) produces well-conditioned denoising trajectories. So the 5-step ablation is likely to be nearly as good as 10-step.
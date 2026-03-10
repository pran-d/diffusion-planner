Changes: local-working-branch vs debug/pipeline
1. Model Architecture
dit1d.py

Removed attn_drop and proj_drop dropout parameters from transformer blocks (also dropped from config.yaml and dfot_trajectory.py construction)
dfot_trajectory.py — CFG / inference path

Removed batched-CFG optimization: debug/pipeline had a fast path — when cfg_w == 1.0, skipped the unconditional pass entirely; otherwise batched cond+uncond in a single forward pass. local-working-branch always runs two separate sequential passes
The unconditional masking sign was also corrected: debug/pipeline used (~external_cond_mask).float() * external_cond (kept non-task dims); local-working-branch uses external_cond_mask * external_cond (zeroes out task dims for uncond)
Dynamic thresholding (cfg_w > 1.0 quantile clamp) moved out of the fast-path block and is now always applied unconditionally in both branches
dfot_trajectory.py — denoising loop

external_conditions_mask construction and _do_sample_step helper were hoisted out of the loop in debug/pipeline (pre-computed once); local-working-branch recomputes them inside every denoising step (m loop)
The mask bit for task dims was inverted: debug/pipeline set [..., -self.task_dim:] = 1; local-working-branch sets [..., :self.task_dim:] = 1 — these index from opposite ends of the external-cond vector
The entire denoising loop was wrapped in torch.no_grad() in debug/pipeline; local-working-branch has no such guard
inpaint feature-map lookup (get_feature_indices) was pre-computed before the loop in debug/pipeline; local-working-branch calls it inside every step
dfot_trajectory.py — inpainting reconstruction

debug/pipeline commented out the delta_xy object XY inpainting path; local-working-branch re-enables it in sbto_utils.py
2. Training
train.py

Added seed_everything() with --seed 42 argument; seeds random, numpy, torch, torch.cuda, and sets cudnn.deterministic = True / benchmark = False
Removed weight_decay=0.01 from AdamW optimizer (now uses PyTorch default)
EMA decay and update_after_step are now read from training_cfg (configurable); were hardcoded 0.9995 / 0 before
Added seeded torch.Generator for DataLoader (both WeightedRandomSampler and shuffle=True path)
Added best_loss-based best model tracking → saves model_best.pth + ema_model_best.pth every time loss improves
EMA shadow params saved as a separate ema_model_{epoch}.pth file on every checkpoint save
Checkpoint dict now includes best_loss, epoch, and seed fields
Removed check_physical_consistency() generative eval + --eval_every / --num_eval_samples args and best_phys_violations tracking (replaced by loss-based best model tracking)
config.yaml

start_timestep: 0 ← 5; end_timestep: -1 removed
State conditioning noise levels ~2.5× larger: joints 0.025 ← 0.01, body_rot6d 0.005 ← 0.0025, obj_rel_pos 0.005 ← 0.0025, obj_rel_rot6d 0.005 ← 0.0025, task_params 0.025 ← 0.01
resample_steps: 2 ← 5 (fewer RePaint re-noising iterations per denoising step)
arrival_ratio (inbetweening): 0.85 ← 0.80
deterministic_inference: True ← False
num_epochs: 750 ← 5000; save_every: 50 ← 10
Removed attn_drop: 0.0, proj_drop: 0.0
motion_generator.py — training loop

Removed batches_per_epoch / max_batches early-stopping logic (now commented out)
Removed _base_norm_stats caching across multiple fit() calls; normalization stats are now always recomputed or loaded fresh on each fit() call
Added task_params argument to fit() for external goal injection into dataset
Scheduler load_state_dict re-enabled on resume (was commented out)
3. Data Pipeline
flexible_dataset.py

Removed end_timestep config support
Removed real_traj_lengths per-trajectory real-length tracking; window index upper bound is now always T_padded - window_size (no per-sample restriction)
Removed _get_single_traj trimming to real_T before padding (was trimming to actual non-padded length)
Removed feature_dims property
Simplified task_magnitude noise: debug/pipeline used symmetric uniform [1−mag, 1+mag] multiplicative jitter; local-working-branch uses torch.rand * mag (one-sided, smaller range)
Added task_params constructor argument for external goal injection
4. Inference / Generation Pipeline
inference.py — major rewrite

debug/pipeline delegated generation to MotionGenerator wrapper; local-working-branch uses RobotDiffuser.getSample() directly with its own autoregressive stitching loop
Added --ema flag (load EMA weights for inference) and --enable_phys_stop (physical-consistency check, ported from MotionGenerator)
Removed --analysis_path arg; analysis data is now always saved as <save_path>_analysis.npz
DEFAULT_LIFT_HEIGHT: 0.5 m ← 0.62 m
Z-profile logic: debug/pipeline used time-based lower ramp (lower_start / lower_end); local-working-branch uses distance-based lowering (no_lower_dist threshold) — lift-only with hold until close to goal
XY displacement scheduling: debug/pipeline used 1/remaining_steps (uniform equal split); local-working-branch uses a trapezoidal velocity profile (accel → cruise → decel)
Default waypoint parameters changed: arrival_ratio 0.85 ← 0.70, lift_start 0.0 ← 0.10, lift_end 0.20 ← 0.40, walk_start_z 0.80 ← 0.25, no_lower_dist 0.5 ← 0.75
motion_generator.py — generate_trajectory()

stitch_steps formula: ceil(target_traj_length / (window_size-1)) ← ceil((target_traj_length-1) / (window_size-1)) — slightly more conservative step count
Always skips t=0 anchor frame (removed if step > 0: guard that previously kept t=0 frames on the first window)
Trajectory smoothing sigma: 2.0 ← 1.25
Removed return_analysis parameter and all per-window analysis export logic
Goal-stop condition: removed _robot_z > 0.71 pelvis-height guard (goal now considered reached by XY error alone)
Removed print(task_cond_np) debug print
Default waypoint parameters updated to match new inference.py defaults
sbto_utils.py

compute_task_params: max_goal_dist parameter default changed from 1.0 (explicit) to None → distance is now clipped to 1.0 unconditionally (hardcoded) rather than via the parameter; length check changed from shape[-1] >= 7 to len(...) >= 7
reconstruct_sbto_trajectory: inpaint delta-XY object position re-encoding block was commented out in debug/pipeline; local-working-branch re-activates it





1) Adding physical losses => smoother
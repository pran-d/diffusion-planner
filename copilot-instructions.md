## Step-by-Step Plan for Your Coding Agent

### Overview
This plan breaks into three major tracks that build on each other:
1. **Intentional Inbetweening Masking:** Implement semantic schemas for targeted feature generation.
2. **Intentional State Conditioning Masking:** Design structured dropout to enable locomotion extraction from loco-manipulation data.
3. **Two-Phase Motion Generation:** Implement per-feature noise levels with a designed hierarchical schedule.

---

### Track 1: Intentional Inbetweening Masking

**1. Structured waypoint schema with semantic intent**
**Target File:** `diffusion_forcing_transformer/dfot_trajectory.py`

Currently, `_generate_waypoint_mask` generates random keyframes and random partial feature groups. Replace this with a semantic waypoint schema that explicitly targets specific generation behaviors.

* Add a new config section `inbetweening.waypoint_schemas` listing named schemas (e.g., "box_only", "locomotion_only", "full") with per-schema probabilities.
* Configure each schema to specify which feature groups are pinned at which frames (first, last, intermediate) and how much noise to add to pinned features.
* **"box_only" schema:** Pin `obj_delta_xy` and `obj_z` at the last frame, leaving all robot features unpinned to force the model to figure out robot motion given a box goal.
* **"locomotion_only" schema:** Leave robot features (`delta_xy`, `delta_yaw`, `body_z`, `joints`) *unpinned* (fully noisy) so the model must generate them. Completely mask/ignore all object features.
* Sample a schema per batch element to allow mixed schemas in a single training batch.
* **Files to modify:** `dfot_trajectory.py` (`_generate_waypoint_mask`), `config/config.yaml` (add `waypoint_schemas`).

**2. Noise-aware partial waypoint injection for box features**
**Target File:** `diffusion_forcing_transformer/dfot_trajectory.py`

The current `_inject_waypoints` uses the same noise level for all partially-known features. Separate the noise level for object features versus robot features.

* Use a lower noise fraction for `obj_delta_xy` and `obj_z` pinned at the last frame (e.g., `waypoint_noise_fraction * 0.5`) to provide a stronger box goal signal.
* Keep the current noise fraction for robot features at intermediate keyframes.
* Add `per_group_noise_fractions` to the config under `inbetweening.partial_masking`.
* **Files to modify:** `dfot_trajectory.py` (`_inject_waypoints`, `_generate_waypoint_mask`), `config/config.yaml`.

---

### Track 2: Intentional State Conditioning Masking

**1. Structured state conditioning dropout by semantic group**
**Target Files:** `models/dfot_trajectory.py`, `datasets/flexible_dataset.py`

State conditioning dropout is currently applied as a single Bernoulli gate over the entire state vector. Replace this with per-group structured dropout.

* Leverage the observation vector layout: `joints`(29) | `body_z`(1) | `body_rot6d`(6) | `obj_rel_pos`(3) | `obj_rel_rot6d`(6) = 45 dims.
* Define conditioning groups: "robot_pose" (`body_z` + `body_rot6d`), "robot_joints" (`joints`), and "object_pose" (`obj_rel_pos` + `obj_rel_rot6d`).
* Update `DFoTTrajectory.forward` to apply independent Bernoulli dropout per group using probabilities from `training.state_conditioning_masking.group_drop_prob`.
* Implement this as an `_apply_structured_state_dropout(state_cond, dropout_probs)` method that masks contiguous slices of the observation vector.
* Wire this into the `apply_condition_dropout` call inside `forward`.
* **Files to modify:** `dfot_trajectory.py`, `config/config.yaml` (extend `state_conditioning_masking`).

**2. Locomotion extraction via object group masking**
**Target File:** `models/dfot_trajectory.py`

To extract pure locomotion trajectories from loco-manipulation data, the model must be trained to generate robot motion without relying on object state conditioning.

* Add a `locomotion_mode` inference flag (passed via `sample()`) that zeros out the `object_pose` group in the state conditioning vector.
* Add a corresponding training regime: with probability `locomotion_dropout_prob` (config, default 0.3), mask the entire `object_pose` group in state conditioning AND simultaneously use the "locomotion_only" waypoint schema.
* At inference, ensure `getSample()` in `model.py` accepts `locomotion_mode=True` and passes it through to `DFoTTrajectory.sample()`.
* **Files to modify:** `dfot_trajectory.py`, `models/model.py`, `motion_generator.py` (`generate_trajectory`), `config/config.yaml`.

**3. Consistent masking between state conditioning and future trajectory**
**Target File:** `models/dfot_trajectory.py`

When object features are dropped in state conditioning (like in locomotion mode), the future trajectory prediction of object features must be deweighted in the loss so the model isn't penalized for failing to predict an unconditioned object.

* When the `object_pose` group is dropped from state conditioning, set the loss weight for `obj_rel_pos` and `obj_rel_rot6d` features in the future trajectory to `0.0` (or a very low `masked_obj_loss_scale`).
* Implement this via a per-feature loss weight tensor computed in `forward` before the final loss reduction (additive to the existing SNR-based loss weighting).
* **Files to modify:** `dfot_trajectory.py` (`forward`, `compute_auxiliary_losses`), `config/config.yaml`.

---

### Track 3: Two-Phase Motion Generation

**Step 3.1 — Per-feature noise level support in forward diffusion**
**Target File:** `diffusion_forcing_transformer/discrete_diffusion.py`

Extend `q_sample` and model predictions to support `(B,T,D)` noise levels for per-feature diffusion (currently assumes scalar or `(B,T)`).

* Modify `q_sample(x_start, k, noise)` to handle `k` of shape `(B,T,D)` by using `extract()` per-feature and indexing `sqrt_alphas_cumprod[k]` with the full shape.
* Apply similar modifications to `predict_start_from_noise`, `predict_noise_from_start`, `predict_v`, `predict_start_from_v`, and `predict_noise_from_v`.
* Add a helper `_expand_k_to_feature_dim(k, D)` that broadcasts `(B,T)` → `(B,T,D)` by repeating, for use when `k` lacks a feature dim.
* **Files to modify:** `discrete_diffusion.py`.

**Step 3.2 — Phase-split noise sampling during training**
**Target File:** `diffusion_forcing_transformer/dfot_trajectory.py`

Refine `_get_training_noise_levels` to make the hierarchical noise sampling principled instead of independent.

* Define Phase 1 features (planning): `delta_xy`, `delta_yaw`, `obj_delta_xy`, `obj_z`, `obj_rel_pos`, `obj_rel_rot6d`.
* Define Phase 2 features (execution): `joints`, `body_z`, `body_rot6d`.
* Sample Phase 1 noise from `[0, k_max_p1]` where `k_max_p1 = phase1_max_ratio * T_total` (default 0.5).
* Sample Phase 2 noise from `[k_min_p2, T_total]` where `k_min_p2 = phase2_min_ratio * T_total` (default 0.0).
* Produce a single `(B,T,D)` noise level tensor by assembling phase 1 and phase 2 noise per feature.
* **Files to modify:** `dfot_trajectory.py` (`_get_training_noise_levels`).

**Step 3.3 — Two-phase DDIM scheduling matrix at inference**
**Target File:** `diffusion_forcing_transformer/dfot_trajectory.py`

Fix `_generate_hierarchical_two_phase_scheduling_matrix` to properly integrate with the sampling loop.

* Ensure the function returns an `(M, B, T, D)` tensor after the repeat in `_sample_sequence`.
* Phase 1 features go from max noise → 0 in the first `phase1_end_ratio` fraction of denoising steps.
* Phase 2 features go from max noise → 0 over all denoising steps.
* Update `from_noise_levels` and `to_noise_levels` in `_sample_sequence` to carry the `D` dimension through to `sample_step` → `ddim_sample_step`.
* In `ddim_sample_step`, index `alpha` and `alpha_next` per-feature using `self.alphas_cumprod[k.clamp(min=0)]` with appropriate reshaping.
* **Files to modify:** `dfot_trajectory.py`, `discrete_diffusion.py` (`ddim_sample_step`, `sample_step`).

**Step 3.4 — Loss weighting for per-feature noise levels**
**Target File:** `diffusion_forcing_transformer/discrete_diffusion.py`

Extend `compute_loss_weights` to process `(B,T,D)` dimensions.

* When `k` has `ndim == 3`, compute SNR per-feature: `snr = self.alphas_cumprod[k] / (1 - self.alphas_cumprod[k])`.
* Apply the fused min-SNR cumulative decay over time along `dim=1` (time axis).
* Return a `(B,T,D)` weight tensor for element-wise loss multiplication in `dfot_trajectory.forward`.
* Add conditional logic: if `k.ndim == 2`, use the scalar path; if `k.ndim == 3`, use the per-feature path.
* **Files to modify:** `discrete_diffusion.py` (`compute_loss_weights`, `_build_buffer`).

**Step 3.5 — Config wiring and inference API**
**Target Files:** `config/config.yaml`, `motion_generator.py`, `inference_mg.py`

* In `config/config.yaml` under `noise_scheduler`, add hierarchical noise configurations.
* In `motion_generator.generate_trajectory`, add `two_phase_inference: bool = True` to set `scheduling_matrix = "hierarchical_two_phase"`.
* In `inference_mg.py`, add a `--two_phase` CLI flag.
* Add a phase_debug visualization to log feature noise levels per step.
* **Files to modify:** `config/config.yaml`, `motion_generator.py`, `inference_mg.py`.

---

### Execution Order for the Copilot Agent

1.  **Step 3.1** — Foundational shape change in `discrete_diffusion.py` (all other steps depend on it).
2.  **Step 3.4** — Loss weighting update (needed before training changes).
3.  **Step 3.2** — Training noise sampling (uses the new shapes).
4.  **Step 3.3** — Inference scheduling matrix.
5.  **Step 3.5** — Config and API wiring.
6.  **Step 1.1** — Semantic waypoint schemas (independent track, no dependencies on Track 3).
7.  **Step 1.2** — Noise-aware injection (depends on 1.1 for schema structure).
8.  **Step 2.1** — Structured state dropout (independent, but logically follows Track 1).
9.  **Step 2.2** — Locomotion extraction mode (depends on 2.1 and 1.1).
10. **Step 2.3** — Consistent loss masking (depends on 2.1 and 2.2).
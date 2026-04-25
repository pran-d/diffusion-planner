## Plan: Two-Phase Kimodo-Style Planner

Add a hierarchical pipeline with **separate models** where `Phase1` predicts coarse task-relevant state and `Phase2` predicts fine robot pose details. This minimizes high-DOF search early, then conditions detailed kinematics on a stable coarse plan. The plan below scopes representation split, model/data interfaces, and training/inference orchestration with minimal disruption to your current diffusion stack.

### Fixed Feature Split
**Phase 1 features**
- `delta_xy`
- `delta_yaw`
- `body_z`
- `obj_rel_pos`
- `obj_rel_rot6d`

**Phase 2 features**
- `joints`
- `body_rot6d`

### Steps
1. Define state split and feature groups in [config/feature_labels.yml](../config/feature_labels.yml), [config/config.yaml](../config/config.yaml), and [config/inference.yaml](../config/inference.yaml) using explicit `phase1_features` and `phase2_features`, fixed to the feature lists above.
2. Add dataset view builders in [datasets/flexible_dataset.py](../datasets/flexible_dataset.py) and [datasets/conditional_dataset.py](../datasets/conditional_dataset.py) to emit phase-specific tensors plus `phase1_context` conditioning for `Phase2`.
3. Introduce two **separate model paths** in [models/model.py](../models/model.py), [models/dfot_trajectory.py](../models/dfot_trajectory.py), and [diffusion_forcing_transformer/base_backbone.py](../diffusion_forcing_transformer/base_backbone.py) with clear `forward_phase1()` / `forward_phase2()` interfaces and independent checkpoints.
4. Wire staged training/inference orchestration in [train.py](../train.py), [inference_mg.py](../inference_mg.py), and [motion_generator.py](../motion_generator.py): run `Phase1`, freeze/sample coarse trajectory, then condition `Phase2` rollout.
5. Add phase-aware masking/guidance and losses in [diffusion_forcing_transformer/guidance_functions.py](../diffusion_forcing_transformer/guidance_functions.py), [run_ablations.py](../run_ablations.py), and [evaluate_ablations.py](../evaluate_ablations.py) to compare single-phase vs two-phase behavior.

### Further Considerations
1. Model strategy is fixed to **separate models** (`Phase1Model`, `Phase2Model`) with separate optimizer/scheduler/checkpoint configs.
2. `Phase2` orientation representation is fixed to `body_rot6d` (no yaw-removed quaternion/axis-angle variants).
3. Choose training schedule: Option A: sequential (`Phase1` then `Phase2`), Option B: joint multi-task, Option C: sequential warm-start then joint fine-tune.

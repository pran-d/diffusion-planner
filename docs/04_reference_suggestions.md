# Reference Paper Analysis & Feature Suggestions

## Overview

This document analyzes two reference papers and identifies features that could improve the current diffusion planner:

1. **PARC** — *Physics-Augmented Robot Choreographer* (terrain-conditioned locomotion diffusion)
2. **PLT** — *Part-Wise Latent Tokens as Adaptable Motion Priors for Physically Simulated Characters* (SIGGRAPH 2025)

---

## Reference 1: PARC (Physics-Augmented Robot Choreographer)

### Paper Summary

PARC generates physically plausible locomotion for legged robots on varied terrains using a **Transformer encoder diffusion model** conditioned on terrain heightmaps. Its key innovation is an **iterative physics augmentation loop**: generate → simulate → fix → retrain.

### Architecture & Method

- **Model**: Transformer encoder (not decoder) diffusion model operating on motion tokens
- **Conditioning**: Terrain heightmaps encoded via a CNN and injected via cross-attention
- **Blended denoising**: Autoregressive generation with temporal overlap. Uses `s=0.65` blend factor — the overlap region merges 65% from the previous window's output and 35% from the current window's prediction
- **Conditioning frames**: 2 frames from the previous window condition the next window (as opposed to your single-frame state condition)
- **Training losses**:
  - Standard diffusion MSE loss
  - **Geometric losses**: foot contact consistency, foot skating penalty, body orientation smoothness
  - **Physics augmentation**: MuJoCo simulation identifies failures (falls, penetrations), kinematic corrections are applied, and corrected trajectories are added back to the training set
- **Iterative augmentation loop**: Train → Generate → Simulate → Correct failures → Add corrected data → Retrain. This loop runs multiple times, progressively improving physical plausibility

### Key Innovations

1. **Terrain heightmap conditioning** via CNN encoder + cross-attention
2. **Blended denoising** for seamless autoregressive windows
3. **Geometric auxiliary losses** during training (foot contact, skating, smoothness)
4. **Iterative physics augmentation** — self-improving data loop
5. **Kinematic correction** of physically infeasible outputs before retraining

---

## Reference 2: PLT (Part-Wise Latent Tokens)

### Paper Summary

PLT decomposes motion priors into **part-specific discrete codebooks** (e.g., upper body, lower body, arms, legs, trunk). A hierarchical control pipeline learns motion skills per body part that can be independently extended, combined, and adapted for new scenarios.

### Architecture & Method

- **Part-wise codebooks**: Body is split into K groups (K=2 for upper/lower, K=5 for trunk + 2 arms + 2 legs). Each group has its own VQ codebook and low-level policy
- **Encoder**: Outputs continuous latent vector y, split into K segments, each quantized via VQ to nearest codebook token
- **Refinement network**: Adds continuous offset Δz to discrete tokens, ensuring coherence across body parts. Regularized by L2 loss to keep refinements small
- **Two-phase training**:
  1. *Imitation learning*: Train encoder + codebooks + low-level policies via online distillation from expert
  2. *Task learning*: Freeze codebooks & low-level policies, train high-level policy with PPO on multi-discrete action space
- **Part-wise adaptation**: Update only the codebook and low-level policy for a specific body part when new motion data arrives — no catastrophic forgetting of other parts
- **Results**: PLT-5 (5 parts) consistently outperforms PLT-2 (2 parts) and baselines (PULSE, NCP) on imitation quality, N-body tracking, and navigation tasks

### Key Innovations

1. **Structured body decomposition** into independent part codebooks
2. **Combinatorial generalization**: Novel motions emerge from combining part-wise tokens
3. **Refinement network** for cross-part coherence (discrete + continuous)
4. **Part-wise incremental adaptation** without forgetting
5. **Multi-discrete action space** for efficient high-level policy training

---

## Feature Suggestions for the Current System

### High Priority — Directly Applicable

#### 1. Blended Denoising for Window Stitching (from PARC)

**Current approach**: Your autoregressive stitching uses hard waypoint injection at frame boundaries. The transition between windows relies on `update_condition()` re-computing SBTO from the last generated frame.

**Suggested improvement**: Adopt PARC's blended denoising with overlap:
- Generate each window with `n_overlap` frames that overlap with the previous window
- Blend predictions: `x_overlap = s * x_prev + (1-s) * x_curr` with `s ≈ 0.65`
- This should produce **smoother transitions** at window boundaries and reduce discontinuity artifacts

**Implementation effort**: Medium. Modify `_sample_sequence()` and the inference loop to maintain overlap buffers and apply blending.

#### 2. Geometric Auxiliary Losses (from PARC)

**Current approach**: Your training uses only the v-prediction MSE loss with fused_min_snr weighting.

**Suggested improvement**: Add geometric consistency losses:
- **Foot/end-effector contact loss**: Penalize foot sliding when the robot should be in ground contact (detectable from `body_z` and joint kinematics via forward kinematics)
- **Temporal smoothness loss**: Penalize jerk (third derivative) in delta_xy, delta_yaw, and joint trajectories
- **Object grasping consistency**: When `obj_rel_pos` indicates the object is grasped (close to hand), penalize relative drift

**Implementation effort**: Medium. Requires forward kinematics computation (you already have `precompute_fk.py` in scripts/) and adding loss terms to the training step. The losses operate on the predicted x₀, not on v directly.

#### 3. Multi-Frame Conditioning (from PARC)

**Current approach**: You condition on a single history frame (`state_history: 1`), providing a 45-dim snapshot.

**Suggested improvement**: Increase to 2–3 conditioning frames to provide velocity/acceleration context:
- Encode 2–3 past frames (potentially with a small temporal encoder or by concatenating)
- This gives the model implicit velocity information, improving prediction of dynamic motions
- PARC uses 2 conditioning frames and finds it critical for maintaining momentum

**Implementation effort**: Low-Medium. Increase `state_history`, modify `state_embedding` to handle multi-frame input (e.g., via a 1D conv or flattened MLP), and adjust the dataset to return more history.

#### 4. Feature-Group-Aware Architecture (inspired by PLT)

**Current approach**: Your waypoint indicator projection treats the 51-dim mask uniformly. The partial masking config already defines feature groups (locomotion, pick_place, etc.).

**Suggested improvement**: Make the architecture explicitly aware of feature groups:
- Use **group-wise embedding layers** in the input embedder (separate linear projections for each feature group, then concatenate)
- This allows the model to learn group-specific representations, analogous to PLT's part-wise codebooks but within the diffusion framework
- Could also add **cross-group attention** within DiT blocks — features from one group attend to features from other groups, improving coordination

**Implementation effort**: Medium. Modify `input_embedder` in DiT1D and potentially add feature-group tokens.

### Medium Priority — Valuable Extensions

#### 5. Iterative Physics Augmentation (from PARC)

**Current approach**: Training data comes from demonstrations or RL rollouts. No self-correction loop.

**Suggested improvement**: Implement PARC's augmentation loop:
1. Train diffusion model on current dataset
2. Generate trajectories for diverse conditions
3. Run MuJoCo simulation on generated trajectories
4. Identify failures (falls, penetrations, object drops)
5. Apply kinematic corrections to failed trajectories
6. Add corrected trajectories to training set
7. Retrain (repeat 2–3 iterations)

**Benefits**: Dramatically improves physical plausibility without requiring more demonstration data. Your codebase already has MuJoCo integration (`mj_model.xml`), making this feasible.

**Implementation effort**: High. Requires MuJoCo rollout pipeline, failure detection, kinematic correction module, and data augmentation integration.

#### 6. Refinement Network for Cross-Feature Coherence (from PLT)

**Current approach**: The diffusion model generates all 51 features jointly. Cross-feature coherence is only implicitly learned.

**Suggested improvement**: Add a lightweight refinement network that post-processes the denoised output:
- Input: raw diffusion output `(T, 51)`
- Output: refined trajectory with continuous offsets
- Regularized by L2 norm (keep refinements small, as in PLT)
- Specifically targets: joint-body consistency, object-hand coordination, locomotion-manipulation coupling

**Implementation effort**: Medium. Add a small MLP or 1D conv network after the final denoising step in `_sample_sequence()`.

#### 7. Terrain/Scene Conditioning (from PARC)

**Current approach**: Only goal position is used as task conditioning. No scene/terrain awareness.

**Suggested improvement**: Add scene context conditioning:
- Encode local terrain heightmap or obstacle map via CNN
- Inject via cross-attention in DiT blocks (add cross-attention layers) or concatenate with external condition
- Enables planning around obstacles, stairs, uneven ground

**Implementation effort**: High. Requires terrain data pipeline, CNN encoder, and modifying DiT blocks to support cross-attention.

### Lower Priority — Future Directions

#### 8. Part-Wise Adaptation Pipeline (from PLT)

**Relevance**: If you need to update the planner for new manipulation skills (e.g., new object shapes) without forgetting locomotion:
- Freeze locomotion-related parts of the model
- Fine-tune only manipulation-related weights
- PLT's approach of per-part codebooks doesn't directly apply to diffusion, but the concept of **feature-group-specific fine-tuning** does

**Adaptation for diffusion**: Use LoRA or adapter layers that target specific feature groups. Train new adapters for new tasks while keeping base model frozen.

#### 9. Discrete Latent Tokens for Motion Primitives (from PLT)

**Relevance**: PLT's VQ codebooks capture discrete motion primitives (walk, reach, etc.).

**Adaptation**: Add a VQ-VAE bottleneck within the DiT architecture:
- Encoder maps trajectory windows to discrete tokens
- Diffusion operates in the latent token space
- Decoder maps back to trajectory features
- Part-wise tokenization (separate codebooks for locomotion vs manipulation features)

This would create a **latent diffusion** model for trajectories (you already have a `latent_diffusion.py` stub in models/).

#### 10. Dynamic Thresholding Improvements (from PARC insights)

**Current approach**: You have 99.5th percentile dynamic thresholding for CFG.

**Suggested improvement**: PARC implicitly handles this through their physics loop. An alternative: **physics-informed thresholding** — instead of purely statistical clamping, use known physical limits:
- Clamp `delta_xy` based on max robot velocity
- Clamp `body_z` based on robot height constraints
- Clamp `joints` based on joint limits
- This provides tighter, more meaningful bounds than percentile-based thresholding

---

## Implementation Roadmap

### Phase 1 (Quick Wins — 1-2 weeks)

1. **Multi-frame conditioning** (#3): Increase `state_history` to 2–3, modify state embedding
2. **Temporal smoothness loss** (part of #2): Add jerk penalty on predicted x₀
3. **Physics-informed thresholding** (#10): Replace percentile thresholding with kinematic limits

### Phase 2 (Core Improvements — 2-4 weeks)

4. **Blended denoising** (#1): Implement overlap-based window stitching
5. **Geometric losses** (#2): Foot contact, object grasping consistency
6. **Feature-group embeddings** (#4): Group-wise input projection in DiT1D

### Phase 3 (Advanced — 1-2 months)

7. **Refinement network** (#6): Post-diffusion refinement MLP
8. **Iterative physics augmentation** (#5): MuJoCo-in-the-loop training
9. **Terrain conditioning** (#7): CNN + cross-attention for scene awareness

### Phase 4 (Research — 2+ months)

10. **Part-wise adaptation with LoRA** (#8)
11. **Latent diffusion with VQ tokens** (#9)

---

## Comparison Table

| Feature                         | PARC | PLT | Current System | Priority |
|---------------------------------|------|-----|----------------|----------|
| Diffusion backbone              | Transformer Encoder | N/A (RL) | DiT1D (✓) | — |
| Terrain conditioning            | ✓ CNN + cross-attn | — | ✗ | Medium |
| Blended denoising (stitching)   | ✓ s=0.65 | — | ✗ hard waypoints | **High** |
| Multi-frame conditioning        | ✓ 2 frames | — | 1 frame | **High** |
| Geometric losses                | ✓ foot contact, skating | — | ✗ | **High** |
| Physics augmentation loop       | ✓ iterative | — | ✗ | Medium |
| Part-wise body decomposition    | — | ✓ K codebooks | Feature groups (partial) | Medium |
| Refinement network              | — | ✓ continuous Δz | ✗ | Medium |
| Discrete motion tokens          | — | ✓ VQ codebooks | ✗ | Low |
| Part-wise adaptation            | — | ✓ incremental | ✗ | Low |
| Partial masking (feature-level) | — | — | ✓ | — |
| Waypoint-guided generation      | — | — | ✓ | — |
| V-prediction + fused_min_snr    | — | — | ✓ | — |
| RePaint resampling              | — | — | ✓ | — |

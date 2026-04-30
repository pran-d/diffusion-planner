# Model Architecture

## Overview

The model follows a three-layer hierarchy:

```
RobotDiffuser (interface)
  └── DFoTTrajectory (waypoint masking + conditioning logic)
        └── DiscreteDiffusion (diffusion process: noising, denoising, loss)
              └── DiT1D (1D Diffusion Transformer backbone)
```

---

## 1. Top-Level Interface: `RobotDiffuser`

**Location:** `models/model.py`

A thin wrapper that:
- Instantiates `DFoTTrajectory` based on config
- Provides `getSample()` for inference: constructs model conditions from current state + task params, then calls `model.sample()`
- Provides `loadWeights()` for checkpoint loading (with `strict=False` for backward compatibility)

### `getSample(current_state, task_params, waypoints, waypoint_mask)`

1. Embeds `current_state` and `task_params` as a tuple `model_cond = (state_cond, task_cond)`
2. Passes optional `waypoints` (clean values) and `waypoint_mask` (boolean mask) for waypoint-guided generation
3. Returns denoised trajectory of shape `(B, T, 51)`

---

## 2. DFoTTrajectory: Main Model

**Location:** `models/dfot_trajectory.py` (~1037 lines)

This class wraps `DiscreteDiffusion` and adds:
- Dual condition embedding (state + task)
- Waypoint mask generation (frame-level + feature-level)
- Waypoint injection with RePaint-style noising
- Modified loss computation (zeroing out fully-known frames)
- Sampling with waypoint inpainting

### 2.1 Condition Embeddings

Two separate MLP embeddings project the conditions into the hidden space:

| Embedding          | Input Dim | Output Dim | Description                                |
|--------------------|-----------|------------|--------------------------------------------|
| `state_embedding`  | 45        | 256        | Projects current robot observation          |
| `task_embedding`   | 3         | 256        | Projects [dir_x, dir_y, distance] goal vec  |

During forward pass:
1. **Condition dropout** for classifier-free guidance training:
   - State dropout: 10% (`state_cond_drop_prob: 0.1`)
   - Task dropout: 20% (`task_cond_drop_prob: 0.2`)
   - When dropped, the condition is replaced with zeros
2. **Concatenation**: `ext_cond = [state_emb, task_emb]` → shape `(B, 512)`

This 512-dim vector is the **external condition** passed into the DiT backbone.

### 2.2 Waypoint Indicator Projection

```python
self.waypoint_indicator_proj = nn.Sequential(
    nn.Linear(num_features, hidden_size),  # 51 → 256
    nn.SiLU(),
    nn.Linear(hidden_size, hidden_size),   # 256 → 256
)
# Zero-initialized (last layer weights and biases set to 0)
```

This MLP projects a per-frame binary mask (shape `(B, T, 51)`) into the hidden space. The output is **added to the hidden states** before the transformer processes them. Zero initialization ensures waypoint conditioning starts as a no-op and gradually learns its influence.

### 2.3 Noise Level Generation

For the `full_sequence` generation mode (default), noise levels are **uniform across all tokens** in a sequence:

```python
noise_levels = torch.rand(B, 1).expand(B, T)  # same k for all timesteps
```

This means every frame in a window gets the same diffusion noise level during training. Fully-known waypoint frames are overridden to `k=0` (clean), and partially-known waypoint frames get a random `k ∈ [0, max_wp_k]`.

### 2.4 Forward Pass (Training)

```
Input: x_0 (B, T, 51), model_cond = (state_cond, task_cond)
   │
   ▼
1. Embed conditions → ext_cond (B, 512)
   │
   ▼
2. Generate noise levels k ~ U(0, 1) for each batch → (B, T)
   │
   ▼
3. Generate waypoint_mask (B, T, 51) via _generate_waypoint_mask()
   - Frame-level: select 0-3 random keyframes
   - Feature-level: partial masking per feature group
   │
   ▼
4. Override noise levels for waypoint frames:
   - Fully known frames → k = 0
   - Partially known frames → k ~ U(0, max_wp_k)
   │
   ▼
5. q_sample: noise x_0 → x_k using noise schedule
   │
   ▼
6. Inject waypoints: overwrite known features with clean/noised values
   │
   ▼
7. Forward through backbone with waypoint indicator:
   - Compute indicator_emb = waypoint_indicator_proj(waypoint_mask.float())
   - hidden = input_embedder(x_k) + pos_emb + indicator_emb
   - Pass through transformer blocks
   │
   ▼
8. Predict v (velocity parameterization)
   │
   ▼
9. Compute MSE loss with fused_min_snr weighting
   - Zero out loss on fully-known waypoint frames
   │
   ▼
Output: weighted scalar loss
```

---

## 3. DiscreteDiffusion: Diffusion Process

**Location:** `diffusion_forcing_transformer/discrete_diffusion.py` (~592 lines)

Implements the core diffusion mechanics: forward noising, reverse denoising, loss computation, and sampling.

### 3.1 Noise Schedule

**Configuration:**
```yaml
noise_scheduler:
  type: sigmoid
  train_timesteps: 200
  inference_timesteps: 10
  beta_start: -3
  beta_end: 3
  v_prediction: true
  deterministic_inference: true
```

**Schedule type: Sigmoid**
```python
betas = torch.sigmoid(torch.linspace(beta_start, beta_end, train_timesteps))
```
After computing betas, **zero terminal SNR** is enforced:
- The final alpha_cumprod is set to 0 (pure noise at t=T)
- Betas are recomputed from the adjusted alpha_cumprod

**Derived quantities** (all precomputed as buffers):
- `alphas_cumprod`: cumulative product of (1 - beta)
- `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`
- `SNR = alphas_cumprod / (1 - alphas_cumprod)`
- `posterior_mean_coef1`, `posterior_mean_coef2` for DDPM posterior

### 3.2 V-Prediction Parameterization

Instead of predicting noise (ε) or the clean signal (x₀), the model predicts **v**:

$$v = \sqrt{\bar{\alpha}_t} \cdot \epsilon - \sqrt{1 - \bar{\alpha}_t} \cdot x_0$$

From v, both x₀ and ε can be recovered:

$$\hat{x}_0 = \sqrt{\bar{\alpha}_t} \cdot x_t - \sqrt{1 - \bar{\alpha}_t} \cdot v$$

$$\hat{\epsilon} = \sqrt{1 - \bar{\alpha}_t} \cdot x_t + \sqrt{\bar{\alpha}_t} \cdot v$$

V-prediction is preferred because it provides more balanced gradients across noise levels and works better with zero terminal SNR.

### 3.3 Forward Process (q_sample)

```python
def q_sample(x_0, noise_level_indices, noise=None):
    """
    x_t = sqrt(alpha_cumprod[t]) * x_0 + sqrt(1 - alpha_cumprod[t]) * noise
    """
```

`noise_level_indices` are integer indices into the 200-step schedule, per-token.

### 3.4 Loss Weighting: Fused Min-SNR

The loss uses a **fused min-SNR** weighting strategy that combines forward and reverse cumulative SNR:

```python
# Forward cumulative SNR (from t=0 to t)
cumsnr_fwd = cumsum(SNR)
# Reverse cumulative SNR (from t=T to t)
cumsnr_bwd = reverse_cumsum(SNR)
# Fused weight
weight[t] = min(cumsnr_fwd[t], cumsnr_bwd[t]) / SNR[t]
```

This biases the loss toward intermediate noise levels where learning signal is strongest, de-emphasizing both very clean (easy) and very noisy (hard) predictions.

### 3.5 DDIM Sampling

The default sampling uses **DDIM** (Denoising Diffusion Implicit Models) with deterministic inference:

```python
# Given model prediction → x_0_pred, eps_pred
x_{t-1} = sqrt(alpha_cumprod[t-1]) * x_0_pred + sqrt(1 - alpha_cumprod[t-1]) * eps_pred
```

When `deterministic_inference: True`, no stochastic noise is added (η=0).

**Timestep sub-sampling**: 200 train steps → 10 inference steps via uniform spacing.

### 3.6 Dynamic Thresholding (for CFG)

When classifier-free guidance scale > 1.0, predictions can exceed the training range. Dynamic thresholding clips x₀ predictions:

```python
if |x_0_pred| > percentile_threshold (99.5th):
    x_0_pred = clip(x_0_pred, -s, s) / s  # where s = 99.5th percentile value
```

### 3.7 Classifier-Free Guidance

During sampling, two forward passes are done (or batched together):

```python
eps_uncond = model(x_t, t, cond=zeros)
eps_cond   = model(x_t, t, cond=ext_cond)
eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

The implementation uses **batched CFG** — both conditional and unconditional inputs are concatenated along the batch dimension for a single forward pass, then split afterward.

---

## 4. DiT1D: 1D Diffusion Transformer Backbone

**Location:** `diffusion_forcing_transformer/dit1d.py` (~213 lines)

A Transformer operating on 1D sequences (trajectories), conditioned via Adaptive Layer Normalization with Zero-initialization (AdaLN-Zero).

### 4.1 Architecture

```
Input: x (B, T, 51)
   │
   ▼
input_embedder: Linear(51, 256)           → (B, T, 256)
   │
   ▼
+ positional_embedding (sinusoidal, T positions)
   │
   ▼
+ [optional] waypoint_indicator_emb       → (B, T, 256)
   │
   ▼
DiTBlock × 8 (depth=8), each conditioned by c
   │
   ▼
FinalLayer (AdaLN + Linear)               → (B, T, 51)
```

**Key dimensions:**
| Parameter     | Value |
|---------------|-------|
| `hidden_size` | 256   |
| `depth`       | 8     |
| `num_heads`   | 4     |
| `head_dim`    | 64    |
| `mlp_ratio`   | 4.0   |
| `mlp_hidden`  | 1024  |

### 4.2 Conditioning Vector `c`

The conditioning vector is the sum of two embeddings:

```python
c = noise_level_emb(k) + external_cond_emb(ext_cond)
```

Where:
- `noise_level_emb`: `StochasticTimeEmbedding` — projects scalar noise level to 256-dim via random Fourier features + MLP
- `external_cond_emb`: `RandomDropoutCondEmbedding` — Linear(512, 256) with training-time dropout

Both produce `(B, 256)` vectors. Their sum `c` is `(B, 256)` and is expanded to `(B, T, 256)` when token-wise conditioning differs (e.g., different noise levels per token).

### 4.3 DiT Blocks with AdaLN-Zero

**Location:** `diffusion_forcing_transformer/dit_blocks.py`

Each `DiTBlock` contains:

```
Input: x (B, T, 256), c (B, [T,] 256)
   │
   ▼
AdaLN-Zero modulation:
   [γ1, β1, α1, γ2, β2, α2] = Linear(c) → 6 × 256 vectors
   │
   ├── Branch 1: Attention
   │   norm1(x) → scale by γ1, shift by β1
   │   → Multi-Head Self-Attention (4 heads, optional RoPE)
   │   → gate by α1
   │   → residual add to x
   │
   └── Branch 2: MLP
       norm2(x) → scale by γ2, shift by β2
       → MLP(256 → 1024 → 256, GELU activation)
       → gate by α2
       → residual add to x
```

**AdaLN-Zero** is critical: the gate vectors (α1, α2) are **initialized to zero**, meaning the block initially acts as an identity function. This stabilizes training by allowing the model to gradually incorporate conditioning.

**Token-wise conditioning**: When noise levels differ per token (autoregressive mode), `c` has shape `(B, T, 256)`, and modulation is applied per-token independently.

### 4.4 Positional Embedding

Sinusoidal positional embedding:
```python
pos_embed = sinusoidal_embedding(max_tokens, hidden_size)  # (T_max, 256)
```
Added to input embeddings before the transformer blocks. Fixed (not learned) by default.

### 4.5 StochasticTimeEmbedding

**Location:** `diffusion_forcing_transformer/embeddings.py`

Projects scalar noise level (continuous in [0, 1]) to hidden dimension:

```
k ∈ [0, 1]
   │
   ▼
Random Fourier features: sin/cos(k * W + b)  → (B, dim)
   │
   ▼
MLP: Linear → SiLU → Linear                  → (B, 256)
```

The random Fourier weights `W` are fixed (not learned), providing a smooth continuous embedding of the noise level.

---

## 5. Configuration Summary

```yaml
model:
  type: dfot
   backbone_type: dit1d
  hidden_size: 256
  depth: 8
  num_heads: 4
  num_features: 51
   num_timesteps: 10
   state_history: 2
   num_observations: 45
   num_task_params: 4
   group_wise_embedding: true
   history_aggregation: attn
```

---

## 6. Model Size Estimate

| Component                  | Parameters (approx.)      |
|----------------------------|---------------------------|
| Input embedder (51→256)    | ~13K                      |
| State MLP (45→256)         | ~24K                      |
| Task MLP (3→256)           | ~2K                       |
| Waypoint indicator MLP     | ~79K                      |
| Noise level embedding      | ~66K                      |
| External cond embedding    | ~131K                     |
| 8 × DiTBlock              | 8 × ~790K ≈ 6.3M         |
| Final layer                | ~79K                      |
| **Total**                  | **~6.7M parameters**      |

---

## 7. Architecture Diagram

```
                    ┌─────────────────┐
                    │  current_state  │ (B, H, 45)
                    │   task_params   │ (B, 4)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ state_embedding │ → (B, 256)
                    │ task_embedding  │ → (B, 256)
                    └────────┬────────┘
                             │ concat
                    ┌────────▼────────┐
                    │    ext_cond     │ (B, 512)
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                     │
   noise_level_emb    ext_cond_emb          waypoint_mask
   (B,[T],256)        (B,256)               (B, T, 51)
        │                    │                     │
        └────────┬───────────┘              waypoint_proj
                 │ sum                      (B, T, 256)
        ┌────────▼────────┐                        │
        │    c vector     │ (B,[T],256)             │
        └────────┬────────┘                        │
                 │                                  │
        ┌────────▼─────────────────────────────────▼──┐
        │                    DiT1D                     │
        │  ┌──────────────────────────────────────┐   │
        │  │ input_embed(x_k) + pos_emb + wp_emb  │   │
        │  └──────────────┬───────────────────────┘   │
        │                 │                            │
        │  ┌──────────────▼───────────────────────┐   │
        │  │      DiTBlock × 8 (AdaLN-Zero)       │   │
        │  │      conditioned by c                 │   │
        │  └──────────────┬───────────────────────┘   │
        │                 │                            │
        │  ┌──────────────▼───────────────────────┐   │
        │  │    FinalLayer → (B, T, 51)            │   │
        │  └──────────────────────────────────────┘   │
        └──────────────────┬───────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │   v-prediction  │ (B, T, 51)
                  └─────────────────┘
```

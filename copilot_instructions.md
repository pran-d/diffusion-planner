# Dual-Branch Cross-Attention DiT for Motion Generation

## Background

The current architecture is a single `DiT1D` backbone: a flat sequence of `DiTBlock`s (self-attention + MLP) where **all 51 features** are treated as one undifferentiated vector per timestep.  
Task/style goals enter via **AdaLN-Zero** conditioning (added to the noise-level embedding), and state history enters via an `MLP → HistoryAggregator` pipeline that also folds into the same AdaLN conditioning vector.

The proposal splits the backbone into two parallel transformer streams:

| Branch | Features | Dim |
|---|---|---|
| **Branch 1 – World** | `delta_xy` (2) + `delta_yaw` (1) + `obj_delta_xy` (2) + `obj_z` (1) + `obj_rel_pos` (3) + `obj_rel_rot6d` (6) | **15** |
| **Branch 2 – Body** | `joints` (29) + `body_z` (1) + `body_rot6d` (6) | **36** |

Keys/values from Branch 1 answer the queries of Branch 2 via cross-attention.  
Both branches receive the *same* task condition via AdaLN-Zero; state history is consumed via dedicated cross-attention layers in each branch.

---

## Design Assessment

### Is this an improvement or overhead?

**Architecturally well-motivated.** The split mirrors a key inductive bias in humanoid loco-manipulation:
- The **world-frame quantities** (base position, object pose) are goal-relevant and relatively low-dimensional.  Routing goal conditioning through them first and then broadcasting to the body branch via cross-attention is a principled separation of concerns.
- **Joints are extremely high-dimensional (29 DOFs)** but largely determined *given* the body's world pose and the object location.  Making Branch 2 attend to Branch 1 encodes this causal hierarchy explicitly.
- The cross-attention gate is low-cost: Branch 2 queries (36→`H`) attend to Branch 1 keys/values (15→`H`), which is a short sequence — `O(T × 15)` per head.

**Potential concerns addressed in the plan:**
1. The 15-dim Branch 1 sequence is short → cross-attention cost is negligible.
2. Keeping Branch 1 architecture *deeper* (more blocks) than Branch 2 is optional but can help since world-frame planning is harder to generalize.
3. The state-history conditioning is currently a single pooled token (HistoryAggregator).  Replacing it with a proper **cross-attention to the raw history sequence** (no pooling) directly in each branch will improve generalization to unseen goals — the model can learn *which* frames of history are informative per-feature-group.

---

## Open Questions

> [!IMPORTANT]
> **Q1 – Shared or separate noise conditioning?**
> Should Branch 1 and Branch 2 share one `StochasticTimeEmbedding` (same noise-level token) or have independent ones? Shared is simpler and correct for synchronous denoising. Confirm before implementation.
> **Answer**: Share the embedding

> [!IMPORTANT]
> **Q2 – Depth split?**
> Current `depth=8`. Proposed: Branch 1 gets 8 blocks (deeper, focuses on the harder world-frame prediction), Branch 2 gets 6 blocks (attends to Branch 1 for guidance). Should this be configurable (`branch1_depth`, `branch2_depth`) or fixed?
> **Answer**: Make them configurable.

> [!NOTE]
> **Q3 – History cross-attention or keep pooled aggregator?**
> The plan proposes replacing `HistoryAggregator` (attention pooling → single token) with a proper cross-attention layer in each branch that attends over all `H` history frames. This is the main change that targets **generalization to unseen goals**. If compute budget is a concern, keeping the pooled aggregator and only adding branch cross-attention is a valid fallback.
> **Status (deferred)**: Current implementation keeps the pooled `HistoryAggregator` (same as `dit1d`). Per-branch history cross-attention is not yet implemented.

> [!NOTE]
> **Q4 – Waypoint indicator injection?**
> Currently `waypoint_indicator_proj` adds to the hidden state of the single backbone. In a dual-branch setup, should it inject into both branches separately (two projectors), or only Branch 1 (world-frame waypoints drive Branch 2)?
> **Answer**:  Only Branch 1.

---

## Proposed Changes

### Feature split definition (`utils/math/sbto_utils.py`)

No code changes — the `build_feature_layout()` dict already defines all slices. The split is derived programmatically in the new backbone.

---

### New building blocks (`diffusion_forcing_transformer/dit_blocks.py`)

#### [MODIFY] [dit_blocks.py](file:///home/pranav/tu-munich/atari/diffusion-planner/diffusion_forcing_transformer/dit_blocks.py)

Add two new block classes:

1. **`CrossAttentionLayer`** — a standard cross-attention module (Q from one stream, K/V from another) with pre-norm (AdaLN or LayerNorm). Reuses the existing `Attention` head implementation (separate Q/K/V projections rather than the fused `qkv`).

2. **`DiTBlockWithCrossAttn`** — a `DiTBlock` extended with an additional cross-attention sublayer inserted between self-attention and MLP:
   ```
   x = x + gate_sa  * self_attn(AdaLN(x, c))
   x = x + gate_ca  * cross_attn(AdaLN(x, c), context)   ← new
   x = x + gate_mlp * mlp(AdaLN(x, c))
   ```
   The gate for the cross-attention sublayer is produced by a separate `AdaLayerNormZero`.

---

### New backbone (`diffusion_forcing_transformer/dit1d_dual.py`) **[NEW]**

New file: `diffusion_forcing_transformer/dit1d_dual.py`

Key design:

```python
class DiT1DDual(BaseBackbone):
    """
    Dual-branch DiT with cross-attention coupling.

    Branch 1 (World): delta_xy, delta_yaw, obj_delta_xy, obj_z,
                      obj_rel_pos, obj_rel_rot6d  → 15 dims
    Branch 2 (Body):  joints, body_z, body_rot6d              → 36 dims

    Information flow:
        1. Both branches embed their feature slices independently.
        2. Both receive task goal via AdaLN-Zero conditioning (same c vector).
        3. State-history cross-attention: each branch attends to the
           embedded history sequence (B, H, hidden).
        4. branch1_blocks: standard DiTBlocks.
        5. branch2_blocks: DiTBlockWithCrossAttn — each block has
           cross-attention (Q from B2, KV from B1 output).
        6. Separate DITFinalLayer heads per branch → outputs concatenated
           back to full (B, T, D) in original feature order.
    """
```

**Module structure:**

| Sub-module | Purpose |
|---|---|
| `self.branch1_input_proj` | `nn.Linear(15, H)` |
| `self.branch2_input_proj` | `nn.Linear(36, H)` |
| `self.pos_emb` | Shared sinusoidal/learned 1D PE |
| `self.history_xattn_b1` | `nn.MultiheadAttention(H, heads)` — B1 attends to history |
| `self.history_xattn_b2` | `nn.MultiheadAttention(H, heads)` — B2 attends to history |
| `self.branch1_blocks` | `nn.ModuleList` of `DiTBlock` (depth=`branch1_depth`) |
| `self.branch2_blocks` | `nn.ModuleList` of `DiTBlockWithCrossAttn` (depth=`branch2_depth`) |
| `self.final_b1` | `DITFinalLayer(H, 15)` |
| `self.final_b2` | `DITFinalLayer(H, 36)` |

**Forward pass:**
```
h1 = branch1_input_proj(x[..., branch1_slice]) + pos_emb
h2 = branch2_input_proj(x[..., branch2_slice]) + pos_emb

# State-history cross-attention (optional, controlled by config)
if history_emb is not None:
    h1 = h1 + history_xattn_b1(h1, history_emb, history_emb)
    h2 = h2 + history_xattn_b2(h2, history_emb, history_emb)

# Branch 1 forward
for block in branch1_blocks:
    h1 = block(h1, c)

# Branch 2 forward (cross-attends to h1 at every block)
for block in branch2_blocks:
    h2 = block(h2, c, context=h1)

# Output heads
out1 = final_b1(h1, c)   # (B, T, 15)
out2 = final_b2(h2, c)   # (B, T, 36)

# Reassemble in original feature order
out = torch.zeros(B, T, D)
out[..., branch1_slice] = out1
out[..., branch2_slice] = out2
return out
```

> [!NOTE]
> The history embedding `history_emb` is computed **upstream** in `DFoTTrajectory`, not inside the backbone. This keeps the backbone stateless with respect to history. The backbone receives it as an optional extra argument — or alternatively it can be folded into the AdaLN `c` vector as before (cheaper but less expressive).

---

### Changes to `DFoTTrajectory` (`models/dfot_trajectory.py`)

#### [MODIFY] [dfot_trajectory.py](file:///home/pranav/tu-munich/atari/diffusion-planner/models/dfot_trajectory.py)

1. **Feature slice registration**: Compute and register `branch1_slice` and `branch2_slice` as `torch.Size`/`slice` objects derived from `feature_index_map` at init time.

2. **Backbone instantiation**: When `backbone_type == "dit1d_dual"`, instantiate `DiT1DDual` instead of `DiT1D`. Pass branch slices and optional config keys `branch1_depth`, `branch2_depth`, `use_history_xattn`.

3. **History conditioning path**: 
   - **Current**: `state_embedding(obs) → HistoryAggregator → single token → added to c`  
   - **Proposed**: `state_embedding(obs) → (B, H, hidden)` sequence is passed as `history_emb` to `DiT1DDual`. No aggregation — the backbone's per-branch cross-attention layers handle it. This is the key change for generalization.
   - Keep the existing path as a fallback (controlled by `use_history_xattn: true/false` in config).

4. **`_model_predictions_with_indicator`**: Update to route through `DiT1DDual.forward()` instead of manually stepping through `DiT1D` blocks. The waypoint indicator is injected into **both** branch hidden states.

5. **Waypoint indicator projection**: Split into `waypoint_indicator_proj_b1` (projects `D`-dim mask → `H`, applied to branch-1-relevant dims) and `waypoint_indicator_proj_b2` (same for branch 2). Alternatively a single `D → H` projection is applied and summed into both, which is simpler.

---

### Config changes (`config/config.yaml`)

#### [MODIFY] [config.yaml](file:///home/pranav/tu-munich/atari/diffusion-planner/config/config.yaml)

```yaml
model:
  backbone_type: "dit1d_dual"   # was "dit1d"
  branch1_depth: 8              # World branch depth
  branch2_depth: 6              # Body branch depth
  use_history_xattn: true       # Replace pooled aggregator with cross-attn
  hidden_size: 256
  num_heads: 4
  mlp_ratio: 4.0
  # branch1/branch2 feature groups (auto-derived if not specified)
  branch1_features:
    - delta_xy
    - delta_yaw
    - obj_delta_xy
    - obj_z
    - obj_rel_pos
    - obj_rel_rot6d
  branch2_features:
    - joints
    - body_z
    - body_rot6d
```

---

## Parameter Budget Comparison

| Component | Current | Proposed |
|---|---|---|
| Input projection | 1 × Linear(51, 256) | 2 × Linear(15/36, 256) |
| Transformer blocks | 8 × DiTBlock | 8 × DiTBlock (B1) + 6 × DiTBlockWithCrossAttn (B2) |
| Cross-attn (history) | 1 × MHA (pooled, O(1) overhead) | 2 × MHA over H frames |
| Cross-attn (B1→B2) | — | 6 × MHA per DiTBlockWithCrossAttn |
| Output heads | 1 × Linear(256, 51) | 2 × Linear(256, 15/36) |
| **Approx total Δ** | **~15M** | **~19–21M** (~+30%)** |

The overhead is moderate and justified by the structural bias gain. The cross-attention from B2 to B1 is cheap because the key/value sequence length is only `T × B1_tokens` (same temporal length, smaller feature space already projected to `H`).

---

## Generalization Strategy

The dual-branch design improves generalization to unseen goals through three mechanisms:

1. **Structural inductive bias**: Branch 1 learns *where* (world-frame navigation + object pose). Branch 2 learns *how* (joint configuration given where). This decomposition is consistent across all tasks, making the goal-conditioning signal easier to generalize — the model never has to disentangle position from joints in a flat embedding.

2. **History cross-attention (replacing pooled aggregator)**: Instead of compressing all history into one token, each branch attends over the full history sequence. For novel goals, the relevant frames of history may differ from training — a cross-attention mechanism is better than a fixed pooling at selecting the right context.

3. **CFG operates cleanly on task AdaLN**: Since the task condition only enters through AdaLN (not mixed into self-attention keys), masking it for classifier-free guidance is clean and complete. This was already the current design and is preserved.

---

## Implementation Status

| Component | Status | Notes |
|---|---|---|
| `CrossAttentionLayer` | ✅ Done | `dit_blocks.py` — AdaLN-Zero gated, Q from x, KV from context |
| `DiTBlockWithCrossAttn` | ✅ Done | `dit_blocks.py` — SA → CA → MLP, gate zero-init |
| `DiT1DDual` backbone | ✅ Done | `dit1d_dual.py` — dual-branch, shared pos_emb + noise cond |
| `discrete_diffusion.py` registration | ✅ Done | `"dit1d_dual"` case added |
| `dfot_trajectory.py` wiring | ✅ Done | branch index derivation, cleaner `_model_predictions_with_indicator` |
| `config/config.yaml` | ✅ Done | `backbone_type: dit1d_dual`, `branch1_depth`, `branch2_depth`, feature lists |
| `DiT1D.forward()` `indicator` kwarg | ✅ Done | Unified indicator injection path for both backbones |
| History cross-attention (Q3) | ⏳ Deferred | Pooled `HistoryAggregator` kept; per-branch cross-attn to history is the next step for generalization |

### Key design choices implemented
- **Q1**: Shared `StochasticTimeEmbedding` (single `c` vector for both branches) ✅
- **Q2**: `branch1_depth` / `branch2_depth` are configurable in config ✅
- **Q4**: Waypoint indicator injected into **Branch 1 only** in `DiT1DDual.forward()` ✅

---

## Verification Plan

### Unit Tests
- Instantiate `DiT1DDual` with a dummy batch; verify output shape `(B, T, 51)` with features in the correct order.
- Verify gradients flow through both branches.
- Verify `DiTBlockWithCrossAttn` cross-attention gate initializes to zero (training stability).

### Integration Test
- Run one training step with `backbone_type: dit1d_dual` and confirm loss is not NaN.
- Compare parameter counts to baseline.

### Training Validation
- Train for ~20 epochs (same as current `config.yaml`), compare training loss curve vs `dit1d` baseline.
- Run inference on a held-out goal and visually inspect trajectory quality.
"""
Dual-Branch Cross-Attention DiT for motion generation.

Branch 1 (World):  delta_xy, delta_yaw, obj_delta_xy, obj_z, obj_rel_pos, obj_rel_rot6d  → 15 dims
Branch 2 (Body):   joints, body_z, body_rot6d                                              → 36 dims

Information flow:
  1. x is split into branch1 / branch2 feature subsets by index.
  2. Each subset is projected independently to hidden_size.
  3. Both branches receive the same task/noise conditioning via AdaLN-Zero (c vector).
  4. Branch 1 runs through standard DiTBlocks.
  5. Branch 2 runs through DiTBlockWithCrossAttn blocks that cross-attend to Branch 1.
  6. Separate final-layer heads per branch; outputs are reassembled in the original
     51-dim feature order before returning.

Waypoint indicator (when provided) is injected only into Branch 1 (Q4 answer).
"""

from typing import Optional, Dict, List
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base_backbone import BaseBackbone
from .dit_blocks import DiTBlock, DiTBlockWithCrossAttn, DITFinalLayer
from .dit1d import SinusoidalPositionalEmbedding


class DiT1DDual(BaseBackbone):
    """
    Dual-branch DiT with cross-attention coupling.

    cfg keys (beyond BaseBackbone defaults):
        branch1_indices   (list[int])  feature indices for Branch 1 (World)
        branch2_indices   (list[int])  feature indices for Branch 2 (Body)
        branch1_depth     (int)        number of DiTBlocks for Branch 1   [default 8]
        branch2_depth     (int)        number of DiTBlockWithCrossAttn for Branch 2 [default 6]
        hidden_size       (int)        shared hidden dimension
        num_heads         (int)        attention heads
        mlp_ratio         (float)      MLP expansion ratio
        pos_emb_type      (str)        "learned_1d" | "sinusoidal_1d"
        use_gradient_checkpointing (bool)
    """

    def __init__(
        self,
        cfg,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask: bool = False,
    ):
        super().__init__(cfg, x_shape, max_tokens, external_cond_dim, use_causal_mask)

        hidden_size = cfg.hidden_size
        num_heads = cfg.num_heads
        mlp_ratio = cfg.mlp_ratio
        branch1_depth = cfg.get("branch1_depth", 8)
        branch2_depth = cfg.get("branch2_depth", 6)
        pos_emb_type = cfg.get("pos_emb_type", "learned_1d")
        self.use_gradient_checkpointing = cfg.get("use_gradient_checkpointing", False)

        # Branch feature index tensors (registered as buffers so they move with .to(device))
        b1_idx = list(cfg.branch1_indices)
        b2_idx = list(cfg.branch2_indices)
        self.register_buffer("branch1_indices", torch.tensor(b1_idx, dtype=torch.long), persistent=False)
        self.register_buffer("branch2_indices", torch.tensor(b2_idx, dtype=torch.long), persistent=False)

        branch1_dim = len(b1_idx)
        branch2_dim = len(b2_idx)
        total_dim = x_shape[0]
        assert branch1_dim + branch2_dim == total_dim, (
            f"branch1 ({branch1_dim}) + branch2 ({branch2_dim}) != total ({total_dim})"
        )

        self.hidden_size = hidden_size
        self.total_dim = total_dim

        # Input projections (one per branch)
        self.branch1_input_proj = nn.Linear(branch1_dim, hidden_size)
        self.branch2_input_proj = nn.Linear(branch2_dim, hidden_size)

        # Mask tokens (one per branch, used during masked-token inpainting)
        self.branch1_mask_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.branch2_mask_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # Shared positional embedding
        learnable = (pos_emb_type == "learned_1d")
        self.pos_emb = SinusoidalPositionalEmbedding(
            embed_dim=hidden_size, shape=(max_tokens,), learnable=learnable
        )

        # Branch 1: standard DiTBlocks
        self.branch1_blocks = nn.ModuleList([
            DiTBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(branch1_depth)
        ])

        # Branch 2: DiTBlockWithCrossAttn — cross-attends to Branch 1 at each block
        self.branch2_blocks = nn.ModuleList([
            DiTBlockWithCrossAttn(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(branch2_depth)
        ])

        # Output heads
        self.final_b1 = DITFinalLayer(hidden_size, branch1_dim)
        self.final_b2 = DITFinalLayer(hidden_size, branch2_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # BaseBackbone abstract properties
    # ------------------------------------------------------------------

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for proj in (self.branch1_input_proj, self.branch2_input_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

        def _mlp_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    def _ckpt(self, module: nn.Module, *args):
        if self.use_gradient_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, T, D=51) noised trajectory
            noise_levels:     (B, T) integer noise levels
            external_cond:    (B, 1, cond_dim) or (B, T, cond_dim)
            external_cond_mask: optional boolean mask for conditioning
            mask:             (B, T) bool — True means token is unknown (uses mask token)
            indicator:        (B, T, hidden) waypoint indicator; injected into Branch 1 only
        Returns:
            (B, T, D) predicted output in original feature order
        """
        B, T, D = x.shape

        # 1. Split features
        x1 = x[..., self.branch1_indices]  # (B, T, branch1_dim)
        x2 = x[..., self.branch2_indices]  # (B, T, branch2_dim)

        # 2. Input projections
        h1 = self.branch1_input_proj(x1)   # (B, T, H)
        h2 = self.branch2_input_proj(x2)   # (B, T, H)

        # 3. Mask tokens (applied before pos_emb)
        if mask is not None:
            h1 = torch.where(mask.unsqueeze(-1), self.branch1_mask_token, h1)
            h2 = torch.where(mask.unsqueeze(-1), self.branch2_mask_token, h2)

        # 4. Positional embedding (shared)
        h1 = self.pos_emb(h1)
        h2 = self.pos_emb(h2)

        # 5. Waypoint indicator → Branch 1 only (Q4)
        if indicator is not None:
            h1 = h1 + indicator

        # 6. Conditioning vector c (shared for both branches)
        c = self.noise_level_pos_embedding(noise_levels)
        if external_cond is not None:
            c = c + self.external_cond_embedding(external_cond, external_cond_mask)

        # 7. Branch 1 forward (standard DiTBlocks)
        for block in self.branch1_blocks:
            h1 = self._ckpt(block, h1, c)

        # 8. Branch 2 forward (cross-attends to h1 at every block)
        for block in self.branch2_blocks:
            h2 = self._ckpt(block, h2, c, h1)

        # 9. Output heads
        out1 = self.final_b1(h1, c)  # (B, T, branch1_dim)
        out2 = self.final_b2(h2, c)  # (B, T, branch2_dim)

        # 10. Reassemble in original feature order
        out = torch.empty(B, T, D, device=x.device, dtype=x.dtype)
        out[..., self.branch1_indices] = out1
        out[..., self.branch2_indices] = out2
        return out

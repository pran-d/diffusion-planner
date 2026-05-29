"""
Dual-Branch DiT with per-feature-group tokens, factorized (time, group) attention,
and group-level cross-attention between branches.

Token grid (per branch b with G_b feature groups):
    (B, T, G_b, H)

Per block:
    1. Temporal self-attention      — Attention over T  (per-group T x T map)
    2. Group self-attention         — Attention over G_b (per-timestep G_b x G_b map)
    3. (Branch 2 only) Group cross-attention — body-group tokens query world-group
       tokens at the same timestep  (per-timestep G_b x G_w map)
    4. MLP

Output: per-group Linear(H -> group_dim) writes into the corresponding slice of
the (B, T, D) output tensor (same reassembly logic as DiT1DDual).
"""

from typing import Optional, List
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Mlp
from functools import partial

from .base_backbone import BaseBackbone
from .dit_blocks import Attention, CrossAttentionLayer, AdaLayerNormZero, DITFinalLayer
from .dit1d import SinusoidalPositionalEmbedding


class _GroupCrossAttn(nn.Module):
    """Per-timestep group-axis cross-attention. Q: (B, T, Gq, H), K/V: (B, T, Gkv, H).

    Implemented with a manual softmax so attention_capture can read `_last_attn`.
    Shape of captured tensor: (B*T, num_heads, Gq, Gkv).
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = AdaLayerNormZero(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Capture state (read by utils/attention_capture.py if a wrapper sets the flag)
        self._capture_attn: bool = False
        self._last_attn: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, ctx: torch.Tensor
    ) -> torch.Tensor:
        B, T, Gq, H = x.shape
        Gkv = ctx.shape[2]
        c_b = c.unsqueeze(2).expand(B, T, Gq, H)
        x_norm, gate = self.norm(x, c_b)

        q = self.q_proj(x_norm).reshape(B * T, Gq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).reshape(B * T, Gkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).reshape(B * T, Gkv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)            # (B*T, h, Gq, Gkv)
        attn = attn.softmax(dim=-1)
        if self._capture_attn:
            self._last_attn = attn.detach()
        out = (attn @ v).transpose(1, 2).reshape(B * T, Gq, H)
        out = self.out_proj(out).reshape(B, T, Gq, H)
        return gate * out


class _GroupedBranchBlock(nn.Module):
    """One transformer block operating on (B, T, G, H) tokens.

    Sublayers (all gated by AdaLN-Zero on the shared (B, T, H) conditioning c):
      - temporal self-attention (over T, per group)
      - group self-attention    (over G, per timestep)
      - optional group cross-attention (body branch only): queries from this
        branch, keys/values from the other branch
      - MLP

    Attention modules are exposed as named submodules so attention_capture can
    find them.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        with_cross: bool = False,
    ):
        super().__init__()
        self.with_cross = with_cross

        self.norm_t = AdaLayerNormZero(hidden_size)
        self.temporal_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        self.norm_g = AdaLayerNormZero(hidden_size)
        self.group_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        if with_cross:
            self.group_cross_attn = _GroupCrossAttn(hidden_size, num_heads)

        self.norm_mlp = AdaLayerNormZero(hidden_size)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=partial(nn.GELU, approximate="tanh"),
        )

        self._init_weights()

    def _init_weights(self):
        def _xavier(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.temporal_attn.apply(_xavier)
        self.group_attn.apply(_xavier)
        self.mlp.apply(_xavier)

    # --------------------------------------------------------------

    def _temporal(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, T, G, H), c: (B, T, H). Attention over T axis per group."""
        B, T, G, H = x.shape
        c_b = c.unsqueeze(2).expand(B, T, G, H)
        x_norm, gate = self.norm_t(x, c_b)                          # (B, T, G, H)
        # (B, T, G, H) -> (B*G, T, H)
        x_flat = x_norm.permute(0, 2, 1, 3).reshape(B * G, T, H)
        out = self.temporal_attn(x_flat)
        out = out.reshape(B, G, T, H).permute(0, 2, 1, 3)           # (B, T, G, H)
        return x + gate * out

    def _group(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, T, G, H), c: (B, T, H). Attention over G axis per timestep."""
        B, T, G, H = x.shape
        c_b = c.unsqueeze(2).expand(B, T, G, H)
        x_norm, gate = self.norm_g(x, c_b)
        # (B, T, G, H) -> (B*T, G, H)
        x_flat = x_norm.reshape(B * T, G, H)
        out = self.group_attn(x_flat).reshape(B, T, G, H)
        return x + gate * out

    def _mlp(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, T, G, H = x.shape
        c_b = c.unsqueeze(2).expand(B, T, G, H)
        x_norm, gate = self.norm_mlp(x, c_b)
        return x + gate * self.mlp(x_norm)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._temporal(x, c)
        x = self._group(x, c)
        if self.with_cross:
            assert ctx is not None, "cross-attn block requires context"
            x = x + self.group_cross_attn(x, c, ctx)
        x = self._mlp(x, c)
        return x


class DiT1DDualGrouped(BaseBackbone):
    """
    cfg keys (beyond BaseBackbone defaults):
        hidden_size, num_heads, mlp_ratio                           (shared)
        branch1_depth, branch2_depth                                (per-branch depth)
        pos_emb_type ("learned_1d" | "sinusoidal_1d")
        use_gradient_checkpointing

        branch1_group_names      list[str]     names of branch-1 feature groups
        branch1_group_starts     list[int]     start index of each group in the full feature vector
        branch1_group_stops      list[int]     stop  index of each group in the full feature vector
        branch2_group_names / starts / stops   same for branch 2
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
        branch1_depth = cfg.get("branch1_depth", 4)
        branch2_depth = cfg.get("branch2_depth", 4)
        pos_emb_type = cfg.get("pos_emb_type", "learned_1d")
        self.use_gradient_checkpointing = cfg.get("use_gradient_checkpointing", False)

        # Group descriptors per branch
        self.branch1_group_names: List[str] = list(cfg.branch1_group_names)
        self.branch2_group_names: List[str] = list(cfg.branch2_group_names)
        b1_starts = list(cfg.branch1_group_starts)
        b1_stops = list(cfg.branch1_group_stops)
        b2_starts = list(cfg.branch2_group_starts)
        b2_stops = list(cfg.branch2_group_stops)

        # Persist as buffers (kept as plain python lists too for indexing convenience)
        self._b1_starts = b1_starts
        self._b1_stops = b1_stops
        self._b2_starts = b2_starts
        self._b2_stops = b2_stops

        self.Gw = len(self.branch1_group_names)
        self.Gb = len(self.branch2_group_names)
        self.hidden_size = hidden_size
        self.total_dim = x_shape[0]

        b1_total = sum(s2 - s1 for s1, s2 in zip(b1_starts, b1_stops))
        b2_total = sum(s2 - s1 for s1, s2 in zip(b2_starts, b2_stops))
        assert b1_total + b2_total == self.total_dim, (
            f"branch1 ({b1_total}) + branch2 ({b2_total}) != total ({self.total_dim})"
        )

        # Per-group input projections
        self.branch1_input_projs = nn.ModuleList([
            nn.Linear(s2 - s1, hidden_size) for s1, s2 in zip(b1_starts, b1_stops)
        ])
        self.branch2_input_projs = nn.ModuleList([
            nn.Linear(s2 - s1, hidden_size) for s1, s2 in zip(b2_starts, b2_stops)
        ])

        # Learnable group embeddings (per branch)
        self.branch1_group_emb = nn.Parameter(torch.randn(1, 1, self.Gw, hidden_size) * 0.02)
        self.branch2_group_emb = nn.Parameter(torch.randn(1, 1, self.Gb, hidden_size) * 0.02)

        # Mask token (per branch, broadcast across G)
        self.branch1_mask_token = nn.Parameter(torch.randn(1, 1, 1, hidden_size) * 0.02)
        self.branch2_mask_token = nn.Parameter(torch.randn(1, 1, 1, hidden_size) * 0.02)

        # Temporal positional embedding (shared)
        learnable = (pos_emb_type == "learned_1d")
        self.pos_emb = SinusoidalPositionalEmbedding(
            embed_dim=hidden_size, shape=(max_tokens,), learnable=learnable
        )

        # Branch 1 blocks (no cross-attn)
        self.branch1_blocks = nn.ModuleList([
            _GroupedBranchBlock(hidden_size, num_heads, mlp_ratio, with_cross=False)
            for _ in range(branch1_depth)
        ])
        # Branch 2 blocks (with group cross-attn to branch 1)
        self.branch2_blocks = nn.ModuleList([
            _GroupedBranchBlock(hidden_size, num_heads, mlp_ratio, with_cross=True)
            for _ in range(branch2_depth)
        ])

        # Per-group output heads
        self.branch1_out_heads = nn.ModuleList([
            DITFinalLayer(hidden_size, s2 - s1) for s1, s2 in zip(b1_starts, b1_stops)
        ])
        self.branch2_out_heads = nn.ModuleList([
            DITFinalLayer(hidden_size, s2 - s1) for s1, s2 in zip(b2_starts, b2_stops)
        ])

        self._init_weights()

    # ----- BaseBackbone properties -----
    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0

    # -----------------------------------

    def _init_weights(self):
        for proj in list(self.branch1_input_projs) + list(self.branch2_input_projs):
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

    def _embed_branch(
        self,
        x: torch.Tensor,
        starts: List[int],
        stops: List[int],
        projs: nn.ModuleList,
        group_emb: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, T, D). Returns (B, T, G, H)."""
        B, T, _ = x.shape
        tokens = []
        for s1, s2, proj in zip(starts, stops, projs):
            tokens.append(proj(x[..., s1:s2]))                       # (B, T, H)
        out = torch.stack(tokens, dim=2)                             # (B, T, G, H)
        out = out + group_emb                                        # broadcast
        return out

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Per-branch, per-group input projections
        h1 = self._embed_branch(x, self._b1_starts, self._b1_stops, self.branch1_input_projs, self.branch1_group_emb)
        h2 = self._embed_branch(x, self._b2_starts, self._b2_stops, self.branch2_input_projs, self.branch2_group_emb)

        # 2. Mask tokens (per timestep, broadcast across G axis)
        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1)                     # (B, T, 1, 1)
            h1 = torch.where(m, self.branch1_mask_token, h1)
            h2 = torch.where(m, self.branch2_mask_token, h2)

        # 3. Temporal positional embedding — add to each group token at the same t
        pe = self.pos_emb(torch.zeros(B, T, self.hidden_size, device=x.device, dtype=x.dtype))
        pe = pe.unsqueeze(2)                                         # (B, T, 1, H)
        h1 = h1 + pe
        h2 = h2 + pe

        # 4. Waypoint indicator → Branch 1 only, broadcast over group axis
        if indicator is not None:
            h1 = h1 + indicator.unsqueeze(2)

        # 5. Conditioning vector c (shared)
        c = self.noise_level_pos_embedding(noise_levels)
        if external_cond is not None:
            c = c + self.external_cond_embedding(external_cond, external_cond_mask)

        # 6. Branch 1 forward
        for block in self.branch1_blocks:
            h1 = self._ckpt(block, h1, c)

        # 7. Branch 2 forward (cross-attends to h1)
        for block in self.branch2_blocks:
            h2 = self._ckpt(block, h2, c, h1)

        # 8. Per-group output heads → write into reassembled output
        out = torch.empty(B, T, D, device=x.device, dtype=x.dtype)
        c_b1 = c.unsqueeze(2).expand(B, T, self.Gw, self.hidden_size)
        c_b2 = c.unsqueeze(2).expand(B, T, self.Gb, self.hidden_size)

        for g, (s1, s2, head) in enumerate(zip(self._b1_starts, self._b1_stops, self.branch1_out_heads)):
            out[..., s1:s2] = head(h1[:, :, g, :], c_b1[:, :, g, :])
        for g, (s1, s2, head) in enumerate(zip(self._b2_starts, self._b2_stops, self.branch2_out_heads)):
            out[..., s1:s2] = head(h2[:, :, g, :], c_b2[:, :, g, :])

        return out

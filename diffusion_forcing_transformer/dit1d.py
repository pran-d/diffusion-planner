from typing import Optional, Tuple, Literal, Dict, List
import numpy as np
import torch
from torch import nn
from .base_backbone import BaseBackbone
from .dit_blocks import DiTBlock, DITFinalLayer
from torch.utils.checkpoint import checkpoint


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Args:
        embed_dim: Embedding dimension.
        pos: Position tensor of shape (...).
    Returns:
        Positional embeddings with shape (-1, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 100**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed(
    embed_dim: int,
    shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Get n-dimensional sinusoidal positional embeddings.
    Args:
        embed_dim: Embedding dimension.
        shape: Shape of the input tensor.
    Returns:
        Positional embeddings with shape (shape_flattened, embed_dim).
    """
    assert embed_dim % (2 * len(shape)) == 0
    grid = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape])
    grid = np.stack(grid, axis=0)  # (ndim, *shape)
    return np.concatenate(
        [
            get_1d_sincos_pos_embed_from_grid(embed_dim // len(shape), grid[i])
            for i in range(len(shape))
        ],
        axis=1,
    )


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, shape: Tuple[int, ...], learnable: bool = False):
        super().__init__()
        if learnable:
            max_tokens = np.prod(shape)
            self.pos_emb = nn.Parameter(
                torch.zeros(1, max_tokens, embed_dim).normal_(std=0.02),
                requires_grad=True,
            )

        else:
            self.register_buffer(
                "pos_emb",
                torch.from_numpy(get_nd_sincos_pos_embed(embed_dim, shape))
                .float()
                .unsqueeze(0),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pos_emb[:, :seq_len]


class GroupWiseInputEmbedder(nn.Module):
    """Per-semantic-group linear projections whose outputs are summed.

    Each feature group (e.g. *delta_xy*, *joints*, *body_rot6d*) gets its own
    ``nn.Linear(group_dim, hidden_size)``.  The outputs are summed element-wise
    to produce a single ``(B, T, hidden_size)`` embedding, giving the model an
    inductive bias that respects the heterogeneous structure of the feature
    vector.

    Falls back to a single ``nn.Linear`` when no group slices are provided.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        group_slices: Dict[str, slice],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.group_slices = group_slices  # e.g. {"delta_xy": slice(0,2), ...}

        # Build one linear projection per group
        self.group_names: List[str] = []
        self.group_projs = nn.ModuleDict()
        covered = set()
        for name, sl in group_slices.items():
            dim = sl.stop - sl.start
            if dim <= 0:
                continue
            self.group_names.append(name)
            self.group_projs[name] = nn.Linear(dim, hidden_size)
            covered.update(range(sl.start, sl.stop))

        # Catch any feature indices not covered by a named group
        uncovered = sorted(set(range(in_channels)) - covered)
        if uncovered:
            self._uncovered_indices = uncovered
            self.group_names.append("_residual")
            self.group_projs["_residual"] = nn.Linear(len(uncovered), hidden_size)
        else:
            self._uncovered_indices = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_channels) → (B, T, hidden_size)"""
        out = torch.zeros(
            *x.shape[:-1], self.hidden_size, device=x.device, dtype=x.dtype
        )
        for name in self.group_names:
            if name == "_residual":
                xi = x[..., self._uncovered_indices]
            else:
                xi = x[..., self.group_slices[name]]
            out = out + self.group_projs[name](xi)
        return out


class DiT1D(BaseBackbone):

    def __init__(
        self,
        cfg,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            external_cond_dim,
            use_causal_mask,
        )

        if use_causal_mask:
            self.use_causal_mask = True
        else:
            self.use_causal_mask = False

        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_size))

        hidden_size = cfg.hidden_size
        in_channels = x_shape[0]
        out_channels = x_shape[0]
        depth = cfg.depth
        num_heads = cfg.num_heads
        mlp_ratio = cfg.mlp_ratio
        use_gradient_checkpointing = cfg.use_gradient_checkpointing
        pos_emb_type = cfg.pos_emb_type

        self.input_embedder = nn.Linear(in_channels, hidden_size)
        self._group_wise = False  # may be replaced by set_group_wise_embedder()
        
        self.out_channels = out_channels # Assuming learn_sigma=False based on DiT1D.init
        self.hidden_size = hidden_size
        self.depth = depth
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Positional Embedding (Simplified for 1D)
        if pos_emb_type == "learned_1d":
            self.pos_emb = SinusoidalPositionalEmbedding(
                embed_dim=hidden_size,
                shape=(max_tokens,),
                learnable=True,
            )
        else:
             # Default to sinusoidal_1d
            self.pos_emb = SinusoidalPositionalEmbedding(
                embed_dim=hidden_size,
                shape=(max_tokens,),
                learnable=False
            )

        # Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        
        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)
        
        self.initialize_weights()

    @property
    def in_channels(self) -> int:
        return self.x_shape[0]

    def set_group_wise_embedder(self, group_slices: Dict[str, slice]) -> None:
        """Replace the flat input_embedder with per-group projections.

        Call this after __init__ (and before training) to switch to
        group-wise input embedding.  The old ``nn.Linear`` is discarded.
        """
        self.input_embedder = GroupWiseInputEmbedder(
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            group_slices=group_slices,
        )
        self._group_wise = True
        # Re-initialise the new projections
        self._input_embedder_init(self.input_embedder)
        n_groups = len(self.input_embedder.group_names)
        print(f"  → Group-wise input embedding enabled ({n_groups} groups)")

    @staticmethod
    def _input_embedder_init(embedder) -> None:
        # Initialize input_embedder like nn.Linear:
        if isinstance(embedder, GroupWiseInputEmbedder):
            for proj in embedder.group_projs.values():
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        else:
            nn.init.xavier_uniform_(embedder.weight)
            nn.init.zeros_(embedder.bias)

    def initialize_weights(self) -> None:
        self._input_embedder_init(self.input_embedder)

        # Initialize noise level embedding and external condition embedding MLPs:
        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0
    
    def _checkpoint(self, module: nn.Module, *args):
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
        use_kv_cache: bool = False,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        # 1. Input Embedding
        # x is (B, T, C)
        x = self.input_embedder(x) # (B, T, hidden)

        if mask is not None:
            # mask: (B, T) boolean tensor where True means masked
            x = torch.where(mask.unsqueeze(-1), self.mask_token, x)

        # 2. Condition Embedding
        c = self.noise_level_pos_embedding(noise_levels)

        if external_cond is not None:
            c = c + self.external_cond_embedding(external_cond, external_cond_mask)
            
        # 3. Positional Embedding
        x = self.pos_emb(x)
        
        # Transformer Blocks
        for block in self.blocks:
            x = self._checkpoint(block, x, c)

        # Placeholder cache hook: enables API-level ablation toggles without
        # changing current numerical behavior.
        if use_kv_cache and kv_cache is not None:
            kv_cache["last_seq_len"] = x.shape[1]
            
        # Final Layer
        x = self.final_layer(x, c)
        
        return x

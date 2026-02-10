from typing import Optional, Tuple, Literal
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
    omega = 1.0 / 10000**omega  # (D/2,)

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

        hidden_size = cfg.hidden_size
        in_channels = x_shape[0]
        out_channels = x_shape[0]
        depth = cfg.depth
        num_heads = cfg.num_heads
        mlp_ratio = cfg.mlp_ratio
        use_gradient_checkpointing = cfg.use_gradient_checkpointing
        pos_emb_type = cfg.pos_emb_type

        self.input_embedder = nn.Linear(in_channels, hidden_size)
        
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

    @staticmethod
    def _input_embedder_init(embedder) -> None:
        # Initialize input_embedder like nn.Linear:
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
    ) -> torch.Tensor:
        # 1. Input Embedding
        # x is (B, T, C)
        x = self.input_embedder(x) # (B, T, hidden)

        # 2. Condition Embedding
        c = self.noise_level_pos_embedding(noise_levels)

        if external_cond is not None:
            c = c + self.external_cond_embedding(external_cond, external_cond_mask)
            
        # 3. Positional Embedding
        x = self.pos_emb(x)
        
        # Transformer Blocks
        for block in self.blocks:
            x = self._checkpoint(block, x, c)
            
        # Final Layer
        x = self.final_layer(x, c)
        
        return x

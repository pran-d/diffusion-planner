import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def normalization(channels):
    # Safe GroupNorm
    num_groups = 32
    while channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, timesteps):
        emb = self._get_sinusoidal_embedding(timesteps, self.embedding_dim)
        return self.main(emb)
    
    @staticmethod
    def _get_sinusoidal_embedding(timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[..., None] * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None, cond_dim=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Conv1d(channels, self.out_channels, 3, padding=1),
        )
        
        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )

        # Conditional embedding layer (FiLM)
        # Assumes cond is already a vector (B, cond_dim)
        if cond_dim is not None and cond_dim > 0:
            self.cond_layers = nn.Sequential(
                SiLU(),
                nn.Linear(cond_dim, self.out_channels * 2) 
            )
        else:
             self.cond_layers = None
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Conv1d(self.out_channels, self.out_channels, 3, padding=1),
        )
        # Initialize output conv to zero for stability
        self.out_layers[-1].weight.data.zero_()
        self.out_layers[-1].bias.data.zero_()
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)
    
    def forward(self, x, emb, cond=None):
        # x: (B, C, T)
        h = self.in_layers(x)
        
        # Add Time Embedding
        emb_out = self.emb_layers(emb).type(h.dtype) 
        
        # Handle both global (B, C) and per-token (B, T, C) embeddings
        if emb_out.ndim == 2:
            # Global: (B, C) -> (B, C, 1) -> broadcasts to (B, C, T)
            h = h + emb_out[:, :, None]
        elif emb_out.ndim == 3:
            # Per-token: (B, T, C) -> (B, C, T) matches (B, C, T)
            h = h + emb_out.permute(0, 2, 1)
        else:
             raise ValueError(f"Unexpected embedding shape: {emb_out.shape}")
             
        # Add Condition (FiLM)
        if cond is not None:
            # Check if condition is valid (not empty tensor)
            if hasattr(cond, 'shape') and cond.shape[-1] > 0:
                 cond_out = self.cond_layers(cond) # (B, 2*C)
                 
                 # Handle broadcasting for 3D tensors (B, T, D) vs 2D (B, D)
                 # self.cond_layers is Linear, so it works on last dim.
                 # If cond is (B, T, D) -> cond_out (B, T, 2*C)
                 # If cond is (B, D) -> cond_out (B, 2*C)
                 
                 if cond_out.ndim == 2:
                     cond_out = cond_out[:, :, None] # (B, 2*C, 1)
                 elif cond_out.ndim == 3:
                     # (B, T, 2*C) -> Need (B, 2*C, T)
                     cond_out = cond_out.permute(0, 2, 1)
                     
                 scale, bias = cond_out.chunk(2, dim=1)
                 h = h * (1 + scale) + bias
            
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        b, c, l = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Compute attention
        # (B, C, L) -> (B, L, C) for matmul
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        
        # We want attention over L
        # dot(Q, K.T)
        attn = torch.matmul(q, k.transpose(1, 2)) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.matmul(attn, v) # (B, L, C)
        h = h.permute(0, 2, 1) # (B, C, L)
        
        h = self.proj_out(h)
        return x + h

class UNet1DModel(nn.Module):
    """
    UNet1D model that preserves temporal dimension (no downsampling on T).
    """
    def __init__(self, in_channels, model_channels, out_channels,
                 num_res_blocks, attention_resolutions, 
                 channel_mult=(1, 2, 4, 8), cond_dim=None):
        super().__init__()
        
        self.model_channels = model_channels
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(model_channels, time_embed_dim)

        # Input blocks
        self.input_blocks = nn.ModuleList([
            nn.Conv1d(in_channels, model_channels, 3, padding=1)
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        
        # Downsampling levels (Channel-wise only if desired, here we just increase channels)
        # Note: We do NOT downsample time (no stride=2).
        
        for level, mult in enumerate(channel_mult):
            out_ch = int(model_channels * mult)
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                    channels=ch, emb_channels=time_embed_dim, 
                    out_channels=out_ch, cond_dim=cond_dim
                )]
                ch = out_ch
                if level in attention_resolutions: 
                    # Use level index instead of resolution since T is constant
                    layers.append(AttentionBlock(ch))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                # Downsample block (usually) - here replaces with Conv but NO stride
                self.input_blocks.append(nn.Conv1d(ch, ch, 3, stride=1, padding=1))
                input_block_chans.append(ch)
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResBlock(channels=ch, emb_channels=time_embed_dim, cond_dim=cond_dim),
            AttentionBlock(ch),
            ResBlock(channels=ch, emb_channels=time_embed_dim, cond_dim=cond_dim),
        )
        
        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = int(model_channels * mult)
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(
                    channels=ch + ich, emb_channels=time_embed_dim, 
                    out_channels=out_ch, cond_dim=cond_dim
                )]
                ch = out_ch
                if level in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                if level and i == num_res_blocks:
                    # Upsample block (usually). Here Conv with stride 1
                    layers.append(nn.Conv1d(ch, ch, 3, stride=1, padding=1))
                    
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Final output
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            nn.Conv1d(ch, out_channels, 3, padding=1),
        )
        # Initialize final conv to zero
        self.out[-1].weight.data.zero_()
        self.out[-1].bias.data.zero_()
    
    def forward(self, x, t, cond=None):
        # x is (B, C, T)
        emb = self.time_embed(t)
        
        hs = []
        h = x
        
        # Encoder
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                # Check inside sequential for ResBlock to pass embeddings
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb, cond)
                    else:
                        h = layer(h)
            elif isinstance(module, ResBlock):
                h = module(h, emb, cond)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        h = self.middle_block[0](h, emb, cond)
        h = self.middle_block[1](h)
        h = self.middle_block[2](h, emb, cond)
        
        # Decoder
        for module in self.output_blocks:
            popped_hs = hs.pop()
            h = torch.cat([h, popped_hs], dim=1)
            
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb, cond)
                    else:
                        h = layer(h)
            elif isinstance(module, ResBlock):
                h = module(h, emb, cond)
            else:
                h = module(h)
                
        return self.out(h)

# Adapter class to match DiT1D interface
class UNet1D(nn.Module):
    def __init__(self, cfg, x_shape, max_tokens, external_cond_dim=0, use_causal_mask=False):
        super().__init__()
        self.in_channels = x_shape[0]
        self.max_tokens = max_tokens
        
        # Map config
        hidden_size = cfg.hidden_size # Base channels
        # If user provides explicit UNet config, use it. Otherwise defaults.
        channel_mult = cfg.get("channel_mult", (1, 2, 4, 8))
        num_res_blocks = cfg.get("num_res_blocks", 2)
        attn_resolutions = cfg.get("attn_resolutions", (2, 3)) # Apply attn at deeper levels
        
        self.unet = UNet1DModel(
            in_channels=self.in_channels,
            model_channels=hidden_size,
            out_channels=self.in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attn_resolutions,
            channel_mult=channel_mult,
            cond_dim=external_cond_dim if external_cond_dim > 0 else None
        )
        
    def forward(self, x, t, cond=None, key_padding_mask=None, **kwargs):
        # x is (B, C, T) coming from simple_trajectory.py
        
        # Flatten conditional if it has shape (B, 1, D)
        if cond is not None and cond.ndim == 3 and cond.shape[1] == 1:
            cond = cond.squeeze(1)
            
        out = self.unet(x, t, cond)
        
        return out


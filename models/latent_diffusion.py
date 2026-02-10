import torch
import torch.nn as nn
from diffusers import UNet1DModel
import torch.nn.functional as F

# ============================================================================
# Network Architecture Components
# ============================================================================

class Mish(nn.Module):
    """Mish activation function."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ResidualFC(nn.Module):
    """Residual fully-connected block."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = Mish()
    
    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + x)


class MHSEChannelAttention(nn.Module):
    """Multi-Head Squeeze-Excitation channel attention."""
    
    def __init__(self, channels, reduction=16, heads=4, fuse='concat'):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.channels = channels
        self.heads = heads
        self.fuse = fuse
        
        self.subc = channels // heads
        
        # SE block for each head
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.subc, self.subc // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(self.subc // reduction, self.subc, bias=False),
                nn.Sigmoid()
            )
            for _ in range(heads)
        ])
        
        if fuse == 'concat':
            self.fuse_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        xs = x.view(B, self.heads, self.subc, H, W)
        
        weights = []
        for i, se in enumerate(self.se_blocks):
            w = se(xs[:, i])
            weights.append(w)
        
        if self.fuse == 'mean':
            w = torch.stack(weights, dim=0).mean(0)
            w = w.view(B, self.subc, 1, 1)
            out = xs * w.unsqueeze(1)
            return out.view(B, C, H, W)
        else:
            out_heads = []
            for i, w in enumerate(weights):
                w = w.view(B, self.subc, 1, 1)
                out_heads.append(xs[:, i] * w)
            out = torch.cat(out_heads, dim=1)
            return self.fuse_conv(out)
        

class CBAMWithMHSE(nn.Module):
    """CBAM with Multi-Head SE channel attention."""
    
    def __init__(self, channels, reduction=16, heads=4, kernel_size=7):
        super().__init__()
        self.ca = MHSEChannelAttention(channels, reduction, heads, fuse='concat')
        
        # Spatial attention
        padding = (kernel_size - 1) // 2
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.ca(x.unsqueeze(1))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([avg_out, max_out], dim=1))
        return x


# ===============================================================
# Upsampler / Downsampler without flattening
# ===============================================================

class TemporalUpsampler1D(nn.Module):
    """Upsamples along the temporal dimension while expanding channels."""
    def __init__(self, in_channels=32, out_channels=128, scale_factor=2, attention_heads=4):
        super().__init__()
        self.output_shape=(1,128,64)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False),
        )
        self.attention = CBAMWithMHSE(
            channels=self.output_shape[0], 
            reduction=16, 
            heads=min(attention_heads, self.output_shape[0]),
            kernel_size=3
        )

    def forward(self, x):
        # x: (B, 32, 32)
        x = self.net(x)         # → (B, 128, 64)
        x = self.attention(x)
        return x


class TemporalDownsampler1D(nn.Module):
    """Downsamples along the temporal dimension and reduces channels."""
    def __init__(self, in_channels=128, out_channels=32, scale_factor=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=1/scale_factor, mode="linear", align_corners=False)
        )

    def forward(self, x):
        # x: (B, 128, 64)
        x = self.net(x)         # → (B, 32, 32)
        return x


# ===============================================================
# Full model: Upsampler → UNet → Downsampler
# ===============================================================

class UNetDiffuser(nn.Module):
    """
    Takes (B, 32, 32), upsamples to (B, 128, 64),
    processes with UNet1D, downsamples to (B, 32, 32).
    """

    def __init__(self, inp_size=32, num_channels=32):
        super().__init__()
        self.upsampler = TemporalUpsampler1D(num_channels, 128, scale_factor=3)

        self.unet = UNet1DModel(
            sample_size=64,
            in_channels=128,
            out_channels=128,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
            use_timestep_embedding=True,
            act_fn="silu",
        )

        self.downsampler = TemporalDownsampler1D(128, num_channels, scale_factor=3)

    def forward(self, x, timesteps):
        # Input: (B, 32, 32)
        latent = self.upsampler(x)                      # (B, 128, 64)
        latent_out = self.unet(latent.squeeze(1), timesteps).sample  # (B, 128, 64)
        out = self.downsampler(latent_out)              # (B, 32, 32)
        return out
    
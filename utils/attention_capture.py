"""Helpers to capture attention weights from Attention / CrossAttentionLayer modules.

Usage:
    captured = enable_capture(backbone)
    # ... run forward ...
    weights = collect(captured)   # {qualified_name: tensor}
"""
from typing import Dict
import torch
import torch.nn as nn

from diffusion_forcing_transformer.dit_blocks import (
    Attention,
    CrossAttentionLayer,
)
from diffusion_forcing_transformer.dit1d_dual_grouped import _GroupCrossAttn


CAPTURABLE = (Attention, CrossAttentionLayer, _GroupCrossAttn)


def enable_capture(model: nn.Module) -> Dict[str, nn.Module]:
    """Flip `_capture_attn=True` on every capturable attention module.

    Returns a dict {qualified_name: module} for use with `collect`.
    """
    captured: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, CAPTURABLE):
            mod._capture_attn = True
            mod._last_attn = None
            captured[name] = mod
    return captured


def disable_capture(captured: Dict[str, nn.Module]) -> None:
    for mod in captured.values():
        mod._capture_attn = False
        mod._last_attn = None


def collect(captured: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
    """Return current `_last_attn` from each captured module and clear it."""
    out: Dict[str, torch.Tensor] = {}
    for name, mod in captured.items():
        if mod._last_attn is not None:
            out[name] = mod._last_attn
            mod._last_attn = None
    return out

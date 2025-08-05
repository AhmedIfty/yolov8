# ultralytics/nn/modules_caf.py
"""
CAFBlock: Context-Aware Fusion for YOLOv8 Neck

This block:
 1) Builds a ResCBAM-Lite attention module for whatever channel count arrives.
 2) Learns a scalar alpha to fuse x and skip.
 3) Builds a Conv(c,c,k=3) to refine the result.
All of that happens on the *first* forward pass, so it works for any scale.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv  # Ultralytics’ own Conv

# A lightweight CBAM (channel + spatial) with residual-style skip
class ResCBAMLite(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.ch = ch
        self.reduction = reduction
        # We'll build these convs in __init__ so they get properly registered
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(ch // reduction, ch, 1, bias=False)
        )
        self.spat = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        ca = (self.mlp(self.avg(x) + self.max(x))).sigmoid()
        x = x * ca
        # Spatial attention
        sa = self.spat(torch.cat([
            x.mean(1, keepdim=True),
            x.max(1, keepdim=True)[0]
        ], 1)).sigmoid()
        return x * sa

class CAFBlock(nn.Module):
    """
    Dynamic Context-Aware Fusion block.
    On first forward, builds its internal attention, alpha, and conv
    to match x.shape[1]. Subsequent forwards reuse them.
    """
    def __init__(self, *args, reduction=16):
        super().__init__()
        # args from YAML are ignored entirely
        self.reduction = reduction
        self.attn = None
        self.alpha = None
        self.conv = None

    def forward(self, x, skip=None):
        # skip → if no neighbour provided (single-input call), just reuse x
        if skip is None:
            skip = x

        # On first call, build submodules with the correct channel count
        if self.attn is None:
            ch = x.shape[1]  # actual incoming channels
            self.attn = ResCBAMLite(ch, reduction=self.reduction).to(x.device)
            # learnable scalar (keep it on the same device)
            self.alpha = nn.Parameter(torch.ones(1, device=x.device))
            # 3×3 conv, same in/out channels
            self.conv = Conv(ch, ch, k=3, s=1).to(x.device)

        x_att = self.attn(x)
        fused = self.alpha * x_att + (1.0 - self.alpha) * skip
        return self.conv(fused)

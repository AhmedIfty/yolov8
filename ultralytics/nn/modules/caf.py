# ultralytics/nn/modules/caf.py
"""Context-Aware Fusion modules for YOLOv8."""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from ultralytics.nn.modules import Conv, C2f
from .block import C2f
from .conv import Conv


# class ResCBAMLite(nn.Module):
#     """
#     Lightweight Residual CBAM (Convolutional Block Attention Module).
#
#     Combines channel and spatial attention with residual connection for
#     better feature refinement without losing information.
#     """
#
#     def __init__(self, ch, reduction=16):
#         """
#         Initialize ResCBAMLite.
#
#         Args:
#             ch (int): Number of input/output channels.
#             reduction (int): Reduction ratio for channel attention MLP.
#         """
#         super().__init__()
#         # Channel attention components
#         self.avg = nn.AdaptiveAvgPool2d(1)
#         self.max = nn.AdaptiveMaxPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(ch, ch // reduction, 1, bias=False),
#             nn.SiLU(),  # Using SiLU (Swish) for YOLOv8 consistency
#             nn.Conv2d(ch // reduction, ch, 1, bias=False),
#         )
#
#         # Spatial attention component
#         self.spat = nn.Conv2d(2, 1, 7, padding=3, bias=False)
#
#     def forward(self, x):
#         """
#         Forward pass with residual CBAM-lite attention.
#
#         Args:
#             x (torch.Tensor): Input tensor [B, C, H, W].
#
#         Returns:
#             torch.Tensor: Output with residual attention applied.
#         """
#         # Channel attention
#         ca = self.mlp(self.avg(x) + self.max(x)).sigmoid()
#         x_ca = x * ca
#
#         # Spatial attention
#         sa = self.spat(
#             torch.cat([x_ca.mean(1, True), x_ca.max(1, True)[0]], 1)
#         ).sigmoid()
#
#         # Residual connection - critical for preserving information
#         return x + x_ca * sa

"""
# math corrected ResCBAMLite class by chatGPT

class ResCBAMLite(nn.Module):
    # CBAM-lite: shared MLP applied separately to avg/max, then summed (+ residual)
    def __init__(self, ch: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        mid = max(ch // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, ch, 1, bias=False),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = (self.mlp(self.avg(x)) + self.mlp(self.max(x))).sigmoid()
        x_ca = x * ca
        sa = self.spatial(torch.cat((x_ca.mean(1, keepdim=True),
                                     x_ca.max(1, keepdim=True)[0]), 1)).sigmoid()
        return x + x_ca * sa
"""

class ResCBAM(nn.Module):
    """Standard CBAM (channel + spatial) with residual, YOLOv8-friendly."""
    def __init__(self, ch: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        mid = max(ch // reduction, 1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, ch, 1, bias=False),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention: MLP(avg) + MLP(max)
        ca = (self.mlp(self.avg(x)) + self.mlp(self.max(x))).sigmoid()
        x_ca = x * ca
        # Spatial attention
        sa = self.spatial(torch.cat((x_ca.mean(1, keepdim=True),
                                     x_ca.max(1, keepdim=True)[0]), 1)).sigmoid()
        return x + x_ca * sa  # residual

class C2fCAF(nn.Module):
    """
    C2f with Context-Aware Fusion.

    Drop-in replacement for C2f in YAML that adds attention mechanism
    while preserving the full capacity of the original C2f module.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        """
        Initialize C2fCAF.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            reduction (int): Reduction ratio for attention module.
        """
        super().__init__()
        self.c2f = C2f(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        # self.attn = ResCBAMLite(c2, reduction=reduction)
        self.attn = ResCBAM(c2, reduction=reduction)  # use proven ResCBAM first

    def forward(self, x):
        """
        Forward pass through C2f followed by attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output with attention applied to C2f features.
        """
        y = self.c2f(x)
        return self.attn(y)


class BiFPNAdd(nn.Module):
    """
    BiFPN-style learned weighted fusion (Optional - for Option B).

    Replaces simple concatenation with learnable weighted sum.
    """

    def __init__(self, num_inputs=2, eps=1e-4):
        """
        Initialize BiFPNAdd.

        Args:
            num_inputs (int): Number of input tensors to fuse.
            eps (float): Small epsilon for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(num_inputs))

    def forward(self, xs):
        """
        Forward pass with learned weighted fusion.

        Args:
            xs (list): List of tensors to fuse.

        Returns:
            torch.Tensor: Weighted sum of input tensors.
        """
        # Ensure all inputs have same spatial dimensions
        h, w = xs[0].shape[-2:]
        aligned = []
        for x in xs:
            if x.shape[-2:] != (h, w):
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            aligned.append(x)

        # Apply learned weights (with ReLU and normalization)
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)

        # Weighted sum
        out = w[0] * aligned[0]
        for i in range(1, len(aligned)):
            out = out + w[i] * aligned[i]

        return out
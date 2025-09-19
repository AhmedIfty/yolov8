
# ultralytics/nn/modules/weighted_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSumFusion(nn.Module):
    """Learnable normalized sum of 2 inputs; both are projected to cout channels."""
    def __init__(self, c0: int, c1: int, cout: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))  # start equal
        # projections (defined here so they're registered with the optimizer)
        self.p0 = nn.Identity() if c0 == cout else nn.Conv2d(c0, cout, 1, bias=False)
        self.p1 = nn.Identity() if c1 == cout else nn.Conv2d(c1, cout, 1, bias=False)
        if hasattr(self.p0, "weight"):
            nn.init.xavier_uniform_(self.p0.weight)
        if hasattr(self.p1, "weight"):
            nn.init.xavier_uniform_(self.p1.weight)

    def forward(self, xs):
        x0, x1 = xs  # same HxW (you upsampled already)
        x0 = self.p0(x0); x1 = self.p1(x1)
        w = F.relu(self.w); w = w / (w.sum() + self.eps)
        return w[0] * x0 + w[1] * x1


# # ultralytics/nn/modules/weighted_fusion.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class WeightedSumFusion(nn.Module):
#     """
#     Learnable weighted sum of 2 feature maps.
#     Automatically handles different channel counts.
#     """
#
#     def __init__(self, eps=1e-4):
#         super().__init__()
#         # Learnable weights for fusion
#         self.w = nn.Parameter(torch.tensor([0.45, 0.55], dtype=torch.float32))
#         self.eps = eps
#         # Projection layers (built lazily on first forward)
#         self.align_channels = None
#
#     def forward(self, x):
#         """
#         Args:
#             x: List of 2 tensors (can have different channels)
#         Returns:
#             Weighted sum with larger channel count
#         """
#         if not isinstance(x, list) or len(x) != 2:
#             return x[0] if isinstance(x, list) else x
#
#         x0, x1 = x[0], x[1]
#
#         # On first forward, build alignment layer if needed
#         if self.align_channels is None:
#             c0, c1 = x0.shape[1], x1.shape[1]
#             if c0 != c1:
#                 # Project smaller to match larger
#                 if c0 > c1:
#                     self.align_channels = nn.Conv2d(c1, c0, 1, bias=False).to(x0.device)
#                     self.out_channels = c0
#                     self.project_second = True
#                 else:
#                     self.align_channels = nn.Conv2d(c0, c1, 1, bias=False).to(x0.device)
#                     self.out_channels = c1
#                     self.project_second = False
#                 # Initialize for information preservation
#                 nn.init.xavier_uniform_(self.align_channels.weight)
#             else:
#                 self.out_channels = c0
#                 self.project_second = None
#
#         # Apply projection if needed
#         if self.align_channels is not None:
#             if self.project_second:
#                 x1 = self.align_channels(x1)
#             else:
#                 x0 = self.align_channels(x0)
#
#         # Normalize weights with ReLU for stability
#         w = F.relu(self.w)
#         w = w / (w.sum() + self.eps)
#
#         return w[0] * x0 + w[1] * x1
#
#
# class WeightedConcat(nn.Module):
#     """
#     Alternative: Weighted concatenation that preserves original behavior.
#     Concatenates with learnable channel-wise weights.
#     """
#
#     def __init__(self, dimension=1):
#         super().__init__()
#         self.d = dimension
#         self.weight_scale = None
#
#     def forward(self, x):
#         if not isinstance(x, list):
#             return x
#
#         # Initialize weights on first forward
#         if self.weight_scale is None:
#             total_channels = sum(xi.shape[self.d] for xi in x)
#             self.weight_scale = nn.Parameter(torch.ones(total_channels, 1, 1))
#             self.weight_scale = self.weight_scale.to(x[0].device)
#
#         # Concatenate
#         out = torch.cat(x, self.d)
#
#         # Apply channel-wise scaling
#         return out * self.weight_scale.view(1, -1, 1, 1)
#
#
# class AdaptiveWeightedFusion(nn.Module):
#     """
#     Alternative: Spatially-adaptive weighted fusion.
#     Learns position-dependent mixing weights.
#     """
#
#     def __init__(self, channels):
#         super().__init__()
#         # Spatial attention for weight generation
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels * 2, channels // 8, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 8, 2, 1),  # 2 weights per position
#             nn.Softmax(dim=1)  # Normalize across the 2 inputs
#         )
#
#     def forward(self, x):
#         """
#         Args:
#             x: List of 2 tensors with same shape [B, C, H, W]
#         Returns:
#             Adaptively weighted sum [B, C, H, W]
#         """
#         if isinstance(x, list):
#             assert len(x) == 2, f"AdaptiveWeightedFusion expects exactly 2 inputs, got {len(x)}"
#             x0, x1 = x[0], x[1]
#         else:
#             return x
#
#         # Compute spatial weights
#         concat = torch.cat([x0, x1], dim=1)
#         weights = self.conv(concat)  # [B, 2, H, W]
#
#         # Apply weights
#         w0 = weights[:, 0:1, :, :]  # [B, 1, H, W]
#         w1 = weights[:, 1:2, :, :]  # [B, 1, H, W]
#
#         return w0 * x0 + w1 * x1
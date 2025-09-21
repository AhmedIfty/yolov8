# ultralytics/nn/modules/cbam_residual.py
import torch
import torch.nn as nn
from .conv import CBAM  # your repo already provides ChannelAttention, SpatialAttention, CBAM(c1, kernel_size=7)

class ResidualCBAM(nn.Module):
    """
    Residual soft-gated CBAM.
    y = x * (alpha * CBAM(x) + (1 - alpha))
    - alpha starts at 0.5 and is learnable (you can freeze if desired).
    - CBAM here is your existing module (channel->spatial, multiplicative mask).
    """
    def __init__(self, c1, c2=None, ksize=7, alpha=0.5):
        super().__init__()
        self.cbam = CBAM(c1, kernel_size=ksize)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=True)

    def forward(self, x):
        a = self.cbam(x)                      # mask in [0,1], same shape as x
        return x * (self.alpha * a + (1.0 - self.alpha))

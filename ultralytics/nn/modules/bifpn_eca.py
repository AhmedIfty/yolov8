# ultralytics/nn/modules/bifpn_eca.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------
# Small building blocks
# ----------------------

class SeparableConv(nn.Module):
    """Depthwise + Pointwise conv, BN, SiLU. Channel-preserving."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class WeightedAdd(nn.Module):
    """Learnable normalized weighted sum (BiFPN-style).
    ReLU(w) then L1 normalization to keep weights positive and stable.
    """

    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, xs):
        # xs: list[Tensor] with same shape (B, C, H, W)
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = 0
        for i, xi in enumerate(xs):
            out = out + w[i] * xi
        return out


class ECA(nn.Module):
    """Efficient Channel Attention: 1D conv on pooled channels, very lightweight."""

    def __init__(self, c: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        t = int(abs((math.log2(c) / gamma) + b))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        y = F.adaptive_avg_pool2d(x, 1)  # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B,1,C] via 1D conv
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # [B,C,1,1]
        return x * y.expand_as(x)


# ----------------------
# Utilities
# ----------------------

def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor upsample to ref's HxW (stable for detection)."""
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode='nearest')


def _downsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Downsample by stride-2 maxpool until spatial size matches ref."""
    h, w = x.shape[2], x.shape[3]
    hr, wr = ref.shape[2], ref.shape[3]
    y = x
    # Typical case is exactly 2x; loop keeps this robust.
    while (y.shape[2] > hr) or (y.shape[3] > wr):
        y = F.max_pool2d(y, kernel_size=2, stride=2)
    if y.shape[2:] != ref.shape[2:]:
        # If one step of pooling overshot due to odd sizes, do a safe interpolate
        y = F.interpolate(y, size=ref.shape[-2:], mode='nearest')
    return y


# ----------------------
# One-layer BiFPN + ECA
# ----------------------

class BiFPN_ECA(nn.Module):
    """
    True BiFPN layer with learnable normalized fusion and ECA gates.
    Expects a 4-level pyramid: [P2, P3, P4, P5] from indices.

    Args:
        channels: List of target channel counts for [P2, P3, P4, P5] outputs

    The module will automatically handle channel projection from input to target channels.
    """

    def __init__(self, channels=256):
        super().__init__()

        # Handle both single int and list inputs for channels
        if isinstance(channels, int):
            # Use same channels for all levels
            c2 = c3 = c4 = c5 = channels
        elif isinstance(channels, (list, tuple)) and len(channels) == 1:
            # If a single-element list is passed, use that value for all levels
            c2 = c3 = c4 = c5 = channels[0]
        elif isinstance(channels, (list, tuple)) and len(channels) == 4:
            c2, c3, c4, c5 = channels
        else:
            raise ValueError(f"channels must be int or list of 1 or 4 ints, got {channels}")

        # Store target channels
        self.target_channels = [c2, c3, c4, c5]

        # Channel projection layers (will be initialized in forward based on input channels)
        self.input_projs = nn.ModuleList()
        self.input_projs_initialized = False

        # Top-down fusion ops
        self.p4_td_w = WeightedAdd(2)
        self.p4_td_c = SeparableConv(c4)
        self.p4_td_eca = ECA(c4)

        self.p3_td_w = WeightedAdd(2)
        self.p3_td_c = SeparableConv(c3)
        self.p3_td_eca = ECA(c3)

        self.p2_td_w = WeightedAdd(2)
        self.p2_td_c = SeparableConv(c2)
        self.p2_td_eca = ECA(c2)

        # Bottom-up fusion ops
        self.p3_out_w = WeightedAdd(3)
        self.p3_out_c = SeparableConv(c3)
        self.p3_out_eca = ECA(c3)

        self.p4_out_w = WeightedAdd(3)
        self.p4_out_c = SeparableConv(c4)
        self.p4_out_eca = ECA(c4)

        self.p5_out_w = WeightedAdd(2)
        self.p5_out_c = SeparableConv(c5)
        self.p5_out_eca = ECA(c5)

        # Optional refinement for P5 start
        self.p5_refine = SeparableConv(c5)

    def _init_input_projs(self, xs):
        """Initialize input projection layers based on actual input channels."""
        if not self.input_projs_initialized:
            for i, x in enumerate(xs):
                in_channels = x.shape[1]
                out_channels = self.target_channels[i]
                if in_channels != out_channels:
                    # Add 1x1 conv to project channels
                    proj = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                else:
                    # Identity if channels match
                    proj = nn.Identity()
                self.input_projs.append(proj.to(xs[0].device))
            self.input_projs_initialized = True

    def forward(self, xs):
        # Handle both list input and single tensor input from Ultralytics
        if not isinstance(xs, (list, tuple)):
            # If single tensor, it should be from Concat or similar
            # This shouldn't happen with our YAML config, but handle it
            raise ValueError("BiFPN_ECA expects a list of 4 tensors [P2, P3, P4, P5]")

        if len(xs) != 4:
            raise ValueError(f"BiFPN_ECA expects exactly 4 features, got {len(xs)}")

        # Initialize input projections on first forward pass
        self._init_input_projs(xs)

        # Project inputs to target channels
        P2 = self.input_projs[0](xs[0])
        P3 = self.input_projs[1](xs[1])
        P4 = self.input_projs[2](xs[2])
        P5 = self.input_projs[3](xs[3])

        # Top-down pathway
        P5_t = self.p5_refine(P5)

        P4_td = self.p4_td_w([P4, _upsample_to(P5_t, P4)])
        P4_td = self.p4_td_eca(self.p4_td_c(P4_td))

        P3_td = self.p3_td_w([P3, _upsample_to(P4_td, P3)])
        P3_td = self.p3_td_eca(self.p3_td_c(P3_td))

        P2_td = self.p2_td_w([P2, _upsample_to(P3_td, P2)])
        P2_o = self.p2_td_eca(self.p2_td_c(P2_td))  # final P2

        # Bottom-up pathway
        P3_o = self.p3_out_w([P3, P3_td, _downsample_to(P2_o, P3)])
        P3_o = self.p3_out_eca(self.p3_out_c(P3_o))

        P4_o = self.p4_out_w([P4, P4_td, _downsample_to(P3_o, P4)])
        P4_o = self.p4_out_eca(self.p4_out_c(P4_o))

        P5_o = self.p5_out_w([P5, _downsample_to(P4_o, P5)])
        P5_o = self.p5_out_eca(self.p5_out_c(P5_o))

        # return [P2_o, P3_o, P4_o, P5_o]

        outputs = [P2_o, P3_o, P4_o, P5_o]

        # IMPORTANT: For Detect head compatibility
        # Detect expects a list during training but might handle differently during export
        return outputs  # Return as list for multi-scale detection
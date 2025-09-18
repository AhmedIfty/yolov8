# ultralytics/nn/modules/bifpn_eca.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv(nn.Module):
    """Depthwise + Pointwise conv with BN and SiLU."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, 1, 1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class WeightedAdd(nn.Module):
    """Learnable weighted sum with ReLU + L1 normalization."""

    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, xs):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = 0
        for i, xi in enumerate(xs):
            out = out + w[i] * xi
        return out


class ECA(nn.Module):
    """Efficient Channel Attention with residual gating."""

    def __init__(self, channels, k=None, alpha=0.5, use_eca=True):
        super().__init__()
        self.use_eca = use_eca
        self.alpha = alpha  # 0=identity, 1=full gate

        if k is None:
            # Adaptive kernel size based on channel count
            k = int(round(math.log2(channels) + 1))
            if k % 2 == 0:
                k += 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if not self.use_eca:
            return x

        # Channel attention
        y = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]

        # Residual gating: x * (alpha * gate + (1 - alpha))
        return x * (self.alpha * y + (1 - self.alpha))


def _upsample_to(x, ref):
    """Upsample x to match ref's spatial dimensions."""
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode='nearest')


def _downsample_to(x, ref):
    """Downsample x to match ref's spatial dimensions."""
    h, w = x.shape[2:]
    hr, wr = ref.shape[2:]
    while h > hr or w > wr:
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        h, w = x.shape[2:]
    if x.shape[2:] != ref.shape[2:]:
        x = F.interpolate(x, size=ref.shape[2:], mode='nearest')
    return x


class BiFPN_ECA(nn.Module):
    """BiFPN refiner layer to be added AFTER PAN."""

    def __init__(self, c=192, use_eca=False, eca_alpha=0.5):
        super().__init__()
        self.c = c
        self.use_eca = use_eca if isinstance(use_eca, bool) else (use_eca[0] if isinstance(use_eca, list) else False)
        self.eca_alpha = eca_alpha

        self.proj = nn.ModuleList()
        self.proj_built = False

        # Top-down fusion
        self.p4_td_w = WeightedAdd(2)
        self.p4_td_c = SeparableConv(c, c)
        self.p3_td_w = WeightedAdd(2)
        self.p3_td_c = SeparableConv(c, c)
        self.p2_td_w = WeightedAdd(2)
        self.p2_td_c = SeparableConv(c, c)

        # Bottom-up fusion
        self.p3_out_w = WeightedAdd(3)
        self.p3_out_c = SeparableConv(c, c)
        self.p4_out_w = WeightedAdd(3)
        self.p4_out_c = SeparableConv(c, c)
        self.p5_out_w = WeightedAdd(2)
        self.p5_out_c = SeparableConv(c, c)

        # ECA modules for each output
        self.eca2 = ECA(c, alpha=eca_alpha, use_eca=self.use_eca)
        self.eca3 = ECA(c, alpha=eca_alpha, use_eca=self.use_eca)
        self.eca4 = ECA(c, alpha=eca_alpha, use_eca=self.use_eca)
        self.eca5 = ECA(c, alpha=eca_alpha, use_eca=self.use_eca)

        self._init_fusion_weights()

    def _init_fusion_weights(self):
        """Initialize fusion weights to bias away from P2."""
        with torch.no_grad():
            self.p4_td_w.w[:] = torch.tensor([0.3, 0.7])  # [P4, up(P5)]
            self.p3_td_w.w[:] = torch.tensor([0.3, 0.7])  # [P3, up(P4_td)]
            self.p2_td_w.w[:] = torch.tensor([0.3, 0.7])  # [P2, up(P3_td)]
            self.p3_out_w.w[:] = torch.tensor([0.4, 0.5, 0.1])  # [P3, P3_td, down(P2)]
            self.p4_out_w.w[:] = torch.tensor([0.4, 0.5, 0.1])  # [P4, P4_td, down(P3)]
            self.p5_out_w.w[:] = torch.tensor([0.6, 0.4])  # [P5, down(P4)]

    def _build_projections(self, in_channels):
        """Build input projection layers based on actual input channels."""
        if not self.proj_built:
            for ch in in_channels:
                self.proj.append(nn.Conv2d(ch, self.c, 1, bias=False))
            self.proj_built = True

    def forward(self, x):
        """Forward pass for a list of 4 tensors [P2, P3, P4, P5] from PAN."""
        features = x if isinstance(x, list) else [x]
        if len(features) != 4:
            raise ValueError(f"BiFPN_ECA expects exactly 4 features, got {len(features)}")

        in_channels = [f.shape[1] for f in features]
        self._build_projections(in_channels)

        # Project to internal channels
        p2, p3, p4, p5 = [proj(f).to(features[0].device) for proj, f in zip(self.proj, features)]

        # Top-down pathway
        p4_td = self.p4_td_c(self.p4_td_w([p4, _upsample_to(p5, p4)]))
        p3_td = self.p3_td_c(self.p3_td_w([p3, _upsample_to(p4_td, p3)]))
        p2_td = self.p2_td_c(self.p2_td_w([p2, _upsample_to(p3_td, p2)]))

        # Bottom-up pathway
        p2_out = self.eca2(p2_td)
        p3_out = self.eca3(self.p3_out_c(self.p3_out_w([p3, p3_td, _downsample_to(p2_out, p3)])))
        p4_out = self.eca4(self.p4_out_c(self.p4_out_w([p4, p4_td, _downsample_to(p3_out, p4)])))
        p5_out = self.eca5(self.p5_out_c(self.p5_out_w([p5, _downsample_to(p4_out, p5)])))

        return [p2_out, p3_out, p4_out, p5_out]


class BiFPNIndex(nn.Module):
    """Extract a specific index from a list output."""

    def __init__(self, index=0):
        super().__init__()
        self.index = index[1] if isinstance(index, list) and len(index) > 1 else (
            index[0] if isinstance(index, list) else index)

    def forward(self, x):
        """Extract tensor at specified index."""
        return x[self.index] if isinstance(x, list) else x
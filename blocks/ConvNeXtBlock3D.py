import torch
import torch.nn as nn

class LayerNorm3d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + \
            self.bias[:, None, None, None]
        return x


class GRN3d(nn.Module):
    """
    Global Response Normalization for 3D feature maps.
    Prevents feature collapse by normalizing each channel
    relative to its global L2 norm across spatial dims.
    From ConvNeXt V2 (Woo et al., 2023).
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        # global L2 norm per channel: (B, C, 1, 1, 1)
        gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        # normalize by mean norm across channels
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXt3DBlock(nn.Module):
    """
    ConvNeXt V2 style block with GRN for 3D medical imaging.
    Safe for batch size 1 — uses LayerNorm3d not BatchNorm.
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        # depthwise large kernel conv
        self.dwconv  = nn.Conv3d(dim, dim, kernel_size=7,
                                 padding=3, groups=dim)
        self.norm    = LayerNorm3d(dim, eps=1e-6)
        # inverted bottleneck
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act     = nn.GELU()
        # GRN replaces the standard mid-block normalization
        self.grn     = GRN3d(4 * dim)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        # layer scale for gradient stability in deep networks
        self.gamma   = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)          # applied after activation, before compression
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma[:, None, None, None]
        return shortcut + x
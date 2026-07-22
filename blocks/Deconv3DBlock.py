from torch import nn as nn
from blocks.SingleConv3DBlock import SingleConv3DBlock
from blocks.SingleDeconv3DBlock import SingleDeconv3DBlock


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.GroupNorm(8, out_planes),   # safe for batch size 1
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
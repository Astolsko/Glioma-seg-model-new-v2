from torch import nn as nn
from blocks.SingleConv3DBlock import SingleConv3DBlock


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.GroupNorm(8, out_planes),   # 8 groups, works at batch size 1
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
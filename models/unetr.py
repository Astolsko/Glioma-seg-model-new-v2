import torch
import torch.nn as nn

from config import cfg
from blocks.Conv3DBlock import Conv3DBlock
from blocks.Deconv3DBlock import Deconv3DBlock
from blocks.CoordAtt3D import CoordAtt3D
from blocks.Transformer import Transformer, DilatedBottleneck
from blocks.SingleDeconv3DBlock import SingleDeconv3DBlock
from blocks.SingleConv3DBlock import SingleConv3DBlock
from blocks.ConvNeXtBlock3D import ConvNeXt3DBlock


class BidirectionalSkip(nn.Module):
    """
    Replaces standard concatenation skip connections with weighted
    bidirectional fusion. Shallow encoder features and deep decoder
    features are weighted separately before fusion, allowing the
    network to learn how much each contributes at each scale.
    """
    def __init__(self, channels):
        super().__init__()
        # learnable scalar weights — initialized to 1 so at start it
        # behaves like equal weighting, then learns during training
        self.w_shallow = nn.Parameter(torch.ones(1))
        self.w_deep = nn.Parameter(torch.ones(1))
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, shallow, deep):
        # sigmoid keeps weights in (0,1) — stable training
        w1 = torch.sigmoid(self.w_shallow)
        w2 = torch.sigmoid(self.w_deep)
        fused = torch.cat([w1 * shallow, w2 * deep], dim=1)
        return self.fuse(fused)


class UNETR(nn.Module):
    def __init__(
        self,
        img_shape=cfg.unetr.img_shape,
        input_dim=cfg.unetr.input_dim,
        output_dim=cfg.unetr.output_dim,
        embed_dim=cfg.unetr.embed_dim,
        patch_size=cfg.unetr.patch_size,
        num_heads=cfg.unetr.num_heads,
        dropout=cfg.unetr.dropout,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = cfg.unetr.num_layers
        self.ext_layers = cfg.unetr.extract_layers

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = Transformer(
            input_dim=input_dim,
            embed_dim=embed_dim,
            cube_size=img_shape,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=self.num_layers,
            dropout=dropout,
            extract_layers=self.ext_layers
        )
        self.dilated_bottleneck = DilatedBottleneck(in_channels=512)

        # U-Net Decoder
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3),
            Conv3DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim, 256),
            Deconv3DBlock(256, 128),
            Deconv3DBlock(128, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim, 512),
            Deconv3DBlock(512, 256),
        )

        self.decoder9 = Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512),
            ConvNeXt3DBlock(512),
            CoordAtt3D(512),
            SingleDeconv3DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256),
            ConvNeXt3DBlock(256),
            CoordAtt3D(256),
            SingleDeconv3DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128),
            ConvNeXt3DBlock(128),
            CoordAtt3D(128),
            SingleDeconv3DBlock(128, 64)
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64),
            Conv3DBlock(64, 64),
            SingleConv3DBlock(64, output_dim, 1)
        )

        self.aux_head_z6 = nn.Sequential(
            nn.Conv3d(128, 3, kernel_size=1),   # z6 decoder outputs 128ch at 24^3
            nn.Upsample(size=cfg.unetr.img_shape, mode='trilinear', align_corners=False)
        )
        self.aux_head_z3 = nn.Sequential(
            nn.Conv3d(64, 3, kernel_size=1),    # z3 decoder outputs 64ch at 48^3
            nn.Upsample(size=cfg.unetr.img_shape, mode='trilinear', align_corners=False)
        )

        self.skip9 = BidirectionalSkip(512)   # for z9 level
        self.skip6 = BidirectionalSkip(256)   # for z6 level
        self.skip3 = BidirectionalSkip(128)   # for z3 level

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z12 = self.dilated_bottleneck(z12)        # multi-scale bottleneck

        z9_enc = self.decoder9(z9)
        z9_fused = self.skip9(z9_enc, z12)
        z9 = self.decoder9_upsampler(
            torch.cat([z9_fused, z12], dim=1)
        )

        z6_enc = self.decoder6(z6)
        z6_fused = self.skip6(z6_enc, z9)
        z6_out = self.decoder6_upsampler(
            torch.cat([z6_fused, z9], dim=1)
        )                                         # (B, 128, 24, 24, 24)

        z3_enc = self.decoder3(z3)
        z3_fused = self.skip3(z3_enc, z6_out)
        z3_out = self.decoder3_upsampler(
            torch.cat([z3_fused, z6_out], dim=1)
        )                                         # (B, 64, 48, 48, 48)

        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3_out], dim=1))

        if self.training:
            aux_z6 = self.aux_head_z6(z6_out)
            aux_z3 = self.aux_head_z3(z3_out)
            return output, aux_z6, aux_z3
        return output

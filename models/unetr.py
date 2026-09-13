"""Modified UNETR for glioma segmentation, with a swappable encoder.

    encoder="vit"    the original UNETR ViT (blocks/Transformer.py): 12 layers
                     over 16^3 patch tokens; layers 3/6/9/12 are reshaped to the
                     1/16 grid and deconvolved up to the decoder's skip scales.
    encoder="mamba"  SegMamba-style hierarchical Vision Mamba
                     (blocks/VisionMamba.py): four stages that already sit at
                     the decoder's skip scales.

Everything after the encoder (dilated bottleneck, bidirectional skips,
ConvNeXt/CoordAtt upsamplers, output header, deep-supervision heads) is the
same module graph for both, so a "vit" run and a "mamba" run differ in the
encoder only. What each encoder hands the shared decoder:

    decoder input            vit                            mamba
    1/2 res, 128 ch  (z3)    layer 3  -> decoder3 (3 up)    stage 1 (48)  -> mamba_skip3
    1/4 res, 256 ch  (z6)    layer 6  -> decoder6 (2 up)    stage 2 (96)  -> mamba_skip6
    1/8 res, 512 ch  (z9)    layer 9  -> decoder9 (1 up)    stage 3 (192) -> mamba_skip9
    1/8 res, 512 ch  (z12)   layer 12 -> decoder12_upsampler  stage 4 (384) -> mamba_hidden
                                                            (768) -> decoder12_upsampler

The mamba_skip*/mamba_hidden blocks are SegMamba's own skip encoders (its
encoder2..encoder5: MONAI UnetrBasicBlock, 3x3x3 residual, instance norm), and
mamba_hidden's 768 channels are SegMamba's hidden_size, which equals the ViT's
embed_dim, so decoder12_upsampler is literally the same layer shape in both.

The ViT path is byte-for-byte the pre-switch model: same attribute names, same
construction order, same parameters, so earlier checkpoints still load.
"""
import torch
import torch.nn as nn
from monai.networks.blocks.unetr_block import UnetrBasicBlock

from config import cfg
from blocks.Conv3DBlock import Conv3DBlock
from blocks.Deconv3DBlock import Deconv3DBlock
from blocks.CoordAtt3D import CoordAtt3D
from blocks.Transformer import Transformer, DilatedBottleneck
from blocks.SingleDeconv3DBlock import SingleDeconv3DBlock
from blocks.SingleConv3DBlock import SingleConv3DBlock
from blocks.ConvNeXtBlock3D import ConvNeXt3DBlock
from blocks.VisionMamba import VisionMambaEncoder

ENCODERS = ("vit", "mamba")


def _segmamba_skip_block(in_channels, out_channels):
    """SegMamba's per-scale skip encoder (encoder2..encoder5 in segmamba.py)."""
    return UnetrBasicBlock(spatial_dims=3, in_channels=in_channels,
                           out_channels=out_channels, kernel_size=3, stride=1,
                           norm_name="instance", res_block=True)


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
        encoder=None,
        mamba_kwargs=None,
    ):
        super().__init__()
        # Read at call time (None sentinel), not bound at import like the
        # defaults above, so a patched/overridden cfg is always honoured.
        encoder = encoder if encoder is not None else cfg.unetr.get("encoder", "vit")
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
        self.encoder_type = encoder
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

        if encoder == "vit":
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
        else:
            if any(int(s) % 16 for s in img_shape):
                raise ValueError(f"the Mamba encoder downsamples 16x; img_shape {img_shape} "
                                 f"must be divisible by 16 on every axis")
            # Its deepest stage is at 1/16 — the same grid as the ViT's 16^3
            # patches — which is what the attention/XAI maps are drawn on.
            self.patch_dim = [int(s) // 16 for s in img_shape]
            mk = dict(cfg.mamba) if mamba_kwargs is None else dict(mamba_kwargs)
            self.mamba_encoder = VisionMambaEncoder(
                in_chans=input_dim, dims=mk["dims"], depths=mk["depths"],
                d_state=mk["d_state"], d_conv=mk["d_conv"], expand=mk["expand"],
                dropout=mk["dropout"],
            )
        self.dilated_bottleneck = DilatedBottleneck(in_channels=512)

        # U-Net Decoder
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3),
            Conv3DBlock(32, 64, 3)
        )

        if encoder == "vit":
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
        else:
            dims = self.mamba_encoder.dims
            self.mamba_skip3 = _segmamba_skip_block(dims[0], 128)        # 1/2 res
            self.mamba_skip6 = _segmamba_skip_block(dims[1], 256)        # 1/4 res
            self.mamba_skip9 = _segmamba_skip_block(dims[2], 512)        # 1/8 res
            self.mamba_hidden = _segmamba_skip_block(dims[3], embed_dim)  # 1/16 res

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
            nn.Conv3d(128, 3, kernel_size=1),   # z6 decoder outputs 128ch at 1/4 res
            nn.Upsample(size=cfg.unetr.img_shape, mode='trilinear', align_corners=False)
        )
        self.aux_head_z3 = nn.Sequential(
            nn.Conv3d(64, 3, kernel_size=1),    # z3 decoder outputs 64ch at 1/2 res
            nn.Upsample(size=cfg.unetr.img_shape, mode='trilinear', align_corners=False)
        )

        self.skip9 = BidirectionalSkip(512)   # for z9 level
        self.skip6 = BidirectionalSkip(256)   # for z6 level
        self.skip3 = BidirectionalSkip(128)   # for z3 level

    def encoder_module_names(self):
        """Top-level modules that make up the encoder (for param counts and
        the XAI randomisation cascade)."""
        return ["transformer"] if self.encoder_type == "vit" else ["mamba_encoder"]

    def _encode(self, x):
        """Encoder plus its projections into the shared decoder. Returns
        (z3 skip @1/2 128ch, z6 skip @1/4 256ch, z9 skip @1/8 512ch,
        bottleneck input @1/8 512ch)."""
        if self.encoder_type == "vit":
            z3, z6, z9, z12 = self.transformer(x)
            z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
            z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
            z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
            z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
            return (self.decoder3(z3), self.decoder6(z6), self.decoder9(z9),
                    self.decoder12_upsampler(z12))

        s1, s2, s3, s4 = self.mamba_encoder(x)
        return (self.mamba_skip3(s1), self.mamba_skip6(s2), self.mamba_skip9(s3),
                self.decoder12_upsampler(self.mamba_hidden(s4)))

    def forward(self, x):
        z3_enc, z6_enc, z9_enc, z12 = self._encode(x)

        z12 = self.dilated_bottleneck(z12)        # multi-scale bottleneck

        z9_fused = self.skip9(z9_enc, z12)
        z9 = self.decoder9_upsampler(
            torch.cat([z9_fused, z12], dim=1)
        )

        z6_fused = self.skip6(z6_enc, z9)
        z6_out = self.decoder6_upsampler(
            torch.cat([z6_fused, z9], dim=1)
        )                                         # (B, 128, 1/4 res)

        z3_fused = self.skip3(z3_enc, z6_out)
        z3_out = self.decoder3_upsampler(
            torch.cat([z3_fused, z6_out], dim=1)
        )                                         # (B, 64, 1/2 res)

        z0 = self.decoder0(x)
        output = self.decoder0_header(torch.cat([z0, z3_out], dim=1))

        if self.training:
            aux_z6 = self.aux_head_z6(z6_out)
            aux_z3 = self.aux_head_z3(z3_out)
            return output, aux_z6, aux_z3
        return output

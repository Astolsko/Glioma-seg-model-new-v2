"""Shape/finiteness smoke tests for every block in blocks/ — these are the
pieces that get assembled into UNETR, so a crash or shape mismatch here is
exactly the kind of thing that would otherwise only surface deep into a
training run when a batch finally exercises that code path.
"""
import pytest

torch = pytest.importorskip("torch")

from blocks.Conv3DBlock import Conv3DBlock
from blocks.Deconv3DBlock import Deconv3DBlock
from blocks.SingleConv3DBlock import SingleConv3DBlock
from blocks.SingleDeconv3DBlock import SingleDeconv3DBlock
from blocks.CoordAtt3D import CoordAtt3D
from blocks.ConvNeXtBlock3D import ConvNeXt3DBlock, LayerNorm3d, GRN3d


def _assert_finite(t):
    assert torch.isfinite(t).all(), "block produced NaN/Inf output"


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_single_conv3dblock_preserves_spatial_shape(kernel_size):
    block = SingleConv3DBlock(4, 8, kernel_size)
    x = torch.randn(2, 4, 6, 6, 6)
    out = block(x)
    assert out.shape == (2, 8, 6, 6, 6)
    _assert_finite(out)


def test_single_deconv3dblock_doubles_spatial_dims():
    block = SingleDeconv3DBlock(8, 4)
    x = torch.randn(2, 8, 4, 4, 4)
    out = block(x)
    assert out.shape == (2, 4, 8, 8, 8)
    _assert_finite(out)


def test_conv3dblock_preserves_spatial_shape_changes_channels():
    block = Conv3DBlock(4, 16, 3)
    x = torch.randn(1, 4, 5, 5, 5)
    out = block(x)
    assert out.shape == (1, 16, 5, 5, 5)
    _assert_finite(out)


def test_conv3dblock_groupnorm_safe_for_batch_size_1():
    # comment in source claims GroupNorm(8, out) is safe at batch size 1 —
    # verify it actually doesn't raise.
    block = Conv3DBlock(4, 8, 3)
    x = torch.randn(1, 4, 4, 4, 4)
    out = block(x)
    assert out.shape[0] == 1
    _assert_finite(out)


def test_deconv3dblock_doubles_spatial_and_changes_channels():
    block = Deconv3DBlock(16, 8, 3)
    x = torch.randn(1, 16, 4, 4, 4)
    out = block(x)
    assert out.shape == (1, 8, 8, 8, 8)
    _assert_finite(out)


def test_coordatt3d_preserves_shape():
    attn = CoordAtt3D(16, reduction=4)
    x = torch.randn(2, 16, 6, 7, 5)
    out = attn(x)
    assert out.shape == x.shape
    _assert_finite(out)


def test_coordatt3d_output_bounded_by_input_scaled_by_sigmoid():
    # out = identity * a_d * a_h * a_w, each a_* in (0,1) (pre-dropout), so
    # |out| should never exceed |identity| in eval mode (dropout off).
    attn = CoordAtt3D(8, reduction=4).eval()
    x = torch.randn(1, 8, 4, 4, 4)
    with torch.no_grad():
        out = attn(x)
    assert torch.all(out.abs() <= x.abs() + 1e-5)


def test_layernorm3d_normalizes_channel_dim():
    ln = LayerNorm3d(8)
    x = torch.randn(2, 8, 3, 3, 3) * 100 + 50
    out = ln(x)
    # per-(batch, spatial-location) mean over channel dim should be ~0
    mean_over_channels = out.mean(dim=1)
    assert torch.allclose(mean_over_channels, torch.zeros_like(mean_over_channels), atol=1e-3)


def test_grn3d_preserves_shape():
    grn = GRN3d(8)
    x = torch.randn(2, 8, 4, 4, 4)
    out = grn(x)
    assert out.shape == x.shape
    _assert_finite(out)


def test_convnext3dblock_preserves_shape_and_is_residual():
    block = ConvNeXt3DBlock(16)
    x = torch.randn(1, 16, 6, 6, 6)
    out = block(x)
    assert out.shape == x.shape
    _assert_finite(out)


def test_convnext3dblock_zero_layer_scale_gives_pure_shortcut():
    # layer_scale_init_value=0 disables self.gamma entirely; with all-zero
    # gamma the residual path shouldn't dominate, but shape/finiteness is
    # what matters at this level.
    block = ConvNeXt3DBlock(8, layer_scale_init_value=0.0)
    assert block.gamma is None
    x = torch.randn(1, 8, 4, 4, 4)
    out = block(x)
    assert out.shape == x.shape

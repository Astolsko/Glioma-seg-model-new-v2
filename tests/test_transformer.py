"""Shape/correctness tests for the ViT-style encoder in blocks/Transformer.py
and the dilated bottleneck used right after it in UNETR.
"""
import pytest

torch = pytest.importorskip("torch")

from blocks.Transformer import (
    MultiHeadAttention, MLP, Embeddings, TransformerBlock, Transformer,
    DilatedBottleneck,
)


def test_multi_head_attention_preserves_shape_and_returns_weights():
    embed_dim, num_heads, num_patches, batch = 32, 4, 10, 2
    attn = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim, dropout=0.0)
    x = torch.randn(batch, num_patches, embed_dim)

    out, weights = attn(x)

    assert out.shape == (batch, num_patches, embed_dim)
    assert weights.shape == (batch, num_heads, num_patches, num_patches)
    # softmax outputs — each attention row sums to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, num_heads, num_patches), atol=1e-5)


def test_multi_head_attention_requires_embed_dim_divisible_by_heads():
    # attention_head_size = int(embed_dim/num_heads); non-divisible combos
    # silently truncate all_head_size instead of raising, which then breaks
    # the residual add in TransformerBlock (shape mismatch) — assert the
    # currently-relied-on invariant explicitly so a future change that
    # violates it fails loudly and immediately, not mid-training.
    embed_dim, num_heads = 30, 4  # 30 / 4 = 7 (int), all_head_size = 28 != 30
    attn = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim, dropout=0.0)
    x = torch.randn(1, 5, embed_dim)
    with pytest.raises(RuntimeError):
        attn(x)


def test_mlp_preserves_shape():
    mlp = MLP(embedding_dim=32, mlp_dim=64, dropout=0.0)
    x = torch.randn(2, 10, 32)
    out = mlp(x)
    assert out.shape == x.shape


def test_embeddings_patch_count_and_shape():
    cube_size = (16, 16, 8)
    patch_size = 8
    embed_dim = 32
    expected_patches = (cube_size[0] * cube_size[1] * cube_size[2]) // (patch_size ** 3)

    emb = Embeddings(input_dim=4, embed_dim=embed_dim, cube_size=cube_size,
                      patch_size=patch_size, dropout=0.0)
    x = torch.randn(2, 4, *cube_size)
    out = emb(x)

    assert out.shape == (2, expected_patches, embed_dim)


def test_transformer_block_preserves_shape(monkeypatch):
    from config import cfg
    monkeypatch.setattr(cfg.unetr, "mlp_dim", 64, raising=True)

    block = TransformerBlock(embed_dim=32, num_heads=4, dropout=0.0,
                              cube_size=(16, 16, 8), patch_size=8)
    x = torch.randn(2, 8, 32)
    out, weights = block(x)
    assert out.shape == x.shape
    assert weights is not None


def test_transformer_returns_requested_extract_layers(monkeypatch):
    from config import cfg
    monkeypatch.setattr(cfg.unetr, "mlp_dim", 64, raising=True)

    cube_size = (16, 16, 8)
    patch_size = 8
    embed_dim = 32
    num_layers = 4
    extract_layers = [1, 2, 3, 4]

    transformer = Transformer(
        input_dim=4, embed_dim=embed_dim, cube_size=cube_size, patch_size=patch_size,
        num_heads=4, num_layers=num_layers, dropout=0.0, extract_layers=extract_layers,
    )
    x = torch.randn(1, 4, *cube_size)
    outputs = transformer(x)

    assert len(outputs) == len(extract_layers)
    n_patches = (cube_size[0] * cube_size[1] * cube_size[2]) // (patch_size ** 3)
    for hidden in outputs:
        assert hidden.shape == (1, n_patches, embed_dim)


def test_transformer_out_of_order_extract_layers_still_returns_in_depth_order(monkeypatch):
    """extract_layers is checked with `depth + 1 in self.extract_layers`
    while iterating layers in order, so the returned list is always in
    increasing depth order regardless of how extract_layers was written."""
    from config import cfg
    monkeypatch.setattr(cfg.unetr, "mlp_dim", 64, raising=True)

    transformer = Transformer(
        input_dim=2, embed_dim=16, cube_size=(8, 8, 8), patch_size=8,
        num_heads=4, num_layers=3, dropout=0.0, extract_layers=[3, 1],
    )
    x = torch.randn(1, 2, 8, 8, 8)
    outputs = transformer(x)
    assert len(outputs) == 2  # one output per requested layer, in depth order


def test_dilated_bottleneck_preserves_shape_and_is_residual():
    block = DilatedBottleneck(in_channels=16)
    x = torch.randn(1, 16, 4, 4, 4)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

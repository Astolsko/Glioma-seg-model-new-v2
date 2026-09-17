"""Full UNETR model tests. Uses a monkeypatched tiny config (see
tiny_unetr_kwargs/tiny_unetr fixtures in conftest.py) so the entire forward
pass — transformer encoder, all three decoder branches, bidirectional skips,
CoordAtt3D/ConvNeXt blocks, aux heads — runs in a couple of seconds on CPU, while
still exercising the exact same code path build_model() uses in real
training. This is the test most likely to catch a shape-mismatch crash
before it happens hours into a real run.
"""
import pytest

torch = pytest.importorskip("torch")

from models.unetr import BidirectionalSkip, UNETR


def test_bidirectional_skip_preserves_shape():
    skip = BidirectionalSkip(channels=16)
    shallow = torch.randn(1, 16, 4, 4, 4)
    deep = torch.randn(1, 16, 4, 4, 4)
    out = skip(shallow, deep)
    assert out.shape == shallow.shape
    assert torch.isfinite(out).all()


def test_bidirectional_skip_weights_start_near_equal():
    # w_shallow/w_deep initialized to 1 -> sigmoid(1) for both -> roughly
    # equal weighting at init, per the class docstring.
    skip = BidirectionalSkip(channels=8)
    assert torch.isclose(skip.w_shallow, skip.w_deep)


def test_unetr_forward_train_mode_returns_main_plus_aux(tiny_unetr, tiny_unetr_kwargs):
    tiny_unetr.train()
    x = torch.randn(1, tiny_unetr_kwargs["input_dim"], *tiny_unetr_kwargs["img_shape"])

    output, aux_z6, aux_z3 = tiny_unetr(x)

    expected_shape = (1, tiny_unetr_kwargs["output_dim"], *tiny_unetr_kwargs["img_shape"])
    assert output.shape == expected_shape
    assert aux_z6.shape == expected_shape
    assert aux_z3.shape == expected_shape
    for t in (output, aux_z6, aux_z3):
        assert torch.isfinite(t).all()


def test_unetr_forward_eval_mode_returns_single_tensor(tiny_unetr, tiny_unetr_kwargs):
    tiny_unetr.eval()
    x = torch.randn(1, tiny_unetr_kwargs["input_dim"], *tiny_unetr_kwargs["img_shape"])

    with torch.no_grad():
        output = tiny_unetr(x)

    assert isinstance(output, torch.Tensor)
    expected_shape = (1, tiny_unetr_kwargs["output_dim"], *tiny_unetr_kwargs["img_shape"])
    assert output.shape == expected_shape
    assert torch.isfinite(output).all()


def test_unetr_backward_pass_produces_finite_grads(tiny_unetr, tiny_unetr_kwargs):
    tiny_unetr.train()
    x = torch.randn(1, tiny_unetr_kwargs["input_dim"], *tiny_unetr_kwargs["img_shape"])
    target = torch.rand(1, tiny_unetr_kwargs["output_dim"], *tiny_unetr_kwargs["img_shape"])

    output, aux_z6, aux_z3 = tiny_unetr(x)
    loss = torch.nn.functional.mse_loss(output, target) + \
        torch.nn.functional.mse_loss(aux_z6, target) + \
        torch.nn.functional.mse_loss(aux_z3, target)
    loss.backward()

    grad_found = False
    for name, p in tiny_unetr.named_parameters():
        if p.grad is not None:
            grad_found = True
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert grad_found, "no gradients were populated — backward pass didn't reach any parameters"


def test_unetr_rejects_batch_with_wrong_input_channels(tiny_unetr, tiny_unetr_kwargs):
    wrong_input_dim = tiny_unetr_kwargs["input_dim"] + 1
    x = torch.randn(1, wrong_input_dim, *tiny_unetr_kwargs["img_shape"])
    with pytest.raises(RuntimeError):
        tiny_unetr(x)


@pytest.mark.slow
def test_unetr_real_config_forward_smoke(device):
    """One forward pass at the ACTUAL config.py dimensions (768 embed dim,
    12 layers, 128x128x96 volume) — slow, but catches issues the tiny model
    can mask (e.g. a channel mismatch that only appears at the real
    embed_dim/patch_dim). Run explicitly with `pytest -m slow`."""
    from config import cfg
    model = UNETR(
        img_shape=cfg.unetr.img_shape,
        input_dim=cfg.unetr.input_dim,
        output_dim=cfg.unetr.output_dim,
        embed_dim=cfg.unetr.embed_dim,
        patch_size=cfg.unetr.patch_size,
        num_heads=cfg.unetr.num_heads,
        dropout=cfg.unetr.dropout,
        encoder="vit",      # this is the ViT's full-size test; the Mamba one lives in test_vision_mamba.py
    ).to(device).eval()

    x = torch.randn(1, cfg.unetr.input_dim, *cfg.unetr.img_shape, device=device)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, cfg.unetr.output_dim, *cfg.unetr.img_shape)
    assert torch.isfinite(output).all()

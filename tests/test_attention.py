"""Tests for utils/attention.py, using the tiny_unetr fixture (see
conftest.py) so a full attention-hook -> extract -> overlay pipeline runs in
seconds. This is the exact code path that only executes every
cfg.attention.every_n_epochs epochs — i.e. the kind of code that, if broken,
crashes a real run only after several epochs of otherwise-successful
training. See utils/attention.py's fixed axis-mismatch bug (attn_map indexed
against the wrong spatial axis) that these tests were written to catch.
"""
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from utils.attention import (
    register_attention_hook, extract_attention_map, save_attention_overlay,
    save_attention_evolution,
)


class _FakeValDataset:
    """Minimal stand-in for a BratsDataset: only `.data[idx]["image"][2]`
    (the t1ce path, used to build the output filename) is touched."""
    def __init__(self, t1ce_path="/fake/path/patient1_t1ce.nii"):
        self.data = [{"image": ["", "", t1ce_path, ""]}]


def _run_forward(model, img_shape, input_dim):
    cache = register_attention_hook(model)
    x = torch.randn(1, input_dim, *img_shape)
    model.eval()
    with torch.no_grad():
        model(x)
    return cache


def test_register_attention_hook_populates_cache_on_forward(tiny_unetr, tiny_unetr_kwargs):
    cache = _run_forward(tiny_unetr, tiny_unetr_kwargs["img_shape"], tiny_unetr_kwargs["input_dim"])
    assert "attn" in cache
    assert cache["attn"].dim() == 4  # (B, num_heads, num_patches, num_patches)


def test_extract_attention_map_upsamples_to_requested_shape(tiny_unetr, tiny_unetr_kwargs):
    cache = _run_forward(tiny_unetr, tiny_unetr_kwargs["img_shape"], tiny_unetr_kwargs["input_dim"])
    attn_map = extract_attention_map(cache, tiny_unetr, tiny_unetr_kwargs["img_shape"])
    assert attn_map.shape == tuple(tiny_unetr_kwargs["img_shape"])
    assert np.isfinite(attn_map).all()


def test_extract_attention_map_upsamples_to_a_different_non_cubic_shape(tiny_unetr, tiny_unetr_kwargs):
    """Validation samples keep their own per-patient (non-fixed) shape — the
    map must be able to upsample to whatever shape that sample actually has,
    not just the model's fixed training img_shape."""
    cache = _run_forward(tiny_unetr, tiny_unetr_kwargs["img_shape"], tiny_unetr_kwargs["input_dim"])
    real_patient_shape = (18, 22, 13)  # deliberately non-cubic, all != tiny_unetr img_shape
    attn_map = extract_attention_map(cache, tiny_unetr, real_patient_shape)
    assert attn_map.shape == real_patient_shape


def test_extract_attention_map_returns_none_when_cache_empty(tiny_unetr, tiny_unetr_kwargs):
    assert extract_attention_map({}, tiny_unetr, tiny_unetr_kwargs["img_shape"]) is None


def test_save_attention_overlay_writes_png_at_model_img_shape(tmp_path, tiny_unetr, tiny_unetr_kwargs):
    img_shape = tiny_unetr_kwargs["img_shape"]
    cache = _run_forward(tiny_unetr, img_shape, tiny_unetr_kwargs["input_dim"])
    val_data = {"image": torch.rand(1, tiny_unetr_kwargs["input_dim"], *img_shape)}
    out_dir = str(tmp_path / "attention")

    save_attention_overlay(cache, tiny_unetr, val_data, _FakeValDataset(), epoch=1,
                            img_shape=img_shape, attention_dir=out_dir, sample_idx=0)

    files = os.listdir(out_dir)
    assert any(f.startswith("attention_epoch0001_patient1") for f in files)


def test_save_attention_overlay_does_not_crash_on_noncubic_per_patient_shape(tmp_path, tiny_unetr, tiny_unetr_kwargs):
    """Regression test for the real crash: a validation sample's actual
    (post-crop) shape is rarely cubic (H, W, D all different), and the
    overlay must handle that without an IndexError."""
    img_shape = tiny_unetr_kwargs["img_shape"]
    cache = _run_forward(tiny_unetr, img_shape, tiny_unetr_kwargs["input_dim"])

    per_patient_shape = (18, 22, 13)  # H != W != D, mirrors a real cropped sample
    val_data = {"image": torch.rand(1, tiny_unetr_kwargs["input_dim"], *per_patient_shape)}
    out_dir = str(tmp_path / "attention_noncubic")

    save_attention_overlay(cache, tiny_unetr, val_data, _FakeValDataset(), epoch=1,
                            img_shape=per_patient_shape, attention_dir=out_dir, sample_idx=0)

    assert any(f.startswith("attention_epoch0001_patient1") for f in os.listdir(out_dir))


def test_save_attention_overlay_skips_gracefully_when_cache_empty(tmp_path, tiny_unetr, tiny_unetr_kwargs, capsys):
    img_shape = tiny_unetr_kwargs["img_shape"]
    val_data = {"image": torch.rand(1, tiny_unetr_kwargs["input_dim"], *img_shape)}
    out_dir = str(tmp_path / "attention_empty")

    save_attention_overlay({}, tiny_unetr, val_data, _FakeValDataset(), epoch=1,
                            img_shape=img_shape, attention_dir=out_dir)

    assert not os.path.isdir(out_dir) or os.listdir(out_dir) == []
    assert "skipping" in capsys.readouterr().out


def test_save_attention_evolution_handles_no_files_gracefully(tmp_path, capsys):
    save_attention_evolution(str(tmp_path))
    assert "no attention files found" in capsys.readouterr().out


def test_save_attention_evolution_collects_saved_overlays(tmp_path, tiny_unetr, tiny_unetr_kwargs):
    img_shape = tiny_unetr_kwargs["img_shape"]
    cache = _run_forward(tiny_unetr, img_shape, tiny_unetr_kwargs["input_dim"])
    val_data = {"image": torch.rand(1, tiny_unetr_kwargs["input_dim"], *img_shape)}
    out_dir = str(tmp_path / "attention")

    for epoch in (1, 2):
        save_attention_overlay(cache, tiny_unetr, val_data, _FakeValDataset(), epoch=epoch,
                                img_shape=img_shape, attention_dir=out_dir, sample_idx=0)

    save_attention_evolution(out_dir)

    assert os.path.exists(os.path.join(out_dir, "attention_evolution.png"))

"""Tests for utils/transforms.py — the depth crop (now on the RAW scan,
before Resized) and the foreground H/W crop (still post-Resize) are the
pieces the user most needs confidence in (it's what tools/crop_visual_check.py
is for), so they get the most scrutiny here.
"""
import numpy as np
import pytest

pytest.importorskip("monai")

from utils.transforms import CropRawDepthd, CropForegroundHWd, ApplyCLAHEAndZscored, fit_to_size
from utils.dataloader import ConvertToMultiChannelBasedOnBratsClassesd


# ---------------------------------------------------------------------------
# CropRawDepthd — the raw-slice depth crop, runs BEFORE Resized
# ---------------------------------------------------------------------------

def _all_foreground_sample(shape):
    """image/label pair where every voxel is > 0, so the foreground bbox
    never collapses to an empty mask (see test documenting that edge case
    below)."""
    image = (np.random.rand(*shape).astype(np.float32) + 0.1)
    label = (np.random.rand(*shape).astype(np.float32) + 0.1)
    return image, label


def test_depth_axis_is_the_smallest_spatial_dimension():
    # (C, H, W, D) = (2, 10, 12, 6) -> D=6 is uniquely smallest -> depth_axis
    # (array axis) should be 3.
    image, label = _all_foreground_sample((2, 10, 12, 6))
    transform = CropRawDepthd(keys=["image", "label"], start_slice=1, num_slices=3)

    out = transform({"image": image, "label": label})

    assert out["image"].shape[3] == 3  # num_slices kept
    assert out["label"].shape[3] == 3
    # H/W are untouched by this transform entirely — that's CropForegroundHWd's job
    assert out["image"].shape[1:3] == (10, 12)


def test_crop_slice_count_matches_num_slices_when_in_bounds():
    # D must be uniquely the smallest dim for depth_axis to land on axis 3 —
    # (H, W, D) = (30, 28, 20).
    image, label = _all_foreground_sample((1, 30, 28, 20))
    transform = CropRawDepthd(keys=["image", "label"], start_slice=5, num_slices=10)

    out = transform({"image": image, "label": label})

    assert out["image"].shape[-1] == 10


def test_crop_silently_clamps_when_window_exceeds_source_depth():
    """Python slicing does NOT raise when start+num_slices > source depth —
    it silently returns fewer slices than requested. CropRawDepthd detects
    this (see the MISMATCH note test below) but the underlying array
    operation still just clamps."""
    depth = 20
    image, label = _all_foreground_sample((1, 30, 28, depth))  # D uniquely smallest
    transform = CropRawDepthd(keys=["image", "label"], start_slice=5, num_slices=95)  # far beyond available depth

    out = transform({"image": image, "label": label})

    actual = out["image"].shape[-1]
    assert actual == depth - 5  # clamped, not the requested 95
    assert actual != 95


def test_depth_axis_tiebreak_when_dims_are_equal():
    """When H==W==D, np.argmin(shape[1:]) picks the FIRST axis (H, array
    index 1) as "depth". This documents that tie-break behavior so a future
    config change that removes the uniqueness of the depth dimension
    doesn't silently corrupt the crop without anyone noticing."""
    image, label = _all_foreground_sample((1, 6, 6, 6))
    transform = CropRawDepthd(keys=["image", "label"], start_slice=0, num_slices=4)

    out = transform({"image": image, "label": label})

    # depth crop was applied to array axis 1 (the first of the tied dims),
    # so the resulting axis-1 size is 4, not axis-3.
    assert out["image"].shape[1] == 4


def test_crop_range_confirmation_prints_once_per_instance(capsys):
    image, label = _all_foreground_sample((1, 8, 8, 10))
    transform = CropRawDepthd(keys=["image", "label"], start_slice=1, num_slices=5)

    transform({"image": image.copy(), "label": label.copy()})
    transform({"image": image.copy(), "label": label.copy()})
    transform({"image": image.copy(), "label": label.copy()})

    captured = capsys.readouterr()
    assert captured.out.count("[CropRawDepthd] one-time check") == 1


def test_crop_range_confirmation_message_matches_requested_and_kept_depth(capsys):
    image, label = _all_foreground_sample((1, 8, 8, 10))
    start, num_slices = 1, 5
    transform = CropRawDepthd(keys=["image", "label"], start_slice=start, num_slices=num_slices)

    transform({"image": image, "label": label})

    out = capsys.readouterr().out
    assert f"[{start}:{start + num_slices})" in out
    assert f"kept depth={num_slices}" in out
    assert "MISMATCH" not in out


def test_crop_range_confirmation_flags_mismatch_when_window_exceeds_depth(capsys):
    # D must be uniquely smallest so depth_axis lands on axis 3 (source depth 8)
    image, label = _all_foreground_sample((1, 20, 22, 8))
    transform = CropRawDepthd(keys=["image", "label"], start_slice=5, num_slices=50)  # only 3 available, not 50

    transform({"image": image, "label": label})

    out = capsys.readouterr().out
    assert "kept depth=3" in out
    assert "MISMATCH" in out


# ---------------------------------------------------------------------------
# CropForegroundHWd — H/W-only crop, runs AFTER Resized
# ---------------------------------------------------------------------------

def test_foreground_bbox_crops_hw_to_nonzero_region():
    # place a single nonzero foreground block away from the array edges and
    # verify the H/W bbox matches it exactly.
    image = np.zeros((2, 10, 10, 4), dtype=np.float32)
    image[:, 2:5, 3:7, :] = 1.0  # foreground block: H in [2,5), W in [3,7)
    label = np.zeros_like(image)

    transform = CropForegroundHWd(keys=["image", "label"], threshold=0)
    out = transform({"image": image, "label": label})

    assert out["image"].shape == (2, 3, 4, 4)  # H: 5-2=3, W: 7-3=4, D: 4 (kept whole)
    assert np.all(out["image"] == 1.0)


def test_foreground_bbox_keeps_full_depth_untouched():
    image = np.zeros((2, 10, 10, 7), dtype=np.float32)
    image[:, 2:5, 3:7, :] = 1.0
    label = np.zeros_like(image)

    transform = CropForegroundHWd(keys=["image", "label"], threshold=0)
    out = transform({"image": image, "label": label})

    assert out["image"].shape[-1] == 7  # untouched


def test_foreground_bbox_raises_on_all_background_image():
    """Documents a real crash risk: if threshold excludes every voxel (e.g.
    an all-zero/background-only volume slips through), coords is empty and
    .min()/.max() raise ValueError. Exactly the kind of mid-run crash this
    test suite exists to surface ahead of time."""
    image = np.zeros((2, 6, 6, 4), dtype=np.float32)
    label = np.zeros_like(image)
    transform = CropForegroundHWd(keys=["image", "label"], threshold=0)

    with pytest.raises(ValueError):
        transform({"image": image, "label": label})


# ---------------------------------------------------------------------------
# Full pipeline integration — confirms the "no compression" design goal:
# the raw-depth crop runs before Resized, so Resized's depth resize should
# be an exact no-op and the final depth should always be exactly
# cfg.unetr.img_shape[-1], regardless of the raw scan's original depth.
# ---------------------------------------------------------------------------

def test_full_pipeline_keeps_exact_target_depth_no_compression(make_nii_patient):
    from config import cfg
    from utils.transforms import build_train_transform, build_val_transform

    target_depth = cfg.unetr.img_shape[-1]
    needed_min_depth = max(cfg.crop.train_start_slice, cfg.crop.val_start_slice) + target_depth
    raw_depth = needed_min_depth + 10
    hw = raw_depth + 20  # keep H/W strictly larger than D, matching real BraTS proportions

    patient_dir = make_nii_patient(patient_id="IntegrationPatient", shape=(hw, hw + 5, raw_depth))
    entry = {
        "image": [f"{patient_dir}/IntegrationPatient_{m}.nii" for m in ("flair", "t1", "t1ce", "t2")],
        "label": f"{patient_dir}/IntegrationPatient_seg.nii",
    }

    train_out = build_train_transform(cfg)(dict(entry))
    assert train_out["image"].shape[-1] == target_depth
    assert train_out["label"].shape[-1] == target_depth

    val_out = build_val_transform(cfg)(dict(entry))
    assert val_out["image"].shape[-1] == target_depth
    assert val_out["label"].shape[-1] == target_depth


# ---------------------------------------------------------------------------
# ConvertToMultiChannelBasedOnBratsClassesd
# ---------------------------------------------------------------------------

def test_convert_brats_labels_builds_tc_wt_et_channels():
    # label values: 0=background, 1=necrotic/non-enhancing, 2=edema, 4=enhancing
    label = np.array([[[0, 1, 2, 4]]], dtype=np.float32)  # shape (1,1,4)
    transform = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])

    out = transform({"label": label})["label"]

    assert out.shape == (3, 1, 1, 4)
    tc, wt, et = out[0, 0, 0], out[1, 0, 0], out[2, 0, 0]
    # TC = label in {1,4}
    assert list(tc) == [0.0, 1.0, 0.0, 1.0]
    # WT = label in {1,2,4}
    assert list(wt) == [0.0, 1.0, 1.0, 1.0]
    # ET = label == 4
    assert list(et) == [0.0, 0.0, 0.0, 1.0]


def test_convert_brats_labels_squeezes_leading_channel_dim():
    label = np.array([[[[0, 1, 2, 4]]]], dtype=np.float32)  # shape (1,1,1,4)
    transform = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])
    out = transform({"label": label})["label"]
    assert out.shape == (3, 1, 1, 4)


# ---------------------------------------------------------------------------
# ApplyCLAHEAndZscored
# ---------------------------------------------------------------------------

def test_apply_clahe_and_zscored_preserves_shape_for_4d_image():
    image = np.random.rand(4, 6, 6, 5).astype(np.float32) * 100
    transform = ApplyCLAHEAndZscored(keys=["image"])
    out = transform({"image": image})["image"]
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_apply_clahe_and_zscored_passes_through_non_4d_arrays_unchanged():
    label = np.random.rand(3, 6, 6).astype(np.float32)  # 3D, not 4D
    transform = ApplyCLAHEAndZscored(keys=["label"])
    out = transform({"label": label})["label"]
    np.testing.assert_array_equal(out, label)


# ---------------------------------------------------------------------------
# fit_to_size
# ---------------------------------------------------------------------------

def test_fit_to_size_pads_smaller_tensor_to_target():
    torch = pytest.importorskip("torch")
    x = torch.ones(1, 1, 4, 4, 4)
    out = fit_to_size(x, (8, 8, 8))
    assert out.shape == (1, 1, 8, 8, 8)
    # original data should be centered, surrounded by zero padding
    assert out[0, 0, 2:6, 2:6, 2:6].eq(1).all()
    assert out.sum().item() == 4 * 4 * 4  # only the original ones remain


def test_fit_to_size_crops_larger_tensor_to_target_centered():
    torch = pytest.importorskip("torch")
    x = torch.arange(8 ** 3, dtype=torch.float32).reshape(1, 1, 8, 8, 8)
    out = fit_to_size(x, (4, 4, 4))
    assert out.shape == (1, 1, 4, 4, 4)
    expected = x[:, :, 2:6, 2:6, 2:6]
    assert torch.equal(out, expected)


def test_fit_to_size_is_noop_when_already_target_size():
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 2, 6, 6, 6)
    out = fit_to_size(x, (6, 6, 6))
    assert torch.equal(out, x)


def test_fit_to_size_handles_mixed_pad_and_crop_dims():
    torch = pytest.importorskip("torch")
    x = torch.ones(1, 1, 4, 10, 4)  # D too small, H too large, W too small
    out = fit_to_size(x, (6, 6, 6))
    assert out.shape == (1, 1, 6, 6, 6)

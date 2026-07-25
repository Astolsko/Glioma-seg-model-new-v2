"""Tests for the inference-time decisions: per-channel thresholding,
connected-component cleanup, and the validation threshold sweep."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from utils.postprocess import binarize, postprocess, search_thresholds


@pytest.fixture
def infer_cfg():
    from easydict import EasyDict
    cfg = EasyDict()
    cfg.infer = EasyDict()
    cfg.infer.thresholds = (0.5, 0.5, 0.5)
    cfg.infer.min_component_voxels = (0, 0, 10)
    cfg.infer.min_total_voxels = (0, 0, 0)
    return cfg


def test_binarize_applies_a_different_threshold_per_channel():
    # every voxel has probability 0.7 in every channel
    logits = torch.full((3, 2, 2, 2), float(np.log(0.7 / 0.3)))

    out = binarize(logits, (0.5, 0.9, 0.1))

    assert out[0].all(), "0.7 > 0.5 should fire"
    assert not out[1].any(), "0.7 < 0.9 should not fire"
    assert out[2].all(), "0.7 > 0.1 should fire"


def test_binarize_handles_batched_and_unbatched_input():
    logits = torch.zeros(3, 4, 4, 4)
    assert binarize(logits, (0.4, 0.4, 0.4)).shape == (3, 4, 4, 4)
    assert binarize(logits.unsqueeze(0), (0.4, 0.4, 0.4)).shape == (1, 3, 4, 4, 4)


def test_postprocess_removes_components_below_min_size(infer_cfg):
    mask = np.zeros((3, 12, 12, 12), dtype=np.float32)
    mask[2, 0:4, 0:4, 0:4] = 1      # 64-voxel blob, must survive
    mask[2, 10, 10, 10] = 1         # 1-voxel speck, must go

    out = postprocess(mask, infer_cfg)

    assert out[2].sum() == 64
    assert out[2, 10, 10, 10] == 0


def test_postprocess_leaves_channels_with_zero_thresholds_untouched(infer_cfg):
    mask = np.zeros((3, 12, 12, 12), dtype=np.float32)
    mask[0, 5, 5, 5] = 1            # TC threshold is 0 -> speck stays

    assert postprocess(mask, infer_cfg)[0].sum() == 1


def test_postprocess_zeroes_channel_when_survivors_are_below_min_total(infer_cfg):
    # This is the ET/HD95 case: a few stray FP voxels on a patient with no ET
    # cost compute_hd95's full 374.0 empty-mismatch penalty.
    infer_cfg.infer.min_component_voxels = (0, 0, 0)
    infer_cfg.infer.min_total_voxels = (0, 0, 50)

    mask = np.zeros((3, 12, 12, 12), dtype=np.float32)
    mask[2, 0:3, 0:3, 0:3] = 1      # 27 voxels, below the 50-voxel floor

    assert postprocess(mask, infer_cfg)[2].sum() == 0


def test_postprocess_round_trips_tensor_input(infer_cfg):
    mask = torch.zeros(3, 8, 8, 8)
    mask[2, 0:4, 0:4, 0:4] = 1

    out = postprocess(mask, infer_cfg)

    assert torch.is_tensor(out)
    assert out.shape == mask.shape


def test_search_thresholds_recovers_the_threshold_that_maximises_dice(infer_cfg):
    # Ground truth fills half the volume. Probabilities are 0.65 inside and
    # 0.45 outside, so only a threshold in (0.45, 0.65] segments it exactly —
    # 0.5 works, 0.7 misses everything, 0.3 floods.
    shape = (3, 4, 4, 4)
    gt = np.zeros(shape, dtype=np.float32)
    gt[:, :2] = 1.0
    prob = np.where(gt > 0.5, 0.65, 0.45).astype(np.float32)
    logits = torch.from_numpy(np.log(prob / (1 - prob))).unsqueeze(0)

    batch = {"image": torch.zeros(1, 4, *shape[1:]),
             "label": torch.from_numpy(gt).unsqueeze(0)}

    best, sweep = search_thresholds(
        model=None, loader=[batch], device="cpu", cfg=infer_cfg,
        inferer=lambda _model, _x: logits,
        candidates=[0.3, 0.5, 0.7],
        apply_postprocess=False,
    )

    assert best == (0.5, 0.5, 0.5)
    assert sweep["dice_mean"][1] == pytest.approx([1.0, 1.0, 1.0])


def test_search_thresholds_returns_config_default_on_empty_loader(infer_cfg):
    best, sweep = search_thresholds(
        model=None, loader=[], device="cpu", cfg=infer_cfg,
        inferer=lambda _model, _x: None,
    )

    assert best == tuple(infer_cfg.infer.thresholds)
    assert sweep == {}

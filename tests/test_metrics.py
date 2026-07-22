import numpy as np
import pytest

pytest.importorskip("medpy")

from utils.metrics import (
    compute_confusion, compute_sensitivity, compute_iou, compute_miou,
    compute_specificity, compute_f1, compute_hd95, compute_roc_auc,
    minmax_normalize,
)


@pytest.fixture
def pred_gt_pair():
    # 4x4 boolean masks with a known overlap:
    # gt:   3x3 square at rows/cols 0-2
    # pred: 3x3 square at rows/cols 1-3 (overlaps gt in the 2x2 region [1:3,1:3])
    gt = np.zeros((4, 4), dtype=bool)
    gt[0:3, 0:3] = True
    pred = np.zeros((4, 4), dtype=bool)
    pred[1:4, 1:4] = True
    return pred, gt


def test_compute_confusion_counts(pred_gt_pair):
    pred, gt = pred_gt_pair
    tp, fp, fn = compute_confusion(pred, gt)
    # overlap region [1:3, 1:3] = 4 pixels -> tp
    assert tp == 4
    # pred true, gt false: pred has 9 true, 4 overlap with gt -> fp = 5
    assert fp == 5
    # gt true, pred false: gt has 9 true, 4 overlap -> fn = 5
    assert fn == 5


def test_compute_sensitivity(pred_gt_pair):
    pred, gt = pred_gt_pair
    tp, fp, fn = compute_confusion(pred, gt)
    assert compute_sensitivity(tp, fn) == pytest.approx(4 / 9)


def test_compute_sensitivity_no_positives_in_gt():
    assert compute_sensitivity(tp=0, fn=0) == 0.0


def test_compute_iou(pred_gt_pair):
    pred, gt = pred_gt_pair
    tp, fp, fn = compute_confusion(pred, gt)
    assert compute_iou(tp, fp, fn) == pytest.approx(4 / (4 + 5 + 5))


def test_compute_iou_empty_masks_is_zero():
    assert compute_iou(tp=0, fp=0, fn=0) == 0.0


def test_compute_f1_matches_dice(pred_gt_pair):
    pred, gt = pred_gt_pair
    tp, fp, fn = compute_confusion(pred, gt)
    f1 = compute_f1(tp, fp, fn)
    assert f1 == pytest.approx(2 * 4 / (2 * 4 + 5 + 5))


def test_compute_specificity(pred_gt_pair):
    pred, gt = pred_gt_pair
    spec = compute_specificity(pred, gt)
    # background (gt False) has 16-9=7 pixels; tn = background & ~pred
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    assert spec == pytest.approx(tn / (tn + fp))


def test_compute_specificity_perfect_prediction():
    gt = np.zeros((4, 4), dtype=bool)
    gt[0, 0] = True
    pred = gt.copy()
    assert compute_specificity(pred, gt) == 1.0


def test_compute_miou_perfect_prediction_is_one():
    gt = np.zeros((4, 4), dtype=bool)
    gt[1:3, 1:3] = True
    pred = gt.copy()
    assert compute_miou(pred, gt) == pytest.approx(1.0)


def test_compute_miou_no_overlap_is_not_one(pred_gt_pair):
    pred, gt = pred_gt_pair
    miou = compute_miou(pred, gt)
    assert 0.0 < miou < 1.0


def test_compute_hd95_both_empty_is_zero():
    pred = np.zeros((5, 5), dtype=bool)
    gt = np.zeros((5, 5), dtype=bool)
    assert compute_hd95(pred, gt, voxel_spacing=(1.0, 1.0)) == 0.0


def test_compute_hd95_one_empty_one_not_returns_sentinel():
    pred = np.zeros((5, 5), dtype=bool)
    gt = np.zeros((5, 5), dtype=bool)
    gt[2, 2] = True
    assert compute_hd95(pred, gt, voxel_spacing=(1.0, 1.0)) == 374.0


def test_compute_hd95_identical_masks_is_zero():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True
    assert compute_hd95(mask, mask.copy(), voxel_spacing=(1.0, 1.0)) == pytest.approx(0.0)


def test_compute_roc_auc_degenerate_single_class_is_zero():
    gt_all_zero = np.zeros((4, 4), dtype=bool)
    prob = np.random.rand(4, 4).astype(np.float32)
    assert compute_roc_auc(prob, gt_all_zero) == 0.0

    gt_all_one = np.ones((4, 4), dtype=bool)
    assert compute_roc_auc(prob, gt_all_one) == 0.0


def test_compute_roc_auc_perfect_separation_is_one():
    gt = np.array([[0, 0], [1, 1]], dtype=bool)
    prob = np.array([[0.0, 0.1], [0.9, 1.0]], dtype=np.float32)
    assert compute_roc_auc(prob, gt) == pytest.approx(1.0)


def test_minmax_normalize_range():
    arr = np.array([1.0, 5.0, 10.0], dtype=np.float32)
    out = minmax_normalize(arr)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_minmax_normalize_constant_array_returns_zeros():
    arr = np.full((3, 3), 7.0, dtype=np.float32)
    out = minmax_normalize(arr)
    assert np.all(out == 0.0)

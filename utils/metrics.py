import numpy as np
from medpy.metric.binary import hd95 as medpy_hd95


def compute_hd95(pred, gt, voxel_spacing):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    if not pred.any() and not gt.any():
        return 0.0
    if pred.any() != gt.any():
        return 374.0
    return float(medpy_hd95(pred, gt, voxelspacing=voxel_spacing))


def compute_sensitivity(tp, fn):
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def compute_iou(tp, fp, fn):
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def compute_miou(pred, gt):
    """
    True mIoU per the formula:
    mIoU = 0.5 * [TP/(TP+FP+FN) + TN/(TN+FN+FP)]
    Computes IoU for both foreground and background then averages.
    """
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    fg_iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    bg_iou = float(tn / (tn + fn + fp)) if (tn + fn + fp) > 0 else 0.0

    return (fg_iou + bg_iou) / 2.0


def compute_confusion(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    return tp, fp, fn


def compute_specificity(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def compute_f1(tp, fp, fn):
    # F1 = Dice = 2TP / (2TP + FP + FN)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def compute_roc_auc(pred_prob, gt):
    from sklearn.metrics import roc_auc_score
    gt_flat = gt.astype(np.bool_).flatten().astype(np.int32)
    prob_flat = pred_prob.flatten()
    if gt_flat.sum() == 0 or gt_flat.sum() == gt_flat.size:
        return 0.0  # AUC undefined if only one class present
    try:
        return float(roc_auc_score(gt_flat, prob_flat))
    except Exception:
        return 0.0


def minmax_normalize(arr):
    arr = arr.astype(np.float32)
    vmin = arr.min()
    vmax = arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)

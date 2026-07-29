"""Turning model logits into a final mask: per-channel thresholding and
connected-component cleanup, plus the validation-split search that picks the
thresholds.

All of this is inference-time only — nothing here needs retraining, and it can
be applied to an already-saved checkpoint via evaluate.py.
"""
import json
import os

import numpy as np
import torch
from scipy import ndimage

CHANNEL_NAMES = ("TC", "WT", "ET")


def binarize(logits, thresholds):
    """Sigmoid + per-channel threshold.

    Accepts (C, H, W, D) or (B, C, H, W, D) — the channel axis is -4 in both,
    so one broadcast shape covers each. MONAI's AsDiscrete only takes a scalar
    threshold, which is why this exists.
    """
    prob = torch.sigmoid(logits.float())
    t = torch.as_tensor(thresholds, dtype=prob.dtype, device=prob.device)
    return (prob > t.view(-1, 1, 1, 1)).to(prob.dtype)


def _clean_channel(mask, min_component, min_total):
    """Drop components below `min_component` voxels, then zero the channel
    entirely if the survivors total less than `min_total` voxels."""
    if min_component > 0 and mask.any():
        labelled, n_components = ndimage.label(mask)
        if n_components:
            sizes = np.bincount(labelled.ravel())
            # sizes[0] is the background count; including label 0 in
            # `too_small` is harmless because mask is already False wherever
            # labelled == 0.
            too_small = np.flatnonzero(sizes < min_component)
            mask = mask & ~np.isin(labelled, too_small)

    if min_total > 0 and mask.sum() < min_total:
        return np.zeros_like(mask)
    return mask


def postprocess(mask, cfg):
    """Apply the connected-component rules to a (C, H, W, D) binary mask.

    Works on numpy or torch input and returns the same type it was given, so
    it can drop into either the metric loop or a plotting path.
    """
    was_tensor = torch.is_tensor(mask)
    arr = mask.detach().cpu().numpy() if was_tensor else np.asarray(mask)
    arr = arr > 0.5

    cleaned = np.stack([
        _clean_channel(arr[c],
                       cfg.infer.min_component_voxels[c],
                       cfg.infer.min_total_voxels[c])
        for c in range(arr.shape[0])
    ])

    if was_tensor:
        return torch.from_numpy(cleaned).to(dtype=mask.dtype, device=mask.device)
    return cleaned.astype(np.float32)


def _dice(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def search_thresholds(model, loader, device, cfg, inferer, candidates=None,
                      apply_postprocess=True, console=None):
    """Sweep a per-channel probability threshold over a split and return the
    triple that maximises mean per-patient Dice.

    Dice is averaged per patient rather than pooled globally so the choice
    matches how `validate()` reports it (MONAI's DiceMetric aggregates per
    sample). `inferer(model, inputs)` must return logits.

    Patients with an EMPTY ground truth in a channel are skipped, because
    MONAI's DiceMetric(ignore_empty=True) — what run_test actually reports —
    skips them too. Scoring both-empty as a free 1.0 (as this did) makes the
    sweep optimise a different metric from the one that gets published, and it
    is not a harmless difference: raising the ET threshold tips one ET-negative
    patient below min_total_voxels, the channel is zeroed, and the sweep
    collects +1.0 for predicting nothing. On run1-new-version that produced a
    lone +0.017 spike at exactly 0.50 in an otherwise monotone curve, and that
    spike is what picked the published ET threshold.

    Presence/absence is a real problem for ET (4 of 70 test patients), but Dice
    with ignore_empty cannot see it in either direction — it belongs to HD95's
    374.0 sentinel and to min_total_voxels, not to this sweep.
    """
    if candidates is None:
        # Floor was 0.30, and run1-new-version's sweep picked exactly 0.30 for
        # BOTH TC and WT — i.e. the search wanted to go lower and the grid
        # stopped it. EMA shrinks logit magnitude, so the optimum sits well
        # below 0.5 and the old range could not reach it.
        candidates = np.round(np.arange(0.05, 0.75, 0.05), 2).tolist()

    dice_sums = np.zeros((len(candidates), 3), dtype=np.float64)
    # Which channels a patient is scored on depends only on its ground truth,
    # not on the threshold, so one count per channel covers every candidate.
    n_scored = np.zeros(3, dtype=np.int64)
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            logits = inferer(model, batch["image"].to(device))
            prob = torch.sigmoid(logits.float())[0].cpu().numpy()
            gt = batch["label"][0].cpu().numpy() > 0.5

            scored = [c for c in range(3) if gt[c].any()]
            for c in scored:
                n_scored[c] += 1

            for i, threshold in enumerate(candidates):
                pred = prob > threshold
                if apply_postprocess:
                    pred = postprocess(pred.astype(np.float32), cfg) > 0.5
                for c in scored:
                    dice_sums[i, c] += _dice(pred[c], gt[c])
            n_samples += 1

    if n_samples == 0 or not n_scored.any():
        return tuple(cfg.infer.thresholds), {}

    dice_mean = np.divide(dice_sums, n_scored,
                          out=np.zeros_like(dice_sums), where=n_scored > 0)
    best_idx = dice_mean.argmax(axis=0)
    best = tuple(float(candidates[i]) for i in best_idx)

    lines = ["Threshold sweep (mean per-patient Dice, empty-GT patients skipped):",
             "scored patients per channel: " + ", ".join(
                 f"{name}={n_scored[c]}/{n_samples}"
                 for c, name in enumerate(CHANNEL_NAMES)),
             f"{'thr':>6}" + "".join(f"{name:>9}" for name in CHANNEL_NAMES)]
    for i, threshold in enumerate(candidates):
        lines.append(f"{threshold:>6.2f}" + "".join(f"{dice_mean[i, c]:>9.4f}"
                                                    for c in range(3)))
    lines.append("best: " + ", ".join(
        f"{name}={best[c]:.2f} (Dice {dice_mean[best_idx[c], c]:.4f})"
        for c, name in enumerate(CHANNEL_NAMES)))
    report = "\n".join(lines)
    print(report, file=console) if console else print(report)

    return best, {
        "candidates": candidates,
        "dice_mean": dice_mean.tolist(),
        "best": best,
        "n_scored": n_scored.tolist(),
        "n_samples": n_samples,
    }


def tune_and_save(model, val_loader, device, cfg, checkpoint_path, eval_dir,
                  inferer, console=None, filename="threshold_sweep.json"):
    """Load the best checkpoint, tune thresholds on val, persist the sweep.

    Loads the checkpoint FIRST because after training the in-memory model holds
    the last epoch's weights — and with EMA enabled, the raw non-averaged ones —
    so tuning against it would tune a model nobody is going to ship.

    Mutates `cfg.infer.thresholds` so the test pass that follows picks them up.
    Shared by train.py (end of a fresh run) and evaluate.py (against a saved
    checkpoint) so the two cannot tune differently.

    `filename` exists because evaluate.py re-tunes into the SAME eval/ folder a
    training run already wrote to. Without a per-tag name a re-tune silently
    destroys the original run's sweep — which is the evidence for whatever the
    re-tune is trying to improve on.
    """
    print("\n=== Tuning inference thresholds on the validation split ===")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    best, sweep = search_thresholds(model, val_loader, device, cfg,
                                    inferer=inferer, console=console)
    cfg.infer.thresholds = best

    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, filename), "w") as f:
        json.dump(sweep, f, indent=2)
    return best


def save_infer_config(cfg, path):
    """Record the exact inference recipe a set of results was produced with."""
    with open(path, "w") as f:
        json.dump({k: list(v) if isinstance(v, tuple) else v
                   for k, v in cfg.infer.items()}, f, indent=2)

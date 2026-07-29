"""No-retrain ET operating-point ablation on run1-new-version.

Model logits are identical across threshold / min_total / min_component settings
— only the post-hoc binarize + connected-component cleanup changes. So run TTA
inference ONCE over the 70 test patients, cache the logits, then sweep ET
operating points instantly.

Reuses the EXACT same primitives run_test uses (binarize, _clean_channel,
compute_hd95 with the 374.0 sentinel, Dice = 2TP/(|P|+|G|) with empty-GT skipped)
so the numbers are directly comparable. TC/WT are held at the sweep's 0.10 and
are untouched by ET-only postproc (channels 0,1 get min_comp/min_total = 0), so
they are constant across the ET grid and computed once.
"""
import os
import numpy as np
import torch

from utils.env_check import ensure_dependencies
ensure_dependencies()

from config import cfg
from utils.dataloader import build_dataloaders
from utils.engine import build_model, get_device, run_inference
from utils.postprocess import binarize, _clean_channel
from utils.metrics import compute_hd95, compute_confusion, compute_sensitivity

CKPT = "logs/run1-new-version/checkpoints/best_metric_model.pth"
SPACING = cfg.metrics.voxel_spacing
TC_THR, WT_THR = 0.10, 0.10   # sweep winners; fixed here


def dice(pred, gt):
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, gt).sum() / denom)


def channel_metrics(pred_c_list, gt_c_list):
    """Mean Dice (empty-GT skipped, pred-empty->0), plus HD95 breakdown, over a
    list of per-patient (pred, gt) boolean arrays for ONE channel."""
    dsum, dn = 0.0, 0
    hd_full_sum, hd_clean_sum = 0.0, 0.0
    n_hal, n_miss, n_clean = 0, 0, 0
    for pred_c, gt_c in zip(pred_c_list, gt_c_list):
        if gt_c.any():
            dsum += dice(pred_c, gt_c); dn += 1
        hd = compute_hd95(pred_c, gt_c, SPACING)
        hd_full_sum += hd
        if pred_c.any() and not gt_c.any():
            n_hal += 1
        elif gt_c.any() and not pred_c.any():
            n_miss += 1
        elif gt_c.any():
            hd_clean_sum += hd; n_clean += 1
    n = len(pred_c_list)
    return {
        "dice": dsum / dn if dn else 0.0,
        "hd95": hd_full_sum / n,
        "hd95_clean": hd_clean_sum / n_clean if n_clean else 0.0,
        "n_hal": n_hal, "n_miss": n_miss, "n_clean": n_clean,
    }


def main():
    device = get_device()
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(CKPT))
    model.eval()
    loaders = build_dataloaders(cfg)

    # --- one inference pass; cache probabilities (post-sigmoid) + labels ---
    probs, gts = [], []
    print("\nCaching test-set predictions (TTA once) ...")
    with torch.no_grad():
        for i, batch in enumerate(loaders["test_loader"]):
            logits = run_inference(model, batch["image"].to(device), cfg)  # (1,3,...)
            probs.append(torch.sigmoid(logits.float())[0].cpu().numpy().astype(np.float32))
            gts.append((batch["label"][0].cpu().numpy() > 0.5))
            print(f"  {i+1}/{len(loaders['test_loader'])}", end="\r")
    print(f"\ncached {len(probs)} patients\n")

    # --- TC / WT constant across the ET grid (thr 0.10, no cleanup) ---
    tc = channel_metrics([p[0] > TC_THR for p in probs], [g[0] for g in gts])
    wt = channel_metrics([p[1] > WT_THR for p in probs], [g[1] for g in gts])
    print(f"TC (thr {TC_THR}): Dice={tc['dice']:.4f} HD95={tc['hd95']:.2f} "
          f"clean={tc['hd95_clean']:.2f} halluc={tc['n_hal']} miss={tc['n_miss']}")
    print(f"WT (thr {WT_THR}): Dice={wt['dice']:.4f} HD95={wt['hd95']:.2f} "
          f"clean={wt['hd95_clean']:.2f} halluc={wt['n_hal']} miss={wt['n_miss']}\n")

    et_gts = [g[2] for g in gts]

    grid_thr = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    grid_pp = [(50, 100), (50, 200), (50, 300), (100, 200), (100, 300)]

    header = (f"{'ET_thr':>6} {'minC':>5} {'minT':>5} | {'hal':>3} {'miss':>4} "
              f"{'clean':>5} | {'ET_Dice':>7} {'ET_HD95':>7} {'ETclean':>7} | "
              f"{'meanDice':>8} {'meanHD95':>8}")
    print(header)
    print("-" * len(header))
    rows = []
    for thr in grid_thr:
        for mc, mt in grid_pp:
            et_preds = []
            for p in probs:
                m = (p[2] > thr).astype(np.float32)
                m = _clean_channel(m > 0.5, mc, mt)   # returns bool array
                et_preds.append(np.asarray(m) > 0.5)
            et = channel_metrics(et_preds, et_gts)
            mean_dice = (tc['dice'] + wt['dice'] + et['dice']) / 3
            mean_hd95 = (tc['hd95'] + wt['hd95'] + et['hd95']) / 3
            tag = "  <- final" if (thr == 0.05 and mc == 50 and mt == 100) else ""
            print(f"{thr:>6.2f} {mc:>5} {mt:>5} | {et['n_hal']:>3} {et['n_miss']:>4} "
                  f"{et['n_clean']:>5} | {et['dice']:>7.4f} {et['hd95']:>7.2f} "
                  f"{et['hd95_clean']:>7.2f} | {mean_dice:>8.4f} {mean_hd95:>8.2f}{tag}")
            rows.append((thr, mc, mt, et, mean_dice, mean_hd95))

    # best by mean HD95, then by mean Dice
    best = min(rows, key=lambda r: (r[5], -r[4]))
    print(f"\nBest mean-HD95 operating point: ET_thr={best[0]} min_comp={best[1]} "
          f"min_total={best[2]} -> meanDice={best[4]:.4f} meanHD95={best[5]:.2f} "
          f"(ET: {best[3]['n_hal']}h+{best[3]['n_miss']}m, Dice {best[3]['dice']:.4f})")


if __name__ == "__main__":
    main()

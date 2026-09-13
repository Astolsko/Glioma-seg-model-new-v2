"""Cache a finished run's probabilities on val + test, plus a per-patient
modality ablation on test, so operating-point analysis needs no GPU.

    python tools/cache_predictions.py --run v3-mamba-30ep --out <dir>
    python tools/cache_predictions.py --run v3-mamba-30ep --out <dir> --limit 2   # smoke test

Thresholds and connected-component cleanup never change the model output, so
one TTA pass per patient covers any grid over them. The probabilities saved are
exactly the sigmoid run_test thresholds: the run's own eval/infer_config.json
(overlap, blending, TTA) and its best_metric_model.pth.

Writes <out>/<run>/:
  {val,test}/NNN_prob.npy   float32 (3, H, W, D), TC/WT/ET
  {val,test}/NNN_gt.npy     bool    (3, H, W, D)
  modality_counts.npy       int64 (N_test, 5, 3, 3): arm (baseline, -FLAIR,
                            -T1, -T1ce, -T2) x channel x [tp, |pred|, |gt|],
                            no TTA and no cleanup, exactly as xai's
                            modality_attribution scores it
  meta.json                 patient ids per split + the inference recipe

Run it in the env the run was trained in (a Mamba run needs `mamba`).
Re-running skips patients already cached. tools/operating_point_study.py
checks the cache against the run's own testing/test_metrics.csv before using it.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_check import configure_cuda_allocator, ensure_dependencies
ensure_dependencies()

from config import cfg
configure_cuda_allocator(cfg.checkpoint.cuda_alloc_conf)

import numpy as np
import torch
from monai.data import DataLoader

from utils.dataloader import BratsDataset
from utils.engine import apply_run_model_config, build_model, get_device, run_inference
from utils.transforms import build_val_transform
from utils.xai import MODALITY_NAMES

SECTIONS = {"val": "validation", "test": "test"}
INFER_KEYS = ("sw_overlap", "sw_mode", "tta_flips", "thresholds",
              "min_component_voxels", "min_total_voxels")


def patient_id(ds, i):
    return os.path.basename(os.path.dirname(ds.data[i]["label"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run folder under logs/")
    parser.add_argument("--out", required=True, help="Cache root; <out>/<run>/ is written")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=list(SECTIONS))
    parser.add_argument("--limit", type=int, default=None, help="First N patients per split")
    parser.add_argument("--no-modality", action="store_true", help="Skip the test modality ablation")
    args = parser.parse_args()

    run_dir = os.path.join(cfg.paths.logs_dir, args.run)
    apply_run_model_config(cfg, run_dir)
    with open(os.path.join(run_dir, "eval", "infer_config.json")) as f:
        shipped = json.load(f)
    for key in INFER_KEYS:
        cfg.infer[key] = tuple(shipped[key]) if isinstance(shipped[key], list) else shipped[key]

    out_dir = os.path.join(args.out, args.run)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    device = get_device()
    model = build_model(cfg, device)
    checkpoint = os.path.join(run_dir, "checkpoints", "best_metric_model.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    meta.update({
        "run": args.run, "encoder": cfg.unetr.encoder, "torch": torch.__version__,
        "checkpoint": checkpoint,
        "infer": {k: list(cfg.infer[k]) if isinstance(cfg.infer[k], tuple) else cfg.infer[k]
                  for k in INFER_KEYS},
    })
    meta.setdefault("ids", {})

    transform = build_val_transform(cfg)
    for split in args.splits:
        ds = BratsDataset(root_dir=cfg.paths.root_dir, section=SECTIONS[split],
                          transform=transform, val_frac=cfg.data.val_frac,
                          test_frac=cfg.data.test_frac, seed=cfg.seed,
                          cache_rate=0.0, num_workers=0)
        n = len(ds) if args.limit is None else min(args.limit, len(ds))
        meta["ids"][split] = [patient_id(ds, i) for i in range(n)]
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        do_modality = split == "test" and not args.no_modality
        counts_path = os.path.join(out_dir, "modality_counts.npy")
        counts = np.full((n, len(MODALITY_NAMES) + 1, 3, 3), -1, dtype=np.int64)
        if do_modality and os.path.exists(counts_path):
            previous = np.load(counts_path)
            k = min(n, len(previous))
            counts[:k] = previous[:k]

        loader = DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=1,
                            shuffle=False, num_workers=4)
        start = time.time()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                prob_path = os.path.join(split_dir, f"{i:03d}_prob.npy")
                need_prob = not os.path.exists(prob_path)
                need_modality = do_modality and (counts[i] < 0).any()
                if not (need_prob or need_modality):
                    continue

                image = batch["image"].to(device)
                gt = batch["label"][0].cpu().numpy() > 0.5

                if need_prob:
                    logits = run_inference(model, image, cfg)
                    prob = torch.sigmoid(logits.float())[0].cpu().numpy().astype(np.float32)
                    np.save(os.path.join(split_dir, f"{i:03d}_gt.npy"), gt)
                    tmp_path = os.path.join(split_dir, f"{i:03d}_prob.tmp.npy")
                    np.save(tmp_path, prob)
                    os.replace(tmp_path, prob_path)   # a killed run never leaves a torn file

                if need_modality:
                    # arm 0 = all modalities; arm k = modality k-1 zeroed, as
                    # utils/xai.py:modality_attribution does it.
                    for arm in range(len(MODALITY_NAMES) + 1):
                        x = image
                        if arm:
                            x = image.clone()
                            x[:, arm - 1] = 0.0
                        prob_m = torch.sigmoid(
                            run_inference(model, x, cfg, tta=False).float())[0].cpu().numpy()
                        for c in range(3):
                            pred = prob_m[c] > cfg.infer.thresholds[c]
                            counts[i, arm, c] = (np.logical_and(pred, gt[c]).sum(),
                                                 pred.sum(), gt[c].sum())
                    np.save(counts_path, counts)

                elapsed = time.time() - start
                print(f"[{args.run} {split}] {i + 1}/{n} {meta['ids'][split][i]} "
                      f"shape={tuple(gt.shape[1:])} elapsed={elapsed / 60:.1f}min", flush=True)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{args.run} {split}] done: {n} patients -> {split_dir}", flush=True)


if __name__ == "__main__":
    main()

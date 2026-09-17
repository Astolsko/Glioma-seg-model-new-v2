"""ET operating point selected on VALIDATION, paired per-patient statistics
between two runs on test, and the modality ablation over the whole test split.
CPU only, from what tools/cache_predictions.py cached.

    python tools/operating_point_study.py --cache <dir> --runs v2-run3 v3-mamba-30ep

Why each part exists:
  * The val threshold sweep maximises Dice with empty-GT patients skipped, so
    it cannot see ET hallucinations, and ET HD95 is dominated by them (374.0
    per empty mismatch). Session 5 picked the ET knee from a TEST-set grid.
    Here the knee is chosen on val with a rule fixed in advance, and test is
    only read at the chosen point (the test grid is printed for transparency).
  * n=70 test patients: a difference between two runs needs a paired test on
    the same patients, not a comparison of two means.
  * The in-train modality ablation reports no confidence intervals and scores
    both-empty as Dice 1.0, which on ET mixes presence/absence into the
    modality effect.

Nothing is trusted until the cache reproduces each run's own
testing/test_metrics.csv at its shipped operating point (section 1).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import warnings
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config import cfg
from utils.metrics import compute_confusion, compute_hd95, compute_iou, compute_sensitivity
from utils.postprocess import _clean_channel

CHANNELS = ("TC", "WT", "ET")
MODALITIES = ("FLAIR", "T1", "T1ce", "T2")
ET = 2
SPACING = tuple(cfg.metrics.voxel_spacing)
THR_GRID = np.round(np.arange(0.05, 0.75, 0.05), 2).tolist()
# (min_component, min_total) for ET, in 1mm voxels. (176, 352) is what shipped.
PP_GRID = [(0, 0), (176, 352), (176, 704), (352, 1056)]
DEFAULT_PP = (176, 352)
N_BOOT = 10000


def grid_key(thr, pp):
    return f"{thr:.2f}/{pp[0]}/{pp[1]}"


# ---------------------------------------------------------------------------
# per patient (worker)
# ---------------------------------------------------------------------------

def _record(pred, gt, hd_memo):
    """One channel of one patient, scored exactly as run_test scores it."""
    tp, fp, fn = compute_confusion(pred, gt)
    pred_vox, gt_vox = int(pred.sum()), int(gt.sum())
    key = hashlib.blake2b(np.packbits(pred).tobytes(), digest_size=16).digest()
    if key not in hd_memo:
        hd_memo[key] = compute_hd95(pred, gt, SPACING)
    if gt_vox == 0:
        status = "halluc" if pred_vox else "empty"
    else:
        status = "clean" if pred_vox else "miss"
    return {
        # MONAI DiceMetric(ignore_empty=True): empty GT is excluded (nan).
        "dice": float("nan") if gt_vox == 0 else float(2.0 * tp / (pred_vox + gt_vox)),
        "hd95": float(hd_memo[key]),
        "sens": float(compute_sensitivity(tp, fn)),
        "iou": float(compute_iou(tp, fp, fn)),
        "status": status, "pred_vox": pred_vox, "gt_vox": gt_vox,
    }


def eval_patient(task):
    split_dir, idx, thresholds, min_comp, min_total = task
    prob = np.load(os.path.join(split_dir, f"{idx:03d}_prob.npy"), mmap_mode="r")
    gt = np.load(os.path.join(split_dir, f"{idx:03d}_gt.npy"))

    out = {"idx": idx, "shipped": [], "diag": [], "et_grid": {}}
    memos = [{}, {}, {}]
    for c in range(3):
        p = np.asarray(prob[c])
        raw = p > thresholds[c]
        pred = _clean_channel(raw, min_comp[c], min_total[c])
        out["shipped"].append(_record(pred, gt[c], memos[c]))
        out["diag"].append({
            "raw_vox": int(raw.sum()),
            "max_prob": float(p.max()),
            "max_prob_in_gt": float(p[gt[c]].max()) if gt[c].any() else None,
        })

    p = np.asarray(prob[ET])
    for thr in THR_GRID:
        raw = p > thr
        for pp in PP_GRID:
            out["et_grid"][grid_key(thr, pp)] = _record(_clean_channel(raw, *pp), gt[ET], memos[ET])
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate(recs):
    dice = np.array([r["dice"] for r in recs], dtype=float)
    clean = [r["hd95"] for r in recs if r["status"] == "clean"]
    return {
        "dice": float(np.nanmean(dice)) if np.isfinite(dice).any() else 0.0,
        "hd95": float(np.mean([r["hd95"] for r in recs])),
        "hd95_clean": float(np.mean(clean)) if clean else 0.0,
        "sens": float(np.mean([r["sens"] for r in recs])),
        "iou": float(np.mean([r["iou"] for r in recs])),
        "n_halluc": int(sum(r["status"] == "halluc" for r in recs)),
        "n_miss": int(sum(r["status"] == "miss" for r in recs)),
        "n_clean": len(clean), "n": len(recs),
    }


def regime(patients, et_key=None):
    """Per-patient (TC, WT, ET) records: shipped TC/WT, ET shipped or from the grid."""
    return [[p["shipped"][0], p["shipped"][1],
             p["shipped"][ET] if et_key is None else p["et_grid"][et_key]] for p in patients]


def headline(rows):
    aggs = [aggregate([r[c] for r in rows]) for c in range(3)]
    return {
        "channels": dict(zip(CHANNELS, aggs)),
        "mean_dice": float(np.mean([a["dice"] for a in aggs])),
        "mean_hd95": float(np.mean([a["hd95"] for a in aggs])),
    }


def anchor_check(run, rows):
    with open(os.path.join(cfg.paths.logs_dir, run, "testing", "test_metrics.csv")) as f:
        logged = next(csv.DictReader(f))
    h = headline(rows)
    max_overlap, max_hd95, counts_ok, lines = 0.0, 0.0, True, []
    for c, name in enumerate(CHANNELS):
        agg, lc = h["channels"][name], name.lower()
        for key, col in (("dice", f"dice_{lc}"), ("hd95", f"hd95_{lc}"),
                         ("hd95_clean", f"hd95_{lc}_clean"), ("sens", f"sens_{lc}"),
                         ("iou", f"iou_{lc}")):
            d = abs(agg[key] - float(logged[col]))
            if key.startswith("hd95"):
                max_hd95 = max(max_hd95, d)
            else:
                max_overlap = max(max_overlap, d)
            lines.append(f"{col}: cache {agg[key]:.5f} logged {float(logged[col]):.5f}")
        for key, col in (("n_halluc", f"n_halluc_{lc}"), ("n_miss", f"n_miss_{lc}")):
            counts_ok &= agg[key] == int(logged[col])
    # Two inference passes are not bit-identical on GPU: a few boundary voxels
    # near the threshold flip. Measured on the Sept-2026 runs, that moves mean
    # HD95 by up to 0.007 mm (Mamba) / 0.0008 mm (ViT) and Dice/sens/IoU by
    # < 1e-4, so HD95 gets a mm-scale tolerance. The counts must match exactly.
    return {"max_abs_diff_overlap": max_overlap, "max_abs_diff_hd95_mm": max_hd95,
            "counts_match": bool(counts_ok),
            "ok": bool(counts_ok and max_overlap < 1e-3 and max_hd95 < 0.05), "detail": lines}


def select_et(val_patients):
    """The rule, fixed before test is read.

    knee:     at the shipped cleanup, the LOWEST ET threshold whose val count of
              hallucinated + missed patients equals the grid minimum (lowest,
              because above the floor a higher threshold only costs Dice).
    min_hd95: the (threshold, cleanup) with the lowest val ET HD95, ties to the
              higher val ET Dice, then the lower threshold.
    """
    aggs = {k: aggregate([p["et_grid"][k] for p in val_patients])
            for k in val_patients[0]["et_grid"]}
    default = [(thr, aggs[grid_key(thr, DEFAULT_PP)]) for thr in THR_GRID]
    floor = min(a["n_halluc"] + a["n_miss"] for _, a in default)
    knee_thr = min(thr for thr, a in default if a["n_halluc"] + a["n_miss"] == floor)
    min_hd95 = min(aggs, key=lambda k: (round(aggs[k]["hd95"], 6), -aggs[k]["dice"],
                                        float(k.split("/")[0])))
    return {"knee": grid_key(knee_thr, DEFAULT_PP), "min_hd95": min_hd95,
            "val_mismatch_floor": int(floor), "val_grid": aggs}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(stat_fn, n, seed=0):
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vals = np.array([stat_fn(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def wilcoxon_p(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(b[mask] != a[mask]):
        return 1.0
    return float(stats.wilcoxon(b[mask], a[mask]).pvalue)


def paired_row(label, a, b):
    """a, b: per-patient values (nan = not scored). Difference is b - a."""
    mask = np.isfinite(a) & np.isfinite(b)
    n = len(a)
    return {
        "metric": label, "a": float(np.nanmean(a)), "b": float(np.nanmean(b)),
        "diff": float(np.nanmean(b) - np.nanmean(a)),
        "ci": bootstrap_ci(lambda idx: np.nanmean(b[idx]) - np.nanmean(a[idx]), n),
        "p": wilcoxon_p(a, b), "n": int(mask.sum()),
    }


def paired_comparison(rows_a, rows_b):
    out = []
    dice_a = np.array([[r[c]["dice"] for c in range(3)] for r in rows_a])
    dice_b = np.array([[r[c]["dice"] for c in range(3)] for r in rows_b])
    hd_a = np.array([[r[c]["hd95"] for c in range(3)] for r in rows_a])
    hd_b = np.array([[r[c]["hd95"] for c in range(3)] for r in rows_b])
    n = len(rows_a)

    def mean_of_channels(x):
        return lambda idx: float(np.mean(np.nanmean(x[idx], axis=0)))

    out.append({
        "metric": "mean Dice (headline)", "a": mean_of_channels(dice_a)(np.arange(n)),
        "b": mean_of_channels(dice_b)(np.arange(n)),
        "diff": mean_of_channels(dice_b)(np.arange(n)) - mean_of_channels(dice_a)(np.arange(n)),
        "ci": bootstrap_ci(lambda idx: mean_of_channels(dice_b)(idx) - mean_of_channels(dice_a)(idx), n),
        "p": None, "n": n,
    })
    for c, name in enumerate(CHANNELS):
        out.append(paired_row(f"Dice {name}", dice_a[:, c], dice_b[:, c]))
    out.append({
        "metric": "mean HD95 (headline, mm)", "a": float(hd_a.mean()), "b": float(hd_b.mean()),
        "diff": float(hd_b.mean() - hd_a.mean()),
        "ci": bootstrap_ci(lambda idx: hd_b[idx].mean() - hd_a[idx].mean(), n),
        "p": None, "n": n,
    })
    for c, name in enumerate(CHANNELS):
        out.append(paired_row(f"HD95 {name} (mm)", hd_a[:, c], hd_b[:, c]))
    for c, name in enumerate(CHANNELS):
        both_clean = np.array([ra[c]["status"] == "clean" and rb[c]["status"] == "clean"
                               for ra, rb in zip(rows_a, rows_b)])
        a = np.where(both_clean, hd_a[:, c], np.nan)
        b = np.where(both_clean, hd_b[:, c], np.nan)
        out.append(paired_row(f"HD95 {name}, clean in both (mm)", a, b))
    return out


def modality_dice(counts, ignore_empty):
    tp, pred, gt = counts[..., 0], counts[..., 1], counts[..., 2]
    denom = pred + gt
    d = np.where(denom == 0, 1.0, 2.0 * tp / np.maximum(denom, 1))
    return np.where(gt == 0, np.nan, d) if ignore_empty else d


def logged_modality_deltas(run):
    with open(os.path.join(cfg.paths.logs_dir, run, "log.txt")) as f:
        block = f.read().rsplit("[XAI: modality]", 1)[-1]
    rows = {}
    for m in re.finditer(r"^\s*-(FLAIR|T1ce|T1|T2)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                         block, re.M):
        rows.setdefault(m.group(1), [float(m.group(i)) for i in (2, 3, 4)])
    return rows


def modality_study(counts, run):
    # Anchor: the in-train XAI runs over the whole test loader and scores
    # both-empty as 1.0; re-derive its deltas that way and compare with log.txt.
    d_all = modality_dice(counts, ignore_empty=False)
    delta_all = (d_all[:, 1:, :] - d_all[:, :1, :]).mean(axis=0)
    logged = logged_modality_deltas(run)
    anchor = max((abs(delta_all[m, c] - logged[name][c])
                  for m, name in enumerate(MODALITIES) if name in logged for c in range(3)),
                 default=float("nan"))

    d = modality_dice(counts, ignore_empty=True)
    delta = d[:, 1:, :] - d[:, :1, :]               # (N, 4, 3)
    table = {"baseline": [float(np.nanmean(d[:, 0, c])) for c in range(3)], "delta": {}}
    for m, name in enumerate(MODALITIES):
        table["delta"][name] = []
        for c in range(3):
            x = delta[:, m, c]
            table["delta"][name].append({
                "mean": float(np.nanmean(x)),
                "ci": bootstrap_ci(lambda idx: np.nanmean(x[idx]), len(x)),
                "n": int(np.isfinite(x).sum()),
            })
    return {"anchor_max_abs_diff_vs_log": float(anchor), "logged_3_patients": logged,
            "full": table, "per_patient_delta": delta}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def fmt_p(p):
    return "—" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def et_table(aggs, header):
    lines = [header, "| ET thr | halluc | miss | clean | ET Dice | ET HD95 | ET HD95 clean |",
             "|---|---|---|---|---|---|---|"]
    for thr in THR_GRID:
        a = aggs[grid_key(thr, DEFAULT_PP)]
        lines.append(f"| {thr:.2f} | {a['n_halluc']} | {a['n_miss']} | {a['n_clean']} | "
                     f"{a['dice']:.4f} | {a['hd95']:.2f} | {a['hd95_clean']:.2f} |")
    return lines


def point_row(label, key, h):
    et = h["channels"]["ET"]
    thr, mc, mt = key.split("/") if key else ("shipped", "", "")
    pp = f"{mc}/{mt}" if key else "176/352"
    return (f"| {label} | {thr} | {pp} | {et['n_halluc']}h + {et['n_miss']}m | {et['dice']:.4f} | "
            f"{et['hd95']:.2f} | {et['hd95_clean']:.2f} | {et['sens']:.4f} | "
            f"{h['mean_dice']:.4f} | {h['mean_hd95']:.2f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="Root passed to cache_predictions.py --out")
    parser.add_argument("--runs", nargs="+", required=True, help="One or two run names (a b)")
    parser.add_argument("--workers", type=int, default=min(28, os.cpu_count() or 1))
    parser.add_argument("--out", default=None, help="Report folder")
    args = parser.parse_args()

    if args.out is None:
        args.out = (os.path.join(cfg.paths.logs_dir, f"compare_{args.runs[0]}_vs_{args.runs[1]}")
                    if len(args.runs) == 2 else os.path.join(cfg.paths.logs_dir, args.runs[0], "eval"))
    os.makedirs(args.out, exist_ok=True)

    results = {}
    with Pool(args.workers) as pool:
        for run in args.runs:
            run_cache = os.path.join(args.cache, run)
            with open(os.path.join(run_cache, "meta.json")) as f:
                meta = json.load(f)
            infer = meta["infer"]
            res = {"meta": meta, "patients": {}}
            # Scoring is the slow part (HD95 over every grid point); reuse it
            # when nothing it depends on has changed.
            signature = json.loads(json.dumps({
                "ids": meta["ids"], "infer": infer, "thr_grid": THR_GRID, "pp_grid": PP_GRID}))
            scored_path = os.path.join(run_cache, "scored.json")
            if os.path.exists(scored_path):
                with open(scored_path) as f:
                    saved = json.load(f)
                if saved["signature"] == signature:
                    res["patients"] = saved["patients"]
                    print(f"[{run}] reusing {scored_path}", flush=True)
            for split in ("val", "test"):
                if split in res["patients"]:
                    continue
                tasks = [(os.path.join(run_cache, split), i, infer["thresholds"],
                          infer["min_component_voxels"], infer["min_total_voxels"])
                         for i in range(len(meta["ids"][split]))]
                print(f"[{run}] scoring {split}: {len(tasks)} patients", flush=True)
                res["patients"][split] = sorted(pool.imap_unordered(eval_patient, tasks),
                                                key=lambda p: p["idx"])
            with open(scored_path, "w") as f:
                json.dump({"signature": signature, "patients": res["patients"]}, f)
            results[run] = res

    report = {}
    md = ["# ET operating point and paired ViT vs Mamba statistics", "",
          "Generated by `tools/operating_point_study.py` from TTA probabilities cached by "
          "`tools/cache_predictions.py` (each run's own checkpoint, env and inference recipe).", ""]

    # 1. anchors -------------------------------------------------------------
    md += ["## 1. Anchor check: cache reproduces `testing/test_metrics.csv`", "",
           "| run | encoder | torch | max abs diff Dice/sens/IoU | max abs diff HD95 (mm) | "
           "halluc/miss counts | verdict |",
           "|---|---|---|---|---|---|---|"]
    for run, res in results.items():
        anchor = anchor_check(run, regime(res["patients"]["test"]))
        res["anchor"] = anchor
        md.append(f"| {run} | {res['meta']['encoder']} | {res['meta']['torch']} | "
                  f"{anchor['max_abs_diff_overlap']:.1e} | {anchor['max_abs_diff_hd95_mm']:.4f} | "
                  f"{'match' if anchor['counts_match'] else 'DIFFER'} | "
                  f"{'OK' if anchor['ok'] else 'MISMATCH, do not use'} |")
    md.append("")

    # 2. ET operating point -------------------------------------------------
    md += ["## 2. ET operating point, selected on validation", "",
           "Rule fixed before reading test. **knee**: at the shipped cleanup (min_component 176, "
           "min_total 352 voxels), the lowest ET threshold that reaches the validation minimum of "
           "hallucinated + missed patients. **min-HD95**: the (threshold, cleanup) with the lowest "
           "validation ET HD95. TC/WT stay at the shipped thresholds throughout.", ""]
    for run, res in results.items():
        val, test = res["patients"]["val"], res["patients"]["test"]
        sel = select_et(val)
        n_neg_val = sum(p["shipped"][ET]["gt_vox"] == 0 for p in val)
        n_neg_test = sum(p["shipped"][ET]["gt_vox"] == 0 for p in test)
        test_grid = {k: aggregate([p["et_grid"][k] for p in test]) for k in test[0]["et_grid"]}
        points = {
            "shipped": headline(regime(test)),
            "knee": headline(regime(test, sel["knee"])),
            "min_hd95": headline(regime(test, sel["min_hd95"])),
        }
        val_points = {
            "shipped": headline(regime(val)),
            "knee": headline(regime(val, sel["knee"])),
            "min_hd95": headline(regime(val, sel["min_hd95"])),
        }
        res["selection"] = sel
        report[run] = {"anchor": res["anchor"], "knee": sel["knee"], "min_hd95": sel["min_hd95"],
                       "val_mismatch_floor": sel["val_mismatch_floor"],
                       "val_points": val_points, "test_points": points,
                       "val_grid": sel["val_grid"], "test_grid": test_grid,
                       "n_et_negative": {"val": int(n_neg_val), "test": int(n_neg_test)}}

        thr = res["meta"]["infer"]["thresholds"]
        md += [f"### {run} ({res['meta']['encoder']})", "",
               f"Shipped thresholds TC/WT/ET {thr[0]}/{thr[1]}/{thr[2]}. ET-negative patients: "
               f"val {n_neg_val}/{len(val)}, test {n_neg_test}/{len(test)} (only these can be hallucinated). "
               f"Val mismatch floor at the shipped cleanup: {sel['val_mismatch_floor']}. "
               f"Selected: knee `{sel['knee']}`, min-HD95 `{sel['min_hd95']}` (thr/min_comp/min_total).", ""]
        for title, pts in (("Validation", val_points), ("**Test**", points)):
            md += [f"{title} at the candidate points:", "",
                   "| point | ET thr | cleanup | ET mismatch | ET Dice | ET HD95 | ET HD95 clean | ET sens | mean Dice | mean HD95 |",
                   "|---|---|---|---|---|---|---|---|---|---|",
                   point_row("shipped (sweep argmax)", None, pts["shipped"]),
                   point_row("val knee", sel["knee"], pts["knee"]),
                   point_row("val min-HD95", sel["min_hd95"], pts["min_hd95"]), ""]
        md += et_table(sel["val_grid"], "Validation grid, shipped cleanup:") + [""]
        md += et_table(test_grid, "Test grid, shipped cleanup (transparency only, not used to select):") + [""]

    # 3. paired ---------------------------------------------------------------
    if len(args.runs) == 2:
        a, b = args.runs
        ids_a, ids_b = results[a]["meta"]["ids"]["test"], results[b]["meta"]["ids"]["test"]
        assert ids_a == ids_b, "test patient order differs between runs; cannot pair"
        md += [f"## 3. Paired test comparison, {b} minus {a} (same {len(ids_a)} patients)", "",
               "Difference is b − a. CI: 95% percentile bootstrap over patients "
               f"({N_BOOT} resamples). p: Wilcoxon signed-rank on per-patient values. Dice skips "
               "empty-GT patients; HD95 includes the 374 mm empty-mismatch sentinel, the 'clean in "
               "both' rows do not.", ""]
        report["paired"] = {}
        for name, key_a, key_b in (("shipped operating points", None, None),
                                   ("val-selected ET knee",
                                    results[a]["selection"]["knee"], results[b]["selection"]["knee"])):
            rows = paired_comparison(regime(results[a]["patients"]["test"], key_a),
                                     regime(results[b]["patients"]["test"], key_b))
            report["paired"][name] = rows
            md += [f"### {name}", "", f"| metric | {a} | {b} | diff | 95% CI | p | n |",
                   "|---|---|---|---|---|---|---|"]
            for r in rows:
                md.append(f"| {r['metric']} | {r['a']:.4f} | {r['b']:.4f} | {r['diff']:+.4f} | "
                          f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {fmt_p(r['p'])} | {r['n']} |")
            md.append("")

        # 4. empty mismatches
        md += ["## 4. Categorical failures on test (shipped operating points)", "",
               f"| patient | channel | GT voxels | {a} | {b} | {a} max prob in GT | {b} max prob in GT |",
               "|---|---|---|---|---|---|---|"]
        for i, pid in enumerate(ids_a):
            for c, ch in enumerate(CHANNELS):
                sa = results[a]["patients"]["test"][i]["shipped"][c]
                sb = results[b]["patients"]["test"][i]["shipped"][c]
                if sa["status"] in ("halluc", "miss") or sb["status"] in ("halluc", "miss"):
                    da = results[a]["patients"]["test"][i]["diag"][c]["max_prob_in_gt"]
                    db = results[b]["patients"]["test"][i]["diag"][c]["max_prob_in_gt"]

                    def cell(s):
                        return (f"**{s['status']}** ({s['pred_vox']} vox)"
                                if s["status"] in ("halluc", "miss") else f"{s['status']} ({s['pred_vox']} vox)")
                    md.append(f"| {pid} | {ch} | {sa['gt_vox']} | {cell(sa)} | {cell(sb)} | "
                              f"{'—' if da is None else f'{da:.3f}'} | {'—' if db is None else f'{db:.3f}'} |")
        md.append("")

    # 4. modality ---------------------------------------------------------------
    md += ["## 5. Modality ablation over the whole test split", "",
           "No TTA, no cleanup, shipped thresholds, over the same test patients as the in-train "
           "`utils/xai.py:modality_attribution`. Dice skips empty-GT patients here; the in-train XAI "
           "counts both-empty as 1.0, and the anchor re-derives its deltas that way from the cache "
           "and compares them with log.txt. Values are Dice change when the modality is zeroed, "
           "mean [95% bootstrap CI].", ""]
    mod = {}
    for run, res in results.items():
        counts_path = os.path.join(args.cache, run, "modality_counts.npy")
        if not os.path.exists(counts_path):
            continue
        counts = np.load(counts_path)
        if (counts < 0).any():
            md.append(f"{run}: modality cache incomplete, skipped.")
            continue
        mod[run] = modality_study(counts, run)
        full = mod[run]["full"]
        report.setdefault(run, {})["modality"] = {k: v for k, v in mod[run].items()
                                                  if k != "per_patient_delta"}
        md += [f"### {run}: n={len(counts)}, anchor max abs diff vs log.txt "
               f"{mod[run]['anchor_max_abs_diff_vs_log']:.4f}", "",
               "| removed | TC | WT | ET |", "|---|---|---|---|",
               "| (baseline Dice) | " + " | ".join(f"{x:.4f}" for x in full["baseline"]) + " |"]
        for name in MODALITIES:
            md.append(f"| -{name} | " + " | ".join(
                f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" for d in full["delta"][name]) + " |")
        md.append("")
    if len(args.runs) == 2 and all(r in mod for r in args.runs):
        a, b = args.runs
        md += [f"Paired difference in the modality effect, {b} minus {a} "
               "(positive = the modality's removal hurts b less):", "",
               "| removed | TC | WT | ET |", "|---|---|---|---|"]
        report["paired_modality"] = {}
        for m, name in enumerate(MODALITIES):
            cells = []
            for c in range(3):
                x = mod[a]["per_patient_delta"][:, m, c]
                y = mod[b]["per_patient_delta"][:, m, c]
                r = paired_row(f"-{name} {CHANNELS[c]}", x, y)
                report["paired_modality"][r["metric"]] = r
                cells.append(f"{r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] p={fmt_p(r['p'])}")
            md.append(f"| -{name} | " + " | ".join(cells) + " |")
        md.append("")

    # per-patient CSVs ----------------------------------------------------------
    for run, res in results.items():
        path = os.path.join(cfg.paths.logs_dir, run, "eval", "per_patient_test.csv")
        knee = res["selection"]["knee"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient"] + [f"{k}_{ch.lower()}" for ch in CHANNELS
                                      for k in ("dice", "hd95", "status", "gt_vox", "pred_vox")]
                       + ["dice_et_knee", "hd95_et_knee", "status_et_knee"])
            for pid, p in zip(res["meta"]["ids"]["test"], res["patients"]["test"]):
                row = [pid]
                for c in range(3):
                    s = p["shipped"][c]
                    row += [s["dice"], s["hd95"], s["status"], s["gt_vox"], s["pred_vox"]]
                k = p["et_grid"][knee]
                w.writerow(row + [k["dice"], k["hd95"], k["status"]])
        print(f"wrote {path}")

    md_path = os.path.join(args.out, "operating_point_study.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.out, "operating_point_study.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {md_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()

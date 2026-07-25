# Run Commands

Every runnable entry point in this repo, what it produces, and roughly what it
costs. Append a new section when a new script or stage lands.

All commands are run from the repo root, inside the env that has
torch/monai/medpy (e.g. `conda activate pytorch2`). Paths come from
`config.py` — `cfg.paths.root_dir` must point at the dataset, and
`cfg.paths.logs_dir` (default `logs/`) is where every run folder is written.

Timings are for the workstation the runs actually happen on: **1x RTX A6000**,
156.27M params, 1314.67 GMac at 128x128x96.

---

## Quick reference

| Command | Retrains? | Cost | Writes to |
|---|---|---|---|
| `pytest` | no | seconds | — |
| `python train.py --name <run>` | **yes** | ~45h | `logs/<run>/` (everything) |
| `python evaluate.py --run <run> --tune-thresholds` | no | ~2h | `logs/<run>/eval/` |
| `python xai.py --run <run>` | no | ~30-45min | `logs/<run>/xai/` |
| `python tools/crop_visual_check.py` | no | seconds | PNGs beside the script |
| `python tools/nifti_viewer.py` | no | seconds | interactive window |

---

## 1. Tests — run this first, always

```bash
pytest                          # whole suite
pytest -m "not gpu"             # skip the CUDA-only tests (works on a laptop)
pytest tests/test_xai.py -v     # one file
pytest -k threshold             # one topic
```

Fast, synthetic-data only. Catches the config/wiring breaks that would
otherwise surface hours into a training run. **Run before kicking off anything
long.**

---

## 2. Training — the full pipeline in one command

```bash
python train.py --name v2-tierB
```

Omit `--name` and it prompts. Runs, in order:

1. **Train** (`cfg.epoch`, currently 100, with warmup + cosine + EMA)
2. **Tune thresholds** on the *validation* split (`cfg.infer.tune_thresholds_after_training`)
3. **Test** with the full inference recipe — tuned thresholds, TTA, post-processing
4. **XAI suite** (`cfg.xai.run_after_training`), all five components

Produces:

```
logs/<run>/
  log.txt                  full stdout capture
  config_snapshot.json     cfg as it was at run start
  metrics.csv              one row per epoch
  checkpoints/             best_metric_model.pth  (EMA weights when EMA is on)
  plots/                   loss / dice / hd95 / iou curves
  visualizations/          pre-training qualitative data checks
  attention/               per-epoch attention overlays
  eval/                    threshold_sweep.json, infer_config.json
  testing/                 test_metrics.csv + qualitative outputs
  xai/                     figures + per-component JSON + summary.json
```

To skip the tail stages, set `cfg.infer.tune_thresholds_after_training = False`
and/or `cfg.xai.run_after_training = False` in `config.py`.

**A failing XAI component cannot discard the run** — it is caught, logged into
`summary.json`, and the checkpoint reloaded.

---

## 3. Evaluation — re-score a saved checkpoint, no retraining

```bash
python evaluate.py --run v1-run3                        # current recipe as-is
python evaluate.py --run v1-run3 --tune-thresholds      # tune on val, then test
python evaluate.py --run v1-run3 --tag notta --no-tta   # ablate flip-TTA
python evaluate.py --run v1-run3 --tag raw --no-postprocess   # ablate CC cleanup
```

Writes to `logs/<run>/eval/test_metrics_<tag>.csv` (default tag `eval`), leaving
the original run's `testing/` untouched. `--tag` is what keeps ablations from
overwriting each other.

Everything it changes lives at inference time: sliding-window overlap and
blending, flip-TTA, per-channel thresholds, connected-component cleanup — see
`cfg.infer`.

**Isolating the HD95 spacing fix.** `cfg.metrics.voxel_spacing` was corrected
from `(1,1,1)` to the true `(1.875, 1.875, 1.0)`, which raises every in-plane
HD95, while post-processing lowers ET HD95. Those two land on the same number.
To tell them apart:

```bash
python evaluate.py --run v1-run3 --tag raw --no-postprocess --no-tta
python evaluate.py --run v1-run3 --tag full --tune-thresholds
```

`raw` isolates the units correction; the difference to `full` is Tier A's real
effect.

---

## 4. Explainability — standalone

```bash
python xai.py --run v1-run3                       # all five components
python xai.py --run v1-run3 --only cam            # iterate on one
python xai.py --run v1-run3 --only modality faithful
```

| Component | What it produces |
|---|---|
| `cam` | X1 Seg-Grad-CAM + HiResCAM overlays, 4 decoder depths x 3 classes |
| `modality` | X2 4x3 ΔDice heatmap — Dice cost of removing each MRI sequence |
| `uncertainty` | X3 MC-dropout entropy maps + error-retention curves |
| `rollout` | X4 attention rollout through all 12 ViT blocks |
| `faithful` | X5 deletion curves vs random null, localisation, randomisation sanity check |

`faithful` always runs **last** regardless of the order you pass — it randomises
the model weights, so anything after it would be explaining a destroyed model.
The checkpoint is reloaded immediately after.

Tunables live in `cfg.xai`: `sample_indices` (default `[0,1,2]` — every extra
sample multiplies the faithfulness sweeps), `cam_layers`, `cam_methods`,
`cam_roi`, `mc_passes`, `deletion_fractions`.

**Read `xai/faithful.json` first.** If the randomisation SSIM does not decay, or
CAM deletion-AUC is no better than the random null, the explanations carry no
information and nothing else in `xai/` is publishable. See `plan.md` §0.7.1,
gates G4/G5.

---

## 5. Manual data-inspection tools

Not part of the pipeline; edit the constants at the top of the script first.

```bash
python tools/crop_visual_check.py    # set SAMPLE_DIR, START_SLICE, THRESHOLD first
python tools/nifti_viewer.py         # file dialogs pick the image + segmentation
```

`crop_visual_check.py` runs the *real* `CropRawDepthd` / `CropForegroundHWd`
classes, not copies, and dumps every raw depth slice before/after the crop. Its
`[CropRawDepthd] one-time check ...` line should match the one printed at the
start of a real training run — if the numbers agree, training is cropping
exactly what the PNGs show.

---

## Recommended sequence on a fresh workstation

```bash
pytest                                                # 1. wiring is sound
python evaluate.py --run v1-run3 --tune-thresholds    # 2. Tier A on the OLD checkpoint (~2h)
python xai.py --run v1-run3                           # 3. XAI baseline on the OLD checkpoint
python train.py --name v2-tierB                       # 4. Tier B retrain (~45h, does 2+3 itself)
```

Steps 2–3 give a Tier-A-only baseline to compare the Tier B run against, which
is what separates "the retrain helped" from "the inference recipe helped".

---

## Appending to this file

One `##` section per entry point. Include: the command, what it writes, roughly
what it costs, and any ordering constraint or footgun. Add a row to the quick
reference table at the top.

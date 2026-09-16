# Run Commands

Every runnable entry point in this repo, what it produces, and roughly what it
costs. Append a new section when a new script or stage lands.

All commands are run from the repo root, inside the env that has
torch/monai/medpy (e.g. `conda activate pytorch2`). **Runs with the Mamba
encoder need the `mamba` env** (`conda activate mamba`, section 6); it has
everything pytorch2 has, so every command below also works there. Paths come from
`config.py` — `cfg.paths.root_dir` must point at the dataset, and
`cfg.paths.logs_dir` (default `logs/`) is where every run folder is written.

Timings are for the workstation the runs actually happen on: **1x RTX A6000**,
156.27M params, 1314.67 GMac at 128x128x96.

---

## Quick reference

| Command | Retrains? | Cost | Writes to |
|---|---|---|---|
| `pytest` | no | seconds | — |
| `python train.py --name <run>` | **yes** | ~45-48h (60 ep) | `logs/<run>/` (everything) |
| `python train.py --name <run> --resume` | **yes** | remaining epochs | `logs/<run>/` (same folder) |
| `tools/train_supervisor.sh <run>` | **yes** | ~45h + restarts | `logs/<run>/` (same folder) |
| `python evaluate.py --run <run> --tune-thresholds` | no | ~2h | `logs/<run>/eval/` |
| `python xai.py --run <run>` | no | ~30-45min | `logs/<run>/xai/` |
| `python tools/smoke_test.py` | no | ~10-15min | temp dirs only |
| `python tools/crosscheck_segmamba.py` | no | ~1min | — |
| `python tools/compare_runs.py v2-run3 <run>` | no | seconds | `logs/compare_<a>_vs_<b>/` |
| `python tools/replot.py --run <run>` | no | seconds (CPU) | `logs/<run>/plots/` |
| `python tools/cache_predictions.py --run <run> --out <dir>` | no | ~1-1.5h per run (GPU) | `<dir>/<run>/` (~8 GB) |
| `python tools/operating_point_study.py --cache <dir> --runs <a> <b>` | no | ~20-30min (CPU) | `logs/compare_<a>_vs_<b>/operating_point_study.{md,json}`, `logs/<run>/eval/per_patient_test.csv` |
| `tools/setup_mamba_env.sh` | no | ~10min | `/DATA/conda_envs/mamba` |
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

1. **Train** (`cfg.epoch`, currently 60, with warmup + cosine + EMA; per-epoch validation at `cfg.infer.val_sw_overlap` = 0.5)
2. **Tune thresholds** on the *validation* split (`cfg.infer.tune_thresholds_after_training`)
3. **Test** with the full inference recipe — tuned thresholds, TTA, post-processing, `cfg.infer.sw_overlap` = 0.75
4. **XAI suite** (`cfg.xai.run_after_training`), all five components

Produces:

```
logs/<run>/
  log.txt                  full stdout capture
  config_snapshot.json     cfg as it was at run start
  metrics.csv              one row per epoch
  train_steps.csv          one row per optimizer step (section 8)
  val_steps.csv            one row per validation patient per epoch (section 8)
  checkpoints/             best_metric_model.pth  (EMA weights when EMA is on)
  plots/                   loss / dice / iou / hd95 / lr / loss_steps curves (redraw: section 8)
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

## 2a. Resuming an interrupted run

Every epoch writes `logs/<run>/checkpoints/last.pth` — the **full training
state**, not just weights: raw weights, the EMA copy, AdamW's moments, the LR
scheduler's position, the GradScaler's scale, the RNG streams, and the
best-metric bookkeeping. A crash costs one epoch, not the run.

```bash
python train.py --name v2-run3 --resume        # continue; fails if there's no last.pth
python train.py --name v2-run3 --auto-resume   # continue if possible, else start fresh
python train.py --name v2-run3 --resume-from logs/other/checkpoints/last.pth
```

A resumed run writes back into the **same** `logs/<run>/` folder (no
`logs/<run>_1`) and appends to the existing `metrics.csv`, so the epoch curves
in `plots/` still cover the whole run. Rows for epochs at or past the resume
point are dropped first, so a replayed epoch cannot appear twice.

`--resume` refuses to continue if `cfg.unetr.img_shape` changed since the
checkpoint was written — that would silently make the two halves of the run
different experiments. Changing `cfg.epoch` is allowed but warns: the LR
schedule was built for the old budget and is restored as-is.

**Unattended runs — restart automatically on any failure:**

```bash
nohup tools/train_supervisor.sh v2-run3 > logs/v2-run3-supervisor.log 2>&1 &
```

Relaunches `train.py --name v2-run3 --auto-resume` on any non-zero exit, up to
20 restarts (override: `tools/train_supervisor.sh v2-run3 40`), waiting
`BACKOFF_SECONDS` (default 60) between attempts so the dead process's GPU
memory is actually released first. A clean finish or a Ctrl-C ends the loop; it
does not restart through an interrupt you asked for.

There are two recovery layers, and they cover different things:

| Layer | Handles | Cost of an event |
|---|---|---|
| In-process OOM skip (`utils/engine.py`) | one unlucky batch or validation volume | one sample dropped from the epoch |
| Supervisor restart | hard OOM past the skip limit, killed process, dead CUDA context | replay of at most one epoch |

`cfg.train_oom_skip_limit` / `cfg.val_oom_skip_limit` (default 5 each) bound the
first layer. Exceeding either re-raises, which is what escalates to the second.

---

## 2b. GPU memory — why v2-run2 died at epoch 33, and what changed

The run did not leak in the Python sense. A live-CUDA-tensor census across
train/validate cycles stayed flat (1616 → 1592 tensors). What ran out was
*usefully shaped* memory:

```
Allocated       4.0 GB       <- actually live
Reserved       23.9 GB       <- held by the caching allocator
Non-releasable  5.1 GB       <- stranded inside partly-used segments (peak 7.1 GB)
Peak allocated 21.9 GB       <- a single training step
```

The cause is two allocation regimes sharing one pool. Training is a fixed
`2x4x128x128x96` batch every step; validation feeds a **whole foreground-cropped
brain**, a different shape for every patient (`130x171x141`, `137x162x130`,
`150x164x131`, …). Alternating them every epoch splits the reserved segments
into blocks neither regime can reuse. With a ~21 GB working set on a 48 GB card
that is survivable for a while — and then, ~33 epochs in, a training step cannot
find a contiguous block and dies with plenty of "free" memory on the card.

Changes:

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**, set by `train.py`
   before CUDA initialises (`cfg.checkpoint.cuda_alloc_conf`). Lets one virtual
   segment grow and be re-carved instead of stranding fixed blocks. Override
   from the shell if you need to: `PYTORCH_CUDA_ALLOC_CONF=... python train.py`.
2. **`empty_cache()` at every train↔validate boundary**, so each phase starts
   against an uncarved pool.
3. **The attention hook is armed, not always-on.** It fired on every
   sliding-window patch of every validation sample — ~27 windows x ~180 patients
   x 50 epochs — cloning a `(1, heads, patches, patches)` tensor each time and
   leaving the last one resident on the GPU for the whole run. It now captures
   only on the epochs that actually save an overlay, and to CPU.
4. **AUC's `.cpu()` copy of the raw logits is gated** on
   `cfg.metrics.auc_every_n_epochs` instead of running every epoch and being
   discarded nine times out of ten.
5. **Per-step tensors are released before the loader hands over the next
   batch**, so the peak stops carrying one extra step of full-resolution
   decoder activations.

**Every epoch line now ends with a memory reading:**

```
gpu_mem alloc=3.4G reserved=23.3G peak=21.4G retries=0 ooms=0
```

`reserved` climbing while `alloc` stays flat is the drift. A non-zero `retries`
means the allocator is already flushing and retrying to satisfy requests — that
is the last warning before an OOM, and it is now in `log.txt` hours ahead of the
crash.

**A crash now leaves its traceback in `logs/<run>/log.txt`.** It previously did
not: `RunLogger.__exit__` restored `sys.stderr` before Python printed the
traceback, so v2-run2's log ends on a tidy `Run finished` with no error in it
anywhere.

If OOM recurs despite all of the above, the levers that actually shrink the
21 GB working set — in increasing order of how much they change the experiment —
are `cfg.crop.num_samples` (2 → 1, halves the effective batch),
`cfg.infer.sw_overlap` (0.5 → 0.25, fewer windows in flight), and
`cfg.unetr.img_shape`. The last one is **not** resumable into an existing run.

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
| `rollout` | X4 attention rollout through all 12 ViT blocks; for a Mamba run, the hidden-attention rollout of its deepest stage (same 8x8x6 grid) |
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

## 6. ViT vs Mamba encoder comparison (SegMamba, Sept 2026)

The UNETR encoder is swappable: `cfg.unetr.encoder = "vit" | "mamba"` in
`config.py` (default now `"mamba"`), or `python train.py --encoder vit|mamba`
for one launch. `"mamba"` is a SegMamba encoder (Xing et al., MICCAI 2024,
BraTS 2023) in `blocks/VisionMamba.py`; the decoder, loss, schedule, inference
recipe and XAI are shared, so the two runs differ in the encoder only.
`evaluate.py`/`xai.py` rebuild whichever encoder a run was trained with from
its `config_snapshot.json`, so old ViT runs keep working unchanged.

**Environment.** The Mamba CUDA kernels cannot load in `pytorch2` (torch 2.11;
every newer mamba_ssm wheel needs glibc >= 2.32, this box has 2.31). The
`mamba` env is pytorch2 with torch 2.6 + mamba_ssm 2.2.4 swapped in; recreate
it with `tools/setup_mamba_env.sh`. `train.py` refuses to start a Mamba run
from an env without the kernels.

**Commands, in order:**

```bash
conda activate mamba
cd "/DATA/Abul Hasan/Glioma Revision"

pytest -q                                    # 1. 243 tests, ~3 min (kernel tests need this env)
python tools/crosscheck_segmamba.py          # 2. encoder == official SegMamba, ~1 min
python tools/smoke_test.py                   # 3. real data: train/val/TTA/XAI + time projection

nohup tools/train_supervisor.sh v3-mamba-30ep > logs/v3-mamba-30ep-supervisor.log 2>&1 &
tail -f logs/v3-mamba-30ep/log.txt           # 4. train 30 ep -> tune -> test -> all 5 XAI

python tools/compare_runs.py v2-run3 v3-mamba-30ep    # 5. side-by-side, after it finishes
```

The supervisor inherits the active env's `python`; with another env active,
prefix it with `PYTHON_BIN=/DATA/conda_envs/mamba/bin/python`.

**Cost** (measured by `tools/smoke_test.py` on this box, Mamba encoder):
4.3 s per training step (data loading 1.6 s, so compute-bound), 10.4 s per
validation volume, 5.7 s per 8-flip TTA volume; peak GPU memory 22.6 GB in
training, 5.4 GB in validation. Projected **~56 min/epoch, ~28 h for 30
epochs**, plus ~1 h for threshold tuning, the TTA test pass and the five XAI
components. The ViT run `v2-run3` ran 41-45 min/epoch. If time gets tighter,
`cfg.val_interval = 2` saves ~9 min/epoch but halves the resolution of the
growth curves this comparison is about.

**What `compare_runs.py` writes** (`logs/compare_v2-run3_vs_v3-mamba-30ep/`):
`curves.png` (val mean/ET Dice, ET HD95 and train loss per epoch, both runs on
one axis), `summary.md` + `summary.csv` (params, min/epoch, val at the
matched epoch 30 and at each run's best, test Dice/HD95/sensitivity with the
HD95 breakdown, XAI modality deltas, uncertainty retention, deletion AUCs).

**Caveats to state in the paper.** The ViT baseline `v2-run3` trained for 50
epochs, this run for 30, each with its own cosine schedule: compare the curves
and the epoch-30 row, not only the final test numbers. The two runs used
different torch builds (2.11 vs 2.6). The Mamba encoder adds dropout that
SegMamba does not have (for MC-dropout parity with the ViT) — the only
deviation from the official architecture, which `crosscheck_segmamba.py`
verifies layer by layer.

---

## 7. ET operating point, paired statistics, full-test modality ablation (no retrain)

```bash
C=/some/scratch/pred_cache            # ~8 GB per run; not under logs/
/DATA/conda_envs/pytorch2/bin/python tools/cache_predictions.py --run v2-run3 --out $C
/DATA/conda_envs/mamba/bin/python    tools/cache_predictions.py --run v3-mamba-30ep --out $C
python tools/operating_point_study.py --cache $C --runs v2-run3 v3-mamba-30ep
```

`cache_predictions.py` runs each run's own checkpoint with its own
`eval/infer_config.json` (TTA, overlap, blending) over val and test once, and
saves the sigmoid probabilities that `run_test` thresholds. On test it also
runs the modality ablation for every patient (no TTA, shipped thresholds) and
saves per-patient `[tp, |pred|, |gt|]` counts. Run each run in the env it was
trained in, so the numbers match its logged test pass. Add `--limit 2` for a
smoke test. Re-running skips patients that are already cached. The two
commands can share the GPU; that way each takes about 1.5 h.

`operating_point_study.py` is CPU only. Its report sections:
1. **Anchor check.** The cache must reproduce `testing/test_metrics.csv` at the
   shipped operating point. If it does not, do not read further.
2. **ET operating point chosen on val.** The rule is fixed in the script: the
   knee is the lowest ET threshold that reaches val's minimum of
   hallucinated + missed patients at the shipped cleanup. It also reports the
   val min-HD95 point. Test is shown at the chosen points, and the test grid
   only for transparency. (Session 5's knee was read off a test-set grid.)
3. **Paired test comparison of the two runs.** Bootstrap CIs and Wilcoxon p on
   the same 70 patients, at the shipped points and at each run's val knee.
4. **Patients with empty mismatches**, side by side.
5. **Modality ablation over the test split** with bootstrap CIs, ET scored with
   empty-GT patients skipped, and a paired difference between the runs. The
   in-train XAI covers the same patients, but it counts both-empty as Dice 1.0
   (on ET that mixes presence/absence into the modality effect) and reports no
   CIs. The anchor re-derives the in-train numbers from the cache.

---

## 8. Per-step CSVs and re-plotting the curves (no retraining)

```bash
python tools/replot.py --run <run>                                    # cfg.plot defaults
python tools/replot.py --run <run> --font-size 16 --fig-size 8 5 --dpi 300 --format png pdf
python tools/replot.py --run <run> --epoch-smoothing 0 --step-smoothing 0 --out figures/<run>
```

Seconds, CPU only. It reads only the run's CSVs and rewrites `logs/<run>/plots/`
(or `--out`). It works on older runs too; they have no `train_steps.csv`, so they
get no per-step curve and no main-head train-loss line.

Every value in the CSVs is raw, never smoothed:

| File | One row per | Columns |
|---|---|---|
| `metrics.csv` | epoch | lr, train/val loss, per-region + mean Dice, HD95, Sens, Spec, IoU, mIoU, F1, AUC |
| `train_steps.csv` | optimizer step | epoch, step, global_step, lr, step time, total loss, main-head loss, both aux-head losses, unweighted Dice / Focal-Tversky / Hausdorff terms, hd_scale |
| `val_steps.csv` | validation patient, per epoch | epoch, sample_index, patient, step time, val loss + its terms, Dice / IoU / mIoU / HD95 / Sens / Spec / F1 / AUC per region + mean |

Footguns:
- `train_loss` in metrics.csv includes the deep-supervision heads
  (+0.3·aux_z6 + 0.15·aux_z3); `val_loss` is the main head only. Compare `val_loss`
  with `loss_main` (the dashed line in `loss.png`), not with `train_loss`.
- Region Dice in metrics.csv averages over patients whose ground truth has that
  region (MONAI's `ignore_empty`). In val_steps.csv that patient's `dice_*` is
  NaN, and the row's `mean_dice` averages the regions present, so it is not the
  epoch's `mean_dice`.
- AUC is computed every `cfg.metrics.auc_every_n_epochs` only. On other epochs
  val_steps.csv leaves `auc_*` blank and metrics.csv writes 0.
- Smoothing (`cfg.plot.epoch_smoothing` 0.3, `step_smoothing` 0.9) is a
  bias-corrected EMA applied when drawing. The raw curve stays visible under it
  and the weight is printed on the figure. Take reported numbers from the CSVs,
  never off a smoothed curve.
- Per-epoch validation runs at `cfg.infer.val_sw_overlap` (0.5), the test pass
  at `cfg.infer.sw_overlap` (0.75), so a val curve and a test number differ in
  overlap as well as split.
- `lr` in metrics.csv (and so `lr.png`) is logged after that epoch's scheduler
  step, i.e. it is the next epoch's LR. `lr` in train_steps.csv is the LR each
  step actually used.

---

## 9. Deliberate data-leak run (assignment demonstration)

`cfg.data.leak = "patient"` (config.py) adds every validation patient to the
training set, in `utils/dataloader.py:BratsDataset._split_datalist`. The val and
test splits themselves do not change, and no test patient is trained on. **Set it
back to `None` for any honest run.**

```bash
conda activate mamba
python train.py --name v5-mamba-leak-demo                 # the name MUST contain "leak"
python train.py --name v5-mamba-leak-demo --auto-resume   # if it stops partway
python tools/replot.py --run v5-mamba-leak-demo
```

Labelled by design: train.py exits if the run name lacks "leak"; log.txt prints a
LEAK banner with the train/val overlap count; `config_snapshot.json` records
`data.leak`; every curve in `plots/` carries a LEAKED SPLIT watermark (replot.py
reads the snapshot, so redrawn plots keep it).

How to read it: per-epoch validation, the best-epoch choice and the tuned
thresholds are all computed on patients the model trained on; the test patients
were not. The gap between this run's val and test metrics, and between its val
curve and an honest run's at matched epochs, is the size of the leak. Expect it
to be small on this dataset: under the honest split 72 of the 105 val patients
already have an identical BraTS19/BraTS20 twin in train (journal, Session 8), so
only 33 are newly leaked.

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

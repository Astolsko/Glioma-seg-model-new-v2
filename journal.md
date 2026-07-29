# Journal — Glioma Segmentation (modified UNETR)

Running log of changes across sessions. Newest entry on top. One entry per
change or decision. Keep it terse: what changed, why, and the measured effect
(fill the result in after the workstation run).

Metric shorthand: Dice/HD95 reported as TC / WT / ET (+ mean). Lower HD95 better.

Baseline to beat (`logs/v1-run3`, 50 ep) — HD95 here is in the OLD, WRONG units
(spacing was `(1,1,1)`; true in-plane is 1.875mm), so multiply in-plane by up to
1.875x before comparing it to anything newer:
- Val Dice (best ep46): 0.821 / 0.892 / **0.783** / 0.832
- Val HD95 ep50 (mm): 4.26 / 4.12 / **21.0** / 9.80
- Test Dice: 0.817 / 0.892 / 0.818 / 0.842
- Test HD95 (mm): 9.10 / 3.97 / **39.7** / 17.6

Current state (`logs/run1-new-version`, 100 ep, EMA weights @ ep91), HD95 in
CORRECT units, test pass with TTA + tuned thresholds + post-processing:
- Val Dice (best ep91): 0.818 / 0.880 / **0.730** / 0.809
- Test Dice: 0.848 / 0.894 / **0.753** / 0.832
- Test HD95 (mm): 4.55 / 4.63 / 24.50 / 11.22
- Test HD95 **excluding empty-mismatch patients**: 4.55 / 4.63 / **3.65**
- 19.6 min/epoch (baseline: 42.0)

---

## 2026-07-29 — Session 4: read the Tier-B run; ET diagnosis overturned

The Tier A+B bundle ran as `logs/run1-new-version` (100 ep, 32.7h). This session
read it, ran three no-retrain ablations off its checkpoint, and found that the
premise of Sessions 1–3 was wrong.

**THE FINDING — ET HD95 was never a boundary problem.**
The new per-direction breakdown in `run_test` decomposes test ET HD95 = 24.50mm as:

| component | contribution |
|---|---|
| 4 patients x 374.0 empty-mismatch sentinel | **21.37mm (87%)** |
| 60 patients x 3.65mm real boundary | 3.13mm |
| 6 patients correctly both-empty (scored 0.0) | 0.00mm |

Verified exact: `4*374 + 60*3.6497 + 6*0 = 1714.98 / 70 = 24.49976`, matching the
reported value to 14 digits. **ET's real boundary quality is 3.65mm — BETTER than
TC (4.55) and WT (4.63).** TC and WT have zero empty-mismatch, so their numbers
were always clean. `plan.md` §0.1's "ET HD95 is the disaster, 5-10x worse than
TC" — the premise for three sessions of Hausdorff annealing, boundary weighting
and connected-component work — is **false**. ET geometry is the best of the
three. What remains is a **presence/absence error on 4 of 70 patients**.
(plan.md §0.1/§0.3 corrected accordingly.)

**Results vs `v1-run3`. Val is the clean comparison (identical protocol);
the test rows are NOT an ablation — the inference recipe changed too.**

| Val, best epoch | v1-run3 ep46 | run1 ep91 | Δ |
|---|---|---|---|
| Dice TC / WT / ET | 0.821 / 0.892 / 0.783 | 0.818 / 0.880 / 0.730 | -0.002 / -0.012 / **-0.053** |
| mean Dice | 0.832 | 0.809 | **-0.023** |
| Sens TC / WT / ET | 0.837 / 0.881 / 0.748 | 0.794 / 0.847 / 0.679 | -0.044 / -0.033 / **-0.069** |

**Sensitivity fell on ALL THREE channels while specificity rose — a uniform
recall regression, not an ET-specific one.** ET shows it worst because ET is
smallest. Strip post-processing out of the test number and ET is 0.7637 vs the
baseline's 0.8184 = **-0.055, matching the val -0.053 almost exactly**: the val
and test ET drops are one phenomenon, and it is training-side.

**What each change actually bought:**
- Aux heads -> Dice only: **19.6 vs 42.0 min/epoch = 2.14x faster.** The one
  clean, uncontaminated win (it removed 2 of 3 scipy distance transforms/step).
- Epochs 50 -> 100: **+0.009 val mean Dice for 16 hours.** 99% of the final value
  was reached at ep50, 99.5% at ep65; ep66-100 moved it 0.002. Not worth it.
- Post-processing: costs **0.011 ET Dice**, buys **10.1mm ET HD95** (mismatches
  ~6 -> 4). Net positive but crude — the 0.011, over ~60 scored patients, is
  about one real-ET patient being zeroed outright.
- HD95 spacing fix: numbers are honest now. **Most of the apparent "ET HD95 2-3x
  better" was post-processing, not the retrain** — raw ET HD95 is 34.63mm.
- The remaining seven training-side changes: net **-0.023 val mean Dice**, and
  **which one caused it is unknown.** Seven changes landed in one run.

**Phase 0 ablations run (all from `run1-new-version`'s EMA checkpoint, no retrain):**

| tag | thresholds | postproc | Dice TC/WT/ET | mean | HD95 ET |
|---|---|---|---|---|---|
| (published) | 0.3/0.3/0.5 | on | 0.845 / 0.895 / 0.753 | 0.831 | 24.50 |
| `nopp` | 0.3/0.3/0.5 | off | 0.845 / 0.895 / 0.764 | 0.834 | 34.63 |
| `wide` | 0.15/0.15/0.5 | on | 0.848 / 0.894 / 0.753 | 0.832 | 24.50 |

TC/WT identical to 4 d.p. across tags => the split is deterministic and these
are true single-variable ablations.

**BUG found in the threshold sweep (fixed).** `search_thresholds._dice` scored
both-empty as **1.0**, but the reported metric — MONAI `DiceMetric(ignore_empty=True)`
in `run_test` — **excludes** empty-GT patients. So the sweep was rewarded for
zeroing ET on ET-negative patients on a metric that never sees it. Symptom: an
otherwise perfectly monotone ET curve with a lone **+0.017 spike at exactly 0.50**
(raise the threshold, one ET-negative patient tips under `min_total=100`, gets
zeroed, collects a free +1.0 ~ +0.01 on ~100 val patients; at 0.55 a different
patient with real ET is zeroed and gives it back). **run1-new-version's published
ET threshold of 0.5 was picked by one lucky patient.**

**Calibration is badly off.** With the grid widened to 0.15, TC and WT both
pinned at the new floor again, monotone decreasing across 0.15-0.70. EMA logit
shrinkage is real and larger than expected; every val Dice ever reported (all at
0.5) understates the model. Grid now extends to 0.05.

**XAI verdicts against `plan.md` §0.7.1 gates:**
- **G6 modality — PASS, and it is the strongest result in the run.** Removing
  T1ce: ET 0.706 -> 0.119 (Δ **-0.588**), TC Δ -0.505, WT only -0.066. Removing
  FLAIR: WT 0.892 -> 0.311 (Δ **-0.581**). Exactly the radiological prior, and
  falsifiable. T2 is near-redundant (Δ -0.03 to -0.05) — worth a sentence.
- **G7 uncertainty — PASS.** Referring the 2% most-uncertain voxels lifts s0 ET
  Dice 0.699 -> 0.952, s1 ET 0.682 -> 0.978. Quote one operating point, not the curve.
- **G4/G5 — VOID, and it is a code bug, not a finding about the model.**
  `component_faithful` uses `cfg.xai.cam_layers[-1]` = `decoder0_header.1`, which
  is **one 1x1x1 conv from the output** (`decoder0_header.2`). So
  d(score)/d(activation) is *exactly zero* outside the ROI and **the CAM support
  IS the ground-truth mask** — `cam.json` confirms it, `mass_in_tumor = 0.9999999`.
  Consequences: the deletion curve measures "deleting the tumour destroys the
  prediction" (trivially true); randomisation SSIM cannot move (0.9860 -> 0.9863
  flat, even after randomising the transformer) because the support is pinned and
  SSIM over a ~95%-zero volume is background-dominated; and `localization`'s
  `top_frac=0.05` selects ~78k voxels against a ~3-8k support, so `elsewhere:
  0.87-0.99` is argsort breaking ties by array index. Also `mass_in_tumor` is a
  tautology under `cam_roi="gt"`.
  **Decision: headline is Dice/HD95, so X1/X5 are CUT from the paper, not
  repaired.** Keep `modality` + `uncertainty` only.

**Decisions taken this session (user):**
1. `v1-run3`'s checkpoint is **gone** — Tier A's isolated effect is permanently
   unrecoverable. Stop trying; do not re-plan around it.
2. **One more training run only.**
3. **Move to native 1mm** (plan §0.7.2 #1). Numbers will drop; that is expected
   and is the point.
4. Paper headline is **Dice/HD95**; XAI is a supporting section.

**Code changed (all no-retrain, all verified byte-compiling):**
- `config.py`: `infer.thresholds` default `(0.5,0.5,0.5)` -> **`(0.3,0.3,0.5)`** so
  an `--no-postprocess` ablation changes ONE thing vs the published run; plus the
  mm^3 rescale warning on `min_component_voxels`/`min_total_voxels`.
- `utils/postprocess.py`: grid `0.30-0.70` -> **`0.05-0.70`**; **empty-GT patients
  now skipped** (matches `DiceMetric(ignore_empty=True)`); sweep JSON gained
  `n_scored`/`n_samples`; `tune_and_save` gained `filename=` so a re-tune cannot
  clobber the sweep it is trying to beat.
- `evaluate.py`: writes `threshold_sweep_<tag>.json`.
- `utils/engine.py`: `run_test` now prints and records the **HD95 breakdown** —
  `n_halluc` (pred non-empty, GT empty) / `n_miss` (pred empty, GT non-empty) /
  `hd95_*_clean` over both-non-empty cases / `n_test`. The two mismatch
  directions need OPPOSITE fixes to `min_total_voxels`, so lumping them was how
  you tune that knob backwards.
- `tests/test_postprocess.py`: new test pinning the empty-GT skip.

---

## NEXT SESSION — start here

### 1. A command was left running. Read its output first.

```bash
pytest -q
python evaluate.py --run run1-new-version --tag final --tune-thresholds   # ~1h
```

Outputs: `logs/run1-new-version/eval/test_metrics_final.csv`,
`eval/threshold_sweep_final.json`, and an `HD95 breakdown ...` line in stdout.
If it was not run, run it — it is the last cheap information available.

**What to look for, and what each answer changes:**

| Look at | If | Then |
|---|---|---|
| `n_halluc_et` vs `n_miss_et` | mostly **hallucinated** | the model invents ET on ET-negative patients. RAISE `min_total_voxels`. Predicted: ~3 halluc + ~1 miss |
| same | mostly **missed** | `min_total=100` is deleting real ET. LOWER it. This is the OPPOSITE of plan.md gate G2's assumption — G2 is written on the old, wrong diagnosis |
| `hd95_et_clean` | still ~3.6mm | confirms ET boundaries are fine; do NOT spend the 1mm run on boundary work |
| ET best threshold | no longer 0.5 | confirms the both-empty bug was what picked it. Expect it to move below 0.5 |
| TC/WT best threshold | pinned at 0.05 again | calibration is worse still; note it, do not chase it — the 1mm run re-tunes anyway |

### 2. Then design the single 1mm run. Change list, and nothing beyond it.

Target is the **recall regression** (-0.055 ET Dice, -0.03 to -0.07 sens on all
three channels). It is the only measured defect left.

1. **Native 1mm.** Delete `transforms.build_resize`; `RandCropByPosNegLabeld(spatial_size=(128,128,128), pos=2, neg=1)`
   for train, `CropForegroundd` + sliding window for val/test. Drop
   `CropRawDepthd` on **all** splits — leaving the fixed `[40:136)` window on
   val/test carries finding D straight into the "honest" numbers.
2. **Aux heads: Dice + Focal-Tversky** (Hausdorff stays OFF). The 2.14x speedup
   came from dropping the scipy distance transform; Tversky is cheap GPU work and
   it was the false-negative pressure term. Keep the speedup, get the recall back.
3. **`RandBiasFieldd` before `ApplyCLAHEAndZscored`, or drop it.** It currently
   runs at `transforms.py:206`, AFTER normalisation at `:175` — a multiplicative
   field on signed z-scored data is not a bias field, and it perturbs exactly the
   T1ce enhancement contrast ET is defined by.
4. **Epochs 50, not 100.** See the measured +0.009-for-16h above.
5. **`cfg.xai.components = ["modality", "uncertainty"]`.** X1/X5 are cut, and
   dropping `faithful` also removes a weight-randomising step from the end of a
   long run.

**Do NOT touch the Hausdorff term.** 3.65 / 4.55 / 4.63mm says it is working.
It was tempting to cut it; the measurement says no.

**Three landmines in the 1mm migration, in order of how silently they corrupt:**
1. `cfg.metrics.voxel_spacing` is **derived** as `(240/img_shape[0], ...)`. With
   `img_shape=(128,128,128)` it computes 1.875 and inflates every HD95 by 1.875x
   while looking correct. **Hard-code `(1,1,1)` and delete the derivation.**
2. `Transformer(cube_size=img_shape)` sizes the positional embedding. Train patch
   size, `cfg.unetr.img_shape` and sliding-window `roi_size` must ALL be
   `(128,128,128)`.
3. `min_component_voxels`/`min_total_voxels` are VOXEL counts. One voxel goes
   3.52mm^3 -> 1mm^3, so the same numbers become a **3.5x weaker** cleanup.
   Rescale to `(0,0,176)` / `(0,0,352)`.

**Cost estimate: 45-60 min/epoch, ~40-50h at 50 epochs** (128^3 is 1.33x the
activations and 1.78x the attention of 128x128x96; sliding-window val over full
volumes is the real blowup, hence `CropForegroundd` on val).

### 3. Known and accepted after that run

- **The recall regression will not have been isolated.** 1mm numbers are not
  comparable to any of the four existing runs, and there is no budget left. If
  1mm ET lands low, reason from the Phase 0 ablations above, not from a
  controlled comparison. This is the priced-in cost of one run.
- **The test split is ~10% (n=70) with no CV.** A ±0.03 ET swing is inside its
  noise. Every test-set claim above carries this caveat. 5-fold CV
  (plan §0.7.2 #2) is still the thing a reviewer will ask for first.

---

## 2026-07-26 — Session 3: XAI suite + Tier A (inference) + Tier B (retrain bundle)

**New findings from reading the code (not in plan.md, ranked):**
1. **Images were resampled with nearest-neighbour.** `Resized(keys=["image","label"], mode="nearest")` passed ONE mode for BOTH keys, so every MRI was point-sampled 240→128 in-plane. A 1.875× downsample with no filtering = aliasing, on exactly the fine T1ce texture ET depends on. Prime suspect for the ET ceiling.
2. **HD95 was reported in the wrong units.** Post-`Resized` in-plane spacing is 240/128 = 1.875 mm, but `cfg.metrics.voxel_spacing` was `(1,1,1)`. Every in-plane HD95 was understated 1.875×.
3. Training + evaluation both happen at 1.875 mm (BraTS scores at 1 mm). Not fixed — user chose to keep the current protocol (Tier C1 deferred).
4. Fixed absolute depth window `[40:136)` for every patient is never verified dataset-wide.
5. **Deep supervision ran the full composite loss on both aux heads** → 3× `HausdorffDTLoss` per step, i.e. 3 scipy CPU distance transforms. Large share of the 2500 s epoch.
6. Inference used MONAI defaults (overlap 0.25, constant blending), no TTA, fixed 0.5 threshold, no post-processing.
7. Minor: `compute_miou` averages background IoU (~0.99, near-meaningless); `RandRotate90(max_k=3)` is anatomically implausible for brain; `GradScaler` under bf16 is a no-op; Adam+`weight_decay` is coupled L2.

**Did — Tier A (inference only, applies to existing checkpoints):**
- `cfg.infer` section; `run_inference` gained `overlap=0.5` + gaussian blending and 8-way flip TTA. TTA averages in **probability** space then maps back via `logit()` so the function's contract stays "always returns logits" — no sometimes-probabilities footgun. Validation passes `tta=False` (8× cost every epoch would exceed the training it monitors).
- New `utils/postprocess.py`: per-channel `binarize`, connected-component `postprocess` (drop small components; zero the channel below a floor — the ET/HD95 374.0 empty-mismatch fix), and `search_thresholds` sweeping per-channel thresholds on val by mean per-patient Dice.
- New `evaluate.py`: re-runs a saved checkpoint with the full recipe into `logs/<run>/eval/`. `--tune-thresholds` tunes on **val**, reports on test.
- **HD95 units fixed**: `cfg.metrics.voxel_spacing` now derived as `(240/img_shape[0], 240/img_shape[1], 1.0)`. Only HD95 reads it — Dice/IoU/sens/spec/F1/AUC are counting metrics and are unchanged. **HD95 will go UP; that is the honest number, not a regression.**

**Did — Tier B (needs one retrain):**
- `Resized` → `mode=("trilinear","nearest")`, `anti_aliasing=(True,False)`, factored into `transforms.build_resize` so train and val cannot drift.
- Adam → **AdamW**; linear warmup (5 ep) → cosine via `SequentialLR`; **EMA** via `torch.optim.swa_utils.AveragedModel` (stdlib, not hand-rolled). Validation, checkpointing and test all use the EMA weights.
- Aux deep-supervision heads now get **Dice only** (`combine_main_and_aux` falls back to the raw callable when there is no `.dice`, so existing tests still pass).
- Epochs 50 → 100 (run3 plateaued ~ep40 with LR already at its 1e-6 floor).
- `RandRotate90` → small-angle `RandAffined` (±15°); added `RandBiasFieldd`.

**Did — XAI (X1–X5), new `utils/xai.py` + `xai.py` CLI, all post-hoc:**
- X1 Seg-Grad-CAM + HiResCAM, ROI-restricted, at 4 decoder depths (`decoder0_header.1`, not `.2` — the last child IS the 1×1 output conv, so a CAM there just redraws the prediction).
- X2 modality ablation → 4×3 ΔDice table (the radiologically checkable result: ET should collapse without T1ce, WT without FLAIR).
- X3 MC-dropout (dropout-only train mode, so the deep-supervision forward signature doesn't change) → mean prob / epistemic std / entropy + error-retention curve.
- X4 attention rollout (Abnar & Zuidema) replacing the query-averaged last-layer map.
- X5 deletion curves + **random-ordering null baseline**, localisation (inside / peritumoral / elsewhere), and the Adebayo cascading-randomisation sanity check.

**Did — wired threshold tuning + XAI into `train.py`** so a single `python train.py` produces the whole artifact set (train → tune on val → test → explain), all inside the one run folder:
- XAI orchestration moved OUT of the `xai.py` script and into `utils/xai.py:run_xai_suite()`; `xai.py` is now a thin CLI over it. One implementation, two entry points — the standalone script and the end of a training run cannot drift.
- Threshold tuning de-duplicated into `utils/postprocess.py:tune_and_save()`, shared by `train.py` and `evaluate.py` for the same reason.
- `RunLogger` gained `eval_dir` and `xai_dir`, so `logs/<run>/` now also holds `eval/{threshold_sweep,infer_config}.json` and `xai/`.
- New flags: `cfg.infer.tune_thresholds_after_training`, `cfg.xai.run_after_training`, `cfg.xai.components`.
- `run_xai_suite` catches a failing component, logs it into the summary and reloads the checkpoint — a broken explanation must never discard a finished ~45 h run. Components always execute in `COMPONENTS` order regardless of what the caller asks, because `faithful` randomises the weights and anything after it would be explaining a destroyed model.
- Modality ablation runs with `tta=False`: it is a relative comparison where every arm gets identical treatment, so 8× TTA would multiply its cost to move no conclusion.

**Files:** `config.py`, `train.py`, `utils/{engine,losses,transforms,plot,run_logger}.py`, new `utils/{postprocess,xai}.py`, new `evaluate.py`, `xai.py`, new `tests/test_{postprocess,xai}.py`, updated `tests/test_{config,engine}.py`.

**Not run here** — laptop still has no torch/monai/pytest. All files byte-compile. **ACTION on workstation: `pytest` first, then `python evaluate.py --run v1-run3 --tune-thresholds` (Tier A, ~2 h, no retrain) before starting the Tier B run.**

**Expected:** Tier A → ET HD95 21 → single digits, mean Dice +0.01–0.02, zero training cost. Tier B → +0.015–0.03 mean Dice. Note Tier A and the spacing fix land on the SAME numbers, so record HD95 before/after the spacing change separately or the two effects are indistinguishable — `evaluate.py --no-postprocess` gives the isolating run.

**Next session — read `plan.md` §0.7 first.** It holds the deferred backlog and the decision gates keyed to what the new run's metrics and XAI outputs actually show.

**Also added `RUN.md`** — every entry point with its command, output layout, cost and ordering constraints, plus the recommended fresh-workstation sequence. Append a `##` section there whenever a new script lands.

---

## 2026-07-24 — Session 2: loss rework (Focal-Tversky + focal-bug fix + HD anneal)

**Did (backlog items 1–3 from Session 1):**
- **Fixed focal double-sigmoid bug** by removing MONAI `FocalLoss` entirely (it was fed `sigmoid(outputs)` but wants logits).
- **Replaced the Focal term with Focal-Tversky** (α=0.7, β=0.3, γ=4/3 ⇒ exponent 0.75), kept at 3 loss components: Dice + Focal-Tversky + Hausdorff-DT. α>β up-weights false negatives → targets ET/TC (their sensitivity was ~0.73). Uniform α/β across channels for now (per-channel weighting is a future knob).
- **Annealed the Hausdorff weight** 0→full over the first 50% of epochs (`hd_anneal_frac=0.5`), via `CombinedLoss.set_epoch()` called each epoch in `run_training`.

**Files:** `utils/losses.py` (rewritten as `CombinedLoss` class), `config.py` (loss section: `focal_weight/focal_gamma` → `tversky_weight/tversky_alpha/tversky_beta/tversky_gamma/hd_anneal_frac`), `utils/engine.py` (set_epoch call), `tests/test_losses.py` + `tests/test_config.py` (updated fixtures, added recall-weighting & anneal tests).

**Not run here** — laptop has no torch/monai/pytest. Byte-compiled OK; arithmetic verified by hand. **ACTION: run `pytest tests/test_losses.py tests/test_config.py` on the workstation before training.**

**Expected effect (to confirm on workstation run):** ET/TC Dice ↑ (recall focus), training more stable early (HD ramped in), focal gradients no longer squashed. Watch: Focal-Tversky can inflate FP → pair with ET post-processing (backlog #4) if ET HD95 doesn't drop.

**Next:** backlog #4 (ET min-volume post-processing) and #5 (AdamW+EMA), pending user go-ahead.

---

## 2026-07-24 — Session 1: codebase audit + plan reconciliation (no code changes)

**Did:** Read the whole pipeline (`models/unetr.py`, `utils/{losses,transforms,dataloader,engine,metrics}.py`, `config.py`, blocks) and cross-checked `plan.md` against the real code and `logs/v1-run3` results. Rewrote `plan.md` §0 as the authoritative reconciliation.

**Key findings (plan.md was stale — written against an older model):**
- ALREADY DONE, plan listed as TODO: CLAHE→z-score (CLAHE is dead code in `dataloader.py`), overlapping sigmoid ET/TC/WT labels, Hausdorff-DT boundary loss, deep supervision (aux z6/z3 heads), dilated-ASPP bottleneck, most augmentation. Model also already has ConvNeXt-V2 (7³ depthwise) + CoordAtt at every decoder stage.
- Plan's baseline "~0.88 mean Dice" is wrong: actual ~0.83.
- Plan's HD95 diagnosis is INVERTED: it says TC HD95 (~13mm) is worst; reality is **ET HD95 is the disaster (21mm val / 40mm test)**, TC/WT ~4mm. ET is the bottleneck on both Dice and HD95.
- ET HD95 pattern = stray FP ET blobs + empty-mismatch (`compute_hd95` returns 374 when exactly one of pred/gt empty). ⇒ inference post-processing is the biggest untapped ET-HD95 lever.

**Real backlog identified (see plan §0.3):**
1. BUG: `losses.py:14` double-sigmoids the Focal term (passes `sigmoid(outputs)` to MONAI FocalLoss, which wants logits).
2. No Focal-Tversky recall weighting for ET/TC false-negatives.
3. Hausdorff term un-annealed (constant 0.2 from ep1).
4. No connected-component / min-volume ET suppression at inference.
5. Plain Adam, no EMA (AdamW + EMA are cheap wins).

**Decision:** implement in order — (1) fix focal bug → (2) Focal-Tversky → (3) ET post-proc + anneal HD → (4) AdamW+EMA → (5) block swap only if ET still <0.90. One change at a time, keep only what improves ET without regressing TC/WT.

**Next:** await user's pick of which step to implement first.

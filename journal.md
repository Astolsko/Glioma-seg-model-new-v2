# Journal — Glioma Segmentation (modified UNETR)

Running log of changes across sessions. Newest entry on top. One entry per
change or decision. Keep it terse: what changed, why, and the measured effect
(fill the result in after the workstation run).

Metric shorthand: Dice/HD95 reported as TC / WT / ET (+ mean). Lower HD95 better.

Baseline to beat (`logs/v1-run3`, 50 ep):
- Val Dice (best ep46): 0.821 / 0.892 / **0.783** / 0.832
- Val HD95 ep50 (mm): 4.26 / 4.12 / **21.0** / 9.80
- Test Dice: 0.817 / 0.892 / 0.818 / 0.842
- Test HD95 (mm): 9.10 / 3.97 / **39.7** / 17.6

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

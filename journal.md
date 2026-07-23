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

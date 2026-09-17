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

Current state, ViT encoder (`logs/v2-run3`, native 1mm, 50 ep, best ep41), HD95
in CORRECT units, test pass with TTA + tuned thresholds + post-processing:
- Val Dice (best ep41): 0.861 / 0.900 / **0.790** / 0.850 (0.841 mean at ep30)
- Test Dice: 0.878 / 0.911 / **0.816** / 0.868
- Test HD95 (mm): 3.94 / 4.84 / **45.76** / 18.18 (ET is sentinel-dominated)
- Test HD95 **excluding empty-mismatch patients**: ET **3.51**; n_halluc/n_miss ET 6/2
- Test ET sensitivity 0.760; 41.8 min/epoch, 35.1 h total
- (Previous: `logs/run1-new-version`, resized, test Dice 0.832 mean. Do NOT
  compare 1mm runs to the resized ones.)

Current state, Mamba encoder (`logs/v3-mamba-30ep`, SegMamba, native 1mm, 30 ep,
best ep23), same test recipe:
- Val Dice (best ep23): 0.881 / 0.911 / **0.810** / 0.868 (0.864 mean at ep30)
- Test Dice: 0.867 / 0.915 / **0.819** / 0.867
- Test HD95 (mm): 10.12 / 5.82 / **35.16** / 17.03 (TC: 1 missed patient; ET 4 halluc + 2 miss)
- Test HD95 clean: TC 4.84, WT 5.82, ET **3.62**
- Test ET sensitivity 0.764; 44.8 min/epoch, 22.4 h training; 102.1M params (encoder 15.7M)
- **Paired vs the ViT on the same 70 test patients: mean Dice -0.0015
  [-0.0108, +0.0052], a tie.** Details in Session 7.

> **Next up:** Session 7 read the Mamba run, then ran the val-selected ET
> operating point, the paired statistics and the full-test modality ablation.
> Start at the Session 7 entry's "Open items". Commands are in `RUN.md` sections 6 and 7.

---

## 2026-09-14 — Session 8: 60 epochs, test overlap 0.75, per-step CSVs + replot; duplicate patients found

- **Data finding (data left as is, user's decision):** `data/combined` holds
  369 `BraTS20_Training_*` + 334 `BraTS19_*` cases, and every BraTS19 case is an
  identical copy of a BraTS20 case (same seg label counts; FLAIR arrays
  identical on the 3 pairs checked). The seed-0 split puts the twin of 72/105
  val and 55/70 test patients in train. Test Dice (TC/WT/ET) for the 15 test
  patients with no twin in train vs the 55 with one: ViT 0.904/0.913/0.824 vs
  0.871/0.910/0.814; Mamba 0.899/0.912/0.817 vs 0.858/0.916/0.820. No detectable
  gain from the twins, but n=15 leaves wide error bars.
- **config.py:** `epoch` 30 → 60, `warmup_epochs` 3 → 5. `infer.sw_overlap`
  0.5 → 0.75 (threshold tuning + test). New `infer.val_sw_overlap = 0.5` keeps
  per-epoch validation at its old cost; 0.75 there would add an estimated day
  over 60 epochs. New `cfg.plot` section: font, size, dpi, formats, and EMA
  smoothing (0.3 per epoch, 0.9 per step).
- **New CSVs** (RUN.md §8): `train_steps.csv` has one row per optimizer step
  (total, main-head, aux and per-term losses, lr). `val_steps.csv` has one row
  per val patient per epoch (loss + terms, every metric per region). Both are
  resume-safe like metrics.csv.
- **Plots** are drawn from the CSVs, and `tools/replot.py` redraws any run.
  Smoothing is applied only when drawing, with the raw curve faint underneath
  and the weight on the figure. `loss.png` adds the main-head train loss, since
  `train_loss` includes the deep-supervision heads and `val_loss` does not. New
  `lr.png` and `loss_steps.png`; IoU and HD95 now plotted per region. The plot
  style no longer leaks into global rcParams, so XAI figures drawn later keep
  matplotlib defaults. A plotting error no longer aborts train.py before
  tuning/test.
- **Comparability:** a 60-epoch run tested at 0.75 overlap differs from
  `v2-run3` / `v3-mamba-30ep` on both counts. `python evaluate.py --run <old>`
  now also runs at 0.75, which separates the overlap effect without retraining.
- **Deliberate leak for an assignment demo (user's requirement):**
  `cfg.data.leak = "patient"` adds every val patient to the training set
  (`BratsDataset._split_datalist`); test is untouched. Labelled by design: the
  run name must contain "leak", log banner, config snapshot, watermarked plots.
  Set `None` for honest runs. RUN.md §9.

## 2026-09-13 — Session 7: Mamba run read out; ET knee on val; paired ViT vs Mamba stats

`v3-mamba-30ep` finished cleanly at 2026-09-13 00:28 IST. Training ended at 23:36;
tuning, the test pass and all 5 XAI components followed. No restarts, 0 allocator
retries, 0 OOMs, peak 22.6 GB. **44.8 min/epoch, not the smoke test's 56**
(ViT 41.8, so Mamba is 7% slower per epoch); 22.4 h of training vs the ViT's 34.9 h.

**Readout** (`logs/compare_v2-run3_vs_v3-mamba-30ep/{summary.md,curves.png}`).
- Val: Mamba is above the ViT at every epoch from ep 2. At ep 30 it has 0.864 vs
  0.841 mean (TC +0.028, WT +0.017, ET +0.024). It passes the ViT's 50-epoch
  best (0.8504) at ep 15 (0.8518), peaks at ep 23 (0.8676) and stays flat after.
- Test at the shipped point (sweep thresholds 0.15/0.10/0.05, TTA, cleanup
  176/352): 0.867 / 0.915 / 0.819 / 0.867, vs the ViT's 0.878 / 0.911 / 0.816 /
  0.868. **The val lead does not carry over to test.**

**New tools** (`RUN.md` section 7).
- `tools/cache_predictions.py` caches each run's TTA probabilities on val and
  test, using the run's own checkpoint, env and infer_config, plus a per-patient
  modality ablation on test.
- `tools/operating_point_study.py` scores the cache on CPU.
- Outputs: `logs/compare_v2-run3_vs_v3-mamba-30ep/operating_point_study.{md,json}`
  and `logs/<run>/eval/per_patient_test.csv`.
- Anchors: the cache reproduces both runs' `testing/test_metrics.csv`. Dice,
  sens and IoU agree within 1e-4, HD95 within 0.007 mm (two GPU inference passes
  are not bit-identical), and the hallucination/miss counts exactly. The
  modality anchor reproduces both in-train log tables to 1e-4.
- The ~17 GB cache lived in the session scratchpad; rebuild it with the tool.

**Paired test comparison, Mamba minus ViT, same 70 patients, shipped points**
(95% bootstrap CI, Wilcoxon signed-rank p):

| metric | ViT | Mamba | diff [95% CI] | p |
|---|---|---|---|---|
| mean Dice | 0.8684 | 0.8669 | -0.0015 [-0.0108, +0.0052] | — |
| Dice TC | 0.8782 | 0.8669 | -0.0112 [-0.0372, +0.0051] | 0.154 |
| Dice WT | 0.9109 | 0.9148 | +0.0038 [+0.0011, +0.0067] | 0.006 |
| Dice ET | 0.8161 | 0.8191 | +0.0029 [-0.0034, +0.0096] | 0.248 |
| mean HD95 (mm) | 18.18 | 17.03 | -1.15 [-7.43, +4.71] | — |
| HD95 TC | 3.94 | 10.11 | +6.17 [-0.12, +17.62] | 0.975 |
| HD95 WT | 4.84 | 5.82 | +0.98 [-0.25, +3.14] | 0.571 |
| HD95 ET | 45.76 | 35.16 | -10.60 [-27.14, +0.85] | 0.101 |
| HD95 ET, clean in both (n=60) | 3.51 | 3.62 | +0.10 [-1.65, +2.59] | 0.279 |

- **Verdict: a tie on test.** Only WT Dice is nominally significant (+0.004,
  p=0.006), and it does not survive Bonferroni over the 9 per-channel tests
  (0.0056).
- The test gaps each come down to one or two patients:
  - **TC Dice -0.011 and TC HD95 +6.2 mm are one patient, `BraTS19_TCIA13_650_1`.**
    Mamba segments its WT (Dice 0.818) but labels none of the 9,759-voxel GT core
    as TC (max TC prob inside GT < 0.0005). The ViT gets TC Dice 0.783, and
    0.783/70 ≈ 0.011. This tumour has almost no enhancement (125 ET voxels).
  - **ET HD95 -10.6 mm is two patients the ViT hallucinates and Mamba does not:**
    `BraTS19_TCIA09_254_1` (592 vox) and `BraTS20_Training_294` (376 vox), both
    just above min_total 352. 2 x 374 / 70 = 10.7.
- Failures both encoders share:
  - **4 ET-negative patients get thousands of predicted ET voxels from both**
    (ViT/Mamba): `BraTS19_2013_29_1` 577/858, `BraTS19_TCIA09_402_1` 5512/5764,
    `BraTS20_Training_312` 4328/5242, `BraTS20_Training_263` 2480/2541. When two
    different encoders agree on blobs this large, check the labels before
    calling them model errors.
  - `BraTS19_TCIA10_632_1` (GT ET 109 vox): both models have prob 1.0 inside GT,
    and min_total=352 zeroes it. `BraTS19_TCIA13_650_1`'s ET (125 vox) goes the
    same way for the ViT (max prob 0.993). **3 of the 4 ET misses are caused by
    the cleanup.**

**ET operating point chosen on VALIDATION.** The rule was fixed in the script
before test was read; Session 5 took its knee from a test grid.

| run | val knee | test at shipped | test at val knee |
|---|---|---|---|
| ViT | 0.20 (val 3h+5m -> 1h+5m) | 6h+2m, ET HD95 45.76, mean Dice 0.8684 | 5h+2m, ET HD95 40.29, mean Dice 0.8687 |
| Mamba | 0.05 = shipped (val flat at 2h+6m for every threshold) | 4h+2m, 35.16, 0.8669 | unchanged |

- **Val cannot calibrate hallucination suppression.** It has 5/105 ET-negative
  patients, against test's 8/70. For both runs val's lowest-HD95 point is NO
  cleanup, because val's 5-6 misses are small real ETs that min_total zeroes. On
  test that point is worse (8 hallucinated). The cleanup's net effect flips
  between the two splits.
- At the val knee the paired result does not change: mean Dice -0.0018
  [-0.0111, +0.0049], ET HD95 -5.1 mm [-16.8, +1.6], p=0.14.
- The test grid is shown for transparency only: both models bottom out at
  3h+2m (ViT from threshold 0.40, Mamba from 0.45).
- Recommendation: headline the shipped points, which is the protocol both runs
  were produced with, and give the ViT val knee as a sensitivity row. Do not
  pick anything from the test grid.

**Modality ablation over the whole test split** (n=70, empty-GT skipped, 95%
CI). The in-train XAI covers the same 70 patients but scores both-empty as 1.0;
its log tables are reproduced exactly.

| removed | ViT TC / WT / ET | Mamba TC / WT / ET | Mamba - ViT (p) |
|---|---|---|---|
| FLAIR | -0.635 / -0.748 / -0.623 | -0.192 / -0.339 / -0.122 | +0.44 / +0.41 / +0.50 (all <0.001) |
| T1ce | -0.556 / -0.084 / -0.825 | -0.441 / -0.054 / -0.827 | +0.12 (<0.001) / +0.03 (0.37) / -0.00 (0.10) |
| T1 | -0.102 / -0.045 / -0.072 | -0.082 / -0.048 / -0.083 | n.s. |
| T2 | -0.080 / -0.036 / -0.060 | -0.020 / -0.012 / -0.015 | +0.06 / +0.02 / +0.05 (all <=0.003) |

- G6 passes for both. ET collapses without T1ce (-0.83 in both), and removing
  FLAIR causes each model's largest WT drop.
- **This is the largest and clearest difference between the encoders, larger
  than anything in Dice.** The ViT uses FLAIR for every region: without it, ET
  drops -0.62, although ET is defined on T1ce. Mamba's reliance is
  region-specific (ET without FLAIR -0.12), and it degrades far less when FLAIR
  or T2 is missing.
- Mamba's relatively heavier T1ce reliance for TC is consistent with its TC miss
  on the barely-enhancing `TCIA13_650_1`. That is a hypothesis; nothing here tests it.
- Caveat: a zeroed sequence is an out-of-distribution input, and neither model
  was trained with modality dropout. This measures reliance, not importance.

**Correction to Session 6 and to compare_runs' XAI rows.** Every file in
`logs/v2-run3/xai/*.json` was rewritten on 2026-08-18 between 10:28 and 10:40 by
a standalone `xai.py` pass, after the run finished at 08:28, and standalone
xai.py does not load the tuned thresholds. So v2-run3's `modality.json` is NOT the
in-train result. Session 6 quoted it as -FLAIR WT -0.761 and -T1ce ET -0.645; the
in-train log says -0.748 and -0.673.

**Other XAI** (3 samples; use as figures, not as claims).
- **Randomisation check:** SSIM stays flat for both models (Mamba 0.959-0.963,
  ViT 0.966). This is Session 4's G4/G5 artefact: the CAM support is pinned to the
  GT ROI, and SSIM is dominated by background. It says nothing about either
  encoder; still void.
- **CAM deletion AUC:** Mamba 0.054-0.058 vs random 0.754; ViT 0.040-0.041 vs
  0.703.
- **Rollout:** Mamba's hidden-attention rollout scores 0.708 vs random 0.754,
  lower in 8 of 9 cases but only just (e.g. 0.905 vs 0.927). The ViT's attention
  rollout scores 0.455 vs 0.703. Mamba's rollout is barely better than random;
  do not lean on it.
- **MC-dropout,** after referring the 2% most uncertain voxels: TC 0.901->0.968,
  WT 0.878->0.922, ET 0.530->0.624. ViT: 0.883->0.959, 0.857->0.913, 0.506->0.591.

**Open items.**
1. Look at the 4 shared ET "hallucinations" and at `BraTS19_TCIA13_650_1`
   (image, GT, both predictions) before writing about any of them.
2. min_total=352 caused 3 of the 4 ET misses (real ETs of 109-125 voxels).
   Retuning it needs a split with more ET-negative patients than val's 5
   (e.g. CV folds); it cannot be tuned honestly on val.
3. There is still no matched-budget (30-epoch) ViT. The val curves favour
   Mamba at every epoch, but the two schedules differ.
4. `xai.py` still ignores `eval/infer_config.json` (Session 6 open item). It
   has now demonstrably overwritten one run's in-train XAI with untuned numbers.
5. Nothing is committed: Sessions 5-7 are all uncommitted on `native-1mm-run`.

---

## 2026-09-12 — Session 6: swappable encoder, SegMamba vs ViT; 30-epoch Mamba run LAUNCHED

User decisions: replace the ViT encoder with a Vision Mamba encoder **as a
swappable block** (keep the ViT for comparison), accurate to how Mamba is used
in brain-tumour papers -> **SegMamba** (Xing et al., MICCAI 2024, BraTS 2023);
**30 epochs**; **no matched ViT rerun** (compare against `v2-run3`, 50 ep);
Mamba kernels via a **separate `mamba` conda env**, not a source build.

**Why a new env, and why torch 2.6 (not 2.10).** Every mamba_ssm/causal_conv1d
wheel after v2.2.4/v1.5.0.post8 was built on Ubuntu 22.04 and needs glibc >=
2.32; this box has 2.31 (import fails "GLIBC_2.32 not found"). v2.2.4 was built
on 20.04 (objdump: needs GLIBC_2.14) and tops out at torch 2.6. `mamba` =
pytorch2's exact pins with torch 2.6.0+cu124, torchvision 0.21, triton 3.2,
sympy 1.13.1, torch 2.6's nvidia-*-12.4 pins, + mamba_ssm 2.2.4,
causal_conv1d 1.5.0.post8, einops, ninja, transformers 4.46.3. Recreate with
`tools/setup_mamba_env.sh`. (Gotchas hit: conda's notices cache is corrupt ->
`CONDA_NUMBER_CHANNEL_NOTICES=0`; pytorch2 itself violates medpy's numpy pin, so
the pins are installed `--no-deps`.)

**Code.**
- `blocks/VisionMamba.py` (new): SegMamba MambaEncoder — stem 7^3/s2, GSC,
  2 x MambaLayer per stage at 48/96/192/384, InstanceNorm + MlpChannel outputs,
  tri-orientated Mamba (fwd / bwd / inter-slice scans, shared in/out proj).
  Deviations: dropout (cfg.mamba.dropout=0.2, for MC-dropout parity — SegMamba
  has none); upstream `mamba_inner_fn` per branch instead of the fork's
  no-out-proj variant (same function, out_proj is bias-free and per token).
  Also: exact chunked reference scan (CPU/tests) and Ali et al. 2024 hidden
  attention.
- `models/unetr.py`: `encoder="vit"|"mamba"`. Mamba stages feed the SAME
  decoder via SegMamba's own skip blocks (UnetrBasicBlock) at 1/2, 1/4, 1/8 and
  a 768-ch 1/16 block into `decoder12_upsampler`. ViT path unchanged:
  v2-run3's checkpoint loads strict, same key order.
- `config.py`: epoch 50 -> **30**, warmup 5 -> 3, `cfg.unetr.encoder="mamba"`,
  `cfg.mamba`, `cfg.xai.components` = all five (to match v2-run3's xai/).
- `utils/engine.py`: build_model refuses a Mamba run without kernels; A_log/D
  in a weight_decay=0 AdamW group (ViT keeps one group); `apply_run_model_config`
  so evaluate.py/xai.py rebuild the encoder a run was trained with.
- `utils/checkpoint.py`: last.pth records the encoder; resuming across encoders
  refuses. `train.py --encoder`.
- XAI: `rollout` = hidden-attention rollout of the deepest stage (same 8x8x6
  grid as the ViT patches) for Mamba; training overlay likewise;
  randomisation cascade ends at `mamba_encoder`; `_reinitialize` resets
  children before parents (neutral for the ViT: no composite resets there).
- Tools: `smoke_test.py` (real data, every stage, time projection),
  `crosscheck_segmamba.py`, `compare_runs.py`, `setup_mamba_env.sh`.

**Verification.**
- `pytest`: 243 passed in `mamba`, 239 passed + 4 skipped (kernel-only) in
  pytorch2. The tests caught a real bug: an `nn.ModuleDict` key "forward"
  collides with Module.forward, so the branches are named fwd/bwd/slice.
- `crosscheck_segmamba.py` vs official SegMamba @cff3597: strict weight load,
  15,718,080 params both. Each of the 8 MambaLayers matches to ~5e-8 given
  identical input, and the four stages end to end to <=3.7e-6 (TF32 off; with
  TF32 on the gap is ~1e-3, i.e. conv rounding, not structure).
- `smoke_test.py` on real data: all stages and all 5 XAI components run.
  4.3 s/step, 10.4 s/val volume, peak 22.6 GB. ~56 min/epoch -> ~28 h + ~1 h.
  Model 102.14M params (encoder 15.72M) vs ViT 156.27M (encoder 79.05M).

**Caveats for the paper.** 30 vs 50 epochs, each with its own cosine schedule:
compare curves and the epoch-30 row (`compare_runs.py`), not only final test.
torch 2.11 (ViT) vs 2.6 (Mamba). Hidden-attention rollout covers the deepest
stage (2 layers); the ViT rollout covers 12 blocks. n=70 test split.

**Launch.** The user ran `python train.py --name v3-mamba-30ep` in the `mamba`
env at 2026-09-12 01:05 IST, in the foreground of terminal pts/4 without nohup,
so closing that terminal or SSH session kills it. At launch the log showed
"Encoder: mamba | Trainable parameters: 102.14M (encoder 15.72M)", the snapshot
showed epoch 30 / warmup 3 / the SegMamba config, and the GPU used ~26 GB.

**Open items left from this session.**
- Standalone `xai.py` does NOT load a run's tuned thresholds. It uses
  `cfg.infer.thresholds` = (0.5, 0.5, 0.5) from config.py. The XAI inside
  `train.py` does use the tuned ones (`tune_and_save` mutates cfg before the
  test and XAI steps). So a standalone XAI rerun can disagree with the
  in-train XAI on modality Dice, the uncertainty mask and the deletion curves.
  The fix (read `eval/infer_config.json` in xai.py) was offered but not made.
- The ET threshold-knee ablation was never run on v2-run3, and has not been
  run on the Mamba run either (the sweep argmax ships for both).
- Nothing from Session 6 is committed. The branch `native-1mm-run` also still
  has uncommitted Session 5 work.

---

## Session-6 handoff (DONE in Session 7): read the Mamba run (`logs/v3-mamba-30ep`)

Launched 2026-09-12 01:05 IST. Projection from the smoke test: ~56 min/epoch,
so 30 epochs is ~28 h, plus ~1 h for tuning, the test pass and XAI. It should
finish around 2026-09-13 06:00 IST. This is a projection; check `run_eta`
in the log.

### 0. Where things stand
```bash
cd "/DATA/Abul Hasan/Glioma Revision"
conda activate mamba                          # REQUIRED: pytorch2 has no Mamba kernels; build_model refuses
ps aux | grep "train.py --name v3-mamba-30ep" | grep -v grep   # still running?
tail -30 logs/v3-mamba-30ep/log.txt           # per-epoch table + run_eta
wc -l logs/v3-mamba-30ep/metrics.csv          # epochs done = lines - 1
grep -n "XAI\] complete\|Run finished" logs/v3-mamba-30ep/log.txt   # finished?
```
- Both lines present: the run is done, go to step 2.
- Not running and not finished: it died, go to step 1.
- Still running: check health against step 3's first rows and wait.

### 1. If it died: resume (it continues from the last completed epoch)
```bash
conda activate mamba
nohup python -u train.py --name v3-mamba-30ep --auto-resume > logs/v3-mamba-30ep.out 2>&1 &
tail -f logs/v3-mamba-30ep.out
```
Use `--auto-resume`, not `--resume`. If no epoch finished (no
`checkpoints/last.pth`), `--resume` exits with an error, and a plain relaunch
without a flag creates a new folder `logs/v3-mamba-30ep_1`. `--auto-resume`
resumes from last.pth if it exists, and otherwise starts fresh in the same folder.
Resuming into the other encoder is refused (last.pth records the encoder).
Memory history: v2-run2 died at epoch 33 from GPU memory (`RUN.md` 2b). The
Mamba run peaked at 22.6 GB in the smoke test.

### 2. When it finishes: what exists, then read in this order
`train.py` runs everything in one go: 30 epochs, then threshold tuning on val,
then the test pass (TTA + tuned thresholds + post-processing), then all 5 XAI
components. The order matters: XAI runs after tuning, so it uses the tuned
thresholds. **No separate evaluate.py or xai.py step is needed.**
1. Compare with the ViT run:
   ```bash
   python tools/compare_runs.py v2-run3 v3-mamba-30ep
   ```
   It writes `logs/compare_v2-run3_vs_v3-mamba-30ep/`:
   - `curves.png`: per-epoch validation curves on shared axes
   - `summary.md`: cost, validation at epoch 30 and at each run's best, test
     metrics with the HD95 breakdown, XAI
   - `summary.csv`: the same table
2. `testing/test_metrics.csv`: Dice/HD95/sensitivity plus the `n_halluc_*`,
   `n_miss_*` and `hd95_*_clean` columns.
3. `eval/threshold_sweep.json`: the tuned thresholds. They are the sweep
   argmax, which ignores hallucinations (Session 5 lesson). If ET HD95 is
   sentinel-dominated, run the ET knee ablation. The template is
   `logs/run1-new-version/eval/et_operating_point_ablation.py` (`min_total`=352
   at 1mm). Rerun with `python evaluate.py --run v3-mamba-30ep --tune-thresholds --tag final`.
4. `xai/`:
   - modality: gate G6, ET collapses without T1ce and WT without FLAIR
   - uncertainty: gate G7
   - faithful: deletion AUCs and the randomisation check
   - rollout and cam: figures
   Standalone reruns use 0.5 thresholds (see Session 6 open items), so prefer
   the in-train outputs.

### 3. What to expect / how to read it
| Look at | Expected / if | Then |
|---|---|---|
| Log header | "Encoder: mamba", 102.14M (encoder 15.72M) | if it says vit/156.27M, the wrong config ran; stop |
| min/epoch | ~56 (ViT: 41.8) | expected; the Mamba run is slower per epoch here. Report it, don't hide it |
| **val mean Dice at ep 30** vs v2-run3's **0.841 at ep 30** | at or above | the main result: Mamba matches or beats ViT at equal epochs with 35% fewer params (102M vs 156M) |
| same | clearly below | the ViT wins at 30 epochs; the curves show whether Mamba was still climbing |
| final test mean Dice vs **0.868** (ViT, 50 ep) | within ~0.03 | inconclusive: n=70 noise and 30 vs 50 epochs. Rely on the curves and the ep-30 row |
| test ET sensitivity vs **0.760** | up | better ET recall, the problem the last runs were chasing |
| `n_halluc_et` vs **6** | higher | ET HD95 will be sentinel-dominated again; compare `hd95_et_clean` (ViT **3.51**) and consider the ET knee |
| xai/modality ET drop without T1ce vs **-0.645**; WT without FLAIR vs **-0.761** | similar collapse | same modality reliance, G6 passes. If ET does NOT collapse, investigate |
| faithful randomisation check | maps change once weights are randomised | if they don't change, the explanations don't depend on the model; flag it in the paper |
| rollout | — | the Mamba rollout covers the deepest stage (2 layers), the ViT rollout 12 blocks. Not like-for-like; say so |

### 4. MAMBA RESULTS (filled in Session 7)
```
v3-mamba-30ep | 30 ep | best ep 23 | 44.8 min/epoch | 22.4 h training (23.4 h incl. tune/test/XAI)
Val  Dice  TC/WT/ET/mean : 0.881 / 0.911 / 0.810 / 0.868   (ep 23; 0.864 at ep 30; ViT at ep 30: 0.841 mean)
Test Dice  TC/WT/ET/mean : 0.867 / 0.915 / 0.819 / 0.867   (ViT: 0.878 / 0.911 / 0.816 / 0.868)
Test HD95  TC/WT/ET      : 10.12 / 5.82 / 35.16   (clean ET: 3.62 ; n_halluc/n_miss ET: 4/2 ; TC: 0/1)
Test Sens  TC/WT/ET      : 0.864 / 0.929 / 0.764   (ViT ET: 0.760)
Tuned thresholds (sweep) : 0.15 / 0.10 / 0.05   | val-selected ET knee: 0.05 (= shipped; val mismatch flat)
XAI (in-train, n=70, both-empty scored 1.0): ET drop w/o T1ce = -0.661 ; WT drop w/o FLAIR = -0.339 ;
     randomisation check: SSIM flat 0.959-0.963 (void, same artefact as the ViT's 0.966)
Verdict: ties the 50-epoch ViT on test (paired mean Dice -0.0015 [-0.0108, +0.0052], n=70)
     and is above it on val at every epoch, with 35% fewer params and 36% less training time.
     The TC gap is 1 patient and the ET HD95 gap 2 patients. The largest difference is
     modality reliance: the ViT needs FLAIR for every region, Mamba does not. See Session 7.
```

---

## Session-5 handoff (HISTORICAL — native-1mm run, done as `logs/v2-run3`)

The native-1mm training run was **built, tested, and ready** but was **PENDING
LAUNCH** at the end of Session 5. It later ran as `logs/v2-run3`, and its results
are in section 4 below. The steps here are kept for the record.

### 0. First, find out where things stand
```bash
cd "/DATA/Abul Hasan/Glioma Revision"
git branch --show-current          # the 1mm code lives on `native-1mm-run`, not main
git log --oneline -3               # was it committed?
ls logs/                           # did the run get launched? look for run2-native-1mm/
ps aux | grep train.py             # is it running right now?
```
- If `logs/run2-native-1mm/` exists with a growing `metrics.csv` → the run is
  underway or done; skip to §2.
- If not → launch it, §1.

### 1. Launch (if not already running)
```bash
git checkout native-1mm-run        # IMPORTANT: the 1mm pipeline is only on this branch
conda activate pytorch2
pytest -q                          # expect 185 passed
nohup python -u train.py --name run2-native-1mm > logs/run2-native-1mm.out 2>&1 &
tail -f logs/run2-native-1mm.out   # watch the per-epoch table + run_eta
```
- ~30–40h, 50 epochs. One command does train → tune thresholds on val → test →
  XAI(modality, uncertainty).
- Outputs land in `logs/run2-native-1mm/`:
  `checkpoints/best_metric_model.pth`, `metrics.csv` (per-epoch),
  `testing/test_metrics.csv` (headline + n_halluc/n_miss/hd95_clean breakdown),
  `eval/{threshold_sweep,infer_config}.json`, `xai/{modality,uncertainty}`,
  `plots/`, `log.txt`, `config_snapshot.json`.

### 2. When it finishes — read outputs in this order
1. `testing/test_metrics.csv` — Dice/HD95 per channel + the `n_halluc_*`,
   `n_miss_*`, `hd95_*_clean` columns (the honest HD95 breakdown).
2. `eval/threshold_sweep.json` — the tuned thresholds. **Do NOT ship the ET
   argmax blindly** — the val sweep is blind to hallucinations (Session-5
   lesson). Re-run the ET knee ablation to pick the operating point:
   `logs/run1-new-version/eval/et_operating_point_ablation.py` is the template
   (point it at the new checkpoint; note `min_total` is now 352 at 1mm).
3. `xai/modality.json` — ET must collapse without T1ce, WT without FLAIR (gate G6).
4. `xai/uncertainty.json` — error-retention curve (gate G7).

### 3. Decision gates for the 1mm results
| Look at | If | Then |
|---|---|---|
| **mean Dice** vs run1's ~0.83 | much lower | **EXPECTED** — 1mm is the harder, honest problem. Compare 1mm-to-1mm ONLY, never to the resized runs. Not a regression. |
| **ET sensitivity** vs run1's 0.72 | **up** | the Focal-Tversky-on-aux + native resolution recovered recall — the run did its job. |
| ET sensitivity | still ≤ ~0.72 | recall is NOT fixable this way. Suspect the flat-ViT ceiling / label quality → plan §0.7.2 #1 (already at 1mm), #5 (block swap), or #2 (5-fold CV). |
| `n_halluc_et` | high | raise `min_total` (now 352) OR ship a higher ET-threshold knee (§2.2). |
| `hd95_et_clean` | ~3–4mm | boundaries fine, as on run1 — do NOT spend effort on boundary work. |
| `xai/modality` | ET does NOT collapse without T1ce | model is right for the wrong reasons — investigate, don't bury (gate G6). |

### 4. RESULTS — APPEND HERE (fill in after the run)
```
v2-run3 (= the native-1mm run) | 50 ep | best ep 41 | 41.8 min/epoch | 35.1 h total
Val  Dice  TC/WT/ET/mean : 0.861 / 0.900 / 0.790 / 0.850   (ep 41; 0.841 at ep 30)
Test Dice  TC/WT/ET/mean : 0.878 / 0.911 / 0.816 / 0.868
Test HD95  TC/WT/ET      : 3.94 / 4.84 / 45.76   (clean ET: 3.51 ; n_halluc/n_miss ET: 6/2)
Test Sens  TC/WT/ET      : 0.879 / 0.923 / 0.760   (run1: 0.72)
Tuned thresholds (sweep) : 0.15 / 0.05 / 0.05   | shipped ET knee: NOT RUN (the sweep argmax shipped)
XAI: ET drop w/o T1ce = -0.645 ; WT drop w/o FLAIR = -0.761 ; uncertainty, Dice after
     referring 2% most-uncertain voxels (3 samples): TC 0.883->0.959, WT 0.857->0.913, ET 0.506->0.591
Verdict: recall UP (ET sens 0.760 vs 0.72); boundaries fine (clean ET HD95 3.51mm);
     hallucinations UP (6 vs 3-5): ET HD95 = (8 x 374 + 60 x 3.51) / 70 = 45.7mm, all
     sentinel. The ET knee ablation (section 2 step 2) was never run on this checkpoint.
```
(Filled in Session 6 from `logs/v2-run3/`; the run's xai/ also holds cam/rollout/faithful.)

### Decisions locked this session (context for whoever reads this)
- **One more training run only** — it is this native-1mm run. `v1-run3`'s
  checkpoint is gone, so the recall regression can never be isolated in a
  controlled comparison; reason from the Phase-0/ET ablations, not A/B.
- **Kept 128×128×96** (not 128³): empty top/bottom slices are removed by the
  adaptive `CropForegroundd`, so patch depth does not change noise exposure —
  96 is cheaper, memory-safe, and keeps `img_shape` unchanged (no transformer
  positional-embedding change). **Dropped `RandBiasFieldd`.** CLAHE removed
  (was already dead code).
- **Aux heads: Dice + Focal-Tversky** (recall pressure); **Hausdorff stays
  main-head only** (keeps the 2.14× speedup). ET boundaries are already good.
- **Headline = Dice/HD95**; **XAI = modality + uncertainty only** (CAM/faithful
  cut — the CAM support is the GT mask, a tautology, not a finding).
- **The residual ET problem is recall/detection + presence-absence, NOT
  boundaries** (ET clean HD95 = 3.10mm, best of the three). Mismatch floor on
  run1 was 4 (≈3 hallucinated + 1 missed).
- **Ship the ET-threshold KNEE, not the sweep argmax** (the sweep maximises
  clean Dice, which is blind to the hallucinations that dominate ET HD95).
- **Caveats that survive any result:** test split is n=70 (±0.03 ET is noise);
  5-fold CV (plan §0.7.2 #2) is the first thing a reviewer will ask for.

---

## 2026-07-29 — Session 5: ran the pending eval; removed CLAHE; ET operating point mis-tuned

Ran `evaluate.py --run run1-new-version --tag final --tune-thresholds` (the command
Session 4 left pending) and removed CLAHE from the preprocessing pipeline. The
eval settles the one open question from Session 4 (gate G2) and surfaces a new,
cheap inference fix.

**CLAHE removed — behavior-neutral.** `ApplyCLAHEAndZscored` only ever called
`zscore_normalize`; `apply_clahe_to_volume` was already dead code (defined, never
invoked). Deleted the dead function + its `skimage` import + its tests, renamed
the transform `ApplyCLAHEAndZscored -> ZScoreNormalized`, added an honest
docstring. Runtime preprocessing is unchanged (per-channel z-score over the brain
mask), so the run1 checkpoint and this eval stay valid. `pytest` green (189, then
33 on the two touched files).

**THE FINDING — the val threshold sweep drove ET to a WORSE operating point.**
With the both-empty bug fixed (empty-GT patients skipped) and the grid extended to
0.05, the sweep picked **TC=0.10, WT=0.10, ET=0.05** (was 0.3/0.3/0.5 published).
The ET sweep is now monotone-decreasing and pinned at the 0.05 floor. But ET Dice
with `ignore_empty` is blind to presence/absence, so it happily minimises the
threshold — and a lower threshold makes MORE ET-negative patients emit stray FP
voxels that clear `min_total=100`. Net effect on the test pass:

| ET operating point | ET Dice | ET HD95 | mismatches | ET HD95 clean |
|---|---|---|---|---|
| published (thr 0.5, pp on) | 0.753 | 24.50 | 4 | 3.65 |
| **final (thr 0.05, pp on)** | **0.758** | **34.71** | **5 halluc + 1 miss** | **3.10** |

Lowering ET 0.5->0.05 bought **+0.005 Dice and cost +10.2mm HD95**. The buggy
sweep's ET=0.5 was, by luck, the better HD95 point. Verified exact:
`6*374 + 60*3.0998 = 2430.0 / 70 = 34.714`, matching to 3 d.p.

**Full `final` test pass (thr 0.1/0.1/0.05, TTA, pp min_comp=(0,0,50) min_total=(0,0,100)):**

| | TC | WT | ET | Mean |
|---|---|---|---|---|
| Dice | 0.852 | 0.892 | **0.758** | 0.834 |
| HD95 (mm) | 4.59 | 4.62 | **34.71** | 14.64 |
| HD95 clean | 4.59 | 4.62 | **3.10** | — |
| Sens | 0.874 | 0.909 | **0.723** | 0.835 |
| IoU | 0.758 | 0.809 | 0.548 | 0.705 |

TC/WT have **zero** empty-mismatch (their HD95 == HD95-clean). ET clean boundary
**3.10mm is the best of the three** — reconfirms Session 4: ET geometry is fine,
the residual ET defect is presence/absence + recall (sens 0.723), NOT boundaries.

**Session-4 predictions, scored:**
1. "~3 halluc + 1 miss" -> **5 halluc + 1 miss.** Direction right (overwhelmingly
   hallucinated), count off. **Gate G2 resolved: hallucinations dominate -> RAISE
   `min_total_voxels`.** The "maybe it's misses, lower it" branch is closed.
2. "ET threshold moves off 0.5" -> **confirmed, to 0.05**; the +0.017 spike at
   exactly 0.50 vanished. The both-empty bug alone picked the published 0.5.
3. "TC/WT slide to the 0.05 floor" -> **partially wrong.** Both landed at **0.10**
   with a shallow interior peak (they DECREASE from 0.10 down to 0.05). TC/WT are
   better calibrated than "pinned at floor" implied; only ET wants the floor.

**Two cheap, no-retrain levers this exposes (NOT training-side):**
- **Raise ET `min_total_voxels`** (100 -> ~150-200 in 1.875mm-voxel space) to kill
  more of the 5 hallucinations. One `evaluate.py` pass, re-runnable.
- **Don't blindly ship the sweep's ET threshold.** The sweep optimises clean Dice,
  which is blind to the hallucination count that dominates ET HD95. A sensible ET
  point trades a hair of clean Dice for far fewer 374.0 sentinels — the sweep
  cannot see that trade. (Its own docstring already warns of exactly this.)

Note both levers get **re-derived in 1mm space anyway** (voxel size changes, so
`min_total` must be rescaled), so they matter mainly as postproc settings to carry
into the 1mm run, not as work to finalise on the soon-superseded 1.875mm protocol.

**ET operating-point ablation (no retrain, `eval/et_operating_point_ablation.{txt,py}`).**
Cached the test-set predictions once (logits are identical across settings) and
swept ET threshold x (min_component, min_total), TC/WT fixed at 0.10. Script
validated: it reproduces `run_test` exactly on two anchors — the `final` point
(0.05/50/100 -> ET 0.7575/34.71, 5h+1m) and the published point (0.50/50/100 ->
ET 0.7529/24.50, 3h+1m).

- **Winner: ET threshold 0.20, min_comp 50, min_total 100.** mean Dice **0.8341**,
  mean HD95 **11.11** (down from `final`'s 14.64, and below published 11.24); ET
  0.7582 / 24.13mm / **3 halluc + 1 miss**. Highest ET Dice of any low-HD95 row
  AND the simplest change — just don't ship the sweep's 0.05.
- **Threshold is a cleaner lever than `min_total`.** Raising thr 0.05->0.20 drops
  hallucinations 5->3 with NO new misses (real ET has high-confidence voxels well
  above 0.20). Raising `min_total` instead (0.05/300) also cuts hallucinations but
  zeroes 2 real small-ET patients (miss 1->2, ET Dice 0.7575->0.7447). So prefer
  the threshold knee; keep `min_total` modest.
- **0.20 is the knee:** below it (0.10, 0.05) hallucinations jump to 5; at/above it
  they hold at 3 while ET Dice falls monotonically to 0.7529 at 0.50.
- **The mismatch floor is 4** on this checkpoint (3h+1m or 2h+2m — trading one for
  the other, never fewer). The 1 miss is a genuine detection failure (ET stays
  empty even at thr 0.05); the ~3 hallucinations are substantial FP blobs. That
  residual == the recall regression, and it is exactly what the 1mm run +
  Focal-Tversky-on-aux targets. No inference knob removes it.

**Lesson carried into the 1mm run:** after tuning, ship the ET-threshold KNEE (the
lowest threshold that holds the hallucination floor), not the sweep's raw argmax;
keep `min_total` modest (rescaled for 1mm voxels).

### Native-1mm run — BUILT + smoke-tested, PENDING LAUNCH (branch `native-1mm-run`)

Implemented the one-more-run change-list. NOT launched — user reviews the diff
first. Design forks the user picked: **keep 128x128x96** patches (not 128^3 — 96
was cheaper/proven and, crucially, the empty top/bottom slices are removed by the
adaptive `CropForegroundd`, not by patch depth, so 128 vs 96 does not change
empty-slice exposure); **drop `RandBiasFieldd`** entirely.

Because `img_shape` STAYS (128,128,96), the transformer positional-embedding
landmine (journal §2 #2) does NOT apply — no model-geometry change at all.

Changes:
- **transforms.py** — native 1mm. Train: `CropForegroundd` (adaptive brain bbox
  on raw intensities) -> `ZScoreNormalized` (whole-brain) -> `SpatialPadd` to the
  patch size -> `RandCropByPosNegLabeld(pos=2,neg=1,num_samples=2)` on the
  single-channel integer label -> multi-channel split -> flips/affine/intensity.
  `Resized`/`CropRawDepthd` gone from the pipeline (classes kept for the tool +
  tests). Val/test: `CropForegroundd` + whole foreground-cropped brain through
  sliding window (no crop imposed).
- **config.py** — `voxel_spacing=(1,1,1)` hard-coded (honest by construction, no
  derivation); `cfg.crop` = {fg_threshold, pos, neg, num_samples}; `min_component`/
  `min_total` rescaled 3.52x to (0,0,176)/(0,0,352) to hold the same PHYSICAL
  cleanup at 1mm; `xai.components=["modality","uncertainty"]`; thresholds default
  (0.5,0.5,0.5) (re-tuned at end of run — ship the KNEE).
- **losses.py** — aux heads now Dice + Focal-Tversky via `CombinedLoss.aux_loss`
  (Hausdorff still off aux, so the 2.14x speedup is kept). `combine_main_and_aux`
  switched from `.dice` to `.aux_loss`; plain-callable fallback preserved.
- **dataloader.py** — `list_data_collate` on the train loader (RandCrop returns a
  list of num_samples); `build_dataloaders` shape-print handles the list.
- **tests** — updated config (crop/spacing), losses (2 new aux tests), transforms
  (new native-1mm integration test); removed the CLAHE/fixed-depth tests.

Verification: `pytest` **185 passed**; real-data smoke test (2 train steps + 2 val
volumes) — collation OK, forward+backward finite, **peak GPU 21.6 GB / 48** (no
OOM), val volumes foreground-cropped to ~(130,171,141) at native 1mm, sliding
window + metrics run. The smoke test caught one real bug pytest could not reach
(`train_ds[0]` is now a list) — fixed.

Epochs to set at launch: **50** (journal measured +0.009 mean-Dice for the 2nd 50).
`train.py` then auto-runs tune-thresholds -> test -> XAI(modality,uncertainty).
Cost est. ~30-40h (128x128x96 is ~1.78x cheaper attention than 128^3; sliding-
window val over full 1mm volumes is the real per-epoch cost).

**Files:** `utils/dataloader.py` (dropped `apply_clahe_to_volume` + `skimage`
import, renamed transform, fixed stale comment), `utils/transforms.py` (import +
2 call sites), `tests/test_dataloader.py` + `tests/test_transforms.py` (dropped
CLAHE tests, renamed). New eval artifacts in `logs/run1-new-version/eval/`:
`test_metrics_final.csv`, `threshold_sweep_final.json`, `infer_config_final.json`.

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

## Session-4 handoff (HISTORICAL — superseded by the "NEXT SESSION" section at the top)

### 1. A command was left running. Read its output first. — DONE (Session 5)

```bash
pytest -q                                                                # green (189)
python evaluate.py --run run1-new-version --tag final --tune-thresholds   # ran, ~1h
```

Outputs written to `logs/run1-new-version/eval/`: `test_metrics_final.csv`,
`threshold_sweep_final.json`, `infer_config_final.json`. **See the Session-5
entry above for the analysis.** Table below records what each row actually fired:

| Look at | Predicted | ACTUAL (Session 5) |
|---|---|---|
| `n_halluc_et` vs `n_miss_et` | ~3 halluc + ~1 miss | **5 halluc + 1 miss** — hallucinations dominate. **RAISE `min_total_voxels`.** Gate G2's raise-direction CONFIRMED |
| `hd95_et_clean` | still ~3.6mm | **3.10mm** — even better; ET boundaries fine, no boundary work in the 1mm run |
| ET best threshold | below 0.5 | **0.05** (floor); the 0.50 spike vanished. But 0.05 is a WORSE HD95 point than the old 0.5 (more hallucinations) — the sweep can't see that |
| TC/WT best threshold | pinned at 0.05 | **0.10** — shallow interior peak, not pinned. Better calibrated than expected |

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

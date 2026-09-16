"""Central configuration for the glioma segmentation pipeline.

Edit values here to change an experiment; train.py snapshots this file's
values into every run folder so past runs stay reproducible.
"""
from easydict import EasyDict

cfg = EasyDict()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
cfg.paths = EasyDict()
cfg.paths.root_dir = "/DATA/Abul Hasan/Glioma Revision/data/combined"
cfg.paths.logs_dir = "logs"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# 60 epochs. The 30-epoch Mamba run (logs/v3-mamba-30ep) ended with the LR
# already at its 1e-6 floor, and the 50-epoch ViT run (logs/v2-run3) was still
# improving at epoch 41 (val mean Dice 0.8413 at epoch 30, 0.8504 at its best),
# so the schedule, not the model, set the ceiling. A 60-epoch run is not
# comparable to those two at their final epoch: compare the per-epoch curves
# (metrics.csv) at matched epochs.
cfg.epoch = 60
cfg.learning_rate = 1e-4
cfg.weight_decay = 1e-4
cfg.patience = 50
cfg.grad_clip_norm = 1.0
cfg.val_interval = 1
cfg.val_amp = True
cfg.seed = 0

# Linear LR warmup for the first N epochs, then cosine over the rest. 50-epoch
# runs plateaued around epoch 40 with the LR already at its 1e-6 floor, i.e.
# the schedule ran out before the model did.
cfg.warmup_epochs = 5   # ~8% of a 60-epoch run (3 on the 30-epoch runs, 5 on the 50-epoch ones)

# Exponential moving average of the weights. Validation, checkpointing and
# testing all use the EMA copy; set to 0 to disable and train/eval the raw
# weights. 0.999 over ~500 steps/epoch is a ~2-epoch averaging horizon.
cfg.ema_decay = 0.999

# How many batches may die of CUDA OOM before the epoch gives up and the run
# fails. A skipped batch is a rounding error on ~500 train / ~180 val samples;
# a failed 45h run is not. But an unbounded skip count would quietly turn "this
# no longer fits on the card" into an epoch that trains on nothing and reports a
# loss anyway, so both are capped. Exceeding the cap re-raises, which is exactly
# the signal `train.py --auto-resume` needs to restart from last.pth.
cfg.train_oom_skip_limit = 5
cfg.val_oom_skip_limit = 5

# ---------------------------------------------------------------------------
# Checkpointing / crash recovery
# ---------------------------------------------------------------------------
cfg.checkpoint = EasyDict()

# Write logs/<run>/checkpoints/last.pth at the end of every epoch: full
# training state (weights + EMA + optimizer + scheduler + scaler + RNG +
# best-metric bookkeeping), which is what `--resume` needs to continue a run
# without restarting the LR schedule or AdamW's moments. ~1.9GB per write for
# this 156M-param model, overwritten in place, atomically. Separate from
# best_metric_model.pth, which stays weights-only and best-only.
cfg.checkpoint.save_last_every_epoch = True

# Force a cycle collection every N validation samples (0 disables).
#
# Measured: six validation samples leave ~0.55GB of CUDA tensors that plain
# refcounting cannot free — whole-brain (1,1,H,W,D) predictions and the float64
# (1,H,W,D) distance transforms HausdorffDTLoss brings back from scipy, all
# unreachable but sitting in reference cycles. CPython's cycle collector
# triggers on object COUNTS, not bytes, so a few dozen 100MB CUDA tensors are
# invisible to its heuristic and can sit uncollected indefinitely.
#
# gc.collect() at the epoch boundary alone would let a full ~180-sample
# validation pass accumulate before anything runs. Every 20 samples caps the
# backlog at ~20 samples' worth for ~9 collections per epoch — a couple of
# seconds against a ~41-minute epoch.
cfg.checkpoint.gc_every_n_val_steps = 20

# Fragmentation guard, applied by train.py before CUDA initialises (see
# utils/env_check.py:configure_cuda_allocator). Training allocates a fixed
# 128x128x96 batch; validation allocates a different whole-brain shape for
# every patient. Under the default allocator those two regimes carve the
# reserved pool into blocks neither can reuse — measured on this run: 23.9GB
# reserved against 4.0GB actually live, with 5-7GB stranded as non-releasable.
# expandable_segments lets one virtual segment grow and be re-carved instead,
# which is the difference between a ~21GB working set fitting comfortably in
# 48GB and dying at epoch 34. Set to "" to use the stock allocator.
cfg.checkpoint.cuda_alloc_conf = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Data / dataloaders
# ---------------------------------------------------------------------------
cfg.data = EasyDict()
cfg.data.val_frac = 0.15
cfg.data.test_frac = 0.10
cfg.data.cache_rate = 0.0
cfg.data.batch_size_train = 1
cfg.data.batch_size_val = 1
cfg.data.batch_size_test = 1
cfg.data.num_workers_train = 4
cfg.data.num_workers_val = 4
cfg.data.num_workers_test = 0

# DELIBERATE DATA LEAKAGE, for the assignment demonstration of how leakage
# inflates validation metrics. Set to None for any honest run (every paper run).
#   "patient": every validation patient is ALSO put in the training set
#   (utils/dataloader.py:BratsDataset._split_datalist). Per-epoch validation,
#   best-checkpoint selection and threshold tuning then all score patients the
#   model trained on. No test patient is trained on, so this run's val-vs-test
#   gap is the demonstration.
# So a leaked run can never pass for an honest one: train.py refuses a run name
# without "leak" in it, log.txt prints a banner, config_snapshot.json records
# this value, and every training-curve plot is watermarked.
# Expect a small effect: 72 of the 105 val patients already have an identical
# BraTS19/BraTS20 twin in train under the honest split (journal, Session 8).
cfg.data.leak = "patient"

# NATIVE 1mm pipeline (Session 5+). The scans are kept at their native 1mm
# 240x240x155 grid — NO Resized downsample, NO fixed depth window. Instead:
#   * CropForegroundd crops every split to the brain's foreground bounding box
#     (adaptive per patient), which removes the near-empty top/bottom slices
#     the old fixed [40:136) window was hand-picked to drop — but adaptively,
#     so it can never clip a brain that sits outside a fixed guess (finding D).
#   * Training samples 128x128x96 patches with RandCropByPosNegLabeld, centred
#     on tumour with probability pos/(pos+neg); val/test feed the whole
#     foreground-cropped volume through sliding-window inference (roi_size =
#     cfg.unetr.img_shape), so no crop window is imposed at eval.
#
# fg_threshold: a voxel counts as brain (foreground) if any modality's RAW
# intensity exceeds this. BraTS background is exactly 0, so 0 is the natural
# cut; CropForegroundd runs BEFORE z-score, on raw intensities.
cfg.crop = EasyDict()
cfg.crop.fg_threshold = 0
# RandCropByPosNegLabeld: tumour-centred vs random crop ratio, and how many
# crops to draw per volume per step. num_samples>1 multiplies the effective
# batch (each crop is a training example), so it needs list_data_collate on the
# train loader (wired in utils/dataloader.py).
cfg.crop.pos = 2
cfg.crop.neg = 1
cfg.crop.num_samples = 2

# ---------------------------------------------------------------------------
# UNETR model
# ---------------------------------------------------------------------------
cfg.unetr = EasyDict()
cfg.unetr.img_shape = (128, 128, 96)
cfg.unetr.input_dim = 4
cfg.unetr.output_dim = 3
cfg.unetr.patch_size = 16
cfg.unetr.embed_dim = 768
cfg.unetr.num_layers = 12
cfg.unetr.num_heads = 12
cfg.unetr.mlp_dim = 2048
cfg.unetr.extract_layers = [3, 6, 9, 12]
cfg.unetr.dropout = 0.2

# Which encoder feeds the shared decoder (models/unetr.py):
#   "vit"   the original UNETR ViT (blocks/Transformer.py)
#   "mamba" SegMamba-style hierarchical Vision Mamba (blocks/VisionMamba.py)
# Everything downstream of the encoder is identical between the two. "mamba"
# needs the mamba_ssm CUDA kernels, i.e. the `mamba` conda env (see RUN.md).
# `python train.py --encoder vit|mamba` overrides this for one launch, and
# evaluate.py / xai.py rebuild whatever encoder a run was TRAINED with from its
# config_snapshot.json, so this value never has to be flipped back to re-read
# an old run.
cfg.unetr.encoder = "mamba"

# ---------------------------------------------------------------------------
# SegMamba encoder (Xing et al., MICCAI 2024, BraTS 2023) — cfg.unetr.encoder="mamba"
# ---------------------------------------------------------------------------
# dims/depths/d_state/d_conv/expand are SegMamba's published values. Its
# hidden_size (768, the channels of the 1/16 bottleneck block) is taken from
# cfg.unetr.embed_dim, which is also 768.
cfg.mamba = EasyDict()
cfg.mamba.dims = [48, 96, 192, 384]
cfg.mamba.depths = [2, 2, 2, 2]
cfg.mamba.d_state = 16
cfg.mamba.d_conv = 4
cfg.mamba.expand = 2
# NOT part of SegMamba (it has no dropout): matched to the ViT's 0.2 so the
# MC-dropout uncertainty XAI samples both encoders comparably.
cfg.mamba.dropout = 0.2

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
cfg.loss = EasyDict()
cfg.loss.dice_weight = 0.5
cfg.loss.tversky_weight = 0.3      # weight of the Focal-Tversky term (was focal_weight)
cfg.loss.hausdorff_weight = 0.2    # target weight; annealed 0 -> this over training
cfg.loss.tversky_alpha = 0.7       # FN weight; alpha > beta => recall-focused (helps ET/TC)
cfg.loss.tversky_beta = 0.3        # FP weight
cfg.loss.tversky_gamma = 4.0 / 3.0  # focal exponent is 1/gamma = 0.75 (Abraham & Khan)
cfg.loss.hd_anneal_frac = 0.5      # HD weight reaches full at 50% of epochs, holds after
cfg.loss.aux_z6_weight = 0.3
cfg.loss.aux_z3_weight = 0.15

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
cfg.metrics = EasyDict()

# TRUE physical spacing of the volumes the metrics actually see, in mm.
#
# Native 1mm pipeline: Spacingd resamples to 1mm and NOTHING downsamples in
# plane afterwards (Resized is gone), so every voxel the metrics see is a real
# 1mm^3 voxel. Hard-coded (1,1,1), honest by construction rather than by
# correction — no derivation to get wrong. Only HD95 reads this; Dice, IoU,
# mIoU, sensitivity, specificity, F1 and AUC are voxel-counting metrics.
#
# (Historical: under the old resized pipeline this was 240/img_shape = 1.875mm
# in plane. Native-1mm HD95 numbers are therefore NOT comparable to the resized
# runs — they are measured on the harder, full-resolution problem.)
cfg.metrics.voxel_spacing = (1.0, 1.0, 1.0)
cfg.metrics.auc_every_n_epochs = 10

# ---------------------------------------------------------------------------
# Inference (evaluation-time only — none of this needs retraining)
# ---------------------------------------------------------------------------
cfg.infer = EasyDict()

# MONAI's sliding_window_inference defaults (overlap=0.25, mode="constant")
# seam patches together with equal weight, so voxels near a patch edge — where
# the model has least context — count as much as voxels at the centre.
#
# 0.75 for threshold tuning and the test pass: the window stride halves along
# each axis, so a typical foreground-cropped brain gets roughly 2-3x the
# windows of 0.5 (on top of the 8x TTA), and every voxel is averaged over more
# patch positions. Inference-only, so `python evaluate.py` applies it to an
# existing checkpoint without retraining.
cfg.infer.sw_overlap = 0.75
# The per-epoch validation loop stays at 0.5. It is already ~18 of ~45 min per
# epoch (10.4 s/volume x 105 volumes), and 2-3x that on each of 60 epochs would
# add roughly a day to the run. So metrics.csv / val_steps.csv are measured at
# this overlap, testing/ at sw_overlap.
cfg.infer.val_sw_overlap = 0.5
cfg.infer.sw_mode = "gaussian"

# Average predictions over the 8 axis-flip combinations. Costs 8x inference,
# so it is applied at test/evaluate time only, never in the per-epoch
# validation loop (see utils/engine.py:run_inference).
cfg.infer.tta_flips = True

# Per-channel (TC, WT, ET) probability thresholds. 0.5 is only optimal if the
# model is perfectly calibrated per class, which a recall-weighted loss makes
# unlikely. Tune with: python evaluate.py --run <name> --tune-thresholds
#
# Fresh 1mm run: this is only the pre-tune default (tune_thresholds_after_
# training re-derives it on val at the end of the run). LESSON from the
# run1-new-version ET ablation (eval/et_operating_point_ablation.txt): the val
# sweep maximises clean-case Dice, which is blind to the hallucination count
# that dominates ET HD95, so it drives the ET threshold to the grid floor and
# INFLATES HD95. When promoting the tuned thresholds, ship the ET-threshold
# KNEE (the lowest threshold that still holds the hallucination floor), not the
# sweep's raw argmax.
cfg.infer.thresholds = (0.5, 0.5, 0.5)

# Search the thresholds above on the VAL split at the end of training, before
# the test pass runs. Tuning on val and reporting on test is the whole point —
# tuning on test would be tuning on the number you are about to publish.
# Costs one extra TTA pass over the val split (~40min); set False to skip and
# keep whatever cfg.infer.thresholds says.
cfg.infer.tune_thresholds_after_training = True

# Connected-component cleanup, per channel (TC, WT, ET), in voxels.
# A component smaller than min_component_voxels is deleted; if what survives
# totals less than min_total_voxels the channel is zeroed outright.
#
# The second rule is the one that matters for ET: compute_hd95 returns the
# 374.0 penalty whenever exactly one of {prediction, ground truth} is empty,
# so a handful of stray FP voxels on an ET-negative patient costs more HD95
# than every correctly segmented patient combined.
#
# RESCALED for native 1mm: a voxel is now 1mm^3 (was 1.875 x 1.875 x 1.0 =
# 3.52mm^3 under the resized pipeline). To keep the SAME PHYSICAL cleanup as
# run1-new-version's (0,0,50)/(0,0,100) — i.e. ~176mm^3 / ~352mm^3 — the voxel
# counts scale by 3.52x: (0,0,176) and (0,0,352). The ET ablation found the
# probability threshold is the cleaner lever anyway (it removes FP scatter
# without zeroing real small-ET patients), so keep min_total at this modest
# physical volume and tune ET recall via the threshold knee, not by inflating
# this floor.
cfg.infer.min_component_voxels = (0, 0, 176)
cfg.infer.min_total_voxels = (0, 0, 352)

# ---------------------------------------------------------------------------
# Training-curve plots (drawn from the CSVs at the end of train.py;
# `python tools/replot.py --run <name>` redraws them with other settings)
# ---------------------------------------------------------------------------
cfg.plot = EasyDict()
cfg.plot.font_family = "serif"
cfg.plot.font_size = 12
cfg.plot.fig_size = (7.0, 4.5)   # inches
cfg.plot.dpi = 150
cfg.plot.formats = ["png"]       # any of png / pdf / svg
# Exponential moving average (TensorBoard-style, bias-corrected), applied when
# drawing only: the CSVs keep the raw values, each raw curve is drawn faintly
# under its smoothed one, and the weight is printed on the figure. 0 = off.
# Per-epoch curves have 60 points and get a light touch; the per-step training
# loss is ~530 random-patch losses per epoch and needs much more to be readable.
cfg.plot.epoch_smoothing = 0.3
cfg.plot.step_smoothing = 0.9

# ---------------------------------------------------------------------------
# Attention overlay visualization
# ---------------------------------------------------------------------------
cfg.attention = EasyDict()
cfg.attention.enabled = True
cfg.attention.every_n_epochs = 10
cfg.attention.sample_idx = 0

# ---------------------------------------------------------------------------
# Qualitative sample visualization (pre-training data checks)
# ---------------------------------------------------------------------------
cfg.visualization = EasyDict()
cfg.visualization.sample_index = 27
cfg.visualization.extra_indices = [0, 5, 10, 15, 20]

# ---------------------------------------------------------------------------
# Explainability (post-hoc — run via xai.py against a saved checkpoint)
# ---------------------------------------------------------------------------
cfg.xai = EasyDict()

# Run the suite automatically at the end of train.py, into logs/<run>/xai/, so
# one `python train.py` produces the full artifact set. Adds ~30-45min to a
# ~45h run. A failing component is caught and logged, never allowed to discard
# a finished training run. Set False to skip and use xai.py separately.
cfg.xai.run_after_training = True
# All five components, so one `python train.py` yields the same xai/ folder
# the ViT run logs/v2-run3 has (its cam/rollout/faithful were produced with
# xai.py afterwards) and the two encoders can be compared file for file. For
# "rollout" the Mamba model reports its hidden-attention rollout (utils/xai.py).
# Session-4 caveat still applies when READING cam/faithful: the CAM at
# decoder0_header.1 sits one 1x1 conv from the output, so its support is the GT
# mask by construction (mass_in_tumor ~ 1.0) — use the deeper decoder layers.
# The headline XAI remains modality + uncertainty.
cfg.xai.components = ["cam", "modality", "uncertainty", "rollout", "faithful"]

# Test-set samples to explain. Keep small: every extra sample multiplies the
# faithfulness sweeps.
cfg.xai.sample_indices = [0, 1, 2]

# Decoder layers to attribute at, coarse -> fine. Dotted attribute paths
# resolved against the model. Showing how localisation sharpens down the
# decoder is the point, so keep more than one.
cfg.xai.cam_layers = [
    "decoder9_upsampler",   # 256ch @ 32x32x24
    "decoder6_upsampler",   # 128ch @ 64x64x48
    "decoder3_upsampler",   #  64ch @ 128x128x96 (full res)
    "decoder0_header.1",    #  64ch, last layer BEFORE the 1x1 output conv
]
# decoder0_header.1 rather than decoder0_header: the module's final child IS
# the 1x1 output conv, so its activation is the logits themselves and a CAM
# there just re-draws the prediction.
# "hires" = HiResCAM (elementwise grad*activation, faithful by construction);
# "grad" = classic Grad-CAM (gradients pooled to per-channel scalars first).
# Both are computed so the paper can compare them.
cfg.xai.cam_methods = ["hires", "grad"]

# Restrict the CAM score to the ground-truth region of the explained class
# ("gt"), the model's own prediction ("pred"), or the whole volume ("all").
# Seg-Grad-CAM (Vinogradova et al., AAAI 2020) sums logits over a region
# rather than over everything — without this the explanation is dominated by
# the background, which is 99% of the voxels.
cfg.xai.cam_roi = "gt"

# MC-dropout: stochastic forward passes with dropout left active at eval.
cfg.xai.mc_passes = 10

# Faithfulness sweep: fractions of highest-attributed voxels to delete.
cfg.xai.deletion_fractions = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# Top-k attribution mass used for the localisation score.
cfg.xai.localization_top_frac = 0.05
# Dilation radius (voxels) for the peritumoral shell in the localisation test.
cfg.xai.peritumoral_radius = 5

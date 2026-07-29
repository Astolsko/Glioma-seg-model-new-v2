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
cfg.epoch = 50   # 50->100 bought only +0.009 mean Dice for ~16h on the resized run
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
cfg.warmup_epochs = 5

# Exponential moving average of the weights. Validation, checkpointing and
# testing all use the EMA copy; set to 0 to disable and train/eval the raw
# weights. 0.999 over ~500 steps/epoch is a ~2-epoch averaging horizon.
cfg.ema_decay = 0.999

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
cfg.infer.sw_overlap = 0.5
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
# Only the two defensible components (Session 4). "cam"/"faithful" are cut: the
# CAM is taken one 1x1 conv from the output, so its support IS the GT mask
# (mass_in_tumor = 0.9999999) and the randomisation SSIM cannot move — the
# faithfulness suite is measuring a tautology, not the model. Dropping
# "faithful" also removes a weight-randomising step from the end of the run.
# "rollout" goes with them (heatmap-only, not load-bearing without X5).
cfg.xai.components = ["modality", "uncertainty"]

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

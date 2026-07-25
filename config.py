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
cfg.epoch = 100
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

# Depth cropping now happens on the RAW scan (right after Orientation+
# Spacing, BEFORE Resized) using absolute raw slice indices — NOT on the
# post-resize volume. This dataset is co-registered (every patient is
# exactly 240x240x155 at 1mm spacing), so a fixed raw-slice window means the
# same physical anatomical range for every patient.
#
# *_start_slice is the only number you set — the window always keeps
# exactly cfg.unetr.img_shape[-1] slices (so end_slice = start_slice +
# cfg.unetr.img_shape[-1], computed automatically in
# utils/transforms.py:CropRawDepthd). That guarantees Resized's depth
# resize is a true no-op (input depth == output depth == img_shape[-1]),
# i.e. no nearest-neighbor slice-dropping — instead of compressing the
# whole 155-slice scan down to img_shape[-1] slices and then cropping a
# window out of THAT lossy result (the old behavior), we crop the raw scan
# first and only resize H/W afterward.
#
# PLACEHOLDER VALUES — pick real ones with tools/crop_visual_check.py
# (it loads one raw sample, applies this exact crop, and dumps every slice
# before/after to PNG so you can see whether the window is cutting off
# brain/tumor at either end).
cfg.crop = EasyDict()
cfg.crop.train_start_slice = 40
cfg.crop.val_start_slice = 40
cfg.crop.threshold = 0.25

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
# Spacingd resamples to 1mm, which for BraTS is the native 240x240x155 grid.
# CropRawDepthd then fixes depth to exactly img_shape[-1] slices, so Resized's
# depth resize is a no-op and depth stays 1mm/voxel. H and W, however, get
# squeezed 240 -> img_shape[0:2] by that same Resized, so one in-plane voxel
# spans 240/128 = 1.875mm.
#
# This used to be hardcoded (1,1,1), which understated every in-plane HD95 by
# 1.875x — the reported "mm" were not mm. Only HD95 reads this value: Dice,
# IoU, mIoU, sensitivity, specificity, F1 and AUC are all voxel-counting
# metrics and are unchanged by the correction. Expect HD95 to go UP after this
# fix; that is the honest number, not a regression.
cfg.metrics.raw_inplane_size = 240   # BraTS in-plane extent at 1mm, post-Spacingd
cfg.metrics.voxel_spacing = (
    cfg.metrics.raw_inplane_size / cfg.unetr.img_shape[0],
    cfg.metrics.raw_inplane_size / cfg.unetr.img_shape[1],
    1.0,
)
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
# than every correctly segmented patient combined. One in-plane voxel is
# 1.875 x 1.875 x 1.0 mm = 3.5mm^3, so 50 voxels is ~176mm^3.
cfg.infer.min_component_voxels = (0, 0, 50)
cfg.infer.min_total_voxels = (0, 0, 100)

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

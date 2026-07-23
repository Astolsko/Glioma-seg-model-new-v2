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
cfg.epoch = 50
cfg.learning_rate = 1e-4
cfg.weight_decay = 1e-4
cfg.patience = 50
cfg.grad_clip_norm = 1.0
cfg.val_interval = 1
cfg.val_amp = True
cfg.seed = 0

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
cfg.metrics.voxel_spacing = (1.0, 1.0, 1.0)
cfg.metrics.auc_every_n_epochs = 10

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

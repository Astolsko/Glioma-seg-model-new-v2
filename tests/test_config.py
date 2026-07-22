"""Sanity checks on config.py itself — catches typos/inconsistent values that
would otherwise only surface hours into a training run (e.g. a crop range
that doesn't match img_shape, or extract_layers that breaks UNETR's forward).
"""
import pytest

from config import cfg


def test_unetr_img_shape_divisible_by_patch_size():
    for dim in cfg.unetr.img_shape:
        assert dim % cfg.unetr.patch_size == 0, (
            f"img_shape {cfg.unetr.img_shape} must be divisible by "
            f"patch_size {cfg.unetr.patch_size} or patch embedding breaks"
        )


def test_unetr_embed_dim_divisible_by_num_heads():
    assert cfg.unetr.embed_dim % cfg.unetr.num_heads == 0


def test_unetr_extract_layers_valid_for_forward_unpacking():
    # UNETR.forward does: z0, z3, z6, z9, z12 = x, *z — requires exactly 4
    # extracted hidden states.
    assert len(cfg.unetr.extract_layers) == 4
    assert all(1 <= layer <= cfg.unetr.num_layers for layer in cfg.unetr.extract_layers)
    assert list(cfg.unetr.extract_layers) == sorted(cfg.unetr.extract_layers)


def test_crop_start_slices_are_nonnegative():
    assert cfg.crop.train_start_slice >= 0
    assert cfg.crop.val_start_slice >= 0


def test_crop_window_fits_within_known_raw_scan_depth():
    """BraTS scans are co-registered to a fixed 240x240x155 grid — verified
    directly against this dataset (every sampled patient in data/combined/
    has exactly that shape at 1mm spacing). CropRawDepthd now crops the RAW
    scan (before Resized), always keeping exactly cfg.unetr.img_shape[-1]
    slices starting at *_start_slice — so start_slice + img_shape[-1] must
    not exceed 155, or the window runs past the available depth (triggering
    CropRawDepthd's MISMATCH warning) and Resized has to compress/stretch
    again, defeating the point of cropping on the raw scan.
    """
    BRATS_RAW_DEPTH = 155
    target_depth = cfg.unetr.img_shape[-1]

    assert cfg.crop.train_start_slice + target_depth <= BRATS_RAW_DEPTH, (
        f"train_start_slice={cfg.crop.train_start_slice} + img_shape[-1]={target_depth} "
        f"exceeds the raw scan depth ({BRATS_RAW_DEPTH}) — use tools/crop_visual_check.py "
        "to pick a start_slice that leaves room for the full window."
    )
    assert cfg.crop.val_start_slice + target_depth <= BRATS_RAW_DEPTH, (
        f"val_start_slice={cfg.crop.val_start_slice} + img_shape[-1]={target_depth} "
        f"exceeds the raw scan depth ({BRATS_RAW_DEPTH}) — use tools/crop_visual_check.py "
        "to pick a start_slice that leaves room for the full window."
    )


def test_loss_weights_are_nonnegative():
    assert cfg.loss.dice_weight >= 0
    assert cfg.loss.focal_weight >= 0
    assert cfg.loss.hausdorff_weight >= 0
    assert cfg.loss.aux_z6_weight >= 0
    assert cfg.loss.aux_z3_weight >= 0


def test_data_fractions_leave_room_for_training_split():
    assert 0 < cfg.data.val_frac < 1
    assert 0 < cfg.data.test_frac < 1
    assert cfg.data.val_frac + cfg.data.test_frac < 1

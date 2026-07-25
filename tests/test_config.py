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
    assert cfg.loss.tversky_weight >= 0
    assert cfg.loss.hausdorff_weight >= 0
    assert cfg.loss.aux_z6_weight >= 0
    assert cfg.loss.aux_z3_weight >= 0


def test_tversky_recall_weighting_and_anneal_are_sane():
    # alpha (FN weight) must exceed beta (FP weight) for the recall focus
    assert cfg.loss.tversky_alpha > cfg.loss.tversky_beta
    assert cfg.loss.tversky_gamma > 0
    assert 0 <= cfg.loss.hd_anneal_frac <= 1


def test_data_fractions_leave_room_for_training_split():
    assert 0 < cfg.data.val_frac < 1
    assert 0 < cfg.data.test_frac < 1
    assert cfg.data.val_frac + cfg.data.test_frac < 1


def test_voxel_spacing_matches_the_resize_the_pipeline_actually_applies():
    """HD95 is the only metric that reads voxel_spacing, and it reports mm.

    Resized squeezes H/W from BraTS's 240 down to img_shape[0:2], so an
    in-plane voxel is 240/128 = 1.875mm, not 1mm. Depth is exempt because
    CropRawDepthd already hands Resized exactly img_shape[-1] slices, making
    the depth resize a no-op. Hardcoding (1,1,1) here understated every
    in-plane HD95 by 1.875x.
    """
    expected = (
        cfg.metrics.raw_inplane_size / cfg.unetr.img_shape[0],
        cfg.metrics.raw_inplane_size / cfg.unetr.img_shape[1],
        1.0,
    )
    assert tuple(cfg.metrics.voxel_spacing) == pytest.approx(expected)
    assert cfg.metrics.voxel_spacing[2] == 1.0, (
        "depth must stay 1mm/voxel — if it doesn't, CropRawDepthd is no longer "
        "handing Resized an exact-depth window and the resize is resampling slices"
    )


def test_inference_settings_have_one_entry_per_output_channel():
    for name in ("thresholds", "min_component_voxels", "min_total_voxels"):
        assert len(cfg.infer[name]) == cfg.unetr.output_dim, (
            f"cfg.infer.{name} is indexed per channel (TC, WT, ET)"
        )
    assert all(0.0 < t < 1.0 for t in cfg.infer.thresholds)
    assert 0.0 <= cfg.infer.sw_overlap < 1.0


def test_warmup_fits_inside_the_training_schedule():
    assert 0 <= cfg.warmup_epochs < cfg.epoch
    assert 0 <= cfg.ema_decay < 1


def test_xai_cam_layers_are_resolvable_paths():
    # xai._resolve_layer walks these against the model; a typo here only
    # surfaces once a checkpoint has been loaded and the run is underway.
    assert cfg.xai.cam_layers
    assert all(part.isdigit() or part.isidentifier()
               for path in cfg.xai.cam_layers for part in path.split("."))
    assert cfg.xai.cam_roi in ("gt", "pred", "all")
    assert all(m in ("hires", "grad") for m in cfg.xai.cam_methods)
    assert cfg.xai.mc_passes >= 2
    assert cfg.xai.deletion_fractions[0] == 0.0, "curve needs an unperturbed anchor"

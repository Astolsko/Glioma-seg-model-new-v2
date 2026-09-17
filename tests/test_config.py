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


def test_crop_config_is_sane_for_native_1mm():
    """Native 1mm pipeline: CropForegroundd (fg_threshold) + RandCropByPosNegLabeld
    (pos/neg/num_samples) replaced the old fixed-depth CropRawDepthd, so there
    are no more *_start_slice knobs — the brain crop is adaptive."""
    assert cfg.crop.fg_threshold >= 0
    assert cfg.crop.pos > 0 and cfg.crop.neg >= 0
    assert cfg.crop.pos + cfg.crop.neg > 0
    assert cfg.crop.num_samples >= 1


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


def test_voxel_spacing_is_native_1mm():
    """HD95 is the only metric that reads voxel_spacing, and it reports mm.

    The native 1mm pipeline keeps full resolution (no Resized), so every voxel
    is a real 1mm^3 voxel and voxel_spacing is (1,1,1), honest by construction.
    (Under the old resized pipeline this was 240/img_shape = 1.875mm in plane.)
    """
    assert tuple(cfg.metrics.voxel_spacing) == pytest.approx((1.0, 1.0, 1.0))


def test_inference_settings_have_one_entry_per_output_channel():
    for name in ("thresholds", "min_component_voxels", "min_total_voxels"):
        assert len(cfg.infer[name]) == cfg.unetr.output_dim, (
            f"cfg.infer.{name} is indexed per channel (TC, WT, ET)"
        )
    assert all(0.0 < t < 1.0 for t in cfg.infer.thresholds)
    assert 0.0 <= cfg.infer.sw_overlap < 1.0
    assert 0.0 <= cfg.infer.val_sw_overlap < 1.0


def test_plot_smoothing_weights_are_valid_ema_weights():
    # utils.plot.smooth_series raises outside [0, 1), and the plots are drawn at
    # the end of a ~45h run, just before threshold tuning and the test pass.
    assert 0.0 <= cfg.plot.epoch_smoothing < 1.0
    assert 0.0 <= cfg.plot.step_smoothing < 1.0


def test_data_leak_is_a_known_mode():
    # utils.dataloader.LEAK_MODES; checked here too so a typo fails before a run.
    assert cfg.data.leak in (None, "patient")


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

"""Tests for the explainability suite.

These run the real tiny UNETR so hook wiring, gradient flow and shape handling
are exercised end to end — the failure modes in XAI code are almost always a
hook that never fires or a map that is silently all-zeros, and neither shows up
in a shape assertion alone.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from utils import xai


@pytest.fixture
def sample(tiny_unetr_kwargs):
    """One synthetic (image, label) pair at the tiny model's input size, with a
    solid cube of tumour so ROI-restricted scoring has something to bite on."""
    shape = tiny_unetr_kwargs["img_shape"]
    image = torch.randn(1, tiny_unetr_kwargs["input_dim"], *shape)
    label = torch.zeros(1, tiny_unetr_kwargs["output_dim"], *shape)
    label[:, :, 4:12, 4:12, 2:6] = 1.0
    return image, label


# --- X1 -------------------------------------------------------------------

@pytest.mark.parametrize("method", ["hires", "grad"])
def test_cam_returns_normalized_map_at_input_resolution(tiny_unetr, sample, method):
    from config import cfg
    image, label = sample
    tiny_unetr.eval()

    cam, logits = xai.cam_for_sample(
        tiny_unetr, image, label, class_idx=2, cfg=cfg,
        layer_path="decoder0_header.1", method=method)

    assert cam.shape == tuple(image.shape[2:])
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert np.isfinite(cam).all()
    assert logits.shape[1] == 3


def test_cam_is_not_all_zeros(tiny_unetr, sample):
    """The classic silent failure: the hook never fires, or the ROI is empty,
    and the 'explanation' is a blank volume that still passes a shape check."""
    from config import cfg
    image, label = sample
    tiny_unetr.eval()

    cam, _ = xai.cam_for_sample(tiny_unetr, image, label, class_idx=2, cfg=cfg,
                                layer_path="decoder0_header.1", method="hires")

    assert cam.max() > 0.0


def test_cam_hook_is_removed_after_use(tiny_unetr):
    layer = xai._resolve_layer(tiny_unetr, "decoder0_header.1")
    before = len(layer._forward_hooks)

    with xai.CAM3D(tiny_unetr, "decoder0_header.1"):
        assert len(layer._forward_hooks) == before + 1

    assert len(layer._forward_hooks) == before


def test_cam_rejects_unknown_method(tiny_unetr, sample):
    image, _ = sample
    tiny_unetr.eval()
    with xai.CAM3D(tiny_unetr, "decoder0_header.1") as cam:
        with pytest.raises(ValueError):
            cam.attribute(image, class_idx=0, method="nope")


def test_resolve_layer_walks_indices_and_attributes(tiny_unetr):
    assert xai._resolve_layer(tiny_unetr, "decoder0_header.1") is tiny_unetr.decoder0_header[1]
    assert xai._resolve_layer(tiny_unetr, "transformer") is tiny_unetr.transformer


# --- X3 -------------------------------------------------------------------

def test_enable_dropout_reactivates_only_dropout_layers(tiny_unetr):
    tiny_unetr.eval()
    count = xai.enable_dropout(tiny_unetr)

    assert count > 0
    assert all(m.training for m in tiny_unetr.modules()
               if isinstance(m, torch.nn.Dropout))
    # everything else must stay in eval, or the forward signature changes
    assert not tiny_unetr.training


def test_mc_dropout_returns_matching_shapes_and_restores_mode(tiny_unetr, sample):
    image, _ = sample
    tiny_unetr.eval()

    mean_prob, std, entropy, n_dropout = xai.mc_dropout_predict(tiny_unetr, image, n_passes=3)

    expected = (3, *image.shape[2:])
    assert mean_prob.shape == std.shape == entropy.shape == expected
    assert (mean_prob >= 0).all() and (mean_prob <= 1).all()
    assert (entropy >= 0).all()
    assert n_dropout > 0
    assert not tiny_unetr.training, "eval mode must be restored"


def test_error_retention_curve_reaches_perfect_dice_when_everything_is_referred():
    gt = np.zeros((4, 4, 4), dtype=bool)
    gt[:2] = True
    pred = np.zeros_like(gt)                      # completely wrong
    uncertainty = np.random.default_rng(0).random(gt.shape)

    curve = xai.error_retention_curve(pred, gt, uncertainty, [0.0, 1.0])

    assert curve[0] == pytest.approx(0.0)
    assert curve[-1] == pytest.approx(1.0)


# --- X4 -------------------------------------------------------------------

def test_attention_rollout_covers_the_input_volume(tiny_unetr, sample):
    from config import cfg
    image, _ = sample
    tiny_unetr.eval()

    relevance = xai.attention_rollout(tiny_unetr, image, cfg)

    assert relevance is not None
    assert relevance.shape == tuple(image.shape[2:])
    assert relevance.min() >= 0.0 and relevance.max() <= 1.0


def test_attention_rollout_removes_its_hooks(tiny_unetr, sample):
    from config import cfg
    image, _ = sample
    before = [len(b.attn._forward_hooks) for b in tiny_unetr.transformer.layer]

    xai.attention_rollout(tiny_unetr.eval(), image, cfg)

    assert [len(b.attn._forward_hooks) for b in tiny_unetr.transformer.layer] == before


# --- X5 -------------------------------------------------------------------

def test_localization_scores_partition_the_top_k_mass():
    gt = np.zeros((8, 8, 8), dtype=bool)
    gt[2:6, 2:6, 2:6] = True
    attribution = gt.astype(np.float32)            # all mass exactly on the tumour

    scores = xai.localization_scores(attribution, gt, top_frac=0.05, peritumoral_radius=2)

    assert scores["inside_tumor"] == pytest.approx(1.0)
    assert sum(scores.values()) == pytest.approx(1.0)


def test_localization_scores_detect_attribution_outside_the_tumor():
    gt = np.zeros((8, 8, 8), dtype=bool)
    gt[0:2, 0:2, 0:2] = True
    attribution = np.zeros((8, 8, 8), dtype=np.float32)
    attribution[7, 7, 7] = 1.0                     # far corner, nowhere near the tumour

    scores = xai.localization_scores(attribution, gt, top_frac=0.002, peritumoral_radius=1)

    assert scores["inside_tumor"] == pytest.approx(0.0)
    assert scores["elsewhere"] > 0.0


def test_deletion_curve_has_one_point_per_fraction(tiny_unetr, sample):
    image, label = sample
    tiny_unetr.eval()
    gt = label[0, 2].numpy() > 0.5
    attribution = np.random.default_rng(0).random(tuple(image.shape[2:])).astype(np.float32)

    fractions = [0.0, 0.1, 0.5]
    curve = xai.deletion_curve(tiny_unetr, image, gt, attribution, class_idx=2,
                               fractions=fractions)

    assert len(curve) == len(fractions)
    assert all(0.0 <= v <= 1.0 for v in curve)


def test_auc_of_a_flat_curve_equals_its_value():
    assert xai.auc([0.5, 0.5, 0.5], [0.0, 0.5, 1.0]) == pytest.approx(0.5)


def test_sanity_check_randomization_reports_one_ssim_per_stage(tiny_unetr, sample):
    """Adebayo et al.'s test. Asserting the SSIM *decays* would be flaky on a
    randomly initialised tiny model (its 'trained' map is already noise), so
    the unit test pins the mechanics; the decay itself is what the real run
    reports and what the paper must show.
    """
    from config import cfg
    image, label = sample
    tiny_unetr.eval()
    order = ["decoder0_header", "decoder3_upsampler"]

    results = xai.sanity_check_randomization(
        tiny_unetr, image, label, class_idx=2, cfg=cfg,
        layer_path="decoder0_header.1", method="hires", order=order)

    assert [r["randomized_through"] for r in results] == order
    assert all(-1.0 <= r["ssim_vs_trained"] <= 1.0 for r in results)


def test_dice_treats_two_empty_masks_as_agreement():
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert xai.dice(empty, empty) == 1.0
    assert xai.dice(np.ones_like(empty), empty) == 0.0


# --- orchestration --------------------------------------------------------

class _OneSampleDataset:
    def __init__(self, image, label):
        self.image, self.label = image[0], label[0]

    def __len__(self):
        return 1

    def __getitem__(self, _idx):
        return {"image": self.image, "label": self.label}


@pytest.fixture
def suite_args(tiny_unetr, sample, tmp_path, monkeypatch):
    from config import cfg
    image, label = sample
    monkeypatch.setattr(cfg.xai, "sample_indices", [0], raising=True)

    checkpoint = tmp_path / "ckpt.pth"
    torch.save(tiny_unetr.state_dict(), str(checkpoint))
    dataset = _OneSampleDataset(image, label)

    return dict(
        model=tiny_unetr.eval(), test_ds=dataset, test_loader=[],
        checkpoint_path=str(checkpoint), out_dir=str(tmp_path / "xai"),
        device="cpu", cfg=cfg, inferer=lambda m, x: m(x),
    )


def test_run_xai_suite_contains_a_failing_component(suite_args, monkeypatch):
    """A broken explanation must never discard a finished training run — the
    suite runs at the very end of train.py, after ~45h of GPU time.
    """
    from utils import xai as xai_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(xai_module, "component_rollout", _boom)

    summary = xai_module.run_xai_suite(components=["rollout"], **suite_args)

    assert "synthetic failure" in summary["rollout"]["error"]
    assert os.path.exists(os.path.join(suite_args["out_dir"], "summary.json"))


def test_run_xai_suite_writes_per_component_json(suite_args):
    summary = xai.run_xai_suite(components=["rollout"], **suite_args)

    assert "rollout" in summary
    assert os.path.exists(os.path.join(suite_args["out_dir"], "rollout.json"))


def test_run_xai_suite_runs_faithful_last_whatever_order_is_asked(suite_args, monkeypatch):
    """component_faithful randomises the model's weights, so ordering is not
    cosmetic: any component after it would explain a destroyed model."""
    from utils import xai as xai_module
    order = []

    for name in ("cam", "rollout", "faithful"):
        monkeypatch.setattr(xai_module, f"component_{name}",
                            lambda *a, _n=name, **k: order.append(_n) or {})

    xai_module.run_xai_suite(components=["faithful", "rollout", "cam"], **suite_args)

    assert order[-1] == "faithful"


def test_informative_slices_prefers_slices_with_the_most_tumor():
    gt = np.zeros((8, 8, 10), dtype=bool)
    gt[:, :, 3] = True                              # densest slice
    gt[0:2, 0:2, 9] = True                          # sparse slice

    chosen = xai.informative_slices(gt, n_slices=1, min_gap=2)

    assert chosen == [3]

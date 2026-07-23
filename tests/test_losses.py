import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from utils.losses import build_loss_fn, combine_main_and_aux


@pytest.fixture
def loss_cfg():
    from easydict import EasyDict
    cfg = EasyDict()
    cfg.loss = EasyDict()
    cfg.loss.dice_weight = 0.5
    cfg.loss.tversky_weight = 0.3
    cfg.loss.hausdorff_weight = 0.2
    cfg.loss.tversky_alpha = 0.7
    cfg.loss.tversky_beta = 0.3
    cfg.loss.tversky_gamma = 4.0 / 3.0
    cfg.loss.hd_anneal_frac = 0.5
    cfg.loss.aux_z6_weight = 0.3
    cfg.loss.aux_z3_weight = 0.15
    return cfg


def test_build_loss_fn_returns_finite_scalar(loss_cfg):
    loss_fn = build_loss_fn(loss_cfg)
    outputs = torch.randn(1, 3, 8, 8, 8)
    labels = (torch.rand(1, 3, 8, 8, 8) > 0.5).float()

    loss = loss_fn(outputs, labels)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_build_loss_fn_is_lower_for_near_perfect_prediction(loss_cfg):
    loss_fn = build_loss_fn(loss_cfg)
    labels = (torch.rand(1, 3, 8, 8, 8) > 0.5).float()

    # near-perfect logits: large positive where label==1, large negative where label==0
    good_logits = (labels * 2 - 1) * 10
    bad_logits = torch.randn(1, 3, 8, 8, 8)

    good_loss = loss_fn(good_logits, labels)
    bad_loss = loss_fn(bad_logits, labels)

    assert good_loss.item() < bad_loss.item()


def test_build_loss_fn_requires_grad_flows(loss_cfg):
    loss_fn = build_loss_fn(loss_cfg)
    outputs = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
    labels = (torch.rand(1, 3, 8, 8, 8) > 0.5).float()

    loss = loss_fn(outputs, labels)
    loss.backward()

    assert outputs.grad is not None
    assert torch.isfinite(outputs.grad).all()


def test_hd_term_is_annealed_from_zero_to_full(loss_cfg):
    loss = build_loss_fn(loss_cfg)
    loss.set_epoch(0, 10)                       # start: HD off
    assert loss._hd_scale == 0.0
    loss.set_epoch(9, 10)                       # end: frac=1.0 -> full
    assert loss._hd_scale == pytest.approx(1.0)
    loss.set_epoch(3, 10)                       # frac=1/3, ramp 0.5 -> 2/3
    assert loss._hd_scale == pytest.approx(min(1.0, (3 / 9) / 0.5))


def _cube(pos_idx, value, n=64):
    """Build a (1,1,4,4,4) tensor with `value` at flat indices `pos_idx`."""
    flat = torch.full((1, 1, n), -value if value > 0 else 0.0)
    for i in pos_idx:
        flat[0, 0, i] = value
    return flat.view(1, 1, 4, 4, 4)


def test_focal_tversky_penalizes_false_negatives_more_than_false_positives(loss_cfg):
    # alpha=0.7 > beta=0.3 => missing positives (recall error, the ET/TC failure
    # mode) must cost more than adding the same number of false positives.
    loss = build_loss_fn(loss_cfg)

    # FN case: gt has 8 positives, prediction hits only 4 -> TP=4, FN=4, FP=0
    gt_fn = _cube(range(8), 1.0)
    pred_fn = _cube(range(4), 10.0)

    # FP case: gt has 4 positives, prediction hits all 4 + 4 extras -> TP=4, FN=0, FP=4
    gt_fp = _cube(range(4), 1.0)
    pred_fp = _cube(list(range(4)) + [8, 9, 10, 11], 10.0)

    l_fn = loss._focal_tversky(pred_fn, gt_fn)
    l_fp = loss._focal_tversky(pred_fp, gt_fp)

    assert l_fn.item() > l_fp.item()


def test_combine_main_and_aux_weights_are_applied_exactly(loss_cfg):
    calls = []

    def fake_loss_fn(pred, target):
        calls.append((pred, target))
        # return the sum of pred as a stand-in "loss" so we can verify the
        # exact weighted-combination arithmetic below
        return pred.sum()

    outputs = torch.tensor(2.0)
    aux_z6 = torch.tensor(3.0)
    aux_z3 = torch.tensor(4.0)
    labels = torch.tensor(0.0)

    total = combine_main_and_aux(fake_loss_fn, outputs, aux_z6, aux_z3, labels, loss_cfg)

    expected = 2.0 + loss_cfg.loss.aux_z6_weight * 3.0 + loss_cfg.loss.aux_z3_weight * 4.0
    assert total.item() == pytest.approx(expected)
    assert len(calls) == 3


def test_combine_main_and_aux_calls_loss_fn_with_correct_args(loss_cfg):
    seen = []

    def fake_loss_fn(pred, target):
        seen.append((pred, target))
        return torch.tensor(0.0)

    outputs, aux_z6, aux_z3, labels = (torch.tensor(1.0), torch.tensor(2.0),
                                        torch.tensor(3.0), torch.tensor(4.0))
    combine_main_and_aux(fake_loss_fn, outputs, aux_z6, aux_z3, labels, loss_cfg)

    assert seen == [(outputs, labels), (aux_z6, labels), (aux_z3, labels)]

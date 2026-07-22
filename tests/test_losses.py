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
    cfg.loss.focal_weight = 0.3
    cfg.loss.hausdorff_weight = 0.2
    cfg.loss.focal_gamma = 2.0
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

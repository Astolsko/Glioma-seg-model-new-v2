"""Tests for utils/engine.py — the actual training/validation loop. These are
the closest thing to "run one real batch through everything" checks: a tiny
synthetic dataset + tiny UNETR run through build_model / train_one_epoch /
validate, exercising the exact same code path a real multi-hour run uses,
in a couple of seconds.

Note: engine.py hardcodes `torch.amp.autocast('cuda', dtype=torch.bfloat16)`
in train_one_epoch/validate/run_inference regardless of what device the
model is actually on, so those tests are marked `gpu` and skipped without a
CUDA device (matching the fact that a real run would also need one).
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from utils.engine import (
    build_training_components, build_model, run_inference, train_one_epoch,
    validate, METRICS_CSV_FIELDS,
)
from utils.losses import build_loss_fn


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n, input_dim, output_dim, shape):
        self.n = n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shape = shape

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.rand(self.input_dim, *self.shape),
            "label": (torch.rand(self.output_dim, *self.shape) > 0.7).float(),
        }


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="engine.py hardcodes cuda autocast — needs a CUDA device to exercise",
)


def test_metrics_csv_fields_include_base_training_row_keys():
    """Static guard: engine.run_training() always writes epoch/lr/train_loss/
    is_best/epoch_time_sec regardless of validation — if one of these were
    ever dropped from METRICS_CSV_FIELDS, csv.DictWriter would raise
    mid-training the first time that row is logged."""
    base_keys = {"epoch", "lr", "train_loss", "is_best", "epoch_time_sec"}
    assert base_keys <= set(METRICS_CSV_FIELDS)


def test_build_model_constructs_and_runs_forward_with_tiny_cfg(tiny_unetr_kwargs, device):
    from config import cfg
    model = build_model(cfg, device)
    assert isinstance(model, torch.nn.Module)

    model.eval()
    x = torch.randn(1, cfg.unetr.input_dim, *cfg.unetr.img_shape, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, cfg.unetr.output_dim, *cfg.unetr.img_shape)


def test_build_training_components_returns_expected_types(tiny_unetr, tiny_unetr_kwargs):
    from config import cfg
    optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans, scaler = \
        build_training_components(tiny_unetr, cfg)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


@requires_cuda
def test_run_inference_matches_expected_shape(tiny_unetr, tiny_unetr_kwargs, device):
    from config import cfg
    model = tiny_unetr.to(device).eval()
    x = torch.randn(1, tiny_unetr_kwargs["input_dim"], *tiny_unetr_kwargs["img_shape"], device=device)

    with torch.no_grad():
        out = run_inference(model, x, cfg)

    assert out.shape == (1, tiny_unetr_kwargs["output_dim"], *tiny_unetr_kwargs["img_shape"])
    assert torch.isfinite(out).all()


@requires_cuda
def test_train_one_epoch_runs_and_updates_model_weights(tiny_unetr, tiny_unetr_kwargs, device):
    from config import cfg
    model = tiny_unetr.to(device)
    dataset = _TinyDataset(3, tiny_unetr_kwargs["input_dim"], tiny_unetr_kwargs["output_dim"],
                            tiny_unetr_kwargs["img_shape"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = build_loss_fn(cfg)

    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    epoch_loss = train_one_epoch(model, loader, dataset, optimizer, scaler, loss_fn, device, cfg, epoch=0)

    assert epoch_loss == epoch_loss  # not NaN
    assert epoch_loss >= 0
    changed = any(not torch.equal(before[name], p.detach()) for name, p in model.named_parameters())
    assert changed, "no parameters changed after a full optimizer step — backward/step likely no-op'd"


@requires_cuda
def test_validate_runs_and_returned_keys_are_subset_of_metrics_csv_fields(
        tiny_unetr, tiny_unetr_kwargs, device, monkeypatch):
    """The regression test for the exact failure mode test_run_logger.py
    documents: if validate()'s return dict ever gains a key that isn't also
    in METRICS_CSV_FIELDS, logging that row crashes mid-training. Running
    the real validate() here (not a hand-written stand-in) means a future
    edit to validate() that adds/renames a metric gets caught immediately."""
    from config import cfg
    monkeypatch.setattr(cfg.attention, "enabled", False, raising=True)  # keep this test focused on metrics

    model = tiny_unetr.to(device)
    dataset = _TinyDataset(2, tiny_unetr_kwargs["input_dim"], tiny_unetr_kwargs["output_dim"],
                            tiny_unetr_kwargs["img_shape"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    _, _, dice_metric, dice_metric_batch, post_trans, _ = build_training_components(model, cfg)
    loss_fn = build_loss_fn(cfg)

    val_metrics = validate(model, loader, dataset, loss_fn, dice_metric, dice_metric_batch,
                            post_trans, device, cfg, epoch=0, attention_cache={}, attention_dir=None)

    assert set(val_metrics.keys()) <= set(METRICS_CSV_FIELDS)
    for key, value in val_metrics.items():
        assert value == value, f"{key} is NaN"  # NaN != NaN


@requires_cuda
def test_validate_with_attention_enabled_does_not_crash_on_noncubic_shape(
        tiny_unetr, tiny_unetr_kwargs, device, monkeypatch, tmp_path):
    """End-to-end regression test for the attention-overlay axis bug fixed
    in utils/attention.py: validate() with attention enabled, fed a batch
    whose spatial shape is non-cubic (like a real post-crop validation
    sample), must not crash."""
    from config import cfg
    monkeypatch.setattr(cfg.attention, "enabled", True, raising=True)
    monkeypatch.setattr(cfg.attention, "every_n_epochs", 1, raising=True)
    monkeypatch.setattr(cfg.metrics, "auc_every_n_epochs", 1, raising=True)

    model = tiny_unetr.to(device)
    noncubic_shape = (24, 32, 16)  # H != W != D
    dataset = _TinyDataset(1, tiny_unetr_kwargs["input_dim"], tiny_unetr_kwargs["output_dim"], noncubic_shape)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    class _FakeValDS:
        data = [{"image": ["", "", "/fake/patientA_t1ce.nii", ""]}]

    from utils.attention import register_attention_hook
    attention_cache = register_attention_hook(model)

    _, _, dice_metric, dice_metric_batch, post_trans, _ = build_training_components(model, cfg)
    loss_fn = build_loss_fn(cfg)

    validate(model, loader, _FakeValDS(), loss_fn, dice_metric, dice_metric_batch,
             post_trans, device, cfg, epoch=0, attention_cache=attention_cache,
             attention_dir=str(tmp_path))

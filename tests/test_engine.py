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
    validate, METRICS_CSV_FIELDS, TRAIN_STEPS_CSV_FIELDS, VAL_STEPS_CSV_FIELDS,
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


class _ConstantModel(torch.nn.Module):
    """Emits the same logit everywhere and counts its forward passes, so the
    TTA plumbing can be checked without a real model in the way."""

    def __init__(self, value, out_channels=3):
        super().__init__()
        self.value = value
        self.out_channels = out_channels
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.full((x.shape[0], self.out_channels, *x.shape[2:]),
                          self.value, device=x.device, dtype=torch.float32)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="engine.py hardcodes cuda autocast — needs a CUDA device to exercise",
)


def test_run_inference_tta_averages_in_probability_space(tiny_unetr_kwargs, monkeypatch):
    """The 8 flip predictions are averaged as probabilities and mapped back
    through logit(). For a model that predicts a constant, every flip agrees,
    so the round trip must return that exact logit — this is what guarantees
    downstream sigmoid+threshold and AUC are unaffected by TTA.
    """
    from config import cfg
    monkeypatch.setattr(cfg, "val_amp", False, raising=True)

    model = _ConstantModel(1.234)
    x = torch.randn(1, 4, *tiny_unetr_kwargs["img_shape"])

    with torch.no_grad():
        out = run_inference(model, x, cfg, tta=True)

    assert torch.allclose(out, torch.full_like(out, 1.234), atol=1e-3)


def test_run_inference_tta_flag_controls_the_number_of_passes(tiny_unetr_kwargs, monkeypatch):
    from config import cfg
    monkeypatch.setattr(cfg, "val_amp", False, raising=True)
    x = torch.randn(1, 4, *tiny_unetr_kwargs["img_shape"])

    plain = _ConstantModel(0.5)
    with torch.no_grad():
        run_inference(plain, x, cfg, tta=False)

    flipped = _ConstantModel(0.5)
    with torch.no_grad():
        run_inference(flipped, x, cfg, tta=True)

    assert flipped.calls == 8 * plain.calls


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
    (optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans, scaler,
     ema_model) = build_training_components(tiny_unetr, cfg)

    assert isinstance(optimizer, torch.optim.AdamW)
    # warmup_epochs > 0 chains LinearLR into CosineAnnealingLR
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler)
    assert (ema_model is None) == (cfg.ema_decay <= 0)


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

    _, _, dice_metric, dice_metric_batch, post_trans, _, _ = build_training_components(model, cfg)
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

    _, _, dice_metric, dice_metric_batch, post_trans, _, _ = build_training_components(model, cfg)
    loss_fn = build_loss_fn(cfg)

    validate(model, loader, _FakeValDS(), loss_fn, dice_metric, dice_metric_batch,
             post_trans, device, cfg, epoch=0, attention_cache=attention_cache,
             attention_dir=str(tmp_path))


def test_run_inference_overlap_argument_overrides_the_config(tiny_unetr_kwargs, monkeypatch):
    """validate() passes cfg.infer.val_sw_overlap while test/tuning use
    cfg.infer.sw_overlap. More overlap means more windows, i.e. more passes."""
    from config import cfg
    monkeypatch.setattr(cfg, "val_amp", False, raising=True)
    x = torch.randn(1, 4, *(2 * s for s in tiny_unetr_kwargs["img_shape"]))

    low, high = _ConstantModel(0.5), _ConstantModel(0.5)
    with torch.no_grad():
        run_inference(low, x, cfg, tta=False, overlap=0.0)
        run_inference(high, x, cfg, tta=False, overlap=0.75)

    assert high.calls > low.calls


@requires_cuda
def test_train_one_epoch_logs_one_csv_row_per_step(tiny_unetr, tiny_unetr_kwargs, device):
    from config import cfg
    model = tiny_unetr.to(device)
    dataset = _TinyDataset(3, tiny_unetr_kwargs["input_dim"], tiny_unetr_kwargs["output_dim"],
                           tiny_unetr_kwargs["img_shape"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = []

    epoch_loss = train_one_epoch(model, loader, dataset, optimizer, torch.amp.GradScaler("cuda"),
                                 build_loss_fn(cfg), device, cfg, epoch=2,
                                 step_logger=rows.append)

    assert [r["step"] for r in rows] == [1, 2, 3]
    assert [r["global_step"] for r in rows] == [7, 8, 9]
    for row in rows:
        assert set(row) <= set(TRAIN_STEPS_CSV_FIELDS)
        assert row["epoch"] == 3
        # total = main head + weighted aux heads, and aux losses are non-negative
        assert row["loss"] >= row["loss_main"] - 1e-6
    assert epoch_loss == pytest.approx(sum(r["loss"] for r in rows) / len(rows))


@requires_cuda
def test_validate_logs_one_row_per_patient_that_the_epoch_row_averages(
        tiny_unetr, tiny_unetr_kwargs, device, monkeypatch):
    from config import cfg
    monkeypatch.setattr(cfg.attention, "enabled", False, raising=True)
    model = tiny_unetr.to(device)
    dataset = _TinyDataset(2, tiny_unetr_kwargs["input_dim"], tiny_unetr_kwargs["output_dim"],
                           tiny_unetr_kwargs["img_shape"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    _, _, dice_metric, dice_metric_batch, post_trans, _, _ = build_training_components(model, cfg)
    rows = []

    val_metrics = validate(model, loader, dataset, build_loss_fn(cfg), dice_metric,
                           dice_metric_batch, post_trans, device, cfg, epoch=0,
                           attention_cache={}, attention_dir=None, step_logger=rows.append)

    assert [r["sample_index"] for r in rows] == [0, 1]
    for row in rows:
        assert set(row) <= set(VAL_STEPS_CSV_FIELDS)
        assert row["epoch"] == 1
    for key in ("val_loss", "dice_tc", "dice_wt", "dice_et", "iou_tc", "hd95_et", "sens_wt"):
        assert val_metrics[key] == pytest.approx(sum(r[key] for r in rows) / len(rows), abs=1e-5)

"""Tests for utils/checkpoint.py — the full-training-state save/load that lets
an interrupted run continue instead of restarting.

The thing worth testing here is not "does torch.save round-trip a tensor" but
the parts a naive resume gets wrong and only notices 20 epochs later: optimizer
moments, the LR scheduler's position in its cosine, the EMA copy, and the
best-metric bookkeeping that decides which checkpoint gets published.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from utils.checkpoint import load_training_state, peek_epoch, save_training_state
from utils.engine import build_training_components
from utils.progress import RunTimeEstimator


def _components(model, cfg):
    (optimizer, lr_scheduler, _, _, _, scaler, ema_model) = build_training_components(model, cfg)
    return optimizer, lr_scheduler, scaler, ema_model


def _save(path, model, cfg, *, epoch=4, best=0.83, best_epoch=3, not_improved=1,
          estimator=None, optimizer=None, lr_scheduler=None, scaler=None, ema_model=None):
    save_training_state(
        path, epoch=epoch, model=model, ema_model=ema_model, optimizer=optimizer,
        lr_scheduler=lr_scheduler, scaler=scaler, best_metric=best,
        best_metric_epoch=best_epoch, not_improved_epoch=not_improved,
        estimator=estimator, elapsed_sec=1234.5, cfg=cfg,
    )


def test_roundtrip_restores_resume_bookkeeping(tiny_unetr, tiny_unetr_kwargs, tmp_path):
    from config import cfg
    model = tiny_unetr
    optimizer, lr_scheduler, scaler, ema_model = _components(model, cfg)
    path = str(tmp_path / "last.pth")

    _save(path, model, cfg, optimizer=optimizer, lr_scheduler=lr_scheduler,
          scaler=scaler, ema_model=ema_model)

    optimizer2, lr_scheduler2, scaler2, ema_model2 = _components(model, cfg)
    state = load_training_state(path, model=model, ema_model=ema_model2,
                                optimizer=optimizer2, lr_scheduler=lr_scheduler2,
                                scaler=scaler2, cfg=cfg)

    # epoch 4 completed -> the resumed loop starts at index 5 (i.e. "epoch 6/N")
    assert state["start_epoch"] == 5
    assert state["best_metric"] == pytest.approx(0.83)
    assert state["best_metric_epoch"] == 3
    assert state["not_improved_epoch"] == 1
    assert state["elapsed_sec"] == pytest.approx(1234.5)


def test_optimizer_moments_survive_the_roundtrip(tiny_unetr, tiny_unetr_kwargs, tmp_path):
    """Resuming with a fresh optimizer restarts AdamW's second-moment estimates
    from zero, which is a large effective-LR spike at the resume epoch. The
    moments have to come back."""
    from config import cfg
    model = tiny_unetr
    optimizer, lr_scheduler, scaler, ema_model = _components(model, cfg)

    # One real step so the optimizer has non-trivial exp_avg/exp_avg_sq state.
    x = torch.randn(1, tiny_unetr_kwargs["input_dim"], *tiny_unetr_kwargs["img_shape"])
    model.train()
    out = model(x)
    loss = (out[0] if isinstance(out, tuple) else out).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    path = str(tmp_path / "last.pth")
    _save(path, model, cfg, optimizer=optimizer, lr_scheduler=lr_scheduler,
          scaler=scaler, ema_model=ema_model)

    saved = optimizer.state_dict()
    optimizer2, lr_scheduler2, scaler2, ema_model2 = _components(model, cfg)
    assert not optimizer2.state_dict()["state"], "fresh optimizer should start empty"

    load_training_state(path, model=model, ema_model=ema_model2, optimizer=optimizer2,
                        lr_scheduler=lr_scheduler2, scaler=scaler2, cfg=cfg)

    restored = optimizer2.state_dict()["state"]
    assert restored, "optimizer moments were not restored"
    for pid, st in saved["state"].items():
        assert torch.allclose(st["exp_avg"], restored[pid]["exp_avg"])
        assert torch.allclose(st["exp_avg_sq"], restored[pid]["exp_avg_sq"])


def test_lr_schedule_resumes_at_the_same_point(tiny_unetr, tiny_unetr_kwargs, tmp_path):
    """A resume that rebuilds the scheduler from scratch restarts warmup and the
    cosine decay — the resumed half of the run would train at a completely
    different LR from the one the crashed run was on."""
    from config import cfg
    model = tiny_unetr
    optimizer, lr_scheduler, scaler, ema_model = _components(model, cfg)

    for _ in range(7):
        lr_scheduler.step()
    expected_lr = lr_scheduler.get_last_lr()[0]

    path = str(tmp_path / "last.pth")
    _save(path, model, cfg, optimizer=optimizer, lr_scheduler=lr_scheduler,
          scaler=scaler, ema_model=ema_model)

    optimizer2, lr_scheduler2, scaler2, ema_model2 = _components(model, cfg)
    assert lr_scheduler2.get_last_lr()[0] != pytest.approx(expected_lr)

    load_training_state(path, model=model, ema_model=ema_model2, optimizer=optimizer2,
                        lr_scheduler=lr_scheduler2, scaler=scaler2, cfg=cfg)
    assert lr_scheduler2.get_last_lr()[0] == pytest.approx(expected_lr)


def test_ema_weights_are_restored_not_reinitialised(tiny_unetr, tiny_unetr_kwargs, tmp_path):
    """The EMA copy IS the published model when cfg.ema_decay > 0. Losing it on
    resume silently restarts the averaging horizon."""
    from config import cfg
    if cfg.ema_decay <= 0:
        pytest.skip("EMA disabled in config")

    model = tiny_unetr
    optimizer, lr_scheduler, scaler, ema_model = _components(model, cfg)
    # Move the EMA away from the raw weights so a failure to restore is visible.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ema_model.update_parameters(model)
    reference = [p.detach().clone() for p in ema_model.module.parameters()]

    path = str(tmp_path / "last.pth")
    _save(path, model, cfg, optimizer=optimizer, lr_scheduler=lr_scheduler,
          scaler=scaler, ema_model=ema_model)

    optimizer2, lr_scheduler2, scaler2, ema_model2 = _components(model, cfg)
    load_training_state(path, model=model, ema_model=ema_model2, optimizer=optimizer2,
                        lr_scheduler=lr_scheduler2, scaler=scaler2, cfg=cfg)

    for want, got in zip(reference, ema_model2.module.parameters()):
        assert torch.allclose(want, got)


def test_estimator_counters_survive(tiny_unetr, tiny_unetr_kwargs, tmp_path):
    from config import cfg
    model = tiny_unetr
    optimizer, lr_scheduler, scaler, ema_model = _components(model, cfg)
    est = RunTimeEstimator(total_train_steps=100, total_val_steps=20)
    for _ in range(9):
        est.record_train_step(2.0)
    est.record_val_step(5.0)

    path = str(tmp_path / "last.pth")
    _save(path, model, cfg, optimizer=optimizer, lr_scheduler=lr_scheduler,
          scaler=scaler, ema_model=ema_model, estimator=est)

    est2 = RunTimeEstimator(total_train_steps=100, total_val_steps=20)
    optimizer2, lr_scheduler2, scaler2, ema_model2 = _components(model, cfg)
    load_training_state(path, model=model, ema_model=ema_model2, optimizer=optimizer2,
                        lr_scheduler=lr_scheduler2, scaler=scaler2, estimator=est2, cfg=cfg)

    assert est2.avg_train_step_time == pytest.approx(2.0)
    assert est2.avg_val_step_time == pytest.approx(5.0)


def test_img_shape_mismatch_refuses_to_resume(tiny_unetr, tiny_unetr_kwargs, tmp_path, monkeypatch):
    """Resuming into a changed geometry would mix two different experiments
    under one run name — and fail deep inside the positional embedding rather
    than at the point the mistake was made."""
    from config import cfg
    model = tiny_unetr
    optimizer, lr_scheduler, scaler, ema_model = _components(model, cfg)
    path = str(tmp_path / "last.pth")
    _save(path, model, cfg, optimizer=optimizer, lr_scheduler=lr_scheduler,
          scaler=scaler, ema_model=ema_model)

    bigger = tuple(d * 2 for d in cfg.unetr.img_shape)
    monkeypatch.setattr(cfg.unetr, "img_shape", bigger, raising=True)

    with pytest.raises(SystemExit, match="img_shape"):
        load_training_state(path, model=model, ema_model=ema_model, optimizer=optimizer,
                            lr_scheduler=lr_scheduler, scaler=scaler, cfg=cfg)


def test_save_is_atomic_and_leaves_no_tmp_file(tiny_unetr, tiny_unetr_kwargs, tmp_path):
    """A crash part-way through a 1.9GB torch.save would otherwise leave a
    truncated file exactly where the resume point is supposed to be, turning a
    one-epoch loss into a whole-run loss."""
    from config import cfg
    model = tiny_unetr
    optimizer, lr_scheduler, scaler, ema_model = _components(model, cfg)
    path = str(tmp_path / "last.pth")
    _save(path, model, cfg, optimizer=optimizer, lr_scheduler=lr_scheduler,
          scaler=scaler, ema_model=ema_model)

    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    assert peek_epoch(path) == 5


def test_rng_streams_are_restored(tiny_unetr, tiny_unetr_kwargs, tmp_path):
    """Without this the augmentation stream after a resume is different from the
    one the crashed run would have drawn — reproducibility of a resumed run
    would depend on where it happened to crash."""
    from config import cfg
    model = tiny_unetr
    optimizer, lr_scheduler, scaler, ema_model = _components(model, cfg)
    path = str(tmp_path / "last.pth")
    _save(path, model, cfg, optimizer=optimizer, lr_scheduler=lr_scheduler,
          scaler=scaler, ema_model=ema_model)

    expected_torch = torch.randn(4)
    expected_numpy = np.random.rand(4)

    load_training_state(path, model=model, ema_model=ema_model, optimizer=optimizer,
                        lr_scheduler=lr_scheduler, scaler=scaler, cfg=cfg)

    assert torch.allclose(torch.randn(4), expected_torch)
    assert np.allclose(np.random.rand(4), expected_numpy)

"""End-to-end resume test: run a few epochs through the REAL run_training loop,
then continue from the checkpoint it wrote and check the second half picks up
exactly where the first left off.

Unit tests on utils/checkpoint.py cover the round-trip of each piece. What they
cannot cover is the wiring — that run_training actually writes last.pth, that
RunLogger reuses the run folder instead of minting logs/<name>_1, that
metrics.csv ends up with one row per epoch across the seam, and that the epoch
counter neither repeats nor skips. Those are the failure modes that only show up
when you actually need the resume, i.e. after a 33-epoch run has already died.
"""
import csv
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from utils.engine import run_training
from utils.losses import build_loss_fn
from utils.run_logger import RunLogger

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="engine.py hardcodes cuda autocast — needs a CUDA device to exercise",
)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n, input_dim, output_dim, shape):
        self.n, self.input_dim, self.output_dim, self.shape = n, input_dim, output_dim, shape

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        return {
            "image": torch.rand(self.input_dim, *self.shape, generator=g),
            "label": (torch.rand(self.output_dim, *self.shape, generator=g) > 0.7).float(),
        }


def _loaders(tiny_unetr_kwargs):
    ds = _TinyDataset(2, tiny_unetr_kwargs["input_dim"], tiny_unetr_kwargs["output_dim"],
                      tiny_unetr_kwargs["img_shape"])
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    return {"train_loader": loader, "train_ds": ds, "val_loader": loader, "val_ds": ds}


def _read_epochs(path):
    with open(path, newline="") as f:
        return [int(float(r["epoch"])) for r in csv.DictReader(f)]


@requires_cuda
def test_run_training_resumes_and_metrics_csv_stays_continuous(
        tiny_unetr, tiny_unetr_kwargs, tmp_path, monkeypatch, device):
    from config import cfg
    monkeypatch.setattr(cfg, "epoch", 2, raising=True)
    monkeypatch.setattr(cfg, "warmup_epochs", 1, raising=True)
    monkeypatch.setattr(cfg.attention, "enabled", False, raising=True)
    monkeypatch.setattr(cfg.checkpoint, "gc_every_n_val_steps", 0, raising=True)

    model = tiny_unetr.to(device)
    loaders = _loaders(tiny_unetr_kwargs)
    loss_fn = build_loss_fn(cfg)

    # --- first half: 2 epochs, then "crash" ---
    with RunLogger(run_name="resume-it", base_dir=str(tmp_path)) as logger:
        run_training(model, loaders, loss_fn, device, cfg, logger)
        first_dir = logger.run_dir
        last_ckpt = logger.last_checkpoint_path
        metrics_csv = logger.metrics_csv_path

    assert os.path.exists(last_ckpt), "run_training must write last.pth every epoch"
    assert _read_epochs(metrics_csv) == [1, 2]

    # --- second half: budget raised to 4, resume ---
    monkeypatch.setattr(cfg, "epoch", 4, raising=True)
    with RunLogger(run_name="resume-it", base_dir=str(tmp_path), resume=True) as logger:
        assert logger.run_dir == first_dir, "a resume must reuse the run folder"
        best, best_epoch, _ = run_training(model, loaders, loss_fn, device, cfg, logger,
                                           resume_from=last_ckpt)

    # No repeated and no skipped epochs across the seam.
    assert _read_epochs(metrics_csv) == [1, 2, 3, 4]
    assert 1 <= best_epoch <= 4
    assert best > -1


@requires_cuda
def test_resume_at_the_epoch_budget_is_a_noop(
        tiny_unetr, tiny_unetr_kwargs, tmp_path, monkeypatch, device):
    """A supervisor that relaunches after a clean finish must not train a second
    time — and must not truncate the finished run's metrics.csv on the way."""
    from config import cfg
    monkeypatch.setattr(cfg, "epoch", 2, raising=True)
    monkeypatch.setattr(cfg, "warmup_epochs", 1, raising=True)
    monkeypatch.setattr(cfg.attention, "enabled", False, raising=True)
    monkeypatch.setattr(cfg.checkpoint, "gc_every_n_val_steps", 0, raising=True)

    model = tiny_unetr.to(device)
    loaders = _loaders(tiny_unetr_kwargs)
    loss_fn = build_loss_fn(cfg)

    with RunLogger(run_name="done-run", base_dir=str(tmp_path)) as logger:
        run_training(model, loaders, loss_fn, device, cfg, logger)
        last_ckpt = logger.last_checkpoint_path
        metrics_csv = logger.metrics_csv_path

    assert _read_epochs(metrics_csv) == [1, 2]

    with RunLogger(run_name="done-run", base_dir=str(tmp_path), resume=True) as logger:
        run_training(model, loaders, loss_fn, device, cfg, logger, resume_from=last_ckpt)

    assert _read_epochs(metrics_csv) == [1, 2]


def test_open_metrics_csv_drops_rows_past_the_resume_point(tmp_path):
    """An epoch whose row was written but whose checkpoint never landed gets
    re-run. Its stale row has to go, or the CSV holds that epoch twice and the
    curves in plots/ double back on themselves."""
    fields = ["epoch", "train_loss"]
    logger = RunLogger(run_name="partial", base_dir=str(tmp_path), resume=True)
    logger.open_metrics_csv(fields)
    for e in range(1, 6):
        logger.log_epoch_metrics({"epoch": e, "train_loss": 0.1 * e})
    logger._metrics_file.close()

    # Checkpoint only made it to epoch 3 -> rows 4 and 5 must be discarded.
    logger2 = RunLogger(run_name="partial", base_dir=str(tmp_path), resume=True)
    assert logger2.run_dir == logger.run_dir
    logger2.open_metrics_csv(fields, resume_from_epoch=3)
    logger2.log_epoch_metrics({"epoch": 4, "train_loss": 0.99})
    logger2._metrics_file.close()

    assert _read_epochs(logger.metrics_csv_path) == [1, 2, 3, 4]


def test_run_logger_logs_the_traceback_into_log_txt(tmp_path):
    """v2-run2's OOM left log.txt ending on a tidy "Run finished" with no error
    in it: __exit__ restored sys.stderr before Python printed the traceback, so
    the crash that killed a 33-epoch run went to the terminal and nowhere else."""
    with pytest.raises(RuntimeError):
        with RunLogger(run_name="boom", base_dir=str(tmp_path)) as logger:
            log_path = logger.log_path
            raise RuntimeError("simulated CUDA out of memory")

    text = open(log_path).read()
    assert "RUN FAILED: RuntimeError" in text
    assert "simulated CUDA out of memory" in text
    assert "Traceback" in text

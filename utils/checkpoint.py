"""Full training-state checkpoints, so a crash costs one epoch instead of a run.

There are two different things a run saves, and conflating them is how resume
logic usually goes wrong:

  * ``best_metric_model.pth`` — WEIGHTS ONLY, the EMA copy when EMA is on. This
    is what evaluate.py / xai.py / run_test load and what gets published. It is
    rewritten only when validation mean Dice improves, so it must NEVER be
    overwritten by a routine per-epoch save.
  * ``last.pth`` — the whole training state: raw weights, EMA weights, optimizer
    moments, LR-scheduler position, GradScaler scale, RNG streams, the
    best-metric bookkeeping and the ETA counters. Rewritten every epoch. Never
    used for reporting.

Resuming from anything less than that full state is a silent quality
regression, not a convenience: dropping the optimizer moments restarts AdamW's
second-moment estimates from zero (a large effective LR spike at the resume
epoch), and dropping the scheduler position restarts the cosine decay.

Writes are atomic (tmp file + ``os.replace``). A 625MB checkpoint takes real
wall time to write; a crash or Ctrl-C part-way through a plain ``torch.save``
leaves a truncated file where the resume point used to be, which turns a
one-epoch loss into a whole-run loss.
"""
import os
import random

import numpy as np
import torch

# Bumped when the dict layout changes in a way older checkpoints can't satisfy.
CHECKPOINT_FORMAT = 1


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    """Restore the RNG streams. Best-effort per stream: a checkpoint moved to a
    box with a different GPU count cannot restore CUDA RNG, and refusing to
    resume over that would be worse than resuming with a fresh augmentation
    stream."""
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"])
                            else state["torch"])
    except Exception as exc:
        print(f"[checkpoint] could not restore CPU RNG state: {exc}")
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            if len(cuda_state) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_state)
            else:
                print(f"[checkpoint] GPU count changed "
                      f"({len(cuda_state)} -> {torch.cuda.device_count()}), "
                      f"skipping CUDA RNG restore")
        except Exception as exc:
            print(f"[checkpoint] could not restore CUDA RNG state: {exc}")


def save_training_state(path, *, epoch, model, ema_model, optimizer, lr_scheduler,
                        scaler, best_metric, best_metric_epoch, not_improved_epoch,
                        estimator=None, elapsed_sec=0.0, cfg=None):
    """Atomically write the full training state. `epoch` is the 0-based index of
    the epoch that just COMPLETED; the resume starts at `epoch + 1`."""
    state = {
        "format": CHECKPOINT_FORMAT,
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "not_improved_epoch": not_improved_epoch,
        "elapsed_sec": elapsed_sec,
        "rng": _rng_state(),
        # Shape/epoch budget are checked on load: resuming a (128,128,96) run
        # into a config that now says (96,96,96) would load positional
        # embeddings of the wrong length and fail deep inside the model.
        "img_shape": tuple(cfg.unetr.img_shape) if cfg is not None else None,
        "encoder": cfg.unetr.get("encoder", "vit") if cfg is not None else None,
        "total_epochs": cfg.epoch if cfg is not None else None,
        "estimator": ({
            "train_steps_done": estimator._train_steps_done,
            "train_time_sum": estimator._train_time_sum,
            "val_steps_done": estimator._val_steps_done,
            "val_time_sum": estimator._val_time_sum,
        } if estimator is not None else None),
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic on POSIX: readers see old or new, never half


def load_training_state(path, *, model, ema_model, optimizer, lr_scheduler, scaler,
                        estimator=None, cfg=None, map_location="cpu"):
    """Restore everything `save_training_state` wrote and return the resume
    bookkeeping. Raises if the checkpoint describes a different model geometry —
    silently resuming into a mismatched config produces a run whose first half
    and second half are not the same experiment.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)

    if state.get("format") != CHECKPOINT_FORMAT:
        print(f"[checkpoint] format {state.get('format')} != {CHECKPOINT_FORMAT}, "
              f"loading best-effort")

    if cfg is not None and state.get("img_shape") is not None:
        if tuple(state["img_shape"]) != tuple(cfg.unetr.img_shape):
            raise SystemExit(
                f"Checkpoint was trained at img_shape={tuple(state['img_shape'])} but "
                f"config.py now says {tuple(cfg.unetr.img_shape)}. Resuming across that "
                f"change would mix two different experiments — start a new run instead."
            )

    if cfg is not None and state.get("encoder") is not None:
        current = cfg.unetr.get("encoder", "vit")
        if state["encoder"] != current:
            raise SystemExit(
                f"Checkpoint was trained with encoder={state['encoder']!r} but the config "
                f"(or --encoder) now says {current!r}. Resume with the same encoder, or "
                f"start a new run."
            )

    model.load_state_dict(state["model"])
    if ema_model is not None and state.get("ema") is not None:
        ema_model.load_state_dict(state["ema"])
    elif ema_model is not None:
        print("[checkpoint] no EMA weights in checkpoint; EMA restarts from current weights")

    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    if estimator is not None and state.get("estimator"):
        est = state["estimator"]
        estimator._train_steps_done = est["train_steps_done"]
        estimator._train_time_sum = est["train_time_sum"]
        estimator._val_steps_done = est["val_steps_done"]
        estimator._val_time_sum = est["val_time_sum"]

    _restore_rng_state(state.get("rng"))

    if cfg is not None and state.get("total_epochs") not in (None, cfg.epoch):
        print(f"[checkpoint] epoch budget changed {state['total_epochs']} -> {cfg.epoch}; "
              f"the LR schedule was built for the old value and is restored as-is")

    return {
        "start_epoch": state["epoch"] + 1,
        "best_metric": state["best_metric"],
        "best_metric_epoch": state["best_metric_epoch"],
        "not_improved_epoch": state["not_improved_epoch"],
        "elapsed_sec": state.get("elapsed_sec", 0.0),
    }


def peek_epoch(path):
    """Read just the resume epoch without materialising the weights — used by
    the CLI to decide whether a `--auto-resume` run has anything to resume."""
    try:
        state = torch.load(path, map_location="meta", weights_only=False)
        return state.get("epoch", -1) + 1
    except Exception:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            return state.get("epoch", -1) + 1
        except Exception:
            return None

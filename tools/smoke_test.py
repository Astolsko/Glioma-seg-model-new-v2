"""Real-data smoke test: every stage of a training run, for a few steps.

    conda activate mamba
    python tools/smoke_test.py                  # the encoder config.py selects
    python tools/smoke_test.py --encoder vit    # e.g. the ViT, for its timing

Runs the same functions train.py runs, on real BraTS volumes:
  1. build_model / build_dataloaders / build_loss_fn / build_training_components;
  2. train_one_epoch over --train-steps real batches (bf16, deep supervision,
     full loss with the Hausdorff term at full weight, AdamW + EMA update),
     after 2 warm-up batches;
  3. validate() over --val-samples real volumes on an attention epoch, so the
     training-time attention / hidden-attention overlay runs as well;
  4. one flip-TTA sliding-window inference (threshold tuning + the test pass);
  5. the whole XAI suite, all five components, on one test sample.
Prints peak GPU memory per phase and projects min/epoch and total hours for
cfg.epoch epochs. Nothing is written under logs/ (temp dirs only).
The weights are untrained, so the metrics it prints mean nothing; what it
checks is that every code path runs and how long and how much memory it takes.
"""
import argparse
import itertools
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils.env_check import configure_cuda_allocator  # noqa: E402
from config import cfg  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", choices=("vit", "mamba"), default=None)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--val-samples", type=int, default=3)
    args = parser.parse_args()
    if args.encoder:
        cfg.unetr.encoder = args.encoder
    configure_cuda_allocator(cfg.checkpoint.cuda_alloc_conf)

    import torch
    from utils import xai
    from utils.attention import register_attention_hook
    from utils.dataloader import build_dataloaders
    from utils.engine import (build_model, build_training_components, get_device,
                              run_inference, train_one_epoch, validate)
    from utils.losses import build_loss_fn

    def sync():
        torch.cuda.synchronize()

    def peak_gb():
        return torch.cuda.max_memory_allocated() / 2 ** 30

    torch.manual_seed(cfg.seed)
    device = get_device()
    model = build_model(cfg, device)
    loaders = build_dataloaders(cfg)
    loss_fn = build_loss_fn(cfg)
    if hasattr(loss_fn, "set_epoch"):
        loss_fn.set_epoch(cfg.epoch - 1, cfg.epoch)       # Hausdorff term at full weight
    (optimizer, _, dice_metric, dice_metric_batch, post_trans, scaler,
     ema_model) = build_training_components(model, cfg)
    eval_model = model if ema_model is None else ema_model.module
    report = {}

    # ---- 1. training steps ------------------------------------------------
    train_iter = iter(loaders["train_loader"])
    warmup = [next(train_iter) for _ in range(2)]
    t0 = time.time()
    timed = [next(train_iter) for _ in range(args.train_steps)]
    load_s = (time.time() - t0) / args.train_steps
    train_one_epoch(model, warmup, loaders["train_ds"], optimizer, scaler, loss_fn,
                    device, cfg, 0, ema_model=ema_model)
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.time()
    train_loss = train_one_epoch(model, timed, loaders["train_ds"], optimizer, scaler,
                                 loss_fn, device, cfg, 0, ema_model=ema_model)
    sync()
    report["train_s"] = (time.time() - t0) / args.train_steps
    report["train_peak"] = peak_gb()
    report["load_s"] = load_s
    print(f"\n[1/5] train: {args.train_steps} steps | loss={train_loss:.4f} | "
          f"{report['train_s']:.2f}s/step compute, {load_s:.2f}s/step data | "
          f"peak {report['train_peak']:.1f} GB")
    del timed, warmup
    torch.cuda.empty_cache()

    # ---- 2. validation + attention overlay ---------------------------------
    val_batches = list(itertools.islice(loaders["val_loader"], args.val_samples))
    attention_cache = register_attention_hook(eval_model) if cfg.attention.enabled else {}
    attention_dir = tempfile.mkdtemp(prefix="smoke_attention_")
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.time()
    validate(eval_model, val_batches, loaders["val_ds"], loss_fn, dice_metric,
             dice_metric_batch, post_trans, device, cfg,
             epoch=cfg.attention.every_n_epochs - 1,
             attention_cache=attention_cache, attention_dir=attention_dir)
    sync()
    report["val_s"] = (time.time() - t0) / len(val_batches)
    report["val_peak"] = peak_gb()
    overlays = os.listdir(attention_dir)
    print(f"[2/5] validate: {len(val_batches)} volumes | {report['val_s']:.1f}s/volume | "
          f"peak {report['val_peak']:.1f} GB | attention overlay: "
          f"{overlays if overlays else 'NOT WRITTEN'}")

    # ---- 3. TTA inference ---------------------------------------------------
    test_batch = next(iter(loaders["test_loader"]))
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.time()
    with torch.no_grad():
        logits = run_inference(eval_model.eval(), test_batch["image"].to(device), cfg, tta=True)
    sync()
    report["tta_s"] = time.time() - t0
    print(f"[3/5] 8-flip TTA inference on {tuple(test_batch['image'].shape[2:])}: "
          f"{report['tta_s']:.1f}s | finite={bool(torch.isfinite(logits).all())} | "
          f"peak {peak_gb():.1f} GB")
    del logits

    # ---- 4. XAI suite ---------------------------------------------------------
    cfg.xai.sample_indices = list(cfg.xai.sample_indices)[:1]
    xai_dir = tempfile.mkdtemp(prefix="smoke_xai_")
    checkpoint = os.path.join(xai_dir, "model.pth")
    torch.save(eval_model.state_dict(), checkpoint)
    t0 = time.time()
    summary = xai.run_xai_suite(
        model=eval_model, test_ds=loaders["test_ds"],
        test_loader=list(itertools.islice(loaders["test_loader"], 1)),
        checkpoint_path=checkpoint, out_dir=xai_dir, device=device, cfg=cfg,
        inferer=lambda m, x: run_inference(m, x, cfg, tta=False),
        components=xai.COMPONENTS,
    )
    report["xai_s"] = time.time() - t0
    failed = {k: v["error"] for k, v in summary.items() if isinstance(v, dict) and "error" in v}
    print(f"[4/5] XAI suite ({', '.join(xai.COMPONENTS)}) on 1 sample: "
          f"{report['xai_s']:.0f}s | failed: {failed if failed else 'none'} | "
          f"{len(os.listdir(xai_dir))} files in {xai_dir}")

    # ---- 5. projection ------------------------------------------------------
    n_train, n_val = len(loaders["train_loader"]), len(loaders["val_loader"])
    n_test = len(loaders["test_loader"])
    step_s = max(report["train_s"], report["load_s"])
    epoch_min = (step_s * n_train + report["val_s"] * n_val) / 60
    train_h = epoch_min * cfg.epoch / 60
    after_h = (n_val + n_test) * report["tta_s"] / 3600          # tune on val + test pass
    print(f"[5/5] projection for encoder={cfg.unetr.encoder!r}, {cfg.epoch} epochs, "
          f"{n_train} train / {n_val} val / {n_test} test volumes:")
    print(f"      ~{epoch_min:.0f} min/epoch  ->  ~{train_h:.1f} h training, "
          f"+ ~{after_h:.1f} h threshold tuning and TTA test pass, + XAI")
    print("      (reference: the ViT run logs/v2-run3 took 41-45 min/epoch)")
    if failed or not overlays:
        raise SystemExit("SMOKE TEST FAILED")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

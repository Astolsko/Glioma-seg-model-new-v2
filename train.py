"""Entry point for training + evaluating the UNETR glioma segmentation
model. Run with:

    python train.py [--name RUN_NAME]
    python train.py --name RUN_NAME --resume        # continue a crashed run
    python train.py --name RUN_NAME --auto-resume   # continue if possible, else start
    python train.py --name RUN_NAME --encoder vit   # pick the encoder (default: config.py)

If --name isn't given you'll be prompted for a run name interactively; it
names the folder under logs/ that holds this run's config snapshot,
per-epoch metrics.csv, plots, visualizations, attention maps, checkpoint,
final test-set results (logs/<name>/testing/), the tuned inference recipe
(logs/<name>/eval/) and the explainability outputs (logs/<name>/xai/).

One command produces the whole artifact set: train -> tune thresholds on val
-> test -> explain. The last two stages can also be re-run standalone against
a saved checkpoint via evaluate.py and xai.py.

Every epoch writes logs/<name>/checkpoints/last.pth — the full training state,
not just weights — so an interrupted run continues from the epoch after the
last completed one. `--auto-resume` is the form to use under a supervisor
(tools/train_supervisor.sh): it resumes when there is something to resume from
and starts fresh when there isn't, so the same command works for both the first
launch and every restart after it.
"""
import argparse
import os

# Must run before any module that imports these packages at module scope.
from utils.env_check import configure_cuda_allocator, ensure_dependencies
ensure_dependencies()

from config import cfg
from utils.dataloader import build_dataloaders
from utils.engine import (
    print_gpu_info, get_device, build_model, run_training, run_test,
    run_inference,
)
from utils.losses import build_loss_fn
from utils.plot import (
    plot_data_distribution, plot_sample_modalities, plot_sample_labels,
    plot_indexed_samples, plot_metrics_from_csv,
)
from utils.postprocess import save_infer_config, tune_and_save
from utils.run_logger import RunLogger
from utils import xai


def print_batch_shapes(train_loader, val_loader):
    train_batch = next(iter(train_loader))
    print("TRAIN")
    print("Image shape :", train_batch["image"].shape)
    print("Label shape :", train_batch["label"].shape)

    val_batch = next(iter(val_loader))
    print("\nVALIDATION")
    print("Image shape :", val_batch["image"].shape)
    print("Label shape :", val_batch["label"].shape)


def resolve_resume(args, cfg):
    """Decide which checkpoint (if any) this launch continues from.

    Returns (resume_path_or_None, reuse_existing_run_dir). The second value is
    what stops a resume from minting logs/<name>_1 and splitting one run's
    artifacts across two folders.
    """
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise SystemExit(f"No checkpoint at {args.resume_from}")
        return args.resume_from, True

    if not (args.resume or args.auto_resume):
        return None, False

    if not args.name:
        raise SystemExit("--resume/--auto-resume need --name to know which run to continue")

    last = os.path.join(cfg.paths.logs_dir, args.name, "checkpoints", "last.pth")
    if os.path.exists(last):
        return last, True

    if args.resume:
        raise SystemExit(
            f"No checkpoint at {last}. Use --auto-resume to start a fresh run when "
            f"there is nothing to resume from."
        )
    # --auto-resume, first launch: nothing to continue, so start the run. Reuse
    # the folder if it already exists (an earlier attempt that died before its
    # first epoch finished) rather than leaving a trail of logs/<name>_N.
    print(f"--auto-resume: no checkpoint at {last}, starting a fresh run")
    return None, os.path.isdir(os.path.join(cfg.paths.logs_dir, args.name))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=None, help="Run name (logs folder); prompted if omitted")
    parser.add_argument("--resume", action="store_true",
                        help="Continue logs/<name> from its checkpoints/last.pth; "
                             "fails if there isn't one")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Continue from checkpoints/last.pth if it exists, otherwise "
                             "start a fresh run. The form to use under a restart supervisor.")
    parser.add_argument("--resume-from", default=None, metavar="PATH",
                        help="Continue from an explicit last.pth (e.g. another run's)")
    parser.add_argument("--encoder", choices=("vit", "mamba"), default=None,
                        help="Override cfg.unetr.encoder for this launch "
                             "(recorded in the run's config_snapshot.json)")
    args = parser.parse_args()
    if args.encoder:
        cfg.unetr.encoder = args.encoder

    # Before ANY CUDA allocation — the allocator reads this once at init.
    configure_cuda_allocator(cfg.checkpoint.cuda_alloc_conf)

    resume_from, reuse_dir = resolve_resume(args, cfg)

    with RunLogger(run_name=args.name, base_dir=cfg.paths.logs_dir,
                   resume=reuse_dir) as run_logger:
        run_logger.save_config_snapshot(cfg)

        print_gpu_info()
        device = get_device()

        model = build_model(cfg, device)

        loaders = build_dataloaders(cfg)
        train_ds, val_ds = loaders["train_ds"], loaders["val_ds"]

        # --- qualitative data checks, carried forward from the notebook ---
        sample = val_ds[cfg.visualization.sample_index]
        print(f"image shape: {sample['image'].shape}")
        plot_sample_modalities(sample, f"{run_logger.vis_dir}/sample_modalities.png")
        print(f"label shape: {sample['label'].shape}")
        plot_sample_labels(sample, f"{run_logger.vis_dir}/sample_labels.png")
        plot_indexed_samples(val_ds, cfg.visualization.extra_indices, run_logger.vis_dir)

        val_frac, test_frac = val_ds.val_frac, val_ds.test_frac
        num_train, num_val = len(train_ds), len(val_ds)
        num_test = int(test_frac * num_val / val_frac)
        plot_data_distribution(num_train, num_val, num_test,
                                f"{run_logger.plots_dir}/data_distribution.png")

        print_batch_shapes(loaders["train_loader"], loaders["val_loader"])

        loss_fn = build_loss_fn(cfg)

        run_training(model, loaders, loss_fn, device, cfg, run_logger,
                     resume_from=resume_from)
        plot_metrics_from_csv(run_logger.metrics_csv_path, run_logger.plots_dir)

        if cfg.infer.tune_thresholds_after_training:
            tune_and_save(
                model, loaders["val_loader"], device, cfg,
                run_logger.checkpoint_path, run_logger.eval_dir,
                inferer=lambda m, x: run_inference(m, x, cfg),
                console=run_logger.console,
            )
        save_infer_config(cfg, os.path.join(run_logger.eval_dir, "infer_config.json"))

        run_test(model, loaders, loss_fn, device, cfg, run_logger)

        if cfg.xai.run_after_training:
            print("\n=== Explainability suite ===")
            xai.run_xai_suite(
                model=model,
                test_ds=loaders["test_ds"],
                test_loader=loaders["test_loader"],
                checkpoint_path=run_logger.checkpoint_path,
                out_dir=run_logger.xai_dir,
                device=device,
                cfg=cfg,
                # Modality ablation compares arms that all get the same
                # treatment, so TTA would multiply its cost 8x to move no
                # conclusion.
                inferer=lambda m, x: run_inference(m, x, cfg, tta=False),
                components=cfg.xai.components,
            )


if __name__ == "__main__":
    main()

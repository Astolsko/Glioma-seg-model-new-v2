"""Entry point for training + evaluating the UNETR glioma segmentation
model. Run with:

    python train.py [--name RUN_NAME]

If --name isn't given you'll be prompted for a run name interactively; it
names the folder under logs/ that holds this run's config snapshot,
per-epoch metrics.csv, plots, visualizations, attention maps, checkpoint,
final test-set results (logs/<name>/testing/), the tuned inference recipe
(logs/<name>/eval/) and the explainability outputs (logs/<name>/xai/).

One command produces the whole artifact set: train -> tune thresholds on val
-> test -> explain. The last two stages can also be re-run standalone against
a saved checkpoint via evaluate.py and xai.py.
"""
import argparse
import os

# Must run before any module that imports these packages at module scope.
from utils.env_check import ensure_dependencies
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=None, help="Run name (logs folder); prompted if omitted")
    args = parser.parse_args()

    with RunLogger(run_name=args.name, base_dir=cfg.paths.logs_dir) as run_logger:
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

        run_training(model, loaders, loss_fn, device, cfg, run_logger)
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

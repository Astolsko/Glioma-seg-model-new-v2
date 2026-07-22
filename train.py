"""Entry point for training + evaluating the UNETR glioma segmentation
model. Run with:

    python train.py [--name RUN_NAME]

If --name isn't given you'll be prompted for a run name interactively; it
names the folder under logs/ that holds this run's config snapshot,
per-epoch metrics.csv, plots, visualizations, attention maps, checkpoint,
and final test-set results (under logs/<name>/testing/).
"""
import argparse

# Must run before any module that imports these packages at module scope.
from utils.env_check import ensure_dependencies
ensure_dependencies()

from config import cfg
from utils.dataloader import build_dataloaders
from utils.engine import (
    print_gpu_info, get_device, build_model, run_training, run_test,
)
from utils.losses import build_loss_fn
from utils.plot import (
    plot_data_distribution, plot_sample_modalities, plot_sample_labels,
    plot_indexed_samples, plot_metrics_from_csv,
)
from utils.run_logger import RunLogger


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

        run_test(model, loaders, loss_fn, device, cfg, run_logger)


if __name__ == "__main__":
    main()

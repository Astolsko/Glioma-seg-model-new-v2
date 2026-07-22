"""Smoke tests for utils/plot.py — every plotting call here runs mid-training
or right after it (data checks, per-epoch curves, qualitative test outputs).
A crash in any of them currently means losing everything that ran before it
in the same script since train.py doesn't isolate these calls. Not checking
pixel content, just that each function completes and writes the file(s) it
promises to.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")

import pytest

torch = pytest.importorskip("torch")

from utils.plot import (
    plot_data_distribution, plot_sample_modalities, plot_sample_labels,
    plot_indexed_samples, plot_test_qualitative, plot_metrics_from_csv,
)


@pytest.fixture
def fake_sample():
    return {
        "image": torch.rand(4, 8, 8, 6),
        "label": torch.rand(3, 8, 8, 6),
    }


class _ListDataset:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def test_plot_data_distribution_creates_file(tmp_path):
    out_path = str(tmp_path / "dist.png")
    plot_data_distribution(10, 3, 2, out_path)
    assert os.path.exists(out_path)


def test_plot_sample_modalities_creates_file(tmp_path, fake_sample):
    out_path = str(tmp_path / "modalities.png")
    plot_sample_modalities(fake_sample, out_path)
    assert os.path.exists(out_path)


def test_plot_sample_labels_creates_file(tmp_path, fake_sample):
    out_path = str(tmp_path / "labels.png")
    plot_sample_labels(fake_sample, out_path)
    assert os.path.exists(out_path)


def test_plot_indexed_samples_creates_one_file_per_index(tmp_path, fake_sample):
    dataset = _ListDataset([fake_sample, fake_sample, fake_sample])
    out_dir = str(tmp_path / "indexed")
    plot_indexed_samples(dataset, [0, 2], out_dir)
    assert os.path.exists(os.path.join(out_dir, "sample_0.png"))
    assert os.path.exists(os.path.join(out_dir, "sample_2.png"))
    assert not os.path.exists(os.path.join(out_dir, "sample_1.png"))


def test_plot_test_qualitative_creates_three_files(tmp_path, fake_sample):
    out_dir = str(tmp_path / "testing_vis")
    val_output = torch.rand(3, 8, 8, 6)
    plot_test_qualitative(fake_sample, val_output, out_dir)
    assert os.path.exists(os.path.join(out_dir, "test_sample_modalities.png"))
    assert os.path.exists(os.path.join(out_dir, "test_sample_labels.png"))
    assert os.path.exists(os.path.join(out_dir, "test_sample_outputs.png"))


def _write_metrics_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_plot_metrics_from_csv_creates_expected_plots(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "mean_dice", "dice_et", "dice_wt",
                  "dice_tc", "mean_hd95", "mean_iou"]
    rows = [
        {"epoch": e, "train_loss": 1.0 / e, "val_loss": 1.2 / e,
         "mean_dice": 0.5 + 0.01 * e, "dice_et": 0.4, "dice_wt": 0.6, "dice_tc": 0.5,
         "mean_hd95": 10.0 / e, "mean_iou": 0.3 + 0.01 * e}
        for e in range(1, 4)
    ]
    _write_metrics_csv(csv_path, fieldnames, rows)

    plots_dir = str(tmp_path / "plots")
    plot_metrics_from_csv(str(csv_path), plots_dir)

    for fname in ("loss.png", "dice.png", "hd95.png", "iou.png"):
        assert os.path.exists(os.path.join(plots_dir, fname))


def test_plot_metrics_from_csv_handles_empty_csv_without_raising(tmp_path):
    csv_path = tmp_path / "empty.csv"
    _write_metrics_csv(csv_path, ["epoch"], [])
    plots_dir = str(tmp_path / "plots")

    plot_metrics_from_csv(str(csv_path), plots_dir)  # should not raise

    assert not os.path.exists(plots_dir)


def test_plot_metrics_from_csv_handles_missing_or_blank_values(tmp_path):
    """log_epoch_metrics() writes a blank string for any fieldname missing
    from a given epoch's row (e.g. an epoch where validation was skipped) —
    the plotting code must tolerate that instead of crashing on float('')."""
    csv_path = tmp_path / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "mean_dice", "dice_et", "dice_wt",
                  "dice_tc", "mean_hd95", "mean_iou"]
    rows = [
        {"epoch": 1, "train_loss": 1.0},  # val fields blank — validation was skipped
        {"epoch": 2, "train_loss": 0.8, "val_loss": 0.9, "mean_dice": 0.5,
         "dice_et": 0.4, "dice_wt": 0.6, "dice_tc": 0.5, "mean_hd95": 5.0, "mean_iou": 0.4},
    ]
    _write_metrics_csv(csv_path, fieldnames, rows)

    plots_dir = str(tmp_path / "plots")
    plot_metrics_from_csv(str(csv_path), plots_dir)  # should not raise

    assert os.path.exists(os.path.join(plots_dir, "loss.png"))

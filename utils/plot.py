import csv
import os

import matplotlib as mpl
# Force the non-interactive Agg backend BEFORE importing pyplot. We only ever
# save figures to disk (no GUI), and if matplotlib picks an interactive backend
# (TkAgg, which it does when $DISPLAY is set under X forwarding), Tk objects get
# garbage-collected off the main thread once DataLoader workers/threads spawn and
# abort the whole process with "Tcl_AsyncDelete: async handler deleted by the
# wrong thread" -> core dump. Agg has no such objects.
mpl.use("Agg")
from matplotlib import pyplot as plt


def _mid_slice(volume):
    vol = volume
    if hasattr(vol, "detach"):
        vol = vol.detach().cpu().numpy()
    slice_2d = vol.take(vol.shape[-1] // 2, axis=-1)
    if slice_2d.ndim == 3:
        slice_2d = slice_2d[slice_2d.shape[0] // 2]
    return slice_2d


def plot_data_distribution(num_train: int, num_val: int, num_test: int, out_path: str):
    """Plot number of data for train-set, val-set, and test-set after splitted"""
    bars = plt.bar(["Train", "Val", "Test"],
                    [num_train, num_val, num_test], align='center', color=['green', 'red', 'blue'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, yval, ha='center', va='bottom')

    plt.ylabel('Number of images')
    plt.title('Data distribution')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_sample_modalities(sample, out_path, image_channels=('FLAIR', 'T1w', 'T1gd', 'T2w')):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor('black')
    for i, ax in enumerate(axes):
        ax.set_facecolor('black')
        slc = _mid_slice(sample["image"][i])
        ax.imshow(slc, cmap="gray")
        ax.set_title(f"{image_channels[i]}", color='white', weight='bold', fontsize=13)
        ax.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, bbox_inches="tight", facecolor='black', dpi=150)
    plt.close()


def plot_sample_labels(sample, out_path, label_channels=('Tumor Core', 'Whole Tumor', 'Enhancing Tumor')):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    label_colors = ['Reds', 'Greens', 'Blues']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('black')
    for i, ax in enumerate(axes):
        ax.set_facecolor('black')
        slc = _mid_slice(sample["label"][i])
        ax.imshow(slc, cmap=label_colors[i], vmin=0, vmax=1)
        ax.set_title(f"{label_channels[i]}", color='white', weight='bold', fontsize=13)
        ax.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, bbox_inches="tight", facecolor='black', dpi=150)
    plt.close()


def plot_indexed_samples(dataset, indices, out_dir, image_channels=('FLAIR', 'T1w', 'T1gd', 'T2w')):
    os.makedirs(out_dir, exist_ok=True)
    for idx in indices:
        sample = dataset[idx]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.patch.set_facecolor('black')
        for i, ax in enumerate(axes):
            ax.set_facecolor('black')
            slc = _mid_slice(sample["image"][i])
            ax.imshow(slc, cmap="gray")
            ax.set_title(f"{image_channels[i]}", color='white', weight='bold', fontsize=11)
            ax.axis('off')
        plt.suptitle(f"Patient index {idx}", color='white', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{idx}.png"),
                    bbox_inches="tight", facecolor='black', dpi=150)
        plt.close()
        print(f"Saved index {idx}")


def plot_test_qualitative(val_data, val_output, out_dir):
    """Save the qualitative image/label/output triptych for one test sample."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(_mid_slice(val_data["image"][i]), cmap="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_sample_modalities.png"), bbox_inches="tight")
    plt.close()

    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(_mid_slice(val_data["label"][i]))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_sample_labels.png"), bbox_inches="tight")
    plt.close()

    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"output channel {i}")
        plt.imshow(_mid_slice(val_output[i]))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_sample_outputs.png"), bbox_inches="tight")
    plt.close()


def _read_metrics_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _as_float_series(rows, key):
    out = []
    for row in rows:
        val = row.get(key, "")
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def plot_metrics_from_csv(csv_path, plots_dir):
    """Read the per-epoch metrics.csv written during training and save the
    same loss/dice/hd95/iou plots the original notebook produced."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })

    rows = _read_metrics_csv(csv_path)
    if not rows:
        print(f"[Plot] no rows found in {csv_path}, skipping metric plots")
        return

    epochs = _as_float_series(rows, "epoch")

    def save_plot(ys, labels, colors, title, xlabel, ylabel, filename):
        os.makedirs(plots_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for y, label, color in zip(ys, labels, colors):
            ax.plot(epochs, y, label=label, color=color, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), bbox_inches='tight')
        plt.close()

    save_plot(
        [_as_float_series(rows, "train_loss"), _as_float_series(rows, "val_loss")],
        ['Train loss (Dice + Focal-Tversky + annealed HausdorffDT)',
         'Val loss (Dice + Focal-Tversky + annealed HausdorffDT)'],
        ['#2c6fad', '#c0392b'],
        'Combined Dice + Focal-Tversky + Hausdorff Loss',
        'Epoch', 'Loss',
        'loss.png'
    )

    save_plot(
        [_as_float_series(rows, "mean_dice"), _as_float_series(rows, "dice_et"),
         _as_float_series(rows, "dice_wt"), _as_float_series(rows, "dice_tc")],
        ['Mean Dice', 'ET', 'WT', 'TC'],
        ['#2c6fad', '#c0392b', '#27ae60', '#e67e22'],
        'Dice scores (validation)',
        'Epoch', 'Dice',
        'dice.png'
    )

    save_plot(
        [_as_float_series(rows, "mean_hd95")],
        ['HD95 mean'],
        ['#8e44ad'],
        'Hausdorff distance 95 (validation)',
        'Epoch', 'HD95 (mm)',
        'hd95.png'
    )

    save_plot(
        [_as_float_series(rows, "mean_iou")],
        ['mIoU mean'],
        ['#16a085'],
        'mIoU (validation)',
        'Epoch', 'mIoU',
        'iou.png'
    )

    print(f"All plots saved to {plots_dir}/")

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


# Categorical slots 1-3 of the validated reference palette (dataviz skill; the
# same slots tools/compare_runs.py uses). Only three slots validate all-pairs,
# so the mean, an aggregate of the three regions rather than a fourth entity,
# is drawn dashed in ink. Colour follows the entity: TC is the same blue in
# every plot, and "train" is slot 1 wherever it appears.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
_INK, _INK_2, _GRID = "#0b0b0b", "#52514e", "#e1e0d9"
_REGIONS = (("tc", "TC", _SERIES[0]), ("wt", "WT", _SERIES[1]), ("et", "ET", _SERIES[2]))

# train.py and tools/replot.py override these from cfg.plot. Smoothing is off
# here so a bare call draws exactly the logged values.
DEFAULT_PLOT_STYLE = {
    "font_family": "serif",
    "font_size": 12,
    "fig_size": (7.0, 4.5),
    "dpi": 150,
    "formats": ("png",),
    "epoch_smoothing": 0.0,
    "step_smoothing": 0.0,
    "watermark": None,   # text stamped across every figure, see leak_watermark
}


def leak_watermark(leak):
    """Watermark for the curves of a run trained with cfg.data.leak set, else None."""
    return f"LEAKED SPLIT (data.leak={leak})" if leak else None


def smooth_series(values, weight):
    """Bias-corrected exponential moving average, i.e. TensorBoard's smoothing
    slider: s_t = weight * s_(t-1) + (1 - weight) * x_t, divided by
    (1 - weight^n) so the first points are not pulled toward zero.

    NaN entries (an epoch without validation) stay NaN and do not advance the
    average. weight=0 returns the values unchanged. Plot-time only: nothing
    smoothed is ever written back to a CSV.
    """
    if not 0.0 <= weight < 1.0:
        raise ValueError(f"smoothing weight must be in [0, 1), got {weight}")
    smoothed, running, n = [], 0.0, 0
    for value in values:
        if value != value:  # NaN
            smoothed.append(float("nan"))
            continue
        running = weight * running + (1.0 - weight) * value
        n += 1
        smoothed.append(running / (1.0 - weight ** n))
    return smoothed


def _epoch_means(step_rows, key, epochs):
    """Per-epoch mean of a train_steps.csv column, aligned to `epochs`."""
    sums, counts = {}, {}
    for row in step_rows:
        try:
            epoch, value = int(float(row["epoch"])), float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        sums[epoch] = sums.get(epoch, 0.0) + value
        counts[epoch] = counts.get(epoch, 0) + 1
    return [sums[int(e)] / counts[int(e)] if e == e and int(e) in counts else float("nan")
            for e in epochs]


def _style_rc(style):
    size = style["font_size"]
    return {
        'font.family': style["font_family"],
        'font.size': size,
        'axes.titlesize': size + 1,
        'axes.labelsize': size,
        'legend.fontsize': size - 1,
        'xtick.labelsize': size - 2,
        'ytick.labelsize': size - 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': style["dpi"],
        'savefig.dpi': style["dpi"],
    }


def _save_curves(series, x, title, xlabel, ylabel, stem, plots_dir, style, smoothing):
    """One figure, one y-axis. `series` holds (y, label, color, linestyle).

    All-NaN series (a column this CSV does not have) are dropped, and a figure
    left with none is not written. With smoothing on, each raw curve is drawn
    faintly under its smoothed curve and the weight is stated on the figure.
    """
    series = [s for s in series if any(v == v for v in s[0])]
    if not series:
        return
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=tuple(style["fig_size"]))
    for y, label, color, linestyle in series:
        if smoothing > 0:
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.3)
            ax.plot(x, smooth_series(y, smoothing), label=label, color=color,
                    linestyle=linestyle, linewidth=2.0)
        else:
            ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2.0)
    note = f"EMA smoothing {smoothing:g}, raw values faint" if smoothing > 0 else None
    ax.set_title(title)
    ax.set_xlabel(xlabel if note is None or len(series) > 1 else f"{xlabel}   ({note})")
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(frameon=False, title=note, title_fontsize="small")
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if style.get("watermark"):
        # Across the plot area, where cropping the figure cannot remove it.
        ax.text(0.5, 0.5, style["watermark"], transform=ax.transAxes, ha="center",
                va="center", rotation=20, fontsize=style["font_size"] * 1.6,
                color=_INK_2, alpha=0.25, zorder=0)
    fig.tight_layout()
    for fmt in style["formats"]:
        fig.savefig(os.path.join(plots_dir, f"{stem}.{fmt}"), bbox_inches='tight')
    plt.close(fig)


def plot_metrics_from_csv(csv_path, plots_dir, train_steps_csv_path=None, style=None):
    """Draw a run's training curves from its CSVs: loss, Dice, IoU, HD95 and LR
    per epoch from metrics.csv, plus the per-step training loss when
    `train_steps_csv_path` exists. Only the CSVs are read, so tools/replot.py
    can redraw a finished run with another font, size, format or smoothing.

    `style` overrides DEFAULT_PLOT_STYLE (train.py passes cfg.plot). The style
    is scoped to these figures and never left in the global rcParams.
    """
    style = {**DEFAULT_PLOT_STYLE, **(style or {})}
    rows = _read_metrics_csv(csv_path)
    if not rows:
        print(f"[Plot] no rows found in {csv_path}, skipping metric plots")
        return
    step_rows = (_read_metrics_csv(train_steps_csv_path)
                 if train_steps_csv_path and os.path.exists(train_steps_csv_path) else [])

    epochs = _as_float_series(rows, "epoch")
    smoothing = style["epoch_smoothing"]

    def col(key):
        return _as_float_series(rows, key)

    def per_region(metric):
        return ([(col(f"{metric}_{key}"), name, color, "-") for key, name, color in _REGIONS]
                + [(col(f"mean_{metric}"), "Mean", _INK, "--")])

    with mpl.rc_context(_style_rc(style)):
        # train_loss includes the deep-supervision heads; val_loss is the main
        # head only. The dashed main-head train curve is the like-for-like one.
        _save_curves(
            [(col("train_loss"), "Train (main + deep-supervision heads)", _SERIES[0], "-"),
             (_epoch_means(step_rows, "loss_main", epochs), "Train (main head only)",
              _SERIES[0], "--"),
             (col("val_loss"), "Validation (main head)", _SERIES[1], "-")],
            epochs, "Combined Dice + Focal-Tversky + Hausdorff Loss", "Epoch", "Loss",
            "loss", plots_dir, style, smoothing)
        _save_curves(per_region("dice"), epochs, "Dice scores (validation)",
                     "Epoch", "Dice", "dice", plots_dir, style, smoothing)
        _save_curves(per_region("iou"), epochs, "IoU (validation)",
                     "Epoch", "IoU", "iou", plots_dir, style, smoothing)
        _save_curves(per_region("hd95"), epochs, "Hausdorff distance 95 (validation)",
                     "Epoch", "HD95 (mm)", "hd95", plots_dir, style, smoothing)
        # A schedule has no noise to smooth.
        _save_curves([(col("lr"), "Learning rate", _INK, "-")], epochs, "Learning rate",
                     "Epoch", "Learning rate", "lr", plots_dir, style, 0.0)

        points = []
        for row in step_rows:
            try:
                points.append((int(float(row["epoch"])), int(float(row["step"])), row))
            except (KeyError, TypeError, ValueError):
                continue
        if points:
            last_step = {}
            for epoch, step, _ in points:
                last_step[epoch] = max(last_step.get(epoch, 1), step)
            # x = epoch - 1 + the fraction of that epoch's steps done, so the
            # last step of epoch 4 sits at x=4, where metrics.csv puts epoch 4.
            x = [epoch - 1 + step / last_step[epoch] for epoch, step, _ in points]
            kept = [row for _, _, row in points]
            _save_curves(
                [(_as_float_series(kept, "loss"), "Total (main + deep-supervision heads)",
                  _SERIES[0], "-"),
                 (_as_float_series(kept, "loss_main"), "Main head only", _SERIES[0], "--")],
                x, "Training loss per step", "Epoch", "Loss", "loss_steps", plots_dir,
                style, style["step_smoothing"])

    print(f"All plots saved to {plots_dir}/")

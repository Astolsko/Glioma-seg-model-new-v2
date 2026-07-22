"""Standalone manual tool for tuning cfg.crop.*_start_slice and
cfg.crop.threshold.

Loads ONE patient sample through Load -> EnsureChannelFirst -> Orientation ->
Spacing — i.e. the RAW scan, BEFORE Resized ever touches it — then crops the
depth axis using the real CropRawDepthd class imported straight from
utils/transforms.py, not a re-implementation, using whatever START_SLICE /
END_SLICE you set below. Every raw depth slice, before and after the crop,
gets dumped to PNG so you can scroll through them and decide whether the
window keeps the brain (and tumor) or cuts into it.

It then runs the real Resize + the real CropForegroundHWd (the H/W
foreground-threshold crop) too, at whatever THRESHOLD you set below, and
saves a before/after PNG with the computed bounding box drawn on it — so you
can check the threshold isn't cutting into the brain or leaving in a ring of
background.

Because this script calls the real CropRawDepthd/CropForegroundHWd, their
normal one-time "[CropRawDepthd] one-time check ..." print (see
utils/transforms.py) will also fire here. Compare that line against the one
printed at the start of a real training run — if the numbers match, the crop
in real training is doing exactly what you see in these PNGs.

Not part of the training pipeline — run directly:

    python tools/crop_visual_check.py

Before running, fill in SAMPLE_DIR below.
"""
import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, Resized

# Let this script be run as `python tools/crop_visual_check.py` from anywhere
# by putting the repo root (parent of tools/) on sys.path, same as it would
# be if train.py (which lives at the repo root) had done the importing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from utils.dataloader import ConvertToMultiChannelBasedOnBratsClassesd
from utils.transforms import CropRawDepthd, CropForegroundHWd

# ---------------------------------------------------------------------------
# PLACEHOLDER — fill this in with ONE patient's data folder, e.g.:
#   SAMPLE_DIR = "/DATA/Abul Hasan/Glioma Revision/data/combined/BraTS19_2013_10_1"
#
# The folder must contain (same layout utils/dataloader.py:load_datalist
# expects, "<id>" = the folder name itself):
#   <id>_flair.nii, <id>_t1.nii, <id>_t1ce.nii, <id>_t2.nii, <id>_seg.nii
# ---------------------------------------------------------------------------
SAMPLE_DIR = r"/DATA/Abul Hasan/Glioma Revision/data/combined/BraTS20_Training_071"  # <-- put your sample's directory path here

# ---------------------------------------------------------------------------
# ENTER THE RAW SLICE WINDOW TO TRY HERE.
# These are indices into the ORIGINAL scan (0..~154 for BraTS — every patient
# is co-registered to the exact same 240x240x155 grid at 1mm spacing, so a
# fixed raw window means the same physical range for every patient).
#
# Keep END_SLICE - START_SLICE == cfg.unetr.img_shape[-1] (currently
# {96}) so Resized's depth resize stays an exact no-op — this script warns
# you if it doesn't.
# ---------------------------------------------------------------------------
START_SLICE = 40  # <-- edit this
END_SLICE = START_SLICE + cfg.unetr.img_shape[-1]  # <-- or hardcode an absolute number instead

# ---------------------------------------------------------------------------
# FOREGROUND THRESHOLD FOR THE H/W CROP — runs AFTER the depth crop + Resize.
# A voxel counts as "foreground" if its intensity is > THRESHOLD; the H/W
# bounding box around all foreground voxels (unioned across modalities and
# depth) is what gets kept. Edit this to try different values without
# touching config.py.
# ---------------------------------------------------------------------------
THRESHOLD = 0.25  # <-- edit this

# Which modality to render (crop itself still runs on all 4 modalities +
# label, exactly like training — this only picks what gets plotted).
MODALITY_NAMES = ["flair", "t1", "t1ce", "t2"]
MODALITY_INDEX = 0

# Overlay the tumor label mask (any nonzero class) on the PNGs.
OVERLAY_LABEL = True

OUTPUT_DIR = "crop_check_output"


def _build_datalist_entry(sample_dir):
    sample_dir = sample_dir.rstrip("/\\")
    patient_id = os.path.basename(sample_dir)
    image_paths = [
        os.path.join(sample_dir, f"{patient_id}_{m}.nii") for m in MODALITY_NAMES
    ]
    label_path = os.path.join(sample_dir, f"{patient_id}_seg.nii")
    missing = [p for p in (*image_paths, label_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Sample is missing expected file(s):\n  " + "\n  ".join(missing)
        )
    return {"image": image_paths, "label": label_path}, patient_id


def _build_raw_transform():
    """Load -> EnsureChannelFirst -> Orientation -> Spacing only — the RAW
    resolution the real pipeline now crops at, before Resized ever runs."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
    ])


def _save_slices(volume, out_dir, label_volume=None, index_offset=0, highlight_range=None):
    os.makedirs(out_dir, exist_ok=True)
    depth = volume.shape[-1]
    for d in range(depth):
        slc = volume[:, :, d]
        real_idx = d + index_offset
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(slc, cmap="gray")
        if label_volume is not None:
            label_slc = label_volume[:, :, d]
            if label_slc.any():
                ax.imshow(np.ma.masked_where(label_slc == 0, label_slc),
                          cmap="autumn", alpha=0.4)
        in_window = highlight_range is not None and highlight_range[0] <= real_idx < highlight_range[1]
        title_color = "green" if in_window else "black"
        ax.set_title(f"raw slice {real_idx}" + (" (kept)" if in_window else ""), color=title_color)
        ax.axis("off")
        fig.savefig(os.path.join(out_dir, f"slice_{real_idx:03d}.png"),
                    bbox_inches="tight", dpi=100)
        plt.close(fig)


def _foreground_bbox(image, threshold):
    """Mirrors CropForegroundHWd's own bbox math exactly, purely so this
    script can draw/print the box — the actual crop below still goes
    through the real CropForegroundHWd class, not this."""
    foreground = image > threshold
    foreground = np.any(foreground, axis=0)
    hw_mask = np.any(foreground, axis=2)
    coords = np.where(hw_mask)
    return coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()


def main():
    if not SAMPLE_DIR:
        raise ValueError(
            "SAMPLE_DIR is empty — open tools/crop_visual_check.py and set "
            "SAMPLE_DIR to one patient's folder before running this script."
        )
    if not os.path.isdir(SAMPLE_DIR):
        raise NotADirectoryError(f"SAMPLE_DIR does not exist: {SAMPLE_DIR}")

    entry, patient_id = _build_datalist_entry(SAMPLE_DIR)
    print(f"Loading sample: {patient_id}")

    raw_transform = _build_raw_transform()
    data = raw_transform(entry)

    image = np.asarray(data["image"])
    label = np.asarray(data["label"])
    print(f"RAW (post Orientation+Spacing, pre-Resize) -> image shape: {image.shape}, label shape: {label.shape}")

    depth_axis = int(np.argmin(image.shape[1:])) + 1
    source_depth = image.shape[depth_axis]
    print(f"Detected depth axis: {depth_axis} (size {source_depth})")
    print(f"Requested RAW crop window: [{START_SLICE}:{END_SLICE}) -> {END_SLICE - START_SLICE} slices requested")

    target_depth = cfg.unetr.img_shape[-1]
    if END_SLICE - START_SLICE != target_depth:
        print(f"NOTE: window length ({END_SLICE - START_SLICE}) != cfg.unetr.img_shape[-1] "
              f"({target_depth}) — Resized will still have to stretch/compress this axis "
              "instead of a no-op. Not wrong, just no longer 'zero compression'.")

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    before_dir = os.path.join(OUTPUT_DIR, "before_crop")
    after_dir = os.path.join(OUTPUT_DIR, "after_crop")

    # any-nonzero label mask, for the overlay (raw label is still the
    # original single-channel {0,1,2,4} values at this point, no need to
    # convert to multi-channel just to check "is there tumor here")
    label_mask = (label[0] > 0).astype(np.float32) if OVERLAY_LABEL else None

    # --- BEFORE crop: every slice of the RAW volume, with the intended
    # window highlighted in green so you can see what's being kept/dropped
    # in context ---
    print(f"Saving {source_depth} raw slices -> {before_dir}/")
    _save_slices(image[MODALITY_INDEX], before_dir, label_volume=label_mask,
                 highlight_range=(START_SLICE, END_SLICE))

    # --- run the REAL crop transform used by training/validation ---
    crop_transform = CropRawDepthd(
        keys=["image", "label"], start_slice=START_SLICE, num_slices=END_SLICE - START_SLICE,
    )
    cropped = crop_transform(data)
    cropped_image = np.asarray(cropped["image"])
    cropped_label = np.asarray(cropped["label"])
    print(f"After CropRawDepthd -> image shape: {cropped_image.shape}, label shape: {cropped_label.shape}")

    kept_depth = cropped_image.shape[depth_axis]
    cropped_label_mask = (cropped_label[0] > 0).astype(np.float32) if OVERLAY_LABEL else None

    # --- AFTER crop: only the kept slices, numbered by their original raw
    # slice index so you can cross-reference directly against before_crop/ ---
    print(f"Saving {kept_depth} post-crop slices -> {after_dir}/")
    _save_slices(cropped_image[MODALITY_INDEX], after_dir,
                 label_volume=cropped_label_mask, index_offset=START_SLICE)

    # --- run the rest of the REAL pipeline (Resize + label convert) so the
    # foreground threshold below sees the same H/W resolution training does ---
    resize_transform = Resized(keys=["image", "label"], spatial_size=cfg.unetr.img_shape, mode="nearest")
    resized = resize_transform(cropped)
    resized = ConvertToMultiChannelBasedOnBratsClassesd(keys="label")(resized)
    resized_image = np.asarray(resized["image"])
    resized_label = np.asarray(resized["label"])

    h_min, h_max, w_min, w_max = _foreground_bbox(resized_image, THRESHOLD)
    print(f"\nForeground bbox at threshold={THRESHOLD}: "
          f"H[{h_min}:{h_max + 1}] W[{w_min}:{w_max + 1}] "
          f"out of full H,W={resized_image.shape[1:3]}")

    # --- run the REAL H/W crop transform used by training/validation ---
    hw_crop_transform = CropForegroundHWd(keys=["image", "label"], threshold=THRESHOLD)
    final = hw_crop_transform(resized)
    final_image = np.asarray(final["image"])
    final_label = np.asarray(final["label"])
    print(f"After CropForegroundHWd -> image shape: {final_image.shape}, label shape: {final_label.shape}")

    hw_dir = os.path.join(OUTPUT_DIR, "foreground_hw_crop")
    os.makedirs(hw_dir, exist_ok=True)
    mid = resized_image.shape[-1] // 2

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(resized_image[MODALITY_INDEX, :, :, mid], cmap="gray")
    tumor_before = np.any(resized_label[:, :, :, mid] > 0, axis=0)
    if OVERLAY_LABEL and tumor_before.any():
        ax.imshow(np.ma.masked_where(~tumor_before, tumor_before), cmap="autumn", alpha=0.4)
    ax.add_patch(plt.Rectangle((w_min, h_min), w_max - w_min + 1, h_max - h_min + 1,
                                edgecolor="lime", facecolor="none", linewidth=2))
    ax.set_title(f"before H/W crop (threshold={THRESHOLD})\nfull size {resized_image.shape[1:3]}, green = kept box")
    ax.axis("off")
    fig.savefig(os.path.join(hw_dir, "before_hw_crop_bbox.png"), bbox_inches="tight", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(final_image[MODALITY_INDEX, :, :, mid], cmap="gray")
    tumor_after = np.any(final_label[:, :, :, mid] > 0, axis=0)
    if OVERLAY_LABEL and tumor_after.any():
        ax.imshow(np.ma.masked_where(~tumor_after, tumor_after), cmap="autumn", alpha=0.4)
    ax.set_title(f"after H/W crop\nfinal size {final_image.shape[1:3]}")
    ax.axis("off")
    fig.savefig(os.path.join(hw_dir, "after_hw_crop.png"), bbox_inches="tight", dpi=120)
    plt.close(fig)

    print("\nDone. Open the PNGs in:")
    print(f"  {before_dir}/       (full raw volume, green title = inside the kept depth window)")
    print(f"  {after_dir}/        (depth crop result, tumor overlaid in yellow/red)")
    print(f"  {hw_dir}/  (green box = what THRESHOLD keeps; after_hw_crop.png = final H/W result)")
    print("Adjust START_SLICE / END_SLICE / THRESHOLD at the top of this script and "
          "re-run until it looks right, then copy START_SLICE into config.py's "
          "cfg.crop.train_start_slice / val_start_slice (end is derived automatically "
          "from cfg.unetr.img_shape[-1]) and THRESHOLD into cfg.crop.threshold.")


if __name__ == "__main__":
    main()

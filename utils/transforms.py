import numpy as np
from monai.transforms import (
    MapTransform,
    Compose,
    LoadImaged,
    Orientationd,
    Resized,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    ToTensord,
    RandFlipd,
    RandAffined,
    RandBiasFieldd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
)

from utils.dataloader import ConvertToMultiChannelBasedOnBratsClassesd, ApplyCLAHEAndZscored


class CropRawDepthd(MapTransform):
    """
    Crops a fixed-length depth window directly on the RAW scan — i.e. right
    after Orientation+Spacing, BEFORE Resized ever touches it — using
    absolute raw slice indices (e.g. slice 20 of the original ~155-slice
    scan, not slice 20 of a resized volume).

    `num_slices` is always cfg.unetr.img_shape[-1] (the depth the model
    consumes), so the window handed to Resized already has exactly the
    target depth. That makes Resized's depth resize an exact no-op (verified:
    nearest-neighbor resize with input size == output size is bit-identical,
    no subtle resampling), so no slices get silently discarded/blended the
    way they were when Resized first compressed the whole raw scan down to
    img_shape[-1] slices and only THEN got cropped further.
    """

    def __init__(self, keys, start_slice, num_slices):
        super().__init__(keys)
        self.start_slice = start_slice
        self.num_slices = num_slices
        # Gate the crop-range confirmation print so it fires once per
        # transform instance instead of on every sample/batch — with
        # num_workers > 0 each DataLoader worker holds its own copy of this
        # transform, so in practice you may see it print once per worker
        # (still a handful of lines, not one per batch).
        self._logged_crop_range = False

    def __call__(self, data):
        d = dict(data)
        end_slice = self.start_slice + self.num_slices

        source_depth = None
        kept_depth = None
        depth_axis = None
        for key in self.keys:
            arr = np.asarray(d[key])
            # image: [C,H,W,D], label: [C,H,W,D]

            # identify depth axis dynamically (smallest spatial dim)
            depth_axis = np.argmin(arr.shape[1:]) + 1

            if key == "image":
                source_depth = arr.shape[depth_axis]

            slices = [slice(None)] * arr.ndim
            slices[depth_axis] = slice(self.start_slice, end_slice)
            arr = arr[tuple(slices)]
            d[key] = arr

            if key == "image":
                kept_depth = arr.shape[depth_axis]

        if not self._logged_crop_range:
            note = "" if kept_depth == self.num_slices else (
                f" — MISMATCH: requested window runs past the available raw "
                f"depth ({source_depth}), so Resized will still have to "
                f"stretch/compress this axis instead of a no-op"
            )
            print(
                f"[CropRawDepthd] one-time check — depth_axis={depth_axis} | "
                f"requested RAW slices [{self.start_slice}:{end_slice}) = "
                f"{self.num_slices} slices | "
                f"source raw depth={source_depth} -> kept depth={kept_depth}{note}"
            )
            self._logged_crop_range = True

        return d


class CropForegroundHWd(MapTransform):
    """
    Crop H/W to the foreground bounding box (union across modalities).
    Depth is left untouched here — it was already fixed to exactly
    cfg.unetr.img_shape[-1] slices by CropRawDepthd, before Resized ran.
    Runs AFTER Resized since H/W only make sense at the resized resolution.
    """

    def __init__(self, keys, threshold=0):
        super().__init__(keys)
        self.threshold = threshold

    def __call__(self, data):
        d = dict(data)

        image = np.asarray(d["image"])
        # image shape: (4,H,W,D)
        foreground = image > self.threshold

        # merge modalities
        foreground = np.any(foreground, axis=0)

        # collapse depth — only keep H/W information
        hw_mask = np.any(foreground, axis=2)
        coords = np.where(hw_mask)

        h_min = coords[0].min()
        h_max = coords[0].max()
        w_min = coords[1].min()
        w_max = coords[1].max()

        # KEEP ENTIRE DEPTH
        for key in self.keys:
            arr = np.asarray(d[key])
            arr = arr[:, h_min:h_max + 1, w_min:w_max + 1, :]
            d[key] = arr

        return d


def build_resize(cfg):
    """Resize H/W down to img_shape — depth is already exact by the time this
    runs (see CropRawDepthd), so this only touches the in-plane axes.

    `mode` is PER KEY here. It used to be a single mode="nearest" covering both
    keys, which point-sampled the MRI from 240 to 128 in-plane: a 1.875x
    downsample with no filtering, i.e. textbook aliasing, applied to exactly
    the fine T1ce enhancement texture that ET segmentation depends on. Labels
    must stay nearest (interpolating class ids is meaningless), but images want
    trilinear, and a downsample additionally wants anti-aliasing — without it
    trilinear still undersamples, it just aliases more smoothly.

    Shared by the train and val/test transforms so the two cannot drift apart.
    """
    return Resized(
        keys=["image", "label"],
        spatial_size=cfg.unetr.img_shape,
        mode=("trilinear", "nearest"),
        align_corners=(False, None),
        anti_aliasing=(True, False),
    )


def build_train_transform(cfg):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropRawDepthd(
            keys=["image", "label"],
            start_slice=cfg.crop.train_start_slice,
            num_slices=cfg.unetr.img_shape[-1],
        ),
        build_resize(cfg),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        CropForegroundHWd(keys=["image", "label"], threshold=cfg.crop.threshold),
        ApplyCLAHEAndZscored(keys="image"),

        # spatial — both image and label.
        # RandRotate90 (max_k=3) used to sit here. A brain never appears
        # rotated 90/180/270 degrees in an RAS-oriented scan, so it spent
        # augmentation budget teaching invariance to poses that do not occur,
        # and after the foreground crop it swapped a non-square H/W before
        # fit_to_size padded the result back. Small-angle affine is the
        # realistic version of the same idea.
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"], prob=0.3,
            rotate_range=(0.26, 0.26, 0.26),   # +/- ~15 degrees per axis
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"), padding_mode="zeros",
        ),

        # intensity — image only
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys="image", prob=0.2,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)
        ),
        RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.3)),
        # Smooth multiplicative intensity drift — simulates the scanner
        # inhomogeneity that survives N4 correction, and is the main
        # cross-scanner nuisance for T1ce enhancement.
        RandBiasFieldd(keys="image", prob=0.3, degree=3, coeff_range=(0.0, 0.1)),

        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


def build_val_transform(cfg):
    """Shared by validation AND test so both splits see identical
    preprocessing (crop-based, no augmentation)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropRawDepthd(
            keys=["image", "label"],
            start_slice=cfg.crop.val_start_slice,
            num_slices=cfg.unetr.img_shape[-1],
        ),
        build_resize(cfg),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        CropForegroundHWd(keys=["image", "label"], threshold=cfg.crop.threshold),
        ApplyCLAHEAndZscored(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


def fit_to_size(x, target):
    """Center pad-or-crop a (B, C, D, H, W)-order tensor so its spatial dims
    match `target`. Only ever applied to training batches — mirrors the
    original notebook's workaround for the depth-crop/resize interaction."""
    import torch.nn.functional as F

    pad = []  # F.pad order: W, H, D
    for dim, t in zip((4, 3, 2), (target[2], target[1], target[0])):
        diff = t - x.shape[dim]
        pad.extend([diff // 2, diff - diff // 2] if diff > 0 else [0, 0])
    if any(pad):
        x = F.pad(x, pad)
    sl = [slice(None), slice(None)]
    for dim, t in zip((2, 3, 4), target):
        start = max((x.shape[dim] - t) // 2, 0)
        sl.append(slice(start, start + t))
    return x[tuple(sl)]

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # non-interactive backend before pyplot; see utils/plot.py
from matplotlib import pyplot as plt

from utils.metrics import minmax_normalize


def register_attention_hook(model):
    """Registers a forward hook on the last transformer block's attention
    module and returns the cache dict the hook writes into (weights shape:
    (B, num_heads, num_patches, num_patches)).

    The hook is ARMED, not always-on: it fires on every sliding-window patch of
    every validation sample, but the map is only ever consumed on the epochs
    that save an overlay. Capturing unconditionally meant cloning a
    (1, heads, patches, patches) tensor tens of thousands of times per run and
    leaving the last one resident in GPU memory for the whole run. Callers arm
    it with `set_attention_capture(cache, True)` for the epoch they want.

    The clone is moved to CPU on capture. It is consumed by
    `extract_attention_map`, whose first move is `.cpu()` anyway, so nothing
    downstream notices — and nothing stays pinned in the GPU pool.
    """
    cache = {"_capture": False}

    if getattr(model, "encoder_type", "vit") == "mamba":
        return _register_mamba_hook(model, cache)

    def _attention_hook(module, inputs, output):
        if not cache.get("_capture"):
            return
        if isinstance(output, (tuple, list)) and len(output) == 2:
            weights = output[1]
            if weights is not None:
                cache["attn"] = weights.detach().to("cpu", copy=True)

    model.transformer.layer[-1].attn.register_forward_hook(_attention_hook)
    return cache


def _register_mamba_hook(model, cache):
    """Mamba counterpart of the hook above, on the last ToM mixer of the
    deepest encoder stage (whose token grid is the ViT's 1/16 patch grid).

    A Mamba layer returns no attention weights: its implicit ("hidden")
    attention has to be computed from the layer's input (Ali et al. 2024, see
    blocks/VisionMamba.py). That is far costlier than the ViT's clone and the
    hook fires on every sliding-window patch while armed, so only the INPUT
    tokens are kept here (overwritten each call, on CPU) and the matrix is
    built once, lazily, in extract_attention_map for the patch that gets
    plotted. Same "last patch of the sample" semantics as the ViT hook.
    """
    mixer = model.mamba_encoder.last_stage_mixers()[-1]

    def _mamba_hook(module, inputs, output):
        if not cache.get("_capture"):
            return
        cache["mamba_tokens"] = inputs[0].detach().to("cpu", copy=True)
        cache["mamba_nslices"] = int(inputs[1])
        cache["mamba_mixer"] = module

    mixer.register_forward_hook(_mamba_hook)
    return cache


_CAPTURE_KEYS = ("attn", "mamba_tokens", "mamba_nslices", "mamba_mixer")


def set_attention_capture(cache, enabled):
    """Arm or disarm the capture hook. Disarming also drops the cached map, so
    a saved overlay does not keep a tensor alive for the rest of the run."""
    if cache is None:
        return
    cache["_capture"] = bool(enabled)
    if not enabled:
        for key in _CAPTURE_KEYS:
            cache.pop(key, None)


def extract_attention_map(cache, model, img_shape):
    attn = cache.get("attn")
    if attn is None and cache.get("mamba_tokens") is not None:
        # (B, L, L) hidden attention -> (B, 1, L, L): one "head", so the
        # reduction below is the same "average attention each token receives".
        mixer = cache["mamba_mixer"]
        device = next(mixer.parameters()).device
        attn = mixer.hidden_attention(cache["mamba_tokens"].to(device),
                                      cache["mamba_nslices"])[:, None]
    if attn is None:
        return None
    attn = attn.detach().cpu()
    if attn.dim() != 4:
        return None
    attn = attn[0]                     # (num_heads, num_patches, num_patches)
    attn = attn.mean(0)                # (num_patches, num_patches) avg over heads
    attn = attn.mean(0)                # (num_patches,) avg attention each patch receives
    patch_d, patch_h, patch_w = model.patch_dim
    attn = attn.reshape(patch_d, patch_h, patch_w)
    attn_map = attn.unsqueeze(0).unsqueeze(0)
    attn_up = F.interpolate(
        attn_map, size=img_shape,
        mode="trilinear", align_corners=False
    )
    return attn_up.squeeze(0).squeeze(0).numpy()


def save_attention_overlay(cache, model, val_data, val_ds, epoch, img_shape, attention_dir, sample_idx=0):
    attn_map = extract_attention_map(cache, model, img_shape)
    if attn_map is None:
        print(f"[Attention] cache empty at epoch {epoch}, skipping")
        return

    try:
        all_data = val_ds.data
        sample = all_data[sample_idx]
        image_paths = sample["image"]
        t1ce_path = image_paths[2]
    except Exception as e:
        print(f"[Attention] could not get path from dataset: {e}")
        return

    patient_id = os.path.basename(t1ce_path)
    patient_id = patient_id.replace("_t1ce.nii.gz", "").replace("_t1ce.nii", "")

    # USE ALREADY TRANSFORMED INPUT — shape = [B,C,H,W,D]
    vol_resized = (
        val_data["image"][sample_idx, 2]   # T1ce modality
        .detach()
        .cpu()
        .numpy()
    )

    # --- pick 5 informative slices based on attention, along the true
    # depth/D axis (attn_map and vol_resized are both (H,W,D) — indexing
    # must clamp against shape[-1], not shape[0]/H, otherwise a non-cubic
    # per-patient shape (H != D, the normal case after the foreground crop)
    # raises IndexError as soon as a chosen index exceeds the D extent) ---
    slice_means = attn_map.mean(axis=(0, 1))
    top_idx = np.argsort(slice_means)[::-1]
    # filter to keep slices at least 10 apart
    chosen = []
    for idx in top_idx:
        if all(abs(idx - c) >= 10 for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == 5:
            break
    depth = attn_map.shape[-1]
    chosen = [min(int(x), depth - 1) for x in sorted(chosen)]  # sort anatomically

    os.makedirs(attention_dir, exist_ok=True)

    # ── Plot 1: 5-slice multi-panel (MRI | overlay) ──────────────────────
    fig, axes = plt.subplots(
        2, 5, figsize=(22, 9),
        gridspec_kw={"hspace": 0.05, "wspace": 0.03}
    )
    fig.patch.set_facecolor("#0d0d0d")
    map_name = ("Mamba Hidden-Attention Map (deepest stage)"
                if getattr(model, "encoder_type", "vit") == "mamba"
                else "Transformer Attention Map")
    fig.suptitle(
        f"{patient_id}  |  {map_name}  |  Epoch {epoch}",
        color="white", fontsize=13, fontweight="bold", y=0.50
    )

    for col, sl in enumerate(chosen):
        mri = minmax_normalize(vol_resized[:, :, sl])
        att = minmax_normalize(attn_map[:, :, sl])

        # row 0 — raw MRI
        axes[0, col].imshow(mri, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Slice {sl}", color="#aaaaaa", fontsize=9, pad=3)
        axes[0, col].axis("off")

        # row 1 — MRI + attention overlay
        axes[1, col].imshow(mri, cmap="gray", origin="lower")
        im = axes[1, col].imshow(att, cmap="inferno", alpha=0.55, origin="lower",
                                  vmin=0, vmax=1)
        axes[1, col].axis("off")

    # shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Attention", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # row labels
    for row, label in enumerate(["T1ce", "Attention\nOverlay"]):
        fig.text(0.01, 0.72 - row * 0.44, label,
                  color="white", fontsize=10, fontweight="bold",
                  va="center", rotation=90)

    out_path = os.path.join(
        attention_dir, f"attention_epoch{epoch:04d}_{patient_id}_multislice.png"
    )
    plt.savefig(out_path, bbox_inches="tight", facecolor="#0d0d0d", dpi=150)
    plt.close()
    print(f"[Attention] saved: {out_path}")


def save_attention_evolution(attention_dir, patient_id_substr=""):
    """
    Collects all saved attention maps for one patient across epochs
    and plots them side by side to show attention evolution.
    """
    pattern = os.path.join(attention_dir, f"attention_epoch*{patient_id_substr}*multislice.png")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print("[Evolution] no attention files found")
        return
    if len(files) > 8:
        # subsample evenly
        idx = np.linspace(0, len(files) - 1, 8, dtype=int)
        files = [files[i] for i in idx]

    from PIL import Image
    imgs = [np.array(Image.open(f)) for f in files]

    # crop to just the overlay row (bottom half of each image)
    crops = [im[im.shape[0] // 2:, :, :] for im in imgs]

    fig, axes = plt.subplots(1, len(crops), figsize=(5 * len(crops), 5))
    fig.patch.set_facecolor("#0d0d0d")
    if len(crops) == 1:
        axes = [axes]

    epoch_labels = []
    for f in files:
        base = os.path.basename(f)
        try:
            ep = int(base.split("epoch")[1].split("_")[0])
            epoch_labels.append(f"Ep {ep}")
        except Exception:
            epoch_labels.append("")

    for ax, crop, label in zip(axes, crops, epoch_labels):
        ax.imshow(crop)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Attention Map Evolution Across Training",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(attention_dir, "attention_evolution.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor="#0d0d0d", dpi=150)
    plt.close()
    print(f"[Evolution] saved: {out_path}")

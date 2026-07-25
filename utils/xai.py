"""Post-hoc explainability for the 3D glioma segmentation model.

Five components, all running against a saved checkpoint — nothing here touches
the training path:

  X1  Seg-Grad-CAM / HiResCAM at several decoder depths   -> cam_*()
  X2  Modality attribution by ablation                    -> modality_attribution()
  X3  MC-dropout predictive uncertainty                   -> mc_dropout_predict()
  X4  Attention rollout through the 12 ViT blocks         -> attention_rollout()
  X5  Quantitative evaluation of the above                -> deletion_curve(),
                                                             localization_scores(),
                                                             sanity_check_randomization()

Why these and not LIME/SHAP: kernel-SHAP on a 128x128x96x4 volume needs
thousands of forward passes per case, and the superpixel decomposition it needs
to be tractable destroys exactly the boundary detail that ET segmentation is
about. Gradient-based (X1, X4) and perturbation-based (X2, X5) families are
both covered here at a fraction of the cost.

The load-bearing part for a paper is X5. A CAM figure alone is unfalsifiable;
deletion curves, localisation scores and the Adebayo randomisation test are
what make the explanations checkable claims.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils.metrics import minmax_normalize
from utils.transforms import fit_to_size

CHANNEL_NAMES = ("TC", "WT", "ET")
MODALITY_NAMES = ("FLAIR", "T1", "T1ce", "T2")
COMPONENTS = ("cam", "modality", "uncertainty", "rollout", "faithful")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_layer(model, path):
    """Resolve a dotted path like 'decoder0_header.1' against the model."""
    module = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _to_numpy(x):
    return x.detach().float().cpu().numpy()


def _normalize(volume):
    """Scale to [0, 1]. Attribution magnitudes are not comparable across
    layers or methods, only their spatial pattern is, so every map is
    normalised before it is plotted or scored."""
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    span = volume.max() - volume.min()
    if span < 1e-12:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - volume.min()) / span).astype(np.float32)


def prepare_sample(sample, cfg, device):
    """Crop/pad one dataset sample to the model's input size and add a batch
    dim. XAI runs the model directly rather than through
    sliding_window_inference, because gradients have to flow back to a single
    well-defined input tensor — a stitched-together sliding window has no such
    thing."""
    image = fit_to_size(sample["image"].unsqueeze(0), cfg.unetr.img_shape).to(device)
    label = fit_to_size(sample["label"].unsqueeze(0), cfg.unetr.img_shape).to(device)
    return image.float(), label.float()


def dice(pred, gt):
    """Binary Dice on numpy arrays; both-empty counts as agreement."""
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, gt).sum() / denom)


# ---------------------------------------------------------------------------
# X1 — Seg-Grad-CAM / HiResCAM
# ---------------------------------------------------------------------------

class CAM3D:
    """Class-specific attribution at one decoder layer, for 3D segmentation.

    Classification Grad-CAM differentiates a single class logit. Segmentation
    has one logit per voxel, so following Seg-Grad-CAM (Vinogradova et al.,
    AAAI 2020) the score is the SUM of the class logits over a region of
    interest M:

        S_c = sum_{v in M} logit_c(v)

    The ROI matters. With M = the whole volume the score is dominated by the
    ~99% of voxels that are background, and the resulting map explains "where
    is brain", not "why this tumour". M is normally the ground-truth region of
    class c (cfg.xai.cam_roi = "gt") or the model's own prediction ("pred").

    Two aggregations, both computed from the same forward/backward pass:

      grad  — classic Grad-CAM. Pool each channel's gradient to a scalar
              weight, w_k = mean_v dS/dA_k(v), then cam = relu(sum_k w_k A_k).
      hires — HiResCAM. Keep the gradient elementwise:
              cam = relu(sum_k dS/dA_k .* A_k).
              Provably reflects the voxels the model actually used; Grad-CAM's
              pooling step can move mass onto regions that did not contribute.

    Used as a context manager so the forward hook is always removed.
    """

    def __init__(self, model, layer_path):
        self.model = model
        self.layer_path = layer_path
        self._activation = None
        self._handle = None

    def __enter__(self):
        layer = _resolve_layer(self.model, self.layer_path)

        def _hook(_module, _inputs, output):
            # Keep the graph-connected tensor, not a copy — autograd.grad needs
            # the node itself. A backward hook would work too but fires in an
            # order that depends on how the module was composed.
            self._activation = output

        self._handle = layer.register_forward_hook(_hook)
        return self

    def __exit__(self, *_exc):
        if self._handle is not None:
            self._handle.remove()
        self._activation = None
        return False

    def attribute(self, image, class_idx, roi=None, method="hires"):
        """Return (cam, logits): cam is a normalised (H, W, D) numpy array at
        the input's resolution, logits is the model output."""
        self.model.zero_grad(set_to_none=True)
        image = image.detach().requires_grad_(False)

        with torch.enable_grad():
            logits = self.model(image)
            if isinstance(logits, (tuple, list)):     # deep supervision heads
                logits = logits[0]

            target = logits[:, class_idx]
            if roi is not None:
                mask = roi.to(target.dtype)
                # An empty ROI would give a constant-zero score and therefore a
                # zero gradient — a blank map that looks like "the model used
                # nothing" rather than "there was nothing to explain".
                score = (target * mask).sum() if mask.sum() > 0 else target.sum()
            else:
                score = target.sum()

            activation = self._activation
            grads = torch.autograd.grad(score, activation, retain_graph=False)[0]

        if method == "grad":
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True)
        elif method == "hires":
            cam = (grads * activation).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"unknown CAM method: {method!r}")

        cam = F.relu(cam)
        cam = F.interpolate(cam.float(), size=image.shape[2:],
                            mode="trilinear", align_corners=False)
        return _normalize(_to_numpy(cam)[0, 0]), logits.detach()


def cam_for_sample(model, image, label, class_idx, cfg, layer_path, method):
    """Run one CAM with the ROI policy from cfg.xai.cam_roi."""
    roi = None
    if cfg.xai.cam_roi == "gt":
        roi = label[:, class_idx] > 0.5
    elif cfg.xai.cam_roi == "pred":
        with torch.no_grad():
            out = model(image)
            out = out[0] if isinstance(out, (tuple, list)) else out
        roi = torch.sigmoid(out[:, class_idx]) > 0.5

    with CAM3D(model, layer_path) as cam:
        return cam.attribute(image, class_idx, roi=roi, method=method)


# ---------------------------------------------------------------------------
# X2 — modality attribution
# ---------------------------------------------------------------------------

def modality_attribution(model, loader, device, cfg, inferer, thresholds=(0.5, 0.5, 0.5)):
    """Dice cost of removing each MRI modality, per tumour region.

    Zeroing a modality sets it to the value the background already has after
    z-scoring, i.e. "this sequence was not acquired". The resulting 4x3 table
    of Dice drops is the most directly checkable explanation this model can
    produce: radiologically, ET is defined by T1ce enhancement and WT by FLAIR
    hyperintensity, so a model that is right for the right reasons must lose
    ET when T1ce goes and WT when FLAIR goes. A model that does not is
    exploiting something else.
    """
    results = {}
    n_modalities = len(MODALITY_NAMES)

    for ablated in [None, *range(n_modalities)]:
        dice_sum = np.zeros(3, dtype=np.float64)
        n_samples = 0

        with torch.no_grad():
            for batch in loader:
                image = batch["image"].to(device)
                if ablated is not None:
                    image = image.clone()
                    image[:, ablated] = 0.0

                prob = torch.sigmoid(inferer(model, image).float())[0].cpu().numpy()
                gt = batch["label"][0].cpu().numpy() > 0.5
                for c in range(3):
                    dice_sum[c] += dice(prob[c] > thresholds[c], gt[c])
                n_samples += 1

        key = "baseline" if ablated is None else MODALITY_NAMES[ablated]
        results[key] = (dice_sum / max(n_samples, 1)).tolist()

    baseline = np.array(results["baseline"])
    results["delta"] = {
        name: (np.array(results[name]) - baseline).tolist()
        for name in MODALITY_NAMES
    }
    return results


# ---------------------------------------------------------------------------
# X3 — MC-dropout uncertainty
# ---------------------------------------------------------------------------

def enable_dropout(model):
    """Put ONLY the dropout layers back in training mode.

    Not model.train() — that would also re-enable deep-supervision heads and
    change the forward signature. The model already carries dropout worth
    sampling: p=0.2 in every transformer block, p=0.1 spatial dropout in each
    CoordAtt3D.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()
            count += 1
    return count


def mc_dropout_predict(model, image, n_passes):
    """N stochastic passes -> (mean_prob, std, entropy), each (C, H, W, D).

    mean_prob is a free self-ensemble and is usually slightly better than the
    deterministic prediction. `std` across passes is the epistemic part (model
    disagreement); `entropy` of the mean is total predictive uncertainty. Both
    are reported because they fail differently: a confidently-wrong model has
    low std but not necessarily low entropy.
    """
    was_training = model.training
    model.eval()
    n_dropout = enable_dropout(model)

    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(image)
            out = out[0] if isinstance(out, (tuple, list)) else out
            probs.append(torch.sigmoid(out.float())[0])
    stacked = torch.stack(probs)
    model.train(was_training)

    mean_prob = stacked.mean(0)
    std = stacked.std(0)
    p = mean_prob.clamp(1e-7, 1 - 1e-7)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log())

    return _to_numpy(mean_prob), _to_numpy(std), _to_numpy(entropy), n_dropout


def error_retention_curve(pred, gt, uncertainty, fractions):
    """Dice after handing the most-uncertain fraction of voxels to a human.

    Simulates clinical triage: the referred voxels are replaced with ground
    truth, so a rising curve means the uncertainty map is pointing at the
    places the model is actually wrong. A flat curve means it is not, however
    pretty the heatmap looks.
    """
    order = np.argsort(uncertainty, axis=None)[::-1]
    curve = []
    for frac in fractions:
        corrected = pred.copy()
        if frac > 0:
            k = int(frac * order.size)
            idx = np.unravel_index(order[:k], pred.shape)
            corrected[idx] = gt[idx]
        curve.append(dice(corrected, gt))
    return curve


# ---------------------------------------------------------------------------
# X4 — attention rollout
# ---------------------------------------------------------------------------

def attention_rollout(model, image, cfg):
    """Relevance per patch, propagated through all 12 transformer blocks.

    The pipeline's existing attention figure averages the LAST layer's weights
    over heads and over queries. That ignores the eleven layers beneath it and
    the residual stream that carries most of the signal past attention, which
    is the standard criticism of raw-attention explanations.

    Rollout (Abnar & Zuidema, ACL 2020) accounts for the residual by mixing in
    the identity, A_hat = 0.5*A + 0.5*I, renormalising rows, and multiplying
    the chain across layers. There is no CLS token here (patch tokens only),
    so relevance is the column mean of the rollout: how much each patch is
    attended to by everything else.

    Granularity is one 16^3 patch, so this is a coarse global-context map and
    is complementary to, not a substitute for, the voxel-resolution CAMs.
    """
    caches, handles = [], []

    def _make_hook(store):
        def _hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)) and len(output) == 2 and output[1] is not None:
                store.append(output[1].detach())
        return _hook

    for block in model.transformer.layer:
        store = []
        caches.append(store)
        handles.append(block.attn.register_forward_hook(_make_hook(store)))

    try:
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()

    rollout = None
    for store in caches:
        if not store:
            continue
        attn = store[0][0].mean(0)                       # heads -> (P, P)
        attn = attn + torch.eye(attn.shape[0], device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn if rollout is None else attn @ rollout

    if rollout is None:
        return None

    relevance = rollout.mean(0)                           # (P,)
    relevance = relevance.reshape(1, 1, *model.patch_dim)
    relevance = F.interpolate(relevance.float(), size=image.shape[2:],
                              mode="trilinear", align_corners=False)
    return _normalize(_to_numpy(relevance)[0, 0])


# ---------------------------------------------------------------------------
# X5 — quantitative evaluation
# ---------------------------------------------------------------------------

def deletion_curve(model, image, gt_channel, attribution, class_idx, fractions,
                   threshold=0.5):
    """Dice as the highest-attributed voxels are progressively deleted.

    Faithfulness test: if a map really identifies what the model used, zeroing
    those voxels first should destroy the prediction faster than zeroing any
    other voxels. Deletion is applied across all four modalities at once, so
    the perturbation is "this location is gone", not "this sequence is gone"
    (that is X2's job).

    Lower area under the curve = more faithful. Compare against
    `random_baseline_curve` — a map that does no better than random ordering
    explains nothing, and reporting that comparison is what separates this
    from a heatmap gallery.
    """
    order = np.argsort(attribution, axis=None)[::-1]
    curve = []

    with torch.no_grad():
        for frac in fractions:
            perturbed = image.clone()
            if frac > 0:
                k = int(frac * order.size)
                idx = np.unravel_index(order[:k], attribution.shape)
                perturbed[0, :, idx[0], idx[1], idx[2]] = 0.0
            out = model(perturbed)
            out = out[0] if isinstance(out, (tuple, list)) else out
            pred = _to_numpy(torch.sigmoid(out[0, class_idx].float())) > threshold
            curve.append(dice(pred, gt_channel))

    return curve


def random_baseline_curve(model, image, gt_channel, class_idx, fractions,
                          seed=0, threshold=0.5):
    """`deletion_curve` with a random voxel ordering — the null hypothesis."""
    rng = np.random.default_rng(seed)
    shape = tuple(image.shape[2:])
    return deletion_curve(model, image, gt_channel,
                          rng.random(shape).astype(np.float32),
                          class_idx, fractions, threshold=threshold)


def auc(curve, fractions):
    """Trapezoidal area under a curve, normalised by the x-range."""
    # np.trapz was renamed to np.trapezoid in numpy 2.0; this env pins 1.23.5.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    span = fractions[-1] - fractions[0]
    return float(trapezoid(curve, fractions) / span) if span > 0 else float("nan")


def localization_scores(attribution, gt_channel, top_frac, peritumoral_radius):
    """Where the top-k% of attribution mass falls.

    Reports three fractions that sum to 1: inside the tumour, in a dilated
    shell around it (peritumoral context — legitimate evidence, oedema and
    mass effect are diagnostic), and elsewhere in the volume (which is the
    part that should be small).
    """
    from scipy import ndimage

    k = max(int(top_frac * attribution.size), 1)
    flat_idx = np.argsort(attribution, axis=None)[::-1][:k]
    top = np.zeros(attribution.size, dtype=bool)
    top[flat_idx] = True
    top = top.reshape(attribution.shape)

    inside = np.logical_and(top, gt_channel).sum()
    if gt_channel.any():
        dilated = ndimage.binary_dilation(gt_channel, iterations=peritumoral_radius)
    else:
        dilated = gt_channel
    shell = np.logical_and(top, np.logical_and(dilated, ~gt_channel)).sum()

    return {
        "inside_tumor": float(inside / k),
        "peritumoral": float(shell / k),
        "elsewhere": float((k - inside - shell) / k),
    }


# Output-first order: randomising the last layer should already destroy a
# faithful map, and each further step should destroy it more.
RANDOMIZATION_ORDER = [
    "decoder0_header",
    "decoder3_upsampler",
    "decoder6_upsampler",
    "decoder9_upsampler",
    "dilated_bottleneck",
    "transformer",
]


def _reinitialize(module):
    """Re-initialise every submodule that knows how, i.e. draw fresh weights
    from the same distribution training started from."""
    for sub in module.modules():
        if hasattr(sub, "reset_parameters"):
            sub.reset_parameters()


def sanity_check_randomization(model, image, label, class_idx, cfg, layer_path,
                               method, order=None):
    """Cascading model-parameter randomisation (Adebayo et al., NeurIPS 2018).

    Randomise the decoder from the output backwards, recomputing the CAM after
    each step, and measure SSIM against the map from the trained model. A
    faithful attribution must decay towards noise. If it does NOT — if the map
    survives having the weights destroyed — then the method is responding to
    image structure, i.e. it is an edge detector wearing an explanation's
    clothes, and every conclusion drawn from it is void.

    This is the cheapest way to pre-empt the most common reviewer objection to
    a saliency-based XAI section, and it is the one experiment that can
    invalidate the rest.

    Mutates the model, so callers must reload the checkpoint afterwards.
    """
    from skimage.metrics import structural_similarity

    reference, _ = cam_for_sample(model, image, label, class_idx, cfg, layer_path, method)
    results = []

    for name in (order or RANDOMIZATION_ORDER):
        _reinitialize(_resolve_layer(model, name))
        current, _ = cam_for_sample(model, image, label, class_idx, cfg, layer_path, method)
        results.append({
            "randomized_through": name,
            "ssim_vs_trained": float(structural_similarity(
                reference, current, data_range=1.0)),
        })

    return results


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def informative_slices(gt_channel, n_slices=5, min_gap=8):
    """Pick slices along the depth axis with the most ground-truth voxels,
    spaced apart so the panels are not five views of one lesion."""
    per_slice = gt_channel.sum(axis=(0, 1))
    if per_slice.sum() == 0:
        per_slice = np.ones_like(per_slice)

    chosen = []
    for idx in np.argsort(per_slice)[::-1]:
        if all(abs(int(idx) - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == n_slices:
            break
    return sorted(chosen)


def save_overlay(mri, attribution, gt_channel, title, out_path, cmap="inferno"):
    """Three-row figure: T1ce, attribution overlay, ground-truth contour.

    Same dark styling as the existing attention figures so the paper's plates
    stay visually consistent.
    """
    slices = informative_slices(gt_channel)
    fig, axes = plt.subplots(3, len(slices), figsize=(4.4 * len(slices), 13),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.03})
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=0.92)

    if len(slices) == 1:
        axes = axes.reshape(3, 1)

    image = None
    for col, sl in enumerate(slices):
        base = minmax_normalize(mri[:, :, sl])
        attr = minmax_normalize(attribution[:, :, sl])

        axes[0, col].imshow(base, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Slice {sl}", color="#aaaaaa", fontsize=9, pad=3)

        axes[1, col].imshow(base, cmap="gray", origin="lower")
        image = axes[1, col].imshow(attr, cmap=cmap, alpha=0.55, origin="lower",
                                    vmin=0, vmax=1)

        axes[2, col].imshow(base, cmap="gray", origin="lower")
        if gt_channel[:, :, sl].any():
            axes[2, col].contour(gt_channel[:, :, sl].astype(float),
                                 levels=[0.5], colors="#00e5ff", linewidths=1.2)

        for row in range(3):
            axes[row, col].axis("off")

    if image is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.5])
        bar = fig.colorbar(image, cax=cbar_ax)
        bar.set_label("Attribution", color="white", fontsize=9)
        plt.setp(bar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    for row, label in enumerate(["T1ce", "Attribution", "Ground truth"]):
        fig.text(0.085, 0.76 - row * 0.26, label, color="white", fontsize=10,
                 fontweight="bold", va="center", rotation=90)

    plt.savefig(out_path, bbox_inches="tight", facecolor="#0d0d0d", dpi=150)
    plt.close()


def save_curve(curves, labels, x_values, title, xlabel, ylabel, out_path):
    """Line plot for the deletion / error-retention curves."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve, label in zip(curves, labels):
        ax.plot(x_values, curve, marker="o", linewidth=1.8, markersize=4, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def save_modality_heatmap(attribution, out_path):
    """4x3 heatmap of Dice drop per (modality, region)."""
    matrix = np.array([attribution["delta"][name] for name in MODALITY_NAMES])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-abs(matrix).max(),
                      vmax=abs(matrix).max())
    ax.set_xticks(range(3), CHANNEL_NAMES)
    ax.set_yticks(range(len(MODALITY_NAMES)), MODALITY_NAMES)
    ax.set_title("Dice change when a modality is removed")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center",
                    fontsize=10)

    fig.colorbar(image, ax=ax, label="ΔDice")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# orchestration — shared by xai.py (standalone) and train.py (end of a run)
# ---------------------------------------------------------------------------

def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def _reload(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def component_cam(model, samples, out_dir, cfg):
    """X1 — class-specific attribution at several decoder depths."""
    results = {}
    for sample_idx, image, label in samples:
        gt = label[0].cpu().numpy() > 0.5
        mri = image[0, 2].cpu().numpy()          # T1ce, the ET-defining sequence

        for class_idx, class_name in enumerate(CHANNEL_NAMES):
            for layer_path in cfg.xai.cam_layers:
                for method in cfg.xai.cam_methods:
                    cam, _ = cam_for_sample(
                        model, image, label, class_idx, cfg, layer_path, method)
                    tag = f"s{sample_idx}_{class_name}_{layer_path.replace('.', '')}_{method}"
                    save_overlay(
                        mri, cam, gt[class_idx],
                        f"{class_name}  |  {method.upper()}-CAM  |  {layer_path}",
                        os.path.join(out_dir, f"cam_{tag}.png"),
                    )
                    results[tag] = {
                        "mass_in_tumor": float(cam[gt[class_idx]].sum() / max(cam.sum(), 1e-9)),
                        "peak_value": float(cam.max()),
                    }
    return results


def component_modality(model, loader, device, out_dir, cfg, inferer):
    """X2 — the radiologically checkable result."""
    attribution = modality_attribution(
        model, loader, device, cfg, inferer=inferer,
        thresholds=cfg.infer.thresholds,
    )
    save_modality_heatmap(attribution, os.path.join(out_dir, "modality_attribution.png"))

    print("  Dice change when a modality is removed (TC / WT / ET):")
    for name in MODALITY_NAMES:
        print(f"    -{name:<6}" + "".join(f"{d:>9.4f}" for d in attribution["delta"][name]))
    return attribution


def component_uncertainty(model, samples, out_dir, cfg):
    """X3 — MC-dropout maps plus the curve that shows they are useful."""
    fractions = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = {}

    for sample_idx, image, label in samples:
        mean_prob, std, entropy, n_dropout = mc_dropout_predict(
            model, image, cfg.xai.mc_passes)
        gt = label[0].cpu().numpy() > 0.5
        mri = image[0, 2].cpu().numpy()

        for class_idx, class_name in enumerate(CHANNEL_NAMES):
            pred = mean_prob[class_idx] > cfg.infer.thresholds[class_idx]
            curve = error_retention_curve(pred, gt[class_idx], entropy[class_idx], fractions)

            save_overlay(
                mri, entropy[class_idx], gt[class_idx],
                f"{class_name}  |  MC-dropout predictive entropy "
                f"({cfg.xai.mc_passes} passes)",
                os.path.join(out_dir, f"uncertainty_s{sample_idx}_{class_name}.png"),
                cmap="viridis",
            )
            save_curve(
                [curve], ["MC-dropout entropy"], fractions,
                f"{class_name} — error retention",
                "Fraction of most-uncertain voxels referred", "Dice",
                os.path.join(out_dir, f"retention_s{sample_idx}_{class_name}.png"),
            )

            results[f"s{sample_idx}_{class_name}"] = {
                "dice_mc_mean": dice(pred, gt[class_idx]),
                "retention_curve": curve,
                "retention_fractions": fractions,
                "mean_entropy": float(entropy[class_idx].mean()),
                "mean_epistemic_std": float(std[class_idx].mean()),
                "n_dropout_layers": n_dropout,
            }
    return results


def component_rollout(model, samples, out_dir, cfg):
    """X4 — the upgrade to the existing raw-attention figure."""
    results = {}
    for sample_idx, image, label in samples:
        relevance = attention_rollout(model, image, cfg)
        if relevance is None:
            print("  attention cache empty — skipping rollout")
            continue

        gt = label[0].cpu().numpy() > 0.5
        save_overlay(
            image[0, 2].cpu().numpy(), relevance, gt[1],
            "Attention rollout (all 12 blocks, residual-corrected)",
            os.path.join(out_dir, f"rollout_s{sample_idx}.png"),
        )
        results[f"s{sample_idx}"] = {
            name: float(relevance[gt[c]].mean()) if gt[c].any() else None
            for c, name in enumerate(CHANNEL_NAMES)
        }
    return results


def component_faithful(model, samples, checkpoint_path, device, out_dir, cfg):
    """X5 — the part that makes X1/X4 falsifiable claims instead of pictures."""
    fractions = cfg.xai.deletion_fractions
    layer_path = cfg.xai.cam_layers[-1]
    results = {}

    for sample_idx, image, label in samples:
        gt = label[0].cpu().numpy() > 0.5

        for class_idx, class_name in enumerate(CHANNEL_NAMES):
            curves, labels, entry = [], [], {}

            for method in cfg.xai.cam_methods:
                cam, _ = cam_for_sample(
                    model, image, label, class_idx, cfg, layer_path, method)
                curve = deletion_curve(
                    model, image, gt[class_idx], cam, class_idx, fractions,
                    threshold=cfg.infer.thresholds[class_idx])
                curves.append(curve)
                labels.append(f"{method}-CAM")
                entry[method] = {
                    "deletion_curve": curve,
                    "deletion_auc": auc(curve, fractions),
                    "localization": localization_scores(
                        cam, gt[class_idx], cfg.xai.localization_top_frac,
                        cfg.xai.peritumoral_radius),
                }

            relevance = attention_rollout(model, image, cfg)
            if relevance is not None:
                curve = deletion_curve(
                    model, image, gt[class_idx], relevance, class_idx, fractions,
                    threshold=cfg.infer.thresholds[class_idx])
                curves.append(curve)
                labels.append("attention rollout")
                entry["rollout"] = {
                    "deletion_curve": curve,
                    "deletion_auc": auc(curve, fractions),
                    "localization": localization_scores(
                        relevance, gt[class_idx], cfg.xai.localization_top_frac,
                        cfg.xai.peritumoral_radius),
                }

            baseline = random_baseline_curve(
                model, image, gt[class_idx], class_idx, fractions,
                threshold=cfg.infer.thresholds[class_idx])
            curves.append(baseline)
            labels.append("random (null)")
            entry["random"] = {
                "deletion_curve": baseline,
                "deletion_auc": auc(baseline, fractions),
            }

            save_curve(
                curves, labels, fractions,
                f"{class_name} — deletion faithfulness (lower is better)",
                "Fraction of highest-attributed voxels deleted", "Dice",
                os.path.join(out_dir, f"deletion_s{sample_idx}_{class_name}.png"),
            )
            results[f"s{sample_idx}_{class_name}"] = entry

    # Randomisation destroys the weights, so it runs last and the checkpoint is
    # reloaded immediately after.
    _, image, label = samples[0]
    method = cfg.xai.cam_methods[0]
    results["sanity_check"] = {
        "layer": layer_path,
        "method": method,
        "class": CHANNEL_NAMES[2],
        "cascade": sanity_check_randomization(
            model, image, label, 2, cfg, layer_path, method),
    }
    _reload(model, checkpoint_path, device)

    print("  Cascading randomisation (SSIM vs trained map — must decay):")
    for step in results["sanity_check"]["cascade"]:
        print(f"    through {step['randomized_through']:<20} "
              f"SSIM={step['ssim_vs_trained']:.4f}")

    return results


def run_xai_suite(model, test_ds, test_loader, checkpoint_path, out_dir, device,
                  cfg, inferer, components=COMPONENTS):
    """Run the selected XAI components and write everything into `out_dir`.

    Shared entry point: xai.py calls this against a finished run, and train.py
    calls it at the end of a fresh one so a single `python train.py` produces
    the whole artifact set. `inferer(model, x) -> logits` is injected rather
    than imported so this module stays independent of utils.engine.

    The model is left holding the checkpoint weights on return — component
    "faithful" randomises them and reloads.
    """
    os.makedirs(out_dir, exist_ok=True)
    _reload(model, checkpoint_path, device)

    samples = []
    for idx in cfg.xai.sample_indices:
        if idx < len(test_ds):
            image, label = prepare_sample(test_ds[idx], cfg, device)
            samples.append((idx, image, label))
    if not samples:
        print("[XAI] no test samples selected — check cfg.xai.sample_indices")
        return {}

    print(f"[XAI] explaining {len(samples)} test sample(s) -> {out_dir}")

    runners = {
        "cam": lambda: component_cam(model, samples, out_dir, cfg),
        "modality": lambda: component_modality(model, test_loader, device, out_dir,
                                               cfg, inferer),
        "uncertainty": lambda: component_uncertainty(model, samples, out_dir, cfg),
        "rollout": lambda: component_rollout(model, samples, out_dir, cfg),
        "faithful": lambda: component_faithful(model, samples, checkpoint_path,
                                               device, out_dir, cfg),
    }

    summary = {}
    # Iterate COMPONENTS, not `components`: "faithful" randomises the weights,
    # so it must run last whatever order the caller asked for.
    for name in [c for c in COMPONENTS if c in components]:
        print(f"\n[XAI: {name}]")
        try:
            summary[name] = runners[name]()
            _save_json(summary[name], os.path.join(out_dir, f"{name}.json"))
        except Exception as exc:                                  # noqa: BLE001
            # A failed explanation must never discard a finished training run.
            print(f"[XAI: {name}] FAILED — {type(exc).__name__}: {exc}")
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"}
            _reload(model, checkpoint_path, device)

    _save_json(summary, os.path.join(out_dir, "summary.json"))
    print(f"[XAI] complete -> {out_dir}")
    return summary

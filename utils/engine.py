import gc
import os
import time

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from tqdm import tqdm

from utils.attention import (
    register_attention_hook, save_attention_overlay, save_attention_evolution,
    set_attention_capture,
)
from utils.checkpoint import load_training_state, save_training_state
from utils.losses import main_and_aux_losses
from utils.metrics import (
    compute_hd95, compute_sensitivity, compute_iou, compute_miou,
    compute_confusion, compute_specificity, compute_f1, compute_roc_auc,
)
from utils.postprocess import binarize, postprocess
from utils.plot import plot_test_qualitative
from utils.progress import RunTimeEstimator, format_duration
from utils.transforms import fit_to_size


def free_gpu_cache(collect=True):
    """Collect unreachable CUDA tensors, then return cached blocks to the driver.

    Called at the train/validation phase boundaries, and the order matters:
    empty_cache() only releases blocks nothing references, so collecting first
    is what makes it worth calling.

    `gc.collect()` is not redundant with refcounting here. A tensor caught in a
    reference cycle is freed only by the cycle collector, and CPython triggers
    that on object COUNTS, not bytes — so a few dozen whole-brain tensors can
    sit unreachable-but-uncollected indefinitely while being GB of VRAM. That is
    visible in this pipeline: a census taken during epoch N+1 still finds epoch
    N's validation-volume tensors live (measured: +1.36GB carried between
    cycles).

    empty_cache() then addresses the other half. Training allocates a fixed
    128x128x96 batch every step; validation allocates a different per-patient
    whole-brain shape every sample. Interleaved in one cached pool they strand
    memory in partially-used segments (measured: 5-7GB non-releasable), which is
    how a ~21GB working set fails to fit on a 48GB card by epoch 34.

    Costs a few hundred ms per epoch — against a ~41-minute epoch, free.
    """
    if collect:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_mem_note():
    """Compact allocator state for the per-epoch summary line. `reserved` minus
    `alloc` is the cached-and-idle pool; `retries` is the number of times the
    allocator had to flush that pool and retry a failed allocation, which is the
    early-warning signal that a run is drifting toward OOM."""
    if not torch.cuda.is_available():
        return ""
    stats = torch.cuda.memory_stats()
    gb = 1024 ** 3
    return (f"gpu_mem alloc={torch.cuda.memory_allocated()/gb:.1f}G "
            f"reserved={torch.cuda.memory_reserved()/gb:.1f}G "
            f"peak={torch.cuda.max_memory_allocated()/gb:.1f}G "
            f"retries={stats.get('num_alloc_retries', 0)} "
            f"ooms={stats.get('num_ooms', 0)}")


def print_gpu_info():
    if torch.cuda.is_available():
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")


def get_device():
    device = torch.device("cpu:0")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    print(device)
    return device


def build_model(cfg, device):
    from models.unetr import UNETR

    encoder = cfg.unetr.get("encoder", "vit")
    if encoder == "mamba" and torch.device(device).type == "cuda":
        from blocks.VisionMamba import mamba_kernels_available
        if not mamba_kernels_available():
            # Fail at startup, not at the first training step after the
            # dataloaders have spent minutes warming up.
            raise SystemExit(
                "cfg.unetr.encoder='mamba' needs the mamba_ssm CUDA kernels, which are "
                "not importable in this Python env. Activate the `mamba` conda env "
                "(conda activate mamba) and rerun, or pass --encoder vit.")

    model = UNETR(
        img_shape=cfg.unetr.img_shape,
        input_dim=cfg.unetr.input_dim,
        output_dim=cfg.unetr.output_dim,
        embed_dim=cfg.unetr.embed_dim,
        patch_size=cfg.unetr.patch_size,
        num_heads=cfg.unetr.num_heads,
        dropout=cfg.unetr.dropout,
        encoder=encoder,
        mamba_kwargs=dict(cfg.mamba) if encoder == "mamba" else None,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for name in model.encoder_module_names()
                         for p in getattr(model, name).parameters())
    print(f"Encoder: {encoder} | Trainable parameters: {total_params/1e6:.2f}M "
          f"(encoder {encoder_params/1e6:.2f}M)")

    input_res = (cfg.unetr.input_dim, *cfg.unetr.img_shape)
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_res=input_res, as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        print(f"FLOPs: {macs} | Params: {params}")
    except Exception as pt_exc:
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy_input = torch.zeros((1, *input_res)).to(next(model.parameters()).device)
            flops = FlopCountAnalysis(model, dummy_input)
            print(f"FLOPs: {flops.total()/1e9:.2f}G")
        except Exception:
            print(f"FLOPs: unavailable ({pt_exc})")
    if encoder == "mamba":
        print("  note: ptflops/fvcore cannot see the fused selective-scan kernels, so the "
              "Mamba FLOP count is an undercount; compare min/epoch instead")

    return model


def apply_run_model_config(cfg, run_dir):
    """Point cfg at the encoder a finished run was TRAINED with.

    evaluate.py and xai.py rebuild the model from config.py, which describes
    the NEXT run, not necessarily the one being re-read. The run's own
    config_snapshot.json is the record; runs from before the encoder switch
    have no `unetr.encoder` key and were all ViT.
    """
    import json

    snapshot = {}
    path = os.path.join(run_dir, "config_snapshot.json")
    if os.path.exists(path):
        with open(path) as f:
            snapshot = json.load(f)
    else:
        print(f"[model] no config_snapshot.json in {run_dir}; assuming encoder='vit'")

    cfg.unetr.encoder = snapshot.get("unetr", {}).get("encoder", "vit")
    if cfg.unetr.encoder == "mamba":
        for key, value in snapshot.get("mamba", {}).items():
            cfg.mamba[key] = value
    print(f"[model] {run_dir}: encoder={cfg.unetr.encoder!r} (from its config_snapshot.json)")
    return cfg.unetr.encoder


def build_lr_scheduler(optimizer, cfg):
    """Linear warmup then cosine decay. Warmup matters more now that training
    runs long enough for the cosine tail to be reached."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epoch - cfg.warmup_epochs, 1), eta_min=1e-6,
    )
    if cfg.warmup_epochs <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=cfg.warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs],
    )


def optimizer_param_groups(model):
    """AdamW parameter groups. Parameters flagged `_no_weight_decay` (the
    Mamba encoder's A_log and D, following Mamba's own convention) go in a
    weight_decay=0 group. A model with none (the ViT) gets exactly one group
    holding model.parameters() in order, i.e. the optimizer it always had, so
    its existing last.pth files still resume."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if getattr(p, "_no_weight_decay", False) else decay).append(p)
    groups = [{"params": decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_training_components(model, cfg):
    # AdamW decouples weight decay from the gradient update; plain Adam with
    # weight_decay applies it as coupled L2, which interacts badly with the
    # per-parameter LR scaling.
    optimizer = torch.optim.AdamW(optimizer_param_groups(model), cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, cfg)

    # torch ships EMA — no reason to hand-roll one. No BatchNorm anywhere in
    # this model (GroupNorm/LayerNorm only), so there are no running stats
    # needing a post-hoc update_bn pass over the training set.
    ema_model = None
    if cfg.ema_decay > 0:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.ema_decay),
        )

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    # Per-channel thresholds instead of MONAI's scalar-only AsDiscrete. The
    # per-epoch loop deliberately does NOT run the connected-component
    # cleanup: its thresholds are tuned on this same split, so folding them in
    # here would let post-processing hyperparameters pick the checkpoint.
    def post_trans(logits):
        return binarize(logits, cfg.infer.thresholds)

    scaler = torch.amp.GradScaler('cuda')
    torch.backends.cudnn.benchmark = True
    return optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans, scaler, ema_model


# The 8 axis-flip combinations over the spatial dims of a (B, C, H, W, D)
# tensor. A flip is its own inverse, so the same dims undo it on the way back.
_TTA_FLIPS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


def run_inference(model, inputs, cfg, tta=None, overlap=None):
    """Sliding-window inference, always returning LOGITS so every caller keeps
    its own sigmoid/threshold — there is exactly one contract here, no
    "sometimes probabilities" mode to trip over.

    With TTA the 8 flip predictions are averaged in PROBABILITY space
    (averaging logits would let the single most over-confident flip dominate)
    and mapped back through logit(), the exact inverse of sigmoid — so
    downstream thresholding and AUC are unaffected by the round trip.

    `tta=None` defers to cfg.infer.tta_flips. The per-epoch validation loop
    passes tta=False explicitly: 8x inference on every epoch would cost more
    than the training step it is meant to be checking.

    `overlap=None` defers to cfg.infer.sw_overlap (test, threshold tuning). The
    per-epoch validation loop passes cfg.infer.val_sw_overlap, for the same
    reason it skips TTA.
    """
    sw_overlap = cfg.infer.sw_overlap if overlap is None else overlap

    def _sliding_window(x):
        def _compute():
            return sliding_window_inference(
                inputs=x,
                roi_size=cfg.unetr.img_shape,
                sw_batch_size=1,
                predictor=model,
                overlap=sw_overlap,
                mode=cfg.infer.sw_mode,
            )

        if cfg.val_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                return _compute()
        return _compute()

    use_tta = cfg.infer.tta_flips if tta is None else tta
    if not use_tta:
        return _sliding_window(inputs)

    prob_sum = None
    for dims in _TTA_FLIPS:
        out = _sliding_window(torch.flip(inputs, dims) if dims else inputs)
        out = torch.sigmoid(out.float())
        if dims:
            out = torch.flip(out, dims)
        prob_sum = out if prob_sum is None else prob_sum + out

    prob = (prob_sum / len(_TTA_FLIPS)).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(prob)


def _recover_from_oom(optimizer=None, scaler=None):
    """Reclaim the memory a failed step was holding and make the next step runnable.

    MUST be called from OUTSIDE the `except` block. While the block is running,
    the live exception's traceback references every frame that was on the stack
    when the allocation failed — including `model.forward()`'s locals, which are
    precisely the activations being freed. Cleaning up in there frees almost
    nothing; the caller's job is to record a flag, drop its own tensor
    references, leave the handler, and then call this.

    Two more things have to be undone, and skipping either turns the next step
    into a different, more confusing failure:

      * the gradients — a backward that died part-way leaves some parameters
        with grads and some without, so the next step would apply an update
        computed from a fraction of the batch;
      * the GradScaler's per-optimizer bookkeeping — if the OOM landed between
        `unscale_()` and `step()`, the scaler still thinks this optimizer has
        been unscaled and the next `unscale_()` raises "unscale_() has already
        been called on this optimizer since the last update()". `update()`
        resets this dict by reassigning it; clearing it here is the same reset
        without needing a successful step to get there.
    """
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler._per_optimizer_states.clear()
    gc.collect()
    free_gpu_cache()


def _loss_terms(loss_fn):
    """The unweighted terms of loss_fn's most recent call as floats — empty for
    a loss that does not record them (CombinedLoss does, in last_terms)."""
    return {k: float(v) for k, v in (getattr(loss_fn, "last_terms", None) or {}).items()}


def _patient_id(ds, index):
    """Patient folder name of item `index`, or "" when the dataset does not
    expose file paths (the synthetic test datasets)."""
    try:
        item = ds.data[index]
        path = item.get("label") or item["image"][0]
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""
    return os.path.basename(os.path.dirname(str(path)))


def train_one_epoch(model, train_loader, train_ds, optimizer, scaler, loss_fn, device, cfg, epoch,
                     estimator=None, console=None, ema_model=None, step_logger=None):
    """One pass over train_loader; returns the mean total loss of the steps that
    ran. `step_logger`, when given, is called once per completed step with a
    TRAIN_STEPS_CSV_FIELDS row (run_training passes RunLogger.log_train_step)."""
    model.train()
    epoch_loss = 0
    step = 0
    oom_skips = 0
    steps_per_epoch = len(train_loader)
    # The scheduler steps once per epoch, so this is the LR of every step below.
    lr = optimizer.param_groups[0]["lr"]
    pbar = tqdm(
        train_loader, total=len(train_loader), file=console, dynamic_ncols=True,
        desc=f"Epoch {epoch + 1}/{cfg.epoch} [train]", leave=False,
    )
    for batch_data in pbar:
        step_start = time.time()
        inputs = labels = outputs = aux_z6 = aux_z3 = loss = None
        loss_main = loss_z6 = loss_z3 = None
        step_parts = {}
        oom_message = None
        try:
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            inputs = fit_to_size(inputs, cfg.unetr.img_shape)
            labels = fit_to_size(labels, cfg.unetr.img_shape)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs, aux_z6, aux_z3 = model(inputs)
                loss, loss_main, loss_z6, loss_z3 = main_and_aux_losses(
                    loss_fn, outputs, aux_z6, aux_z3, labels, cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                ema_model.update_parameters(model)
            loss_value = loss.item()
            if step_logger is not None:
                step_parts = {
                    "loss_main": loss_main.item(), "loss_aux_z6": loss_z6.item(),
                    "loss_aux_z3": loss_z3.item(), **_loss_terms(loss_fn),
                }
        except torch.cuda.OutOfMemoryError as exc:
            # Record and get out. The cleanup deliberately happens after the
            # handler exits — see _recover_from_oom on why doing it in here
            # frees almost nothing.
            oom_message = str(exc)
            inputs = labels = outputs = aux_z6 = aux_z3 = loss = None
            loss_main = loss_z6 = loss_z3 = None

        if oom_message is not None:
            # A single unlucky batch must not cost the run. Drop it, clean up,
            # and carry on — one skipped batch out of ~500 per epoch is far
            # cheaper than restarting from the last checkpoint. The counter is
            # the guard against papering over a genuine "this config no longer
            # fits", where every batch would OOM and the epoch would be a no-op.
            oom_skips += 1
            _recover_from_oom(optimizer, scaler)
            print(f"\n[OOM] epoch {epoch + 1} step {step + 1}: skipped batch "
                  f"({oom_skips}/{cfg.train_oom_skip_limit} allowed). {oom_message}")
            if oom_skips > cfg.train_oom_skip_limit:
                pbar.close()
                raise torch.cuda.OutOfMemoryError(
                    f"{oom_skips} OOM-skipped batches in epoch {epoch + 1} exceeds "
                    f"cfg.train_oom_skip_limit={cfg.train_oom_skip_limit}. "
                    f"Last error: {oom_message}"
                )
            continue

        step += 1
        epoch_loss += loss_value

        if step_logger is not None:
            step_logger({
                "epoch": epoch + 1, "step": step,
                "global_step": epoch * steps_per_epoch + step, "lr": lr,
                "step_time_sec": time.time() - step_start,
                "loss": loss_value, **step_parts,
            })

        if estimator is not None:
            estimator.record_train_step(time.time() - step_start)
            pbar.set_postfix_str(f"loss={loss_value:.4f}  run_eta={estimator.eta_string()}")
        else:
            pbar.set_postfix_str(f"loss={loss_value:.4f}")

        # Release this step's activations BEFORE the loader hands over the next
        # batch. Without this the previous outputs/inputs stay referenced while
        # the next batch is allocated, so the peak carries one extra step's
        # worth of full-resolution decoder tensors for no reason.
        del inputs, labels, outputs, aux_z6, aux_z3, loss, loss_main, loss_z6, loss_z3
    pbar.close()
    if oom_skips:
        print(f"[OOM] epoch {epoch + 1}: {oom_skips} batch(es) skipped")
    epoch_loss /= max(step, 1)
    return epoch_loss


_CHANNELS = ("tc", "wt", "et")
# Per-patient metrics in val_steps.csv, named as in METRICS_CSV_FIELDS.
_VAL_STEP_METRICS = ("dice", "iou", "miou", "hd95", "sens", "spec", "f1", "auc")

TRAIN_STEPS_CSV_FIELDS = [
    "epoch", "step", "global_step", "lr", "step_time_sec",
    "loss", "loss_main", "loss_aux_z6", "loss_aux_z3",
    "loss_dice", "loss_focal_tversky", "loss_hausdorff", "hd_scale",
]

VAL_STEPS_CSV_FIELDS = [
    "epoch", "sample_index", "patient", "step_time_sec",
    "val_loss", "loss_dice", "loss_focal_tversky", "loss_hausdorff", "hd_scale",
] + [key for metric in _VAL_STEP_METRICS
     for key in (*(f"{metric}_{c}" for c in _CHANNELS), f"mean_{metric}")]


def _val_step_row(epoch, sample_index, patient, step_time_sec, val_loss, loss_parts,
                  sample_dice, sample_metrics, with_auc):
    """One val_steps.csv row: this patient's loss and every per-region metric,
    plus the mean over the regions that are finite (Dice is NaN for a region
    absent from the ground truth, exactly as DiceMetric counts it)."""
    row = {"epoch": epoch + 1, "sample_index": sample_index, "patient": patient,
           "step_time_sec": step_time_sec, "val_loss": val_loss, **loss_parts}
    for name, values in {"dice": sample_dice, **sample_metrics}.items():
        if name == "auc" and not with_auc:
            continue  # only computed every cfg.metrics.auc_every_n_epochs
        values = np.asarray(values, dtype=np.float64)
        for c, channel in enumerate(_CHANNELS):
            row[f"{name}_{channel}"] = float(values[c])
        finite = values[np.isfinite(values)]
        row[f"mean_{name}"] = float(finite.mean()) if finite.size else float("nan")
    return row


def validate(model, val_loader, val_ds, loss_fn, dice_metric, dice_metric_batch, post_trans,
             device, cfg, epoch, attention_cache, attention_dir, estimator=None, console=None,
             step_logger=None):
    """The per-epoch validation pass; returns its metrics.csv row. `step_logger`,
    when given, also gets one VAL_STEPS_CSV_FIELDS row per patient (run_training
    passes RunLogger.log_val_step)."""
    model.eval()
    val_loss_epoch = 0.0
    val_steps = 0
    metric_sums = {
        "hd95": np.zeros(3, dtype=np.float64),
        "sens": np.zeros(3, dtype=np.float64),
        "iou": np.zeros(3, dtype=np.float64),
        "miou": np.zeros(3, dtype=np.float64),
        "spec": np.zeros(3, dtype=np.float64),
        "f1": np.zeros(3, dtype=np.float64),
        "auc": np.zeros(3, dtype=np.float64),
    }
    sample_count = 0
    attention_saved = False
    batch_size = getattr(val_loader, "batch_size", None) or 1
    compute_auc_this_epoch = ((epoch + 1) % cfg.metrics.auc_every_n_epochs == 0)
    attention_this_epoch = cfg.attention.enabled and ((epoch + 1) % cfg.attention.every_n_epochs == 0)

    # The hook that fills attention_cache runs on every sliding-window patch of
    # every validation sample. Cloning a (1, heads, patches, patches) tensor
    # each time — ~27 windows x ~180 patients x 50 epochs — is pure allocator
    # churn for a picture we save once every cfg.attention.every_n_epochs. Arm
    # it only on the epochs that actually save one.
    set_attention_capture(attention_cache, attention_this_epoch)

    pbar = tqdm(
        val_loader, total=len(val_loader), file=console, dynamic_ncols=True,
        desc=f"Epoch {epoch + 1}/{cfg.epoch} [val]", leave=False,
    )
    oom_skips = 0
    with torch.no_grad():
        for val_index, val_data in enumerate(pbar):
            step_start = time.time()
            val_inputs = val_labels = val_outputs = None
            oom_message = None
            try:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                # tta=False: the per-epoch loop runs every epoch, so 8x flip
                # inference here would cost more than the training it monitors.
                # Test/evaluate.py turn it on. The overlap is kept lower than
                # the test pass's for the same reason (cfg.infer.val_sw_overlap).
                val_outputs = run_inference(model, val_inputs, cfg, tta=False,
                                            overlap=cfg.infer.get("val_sw_overlap"))
                val_loss_value = loss_fn(val_outputs, val_labels).item()
                val_loss_parts = _loss_terms(loss_fn)
                val_loss_epoch += val_loss_value
                val_steps += 1
                # The raw probabilities are only read for AUC, which runs every
                # cfg.metrics.auc_every_n_epochs epochs. Copying a whole
                # foreground-cropped volume off the GPU on the other nine tenths
                # of epochs buys nothing.
                val_outputs_raw = ([i.detach().cpu() for i in decollate_batch(val_outputs)]
                                   if compute_auc_this_epoch
                                   else [None] * val_outputs.shape[0])
                val_outputs_disc = [post_trans(i) for i in decollate_batch(val_outputs)]
                val_labels_list = decollate_batch(val_labels)
                dice_metric(y_pred=val_outputs_disc, y=val_labels)
                # The per-call return is this batch's per-patient Dice, exactly
                # as the aggregate counts it (NaN where the ground truth is empty).
                sample_dice = dice_metric_batch(
                    y_pred=val_outputs_disc, y=val_labels).detach().cpu().numpy()
            except torch.cuda.OutOfMemoryError as exc:
                oom_message = str(exc)
                val_inputs = val_labels = val_outputs = None

            if oom_message is not None:
                # Validation feeds whole per-patient volumes, so the largest
                # brains in the split are the ones that OOM. Dropping one from
                # the epoch's averages is a rounding error on ~180 samples;
                # losing the run is not.
                oom_skips += 1
                _recover_from_oom()
                print(f"\n[OOM] epoch {epoch + 1} validation: skipped one sample "
                      f"({oom_skips}/{cfg.val_oom_skip_limit} allowed). {oom_message}")
                if oom_skips > cfg.val_oom_skip_limit:
                    pbar.close()
                    raise torch.cuda.OutOfMemoryError(
                        f"{oom_skips} OOM-skipped validation samples in epoch "
                        f"{epoch + 1} exceeds cfg.val_oom_skip_limit="
                        f"{cfg.val_oom_skip_limit}. Last error: {oom_message}"
                    )
                continue

            for b, (pred_raw, pred, gt) in enumerate(
                    zip(val_outputs_raw, val_outputs_disc, val_labels_list)):
                pred_np = pred.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()
                pred_prob_np = torch.sigmoid(pred_raw).numpy() if pred_raw is not None else None
                sample_metrics = {name: np.zeros(3, dtype=np.float64) for name in metric_sums}
                for c in range(3):
                    pred_c = pred_np[c] > 0.5
                    gt_c = gt_np[c] > 0.5
                    tp, fp, fn = compute_confusion(pred_c, gt_c)
                    sample_metrics["hd95"][c] = compute_hd95(pred_c, gt_c, cfg.metrics.voxel_spacing)
                    sample_metrics["sens"][c] = compute_sensitivity(tp, fn)
                    sample_metrics["iou"][c] = compute_iou(tp, fp, fn)
                    sample_metrics["miou"][c] = compute_miou(pred_c, gt_c)
                    sample_metrics["spec"][c] = compute_specificity(pred_c, gt_c)
                    sample_metrics["f1"][c] = compute_f1(tp, fp, fn)
                    if compute_auc_this_epoch:
                        sample_metrics["auc"][c] = compute_roc_auc(pred_prob_np[c], gt_c)
                for name, values in sample_metrics.items():
                    metric_sums[name] += values
                sample_count += 1

                if step_logger is not None:
                    # val_loss is the batch mean, i.e. this patient's loss at
                    # the pipeline's validation batch size of 1.
                    index = val_index * batch_size + b
                    step_logger(_val_step_row(
                        epoch, index, _patient_id(val_ds, index), time.time() - step_start,
                        val_loss_value, val_loss_parts, sample_dice[b], sample_metrics,
                        compute_auc_this_epoch))

            if attention_this_epoch and not attention_saved:
                # Validation volumes keep their per-patient cropped shape (no
                # fit_to_size like training batches get), so the attention map
                # must be upsampled to THIS sample's actual shape, not the
                # fixed cfg.unetr.img_shape — otherwise slice indices computed
                # from the map can fall outside the real volume.
                save_attention_overlay(
                    attention_cache, model, val_data, val_ds, epoch + 1,
                    tuple(val_inputs.shape[2:]), attention_dir,
                    sample_idx=cfg.attention.sample_idx,
                )
                attention_saved = True
                # Picture saved: stop capturing and drop the cached map so it
                # is not held for the remaining ~180 samples and every epoch in
                # between.
                set_attention_capture(attention_cache, False)

            if estimator is not None:
                estimator.record_val_step(time.time() - step_start)
                pbar.set_postfix_str(f"run_eta={estimator.eta_string()}")

            # val_labels_list holds views onto val_labels, so dropping the batch
            # tensor alone would not free it.
            del (val_inputs, val_labels, val_labels_list, val_outputs,
                 val_outputs_raw, val_outputs_disc)

            # See cfg.checkpoint.gc_every_n_val_steps: this loop leaves ~0.09GB
            # per sample of unreachable-but-uncollectable CUDA tensors, and the
            # cycle collector's own heuristic cannot see them.
            gc_every = cfg.checkpoint.gc_every_n_val_steps
            if gc_every and val_steps and val_steps % gc_every == 0:
                gc.collect()
    pbar.close()
    set_attention_capture(attention_cache, False)
    if oom_skips:
        print(f"[OOM] epoch {epoch + 1}: {oom_skips} validation sample(s) skipped")

    metric = dice_metric.aggregate().item()
    metric_batch = dice_metric_batch.aggregate()
    metric_tc, metric_wt, metric_et = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item()
    dice_metric.reset()
    dice_metric_batch.reset()

    val_loss_epoch = val_loss_epoch / max(val_steps, 1)

    if sample_count > 0:
        hd95_tc, hd95_wt, hd95_et = (metric_sums["hd95"] / sample_count).tolist()
        sens_tc, sens_wt, sens_et = (metric_sums["sens"] / sample_count).tolist()
        iou_tc, iou_wt, iou_et = (metric_sums["iou"] / sample_count).tolist()
        miou_tc, miou_wt, miou_et = (metric_sums["miou"] / sample_count).tolist()
        spec_tc, spec_wt, spec_et = (metric_sums["spec"] / sample_count).tolist()
        f1_tc, f1_wt, f1_et = (metric_sums["f1"] / sample_count).tolist()
        auc_tc, auc_wt, auc_et = (metric_sums["auc"] / sample_count).tolist()
    else:
        hd95_tc = hd95_wt = hd95_et = 0.0
        sens_tc = sens_wt = sens_et = 0.0
        iou_tc = iou_wt = iou_et = 0.0
        miou_tc = miou_wt = miou_et = 0.0
        spec_tc = spec_wt = spec_et = 0.0
        f1_tc = f1_wt = f1_et = 0.0
        auc_tc = auc_wt = auc_et = 0.0

    dice_et, dice_wt, dice_tc = metric_et, metric_wt, metric_tc
    mean_dice = float(np.mean([dice_et, dice_wt, dice_tc]))
    mean_hd95 = float(np.mean([hd95_et, hd95_wt, hd95_tc]))
    mean_sens = float(np.mean([sens_et, sens_wt, sens_tc]))
    mean_spec = float(np.mean([spec_et, spec_wt, spec_tc]))
    mean_iou = float(np.mean([iou_et, iou_wt, iou_tc]))
    mean_miou = float(np.mean([miou_et, miou_wt, miou_tc]))
    mean_f1 = float(np.mean([f1_et, f1_wt, f1_tc]))
    mean_auc = float(np.mean([auc_et, auc_wt, auc_tc]))

    return {
        "val_loss": val_loss_epoch,
        "mean_dice": mean_dice, "dice_tc": dice_tc, "dice_wt": dice_wt, "dice_et": dice_et,
        "mean_hd95": mean_hd95, "hd95_tc": hd95_tc, "hd95_wt": hd95_wt, "hd95_et": hd95_et,
        "mean_sens": mean_sens, "sens_tc": sens_tc, "sens_wt": sens_wt, "sens_et": sens_et,
        "mean_spec": mean_spec, "spec_tc": spec_tc, "spec_wt": spec_wt, "spec_et": spec_et,
        "mean_iou": mean_iou, "iou_tc": iou_tc, "iou_wt": iou_wt, "iou_et": iou_et,
        "mean_miou": mean_miou, "miou_tc": miou_tc, "miou_wt": miou_wt, "miou_et": miou_et,
        "mean_f1": mean_f1, "f1_tc": f1_tc, "f1_wt": f1_wt, "f1_et": f1_et,
        "mean_auc": mean_auc, "auc_tc": auc_tc, "auc_wt": auc_wt, "auc_et": auc_et,
    }


METRICS_CSV_FIELDS = [
    "epoch", "lr", "epoch_time_sec", "train_loss", "val_loss",
    "dice_tc", "dice_wt", "dice_et", "mean_dice",
    "hd95_tc", "hd95_wt", "hd95_et", "mean_hd95",
    "sens_tc", "sens_wt", "sens_et", "mean_sens",
    "spec_tc", "spec_wt", "spec_et", "mean_spec",
    "iou_tc", "iou_wt", "iou_et", "mean_iou",
    "miou_tc", "miou_wt", "miou_et", "mean_miou",
    "f1_tc", "f1_wt", "f1_et", "mean_f1",
    "auc_tc", "auc_wt", "auc_et", "mean_auc",
    "is_best",
]


def _format_metrics_table(row):
    header = f"{'':6}{'Dice':>8}{'F1':>8}{'HD95':>8}{'Sens':>8}{'Spec':>8}{'IoU':>8}{'mIoU':>8}{'AUC':>8}"
    lines = [header]
    for label, key in (("TC", "tc"), ("WT", "wt"), ("ET", "et"), ("Mean", "mean")):
        dice = row[f"{'mean_dice' if key == 'mean' else 'dice_' + key}"]
        f1 = row[f"{'mean_f1' if key == 'mean' else 'f1_' + key}"]
        hd95 = row[f"{'mean_hd95' if key == 'mean' else 'hd95_' + key}"]
        sens = row[f"{'mean_sens' if key == 'mean' else 'sens_' + key}"]
        spec = row[f"{'mean_spec' if key == 'mean' else 'spec_' + key}"]
        iou = row[f"{'mean_iou' if key == 'mean' else 'iou_' + key}"]
        miou = row[f"{'mean_miou' if key == 'mean' else 'miou_' + key}"]
        auc = row[f"{'mean_auc' if key == 'mean' else 'auc_' + key}"]
        lines.append(
            f"{label:<6}{dice:>8.3f}{f1:>8.3f}{hd95:>8.2f}{sens:>8.3f}{spec:>8.3f}{iou:>8.3f}{miou:>8.3f}{auc:>8.3f}"
        )
    return "\n".join(lines)


def _print_epoch_summary(epoch, cfg, row, best_metric, best_metric_epoch, estimator):
    header = f"Epoch {epoch + 1}/{cfg.epoch} | lr={row['lr']:.2e} | train_loss={row['train_loss']:.4f}"
    lines = [header]
    if "val_loss" in row:
        lines[-1] += f" | val_loss={row['val_loss']:.4f} | time={format_duration(row['epoch_time_sec'])}"
        lines.append(_format_metrics_table(row))
        best_note = " | new_best=True (model saved)" if row["is_best"] else ""
        lines.append(f"best_mean_dice={best_metric:.4f} at_epoch={best_metric_epoch}{best_note}")
    else:
        lines[-1] += f" | time={format_duration(row['epoch_time_sec'])}"
    lines.append(f"run_eta={estimator.eta_string()}")
    # Memory state on every epoch line. A run that is drifting toward OOM shows
    # it here — rising `reserved` with flat `alloc`, then a non-zero `retries` —
    # hours before it actually dies, which is the difference between diagnosing
    # this from the log and having to reproduce it.
    mem = gpu_mem_note()
    if mem:
        lines.append(mem)
    print("\n".join(lines))


def run_training(model, loaders, loss_fn, device, cfg, run_logger, resume_from=None):
    """Train, validating every cfg.val_interval epochs.

    `resume_from` is a path to a `last.pth` written by a previous attempt at
    this same run. Everything that defines where training is — weights, EMA,
    AdamW moments, the cosine schedule's position, the GradScaler's scale, the
    RNG streams, the best-metric bookkeeping — comes back from it, so a resumed
    epoch 34 is the epoch 34 the crashed run would have produced. Restoring
    only the weights would restart AdamW's second-moment estimates from zero
    and restart the LR schedule, which is a different (worse) experiment
    wearing the same run name.
    """
    train_loader, train_ds = loaders["train_loader"], loaders["train_ds"]
    val_loader, val_ds = loaders["val_loader"], loaders["val_ds"]

    (optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans, scaler,
     ema_model) = build_training_components(model, cfg)

    # Everything downstream of training — validation, attention hooks, the
    # saved checkpoint — looks at the EMA weights when EMA is enabled, so the
    # metrics that pick the best epoch describe the weights that get shipped.
    eval_model = model if ema_model is None else ema_model.module

    num_val_runs = sum(1 for e in range(cfg.epoch) if (e + 1) % cfg.val_interval == 0)
    estimator = RunTimeEstimator(
        total_train_steps=cfg.epoch * len(train_loader),
        total_val_steps=num_val_runs * len(val_loader),
    )

    start_epoch = 0
    best_metric = -1
    best_metric_epoch = -1
    not_improved_epoch = 0
    elapsed_before = 0.0

    if resume_from:
        state = load_training_state(
            resume_from, model=model, ema_model=ema_model, optimizer=optimizer,
            lr_scheduler=lr_scheduler, scaler=scaler, estimator=estimator, cfg=cfg,
            map_location=device,
        )
        start_epoch = state["start_epoch"]
        best_metric = state["best_metric"]
        best_metric_epoch = state["best_metric_epoch"]
        not_improved_epoch = state["not_improved_epoch"]
        elapsed_before = state["elapsed_sec"]
        print(f"Resumed from {resume_from} | next epoch {start_epoch + 1}/{cfg.epoch} "
              f"| best_mean_dice={best_metric:.4f} @ epoch {best_metric_epoch} "
              f"| {format_duration(elapsed_before)} already spent")
        if start_epoch >= cfg.epoch:
            print("Checkpoint is already at the epoch budget — nothing left to train.")
            return best_metric, best_metric_epoch, elapsed_before

    attention_cache = register_attention_hook(eval_model) if cfg.attention.enabled else {}

    run_logger.open_metrics_csv(METRICS_CSV_FIELDS, resume_from_epoch=start_epoch)
    run_logger.open_step_csvs(TRAIN_STEPS_CSV_FIELDS, VAL_STEPS_CSV_FIELDS,
                              resume_from_epoch=start_epoch)

    training = True

    total_start = time.time()
    for epoch in range(start_epoch, cfg.epoch):
        epoch_start = time.time()

        # anneal the Hausdorff term (loss_fn holds the schedule; no-op if absent)
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch, cfg.epoch)

        epoch_loss = train_one_epoch(
            model, train_loader, train_ds, optimizer, scaler, loss_fn, device, cfg, epoch,
            estimator=estimator, console=run_logger.console, ema_model=ema_model,
            step_logger=run_logger.log_train_step,
        )
        lr_scheduler.step()

        row = {
            "epoch": epoch + 1,
            "lr": lr_scheduler.get_last_lr()[0],
            "train_loss": epoch_loss,
            "is_best": False,
        }

        # Hand the validation phase an uncarved pool. Training allocates a fixed
        # 128x128x96 shape; validation allocates a different, per-patient shape
        # for every sample. Interleaving the two in one cached pool is what
        # strands memory in partially-used segments (measured: 5-7GB
        # non-releasable), and that is the failure v2-run2 hit at epoch 34 with
        # 48GB of card and a ~21GB working set.
        free_gpu_cache()

        if (epoch + 1) % cfg.val_interval == 0:
            val_metrics = validate(
                eval_model, val_loader, val_ds, loss_fn, dice_metric, dice_metric_batch,
                post_trans, device, cfg, epoch, attention_cache, run_logger.attention_dir,
                estimator=estimator, console=run_logger.console,
                step_logger=run_logger.log_val_step,
            )
            row.update(val_metrics)
            metric = val_metrics["mean_dice"]

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                not_improved_epoch = 0
                row["is_best"] = True
                torch.save(eval_model.state_dict(), run_logger.checkpoint_path)
            else:
                not_improved_epoch += 1
                if not_improved_epoch >= cfg.patience:
                    training = False

        free_gpu_cache()

        row["epoch_time_sec"] = time.time() - epoch_start
        run_logger.log_epoch_metrics(row)

        # Written AFTER metrics.csv so the two can only disagree in the safe
        # direction: a crash between them replays one epoch, whereas the
        # reverse order would resume past an epoch whose row never landed.
        if cfg.checkpoint.save_last_every_epoch:
            save_training_state(
                run_logger.last_checkpoint_path,
                epoch=epoch, model=model, ema_model=ema_model, optimizer=optimizer,
                lr_scheduler=lr_scheduler, scaler=scaler, best_metric=best_metric,
                best_metric_epoch=best_metric_epoch, not_improved_epoch=not_improved_epoch,
                estimator=estimator,
                elapsed_sec=elapsed_before + (time.time() - total_start),
                cfg=cfg,
            )

        _print_epoch_summary(epoch, cfg, row, best_metric, best_metric_epoch, estimator)

        if not training:
            print("Training stopped early: no improvement within patience window.")
            break

    total_time = elapsed_before + (time.time() - total_start)
    print(f"Train completed | best_metric: {best_metric:.4f} @ epoch {best_metric_epoch} | total time: {format_duration(total_time)}")

    if cfg.attention.enabled:
        save_attention_evolution(run_logger.attention_dir)

    free_gpu_cache()
    return best_metric, best_metric_epoch, total_time


def run_test(model, loaders, loss_fn, device, cfg, run_logger):
    test_loader, test_ds = loaders["test_loader"], loaders["test_ds"]

    model.load_state_dict(torch.load(run_logger.checkpoint_path))
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    # Unlike the per-epoch validation loop, the final test pass applies the
    # full inference recipe: tuned per-channel thresholds AND the
    # connected-component cleanup that suppresses stray ET blobs.
    def post_trans(logits):
        return postprocess(binarize(logits, cfg.infer.thresholds), cfg)

    print(f"Test inference | thresholds={tuple(cfg.infer.thresholds)} "
          f"| tta_flips={cfg.infer.tta_flips} | sw_overlap={cfg.infer.sw_overlap} "
          f"({cfg.infer.sw_mode}) | min_component={tuple(cfg.infer.min_component_voxels)} "
          f"| min_total={tuple(cfg.infer.min_total_voxels)}")

    sample = test_ds[0]
    print("image shape:", sample["image"].shape)
    print("label shape:", sample["label"].shape)

    with torch.no_grad():
        val_input = sample["image"].unsqueeze(0).to(device)
        val_output = run_inference(model, val_input, cfg)
        val_output = post_trans(val_output[0])
        plot_test_qualitative(sample, val_output, run_logger.testing_vis_dir)

    metric_sums = {
        "hd95": np.zeros(3, dtype=np.float64),
        "hd95_clean": np.zeros(3, dtype=np.float64),
        "sens": np.zeros(3, dtype=np.float64),
        "iou": np.zeros(3, dtype=np.float64),
    }
    sample_count = 0

    # The mean HD95 averages two populations that have nothing to do with each
    # other. compute_hd95 returns the 374.0 sentinel whenever exactly one of
    # {pred, gt} is empty, so on a channel like ET — where a good fraction of
    # patients simply have no enhancing tumour — the reported millimetres are
    # mostly a PATIENT COUNT in disguise: a couple of empty-mismatch cases
    # outweigh every correctly segmented boundary in the split. Counting them
    # separately, and averaging HD95 over the both-non-empty cases only, splits
    # "how many patients did we get categorically wrong" from "how good are the
    # boundaries when we get it right" — two numbers, both readable.
    # Split by direction, because the two need opposite fixes: a hallucination
    # (we found a tumour that is not there) wants min_total_voxels RAISED, a
    # miss (we deleted or never found a real one) wants it LOWERED. Lumping
    # them together is how you tune that knob in the wrong direction.
    n_halluc = np.zeros(3, dtype=np.int64)       # pred non-empty, gt empty
    n_miss = np.zeros(3, dtype=np.int64)         # pred empty, gt non-empty
    n_clean = np.zeros(3, dtype=np.int64)        # both non-empty

    dice_metric.reset()
    dice_metric_batch.reset()

    pbar = tqdm(test_loader, total=len(test_loader), file=run_logger.console,
                dynamic_ncols=True, desc="Test", leave=False)
    with torch.no_grad():
        for test_data in pbar:
            test_inputs, test_labels = (
                test_data["image"].to(device),
                test_data["label"].to(device),
            )
            test_outputs = run_inference(model, test_inputs, cfg)
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            test_labels_list = decollate_batch(test_labels)

            dice_metric(y_pred=test_outputs, y=test_labels_list)
            dice_metric_batch(y_pred=test_outputs, y=test_labels_list)

            for pred, gt in zip(test_outputs, test_labels_list):
                pred_np = pred.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()
                for c in range(3):
                    pred_c = pred_np[c] > 0.5
                    gt_c = gt_np[c] > 0.5
                    tp, fp, fn = compute_confusion(pred_c, gt_c)
                    metric_sums["sens"][c] += compute_sensitivity(tp, fn)
                    metric_sums["iou"][c] += compute_iou(tp, fp, fn)
                    hd = compute_hd95(pred_c, gt_c, cfg.metrics.voxel_spacing)
                    metric_sums["hd95"][c] += hd
                    if pred_c.any() and not gt_c.any():
                        n_halluc[c] += 1
                    elif gt_c.any() and not pred_c.any():
                        n_miss[c] += 1
                    elif gt_c.any():          # both non-empty: a real distance
                        metric_sums["hd95_clean"][c] += hd
                        n_clean[c] += 1
                sample_count += 1
    pbar.close()

    metric = dice_metric.aggregate().item()
    metric_batch = dice_metric_batch.aggregate()
    dice_metric.reset()
    dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item()

    if sample_count > 0:
        hd95_tc, hd95_wt, hd95_et = (metric_sums["hd95"] / sample_count).tolist()
        sens_tc, sens_wt, sens_et = (metric_sums["sens"] / sample_count).tolist()
        iou_tc, iou_wt, iou_et = (metric_sums["iou"] / sample_count).tolist()
    else:
        hd95_tc = hd95_wt = hd95_et = 0.0
        sens_tc = sens_wt = sens_et = 0.0
        iou_tc = iou_wt = iou_et = 0.0

    dice_et, dice_wt, dice_tc = metric_et, metric_wt, metric_tc
    mean_dice = float(np.mean([dice_et, dice_wt, dice_tc]))
    mean_hd95 = float(np.mean([hd95_et, hd95_wt, hd95_tc]))
    mean_sens = float(np.mean([sens_et, sens_wt, sens_tc]))
    mean_iou = float(np.mean([iou_et, iou_wt, iou_tc]))

    print(
        f"Test | ET: Dice={dice_et:.3f} HD95={hd95_et:.2f} Sens={sens_et:.3f} IoU={iou_et:.3f} "
        f"| WT: Dice={dice_wt:.3f} HD95={hd95_wt:.2f} Sens={sens_wt:.3f} IoU={iou_wt:.3f} "
        f"| TC: Dice={dice_tc:.3f} HD95={hd95_tc:.2f} Sens={sens_tc:.3f} IoU={iou_tc:.3f} "
        f"| Mean: Dice={mean_dice:.3f} HD95={mean_hd95:.2f} Sens={mean_sens:.3f} IoU={mean_iou:.3f} mIoU={mean_iou:.3f}"
    )
    hd95_clean = np.divide(metric_sums["hd95_clean"], n_clean,
                           out=np.zeros(3), where=n_clean > 0).tolist()
    print(
        f"HD95 breakdown (n={sample_count}) | "
        + " | ".join(
            f"{name}: {n_halluc[c]} hallucinated + {n_miss[c]} missed "
            f"(374.0 each), {hd95_clean[c]:.2f}mm over the {n_clean[c]} clean cases"
            for c, name in enumerate(("TC", "WT", "ET"))
        )
    )
    print("Metric on test image: ", metric)
    print(f"metric_tc: {metric_tc:.4f}")
    print(f"metric_wt: {metric_wt:.4f}")
    print(f"metric_et: {metric_et:.4f}")

    test_row = {
        "dice_tc": dice_tc, "dice_wt": dice_wt, "dice_et": dice_et, "mean_dice": mean_dice,
        "hd95_tc": hd95_tc, "hd95_wt": hd95_wt, "hd95_et": hd95_et, "mean_hd95": mean_hd95,
        "sens_tc": sens_tc, "sens_wt": sens_wt, "sens_et": sens_et, "mean_sens": mean_sens,
        "iou_tc": iou_tc, "iou_wt": iou_wt, "iou_et": iou_et, "mean_iou": mean_iou,
        "hd95_tc_clean": hd95_clean[0], "hd95_wt_clean": hd95_clean[1],
        "hd95_et_clean": hd95_clean[2],
        "n_halluc_tc": int(n_halluc[0]), "n_halluc_wt": int(n_halluc[1]),
        "n_halluc_et": int(n_halluc[2]),
        "n_miss_tc": int(n_miss[0]), "n_miss_wt": int(n_miss[1]),
        "n_miss_et": int(n_miss[2]), "n_test": sample_count,
    }
    run_logger.write_test_metrics(test_row)
    return test_row

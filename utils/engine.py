import time

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from tqdm import tqdm

from utils.attention import register_attention_hook, save_attention_overlay, save_attention_evolution
from utils.losses import combine_main_and_aux
from utils.metrics import (
    compute_hd95, compute_sensitivity, compute_iou, compute_miou,
    compute_confusion, compute_specificity, compute_f1, compute_roc_auc,
)
from utils.plot import plot_test_qualitative
from utils.progress import RunTimeEstimator, format_duration
from utils.transforms import fit_to_size


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

    model = UNETR(
        img_shape=cfg.unetr.img_shape,
        input_dim=cfg.unetr.input_dim,
        output_dim=cfg.unetr.output_dim,
        embed_dim=cfg.unetr.embed_dim,
        patch_size=cfg.unetr.patch_size,
        num_heads=cfg.unetr.num_heads,
        dropout=cfg.unetr.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params/1e6:.2f}M")

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

    return model


def build_training_components(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=1e-6)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    scaler = torch.amp.GradScaler('cuda')
    torch.backends.cudnn.benchmark = True
    return optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans, scaler


def run_inference(model, inputs, cfg):
    def _compute():
        return sliding_window_inference(
            inputs=inputs,
            roi_size=cfg.unetr.img_shape,
            sw_batch_size=1,
            predictor=model,
        )

    if cfg.val_amp:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return _compute()
    return _compute()


def train_one_epoch(model, train_loader, train_ds, optimizer, scaler, loss_fn, device, cfg, epoch,
                     estimator=None, console=None):
    model.train()
    epoch_loss = 0
    step = 0
    pbar = tqdm(
        train_loader, total=len(train_loader), file=console, dynamic_ncols=True,
        desc=f"Epoch {epoch + 1}/{cfg.epoch} [train]", leave=False,
    )
    for batch_data in pbar:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        inputs = fit_to_size(inputs, cfg.unetr.img_shape)
        labels = fit_to_size(labels, cfg.unetr.img_shape)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs, aux_z6, aux_z3 = model(inputs)
            loss = combine_main_and_aux(loss_fn, outputs, aux_z6, aux_z3, labels, cfg)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_loss += loss.item()

        if estimator is not None:
            estimator.record_train_step(time.time() - step_start)
            pbar.set_postfix_str(f"loss={loss.item():.4f}  run_eta={estimator.eta_string()}")
        else:
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
    pbar.close()
    epoch_loss /= step
    return epoch_loss


def validate(model, val_loader, val_ds, loss_fn, dice_metric, dice_metric_batch, post_trans,
             device, cfg, epoch, attention_cache, attention_dir, estimator=None, console=None):
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
    compute_auc_this_epoch = ((epoch + 1) % cfg.metrics.auc_every_n_epochs == 0)
    attention_this_epoch = cfg.attention.enabled and ((epoch + 1) % cfg.attention.every_n_epochs == 0)

    pbar = tqdm(
        val_loader, total=len(val_loader), file=console, dynamic_ncols=True,
        desc=f"Epoch {epoch + 1}/{cfg.epoch} [val]", leave=False,
    )
    with torch.no_grad():
        for val_data in pbar:
            step_start = time.time()
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = run_inference(model, val_inputs, cfg)
            val_loss_epoch += loss_fn(val_outputs, val_labels).item()
            val_steps += 1
            val_outputs_raw = [i.detach().cpu() for i in decollate_batch(val_outputs)]
            val_outputs_disc = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels_list = decollate_batch(val_labels)
            dice_metric(y_pred=val_outputs_disc, y=val_labels)
            dice_metric_batch(y_pred=val_outputs_disc, y=val_labels)

            for pred_raw, pred, gt in zip(val_outputs_raw, val_outputs_disc, val_labels_list):
                pred_prob_np = torch.sigmoid(pred_raw).numpy()
                pred_np = pred.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()
                for c in range(3):
                    pred_c = pred_np[c] > 0.5
                    gt_c = gt_np[c] > 0.5
                    tp, fp, fn = compute_confusion(pred_c, gt_c)
                    metric_sums["hd95"][c] += compute_hd95(pred_c, gt_c, cfg.metrics.voxel_spacing)
                    metric_sums["sens"][c] += compute_sensitivity(tp, fn)
                    metric_sums["iou"][c] += compute_iou(tp, fp, fn)
                    metric_sums["miou"][c] += compute_miou(pred_c, gt_c)
                    metric_sums["spec"][c] += compute_specificity(pred_c, gt_c)
                    metric_sums["f1"][c] += compute_f1(tp, fp, fn)
                    if compute_auc_this_epoch:
                        metric_sums["auc"][c] += compute_roc_auc(pred_prob_np[c], gt_c)
                sample_count += 1

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

            if estimator is not None:
                estimator.record_val_step(time.time() - step_start)
                pbar.set_postfix_str(f"run_eta={estimator.eta_string()}")
    pbar.close()

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
    print("\n".join(lines))


def run_training(model, loaders, loss_fn, device, cfg, run_logger):
    train_loader, train_ds = loaders["train_loader"], loaders["train_ds"]
    val_loader, val_ds = loaders["val_loader"], loaders["val_ds"]

    optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans, scaler = build_training_components(model, cfg)

    attention_cache = register_attention_hook(model) if cfg.attention.enabled else {}

    run_logger.open_metrics_csv(METRICS_CSV_FIELDS)

    num_val_runs = sum(1 for e in range(cfg.epoch) if (e + 1) % cfg.val_interval == 0)
    estimator = RunTimeEstimator(
        total_train_steps=cfg.epoch * len(train_loader),
        total_val_steps=num_val_runs * len(val_loader),
    )

    training = True
    best_metric = -1
    best_metric_epoch = -1
    not_improved_epoch = 0

    total_start = time.time()
    for epoch in range(cfg.epoch):
        epoch_start = time.time()

        # anneal the Hausdorff term (loss_fn holds the schedule; no-op if absent)
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch, cfg.epoch)

        epoch_loss = train_one_epoch(
            model, train_loader, train_ds, optimizer, scaler, loss_fn, device, cfg, epoch,
            estimator=estimator, console=run_logger.console,
        )
        lr_scheduler.step()

        row = {
            "epoch": epoch + 1,
            "lr": lr_scheduler.get_last_lr()[0],
            "train_loss": epoch_loss,
            "is_best": False,
        }

        if (epoch + 1) % cfg.val_interval == 0:
            val_metrics = validate(
                model, val_loader, val_ds, loss_fn, dice_metric, dice_metric_batch,
                post_trans, device, cfg, epoch, attention_cache, run_logger.attention_dir,
                estimator=estimator, console=run_logger.console,
            )
            row.update(val_metrics)
            metric = val_metrics["mean_dice"]

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                not_improved_epoch = 0
                row["is_best"] = True
                torch.save(model.state_dict(), run_logger.checkpoint_path)
            else:
                not_improved_epoch += 1
                if not_improved_epoch >= cfg.patience:
                    training = False

        row["epoch_time_sec"] = time.time() - epoch_start
        run_logger.log_epoch_metrics(row)
        _print_epoch_summary(epoch, cfg, row, best_metric, best_metric_epoch, estimator)

        if not training:
            print("Training stopped early: no improvement within patience window.")
            break

    total_time = time.time() - total_start
    print(f"Train completed | best_metric: {best_metric:.4f} @ epoch {best_metric_epoch} | total time: {format_duration(total_time)}")

    if cfg.attention.enabled:
        save_attention_evolution(run_logger.attention_dir)

    return best_metric, best_metric_epoch, total_time


def run_test(model, loaders, loss_fn, device, cfg, run_logger):
    test_loader, test_ds = loaders["test_loader"], loaders["test_ds"]

    model.load_state_dict(torch.load(run_logger.checkpoint_path))
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

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
        "sens": np.zeros(3, dtype=np.float64),
        "iou": np.zeros(3, dtype=np.float64),
    }
    sample_count = 0

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
                    metric_sums["hd95"][c] += compute_hd95(pred_c, gt_c, cfg.metrics.voxel_spacing)
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
    print("Metric on test image: ", metric)
    print(f"metric_tc: {metric_tc:.4f}")
    print(f"metric_wt: {metric_wt:.4f}")
    print(f"metric_et: {metric_et:.4f}")

    test_row = {
        "dice_tc": dice_tc, "dice_wt": dice_wt, "dice_et": dice_et, "mean_dice": mean_dice,
        "hd95_tc": hd95_tc, "hd95_wt": hd95_wt, "hd95_et": hd95_et, "mean_hd95": mean_hd95,
        "sens_tc": sens_tc, "sens_wt": sens_wt, "sens_et": sens_et, "mean_sens": mean_sens,
        "iou_tc": iou_tc, "iou_wt": iou_wt, "iou_et": iou_et, "mean_iou": mean_iou,
    }
    run_logger.write_test_metrics(test_row)
    return test_row

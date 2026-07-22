import torch
from monai.losses import DiceLoss, FocalLoss, HausdorffDTLoss


def build_loss_fn(cfg):
    """Weighted combination of Dice + Focal + Hausdorff-DT losses."""
    dice_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True)
    focal_loss_fn = FocalLoss(gamma=cfg.loss.focal_gamma, reduction="mean")
    hausdorff_loss_fn = HausdorffDTLoss(to_onehot_y=False, sigmoid=True)

    def loss_function(outputs, labels):
        sig = torch.sigmoid(outputs)
        l_dice = dice_loss_fn(outputs, labels)
        l_focal = focal_loss_fn(sig, labels.float())
        l_hd = hausdorff_loss_fn(outputs, labels)
        return (
            cfg.loss.dice_weight * l_dice
            + cfg.loss.focal_weight * l_focal
            + cfg.loss.hausdorff_weight * l_hd
        )

    return loss_function


def combine_main_and_aux(loss_function, outputs, aux_z6, aux_z3, labels, cfg):
    loss_main = loss_function(outputs, labels)
    loss_aux6 = loss_function(aux_z6, labels)
    loss_aux3 = loss_function(aux_z3, labels)
    return (
        loss_main
        + cfg.loss.aux_z6_weight * loss_aux6
        + cfg.loss.aux_z3_weight * loss_aux3
    )

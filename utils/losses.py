import torch
from monai.losses import DiceLoss, TverskyLoss, HausdorffDTLoss


class CombinedLoss:
    """Dice + Focal-Tversky + (annealed) Hausdorff-DT, multi-label sigmoid.

    Changes vs the old Dice+Focal+HD combo:
      * Focal-Tversky REPLACES the plain Focal term. Tversky with alpha > beta
        up-weights false negatives — the ET/TC failure mode (their sensitivity
        was ~0.73, i.e. FN-dominated). The focal exponent 1/gamma then focuses
        learning on the hard, poorly-segmented channels (Abraham & Khan, ISBI
        2019). alpha/beta are uniform across TC/WT/ET; recall weighting mostly
        moves ET/TC anyway since WT already has high recall.
        ponytail: uniform alpha/beta, per-channel Tversky weights if ET still lags.
      * The old code fed sigmoid(outputs) into MONAI FocalLoss, which expects
        raw logits (double-sigmoid bug). Every term here takes raw logits with
        the loss's own sigmoid=True.
      * Hausdorff-DT weight is annealed 0 -> full over the first hd_anneal_frac
        of training. Boundary/distance losses destabilize at full strength from
        step 0 (Kervadec, MIDL 2019); ramping keeps Dice+FT as the early anchor.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dice = DiceLoss(to_onehot_y=False, sigmoid=True)
        # reduction="none" -> per-(batch, channel) Tversky loss (= 1 - TI_c),
        # so the focal exponent can be applied per channel before averaging.
        self.tversky = TverskyLoss(
            to_onehot_y=False, sigmoid=True,
            alpha=cfg.loss.tversky_alpha, beta=cfg.loss.tversky_beta,
            reduction="none",
        )
        self.hausdorff = HausdorffDTLoss(to_onehot_y=False, sigmoid=True)
        self._hd_scale = 1.0  # full unless set_epoch() anneals it (train loop)

    def set_epoch(self, epoch, total_epochs):
        """Linear anneal: HD weight ramps 0 -> full over the first
        `hd_anneal_frac` of training, then holds at full."""
        frac = epoch / max(total_epochs - 1, 1)
        ramp = self.cfg.loss.hd_anneal_frac
        self._hd_scale = 1.0 if ramp <= 0 else min(1.0, frac / ramp)

    def _focal_tversky(self, outputs, labels):
        tv = self.tversky(outputs, labels)          # (B, C, 1, 1, 1) = 1 - TI_c
        return tv.pow(1.0 / self.cfg.loss.tversky_gamma).mean()

    def __call__(self, outputs, labels):
        l_dice = self.dice(outputs, labels)
        l_ft = self._focal_tversky(outputs, labels.float())
        l_hd = self.hausdorff(outputs, labels)
        return (
            self.cfg.loss.dice_weight * l_dice
            + self.cfg.loss.tversky_weight * l_ft
            + self.cfg.loss.hausdorff_weight * self._hd_scale * l_hd
        )


def build_loss_fn(cfg):
    """Weighted combination of Dice + Focal-Tversky + Hausdorff-DT losses."""
    return CombinedLoss(cfg)


def combine_main_and_aux(loss_function, outputs, aux_z6, aux_z3, labels, cfg):
    loss_main = loss_function(outputs, labels)
    loss_aux6 = loss_function(aux_z6, labels)
    loss_aux3 = loss_function(aux_z3, labels)
    return (
        loss_main
        + cfg.loss.aux_z6_weight * loss_aux6
        + cfg.loss.aux_z3_weight * loss_aux3
    )

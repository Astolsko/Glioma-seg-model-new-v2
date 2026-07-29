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

    def aux_loss(self, outputs, labels):
        """Loss for the deep-supervision heads: Dice + Focal-Tversky, NO
        Hausdorff. The Hausdorff term stays off the aux heads on purpose — it is
        the expensive one (a scipy CPU distance transform per channel per
        sample) and its own authors ramp it in on the MAIN output only.

        Dice-only aux (the previous behaviour) supplies overlap gradients but no
        recall pressure, and the recall regression is exactly what the fine
        decoder scales the aux heads feed need to fix. Adding Focal-Tversky
        (alpha>beta => false-negative weighting) puts that pressure on those
        scales for near-zero cost (Tversky is cheap GPU work)."""
        l_dice = self.dice(outputs, labels)
        l_ft = self._focal_tversky(outputs, labels.float())
        return (self.cfg.loss.dice_weight * l_dice
                + self.cfg.loss.tversky_weight * l_ft)


def build_loss_fn(cfg):
    """Weighted combination of Dice + Focal-Tversky + Hausdorff-DT losses."""
    return CombinedLoss(cfg)


def combine_main_and_aux(loss_function, outputs, aux_z6, aux_z3, labels, cfg):
    """Main head gets the full composite loss; the deep-supervision heads get
    Dice + Focal-Tversky (NO Hausdorff — see CombinedLoss.aux_loss).

    Running the full composite on the aux heads too would mean three
    HausdorffDTLoss evaluations per step instead of one — each a scipy CPU
    distance transform per channel per sample, a large fraction of the epoch.
    Keeping HD on the main head only preserves that speedup while restoring the
    Focal-Tversky recall pressure on the fine decoder scales the aux heads feed.

    Falls back to calling `loss_function` itself when it exposes no `aux_loss`
    attribute, so a plain callable still works (the tests rely on this).
    """
    aux_loss_fn = getattr(loss_function, "aux_loss", loss_function)
    return (
        loss_function(outputs, labels)
        + cfg.loss.aux_z6_weight * aux_loss_fn(aux_z6, labels)
        + cfg.loss.aux_z3_weight * aux_loss_fn(aux_z3, labels)
    )

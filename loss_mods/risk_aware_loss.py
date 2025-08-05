"""
Risk-Aware BCE classification loss for YOLOv8
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss, DFLoss

# ─── class weights (glass, medical, metal, organic, paper, plastic, sharp-object)
CLS_WEIGHTS = torch.tensor([4.47, 5.0, 3.0, 2.19, 2.57, 2.34, 4.96])


class RiskAwareLoss(v8DetectionLoss):
    """v8DetectionLoss drop-in with weighted BCE on classification term."""

    def __init__(self, model):
        super().__init__(model)  # build stock loss
        self.register_buffer("cls_w", CLS_WEIGHTS.view(1, 1, -1))
        print("✅  RiskAwareLoss initialised", flush=True)

    # --------------------------------------------------------------------- #
    #  Override __call__: reuse box & dfl terms, replace cls term only
    # --------------------------------------------------------------------- #
    def __call__(self, preds, batch):
        loss, items = super().__call__(preds, batch)        # box, cls, dfl

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_scores = torch.cat(
            [x.view(feats[0].shape[0], self.no, -1) for x in feats], dim=2
        )[:, self.reg_max * 4 :, :].permute(0, 2, 1)

        target_scores = batch["cls"].unsqueeze(-1).eq(
            torch.arange(self.nc, device=pred_scores.device)
        ).float().expand_as(pred_scores)

        # weighted BCE
        cls_loss = self.bce(pred_scores, target_scores) * self.cls_w.to(pred_scores.dtype)
        items[1] = cls_loss.mean()                         # replace cls item

        loss = items[0] * self.hyp.box + items[1] * self.hyp.cls + items[2]
        return loss, items


# ─── patch Ultralytics at import-time ───────────────────────────────────── #
import ultralytics.utils.loss as _ul_loss

RiskAwareLoss.__module__ = "ultralytics.utils.loss"
RiskAwareLoss.__qualname__ = "v8DetectionLoss"

_ul_loss.v8DetectionLoss = RiskAwareLoss         # every later import sees ours

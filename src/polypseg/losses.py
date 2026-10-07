"""Segmentation losses: soft Dice, BCE and their weighted combination."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice, computed per image and averaged over the batch.

    Dice = 2|P ∩ T| / (|P| + |T|) with probabilities instead of hard predictions, so it
    is differentiable. Being overlap-based it ignores the (easy, abundant) background,
    which is what makes it useful for small foreground objects.
    """
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5
                  ) -> torch.Tensor:
    """BCE keeps per-pixel gradients everywhere; Dice optimises the overlap we report."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce_weight * bce + (1.0 - bce_weight) * soft_dice_loss(logits, target)


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return bce_dice_loss(logits, target, self.bce_weight)

"""Binary segmentation metrics on hard masks: Dice, IoU, precision, recall, accuracy.

Two aggregation conventions matter in the polyp literature, so both are provided:
- ``mean of per-image`` scores (what most Kvasir-SEG papers report as mDice / mIoU);
- ``dataset-level`` (micro) scores from the pooled confusion counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @classmethod
    def from_masks(cls, pred: np.ndarray, target: np.ndarray) -> Confusion:
        p, t = pred.astype(bool), target.astype(bool)
        return cls(tp=int((p & t).sum()), fp=int((p & ~t).sum()),
                   fn=int((~p & t).sum()), tn=int((~p & ~t).sum()))

    def __add__(self, other: Confusion) -> Confusion:
        return Confusion(self.tp + other.tp, self.fp + other.fp,
                         self.fn + other.fn, self.tn + other.tn)

    def dice(self, eps: float = 1e-7) -> float:
        # Empty prediction + empty target is a perfect match, not a zero.
        if self.tp + self.fp + self.fn == 0:
            return 1.0
        return 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + eps)

    def iou(self, eps: float = 1e-7) -> float:
        if self.tp + self.fp + self.fn == 0:
            return 1.0
        return self.tp / (self.tp + self.fp + self.fn + eps)

    def precision(self, eps: float = 1e-7) -> float:
        return self.tp / (self.tp + self.fp + eps) if self.tp + self.fp else 0.0

    def recall(self, eps: float = 1e-7) -> float:
        return self.tp / (self.tp + self.fn + eps) if self.tp + self.fn else 0.0

    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 1.0

    def as_dict(self) -> dict[str, float]:
        return {"dice": self.dice(), "iou": self.iou(), "precision": self.precision(),
                "recall": self.recall(), "accuracy": self.accuracy()}


def binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(np.uint8)


def image_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return Confusion.from_masks(pred, target).as_dict()


def summarize(per_image: list[dict[str, float]], pooled: Confusion) -> dict[str, dict[str, float]]:
    """Mean-of-images and dataset-level metrics side by side."""
    keys = ("dice", "iou", "precision", "recall", "accuracy")
    mean = {k: float(np.mean([m[k] for m in per_image])) for k in keys} if per_image else {}
    return {"mean_per_image": mean, "dataset_level": pooled.as_dict(),
            "num_images": len(per_image)}


def polyp_size_fraction(target: np.ndarray) -> float:
    """Polyp area as a fraction of the image; used to stratify results by polyp size."""
    return float(target.astype(bool).mean())


def size_bucket(fraction: float) -> str:
    if fraction < 0.05:
        return "small (<5%)"
    if fraction < 0.20:
        return "medium (5-20%)"
    return "large (>20%)"

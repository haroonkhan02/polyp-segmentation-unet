"""Monte-Carlo dropout uncertainty: maps, summary statistics and error correlation.

With dropout kept active at test time, each forward pass samples a slightly different
network. The spread of the T sampled probability maps estimates *epistemic* (model)
uncertainty (Gal & Ghahramani, 2016). Two maps are derived:

- ``std``: per-pixel standard deviation of the sampled probabilities;
- ``entropy``: binary entropy of the mean probability (captures both the model
  disagreement and the inherent ambiguity of the pixel).
"""

from __future__ import annotations

import numpy as np


def mean_std(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """samples: (T, H, W) probabilities -> (mean, std) maps."""
    return samples.mean(axis=0), samples.std(axis=0)


def binary_entropy(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """H(p) in nats, maximal (ln 2) at p = 0.5."""
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def summarize_uncertainty(std: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Scalars a clinician-facing UI could show next to the mask."""
    region = pred.astype(bool)
    boundary = _boundary(pred)
    return {
        "mean_std": float(std.mean()),
        "mean_std_in_polyp": float(std[region].mean()) if region.any() else 0.0,
        "mean_std_on_boundary": float(std[boundary].mean()) if boundary.any() else 0.0,
        "uncertain_fraction": float((std > 0.1).mean()),
    }


def _boundary(mask: np.ndarray, width: int = 3) -> np.ndarray:
    import cv2

    m = mask.astype(np.uint8)
    kernel = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    return (cv2.dilate(m, kernel) - cv2.erode(m, kernel)).astype(bool)


def error_uncertainty_stats(uncertainty: np.ndarray, pred: np.ndarray, target: np.ndarray
                            ) -> dict[str, float]:
    """Is the model uncertain where it is wrong? Compares uncertainty on error vs correct pixels.

    Returns the mean uncertainty on mis-classified and correctly classified pixels and the
    AUROC of the uncertainty map as a detector of pixel errors (0.5 = useless, 1 = perfect).
    """
    errors = pred.astype(bool) != target.astype(bool)
    u = uncertainty.ravel()
    e = errors.ravel()
    out = {"mean_unc_on_errors": float(u[e].mean()) if e.any() else 0.0,
           "mean_unc_on_correct": float(u[~e].mean()) if (~e).any() else 0.0,
           "error_fraction": float(e.mean())}
    out["auroc"] = auroc(u, e) if e.any() and (~e).any() else float("nan")
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U) without scikit-learn. Ties get average ranks."""
    scores = scores.astype(np.float64)
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def subsample_for_auroc(uncertainty: np.ndarray, pred: np.ndarray, target: np.ndarray,
                        max_pixels: int = 200_000, seed: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Random pixel subsample so the pooled AUROC over a whole test set fits in memory."""
    rng = np.random.default_rng(seed)
    u = uncertainty.ravel()
    e = (pred.astype(bool) != target.astype(bool)).ravel()
    if len(u) > max_pixels:
        idx = rng.choice(len(u), max_pixels, replace=False)
        return u[idx], e[idx]
    return u, e

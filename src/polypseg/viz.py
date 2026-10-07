"""Overlays, uncertainty heatmaps and gallery mosaics (RGB uint8 in, RGB uint8 out)."""

from __future__ import annotations

import cv2
import numpy as np

GREEN = (0, 220, 0)
RED = (230, 40, 40)
BLUE = (40, 120, 255)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=GREEN, alpha: float = 0.4,
                 contour: bool = True) -> np.ndarray:
    """Tint the mask area and draw its contour."""
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if contour:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def overlay_comparison(rgb: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Green = correct polyp (TP), red = missed (FN), blue = false alarm (FP)."""
    p, t = pred.astype(bool), target.astype(bool)
    out = rgb.copy().astype(np.float32)
    for region, color in ((p & t, GREEN), (~p & t, RED), (p & ~t, BLUE)):
        out[region] = out[region] * 0.55 + np.array(color) * 0.45
    return out.astype(np.uint8)


def heatmap(values: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Scalar map -> RGB colour map (inferno-like via OpenCV's MAGMA)."""
    vmax = vmax or max(float(values.max()), 1e-6)
    v = np.clip(values / vmax, 0, 1)
    cm = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def blend_heatmap(rgb: np.ndarray, values: np.ndarray, vmax: float | None = None,
                  alpha: float = 0.55) -> np.ndarray:
    hm = heatmap(values, vmax)
    return cv2.addWeighted(rgb, 1 - alpha, hm, alpha, 0)


def label(img: np.ndarray, text: str, scale: float = 0.6) -> np.ndarray:
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(out, (0, 0), (tw + 10, th + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (5, th + 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def tile(images: list[np.ndarray], size: int = 256) -> np.ndarray:
    """Resize each image to size x size and concatenate horizontally."""
    return np.concatenate([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
                           for im in images], axis=1)


def mosaic(rows: list[np.ndarray]) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    return np.concatenate(padded, axis=0)


def case_row(rgb: np.ndarray, pred: np.ndarray, target: np.ndarray,
             uncertainty: np.ndarray | None, title: str, size: int = 256) -> np.ndarray:
    """image | ground truth | prediction | TP/FN/FP comparison | uncertainty.

    Panels are resized *before* labelling so the text has the same size on every frame.
    """
    panels = [(rgb, title), (overlay_mask(rgb, target), "ground truth"),
              (overlay_mask(rgb, pred), "prediction"),
              (overlay_comparison(rgb, pred, target), "green TP  red FN  blue FP")]
    if uncertainty is not None:
        panels.append((blend_heatmap(rgb, uncertainty, vmax=0.5), "MC-dropout std"))
    return np.concatenate(
        [label(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA), text, 0.45)
         for im, text in panels], axis=1)


def save_rgb(path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

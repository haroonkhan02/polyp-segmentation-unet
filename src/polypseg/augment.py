"""Paired image/mask augmentation with OpenCV (no extra dependency).

Every geometric transform is applied to the image and the mask with the *same*
parameters; the mask is warped with nearest-neighbour interpolation so it stays binary.
Photometric transforms touch the image only. Polyps have no canonical orientation, so
vertical flips and 90-degree rotations are valid here (unlike, say, road scenes).
"""

from __future__ import annotations

import cv2
import numpy as np


def random_flips(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                 p_h: float, p_v: float, p_rot90: float) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < p_h:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < p_v:
        img, mask = img[::-1], mask[::-1]
    if rng.random() < p_rot90:
        k = int(rng.integers(1, 4))
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def random_affine(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                  max_rotate: float, scale: tuple[float, float], shift: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate / scale / translate about the image centre; borders are reflected."""
    h, w = mask.shape[:2]
    angle = float(rng.uniform(-max_rotate, max_rotate))
    s = float(rng.uniform(*scale))
    tx, ty = rng.uniform(-shift, shift, 2) * (w, h)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
    m[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT_101)
    return img, mask


def color_jitter(img: np.ndarray, rng: np.random.Generator, brightness: float,
                 contrast: float, hue: float, saturation: float) -> np.ndarray:
    """Brightness/contrast in RGB, then hue/saturation in HSV. ``img`` is uint8 RGB."""
    out = img.astype(np.float32)
    alpha = 1.0 + float(rng.uniform(-contrast, contrast))
    beta = float(rng.uniform(-brightness, brightness)) * 255.0
    out = np.clip(alpha * (out - 128.0) + 128.0 + beta, 0, 255).astype(np.uint8)
    if hue > 0 or saturation > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-hue, hue))) % 180
        sat = hsv[..., 1].astype(np.float32) * (1.0 + float(rng.uniform(-saturation, saturation)))
        hsv[..., 1] = np.clip(sat, 0, 255).astype(np.int16)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def random_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(img, (k, k), 0)


def augment_pair(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                 params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full augmentation policy described by ``params`` (see TrainConfig)."""
    img, mask = random_flips(img, mask, rng, params["hflip"], params["vflip"],
                             params["rotate90"])
    if rng.random() < params["affine"]:
        img, mask = random_affine(img, mask, rng, params["max_rotate"],
                                  tuple(params["scale"]), params["shift"])
    if rng.random() < params["color"]:
        img = color_jitter(img, rng, params["brightness"], params["contrast"],
                           params["hue"], params["saturation"])
    if rng.random() < params["blur"]:
        img = random_blur(img, rng)
    return img, mask

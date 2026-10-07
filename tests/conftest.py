"""Synthetic colonoscopy-like fixtures: no dataset, GPU or download needed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from polypseg.config import TrainConfig
from polypseg.model import build_model, save_checkpoint


def synthetic_case(size: int = 96, radius: int = 22, center=None, seed: int = 0):
    """Pink mucosa with a darker reddish 'polyp' disk; returns (rgb uint8, mask {0,1})."""
    rng = np.random.default_rng(seed)
    rgb = np.full((size, size, 3), (210, 140, 150), np.uint8)
    rgb = np.clip(rgb.astype(int) + rng.integers(-15, 15, rgb.shape), 0, 255).astype(np.uint8)
    cx, cy = center or (size // 2, size // 2)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, -1)
    rgb[mask == 1] = (150, 60, 80)
    return rgb, mask


@pytest.fixture
def case():
    return synthetic_case()


@pytest.fixture
def fake_kvasir(tmp_path):
    """A 6-image Kvasir-SEG-shaped folder with JPEG masks (so thresholding is exercised)."""
    root = tmp_path / "Kvasir-SEG"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for i in range(6):
        rgb, mask = synthetic_case(80, radius=8 + 3 * i, center=(30 + 4 * i, 40), seed=i)
        cv2.imwrite(str(root / "images" / f"img{i}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "masks" / f"img{i}.jpg"), mask * 255)
    return root


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Untrained resnet18 U-Net with decoder dropout, 64 px input."""
    cfg = TrainConfig(encoder="resnet18", pretrained=False, image_size=64, dropout=0.3)
    torch.manual_seed(0)
    model = build_model(cfg.encoder, cfg.pretrained, cfg.dropout)
    path = tmp_path_factory.mktemp("models") / "tiny.pt"
    save_checkpoint(model, path, cfg, {"epoch": 1, "val_dice": 0.0})
    return path

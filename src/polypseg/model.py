"""U-Net with a pretrained encoder, decoder dropout for MC sampling, and inference wrapper."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn

from .config import TrainConfig
from .data import preprocess


def pick_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DropoutBlock(nn.Module):
    """Wraps a decoder block so Dropout2d follows it (signature-agnostic across smp versions)."""

    def __init__(self, block: nn.Module, p: float):
        super().__init__()
        self.block = block
        self.dropout = nn.Dropout2d(p)

    def forward(self, *args, **kwargs):
        return self.dropout(self.block(*args, **kwargs))


def build_model(encoder: str = "resnet34", pretrained: bool = True, dropout: float = 0.2
                ) -> nn.Module:
    """Binary U-Net (1 logit channel). ``dropout`` > 0 adds Dropout2d after every decoder block.

    Spatial (channel-wise) dropout is used instead of per-pixel dropout: dropping whole
    feature maps produces coherent alternative segmentations when sampled at test time,
    which is what Monte-Carlo dropout needs to estimate epistemic uncertainty.
    """
    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                     in_channels=3, classes=1)
    if dropout > 0:
        model.decoder.blocks = nn.ModuleList(
            [DropoutBlock(b, dropout) for b in model.decoder.blocks])
    return model


def enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode everywhere (BatchNorm uses running stats) except the dropout layers."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout | nn.Dropout2d):
            m.train()


def save_checkpoint(model: nn.Module, path: Path, cfg: TrainConfig, extra: dict | None = None
                    ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "config": cfg.to_dict(), **(extra or {})}, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None
                    ) -> tuple[nn.Module, TrainConfig, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg.encoder, pretrained=False, dropout=cfg.dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device or torch.device("cpu")).eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "config")}
    return model, cfg, meta


class PolypSegmenter:
    """Loads a checkpoint and predicts polyp probability maps at the original resolution."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = pick_device(device)
        self.model, self.cfg, self.meta = load_checkpoint(checkpoint, self.device)
        self.size = self.cfg.image_size
        self.last_inference_ms = 0.0

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x.to(self.device)))[:, 0]

    def _to_image(self, probs: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
        p = probs.float().cpu().numpy()
        return cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Deterministic probability map (HxW float32 in [0, 1]) for an RGB uint8 image."""
        self.model.eval()
        x = preprocess(rgb, self.size)
        start = time.perf_counter()
        probs = self._forward(x)[0]
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        return self._to_image(probs, rgb.shape[:2])

    def predict_mc(self, rgb: np.ndarray, samples: int = 20) -> np.ndarray:
        """``samples`` stochastic forward passes with dropout on -> (samples, H, W) probs."""
        enable_mc_dropout(self.model)
        x = preprocess(rgb, self.size).repeat(samples, 1, 1, 1)
        try:
            probs = self._forward(x)
        finally:
            self.model.eval()
        return np.stack([self._to_image(p, rgb.shape[:2]) for p in probs])

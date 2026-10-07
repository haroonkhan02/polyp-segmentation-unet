"""Paths, dataset constants and the YAML-backed training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
KVASIR_DIR = DATA_DIR / "Kvasir-SEG"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"
OUTPUTS_DIR = ROOT / "outputs"

# Kvasir-SEG (Jha et al., 2020): 1,000 colonoscopy images with pixel-wise polyp masks.
KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_ZIP_BYTES = 46_227_172
KVASIR_NUM_IMAGES = 1000
SPLIT_FILE = CONFIGS_DIR / "split.json"
DEFAULT_CONFIG = CONFIGS_DIR / "kvasir_unet.yaml"

# ImageNet statistics used by the pretrained encoders.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MASK_THRESHOLD = 127  # Kvasir masks are JPEGs: compression leaves values in 1..254


@dataclass
class TrainConfig:
    """Everything that defines a training run; saved into the checkpoint and the history."""

    encoder: str = "resnet34"
    pretrained: bool = True
    image_size: int = 320
    epochs: int = 40
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2  # Dropout2d after every decoder block (used for MC dropout)
    bce_weight: float = 0.5  # loss = bce_weight * BCE + (1 - bce_weight) * Dice
    workers: int = 4
    seed: int = 0
    augment: dict = field(default_factory=lambda: {
        "hflip": 0.5, "vflip": 0.5, "rotate90": 0.5, "affine": 0.7, "max_rotate": 25,
        "scale": [0.8, 1.2], "shift": 0.1, "color": 0.8, "brightness": 0.2,
        "contrast": 0.2, "hue": 8, "saturation": 0.2, "blur": 0.2,
    })

    @classmethod
    def load(cls, path: str | Path | None = None) -> TrainConfig:
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        cfg = cls(**{k: v for k, v in data.items() if k != "augment"})
        if "augment" in data:
            cfg.augment = {**cfg.augment, **data["augment"]}
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

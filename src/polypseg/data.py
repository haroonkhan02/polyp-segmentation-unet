"""Kvasir-SEG listing, deterministic train/val/test split and the PyTorch dataset."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment_pair
from .config import IMAGENET_MEAN, IMAGENET_STD, KVASIR_DIR, MASK_THRESHOLD, SPLIT_FILE

SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def list_pairs(root: Path = KVASIR_DIR) -> list[str]:
    """Sorted image stems that have both an image and a mask."""
    images = {p.stem for p in (root / "images").glob("*.jpg")}
    masks = {p.stem for p in (root / "masks").glob("*.jpg")}
    return sorted(images & masks)


def make_split(stems: list[str], seed: int = 0,
               fractions: dict[str, float] | None = None) -> dict[str, list[str]]:
    """Shuffle the sorted stems with a fixed seed and cut train/val/test (80/10/10)."""
    fractions = fractions or SPLIT_FRACTIONS
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    order = np.random.default_rng(seed).permutation(len(stems))
    shuffled = [stems[i] for i in order]
    n_train = int(round(fractions["train"] * len(stems)))
    n_val = int(round(fractions["val"] * len(stems)))
    return {"train": sorted(shuffled[:n_train]),
            "val": sorted(shuffled[n_train:n_train + n_val]),
            "test": sorted(shuffled[n_train + n_val:])}


def load_or_create_split(root: Path = KVASIR_DIR, path: Path = SPLIT_FILE,
                         seed: int = 0) -> dict[str, list[str]]:
    """The committed split file wins, so every run and reader uses the same test set."""
    if path.exists():
        split = json.loads(path.read_text())
        present = set(list_pairs(root))
        return {k: [s for s in v if s in present] for k, v in split.items()}
    split = make_split(list_pairs(root), seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))
    return split


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    """Binary uint8 mask {0, 1}. Kvasir masks are JPEGs, so threshold the artefacts away."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > MASK_THRESHOLD).astype(np.uint8)


def resize_pair(img: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 RGB HxWx3 -> float CxHxW normalised with ImageNet statistics."""
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def preprocess(img: np.ndarray, size: int) -> torch.Tensor:
    """Inference preprocessing for a single RGB image: resize + normalise -> 1x3xSxS."""
    return to_tensor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)).unsqueeze(0)


class PolypDataset(Dataset):
    """Returns (image tensor 3xSxS, mask tensor 1xSxS float {0,1}) for a list of stems."""

    def __init__(self, stems: list[str], root: Path = KVASIR_DIR, size: int = 320,
                 augment: dict | None = None, seed: int = 0):
        self.stems = list(stems)
        self.root = Path(root)
        self.size = size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.stems)

    def paths(self, idx: int) -> tuple[Path, Path]:
        stem = self.stems[idx]
        return self.root / "images" / f"{stem}.jpg", self.root / "masks" / f"{stem}.jpg"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.paths(idx)
        img, mask = resize_pair(read_image(img_path), read_mask(mask_path), self.size)
        if self.augment:
            img, mask = augment_pair(img, mask, self.rng, self.augment)
        return to_tensor(img), torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

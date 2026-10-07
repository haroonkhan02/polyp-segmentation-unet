"""Train the polyp U-Net on Kvasir-SEG.

Loss = BCE + soft Dice. The checkpoint with the best validation mean Dice is kept, and the
per-epoch history is written to reports/training_history.json.

Example:
    polyp-train --config configs/kvasir_unet.yaml --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_CONFIG, KVASIR_DIR, MODELS_DIR, REPORTS_DIR, ROOT, TrainConfig
from .data import PolypDataset, load_or_create_split
from .losses import BCEDiceLoss
from .metrics import Confusion, binarize
from .model import build_model, pick_device, save_checkpoint


def _rel(value) -> str:
    """Paths relative to the repo, so reports don't leak local directories."""
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(ROOT))
        except ValueError:
            return value.name
    return str(value)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                    criterion=None) -> dict[str, float]:
    """Mean per-image Dice/IoU (and loss) over a loader with deterministic dropout-off passes."""
    model.eval()
    dices, ious, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            losses.append(criterion(logits, y).item())
        probs = torch.sigmoid(logits)[:, 0].float().cpu().numpy()
        targets = y[:, 0].cpu().numpy()
        for p, t in zip(probs, targets, strict=True):
            c = Confusion.from_masks(binarize(p), t)
            dices.append(c.dice())
            ious.append(c.iou())
    out = {"dice": float(np.mean(dices)), "iou": float(np.mean(ious))}
    if losses:
        out["loss"] = float(np.mean(losses))
    return out


def train_one_epoch(model, loader, criterion, optim, sched, device) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if sched is not None:
            sched.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=KVASIR_DIR)
    parser.add_argument("--epochs", type=int, help="override the config")
    parser.add_argument("--encoder", help="override the config")
    parser.add_argument("--batch-size", type=int, help="override the config")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "unet_resnet34.pt")
    parser.add_argument("--history", type=Path, default=REPORTS_DIR / "training_history.json")
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config if args.config.exists() else None)
    for key in ("epochs", "encoder", "batch_size"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    set_seed(cfg.seed)

    split = load_or_create_split(args.root, seed=cfg.seed)
    if not split["train"] or not split["val"]:
        raise SystemExit("No Kvasir-SEG images found. Run `polyp-download` first.")
    print(f"train {len(split['train'])}  val {len(split['val'])}  test {len(split['test'])}")

    device = pick_device(args.device)
    train_loader = DataLoader(
        PolypDataset(split["train"], args.root, cfg.image_size, cfg.augment, cfg.seed),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers,
        persistent_workers=cfg.workers > 0, drop_last=True)
    val_loader = DataLoader(PolypDataset(split["val"], args.root, cfg.image_size),
                            batch_size=cfg.batch_size, num_workers=cfg.workers)

    model = build_model(cfg.encoder, cfg.pretrained, cfg.dropout).to(device)
    criterion = BCEDiceLoss(cfg.bce_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=cfg.lr, pct_start=0.1,
                                                total_steps=cfg.epochs * len(train_loader))

    history, best, best_epoch = [], -1.0, 0
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optim, sched, device)
        val = evaluate_loader(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_dice": val["dice"], "val_iou": val["iou"],
                        "lr": optim.param_groups[0]["lr"]})
        flag = ""
        if val["dice"] > best:
            best, best_epoch = val["dice"], epoch
            save_checkpoint(model, args.output, cfg, {"epoch": epoch, "val_dice": best,
                                                      "val_iou": val["iou"]})
            flag = "  *best*"
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val['loss']:.4f}  "
              f"val dice {val['dice']:.4f}  val iou {val['iou']:.4f}  "
              f"[{(time.time() - start) / 60:.1f} min]{flag}", flush=True)

    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(json.dumps(
        {"config": cfg.to_dict(), "device": str(device),
         "train_images": len(split["train"]), "val_images": len(split["val"]),
         "best_epoch": best_epoch, "best_val_dice": best, "minutes": (time.time() - start) / 60,
         "checkpoint": _rel(args.output), "history": history}, indent=2))
    print(f"Best val Dice {best:.4f} at epoch {best_epoch}; saved {args.output}")


if __name__ == "__main__":
    main()

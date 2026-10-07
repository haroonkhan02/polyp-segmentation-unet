"""Matplotlib figures for the reports folder (headless backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_training_curves(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("BCE + Dice loss"), axes[0].legend()
    axes[1].plot(epochs, [h["val_dice"] for h in history], label="val Dice")
    axes[1].plot(epochs, [h["val_iou"] for h in history], label="val IoU")
    axes[1].set_xlabel("epoch"), axes[1].set_ylim(0, 1), axes[1].legend()
    best = max(history, key=lambda h: h["val_dice"])
    axes[1].axvline(best["epoch"], color="grey", ls="--", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dice_histogram(dices: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(dices, bins=np.linspace(0, 1, 21), color="#2a9d8f", edgecolor="white")
    ax.axvline(float(np.mean(dices)), color="k", ls="--", lw=1,
               label=f"mean {np.mean(dices):.3f}")
    ax.axvline(float(np.median(dices)), color="#e76f51", ls=":", lw=1.2,
               label=f"median {np.median(dices):.3f}")
    ax.set_xlabel("per-image Dice"), ax.set_ylabel("images"), ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dice_vs_size_and_uncertainty(per_image: list[dict], path: Path) -> None:
    """Two error-analysis views: Dice against polyp size and against mean MC-dropout std."""
    size = [m["polyp_fraction"] * 100 for m in per_image]
    dice = [m["dice"] for m in per_image]
    unc = [m["uncertainty"]["mean_std"] for m in per_image]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].scatter(size, dice, s=14, alpha=0.7, color="#264653")
    axes[0].set_xscale("log"), axes[0].set_xlabel("polyp area (% of image, log)")
    axes[0].set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"], minor=False)
    axes[0].set_xticks([], minor=True)
    axes[0].set_ylabel("Dice"), axes[0].set_ylim(0, 1.02)
    axes[1].scatter(unc, dice, s=14, alpha=0.7, color="#e76f51")
    axes[1].set_xlabel("mean MC-dropout std over the image"), axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

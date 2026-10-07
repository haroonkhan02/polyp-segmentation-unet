"""Evaluate a checkpoint on a split: metrics, size-stratified results, MC-dropout uncertainty
analysis, failure-case galleries and plots. Everything goes to reports/.

Example:
    polyp-evaluate --split test --mc-samples 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import KVASIR_DIR, MODELS_DIR, REPORTS_DIR
from .data import load_or_create_split, read_image, read_mask
from .metrics import Confusion, binarize, polyp_size_fraction, size_bucket, summarize
from .model import PolypSegmenter
from .plots import plot_dice_histogram, plot_dice_vs_size_and_uncertainty, plot_training_curves
from .uncertainty import (
    auroc,
    error_uncertainty_stats,
    mean_std,
    subsample_for_auroc,
    summarize_uncertainty,
)
from .viz import case_row, mosaic, save_rgb


def evaluate_image(seg: PolypSegmenter, rgb: np.ndarray, target: np.ndarray, threshold: float,
                   mc_samples: int) -> tuple[dict, dict]:
    """Metrics for one image (deterministic + MC-mean predictions) and the arrays for galleries."""
    probs = seg.predict(rgb)
    det_ms = seg.last_inference_ms
    pred = binarize(probs, threshold)
    record = {"dice": None, "polyp_fraction": polyp_size_fraction(target), "ms": det_ms}
    record.update(Confusion.from_masks(pred, target).as_dict())
    arrays = {"probs": probs, "pred": pred}
    if mc_samples > 0:
        start = time.perf_counter()
        samples = seg.predict_mc(rgb, mc_samples)
        record["mc_ms"] = (time.perf_counter() - start) * 1000
        mean, std = mean_std(samples)
        mc_pred = binarize(mean, threshold)
        record["mc_mean"] = Confusion.from_masks(mc_pred, target).as_dict()
        record["uncertainty"] = summarize_uncertainty(std, pred)
        record["uncertainty"].update(error_uncertainty_stats(std, pred, target))
        arrays.update({"std": std, "mc_pred": mc_pred})
    return record, arrays


def stratify_by_size(per_image: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for m in per_image:
        buckets.setdefault(size_bucket(m["polyp_fraction"]), []).append(m)
    return {k: {"num_images": len(v), "dice": float(np.mean([m["dice"] for m in v])),
                "iou": float(np.mean([m["iou"] for m in v])),
                "recall": float(np.mean([m["recall"] for m in v]))}
            for k, v in sorted(buckets.items())}


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def build_galleries(cases: list[dict], out_dir: Path, n: int) -> None:
    """gallery_test.jpg = best + median cases; failure_cases.jpg = the worst Dice scores."""
    ranked = sorted(cases, key=lambda c: c["dice"])
    mid = len(ranked) // 2
    n_best = n // 2
    n_med = n - n_best
    picks = {"gallery_test.jpg": ranked[-n_best:][::-1] + ranked[mid - n_med // 2:][:n_med],
             "failure_cases.jpg": ranked[:n]}
    for name, chosen in picks.items():
        rows = [case_row(c["rgb"], c["pred"], c["target"], c.get("std"),
                         f"{c['stem'][:12]}  Dice {c['dice']:.2f}") for c in chosen]
        if rows:
            save_rgb(out_dir / name, mosaic(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "unet_resnet34.pt")
    parser.add_argument("--root", type=Path, default=KVASIR_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=20, help="0 disables MC dropout")
    parser.add_argument("--gallery-size", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    stems = load_or_create_split(args.root)[args.split]
    if not stems:
        raise SystemExit("No images for this split. Run `polyp-download` first.")
    seg = PolypSegmenter(args.checkpoint, args.device)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    per_image, cases, pooled, mc_pooled = [], [], Confusion(0, 0, 0, 0), Confusion(0, 0, 0, 0)
    unc_pixels, err_pixels = [], []
    for stem in tqdm(stems, desc=f"evaluate {args.split}"):
        rgb = read_image(args.root / "images" / f"{stem}.jpg")
        target = read_mask(args.root / "masks" / f"{stem}.jpg")
        record, arrays = evaluate_image(seg, rgb, target, args.threshold, args.mc_samples)
        record["stem"] = stem
        per_image.append(record)
        pooled = pooled + Confusion.from_masks(arrays["pred"], target)
        if "std" in arrays:
            mc_pooled = mc_pooled + Confusion.from_masks(arrays["mc_pred"], target)
            u, e = subsample_for_auroc(arrays["std"], arrays["pred"], target, 20_000)
            unc_pixels.append(u), err_pixels.append(e)
        cases.append({"stem": stem, "rgb": rgb, "target": target, "dice": record["dice"],
                      **{k: arrays[k] for k in ("pred", "std") if k in arrays}})

    results = {"split": args.split, "threshold": args.threshold,
               "checkpoint": {"path": args.checkpoint.name, **seg.meta,
                              "config": seg.cfg.to_dict()},
               "device": str(seg.device), "deterministic": summarize(per_image, pooled),
               "by_polyp_size": stratify_by_size(per_image),
               "timing_ms": {"deterministic_mean": float(np.mean([m["ms"] for m in per_image]))}}
    if args.mc_samples > 0:
        mc_records = [m["mc_mean"] for m in per_image]
        results["mc_dropout"] = {
            "samples": args.mc_samples, "mean_prediction": summarize(mc_records, mc_pooled),
            "pixel_error_auroc": auroc(np.concatenate(unc_pixels), np.concatenate(err_pixels)),
            "mean_std_on_error_pixels": float(np.mean(
                [m["uncertainty"]["mean_unc_on_errors"] for m in per_image])),
            "mean_std_on_correct_pixels": float(np.mean(
                [m["uncertainty"]["mean_unc_on_correct"] for m in per_image])),
            "spearman_dice_vs_mean_std": spearman(
                [m["dice"] for m in per_image],
                [m["uncertainty"]["mean_std"] for m in per_image]),
        }
        results["timing_ms"]["mc_mean"] = float(np.mean([m["mc_ms"] for m in per_image]))
    results["per_image"] = per_image
    (args.reports_dir / f"results_{args.split}.json").write_text(json.dumps(results, indent=2))

    build_galleries(cases, args.reports_dir, args.gallery_size)
    plot_dice_histogram([m["dice"] for m in per_image], args.reports_dir / "dice_histogram.png")
    if args.mc_samples > 0:
        plot_dice_vs_size_and_uncertainty(per_image, args.reports_dir / "error_analysis.png")
    history_path = args.reports_dir / "training_history.json"
    if history_path.exists():
        plot_training_curves(json.loads(history_path.read_text())["history"],
                             args.reports_dir / "training_curves.png")

    det = results["deterministic"]["mean_per_image"]
    print(f"\n{args.split}: {len(per_image)} images  mean Dice {det['dice']:.4f}  "
          f"mIoU {det['iou']:.4f}  precision {det['precision']:.4f}  recall {det['recall']:.4f}  "
          f"{results['timing_ms']['deterministic_mean']:.1f} ms/img")
    for bucket, stats in results["by_polyp_size"].items():
        print(f"  {bucket:16s} n={stats['num_images']:3d}  Dice {stats['dice']:.4f}")
    if args.mc_samples > 0:
        mc = results["mc_dropout"]
        print(f"MC dropout ({args.mc_samples} samples): mean-pred Dice "
              f"{mc['mean_prediction']['mean_per_image']['dice']:.4f}  pixel-error AUROC "
              f"{mc['pixel_error_auroc']:.3f}  std on errors {mc['mean_std_on_error_pixels']:.3f} "
              f"vs correct {mc['mean_std_on_correct_pixels']:.3f}")


if __name__ == "__main__":
    main()

"""Export the U-Net to ONNX (FP32 / FP16 / INT8) and CoreML, check parity, re-score the test set.

For every variant the report gives the file size, how far its probabilities and masks move
from the PyTorch model (max probability difference, share of pixels whose mask flips, Dice
between the two masks) and the test-split metrics recomputed from the variant's own masks.

Example:
    polyp-export                    # all variants, test split
    polyp-export --no-coreml --skip-export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import KVASIR_DIR, MODELS_DIR, REPORTS_DIR
from .data import load_or_create_split, read_image, read_mask
from .metrics import Confusion, binarize, summarize

EXPORT_DIR = MODELS_DIR / "onnx"
DEFAULT_CHECKPOINT = MODELS_DIR / "unet_resnet34.pt"
FILES = {"onnx_fp32": "polyp_unet_fp32.onnx", "onnx_fp16": "polyp_unet_fp16.onnx",
         "onnx_int8": "polyp_unet_int8.onnx", "coreml_fp16": "polyp_unet_fp16.mlpackage",
         "coreml_fp32": "polyp_unet_fp32.mlpackage"}


def file_mb(path: Path) -> float:
    if path.is_dir():  # .mlpackage
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
    return path.stat().st_size / 1e6


def mask_agreement(ref: np.ndarray, other: np.ndarray, threshold: float = 0.5) -> dict:
    """Probability and mask differences between two HxW probability maps."""
    a, b = binarize(ref, threshold), binarize(other, threshold)
    return {"max_prob_diff": float(np.abs(ref - other).max()),
            "flipped_pixels": float((a != b).mean()),
            "mask_dice": Confusion.from_masks(b, a).dice()}


def calibration_batches(stems: list[str], root: Path, size: int, n: int, seed: int = 0):
    from .export import image_batch

    picks = np.random.default_rng(seed).choice(len(stems), size=min(n, len(stems)),
                                               replace=False)
    for i in picks:
        yield image_batch(read_image(root / "images" / f"{stems[i]}.jpg"), size)


def build_variants(checkpoint: Path, train: list[str], root: Path, out_dir: Path, coreml: bool,
                   calib: int) -> dict[str, Path]:
    from . import export

    wrapper, size = export.load_wrapped(checkpoint)
    paths = {"onnx_fp32": export.export_onnx(wrapper, size, out_dir / FILES["onnx_fp32"])}
    paths["onnx_fp16"] = export.to_fp16(paths["onnx_fp32"], out_dir / FILES["onnx_fp16"])
    paths["onnx_int8"] = export.to_int8(paths["onnx_fp32"], out_dir / FILES["onnx_int8"],
                                        calibration_batches(train, root, size, calib))
    if coreml:
        try:
            for bits in (16, 32):
                key = f"coreml_fp{bits}"
                paths[key] = export.export_coreml(wrapper, size, out_dir / FILES[key],
                                                  fp16=bits == 16)
        except Exception as err:  # coremltools lags behind new torch releases
            print(f"CoreML export failed: {err}")
    return paths


def load_segmenter(kind: str, path: Path, size: int):
    from .export import CoreMLSegmenter, OnnxSegmenter

    seg = OnnxSegmenter(path) if kind.startswith("onnx") else CoreMLSegmenter(path, size)
    seg.name = kind
    return seg


def score(probs: list[np.ndarray], targets: list[np.ndarray], threshold: float) -> dict:
    per_image, pooled = [], Confusion(0, 0, 0, 0)
    for p, t in zip(probs, targets, strict=True):
        conf = Confusion.from_masks(binarize(p, threshold), t)
        per_image.append(conf.as_dict())
        pooled = pooled + conf
    out = summarize(per_image, pooled)
    out["below_0.5_dice"] = int(sum(m["dice"] < 0.5 for m in per_image))
    return out


def markdown(report: dict) -> str:
    lines = ["# Exported models: parity and test-split accuracy\n",
             f"{report['test_images']} test images (Kvasir-SEG, committed split), threshold "
             f"{report['threshold']}. Parity is measured against the PyTorch model on CPU, at the "
             "original image resolution.\n",
             "| Variant | Size (MB) | Max prob diff | Flipped pixels (mean / max) | Mask Dice vs "
             "PyTorch (min) | Mean Dice | Mean IoU | Images < 0.5 Dice |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in report["variants"].items():
        m, p = r["metrics"]["mean_per_image"], r.get("parity")
        par = (f"{p['max_prob_diff']:.2g} | {100 * p['flipped_mean']:.3f} % / "
               f"{100 * p['flipped_max']:.2f} % | {p['mask_dice_min']:.4f}"
               if p else "reference | | ")
        lines.append(f"| {name} | {r['size_mb']:.1f} | {par} | {m['dice']:.4f} | "
                     f"{m['iou']:.4f} | {r['metrics']['below_0.5_dice']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--root", type=Path, default=KVASIR_DIR)
    parser.add_argument("--out-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calib", type=int, default=128, help="INT8 calibration images (train)")
    parser.add_argument("--no-coreml", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=8,
                        help="samples in the MC-dropout graph (0 skips it)")
    parser.add_argument("--skip-export", action="store_true", help="reuse files in --out-dir")
    args = parser.parse_args()

    from .model import PolypSegmenter

    split = load_or_create_split(args.root)
    stems = split[args.split]
    if not stems:
        raise SystemExit(f"No images under {args.root}. Run `polyp-download`.")
    if args.skip_export:
        paths = {k: args.out_dir / v for k, v in FILES.items() if (args.out_dir / v).exists()}
    else:
        paths = build_variants(args.checkpoint, split["train"], args.root, args.out_dir,
                               not args.no_coreml, args.calib)

    mc_paths = {}
    if args.mc_samples > 0:
        from .mc_export import MC_FILES, build_mc_variants

        if args.skip_export:
            mc_paths = {k: args.out_dir / v.format(t=args.mc_samples)
                        for k, v in MC_FILES.items()}
        else:
            from .export import load_wrapped

            mc_paths = build_mc_variants(
                args.checkpoint, args.out_dir, args.mc_samples,
                calibration_batches(split["train"], args.root, load_wrapped(args.checkpoint)[1],
                                    args.calib))

    images = [read_image(args.root / "images" / f"{s}.jpg") for s in stems]
    targets = [read_mask(args.root / "masks" / f"{s}.jpg") for s in stems]
    reference = PolypSegmenter(args.checkpoint, device="cpu")
    ref_probs = [reference.predict(im) for im in images]
    report = {"checkpoint": str(args.checkpoint), "split": args.split,
              "test_images": len(stems), "threshold": args.threshold, "variants": {}}
    report["variants"]["pytorch_fp32"] = {"size_mb": file_mb(args.checkpoint),
                                          "metrics": score(ref_probs, targets, args.threshold)}
    for kind, path in paths.items():
        seg = load_segmenter(kind, path, reference.size)
        probs = [seg.predict(im) for im in images]
        agree = [mask_agreement(r, p, args.threshold) for r, p in zip(ref_probs, probs,
                                                                       strict=True)]
        flipped = [a["flipped_pixels"] for a in agree]
        report["variants"][kind] = {
            "file": path.name, "size_mb": file_mb(path),
            "parity": {"max_prob_diff": max(a["max_prob_diff"] for a in agree),
                       "flipped_mean": float(np.mean(flipped)),
                       "flipped_max": float(np.max(flipped)),
                       "mask_dice_min": min(a["mask_dice"] for a in agree),
                       "mask_dice_mean": float(np.mean([a["mask_dice"] for a in agree]))},
            "metrics": score(probs, targets, args.threshold)}
        print(kind, json.dumps(report["variants"][kind]["parity"]),
              f"mean Dice {report['variants'][kind]['metrics']['mean_per_image']['dice']:.4f}")

    text = markdown(report)
    if mc_paths:
        from . import mc_export

        report["mc_dropout"] = mc_export.evaluate_mc(args.checkpoint, mc_paths, images, targets,
                                                     ref_probs, args.mc_samples, args.threshold)
        text += mc_export.markdown(report["mc_dropout"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "export.json").write_text(json.dumps(report, indent=2))
    (REPORTS_DIR / "export.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

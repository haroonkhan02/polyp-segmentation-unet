"""Segment polyps in an image or a folder of images and save overlays (+ uncertainty maps).

Example:
    polyp-predict data/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg --mc-samples 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MODELS_DIR, OUTPUTS_DIR
from .data import read_image
from .metrics import binarize
from .model import PolypSegmenter
from .uncertainty import mean_std, summarize_uncertainty
from .viz import blend_heatmap, label, overlay_mask, save_rgb, tile

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [source]


def predict_file(seg: PolypSegmenter, path: Path, out_dir: Path, threshold: float,
                 mc_samples: int) -> dict:
    rgb = read_image(path)
    probs = seg.predict(rgb)
    pred = binarize(probs, threshold)
    panels = [label(rgb, path.stem[:16]), label(overlay_mask(rgb, pred), "prediction"),
              label(blend_heatmap(rgb, probs, vmax=1.0), "probability")]
    info = {"file": path.name, "polyp_fraction": float(pred.mean()),
            "inference_ms": seg.last_inference_ms}
    if mc_samples > 0:
        mean, std = mean_std(seg.predict_mc(rgb, mc_samples))
        panels.append(label(blend_heatmap(rgb, std, vmax=0.5), f"MC-dropout std ({mc_samples})"))
        info["uncertainty"] = summarize_uncertainty(std, pred)
    save_rgb(out_dir / f"{path.stem}_pred.jpg", tile(panels, 320))
    save_rgb(out_dir / f"{path.stem}_mask.png", (pred * 255).astype("uint8"))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="image file or folder")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "unet_resnet34.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("-o", "--output", type=Path, default=OUTPUTS_DIR / "predictions")
    args = parser.parse_args()

    seg = PolypSegmenter(args.checkpoint, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    infos = [predict_file(seg, p, args.output, args.threshold, args.mc_samples)
             for p in collect_images(args.source)]
    (args.output / "predictions.json").write_text(json.dumps(infos, indent=2))
    for info in infos:
        print(f"{info['file']}: polyp covers {info['polyp_fraction'] * 100:.1f}% of the image "
              f"({info['inference_ms']:.1f} ms)")
    print(f"wrote {len(infos)} overlays to {args.output}")


if __name__ == "__main__":
    main()

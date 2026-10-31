"""Check the browser demo's JavaScript pipeline against the Python one on the test split.

Dumps decoded test frames, runs ``scripts/web_parity/parity.mjs`` (web/polyp.js +
onnxruntime-web WASM, the same code the static Space runs) and compares its maps with
``OnnxSegmenter`` on the same ONNX file: flipped mask pixels, mask Dice between the two, and the
test metrics computed from each. ``--mc`` does the same for the MC-dropout graph, where only
the scores can be compared (sampling is random).

Example (after `npm install` in scripts/web_parity):
    python scripts/web_parity.py
    python scripts/web_parity.py --mc --model models/onnx/polyp_unet_mc8_int8.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from polypseg.config import KVASIR_DIR, OUTPUTS_DIR, REPORTS_DIR
from polypseg.data import load_or_create_split, read_image, read_mask
from polypseg.export import OnnxSegmenter
from polypseg.export_models import mask_agreement, score
from polypseg.mc_export import OnnxMC, score_mc

ROOT = Path(__file__).resolve().parents[1]


def dump(images: list[np.ndarray], out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    meta = []
    for i, rgb in enumerate(images):
        rgb.tofile(out / f"{i}.rgb")
        meta.append({"file": f"{i}.rgb", "width": rgb.shape[1], "height": rgb.shape[0]})
    (out / "meta.json").write_text(json.dumps({"images": meta}))


def js_maps(work: Path, i: int, shape: tuple[int, int], count: int) -> list[np.ndarray]:
    return [np.fromfile(work / f"{i}.rgb.js{k}", np.float32).reshape(shape)
            for k in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "models/onnx/polyp_unet_int8.onnx")
    parser.add_argument("--mc", action="store_true", help="the model is the MC-dropout graph")
    parser.add_argument("--root", type=Path, default=KVASIR_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--work", type=Path, default=OUTPUTS_DIR / "web_parity")
    args = parser.parse_args()

    stems = load_or_create_split(args.root)["test"]
    images = [read_image(args.root / "images" / f"{s}.jpg") for s in stems]
    targets = [read_mask(args.root / "masks" / f"{s}.jpg") for s in stems]
    dump(images, args.work)
    cmd = ["node", str(ROOT / "scripts/web_parity/parity.mjs"), str(args.work), str(args.model)]
    subprocess.run(cmd + (["--mc"] if args.mc else []), check=True)
    js = json.loads((args.work / "js_meta.json").read_text())
    report = {"model": args.model.name, "images": len(stems),
              "onnxruntime_web": js["ort_version"],
              "js_ms_per_frame_node_wasm": js["ms_per_frame"]}

    if args.mc:
        int8 = OnnxSegmenter(ROOT / "models/onnx/polyp_unet_int8.onnx")
        det = [int8.predict(im) for im in images]  # errors of the model the demo shows
        run = OnnxMC(args.model)
        js_mc = [tuple(js_maps(args.work, i, im.shape[:2], 2)) for i, im in enumerate(images)]
        report["errors_of"] = "polyp_unet_int8.onnx"
        report["python"] = score_mc([run(im) for im in images], det, targets, args.threshold)
        report["javascript"] = score_mc(js_mc, det, targets, args.threshold)
        out_name = "web_parity_mc.json"
    else:
        seg = OnnxSegmenter(args.model)
        py = [seg.predict(im) for im in images]
        js_probs = [js_maps(args.work, i, im.shape[:2], 1)[0] for i, im in enumerate(images)]
        agree = [mask_agreement(p, j, args.threshold) for p, j in zip(py, js_probs, strict=True)]
        report.update({
            "max_prob_diff": max(a["max_prob_diff"] for a in agree),
            "flipped_pixels_mean": float(np.mean([a["flipped_pixels"] for a in agree])),
            "flipped_pixels_max": max(a["flipped_pixels"] for a in agree),
            "mask_dice_min": min(a["mask_dice"] for a in agree),
            "python": score(py, targets, args.threshold),
            "javascript": score(js_probs, targets, args.threshold)})
        out_name = "web_parity.json"
    (REPORTS_DIR / out_name).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("python", "javascript")},
                     indent=2))
    for side in ("python", "javascript"):
        r = report[side]
        m = r.get("mean_per_image") or r["mean_prediction"]["mean_per_image"]
        extra = f", AUROC {r['pixel_error_auroc']:.3f}" if args.mc else ""
        print(f"{side}: mean Dice {m['dice']:.4f}{extra}")
    shutil.rmtree(args.work)


if __name__ == "__main__":
    main()

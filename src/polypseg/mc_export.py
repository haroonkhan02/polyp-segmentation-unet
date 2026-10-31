"""Export and check the Monte-Carlo-dropout graph (mean + std maps) used by the browser demo.

Sampling is random, so parity cannot be checked pixel by pixel. Instead the PyTorch MC
prediction and each exported MC graph are scored the same way ``polyp-evaluate`` scores
uncertainty: Dice of the mean prediction, and the pooled pixel-level AUROC of the std map as a
detector of the deterministic model's errors (20,000 random pixels per image, seed 0).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .metrics import Confusion, binarize, summarize
from .uncertainty import auroc, mean_std, subsample_for_auroc

MC_FILES = {"onnx_mc_fp32": "polyp_unet_mc{t}_fp32.onnx",
            "onnx_mc_int8": "polyp_unet_mc{t}_int8.onnx"}


def build_mc_variants(checkpoint: Path, out_dir: Path, samples: int, calibration) -> dict:
    from .export import export_mc_onnx, to_int8

    fp32 = export_mc_onnx(checkpoint, out_dir / MC_FILES["onnx_mc_fp32"].format(t=samples),
                          samples)
    int8 = to_int8(fp32, out_dir / MC_FILES["onnx_mc_int8"].format(t=samples), calibration)
    return {"onnx_mc_fp32": fp32, "onnx_mc_int8": int8}


class OnnxMC:
    """Runs an exported MC graph and returns (mean, std) at the image's resolution."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        from .export import INPUT_NAME, MC_OUTPUTS

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input, self.outputs = INPUT_NAME, list(MC_OUTPUTS)
        self.size = int(self.session.get_inputs()[0].shape[2])

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from .export import image_batch

        mean, std = self.session.run(self.outputs, {self.input: image_batch(rgb, self.size)})
        h, w = rgb.shape[:2]
        return tuple(cv2.resize(m[0, 0].astype(np.float32), (w, h),
                                interpolation=cv2.INTER_LINEAR) for m in (mean, std))


def score_mc(maps, det_probs, targets, threshold: float) -> dict:
    """Mean-prediction metrics + std-vs-error AUROC, scored like ``polyp-evaluate``."""
    per_image, pooled, unc, err = [], Confusion(0, 0, 0, 0), [], []
    for (mean, std), probs, target in zip(maps, det_probs, targets, strict=True):
        conf = Confusion.from_masks(binarize(mean, threshold), target)
        per_image.append(conf.as_dict())
        pooled = pooled + conf
        u, e = subsample_for_auroc(std, binarize(probs, threshold), target, 20_000)
        unc.append(u)
        err.append(e)
    unc_all, err_all = np.concatenate(unc), np.concatenate(err)
    return {"mean_prediction": summarize(per_image, pooled),
            "pixel_error_auroc": auroc(unc_all, err_all),
            "mean_std_on_error_pixels": float(unc_all[err_all].mean()),
            "mean_std_on_correct_pixels": float(unc_all[~err_all].mean())}


def evaluate_mc(checkpoint: Path, paths: dict, images, targets, det_probs, samples: int,
                threshold: float, seed: int = 0) -> dict:
    from .export_models import file_mb
    from .model import PolypSegmenter

    torch.manual_seed(seed)
    seg = PolypSegmenter(checkpoint, device="cpu")
    out = {"samples": samples, "variants": {
        "pytorch_mc": score_mc([mean_std(seg.predict_mc(im, samples)) for im in images],
                               det_probs, targets, threshold)}}
    for kind, path in paths.items():
        run = OnnxMC(path)
        out["variants"][kind] = {"file": path.name, "size_mb": file_mb(path),
                                 **score_mc([run(im) for im in images], det_probs, targets,
                                            threshold)}
    return out


def markdown(mc: dict) -> str:
    lines = [f"\n## Monte-Carlo dropout graph ({mc['samples']} samples)\n",
             "Random sampling, so this compares scores, not pixels. AUROC: std map as a detector "
             "of the deterministic PyTorch model's pixel errors. With 8 samples the AUROC of the "
             "same model moves by up to 0.02 between random seeds, so differences of that size "
             "between rows are sampling noise.\n",
             "| Variant | Size (MB) | Mean-prediction Dice | Pixel-error AUROC | Std on errors / "
             "correct |", "|---|---|---|---|---|"]
    for name, r in mc["variants"].items():
        size = f"{r['size_mb']:.1f}" if "size_mb" in r else "(checkpoint)"
        lines.append(f"| {name} | {size} | "
                     f"{r['mean_prediction']['mean_per_image']['dice']:.4f} | "
                     f"{r['pixel_error_auroc']:.3f} | {r['mean_std_on_error_pixels']:.3f} / "
                     f"{r['mean_std_on_correct_pixels']:.4f} |")
    return "\n".join(lines) + "\n"

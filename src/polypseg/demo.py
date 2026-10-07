"""Gradio demo: upload a colonoscopy frame, get the polyp mask, probability map and
MC-dropout uncertainty map. Research demo only: not a medical device.

Example:
    polyp-demo --checkpoint models/unet_resnet34.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import MODELS_DIR
from .metrics import binarize
from .model import PolypSegmenter
from .uncertainty import mean_std, summarize_uncertainty
from .viz import blend_heatmap, overlay_mask

DISCLAIMER = (
    "**Not for clinical use.** This is a research/portfolio demo trained on 800 Kvasir-SEG "
    "images. It has not been validated on any other scanner, population or clinical setting."
)


def build_app(seg: PolypSegmenter):
    import gradio as gr

    def run(image: np.ndarray, threshold: float, mc_samples: int):
        if image is None:
            return None, None, None, "Upload an image."
        probs = seg.predict(image)
        pred = binarize(probs, threshold)
        overlay = overlay_mask(image, pred)
        prob_map = blend_heatmap(image, probs, vmax=1.0)
        text = (f"Polyp covers **{pred.mean() * 100:.1f}%** of the frame · inference "
                f"{seg.last_inference_ms:.0f} ms on {seg.device}")
        if mc_samples > 0:
            _, std = mean_std(seg.predict_mc(image, int(mc_samples)))
            unc_map = blend_heatmap(image, std, vmax=0.5)
            stats = summarize_uncertainty(std, pred)
            text += (f"\n\nMC dropout ({int(mc_samples)} passes): mean std "
                     f"{stats['mean_std']:.3f}, on the boundary "
                     f"{stats['mean_std_on_boundary']:.3f}, uncertain pixels (std > 0.1): "
                     f"{stats['uncertain_fraction'] * 100:.1f}%")
        else:
            unc_map = None
        return overlay, prob_map, unc_map, text

    with gr.Blocks(title="Polyp segmentation") as app:
        gr.Markdown("# Colon polyp segmentation (U-Net, Kvasir-SEG)\n" + DISCLAIMER)
        with gr.Row():
            inp = gr.Image(label="Colonoscopy frame", type="numpy")
            out_overlay = gr.Image(label="Predicted polyp")
        with gr.Row():
            out_prob = gr.Image(label="Probability map")
            out_unc = gr.Image(label="MC-dropout uncertainty (std)")
        threshold = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="Mask threshold")
        samples = gr.Slider(0, 50, 20, step=5, label="MC-dropout passes (0 = off)")
        summary = gr.Markdown()
        btn = gr.Button("Segment", variant="primary")
        btn.click(run, [inp, threshold, samples], [out_overlay, out_prob, out_unc, summary])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "unet_resnet34.pt")
    parser.add_argument("--device")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    seg = PolypSegmenter(args.checkpoint, args.device)
    build_app(seg).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

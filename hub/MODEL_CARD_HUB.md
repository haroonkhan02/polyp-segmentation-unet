---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - polyp-segmentation
  - kvasir-seg
  - unet
  - onnx
  - uncertainty
---

# Polyp segmentation U-Net (ResNet-34, Kvasir-SEG)

Binary colon-polyp segmentation: a U-Net with an ImageNet-pretrained ResNet-34 encoder and
`Dropout2d(0.2)` after each decoder block, trained on 800 Kvasir-SEG frames. Training code,
evaluation and error analysis: [github.com/haroonkhan02/polyp-segmentation-unet](https://github.com/haroonkhan02/polyp-segmentation-unet).
Browser demo: [spaces/haroonx02/polyp-segmentation-browser](https://huggingface.co/spaces/haroonx02/polyp-segmentation-browser).

**Not for clinical use.** Research and portfolio model trained on one hospital's data, with no
validation on other endoscopes or populations and no regulatory clearance.

## Files

| File | What it is | Size | Test mean Dice |
|---|---|---|---|
| `unet_resnet34.pt` | PyTorch checkpoint (`state_dict` + training config) | 98 MB | 0.904 |
| `onnx/polyp_unet_fp16.onnx` | ONNX, FP16 weights, float32 input/output | 49 MB | 0.904 |
| `onnx/polyp_unet_int8.onnx` | ONNX, static INT8 (QDQ, per-channel, 128 train frames for calibration) | 25 MB | 0.904 |
| `onnx/polyp_unet_mc8_int8.onnx` | INT8 Monte-Carlo-dropout graph: 8 decoder samples, outputs `mean` and `std` | 25 MB | 0.904 (mean) |

ONNX input `image`: float32 `(1, 3, 320, 320)`, RGB in [0, 1] (resize with area interpolation,
divide by 255; ImageNet normalisation is inside the graph). Output `probs`: `(1, 1, 320, 320)`
polyp probability (sigmoid inside the graph); resize back bilinearly and threshold at 0.5.

## Evaluation

Held-out test split of 100 Kvasir-SEG images (committed seeded split), threshold 0.5, original
resolution. Mean per-image Dice 0.904 and IoU 0.844 for the PyTorch model; the ONNX FP32/FP16
and INT8 files score the same to within 0.001 (INT8: 0.9043 vs 0.9036). By polyp size: small
(< 5 % of the frame) 0.868, medium 0.938, large 0.867. The MC-dropout std map detects the
model's pixel errors with an AUROC of about 0.92–0.94 (8 samples; it varies by up to 0.02
between random seeds). It flags uncertain boundaries well but misses regions the model is
confidently wrong about.

## Limitations

Single source (one Norwegian hospital), every frame contains a polyp (false-positive rate on
healthy mucosa unknown), 100 test images (about ±0.025 on mean Dice), errors are mostly
under-segmentation of flat polyps.

## Data

Kvasir-SEG: D. Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MMM 2020. CC BY 4.0.

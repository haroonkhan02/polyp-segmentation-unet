---
title: Polyp Segmentation in the Browser
emoji: 🔬
colorFrom: pink
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Kvasir-SEG U-Net with MC-dropout uncertainty, in your browser
models:
  - haroonx02/polyp-segmentation-unet-resnet34
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

A U-Net polyp segmentation model (ResNet-34 encoder, trained on Kvasir-SEG) exported to ONNX,
quantised to INT8 and run client-side with onnxruntime-web (WebAssembly). A second graph keeps
decoder dropout on and returns a Monte-Carlo uncertainty map. Pre- and post-processing are
ported to JavaScript and checked against the Python pipeline.

Research demo only, not for clinical use. Code, training, evaluation and error analysis:
[github.com/haroonkhan02/polyp-segmentation-unet](https://github.com/haroonkhan02/polyp-segmentation-unet).

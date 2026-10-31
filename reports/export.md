# Exported models: parity and test-split accuracy

100 test images (Kvasir-SEG, committed split), threshold 0.5. Parity is measured against the PyTorch model on CPU, at the original image resolution.

| Variant | Size (MB) | Max prob diff | Flipped pixels (mean / max) | Mask Dice vs PyTorch (min) | Mean Dice | Mean IoU | Images < 0.5 Dice |
|---|---|---|---|---|---|---|---|
| pytorch_fp32 | 97.9 | reference | |  | 0.9036 | 0.8436 | 4 |
| onnx_fp32 | 97.7 | 1.6e-05 | 0.000 % / 0.00 % | 1.0000 | 0.9036 | 0.8436 | 4 |
| onnx_fp16 | 48.9 | 0.0049 | 0.001 % / 0.04 % | 0.9992 | 0.9036 | 0.8436 | 4 |
| onnx_int8 | 24.7 | 0.41 | 0.145 % / 2.01 % | 0.8856 | 0.9043 | 0.8443 | 4 |
| coreml_fp16 | 48.9 | 0.031 | 0.006 % / 0.12 % | 0.9936 | 0.9037 | 0.8437 | 4 |
| coreml_fp32 | 97.8 | 8.5e-06 | 0.000 % / 0.00 % | 1.0000 | 0.9036 | 0.8436 | 4 |

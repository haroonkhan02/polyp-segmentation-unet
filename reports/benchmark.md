# Model latency per backend

arm64 macOS-27.0.1-arm64-arm-64bit; input 1x3x320x320; median and 90th percentile of 100 runs after 10 warm-up runs, network only.

| Backend | Size (MB) | Median (ms) | p90 (ms) |
|---|---|---|---|
| pytorch_fp32 (cpu) | 97.9 | 49.5 | 50.2 |
| pytorch_fp32 (mps) | 97.9 | 12.9 | 13.5 |
| onnx_fp32 (onnxruntime CPU) | 97.7 | 30.8 | 34.7 |
| onnx_fp16 (onnxruntime CPU) | 48.9 | 30.6 | 34.2 |
| onnx_int8 (onnxruntime CPU) | 24.7 | 21.7 | 24.2 |
| coreml_fp16 (all) | 48.9 | 1.9 | 2.0 |
| coreml_fp16 (cpu_and_gpu) | 48.9 | 4.6 | 4.8 |
| coreml_fp16 (cpu_only) | 48.9 | 10.8 | 10.9 |
| coreml_fp32 (all) | 97.8 | 7.7 | 7.8 |
| coreml_fp32 (cpu_and_gpu) | 97.8 | 7.7 | 7.8 |
| coreml_fp32 (cpu_only) | 97.8 | 24.6 | 24.9 |

"""Model latency per backend: PyTorch (MPS / CPU), ONNX Runtime CPU, CoreML.

Times the network alone on one real 320x320 test input (resizing and thresholding are the
same for every backend): warm-up runs first, then the median and 90th percentile.

Example:
    polyp-benchmark --runs 100
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .config import KVASIR_DIR, REPORTS_DIR
from .data import load_or_create_split, read_image
from .export_models import DEFAULT_CHECKPOINT, EXPORT_DIR, FILES, file_mb


def time_fn(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


def torch_runner(checkpoint: Path, device: str, batch: np.ndarray):
    from .export import load_wrapped

    wrapper, _ = load_wrapped(checkpoint)
    wrapper.to(device)
    x = torch.from_numpy(batch).to(device)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)

    @torch.no_grad()
    def run():
        wrapper(x)
        sync()

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--models", type=Path, default=EXPORT_DIR)
    parser.add_argument("--root", type=Path, default=KVASIR_DIR)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    from .export import image_batch, load_wrapped

    _, size = load_wrapped(args.checkpoint)
    stem = load_or_create_split(args.root)["test"][0]
    batch = image_batch(read_image(args.root / "images" / f"{stem}.jpg"), size)

    rows = {}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        rows[f"pytorch_fp32 ({dev})"] = {
            "size_mb": file_mb(args.checkpoint),
            **time_fn(torch_runner(args.checkpoint, dev, batch), args.runs, args.warmup)}
    import onnxruntime as ort

    for variant in ("fp32", "fp16", "int8"):
        path = args.models / FILES[f"onnx_{variant}"]
        if path.exists():
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            rows[f"onnx_{variant} (onnxruntime CPU)"] = {
                "size_mb": file_mb(path),
                **time_fn(lambda s=sess: s.run(None, {"image": batch}), args.runs, args.warmup)}
    try:
        import coremltools as ct
    except ImportError:
        ct = None
    for variant in ("fp16", "fp32"):
        path = args.models / FILES[f"coreml_{variant}"]
        if ct is None or not path.exists():
            continue
        for units in ("ALL", "CPU_AND_GPU", "CPU_ONLY"):
            model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
            rows[f"coreml_{variant} ({units.lower()})"] = {
                "size_mb": file_mb(path),
                **time_fn(lambda m=model: m.predict({"image": batch}), args.runs, args.warmup)}

    report = {"machine": f"{platform.machine()} {platform.platform()}",
              "onnxruntime": ort.__version__, "torch": torch.__version__,
              "input": f"1x3x{size}x{size}", "runs": args.runs, "results": rows}
    lines = ["# Model latency per backend\n",
             f"{report['machine']}; input {report['input']}; median and 90th percentile of "
             f"{args.runs} runs after {args.warmup} warm-up runs, network only.\n",
             "| Backend | Size (MB) | Median (ms) | p90 (ms) |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['size_mb']:.1f} | {v['median_ms']:.1f} | {v['p90_ms']:.1f} |"
              for k, v in rows.items()]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark.json").write_text(json.dumps(report, indent=2))
    (REPORTS_DIR / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

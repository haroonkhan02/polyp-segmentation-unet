"""Export the polyp U-Net to ONNX (FP32, FP16, INT8) and CoreML, and run the exported files.

The exported graph takes an RGB image in [0, 1] (1, 3, S, S) and returns the polyp
probability map (1, 1, S, S): ImageNet normalisation and the sigmoid live inside the graph, so
a browser or mobile client only has to resize the image and divide by 255. Dropout is exported
in eval mode (identity), i.e. the deterministic prediction. ``export_mc_onnx`` writes a second
graph that keeps decoder dropout on and returns the Monte-Carlo mean and standard deviation.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD
from .model import DropoutBlock, load_checkpoint

INPUT_NAME, OUTPUT_NAME = "image", "probs"
MC_OUTPUTS = ("mean", "std")


class ExportWrapper(torch.nn.Module):
    """Normalise -> U-Net -> sigmoid, so the exported graph is self-contained."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((x - self.mean) / self.std))


class _SampledDropout2d(torch.nn.Module):
    """Channel dropout that stays active in eval mode and traces to ONNX RandomUniformLike.

    ``nn.Dropout2d`` exports as an identity in eval mode and cannot be exported in training
    mode, so the MC graph draws its own per-channel keep mask.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep = (torch.rand_like(x[:, :, :1, :1]) >= self.p).to(x.dtype)
        return x * keep / (1.0 - self.p)


class MCDropoutWrapper(torch.nn.Module):
    """Normalise -> encoder once -> ``samples`` decoder passes with dropout -> mean and std.

    Dropout only sits in the decoder, so the encoder features are computed once and repeated:
    the same samples as ``PolypSegmenter.predict_mc`` at a fraction of the cost. Outputs are the
    mean probability and the per-pixel standard deviation, each (1, 1, S, S).
    """

    def __init__(self, model: torch.nn.Module, samples: int = 8):
        super().__init__()
        self.model = copy.deepcopy(model).eval()
        for block in self.model.decoder.blocks:
            if isinstance(block, DropoutBlock):
                block.dropout = _SampledDropout2d(block.dropout.p)
        self.samples = samples
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.encoder((x - self.mean) / self.std)
        features = [f.repeat(self.samples, 1, 1, 1) for f in features]
        probs = torch.sigmoid(self.model.segmentation_head(self.model.decoder(features)))
        mean = probs.mean(dim=0, keepdim=True)
        std = ((probs - mean) ** 2).mean(dim=0, keepdim=True).sqrt()  # population std, as numpy
        return mean, std


def load_wrapped(checkpoint: str | Path) -> tuple[ExportWrapper, int]:
    """Checkpoint -> (eval-mode wrapper on CPU, square input size)."""
    model, cfg, _ = load_checkpoint(checkpoint)
    return ExportWrapper(model).eval(), cfg.image_size


def export_mc_onnx(checkpoint: str | Path, out: Path, samples: int = 8, opset: int = 17) -> Path:
    """MC-dropout graph: input ``image`` -> outputs ``mean`` and ``std``."""
    import onnx

    model, cfg, _ = load_checkpoint(checkpoint)
    wrapper = MCDropoutWrapper(model, samples).eval()
    size = cfg.image_size
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, (torch.rand(1, 3, size, size),), str(out),
                      input_names=[INPUT_NAME], output_names=list(MC_OUTPUTS),
                      opset_version=opset, dynamo=False)
    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(out))
    return out


def image_batch(rgb: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 image -> (1, 3, S, S) float32 in [0, 1], the exported graph's input."""
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray((small.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])


def export_onnx(wrapper: ExportWrapper, size: int, out: Path, opset: int = 17) -> Path:
    """Trace the wrapper to a single-file ONNX model with a fixed 1x3xSxS input."""
    import onnx

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, (torch.rand(1, 3, size, size),), str(out),
                      input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                      opset_version=opset, dynamo=False)
    model = onnx.load(str(out))  # re-save so weights are embedded, not external data
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    return out


def to_fp16(src: Path, out: Path) -> Path:
    """FP16 weights and activations; input and output stay float32 for the callers."""
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(out))
    return out


class _Calibration:
    """onnxruntime CalibrationDataReader over a list of preprocessed batches."""

    def __init__(self, batches: Iterable[np.ndarray]):
        self._it = iter(batches)

    def get_next(self):
        batch = next(self._it, None)
        return None if batch is None else {INPUT_NAME: batch}


def to_int8(src: Path, out: Path, calibration: Iterable[np.ndarray]) -> Path:
    """Static INT8 (QDQ, per-channel weights) calibrated on real images.

    Everything after the last convolution (the sigmoid, and in the MC graph the mean and
    standard deviation) stays in float: an 8-bit probability would put the 0.5 threshold on a
    coarse grid, and a quantised std would round small uncertainties to zero.
    """
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = out.with_name(out.stem + "_prep.onnx")
    quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
    nodes = list(onnx.load(str(prep)).graph.node)  # topologically sorted
    last_conv = max(i for i, n in enumerate(nodes) if n.op_type == "Conv")
    keep_float = [n.name for n in nodes[last_conv + 1:]]
    quantize_static(str(prep), str(out), _Calibration(calibration),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=keep_float)
    prep.unlink()
    return out


def export_coreml(wrapper: ExportWrapper, size: int, out: Path, fp16: bool = True) -> Path:
    """CoreML ML Program (Apple Neural Engine / GPU / CPU chosen by the OS at run time)."""
    import coremltools as ct

    traced = torch.jit.trace(wrapper, torch.rand(1, 3, size, size))
    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(name=INPUT_NAME, shape=(1, 3, size, size))],
                         outputs=[ct.TensorType(name=OUTPUT_NAME)],
                         convert_to="mlprogram", compute_precision=precision,
                         minimum_deployment_target=ct.target.macOS13)
    mlmodel.save(str(out))
    return out


class _ExportedSegmenter:
    """Same interface as ``PolypSegmenter.predict``: RGB image -> HxW probability map."""

    name = "exported"

    def __init__(self, size: int):
        self.size = size
        self.last_inference_ms = 0.0

    def run(self, batch: np.ndarray) -> np.ndarray:  # (1, 1, S, S) probabilities
        raise NotImplementedError

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        batch = image_batch(rgb, self.size)
        start = time.perf_counter()
        probs = np.asarray(self.run(batch), dtype=np.float32)[0, 0]
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        h, w = rgb.shape[:2]
        return cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)


class OnnxSegmenter(_ExportedSegmenter):
    def __init__(self, path: str | Path, threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        super().__init__(int(self.session.get_inputs()[0].shape[2]))
        self.name = f"onnx:{Path(path).stem}"

    def run(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0]


class CoreMLSegmenter(_ExportedSegmenter):
    def __init__(self, path: str | Path, size: int, compute_units: str = "ALL"):
        import coremltools as ct

        self.model = ct.models.MLModel(str(path),
                                       compute_units=getattr(ct.ComputeUnit, compute_units))
        super().__init__(size)
        self.name = f"coreml:{compute_units.lower()}"

    def run(self, batch: np.ndarray) -> np.ndarray:
        return self.model.predict({INPUT_NAME: batch})[OUTPUT_NAME]

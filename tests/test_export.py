import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxconverter_common")

from polypseg import export  # noqa: E402
from polypseg.data import preprocess  # noqa: E402
from polypseg.model import PolypSegmenter  # noqa: E402

from .conftest import synthetic_case  # noqa: E402


@pytest.fixture(scope="module")
def onnx_fp32(tiny_checkpoint, tmp_path_factory):
    wrapper, size = export.load_wrapped(tiny_checkpoint)
    return export.export_onnx(wrapper, size, tmp_path_factory.mktemp("onnx") / "m.onnx")


def _image(seed: int = 0) -> np.ndarray:
    rgb, _ = synthetic_case(size=200, radius=40, seed=seed)
    return rgb


def test_wrapper_matches_segmenter(tiny_checkpoint):
    wrapper, size = export.load_wrapped(tiny_checkpoint)
    rgb = _image()
    with torch.no_grad():
        probs = wrapper(torch.from_numpy(export.image_batch(rgb, size)))[0, 0].numpy()
    seg = PolypSegmenter(tiny_checkpoint, device="cpu")
    with torch.no_grad():  # the segmenter's map before it is resized back to the image
        ref = torch.sigmoid(seg.model(preprocess(rgb, size)))[0, 0].numpy()
    np.testing.assert_allclose(probs, ref, atol=1e-5)
    assert wrapper.training is False


def test_onnx_fp32_parity_and_io(onnx_fp32, tiny_checkpoint):
    det = export.OnnxSegmenter(onnx_fp32)
    assert det.size == 64
    rgb = _image(1)
    ref = PolypSegmenter(tiny_checkpoint, device="cpu").predict(rgb)
    probs = det.predict(rgb)
    assert probs.shape == rgb.shape[:2] and probs.dtype == np.float32
    assert np.abs(probs - ref).max() < 1e-4


def test_onnx_is_deterministic_despite_dropout(onnx_fp32):
    det = export.OnnxSegmenter(onnx_fp32)
    rgb = _image(2)
    np.testing.assert_array_equal(det.predict(rgb), det.predict(rgb))


def test_fp16_halves_the_file_and_stays_close(onnx_fp32, tmp_path):
    fp16 = export.to_fp16(onnx_fp32, tmp_path / "m16.onnx")
    assert fp16.stat().st_size < 0.6 * onnx_fp32.stat().st_size
    rgb = _image(3)
    a = export.OnnxSegmenter(onnx_fp32).predict(rgb)
    b = export.OnnxSegmenter(fp16).predict(rgb)
    assert b.dtype == np.float32 and np.abs(a - b).max() < 0.05


def test_int8_quantises_and_keeps_output_sigmoid_in_float(onnx_fp32, tmp_path):
    import onnx

    batches = [export.image_batch(_image(s), 64) for s in range(4)]
    int8 = export.to_int8(onnx_fp32, tmp_path / "m8.onnx", batches)
    assert int8.stat().st_size < 0.4 * onnx_fp32.stat().st_size
    assert not (tmp_path / "m8_prep.onnx").exists()
    graph = onnx.load(str(int8)).graph
    last = next(n for n in graph.node if export.OUTPUT_NAME in n.output)
    assert last.op_type == "Sigmoid"
    probs = export.OnnxSegmenter(int8).predict(_image(5))
    assert probs.min() >= 0.0 and probs.max() <= 1.0

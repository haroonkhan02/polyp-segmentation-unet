import numpy as np
import torch

from polypseg.model import (
    DropoutBlock,
    PolypSegmenter,
    build_model,
    enable_mc_dropout,
    load_checkpoint,
)


def test_build_model_wraps_decoder_blocks_with_dropout():
    model = build_model("resnet18", pretrained=False, dropout=0.2)
    assert all(isinstance(b, DropoutBlock) for b in model.decoder.blocks)
    plain = build_model("resnet18", pretrained=False, dropout=0.0)
    assert not any(isinstance(b, DropoutBlock) for b in plain.decoder.blocks)
    with torch.no_grad():
        out = model.eval()(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_eval_is_deterministic_but_mc_dropout_is_stochastic():
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False, dropout=0.5).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a, b = model(x), model(x)
        enable_mc_dropout(model)
        c, d = model(x), model(x)
    assert torch.equal(a, b)
    assert not torch.equal(c, d)
    # BatchNorm must stay in eval mode during MC dropout.
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))


def test_checkpoint_roundtrip_and_segmenter(tiny_checkpoint):
    model, cfg, meta = load_checkpoint(tiny_checkpoint)
    assert cfg.encoder == "resnet18" and cfg.image_size == 64 and meta["epoch"] == 1
    seg = PolypSegmenter(tiny_checkpoint, device="cpu")
    rgb = np.random.default_rng(0).integers(0, 255, (90, 120, 3), dtype=np.uint8)
    probs = seg.predict(rgb)
    assert probs.shape == (90, 120) and probs.dtype == np.float32
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert seg.last_inference_ms > 0
    samples = seg.predict_mc(rgb, samples=4)
    assert samples.shape == (4, 90, 120)
    assert samples.std(axis=0).mean() > 0
    # predict_mc must leave the model in plain eval mode afterwards.
    assert not any(m.training for m in seg.model.modules())

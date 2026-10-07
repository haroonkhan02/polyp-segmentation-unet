"""End-to-end pieces on synthetic data: evaluation records, galleries, prediction CLI, viz."""

import json

import numpy as np

from polypseg.evaluate import build_galleries, evaluate_image, spearman, stratify_by_size
from polypseg.model import PolypSegmenter
from polypseg.predict import collect_images, predict_file
from polypseg.viz import blend_heatmap, case_row, heatmap, overlay_comparison, overlay_mask
from tests.conftest import synthetic_case


def test_viz_shapes(case):
    rgb, mask = case
    assert overlay_mask(rgb, mask).shape == rgb.shape
    assert overlay_comparison(rgb, mask, np.roll(mask, 5, axis=1)).shape == rgb.shape
    assert heatmap(np.random.rand(96, 96)).dtype == np.uint8
    assert blend_heatmap(rgb, np.random.rand(96, 96)).shape == rgb.shape
    row = case_row(rgb, mask, mask, np.zeros((96, 96)), "t", size=64)
    assert row.shape == (64, 5 * 64, 3)


def test_evaluate_image_records_and_galleries(tiny_checkpoint, tmp_path):
    seg = PolypSegmenter(tiny_checkpoint, device="cpu")
    cases = []
    for i in range(4):
        rgb, target = synthetic_case(72, radius=6 + 6 * i, seed=i)
        record, arrays = evaluate_image(seg, rgb, target, 0.5, mc_samples=3)
        assert {"dice", "iou", "precision", "recall", "polyp_fraction", "ms", "mc_ms",
                "mc_mean", "uncertainty"} <= set(record)
        assert arrays["pred"].shape == target.shape and arrays["std"].shape == target.shape
        record["stem"] = f"case{i}"
        cases.append({"stem": record["stem"], "rgb": rgb, "target": target,
                      "dice": record["dice"], "pred": arrays["pred"], "std": arrays["std"]})
    buckets = stratify_by_size([{"dice": c["dice"], "iou": 0.5, "recall": 0.5,
                                 "polyp_fraction": c["target"].mean()} for c in cases])
    assert sum(b["num_images"] for b in buckets.values()) == 4
    build_galleries(cases, tmp_path, n=2)
    assert (tmp_path / "gallery_test.jpg").exists() and (tmp_path / "failure_cases.jpg").exists()
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9


def test_predict_cli_writes_outputs(tiny_checkpoint, fake_kvasir, tmp_path):
    seg = PolypSegmenter(tiny_checkpoint, device="cpu")
    images = collect_images(fake_kvasir / "images")
    assert len(images) == 6
    info = predict_file(seg, images[0], tmp_path, 0.5, mc_samples=2)
    assert (tmp_path / "img0_pred.jpg").exists() and (tmp_path / "img0_mask.png").exists()
    assert "uncertainty" in info and json.dumps(info)

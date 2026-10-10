import json

import numpy as np
import torch

from polypseg.augment import augment_pair, color_jitter, random_affine, random_flips
from polypseg.data import (
    PolypDataset,
    list_pairs,
    load_or_create_split,
    make_split,
    preprocess,
    read_mask,
)
from tests.conftest import synthetic_case


def test_list_pairs_and_split_are_deterministic(fake_kvasir):
    stems = list_pairs(fake_kvasir)
    assert stems == [f"img{i}" for i in range(6)]
    a = make_split(stems, seed=0, fractions={"train": 0.5, "val": 0.25, "test": 0.25})
    b = make_split(stems, seed=0, fractions={"train": 0.5, "val": 0.25, "test": 0.25})
    assert a == b
    assert sorted(a["train"] + a["val"] + a["test"]) == stems
    assert not (set(a["train"]) & set(a["test"]))
    assert make_split(stems, seed=1) != a or len(stems) < 3


def test_split_file_is_created_then_reused(fake_kvasir, tmp_path):
    path = tmp_path / "split.json"
    first = load_or_create_split(fake_kvasir, path)
    assert path.exists()
    # Images missing on disk are dropped, the committed order is otherwise kept.
    stored = json.loads(path.read_text())
    stored["train"].append("not_on_disk")
    path.write_text(json.dumps(stored))
    again = load_or_create_split(fake_kvasir, path)
    assert again == first


def test_jpeg_mask_is_thresholded_to_binary(fake_kvasir):
    mask = read_mask(fake_kvasir / "masks" / "img3.jpg")
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 1}
    assert 0.02 < mask.mean() < 0.3


def test_dataset_items(fake_kvasir):
    cfg_aug = {"hflip": 1.0, "vflip": 0.5, "rotate90": 0.5, "affine": 1.0, "max_rotate": 20,
               "scale": [0.9, 1.1], "shift": 0.05, "color": 1.0, "brightness": 0.2,
               "contrast": 0.2, "hue": 5, "saturation": 0.2, "blur": 1.0}
    for aug in (None, cfg_aug):
        ds = PolypDataset(list_pairs(fake_kvasir), fake_kvasir, 48, aug, seed=0)
        x, y = ds[2]
        assert x.shape == (3, 48, 48) and x.dtype == torch.float32
        assert y.shape == (1, 48, 48) and set(torch.unique(y).tolist()) <= {0.0, 1.0}
        assert y.sum() > 0


def test_geometric_augmentation_keeps_image_and_mask_aligned():
    rgb, mask = synthetic_case(96)
    rng = np.random.default_rng(3)
    params = {"hflip": 0.5, "vflip": 0.5, "rotate90": 0.5, "affine": 1.0, "max_rotate": 30,
              "scale": [0.8, 1.2], "shift": 0.1, "color": 0.0, "brightness": 0, "contrast": 0,
              "hue": 0, "saturation": 0, "blur": 0.0}
    for _ in range(10):
        img2, mask2 = augment_pair(rgb, mask, rng, params)
        # The polyp is the only dark-red region, so recover it from the image and compare.
        recovered = (img2[..., 1] < 100).astype(np.uint8)
        inter, union = (recovered & mask2).sum(), (recovered | mask2).sum()
        assert union > 0 and inter / union > 0.9
        assert mask2.shape == mask.shape and set(np.unique(mask2)) <= {0, 1}


def test_flips_are_exact_and_affine_preserves_area_roughly():
    rgb, mask = synthetic_case(96)
    img2, mask2 = random_flips(rgb, mask, np.random.default_rng(0), 1.0, 0.0, 0.0)
    np.testing.assert_array_equal(mask2, mask[:, ::-1])
    np.testing.assert_array_equal(img2, rgb[:, ::-1])
    _, mask3 = random_affine(rgb, mask, np.random.default_rng(0), 0, (1.0, 1.0), 0.0)
    assert abs(mask3.sum() - mask.sum()) / mask.sum() < 0.02


def test_color_jitter_changes_only_pixels_not_shape():
    rgb, _ = synthetic_case(64)
    out = color_jitter(rgb, np.random.default_rng(1), 0.3, 0.3, 10, 0.3)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert not np.array_equal(out, rgb)


def test_preprocess_normalises_with_imagenet_stats():
    rgb = np.full((50, 70, 3), 255, np.uint8)
    x = preprocess(rgb, 32)
    assert x.shape == (1, 3, 32, 32)
    expected = (1.0 - 0.485) / 0.229
    assert abs(x[0, 0, 0, 0].item() - expected) < 1e-5


def test_seed_worker_gives_each_worker_its_own_rng(monkeypatch, fake_kvasir):
    from types import SimpleNamespace

    import polypseg.data as data

    draws = []
    for worker_id in range(2):
        ds = data.PolypDataset(["img0"], fake_kvasir, size=64, augment={"hflip": 0.5}, seed=0)
        info = SimpleNamespace(seed=1234 + worker_id, dataset=ds)
        monkeypatch.setattr(data.torch.utils.data, "get_worker_info", lambda info=info: info)
        data.seed_worker(worker_id)
        draws.append(ds.rng.random(5).tolist())
    assert draws[0] != draws[1]

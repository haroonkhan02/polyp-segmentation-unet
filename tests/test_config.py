import pytest

from polypseg.config import TrainConfig


def test_yaml_overrides_merge_with_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("encoder: resnet18\nepochs: 3\naugment:\n  hflip: 0.0\n")
    cfg = TrainConfig.load(path)
    assert cfg.encoder == "resnet18" and cfg.epochs == 3
    assert cfg.augment["hflip"] == 0.0 and cfg.augment["vflip"] == 0.5  # untouched default
    assert TrainConfig(**cfg.to_dict()) == cfg


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.load(path)

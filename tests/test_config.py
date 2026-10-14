"""Config loading, validation and fingerprints."""

import copy

import pytest
import yaml
from conftest import DEV_CONFIG

from prs.data import PREPROCESS_VERSION
from prs.utils.config import ConfigError, data_config_from_dict, load_config
from prs.utils.fingerprint import data_fingerprint


@pytest.fixture
def raw_data() -> dict:
    return yaml.safe_load(DEV_CONFIG.read_text())["data"]


def test_dev_config_derived_values():
    cfg = load_config(DEV_CONFIG).data
    assert (cfg.history_steps, cfg.history_len, cfg.future_steps) == (20, 21, 60)
    assert cfg.target_type_ids == (2, 3)
    assert cfg.dt == pytest.approx(0.1)


def test_config_rejects_invalid_window(raw_data):
    raw_data["current_index"] = 10  # 2.0 s of history needs t0 >= 20
    with pytest.raises(ConfigError, match="window around t0=10"):
        data_config_from_dict(raw_data)
    raw_data["current_index"] = 31  # 6.0 s of future needs t0 <= 30
    with pytest.raises(ConfigError, match="window around t0=31"):
        data_config_from_dict(raw_data)


def test_config_rejects_fractional_frames(raw_data):
    raw_data["history_seconds"] = 2.05
    with pytest.raises(ConfigError, match="whole number of frames"):
        data_config_from_dict(raw_data)


def test_config_rejects_unknown_and_missing_keys(raw_data):
    with pytest.raises(ConfigError, match="unknown keys"):
        data_config_from_dict({**raw_data, "map_radius": 50.0})
    del raw_data["map_radius_m"]
    with pytest.raises(ConfigError, match="missing keys"):
        data_config_from_dict(raw_data)


def test_config_rejects_wrong_types(raw_data):
    raw_data["max_neighbors"] = "16"
    with pytest.raises(ConfigError, match="expected an integer"):
        data_config_from_dict(raw_data)


def test_config_rejects_overlapping_splits(raw_data):
    raw_data["splits"]["test"]["shards"] = [0]  # same validation shard as val
    with pytest.raises(ConfigError, match="disjoint"):
        data_config_from_dict(raw_data)


def test_config_rejects_test_split_from_training(raw_data):
    raw_data["splits"]["test"] = {"source": "training", "shards": [5]}
    with pytest.raises(ConfigError, match="must use source 'validation'"):
        data_config_from_dict(raw_data)


def test_fingerprint_tracks_content_settings_only(raw_data):
    base = data_fingerprint(data_config_from_dict(raw_data), PREPROCESS_VERSION)
    assert base == data_fingerprint(data_config_from_dict(raw_data), PREPROCESS_VERSION)

    moved = copy.deepcopy(raw_data)
    moved["raw_root"] = "/elsewhere"
    moved["splits"]["train"]["shards"] = [0, 1, 2]
    assert data_fingerprint(data_config_from_dict(moved), PREPROCESS_VERSION) == base

    changed = copy.deepcopy(raw_data)
    changed["map_radius_m"] = 60.0
    assert data_fingerprint(data_config_from_dict(changed), PREPROCESS_VERSION) != base
    cfg = data_config_from_dict(raw_data)
    assert data_fingerprint(cfg, PREPROCESS_VERSION + 1) != base


def test_model_config_references_data_config():
    cfg = load_config(DEV_CONFIG.parent / "baseline_cv.yaml")
    assert cfg.require_model().name == "constant_velocity"
    assert cfg.require_evaluation().miss_threshold_m == 2.0
    assert cfg.data == load_config(DEV_CONFIG).data


def test_config_rejects_data_and_data_config_together(tmp_path):
    path = tmp_path / "both.yaml"
    path.write_text(f"data_config: {DEV_CONFIG}\ndata: {{}}\n")
    with pytest.raises(ConfigError, match="not both"):
        load_config(path)


def test_config_rejects_unknown_model(tmp_path, raw_data):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({"data": raw_data, "model": {"name": "transformer_xl"}}))
    with pytest.raises(ConfigError, match="model.name must be one of"):
        load_config(path)


def test_data_only_config_has_no_model():
    with pytest.raises(ConfigError, match="no `model` section"):
        load_config(DEV_CONFIG).require_model()


def test_shard_ranges(raw_data):
    raw_data["splits"]["train"]["shards"] = "0-3,7,10-11"
    assert data_config_from_dict(raw_data).splits["train"].shards == (0, 1, 2, 3, 7, 10, 11)
    for bad in ("5-3", "1-x", "1--2"):
        raw_data["splits"]["train"]["shards"] = bad
        with pytest.raises(ConfigError):
            data_config_from_dict(raw_data)


def test_dev_and_exp_tiers_share_content_settings():
    """Different shards only: same fingerprint, so caches are shared and dev ⊂ exp."""
    dev, exp = load_config(DEV_CONFIG), load_config(DEV_CONFIG.parent / "data_exp.yaml")
    assert data_fingerprint(dev.data, PREPROCESS_VERSION) == data_fingerprint(
        exp.data, PREPROCESS_VERSION
    )
    for split in ("train", "val", "test"):
        assert set(dev.data.splits[split].shards) <= set(exp.data.splits[split].shards)


def test_data_config_override_sets_tier():
    model_config = DEV_CONFIG.parent / "baseline_cv.yaml"
    assert load_config(model_config).tier == "data_dev"
    exp = load_config(model_config, data_config=DEV_CONFIG.parent / "data_exp.yaml")
    assert exp.tier == "data_exp" and len(exp.data.splits["test"].shards) == 30
    assert exp.require_model().name == "constant_velocity"

"""Config loading, validation and fingerprints."""

import copy
import dataclasses

import pytest
import yaml
from conftest import DEV_CONFIG

from prs.data import PREPROCESS_VERSION
from prs.utils.config import (
    MULTIMODAL_TRAINING_PARAMS,
    ConfigError,
    RiskConfig,
    config_from_dict,
    config_to_dict,
    data_config_from_dict,
    load_config,
)
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


def test_lstm_config_loads():
    cfg = load_config(DEV_CONFIG.parent / "lstm.yaml")
    model, training = cfg.require_model(), cfg.require_training()
    assert (model.name, model.hidden_dim, model.num_layers, model.decoder_hidden_dim) == (
        "lstm", 256, 2, 512,
    )  # fmt: skip
    assert (training.batch_size, training.epochs, cfg.device) == (64, 30, "auto")


def test_model_section_needs_exactly_its_hyperparameters(tmp_path, raw_data):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({"data": raw_data, "model": {"name": "lstm", "hidden_dim": 8}}))
    with pytest.raises(ConfigError, match="model.num_layers is required"):
        load_config(path)
    path.write_text(
        yaml.safe_dump({"data": raw_data, "model": {"name": "constant_velocity", "dropout": 0.1}})
    )
    with pytest.raises(ConfigError, match="model.dropout is not used"):
        load_config(path)


def test_lstm_map_config_checks_map_settings(tmp_path, raw_data):
    model = load_config(DEV_CONFIG.parent / "lstm_map.yaml").require_model()
    assert (model.map_point_dim, model.map_attention_heads) == (128, 8)
    path = tmp_path / "model.yaml"
    section = {"name": "lstm_map", "hidden_dim": 256, "num_layers": 2, "dropout": 0.1,
               "decoder_hidden_dim": 512, "map_point_dim": 128,
               "map_attention_heads": 6}  # fmt: skip
    path.write_text(yaml.safe_dump({"data": raw_data, "model": section}))
    with pytest.raises(ConfigError, match="divisible by model.map_attention_heads"):
        load_config(path)
    path.write_text(yaml.safe_dump({"data": raw_data, "model": {**section, "map_point_dim": 127}}))
    with pytest.raises(ConfigError, match="map_point_dim must be even"):
        load_config(path)
    lstm = {"name": "lstm", "hidden_dim": 8, "num_layers": 1, "dropout": 0.0,
            "decoder_hidden_dim": 8, "map_attention_heads": 2}  # fmt: skip
    path.write_text(yaml.safe_dump({"data": raw_data, "model": lstm}))
    with pytest.raises(ConfigError, match="model.map_attention_heads is not used by model 'lstm'"):
        load_config(path)


def test_interaction_configs_differ_only_in_their_switches():
    stems = ("interaction", "interaction_map", "interaction_neighbors", "interaction_history")
    configs = {stem: load_config(DEV_CONFIG.parent / f"{stem}.yaml") for stem in stems}
    switches = {stem: (c.require_model().use_map, c.require_model().use_neighbors)
                for stem, c in configs.items()}  # fmt: skip
    assert switches == {"interaction": (True, True), "interaction_map": (True, False),
                        "interaction_neighbors": (False, True),
                        "interaction_history": (False, False)}  # fmt: skip
    rest = []
    for cfg in configs.values():
        plain = config_to_dict(cfg)
        for key in ("use_map", "use_neighbors", "map_point_dim"):
            plain["model"].pop(key, None)
        rest.append(plain)
    assert all(plain == rest[0] for plain in rest)


def test_interaction_config_checks_its_settings(tmp_path, raw_data):
    path = tmp_path / "model.yaml"
    section = {"name": "interaction", "use_map": True, "use_neighbors": True, "hidden_dim": 256,
               "num_layers": 2, "dropout": 0.1, "decoder_hidden_dim": 512, "map_point_dim": 128,
               "transformer_layers": 3, "transformer_heads": 8,
               "transformer_ff_dim": 1024}  # fmt: skip
    path.write_text(yaml.safe_dump({"data": raw_data, "model": section}))
    load_config(path)
    no_map = {**section, "use_map": False}
    path.write_text(yaml.safe_dump({"data": raw_data, "model": no_map}))
    with pytest.raises(ConfigError, match="model.map_point_dim is not used"):
        load_config(path)
    del no_map["map_point_dim"]
    path.write_text(yaml.safe_dump({"data": raw_data, "model": no_map}))
    load_config(path)
    del section["map_point_dim"]
    path.write_text(yaml.safe_dump({"data": raw_data, "model": section}))
    with pytest.raises(ConfigError, match="model.map_point_dim is required"):
        load_config(path)
    for bad, message in (
        ({"transformer_heads": 6}, "divisible by model.transformer_heads"),
        ({"use_map": "yes"}, "expected true/false"),
        ({"transformer_layers": 0}, "transformer_layers must be positive"),
    ):
        path.write_text(yaml.safe_dump({"data": raw_data, "model": {**no_map, **bad}}))
        with pytest.raises(ConfigError, match=message):
            load_config(path)


def test_multimodal_config_is_the_full_interaction_config_with_modes():
    interaction = config_to_dict(load_config(DEV_CONFIG.parent / "interaction.yaml"))
    multimodal = config_to_dict(load_config(DEV_CONFIG.parent / "multimodal.yaml"))
    assert multimodal["model"].pop("name") == "multimodal" and interaction["model"].pop("name")
    assert multimodal["model"].pop("num_modes") == 6
    assert multimodal["training"].pop("epochs") == 50 and interaction["training"].pop("epochs")
    for key in MULTIMODAL_TRAINING_PARAMS:
        multimodal["training"].pop(key)
    assert multimodal == interaction


def test_multimodal_settings_are_checked(tmp_path, raw_data):
    path = tmp_path / "model.yaml"
    model = {"name": "multimodal", "use_map": False, "use_neighbors": True, "hidden_dim": 256,
             "num_layers": 2, "dropout": 0.1, "decoder_hidden_dim": 512, "transformer_layers": 3,
             "transformer_heads": 8, "transformer_ff_dim": 1024, "num_modes": 6}  # fmt: skip
    training = {"batch_size": 64, "epochs": 30, "learning_rate": 3e-4, "weight_decay": 1e-4,
                "gradient_clip_norm": 5.0, "smooth_l1_beta": 1.0, "num_workers": 4,
                "lambda_cls": 1.0, "wta_selection": "ade", "collapse_min_win_rate": 0.02,
                "collapse_check_after_epochs": 3}  # fmt: skip
    deterministic = {k: v for k, v in training.items() if k not in MULTIMODAL_TRAINING_PARAMS}
    interaction = {k: v for k, v in model.items() if k != "num_modes"} | {"name": "interaction"}

    def load(model_section, training_section):
        sections = {"data": raw_data, "model": model_section, "training": training_section}
        path.write_text(yaml.safe_dump(sections))
        return load_config(path)

    assert load(model, training).require_training().wta_selection == "ade"
    assert load(interaction, deterministic).require_training().lambda_cls is None
    for model_section, training_section, message in (
        (model, deterministic, "training.lambda_cls is required by model 'multimodal'"),
        (interaction, training, "training.lambda_cls is not used by model 'interaction'"),
        ({**interaction, "num_modes": 6}, deterministic, "model.num_modes is not used"),
        ({**model, "num_modes": 1}, training, "num_modes must be at least 2"),
        (model, {**training, "wta_selection": "mse"}, "wta_selection must be one of"),
        (model, {**training, "collapse_min_win_rate": 1.0}, r"must be in \[0, 1\)"),
        (model, {**training, "collapse_check_after_epochs": 0}, "must be positive"),
        (model, {**training, "lambda_cls": -1.0}, "lambda_cls must be >= 0"),
    ):
        with pytest.raises(ConfigError, match=message):
            load(model_section, training_section)


def test_model_configs_share_one_conflict_rule():
    paths = [p for p in sorted(DEV_CONFIG.parent.glob("*.yaml")) if not p.stem.startswith("data_")]
    rules = {load_config(path).require_risk() for path in paths}
    assert rules == {RiskConfig("ego_box", 0.4, 3.0, 4.0, 1.0, 50.0)}  # doc §13, 3 m: §22.6
    with pytest.raises(ConfigError, match="no `risk` section"):
        load_config(DEV_CONFIG).require_risk()


def test_risk_settings_are_checked(tmp_path, raw_data):
    path = tmp_path / "risk.yaml"
    risk = {"distance_mode": "ego_box", "vru_radius_m": 0.4, "conflict_distance_m": 1.0,
            "max_time_to_closest_approach_s": 4.0, "min_ego_speed_mps": 1.0,
            "max_sdc_distance_m": 50.0}  # fmt: skip

    def load(section):
        path.write_text(yaml.safe_dump({"data": raw_data, "risk": section}))
        return load_config(path)

    assert load({**risk, "distance_mode": "point"}).require_risk().distance_mode == "point"
    assert load({**risk, "vru_radius_m": 0}).require_risk().vru_radius_m == 0.0
    for section, message in (
        ({**risk, "distance_mode": "circle"}, "distance_mode must be one of"),
        ({**risk, "vru_radius_m": -0.1}, "vru_radius_m must be >= 0"),
        ({**risk, "min_ego_speed_mps": -1}, "min_ego_speed_mps must be >= 0"),
        ({**risk, "conflict_distance_m": 0}, "conflict_distance_m must be positive"),
        ({**risk, "max_time_to_closest_approach_s": 0}, "must be positive"),
        ({**risk, "max_sdc_distance_m": -5}, "max_sdc_distance_m must be positive"),
        ({k: v for k, v in risk.items() if k != "vru_radius_m"}, "missing keys"),
        ({**risk, "radius": 1}, "unknown keys"),
    ):
        with pytest.raises(ConfigError, match=message):
            load(section)


def test_config_rejects_unknown_device(tmp_path, raw_data):
    path = tmp_path / "device.yaml"
    path.write_text(yaml.safe_dump({"data": raw_data, "device": "tpu"}))
    with pytest.raises(ConfigError, match="device must be one of"):
        load_config(path)


def test_config_to_dict_round_trips():
    cfg = load_config(
        DEV_CONFIG.parent / "lstm.yaml", data_config=DEV_CONFIG.parent / "data_exp.yaml"
    )
    plain = config_to_dict(cfg)
    assert yaml.safe_load(yaml.safe_dump(plain)) == plain  # YAML-safe types only
    assert config_from_dict(plain) == dataclasses.replace(cfg, data_source=None)
    assert plain["data"]["splits"]["train"]["shards"] == list(range(100))

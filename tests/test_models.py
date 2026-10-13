"""The shared model interface and its leakage guard (design doc §14.1, §40)."""

import inspect

import numpy as np

from prs.models import build_model
from prs.models.base import MODEL_INPUT_FIELDS, ForecastModel, model_inputs
from prs.models.constant_velocity import ConstantVelocity
from prs.utils.config import MODEL_NAMES, ModelConfig

ALL_MODELS = (ForecastModel, ConstantVelocity)


def test_model_forward_signature_excludes_future():
    for cls in ALL_MODELS:
        params = [p for p in inspect.signature(cls.forward).parameters if p != "self"]
        assert tuple(params) == MODEL_INPUT_FIELDS, cls.__name__
    assert not any("future" in name or "ego" in name for name in MODEL_INPUT_FIELDS)


def test_model_inputs_drops_everything_else():
    batch = {
        "target_history": np.zeros((2, 21, 9), np.float32),
        "target_history_valid": np.ones((2, 21), bool),
        "target_type": np.array([2, 3]),
        "target_future": np.ones((2, 60, 2), np.float32),
        "ego_future": np.ones((2, 60, 5), np.float32),
        "meta": ["a", "b"],
    }
    inputs = model_inputs(batch)
    assert set(inputs) == set(MODEL_INPUT_FIELDS)
    assert inputs["neighbors"] is None and inputs["map_polylines"] is None
    assert inputs["target_history"].shape == (2, 21, 9)


def test_every_configured_model_builds(data_cfg):
    for name in MODEL_NAMES:
        model = build_model(ModelConfig(name), data_cfg)
        assert isinstance(model, ForecastModel)

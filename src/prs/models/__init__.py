"""Forecasting models. Every model implements `ForecastModel` (design doc §40)."""

from __future__ import annotations

from prs.models.base import ForecastModel
from prs.models.constant_velocity import ConstantVelocity
from prs.utils.config import DataConfig, ModelConfig


def build_model(model_cfg: ModelConfig, data_cfg: DataConfig) -> ForecastModel:
    if model_cfg.name == "constant_velocity":
        return ConstantVelocity(future_steps=data_cfg.future_steps, dt=data_cfg.dt)
    raise ValueError(f"unknown model {model_cfg.name!r}")

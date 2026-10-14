"""Forecasting models. Every model implements `ForecastModel` (design doc §40)."""

from __future__ import annotations

from prs.data.types import F_AGENT
from prs.models.base import ForecastModel
from prs.models.constant_velocity import ConstantVelocity
from prs.models.lstm_forecaster import LSTMForecaster
from prs.utils.config import DataConfig, ModelConfig


def build_model(model_cfg: ModelConfig, data_cfg: DataConfig) -> ForecastModel:
    if model_cfg.name == "constant_velocity":
        return ConstantVelocity(future_steps=data_cfg.future_steps, dt=data_cfg.dt)
    if model_cfg.name == "lstm":
        return LSTMForecaster(
            agent_features=F_AGENT,
            future_steps=data_cfg.future_steps,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            decoder_hidden_dim=model_cfg.decoder_hidden_dim,
        )
    raise ValueError(f"unknown model {model_cfg.name!r}")


def is_learned(model: ForecastModel) -> bool:
    """True if the model has trainable parameters (and so needs a checkpoint to evaluate)."""
    return any(True for _ in model.parameters())

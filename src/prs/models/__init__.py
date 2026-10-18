"""Forecasting models. Every model implements `ForecastModel` (design doc §40)."""

from __future__ import annotations

from prs.data.map_types import NUM_MAP_TYPES
from prs.data.types import F_AGENT, F_MAP, ObjectType
from prs.models.base import ForecastModel
from prs.models.constant_velocity import ConstantVelocity
from prs.models.interaction import InteractionForecaster
from prs.models.lstm_forecaster import LSTMForecaster
from prs.models.lstm_map_forecaster import LSTMMapForecaster
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
    if model_cfg.name == "lstm_map":
        return LSTMMapForecaster(
            agent_features=F_AGENT,
            map_features=F_MAP,
            num_map_types=NUM_MAP_TYPES,
            future_steps=data_cfg.future_steps,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            decoder_hidden_dim=model_cfg.decoder_hidden_dim,
            map_point_dim=model_cfg.map_point_dim,
            map_attention_heads=model_cfg.map_attention_heads,
        )
    if model_cfg.name in ("interaction", "multimodal"):
        return InteractionForecaster(
            agent_features=F_AGENT,
            map_features=F_MAP,
            num_agent_types=len(ObjectType),
            num_map_types=NUM_MAP_TYPES,
            future_steps=data_cfg.future_steps,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            decoder_hidden_dim=model_cfg.decoder_hidden_dim,
            use_map=model_cfg.use_map,
            use_neighbors=model_cfg.use_neighbors,
            transformer_layers=model_cfg.transformer_layers,
            transformer_heads=model_cfg.transformer_heads,
            transformer_ff_dim=model_cfg.transformer_ff_dim,
            map_point_dim=model_cfg.map_point_dim,
            num_modes=model_cfg.num_modes or 1,
        )
    raise ValueError(f"unknown model {model_cfg.name!r}")


def model_label(model_cfg: ModelConfig) -> str:
    """The model's name in tables and figures. Interaction and multimodal variants (one
    architecture, design doc §19) also name the context they use."""
    if model_cfg.name not in ("interaction", "multimodal"):
        return model_cfg.name
    context = [name for name, used in (("map", model_cfg.use_map),
                                       ("neighbors", model_cfg.use_neighbors)) if used]  # fmt: skip
    return f"{model_cfg.name} ({' + '.join(context) or 'history only'})"


def is_learned(model: ForecastModel) -> bool:
    """True if the model has trainable parameters (and so needs a checkpoint to evaluate)."""
    return any(True for _ in model.parameters())

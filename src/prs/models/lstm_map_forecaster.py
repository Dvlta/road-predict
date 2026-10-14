"""Map-aware LSTM forecaster (design doc §18): the §17 LSTM plus the target's map context."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from prs.models.base import ForecastModel
from prs.models.lstm_forecaster import TrajectoryEncoder
from prs.models.map_encoder import MapAttentionPool, PolylineEncoder


class LSTMMapForecaster(ForecastModel):
    """History encoder (as in §17) + polyline encoder + attention pooling with the target
    embedding as the query. The decoder reads the concatenated [target embedding, map context].

    The map-disabled model of the ablation is `LSTMForecaster` (model `lstm`): the same
    history encoder and decoder layout without the map branch.
    """

    num_modes = 1
    input_fields = (
        "target_history", "target_history_valid", "target_type",
        "map_polylines", "map_valid", "map_types",
    )  # fmt: skip

    def __init__(
        self,
        agent_features: int,
        map_features: int,
        num_map_types: int,
        future_steps: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        decoder_hidden_dim: int,
        map_point_dim: int,
        map_attention_heads: int,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = TrajectoryEncoder(agent_features, hidden_dim, num_layers, dropout)
        self.map_encoder = PolylineEncoder(map_features, map_point_dim, hidden_dim, num_map_types)
        self.map_pool = MapAttentionPool(hidden_dim, map_attention_heads)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, future_steps * 2),
        )

    def forward(
        self,
        target_history: Tensor,
        target_history_valid: Tensor,
        target_type: Tensor,
        neighbors: Tensor | None = None,
        neighbor_valid: Tensor | None = None,
        neighbor_types: Tensor | None = None,
        neighbor_is_sdc: Tensor | None = None,
        map_polylines: Tensor | None = None,
        map_valid: Tensor | None = None,
        map_types: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if map_polylines is None or map_valid is None or map_types is None:
            raise ValueError("lstm_map needs map_polylines, map_valid and map_types")
        embedding = self.encoder(target_history, target_history_valid)
        chunks, chunk_valid = self.map_encoder(map_polylines, map_valid, map_types)
        context = self.map_pool(embedding, chunks, chunk_valid)
        decoded = self.decoder(torch.cat([embedding, context], dim=-1))
        trajectories = decoded.view(-1, 1, self.future_steps, 2)
        return {
            "trajectories": trajectories,
            "logits": torch.zeros(len(trajectories), 1, device=trajectories.device),
        }

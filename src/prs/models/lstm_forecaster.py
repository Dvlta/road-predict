"""Single-agent LSTM forecaster (design doc §17): the target's own history, no map or neighbors."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from prs.models.base import ForecastModel


class TrajectoryEncoder(nn.Module):
    """Masked per-frame projection, then a multi-layer LSTM over the history frames.

    Returns the top layer's hidden state at the last frame (t0). Frames are zeroed by their
    mask before use, so padded values never reach the output; the `valid_flag` feature tells
    the LSTM which frames are padding. Gaps can be leading or internal, so the sequence is not
    packed (design doc §17). Reused unchanged as the agent encoder in §19.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, history: Tensor, valid: Tensor) -> Tensor:
        """history [B, H, F], valid [B, H] -> [B, hidden_dim]."""
        features = history * valid[..., None].to(history.dtype)
        _, (hidden, _) = self.lstm(self.projection(features))
        return hidden[-1]


class LSTMForecaster(ForecastModel):
    """History encoder + MLP decoder to T future positions in the target frame."""

    num_modes = 1
    input_fields = ("target_history", "target_history_valid", "target_type")

    def __init__(
        self,
        agent_features: int,
        future_steps: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        decoder_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = TrajectoryEncoder(agent_features, hidden_dim, num_layers, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
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
        embedding = self.encoder(target_history, target_history_valid)
        trajectories = self.decoder(embedding).view(-1, 1, self.future_steps, 2)
        return {
            "trajectories": trajectories,
            "logits": torch.zeros(len(trajectories), 1, device=trajectories.device),
        }

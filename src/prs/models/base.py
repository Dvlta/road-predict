"""The interface every forecaster implements (design doc §40)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

# The only batch fields a model may receive: history and static map, never anything derived
# from the future (design doc §14.1). Every model's forward() takes exactly these arguments.
MODEL_INPUT_FIELDS = (
    "target_history",
    "target_history_valid",
    "target_type",
    "neighbors",
    "neighbor_valid",
    "neighbor_types",
    "neighbor_is_sdc",
    "map_polylines",
    "map_valid",
    "map_types",
)


class ForecastModel(nn.Module):
    """Predicts K futures for a batch of targets, in each target's local frame.

    forward() returns {"trajectories": Tensor[B, K, T, 2], "logits": Tensor[B, K]}.
    Neighbor and map arguments are None when the batch has no context (before Stage 3).
    """

    num_modes: int = 1
    # Fields this model actually reads; the data loader loads only these (plus labels).
    input_fields: tuple[str, ...] = MODEL_INPUT_FIELDS

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
        raise NotImplementedError


def model_inputs(batch: Mapping[str, Any], device: torch.device | str = "cpu") -> dict[str, Any]:
    """Select the model's arguments from a batch. The only path from data to a model."""
    return {
        name: None if batch.get(name) is None else torch.as_tensor(batch[name]).to(device)
        for name in MODEL_INPUT_FIELDS
    }

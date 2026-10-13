"""Constant-velocity baseline (design doc §16). No learned parameters."""

from __future__ import annotations

import torch
from torch import Tensor

from prs.models.base import ForecastModel


class ConstantVelocity(ForecastModel):
    """p(t0 + k dt) = p(t0) + v(t0) * k * dt for k = 1..T, in the target frame."""

    num_modes = 1
    input_fields = ("target_history", "target_history_valid", "target_type")

    def __init__(self, future_steps: int, dt: float) -> None:
        super().__init__()
        offsets = torch.arange(1, future_steps + 1, dtype=torch.float32) * dt
        self.register_buffer("offsets", offsets, persistent=False)  # seconds after t0

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
        if not bool(target_history_valid[:, -1].all()):
            raise ValueError("every target must be valid at t0 (last history frame)")
        current = target_history[:, -1]  # [B, F_AGENT]: x, y, vx, vy, ...
        position, velocity = current[:, None, 0:2], current[:, None, 2:4]
        trajectories = position + velocity * self.offsets[None, :, None]  # [B, T, 2]
        return {
            "trajectories": trajectories[:, None],
            "logits": torch.zeros(len(current), 1, device=current.device),
        }

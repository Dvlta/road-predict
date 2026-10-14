"""Deterministic trajectory loss (design doc §17)."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional


def masked_smooth_l1(prediction: Tensor, target: Tensor, valid: Tensor, beta: float) -> Tensor:
    """Smooth L1 between positions, summed over x and y and averaged over the batch's valid
    future frames only. prediction, target [B, T, 2]; valid [B, T]."""
    valid = valid.to(prediction.dtype)
    count = valid.sum()
    if count.item() == 0:
        raise ValueError("no valid future frames in the batch")
    per_frame = functional.smooth_l1_loss(prediction, target, reduction="none", beta=beta).sum(-1)
    return (per_frame * valid).sum() / count

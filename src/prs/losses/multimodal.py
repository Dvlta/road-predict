"""Winner-takes-all loss for K predicted futures (design doc §20)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from prs.losses.trajectory import masked_smooth_l1


def select_winners(trajectories: Tensor, future: Tensor, valid: Tensor, selection: str) -> Tensor:
    """The mode closest to the ground truth, per sample: [B]. `selection` is "ade" (mean error
    over the valid frames) or "fde" (error at the final frame, always valid, §5.2).
    trajectories [B, K, T, 2]; future [B, T, 2]; valid [B, T]."""
    errors = (trajectories - future[:, None]).norm(dim=-1)  # [B, K, T]
    if selection == "ade":
        frames = valid[:, None].to(errors.dtype)
        score = (errors * frames).sum(-1) / frames.sum(-1)
    elif selection == "fde":
        score = errors[..., -1]
    else:
        raise ValueError(f"unknown selection {selection!r}")
    return score.argmin(dim=1)


def winner_takes_all(
    trajectories: Tensor,
    logits: Tensor,
    future: Tensor,
    valid: Tensor,
    beta: float,
    lambda_cls: float,
    selection: str,
) -> tuple[Tensor, Tensor]:
    """SmoothL1 on the winning mode only, plus `lambda_cls` × cross-entropy of the logits
    against the winner. Only the winner is pulled toward the truth, so the other modes stay
    free to cover other futures. With one mode this is exactly the deterministic loss (the
    cross-entropy of a single logit is zero). Returns the loss and the winners [B]."""
    with torch.no_grad():
        winners = select_winners(trajectories, future, valid, selection)
    chosen = trajectories[torch.arange(len(winners), device=winners.device), winners]
    loss = masked_smooth_l1(chosen, future, valid, beta)
    return loss + lambda_cls * functional.cross_entropy(logits, winners), winners

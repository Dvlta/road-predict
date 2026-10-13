"""Constant-velocity baseline (design doc §16, §27)."""

import numpy as np
import pytest
import torch
from conftest import CYCLIST, PED

from prs.data.collate import collate_samples
from prs.data.targets import build_target_sample
from prs.data.windows import make_window
from prs.metrics.breakdowns import per_sample_metrics
from prs.models.base import model_inputs
from prs.models.constant_velocity import ConstantVelocity

T, DT = 60, 0.1


def _history(position, velocity, batch=1):
    history = torch.zeros(batch, 21, 9)
    history[:, -1, 0:2] = torch.tensor(position)
    history[:, -1, 2:4] = torch.tensor(velocity)
    history[:, -1, -1] = 1.0
    return history, torch.ones(batch, 21, dtype=torch.bool), torch.full((batch,), 2)


def test_constant_velocity_stationary():
    history, valid, types = _history([0.0, 0.0], [0.0, 0.0])
    out = ConstantVelocity(T, DT)(history, valid, types)
    assert out["trajectories"].shape == (1, 1, T, 2) and out["logits"].shape == (1, 1)
    assert torch.count_nonzero(out["trajectories"]) == 0


def test_constant_velocity_linear_motion():
    history, valid, types = _history([0.5, -1.0], [1.4, 0.3])
    trajectory = ConstantVelocity(T, DT)(history, valid, types)["trajectories"][0, 0]
    seconds = torch.arange(1, T + 1) * DT
    expected = torch.tensor([0.5, -1.0]) + seconds[:, None] * torch.tensor([1.4, 0.3])
    torch.testing.assert_close(trajectory, expected)
    # First step is one frame (0.1 s) ahead, last step is 6 s ahead.
    torch.testing.assert_close(trajectory[0], torch.tensor([0.64, -0.97]))
    torch.testing.assert_close(trajectory[-1], torch.tensor([8.9, 0.8]))


def test_constant_velocity_requires_valid_t0():
    history, valid, types = _history([0.0, 0.0], [1.0, 0.0])
    valid[:, -1] = False
    with pytest.raises(ValueError, match="valid at t0"):
        ConstantVelocity(T, DT)(history, valid, types)


def test_constant_velocity_is_exact_on_constant_motion(scenario, data_cfg):
    """The synthetic targets move at constant velocity, so CV must reproduce them exactly,
    including across the cyclist's invalid future frame."""
    window = make_window(data_cfg)
    samples = [build_target_sample(scenario, i, data_cfg, window) for i in (PED, CYCLIST)]
    batch = collate_samples(samples)
    out = ConstantVelocity(data_cfg.future_steps, data_cfg.dt)(**model_inputs(batch))
    metrics = per_sample_metrics(
        out["trajectories"].numpy(), out["logits"].numpy(),
        batch["target_future"], batch["target_future_valid"],
    )  # fmt: skip
    np.testing.assert_allclose(metrics["ade"], 0.0, atol=1e-4)
    np.testing.assert_allclose(metrics["fde"], 0.0, atol=1e-4)

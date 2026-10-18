"""Trajectory losses: masked SmoothL1 (design doc §17) and winner-takes-all (§20)."""

import math

import numpy as np
import pytest
import torch

from prs.losses.multimodal import select_winners, winner_takes_all
from prs.losses.trajectory import masked_smooth_l1
from prs.metrics.multimodal import winners


def test_smooth_l1_known_value():
    prediction = torch.tensor([[[0.5, 2.0]]])  # errors 0.5 (quadratic) and 2.0 (linear)
    loss = masked_smooth_l1(prediction, torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=bool), 1.0)
    assert loss.item() == pytest.approx(0.5 * 0.5**2 + (2.0 - 0.5))


def test_invalid_frames_do_not_count():
    target = torch.zeros(2, 3, 2)
    prediction = torch.zeros(2, 3, 2)
    prediction[0, 1] = 1e6  # an invalid frame with a huge error
    prediction[1, 2] = 0.5
    valid = torch.ones(2, 3, dtype=bool)
    valid[0, 1] = False
    loss = masked_smooth_l1(prediction, target, valid, beta=1.0)
    assert loss.item() == pytest.approx(2 * 0.125 / 5)  # one valid frame errs; 5 valid frames


def test_no_valid_frames_fails():
    with pytest.raises(ValueError, match="no valid future frames"):
        masked_smooth_l1(
            torch.zeros(1, 3, 2), torch.zeros(1, 3, 2), torch.zeros(1, 3, dtype=bool), 1.0
        )


def _two_modes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mode 0 tracks the truth closely but ends 3 m off; mode 1 is 2 m off throughout but
    ends exactly on it. So ADE picks mode 0 and FDE picks mode 1."""
    future = torch.zeros(1, 4, 2)
    modes = torch.zeros(1, 2, 4, 2)
    modes[0, 0, -1, 0] = 3.0
    modes[0, 1, :-1, 1] = 2.0
    return modes, future, torch.ones(1, 4, dtype=torch.bool)


def test_wta_selection_by_ade_or_fde():
    modes, future, valid = _two_modes()
    assert select_winners(modes, future, valid, "ade").tolist() == [0]
    assert select_winners(modes, future, valid, "fde").tolist() == [1]
    valid[0, -1] = False  # ADE ignores invalid frames: now mode 0 is exact
    assert select_winners(modes, future, valid, "ade").tolist() == [0]
    valid[0] = torch.tensor([False, False, False, True])  # only the end: mode 1 is exact
    modes[0, 1, 0, 0] = 1e6  # and its huge error in an invalid frame does not count
    assert select_winners(modes, future, valid, "ade").tolist() == [1]


def test_wta_with_one_mode_is_the_deterministic_loss():
    torch.manual_seed(0)
    prediction, future = torch.randn(4, 1, 6, 2), torch.randn(4, 6, 2)
    valid = torch.rand(4, 6) > 0.2
    valid[:, -1] = True
    loss, winners = winner_takes_all(prediction, torch.randn(4, 1), future, valid, 1.0, 1.0, "ade")
    assert winners.tolist() == [0, 0, 0, 0]
    assert loss.item() == pytest.approx(
        masked_smooth_l1(prediction[:, 0], future, valid, 1.0).item()
    )


def test_wta_trains_only_the_winner():
    modes, future, valid = _two_modes()
    modes.requires_grad_(True)
    logits = torch.zeros(1, 2, requires_grad=True)
    loss, winners = winner_takes_all(modes, logits, future, valid, 1.0, 0.5, "fde")
    loss.backward()
    assert winners.tolist() == [1]
    assert modes.grad[0, 0].abs().sum() == 0 and modes.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 1] < 0 < logits.grad[0, 0]  # raises the winner's probability
    expected = masked_smooth_l1(modes[:, 1], future, valid, 1.0) + 0.5 * math.log(2)
    assert loss.item() == pytest.approx(expected.item())


def test_wta_winners_match_the_metric_winners():
    rng = np.random.default_rng(0)
    modes, future = rng.normal(size=(50, 4, 8, 2)), rng.normal(size=(50, 8, 2))
    valid = rng.random((50, 8)) > 0.3
    valid[:, -1] = True
    for selection in ("ade", "fde"):
        expected = winners(modes, future, valid, selection)
        actual = select_winners(
            torch.from_numpy(modes), torch.from_numpy(future), torch.from_numpy(valid), selection
        )
        np.testing.assert_array_equal(actual.numpy(), expected)

"""Masked trajectory loss (design doc §17)."""

import pytest
import torch

from prs.losses.trajectory import masked_smooth_l1


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

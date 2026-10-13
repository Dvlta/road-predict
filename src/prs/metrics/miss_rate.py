"""Miss definition (design doc §21). A project metric, not the official Waymo one."""

from __future__ import annotations

import numpy as np


def misses(min_fde: np.ndarray, threshold_m: float) -> np.ndarray:
    """True where even the best mode ends more than `threshold_m` from the ground truth."""
    return np.asarray(min_fde) > threshold_m

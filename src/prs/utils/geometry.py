"""Low-level 2D geometry helpers. All arithmetic is float64."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles in radians to [-pi, pi)."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def rotate(vectors: np.ndarray, theta: float) -> np.ndarray:
    """Rotate 2D vectors of shape [..., 2] counter-clockwise by `theta` radians."""
    v = np.asarray(vectors, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(v)
    out[..., 0] = c * v[..., 0] - s * v[..., 1]
    out[..., 1] = s * v[..., 0] + c * v[..., 1]
    return out

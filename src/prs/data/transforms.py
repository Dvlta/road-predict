"""Target-centric frame transforms (design doc §6).

All arithmetic is float64. Callers cast to float32 only after transforming into the local
frame, because float32 cannot hold WOMD's global coordinates (tens of km) to 1e-4 m (§6.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prs.utils.geometry import rotate, wrap_angle


@dataclass(frozen=True, eq=False)
class LocalFrame:
    """Frame whose origin is `origin` (global) and whose +x axis points along `theta`."""

    origin: np.ndarray  # float64[2]
    theta: float  # radians, global heading of the local +x axis

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=np.float64)
        if origin.shape != (2,):
            raise ValueError(f"origin must have shape (2,), got {origin.shape}")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "theta", float(self.theta))

    def points_to_local(self, points: np.ndarray) -> np.ndarray:
        return rotate(np.asarray(points, dtype=np.float64) - self.origin, -self.theta)

    def points_to_global(self, points: np.ndarray) -> np.ndarray:
        return rotate(points, self.theta) + self.origin

    def vectors_to_local(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate only: vectors (velocities, directions) have no position."""
        return rotate(vectors, -self.theta)

    def vectors_to_global(self, vectors: np.ndarray) -> np.ndarray:
        return rotate(vectors, self.theta)

    def headings_to_local(self, headings: np.ndarray | float) -> np.ndarray:
        return wrap_angle(np.asarray(headings, dtype=np.float64) - self.theta)

    def headings_to_global(self, headings: np.ndarray | float) -> np.ndarray:
        return wrap_angle(np.asarray(headings, dtype=np.float64) + self.theta)

"""Distances between VRU positions and the ego vehicle (design doc §22.3). All arithmetic is
float64."""

from __future__ import annotations

import numpy as np


def closest_point_on_box(
    points: np.ndarray,
    centers: np.ndarray,
    headings: np.ndarray,
    length: np.ndarray | float,
    width: np.ndarray | float,
) -> np.ndarray:
    """The point of an oriented rectangle nearest to each point (the point itself when inside):
    [..., 2].

    `points` and `centers` are [..., 2] and `headings` is [...] (radians, the direction of the
    rectangle's length); `length` and `width` broadcast against `headings`.
    """
    centers = np.asarray(centers, dtype=np.float64)
    offset = np.asarray(points, dtype=np.float64) - centers
    headings = np.asarray(headings, dtype=np.float64)
    half_length = 0.5 * np.asarray(length, dtype=np.float64)
    half_width = 0.5 * np.asarray(width, dtype=np.float64)
    c, s = np.cos(headings), np.sin(headings)
    along = np.clip(c * offset[..., 0] + s * offset[..., 1], -half_length, half_length)
    across = np.clip(-s * offset[..., 0] + c * offset[..., 1], -half_width, half_width)
    return centers + np.stack([c * along - s * across, s * along + c * across], axis=-1)


def point_to_box_distance(
    points: np.ndarray,
    centers: np.ndarray,
    headings: np.ndarray,
    length: np.ndarray | float,
    width: np.ndarray | float,
) -> np.ndarray:
    """Distance from each point to an oriented rectangle, 0 inside or on its edge: [...].
    Arguments as for `closest_point_on_box`."""
    nearest = closest_point_on_box(points, centers, headings, length, width)
    return point_distance(points, nearest)


def point_distance(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Center-to-center distance: [..., 2] -> [...]."""
    offset = np.asarray(points, dtype=np.float64) - np.asarray(centers, dtype=np.float64)
    return np.hypot(offset[..., 0], offset[..., 1])


def boxes_overlap(
    centers_a: np.ndarray,
    headings_a: np.ndarray,
    size_a: np.ndarray,
    centers_b: np.ndarray,
    headings_b: np.ndarray,
    size_b: np.ndarray,
) -> np.ndarray:
    """Whether two oriented rectangles overlap (touching counts), by the separating-axis test:
    [...]. Centers are [..., 2], headings [...] and sizes [..., 2] (length, width); all
    broadcast against each other."""
    offset = np.asarray(centers_b, dtype=np.float64) - np.asarray(centers_a, dtype=np.float64)
    boxes = [_box_axes(headings_a, size_a), _box_axes(headings_b, size_b)]
    overlap = np.ones(offset.shape[:-1], dtype=bool)
    for axis in (axis for along, across, _ in boxes for axis in (along, across)):
        reach = sum(
            half[..., 0] * np.abs((axis * along).sum(-1))
            + half[..., 1] * np.abs((axis * across).sum(-1))
            for along, across, half in boxes
        )
        overlap = overlap & (np.abs((axis * offset).sum(-1)) <= reach)
    return overlap


def _box_axes(headings: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A rectangle's unit axes along and across its heading ([..., 2] each), and its half
    length and width [..., 2]."""
    c, s = np.cos(headings), np.sin(headings)
    half = 0.5 * np.asarray(size, dtype=np.float64)
    return np.stack([c, s], axis=-1), np.stack([-s, c], axis=-1), half

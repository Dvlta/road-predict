"""Map extraction (design doc §9).

Per scene (once): close polygons, resample every feature along arc length, split into chunks
of at most P points, all in global coordinates. Resampling commutes with the rigid
target-frame transform, so it does not need to be repeated per target.

Per target: transform the chunks into the target frame, keep those with a point inside the
radius, take the nearest M, and encode (x, y, dx, dy) per point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prs.data.map_types import MapType, map_type
from prs.data.transforms import LocalFrame
from prs.data.types import F_MAP, RawScenario
from prs.utils.config import DataConfig


@dataclass(frozen=True, eq=False)
class SceneMap:
    """Every map chunk of one scene, global coordinates, zero-padded to P points."""

    points: np.ndarray  # float64[C, P, 2]
    valid: np.ndarray  # bool[C, P]; valid points always form a prefix
    types: np.ndarray  # int64[C], MapType


def resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """Evenly spaced points at most `spacing_m` apart along the polyline, endpoints included."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.copy()
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.concatenate([[True], steps > 1e-9])]  # drop repeated points
    if len(points) < 2:
        return points[:1].copy()
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    samples = np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / spacing_m)) + 1)
    return np.stack(
        [np.interp(samples, arc, points[:, 0]), np.interp(samples, arc, points[:, 1])], 1
    )


def split_polyline(points: np.ndarray, max_points: int) -> list[np.ndarray]:
    """Chunks of at most `max_points` points; consecutive chunks share one endpoint."""
    if len(points) <= max_points:
        return [points]
    step = max_points - 1
    return [points[start : start + max_points] for start in range(0, len(points) - 1, step)]


def prepare_scene_map(scenario: RawScenario, cfg: DataConfig) -> SceneMap:
    P = cfg.max_points_per_polyline
    chunks: list[np.ndarray] = []
    types: list[int] = []
    for feature in scenario.map_features:
        if len(feature.points) == 0:
            continue
        points = feature.points
        if feature.is_polygon and len(points) >= 2:
            points = np.vstack([points, points[:1]])  # polygons are closed (design doc §9)
        chunk_type = int(map_type(feature.kind, feature.subtype))
        for chunk in split_polyline(resample_polyline(points, cfg.map_resample_spacing_m), P):
            chunks.append(chunk)
            types.append(chunk_type)
    points_arr = np.zeros((len(chunks), P, 2))
    valid = np.zeros((len(chunks), P), dtype=bool)
    for c, chunk in enumerate(chunks):
        points_arr[c, : len(chunk)] = chunk
        valid[c, : len(chunk)] = True
    return SceneMap(points_arr, valid, np.array(types, dtype=np.int64))


def extract_map(
    scene_map: SceneMap, frame: LocalFrame, cfg: DataConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """map_polylines [M, P, F_MAP] float32, map_valid [M, P], map_types [M], and the number of
    chunks inside the radius before the cap."""
    M, P = cfg.max_map_polylines, cfg.max_points_per_polyline
    local = frame.points_to_local(scene_map.points)  # float64; padded points are masked below
    distance = np.where(scene_map.valid, np.linalg.norm(local, axis=-1), np.inf)
    nearest = distance.min(axis=1)
    candidates = np.flatnonzero(nearest <= cfg.map_radius_m)
    chosen = candidates[np.argsort(nearest[candidates], kind="stable")][:M]

    points, valid = local[chosen], scene_map.valid[chosen]
    direction = np.zeros_like(points)
    direction[:, :-1] = np.where(valid[:, 1:, None], points[:, 1:] - points[:, :-1], 0.0)
    counts = valid.sum(axis=1)
    rows = np.flatnonzero(counts >= 2)  # the last point repeats its predecessor's direction
    direction[rows, counts[rows] - 1] = direction[rows, counts[rows] - 2]
    features = np.concatenate([points, direction], axis=-1)
    features[~valid] = 0.0

    map_polylines = np.zeros((M, P, F_MAP), dtype=np.float32)
    map_valid = np.zeros((M, P), dtype=bool)
    map_types = np.full(M, int(MapType.PAD), dtype=np.int64)
    map_polylines[: len(chosen)] = features
    map_valid[: len(chosen)] = valid
    map_types[: len(chosen)] = scene_map.types[chosen]
    return map_polylines, map_valid, map_types, len(candidates)

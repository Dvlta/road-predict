"""Core data types: raw scenarios from the reader and processed samples (design doc §7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from prs.data.transforms import LocalFrame


class ObjectType(IntEnum):
    """WOMD `Track.ObjectType` values."""

    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


OBJECT_TYPE_BY_NAME = {t.name.lower(): t for t in ObjectType if t is not ObjectType.UNSET}

# Raw map feature kinds, named after the `MapFeature.feature_data` oneof fields (§4.3).
POLYLINE_KINDS = ("lane", "road_line", "road_edge")
POLYGON_KINDS = ("crosswalk", "speed_bump", "driveway")
POINT_KINDS = ("stop_sign",)
MAP_KINDS = POLYLINE_KINDS + POLYGON_KINDS + POINT_KINDS

AGENT_FEATURES = (
    "x", "y", "vx", "vy", "heading_sin", "heading_cos", "length", "width", "valid_flag",
)  # fmt: skip
EGO_FUTURE_FEATURES = ("x", "y", "heading", "vx", "vy")
F_AGENT = len(AGENT_FEATURES)
F_EGO = len(EGO_FUTURE_FEATURES)
F_MAP = 4  # x, y, dx, dy


@dataclass(frozen=True, eq=False)
class MapFeature:
    id: int
    kind: str  # one of MAP_KINDS
    subtype: int  # lane / road_line / road_edge type enum value; 0 for other kinds
    points: np.ndarray  # float64[P, 2], global; polygons are not closed here

    @property
    def is_polygon(self) -> bool:
        return self.kind in POLYGON_KINDS


@dataclass(eq=False)
class RawScenario:
    """One WOMD scene as plain NumPy arrays. A = tracks, F = frames."""

    scenario_id: str
    timestamps: np.ndarray  # float64[F]
    current_time_index: int  # official "now"; metadata only (§5.1)
    sdc_index: int  # index into the track axis
    track_ids: np.ndarray  # int64[A]
    track_types: np.ndarray  # int64[A], ObjectType values
    xy: np.ndarray  # float64[A, F, 2]
    heading: np.ndarray  # float64[A, F]
    velocity: np.ndarray  # float64[A, F, 2]
    size: np.ndarray  # float64[A, F, 2], (length, width)
    valid: np.ndarray  # bool[A, F]
    tracks_to_predict: np.ndarray  # int64[R], track indices
    objects_of_interest: np.ndarray  # int64[I], track ids
    map_features: list[MapFeature]
    unknown_field_numbers: tuple[int, ...] = ()  # Scenario fields absent from the vendored proto

    @property
    def num_tracks(self) -> int:
        return int(self.track_ids.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.timestamps.shape[0])

    def track_index(self, track_id: int) -> int:
        matches = np.flatnonzero(self.track_ids == track_id)
        if matches.size != 1:
            raise KeyError(f"track id {track_id} not found in scenario {self.scenario_id}")
        return int(matches[0])


@dataclass(frozen=True)
class SampleMeta:
    """Per-sample metadata (design doc §5.3). Never a model input."""

    shard: str
    scenario_id: str
    target_track_id: int
    target_type: int
    t0: int
    current_speed_mps: float
    future_path_length_m: float  # derived from the future: analysis only
    is_tracks_to_predict: bool  # official selection is biased by future behavior
    sdc_distance_m: float  # NaN if the SDC is invalid at t0
    frame_heading_source_used: str  # "box" | "velocity"
    # Candidates inside the radius before the caps (cap-hit logging, crowded-vs-sparse
    # breakdowns). -1 for samples built without context.
    neighbors_within_radius: int = -1
    map_chunks_within_radius: int = -1


@dataclass(eq=False)
class ProcessedSample:
    """Target-centric sample (design doc §7).

    Invalid frames hold all-zero features and a False mask; padded neighbor slots have type 0
    and padded map chunks type MapType.PAD (0). Neighbor and map context is None only for
    target-only samples (`build_target_sample`).
    """

    meta: SampleMeta
    target_type: int
    target_history: np.ndarray  # float32[H, F_AGENT]
    target_history_valid: np.ndarray  # bool[H]
    target_future: np.ndarray  # float32[T, 2]
    target_future_valid: np.ndarray  # bool[T]
    ego_history: np.ndarray  # float32[H, F_AGENT]
    ego_history_valid: np.ndarray  # bool[H]
    ego_future: np.ndarray  # float32[T, F_EGO]; risk module only, never a model input
    ego_future_valid: np.ndarray  # bool[T]
    ego_size: np.ndarray  # float32[2], (length, width)
    origin_global: np.ndarray  # float64[2]
    heading_global: float
    neighbors: np.ndarray | None = None  # float32[N, H, F_AGENT]
    neighbor_valid: np.ndarray | None = None  # bool[N, H]
    neighbor_types: np.ndarray | None = None  # int64[N]
    neighbor_is_sdc: np.ndarray | None = None  # bool[N]
    map_polylines: np.ndarray | None = None  # float32[M, P, F_MAP]
    map_valid: np.ndarray | None = None  # bool[M, P]
    map_types: np.ndarray | None = None  # int64[M]

    @property
    def frame(self) -> LocalFrame:
        return LocalFrame(self.origin_global, self.heading_global)

    def validate(self) -> None:
        """Fail loudly on any schema violation (design doc §7, §31)."""
        H = self.target_history.shape[0]
        T = self.target_future.shape[0]
        _check(self, "target_history", (H, F_AGENT), np.float32)
        _check(self, "target_history_valid", (H,), np.bool_)
        _check(self, "target_future", (T, 2), np.float32)
        _check(self, "target_future_valid", (T,), np.bool_)
        _check(self, "ego_history", (H, F_AGENT), np.float32)
        _check(self, "ego_history_valid", (H,), np.bool_)
        _check(self, "ego_future", (T, F_EGO), np.float32)
        _check(self, "ego_future_valid", (T,), np.bool_)
        _check(self, "ego_size", (2,), np.float32)
        _check(self, "origin_global", (2,), np.float64)
        for features, valid in (
            ("target_history", "target_history_valid"),
            ("ego_history", "ego_history_valid"),
        ):
            mask = getattr(self, valid)
            if not np.array_equal(getattr(self, features)[:, -1], mask.astype(np.float32)):
                raise ValueError(f"{features}: valid_flag column disagrees with {valid}")
        for features, valid in (
            ("target_history", "target_history_valid"),
            ("target_future", "target_future_valid"),
            ("ego_history", "ego_history_valid"),
            ("ego_future", "ego_future_valid"),
        ):
            if np.any(getattr(self, features)[~getattr(self, valid)] != 0.0):
                raise ValueError(f"{features}: padded (invalid) entries must be zero")
        if not self.target_history_valid[-1]:
            raise ValueError("target must be valid at t0 (last history frame)")
        if not self.target_future_valid.any():
            raise ValueError("sample has no valid future points")

        neighbor_fields = ("neighbors", "neighbor_valid", "neighbor_types", "neighbor_is_sdc")
        if _context_present(self, neighbor_fields):
            N = self.neighbors.shape[0]
            _check(self, "neighbors", (N, H, F_AGENT), np.float32)
            _check(self, "neighbor_valid", (N, H), np.bool_)
            _check(self, "neighbor_types", (N,), np.int64)
            _check(self, "neighbor_is_sdc", (N,), np.bool_)
            if not np.array_equal(self.neighbors[..., -1], self.neighbor_valid.astype(np.float32)):
                raise ValueError("neighbors: valid_flag column disagrees with neighbor_valid")
            if np.any(self.neighbors[~self.neighbor_valid] != 0.0):
                raise ValueError("neighbors: padded (invalid) entries must be zero")
            empty = ~self.neighbor_valid.any(axis=1)
            if np.any(self.neighbor_types[empty] != 0) or self.neighbor_is_sdc[empty].any():
                raise ValueError("padded neighbor slots must have type 0 and is_sdc False")
        map_fields = ("map_polylines", "map_valid", "map_types")
        if _context_present(self, map_fields):
            M, P = self.map_polylines.shape[:2]
            _check(self, "map_polylines", (M, P, F_MAP), np.float32)
            _check(self, "map_valid", (M, P), np.bool_)
            _check(self, "map_types", (M,), np.int64)
            if np.any(self.map_polylines[~self.map_valid] != 0.0):
                raise ValueError("map_polylines: padded (invalid) entries must be zero")
            empty = ~self.map_valid.any(axis=1)
            if np.any(self.map_types[empty] != 0) or np.any(self.map_types[~empty] == 0):
                raise ValueError("map_types must be 0 (PAD) exactly for empty chunks")


def _check(sample: ProcessedSample, name: str, shape: tuple[int, ...], dtype: type) -> None:
    value = getattr(sample, name)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(value).__name__}")
    if value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {np.dtype(dtype)}, got {value.dtype}")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{name}: contains NaN or inf")


def _context_present(sample: ProcessedSample, names: tuple[str, ...]) -> bool:
    present = [getattr(sample, name) is not None for name in names]
    if any(present) and not all(present):
        missing = [n for n, p in zip(names, present, strict=True) if not p]
        raise ValueError(f"context fields must be all set or all None; missing {missing}")
    return all(present)

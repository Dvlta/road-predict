"""WOMD `Scenario` protobuf -> `RawScenario` (design doc §11, Layer A).

This is the only module that imports protobuf code. Everything downstream consumes RawScenario.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from operator import attrgetter
from pathlib import Path

import numpy as np
from google.protobuf import unknown_fields

from prs.data.raw.protos import scenario_pb2
from prs.data.raw.tfrecord import read_records
from prs.data.types import POINT_KINDS, POLYGON_KINDS, POLYLINE_KINDS, MapFeature, RawScenario

# ObjectState fields in the column order used by scenario_from_proto.
_STATE_FIELDS = (
    "center_x", "center_y", "heading", "velocity_x", "velocity_y", "length", "width", "valid",
)  # fmt: skip
_get_state = attrgetter(*_STATE_FIELDS)


def parse_scenario(record: bytes) -> RawScenario:
    return scenario_from_proto(scenario_pb2.Scenario.FromString(record))


def iter_scenarios(path: str | Path, scenario_ids: set[str] | None = None) -> Iterator[RawScenario]:
    """Yield the scenes in one shard, optionally only those with the given ids."""
    for record in read_records(path):
        proto = scenario_pb2.Scenario.FromString(record)
        if scenario_ids is None or proto.scenario_id in scenario_ids:
            yield scenario_from_proto(proto)


def find_scenario(paths: Iterable[str | Path], scenario_id: str) -> tuple[RawScenario, Path]:
    """Return the scene with `scenario_id` and the shard it came from."""
    paths = [Path(p) for p in paths]
    for path in paths:
        for scenario in iter_scenarios(path, {scenario_id}):
            return scenario, path
    raise KeyError(f"scenario {scenario_id} not found in {[str(p) for p in paths]}")


def scenario_from_proto(proto: scenario_pb2.Scenario) -> RawScenario:
    num_frames = len(proto.timestamps_seconds)
    num_tracks = len(proto.tracks)
    if num_frames == 0:
        raise ValueError(f"scenario {proto.scenario_id} has no timestamps")
    if not 0 <= proto.sdc_track_index < num_tracks:
        raise ValueError(f"scenario {proto.scenario_id}: sdc_track_index out of range")

    states = np.empty((num_tracks, num_frames, len(_STATE_FIELDS)), dtype=np.float64)
    for i, track in enumerate(proto.tracks):
        if len(track.states) != num_frames:
            raise ValueError(
                f"scenario {proto.scenario_id}: track {track.id} has {len(track.states)} "
                f"states, expected {num_frames}"
            )
        states[i] = [_get_state(s) for s in track.states]

    return RawScenario(
        scenario_id=proto.scenario_id,
        timestamps=np.array(list(proto.timestamps_seconds), dtype=np.float64),
        current_time_index=proto.current_time_index,
        sdc_index=proto.sdc_track_index,
        track_ids=np.array([t.id for t in proto.tracks], dtype=np.int64),
        track_types=np.array([t.object_type for t in proto.tracks], dtype=np.int64),
        xy=states[..., 0:2].copy(),
        heading=states[..., 2].copy(),
        velocity=states[..., 3:5].copy(),
        size=states[..., 5:7].copy(),
        valid=states[..., 7] > 0.5,
        tracks_to_predict=np.array([r.track_index for r in proto.tracks_to_predict], np.int64),
        objects_of_interest=np.array(list(proto.objects_of_interest), dtype=np.int64),
        map_features=[f for f in map(_map_feature, proto.map_features) if f is not None],
        unknown_field_numbers=tuple(
            sorted({field.field_number for field in unknown_fields.UnknownFieldSet(proto)})
        ),
    )


def _map_feature(feature) -> MapFeature | None:
    kind = feature.WhichOneof("feature_data")
    if kind is None:
        return None
    data = getattr(feature, kind)
    if kind in POLYLINE_KINDS:
        points, subtype = data.polyline, data.type
    elif kind in POLYGON_KINDS:
        points, subtype = data.polygon, 0
    elif kind in POINT_KINDS:
        points, subtype = [data.position], 0
    else:
        raise ValueError(f"unknown map feature kind {kind!r}")
    xy = np.array([(p.x, p.y) for p in points], dtype=np.float64).reshape(-1, 2)
    return MapFeature(id=feature.id, kind=kind, subtype=int(subtype), points=xy)

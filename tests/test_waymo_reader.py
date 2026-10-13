"""Scenario proto -> RawScenario conversion."""

import numpy as np
import pytest
from conftest import TRAINING_SHARD_0

from prs.data.raw.protos import scenario_pb2
from prs.data.raw.tfrecord import read_records
from prs.data.raw.waymo_reader import parse_scenario
from prs.data.types import ObjectType


def _tiny_scenario_bytes() -> bytes:
    proto = scenario_pb2.Scenario(scenario_id="tiny", current_time_index=10, sdc_track_index=1)
    proto.timestamps_seconds.extend(i * 0.1 for i in range(91))
    for track_id, object_type in ((5, ObjectType.PEDESTRIAN), (8, ObjectType.VEHICLE)):
        track = proto.tracks.add(id=track_id, object_type=int(object_type))
        for i in range(91):
            track.states.add(
                center_x=24_000.0 + i, center_y=-18_000.0, heading=0.5, velocity_x=1.0,
                velocity_y=-2.0, length=4.0, width=2.0, valid=i != 3,
            )  # fmt: skip
    proto.tracks_to_predict.add(track_index=0)
    proto.objects_of_interest.extend([5, 8])
    lane = proto.map_features.add(id=1).lane
    lane.type = 3  # bike lane
    for x in range(4):
        lane.polyline.add(x=float(x), y=0.0)
    crosswalk = proto.map_features.add(id=2).crosswalk
    for x, y in ((0, 0), (4, 0), (4, 3), (0, 3)):
        crosswalk.polygon.add(x=float(x), y=float(y))
    proto.map_features.add(id=3).stop_sign.position.x = 7.0
    # Field 12 (stripped LiDAR) as a length-delimited unknown field: tag, length, payload.
    return proto.SerializeToString() + bytes([12 << 3 | 2, 3]) + b"abc"


def test_reader_converts_scenario():
    s = parse_scenario(_tiny_scenario_bytes())
    assert s.scenario_id == "tiny" and s.num_frames == 91 and s.num_tracks == 2
    assert s.current_time_index == 10 and s.sdc_index == 1
    assert s.track_ids.tolist() == [5, 8]
    assert s.track_types.tolist() == [ObjectType.PEDESTRIAN, ObjectType.VEHICLE]
    assert s.xy.shape == (2, 91, 2) and s.xy.dtype == np.float64
    assert s.xy[0, 4].tolist() == [24_004.0, -18_000.0]
    assert np.isclose(s.heading[0, 0], 0.5) and s.velocity[0, 0].tolist() == [1.0, -2.0]
    assert s.size[1, 0].tolist() == [4.0, 2.0]
    assert not s.valid[0, 3] and s.valid[0, 4]
    assert s.tracks_to_predict.tolist() == [0] and s.objects_of_interest.tolist() == [5, 8]
    kinds = {f.kind: f for f in s.map_features}
    assert kinds["lane"].subtype == 3 and kinds["lane"].points.shape == (4, 2)
    assert kinds["crosswalk"].is_polygon and kinds["crosswalk"].points.shape == (4, 2)
    assert kinds["stop_sign"].points.tolist() == [[7.0, 0.0]]
    assert s.unknown_field_numbers == (12,)


@pytest.mark.data
@pytest.mark.skipif(not TRAINING_SHARD_0.exists(), reason="training shard 0 not downloaded")
def test_reads_real_shard():
    s = parse_scenario(next(read_records(TRAINING_SHARD_0)))
    assert s.num_frames == 91 and s.current_time_index == 10
    assert s.track_types[s.sdc_index] == ObjectType.VEHICLE

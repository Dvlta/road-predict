"""Map types, resampling, chunking and per-target chunk selection (design doc §9, §27)."""

import dataclasses

import numpy as np
import pytest
from conftest import PED, make_scenario

from prs.data.map_extraction import (
    extract_map,
    prepare_scene_map,
    resample_polyline,
    split_polyline,
)
from prs.data.map_types import MapType, map_type
from prs.data.preprocess import build_sample
from prs.data.windows import make_window


def test_map_type_mapping():
    assert map_type("lane", 3) == MapType.LANE_BIKE
    assert map_type("road_line", 1) == MapType.ROAD_LINE_BROKEN
    assert map_type("road_line", 7) == MapType.ROAD_LINE_SOLID
    assert map_type("road_line", 8) == MapType.ROAD_LINE_SOLID  # passing double yellow
    assert map_type("road_edge", 2) == MapType.ROAD_EDGE_MEDIAN
    assert map_type("crosswalk", 0) == MapType.CROSSWALK
    assert map_type("stop_sign", 0) == MapType.STOP_SIGN
    assert MapType.PAD == 0
    with pytest.raises(ValueError, match="subtype"):
        map_type("lane", 9)
    with pytest.raises(ValueError, match="kind"):
        map_type("traffic_light", 0)


def test_resample_spacing():
    line = np.array([[0.0, 0.0], [4.0, 0.0], [10.0, 0.0]])
    resampled = resample_polyline(line, 1.0)
    np.testing.assert_allclose(resampled[:, 0], np.arange(11.0))
    short = resample_polyline(np.array([[0.0, 0.0], [2.5, 0.0]]), 1.0)
    assert len(short) == 4  # 3 equal steps of 0.833 m; both endpoints kept
    np.testing.assert_allclose(short[[0, -1]], [[0.0, 0.0], [2.5, 0.0]])


def test_resample_drops_repeated_points():
    line = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    resampled = resample_polyline(line, 1.0)
    assert len(resampled) == 6 and np.allclose(resampled[-1], [3.0, 4.0])
    assert len(resample_polyline(np.zeros((3, 2)), 1.0)) == 1


def test_polyline_chunking():
    points = np.arange(40.0)[:, None].repeat(2, axis=1)
    chunks = split_polyline(points, 20)
    assert [len(c) for c in chunks] == [20, 20, 2]
    for first, second in zip(chunks, chunks[1:], strict=False):
        np.testing.assert_array_equal(first[-1], second[0])  # shared endpoints: no gaps
    assert [len(c) for c in split_polyline(points[:20], 20)] == [20]
    assert [len(c) for c in split_polyline(points[:21], 20)] == [20, 2]


def test_polygon_closed(scenario, data_cfg):
    scene_map = prepare_scene_map(scenario, data_cfg)
    crosswalk = np.flatnonzero(scene_map.types == MapType.CROSSWALK)
    assert len(crosswalk) == 1  # perimeter 14 m at 1 m spacing: 15 points, one chunk
    points = scene_map.points[crosswalk[0], scene_map.valid[crosswalk[0]]]
    assert len(points) == 15
    np.testing.assert_allclose(points[0], points[-1])


def test_stop_sign_single_point(scenario, data_cfg):
    sample = build_sample(scenario, PED, data_cfg, make_window(data_cfg))
    stop = np.flatnonzero(sample.map_types == MapType.STOP_SIGN)
    assert len(stop) == 1 and sample.map_valid[stop[0]].sum() == 1
    np.testing.assert_array_equal(sample.map_polylines[stop[0], 0, 2:], [0.0, 0.0])  # no direction


def test_map_in_target_frame(scenario, data_cfg):
    """The lane runs east-west 5 m south of the north-facing pedestrian: in the target frame
    it lies 5 m behind (x = -5) and runs toward -y (east is to the right)."""
    sample = build_sample(scenario, PED, data_cfg, make_window(data_cfg))
    lane_chunks = np.flatnonzero(sample.map_types == MapType.LANE_SURFACE_STREET)
    assert len(lane_chunks) == 5  # 80 m of lane, 1 m spacing, chunks of up to 20 points
    for c in lane_chunks:
        valid = sample.map_valid[c]
        np.testing.assert_allclose(sample.map_polylines[c, valid, 0], -5.0, atol=1e-4)
        directions = sample.map_polylines[c, valid, 2:]
        np.testing.assert_allclose(
            directions, np.tile([0.0, -1.0], (len(directions), 1)), atol=1e-4
        )


def test_map_selection_nearest_and_capped(scenario, data_cfg):
    frame = build_sample(scenario, PED, data_cfg, make_window(data_cfg)).frame
    scene_map = prepare_scene_map(scenario, data_cfg)
    capped = dataclasses.replace(data_cfg, max_map_polylines=2)
    polylines, valid, types, candidates = extract_map(scene_map, frame, capped)
    assert candidates == 7  # 5 lane chunks + crosswalk + stop sign, all within 50 m
    # Nearest two: the crosswalk (≈ 2.2 m away) and a lane chunk (5 m behind).
    assert set(types.tolist()) == {MapType.CROSSWALK, MapType.LANE_SURFACE_STREET}
    nearest = [np.linalg.norm(polylines[c, valid[c], :2], axis=1).min() for c in range(2)]
    assert nearest == sorted(nearest)

    small = dataclasses.replace(data_cfg, map_radius_m=4.0)
    polylines, valid, types, candidates = extract_map(scene_map, frame, small)
    assert candidates == 1 and types[0] == MapType.CROSSWALK
    assert (types[1:] == MapType.PAD).all() and not valid[1:].any() and not polylines[1:].any()


def test_empty_map(data_cfg):
    scenario = dataclasses.replace(make_scenario(), map_features=[])
    sample = build_sample(scenario, PED, data_cfg, make_window(data_cfg))
    assert not sample.map_valid.any() and (sample.map_types == MapType.PAD).all()
    assert sample.meta.map_chunks_within_radius == 0

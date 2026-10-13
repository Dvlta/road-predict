"""Windows, target filters and target-centric sample construction (design doc §5-§7, §27)."""

import dataclasses

import numpy as np
import pytest
from conftest import (
    CYCLIST,
    PARKED_CAR,
    PED,
    PED_FINAL_INVALID,
    PED_INVALID_T0,
    PED_SHORT_HISTORY,
    SDC,
    T0,
    TRAINING_SHARD_0,
    make_scenario,
)

from prs.data.preprocess import build_sample
from prs.data.raw.waymo_reader import iter_scenarios
from prs.data.targets import ACCEPTED, build_target_sample, frame_heading, select_targets
from prs.data.windows import Window, make_window
from prs.models.base import MODEL_INPUT_FIELDS


@pytest.fixture
def window(data_cfg) -> Window:
    return make_window(data_cfg)


def test_window_indices(window):
    history = np.arange(91)[window.history_slice]
    future = np.arange(91)[window.future_slice]
    assert history.tolist() == list(range(0, 21)) and future.tolist() == list(range(21, 81))
    assert history.max() == window.t0 < future.min()
    assert window.history_len == 21 and window.final_future_index == 80


def test_window_rejects_out_of_range(data_cfg):
    make_window(data_cfg, t0=30)
    for t0 in (19, 31):
        with pytest.raises(ValueError, match="does not fit"):
            make_window(data_cfg, t0=t0)


def test_target_type_filter(scenario, data_cfg, window):
    accepted, stats = select_targets(scenario, data_cfg, window)
    assert accepted == [PED, CYCLIST]
    assert stats.as_dict() == {
        "vehicle": {"not_target_type": 2},
        "pedestrian": {
            ACCEPTED: 1,
            "invalid_at_t0": 1,
            "final_future_invalid": 1,
            "insufficient_history": 1,
        },
        "cyclist": {ACCEPTED: 1},
    }
    assert PARKED_CAR not in accepted and SDC not in accepted


def test_require_final_future_valid(scenario, data_cfg, window):
    lenient = dataclasses.replace(data_cfg, require_final_future_valid=False)
    accepted, _ = select_targets(scenario, lenient, window)
    assert PED_FINAL_INVALID in accepted


def test_build_rejects_invalid_targets(scenario, data_cfg, window):
    for index in (PED_INVALID_T0, PED_SHORT_HISTORY, PARKED_CAR, SDC):
        with pytest.raises(ValueError, match="not a valid target"):
            build_target_sample(scenario, index, data_cfg, window)


def test_irregular_timestamps_rejected(scenario, data_cfg, window):
    dropped = scenario.timestamps.copy()
    dropped[30:] += 0.1  # a dropped frame inside the window: one 0.2 s step
    accepted, stats = select_targets(
        dataclasses.replace(scenario, timestamps=dropped), data_cfg, window
    )
    assert accepted == []
    # Every track in the scene: 4 pedestrians, 1 cyclist, 2 vehicles.
    assert stats.as_dict() == {
        "pedestrian": {"irregular_timestamps": 4},
        "cyclist": {"irregular_timestamps": 1},
        "vehicle": {"irregular_timestamps": 2},
    }

    late = scenario.timestamps.copy()
    late[85:] += 0.1  # after the window's last frame (80): irrelevant
    accepted, _ = select_targets(dataclasses.replace(scenario, timestamps=late), data_cfg, window)
    assert accepted == [PED, CYCLIST]


def test_scene_length_mismatch(scenario, data_cfg, window):
    short = dataclasses.replace(scenario, timestamps=scenario.timestamps[:80])
    with pytest.raises(ValueError, match="80 frames"):
        select_targets(short, data_cfg, window)


def test_frame_heading_auto_rule():
    velocity = np.array([0.0, 1.4])
    assert frame_heading(0.3, velocity, "auto", 0.5) == (pytest.approx(np.pi / 2), "velocity")
    assert frame_heading(0.3, velocity * 0.1, "auto", 0.5) == (0.3, "box")
    assert frame_heading(0.3, velocity, "box", 0.5) == (0.3, "box")
    assert frame_heading(0.3, np.zeros(2), "velocity", 0.5) == (0.3, "box")


def test_history_future_shapes(scenario, data_cfg, window):
    sample = build_target_sample(scenario, PED, data_cfg, window)
    assert sample.target_history.shape == (21, 9) and sample.target_future.shape == (60, 2)
    assert sample.ego_history.shape == (21, 9) and sample.ego_future.shape == (60, 5)
    assert sample.neighbors is None and sample.map_polylines is None  # Stage 3


def test_target_at_origin_facing_x(scenario, data_cfg, window):
    sample = build_target_sample(scenario, PED, data_cfg, window)
    x, y, vx, vy, sin, cos, length, width, flag = sample.target_history[-1]
    np.testing.assert_allclose([x, y, vx, vy, sin, cos], [0, 0, 1.4, 0, 0, 1], atol=1e-5)
    assert (length, width, flag) == pytest.approx((0.6, 0.6, 1.0))
    # Walking straight north at 1.4 m/s: 2 s behind, 6 s ahead, all along the x axis.
    np.testing.assert_allclose(sample.target_history[0, :2], [-2.8, 0.0], atol=1e-4)
    np.testing.assert_allclose(sample.target_future[-1], [8.4, 0.0], atol=1e-4)
    np.testing.assert_allclose(sample.target_future[:, 1], 0.0, atol=1e-4)


def test_invalid_state_mask(scenario, data_cfg, window):
    sample = build_target_sample(scenario, CYCLIST, data_cfg, window)
    assert sample.target_history_valid.tolist() == [i not in (5, 6) for i in range(21)]
    assert not sample.target_history[[5, 6]].any()  # placeholders replaced by zeros
    assert sample.target_history[4, -1] == 1.0 and sample.target_history[5, -1] == 0.0
    future_index = 40 - (T0 + 1)
    assert not sample.target_future_valid[future_index]
    assert not sample.target_future[future_index].any()


def test_ego_extraction(scenario, data_cfg, window):
    sample = build_target_sample(scenario, PED, data_cfg, window)
    # The pedestrian faces north, so the SDC's eastward motion is toward the target's right.
    np.testing.assert_allclose(sample.ego_future[:, 3:5], np.tile([0.0, -8.0], (60, 1)), atol=1e-4)
    np.testing.assert_allclose(sample.ego_future[:, 2], -np.pi / 2, atol=1e-6)
    # At t0 the SDC is 30 m west, 5 m south: 5 m behind and 30 m to the left.
    np.testing.assert_allclose(sample.ego_history[-1, :2], [-5.0, 30.0], atol=1e-4)
    np.testing.assert_allclose(sample.ego_size, [4.8, 2.0])


def test_metadata(scenario, data_cfg, window):
    meta = build_target_sample(scenario, PED, data_cfg, window, shard="s0").meta
    assert (meta.shard, meta.scenario_id, meta.target_track_id, meta.t0) == (
        "s0",
        "synthetic",
        7,
        20,
    )
    assert meta.current_speed_mps == pytest.approx(1.4)
    assert meta.future_path_length_m == pytest.approx(8.4, abs=1e-6)
    assert meta.sdc_distance_m == pytest.approx(np.hypot(30.0, 5.0))
    assert meta.is_tracks_to_predict and meta.frame_heading_source_used == "velocity"
    assert not build_target_sample(scenario, CYCLIST, data_cfg, window).meta.is_tracks_to_predict


def test_sample_builder_no_leakage(data_cfg, window):
    """Design doc §14.3 check 2: changing anything after t0 must not change model inputs."""
    clean = make_scenario()
    perturbed = make_scenario()
    rng = np.random.default_rng(0)
    after = slice(window.t0 + 1, None)
    for name in ("xy", "heading", "velocity", "size"):
        array = getattr(perturbed, name)
        array[:, after] += rng.normal(scale=5.0, size=array[:, after].shape)

    for index in (PED, CYCLIST):
        a = build_sample(clean, index, data_cfg, window)
        b = build_sample(perturbed, index, data_cfg, window)
        for name in (*MODEL_INPUT_FIELDS, "ego_history", "ego_history_valid", "origin_global"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        assert a.heading_global == b.heading_global
        assert a.meta.current_speed_mps == b.meta.current_speed_mps
        assert not np.array_equal(a.target_future, b.target_future)  # the perturbation landed


@pytest.mark.data
@pytest.mark.skipif(not TRAINING_SHARD_0.exists(), reason="training shard 0 not downloaded")
def test_real_shard_samples_validate(data_cfg, window):
    built = 0
    for scenario in iter_scenarios(TRAINING_SHARD_0):
        accepted, _ = select_targets(scenario, data_cfg, window)
        for index in accepted:
            sample = build_sample(scenario, index, data_cfg, window)  # validates itself
            np.testing.assert_allclose(sample.target_history[-1, :2], 0.0, atol=1e-6)
            assert sample.neighbor_valid.any(axis=1).sum() == min(
                sample.meta.neighbors_within_radius, data_cfg.max_neighbors
            )
            assert sample.map_valid.any(axis=1).sum() == min(
                sample.meta.map_chunks_within_radius, data_cfg.max_map_polylines
            )
            built += 1
        if built >= 200:
            break
    assert built > 0

"""Ego-VRU conflict analysis and its ground-truth evaluation (design doc §22, §27)."""

import dataclasses

import numpy as np
import pytest
from conftest import PED, SDC, T0
from PIL import Image

from prs.data.targets import build_target_sample
from prs.data.windows import make_window
from prs.risk.conflict import analyze_conflict, mode_conflicts, separations
from prs.risk.evaluation import (
    analyzed,
    average_precision,
    compare_conflict_scores,
    conflict_table,
    ground_truth_conflicts,
    precision_recall_at,
    predicted_conflict_scores,
    scene_bootstrap,
)
from prs.risk.geometry import closest_point_on_box, point_to_box_distance
from prs.utils.config import RiskConfig
from prs.visualization.animation import END_HOLD_MS, FRAME_MS, T0_HOLD_MS, render_animation
from prs.visualization.conflict import (
    ClosestApproach,
    draw_closest_approach,
    draw_time_ticks,
    render_precision_recall,
)
from prs.visualization.predictions import render_sample

DT = 0.1
T = 60
RISK = RiskConfig(
    distance_mode="ego_box",
    vru_radius_m=0.4,
    conflict_distance_m=1.0,
    max_time_to_closest_approach_s=4.0,
    min_ego_speed_mps=1.0,
    max_sdc_distance_m=50.0,
)
EGO_SIZE = np.array([4.8, 2.0])  # a car: its bumper is 2.4 m ahead of its center
ALL_VALID = np.ones(T, dtype=bool)
TIMES = (np.arange(T) + 1) * DT  # index i is frame t0 + 1 + i


def ego_track(start, velocity) -> np.ndarray:
    """ego_future [T, 5] (x, y, heading, vx, vy) for constant-velocity driving."""
    start, velocity = np.asarray(start, float), np.asarray(velocity, float)
    xy = start + TIMES[:, None] * velocity
    heading = np.full(T, np.arctan2(velocity[1], velocity[0]))
    return np.column_stack([xy, heading, np.tile(velocity, (T, 1))])


def vru_track(start, velocity) -> np.ndarray:
    """A VRU path [T, 2] at constant velocity from `start` at t0."""
    return np.asarray(start, float) + TIMES[:, None] * np.asarray(velocity, float)


EGO_EAST = ego_track([-20.0, 0.0], [8.0, 0.0])  # center reaches x = 0 at 2.5 s
CROSSING = vru_track([0.0, -3.0], [0.0, 1.2])  # reaches the ego's lane center at 2.5 s


def analyze(modes, probabilities=None, ego=EGO_EAST, valid=ALL_VALID, risk=RISK):
    modes = np.asarray(modes, float).reshape(-1, T, 2)
    if probabilities is None:
        probabilities = np.full(len(modes), 1.0 / len(modes))
    return analyze_conflict(modes, probabilities, ego, valid, EGO_SIZE, risk, DT)


def test_point_to_box_distance():
    def distance(point, heading=0.0):
        return float(point_to_box_distance(np.array(point), np.zeros(2), heading, 4.8, 2.0))

    assert distance([0.5, 0.5]) == 0.0  # inside
    assert distance([2.4, 0.0]) == 0.0  # on the bumper
    assert distance([3.4, 0.0]) == pytest.approx(1.0)  # 1 m ahead of the bumper
    assert distance([0.0, -1.5]) == pytest.approx(0.5)  # beside the car
    assert distance([2.4 + 3.0, 1.0 + 4.0]) == pytest.approx(5.0)  # off a corner
    assert distance([0.0, 3.4], np.pi / 2) == pytest.approx(1.0)  # facing +y: length along y
    assert distance([2.0, 0.0], np.pi / 2) == pytest.approx(1.0)


def test_closest_point_on_box():
    def nearest(point, heading=0.0):
        return closest_point_on_box(np.array(point), np.zeros(2), heading, 4.8, 2.0)

    np.testing.assert_allclose(nearest([0.5, -0.5]), [0.5, -0.5])  # inside: the point itself
    np.testing.assert_allclose(nearest([5.0, 0.3]), [2.4, 0.3])  # in front: on the bumper
    np.testing.assert_allclose(nearest([5.0, 4.0]), [2.4, 1.0])  # off a corner: the corner
    np.testing.assert_allclose(nearest([0.0, 5.0], np.pi / 2), [0.0, 2.4], atol=1e-12)


def test_geometric_crossing():
    result = analyze([CROSSING])
    assert result.conflicting_modes == [0] and result.score == pytest.approx(1.0)
    assert result.min_separation_m == 0.0  # the pedestrian ends up inside the footprint
    assert result.time_to_closest_approach_s < 2.5  # contact begins before the centers meet
    x, y = result.closest_point_xy
    assert x == 0.0 and -3.0 < y < 0.0


def test_parallel_non_conflict():
    result = analyze([vru_track([0.0, 8.0], [1.4, 0.0])])
    assert result.conflicting_modes == [] and result.score == 0.0
    assert result.min_separation_m == pytest.approx(8.0 - 1.0 - 0.4)  # less half width, radius


def test_temporal_non_conflict():
    # The same path across the ego's lane, but the pedestrian is past it 2 s before the car.
    early = vru_track([0.0, -1.0], [0.0, 2.0])
    result = analyze([early])
    assert result.conflicting_modes == [] and result.min_separation_m > 1.5
    assert analyze([CROSSING]).conflicting_modes == [0]


def test_multimodal_conflict_score():
    far = vru_track([0.0, 15.0], [1.0, 0.0])
    farther = vru_track([5.0, -20.0], [0.0, -1.0])
    one = analyze([far, CROSSING, farther], [0.5, 0.2, 0.3])
    assert one.score == pytest.approx(0.2) and one.conflicting_modes == [1]
    assert one.highest_risk_mode == 1
    two = analyze([CROSSING, far, CROSSING + [0.0, 0.5]], [0.2, 0.5, 0.3])
    assert two.score == pytest.approx(0.5) and two.conflicting_modes == [0, 2]
    assert two.highest_risk_mode == 2  # the more probable conflicting mode


def test_highest_risk_mode_without_conflicts_is_the_closest():
    result = analyze(
        [vru_track([0.0, 15.0], [1.0, 0.0]), vru_track([0.0, 6.0], [1.0, 0.0])], [0.9, 0.1]
    )
    assert result.score == 0.0 and result.highest_risk_mode == 1
    assert result.min_separation_m == pytest.approx(6.0 - 1.0 - 0.4)


def test_invalid_ego_frames_skipped():
    # Whatever an unobserved frame holds is ignored, even an ego driving through the target.
    ego = ego_track([-40.0, 20.0], [8.0, 0.0])
    standing = vru_track([0.0, 0.0], [0.0, 0.0])
    on_top = ego.copy()
    on_top[10:30, :2] = 0.0
    valid = ALL_VALID.copy()
    valid[10:30] = False
    assert analyze([standing], ego=on_top).conflicting_modes == [0]
    assert analyze([standing], ego=on_top, valid=valid).conflicting_modes == []
    # Unobserved frames around the meeting hide the crossing conflict too.
    meeting = (TIMES > 1.5) & (TIMES < 3.5)
    assert analyze([CROSSING], valid=~meeting).conflicting_modes == []


def test_no_valid_ego_frame():
    result = analyze([CROSSING], valid=np.zeros(T, dtype=bool))
    assert result.conflicting_modes == [] and result.min_separation_m == np.inf
    assert np.isnan(result.time_to_closest_approach_s)


def test_bumper_conflict_ego_box():
    # A pedestrian 0.5 m ahead of the bumper, keeping pace with a car driving at 2 m/s.
    ego = ego_track([0.0, 0.0], [2.0, 0.0])
    ahead = vru_track([2.4 + 0.5, 0.0], [2.0, 0.0])
    box = analyze([ahead], ego=ego)
    assert box.conflicting_modes == [0]
    assert box.min_separation_m == pytest.approx(0.5 - 0.4)
    assert box.time_to_closest_approach_s == pytest.approx(DT)
    point = analyze([ahead], ego=ego, risk=dataclasses.replace(RISK, distance_mode="point"))
    assert point.conflicting_modes == [] and point.min_separation_m == pytest.approx(2.5)


def test_stationary_ego_no_conflict():
    parked = ego_track([0.0, 0.0], [0.0, 0.0])
    passing = vru_track([2.9, -3.0], [0.0, 1.2])  # walks across 0.5 m ahead of the bumper
    result = analyze([passing], ego=parked)
    assert result.min_separation_m < 1.0 and result.conflicting_modes == []
    creeping = parked.copy()
    creeping[:, 3] = 1.0  # same footprint, but the ego is moving at the speed threshold
    assert analyze([passing], ego=creeping).conflicting_modes == [0]


def test_closest_approach_time_limit():
    # The VRU is far from the ego except at one index, so its closest approach is that index.
    ego = ego_track([0.0, 0.0], [2.0, 0.0])
    for index, conflicts in ((39, True), (40, False)):  # 4.0 s and 4.1 s after t0
        path = np.full((T, 2), 100.0)
        path[index] = ego[index, :2]
        result = analyze([path], ego=ego)
        assert result.time_to_closest_approach_s == pytest.approx(TIMES[index])
        assert (result.conflicting_modes == [0]) is conflicts


def test_time_alignment(scenario, data_cfg):
    """Index i of the target future and of the ego future is the same frame, t0 + 1 + i: the
    separations match ones computed from the raw scene frame by frame."""
    sample = build_target_sample(scenario, PED, data_cfg, make_window(data_cfg))
    gaps = separations(
        sample.target_future[None, None], sample.ego_future[None],
        sample.ego_future_valid[None], sample.ego_size[None], RISK,
    )[0, 0]  # fmt: skip
    frames = np.arange(T0 + 1, T0 + 1 + T)
    # Distances do not depend on the frame, so the raw global states give the same values.
    expected = np.maximum(
        point_to_box_distance(
            scenario.xy[PED, frames], scenario.xy[SDC, frames],
            scenario.heading[SDC, frames], *scenario.size[SDC, T0],
        ) - RISK.vru_radius_m,
        0.0,
    )  # fmt: skip
    np.testing.assert_allclose(gaps, expected, atol=1e-3)  # the sample is float32
    modes = mode_conflicts(
        sample.target_future[None, None], sample.ego_future[None],
        sample.ego_future_valid[None], sample.ego_size[None], RISK, DT,
    )  # fmt: skip
    closest_frame = frames[np.argmin(expected)]
    assert modes["time_to_closest_s"][0, 0] == pytest.approx((closest_frame - T0) * DT)


def test_ground_truth_conflict_label():
    future = np.stack([CROSSING, CROSSING, vru_track([0.0, 8.0], [1.4, 0.0])])
    target_valid = np.ones((3, T), dtype=bool)
    target_valid[1, (TIMES > 1.5) & (TIMES < 3.5)] = False  # unobserved while it meets the car
    ego = np.stack([EGO_EAST] * 3)
    labels = ground_truth_conflicts(
        future, target_valid, ego, np.ones((3, T), bool), np.stack([EGO_SIZE] * 3), RISK, DT
    )
    assert labels.tolist() == [True, False, False]
    # The forecast score uses the same rule, but never the target's future validity.
    scores = predicted_conflict_scores(
        future[:, None], np.zeros((3, 1)), ego, np.ones((3, T), bool),
        np.stack([EGO_SIZE] * 3), RISK, DT,
    )  # fmt: skip
    assert scores.tolist() == [1.0, 1.0, 0.0]


def test_analysis_radius():
    distances = np.array([np.nan, 10.0, 50.0, 50.1])
    assert analyzed(distances, RISK).tolist() == [False, True, True, False]


def test_average_precision_by_hand():
    labels = np.array([True, False, True, False])
    assert average_precision(np.array([0.9, 0.8, 0.7, 0.6]), labels) == pytest.approx(
        0.5 * 1.0 + 0.5 * 2 / 3
    )
    # Tied scores count as one threshold; a binary score has a single operating point.
    tied = average_precision(np.array([1.0, 1.0, 0.0]), np.array([True, False, True]))
    assert tied == pytest.approx(0.5 * 0.5 + 0.5 * 2 / 3)
    assert average_precision(np.array([0.3, 0.2]), np.array([False, False])) is None


def test_precision_recall_at_threshold():
    scores, labels = np.array([0.9, 0.6, 0.4, 0.1]), np.array([True, False, True, False])
    assert precision_recall_at(scores, labels, 0.5) == (0.5, 0.5)
    assert precision_recall_at(scores, labels, 0.95) == (None, 0.0)
    assert precision_recall_at(scores, np.zeros(4, bool), 0.5) == (0.0, None)


def test_scene_bootstrap_resamples_whole_scenes_reproducibly():
    clusters = np.array(["a", "b", "a", "c", "c", "c"])
    first = list(scene_bootstrap(clusters, draws=50, seed=3))
    again = list(scene_bootstrap(clusters, draws=50, seed=3))
    assert all(np.array_equal(x, y) for x, y in zip(first, again, strict=True))
    sizes = {"a": 2, "b": 1, "c": 3}
    for draw in first:
        counts = {scene: int((clusters[draw] == scene).sum()) for scene in sizes}
        assert all(counts[scene] % sizes[scene] == 0 for scene in sizes)  # whole scenes only
        assert sum(counts.values()) == len(draw)
        for scene, size in sizes.items():  # each copy of a scene holds all of its samples
            members = np.sort(draw[clusters[draw] == scene])
            assert np.array_equal(members, np.repeat(np.flatnonzero(clusters == scene),
                                                     counts[scene] // size))  # fmt: skip


def test_compare_conflict_scores_pairs_the_models():
    rng = np.random.default_rng(0)
    labels = rng.random(400) < 0.2
    clusters = np.repeat(np.arange(100), 4)
    perfect = labels.astype(float)
    noisy = np.clip(perfect + rng.normal(0, 0.6, 400), 0, 1)
    result = compare_conflict_scores(
        {"noisy": noisy, "copy": noisy.copy(), "perfect": perfect}, labels, clusters, draws=200
    )
    assert "average_precision_diff" not in result["noisy"]
    assert result["copy"]["average_precision_diff"] == 0.0
    assert result["copy"]["average_precision_diff_ci95"] == [0.0, 0.0]  # paired: same draws
    assert result["perfect"]["average_precision"] == 1.0
    low, high = result["perfect"]["average_precision_diff_ci95"]
    assert 0 < low <= result["perfect"]["average_precision_diff"] <= high
    table = conflict_table({"overall": result})
    assert "| overall | perfect | 400 |" in table and "*" in table


def test_conflict_figures_render(scenario, data_cfg, tmp_path):
    sample = build_target_sample(scenario, PED, data_cfg, make_window(data_cfg))
    predictions = np.stack([sample.target_future, sample.target_future + [0.0, 1.0]])

    def overlay(ax):
        draw_time_ticks(ax, sample.ego_future[:, :2], sample.ego_future_valid, DT, "purple")
        point = predictions[1, 30]
        ego, size = sample.ego_future, sample.ego_size
        approach = ClosestApproach(30, point, "1.0 m", "red", "closest")
        return [draw_closest_approach(ax, ego, size, approach)]

    out = tmp_path / "conflict.png"
    render_sample(scenario, sample, data_cfg, out, predictions, np.array([0.7, 0.3]), "test",
                  highlighted_modes=[1], overlay=overlay)  # fmt: skip
    assert out.stat().st_size > 10_000
    curve = {"recall": np.array([0.5, 1.0]), "precision": np.array([1.0, 0.5])}
    pr = tmp_path / "pr.png"
    render_precision_recall({"model": curve}, {"model": (1.0, 0.5)}, 0.25, pr, "test")
    assert pr.stat().st_size > 10_000


def test_animation_plays_history_then_forecast_then_future(scenario, data_cfg, tmp_path):
    window = make_window(data_cfg)
    sample = build_target_sample(scenario, PED, data_cfg, window)
    predictions = np.stack([sample.target_future, sample.target_future + [0.0, 1.0]])
    approach = ClosestApproach(30, predictions[1, 30], "1.0 m", "red", "closest")
    out = tmp_path / "conflict.gif"
    frames = render_animation(scenario, sample, data_cfg, out, predictions, np.array([0.7, 0.3]),
                              "test", highlighted_modes=[1], approaches=[approach])  # fmt: skip
    # Every history frame up to t0, t0 again with the forecast, then every future frame.
    assert frames == window.history_len + 1 + window.future_steps
    with Image.open(out) as gif:
        assert gif.n_frames == frames
        durations = []
        for index in range(frames):
            gif.seek(index)
            durations.append(gif.info["duration"])
    assert set(durations[: window.history_len]) == {FRAME_MS}
    assert durations[window.history_len] == T0_HOLD_MS
    assert durations[-1] == END_HOLD_MS

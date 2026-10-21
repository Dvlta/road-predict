"""Counterfactual ego plans and synthetic crash labels."""

import numpy as np
from conftest import PED

from prs.data.targets import build_target_sample
from prs.data.windows import make_window
from prs.risk.counterfactual import (
    CONTACT_DISTANCE_M,
    MAX_CURVATURE_PER_M,
    MIN_POINT_SPACING_M,
    TRAFFIC_GAP_M,
    Traffic,
    contact_rule,
    no_yield_plan,
    no_yield_plans,
    no_yield_speeds,
)
from prs.risk.evaluation import ground_truth_approaches
from prs.risk.geometry import boxes_overlap
from prs.utils.config import RiskConfig
from prs.visualization.animation import render_animation

DT = 0.1
H, T = 21, 60
RISK = RiskConfig("ego_box", 0.4, 3.0, 4.0, 1.0, 50.0)
EGO_SIZE = np.array([4.8, 2.0])


def car(xy: np.ndarray, heading: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ego_history [H, 9] and ego_future [T, 5] from positions, headings and speeds over
    H + T frames (t0 is frame H - 1)."""
    velocity = speed[:, None] * np.column_stack([np.cos(heading), np.sin(heading)])
    history = np.zeros((H, 9))
    history[:, :2], history[:, 2:4] = xy[:H], velocity[:H]
    history[:, 4], history[:, 5] = np.sin(heading[:H]), np.cos(heading[:H])
    history[:, 6:8], history[:, 8] = EGO_SIZE, 1.0
    future = np.column_stack([xy[H:], heading[H:], velocity[H:]])
    return history, future


def distance_driven(speeds: np.ndarray) -> np.ndarray:
    """Trapezoid-integrated distance at each frame, 0 at t0."""
    distance = np.r_[0.0, np.cumsum(0.5 * (speeds[1:] + speeds[:-1]) * DT)]
    return distance - distance[H - 1]


def straight(speeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A car driving east along y = 0 at `speeds` [H + T], at x = 0 at t0."""
    x = distance_driven(speeds)
    return car(np.column_stack([x, np.zeros_like(x)]), np.zeros_like(x), speeds)


def plan_of(history, future, accel=1.5, traffic=None):
    return no_yield_plan(history, np.ones(H, bool), future, np.ones(T, bool), DT, accel, traffic)


def parked(x: float, y: float) -> Traffic:
    """One parked car facing east, centered at (x, y)."""
    return Traffic(
        centers=np.tile([x, y], (1, T, 1)).astype(float), headings=np.zeros((1, T)),
        sizes=np.tile([4.5, 1.9], (1, T, 1)), valid=np.ones((1, T), bool),
    )  # fmt: skip


def test_no_yield_speeds():
    logged = np.array([5.0, 4.0, 2.0, 0.0, 0.0, 6.0])
    speeds = no_yield_speeds(logged, start_speed=5.0, top_speed=6.0, accel=1.0, dt=1.0)
    assert speeds.tolist() == [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    speeds = no_yield_speeds(logged, start_speed=0.0, top_speed=8.0, accel=1.0, dt=1.0)
    assert speeds.tolist() == [5.0, 4.0, 3.0, 4.0, 5.0, 6.0]


def test_plan_matches_the_log_when_the_car_never_slows():
    history, future = straight(np.full(H + T, 10.0))
    plan, valid, extended, _ = plan_of(history, future)
    np.testing.assert_allclose(plan, future, atol=1e-6)
    assert valid.all() and not extended.any()


def test_no_yield_plan_drives_through_a_stop():
    # 8 m/s, braking from t0 to a stop 2 s later, then waiting.
    speeds = np.r_[np.full(H, 8.0), np.clip(8.0 - 4.0 * (np.arange(T) + 1) * DT, 0.0, None)]
    history, future = straight(speeds)
    plan, valid, extended, _ = plan_of(history, future)
    plan_speed = np.hypot(plan[:, 3], plan[:, 4])
    np.testing.assert_allclose(plan_speed, 8.0)  # never slower than its speed at t0
    # Exact while moving; near the stop, frames closer than MIN_POINT_SPACING_M to the last
    # kept path point take its arc length, so the plan may trail by up to that much.
    expected = 8.0 * (np.arange(T) + 1) * DT
    np.testing.assert_allclose(plan[:15, 0], expected[:15], atol=1e-9)
    assert np.all(np.abs(plan[:, 0] - expected) <= MIN_POINT_SPACING_M)
    np.testing.assert_allclose(plan[:, 1:3], 0.0, atol=1e-9)  # straight on past the stop
    stop = future[-1, 0]
    # The path ends at its last kept point, within MIN_POINT_SPACING_M of the stop.
    assert extended[plan[:, 0] > stop].all() and extended[-1]
    assert not extended[plan[:, 0] < stop - MIN_POINT_SPACING_M].any()


def test_extension_continues_a_turn():
    # A left turn on a 20 m circle at 6 m/s, stopping 1.5 s after t0.
    radius = 20.0
    speeds = np.r_[np.full(H + 15, 6.0), np.zeros(T - 15)]
    angle = distance_driven(speeds) / radius
    xy = radius * np.column_stack([np.sin(angle), 1.0 - np.cos(angle)])
    history, future = car(xy, angle, speeds)
    plan, _, extended, _ = plan_of(history, future)
    assert extended.any()
    distance_from_center = np.hypot(plan[extended, 0], plan[extended, 1] - radius)
    np.testing.assert_allclose(distance_from_center, radius, atol=0.3)
    turned = np.unwrap(plan[:, 2])
    assert np.all(np.diff(turned) >= -1e-9)  # keeps turning left, never back


def test_tight_curvature_is_capped():
    radius = 3.0  # tighter than any car turns at speed: treated as noise
    speeds = np.r_[np.full(H + 5, 3.0), np.zeros(T - 5)]
    angle = distance_driven(speeds) / radius
    xy = radius * np.column_stack([np.sin(angle), 1.0 - np.cos(angle)])
    history, future = car(xy, angle, speeds)
    plan, _, extended, _ = plan_of(history, future)
    step = np.diff(np.unwrap(plan[extended, 2]))
    distance = np.hypot(*np.diff(plan[extended, :2], axis=0).T)
    np.testing.assert_allclose(step / distance, MAX_CURVATURE_PER_M, rtol=0.02)


def test_parked_car_stays_put():
    history, future = straight(np.zeros(H + T))
    plan, _, extended, _ = plan_of(history, future)
    np.testing.assert_allclose(plan[:, :2], 0.0, atol=1e-9)
    np.testing.assert_allclose(plan[:, 3:], 0.0)
    assert not extended.any()


def test_unobserved_car_keeps_its_logged_future():
    history, future = straight(np.full(H + T, 5.0))
    history_valid = np.ones(H, bool)
    history_valid[-1] = False
    future_valid = np.ones(T, bool)
    future_valid[10] = False
    plan, valid, extended, held_from = no_yield_plan(
        history, history_valid, future, future_valid, DT
    )
    np.testing.assert_array_equal(plan, future)
    np.testing.assert_array_equal(valid, future_valid)
    assert not extended.any() and held_from == -1


def test_batched_plans_match_single_plans():
    histories, futures = zip(*(straight(np.full(H + T, v)) for v in (0.0, 4.0, 9.0)), strict=True)
    histories, futures = np.stack(histories), np.stack(futures)
    plans, valid, extended, held_from = no_yield_plans(
        histories, np.ones((3, H), bool), futures, np.ones((3, T), bool), DT
    )
    assert plans.shape == (3, T, 5) and valid.shape == extended.shape == (3, T)
    assert held_from.tolist() == [-1, -1, -1]
    np.testing.assert_allclose(plans[1], plan_of(histories[1], futures[1])[0])


def test_yielding_car_becomes_a_crash_without_yielding():
    # The car stops 6 m before the crossing at x = 20; a pedestrian walks across it.
    speeds = np.r_[np.full(H, 8.0), np.clip(8.0 - 3.0 * (np.arange(T) + 1) * DT, 0.0, None)]
    history, future = straight(speeds)
    stop = future[-1, 0]
    crossing_x = stop + 6.0
    times = (np.arange(T) + 1) * DT
    walker = np.column_stack([np.full(T, crossing_x), -4.0 + 1.4 * times])
    rule = contact_rule(RISK, horizon_s=6.0)
    assert rule.conflict_distance_m == CONTACT_DISTANCE_M
    assert rule.max_time_to_closest_approach_s == 6.0

    def contact(ego_future):
        return ground_truth_approaches(
            walker[None], np.ones((1, T), bool), ego_future[None], np.ones((1, T), bool),
            EGO_SIZE[None], rule, DT,
        )  # fmt: skip

    assert not contact(future)["conflict"][0]
    plan = plan_of(history, future).future
    crash = contact(plan)
    assert crash["conflict"][0] and crash["min_separation_m"][0] == 0.0
    # At 8 m/s the car's front reaches the crossing at 1.7 s and its rear clears it at 2.4 s;
    # the walker's disc enters the car's lane at 1.9 s.
    assert 1.8 <= crash["time_to_closest_s"][0] <= 2.0


def test_zero_acceleration_still_never_slows():
    speeds = np.r_[np.full(H, 6.0), np.linspace(6.0, 0.0, T)]
    history, future = straight(speeds)
    plan, *_ = plan_of(history, future, accel=0.0)
    np.testing.assert_allclose(np.hypot(plan[:, 3], plan[:, 4]), 6.0)


def test_boxes_overlap():
    square = dict(centers_a=[0.0, 0.0], headings_a=0.0, size_a=[2.0, 2.0], size_b=[2.0, 2.0])
    assert boxes_overlap(**square, centers_b=[1.9, 0.0], headings_b=0.0)
    assert not boxes_overlap(**square, centers_b=[2.1, 0.0], headings_b=0.0)
    diamond = np.pi / 4  # reaches 1 + sqrt(2) = 2.41 m along x
    assert boxes_overlap(**square, centers_b=[2.3, 0.0], headings_b=diamond)
    assert not boxes_overlap(**square, centers_b=[2.5, 0.0], headings_b=diamond)
    # Separated only along the diamond's own axis: corners nearly touch but do not.
    assert not boxes_overlap(**square, centers_b=[1.8, 1.8], headings_b=diamond)
    batch = boxes_overlap(np.zeros((1, 5, 2)), np.zeros((1, 5)), [4.0, 2.0], np.zeros((7, 5, 2)),
                          np.zeros((7, 5)), np.ones((7, 5, 2)))  # fmt: skip
    assert batch.shape == (7, 5) and batch.all()


def test_plan_stops_behind_a_parked_car():
    # The logged car slows to a stop behind a car parked at x = 18; the plan must not drive
    # through it, but stops with its front TRAFFIC_GAP_M short of the parked car's rear.
    speeds = np.r_[np.full(H, 8.0), np.clip(8.0 - 3.0 * (np.arange(T) + 1) * DT, 0.0, None)]
    history, future = straight(speeds)
    plan, _, _, held_from = plan_of(history, future, traffic=parked(18.0, 0.0))
    assert held_from > 0
    rear_of_parked = 18.0 - 2.25
    front = plan[:, 0] + EGO_SIZE[0] / 2
    assert front.max() <= rear_of_parked - TRAFFIC_GAP_M + 8.0 * DT  # at most one frame late
    assert front.max() > rear_of_parked - TRAFFIC_GAP_M - 8.0 * DT
    np.testing.assert_allclose(np.hypot(plan[held_from:, 3], plan[held_from:, 4]), 0.0)
    assert np.all(np.diff(plan[:, 0]) >= -1e-9)  # never backs up


def test_car_in_the_next_lane_does_not_stop_the_plan():
    speeds = np.r_[np.full(H, 8.0), np.clip(8.0 - 3.0 * (np.arange(T) + 1) * DT, 0.0, None)]
    history, future = straight(speeds)
    alone = plan_of(history, future)
    beside = plan_of(history, future, traffic=parked(18.0, 3.5))
    assert beside.held_from == -1
    np.testing.assert_array_equal(beside.future, alone.future)


def test_animation_follows_a_counterfactual_plan(scenario, data_cfg, tmp_path):
    window = make_window(data_cfg)
    sample = build_target_sample(scenario, PED, data_cfg, window)
    plan = sample.ego_future.copy()
    plan[:, 0] += 5.0
    out = tmp_path / "crash.gif"
    frames = render_animation(scenario, sample, data_cfg, out, sample.target_future[None],
                              np.ones(1), "test", ego_plan=plan)  # fmt: skip
    assert frames == window.history_len + 1 + window.future_steps
    assert out.stat().st_size > 10_000

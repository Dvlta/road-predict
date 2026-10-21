"""Counterfactual ego plans: synthetic stress tests of the conflict analysis.

WOMD holds no collisions: the logged SDC slowed or stopped for every VRU it met. Here the SDC
drives its own logged path with a speed profile that does not yield, so the logged VRU future
can be checked against a plan the SDC did not take. The plan starts from the SDC's real
position and speed at t0, and the forecast's inputs end at t0, so the saved predictions apply
unchanged.

**Open loop.** The VRU's logged future reacted to the real SDC; facing a car that does not
yield, a person would step back or run. A "contact" therefore means the person's real path
meets this hypothetical plan, not a simulated crash.

**Path.** The logged positions from the history through the future, with the logged headings.
A plan that drives further than the log continues past its end along an arc with the
curvature of the path's last few metres (a straight line on a straight road). Contacts past
the end of the logged path rest on that extension and are reported separately.

**Other vehicles.** The plan ignores pedestrians and cyclists, not cars: where its front would
come within `TRAFFIC_GAP_M` of another vehicle's logged position, it stops there for good
(or follows the log, if the log is further on). A car queued behind another is therefore not
driven through it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from prs.data.raw.waymo_reader import iter_scenarios
from prs.data.transforms import LocalFrame
from prs.data.types import AGENT_FEATURES, ObjectType, RawScenario, SampleMeta
from prs.data.windows import Window, make_window
from prs.risk.geometry import boxes_overlap
from prs.utils.config import DataConfig, RiskConfig

NO_YIELD_ACCEL_MPS2 = 1.5  # comfortable acceleration toward the plan's top speed
MIN_POINT_SPACING_M = 0.1  # closer path points are dropped: a stopped car's jitter
CURVATURE_WINDOW_M = 5.0  # the extension keeps the curvature of the path's last metres
MIN_CURVATURE_PATH_M = 1.0  # shorter paths extend in a straight line
MAX_CURVATURE_PER_M = 0.2  # a 5 m turning radius: tighter estimates are noise
CONTACT_DISTANCE_M = 1e-6  # separation (already less the VRU radius) below this = contact
TRAFFIC_GAP_M = 1.0  # the plan stops before its front comes this close to another vehicle
EGO_PLAN_FIELDS = ("ego_history", "ego_history_valid", "ego_future", "ego_future_valid")

_X, _Y, _VX, _VY, _SIN, _COS, _LENGTH, _WIDTH = (
    AGENT_FEATURES.index(n)
    for n in ("x", "y", "vx", "vy", "heading_sin", "heading_cos", "length", "width")
)


class Plan(NamedTuple):
    """A counterfactual ego future [T, 5] (x, y, heading, vx, vy) with its validity [T], the
    frames past the end of the logged path [T], and the first frame at which it stopped for
    another vehicle (-1 if never). Batched plans stack each field."""

    future: np.ndarray
    valid: np.ndarray
    extended: np.ndarray
    held_from: np.ndarray | int


@dataclass(frozen=True)
class Traffic:
    """Other vehicles over the future frames, in the target frame: centers [A, T, 2],
    headings [A, T], sizes [A, T, 2] (length, width), valid [A, T]."""

    centers: np.ndarray
    headings: np.ndarray
    sizes: np.ndarray
    valid: np.ndarray


def traffic_from_scenario(scenario: RawScenario, frame: LocalFrame, window: Window) -> Traffic:
    """Every vehicle but the SDC over the window's future frames, in `frame`."""
    vehicles = [
        i for i in range(scenario.num_tracks)
        if i != scenario.sdc_index and scenario.track_types[i] == ObjectType.VEHICLE
    ]  # fmt: skip
    future = window.future_slice
    return Traffic(
        centers=frame.points_to_local(scenario.xy[vehicles, future]),
        headings=frame.headings_to_local(scenario.heading[vehicles, future]),
        sizes=scenario.size[vehicles, future].astype(np.float64),
        valid=scenario.valid[vehicles, future],
    )


@dataclass(frozen=True)
class EgoPath:
    """The SDC's logged path: points [P, 2] in driving order with their headings [P]
    (unwrapped) and arc lengths [P]; `start` is the arc length of the SDC at t0."""

    points: np.ndarray
    headings: np.ndarray
    arc: np.ndarray
    start: float

    @property
    def length(self) -> float:
        return float(self.arc[-1])

    def curvature(self) -> float:
        """Heading change per metre over the path's last `CURVATURE_WINDOW_M`, 0 on paths
        shorter than `MIN_CURVATURE_PATH_M`."""
        if self.length < MIN_CURVATURE_PATH_M:
            return 0.0
        back = max(0.0, self.length - CURVATURE_WINDOW_M)
        turn = self.headings[-1] - np.interp(back, self.arc, self.headings)
        return float(np.clip(turn / (self.length - back), -MAX_CURVATURE_PER_M,
                             MAX_CURVATURE_PER_M))  # fmt: skip

    def pose(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions [N, 2] and headings [N] at arc lengths `s`; past the end, along the
        extension arc."""
        s = np.asarray(s, dtype=np.float64)
        xy = np.column_stack(
            [np.interp(s, self.arc, self.points[:, 0]), np.interp(s, self.arc, self.points[:, 1])]
        )
        heading = np.interp(s, self.arc, self.headings)
        beyond = np.maximum(s - self.length, 0.0)
        end_heading, kappa = self.headings[-1], self.curvature()
        if abs(kappa) < 1e-9:
            offset = beyond[:, None] * [np.cos(end_heading), np.sin(end_heading)]
        else:
            turned = end_heading + kappa * beyond
            offset = np.column_stack([np.sin(turned) - np.sin(end_heading),
                                      np.cos(end_heading) - np.cos(turned)]) / kappa  # fmt: skip
        extended = beyond > 0
        xy[extended] = self.points[-1] + offset[extended]
        heading[extended] = end_heading + kappa * beyond[extended]
        return xy, heading


def ego_path(
    ego_history: np.ndarray,
    ego_history_valid: np.ndarray,
    ego_future: np.ndarray,
    ego_future_valid: np.ndarray,
) -> tuple[EgoPath, np.ndarray] | None:
    """The logged path through t0, and the arc length of every future frame along it [T]
    (that of the last kept point for frames dropped as jitter, NaN for invalid frames). None
    when the SDC is not observed at t0."""
    if not ego_history_valid[-1]:
        return None
    history = np.asarray(ego_history, dtype=np.float64)
    future = np.asarray(ego_future, dtype=np.float64)
    past = history[ego_history_valid]
    past_heading = np.arctan2(past[:, _SIN], past[:, _COS])
    # Thin the history backward and the future forward from t0, so t0 is always a point.
    back = _thin(past[::-1, [_X, _Y]])
    points = list(past[::-1][back][::-1, [_X, _Y]])
    headings = list(past_heading[::-1][back][::-1])
    start_index = len(points) - 1
    frame_point = np.full(len(future), -1)
    for i in np.flatnonzero(ego_future_valid):
        if np.hypot(*(future[i, :2] - points[-1])) >= MIN_POINT_SPACING_M:
            points.append(future[i, :2])
            headings.append(future[i, 2])
        frame_point[i] = len(points) - 1
    points = np.asarray(points)
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    path = EgoPath(points, np.unwrap(np.asarray(headings)), arc, float(arc[start_index]))
    frame_arc = np.where(frame_point >= 0, arc[np.maximum(frame_point, 0)], np.nan)
    return path, frame_arc


def no_yield_speeds(
    logged_speeds: np.ndarray, start_speed: float, top_speed: float, accel: float, dt: float
) -> np.ndarray:
    """A speed profile that never slows for anyone [T]: the logged speed or, if higher,
    `start_speed` rising at `accel` up to `top_speed` (index i is (i + 1) * dt after t0)."""
    times = (np.arange(len(logged_speeds)) + 1) * dt
    return np.maximum(logged_speeds, np.minimum(start_speed + accel * times, top_speed))


def no_yield_plan(
    ego_history: np.ndarray,
    ego_history_valid: np.ndarray,
    ego_future: np.ndarray,
    ego_future_valid: np.ndarray,
    dt: float,
    accel: float = NO_YIELD_ACCEL_MPS2,
    traffic: Traffic | None = None,
) -> Plan:
    """The SDC's logged path driven without yielding, valid at every frame.

    The top speed is the fastest the SDC drove in the window, so a car that never moved
    stays put. The plan adds the distance its extra speed gains to the logged progress, so it
    matches the log until the logged speed first falls below the profile. With `traffic`,
    it stops for other vehicles (see the module docstring). When the SDC is not observed at
    t0 the logged future is returned unchanged."""
    future = np.asarray(ego_future, dtype=np.float64)
    built = ego_path(ego_history, ego_history_valid, future, ego_future_valid)
    if built is None:
        valid = np.asarray(ego_future_valid, dtype=bool).copy()
        return Plan(future.copy(), valid, np.zeros(len(future), dtype=bool), -1)
    path, frame_arc = built
    history = np.asarray(ego_history, dtype=np.float64)
    past = history[ego_history_valid]
    logged = np.where(ego_future_valid, np.hypot(future[:, 3], future[:, 4]), 0.0)
    start_speed = float(np.hypot(history[-1, _VX], history[-1, _VY]))
    top_speed = max(start_speed, float(np.hypot(past[:, _VX], past[:, _VY]).max()),
                    float(logged.max()))  # fmt: skip
    speeds = no_yield_speeds(logged, start_speed, top_speed, accel, dt)
    # Logged progress (carried over invalid frames) plus the extra distance driven.
    progress = _fill_forward(frame_arc, path.start)
    gained = np.cumsum(0.5 * (np.r_[start_speed, speeds[:-1]] - np.r_[start_speed, logged[:-1]]
                              + speeds - logged) * dt)  # fmt: skip
    s = progress + gained
    held_from = -1
    if traffic is not None and len(traffic.centers):
        size = history[-1, [_LENGTH, _WIDTH]]
        blocked = _meets_traffic(path, s, size, traffic) & (s > progress + 1e-6)
        if blocked.any():
            held_from = int(np.argmax(blocked))
            held = s[held_from - 1] if held_from > 0 else path.start
            s[held_from:] = np.maximum(progress[held_from:], held)
            speeds[held_from:] = np.where(progress[held_from:] >= held, logged[held_from:], 0.0)
    xy, heading = path.pose(s)
    plan = np.column_stack([xy, heading, speeds * np.cos(heading), speeds * np.sin(heading)])
    return Plan(plan, np.ones(len(plan), dtype=bool), s > path.length + 1e-9, held_from)


def no_yield_plans(
    ego_history: np.ndarray,
    ego_history_valid: np.ndarray,
    ego_future: np.ndarray,
    ego_future_valid: np.ndarray,
    dt: float,
    accel: float = NO_YIELD_ACCEL_MPS2,
    traffic: Sequence[Traffic | None] | None = None,
) -> Plan:
    """`no_yield_plan` for every sample, with its own traffic (None: none): fields stacked,
    future [B, T, 5], valid and extended [B, T], held_from [B]."""
    traffic = [None] * len(ego_history) if traffic is None else traffic
    built = [
        no_yield_plan(h, hv, f, fv, dt, accel, t)
        for h, hv, f, fv, t in zip(ego_history, ego_history_valid, ego_future, ego_future_valid,
                                   traffic, strict=True)
    ]  # fmt: skip
    return Plan(*(np.stack(parts) for parts in zip(*built, strict=True)))


def split_no_yield_plans(
    cfg: DataConfig,
    split: str,
    metas: Sequence[SampleMeta],
    columns: Mapping[str, np.ndarray],
    dt: float,
    accels: Sequence[float],
    wanted: np.ndarray,
) -> dict[float, Plan]:
    """No-yield plans for every sample of a cached split, at each acceleration. `wanted`
    samples stop for the vehicles of their raw scene, read from the split's raw shards one
    scene at a time; the others are planned without traffic. `columns` holds the cached
    ego history and future and the samples' `origin_global` and `heading_global`."""
    ego = [columns[name] for name in EGO_PLAN_FIELDS]
    plans = {accel: [no_yield_plan(*(c[i] for c in ego), dt, accel) for i in range(len(metas))]
             for accel in accels}  # fmt: skip
    scenes: dict[str, dict[str, list[int]]] = {}
    for i in np.flatnonzero(wanted):
        scenes.setdefault(metas[i].shard, {}).setdefault(metas[i].scenario_id, []).append(i)
    source = cfg.splits[split].source
    for shard, by_scene in sorted(scenes.items()):
        for scenario in iter_scenarios(cfg.raw_root / source / shard, set(by_scene)):
            for i in by_scene[scenario.scenario_id]:
                frame = LocalFrame(columns["origin_global"][i], columns["heading_global"][i])
                traffic = traffic_from_scenario(scenario, frame, make_window(cfg, metas[i].t0))
                for accel in accels:
                    plans[accel][i] = no_yield_plan(*(c[i] for c in ego), dt, accel, traffic)
    return {
        accel: Plan(*(np.stack(parts) for parts in zip(*built, strict=True)))
        for accel, built in plans.items()
    }


def contact_rule(risk: RiskConfig, horizon_s: float) -> RiskConfig:
    """The conflict rule narrowed to contact (the VRU's disc touches the ego footprint) at any
    time within `horizon_s`, while the ego moves at `risk.min_ego_speed_mps` or more."""
    return dataclasses.replace(
        risk, conflict_distance_m=CONTACT_DISTANCE_M, max_time_to_closest_approach_s=horizon_s
    )


def _meets_traffic(path: EgoPath, s: np.ndarray, size: np.ndarray, traffic: Traffic) -> np.ndarray:
    """Per frame [T]: whether the ego at arc length `s`, its front extended by
    `TRAFFIC_GAP_M`, overlaps a vehicle's logged footprint at that frame."""
    xy, heading = path.pose(s)
    forward = np.column_stack([np.cos(heading), np.sin(heading)])
    front = xy + 0.5 * TRAFFIC_GAP_M * forward
    padded = np.asarray(size, dtype=np.float64) + [TRAFFIC_GAP_M, 0.0]
    overlap = boxes_overlap(
        front[None], heading[None], padded, traffic.centers, traffic.headings, traffic.sizes
    )
    return (overlap & traffic.valid).any(axis=0)


def _thin(points: np.ndarray) -> np.ndarray:
    """Indices of points kept in order, each at least `MIN_POINT_SPACING_M` from the last."""
    keep = [0]
    for i in range(1, len(points)):
        if np.hypot(*(points[i] - points[keep[-1]])) >= MIN_POINT_SPACING_M:
            keep.append(i)
    return np.asarray(keep)


def _fill_forward(values: np.ndarray, first: float) -> np.ndarray:
    """NaNs replaced by the previous finite value (`first` before any)."""
    out = np.empty_like(values)
    last = first
    for i, value in enumerate(values):
        last = value if np.isfinite(value) else last
        out[i] = last
    return out

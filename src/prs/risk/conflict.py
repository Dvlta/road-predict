"""Conflicts between forecast VRU trajectories and the ego's logged future (design doc
§22.3-§22.5).

Index i of a forecast and of `ego_future` both refer to frame t0 + 1 + i, at time (i + 1) * dt
after t0: the sample builder cuts both with `Window.future_slice`. Functions are batched over
samples B and modes K; `analyze_conflict` is the single-target interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prs.risk.geometry import point_distance, point_to_box_distance
from prs.utils.config import RiskConfig

TIME_TOLERANCE_S = 1e-9  # (i + 1) * dt is not exact in floating point
TIE_TOLERANCE_M = 1e-6  # separations this close to the minimum are ties: the earliest wins


@dataclass(frozen=True)
class ConflictResult:
    """One target's conflict analysis (design doc §22.5)."""

    score: float  # sum of the conflicting modes' probabilities: a score, not a probability
    highest_risk_mode: int  # the most probable conflicting mode, else the closest mode
    min_separation_m: float  # of the highest-risk mode; inf if no ego frame is valid
    time_to_closest_approach_s: float  # of the highest-risk mode; NaN if no ego frame is valid
    closest_point_xy: tuple[float, float]  # where that mode is at its closest approach
    conflicting_modes: list[int]


def separations(
    trajectories: np.ndarray,
    ego_future: np.ndarray,
    frame_valid: np.ndarray,
    ego_size: np.ndarray,
    risk: RiskConfig,
) -> np.ndarray:
    """d_k(i), each mode's distance to the ego at the same index less the VRU radius, floored at
    0: [B, K, T] from trajectories [B, K, T, 2], ego_future [B, T, 5] and ego_size [B, 2]. inf
    where `frame_valid` [B, T] is false, so those frames are skipped."""
    points = np.asarray(trajectories, dtype=np.float64)
    ego = np.asarray(ego_future, dtype=np.float64)
    if points.ndim != 4 or ego.ndim != 3 or points.shape[2] != ego.shape[1]:
        raise ValueError(f"shape mismatch: trajectories {points.shape}, ego_future {ego.shape}")
    centers = ego[:, None, :, :2]  # [B, 1, T, 2]
    if risk.distance_mode == "ego_box":
        size = np.asarray(ego_size, dtype=np.float64)[:, None, None]  # [B, 1, 1, 2]
        distance = point_to_box_distance(
            points, centers, ego[:, None, :, 2], size[..., 0], size[..., 1]
        )
    else:
        distance = point_distance(points, centers)
    gap = np.maximum(distance - risk.vru_radius_m, 0.0)
    return np.where(np.asarray(frame_valid, dtype=bool)[:, None], gap, np.inf)


def mode_conflicts(
    trajectories: np.ndarray,
    ego_future: np.ndarray,
    frame_valid: np.ndarray,
    ego_size: np.ndarray,
    risk: RiskConfig,
    dt: float,
) -> dict[str, np.ndarray]:
    """Per mode [B, K]: the minimum separation, the index and time of closest approach (the
    earliest one on ties), the ego speed then, the mode's position then ([B, K, 2]), and
    whether the mode conflicts (design doc §22.4). Modes that never meet a valid frame get an
    infinite separation, NaN time, speed and position, and no conflict."""
    gaps = separations(trajectories, ego_future, frame_valid, ego_size, risk)
    min_separation = gaps.min(axis=2)
    # The first index at the minimum. A steady gap varies by rounding error from frame to
    # frame, which would otherwise pick an arbitrary later frame.
    index = (gaps <= min_separation[..., None] + TIE_TOLERANCE_M).argmax(axis=2)
    reached = np.isfinite(min_separation)
    ego = np.asarray(ego_future, dtype=np.float64)
    speeds = np.take_along_axis(np.hypot(ego[..., 3], ego[..., 4]), index, axis=1)
    points = np.asarray(trajectories, dtype=np.float64)
    closest = np.take_along_axis(points, index[..., None, None], axis=2)[:, :, 0]
    time = np.where(reached, (index + 1) * dt, np.nan)
    speeds = np.where(reached, speeds, np.nan)
    conflict = (
        reached
        & (min_separation < risk.conflict_distance_m)
        & (
            np.nan_to_num(time, nan=np.inf)
            <= risk.max_time_to_closest_approach_s + TIME_TOLERANCE_S
        )
        & (np.nan_to_num(speeds, nan=-np.inf) >= risk.min_ego_speed_mps)
    )
    return {
        "min_separation_m": min_separation,
        "closest_index": index,
        "time_to_closest_s": time,
        "ego_speed_mps": speeds,
        "closest_point_xy": np.where(reached[..., None], closest, np.nan),
        "conflict": conflict,
    }


def conflict_scores(conflict: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Sum of the conflicting modes' probabilities, in [0, 1]: [B]."""
    return np.where(conflict, np.asarray(probabilities, dtype=np.float64), 0.0).sum(axis=1)


def highest_risk_modes(
    conflict: np.ndarray, probabilities: np.ndarray, min_separation: np.ndarray
) -> np.ndarray:
    """The most probable conflicting mode, or the mode that comes closest when none
    conflicts: [B]."""
    most_probable = np.where(conflict, probabilities, -np.inf).argmax(axis=1)
    return np.where(conflict.any(axis=1), most_probable, np.asarray(min_separation).argmin(axis=1))


def analyze_conflict(
    predicted_trajectories: np.ndarray,
    mode_probabilities: np.ndarray,
    ego_future: np.ndarray,
    ego_future_valid: np.ndarray,
    ego_size: np.ndarray,
    risk: RiskConfig,
    dt: float,
) -> ConflictResult:
    """Conflict analysis of one target's K predicted modes [K, T, 2] with probabilities [K],
    against ego_future [T, 5] (x, y, heading, vx, vy), ego_future_valid [T] and ego_size
    (length, width). Every mode is analyzed, however unlikely (§22.5)."""
    modes = mode_conflicts(
        np.asarray(predicted_trajectories)[None], np.asarray(ego_future)[None],
        np.asarray(ego_future_valid)[None], np.asarray(ego_size)[None], risk, dt,
    )  # fmt: skip
    probabilities = np.asarray(mode_probabilities, dtype=np.float64)[None]
    conflict = modes["conflict"]
    k = int(highest_risk_modes(conflict, probabilities, modes["min_separation_m"])[0])
    x, y = modes["closest_point_xy"][0, k]
    return ConflictResult(
        score=float(conflict_scores(conflict, probabilities)[0]),
        highest_risk_mode=k,
        min_separation_m=float(modes["min_separation_m"][0, k]),
        time_to_closest_approach_s=float(modes["time_to_closest_s"][0, k]),
        closest_point_xy=(float(x), float(y)),
        conflicting_modes=[int(m) for m in np.flatnonzero(conflict[0])],
    )

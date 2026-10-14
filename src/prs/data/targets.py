"""Target selection and target-centric sample construction (design doc §5, §6, §7).

Stage 1 builds the target and ego parts of a sample. Neighbor and map context is added in
Stage 3 (`ProcessedSample` keeps those fields None until then).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from prs.data.transforms import LocalFrame
from prs.data.types import ObjectType, ProcessedSample, RawScenario, SampleMeta
from prs.data.windows import Window
from prs.utils.config import DataConfig

ACCEPTED = "accepted"
# Filters in the order they are applied; a rejected track counts under the first one it fails.
REJECTION_REASONS = (
    "irregular_timestamps",
    "not_target_type",
    "is_sdc",
    "invalid_at_t0",
    "insufficient_history",
    "insufficient_future",
    "final_future_invalid",
)


@dataclass
class FilterStats:
    """Target-filter outcomes keyed by (object type name, reason)."""

    counts: Counter[tuple[str, str]] = field(default_factory=Counter)

    def add(self, object_type: int, reason: str) -> None:
        self.counts[(ObjectType(object_type).name.lower(), reason)] += 1

    def update(self, other: FilterStats) -> None:
        self.counts.update(other.counts)

    def accepted(self, type_name: str) -> int:
        return self.counts[(type_name, ACCEPTED)]

    def as_dict(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for (type_name, reason), n in sorted(self.counts.items()):
            out.setdefault(type_name, {})[reason] = n
        return out


def check_scene(scenario: RawScenario, cfg: DataConfig) -> None:
    if scenario.num_frames != cfg.scene_num_frames:
        raise ValueError(
            f"scenario {scenario.scenario_id} has {scenario.num_frames} frames, "
            f"config expects {cfg.scene_num_frames}"
        )


def timestamps_regular(scenario: RawScenario, window: Window, cfg: DataConfig) -> bool:
    """True if every frame step inside the window is dt apart (within tolerance).

    About 3% of WOMD scenes drop a frame (a 0.2 s step); there, frame index no longer equals
    time, which would misalign labels, constant-velocity math and conflict timing (§5.2).
    """
    first = window.t0 - window.history_steps
    steps = np.diff(scenario.timestamps[first : window.final_future_index + 1])
    return bool(np.abs(steps - cfg.dt).max() <= cfg.max_timestep_error_s)


def check_target(scenario: RawScenario, index: int, cfg: DataConfig, window: Window) -> str:
    """Return ACCEPTED or the first filter the track fails (design doc §5.2)."""
    if not timestamps_regular(scenario, window, cfg):
        return "irregular_timestamps"
    if scenario.track_types[index] not in cfg.target_type_ids:
        return "not_target_type"
    if index == scenario.sdc_index:
        return "is_sdc"
    valid = scenario.valid[index]
    if not valid[window.t0]:
        return "invalid_at_t0"
    if valid[window.history_slice].mean() < cfg.min_history_valid_fraction:
        return "insufficient_history"
    if valid[window.future_slice].mean() < cfg.min_future_valid_fraction:
        return "insufficient_future"
    if cfg.require_final_future_valid and not valid[window.final_future_index]:
        return "final_future_invalid"
    return ACCEPTED


def select_targets(
    scenario: RawScenario, cfg: DataConfig, window: Window
) -> tuple[list[int], FilterStats]:
    """Track indices that pass every target filter, plus per-filter counts."""
    check_scene(scenario, cfg)
    stats = FilterStats()
    accepted = []
    for index in range(scenario.num_tracks):
        reason = check_target(scenario, index, cfg, window)
        stats.add(int(scenario.track_types[index]), reason)
        if reason == ACCEPTED:
            accepted.append(index)
    return accepted, stats


def frame_heading(
    box_heading: float, velocity: np.ndarray, source: str, min_speed_mps: float
) -> tuple[float, str]:
    """Global heading of the target frame's +x axis (design doc §6.1).

    Returns (theta, source actually used). Velocity direction is used only when the target
    moves fast enough for it to be meaningful; otherwise the box heading is the fallback.
    """
    speed = float(np.hypot(velocity[0], velocity[1]))
    use_velocity = (source == "velocity" and speed > 0.0) or (
        source == "auto" and speed >= min_speed_mps
    )
    if use_velocity:
        return float(np.arctan2(velocity[1], velocity[0])), "velocity"
    return float(box_heading), "box"


def encode_agent_states(
    scenario: RawScenario, index: int, frames: slice, frame: LocalFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Agent features float32[L, F_AGENT] in the local frame, and the validity mask.

    Invalid frames are all-zero, including valid_flag (the padding defined in design doc §7).
    """
    valid = scenario.valid[index, frames].copy()
    heading = frame.headings_to_local(scenario.heading[index, frames])
    features = np.concatenate(
        [
            frame.points_to_local(scenario.xy[index, frames]),
            frame.vectors_to_local(scenario.velocity[index, frames]),
            np.sin(heading)[:, None],
            np.cos(heading)[:, None],
            scenario.size[index, frames],
            np.ones((valid.shape[0], 1)),
        ],
        axis=1,
    )
    features[~valid] = 0.0
    return features.astype(np.float32), valid


def build_target_sample(
    scenario: RawScenario, index: int, cfg: DataConfig, window: Window, shard: str = ""
) -> ProcessedSample:
    """Target-centric sample for one target track (target + ego parts; design doc §7)."""
    check_scene(scenario, cfg)
    reason = check_target(scenario, index, cfg, window)
    if reason != ACCEPTED:
        raise ValueError(
            f"track {scenario.track_ids[index]} in scenario {scenario.scenario_id} is not a "
            f"valid target at t0={window.t0}: {reason}"
        )
    t0, sdc = window.t0, scenario.sdc_index
    theta, heading_source = frame_heading(
        scenario.heading[index, t0],
        scenario.velocity[index, t0],
        cfg.frame_heading_source,
        cfg.velocity_heading_min_speed_mps,
    )
    frame = LocalFrame(scenario.xy[index, t0], theta)

    target_history, target_history_valid = encode_agent_states(
        scenario, index, window.history_slice, frame
    )
    target_future_valid = scenario.valid[index, window.future_slice].copy()
    target_future_xy = frame.points_to_local(scenario.xy[index, window.future_slice])
    target_future_xy[~target_future_valid] = 0.0

    ego_history, ego_history_valid = encode_agent_states(scenario, sdc, window.history_slice, frame)
    ego_future_valid = scenario.valid[sdc, window.future_slice].copy()
    ego_future = np.concatenate(
        [
            frame.points_to_local(scenario.xy[sdc, window.future_slice]),
            frame.headings_to_local(scenario.heading[sdc, window.future_slice])[:, None],
            frame.vectors_to_local(scenario.velocity[sdc, window.future_slice]),
        ],
        axis=1,
    )
    ego_future[~ego_future_valid] = 0.0

    path = np.concatenate([np.zeros((1, 2)), target_future_xy[target_future_valid]])
    sdc_valid_now = bool(scenario.valid[sdc, t0])
    meta = SampleMeta(
        shard=shard,
        scenario_id=scenario.scenario_id,
        target_track_id=int(scenario.track_ids[index]),
        target_type=int(scenario.track_types[index]),
        t0=t0,
        current_speed_mps=float(np.hypot(*scenario.velocity[index, t0])),
        future_path_length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        is_tracks_to_predict=bool(np.isin(index, scenario.tracks_to_predict)),
        sdc_distance_m=(
            float(np.linalg.norm(scenario.xy[sdc, t0] - scenario.xy[index, t0]))
            if sdc_valid_now
            else float("nan")
        ),
        frame_heading_source_used=heading_source,
    )
    sample = ProcessedSample(
        meta=meta,
        target_type=meta.target_type,
        target_history=target_history,
        target_history_valid=target_history_valid,
        target_future=target_future_xy.astype(np.float32),
        target_future_valid=target_future_valid,
        ego_history=ego_history,
        ego_history_valid=ego_history_valid,
        ego_future=ego_future.astype(np.float32),
        ego_future_valid=ego_future_valid,
        ego_size=_ego_size(scenario, t0).astype(np.float32),
        origin_global=frame.origin.copy(),
        heading_global=frame.theta,
    )
    sample.validate()
    return sample


def _ego_size(scenario: RawScenario, t0: int) -> np.ndarray:
    """SDC (length, width) at t0, or at its latest valid frame before t0."""
    sdc = scenario.sdc_index
    valid_frames = np.flatnonzero(scenario.valid[sdc, : t0 + 1])
    if valid_frames.size == 0:
        raise ValueError(f"scenario {scenario.scenario_id}: SDC has no valid state up to t0")
    return scenario.size[sdc, valid_frames[-1]]

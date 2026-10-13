"""Shared fixtures: the dev config and a synthetic scene whose correct answers are known exactly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prs.data.types import MapFeature, ObjectType, RawScenario
from prs.utils.config import DataConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_CONFIG = REPO_ROOT / "configs" / "data_dev.yaml"
TRAINING_SHARD_0 = REPO_ROOT / "data/raw/womd_v1_3_1/training/training.tfrecord-00000-of-01000"

# Realistic global magnitude (WOMD coordinates reach ~25 km; design doc §42 Q2).
ORIGIN = np.array([24_620.9, -18_034.7])
NUM_FRAMES = 91
T0 = 20
DT = 0.1

# Track indices in the synthetic scene.
SDC, PED, CYCLIST, PED_INVALID_T0, PED_FINAL_INVALID, PARKED_CAR, PED_SHORT_HISTORY = range(7)
TRACK_IDS = (100, 7, 9, 11, 12, 13, 14)
PLACEHOLDER = -1.0  # what an invalid raw state holds here; real data is measured in §42 Q4


@pytest.fixture
def data_cfg() -> DataConfig:
    return load_config(DEV_CONFIG).data


def _straight_track(start: np.ndarray, velocity: np.ndarray, size: tuple[float, float]):
    """xy, heading, velocity, size arrays for constant-velocity motion through all frames."""
    t = (np.arange(NUM_FRAMES) - T0) * DT  # seconds relative to t0
    xy = start + t[:, None] * velocity
    heading = np.full(NUM_FRAMES, np.arctan2(velocity[1], velocity[0]))
    return xy, heading, np.tile(velocity, (NUM_FRAMES, 1)), np.tile(size, (NUM_FRAMES, 1))


def make_scenario() -> RawScenario:
    """Synthetic scene, positions given at t0:

    - SDC: 30 m west and 5 m south of the pedestrian, driving east at 8 m/s.
    - PED: at ORIGIN at t0, walking north at 1.4 m/s.
    - CYCLIST: 6 m north, riding west at 5 m/s; invalid at frames 5-6 and 40.
    - PED_INVALID_T0: invalid at t0.
    - PED_FINAL_INVALID: invalid at the final future frame only.
    - PARKED_CAR: a stationary vehicle.
    - PED_SHORT_HISTORY: only valid from frame 15 on.
    """
    specs = [
        (ObjectType.VEHICLE, ORIGIN + [-30.0, -5.0], [8.0, 0.0], (4.8, 2.0)),
        (ObjectType.PEDESTRIAN, ORIGIN, [0.0, 1.4], (0.6, 0.6)),
        (ObjectType.CYCLIST, ORIGIN + [10.0, 6.0], [-5.0, 0.0], (1.8, 0.7)),
        (ObjectType.PEDESTRIAN, ORIGIN + [3.0, 3.0], [1.0, 0.0], (0.6, 0.6)),
        (ObjectType.PEDESTRIAN, ORIGIN + [-3.0, 2.0], [0.0, -1.0], (0.6, 0.6)),
        (ObjectType.VEHICLE, ORIGIN + [5.0, -4.0], [0.0, 0.0], (4.5, 1.9)),
        (ObjectType.PEDESTRIAN, ORIGIN + [-6.0, -2.0], [1.2, 0.0], (0.6, 0.6)),
    ]
    tracks = [_straight_track(np.asarray(s), np.asarray(v, float), size) for _, s, v, size in specs]
    valid = np.ones((len(specs), NUM_FRAMES), dtype=bool)
    valid[CYCLIST, [5, 6, 40]] = False
    valid[PED_INVALID_T0, T0] = False
    valid[PED_FINAL_INVALID, T0 + 60] = False
    valid[PED_SHORT_HISTORY, :15] = False

    xy = np.stack([t[0] for t in tracks])
    heading = np.stack([t[1] for t in tracks])
    velocity = np.stack([t[2] for t in tracks])
    size = np.stack([t[3] for t in tracks])
    for array in (xy, heading, velocity, size):
        array[~valid] = PLACEHOLDER

    crosswalk = MapFeature(1, "crosswalk", 0, ORIGIN + np.array([[2, 1], [6, 1], [6, 4], [2, 4.0]]))
    lane = MapFeature(
        2, "lane", 2, ORIGIN + np.stack([np.linspace(-40, 40, 81), np.full(81, -5.0)], 1)
    )
    stop = MapFeature(3, "stop_sign", 0, ORIGIN + np.array([[8.0, -2.0]]))
    return RawScenario(
        scenario_id="synthetic",
        timestamps=np.arange(NUM_FRAMES) * DT,
        current_time_index=10,
        sdc_index=SDC,
        track_ids=np.array(TRACK_IDS, dtype=np.int64),
        track_types=np.array([int(s[0]) for s in specs], dtype=np.int64),
        xy=xy,
        heading=heading,
        velocity=velocity,
        size=size,
        valid=valid,
        tracks_to_predict=np.array([PED], dtype=np.int64),
        objects_of_interest=np.array([7, 100], dtype=np.int64),
        map_features=[crosswalk, lane, stop],
    )


@pytest.fixture
def scenario() -> RawScenario:
    return make_scenario()

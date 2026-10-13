"""Neighbor selection and encoding (design doc §8, §27)."""

import dataclasses

import numpy as np
from conftest import (
    CYCLIST,
    PARKED_CAR,
    PED,
    PED_FINAL_INVALID,
    PED_INVALID_T0,
    PED_SHORT_HISTORY,
    SDC,
    T0,
)

from prs.data.neighbor_selection import select_neighbors
from prs.data.preprocess import build_sample
from prs.data.types import ObjectType
from prs.data.windows import make_window

# Distances from the pedestrian at t0: PED_FINAL_INVALID 3.6 m, PED_SHORT_HISTORY 6.32 m,
# PARKED_CAR 6.40 m, CYCLIST 11.7 m, SDC 30.4 m. PED_INVALID_T0 is invalid at t0.
NEAREST_FIRST = [PED_FINAL_INVALID, PED_SHORT_HISTORY, PARKED_CAR, CYCLIST]


def test_sdc_first_then_nearest(scenario, data_cfg):
    selected, candidates = select_neighbors(scenario, PED, data_cfg, T0)
    assert selected.tolist() == [SDC, *NEAREST_FIRST] and candidates == 5
    assert PED not in selected and PED_INVALID_T0 not in selected


def test_sdc_by_distance_when_not_forced(scenario, data_cfg):
    cfg = dataclasses.replace(data_cfg, always_include_sdc=False)
    assert select_neighbors(scenario, PED, cfg, T0)[0].tolist() == [*NEAREST_FIRST, SDC]


def test_neighbor_cap_and_radius(scenario, data_cfg):
    capped = dataclasses.replace(data_cfg, max_neighbors=2)
    selected, candidates = select_neighbors(scenario, PED, capped, T0)
    assert selected.tolist() == [SDC, PED_FINAL_INVALID] and candidates == 5
    near = dataclasses.replace(data_cfg, neighbor_radius_m=5.0)  # the SDC is out of range
    assert select_neighbors(scenario, PED, near, T0)[0].tolist() == [PED_FINAL_INVALID]


def test_neighbor_padding(scenario, data_cfg):
    sample = build_sample(scenario, PED, data_cfg, make_window(data_cfg))
    assert sample.neighbors.shape == (48, 21, 9) and sample.meta.neighbors_within_radius == 5
    filled = sample.neighbor_valid.any(axis=1)
    assert filled.tolist() == [True] * 5 + [False] * 43
    assert not sample.neighbors[5:].any() and (sample.neighbor_types[5:] == 0).all()
    assert sample.neighbor_is_sdc.tolist() == [True] + [False] * 47
    assert sample.neighbor_types[:5].tolist() == [
        ObjectType.VEHICLE, ObjectType.PEDESTRIAN, ObjectType.PEDESTRIAN, ObjectType.VEHICLE,
        ObjectType.CYCLIST,
    ]  # fmt: skip


def test_neighbor_frames_masked_and_in_target_frame(scenario, data_cfg):
    sample = build_sample(scenario, PED, data_cfg, make_window(data_cfg))
    # Slot 2 is PED_SHORT_HISTORY, valid only from frame 15: the earlier frames are padding.
    assert sample.neighbor_valid[2].tolist() == [i >= 15 for i in range(21)]
    assert not sample.neighbors[2, :15].any()
    # Slot 0 is the SDC: same local position as the ego history (5 m behind, 30 m left).
    np.testing.assert_allclose(sample.neighbors[0, -1, :2], [-5.0, 30.0], atol=1e-4)
    np.testing.assert_array_equal(sample.neighbors[0], sample.ego_history)

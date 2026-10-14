"""Neighbor selection (design doc §8): the SDC first, then the nearest agents."""

from __future__ import annotations

import numpy as np

from prs.data.targets import encode_agent_states
from prs.data.transforms import LocalFrame
from prs.data.types import F_AGENT, RawScenario
from prs.data.windows import Window
from prs.utils.config import DataConfig


def neighbor_candidates(
    scenario: RawScenario, target_index: int, cfg: DataConfig, t0: int
) -> np.ndarray:
    """Track indices of every agent valid at t0 within the radius, in selection order: the SDC
    first (if configured), then nearest first."""
    positions = scenario.xy[:, t0]
    distances = np.linalg.norm(positions - positions[target_index], axis=1)
    candidates = scenario.valid[:, t0] & (distances <= cfg.neighbor_radius_m)
    candidates[target_index] = False
    order = np.flatnonzero(candidates)
    order = order[np.argsort(distances[order], kind="stable")]  # ties: lower track index first
    sdc = scenario.sdc_index
    if cfg.always_include_sdc and candidates[sdc]:
        order = np.concatenate([[sdc], order[order != sdc]])
    return order


def select_neighbors(
    scenario: RawScenario, target_index: int, cfg: DataConfig, t0: int
) -> tuple[np.ndarray, int]:
    """The first `max_neighbors` candidates, and how many candidates there were before the cap."""
    order = neighbor_candidates(scenario, target_index, cfg, t0)
    return order[: cfg.max_neighbors], len(order)


def encode_neighbors(
    scenario: RawScenario, indices: np.ndarray, cfg: DataConfig, window: Window, frame: LocalFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """neighbors [N, H, F_AGENT], neighbor_valid [N, H], neighbor_types [N], neighbor_is_sdc [N].

    Slots beyond the selected agents are padding: zero features, False masks, type 0 (UNSET).
    """
    N, H = cfg.max_neighbors, window.history_len
    neighbors = np.zeros((N, H, F_AGENT), dtype=np.float32)
    valid = np.zeros((N, H), dtype=bool)
    types = np.zeros(N, dtype=np.int64)
    is_sdc = np.zeros(N, dtype=bool)
    for slot, index in enumerate(indices):
        neighbors[slot], valid[slot] = encode_agent_states(
            scenario, int(index), window.history_slice, frame
        )
        types[slot] = scenario.track_types[index]
        is_sdc[slot] = index == scenario.sdc_index
    return neighbors, valid, types, is_sdc

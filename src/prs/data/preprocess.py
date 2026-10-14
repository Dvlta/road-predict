"""Build complete target-centric samples from raw shards (design doc §7-§9, §11)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from prs.data.map_extraction import SceneMap, extract_map, prepare_scene_map
from prs.data.neighbor_selection import encode_neighbors, select_neighbors
from prs.data.raw.waymo_reader import find_scenario, iter_scenarios
from prs.data.splits import shard_paths
from prs.data.targets import FilterStats, build_target_sample, select_targets
from prs.data.types import ObjectType, ProcessedSample, RawScenario
from prs.data.windows import Window, make_window
from prs.utils.config import DataConfig

VRU_TYPES = (ObjectType.PEDESTRIAN, ObjectType.CYCLIST)


def split_t0s(cfg: DataConfig, split: str) -> tuple[int, ...]:
    """Training may use several t0 per scene; val/test use exactly one (design doc §5.1)."""
    return cfg.train_current_indices if split == "train" else (cfg.current_index,)


def build_sample(
    scenario: RawScenario,
    index: int,
    cfg: DataConfig,
    window: Window,
    shard: str = "",
    scene_map: SceneMap | None = None,
) -> ProcessedSample:
    """Complete sample: target, ego, neighbors and map. Pass `scene_map` to reuse the scene's
    prepared map chunks across its targets."""
    sample = build_target_sample(scenario, index, cfg, window, shard=shard)
    frame = sample.frame
    selected, num_neighbors = select_neighbors(scenario, index, cfg, window.t0)
    neighbors, neighbor_valid, neighbor_types, neighbor_is_sdc = encode_neighbors(
        scenario, selected, cfg, window, frame
    )
    if scene_map is None:
        scene_map = prepare_scene_map(scenario, cfg)
    map_polylines, map_valid, map_types, num_chunks = extract_map(scene_map, frame, cfg)
    sample = dataclasses.replace(
        sample,
        meta=dataclasses.replace(
            sample.meta, neighbors_within_radius=num_neighbors, map_chunks_within_radius=num_chunks
        ),
        neighbors=neighbors,
        neighbor_valid=neighbor_valid,
        neighbor_types=neighbor_types,
        neighbor_is_sdc=neighbor_is_sdc,
        map_polylines=map_polylines,
        map_valid=map_valid,
        map_types=map_types,
    )
    sample.validate()
    return sample


def iter_shard_samples(
    path: Path, cfg: DataConfig, t0s: tuple[int, ...], stats: FilterStats | None = None
) -> Iterator[ProcessedSample]:
    windows = [make_window(cfg, t0) for t0 in t0s]
    for scenario in iter_scenarios(path):
        scene_map = None  # prepared lazily: scenes without targets never need it
        for window in windows:
            accepted, scene_stats = select_targets(scenario, cfg, window)
            if stats is not None:
                stats.update(scene_stats)
            for index in accepted:
                if scene_map is None:
                    scene_map = prepare_scene_map(scenario, cfg)
                yield build_sample(scenario, index, cfg, window, path.name, scene_map)


def iter_split_samples(
    cfg: DataConfig, split: str, stats: FilterStats | None = None
) -> Iterator[ProcessedSample]:
    """Every sample of a project split ("train", "val" or "test"), shard by shard."""
    for path in shard_paths(cfg, split):
        yield from iter_shard_samples(path, cfg, split_t0s(cfg, split), stats)


def find_sample(
    cfg: DataConfig, split: str, scenario_id: str, track_id: int
) -> tuple[RawScenario, ProcessedSample]:
    """The sample for one specific target, at the split's evaluation t0 (fails loudly if the
    scene, track or target filters say no)."""
    scenario, path = find_scenario(shard_paths(cfg, split), scenario_id)
    window = make_window(cfg, split_t0s(cfg, split)[0])
    sample = build_sample(scenario, scenario.track_index(track_id), cfg, window, path.name)
    return scenario, sample


def pick_examples(
    cfg: DataConfig, split: str, per_type: int, min_speed_mps: float = 0.0
) -> list[tuple[RawScenario, ProcessedSample]]:
    """Up to `per_type` pedestrian and cyclist samples, each from a different scene."""
    window = make_window(cfg)
    chosen: dict[ObjectType, list[tuple[RawScenario, ProcessedSample]]] = {t: [] for t in VRU_TYPES}
    for path in shard_paths(cfg, split):
        for scenario in iter_scenarios(path):
            accepted, _ = select_targets(scenario, cfg, window)
            for index in accepted:
                object_type = ObjectType(scenario.track_types[index])
                speed = float(np.hypot(*scenario.velocity[index, window.t0]))
                wanted = object_type in chosen and len(chosen[object_type]) < per_type
                if wanted and speed >= min_speed_mps:
                    sample = build_sample(scenario, index, cfg, window, path.name)
                    chosen[object_type].append((scenario, sample))
                    break  # one target per scene
            if all(len(v) == per_type for v in chosen.values()):
                return [example for t in VRU_TYPES for example in chosen[t]]
    return [example for t in VRU_TYPES for example in chosen[t]]

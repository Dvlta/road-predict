"""Measure how often map-chunk caps drop chunks near the target's actual path (design doc §9).

A chunk is "path-relevant" if any of its points lies within 5 m of the target's position at
t0 or any of its ground-truth future positions. This uses the future for analysis only; it is
never a model input.

    uv run python scripts/analyze_map_caps.py --config configs/data_dev.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from prs.data.map_extraction import prepare_scene_map
from prs.data.raw.waymo_reader import iter_scenarios
from prs.data.splits import shard_paths
from prs.data.targets import build_target_sample, select_targets
from prs.data.types import ObjectType
from prs.data.windows import make_window
from prs.utils.config import SPLIT_NAMES, load_config

RELEVANCE_M = 5.0
POLICIES = [  # (radius m, cap)
    (50.0, 64), (50.0, 128), (50.0, 192), (50.0, 256), (30.0, 64), (30.0, 128),
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config).data
    window = make_window(cfg)

    # (class, policy) -> [relevant chunks, dropped relevant chunks, targets losing one, over cap]
    stats: dict = defaultdict(lambda: np.zeros(4))
    targets: dict[str, int] = defaultdict(int)
    candidates_per_target: dict[str, list[int]] = defaultdict(list)
    for path in (p for split in SPLIT_NAMES for p in shard_paths(cfg, split)):
        for scenario in iter_scenarios(path):
            accepted, _ = select_targets(scenario, cfg, window)
            if not accepted:
                continue
            scene_map = prepare_scene_map(scenario, cfg)
            for index in accepted:
                name = ObjectType(scenario.track_types[index]).name.lower()
                sample = build_target_sample(scenario, index, cfg, window)
                local = sample.frame.points_to_local(scene_map.points)  # [C, P, 2]
                nearest = np.where(scene_map.valid, np.linalg.norm(local, axis=-1), np.inf).min(1)
                trail = np.vstack([[0.0, 0.0], sample.target_future[sample.target_future_valid]])
                gaps = np.linalg.norm(local[:, :, None, :] - trail[None, None], axis=-1)
                gaps = np.where(scene_map.valid[..., None], gaps, np.inf)
                relevant = gaps.min(axis=(1, 2)) <= RELEVANCE_M
                targets[name] += 1
                candidates_per_target[name].append(int((nearest <= cfg.map_radius_m).sum()))
                for radius, cap in POLICIES:
                    inside = np.flatnonzero(nearest <= radius)
                    kept = np.zeros(len(nearest), dtype=bool)
                    kept[inside[np.argsort(nearest[inside], kind="stable")][:cap]] = True
                    dropped = int((relevant & ~kept).sum())
                    stats[(name, radius, cap)] += [relevant.sum(), dropped, dropped > 0,
                                                   len(inside) > cap]  # fmt: skip

    for name in targets:
        counts = np.array(candidates_per_target[name])
        relevant_mean = stats[(name, *POLICIES[0])][0] / targets[name]
        print(
            f"\n{name}: {targets[name]} targets · chunks within {cfg.map_radius_m:.0f} m: "
            f"p50 {np.median(counts):.0f}, p90 {np.percentile(counts, 90):.0f}, "
            f"max {counts.max()} · path-relevant chunks per target: {relevant_mean:.1f}"
        )
        print(f"  {'policy':<18}{'over cap':>10}{'relevant dropped':>18}{'targets losing one':>20}")
        for radius, cap in POLICIES:
            relevant, dropped, losing, over = stats[(name, radius, cap)]
            policy = f"M={cap}, {radius:.0f} m"
            print(
                f"  {policy:<18}{over / targets[name]:>10.1%}"
                f"{dropped / max(relevant, 1):>18.1%}{losing / targets[name]:>20.1%}"
            )


if __name__ == "__main__":
    main()

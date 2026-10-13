"""Measure the dataset properties the design depends on (design doc §42).

Writes one JSON report per shard plus summary.json, and prints a summary keyed to the §42
question numbers.

    uv run python scripts/inspect_waymo.py --config configs/data_dev.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from prs.data.raw.waymo_reader import iter_scenarios
from prs.data.splits import shard_paths
from prs.data.targets import FilterStats, select_targets
from prs.data.types import POLYGON_KINDS, ObjectType, RawScenario
from prs.data.windows import Window, make_window
from prs.utils.config import SPLIT_NAMES, DataConfig, load_config
from prs.utils.geometry import wrap_angle

SPACING_BINS = np.arange(0.0, 10.0 + 1e-9, 0.05)  # meters; larger spacings land in the last bin
INVALID_FIELDS = ("x", "y", "heading", "vx", "vy", "length", "width")


def inspect_shard(
    path: Path, cfg: DataConfig, window: Window, max_scenes: int | None, stationary_mps: float
) -> tuple[dict, dict[str, list]]:
    """Return (shard report, per-target measurements)."""
    start = time.perf_counter()
    frames, current, unknown, track_types = Counter(), Counter(), Counter(), Counter()
    scenes_with, sdc_types, map_features, map_points = Counter(), Counter(), Counter(), Counter()
    invalid_values = {name: Counter() for name in INVALID_FIELDS}
    spacing_hist = defaultdict(lambda: np.zeros(len(SPACING_BINS) - 1))
    filter_stats = FilterStats()
    targets: dict[str, list] = defaultdict(list)
    num_scenes, sdc_valid_at_t0, max_abs_coord, max_step_error = 0, 0, 0.0, 0.0

    for scenario in tqdm(iter_scenarios(path), desc=path.name, unit="scene", disable=None):
        num_scenes += 1
        frames[scenario.num_frames] += 1
        current[scenario.current_time_index] += 1
        unknown.update(scenario.unknown_field_numbers)
        types_here = Counter(ObjectType(t).name.lower() for t in scenario.track_types)
        track_types.update(types_here)
        scenes_with.update(name for name in types_here)
        max_step_error = max(
            max_step_error, float(np.abs(np.diff(scenario.timestamps) - cfg.dt).max())
        )
        valid = scenario.valid
        max_abs_coord = max(max_abs_coord, float(np.abs(scenario.xy[valid]).max(initial=0.0)))
        _count_invalid_values(scenario, invalid_values)
        sdc_types[ObjectType(scenario.track_types[scenario.sdc_index]).name.lower()] += 1
        sdc_valid_at_t0 += bool(valid[scenario.sdc_index, window.t0])
        for feature in scenario.map_features:
            map_features[feature.kind] += 1
            map_points[feature.kind] += len(feature.points)
            if len(feature.points) >= 2:
                spacing = np.linalg.norm(np.diff(feature.points, axis=0), axis=1)
                spacing_hist[feature.kind] += np.histogram(
                    np.minimum(spacing, SPACING_BINS[-1] - 1e-9), bins=SPACING_BINS
                )[0]

        accepted, stats = select_targets(scenario, cfg, window)
        filter_stats.update(stats)
        if accepted:
            _measure_targets(scenario, accepted, cfg, window, stationary_mps, targets)
        if max_scenes is not None and num_scenes >= max_scenes:
            break

    report = {
        "shard": path.name,
        "bytes": path.stat().st_size,
        "scenes": num_scenes,
        "seconds": round(time.perf_counter() - start, 1),
        "frames_per_scene": dict(frames),
        "current_time_index": dict(current),
        "max_timestamp_step_error_s": max_step_error,
        "unknown_scenario_field_numbers": {str(k): v for k, v in sorted(unknown.items())},
        "tracks_by_type": dict(track_types),
        "scene_fraction_with_type": {k: v / num_scenes for k, v in scenes_with.items()},
        "max_abs_global_coordinate_m": max_abs_coord,
        "invalid_state_values": {
            name: _fractions(counter) for name, counter in invalid_values.items()
        },
        "sdc": {"types": dict(sdc_types), "valid_at_t0_fraction": sdc_valid_at_t0 / num_scenes},
        "map": {
            kind: {
                "features": map_features[kind],
                "points": map_points[kind],
                "median_spacing_m": _hist_median(spacing_hist[kind]),
            }
            for kind in sorted(map_features)
        },
        "targets": {
            "t0": window.t0,
            "filter_counts": filter_stats.as_dict(),
            "accepted": {name: filter_stats.accepted(name) for name in cfg.target_types},
        },
    }
    return report, targets


def _count_invalid_values(scenario: RawScenario, counters: dict[str, Counter]) -> None:
    invalid = ~scenario.valid
    columns = {
        "x": scenario.xy[..., 0], "y": scenario.xy[..., 1], "heading": scenario.heading,
        "vx": scenario.velocity[..., 0], "vy": scenario.velocity[..., 1],
        "length": scenario.size[..., 0], "width": scenario.size[..., 1],
    }  # fmt: skip
    for name, values in columns.items():
        v = values[invalid]
        counters[name]["-1"] += int((v == -1).sum())
        counters[name]["0"] += int((v == 0).sum())
        counters[name]["other"] += int(((v != -1) & (v != 0)).sum())


def _measure_targets(
    scenario: RawScenario,
    accepted: list[int],
    cfg: DataConfig,
    window: Window,
    stationary_mps: float,
    out: dict[str, list],
) -> None:
    t0, sdc = window.t0, scenario.sdc_index
    positions = scenario.xy[:, t0]
    valid_now = scenario.valid[:, t0]
    segments = _map_segments(scenario)
    stop_signs = np.array(
        [f.points[0] for f in scenario.map_features if f.kind == "stop_sign" and len(f.points)]
    ).reshape(-1, 2)
    chunk_length = (cfg.max_points_per_polyline - 1) * cfg.map_resample_spacing_m

    for index in accepted:
        p0 = positions[index]
        velocity = scenario.velocity[index, t0]
        speed = float(np.hypot(*velocity))
        out["type"].append(ObjectType(scenario.track_types[index]).name.lower())
        out["speed"].append(speed)
        out["stationary"].append(speed < stationary_mps)
        out["sdc_distance"].append(
            float(np.linalg.norm(positions[sdc] - p0)) if valid_now[sdc] else math.nan
        )
        heading_diff = math.nan
        if speed >= cfg.velocity_heading_min_speed_mps:
            diff = wrap_angle(scenario.heading[index, t0] - np.arctan2(velocity[1], velocity[0]))
            heading_diff = float(np.degrees(np.abs(diff)))
        out["heading_diff_deg"].append(heading_diff)

        distances = np.linalg.norm(positions - p0, axis=1)
        nearby = valid_now & (distances <= cfg.neighbor_radius_m)
        nearby[index] = False
        out["neighbors"].append(int(nearby.sum()))

        # Map chunk estimate: polyline length inside the radius / chunk length, per feature.
        starts, ends, feature_ids, lengths = segments
        inside = (np.linalg.norm(starts - p0, axis=1) <= cfg.map_radius_m) & (
            np.linalg.norm(ends - p0, axis=1) <= cfg.map_radius_m
        )
        length_in = np.bincount(
            feature_ids[inside], weights=lengths[inside], minlength=len(scenario.map_features)
        )
        chunks = int(np.ceil(length_in[length_in > 0] / chunk_length).sum())
        chunks += int((np.linalg.norm(stop_signs - p0, axis=1) <= cfg.map_radius_m).sum())
        out["map_chunks_est"].append(chunks)


def _map_segments(scenario: RawScenario) -> tuple[np.ndarray, ...]:
    starts, ends, ids = [], [], []
    for feature_id, feature in enumerate(scenario.map_features):
        points = feature.points
        if len(points) < 2:
            continue
        if feature.kind in POLYGON_KINDS:
            points = np.concatenate([points, points[:1]])
        starts.append(points[:-1])
        ends.append(points[1:])
        ids.append(np.full(len(points) - 1, feature_id))
    if not starts:
        empty = np.zeros((0, 2))
        return empty, empty, np.zeros(0, dtype=np.int64), np.zeros(0)
    starts_arr, ends_arr = np.concatenate(starts), np.concatenate(ends)
    lengths = np.linalg.norm(ends_arr - starts_arr, axis=1)
    return starts_arr, ends_arr, np.concatenate(ids), lengths


def _fractions(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {k: (v / total if total else 0.0) for k, v in counter.items()}


def _hist_median(hist: np.ndarray) -> float | None:
    if hist.sum() == 0:
        return None
    index = int(np.searchsorted(np.cumsum(hist), hist.sum() / 2))
    return round(float((SPACING_BINS[index] + SPACING_BINS[index + 1]) / 2), 3)


def _percentiles(values: list[float]) -> dict[str, float]:
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return {"count": 0}
    q = np.percentile(v, [10, 50, 90, 99])
    return {"count": int(v.size), "p10": q[0], "p50": q[1], "p90": q[2], "p99": q[3],
            "max": float(v.max())}  # fmt: skip


def summarize_targets(targets: dict[str, list], cfg: DataConfig) -> dict:
    types = np.array(targets["type"])
    summary: dict = {"by_type": {}}
    for name in cfg.target_types:
        mask = types == name
        if not mask.any():
            summary["by_type"][name] = {"count": 0}
            continue
        heading = np.array(targets["heading_diff_deg"])[mask]
        moving = heading[~np.isnan(heading)]
        summary["by_type"][name] = {
            "count": int(mask.sum()),
            "stationary_fraction": float(np.mean(np.array(targets["stationary"])[mask])),
            "heading_vs_velocity_deg_moving": {
                **_percentiles(list(moving)),
                "fraction_over_45": float(np.mean(moving > 45)) if moving.size else None,
                "fraction_over_135": float(np.mean(moving > 135)) if moving.size else None,
            },
        }
    sdc = np.array(targets["sdc_distance"], dtype=np.float64)
    neighbors = np.array(targets["neighbors"])
    chunks = np.array(targets["map_chunks_est"])
    summary["sdc_distance_m"] = {
        **_percentiles(list(sdc)),
        **{f"fraction_within_{r}m": float(np.mean(sdc[~np.isnan(sdc)] <= r)) for r in (10, 25, 50)},
    }
    summary["neighbors_within_radius"] = {
        **_percentiles(list(neighbors)),
        "fraction_over_cap": float(np.mean(neighbors > cfg.max_neighbors)),
    }
    summary["map_chunks_within_radius_est"] = {
        **_percentiles(list(chunks)),
        "fraction_over_cap": float(np.mean(chunks > cfg.max_map_polylines)),
    }
    return summary


def print_summary(reports: list[dict], summary: dict, cfg: DataConfig) -> None:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    print("\n=== Design doc §42 answers ===")
    print(
        "Q1  frames/scene:",
        _merge(r["frames_per_scene"] for r in reports),
        "| current_time_index:",
        _merge(r["current_time_index"] for r in reports),
        f"| max timestep error: {max(r['max_timestamp_step_error_s'] for r in reports):.2e} s",
    )
    print(
        f"Q2  max |global coordinate|: "
        f"{max(r['max_abs_global_coordinate_m'] for r in reports):,.1f} m"
    )
    print(
        "Q3  median map point spacing (m):",
        {k: v["median_spacing_m"] for k, v in reports[0]["map"].items()},
        f"({reports[0]['shard']})",
    )
    print(
        "Q4  invalid-state values:", reports[0]["invalid_state_values"], f"({reports[0]['shard']})"
    )
    for r in reports:
        print(
            f"Q5  {r['shard']}: {r['bytes'] / 1e6:.0f} MB, {r['scenes']} scenes, {r['seconds']} s"
            f" | Q6 accepted targets: {r['targets']['accepted']}"
        )
    print("Q6  filter counts (all shards):")
    merged = defaultdict(Counter)
    for r in reports:
        for type_name, counts in r["targets"]["filter_counts"].items():
            merged[type_name].update(counts)
    for type_name in cfg.target_types:
        print(f"      {type_name}: {dict(merged[type_name])}")
    for name, s in summary["by_type"].items():
        if s["count"] == 0:
            continue
        h = s["heading_vs_velocity_deg_moving"]
        print(f"Q7  {name}: {s['count']} targets, stationary {pct(s['stationary_fraction'])}")
        if h["count"]:
            print(
                f"Q9  {name} (moving): box-vs-velocity heading p50 {h['p50']:.1f}°, "
                f"p90 {h['p90']:.1f}°, >45° {pct(h['fraction_over_45'])}, "
                f">135° {pct(h['fraction_over_135'])}"
            )
    s = summary["sdc_distance_m"]
    print(
        f"Q8  target-SDC distance: p10 {s['p10']:.1f} m, p50 {s['p50']:.1f} m, "
        f"p90 {s['p90']:.1f} m | within 10/25/50 m: {pct(s['fraction_within_10m'])} / "
        f"{pct(s['fraction_within_25m'])} / {pct(s['fraction_within_50m'])}"
    )
    m = summary["map_chunks_within_radius_est"]
    print(
        f"Q10 map chunks within {cfg.map_radius_m:.0f} m (estimate): p50 {m['p50']:.0f}, "
        f"p90 {m['p90']:.0f}, max {m['max']:.0f} | over cap {cfg.max_map_polylines}: "
        f"{pct(m['fraction_over_cap'])}"
    )
    n = summary["neighbors_within_radius"]
    print(
        f"Q11 neighbors within {cfg.neighbor_radius_m:.0f} m: p50 {n['p50']:.0f}, "
        f"p90 {n['p90']:.0f}, max {n['max']:.0f} | over cap {cfg.max_neighbors}: "
        f"{pct(n['fraction_over_cap'])}"
    )
    print("Q12 unknown Scenario field numbers (scenes):",
          _merge(r["unknown_scenario_field_numbers"] for r in reports))  # fmt: skip


def _merge(dicts) -> dict:
    total: Counter = Counter()
    for d in dicts:
        total.update(d)
    return dict(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="all", choices=("all", *SPLIT_NAMES))
    parser.add_argument("--max-scenes", type=int, help="per shard, for quick runs")
    parser.add_argument("--stationary-threshold", type=float, default=0.5, help="m/s (doc §5.3)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/inspection"))
    args = parser.parse_args()

    cfg = load_config(args.config).data
    window = make_window(cfg)
    splits = SPLIT_NAMES if args.split == "all" else (args.split,)
    paths = [p for split in splits for p in shard_paths(cfg, split)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing shards (run scripts/download_shards.py): {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports, targets = [], defaultdict(list)
    for path in paths:
        report, shard_targets = inspect_shard(
            path, cfg, window, args.max_scenes, args.stationary_threshold
        )
        reports.append(report)
        for key, values in shard_targets.items():
            targets[key].extend(values)
        (args.out_dir / f"{path.name}.json").write_text(json.dumps(report, indent=2))

    summary = summarize_targets(targets, cfg)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"shards": [r["shard"] for r in reports], **summary}, indent=2, default=float)
    )
    print_summary(reports, summary, cfg)
    print(f"\nreports written to {args.out_dir}/")


if __name__ == "__main__":
    main()

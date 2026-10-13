"""Render target-centric BEV figures: the Stage 1 acceptance test (design doc §41).

    uv run python scripts/visualize_sample.py --config configs/data_dev.yaml --examples 3
    uv run python scripts/visualize_sample.py --config configs/data_dev.yaml \
        --scenario-id <id> --target-track-id <id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prs.data.preprocess import find_sample, pick_examples
from prs.data.types import ObjectType
from prs.utils.config import SPLIT_NAMES, load_config
from prs.visualization.predictions import render_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=SPLIT_NAMES)
    parser.add_argument("--scenario-id")
    parser.add_argument("--target-track-id", type=int)
    parser.add_argument("--examples", type=int, default=3, help="targets per VRU type")
    parser.add_argument("--min-speed", type=float, default=0.0, help="m/s, examples mode only")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures/samples"))
    args = parser.parse_args()
    if (args.scenario_id is None) != (args.target_track_id is None):
        parser.error("--scenario-id and --target-track-id must be given together")

    cfg = load_config(args.config).data
    if args.scenario_id is not None:
        examples = [find_sample(cfg, args.split, args.scenario_id, args.target_track_id)]
    else:
        examples = pick_examples(cfg, args.split, args.examples, args.min_speed)

    for scenario, sample in examples:
        meta = sample.meta
        out = args.out_dir / f"{meta.scenario_id}_{meta.target_track_id}.png"
        render_sample(scenario, sample, cfg, out)
        print(
            f"{ObjectType(meta.target_type).name.lower():<10} scenario {meta.scenario_id} "
            f"track {meta.target_track_id:<5} speed {meta.current_speed_mps:5.2f} m/s  -> {out}"
        )


if __name__ == "__main__":
    main()

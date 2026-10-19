"""Overlay a model's predicted futures on target-centric BEV figures (design doc §16, §23).

    uv run python scripts/visualize_prediction.py --config configs/baseline_cv.yaml --examples 2
    uv run python scripts/visualize_prediction.py --config configs/baseline_cv.yaml \
        --scenario-id <id> --target-track-id <id>
    uv run python scripts/visualize_prediction.py --config configs/lstm.yaml \
        --checkpoint outputs/runs/<run>/best.pt --examples 2 --animate

--animate also writes each example as a GIF: the history, then the forecast against the logged
future.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from prs.data import PREPROCESS_VERSION
from prs.data.collate import collate_samples
from prs.data.preprocess import find_sample, pick_examples
from prs.data.types import ObjectType
from prs.metrics.breakdowns import per_sample_metrics
from prs.models import build_model, is_learned, model_label
from prs.models.base import model_inputs
from prs.training.checkpoint import load_checkpoint, restore_model
from prs.utils.config import SPLIT_NAMES, config_to_dict, load_config
from prs.utils.fingerprint import data_fingerprint
from prs.visualization.animation import render_animation
from prs.visualization.predictions import render_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=SPLIT_NAMES)
    parser.add_argument("--scenario-id")
    parser.add_argument("--target-track-id", type=int)
    parser.add_argument("--examples", type=int, default=2, help="targets per VRU type")
    parser.add_argument("--min-speed", type=float, default=0.0, help="m/s, examples mode only")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures/predictions"))
    parser.add_argument("--checkpoint", type=Path, help="weights for a learned model")
    parser.add_argument("--animate", action="store_true", help="also write a GIF per example")
    args = parser.parse_args()
    if (args.scenario_id is None) != (args.target_track_id is None):
        parser.error("--scenario-id and --target-track-id must be given together")

    cfg = load_config(args.config)
    model_cfg = cfg.require_model()
    model = build_model(model_cfg, cfg.data).eval()
    if is_learned(model):
        if args.checkpoint is None:
            parser.error(f"model {model_cfg.name!r} is learned: pass --checkpoint")
        fingerprint = data_fingerprint(cfg.data, PREPROCESS_VERSION)
        restore_model(
            model, load_checkpoint(args.checkpoint), config_to_dict(cfg)["model"], fingerprint
        )
    if args.scenario_id is not None:
        examples = [find_sample(cfg.data, args.split, args.scenario_id, args.target_track_id)]
    else:
        examples = pick_examples(cfg.data, args.split, args.examples, args.min_speed)

    for scenario, sample in examples:
        batch = collate_samples([sample])
        with torch.no_grad():
            output = model(**model_inputs(batch))
        trajectories = output["trajectories"].numpy()
        logits = output["logits"].numpy()
        metrics = per_sample_metrics(
            trajectories, logits, batch["target_future"], batch["target_future_valid"]
        )
        ade, fde = float(metrics["ade"][0]), float(metrics["fde"][0])
        subtitle = f"{model_label(model_cfg)} · ADE {ade:.2f} m · FDE {fde:.2f} m"
        num_modes = trajectories.shape[1]
        if num_modes > 1:
            subtitle += (
                f" (top mode) · minADE@{num_modes} {float(metrics['min_ade'][0]):.2f} m · "
                f"minFDE@{num_modes} {float(metrics['min_fde'][0]):.2f} m"
            )
        probabilities = torch.softmax(output["logits"], dim=-1).numpy()[0]
        meta = sample.meta
        out = args.out_dir / args.config.stem / f"{meta.scenario_id}_{meta.target_track_id}.png"
        render_sample(
            scenario,
            sample,
            cfg.data,
            out,
            predictions=trajectories[0],
            probabilities=probabilities,
            subtitle=subtitle,
        )
        if args.animate:
            title = (
                f"{ObjectType(meta.target_type).name.lower()} track {meta.target_track_id} · "
                f"scenario {meta.scenario_id}\n{subtitle}"
            )
            render_animation(
                scenario, sample, cfg.data, out.with_suffix(".gif"), trajectories[0],
                probabilities, title,
            )  # fmt: skip
        print(
            f"{ObjectType(meta.target_type).name.lower():<10} scenario {meta.scenario_id} "
            f"track {meta.target_track_id:<5} ADE {ade:5.2f} m  FDE {fde:5.2f} m  -> {out}"
        )


if __name__ == "__main__":
    main()

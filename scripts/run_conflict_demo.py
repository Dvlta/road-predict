"""Draw conflict examples as BEV figures (design doc §22, §23).

    uv run python scripts/run_conflict_demo.py --config configs/multimodal.yaml \
        --checkpoint outputs/runs/multimodal_exp/best.pt --data-config configs/data_exp.yaml \
        --split test --num-samples 20 --only-conflicts
    uv run python scripts/run_conflict_demo.py --config configs/multimodal.yaml \
        --checkpoint outputs/runs/multimodal_exp/best.pt --data-config configs/data_exp.yaml \
        --split test --scenario-id <id> --target-track-id <id>

Runs the model over the cached split and scores every target against the SDC's logged future
with the config's `risk` rule. Each figure shows the predicted modes (conflicting ones in magenta),
the SDC's future with a mark every second, the SDC's footprint at the highest-risk mode's
closest approach and at the true closest approach, and the conflict score beside the
ground-truth label. --only-conflicts draws targets that truly conflict or that the model flags:
a mix of caught conflicts, false alarms and missed conflicts. Otherwise it draws a random
sample of the analyzed targets. --animate also writes each example as a GIF that plays the
history, the forecast and the logged future (prs.visualization.animation).

Figures go to outputs/figures/conflicts/<config>/<tier>_<split>/, named by outcome and score.
The map comes from the raw shards, so the split's raw shards must be present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from prs.data import PREPROCESS_VERSION
from prs.data.dataset import LABEL_FIELDS, CachedDataset, collate_batch
from prs.data.preprocess import build_sample
from prs.data.raw.waymo_reader import iter_scenarios
from prs.data.types import ObjectType
from prs.data.windows import make_window
from prs.metrics.multimodal import mode_probabilities
from prs.models import build_model, is_learned, model_label
from prs.risk.conflict import analyze_conflict, mode_conflicts
from prs.risk.evaluation import (
    SCORE_THRESHOLD,
    analyzed,
    ground_truth_conflicts,
    predicted_conflict_scores,
)
from prs.training.checkpoint import load_checkpoint, restore_model
from prs.training.device import select_device
from prs.training.inference import predict
from prs.utils.config import DEVICES, SPLIT_NAMES, config_to_dict, load_config
from prs.utils.fingerprint import data_fingerprint
from prs.visualization.animation import render_animation
from prs.visualization.bev import CONFLICT_COLOR, SDC_COLOR, TARGET_COLOR
from prs.visualization.conflict import ClosestApproach, draw_closest_approach, draw_time_ticks
from prs.visualization.predictions import PREDICTION_COLOR, render_sample

EGO_FIELDS = ("ego_future", "ego_future_valid", "ego_size")
OUTCOMES = ("caught", "false_alarm", "missed")  # the --only-conflicts mix, in turn


def outcome(conflict: bool, flagged: bool) -> str:
    if conflict:
        return "caught" if flagged else "missed"
    return "false_alarm" if flagged else "no_conflict"


def choose(
    labels: np.ndarray,
    flagged: np.ndarray,
    inside: np.ndarray,
    num_samples: int,
    only_conflicts: bool,
    rng: np.random.Generator,
) -> list[int]:
    """Sample indices to draw: with `only_conflicts`, caught conflicts, false alarms and missed
    conflicts in turn (each group in random order); otherwise random analyzed targets."""
    if not only_conflicts:
        pool = np.flatnonzero(inside)
        return sorted(rng.choice(pool, min(num_samples, len(pool)), replace=False).tolist())
    groups = {
        "caught": inside & labels & flagged,
        "false_alarm": inside & ~labels & flagged,
        "missed": inside & labels & ~flagged,
    }
    queues = {name: rng.permutation(np.flatnonzero(mask)).tolist() for name, mask in groups.items()}
    chosen: list[int] = []
    while len(chosen) < num_samples and any(queues.values()):
        for name in OUTCOMES:
            if queues[name] and len(chosen) < num_samples:
                chosen.append(queues[name].pop(0))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="config with a `risk` section")
    parser.add_argument("--checkpoint", type=Path, help="weights for a learned model")
    parser.add_argument("--data-config", type=Path, help="override the config's data tier")
    parser.add_argument("--split", default="test", choices=SPLIT_NAMES)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--only-conflicts", action="store_true", help="targets that conflict or are flagged"
    )
    parser.add_argument("--scenario-id")
    parser.add_argument("--target-track-id", type=int)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--animate", action="store_true", help="also write a GIF per example")
    parser.add_argument("--seed", type=int, default=0, help="which examples are drawn")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=DEVICES, help="override the config's device")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures/conflicts"))
    args = parser.parse_args()
    if (args.scenario_id is None) != (args.target_track_id is None):
        parser.error("--scenario-id and --target-track-id must be given together")

    cfg = load_config(args.config, data_config=args.data_config)
    model_cfg, risk, dt = cfg.require_model(), cfg.require_risk(), cfg.data.dt
    model = build_model(model_cfg, cfg.data)
    if is_learned(model):
        if args.checkpoint is None:
            parser.error(f"model {model_cfg.name!r} is learned: pass --checkpoint")
        fingerprint = data_fingerprint(cfg.data, PREPROCESS_VERSION)
        restore_model(
            model, load_checkpoint(args.checkpoint), config_to_dict(cfg)["model"], fingerprint
        )
    device = select_device(args.device or cfg.device)
    model = model.to(device).eval()

    dataset = CachedDataset(cfg.data, args.split, fields={*model.input_fields, *LABEL_FIELDS})
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    results = predict(model, loader, device)
    metas = dataset.metas()
    ego_future, ego_valid, ego_size = (dataset.column(name) for name in EGO_FIELDS)
    future, future_valid = results["target_future"], results["target_future_valid"]
    labels = ground_truth_conflicts(future, future_valid, ego_future, ego_valid, ego_size, risk, dt)
    scores = predicted_conflict_scores(
        results["trajectories"], results["logits"], ego_future, ego_valid, ego_size, risk, dt
    )
    inside = analyzed(np.array([m.sdc_distance_m for m in metas]), risk)
    flagged = scores >= args.threshold
    print(
        f"{int(inside.sum())} of {len(metas)} targets analyzed: "
        f"{int((inside & labels & flagged).sum())} conflicts caught, "
        f"{int((inside & labels & ~flagged).sum())} missed, "
        f"{int((inside & ~labels & flagged).sum())} false alarms"
    )

    if args.scenario_id is not None:
        chosen = [
            i for i, m in enumerate(metas)
            if m.scenario_id == args.scenario_id and m.target_track_id == args.target_track_id
        ]  # fmt: skip
        if not chosen:
            raise SystemExit(f"no {args.split} target {args.target_track_id} in {args.scenario_id}")
    else:
        rng = np.random.default_rng(args.seed)
        chosen = choose(labels, flagged, inside, args.num_samples, args.only_conflicts, rng)

    source = cfg.data.splits[args.split].source
    wanted: dict[str, set[str]] = {}
    for i in chosen:
        wanted.setdefault(metas[i].shard, set()).add(metas[i].scenario_id)
    scenarios = {}
    for shard, ids in sorted(wanted.items()):
        for scenario in iter_scenarios(cfg.data.raw_root / source / shard, ids):
            scenarios[scenario.scenario_id] = scenario

    out_dir = args.out_dir / args.config.stem / f"{cfg.tier}_{args.split}"
    for i in chosen:
        meta = metas[i]
        scenario = scenarios[meta.scenario_id]
        window = make_window(cfg.data, meta.t0)
        index = scenario.track_index(meta.target_track_id)
        sample = build_sample(scenario, index, cfg.data, window, meta.shard)
        if not np.allclose(sample.target_future, future[i], atol=1e-4):
            raise RuntimeError(f"raw and cached samples differ for {meta.scenario_id}")
        predictions = results["trajectories"][i]
        probabilities = mode_probabilities(results["logits"][i : i + 1])[0]
        result = analyze_conflict(
            predictions, probabilities, sample.ego_future, sample.ego_future_valid,
            sample.ego_size, risk, dt,
        )  # fmt: skip
        modes = mode_conflicts(
            predictions[None], sample.ego_future[None], sample.ego_future_valid[None],
            sample.ego_size[None], risk, dt,
        )  # fmt: skip
        truth = mode_conflicts(
            sample.target_future[None, None], sample.ego_future[None],
            (sample.target_future_valid & sample.ego_future_valid)[None],
            sample.ego_size[None], risk, dt,
        )  # fmt: skip
        k = result.highest_risk_mode
        risky_color = CONFLICT_COLOR if result.conflicting_modes else PREDICTION_COLOR
        predicted = bool(np.isfinite(result.min_separation_m))
        approaches = []
        if predicted:
            closest = int(modes["closest_index"][0, k])
            approaches.append(ClosestApproach(
                closest, predictions[k, closest],
                f"mode {k}: {result.min_separation_m:.1f} m at "
                f"{result.time_to_closest_approach_s:.1f} s",
                risky_color, "SDC at the predicted closest approach", text_offset=(8, 10),
            ))  # fmt: skip
        if np.isfinite(truth["min_separation_m"][0, 0]):
            closest = int(truth["closest_index"][0, 0])
            approaches.append(ClosestApproach(
                closest, sample.target_future[closest],
                f"truth: {truth['min_separation_m'][0, 0]:.1f} m at "
                f"{truth['time_to_closest_s'][0, 0]:.1f} s",
                TARGET_COLOR, "SDC at the true closest approach",
            ))  # fmt: skip

        def overlay(ax, sample=sample, path=predictions[k], predicted=predicted,
                    approaches=approaches, color=risky_color):  # fmt: skip
            draw_time_ticks(ax, sample.ego_future[:, :2], sample.ego_future_valid, dt, SDC_COLOR)
            if predicted:
                draw_time_ticks(ax, path, np.ones(len(path), bool), dt, color)
            return [
                draw_closest_approach(ax, sample.ego_future, sample.ego_size, a) for a in approaches
            ]

        verdict = outcome(bool(labels[i]), bool(flagged[i]))
        conflicting = ", ".join(str(m) for m in result.conflicting_modes) or "none"
        true_speed = truth["ego_speed_mps"][0, 0]
        subtitle = "\n".join([
            f"{model_label(model_cfg)} · conflict score {result.score:.2f} (conflicting modes: "
            f"{conflicting}) · {'flagged' if flagged[i] else 'not flagged'} at "
            f">= {args.threshold:g}",
            f"ground truth: {'conflict' if labels[i] else 'no conflict'} (SDC at "
            f"{true_speed:.1f} m/s then) · outcome: {verdict.replace('_', ' ')}",
            f"rule: closer than {risk.conflict_distance_m:g} m to the SDC footprint within "
            f"{risk.max_time_to_closest_approach_s:g} s, SDC moving >= "
            f"{risk.min_ego_speed_mps:g} m/s",
        ])  # fmt: skip
        out = (
            out_dir / f"{verdict}_{result.score:.2f}_{meta.scenario_id}_{meta.target_track_id}.png"
        )
        render_sample(
            scenario, sample, cfg.data, out, predictions=predictions,
            probabilities=probabilities, subtitle=subtitle,
            highlighted_modes=result.conflicting_modes, overlay=overlay,
        )  # fmt: skip
        if args.animate:
            target_type = ObjectType(meta.target_type).name.lower()
            title = "\n".join([
                f"{target_type} track {meta.target_track_id} · scenario {meta.scenario_id}",
                f"{model_label(model_cfg)} · conflict score {result.score:.2f} · ground truth: "
                f"{'conflict' if labels[i] else 'no conflict'} · {verdict.replace('_', ' ')}",
                f"conflict: closer than {risk.conflict_distance_m:g} m to the SDC footprint "
                f"within {risk.max_time_to_closest_approach_s:g} s, SDC moving >= "
                f"{risk.min_ego_speed_mps:g} m/s",
            ])  # fmt: skip
            render_animation(
                scenario, sample, cfg.data, out.with_suffix(".gif"), predictions, probabilities,
                title, result.conflicting_modes, approaches,
            )  # fmt: skip
        print(
            f"{verdict:<12} score {result.score:.2f}  "
            f"{ObjectType(meta.target_type).name.lower():<10} {meta.scenario_id} "
            f"track {meta.target_track_id:<5} -> {out}"
        )


if __name__ == "__main__":
    main()

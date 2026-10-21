"""Score saved predictions against ground-truth conflicts (design doc §22.6, §24.4).

    uv run python scripts/evaluate_conflicts.py --config configs/multimodal.yaml \
        --data-config configs/data_exp.yaml \
        outputs/metrics/baseline_cv/data_exp/test.json \
        outputs/metrics/multimodal/data_exp/test.json

Each metrics file (from scripts/evaluate.py) points to its model's saved predictions. Every model
is scored on the same samples, the targets within the analysis radius of the SDC, against the
same labels: the conflict rule applied to the logged target and ego futures. The rule is the
config's `risk` section; the --conflict-distance-m, --max-time-s, --min-ego-speed-mps and
--distance-mode options override it for sensitivity checks. The first file is the reference
for AP differences.

Writes outputs/conflicts/<tier>/<split>_<rule>.json and .md, and a precision-recall figure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from prs.risk.evaluation import (
    BOOTSTRAP_DRAWS,
    SCORE_THRESHOLD,
    analyzed,
    compare_conflict_scores,
    conflict_table,
    ground_truth_conflicts,
    precision_recall_at,
    precision_recall_curve,
    predicted_conflict_scores,
)
from prs.risk.inputs import load_scoring_inputs
from prs.utils.config import DISTANCE_MODES, RiskConfig, load_config
from prs.visualization.conflict import render_precision_recall

EGO_FIELDS = ("ego_future", "ego_future_valid", "ego_size")


def rule_name(risk: RiskConfig) -> str:
    return (
        f"{risk.distance_mode}_{risk.conflict_distance_m:g}m_"
        f"{risk.max_time_to_closest_approach_s:g}s_{risk.min_ego_speed_mps:g}mps"
    )


def rule_text(risk: RiskConfig) -> str:
    distance = "the ego's footprint" if risk.distance_mode == "ego_box" else "the ego's center"
    return (
        f"closer than {risk.conflict_distance_m:g} m to {distance} (less a "
        f"{risk.vru_radius_m:g} m VRU radius), closest approach within "
        f"{risk.max_time_to_closest_approach_s:g} s, ego moving at >= "
        f"{risk.min_ego_speed_mps:g} m/s then; targets within {risk.max_sdc_distance_m:g} m "
        "of the SDC at t0"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics JSON files, reference first")
    parser.add_argument("--config", required=True, type=Path, help="config with a `risk` section")
    parser.add_argument("--data-config", type=Path, help="override the config's data tier")
    parser.add_argument("--conflict-distance-m", type=float)
    parser.add_argument("--max-time-s", type=float, help="max time to closest approach")
    parser.add_argument("--min-ego-speed-mps", type=float)
    parser.add_argument("--distance-mode", choices=DISTANCE_MODES)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS, help="bootstrap resamples")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/conflicts"))
    args = parser.parse_args()

    cfg = load_config(args.config, data_config=args.data_config)
    try:
        inputs = load_scoring_inputs(args.metrics, cfg)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from None
    overrides = {
        "conflict_distance_m": args.conflict_distance_m,
        "max_time_to_closest_approach_s": args.max_time_s,
        "min_ego_speed_mps": args.min_ego_speed_mps,
        "distance_mode": args.distance_mode,
    }
    risk = dataclasses.replace(
        cfg.require_risk(), **{k: v for k, v in overrides.items() if v is not None}
    )

    split, columns, names = inputs.split, inputs.columns, inputs.models
    ego = [columns[name] for name in EGO_FIELDS]
    labels = ground_truth_conflicts(
        columns["target_future"], columns["target_future_valid"], *ego, risk, cfg.data.dt
    )
    scores = {
        name: predicted_conflict_scores(
            predictions["trajectories"], predictions["logits"], *ego, risk, cfg.data.dt
        )
        for name, predictions in inputs.predictions.items()
    }
    inside = analyzed(np.array([m.sdc_distance_m for m in inputs.metas]), risk)
    groups = inputs.groups(inside, cfg.data.target_types)
    results = {
        group: compare_conflict_scores(
            {name: s[mask] for name, s in scores.items()}, labels[mask], inputs.scenes[mask],
            args.threshold, args.draws,
        )
        for group, mask in groups.items()
    }  # fmt: skip

    stem = f"{split}_{rule_name(risk)}"
    out = args.out_dir / inputs.tier / stem
    overall = results["overall"][names[0]]
    header = (
        f"Split `{split}` of tier `{inputs.tier}`, manifest `{inputs.manifest['sha256']}`: "
        f"{int(inside.sum())} of {len(inside)} samples analyzed, {overall['positives']} "
        f"ground-truth conflicts. Conflict rule: {rule_text(risk)}. Flagged = "
        f"conflict_score >= {args.threshold:g}; precision and recall are at that threshold. "
        "Intervals are 95% scene-bootstrap intervals; `*` = the AP difference from the first "
        "model excludes zero. Open loop: the logged ego already reacted to the VRU, so these "
        "are mostly near-misses, not collisions."
    )
    markdown = header + "\n\n" + conflict_table(results) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(markdown)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "split": split,
                "tier": inputs.tier,
                "manifest": inputs.manifest,
                "risk": dataclasses.asdict(risk),
                "threshold": args.threshold,
                "bootstrap_draws": args.draws,
                "metrics_files": inputs.metrics_files,
                "num_samples": len(inside),
                "num_analyzed": int(inside.sum()),
                "groups": results,
            },
            indent=2,
        )
    )
    curves = {name: precision_recall_curve(s[inside], labels[inside]) for name, s in scores.items()}
    points = {
        name: precision_recall_at(s[inside], labels[inside], args.threshold)
        for name, s in scores.items()
    }
    figure = out.with_suffix(".png")
    if overall["positives"]:
        render_precision_recall(
            curves, points, overall["base_rate"], figure,
            f"conflict scores vs ground truth · {split} · {rule_name(risk)} · "
            f"{overall['positives']} conflicts in {overall['count']} samples",
        )  # fmt: skip
    print(markdown)
    print(
        f"written to {out.with_suffix('.md')}" + (f" and {figure}" if overall["positives"] else "")
    )


if __name__ == "__main__":
    main()

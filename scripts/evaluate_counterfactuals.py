"""Score saved predictions against synthetic crashes from counterfactual ego plans.

    uv run python scripts/evaluate_counterfactuals.py --config configs/multimodal.yaml \
        --data-config configs/data_exp.yaml \
        outputs/metrics/baseline_cv/data_exp/test.json \
        outputs/metrics/interaction/data_exp/test.json \
        outputs/metrics/multimodal/data_exp/test.json

WOMD holds no collisions, so the SDC is replayed along its logged path without yielding
(prs.risk.counterfactual): never slower than logged, accelerating at --accel up to the fastest
it drove, but stopping for other vehicles. Their logged positions come from the split's raw
shards, which must be present; reading them takes a minute or two. A crash is the logged VRU
future touching the plan's footprint while the SDC moves, within the forecast horizon. The
labels are synthetic and open loop: the VRU reacted to the real SDC, not to this plan.

Each model's saved forecast is scored against the plan in two ways:
- contact score: the probability of the predicted modes that touch the plan, which is the
  label's own rule and the headline score;
- conflict score: the config's `risk` rule, the near-miss warning of scripts/evaluate_conflicts.py.

Reported on the targets within the analysis radius:
- AP and precision/recall at --threshold per group, with scene-bootstrap intervals (the first
  file is the reference for AP differences);
- plan ranking: whether each crash's plan scores higher than the logged plan, i.e. whether the
  forecast would have picked the safer plan;
- recall by time to contact, and by whether contact falls on the logged path or on the
  extension past its end;
- for every --accel after the first, the crash count and AP.

Writes outputs/counterfactuals/<tier>/<split>_no_yield.{md,json,png}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from prs.risk.counterfactual import (
    EGO_PLAN_FIELDS,
    NO_YIELD_ACCEL_MPS2,
    TRAFFIC_GAP_M,
    contact_rule,
    split_no_yield_plans,
)
from prs.risk.evaluation import (
    BOOTSTRAP_DRAWS,
    SCORE_THRESHOLD,
    analyzed,
    compare_conflict_scores,
    conflict_table,
    ground_truth_approaches,
    precision_recall_at,
    precision_recall_curve,
    predicted_conflict_scores,
)
from prs.risk.inputs import LOGGED_FIELDS, load_scoring_inputs
from prs.utils.config import load_config
from prs.visualization.conflict import render_precision_recall

TIME_BUCKETS_S = ((0.0, 2.0), (2.0, 4.0), (4.0, np.inf))  # time to contact, (low, high]
SCORE_KINDS = ("contact", "conflict")


def bucket_name(low: float, high: float) -> str:
    return f"> {low:g} s" if np.isinf(high) else f"{low:g}–{high:g} s"


def share(mask: np.ndarray, of: np.ndarray) -> float | None:
    return float(mask[of].mean()) if of.any() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics JSON files, reference first")
    parser.add_argument("--config", required=True, type=Path, help="config with a `risk` section")
    parser.add_argument("--data-config", type=Path, help="override the config's data tier")
    parser.add_argument(
        "--accel", type=float, nargs="+", default=[NO_YIELD_ACCEL_MPS2, 1.0, 3.0],
        help="no-yield accelerations in m/s^2; the first is the main plan, the rest sensitivity",
    )  # fmt: skip
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS, help="bootstrap resamples")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/counterfactuals"))
    args = parser.parse_args()

    cfg = load_config(args.config, data_config=args.data_config)
    fields = (*LOGGED_FIELDS, *EGO_PLAN_FIELDS, "origin_global", "heading_global")
    try:
        inputs = load_scoring_inputs(args.metrics, cfg, fields=tuple(dict.fromkeys(fields)))
    except ValueError as error:
        raise SystemExit(f"error: {error}") from None
    risk, dt, columns = cfg.require_risk(), cfg.data.dt, inputs.columns
    rules = {"contact": contact_rule(risk, cfg.data.future_seconds), "conflict": risk}
    target = (columns["target_future"], columns["target_future_valid"])
    ego_size = columns["ego_size"]
    inside = analyzed(np.array([m.sdc_distance_m for m in inputs.metas]), risk)
    groups, scenes = inputs.groups(inside, cfg.data.target_types), inputs.scenes

    def scores_under(plan: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
        return {
            f"{model} · {kind}": predicted_conflict_scores(
                predictions["trajectories"], predictions["logits"], plan, valid, ego_size,
                rules[kind], dt,
            )
            for kind in SCORE_KINDS
            for model, predictions in inputs.predictions.items()
        }  # fmt: skip

    logged = (columns["ego_future"], columns["ego_future_valid"])
    logged_contacts = ground_truth_approaches(*target, *logged, ego_size, rules["contact"], dt)
    logged_scores = scores_under(*logged)
    runs = {}
    all_plans = split_no_yield_plans(
        cfg.data, inputs.split, inputs.metas, columns, dt, args.accel, inside
    )
    for accel, (plans, valid, extended, held_from) in all_plans.items():
        truth = ground_truth_approaches(*target, plans, valid, ego_size, rules["contact"], dt)
        index = truth["closest_index"]
        moved = np.hypot(*(plans[..., :2] - logged[0][..., :2]).transpose(2, 0, 1))
        runs[accel] = {
            "labels": truth["conflict"],
            "time_to_contact_s": truth["time_to_closest_s"],
            "on_extension": extended[np.arange(len(index)), index],
            "differs": (np.where(logged[1], moved, 0.0) > 1.0).any(axis=1),
            "held": held_from >= 0,
            "scores": scores_under(plans, valid),
        }

    main_accel = args.accel[0]
    main = runs[main_accel]
    labels, scores = main["labels"], main["scores"]
    results = {
        group: compare_conflict_scores(
            {name: s[mask] for name, s in scores.items()}, labels[mask], scenes[mask],
            args.threshold, args.draws,
        )
        for group, mask in groups.items()
    }  # fmt: skip

    crashes = labels & inside
    time, on_extension = main["time_to_contact_s"], main["on_extension"]
    where = {"logged path": crashes & ~on_extension, "extension": crashes & on_extension}
    when = {bucket_name(lo, hi): crashes & (time > lo) & (time <= hi) for lo, hi in TIME_BUCKETS_S}
    analysis = {}
    for name, s in scores.items():
        flagged = s >= args.threshold
        analysis[name] = {
            "plan_scored_higher": share(s > logged_scores[name], crashes),
            "plan_scored_equal": share(s == logged_scores[name], crashes),
            "plan_scored_lower": share(s < logged_scores[name], crashes),
            "recall_by_time": {k: share(flagged, m) for k, m in when.items()},
            "recall_by_location": {k: share(flagged, m) for k, m in where.items()},
            "median_time_to_contact_flagged_s": (
                float(np.median(time[crashes & flagged])) if (crashes & flagged).any() else None
            ),
        }
    sensitivity = {
        accel: {
            "crashes": int((run["labels"] & inside).sum()),
            "on_extension": int((run["labels"] & inside & run["on_extension"]).sum()),
            "overall": compare_conflict_scores(
                {name: s[inside] for name, s in run["scores"].items()}, run["labels"][inside],
                scenes[inside], args.threshold, args.draws,
            ),
        }
        for accel, run in runs.items()
        if accel != main_accel
    }  # fmt: skip

    out = args.out_dir / inputs.tier / f"{inputs.split}_no_yield"
    markdown = _report(
        inputs, risk, args.threshold, main_accel, inside, main, logged_contacts, results,
        analysis, sensitivity, where, when,
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(markdown)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "split": inputs.split,
                "tier": inputs.tier,
                "manifest": inputs.manifest,
                "risk": dataclasses.asdict(risk),
                "plan": "no_yield",
                "accel_mps2": main_accel,
                "horizon_s": cfg.data.future_seconds,
                "threshold": args.threshold,
                "bootstrap_draws": args.draws,
                "metrics_files": inputs.metrics_files,
                "num_samples": len(inside),
                "num_analyzed": int(inside.sum()),
                "plans_differing_from_log": int((main["differs"] & inside).sum()),
                "plans_stopped_for_a_vehicle": int((main["held"] & inside).sum()),
                "logged_plan_contacts": int((logged_contacts["conflict"] & inside).sum()),
                "crashes": int(crashes.sum()),
                "crashes_on_extension": int(where["extension"].sum()),
                "crashes_by_time": {k: int(m.sum()) for k, m in when.items()},
                "groups": results,
                "per_score": analysis,
                "sensitivity": {str(k): v for k, v in sensitivity.items()},
            },
            indent=2,
        )
    )
    figure = out.with_suffix(".png")
    if crashes.any():
        contact_scores = {n: s[inside] for n, s in scores.items() if n.endswith("· contact")}
        render_precision_recall(
            {n: precision_recall_curve(s, labels[inside]) for n, s in contact_scores.items()},
            {n: precision_recall_at(s, labels[inside], args.threshold)
             for n, s in contact_scores.items()},
            float(crashes.sum() / inside.sum()), figure,
            f"contact scores vs synthetic crashes · {inputs.split} · no-yield plan at "
            f"{main_accel:g} m/s² · {int(crashes.sum())} crashes in {int(inside.sum())} samples",
        )  # fmt: skip
    print(markdown)
    print(f"written to {out.with_suffix('.md')}" + (f" and {figure}" if crashes.any() else ""))


def _report(
    inputs, risk, threshold, accel, inside, main, logged_contacts, results, analysis, sensitivity,
    where, when,
) -> str:  # fmt: skip
    crashes = main["labels"] & inside
    header = (
        f"Split `{inputs.split}` of tier `{inputs.tier}`, manifest "
        f"`{inputs.manifest['sha256']}`: {int(inside.sum())} of {len(inside)} samples analyzed "
        f"(targets within {risk.max_sdc_distance_m:g} m of the SDC at t0). **Synthetic, open "
        f"loop:** the SDC drives its logged path without yielding (never slower than logged, "
        f"accelerating at {accel:g} m/s² up to the fastest it drove, stopping before its front "
        f"comes within {TRAFFIC_GAP_M:g} m of another vehicle); the plan differs from the log by "
        f"more than 1 m for {int((main['differs'] & inside).sum())} targets and stops for a "
        f"vehicle in {int((main['held'] & inside).sum())}. A crash is "
        f"the logged VRU future touching the plan's footprint (less a {risk.vru_radius_m:g} m "
        f"VRU radius) while the SDC moves at >= {risk.min_ego_speed_mps:g} m/s, within the "
        f"forecast horizon: {int(crashes.sum())} crashes, {int(where['extension'].sum())} of "
        f"them past the end of the logged path. The logged plans give "
        f"{int((logged_contacts['conflict'] & inside).sum())}. The VRU reacted to the real SDC, "
        f"not to this plan, so a crash means its real path meets the plan, not a simulated "
        f"collision.\n\n"
        f"Scores under the plan: `contact` = probability of the predicted modes that touch it "
        f"(the label's rule); `conflict` = the near-miss rule (closer than "
        f"{risk.conflict_distance_m:g} m within {risk.max_time_to_closest_approach_s:g} s). "
        f"Flagged = score >= {threshold:g}. Intervals are 95% scene-bootstrap intervals; `*` = "
        f"the AP difference from the first row excludes zero."
    )
    when_cols = " | ".join(f"{k} ({int(m.sum())})" for k, m in when.items())
    where_cols = " | ".join(f"{k} ({int(m.sum())})" for k, m in where.items())
    lines = [
        f"| Score | Plan scored above the logged plan | Equal | Below | Recall {when_cols} | "
        f"Recall {where_cols} | Median time to contact, flagged |",
        "|---|---:|---:|---:|" + "---:|" * (len(when) + len(where)) + "---:|",
    ]
    for name, a in analysis.items():
        cells = [_pct(a[f"plan_scored_{k}"]) for k in ("higher", "equal", "lower")]
        cells += [_pct(v) for v in a["recall_by_time"].values()]
        cells += [_pct(v) for v in a["recall_by_location"].values()]
        median = a["median_time_to_contact_flagged_s"]
        cells.append("—" if median is None else f"{median:.1f} s")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    ranking = (
        "Per crash: does the forecast score the no-yield plan above the logged plan (would it "
        "have picked the safer plan)? Recall at the threshold by time to contact and by where "
        "contact falls.\n\n" + "\n".join(lines)
    )
    sections = [header, conflict_table(results, positives="Crashes"), ranking]
    if sensitivity:
        accels = list(sensitivity)
        rows = [
            "| Score | " + " | ".join(
                f"{a:g} m/s² ({s['crashes']} crashes, {s['on_extension']} on the extension)"
                for a, s in sensitivity.items()
            ) + " |",
            "|---|" + "---:|" * len(accels),
        ]  # fmt: skip
        for name in analysis:
            cells = [_ap(sensitivity[a]["overall"][name]) for a in accels]
            rows.append(f"| {name} | " + " | ".join(cells) + " |")
        sections.append("Sensitivity to the plan's acceleration, overall AP:\n\n" + "\n".join(rows))
    return "\n\n".join(sections) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _ap(summary: dict) -> str:
    value, interval = summary["average_precision"], summary["average_precision_ci95"]
    if value is None:
        return "—"
    return f"{value:.3f}" + ("" if interval is None else f" ({interval[0]:.3f}–{interval[1]:.3f})")


if __name__ == "__main__":
    main()

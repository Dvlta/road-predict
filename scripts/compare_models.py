"""Paired comparison of two models on the same sample set (design doc §24.3).

    uv run python scripts/compare_models.py \
        outputs/metrics/baseline_cv/data_exp/test.json outputs/metrics/lstm/data_exp/test.json

Reports B - A per group with a scene-clustered 95% interval; `*` marks differences whose
interval excludes zero. Refuses files computed on different sample sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prs.metrics.breakdowns import group_masks
from prs.metrics.paired import align, paired_comparison, paired_table
from prs.metrics.tables import ManifestMismatchError, check_comparable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path, help="metrics JSON of the reference model")
    parser.add_argument("b", type=Path, help="metrics JSON of the model being tested")
    parser.add_argument("--out", type=Path, help="also write the Markdown here")
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in (args.a, args.b)]
    try:
        check_comparable(reports)
    except ManifestMismatchError as error:
        raise SystemExit(f"error: {error}") from None
    thresholds = {json.dumps(r["evaluation"], sort_keys=True) for r in reports}
    if len(thresholds) > 1:
        raise SystemExit("error: the two files use different evaluation thresholds")
    per_sample = [dict(np.load(r["per_sample"])) for r in reports]
    a, b = per_sample[0], align(per_sample[0], per_sample[1])

    evaluation = reports[0]["evaluation"]
    class_names = list(reports[0]["by_class"])
    masks = group_masks(
        a["target_type"],
        a["current_speed_mps"],
        class_names,
        evaluation["stationary_speed_threshold_mps"],
    )
    comparison = paired_comparison(a, b, masks, a["scenario_id"], evaluation["miss_threshold_m"])
    name_a, name_b = (r["model"] for r in reports)
    first = reports[0]
    markdown = (
        f"A = `{name_a}` ({args.a}), B = `{name_b}` ({args.b}). Split `{first['split']}` of "
        f"tier `{first['tier']}`, {first['manifest']['num_samples']} samples, manifest "
        f"`{first['manifest']['sha256']}`. B − A cells are mean paired differences ± 95% "
        "intervals clustered by scene; negative is better for B; `*` = interval excludes "
        "zero.\n\n" + paired_table(comparison, "A", "B") + "\n"
    )
    print(markdown)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()

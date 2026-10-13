"""Evaluate a forecaster on one split and write a metrics JSON (design doc §16, §24).

    uv run python scripts/evaluate.py --config configs/baseline_cv.yaml --split val

Reads samples from the processed cache (build it with scripts/preprocess_waymo.py). Writes
outputs/metrics/<config name>/<split>.json, which records the exact sample set (manifest
hash), the data fingerprint and the git commit, so results are reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prs.data import PREPROCESS_VERSION
from prs.data.dataset import LABEL_FIELDS, CachedDataset, collate_batch
from prs.data.splits import shard_paths
from prs.metrics.breakdowns import breakdown, per_sample_metrics
from prs.metrics.manifest import manifest_hash, sample_key
from prs.models import build_model
from prs.models.base import ForecastModel, model_inputs
from prs.utils.config import SPLIT_NAMES, load_config
from prs.utils.fingerprint import data_fingerprint
from prs.utils.provenance import git_commit


def predict(model: ForecastModel, loader: DataLoader) -> dict[str, np.ndarray]:
    """Model outputs and labels for every sample, in dataset order."""
    collected: dict[str, list[np.ndarray]] = {
        "trajectories": [], "logits": [], "target_future": [], "target_future_valid": [],
        "target_type": [],
    }  # fmt: skip
    with torch.no_grad():
        for batch in loader:
            output = model(**model_inputs(batch))
            for key in ("trajectories", "logits"):
                collected[key].append(output[key].cpu().numpy())
            for key in LABEL_FIELDS:
                collected[key].append(batch[key].numpy())
    return {key: np.concatenate(values) for key, values in collected.items()}


def print_report(report: dict) -> None:
    manifest = report["manifest"]
    print(f"\n{report['model']} · {report['split']} · {manifest['num_samples']} samples · "
          f"manifest {manifest['sha256']}")  # fmt: skip
    print(f"{'group':<26}{'count':>7}{'ADE (m)':>10}{'FDE (m)':>10}{'miss rate':>11}")
    rows = [("overall", report["overall"])]
    for name, summary in report["by_class"].items():
        rows.append((name, summary))
        rows += [
            (f"  {key}", value)
            for key, value in report["by_class_and_speed"].items()
            if key.startswith(f"{name}/")
        ]
    for label, s in rows:
        if s["count"] == 0:
            print(f"{label:<26}{0:>7}")
            continue
        print(f"{label:<26}{s['count']:>7}{s['ade']:>10.2f}{s['fde']:>10.2f}"
              f"{s['miss_rate']:>11.1%}")  # fmt: skip
    sanity = report["sanity"]
    print(f"first-step (+0.1 s) error: mean {sanity['first_step_error_mean_m']:.3f} m, "
          f"median {sanity['first_step_error_median_m']:.3f} m")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=SPLIT_NAMES)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg, eval_cfg = cfg.require_model(), cfg.require_evaluation()
    model = build_model(model_cfg, cfg.data).eval()

    dataset = CachedDataset(cfg.data, args.split, fields={*model.input_fields, *LABEL_FIELDS})
    if len(dataset) == 0:
        raise SystemExit(f"no samples in split {args.split!r}")
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    results = predict(model, loader)

    metas = dataset.metas()
    per_sample = per_sample_metrics(
        results["trajectories"], results["logits"],
        results["target_future"], results["target_future_valid"],
    )  # fmt: skip
    metrics = breakdown(
        per_sample,
        results["target_type"],
        np.array([m.current_speed_mps for m in metas]),
        cfg.data.target_types,
        eval_cfg,
        model.num_modes,
    )
    keys = [sample_key(m) for m in metas]
    report = {
        "model": model_cfg.name,
        "config": str(args.config),
        "split": args.split,
        "shards": [p.name for p in shard_paths(cfg.data, args.split)],
        "data_fingerprint": data_fingerprint(cfg.data, PREPROCESS_VERSION),
        "git_commit": git_commit(),
        "manifest": {"num_samples": len(keys), "sha256": manifest_hash(keys)},
        "evaluation": {
            "miss_threshold_m": eval_cfg.miss_threshold_m,
            "stationary_speed_threshold_mps": eval_cfg.stationary_speed_threshold_mps,
        },
        **metrics,
        "filter_counts": dataset.filter_counts(),
    }
    out = args.out_dir / args.config.stem / f"{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

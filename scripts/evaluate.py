"""Evaluate a forecaster on one split and write a metrics JSON (design doc §16, §24).

    uv run python scripts/evaluate.py --config configs/baseline_cv.yaml --split val
    uv run python scripts/evaluate.py --config configs/baseline_cv.yaml \
        --data-config configs/data_exp.yaml --split test

Reads samples from the processed cache (build it with scripts/preprocess_waymo.py). Writes
outputs/metrics/<config name>/<data tier>/<split>.json, which records the exact sample set
(manifest hash; the key list is saved under outputs/manifests/), the data fingerprint and the
git commit, so results are reproducible.
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
from prs.metrics.manifest import save_manifest
from prs.metrics.tables import breakdown_table
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
    print(f"\n{report['model']}")
    print(breakdown_table([report]))
    sanity, manifest = report["sanity"], report["manifest"]
    print(
        f"first-step (+0.1 s) error: mean {sanity['first_step_error_mean_m']:.3f} m, "
        f"median {sanity['first_step_error_median_m']:.3f} m · "
        f"{manifest['num_samples']} samples, manifest {manifest['sha256']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-config", type=Path, help="override the config's data tier")
    parser.add_argument("--split", default="val", choices=SPLIT_NAMES)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/manifests"))
    args = parser.parse_args()

    cfg = load_config(args.config, data_config=args.data_config)
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
        clusters=np.array([m.scenario_id for m in metas]),
    )
    shards = [p.name for p in shard_paths(cfg.data, args.split)]
    fingerprint = data_fingerprint(cfg.data, PREPROCESS_VERSION)
    report = {
        "model": model_cfg.name,
        "model_info": {
            "uses_map": "map_polylines" in model.input_fields,
            "uses_neighbors": "neighbors" in model.input_fields,
            "num_modes": model.num_modes,
        },
        "config": str(args.config),
        "tier": cfg.tier,
        "split": args.split,
        "shards": shards,
        "data_fingerprint": fingerprint,
        "git_commit": git_commit(),
        "manifest": save_manifest(
            args.manifest_dir / cfg.tier,
            metas,
            args.split,
            {"tier": cfg.tier, "data_fingerprint": fingerprint, "shards": shards},
        ),
        "evaluation": {
            "miss_threshold_m": eval_cfg.miss_threshold_m,
            "stationary_speed_threshold_mps": eval_cfg.stationary_speed_threshold_mps,
        },
        **metrics,
        "filter_counts": dataset.filter_counts(),
    }
    out = args.out_dir / args.config.stem / cfg.tier / f"{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

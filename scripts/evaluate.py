"""Evaluate a forecaster on one split and write a metrics JSON (design doc §16, §24).

    uv run python scripts/evaluate.py --config configs/baseline_cv.yaml --split val
    uv run python scripts/evaluate.py --config configs/baseline_cv.yaml \
        --data-config configs/data_exp.yaml --split test
    uv run python scripts/evaluate.py --config configs/lstm.yaml \
        --checkpoint outputs/runs/<run>/best.pt --data-config configs/data_exp.yaml --split test

Reads samples from the processed cache (build it with scripts/preprocess_waymo.py). Writes
outputs/metrics/<config name>/<data tier>/<split>.json, which records the exact sample set
(manifest hash; the key list is saved under outputs/manifests/), the data fingerprint, the git
commit and the checkpoint, so results are reproducible. Per-sample errors go to
<split>.per_sample.npz beside it, for paired comparisons (scripts/compare_models.py).

--shuffle-map SEED gives every sample another sample's map (design doc §14.3 check 6) and writes
<split>_shuffled_map.json instead; compare it with the normal file to see how much the model
relies on the map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from prs.data import PREPROCESS_VERSION
from prs.data.dataset import LABEL_FIELDS, CachedDataset, ShuffledMapDataset, collate_batch
from prs.data.splits import shard_paths
from prs.metrics.breakdowns import breakdown, per_sample_metrics
from prs.metrics.manifest import sample_key, save_manifest
from prs.metrics.tables import breakdown_table
from prs.models import build_model, is_learned, model_label
from prs.training.checkpoint import load_checkpoint, restore_model
from prs.training.device import select_device
from prs.training.inference import predict
from prs.utils.config import DEVICES, SPLIT_NAMES, config_to_dict, load_config
from prs.utils.fingerprint import data_fingerprint
from prs.utils.provenance import git_commit


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
    parser.add_argument("--checkpoint", type=Path, help="weights for a learned model")
    parser.add_argument("--device", choices=DEVICES, help="override the config's device")
    parser.add_argument(
        "--shuffle-map", type=int, metavar="SEED", help="evaluate with maps shuffled across samples"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, data_config=args.data_config)
    model_cfg, eval_cfg = cfg.require_model(), cfg.require_evaluation()
    model = build_model(model_cfg, cfg.data)
    fingerprint = data_fingerprint(cfg.data, PREPROCESS_VERSION)
    checkpoint_info = None
    if is_learned(model):
        if args.checkpoint is None:
            raise SystemExit(f"model {model_cfg.name!r} is learned: pass --checkpoint")
        checkpoint = load_checkpoint(args.checkpoint)
        restore_model(model, checkpoint, config_to_dict(cfg)["model"], fingerprint)
        checkpoint_info = {
            "path": str(args.checkpoint),
            "epoch": checkpoint["epoch"],
            "git_commit": checkpoint["git_commit"],
            "train_metrics": checkpoint["metrics"],
            "train_shards": checkpoint["config"]["data"]["splits"]["train"],
        }
    elif args.checkpoint is not None:
        raise SystemExit(f"model {model_cfg.name!r} has no weights; drop --checkpoint")
    device = select_device(args.device or cfg.device)
    model = model.to(device).eval()

    dataset = CachedDataset(cfg.data, args.split, fields={*model.input_fields, *LABEL_FIELDS})
    if len(dataset) == 0:
        raise SystemExit(f"no samples in split {args.split!r}")
    stem = args.split
    if args.shuffle_map is not None:
        if "map_polylines" not in model.input_fields:
            raise SystemExit(f"model {model_cfg.name!r} does not read the map; drop --shuffle-map")
        dataset = ShuffledMapDataset(dataset, args.shuffle_map)
        stem = f"{args.split}_shuffled_map"
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    results = predict(model, loader, device)
    if not np.array_equal(results["index"], np.arange(len(dataset))):
        raise RuntimeError("predictions are not in dataset order")

    metas = dataset.metas()
    speeds = np.array([m.current_speed_mps for m in metas])
    scenes = np.array([m.scenario_id for m in metas])
    per_sample = per_sample_metrics(
        results["trajectories"], results["logits"],
        results["target_future"], results["target_future_valid"],
    )  # fmt: skip
    metrics = breakdown(
        per_sample,
        results["target_type"],
        speeds,
        cfg.data.target_types,
        eval_cfg,
        model.num_modes,
        clusters=scenes,
    )
    shards = [p.name for p in shard_paths(cfg.data, args.split)]
    out = args.out_dir / args.config.stem / cfg.tier / f"{stem}.json"
    per_sample_path = out.with_name(f"{stem}.per_sample.npz")
    shuffled = args.shuffle_map is not None
    report = {
        "model": model_label(model_cfg) + (" (shuffled map)" if shuffled else ""),
        "map_shuffle_seed": args.shuffle_map,
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
        "device": str(device),
        "checkpoint": checkpoint_info,
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
        "per_sample": str(per_sample_path),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        per_sample_path,
        keys=np.array([sample_key(m) for m in metas]),
        scenario_id=scenes,
        target_type=results["target_type"],
        current_speed_mps=speeds,
        **{name: per_sample[name] for name in ("ade", "fde", "min_ade", "min_fde")},
    )
    out.write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

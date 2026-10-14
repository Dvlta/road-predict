"""Train a forecaster (design doc §17, §25).

    uv run python scripts/train.py --config configs/lstm.yaml
    uv run python scripts/train.py --config configs/lstm.yaml --data-config configs/data_exp.yaml
    uv run python scripts/train.py --resume outputs/runs/<name> [--epochs 40]

    # Overfit check (§14.3 check 5): train and validate on the first 100 training samples
    uv run python scripts/train.py --config configs/lstm.yaml --subset 100 --epochs 300

Writes outputs/runs/<name>/ (config.yaml, metrics.json, train.log, tensorboard/, best.pt,
last.pt). Resuming reads the run's own config.yaml, so it continues the identical setup;
--epochs may raise the epoch count. View curves with `uv run tensorboard --logdir outputs/runs`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import yaml
from torch.utils.data import Subset

from prs.data import PREPROCESS_VERSION
from prs.data.dataset import LABEL_FIELDS, CachedDataset
from prs.data.splits import shard_paths
from prs.metrics.manifest import save_manifest
from prs.models import build_model, is_learned
from prs.training.device import device_info, select_device
from prs.training.logging import RunLogger
from prs.training.trainer import Trainer, check_run_dir
from prs.utils.config import DEVICES, config_from_dict, config_to_dict, load_config
from prs.utils.fingerprint import data_fingerprint
from prs.utils.provenance import git_commit
from prs.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue this run")
    parser.add_argument("--data-config", type=Path, help="override the config's data tier")
    parser.add_argument("--name", help="run directory name (default: <config>_<tier>_<time>)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/manifests"))
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument("--device", choices=DEVICES, help="override the config's device")
    parser.add_argument(
        "--subset", type=int, help="overfit check: train and validate on the first N samples"
    )
    args = parser.parse_args()

    if args.resume is not None:
        if args.data_config or args.name or args.subset or args.device:
            parser.error("--resume reuses the run's settings; only --epochs may change")
        run_dir = args.resume
        check_run_dir(run_dir, resume=True)
        cfg = config_from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
        run_info = json.loads((run_dir / "metrics.json").read_text())["run"]
        resumed = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit()}
        run_info["resumed"] = [*run_info.get("resumed", []), resumed]
    else:
        cfg = load_config(args.config, data_config=args.data_config)
        if args.device is not None:
            cfg = dataclasses.replace(cfg, device=args.device)
        name = args.name or f"{args.config.stem}_{cfg.tier}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = args.out_dir / name
        check_run_dir(run_dir, resume=False)
        run_info = {
            "name": name,
            "config_file": str(args.config),
            "tier": cfg.tier,
            "subset": args.subset,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    if args.epochs is not None:
        cfg = dataclasses.replace(
            cfg, training=dataclasses.replace(cfg.require_training(), epochs=args.epochs)
        )
    training = cfg.require_training()
    tier = run_info["tier"]  # a resumed config is inline, so the tier comes from the run record

    logger = RunLogger(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
    set_seed(cfg.seed)
    model = build_model(cfg.require_model(), cfg.data)
    if not is_learned(model):
        raise SystemExit(f"model {cfg.require_model().name!r} has nothing to train")
    device = select_device(cfg.device)
    fingerprint = data_fingerprint(cfg.data, PREPROCESS_VERSION)

    fields = {*model.input_fields, *LABEL_FIELDS}
    train_set = CachedDataset(cfg.data, "train", fields=fields)
    train_metas = train_set.metas()
    if run_info["subset"]:
        n = run_info["subset"]
        if n > len(train_set):
            raise SystemExit(f"--subset {n} exceeds the {len(train_set)} training samples")
        train_set, train_metas = Subset(train_set, range(n)), train_metas[:n]
        val_set, val_split, val_metas = train_set, "train", train_metas
    else:
        val_set = CachedDataset(cfg.data, "val", fields=fields)
        val_split, val_metas = "val", val_set.metas()

    manifests = tier if run_info["subset"] is None else f"{tier}_subset"
    details = {"tier": tier, "data_fingerprint": fingerprint}
    provenance = {
        "run": run_info,
        "seed": cfg.seed,
        "git_commit": git_commit(),
        **device_info(device),
        "data_fingerprint": fingerprint,
        "train_shards": [p.name for p in shard_paths(cfg.data, "train")],
        "train_manifest": save_manifest(
            args.manifest_dir / manifests, train_metas, "train", details
        ),
        "val_split": val_split,
        "val_manifest": save_manifest(args.manifest_dir / manifests, val_metas, val_split, details),
    }
    trainer = Trainer(model, cfg, train_set, val_set, device, logger, fingerprint, provenance)
    if args.resume is not None:
        trainer.resume()
    else:
        parameters = sum(p.numel() for p in model.parameters())
        logger.info(
            f"run {run_dir} | {cfg.require_model().name}, {parameters:,} parameters | "
            f"device {device} | tier {tier}: {len(train_set)} train / {len(val_set)} "
            f"{val_split} samples | git {provenance['git_commit']}"
        )
    best = trainer.fit(training.epochs)
    logger.info(
        f"done. best epoch {best['epoch']}: val ADE {best['val_ade']:.3f} m, "
        f"FDE {best['val_fde']:.3f} m -> {run_dir / 'best.pt'}"
    )
    logger.close()


if __name__ == "__main__":
    main()

"""Build the processed-sample cache for the shards in a data config (design doc §11).

    uv run python scripts/preprocess_waymo.py --config configs/data_dev.yaml

Shards whose cache already exists for the current settings are skipped. Changing any
preprocessing setting changes the fingerprint, and therefore the cache directory.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from prs.data import PREPROCESS_VERSION
from prs.data.cache import build_shard_cache, cache_dir, shard_cache_dir
from prs.data.splits import shard_paths
from prs.utils.config import SPLIT_NAMES, load_config
from prs.utils.fingerprint import data_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_NAMES), choices=SPLIT_NAMES)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing caches")
    parser.add_argument(
        "--delete-raw-after", action="store_true", help="delete each raw shard once cached"
    )
    args = parser.parse_args()

    cfg = load_config(args.config).data
    jobs = []
    for split in args.splits:
        for path in shard_paths(cfg, split):
            if (shard_cache_dir(cfg, path.name) / "meta.json").exists() and not args.overwrite:
                print(f"cached   {path.name}")
                continue
            if not path.exists():
                raise SystemExit(f"missing raw shard {path} (run scripts/download_shards.py)")
            jobs.append((path, split))

    print(f"cache: {cache_dir(cfg)}  (fingerprint {data_fingerprint(cfg, PREPROCESS_VERSION)})")
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs) or 1))) as pool:
        futures = [(path, pool.submit(build_shard_cache, path, cfg, split)) for path, split in jobs]
        for path, future in futures:
            info = future.result()
            print(
                f"built    {path.name}: {info['num_samples']} samples in {info['build_seconds']} s"
                f" | over neighbor cap {_pct(info['neighbor_cap_hit_fraction'])}"
                f" | over map cap {_pct(info['map_cap_hit_fraction'])}"
            )
            if args.delete_raw_after:
                path.unlink()
                print(f"deleted  {path}")


def _pct(fraction: float | None) -> str:
    return "n/a" if fraction is None else f"{100 * fraction:.1f}%"


if __name__ == "__main__":
    main()

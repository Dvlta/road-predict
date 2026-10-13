"""Download the WOMD shards listed in a data config and verify their MD5 (design doc §10.1).

Files already present with a matching MD5 are skipped.

    uv run python scripts/download_shards.py --config configs/data_dev.yaml
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from prs.data.splits import gcs_url, shard_filename
from prs.utils.config import SPLIT_NAMES, load_config


def remote_md5(url: str) -> str:
    result = subprocess.run(
        ["gcloud", "storage", "objects", "describe", url, "--format=value(md5_hash)"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def local_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_NAMES), choices=SPLIT_NAMES)
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()

    cfg = load_config(args.config).data
    # Sliced parallel downloads failed hash validation on macOS (design doc §10.1).
    env = {**os.environ, "CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD": "0"}
    for split in args.splits:
        spec = cfg.splits[split]
        for index in spec.shards:
            url = gcs_url(cfg.dataset_version, spec.source, index)
            dest = cfg.raw_root / spec.source / shard_filename(spec.source, index)
            if args.dry_run:
                print(f"gcloud storage cp {url} {dest}")
                continue
            expected = remote_md5(url)
            if dest.exists() and local_md5(dest) == expected:
                print(f"ok       {dest}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["gcloud", "storage", "cp", url, str(dest)], check=True, env=env)
            if local_md5(dest) != expected:
                dest.unlink()
                sys.exit(f"MD5 mismatch for {dest}; deleted it")
            print(f"fetched  {dest}")


if __name__ == "__main__":
    main()

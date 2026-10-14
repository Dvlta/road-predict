"""Shard file names, local paths and GCS URLs for project splits (design doc §10)."""

from __future__ import annotations

from pathlib import Path

from prs.utils.config import SHARD_COUNTS, DataConfig


def shard_filename(source: str, index: int) -> str:
    return f"{source}.tfrecord-{index:05d}-of-{SHARD_COUNTS[source]:05d}"


def shard_paths(cfg: DataConfig, split: str) -> list[Path]:
    """Local raw-shard paths for a project split ("train", "val" or "test")."""
    spec = cfg.splits[split]
    return [cfg.raw_root / spec.source / shard_filename(spec.source, i) for i in spec.shards]


def gcs_url(dataset_version: str, source: str, index: int) -> str:
    """e.g. v1.3.1 -> gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/..."""
    bucket = "waymo_open_dataset_motion_v_" + dataset_version.removeprefix("v").replace(".", "_")
    return f"gs://{bucket}/uncompressed/scenario/{source}/{shard_filename(source, index)}"

"""Checkpoints (design doc §25). Loading one that does not match the current model or data
settings fails loudly (§31)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """A checkpoint that cannot be used with the current model or data."""


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    data_fingerprint: str,
    git_commit: str | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write atomically, so an interrupted save never leaves a corrupt file. `epoch` is the
    number of completed epochs."""
    payload = {
        "format_version": FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
        "data_fingerprint": data_fingerprint,
        "git_commit": git_commit,
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CheckpointError(f"no checkpoint at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(
            f"{path}: format version {checkpoint.get('format_version')}, expected {FORMAT_VERSION}"
        )
    return checkpoint


def restore_model(
    model: nn.Module,
    checkpoint: dict[str, Any],
    model_config: dict[str, Any],
    data_fingerprint: str,
) -> None:
    """Load weights after checking the checkpoint was made for this exact model configuration
    and these data settings."""
    saved = checkpoint["config"].get("model")
    if saved != model_config:
        raise CheckpointError(f"checkpoint model config {saved} != current {model_config}")
    if checkpoint["data_fingerprint"] != data_fingerprint:
        raise CheckpointError(
            f"checkpoint was trained on data fingerprint {checkpoint['data_fingerprint']}, "
            f"current data settings give {data_fingerprint}"
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise CheckpointError(f"checkpoint does not match the model: {error}") from None

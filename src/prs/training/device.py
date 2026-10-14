"""Device selection (design doc §30): cuda, then mps, then cpu, with an explicit override."""

from __future__ import annotations

import os

import torch


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device(name)


def device_info(device: torch.device) -> dict[str, str | bool | None]:
    """What a run records about its hardware (design doc §30, §32)."""
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "torch_version": torch.__version__,
    }


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work, so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

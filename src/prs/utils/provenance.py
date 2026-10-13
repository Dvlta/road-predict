"""Run provenance (design doc §32)."""

from __future__ import annotations

import subprocess


def git_commit() -> str | None:
    """Current commit hash, with "-dirty" if there are uncommitted changes; None outside git
    or before the first commit."""
    try:
        sha = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

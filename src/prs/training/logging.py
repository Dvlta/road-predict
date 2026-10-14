"""Run directory logging (design doc §26): train.log, metrics.json and TensorBoard.

outputs/runs/<name>/
├── config.yaml      resolved config, including the data section and shard lists
├── metrics.json     run provenance, per-epoch history, best epoch
├── train.log
├── tensorboard/
├── best.pt
└── last.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._log = (run_dir / "train.log").open("a")
        self._writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def info(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def scalars(self, values: dict[str, float], step: int, prefix: str = "") -> None:
        for name, value in values.items():
            self._writer.add_scalar(f"{prefix}{name}", value, step)
        self._writer.flush()

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        path = self.run_dir / "metrics.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(metrics, indent=2))
        tmp.replace(path)

    def close(self) -> None:
        self._writer.close()
        self._log.close()

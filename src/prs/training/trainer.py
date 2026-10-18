"""Training loop (design doc §17, §25).

AdamW with gradient clipping; a finite-loss check every step (abort and dump the batch);
validation after every epoch; the best checkpoint chosen by validation ADE; `last.pt` written
every epoch, so an interrupted run resumes where it stopped.

Each epoch reseeds the RNGs and the shuffle order from (seed, epoch), so on the CPU a resumed
run matches an uninterrupted one exactly. MPS and CUDA runs are best-effort reproducible.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from prs.data.dataset import collate_batch
from prs.losses.multimodal import winner_takes_all
from prs.losses.trajectory import masked_smooth_l1
from prs.metrics.breakdowns import per_sample_metrics, summarize
from prs.metrics.multimodal import mode_statistics
from prs.models.base import ForecastModel, model_inputs
from prs.training.checkpoint import load_checkpoint, restore_model, save_checkpoint
from prs.training.device import synchronize
from prs.training.inference import predict
from prs.training.logging import RunLogger
from prs.utils.config import Config, TrainingConfig, config_to_dict
from prs.utils.seed import set_seed

STEP_LOG_EVERY = 50
# Seconds a loader may wait for one batch from its workers. A worker that fails to deliver
# (for example, out of file descriptors) would otherwise leave the run waiting forever; a
# batch normally takes well under a second, even from a cold network volume.
LOADER_TIMEOUT_S = 600


class NonFiniteLossError(RuntimeError):
    """The loss or gradient norm became NaN or inf (design doc §14.3 check 9)."""


def best_metric(num_modes: int) -> str:
    """The validation metric that picks the best epoch (lower is better): ADE for one mode,
    minADE@K for several, since only the best mode of a multimodal forecast is meant to be
    right."""
    return "val_ade" if num_modes == 1 else "val_min_ade"


def trajectory_loss(
    trajectories: Tensor, logits: Tensor, future: Tensor, valid: Tensor, training: TrainingConfig
) -> Tensor:
    """SmoothL1 for one mode (§17); winner-takes-all for several (§20)."""
    if trajectories.shape[1] == 1:
        return masked_smooth_l1(trajectories[:, 0], future, valid, training.smooth_l1_beta)
    loss, _ = winner_takes_all(
        trajectories, logits, future, valid, training.smooth_l1_beta, training.lambda_cls,
        training.wta_selection,
    )  # fmt: skip
    return loss


def batch_loss(
    model: ForecastModel, batch: dict[str, Tensor], device: torch.device, training: TrainingConfig
) -> Tensor:
    output = model(**model_inputs(batch, device))
    return trajectory_loss(
        output["trajectories"],
        output["logits"],
        batch["target_future"].to(device),
        batch["target_future_valid"].to(device),
        training,
    )


class Trainer:
    def __init__(
        self,
        model: ForecastModel,
        cfg: Config,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
        logger: RunLogger,
        data_fingerprint: str,
        provenance: dict[str, Any],
    ) -> None:
        self.cfg, self.training = cfg, cfg.require_training()
        self.eval_cfg = cfg.require_evaluation()
        self.model = model.to(device)
        self.best_metric = best_metric(model.num_modes)
        self.device, self.logger = device, logger
        self.run_dir = logger.run_dir
        self.data_fingerprint, self.provenance = data_fingerprint, provenance
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
        )
        # The shuffle order comes only from this generator, reseeded every epoch. Each loader
        # also gets a private generator for its own draws (worker seeds, made whenever it starts
        # its workers), so neither the order nor the global RNG (dropout) depends on whether a
        # loader is fresh (a resumed run) or reuses persistent workers, or on the worker count.
        self.generator = torch.Generator()
        workers = self.training.num_workers
        loader_args = {
            "batch_size": self.training.batch_size,
            "collate_fn": collate_batch,
            "num_workers": workers,
            "persistent_workers": workers > 0,
            "timeout": LOADER_TIMEOUT_S if workers > 0 else 0,  # allowed only with workers
        }
        self.train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset, generator=self.generator),
            generator=torch.Generator(),
            **loader_args,
        )
        self.val_loader = DataLoader(
            val_dataset, shuffle=False, generator=torch.Generator(), **loader_args
        )
        self.epoch = 0  # completed epochs
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.global_step = 0

    # ------------------------------------------------------------------ resume

    def resume(self) -> None:
        """Continue from the run directory's last.pt."""
        checkpoint = load_checkpoint(self.run_dir / "last.pt")
        restore_model(
            self.model, checkpoint, config_to_dict(self.cfg)["model"], self.data_fingerprint
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.history = checkpoint["history"]
        self.best = checkpoint["best"]
        self.global_step = checkpoint["global_step"]
        self.logger.info(f"resumed from {self.run_dir / 'last.pt'} after epoch {self.epoch}")

    # ------------------------------------------------------------------ loop

    def fit(self, epochs: int) -> dict[str, Any] | None:
        """Train until `epochs` epochs are complete; returns the best epoch's record."""
        while self.epoch < epochs:
            set_seed(self.cfg.seed + self.epoch)
            self.generator.manual_seed(self.cfg.seed + self.epoch)
            record = {"epoch": self.epoch + 1, **self.train_one_epoch(), **self.validate()}
            if self.model.num_modes > 1:
                record["collapsed_modes"] = self._collapsed_modes(record)
            self.epoch += 1
            self.history.append(record)
            metric = self.best_metric
            improved = self.best is None or record[metric] < self.best[metric]
            if improved:
                self.best = record
                self._save("best.pt", record)
            self._save("last.pt", record)
            self.logger.scalars(
                {k: v for k, v in record.items() if isinstance(v, float)},
                self.epoch,
                prefix="epoch/",
            )
            if "val_win_rates" in record:
                self.logger.scalars(
                    {f"val_win_rate/{k}": r for k, r in enumerate(record["val_win_rates"])},
                    self.epoch,
                    prefix="epoch/",
                )
            self.logger.write_metrics(self.metrics())
            self.logger.info(
                f"epoch {self.epoch}/{epochs}  train loss {record['train_loss']:.4f}  "
                f"val loss {record['val_loss']:.4f}  val ADE {record['val_ade']:.3f} m  "
                f"FDE {record['val_fde']:.3f} m  miss {record['val_miss_rate']:.1%}  "
                + self._modes_summary(record)
                + f"({record['train_seconds']:.0f} s, {record['train_samples_per_s']:.0f} "
                f"samples/s){'  *best' if improved else ''}"
            )
            if record.get("collapsed_modes"):
                self.logger.info(
                    f"mode collapse: modes {record['collapsed_modes']} win less than "
                    f"{self.training.collapse_min_win_rate:.1%} of validation samples (§20)"
                )
        return self.best

    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        clip = self.training.gradient_clip_norm
        total_loss, total_samples = 0.0, 0
        start = time.perf_counter()
        progress = tqdm(
            self.train_loader, desc=f"epoch {self.epoch + 1}", leave=False, disable=None
        )
        for batch in progress:
            loss = batch_loss(self.model, batch, self.device, self.training)
            loss_value = self._check_finite(loss, "loss", batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self._check_finite(grad_norm, "gradient norm", batch)
            self.optimizer.step()

            n = len(batch["target_type"])
            total_loss += loss_value * n
            total_samples += n
            self.global_step += 1
            if self.global_step % STEP_LOG_EVERY == 0:
                self.logger.scalars({"train_loss": loss_value}, self.global_step, prefix="step/")
                progress.set_postfix(loss=f"{loss_value:.3f}")
        synchronize(self.device)
        seconds = time.perf_counter() - start
        return {
            "train_loss": total_loss / total_samples,
            "train_seconds": seconds,
            "train_samples_per_s": total_samples / seconds,
        }

    def validate(self) -> dict[str, float]:
        results = predict(self.model, self.val_loader, self.device)
        loss = trajectory_loss(
            torch.from_numpy(results["trajectories"]),
            torch.from_numpy(results["logits"]),
            torch.from_numpy(results["target_future"]),
            torch.from_numpy(results["target_future_valid"]),
            self.training,
        )
        per_sample = per_sample_metrics(
            results["trajectories"],
            results["logits"],
            results["target_future"],
            results["target_future_valid"],
        )
        summary = summarize(
            per_sample,
            np.ones(len(results["index"]), dtype=bool),
            self.eval_cfg.miss_threshold_m,
            self.model.num_modes,
        )
        record = {
            "val_loss": float(loss),
            "val_ade": summary["ade"],
            "val_fde": summary["fde"],
            "val_miss_rate": summary["miss_rate"],
        }
        if self.model.num_modes > 1:
            modes = mode_statistics(
                results["trajectories"],
                results["logits"],
                results["target_future"],
                results["target_future_valid"],
                self.training.wta_selection,
            )
            record |= {
                "val_min_ade": summary["min_ade"],
                "val_min_fde": summary["min_fde"],
                "val_win_rates": modes["win_rates"],
                "val_endpoint_spread_m": modes["endpoint_spread_m"],
                "val_mode_entropy": modes["entropy"],
                "val_top1_accuracy": modes["top1_accuracy"],
                "val_winner_nll": modes["winner_nll"],
            }
        return record

    # ------------------------------------------------------------------ helpers

    def metrics(self) -> dict[str, Any]:
        return {
            **self.provenance,
            "best_metric": self.best_metric,
            "best": self.best,
            "history": self.history,
        }

    def _collapsed_modes(self, record: dict[str, Any]) -> list[int]:
        """Modes that win less often than collapse_min_win_rate, checked once
        collapse_check_after_epochs epochs are done (§20)."""
        if record["epoch"] < self.training.collapse_check_after_epochs:
            return []
        threshold = self.training.collapse_min_win_rate
        return [k for k, rate in enumerate(record["val_win_rates"]) if rate < threshold]

    @staticmethod
    def _modes_summary(record: dict[str, Any]) -> str:
        if "val_win_rates" not in record:
            return ""
        rates = record["val_win_rates"]
        return (
            f"minADE {record['val_min_ade']:.3f} m  minFDE {record['val_min_fde']:.3f} m  "
            f"spread {record['val_endpoint_spread_m']:.2f} m  "
            f"entropy {record['val_mode_entropy']:.2f}  wins {min(rates):.1%}-{max(rates):.1%}  "
            f"top-1 hit {record['val_top1_accuracy']:.1%}  NLL {record['val_winner_nll']:.3f}  "
        )

    def _save(self, name: str, record: dict[str, Any]) -> None:
        save_checkpoint(
            self.run_dir / name,
            self.model,
            self.optimizer,
            self.epoch,
            config_to_dict(self.cfg),
            record,
            self.data_fingerprint,
            self.provenance.get("git_commit"),
            extra={"history": self.history, "best": self.best, "global_step": self.global_step},
        )

    def _check_finite(self, value: Tensor, what: str, batch: dict[str, Tensor]) -> float:
        number = float(value.item())
        if not math.isfinite(number):
            dump = self.run_dir / "nonfinite_batch.pt"
            torch.save({key: tensor.cpu() for key, tensor in batch.items()}, dump)
            raise NonFiniteLossError(
                f"{what} is {number} at epoch {self.epoch + 1}, step {self.global_step + 1}; "
                f"batch (with dataset indices) saved to {dump}"
            )
        return number


def check_run_dir(run_dir: Path, resume: bool) -> None:
    if resume and not (run_dir / "last.pt").exists():
        raise SystemExit(f"cannot resume: no last.pt in {run_dir}")
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} already exists; pass --resume to continue it")

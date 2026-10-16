"""Training loop, checkpoints and resume (design doc §14.3 checks 5 and 9, §25)."""

import dataclasses
import time

import numpy as np
import pytest
import torch
from conftest import REPO_ROOT
from torch.utils.data import Dataset, Subset

from prs.data import PREPROCESS_VERSION
from prs.data.cache import cache_dir
from prs.data.dataset import LABEL_FIELDS, CachedDataset
from prs.models import build_model
from prs.training.checkpoint import (
    CheckpointError,
    load_checkpoint,
    restore_model,
    save_checkpoint,
)
from prs.training.logging import RunLogger
from prs.training.trainer import NonFiniteLossError, Trainer
from prs.utils.config import config_to_dict, load_config
from prs.utils.fingerprint import data_fingerprint
from prs.utils.seed import set_seed

H, T = 21, 60
FINGERPRINT = "synthetic"


class LinearTracks(Dataset):
    """Targets moving at constant velocity, so the future is a learnable function of the past."""

    def __init__(self, n: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        velocity = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)
        past = (np.arange(H) - (H - 1)) * 0.1
        future = np.arange(1, T + 1) * 0.1
        self.history = np.zeros((n, H, 9), np.float32)
        self.history[..., :2] = past[None, :, None] * velocity[:, None]
        self.history[..., 2:4] = velocity[:, None]
        self.history[..., 5] = 1.0
        self.history[..., 8] = 1.0
        self.future = (future[None, :, None] * velocity[:, None]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, i: int) -> dict:
        return {
            "target_history": self.history[i],
            "target_history_valid": np.ones(H, bool),
            "target_type": np.int64(2),
            "target_future": self.future[i],
            "target_future_valid": np.ones(T, bool),
            "index": i,
        }


class StalledTracks(LinearTracks):
    """A loader worker that never delivers a batch, like one out of file descriptors."""

    def __getitem__(self, i: int) -> dict:
        time.sleep(3600)
        return super().__getitem__(i)


def _cfg(**training):
    cfg = load_config(REPO_ROOT / "configs" / "lstm.yaml")
    small = dataclasses.replace(cfg.require_model(), hidden_dim=32, decoder_hidden_dim=64)
    defaults = {"batch_size": 32, "epochs": 2, "num_workers": 0, "learning_rate": 1e-3}
    return dataclasses.replace(
        cfg,
        model=small,
        device="cpu",
        training=dataclasses.replace(cfg.require_training(), **{**defaults, **training}),
    )


def _trainer(cfg, run_dir, train, val=None, fingerprint=FINGERPRINT):
    set_seed(cfg.seed)
    model = build_model(cfg.require_model(), cfg.data)
    return Trainer(
        model, cfg, train, val if val is not None else train, torch.device("cpu"),
        RunLogger(run_dir), fingerprint, {"git_commit": None},
    )  # fmt: skip


def test_training_loss_decreases(tmp_path):
    trainer = _trainer(_cfg(epochs=12), tmp_path, LinearTracks(256))
    trainer.fit(12)
    losses = [record["train_loss"] for record in trainer.history]
    assert losses[-1] < 0.5 * losses[0]
    assert (tmp_path / "best.pt").exists() and (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_matches_uninterrupted_run(tmp_path, workers):
    """With loader workers too: a resumed loader starts fresh, while an uninterrupted run
    reuses its persistent workers, and the shuffle order must not depend on that."""
    cfg, data = _cfg(num_workers=workers), LinearTracks(96)
    straight = _trainer(cfg, tmp_path / "straight", data)
    straight.fit(2)

    first = _trainer(cfg, tmp_path / "resumed", data)
    first.fit(1)
    second = _trainer(cfg, tmp_path / "resumed", data)
    second.resume()
    assert second.epoch == 1 and len(second.history) == 1
    second.fit(2)

    for name, value in straight.model.state_dict().items():
        torch.testing.assert_close(second.model.state_dict()[name], value, rtol=0, atol=0)
    assert second.history == [
        {**r, "train_seconds": s["train_seconds"], "train_samples_per_s": s["train_samples_per_s"]}
        for r, s in zip(straight.history, second.history, strict=True)
    ]


def test_worker_count_does_not_change_training(tmp_path):
    """So `train.py --num-workers` may differ between machines, even when resuming a run."""
    data, runs = LinearTracks(96), []
    for workers in (0, 2):
        trainer = _trainer(_cfg(num_workers=workers), tmp_path / str(workers), data)
        trainer.fit(2)
        runs.append(trainer.model.state_dict())
    for name, value in runs[0].items():
        torch.testing.assert_close(runs[1][name], value, rtol=0, atol=0)


def test_stalled_loader_worker_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("prs.training.trainer.LOADER_TIMEOUT_S", 2)
    trainer = _trainer(_cfg(num_workers=1), tmp_path, StalledTracks(32))
    with pytest.raises(RuntimeError, match="timed out"):
        trainer.fit(1)


def test_nonfinite_loss_aborts_and_dumps_batch(tmp_path):
    trainer = _trainer(_cfg(), tmp_path, LinearTracks(64))
    with torch.no_grad():
        trainer.model.decoder[-1].bias.fill_(float("nan"))
    with pytest.raises(NonFiniteLossError, match="loss is nan"):
        trainer.fit(1)
    dump = torch.load(tmp_path / "nonfinite_batch.pt")
    assert "index" in dump and len(dump["index"]) == 32


def test_checkpoint_round_trip(tmp_path):
    cfg = _cfg()
    model = build_model(cfg.require_model(), cfg.data)
    model_dict = config_to_dict(cfg)["model"]
    save_checkpoint(tmp_path / "x.pt", model, None, 3, config_to_dict(cfg), {}, FINGERPRINT, None)
    restored = build_model(cfg.require_model(), cfg.data)
    checkpoint = load_checkpoint(tmp_path / "x.pt")
    restore_model(restored, checkpoint, model_dict, FINGERPRINT)
    assert checkpoint["epoch"] == 3
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_checkpoint_mismatch_fails_loudly(tmp_path):
    cfg = _cfg()
    model = build_model(cfg.require_model(), cfg.data)
    save_checkpoint(tmp_path / "x.pt", model, None, 1, config_to_dict(cfg), {}, FINGERPRINT, None)
    checkpoint = load_checkpoint(tmp_path / "x.pt")

    wider_cfg = dataclasses.replace(cfg.require_model(), hidden_dim=48)
    wider = build_model(wider_cfg, cfg.data)
    wider_dict = config_to_dict(dataclasses.replace(cfg, model=wider_cfg))["model"]
    with pytest.raises(CheckpointError, match="model config"):
        restore_model(wider, checkpoint, wider_dict, FINGERPRINT)
    with pytest.raises(CheckpointError, match="does not match the model"):  # weights only
        restore_model(wider, checkpoint, config_to_dict(cfg)["model"], FINGERPRINT)
    with pytest.raises(CheckpointError, match="data fingerprint"):
        restore_model(model, checkpoint, config_to_dict(cfg)["model"], "other")
    with pytest.raises(CheckpointError, match="no checkpoint"):
        load_checkpoint(tmp_path / "missing.pt")


@pytest.mark.slow
@pytest.mark.data
# Full-batch steps per model: about twice the steps needed to first reach 0.1 m on the CPU (≈ 200
# for lstm_map, whose map encoder makes each step ≈ 0.7 s).
@pytest.mark.parametrize(("config_name", "steps"), [("lstm.yaml", 1000), ("lstm_map.yaml", 400)])
def test_overfit_tiny(tmp_path, config_name, steps):
    """Check 5: each full-size learned model memorizes 100 real training samples (dev cache).
    Full-batch steps at a raised learning rate, without dropout, keep this short on the CPU."""
    cfg = _cfg(batch_size=100, learning_rate=3e-3)
    full_size = load_config(REPO_ROOT / "configs" / config_name).require_model()
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(full_size, dropout=0.0))
    if not cache_dir(cfg.data).exists():
        pytest.skip("dev cache not built")
    model = build_model(cfg.require_model(), cfg.data)
    fields = {*model.input_fields, *LABEL_FIELDS}
    subset = Subset(CachedDataset(cfg.data, "train", fields=fields), range(100))
    fingerprint = data_fingerprint(cfg.data, PREPROCESS_VERSION)
    trainer = _trainer(cfg, tmp_path, subset, fingerprint=fingerprint)
    best = trainer.fit(steps)
    assert best["val_ade"] < 0.1, best

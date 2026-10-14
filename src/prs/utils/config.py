"""Typed YAML configuration (design doc §13).

Unknown keys, missing keys, wrong types and windows that do not fit the scene all fail loudly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from prs.data.types import OBJECT_TYPE_BY_NAME

FRAME_HEADING_SOURCES = ("box", "velocity", "auto")
MAP_DTYPES = ("float32", "float16")
MODEL_NAMES = ("constant_velocity",)
SPLIT_NAMES = ("train", "val", "test")
# Official source split each project split must come from (design doc §10.2).
SPLIT_SOURCES = {"train": "training", "val": "validation", "test": "validation"}
# Shards per official split; verified against the v1.3.1 bucket on 2026-10-12 (design doc §4.1).
SHARD_COUNTS = {"training": 1000, "validation": 150}


class ConfigError(ValueError):
    """Invalid configuration."""


@dataclass(frozen=True)
class SplitSpec:
    source: str  # official split: "training" | "validation"
    shards: tuple[int, ...]


@dataclass(frozen=True)
class DataConfig:
    dataset_version: str
    raw_root: Path
    cache_root: Path
    sample_hz: int
    scene_num_frames: int
    max_timestep_error_s: float
    current_index: int
    train_current_indices: tuple[int, ...]
    history_seconds: float
    future_seconds: float
    target_types: tuple[str, ...]
    min_history_valid_fraction: float
    min_future_valid_fraction: float
    require_final_future_valid: bool
    frame_heading_source: str
    velocity_heading_min_speed_mps: float
    max_neighbors: int
    neighbor_radius_m: float
    always_include_sdc: bool
    map_radius_m: float
    map_resample_spacing_m: float
    max_map_polylines: int
    max_points_per_polyline: int
    map_dtype: str
    splits: dict[str, SplitSpec]

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_hz

    @property
    def history_steps(self) -> int:
        """Past frames before t0, excluding t0."""
        return _whole_steps(self.history_seconds, self.sample_hz, "history_seconds")

    @property
    def history_len(self) -> int:
        """H: history tensor length, including t0."""
        return self.history_steps + 1

    @property
    def future_steps(self) -> int:
        """T: future tensor length."""
        return _whole_steps(self.future_seconds, self.sample_hz, "future_seconds")

    @property
    def target_type_ids(self) -> tuple[int, ...]:
        return tuple(int(OBJECT_TYPE_BY_NAME[name]) for name in self.target_types)

    def __post_init__(self) -> None:
        _require(self.sample_hz > 0, "sample_hz must be positive")
        _require(self.history_seconds > 0, "history_seconds must be positive")
        _require(self.future_seconds > 0, "future_seconds must be positive")
        for t0 in (self.current_index, *self.train_current_indices):
            first, last = t0 - self.history_steps, t0 + self.future_steps
            _require(
                first >= 0 and last <= self.scene_num_frames - 1,
                f"window around t0={t0} needs frames {first}..{last}, but scenes have frames "
                f"0..{self.scene_num_frames - 1} (design doc §5.1)",
            )
        _require(len(self.train_current_indices) > 0, "train_current_indices must not be empty")
        _require(len(self.target_types) > 0, "target_types must not be empty")
        for name in self.target_types:
            _require(name in OBJECT_TYPE_BY_NAME, f"unknown target type {name!r}")
        for name in ("min_history_valid_fraction", "min_future_valid_fraction"):
            _require(0.0 <= getattr(self, name) <= 1.0, f"{name} must be in [0, 1]")
        _require(
            self.frame_heading_source in FRAME_HEADING_SOURCES,
            f"frame_heading_source must be one of {FRAME_HEADING_SOURCES}",
        )
        for name in (
            "neighbor_radius_m",
            "map_radius_m",
            "map_resample_spacing_m",
            "max_neighbors",
            "max_map_polylines",
            "max_points_per_polyline",
        ):
            _require(getattr(self, name) > 0, f"{name} must be positive")
        _require(self.velocity_heading_min_speed_mps >= 0, "velocity_heading_min_speed_mps < 0")
        _require(self.max_timestep_error_s >= 0, "max_timestep_error_s must be >= 0")
        _require(self.map_dtype in MAP_DTYPES, f"map_dtype must be one of {MAP_DTYPES}")
        self._validate_splits()

    def _validate_splits(self) -> None:
        _require(
            set(self.splits) == set(SPLIT_NAMES),
            f"splits must define exactly {SPLIT_NAMES}, got {sorted(self.splits)}",
        )
        used: dict[tuple[str, int], str] = {}
        for name, spec in self.splits.items():
            expected = SPLIT_SOURCES[name]
            _require(spec.source == expected, f"split {name!r} must use source {expected!r}")
            _require(len(spec.shards) > 0, f"split {name!r} has no shards")
            _require(len(set(spec.shards)) == len(spec.shards), f"split {name!r} repeats shards")
            for index in spec.shards:
                _require(
                    0 <= index < SHARD_COUNTS[spec.source],
                    f"split {name!r}: shard {index} out of range for {spec.source}",
                )
                other = used.setdefault((spec.source, index), name)
                _require(
                    other == name,
                    f"{spec.source} shard {index} is in splits {other!r} and {name!r} "
                    "(splits must be disjoint)",
                )


@dataclass(frozen=True)
class ModelConfig:
    name: str

    def __post_init__(self) -> None:
        _require(self.name in MODEL_NAMES, f"model.name must be one of {MODEL_NAMES}")


@dataclass(frozen=True)
class EvaluationConfig:
    miss_threshold_m: float
    stationary_speed_threshold_mps: float

    def __post_init__(self) -> None:
        _require(self.miss_threshold_m > 0, "evaluation.miss_threshold_m must be positive")
        _require(
            self.stationary_speed_threshold_mps >= 0,
            "evaluation.stationary_speed_threshold_mps must be >= 0",
        )


@dataclass(frozen=True)
class Config:
    data: DataConfig
    seed: int = 42
    model: ModelConfig | None = None
    evaluation: EvaluationConfig | None = None
    data_source: Path | None = None  # file the `data` section came from (names the data tier)

    @property
    def tier(self) -> str:
        """Short name of the data tier, e.g. "data_dev", taken from the data config file."""
        return self.data_source.stem if self.data_source is not None else "inline"

    def require_model(self) -> ModelConfig:
        if self.model is None:
            raise ConfigError("config has no `model` section")
        return self.model

    def require_evaluation(self) -> EvaluationConfig:
        if self.evaluation is None:
            raise ConfigError("config has no `evaluation` section")
        return self.evaluation


def load_config(path: str | Path, data_config: str | Path | None = None) -> Config:
    """Load a config. `data_config: <file>` pulls the `data` section from another config,
    resolved relative to this file, so a model config can switch data tiers in one line.
    The `data_config` argument overrides that reference (e.g. `--data-config` on the CLI)."""
    path = Path(path)
    raw = _load_mapping(path)
    data_source = path
    if "data_config" in raw or data_config is not None:
        referenced_name = raw.pop("data_config", None)
        if "data" in raw:
            raise ConfigError(f"{path}: give either `data` or `data_config`, not both")
        if data_config is not None:
            data_source = Path(data_config)
        else:
            data_source = path.parent / _as_str(referenced_name, "data_config")
        referenced = _load_mapping(data_source)
        if "data" not in referenced:
            raise ConfigError(f"{data_source}: no `data` section")
        raw["data"] = referenced["data"]
    return dataclasses.replace(config_from_dict(raw), data_source=data_source)


def config_from_dict(raw: dict[str, Any]) -> Config:
    allowed = {"data", "seed", "model", "evaluation"}
    _check_keys(raw, allowed, required={"data"}, where="config")
    return Config(
        data=data_config_from_dict(raw["data"]),
        seed=_as_int(raw.get("seed", 42), "seed"),
        model=_section(raw, "model", ModelConfig, {"name": _as_str}),
        evaluation=_section(
            raw,
            "evaluation",
            EvaluationConfig,
            {"miss_threshold_m": _as_float, "stationary_speed_threshold_mps": _as_float},
        ),
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return raw


def _section(
    raw: dict[str, Any], name: str, cls: type, converters: dict[str, Callable[[Any, str], Any]]
) -> Any:
    """Build an optional, fully specified config section."""
    if name not in raw:
        return None
    section = raw[name]
    if not isinstance(section, dict):
        raise ConfigError(f"{name}: expected a mapping")
    _check_keys(section, set(converters), required=set(converters), where=name)
    return cls(
        **{key: convert(section[key], f"{name}.{key}") for key, convert in converters.items()}
    )


def data_config_from_dict(raw: dict[str, Any]) -> DataConfig:
    if not isinstance(raw, dict):
        raise ConfigError("data: expected a mapping")
    names = {f.name for f in fields(DataConfig)}
    _check_keys(raw, names, required=names, where="data")
    converters: dict[str, Callable[[Any, str], Any]] = {
        "dataset_version": _as_str,
        "raw_root": _as_path,
        "cache_root": _as_path,
        "sample_hz": _as_int,
        "scene_num_frames": _as_int,
        "max_timestep_error_s": _as_float,
        "current_index": _as_int,
        "train_current_indices": _as_int_tuple,
        "history_seconds": _as_float,
        "future_seconds": _as_float,
        "target_types": _as_str_tuple,
        "min_history_valid_fraction": _as_float,
        "min_future_valid_fraction": _as_float,
        "require_final_future_valid": _as_bool,
        "frame_heading_source": _as_str,
        "velocity_heading_min_speed_mps": _as_float,
        "max_neighbors": _as_int,
        "neighbor_radius_m": _as_float,
        "always_include_sdc": _as_bool,
        "map_radius_m": _as_float,
        "map_resample_spacing_m": _as_float,
        "max_map_polylines": _as_int,
        "max_points_per_polyline": _as_int,
        "map_dtype": _as_str,
        "splits": _as_splits,
    }
    return DataConfig(**{name: converters[name](raw[name], f"data.{name}") for name in names})


def _whole_steps(seconds: float, hz: int, name: str) -> int:
    steps = seconds * hz
    if abs(steps - round(steps)) > 1e-9:
        raise ConfigError(f"{name}={seconds} is not a whole number of frames at {hz} Hz")
    return int(round(steps))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_keys(raw: dict[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    missing = sorted(required - set(raw))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    if missing:
        raise ConfigError(f"{where}: missing keys {missing}")


def _as_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name}: expected a string, got {value!r}")
    return value


def _as_path(value: Any, name: str) -> Path:
    return Path(_as_str(value, name))


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name}: expected an integer, got {value!r}")
    return value


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name}: expected a number, got {value!r}")
    return float(value)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name}: expected true/false, got {value!r}")
    return value


def _as_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{name}: expected a list, got {value!r}")
    return tuple(_as_int(v, f"{name}[{i}]") for i, v in enumerate(value))


def _as_str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{name}: expected a list, got {value!r}")
    return tuple(_as_str(v, f"{name}[{i}]") for i, v in enumerate(value))


def _as_splits(value: Any, name: str) -> dict[str, SplitSpec]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: expected a mapping, got {value!r}")
    splits = {}
    for split, spec in value.items():
        where = f"{name}.{split}"
        if not isinstance(spec, dict):
            raise ConfigError(f"{where}: expected a mapping, got {spec!r}")
        _check_keys(spec, {"source", "shards"}, required={"source", "shards"}, where=where)
        splits[split] = SplitSpec(
            source=_as_str(spec["source"], f"{where}.source"),
            shards=_as_shards(spec["shards"], f"{where}.shards"),
        )
    return splits


def _as_shards(value: Any, name: str) -> tuple[int, ...]:
    """A list of shard indices, or a string of inclusive ranges such as "0-14,20"."""
    if not isinstance(value, str):
        return _as_int_tuple(value, name)
    shards: list[int] = []
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) not in (1, 2) or not all(b.strip().isdigit() for b in bounds):
            raise ConfigError(f"{name}: cannot parse {part!r} in {value!r}")
        first, last = int(bounds[0]), int(bounds[-1])
        if last < first:
            raise ConfigError(f"{name}: empty range {part!r}")
        shards.extend(range(first, last + 1))
    return tuple(shards)

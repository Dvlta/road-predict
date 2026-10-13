"""ProcessedSample schema validation (design doc §7, §31)."""

import dataclasses

import numpy as np
import pytest
from conftest import CYCLIST, PED

from prs.data.preprocess import build_sample
from prs.data.targets import build_target_sample
from prs.data.windows import make_window


@pytest.fixture
def sample(scenario, data_cfg):
    return build_target_sample(scenario, CYCLIST, data_cfg, make_window(data_cfg))


def test_valid_sample_passes(sample):
    sample.validate()


def test_validate_catches_bad_shape(sample):
    bad = dataclasses.replace(sample, target_future=sample.target_future[:59])
    with pytest.raises(ValueError, match="target_future_valid: expected shape"):
        bad.validate()


def test_validate_catches_bad_dtype(sample):
    bad = dataclasses.replace(sample, target_future=sample.target_future.astype(np.float64))
    with pytest.raises(ValueError, match="expected dtype float32"):
        bad.validate()


def test_validate_catches_nonzero_padding(sample):
    history = sample.target_history.copy()
    history[5, 0] = -1.0  # frame 5 is invalid for the cyclist
    with pytest.raises(ValueError, match="padded"):
        dataclasses.replace(sample, target_history=history).validate()


def test_validate_catches_nan(scenario, data_cfg):
    sample = build_target_sample(scenario, PED, data_cfg, make_window(data_cfg))
    future = sample.target_future.copy()
    future[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        dataclasses.replace(sample, target_future=future).validate()


def test_validate_context_all_or_nothing(sample):
    partial = dataclasses.replace(sample, neighbors=np.zeros((16, 21, 9), np.float32))
    with pytest.raises(ValueError, match="all set or all None"):
        partial.validate()


def test_validate_catches_bad_neighbor_padding(scenario, data_cfg):
    sample = build_sample(scenario, PED, data_cfg, make_window(data_cfg))
    types = sample.neighbor_types.copy()
    types[-1] = 1  # a padded slot claiming to be a vehicle
    with pytest.raises(ValueError, match="padded neighbor slots"):
        dataclasses.replace(sample, neighbor_types=types).validate()


def test_validate_catches_bad_map_types(scenario, data_cfg):
    sample = build_sample(scenario, PED, data_cfg, make_window(data_cfg))
    types = sample.map_types.copy()
    types[0] = 0  # a real chunk marked as padding
    with pytest.raises(ValueError, match="PAD"):
        dataclasses.replace(sample, map_types=types).validate()

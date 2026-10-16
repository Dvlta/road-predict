"""The shared model interface and the per-model guardrails (design doc §14, §40):
leakage, padding invariance and CPU/MPS parity."""

import inspect

import numpy as np
import pytest
import torch
from conftest import REPO_ROOT

from prs.data.map_types import NUM_MAP_TYPES, MapType
from prs.data.types import F_AGENT, ObjectType
from prs.models import build_model
from prs.models.base import MODEL_INPUT_FIELDS, ForecastModel, model_inputs
from prs.models.constant_velocity import ConstantVelocity
from prs.models.interaction import InteractionForecaster
from prs.models.lstm_forecaster import LSTMForecaster
from prs.models.lstm_map_forecaster import LSTMMapForecaster
from prs.utils.config import MODEL_NAMES, load_config

ALL_MODELS = (
    ForecastModel, ConstantVelocity, LSTMForecaster, LSTMMapForecaster, InteractionForecaster
)  # fmt: skip
INTERACTION_CONFIGS = (
    "interaction_history.yaml", "interaction_map.yaml", "interaction_neighbors.yaml",
    "interaction.yaml",
)  # fmt: skip
LEARNED_CONFIGS = ("lstm.yaml", "lstm_map.yaml", *INTERACTION_CONFIGS)
MODEL_CONFIGS = ("baseline_cv.yaml", *LEARNED_CONFIGS)
H, T, N, M, P = 21, 60, 6, 6, 20


def test_model_forward_signature_excludes_future():
    for cls in ALL_MODELS:
        params = [p for p in inspect.signature(cls.forward).parameters if p != "self"]
        assert tuple(params) == MODEL_INPUT_FIELDS, cls.__name__
    assert not any("future" in name or "ego" in name for name in MODEL_INPUT_FIELDS)


def test_model_inputs_drops_everything_else():
    batch = {
        "target_history": np.zeros((2, 21, 9), np.float32),
        "target_history_valid": np.ones((2, 21), bool),
        "target_type": np.array([2, 3]),
        "target_future": np.ones((2, 60, 2), np.float32),
        "ego_future": np.ones((2, 60, 5), np.float32),
        "meta": ["a", "b"],
    }
    inputs = model_inputs(batch)
    assert set(inputs) == set(MODEL_INPUT_FIELDS)
    assert inputs["neighbors"] is None and inputs["map_polylines"] is None
    assert inputs["target_history"].shape == (2, 21, 9)


def _model(config_name: str, seed: int = 0) -> ForecastModel:
    cfg = load_config(REPO_ROOT / "configs" / config_name)
    torch.manual_seed(seed)
    return build_model(cfg.require_model(), cfg.data).eval()


def _agents(rng: np.random.Generator, shape: tuple[int, ...], valid: np.ndarray) -> np.ndarray:
    """Agent states as the cache stores them: valid_flag set, padded frames all zero."""
    states = rng.normal(size=(*shape, H, F_AGENT)).astype(np.float32)
    states[..., 8] = valid
    states[~valid] = 0.0
    return states


def _batch(batch_size: int = 5, seed: int = 0) -> dict[str, np.ndarray]:
    """A full batch with gaps in the history (leading and internal), neighbor slots as the
    cache fills them (the SDC first, mixed types, trailing padded slots), and future-only
    fields."""
    rng = np.random.default_rng(seed)
    history_valid = np.ones((batch_size, H), bool)
    history_valid[0, :8] = False  # leading gap
    history_valid[1, 5:9] = False  # internal gap
    history_valid[2, :-1] = False  # only t0 observed
    history = _agents(rng, (batch_size,), history_valid)
    neighbor_valid = rng.random((batch_size, N, H)) > 0.3
    neighbor_valid[..., -1] = True  # selected neighbors are valid at t0 (design doc §8)
    neighbor_valid[:, -2:] = False  # two padded slots per sample
    neighbor_types = rng.integers(ObjectType.VEHICLE, len(ObjectType), size=(batch_size, N))
    neighbor_types[:, 0] = ObjectType.VEHICLE
    neighbor_types[:, -2:] = ObjectType.UNSET
    neighbor_is_sdc = np.zeros((batch_size, N), bool)
    neighbor_is_sdc[:, 0] = True
    map_valid = rng.random((batch_size, M, P)) > 0.3
    map_valid[:, -2:] = False  # two padded chunks per sample
    map_types = rng.integers(1, NUM_MAP_TYPES, size=(batch_size, M))
    map_types[~map_valid.any(axis=-1)] = MapType.PAD
    return {
        "target_history": history,
        "target_history_valid": history_valid,
        "target_type": np.full(batch_size, 2),
        "neighbors": _agents(rng, (batch_size, N), neighbor_valid),
        "neighbor_valid": neighbor_valid,
        "neighbor_types": neighbor_types,
        "neighbor_is_sdc": neighbor_is_sdc,
        "map_polylines": rng.normal(size=(batch_size, M, P, 4)).astype(np.float32),
        "map_valid": map_valid,
        "map_types": map_types,
        "target_future": rng.normal(size=(batch_size, T, 2)).astype(np.float32),
        "target_future_valid": np.ones((batch_size, T), bool),
        "ego_future": rng.normal(size=(batch_size, T, 5)).astype(np.float32),
    }


def _predict(model: ForecastModel, batch: dict, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        return model.to(device)(**model_inputs(batch, device))["trajectories"].cpu().numpy()


def test_every_model_name_has_a_config_that_builds():
    names = set()
    for config_name in MODEL_CONFIGS:
        model = _model(config_name)
        assert isinstance(model, ForecastModel)
        names.add(load_config(REPO_ROOT / "configs" / config_name).require_model().name)
    assert names == set(MODEL_NAMES)


@pytest.mark.parametrize("config_name", MODEL_CONFIGS)
def test_output_shapes_and_finite(config_name):
    model, batch = _model(config_name), _batch()
    with torch.no_grad():
        output = model(**model_inputs(batch))
    k = model.num_modes
    assert output["trajectories"].shape == (5, k, T, 2)
    assert output["logits"].shape == (5, k)
    assert torch.isfinite(output["trajectories"]).all()  # incl. the t0-only history


@pytest.mark.parametrize("config_name", MODEL_CONFIGS)
def test_future_fields_cannot_change_outputs(config_name):
    """Check 2 for models: perturbing every future-derived field leaves outputs identical."""
    model, batch = _model(config_name), _batch()
    perturbed = {**batch, "target_future": batch["target_future"] + 100.0,
                 "target_future_valid": ~batch["target_future_valid"],
                 "ego_future": batch["ego_future"] * -3.0}  # fmt: skip
    np.testing.assert_array_equal(_predict(model, batch), _predict(model, perturbed))


@pytest.mark.parametrize("config_name", MODEL_CONFIGS)
def test_padding_invariance(config_name):
    """Check 3: random values in padded slots (history, neighbors, map) change nothing."""
    model, batch = _model(config_name), _batch()
    rng = np.random.default_rng(1)
    noisy = {key: value.copy() for key, value in batch.items()}
    masks = {"target_history": "target_history_valid", "neighbors": "neighbor_valid",
             "map_polylines": "map_valid"}  # fmt: skip
    for field, mask in masks.items():
        padded = ~batch[mask]
        noisy[field][padded] = rng.normal(scale=50.0, size=noisy[field][padded].shape)
    padded_chunks = ~batch["map_valid"].any(axis=-1)
    noisy["map_types"][padded_chunks] = rng.integers(0, NUM_MAP_TYPES, size=padded_chunks.sum())
    padded_slots = ~batch["neighbor_valid"].any(axis=-1)
    noisy["neighbor_types"][padded_slots] = rng.integers(0, len(ObjectType), padded_slots.sum())
    noisy["neighbor_is_sdc"][padded_slots] = True
    np.testing.assert_array_equal(_predict(model, batch), _predict(model, noisy))


@pytest.mark.parametrize("config_name", MODEL_CONFIGS)
def test_map_reaches_exactly_the_map_models(config_name):
    """Moving the valid map points changes the output iff the model declares map inputs, so
    `input_fields` (what the loader reads) matches what the model uses, and the map-disabled
    ablation (`lstm`) really ignores the map."""
    model, batch = _model(config_name), _batch()
    moved = {**batch, "map_polylines": batch["map_polylines"] + 3.0}
    changed = not np.array_equal(_predict(model, batch), _predict(model, moved))
    assert changed == ("map_polylines" in model.input_fields)


@pytest.mark.parametrize("config_name", MODEL_CONFIGS)
def test_neighbors_reach_exactly_the_neighbor_models(config_name):
    """As above for neighbors: moving the valid neighbor states changes the output iff the
    model declares neighbor inputs."""
    model, batch = _model(config_name), _batch()
    neighbors = batch["neighbors"].copy()
    neighbors[..., :2][batch["neighbor_valid"]] += 3.0
    moved = {**batch, "neighbors": neighbors}
    changed = not np.array_equal(_predict(model, batch), _predict(model, moved))
    assert changed == ("neighbors" in model.input_fields)


@pytest.mark.parametrize("config_name", LEARNED_CONFIGS)
def test_zero_neighbors_zero_map_finite(config_name):
    """Check 10: samples with no neighbor and no valid map chunk still get finite outputs and
    gradients, and their padded neighbor and map content is ignored."""
    model, batch = _model(config_name).train(), _batch()
    empty = {**batch, "map_valid": np.zeros_like(batch["map_valid"]),
             "map_types": np.zeros_like(batch["map_types"]),
             "neighbors": np.zeros_like(batch["neighbors"]),
             "neighbor_valid": np.zeros_like(batch["neighbor_valid"]),
             "neighbor_types": np.zeros_like(batch["neighbor_types"]),
             "neighbor_is_sdc": np.zeros_like(batch["neighbor_is_sdc"])}  # fmt: skip
    output = model(**model_inputs(empty))["trajectories"]
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    other = {**empty, "map_polylines": batch["map_polylines"] * 7.0 + 1.0,
             "neighbors": batch["neighbors"] * 7.0 + 1.0}  # fmt: skip
    np.testing.assert_array_equal(_predict(model, empty), _predict(model, other))


@pytest.mark.parametrize("config_name", MODEL_CONFIGS)
def test_extra_padded_slots_change_nothing(config_name):
    """More padding (larger N and M, the new slots empty) leaves outputs unchanged. Padded
    tokens are zeroed, so the padding test alone cannot tell whether they are also excluded
    from attention; this one can, since extra attended keys would shift every softmax."""
    model, batch = _model(config_name), _batch()
    wider = dict(batch)
    for fields, extra in (
        (("neighbors", "neighbor_valid", "neighbor_types", "neighbor_is_sdc"), 3),
        (("map_polylines", "map_valid", "map_types"), 5),
    ):
        for field in fields:
            value = batch[field]
            empty = np.zeros((len(value), extra, *value.shape[2:]), value.dtype)
            wider[field] = np.concatenate([value, empty], axis=1)
    np.testing.assert_allclose(_predict(model, wider), _predict(model, batch), atol=1e-5)


@pytest.mark.parametrize("config_name", INTERACTION_CONFIGS)
def test_neighbor_and_map_slot_order_does_not_matter(config_name):
    """Tokens carry no slot position: the SDC is known by its flag, not by being slot 0, and
    reordering neighbors or map chunks only changes floating-point summation order."""
    model, batch = _model(config_name), _batch()
    rng = np.random.default_rng(2)
    neighbor_order, chunk_order = rng.permutation(N), rng.permutation(M)
    shuffled = dict(batch)
    for field in ("neighbors", "neighbor_valid", "neighbor_types", "neighbor_is_sdc"):
        shuffled[field] = batch[field][:, neighbor_order]
    for field in ("map_polylines", "map_valid", "map_types"):
        shuffled[field] = batch[field][:, chunk_order]
    np.testing.assert_allclose(_predict(model, shuffled), _predict(model, batch), atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("config_name", MODEL_CONFIGS)
def test_cpu_mps_parity(config_name):
    """Check 7: the same weights give the same outputs on the CPU and on Apple's GPU."""
    batch = _batch(batch_size=64, seed=3)
    cpu = _predict(_model(config_name), batch, "cpu")
    mps = _predict(_model(config_name), batch, "mps")
    np.testing.assert_allclose(mps, cpu, rtol=1e-4, atol=1e-4)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mlx.core as mx  # noqa: E402
from bot_training.features.build_features import INPUT_COLUMNS  # noqa: E402
from bot_training.models.pvp_sequence_model import PvPSequenceModel  # noqa: E402


BATCH_SIZE = 2
WINDOW_SIZE = 6
FEATURE_COUNT = len(INPUT_COLUMNS)
BOOLEAN_ACTION_COUNT = 9


def _make_model() -> PvPSequenceModel:
    return PvPSequenceModel(
        input_feature_count=FEATURE_COUNT,
        boolean_action_count=BOOLEAN_ACTION_COUNT,
    )


def _make_inputs() -> np.ndarray:
    return mx.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(BATCH_SIZE, WINDOW_SIZE, FEATURE_COUNT),
        dtype=mx.float32,
    )


def test_model_initialization() -> None:
    model = _make_model()

    assert isinstance(model, PvPSequenceModel)


def test_forward_pass_returns_outputs() -> None:
    model = _make_model()
    outputs = model(_make_inputs())

    assert outputs["binary_probabilities"] is not None
    assert outputs["slot_probabilities"] is not None
    assert outputs["continuous_deltas"] is not None


def test_output_shapes_match_requested_heads() -> None:
    model = _make_model()
    outputs = model(_make_inputs())

    assert outputs["binary_probabilities"].shape == (BATCH_SIZE, BOOLEAN_ACTION_COUNT)
    assert outputs["slot_probabilities"].shape == (BATCH_SIZE, 9)
    assert outputs["continuous_deltas"].shape == (BATCH_SIZE, 2)


def test_output_bounds_for_probability_heads() -> None:
    model = _make_model()
    outputs = model(_make_inputs())

    binary = np.asarray(outputs["binary_probabilities"])
    slots = np.asarray(outputs["slot_probabilities"])

    assert np.all(binary >= 0.0)
    assert np.all(binary <= 1.0)
    assert np.all(slots >= 0.0)
    assert np.allclose(slots.sum(axis=-1), 1.0)


def test_model_compiles_and_executes() -> None:
    model = _make_model()

    compiled_forward = mx.compile(lambda inputs: model(inputs))
    outputs = compiled_forward(_make_inputs())

    assert outputs["binary_probabilities"].shape == (BATCH_SIZE, BOOLEAN_ACTION_COUNT)
    assert outputs["slot_probabilities"].shape == (BATCH_SIZE, 9)
    assert outputs["continuous_deltas"].shape == (BATCH_SIZE, 2)


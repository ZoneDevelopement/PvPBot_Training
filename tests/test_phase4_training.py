from __future__ import annotations

import sys
from pathlib import Path

import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bot_training.features.build_features import INPUT_COLUMNS  # noqa: E402
from bot_training.models.pvp_sequence_model import PvPSequenceModel  # noqa: E402
from bot_training.training.phase4 import (  # noqa: E402
    SequenceDataset,
    binary_cross_entropy,
    categorical_cross_entropy,
    iter_batches,
    mean_squared_error,
    predict_mechanics,
    split_dataset_by_match,
    total_loss,
    train_on_batch,
    train_phase4_model,
)

FEATURE_COUNT = len(INPUT_COLUMNS)
CATEGORICAL_FEATURE_COUNT = 38
BOOLEAN_ACTION_COUNT = 9
WINDOW_SIZE = 5


def _make_synthetic_dataset(sample_count: int = 96, seed: int = 7) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(sample_count, WINDOW_SIZE, FEATURE_COUNT)).astype(np.float32)
    categorical_windows = np.zeros((sample_count, WINDOW_SIZE, CATEGORICAL_FEATURE_COUNT), dtype=np.int32)
    last = windows[:, -1, :]

    binary_targets = np.zeros((sample_count, BOOLEAN_ACTION_COUNT), dtype=np.float32)
    binary_targets[:, 0] = (last[:, 10] > 0.0).astype(np.float32)
    binary_targets[:, 4] = (last[:, 12] > 0.15).astype(np.float32)
    binary_targets[:, 6] = (last[:, 13] > 0.2).astype(np.float32)
    binary_targets[:, 7] = (np.abs(last[:, 10]) < 0.2).astype(np.float32)
    binary_targets[:, 8] = (last[:, 0] < -0.2).astype(np.float32)

    slot_targets = np.argmax(last[:, 11:20], axis=1).astype(np.int32)
    continuous_targets = np.stack([last[:, 11] * 0.4, last[:, 12] * 0.6], axis=1).astype(np.float32)

    match_ids = np.asarray([f"match_{i // 8}" for i in range(sample_count)], dtype=object)
    return SequenceDataset(windows, categorical_windows, binary_targets, slot_targets, continuous_targets, match_ids)


def _make_model() -> PvPSequenceModel:
    return PvPSequenceModel(
        input_feature_count=FEATURE_COUNT,
        boolean_action_count=BOOLEAN_ACTION_COUNT,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
    )


def _dataset_targets(dataset: SequenceDataset) -> dict[str, np.ndarray]:
    return {
        "binary_targets": dataset.binary_targets,
        "slot_targets": dataset.slot_targets,
        "continuous_targets": dataset.continuous_targets,
    }


def test_loss_functions_return_exact_zero_for_identical_predictions_and_targets() -> None:
    binary = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    slot = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    continuous = np.array([[0.25, -0.5]], dtype=np.float32)

    outputs = {
        "binary_probabilities": binary,
        "slot_probabilities": slot,
        "continuous_deltas": continuous,
    }
    targets = {
        "binary_targets": binary,
        "slot_targets": slot,
        "continuous_targets": continuous,
    }

    assert binary_cross_entropy(binary, binary) == 0.0
    assert categorical_cross_entropy(slot, slot) == 0.0
    assert mean_squared_error(continuous, continuous) == 0.0
    assert total_loss(outputs, targets) == 0.0


def test_dataset_split_is_match_safe_and_uses_an_80_20_partition() -> None:
    dataset = _make_synthetic_dataset(sample_count=80)
    split = split_dataset_by_match(dataset, train_ratio=0.8, seed=13)

    train_matches = set(split.train.match_ids.tolist())
    validation_matches = set(split.validation.match_ids.tolist())

    assert train_matches.isdisjoint(validation_matches)
    assert len(train_matches) == 8
    assert len(validation_matches) == 2


def test_batch_iterator_obeys_batch_size() -> None:
    dataset = _make_synthetic_dataset(sample_count=130)
    batches = list(iter_batches(dataset, batch_size=64, shuffle=False))

    assert len(batches) == 3
    assert batches[0].inputs.shape[0] == 64
    assert batches[1].inputs.shape[0] == 64
    assert batches[2].inputs.shape[0] == 2


def test_gradient_updates_reduce_loss_on_a_single_batch() -> None:
    dataset = _make_synthetic_dataset(sample_count=64)
    model = _make_model()
    optimizer = optim.Adam(learning_rate=0.001)

    initial_loss = total_loss(
        predict_mechanics(model, dataset.inputs, dataset.categorical_inputs),
        _dataset_targets(dataset),
    )
    for _ in range(50):
        train_on_batch(model, optimizer, dataset)
    final_loss = total_loss(
        predict_mechanics(model, dataset.inputs, dataset.categorical_inputs),
        _dataset_targets(dataset),
    )

    assert final_loss < initial_loss


def test_predict_mechanics_returns_expected_head_shapes() -> None:
    dataset = _make_synthetic_dataset(sample_count=12)
    model = _make_model()

    outputs = predict_mechanics(model, dataset.inputs[:4], dataset.categorical_inputs[:4])

    assert outputs["binary_probabilities"].shape == (4, BOOLEAN_ACTION_COUNT)
    assert outputs["slot_probabilities"].shape == (4, 9)
    assert outputs["continuous_deltas"].shape == (4, 2)


def test_training_loop_emits_per_epoch_log(capsys) -> None:
    dataset = _make_synthetic_dataset(sample_count=64)
    model = _make_model()

    train_phase4_model(
        dataset,
        model,
        epochs=1,
        batch_size=64,
        learning_rate=0.001,
        validation_ratio=0.2,
        seed=19,
        checkpoint_path=None,
    )

    captured = capsys.readouterr().out
    assert "Epoch 1/1" in captured
    assert "train_loss=" in captured
    assert "val_loss=" in captured
    assert "best_val_loss=" in captured

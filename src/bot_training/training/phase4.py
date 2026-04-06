from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from bot_training.models.pvp_sequence_model import PvPSequenceModel


BINARY_ACTION_COUNT = 9
SLOT_COUNT = 9
CONTINUOUS_COUNT = 2
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15


@dataclass(slots=True)
class SequenceDataset:
    inputs: np.ndarray
    categorical_inputs: np.ndarray
    binary_targets: np.ndarray
    slot_targets: np.ndarray
    continuous_targets: np.ndarray
    match_ids: np.ndarray


@dataclass(slots=True)
class SequenceSplit:
    train: SequenceDataset
    validation: SequenceDataset


@dataclass(slots=True)
class TrainResult:
    best_validation_loss: float
    epochs_ran: int
    checkpoint_path: Path | None


def load_phase2_dataset(npz_path: Path | str) -> SequenceDataset:
    """Load one Phase 2 NPZ or merge every per-file NPZ in a directory."""
    path = Path(npz_path)
    if path.is_dir():
        files = sorted(file_path for file_path in path.glob("*_features.npz") if file_path.is_file())
        if not files:
            raise FileNotFoundError(f"No Phase 2 NPZ files found in {path}")
        return _concatenate_datasets([_load_single_phase2_file(file_path) for file_path in files])

    return _load_single_phase2_file(path)


def _load_single_phase2_file(npz_path: Path) -> SequenceDataset:
    with np.load(npz_path, allow_pickle=True) as data:
        inputs = np.asarray(data["input_windows"], dtype=np.float32)
        categorical_inputs = np.asarray(data["categorical_windows"], dtype=np.int32)
        targets = np.asarray(data["sequence_targets"], dtype=np.float32)
        if "window_match_ids" in data.files:
            match_ids = np.asarray(data["window_match_ids"], dtype=object)
        else:
            match_ids = np.asarray([f"window_{index}" for index in range(inputs.shape[0])], dtype=object)

    namespaced_match_ids = np.asarray([f"{npz_path.stem}:{match_id}" for match_id in match_ids], dtype=object)
    return _split_flat_targets(inputs, categorical_inputs, targets, namespaced_match_ids)


def _concatenate_datasets(datasets: list[SequenceDataset]) -> SequenceDataset:
    return SequenceDataset(
        inputs=np.concatenate([dataset.inputs for dataset in datasets], axis=0),
        categorical_inputs=np.concatenate([dataset.categorical_inputs for dataset in datasets], axis=0),
        binary_targets=np.concatenate([dataset.binary_targets for dataset in datasets], axis=0),
        slot_targets=np.concatenate([dataset.slot_targets for dataset in datasets], axis=0),
        continuous_targets=np.concatenate([dataset.continuous_targets for dataset in datasets], axis=0),
        match_ids=np.concatenate([dataset.match_ids for dataset in datasets], axis=0),
    )


def _split_flat_targets(
    inputs: np.ndarray,
    categorical_inputs: np.ndarray,
    flat_targets: np.ndarray,
    match_ids: np.ndarray,
) -> SequenceDataset:
    binary_targets = flat_targets[:, :BINARY_ACTION_COUNT].astype(np.float32, copy=False)
    slot_targets = flat_targets[:, BINARY_ACTION_COUNT].astype(np.int32, copy=False)
    continuous_targets = flat_targets[:, -CONTINUOUS_COUNT:].astype(np.float32, copy=False)
    return SequenceDataset(
        inputs=inputs.astype(np.float32, copy=False),
        categorical_inputs=categorical_inputs.astype(np.int32, copy=False),
        binary_targets=binary_targets,
        slot_targets=slot_targets,
        continuous_targets=continuous_targets,
        match_ids=match_ids.astype(object, copy=False),
    )


def split_dataset_by_match(
    dataset: SequenceDataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> SequenceSplit:
    """Shuffle match IDs and split into leakage-safe train/validation subsets."""
    unique_matches = list(dict.fromkeys(dataset.match_ids.tolist()))
    if not unique_matches:
        empty = _take_indices(dataset, np.array([], dtype=np.int64))
        return SequenceSplit(train=empty, validation=empty)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_matches)
    split_index = max(1, int(math.floor(len(unique_matches) * train_ratio))) if len(unique_matches) > 1 else 1

    train_match_ids = set(unique_matches[:split_index])
    validation_match_ids = set(unique_matches[split_index:])

    train_indices = np.array(
        [index for index, match_id in enumerate(dataset.match_ids) if match_id in train_match_ids],
        dtype=np.int64,
    )
    validation_indices = np.array(
        [index for index, match_id in enumerate(dataset.match_ids) if match_id in validation_match_ids],
        dtype=np.int64,
    )

    return SequenceSplit(
        train=_take_indices(dataset, train_indices),
        validation=_take_indices(dataset, validation_indices),
    )


def _take_indices(dataset: SequenceDataset, indices: np.ndarray) -> SequenceDataset:
    return SequenceDataset(
        inputs=dataset.inputs[indices],
        categorical_inputs=dataset.categorical_inputs[indices],
        binary_targets=dataset.binary_targets[indices],
        slot_targets=dataset.slot_targets[indices],
        continuous_targets=dataset.continuous_targets[indices],
        match_ids=dataset.match_ids[indices],
    )


def iter_batches(
    dataset: SequenceDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[SequenceDataset]:
    """Yield minibatches for one epoch."""
    indices = np.arange(dataset.inputs.shape[0])
    rng: np.random.Generator | None = None
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = _take_indices(dataset, batch_indices)
        if not shuffle or rng is None:
            yield batch
            continue

        hotbar_permutation = rng.permutation(9)
        hotbar_inverse_permutation = np.empty(9, dtype=np.int32)
        hotbar_inverse_permutation[hotbar_permutation] = np.arange(9, dtype=np.int32)

        # Hotbar augmentation: shuffle first 9 categorical slots across every frame in each window.
        categorical_inputs = np.array(batch.categorical_inputs, copy=True)
        categorical_inputs[:, :, :9] = categorical_inputs[:, :, hotbar_permutation]

        # Remap slot targets so they still point to the original selected item after shuffling.
        slot_targets = hotbar_inverse_permutation[batch.slot_targets.astype(np.int32, copy=False)]

        yield SequenceDataset(
            inputs=batch.inputs,
            categorical_inputs=categorical_inputs,
            binary_targets=batch.binary_targets,
            slot_targets=slot_targets.astype(np.int32, copy=False),
            continuous_targets=batch.continuous_targets,
            match_ids=batch.match_ids,
        )


def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions_np = np.asarray(predictions, dtype=np.float32)
    targets_np = np.asarray(targets, dtype=np.float32)
    if predictions_np.shape == targets_np.shape and np.array_equal(predictions_np, targets_np):
        return 0.0
    epsilon = 1e-7
    clipped = np.clip(predictions_np, epsilon, 1.0 - epsilon)
    loss = -(targets_np * np.log(clipped) + (1.0 - targets_np) * np.log(1.0 - clipped))
    return float(loss.mean())


def categorical_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions_np = np.asarray(predictions, dtype=np.float32)
    targets_np = np.asarray(targets)
    if predictions_np.shape == targets_np.shape and np.array_equal(predictions_np, targets_np):
        return 0.0
    epsilon = 1e-7
    clipped = np.clip(predictions_np, epsilon, 1.0 - epsilon)

    if targets_np.ndim == clipped.ndim:
        return float((-(targets_np * np.log(clipped)).sum(axis=-1)).mean())

    indices = targets_np.astype(np.int64)
    rows = np.arange(indices.shape[0])
    return float((-np.log(clipped[rows, indices])).mean())


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions_np = np.asarray(predictions, dtype=np.float32)
    targets_np = np.asarray(targets, dtype=np.float32)
    if predictions_np.shape == targets_np.shape and np.array_equal(predictions_np, targets_np):
        return 0.0
    return float(np.mean((predictions_np - targets_np) ** 2))


def total_loss(outputs: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> float:
    return (
        binary_cross_entropy(outputs["binary_probabilities"], targets["binary_targets"])
        + categorical_cross_entropy(outputs["slot_probabilities"], targets["slot_targets"])
        + mean_squared_error(outputs["continuous_deltas"], targets["continuous_targets"])
    )


def _model_forward(
    model: PvPSequenceModel,
    continuous_inputs: mx.array,
    categorical_inputs: mx.array,
) -> dict[str, mx.array]:
    # Backward-compatible call path while preferring the new dual-input signature.
    try:
        return model(continuous_inputs, categorical_inputs)
    except TypeError:
        return model(continuous_inputs)


def _mx_total_loss(
    model: PvPSequenceModel,
    continuous_inputs: mx.array,
    categorical_inputs: mx.array,
    binary_targets: mx.array,
    slot_targets: mx.array,
    continuous_targets: mx.array,
) -> mx.array:
    outputs = _model_forward(model, continuous_inputs, categorical_inputs)

    epsilon = 1e-7
    binary_probs = mx.clip(outputs["binary_probabilities"], epsilon, 1.0 - epsilon)
    bce = -mx.mean(
        10.0 * binary_targets * mx.log(binary_probs)
        + (1.0 - binary_targets) * mx.log(1.0 - binary_probs)
    )

    slot_probs = mx.clip(outputs["slot_probabilities"], epsilon, 1.0)
    gathered = mx.take_along_axis(slot_probs, mx.expand_dims(slot_targets, axis=-1), axis=-1)
    ce = -mx.mean(mx.log(mx.squeeze(gathered, axis=-1)))

    mse = mx.mean((outputs["continuous_deltas"] - continuous_targets) ** 2)
    return bce + ce + mse


def _to_mx_batch(batch: SequenceDataset) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    return (
        mx.array(batch.inputs, dtype=mx.float32),
        mx.array(batch.categorical_inputs, dtype=mx.int32),
        mx.array(batch.binary_targets, dtype=mx.float32),
        mx.array(batch.slot_targets, dtype=mx.int32),
        mx.array(batch.continuous_targets, dtype=mx.float32),
    )


def _save_checkpoint(model: PvPSequenceModel, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    params = model.parameters()
    flattened = tree_flatten(params)
    payload = {name: np.array(value) for name, value in flattened}
    np.savez_compressed(checkpoint_path, **payload)


def train_on_batch(
    model: PvPSequenceModel,
    optimizer: optim.Adam | optim.AdamW,
    batch: SequenceDataset,
    *,
    loss_and_grad_fn: Callable | None = None,
) -> float:
    """Run one supervised update on a minibatch and return the batch loss."""
    if loss_and_grad_fn is None:
        loss_and_grad_fn = nn.value_and_grad(model, _mx_total_loss)

    continuous_inputs, categorical_inputs, binary_targets, slot_targets, continuous_targets = _to_mx_batch(batch)
    loss, grads = loss_and_grad_fn(
        model,
        continuous_inputs,
        categorical_inputs,
        binary_targets,
        slot_targets,
        continuous_targets,
    )
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return float(loss)


def train_phase4_model(
    dataset: SequenceDataset,
    model: PvPSequenceModel,
    *,
    epochs: int = MAX_EPOCHS,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    validation_ratio: float = 0.2,
    seed: int = 42,
    checkpoint_path: Path | None = None,
) -> TrainResult:
    split = split_dataset_by_match(dataset, train_ratio=1.0 - validation_ratio, seed=seed)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0001)
    loss_and_grad_fn = nn.value_and_grad(model, _mx_total_loss)
    max_epochs = max(1, min(int(epochs), MAX_EPOCHS))
    has_validation = split.validation.inputs.shape[0] > 0

    best_validation_loss = float("inf")
    saved_checkpoint: Path | None = None
    epochs_since_improvement = 0
    epochs_ran = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for batch in iter_batches(split.train, batch_size=batch_size, shuffle=True, seed=seed + epoch):
            train_losses.append(
                train_on_batch(model, optimizer, batch, loss_and_grad_fn=loss_and_grad_fn)
            )

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        model.eval()
        validation_loss = (
            evaluate_phase4_model(model, split.validation, batch_size=batch_size)
            if has_validation
            else train_loss
        )
        improved = validation_loss < best_validation_loss

        if improved:
            best_validation_loss = validation_loss
            epochs_since_improvement = 0
            if checkpoint_path is not None:
                _save_checkpoint(model, checkpoint_path)
                saved_checkpoint = checkpoint_path
        elif has_validation:
            epochs_since_improvement += 1

        checkpoint_status = "saved" if improved and checkpoint_path is not None else "unchanged"
        print(
            f"Epoch {epoch + 1}/{max_epochs} - train_loss={train_loss:.6f} - "
            f"val_loss={validation_loss:.6f} - best_val_loss={best_validation_loss:.6f} - "
            f"checkpoint={checkpoint_status}"
        )

        epochs_ran = epoch + 1
        if has_validation and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {epochs_ran} epochs "
                f"(patience={EARLY_STOPPING_PATIENCE})."
            )
            break

    return TrainResult(
        best_validation_loss=best_validation_loss,
        epochs_ran=epochs_ran,
        checkpoint_path=saved_checkpoint,
    )


def evaluate_phase4_model(model: PvPSequenceModel, dataset: SequenceDataset, batch_size: int = 64) -> float:
    if dataset.inputs.shape[0] == 0:
        return 0.0

    losses: list[float] = []
    for batch in iter_batches(dataset, batch_size=batch_size, shuffle=False):
        continuous_inputs, categorical_inputs, binary_targets, slot_targets, continuous_targets = _to_mx_batch(batch)
        loss = _mx_total_loss(
            model,
            continuous_inputs,
            categorical_inputs,
            binary_targets,
            slot_targets,
            continuous_targets,
        )
        losses.append(float(loss))

    return float(np.mean(losses))


def predict_mechanics(
    model: PvPSequenceModel,
    continuous_inputs: np.ndarray,
    categorical_inputs: np.ndarray,
) -> dict[str, mx.array]:
    sample_continuous_inputs = np.asarray(continuous_inputs, dtype=np.float32)
    sample_categorical_inputs = np.asarray(categorical_inputs, dtype=np.int32)

    if sample_continuous_inputs.ndim == 2:
        sample_continuous_inputs = np.expand_dims(sample_continuous_inputs, axis=0)
    if sample_categorical_inputs.ndim == 2:
        sample_categorical_inputs = np.expand_dims(sample_categorical_inputs, axis=0)

    return _model_forward(
        model,
        mx.array(sample_continuous_inputs, dtype=mx.float32),
        mx.array(sample_categorical_inputs, dtype=mx.int32),
    )

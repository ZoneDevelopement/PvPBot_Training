"""Feature engineering helpers for sequence model training."""

from __future__ import annotations

import json
import re
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_COLUMNS: tuple[str, ...] = (
    "health",
    "posX",
    "posY",
    "posZ",
    "velX",
    "velY",
    "velZ",
    "yaw",
    "pitch",
    "isOnGround",
    "targetDistance",
    "targetRelX",
    "targetRelY",
    "targetRelZ",
    "targetYaw",
    "targetPitch",
    "targetVelX",
    "targetVelY",
    "targetVelZ",
    "targetHealth",
)

MATCH_ID_COLUMN: str = "match_id"

BINARY_ACTION_COLUMNS: tuple[str, ...] = (
    "inputForward",
    "inputBackward",
    "inputLeft",
    "inputRight",
    "inputJump",
    "inputSneak",
    "inputSprint",
    "inputLmb",
    "inputRmb",
)

# Backward-compatible alias used by the existing feature tests and exports.
ACTION_COLUMNS: tuple[str, ...] = BINARY_ACTION_COLUMNS

SLOT_COLUMN: str = "inputSlot"
SLOT_COUNT: int = 9

BOOLEAN_INPUT_COLUMNS: tuple[str, ...] = ("isOnGround",)
CONTINUOUS_INPUT_COLUMNS: tuple[str, ...] = tuple(col for col in INPUT_COLUMNS if col not in BOOLEAN_INPUT_COLUMNS)
DELTA_COLUMNS: tuple[str, ...] = ("deltaYaw", "deltaPitch")
TARGET_COLUMNS: tuple[str, ...] = ACTION_COLUMNS + (SLOT_COLUMN,) + DELTA_COLUMNS

HEALTH_COLUMNS: tuple[str, ...] = ("health", "targetHealth")
YAW_COLUMNS: tuple[str, ...] = ("yaw", "targetYaw")
PITCH_COLUMNS: tuple[str, ...] = ("pitch", "targetPitch")
SPATIAL_COLUMNS: tuple[str, ...] = (
    "posX",
    "posY",
    "posZ",
    "targetDistance",
    "targetRelX",
    "targetRelY",
    "targetRelZ",
)
VELOCITY_COLUMNS: tuple[str, ...] = (
    "velX",
    "velY",
    "velZ",
    "targetVelX",
    "targetVelY",
    "targetVelZ",
)

_TRUE_VALUES = {"true", "1", "t", "yes", "y"}
_EMPTY_ITEM_VALUES = {"", "air", "none", "null", "nan"}
UNKNOWN_ITEM_TOKEN: str = "UNKNOWN"

_LEADING_SLOT_PREFIX_PATTERN = re.compile(r"^\s*\d+\s*=\s*")
_TRAILING_QUANTITY_PATTERN = re.compile(r"\s*[xX]\d+\s*$")

MAIN_HAND_COLUMN: str = "mainHandItem"
OFF_HAND_COLUMN: str = "offHandItem"
HOTBAR_COLUMNS: tuple[str, ...] = tuple(f"hotbar{slot}" for slot in range(9))
INVENTORY_BAG_COLUMN: str = "inventoryBag"
INVENTORY_BAG_SLOT_COUNT: int = 27

DIRECT_ITEM_COLUMNS: tuple[str, ...] = (MAIN_HAND_COLUMN, OFF_HAND_COLUMN) + HOTBAR_COLUMNS
INVENTORY_BAG_SLOT_COLUMNS: tuple[str, ...] = tuple(
    f"inventoryBagSlot{slot}" for slot in range(INVENTORY_BAG_SLOT_COUNT)
)
ITEM_SLOT_COLUMNS: tuple[str, ...] = DIRECT_ITEM_COLUMNS + INVENTORY_BAG_SLOT_COLUMNS


@dataclass(slots=True)
class FeatureEngineeringResult:
    """Container for model-ready arrays."""

    inputs: np.ndarray
    categorical_inputs: np.ndarray
    targets: np.ndarray
    input_windows: np.ndarray
    categorical_windows: np.ndarray
    sequence_targets: np.ndarray
    window_match_ids: np.ndarray
    item_vocabulary: dict[str, int]


def _require_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def to_binary_int(series: pd.Series) -> pd.Series:
    """Convert mixed boolean/string/numeric values to 0/1 integer flags."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.int8)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0).astype(np.int8)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(_TRUE_VALUES).astype(np.int8)


def extract_input_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract and coerce game-state feature columns."""
    _require_columns(dataframe, INPUT_COLUMNS)
    features = dataframe.loc[:, INPUT_COLUMNS].copy()
    for column in BOOLEAN_INPUT_COLUMNS:
        features[column] = to_binary_int(features[column])
    for column in CONTINUOUS_INPUT_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="raise")
    return features


def extract_action_targets(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract action columns and encode them as 0/1 integers."""
    _require_columns(dataframe, BINARY_ACTION_COLUMNS)
    actions = dataframe.loc[:, BINARY_ACTION_COLUMNS].copy()
    for column in BINARY_ACTION_COLUMNS:
        actions[column] = to_binary_int(actions[column])
    return actions.astype(np.int8)


def extract_slot_targets(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract the hotbar slot selection as an integer target in [0, 8]."""
    _require_columns(dataframe, (SLOT_COLUMN,))
    slots = pd.to_numeric(dataframe.loc[:, [SLOT_COLUMN]][SLOT_COLUMN], errors="raise")
    if slots.isna().any():
        raise ValueError("inputSlot contains missing values")
    if not slots.between(0, SLOT_COUNT - 1).all():
        raise ValueError("inputSlot must be an integer in the range [0, 8]")
    return pd.DataFrame({SLOT_COLUMN: slots.astype(np.int8)}, index=dataframe.index)


def compute_delta_targets(dataframe: pd.DataFrame, yaw_column: str = "yaw", pitch_column: str = "pitch") -> pd.DataFrame:
    """Build frame-to-frame mouse movement targets from absolute view angles."""
    _require_columns(dataframe, (yaw_column, pitch_column))
    yaw = pd.to_numeric(dataframe[yaw_column], errors="raise")
    pitch = pd.to_numeric(dataframe[pitch_column], errors="raise")
    if MATCH_ID_COLUMN in dataframe.columns:
        grouped = dataframe.loc[:, [MATCH_ID_COLUMN]].copy()
        grouped["yaw"] = yaw
        grouped["pitch"] = pitch

        delta_yaw = grouped.groupby(MATCH_ID_COLUMN, sort=False)["yaw"].shift(-1) - grouped["yaw"]
        delta_pitch = grouped.groupby(MATCH_ID_COLUMN, sort=False)["pitch"].shift(-1) - grouped["pitch"]
    else:
        delta_yaw = yaw.shift(-1) - yaw
        delta_pitch = pitch.shift(-1) - pitch
    deltas = pd.DataFrame(
        {
            "deltaYaw": delta_yaw,
            "deltaPitch": delta_pitch,
        },
        index=dataframe.index,
    )
    return deltas


def build_targets_with_deltas(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Concatenate action, slot, and next-frame yaw/pitch delta targets."""
    actions = extract_action_targets(dataframe)
    slot = extract_slot_targets(dataframe)
    deltas = compute_delta_targets(dataframe)
    targets = pd.concat([actions, slot, deltas], axis=1)
    return targets.loc[:, TARGET_COLUMNS]


def align_next_frame_rows(inputs: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows where next-frame deltas are undefined (last frame)."""
    valid_mask = targets.loc[:, DELTA_COLUMNS].notna().all(axis=1)
    return inputs.loc[valid_mask].reset_index(drop=True), targets.loc[valid_mask].reset_index(drop=True)


def _resolve_match_ids(dataframe: pd.DataFrame, row_count: int) -> pd.Series:
    if MATCH_ID_COLUMN in dataframe.columns:
        return dataframe.loc[:, MATCH_ID_COLUMN].astype(str)
    return pd.Series(["__single_match__"] * row_count, index=dataframe.index)


def _normalize_item_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = str(value).strip()
    normalized = _LEADING_SLOT_PREFIX_PATTERN.sub("", normalized)
    normalized = _TRAILING_QUANTITY_PATTERN.sub("", normalized).strip()
    if normalized.lower() in _EMPTY_ITEM_VALUES:
        return ""
    return normalized


def _parse_inventory_bag_value(value: object) -> list[str]:
    if value is None:
        return [""] * INVENTORY_BAG_SLOT_COUNT

    if not isinstance(value, (list, tuple, np.ndarray, dict)):
        try:
            if pd.isna(value):
                return [""] * INVENTORY_BAG_SLOT_COUNT
        except TypeError:
            pass

    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        raw = str(value).strip()
        if not raw:
            parsed = []
        else:
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    decoded = literal_eval(raw)
                except (ValueError, SyntaxError):
                    decoded = [part.strip() for part in raw.replace(";", ",").split(",")]
            if isinstance(decoded, dict):
                parsed = [decoded.get(index, decoded.get(str(index), "")) for index in range(INVENTORY_BAG_SLOT_COUNT)]
            elif isinstance(decoded, (list, tuple, np.ndarray)):
                parsed = list(decoded)
            else:
                parsed = [decoded]

    normalized = [_normalize_item_name(item) for item in parsed[:INVENTORY_BAG_SLOT_COUNT]]
    if len(normalized) < INVENTORY_BAG_SLOT_COUNT:
        normalized.extend([""] * (INVENTORY_BAG_SLOT_COUNT - len(normalized)))
    return normalized


def parse_inventory_bag_slots(series: pd.Series) -> pd.DataFrame:
    """Parse serialized inventory bags into 27 fixed slot columns."""
    rows = [_parse_inventory_bag_value(value) for value in series]
    return pd.DataFrame(rows, columns=INVENTORY_BAG_SLOT_COLUMNS, index=series.index, dtype="string")


def extract_categorical_item_slots(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract all 38 categorical item slots from direct and bag inventory sources."""
    _require_columns(dataframe, DIRECT_ITEM_COLUMNS + (INVENTORY_BAG_COLUMN,))

    direct_slots = dataframe.loc[:, DIRECT_ITEM_COLUMNS].copy()
    for column in DIRECT_ITEM_COLUMNS:
        direct_slots[column] = direct_slots[column].map(_normalize_item_name)

    bag_slots = parse_inventory_bag_slots(dataframe.loc[:, INVENTORY_BAG_COLUMN])
    combined = pd.concat([direct_slots, bag_slots], axis=1)
    return combined.loc[:, ITEM_SLOT_COLUMNS].astype("string")


def build_item_vocabulary(categorical_slots: pd.DataFrame) -> dict[str, int]:
    """Create a stable item vocabulary with reserved EMPTY/UNKNOWN identifiers."""
    if tuple(categorical_slots.columns) != ITEM_SLOT_COLUMNS:
        categorical_slots = categorical_slots.loc[:, ITEM_SLOT_COLUMNS]
    unique_items = sorted(
        {
            item
            for item in categorical_slots.to_numpy().ravel()
            if item and item != UNKNOWN_ITEM_TOKEN
        }
    )
    vocabulary: dict[str, int] = {"": 0, UNKNOWN_ITEM_TOKEN: 1}
    vocabulary.update({item: idx for idx, item in enumerate(unique_items, start=2)})
    return vocabulary


def save_item_vocabulary(vocabulary: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(vocabulary, indent=2, sort_keys=True), encoding="utf-8")


def apply_item_vocabulary(categorical_slots: pd.DataFrame, vocabulary: dict[str, int]) -> np.ndarray:
    """Map 38 categorical item slots to integer ids."""
    if tuple(categorical_slots.columns) != ITEM_SLOT_COLUMNS:
        categorical_slots = categorical_slots.loc[:, ITEM_SLOT_COLUMNS]
    unknown_id = vocabulary.get(UNKNOWN_ITEM_TOKEN, 1)
    normalized = categorical_slots.fillna("").astype("string")
    encoded = normalized.apply(lambda column: column.map(vocabulary).fillna(unknown_id).astype(np.int32))
    return encoded.to_numpy(dtype=np.int32)


def _build_match_aware_windows(
    aligned_inputs: pd.DataFrame,
    aligned_categorical_inputs: pd.DataFrame,
    aligned_targets: pd.DataFrame,
    match_ids: pd.Series,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window_blocks: list[np.ndarray] = []
    categorical_window_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    window_match_ids: list[str] = []

    for match_id, group_index in match_ids.groupby(match_ids, sort=False).groups.items():
        group_inputs = aligned_inputs.loc[group_index]
        group_categorical_inputs = aligned_categorical_inputs.loc[group_index]
        group_targets = aligned_targets.loc[group_index]

        group_input_array = group_inputs.to_numpy(dtype=np.float32)
        group_categorical_array = group_categorical_inputs.to_numpy(dtype=np.int32)
        group_target_array = group_targets.to_numpy(dtype=np.float32)
        if group_input_array.shape[0] < window_size:
            continue

        group_windows = generate_sliding_windows(group_input_array, window_size=window_size)
        group_categorical_windows = generate_sliding_windows(group_categorical_array, window_size=window_size)
        group_targets_windowed = group_target_array[window_size - 1 :]

        window_blocks.append(group_windows)
        categorical_window_blocks.append(group_categorical_windows)
        target_blocks.append(group_targets_windowed)
        window_match_ids.extend([str(match_id)] * group_windows.shape[0])

    if not window_blocks:
        feature_width = aligned_inputs.shape[1]
        categorical_feature_width = aligned_categorical_inputs.shape[1]
        target_width = aligned_targets.shape[1]
        return (
            np.empty((0, window_size, feature_width), dtype=np.float32),
            np.empty((0, window_size, categorical_feature_width), dtype=np.int32),
            np.empty((0, target_width), dtype=np.float32),
            np.empty((0,), dtype=object),
        )

    return (
        np.concatenate(window_blocks, axis=0),
        np.concatenate(categorical_window_blocks, axis=0),
        np.concatenate(target_blocks, axis=0),
        np.asarray(window_match_ids, dtype=object),
    )


def normalize_continuous_inputs(inputs: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic Minecraft-aware scaling to continuous features."""
    normalized = inputs.copy().astype(np.float32)
    normalized.loc[:, HEALTH_COLUMNS] = normalized.loc[:, HEALTH_COLUMNS] / 20.0
    normalized.loc[:, YAW_COLUMNS] = normalized.loc[:, YAW_COLUMNS] / 180.0
    normalized.loc[:, PITCH_COLUMNS] = normalized.loc[:, PITCH_COLUMNS] / 90.0
    normalized.loc[:, SPATIAL_COLUMNS] = np.minimum(normalized.loc[:, SPATIAL_COLUMNS] / 50.0, 1.0)
    normalized.loc[:, VELOCITY_COLUMNS] = np.minimum(normalized.loc[:, VELOCITY_COLUMNS] / 4.0, 1.0)
    return normalized


def fit_transform_inputs(inputs: pd.DataFrame) -> np.ndarray:
    """Scale inputs with fixed bounds and return the normalized array."""
    normalized = normalize_continuous_inputs(inputs)
    return normalized.to_numpy(dtype=np.float32)


def generate_sliding_windows(array: np.ndarray, window_size: int = 20) -> np.ndarray:
    """Create overlapping 3D windows [num_windows, window_size, feature_count]."""
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")
    if array.ndim != 2:
        raise ValueError("Expected a 2D array with shape [rows, features]")
    rows, features = array.shape
    if rows < window_size:
        return np.empty((0, window_size, features), dtype=array.dtype)
    windows = np.lib.stride_tricks.sliding_window_view(array, window_shape=window_size, axis=0)
    return np.ascontiguousarray(np.swapaxes(windows, 1, 2))


def engineer_feature_tensors(
    dataframe: pd.DataFrame,
    window_size: int = 20,
    vocabulary_path: Path | None = None,
    item_vocabulary: dict[str, int] | None = None,
) -> FeatureEngineeringResult:
    """Full pipeline from filtered dataframe to model-ready sequence tensors."""
    inputs = extract_input_features(dataframe)
    categorical_slots = extract_categorical_item_slots(dataframe)
    resolved_vocabulary = item_vocabulary if item_vocabulary is not None else build_item_vocabulary(categorical_slots)
    if vocabulary_path is not None:
        save_item_vocabulary(resolved_vocabulary, vocabulary_path)
    categorical_inputs = apply_item_vocabulary(categorical_slots, resolved_vocabulary)

    targets = build_targets_with_deltas(dataframe)
    aligned_inputs, aligned_targets = align_next_frame_rows(inputs, targets)
    valid_mask = targets.loc[:, DELTA_COLUMNS].notna().all(axis=1)
    aligned_match_ids = _resolve_match_ids(dataframe, len(dataframe)).loc[valid_mask].reset_index(drop=True)
    aligned_categorical_inputs = pd.DataFrame(
        categorical_inputs,
        columns=ITEM_SLOT_COLUMNS,
    ).loc[valid_mask].reset_index(drop=True)

    normalized_inputs = fit_transform_inputs(aligned_inputs)
    target_array = aligned_targets.to_numpy(dtype=np.float32)
    input_windows, categorical_windows, sequence_targets, window_match_ids = _build_match_aware_windows(
        pd.DataFrame(normalized_inputs, columns=aligned_inputs.columns),
        aligned_categorical_inputs,
        aligned_targets,
        aligned_match_ids,
        window_size=window_size,
    )

    return FeatureEngineeringResult(
        inputs=normalized_inputs,
        categorical_inputs=aligned_categorical_inputs.to_numpy(dtype=np.int32),
        targets=target_array,
        input_windows=input_windows,
        categorical_windows=categorical_windows,
        sequence_targets=sequence_targets,
        window_match_ids=window_match_ids,
        item_vocabulary=resolved_vocabulary,
    )


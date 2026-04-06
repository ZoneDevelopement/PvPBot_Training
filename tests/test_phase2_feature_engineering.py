from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bot_training.features.build_features import (  # noqa: E402
    ACTION_COLUMNS,
    DIRECT_ITEM_COLUMNS,
    INPUT_COLUMNS,
    INVENTORY_BAG_SLOT_COUNT,
    ITEM_SLOT_COLUMNS,
    UNKNOWN_ITEM_TOKEN,
    apply_item_vocabulary,
    build_item_vocabulary,
    compute_delta_targets,
    extract_categorical_item_slots,
    extract_action_targets,
    extract_input_features,
    extract_slot_targets,
    fit_transform_inputs,
    generate_sliding_windows,
    parse_inventory_bag_slots,
)


def _mock_dataframe(row_count: int = 3) -> pd.DataFrame:
    def repeat_pattern(pattern: list[object]) -> list[object]:
        return [pattern[index % len(pattern)] for index in range(row_count)]

    data = {
        "health": np.linspace(20, 10, row_count),
        "foodLevel": np.linspace(20, 12, row_count),
        "damageDealt": np.linspace(0, 6, row_count),
        "posX": np.arange(row_count),
        "posY": np.arange(row_count) + 64,
        "posZ": np.arange(row_count) + 100,
        "velX": np.linspace(0.1, 0.3, row_count),
        "velY": np.linspace(0.0, 0.2, row_count),
        "velZ": np.linspace(-0.1, 0.1, row_count),
        "yaw": np.linspace(10, 30, row_count),
        "pitch": np.linspace(5, 15, row_count),
        "isOnGround": [True] * row_count,
        "targetDistance": np.linspace(6, 4, row_count),
        "targetRelX": np.linspace(1.0, 2.0, row_count),
        "targetRelY": np.linspace(-0.5, 0.5, row_count),
        "targetRelZ": np.linspace(0.5, 1.5, row_count),
        "nearestProjectileDx": np.linspace(-2.0, -1.0, row_count),
        "nearestProjectileDy": np.linspace(0.5, 1.5, row_count),
        "nearestProjectileDz": np.linspace(4.0, 2.0, row_count),
        "targetYaw": np.linspace(20, 40, row_count),
        "targetPitch": np.linspace(-10, 10, row_count),
        "targetVelX": np.linspace(0.0, 0.4, row_count),
        "targetVelY": np.linspace(0.0, 0.3, row_count),
        "targetVelZ": np.linspace(0.0, 0.2, row_count),
        "targetHealth": np.linspace(20, 8, row_count),
        "inputForward": repeat_pattern([True, False, True]),
        "inputBackward": repeat_pattern([False]),
        "inputLeft": repeat_pattern([False, True, False]),
        "inputRight": repeat_pattern([True, False, False]),
        "inputJump": repeat_pattern([False, True, False]),
        "inputSneak": repeat_pattern([False, False, True]),
        "inputSprint": repeat_pattern([True, True, False]),
        "inputLmb": repeat_pattern([True, False, True]),
        "inputRmb": repeat_pattern([False, True, False]),
        "inputSlot": repeat_pattern([0, 1, 8]),
        "mainHandItem": repeat_pattern(["diamond_sword", "diamond_sword", "bow"]),
        "offHandItem": repeat_pattern(["shield", "", "shield"]),
        "hotbar0": repeat_pattern(["diamond_sword", "diamond_sword", "bow"]),
        "hotbar1": repeat_pattern(["splash_healing", "splash_healing", "splash_healing"]),
        "hotbar2": repeat_pattern(["potion_healing", "potion_healing", "potion_healing"]),
        "hotbar3": repeat_pattern(["cooked_beef", "cooked_beef", "cooked_beef"]),
        "hotbar4": repeat_pattern(["golden_apple", "golden_apple", "golden_apple"]),
        "hotbar5": repeat_pattern(["", "", ""]),
        "hotbar6": repeat_pattern(["", "", ""]),
        "hotbar7": repeat_pattern(["", "", ""]),
        "hotbar8": repeat_pattern(["", "", ""]),
        "inventoryBag": repeat_pattern(
            [
                str(["", "ender_pearl", "cobblestone"] + [""] * 24),
                str(["arrow", "", ""] + [""] * 24),
                str(["", "", ""] + [""] * 24),
            ]
        ),
    }
    return pd.DataFrame(data)


def test_input_extraction_shape_matches_feature_count() -> None:
    dataframe = _mock_dataframe(row_count=5)
    inputs = extract_input_features(dataframe)

    assert inputs.shape == (5, len(INPUT_COLUMNS))


def test_boolean_conversion_outputs_binary_ints() -> None:
    dataframe = pd.DataFrame(
        {
            "inputForward": [True],
            "inputBackward": [False],
            "inputLeft": [True],
            "inputRight": [False],
            "inputJump": [True],
            "inputSneak": [False],
            "inputSprint": [True],
            "inputLmb": [False],
            "inputRmb": [True],
        }
    )

    actions = extract_action_targets(dataframe)

    assert actions.shape == (1, len(ACTION_COLUMNS))
    assert actions.to_numpy().dtype == np.int8
    assert np.array_equal(actions.to_numpy(), np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.int8))


def test_slot_extraction_outputs_integer_class() -> None:
    dataframe = pd.DataFrame({"inputSlot": [0, 4, 8]})

    slots = extract_slot_targets(dataframe)

    assert slots.shape == (3, 1)
    assert slots.to_numpy().dtype == np.int8
    assert np.array_equal(slots.to_numpy().ravel(), np.array([0, 4, 8], dtype=np.int8))


def test_delta_calculation_matches_expected_difference() -> None:
    dataframe = pd.DataFrame({"yaw": [10.0, 13.5], "pitch": [-2.0, 4.5]})

    deltas = compute_delta_targets(dataframe)

    assert deltas.loc[0, "deltaYaw"] == 3.5
    assert deltas.loc[0, "deltaPitch"] == 6.5


def test_normalizer_applies_fixed_minecraft_scaling() -> None:
    dataframe = _mock_dataframe(row_count=3)
    inputs = extract_input_features(dataframe)

    scaled = fit_transform_inputs(inputs)
    feature_idx = {name: idx for idx, name in enumerate(INPUT_COLUMNS)}

    assert np.isclose(scaled[0, feature_idx["health"]], 1.0)
    assert np.isclose(scaled[0, feature_idx["foodLevel"]], 1.0)
    assert np.isclose(scaled[0, feature_idx["damageDealt"]], 0.0)
    assert np.isclose(scaled[0, feature_idx["targetHealth"]], 1.0)
    assert np.isclose(scaled[0, feature_idx["yaw"]], 10.0 / 180.0)
    assert np.isclose(scaled[0, feature_idx["pitch"]], 5.0 / 90.0)
    assert np.isclose(scaled[0, feature_idx["targetDistance"]], 6.0 / 50.0)
    assert np.isclose(scaled[0, feature_idx["velX"]], 0.1 / 4.0)
    assert np.isclose(scaled[0, feature_idx["targetVelX"]], 0.0)
    assert np.isclose(scaled[0, feature_idx["posZ"]], 1.0)
    assert np.isclose(scaled[0, feature_idx["targetRelY"]], -0.5 / 50.0)
    assert np.isclose(scaled[0, feature_idx["nearestProjectileDx"]], -2.0 / 50.0)


def test_sequence_generator_creates_expected_window_count() -> None:
    array = np.arange(50 * 3, dtype=np.float32).reshape(50, 3)

    windows = generate_sliding_windows(array, window_size=20)

    assert windows.shape == (31, 20, 3)


def test_inventory_bag_parser_outputs_27_slots() -> None:
    series = pd.Series([
        str(["arrow", "healing_potion", ""] + [""] * 24),
        "",
    ])

    parsed = parse_inventory_bag_slots(series)

    assert parsed.shape == (2, INVENTORY_BAG_SLOT_COUNT)
    assert parsed.iloc[0, 0] == "arrow"
    assert parsed.iloc[0, 1] == "healing_potion"
    assert parsed.iloc[1].tolist() == [""] * INVENTORY_BAG_SLOT_COUNT


def test_categorical_inventory_features_have_38_slots_per_frame() -> None:
    dataframe = _mock_dataframe(row_count=40)

    categorical_slots = extract_categorical_item_slots(dataframe)
    vocabulary = build_item_vocabulary(categorical_slots)
    categorical_array = apply_item_vocabulary(categorical_slots, vocabulary)
    windows = generate_sliding_windows(categorical_array, window_size=20)

    assert tuple(categorical_slots.columns) == ITEM_SLOT_COLUMNS
    assert len(DIRECT_ITEM_COLUMNS) + INVENTORY_BAG_SLOT_COUNT == 38
    assert categorical_array.shape == (40, 38)
    assert windows.shape == (21, 20, 38)
    assert vocabulary[""] == 0
    assert vocabulary[UNKNOWN_ITEM_TOKEN] == 1


def test_item_name_normalizer_removes_slot_prefix_and_quantity_suffix() -> None:
    dataframe = _mock_dataframe(row_count=1)
    dataframe.loc[0, "mainHandItem"] = " 8=BOWx3 "
    dataframe.loc[0, "offHandItem"] = "1=DIAMOND SWORD x12"
    dataframe.loc[0, "hotbar0"] = "0=airx1"
    dataframe.loc[0, "inventoryBag"] = str(["9=ENDER PEARLx16"] + [""] * 26)

    categorical_slots = extract_categorical_item_slots(dataframe)

    assert categorical_slots.loc[0, "mainHandItem"] == "BOW"
    assert categorical_slots.loc[0, "offHandItem"] == "DIAMOND SWORD"
    assert categorical_slots.loc[0, "hotbar0"] == ""
    assert categorical_slots.loc[0, "inventoryBagSlot0"] == "ENDER PEARL"


def test_apply_item_vocabulary_maps_missing_items_to_unknown_id() -> None:
    slots = pd.DataFrame(
        {
            column: [""]
            for column in ITEM_SLOT_COLUMNS
        }
    )
    slots.loc[0, "mainHandItem"] = "known_item"
    slots.loc[0, "offHandItem"] = "new_item_not_in_vocab"

    vocabulary = {"": 0, UNKNOWN_ITEM_TOKEN: 1, "known_item": 2}
    encoded = apply_item_vocabulary(slots, vocabulary)

    assert encoded.shape == (1, len(ITEM_SLOT_COLUMNS))
    assert encoded[0, ITEM_SLOT_COLUMNS.index("mainHandItem")] == 2
    assert encoded[0, ITEM_SLOT_COLUMNS.index("offHandItem")] == 1



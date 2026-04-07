"""Run scenario-based sanity checks against the trained Phase 4 MLX model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import mlx.core as mx

import _bootstrap  # noqa: F401

from bot_training.config import CHECKPOINTS_DIR, EXPORTS_DIR
from bot_training.features.build_features import (
    BINARY_ACTION_COLUMNS,
    INPUT_COLUMNS,
    normalize_continuous_inputs,
)
from bot_training.models.pvp_sequence_model import PvPSequenceModel


WINDOW_SIZE = 20
INVENTORY_SLOT_COUNT = 38


@dataclass(slots=True)
class ScenarioResult:
    name: str
    passed: bool
    details: str


class ScenarioRunner:
    def __init__(
        self,
        model: PvPSequenceModel,
        args: argparse.Namespace,
        item_vocabulary: dict[str, int],
    ) -> None:
        self.model = model
        self.args = args
        self.item_vocabulary = item_vocabulary
        self.feature_index = {name: idx for idx, name in enumerate(INPUT_COLUMNS)}
        self.action_index = {name: idx for idx, name in enumerate(BINARY_ACTION_COLUMNS)}
        self.missing_capabilities: list[str] = []
        self.potion_item_id = self._item_id("POTION")
        self.splash_potion_item_id = self._item_id("SPLASH_POTION")
        self.food_item_id = self._item_id("COOKED_BEEF")
        self.golden_apple_item_id = self._item_id("GOLDEN_APPLE")
        self.sword_item_id = self._item_id("DIAMOND_SWORD")
        self.filler_item_id = self._item_id("OBSIDIAN")

    def _item_id(self, item_name: str) -> int:
        item_id = self.item_vocabulary.get(item_name)
        if item_id is None:
            raise KeyError(f"Required item '{item_name}' was not found in the item vocabulary.")
        if int(item_id) >= int(self.model.item_vocabulary_size):
            raise ValueError(
                f"Item '{item_name}' id {item_id} exceeds model item vocabulary size {self.model.item_vocabulary_size}."
            )
        return int(item_id)

    def make_neutral_window(self) -> tuple[np.ndarray, np.ndarray]:
        continuous_window = np.zeros((WINDOW_SIZE, len(INPUT_COLUMNS)), dtype=np.float32)
        categorical_window = np.zeros((WINDOW_SIZE, INVENTORY_SLOT_COUNT), dtype=np.int32)
        categorical_window.fill(self.filler_item_id)
        self._set_inventory_item(categorical_window, 0, self.sword_item_id)
        self._set_inventory_item(categorical_window, 2, self.sword_item_id)
        self._set(continuous_window, "health", 20.0, final_only=False)
        self._set(continuous_window, "foodLevel", 20.0, final_only=False)
        self._set(continuous_window, "isOnGround", 1.0, final_only=False)
        self._set(continuous_window, "targetDistance", 30.0, final_only=False)
        self._set(continuous_window, "targetHealth", 20.0, final_only=False)
        self._set(continuous_window, "targetRelX", 0.0, final_only=False)
        self._set(continuous_window, "targetRelY", 0.0, final_only=False)
        self._set(continuous_window, "targetRelZ", 1.0, final_only=False)
        return continuous_window, categorical_window

    @staticmethod
    def _set_inventory_item(categorical_window: np.ndarray, slot: int, item_id: int) -> None:
        if slot < 0 or slot >= categorical_window.shape[1]:
            raise IndexError(f"Inventory slot index {slot} is out of bounds for mock window.")
        categorical_window[:, slot] = int(item_id)

    def _set_hotbar_item(self, categorical_window: np.ndarray, hotbar_slot: int, item_id: int) -> None:
        self._set_inventory_item(categorical_window, hotbar_slot + 2, item_id)

    def _set(self, window: np.ndarray, feature: str, value: float, *, final_only: bool = False) -> bool:
        idx = self.feature_index.get(feature)
        if idx is None:
            return False
        if final_only:
            window[-1, idx] = value
        else:
            window[:, idx] = value
        return True

    def _expect_feature(self, feature: str, capability_name: str) -> None:
        if feature not in self.feature_index:
            self.missing_capabilities.append(capability_name)

    def _predict(self, continuous_window: np.ndarray, categorical_window: np.ndarray) -> dict[str, np.ndarray]:
        batched_continuous = np.expand_dims(continuous_window.astype(np.float32), axis=0)
        batched_categorical = np.expand_dims(categorical_window.astype(np.int32), axis=0)
        frame = pd.DataFrame(batched_continuous[0], columns=INPUT_COLUMNS)
        batched_continuous[0] = normalize_continuous_inputs(frame).to_numpy(dtype=np.float32)

        outputs = self.model(
            mx.array(batched_continuous, dtype=mx.float32),
            mx.array(batched_categorical, dtype=mx.int32),
        )
        return {
            "binary_probabilities": np.asarray(outputs["binary_probabilities"])[0],
            "slot_probabilities": np.asarray(outputs["slot_probabilities"])[0],
            "continuous_deltas": np.asarray(outputs["continuous_deltas"])[0],
        }

    def _bin(self, outputs: dict[str, np.ndarray], action: str) -> float:
        return float(outputs["binary_probabilities"][self.action_index[action]])

    @staticmethod
    def _slot(outputs: dict[str, np.ndarray]) -> int:
        return int(np.argmax(outputs["slot_probabilities"]))

    @staticmethod
    def _yaw(outputs: dict[str, np.ndarray]) -> float:
        return float(outputs["continuous_deltas"][0])

    @staticmethod
    def _pitch(outputs: dict[str, np.ndarray]) -> float:
        return float(outputs["continuous_deltas"][1])

    def _ok(self, condition: bool, name: str, details: str) -> ScenarioResult:
        return ScenarioResult(name=name, passed=condition, details=details)

    def scenario_chasing_enemy(self) -> ScenarioResult:
        continuous_window, categorical_window = self.make_neutral_window()
        self._set(continuous_window, "targetDistance", 15.0)
        self._set(continuous_window, "targetRelX", 0.0)
        self._set(continuous_window, "targetRelY", 0.0)
        self._set(continuous_window, "targetRelZ", 1.0)
        out = self._predict(continuous_window, categorical_window)

        forward = self._bin(out, "inputForward")
        sprint = self._bin(out, "inputSprint")
        passed = forward >= self.args.high_prob and sprint >= self.args.high_prob
        return self._ok(
            passed,
            "Step 2 - Chasing Enemy",
            f"forward={forward:.3f}, sprint={sprint:.3f}, threshold={self.args.high_prob:.2f}",
        )

    def scenario_melee_combat(self) -> ScenarioResult:
        continuous_window, categorical_window = self.make_neutral_window()
        closing_distance = np.linspace(6.0, 2.0, WINDOW_SIZE, dtype=np.float32)
        continuous_window[:, self.feature_index["targetDistance"]] = closing_distance
        self._set(continuous_window, "targetRelX", 0.0)
        continuous_window[:, self.feature_index["targetRelZ"]] = closing_distance
        self._set(continuous_window, "velZ", 0.2)
        self._set(continuous_window, "targetVelZ", -0.2)
        out = self._predict(continuous_window, categorical_window)

        lmb = self._bin(out, "inputLmb")
        passed = lmb >= self.args.high_prob
        return self._ok(passed, "Step 3 - Melee Combat", f"lmb={lmb:.3f}, threshold={self.args.high_prob:.2f}")

    def scenario_aiming(self) -> ScenarioResult:
        continuous_window, categorical_window = self.make_neutral_window()
        self._set(continuous_window, "targetRelX", 4.0)
        self._set(continuous_window, "targetRelY", 2.0)
        self._set(continuous_window, "targetRelZ", 6.0)
        out = self._predict(continuous_window, categorical_window)

        delta_yaw = self._yaw(out)
        delta_pitch = self._pitch(out)
        passed = abs(delta_yaw) > 0.01 or abs(delta_pitch) > 0.01
        return self._ok(
            passed,
            "Step 4 - Aiming",
            f"deltaYaw={delta_yaw:.3f}, deltaPitch={delta_pitch:.3f} (expected movement)",
        )

    def scenario_obstacle_jumping(self) -> ScenarioResult:
        continuous_window, categorical_window = self.make_neutral_window()
        self._set(continuous_window, "targetDistance", 1.414)
        self._set(continuous_window, "targetRelY", 1.0)
        self._set(continuous_window, "targetRelZ", 1.0)
        self._set(continuous_window, "velZ", 0.2, final_only=False)
        self._set(continuous_window, "velZ", 0.0, final_only=True)
        out = self._predict(continuous_window, categorical_window)

        jump = self._bin(out, "inputJump")
        passed = jump >= self.args.high_prob
        return self._ok(passed, "Step 5 - Obstacle Jumping", f"jump={jump:.3f}, threshold={self.args.high_prob:.2f}")

    def scenario_drinking_potion(self) -> ScenarioResult:
        continuous_window, categorical_window = self.make_neutral_window()
        continuous_window[:, self.feature_index["health"]] = np.linspace(10.0, 4.0, WINDOW_SIZE, dtype=np.float32)
        self._set(continuous_window, "targetDistance", 30.0)
        self._set_inventory_item(categorical_window, 0, self.potion_item_id)
        self._set_hotbar_item(categorical_window, self.args.drink_slot, self.potion_item_id)
        out = self._predict(continuous_window, categorical_window)

        slot = self._slot(out)
        rmb = self._bin(out, "inputRmb")
        passed = slot == self.args.drink_slot and rmb >= self.args.high_prob
        return self._ok(
            passed,
            "Step 6 - Drinking Potion",
            f"slot={slot}, expected={self.args.drink_slot}, rmb={rmb:.3f}",
        )

    def scenario_splash_potion_attack(self) -> ScenarioResult:
        continuous_window, categorical_window = self.make_neutral_window()
        self._set(continuous_window, "health", 12.0)
        self._set(continuous_window, "targetDistance", 6.0)
        self._set(continuous_window, "targetRelX", 2.0)
        self._set_inventory_item(categorical_window, 0, self.splash_potion_item_id)
        self._set_hotbar_item(categorical_window, self.args.splash_slot, self.splash_potion_item_id)
        out = self._predict(continuous_window, categorical_window)

        slot = self._slot(out)
        rmb = self._bin(out, "inputRmb")
        delta_yaw = self._yaw(out)
        passed = slot == self.args.splash_slot and rmb >= self.args.high_prob and delta_yaw > 0.0
        return self._ok(
            passed,
            "Step 7 - Splash Potion Attack",
            f"slot={slot}, expected={self.args.splash_slot}, rmb={rmb:.3f}, deltaYaw={delta_yaw:.3f}",
        )

    def scenario_splash_potion_self_heal(self) -> ScenarioResult:
        continuous_window, categorical_window = self.make_neutral_window()
        continuous_window[:, self.feature_index["health"]] = np.linspace(8.0, 2.0, WINDOW_SIZE, dtype=np.float32)
        self._set(continuous_window, "targetDistance", 3.0)
        self._set_inventory_item(categorical_window, 0, self.splash_potion_item_id)
        self._set_hotbar_item(categorical_window, self.args.splash_slot, self.splash_potion_item_id)
        out = self._predict(continuous_window, categorical_window)

        slot = self._slot(out)
        rmb = self._bin(out, "inputRmb")
        delta_pitch = self._pitch(out)
        passed = (
            slot == self.args.splash_slot
            and rmb >= self.args.high_prob
            and delta_pitch >= self.args.very_large_positive_pitch
        )
        return self._ok(
            passed,
            "Step 8 - Splash Potion Self Heal",
            (
                f"slot={slot}, expected={self.args.splash_slot}, rmb={rmb:.3f}, "
                f"deltaPitch={delta_pitch:.3f}, threshold={self.args.very_large_positive_pitch:.2f}"
            ),
        )

    def scenario_food_eating(self) -> ScenarioResult:
        self._expect_feature("foodLevel", "food_level_signal")

        continuous_window, categorical_window = self.make_neutral_window()
        continuous_window[:, self.feature_index["health"]] = np.linspace(18.0, 13.0, WINDOW_SIZE, dtype=np.float32)
        self._set(continuous_window, "targetDistance", 25.0)
        self._set_inventory_item(categorical_window, 0, self.food_item_id)
        self._set_hotbar_item(categorical_window, self.args.food_slot, self.food_item_id)
        out = self._predict(continuous_window, categorical_window)

        slot = self._slot(out)
        rmb = self._bin(out, "inputRmb")
        passed = slot == self.args.food_slot and rmb >= self.args.high_prob
        return self._ok(
            passed,
            "Step 9 - Food Eating",
            f"slot={slot}, expected={self.args.food_slot}, rmb={rmb:.3f}",
        )

    def scenario_golden_apple_prebuff(self) -> ScenarioResult:
        continuous_window, categorical_window = self.make_neutral_window()
        idx = self.feature_index.get("targetDistance")
        if idx is not None:
            continuous_window[:, idx] = np.linspace(30.0, 10.0, WINDOW_SIZE, dtype=np.float32)
        self._set(continuous_window, "targetVelZ", -1.0)
        self._set_inventory_item(categorical_window, 0, self.golden_apple_item_id)
        self._set_hotbar_item(categorical_window, self.args.golden_apple_slot, self.golden_apple_item_id)
        out = self._predict(continuous_window, categorical_window)

        slot = self._slot(out)
        rmb = self._bin(out, "inputRmb")
        passed = slot == self.args.golden_apple_slot and rmb >= self.args.high_prob
        return self._ok(
            passed,
            "Step 10 - Golden Apple Pre Buff",
            f"slot={slot}, expected={self.args.golden_apple_slot}, rmb={rmb:.3f}",
        )

    def scenario_sprint_reset(self) -> ScenarioResult:
        self._expect_feature("damageDealt", "damage_dealt_signal")

        first_continuous, first_categorical = self.make_neutral_window()
        self._set(first_continuous, "targetDistance", 2.0)
        self._set(first_continuous, "targetRelZ", 2.0)
        self._set(first_continuous, "targetHealth", 20.0)
        out1 = self._predict(first_continuous, first_categorical)

        second_continuous, second_categorical = self.make_neutral_window()
        self._set(second_continuous, "targetDistance", 2.5)
        self._set(second_continuous, "targetRelZ", 2.5)
        self._set(second_continuous, "targetHealth", 18.0)
        out2 = self._predict(second_continuous, second_categorical)

        sprint_1 = self._bin(out1, "inputSprint")
        sprint_2 = self._bin(out2, "inputSprint")
        forward_1 = self._bin(out1, "inputForward")
        forward_2 = self._bin(out2, "inputForward")

        drop_detected = sprint_1 <= self.args.drop_prob or forward_1 <= self.args.drop_prob
        rise_detected = sprint_2 >= self.args.rise_prob or forward_2 >= self.args.rise_prob
        passed = drop_detected and rise_detected
        return self._ok(
            passed,
            "Step 11 - Sprint Reset",
            (
                f"sprint: {sprint_1:.3f}->{sprint_2:.3f}, "
                f"forward: {forward_1:.3f}->{forward_2:.3f}, "
                "targetHealth: 20.0->18.0, "
                f"drop<={self.args.drop_prob:.2f}, rise>={self.args.rise_prob:.2f}"
            ),
        )

    def scenario_block_hitting(self) -> ScenarioResult:
        first_continuous, first_categorical = self.make_neutral_window()
        self._set(first_continuous, "targetDistance", 2.0)
        self._set(first_continuous, "targetRelX", -1.0)
        self._set(first_continuous, "targetRelZ", 1.732)
        out1 = self._predict(first_continuous, first_categorical)

        second_continuous, second_categorical = self.make_neutral_window()
        self._set(second_continuous, "targetDistance", 2.0)
        self._set(second_continuous, "targetRelX", 1.0)
        self._set(second_continuous, "targetRelZ", 1.732)
        out2 = self._predict(second_continuous, second_categorical)

        lmb1 = self._bin(out1, "inputLmb")
        rmb1 = self._bin(out1, "inputRmb")
        lmb2 = self._bin(out2, "inputLmb")
        rmb2 = self._bin(out2, "inputRmb")

        has_high_lmb = max(lmb1, lmb2) >= self.args.high_prob
        has_high_rmb = max(rmb1, rmb2) >= self.args.high_prob
        switches_dominant_hand = (lmb1 > rmb1 and rmb2 > lmb2) or (rmb1 > lmb1 and lmb2 > rmb2)
        passed = has_high_lmb and has_high_rmb and switches_dominant_hand
        return self._ok(
            passed,
            "Step 12 - Block Hitting",
            (
                f"lmb=({lmb1:.3f},{lmb2:.3f}), rmb=({rmb1:.3f},{rmb2:.3f}), "
                f"threshold={self.args.high_prob:.2f}"
            ),
        )

    def scenario_projectile_dodging(self) -> ScenarioResult:
        self._expect_feature("nearestProjectileDx", "projectile_tracking_signal")
        self._expect_feature("nearestProjectileDy", "projectile_tracking_signal")
        self._expect_feature("nearestProjectileDz", "projectile_tracking_signal")

        continuous_window, categorical_window = self.make_neutral_window()
        self._set(continuous_window, "targetDistance", 12.0)
        self._set(continuous_window, "targetRelX", 4.0)
        self._set(continuous_window, "targetRelZ", 8.0)
        self._set(continuous_window, "nearestProjectileDx", -1.5)
        self._set(continuous_window, "nearestProjectileDy", 0.3)
        self._set(continuous_window, "nearestProjectileDz", 2.0)
        out = self._predict(continuous_window, categorical_window)

        move_left = self._bin(out, "inputLeft")
        move_right = self._bin(out, "inputRight")
        passed = max(move_left, move_right) >= self.args.high_prob
        return self._ok(
            passed,
            "Step 13 - Projectile Dodging",
            f"left={move_left:.3f}, right={move_right:.3f}, threshold={self.args.high_prob:.2f}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario checks against the trained Phase 4 model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "phase4_best_weights.npz",
        help="Path to the trained MLX checkpoint (.npz).",
    )
    parser.add_argument(
        "--item-vocab",
        type=Path,
        default=EXPORTS_DIR / "phase2_item_vocabulary.json",
        help="Path to the Phase 2 exported item vocabulary JSON.",
    )
    parser.add_argument("--high-prob", type=float, default=0.01)
    parser.add_argument("--drop-prob", type=float, default=0.05)
    parser.add_argument("--rise-prob", type=float, default=0.1)
    parser.add_argument("--very-large-positive-pitch", type=float, default=0.5)
    parser.add_argument("--drink-slot", type=int, default=6)
    parser.add_argument("--splash-slot", type=int, default=6)
    parser.add_argument("--food-slot", type=int, default=6)
    parser.add_argument("--golden-apple-slot", type=int, default=3)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Do not fail fast when assertions fail; prints results and exits 0.",
    )
    return parser.parse_args()


def load_item_vocabulary(item_vocab_path: Path) -> dict[str, int]:
    with item_vocab_path.open("r", encoding="utf-8") as handle:
        raw_vocabulary = json.load(handle)

    if not isinstance(raw_vocabulary, dict):
        raise ValueError(f"Item vocabulary must be a JSON object: {item_vocab_path}")

    vocabulary: dict[str, int] = {}
    for key, value in raw_vocabulary.items():
        vocabulary[str(key)] = int(value)
    return vocabulary


def infer_checkpoint_item_vocab_size(checkpoint_path: Path) -> int | None:
    with np.load(checkpoint_path, allow_pickle=False) as weights:
        embedding = weights.get("item_embedding.weight")
        if embedding is None:
            return None
        return int(embedding.shape[0])


def build_model(checkpoint_path: Path, item_vocabulary: dict[str, int]) -> PvPSequenceModel:
    checkpoint_vocab_size = infer_checkpoint_item_vocab_size(checkpoint_path)
    vocabulary_id_span = max(item_vocabulary.values(), default=0) + 1
    effective_vocab_size = max(
        len(item_vocabulary),
        vocabulary_id_span,
        checkpoint_vocab_size or 0,
    )

    model = PvPSequenceModel(
        input_feature_count=len(INPUT_COLUMNS),
        boolean_action_count=len(BINARY_ACTION_COLUMNS),
        item_slot_count=INVENTORY_SLOT_COUNT,
        item_vocabulary_size=effective_vocab_size,
    )
    model.load_weights(str(checkpoint_path), strict=True)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.item_vocab.exists():
        raise FileNotFoundError(f"Item vocabulary not found: {args.item_vocab}")

    item_vocabulary = load_item_vocabulary(args.item_vocab)
    checkpoint_vocab_size = infer_checkpoint_item_vocab_size(args.checkpoint)
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != len(item_vocabulary):
        print(
            "[WARN] Item vocabulary entry count does not match checkpoint embedding rows: "
            f"vocab={len(item_vocabulary)} vs checkpoint={checkpoint_vocab_size}. "
            "Using checkpoint-compatible model size while preserving vocabulary IDs for scenario mocks."
        )
    model = build_model(args.checkpoint, item_vocabulary)

    runner = ScenarioRunner(model=model, args=args, item_vocabulary=item_vocabulary)
    checks: list[Callable[[], ScenarioResult]] = [
        runner.scenario_chasing_enemy,
        runner.scenario_melee_combat,
        runner.scenario_aiming,
        runner.scenario_obstacle_jumping,
        runner.scenario_drinking_potion,
        runner.scenario_splash_potion_attack,
        runner.scenario_splash_potion_self_heal,
        runner.scenario_food_eating,
        runner.scenario_golden_apple_prebuff,
        runner.scenario_sprint_reset,
        runner.scenario_block_hitting,
        runner.scenario_projectile_dodging,
    ]

    # Step 1 helper validation: ensure neutral mock windows have expected dual-input shapes.
    neutral_continuous, neutral_categorical = runner.make_neutral_window()
    assert neutral_continuous.shape == (
        WINDOW_SIZE,
        len(INPUT_COLUMNS),
    ), "Continuous neutral window helper must return 20-frame windows"
    assert neutral_categorical.shape == (
        WINDOW_SIZE,
        INVENTORY_SLOT_COUNT,
    ), "Categorical neutral window helper must return 20x38 windows"

    results = [check() for check in checks]

    failed = [result for result in results if not result.passed]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.details}")

    if runner.missing_capabilities:
        unique = sorted(set(runner.missing_capabilities))
        print("[WARN] Dataset features do not expose the following explicit scenario signals:")
        for capability in unique:
            print(f"  - {capability}")
        print("       Some checks use proxy signals because these features are absent from INPUT_COLUMNS.")

    if failed and not args.allow_failures:
        failed_names = ", ".join(result.name for result in failed)
        raise AssertionError(f"Scenario assertions failed: {failed_names}")

    print(f"Completed {len(results)} scenario checks with {len(failed)} failure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


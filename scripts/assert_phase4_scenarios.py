"""Run scenario-based sanity checks against the trained Phase 4 MLX model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import mlx.core as mx

import _bootstrap  # noqa: F401

from bot_training.config import CHECKPOINTS_DIR
from bot_training.features.build_features import (
    BINARY_ACTION_COLUMNS,
    INPUT_COLUMNS,
    normalize_continuous_inputs,
)
from bot_training.models.pvp_sequence_model import PvPSequenceModel


WINDOW_SIZE = 20


@dataclass(slots=True)
class ScenarioResult:
    name: str
    passed: bool
    details: str


class ScenarioRunner:
    def __init__(self, model: PvPSequenceModel, args: argparse.Namespace) -> None:
        self.model = model
        self.args = args
        self.feature_index = {name: idx for idx, name in enumerate(INPUT_COLUMNS)}
        self.action_index = {name: idx for idx, name in enumerate(BINARY_ACTION_COLUMNS)}
        self.missing_capabilities: list[str] = []

    def make_neutral_window(self) -> np.ndarray:
        window = np.zeros((WINDOW_SIZE, len(INPUT_COLUMNS)), dtype=np.float32)
        self._set(window, "health", 20.0, final_only=False)
        self._set(window, "isOnGround", 1.0, final_only=False)
        self._set(window, "targetDistance", 30.0, final_only=False)
        self._set(window, "targetHealth", 20.0, final_only=False)
        self._set(window, "targetRelX", 0.0, final_only=False)
        self._set(window, "targetRelY", 0.0, final_only=False)
        self._set(window, "targetRelZ", 1.0, final_only=False)
        return window

    def _set(self, window: np.ndarray, feature: str, value: float, *, final_only: bool = True) -> bool:
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

    def _predict(self, window: np.ndarray) -> dict[str, np.ndarray]:
        batched = np.expand_dims(window.astype(np.float32), axis=0)
        frame = pd.DataFrame(batched[0], columns=INPUT_COLUMNS)
        batched[0] = normalize_continuous_inputs(frame).to_numpy(dtype=np.float32)

        outputs = self.model(mx.array(batched, dtype=mx.float32))
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
        window = self.make_neutral_window()
        self._set(window, "targetDistance", 15.0)
        self._set(window, "targetRelX", 0.0)
        self._set(window, "targetRelY", 0.0)
        self._set(window, "targetRelZ", 1.0)
        out = self._predict(window)

        forward = self._bin(out, "inputForward")
        sprint = self._bin(out, "inputSprint")
        passed = forward >= self.args.high_prob and sprint >= self.args.high_prob
        return self._ok(
            passed,
            "Step 2 - Chasing Enemy",
            f"forward={forward:.3f}, sprint={sprint:.3f}, threshold={self.args.high_prob:.2f}",
        )

    def scenario_melee_combat(self) -> ScenarioResult:
        window = self.make_neutral_window()
        self._set(window, "targetDistance", 2.0)
        self._set(window, "targetRelX", 0.0)
        self._set(window, "targetRelZ", 1.0)
        out = self._predict(window)

        lmb = self._bin(out, "inputLmb")
        passed = lmb >= self.args.high_prob
        return self._ok(passed, "Step 3 - Melee Combat", f"lmb={lmb:.3f}, threshold={self.args.high_prob:.2f}")

    def scenario_aiming(self) -> ScenarioResult:
        window = self.make_neutral_window()
        self._set(window, "targetRelX", 4.0)
        self._set(window, "targetRelY", 2.0)
        self._set(window, "targetRelZ", 6.0)
        out = self._predict(window)

        delta_yaw = self._yaw(out)
        delta_pitch = self._pitch(out)
        passed = delta_yaw > 0.0 and delta_pitch > 0.0
        return self._ok(
            passed,
            "Step 4 - Aiming",
            f"deltaYaw={delta_yaw:.3f}, deltaPitch={delta_pitch:.3f} (expected positive)",
        )

    def scenario_obstacle_jumping(self) -> ScenarioResult:
        window = self.make_neutral_window()
        self._set(window, "targetDistance", 4.0)
        self._set(window, "targetRelY", 5.0)
        out = self._predict(window)

        jump = self._bin(out, "inputJump")
        passed = jump >= self.args.high_prob
        return self._ok(passed, "Step 5 - Obstacle Jumping", f"jump={jump:.3f}, threshold={self.args.high_prob:.2f}")

    def scenario_drinking_potion(self) -> ScenarioResult:
        window = self.make_neutral_window()
        self._set(window, "health", 4.0)
        self._set(window, "targetDistance", 30.0)
        out = self._predict(window)

        slot = self._slot(out)
        rmb = self._bin(out, "inputRmb")
        passed = slot == self.args.drink_slot and rmb >= self.args.high_prob
        return self._ok(
            passed,
            "Step 6 - Drinking Potion",
            f"slot={slot}, expected={self.args.drink_slot}, rmb={rmb:.3f}",
        )

    def scenario_splash_potion_attack(self) -> ScenarioResult:
        window = self.make_neutral_window()
        self._set(window, "health", 12.0)
        self._set(window, "targetDistance", 6.0)
        self._set(window, "targetRelX", 2.0)
        out = self._predict(window)

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
        window = self.make_neutral_window()
        self._set(window, "health", 2.0)
        self._set(window, "targetDistance", 3.0)
        out = self._predict(window)

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

        window = self.make_neutral_window()
        self._set(window, "health", 13.0)
        self._set(window, "targetDistance", 25.0)
        out = self._predict(window)

        slot = self._slot(out)
        rmb = self._bin(out, "inputRmb")
        passed = slot == self.args.food_slot and rmb >= self.args.high_prob
        return self._ok(
            passed,
            "Step 9 - Food Eating",
            f"slot={slot}, expected={self.args.food_slot}, rmb={rmb:.3f}",
        )

    def scenario_golden_apple_prebuff(self) -> ScenarioResult:
        window = self.make_neutral_window()
        idx = self.feature_index.get("targetDistance")
        if idx is not None:
            window[:, idx] = np.linspace(30.0, 10.0, WINDOW_SIZE, dtype=np.float32)
        self._set(window, "targetVelZ", -1.0)
        out = self._predict(window)

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

        first = self.make_neutral_window()
        self._set(first, "targetDistance", 2.0)
        self._set(first, "targetRelZ", 1.0)
        out1 = self._predict(first)

        second = self.make_neutral_window()
        self._set(second, "targetDistance", 2.5)
        self._set(second, "targetRelZ", 1.0)
        out2 = self._predict(second)

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
                f"drop<={self.args.drop_prob:.2f}, rise>={self.args.rise_prob:.2f}"
            ),
        )

    def scenario_block_hitting(self) -> ScenarioResult:
        first = self.make_neutral_window()
        self._set(first, "targetDistance", 2.0)
        self._set(first, "targetRelX", -1.0)
        out1 = self._predict(first)

        second = self.make_neutral_window()
        self._set(second, "targetDistance", 2.0)
        self._set(second, "targetRelX", 1.0)
        out2 = self._predict(second)

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
        self._expect_feature("nearestProjectileDeltaX", "projectile_tracking_signal")
        self._expect_feature("nearestProjectileDeltaY", "projectile_tracking_signal")
        self._expect_feature("nearestProjectileDeltaZ", "projectile_tracking_signal")

        window = self.make_neutral_window()
        self._set(window, "targetDistance", 12.0)
        self._set(window, "targetRelX", 4.0)
        self._set(window, "targetRelZ", 8.0)
        out = self._predict(window)

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
    parser.add_argument("--high-prob", type=float, default=0.8)
    parser.add_argument("--drop-prob", type=float, default=0.1)
    parser.add_argument("--rise-prob", type=float, default=0.9)
    parser.add_argument("--very-large-positive-pitch", type=float, default=0.5)
    parser.add_argument("--drink-slot", type=int, default=1)
    parser.add_argument("--splash-slot", type=int, default=2)
    parser.add_argument("--food-slot", type=int, default=3)
    parser.add_argument("--golden-apple-slot", type=int, default=4)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Do not fail fast when assertions fail; prints results and exits 0.",
    )
    return parser.parse_args()


def build_model(checkpoint_path: Path) -> PvPSequenceModel:
    model = PvPSequenceModel(
        input_feature_count=len(INPUT_COLUMNS),
        boolean_action_count=len(BINARY_ACTION_COLUMNS),
    )
    model.load_weights(str(checkpoint_path), strict=True)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = build_model(args.checkpoint)

    runner = ScenarioRunner(model=model, args=args)
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

    # Step 1 helper validation: ensure neutral mock windows have shape [20, feature_count].
    neutral = runner.make_neutral_window()
    assert neutral.shape == (WINDOW_SIZE, len(INPUT_COLUMNS)), "Neutral window helper must return 20-frame windows"

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


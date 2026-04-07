"""FastAPI inference server for the Phase 4 PvP sequence model."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from bot_training.features.build_features import (
    BINARY_ACTION_COLUMNS,
    INPUT_COLUMNS,
    ITEM_SLOT_COLUMNS,
    normalize_continuous_inputs,
)
from bot_training.models.pvp_sequence_model import PvPSequenceModel


WINDOW_SIZE = 20
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "phase4_best_weights.npz"
ITEM_VOCAB_PATH = PROJECT_ROOT / "models" / "exports" / "phase2_item_vocabulary.json"

app = FastAPI(title="PvP Sequence Model Inference API")

# Rolling per-bot processed frame buffers used to build 20-frame model windows.
state_buffers: dict[str, list] = {}

model: PvPSequenceModel | None = None
item_vocabulary: dict[str, int] = {}
air_item_id: int = 0


class RawEntityState(BaseModel):
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    vel_x: float
    vel_y: float
    vel_z: float
    health: float
    food: float = 20.0
    is_on_ground: bool = True


class InventoryState(BaseModel):
    main_hand: str
    off_hand: str
    hotbar: list[str]

    @field_validator("hotbar")
    @classmethod
    def _validate_hotbar(cls, value: list[str]) -> list[str]:
        if len(value) != 9:
            raise ValueError("hotbar must contain exactly 9 item names")
        return value


class GameState(BaseModel):
    bot_id: str
    bot: RawEntityState
    target: RawEntityState
    inventory: InventoryState


class BotPrediction(BaseModel):
    deltaYaw: float
    deltaPitch: float
    inputForward: float
    inputBackward: float
    inputLeft: float
    inputRight: float
    inputJump: float
    inputSneak: float
    inputSprint: float
    inputLmb: float
    inputRmb: float
    inputSlot: int


def _load_item_vocabulary(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Item vocabulary must be a JSON object: {path}")
    return {str(key): int(value) for key, value in raw.items()}


def _infer_checkpoint_vocab_size(path: Path) -> int | None:
    with np.load(path, allow_pickle=False) as weights:
        embedding = weights.get("item_embedding.weight")
        if embedding is None:
            return None
        return int(embedding.shape[0])


def _normalize_item_name(name: str) -> str:
    return name.strip()


def _calculate_wrapped_yaw_delta(current_yaw: float, previous_yaw: float) -> float:
    return ((current_yaw - previous_yaw + 180.0) % 360.0) - 180.0


def _state_to_continuous_row(state: GameState) -> tuple[list[float], float, float]:
    dx = float(state.target.x - state.bot.x)
    dy = float(state.target.y - state.bot.y)
    dz = float(state.target.z - state.bot.z)
    target_distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))

    # These deltas are computed for live-tick continuity and can be exposed later if needed.
    previous_entry = state_buffers.get(state.bot_id, [])
    if previous_entry:
        previous_state = previous_entry[-1]["raw_state"]
        delta_yaw = _calculate_wrapped_yaw_delta(float(state.bot.yaw), float(previous_state.bot.yaw))
        delta_pitch = float(state.bot.pitch - previous_state.bot.pitch)
    else:
        delta_yaw = 0.0
        delta_pitch = 0.0

    feature_values = {
        "health": float(state.bot.health),
        "foodLevel": float(state.bot.food),
        "damageDealt": 0.0,
        "velX": float(state.bot.vel_x),
        "velY": float(state.bot.vel_y),
        "velZ": float(state.bot.vel_z),
        "yaw": float(state.bot.yaw),
        "pitch": float(state.bot.pitch),
        "isOnGround": 1.0 if state.bot.is_on_ground else 0.0,
        "targetDistance": target_distance,
        "targetRelX": dx,
        "targetRelY": dy,
        "targetRelZ": dz,
        "nearestProjectileDx": 0.0,
        "nearestProjectileDy": 0.0,
        "nearestProjectileDz": 0.0,
        "targetYaw": float(state.target.yaw),
        "targetPitch": float(state.target.pitch),
        "targetVelX": float(state.target.vel_x),
        "targetVelY": float(state.target.vel_y),
        "targetVelZ": float(state.target.vel_z),
        "targetHealth": float(state.target.health),
    }
    return [float(feature_values[name]) for name in INPUT_COLUMNS], delta_yaw, delta_pitch


def _state_to_categorical_row(state: GameState) -> list[int]:
    bag = ["AIR"] * 27
    item_names = [
        state.inventory.main_hand,
        state.inventory.off_hand,
        *state.inventory.hotbar,
        *bag,
    ]

    ids: list[int] = []
    for item_name in item_names:
        normalized = _normalize_item_name(str(item_name))
        ids.append(int(item_vocabulary.get(normalized, air_item_id)))
    return ids


@app.on_event("startup")
def startup_event() -> None:
    global model, item_vocabulary, air_item_id

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not ITEM_VOCAB_PATH.exists():
        raise FileNotFoundError(f"Item vocabulary not found: {ITEM_VOCAB_PATH}")

    item_vocabulary = _load_item_vocabulary(ITEM_VOCAB_PATH)
    air_item_id = int(item_vocabulary.get("AIR", item_vocabulary.get("", 0)))

    checkpoint_vocab_size = _infer_checkpoint_vocab_size(CHECKPOINT_PATH)
    vocabulary_id_span = max(item_vocabulary.values(), default=0) + 1
    effective_vocab_size = max(
        len(item_vocabulary),
        vocabulary_id_span,
        checkpoint_vocab_size or 0,
    )

    model = PvPSequenceModel(
        input_feature_count=len(INPUT_COLUMNS),
        boolean_action_count=len(BINARY_ACTION_COLUMNS),
        item_slot_count=len(ITEM_SLOT_COLUMNS),
        item_vocabulary_size=effective_vocab_size,
    )
    model.load_weights(str(CHECKPOINT_PATH), strict=True)
    model.eval()


@app.post("/predict", response_model=BotPrediction)
def predict(state: GameState) -> BotPrediction:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    continuous_row, delta_yaw, delta_pitch = _state_to_continuous_row(state)
    categorical_row = _state_to_categorical_row(state)

    history = state_buffers.setdefault(state.bot_id, [])
    history.append(
        {
            "raw_state": state,
            "continuous": continuous_row,
            "categorical": categorical_row,
            "deltaYaw": delta_yaw,
            "deltaPitch": delta_pitch,
        }
    )
    if len(history) > WINDOW_SIZE:
        history.pop(0)

    # Left-pad short histories by repeating the first observed frame.
    window_entries = list(history)
    if len(window_entries) < WINDOW_SIZE:
        padding = [window_entries[0]] * (WINDOW_SIZE - len(window_entries))
        window_entries = padding + window_entries

    continuous_window = np.asarray(
        [entry["continuous"] for entry in window_entries],
        dtype=np.float32,
    )
    categorical_window = np.asarray(
        [entry["categorical"] for entry in window_entries],
        dtype=np.int32,
    )

    frame_df = pd.DataFrame(continuous_window, columns=INPUT_COLUMNS)
    continuous_window = normalize_continuous_inputs(frame_df).to_numpy(dtype=np.float32)

    outputs = model(
        mx.array(np.expand_dims(continuous_window, axis=0), dtype=mx.float32),
        mx.array(np.expand_dims(categorical_window, axis=0), dtype=mx.int32),
    )

    # Model heads are computed from the final timestep internally.
    binary_probabilities = np.asarray(outputs["binary_probabilities"])[0]
    slot_probabilities = np.asarray(outputs["slot_probabilities"])[0]
    continuous_deltas = np.asarray(outputs["continuous_deltas"])[0]

    return BotPrediction(
        deltaYaw=float(continuous_deltas[0]),
        deltaPitch=float(continuous_deltas[1]),
        inputForward=float(binary_probabilities[0]),
        inputBackward=float(binary_probabilities[1]),
        inputLeft=float(binary_probabilities[2]),
        inputRight=float(binary_probabilities[3]),
        inputJump=float(binary_probabilities[4]),
        inputSneak=float(binary_probabilities[5]),
        inputSprint=float(binary_probabilities[6]),
        inputLmb=float(binary_probabilities[7]),
        inputRmb=float(binary_probabilities[8]),
        inputSlot=int(np.argmax(slot_probabilities)),
    )



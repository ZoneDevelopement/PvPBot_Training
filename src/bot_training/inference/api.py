"""FastAPI inference server for the Phase 4 PvP sequence model."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

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

# Rolling per-bot history buffers used to build 20-frame model windows.
state_buffers: dict[str, list] = {}

model: PvPSequenceModel | None = None
item_vocabulary: dict[str, int] = {}
unknown_item_id: int = 1


class GameState(BaseModel):
    bot_id: str

    # Continuous model inputs
    health: float
    foodLevel: float
    damageDealt: float
    velX: float
    velY: float
    velZ: float
    yaw: float
    pitch: float
    isOnGround: bool
    targetDistance: float
    targetRelX: float
    targetRelY: float
    targetRelZ: float
    nearestProjectileDx: float
    nearestProjectileDy: float
    nearestProjectileDz: float
    targetYaw: float
    targetPitch: float
    targetVelX: float
    targetVelY: float
    targetVelZ: float
    targetHealth: float

    # Categorical inventory inputs (2 direct-hand + 9 hotbar + 27 bag slots = 38)
    mainHandItem: str
    offHandItem: str
    hotbar0: str
    hotbar1: str
    hotbar2: str
    hotbar3: str
    hotbar4: str
    hotbar5: str
    hotbar6: str
    hotbar7: str
    hotbar8: str
    inventoryBag: list[str] = Field(default_factory=lambda: [""] * 27)

    @field_validator("inventoryBag")
    @classmethod
    def _validate_inventory_bag(cls, value: list[str]) -> list[str]:
        if len(value) != 27:
            raise ValueError("inventoryBag must contain exactly 27 item names")
        return value


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


def _state_to_continuous_row(state: GameState) -> list[float]:
    return [
        float(state.health),
        float(state.foodLevel),
        float(state.damageDealt),
        float(state.velX),
        float(state.velY),
        float(state.velZ),
        float(state.yaw),
        float(state.pitch),
        1.0 if state.isOnGround else 0.0,
        float(state.targetDistance),
        float(state.targetRelX),
        float(state.targetRelY),
        float(state.targetRelZ),
        float(state.nearestProjectileDx),
        float(state.nearestProjectileDy),
        float(state.nearestProjectileDz),
        float(state.targetYaw),
        float(state.targetPitch),
        float(state.targetVelX),
        float(state.targetVelY),
        float(state.targetVelZ),
        float(state.targetHealth),
    ]


def _state_to_categorical_row(state: GameState) -> list[int]:
    bag = list(state.inventoryBag)
    item_names = [
        state.mainHandItem,
        state.offHandItem,
        state.hotbar0,
        state.hotbar1,
        state.hotbar2,
        state.hotbar3,
        state.hotbar4,
        state.hotbar5,
        state.hotbar6,
        state.hotbar7,
        state.hotbar8,
        *bag,
    ]

    ids: list[int] = []
    for item_name in item_names:
        normalized = _normalize_item_name(str(item_name))
        ids.append(int(item_vocabulary.get(normalized, unknown_item_id)))
    return ids


@app.on_event("startup")
def startup_event() -> None:
    global model, item_vocabulary, unknown_item_id

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not ITEM_VOCAB_PATH.exists():
        raise FileNotFoundError(f"Item vocabulary not found: {ITEM_VOCAB_PATH}")

    item_vocabulary = _load_item_vocabulary(ITEM_VOCAB_PATH)
    unknown_item_id = int(item_vocabulary.get("UNKNOWN", 1))

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

    history = state_buffers.setdefault(state.bot_id, [])
    history.append(state)
    if len(history) > WINDOW_SIZE:
        history.pop(0)

    # Left-pad short histories by repeating the first observed frame.
    window_frames = list(history)
    if len(window_frames) < WINDOW_SIZE:
        padding = [window_frames[0]] * (WINDOW_SIZE - len(window_frames))
        window_frames = padding + window_frames

    continuous_window = np.asarray(
        [_state_to_continuous_row(frame) for frame in window_frames],
        dtype=np.float32,
    )
    categorical_window = np.asarray(
        [_state_to_categorical_row(frame) for frame in window_frames],
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



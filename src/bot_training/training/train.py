"""Phase 4 training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from bot_training.config import CHECKPOINTS_DIR, PROCESSED_DATA_DIR
from bot_training.features.build_features import INPUT_COLUMNS
from bot_training.models.pvp_sequence_model import PvPSequenceModel
from bot_training.training.phase4 import load_phase2_dataset, train_phase4_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 4 MLX PvP sequence model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROCESSED_DATA_DIR / "phase2_feature_tensors_per_file",
        help="Path to the Phase 2 NPZ directory or a specific NPZ file to train on.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "phase4_best_weights.npz",
        help="Path where the best weights should be saved.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs to train (capped at 50).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    dataset = load_phase2_dataset(args.dataset)
    model = PvPSequenceModel(
        input_feature_count=len(INPUT_COLUMNS),
        boolean_action_count=9,
    )
    result = train_phase4_model(
        dataset,
        model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    print(f"Best validation loss: {result.best_validation_loss:.6f}")
    if result.checkpoint_path is not None:
        print(f"Saved checkpoint: {result.checkpoint_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


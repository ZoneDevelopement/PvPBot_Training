"""Rebuild Phase 2 artifacts and retrain the Phase 4 model in one command."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from bot_training.config import CHECKPOINTS_DIR, EXPORTS_DIR, PROCESSED_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete generated Phase 2/Phase 4 artifacts, regenerate Phase 2 tensors, "
            "and retrain the Phase 4 model."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "phase1_clean_matches_per_file",
        help="Directory with cleaned Phase 1 CSVs.",
    )
    parser.add_argument("--input-pattern", default="*_clean.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "phase2_feature_tensors_per_file",
        help="Phase 2 per-file NPZ output directory.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "phase2_feature_manifest.json",
        help="Phase 2 manifest JSON output path.",
    )
    parser.add_argument(
        "--vocabulary-file",
        type=Path,
        default=EXPORTS_DIR / "phase2_item_vocabulary.json",
        help="Global Phase 2 vocabulary JSON output path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "phase4_best_weights.npz",
        help="Phase 4 checkpoint output path.",
    )
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for Phase 2 build input files (useful for quick experiments).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without deleting files or launching subprocesses.",
    )
    return parser.parse_args()


def _remove_path(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return

    if dry_run:
        print(f"[DRY-RUN] delete: {path}")
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"Deleted: {path}")


def _run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    if dry_run:
        print(f"[DRY-RUN] run: {printable}")
        return

    print(f"Running: {printable}")
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    cleanup_targets = [
        args.output_dir,
        args.manifest_file,
        args.vocabulary_file,
        PROCESSED_DATA_DIR / "phase2_feature_tensors.npz",
        args.checkpoint,
    ]

    print("Step 1/3 - Removing old Phase 2 and Phase 4 artifacts...")
    for target in cleanup_targets:
        _remove_path(target, dry_run=args.dry_run)

    build_command = [
        sys.executable,
        str(project_root / "scripts" / "build_features.py"),
        "--input-dir",
        str(args.input_dir),
        "--input-pattern",
        args.input_pattern,
        "--output-dir",
        str(args.output_dir),
        "--manifest-file",
        str(args.manifest_file),
        "--vocabulary-file",
        str(args.vocabulary_file),
        "--window-size",
        str(args.window_size),
    ]
    if args.max_files is not None:
        build_command.extend(["--max-files", str(args.max_files)])

    print("Step 2/3 - Regenerating Phase 2 feature tensors...")
    _run_command(build_command, dry_run=args.dry_run)

    train_command = [
        sys.executable,
        str(project_root / "scripts" / "train_model.py"),
        "--dataset",
        str(args.output_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--validation-ratio",
        str(args.validation_ratio),
        "--seed",
        str(args.seed),
    ]

    print("Step 3/3 - Training Phase 4 model...")
    _run_command(train_command, dry_run=args.dry_run)

    if args.dry_run:
        print("Done (dry-run).")
    else:
        print("Done. Phase 2 regenerated and Phase 4 retrained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


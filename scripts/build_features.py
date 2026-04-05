"""Build model-ready tensors from filtered Phase 1 PvP data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from bot_training.config import EXPORTS_DIR, PROCESSED_DATA_DIR
from bot_training.features.build_features import FeatureEngineeringResult, engineer_feature_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering pipeline for sequence model inputs.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Single filtered CSV produced by phase 1 cleaning.",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of per-file cleaned CSVs from phase 1.",
    )
    parser.add_argument(
        "--input-pattern",
        default="*_clean.csv",
        help="Glob pattern used when --input-dir is provided.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for number of discovered files when --input-dir is used.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "phase2_feature_tensors.npz",
        help="Destination NPZ for --input-file mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "phase2_feature_tensors_per_file",
        help="Destination directory for batch NPZ files when --input-dir is used.",
    )
    parser.add_argument(
        "--scaler-file",
        type=Path,
        default=EXPORTS_DIR / "phase2_minmax_scaler.joblib",
        help="Destination scaler path for --input-file mode.",
    )
    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=EXPORTS_DIR / "phase2_item_vocabulary.json",
        help="Destination item vocabulary JSON for --input-file mode.",
    )
    parser.add_argument(
        "--scaler-dir",
        type=Path,
        default=EXPORTS_DIR / "phase2_scalers_per_file",
        help="Destination directory for batch scaler artifacts when --input-dir is used.",
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=EXPORTS_DIR / "phase2_vocabs_per_file",
        help="Destination directory for per-file item vocabulary JSON artifacts.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "phase2_feature_manifest.json",
        help="JSON summary written in --input-dir mode.",
    )
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window length for sequence tensors.")
    return parser.parse_args()


def _read_dataframe(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)  # type: ignore[call-arg]
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame from input CSV")
    return dataframe


def _save_result_arrays(output_file: Path, result: FeatureEngineeringResult) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        inputs=result.inputs,
        categorical_inputs=result.categorical_inputs,
        targets=result.targets,
        input_windows=result.input_windows,
        categorical_windows=result.categorical_windows,
        sequence_targets=result.sequence_targets,
        window_match_ids=result.window_match_ids,
    )


def _run_single_file(
    input_file: Path,
    output_file: Path,
    scaler_file: Path,
    vocab_file: Path,
    window_size: int,
) -> None:
    dataframe = _read_dataframe(input_file)
    result: FeatureEngineeringResult = engineer_feature_tensors(
        dataframe,
        window_size=window_size,
        scaler_path=scaler_file,
        vocabulary_path=vocab_file,
    )
    _save_result_arrays(output_file, result)
    print(f"Rows transformed: {result.inputs.shape[0]}")
    print(f"Feature width: {result.inputs.shape[1]}")
    print(f"Categorical feature width: {result.categorical_inputs.shape[1]}")
    print(f"Windows: {result.input_windows.shape[0]}")
    print(f"Saved tensors: {output_file.resolve()}")
    print(f"Saved scaler: {scaler_file.resolve()}")
    print(f"Saved item vocabulary: {vocab_file.resolve()}")


def _discover_input_files(input_dir: Path, pattern: str, max_files: int | None) -> list[Path]:
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if max_files is not None:
        return files[:max_files]
    return files


def _run_batch(args: argparse.Namespace) -> int:
    input_files = _discover_input_files(args.input_dir, args.input_pattern, args.max_files)
    if not input_files:
        raise FileNotFoundError(
            f"No input files found in {args.input_dir} with pattern '{args.input_pattern}'."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scaler_dir.mkdir(parents=True, exist_ok=True)
    args.vocab_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for index, input_file in enumerate(input_files, start=1):
        stem = input_file.stem
        output_file = args.output_dir / f"{stem}_features.npz"
        scaler_file = args.scaler_dir / f"{stem}_scaler.joblib"
        vocab_file = args.vocab_dir / f"{stem}_item_vocabulary.json"

        dataframe = _read_dataframe(input_file)
        result: FeatureEngineeringResult = engineer_feature_tensors(
            dataframe,
            window_size=args.window_size,
            scaler_path=scaler_file,
            vocabulary_path=vocab_file,
        )
        _save_result_arrays(output_file, result)

        manifest.append(
            {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "scaler_file": str(scaler_file),
                "vocabulary_file": str(vocab_file),
                "rows_transformed": int(result.inputs.shape[0]),
                "feature_width": int(result.inputs.shape[1]),
                "categorical_feature_width": int(result.categorical_inputs.shape[1]),
                "windows": int(result.input_windows.shape[0]),
                "unique_match_ids": int(len(set(result.window_match_ids.tolist()))) if result.window_match_ids.size else 0,
            }
        )
        print(f"[{index}/{len(input_files)}] {input_file.name} -> {output_file.name}")

    args.manifest_file.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Processed files: {len(input_files)}")
    print(f"Saved manifest: {args.manifest_file.resolve()}")
    print(f"Batch tensor dir: {args.output_dir.resolve()}")
    print(f"Batch scaler dir: {args.scaler_dir.resolve()}")
    print(f"Batch vocab dir: {args.vocab_dir.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    if args.input_file is not None:
        _run_single_file(
            input_file=args.input_file,
            output_file=args.output_file,
            scaler_file=args.scaler_file,
            vocab_file=args.vocab_file,
            window_size=args.window_size,
        )
        return 0

    return _run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())



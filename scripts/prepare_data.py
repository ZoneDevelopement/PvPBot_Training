"""Phase 1 CSV processor for Minecraft PvP match data."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bot_training.config import DATA_DIR, RAW_DATA_DIR
from bot_training.data.preprocessing import (
    Phase1Config,
    Phase1Progress,
    RejectionBreakdown,
    discover_csv_files,
    process_phase1_csv_file,
    process_phase1_csv_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked Phase 1 PvP data cleaner.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR, help="Directory containing raw CSV files.")
    parser.add_argument(
        "--output-mode",
        choices=["merged", "per-file"],
        default="merged",
        help="`merged` writes one combined output CSV, `per-file` writes one clean CSV per input file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DATA_DIR / "processed" / "phase1_clean_matches.csv",
        help="Clean CSV output path when --output-mode=merged.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "processed" / "phase1_clean_matches_per_file",
        help="Output directory when --output-mode=per-file.",
    )
    parser.add_argument("--chunksize", type=int, default=50_000, help="Rows to read per pandas chunk.")
    parser.add_argument(
        "--timestamp-gap",
        type=float,
        default=5_000.0,
        help="Gap threshold that starts a new match when timestamps jump beyond this amount.",
    )
    parser.add_argument("--min-frames", type=int, default=1_000, help="Minimum rows per match.")
    parser.add_argument(
        "--max-damage-taken",
        type=float,
        default=40.0,
        help="Discard matches where total damageTaken exceeds this threshold.",
    )
    parser.add_argument(
        "--min-attack-accuracy",
        type=float,
        default=0.40,
        help="Minimum hits/clicks ratio required to keep a match.",
    )
    parser.add_argument(
        "--min-sprint-uptime",
        type=float,
        default=0.60,
        help="Minimum fraction of frames with inputSprint=True.",
    )
    parser.add_argument("--timestamp-col", default="timestamp", help="Name of the timestamp column.")
    parser.add_argument("--player-col", default="playerName", help="Name of the player name column.")
    parser.add_argument("--damage-taken-col", default="damageTaken", help="Name of the damage taken column.")
    parser.add_argument("--damage-dealt-col", default="damageDealt", help="Name of the damage dealt column.")
    parser.add_argument("--input-lmb-col", default="inputLmb", help="Name of the left mouse button column.")
    parser.add_argument("--input-sprint-col", default="inputSprint", help="Name of the sprint input column.")
    parser.add_argument(
        "--split-on-player-change",
        action="store_true",
        help="Start a new candidate match whenever playerName changes. Disabled by default because many files alternate players each tick.",
    )
    parser.add_argument("--progress", action="store_true", help="Print periodic progress updates while processing.")
    parser.add_argument(
        "--progress-every-chunks",
        type=int,
        default=10,
        help="Emit progress every N chunks per file when --progress is enabled.",
    )
    return parser.parse_args()


def _format_rejection_reasons(reasons: RejectionBreakdown) -> str:
    return (
        f"min_frames={reasons.min_frames} "
        f"damage={reasons.damage} "
        f"accuracy={reasons.accuracy} "
        f"sprint={reasons.sprint}"
    )


def _progress_printer(progress: Phase1Progress) -> None:
    print(
        "[progress] "
        f"file {progress.file_index}/{progress.total_files} "
        f"({progress.current_file.name}) | "
        f"rows={progress.rows_read} candidates={progress.candidate_matches} "
        f"kept={progress.kept_matches} rejected={progress.rejected_matches} | "
        f"reasons: {_format_rejection_reasons(progress.rejection_reasons)}"
    )


def main() -> int:
    args = parse_args()
    config = Phase1Config(
        chunksize=args.chunksize,
        timestamp_gap=args.timestamp_gap,
        min_frames=args.min_frames,
        max_damage_taken=args.max_damage_taken,
        min_attack_accuracy=args.min_attack_accuracy,
        min_sprint_uptime=args.min_sprint_uptime,
        timestamp_col=args.timestamp_col,
        player_col=args.player_col,
        damage_taken_col=args.damage_taken_col,
        damage_dealt_col=args.damage_dealt_col,
        input_lmb_col=args.input_lmb_col,
        input_sprint_col=args.input_sprint_col,
        split_on_player_change=args.split_on_player_change,
    )
    csv_files = discover_csv_files(args.input_dir)
    progress_callback = _progress_printer if args.progress else None

    if args.output_mode == "merged":
        result = process_phase1_csv_files(
            args.input_dir,
            args.output_file,
            config,
            progress_callback=progress_callback,
            progress_every_chunks=args.progress_every_chunks,
        )
        print(
            "Processed "
            f"{result.files_processed} files, {result.rows_read} rows, "
            f"{result.candidate_matches} candidate matches -> "
            f"{result.kept_matches} kept / {result.rejected_matches} rejected."
        )
        print(f"Rejections by reason: {_format_rejection_reasons(result.rejection_reasons)}")
        print(f"Output mode: merged -> {args.output_file.resolve()}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_files = 0
    combined_rows = 0
    combined_candidates = 0
    combined_kept = 0
    combined_rejected = 0
    combined_reasons = RejectionBreakdown()
    for index, csv_file in enumerate(csv_files, start=1):
        per_file_output = args.output_dir / f"{csv_file.stem}_clean.csv"
        file_result, _ = process_phase1_csv_file(
            csv_file,
            per_file_output,
            config,
            append=False,
            start_match_id=1,
            progress_callback=progress_callback,
            file_index=index,
            total_files=len(csv_files),
            progress_every_chunks=args.progress_every_chunks,
        )
        combined_files += file_result.files_processed
        combined_rows += file_result.rows_read
        combined_candidates += file_result.candidate_matches
        combined_kept += file_result.kept_matches
        combined_rejected += file_result.rejected_matches
        combined_reasons.min_frames += file_result.rejection_reasons.min_frames
        combined_reasons.damage += file_result.rejection_reasons.damage
        combined_reasons.accuracy += file_result.rejection_reasons.accuracy
        combined_reasons.sprint += file_result.rejection_reasons.sprint

    print(
        "Processed "
        f"{combined_files} files, {combined_rows} rows, "
        f"{combined_candidates} candidate matches -> "
        f"{combined_kept} kept / {combined_rejected} rejected."
    )
    print(f"Rejections by reason: {_format_rejection_reasons(combined_reasons)}")
    print(f"Output mode: per-file -> {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


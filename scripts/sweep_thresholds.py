"""Run an automatic sweep over Phase 1 filtering thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401

from bot_training.config import DATA_DIR, METRICS_DIR, RAW_DATA_DIR
from bot_training.data.preprocessing import Phase1Config
from bot_training.data.threshold_sweep import (
    parse_float_grid,
    parse_int_grid,
    parse_score_weights,
    run_threshold_sweep,
    sample_csv_files,
    to_report_rows,
)
from bot_training.data.preprocessing import discover_csv_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Phase 1 thresholds and rank configurations.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR, help="Directory with raw CSV data.")
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=METRICS_DIR / "phase1_threshold_sweep.csv",
        help="Where to write the ranked sweep results CSV.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DATA_DIR / "interim" / "phase1_sweep_runs",
        help="Temporary output directory used during sweep runs.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many top configurations to print.")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of input CSV files to sample randomly for faster sweeps (e.g. 0.1 for 1/10).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic file sampling when --sample-fraction < 1.",
    )

    parser.add_argument("--min-frames-grid", default="400,700,1000", help="Comma-separated integer grid.")
    parser.add_argument("--max-damage-grid", default="40,50,60", help="Comma-separated float grid.")
    parser.add_argument("--min-attack-accuracy-grid", default="0.20,0.30,0.40", help="Comma-separated float grid.")
    parser.add_argument("--min-sprint-uptime-grid", default="0.15,0.30,0.60", help="Comma-separated float grid.")
    parser.add_argument(
        "--score-weights",
        default="0.25,0.25,0.25,0.25",
        help="Penalty weights: min_frames,damage,accuracy,sprint",
    )

    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--timestamp-gap", type=float, default=5_000.0)
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--player-col", default="playerName")
    parser.add_argument("--damage-taken-col", default="damageTaken")
    parser.add_argument("--damage-dealt-col", default="damageDealt")
    parser.add_argument("--input-lmb-col", default="inputLmb")
    parser.add_argument("--input-sprint-col", default="inputSprint")
    parser.add_argument("--split-on-player-change", action="store_true")

    parser.add_argument(
        "--keep-intermediate-outputs",
        action="store_true",
        help="Keep per-run cleaned CSV outputs in work-dir instead of deleting them.",
    )
    return parser.parse_args()


def write_report_csv(report_csv: Path, rows: list[dict[str, str]]) -> None:
    report_csv = report_csv.expanduser().resolve()
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top_runs(rows: list[dict[str, str]], top_k: int) -> None:
    top = rows[: max(top_k, 0)]
    if not top:
        print("No sweep rows to show.")
        return

    print("Top threshold configurations:")
    for row in top:
        print(
            f"#{row['rank']} score={row['quality_score']} keep_rate={row['keep_rate']} "
            f"kept={row['kept_matches']}/{row['candidate_matches']} "
            f"frames>={row['min_frames']} maxDmg<={row['max_damage_taken']} "
            f"acc>={row['min_attack_accuracy']} sprint>={row['min_sprint_uptime']}"
        )


def main() -> int:
    args = parse_args()

    min_frames_grid = parse_int_grid(args.min_frames_grid)
    max_damage_grid = parse_float_grid(args.max_damage_grid)
    min_attack_accuracy_grid = parse_float_grid(args.min_attack_accuracy_grid)
    min_sprint_uptime_grid = parse_float_grid(args.min_sprint_uptime_grid)
    weights = parse_score_weights(args.score_weights)

    base_config = Phase1Config(
        chunksize=args.chunksize,
        timestamp_gap=args.timestamp_gap,
        timestamp_col=args.timestamp_col,
        player_col=args.player_col,
        damage_taken_col=args.damage_taken_col,
        damage_dealt_col=args.damage_dealt_col,
        input_lmb_col=args.input_lmb_col,
        input_sprint_col=args.input_sprint_col,
        split_on_player_change=args.split_on_player_change,
    )

    total_runs = (
        len(min_frames_grid)
        * len(max_damage_grid)
        * len(min_attack_accuracy_grid)
        * len(min_sprint_uptime_grid)
    )
    all_csv_files = discover_csv_files(args.input_dir)
    sampled_csv_files = sample_csv_files(all_csv_files, args.sample_fraction, args.sample_seed)

    print(
        f"Running threshold sweep with {total_runs} combinations on "
        f"{len(sampled_csv_files)}/{len(all_csv_files)} files "
        f"(sample_fraction={args.sample_fraction}, seed={args.sample_seed})..."
    )

    runs = run_threshold_sweep(
        input_dir=args.input_dir,
        csv_files=sampled_csv_files,
        base_config=base_config,
        min_frames_grid=min_frames_grid,
        max_damage_grid=max_damage_grid,
        min_attack_accuracy_grid=min_attack_accuracy_grid,
        min_sprint_uptime_grid=min_sprint_uptime_grid,
        weights=weights,
        working_dir=args.work_dir,
        keep_intermediate_outputs=args.keep_intermediate_outputs,
    )

    rows = to_report_rows(runs)
    write_report_csv(args.report_csv, rows)
    print_top_runs(rows, args.top_k)
    print(f"Sweep report written to: {args.report_csv.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



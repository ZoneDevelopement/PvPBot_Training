"""Threshold sweep utilities for Phase 1 filtering experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
from typing import Iterable

from bot_training.data.preprocessing import (
    Phase1Config,
    Phase1Result,
    RejectionBreakdown,
    discover_csv_files,
    process_phase1_csv_files,
)


@dataclass(slots=True)
class SweepRunResult:
    """Outcome of one threshold configuration run."""

    min_frames: int
    max_damage_taken: float
    min_attack_accuracy: float
    min_sprint_uptime: float
    result: Phase1Result
    keep_rate: float
    quality_score: float
    rank: int = 0


@dataclass(slots=True)
class ScoreWeights:
    """Penalty weights for rejection reasons in quality scoring."""

    min_frames: float = 0.25
    damage: float = 0.25
    accuracy: float = 0.25
    sprint: float = 0.25


def parse_int_grid(raw: str) -> list[int]:
    """Parse a comma-separated integer grid string."""

    values: list[int] = []
    for token in raw.split(","):
        value = token.strip()
        if not value:
            continue
        values.append(int(value))
    if not values:
        raise ValueError("Grid must contain at least one integer value.")
    return values


def parse_float_grid(raw: str) -> list[float]:
    """Parse a comma-separated float grid string."""

    values: list[float] = []
    for token in raw.split(","):
        value = token.strip()
        if not value:
            continue
        values.append(float(value))
    if not values:
        raise ValueError("Grid must contain at least one float value.")
    return values


def parse_score_weights(raw: str) -> ScoreWeights:
    """Parse min_frames,damage,accuracy,sprint weights from a CSV string."""

    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != 4:
        raise ValueError("--score-weights must provide exactly 4 comma-separated numbers.")
    return ScoreWeights(min_frames=values[0], damage=values[1], accuracy=values[2], sprint=values[3])


def sample_csv_files(csv_files: list[Path], sample_fraction: float, sample_seed: int) -> list[Path]:
    """Return a deterministic random subset of CSV files based on a fraction."""

    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1].")
    if not csv_files:
        return []
    if sample_fraction >= 1:
        return sorted(csv_files)

    target = max(1, int(round(len(csv_files) * sample_fraction)))
    rng = random.Random(sample_seed)
    indices = sorted(rng.sample(range(len(csv_files)), target))
    sorted_files = sorted(csv_files)
    return [sorted_files[index] for index in indices]


def compute_quality_score(result: Phase1Result, weights: ScoreWeights) -> tuple[float, float]:
    """Return (keep_rate, quality_score) for a phase1 run result."""

    candidates = max(result.candidate_matches, 1)
    keep_rate = result.kept_matches / candidates

    reasons = result.rejection_reasons
    penalty = (
        weights.min_frames * (reasons.min_frames / candidates)
        + weights.damage * (reasons.damage / candidates)
        + weights.accuracy * (reasons.accuracy / candidates)
        + weights.sprint * (reasons.sprint / candidates)
    )
    return keep_rate, keep_rate - penalty


def run_threshold_sweep(
    *,
    input_dir: Path,
    csv_files: list[Path] | None,
    base_config: Phase1Config,
    min_frames_grid: Iterable[int],
    max_damage_grid: Iterable[float],
    min_attack_accuracy_grid: Iterable[float],
    min_sprint_uptime_grid: Iterable[float],
    weights: ScoreWeights,
    working_dir: Path,
    keep_intermediate_outputs: bool,
) -> list[SweepRunResult]:
    """Execute threshold combinations and return ranked sweep results."""

    working_dir = working_dir.expanduser().resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    selected_files = [path.expanduser().resolve() for path in csv_files] if csv_files is not None else discover_csv_files(input_dir)

    runs: list[SweepRunResult] = []
    run_index = 0

    for min_frames, max_damage, min_accuracy, min_sprint in product(
        min_frames_grid,
        max_damage_grid,
        min_attack_accuracy_grid,
        min_sprint_uptime_grid,
    ):
        run_index += 1
        output_file = working_dir / f"sweep_run_{run_index:04d}.csv"

        config = Phase1Config(
            chunksize=base_config.chunksize,
            timestamp_gap=base_config.timestamp_gap,
            min_frames=int(min_frames),
            max_damage_taken=float(max_damage),
            min_attack_accuracy=float(min_accuracy),
            min_sprint_uptime=float(min_sprint),
            timestamp_col=base_config.timestamp_col,
            player_col=base_config.player_col,
            damage_taken_col=base_config.damage_taken_col,
            damage_dealt_col=base_config.damage_dealt_col,
            input_lmb_col=base_config.input_lmb_col,
            input_sprint_col=base_config.input_sprint_col,
            split_on_player_change=base_config.split_on_player_change,
        )

        phase_result = process_phase1_csv_files(input_dir, output_file, config, csv_files=selected_files)
        keep_rate, quality_score = compute_quality_score(phase_result, weights)
        runs.append(
            SweepRunResult(
                min_frames=int(min_frames),
                max_damage_taken=float(max_damage),
                min_attack_accuracy=float(min_accuracy),
                min_sprint_uptime=float(min_sprint),
                result=phase_result,
                keep_rate=keep_rate,
                quality_score=quality_score,
            )
        )

        if not keep_intermediate_outputs and output_file.exists():
            output_file.unlink()

    runs.sort(
        key=lambda run: (
            run.quality_score,
            run.keep_rate,
            run.result.kept_matches,
            -run.result.rejected_matches,
        ),
        reverse=True,
    )
    for index, run in enumerate(runs, start=1):
        run.rank = index
    return runs


def to_report_rows(runs: Iterable[SweepRunResult]) -> list[dict[str, str]]:
    """Convert sweep results to report-friendly row dictionaries."""

    rows: list[dict[str, str]] = []
    for run in runs:
        reasons: RejectionBreakdown = run.result.rejection_reasons
        rows.append(
            {
                "rank": str(run.rank),
                "quality_score": f"{run.quality_score:.6f}",
                "keep_rate": f"{run.keep_rate:.6f}",
                "kept_matches": str(run.result.kept_matches),
                "rejected_matches": str(run.result.rejected_matches),
                "candidate_matches": str(run.result.candidate_matches),
                "min_frames": str(run.min_frames),
                "max_damage_taken": f"{run.max_damage_taken:.6f}",
                "min_attack_accuracy": f"{run.min_attack_accuracy:.6f}",
                "min_sprint_uptime": f"{run.min_sprint_uptime:.6f}",
                "rej_min_frames": str(reasons.min_frames),
                "rej_damage": str(reasons.damage),
                "rej_accuracy": str(reasons.accuracy),
                "rej_sprint": str(reasons.sprint),
            }
        )
    return rows



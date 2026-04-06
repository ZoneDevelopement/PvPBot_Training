"""Phase 1 chunked preprocessing for Minecraft PvP match data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import csv
from typing import Callable, Iterable, Iterator, TextIO, cast
import pandas as pd

@dataclass(slots=True)
class Phase1Config:
    """Configuration for chunked match extraction and filtering."""

    chunksize: int = 50_000
    timestamp_gap: float = 5_000.0
    min_frames: int = 1_000
    max_damage_taken: float = 40.0
    min_attack_accuracy: float = 0.40
    min_sprint_uptime: float = 0.60
    timestamp_col: str = "timestamp"
    player_col: str = "playerName"
    damage_taken_col: str = "damageTaken"
    damage_dealt_col: str = "damageDealt"
    input_lmb_col: str = "inputLmb"
    input_sprint_col: str = "inputSprint"
    split_on_player_change: bool = False


@dataclass(slots=True)
class RejectionBreakdown:
    """Counters for why candidate matches were rejected."""

    min_frames: int = 0
    damage: int = 0
    accuracy: int = 0
    sprint: int = 0


@dataclass(slots=True)
class Phase1Result:
    """Summary of a Phase 1 processing run."""

    files_processed: int = 0
    rows_read: int = 0
    candidate_matches: int = 0
    kept_matches: int = 0
    rejected_matches: int = 0
    rejection_reasons: RejectionBreakdown = field(default_factory=RejectionBreakdown)


@dataclass(slots=True)
class MatchMetrics:
    """Quality metrics for a single candidate match."""

    frame_count: int
    total_damage_taken: float
    clicks: int
    hits: int
    attack_accuracy: float
    sprint_uptime: float


@dataclass(slots=True)
class Phase1Progress:
    """Progress snapshot emitted while processing large datasets."""

    file_index: int
    total_files: int
    current_file: Path
    rows_read: int
    candidate_matches: int
    kept_matches: int
    rejected_matches: int
    rejection_reasons: RejectionBreakdown


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _coerce_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _parse_timestamp_value(value: object) -> float:
    if value is None:
        raise ValueError("Missing timestamp value.")

    text = str(value).strip()
    if not text:
        raise ValueError("Missing timestamp value.")

    try:
        return float(text)
    except ValueError:
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()


def _iter_chunks_from_csv(csv_file: Path, chunksize: int) -> Iterator[tuple[list[dict[str, str]], list[str]]]:
    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        while True:
            chunk: list[dict[str, str]] = []
            for _ in range(chunksize):
                try:
                    chunk.append(next(reader))
                except StopIteration:
                    break
            if not chunk:
                break
            yield chunk, fieldnames


def _iter_chunks(csv_file: Path, chunksize: int) -> Iterator[tuple[list[dict[str, str]], list[str]]]:
    if pd is None:
        yield from _iter_chunks_from_csv(csv_file, chunksize)
        return

    for chunk in pd.read_csv(filepath_or_buffer=csv_file, chunksize=chunksize, iterator=True):  # type: ignore[call-arg]
        yield chunk.to_dict(orient="records"), list(chunk.columns)


def _ensure_required_columns(fieldnames: Iterable[str], config: Phase1Config) -> None:
    required = {
        config.timestamp_col,
        config.player_col,
        config.damage_taken_col,
        config.damage_dealt_col,
        config.input_lmb_col,
        config.input_sprint_col,
    }
    missing = required.difference(fieldnames)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def summarize_match_candidate(rows: list[dict[str, object]], config: Phase1Config) -> MatchMetrics:
    """Calculate match-level quality metrics used for filtering."""

    frame_count = len(rows)
    total_damage_taken = sum(_coerce_float(row.get(config.damage_taken_col)) for row in rows)
    clicks = sum(_coerce_bool(row.get(config.input_lmb_col)) for row in rows)
    hits = sum(_coerce_float(row.get(config.damage_dealt_col)) > 0 for row in rows)
    sprint_uptime = (sum(_coerce_bool(row.get(config.input_sprint_col)) for row in rows) / frame_count) if frame_count else 0.0
    attack_accuracy = float(hits / clicks) if clicks > 0 else 0.0

    return MatchMetrics(
        frame_count=frame_count,
        total_damage_taken=float(total_damage_taken),
        clicks=int(clicks),
        hits=int(hits),
        attack_accuracy=attack_accuracy,
        sprint_uptime=float(sprint_uptime),
    )


def match_passes_filters(metrics: MatchMetrics, config: Phase1Config) -> bool:
    """Return ``True`` when a candidate match satisfies all quality filters."""

    return (
        metrics.frame_count >= config.min_frames
        and metrics.total_damage_taken <= config.max_damage_taken
        and metrics.attack_accuracy >= config.min_attack_accuracy
        and metrics.sprint_uptime >= config.min_sprint_uptime
    )


def match_rejection_reasons(metrics: MatchMetrics, config: Phase1Config) -> list[str]:
    """Return all filter reasons that caused a candidate to be rejected."""

    reasons: list[str] = []
    if metrics.frame_count < config.min_frames:
        reasons.append("min_frames")
    if metrics.total_damage_taken > config.max_damage_taken:
        reasons.append("damage")
    if metrics.attack_accuracy < config.min_attack_accuracy:
        reasons.append("accuracy")
    if metrics.sprint_uptime < config.min_sprint_uptime:
        reasons.append("sprint")
    return reasons


def _apply_rejection_reasons(result: Phase1Result, reasons: list[str]) -> None:
    result.rejected_matches += 1
    for reason in reasons:
        if reason == "min_frames":
            result.rejection_reasons.min_frames += 1
        elif reason == "damage":
            result.rejection_reasons.damage += 1
        elif reason == "accuracy":
            result.rejection_reasons.accuracy += 1
        elif reason == "sprint":
            result.rejection_reasons.sprint += 1


def _snapshot_rejection_reasons(result: Phase1Result) -> RejectionBreakdown:
    return RejectionBreakdown(
        min_frames=result.rejection_reasons.min_frames,
        damage=result.rejection_reasons.damage,
        accuracy=result.rejection_reasons.accuracy,
        sprint=result.rejection_reasons.sprint,
    )


def _write_match(
    writer: csv.DictWriter,
    rows: list[dict[str, object]],
    match_id: int,
) -> None:
    for row in rows:
        output_row: dict[str, str] = {"match_id": str(match_id)}
        for key, value in row.items():
            output_row[key] = "" if value is None else str(value)
        writer.writerow(output_row)


def discover_csv_files(input_path: Path) -> list[Path]:
    """Return CSV files from a directory tree or a single CSV path."""

    resolved = input_path.expanduser().resolve()
    if resolved.is_file():
        return [resolved] if resolved.suffix.lower() == ".csv" else []
    return sorted(resolved.rglob("*.csv"))


def process_phase1_csv_file(
    csv_file: Path,
    output_file: Path,
    config: Phase1Config,
    *,
    append: bool,
    start_match_id: int,
    progress_callback: Callable[[Phase1Progress], None] | None = None,
    file_index: int = 1,
    total_files: int = 1,
    progress_every_chunks: int = 10,
) -> tuple[Phase1Result, int]:
    """Process one CSV file and append passing matches to output."""

    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = Phase1Result(files_processed=1)
    next_match_id = start_match_id
    active_matches: dict[str, list[dict[str, object]]] = {}
    previous_timestamps: dict[str, float] = {}
    chunk_counter = 0

    output_handle: TextIO | None = None
    writer: csv.DictWriter | None = None

    def ensure_writer(fieldnames: list[str]) -> csv.DictWriter:
        nonlocal output_handle, writer
        if writer is not None:
            return writer

        if not append:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_file.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(cast(TextIO, output_handle), fieldnames=["match_id"] + fieldnames)
            writer.writeheader()
            return cast(csv.DictWriter, writer)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_file.exists()
        has_content = file_exists and output_file.stat().st_size > 0
        output_handle = output_file.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(cast(TextIO, output_handle), fieldnames=["match_id"] + fieldnames)
        if not has_content:
            writer.writeheader()
        return cast(csv.DictWriter, writer)

    try:

        for chunk, fieldnames in _iter_chunks(csv_file, config.chunksize):
            chunk_counter += 1
            _ensure_required_columns(fieldnames, config)
            result.rows_read += len(chunk)

            for row in chunk:
                row_player = row.get(config.player_col)
                if row_player is None or str(row_player).strip() == "":
                    continue

                try:
                    row_timestamp = _parse_timestamp_value(row.get(config.timestamp_col))
                except ValueError:
                    continue

                # Fetch or initialize this player's match buffer and last timestamp
                player_key = str(row_player).strip()
                player_match_rows = active_matches.get(player_key, [])
                player_previous_timestamp = previous_timestamps.get(player_key)

                should_start_new_match = (
                    not player_match_rows
                    or (config.split_on_player_change and row_player != player_key)
                    or player_previous_timestamp is None
                    or row_timestamp < player_previous_timestamp
                    or (row_timestamp - player_previous_timestamp) > config.timestamp_gap
                )

                if should_start_new_match and player_match_rows:
                    result.candidate_matches += 1
                    metrics = summarize_match_candidate(player_match_rows, config)
                    if match_passes_filters(metrics, config):
                        current_writer = ensure_writer(list(player_match_rows[0].keys()))
                        _write_match(current_writer, player_match_rows, next_match_id)
                        next_match_id += 1
                        result.kept_matches += 1
                    else:
                        _apply_rejection_reasons(result, match_rejection_reasons(metrics, config))
                    player_match_rows = []

                player_match_rows.append(row)
                active_matches[player_key] = player_match_rows
                previous_timestamps[player_key] = row_timestamp

            if progress_callback is not None and chunk_counter % max(progress_every_chunks, 1) == 0:
                progress_callback(
                    Phase1Progress(
                        file_index=file_index,
                        total_files=total_files,
                        current_file=csv_file,
                        rows_read=result.rows_read,
                        candidate_matches=result.candidate_matches,
                        kept_matches=result.kept_matches,
                        rejected_matches=result.rejected_matches,
                        rejection_reasons=_snapshot_rejection_reasons(result),
                    )
                )

        # Process any remaining match buffers for all players
        for player_key, player_match_rows in active_matches.items():
            if player_match_rows:
                result.candidate_matches += 1
                metrics = summarize_match_candidate(player_match_rows, config)
                if match_passes_filters(metrics, config):
                    current_writer = ensure_writer(list(player_match_rows[0].keys()))
                    _write_match(current_writer, player_match_rows, next_match_id)
                    next_match_id += 1
                    result.kept_matches += 1
                else:
                    _apply_rejection_reasons(result, match_rejection_reasons(metrics, config))
    finally:
        if output_handle is not None:
            cast(TextIO, output_handle).close()

    if progress_callback is not None:
        progress_callback(
            Phase1Progress(
                file_index=file_index,
                total_files=total_files,
                current_file=csv_file,
                rows_read=result.rows_read,
                candidate_matches=result.candidate_matches,
                kept_matches=result.kept_matches,
                rejected_matches=result.rejected_matches,
                rejection_reasons=_snapshot_rejection_reasons(result),
            )
        )

    return result, next_match_id


def process_phase1_csv_files(
    input_dir: Path,
    output_file: Path,
    config: Phase1Config,
    progress_callback: Callable[[Phase1Progress], None] | None = None,
    progress_every_chunks: int = 10,
    csv_files: list[Path] | None = None,
) -> Phase1Result:
    """Stream CSV files in chunks, filter candidate matches, and append clean rows."""

    input_dir = input_dir.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    selected_files = [path.expanduser().resolve() for path in csv_files] if csv_files is not None else discover_csv_files(input_dir)
    result = Phase1Result(files_processed=len(selected_files))
    next_match_id = 1
    for index, csv_file in enumerate(selected_files, start=1):
        file_result, next_match_id = process_phase1_csv_file(
            csv_file,
            output_file,
            config,
            append=index > 1,
            start_match_id=next_match_id,
            progress_callback=progress_callback,
            file_index=index,
            total_files=len(selected_files),
            progress_every_chunks=progress_every_chunks,
        )
        result.rows_read += file_result.rows_read
        result.candidate_matches += file_result.candidate_matches
        result.kept_matches += file_result.kept_matches
        result.rejected_matches += file_result.rejected_matches
        result.rejection_reasons.min_frames += file_result.rejection_reasons.min_frames
        result.rejection_reasons.damage += file_result.rejection_reasons.damage
        result.rejection_reasons.accuracy += file_result.rejection_reasons.accuracy
        result.rejection_reasons.sprint += file_result.rejection_reasons.sprint

    return result


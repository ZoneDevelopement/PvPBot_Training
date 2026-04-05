"""Helpers for locating raw PvP match datasets."""

from __future__ import annotations

from pathlib import Path

from bot_training.config import RAW_DATA_DIR


def list_raw_csv_files(dataset_name: str | None = None) -> list[Path]:
    """Return raw CSV files, optionally filtered by dataset folder name."""
    base = RAW_DATA_DIR if dataset_name is None else RAW_DATA_DIR / dataset_name
    if not base.exists():
        return []
    return sorted(base.glob("*.csv"))


"""One-off model evaluation script."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from bot_training.evaluation.evaluate import main as evaluation_main


if __name__ == "__main__":
    raise SystemExit(evaluation_main())


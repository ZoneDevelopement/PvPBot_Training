"""One-off model training script."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from bot_training.training.train import main as training_main


if __name__ == "__main__":
    raise SystemExit(training_main())


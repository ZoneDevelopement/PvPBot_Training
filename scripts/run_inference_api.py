"""Run the FastAPI inference server for the PvP sequence model."""

from __future__ import annotations
import _bootstrap  # noqa: F401
import uvicorn

from bot_training.inference.api import app

if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)


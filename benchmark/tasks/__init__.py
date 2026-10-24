"""Canonical Balanced120 TaskPack inventory."""

from pathlib import Path

TASKS_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = TASKS_ROOT / "manifest.json"
INTERACTION_OPENINGS_PATH = TASKS_ROOT / "interaction_openings.json"

__all__ = [
    "INTERACTION_OPENINGS_PATH",
    "MANIFEST_PATH",
    "TASKS_ROOT",
]

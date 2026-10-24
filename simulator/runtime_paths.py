"""Writable runtime paths kept outside the immutable release tree."""

from __future__ import annotations

import os
from pathlib import Path


_OUTPUT_ROOT_ENV = "INTERACTIVEWORLD_OUTPUT_DIR"
_DEFAULT_OUTPUT_ROOT = Path("/tmp/interactiveworld-output")


def runtime_output_root() -> Path:
    """Return the absolute directory reserved for runtime-generated artifacts."""
    configured = os.environ.get(_OUTPUT_ROOT_ENV, "").strip()
    root = Path(configured).expanduser() if configured else _DEFAULT_OUTPUT_ROOT
    if not root.is_absolute():
        raise RuntimeError(f"{_OUTPUT_ROOT_ENV} must be an absolute directory path")
    return root.resolve()


def autoexposure_cache_path() -> Path:
    """Return the cache path beneath the configured worker/run output."""
    return runtime_output_root() / "fpv_outputs" / "autoexp_cache.json"


__all__ = ["autoexposure_cache_path", "runtime_output_root"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


def behavior_mdl_dir() -> Path | None:
    """Return the OmniGibson MDL material directory used by baked BEHAVIOR assets.

    Without it every BEHAVIOR surface renders as the flat-red MDL fallback.
    ``launch/prepare_assets.sh`` copies the materials to ``<cache>/materials``.
    """
    configured = os.environ.get("INTERACTIVEWORLD_MDL_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    cache = os.environ.get("INTERACTIVEWORLD_BEHAVIOR_CACHE", "").strip()
    if cache:
        candidate = Path(cache).expanduser() / "materials"
        if candidate.is_dir():
            return candidate
    return None


__all__ = ["autoexposure_cache_path", "behavior_mdl_dir", "runtime_output_root"]

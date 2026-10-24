# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve immutable visual assets referenced by released TaskPacks."""

from __future__ import annotations

import os
from pathlib import Path


_LABEL_ROOT_ENV = "INTERACTIVEWORLD_LABEL_ASSET_ROOT"
_GENERATED_LABEL_CACHE_ENV = "INTERACTIVEWORLD_GENERATED_LABEL_CACHE"
_BUNDLED_LABEL_ROOT = (
    Path(__file__).resolve().parents[2] / "benchmark" / "tasks" / "assets" / "labels"
)
_DEFAULT_GENERATED_LABEL_CACHE = Path("/tmp/interactiveworld-output/labels")


def label_asset_root() -> Path:
    """Return the configured label root, rejecting ambiguous relative overrides."""
    configured = os.environ.get(_LABEL_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise RuntimeError(f"{_LABEL_ROOT_ENV} must be an absolute directory path")
    else:
        root = _BUNDLED_LABEL_ROOT
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"TaskPack label asset directory does not exist: {root}. "
            f"Install the bundled benchmark assets or set {_LABEL_ROOT_ENV}."
        )
    return root


def resolve_label_asset(source: str) -> Path:
    """Resolve one TaskPack-relative label path without permitting root escape."""
    relative = Path(source)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("TaskPack label source must be a safe relative path")
    root = label_asset_root()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("TaskPack label source escapes the configured asset root")
    if not candidate.is_file():
        raise FileNotFoundError(f"TaskPack label asset does not exist: {source}")
    return candidate


def generated_label_cache() -> Path:
    """Return the writable cache for labels generated during world construction."""
    configured = os.environ.get(_GENERATED_LABEL_CACHE_ENV)
    path = (
        Path(configured).expanduser() if configured else _DEFAULT_GENERATED_LABEL_CACHE
    )
    if not path.is_absolute():
        raise RuntimeError(
            f"{_GENERATED_LABEL_CACHE_ENV} must be an absolute directory path"
        )
    return path.resolve()


__all__ = ["generated_label_cache", "label_asset_root", "resolve_label_asset"]

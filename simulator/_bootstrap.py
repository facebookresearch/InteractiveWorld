# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Centralized compatibility paths for the frozen simulator implementation.

The simulator implementation uses sibling imports because Isaac Kit executes
files as scripts.  Keep that compatibility in one place while exposing
the release as ordinary, functionally grouped Python packages.
"""

from __future__ import annotations

import sys
from pathlib import Path


_SIMULATOR_ROOT = Path(__file__).resolve().parent
_RUNTIME_MODULE_DIRS = (
    "physical_world",
    "environment",
    "interaction",
    "transport",
    "taskpack_support",
    "world_catalog",
    "human_model",
)


def activate_runtime_paths() -> None:
    """Expose frozen sibling modules to the Isaac script-style runtime."""
    for directory in reversed(_RUNTIME_MODULE_DIRS):
        path = str(_SIMULATOR_ROOT / directory)
        if path not in sys.path:
            sys.path.insert(0, path)

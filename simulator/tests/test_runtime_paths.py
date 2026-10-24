# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulator.runtime_paths import autoexposure_cache_path


class RuntimePathsTest(unittest.TestCase):
    def test_autoexposure_cache_is_beneath_configured_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "worker-0" / "run"
            with patch.dict(
                os.environ,
                {"INTERACTIVEWORLD_OUTPUT_DIR": str(output_root)},
            ):
                self.assertEqual(
                    output_root / "fpv_outputs" / "autoexp_cache.json",
                    autoexposure_cache_path(),
                )

    def test_autoexposure_cache_default_does_not_use_release_tree(self) -> None:
        release_root = Path(__file__).resolve().parents[2]
        with patch.dict(os.environ):
            os.environ.pop("INTERACTIVEWORLD_OUTPUT_DIR", None)
            cache_path = autoexposure_cache_path()

        self.assertEqual(
            Path("/tmp/interactiveworld-output/fpv_outputs/autoexp_cache.json"),
            cache_path,
        )
        self.assertFalse(cache_path.is_relative_to(release_root))

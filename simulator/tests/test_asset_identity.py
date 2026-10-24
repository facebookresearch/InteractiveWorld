# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulator.physical_world import scene_furniture_units as sfu


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class CanonicalAssetIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        root = Path(self._temporary.name)
        source = root / "behavior-1k-assets/objects/fridge/abcdef/usd/abcdef.encrypted.usd"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"encrypted source")
        self.data_root = root
        self.frozen = {
            "fridge/abcdef.usd": {
                "source_sha256": _digest(b"encrypted source"),
                "baked_sha256": _digest(b"reference bake"),
            }
        }

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _canonical(self, digest: str) -> str:
        with (
            patch.object(sfu, "_BEHAVIOR_DATA_ROOT", str(self.data_root)),
            patch.object(sfu, "_FROZEN_IDENTITIES", self.frozen),
            patch.dict(sfu._SOURCE_DIGESTS, clear=True),
        ):
            return sfu.canonical_asset_sha256("fridge/abcdef.usd", digest)

    def test_local_bake_of_frozen_source_carries_frozen_digest(self) -> None:
        self.assertEqual(
            _digest(b"reference bake"), self._canonical(_digest(b"local bake"))
        )

    def test_reference_bake_is_unchanged(self) -> None:
        self.assertEqual(
            _digest(b"reference bake"), self._canonical(_digest(b"reference bake"))
        )

    def test_different_source_keeps_its_own_digest(self) -> None:
        source = next(self.data_root.rglob("*.encrypted.usd"))
        source.write_bytes(b"another dataset version")
        self.assertEqual(_digest(b"local bake"), self._canonical(_digest(b"local bake")))

    def test_unpinned_asset_keeps_its_own_digest(self) -> None:
        with patch.object(sfu, "_FROZEN_IDENTITIES", self.frozen):
            self.assertEqual(
                "sha256:x", sfu.canonical_asset_sha256("table/zzzzzz.usd", "sha256:x")
            )

    def test_release_manifest_pins_every_scene_bound_taskpack_asset(self) -> None:
        release_root = Path(__file__).resolve().parents[2]
        manifest = json.loads(
            (release_root / "simulator/world_catalog/behavior_assets.json").read_text(
                encoding="utf-8"
            )
        )
        pinned = {}
        for path in (release_root / "benchmark/tasks/taskpacks").rglob("*.json"):
            stack = [json.loads(path.read_text(encoding="utf-8"))]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    if "asset_sha256" in value and "name" in value:
                        cat, model, _ = str(value["name"]).rsplit("_", 2)
                        pinned[f"{cat}/{model}.usd"] = value["asset_sha256"]
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
        identities = manifest["frozen_identities"]
        self.assertEqual(set(pinned), set(identities))
        for ref, digest in pinned.items():
            self.assertEqual(digest, identities[ref]["baked_sha256"])
            self.assertIn(ref.removesuffix(".usd"), manifest["bake_models"])

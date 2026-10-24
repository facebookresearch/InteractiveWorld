from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


_SIMULATOR_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_ROOT = _SIMULATOR_ROOT.parent


class ReleaseContractTest(unittest.TestCase):
    def test_manifests_are_valid_json(self) -> None:
        for name in ("component_manifest.json", "external_dependencies.json"):
            value = json.loads((_SIMULATOR_ROOT / name).read_text(encoding="utf-8"))
            self.assertIsInstance(value, dict)
            self.assertTrue(value["schema_version"].startswith("interactiveworld-"))

    def test_public_entrypoints_exist(self) -> None:
        expected = (
            _SIMULATOR_ROOT / "api.py",
            _SIMULATOR_ROOT / "physical_world" / "episode_runtime.py",
            _SIMULATOR_ROOT / "environment" / "isaac_env.py",
            _SIMULATOR_ROOT / "interaction" / "episode_loop.py",
            _RELEASE_ROOT / "launch" / "run_taskpack.py",
            _RELEASE_ROOT / "launch" / "run_sim_server.py",
        )
        self.assertTrue(all(path.is_file() for path in expected))

    def test_no_markdown_or_bytecode_is_packaged(self) -> None:
        forbidden = []
        for root in (_SIMULATOR_ROOT, _RELEASE_ROOT / "launch"):
            forbidden.extend(
                path
                for path in root.rglob("*")
                if path.suffix in {".md", ".pyc"} or path.name == "__pycache__"
            )
        self.assertEqual([], forbidden)

    def test_no_restricted_human_data_is_bundled(self) -> None:
        human_root = _SIMULATOR_ROOT / "human_model"
        restricted_suffixes = {".pkl", ".npz", ".obj", ".fbx", ".glb", ".gltf"}
        packaged = [
            path for path in human_root.rglob("*") if path.suffix in restricted_suffixes
        ]
        self.assertEqual([], packaged)
        dependencies = json.loads(
            (_SIMULATOR_ROOT / "external_dependencies.json").read_text(encoding="utf-8")
        )
        external_names = {
            str(entry["name"]) for entry in dependencies["required_external_assets"]
        }
        self.assertIn("SMPL+H male body model", external_names)
        self.assertIn("ACCAD/AMASS forward walk clip", external_names)

    def test_runtime_label_paths_are_consistent(self) -> None:
        from simulator.physical_world import (
            asset_preflight,
            behavior_import,
            task_world_builder,
        )
        from simulator.taskpack_support.bundled_assets import (
            generated_label_cache,
            label_asset_root,
            resolve_label_asset,
        )

        generated = generated_label_cache()
        self.assertEqual(generated, Path(task_world_builder.GENERATED_LABEL_DIR))
        self.assertEqual(generated, Path(behavior_import.GENERATED_LABEL_DIR))
        self.assertEqual(generated, Path(asset_preflight.GENERATED_LABEL_DIR))
        self.assertNotEqual(generated, label_asset_root())

        relative = "ask81019/requisition.png"
        expected = resolve_label_asset(relative)
        self.assertEqual(label_asset_root() / relative, expected)

    def test_scene_capability_index_is_available(self) -> None:
        from simulator.physical_world.task_world_builder import recon_scenes

        scenes = recon_scenes()
        self.assertEqual(51, len(scenes))
        self.assertIn("Rs_int", scenes)
        self.assertIn("work_surfaces", scenes["Rs_int"])

    def test_simulator_tree_digest(self) -> None:
        manifest = json.loads(
            (_SIMULATOR_ROOT / "component_manifest.json").read_text(encoding="utf-8")
        )
        files = [
            path
            for path in _SIMULATOR_ROOT.rglob("*")
            if path.is_file()
            and path.name != "component_manifest.json"
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
        ]
        digest = hashlib.sha256()
        for path in sorted(
            files, key=lambda value: value.relative_to(_SIMULATOR_ROOT).as_posix()
        ):
            relative = path.relative_to(_SIMULATOR_ROOT).as_posix().encode("utf-8")
            data = path.read_bytes()
            digest.update(relative)
            digest.update(b"\0")
            digest.update(str(len(data)).encode("ascii"))
            digest.update(b"\0")
            digest.update(data)
        integrity = manifest["tree_integrity"]
        self.assertEqual(integrity["file_count"], len(files))
        self.assertEqual(integrity["sha256"], digest.hexdigest())

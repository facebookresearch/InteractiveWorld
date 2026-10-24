# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest import mock

from simulator.catalog import (
    INTERACTION_OPENINGS_VERSION,
    MANIFEST_VERSION,
    RUNTIME_CONTRACT_VERSION,
    TaskCatalog,
    TaskInfo,
    TASKPACK_VERSION,
    TaskSpec,
)
from simulator.errors import TaskCatalogError, TaskNotFoundError, TaskValidationError
from simulator.interaction.task_runtime import digest_json
from simulator.taskpack_support.runtime_adapter import ADAPTER_VERSION, canonical_digest


class TaskCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = TaskCatalog.bundled()
        self.info = self.catalog.list()[0]

    def test_bundled_catalog_lists_and_filters_immutable_metadata(self) -> None:
        tasks = self.catalog.list()
        accessibility = self.catalog.list(category="accessibility")
        watch = self.catalog.list(source_family="watch")
        combined = self.catalog.list(
            category=tasks[0].category.lower(),
            source_family=tasks[0].source_family.lower(),
        )

        self.assertEqual(120, len(self.catalog))
        self.assertEqual(120, len(tasks))
        self.assertEqual(20, len(accessibility))
        self.assertEqual(22, len(watch))
        self.assertTrue(combined)
        self.assertTrue(
            all(
                task.category == tasks[0].category
                and task.source_family == tasks[0].source_family
                for task in combined
            )
        )
        with self.assertRaises(FrozenInstanceError):
            tasks[0].__setattr__("task_id", "changed")

    def test_get_verifies_and_projects_a_bundled_task(self) -> None:
        spec = self.catalog.get(self.info.task_id)

        self.assertIsInstance(self.info, TaskInfo)
        self.assertIsInstance(spec, TaskSpec)
        self.assertEqual(self.info.task_id, spec.task_id)
        self.assertEqual(self.info.taskpack_sha256, spec.source_sha256)
        self.assertEqual(TASKPACK_VERSION, spec.taskpack_version)
        self.assertEqual(RUNTIME_CONTRACT_VERSION, spec.runtime_contract_version)
        self.assertEqual(self.info, spec.info)
        self.assertEqual(spec.task_id, spec.taskpack["task_id"])
        self.assertEqual(spec.task_id, spec.builder_spec["task_id"])
        self.assertEqual(spec.scene, spec.builder_spec["scene"])
        self.assertEqual(
            ADAPTER_VERSION, spec.builder_spec["runtime_adapter"]["version"]
        )
        self.assertEqual(canonical_digest(spec.builder_spec), spec.builder_spec_digest)

        detached = spec.taskpack
        detached["task_id"] = "changed"
        self.assertEqual(self.info.task_id, spec.taskpack["task_id"])
        with self.assertRaises(FrozenInstanceError):
            spec.__setattr__("task_id", "changed")

    def test_every_bundled_taskpack_satisfies_the_public_contract(self) -> None:
        specs = tuple(self.catalog.get(info.task_id) for info in self.catalog.list())

        self.assertEqual(120, len(specs))
        self.assertTrue(all(spec.taskpack["story"] for spec in specs))
        self.assertEqual(
            22,
            sum(spec.preparation_mode == "continuous_watch" for spec in specs),
        )
        self.assertEqual(
            98,
            sum(spec.preparation_mode == "prologue" for spec in specs),
        )

    def test_bundled_preparation_mode_is_self_describing(self) -> None:
        for info in self.catalog.list():
            with self.subTest(task_id=info.task_id):
                catalog_spec = self.catalog.get(info.task_id)
                file_spec = TaskSpec.load(
                    info.taskpack_path,
                    expected_sha256=info.taskpack_sha256,
                    expected_task_id=info.task_id,
                )
                self.assertEqual(
                    catalog_spec.preparation_mode,
                    file_spec.preparation_mode,
                )

    def test_unknown_task_has_a_public_error(self) -> None:
        with self.assertRaisesRegex(TaskNotFoundError, "not present"):
            self.catalog.get("WATCH-DOES-NOT-EXIST")

    def test_explicit_file_requires_the_exact_sha256_and_task_id(self) -> None:
        spec = TaskSpec.load(
            self.info.taskpack_path,
            expected_sha256=self.info.taskpack_sha256,
            expected_task_id=self.info.task_id,
        )

        self.assertEqual(self.info.task_id, spec.task_id)
        self.assertIsNone(spec.info)
        with self.assertRaisesRegex(TaskValidationError, "SHA256 mismatch"):
            TaskSpec.load(
                self.info.taskpack_path,
                expected_sha256="sha256:" + "0" * 64,
            )
        with self.assertRaisesRegex(TaskValidationError, "task_id mismatch"):
            TaskSpec.load(
                self.info.taskpack_path,
                expected_sha256=self.info.taskpack_sha256,
                expected_task_id="WATCH-DOES-NOT-MATCH",
            )

    def test_inline_taskpack_uses_canonical_digest_and_owns_its_snapshot(self) -> None:
        taskpack = self._read_source_taskpack()
        expected_digest = digest_json(taskpack)

        spec = TaskSpec.from_mapping(taskpack, expected_sha256=expected_digest)
        taskpack["task_id"] = "changed-after-load"

        self.assertEqual(self.info.task_id, spec.task_id)
        self.assertIsNone(spec.source_sha256)
        self.assertEqual(expected_digest, spec.taskpack_digest)
        self.assertIsNone(spec.source_path)
        self.assertEqual(self.info.task_id, spec.taskpack["task_id"])
        with self.assertRaisesRegex(TaskValidationError, "canonical SHA256 mismatch"):
            TaskCatalog.load_taskpack(
                spec.taskpack,
                expected_sha256="sha256:" + "0" * 64,
            )

    def test_rejects_version_drift(self) -> None:
        cases = (
            ("taskpack_version", self._with_taskpack_version_drift()),
            ("runtime contract", self._with_runtime_version_drift()),
        )
        with tempfile.TemporaryDirectory() as directory:
            for expected_message, taskpack in cases:
                with self.subTest(expected_message=expected_message):
                    path, expected_hash = self._write_taskpack(
                        Path(directory), expected_message, taskpack
                    )
                    with self.assertRaisesRegex(TaskValidationError, expected_message):
                        TaskSpec.load(path, expected_sha256=expected_hash)

    def test_rejects_missing_or_malformed_public_task_fields(self) -> None:
        source = self._read_source_taskpack()
        cases: list[tuple[str, dict[str, Any]]] = []

        missing_story = copy.deepcopy(source)
        missing_story.pop("story")
        cases.append(("story must be an object", missing_story))

        empty_goal = copy.deepcopy(source)
        empty_goal["story"]["assistant_goal"] = "  "
        cases.append(("story.assistant_goal must be a non-empty string", empty_goal))

        malformed_actions = copy.deepcopy(source)
        malformed_actions["story"]["prologue"]["actions"] = "[Wait]"
        cases.append(("story.prologue.actions must be an array", malformed_actions))

        invalid_action = copy.deepcopy(source)
        invalid_action["story"]["prologue"]["actions"] = [""]
        cases.append((r"story.prologue.actions\[0\]", invalid_action))

        malformed_prologue_say = copy.deepcopy(source)
        malformed_prologue_say["story"]["prologue"]["say"] = None
        cases.append(("story.prologue.say must be a string", malformed_prologue_say))

        invalid_tick_budget = copy.deepcopy(source)
        invalid_tick_budget["runtime"]["tick_budget"] = 0
        cases.append(
            ("runtime.tick_budget must be a positive integer", invalid_tick_budget)
        )

        missing_human_policy = copy.deepcopy(source)
        missing_human_policy["runtime"].pop("human_policy")
        cases.append(("runtime.human_policy must be an object", missing_human_policy))

        unknown_policy_version = copy.deepcopy(source)
        unknown_policy_version["runtime"]["human_policy"]["version"] = (
            "human-policy/999"
        )
        cases.append(
            ("unsupported runtime.human_policy.version", unknown_policy_version)
        )

        empty_default_plan = copy.deepcopy(source)
        empty_default_plan["runtime"]["human_policy"]["default_plan"] = []
        cases.append(
            ("runtime.human_policy.default_plan must not be empty", empty_default_plan)
        )

        malformed_transitions = copy.deepcopy(source)
        malformed_transitions["runtime"]["human_policy"]["transitions"] = ["turn0"]
        cases.append((r"runtime.human_policy.transitions\[0\]", malformed_transitions))

        for expected_message, taskpack in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(TaskValidationError, expected_message):
                    TaskSpec.from_mapping(
                        taskpack,
                        expected_sha256=digest_json(taskpack),
                    )

    def test_custom_taskpack_declares_or_defaults_its_preparation_mode(self) -> None:
        default_taskpack = self._read_source_taskpack()
        explicit_taskpack = copy.deepcopy(default_taskpack)
        explicit_taskpack["runtime"]["public_preparation"] = {
            "mode": "continuous_watch"
        }

        default_spec = TaskSpec.from_mapping(
            default_taskpack,
            expected_sha256=digest_json(default_taskpack),
        )
        explicit_spec = TaskSpec.from_mapping(
            explicit_taskpack,
            expected_sha256=digest_json(explicit_taskpack),
        )

        self.assertEqual("prologue", default_spec.preparation_mode)
        self.assertEqual("continuous_watch", explicit_spec.preparation_mode)

    def test_rejects_an_unknown_custom_preparation_mode(self) -> None:
        cases = (
            (None, "runtime.public_preparation must be an object"),
            ({"mode": "guess"}, "runtime.public_preparation.mode"),
        )
        for value, expected_message in cases:
            with self.subTest(value=value):
                taskpack = self._read_source_taskpack()
                taskpack["runtime"]["public_preparation"] = value
                with self.assertRaisesRegex(TaskValidationError, expected_message):
                    TaskSpec.from_mapping(
                        taskpack,
                        expected_sha256=digest_json(taskpack),
                    )

    def test_bundled_interaction_metadata_uses_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, bundle_root = self._write_single_task_bundle(Path(directory))
            (bundle_root / "interaction_openings.json").write_text(
                '{"version":"taskpack-interaction-openings/2.1","tasks":{},"tasks":{}}',
                encoding="utf-8",
            )

            with (
                mock.patch(
                    "simulator.catalog._bundled_manifest_path",
                    return_value=manifest_path,
                ),
                self.assertRaisesRegex(TaskCatalogError, "duplicate JSON field"),
            ):
                TaskCatalog.bundled()

    def test_bundled_interaction_metadata_cannot_escape_manifest_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_root = self._write_single_task_bundle(root)
            external = root / "outside.json"
            external.write_text(
                json.dumps({"version": INTERACTION_OPENINGS_VERSION, "tasks": {}}),
                encoding="utf-8",
            )
            (bundle_root / "interaction_openings.json").symlink_to(external)

            with (
                mock.patch(
                    "simulator.catalog._bundled_manifest_path",
                    return_value=manifest_path,
                ),
                self.assertRaisesRegex(TaskCatalogError, "escapes"),
            ):
                TaskCatalog.bundled()

    def test_bundled_preparation_mode_cannot_conflict_with_taskpack(self) -> None:
        taskpack = self._read_source_taskpack()
        taskpack["runtime"]["public_preparation"] = {"mode": "prologue"}
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, bundle_root = self._write_single_task_bundle(
                Path(directory), taskpack=taskpack
            )
            (bundle_root / "interaction_openings.json").write_text(
                json.dumps(
                    {
                        "version": INTERACTION_OPENINGS_VERSION,
                        "tasks": {
                            self.info.task_id: {"continuous_watch": True},
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "simulator.catalog._bundled_manifest_path",
                return_value=manifest_path,
            ):
                catalog = TaskCatalog.bundled()
                with self.assertRaisesRegex(TaskValidationError, "conflicts"):
                    catalog.get(self.info.task_id)

    def test_rejects_a_runtime_adapter_incompatible_spec(self) -> None:
        taskpack = self._read_source_taskpack()
        executable_spec = taskpack["executable_spec"]
        executable_spec["spec_version"] = "unsupported-bts-version"
        taskpack["lineage"]["executable_spec_digest"] = canonical_digest(
            executable_spec
        )

        with tempfile.TemporaryDirectory() as directory:
            path, expected_hash = self._write_taskpack(
                Path(directory), "adapter", taskpack
            )
            with self.assertRaisesRegex(
                TaskValidationError, "invalid runtime builder spec"
            ):
                TaskSpec.load(path, expected_sha256=expected_hash)

    def _read_source_taskpack(self) -> dict[str, Any]:
        value = json.loads(self.info.taskpack_path.read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        return value

    def _with_taskpack_version_drift(self) -> dict[str, Any]:
        taskpack = self._read_source_taskpack()
        taskpack["taskpack_version"] = "data-generate-taskpack/999"
        return taskpack

    def _with_runtime_version_drift(self) -> dict[str, Any]:
        taskpack = self._read_source_taskpack()
        taskpack["runtime"]["contract_version"] = "compiled-runtime-contract/999"
        return taskpack

    def _write_taskpack(
        self, directory: Path, name: str, taskpack: dict[str, Any]
    ) -> tuple[Path, str]:
        path = directory / f"{name.replace(' ', '-')}.json"
        data = json.dumps(taskpack, ensure_ascii=False).encode("utf-8")
        path.write_bytes(data)
        return path, f"sha256:{hashlib.sha256(data).hexdigest()}"

    def _write_single_task_bundle(
        self,
        directory: Path,
        *,
        taskpack: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        bundle_root = directory / "bundle"
        taskpack_root = bundle_root / "taskpacks"
        taskpack_root.mkdir(parents=True)
        taskpack_path, taskpack_sha256 = self._write_taskpack(
            taskpack_root,
            self.info.task_id,
            taskpack if taskpack is not None else self._read_source_taskpack(),
        )
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "release_id": "test-release",
            "taskpack_root": "taskpacks",
            "task_count": 1,
            "tasks": [
                {
                    "task_id": self.info.task_id,
                    "assigned_family": self.info.assigned_family,
                    "source_family": self.info.source_family,
                    "taskpack_path": str(taskpack_path.relative_to(bundle_root)),
                    "taskpack_sha256": taskpack_sha256,
                    "taskpack_version": self.info.taskpack_version,
                    "runtime_contract_version": self.info.runtime_contract_version,
                    "completion_protocol": self.info.completion_protocol,
                    "performance_rubric_version": self.info.performance_rubric_version,
                }
            ],
        }
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, bundle_root

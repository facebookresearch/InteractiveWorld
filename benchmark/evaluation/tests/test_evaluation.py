# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmark.evaluation import (
    aggregate_results,
    EVIDENCE_SCHEMA_VERSION,
    load_rubric,
    prepare_evidence,
    score_evidence,
    task_index,
    validate_task_manifest,
)
from benchmark.evaluation.cli import _read_json_or_jsonl, main as cli_main
from benchmark.evaluation.run_taskpack import (
    _expected_taskpack_sha256,
    load_frozen_tasks,
    main as run_taskpack_main,
    TaskPackEvalError,
)

_RUN_EVIDENCE_SHA256 = "sha256:" + "a" * 64


class _StopBeforeIsaac(RuntimeError):
    pass


class TaskpackRuntimeConfigTest(unittest.TestCase):
    def test_main_configures_absolute_output_root_before_startup_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "worker-0" / "run"
            with (
                patch.dict(
                    os.environ,
                    {"INTERACTIVEWORLD_OUTPUT_DIR": "/tmp/stale-output"},
                ),
                patch(
                    "simulator.interaction.episode_loop.assert_frame_budget",
                    side_effect=_StopBeforeIsaac,
                ),
            ):
                with self.assertRaises(_StopBeforeIsaac):
                    run_taskpack_main(
                        [
                            "--task-id",
                            "ASK-13000",
                            "--assistant",
                            "oracle-verbal",
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

                self.assertEqual(
                    str(output_dir.resolve()),
                    os.environ["INTERACTIVEWORLD_OUTPUT_DIR"],
                )
                self.assertTrue(output_dir.is_dir())

    def test_main_establishes_default_and_explicit_frame_budget_before_gate(
        self,
    ) -> None:
        base_args = ["--task-id", "ASK-13000", "--assistant", "oracle-verbal"]
        cases = (([], "24"), (["--frame-budget", "41"], "41"))

        for extra_args, expected in cases:
            with (
                self.subTest(frame_budget=expected),
                patch.dict(os.environ, {"ISAAC_FRAME_BUDGET": "999"}),
                patch(
                    "simulator.interaction.episode_loop.assert_frame_budget",
                    side_effect=_StopBeforeIsaac,
                ) as startup_gate,
            ):
                with self.assertRaises(_StopBeforeIsaac):
                    run_taskpack_main([*base_args, *extra_args])

                self.assertEqual(expected, os.environ.get("ISAAC_FRAME_BUDGET"))
                startup_gate.assert_called_once_with()

    def test_main_rejects_nonpositive_frame_budget_before_simulator_import(
        self,
    ) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as context:
            run_taskpack_main(
                [
                    "--task-id",
                    "ASK-13000",
                    "--assistant",
                    "oracle-verbal",
                    "--frame-budget",
                    "0",
                ]
            )

        self.assertEqual(2, context.exception.code)
        self.assertIn("--frame-budget must be positive", stderr.getvalue())


def valid_checks() -> dict[str, bool]:
    return {
        "scene_setup_ok": True,
        "runtime_complete": True,
        "trace_complete": True,
        "frames_usable": True,
        "assistant_transport_ok": True,
        "human_executor_ok": True,
        "simulator_ok": True,
        "no_material_human_leakage": True,
    }


def valid_native() -> dict[str, bool]:
    return {
        "contract_world_success": True,
        "process_requirements_met": True,
        "verification_points_met": True,
        "assistant_declared_done": True,
        "requires_assistant_done": True,
    }


class CustomTaskpackIdentityTest(unittest.TestCase):
    _RELEASE = {"ASK-12802": {"expected_taskpack_sha256": "sha256:" + "a" * 64}}

    def test_frozen_task_id_requires_the_frozen_file(self) -> None:
        self.assertEqual(
            "sha256:" + "a" * 64,
            _expected_taskpack_sha256(self._RELEASE, "ASK-12802", "sha256:" + "a" * 64),
        )
        with self.assertRaises(TaskPackEvalError):
            _expected_taskpack_sha256(self._RELEASE, "ASK-12802", "sha256:" + "b" * 64)

    def test_custom_task_id_is_pinned_by_the_supplied_digest(self) -> None:
        self.assertEqual(
            "sha256:" + "c" * 64,
            _expected_taskpack_sha256(self._RELEASE, "MY-TASK", "sha256:" + "c" * 64),
        )


class RubricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rubric = load_rubric()
        cls.tasks = task_index(cls.rubric)

    def evidence(self, task_id: str, release: dict) -> dict:
        task = self.tasks[task_id]
        release = {
            "rubric_version": task["rubric_version"],
            "evidence_mode": "trace_adjudicated",
            "adjudication_required": True,
            "adjudication_status": "COMPLETE",
            **release,
        }
        return {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "task_id": task_id,
            "taskpack_sha256": task["expected_taskpack_sha256"],
            "run_evidence_sha256": _RUN_EVIDENCE_SHA256,
            "validity": valid_checks(),
            "native_taskpack": valid_native(),
            "release": release,
        }

    def criterion_results(
        self, task_id: str, field: str, values: list[object]
    ) -> dict[str, object]:
        spec_name = {
            "group_results": "groups",
            "gate_results": "gates",
            "bundle_results": "bundles",
            "terminal_group_results": "terminal_groups",
            "transition_gate_results": "transition_gates",
            "atom_results": "atoms",
            "condition_results": "conditions",
            "causal_checks": "checks",
        }[field]
        criteria = self.tasks[task_id]["score_spec"][spec_name]
        self.assertEqual(len(criteria), len(values))
        return {
            criterion["id"]: value
            for criterion, value in zip(criteria, values, strict=True)
        }

    def passing_evidence(self, task_id: str) -> dict:
        task = self.tasks[task_id]
        template = prepare_evidence(
            {
                "task_id": task_id,
                "taskpack_file_sha256": task["expected_taskpack_sha256"],
                "run_evidence_sha256": _RUN_EVIDENCE_SHA256,
                "validity_evidence": valid_checks(),
                "native_taskpack_outcome": valid_native(),
            },
            self.rubric,
        )
        template["release"]["adjudication_status"] = "COMPLETE"
        for name, values in template["release"].items():
            if isinstance(values, dict):
                passing = 1.0 if name == "group_results" else True
                for criterion_id in values:
                    values[criterion_id] = passing
        return template

    def test_frozen_inventory(self) -> None:
        self.assertEqual(len(self.rubric["tasks"]), 120)
        self.assertEqual(len(self.tasks), 120)
        self.assertEqual(
            Counter(t["assigned_category"] for t in self.rubric["tasks"]),
            self.rubric["assigned_category_counts"],
        )
        self.assertEqual(
            Counter(t["source_family"] for t in self.rubric["tasks"]),
            self.rubric["source_family_counts"],
        )
        self.assertEqual(
            Counter(t["protocol"] for t in self.rubric["tasks"]),
            {
                "StateScore-v2": 26,
                "StateBundle-v1": 16,
                "CFSS-StateScore": 20,
                "ASK-PCS": 19,
                "WATCH-binary": 22,
                "FIND-C-pass-binary": 17,
            },
        )

    def test_task_manifest_ids_metadata_and_files_match_rubric(self) -> None:
        validate_task_manifest(self.rubric)

    def test_every_criterion_has_a_stable_english_id_and_description(self) -> None:
        collection_names = (
            "groups",
            "gates",
            "bundles",
            "terminal_groups",
            "transition_gates",
            "atoms",
            "conditions",
            "checks",
        )
        for task in self.rubric["tasks"]:
            with self.subTest(task_id=task["task_id"]):
                criteria = [
                    criterion
                    for name in collection_names
                    for criterion in task["score_spec"].get(name, [])
                ]
                self.assertEqual(task["criteria_count"], len(criteria))
                self.assertEqual(
                    len(criteria), len({criterion["id"] for criterion in criteria})
                )
                self.assertTrue(
                    all(criterion["description"].strip() for criterion in criteria)
                )
                self.assertFalse(
                    any(
                        "\u4e00" <= character <= "\u9fff"
                        for criterion in criteria
                        for character in criterion["description"]
                    )
                )

    def test_native_and_release_outcomes_are_independent(self) -> None:
        evidence = self.evidence(
            "GUIDE-102000",
            {
                "group_results": self.criterion_results(
                    "GUIDE-102000", "group_results", [1, 1, 1]
                ),
                "gate_results": self.criterion_results(
                    "GUIDE-102000", "gate_results", [True]
                ),
            },
        )
        evidence["native_taskpack"] = {
            "contract_world_success": True,
            "process_requirements_met": True,
            "verification_points_met": True,
            "assistant_declared_done": False,
            "requires_assistant_done": True,
        }
        result = score_evidence(evidence, self.rubric)
        self.assertFalse(result["native_taskpack"]["strict_success"])
        self.assertTrue(result["release"]["strict_pass"])
        self.assertAlmostEqual(result["release"]["completion_score"], 1.0)

    def test_stable_criterion_id_mapping_is_order_independent(self) -> None:
        task = self.tasks["GUIDE-102000"]
        groups = {
            criterion["id"]: 1.0 for criterion in reversed(task["score_spec"]["groups"])
        }
        gates = {
            criterion["id"]: True for criterion in reversed(task["score_spec"]["gates"])
        }
        result = score_evidence(
            self.evidence(
                "GUIDE-102000",
                {"group_results": groups, "gate_results": gates},
            ),
            self.rubric,
        )
        self.assertEqual(result["status"], "SCORED")
        self.assertTrue(result["release"]["strict_pass"])

    def test_missing_criterion_id_fails_closed(self) -> None:
        task = self.tasks["GUIDE-102000"]
        groups = {criterion["id"]: 1.0 for criterion in task["score_spec"]["groups"]}
        groups.pop(next(iter(groups)))
        gates = {criterion["id"]: True for criterion in task["score_spec"]["gates"]}
        with self.assertRaisesRegex(ValueError, "missing fields"):
            score_evidence(
                self.evidence(
                    "GUIDE-102000",
                    {"group_results": groups, "gate_results": gates},
                ),
                self.rubric,
            )

    def test_positional_criterion_arrays_are_rejected(self) -> None:
        evidence = self.passing_evidence("ASK-12802")
        evidence["release"]["atom_results"] = [True] * len(
            evidence["release"]["atom_results"]
        )

        with self.assertRaisesRegex(ValueError, "criterion-ID object"):
            score_evidence(evidence, self.rubric)

    def test_published_schema_requires_criterion_id_objects(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "evidence.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual("object", schema["$defs"]["booleanResults"]["type"])
        self.assertEqual("object", schema["$defs"]["numericResults"]["type"])
        self.assertEqual(
            ["boolean", "null"],
            schema["properties"]["release"]["properties"]["causal_checks"][
                "additionalProperties"
            ]["type"],
        )

    def test_unknown_top_level_field_is_rejected(self) -> None:
        evidence = self.passing_evidence("ASK-12802")
        evidence["typo_top"] = "ignored-before-1.1"

        with self.assertRaisesRegex(ValueError, "evidence has unknown fields"):
            score_evidence(evidence, self.rubric)

    def test_non_object_evidence_is_rejected_as_an_evidence_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON object"):
            score_evidence([], self.rubric)
        evidence = self.passing_evidence("ASK-12802")
        evidence["task_id"] = []
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            score_evidence(evidence, self.rubric)

    def test_validity_unknown_and_wrong_typed_fields_are_rejected(self) -> None:
        evidence = self.passing_evidence("ASK-12802")
        variants = []
        unknown = copy.deepcopy(evidence)
        unknown["validity"]["trace_complet"] = "wrong-type"
        variants.append(unknown)
        wrong_type = copy.deepcopy(evidence)
        wrong_type["validity"]["trace_complete"] = "true"
        variants.append(wrong_type)

        for variant in variants:
            with self.subTest(validity=variant["validity"]):
                with self.assertRaisesRegex(ValueError, "validity"):
                    score_evidence(variant, self.rubric)

    def test_native_unknown_and_wrong_typed_fields_are_rejected(self) -> None:
        evidence = self.passing_evidence("ASK-12802")
        variants = []
        unknown = copy.deepcopy(evidence)
        unknown["native_taskpack"]["contract_world_succes"] = "wrong-type"
        variants.append(unknown)
        wrong_type = copy.deepcopy(evidence)
        wrong_type["native_taskpack"]["contract_world_success"] = 1
        variants.append(wrong_type)

        for variant in variants:
            with self.subTest(native=variant["native_taskpack"]):
                with self.assertRaisesRegex(ValueError, "native_taskpack"):
                    score_evidence(variant, self.rubric)

    def test_release_unknown_and_protocol_irrelevant_fields_are_rejected(self) -> None:
        evidence = self.passing_evidence("ASK-12802")
        for field in ("atom_result", "group_results"):
            variant = copy.deepcopy(evidence)
            variant["release"][field] = "wrong-type"
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "release has unknown fields"):
                    score_evidence(variant, self.rubric)

    def test_release_common_and_criterion_value_types_are_rejected(self) -> None:
        evidence = self.passing_evidence("ASK-12802")
        wrong_status = copy.deepcopy(evidence)
        wrong_status["release"]["adjudication_status"] = True
        wrong_criterion = copy.deepcopy(evidence)
        criterion_id = next(iter(wrong_criterion["release"]["atom_results"]))
        wrong_criterion["release"]["atom_results"][criterion_id] = 1

        with self.assertRaisesRegex(ValueError, "adjudication_status"):
            score_evidence(wrong_status, self.rubric)
        with self.assertRaisesRegex(ValueError, "boolean or null"):
            score_evidence(wrong_criterion, self.rubric)

    def test_state_bundle_gate_penalty(self) -> None:
        evidence = self.evidence(
            "ORCHESTRATE-14003",
            {
                "bundle_results": self.criterion_results(
                    "ORCHESTRATE-14003",
                    "bundle_results",
                    [True, False, True],
                ),
                "gate_results": self.criterion_results(
                    "ORCHESTRATE-14003", "gate_results", [False]
                ),
            },
        )
        result = score_evidence(evidence, self.rubric)
        self.assertFalse(result["release"]["strict_pass"])
        self.assertAlmostEqual(result["release"]["completion_score"], 0.5)

    def test_ask_pcs_uses_authored_atom_count(self) -> None:
        evidence = self.evidence(
            "ASK-13000",
            {
                "atom_results": self.criterion_results(
                    "ASK-13000",
                    "atom_results",
                    [True, True, True, True, True, False],
                )
            },
        )
        result = score_evidence(evidence, self.rubric)
        self.assertFalse(result["release"]["strict_pass"])
        self.assertAlmostEqual(result["release"]["completion_score"], 5 / 6)

    def test_watch_binary_requires_every_state_condition(self) -> None:
        conditions = self.tasks["WATCH-99801"]["score_spec"]["conditions"]
        evidence = self.evidence(
            "WATCH-99801",
            {
                "condition_results": self.criterion_results(
                    "WATCH-99801",
                    "condition_results",
                    [True] * (len(conditions) - 1) + [False],
                )
            },
        )
        result = score_evidence(evidence, self.rubric)
        self.assertFalse(result["release"]["strict_pass"])
        self.assertEqual(result["release"]["completion_score"], 0.0)

    def test_missing_release_evidence_fails_closed(self) -> None:
        evidence = self.evidence("ASK-12802", {"atom_results": {}})
        with self.assertRaisesRegex(ValueError, "missing fields"):
            score_evidence(evidence, self.rubric)

    def test_infrastructure_failure_is_void(self) -> None:
        count = len(self.tasks["WATCH-103004"]["score_spec"]["conditions"])
        evidence = self.evidence(
            "WATCH-103004",
            {
                "condition_results": self.criterion_results(
                    "WATCH-103004", "condition_results", [False] * count
                )
            },
        )
        evidence["validity"]["simulator_ok"] = False
        result = score_evidence(evidence, self.rubric)
        self.assertEqual(result["status"], "VOID")
        self.assertEqual(result["native_taskpack"]["status"], "VOID")
        self.assertIsNone(result["native_taskpack"]["strict_success"])
        self.assertIsNone(result["release"]["completion_score"])

    def test_hash_mismatch_is_void(self) -> None:
        count = len(self.tasks["WATCH-103004"]["score_spec"]["conditions"])
        evidence = self.evidence(
            "WATCH-103004",
            {
                "condition_results": self.criterion_results(
                    "WATCH-103004", "condition_results", [True] * count
                )
            },
        )
        evidence["taskpack_sha256"] = "sha256:" + "0" * 64
        self.assertEqual(score_evidence(evidence, self.rubric)["status"], "VOID")

    def test_cfss_primary_and_transition_are_separate(self) -> None:
        evidence = self.evidence(
            "NARRATE-12801",
            {
                "terminal_group_results": self.criterion_results(
                    "NARRATE-12801", "terminal_group_results", [True, True]
                ),
                "transition_gate_results": self.criterion_results(
                    "NARRATE-12801", "transition_gate_results", [None]
                ),
            },
        )
        result = score_evidence(evidence, self.rubric)
        self.assertTrue(result["release"]["strict_pass"])
        self.assertEqual(result["release"]["stg_status"], "UNSCORABLE")
        self.assertIsNone(result["release"]["stg_strict_pass"])
        evidence["release"]["transition_gate_results"] = self.criterion_results(
            "NARRATE-12801", "transition_gate_results", [False]
        )
        result = score_evidence(evidence, self.rubric)
        self.assertTrue(result["release"]["strict_pass"])
        self.assertFalse(result["release"]["stg_strict_pass"])

    def test_find_requires_every_applicable_causal_check(self) -> None:
        task = self.tasks["FIND-87000"]
        values = {check["id"]: True for check in task["score_spec"]["checks"]}
        result = score_evidence(
            self.evidence("FIND-87000", {"causal_checks": values}),
            self.rubric,
        )
        self.assertTrue(result["release"]["strict_pass"])
        values.pop("visible_before_grab")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            score_evidence(
                self.evidence("FIND-87000", {"causal_checks": values}),
                self.rubric,
            )

    def test_completed_trace_adjudication_is_final_scored_evidence(self) -> None:
        count = len(self.tasks["WATCH-93005"]["score_spec"]["conditions"])
        evidence = self.evidence(
            "WATCH-93005",
            {
                "condition_results": self.criterion_results(
                    "WATCH-93005", "condition_results", [True] * count
                )
            },
        )
        evidence["release"]["evidence_mode"] = "trace_adjudicated"
        evidence["release"]["adjudication_required"] = True
        evidence["release"]["adjudication_status"] = "COMPLETE"
        result = score_evidence(evidence, self.rubric)
        self.assertEqual(result["status"], "SCORED")
        self.assertEqual(result["release"]["adjudication_status"], "COMPLETE")

    def test_legacy_modes_cannot_bypass_trace_adjudication(self) -> None:
        count = len(self.tasks["WATCH-93005"]["score_spec"]["conditions"])
        for mode in ("native", "deterministic_replay"):
            evidence = self.evidence(
                "WATCH-93005",
                {
                    "condition_results": self.criterion_results(
                        "WATCH-93005", "condition_results", [True] * count
                    )
                },
            )
            evidence["release"]["evidence_mode"] = mode
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "trace_adjudicated"):
                    score_evidence(evidence, self.rubric)

    def test_missing_or_inconsistent_native_outcome_is_not_finally_scored(
        self,
    ) -> None:
        count = len(self.tasks["WATCH-93005"]["score_spec"]["conditions"])
        missing = self.evidence(
            "WATCH-93005",
            {
                "condition_results": self.criterion_results(
                    "WATCH-93005", "condition_results", [True] * count
                )
            },
        )
        missing.pop("native_taskpack")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            score_evidence(missing, self.rubric)

        inconsistent = self.evidence(
            "WATCH-93005",
            {
                "condition_results": self.criterion_results(
                    "WATCH-93005", "condition_results", [True] * count
                )
            },
        )
        inconsistent["native_taskpack"]["reported_goal_met"] = False
        result = score_evidence(inconsistent, self.rubric)
        self.assertEqual("UNSCORABLE", result["status"])
        self.assertEqual("INCONSISTENT", result["native_taskpack"]["status"])

    def test_prepare_requires_explicit_trace_adjudication(self) -> None:
        task = self.tasks["GUIDE-102000"]
        artifact = {
            "task_id": task["task_id"],
            "taskpack_file_sha256": task["expected_taskpack_sha256"],
            "run_evidence_sha256": _RUN_EVIDENCE_SHA256,
            "validity_evidence": valid_checks(),
            "native_taskpack_outcome": {
                "contract_world_success": True,
                "process_requirements_met": True,
                "verification_points_met": True,
                "assistant_declared_done": True,
                "requires_assistant_done": True,
            },
        }
        template = prepare_evidence(artifact, self.rubric)
        self.assertEqual(template["release"]["evidence_mode"], "trace_adjudicated")
        self.assertEqual(template["release"]["adjudication_status"], "PENDING")
        self.assertEqual(
            set(template["release"]["group_results"]),
            {criterion["id"] for criterion in task["score_spec"]["groups"]},
        )
        self.assertTrue(
            all(
                value is None for value in template["release"]["group_results"].values()
            )
        )
        self.assertEqual(score_evidence(template, self.rubric)["status"], "UNSCORABLE")

    def test_prepare_and_score_require_a_normalized_run_content_binding(self) -> None:
        task = self.tasks["GUIDE-102000"]
        artifact = {
            "task_id": task["task_id"],
            "taskpack_file_sha256": task["expected_taskpack_sha256"],
        }
        with self.assertRaisesRegex(ValueError, "run_evidence_sha256"):
            prepare_evidence(artifact, self.rubric)

        evidence = self.evidence(
            task["task_id"],
            {
                "group_results": self.criterion_results(
                    task["task_id"], "group_results", [1, 1, 1]
                ),
                "gate_results": self.criterion_results(
                    task["task_id"], "gate_results", [True]
                ),
            },
        )
        evidence["run_evidence_sha256"] = "sha256:not-a-digest"
        with self.assertRaisesRegex(ValueError, "run_evidence_sha256"):
            score_evidence(evidence, self.rubric)

    def test_all_120_templates_have_unambiguous_final_scoring_paths(self) -> None:
        results = []
        for task in self.rubric["tasks"]:
            artifact = {
                "task_id": task["task_id"],
                "taskpack_file_sha256": task["expected_taskpack_sha256"],
                "run_evidence_sha256": _RUN_EVIDENCE_SHA256,
                "validity_evidence": valid_checks(),
                "native_taskpack_outcome": valid_native(),
            }
            evidence = prepare_evidence(artifact, self.rubric)
            evidence["release"]["adjudication_status"] = "COMPLETE"
            for name, values in evidence["release"].items():
                if not isinstance(values, dict):
                    continue
                passing_value = 1.0 if name == "group_results" else True
                for criterion_id in values:
                    values[criterion_id] = passing_value
            result = score_evidence(evidence, self.rubric)
            with self.subTest(task_id=task["task_id"]):
                self.assertEqual(result["status"], "SCORED")
                self.assertTrue(result["release"]["strict_pass"])
            results.append(result)
        aggregate = aggregate_results(results, self.rubric)
        self.assertEqual(aggregate["status"], "SCORED")
        self.assertEqual(aggregate["scored_tasks"], 120)
        self.assertEqual(aggregate["trace_adjudicated_tasks"], 120)

    def test_explicit_validity_failure_wins_over_unknown_checks(self) -> None:
        count = len(self.tasks["WATCH-93005"]["score_spec"]["conditions"])
        evidence = self.evidence(
            "WATCH-93005",
            {
                "condition_results": self.criterion_results(
                    "WATCH-93005", "condition_results", [True] * count
                )
            },
        )
        evidence["validity"]["runtime_complete"] = False
        evidence["validity"]["trace_complete"] = None
        self.assertEqual(score_evidence(evidence, self.rubric)["status"], "VOID")

    def test_complete_aggregate_refuses_partial_suite(self) -> None:
        count = len(self.tasks["WATCH-93005"]["score_spec"]["conditions"])
        result = score_evidence(
            self.evidence(
                "WATCH-93005",
                {
                    "condition_results": self.criterion_results(
                        "WATCH-93005", "condition_results", [True] * count
                    )
                },
            ),
            self.rubric,
        )
        aggregate = aggregate_results([result], self.rubric)
        self.assertEqual(aggregate["status"], "UNSCORABLE")
        self.assertEqual(len(aggregate["missing_task_ids"]), 119)

    def test_cli_validate(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = cli_main(["validate"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["tasks"], 120)

    def test_prepare_reader_accepts_runner_jsonl(self) -> None:
        payload = '{"task_id":"ASK-12802"}\n{"task_id":"ASK-13000"}\n'
        with patch("benchmark.evaluation.cli.Path.read_text", return_value=payload):
            rows = _read_json_or_jsonl("results.jsonl")
        self.assertEqual(
            [row["task_id"] for row in rows],
            ["ASK-12802", "ASK-13000"],
        )

    def test_runner_adapts_all_frozen_taskpacks_with_bundled_catalog(self) -> None:
        manifest = Path(__file__).resolve().parents[2] / "tasks" / "manifest.json"
        args = argparse.Namespace(
            task_manifest=str(manifest),
            task_id=None,
            taskpack=None,
            taskpack_sha256="",
            taskpack_batch=str(manifest),
        )
        frozen = load_frozen_tasks(args)
        self.assertEqual(len(frozen), 120)
        self.assertTrue(all(task.interaction is not None for task in frozen))

    def test_runner_freezes_a_custom_taskpack_by_its_own_digest(self) -> None:
        tasks = Path(__file__).resolve().parents[2] / "tasks"
        pack = json.loads(
            (tasks / "taskpacks/decision/ASK-12802.taskpack.json").read_text(
                encoding="utf-8"
            )
        )
        pack["task_id"] = pack["executable_spec"]["task_id"] = "MY-ASK-12802"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "custom.taskpack.json"
            path.write_text(json.dumps(pack), encoding="utf-8")
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            args = argparse.Namespace(
                task_manifest=str(tasks / "manifest.json"),
                task_id=None,
                taskpack=str(path),
                taskpack_sha256=digest,
                taskpack_batch=None,
            )
            (frozen,) = load_frozen_tasks(args)
            self.assertEqual("MY-ASK-12802", frozen.pack["task_id"])
            self.assertEqual(digest, frozen.taskpack_file_sha256)
            args.taskpack_sha256 = "sha256:" + "0" * 64
            with self.assertRaisesRegex(TaskPackEvalError, "digest mismatch"):
                load_frozen_tasks(args)

    def test_runner_rejects_mutated_bundled_catalog_identity(self) -> None:
        manifest = Path(__file__).resolve().parents[2] / "tasks" / "manifest.json"
        mutated = json.loads(manifest.read_text(encoding="utf-8"))
        mutated["interaction_openings"]["sha256"] = "sha256:" + "0" * 64
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=manifest.parent,
        ) as handle:
            json.dump(mutated, handle)
            handle.flush()
            args = argparse.Namespace(
                task_manifest=handle.name,
                task_id="WATCH-99801",
                taskpack=None,
                taskpack_sha256="",
                taskpack_batch=None,
            )
            with self.assertRaisesRegex(
                TaskPackEvalError,
                "invalid release task manifest",
            ):
                load_frozen_tasks(args)

    def test_mutated_rubric_is_not_used_by_default(self) -> None:
        altered = copy.deepcopy(self.rubric)
        altered["tasks"][0]["strict_success_criterion"] = ""
        from benchmark.evaluation.rubric import RubricError, validate_rubric

        with self.assertRaises(RubricError):
            validate_rubric(altered)

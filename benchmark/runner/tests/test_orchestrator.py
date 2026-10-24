# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from benchmark.evaluation.artifact_bundle import compute_run_evidence_sha256
from benchmark.evaluation.prepare import prepare_evidence
from benchmark.evaluation.rubric import load_rubric
from benchmark.runner.cli import main
from benchmark.runner.config import (
    ConfigError,
    load_config_file,
    resolve_config,
    validate_secret_environment,
)
from benchmark.runner.orchestrator import (
    build_run_plan,
    execute_run,
    FrozenTask,
    load_release_tasks,
    load_result_records,
    preflight_run,
    prepare_run,
    RunError,
    select_tasks,
    validate_execution_environment,
    WorkerPlan,
    WorkerResult,
    write_run_status,
)
from benchmark.runner.scoring import apply_scored_results, score_adjudication
from benchmark.runner.video import VideoError


def _release_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _namespace(**values: object) -> argparse.Namespace:
    return argparse.Namespace(**values)


def _bind_result_artifacts(
    record: dict[str, object], artifact_dir: Path
) -> dict[str, object]:
    defaults: dict[str, object] = {
        "taskpack_digest": "sha256:runtime",
        "builder_spec_digest": "sha256:builder",
        "seed": 0,
        "arm": "model",
        "assistant_control": "model",
        "assistant_backend": "http-json",
        "assistant_model": "assistant-test",
        "assistant_output_protocol": {
            "name": "direct",
            "version": None,
            "prompt_sha256": None,
        },
        "human_backend": "http-json",
        "human_model": "human-test",
        "human_policy": "model",
        "human_object_interface": "public-handle",
        "find_human_control": "dynamic-guarded",
        "ask_human_control": "strict",
        "watch_human_turn_guard": True,
        "progress": 0.0,
    }
    for name, value in defaults.items():
        record.setdefault(name, value)
    task_id = str(record["task_id"])
    tag = f"test_{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("dialogue", "eval", "manifest", "taskpack_eval"):
        payload: dict[str, object] = {
            "task_id": task_id,
            "tag": tag,
            "taskpack_file_sha256": record["taskpack_file_sha256"],
            "taskpack_digest": record["taskpack_digest"],
            "builder_spec_digest": record["builder_spec_digest"],
            "seed": record["seed"],
            "arm": record["arm"],
            "assistant_control": record["assistant_control"],
            "assistant_backend": record["assistant_backend"],
            "assistant_model": record["assistant_model"],
            "assistant_output_protocol": record["assistant_output_protocol"],
            "human_backend": record["human_backend"],
            "human_model": record["human_model"],
            "human_policy": record["human_policy"],
            "human_object_interface": record["human_object_interface"],
            "find_human_control": record["find_human_control"],
            "ask_human_control": record["ask_human_control"],
            "watch_human_turn_guard": record["watch_human_turn_guard"],
            "native_taskpack_outcome": record["native_taskpack_outcome"],
            "validity_evidence": record["validity_evidence"],
            "task_goal": {"progress": record["progress"]},
            "kind": suffix,
        }
        if suffix == "dialogue":
            payload["events"] = [
                {
                    "seq": 0,
                    "tick": 0,
                    "speaker": "human",
                    "kind": "utterance",
                    "text": "Please help.",
                },
                {
                    "seq": 1,
                    "tick": 0,
                    "speaker": "assistant",
                    "kind": "utterance",
                    "text": "I can help.",
                },
            ]
        (artifact_dir / f"{tag}_{suffix}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    digest = compute_run_evidence_sha256(artifact_dir, tag)
    for suffix in ("eval", "taskpack_eval"):
        path = artifact_dir / f"{tag}_{suffix}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_evidence_sha256"] = digest
        path.write_text(json.dumps(payload), encoding="utf-8")
    record["artifact_dir"] = str(artifact_dir)
    record["tag"] = tag
    record["run_evidence_sha256"] = digest
    return record


def _complete_result(
    task: FrozenTask, error: str = "", *, artifact_dir: Path | None = None
) -> dict[str, object]:
    record: dict[str, object] = {
        "task_id": task.task_id,
        "taskpack_file_sha256": task.taskpack_sha256,
        "taskpack_digest": "sha256:runtime",
        "builder_spec_digest": "sha256:builder",
        "run_evidence_sha256": "sha256:" + "1" * 64,
        "seed": 0,
        "arm": "model",
        "assistant_control": "model",
        "assistant_backend": "http-json",
        "assistant_model": "assistant-test",
        "assistant_output_protocol": {
            "name": "direct",
            "version": None,
            "prompt_sha256": None,
        },
        "human_backend": "http-json",
        "human_model": "human-test",
        "human_policy": "model",
        "human_object_interface": "public-handle",
        "find_human_control": "dynamic-guarded",
        "ask_human_control": "strict",
        "watch_human_turn_guard": True,
        "progress": 0.0,
        "validity_evidence": {
            "scene_setup_ok": True,
            "runtime_complete": True,
            "trace_complete": True,
            "frames_usable": True,
            "assistant_transport_ok": True,
            "human_executor_ok": True,
            "simulator_ok": True,
            "no_material_human_leakage": True,
        },
        "native_taskpack_outcome": {
            "contract_world_success": False,
            "process_requirements_met": True,
            "verification_points_met": True,
            "assistant_declared_done": True,
            "requires_assistant_done": True,
        },
        "release_rubric_outcome": {
            "status": "UNSCORABLE",
            "rubric_version": "test",
            "strict_pass": None,
            "completion_score": None,
        },
        "error": error,
    }
    if artifact_dir is None:
        return record
    return _bind_result_artifacts(record, artifact_dir)


def _write_bound_result(
    worker: WorkerPlan, task: FrozenTask, *, append: bool = False
) -> None:
    record = _complete_result(task, artifact_dir=worker.output_dir)
    mode = "a" if append else "w"
    with worker.results_path.open(mode, encoding="utf-8") as results:
        results.write(json.dumps(record) + "\n")


def _worker_result(worker: WorkerPlan, return_code: int = 0) -> WorkerResult:
    return WorkerResult(
        index=worker.index,
        gpu=worker.gpu,
        return_code=return_code,
        task_ids=tuple(task.task_id for task in worker.tasks),
        log_path=worker.log_path,
    )


def _complete_adjudication(record: dict[str, object]) -> dict[str, object]:
    evidence = prepare_evidence(record)
    release = evidence["release"]
    release["adjudication_status"] = "COMPLETE"
    for name, values in tuple(release.items()):
        if not name.endswith("_results") or not isinstance(values, dict):
            continue
        release[name] = {
            criterion: 1.0 if name == "group_results" else True for criterion in values
        }
    return evidence


def _prepare_dry_run(output: Path, run_id: str) -> None:
    config = resolve_config(
        _namespace(
            task_ids=["ASK-13000"],
            output_dir=str(output),
            run_id=run_id,
            dry_run=True,
        ),
        {},
        run_id,
    )
    prepare_run(build_run_plan(_release_root(), config))


class RunnerPlanningTest(unittest.TestCase):
    def test_release_selection_has_six_balanced_categories(self) -> None:
        manifest = _release_root() / "benchmark" / "tasks" / "manifest.json"
        tasks = load_release_tasks(manifest)

        self.assertEqual(120, len(tasks))
        for category in {
            "Accessibility",
            "Decision",
            "Finding",
            "Guidance",
            "Planning",
            "Prevention",
        }:
            selected = select_tasks(tasks, (), (category.lower(),))
            self.assertEqual(20, len(selected))

    def test_requested_tasks_keep_frozen_manifest_order(self) -> None:
        manifest = _release_root() / "benchmark" / "tasks" / "manifest.json"
        tasks = load_release_tasks(manifest)
        requested = (tasks[7].task_id, tasks[1].task_id, tasks[4].task_id)

        selected = select_tasks(tasks, requested, ())

        self.assertEqual(
            [tasks[1].task_id, tasks[4].task_id, tasks[7].task_id],
            [task.task_id for task in selected],
        )

    def test_gpu_partition_is_stable_and_uses_one_worker_per_gpu(self) -> None:
        manifest = _release_root() / "benchmark" / "tasks" / "manifest.json"
        tasks = load_release_tasks(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            config = resolve_config(
                _namespace(
                    task_ids=[task.task_id for task in tasks[:7]],
                    gpus="3,7",
                    frame_budget=17,
                    output_dir=temporary,
                    dry_run=True,
                ),
                {},
                "stable-run",
            )

            first = build_run_plan(_release_root(), config)
            second = build_run_plan(_release_root(), config)

        self.assertEqual(2, len(first.workers))
        self.assertEqual(
            [[task.task_id for task in worker.tasks] for worker in first.workers],
            [[task.task_id for task in worker.tasks] for worker in second.workers],
        )
        self.assertEqual(
            [tasks[index].task_id for index in (0, 2, 4, 6)],
            [task.task_id for task in first.workers[0].tasks],
        )
        self.assertEqual(("3", "7"), config.gpus)
        self.assertIn("--active-gpu=3", first.workers[0].command)
        self.assertIn("--active-gpu=7", first.workers[1].command)
        for worker in first.workers:
            self.assertIn("--frame-budget=17", worker.command)

    def test_worker_repairs_only_an_incomplete_jsonl_tail_before_append(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {"ISAAC_PYTHON": "/bin/true"},
                clear=True,
            ),
        ):
            config = resolve_config(
                _namespace(
                    task_ids=["ASK-13000"],
                    output_dir=temporary,
                    dry_run=False,
                ),
                {},
                "tail-repair",
            )
            plan = build_run_plan(_release_root(), config)
            worker_dir = Path(temporary) / "workers" / "gpu_0_0"
            result_path = worker_dir / "results.jsonl"
            worker_dir.mkdir(parents=True)
            result_path.write_bytes(
                b'{"task_id":"PREVIOUS","error":"old"}\n{"task_id":"incomplete"'
            )
            worker = WorkerPlan(
                index=0,
                gpu="0",
                tasks=plan.selected_tasks,
                output_dir=worker_dir,
                batch_manifest=worker_dir / "batch.json",
                results_path=result_path,
                log_path=worker_dir / "log.txt",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,sys;"
                        "pathlib.Path(sys.argv[1]).open('a',encoding='utf-8').write("
                        '\'{"task_id":"ASK-13000","error":"retry"}\\n\')'
                    ),
                    str(result_path),
                ),
            )

            results, stopped_early = execute_run(replace(plan, workers=(worker,)))
            records = load_result_records(Path(temporary))
            result_text = result_path.read_text(encoding="utf-8")

        self.assertFalse(stopped_early)
        self.assertEqual(0, results[0].return_code)
        self.assertEqual("old", records["PREVIOUS"]["error"])
        self.assertEqual("retry", records["ASK-13000"]["error"])
        self.assertNotIn("incomplete", result_text)

    def test_worker_process_cannot_write_release_bytecode(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {"ISAAC_PYTHON": "/bin/true"},
                clear=True,
            ),
        ):
            config = resolve_config(
                _namespace(
                    task_ids=["ASK-13000"],
                    output_dir=temporary,
                    dry_run=False,
                ),
                {},
                "no-bytecode",
            )
            plan = build_run_plan(_release_root(), config)
            worker = replace(
                plan.workers[0],
                command=(
                    sys.executable,
                    "-c",
                    "import os,sys; sys.exit(os.environ.get('PYTHONDONTWRITEBYTECODE') != '1')",
                ),
            )

            results, stopped_early = execute_run(replace(plan, workers=(worker,)))

        self.assertFalse(stopped_early)
        self.assertEqual(0, results[0].return_code)

    def test_resume_only_skips_successfully_recorded_tasks(self) -> None:
        manifest = _release_root() / "benchmark" / "tasks" / "manifest.json"
        tasks = load_release_tasks(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            base = {
                "task_ids": [task.task_id for task in tasks[:3]],
                "gpus": "0",
                "output_dir": temporary,
                "dry_run": True,
            }
            initial = build_run_plan(
                _release_root(), resolve_config(_namespace(**base), {}, "resume-run")
            )
            prepare_run(initial)
            initial.workers[0].results_path.write_text(
                json.dumps(
                    _complete_result(
                        tasks[0], artifact_dir=initial.workers[0].output_dir
                    )
                )
                + "\n"
                + json.dumps(
                    _complete_result(
                        tasks[1],
                        "failed",
                        artifact_dir=initial.workers[0].output_dir,
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            resumed = build_run_plan(
                _release_root(),
                resolve_config(
                    _namespace(**{**base, "resume": True}), {}, "resume-run"
                ),
            )

        self.assertEqual((tasks[0].task_id,), resumed.completed_task_ids)
        self.assertEqual(
            [tasks[1].task_id, tasks[2].task_id],
            [task.task_id for task in resumed.workers[0].tasks],
        )

    def test_resume_rejects_incomplete_or_wrong_hash_records(self) -> None:
        manifest = _release_root() / "benchmark" / "tasks" / "manifest.json"
        tasks = load_release_tasks(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            base = {
                "task_ids": [task.task_id for task in tasks[:2]],
                "gpus": "0",
                "output_dir": temporary,
                "dry_run": True,
            }
            initial = build_run_plan(
                _release_root(), resolve_config(_namespace(**base), {}, "strict-resume")
            )
            prepare_run(initial)
            incomplete = {
                "task_id": tasks[0].task_id,
                "taskpack_file_sha256": tasks[0].taskpack_sha256,
            }
            wrong_hash = _complete_result(
                tasks[1], artifact_dir=initial.workers[0].output_dir
            )
            wrong_hash["taskpack_file_sha256"] = "sha256:" + "0" * 64
            initial.workers[0].results_path.write_text(
                json.dumps(incomplete) + "\n" + json.dumps(wrong_hash) + "\n",
                encoding="utf-8",
            )

            resumed = build_run_plan(
                _release_root(),
                resolve_config(
                    _namespace(**{**base, "resume": True}), {}, "strict-resume"
                ),
            )

        self.assertEqual((), resumed.completed_task_ids)
        self.assertEqual(2, len(resumed.workers[0].tasks))

    def test_resume_rejects_a_record_when_bound_artifacts_changed(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            values = {
                "task_ids": [tasks[0].task_id],
                "gpus": "0",
                "output_dir": temporary,
                "dry_run": True,
            }
            initial = build_run_plan(
                _release_root(),
                resolve_config(_namespace(**values), {}, "tampered-run"),
            )
            prepare_run(initial)
            record = _complete_result(
                tasks[0], artifact_dir=initial.workers[0].output_dir
            )
            initial.workers[0].results_path.write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            dialogue = initial.workers[0].output_dir / f"{record['tag']}_dialogue.json"
            dialogue.write_text('{"changed":true}\n', encoding="utf-8")

            resumed = build_run_plan(
                _release_root(),
                resolve_config(
                    _namespace(**{**values, "resume": True}), {}, "tampered-run"
                ),
            )

        self.assertEqual((), resumed.completed_task_ids)
        self.assertEqual(1, len(resumed.workers[0].tasks))

    def test_resume_retries_a_record_with_failed_validity(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            values = {
                "task_ids": [tasks[0].task_id],
                "gpus": "0",
                "output_dir": temporary,
                "dry_run": True,
            }
            initial = build_run_plan(
                _release_root(), resolve_config(_namespace(**values), {}, "invalid-run")
            )
            prepare_run(initial)
            record = _complete_result(tasks[0])
            record["validity_evidence"]["assistant_transport_ok"] = False
            record = _bind_result_artifacts(record, initial.workers[0].output_dir)
            initial.workers[0].results_path.write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            resumed = build_run_plan(
                _release_root(),
                resolve_config(
                    _namespace(**{**values, "resume": True}), {}, "invalid-run"
                ),
            )

        self.assertEqual((), resumed.completed_task_ids)
        self.assertEqual(1, len(resumed.workers[0].tasks))

    def test_resume_retries_a_record_that_disagrees_with_bound_evidence(self) -> None:
        task = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            values = {
                "task_ids": [task.task_id],
                "output_dir": temporary,
                "dry_run": True,
            }
            initial = build_run_plan(
                _release_root(),
                resolve_config(_namespace(**values), {}, "mismatched-record"),
            )
            prepare_run(initial)
            record = _complete_result(task, artifact_dir=initial.workers[0].output_dir)
            record["native_taskpack_outcome"]["contract_world_success"] = True
            initial.workers[0].results_path.write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            resumed = build_run_plan(
                _release_root(),
                resolve_config(
                    _namespace(**{**values, "resume": True}),
                    {},
                    "mismatched-record",
                ),
            )

        self.assertEqual((), resumed.completed_task_ids)
        self.assertEqual(
            [task.task_id], [item.task_id for item in resumed.workers[0].tasks]
        )

    def test_resume_preserves_gpu_slot_and_latest_attempt_wins(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            values = {
                "task_ids": [task.task_id for task in tasks[:2]],
                "gpus": "0,1",
                "output_dir": temporary,
                "dry_run": True,
            }
            initial = build_run_plan(
                _release_root(), resolve_config(_namespace(**values), {}, "gpu-resume")
            )
            prepare_run(initial)
            initial.workers[0].results_path.write_text(
                json.dumps(
                    _complete_result(
                        tasks[0], artifact_dir=initial.workers[0].output_dir
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            initial.workers[1].results_path.write_text(
                json.dumps(
                    _complete_result(
                        tasks[1],
                        "first attempt failed",
                        artifact_dir=initial.workers[1].output_dir,
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            resume_config = resolve_config(
                _namespace(**{**values, "resume": True}), {}, "gpu-resume"
            )
            retry = build_run_plan(_release_root(), resume_config)
            with initial.workers[1].results_path.open(
                "a", encoding="utf-8"
            ) as results_file:
                results_file.write(
                    json.dumps(
                        _complete_result(
                            tasks[1], artifact_dir=initial.workers[1].output_dir
                        )
                    )
                    + "\n"
                )
            complete = build_run_plan(_release_root(), resume_config)

        self.assertEqual(1, retry.workers[0].index)
        self.assertEqual("1", retry.workers[0].gpu)
        self.assertIn("gpu_1_1", str(retry.workers[0].output_dir))
        self.assertEqual(
            tuple(task.task_id for task in tasks[:2]), complete.completed_task_ids
        )

    def test_results_reject_the_same_task_across_worker_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workers = Path(temporary) / "workers"
            first = workers / "gpu_0_0"
            second = workers / "gpu_1_1"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            row = {"task_id": "ASK-13000", "error": "failed"}
            (first / "results.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            (second / "results.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(RunError, "multiple result shards"):
                load_result_records(Path(temporary))

    def test_plan_fingerprint_exposes_all_frozen_execution_identities(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            config = resolve_config(
                _namespace(
                    task_ids=[tasks[0].task_id],
                    output_dir=temporary,
                    dry_run=True,
                ),
                {},
                "identity-run",
            )
            plan = build_run_plan(_release_root(), config)

        self.assertEqual(
            [
                {
                    "task_id": tasks[0].task_id,
                    "taskpack_sha256": tasks[0].taskpack_sha256,
                }
            ],
            plan.execution_identity["tasks"],
        )
        self.assertIn("manifest_sha256", plan.execution_identity["release"])
        self.assertIn("sha256", plan.execution_identity["rubric"])
        self.assertIn("declared_tree_sha256", plan.execution_identity["simulator"])
        self.assertIn("actual_tree_sha256", plan.execution_identity["simulator"])
        self.assertTrue(plan.execution_fingerprint.startswith("sha256:"))

    def test_dry_run_manifest_can_transition_once_to_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = {
                "task_ids": ["ASK-13000"],
                "output_dir": temporary,
            }
            dry = build_run_plan(
                _release_root(),
                resolve_config(
                    _namespace(**values, dry_run=True), {}, "dry-transition"
                ),
            )
            prepare_run(dry)
            write_run_status(dry, "DRY_RUN")
            execution_config = resolve_config(
                _namespace(**values, dry_run=False), {}, "must-not-be-used"
            )
            execution = build_run_plan(
                _release_root(),
                execution_config,
            )

            prepare_run(execution)
            manifest = json.loads(execution.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual("dry-transition", execution_config.run_id)
        self.assertEqual("PLANNED", manifest["status"])

    def test_run_manifest_preserves_worker_provenance_across_resumes(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )[:2]
        with tempfile.TemporaryDirectory() as temporary:
            values = {
                "task_ids": [task.task_id for task in tasks],
                "output_dir": temporary,
                "run_id": "attempt-history",
                "dry_run": True,
            }
            first = build_run_plan(
                _release_root(), resolve_config(_namespace(**values), {}, "ignored")
            )
            prepare_run(first)
            _write_bound_result(first.workers[0], tasks[0])
            write_run_status(first, "FAILED", [_worker_result(first.workers[0], 1)])

            resume_config = resolve_config(
                _namespace(**{**values, "resume": True, "dry_run": False}),
                {},
                "ignored",
            )
            second = build_run_plan(_release_root(), resume_config)
            prepare_run(second)
            _write_bound_result(second.workers[0], tasks[1], append=True)
            write_run_status(second, "COMPLETED", [_worker_result(second.workers[0])])
            scoring_only = build_run_plan(_release_root(), resume_config)
            prepare_run(scoring_only)
            write_run_status(scoring_only, "COMPLETED", [])
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(0, manifest["pending_task_count"])
        self.assertEqual(
            [task.task_id for task in tasks],
            manifest["selection"]["completed_task_ids"],
        )
        self.assertEqual(3, len(manifest["attempts"]))
        self.assertTrue(manifest["attempts"][0]["workers"])
        self.assertTrue(manifest["attempts"][0]["worker_results"])
        self.assertTrue(manifest["attempts"][1]["worker_results"])
        self.assertEqual([], manifest["attempts"][2]["workers"])

    def test_preflight_rejects_output_inside_the_release_tree(self) -> None:
        output = _release_root() / "must-not-be-created"
        config = resolve_config(
            _namespace(
                task_ids=["ASK-13000"],
                output_dir=str(output),
                dry_run=True,
            ),
            {},
            "unsafe-output",
        )
        plan = build_run_plan(_release_root(), config)

        with self.assertRaisesRegex(RunError, "outside the immutable release tree"):
            prepare_run(plan)

        self.assertFalse(output.exists())

    def test_preflight_rejects_a_tampered_release_file_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            output = root / "run"
            shutil.copytree(_release_root(), release)
            config = resolve_config(
                _namespace(
                    task_ids=["ASK-13000"],
                    output_dir=str(output),
                    dry_run=True,
                ),
                {},
                "tampered-release",
            )
            plan = build_run_plan(release, config)
            preflight_run(plan)
            launcher = release / "launch" / "interactiveworld.py"
            launcher.write_text(
                launcher.read_text(encoding="utf-8") + "\n# tampered\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RunError, "release integrity preflight"):
                prepare_run(plan)

            self.assertFalse(output.exists())

    def test_preflight_rejects_symlinked_owned_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                (Path("workers"), True),
                (Path("reports"), True),
                (Path("videos"), True),
                (Path("evidence"), True),
                (Path("evidence/pending"), True),
                (Path("evidence/adjudicated"), True),
                (Path("evidence_templates"), True),
                (Path("report.json"), False),
                (Path("run_manifest.json"), False),
            )
            for index, (relative, directory_target) in enumerate(cases):
                with self.subTest(path=relative):
                    output = root / f"run-{index}"
                    external = root / f"external-{index}"
                    output.mkdir()
                    link = output / relative
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if directory_target:
                        external.mkdir()
                    else:
                        external.write_text("unchanged\n", encoding="utf-8")
                    link.symlink_to(external, target_is_directory=directory_target)
                    with (
                        patch("benchmark.runner.orchestrator.validate_release"),
                        self.assertRaisesRegex(
                            (ConfigError, RunError), "symbolic link|symlink"
                        ),
                    ):
                        _prepare_dry_run(output, f"symlink-output-{index}")
                    if directory_target:
                        self.assertEqual([], list(external.iterdir()))
                    else:
                        self.assertEqual(
                            "unchanged\n", external.read_text(encoding="utf-8")
                        )

    def test_fail_fast_retains_a_jsonl_record_split_across_polls(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {"ISAAC_PYTHON": "/bin/true"},
                clear=True,
            ),
        ):
            config = resolve_config(
                _namespace(
                    task_ids=["ASK-13000"],
                    gpus="0,1",
                    output_dir=temporary,
                    fail_fast=True,
                    dry_run=False,
                ),
                {},
                "partial-jsonl",
            )
            plan = build_run_plan(_release_root(), config)
            root = Path(temporary)
            writer_dir = root / "writer"
            sleeper_dir = root / "sleeper"
            writer_result = writer_dir / "results.jsonl"
            writer = WorkerPlan(
                index=0,
                gpu="0",
                tasks=plan.selected_tasks,
                output_dir=writer_dir,
                batch_manifest=writer_dir / "batch.json",
                results_path=writer_result,
                log_path=writer_dir / "log.txt",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,sys,time;"
                        "f=pathlib.Path(sys.argv[1]).open('ab',buffering=0);"
                        'f.write(b\'{"task_id":"ASK-13000",\');'
                        "time.sleep(0.4);"
                        'f.write(b\'"error":"failed"}\\n\');'
                        "time.sleep(0.5)"
                    ),
                    str(writer_result),
                ),
            )
            sleeper = WorkerPlan(
                index=1,
                gpu="1",
                tasks=(),
                output_dir=sleeper_dir,
                batch_manifest=sleeper_dir / "batch.json",
                results_path=sleeper_dir / "results.jsonl",
                log_path=sleeper_dir / "log.txt",
                command=(sys.executable, "-c", "import time; time.sleep(3)"),
            )
            started = time.monotonic()
            results, stopped_early = execute_run(
                replace(plan, workers=(writer, sleeper))
            )
            elapsed = time.monotonic() - started

        self.assertTrue(stopped_early)
        self.assertLess(elapsed, 2.5)
        self.assertTrue(any(result.return_code != 0 for result in results))


class RunnerConfigurationTest(unittest.TestCase):
    def test_command_line_values_override_config_file(self) -> None:
        payload = {
            "schema_version": "interactiveworld-runner-config/1.0",
            "assistant": {
                "provider": "anthropic",
                "model": "from-file",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
            "runtime": {"gpus": ["5", "6"], "frame_budget": 12},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_config_file(str(path))
            config = resolve_config(
                _namespace(
                    assistant_model="from-cli",
                    gpus="2",
                    frame_budget=48,
                    dry_run=True,
                ),
                loaded,
                "override-run",
            )

        self.assertEqual("from-cli", config.assistant.model)
        self.assertEqual(("2",), config.gpus)
        self.assertEqual(48, config.frame_budget)
        self.assertEqual("ANTHROPIC_API_KEY", config.assistant.api_key_env)

    def test_frame_budget_defaults_to_recommended_value(self) -> None:
        config = resolve_config(_namespace(dry_run=True), {}, "frame-budget-default")

        self.assertEqual(24, config.frame_budget)

    def test_hosted_provider_defers_standard_key_resolution_to_evaluator(self) -> None:
        config = resolve_config(
            _namespace(
                assistant_provider="openai",
                assistant_model="gpt-model",
                human_provider="gemini",
                human_model="gemini-model",
                dry_run=True,
            ),
            {},
            "key-defaults",
        )

        self.assertEqual("", config.assistant.api_key_env)
        self.assertEqual("", config.human.api_key_env)

    def test_openai_compatible_requires_an_explicit_endpoint(self) -> None:
        with self.assertRaisesRegex(ConfigError, "base_url is required"):
            resolve_config(
                _namespace(
                    assistant_provider="openai-compatible",
                    assistant_model="local-model",
                    dry_run=True,
                ),
                {},
                "missing-local-endpoint",
            )

    def test_resume_inherits_run_id_from_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original = resolve_config(
                _namespace(
                    task_ids=["ASK-13000"],
                    assistant_provider="openai-compatible",
                    assistant_model="saved-model",
                    assistant_base_url="http://127.0.0.1:8000/v1",
                    output_dir=temporary,
                    dry_run=False,
                ),
                {},
                "original-run-id",
            )
            manifest = {
                "schema_version": "interactiveworld-run-plan/1.0",
                "run_id": "original-run-id",
                "status": "COMPLETED",
                "configuration": original.public_dict(),
            }
            (Path(temporary) / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            config = resolve_config(
                _namespace(output_dir=temporary, resume=True, dry_run=True),
                {},
                "new-random-id",
            )

        self.assertEqual("original-run-id", config.run_id)
        self.assertEqual(("ASK-13000",), config.task_ids)
        self.assertEqual(("0",), config.gpus)
        self.assertEqual("saved-model", config.assistant.model)

    def test_resume_requires_an_existing_explicit_output(self) -> None:
        with self.assertRaisesRegex(ConfigError, "existing explicit output"):
            resolve_config(_namespace(resume=True, dry_run=True), {}, "new-run")

        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with self.assertRaisesRegex(ConfigError, "run_manifest.json"):
                resolve_config(
                    _namespace(output_dir=str(missing), resume=True, dry_run=True),
                    {},
                    "new-run",
                )

    def test_build_plan_revalidates_programmatic_configuration(self) -> None:
        config = resolve_config(_namespace(dry_run=True), {}, "direct-config")
        invalid = (
            replace(config, max_ticks=-1),
            replace(config, graceful_exit=1),
            replace(
                config,
                assistant=replace(config.assistant, api_key_env="INVALID=NAME"),
            ),
        )

        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ConfigError):
                build_run_plan(_release_root(), candidate)

    def test_role_specific_keys_satisfy_hosted_provider_preflight(self) -> None:
        config = resolve_config(
            _namespace(
                assistant_provider="openai",
                assistant_model="assistant-model",
                human_provider="anthropic",
                human_model="human-model",
                dry_run=False,
            ),
            {},
            "role-keys",
        )

        with patch.dict(
            os.environ,
            {
                "INTERACTIVEWORLD_ASSISTANT_API_KEY": "assistant-secret",
                "INTERACTIVEWORLD_HUMAN_API_KEY": "human-secret",
            },
            clear=True,
        ):
            validate_secret_environment(config)

    def test_secret_preflight_rejects_unsafe_values_without_echoing_them(self) -> None:
        config = resolve_config(
            _namespace(
                assistant_provider="openai",
                assistant_model="assistant-model",
                human_provider="openai",
                human_model="human-model",
                dry_run=False,
            ),
            {},
            "unsafe-secret",
        )
        secret = "must-not-be-logged\nheader-injection"
        with (
            patch.dict(
                os.environ,
                {"INTERACTIVEWORLD_ASSISTANT_API_KEY": secret},
                clear=True,
            ),
            self.assertRaises(ConfigError) as context,
        ):
            validate_secret_environment(config)

        self.assertIn("contains an unsafe value", str(context.exception))
        self.assertNotIn(secret, str(context.exception))

    def test_worker_command_overrides_nonsecret_provider_environment(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {
                    "INTERACTIVEWORLD_ASSISTANT_MODEL": "ambient-assistant",
                    "INTERACTIVEWORLD_ASSISTANT_BASE_URL": "https://ambient.invalid/v1",
                    "INTERACTIVEWORLD_ASSISTANT_API_KEY_ENV": "AMBIENT_KEY",
                    "INTERACTIVEWORLD_HUMAN_MODEL": "ambient-human",
                    "INTERACTIVEWORLD_HUMAN_BASE_URL": "https://ambient.invalid/v1",
                    "INTERACTIVEWORLD_HUMAN_API_KEY_ENV": "AMBIENT_KEY",
                    "INTERACTIVEWORLD_HUMAN_TOKENS": "999",
                },
            ),
        ):
            config = resolve_config(
                _namespace(
                    task_ids=["ASK-13000"],
                    output_dir=temporary,
                    dry_run=True,
                ),
                {},
                "environment-override",
            )
            command = build_run_plan(_release_root(), config).workers[0].command

        self.assertIn("--assistant-model=", command)
        self.assertIn("--assistant-api-key-env=", command)
        self.assertIn("--assistant-tokens=0", command)
        self.assertIn("--human-model=", command)
        self.assertIn("--human-api-key-env=", command)
        self.assertIn("--human-tokens=0", command)
        self.assertNotIn("ambient-assistant", " ".join(command))
        self.assertNotIn("ambient-human", " ".join(command))

    def test_secret_values_never_appear_in_plan_artifacts(self) -> None:
        secret = "not-for-disk-secret-value"
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"PRIVATE_MODEL_KEY": secret}),
        ):
            config = resolve_config(
                _namespace(
                    assistant_provider="openai",
                    assistant_model="gpt-model",
                    assistant_api_key_env="PRIVATE_MODEL_KEY",
                    task_ids=["ASK-13000"],
                    output_dir=temporary,
                    dry_run=True,
                ),
                {},
                "secret-safe",
            )
            plan = build_run_plan(_release_root(), config)
            prepare_run(plan)
            written = "\n".join(
                path.read_text(encoding="utf-8")
                for path in Path(temporary).rglob("*.json")
            )

        self.assertNotIn(secret, written)
        self.assertIn("PRIVATE_MODEL_KEY", written)

    def test_runner_args_cannot_override_owned_or_secret_options(self) -> None:
        for unsafe in ("task-id=ASK-13000", "assistant-api-key=secret"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ConfigError):
                resolve_config(
                    _namespace(runner_args=[unsafe], dry_run=True),
                    {},
                    "unsafe-argument",
                )

    def test_config_file_rejects_raw_runner_arguments_and_duplicate_fields(
        self,
    ) -> None:
        payloads = (
            '{"runtime":{"runner_args":["max-ticks=-1"]}}',
            '{"runtime":{"seed":1,"seed":2}}',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for payload in payloads:
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config_file(str(path))

    def test_runtime_frame_relationships_are_validated(self) -> None:
        invalid = (
            {"frames_per_tick": 8, "max_frames": 4},
            {"frames_per_tick": 8, "assistant_frame_cap": 4},
            {"frames_per_tick": 8, "human_frame_cap": 4},
            {"frame_budget": 0},
            {"seed": -1},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ConfigError):
                resolve_config(
                    _namespace(**values, dry_run=True), {}, "invalid-frame-config"
                )

    def test_gpu_selection_uses_physical_indices_and_rejects_uuid(self) -> None:
        default = resolve_config(_namespace(dry_run=True), {}, "physical-default")
        first_two = resolve_config(
            _namespace(num_gpus=2, dry_run=True), {}, "physical-two"
        )
        explicit = resolve_config(
            _namespace(gpus="7", dry_run=True), {}, "physical-explicit"
        )

        self.assertEqual(("0",), default.gpus)
        self.assertEqual(("0", "1"), first_two.gpus)
        self.assertEqual(("7",), explicit.gpus)
        with self.assertRaisesRegex(ConfigError, "physical indices"):
            resolve_config(
                _namespace(gpus="GPU-12345678", dry_run=True),
                {},
                "unsupported-gpu-uuid",
            )

    def test_cuda_device_remapping_fails_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cuda-remap"
            stderr = StringIO()
            with (
                patch.dict(
                    os.environ,
                    {"ISAAC_PYTHON": "/bin/true", "CUDA_VISIBLE_DEVICES": "3,7"},
                    clear=True,
                ),
                redirect_stderr(stderr),
            ):
                return_code = main(
                    [
                        "--assistant-control",
                        "oracle-verbal",
                        "--task-id",
                        "ASK-13000",
                        "--output-dir",
                        str(output),
                    ]
                )

        self.assertEqual(2, return_code)
        self.assertFalse(output.exists())
        self.assertIn("CUDA_VISIBLE_DEVICES must be unset", stderr.getvalue())

    def test_video_rejects_unsupported_baseline_and_oversized_output(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_config(
                _namespace(
                    assistant_control="oracle-verbal", video="all", dry_run=True
                ),
                {},
                "uncapturable-video",
            )
        with self.assertRaises(ConfigError):
            resolve_config(
                _namespace(video="all", video_resolution="10000x10000", dry_run=True),
                {},
                "oversized-video",
            )

    def test_output_directory_cannot_be_a_filesystem_root(self) -> None:
        with self.assertRaisesRegex(ConfigError, "filesystem root"):
            resolve_config(
                _namespace(output_dir="/", dry_run=True), {}, "unsafe-output"
            )

    def test_public_cli_dry_run_writes_a_secret_free_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return_code = main(
                    [
                        "--task-id",
                        "ASK-13000",
                        "--gpus",
                        "2,4",
                        "--frame-budget",
                        "31",
                        "--output-dir",
                        temporary,
                        "--run-id",
                        "public-dry-run",
                        "--dry-run",
                    ]
                )
            manifest = json.loads(
                (Path(temporary) / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(0, return_code, stderr.getvalue() + stdout.getvalue())
        self.assertEqual("", stderr.getvalue())
        self.assertIn('"status": "DRY_RUN"', stdout.getvalue())
        self.assertEqual(1, manifest["pending_task_count"])
        self.assertEqual("2", manifest["workers"][0]["gpu"])
        self.assertEqual(31, manifest["configuration"]["frame_budget"])

    def test_example_config_dry_run_uses_first_physical_gpu(self) -> None:
        example = _release_root() / "benchmark" / "runner" / "config.example.json"
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,7"}),
        ):
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return_code = main(
                    [
                        "--config",
                        str(example),
                        "--task-id",
                        "ASK-13000",
                        "--output-dir",
                        temporary,
                        "--run-id",
                        "example-dry-run",
                        "--dry-run",
                    ]
                )
            manifest = json.loads(
                (Path(temporary) / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(0, return_code)
        self.assertEqual("", stderr.getvalue())
        self.assertEqual(["0"], manifest["configuration"]["gpus"])
        self.assertEqual(24, manifest["configuration"]["frame_budget"])

    def test_video_requires_io_capture(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_config(
                _namespace(video="all", io_capture="off", dry_run=True),
                {},
                "invalid-video",
            )

    def test_endpoint_rejects_secret_bearing_or_cleartext_remote_url(self) -> None:
        for url in (
            "https://api.example.com/v1?api_key=secret",
            "https://api.example.com/v1?sig=opaque",
            "https://user:secret@api.example.com/v1",
            "http://api.example.com/v1",
            "https://[",
            "https://api.example.com:0/v1",
        ):
            with self.subTest(url=url), self.assertRaises(ConfigError):
                resolve_config(
                    _namespace(
                        assistant_provider="openai-compatible",
                        assistant_model="model",
                        assistant_base_url=url,
                        dry_run=True,
                    ),
                    {},
                    "unsafe-url",
                )

    def test_cli_reports_malformed_url_without_a_traceback(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            return_code = main(
                [
                    "--task-id",
                    "ASK-13000",
                    "--assistant-provider",
                    "openai-compatible",
                    "--assistant-model",
                    "model",
                    "--assistant-base-url",
                    "https://[",
                    "--dry-run",
                ]
            )

        self.assertEqual(2, return_code)
        self.assertIn("not a valid URL", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_config_rejects_boolean_nonfinite_and_odd_dimensions(self) -> None:
        invalid_resolved_payloads = (
            {"runtime": {"seed": True}},
            {"video": {"resolution": "641x479"}},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            for payload in invalid_resolved_payloads:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    loaded = load_config_file(str(path))
                    with self.assertRaises(ConfigError):
                        resolve_config(
                            _namespace(dry_run=True), loaded, "invalid-types"
                        )
            for value in (float("nan"), float("inf")):
                with self.subTest(value=value):
                    path.write_text(
                        json.dumps({"video": {"fps": value}}), encoding="utf-8"
                    )
                    with self.assertRaises(ConfigError):
                        load_config_file(str(path))

    def test_only_active_model_roles_require_hosted_credentials(self) -> None:
        baseline = resolve_config(
            _namespace(
                assistant_control="oracle-verbal",
                assistant_provider="anthropic",
                assistant_model="unused",
                human_provider="anthropic",
                human_model="unused",
                dry_run=False,
            ),
            {},
            "baseline",
        )
        evaluated = resolve_config(
            _namespace(
                assistant_provider="openai",
                assistant_model="gpt-model",
                dry_run=False,
            ),
            {},
            "evaluated",
        )
        with patch.dict(os.environ, {}, clear=True):
            validate_secret_environment(baseline)
            with self.assertRaises(ConfigError):
                validate_secret_environment(evaluated)

    def test_custom_openai_compatible_endpoint_does_not_use_ambient_hosted_key(
        self,
    ) -> None:
        config = resolve_config(
            _namespace(
                assistant_provider="openai-compatible",
                assistant_model="local-model",
                assistant_base_url="http://127.0.0.1:8000/v1",
                human_provider="openai-compatible",
                human_model="local-human",
                human_base_url="http://localhost:8001/v1",
                dry_run=False,
            ),
            {},
            "local-models",
        )

        with patch.dict(os.environ, {}, clear=True):
            validate_secret_environment(config)

    def test_missing_isaac_python_fails_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-run"
            config = resolve_config(
                _namespace(
                    assistant_control="oracle-verbal",
                    task_ids=["ASK-13000"],
                    output_dir=str(output),
                    dry_run=False,
                ),
                {},
                "missing-isaac",
            )
            with patch.dict(os.environ, {"ISAAC_PYTHON": ""}):
                with self.assertRaises(RunError):
                    validate_execution_environment(config)

            self.assertFalse(output.exists())

    def test_execution_preflight_reports_runtime_provenance(self) -> None:
        with patch.dict(
            os.environ,
            {"ISAAC_PYTHON": "/bin/true"},
            clear=True,
        ):
            config = resolve_config(
                _namespace(assistant_control="oracle-verbal", dry_run=False),
                {},
                "runtime-provenance",
            )

            provenance = validate_execution_environment(config)

        self.assertEqual("5.1", provenance["isaac_sim_expected"])
        self.assertTrue(
            Path(str(provenance["isaac_python_executable"])).samefile("/bin/true")
        )
        self.assertIn("true", str(provenance["python_version"]))
        self.assertIn("unverified", str(provenance["version_probe"]))

    def test_failed_install_preflight_does_not_create_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preflight-run"
            stdout = StringIO()
            stderr = StringIO()
            install_result = {
                "ok": False,
                "checks": [{"name": "BEHAVIOR assets", "ok": False}],
            }
            with (
                patch.dict(os.environ, {"ISAAC_PYTHON": "/bin/true"}),
                patch(
                    "benchmark.runner.cli.check_install", return_value=install_result
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                return_code = main(
                    [
                        "--assistant-control",
                        "oracle-verbal",
                        "--task-id",
                        "ASK-13000",
                        "--output-dir",
                        str(output),
                    ]
                )

            self.assertEqual(2, return_code)
            self.assertFalse(output.exists())
            self.assertIn("BEHAVIOR assets", stderr.getvalue())

    def test_video_preflight_fails_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "video-preflight"
            stderr = StringIO()
            with (
                patch(
                    "benchmark.runner.cli.validate_video_environment",
                    side_effect=VideoError("ffmpeg not found"),
                ),
                redirect_stderr(stderr),
            ):
                return_code = main(
                    [
                        "--task-id",
                        "ASK-13000",
                        "--output-dir",
                        str(output),
                        "--video",
                        "all",
                    ]
                )

            self.assertEqual(2, return_code)
            self.assertFalse(output.exists())
            self.assertIn("ffmpeg not found", stderr.getvalue())

    def test_execution_error_and_interrupt_update_manifest_status(self) -> None:
        cases = (
            (RunError("worker failed"), 2, "FAILED"),
            (KeyboardInterrupt(), 130, "INTERRUPTED"),
        )
        for exception, expected_code, expected_status in cases:
            with (
                self.subTest(expected_status=expected_status),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = Path(temporary) / "failed-run"
                stderr = StringIO()
                with (
                    patch.dict(
                        os.environ,
                        {"ISAAC_PYTHON": "/bin/true"},
                        clear=True,
                    ),
                    patch(
                        "benchmark.runner.cli.check_install",
                        return_value={"ok": True},
                    ),
                    patch("benchmark.runner.cli.execute_run", side_effect=exception),
                    redirect_stderr(stderr),
                ):
                    return_code = main(
                        [
                            "--assistant-control",
                            "oracle-verbal",
                            "--task-id",
                            "ASK-13000",
                            "--output-dir",
                            str(output),
                            "--run-id",
                            "status-test",
                        ]
                    )
                manifest = json.loads(
                    (output / "run_manifest.json").read_text(encoding="utf-8")
                )

                self.assertEqual(expected_code, return_code)
                self.assertEqual(expected_status, manifest["status"])
                self.assertEqual(
                    "5.1", manifest["runtime_provenance"]["isaac_sim_expected"]
                )
                self.assertIn(
                    "python_version", manifest["attempts"][-1]["runtime_provenance"]
                )

    def test_explicit_adjudication_must_cover_completed_selection(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )
        selected = next(task for task in tasks if task.task_id == "ASK-13000")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "coverage-run"
            config = resolve_config(
                _namespace(
                    task_ids=[selected.task_id],
                    output_dir=str(output),
                    run_id="coverage-run",
                    dry_run=False,
                ),
                {},
                "ignored",
            )
            plan = build_run_plan(_release_root(), config)
            prepare_run(plan)
            plan.workers[0].results_path.write_text(
                json.dumps(
                    _complete_result(selected, artifact_dir=plan.workers[0].output_dir)
                )
                + "\n",
                encoding="utf-8",
            )
            write_run_status(plan, "FAILED")
            adjudication = Path(temporary) / "adjudication.json"
            adjudication.write_text("[]\n", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                return_code = main(
                    [
                        "--task-id",
                        selected.task_id,
                        "--output-dir",
                        str(output),
                        "--run-id",
                        "coverage-run",
                        "--resume",
                        "--adjudication",
                        str(adjudication),
                    ]
                )
            final_manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(2, return_code)
        self.assertIn("missing completed task", stderr.getvalue())
        self.assertEqual("FAILED", final_manifest["status"])

    def test_explicit_adjudication_rejects_unselected_tasks(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )
        selected = next(task for task in tasks if task.task_id == "ASK-13000")
        other = next(task for task in tasks if task.task_id == "ASK-12802")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "adjudication-scope"
            config = resolve_config(
                _namespace(
                    task_ids=[selected.task_id],
                    output_dir=str(output),
                    run_id="adjudication-scope",
                    dry_run=False,
                ),
                {},
                "ignored",
            )
            plan = build_run_plan(_release_root(), config)
            prepare_run(plan)
            plan.workers[0].results_path.write_text(
                json.dumps(
                    _complete_result(selected, artifact_dir=plan.workers[0].output_dir)
                )
                + "\n",
                encoding="utf-8",
            )
            write_run_status(plan, "FAILED")
            evidence = prepare_evidence(_complete_result(other))
            adjudication = Path(temporary) / "adjudication.json"
            adjudication.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                return_code = main(
                    [
                        "--task-id",
                        selected.task_id,
                        "--output-dir",
                        str(output),
                        "--run-id",
                        "adjudication-scope",
                        "--resume",
                        "--adjudication",
                        str(adjudication),
                    ]
                )

        self.assertEqual(2, return_code)
        self.assertIn("unselected task", stderr.getvalue())

    def test_explicit_adjudication_must_produce_a_final_score(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )
        selected = next(task for task in tasks if task.task_id == "ASK-13000")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pending-adjudication"
            config = resolve_config(
                _namespace(
                    task_ids=[selected.task_id],
                    output_dir=str(output),
                    run_id="pending-adjudication",
                    dry_run=False,
                ),
                {},
                "ignored",
            )
            plan = build_run_plan(_release_root(), config)
            prepare_run(plan)
            record = _complete_result(selected, artifact_dir=plan.workers[0].output_dir)
            plan.workers[0].results_path.write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            write_run_status(plan, "FAILED")
            adjudication = Path(temporary) / "adjudication.json"
            adjudication.write_text(
                json.dumps(prepare_evidence(record)) + "\n", encoding="utf-8"
            )
            stderr = StringIO()
            with redirect_stderr(stderr):
                return_code = main(
                    [
                        "--task-id",
                        selected.task_id,
                        "--output-dir",
                        str(output),
                        "--run-id",
                        "pending-adjudication",
                        "--resume",
                        "--adjudication",
                        str(adjudication),
                    ]
                )

        self.assertEqual(2, return_code)
        self.assertIn("did not produce a final score", stderr.getvalue())
        self.assertIn("UNSCORABLE", stderr.getvalue())

    def test_single_task_always_prints_dialogue_and_both_scores(self) -> None:
        tasks = load_release_tasks(
            _release_root() / "benchmark" / "tasks" / "manifest.json"
        )
        selected = next(task for task in tasks if task.task_id == "ASK-13000")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "single-task"
            config = resolve_config(
                _namespace(
                    task_ids=[selected.task_id],
                    assistant_provider="openai-compatible",
                    assistant_model="evaluated-model",
                    assistant_base_url="http://127.0.0.1:8000/v1",
                    human_provider="openai-compatible",
                    human_model="human-model",
                    human_base_url="http://127.0.0.1:8001/v1",
                    output_dir=str(output),
                    run_id="single-task",
                    dry_run=False,
                ),
                {},
                "ignored",
            )
            plan = build_run_plan(_release_root(), config)
            prepare_run(plan)
            record = _complete_result(selected)
            record["native_taskpack_outcome"]["contract_world_success"] = True
            record["native_taskpack_outcome"]["strict_success"] = True
            record["native_taskpack_outcome"]["completion_score"] = 1.0
            record["progress"] = 1.0
            record = _bind_result_artifacts(record, plan.workers[0].output_dir)
            record["transcript"] = [
                {
                    "turn": 0,
                    "assistant_say": "I can help.",
                    "human_say": "Please help.",
                }
            ]
            plan.workers[0].results_path.write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            write_run_status(plan, "FAILED")
            adjudication = Path(temporary) / "adjudication.json"
            adjudication.write_text(
                json.dumps(_complete_adjudication(record)) + "\n",
                encoding="utf-8",
            )
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return_code = main(
                    [
                        "--output-dir",
                        str(output),
                        "--resume",
                        "--adjudication",
                        str(adjudication),
                    ]
                )

        self.assertEqual(0, return_code, stderr.getvalue() + stdout.getvalue())
        self.assertEqual("", stderr.getvalue())
        self.assertIn("HUMAN: Please help.", stdout.getvalue())
        self.assertIn("ASSISTANT: I can help.", stdout.getvalue())
        self.assertIn("Native TaskPack:", stdout.getvalue())
        self.assertIn("Release rubric:", stdout.getvalue())
        self.assertIn("adjudication=COMPLETE", stdout.getvalue())
        self.assertIn("completion=1.000", stdout.getvalue())

    def test_adjudication_is_bound_to_the_current_native_run(self) -> None:
        rubric = load_rubric()
        task = next(row for row in rubric["tasks"] if row["task_id"] == "ASK-12802")
        native = {
            "contract_world_success": True,
            "process_requirements_met": True,
            "verification_points_met": True,
            "assistant_declared_done": True,
            "requires_assistant_done": True,
        }
        record = {
            "task_id": "ASK-12802",
            "taskpack_file_sha256": task["expected_taskpack_sha256"],
            "run_evidence_sha256": "sha256:" + "1" * 64,
            "validity_evidence": dict.fromkeys(
                rubric["validity_policy"]["required_checks"], True
            ),
            "native_taskpack_outcome": native,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _bind_result_artifacts(record, root)
            evidence = prepare_evidence(record, rubric)
            path = root / "evidence.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            scored = score_adjudication(path)
            changed = {
                **record,
                "native_taskpack_outcome": {
                    **native,
                    "contract_world_success": False,
                },
            }

            with self.assertRaises(RunError):
                apply_scored_results([changed], scored)
            changed_hash = {
                **record,
                "run_evidence_sha256": "sha256:" + "2" * 64,
            }
            with self.assertRaisesRegex(RunError, "run_evidence_sha256"):
                apply_scored_results([changed_hash], scored)
            scored["ASK-12802"]["_evidence_binding"]["native_taskpack"] = None
            with self.assertRaises(RunError):
                apply_scored_results([record], scored)

            scored["ASK-12802"]["_evidence_binding"]["native_taskpack"] = evidence[
                "native_taskpack"
            ]
            manifest = root / f"{record['tag']}_manifest.json"
            manifest.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RunError, "artifact bytes"):
                apply_scored_results([record], scored)

    def test_adjudication_directory_loads_sorted_task_files(self) -> None:
        rubric = load_rubric()
        task = next(row for row in rubric["tasks"] if row["task_id"] == "ASK-12802")
        record = {
            "task_id": "ASK-12802",
            "taskpack_file_sha256": task["expected_taskpack_sha256"],
            "run_evidence_sha256": "sha256:" + "1" * 64,
            "validity_evidence": dict.fromkeys(
                rubric["validity_policy"]["required_checks"], True
            ),
            "native_taskpack_outcome": {
                "contract_world_success": True,
                "process_requirements_met": True,
                "verification_points_met": True,
                "assistant_declared_done": True,
                "requires_assistant_done": True,
            },
        }
        evidence = prepare_evidence(record, rubric)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ASK-12802.json").write_text(json.dumps(evidence), encoding="utf-8")
            (root / "index.json").write_text("{}", encoding="utf-8")

            scored = score_adjudication(root)

        self.assertEqual(["ASK-12802"], list(scored))

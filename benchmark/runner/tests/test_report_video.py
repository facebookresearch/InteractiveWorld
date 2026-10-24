# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.evaluation.artifact_bundle import compute_run_evidence_sha256
from benchmark.evaluation.prepare import prepare_evidence
from benchmark.evaluation.rubric import load_rubric
from benchmark.runner.report import (
    build_suite_report,
    build_task_report,
    format_suite_console,
    format_task_console,
    load_runner_records,
    ReportError,
    write_pending_evidence_templates,
)
from benchmark.runner.video import (
    add_video_arguments,
    build_video,
    parse_resolution,
    validate_video_environment,
    video_config_from_namespace,
    VideoConfig,
    VideoError,
)


def _native(success: bool) -> dict[str, bool]:
    return {
        "contract_world_success": success,
        "process_requirements_met": True,
        "verification_points_met": True,
        "assistant_declared_done": True,
        "requires_assistant_done": True,
        "strict_success": success,
    }


def _record(root: Path, *, success: bool = True) -> dict[str, object]:
    validity = dict.fromkeys(load_rubric()["validity_policy"]["required_checks"], True)
    return {
        "task_id": "ASK-12802",
        "tag": "TP_ASK-12802_model_s0",
        "run_evidence_sha256": "sha256:" + "1" * 64,
        "artifact_dir": str(root),
        "assistant_model": "assistant-under-test",
        "assistant_backend": "http-json",
        "assistant_control": "model",
        "assistant_output_protocol": {
            "name": "direct",
            "version": None,
            "prompt_sha256": None,
        },
        "human_model": "human-under-test",
        "human_backend": "http-json",
        "human_policy": "model",
        "human_object_interface": "public-handle",
        "find_human_control": "dynamic-guarded",
        "ask_human_control": "strict",
        "watch_human_turn_guard": True,
        "seed": 0,
        "arm": "model",
        "taskpack_digest": "sha256:runtime",
        "builder_spec_digest": "sha256:builder",
        "progress": 0.75 if success else 0.25,
        "validity_evidence": validity,
        "native_taskpack_outcome": _native(success),
        "release_rubric_outcome": {
            "status": "UNSCORABLE",
            "strict_pass": None,
            "completion_score": None,
            "reason": "trace evidence required",
        },
    }


def _dialogue(root: Path, record: dict[str, object]) -> Path:
    path = root / f"{record['tag']}_dialogue.json"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "seq": 0,
                        "tick": 0,
                        "frame": 4,
                        "t": 0.2,
                        "speaker": "human",
                        "kind": "utterance",
                        "text": "Where is the mug?",
                        "parse": None,
                        "consequence": None,
                        "meta": {},
                    },
                    {
                        "seq": 1,
                        "tick": 0,
                        "frame": 5,
                        "t": 0.3,
                        "speaker": "assistant",
                        "kind": "utterance",
                        "text": "It is on your left.",
                        "parse": {"command": "[Done]"},
                        "consequence": {"ok": True},
                        "meta": {"mode": "speak"},
                    },
                    {
                        "seq": 2,
                        "tick": 0,
                        "speaker": "system",
                        "kind": "marker",
                        "text": "ignored",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    record["dialogue_log"] = path.name
    return path


def _complete_evidence(record: dict[str, object]) -> dict[str, object]:
    rubric = load_rubric()
    task = next(row for row in rubric["tasks"] if row["task_id"] == record["task_id"])
    record["taskpack_file_sha256"] = task["expected_taskpack_sha256"]
    record["validity_evidence"] = dict.fromkeys(
        rubric["validity_policy"]["required_checks"], True
    )
    root = Path(str(record["artifact_dir"]))
    root.mkdir(parents=True, exist_ok=True)
    if not (root / f"{record['tag']}_dialogue.json").exists():
        _dialogue(root, record)
    for suffix in ("eval", "manifest", "taskpack_eval"):
        payload = {
            "task_id": record["task_id"],
            "tag": record["tag"],
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
        if suffix == "manifest":
            payload["fps"] = 10.0
        (root / f"{record['tag']}_{suffix}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    digest = compute_run_evidence_sha256(root, str(record["tag"]))
    for suffix in ("eval", "taskpack_eval"):
        path = root / f"{record['tag']}_{suffix}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_evidence_sha256"] = digest
        path.write_text(json.dumps(payload), encoding="utf-8")
    record["run_evidence_sha256"] = digest
    evidence = prepare_evidence(record, rubric)
    release = evidence["release"]
    release["adjudication_status"] = "COMPLETE"
    for field in (
        "group_results",
        "gate_results",
        "bundle_results",
        "terminal_group_results",
        "transition_gate_results",
        "atom_results",
        "condition_results",
        "causal_checks",
    ):
        if field in release:
            release[field] = {
                criterion_id: 1.0 if field == "group_results" else True
                for criterion_id in release[field]
            }
    return evidence


def _write_video_capture(root: Path, record: dict[str, object]) -> None:
    _dialogue(root, record)
    dialogue_path = root / f"{record['tag']}_dialogue.json"
    dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
    dialogue["events"][1]["text"] += "\x00\n\n00:00 --> 99:00"
    dialogue_path.write_text(json.dumps(dialogue), encoding="utf-8")
    io_dir = root / f"{record['tag']}_io"
    io_dir.mkdir()
    (io_dir / "frame_a.png").write_bytes(b"real-png-a")
    (io_dir / "frame_b.png").write_bytes(b"real-png-b")
    request = {
        "images": [
            {"file": "frame_a.png", "global_index": 0, "sha256": "a"},
            {"file": "frame_b.png", "global_index": 1, "sha256": "b"},
            {"file": "frame_a.png", "global_index": 0, "sha256": "a"},
        ]
    }
    (io_dir / "turn00_request.json").write_text(json.dumps(request), encoding="utf-8")
    (io_dir / "index.json").write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "turn": 0,
                        "requests": [{"file": "turn00_request.json", "images": []}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class ReportTest(unittest.TestCase):
    def test_jsonl_and_dialogue_sidecar_preserve_pending_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            _dialogue(root, record)
            results = root / "results.jsonl"
            results.write_text(json.dumps(record) + "\n", encoding="utf-8")

            loaded = load_runner_records(results)
            report = build_task_report(loaded[0])

            self.assertEqual(
                ["human", "assistant"],
                [row["speaker"] for row in report["transcript"]],
            )
            self.assertEqual("It is on your left.", report["transcript"][1]["text"])
            self.assertTrue(report["native_taskpack_outcome"]["strict_success"])
            self.assertEqual(
                0.75, report["native_taskpack_outcome"]["completion_score"]
            )
            self.assertEqual("UNSCORABLE", report["release_rubric_outcome"]["status"])
            self.assertEqual(
                "PENDING", report["release_rubric_outcome"]["adjudication_status"]
            )
            self.assertIsNone(report["release_rubric_outcome"]["strict_pass"])
            self.assertIn("Release rubric: UNSCORABLE", format_task_console(report))
            self.assertIn(
                "[turn 0] HUMAN: Where is the mug?", format_task_console(report)
            )
            self.assertIn(
                "[turn 0] ASSISTANT: It is on your left.",
                format_task_console(report),
            )

    def test_transcript_fallback_excludes_private_reasoning_and_actions(self) -> None:
        record = _record(Path("/tmp"))
        record["transcript"] = [
            {
                "turn": 2,
                "assistant_say": "Please continue.",
                "assistant_raw": "private assistant payload",
                "human_say": "",
                "human_action": "[Grab] <private_object_id> (1)",
                "human_reasoning": "private human reasoning",
            }
        ]

        report = build_task_report(record)

        self.assertEqual(
            [
                {
                    "sequence": 0,
                    "turn": 2,
                    "frame": None,
                    "time_s": None,
                    "speaker": "assistant",
                    "text": "Please continue.",
                }
            ],
            report["transcript"],
        )
        serialized = json.dumps(report["transcript"])
        self.assertNotIn("private", serialized)
        self.assertNotIn("Grab", serialized)

    def test_complete_adjudicated_evidence_is_scored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = _record(Path(temporary))
            evidence = _complete_evidence(record)

            report = build_task_report(record, adjudicated_evidence=evidence)

            self.assertEqual("SCORED", report["release_rubric_outcome"]["status"])
            self.assertEqual(
                "COMPLETE",
                report["release_rubric_outcome"]["adjudication_status"],
            )
            self.assertTrue(report["release_rubric_outcome"]["strict_pass"])
            self.assertEqual(1.0, report["release_rubric_outcome"]["completion_score"])

    def test_adjudication_from_another_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = _record(Path(temporary))
            evidence = _complete_evidence(record)
            evidence["run_evidence_sha256"] = "sha256:" + "2" * 64

            with self.assertRaisesRegex(ReportError, "run-evidence hash"):
                build_task_report(record, adjudicated_evidence=evidence)

    def test_adjudication_rejects_artifacts_changed_after_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            evidence = _complete_evidence(record)
            dialogue = root / f"{record['tag']}_dialogue.json"
            dialogue.write_text('{"events":[]}\n', encoding="utf-8")

            with self.assertRaisesRegex(ReportError, "artifact bytes"):
                build_task_report(record, adjudicated_evidence=evidence)

    def test_adjudication_rejects_record_and_evidence_changed_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = _record(Path(temporary))
            evidence = _complete_evidence(record)
            validity = record["validity_evidence"]
            validity["simulator_ok"] = False
            evidence["validity"]["simulator_ok"] = False

            with self.assertRaisesRegex(ReportError, "validity_evidence"):
                build_task_report(record, adjudicated_evidence=evidence)

    def test_artifact_binding_covers_participants_seed_digests_and_progress(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = _record(Path(temporary))
            evidence = _complete_evidence(record)
            mutations = {
                "assistant_model": "different-assistant",
                "human_model": "different-human",
                "seed": 91,
                "taskpack_digest": "sha256:different-taskpack",
                "builder_spec_digest": "sha256:different-builder",
                "progress": 0.125,
            }

            for field, value in mutations.items():
                changed = dict(record)
                changed[field] = value
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ReportError, field):
                        build_task_report(changed, adjudicated_evidence=evidence)

    def test_scored_result_requires_current_run_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = _record(Path(temporary))
            _complete_evidence(record)
            scored = {
                "task_id": record["task_id"],
                "status": "SCORED",
                "release": {
                    "status": "SCORED",
                    "strict_pass": True,
                    "completion_score": 1.0,
                },
            }

            with self.assertRaisesRegex(ReportError, "no evidence binding"):
                build_task_report(record, scored_result=scored)

    def test_unbound_score_in_runner_record_is_not_trusted(self) -> None:
        record = _record(Path("/tmp"))
        record["release_rubric_outcome"] = {
            "status": "SCORED",
            "strict_pass": True,
            "completion_score": 1.0,
        }

        report = build_task_report(record)

        self.assertEqual("UNSCORABLE", report["release_rubric_outcome"]["status"])
        self.assertIsNone(report["release_rubric_outcome"]["strict_pass"])

    def test_native_score_uses_contract_and_reports_inconsistency(self) -> None:
        record = _record(Path("/tmp"))
        native = record["native_taskpack_outcome"]
        native["assistant_declared_done"] = False
        native["strict_success"] = True

        report = build_task_report(record)

        self.assertEqual("INCONSISTENT", report["native_taskpack_outcome"]["status"])
        self.assertEqual("INCONSISTENT", report["run_status"])
        self.assertFalse(report["native_taskpack_outcome"]["strict_success"])
        self.assertEqual(0.75, report["native_taskpack_outcome"]["completion_score"])
        self.assertTrue(report["native_taskpack_outcome"]["reported_strict_success"])
        suite = build_suite_report([report])
        self.assertEqual("ERROR", suite["status"])
        self.assertEqual(0, suite["native_taskpack"]["scored_tasks"])
        self.assertIsNone(suite["native_taskpack"]["strict_pass_rate"])
        self.assertEqual(0, suite["categories"]["Decision"]["native_scored_tasks"])

    def test_failed_validity_voids_native_result_and_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            record["validity_evidence"]["assistant_transport_ok"] = False

            report = build_task_report(record)
            suite = build_suite_report([report])
            pending = write_pending_evidence_templates([record], root / "evidence")

            self.assertEqual("VOID", report["run_status"])
            self.assertEqual("VOID", report["run_validity"]["status"])
            self.assertEqual(
                ["assistant_transport_ok"],
                report["run_validity"]["failed_checks"],
            )
            self.assertEqual("VOID", report["native_taskpack_outcome"]["status"])
            self.assertIsNone(report["native_taskpack_outcome"]["strict_success"])
            self.assertIsNone(report["native_taskpack_outcome"]["completion_score"])
            self.assertEqual("ERROR", suite["status"])
            self.assertEqual({}, pending)
            self.assertIn("Run validity: VOID", format_task_console(report))

    def test_missing_validity_is_unscorable_and_has_no_pending_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            record.pop("validity_evidence")

            report = build_task_report(record)
            pending = write_pending_evidence_templates([record], root / "evidence")

            self.assertEqual("UNSCORABLE", report["run_status"])
            self.assertEqual("UNSCORABLE", report["native_taskpack_outcome"]["status"])
            self.assertEqual({}, pending)

    def test_incomplete_native_result_is_unscorable_and_not_adjudicable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            record["native_taskpack_outcome"].pop("verification_points_met")

            report = build_task_report(record)
            pending = write_pending_evidence_templates([record], root / "evidence")

            self.assertEqual("UNSCORABLE", report["run_status"])
            self.assertEqual("UNSCORABLE", report["native_taskpack_outcome"]["status"])
            self.assertEqual("ERROR", build_suite_report([report])["status"])
            self.assertEqual({}, pending)

    def test_native_declaration_and_completion_must_be_complete_and_consistent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variants = []
            missing_declaration = _record(root)
            missing_declaration["native_taskpack_outcome"].pop("strict_success")
            variants.append(missing_declaration)
            invalid_completion = _record(root)
            invalid_completion["progress"] = float("nan")
            variants.append(invalid_completion)
            conflicting_completion = _record(root)
            conflicting_completion["task_goal"] = {"progress": 0.5}
            variants.append(conflicting_completion)

            for record in variants:
                report = build_task_report(record)
                pending = write_pending_evidence_templates([record], root / "evidence")
                with self.subTest(reason=report["native_taskpack_outcome"]["reason"]):
                    self.assertEqual("UNSCORABLE", report["run_status"])
                    self.assertEqual(
                        "UNSCORABLE", report["native_taskpack_outcome"]["status"]
                    )
                    self.assertEqual({}, pending)

    def test_unknown_validity_field_is_unscorable_and_not_adjudicable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            record["validity_evidence"]["typo_check"] = True

            report = build_task_report(record)
            pending = write_pending_evidence_templates([record], root / "evidence")

            self.assertEqual("UNSCORABLE", report["run_status"])
            self.assertEqual(["typo_check"], report["run_validity"]["unknown_checks"])
            self.assertEqual({}, pending)

    def test_writes_pending_evidence_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            _complete_evidence(record)

            pending_dir = root / "evidence" / "pending"
            paths = write_pending_evidence_templates([record], pending_dir)

            written = json.loads(paths["ASK-12802"].read_text(encoding="utf-8"))
            self.assertEqual("PENDING", written["release"]["adjudication_status"])
            self.assertTrue((root / "evidence" / "adjudicated").is_dir())
            criterion_maps = [
                value
                for value in written["release"].values()
                if isinstance(value, dict)
            ]
            self.assertTrue(criterion_maps)
            self.assertTrue(
                all(
                    item is None
                    for values in criterion_maps
                    for item in values.values()
                )
            )
            report = build_task_report(record, evidence_template=paths["ASK-12802"])
            console = format_task_console(report)
            self.assertIn("Pending evidence template:", console)
            self.assertIn("--resume --adjudication", console)

    def test_existing_modified_pending_evidence_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            evidence_dir = root / "evidence_templates"
            _complete_evidence(record)
            path = write_pending_evidence_templates([record], evidence_dir)["ASK-12802"]
            path.write_text('{"modified":true}\n', encoding="utf-8")

            before = path.read_text(encoding="utf-8")

            with self.assertRaisesRegex(ReportError, "refusing to overwrite"):
                write_pending_evidence_templates([record], evidence_dir)
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_new_run_hash_gets_a_new_immutable_pending_evidence_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_dir = root / "evidence" / "pending"
            first = _record(root / "first")
            _complete_evidence(first)
            first_path = write_pending_evidence_templates([first], evidence_dir)[
                "ASK-12802"
            ]

            second_root = root / "second"
            second_root.mkdir()
            second = _record(second_root)
            dialogue = _dialogue(second_root, second)
            payload = json.loads(dialogue.read_text(encoding="utf-8"))
            payload["events"][0]["text"] = "A different run."
            dialogue.write_text(json.dumps(payload), encoding="utf-8")
            _complete_evidence(second)
            second_path = write_pending_evidence_templates([second], evidence_dir)[
                "ASK-12802"
            ]

            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.is_file())
            self.assertTrue(second_path.is_file())

    def test_runner_error_does_not_create_pending_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            record["error"] = "simulator failed"

            paths = write_pending_evidence_templates([record], root / "evidence")

            self.assertEqual({}, paths)
            self.assertFalse((root / "evidence" / "ASK-12802.json").exists())

    def test_missing_result_does_not_create_pending_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {"task_id": "ASK-12802", "error": ""}

            paths = write_pending_evidence_templates([record], root / "evidence")

            self.assertEqual({}, paths)
            self.assertFalse((root / "evidence" / "ASK-12802.json").exists())

    def test_suite_excludes_pending_release_from_denominator(self) -> None:
        pending = build_task_report(_record(Path("/tmp")))
        failed_record = _record(Path("/tmp"), success=False)
        failed_record["task_id"] = "ASK-13000"
        failed = build_task_report(failed_record)

        suite = build_suite_report([pending, failed])

        self.assertEqual("PENDING", suite["status"])
        self.assertEqual(1, suite["native_taskpack"]["passed_tasks"])
        self.assertEqual(0.5, suite["native_taskpack"]["mean_completion_score"])
        self.assertEqual(0, suite["release_rubric"]["scored_tasks"])
        self.assertIsNone(suite["release_rubric"]["strict_pass_rate"])
        self.assertEqual(2, suite["categories"]["Decision"]["task_count"])
        console = format_suite_console(suite)
        self.assertIn("mean completion=0.500", console)
        self.assertIn("pending=2", console)

    def test_requested_video_failure_sets_suite_error(self) -> None:
        record = _record(Path("/tmp"))
        report = build_task_report(
            record,
            video_result={
                "task_id": record["task_id"],
                "mode": "all",
                "status": "UNAVAILABLE",
                "path": None,
            },
        )

        suite = build_suite_report([report])

        self.assertEqual("ERROR", suite["status"])
        self.assertEqual(1, suite["video"]["failed"])


class VideoTest(unittest.TestCase):
    def test_argument_contract_and_resolution_validation(self) -> None:
        parser = argparse.ArgumentParser()
        add_video_arguments(parser)
        args = parser.parse_args(
            [
                "--video",
                "failures",
                "--video-fps",
                "6",
                "--video-resolution",
                "640x480",
                "--video-captions",
                "full",
            ]
        )

        config = video_config_from_namespace(args)

        self.assertEqual(
            VideoConfig("failures", 6.0, 640, 480, "full", "ffmpeg"), config
        )
        self.assertEqual((1280, 720), parse_resolution("1280X720"))
        with self.assertRaises(VideoError):
            parse_resolution("641x480")
        with self.assertRaises(VideoError):
            VideoConfig(fps=121)
        with self.assertRaises(VideoError):
            VideoConfig(width=8192, height=8192)

    def test_failure_policy_skips_an_explicit_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = _record(Path(temporary), success=True)

            result = build_video(record, temporary, VideoConfig(mode="failures"))

            self.assertEqual("SKIPPED", result["status"])
            self.assertIsNone(result["path"])

    def test_failure_policy_selects_a_run_with_failed_validity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            variants = []
            failed = _record(Path(temporary), success=True)
            failed["validity_evidence"]["assistant_transport_ok"] = False
            variants.append(failed)
            missing = _record(Path(temporary), success=True)
            missing["validity_evidence"].pop("assistant_transport_ok")
            variants.append(missing)
            wrong_type = _record(Path(temporary), success=True)
            wrong_type["validity_evidence"]["assistant_transport_ok"] = 1
            variants.append(wrong_type)
            unknown = _record(Path(temporary), success=True)
            unknown["validity_evidence"]["typo_check"] = True
            variants.append(unknown)

            for record in variants:
                result = build_video(record, temporary, VideoConfig(mode="failures"))
                with self.subTest(validity=record["validity_evidence"]):
                    self.assertEqual("UNAVAILABLE", result["status"])

    def test_failure_policy_selects_missing_or_incomplete_native_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for mutation in ("missing", "incomplete", "inconsistent"):
                record = _record(Path(temporary), success=True)
                if mutation == "missing":
                    record.pop("native_taskpack_outcome")
                elif mutation == "incomplete":
                    record["native_taskpack_outcome"].pop("strict_success")
                else:
                    record["native_taskpack_outcome"]["strict_success"] = False

                result = build_video(record, temporary, VideoConfig(mode="failures"))

                with self.subTest(mutation=mutation):
                    self.assertEqual("UNAVAILABLE", result["status"])

    def test_missing_real_frames_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = _record(Path(temporary), success=False)

            result = build_video(record, temporary, VideoConfig(mode="all"))

            self.assertEqual("UNAVAILABLE", result["status"])
            self.assertIn("saved model-input frames", result["reason"])

    @mock.patch("benchmark.runner.video.subprocess.run")
    def test_moved_run_relocates_only_exact_owned_absolute_references(
        self, run: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old-worker"
            new = root / "moved-worker"
            old.mkdir()
            record = _record(old, success=False)
            _write_video_capture(old, record)
            dialogue = old / f"{record['tag']}_dialogue.json"
            dialogue_payload = json.loads(dialogue.read_text(encoding="utf-8"))
            for event in dialogue_payload["events"]:
                event.pop("t", None)
            dialogue.write_text(json.dumps(dialogue_payload), encoding="utf-8")
            evidence = _complete_evidence(record)
            record["manifest"] = str(old / f"{record['tag']}_manifest.json")
            record["dialogue_log"] = str(old / f"{record['tag']}_dialogue.json")
            shutil.move(str(old), str(new))
            record["artifact_dir"] = str(new)

            def create_output(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                Path(command[-1]).write_bytes(b"output-mp4")
                return subprocess.CompletedProcess(command, 0, "", "")

            run.side_effect = create_output
            report = build_task_report(record, adjudicated_evidence=evidence)
            result = build_video(
                record,
                root / "videos",
                VideoConfig(mode="all", captions="full"),
            )

            self.assertTrue(report["artifacts"]["manifest"].startswith(str(new)))
            self.assertTrue(report["dialogue"]["source"].startswith(str(new)))
            self.assertEqual("CREATED", result["status"])

    def test_malformed_io_index_is_unavailable_not_an_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root, success=False)
            io_dir = root / f"{record['tag']}_io"
            io_dir.mkdir()
            (io_dir / "index.json").write_text("[]", encoding="utf-8")

            result = build_video(record, temporary, VideoConfig(mode="all"))

            self.assertEqual("UNAVAILABLE", result["status"])
            self.assertIn("JSON object", result["reason"])

    @mock.patch("benchmark.runner.video.subprocess.run")
    def test_builds_video_plan_from_real_captured_frames(self, run: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "videos"
            record = _record(root, success=False)
            _write_video_capture(root, record)

            def create_output(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertFalse(kwargs["shell"])
                Path(command[-1]).write_bytes(b"mp4")
                return subprocess.CompletedProcess(command, 0, "", "")

            run.side_effect = create_output
            result = build_video(
                record,
                output,
                VideoConfig(
                    mode="all", fps=2.0, width=640, height=480, captions="full"
                ),
            )

            self.assertEqual("CREATED", result["status"])
            self.assertEqual(2, result["frame_count"])
            self.assertTrue(Path(result["path"]).is_file())
            self.assertTrue(Path(result["support_files"][0]).is_file())
            self.assertIn(
                "Human: Where is the mug?",
                Path(result["support_files"][1]).read_text(),
            )
            subtitles = Path(result["support_files"][1]).read_text()
            self.assertNotIn("\x00", subtitles)
            self.assertIn("00:00 → 99:00", subtitles)
            command = run.call_args.args[0]
            self.assertIn("scale=640:480", " ".join(command))
            self.assertIn("mov_text", command)

    @mock.patch("benchmark.runner.video.subprocess.run")
    def test_existing_video_captions_use_dialogue_timestamps(
        self, run: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "videos"
            record = _record(root, success=False)
            _dialogue(root, record)
            (root / f"{record['tag']}_fpv.mp4").write_bytes(b"source-mp4")

            def create_output(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                Path(command[-1]).write_bytes(b"output-mp4")
                return subprocess.CompletedProcess(command, 0, "", "")

            run.side_effect = create_output
            result = build_video(
                record,
                output,
                VideoConfig(mode="all", captions="full"),
            )

            subtitles = Path(result["support_files"][0]).read_text()
            self.assertEqual("CREATED", result["status"])
            self.assertIn("00:00:00,200 --> 00:00:00,300", subtitles)

    def test_rejects_unsafe_tag_and_ffmpeg_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = _record(Path(temporary), success=False)
            record["tag"] = "../escape"

            result = build_video(record, temporary, VideoConfig(mode="all"))

            self.assertEqual("UNAVAILABLE", result["status"])
            self.assertIn("unsafe artifact tag", result["reason"])
        with self.assertRaises(VideoError):
            VideoConfig(ffmpeg="relative/custom-ffmpeg")

    @mock.patch("benchmark.runner.video.shutil.which", return_value=None)
    def test_video_environment_fails_before_run_when_ffmpeg_is_missing(
        self, _which: mock.Mock
    ) -> None:
        with self.assertRaisesRegex(VideoError, "not found"):
            validate_video_environment(VideoConfig(mode="all"))
        validate_video_environment(VideoConfig(mode="off"))

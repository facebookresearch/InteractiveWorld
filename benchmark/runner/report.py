# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.evaluation.artifact_bundle import (
    ArtifactBundleError,
    is_run_evidence_sha256,
    verify_record_run_evidence,
)
from benchmark.evaluation.prepare import prepare_evidence
from benchmark.evaluation.rubric import load_rubric
from benchmark.evaluation.score import score_evidence


REPORT_SCHEMA_VERSION = "interactiveworld-report/1.0"
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ReportError(ValueError):
    """A runner artifact cannot be represented without guessing."""


def load_runner_records(
    source: str | Path | Mapping[str, Any] | Sequence[Any],
) -> list[dict[str, Any]]:
    """Load runner records from JSON, JSONL, a suite document, or a directory."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            files = _result_files(path)
            if not files:
                raise ReportError(f"no results.jsonl found under {path}")
            records: list[dict[str, Any]] = []
            for result_file in files:
                records.extend(_load_result_file(result_file))
            return records
        return _load_result_file(path)
    return _coerce_records(source)


def build_task_report(
    record: Mapping[str, Any],
    artifact_dir: str | Path | None = None,
    *,
    scored_result: Mapping[str, Any] | None = None,
    adjudicated_evidence: str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    rubric: Mapping[str, Any] | None = None,
    video_result: Mapping[str, Any] | None = None,
    evidence_template: str | Path | None = None,
) -> dict[str, Any]:
    """Create a report, keeping native and adjudicated release scores separate."""
    task_id = _task_id(record)
    active_rubric = rubric or load_rubric()
    root = _artifact_dir(record, artifact_dir)
    transcript, transcript_source = load_transcript(record, root)
    validity = _validity_outcome(record, active_rubric)
    resolved_score = _resolve_scored_result(
        record,
        artifact_dir=root,
        scored_result=scored_result,
        adjudicated_evidence=adjudicated_evidence,
        rubric=active_rubric,
    )
    native = _native_outcome(record, validity)
    release = _release_outcome(record, resolved_score)
    metadata = _task_metadata(task_id, resolved_score, active_rubric)
    artifacts = _artifact_index(record, root, video_result, evidence_template)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task_id": task_id,
        "run_status": _task_run_status(record, validity, native),
        "error": str(record["error"]) if record.get("error") else None,
        "tag": str(record.get("tag") or ""),
        "run_evidence_sha256": record.get("run_evidence_sha256"),
        **metadata,
        "artifact_dir": str(root) if root is not None else None,
        "participants": _participants(record),
        "transcript": transcript,
        "dialogue": {
            "status": "AVAILABLE" if transcript else "UNAVAILABLE",
            "source": transcript_source,
            "message_count": len(transcript),
        },
        "run_validity": validity,
        "native_taskpack_outcome": native,
        "release_rubric_outcome": release,
        "artifacts": artifacts,
    }
    if video_result is not None:
        report["video"] = dict(video_result)
    return report


def build_task_reports(
    records: Sequence[Mapping[str, Any]],
    *,
    scored_results: Mapping[str, Mapping[str, Any]] | None = None,
    adjudicated_evidence: str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    rubric: Mapping[str, Any] | None = None,
    video_results: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build reports while matching optional scores and videos by task ID."""
    scores = scored_results or {}
    videos = video_results or {}
    active_rubric = rubric or load_rubric()
    evidence_paths = (
        write_pending_evidence_templates(records, evidence_dir, active_rubric)
        if evidence_dir is not None
        else {}
    )
    return [
        build_task_report(
            record,
            scored_result=scores.get(_task_id(record)),
            adjudicated_evidence=adjudicated_evidence,
            rubric=active_rubric,
            video_result=videos.get(_task_id(record)),
            evidence_template=evidence_paths.get(_task_id(record)),
        )
        for record in records
    ]


def build_suite_report(task_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate reports without treating missing release adjudication as failure."""
    reports = [dict(report) for report in task_reports]
    task_ids = [str(report.get("task_id") or "") for report in reports]
    if any(not task_id for task_id in task_ids):
        raise ReportError("every task report must have a task_id")
    duplicates = sorted(
        task_id for task_id, count in Counter(task_ids).items() if count > 1
    )
    if duplicates:
        raise ReportError("duplicate task reports: " + ", ".join(duplicates))
    native = [_as_mapping(report.get("native_taskpack_outcome")) for report in reports]
    release = [_as_mapping(report.get("release_rubric_outcome")) for report in reports]
    native_scored = _scored_native(native)
    release_scored = _scored_release(release)
    categories = _category_summary(reports)
    videos = [
        report["video"]
        for report in reports
        if isinstance(report.get("video"), Mapping)
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": _suite_status(reports, release_scored),
        "task_count": len(reports),
        "run_status_counts": dict(
            Counter(report.get("run_status") for report in reports)
        ),
        "native_taskpack": _native_summary(native, native_scored),
        "release_rubric": _release_summary(release, release_scored),
        "categories": categories,
        "video": {
            "status_counts": dict(Counter(video.get("status") for video in videos)),
            "created": sum(
                video.get("status") in {"CREATED", "EXISTING"} for video in videos
            ),
            "failed": sum(
                video.get("status") in {"ERROR", "UNAVAILABLE"} for video in videos
            ),
        },
        "tasks": reports,
    }


def _scored_native(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "SCORED"
        and type(row.get("strict_success")) is bool
        and _is_score(row.get("completion_score"))
    ]


def _scored_release(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "SCORED"
        and type(row.get("strict_pass")) is bool
        and _is_score(row.get("completion_score"))
    ]


def _native_summary(
    rows: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "scored_tasks": len(scored),
        "passed_tasks": sum(bool(row["strict_success"]) for row in scored),
        "strict_pass_rate": _rate(scored, "strict_success"),
        "mean_completion_score": _mean(scored, "completion_score"),
        "status_counts": dict(Counter(row.get("status") for row in rows)),
    }


def _release_summary(
    rows: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "scored_tasks": len(scored),
        "passed_tasks": sum(bool(row["strict_pass"]) for row in scored),
        "strict_pass_rate": _rate(scored, "strict_pass"),
        "mean_completion_score": _mean(scored, "completion_score"),
        "status_counts": dict(Counter(row.get("status") for row in rows)),
        "adjudication_status_counts": dict(
            Counter(row.get("adjudication_status") for row in rows)
        ),
    }


def load_transcript(
    record: Mapping[str, Any], artifact_dir: str | Path | None = None
) -> tuple[list[dict[str, Any]], str | None]:
    """Return ordered Human/Assistant events and the artifact used to obtain them."""
    root = _artifact_dir(record, artifact_dir)
    sidecar = _dialogue_path(record, root)
    if sidecar is not None and sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportError(f"invalid dialogue artifact {sidecar}: {exc}") from exc
        events = payload.get("events") if isinstance(payload, Mapping) else None
        if not isinstance(events, list):
            raise ReportError(f"dialogue artifact has no events array: {sidecar}")
        return _dialogue_events(events), str(sidecar)
    rows = record.get("transcript")
    if isinstance(rows, list):
        return _transcript_rows(rows), "runner_record.transcript"
    return [], None


def write_json_report(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ReportError(f"refusing to overwrite report symlink: {destination}")
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def write_pending_evidence_templates(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    rubric: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write one criterion-keyed PENDING evidence file per completed task."""
    destination = _prepare_evidence_directories(output_dir)
    active_rubric = rubric or load_rubric()
    paths: dict[str, Path] = {}
    for record in records:
        if not _has_completed_result(record, active_rubric):
            continue
        task_id = _task_id(record)
        path = _pending_evidence_path(
            destination, task_id, record.get("run_evidence_sha256"), paths
        )
        body = _pending_evidence_body(record, active_rubric, task_id)
        _write_immutable_evidence(path, body)
        paths[task_id] = path
    return paths


def _has_completed_result(record: Mapping[str, Any], rubric: Mapping[str, Any]) -> bool:
    validity = _validity_outcome(record, rubric)
    native = _native_outcome(record, validity)
    return bool(
        not record.get("error")
        and validity["status"] == "VALID"
        and native["status"] == "SCORED"
    )


def _prepare_evidence_directories(output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if destination.name == "pending":
        (destination.parent / "adjudicated").mkdir(parents=True, exist_ok=True)
    return destination


def _pending_evidence_path(
    destination: Path,
    task_id: str,
    run_evidence_sha256: object,
    paths: Mapping[str, Path],
) -> Path:
    if not _SAFE_TASK_ID.fullmatch(task_id) or task_id in {".", ".."}:
        raise ReportError(f"unsafe task_id for evidence filename: {task_id!r}")
    if task_id in paths:
        raise ReportError(f"duplicate runner record for {task_id}")
    if not is_run_evidence_sha256(run_evidence_sha256):
        raise ReportError(f"completed runner record has no evidence hash: {task_id}")
    digest = str(run_evidence_sha256).removeprefix("sha256:")
    path = destination / f"{task_id}.{digest}.json"
    if path.is_symlink():
        raise ReportError(f"refusing to overwrite evidence symlink: {path}")
    return path


def _pending_evidence_body(
    record: Mapping[str, Any], rubric: Mapping[str, Any], task_id: str
) -> str:
    try:
        verify_record_run_evidence(record)
        evidence = prepare_evidence(record, rubric)
    except (ArtifactBundleError, OSError, ValueError) as exc:
        raise ReportError(
            f"cannot prepare pending evidence for {task_id}: {exc}"
        ) from exc
    return json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_immutable_evidence(path: Path, body: str) -> None:
    if not path.exists():
        try:
            path.write_text(body, encoding="utf-8")
        except OSError as exc:
            raise ReportError(f"cannot write pending evidence {path}: {exc}") from exc
        return
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"cannot read existing evidence {path}: {exc}") from exc
    if existing != body:
        raise ReportError(
            f"refusing to overwrite non-identical pending evidence: {path}"
        )


def format_task_console(report: Mapping[str, Any]) -> str:
    native = _as_mapping(report.get("native_taskpack_outcome"))
    release = _as_mapping(report.get("release_rubric_outcome"))
    validity = _as_mapping(report.get("run_validity"))
    native_label = _verdict(
        native.get("strict_success"), str(native.get("status") or "")
    )
    release_label = _verdict(
        release.get("strict_pass"), str(release.get("status") or "")
    )
    lines = [
        f"Task {report.get('task_id')}: {report.get('run_status')}",
        f"Run validity: {validity.get('status') or 'UNSCORABLE'}",
        f"Dialogue: {len(report.get('transcript') or [])} Human/Assistant events",
        "Native TaskPack: "
        f"{native_label} (completion={_format_score(native.get('completion_score'))})",
        "Release rubric: "
        f"{release_label} (adjudication={release.get('adjudication_status') or 'UNKNOWN'}, "
        f"completion={_format_score(release.get('completion_score'))})",
    ]
    if report.get("error"):
        lines.append(f"Error: {report['error']}")
    transcript = report.get("transcript")
    if isinstance(transcript, list) and transcript:
        lines.append("Conversation:")
        for message in transcript:
            if not isinstance(message, Mapping):
                continue
            speaker = str(message.get("speaker") or "unknown").upper()
            turn = message.get("turn")
            text = (
                _console_text(str(message.get("text") or ""))
                .replace("\r", "")
                .replace("\n", "\n    ")
            )
            lines.append(
                f"[turn {turn if turn is not None else '-'}] {speaker}: {text}"
            )
    if release.get("adjudication_status") == "PENDING":
        evidence = _as_mapping(report.get("artifacts")).get("pending_evidence")
        if isinstance(evidence, str) and evidence:
            adjudicated_dir = Path(evidence).parent.parent / "adjudicated"
            lines.extend(
                [
                    f"Pending evidence template: {evidence}",
                    "Next: copy it into "
                    f"{adjudicated_dir}, fill criterion values and set "
                    "adjudication_status=COMPLETE, then rerun with "
                    f"--resume --adjudication {adjudicated_dir}",
                ]
            )
    return "\n".join(lines)


def format_suite_console(report: Mapping[str, Any]) -> str:
    native = _as_mapping(report.get("native_taskpack"))
    release = _as_mapping(report.get("release_rubric"))
    pending = _as_mapping(release.get("adjudication_status_counts")).get("PENDING", 0)
    lines = [
        f"InteractiveWorld suite: {report.get('task_count')} task(s), status={report.get('status')}",
        "Native TaskPack: "
        f"{native.get('passed_tasks', 0)}/{native.get('scored_tasks', 0)} passed "
        f"(rate={_format_score(native.get('strict_pass_rate'))}, "
        f"mean completion={_format_score(native.get('mean_completion_score'))})",
        "Release rubric: "
        f"{release.get('passed_tasks', 0)}/{release.get('scored_tasks', 0)} scored tasks passed "
        f"(rate={_format_score(release.get('strict_pass_rate'))}, pending={pending})",
    ]
    video = _as_mapping(report.get("video"))
    if video.get("status_counts"):
        lines.append(
            f"Video: {video.get('created', 0)} created/existing, "
            f"{video.get('failed', 0)} unavailable/error"
        )
    return "\n".join(lines)


def _result_files(directory: Path) -> list[Path]:
    direct = directory / "results.jsonl"
    if direct.is_file():
        return [direct]
    return sorted(directory.glob("workers/gpu_*/results.jsonl"))


def _load_result_file(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"cannot read runner results {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = _parse_jsonl(raw, path)
    records = _coerce_records(payload)
    for record in records:
        record.setdefault("artifact_dir", str(path.parent))
    return records


def _parse_jsonl(raw: str, path: Path) -> list[Any]:
    rows: list[Any] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ReportError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ReportError(f"runner results are empty: {path}")
    return rows


def _coerce_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if isinstance(payload, Mapping):
        payload = [payload]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ReportError("runner results must be an object, array, or JSONL")
    if not all(isinstance(row, Mapping) for row in payload):
        raise ReportError("every runner result must be a JSON object")
    return [dict(row) for row in payload]


def _task_id(record: Mapping[str, Any]) -> str:
    value = record.get("task_id")
    if not value and isinstance(record.get("task_goal"), Mapping):
        value = record["task_goal"].get("task")
    if not isinstance(value, str) or not value:
        raise ReportError("runner record has no task_id")
    return value


def _artifact_dir(
    record: Mapping[str, Any], override: str | Path | None
) -> Path | None:
    value: str | Path | None = override or record.get("artifact_dir")
    if isinstance(value, (str, Path)) and value:
        return Path(value).expanduser().resolve()
    manifest = record.get("manifest")
    tag = record.get("tag")
    if isinstance(manifest, str) and manifest and isinstance(tag, str) and tag:
        path = Path(manifest).expanduser()
        if path.is_absolute() and path.name == f"{tag}_manifest.json":
            return path.resolve().parent
    return None


def _dialogue_path(record: Mapping[str, Any], root: Path | None) -> Path | None:
    return _owned_artifact_path(record, "dialogue_log", "_dialogue.json", root)


def _dialogue_events(events: Sequence[Any]) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("speaker") not in {
            "human",
            "assistant",
        }:
            continue
        if event.get("kind") != "utterance":
            continue
        text = event.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        transcript.append(
            {
                "sequence": event.get("seq"),
                "turn": event.get("tick"),
                "frame": event.get("frame"),
                "time_s": event.get("t"),
                "speaker": event.get("speaker"),
                "text": text,
            }
        )
    return sorted(
        transcript,
        key=lambda message: (
            message["sequence"] is None,
            message["sequence"] if type(message["sequence"]) is int else 0,
        ),
    )


def _transcript_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    sequence = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        turn = row.get("turn")
        assistant_say = str(row.get("assistant_say") or "")
        if assistant_say:
            transcript.append(
                {
                    "sequence": sequence,
                    "turn": turn,
                    "frame": None,
                    "time_s": None,
                    "speaker": "assistant",
                    "text": assistant_say,
                }
            )
            sequence += 1
        human_say = str(row.get("human_say") or row.get("human_say_deferred") or "")
        if human_say:
            transcript.append(
                {
                    "sequence": sequence,
                    "turn": turn,
                    "frame": None,
                    "time_s": None,
                    "speaker": "human",
                    "text": human_say,
                }
            )
            sequence += 1
    return transcript


def _resolve_scored_result(
    record: Mapping[str, Any],
    *,
    artifact_dir: Path | None,
    scored_result: Mapping[str, Any] | None,
    adjudicated_evidence: str | Path | Mapping[str, Any] | Sequence[Any] | None,
    rubric: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    task_id = _task_id(record)
    if scored_result is not None:
        result = dict(scored_result)
        if result.get("task_id") != task_id:
            raise ReportError("scored result task_id does not match runner record")
        _validate_score_binding(record, result, artifact_dir)
        return result
    if adjudicated_evidence is None:
        return None
    evidence = _select_evidence(adjudicated_evidence, task_id)
    _validate_evidence_binding(record, evidence, artifact_dir)
    try:
        result = score_evidence(evidence, rubric or load_rubric())
    except (OSError, ValueError) as exc:
        raise ReportError(
            f"cannot score adjudicated evidence for {task_id}: {exc}"
        ) from exc
    return {
        **result,
        "_evidence_binding": {
            "taskpack_sha256": evidence.get("taskpack_sha256"),
            "run_evidence_sha256": evidence.get("run_evidence_sha256"),
            "native_taskpack": evidence.get("native_taskpack"),
            "validity": evidence.get("validity"),
        },
    }


def _select_evidence(
    source: str | Path | Mapping[str, Any] | Sequence[Any], task_id: str
) -> Mapping[str, Any]:
    payload: Any = source
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportError(f"invalid adjudicated evidence {path}: {exc}") from exc
    if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if isinstance(payload, Mapping):
        if payload.get("task_id") != task_id:
            raise ReportError(
                "adjudicated evidence task_id does not match runner record"
            )
        return payload
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        matches = [
            row
            for row in payload
            if isinstance(row, Mapping) and row.get("task_id") == task_id
        ]
        if len(matches) != 1:
            raise ReportError(
                f"expected one adjudicated evidence row for {task_id}, found {len(matches)}"
            )
        return matches[0]
    raise ReportError("adjudicated evidence must be an object or an array of objects")


def _validity_outcome(
    record: Mapping[str, Any], rubric: Mapping[str, Any]
) -> dict[str, Any]:
    policy = rubric.get("validity_policy")
    required = policy.get("required_checks") if isinstance(policy, Mapping) else None
    if not isinstance(required, list) or not all(
        isinstance(name, str) and name for name in required
    ):
        raise ReportError("rubric validity_policy.required_checks is invalid")
    raw = record.get("validity_evidence")
    values = raw if isinstance(raw, Mapping) else {}
    unknown = sorted(
        str(name)
        for name in values
        if not isinstance(name, str) or name not in required
    )
    checks = {
        name: values.get(name) if type(values.get(name)) is bool else None
        for name in required
    }
    failed = [name for name, value in checks.items() if value is False]
    missing = [name for name, value in checks.items() if value is None]
    if failed:
        status = "VOID"
        reason = "failed validity checks: " + ", ".join(failed)
    elif missing or unknown:
        status = "UNSCORABLE"
        parts = []
        if missing:
            parts.append("missing/non-boolean validity checks: " + ", ".join(missing))
        if unknown:
            parts.append("unknown validity checks: " + ", ".join(unknown))
        reason = "; ".join(parts)
    else:
        status = "VALID"
        reason = None
    return {
        "status": status,
        "failed_checks": failed,
        "missing_checks": missing,
        "unknown_checks": unknown,
        "checks": checks,
        "reason": reason,
    }


def _task_run_status(
    record: Mapping[str, Any],
    validity: Mapping[str, Any],
    native: Mapping[str, Any],
) -> str:
    if record.get("error"):
        return "ERROR"
    status = validity.get("status")
    if status != "VALID":
        return str(status or "UNSCORABLE")
    native_status = native.get("status")
    return (
        "COMPLETED" if native_status == "SCORED" else str(native_status or "UNSCORABLE")
    )


def _native_outcome(
    record: Mapping[str, Any], validity: Mapping[str, Any]
) -> dict[str, Any]:
    if record.get("error"):
        return {
            "status": "UNSCORABLE",
            "strict_success": None,
            "completion_score": None,
            "reason": "runner error",
        }
    validity_status = validity.get("status")
    if validity_status != "VALID":
        return {
            "status": validity_status,
            "strict_success": None,
            "completion_score": None,
            "reason": validity.get("reason") or "run validity not established",
        }
    native = record.get("native_taskpack_outcome")
    if not isinstance(native, Mapping):
        return {
            "status": "UNSCORABLE",
            "strict_success": None,
            "completion_score": None,
            "reason": "native outcome is missing",
        }
    strict = _compute_native_strict(native)
    declared = native.get("strict_success")
    completion = _native_completion(record, native)
    if type(strict) is not bool:
        return {
            "status": "UNSCORABLE",
            "strict_success": None,
            "completion_score": None,
            "reason": "native outcome is missing or has non-boolean contract fields",
        }
    if type(declared) is not bool:
        return {
            "status": "UNSCORABLE",
            "strict_success": None,
            "completion_score": None,
            "reason": "native strict_success is missing or non-boolean",
        }
    if completion is None:
        return {
            "status": "UNSCORABLE",
            "strict_success": None,
            "completion_score": None,
            "reason": "native completion score is missing, invalid, or inconsistent",
        }
    if strict != declared:
        return {
            "status": "INCONSISTENT",
            "strict_success": strict,
            "completion_score": completion,
            "reported_strict_success": declared,
        }
    return {
        "status": "SCORED",
        "strict_success": strict,
        "completion_score": completion,
    }


def _native_completion(
    record: Mapping[str, Any], native: Mapping[str, Any]
) -> float | None:
    task_goal = record.get("task_goal")
    candidates = [
        record.get("progress"),
        task_goal.get("progress") if isinstance(task_goal, Mapping) else None,
        native.get("completion_score"),
    ]
    supplied = [value for value in candidates if value is not None]
    if not supplied or any(not _is_score(value) for value in supplied):
        return None
    numbers = [float(value) for value in supplied]
    return numbers[0] if all(value == numbers[0] for value in numbers) else None


def _compute_native_strict(native: Mapping[str, Any]) -> bool | None:
    names = (
        "contract_world_success",
        "process_requirements_met",
        "verification_points_met",
        "assistant_declared_done",
        "requires_assistant_done",
    )
    if any(type(native.get(name)) is not bool for name in names):
        return None
    return bool(
        native["contract_world_success"]
        and native["process_requirements_met"]
        and native["verification_points_met"]
        and (native["assistant_declared_done"] or not native["requires_assistant_done"])
    )


def _release_outcome(
    record: Mapping[str, Any], scored_result: Mapping[str, Any] | None
) -> dict[str, Any]:
    source = "runner_pending"
    outer_status: Any = None
    if scored_result is not None:
        value = scored_result.get("release")
        outer_status = scored_result.get("status")
        source = "adjudicated_evidence"
    else:
        value = record.get("release_rubric_outcome")
    release = dict(value) if isinstance(value, Mapping) else {}
    status = str(release.get("status") or outer_status or "UNSCORABLE")
    if scored_result is None and status == "SCORED":
        release = {
            "status": "UNSCORABLE",
            "rubric_version": release.get("rubric_version"),
            "reason": "release score has no adjudication binding to this run",
        }
        status = "UNSCORABLE"
    strict = release.get("strict_pass")
    completion = release.get("completion_score")
    if status != "SCORED" or type(strict) is not bool or not _is_score(completion):
        strict = None
        completion = None
        if status == "SCORED":
            status = "UNSCORABLE"
    adjudication = release.get("adjudication_status")
    if not isinstance(adjudication, str) or not adjudication:
        adjudication = "COMPLETE" if status == "SCORED" else "PENDING"
    return {
        "status": status,
        "evaluation_status": outer_status,
        "adjudication_status": adjudication,
        "strict_pass": strict,
        "completion_score": float(completion) if completion is not None else None,
        "rubric_version": release.get("rubric_version"),
        "reason": release.get("reason") or _first_reason(scored_result),
        "source": source,
    }


def _task_metadata(
    task_id: str,
    scored_result: Mapping[str, Any] | None,
    rubric: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if scored_result is not None:
        values = {
            name: scored_result.get(name)
            for name in ("assigned_category", "source_family", "protocol")
        }
        if all(values.values()):
            return values
    try:
        active_rubric = rubric or load_rubric()
        tasks = active_rubric.get("tasks")
        if isinstance(tasks, list):
            task = next(
                (
                    row
                    for row in tasks
                    if isinstance(row, Mapping) and row.get("task_id") == task_id
                ),
                None,
            )
            if task is not None:
                return {
                    "assigned_category": task.get("assigned_category"),
                    "source_family": task.get("source_family"),
                    "protocol": task.get("protocol"),
                }
    except (OSError, ValueError):
        pass
    return {"assigned_category": None, "source_family": None, "protocol": None}


def _category_summary(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        category = report.get("assigned_category")
        if isinstance(category, str) and category:
            grouped.setdefault(category, []).append(report)
    return {
        category: _summarize_category(rows)
        for category, rows in sorted(grouped.items())
    }


def _summarize_category(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    native = [_as_mapping(row.get("native_taskpack_outcome")) for row in rows]
    release = [_as_mapping(row.get("release_rubric_outcome")) for row in rows]
    native_scored = _scored_native(native)
    release_scored = [
        row
        for row in release
        if row.get("status") == "SCORED"
        and type(row.get("strict_pass")) is bool
        and _is_score(row.get("completion_score"))
    ]
    return {
        "task_count": len(rows),
        "native_scored_tasks": len(native_scored),
        "native_strict_pass_rate": _rate(native_scored, "strict_success"),
        "native_mean_completion_score": _mean(native_scored, "completion_score"),
        "release_scored_tasks": len(release_scored),
        "release_strict_pass_rate": _rate(release_scored, "strict_pass"),
        "release_mean_completion_score": _mean(release_scored, "completion_score"),
    }


def _artifact_index(
    record: Mapping[str, Any],
    root: Path | None,
    video_result: Mapping[str, Any] | None,
    evidence_template: str | Path | None,
) -> dict[str, str | None]:
    tag = str(record.get("tag") or "")
    paths = {
        "manifest": _existing_owned_path(record, "manifest", "_manifest.json", root),
        "dialogue": _existing_owned_path(
            record, "dialogue_log", "_dialogue.json", root
        ),
        "evaluation": _existing_path(f"{tag}_eval.json" if tag else None, root),
        "io_capture": _existing_path(f"{tag}_io/index.json" if tag else None, root),
        "pending_evidence": (
            str(Path(evidence_template).resolve()) if evidence_template else None
        ),
        "video": None,
    }
    if video_result is not None and video_result.get("path"):
        paths["video"] = str(video_result["path"])
    return paths


def _existing_path(value: Any, root: Path | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    resolved = _artifact_path(value, root)
    if resolved is None:
        return None
    return str(resolved) if resolved.exists() else None


def _existing_owned_path(
    record: Mapping[str, Any], key: str, suffix: str, root: Path | None
) -> str | None:
    resolved = _owned_artifact_path(record, key, suffix, root)
    return str(resolved) if resolved is not None and resolved.exists() else None


def _owned_artifact_path(
    record: Mapping[str, Any], key: str, suffix: str, root: Path | None
) -> Path | None:
    tag = record.get("tag")
    if root is None or not isinstance(tag, str) or not tag:
        return None
    expected = f"{tag}{suffix}"
    value = record.get(key)
    if isinstance(value, str) and value:
        if "\x00" in value:
            raise ReportError(f"invalid {key} artifact reference")
        supplied = Path(value).expanduser()
        if supplied.name != expected or (
            not supplied.is_absolute() and supplied != Path(expected)
        ):
            raise ReportError(f"{key} does not identify the task-owned artifact")
    return _artifact_path(expected, root)


def _artifact_path(value: str, root: Path | None) -> Path | None:
    if "\x00" in value or root is None:
        return None
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ReportError(f"artifact path escapes artifact_dir: {value!r}") from exc
    return resolved


def _participants(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "human": {
            "model": record.get("human_model"),
            "backend": record.get("human_backend"),
            "policy": record.get("human_policy"),
        },
        "assistant": {
            "model": record.get("assistant_model"),
            "backend": record.get("assistant_backend"),
            "control": record.get("assistant_control") or record.get("arm"),
        },
    }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_reason(result: Mapping[str, Any] | None) -> str | None:
    if result is None:
        return "release trace adjudication has not been supplied"
    reasons = result.get("reasons")
    return str(reasons[0]) if isinstance(reasons, list) and reasons else None


def _validate_score_binding(
    record: Mapping[str, Any], result: Mapping[str, Any], artifact_dir: Path | None
) -> None:
    binding = result.get("_evidence_binding")
    if not isinstance(binding, Mapping):
        raise ReportError("scored result has no evidence binding to the current run")
    _validate_evidence_binding(record, binding, artifact_dir)


def _validate_evidence_binding(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    artifact_dir: Path | None,
) -> None:
    try:
        verify_record_run_evidence(record, artifact_dir)
    except (ArtifactBundleError, OSError) as exc:
        raise ReportError(
            f"current runner artifacts do not match their run binding: {exc}"
        ) from exc
    record_run_hash = record.get("run_evidence_sha256")
    evidence_run_hash = evidence.get("run_evidence_sha256")
    if not isinstance(record_run_hash, str) or evidence_run_hash != record_run_hash:
        raise ReportError(
            "adjudication run-evidence hash does not match the current run"
        )
    record_hash = record.get("taskpack_file_sha256") or record.get("taskpack_sha256")
    evidence_hash = evidence.get("taskpack_sha256")
    if not isinstance(record_hash, str) or evidence_hash != record_hash:
        raise ReportError("adjudication TaskPack hash does not match the current run")
    _compare_bound_mapping(
        "native_taskpack",
        record.get("native_taskpack_outcome"),
        evidence.get("native_taskpack"),
    )
    _compare_bound_mapping(
        "validity",
        record.get("validity_evidence"),
        evidence.get("validity"),
    )


def _compare_bound_mapping(label: str, current: Any, bound: Any) -> None:
    if not isinstance(current, Mapping) or not isinstance(bound, Mapping):
        raise ReportError(f"adjudication {label} is not bound to the current run")
    mismatched = sorted(
        name
        for name, value in bound.items()
        if name in current and current.get(name) != value
    )
    missing = sorted(name for name in bound if name not in current)
    if mismatched or missing:
        details = ", ".join(missing + mismatched)
        raise ReportError(f"adjudication {label} differs from current run: {details}")


def _console_text(value: str) -> str:
    return "".join(
        character
        for character in value
        if character in {"\n", "\t"} or character.isprintable()
    )


def _suite_status(
    reports: Sequence[Mapping[str, Any]], release_scored: Sequence[Mapping[str, Any]]
) -> str:
    if not reports:
        return "EMPTY"
    if any(report.get("run_status") != "COMPLETED" for report in reports):
        return "ERROR"
    if any(
        isinstance(report.get("video"), Mapping)
        and report["video"].get("status") in {"ERROR", "UNAVAILABLE"}
        for report in reports
    ):
        return "ERROR"
    return "SCORED" if len(release_scored) == len(reports) else "PENDING"


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else None


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else None


def _is_score(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _verdict(value: Any, status: str) -> str:
    if type(value) is bool:
        return "PASS" if value else "FAIL"
    return status or "UNSCORABLE"


def _format_score(value: Any) -> str:
    return f"{float(value):.3f}" if type(value) in (int, float) else "n/a"

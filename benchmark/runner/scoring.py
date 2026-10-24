# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Optional application of human-adjudicated release evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.evaluation.artifact_bundle import (
    ArtifactBundleError,
    verify_record_run_evidence,
)
from benchmark.evaluation.prepare import prepare_evidence
from benchmark.evaluation.rubric import load_rubric
from benchmark.evaluation.score import EvidenceError, score_evidence

from .orchestrator import RunError


def _evidence_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("results"), list):
        value = value["results"]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if all(isinstance(row, Mapping) for row in value):
            return list(value)
    raise RunError("adjudication must be one evidence object or an array of objects")


def score_adjudication(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payloads = _load_payloads(path)
    rubric = load_rubric()
    results: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        _score_rows(_evidence_rows(payload), rubric, results)
    return results


def _load_payloads(path: Path) -> list[Any]:
    if path.is_dir():
        ignored = {"aggregate.json", "index.json", "report.json", "summary.json"}
        paths = [
            child
            for child in sorted(path.glob("*.json"))
            if child.name not in ignored and not child.name.startswith("_")
        ]
        if not paths:
            raise RunError(f"adjudication directory contains no evidence JSON: {path}")
        return [_read_payload(child, allow_array=False) for child in paths]
    return [_read_payload(path, allow_array=True)]


def _read_payload(path: Path, *, allow_array: bool) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"cannot read adjudication {path}: {exc}") from exc
    if not allow_array and not isinstance(payload, Mapping):
        raise RunError(f"adjudication file must contain one evidence object: {path}")
    return payload


def _score_rows(
    rows: list[Mapping[str, Any]],
    rubric: Mapping[str, Any],
    results: dict[str, dict[str, Any]],
) -> None:
    for row in rows:
        try:
            scored = score_evidence(row, rubric)
        except EvidenceError as exc:
            raise RunError(f"invalid adjudication evidence: {exc}") from exc
        task_id = str(scored.get("task_id") or "")
        if task_id in results:
            raise RunError(f"duplicate adjudication evidence for {task_id}")
        scored["_evidence_binding"] = {
            "taskpack_sha256": row.get("taskpack_sha256"),
            "run_evidence_sha256": row.get("run_evidence_sha256"),
            "native_taskpack": row.get("native_taskpack"),
            "validity": row.get("validity"),
        }
        results[task_id] = scored


def apply_scored_results(
    records: list[dict[str, Any]],
    scored_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for record in records:
        task_id = str(record.get("task_id") or "")
        scored = scored_results.get(task_id)
        if scored is None:
            merged.append(record)
            continue
        _validate_binding(record, scored)
        updated = dict(record)
        updated["release_rubric_outcome"] = dict(scored.get("release") or {})
        updated["release_score_status"] = scored.get("status")
        merged.append(updated)
    return merged


def _validate_binding(record: Mapping[str, Any], scored: Mapping[str, Any]) -> None:
    binding = scored.get("_evidence_binding")
    if not isinstance(binding, Mapping):
        raise RunError("scored adjudication is missing its run binding")
    try:
        current = prepare_evidence(record)
    except (OSError, ValueError) as exc:
        raise RunError(
            f"cannot bind adjudication to current runner record: {exc}"
        ) from exc
    task_id = str(record.get("task_id") or "")
    for name in (
        "taskpack_sha256",
        "run_evidence_sha256",
        "native_taskpack",
        "validity",
    ):
        if name not in binding or binding[name] is None:
            raise RunError(f"adjudication {name} is missing for {task_id}")
        if binding[name] != current.get(name):
            raise RunError(
                f"adjudication {name} does not match current run for {task_id}"
            )
    try:
        verify_record_run_evidence(record)
    except (ArtifactBundleError, OSError) as exc:
        raise RunError(
            f"current runner artifacts do not match their run binding: {exc}"
        ) from exc

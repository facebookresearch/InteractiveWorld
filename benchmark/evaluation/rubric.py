# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

RUBRIC_SCHEMA_VERSION = "interactiveworld-balanced120-rubric/1.0"
RUBRIC_PATH = Path(__file__).with_name("rubric.json")

PROTOCOL_BY_SOURCE = {
    "GUIDE": "StateScore-v2",
    "ORCHESTRATE": "StateBundle-v1",
    "NARRATE": "CFSS-StateScore",
    "ASK": "ASK-PCS",
    "WATCH": "WATCH-binary",
    "FIND": "FIND-C-pass-binary",
}
EXPECTED_CATEGORIES = {
    "Guidance": 20,
    "Planning": 20,
    "Accessibility": 20,
    "Decision": 20,
    "Prevention": 20,
    "Finding": 20,
}
EXPECTED_SOURCES = {
    "GUIDE": 26,
    "ORCHESTRATE": 16,
    "NARRATE": 20,
    "ASK": 19,
    "WATCH": 22,
    "FIND": 17,
}
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_CRITERION_ID_RE = re.compile(r"[a-z][a-z0-9_]*")


class RubricError(ValueError):
    pass


def load_rubric(path: str | Path | None = None) -> dict[str, Any]:
    rubric_path = Path(path) if path is not None else RUBRIC_PATH
    try:
        data = json.loads(rubric_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RubricError(f"cannot load rubric {rubric_path}: {exc}") from exc
    validate_rubric(data)
    return data


def task_index(rubric: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {task["task_id"]: task for task in rubric["tasks"]}


def validate_task_manifest(  # noqa: C901
    rubric: Mapping[str, Any],
    path: str | Path | None = None,
    *,
    verify_files: bool = True,
) -> None:
    manifest_path = (
        Path(path)
        if path is not None
        else RUBRIC_PATH.parent.parent / "tasks" / "manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RubricError(f"cannot load task manifest {manifest_path}: {exc}") from exc
    rows = manifest.get("tasks") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 120:
        raise RubricError("task manifest must contain exactly 120 tasks")
    rubric_tasks = task_index(rubric)
    manifest_tasks: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), str):
            raise RubricError("task manifest row is missing task_id")
        task_id = row["task_id"]
        if task_id in manifest_tasks:
            raise RubricError(f"duplicate task manifest row: {task_id}")
        manifest_tasks[task_id] = row
    if set(manifest_tasks) != set(rubric_tasks):
        raise RubricError("task manifest IDs differ from rubric IDs")
    for task_id, task in rubric_tasks.items():
        row = manifest_tasks[task_id]
        expected_fields = {
            "source_family": task["source_family"],
            "assigned_family": task["assigned_category"],
            "completion_protocol": task["protocol"],
            "performance_rubric_version": task["rubric_version"],
            "taskpack_sha256": task["expected_taskpack_sha256"],
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise RubricError(f"{task_id}: manifest {field} does not match rubric")
        if verify_files:
            relative = row.get("taskpack_path")
            if not isinstance(relative, str) or not relative:
                raise RubricError(f"{task_id}: taskpack_path is missing")
            taskpack_path = manifest_path.parent / relative
            if not taskpack_path.is_file():
                raise RubricError(f"{task_id}: TaskPack is missing at {taskpack_path}")
            actual = _file_sha256(taskpack_path)
            if actual != row["taskpack_sha256"]:
                raise RubricError(f"{task_id}: TaskPack file hash mismatch")

    openings_entry = manifest.get("interaction_openings")
    if not isinstance(openings_entry, Mapping):
        raise RubricError("task manifest is missing interaction_openings")
    openings_path_value = openings_entry.get("path")
    openings_digest = openings_entry.get("sha256")
    if not isinstance(openings_path_value, str) or not openings_path_value:
        raise RubricError("interaction_openings.path is missing")
    if (
        not isinstance(openings_digest, str)
        or _HASH_RE.fullmatch(openings_digest) is None
    ):
        raise RubricError("interaction_openings.sha256 is invalid")
    openings_path = manifest_path.parent / openings_path_value
    if not openings_path.is_file():
        raise RubricError(f"interaction opening catalog is missing at {openings_path}")
    if verify_files and _file_sha256(openings_path) != openings_digest:
        raise RubricError("interaction opening catalog hash mismatch")
    try:
        openings = json.loads(openings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RubricError(f"cannot load interaction opening catalog: {exc}") from exc
    catalog_tasks = openings.get("tasks") if isinstance(openings, Mapping) else None
    if not isinstance(catalog_tasks, Mapping):
        raise RubricError("interaction opening catalog must contain a tasks object")
    watch_ids = {
        task_id
        for task_id, task in rubric_tasks.items()
        if task["source_family"] == "WATCH"
    }
    if set(catalog_tasks) != watch_ids:
        raise RubricError(
            "interaction opening catalog must cover exactly the 22 WATCH tasks"
        )
    if openings.get("version") != openings_entry.get("schema_version"):
        raise RubricError("interaction opening catalog schema version mismatch")
    if openings_entry.get("task_count") != len(watch_ids):
        raise RubricError("interaction opening catalog task_count mismatch")
    if openings_entry.get("source_family") != "WATCH":
        raise RubricError("interaction opening catalog source_family must be WATCH")
    runtime_identity = rubric["provenance"]["runtime_identity"]
    if openings_digest != runtime_identity["interaction_openings_sha256"]:
        raise RubricError("interaction opening catalog differs from rubric provenance")
    _validate_asset_bundle(
        manifest_path.parent,
        rows,
        manifest.get("assets"),
        runtime_identity["asset_manifest_sha256"],
        verify_files=verify_files,
    )


def _referenced_pngs(value: Any) -> set[PurePosixPath]:
    references: set[PurePosixPath] = set()
    if isinstance(value, Mapping):
        source = value.get("source")
        texture = value.get("texture")
        candidate = (
            source
            if value.get("renderer") == "document_scan"
            else texture
            if value.get("kind") == "water_film"
            else None
        )
        if isinstance(candidate, str) and candidate.lower().endswith(".png"):
            relative = PurePosixPath(candidate)
            if relative.is_absolute() or ".." in relative.parts:
                raise RubricError(f"unsafe TaskPack asset reference: {candidate}")
            references.add(relative)
        for child in value.values():
            references.update(_referenced_pngs(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_referenced_pngs(child))
    return references


def _validate_asset_bundle(  # noqa: C901
    tasks_root: Path,
    task_rows: list[Any],
    entry: Any,
    expected_manifest_sha256: str,
    *,
    verify_files: bool,
) -> None:
    if not isinstance(entry, Mapping):
        raise RubricError("task manifest is missing assets")
    manifest_value = entry.get("manifest_path")
    manifest_digest = entry.get("manifest_sha256")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise RubricError("assets.manifest_path is missing")
    relative_manifest = PurePosixPath(manifest_value)
    if relative_manifest.is_absolute() or ".." in relative_manifest.parts:
        raise RubricError("assets.manifest_path is unsafe")
    if manifest_digest != expected_manifest_sha256:
        raise RubricError("asset manifest differs from rubric provenance")
    asset_manifest_path = tasks_root.joinpath(*relative_manifest.parts)
    if not asset_manifest_path.is_file():
        raise RubricError("bundled asset manifest is missing")
    if verify_files and _file_sha256(asset_manifest_path) != manifest_digest:
        raise RubricError("bundled asset manifest hash mismatch")
    try:
        asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RubricError(f"cannot load bundled asset manifest: {exc}") from exc
    if not isinstance(asset_manifest, Mapping):
        raise RubricError("bundled asset manifest must be an object")
    if asset_manifest.get("schema_version") != entry.get("schema_version"):
        raise RubricError("bundled asset manifest schema version mismatch")
    files = asset_manifest.get("files")
    if not isinstance(files, list) or len(files) != entry.get("file_count"):
        raise RubricError("bundled asset count mismatch")
    asset_root_value = asset_manifest.get("asset_root")
    if not isinstance(asset_root_value, str) or not asset_root_value:
        raise RubricError("bundled asset_root is missing")
    asset_root = asset_manifest_path.parent / asset_root_value
    declared: set[PurePosixPath] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise RubricError("bundled asset manifest row must be an object")
        relative = PurePosixPath(str(row.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RubricError(f"unsafe bundled asset path: {relative}")
        if relative in declared:
            raise RubricError(f"duplicate bundled asset path: {relative}")
        declared.add(relative)
        asset_path = asset_root.joinpath(*relative.parts)
        if not asset_path.is_file():
            raise RubricError(f"missing bundled asset: {relative}")
        if verify_files and _file_sha256(asset_path) != row.get("sha256"):
            raise RubricError(f"bundled asset hash mismatch: {relative}")
        if asset_path.stat().st_size != row.get("size_bytes"):
            raise RubricError(f"bundled asset size mismatch: {relative}")
    referenced: set[PurePosixPath] = set()
    for task in task_rows:
        taskpack_path = tasks_root / task["taskpack_path"]
        taskpack = json.loads(taskpack_path.read_text(encoding="utf-8"))
        referenced.update(_referenced_pngs(taskpack))
    if referenced != declared:
        raise RubricError("bundled assets do not exactly cover TaskPack PNG references")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def validate_rubric(rubric: Mapping[str, Any]) -> None:  # noqa: C901
    if rubric.get("schema_version") != RUBRIC_SCHEMA_VERSION:
        raise RubricError("unsupported rubric schema_version")
    tasks = rubric.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 120:
        raise RubricError("rubric must contain exactly 120 tasks")
    if rubric.get("task_count") != 120:
        raise RubricError("task_count must be 120")
    if set(rubric.get("protocols", {})) != set(PROTOCOL_BY_SOURCE.values()):
        raise RubricError("rubric must define exactly the six frozen protocols")
    if rubric.get("task_manifest_version") != "interactive-world-task-manifest/1.0":
        raise RubricError("unsupported task_manifest_version")
    _validate_provenance(rubric.get("provenance"))
    workflow = rubric.get("release_evidence_workflow")
    if not isinstance(workflow, Mapping) or set(workflow) != {
        "runner_scope",
        "prepare_scope",
        "trace_adjudication",
        "finalization",
    }:
        raise RubricError("release_evidence_workflow is missing or incomplete")

    ids: set[str] = set()
    categories: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    forbidden = {
        "observed_taskpack_hashes",
        "observed_taskpack_hash_by_model",
        "evaluation_status_by_model",
        "model_results",
    }
    for task in tasks:
        if not isinstance(task, Mapping):
            raise RubricError("each task must be an object")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RubricError("every task needs a task_id")
        if task_id in ids:
            raise RubricError(f"duplicate task_id: {task_id}")
        ids.add(task_id)
        source = task.get("source_family")
        if source not in PROTOCOL_BY_SOURCE:
            raise RubricError(f"{task_id}: unknown source_family {source!r}")
        if task.get("protocol") != PROTOCOL_BY_SOURCE[source]:
            raise RubricError(f"{task_id}: protocol does not preserve source family")
        if not task_id.startswith(source + "-"):
            raise RubricError(f"{task_id}: source_family does not match task ID")
        digest = task.get("expected_taskpack_sha256")
        if not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None:
            raise RubricError(f"{task_id}: invalid expected TaskPack hash")
        if not task.get("strict_success_criterion"):
            raise RubricError(f"{task_id}: missing strict success criterion")
        if forbidden.intersection(task):
            raise RubricError(f"{task_id}: historical result fields are forbidden")
        score_spec = task.get("score_spec")
        _validate_score_spec(task_id, task["protocol"], score_spec)
        if task.get("criteria_count") != _criterion_count(score_spec):
            raise RubricError(f"{task_id}: criteria_count does not match score_spec")
        categories[task.get("assigned_category")] += 1
        sources[source] += 1

    if dict(categories) != EXPECTED_CATEGORIES:
        raise RubricError(f"assigned category counts differ: {dict(categories)}")
    if dict(sources) != EXPECTED_SOURCES:
        raise RubricError(f"source family counts differ: {dict(sources)}")
    if rubric.get("assigned_category_counts") != EXPECTED_CATEGORIES:
        raise RubricError("declared assigned_category_counts are not frozen 20x6")
    if rubric.get("source_family_counts") != EXPECTED_SOURCES:
        raise RubricError("declared source_family_counts are not frozen")


def _validate_provenance(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RubricError("provenance must be an object")
    if set(value) != {"selection", "runtime_identity", "task_criteria"}:
        raise RubricError("provenance has missing or unknown fields")
    selection = value.get("selection")
    criteria = value.get("task_criteria")
    runtime = value.get("runtime_identity")
    if not all(isinstance(item, Mapping) for item in (selection, criteria, runtime)):
        raise RubricError("selection, criteria, and runtime provenance are required")
    if set(selection) != {"source", "used_fields", "excluded_fields"}:
        raise RubricError("selection provenance has missing or unknown fields")
    if set(criteria) != {"source", "rule"}:
        raise RubricError("task criteria provenance has missing or unknown fields")
    if set(runtime) != {
        "path",
        "rule",
        "interaction_openings_sha256",
        "asset_manifest_sha256",
    }:
        raise RubricError("runtime provenance has missing or unknown fields")
    if selection.get("used_fields") != [
        "task_id",
        "source_family",
        "assigned_family",
        "completion_protocol",
        "performance_rubric_version",
    ]:
        raise RubricError("selection provenance includes non-release fields")
    for label, text in (
        ("selection", selection.get("source")),
        ("task criteria", criteria.get("source")),
        ("runtime", runtime.get("path")),
    ):
        if not isinstance(text, str) or not text:
            raise RubricError(f"missing {label} provenance source")
    for label, digest in (
        ("interaction openings", runtime.get("interaction_openings_sha256")),
        ("asset manifest", runtime.get("asset_manifest_sha256")),
    ):
        if not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None:
            raise RubricError(f"invalid {label} provenance hash")


def _validate_score_spec(  # noqa: C901
    task_id: str, protocol: str, spec: Any
) -> None:
    if not isinstance(spec, Mapping):
        raise RubricError(f"{task_id}: score_spec must be an object")
    kind = spec.get("kind")
    expected_kind = {
        "StateScore-v2": "weighted_groups",
        "StateBundle-v1": "state_bundles",
        "CFSS-StateScore": "cfss_groups",
        "ASK-PCS": "terminal_atoms",
        "WATCH-binary": "state_conjunction",
        "FIND-C-pass-binary": "causal_checks",
    }[protocol]
    if kind != expected_kind:
        raise RubricError(f"{task_id}: expected score_spec kind {expected_kind}")
    if kind == "weighted_groups":
        groups = spec.get("groups")
        if not isinstance(groups, list) or not groups:
            raise RubricError(f"{task_id}: weighted groups cannot be empty")
        weights = [
            group.get("weight") for group in groups if isinstance(group, Mapping)
        ]
        if len(weights) != len(groups) or not all(
            isinstance(value, (int, float)) and value > 0 for value in weights
        ):
            raise RubricError(f"{task_id}: invalid group weight")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise RubricError(f"{task_id}: group weights must sum to one")
        _validate_named_criteria(task_id, groups, "groups")
        _validate_named_criteria(task_id, spec.get("gates"), "gates", allow_empty=True)
    elif kind == "state_bundles":
        _validate_named_criteria(task_id, spec.get("bundles"), "bundles")
        _validate_named_criteria(task_id, spec.get("gates"), "gates", allow_empty=True)
    elif kind == "cfss_groups":
        _validate_named_criteria(
            task_id, spec.get("terminal_groups"), "terminal_groups"
        )
        _validate_named_criteria(
            task_id,
            spec.get("transition_gates"),
            "transition_gates",
            allow_empty=True,
        )
    elif kind == "terminal_atoms":
        _validate_named_criteria(task_id, spec.get("atoms"), "atoms")
    elif kind == "state_conjunction":
        _validate_named_criteria(task_id, spec.get("conditions"), "conditions")
    elif kind == "causal_checks":
        _validate_named_criteria(task_id, spec.get("checks"), "checks")
    all_ids = [
        criterion["id"]
        for key in (
            "groups",
            "gates",
            "bundles",
            "terminal_groups",
            "transition_gates",
            "atoms",
            "conditions",
            "checks",
        )
        for criterion in spec.get(key, [])
    ]
    if len(all_ids) != len(set(all_ids)):
        raise RubricError(f"{task_id}: criterion IDs must be unique within the task")


def _criterion_count(spec: Mapping[str, Any]) -> int:
    return sum(
        len(spec.get(key, []))
        for key in (
            "groups",
            "gates",
            "bundles",
            "terminal_groups",
            "transition_gates",
            "atoms",
            "conditions",
            "checks",
        )
    )


def _validate_named_criteria(
    task_id: str,
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        cardinality = "possibly empty" if allow_empty else "non-empty"
        raise RubricError(f"{task_id}: {name} must be a {cardinality} list")
    ids: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RubricError(f"{task_id}: every {name} entry must be an object")
        criterion_id = item.get("id")
        description = item.get("description")
        if not isinstance(criterion_id, str) or not criterion_id:
            raise RubricError(f"{task_id}: every {name} entry needs an id")
        if _CRITERION_ID_RE.fullmatch(criterion_id) is None:
            raise RubricError(
                f"{task_id}: invalid stable criterion ID {criterion_id!r}"
            )
        if not isinstance(description, str) or not description:
            raise RubricError(f"{task_id}: every {name} entry needs a description")
        ids.append(criterion_id)
    if len(ids) != len(set(ids)):
        raise RubricError(f"{task_id}: duplicate IDs in {name}")

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Validate the frozen InteractiveWorld Balanced120 TaskPack release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_VERSION = "interactive-world-task-manifest/1.0"
RELEASE_ID = "InteractiveWorld-Balanced120"
TASKPACK_VERSION = "data-generate-taskpack/1.0"
RUNTIME_CONTRACT_VERSION = "compiled-runtime-contract/1.8"

ASSIGNED_COUNTS = {
    "Accessibility": 20,
    "Decision": 20,
    "Finding": 20,
    "Guidance": 20,
    "Planning": 20,
    "Prevention": 20,
}
SOURCE_COUNTS = {
    "ASK": 19,
    "FIND": 17,
    "GUIDE": 26,
    "NARRATE": 20,
    "ORCHESTRATE": 16,
    "WATCH": 22,
}
EXPECTED_FORMAT_COUNTS = {
    "taskpack_versions": {TASKPACK_VERSION: 120},
    "runtime_contract_versions": {RUNTIME_CONTRACT_VERSION: 120},
}
TASK_ID_RE = re.compile(r"^(ASK|FIND|GUIDE|NARRATE|ORCHESTRATE|WATCH)-[0-9]+$")
SHA256_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
COMPLETION_PROTOCOLS = {
    "ASK-PCS",
    "CFSS-StateScore",
    "FIND-C-pass-binary",
    "StateBundle-v1",
    "StateScore-v2",
    "WATCH-binary",
}
TOP_LEVEL_KEYS = {
    "manifest_version",
    "release_id",
    "schema_file",
    "taskpack_root",
    "task_count",
    "assigned_family_counts",
    "source_family_counts",
    "format_counts",
    "hash_policy",
    "interaction_openings",
    "assets",
    "tasks",
}
TASK_KEYS = {
    "task_id",
    "assigned_family",
    "source_family",
    "taskpack_path",
    "taskpack_sha256",
    "taskpack_version",
    "runtime_contract_version",
    "completion_protocol",
    "performance_rubric_version",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read valid JSON from {path}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _referenced_pngs(value: Any) -> set[PurePosixPath]:
    references: set[PurePosixPath] = set()
    if isinstance(value, dict):
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
            path = PurePosixPath(candidate)
            require(
                not path.is_absolute() and ".." not in path.parts,
                f"unsafe PNG reference: {candidate}",
            )
            references.add(path)
        for child in value.values():
            references.update(_referenced_pngs(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_referenced_pngs(child))
    return references


def _validate_asset_bundle(
    root: Path,
    tasks: list[dict[str, Any]],
    entry: Any,
) -> None:
    expected = {
        "manifest_path": "assets/manifest.json",
        "manifest_sha256": "sha256:1eb9cf32bd6b08fa908c18a6bd41a84989f2834a8f30a9a63643614099f35d04",
        "schema_version": "interactiveworld-task-assets/1.0",
        "file_count": 85,
    }
    require(entry == expected, "invalid bundled asset manifest identity")
    asset_manifest_path = root / entry["manifest_path"]
    require(asset_manifest_path.is_file(), "bundled asset manifest is missing")
    require(not asset_manifest_path.is_symlink(), "bundled asset manifest is a symlink")
    require(
        sha256(asset_manifest_path) == entry["manifest_sha256"],
        "asset manifest SHA256 mismatch",
    )
    asset_manifest = load_json(asset_manifest_path)
    require(
        isinstance(asset_manifest, dict)
        and asset_manifest.get("schema_version") == entry["schema_version"],
        "bundled asset manifest schema mismatch",
    )
    files = asset_manifest.get("files")
    require(
        isinstance(files, list) and len(files) == entry["file_count"],
        "asset count mismatch",
    )
    asset_root = asset_manifest_path.parent / str(
        asset_manifest.get("asset_root") or ""
    )
    declared: set[PurePosixPath] = set()
    for row in files:
        require(isinstance(row, dict), "asset manifest rows must be objects")
        relative = PurePosixPath(str(row.get("path") or ""))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe asset path: {relative}",
        )
        path = asset_root.joinpath(*relative.parts)
        require(
            path.is_file() and not path.is_symlink(),
            f"missing bundled asset: {relative}",
        )
        require(
            sha256(path) == row.get("sha256"),
            f"bundled asset SHA256 mismatch: {relative}",
        )
        require(
            path.stat().st_size == row.get("size_bytes"),
            f"bundled asset size mismatch: {relative}",
        )
        declared.add(relative)
    require(len(declared) == len(files), "duplicate bundled asset path")

    referenced: set[PurePosixPath] = set()
    for task in tasks:
        referenced.update(_referenced_pngs(load_json(root / task["taskpack_path"])))
    require(
        referenced == declared,
        "bundled assets must exactly match TaskPack PNG references",
    )
    actual = {
        path.relative_to(asset_root) for path in asset_root.rglob("*") if path.is_file()
    }
    require(
        actual == declared, "bundled asset directory contains missing or extra files"
    )


def validate_task(root: Path, task: Any) -> tuple[str, str, str, str, str, Path]:
    require(isinstance(task, dict), "each tasks[] entry must be an object")
    require(
        set(task) == TASK_KEYS, f"{task.get('task_id', '<unknown>')}: invalid fields"
    )

    task_id = task["task_id"]
    task_id_match = TASK_ID_RE.fullmatch(task_id)
    require(task_id_match is not None, f"invalid task_id: {task_id!r}")
    source_family = task["source_family"]
    require(
        source_family == task_id_match.group(1), f"{task_id}: source family mismatch"
    )
    require(source_family in SOURCE_COUNTS, f"{task_id}: unknown source family")

    assigned_family = task["assigned_family"]
    require(assigned_family in ASSIGNED_COUNTS, f"{task_id}: unknown assigned family")
    relative_path = PurePosixPath(task["taskpack_path"])
    expected_path = PurePosixPath(
        "taskpacks", assigned_family.lower(), f"{task_id}.taskpack.json"
    )
    require(relative_path == expected_path, f"{task_id}: non-canonical taskpack_path")
    require(not relative_path.is_absolute(), f"{task_id}: absolute taskpack_path")
    require(".." not in relative_path.parts, f"{task_id}: escaping taskpack_path")

    taskpack_path = root.joinpath(*relative_path.parts)
    require(taskpack_path.is_file(), f"{task_id}: missing {taskpack_path}")
    require(
        not taskpack_path.is_symlink(), f"{task_id}: TaskPack must not be a symlink"
    )

    runtime_hash = task["taskpack_sha256"]
    hash_match = SHA256_RE.fullmatch(runtime_hash)
    require(hash_match is not None, f"{task_id}: invalid taskpack_sha256")
    require(
        sha256(taskpack_path) == runtime_hash, f"{task_id}: TaskPack SHA256 mismatch"
    )

    require(
        task["taskpack_version"] == TASKPACK_VERSION,
        f"{task_id}: manifest TaskPack version",
    )
    require(
        task["runtime_contract_version"] == RUNTIME_CONTRACT_VERSION,
        f"{task_id}: manifest runtime contract version",
    )
    taskpack = load_json(taskpack_path)
    require(isinstance(taskpack, dict), f"{task_id}: TaskPack root must be an object")
    require(taskpack.get("task_id") == task_id, f"{task_id}: TaskPack task_id mismatch")
    require(
        taskpack.get("taskpack_version") == TASKPACK_VERSION,
        f"{task_id}: TaskPack schema version mismatch",
    )
    runtime = taskpack.get("runtime")
    require(isinstance(runtime, dict), f"{task_id}: missing runtime object")
    require(
        runtime.get("contract_version") == RUNTIME_CONTRACT_VERSION,
        f"{task_id}: runtime contract version mismatch",
    )

    require(
        task["completion_protocol"] in COMPLETION_PROTOCOLS,
        f"{task_id}: unknown completion protocol",
    )
    require(
        isinstance(task["performance_rubric_version"], str)
        and bool(task["performance_rubric_version"]),
        f"{task_id}: missing rubric version",
    )
    return (
        task_id,
        assigned_family,
        source_family,
        task["taskpack_version"],
        task["runtime_contract_version"],
        taskpack_path,
    )


def validate(manifest_path: Path) -> None:
    root = manifest_path.parent.resolve()
    manifest = load_json(manifest_path)
    require(isinstance(manifest, dict), "manifest root must be an object")
    require(set(manifest) == TOP_LEVEL_KEYS, "manifest has missing or unknown fields")
    require(
        manifest["manifest_version"] == MANIFEST_VERSION, "manifest version mismatch"
    )
    require(manifest["release_id"] == RELEASE_ID, "release ID mismatch")
    require(manifest["schema_file"] == "manifest.schema.json", "schema path mismatch")
    require(manifest["taskpack_root"] == "taskpacks", "taskpack root mismatch")
    require(manifest["task_count"] == 120, "task_count must be 120")
    require(load_json(root / manifest["schema_file"]), "manifest schema is empty")

    hash_policy = manifest["hash_policy"]
    require(
        hash_policy
        == {
            "authoritative_field": "tasks[].taskpack_sha256",
            "taskpack_sha256_algorithm": "SHA-256",
        },
        "invalid hash policy",
    )

    openings_entry = manifest["interaction_openings"]
    require(
        openings_entry
        == {
            "path": "interaction_openings.json",
            "sha256": "sha256:a9c92de7bbc460fc15fcbcf7486dbc37ebda8ba156381d1c29a43f709c8dd485",
            "schema_version": "taskpack-interaction-openings/2.1",
            "task_count": 22,
            "source_family": "WATCH",
        },
        "invalid interaction opening catalog identity",
    )
    openings_path = root / openings_entry["path"]
    require(openings_path.is_file(), "interaction opening catalog is missing")
    require(not openings_path.is_symlink(), "interaction opening catalog is a symlink")
    require(
        sha256(openings_path) == openings_entry["sha256"],
        "opening catalog SHA256 mismatch",
    )
    openings = load_json(openings_path)
    require(
        isinstance(openings, dict)
        and openings.get("version") == openings_entry["schema_version"],
        "interaction opening catalog schema mismatch",
    )

    tasks = manifest["tasks"]
    require(
        isinstance(tasks, list) and len(tasks) == 120, "tasks must contain 120 entries"
    )
    rows = [validate_task(root, task) for task in tasks]

    task_ids = [row[0] for row in rows]
    task_paths = [row[5] for row in rows]
    task_hashes = [task["taskpack_sha256"] for task in tasks]
    require(len(set(task_ids)) == 120, "task IDs are not unique")
    require(len(set(task_paths)) == 120, "TaskPack paths are not unique")
    require(len(set(task_hashes)) == 120, "TaskPack hashes are not unique")
    family_order = {family: index for index, family in enumerate(ASSIGNED_COUNTS)}
    ordering_keys = [
        (family_order[task["assigned_family"]], task["task_id"]) for task in tasks
    ]
    require(
        ordering_keys == sorted(ordering_keys),
        "tasks are not grouped by assigned family and task_id",
    )

    assigned_counts = Counter(row[1] for row in rows)
    source_counts = Counter(row[2] for row in rows)
    version_counts = Counter(row[3] for row in rows)
    runtime_counts = Counter(row[4] for row in rows)
    require(
        dict(assigned_counts) == ASSIGNED_COUNTS,
        "assigned families must each contain 20 tasks",
    )
    require(
        dict(source_counts) == SOURCE_COUNTS,
        "source family counts do not match frozen set",
    )
    require(
        dict(version_counts) == {TASKPACK_VERSION: 120},
        "TaskPack version count mismatch",
    )
    require(
        dict(runtime_counts) == {RUNTIME_CONTRACT_VERSION: 120},
        "runtime contract version count mismatch",
    )
    require(
        manifest["assigned_family_counts"] == ASSIGNED_COUNTS,
        "assigned count metadata mismatch",
    )
    require(
        manifest["source_family_counts"] == SOURCE_COUNTS,
        "source count metadata mismatch",
    )
    require(
        manifest["format_counts"] == EXPECTED_FORMAT_COUNTS,
        "format count metadata mismatch",
    )
    watch_ids = {task["task_id"] for task in tasks if task["source_family"] == "WATCH"}
    opening_rows = openings.get("tasks") if isinstance(openings, dict) else None
    require(isinstance(opening_rows, dict), "opening catalog tasks must be an object")
    require(
        set(opening_rows) == watch_ids,
        "opening catalog must cover exactly the 22 WATCH tasks",
    )
    _validate_asset_bundle(root, tasks, manifest["assets"])

    expected_files = set(task_paths)
    actual_files: set[Path] = set()
    taskpack_root = root / "taskpacks"
    require(taskpack_root.is_dir(), "missing taskpacks directory")
    for family in ASSIGNED_COUNTS:
        family_dir = taskpack_root / family.lower()
        require(family_dir.is_dir(), f"missing category directory: {family_dir}")
        require(
            not family_dir.is_symlink(),
            f"category directory is a symlink: {family_dir}",
        )
        for entry in family_dir.iterdir():
            require(
                entry.is_file(), f"unexpected non-file in category directory: {entry}"
            )
            require(not entry.is_symlink(), f"unexpected symlink: {entry}")
            actual_files.add(entry.resolve())
    require(
        actual_files == {path.resolve() for path in expected_files},
        "extra or missing TaskPack file",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
    )
    args = parser.parse_args()
    try:
        validate(args.manifest.resolve())
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    print(
        "validated InteractiveWorld-Balanced120: "
        "120 unique TaskPacks, six categories x 20, canonical SHA256 and runtime schemas OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

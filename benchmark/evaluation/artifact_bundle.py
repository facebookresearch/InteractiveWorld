# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Content-address one task's persisted evidence without exposing its contents."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUN_EVIDENCE_BINDING_VERSION = "interactiveworld-run-evidence/1.0"
RUN_EVIDENCE_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REQUIRED_FILE_SUFFIXES = (
    "_dialogue.json",
    "_eval.json",
    "_manifest.json",
    "_taskpack_eval.json",
)
_OPTIONAL_FILE_SUFFIXES = ("_fpv.mp4", "_replay.json.gz", "_tpv.mp4")
_DIRECTORY_SUFFIXES = ("_frames", "_io", "_tpv_frames")
_RECORD_BOUND_FIELDS = (
    "task_id",
    "tag",
    "taskpack_file_sha256",
    "taskpack_digest",
    "builder_spec_digest",
    "seed",
    "arm",
    "assistant_control",
    "assistant_backend",
    "assistant_model",
    "assistant_output_protocol",
    "human_backend",
    "human_model",
    "human_policy",
    "human_object_interface",
    "find_human_control",
    "ask_human_control",
    "watch_human_turn_guard",
    "native_taskpack_outcome",
    "validity_evidence",
)


class ArtifactBundleError(ValueError):
    """The persisted task evidence cannot be bound safely."""


def compute_run_evidence_sha256(artifact_dir: str | Path, tag: str) -> str:
    """Return a relocation-safe digest of all persisted evidence for ``tag``.

    The digest is a content binding for accidental-mismatch detection, not an
    authenticity signature. Paths and evidence bytes are never returned.
    """
    validate_artifact_tag(tag)
    root = Path(artifact_dir).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactBundleError("task artifact directory is missing")
    paths = _bundle_paths(root, tag)
    _require_core_artifacts(root, tag, paths)
    digest = hashlib.sha256()
    digest.update(RUN_EVIDENCE_BINDING_VERSION.encode("ascii") + b"\x00")
    for path in paths:
        relative = path.relative_to(root).as_posix()
        content = _content_bytes(path)
        digest.update(relative.encode("utf-8") + b"\x00")
        digest.update(str(len(content)).encode("ascii") + b"\x00")
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def is_run_evidence_sha256(value: object) -> bool:
    """Return whether ``value`` is a normalized run-evidence digest."""
    return (
        isinstance(value, str)
        and RUN_EVIDENCE_SHA256_PATTERN.fullmatch(value) is not None
    )


def verify_run_evidence_sha256(
    artifact_dir: str | Path, tag: str, expected: object
) -> None:
    """Verify artifact bytes and the two embedded binding copies."""
    if not is_run_evidence_sha256(expected):
        raise ArtifactBundleError("invalid expected run-evidence digest")
    actual = compute_run_evidence_sha256(artifact_dir, tag)
    if actual != expected:
        raise ArtifactBundleError("run-evidence digest does not match artifact bytes")
    root = Path(artifact_dir).expanduser().resolve()
    for suffix in ("_eval.json", "_taskpack_eval.json"):
        path = root / f"{tag}{suffix}"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactBundleError(
                f"cannot validate embedded binding in {path.name}"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("run_evidence_sha256") != expected
        ):
            raise ArtifactBundleError(
                f"embedded run-evidence digest is missing or mismatched in {path.name}"
            )


def verify_record_run_evidence(
    record: Mapping[str, Any], artifact_dir: str | Path | None = None
) -> None:
    """Verify a runner record against the persisted bundle it identifies."""
    root = artifact_dir if artifact_dir is not None else record.get("artifact_dir")
    tag = record.get("tag")
    expected = record.get("run_evidence_sha256")
    if not isinstance(root, (str, Path)) or not isinstance(tag, str):
        raise ArtifactBundleError(
            "runner record is missing artifact_dir or task artifact tag"
        )
    verify_run_evidence_sha256(root, tag, expected)
    path = Path(root).expanduser().resolve() / f"{tag}_taskpack_eval.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactBundleError(
            "cannot validate runner record against taskpack evaluation"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ArtifactBundleError("taskpack evaluation must be a JSON object")
    for name in _RECORD_BOUND_FIELDS:
        if name not in record or name not in payload or record[name] != payload[name]:
            raise ArtifactBundleError(
                f"runner record {name} does not match taskpack evaluation"
            )
    task_goal = payload.get("task_goal")
    if (
        not isinstance(task_goal, Mapping)
        or "progress" not in record
        or record["progress"] != task_goal.get("progress")
    ):
        raise ArtifactBundleError(
            "runner record progress does not match taskpack evaluation"
        )


def validate_artifact_tag(tag: str) -> None:
    """Reject task tags that could address anything outside their output root."""
    if not _SAFE_TAG.fullmatch(tag) or tag in {".", ".."}:
        raise ArtifactBundleError(f"unsafe task artifact tag: {tag!r}")


def reset_task_artifacts(artifact_dir: str | Path, tag: str) -> None:
    """Remove only the named task's owned artifacts before a retry."""
    validate_artifact_tag(tag)
    root = Path(artifact_dir).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactBundleError("task artifact directory is missing")
    for path in _owned_roots(root, tag):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            raise ArtifactBundleError(
                f"unsupported existing task artifact: {path.name}"
            )


def _bundle_paths(root: Path, tag: str) -> list[Path]:
    paths: list[Path] = []
    for child in _owned_roots(root, tag):
        if not child.exists() and not child.is_symlink():
            continue
        if child.is_symlink():
            raise ArtifactBundleError(f"task evidence contains a symlink: {child.name}")
        if child.is_file():
            paths.append(child)
            continue
        if not child.is_dir():
            continue
        for path in sorted(child.rglob("*"), key=lambda value: value.as_posix()):
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                raise ArtifactBundleError(
                    f"task evidence contains a symlink: {relative}"
                )
            if path.is_file():
                paths.append(path)
    return sorted(paths, key=lambda value: value.relative_to(root).as_posix())


def _owned_roots(root: Path, tag: str) -> list[Path]:
    suffixes = (
        *_REQUIRED_FILE_SUFFIXES,
        *_OPTIONAL_FILE_SUFFIXES,
        *_DIRECTORY_SUFFIXES,
    )
    return [root / f"{tag}{suffix}" for suffix in suffixes]


def _require_core_artifacts(root: Path, tag: str, paths: list[Path]) -> None:
    available = {path.name for path in paths if path.parent == root}
    missing = [
        f"{tag}{suffix}"
        for suffix in _REQUIRED_FILE_SUFFIXES
        if f"{tag}{suffix}" not in available
    ]
    if missing:
        raise ArtifactBundleError(
            "task evidence is missing required artifact(s): " + ", ".join(missing)
        )


def _content_bytes(path: Path) -> bytes:
    if path.suffix != ".json":
        return path.read_bytes()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactBundleError(f"invalid JSON evidence file: {path.name}") from exc
    normalized = (
        _without_binding(value)
        if path.name.endswith(("_eval.json", "_taskpack_eval.json"))
        else value
    )
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactBundleError(
            f"non-canonical JSON evidence file: {path.name}"
        ) from exc
    return text.encode("utf-8")


def _without_binding(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        str(key): child for key, child in value.items() if key != "run_evidence_sha256"
    }

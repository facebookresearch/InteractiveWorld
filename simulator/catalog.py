# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Public discovery and verified loading for InteractiveWorld TaskPacks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simulator.errors import TaskCatalogError, TaskNotFoundError, TaskValidationError
from simulator.interaction.task_runtime import builder_spec_from_taskpack, digest_json
from simulator.taskpack_support.runtime_adapter import canonical_digest


MANIFEST_VERSION = "interactive-world-task-manifest/1.0"
TASKPACK_VERSION = "data-generate-taskpack/1.0"
RUNTIME_CONTRACT_VERSION = "compiled-runtime-contract/1.8"
INTERACTION_OPENINGS_VERSION = "taskpack-interaction-openings/2.1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HUMAN_POLICY_VERSION = "human-policy/1.0"
_PREPARATION_MODES = frozenset({"prologue", "continuous_watch"})
_REQUIRED_INFO_FIELDS = (
    "task_id",
    "assigned_family",
    "source_family",
    "taskpack_path",
    "taskpack_sha256",
    "taskpack_version",
    "runtime_contract_version",
    "completion_protocol",
    "performance_rubric_version",
)


@dataclass(frozen=True)
class TaskInfo:
    """Immutable catalog metadata for one TaskPack."""

    task_id: str
    assigned_family: str
    source_family: str
    taskpack_path: Path
    taskpack_sha256: str
    taskpack_version: str
    runtime_contract_version: str
    completion_protocol: str
    performance_rubric_version: str

    @property
    def category(self) -> str:
        """Return the benchmark category under its user-facing name."""
        return self.assigned_family


@dataclass(frozen=True)
class TaskSpec:
    """An immutable, fully verified TaskPack and its runtime projection.

    ``source_sha256`` binds original file bytes when the document came from a
    file. ``taskpack_digest`` always binds canonical JSON and is the wire identity.
    """

    task_id: str
    taskpack_version: str
    runtime_contract_version: str
    source_sha256: str | None
    taskpack_digest: str
    builder_spec_digest: str
    scene: str
    source_path: Path | None
    info: TaskInfo | None = None
    preparation_mode: str = "prologue"
    _taskpack_json: str = field(repr=False, compare=False, default="{}")
    _builder_spec_json: str = field(repr=False, compare=False, default="{}")

    @property
    def taskpack(self) -> dict[str, Any]:
        """Return a detached mutable copy of the verified TaskPack document."""
        return _object_from_snapshot(self._taskpack_json, "TaskPack snapshot")

    @property
    def builder_spec(self) -> dict[str, Any]:
        """Return a detached mutable copy of the verified runtime builder view."""
        return _object_from_snapshot(self._builder_spec_json, "builder snapshot")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_task_id: str | None = None,
    ) -> TaskSpec:
        """Load an explicit TaskPack whose expected file digest is mandatory."""
        return _load_taskpack(
            Path(path),
            expected_sha256=expected_sha256,
            expected_task_id=expected_task_id,
            info=None,
        )

    @classmethod
    def from_mapping(
        cls,
        taskpack: Mapping[str, Any],
        *,
        expected_sha256: str,
        source_path: Path | None = None,
        expected_task_id: str | None = None,
    ) -> TaskSpec:
        """Load an inline TaskPack bound to its canonical JSON SHA256."""
        return load_taskpack(
            taskpack,
            expected_sha256=expected_sha256,
            source_path=source_path,
            expected_task_id=expected_task_id,
        )


class TaskCatalog:
    """A lightweight catalog backed by a versioned TaskPack manifest."""

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        is_bundled = manifest_path is None
        selected_path = (
            _bundled_manifest_path() if manifest_path is None else Path(manifest_path)
        )
        self._manifest_path = selected_path.resolve()
        manifest = _read_json_object(self._manifest_path, catalog=True)
        self._manifest_version = _required_text(
            manifest, "manifest_version", self._manifest_path, catalog=True
        )
        if self._manifest_version != MANIFEST_VERSION:
            raise TaskCatalogError(
                "unsupported task manifest version "
                f"{self._manifest_version!r}; expected {MANIFEST_VERSION!r}"
            )
        self._release_id = _required_text(
            manifest, "release_id", self._manifest_path, catalog=True
        )
        self._tasks = _parse_manifest_tasks(manifest, self._manifest_path)
        self._by_id = {task.task_id: task for task in self._tasks}
        self._preparation_modes = (
            _bundled_preparation_modes(self._manifest_path, self._tasks)
            if is_bundled
            else {}
        )

    @classmethod
    def bundled(cls) -> TaskCatalog:
        """Open the TaskPack manifest shipped with this release."""
        return cls()

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def manifest_version(self) -> str:
        return self._manifest_version

    @property
    def release_id(self) -> str:
        return self._release_id

    def __len__(self) -> int:
        return len(self._tasks)

    def list(
        self,
        *,
        category: str | None = None,
        source_family: str | None = None,
    ) -> tuple[TaskInfo, ...]:
        """List tasks, optionally filtering by category and source family."""
        category_key = _optional_filter(category, "category")
        source_key = _optional_filter(source_family, "source_family")
        return tuple(
            task
            for task in self._tasks
            if (category_key is None or task.assigned_family.casefold() == category_key)
            and (source_key is None or task.source_family.casefold() == source_key)
        )

    def get(self, task_id: str) -> TaskSpec:
        """Load one bundled task by ID and verify it against the manifest."""
        if not isinstance(task_id, str) or not task_id.strip():
            raise TaskNotFoundError("task_id must be a non-empty string")
        normalized = task_id.strip()
        info = self._by_id.get(normalized)
        if info is None:
            raise TaskNotFoundError(
                f"task {normalized!r} is not present in {self._release_id}"
            )
        return _load_taskpack(
            info.taskpack_path,
            expected_sha256=info.taskpack_sha256,
            expected_task_id=info.task_id,
            info=info,
            preparation_mode=self._preparation_modes.get(info.task_id),
        )

    @staticmethod
    def load(
        path: str | Path,
        *,
        expected_sha256: str,
        expected_task_id: str | None = None,
    ) -> TaskSpec:
        """Load an explicit TaskPack without requiring a catalog manifest."""
        return TaskSpec.load(
            path,
            expected_sha256=expected_sha256,
            expected_task_id=expected_task_id,
        )

    @staticmethod
    def load_taskpack(
        taskpack: Mapping[str, Any],
        *,
        expected_sha256: str,
        source_path: Path | None = None,
        expected_task_id: str | None = None,
    ) -> TaskSpec:
        """Load an inline TaskPack bound to its canonical JSON SHA256."""
        return load_taskpack(
            taskpack,
            expected_sha256=expected_sha256,
            source_path=source_path,
            expected_task_id=expected_task_id,
        )


def _bundled_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "benchmark" / "tasks" / "manifest.json"
    )


def _bundled_preparation_modes(
    manifest_path: Path,
    tasks: tuple[TaskInfo, ...],
) -> dict[str, str]:
    manifest_root = manifest_path.parent.resolve()
    openings_path = (manifest_root / "interaction_openings.json").resolve()
    if openings_path.parent != manifest_root:
        raise TaskCatalogError(
            f"interaction openings path escapes the task manifest directory: {openings_path}"
        )
    if not openings_path.is_file():
        raise TaskCatalogError(f"interaction openings file is missing: {openings_path}")
    document = _read_json_object(openings_path, catalog=True)
    version = _required_text(document, "version", openings_path, catalog=True)
    if version != INTERACTION_OPENINGS_VERSION:
        raise TaskCatalogError(
            f"unsupported interaction openings version {version!r}; "
            f"expected {INTERACTION_OPENINGS_VERSION!r}"
        )
    rows = document.get("tasks")
    if not isinstance(rows, Mapping):
        raise TaskCatalogError(f"{openings_path}: tasks must be an object")
    known_task_ids = {task.task_id for task in tasks}
    modes = dict.fromkeys(known_task_ids, "prologue")
    for task_id, value in rows.items():
        if not task_id.strip() or task_id not in known_task_ids:
            raise TaskCatalogError(
                f"{openings_path}: unknown interaction task_id {task_id!r}"
            )
        if not isinstance(value, Mapping):
            raise TaskCatalogError(
                f"{openings_path}: interaction task {task_id!r} must be an object"
            )
        continuous_watch = value.get("continuous_watch")
        if type(continuous_watch) is not bool:
            raise TaskCatalogError(
                f"{openings_path}: {task_id}.continuous_watch must be a boolean"
            )
        if continuous_watch:
            modes[task_id] = "continuous_watch"
    return modes


def _parse_manifest_tasks(
    manifest: Mapping[str, Any], manifest_path: Path
) -> tuple[TaskInfo, ...]:
    rows = manifest.get("tasks")
    if not isinstance(rows, list):
        raise TaskCatalogError(f"{manifest_path}: tasks must be an array")
    declared_count = manifest.get("task_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise TaskCatalogError(f"{manifest_path}: task_count must be an integer")
    if declared_count != len(rows):
        raise TaskCatalogError(
            f"{manifest_path}: task_count is {declared_count}, found {len(rows)} rows"
        )
    taskpack_root = _taskpack_root(manifest, manifest_path)
    tasks = tuple(
        _parse_manifest_row(row, manifest_path, taskpack_root) for row in rows
    )
    _require_unique(tasks, manifest_path)
    return tasks


def _taskpack_root(manifest: Mapping[str, Any], manifest_path: Path) -> Path:
    relative = _required_text(manifest, "taskpack_root", manifest_path, catalog=True)
    raw_path = Path(relative)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise TaskCatalogError(f"{manifest_path}: unsafe taskpack_root {relative!r}")
    root = (manifest_path.parent / raw_path).resolve()
    if not root.is_dir():
        raise TaskCatalogError(f"{manifest_path}: taskpack_root is missing: {root}")
    return root


def _parse_manifest_row(
    value: Any, manifest_path: Path, taskpack_root: Path
) -> TaskInfo:
    if not isinstance(value, dict):
        raise TaskCatalogError(f"{manifest_path}: every tasks entry must be an object")
    fields = {
        name: _required_text(value, name, manifest_path, catalog=True)
        for name in _REQUIRED_INFO_FIELDS
    }
    if fields["taskpack_version"] != TASKPACK_VERSION:
        raise TaskCatalogError(
            f"{fields['task_id']}: unsupported taskpack_version "
            f"{fields['taskpack_version']!r}"
        )
    if fields["runtime_contract_version"] != RUNTIME_CONTRACT_VERSION:
        raise TaskCatalogError(
            f"{fields['task_id']}: unsupported runtime_contract_version "
            f"{fields['runtime_contract_version']!r}"
        )
    _validate_sha256(fields["taskpack_sha256"], "manifest taskpack_sha256")
    taskpack_path = _safe_taskpack_path(
        manifest_path.parent, taskpack_root, fields["taskpack_path"], fields["task_id"]
    )
    return TaskInfo(
        task_id=fields["task_id"],
        assigned_family=fields["assigned_family"],
        source_family=fields["source_family"],
        taskpack_path=taskpack_path,
        taskpack_sha256=fields["taskpack_sha256"],
        taskpack_version=fields["taskpack_version"],
        runtime_contract_version=fields["runtime_contract_version"],
        completion_protocol=fields["completion_protocol"],
        performance_rubric_version=fields["performance_rubric_version"],
    )


def _safe_taskpack_path(
    manifest_root: Path, taskpack_root: Path, relative: str, task_id: str
) -> Path:
    raw_path = Path(relative)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise TaskCatalogError(f"{task_id}: unsafe taskpack path {relative!r}")
    path = (manifest_root / raw_path).resolve()
    if not path.is_relative_to(taskpack_root):
        raise TaskCatalogError(f"{task_id}: taskpack path escapes taskpack_root")
    if not path.is_file():
        raise TaskCatalogError(f"{task_id}: TaskPack is missing: {path}")
    return path


def _require_unique(tasks: tuple[TaskInfo, ...], manifest_path: Path) -> None:
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise TaskCatalogError(f"{manifest_path}: duplicate task_id")
    taskpack_paths = [task.taskpack_path for task in tasks]
    if len(taskpack_paths) != len(set(taskpack_paths)):
        raise TaskCatalogError(f"{manifest_path}: duplicate taskpack_path")


def _load_taskpack(
    path: Path,
    *,
    expected_sha256: str,
    expected_task_id: str | None,
    info: TaskInfo | None,
    preparation_mode: str | None = None,
) -> TaskSpec:
    _validate_sha256(expected_sha256, "expected_sha256")
    source_path = path.resolve()
    actual_sha256 = _file_sha256(source_path)
    if actual_sha256 != expected_sha256:
        raise TaskValidationError(
            f"TaskPack SHA256 mismatch for {source_path}: expected "
            f"{expected_sha256}, got {actual_sha256}"
        )
    taskpack = _read_json_object(source_path, catalog=False)
    return _build_task_spec(
        taskpack,
        source_sha256=actual_sha256,
        source_path=source_path,
        expected_task_id=expected_task_id,
        info=info,
        preparation_mode=preparation_mode,
    )


def load_taskpack(
    taskpack: Mapping[str, Any],
    *,
    expected_sha256: str,
    source_path: Path | None = None,
    expected_task_id: str | None = None,
) -> TaskSpec:
    """Verify an inline TaskPack using the SHA256 of its canonical JSON form."""
    _validate_sha256(expected_sha256, "expected_sha256")
    document = dict(taskpack)
    try:
        taskpack_digest = digest_json(document)
        snapshot = _snapshot(document)
    except (TypeError, ValueError) as error:
        raise TaskValidationError(f"TaskPack is not canonical JSON: {error}") from error
    if taskpack_digest != expected_sha256:
        raise TaskValidationError(
            "TaskPack canonical SHA256 mismatch: "
            f"expected {expected_sha256}, got {taskpack_digest}"
        )
    normalized = _object_from_snapshot(snapshot, "TaskPack snapshot")
    return _build_task_spec(
        normalized,
        source_sha256=None,
        source_path=source_path.resolve() if source_path is not None else None,
        expected_task_id=expected_task_id,
        info=None,
    )


def _build_task_spec(
    taskpack: dict[str, Any],
    *,
    source_sha256: str | None,
    source_path: Path | None,
    expected_task_id: str | None,
    info: TaskInfo | None,
    preparation_mode: str | None = None,
) -> TaskSpec:
    source_label = source_path or Path("<inline TaskPack>")
    task_id = _required_text(taskpack, "task_id", source_label, catalog=False)
    if expected_task_id is not None and task_id != expected_task_id:
        raise TaskValidationError(
            f"TaskPack task_id mismatch: expected {expected_task_id!r}, got {task_id!r}"
        )
    taskpack_version = _required_text(
        taskpack, "taskpack_version", source_label, catalog=False
    )
    if taskpack_version != TASKPACK_VERSION:
        raise TaskValidationError(
            f"{task_id}: unsupported taskpack_version {taskpack_version!r}; "
            f"expected {TASKPACK_VERSION!r}"
        )
    runtime = _required_mapping(taskpack, "runtime", source_label)
    runtime_version = _required_text(
        runtime, "contract_version", source_label, catalog=False
    )
    if runtime_version != RUNTIME_CONTRACT_VERSION:
        raise TaskValidationError(
            f"{task_id}: unsupported runtime contract {runtime_version!r}; "
            f"expected {RUNTIME_CONTRACT_VERSION!r}"
        )
    declared_preparation_mode = _validate_public_task_fields(taskpack, runtime, task_id)
    if (
        preparation_mode is not None
        and "public_preparation" in runtime
        and declared_preparation_mode != preparation_mode
    ):
        raise TaskValidationError(
            f"{task_id}: runtime.public_preparation.mode "
            f"{declared_preparation_mode!r} conflicts with bundled interaction "
            f"metadata {preparation_mode!r}"
        )
    resolved_preparation_mode = preparation_mode or declared_preparation_mode
    builder_spec, builder_digest = _validated_builder(taskpack, source_label, task_id)
    scene = _required_text(builder_spec, "scene", source_label, catalog=False)
    return TaskSpec(
        task_id=task_id,
        taskpack_version=taskpack_version,
        runtime_contract_version=runtime_version,
        source_sha256=source_sha256,
        taskpack_digest=digest_json(taskpack),
        builder_spec_digest=builder_digest,
        scene=scene,
        source_path=source_path,
        info=info,
        preparation_mode=resolved_preparation_mode,
        _taskpack_json=_snapshot(taskpack),
        _builder_spec_json=_snapshot(builder_spec),
    )


def _validate_public_task_fields(
    taskpack: Mapping[str, Any],
    runtime: Mapping[str, Any],
    task_id: str,
) -> str:
    story = _public_mapping(taskpack, "story", task_id, "")
    _public_text(story, "premise", task_id, "story")
    _public_text(story, "assistant_goal", task_id, "story")
    prologue = _public_mapping(story, "prologue", task_id, "story")
    actions = _public_array(prologue, "actions", task_id, "story.prologue")
    _validate_text_items(actions, task_id, "story.prologue.actions")
    say = prologue.get("say")
    if not isinstance(say, str):
        raise TaskValidationError(f"{task_id}: story.prologue.say must be a string")

    tick_budget = runtime.get("tick_budget")
    if type(tick_budget) is not int or tick_budget <= 0:
        raise TaskValidationError(
            f"{task_id}: runtime.tick_budget must be a positive integer"
        )
    human_policy = _public_mapping(runtime, "human_policy", task_id, "runtime")
    _validate_human_policy(human_policy, task_id)
    return _public_preparation_mode(runtime, task_id)


def _public_preparation_mode(runtime: Mapping[str, Any], task_id: str) -> str:
    if "public_preparation" not in runtime:
        return "prologue"
    value = runtime["public_preparation"]
    if not isinstance(value, Mapping):
        raise TaskValidationError(
            f"{task_id}: runtime.public_preparation must be an object"
        )
    mode = value.get("mode")
    if not isinstance(mode, str) or mode not in _PREPARATION_MODES:
        raise TaskValidationError(
            f"{task_id}: runtime.public_preparation.mode must be one of "
            f"{sorted(_PREPARATION_MODES)!r}"
        )
    return mode


def _validate_human_policy(policy: Mapping[str, Any], task_id: str) -> None:
    version = _public_text(policy, "version", task_id, "runtime.human_policy")
    if version != _HUMAN_POLICY_VERSION:
        raise TaskValidationError(
            f"{task_id}: unsupported runtime.human_policy.version {version!r}; "
            f"expected {_HUMAN_POLICY_VERSION!r}"
        )
    initial_state = _public_mapping(
        policy, "initial_state", task_id, "runtime.human_policy"
    )
    if "facing" not in initial_state:
        raise TaskValidationError(
            f"{task_id}: runtime.human_policy.initial_state.facing is required"
        )
    facing = initial_state.get("facing")
    if facing is not None and (not isinstance(facing, str) or not facing.strip()):
        raise TaskValidationError(
            f"{task_id}: runtime.human_policy.initial_state.facing "
            "must be null or a non-empty string"
        )
    _public_text(
        initial_state, "posture", task_id, "runtime.human_policy.initial_state"
    )
    held = _public_array(
        initial_state, "held", task_id, "runtime.human_policy.initial_state"
    )
    _validate_text_items(held, task_id, "runtime.human_policy.initial_state.held")

    default_plan = _public_array(
        policy, "default_plan", task_id, "runtime.human_policy"
    )
    _validate_text_items(
        default_plan,
        task_id,
        "runtime.human_policy.default_plan",
        require_nonempty=True,
    )
    intervention = _public_mapping(
        policy, "intervention_window", task_id, "runtime.human_policy"
    )
    _public_text(
        intervention,
        "after_beat",
        task_id,
        "runtime.human_policy.intervention_window",
    )
    _public_text(
        intervention,
        "before_beat",
        task_id,
        "runtime.human_policy.intervention_window",
    )
    transitions = _public_array(policy, "transitions", task_id, "runtime.human_policy")
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            raise TaskValidationError(
                f"{task_id}: runtime.human_policy.transitions[{index}] "
                "must be an object"
            )
    terminal_beats = _public_array(
        policy, "terminal_beats", task_id, "runtime.human_policy"
    )
    _validate_text_items(
        terminal_beats,
        task_id,
        "runtime.human_policy.terminal_beats",
        require_nonempty=True,
    )


def _public_mapping(
    value: Mapping[str, Any],
    field_name: str,
    task_id: str,
    parent_path: str,
) -> Mapping[str, Any]:
    field_value = value.get(field_name)
    field_path = f"{parent_path}.{field_name}" if parent_path else field_name
    if not isinstance(field_value, Mapping):
        raise TaskValidationError(f"{task_id}: {field_path} must be an object")
    return field_value


def _public_array(
    value: Mapping[str, Any],
    field_name: str,
    task_id: str,
    parent_path: str,
) -> list[Any]:
    field_value = value.get(field_name)
    field_path = f"{parent_path}.{field_name}" if parent_path else field_name
    if not isinstance(field_value, list):
        raise TaskValidationError(f"{task_id}: {field_path} must be an array")
    return field_value


def _public_text(
    value: Mapping[str, Any],
    field_name: str,
    task_id: str,
    parent_path: str,
) -> str:
    field_value = value.get(field_name)
    field_path = f"{parent_path}.{field_name}" if parent_path else field_name
    if not isinstance(field_value, str) or not field_value.strip():
        raise TaskValidationError(f"{task_id}: {field_path} must be a non-empty string")
    return field_value.strip()


def _validate_text_items(
    values: list[Any],
    task_id: str,
    field_path: str,
    *,
    require_nonempty: bool = False,
) -> None:
    if require_nonempty and not values:
        raise TaskValidationError(f"{task_id}: {field_path} must not be empty")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise TaskValidationError(
                f"{task_id}: {field_path}[{index}] must be a non-empty string"
            )


def _validated_builder(
    taskpack: Mapping[str, Any], source_path: Path, task_id: str
) -> tuple[dict[str, Any], str]:
    executable_spec = _required_mapping(taskpack, "executable_spec", source_path)
    if executable_spec.get("task_id") != task_id:
        raise TaskValidationError(
            f"{task_id}: executable_spec.task_id must match the TaskPack task_id"
        )
    try:
        builder_spec, builder_digest = builder_spec_from_taskpack(taskpack)
    except (KeyError, TypeError, ValueError) as error:
        raise TaskValidationError(
            f"{task_id}: invalid runtime builder spec: {error}"
        ) from error
    actual_builder_digest = canonical_digest(builder_spec)
    if builder_digest != actual_builder_digest:
        raise TaskValidationError(
            f"{task_id}: runtime builder digest mismatch: expected "
            f"{builder_digest}, got {actual_builder_digest}"
        )
    return builder_spec, builder_digest


def _read_json_object(path: Path, *, catalog: bool) -> dict[str, Any]:
    error_type = TaskCatalogError if catalog else TaskValidationError
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TaskValidationError) as error:
        raise error_type(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise error_type(f"{path}: document root must be an object")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TaskValidationError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise TaskValidationError(f"non-finite JSON number is not allowed: {value}")


def _required_mapping(
    value: Mapping[str, Any], field_name: str, source: Path
) -> Mapping[str, Any]:
    field_value = value.get(field_name)
    if not isinstance(field_value, Mapping):
        raise TaskValidationError(f"{source}: {field_name} must be an object")
    return field_value


def _required_text(
    value: Mapping[str, Any],
    field_name: str,
    source: Path,
    *,
    catalog: bool,
) -> str:
    field_value = value.get(field_name)
    if not isinstance(field_value, str) or not field_value.strip():
        error_type = TaskCatalogError if catalog else TaskValidationError
        raise error_type(f"{source}: {field_name} must be a non-empty string")
    return field_value.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise TaskValidationError(f"cannot hash TaskPack {path}: {error}") from error
    return f"sha256:{digest.hexdigest()}"


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TaskValidationError(
            f"{label} must use the form 'sha256:' followed by 64 lowercase hex digits"
        )


def _optional_filter(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TaskCatalogError(f"{label} filter must be a non-empty string")
    return value.strip().casefold()


def _snapshot(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _object_from_snapshot(value: str, label: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise TaskValidationError(f"{label} is not an object")
    return decoded


__all__ = [
    "INTERACTION_OPENINGS_VERSION",
    "MANIFEST_VERSION",
    "RUNTIME_CONTRACT_VERSION",
    "TASKPACK_VERSION",
    "TaskCatalog",
    "TaskInfo",
    "TaskSpec",
    "load_taskpack",
]

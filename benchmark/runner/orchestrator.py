"""Deterministic multi-GPU orchestration for the frozen benchmark."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from benchmark.evaluation.artifact_bundle import (
    ArtifactBundleError,
    verify_record_run_evidence,
)
from validate_release import ReleaseValidationError, validate as validate_release

from .config import (
    BenchmarkConfig,
    ConfigError,
    EndpointConfig,
    validate_config,
    validate_secret_environment,
)

_LOG: logging.Logger = logging.getLogger(__name__)
_PLAN_SCHEMA = "interactiveworld-run-plan/1.0"
_SHARD_SCHEMA = "interactiveworld-task-shard/1.0"
_RUN_STATUSES = {
    "PLANNED",
    "DRY_RUN",
    "RUNNING",
    "FAILED",
    "INTERRUPTED",
    "COMPLETED",
}
_REQUIRED_VALIDITY_FIELDS = (
    "scene_setup_ok",
    "runtime_complete",
    "trace_complete",
    "frames_usable",
    "assistant_transport_ok",
    "human_executor_ok",
    "simulator_ok",
    "no_material_human_leakage",
)
_REQUIRED_NATIVE_FIELDS = (
    "contract_world_success",
    "process_requirements_met",
    "verification_points_met",
    "assistant_declared_done",
    "requires_assistant_done",
)


class RunError(RuntimeError):
    """The benchmark run could not be planned or executed safely."""


@dataclass(frozen=True)
class FrozenTask:
    task_id: str
    assigned_category: str
    source_family: str
    taskpack_path: Path
    taskpack_sha256: str

    def batch_row(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "taskpack_path": str(self.taskpack_path),
            "taskpack_sha256": self.taskpack_sha256,
        }


@dataclass(frozen=True)
class WorkerPlan:
    index: int
    gpu: str
    tasks: tuple[FrozenTask, ...]
    output_dir: Path
    batch_manifest: Path
    results_path: Path
    log_path: Path
    command: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "gpu": self.gpu,
            "task_ids": [task.task_id for task in self.tasks],
            "output_dir": str(self.output_dir),
            "batch_manifest": str(self.batch_manifest),
            "results": str(self.results_path),
            "log": str(self.log_path),
            "command": list(self.command),
        }


@dataclass(frozen=True)
class RunPlan:
    release_root: Path
    task_manifest: Path
    config: BenchmarkConfig
    selected_tasks: tuple[FrozenTask, ...]
    completed_task_ids: tuple[str, ...]
    workers: tuple[WorkerPlan, ...]
    execution_identity: dict[str, object]
    execution_fingerprint: str
    runtime_provenance: dict[str, object]

    @property
    def manifest_path(self) -> Path:
        return self.config.output_dir / "run_manifest.json"

    def public_dict(
        self,
        status: str = "PLANNED",
        completed_task_ids: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        completed = (
            self.completed_task_ids
            if completed_task_ids is None
            else completed_task_ids
        )
        return {
            "schema_version": _PLAN_SCHEMA,
            "run_id": self.config.run_id,
            "status": status,
            "release_root": str(self.release_root),
            "task_manifest": str(self.task_manifest),
            "task_count": len(self.selected_tasks),
            "pending_task_count": len(self.selected_tasks) - len(completed),
            "selection": {
                "task_ids": [task.task_id for task in self.selected_tasks],
                "completed_task_ids": list(completed),
            },
            "configuration": self.config.public_dict(),
            "execution_identity": self.execution_identity,
            "execution_fingerprint": self.execution_fingerprint,
            "runtime_provenance": self.runtime_provenance,
            "workers_scope": "current_attempt",
            "workers": [worker.public_dict() for worker in self.workers],
        }


@dataclass(frozen=True)
class WorkerResult:
    index: int
    gpu: str
    return_code: int
    task_ids: tuple[str, ...]
    log_path: Path

    def public_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "gpu": self.gpu,
            "return_code": self.return_code,
            "task_ids": list(self.task_ids),
            "log": str(self.log_path),
        }


@dataclass
class _ProcessState:
    worker: WorkerPlan
    process: subprocess.Popen[bytes]
    log_handle: BinaryIO
    result_offset: int
    result_buffer: bytes = b""


def _reject_json_constant(value: str) -> None:
    raise RunError(f"non-finite JSON number is not allowed: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RunError(f"duplicate JSON field: {key!r}")
        value[key] = item
    return value


def _decode_json(text: str, source: Path | str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RunError) as exc:
        raise RunError(f"cannot read JSON {source}: {exc}") from exc


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunError(f"cannot read JSON {path}: {exc}") from exc
    return _decode_json(text, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RunError(f"cannot hash release artifact {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _tree_sha256(root: Path, excluded: set[Path]) -> str:
    digest = hashlib.sha256()
    files: list[tuple[Path, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RunError(f"release identity tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in excluded or "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append((relative, path))
    for relative, path in sorted(files, key=lambda item: item[0].as_posix()):
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise RunError(f"cannot hash release artifact {path}: {exc}") from exc
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
    return "sha256:" + digest.hexdigest()


def load_release_tasks(manifest_path: Path) -> tuple[FrozenTask, ...]:
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise RunError(f"invalid task manifest: {manifest_path}")
    if payload.get("manifest_version") != "interactive-world-task-manifest/1.0":
        raise RunError("release task manifest has an unsupported version")
    if payload.get("release_id") != "InteractiveWorld-Balanced120":
        raise RunError("release task manifest has an unexpected release_id")
    if payload.get("taskpack_root") != "taskpacks":
        raise RunError("release task manifest taskpack_root must be 'taskpacks'")
    tasks = tuple(_parse_task(row, manifest_path) for row in payload["tasks"])
    declared = payload.get("task_count")
    if declared != len(tasks) or len(tasks) != 120:
        raise RunError(
            f"release manifest must contain exactly 120 tasks, found {len(tasks)}"
        )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RunError("release task manifest contains duplicate task IDs")
    if len({task.taskpack_path for task in tasks}) != len(tasks):
        raise RunError("release task manifest contains duplicate TaskPack paths")
    expected_categories = {
        "Accessibility",
        "Decision",
        "Finding",
        "Guidance",
        "Planning",
        "Prevention",
    }
    counts = Counter(task.assigned_category for task in tasks)
    if set(counts) != expected_categories or any(
        count != 20 for count in counts.values()
    ):
        raise RunError("release task manifest must contain six categories of 20 tasks")
    return tasks


def _parse_task(value: Any, manifest_path: Path) -> FrozenTask:
    if not isinstance(value, dict):
        raise RunError("every release task entry must be an object")
    try:
        task_id = value["task_id"]
        assigned_category = value["assigned_family"]
        source_family = value["source_family"]
        relative_path = value["taskpack_path"]
        expected_hash = value["taskpack_sha256"]
    except KeyError as exc:
        raise RunError(f"task manifest entry is missing {exc.args[0]!r}") from exc
    fields = (task_id, assigned_category, source_family, relative_path, expected_hash)
    if not all(isinstance(field, str) and field for field in fields):
        raise RunError("task manifest identity fields must be non-empty strings")
    if not _is_sha256(expected_hash):
        raise RunError(f"invalid TaskPack hash for {task_id}: {expected_hash!r}")
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise RunError(f"TaskPack path for {task_id} must be relative")
    taskpack_root = (manifest_path.parent / "taskpacks").resolve()
    path = (manifest_path.parent / raw_path).resolve()
    if not path.is_relative_to(taskpack_root) or not path.is_file():
        raise RunError(f"TaskPack path for {task_id} is missing or escapes taskpacks/")
    actual_hash = _file_sha256(path)
    if actual_hash != expected_hash:
        raise RunError(
            f"TaskPack hash mismatch for {task_id}: expected {expected_hash}, "
            f"got {actual_hash}"
        )
    return FrozenTask(
        task_id=task_id,
        assigned_category=assigned_category,
        source_family=source_family,
        taskpack_path=path,
        taskpack_sha256=expected_hash,
    )


def _is_sha256(value: str) -> bool:
    prefix = "sha256:"
    if not value.startswith(prefix) or len(value) != len(prefix) + 64:
        return False
    return all(character in "0123456789abcdef" for character in value[len(prefix) :])


def select_tasks(
    tasks: tuple[FrozenTask, ...],
    task_ids: tuple[str, ...],
    categories: tuple[str, ...],
) -> tuple[FrozenTask, ...]:
    if task_ids and categories:
        raise ConfigError("task IDs and categories are mutually exclusive selectors")
    if len(task_ids) != len(set(task_ids)):
        raise ConfigError("task IDs must not be repeated")
    known_ids = {task.task_id for task in tasks}
    unknown_ids = sorted(set(task_ids) - known_ids)
    if unknown_ids:
        raise ConfigError("unknown task ID(s): " + ", ".join(unknown_ids))
    if task_ids:
        wanted = set(task_ids)
        return tuple(task for task in tasks if task.task_id in wanted)
    if not categories:
        return tasks
    canonical = {
        task.assigned_category.casefold(): task.assigned_category for task in tasks
    }
    requested = {category.casefold() for category in categories}
    unknown = sorted(requested - set(canonical))
    if unknown:
        raise ConfigError("unknown category name(s): " + ", ".join(unknown))
    return tuple(
        task for task in tasks if task.assigned_category.casefold() in requested
    )


def _release_identity(
    release_root: Path, selected_tasks: tuple[FrozenTask, ...]
) -> dict[str, object]:
    release_path = release_root / "release_manifest.json"
    rubric_path = release_root / "benchmark/evaluation/rubric.json"
    simulator_path = release_root / "simulator" / "component_manifest.json"
    task_manifest_path = release_root / "benchmark/tasks/manifest.json"
    release = _identity_manifest(release_path)
    rubric = _identity_manifest(rubric_path)
    simulator = _identity_manifest(simulator_path)
    simulator_tree = _identity_manifest_field(
        simulator, "tree_integrity", simulator_path
    )
    _validate_identity_values(release, rubric, simulator, simulator_tree)
    return {
        "release": {
            "release_id": release.get("release_id"),
            "release_version": release.get("release_version"),
            "source_revision": release.get("source_revision"),
            "manifest_sha256": _file_sha256(release_path),
        },
        "task_manifest_sha256": _file_sha256(task_manifest_path),
        "rubric": {
            "schema_version": rubric.get("schema_version"),
            "benchmark_id": rubric.get("benchmark_id"),
            "sha256": _file_sha256(rubric_path),
        },
        "simulator": {
            "schema_version": simulator.get("schema_version"),
            "release_version": simulator.get("release_version"),
            "declared_tree_sha256": simulator_tree.get("sha256"),
            "actual_tree_sha256": _tree_sha256(
                release_root / "simulator", {Path("component_manifest.json")}
            ),
            "manifest_sha256": _file_sha256(simulator_path),
        },
        "tasks": [
            {"task_id": task.task_id, "taskpack_sha256": task.taskpack_sha256}
            for task in selected_tasks
        ],
    }


def _validate_identity_values(
    release: Mapping[str, Any],
    rubric: Mapping[str, Any],
    simulator: Mapping[str, Any],
    simulator_tree: Mapping[str, Any],
) -> None:
    required = (
        release.get("release_id"),
        release.get("release_version"),
        release.get("source_revision"),
        rubric.get("schema_version"),
        rubric.get("benchmark_id"),
        simulator.get("schema_version"),
        simulator.get("release_version"),
        simulator_tree.get("sha256"),
    )
    if not all(isinstance(value, str) and value for value in required):
        raise RunError("release identity manifests contain missing identity fields")


def _identity_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise RunError(f"release identity manifest must be a JSON object: {path}")
    return value


def _identity_manifest_field(
    manifest: Mapping[str, Any], field: str, path: Path
) -> dict[str, Any]:
    value = manifest.get(field)
    if not isinstance(value, dict):
        raise RunError(f"release identity manifest {path} is missing {field}")
    return value


def _execution_fingerprint(
    config: BenchmarkConfig, execution_identity: Mapping[str, object]
) -> str:
    public = config.public_dict()
    for key in ("resume", "dry_run", "adjudication", "video"):
        public.pop(key, None)
    payload = {
        "configuration": public,
        "execution_identity": execution_identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        incomplete_last_line = bool(text and not text.endswith(("\n", "\r")))
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                value = _decode_json(line, f"{path}:{line_number}")
            except RunError:
                if incomplete_last_line and line_number == len(lines):
                    break
                raise
            if not isinstance(value, dict):
                raise RunError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    except OSError as exc:
        raise RunError(f"cannot read runner results {path}: {exc}") from exc
    return rows


def load_result_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load latest attempts, rejecting a task that appears in multiple shards."""
    records: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    workers_dir = output_dir / "workers"
    if workers_dir.is_symlink():
        raise RunError(f"workers directory must not be a symbolic link: {workers_dir}")
    if not workers_dir.is_dir():
        return records
    for path in sorted(workers_dir.glob("*/results.jsonl")):
        if path.parent.is_symlink() or path.is_symlink():
            raise RunError(f"runner result path must not be a symbolic link: {path}")
        for value in _load_jsonl(path):
            task_id = str(value.get("task_id") or "")
            if task_id:
                previous_source = sources.get(task_id)
                if previous_source is not None and previous_source != path:
                    raise RunError(
                        f"task {task_id} appears in multiple result shards: "
                        f"{previous_source} and {path}"
                    )
                records[task_id] = {**value, "artifact_dir": str(path.parent)}
                sources[task_id] = path
    return records


def _complete_record(task: FrozenTask, row: Mapping[str, Any]) -> bool:
    if row.get("task_id") != task.task_id:
        return False
    if row.get("taskpack_file_sha256") != task.taskpack_sha256 or row.get("error"):
        return False
    evidence_hash = row.get("run_evidence_sha256")
    if not isinstance(evidence_hash, str) or not _is_sha256(evidence_hash):
        return False
    artifact_dir = row.get("artifact_dir")
    tag = row.get("tag")
    if not isinstance(artifact_dir, str) or not isinstance(tag, str):
        return False
    try:
        verify_record_run_evidence(row, artifact_dir)
    except (ArtifactBundleError, OSError):
        return False
    validity = row.get("validity_evidence")
    if not isinstance(validity, Mapping) or any(
        validity.get(name) is not True for name in _REQUIRED_VALIDITY_FIELDS
    ):
        return False
    native = row.get("native_taskpack_outcome")
    if not isinstance(native, Mapping) or any(
        type(native.get(name)) is not bool for name in _REQUIRED_NATIVE_FIELDS
    ):
        return False
    release = row.get("release_rubric_outcome")
    if not isinstance(release, Mapping) or any(
        name not in release
        for name in ("status", "rubric_version", "strict_pass", "completion_score")
    ):
        return False
    return all(
        isinstance(row.get(name), str) and bool(row[name])
        for name in ("taskpack_digest", "builder_spec_digest")
    )


def _completed_ids(
    config: BenchmarkConfig, selected_tasks: tuple[FrozenTask, ...]
) -> set[str]:
    if not config.resume:
        return set()
    records = load_result_records(config.output_dir)
    return {
        task.task_id
        for task in selected_tasks
        if (row := records.get(task.task_id)) is not None
        and _complete_record(task, row)
    }


def _partition(
    selected: tuple[FrozenTask, ...],
    gpus: tuple[str, ...],
    completed: set[str],
) -> list[tuple[int, str, tuple[FrozenTask, ...]]]:
    partitions: list[list[FrozenTask]] = [[] for _gpu in gpus]
    for index, task in enumerate(selected):
        if task.task_id not in completed:
            partitions[index % len(gpus)].append(task)
    return [
        (index, gpu, tuple(tasks))
        for index, (gpu, tasks) in enumerate(zip(gpus, partitions))
        if tasks
    ]


def _append_endpoint(command: list[str], role: str, config: EndpointConfig) -> None:
    command.append(f"--{role}-provider={config.provider}")
    command.append(f"--{role}-model={config.model}")
    command.append(f"--{role}-base-url={config.base_url}")
    command.append(f"--{role}-api-key-env={config.api_key_env}")
    command.append(f"--{role}-timeout-s={config.timeout_s}")
    command.append(f"--{role}-frame-cap={config.frame_cap}")
    command.append(f"--{role}-tokens={config.tokens}")


def _worker_command(
    release_root: Path,
    config: BenchmarkConfig,
    task_manifest: Path,
    batch_manifest: Path,
    output_dir: Path,
    results_path: Path,
    worker_index: int,
    gpu: str,
) -> tuple[str, ...]:
    command = [
        str(release_root / "launch" / "run_taskpack.sh"),
        f"--taskpack-batch={batch_manifest}",
        f"--task-manifest={task_manifest}",
        f"--assistant={config.assistant_control}",
    ]
    _append_endpoint(command, "assistant", config.assistant)
    _append_endpoint(command, "human", config.human)
    command.extend(
        [
            f"--seed={config.seed}",
            f"--max-ticks={config.max_ticks}",
            f"--frame-budget={config.frame_budget}",
            f"--per-interval={config.frames_per_tick}",
            f"--max-frames={config.max_frames}",
            f"--frame-max-side={config.frame_max_side}",
            f"--human-invalid-attempt-limit={config.human_invalid_attempt_limit}",
            f"--human-http-attempt-limit={config.human_http_attempt_limit}",
            f"--human-http-retry-backoff-s={config.human_http_retry_backoff_s}",
            f"--active-gpu={gpu}",
            f"--output-dir={output_dir}",
            f"--results={results_path}",
            f"--tag-prefix=IW_{config.run_id}_w{worker_index}",
            f"--io-capture={config.io_capture}",
        ]
    )
    if config.graceful_exit:
        command.append("--graceful-exit")
    return tuple(command)


def _worker_plan(
    release_root: Path,
    task_manifest: Path,
    config: BenchmarkConfig,
    index: int,
    gpu: str,
    tasks: tuple[FrozenTask, ...],
) -> WorkerPlan:
    gpu_label = gpu.replace("/", "_")
    worker_dir = config.output_dir / "workers" / f"gpu_{index}_{gpu_label}"
    batch = worker_dir / "task_batch.json"
    results = worker_dir / "results.jsonl"
    command = _worker_command(
        release_root, config, task_manifest, batch, worker_dir, results, index, gpu
    )
    return WorkerPlan(
        index=index,
        gpu=gpu,
        tasks=tasks,
        output_dir=worker_dir,
        batch_manifest=batch,
        results_path=results,
        log_path=worker_dir / "runner.log",
        command=command,
    )


def build_run_plan(release_root: Path, config: BenchmarkConfig) -> RunPlan:
    validate_config(config)
    root = release_root.resolve()
    manifest = root / "benchmark" / "tasks" / "manifest.json"
    selected = select_tasks(
        load_release_tasks(manifest), config.task_ids, config.categories
    )
    completed = _completed_ids(config, selected)
    partitions = _partition(selected, config.gpus, completed)
    workers = tuple(
        _worker_plan(root, manifest, config, index, gpu, tasks)
        for index, gpu, tasks in partitions
    )
    execution_identity = _release_identity(root, selected)
    return RunPlan(
        release_root=root,
        task_manifest=manifest,
        config=config,
        selected_tasks=selected,
        completed_task_ids=tuple(
            task.task_id for task in selected if task.task_id in completed
        ),
        workers=workers,
        execution_identity=execution_identity,
        execution_fingerprint=_execution_fingerprint(config, execution_identity),
        runtime_provenance={
            "isaac_sim_expected": "5.1",
            "isaac_python_executable": os.environ.get("ISAAC_PYTHON") or None,
            "python_version": None,
            "version_probe": "not_run",
        },
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _check_existing_plan(plan: RunPlan) -> None:
    if plan.config.output_dir.exists() and not plan.config.output_dir.is_dir():
        raise RunError(f"output path is not a directory: {plan.config.output_dir}")
    if not plan.manifest_path.exists():
        _check_fresh_output(plan)
        return
    previous = _read_json(plan.manifest_path)
    if not isinstance(previous, dict):
        raise RunError("existing run manifest must be a JSON object")
    _check_previous_plan(plan, previous)


def _check_fresh_output(plan: RunPlan) -> None:
    if plan.config.resume:
        raise RunError("resume requires an existing run manifest")
    owned = (
        plan.config.output_dir / "workers",
        plan.config.output_dir / "reports",
        plan.config.output_dir / "videos",
        plan.config.output_dir / "evidence",
        plan.config.output_dir / "evidence_templates",
        plan.config.output_dir / "report.json",
    )
    if any(path.exists() for path in owned):
        raise RunError(
            "output directory contains runner artifacts but no compatible run manifest"
        )


def _check_previous_plan(plan: RunPlan, previous: Mapping[str, Any]) -> None:
    if previous.get("schema_version") != _PLAN_SCHEMA:
        raise RunError("existing run manifest has an unsupported schema version")
    if previous.get("execution_fingerprint") != plan.execution_fingerprint:
        raise RunError(
            "requested configuration does not match the existing run manifest"
        )
    previous_status = previous.get("status")
    if previous_status not in _RUN_STATUSES:
        raise RunError(f"existing run manifest has invalid status: {previous_status!r}")
    if previous.get("run_id") != plan.config.run_id:
        raise RunError("run_id does not match the existing run manifest")
    if previous_status == "DRY_RUN":
        _validate_dry_run_artifacts(plan.config.output_dir)
        return
    if plan.config.dry_run:
        raise RunError("a dry run cannot replace a manifest from an executed run")
    if not plan.config.resume:
        raise RunError(
            f"run manifest already exists at {plan.manifest_path}; use --resume "
            "or another output directory"
        )


def _validate_dry_run_artifacts(output_dir: Path) -> None:
    forbidden = (
        output_dir / "reports",
        output_dir / "videos",
        output_dir / "evidence",
        output_dir / "evidence_templates",
        output_dir / "report.json",
    )
    if any(path.exists() for path in forbidden):
        raise RunError("DRY_RUN manifest is mixed with execution artifacts")
    workers_dir = output_dir / "workers"
    if not workers_dir.is_dir():
        return
    for worker_dir in workers_dir.iterdir():
        if not worker_dir.is_dir():
            raise RunError("DRY_RUN workers directory contains an unexpected entry")
        unexpected = [
            path for path in worker_dir.iterdir() if path.name != "task_batch.json"
        ]
        if unexpected:
            raise RunError("DRY_RUN manifest is mixed with worker execution artifacts")


def preflight_run(plan: RunPlan) -> None:
    """Validate output ownership and the bundled worker entrypoint without writes."""
    release_root = plan.release_root.resolve()
    if plan.config.output_dir.is_symlink():
        raise RunError("output directory must not be a symbolic link")
    output_root = plan.config.output_dir.resolve()
    if output_root == Path(output_root.anchor):
        raise RunError("output directory must not be a filesystem root")
    if output_root == release_root or output_root.is_relative_to(release_root):
        raise RunError("output directory must be outside the immutable release tree")
    _validate_owned_output_paths(output_root)
    try:
        validate_release(release_root / "release_manifest.json", check_hygiene=False)
    except (
        ReleaseValidationError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        raise RunError(f"release integrity preflight failed: {exc}") from exc
    _check_existing_plan(plan)
    _validate_worker_paths(plan)
    launcher = plan.release_root / "launch" / "run_taskpack.sh"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise RunError(f"benchmark worker launcher is not executable: {launcher}")


def _validate_owned_output_paths(output_root: Path) -> None:
    directory_paths = (
        output_root / "workers",
        output_root / "reports",
        output_root / "videos",
        output_root / "evidence",
        output_root / "evidence" / "pending",
        output_root / "evidence" / "adjudicated",
        output_root / "evidence_templates",
    )
    file_paths = (
        output_root / "report.json",
        output_root / "run_manifest.json",
    )
    for path in (*directory_paths, *file_paths):
        if path.is_symlink():
            raise RunError(f"runner-owned output path must not be a symlink: {path}")
        if path.exists() and not path.resolve().is_relative_to(output_root):
            raise RunError(f"runner-owned output path escapes output directory: {path}")
    for path in directory_paths:
        if path.exists() and not path.is_dir():
            raise RunError(f"runner-owned output directory is not a directory: {path}")
    for path in file_paths:
        if path.exists() and not path.is_file():
            raise RunError(f"runner-owned output file is not a regular file: {path}")


def _validate_worker_paths(plan: RunPlan) -> None:
    output_root = plan.config.output_dir.resolve()
    for worker in plan.workers:
        paths = (
            worker.output_dir,
            worker.batch_manifest,
            worker.results_path,
            worker.log_path,
        )
        if any(path.is_symlink() for path in paths):
            raise RunError("worker output paths must not be symbolic links")
        if any(not path.resolve().is_relative_to(output_root) for path in paths):
            raise RunError("worker output path escapes the configured output directory")


def prepare_run(plan: RunPlan) -> None:
    preflight_run(plan)
    manifest = _manifest_for_new_attempt(plan)
    plan.config.output_dir.mkdir(parents=True, exist_ok=True)
    for worker in plan.workers:
        worker.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            worker.batch_manifest,
            {
                "schema_version": _SHARD_SCHEMA,
                "run_id": plan.config.run_id,
                "gpu": worker.gpu,
                "tasks": [task.batch_row() for task in worker.tasks],
            },
        )
    _atomic_write_json(plan.manifest_path, manifest)


def _manifest_for_new_attempt(plan: RunPlan) -> dict[str, object]:
    attempts = _existing_attempts(plan.manifest_path)
    attempt_index = len(attempts) + 1
    attempts.append(_attempt_payload(plan, attempt_index, "PLANNED"))
    payload = plan.public_dict()
    payload["attempt_index"] = attempt_index
    payload["attempts"] = attempts
    return payload


def _attempt_payload(
    plan: RunPlan, attempt_index: int, status: str
) -> dict[str, object]:
    return {
        "attempt_index": attempt_index,
        "status": status,
        "preexisting_completed_task_ids": list(plan.completed_task_ids),
        "scheduled_task_ids": [
            task.task_id for worker in plan.workers for task in worker.tasks
        ],
        "runtime_provenance": plan.runtime_provenance,
        "workers": [worker.public_dict() for worker in plan.workers],
        "worker_results": [],
    }


def _existing_attempts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    previous = _read_json(path)
    if not isinstance(previous, dict):
        raise RunError("existing run manifest must be a JSON object")
    attempts = previous.get("attempts")
    if attempts is None:
        return [_legacy_attempt(previous)]
    if not isinstance(attempts, list) or not all(
        isinstance(attempt, dict) for attempt in attempts
    ):
        raise RunError("existing run manifest has invalid attempt history")
    for index, attempt in enumerate(attempts, 1):
        _validate_attempt(attempt, index)
    return [dict(attempt) for attempt in attempts]


def _validate_attempt(attempt: Mapping[str, Any], expected_index: int) -> None:
    if attempt.get("attempt_index") != expected_index:
        raise RunError("existing run manifest has non-contiguous attempt history")
    if attempt.get("status") not in _RUN_STATUSES:
        raise RunError("existing run manifest has invalid attempt status")
    for name in (
        "preexisting_completed_task_ids",
        "scheduled_task_ids",
        "workers",
        "worker_results",
    ):
        if not isinstance(attempt.get(name), list):
            raise RunError(f"existing run manifest attempt has invalid {name}")


def _legacy_attempt(previous: Mapping[str, Any]) -> dict[str, object]:
    workers = previous.get("workers", [])
    results = previous.get("worker_results", [])
    if not isinstance(workers, list) or not isinstance(results, list):
        raise RunError("existing run manifest has invalid worker provenance")
    selection = previous.get("selection")
    completed = (
        selection.get("completed_task_ids", []) if isinstance(selection, dict) else []
    )
    scheduled = _legacy_scheduled_task_ids(workers)
    return {
        "attempt_index": 1,
        "status": previous.get("status"),
        "preexisting_completed_task_ids": completed,
        "scheduled_task_ids": scheduled,
        "workers": workers,
        "worker_results": results,
    }


def _legacy_scheduled_task_ids(workers: list[Any]) -> list[str]:
    task_ids: list[str] = []
    for worker in workers:
        if not isinstance(worker, dict) or not isinstance(worker.get("task_ids"), list):
            raise RunError("existing run manifest has invalid worker provenance")
        values = worker["task_ids"]
        if not all(isinstance(task_id, str) for task_id in values):
            raise RunError("existing run manifest has invalid worker task IDs")
        task_ids.extend(values)
    return task_ids


def _start_worker(worker: WorkerPlan) -> _ProcessState:
    worker.output_dir.mkdir(parents=True, exist_ok=True)
    _repair_jsonl_tail(worker.results_path)
    log_handle = worker.log_path.open("ab")
    environment = os.environ.copy()
    # SimulationApp and PhysX share a physical active_gpu index. CUDA remapping
    # makes CUDA device 0 disagree with the renderer's physical GPU on workers
    # assigned to GPU > 0, so the evaluator receives the physical index in its
    # command and the child must not inherit CUDA_VISIBLE_DEVICES.
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    offset = worker.results_path.stat().st_size if worker.results_path.exists() else 0
    try:
        # This release intentionally stays stdlib-only. The executable is fixed by
        # the release plan, arguments are separately validated, and no shell runs.
        process = subprocess.Popen(
            list(worker.command),
            shell=False,
            cwd=str(worker.output_dir),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        log_handle.close()
        raise
    return _ProcessState(worker, process, log_handle, offset)


def _repair_jsonl_tail(path: Path) -> None:
    """Prevent a crashed final write from corrupting the next append attempt."""
    if not path.exists():
        return
    if path.is_symlink():
        raise RunError(f"runner results must not be a symbolic link: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RunError(f"cannot inspect runner results {path}: {exc}") from exc
    if not data or data.endswith(b"\n"):
        return
    tail_start = data.rfind(b"\n") + 1
    tail = data[tail_start:]
    valid_object = False
    try:
        value = _decode_json(tail.decode("utf-8"), path)
        valid_object = isinstance(value, dict)
    except (RunError, UnicodeDecodeError):
        value = None
    if value is not None and not valid_object:
        raise RunError(f"runner result line must be a JSON object: {path}")
    try:
        with path.open("r+b") as handle:
            if valid_object:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
            else:
                handle.truncate(tail_start)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RunError(f"cannot repair runner results {path}: {exc}") from exc


def _new_result_error(state: _ProcessState) -> bool:
    path = state.worker.results_path
    if not path.exists():
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(state.result_offset)
            block = handle.read()
            state.result_offset = handle.tell()
    except OSError as exc:
        raise RunError(f"cannot monitor {path}: {exc}") from exc
    pending = state.result_buffer + block
    lines = pending.split(b"\n")
    if pending.endswith(b"\n"):
        state.result_buffer = b""
    else:
        state.result_buffer = lines.pop()
    if state.process.poll() is not None and state.result_buffer.strip():
        lines.append(state.result_buffer)
        state.result_buffer = b""
    for line in lines:
        if not line.strip():
            continue
        try:
            value = _decode_json(line.decode("utf-8"), path)
        except UnicodeDecodeError as exc:
            raise RunError(f"runner results are not UTF-8 JSONL: {path}") from exc
        if isinstance(value, dict) and value.get("error"):
            return True
        if not isinstance(value, dict):
            raise RunError(f"runner result line must be a JSON object: {path}")
    return False


def _stop_workers(states: list[_ProcessState]) -> None:
    running = [state for state in states if state.process.poll() is None]
    for state in running:
        state.process.terminate()
    deadline = time.monotonic() + 10.0
    while running and time.monotonic() < deadline:
        running = [state for state in running if state.process.poll() is None]
        if running:
            time.sleep(0.1)
    for state in running:
        state.process.kill()
    for state in states:
        try:
            state.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            raise RunError(f"worker {state.worker.index} did not stop") from exc


def _wait_workers(states: list[_ProcessState], fail_fast: bool) -> bool:
    stopped_early = False
    while any(state.process.poll() is None for state in states):
        failed = any(
            state.process.poll() not in (None, 0) or _new_result_error(state)
            for state in states
        )
        if fail_fast and failed:
            stopped_early = True
            _stop_workers(states)
            break
        time.sleep(0.25)
    return stopped_early


def execute_run(plan: RunPlan) -> tuple[list[WorkerResult], bool]:
    if plan.config.dry_run or not plan.workers:
        return [], False
    _validate_worker_paths(plan)
    validate_execution_environment(plan.config)
    states: list[_ProcessState] = []
    try:
        for worker in plan.workers:
            _LOG.info("starting worker %s on GPU %s", worker.index, worker.gpu)
            states.append(_start_worker(worker))
        stopped_early = _wait_workers(states, plan.config.fail_fast)
    except (OSError, RunError, KeyboardInterrupt):
        _stop_workers(states)
        raise
    finally:
        for state in states:
            state.log_handle.close()
    results = [
        WorkerResult(
            index=state.worker.index,
            gpu=state.worker.gpu,
            return_code=state.process.wait(),
            task_ids=tuple(task.task_id for task in state.worker.tasks),
            log_path=state.worker.log_path,
        )
        for state in states
    ]
    return results, stopped_early


def validate_execution_environment(config: BenchmarkConfig) -> dict[str, object]:
    validate_secret_environment(config)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RunError(
            "CUDA_VISIBLE_DEVICES must be unset; InteractiveWorld passes physical "
            "GPU indices directly to the Isaac renderer and PhysX"
        )
    value = os.environ.get("ISAAC_PYTHON", "")
    if not value:
        raise RunError("ISAAC_PYTHON must name the Isaac Sim 5.1 Python executable")
    executable = Path(value)
    if not executable.is_absolute():
        raise RunError("ISAAC_PYTHON must be an absolute path")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RunError(f"ISAAC_PYTHON is not executable: {executable}")
    return {
        "isaac_sim_expected": "5.1",
        "isaac_python_executable": str(executable.resolve()),
        "python_version": _python_version(executable),
        "version_probe": "python --version only; Isaac Sim package version unverified",
    }


def _python_version(executable: Path) -> str:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        # Public releases cannot depend on Meta's process wrapper. The executable
        # is the already validated absolute ISAAC_PYTHON path and no shell runs.
        result = subprocess.run(
            [str(executable), "--version"],
            shell=False,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunError(f"cannot query ISAAC_PYTHON version: {exc}") from exc
    output = result.stdout.decode("utf-8", errors="replace").strip()
    if result.returncode != 0 or not output:
        raise RunError("ISAAC_PYTHON --version did not complete successfully")
    normalized = " ".join(output.splitlines())
    if len(normalized) > 512 or any(ord(character) < 32 for character in normalized):
        raise RunError("ISAAC_PYTHON --version returned unsupported output")
    return normalized


def write_run_status(
    plan: RunPlan,
    status: str,
    workers: list[WorkerResult] | None = None,
) -> None:
    if status not in _RUN_STATUSES:
        raise RunError(f"invalid run status: {status!r}")
    completed = _current_completed_task_ids(plan, status)
    payload = plan.public_dict(status=status, completed_task_ids=completed)
    attempts = _existing_attempts(plan.manifest_path)
    if not attempts:
        attempts.append(_attempt_payload(plan, 1, status))
    current = dict(attempts[-1])
    current["status"] = status
    current["completed_task_ids"] = list(completed)
    if workers is not None:
        current["worker_results"] = [worker.public_dict() for worker in workers]
        payload["worker_results_scope"] = "current_attempt"
        payload["worker_results"] = current["worker_results"]
    attempts[-1] = current
    payload["attempt_index"] = current["attempt_index"]
    payload["attempts"] = attempts
    _atomic_write_json(plan.manifest_path, payload)


def _current_completed_task_ids(plan: RunPlan, status: str) -> tuple[str, ...]:
    if status not in {"FAILED", "INTERRUPTED", "COMPLETED"}:
        return plan.completed_task_ids
    try:
        records = load_result_records(plan.config.output_dir)
    except RunError as exc:
        _LOG.warning("cannot update completed task count: %s", exc)
        return plan.completed_task_ids
    return tuple(
        task.task_id
        for task in plan.selected_tasks
        if (row := records.get(task.task_id)) is not None
        and _complete_record(task, row)
    )


def write_json(path: Path, value: Any) -> None:
    _atomic_write_json(path, value)

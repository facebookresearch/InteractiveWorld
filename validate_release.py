#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fail-closed validation for the complete InteractiveWorld release tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "release_manifest.json"
FORBIDDEN_FILE_SUFFIXES = {".pyc", ".pyo"}
ALLOWED_DOCUMENTATION_FILES = {
    Path("README.md"),
    Path("CODE_OF_CONDUCT.md"),
    Path("CONTRIBUTING.md"),
}
DOCUMENTATION_DIRECTORY = Path("docs")
FORBIDDEN_PATH_PARTS = {"__pycache__"}
REPOSITORY_METADATA_PATH_PARTS = {".git"}
FORBIDDEN_FILE_FRAGMENTS = {".partial."}
FORBIDDEN_SIMULATOR_ASSETS = {
    "SMPLH_male.pkl",
    "walk_cycle.npz",
}
BUNDLED_HUMAN_ASSET_ROOT = PurePosixPath("human_model/assets/atlas")
# Absolute paths of a developer machine must never ship in release sources.
PRIVATE_PATH_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(r"/data/(?:repos|users)/"),
)
REQUIRED_ENTRYPOINTS = {
    "validate_release": "validate_release.py",
    "validate_install": "launch/check_install.py",
    "run_benchmark": "launch/interactiveworld.sh",
    "run_benchmark_python": "launch/interactiveworld.py",
    "runner_config": "benchmark/runner/config.example.json",
    "run_taskpack": "launch/run_taskpack.sh",
    "run_taskpack_python": "launch/run_taskpack.py",
    "run_simulator_server": "launch/run_sim_server.py",
    "run_simulator": "launch/simulator.sh",
    "run_simulator_python": "launch/simulator.py",
    "evaluate": "benchmark/evaluation/__main__.py",
}
EXECUTABLE_ENTRYPOINTS = {
    "validate_install",
    "run_benchmark",
    "run_benchmark_python",
    "run_taskpack",
    "run_taskpack_python",
    "run_simulator_server",
    "run_simulator",
    "run_simulator_python",
}
VIDEO_MODES = ("off", "failures", "all")
VIDEO_CAPTION_MODES = ("off", "speaker", "full")
API_KEY_ENVIRONMENT_NAMES = {
    "INTERACTIVEWORLD_ASSISTANT_API_KEY",
    "INTERACTIVEWORLD_HUMAN_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
}
PROVIDER_ENDPOINT_DESCRIPTIONS = {
    "http-json": (
        "Role-specific base_url using the provider-neutral /generate contract; "
        "loopback defaults are available."
    ),
    "openai-compatible": (
        "Role-specific base_url for a self-hosted OpenAI-compatible Chat "
        "Completions endpoint."
    ),
    "openai": "Hosted OpenAI endpoint, or an explicit compatible base_url.",
    "anthropic": "Hosted Anthropic Messages endpoint, or an explicit base_url.",
    "gemini": "Hosted Google Gemini generateContent endpoint, or an explicit base_url.",
}
ENDPOINT_POLICY = (
    "Remote endpoints require HTTPS; cleartext HTTP is accepted only for loopback "
    "self-hosted endpoints."
)
REQUIRED_RUNNER_OPTIONS = {
    "--config",
    "--task-id",
    "--category",
    "--all",
    "--list-tasks",
    "--assistant-control",
    "--assistant-provider",
    "--assistant-model",
    "--assistant-base-url",
    "--assistant-api-key-env",
    "--assistant-timeout-s",
    "--assistant-frame-cap",
    "--assistant-tokens",
    "--human-provider",
    "--human-model",
    "--human-base-url",
    "--human-api-key-env",
    "--human-timeout-s",
    "--human-frame-cap",
    "--human-tokens",
    "--gpus",
    "--num-gpus",
    "--seed",
    "--max-ticks",
    "--frame-budget",
    "--frames-per-tick",
    "--max-frames",
    "--frame-max-side",
    "--human-invalid-attempt-limit",
    "--human-http-attempt-limit",
    "--human-http-retry-backoff-s",
    "--io-capture",
    "--graceful-exit",
    "--fail-fast",
    "--output-dir",
    "--run-id",
    "--video",
    "--video-fps",
    "--video-resolution",
    "--video-captions",
    "--adjudication",
    "--resume",
    "--dry-run",
    "--verbose",
}


class ReleaseValidationError(RuntimeError):
    """The release is incomplete, inconsistent, or contains transient files."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseValidationError(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseValidationError(f"cannot load JSON {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _release_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    _require(
        not path.is_absolute() and ".." not in path.parts, f"unsafe path: {relative}"
    )
    resolved = root.joinpath(*path.parts)
    _require(resolved.is_file(), f"missing release file: {relative}")
    _require(
        not resolved.is_symlink(), f"release file must not be a symlink: {relative}"
    )
    return resolved


def _tree_integrity(root: Path, excluded_files: set[Path]) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in excluded_files:
            continue
        if REPOSITORY_METADATA_PATH_PARTS.intersection(relative.parts):
            continue
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        files.append((relative, path))
    for relative, path in sorted(files, key=lambda item: item[0].as_posix()):
        data = path.read_bytes()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
    return len(files), digest.hexdigest()


def _validate_declared_tree_integrity(
    root: Path,
    declared: Any,
    excluded_files: set[Path],
    label: str,
) -> None:
    _require(isinstance(declared, dict), f"{label} tree integrity is missing")
    count, digest = _tree_integrity(root, excluded_files)
    _require(declared.get("file_count") == count, f"{label} tree file count mismatch")
    _require(declared.get("sha256") == digest, f"{label} tree SHA256 mismatch")


def _validate_tree_hygiene(
    root: Path, *, allow_bytecode_cache: bool = False
) -> tuple[int, int]:
    files = 0
    python_files = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if REPOSITORY_METADATA_PATH_PARTS.intersection(relative.parts):
            continue
        if FORBIDDEN_PATH_PARTS.intersection(relative.parts):
            if allow_bytecode_cache:
                continue
            raise ReleaseValidationError(
                f"transient directory is forbidden: {relative}"
            )
        _require(not path.is_symlink(), f"release symlink is forbidden: {relative}")
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if allow_bytecode_cache and suffix in {".pyc", ".pyo"}:
            continue
        files += 1
        if suffix in {".md", ".markdown"}:
            _require(
                relative in ALLOWED_DOCUMENTATION_FILES
                or relative.parent == DOCUMENTATION_DIRECTORY,
                f"unexpected documentation file: {relative}",
            )
        _require(
            suffix not in FORBIDDEN_FILE_SUFFIXES, f"forbidden file type: {relative}"
        )
        _require(
            not any(fragment in path.name for fragment in FORBIDDEN_FILE_FRAGMENTS),
            f"partial file is forbidden: {relative}",
        )
        _validate_release_source(path, relative, suffix)
        if suffix == ".py":
            python_files += 1
    return files, python_files


def _validate_release_source(path: Path, relative: Path, suffix: str) -> None:
    if "simulator" in relative.parts:
        _require(
            path.name not in FORBIDDEN_SIMULATOR_ASSETS,
            f"no-distribution simulator asset is forbidden: {relative}",
        )
    if suffix not in {".py", ".json", ".sh", ".md", ".txt"}:
        return
    try:
        text = path.read_text(encoding="utf-8")
        if suffix == ".py":
            ast.parse(text, filename=str(relative))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ReleaseValidationError(
            f"cannot parse release source {relative}: {error}"
        ) from error
    for pattern in PRIVATE_PATH_PATTERNS:
        _require(
            pattern.search(text) is None, f"private source path remains in {relative}"
        )


def _validate_entrypoints(manifest: dict[str, Any], root: Path) -> None:
    entrypoints = manifest.get("entrypoints")
    _require(
        entrypoints == REQUIRED_ENTRYPOINTS,
        "release entrypoints are incomplete or have unexpected targets",
    )
    for name, relative in REQUIRED_ENTRYPOINTS.items():
        path = _release_file(root, relative)
        if name in EXECUTABLE_ENTRYPOINTS:
            _require(
                bool(path.stat().st_mode & 0o111),
                f"release launcher must be executable: {relative}",
            )


def _validate_bundled_human_assets(dependencies: dict[str, Any], root: Path) -> None:
    """Only the hash-pinned grasp atlas may ship under the human model."""
    entries = dependencies.get("bundled_human_assets")
    _require(isinstance(entries, list), "bundled human asset list is missing")
    declared = set()
    for entry in entries:
        _require(isinstance(entry, dict), "bundled human asset must be an object")
        relative = PurePosixPath(str(entry.get("path") or ""))
        _require(
            BUNDLED_HUMAN_ASSET_ROOT in relative.parents,
            f"unexpected bundled human asset: {relative}",
        )
        path = _release_file(root, relative.as_posix())
        _require(
            _sha256(path) == f"sha256:{entry.get('sha256')}",
            f"bundled human asset hash mismatch: {relative}",
        )
        declared.add(relative)
    packaged = {
        PurePosixPath(path.relative_to(root).as_posix())
        for path in (root / BUNDLED_HUMAN_ASSET_ROOT).rglob("*")
        if path.is_file()
    }
    _require(packaged == declared, "bundled human assets differ from their declaration")


def _validate_component_metadata(components: dict[str, Any]) -> None:
    public_api = components.get("public_api")
    _require(isinstance(public_api, dict), "simulator public API is missing")
    expected_api = {
        "simulator_facade": "simulator.api:Simulator",
        "simulator_client": "simulator.client:SimulatorClient",
        "task_catalog": "simulator.catalog:TaskCatalog",
        "world_catalog": "simulator.api:available_worlds",
        "semantic_action": "simulator.actions:Action",
        "task_preparation": "simulator.api:TaskPreparation",
        "benchmark_runner": "benchmark.runner.cli:main",
        "benchmark_report": "benchmark.runner.report:build_suite_report",
        "benchmark_video": "benchmark.runner.video:build_videos",
        "evaluation_evidence": "benchmark.evaluation.prepare:prepare_evidence",
    }
    _require(
        all(public_api.get(name) == value for name, value in expected_api.items()),
        "benchmark public API declarations are incomplete",
    )
    layers = components.get("layers")
    runner = layers.get("benchmark_runner") if isinstance(layers, dict) else None
    _require(
        runner
        == {
            "entrypoint": "benchmark.runner.cli:main",
            "purpose": (
                "One-command task selection, independent Human and Assistant "
                "model endpoints, deterministic multi-GPU sharding, reporting, "
                "evidence adjudication, and optional post-run video"
            ),
        },
        "benchmark runner component layer is missing",
    )
    versions = components.get("frozen_contract_versions")
    expected_versions = {
        "runner_config": "interactiveworld-runner-config/1.0",
        "report": "interactiveworld-report/1.0",
        "evaluation_evidence": "interactiveworld-evidence/1.1",
    }
    _require(
        isinstance(versions, dict)
        and all(
            versions.get(name) == value for name, value in expected_versions.items()
        ),
        "benchmark component contract versions are incomplete",
    )


def _validate_api_dependency_metadata(dependencies: dict[str, Any]) -> None:
    api = dependencies.get("model_api_configuration")
    expected_fields = {
        "credential_values_bundled",
        "api_key_environment_variable_names",
        "api_key_name_fields",
        "endpoint_fields",
        "endpoint_policy",
        "provider_endpoints",
    }
    _require(
        isinstance(api, dict) and set(api) == expected_fields,
        "model API dependency metadata is incomplete",
    )
    _require(
        api["credential_values_bundled"] is False, "API secrets must not be bundled"
    )
    names = api["api_key_environment_variable_names"]
    _require(
        isinstance(names, list)
        and len(names) == len(API_KEY_ENVIRONMENT_NAMES)
        and set(names) == API_KEY_ENVIRONMENT_NAMES,
        "API-key environment variable names are incomplete",
    )
    _require(
        api["api_key_name_fields"] == ["assistant.api_key_env", "human.api_key_env"],
        "API-key configuration must contain environment names only",
    )
    _require(
        api["endpoint_fields"] == ["assistant.base_url", "human.base_url"],
        "model endpoint configuration fields are incomplete",
    )
    _require(
        api["endpoint_policy"] == ENDPOINT_POLICY
        and api["provider_endpoints"] == PROVIDER_ENDPOINT_DESCRIPTIONS,
        "provider endpoint descriptions are incomplete",
    )


def _validate_ffmpeg_dependency(dependencies: dict[str, Any]) -> None:
    tools = dependencies.get("optional_system_tools")
    _require(isinstance(tools, list), "optional system tools are missing")
    ffmpeg = next(
        (
            entry
            for entry in tools
            if isinstance(entry, dict) and entry.get("name") == "ffmpeg"
        ),
        None,
    )
    _require(
        isinstance(ffmpeg, dict)
        and ffmpeg.get("command") == "ffmpeg"
        and ffmpeg.get("bundled") is False
        and ffmpeg.get("required_for")
        == ["benchmark video mode failures", "benchmark video mode all"],
        "ffmpeg optional dependency metadata is invalid",
    )


def _validate_runner_metadata(
    manifest: dict[str, Any], dependencies: dict[str, Any], root: Path
) -> None:
    from benchmark.runner.config import (
        CONFIG_SCHEMA_VERSION,
        load_config_file,
        PROVIDERS,
        VIDEO_CAPTIONS,
        VIDEO_MODES,
    )

    runner = manifest.get("benchmark_runner")
    _require(isinstance(runner, dict), "benchmark runner metadata is missing")
    _require(
        set(runner)
        == {
            "config_schema_version",
            "participant_roles",
            "providers",
            "task_selection",
            "gpu_execution",
            "video",
            "evidence",
        },
        "benchmark runner metadata has unexpected fields",
    )
    _require(
        runner.get("config_schema_version") == CONFIG_SCHEMA_VERSION,
        "benchmark runner config schema mismatch",
    )
    providers = runner.get("providers")
    _require(
        isinstance(providers, list) and tuple(providers) == tuple(PROVIDERS),
        "runner providers mismatch",
    )
    _validate_runner_semantics(runner, tuple(VIDEO_MODES), tuple(VIDEO_CAPTIONS))
    _validate_runner_cli_options()
    example = load_config_file(
        str(_release_file(root, REQUIRED_ENTRYPOINTS["runner_config"]))
    )
    _validate_runner_example(
        example, tuple(PROVIDERS), tuple(VIDEO_MODES), tuple(VIDEO_CAPTIONS)
    )
    _validate_ffmpeg_dependency(dependencies)
    _validate_api_dependency_metadata(dependencies)


def _validate_runner_cli_options() -> None:
    from benchmark.runner.cli import build_parser

    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    _require(
        REQUIRED_RUNNER_OPTIONS <= options,
        "benchmark runner CLI options are incomplete",
    )


def _validate_runner_semantics(
    runner: dict[str, Any], video_modes: tuple[str, ...], caption_modes: tuple[str, ...]
) -> None:
    _require(
        runner.get("participant_roles") == ["human", "assistant"],
        "runner roles mismatch",
    )
    _require(
        runner.get("task_selection")
        == {
            "all_120_by_default": True,
            "supports_exact_task_ids": True,
            "supports_six_categories": True,
        },
        "runner task-selection semantics mismatch",
    )
    _require(
        runner.get("gpu_execution")
        == {
            "selection_flags": ["--gpus", "--num-gpus"],
            "physical_indices": True,
            "requires_cuda_visible_devices_unset": True,
            "one_worker_per_selected_gpu": True,
            "deterministic_round_robin_sharding": True,
        },
        "runner GPU semantics mismatch",
    )
    video = runner.get("video")
    expected_video = {
        "modes": list(video_modes),
        "caption_modes": list(caption_modes),
        "encoder": "ffmpeg",
        "uses_existing_run_artifacts": True,
        "requires_io_capture_without_existing_fpv_video": True,
        "requested_video_failure_fails_run": True,
    }
    _require(
        video == expected_video,
        "runner video semantics mismatch",
    )
    _validate_evidence_semantics(runner.get("evidence"))


def _validate_evidence_semantics(value: Any) -> None:
    expected = {
        "schema_version": "interactiveworld-evidence/1.1",
        "native_and_release_scores_are_separate": True,
        "complete_trace_adjudication_required": True,
        "run_evidence_sha256_binding_required": True,
        "pending_directory": "evidence/pending",
        "adjudicated_directory": "evidence/adjudicated",
    }
    _require(value == expected, "runner evidence semantics mismatch")


def _validate_runner_example(
    config: dict[str, Any],
    providers: tuple[str, ...],
    video_modes: tuple[str, ...],
    caption_modes: tuple[str, ...],
) -> None:
    for role in ("assistant", "human"):
        endpoint = config.get(role)
        _require(
            isinstance(endpoint, dict) and endpoint.get("provider") in providers,
            f"runner example has unsupported {role} provider",
        )
    video = config.get("video")
    _require(
        isinstance(video, dict)
        and video.get("mode") in video_modes
        and video.get("captions") in caption_modes,
        "runner example has unsupported video configuration",
    )


def _validate_release_identity(manifest: dict[str, Any]) -> None:
    _require(
        manifest.get("schema_version") == "interactiveworld-release/1.0",
        "unsupported release manifest schema",
    )
    _require(
        manifest.get("release_id") == "InteractiveWorld-Balanced120",
        "release ID mismatch",
    )
    _require(manifest.get("release_version") == "1.0.0", "release version mismatch")
    distribution = manifest.get("distribution")
    _require(isinstance(distribution, dict), "distribution policy is missing")
    _require(
        distribution.get("restricted_human_assets_bundled") is False,
        "restricted human assets must not be bundled",
    )


def _validate_frozen_artifacts(manifest: dict[str, Any], root: Path) -> None:
    artifacts = manifest.get("frozen_artifacts")
    _require(
        isinstance(artifacts, dict) and artifacts, "frozen_artifacts must be non-empty"
    )
    for name, entry in artifacts.items():
        _require(isinstance(entry, dict), f"artifact {name} must be an object")
        path = _release_file(root, str(entry.get("path") or ""))
        expected = entry.get("sha256")
        _require(isinstance(expected, str), f"artifact {name} has no SHA256")
        _require(_sha256(path) == expected, f"artifact hash mismatch: {name}")


def _load_benchmark_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task_manifest_path = _release_file(root, "benchmark/tasks/manifest.json")
    evaluation_rubric_path = _release_file(root, "benchmark/evaluation/rubric.json")
    from benchmark.evaluation.rubric import load_rubric
    from benchmark.tasks.validate import validate as validate_tasks

    validate_tasks(task_manifest_path)
    task_manifest = _load_json(task_manifest_path)
    rubric = load_rubric(evaluation_rubric_path)
    _require(isinstance(task_manifest, dict), "task manifest must be an object")
    _require(isinstance(rubric, dict), "evaluation rubric must be an object")
    return task_manifest, rubric


def _validate_task_rubric_alignment(
    task_manifest: dict[str, Any], rubric: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    task_rows = {row["task_id"]: row for row in task_manifest["tasks"]}
    rubric_rows = {row["task_id"]: row for row in rubric["tasks"]}
    _require(
        task_rows.keys() == rubric_rows.keys(), "TaskPack and rubric task sets differ"
    )
    for task_id, task in task_rows.items():
        scored = rubric_rows[task_id]
        _require(
            scored["expected_taskpack_sha256"] == task["taskpack_sha256"],
            f"{task_id}: evaluator hash differs from frozen TaskPack",
        )
        _require(
            scored["assigned_category"] == task["assigned_family"],
            f"{task_id}: assigned category differs between manifests",
        )
        _require(
            scored["source_family"] == task["source_family"],
            f"{task_id}: source family differs between manifests",
        )
        _require(
            scored["protocol"] == task["completion_protocol"],
            f"{task_id}: completion protocol differs between manifests",
        )
    return task_rows


def _validate_benchmark_metadata(
    manifest: dict[str, Any],
    task_manifest: dict[str, Any],
    rubric: dict[str, Any],
    task_rows: dict[str, dict[str, Any]],
) -> None:
    benchmark = manifest.get("benchmark")
    _require(isinstance(benchmark, dict), "benchmark summary is missing")
    _require(
        benchmark.get("task_count") == len(task_rows) == 120,
        "benchmark task count mismatch",
    )
    _require(
        benchmark.get("assigned_categories") == task_manifest["assigned_family_counts"],
        "benchmark category summary mismatch",
    )
    _require(
        benchmark.get("taskpack_version") == "data-generate-taskpack/1.0"
        and benchmark.get("runtime_contract_version")
        == "compiled-runtime-contract/1.8",
        "benchmark contract version mismatch",
    )
    _require(
        benchmark.get("watch_interaction_openings") == 22
        and benchmark.get("bundled_label_assets") == 85,
        "benchmark bundled resource summary mismatch",
    )
    evaluation = manifest.get("evaluation")
    _require(isinstance(evaluation, dict), "evaluation summary is missing")
    _require(
        evaluation.get("criteria_count")
        == sum(row["criteria_count"] for row in rubric["tasks"])
        == 464,
        "evaluation criterion count mismatch",
    )
    _require(
        set(evaluation.get("protocols") or ()) == set(rubric["protocols"]),
        "evaluation protocol summary mismatch",
    )
    _require(
        evaluation.get("native_taskpack_and_release_scores_are_separate") is True
        and evaluation.get("trace_adjudication_must_be_complete_before_scoring") is True
        and evaluation.get("adjudication_is_bound_to_run_evidence_sha256") is True,
        "release evaluation semantics are incomplete",
    )


def _validate_simulator_metadata(manifest: dict[str, Any], root: Path) -> None:
    component_path = _release_file(root, "simulator/component_manifest.json")
    dependency_path = _release_file(root, "simulator/external_dependencies.json")
    components = _load_json(component_path)
    dependencies = _load_json(dependency_path)
    _require(isinstance(components, dict), "simulator component manifest is invalid")
    _require(isinstance(dependencies, dict), "dependency manifest is invalid")
    _require(
        components.get("schema_version") == "interactiveworld-simulator-components/1.0",
        "simulator component manifest schema mismatch",
    )
    _require(
        dependencies.get("schema_version")
        == "interactiveworld-external-dependencies/1.0",
        "external dependency manifest schema mismatch",
    )
    _validate_bundled_human_assets(dependencies, root / "simulator")
    simulator = manifest.get("simulator")
    _require(isinstance(simulator, dict), "simulator release metadata is missing")
    _require(
        simulator.get("protocol_version") == "interactiveworld-sim/1.0"
        and simulator.get("python_api") == "simulator.api:Simulator"
        and simulator.get("task_catalog") == "simulator.catalog:TaskCatalog"
        and simulator.get("process_isolated_by_default") is True
        and simulator.get("supports_task_free_worlds") is True
        and simulator.get("supports_bundled_and_custom_taskpacks") is True
        and simulator.get("task_preparation_modes") == ["prologue", "continuous_watch"]
        and simulator.get("observation_profiles") == ["agent", "debug"],
        "simulator public contract metadata is incomplete",
    )
    _validate_component_metadata(components)
    _validate_runner_metadata(manifest, dependencies, root)
    _validate_declared_tree_integrity(
        root / "simulator",
        components.get("tree_integrity"),
        {Path("component_manifest.json")},
        "simulator",
    )


def validate(
    manifest_path: Path = MANIFEST_PATH, *, check_hygiene: bool = True
) -> dict[str, Any]:
    resolved_manifest = manifest_path.resolve()
    root = resolved_manifest.parent
    manifest = _load_json(resolved_manifest)
    _require(isinstance(manifest, dict), "release manifest must be an object")
    _validate_release_identity(manifest)
    _validate_declared_tree_integrity(
        root,
        manifest.get("tree_integrity"),
        {Path("release_manifest.json")},
        "release",
    )
    _validate_frozen_artifacts(manifest, root)
    task_manifest, rubric = _load_benchmark_contract(root)
    task_rows = _validate_task_rubric_alignment(task_manifest, rubric)
    _validate_benchmark_metadata(manifest, task_manifest, rubric, task_rows)
    _validate_simulator_metadata(manifest, root)
    _validate_entrypoints(manifest, root)

    files, python_files = _validate_tree_hygiene(
        root, allow_bytecode_cache=not check_hygiene
    )
    return {
        "status": "OK",
        "release_id": manifest["release_id"],
        "release_version": manifest["release_version"],
        "tasks": len(task_rows),
        "assigned_categories": task_manifest["assigned_family_counts"],
        "files": files,
        "python_files": python_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)
    try:
        result = validate(args.manifest.resolve())
    except ReleaseValidationError as error:
        sys.stderr.write(f"release validation failed: {error}\n")
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

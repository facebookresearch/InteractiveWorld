#!/usr/bin/env python3
"""Command-line interface for exploring and driving InteractiveWorld."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO


def _activate_release() -> None:
    release_root = str(Path(__file__).resolve().parents[1])
    if release_root not in sys.path:
        sys.path.insert(0, release_root)


def _load_public_api() -> SimpleNamespace:
    """Load the lightweight API only after a command has been selected."""
    _activate_release()
    public = importlib.import_module("simulator")

    return SimpleNamespace(
        Simulator=public.Simulator,
        SimulatorConfig=public.SimulatorConfig,
        TaskCatalog=public.TaskCatalog,
        TaskSpec=public.TaskSpec,
        action_capabilities=public.action_capabilities,
        available_worlds=public.available_worlds,
    )


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(value: object, output: TextIO) -> None:
    json.dump(_json_value(value), output, ensure_ascii=False, indent=2, sort_keys=True)
    output.write("\n")


def _taskpack_mapping(task: object) -> Mapping[str, Any]:
    value = getattr(task, "taskpack", {})
    return value if isinstance(value, Mapping) else {}


def _task_summary(task: object) -> dict[str, object]:
    taskpack = _taskpack_mapping(task)
    executable = taskpack.get("executable_spec")
    executable = executable if isinstance(executable, Mapping) else {}
    meta = executable.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    source_path = getattr(task, "source_path", None)
    summary = {
        "task_id": str(getattr(task, "task_id", taskpack.get("task_id") or "")),
        "category": str(
            getattr(
                task,
                "category",
                source_path.parent.name if isinstance(source_path, Path) else "",
            )
        ),
        "source_family": str(getattr(task, "source_family", meta.get("family") or "")),
        "taskpack_sha256": str(
            getattr(task, "taskpack_sha256", None)
            or getattr(task, "source_sha256", None)
            or ""
        ),
    }
    for name in (
        "taskpack_version",
        "runtime_contract_version",
        "completion_protocol",
        "preparation_mode",
    ):
        value = getattr(task, name, None)
        if value is not None:
            summary[name] = str(value)
    scene = getattr(task, "scene", meta.get("scene"))
    if scene:
        summary["scene"] = str(scene)
    difficulty = meta.get("difficulty")
    if difficulty:
        summary["difficulty"] = str(difficulty)
    return summary


def _task_detail(task: object) -> dict[str, object]:
    detail = _task_summary(task)
    for name in (
        "source_sha256",
        "taskpack_digest",
        "builder_spec_digest",
        "source_path",
        "info",
    ):
        value = getattr(task, name, None)
        if value is not None:
            detail[name] = value
    detail["taskpack"] = getattr(task, "taskpack", {})
    detail["builder_spec"] = getattr(task, "builder_spec", {})
    return detail


def _tasks_list(args: argparse.Namespace, output: TextIO) -> int:
    public = _load_public_api()
    catalog = public.TaskCatalog.bundled()
    tasks = catalog.list(category=args.category, source_family=args.source_family)
    summaries = [_task_summary(task) for task in tasks]
    _write_json({"count": len(summaries), "tasks": summaries}, output)
    return 0


def _tasks_show(args: argparse.Namespace, output: TextIO) -> int:
    public = _load_public_api()
    task = public.TaskCatalog.bundled().get(args.task)
    _write_json(_task_detail(task), output)
    return 0


def _capabilities(args: argparse.Namespace, output: TextIO) -> int:
    public = _load_public_api()
    actions = public.action_capabilities(include_unavailable=args.all)
    _write_json({"count": len(actions), "actions": actions}, output)
    return 0


def _worlds(_args: argparse.Namespace, output: TextIO) -> int:
    public = _load_public_api()
    worlds = public.available_worlds()
    _write_json({"count": len(worlds), "worlds": worlds}, output)
    return 0


def _doctor(_args: argparse.Namespace, output: TextIO) -> int:
    _activate_release()
    from launch.check_install import check_install

    result = check_install()
    _write_json(result, output)
    return 0 if result["ok"] else 1


def _run(args: argparse.Namespace, output: TextIO) -> int:
    public = _load_public_api()
    config = public.SimulatorConfig(
        gpu=args.gpu,
        seed=args.seed,
        render_observations=args.render,
        observation_profile=args.observation_profile,
        frames_per_action=args.frames_per_action,
        render_selected_only=args.render_selected_only,
        visibility_gating=args.visibility_gating,
        visibility_checks=args.visibility_checks,
        strict_acceptance=args.strict_acceptance,
        output_dir=args.output_dir,
        isaac_python=args.isaac_python,
        isaac_env_dir=args.isaac_env_dir,
    )
    if args.scene is not None:
        launch_arguments = {"scene": args.scene}
        task_id = None
    elif args.taskpack is not None:
        task = public.TaskSpec.load(
            args.taskpack,
            expected_sha256=args.taskpack_sha256,
        )
        launch_arguments = {"task": task}
        task_id = task.task_id
    else:
        launch_arguments = {"task_id": args.task}
        task_id = args.task
    steps: list[dict[str, object]] = []
    with public.Simulator.launch(config=config, **launch_arguments) as simulator:
        preparation = simulator.prepare_task() if args.prepare_task else None
        initial_observation = (
            preparation.observation if preparation is not None else simulator.observe()
        )
        for command in args.action:
            result = simulator.step(command)
            steps.append({"action": command, "result": result})
            if result.terminated or result.truncated:
                break
        final_score = simulator.score() if task_id is not None else None
        current_scene = simulator.current_scene
    _write_json(
        {
            "task_id": task_id,
            "scene": current_scene,
            "prepared": preparation is not None,
            "preparation": (
                {
                    "action_count": len(preparation.steps),
                    "score_baseline": preparation.score_baseline,
                }
                if preparation is not None
                else None
            ),
            "initial_observation": initial_observation,
            "steps": steps,
            "score": final_score,
        },
        output,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interactiveworld-simulator",
        description="Inspect tasks or directly drive an InteractiveWorld scene.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    tasks = commands.add_parser("tasks", help="inspect bundled TaskPacks")
    task_commands = tasks.add_subparsers(dest="tasks_command", required=True)
    task_list = task_commands.add_parser("list", help="list bundled tasks")
    task_list.add_argument("--category")
    task_list.add_argument("--source-family")
    task_list.set_defaults(handler=_tasks_list)
    task_show = task_commands.add_parser("show", help="show one TaskPack")
    task_show.add_argument("task")
    task_show.set_defaults(handler=_tasks_show)

    capabilities = commands.add_parser(
        "capabilities", help="list semantic human actions"
    )
    capabilities.add_argument(
        "--all",
        action="store_true",
        help="also show actions documented but not currently executable",
    )
    capabilities.set_defaults(handler=_capabilities)

    worlds = commands.add_parser(
        "worlds", help="list physical scenes available without a TaskPack"
    )
    worlds.set_defaults(handler=_worlds)

    doctor = commands.add_parser(
        "doctor", help="check Isaac, assets, and bundled task resources"
    )
    doctor.set_defaults(handler=_doctor)

    run = commands.add_parser("run", help="open one world or task and execute actions")
    task_source = run.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--task", help="bundled task ID")
    task_source.add_argument("--taskpack", type=Path, help="custom TaskPack JSON")
    task_source.add_argument("--scene", help="physical scene without a TaskPack")
    run.add_argument(
        "--taskpack-sha256",
        help="required sha256:... file digest for --taskpack",
    )
    run.add_argument(
        "--action",
        action="append",
        default=[],
        metavar="COMMAND",
        help="semantic command; repeat to execute a sequence",
    )
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("--isaac-python", type=Path)
    run.add_argument("--isaac-env-dir", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include rendered observations",
    )
    run.add_argument(
        "--observation-profile",
        choices=("agent", "debug"),
        default="agent",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--frames-per-action", type=int, default=4)
    run.add_argument(
        "--render-selected-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument(
        "--visibility-gating",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run.add_argument(
        "--visibility-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument(
        "--strict-acceptance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument(
        "--prepare-task",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "apply a non-continuous TaskPack prologue and capture its score baseline; "
            "continuous WATCH tasks must use the benchmark runner"
        ),
    )
    run.set_defaults(handler=_run)
    return parser


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and bool(args.taskpack) != bool(args.taskpack_sha256):
        parser.error("--taskpack and --taskpack-sha256 must be provided together")
    if args.command == "run" and args.scene and args.prepare_task:
        parser.error("--prepare-task is only valid with --task or --taskpack")
    output = stdout if stdout is not None else sys.stdout
    return int(args.handler(args, output))


if __name__ == "__main__":
    raise SystemExit(main())

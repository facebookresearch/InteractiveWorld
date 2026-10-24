"""One-command public interface for running InteractiveWorld evaluations."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.evaluation.score import EvidenceError
from launch.check_install import check_install

from .config import (
    BenchmarkConfig,
    ConfigError,
    load_config_file,
    PROVIDERS,
    resolve_config,
)
from .orchestrator import (
    build_run_plan,
    execute_run,
    load_release_tasks,
    load_result_records,
    preflight_run,
    prepare_run,
    RunError,
    RunPlan,
    validate_execution_environment,
    WorkerResult,
    write_run_status,
)
from .report import (
    build_suite_report,
    build_task_reports,
    format_suite_console,
    format_task_console,
    ReportError,
    write_json_report,
)
from .scoring import apply_scored_results, score_adjudication
from .video import (
    add_video_arguments,
    build_videos,
    validate_video_environment,
    video_config_from_namespace,
    VideoError,
)

_LOG: logging.Logger = logging.getLogger(__name__)


def _add_boolean(
    parser: argparse.ArgumentParser,
    name: str,
    help_text: str,
) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}", dest=destination, action="store_true", help=help_text
    )
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(**{destination: None})


def _add_endpoint_arguments(parser: argparse.ArgumentParser, role: str) -> None:
    title = role.capitalize()
    parser.add_argument(
        f"--{role}-provider",
        choices=PROVIDERS,
        help=f"{title} API provider; openai-compatible supports vLLM/Ollama",
    )
    parser.add_argument(f"--{role}-model", help=f"{title} model identifier")
    parser.add_argument(
        f"--{role}-base-url",
        help=f"{title} API base URL or provider-neutral /generate endpoint",
    )
    parser.add_argument(
        f"--{role}-api-key-env",
        metavar="ENV_NAME",
        help=f"environment variable containing the {title} API key",
    )
    parser.add_argument(f"--{role}-timeout-s", type=float)
    parser.add_argument(f"--{role}-frame-cap", type=int)
    parser.add_argument(
        f"--{role}-tokens",
        type=int,
        help=f"maximum new {title} tokens; 0 uses the provider/runtime default",
    )


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_argument_group("task selection")
    selection.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="exact frozen task ID; repeat for multiple tasks",
    )
    selection.add_argument(
        "--category",
        dest="categories",
        action="append",
        help="one of the six assigned benchmark categories",
    )
    selection.add_argument(
        "--all",
        dest="all_tasks",
        action="store_true",
        default=None,
        help="run all 120 tasks (the default when no selector is given)",
    )
    selection.add_argument(
        "--list-tasks", action="store_true", help="print the frozen task catalog"
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    runtime = parser.add_argument_group("runtime")
    gpu_selection = runtime.add_mutually_exclusive_group()
    gpu_selection.add_argument(
        "--gpus",
        help=(
            "comma-separated physical GPU indices; CUDA_VISIBLE_DEVICES must be "
            "unset for execution"
        ),
    )
    gpu_selection.add_argument(
        "--num-gpus",
        type=int,
        help="use physical GPUs 0 through N-1",
    )
    runtime.add_argument("--seed", type=int)
    runtime.add_argument("--max-ticks", type=int)
    runtime.add_argument(
        "--frame-budget",
        type=int,
        help="total egocentric-frame budget distributed across an episode",
    )
    runtime.add_argument("--frames-per-tick", type=int)
    runtime.add_argument("--max-frames", type=int)
    runtime.add_argument("--frame-max-side", type=int)
    runtime.add_argument("--human-invalid-attempt-limit", type=int)
    runtime.add_argument("--human-http-attempt-limit", type=int)
    runtime.add_argument("--human-http-retry-backoff-s", type=float)
    runtime.add_argument("--io-capture", choices=("auto", "off"))
    _add_boolean(parser, "graceful-exit", "ask Isaac Kit to perform normal teardown")
    _add_boolean(parser, "fail-fast", "stop all GPU workers after the first run error")


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        help=(
            "run directory; default is $INTERACTIVEWORLD_RUN_ROOT/<run-id> "
            "or the system temporary directory"
        ),
    )
    output.add_argument("--run-id", help="stable run identifier")
    output.add_argument(
        "--adjudication",
        help="criterion-level evidence JSON or evidence/adjudicated directory",
    )
    _add_boolean(parser, "resume", "resume compatible successful tasks in output-dir")


def _usage_examples() -> str:
    return """examples:
  (after exporting OPENAI_API_KEY) env -u CUDA_VISIBLE_DEVICES ./launch/interactiveworld.sh --assistant-provider openai --assistant-model gpt-4.1 --human-provider openai --human-model gpt-4.1 --num-gpus 4
  env -u CUDA_VISIBLE_DEVICES ./launch/interactiveworld.sh --task-id ASK-13000 --assistant-provider openai-compatible --assistant-model local-vlm --assistant-base-url http://127.0.0.1:8000/v1 --human-provider openai-compatible --human-model local-human --human-base-url http://127.0.0.1:8001/v1 --video all
  ./launch/interactiveworld.sh --config benchmark/runner/config.example.json --dry-run

scoring:
  The first run prints the native TaskPack result immediately. The official
  release score is produced only after complete-trace adjudication:
  ./launch/interactiveworld.sh --output-dir /tmp/interactiveworld-runs/my-run --run-id my-run --resume --adjudication /tmp/interactiveworld-runs/my-run/evidence/adjudicated
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interactiveworld",
        description="Run one task or the complete frozen InteractiveWorld benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_usage_examples(),
    )
    parser.add_argument("--config", help="runner-config/1.0 JSON file")
    _add_selection_arguments(parser)
    parser.add_argument(
        "--assistant-control",
        choices=("model", "oracle-verbal", "silent"),
        help="model evaluation or a deterministic baseline",
    )
    _add_endpoint_arguments(parser, "assistant")
    _add_endpoint_arguments(parser, "human")
    _add_runtime_arguments(parser)
    _add_output_arguments(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="write and print the complete execution plan without starting Isaac",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="show worker lifecycle logs"
    )
    add_video_arguments(parser)
    parser.set_defaults(
        video=None, video_fps=None, video_resolution=None, video_captions=None
    )
    return parser


def _release_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _generated_run_id() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return "run-" + now.strftime("%Y%m%dT%H%M%SZ-%f")


def _print_json(value: Any) -> None:
    sys.stdout.write(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _list_tasks(root: Path) -> int:
    tasks = load_release_tasks(root / "benchmark" / "tasks" / "manifest.json")
    _print_json(
        {
            "task_count": len(tasks),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "category": task.assigned_category,
                    "source_family": task.source_family,
                }
                for task in tasks
            ],
        }
    )
    return 0


def _ordered_records(plan: RunPlan) -> list[dict[str, Any]]:
    by_id = load_result_records(plan.config.output_dir)
    worker_dirs = {
        task.task_id: worker.output_dir
        for worker in plan.workers
        for task in worker.tasks
    }
    rows: list[dict[str, Any]] = []
    for task in plan.selected_tasks:
        record = by_id.get(task.task_id)
        if record is not None:
            rows.append(record)
            continue
        rows.append(
            {
                "task_id": task.task_id,
                "error": "runner produced no result record",
                "artifact_dir": str(
                    worker_dirs.get(task.task_id, plan.config.output_dir)
                ),
            }
        )
    return rows


def _video_configuration(config: BenchmarkConfig) -> Any:
    return video_config_from_namespace(
        argparse.Namespace(
            video=config.video_mode,
            video_fps=config.video_fps,
            video_resolution=config.video_resolution,
            video_captions=config.video_captions,
        )
    )


def _build_reports(
    plan: RunPlan,
    worker_results: list[WorkerResult],
    scored: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _ordered_records(plan)
    _validate_adjudication_coverage(plan, records, scored)
    scored_records = apply_scored_results(records, scored)
    video_rows = build_videos(
        scored_records,
        plan.config.output_dir / "videos",
        _video_configuration(plan.config),
    )
    videos = {str(row.get("task_id") or ""): row for row in video_rows}
    reports = build_task_reports(
        records,
        scored_results=scored,
        video_results=videos,
        evidence_dir=plan.config.output_dir / "evidence" / "pending",
    )
    suite = build_suite_report(reports)
    suite["run_id"] = plan.config.run_id
    suite["run_manifest"] = str(plan.manifest_path)
    suite["worker_results_scope"] = "current_attempt"
    suite["worker_results"] = [worker.public_dict() for worker in worker_results]
    reports_dir = plan.config.output_dir / "reports"
    for report in reports:
        write_json_report(report, reports_dir / f"{report['task_id']}.json")
    write_json_report(suite, plan.config.output_dir / "report.json")
    return suite, reports


def _validate_adjudication_coverage(
    plan: RunPlan,
    records: list[dict[str, Any]],
    scored: dict[str, dict[str, Any]],
) -> None:
    if plan.config.adjudication is None:
        return
    selected = _validate_adjudication_selection(plan, scored)
    completed = {
        str(record.get("task_id"))
        for record in records
        if record.get("task_id") in selected and not record.get("error")
    }
    _require_scored_adjudication(completed, scored)


def _require_scored_adjudication(
    completed: set[str], scored: dict[str, dict[str, Any]]
) -> None:
    missing = sorted(completed - set(scored))
    if missing:
        raise RunError(
            "adjudication is missing completed task(s): " + ", ".join(missing)
        )
    non_scored = []
    for task_id in sorted(completed):
        result = scored[task_id]
        if result.get("status") == "SCORED":
            continue
        reasons = result.get("reasons")
        detail = "; ".join(str(reason) for reason in reasons or ())
        non_scored.append(
            f"{task_id}={result.get('status') or 'UNKNOWN'}"
            + (f" ({detail})" if detail else "")
        )
    if non_scored:
        raise RunError(
            "adjudication did not produce a final score: " + ", ".join(non_scored)
        )


def _validate_adjudication_selection(
    plan: RunPlan, scored: dict[str, dict[str, Any]]
) -> set[str]:
    selected = {task.task_id for task in plan.selected_tasks}
    if plan.config.adjudication is None:
        return selected
    unexpected = sorted(set(scored) - selected)
    if unexpected:
        raise RunError(
            "adjudication contains unselected task(s): " + ", ".join(unexpected)
        )
    return selected


def _console_report(
    plan: RunPlan, suite: dict[str, Any], reports: list[dict[str, Any]]
) -> None:
    if len(reports) == 1:
        body = format_task_console(reports[0])
    else:
        body = format_suite_console(suite)
    sys.stdout.write(body + "\n")
    sys.stdout.write(f"Report: {plan.config.output_dir / 'report.json'}\n")


def _failed_run(
    suite: dict[str, Any], worker_results: list[WorkerResult], video_mode: str
) -> bool:
    if suite.get("status") == "ERROR":
        return True
    if any(worker.return_code != 0 for worker in worker_results):
        return True
    counts = suite.get("run_status_counts")
    if isinstance(counts, dict) and int(counts.get("ERROR", 0)) > 0:
        return True
    if video_mode == "off":
        return False
    tasks = suite.get("tasks")
    if not isinstance(tasks, list):
        return True
    return any(
        isinstance(task, dict)
        and isinstance(task.get("video"), dict)
        and task["video"].get("status") in {"ERROR", "UNAVAILABLE"}
        for task in tasks
    )


def _run(args: argparse.Namespace) -> int:
    root = _release_root()
    if args.list_tasks:
        return _list_tasks(root)
    file_config = load_config_file(args.config)
    config = resolve_config(args, file_config, _generated_run_id())
    plan = build_run_plan(root, config)
    preflight_run(plan)
    scored = score_adjudication(config.adjudication)
    _validate_adjudication_selection(plan, scored)
    if config.adjudication is not None:
        _require_scored_adjudication(set(plan.completed_task_ids), scored)
    video_config = _video_configuration(config)
    if not config.dry_run:
        validate_video_environment(video_config)
    if not config.dry_run and plan.workers:
        provenance = validate_execution_environment(config)
        plan = replace(plan, runtime_provenance=provenance)
        _validate_install()
    prepare_run(plan)
    if config.dry_run:
        write_run_status(plan, "DRY_RUN")
        _print_json(plan.public_dict(status="DRY_RUN"))
        return 0
    return _execute_prepared(plan, scored)


def _execute_prepared(plan: RunPlan, scored: dict[str, dict[str, Any]]) -> int:
    write_run_status(plan, "RUNNING")
    workers: list[WorkerResult] = []
    try:
        workers, stopped_early = execute_run(plan)
        suite, reports = _build_reports(plan, workers, scored)
    except KeyboardInterrupt:
        _record_terminal_status(plan, "INTERRUPTED", workers)
        raise
    except Exception:  # noqa: BLE001 - preserve lifecycle status for every run failure
        _record_terminal_status(plan, "FAILED", workers)
        raise
    failed = stopped_early or _failed_run(suite, workers, plan.config.video_mode)
    write_run_status(plan, "FAILED" if failed else "COMPLETED", workers)
    _console_report(plan, suite, reports)
    return 1 if failed else 0


def _record_terminal_status(
    plan: RunPlan, status: str, workers: list[WorkerResult]
) -> None:
    try:
        write_run_status(plan, status, workers)
    except (OSError, RunError) as exc:
        _LOG.error("could not persist %s run status: %s", status, exc)


def _validate_install() -> None:
    result = check_install()
    if result.get("ok") is True:
        return
    checks = result.get("checks")
    failures = [
        str(check.get("name") or "unknown dependency")
        for check in checks or ()
        if isinstance(check, dict) and check.get("ok") is not True
    ]
    detail = ", ".join(failures) if failures else "unknown installation error"
    raise RunError(
        f"installation preflight failed: {detail}; run launch/check_install.py --strict"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        return _run(args)
    except (
        ConfigError,
        RunError,
        ReportError,
        VideoError,
        EvidenceError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

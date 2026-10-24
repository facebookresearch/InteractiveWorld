from __future__ import annotations

from typing import Any, Mapping

from .artifact_bundle import is_run_evidence_sha256
from .rubric import load_rubric, task_index, validate_rubric

EVIDENCE_SCHEMA_VERSION = "interactiveworld-evidence/1.1"


class PreparationError(ValueError):
    pass


def prepare_evidence(
    run_record: Mapping[str, Any],
    rubric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a criterion-keyed evidence template from one runner artifact.

    Fields that the runner proves are copied. Fields needing the revised
    protocol evaluator are emitted as null and must be adjudicated before
    ``score_evidence`` will return a score.
    """
    if rubric is None:
        rubric = load_rubric()
    else:
        validate_rubric(rubric)
    task_id = _task_id(run_record)
    task = task_index(rubric).get(task_id)
    if task is None:
        raise PreparationError(
            f"runner artifact task is not in Balanced120: {task_id!r}"
        )
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "task_id": task_id,
        "taskpack_sha256": _taskpack_hash(run_record),
        "run_evidence_sha256": _run_evidence_hash(run_record),
        "validity": _validity_template(run_record, rubric),
        "native_taskpack": _native_taskpack(run_record),
        "release": {
            "rubric_version": task["rubric_version"],
            "evidence_mode": "trace_adjudicated",
            "adjudication_required": True,
            "adjudication_status": "PENDING",
            **_release_template(task["score_spec"]),
        },
    }


def _task_id(record: Mapping[str, Any]) -> str:
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        task_goal = record.get("task_goal")
        task_id = task_goal.get("task") if isinstance(task_goal, Mapping) else None
    if not isinstance(task_id, str) or not task_id:
        raise PreparationError("runner artifact has no task_id")
    return task_id


def _taskpack_hash(record: Mapping[str, Any]) -> str | None:
    for name in ("taskpack_file_sha256", "taskpack_sha256"):
        value = record.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _run_evidence_hash(record: Mapping[str, Any]) -> str:
    value = record.get("run_evidence_sha256")
    if not is_run_evidence_sha256(value):
        raise PreparationError(
            "runner artifact has no valid run_evidence_sha256 content binding"
        )
    return value


def _validity_template(
    record: Mapping[str, Any], rubric: Mapping[str, Any]
) -> dict[str, bool | None]:
    explicit = record.get("validity_evidence")
    if isinstance(explicit, Mapping):
        return {
            name: explicit.get(name) if type(explicit.get(name)) is bool else None
            for name in rubric["validity_policy"]["required_checks"]
        }
    failed_runtime = bool(record.get("error"))
    values: dict[str, bool | None] = dict.fromkeys(
        rubric["validity_policy"]["required_checks"], None
    )
    if failed_runtime:
        values["runtime_complete"] = False
        return values
    if record.get("build") == "OK":
        values["runtime_complete"] = True
    if type(record.get("episode_valid")) is bool:
        values["scene_setup_ok"] = record["episode_valid"]
        values["frames_usable"] = record["episode_valid"]
    if record.get("eval_valid") is True:
        values["assistant_transport_ok"] = True
        values["human_executor_ok"] = True
        values["no_material_human_leakage"] = True
    if type(record.get("benchmark_usable")) is bool:
        values["simulator_ok"] = record["benchmark_usable"]
    return values


def _native_taskpack(record: Mapping[str, Any]) -> dict[str, bool | None]:
    native = record.get("native_taskpack_outcome")
    if not isinstance(native, Mapping):
        native = record.get("completion_contract")
    if not isinstance(native, Mapping):
        native = record.get("task_goal")
    if not isinstance(native, Mapping):
        native = {}
    aliases = {
        "contract_world_success": ("contract_world_success", "physical_met"),
        "process_requirements_met": ("process_requirements_met",),
        "verification_points_met": ("verification_points_met",),
        "assistant_declared_done": ("assistant_declared_done",),
        "requires_assistant_done": (
            "requires_assistant_done",
            "score_requires_assistant_done",
        ),
    }
    return {
        name: _first_boolean(native, candidates) for name, candidates in aliases.items()
    }


def _first_boolean(value: Mapping[str, Any], names: tuple[str, ...]) -> bool | None:
    for name in names:
        if type(value.get(name)) is bool:
            return value[name]
    return None


def _criterion_template(criteria: list[Mapping[str, Any]]) -> dict[str, None]:
    return dict.fromkeys((str(criterion["id"]) for criterion in criteria), None)


def _release_template(spec: Mapping[str, Any]) -> dict[str, Any]:
    kind = spec["kind"]
    if kind == "weighted_groups":
        return {
            "group_results": _criterion_template(spec["groups"]),
            "gate_results": _criterion_template(spec["gates"]),
        }
    if kind == "state_bundles":
        return {
            "bundle_results": _criterion_template(spec["bundles"]),
            "gate_results": _criterion_template(spec["gates"]),
        }
    if kind == "cfss_groups":
        return {
            "terminal_group_results": _criterion_template(spec["terminal_groups"]),
            "transition_gate_results": _criterion_template(spec["transition_gates"]),
        }
    if kind == "terminal_atoms":
        return {"atom_results": _criterion_template(spec["atoms"])}
    if kind == "state_conjunction":
        return {"condition_results": _criterion_template(spec["conditions"])}
    if kind == "causal_checks":
        return {"causal_checks": _criterion_template(spec["checks"])}
    raise PreparationError(f"unsupported score_spec kind: {kind}")

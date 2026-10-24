from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .artifact_bundle import is_run_evidence_sha256
from .prepare import EVIDENCE_SCHEMA_VERSION
from .rubric import load_rubric, task_index, validate_rubric

EVIDENCE_MODE = "trace_adjudicated"
_ROOT_FIELDS = {
    "evidence_schema_version",
    "task_id",
    "taskpack_sha256",
    "run_evidence_sha256",
    "validity",
    "native_taskpack",
    "release",
}
_NATIVE_FIELDS = {
    "contract_world_success",
    "process_requirements_met",
    "verification_points_met",
    "assistant_declared_done",
    "requires_assistant_done",
}
_RELEASE_COMMON_FIELDS = {
    "rubric_version",
    "evidence_mode",
    "adjudication_required",
    "adjudication_status",
}
_PROTOCOL_RESULT_FIELDS = {
    "StateScore-v2": ("group_results", "gate_results"),
    "StateBundle-v1": ("bundle_results", "gate_results"),
    "CFSS-StateScore": ("terminal_group_results", "transition_gate_results"),
    "ASK-PCS": ("atom_results",),
    "WATCH-binary": ("condition_results",),
    "FIND-C-pass-binary": ("causal_checks",),
}
_RESULT_CRITERIA_FIELDS = {
    "group_results": "groups",
    "gate_results": "gates",
    "bundle_results": "bundles",
    "terminal_group_results": "terminal_groups",
    "transition_gate_results": "transition_gates",
    "atom_results": "atoms",
    "condition_results": "conditions",
    "causal_checks": "checks",
}


class EvidenceError(ValueError):
    pass


def score_evidence(
    evidence: Mapping[str, Any], rubric: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    active_rubric = _active_rubric(rubric)
    task_id, task, run_evidence_sha256 = _validated_evidence(evidence, active_rubric)
    base = _result_base(task_id, task, run_evidence_sha256)
    if evidence.get("taskpack_sha256") != task["expected_taskpack_sha256"]:
        return _invalid_identity_result(base, task)
    validity = _audit_validity(evidence.get("validity"), active_rubric)
    if validity["status"] != "VALID":
        return _invalid_validity_result(base, task, validity)
    native = _score_native(evidence.get("native_taskpack"))
    base["native_taskpack"] = native
    if native.get("status") != "SCORED":
        return _invalid_native_result(base, task, native)
    return _release_result(base, task, evidence["release"])


def _active_rubric(
    rubric: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if rubric is None:
        return load_rubric()
    validate_rubric(rubric)
    return rubric


def _validated_evidence(
    evidence: Mapping[str, Any], rubric: Mapping[str, Any]
) -> tuple[str, Mapping[str, Any], str]:
    if not isinstance(evidence, Mapping):
        raise EvidenceError("evidence must be a JSON object")
    if evidence.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceError(
            f"evidence_schema_version must be {EVIDENCE_SCHEMA_VERSION!r}"
        )
    run_evidence_sha256 = evidence.get("run_evidence_sha256")
    if not is_run_evidence_sha256(run_evidence_sha256):
        raise EvidenceError(
            "run_evidence_sha256 must be a normalized sha256 content binding"
        )
    task_id = evidence.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise EvidenceError("task_id must be a non-empty string")
    task = task_index(rubric).get(task_id)
    if task is None:
        raise EvidenceError(f"unknown task_id: {task_id!r}")
    _validate_evidence_shape(evidence, task, rubric)
    return task_id, task, str(run_evidence_sha256)


def _result_base(
    task_id: str, task: Mapping[str, Any], run_evidence_sha256: str
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "run_evidence_sha256": run_evidence_sha256,
        "source_family": task["source_family"],
        "assigned_category": task["assigned_category"],
        "protocol": task["protocol"],
    }


def _invalid_identity_result(
    base: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    reason = "TaskPack hash is missing or does not match the frozen task identity"
    return {
        **base,
        "status": "VOID",
        "reasons": [reason],
        "native_taskpack": _unscored_native("invalid TaskPack identity", "VOID"),
        "release": _unscored_release(task, "invalid TaskPack identity"),
    }


def _invalid_validity_result(
    base: Mapping[str, Any],
    task: Mapping[str, Any],
    validity: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(validity["status"])
    return {
        **base,
        "status": status,
        "reasons": validity["reasons"],
        "native_taskpack": _unscored_native("run validity not established", status),
        "release": _unscored_release(task, "run validity not established"),
    }


def _invalid_native_result(
    base: Mapping[str, Any],
    task: Mapping[str, Any],
    native: Mapping[str, Any],
) -> dict[str, Any]:
    reason = str(
        native.get("reason")
        or "native TaskPack outcome is missing, incomplete, or inconsistent"
    )
    return {
        **base,
        "status": "UNSCORABLE",
        "reasons": [reason],
        "release": _unscored_release(task, reason),
    }


def _release_result(
    base: Mapping[str, Any], task: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        release = _score_release(task, evidence)
    except EvidenceError as exc:
        return {
            **base,
            "status": "UNSCORABLE",
            "reasons": [str(exc)],
            "release": _unscored_release(task, str(exc)),
        }
    return {**base, "status": "SCORED", "reasons": [], "release": release}


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if any(not isinstance(name, str) for name in value):
        raise EvidenceError(f"{label} field names must be strings")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise EvidenceError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise EvidenceError(f"{label} has unknown fields: {', '.join(unknown)}")


def _validate_evidence_shape(
    evidence: Mapping[str, Any],
    task: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> None:
    _exact_fields(evidence, _ROOT_FIELDS, "evidence")
    task_id = evidence.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise EvidenceError("task_id must be a non-empty string")
    taskpack_sha256 = evidence.get("taskpack_sha256")
    if taskpack_sha256 is not None and not is_run_evidence_sha256(taskpack_sha256):
        raise EvidenceError("taskpack_sha256 must be null or a normalized sha256")
    _validate_validity_shape(evidence.get("validity"), rubric)
    _validate_native_shape(evidence.get("native_taskpack"))
    _validate_release_shape(evidence.get("release"), task)


def _validate_validity_shape(value: Any, rubric: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise EvidenceError("validity must be an object")
    expected = set(rubric["validity_policy"]["required_checks"])
    _exact_fields(value, expected, "validity")
    invalid = sorted(
        name for name in expected if type(value.get(name)) not in (bool, type(None))
    )
    if invalid:
        raise EvidenceError(
            "validity fields must be boolean or null: " + ", ".join(invalid)
        )


def _validate_native_shape(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise EvidenceError("native_taskpack must be an object")
    if any(not isinstance(name, str) for name in value):
        raise EvidenceError("native_taskpack field names must be strings")
    allowed = _NATIVE_FIELDS | {"reported_goal_met"}
    actual = set(value)
    missing = sorted(_NATIVE_FIELDS - actual)
    unknown = sorted(actual - allowed)
    if missing:
        raise EvidenceError("native_taskpack is missing fields: " + ", ".join(missing))
    if unknown:
        raise EvidenceError("native_taskpack has unknown fields: " + ", ".join(unknown))
    invalid = sorted(
        name for name, item in value.items() if type(item) not in (bool, type(None))
    )
    if invalid:
        raise EvidenceError(
            "native_taskpack fields must be boolean or null: " + ", ".join(invalid)
        )


def _validate_release_shape(value: Any, task: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise EvidenceError("release must be an object")
    protocol = str(task["protocol"])
    result_fields = _PROTOCOL_RESULT_FIELDS.get(protocol)
    if result_fields is None:
        raise EvidenceError(f"unsupported protocol: {protocol}")
    expected = _RELEASE_COMMON_FIELDS | set(result_fields)
    _exact_fields(value, expected, "release")
    _validate_release_common(value)
    spec = task["score_spec"]
    for field in result_fields:
        criteria_name = _RESULT_CRITERIA_FIELDS[field]
        _validate_result_map(value.get(field), spec[criteria_name], field)


def _validate_release_common(value: Mapping[str, Any]) -> None:
    if not isinstance(value.get("rubric_version"), str):
        raise EvidenceError("release rubric_version must be a string")
    if value.get("evidence_mode") != EVIDENCE_MODE:
        raise EvidenceError("evidence_mode must be trace_adjudicated")
    if value.get("adjudication_required") is not True:
        raise EvidenceError("adjudication_required must be true")
    if value.get("adjudication_status") not in {"PENDING", "COMPLETE"}:
        raise EvidenceError("adjudication_status must be PENDING or COMPLETE")


def _validate_result_map(value: Any, criteria: Sequence[object], name: str) -> None:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{name} must be a criterion-ID object")
    criterion_ids = set(_criterion_ids(criteria))
    _exact_fields(value, criterion_ids, name)
    for criterion_id, item in value.items():
        if name == "group_results":
            valid = item is None or (
                type(item) in (int, float)
                and math.isfinite(float(item))
                and 0 <= item <= 1
            )
        else:
            valid = item is None or type(item) is bool
        if not valid:
            expected = (
                "numeric in [0, 1] or null"
                if name == "group_results"
                else "boolean or null"
            )
            raise EvidenceError(f"{name}.{criterion_id} must be {expected}")


def _audit_validity(value: Any, rubric: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "UNSCORABLE", "reasons": ["validity evidence is missing"]}
    required = rubric["validity_policy"]["required_checks"]
    failed = [name for name in required if value.get(name) is False]
    if failed:
        return {
            "status": "VOID",
            "reasons": ["failed validity checks: " + ", ".join(failed)],
        }
    missing = [name for name in required if type(value.get(name)) is not bool]
    if missing:
        return {
            "status": "UNSCORABLE",
            "reasons": ["missing/non-boolean validity checks: " + ", ".join(missing)],
        }
    return {"status": "VALID", "reasons": []}


def _score_native(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "UNAVAILABLE", "strict_success": None}
    if not isinstance(value, Mapping):
        return {
            "status": "UNSCORABLE",
            "strict_success": None,
            "reason": "native_taskpack must be an object",
        }
    names = [
        "contract_world_success",
        "process_requirements_met",
        "verification_points_met",
        "assistant_declared_done",
        "requires_assistant_done",
    ]
    missing = [name for name in names if type(value.get(name)) is not bool]
    if missing:
        return {
            "status": "UNSCORABLE",
            "strict_success": None,
            "reason": "missing/non-boolean native fields: " + ", ".join(missing),
        }
    success = (
        value["contract_world_success"]
        and value["process_requirements_met"]
        and value["verification_points_met"]
        and (value["assistant_declared_done"] or not value["requires_assistant_done"])
    )
    if "reported_goal_met" in value and type(value["reported_goal_met"]) is not bool:
        return {
            "status": "UNSCORABLE",
            "strict_success": None,
            "reason": "reported_goal_met must be boolean",
        }
    if "reported_goal_met" in value and value["reported_goal_met"] != success:
        return {
            "status": "INCONSISTENT",
            "strict_success": success,
            "reported_goal_met": value["reported_goal_met"],
        }
    return {"status": "SCORED", "strict_success": success}


def _unscored_native(reason: str, status: str) -> dict[str, Any]:
    return {"status": status, "strict_success": None, "reason": reason}


def _score_release(  # noqa: C901
    task: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    if evidence.get("rubric_version") != task["rubric_version"]:
        raise EvidenceError("release rubric_version is missing or mismatched")
    mode = evidence.get("evidence_mode")
    if mode != EVIDENCE_MODE:
        raise EvidenceError("evidence_mode must be trace_adjudicated")
    if evidence.get("adjudication_required") is not True:
        raise EvidenceError(
            "trace_adjudicated evidence must set adjudication_required to true"
        )
    if evidence.get("adjudication_status") != "COMPLETE":
        raise EvidenceError(
            "trace_adjudicated evidence remains pending until "
            "adjudication_status is COMPLETE"
        )
    protocol = task["protocol"]
    spec = task["score_spec"]
    extra: dict[str, Any] = {}

    if protocol == "StateScore-v2":
        groups = _number_results(
            evidence.get("group_results"), spec["groups"], "group_results"
        )
        gates = _gate_results(evidence, spec["gates"])
        completion = sum(
            group["weight"] * result for group, result in zip(spec["groups"], groups)
        )
        strict = all(result == 1.0 for result in groups) and gates
    elif protocol == "StateBundle-v1":
        bundles = _bool_results(
            evidence.get("bundle_results"), spec["bundles"], "bundle_results"
        )
        gates = _gate_results(evidence, spec["gates"])
        active = sum(bundles) / len(bundles)
        completion = active if gates else 0.75 * active
        strict = all(bundles) and gates
        extra["active_result_fraction"] = active
    elif protocol == "CFSS-StateScore":
        groups = _bool_results(
            evidence.get("terminal_group_results"),
            spec["terminal_groups"],
            "terminal_group_results",
        )
        completion = sum(groups) / len(groups)
        strict = all(groups)
        transition_gates = spec["transition_gates"]
        if transition_gates:
            transition = evidence.get("transition_gate_results")
            if _optional_bool_results(transition, transition_gates):
                transition = _bool_results(
                    transition,
                    transition_gates,
                    "transition_gate_results",
                )
                extra["stg_status"] = "SCORED"
                extra["stg_strict_pass"] = strict and all(transition)
            else:
                extra["stg_status"] = "UNSCORABLE"
                extra["stg_strict_pass"] = None
        else:
            extra["stg_status"] = "NOT_APPLICABLE"
            extra["stg_strict_pass"] = strict
    elif protocol == "ASK-PCS":
        atoms = _bool_results(
            evidence.get("atom_results"), spec["atoms"], "atom_results"
        )
        completion = sum(atoms) / len(atoms)
        strict = all(atoms)
    elif protocol == "WATCH-binary":
        conditions = _bool_results(
            evidence.get("condition_results"),
            spec["conditions"],
            "condition_results",
        )
        strict = all(conditions)
        completion = float(strict)
    elif protocol == "FIND-C-pass-binary":
        values = evidence.get("causal_checks")
        if not isinstance(values, Mapping):
            raise EvidenceError("causal_checks must be an object")
        required = {check["id"] for check in spec["checks"]}
        missing = sorted(
            name for name in required if type(values.get(name)) is not bool
        )
        unknown = sorted(set(values) - required)
        if missing:
            raise EvidenceError(
                "missing/non-boolean causal checks: " + ", ".join(missing)
            )
        if unknown:
            raise EvidenceError("unknown causal checks: " + ", ".join(unknown))
        strict = all(values[name] for name in required)
        completion = float(strict)
    else:
        raise EvidenceError(f"unsupported protocol: {protocol}")

    completion = min(1.0, max(0.0, float(completion)))
    return {
        "status": "SCORED",
        "rubric_version": task["rubric_version"],
        "evidence_mode": mode,
        "adjudication_required": True,
        "adjudication_status": "COMPLETE",
        "strict_pass": bool(strict),
        "completion_score": completion,
        **extra,
    }


def _gate_results(evidence: Mapping[str, Any], gates: Sequence[object]) -> bool:
    values = _bool_results(evidence.get("gate_results"), gates, "gate_results")
    return all(values)


def _criterion_ids(criteria: Sequence[object]) -> list[str]:
    return [
        str(criterion["id"]) for criterion in criteria if isinstance(criterion, Mapping)
    ]


def _ordered_results(value: Any, criteria: Sequence[object], name: str) -> list[Any]:
    criterion_ids = _criterion_ids(criteria)
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{name} must be a criterion-ID object")
    missing = sorted(set(criterion_ids) - set(value))
    unknown = sorted(set(value) - set(criterion_ids))
    if missing:
        raise EvidenceError(f"{name} is missing criterion IDs: {', '.join(missing)}")
    if unknown:
        raise EvidenceError(f"{name} has unknown criterion IDs: {', '.join(unknown)}")
    return [value[criterion_id] for criterion_id in criterion_ids]


def _bool_results(value: Any, criteria: Sequence[object], name: str) -> list[bool]:
    values = _ordered_results(value, criteria, name)
    if any(type(item) is not bool for item in values):
        raise EvidenceError(f"every {name} value must be boolean")
    return values


def _optional_bool_results(value: Any, criteria: Sequence[object]) -> bool:
    try:
        _bool_results(value, criteria, "transition_gate_results")
    except EvidenceError:
        return False
    return True


def _number_results(value: Any, criteria: Sequence[object], name: str) -> list[float]:
    values = _ordered_results(value, criteria, name)
    if any(
        type(item) not in (int, float)
        or not math.isfinite(float(item))
        or not 0 <= item <= 1
        for item in values
    ):
        raise EvidenceError(f"every {name} value must be numeric in [0, 1]")
    return [float(item) for item in values]


def _unscored_release(task: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "status": "UNSCORABLE",
        "rubric_version": task["rubric_version"],
        "strict_pass": None,
        "completion_score": None,
        "reason": reason,
    }


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    rubric: Mapping[str, Any] | None = None,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    if rubric is None:
        rubric = load_rubric()
    else:
        validate_rubric(rubric)
    expected = task_index(rubric)
    by_id: dict[str, Mapping[str, Any]] = {}
    for result in results:
        task_id = result.get("task_id")
        if task_id in by_id:
            raise EvidenceError(f"duplicate result for {task_id}")
        if task_id not in expected:
            raise EvidenceError(f"unknown task_id in results: {task_id!r}")
        by_id[task_id] = result
    missing = sorted(set(expected) - set(by_id))
    non_scored = sorted(
        task_id for task_id, result in by_id.items() if result.get("status") != "SCORED"
    )
    complete = not missing and not non_scored and len(by_id) == 120
    if require_complete and not complete:
        return {
            "status": "UNSCORABLE",
            "complete": False,
            "missing_task_ids": missing,
            "non_scored_task_ids": non_scored,
        }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in by_id.values():
        if result.get("status") == "SCORED":
            grouped[result["assigned_category"]].append(result)
    categories = {}
    for category in rubric["assigned_category_counts"]:
        rows = grouped.get(category, [])
        categories[category] = {
            "scored_tasks": len(rows),
            "strict_pass_rate": (
                sum(row["release"]["strict_pass"] for row in rows) / len(rows)
                if rows
                else None
            ),
            "mean_completion_rate": (
                sum(row["release"]["completion_score"] for row in rows) / len(rows)
                if rows
                else None
            ),
        }
    scored = [row for rows in grouped.values() for row in rows]
    trace_adjudicated = sum(
        row["release"].get("evidence_mode") == "trace_adjudicated" for row in scored
    )
    return {
        "status": "SCORED" if complete else "PARTIAL",
        "complete": complete,
        "scored_tasks": len(scored),
        "trace_adjudicated_tasks": trace_adjudicated,
        "status_counts": dict(
            Counter(result.get("status") for result in by_id.values())
        ),
        "strict_pass_rate": (
            sum(row["release"]["strict_pass"] for row in scored) / len(scored)
            if scored
            else None
        ),
        "mean_completion_rate": (
            sum(row["release"]["completion_score"] for row in scored) / len(scored)
            if scored
            else None
        ),
        "categories": categories,
        "missing_task_ids": missing,
        "non_scored_task_ids": non_scored,
    }

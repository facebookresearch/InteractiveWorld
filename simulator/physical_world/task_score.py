"""Shared task scoring for benchmark manifests and the persistent RL environment.

The physical relation set, positive predicates, and negative constraints must reduce to one verdict.
This module keeps that reduction Isaac-free and reuses the benchmark's existing negative-constraint and
achievement/prohibition semantics instead of maintaining a second training-only interpretation.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import constraint_evaluator
import isaac_goal


Verdict = dict[str, object]


@dataclass(frozen=True)
class ScoreRecord:
    """One canonical reduction of positive goals and negative constraints."""

    success: bool
    scoreable: bool
    n_done: int
    n_total: int
    progress: float
    n_positive_done: int
    n_positive_total: int
    progress_positive: float
    n_sub_goals: int
    n_sub_goals_ok: int
    sub_goal_scoring: str
    n_achievements: int
    n_achievements_ok: int
    progress_achievements: float
    n_prohibitions: int
    n_prohibitions_ok: int
    prohibitions_ok: bool
    progress_sub_goals: float
    goals: tuple[Verdict, ...]
    aggregates: tuple[Verdict, ...]
    outcomes: tuple[Verdict, ...]
    neg: tuple[Verdict, ...]
    sub_goals: tuple[Verdict, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-ready canonical score used by the RL environment."""
        return {
            "success": self.success,
            "scoreable": self.scoreable,
            "n_done": self.n_done,
            "n_total": self.n_total,
            "progress": self.progress,
            "progress_denominator": "achievements_gated_by_prohibitions",
            "n_positive_done": self.n_positive_done,
            "n_positive_total": self.n_positive_total,
            "progress_positive": self.progress_positive,
            **self._sub_goal_fields(),
            "goals": [dict(r) for r in self.goals],
            "aggregates": [dict(r) for r in self.aggregates],
            "outcomes": [dict(r) for r in self.outcomes],
            "neg": [dict(r) for r in self.neg],
            "sub_goals": [dict(r) for r in self.sub_goals],
        }

    def to_manifest_dict(self) -> dict[str, object]:
        """Benchmark-compatible fields backed by this same canonical record."""
        out: dict[str, object] = {
            "met": self.success,
            "progress": self.progress_positive,
            "n": self.n_positive_total,
            "progress_denominator": "positive_goals",
            "n_sub_goals": self.n_sub_goals,
            "n_sub_goals_ok": self.n_sub_goals_ok,
            **self._sub_goal_fields(),
            "goals": [dict(r) for r in self.goals],
            "aggregates": [dict(r) for r in self.aggregates],
            "outcomes": [dict(r) for r in self.outcomes],
            "sub_goals": [dict(r) for r in self.sub_goals],
        }
        if self.neg:
            out["neg"] = [dict(r) for r in self.neg]
            out["neg_ok"] = self.prohibitions_ok and all(
                bool(r["ok"]) for r in self.neg
            )
        return out

    def _sub_goal_fields(self) -> dict[str, object]:
        return {
            "sub_goal_scoring": self.sub_goal_scoring,
            "n_achievements": self.n_achievements,
            "n_achievements_ok": self.n_achievements_ok,
            "progress_achievements": self.progress_achievements,
            "n_prohibitions": self.n_prohibitions,
            "n_prohibitions_ok": self.n_prohibitions_ok,
            "prohibitions_ok": self.prohibitions_ok,
            "progress_sub_goals": self.progress_sub_goals,
        }


def score_task(
    goals: Sequence[isaac_goal.Predicate],
    neg_constraints: Sequence[Mapping[str, object]],
    relations: set[tuple],
    trace_outcomes: Sequence[Mapping[str, object]] = (),
    trace_state: Mapping[str, object] | None = None,
    aggregate_goals: Sequence[Mapping[str, object]] = (),
    baseline_goal_indices: Collection[int] = (),
) -> ScoreRecord:
    """Score one state, excluding baseline-satisfied goals from earned progress."""
    positive = isaac_goal.goal_report(list(goals), relations)
    baseline = set(baseline_goal_indices)
    for index, row in enumerate(positive):
        if index in baseline:
            row["baseline_excluded"] = True
    aggregates = _aggregate_goal_report(aggregate_goals, relations)
    outcomes = _trace_outcome_report(trace_outcomes, trace_state or {})
    neg_input = [dict(c) for c in neg_constraints]
    negative = constraint_evaluator.eval_neg_constraints(neg_input, relations)
    sub_goals = _join_sub_goals(positive, aggregates, outcomes, negative)
    split = constraint_evaluator.score_sub_goals(sub_goals)
    gate = bool(split["prohibitions_ok"])
    achievement_progress = [
        float(row.get("progress", 1.0 if row.get("ok") else 0.0))
        for row in sub_goals
        if constraint_evaluator.sub_goal_class(row) == constraint_evaluator.ACHIEVEMENT
    ]
    scoreable = bool(achievement_progress)
    progress_achievements = (
        sum(achievement_progress) / len(achievement_progress)
        if achievement_progress
        else 1.0
    )
    progress = progress_achievements if scoreable and gate else 0.0
    n_done = int(split["n_achievements_ok"]) if scoreable and gate else 0
    positive_rows = [
        row
        for row in [*positive, *aggregates, *outcomes]
        if constraint_evaluator.sub_goal_class(row) == constraint_evaluator.ACHIEVEMENT
    ]
    positive_progress = [
        float(row.get("progress", 1.0 if row.get("ok") else 0.0))
        for row in positive_rows
    ]
    return ScoreRecord(
        success=scoreable and all(bool(r["ok"]) for r in sub_goals),
        scoreable=scoreable,
        n_done=n_done,
        n_total=int(split["n_achievements"]),
        progress=progress,
        n_positive_done=sum(1 for r in positive_rows if r["ok"]),
        n_positive_total=len(positive_rows),
        progress_positive=(sum(positive_progress) / len(positive_progress))
        if positive_rows
        else 1.0,
        n_sub_goals=len(sub_goals),
        n_sub_goals_ok=sum(1 for r in sub_goals if r["ok"]),
        sub_goal_scoring=str(split["sub_goal_scoring"]),
        n_achievements=int(split["n_achievements"]),
        n_achievements_ok=int(split["n_achievements_ok"]),
        progress_achievements=progress_achievements,
        n_prohibitions=int(split["n_prohibitions"]),
        n_prohibitions_ok=int(split["n_prohibitions_ok"]),
        prohibitions_ok=gate,
        progress_sub_goals=progress,
        goals=tuple(dict(r) for r in positive),
        aggregates=tuple(dict(r) for r in aggregates),
        outcomes=tuple(dict(r) for r in outcomes),
        neg=tuple(dict(r) for r in negative),
        sub_goals=tuple(dict(r) for r in sub_goals),
    )


def satisfied_goal_indices(
    goals: Sequence[isaac_goal.Predicate], relations: set[tuple]
) -> frozenset[int]:
    """Return goal positions already satisfied at an episode's scoring baseline."""
    return frozenset(
        index
        for index, goal in enumerate(goals)
        if isaac_goal.satisfied(goal, relations)
    )


def _join_sub_goals(
    positive: list[dict],
    aggregates: list[dict],
    outcomes: list[dict],
    negative: list[dict],
) -> list[dict]:
    achievements = [
        {**dict(row), "idx": index}
        for index, row in enumerate([*positive, *aggregates, *outcomes])
    ]
    return achievements + [
        {
            "idx": len(achievements) + int(r["idx"]),
            "kind": r["kind"],
            "label": r["label"],
            "ok": bool(r["ok"]),
            "offenders": list(r["offenders"]),
        }
        for r in negative
    ]


def _aggregate_goal_report(
    contracts: Sequence[Mapping[str, object]], relations: set[tuple]
) -> list[dict]:
    rows = [
        _aggregate_goal_verdict(index, contract, relations)
        for index, contract in enumerate(contracts)
    ]
    _invalidate_reused_exact_count_candidates(rows)
    return rows


def _invalidate_reused_exact_count_candidates(rows: list[dict]) -> None:
    """Reject one candidate satisfying the same class in multiple destinations."""
    conflicts: list[set[str]] = [set() for _row in rows]
    for left_index, left in enumerate(rows):
        if left.get("kind") != "exact_count_inside":
            continue
        left_candidates = tuple(left.get("candidates") or ())
        left_inside = set(left.get("inside_candidates") or ())
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            if (
                right.get("kind") != "exact_count_inside"
                or tuple(right.get("candidates") or ()) != left_candidates
                or right.get("target") == left.get("target")
            ):
                continue
            repeated = left_inside.intersection(right.get("inside_candidates") or ())
            conflicts[left_index].update(repeated)
            conflicts[right_index].update(repeated)
    for row, repeated in zip(rows, conflicts):
        if not repeated:
            continue
        row["ok"] = False
        row["progress"] = 0.0
        row["candidate_in_multiple_targets"] = sorted(repeated)


def _aggregate_goal_verdict(
    index: int, contract: Mapping[str, object], relations: set[tuple]
) -> dict:
    kind = str(contract.get("kind") or "")
    candidates = {
        str(candidate)
        for candidate in contract.get("candidates", ())
        if isinstance(candidate, str)
    }
    target = str(contract.get("target") or "")
    count_value = contract.get("count")
    count = (
        int(count_value)
        if isinstance(count_value, int) and not isinstance(count_value, bool)
        else 0
    )
    inside = sorted(
        {
            str(subject)
            for pred, subject, relation_target in relations
            if str(getattr(pred, "value", pred)).casefold() == "inside"
            and str(subject) in candidates
            and str(relation_target) == target
        }
    )
    actual = len(inside)
    supported = kind == "exact_count_inside" and bool(candidates) and count > 0
    ok = supported and actual == count
    progress = min(1.0, actual / count) if supported and actual <= count else 0.0
    return {
        "idx": index,
        "pred": kind or "unknown_aggregate_goal",
        "kind": kind or "unknown_aggregate_goal",
        "label": str(contract.get("label") or kind or "aggregate_goal"),
        "candidates": sorted(candidates),
        "target": target,
        "count": count,
        "actual": actual,
        "inside_candidates": inside,
        "ok": ok,
        "progress": progress,
        **({} if supported else {"error": "unsupported aggregate goal"}),
    }


def _trace_outcome_report(
    contracts: Sequence[Mapping[str, object]], trace_state: Mapping[str, object]
) -> list[dict]:
    events = [row for row in trace_state.get("events", ()) if isinstance(row, Mapping)]
    return [
        _trace_outcome_verdict(index, contract, events, trace_state)
        for index, contract in enumerate(contracts)
    ]


def _trace_outcome_verdict(
    index: int,
    contract: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
    trace_state: Mapping[str, object],
) -> dict:
    kind = str(contract.get("kind") or "")
    if kind == "halt_on_absent":
        return _halt_outcome(index, contract, events)
    if kind == "ordered_route":
        return _route_outcome(index, contract, events, trace_state)
    if kind == "detached_then_reattached":
        return _detached_then_reattached_outcome(index, contract, events)
    if kind == "attachment_precedence":
        return _attachment_precedence_outcome(index, contract, events)
    return {
        "idx": index,
        "pred": kind or "unknown_trace_outcome",
        "kind": kind or "unknown_trace_outcome",
        "label": str(contract.get("label") or kind or "trace_outcome"),
        "ok": False,
        "error": "unsupported trace outcome",
    }


def _halt_outcome(
    index: int,
    contract: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
) -> dict:
    halt_index = next(
        (
            event_index
            for event_index, event in enumerate(events)
            if event.get("event") == "policy_halted"
            or "policy_halted" in _event_effect_ops(event)
        ),
        None,
    )
    considered = events if halt_index is None else events[: halt_index + 1]
    facts = {
        str(fact)
        for event in considered
        for fact in event.get("required_fact_ids", ())
        if isinstance(fact, str)
    }
    required = {
        str(fact)
        for fact in contract.get("required_fact_ids", ())
        if isinstance(fact, str)
    }
    effect_ops = {op for event in considered for op in _event_effect_ops(event)}
    forbidden = {
        str(op)
        for op in contract.get("forbidden_effect_ops", ())
        if isinstance(op, str)
    }
    missing = sorted(required - facts)
    violations = sorted(forbidden & effect_ops)
    return {
        "idx": index,
        "pred": "halt_on_absent",
        "kind": "halt_on_absent",
        "label": str(contract.get("label") or "halt_on_absent"),
        "ok": halt_index is not None and not missing and not violations,
        "terminal": "policy_halted" if halt_index is not None else None,
        "required_fact_ids": sorted(required),
        "observed_fact_ids": sorted(facts),
        "missing_fact_ids": missing,
        "forbidden_effect_ops": sorted(forbidden),
        "forbidden_effect_events": violations,
    }


def _route_outcome(
    index: int,
    contract: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
    trace_state: Mapping[str, object],
) -> dict:
    ordered = [
        str(value)
        for value in contract.get("ordered_waypoints", ())
        if isinstance(value, str)
    ]
    visits = [waypoint for event in events for waypoint in _event_waypoints(event)]
    cursor = 0
    violation: str | None = None
    for waypoint in visits:
        if cursor < len(ordered) and waypoint == ordered[cursor]:
            cursor += 1
            continue
        if waypoint in ordered[cursor + 1 :]:
            violation = waypoint
            break
    current_at = {
        str(value)
        for value in trace_state.get("current_at", ())
        if isinstance(value, str)
    }
    terminal_target = ordered[-1] if ordered else None
    terminal_ok = terminal_target is not None and terminal_target in current_at
    return {
        "idx": index,
        "pred": "ordered_route",
        "kind": "ordered_route",
        "label": str(contract.get("label") or "ordered_route"),
        "ok": bool(ordered)
        and cursor == len(ordered)
        and violation is None
        and terminal_ok,
        "ordered_waypoints": ordered,
        "visited_waypoints": visits,
        "ordered_waypoints_done": cursor,
        "ordered_waypoints_total": len(ordered),
        "order_violation": violation,
        "terminal": {
            "pred": "at",
            "subject": "human",
            "target": terminal_target,
            "ok": terminal_ok,
        },
    }


def _detached_then_reattached_outcome(
    index: int,
    contract: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
) -> dict:
    subject = str(contract.get("subject") or "")
    target = str(contract.get("target") or "")
    connector = str(contract.get("connector") or "")
    detached_index = next(
        (
            event_index
            for event_index, event in enumerate(events)
            if _event_has_effect(event, "not_attached", subject, target)
        ),
        None,
    )
    reattached_index = (
        next(
            (
                event_index
                for event_index, event in enumerate(events)
                if event_index > detached_index
                and _event_has_exact_attachment(event, subject, target, connector)
            ),
            None,
        )
        if detached_index is not None
        else None
    )
    return {
        "idx": index,
        "pred": "detached_then_reattached",
        "kind": "detached_then_reattached",
        "label": str(contract.get("label") or "detached_then_reattached"),
        "subject": subject,
        "target": target,
        "connector": connector,
        "ok": detached_index is not None and reattached_index is not None,
        "detached_event_index": detached_index,
        "reattached_event_index": reattached_index,
        "detached_beat_id": (
            str(events[detached_index].get("beat_id") or "")
            if detached_index is not None
            else None
        ),
        "reattached_beat_id": (
            str(events[reattached_index].get("beat_id") or "")
            if reattached_index is not None
            else None
        ),
    }


def _event_has_exact_attachment(
    event: Mapping[str, object],
    subject: str,
    target: str,
    connector: str,
    child_point_idx: int = 0,
    parent_point_idx: int = 0,
) -> bool:
    return any(
        isinstance(effect, Mapping)
        and effect.get("op") == "attached"
        and effect.get("subject") == subject
        and effect.get("target") == target
        and effect.get("physical_attached") is True
        and effect.get("child_point") == f"{connector}M"
        and effect.get("parent_point") == f"{connector}F"
        and effect.get("child_point_idx") == child_point_idx
        and effect.get("parent_point_idx") == parent_point_idx
        for effect in event.get("actual_effects", ())
    )


def _attachment_precedence_outcome(
    index: int,
    contract: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
) -> dict:
    before = _attachment_edge(contract.get("before"))
    after = _attachment_edge(contract.get("after"))
    before_position: tuple[int, int] | None = None
    after_position: tuple[int, int] | None = None
    violation_position: tuple[int, int] | None = None
    if before is not None and after is not None and before != after:
        for event_index, event in enumerate(events):
            for effect_index, effect in enumerate(event.get("actual_effects", ())):
                if not isinstance(effect, Mapping):
                    continue
                if (
                    _effect_matches_attachment(effect, after)
                    and before_position is None
                ):
                    violation_position = (event_index, effect_index)
                    break
                if before_position is None and _effect_matches_attachment(
                    effect, before
                ):
                    before_position = (event_index, effect_index)
                    continue
                if before_position is not None and _effect_matches_attachment(
                    effect, after
                ):
                    after_position = (event_index, effect_index)
                    break
            if violation_position is not None or after_position is not None:
                break
    return {
        "idx": index,
        "pred": "attachment_precedence",
        "kind": "attachment_precedence",
        "label": str(contract.get("label") or "attachment_precedence"),
        "before": before,
        "after": after,
        "ok": (
            before_position is not None
            and after_position is not None
            and violation_position is None
        ),
        "before_event_index": _position_value(before_position, 0),
        "before_effect_index": _position_value(before_position, 1),
        "before_beat_id": _position_beat(events, before_position),
        "after_event_index": _position_value(after_position, 0),
        "after_effect_index": _position_value(after_position, 1),
        "after_beat_id": _position_beat(events, after_position),
        "order_violation_event_index": _position_value(violation_position, 0),
        "order_violation_effect_index": _position_value(violation_position, 1),
        "order_violation_beat_id": _position_beat(events, violation_position),
    }


def _attachment_edge(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    child = value.get("child")
    parent = value.get("parent")
    connector = value.get("connector")
    child_point_idx = value.get("child_point_idx")
    parent_point_idx = value.get("parent_point_idx")
    if (
        not isinstance(child, str)
        or not child
        or not isinstance(parent, str)
        or not parent
        or not isinstance(connector, str)
        or not connector
        or not isinstance(child_point_idx, int)
        or isinstance(child_point_idx, bool)
        or child_point_idx < 0
        or not isinstance(parent_point_idx, int)
        or isinstance(parent_point_idx, bool)
        or parent_point_idx < 0
    ):
        return None
    return {
        "child": child,
        "parent": parent,
        "connector": connector,
        "child_point_idx": child_point_idx,
        "parent_point_idx": parent_point_idx,
    }


def _effect_matches_attachment(
    effect: Mapping[str, object], edge: Mapping[str, object]
) -> bool:
    return (
        effect.get("op") == "attached"
        and effect.get("subject") == edge["child"]
        and effect.get("target") == edge["parent"]
        and effect.get("physical_attached") is True
        and effect.get("child_point") == f"{edge['connector']}M"
        and effect.get("parent_point") == f"{edge['connector']}F"
        and effect.get("child_point_idx") == edge["child_point_idx"]
        and effect.get("parent_point_idx") == edge["parent_point_idx"]
    )


def _position_value(position: tuple[int, int] | None, index: int) -> int | None:
    return position[index] if position is not None else None


def _position_beat(
    events: Sequence[Mapping[str, object]], position: tuple[int, int] | None
) -> str | None:
    if position is None:
        return None
    return str(events[position[0]].get("beat_id") or "")


def _event_has_effect(
    event: Mapping[str, object], op: str, subject: str, target: str
) -> bool:
    return any(
        isinstance(effect, Mapping)
        and effect.get("op") == op
        and effect.get("subject") == subject
        and effect.get("target") == target
        for effect in event.get("actual_effects", ())
    )


def _event_effect_ops(event: Mapping[str, object]) -> set[str]:
    result: set[str] = set()
    if isinstance(event.get("effect_op"), str):
        result.add(str(event["effect_op"]))
    for effect in event.get("actual_effects", ()):
        if isinstance(effect, Mapping) and isinstance(effect.get("op"), str):
            result.add(str(effect["op"]))
    return result


def _event_waypoints(event: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    if isinstance(event.get("waypoint"), str):
        result.append(str(event["waypoint"]))
    for effect in event.get("actual_effects", ()):
        if (
            isinstance(effect, Mapping)
            and effect.get("op") == "at"
            and effect.get("subject") == "human"
            and isinstance(effect.get("target"), str)
        ):
            waypoint = str(effect["target"])
            if waypoint not in result:
                result.append(waypoint)
    return result

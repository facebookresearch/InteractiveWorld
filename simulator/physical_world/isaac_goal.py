# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Symbolic goal predicates + a physics-agnostic goal checker for the benchmark / RL loop.

A goal is a conjunction of relational predicates over the scene's CURRENT symbolic relations (the set
``scene_state.relations()`` produces). This REPLACES VirtualHome's ``check_goal`` (which read a VH
``EnvironmentGraph`` that Isaac lacks); here the relation set is derived from Isaac PhysX contacts /
scene-query overlaps / articulation joint state (see ``scene_state``). These functions are PURE (a
relation set -> a verdict), so they are offline-testable with synthetic relations -- no Isaac needed.

Relations are represented as a set of ``(Pred, a, b)`` tuples; unary predicates use b=None.

    python3 simulator/physical_world/isaac_goal.py   # offline selftest
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Mapping

try:
    from . import runtime_capabilities
except ImportError:
    import runtime_capabilities


class Pred(enum.Enum):
    """The relational predicates scored by the benchmark."""

    ON = "on"  # (obj, surface): obj rests on surface's top face
    INSIDE = "inside"  # (obj, container): obj is within a container's interior
    HELD = "held"  # (obj, None): obj is currently grasped by the agent
    OPEN = "open"  # (openable, None): a drawer/door joint is past its open threshold
    CLOSED = (
        "closed"  # (openable, None): NOT open (derived, never emitted by relations())
    )
    OFF = (
        "off"  # (toggleable, None): NOT running (derived, never emitted by relations())
    )
    NEAR = "near"  # (a, b): a is within reach distance of b
    SITTING = "sitting"  # (seat, None): the agent is seated on `seat`
    CUT = (
        "cut"  # (obj, None): a CONTACT cut action completed on obj (declared, symbolic)
    )
    COOKED = (
        "cooked"  # (obj, None): a COOK action completed on obj (declared, symbolic)
    )
    RUNNING = "running"  # (device, None): a toggleable device has been turned on
    ATTACHED = "attached"  # (child, parent): child's male attach-point is mated to parent's female (live world-pose coincidence)
    RAISED = "raised"  # (assembly root, None): live orientation differs from its episode baseline
    UNMOVED = "unmoved"  # (entity, None): maximum episode pose deviation stayed within tolerance


@dataclass(frozen=True)
class Predicate:
    """One goal predicate. ``b`` is None for unary predicates."""

    pred: Pred
    a: str
    b: str | None = None

    def as_tuple(self) -> tuple:
        return (self.pred, self.a, self.b)


# ---- convenience builders --------------------------------------------------------------------------
def on(obj: str, surface: str) -> Predicate:
    return Predicate(Pred.ON, obj, surface)


def inside(obj: str, container: str) -> Predicate:
    return Predicate(Pred.INSIDE, obj, container)


def held(obj: str) -> Predicate:
    return Predicate(Pred.HELD, obj)


def is_open(openable: str) -> Predicate:
    return Predicate(Pred.OPEN, openable)


def is_closed(openable: str) -> Predicate:
    return Predicate(Pred.CLOSED, openable)


def is_off(toggleable: str) -> Predicate:
    return Predicate(Pred.OFF, toggleable)


def near(a: str, b: str) -> Predicate:
    return Predicate(Pred.NEAR, a, b)


def is_sitting(seat: str) -> Predicate:
    return Predicate(Pred.SITTING, seat)


def is_cut(obj: str) -> Predicate:
    return Predicate(Pred.CUT, obj)


def is_cooked(obj: str) -> Predicate:
    return Predicate(Pred.COOKED, obj)


def is_running(device: str) -> Predicate:
    return Predicate(Pred.RUNNING, device)


def attached(child: str, parent: str) -> Predicate:
    return Predicate(Pred.ATTACHED, child, parent)


def is_raised(obj: str) -> Predicate:
    return Predicate(Pred.RAISED, obj)


def is_unmoved(obj: str) -> Predicate:
    return Predicate(Pred.UNMOVED, obj)


def from_row(row: Mapping[str, object]) -> Predicate:
    """Convert one TaskPack goal row using the shared predicate registry."""
    canonical = runtime_capabilities.canonical_predicate(row.get("pred"))
    if canonical is None:
        raise ValueError(f"unknown goal predicate {row.get('pred')!r}")
    capability = runtime_capabilities.predicate_by_name()[
        runtime_capabilities.normalize(canonical)
    ]
    subject = str(row.get("subject") or "")
    target = str(row.get("target") or "") if capability.arity == 2 else None
    if not subject or (capability.arity == 2 and not target):
        raise ValueError(f"malformed {canonical} goal row: {dict(row)!r}")
    return Predicate(Pred(canonical), subject, target)


def from_rows(rows) -> list[Predicate]:
    return [from_row(row) for row in rows or ()]


# ---- checking (pure over a relation set) -----------------------------------------------------------
def satisfied(pred: Predicate, relations: set) -> bool:
    """Return whether one predicate holds, including absence-derived states."""
    if pred.pred is Pred.CLOSED:
        return (Pred.OPEN, pred.a, None) not in relations
    if pred.pred is Pred.OFF:
        return (Pred.RUNNING, pred.a, None) not in relations
    return pred.as_tuple() in relations


def check_goal(goal: list[Predicate], relations: set) -> bool:
    """True iff EVERY predicate in the (conjunctive) goal holds. Empty goal is trivially satisfied."""
    return all(satisfied(p, relations) for p in goal)


def goal_progress(goal: list[Predicate], relations: set) -> float:
    """Fraction of the goal's predicates satisfied (a dense RL/eval signal). Empty goal -> 1.0."""
    if not goal:
        return 1.0
    return sum(satisfied(p, relations) for p in goal) / len(goal)


def unmet(goal: list[Predicate], relations: set) -> list[Predicate]:
    """The predicates NOT yet satisfied (for failure diagnosis / recovery)."""
    return [p for p in goal if not satisfied(p, relations)]


def goal_report(goal: list[Predicate], relations: set, offset: int = 0) -> list[dict]:
    """Per-predicate verdicts as JSON-ready records: ``{idx, pred, subject, target, ok}``.

    ``check_goal``/``goal_progress`` reduce the same per-predicate booleans through ``all``/``sum`` and
    throw them away, so a caller that wants to know WHICH predicate failed (per-sub-goal scoring, a
    capability breakdown, a dense reward) had to recompute them. This exposes them once, in the goal's
    own order, so a consumer can join a verdict back to the goal that produced it by index.

    ``offset`` shifts ``idx`` when these predicates are one segment of a larger scoreable array (e.g.
    positives followed by negative constraints). ``target`` is None for unary predicates.
    """
    return [
        {
            "idx": offset + i,
            "pred": p.pred.value,
            "subject": p.a,
            "target": p.b,
            "ok": bool(satisfied(p, relations)),
        }
        for i, p in enumerate(goal)
    ]


# ---- offline selftest (no Isaac; pure relation-set logic) ------------------------------------------
def _selftest() -> int:
    ok = True

    def _chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    # A cup starts held, then rests on deskA, then is moved onto deskB; a drawer opens.
    goal = [on("cup", "deskB")]
    rel_start = {(Pred.HELD, "cup", None)}
    rel_deskA = {(Pred.ON, "cup", "deskA")}
    rel_deskB = {(Pred.ON, "cup", "deskB")}

    _chk(not check_goal(goal, rel_start), "goal ON(cup,deskB) NOT met while held")
    _chk(not check_goal(goal, rel_deskA), "goal NOT met while on deskA")
    _chk(check_goal(goal, rel_deskB), "goal MET once on deskB")
    _chk(goal_progress(goal, rel_deskA) == 0.0, "progress 0 on deskA")
    _chk(goal_progress(goal, rel_deskB) == 1.0, "progress 1 on deskB")

    # Multi-predicate goal + CLOSED derivation + unmet reporting.
    goal2 = [on("cup", "deskB"), held("pen"), is_closed("drawer")]
    rel = {(Pred.ON, "cup", "deskB")}  # pen not held, drawer not open (=> closed)
    _chk(satisfied(is_closed("drawer"), rel), "CLOSED true when OPEN absent")
    _chk(
        not satisfied(is_closed("drawer"), {(Pred.OPEN, "drawer", None)}),
        "CLOSED false when OPEN present",
    )
    _chk(satisfied(is_off("stove"), rel), "OFF true when RUNNING absent")
    _chk(
        not satisfied(is_off("stove"), {(Pred.RUNNING, "stove", None)}),
        "OFF false when RUNNING present",
    )
    _chk(
        abs(goal_progress(goal2, rel) - 2.0 / 3.0) < 1e-9,
        "progress 2/3 (cup+drawer, no pen)",
    )
    _chk([p.a for p in unmet(goal2, rel)] == ["pen"], "unmet reports exactly [pen]")

    # goal_report: one record per predicate, in goal order, agreeing with the aggregates.
    rep = goal_report(goal2, rel)
    _chk(
        [r["idx"] for r in rep] == [0, 1, 2], "goal_report idx is 0..n-1 in goal order"
    )
    _chk(
        [r["ok"] for r in rep] == [True, False, True],
        "goal_report ok agrees per predicate (cup yes, pen no, drawer yes)",
    )
    _chk(
        rep[0]
        == {"idx": 0, "pred": "on", "subject": "cup", "target": "deskB", "ok": True},
        "goal_report record shape {idx,pred,subject,target,ok}",
    )
    _chk(rep[1]["target"] is None, "unary predicate reports target=None")
    _chk(
        all(r["ok"] for r in rep) == check_goal(goal2, rel)
        and sum(r["ok"] for r in rep) / len(rep) == goal_progress(goal2, rel),
        "goal_report reduces to check_goal / goal_progress exactly",
    )
    _chk(
        [r["idx"] for r in goal_report(goal2, rel, offset=5)] == [5, 6, 7],
        "goal_report offset shifts idx into a larger scoreable array",
    )
    _chk(goal_report([], rel) == [], "goal_report of an empty goal is empty")

    print(f"[isaac_goal] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())

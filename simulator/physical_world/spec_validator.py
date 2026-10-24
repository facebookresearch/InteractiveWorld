# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""OFFLINE benchmark task-spec validator -- catch a broken task in seconds, not 40 minutes.

Every rule here defends a VERIFIED silent-failure path of the render harness: an answer key that
names an entity the scene never built, a negative constraint that can never fire, a goal that is
already true at t=0, a scoring surface nothing scores. Each of those otherwise costs a full RTX render
before it shows up -- and several never show up at all, they just publish a free 1.00.

STANDALONE + READ-ONLY. This module is NOT wired into the render path and changes no gate, no scorer
and no action: it imports a task spec, normalises it, and prints findings. Running it is free.

    python3 simulator/physical_world/spec_validator.py                  # selftest (pure, no Isaac, no cache)
    python3 simulator/physical_world/spec_validator.py --all            # validate every SHIPPED_SPECS module
    python3 simulator/physical_world/spec_validator.py --spec <task_module> --json

Exit codes: 0 = no ERROR, 1 = at least one ERROR finding, 2 = selftest failure.

WHAT IS NOT KNOWABLE HERE. Two checks need a BUILT stage and are marked ``needs_stage``; both are
implemented as pure kernels (``container_xy_overlap_pairs`` / ``pre_satisfied_predicates``) that are
unit-tested below and are ready to be called with real numbers the moment a build-time hook exists:
  * CON005 container xy-footprint overlap -- imported containers are positioned by ``_find_slot`` at
    build time, so no declared coordinate exists to compare offline.
  * GOAL010 geometric pre-satisfaction -- an object's settled pose at t=0 is a physics result.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
from dataclasses import dataclass, field

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import asset_preflight  # noqa: E402
from constraint_evaluator import eval_neg_constraints, NEG_KINDS  # noqa: E402
from isaac_goal import check_goal, inside, Pred  # noqa: E402

ERROR = "ERROR"
WARN = "WARN"
INFO = "INFO"

SHIPPED_SPECS: tuple[str, ...] = ("constraint_evaluator",)

# The negative-constraint kinds ``constraint_evaluator.eval_neg_constraints`` actually implements -- ALIASED
# from the evaluator's own module so this table cannot drift from the branches it describes. Anything else
# evaluates to ``ok=False`` with an ``unimplemented_kind:<name>`` offender, so the render cannot publish a
# free pass; NEG001 below still names it at AUTHORING time, before the 40-minute render.
IMPLEMENTED_NEG_KINDS: frozenset[str] = NEG_KINDS

# The fields each implemented kind's evaluator branch reads, and which NEG002 therefore requires to be
# non-empty. INVARIANT: exactly one entry per IMPLEMENTED_NEG_KINDS member, in BOTH directions --
# ``_evaluator_selftest`` fails on either kind of drift, since the registry lives in another module and
# grows there. A registered kind with no entry here is not field-checkable, so ``rule_neg_kinds`` reports
# NEG009 rather than indexing it.
REQUIRED_NEG_FIELDS: dict[str, tuple[str, ...]] = {
    "not_inside": ("objs", "containers"),
    "all_placed": ("objs",),
    "no_share_group": ("group_of", "incompatible"),
    "capacity": ("cap",),
}

# Predicates ``scene_state.relations(source="physics")`` can actually EMIT. ON is never emitted (no
# contact report is passed) and OPEN is pinned to 0.0, so a goal using ON/OPEN/CLOSED is either
# permanently unsatisfiable or permanently vacuous.
EMITTED_PREDS: frozenset[str] = frozenset({"inside", "attached"})

# Keys of ``scene.behavior_task`` the gated scorer in ``episode_runtime`` reads. A spec that
# returns its negatives under any other key is declaring constraints nothing will ever evaluate.
SCORER_NEG_KEY = "neg_constraints"
SCORER_GOAL_KEYS: frozenset[str] = frozenset({"goals", "carcass", "grasp", "container"})


@dataclass(frozen=True)
class Finding:
    """One structured verdict. ``needs_stage`` marks a check that can only run on a built scene."""

    rule: str
    severity: str
    entity: str
    message: str
    fix: str
    needs_stage: bool = False

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Goal:
    """One scored predicate. ``target`` is None for unary predicates."""

    pred: str
    subject: str
    target: str | None = None


@dataclass
class TaskSpec:
    """A task spec normalised across the three incompatible shapes the shipped modules use."""

    name: str
    scene: str
    goals: list[Goal] = field(default_factory=list)
    negs: list[dict] = field(default_factory=list)
    negs_key: str = SCORER_NEG_KEY
    return_shape: str = "tuple"
    items: dict[str, dict] = field(default_factory=dict)
    containers: dict[str, dict] = field(default_factory=dict)
    parts: dict[str, dict] = field(default_factory=dict)
    decor: list[dict] = field(default_factory=list)
    carcass_edges: list[tuple[str, str]] = field(default_factory=list)
    unscored_surfaces: dict[str, list] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    deferred: dict[str, object] = field(default_factory=dict)

    def entity_keys(self) -> set[str]:
        return set(self.items) | set(self.parts)

    def known_keys(self) -> set[str]:
        return self.entity_keys() | set(self.containers)

    def goal_targets(self) -> set[str]:
        return {g.target for g in self.goals if g.target}


# ---- pure kernels (offline-testable; fed real numbers only from a BUILT stage) ----------------------
def container_xy_overlap_pairs(
    boxes: dict[str, tuple[float, float, float, float, float, float]],
    margin: float = 0.0,
) -> list[tuple[str, str, float]]:
    """Container pairs whose xy footprints AND z spans intersect -> the scorer cannot tell them apart.

    ``scene_state``'s ``_relations_inside_live`` is one outer-AABB xy test with no exclusivity,
    so an object in the intersection scores INSIDE BOTH -- ``capacity`` double-counts and
    ``no_share_group`` phantom-fires. boxes: key -> (xmin, ymin, zmin, xmax, ymax, zmax).
    Returns (a, b, overlap_area) sorted, for every offending pair.
    """
    out: list[tuple[str, str, float]] = []
    keys = sorted(boxes)
    for i, a in enumerate(keys):
        ax0, ay0, az0, ax1, ay1, az1 = boxes[a]
        for b in keys[i + 1 :]:
            bx0, by0, bz0, bx1, by1, bz1 = boxes[b]
            dx = min(ax1, bx1) - max(ax0, bx0) - margin
            dy = min(ay1, by1) - max(ay0, by0) - margin
            dz = min(az1, bz1) - max(az0, bz0)
            if dx > 0.0 and dy > 0.0 and dz > 0.0:
                out.append((a, b, round(dx * dy, 6)))
    return out


def pre_satisfied_predicates(goals: list[Goal], relations_t0: set) -> list[Goal]:
    """Goal predicates ALREADY true in the t=0 relation set -- a task that starts solved scores free."""
    pred_of = {"inside": Pred.INSIDE, "attached": Pred.ATTACHED, "on": Pred.ON}
    return [
        g
        for g in goals
        if (pred_of.get(g.pred), g.subject, g.target) in relations_t0
        and g.pred in pred_of
    ]


# ---- spec normalisation ----------------------------------------------------------------------------
def _goals_from_dict(fields: dict) -> tuple[list[Goal], list[tuple[str, str]]]:
    goals: list[Goal] = []
    carcass = [(str(c), str(p)) for c, p in fields.get("carcass") or []]
    goals += [Goal("attached", c, p) for c, p in carcass]
    goals += [Goal("inside", str(o), str(c)) for o, c in fields.get("goals") or []]
    return goals, carcass


def _negs_from_dict(fields: dict) -> tuple[list[dict], str]:
    """Return (negatives, the KEY they were declared under) -- the key itself is a finding (SHAPE002)."""
    for key in (SCORER_NEG_KEY, "negs", "neg", "negatives"):
        if key in fields:
            return list(fields[key] or []), key
    return [], SCORER_NEG_KEY


def spec_from_module(mod, name: str | None = None) -> TaskSpec:
    """Normalise a task module. Tolerates the tuple-returning and dict-returning conventions."""
    fields = mod.build_task_fields()
    if isinstance(fields, tuple):
        raw_goals, raw_negs = fields
        goals = [Goal("inside", str(o), str(c)) for o, c in raw_goals]
        negs, negs_key, shape, carcass = (
            list(raw_negs or []),
            SCORER_NEG_KEY,
            "tuple",
            [],
        )
        unscored: dict[str, list] = {}
    elif isinstance(fields, dict):
        goals, carcass = _goals_from_dict(fields)
        negs, negs_key = _negs_from_dict(fields)
        shape = "dict"
        unscored = {
            k: list(v)
            for k, v in fields.items()
            if k not in SCORER_GOAL_KEYS
            and k != negs_key
            and isinstance(v, (list, tuple))
            and v
        }
    else:
        raise TypeError(
            f"{name or mod.__name__}.build_task_fields() returned {type(fields).__name__}; "
            "expected a (goals, negs) tuple or a dict"
        )
    return TaskSpec(
        name=name or mod.__name__,
        scene=str(getattr(mod, "SCENE", "")),
        goals=goals,
        negs=negs,
        negs_key=negs_key,
        return_shape=shape,
        items=dict(getattr(mod, "ITEMS", {}) or {}),
        containers=dict(getattr(mod, "CONTAINERS", {}) or {}),
        parts=dict(getattr(mod, "PARTS", {}) or {}),
        decor=list(getattr(mod, "DECOR", []) or []),
        carcass_edges=carcass,
        unscored_surfaces=unscored,
        labels=dict(getattr(mod, "LABELS", {}) or {}),
        deferred=dict(getattr(mod, "DEFERRED_SCORING", {}) or {}),
    )


def load_spec(module_name: str) -> TaskSpec:
    return spec_from_module(importlib.import_module(module_name), module_name)


# ---- rules -----------------------------------------------------------------------------------------
def rule_shape(spec: TaskSpec) -> list[Finding]:
    """SHAPE001-004: the spec<->scorer contract (return type, key names, declared-but-unscored)."""
    out: list[Finding] = []
    if spec.return_shape != "tuple":
        out.append(
            Finding(
                "SHAPE001",
                WARN,
                spec.name,
                f"build_task_fields() returns a {spec.return_shape}; the other shipped specs return a "
                "(goals, negs) tuple. Generic tooling must branch on the type.",
                "Agree one return type across all specs (recommend the dict -- it is the superset).",
            )
        )
    if spec.negs and spec.negs_key != SCORER_NEG_KEY:
        out.append(
            Finding(
                "SHAPE002",
                ERROR,
                f"{spec.name}.{spec.negs_key}",
                f"{len(spec.negs)} negative constraint(s) declared under key {spec.negs_key!r}, but the "
                f"scorer reads {SCORER_NEG_KEY!r} -- they are never evaluated (silent all-pass).",
                f"Rename the key to {SCORER_NEG_KEY!r} in build_task_fields() and in the builder that "
                "copies it into scene.behavior_task.",
            )
        )
    for key, vals in spec.unscored_surfaces.items():
        out.append(
            Finding(
                "SHAPE003",
                ERROR,
                f"{spec.name}.{key}",
                f"{len(vals)} entity/entities declared under {key!r}, which no scorer branch reads "
                f"(scorer keys: {sorted(SCORER_GOAL_KEYS)}) -- a declared answer key that scores nothing.",
                f"Either implement a scorer branch for {key!r} or drop it from build_task_fields().",
            )
        )
    return out + _rule_start_containers(spec)


def _rule_start_containers(spec: TaskSpec) -> list[Finding]:
    """SHAPE004: an item whose declared start is ``in:<X>`` must name a declared container."""
    out: list[Finding] = []
    for key, item in spec.items.items():
        place = str(item.get("place", ""))
        if place.startswith("in:") and place[3:] not in spec.containers:
            out.append(
                Finding(
                    "SHAPE004",
                    ERROR,
                    key,
                    f"starts {place!r} but {place[3:]!r} is not a declared container -- the builder has "
                    "nowhere to seat it, so the item is dropped or spawns in mid-air.",
                    f"Declare {place[3:]!r} in CONTAINERS, or fix the typo.",
                )
            )
    return out


def rule_deferred(spec: TaskSpec) -> list[Finding]:
    """SHAPE005: keep DECLARED-BUT-UNSCORED surfaces visible on every run.

    A spec may legitimately design more than the engine can evaluate yet (bookcase's per-compartment
    stocking). Moving that half into ``DEFERRED_SCORING`` is right -- returned as live fields it would be
    either a free pass (an answer key nothing reads) or an automatic fail (an unimplemented negative kind).
    But silence about it is how a partially-scored task gets mistaken for a fully-scored one, so the gap is
    reported every run, not once in a comment.
    """
    if not spec.deferred:
        return []
    surfaces: list[str] = []
    for key, val in sorted(spec.deferred.items()):
        if key == "why" or not isinstance(val, (list, tuple)) or not val:
            continue
        if key == SCORER_NEG_KEY:
            kinds = sorted({str(c.get("kind")) for c in val if isinstance(c, dict)})
            surfaces.append(f"{key} ({len(val)}: kinds {kinds})")
        else:
            surfaces.append(f"{key} ({len(val)})")
    if not surfaces:
        return []
    why = str(spec.deferred.get("why") or "no reason recorded")
    return [
        Finding(
            "SHAPE005",
            WARN,
            f"{spec.name}.DEFERRED_SCORING",
            f"declared but NOT scored: {'; '.join(surfaces)}. Reason: {why}. The published score covers "
            f"only the {len(spec.goals)} live predicate(s) -- a run can be 1.00 and still have done none "
            "of the deferred work.",
            "Implement the missing evaluator(s) and move the block into build_task_fields(), or drop it "
            "from the task design. Until then treat this task's score as partial.",
        )
    ]


def rule_goal_vocabulary(spec: TaskSpec) -> list[Finding]:
    """GOAL001-003, GOAL005-006: the goal exists, is scorable, and every name resolves."""
    out: list[Finding] = []
    if not spec.goals:
        out.append(
            Finding(
                "GOAL001",
                ERROR,
                spec.name,
                "empty goal list -- check_goal()/goal_progress() report met=True, progress=1.0 for an "
                "empty conjunction, so the episode scores a free 1.00 (plan §2.1 F1).",
                "Emit at least one predicate, or have the builder mark the task scored=False.",
            )
        )
    seen: set[tuple] = set()
    for g in spec.goals:
        sig = (g.pred, g.subject, g.target)
        if sig in seen:
            out.append(
                Finding(
                    "GOAL002",
                    WARN,
                    f"{g.subject}->{g.target}",
                    "duplicate goal predicate -- inflates the progress denominator and double-counts.",
                    "De-duplicate the goal list.",
                )
            )
        seen.add(sig)
        if g.pred not in EMITTED_PREDS:
            out.append(
                Finding(
                    "GOAL003",
                    ERROR,
                    f"{g.subject}->{g.target}",
                    f"predicate {g.pred!r} is never emitted by relations(source='physics') -- the goal "
                    "is permanently unsatisfiable (and its negation permanently vacuous).",
                    f"Restrict the goal to {sorted(EMITTED_PREDS)} until the emitter lands (plan F7).",
                )
            )
    return out + _rule_goal_names(spec)


def _rule_goal_names(spec: TaskSpec) -> list[Finding]:
    out: list[Finding] = []
    entities, containers = spec.entity_keys(), set(spec.containers)
    for g in spec.goals:
        if g.subject not in entities:
            out.append(
                Finding(
                    "GOAL005",
                    ERROR,
                    g.subject,
                    "goal subject is not a declared ITEM/PART -- nothing will ever be registered under "
                    "that name, so the predicate can never become true.",
                    f"Declare {g.subject!r}, or fix the typo (known: {sorted(entities)[:6]}...).",
                )
            )
        legal = containers if g.pred == "inside" else entities
        if g.target is not None and g.target not in legal:
            out.append(
                Finding(
                    "GOAL006",
                    ERROR,
                    str(g.target),
                    f"goal target of {g.pred.upper()}({g.subject}, {g.target}) is not a declared "
                    f"{'container' if g.pred == 'inside' else 'entity'} -- permanently unmet.",
                    f"Declare {g.target!r}, or fix the typo (known: {sorted(legal)[:6]}...).",
                )
            )
    return out


def rule_pre_satisfied(spec: TaskSpec) -> list[Finding]:
    """GOAL004 (declared, offline) + GOAL010 (geometric, needs a built stage)."""
    out: list[Finding] = []
    for g in spec.goals:
        if g.pred != "inside":
            continue
        place = str(spec.items.get(g.subject, {}).get("place", ""))
        if place.startswith("in:") and place[3:] == g.target:
            out.append(
                Finding(
                    "GOAL004",
                    ERROR,
                    g.subject,
                    f"starts INSIDE its own goal container {g.target!r} -- the predicate is true at t=0, "
                    "so this share of the score is free and the episode need not act.",
                    "Move the start placement out of the goal container, or drop the predicate.",
                )
            )
    out.append(
        Finding(
            "GOAL010",
            INFO,
            spec.name,
            "geometric pre-satisfaction (an object that physically settles inside its goal container at "
            "t=0) is NOT checkable offline -- settled poses are a physics result. Kernel ready: "
            "pre_satisfied_predicates(spec.goals, relations_t0).",
            "REQUEST: call it at build time with scene.relations(source='physics') and reject/displace "
            "any already-true predicate; record goal_t0.pre_satisfied_preds in the manifest.",
            needs_stage=True,
        )
    )
    return out


def rule_answer_key(spec: TaskSpec) -> list[Finding]:
    """GOAL007-009: the reference solution must satisfy the goal and violate none of its own negatives."""
    out: list[Finding] = []
    ref = _reference_relations(spec)
    if spec.goals and not check_goal(
        [inside(g.subject, g.target) for g in spec.goals if g.pred == "inside"], ref
    ):
        out.append(
            Finding(
                "GOAL008",
                ERROR,
                spec.name,
                "the answer key does not satisfy its own goal conjunction -- the task is unsolvable as "
                "specified.",
                "Re-derive goals and the answer key from one source of truth.",
            )
        )
    for verdict in eval_neg_constraints(spec.negs, ref):
        # An unimplemented kind evaluates to ok=False for EVERY world including this one; NEG001 diagnoses
        # it precisely, so re-reporting it as "the answer key violates it" would be actively misleading.
        if str(verdict["kind"]) not in IMPLEMENTED_NEG_KINDS:
            continue
        if not verdict["ok"]:
            out.append(
                Finding(
                    "GOAL009",
                    ERROR,
                    str(verdict["label"]),
                    f"the reference solution VIOLATES this negative constraint "
                    f"(offenders={verdict['offenders']}) -- a perfect run still fails.",
                    "Relax the constraint or re-route the answer key.",
                )
            )
    return out + _rule_capacity(spec) + _rule_unscored_items(spec)


def _reference_relations(spec: TaskSpec) -> set:
    rel = {(Pred.INSIDE, g.subject, g.target) for g in spec.goals if g.pred == "inside"}
    return rel | {
        (Pred.ATTACHED, g.subject, g.target) for g in spec.goals if g.pred == "attached"
    }


def _rule_capacity(spec: TaskSpec) -> list[Finding]:
    """CAP: a declared capacity below (or exactly at) the answer key's own load."""
    caps: dict[str, int] = {}
    for c in spec.negs:
        if c.get("kind") == "capacity":
            caps.update({str(k): int(v) for k, v in (c.get("cap") or {}).items()})
    load: dict[str, int] = {}
    for g in spec.goals:
        if g.pred == "inside" and g.target:
            load[g.target] = load.get(g.target, 0) + 1
    out: list[Finding] = []
    for unit, cap in sorted(caps.items()):
        n = load.get(unit, 0)
        if n > cap:
            out.append(
                Finding(
                    "GOAL011",
                    ERROR,
                    unit,
                    f"answer key routes {n} item(s) here but capacity is {cap} -- the reference solution "
                    "trips its own capacity constraint.",
                    f"Raise cap[{unit!r}] to >= {n}, or re-route items.",
                )
            )
        elif n == cap and cap > 0:
            out.append(
                Finding(
                    "GOAL012",
                    WARN,
                    unit,
                    f"zero capacity headroom ({n}/{cap}). The live INSIDE test has no exclusivity, so one "
                    "spurious containment from an overlapping neighbour flips this to a violation.",
                    f"Raise cap[{unit!r}] by one, or land container exclusivity first.",
                )
            )
    return out


def _rule_unscored_items(spec: TaskSpec) -> list[Finding]:
    """GOAL007: an item that declares a destination but has no goal predicate is never scored."""
    scored = {g.subject for g in spec.goals}
    out: list[Finding] = []
    for key, item in spec.items.items():
        target = item.get("target") or item.get("ref_target")
        if target and key not in scored:
            out.append(
                Finding(
                    "GOAL007",
                    WARN,
                    key,
                    f"declares destination {target!r} but no goal predicate scores it -- moving it "
                    "correctly earns nothing and leaving it earns nothing.",
                    "Add it to build_task_fields()'s goal list, or drop the destination field.",
                )
            )
    return out


def rule_neg_kinds(spec: TaskSpec) -> list[Finding]:
    """NEG001-002, NEG004, NEG009: kind implemented + field-checkable, fields present, labels unique."""
    out: list[Finding] = []
    labels: dict[str, int] = {}
    for c in spec.negs:
        kind, label = str(c.get("kind")), str(c.get("label", c.get("kind")))
        labels[label] = labels.get(label, 0) + 1
        if kind not in IMPLEMENTED_NEG_KINDS:
            out.append(
                Finding(
                    "NEG001",
                    ERROR,
                    label,
                    f"kind {kind!r} is not implemented by eval_neg_constraints -- it evaluates to "
                    "ok=False (offender 'unimplemented_kind') for every world, so the task can never "
                    "pass, whatever the run does.",
                    f"Implement {kind!r}, or express it with {sorted(IMPLEMENTED_NEG_KINDS)}.",
                )
            )
            continue
        if kind not in REQUIRED_NEG_FIELDS:
            out.append(
                Finding(
                    "NEG009",
                    ERROR,
                    label,
                    f"kind {kind!r} is registered in NEG_KINDS but has no REQUIRED_NEG_FIELDS entry, so "
                    "NEG002 cannot tell whether this constraint's fields are populated -- an empty one "
                    "would validate clean here and then never fire at score time.",
                    f"Add {kind!r} to spec_validator.REQUIRED_NEG_FIELDS, naming the fields its "
                    "eval_neg_constraints branch reads.",
                )
            )
            continue
        missing = _missing_neg_fields(kind, c)
        if missing:
            out.append(
                Finding(
                    "NEG002",
                    ERROR,
                    label,
                    f"kind {kind!r} is missing/empty required field(s) {missing} -- it evaluates over an "
                    "empty set and can never fire.",
                    f"Populate {missing} on the constraint dict.",
                )
            )
    for label, n in sorted(labels.items()):
        if n > 1:
            out.append(
                Finding(
                    "NEG004",
                    WARN,
                    label,
                    f"{n} negative constraints share this label -- verdicts are indistinguishable in the "
                    "manifest.",
                    "Give each constraint a unique label.",
                )
            )
    return out


def _missing_neg_fields(kind: str, c: dict) -> list[str]:
    """Required fields of ``kind`` that ``c`` leaves empty. Total: an unlisted kind requires nothing."""
    return [f for f in REQUIRED_NEG_FIELDS.get(kind, ()) if not c.get(f)]


def rule_neg_fireable(spec: TaskSpec) -> list[Finding]:
    """NEG003: PROVE each constraint can fire, by feeding the evaluator a synthetic violation."""
    out: list[Finding] = []
    for c in spec.negs:
        if str(c.get("kind")) not in IMPLEMENTED_NEG_KINDS:
            continue
        label = str(c.get("label", c.get("kind")))
        probe = _synthetic_violation(c)
        if probe is None:
            out.append(
                Finding(
                    "NEG003",
                    ERROR,
                    label,
                    "no violating world state can be constructed from this constraint's own vocabulary "
                    "-- it is unfireable by construction and will report ok=True forever.",
                    "Populate the constraint's object/container/unit sets.",
                )
            )
            continue
        verdicts = eval_neg_constraints([c], probe)
        if verdicts and verdicts[0]["ok"]:
            out.append(
                Finding(
                    "NEG003",
                    ERROR,
                    label,
                    "the evaluator reports ok=True even for a synthetic violating relation set -- this "
                    "constraint is DEAD (the 'sink trap': it can never report a failure).",
                    "Fix the constraint's fields, or the evaluator branch for this kind.",
                )
            )
    return out


def _synthetic_violation(c: dict) -> set | None:
    """A relation set that MUST violate ``c``. None when the constraint's own vocabulary is too thin."""
    kind = str(c.get("kind"))
    if kind == "not_inside":
        objs, conts = sorted(c.get("objs") or []), sorted(c.get("containers") or [])
        return {(Pred.INSIDE, objs[0], conts[0])} if objs and conts else None
    if kind == "all_placed":
        return set() if c.get("objs") else None  # nothing placed -> every obj missing
    if kind == "no_share_group":
        return _share_group_violation(c)
    if kind == "capacity":
        caps = c.get("cap") or {}
        if not caps:
            return None
        unit = sorted(caps)[0]
        return {
            (Pred.INSIDE, f"__probe{i}__", unit) for i in range(int(caps[unit]) + 1)
        }
    return None


def _share_group_violation(c: dict) -> set | None:
    gof = c.get("group_of") or {}
    units = sorted(c.get("units") or [])
    by_group: dict[str, list[str]] = {}
    for obj, grp in gof.items():
        by_group.setdefault(str(grp), []).append(str(obj))
    for pair in c.get("incompatible") or []:
        g1, g2 = str(pair[0]), str(pair[1])
        if by_group.get(g1) and by_group.get(g2) and units:
            u = units[0]
            return {
                (Pred.INSIDE, sorted(by_group[g1])[0], u),
                (Pred.INSIDE, sorted(by_group[g2])[0], u),
            }
    return None


def rule_neg_grounding(spec: TaskSpec) -> list[Finding]:
    """NEG005-007: every name a negative mentions must be a declared entity/container."""
    out: list[Finding] = []
    entities, containers = spec.entity_keys(), set(spec.containers)
    for c in spec.negs:
        label = str(c.get("label", c.get("kind")))
        for obj in sorted(c.get("objs") or []) + sorted((c.get("group_of") or {})):
            if obj not in entities:
                out.append(_undeclared(label, str(obj), "object", "ITEMS/PARTS"))
        named = sorted(c.get("containers") or []) + sorted(c.get("units") or [])
        named += sorted((c.get("cap") or {}))
        for cont in named:
            if cont not in containers:
                out.append(_undeclared(label, str(cont), "container", "CONTAINERS"))
    return out + _rule_neg_groups(spec)


def _undeclared(label: str, name: str, what: str, where: str) -> Finding:
    return Finding(
        "NEG005",
        ERROR,
        name,
        f"negative {label!r} names {what} {name!r}, which is not declared in {where}. An unregistered "
        "container makes the constraint UNFIREABLE -- it reports ok=True forever (the sink trap).",
        f"Declare {name!r} in {where}, or fix the typo.",
    )


def _rule_neg_groups(spec: TaskSpec) -> list[Finding]:
    """NEG007: an incompatibility pair whose groups no declared object carries can never fire."""
    out: list[Finding] = []
    for c in spec.negs:
        if c.get("kind") != "no_share_group":
            continue
        present = {str(g) for g in (c.get("group_of") or {}).values()}
        label = str(c.get("label", c.get("kind")))
        for pair in c.get("incompatible") or []:
            missing = [str(g) for g in pair if str(g) not in present]
            if missing:
                out.append(
                    Finding(
                        "NEG007",
                        WARN,
                        f"{label}:{'+'.join(str(g) for g in pair)}",
                        f"no declared object carries group(s) {missing} -- this incompatibility pair can "
                        "never fire.",
                        "Drop the pair, or assign the group to an object.",
                    )
                )
        pairs = {frozenset((str(a), str(b))) for a, b in (c.get("incompatible") or [])}
        out += _reachable_compatible_pairs(spec, c, label, pairs)
    return out


def _reachable_compatible_pairs(
    spec: TaskSpec, c: dict, label: str, pairs: set
) -> list[Finding]:
    """NEG008: only report a NON-incompatible group pair the spec itself co-locates (start or answer key).

    Enumerating every unordered pair of declared groups is noise; a pair matters when the task's OWN
    start state or reference solution puts both groups in one unit and no rule objects.
    """
    gof = {str(k): str(v) for k, v in (c.get("group_of") or {}).items()}
    units = {str(u) for u in (c.get("units") or [])}
    per_unit: dict[str, set[str]] = {}
    for key, item in spec.items.items():
        place = str(item.get("place", ""))
        if key in gof and place.startswith("in:"):
            per_unit.setdefault(place[3:], set()).add(gof[key])
    for g in spec.goals:
        if g.pred == "inside" and g.subject in gof and g.target:
            per_unit.setdefault(g.target, set()).add(gof[g.subject])
    out: list[Finding] = []
    for unit in sorted(per_unit):
        if units and unit not in units:
            continue
        groups = sorted(per_unit[unit])
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1 :]:
                if frozenset((g1, g2)) in pairs:
                    continue
                out.append(
                    Finding(
                        "NEG008",
                        INFO,
                        f"{label}:{unit}:{g1}+{g2}",
                        f"the spec's own start state / answer key co-locates groups {g1!r} and {g2!r} in "
                        f"{unit!r}, and that pair is not in `incompatible` -- so this co-location is never "
                        "flagged, whoever causes it.",
                        "Confirm it is intended; otherwise add the pair to `incompatible`.",
                    )
                )
    return out


def rule_containers(spec: TaskSpec) -> list[Finding]:
    """CON001-005: scene-sourced containers resolve, imported ones are fully specified, overlap check."""
    out: list[Finding] = []
    scene_cats: dict[str, list[str]] = {}
    for key, c in sorted(spec.containers.items()):
        source = str(c.get("source", "import"))
        if source == "import" and not c.get("model"):
            out.append(
                Finding(
                    "CON004",
                    ERROR,
                    key,
                    "source='import' but no model is declared -- behavior_import cannot resolve a baked "
                    "USD and drops it silently.",
                    "Add the model id, or switch to source='scene'.",
                )
            )
        if source == "scene":
            scene_cats.setdefault(str(c.get("cat")), []).append(key)
    out += _rule_scene_supply(spec, scene_cats)
    out.append(
        Finding(
            "CON005",
            INFO,
            spec.name,
            "container xy-footprint overlap is NOT checkable offline -- imported containers are placed by "
            "_find_slot at build time, so no declared coordinate exists. Kernel ready: "
            "container_xy_overlap_pairs({key: world_aabb}).",
            "REQUEST: call it after container registration with each Role.CONTAINER's world AABB; record "
            "scene_integrity.container_xy_overlaps and mark affected INSIDE verdicts ambiguous.",
            needs_stage=True,
        )
    )
    return out


def _rule_scene_supply(
    spec: TaskSpec, scene_cats: dict[str, list[str]]
) -> list[Finding]:
    """CON001-003: the scene must actually contain enough instances of each scene-sourced category."""
    if not scene_cats or not spec.scene:
        return []
    try:
        counts = asset_preflight.scene_category_counts(spec.scene)
    except (OSError, ValueError, KeyError) as exc:
        return [
            Finding(
                "CON001",
                WARN,
                spec.scene,
                f"could not read the scene layout to verify scene-sourced containers: {exc}",
                "Check the BEHAVIOR scene JSON path.",
            )
        ]
    out: list[Finding] = []
    for cat, keys in sorted(scene_cats.items()):
        have = counts.get(cat, 0)
        if have == 0:
            out.append(
                Finding(
                    "CON001",
                    ERROR,
                    f"{','.join(keys)} ({cat})",
                    f"scene {spec.scene!r} contains no loadable object of category {cat!r} -- the "
                    "container never registers, so every goal on it is unmet and every negative on it is "
                    "UNFIREABLE (reports ok=True forever).",
                    "Pick a category the scene has, or import the unit instead of sourcing it.",
                )
            )
        elif have < len(keys):
            out.append(
                Finding(
                    "CON002",
                    ERROR,
                    f"{','.join(keys)} ({cat})",
                    f"{len(keys)} containers claim category {cat!r} but the scene has only {have} "
                    "instance(s) -- at least one cannot be assigned.",
                    "Import the surplus units, or reduce the declarations.",
                )
            )
        elif len(keys) > 1 and (defect := _pick_defect(spec, keys)):
            out.append(
                Finding(
                    "CON003",
                    WARN,
                    f"{','.join(keys)} ({cat})",
                    f"{len(keys)} containers resolve by the same category {cat!r} and are not totally "
                    f"ordered ({defect}); which of the scene's {have} instances becomes which is decided "
                    "at build time and is not reproducible from the spec, while the answer key is fixed.",
                    "Give each an explicit `pick`: {'by': <selection rule>, 'rank': <distinct int>}, and "
                    "make the builder assign the ranked instances to it.",
                )
            )
    n_scene = sum(len(v) for v in scene_cats.values())
    out.append(
        Finding(
            "CON006",
            INFO,
            spec.name,
            f"{n_scene} scene-sourced container(s) pass the category check, but REGISTRATION (Role.CONTAINER "
            "with a non-degenerate AABB) is only provable on a built stage. An unregistered container is the "
            "sink trap: its negatives report ok=True forever.",
            "REQUEST: after registration, assert every container named in goals/negatives is in "
            "scene.entities with Role.CONTAINER; record scene_integrity.containers_missing.",
            needs_stage=True,
        )
    )
    return out


def _pick_defect(spec: TaskSpec, keys: list[str]) -> str | None:
    """Why same-category containers ``keys`` are still ambiguous; None when ``pick`` totally orders them."""
    picks = {k: (spec.containers[k].get("pick") or {}) for k in keys}
    undeclared = sorted(k for k, p in picks.items() if not p)
    if undeclared:
        return f"no `pick` disambiguator on {undeclared}"
    rules = sorted({str(p.get("by")) for p in picks.values()})
    if len(rules) != 1:
        return f"conflicting `pick.by` rules {rules}"
    ranks = [p.get("rank") for p in picks.values()]
    if any(not isinstance(r, int) or isinstance(r, bool) for r in ranks):
        return f"`pick.rank` must be an int (got {ranks})"
    if len(set(ranks)) != len(ranks):
        return f"duplicate `pick.rank` {sorted(ranks)}"
    return None


def rule_scene(spec: TaskSpec) -> list[Finding]:
    """SCENE001-002: the declared scene must exist on disk and expose a readable layout."""
    if not spec.scene:
        return [
            Finding(
                "SCENE001",
                ERROR,
                spec.name,
                "no SCENE declared -- nothing can resolve scene-sourced containers or pre-flight assets.",
                "Set SCENE to a BEHAVIOR scene key.",
            )
        ]
    try:
        n = len(asset_preflight.scene_loadable_models(spec.scene))
    except (OSError, ValueError, KeyError) as exc:
        return [
            Finding(
                "SCENE002",
                ERROR,
                spec.scene,
                f"declared SCENE is not a readable BEHAVIOR scene: {type(exc).__name__}: {exc}",
                "Fix the scene key (it must name a directory under the BEHAVIOR scenes root).",
            )
        ]
    return (
        []
        if n
        else [
            Finding(
                "SCENE002",
                ERROR,
                spec.scene,
                "the declared SCENE has no loadable DatasetObjects.",
                "Fix the scene key.",
            )
        ]
    )


def rule_carcass(spec: TaskSpec) -> list[Finding]:
    """DAG001-003: assembly precedence must be acyclic, rooted, and WRITTEN in a buildable order."""
    edges = spec.carcass_edges
    if not edges:
        return []
    out: list[Finding] = []
    children = {c for c, _p in edges}
    for c, p in edges:
        for name in (c, p):
            if name not in spec.parts:
                out.append(
                    Finding(
                        "DAG003",
                        ERROR,
                        name,
                        f"carcass edge ({c} -> {p}) references {name!r}, which is not a declared PART.",
                        f"Declare {name!r} in PARTS, or fix the typo.",
                    )
                )
    attached = {
        p for _c, p in edges if p not in children
    }  # roots are parents that are never children
    for c, p in edges:
        if p not in attached:
            out.append(
                Finding(
                    "DAG002",
                    ERROR,
                    f"{c}->{p}",
                    f"edge is written before its parent {p!r} is attached -- the parent's Female point has "
                    "no world pose yet, so the program is unsatisfiable. The edge list is never "
                    "topologically sorted; order is load-bearing.",
                    "Reorder the edge list (or sort it topologically in the builder).",
                )
            )
        attached.add(c)
    if len(attached) < len(children) + 1:
        out.append(
            Finding(
                "DAG001",
                ERROR,
                spec.name,
                "the assembly graph is cyclic or has no root -- no build order exists.",
                "Make the edge set a tree rooted at one part.",
            )
        )
    return out


def rule_assets(spec: TaskSpec) -> list[Finding]:
    """ASSET/LABEL: delegate to the pre-flight (skipped, loudly, when the bake cache is absent)."""
    if not os.path.isdir(asset_preflight.CACHE_USD):
        return [
            Finding(
                "ASSET000",
                INFO,
                asset_preflight.CACHE_USD,
                "bake cache not present -- asset and label checks skipped.",
                "Set INTERACTIVEWORLD_BEHAVIOR_CACHE, or pass --no-assets.",
            )
        ]
    sev = {"CRITICAL": ERROR, "IMPORTANT": WARN, "COSMETIC": INFO}
    return [
        Finding(f.rule, sev[f.criticality], f.entity, f.message, f.fix)
        for f in asset_preflight.spec_findings(spec)
    ]


def validate(spec: TaskSpec, with_assets: bool = True) -> list[Finding]:
    """Run every rule. Order is stable so two runs of the same spec diff cleanly."""
    rules = [
        rule_shape,
        rule_deferred,
        rule_goal_vocabulary,
        rule_pre_satisfied,
        rule_answer_key,
        rule_neg_kinds,
        rule_neg_fireable,
        rule_neg_grounding,
        rule_containers,
        rule_carcass,
    ]
    if os.path.isdir(asset_preflight.SCENE_ROOT):
        rules.append(rule_scene)
    if with_assets:
        rules.append(rule_assets)
    out: list[Finding] = []
    for rule in rules:
        out += rule(spec)
    return out


# ---- reporting ---------------------------------------------------------------------------------------
def _counts(findings: list[Finding]) -> dict[str, int]:
    out = {ERROR: 0, WARN: 0, INFO: 0}
    for f in findings:
        out[f.severity] = out.get(f.severity, 0) + 1
    return out


def report(spec: TaskSpec, findings: list[Finding]) -> None:
    print(
        f"\n=== {spec.name} (scene={spec.scene or '?'}, {len(spec.goals)} goal predicate(s), "
        f"{len(spec.negs)} negative(s), returns a {spec.return_shape}) ==="
    )
    order = {ERROR: 0, WARN: 1, INFO: 2}
    for f in sorted(
        findings, key=lambda x: (order.get(x.severity, 3), x.rule, x.entity)
    ):
        tag = " [needs a built stage]" if f.needs_stage else ""
        print(
            f"  {f.severity:5s} {f.rule:9s} {f.entity}{tag}\n        {f.message}\n        FIX: {f.fix}"
        )
    c = _counts(findings)
    print(f"  -- {c[ERROR]} ERROR / {c[WARN]} WARN / {c[INFO]} INFO")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline benchmark task-spec validator (no Isaac)."
    )
    ap.add_argument(
        "--spec", action="append", default=[], help="task module name, repeatable"
    )
    ap.add_argument(
        "--all", action="store_true", help=f"validate {', '.join(SHIPPED_SPECS)}"
    )
    ap.add_argument(
        "--no-assets", action="store_true", help="skip bake-cache/label checks"
    )
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args(argv)
    names = list(args.spec) + (list(SHIPPED_SPECS) if args.all else [])
    if not names:
        return _selftest()
    payload: dict[str, list[dict]] = {}
    rc = 0
    for name in names:
        spec = load_spec(name)
        findings = validate(spec, with_assets=not args.no_assets)
        payload[name] = [f.as_dict() for f in findings]
        if not args.json:
            report(spec, findings)
        rc = 1 if any(f.severity == ERROR for f in findings) else rc
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


# =====================================================================================================
# Pure selftest (no Isaac, no bake cache): kernels + NEGATIVE specs that MUST be caught.
# =====================================================================================================
def _bad_spec(**over) -> TaskSpec:
    """A minimal well-formed spec, overridable field-by-field to inject one defect at a time."""
    base = {
        "name": "probe",
        "scene": "Rs_int",  # a real scene key: the base spec must be clean under every rule
        "goals": [Goal("inside", "cup", "bin")],
        "negs": [{"kind": "all_placed", "objs": ["cup"], "label": "placed"}],
        "items": {"cup": {"cat": "cup", "model": "aaaaaa", "place": "table"}},
        "containers": {
            "bin": {"source": "import", "cat": "trash_can", "model": "bbbbbb"}
        },
    }
    base.update(over)
    return TaskSpec(**base)


def _rules_selftest(chk) -> None:
    ids = lambda fs: {f.rule for f in fs}  # noqa: E731
    clean = validate(_bad_spec(), with_assets=False)
    chk(
        not [f for f in clean if f.severity == ERROR],
        "a well-formed spec produces no ERROR",
    )

    typo = _bad_spec(goals=[Goal("inside", "cup", "bni")])
    chk(
        "GOAL006" in ids(validate(typo, with_assets=False)),
        "NEGATIVE: typo'd goal container -> GOAL006",
    )
    typo_neg = _bad_spec(
        negs=[
            {
                "kind": "not_inside",
                "objs": ["cup"],
                "containers": ["snik"],
                "label": "x",
            }
        ]
    )
    chk(
        "NEG005" in ids(validate(typo_neg, with_assets=False)),
        "NEGATIVE: typo'd sink container -> NEG005",
    )

    unknown = _bad_spec(
        negs=[{"kind": "all_shelved", "objs": ["cup"], "label": "shelved"}]
    )
    chk(
        "NEG001" in ids(validate(unknown, with_assets=False)),
        "NEGATIVE: unimplemented neg kind -> NEG001",
    )

    pre = _bad_spec(items={"cup": {"cat": "cup", "model": "a", "place": "in:bin"}})
    chk(
        "GOAL004" in ids(validate(pre, with_assets=False)),
        "NEGATIVE: goal satisfied at t=0 -> GOAL004",
    )

    chk(
        "GOAL001" in ids(validate(_bad_spec(goals=[]), with_assets=False)),
        "NEGATIVE: empty goal -> GOAL001",
    )
    chk(
        "GOAL003"
        in ids(
            validate(_bad_spec(goals=[Goal("on", "cup", "bin")]), with_assets=False)
        ),
        "NEGATIVE: ON goal (never emitted) -> GOAL003",
    )
    dead = _bad_spec(
        negs=[{"kind": "not_inside", "objs": ["cup"], "containers": [], "label": "d"}]
    )
    chk(
        "NEG002" in ids(validate(dead, with_assets=False)),
        "NEGATIVE: empty container set -> NEG002",
    )
    over = _bad_spec(negs=[{"kind": "capacity", "cap": {"bin": 0}, "label": "cap"}])
    chk(
        "GOAL011" in ids(validate(over, with_assets=False)),
        "NEGATIVE: answer key over capacity -> GOAL011",
    )
    chk(
        "SCENE002" in ids(rule_scene(_bad_spec(scene="__no_such_scene__"))),
        "NEGATIVE: nonexistent SCENE key -> SCENE002",
    )
    chk(
        "SCENE001" in ids(rule_scene(_bad_spec(scene=""))),
        "NEGATIVE: missing SCENE -> SCENE001",
    )
    _rules_selftest_pick(chk, ids)
    _rules_selftest_deferred(chk, ids)


# A scene with THREE fridges, so the CON003 probes below reach the same-category branch rather than
# CON002 (fewer instances than declarations).
_TWIN_SCENE = "restaurant_hotel"


def _twin_fridges(**pick) -> TaskSpec:
    """Two containers on one scene category -- the shape CON003 exists to catch."""
    base = {"source": "scene", "cat": "fridge", "room": "kitchen"}
    return _bad_spec(
        scene=_TWIN_SCENE,
        goals=[Goal("inside", "cup", "raw")],
        containers={
            "raw": base | {"pick": {"by": "nearest_room_anchor", "rank": 0} | pick},
            "rte": base | {"pick": {"by": "nearest_room_anchor", "rank": 1}},
        },
    )


def _rules_selftest_pick(chk, ids) -> None:
    """CON003 must fire on an undeclared/duplicated ordering and stay silent on a total one."""
    plain = {"source": "scene", "cat": "fridge", "room": "kitchen"}
    bare = _bad_spec(
        scene=_TWIN_SCENE,
        goals=[Goal("inside", "cup", "raw")],
        containers={"raw": dict(plain), "rte": dict(plain)},
    )
    chk(
        "CON003" in ids(_rule_scene_supply(bare, {"fridge": ["raw", "rte"]})),
        "NEGATIVE: two containers on one scene category, no disambiguator -> CON003",
    )
    dup = _twin_fridges(rank=1)
    chk(
        "CON003" in ids(_rule_scene_supply(dup, {"fridge": ["raw", "rte"]})),
        "NEGATIVE: duplicate pick.rank is not a total order -> CON003",
    )
    conflict = _twin_fridges(by="farthest_from_hood")
    chk(
        "CON003" in ids(_rule_scene_supply(conflict, {"fridge": ["raw", "rte"]})),
        "NEGATIVE: conflicting pick.by rules -> CON003",
    )
    chk(
        "CON003"
        not in ids(_rule_scene_supply(_twin_fridges(), {"fridge": ["raw", "rte"]})),
        "a declared, distinctly-ranked pick clears CON003",
    )


def _rules_selftest_deferred(chk, ids) -> None:
    """SHAPE005 must report a deferred block once, and stay silent when there is none."""
    chk(not rule_deferred(_bad_spec()), "no DEFERRED_SCORING -> no SHAPE005")
    empty = _bad_spec(
        deferred={"why": "not built", "provision": [], "neg_constraints": []}
    )
    chk(
        not rule_deferred(empty),
        "an EMPTY deferred block is not reported (nothing is withheld)",
    )
    real = _bad_spec(
        deferred={
            "why": "needs an evaluator that does not exist",
            "provision": ["a", "b"],
            "neg_constraints": [{"kind": "all_shelved", "label": "x"}],
        }
    )
    found = rule_deferred(real)
    chk("SHAPE005" in ids(found), "NEGATIVE: a non-empty deferred block -> SHAPE005")
    chk(len(found) == 1, "SHAPE005 is reported ONCE per spec, not once per surface")
    chk(
        "all_shelved" in found[0].message and "provision (2)" in found[0].message,
        "SHAPE005 names the withheld surfaces AND the unimplemented kinds",
    )


def _kernel_selftest(chk) -> None:
    boxes = {
        "bin_a": (0.0, 0.0, 0.0, 1.0, 1.0, 0.5),
        "bin_b": (0.5, 0.5, 0.0, 1.5, 1.5, 0.5),  # overlaps a in xy AND z
        "bin_c": (5.0, 5.0, 0.0, 6.0, 6.0, 0.5),  # far away
        "shelf": (0.0, 0.0, 2.0, 1.0, 1.0, 2.5),  # same xy as a, disjoint z
    }
    pairs = container_xy_overlap_pairs(boxes)
    chk(
        [(a, b) for a, b, _ in pairs] == [("bin_a", "bin_b")],
        "NEGATIVE: overlapping container footprints -> exactly the (bin_a, bin_b) pair",
    )
    chk(abs(pairs[0][2] - 0.25) < 1e-9, "overlap area is reported (0.25 m^2)")
    chk(
        not container_xy_overlap_pairs(boxes, margin=1.0),
        "a generous margin suppresses the pair",
    )

    goals = [Goal("inside", "cup", "bin"), Goal("inside", "pen", "bin")]
    t0 = {(Pred.INSIDE, "cup", "bin")}
    chk(
        [g.subject for g in pre_satisfied_predicates(goals, t0)] == ["cup"],
        "NEGATIVE: pre-satisfied predicate kernel reports exactly [cup]",
    )


def _evaluator_selftest(chk) -> None:
    """The IMPLEMENTED_NEG_KINDS table must match the LIVE evaluator, in both directions."""
    for kind, c in (
        (
            "not_inside",
            {"kind": "not_inside", "objs": ["a"], "containers": ["u"], "label": "l"},
        ),
        ("all_placed", {"kind": "all_placed", "objs": ["a"], "label": "l"}),
        (
            "no_share_group",
            {
                "kind": "no_share_group",
                "group_of": {"a": "g1", "b": "g2"},
                "incompatible": [["g1", "g2"]],
                "units": ["u"],
                "label": "l",
            },
        ),
        ("capacity", {"kind": "capacity", "cap": {"u": 1}, "label": "l"}),
    ):
        probe = _synthetic_violation(c)
        chk(probe is not None, f"a synthetic violation exists for kind {kind!r}")
        chk(
            not eval_neg_constraints([c], probe)[0]["ok"],
            f"the evaluator FIRES for kind {kind!r}",
        )
    bogus = {"kind": "__not_a_kind__", "label": "l"}
    verdict = eval_neg_constraints([bogus], set())[0]
    chk(
        not verdict["ok"]
        and verdict["offenders"] == ["unimplemented_kind:__not_a_kind__"],
        "the evaluator reports an UNKNOWN kind VIOLATED, never a silent pass (NEG001 is now "
        "belt-and-braces: it names the kind before the render, not after)",
    )
    chk(
        IMPLEMENTED_NEG_KINDS is NEG_KINDS,
        "the kind table is the evaluator's own registry, not a local copy that can drift",
    )
    unlisted = sorted(set(IMPLEMENTED_NEG_KINDS) - set(REQUIRED_NEG_FIELDS))
    stale = sorted(set(REQUIRED_NEG_FIELDS) - set(IMPLEMENTED_NEG_KINDS))
    chk(
        not unlisted and not stale,
        "REQUIRED_NEG_FIELDS covers the registry in BOTH directions"
        + (f" (unlisted={unlisted}, stale={stale})" if (unlisted or stale) else ""),
    )


def _fifth_kind_selftest(chk) -> None:
    """A kind registered in NEG_KINDS before REQUIRED_NEG_FIELDS catches up (e.g. mid-authoring).

    Widening the registry is the only way to reach that state while the two tables agree, so the probe kind
    is injected rather than declared. The contract: every rule still RUNS and the gap is REPORTED (NEG009);
    a validator that dies on a spec it was written to diagnose protects nothing.
    """
    global IMPLEMENTED_NEG_KINDS

    saved = IMPLEMENTED_NEG_KINDS
    IMPLEMENTED_NEG_KINDS = frozenset(set(saved) | {"__fifth_kind__"})
    try:
        spec = _bad_spec(negs=[{"kind": "__fifth_kind__", "label": "fifth"}])
        rules = {f.rule for f in validate(spec, with_assets=False)}
    finally:
        IMPLEMENTED_NEG_KINDS = saved
    chk(
        "NEG009" in rules,
        "NEGATIVE: a fifth registered kind with no required-fields entry -> NEG009, not a KeyError",
    )
    chk(
        "NEG001" not in rules,
        "a registered fifth kind is not ALSO mis-reported as unimplemented (NEG001)",
    )
    chk(
        IMPLEMENTED_NEG_KINDS is NEG_KINDS,
        "the injected registry is restored after the probe",
    )


def _selftest() -> int:
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    _kernel_selftest(chk)
    _evaluator_selftest(chk)
    _fifth_kind_selftest(chk)
    _rules_selftest(chk)
    for name in SHIPPED_SPECS:
        spec = load_spec(name)
        chk(
            bool(spec.goals),
            f"{name}: normalises to a non-empty goal list ({len(spec.goals)})",
        )
        chk(
            bool(validate(spec, with_assets=False) is not None),
            f"{name}: every rule runs without raising",
        )
    print(
        f"[spec_validator] selftest {'PASS' if ok else 'FAIL'} ({len(SHIPPED_SPECS)} shipped specs loaded)"
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

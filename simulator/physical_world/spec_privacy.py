#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Spec de-leak stage -- close the data-structure leaks a model-free (blank-frame) policy can exploit.

Runs inside the spec normalisation step as a pure function of the assembled specs plus
``DELEAK_SALT``: no clock, no ``random`` module state, no filesystem, so a second run of
the producer reproduces every byte.

* **L1** the unit-letter convention.  Container role keys are renamed to a neutral
  ``unit_<letter>`` family under a seeded permutation, and the permutation is chosen by a
  portfolio-level allocator so that "always answer the alphabetically-first unit" scores
  the chance level rather than 0.375.  The original key is kept in ``unit_roles`` (hidden)
  so the role a scene has to build is never lost.
* **L2** roster order.  ``roster.containers`` is re-ordered by a seeded permutation whose
  answer index is balanced across the portfolio.
* **L3** the elimination set.  Every field is labelled ``visible`` / ``hidden``, and where
  ``not_inside`` pinned the answer a plausible unlisted distractor container is added.
* **L4** the literal class map and answer-naming prose: both live in hidden fields, and the
  visible projection redacts every container token out of the prose it does publish.
* **L5** the chance level, recomputed and written back as the canonical value.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
from typing import Any, Callable

DELEAK_SALT: str = "interactiveworld/spec-privacy/1.0"

# THE frozen-world constant. Every world draw in the producers -- the label/date cues, the
# stand-in environment in ``constraint_resolver.synth_env``, and the builder's own
# appearance permutations -- is evaluated at this ONE value, once, at authoring time. A benchmark
# task has exactly one environment: object placement and the human's start pose are a property of
# the task, not of the run. Anything that still wants to differ run-to-run is a MODEL-side seed and
# does not belong in a spec. Kept next to ``DELEAK_SALT`` because it is the same kind of object --
# an authoring-time constant whose value is arbitrary but whose FIXEDNESS is load-bearing.
FROZEN_WORLD_SEED: int = 0

LETTERS: str = "abcdefghijklmnopqrstuvwxyz"
NONE_TOKEN: str = "NONE"
UNIT_PLACEHOLDER: str = "<unit>"
# The admission window's per-row tolerance (``floor <= chance + WINDOW_SLACK``), mirrored here so
# the allocators optimise the quantity the gate actually measures rather than a portfolio average.
WINDOW_SLACK: float = 0.25

# task_world_builder.py binds these nine rows BY ROLE KEY (``UnitRecipe`` / ``PreloadRecipe`` /
# ``NumberRecipe``) and asserts "the recipe covers exactly the spec roster", so renaming
# their containers would fail the build.  Their permutation is pinned to the identity and
# the portfolio balance is carried by the other 118 specs.
BUILDER_PINNED: frozenset[str] = frozenset(
    {
        "T1-CHEM-02",
        "T1-DINR-01",
        "T1-GCAF-01",
        "T3-GROC-02",
        "T3-HALL-01",
        "T6-HSF-02",
        "T6-RDIN-01",
        "T6-RHOT-01",
        "T6-RHOT-02",
    }
)

# A pinned row whose role keys were re-keyed BY HAND, in lockstep with its task_world_builder recipe,
# because the seeded permutation cannot reach it. Authored rather than salted so it can never
# drift: ``T6-RHOT-02``'s two racks are not interchangeable (one IS the overflow rack), and with
# the answer keyed ``unit_a`` the alphabetically-first constant scored a free 1.00.
PINNED_RENAME: dict[str, dict[str, str]] = {
    "T6-RHOT-02": {"unit_a": "unit_b", "unit_overflow": "unit_a"},
}

VISIBLE_FIELDS: frozenset[str] = frozenset(
    {
        "spec_version",
        "task_id",
        "family",
        "template",
        "scene",
        "wave",
        "difficulty",
        "roster",
        "chance",
        "visibility",
        "redact_in_visible_text",
    }
)
VISIBLE_DESIGN: frozenset[str] = frozenset({"premise", "assistant_must"})


def _rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join((DELEAK_SALT, *parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _push(seq: list[str], key: str | None) -> None:
    if key and key not in seq:
        seq.append(key)


def menu_of(spec: dict) -> list[str]:
    """Every target the assistant may legally name, in a deterministic order."""
    out: list[str] = []
    roster = spec["roster"]
    for name in (
        list(roster["containers"])
        + list(roster.get("nested_containers") or [])
        + [g["target"] for g in spec["answer_key"]]
    ):
        _push(out, name)
    return out


def key_targets(spec: dict) -> list[str]:
    return [str(g["target"]) for g in spec["answer_key"]]


def pinned_objects(spec: dict) -> set[str]:
    return {str(g["subject"]) for g in spec["goals"] if g.get("pred") == "inside"}


def free_objects(spec: dict) -> list[str]:
    """Roster objects no INSIDE goal pins -- the ones a new container gives a new option."""
    pinned = pinned_objects(spec)
    return [o for o in spec["roster"]["objects"] if o not in pinned]


def forbidden_targets(spec: dict) -> set[str]:
    out: set[str] = set()
    for neg in spec["negs"]:
        if neg["kind"] == "not_inside":
            out |= {str(c) for c in neg.get("containers") or []}
    return out


# ---- L5a: the arms of one row share one world, so they must share one menu -----------------------
def union_arm_menus(specs: list[dict]) -> dict[str, list[str]]:
    """Give every arm of a split row the full container menu of the world it is filmed in.

    An arm whose only literal target is the HOLD bay would otherwise get a one-container
    roster: chance is 1.00 and the row measures nothing.  The arms are the SAME scene, so the
    union is a statement of fact, not a new difficulty.  Skipped where the spec has a free
    object, since there an extra container would also be an extra legal assignment.
    """
    by_src: dict[str, list[dict]] = {}
    for spec in specs:
        by_src.setdefault(str(spec["source_task_id"]), []).append(spec)
    added: dict[str, list[str]] = {}
    for _src, group in sorted(by_src.items()):
        if len(group) < 2:
            continue
        union: list[str] = []
        for spec in group:
            for name in spec["roster"]["containers"]:
                _push(union, name)
        for spec in group:
            if free_objects(spec):
                continue
            miss = [c for c in union if c not in spec["roster"]["containers"]]
            if miss:
                spec["roster"]["containers"].extend(miss)
                added[str(spec["task_id"])] = miss
    return added


# ---- L3b: a plausible but unlisted distractor ----------------------------------------------------
def _needs_distractor(spec: dict) -> str:
    """Why this spec needs one more container -- '' when it does not."""
    menu, targets = menu_of(spec), set(key_targets(spec))
    if len(menu) <= 1:
        return "degenerate menu: one container, so every constant policy scores 1.00"
    forb = forbidden_targets(spec)
    if forb and set(menu) - forb == targets and len(targets) == 1:
        return "not_inside pinned the answer by elimination"
    return ""


def add_distractors(
    specs: list[dict],
    rebuild: Callable[[dict], None],
) -> dict[str, dict[str, str]]:
    """Add one unlisted decoy container wherever elimination was decisive.

    The decoy is excluded for the FREE objects only (``not_inside``), never for an object an
    INSIDE goal already pins, so the wrong-answer check still enumerates exactly one legal
    assignment while the answer is no longer deducible by crossing off the listed units.
    """
    out: dict[str, dict[str, str]] = {}
    for spec in specs:
        why = _needs_distractor(spec)
        if not why or str(spec["task_id"]) in BUILDER_PINNED:
            continue
        name = "unit_distractor_1"
        while name in menu_of(spec) or name in spec["roster"]["objects"]:
            name += "x"
        spec["roster"]["containers"].append(name)
        spec["distractor_containers"] = [*spec.get("distractor_containers", []), name]
        free = free_objects(spec)
        if free:
            spec["negs"].append(
                {
                    "kind": "not_inside",
                    "objs": list(free),
                    "containers": [name],
                    "label": "not_inside_distractor",
                }
            )
            _push(spec["neg_kinds_declared"], "not_inside")
            rebuild(spec)
        out[str(spec["task_id"])] = {"container": name, "why": why}
    return out


# ---- L1: the letter suffix is not the answer -----------------------------------------------------
def _renamable(spec: dict) -> list[str]:
    """Containers whose key is a pure role handle: everything except a nested container,
    which is also a graspable and keeps its object name."""
    if str(spec["task_id"]) in BUILDER_PINNED:
        return []
    nested = set(spec["roster"].get("nested_containers") or [])
    return [c for c in spec["roster"]["containers"] if c not in nested]


def _alpha_first_options(spec: dict) -> list[tuple[str | None, float]]:
    """(token that gets the lexicographically first new name, resulting always-first score).

    The SET of new names does not depend on the permutation, so the rank-0 name is fixed and
    the only choice is which role key receives it.
    """
    renam = _renamable(spec)
    fixed = [c for c in menu_of(spec) if c not in renam]
    names = [f"unit_{LETTERS[i]}" for i in range(len(renam))]
    if not names:
        return []
    first = sorted(names + fixed)[0]
    if first in fixed:
        return []
    targets = key_targets(spec)
    opts: list[tuple[str | None, float]] = []
    if any(t not in targets for t in renam):
        opts.append((None, 0.0))
    for tok in sorted(set(targets) & set(renam)):
        opts.append((tok, targets.count(tok) / len(targets)))
    return opts


def assign_alpha_first(
    specs: list[dict], chance: dict[str, float]
) -> dict[str, tuple[str | None, float]]:
    """Pick, per spec, which role key gets ``unit_a`` so the portfolio sum of the
    always-alphabetically-first score tracks the portfolio sum of the chance level.

    A 1.00 outcome is refused whenever any alternative exists: on a single-target spec the
    policy is binary, so the specs that carry the rank-0 hits are the multi-target ones.

    The last tie-break is SEEDED, not lexical.  A lexical tie-break (``str(token)``) would, on a row
    whose targets all score the same, hand ``unit_a`` back to the alphabetically-first target --
    i.e. back to the name the row was authored with -- which is the L6 index-map leak.

    A PORTFOLIO sum sitting at chance is not the property the admission gate checks.  The gate is
    PER ROW, and a sum-only objective is free to pay for a row it drove to 0.00 by parking 0.60 on
    the next one.  The admissible set is therefore filtered to options at or under the window
    before the sum is balanced, so the leftover is SPREAD instead of concentrated.  The filter is a
    preference, not a constraint: a row whose every option is over the window (all of its targets
    are the same unit) falls back to the unfiltered options rather than silently reporting a
    number it did not achieve.
    """
    order = sorted(specs, key=lambda s: _rng("order", str(s["task_id"])).random())
    got = want = 0.0
    out: dict[str, tuple[str | None, float]] = {}
    for spec in order:
        tid = str(spec["task_id"])
        want += chance[tid]
        opts = _alpha_first_options(spec)
        if not opts:
            got += _forced_alpha_first(spec)
            continue
        safe = [o for o in opts if o[1] < 1.0] or opts
        admissible = [o for o in safe if o[1] <= chance[tid] + WINDOW_SLACK] or safe
        pick = min(
            admissible,
            key=lambda o: (
                abs(got + o[1] - want),
                o[1],
                _rng("first-tie", tid, str(o[0])).random(),
            ),
        )
        out[tid], got = pick, got + pick[1]
    return out


def _forced_alpha_first(spec: dict) -> float:
    """The always-first score of a spec whose rank-0 name cannot be moved."""
    menu, targets = menu_of(spec), key_targets(spec)
    if not menu or not targets:
        return 0.0
    return targets.count(sorted(menu)[0]) / len(targets)


# The rename itself is chosen by ``assign_rename`` (L6): ``assign_alpha_first`` fixes WHICH role key
# receives the rank-0 name, and every permutation of the remaining names scores the same
# always-alphabetically-first value, so L6 spends that leftover freedom without touching L1.


def _rewrite(node: Any, rx: re.Pattern[str], mapping: dict[str, str]) -> Any:
    if isinstance(node, str):
        return rx.sub(lambda m: mapping[m.group(0)], node)
    if isinstance(node, list):
        return [_rewrite(v, rx, mapping) for v in node]
    if isinstance(node, dict):
        return {
            (_rewrite(k, rx, mapping) if isinstance(k, str) else k): _rewrite(
                v, rx, mapping
            )
            for k, v in node.items()
        }
    return node


def apply_rename(spec: dict, mapping: dict[str, str]) -> dict:
    """One single-pass rewrite over every string and every dict key in the spec.

    Single pass on purpose: renaming a->c and then c->a in two passes would silently
    collapse two role keys onto one.
    """
    if not mapping:
        return spec
    rx = re.compile(
        r"\b("
        + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        + r")\b"
    )
    out = _rewrite(spec, rx, mapping)
    out["unit_roles"] = {v: k for k, v in sorted(mapping.items())}
    out["unit_rename"] = dict(sorted(mapping.items()))
    return out


# ---- L2: roster order is a seeded permutation ----------------------------------------------------
def assign_roster_index(specs: list[dict]) -> dict[str, int]:
    """Balanced target index for the answer inside ``roster.containers``.

    Round robin within each container-count bucket, so the index of the correct target is as
    close to uniform as an integer allocation allows instead of being 0 in nearly every spec.
    """
    buckets: dict[int, list[dict]] = {}
    for spec in specs:
        buckets.setdefault(len(spec["roster"]["containers"]), []).append(spec)
    out: dict[str, int] = {}
    for size, group in sorted(buckets.items()):
        order = sorted(group, key=lambda s: _rng("idx", str(s["task_id"])).random())
        for i, spec in enumerate(order):
            out[str(spec["task_id"])] = i % max(size, 1)
    return out


def permute_roster(spec: dict, index: int) -> None:
    conts = list(spec["roster"]["containers"])
    targets = [t for t in key_targets(spec) if t in conts]
    if not conts:
        return
    head = targets[0] if targets else conts[0]
    rest = [c for c in conts if c != head]
    rest.sort(key=lambda c: _rng("roster", str(spec["task_id"]), c).random())
    slot = min(index, len(conts) - 1)
    spec["roster"]["containers"] = rest[:slot] + [head] + rest[slot:]


# ---- L5b: the canonical chance level -------------------------------------------------------------
def _legal_assignment_count(spec: dict) -> int:
    """How many placements survive the goals and ``not_inside``: 1 per pinned object, and one
    branch per allowed container (plus 'nowhere') for every object nothing pins."""
    menu = menu_of(spec)
    total = 1
    for obj in free_objects(spec):
        banned = {
            str(c)
            for neg in spec["negs"]
            if neg["kind"] == "not_inside" and obj in (neg.get("objs") or [])
            for c in neg.get("containers") or []
        }
        total *= len([c for c in menu if c not in banned]) + 1
    return total


def chance_block(spec: dict) -> dict:
    """The canonical published chance level, recomputed from the legal answer alphabet."""
    menu = menu_of(spec)
    placed: set[str] = set()
    for neg in spec["negs"]:
        if neg["kind"] == "all_placed":
            placed |= {str(o) for o in neg.get("objs") or []}
    allow_none = not set(spec["roster"]["objects"]) <= placed
    alpha = len(menu) + (1 if allow_none else 0)
    n_obj = len(spec["roster"]["objects"])
    return {
        "item": 1.0 / max(alpha, 1),
        "episode": float(max(alpha, 1) ** -n_obj) if n_obj else 1.0,
        "alphabet": max(alpha, 1),
        "menu": len(menu),
        "allow_none": allow_none,
        "n_objects": n_obj,
        "n_key": len(spec["answer_key"]),
        "n_legal_assignments": _legal_assignment_count(spec),
        "degenerate": max(alpha, 1) <= 1,
        "source": "spec_privacy.chance_block -- 1/|legal answer alphabet| per scored item",
    }


# ---- L3a / L4: what a harness may show the assistant ---------------------------------------------
def visibility_block(spec: dict) -> dict[str, str]:
    """Every field labelled.  Fail-closed: anything not on the allow-list is hidden."""
    keys = sorted(set(spec) | {"visibility"})
    out = {k: ("visible" if k in VISIBLE_FIELDS else "hidden") for k in keys}
    for k in sorted(spec.get("design") or {}):
        out[f"design.{k}"] = "visible" if k in VISIBLE_DESIGN else "hidden"
    return out


def visible_projection(spec: dict) -> dict:
    """The ONLY part of a spec a harness may put in front of the assistant.

    Container tokens are redacted out of the prose it does publish: prose that names a correct
    target in ``premise`` / ``assistant_must`` with no wrong target beside it hands the answer
    to a blank-frame attacker.  The menu itself stays -- in ``roster``, permuted.
    """
    vis = spec.get("visibility") or visibility_block(spec)
    out = {k: v for k, v in spec.items() if vis.get(k) == "visible"}
    design = {
        k: v
        for k, v in (spec.get("design") or {}).items()
        if vis.get(f"design.{k}") == "visible"
    }
    tokens = list(spec.get("redact_in_visible_text") or [])
    if tokens:
        rx = re.compile(
            r"\b("
            + "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
            + r")\b"
        )
        design = {k: rx.sub(UNIT_PLACEHOLDER, str(v)) for k, v in design.items()}
    out["design"] = design
    return out


def answer_bearing_visible(spec: dict) -> list[str]:
    """The machine check: no answer-bearing field, and no answer token, in the projection."""
    proj = visible_projection(spec)
    bad = [
        f
        for f in ("goals", "negs", "answer_key", "answer_key_raw", "provenance")
        if f in proj
    ]
    findings = [f"{f} is present in the visible projection" for f in bad]
    text = " ".join(str(v) for v in proj.get("design", {}).values())
    for tok in sorted(set(key_targets(spec))):
        if re.search(rf"\b{re.escape(tok)}\b", text):
            findings.append(f"visible prose names the answer target {tok}")
    for neg in spec["negs"]:
        if (neg.get("group_of_expr") or {}).get("op") == "class_map" and "negs" in proj:
            findings.append(
                "the literal class map is reachable from the visible projection"
            )
    return findings


# ---- L6: the answer key is not the index map -----------------------------------------------------
# An attacker who has seen no pixel can still write down ``obj_1 -> unit_a, obj_2 -> unit_b``.  Two
# producer habits make that pay: an L1 tie-break on the role key's own NAME hands ``unit_a`` back
# to the key the row was authored with, and ``build_roster`` emits ``roster.objects`` in ANSWER-KEY
# order, so a row routing consecutive pairs to consecutive units prints its own key in the object
# list.  Both are countered by allocation, not by inversion: the score is steered onto the chance
# level rather than to zero, because a portfolio where the index map is always WRONG is just as
# exploitable as one where it is always right.
#
# The policy is scored as a FAMILY, not as the one map ``obj_1 -> unit_a``.  Steering only that
# single map onto chance just moves the leak onto another member, such as "shift the letters by two
# and read the menu backwards", which can then solve whole keys outright.  The family is the
# dihedral one an attacker can enumerate for free: objects in the roster's order or sorted (or
# either reversed), units in menu order or reversed, and every cyclic shift.
_MAX_SHIFT: int = 6
_OBJ_ORDERS: tuple[str, ...] = ("sorted", "sortedrev", "roster", "rosterrev")
# ``menu``  the attacker's own canonicalisation, sorted by name;
# ``roster`` the order a roster-listing prompt prints, i.e. ``roster.containers`` after L2;
# ``probe``  the order a shuffled-menu prompt prints, a hash shuffle keyed on the task id;
# ``cap``    the shorter list a capacity clause publishes.
# The two PRINTED orders must be scored too: an allocator that decorrelates the key only from the
# sorted menu, with no term for the orders the harness actually renders, is free to align the key
# with those instead, and the answer can then be read straight off the prompt.
_UNIT_SOURCES: tuple[str, ...] = ("menu", "roster", "probe", "cap")
_UNIT_ORDERS: tuple[str, ...] = ("fwd", "rev")
# ``spread`` shares the objects out over every unit (right when objects outnumber units);
# ``prefix`` pairs rank to rank and stops (right when they are equal, or units outnumber objects).
_BLOCK_MODES: tuple[str, ...] = ("spread", "prefix")
SUFFIX_POLICY: str = "suffix_match"


def _policy_names(obj_orders: tuple[str, ...], suffix: bool = False) -> tuple[str, ...]:
    names = tuple(
        f"{o}_{s}_{u}_{b}_k{k}"
        for o in obj_orders
        for s in _UNIT_SOURCES
        for u in _UNIT_ORDERS
        for b in _BLOCK_MODES
        for k in range(_MAX_SHIFT)
    )
    return names + ((SUFFIX_POLICY,) if suffix else ())


INDEX_POLICIES: tuple[str, ...] = _policy_names(_OBJ_ORDERS, suffix=True)
CANONICAL_POLICY: str = "sorted_menu_fwd_prefix_k0"
# The suffix map (``tool_a -> unit_a``) is a property of the rename alone, so only the rename stage
# can move it; the object-order stage carries the two printed object orders.
_SORTED_POLICIES: tuple[str, ...] = _policy_names(("sorted", "sortedrev"), suffix=True)
_ROSTER_POLICIES: tuple[str, ...] = _policy_names(("roster", "rosterrev"))
_MAX_CANDIDATES: int = 480


def _index_blocks(n_objs: int, n_units: int, mode: str) -> list[int]:
    """Which unit slot the i-th object takes under one of the two natural readings."""
    if n_units <= 0 or n_objs <= 0:
        return []
    if mode == "prefix":
        return [min(i, n_units - 1) for i in range(n_objs)]
    return [min((i * n_units) // n_objs, n_units - 1) for i in range(n_objs)]


def _shift_recalls(
    objs: list[str], units: list[str], key: dict[str, str], mode: str = "spread"
) -> list[float]:
    """Recall of the index map at EVERY cyclic shift, in one pass.

    The i-th object lands on ``units[(block_i + k) % n]``, so a key goal hits at exactly one shift:
    ``k = index(target) - block_i``.  Counting those residues prices all n shifts at once.
    """
    n = len(units)
    slots = _index_blocks(len(objs), n, mode)
    out = [0.0] * _MAX_SHIFT
    if not slots or not key:
        return out
    where = {u: i for i, u in enumerate(units)}
    counts = [0] * n
    for i, obj in enumerate(objs):
        target = key.get(obj)
        if target in where:
            counts[(where[target] - slots[i]) % n] += 1
    return [counts[k % n] / len(key) for k in range(_MAX_SHIFT)]


def _cap_units(spec: dict) -> list[str]:
    """The units a capacity clause names.  A clause that budgets three of four units publishes
    which unit is dead, and a prompt that renders the clause verbatim exposes that shorter list
    as one more list a model can index into."""
    out: list[str] = []
    for neg in spec["negs"]:
        if neg["kind"] == "capacity":
            for unit in neg.get("cap_expr") or neg.get("cap") or {}:
                _push(out, str(unit))
    return out


def probe_order(task_id: str, alphabet: list[str]) -> list[str]:
    """The order the shuffled-menu prompts print the menu in.

    Mirrors the harness's stable shuffle: a hash sort keyed on the task id.  Duplicated here on
    purpose -- the producer must be able to see the order the harness will render WITHOUT
    importing the attack tooling, so the two implementations must be kept in sync.
    """
    return sorted(
        alphabet, key=lambda s: hashlib.sha256(f"{task_id}|{s}".encode()).hexdigest()
    )


def _allows_none(spec: dict) -> bool:
    placed: set[str] = set()
    for neg in spec["negs"]:
        if neg["kind"] == "all_placed":
            placed |= {str(o) for o in neg.get("objs") or []}
    return not set(spec["roster"]["objects"]) <= placed


def _suffix_recall(objs: list[str], units: list[str], key: dict[str, str]) -> float:
    """``tool_a -> unit_a``: match the object's trailing token to the unit's trailing token.

    Not an index map -- it needs no ordering at all -- but the same free lunch: it can solve a
    row without a single pixel.
    """
    if not key:
        return 0.0
    tail = {u.rsplit("_", 1)[-1]: u for u in units}
    hits = sum(
        1 for o in objs if key.get(o) and key[o] == tail.get(o.rsplit("_", 1)[-1])
    )
    return hits / len(key)


def _unit_sources(spec: dict, menu: list[str], cap: list[str]) -> dict[str, list[str]]:
    """The four lists an attacker can index into, each exactly as it reaches the page."""
    alpha = menu + [NONE_TOKEN] if _allows_none(spec) else list(menu)
    return {
        "menu": sorted(menu),
        "roster": alpha,
        "probe": probe_order(str(spec["task_id"]), alpha),
        # A one-unit capacity list is not an index map at all, it is the constant "always the
        # capacity unit", which is L1's policy; scoring it here would peg the row at 1.00 whatever
        # is permuted and leave the allocator no admissible candidate.
        "cap": sorted(cap) if len(cap) > 1 else sorted(menu),
    }


def index_policies(
    spec: dict, mapping: dict[str, str] | None = None, objs: list[str] | None = None
) -> dict[str, float]:
    """Item recall of every member of the index-map family, for a candidate rename (``mapping``)
    and a candidate object order (``objs``)."""
    ren = mapping or {}
    key = {
        str(g["subject"]): ren.get(str(g["target"]), str(g["target"]))
        for g in spec["answer_key"]
    }
    if not key:
        return dict.fromkeys(INDEX_POLICIES, 0.0)
    menu: list[str] = []
    for name in menu_of(spec):
        _push(menu, ren.get(name, name))
    cap: list[str] = []
    for name in _cap_units(spec):
        _push(cap, ren.get(name, name))
    sources = _unit_sources(spec, menu, cap)
    roster = list(objs if objs is not None else spec["roster"]["objects"])
    orders = {
        "sorted": sorted(roster),
        "sortedrev": sorted(roster, reverse=True),
        "roster": roster,
        "rosterrev": roster[::-1],
    }
    out: dict[str, float] = {SUFFIX_POLICY: _suffix_recall(roster, menu, key)}
    for oname, olist in orders.items():
        for sname, slist in sources.items():
            for uname, ulist in (("fwd", slist), ("rev", slist[::-1])):
                for bname in _BLOCK_MODES:
                    vals = _shift_recalls(olist, ulist, key, bname)
                    for k, val in enumerate(vals):
                        out[f"{oname}_{sname}_{uname}_{bname}_k{k}"] = val
    return out


def _first_pool(spec: dict, renam: list[str]) -> list[str]:
    """The role keys allowed to receive ``unit_a`` when the L1 allocator asked for a non-target."""
    targets = set(key_targets(spec))
    return [c for c in renam if c not in targets] or list(renam)


def _equivalent_anchors(spec: dict, renam: list[str], score: float | None) -> list[str]:
    """Every role key that can take the rank-0 name for the always-first score L1 settled on.

    L1 cares only about the SCORE. Two targets that appear in the key the same number of times are
    interchangeable to it, and so is any non-target when it chose 0.  Handing that whole set to L6
    is what lets a 2-unit row escape the index map without moving the L1 number by a thousandth.
    """
    if score is None:
        return _first_pool(spec, renam)
    out: list[str] = []
    for tok, val in _alpha_first_options(spec):
        if abs(val - score) > 1e-9:
            continue
        out.extend(_first_pool(spec, renam) if tok is None else [tok])
    return out or _first_pool(spec, renam)


def _rename_candidates(
    spec: dict, choice: tuple[str | None, float] | None
) -> list[dict[str, str]]:
    """Every neutral-key permutation the L1 allocation still leaves open.

    ``choice`` is ``assign_alpha_first``'s (token, score) pick.  Every candidate here scores that
    same always-alphabetically-first value, so L6 spends the freedom L1 did not need and cannot
    undo L1.
    """
    renam = _renamable(spec)
    if not renam:
        return []
    names = [f"unit_{LETTERS[i]}" for i in range(len(renam))]
    anchors = _equivalent_anchors(spec, renam, choice[1] if choice else None)
    out: list[dict[str, str]] = []
    for anchor in anchors:
        rest = [c for c in renam if c != anchor]
        for perm in itertools.permutations(names[1:]):
            out.append({anchor: names[0], **dict(zip(rest, perm))})
    if len(out) > _MAX_CANDIDATES:
        out = sorted(
            out,
            key=lambda c: _rng(
                "cand", str(spec["task_id"]), str(sorted(c.items()))
            ).random(),
        )
        out = out[:_MAX_CANDIDATES]
    return out


def _allocate(
    specs: list[dict],
    chance: dict[str, float],
    policies: tuple[str, ...],
    candidates: Callable[[dict], list[Any]],
    score: Callable[[dict, Any], dict[str, float]],
    tag: str,
    protected: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Shared portfolio allocator: walk the specs in a seeded order and pick, per spec, the
    candidate that keeps the running sum of every named policy closest to the running sum of the
    chance level.  A candidate that solves a whole answer key is refused whenever another exists.

    Two sums are tracked, not one: the PER-SPEC mean (what a per-task report averages) and the
    PER-GOAL rate (goals won / goals scored). Balancing only the first lets the multi-goal rows
    run hot while the headline still looks clean, which hides exactly the leak this stage exists
    to remove.
    """
    order = sorted(
        specs, key=lambda s: _rng(f"{tag}-order", str(s["task_id"])).random()
    )
    scale = max(sum(len(s["answer_key"]) for s in specs) / max(len(specs), 1), 1.0)
    got = {(p, w): 0.0 for p in policies for w in ("spec", "goal")}
    want = {"spec": 0.0, "goal": 0.0}
    wins = dict.fromkeys(policies, 0)
    out: dict[str, Any] = {}
    for spec in order:
        tid, n_key = str(spec["task_id"]), len(spec["answer_key"])
        want["spec"] += chance[tid]
        want["goal"] += chance[tid] * n_key
        cands = candidates(spec)
        if not cands:
            for pol, val in score(spec, None).items():
                if pol in policies:
                    got[(pol, "spec")] += val
                    got[(pol, "goal")] += val * n_key
            continue

        def deviation(scores: dict[str, float], n: int = n_key) -> float:
            return sum(
                abs(got[(p, "spec")] + scores[p] - want["spec"])
                + abs(got[(p, "goal")] + scores[p] * n - want["goal"]) / scale
                for p in policies
            )

        def winners(
            scores: dict[str, float], pols: tuple[str, ...], n: int = n_key
        ) -> list[str]:
            return [p for p in pols if scores[p] >= 1.0] if n > 1 else []

        def hotness(scores: dict[str, float]) -> int:
            """How lopsided the family becomes if this candidate is taken.

            Minimising only the NUMBER of members that hold a whole key still lets the same member
            collect all of them (e.g. most unavoidable wins can land on 'reverse the menu and
            shift by two').  Preferring the least-used winner spreads them instead.
            """
            hit = winners(scores, policies)
            return max((wins[p] + 1 for p in hit), default=0)

        scored = [(c, score(spec, c)) for c in cands]
        # Priority order: never let the map an attacker would name FIRST hold a whole multi-goal
        # key, then hold down the rest of the family, then spread what is left, then track chance.
        # A hard veto cannot be used at any level: on a row with as many units as objects the
        # family is the entire symmetric group, so some member wins whatever is picked.
        pick = min(
            scored,
            key=lambda x: (
                len(winners(x[1], protected)),
                len(winners(x[1], policies)),
                hotness(x[1]),
                deviation(x[1]),
                _rng(f"{tag}-tie", tid, str(x[0])).random(),
            ),
        )
        out[tid] = pick[0]
        for pol in winners(pick[1], policies):
            wins[pol] += 1
        for pol in policies:
            got[(pol, "spec")] += pick[1][pol]
            got[(pol, "goal")] += pick[1][pol] * n_key
    return out


def assign_rename(
    specs: list[dict],
    firsts: dict[str, tuple[str | None, float]],
    chance: dict[str, float],
) -> dict[str, dict[str, str]]:
    """role key -> neutral key, chosen so ``obj_i -> unit_<i-th letter>`` scores the chance level."""
    return _allocate(
        specs,
        chance,
        _SORTED_POLICIES,
        lambda s: _rename_candidates(s, firsts.get(str(s["task_id"]))),
        lambda s, c: index_policies(s, mapping=c),
        "rename",
        protected=(
            "sorted_menu_fwd_prefix_k0",
            "sorted_menu_fwd_spread_k0",
            "sorted_probe_fwd_prefix_k0",
            "sorted_cap_fwd_spread_k0",
            SUFFIX_POLICY,
        ),
    )


def _object_orders(spec: dict) -> list[list[str]]:
    objs = list(spec["roster"]["objects"])
    if len(objs) < 2:
        return []
    seen = {tuple(objs): objs}
    for i in range(_MAX_CANDIDATES // 12):
        cand = sorted(
            objs,
            key=lambda o: _rng("objorder", str(spec["task_id"]), str(i), o).random(),
        )
        seen.setdefault(tuple(cand), cand)
    return list(seen.values())


def assign_object_order(
    specs: list[dict], chance: dict[str, float]
) -> dict[str, list[str]]:
    """Seeded print order for ``roster.objects``.

    ``build_roster`` emits the objects in answer-key order, so the roster can list the key's
    pairs adjacently and a model can read the whole key straight off the object list with no
    visual cue at all.
    """
    return _allocate(
        specs,
        chance,
        _ROSTER_POLICIES,
        _object_orders,
        lambda s, c: index_policies(s, objs=c),
        "objorder",
        protected=(
            "roster_roster_fwd_prefix_k0",
            "roster_probe_fwd_prefix_k0",
            "roster_menu_fwd_prefix_k0",
            "roster_cap_fwd_spread_k0",
            "roster_roster_fwd_spread_k0",
        ),
    )


# ---- driver --------------------------------------------------------------------------------------
def deleak(specs: list[dict], rebuild: Callable[[dict], None]) -> dict[str, Any]:
    """Run every de-leak step, in the one order that makes them commute."""
    report: dict[str, Any] = {"arm_menu_union": union_arm_menus(specs)}
    report["distractors"] = add_distractors(specs, rebuild)
    specs[:] = [
        apply_rename(s, PINNED_RENAME.get(str(s["task_id"]), {})) for s in specs
    ]
    chance = {str(s["task_id"]): chance_block(s)["item"] for s in specs}
    firsts = assign_alpha_first(specs, chance)
    indexes = assign_roster_index(specs)
    renames = assign_rename(specs, firsts, chance)
    out: list[dict] = []
    for spec in specs:
        tid = str(spec["task_id"])
        mapping = {k: v for k, v in (renames.get(tid) or {}).items() if k != v}
        spec = apply_rename(spec, mapping)
        permute_roster(spec, indexes[tid])
        out.append(spec)
    specs[:] = out
    orders = assign_object_order(specs, chance)
    out = []
    for spec in specs:
        tid = str(spec["task_id"])
        if orders.get(tid):
            spec["roster"]["objects"] = list(orders[tid])
        spec["chance"] = chance_block(spec)
        spec["redact_in_visible_text"] = sorted(menu_of(spec))
        spec["visibility"] = visibility_block(spec)
        out.append(spec)
    specs[:] = out
    report["degenerate_chance"] = [
        s["task_id"] for s in specs if s["chance"]["degenerate"]
    ]
    report["visible_leaks"] = {
        str(s["task_id"]): f for s in specs if (f := answer_bearing_visible(s))
    }
    return report

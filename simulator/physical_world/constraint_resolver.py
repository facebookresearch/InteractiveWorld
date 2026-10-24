#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Build-time RESOLVER for seed-dependent negative constraints (see the design record).

A negative constraint whose value is a function of the SEED cannot be authored as a static field.
Today those specs ship the field EMPTY, and the shipped ``eval_neg_constraints`` reads an empty
``group_of`` as "no object carries a class" -> ok=True forever: a SILENT PASS. This module gives the
spec a declarative ``<field>_expr`` instead, evaluated by the builder AFTER preloads are placed and
the seeded cue draw is known, and makes it IMPOSSIBLE for an unresolved field to reach the scorer.

This module is ONE STAGE of the spec pipeline, not the pipeline. Regenerate specs with
``python3 build_specs.py``; the stages below are exposed for debugging and for the guard.

Usage:
    python3 build_specs.py               # THE entry point: normalise -> resolve -> guard -> prove
    python3 constraint_resolver.py --migrate      # stage only: author the _expr blocks into specs/*.json
    python3 constraint_resolver.py --check        # GUARD: refuse specs that skipped the resolver stage
    python3 constraint_resolver.py --prove        # run the three proofs, write constraint_resolver_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(os.path.dirname(_HERE), "simulator.physical_world")
for _p in (_CODE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_privacy  # noqa: E402
from constraint_evaluator import eval_neg_constraints, validate_neg_kinds  # noqa: E402
from isaac_goal import Pred  # noqa: E402
from spec_validator import (  # noqa: E402
    _synthetic_violation,
    ERROR,
    Finding,
    REQUIRED_NEG_FIELDS,
    WARN,
)

SPEC_DIR = os.path.join(_HERE, "specs")

# Every field the resolver may fill, and the expression key that fills it. The original field is
# authored EMPTY; only ``resolve_negs`` writes it, so "field is empty" == "the resolver has not run".
RESOLVABLE: dict[str, str] = {
    "cap": "cap_expr",
    "group_of": "group_of_expr",
    "objs": "objs_expr",
    "containers": "containers_expr",
}
INT_OPS = frozenset({"const", "preload", "cue_number", "key_load", "sum"})
SET_SOURCES = frozenset({"pool", "unit_set", "roster", "key_subjects", "lit"})

# Stamped into every spec by ``migrate_spec``. A spec WITHOUT it was written by a stage that runs
# before the resolver, so its seed-dependent fields are the EMPTY ones the scorer reads as "nothing
# violates this". ``assert_spec_resolved`` refuses such a spec by name instead of scoring it.
STAGE_KEY = "pipeline_stage"
PIPELINE_STAGE = "g0.resolved"
ENTRY_POINT = "python3 build_specs.py"


class ResolveError(ValueError):
    """A negative constraint could not be resolved to a concrete, fireable value. Always a BUILD
    error: raising here stops the episode before it is rendered, which is the only place a raise is
    not swallowed (episode_runtime's blanket ``except Exception`` around the goal block turns a score-time
    raise into ``task_goal: null`` with ``episode_valid`` still true -- a false pass by another route)."""


class UnresolvedSpecError(ResolveError):
    """A spec FILE reached a consumer without the resolver stage having run on it. Distinct from
    ``ResolveError`` so a caller can tell "the producers were run in the wrong order" apart from
    "this constraint cannot be resolved at all"."""


@dataclass(frozen=True)
class BuildEnv:
    """The ONLY channel between the builder and the resolver. Published after preload placement,
    physics settle and the seeded cue draw. The resolver may read nothing outside it."""

    preload: dict[str, int] = field(default_factory=dict)
    cue_number: dict[str, dict[str, int]] = field(default_factory=dict)
    cue_class: dict[str, dict[str, str]] = field(default_factory=dict)
    pool: dict[str, list[str]] = field(default_factory=dict)
    unit_set: dict[str, list[str]] = field(default_factory=dict)


# ---- expression evaluation -------------------------------------------------------------------
def _key_load(spec: dict, unit: str) -> int:
    """How many INSIDE goals the answer key routes to ``unit``. ATTACHED goals are excluded: capacity
    counts INSIDE pairs only (constraint_evaluator.py:460-468), so counting them would overstate the load."""
    return sum(
        1
        for g in spec.get("answer_key", [])
        if g.get("target") == unit and g.get("pred", "inside") == "inside"
    )


def _lookup(table: dict, name: str, what: str) -> object:
    if name not in table:
        raise ResolveError(
            f"build env is missing {what} {name!r}; available: {sorted(table)}. The builder must "
            f"publish it before resolve_negs() runs."
        )
    return table[name]


def eval_int(expr: dict, spec: dict, env: BuildEnv) -> int:
    """INT expressions -- the whole arithmetic of the language is ``+``."""
    op = str(expr.get("op"))
    if op not in INT_OPS:
        raise ResolveError(f"unknown INT op {op!r}; implemented: {sorted(INT_OPS)}")
    if op == "const":
        return int(expr["v"])
    if op == "preload":
        return int(_lookup(env.preload, str(expr["unit"]), "preload for unit"))
    if op == "key_load":
        return _key_load(spec, str(expr["unit"]))
    if op == "cue_number":
        chan = _lookup(env.cue_number, str(expr["channel"]), "cue_number channel")
        return int(_lookup(dict(chan), str(expr["unit"]), "cue_number entry"))
    return sum(eval_int(a, spec, env) for a in expr["args"])


def eval_source(src: dict, spec: dict, env: BuildEnv) -> list[str]:
    """A SET source: exactly one of pool / unit_set / roster / key_subjects / lit."""
    named = [k for k in src if k in SET_SOURCES]
    if len(named) != 1:
        raise ResolveError(
            f"a set source needs exactly one of {sorted(SET_SOURCES)}, got {sorted(src)}"
        )
    key = named[0]
    if key == "pool":
        return sorted(str(x) for x in _lookup(env.pool, str(src[key]), "pool"))
    if key == "unit_set":
        return sorted(str(x) for x in _lookup(env.unit_set, str(src[key]), "unit_set"))
    if key == "roster":
        return sorted(str(x) for x in spec.get("roster", {}).get(str(src[key]), []))
    if key == "key_subjects":
        return sorted({str(g["subject"]) for g in spec.get("answer_key", [])})
    return sorted(str(x) for x in src[key])


def eval_set(expr: dict, spec: dict, env: BuildEnv) -> list[str]:
    if str(expr.get("op")) != "set":
        raise ResolveError(f"unknown SET op {expr.get('op')!r}; implemented: ['set']")
    out = set(eval_source(dict(expr["of"]), spec, env))
    for m in expr.get("minus", []):
        out -= set(eval_source(dict(m), spec, env))
    return sorted(out)


def eval_map(expr: dict, spec: dict, env: BuildEnv) -> dict[str, str]:
    """MAP expressions. ``class_map`` is the literal (the MAP counterpart of const/lit): a class that
    is INTRINSIC to the object -- an acid is an acid in every world -- is authored, not drawn."""
    op = str(expr.get("op"))
    if op == "class_map":
        return {str(k): str(v) for k, v in dict(expr["v"]).items()}
    if op != "cue_class":
        raise ResolveError(
            f"unknown MAP op {op!r}; implemented: ['cue_class', 'class_map']"
        )
    chan = dict(_lookup(env.cue_class, str(expr["channel"]), "cue_class channel"))
    domain = eval_set(dict(expr["domain"]), spec, env)
    missing = [k for k in domain if k not in chan]
    if missing:
        raise ResolveError(
            f"cue_class channel {expr['channel']!r} does not classify {missing}. Every object in the "
            f"domain must carry a rendered class, or the constraint is partly blind."
        )
    return {k: str(chan[k]) for k in domain}


# ---- resolution + the fail-closed gates ------------------------------------------------------
def resolve_one(c: dict, spec: dict, env: BuildEnv) -> dict:
    """Fill every ``<field>_expr`` this constraint carries. Returns a new dict; never mutates."""
    out = dict(c)
    for fld, ekey in RESOLVABLE.items():
        expr = c.get(ekey)
        if not expr:
            continue
        if fld == "cap":
            out[fld] = {
                str(u): eval_int(dict(e), spec, env) for u, e in dict(expr).items()
            }
        elif fld == "group_of":
            out[fld] = eval_map(dict(expr), spec, env)
        else:
            out[fld] = eval_set(dict(expr), spec, env)
    out["resolved"] = True
    out.pop("unresolved_reason", None)
    return out


def assert_resolved(c: dict) -> None:
    """GATE 1 -- no EMPTY required field may leave the resolver. This is the assertion that turns the
    silent pass into a hard build error: ``group_of`` is a required field of ``no_share_group``, so an
    empty one (authored or resolved-to-nothing) raises here instead of scoring 1.00 at run time.
    Its build-artefact sibling is ``assert_spec_resolved``, which guards the spec FILE."""
    kind = str(c.get("kind"))
    empty = [f for f in REQUIRED_NEG_FIELDS.get(kind, ()) if not c.get(f)]
    if empty:
        raise ResolveError(
            f"constraint {c.get('label', kind)!r} (kind={kind}) has EMPTY required field(s) {empty} "
            f"after resolution. eval_neg_constraints would read that as 'nothing to check' and report "
            f"ok=True for every world -- a SILENT PASS. Author {RESOLVABLE.get(empty[0], 'the field')}."
        )


def unfilled_fields(c: dict) -> list[str]:
    """Required negative-constraint fields that are EMPTY and that no ``<field>_expr`` will fill."""
    kind = str(c.get("kind"))
    return [
        f
        for f in REQUIRED_NEG_FIELDS.get(kind, ())
        if not c.get(f) and not c.get(RESOLVABLE.get(f, "\x00"))
    ]


def assert_spec_resolved(spec: dict) -> None:
    """GATE 1b -- the BUILD-ARTEFACT gate, on a spec as it sits on disk. ``assert_resolved`` guards a
    constraint the resolver already ran on; this guards the spec FILE, which is where the wrong run
    order does its damage, and names the offending spec and field."""
    tid = str(spec.get("task_id", "<no task_id>"))
    # The field check comes first: it names the actual damage. The stamp check is the backstop for a
    # spec whose constraints happen to be all-static, where no field can betray the missing stage.
    for i, c in enumerate(spec.get("negs") or []):
        for fld in unfilled_fields(c):
            raise UnresolvedSpecError(
                f"{tid}: negs[{i}] kind={str(c.get('kind'))!r} label={str(c.get('label'))!r}: "
                f"required field {fld!r} is EMPTY and no {fld}_expr fills it. eval_neg_constraints "
                f"reads that as 'nothing violates this' and returns ok=True for every world -- a "
                f"SILENT PASS. Regenerate with `{ENTRY_POINT}`."
            )
    stage = spec.get(STAGE_KEY)
    if stage != PIPELINE_STAGE:
        raise UnresolvedSpecError(
            f"{tid}: {STAGE_KEY}={stage!r}, expected {PIPELINE_STAGE!r}. This spec was written by a "
            f"stage that runs BEFORE the resolver (g0_normalise.py on its own), so its seed-dependent "
            f"negative constraints are unresolved. Regenerate with `{ENTRY_POINT}`."
        )


def assert_specs_resolved(specs: list[dict]) -> int:
    """Run the artefact gate over a whole portfolio. Raises on the FIRST offender, but counts them
    all first so the message can say how much of the portfolio is affected."""
    bad: list[tuple[int, str, UnresolvedSpecError]] = []
    for spec in specs:
        try:
            assert_spec_resolved(spec)
        except UnresolvedSpecError as exc:
            empty = any(unfilled_fields(c) for c in spec.get("negs") or [])
            bad.append((0 if empty else 1, str(spec.get("task_id")), exc))
    if bad:
        bad.sort(
            key=lambda t: t[0]
        )  # quote a spec with an EMPTY field: that is the live damage
        n_empty = sum(1 for rank, _, _ in bad if rank == 0)
        names = ", ".join(t for _, t, _ in bad[:6]) + (" ..." if len(bad) > 6 else "")
        raise UnresolvedSpecError(
            f"{len(bad)}/{len(specs)} spec(s) never went through the resolver stage, {n_empty} of "
            f"them with an EMPTY negative-constraint field [{names}]. First offender:\n  {bad[0][2]}"
        ) from bad[0][2]
    return len(specs)


def assert_fireable(c: dict) -> None:
    """GATE 2 -- kind-agnostic. Feed the constraint a synthetic world built from its OWN vocabulary
    that must violate it; if the shipped evaluator still says ok=True the constraint is dead. This
    also catches a future 5th kind that resolves to something structurally inert."""
    probe = _synthetic_violation(c)
    label = c.get("label", c.get("kind"))
    if probe is None:
        raise ResolveError(
            f"constraint {label!r} is UNFIREABLE: no violating world can be built from its own "
            f"vocabulary, so it reports ok=True forever."
        )
    if eval_neg_constraints([c], probe)[0]["ok"]:
        raise ResolveError(
            f"constraint {label!r} is DEAD: the shipped evaluator reports ok=True even for a "
            f"synthetic violating relation set."
        )


def resolve_negs(negs: list[dict], spec: dict, env: BuildEnv) -> list[dict]:
    """THE chokepoint. The builder calls this once, after preloads are placed and the cue draw is
    known, and hands the RESULT to the scorer. Nothing else may construct the scorer's neg list."""
    validate_neg_kinds(
        negs
    )  # GATE 0: an unregistered kind is a build error, not a guaranteed fail
    out = [resolve_one(c, spec, env) for c in negs]
    for c in out:
        assert_resolved(c)
        assert_fireable(c)
    return out


# ---- one-time authoring: migrate the legacy cap_expr / empty fields to the op language ---------
_CLASS_CHANNEL = "item_class"


def _cap_expr_ops(legacy: dict, unit: str) -> dict:
    """Legacy cap_expr shapes -> INT ops. B1 preload+key_load, B2 preload+k, B3 rendered scalar."""
    e = dict(legacy)
    if "literal" in e:
        return {"op": "const", "v": int(e["literal"])}
    pre = {"op": "preload", "unit": unit}
    if "key_load" in e:  # B1: authored as "n + preload"
        return {"op": "sum", "args": [pre, {"op": "key_load", "unit": unit}]}
    if "offset" in e:  # B2: "facings + 2" / "occupants + 1"
        return {"op": "sum", "args": [pre, {"op": "const", "v": int(e["offset"])}]}
    return {
        "op": "cue_number",
        "channel": "facing_limit",
        "unit": unit,
    }  # B3: a drawn scalar


def _pool_minus_key(pool: str = "items") -> dict:
    return {"op": "set", "of": {"pool": pool}, "minus": [{"key_subjects": True}]}


def _already_migrated(c: dict) -> bool:
    """True once every ``_expr`` on this constraint is in the op language. Without this guard a second
    --migrate pass re-reads its own output as a legacy shape and silently rewrites the semantics."""
    exprs = [c[k] for k in c if k.endswith("_expr") and c[k]]
    if not exprs:
        return False
    flat = [v for e in exprs for v in (e.values() if "op" not in e else [e])]
    return all(isinstance(v, dict) and "op" in v for v in flat)


def migrate_neg(c: dict, spec: dict) -> dict:
    """Author the ``<field>_expr`` for one constraint. Idempotent."""
    if _already_migrated(c):
        return dict(c)
    out = dict(c)
    kind = str(c.get("kind"))
    if kind == "capacity" and c.get("cap_expr"):
        out["cap_expr"] = {
            u: _cap_expr_ops(e, u) for u, e in dict(c["cap_expr"]).items()
        }
        out["cap"] = {}
    elif kind == "capacity" and c.get("cap"):
        out["cap_expr"] = {
            u: {"op": "const", "v": int(v)} for u, v in dict(c["cap"]).items()
        }
        out["cap"] = {}
    if kind == "no_share_group" and c.get("group_of"):
        out["group_of_expr"] = {"op": "class_map", "v": dict(c["group_of"])}
        out["group_of"] = {}
    elif kind == "no_share_group":
        # domain = "staged": the movable items AND the preload already sitting in the units. A class
        # map that skips the preload is blind to exactly the object the overflow row is about.
        out["group_of_expr"] = {
            "op": "cue_class",
            "channel": _CLASS_CHANNEL,
            "domain": {"op": "set", "of": {"pool": "staged"}},
        }
    if kind == "not_inside":
        out.update(_migrate_not_inside(c, spec))
    elif kind == "all_placed" and c.get("objs"):
        out["objs_expr"] = {"op": "set", "of": {"lit": sorted(c["objs"])}}
        out["objs"] = []
    return out


def _migrate_not_inside(c: dict, spec: dict) -> dict:
    """C1 'the N spares' -> pool minus the scored subjects. C2 'the live plans' bays' -> a unit_set."""
    why = str(c.get("unresolved_reason", ""))
    out: dict = {}
    if not c.get("objs") and "objs=[" in why:
        out["objs_expr"] = _pool_minus_key()
    if not c.get("containers") and "containers=[" in why:
        # The exclusion is "every live bay EXCEPT the hold bay", and the hold bay is what the
        # answer key routes to.  Reading it off the roster only worked while the roster WAS the
        # single hold bay; once the arm carries its world's full menu that subtracted everything.
        hold = sorted({str(g["target"]) for g in spec.get("answer_key") or []})
        out["containers_expr"] = {
            "op": "set",
            "of": {"unit_set": "live_plan_units"},
            "minus": [{"lit": hold}],
        }
    if c.get("objs") and "objs_expr" not in out:
        out["objs_expr"] = {"op": "set", "of": {"lit": sorted(c["objs"])}}
        out["objs"] = []
    if c.get("containers") and "containers_expr" not in out:
        out["containers_expr"] = {"op": "set", "of": {"lit": sorted(c["containers"])}}
        out["containers"] = []
    return out


def migrate_spec(spec: dict) -> dict:
    """Rewrite one spec's negs into the op language and record the declared env contract."""
    out = dict(spec)
    out["negs"] = [migrate_neg(c, spec) for c in spec.get("negs", [])]
    hint = _spare_count(spec)
    if hint:
        out["resolver_hints"] = {"pool_size": len(spec.get("answer_key", [])) + hint}
    out["resolver_env"] = sorted(_declared_channels(out["negs"]))
    out["requires_resolver"] = bool(out["resolver_env"]) or any(
        c.get(e) for c in out["negs"] for e in RESOLVABLE.values()
    )
    cleared = [
        b for b in spec.get("g0_blocking", []) if b.get("what") == "unresolved_neg"
    ]
    if cleared:
        out["g0_blocking"] = [
            b for b in spec.get("g0_blocking", []) if b.get("what") != "unresolved_neg"
        ]
        out["g0_blocking_cleared"] = cleared
    out[STAGE_KEY] = PIPELINE_STAGE
    return out


def _declared_channels(negs: list[dict]) -> set[str]:
    """Which BuildEnv tables this spec's expressions actually read (the offline contract for G1)."""
    seen: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            op = str(node.get("op", ""))
            for table, k in (
                ("preload", "unit"),
                ("cue_number", "channel"),
                ("cue_class", "channel"),
            ):
                if op == table:
                    seen.add(f"{table}:{node.get(k)}")
            for table in ("pool", "unit_set"):
                if table in node and isinstance(node.get(table), str):
                    seen.add(f"{table}:{node[table]}")
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(negs)
    return seen


# ---- offline stand-in for the builder: a deterministic, key-consistent seeded draw --------------
def _rng_bit(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def _preload_names(unit: str, n: int) -> list[str]:
    return [f"__pre_{unit}_{i}__" for i in range(n)]


def synth_env(spec: dict, seed: int = spec_privacy.FROZEN_WORLD_SEED) -> BuildEnv:
    """The offline STAND-IN the real builder replaces. It draws exactly what the builder draws --
    preload counts, the per-item class decals, the staged pool, the live-plan unit set -- from a
    deterministic hash of (task_id, seed), and keeps the class draw CONSISTENT with the answer key
    (the real causation is cue -> key; the specs froze the key, so the stand-in inverts it).

    The world is frozen, so the only value a gate may judge a spec at is the default: any other
    ``seed`` synthesises an environment the benchmark will never build."""
    tid = str(spec.get("task_id"))
    preload = _synth_preload(spec, tid, seed)
    pool = _synth_pool(spec, preload)
    classes = _synth_classes(spec, tid, seed, pool["staged"], preload)
    return BuildEnv(
        preload=preload,
        cue_number={"facing_limit": {u: n for u, n in preload.items()}},
        cue_class={_CLASS_CHANNEL: classes} if classes else {},
        pool=pool,
        unit_set=_synth_unit_sets(spec, tid, seed),
    )


def _synth_preload(spec: dict, tid: str, seed: int) -> dict[str, int]:
    """Every unit a preload-dependent cap mentions gets 1..2 seeded occupants."""
    out: dict[str, int] = {}
    for c in spec.get("negs", []):
        for unit, e in dict(c.get("cap_expr") or {}).items():
            if f'"preload"' in json.dumps(e) or '"cue_number"' in json.dumps(e):
                out[unit] = 1 + _rng_bit(tid, str(seed), "pre", unit) % 2
    return out


def _spare_count(spec: dict) -> int:
    """The N of 'the N spares' / 'the N lab spares', read off the fragment the normaliser could not
    parse. The builder knows this number directly (it staged the pouches); offline we recover it."""
    n = 0
    for c in spec.get("negs", []):
        why = str(c.get("unresolved_reason", ""))
        if "spares" not in why:
            continue
        for tok in why.replace("'", " ").replace("[", " ").split():
            if tok.isdigit():
                n = max(n, int(tok))
    return n


def _synth_pool(spec: dict, preload: dict[str, int]) -> dict[str, list[str]]:
    """``items`` = every graspable staged for the human (scored + distractors); ``staged`` = that
    plus the preload already inside the units. Specs whose prose says 'the 8 spares' stage 12."""
    objs = sorted(spec.get("roster", {}).get("objects", []))
    total = int(spec.get("resolver_hints", {}).get("pool_size") or 0)
    if not total:
        total = len(spec.get("answer_key", [])) + _spare_count(spec)
    items = [f"item_{i}" for i in range(1, total + 1)] if total > len(objs) else objs
    pre = [n for u, k in sorted(preload.items()) for n in _preload_names(u, k)]
    # A NAMED pre-placed object is staged exactly like the anonymous preload: it exists at t=0 and a
    # constraint may name it, so leaving it out of `staged` reads as an undeclared entity (NEG005).
    seated = sorted(dict(spec.get("start_placements") or {}))
    return {
        "items": sorted(items),
        "staged": sorted(set(items) | set(pre) | set(seated)),
    }


def _synth_unit_sets(spec: dict, tid: str, seed: int) -> dict[str, list[str]]:
    """``live_plan_units``: the bays the surviving plans still route to, plus the hold bay. Offline it
    is read off the DETERMINED sibling (whose goals name the real bays); the builder computes it from
    the plan table and the prologue it actually rendered."""
    if not str(spec.get("task_id", "")).endswith(".ambiguous"):
        return {}
    sib = os.path.join(
        SPEC_DIR, f"{spec['task_id'][: -len('.ambiguous')]}.determined.json"
    )
    if not os.path.exists(sib):
        return {}
    with open(sib) as fh:
        det = json.load(fh)
    bays = sorted({str(g["target"]) for g in det.get("answer_key", [])})
    pinned = bays[_rng_bit(tid, str(seed), "pin") % len(bays)] if bays else None
    live = [b for b in bays if b != pinned]
    return {
        "live_plan_units": sorted(
            set(live) | set(spec.get("roster", {}).get("containers", []))
        )
    }


def _synth_classes(
    spec: dict, tid: str, seed: int, pool: list[str], preload: dict[str, int]
) -> dict[str, str]:
    """Draw the item->class decals. Objects the answer key routes to a constrained unit take that
    unit's class (so a unit is class-homogeneous and the REFERENCE solution never trips its own
    rule); everything else -- distractors, preload -- is a free seeded coin flip."""
    nsg = [c for c in spec.get("negs", []) if c.get("kind") == "no_share_group"]
    if not nsg or not nsg[0].get("incompatible"):
        return {}
    pair = [str(g) for g in nsg[0]["incompatible"][0]]
    units = [str(u) for u in nsg[0].get("units") or []]
    key = {str(g["subject"]): str(g["target"]) for g in spec.get("answer_key", [])}
    by_unit = {u: pair[i % len(pair)] for i, u in enumerate(units)}
    out: dict[str, str] = {}
    for obj in sorted(set(pool) | set(key)):
        tgt = key.get(obj)
        out[obj] = (
            by_unit.get(str(tgt)) or pair[_rng_bit(tid, str(seed), obj) % len(pair)]
        )
    for unit, n in preload.items():
        for name in _preload_names(unit, n):
            out[name] = (
                by_unit.get(unit) or pair[_rng_bit(tid, str(seed), name) % len(pair)]
            )
    return out


# ---- reference / counter-example worlds --------------------------------------------------------
def reference_relations(spec: dict, env: BuildEnv) -> set:
    """The world a PERFECT run leaves behind: the seeded preload plus the answer key. The preload
    pairs are load-bearing -- capacity is TOTAL occupancy, so a reference world that omits them
    understates every count by exactly the number the cap was raised by."""
    rel = {
        (Pred.INSIDE, str(g["subject"]), str(g["target"]))
        for g in spec.get("answer_key", [])
        if g.get("pred", "inside") == "inside" and g.get("target")
    }
    for unit, n in env.preload.items():
        rel |= {(Pred.INSIDE, name, unit) for name in _preload_names(unit, n)}
    # A declared pre-placed object is still there when the run ends -- nothing ever grabs it -- and a
    # no_share_group that names it must be evaluated against the seat it actually occupies.
    for name, place in dict(spec.get("start_placements") or {}).items():
        if str(place).startswith("in:"):
            rel.add((Pred.INSIDE, str(name), str(place)[3:]))
    return rel


def wrong_relations(spec: dict, neg: dict, env: BuildEnv) -> set | None:
    """A DELIBERATELY WRONG solution for a no_share_group spec: keep the answer key, but re-route one
    item of the opposite class into a constrained unit. Nothing else changes -- every INSIDE goal is
    still satisfied, so the only thing that can catch it is the constraint under test."""
    rel = reference_relations(spec, env)
    gof = dict(neg.get("group_of") or {})
    units = [str(u) for u in neg.get("units") or []]
    for g1, g2 in [tuple(p) for p in neg.get("incompatible", [])]:
        for unit in units:
            here = {gof.get(a) for (_p, a, b) in rel if b == unit}
            other = (
                str(g2) if str(g1) in here else (str(g1) if str(g2) in here else None)
            )
            if other is None:
                continue
            movers = sorted(
                a
                for a, g in gof.items()
                if g == other and (Pred.INSIDE, a, unit) not in rel
            )
            if movers:
                return rel | {(Pred.INSIDE, movers[0], unit)}
    return None


# ---- validator extension (additive sibling of spec_validator; see doc section 4) ---------------
def check_resolvable(spec: dict, env: BuildEnv) -> list[Finding]:
    """RES001-004: catch an unresolvable resolver, and count the PRELOAD in the capacity check --
    the two things spec_validator cannot see (it reads shipped MODULES and counts only the answer
    key's own load, so a preload-coupled cap passes offline and fails at run time)."""
    out: list[Finding] = []
    negs = spec.get("negs", [])
    try:
        resolved = resolve_negs(negs, spec, env)
    except (ResolveError, ValueError) as exc:
        return [
            Finding(
                "RES001",
                ERROR,
                str(spec.get("task_id")),
                f"the resolver cannot produce a concrete, fireable constraint: {exc}",
                "Author the missing <field>_expr, or publish the missing build-env channel.",
            )
        ]
    for c in negs:
        out += _check_no_static_bypass(c)
    out += _check_capacity_preload(spec, resolved, env)
    return out


def _check_no_static_bypass(c: dict) -> list[Finding]:
    """RES002: a required field carrying a STATIC value with no ``_expr`` behind it. Legal today, but
    it means two code paths fill the same field and 'empty' stops meaning 'the resolver has not run'."""
    kind = str(c.get("kind"))
    bad = [
        f
        for f in REQUIRED_NEG_FIELDS.get(kind, ())
        if f in RESOLVABLE and c.get(f) and not c.get(RESOLVABLE[f])
    ]
    if not bad:
        return []
    return [
        Finding(
            "RES002",
            WARN,
            str(c.get("label", kind)),
            f"field(s) {bad} are filled statically with no sibling _expr, so the resolver is not the "
            "only writer and an empty field no longer proves the resolver was skipped.",
            f"Author {[RESOLVABLE[f] for f in bad]} (a literal is {{'op':'const'}} / {{'lit':[...]}}).",
        )
    ]


def _check_capacity_preload(
    spec: dict, resolved: list[dict], env: BuildEnv
) -> list[Finding]:
    """RES003/RES004: the capacity check spec_validator._rule_capacity should have made -- against
    preload + the answer key's own load, not the answer key alone."""
    out: list[Finding] = []
    for c in resolved:
        if str(c.get("kind")) != "capacity":
            continue
        for unit, cap in sorted(dict(c.get("cap") or {}).items()):
            need = env.preload.get(unit, 0) + _key_load(spec, unit)
            if need > int(cap):
                out.append(
                    Finding(
                        "RES003",
                        ERROR,
                        f"{spec.get('task_id')}:{unit}",
                        f"preload({env.preload.get(unit, 0)}) + answer-key load({_key_load(spec, unit)}) "
                        f"= {need} EXCEEDS the resolved cap {cap} -- a perfect run trips its own "
                        "capacity constraint. Invisible to spec_validator, which ignores the preload.",
                        f"Raise the offset for {unit!r} to >= the key load, or re-route the key.",
                    )
                )
            elif need == int(cap) and cap > 0:
                out.append(
                    Finding(
                        "RES004",
                        WARN,
                        f"{spec.get('task_id')}:{unit}",
                        f"zero capacity headroom ({need}/{cap}) once the preload is counted; one "
                        "spurious containment from an overlapping neighbour flips this to a violation.",
                        f"Raise cap[{unit!r}] by one, or land container exclusivity first.",
                    )
                )
    return out


# ---- driver ------------------------------------------------------------------------------------
def load_specs(spec_dir: str = "") -> list[dict]:
    out = []
    for name in sorted(os.listdir(spec_dir or SPEC_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(spec_dir or SPEC_DIR, name)) as fh:
                out.append(json.load(fh))
    return out


def dump_spec(path: str, spec: dict) -> bool:
    """THE spec serialisation, shared with g0_normalise.emit so the two producers cannot churn each
    other's whitespace. Returns True when the bytes actually changed."""
    text = json.dumps(spec, indent=1, ensure_ascii=False) + "\n"
    old = open(path).read() if os.path.exists(path) else None
    if old == text:
        return False
    with open(path, "w") as fh:
        fh.write(text)
    return True


def cmd_migrate(spec_dir: str = "") -> int:
    """Idempotent, and a no-op after g0_normalise (which now runs this stage itself)."""
    root = spec_dir or SPEC_DIR
    changed = [
        s["task_id"]
        for s in load_specs(root)
        if dump_spec(os.path.join(root, f"{s['task_id']}.json"), migrate_spec(s))
    ]
    print(f"migrate: {len(changed)} spec file(s) rewritten into the op language")
    return 0


def cmd_check(spec_dir: str = "") -> int:
    """THE GUARD. Refuses a spec portfolio that skipped the resolver stage, by name."""
    specs = load_specs(spec_dir or SPEC_DIR)
    n_fields = sum(len(unfilled_fields(c)) for s in specs for c in s.get("negs") or [])
    n = assert_specs_resolved(specs)
    print(
        f"check: {n}/{n} spec(s) carry {STAGE_KEY}={PIPELINE_STAGE!r}; "
        f"{n_fields} unresolved negative-constraint field(s)"
    )
    return 0


def _prove_one(spec: dict) -> dict:
    """Per-spec proof row: does it resolve, is the reference clean, does a wrong world now fire?"""
    env = synth_env(spec)
    row: dict = {"task_id": spec["task_id"], "findings": []}
    try:
        resolved = resolve_negs(spec.get("negs", []), spec, env)
    except (ResolveError, ValueError) as exc:
        row["resolve"] = f"ERROR: {exc}"
        return row
    row["resolve"] = "ok"
    ref = reference_relations(spec, env)
    row["ref_violations"] = sorted(
        f"{v['label']}:{v['offenders']}"
        for v in eval_neg_constraints(resolved, ref)
        if not v["ok"]
    )
    row["findings"] = [
        f"{f.rule}/{f.severity} {f.entity}: {f.message}"
        for f in check_resolvable(spec, env)
    ]
    row |= _prove_counterexample(spec, resolved, env)
    return row


def _pre_resolver_neg(neg: dict, spec: dict) -> dict:
    """The constraint exactly as it shipped ON DISK before this work: an EMPTY group_of for the specs
    whose class map was seeded (the silent-pass class), the authored literal map for the rest."""
    out = {k: v for k, v in neg.items() if not k.endswith("_expr")}
    was_blocked = any(
        b.get("kind") == "no_share_group" for b in spec.get("g0_blocking_cleared", [])
    )
    expr = dict(neg.get("group_of_expr") or {})
    out["group_of"] = {} if was_blocked else dict(expr.get("v") or {})
    return out


def _prove_counterexample(spec: dict, resolved: list[dict], env: BuildEnv) -> dict:
    """(c) the whole point. One WRONG solution -- every INSIDE goal still satisfied, one item of the
    opposite class re-routed into a constrained unit -- scored three ways: as the spec SHIPPED, with
    the resolver SKIPPED, and RESOLVED."""
    for cur, after in zip(spec.get("negs", []), resolved):
        if str(after.get("kind")) != "no_share_group":
            continue
        wrong = wrong_relations(spec, after, env)
        if wrong is None:
            return {"counterexample": "NONE CONSTRUCTIBLE"}
        verdict = eval_neg_constraints([after], wrong)[0]
        return {
            "counterexample": sorted(wrong - reference_relations(spec, env))[0][1:],
            "before": eval_neg_constraints([_pre_resolver_neg(after, spec)], wrong)[0][
                "ok"
            ],
            "after": verdict["ok"],
            "after_offenders": verdict["offenders"],
            "skip_raises": _skip_raises(cur),
        }
    return {}


def _skip_raises(neg: dict) -> bool:
    """The fail-closed gate: hand the scorer the UNRESOLVED constraint and confirm it is refused."""
    try:
        assert_resolved(neg)
    except ResolveError:
        return True
    return False


def prove_summary(rows: list[dict]) -> dict:
    """The four headline counters, so a caller can assert on them instead of scraping stdout."""
    ces = [r for r in rows if "before" in r]
    silent = [
        r for r in ces if r["before"] is True
    ]  # the specs that shipped an empty group_of
    return {
        "specs": len(rows),
        "resolved": len(rows) - sum(1 for r in rows if r["resolve"] != "ok"),
        "reference_clean": len(rows) - sum(1 for r in rows if r.get("ref_violations")),
        "silent_specs": len(silent),
        "silent_now_fire": sum(1 for r in silent if r["after"] is False),
        "already_firing": len(ces) - len(silent),
        "already_firing_still_fire": sum(
            1 for r in ces if r["before"] is False and r["after"] is False
        ),
        "counterexamples": len(ces),
        "skip_raises": sum(1 for r in ces if r["skip_raises"]),
    }


def cmd_prove(spec_dir: str = "") -> int:
    rows = [_prove_one(s) for s in load_specs(spec_dir)]
    bad_resolve = [r for r in rows if r["resolve"] != "ok"]
    dirty_ref = [r for r in rows if r.get("ref_violations")]
    ces = [r for r in rows if "before" in r]
    s = prove_summary(rows)
    print(f"(a) specs resolved            : {s['resolved']}/{s['specs']}")
    print(f"(b) reference solutions clean : {s['reference_clean']}/{s['specs']}")
    print(
        f"(c) silent-pass specs now FIRE: {s['silent_now_fire']}/{s['silent_specs']}"
        f"   (+{s['already_firing_still_fire']}/{s['already_firing']} already-firing specs still fire)"
    )
    print(
        f"(d) skipping the resolver now RAISES: {s['skip_raises']}/{s['counterexamples']}"
    )
    for r in bad_resolve + dirty_ref:
        print(
            f"    !! {r['task_id']}: {r.get('resolve')} {r.get('ref_violations', '')}"
        )
    for r in sorted(ces, key=lambda x: x["task_id"]):
        print(
            f"    {r['task_id']:24s} move {r['counterexample'][0]} -> {r['counterexample'][1]}"
            f" | before ok={r['before']} | after ok={r['after']} {r['after_offenders']}"
        )
    for r in rows:
        for f in r["findings"]:
            print(f"    {f}")
    report = os.path.join(
        os.path.dirname(spec_dir) if spec_dir else _HERE,
        "constraint_resolver_report.json",
    )
    with open(report, "w") as fh:
        fh.write(json.dumps(rows, indent=1) + "\n")
    return 0 if not bad_resolve and not dirty_ref else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--migrate", action="store_true", help="author _expr blocks into specs/*.json"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="GUARD: refuse specs that skipped this stage",
    )
    ap.add_argument("--prove", action="store_true", help="run the three proofs")
    ap.add_argument(
        "--spec-dir", default="", help="directory containing task specifications"
    )
    args = ap.parse_args(argv)
    rc = 0
    if args.migrate:
        rc |= cmd_migrate(args.spec_dir)
    if args.check:
        rc |= cmd_check(args.spec_dir)
    if args.prove:
        rc |= cmd_prove(args.spec_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

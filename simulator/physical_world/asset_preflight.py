# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""OFFLINE asset pre-flight -- prove every asset a task spec names is baked BEFORE a render is launched.

An unbaked asset is a SILENT drop: ``behavior_import`` catches the ``FileNotFoundError`` from
``baked_usd`` and ``continue``s while the answer key is copied verbatim, so the goal ends up quantified
over an entity that does not exist -- permanently unmet if it is a goal subject, or permanently VACUOUS
if it is only a negative's container (the "sink trap": an unregistered container makes ``not_inside``
unfireable, so it reports ok=True forever).

THE QUERY MATTERS. ``behavior_bake.scene_models()`` keeps entries whose ``class_name`` is None, while
the LOADER ``behavior_import.scene_objects()`` requires ``class_name == "DatasetObject"`` AND non-null
pos/ori -- so a pre-flight built on the bake query can report models that never load. This module
therefore calls the LOADER's own function (never a copy of it) and additionally drops
``behavior_import._SKIP_CATS``, which the compose pass skips before it ever asks for a baked USD.
``scene_query_divergence()`` measures the two sets so the choice stays auditable.

STANDALONE + READ-ONLY: no Isaac, no stage, no write. Runs in seconds.

    python3 simulator/physical_world/asset_preflight.py                          # selftest
    python3 simulator/physical_world/asset_preflight.py --spec <task_module>     # a task spec
    python3 simulator/physical_world/asset_preflight.py --scene school_chemistry # a whole scene
    python3 simulator/physical_world/asset_preflight.py --scene Rs_int --divergence

Exit codes: 0 = clean, 1 = at least one CRITICAL finding, 2 = selftest failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import behavior_import  # noqa: E402  (stdlib-only at import time; the loader's own filter)
from simulator.taskpack_support.bundled_assets import (  # noqa: E402
    generated_label_cache,
)

CACHE_ROOT = os.environ.get(
    "INTERACTIVEWORLD_BEHAVIOR_CACHE", "/opt/interactiveworld/behavior-cache"
)
CACHE_USD = os.path.join(CACHE_ROOT, "usd")
CACHE_SPEC = os.path.join(CACHE_ROOT, "spec")
GENERATED_LABEL_DIR = str(generated_label_cache())
OBJ_ROOT = os.path.join(
    os.environ.get(
        "INTERACTIVEWORLD_BEHAVIOR_DATA", "/opt/interactiveworld/behavior-data"
    ),
    "behavior-1k-assets",
    "objects",
)
SCENE_ROOT = behavior_import.SCENE_ROOT

BAKED = "baked"
BAKED_NO_SPEC = "baked_no_spec"
UNBAKED_HAS_SOURCE = "unbaked_has_source"
SOURCE_ABSENT = "source_absent"

CRITICAL = "CRITICAL"
IMPORTANT = "IMPORTANT"
COSMETIC = "COSMETIC"

BAKE_CMD = "install the frozen BEHAVIOR cache containing"


@dataclass(frozen=True)
class AssetFinding:
    rule: str
    criticality: str
    entity: str
    message: str
    fix: str

    def as_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


def bake_status(cat: str, model: str) -> str:
    """Exactly what ``behavior_import.baked_usd`` will find, plus whether the baked spec.json exists."""
    if os.path.exists(os.path.join(CACHE_USD, cat, f"{model}.usd")):
        has_spec = os.path.exists(os.path.join(CACHE_SPEC, cat, f"{model}.json"))
        return BAKED if has_spec else BAKED_NO_SPEC
    return (
        UNBAKED_HAS_SOURCE
        if os.path.isdir(os.path.join(OBJ_ROOT, cat, model))
        else SOURCE_ABSENT
    )


def bake_command(pairs: list[tuple[str, str]]) -> str:
    """The exact command that fixes a missing list. Empty list -> empty string (nothing to run)."""
    uniq = sorted({f"{c}/{m}" for c, m in pairs})
    return f"{BAKE_CMD} {' '.join(uniq)}" if uniq else ""


def scene_loadable_models(
    scene: str, include_skipped: bool = False
) -> list[tuple[str, str]]:
    """Unique (cat, model) the loader will ACTUALLY request for ``scene`` -- the correct pre-flight query."""
    skip: set[str] = (
        set() if include_skipped else set(getattr(behavior_import, "_SKIP_CATS", ()))
    )
    seen: set[tuple[str, str]] = set()
    for o in behavior_import.scene_objects(scene):
        if o["cat"] in skip:
            continue
        seen.add((o["cat"], o["model"]))
    return sorted(seen)


def scene_category_counts(scene: str, include_skipped: bool = False) -> dict[str, int]:
    """Loadable INSTANCE count per category -- how many prims a scene-sourced container can be assigned."""
    skip: set[str] = (
        set() if include_skipped else set(getattr(behavior_import, "_SKIP_CATS", ()))
    )
    out: dict[str, int] = {}
    for o in behavior_import.scene_objects(scene):
        if o["cat"] in skip:
            continue
        out[o["cat"]] = out.get(o["cat"], 0) + 1
    return out


def scene_category_models(scene: str, cat: str) -> list[str]:
    return sorted(
        {o["model"] for o in behavior_import.scene_objects(scene) if o["cat"] == cat}
    )


def scene_query_divergence(scene: str) -> dict[str, object]:
    """Measure the bake-side query against the loader-side one, so the choice of query stays auditable."""
    loader = set(scene_loadable_models(scene, include_skipped=True))
    try:
        import behavior_bake

        bake = set(behavior_bake.scene_models(scene))
    except (ImportError, OSError, ValueError, KeyError) as exc:
        return {
            "scene": scene,
            "error": f"{type(exc).__name__}: {exc}",
            "loader": len(loader),
        }
    return {
        "scene": scene,
        "loader": len(loader),
        "bake": len(bake),
        "bake_only": sorted(f"{c}/{m}" for c, m in bake - loader),
        "loader_only": sorted(f"{c}/{m}" for c, m in loader - bake),
    }


# ---- spec-driven pre-flight --------------------------------------------------------------------------
def _safe(name: str) -> str:
    """Mirrors ``behavior_import._safe`` -- the decal filename is built from the SANITISED entity key."""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


def _spec_refs(spec) -> list[tuple[str, str, str, str, str]]:
    """(origin, key, cat, model, criticality) for every DIRECT baked-asset reference a spec makes."""
    scored = {g.subject for g in spec.goals} | spec.goal_targets()
    neg_named: set[str] = set()
    for c in spec.negs:
        neg_named |= {str(x) for x in (c.get("containers") or [])}
        neg_named |= {str(x) for x in (c.get("units") or [])}
        neg_named |= {str(x) for x in (c.get("cap") or {})}
    out: list[tuple[str, str, str, str, str]] = []
    for origin, table in (("item", spec.items), ("part", spec.parts)):
        for key, v in sorted(table.items()):
            if v.get("cat") and v.get("model"):
                crit = CRITICAL if key in scored else IMPORTANT
                out.append((origin, key, str(v["cat"]), str(v["model"]), crit))
    for key, v in sorted(spec.containers.items()):
        if (
            str(v.get("source", "import")) == "import"
            and v.get("cat")
            and v.get("model")
        ):
            crit = CRITICAL if (key in scored or key in neg_named) else IMPORTANT
            out.append(("container", key, str(v["cat"]), str(v["model"]), crit))
    for i, d in enumerate(spec.decor):
        if d.get("cat") and d.get("model"):
            out.append(
                ("decor", f"decor[{i}]", str(d["cat"]), str(d["model"]), COSMETIC)
            )
    return out


def spec_findings(spec) -> list[AssetFinding]:
    """Every asset problem in one spec: direct refs, scene-sourced supply, and label PNGs."""
    out: list[AssetFinding] = []
    for origin, key, cat, model, crit in _spec_refs(spec):
        status = bake_status(cat, model)
        if status == BAKED:
            continue
        rule = "ASSET001" if status != BAKED_NO_SPEC else "ASSET002"
        detail = {
            BAKED_NO_SPEC: "baked USD present but its spec.json is missing (role_hint/bbox/mass unknown)",
            UNBAKED_HAS_SOURCE: "NOT baked (source model exists) -- silently dropped at build",
            SOURCE_ABSENT: "NOT baked and NO source model on disk -- cannot be baked",
        }[status]
        out.append(
            AssetFinding(
                rule,
                crit,
                f"{key} ({origin})",
                f"{cat}/{model}: {detail}. The answer key is copied verbatim, so a dropped asset leaves a "
                "predicate over an entity that does not exist.",
                bake_command([(cat, model)])
                or f"restore the source model at {OBJ_ROOT}/{cat}/{model}",
            )
        )
    return out + _scene_container_findings(spec) + label_findings(spec)


def _scene_container_findings(spec) -> list[AssetFinding]:
    """A scene-sourced container vanishes if every instance of its category is unbaked."""
    if not spec.scene:
        return []
    try:
        counts = scene_category_counts(spec.scene)
    except (OSError, ValueError, KeyError) as exc:
        return [
            AssetFinding(
                "ASSET003",
                IMPORTANT,
                spec.scene,
                f"scene layout unreadable: {exc}",
                "check the scene key",
            )
        ]
    out: list[AssetFinding] = []
    for key, c in sorted(spec.containers.items()):
        if str(c.get("source", "import")) != "scene":
            continue
        cat = str(c.get("cat"))
        models = scene_category_models(spec.scene, cat)
        unbaked = [
            (cat, m)
            for m in models
            if bake_status(cat, m) not in (BAKED, BAKED_NO_SPEC)
        ]
        if models and len(unbaked) == len(models):
            out.append(
                AssetFinding(
                    "ASSET004",
                    CRITICAL,
                    key,
                    f"every {cat!r} instance in {spec.scene!r} ({counts.get(cat, 0)} of them) is unbaked, so "
                    "this container never registers -- its goals are permanently unmet and its negatives "
                    "permanently report ok=True (the sink trap).",
                    bake_command(unbaked),
                )
            )
        elif unbaked:
            out.append(
                AssetFinding(
                    "ASSET005",
                    IMPORTANT,
                    key,
                    f"{len(unbaked)}/{len(models)} {cat!r} model(s) in {spec.scene!r} are unbaked -- the "
                    "container still registers, but on a reduced and build-order-dependent candidate set.",
                    bake_command(unbaked),
                )
            )
    return out


def label_findings(spec) -> list[AssetFinding]:
    """Label PNGs are hand-made and a missing one is a silent ``continue`` at build time."""
    out: list[AssetFinding] = []
    for key, c in sorted(spec.containers.items()):
        sign = c.get("sign")
        if sign and not os.path.exists(
            os.path.join(GENERATED_LABEL_DIR, f"{sign}.png")
        ):
            out.append(
                _label_missing(
                    "LABEL001", CRITICAL, key, f"{sign}.png", "container sign"
                )
            )
    for key, item in sorted(spec.items.items()):
        if item.get("needs_lookat") and not os.path.exists(
            os.path.join(GENERATED_LABEL_DIR, f"reagent_{_safe(key)}.png")
        ):
            out.append(
                _label_missing(
                    "LABEL002",
                    CRITICAL,
                    key,
                    f"reagent_{_safe(key)}.png",
                    "needs_lookat item decal",
                )
            )
    for key, base in sorted(spec.labels.items()):
        if base and not os.path.exists(
            os.path.join(GENERATED_LABEL_DIR, f"{base}.png")
        ):
            out.append(
                _label_missing(
                    "LABEL003", CRITICAL, key, f"{base}.png", "part letter placard"
                )
            )
    return out


def _label_missing(rule: str, crit: str, key: str, png: str, what: str) -> AssetFinding:
    return AssetFinding(
        rule,
        crit,
        key,
        f"{what} {png!r} is absent from {GENERATED_LABEL_DIR} -- the decal is silently skipped, so the observation "
        "the spec says the assistant must READ does not exist and the task is unsolvable by design.",
        f"Generate {os.path.join(GENERATED_LABEL_DIR, png)} before launching the episode.",
    )


# ---- scene-driven pre-flight ---------------------------------------------------------------------------
def scene_findings(scene: str) -> list[AssetFinding]:
    """Every model a scene will request, classified. All IMPORTANT -- criticality needs a spec."""
    out: list[AssetFinding] = []
    missing: list[tuple[str, str]] = []
    for cat, model in scene_loadable_models(scene):
        status = bake_status(cat, model)
        if status in (BAKED, BAKED_NO_SPEC):
            continue
        missing.append((cat, model))
        out.append(
            AssetFinding(
                "ASSET001",
                IMPORTANT,
                f"{cat}/{model}",
                f"{status} -- silently dropped when {scene!r} is composed.",
                bake_command([(cat, model)])
                if status == UNBAKED_HAS_SOURCE
                else "no source model on disk",
            )
        )
    if missing:
        out.append(
            AssetFinding(
                "ASSET006",
                IMPORTANT,
                scene,
                f"{len(missing)} model(s) missing",
                bake_command(missing),
            )
        )
    return out


def report(title: str, findings: list[AssetFinding]) -> int:
    print(f"\n=== asset pre-flight: {title} ===")
    order = {CRITICAL: 0, IMPORTANT: 1, COSMETIC: 2}
    for f in sorted(
        findings, key=lambda x: (order.get(x.criticality, 3), x.rule, x.entity)
    ):
        print(
            f"  {f.criticality:9s} {f.rule:9s} {f.entity}\n        {f.message}\n        FIX: {f.fix}"
        )
    n_crit = sum(1 for f in findings if f.criticality == CRITICAL)
    print(f"  -- {n_crit} CRITICAL / {len(findings)} finding(s)")
    return n_crit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline BEHAVIOR asset pre-flight (no Isaac)."
    )
    ap.add_argument(
        "--spec", action="append", default=[], help="task module name, repeatable"
    )
    ap.add_argument(
        "--scene", action="append", default=[], help="BEHAVIOR scene key, repeatable"
    )
    ap.add_argument(
        "--divergence",
        action="store_true",
        help="bake-query vs loader-query diff per scene",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not (args.spec or args.scene):
        return _selftest()
    payload: dict[str, object] = {}
    n_crit = 0
    for name in args.spec:
        # Lazy: spec_validator imports THIS module, so the dependency is one-way at import time.
        import spec_validator

        findings = spec_findings(spec_validator.load_spec(name))
        payload[name] = [f.as_dict() for f in findings]
        n_crit += report(name, findings) if not args.json else 0
    for scene in args.scene:
        findings = scene_findings(scene)
        payload[scene] = [f.as_dict() for f in findings]
        if args.divergence:
            payload[f"{scene}:divergence"] = scene_query_divergence(scene)
        n_crit += report(scene, findings) if not args.json else 0
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 1 if n_crit else 0


# =====================================================================================================
# Pure selftest: the query choice, the status classifier, and NEGATIVE cases on a synthetic spec.
# =====================================================================================================
@dataclass
class _FakeSpec:
    """A spec-shaped stand-in so the selftest never depends on a real module or the bake cache."""

    name: str = "fake"
    scene: str = ""
    goals: list = dataclasses.field(default_factory=list)
    negs: list = dataclasses.field(default_factory=list)
    items: dict = dataclasses.field(default_factory=dict)
    containers: dict = dataclasses.field(default_factory=dict)
    parts: dict = dataclasses.field(default_factory=dict)
    decor: list = dataclasses.field(default_factory=list)
    labels: dict = dataclasses.field(default_factory=dict)

    def goal_targets(self) -> set[str]:
        return {g.target for g in self.goals if g.target}


def _selftest() -> int:
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    chk(
        bake_status("__no_such_cat__", "zzzz") == SOURCE_ABSENT,
        "unknown cat/model -> source_absent",
    )
    chk(
        bake_command([("b", "2"), ("a", "1"), ("a", "1")]) == f"{BAKE_CMD} a/1 b/2",
        "bake_command is de-duplicated, sorted, and copy-pasteable",
    )
    chk(bake_command([]) == "", "bake_command is empty when nothing is missing")

    from spec_validator import (  # local: spec_validator imports this module at its top
        Goal,
    )

    spec = _FakeSpec(
        goals=[Goal("inside", "widget", "bin")],
        items={
            "widget": {"cat": "__no_such_cat__", "model": "zzzz", "needs_lookat": True}
        },
        containers={
            "bin": {
                "source": "import",
                "cat": "__no_such_cat__",
                "model": "yyyy",
                "sign": "__nope__",
            }
        },
        decor=[{"cat": "__no_such_cat__", "model": "xxxx"}],
        labels={"partq": "__no_such_label__"},
    )
    f = spec_findings(spec)
    rules = {x.rule for x in f}
    chk("ASSET001" in rules, "NEGATIVE: unbaked goal subject + target -> ASSET001")
    chk(
        {x.criticality for x in f if x.rule == "ASSET001"} == {CRITICAL, COSMETIC},
        "NEGATIVE: a goal asset is CRITICAL while decor is COSMETIC (never blocks a batch)",
    )
    chk("LABEL001" in rules, "NEGATIVE: missing container sign PNG -> LABEL001")
    chk(
        "LABEL002" in rules, "NEGATIVE: needs_lookat item with no decal PNG -> LABEL002"
    )
    chk("LABEL003" in rules, "NEGATIVE: missing part placard PNG -> LABEL003")
    chk(
        all(BAKE_CMD in x.fix or "restore" in x.fix or "Author" in x.fix for x in f),
        "every finding ships a fix",
    )

    if os.path.isdir(CACHE_USD):
        div = scene_query_divergence("Rs_int")
        chk(
            div.get("bake_only") == [] and div.get("loader_only") == [],
            f"Rs_int query divergence: {div}",
        )
        chk(
            len(scene_loadable_models("Rs_int")) > 0,
            "the loader query returns models for Rs_int",
        )
    else:
        print("  [skip] bake cache absent -> scene queries not exercised")
    print(f"[asset_preflight] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

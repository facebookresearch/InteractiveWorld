# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve a DESCRIPTION written where an object NAME was expected.

A real person never types a name. They walk to the thing they can see. The naming indirection is an
artefact of the simulation: scored objects are renamed to opaque keys (``item_1``, ``unit_a..unit_f``)
while decoys keep their descriptive BEHAVIOR names, so a perfect English instruction -- "the locker
with the LINE CHARLIE placard" -- has 107 lexical matches in the person's vocabulary and the answer is
not one of them. This module removes the indirection: the person may write what it SEES between the
angle brackets and the engine binds it.

    [Walk] <the locker with the LINE CHARLIE placard> (1)   ->  unit_b
    [Grab] <the tupperware on the bench> (1)                ->  item_1

THREE OUTCOMES, and the middle one is the point:

    exactly one match  ->  act on it
    two or more        ->  AMBIGUOUS. Do not guess -- the person is told to ask "Which one do you
                           mean?". Without this the question almost never arises, because a person
                           cannot ask which of two candidates you mean about a set it cannot
                           enumerate. Here the engine enumerates, so "two of them fit" becomes a
                           STRUCTURAL outcome.
    zero               ->  the existing honest refusal ("I don't see anything like that")

⚠️  THE LEAK RULE -- the one way to ruin this.

    The person may know what things LOOK LIKE and WHERE they are.
    The person must NEVER know what they are FOR.

The resolver serves the PERSON, which is the measuring instrument and is not scored. It must never
serve the ASSISTANT, which is the subject. So it matches on FOUR features and no others:

    1. the object's category noun   ("locker", "reagent bottle")   -- appearance
    2. the text printed on it       ("LINE CHARLIE")               -- appearance (it is painted on)
    3. what it is standing on       ("on the lab bench")           -- position
    4. bearing + distance from the person ("8 m to your right")    -- position

and it is FORBIDDEN the answer key, the goal state, the routing rule, an object's destination and its
contents. ``[PutInside] <it> <the correct hazard cabinet>`` must NOT resolve: "correct" is task
semantics and deciding it is exactly what the assistant is scored on. That class is rejected by
``_VETO`` below, loudly, rather than being silently ignored -- see ``_selftest_leak``.

The prohibition is ENFORCED, not asserted: ``_selftest_audit`` greps this module's own source for the
forbidden ``behavior_task`` keys and fails if any of them is ever read.

DEFAULT OFF. ``command_grammar`` consults this module only when ``ISAAC_DESC_RESOLVE=1``, so every
other consumer of the grammar -- offline checks, training rollouts, other grounding modes -- runs on a
byte-identical path.

    bash launch/run_isaac.sh simulator/physical_world/desc_resolve.py    # offline selftest (no Isaac)
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------------------------------
#  the ONE hook onto live state that the scene object cannot give us
# ---------------------------------------------------------------------------------------------------
# Bearing and distance are measured from where the person is standing RIGHT NOW, and ``SceneState``
# does not know where that is (the rig does). A caller that wants position features installs a getter
# returning ``((x, y), yaw_deg)``; with none installed the position features are simply unavailable and
# the resolver degrades to category + placard + support, which can only make it ask MORE often. It can
# never make it act on something it should not have.
ACTOR_POSE: Callable[[], tuple[tuple[float, float] | None, float]] | None = None

# Optional narrowing of the candidate set to the vocabulary the person was actually given this turn
# (e.g. a room-level scope filter). None = the whole registry.
#
# ⚠️ INVARIANT ON WHOEVER SETS THIS, and it is not optional: the set handed over must already have
# been checked to contain every scored answer. A scope is a PROXY, and a proxy that drops a scored
# container turns "the resolver refused" into a measurement of the scope rather than of the resolver --
# invisibly, because the candidate list is still full and well formed and only the one name that
# mattered is gone. Example: moving six AABB centres by centimetres can flip a k-nearest room vote and
# take verbal binding 1.00 -> 0.00 with the fallback counter reading zero throughout. This module
# cannot check the invariant itself -- the answer key is exactly what it is forbidden to read -- so the
# check lives in the harness, which legitimately owns it.
SCOPE_NAMES: set[str] | None = None

# Every resolution attempt, in order, for the measurement. The engine acts on the RESOLVED key while
# the transcript records the raw text the person typed, so binding accuracy is only recoverable from
# this log. Reset per episode by the runner.
LOG: list[dict] = []


def enabled() -> bool:
    """``ISAAC_DESC_RESOLVE=1`` arms the fallback. Read per call: a probe process may flip it."""
    return os.getenv("ISAAC_DESC_RESOLVE", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def reset_log() -> None:
    LOG.clear()


# ---------------------------------------------------------------------------------------------------
#  vocabulary
# ---------------------------------------------------------------------------------------------------
_STOP: frozenset[str] = frozenset(
    """a an the this that these those it its is are was be to of and or for from you your yours
    my mine me i we our there here one ones thing things some any about approximately roughly
    please just now then over near side sided m metre metres meter meters metres away go get
    which what where s""".split()
)

# Bearing / distance clauses, removed before ANYTHING else looks at the words. "to your right" must
# not be read as the adjective "right" by the leak veto below, and "about 8 m" must not be read as a
# noun by the category filter.
_BEARING_RE: re.Pattern = re.compile(
    r"\b(?:right\s+in\s+front\s+of\s+(?:you|me)|in\s+front\s+of\s+(?:you|me)"
    r"|(?:over\s+)?(?:to|on)\s+(?:your|the|my)\s+(?:left|right)(?:\s+hand\s+side)?"
    r"|ahead\s+of\s+(?:you|me)|behind\s+(?:you|me)|straight\s+ahead|right\s+behind\s+(?:you|me))\b"
)
_DIST_RE: re.Pattern = re.compile(
    r"\b(?:about|around|roughly|approx\w*)?\s*(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)\b"
)

# Where a head noun phrase stops and a modifier begins. Split on the FIRST of these.
_MODIFIER_RE: re.Pattern = re.compile(
    r"\s+(?:with|on|in|inside|next\s+to|beside|by|under|underneath|above|over|near|at|labell?ed|"
    r"marked|reading|that\s+says|which\s+says|says)\s+"
)
# "on the lab bench" / "in the fridge" -- the SUPPORT clause specifically.
_SUPPORT_RE: re.Pattern = re.compile(
    r"\b(?:on|in|inside|on\s+top\s+of)\s+(?:the\s+|a\s+|an\s+)?([a-z0-9 \-]+)"
)

# A small, deliberately modest synonym map so the resolver is not merely an echo of the phrasing the
# oracle happens to generate. Symmetric, and it has no promiscuous hub ("container" is deliberately
# absent -- it would join bins to tupperware to cabinets and turn every description ambiguous).
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("cabinet", "locker", "cupboard", "closet", "wardrobe", "press"),
    ("bottle", "flask", "vial", "jar"),
    ("bin", "trashcan", "wastebasket", "wastebin", "dustbin"),
    ("shelf", "shelving", "rack", "bookcase"),
    ("fridge", "refrigerator", "freezer"),
    ("bench", "counter", "countertop", "worktop", "table"),
    ("tupperware", "tub", "tuppeware"),
    ("case", "display"),
    ("sofa", "couch"),
    ("bag", "sack"),
)


def _synonyms() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for grp in _SYNONYM_GROUPS:
        for w in grp:
            out.setdefault(w, set()).update(grp)
    return out


# ---------------------------------------------------------------------------------------------------
#  THE LEAK VETO -- phrasings that ask the engine to make the assistant's decision
# ---------------------------------------------------------------------------------------------------
# These are refused OUTRIGHT rather than allowed to fall through. Falling through would usually be
# harmless (a word the resolver does not understand simply fails to narrow, so the outcome is "ask"),
# but "usually" is not a guarantee: in a scene with exactly one cabinet, "the correct cabinet" would
# resolve, and the assistant would have been handed the routing decision it is scored on. The refusal
# is the contract, and it is tested adversarially in ``_selftest_leak``.
_VETO: tuple[tuple[str, str], ...] = (
    (r"\bcorrect\b", "which one is correct is the task, not the appearance"),
    (r"\bincorrect\b", "which one is correct is the task, not the appearance"),
    (r"\bwrong\b", "which one is wrong is the task, not the appearance"),
    (
        r"\bright\b",
        "'the right one' is a task judgement (bearings are stripped before this test)",
    ),
    (
        r"\bproper\b|\bappropriate\b|\bsuitable\b|\bdesignated\b|\bintended\b|\brequired\b",
        "a fitness judgement is the task, not the appearance",
    ),
    (
        r"\bmatching\b|\bmatches\b|\bthat\s+match",
        "'matching' compares against the routing rule",
    ),
    (
        r"\bbest\b|\bsafest\b|\bsafe\b",
        "a fitness judgement is the task, not the appearance",
    ),
    (
        r"\bbelongs?\b|\bsupposed\s+to\b|\bmeant\s+(?:for|to)\b|\bshould\s+go\b|\bgoes\s+(?:in|on|into|inside)\b",
        "where it goes is the answer key",
    ),
    (
        r"\bthe\s+(?:answer|target|destination|goal|solution)\b",
        "that names the answer key",
    ),
    (
        r"\bclass\b|\bcategory\s+for\b|\bhazard\s+for\b",
        "a routing class is the rule, not the appearance",
    ),
    (
        r"\bcontaining\b|\bfull\s+of\b|\bwith\s+the\s+[a-z ]+\s+in\s+it\b|\bwith\s+[a-z ]+\s+inside\b",
        "contents are hidden from the person",
    ),
    (r"\bwhere\s+(?:it|they|this|that)\b", "where it goes is the answer key"),
    (r"\b(?:other|remaining)\b", "'the other one' is a deduction over the answer set"),
    # "the locker FOR THE OXIDISERS" -- a plural class noun after "for" names the ROUTING RULE. A
    # singular one is left alone ("the bin for paper" could be painted on the bin), and it is safe to:
    # an unrecognised word cannot select, it can only fail to narrow, so the outcome is "ask".
    (
        r"\bfor\s+(?:the\s+|any\s+|all\s+)?\w{3,}s\b",
        "a class of contents is the routing rule",
    ),
)


# ---------------------------------------------------------------------------------------------------
#  the result
# ---------------------------------------------------------------------------------------------------
@dataclass
class Resolution:
    """What one description resolved to, and enough of the working to audit it after the fact."""

    outcome: str  # "unique" | "ambiguous" | "none" | "refused" | "off"
    key: str | None = None
    message: str = ""
    candidates: list[str] = field(
        default_factory=list
    )  # capped for the log; n_cand is the truth
    n_cand: int = 0
    raw: str = ""
    why: str = ""


# ---------------------------------------------------------------------------------------------------
#  entity APPEARANCE features -- category noun, printed placard, support. Cached per scene.
# ---------------------------------------------------------------------------------------------------
_FEATS: dict[int, tuple[int, dict[str, set[str]]]] = {}


def _clean_cat(cat: str) -> str:
    """``reagent_bottle/abcdef`` / ``lab_bench_zzabcd_0`` -> ``"reagent bottle"`` / ``"lab bench"``.

    BEHAVIOR's 6-letter model hash and the instance number carry no appearance, so they come off.
    Same derivation the oracle uses for its own English, deliberately: the person's binding and the
    assistant's sentence must be talking about the same word.
    """
    cat = (cat or "").split("/")[0].strip()
    cat = re.sub(r"_[a-z]{4,8}_\d+$", "", cat)
    cat = re.sub(r"_\d+$", "", cat)
    cat = re.sub(r"_[a-z]{6}$", "", cat)
    return re.sub(r"[_\-]+", " ", cat).strip()


def _usd_category(scene: Any, key: str) -> str:
    """``/World/bu_unit_b`` -> ``"locker"``, read back off the prim's USD reference.

    This is the ONLY channel by which an opaque key (``unit_b``) has a visible category at all: the
    key was renamed precisely so it would not say what it is, and the asset it references still does.
    """
    ent = (getattr(scene, "entities", None) or {}).get(key)
    path = getattr(ent, "prim_path", None)
    stage = getattr(scene, "stage", None)
    if not path or stage is None:
        return ""
    try:
        prim = stage.GetPrimAtPath(path)
        for spec in prim.GetPrimStack():
            for ref in getattr(spec, "referenceList", None).prependedItems or []:
                parts = str(ref.assetPath).replace(".usd", "").split("/")
                if len(parts) >= 2:
                    return _clean_cat(parts[-2])
    except Exception:  # noqa: BLE001 - a missing category downgrades the match, never the run
        return ""
    return ""


def _grasp_category(scene: Any) -> dict[str, str]:
    """``scene.grasp_objs`` -> key -> category, for the graspables the builder sited itself."""
    out: dict[str, str] = {}
    for g in getattr(scene, "grasp_objs", None) or []:
        key = str(g.get("key") or "")
        body = str((g.get("build") or {}).get("behavior") or "")
        if key and body:
            out[key] = _clean_cat(body)
    return out


# A key contributes WORDS only when it is a real BEHAVIOR instance name -- ``<category>_<6-letter model
# hash>_<index>``, e.g. ``locker_gemgfz_0``. Every other spelling in the registry is the benchmark's own
# bookkeeping (``unit_b``, ``item_1``, ``tool_a``, ``p_1``: ~25 prefixes, no fixed list worth
# enumerating), and bookkeeping is not appearance. Letting `unit_b` answer to "the unit" would hand the
# person the ROSTER of which six things are special, which is itself a task defect.
# The rule is positive and fails CLOSED: an anonymised key is describable by its ASSET and its placard
# or not at all.
_BEHAVIOR_KEY_RE: re.Pattern = re.compile(r"_[a-z]{4,8}_\d+$")


def _noun_tokens(scene: Any) -> dict[str, set[str]]:
    """key -> the words a person would use for what that thing IS. Cached; bodies never change."""
    ents = getattr(scene, "entities", None) or {}
    hit = _FEATS.get(id(scene))
    if hit is not None and hit[0] == len(ents):
        return hit[1]
    gcat = _grasp_category(scene)
    syn = _synonyms()
    feats: dict[str, set[str]] = {}
    for key in (str(k) for k in ents):
        sources = [gcat.get(key, ""), _usd_category(scene, key)]
        if _BEHAVIOR_KEY_RE.search(key):
            sources.append(_clean_cat(key))
        words: set[str] = set()
        for cat in sources:
            words.update(w for w in cat.split() if len(w) >= 3 and w not in _STOP)
        expanded = set(words)
        for w in words:
            expanded.update(syn.get(w, ()))
            expanded.add(w.rstrip("s") if w.endswith("s") else w + "s")
        feats[key] = expanded
    _FEATS[id(scene)] = (len(ents), feats)
    return feats


def category_report(scene: Any) -> dict[str, list[str]]:
    """key -> its derived appearance words, for the run log.

    Printed at episode start because the whole approach rests on one fact that is cheap to check and
    fatal to assume: an ANONYMISED key has a category only if its USD reference can be read back. If
    that lookup fails, `unit_b` has no words at all and no description can ever reach it -- the
    approach fails CLOSED (the person cannot act) rather than silently binding the wrong thing, but
    the run would be measuring the lookup, not the idea.
    """
    return {k: sorted(v) for k, v in _noun_tokens(scene).items()}


def _zone_text(scene: Any) -> dict[str, str]:
    """key -> the string PRINTED on that unit's placard.

    ⚠️ ``behavior_task["zones"]`` is the only member of that dict this module may read, and this is the
    only place it is read. A placard is paint: the person can see it on the video and quote it back --
    it demonstrably does. What a placard MEANS -- which placard a nitric-acid bottle belongs under --
    is the routing rule, lives in ``answer_key`` / ``goals`` / ``cue_class``, and is never touched.
    """
    bt = getattr(scene, "behavior_task", None) or {}
    return {
        str(k): str(v)
        for k, v in (bt.get("zones") or {}).items()
        if str(v).strip() not in ("", "-")
    }


# ---------------------------------------------------------------------------------------------------
#  description PARSING
# ---------------------------------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """The person's raw argument -> a flat lowercase phrase.

    ``command_grammar.parse_line`` lowercases and turns spaces into underscores before we ever see it,
    so the underscores come back out first. Quotes and punctuation go; hyphens stay, because placard
    strings carry them (``ALLERGEN-FREE``, ``HOT LINE - NO STORAGE``).
    """
    s = (text or "").replace("_", " ").lower()
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    return " ".join(s.split())


def _tokens(phrase: str) -> set[str]:
    return {
        w
        for w in re.split(r"[\s\-]+", phrase)
        if w and w not in _STOP and not w.isdigit()
    }


@dataclass
class Parsed:
    """The description, taken apart into the four features the resolver is allowed to use."""

    flat: str = ""  # everything, bearings and distances removed
    head: str = ""  # the noun phrase before the first modifier
    support: str = ""  # the "on the X" clause, if any
    side: str = ""  # "left" | "right" | "ahead" | "behind"
    dist_m: float | None = None


def _parse(text: str) -> Parsed:
    raw = _normalise(text)
    side = ""
    mb = _BEARING_RE.search(raw)
    if mb:
        seg = mb.group(0)
        side = (
            "left"
            if "left" in seg
            else "right"
            if "right" in seg and "in front" not in seg
            else "behind"
            if "behind" in seg
            else "ahead"
        )
    md = _DIST_RE.search(raw)
    dist = float(md.group(1)) if md else None
    flat = " ".join(_DIST_RE.sub(" ", _BEARING_RE.sub(" ", raw)).split())
    ms = _MODIFIER_RE.search(flat)
    head = (flat[: ms.start()] if ms else flat).strip()
    sup = ""
    msup = _SUPPORT_RE.search(flat)
    if msup:
        sup = msup.group(1).strip()
    return Parsed(flat=flat, head=head or flat, support=sup, side=side, dist_m=dist)


def _veto(p: Parsed) -> str:
    """The refusal reason for a task-semantic phrasing, or "" when the description is pure appearance."""
    for pat, why in _VETO:
        if re.search(pat, p.flat):
            return why
    return ""


# ---------------------------------------------------------------------------------------------------
#  POSITION features -- only ever a tie-break, never an eliminator
# ---------------------------------------------------------------------------------------------------
def _actor() -> tuple[tuple[float, float] | None, float]:
    if ACTOR_POSE is None:
        return None, 0.0
    try:
        xy, yaw = ACTOR_POSE()
        return (None if xy is None else (float(xy[0]), float(xy[1]))), float(yaw)
    except Exception:  # noqa: BLE001 - a missing pose drops the tie-break, never the run
        return None, 0.0


def _xy(scene: Any, key: str) -> tuple[float, float] | None:
    try:
        p = scene.object_pos(key)
        return (float(p[0]), float(p[1]))
    except Exception:  # noqa: BLE001 - same
        ent = (getattr(scene, "entities", None) or {}).get(key)
        xy = getattr(ent, "xy", None)
        return (float(xy[0]), float(xy[1])) if xy else None


def _side_of(
    actor: tuple[float, float], yaw_deg: float, xy: tuple[float, float]
) -> tuple[str, float]:
    """(which way it lies from the person, how far). Same convention as the oracle's own phrasing.

    ⚠️ The rig's forward is **-Y**, not +X (``humanoid_motion._body_forward`` is
    ``[sin(yaw), -cos(yaw)]``), so the obvious ``atan2(dy, dx) - yaw`` reads the world 90 degrees
    rotated -- it calls a thing STRAIGHT AHEAD "to your right" at every yaw. This helper must agree
    exactly with the oracle's bearing convention, or the resolver would be filtering on one convention
    while the assistant speaks another.
    Pinned by ``_selftest_bearing`` against the rig's own forward vector.
    """
    dist = math.hypot(xy[0] - actor[0], xy[1] - actor[1])
    rel = math.degrees(math.atan2(xy[0] - actor[0], -(xy[1] - actor[1]))) - yaw_deg
    rel = (rel + 180.0) % 360.0 - 180.0
    if rel > 135.0 or rel < -135.0:
        return "behind", dist
    if rel > 45.0:
        return "left", dist
    if rel < -45.0:
        return "right", dist
    return "ahead", dist


# ---------------------------------------------------------------------------------------------------
#  the resolver
# ---------------------------------------------------------------------------------------------------
def _candidates(scene: Any) -> list[str]:
    names = [str(k) for k in (getattr(scene, "entities", None) or {})]
    if SCOPE_NAMES:
        held = {str(h) for h in (getattr(scene, "held", None) or ())}
        names = [n for n in names if n in SCOPE_NAMES or n in held]
    return sorted(names)


def _narrow(keep: list[str], sub: list[str]) -> list[str]:
    """Apply a filter, but never let it EMPTY the set.

    Every filter here except the category one is soft. A person who has turned since the assistant
    spoke has a stale bearing, and a bearing that deletes the right answer turns a wrong choice into
    an unsayable one -- the same failure mode that rules out a visible-only candidate set.
    """
    return sub if sub else keep


def _by_placard(scene: Any, keep: list[str], p: Parsed) -> tuple[list[str], str]:
    """Keep the candidates whose PRINTED placard string appears in the description."""
    zones = _zone_text(scene)
    padded = f" {p.flat} "
    hit = [k for k in keep if k in zones and f" {_normalise(zones[k])} " in padded]
    if not hit:
        return keep, ""
    return hit, f'placard "{zones[hit[0]]}"'


def _by_support(scene: Any, keep: list[str], p: Parsed) -> tuple[list[str], str]:
    """Keep the candidates resting on something the description's "on the X" clause names."""
    if not p.support:
        return keep, ""
    want = _tokens(p.support)
    if not want:
        return keep, ""
    on = getattr(scene, "on", None) or {}
    feats = _noun_tokens(scene)
    hit = [k for k in keep if on.get(k) and (feats.get(str(on[k]), set()) & want)]
    if not hit:
        return keep, ""
    return hit, f"on the {p.support}"


def _by_position(scene: Any, keep: list[str], p: Parsed) -> tuple[list[str], str]:
    """Bearing then distance, each soft. Only ever narrows a set that is already category-correct."""
    actor, yaw = _actor()
    if actor is None or (not p.side and p.dist_m is None):
        return keep, ""
    poses = {k: _xy(scene, k) for k in keep}
    geom = {k: _side_of(actor, yaw, xy) for k, xy in poses.items() if xy is not None}
    if not geom:
        return keep, ""
    out, why = keep, ""
    if p.side:
        hit = [k for k in keep if geom.get(k, ("", 0.0))[0] == p.side]
        if hit:
            out, why = hit, p.side
    if p.dist_m is not None:
        tol = max(1.5, 0.35 * p.dist_m)
        hit = [k for k in out if k in geom and abs(geom[k][1] - p.dist_m) <= tol]
        if hit:
            out = hit
            why = f"{why}, {p.dist_m:g} m".strip(", ")
    return out, why


def resolve(scene: Any, text: str) -> Resolution:
    """One description -> a Resolution. Never raises; the caller decides what to do with the outcome."""
    p = _parse(text)
    bad = _veto(p)
    if bad:
        return Resolution(
            outcome="refused",
            raw=text,
            why=bad,
            message=(
                f'Refused "{p.flat}": you can only go by what a thing LOOKS like and where it '
                f"is, not by what it is for. Ask the assistant to describe it."
            ),
        )
    head = _tokens(p.head)
    if not head:
        return Resolution(
            outcome="none",
            raw=text,
            why="no noun",
            message=f'Nothing here matches "{p.flat}".',
        )
    feats = _noun_tokens(scene)
    keep = [k for k in _candidates(scene) if feats.get(k, set()) & head]
    if not keep:
        # A modifier-only description ("the one with the LINE CHARLIE placard") still deserves a try.
        allw = _tokens(p.flat)
        keep = [k for k in _candidates(scene) if feats.get(k, set()) & allw]
    if not keep:
        return Resolution(
            outcome="none",
            raw=text,
            why="no category match",
            message=f'Nothing here matches "{p.flat}". Say you cannot see anything like that.',
        )
    trail: list[str] = []
    for fn in (_by_placard, _by_support):
        keep, why = fn(scene, keep, p)
        if why:
            trail.append(why)
    if len(keep) > 1:
        keep, why = _by_position(scene, keep, p)
        if why:
            trail.append(why)
    if len(keep) == 1:
        return Resolution(
            outcome="unique",
            key=keep[0],
            raw=text,
            candidates=keep,
            n_cand=1,
            why=" + ".join(trail) or "category",
            message=f'"{p.flat}" -> {keep[0]}',
        )
    # ⚠️ ONLY THE FIRST LINE, AND ONLY ITS FIRST 160 CHARACTERS, REACHES THE PERSON. The driver builds
    # ``Your last command <cmd> was REJECTED: {str(exc).splitlines()[0][:160]}. Re-issue it correctly
    # this turn.`` (``episode_loop``) -- and that trailing sentence is a direct instruction to
    # RE-ISSUE, which is the opposite of what an ambiguity calls for. Observed: on an ambiguity the
    # person tends to re-describe instead of asking. Since the suffix is not ours to change, line 1 has
    # to countermand it in place, inside the budget, and name the exact sentence to say. The full
    # candidate list stays on line 2 for the log, where the truncation drops it.
    return Resolution(
        outcome="ambiguous",
        raw=text,
        candidates=sorted(keep)[:12],
        n_cand=len(keep),
        why=" + ".join(trail),
        message=(
            f'AMBIGUOUS: {len(keep)} things fit "{p.flat[:20]}" -- you cannot tell them apart. '
            f'Do NOT re-issue; say exactly: "Which one do you mean?"'
            f"\ncandidates: {', '.join(sorted(keep)[:8])}"
        ),
    )


def resolve_logged(scene: Any, text: str) -> Resolution:
    """``resolve`` plus the audit row. The transcript records the DESCRIPTION the person typed while
    the engine acts on the resolved KEY, so binding accuracy is only recoverable from this log."""
    r = resolve(scene, text)
    LOG.append(
        {
            "raw": text,
            "flat": _parse(text).flat,
            "outcome": r.outcome,
            "key": r.key,
            "n_cand": r.n_cand,
            "candidates": r.candidates[:12],
            "why": r.why,
        }
    )
    return r


# ===================================================================================================
#  Offline selftest (no Isaac): a fake scene, the happy path, the ambiguity path, the leak veto, and
#  a source-level audit that this module never reads a forbidden fact.
# ===================================================================================================
class _FakeEnt:
    def __init__(self, xy: tuple[float, float], prim_path: str | None = None) -> None:
        self.xy = xy
        self.prim_path = prim_path


class _FakeRef:
    def __init__(self, path: str) -> None:
        self.assetPath = path  # noqa: N815 - USD's own spelling


class _FakeSpec:
    def __init__(self, path: str) -> None:
        self.referenceList = type("_L", (), {"prependedItems": [_FakeRef(path)]})()  # noqa: N815


class _FakePrim:
    def __init__(self, path: str) -> None:
        self._p = path

    def GetPrimStack(self) -> list:  # noqa: N802 - USD's own spelling
        return [_FakeSpec(self._p)]


class _FakeStage:
    """Just enough USD to exercise ``_usd_category`` -- the one lookup the whole approach rests on."""

    def __init__(self, by_path: dict[str, str]) -> None:
        self._by_path = by_path

    def GetPrimAtPath(self, path: str) -> _FakePrim:  # noqa: N802 - USD's own spelling
        return _FakePrim(self._by_path.get(path, "unknown/xxxxxx.usd"))


class _FakeScene:
    """The shape of ``school_chemistry`` in miniature: six opaque task units that are all lockers, one
    with each placard, plus the decoy lockers whose descriptive names are the whole problem."""

    def __init__(self) -> None:
        self.entities: dict[str, Any] = {}
        self.held: set[str] = set()
        self.on: dict[str, str] = {}
        by_path: dict[str, str] = {}
        zones = (
            "LINE ALPHA",
            "LINE BRAVO",
            "LINE CHARLIE",
            "LINE DELTA",
            "LINE ECHO",
            "LINE FOX",
        )
        for i, _z in enumerate(zones):
            k = f"unit_{chr(ord('a') + i)}"
            self.entities[k] = _FakeEnt((3.0 + i, -4.0), prim_path=f"/World/bu_{k}")
            by_path[f"/World/bu_{k}"] = "/opt/bcache/usd/locker/yhthgr.usd"
        for i in range(8):
            self.entities[f"locker_gemgfz_{i}"] = _FakeEnt((-30.0 - i, 12.0))
        self.entities["lab_bench_zzabcd_0"] = _FakeEnt((1.0, -1.0))
        self.entities["item_1"] = _FakeEnt((1.0, -1.1), prim_path="/World/Obj_item_1")
        by_path["/World/Obj_item_1"] = "/opt/bcache/usd/tupperware/aabbcc.usd"
        self.entities["tupperware_qqwwee_3"] = _FakeEnt((-25.0, 9.0))
        self.entities["tool_a"] = _FakeEnt(
            (2.0, -2.0)
        )  # a spec pseudonym with NO asset behind it
        self.on["item_1"] = "lab_bench_zzabcd_0"
        self.stage = _FakeStage(by_path)
        self.grasp_objs = [
            {"key": "item_1", "build": {"behavior": "tupperware/aabbcc"}}
        ]
        self.behavior_task = {
            "zones": {f"unit_{chr(ord('a') + i)}": z for i, z in enumerate(zones)},
            # present on purpose: the resolver must never look at any of these
            "goals": [("item_1", "unit_c")],
            "answer_key": {"item_1": "unit_c"},
            "answered": {"q1": "unit_c"},
            "cue_class": {"item_1": "OXIDISER"},
        }

    def object_pos(self, key: str):
        return (*self.entities[key].xy, 0.9)


def _selftest_happy(sc: _FakeScene, chk) -> None:
    """The two sentences the oracle actually generates, and the paraphrases a person would use."""
    for text, want, why in (
        (
            "the locker with the LINE CHARLIE placard",
            "unit_c",
            "the headline case: 14 lockers, 1 placard",
        ),
        ('the locker with the "LINE CHARLIE" placard', "unit_c", "quoted placard"),
        (
            "the_cupboard_with_the_line_charlie_placard",
            "unit_c",
            "underscored + a synonym for locker",
        ),
        ("the cabinet labelled LINE ECHO", "unit_e", "a different preposition"),
        (
            "the tupperware on the lab bench",
            "item_1",
            "support clause separates two tupperwares",
        ),
        ("the tupperware on the bench", "item_1", "support clause, shortened noun"),
        ("tupperware", None, "a bare noun with two of them is AMBIGUOUS, not a guess"),
    ):
        r = resolve(sc, text)
        if want is None:
            chk(
                r.outcome == "ambiguous",
                f"{why}: {text!r} -> {r.outcome} {r.candidates}",
            )
        else:
            chk(
                r.outcome == "unique" and r.key == want,
                f"{why}: {text!r} -> {r.outcome} {r.key} (want {want})",
            )


def _selftest_ambiguous(sc: _FakeScene, chk) -> None:
    """The mandatory case: a deliberately vague assistant against six in-scope cabinets."""
    for text, n_min in (("the cabinet", 14), ("the locker", 14), ("the cupboard", 14)):
        r = resolve(sc, text)
        chk(
            r.outcome == "ambiguous" and r.n_cand >= n_min,
            f"vague {text!r} -> {r.outcome} over {r.n_cand} candidates (>= {n_min})",
        )
        chk(
            "Which one do you mean?" in r.message,
            f"vague {text!r} message tells the person the exact sentence to say",
        )
    for pseudo in ("the unit", "the item", "the tool"):
        r = resolve(sc, pseudo)
        chk(
            r.outcome == "none",
            f"{pseudo!r} must NOT name a spec pseudonym: -> {r.outcome} (a key spelling is not appearance)",
        )
    r = resolve(sc, "the purple crate")
    chk(
        r.outcome == "none",
        f"unmatchable 'the purple crate' -> {r.outcome} (honest refusal)",
    )


def _selftest_leak(sc: _FakeScene, chk) -> None:
    """Adversarial phrasings. Every one of these asks the engine to make the assistant's decision."""
    for text in (
        "the correct hazard cabinet",
        "the right cabinet for this",
        "the cabinet where it belongs",
        "the cabinet this bottle should go in",
        "the locker that matches what I am holding",
        "the appropriate locker",
        "the safest cabinet",
        "the cabinet containing the acids",
        "the locker with the oxidisers inside",
        "the destination cabinet",
        "the other one",
        "the other cabinet",
        "the cabinet for the class of this item",
        "the locker for the oxidisers",
        "the cabinet where this goes",
        "the designated hazard locker",
        "the safest place for it",
        "the destination",
        "the best cabinet for this",
        "the cabinet it belongs in",
        "the right locker for this bottle",
        "the cabinet that matches the item i am holding",
    ):
        r = resolve(sc, text)
        chk(
            r.outcome == "refused",
            f"LEAK VETO {text!r} -> {r.outcome} ({r.why or r.message[:40]})",
        )
    # ... while the SPATIAL word "right" still works, which is why the veto runs after bearings go.
    r = resolve(sc, "the locker with the LINE DELTA placard about 5 m to your right")
    chk(
        r.outcome == "unique" and r.key == "unit_d",
        f"'to your right' is a bearing, not the adjective: -> {r.outcome} {r.key}",
    )


def _code_only(src: str) -> str:
    """The module's executable lines: comments, docstrings and the selftest itself removed.

    The audit has to read CODE, not prose -- this file discusses the forbidden facts at length and
    saying their names in a comment is the opposite of reading them.
    """
    body = src.split("#  Offline selftest", 1)[0]
    body = re.sub(r'"""(?:.|\n)*?"""', " ", body)
    return "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())


def _selftest_audit(chk) -> None:
    """Grep this module's own source. The leak rule is enforced, not asserted."""
    with open(os.path.abspath(__file__)) as fh:
        code = _code_only(fh.read())
    for forbidden in (
        "answer_key",
        "answered",
        "cue_class",
        "cue_number",
        "goals",
        "carcass",
        "neg_constraints",
        "sub_goals",
        "preload",
        "preload_keys",
        "container",
        "attached",
        "inside",
    ):
        chk(
            f'"{forbidden}"' not in code and f"'{forbidden}'" not in code,
            f"source audit: the resolver never reads behavior_task[{forbidden!r}]",
        )
    chk(
        code.count("behavior_task") == 1,
        f"source audit: behavior_task is read in exactly ONE place (found {code.count('behavior_task')})",
    )
    chk(
        code.count('"zones"') == 1,
        f'source audit: "zones" is the only member read (found {code.count(chr(34) + "zones" + chr(34))})',
    )


def _selftest_bearing(chk) -> None:
    """Pin the bearing convention to the rig's OWN forward vector, at four yaws.

    ``humanoid_motion._body_forward(yaw) = [sin(yaw), -cos(yaw)]`` and left is forward turned +90
    degrees. A thing placed ON the forward ray must read "ahead" and one on the left ray must read
    "left", at every yaw. The naive ``atan2(dy, dx) - yaw`` formula fails this at all four and calls
    everything ahead of the wearer "right" -- and this resolver would then be filtering on a convention
    the assistant's own sentence does not use.
    """
    for yaw in (0.0, 90.0, 180.0, -135.0):
        r = math.radians(yaw)
        fwd = (math.sin(r), -math.cos(r))
        left = (-fwd[1], fwd[0])
        for vec, want in (
            (fwd, "ahead"),
            (left, "left"),
            ((-left[0], -left[1]), "right"),
            ((-fwd[0], -fwd[1]), "behind"),
        ):
            got, _d = _side_of((0.0, 0.0), yaw, (5.0 * vec[0], 5.0 * vec[1]))
            chk(got == want, f"bearing yaw={yaw:+.0f} {want:<7}-> {got}")


def _selftest_position(sc: _FakeScene, chk) -> None:
    """Bearing/distance narrow, but they may never delete the only right answer."""
    global ACTOR_POSE
    ACTOR_POSE = lambda: ((0.0, -4.0), 0.0)  # noqa: E731 - a one-line test stub
    try:
        r = resolve(sc, "the locker about 3 m ahead of you")
        chk(
            r.outcome in ("unique", "ambiguous"),
            f"position tie-break ran -> {r.outcome} {r.candidates[:4]}",
        )
        # a STALE bearing (the person turned) must not make the answer unsayable
        r2 = resolve(sc, "the locker with the LINE CHARLIE placard 40 m behind you")
        chk(
            r2.outcome == "unique" and r2.key == "unit_c",
            f"a stale bearing is soft: still -> {r2.outcome} {r2.key}",
        )
    finally:
        ACTOR_POSE = None


def _selftest() -> int:
    ok = True

    def chk(cond: Any, msg: str) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    sc = _FakeScene()
    _selftest_bearing(chk)
    _selftest_happy(sc, chk)
    _selftest_ambiguous(sc, chk)
    _selftest_leak(sc, chk)
    _selftest_position(sc, chk)
    _selftest_audit(chk)
    print(f"[desc_resolve] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())

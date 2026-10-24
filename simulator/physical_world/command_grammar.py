# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""VirtualHome-style command grammar -> Goal sequence (the deterministic front-door of the executor).

Parses fixed structured commands -- `[Action] <name> (id)` and the two-arg `[Put] <cup> (1) <table> (1)`
-- into `goal_schema.Goal`, resolving object names via `scene_state` grounding and AUTO-INSERTING the
implicit navigate before a manipulation whose target we are not already standing at. This is a fixed
grammar (NOT free-form NL) so parsing is deterministic and reproducible -- the human/policy emits valid
commands; we only translate.

Grammar mirrors VirtualHome's parser verbatim (``evolving_graph/scripts.py:parse_script_line``):
  patt_action = r'^\[(\w+)\]'   (verb token = the Action enum member name, upper-cased)
  patt_params = r'\<(.+?)\>\s*\((.+?)\)'   (each object arg = <class_name> (instance_id))
  multiple lines separated by a newline OR VirtualHome's ', ' (comma-space).

The GROUNDED core is Walk/Run -> navigate, Grab -> pick, Put/PutBack -> place, Drop/Release -> release.
The full verb set, including the ANCHORED verbs (Open/Close/...), comes from the shared registry.

EGOCENTRIC LOCOMOTION ([Forward] / [TurnLeft] / [TurnRight] / [TurnAround]) is the one family that names
NO entity: it is expressed entirely in the body frame, so it can be issued to a human who cannot see the
room and therefore cannot name anything in it. Their optional argument is a bare number rather than a
`<class> (instance)` pair -- see `_PATT_SCALAR`:

    [Forward]            one pace (goal_schema.FORWARD_STEP_M metres)
    [Forward] 3          three paces      [Forward] 1.5m   an explicit distance
    [TurnLeft]           a quarter turn   [TurnLeft] 45    a partial turn, degrees
    [TurnRight]          a quarter turn right
    [TurnAround]         a fixed half turn (no argument)
    [Wait]               three seconds     [Wait] 30s     [Wait] <seconds> (30)

    bash launch/run_isaac.sh simulator/physical_world/command_grammar.py   # offline selftest (no Isaac)
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import runtime_capabilities  # noqa: E402
from goal_schema import (  # noqa: E402
    FORWARD_STEP_M,
    Goal,
    Primitive,
    TURN_AROUND_DEG,
    TURN_DEG,
    WAIT_MAX_S,
    WAIT_MIN_S,
)

# VirtualHome's two regexes, verbatim (from its evolving_graph/scripts.py).
_PATT_ACTION = re.compile(r"^\[(\w+)\]")
_PATT_PARAMS = re.compile(r"\<(.+?)\>\s*\((.+?)\)")
# The EGOCENTRIC-LOCOMOTION scalar argument. These four verbs name no entity, so their optional argument
# is a plain NUMBER with an optional unit rather than VirtualHome's `<class> (instance)` pair:
#     [Forward] 3        [Forward] 3 steps      [Forward] 1.5m        [TurnLeft] 45      [TurnLeft] 45 deg
# The VH-shaped spelling `<steps> (3)` / `<m> (2)` / `<deg> (45)` is accepted too (integers only), so a
# spec author who tokenises everything the same way does not have to special-case them.
_PATT_SCALAR = re.compile(
    r"^([-+]?(?:\d+\.?\d*|\.\d+))\s*"
    r"(m|metre|metres|meter|meters|step|steps|deg|degree|degrees|s|sec|secs|second|seconds)?$",
    re.IGNORECASE,
)
_PATT_PLACE_SPOT = re.compile(
    r"(?:^|\s)@place_spot="
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?):"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_UNIT_CANON = {
    "m": "m",
    "metre": "m",
    "metres": "m",
    "meter": "m",
    "meters": "m",
    "step": "steps",
    "steps": "steps",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
}

# Command spelling, arity and executor metadata live in one shared registry. This module only turns
# the registry's primitive strings into the local enum used by compiled goals.
_VERB = {
    verb: Primitive(capability.primitive)
    for verb, capability in runtime_capabilities.action_by_verb().items()
}

# Sign of a turn token in the world XY frame. The body's forward is [sin(yaw), -cos(yaw)] and up is +Z,
# so the walker's LEFT hand is up x forward = +X, which is yaw = +90: turning left INCREASES yaw.
_TURN_SIGN = {
    "TURNLEFT": +1.0,
    "TURN_LEFT": +1.0,
    "TURNRIGHT": -1.0,
    "TURN_RIGHT": -1.0,
    "TURNAROUND": +1.0,
    "TURN_AROUND": +1.0,
}
# [TurnAround] is a FIXED half turn: a parameterised one would just be [TurnLeft] <deg>.
_TURN_FIXED = ("TURNAROUND", "TURN_AROUND")

# The verbs whose Goal carries no entity at all. Consumers that walk a compiled program looking for the
# places the human STANDS (episode_runtime._stand_targets) must skip these -- there is nothing to resolve.
LOCOMOTION_PRIMS = frozenset({Primitive.FORWARD, Primitive.TURN})

# Object-argument count per primitive (VH enforces arity too): navigate/pick take 1, place takes 2
# (<obj> <surface>), release takes 0.
_ARITIES = {
    Primitive(capability.primitive): capability.arities
    for capability in runtime_capabilities.action_by_primitive().values()
}
_ARITY = {primitive: arities[0] for primitive, arities in _ARITIES.items()}

# CONTACT verbs with two entity arguments. CUT/STIR/POUR retain the historical
# <target> <tool> order; WIPE's public spelling is <tool> <target>.
_CONTACT_2ARG = (Primitive.CUT, Primitive.STIR, Primitive.POUR, Primitive.WIPE)
_ASSEMBLY_PRIMS = (Primitive.ATTACH, Primitive.DETACH, Primitive.RAISE)
_DIRECT_INTERACTION_PRIMS = (*_CONTACT_2ARG, Primitive.LOOK_AT)
_PLACE_PRIMS = (Primitive.PLACE, Primitive.PLACE_INSIDE)
_MOVE_FIXTURE_PRIMS = (Primitive.PUSH, Primitive.PULL)
_STATE_TRANSITION_PRIMS = (Primitive.FALL_TO_FLOOR, Primitive.PLACE_ON_FLOOR)
_SINGLE_TARGET_PRIMS = (
    Primitive.PICK,
    Primitive.FLIP,
    Primitive.OPEN,
    Primitive.CLOSE,
    Primitive.OPENDOOR,
    Primitive.CLOSEDOOR,
    Primitive.TURN_ON,
    Primitive.TURN_OFF,
    Primitive.SIT,
    Primitive.COOK,
    Primitive.USE_COMPUTER,
    Primitive.ANSWER_PHONE,
)
_NOARG_PRIMS = (
    Primitive.RELEASE,
    Primitive.STANDUP,
    Primitive.LOOK_AROUND,
    Primitive.LOOK_SCAN,
    Primitive.SQUAT,
    Primitive.CONFIRM,
    Primitive.WAIT,
)


class CommandParseError(ValueError):
    """A command line that does not match the grammar / an unsupported verb / wrong arity."""


@dataclass(frozen=True)
class CompiledCommand:
    """One source command and the contiguous Goal range it produced."""

    index: int
    source: str
    goal_start: int
    goal_end: int


def navigation_followup(
    goals: Sequence[Goal],
    commands: Sequence[CompiledCommand],
    goal_index: int,
) -> Goal | None:
    """Return the next non-navigation goal in the same source command."""
    if not 0 <= goal_index < len(goals):
        return None
    if goals[goal_index].prim is not Primitive.NAVIGATE_TO:
        return None
    command = next(
        (
            candidate
            for candidate in commands
            if candidate.goal_start <= goal_index < candidate.goal_end
        ),
        None,
    )
    if command is None:
        return None
    for candidate in goals[goal_index + 1 : command.goal_end]:
        if candidate.prim is not Primitive.NAVIGATE_TO:
            return candidate
    return None


def navigation_action_followup(
    goals: Sequence[Goal],
    commands: Sequence[CompiledCommand],
    goal_index: int,
    scene=None,
) -> Goal | None:
    """Return the action an explicit or implicit navigation is preparing for."""
    direct = navigation_followup(goals, commands, goal_index)
    if direct is not None:
        return direct
    if not 0 <= goal_index < len(goals):
        return None
    navigation = goals[goal_index]
    if navigation.prim is not Primitive.NAVIGATE_TO or navigation.target_key is None:
        return None
    command_index = next(
        (
            index
            for index, command in enumerate(commands)
            if command.goal_start <= goal_index < command.goal_end
        ),
        None,
    )
    if command_index is None or command_index + 1 >= len(commands):
        return None
    next_command = commands[command_index + 1]
    action = next(
        (
            candidate
            for candidate in goals[next_command.goal_start : next_command.goal_end]
            if candidate.prim is not Primitive.NAVIGATE_TO
        ),
        None,
    )
    if action is None:
        return None
    target = action.meta.get("surface") or action.obj_key
    if target == navigation.target_key:
        return action
    if scene is None or action.obj_key is None:
        return None
    if action.prim not in (Primitive.PICK, Primitive.LOOK_AT, Primitive.FLIP):
        return None
    parent = (getattr(scene, "on", None) or {}).get(action.obj_key) or (
        getattr(scene, "inside", None) or {}
    ).get(action.obj_key)
    return action if parent == navigation.target_key else None


def parse_line(line: str) -> tuple[str, list[tuple[str, int]]] | None:
    """One command line -> (VERB_TOKEN_UPPER, [(name, instance), ...]); None if it is not a command."""
    s = line.strip()
    m = _PATT_ACTION.search(s)
    if not m:
        return None
    verb = m.group(1).upper()
    params = [
        (g.group(1).lower().replace(" ", "_"), int(g.group(2)))
        for g in _PATT_PARAMS.finditer(s[m.end(1) :])
    ]
    return verb, params


def _place_spot_arg(line: str) -> tuple[float, float] | None:
    """Parse the adapter-only normalized placement suffix.

    The colon spelling deliberately contains no comma because ``, `` separates
    VirtualHome commands. ``u`` runs left-to-right in the target's front-facing
    frame; ``v`` runs from its front edge (0) to back edge (1).
    """
    if "@place_spot" not in line:
        return None
    match = _PATT_PLACE_SPOT.search(line.strip())
    if match is None:
        raise CommandParseError(
            "place_spot must use @place_spot=<u>:<v> with numeric coordinates"
        )
    u, v = float(match.group(1)), float(match.group(2))
    if not -1.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
        raise CommandParseError("place_spot must satisfy -1 <= u <= 1 and 0 <= v <= 1")
    return u, v


def split_program(text: str) -> list[str]:
    """Split a program into command lines: newlines OR VirtualHome's ', ' (comma-space) separator.

    The split is BRACKET-AWARE: a ', ' inside `<...>` is not a separator. For a VirtualHome program
    this changes nothing -- an object class never contains a comma -- but a DESCRIPTION does, and
    natural English puts one exactly where it hurts:

        [Walk] <the cabinet with the "LINE ECHO" placard, located about 24 m behind me> (1)

    Split naively, that becomes `[Walk] <the cabinet with the "LINE ECHO" placard` plus an orphan
    fragment, and the first half raises `[WALK] expects 1 object arg(s), got 0` -- a message about
    arity, for a command whose only fault was a comma. Measured: four consecutive turns lost that way,
    and the resolver was never consulted at all, so the failure looked like the description not
    binding when the description had never arrived. Depth tracking makes it a non-issue and cannot
    change any input that was already well formed.
    """
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    src = text.replace("\n", ", ")
    i = 0
    while i < len(src):
        ch = src[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if depth == 0 and ch == "," and src[i + 1 : i + 2] == " ":
            out.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf).strip())
    return [c for c in out if c]


# Sanity cap on a single [Forward]. A guided walk is issued in paces; anything past this is a typo, and
# a typo should be a parse error rather than a 1 km stride through the building.
_FORWARD_MAX_M = 20.0


def _scalar_arg(
    verb: str, line: str, params: list[tuple[str, int]]
) -> tuple[float, str] | None:
    """The OPTIONAL scalar of a locomotion command -> (value, unit), or None when the token is bare.

    ``unit`` is "" (unqualified), "m", "steps", "deg" or "s". Anything that is not a number-with-optional-unit
    raises, so a mistyped ``[Forward] <cup> (1)`` is a parse error instead of a silent default pace."""
    if params:
        if len(params) != 1:
            raise CommandParseError(
                f"[{verb}] takes at most ONE argument, got {len(params)}"
            )
        unit = _UNIT_CANON.get(params[0][0])
        if unit is None:
            raise CommandParseError(
                f"[{verb}] argument <{params[0][0]}> is not a unit: use <steps>/<m>/<deg>/<seconds> (id = the count) "
                f"or a bare number"
            )
        return float(params[0][1]), unit
    s = line.strip()
    m = _PATT_ACTION.search(s)
    rest = s[m.end() :].strip() if m else ""
    if not rest:
        return None
    sm = _PATT_SCALAR.match(rest)
    if sm is None:
        raise CommandParseError(
            f"[{verb}] expects at most one number with an optional unit (e.g. '3', '1.5m', '45 deg', '30s'), "
            f"got {rest!r}"
        )
    return float(sm.group(1)), _UNIT_CANON.get((sm.group(2) or "").lower(), "")


def _forward_goal(verb: str, scalar: tuple[float, str] | None) -> Goal:
    """[Forward] -> a FORWARD goal carrying an absolute distance in metres.

    Bare = one pace. An unqualified or <steps> number counts PACES (``FORWARD_STEP_M`` each), which is how
    a person is actually guided; an "m"-suffixed number is metres."""
    if scalar is None:
        return Goal(Primitive.FORWARD, meta={"dist_m": FORWARD_STEP_M, "verb": verb})
    val, unit = scalar
    if unit == "deg":
        raise CommandParseError(f"[{verb}] takes a distance, not an angle in degrees")
    if unit == "s":
        raise CommandParseError(f"[{verb}] takes a distance, not a duration")
    dist = val if unit == "m" else val * FORWARD_STEP_M
    if not 0.0 < dist <= _FORWARD_MAX_M:
        raise CommandParseError(
            f"[{verb}] distance must be in (0, {_FORWARD_MAX_M:g}] m, got {dist:g} m"
        )
    return Goal(Primitive.FORWARD, meta={"dist_m": dist, "verb": verb})


def _turn_goal(verb: str, scalar: tuple[float, str] | None) -> Goal:
    """[TurnLeft]/[TurnRight]/[TurnAround] -> a TURN goal carrying a SIGNED angle (+ = left/CCW)."""
    sign = _TURN_SIGN[verb]
    if verb in _TURN_FIXED:
        if scalar is not None:
            raise CommandParseError(
                f"[{verb}] takes no argument (it is a fixed {TURN_AROUND_DEG:g} deg half turn); "
                f"use [TurnLeft]/[TurnRight] with degrees for a partial turn"
            )
        return Goal(
            Primitive.TURN, meta={"turn_deg": sign * TURN_AROUND_DEG, "verb": verb}
        )
    if scalar is None:
        deg = TURN_DEG
    else:
        val, unit = scalar
        if unit in ("m", "steps"):
            raise CommandParseError(
                f"[{verb}] takes an angle in degrees, not a distance in {unit}"
            )
        if unit == "s":
            raise CommandParseError(f"[{verb}] takes an angle, not a duration")
        deg = val
    if not 0.0 < deg <= 360.0:
        raise CommandParseError(
            f"[{verb}] angle must be in (0, 360] degrees, got {deg:g}"
        )
    return Goal(Primitive.TURN, meta={"turn_deg": sign * deg, "verb": verb})


def _locomotion_goal(verb: str, line: str, params: list[tuple[str, int]]) -> Goal:
    """One egocentric-locomotion command line -> its Goal (no scene grounding: it names no entity)."""
    scalar = _scalar_arg(verb, line, params)
    if _VERB[verb] is Primitive.FORWARD:
        return _forward_goal(verb, scalar)
    return _turn_goal(verb, scalar)


def _wait_goal(verb: str, line: str, params: list[tuple[str, int]]) -> Goal:
    scalar = _scalar_arg(verb, line, params)
    if scalar is None:
        return Goal(Primitive.WAIT)
    duration_s, unit = scalar
    if unit not in {"", "s"}:
        raise CommandParseError(f"[{verb}] takes a duration in seconds, not {unit}")
    if not WAIT_MIN_S <= duration_s <= WAIT_MAX_S:
        raise CommandParseError(
            f"[{verb}] duration must be in [{WAIT_MIN_S:g}, {WAIT_MAX_S:g}] seconds, "
            f"got {duration_s:g}"
        )
    return Goal(Primitive.WAIT, meta={"duration_s": duration_s})


def _resolve(scene, name: str, inst: int) -> str:
    """Map a `<name> (inst)` reference to a registered entity key: try '<name>' then '<name><inst>',
    case-insensitively (VH lowercases names; our scene keys may be mixed-case like 'deskB').

    A name that does not resolve MAY be a DESCRIPTION of the thing the speaker can see -- "the locker
    with the LINE CHARLIE placard" -- when ``ISAAC_DESC_RESOLVE=1``. That fallback runs only after
    every exact-name attempt has failed, so a command naming a real entity behaves exactly as it did
    before, byte for byte. See ``desc_resolve`` for the resolution rule and the leak veto."""
    lower_map = {k.lower(): k for k in scene.entities}
    for cand in (name, f"{name}{inst}"):
        if cand in scene.entities:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    hit = _resolve_description(scene, name)
    if hit is not None:
        return hit
    raise CommandParseError(
        f"no scene entity for <{name}> ({inst}) (have {sorted(scene.entities)})"
    )


def _resolve_description(scene, name: str):
    """The description fallback: ``<a description>`` -> the one entity it can only be, or a rejection.

    Returns the key on a UNIQUE match and None when the fallback is switched off, so the caller's
    existing "no scene entity" error is what a disabled build still raises. Everything else raises
    here, because the three non-unique outcomes each need their own sentence back to the speaker:
    two-or-more must produce a QUESTION, zero must produce the honest refusal, and a phrasing that
    asks the engine to make the assistant's decision must produce a hard no."""
    if os.getenv("ISAAC_DESC_RESOLVE", "0").strip().lower() not in (
        "1",
        "true",
        "on",
        "yes",
    ):
        return None  # tested BEFORE the import, so an off build never even loads the module
    import desc_resolve  # noqa: PLC0415 - flag-gated

    res = desc_resolve.resolve_logged(scene, name)
    if res.outcome == "unique":
        return res.key
    raise CommandParseError(res.message)


def _validate_command_arity(
    verb: str, prim: Primitive, params: list[tuple[str, int]]
) -> None:
    if prim is Primitive.PLACE_INSIDE:
        if len(params) not in (2, 3):
            raise CommandParseError(
                f"[{verb}] expects 2 or 3 args (<obj> <target> [<space>]), "
                f"got {len(params)}"
            )
        return
    if prim is Primitive.ATTACH:
        if len(params) not in (4, 5):
            raise CommandParseError(
                f"[{verb}] expects 4 args, plus optional "
                "<orientation_error_deg> (degrees), "
                f"got {len(params)}"
            )
        if len(params) == 5 and params[4][0] != "orientation_error_deg":
            raise CommandParseError(
                f"[{verb}] fifth arg must be <orientation_error_deg>, "
                f"got <{params[4][0]}>"
            )
        return
    if len(params) not in _ARITIES[prim]:
        raise CommandParseError(
            f"[{verb}] expects {_ARITY[prim]} object arg(s), got {len(params)}"
        )


def _compile_assembly_goal(prim: Primitive, params, scene, navigate) -> Goal:
    if prim is Primitive.ATTACH:
        child = _resolve(scene, *params[0])
        parent = _resolve(scene, *params[2])
        navigate(parent)
        orientation_error_deg = float(params[4][1]) if len(params) == 5 else 0.0
        if not 0.0 <= abs(orientation_error_deg) <= 180.0:
            raise CommandParseError(
                "[ATTACH] orientation error must be within [-180, 180] deg, "
                f"got {orientation_error_deg:g}"
            )
        return Goal(
            Primitive.ATTACH,
            meta={
                "child": child,
                "parent": parent,
                "child_point": params[1][0],
                "child_point_idx": int(params[1][1]),
                "parent_point": params[3][0],
                "parent_point_idx": int(params[3][1]),
                "orientation_error_deg": orientation_error_deg,
                "orientation_error_axis": (
                    "connector_y" if orientation_error_deg else None
                ),
                "injected_error_kind": (
                    "wrong_orientation" if orientation_error_deg else None
                ),
            },
        )
    child = _resolve(scene, *params[0])
    navigate(child)
    if prim is Primitive.DETACH:
        parent = _resolve(scene, *params[1])
        return Goal(
            Primitive.DETACH,
            obj_key=child,
            meta={"child": child, "parent": parent},
        )
    return Goal(Primitive.RAISE, meta={"root": child})


def _compile_direct_interaction_goal(
    prim: Primitive,
    params,
    scene,
    navigate,
    *,
    held_key: str | None = None,
) -> Goal:
    first = _resolve(scene, *params[0])
    if prim is Primitive.LOOK_AT:
        return Goal(prim, obj_key=first)
    second = _resolve(scene, *params[1])
    if prim is Primitive.WIPE:

        def _is_surface(key: str) -> bool:
            entity = scene.resolve(key)
            roles = [entity.role, *(entity.affordances or {})]
            if entity.target is not None:
                roles.append(entity.target.role)
            return any(getattr(role, "value", role) == "surface" for role in roles)

        first_surface = _is_surface(first)
        second_surface = _is_surface(second)
        legacy_order = held_key == second or (
            held_key is None and first_surface and not second_surface
        )
        tool, target = (second, first) if legacy_order else (first, second)
    else:
        target, tool = first, second
    navigate(target)
    return Goal(prim, obj_key=target, meta={"tool": tool})


def _compile_place_goal(
    prim: Primitive, params, scene, navigate, place_spot=None
) -> Goal:
    object_key = _resolve(scene, *params[0])
    target = _resolve(scene, *params[1])
    navigate(target)
    meta = {"surface": target, "_held_object_key": object_key}
    place_xy = None
    if place_spot is not None:
        space = params[2][0] if len(params) == 3 else None
        meta["place_spot"] = place_spot
        resolver = getattr(scene, "place_xy_from_spot", None)
        if callable(resolver):
            place_xy = resolver(
                target,
                place_spot,
                object_name=object_key,
                space=space,
            )
    if prim is Primitive.PLACE:
        return Goal(prim, place_xy=place_xy, meta=meta)
    space = params[2][0] if len(params) == 3 else None
    meta["space"] = space
    return Goal(prim, place_xy=place_xy, meta=meta)


def _compile_move_fixture_goal(prim: Primitive, params, scene, navigate) -> Goal:
    mover = _resolve(scene, *params[0])
    destination = _resolve(scene, *params[1])
    navigate(mover)
    return Goal(prim, obj_key=mover, meta={"destination": destination})


def _compile_state_transition_goal(
    prim: Primitive, params, scene, navigate, place_spot=None
) -> Goal:
    obj = _resolve(scene, *params[0])
    anchor = _resolve(scene, *params[1])
    if prim is Primitive.PLACE_ON_FLOOR:
        navigate(anchor)
    meta = {"anchor": anchor}
    place_xy = None
    if place_spot is not None:
        meta["place_spot"] = place_spot
        resolver = getattr(scene, "place_xy_from_spot", None)
        if callable(resolver):
            place_xy = resolver(anchor, place_spot, object_name=obj)
    return Goal(prim, obj_key=obj, place_xy=place_xy, meta=meta)


def _compile_non_locomotion_goal(
    prim: Primitive,
    params,
    scene,
    navigate,
    place_spot=None,
    *,
    held_key: str | None = None,
) -> Goal | None:
    if prim is Primitive.NAVIGATE_TO:
        navigate(_resolve(scene, *params[0]))
        return None
    if prim is Primitive.ANSWER_PHONE:
        return Goal(prim, obj_key=_resolve(scene, *params[0]))
    if prim in _SINGLE_TARGET_PRIMS:
        key = _resolve(scene, *params[0])
        navigation_key = key
        if prim is Primitive.USE_COMPUTER:
            entity = scene.resolve(key)
            anchor = str((entity.meta or {}).get("interaction_anchor") or "")
            if anchor:
                scene.resolve(anchor)
                navigation_key = anchor
        navigate(navigation_key)
        return Goal(prim, obj_key=key)
    if prim in _PLACE_PRIMS:
        return _compile_place_goal(prim, params, scene, navigate, place_spot)
    if prim in _MOVE_FIXTURE_PRIMS:
        return _compile_move_fixture_goal(prim, params, scene, navigate)
    if prim in _STATE_TRANSITION_PRIMS:
        return _compile_state_transition_goal(prim, params, scene, navigate, place_spot)
    if prim in _ASSEMBLY_PRIMS:
        return _compile_assembly_goal(prim, params, scene, navigate)
    if prim in _DIRECT_INTERACTION_PRIMS:
        return _compile_direct_interaction_goal(
            prim,
            params,
            scene,
            navigate,
            held_key=held_key,
        )
    if prim in _NOARG_PRIMS:
        return Goal(prim)
    raise CommandParseError(f"no compiler for primitive {prim.value!r}")


def compile_program(text: str, scene) -> list[Goal]:
    """Compile a program while preserving the historical Goal-list API."""
    goals, _commands = compile_program_with_commands(text, scene)
    return goals


def _held_hint_after_goal(
    held_key: str | None,
    prim: Primitive,
    goal: Goal,
) -> str | None:
    if prim is Primitive.PICK:
        return goal.obj_key
    if prim in (*_PLACE_PRIMS, Primitive.RELEASE):
        return None
    if prim is Primitive.WIPE and held_key is None:
        return str(goal.meta["tool"])
    return held_key


def compile_program_with_commands(
    text: str,
    scene,
    held_key: str | None = None,
) -> tuple[list[Goal], tuple[CompiledCommand, ...]]:
    """VirtualHome command string -> Goal list. Names are grounded via ``scene`` (scene_state), and the
    implicit navigate is auto-inserted before a manipulation whose target we are not already standing at.

    The auto-inserted navigate is UNRESOLVED -- it carries ``target_key`` (the entity to stand at) and
    leaves ``target_xy`` None. The concrete stand pose is chosen at RUNTIME, nearest-reachable from the
    human's LIVE position (``init_validation.pick_stand_runtime`` in the render loop). Offline callers
    with no physics handle (selftests / demos via ``execute``) resolve the key from ``scene.stand_pose_for``
    inside ``produce_navigate``. Stands are never chosen position-agnostically at init/compile time."""
    goals: list[Goal] = []
    commands: list[CompiledCommand] = []
    last_nav: str | None = None
    held_hint = held_key

    def _record_command(source: str, goal_start: int) -> None:
        commands.append(
            CompiledCommand(
                index=len(commands),
                source=source,
                goal_start=goal_start,
                goal_end=len(goals),
            )
        )

    def _nav(key: str) -> None:
        nonlocal last_nav
        if last_nav == key:
            return  # already standing there -> no redundant walk
        goals.append(Goal(Primitive.NAVIGATE_TO, target_key=key))
        last_nav = key

    program = tuple((line, parse_line(line)) for line in split_program(text))
    for command_offset, (line, parsed) in enumerate(program):
        if parsed is None:
            continue
        command_start = len(goals)
        verb, params = parsed
        prim = _VERB.get(verb)
        if prim is None:
            raise CommandParseError(
                f"unsupported verb [{verb}] (P0 supports {sorted(_VERB)})"
            )
        place_spot = _place_spot_arg(line)
        if (
            place_spot is not None
            and prim not in _PLACE_PRIMS
            and prim is not Primitive.FALL_TO_FLOOR
        ):
            raise CommandParseError(
                f"[{verb}] cannot carry @place_spot; only placement and FallToFloor support it"
            )
        if prim in LOCOMOTION_PRIMS:
            # No target -> no implicit walk BEFORE it, and it invalidates the "already standing there"
            # memory, so a manipulation issued AFTER a step or a turn gets its walk back.
            goals.append(_locomotion_goal(verb, line, params))
            last_nav = None
            _record_command(line, command_start)
            continue
        if prim is Primitive.WAIT:
            goals.append(_wait_goal(verb, line, params))
            _record_command(line, command_start)
            continue
        _validate_command_arity(verb, prim, params)
        if prim is Primitive.NAVIGATE_TO:
            key = _resolve(scene, *params[0])
            entity = scene.resolve(key)
            close_reposition = bool(
                (getattr(entity, "meta", None) or {}).get("close_reposition")
            )
            next_parsed = (
                program[command_offset + 1][1]
                if command_offset + 1 < len(program)
                else None
            )
            next_is_matching_close = False
            if close_reposition and next_parsed is not None:
                next_verb, next_params = next_parsed
                next_prim = _VERB.get(next_verb)
                next_is_matching_close = bool(
                    next_prim is Primitive.CLOSE
                    and len(next_params) == 1
                    and _resolve(scene, *next_params[0]) == key
                )
            if next_is_matching_close:
                goals.append(
                    Goal(
                        Primitive.NAVIGATE_TO,
                        target_key=key,
                        meta={"_close_reposition_walk": key},
                    )
                )
                last_nav = key
            else:
                _nav(key)
            _record_command(line, command_start)
            continue
        goal = _compile_non_locomotion_goal(
            prim,
            params,
            scene,
            _nav,
            place_spot,
            held_key=held_hint,
        )
        if goal is not None:
            goals.append(goal)
            held_hint = _held_hint_after_goal(held_hint, prim, goal)
        _record_command(line, command_start)
    return goals, tuple(commands)


def _selftest_locomotion(ss, _chk) -> None:
    """The four egocentric-locomotion tokens: bare defaults, every optional-argument spelling, the
    sign convention, that they insert NO walk, that they RESET the standing-there memory, and the
    arity/unit rejections."""
    gl = compile_program(
        "[Forward], [TurnLeft], [TurnRight], [TurnAround]",
        ss,
    )
    _chk(
        [g.prim for g in gl]
        == [Primitive.FORWARD, Primitive.TURN, Primitive.TURN, Primitive.TURN],
        f"bare locomotion program -> {[g.prim.value for g in gl]} (no walk auto-inserted)",
    )
    _chk(
        gl[0].meta["dist_m"] == FORWARD_STEP_M,
        f"[Forward] default = one pace ({FORWARD_STEP_M} m)",
    )
    _chk(
        [g.meta["turn_deg"] for g in gl[1:]] == [TURN_DEG, -TURN_DEG, TURN_AROUND_DEG],
        f"turn signs (+ = left) -> {[g.meta['turn_deg'] for g in gl[1:]]}",
    )
    _chk(
        all(g.target_key is None and g.obj_key is None for g in gl),
        "locomotion goals carry NO entity (target_key/obj_key stay None)",
    )

    # Optional argument, every accepted spelling: paces (bare / <steps>), metres, degrees.
    ga = compile_program(
        "[Forward] 3, [Forward] <steps> (2), [Forward] 1.5m, [Forward] <m> (2), "
        "[TurnLeft] 45, [TurnRight] <deg> (30), [TurnLeft] 45 deg",
        ss,
    )
    _chk(
        [round(g.meta.get("dist_m", 0.0), 6) for g in ga[:4]]
        == [round(3 * FORWARD_STEP_M, 6), round(2 * FORWARD_STEP_M, 6), 1.5, 2.0],
        f"[Forward] arg forms -> {[round(g.meta.get('dist_m', 0.0), 3) for g in ga[:4]]} m",
    )
    _chk(
        [g.meta["turn_deg"] for g in ga[4:]] == [45.0, -30.0, 45.0],
        f"[Turn*] arg forms -> {[g.meta['turn_deg'] for g in ga[4:]]} deg",
    )

    # A step or a turn moves the body, so the NEXT manipulation must get its walk back even though the
    # program already walked to that entity earlier (the 'already standing there' dedup is invalidated).
    gr = compile_program("[Walk] <cup> (1), [Forward] 2, [Grab] <cup> (1)", ss)
    _chk(
        [g.prim for g in gr]
        == [
            Primitive.NAVIGATE_TO,
            Primitive.FORWARD,
            Primitive.NAVIGATE_TO,
            Primitive.PICK,
        ],
        f"[Forward] resets the stand memory -> {[g.prim.value for g in gr]}",
    )

    # Rejections: [TurnAround] is fixed-arity-0, the units are checked, and an entity is not a scalar.
    for bad, why in (
        ("[TurnAround] 90", "[TurnAround] with an argument (wrong arity)"),
        ("[Forward] 45 deg", "[Forward] given an angle"),
        ("[TurnLeft] 2m", "[TurnLeft] given a distance"),
        ("[Forward] <cup> (1)", "[Forward] given an entity, not a unit"),
        ("[Forward] 0", "[Forward] with a non-positive distance"),
        ("[Forward] 100m", "[Forward] beyond the sanity cap"),
        ("[TurnLeft] 900", "[TurnLeft] beyond one full turn"),
        ("[Forward] 2 3", "[Forward] with two numbers"),
    ):
        try:
            compile_program(bad, ss)
            _chk(False, f"{why} should raise")
        except CommandParseError:
            _chk(True, f"{why} raises CommandParseError")


def _selftest_description(_chk) -> None:
    """The description fallback, end to end through ``compile_program``: OFF changes nothing, ON binds a
    description to the one entity it can only be, and the two non-unique outcomes still raise."""
    import desc_resolve

    sc = desc_resolve._FakeScene()
    prog = "[Walk] <the locker with the LINE CHARLIE placard> (1)"

    os.environ.pop("ISAAC_DESC_RESOLVE", None)
    try:
        compile_program(prog, sc)
        _chk(False, "flag OFF: a description must still be 'no scene entity'")
    except CommandParseError as exc:
        _chk(
            "no scene entity" in str(exc),
            f"flag OFF: unchanged error path ({str(exc)[:38]}...)",
        )

    # A ', ' INSIDE a description is not a statement separator. Measured live: four consecutive turns
    # lost to `[WALK] expects 1 object arg(s), got 0` on a perfectly good description of the right
    # cabinet, with the resolver never consulted -- so the failure read as "the description did not
    # bind" when the description had never arrived.
    _chk(
        split_program("[Walk] <the cabinet with the placard, about 24 m behind me> (1)")
        == ["[Walk] <the cabinet with the placard, about 24 m behind me> (1)"],
        "a comma INSIDE <> does not split the command",
    )
    _chk(
        split_program("[Walk] <cup> (1), [Grab] <cup> (1)")
        == ["[Walk] <cup> (1)", "[Grab] <cup> (1)"],
        "a comma BETWEEN commands still splits (VirtualHome separator intact)",
    )
    _chk(
        split_program("[Walk] <a, b> (1), [Grab] <c> (1)")
        == ["[Walk] <a, b> (1)", "[Grab] <c> (1)"],
        "both at once: split between, not inside",
    )

    os.environ["ISAAC_DESC_RESOLVE"] = "1"
    try:
        gl = compile_program(prog, sc)
        _chk(
            [g.prim for g in gl] == [Primitive.NAVIGATE_TO]
            and gl[0].target_key == "unit_c",
            f"flag ON: description -> {gl[0].target_key} (the placard is the only thing that tells "
            "14 lockers apart)",
        )
        gg = compile_program("[Grab] <the tupperware on the lab bench> (1)", sc)
        _chk(
            gg[-1].obj_key == "item_1",
            f"flag ON: 2-clause description -> {gg[-1].obj_key}",
        )
        # An EXACT name must not take the fallback at all -- byte-identical to a build with it off.
        ge = compile_program("[Walk] <unit_c> (1)", sc)
        _chk(
            ge[0].target_key == "unit_c",
            "flag ON: an exact name still resolves exactly",
        )
        for bad, want, why in (
            ("[Walk] <the cabinet> (1)", "AMBIGUOUS", "14 fit -> the person must ask"),
            (
                "[Walk] <the purple crate> (1)",
                "Nothing here matches",
                "0 fit -> honest refusal",
            ),
            (
                "[Walk] <the correct hazard cabinet> (1)",
                "Refused",
                "task semantics -> hard no",
            ),
        ):
            try:
                compile_program(bad, sc)
                _chk(False, f"{why}: should raise")
            except CommandParseError as exc:
                _chk(want in str(exc), f"{why}: {str(exc)[:60]}")
    finally:
        os.environ.pop("ISAAC_DESC_RESOLVE", None)


# =====================================================================================================
# Offline selftest (no Isaac): parse fidelity + name resolution + auto-navigate insertion.
# =====================================================================================================
def _selftest() -> int:
    from scene_state import Role, SceneState

    ok = True

    def _chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    ss = SceneState(start_xy=(0.0, 1.05))
    ss.register("cup", Role.GRASPABLE, xy=(0.0, -1.36), z=0.75, approach_yaw=0.0)
    ss.register("deskB", Role.SURFACE, xy=(0.0, 1.36), z=0.75, approach_yaw=180.0)

    # Regex fidelity (VH grammar): action token upper-cased, two <name>(id) args parsed.
    v, p = parse_line("[Put] <cup> (1) <table> (2)")
    _chk(v == "PUT" and p == [("cup", 1), ("table", 2)], f"parse_line -> {v} {p}")

    # Full VH program (', '-separated) -> the same 4-goal sequence the hand-built demo used.
    prog = "[Walk] <cup> (1), [Grab] <cup> (1), [Walk] <deskB> (1), [Put] <cup> (1) <deskB> (1)"
    goals = compile_program(prog, ss)
    prims = [g.prim for g in goals]
    _chk(
        prims
        == [
            Primitive.NAVIGATE_TO,
            Primitive.PICK,
            Primitive.NAVIGATE_TO,
            Primitive.PLACE,
        ],
        f"program -> {[x.value for x in prims]}",
    )
    _chk(goals[1].obj_key == "cup", "Grab resolved obj_key=cup")
    _chk(goals[3].meta.get("surface") == "deskB", "Put resolved surface=deskB")
    # The auto-inserted navigate is UNRESOLVED -- it carries the entity KEY, and the concrete
    # (target_xy, target_yaw) stand pose is picked at RUNTIME (nearest-reachable from the human's live
    # position), not at compile time. So target_xy is None here; only target_key is set.
    _chk(
        goals[0].target_key == "cup" and goals[0].target_xy is None,
        "Walk left UNRESOLVED (target_key=cup, target_xy=None) for the runtime stand pick",
    )

    # Auto-insert: a bare [Grab] with no preceding [Walk] gets a navigate prepended.
    g2 = compile_program("[Grab] <cup> (1)", ss)
    _chk(
        [g.prim for g in g2] == [Primitive.NAVIGATE_TO, Primitive.PICK],
        "bare Grab auto-inserts navigate",
    )

    # Door verbs: [OpenDoor]/[CloseDoor] parse to the door primitives and auto-insert ONE navigate to
    # the door (the second op is at the same key -> _nav dedups), mirroring Open/Close.
    ss.register("door1", Role.OPENABLE, xy=(1.0, 0.0), z=1.0, approach_yaw=0.0)
    gd = compile_program("[OpenDoor] <door> (1), [CloseDoor] <door> (1)", ss)
    _chk(
        [g.prim for g in gd]
        == [Primitive.NAVIGATE_TO, Primitive.OPENDOOR, Primitive.CLOSEDOOR],
        f"door program -> {[g.prim.value for g in gd]}",
    )

    # PutInside is a DISTINCT action: [PutInside] <obj> <container> [<space>] -> navigate + a PLACE_INSIDE
    # goal carrying meta['surface'] + meta['space'] (the space NAME; None when omitted -> auto-distribute).
    ss.register("bin", Role.CONTAINER, xy=(2.0, 0.0), z=0.2, approach_yaw=0.0)
    gi = compile_program("[PutInside] <cup> (1) <bin> (1)", ss)
    _chk(
        [g.prim for g in gi] == [Primitive.NAVIGATE_TO, Primitive.PLACE_INSIDE],
        f"PutInside -> {[g.prim.value for g in gi]}",
    )
    _chk(
        gi[-1].meta.get("surface") == "bin" and gi[-1].meta.get("space") is None,
        "PutInside(no space) -> PLACE_INSIDE surface=bin space=None",
    )
    gil = compile_program("[PutInside] <cup> (1) <bin> (1) <bin_space2> (1)", ss)
    _chk(
        gil[-1].prim is Primitive.PLACE_INSIDE
        and gil[-1].meta.get("space") == "bin_space2",
        "PutInside(<bin_space2>) -> PLACE_INSIDE space=bin_space2",
    )
    gp = compile_program("[Put] <cup> (1) <deskB> (1)", ss)
    _chk(
        gp[-1].prim is Primitive.PLACE and gp[-1].meta.get("into") is None,
        "plain Put stays PLACE with no into flag",
    )

    # Attach is a DISTINCT action: [Attach] <A> <A1> <B> <B1> -> navigate + an ATTACH goal carrying
    # explicit child/parent connector pairs (A1 name/idx + B1 name/idx).
    ss.register("panel", Role.GRASPABLE, xy=(3.0, 0.0), z=0.5, approach_yaw=0.0)
    ss.register("hub", Role.SURFACE, xy=(3.0, 1.0), z=0.5, approach_yaw=180.0)
    ga = compile_program("[Attach] <panel>(1) <panelM>(0) <hub>(1) <hubF>(0)", ss)
    _chk(
        [g.prim for g in ga] == [Primitive.NAVIGATE_TO, Primitive.ATTACH],
        f"Attach -> {[g.prim.value for g in ga]}",
    )
    _chk(
        ga[-1].meta.get("parent") == "hub"
        and ga[-1].meta.get("child_point") == "panelm"
        and ga[-1].meta.get("child_point_idx") == 0
        and ga[-1].meta.get("parent_point") == "hubf"
        and ga[-1].meta.get("parent_point_idx") == 0,
        "Attach 4-arg -> ATTACH parent=hub child_point=panelm(0) parent_point=hubf(0)",
    )
    gap = compile_program("[Attach] <panel>(1) <panelM>(0) <hub>(1) <slot>(2)", ss)
    _chk(
        gap[-1].prim is Primitive.ATTACH
        and gap[-1].meta.get("parent_point") == "slot"
        and gap[-1].meta.get("parent_point_idx") == 2,
        "Attach 4-arg with (slot)(2) -> parent_point=slot parent_point_idx=2",
    )

    _selftest_locomotion(ss, _chk)
    _selftest_description(_chk)

    # Errors: unsupported verb + wrong arity.
    for bad, why in (
        ("[Fly] <cup> (1)", "unsupported verb"),
        ("[Put] <cup> (1)", "wrong arity"),
    ):
        try:
            compile_program(bad, ss)
            _chk(False, f"{why} should raise")
        except CommandParseError:
            _chk(True, f"{why} raises CommandParseError")

    print(f"[command_grammar] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())

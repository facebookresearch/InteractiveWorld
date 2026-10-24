# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compile blind-guidance ego legs from the LIVE nav grid instead of authored dead reckoning.

WHY THIS EXISTS. The "guide a person who is shown no image" family is scored on whether an
assistant can turn what it sees into instructions a body can execute without sight. Those
instructions are the four egocentric verbs -- ``[TurnLeft] / [TurnRight] / [TurnAround] /
[Forward] <steps>(n)`` -- and they are the only verbs in the grammar that do not name a
destination, so they are the only ones that can be given to someone who cannot see the room.

An AUTHOR cannot write them. Legs dead-reckoned against an imagined layout ("turn left, three
steps") disagree with the built scene: a human who starts flush against a container can walk 0 of 5
authored ``[Forward]`` commands, blocked at 0.00 m clear. Every one of those legs fails as
``path_blocked`` with zero frames -- and the row would still score 1.00, because the manipulation
verbs auto-insert a ``[Walk]`` that carries the body there anyway. The guidance would be decorative.

So the legs are compiled HERE, from the same nav grid the walker and the ``[Forward]`` clearance
oracle consume, at render time, once the scene actually exists:

  authored:  [TurnRight] <deg>(60), [Forward] <steps>(2), [LookAt] <obj_1>, [Grab] <obj_1>
  compiled:  [TurnLeft] <deg>(30), [Forward] <steps>(2), [Forward] <steps>(1), [LookAt] <obj_1>, ...

The rewrite is confined to the ego legs. Manipulation verbs, their arguments, their order and the
answer slots are untouched, so a compiled program scores exactly what the authored one scored.

CONTRACT the compiler respects, all of it measured rather than assumed:

* one pace is ``goal_schema.FORWARD_STEP_M`` (0.6 m) -- the engine constant, not the author's 0.45;
* a ``[Forward]`` that does not fit fails WHOLE, zero frames (``produce_forward``), so every leg is
  measured with ``nav_planner.free_run_ahead`` -- the same oracle -- before it is emitted;
* the published bearing convention is clock positions, i.e. 30-degree hours clamped to the 9/3
  endpoints with 6 o'clock as ``[TurnAround]``, so a compiled turn is quantised to it and the
  residual heading error is absorbed by re-planning after every run;
* a run is capped at ``_CHUNK`` paces, which is what BOUNDS that residual: 15 degrees of worst-case
  quantisation error over 1.2 m is 0.31 m of drift, and the next run re-aims from the true pose.

The deliberate MIS-ROUTE control (``BENCH_EGO_MISROUTE``) mirrors a compiled turn on purpose. It is
the wrong-guidance arm: the body walks off in the wrong direction, the legs stop fitting, and --
because ``semantic_executor``'s ego lock refuses to teleport a body the ego channel has taken over
-- the row FAILS instead of being rescued by an auto-``[Walk]``. That failure is the proof the
guidance is load-bearing.
"""

from __future__ import annotations

import math
import os
import re

from goal_schema import FORWARD_STEP_M, PrimitiveError, PrimitiveFailure

# How far a NAMED navigate may still move a body the ego channel has taken over. It is not slack for
# a shortcut -- it is the final square-up onto a stand that the inflated grid marks as blocked (a
# stand legitimately hugs the counter it works at), which no [Forward] can enter. Anything past this
# is a traverse the guidance failed to perform, and semantic_executor fails it loudly.
EGO_ARRIVE_M: float = 1.25
# What the COMPILER aims for, deliberately tighter than the lock. The runtime stand picker ranks
# candidates nearest the body's LIVE position, so it can legitimately land on a stand the compiler did
# not target; the gap between these two numbers is the headroom for that disagreement, and nothing else.
_COMPILE_ARRIVE_M: float = 0.75

_HOUR_DEG: float = 30.0  # one clock hour
_QUARTER_DEG: float = (
    90.0  # the 9 / 3 o'clock endpoints == a bare [TurnLeft] / [TurnRight]
)
_TURNAROUND_DEG: float = 150.0  # at or past this, 6 o'clock -> [TurnAround]
_CHUNK: int = (
    2  # paces per [Forward]; bounds the quantised-heading drift (see module docstring)
)
_MAX_LEGS: int = (
    40  # a route needing more runs than this is not expressible as guidance
)
# How far off an islanded stand a reachable stand-in may be. Deliberately the SAME budget the ego lock
# allows a navigate to close: aiming further is the compiler quietly handing the walk back to the
# harness. A 2.5 m budget can aim the guidance metres short of its own target and then fail the pick
# on seat error -- a silent shortfall dressed up as a compiled route.
_TARGET_SNAP_R: float = EGO_ARRIVE_M
_CLEAR_MARGIN: float = (
    0.05  # keep the emitted leg strictly inside the measured clearance
)
_MIN_PROGRESS_M: float = (
    0.05  # a run must shorten the remaining route by at least this, or it is a stall
)

_TOK_RE = re.compile(r"\[[A-Za-z_]+\](?:\s*<[^>]*>\s*\(\s*\d+\s*\))*")
_EGO_RE = re.compile(r"^\[(Forward|TurnLeft|TurnRight|TurnAround)\]", re.IGNORECASE)
_OBSERVE_RE = re.compile(r"^\[(LookAround|LookScan)\]", re.IGNORECASE)

# Which argument of a verb names the entity the body has to STAND at. Mirrors command_grammar's
# auto-navigate exactly (and [LookAt], whose producer squares the body onto the target's stand), so
# the compiler routes to the same place the executor will.
_STAND_ARG: dict[str, int] = {
    "walk": 0,
    "run": 0,
    "walktowards": 0,
    "grab": 0,
    "open": 0,
    "close": 0,
    "opendoor": 0,
    "closedoor": 0,
    "sit": 0,
    "cook": 0,
    "raise": 0,
    "lookat": 0,
    "put": 1,
    "putback": 1,
    "putinside": 1,
    "attach": 2,
}


def tokens(program: str) -> list[str]:
    """Split a program string into whole ``[Verb] <a>(1) <b>(1)`` commands."""
    return [" ".join(t.split()) for t in _TOK_RE.findall(str(program))]


def is_ego(tok: str) -> bool:
    return _EGO_RE.match(tok) is not None


def has_ego_channel(program: str) -> bool:
    """True iff the program steers the body egocentrically anywhere -- the whole gate on this file."""
    return any(is_ego(t) for t in tokens(program))


def _verb(tok: str) -> str:
    return tok.split("]", 1)[0].lstrip("[").lower()


def _args(tok: str) -> list[str]:
    return re.findall(r"<([^>]*)>", tok)


def stand_key(tok: str) -> str | None:
    """The entity this command makes the body stand at, or None if it moves nobody."""
    idx = _STAND_ARG.get(_verb(tok))
    if idx is None:
        return None
    args = _args(tok)
    if idx >= len(args):
        return None
    # An UNFILLED answer slot names nothing. It cannot occur in a built program (fill_answer_slots
    # resolves every one and prepends the [Walk] this compiler then routes), only in a raw spec.
    return args[idx] if args[idx] != "?" else None


def _yaw_for_dir(dx: float, dy: float) -> float:
    """Body yaw that faces (dx, dy). The exact inverse of ``nav_planner.forward_xy``."""
    return math.degrees(math.atan2(float(dx), -float(dy)))


def _ang_delta(a: float, b: float) -> float:
    """Signed a -> b turn in (-180, 180]."""
    return (float(b) - float(a) + 180.0) % 360.0 - 180.0


def quantise_turn(dyaw: float) -> float:
    """A needed turn, snapped to the published clock convention. 0.0 means "no turn verb"."""
    mag, sign = abs(dyaw), (1.0 if dyaw >= 0.0 else -1.0)
    if mag >= _TURNAROUND_DEG:
        return 180.0
    if mag >= _QUARTER_DEG:
        return sign * _QUARTER_DEG
    return sign * round(mag / _HOUR_DEG) * _HOUR_DEG


def turn_tokens(qyaw: float) -> list[str]:
    """The grammar spelling of a quantised turn. ``[TurnAround]`` is arity-0; a partial turn MUST use
    ``[TurnLeft]`` or ``[TurnRight]`` with an explicit degree scalar.

    A bearing past the convention's 9/3 endpoints is spelled as the endpoint plus the remainder --
    "turn left, then a bit more left" -- which is a thing a guide can say and a single verb is not."""
    if abs(qyaw) < 1e-6:
        return []
    if abs(qyaw) >= 180.0 - 1e-6:
        return ["[TurnAround]"]
    verb = "[TurnLeft]" if qyaw > 0.0 else "[TurnRight]"
    mag = abs(qyaw)
    if abs(mag - _QUARTER_DEG) < 1e-6:
        return [verb]
    if mag < _QUARTER_DEG:
        return [f"{verb} <deg> ({int(round(mag))})"]
    return [verb, f"{verb} <deg> ({int(round(mag - _QUARTER_DEG))})"]


def forward_tokens(steps: int) -> list[str]:
    """A run of paces as ``[Forward] <steps>(n)`` commands, at most ``_CHUNK`` paces each."""
    full, rest = divmod(int(steps), _CHUNK)
    out = [f"[Forward] <steps> ({_CHUNK})"] * full
    if rest:
        out.append(
            "[Forward] <steps> (1)" if rest == 1 else f"[Forward] <steps> ({rest})"
        )
    return out


def compile_route(
    grid, start_xy, start_yaw: float, goal_xy, arrive_m: float = _COMPILE_ARRIVE_M
) -> tuple[list[str], tuple[float, float], float]:
    """Route on the nav grid -> ego legs that genuinely execute. Returns (tokens, end_xy, end_yaw).

    Greedy and re-planning: aim at the next string-pulled waypoint, quantise the turn to the clock
    convention, then walk only as far as ``free_run_ahead`` -- ``produce_forward``'s own oracle --
    says is clear on that quantised heading, and re-plan from the pose that produces. Raises
    ``PrimitiveFailure(PATH_BLOCKED)`` when there is no route at all (from ``plan_path``) or when the
    convention cannot express one; both are real, nameable defects, not silent truncations."""
    import nav_planner

    out: list[str] = []
    cur = (float(start_xy[0]), float(start_xy[1]))
    yaw = float(start_yaw)
    goal = (float(goal_xy[0]), float(goal_xy[1]))
    for _ in range(_MAX_LEGS):
        gap = math.dist(cur, goal)
        if gap <= arrive_m:
            return out, cur, yaw
        path = nav_planner.plan_path(grid, cur, goal)
        qy, steps, cur = _best_bearing(grid, cur, yaw, path)
        if steps <= 0:
            raise PrimitiveFailure(
                PrimitiveError.PATH_BLOCKED,
                f"ego route stalled at ({cur[0]:.2f},{cur[1]:.2f}) yaw={yaw:.0f}, {gap:.2f} m "
                f"short of ({goal[0]:.2f},{goal[1]:.2f}): no clock bearing both clears a pace and "
                f"makes progress along the route",
            )
        out += turn_tokens(qy) + forward_tokens(steps)
        yaw = (yaw + qy + 180.0) % 360.0 - 180.0
    raise PrimitiveFailure(
        PrimitiveError.PATH_BLOCKED,
        f"ego route to ({goal[0]:.2f},{goal[1]:.2f}) needs more than {_MAX_LEGS} runs",
    )


def _best_bearing(grid, cur, yaw: float, path):
    """Pick the next run by MEASURING all twelve clock bearings, not by geometry. Returns
    (turn, paces, landing); paces 0 means every bearing is a dead end.

    Aiming straight at the next waypoint and quantising is what a person would do, and it deadlocks:
    a body delivered into a pocket (a stand that hugs a container) has no clear pace on the bearing
    that points at the goal, and the sideways pace that would free it points AWAY from the goal, so a
    progress-toward-the-goal test rejects it. Scoring on REMAINING ROUTE LENGTH instead accepts the
    sidestep -- it is progress along the path even though it is not progress toward the point -- which
    is exactly the difference between guiding someone out of a corner and telling them they are
    stuck."""
    import nav_planner

    rem0 = _path_remaining(path, cur)
    best = (0.0, 0, cur, rem0 - _MIN_PROGRESS_M)
    for hours in range(-6, 6):
        qy = hours * _HOUR_DEG
        bear = (yaw + qy + 180.0) % 360.0 - 180.0
        clear = nav_planner.free_run_ahead(
            grid, cur, bear, _CHUNK * FORWARD_STEP_M + _CLEAR_MARGIN
        )
        steps = int(max(0.0, clear - _CLEAR_MARGIN) / FORWARD_STEP_M + 1e-9)
        if steps <= 0:
            continue
        fx, fy = nav_planner.forward_xy(bear)
        for n in range(min(steps, _CHUNK), 0, -1):
            land = (cur[0] + fx * n * FORWARD_STEP_M, cur[1] + fy * n * FORWARD_STEP_M)
            rem = _path_remaining(path, land)
            if rem < best[3]:
                best = (qy, n, land, rem)
    return best[0], best[1], best[2]


def _path_remaining(path, p) -> float:
    """Route length still to walk from ``p``: the shortest (off-path hop + remainder of the polyline).

    The off-path hop is included so a landing point that is closer to the goal in a straight line but
    on the wrong side of a wall does not read as progress -- the polyline is the only route there is."""
    tail = [0.0] * len(path)
    for i in range(len(path) - 2, -1, -1):
        tail[i] = tail[i + 1] + math.dist(path[i], path[i + 1])
    best = math.dist(p, path[-1]) + 0.0 if len(path) == 1 else float("inf")
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg = (b[0] - a[0], b[1] - a[1])
        ln2 = seg[0] * seg[0] + seg[1] * seg[1]
        t = (
            0.0
            if ln2 < 1e-12
            else max(
                0.0, min(1.0, ((p[0] - a[0]) * seg[0] + (p[1] - a[1]) * seg[1]) / ln2)
            )
        )
        q = (a[0] + seg[0] * t, a[1] + seg[1] * t)
        best = min(best, math.dist(p, q) + (1.0 - t) * math.dist(a, b) + tail[i + 1])
    return best


def _stand_xy(scene, key: str) -> tuple[tuple[float, float], float] | None:
    """The declared stand pose for an entity, or None when the scene cannot name one."""
    try:
        stand, yaw = scene.stand_pose_for(key)
    except Exception as exc:  # noqa: BLE001 -- any resolution failure means "cannot route here"
        print(f"[ego] no stand pose for <{key}> ({exc}); leaving its legs authored")
        return None
    return (float(stand[0]), float(stand[1])), float(yaw)


def _reachable_cells(grid, cur) -> set:
    """Every grid cell a body at ``cur`` can actually walk to (4-connected flood).

    A subset of what ``plan_path``'s 8-connected A* can reach, so a cell in here is guaranteed
    routable -- which is the point. One flood answers "is this target reachable" for every candidate
    at once, instead of paying an A* per candidate."""
    from collections import deque

    si, sj = grid.world_to_cell(*cur)
    if not grid.is_free(si, sj):
        snap = grid.nearest_free(si, sj)
        if snap is None:
            return set()
        si, sj = snap
    seen = {(si, sj)}
    q = deque([(si, sj)])
    while q:
        ci, cj = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (ci + di, cj + dj)
            if nxt not in seen and grid.is_free(*nxt):
                seen.add(nxt)
                q.append(nxt)
    return seen


def _entity_contact_xy(scene, key: str):
    """The point on the entity a stand has to be within arm's reach of, in XY. The nav grid is built
    so a free cell near EVERY such contact is reachable from the start, which is what makes it the
    dependable anchor when the DECLARED stand turns out to be islanded."""
    try:
        import init_validation as iv

        c = iv._entity_contact(scene, key)
    except Exception:  # noqa: BLE001 -- an unregistered / contactless entity simply has none
        return None
    return None if c is None else (float(c[0]), float(c[1]))


def _route_target(scene, grid, key: str, cur, stand_fn=None):
    """Where the legs should actually deliver the body for ``key``: the stand the RUNTIME will pick if
    it can be predicted, else the declared stand, else the closest reachable cell to the entity.

    Predicting matters because the program is open loop. The runtime picker re-picks nearest-reachable
    from the body's LIVE position, so guiding to some other point leaves a residual the next run then
    compounds; asking the SAME picker with the SAME live position is a fixed point instead. And the
    declared ``stand_pose_for`` is only a geometric ideal -- a declared stand can have no route at
    all from the start. Never returns an unvalidated point."""
    reach = _reachable_cells(grid, cur)
    if not reach:
        return None
    anchors = []
    if stand_fn is not None:
        picked = stand_fn(key, cur)
        if picked is not None:
            anchors.append((float(picked[0][0]), float(picked[0][1])))
    anchors += [
        a
        for a in ((_stand_xy(scene, key) or (None,))[0], _entity_contact_xy(scene, key))
        if a
    ]
    for base in anchors:
        if grid.world_to_cell(*base) in reach:
            return base
    for base in anchors:
        near = min(reach, key=lambda c: math.dist(grid.cell_center(*c), base))
        pt = grid.cell_center(*near)
        if math.dist(pt, base) <= _TARGET_SNAP_R:
            print(
                f"[ego] <{key}>: point ({base[0]:.2f},{base[1]:.2f}) has no route; "
                f"aiming at the nearest reachable cell ({pt[0]:.2f},{pt[1]:.2f})"
            )
            return pt
    return None


def _misroute(toks: list[str]) -> list[str]:
    """``BENCH_EGO_MISROUTE``: the WRONG-GUIDANCE arm. ``first`` (default) mirrors the first compiled
    turn -- the classic swapped left/right -- and ``all`` mirrors every one. Nothing else moves, so a
    failure is attributable to the route and to nothing else."""
    mode = (os.environ.get("BENCH_EGO_MISROUTE") or "").strip().lower()
    if mode in ("", "0", "off", "false"):
        return toks
    flip = {"[TurnLeft]": "[TurnRight]", "[TurnRight]": "[TurnLeft]"}
    out, done = [], False
    for t in toks:
        head, _, rest = t.partition(" ")
        if head in flip and not (done and mode != "all"):
            out.append((flip[head] + (" " + rest if rest else "")).strip())
            done = True
            continue
        out.append(t)
    print(
        f"[ego] BENCH_EGO_MISROUTE={mode}: turns mirrored (wrong-guidance control ON)"
    )
    return out


def recompile(
    program: str, scene, start_xy, start_yaw: float = 0.0, stand_fn=None
) -> str:
    """Replace every authored ego run with one compiled from the scene's live nav grid.

    Returns ``program`` UNCHANGED for any program with no ego verb (every non-guidance family) and for a
    scene with no nav grid (nothing to measure against), so this is inert everywhere it is not the
    point. Each run is routed to the stand of the NEXT command that stands somewhere -- the same
    place ``command_grammar``'s auto-navigate would have gone -- which is exactly what makes the
    legs, and not the auto-navigate, do the travelling."""
    toks = tokens(program)
    if not any(is_ego(t) for t in toks):
        return program
    # KNOWN ROOT CAUSE, NOT YET FIXED -- read before "fixing" a door-related `path_blocked`.
    # This is the CLOSED-doors grid, while ``init_validation.pick_stand_runtime`` resolves stands
    # against ``nav_grid.passable_view`` (doors OPEN) -- see its ``_flood``. The compiler and the
    # runtime picker therefore disagree about what is reachable whenever a closed door lies between the
    # body and a target, and the disagreement costs the row every command behind that door:
    #
    #   compile : target behind a closed door is unreachable -> "no routable stand: kept as authored"
    #   runtime : the SAME target gets a stand (doors-open view) -> a walk is attempted
    #   runtime : that walk needs `_runtime_open_doors`, whose door-approach navigate is refused by
    #             `semantic_executor._assert_ego_unlocked` -- a door is never a key in `ego_delivery`,
    #             so the lock measures the body against the door stand itself and calls it a carry
    #   result  : `path_blocked`
    #
    # The lock is RIGHT to refuse; the defect is that guidance was planned in a different world from
    # the one it runs in. Fixing it means compiling on the passable view AND making each crossed door
    # an explicit guided step, so the person is told to open it -- NOT exempting doors from the lock,
    # which would restore the silent-rescue hole the lock exists to close.
    grid = getattr(scene, "nav_grid", None)
    if grid is None:
        print(
            "[ego] no nav grid on this scene: ego legs left as authored (unmeasurable)"
        )
        return program
    out: list[str] = []
    pose = ((float(start_xy[0]), float(start_xy[1])), float(start_yaw))
    # Where the guidance MEANT to leave the body for each target. This, not the stand, is what the ego
    # lock measures against at runtime: whether the legs did their job is a question about the route
    # they were given, and a stand that no whole pace can enter is geometry's fault, not guidance's.
    delivery: dict[str, tuple[float, float]] = {}
    scene.ego_delivery = delivery
    i, n_auth, n_comp = 0, 0, 0
    while i < len(toks):
        if not is_ego(toks[i]):
            out.append(toks[i])
            key = stand_key(toks[i])
            # The runtime navigate this verb triggers ends the body at ITS stand, not at the end of
            # the legs, so the model of "where we are" has to follow it or every later run is aimed
            # from a place the body is not. That stand is often a cell the inflated grid calls
            # BLOCKED (it hugs the counter) -- which is fine to stand on and fine to step off, so it
            # is the honest start for the next run even though no route may end there.
            pose = _landing_pose(scene, grid, key, pose, stand_fn) if key else pose
            i += 1
            continue
        j, obs = i, []
        while j < len(toks) and (is_ego(toks[j]) or _OBSERVE_RE.match(toks[j])):
            if _OBSERVE_RE.match(toks[j]):
                obs.append(toks[j])
            j += 1
        n_auth += sum(1 for t in toks[i:j] if is_ego(t))
        legs, pose = _run_legs(toks, i, j, scene, grid, pose, stand_fn, delivery)
        n_comp += len(legs)
        out += _interleave(legs, obs)
        i = j
    out = _misroute(out)
    print(
        f"[ego] legs recompiled from the nav grid: {n_auth} authored -> {n_comp} measured"
    )
    return ", ".join(out)


def _landing_pose(scene, grid, key: str, pose, stand_fn):
    """Where the body will be standing once ``key``'s own navigate has run. Distinct from the leg
    DESTINATION: the legs must end somewhere routable, the body ends on the picked stand."""
    if stand_fn is not None:
        picked = stand_fn(key, pose[0])
        if picked is not None:
            return (float(picked[0][0]), float(picked[0][1])), float(picked[1])
    declared = _stand_xy(scene, key)
    if declared is not None:
        return declared
    target = _route_target(scene, grid, key, pose[0], stand_fn)
    return (target, pose[1]) if target else pose


def _interleave(legs: list[str], obs: list[str]) -> list[str]:
    """Put the run's own ``[LookAround]`` scans back, spread evenly through the compiled legs. The
    authored positions cannot survive (the legs are different legs), but the count and the
    scan-as-you-go character do, and an observation verb moves nobody so nothing else shifts."""
    if not obs:
        return legs
    out, k = [], 0
    for idx, leg in enumerate(legs):
        out.append(leg)
        while k < len(obs) and (k + 1) * len(legs) <= (idx + 1) * len(obs):
            out.append(obs[k])
            k += 1
    return out + obs[k:]


def _run_legs(toks, i: int, j: int, scene, grid, pose, stand_fn=None, delivery=None):
    """Compile ONE authored run of ego tokens into measured legs.

    Keeps the AUTHORED run when the geometry cannot be guided (no target, or no route the clock
    convention can walk) and says so loudly. It does not raise: an uncompilable run is one row's
    honest failure -- the ego lock then refuses the rescue walk and the row scores 0 -- not a reason
    to kill an episode that has already paid for its scene."""
    key = next((stand_key(t) for t in toks[j:] if stand_key(t)), None)
    goal = _route_target(scene, grid, key, pose[0], stand_fn) if key else None
    if goal is None:
        print(
            f"[ego] run at token {i} leads to <{key}>, which has no routable stand: kept as authored"
        )
        return [t for t in toks[i:j] if is_ego(t)], pose
    try:
        legs, end_xy, end_yaw = compile_route(grid, pose[0], pose[1], goal)
    except PrimitiveFailure as exc:
        print(f"[ego] cannot guide a body to <{key}>: {exc}; kept as authored")
        return [t for t in toks[i:j] if is_ego(t)], pose
    if delivery is not None:
        delivery[key] = end_xy
    print(
        f"[ego] -> <{key}> at ({goal[0]:.2f},{goal[1]:.2f}): {len(legs)} legs, "
        f"{sum(_pace_count(t) for t in legs) * FORWARD_STEP_M:.2f} m walked, "
        f"delivering to ({end_xy[0]:.2f},{end_xy[1]:.2f}), residual {math.dist(end_xy, goal):.2f} m"
    )
    return legs, (end_xy, end_yaw)


def _pace_count(tok: str) -> int:
    m = re.match(r"^\[Forward\]\s*<[^>]*>\s*\(\s*(\d+)\s*\)", tok, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 1 if re.match(r"^\[Forward\]\s*$", tok, re.IGNORECASE) else 0


# =====================================================================================================
# Offline selftest: a synthetic grid, no Isaac. Proves the compiler emits legs that its own oracle
# says fit, stops inside the arrival tolerance, and refuses to invent a route through a wall.
# =====================================================================================================
class _FakeGrid:
    """A metre grid with an axis-aligned blocked rectangle -- enough to exercise plan_path + clearance."""

    def __init__(self, cell=0.1, blocks=((1.0, -0.5, 1.4, 2.0),), extent=6.0):
        self.cell = cell
        self.inflate = 0.0
        self.blocks = blocks
        self.n = int(2 * extent / cell)
        self.o = -extent
        self.sdf = _ZeroSdf()
        self.door_cost = None

    def world_to_cell(self, x, y):
        return int((x - self.o) / self.cell), int((y - self.o) / self.cell)

    def cell_center(self, i, j):
        return (self.o + (i + 0.5) * self.cell, self.o + (j + 0.5) * self.cell)

    def is_free(self, i, j):
        if not (0 <= i < self.n and 0 <= j < self.n):
            return False
        x, y = self.cell_center(i, j)
        return not any(
            x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in self.blocks
        )

    def nearest_free(self, i, j):
        """Truly nearest by distance, like a real NavGrid -- and, like a real NavGrid, with no idea
        which side of a wall it lands on. That indifference is the defect ``plan_path`` retries around."""
        best = None
        for r in range(1, 40):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r or not self.is_free(i + di, j + dj):
                        continue
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, (i + di, j + dj))
            if best is not None:
                return best[1]
        return None


class _ZeroSdf:
    def __getitem__(self, idx):
        return 1.0


def _selftest() -> int:
    import nav_planner

    fails = []

    def chk(cond, what):
        print(f"  {'PASS' if cond else 'FAIL'}  {what}")
        if not cond:
            fails.append(what)

    chk(quantise_turn(14.0) == 0.0, "a sub-half-hour turn issues no verb")
    chk(quantise_turn(-44.0) == -30.0, "44 deg right -> 1 o'clock hour")
    chk(quantise_turn(100.0) == 90.0, "100 deg left -> the 9 o'clock endpoint")
    chk(quantise_turn(-170.0) == 180.0, "170 deg -> the 6 o'clock extension")
    chk(turn_tokens(90.0) == ["[TurnLeft]"], "a quarter turn is the BARE verb")
    chk(
        turn_tokens(-30.0) == ["[TurnRight] <deg> (30)"],
        "a partial turn uses the bracket form",
    )
    chk(turn_tokens(180.0) == ["[TurnAround]"], "[TurnAround] takes no argument")
    chk(
        forward_tokens(5)
        == ["[Forward] <steps> (2)", "[Forward] <steps> (2)", "[Forward] <steps> (1)"],
        "5 paces -> chunks of at most 2",
    )
    chk(
        stand_key("[PutInside] <obj_1> (1) <unit_c> (1)") == "unit_c",
        "[PutInside] stands at the container",
    )
    chk(stand_key("[TurnLeft]") is None, "a turn stands nowhere")
    chk(
        has_ego_channel("[Grab] <a> (1)") is False,
        "a program with no ego verb declares no channel",
    )

    grid = _FakeGrid()
    legs, end, _yaw = compile_route(grid, (0.0, 0.0), 0.0, (3.0, 0.0))
    chk(bool(legs), "a route around the block compiles to legs")
    chk(
        math.dist(end, (3.0, 0.0)) <= EGO_ARRIVE_M,
        f"the route arrives ({math.dist(end, (3.0, 0.0)):.2f} m)",
    )

    # Every emitted [Forward] must fit, replayed through produce_forward's own oracle.
    xy, yw, bad, paces = (0.0, 0.0), 0.0, 0, 0
    for t in legs:
        v = _verb(t)
        if v.startswith("turn"):
            arg = _args(t)
            deg = (
                float(re.findall(r"\((\d+)\)", t)[0])
                if arg
                else (180.0 if v == "turnaround" else 90.0)
            )
            yw = (
                yw
                + (deg if v == "turnleft" else -deg if v == "turnright" else 180.0)
                + 180.0
            ) % 360.0 - 180.0
            continue
        d = _pace_count(t) * FORWARD_STEP_M
        if nav_planner.free_run_ahead(grid, xy, yw, d) < d - 1e-3:
            bad += 1
        fx, fy = nav_planner.forward_xy(yw)
        xy = (xy[0] + fx * d, xy[1] + fy * d)
        paces += _pace_count(t)
    chk(bad == 0, f"all {paces} compiled paces clear free_run_ahead ({bad} blocked)")

    chk(
        turn_tokens(120.0) == ["[TurnLeft]", "[TurnLeft] <deg> (30)"],
        "a bearing past the 9 o'clock endpoint composes two verbs",
    )

    # POCKET: the body starts in a slot open only sideways, so the bearing that points at the goal is
    # walled and the pace that frees it points away from it: the start-pocket deadlock.
    pocket = _FakeGrid(
        blocks=(
            (-0.5, -0.6, 0.5, -0.4),
            (-0.5, 0.4, 0.5, 0.6),
            (-0.5, -0.6, -0.3, 0.6),
        ),
        extent=8.0,
    )
    plegs, pend, _ = compile_route(pocket, (0.0, 0.0), 180.0, (4.0, 3.0))
    chk(
        math.dist(pend, (4.0, 3.0)) <= EGO_ARRIVE_M,
        f"a body in a pocket is guided out ({len(plegs)} legs)",
    )

    sealed = _FakeGrid(
        blocks=(
            (2.5, 2.5, 3.5, 2.7),
            (2.5, 3.3, 3.5, 3.5),
            (2.5, 2.5, 2.7, 3.5),
            (3.3, 2.5, 3.5, 3.5),
        )
    )
    try:
        compile_route(
            sealed, (0.0, 0.0), 0.0, (3.0, 3.0)
        )  # walled in on all four sides
        chk(False, "a walled-off goal raises rather than inventing a route")
    except PrimitiveFailure:
        chk(True, "a walled-off goal raises rather than inventing a route")

    # nav_planner: a BLOCKED goal inside a sealed wall, positioned so nearest_free snaps to the FAR
    # (unreachable) side. The first attempt fails; the one-stride retry finds the near-side cell.
    walled = _FakeGrid(blocks=((0.90, -8.0, 1.20, 8.0),))
    chk(
        walled.cell_center(*walled.nearest_free(*walled.world_to_cell(1.18, 0.0)))[0]
        > 1.2,
        "nearest_free does snap the test goal to the FAR side (the defect is reproduced)",
    )
    p = nav_planner.plan_path(walled, (0.0, 0.0), (1.18, 0.0))
    chk(
        p[-1] == (1.18, 0.0),
        "a blocked goal snapped to the wrong side of a wall still routes",
    )
    chk(p[-2][0] < 0.9, "and it routes to the NEAR side, not through the wall")
    try:
        nav_planner.plan_path(walled, (0.0, 0.0), (3.0, 0.0))
        chk(False, "a goal a stride past the wall is still honestly unreachable")
    except PrimitiveFailure:
        chk(True, "a goal a stride past the wall is still honestly unreachable")

    _selftest_route_gate(chk, walled)
    _selftest_ego_lock(chk)
    print(f"[ego_route] selftest: {len(fails)} FAIL")
    return 1 if fails else 0


def _selftest_route_gate(chk, walled) -> None:
    """``init_validation._stand_routes`` -- the oracle behind ``pick_stand_runtime(route_gate=True)``.

    It lives in ``init_validation``, which has no offline harness of its own (its ``--positive-control``
    boots Isaac), while the grid fixture it needs is right here. Three properties, and the third is the
    one that makes the gate safe to enable: it may only ever REJECT, never invent a reason to accept."""
    import init_validation as _iv

    chk(
        _iv._stand_routes(walled, (0.0, 0.0), 1.18, 0.0),
        "route gate ACCEPTS a stand the router can reach (blocked cell, near side of the wall)",
    )
    chk(
        not _iv._stand_routes(walled, (0.0, 0.0), 3.0, 0.0),
        "route gate REJECTS a stand a stride past the wall",
    )

    class _Exploding:
        """Anything that is not PATH_BLOCKED means the route is UNMEASURABLE, not absent."""

        def world_to_cell(self, *_a):
            raise RuntimeError("grid unavailable")

    chk(
        _iv._stand_routes(_Exploding(), (0.0, 0.0), 1.0, 1.0),
        "an unmeasurable route fails OPEN: the gate can only remove stands that were already failing",
    )


def _selftest_ego_lock(chk) -> None:
    """The ego lock: a body the ego channel owns cannot be carried by a named navigate."""
    import semantic_executor as se
    from goal_schema import Goal, Primitive

    class _Sc:
        ego_delivery: dict = {}

    sc = _Sc()
    chk(se._EGO_SQ_M == EGO_ARRIVE_M, "the square-up budget equals the navigate budget")
    st = se.SchedState(root_xy=(0.0, 0.0), root_yaw=0.0, stand_z=0.0, floor_z=0.0)
    far = Goal(Primitive.NAVIGATE_TO, target_xy=(5.0, 0.0), target_key="unit_a")
    near = Goal(Primitive.NAVIGATE_TO, target_xy=(0.5, 0.0), target_key="unit_a")
    chk(
        st.ego_locked is False,
        "a fresh state is UNLOCKED (every non-ego program is unchanged)",
    )
    se._assert_ego_unlocked(sc, st, far, False)
    chk(True, "unlocked: a 5 m navigate is allowed, exactly as before")
    st.ego_locked = True
    try:
        se._assert_ego_unlocked(sc, st, far, False)
        chk(False, "locked: a 5 m navigate FAILS with the target named")
    except PrimitiveFailure as exc:
        chk(
            "unit_a" in str(exc) and "5.00 m short" in str(exc),
            "locked: a 5 m navigate FAILS with the target named",
        )
    se._assert_ego_unlocked(sc, st, near, False)
    chk(True, "locked: the sub-metre square-up onto a stand still runs")
    se._assert_ego_unlocked(sc, st, far, True)
    chk(True, "locked: produce_forward's own straight leg is exempt")
    # The delivery point, not the stand, is the reference: a stand no pace can enter is forgiven, and
    # a body that never left the start is not.
    sc.ego_delivery = {"unit_a": (4.9, 0.0)}
    st.root_xy = (4.8, 0.0)
    se._assert_ego_unlocked(sc, st, far, False)
    chk(
        True,
        "locked: an unenterable stand is forgiven when the legs delivered as planned",
    )
    st.root_xy = (0.0, 0.0)
    try:
        se._assert_ego_unlocked(sc, st, far, False)
        chk(False, "locked: a body that never left the start still FAILS")
    except PrimitiveFailure:
        chk(True, "locked: a body that never left the start still FAILS")


if __name__ == "__main__":
    raise SystemExit(_selftest())

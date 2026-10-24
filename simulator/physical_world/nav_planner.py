"""Grid path planning + planned multi-leg walk + FPV eye push-out (all additive, gated on scene.nav_grid).

- plan_path: 8-connected grid A* over the NavGrid free cells (clearance-weighted cost so paths hug open
  space), then a line-of-sight string-pull to a minimal waypoint polyline. Endpoints are the EXACT
  start/goal (snapped to nearest free only if blocked); raises PrimitiveFailure(PATH_BLOCKED) on no route.
- walk_planned: THE walk producer for every NAVIGATE_TO, with or without a grid. It follows the
  polyline while ``heading_controller`` steers the body heading per frame, so there are no depart /
  corner / arrival turn BEATS -- the body turns while it walks, the way habitat-lab and IsaacLab both
  do it. Writes the SchedState end-state + carry threading.
- push_eye_out: clamp the forward-pushed FPV eye back out of walls along the head->eye axis using the
  grid's two-sided SDF (never behind the head). Grid-based, not raycast (the repo has no raycast wiring).

gait_player / humanoid_motion / semantic_executor are imported lazily inside the functions to avoid
the produce_navigate <-> nav_planner import cycle.

OFFLINE GATE: ``python simulator.physical_world/nav_planner.py`` (no Isaac, ~40 ms). It checks the router itself AND
``init_validation --nav-proxy-control``, which proves that the stand search's cheap connectivity proxy
disagrees with this router in both directions. **Run it after touching ``_GOAL_SNAP_R``, the nav grid
inflation, or ``init_validation._shuffle_tol``** -- those three constants are what set the disagreement
band, and it is a structural gap, not a tuning artefact: the proxy's radius is ``_shuffle_tol`` (0.25 m
at the 0.10 m inflation) while a blocked goal here is snapped up to ``_GOAL_SNAP_R`` (0.60 m) before
routing, so **the proxy is stricter than the router it stands in for over a 0.35 m band, by
construction.**
"""

from __future__ import annotations

import heapq
import math
from dataclasses import replace

import numpy as np
from goal_schema import PrimitiveError, PrimitiveFailure

_NBRS = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
)


# REMOVED: the (cell, heading) pose-lattice search and its arrival-lane ladder.
#
# It planned a straight "approach lane" onto the goal heading so the last leg would already be
# aimed at the target. It never once fired. The detour budget was a RATIO of the direct path
# (1.6x) while the shortest lane was an ABSOLUTE 0.5 m, so honouring the heading needed a direct
# path of at least ~1.03 m from a side-on start; measured on cam0815_G, every rung was rejected on
# every navigate, and lengthening the ladder from 0.8 m to 3.0 m changed the rendered arrival
# pivots by under 4 deg. Arrival heading is now the CONTROLLER's job (see ``heading_controller``),
# where it is a per-frame rate-limited slew that cannot be priced out by a budget.


def signed_yaw_delta(from_yaw: float, to_yaw: float) -> float:
    """Shortest signed rotation from one heading to another, in degrees."""
    return (float(to_yaw) - float(from_yaw) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------------------------
# A* + string-pull
# ---------------------------------------------------------------------------------------------
# How far off a BLOCKED goal the retry snap may land. One stride: the final hop from the reached free
# cell to the goal point crosses the body-inflated band either way (that is what standing at a counter
# is), so it must stay short enough that it is a step-in and not a traverse.
_GOAL_SNAP_R: float = 0.6
_FINAL_APPROACH_MIN_FRAMES = 4
_FINAL_APPROACH_MAX_FRAMES = 6
_FINAL_APPROACH_STEP_M = 0.10
_PATH_OVERRIDE_START_TOL_M = 0.05


def plan_path(
    grid,
    start_xy,
    goal_xy,
    clearance_margin: float = 0.35,
    clearance_weight: float = 8.0,
) -> list[tuple[float, float]]:
    """A* + string-pull from start_xy to goal_xy over grid free cells. Returns world waypoints
    [start_exact, wp1, ..., goal_exact]. Raises PrimitiveFailure(PATH_BLOCKED) if no route exists.

    A BLOCKED goal (a stand hugging the counter it works at) is snapped to the nearest free cell within
    one stride. That snap can land on the wrong side of a wall, and when it does A* reports "no route"
    for a goal that is perfectly reachable -- measured on
    ``T5-HSS-01``, where ``init_validation.pick_stand_runtime`` (whose own gate asks for a
    START-REACHABLE free cell near the stand) returned three stands the router then refused, failing
    ten commands and scoring the row 0.00. So on failure the search is RETRIED against every free cell
    within one stride of the goal, terminating at whichever is reached first. Purely additive: a route
    that exists today is found by attempt one and is byte-identical."""
    si, sj = grid.world_to_cell(*start_xy)
    gi, gj = grid.world_to_cell(*goal_xy)
    if not grid.is_free(si, sj):
        snap = grid.nearest_free(si, sj)
        if snap is None:
            raise PrimitiveFailure(
                PrimitiveError.PATH_BLOCKED, "start cell has no free neighbour"
            )
        si, sj = snap
    blocked_goal = not grid.is_free(gi, gj)
    near: set[tuple[int, int]] = set()
    if blocked_goal:
        near = _free_cells_within(grid, goal_xy, _GOAL_SNAP_R)
        if not near:
            raise PrimitiveFailure(
                PrimitiveError.PATH_BLOCKED, "goal cell has no free neighbour"
            )
        gi, gj = min(
            near,
            key=lambda cell: (
                math.dist(grid.cell_center(*cell), goal_xy),
                cell[0],
                cell[1],
            ),
        )

    args = (grid, clearance_margin, clearance_weight)
    cells = _astar(*args, (si, sj), {(gi, gj)}, (gi, gj))
    if cells is None and blocked_goal:
        cells = _astar(*args, (si, sj), near, (gi, gj))
    if cells is None:
        raise PrimitiveFailure(
            PrimitiveError.PATH_BLOCKED,
            f"no route {tuple(round(v, 2) for v in start_xy)}->{tuple(round(v, 2) for v in goal_xy)}",
        )

    return _world_waypoints(
        grid,
        _string_pull(grid, cells, clearance_margin, clearance_weight),
        start_xy,
        goal_xy,
        blocked_start=not grid.is_free(*grid.world_to_cell(*start_xy)),
        blocked_goal=blocked_goal,
    )


def _world_waypoints(
    grid, cells, start_xy, goal_xy, *, blocked_start: bool, blocked_goal: bool
) -> list[tuple[float, float]]:
    """Convert routed cells while retaining safe endpoints beside blocked stands."""
    exact_start = (float(start_xy[0]), float(start_xy[1]))
    pts = [exact_start]
    if blocked_start:
        pts.append(grid.cell_center(*cells[0]))
    if not blocked_goal:
        for i, j in cells[1:-1]:
            pts.append(grid.cell_center(i, j))
        pts.append((float(goal_xy[0]), float(goal_xy[1])))
        return _dedup(pts, 0.5 * grid.cell)

    for i, j in cells[1:]:
        pts.append(grid.cell_center(i, j))
    pts = _dedup(pts, 0.5 * grid.cell)
    if pts[0] != exact_start:
        pts.insert(0, exact_start)
    safe_goal = tuple(float(v) for v in grid.cell_center(*cells[-1]))
    if math.dist(pts[-1], safe_goal) > 1e-9:
        pts.append(safe_goal)
    pts.append((float(goal_xy[0]), float(goal_xy[1])))
    return pts


def _free_cells_within(grid, xy, radius: float) -> set:
    """Every free cell whose centre is within ``radius`` of ``xy``. The retry goal set."""
    ci, cj = grid.world_to_cell(*xy)
    r = max(1, int(math.ceil(radius / grid.cell)))
    out = set()
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            i, j = ci + di, cj + dj
            if grid.is_free(i, j) and math.dist(grid.cell_center(i, j), xy) <= radius:
                out.add((i, j))
    return out


def _astar(
    grid, clearance_margin: float, clearance_weight: float, start, goals: set, aim
):
    """8-connected clearance-weighted A* to ANY cell in ``goals``, heuristic aimed at ``aim``.
    Returns the cell chain start..reached, or None when no goal cell is reachable."""
    cell = grid.cell
    ai, aj = aim

    def heur(i: int, j: int) -> float:
        return math.hypot(i - ai, j - aj) * cell

    openq = [(heur(*start), 0.0, start)]
    gcost = {start: 0.0}
    came: dict = {}
    hit = None
    while openq:
        _f, gc, (ci, cj) = heapq.heappop(openq)
        if (ci, cj) in goals:
            hit = (ci, cj)
            break
        if gc > gcost.get((ci, cj), 1e18):
            continue
        for di, dj, w in _NBRS:
            ni, nj = ci + di, cj + dj
            if not grid.is_free(ni, nj):
                continue
            if (
                di != 0 and dj != 0
            ):  # no diagonal corner-cutting through a blocked orthogonal cell
                if not grid.is_free(ci + di, cj) or not grid.is_free(ci, cj + dj):
                    continue
            pen = clearance_weight * max(
                0.0, clearance_margin - float(grid.sdf[nj, ni])
            )
            # Extra cost to route through a CLOSED door (prefer an already-open doorway of similar length).
            # door_cost is None for every doorless grid -> exact +0.0 -> A* is byte-identical (IEEE754 x+0=x).
            dcost = (
                float(grid.door_cost[nj, ni])
                if getattr(grid, "door_cost", None) is not None
                else 0.0
            )
            ng = gc + (w + pen + dcost) * cell
            if ng < gcost.get((ni, nj), 1e18):
                gcost[(ni, nj)] = ng
                came[(ni, nj)] = (ci, cj)
                heapq.heappush(openq, (ng + heur(ni, nj), ng, (ni, nj)))
    if hit is None:
        return None
    chain = [hit]
    while chain[-1] != start:
        chain.append(came[chain[-1]])
    chain.reverse()
    return chain


def _cell_step_cost(grid, source, destination, clearance_margin, clearance_weight):
    di = int(destination[0]) - int(source[0])
    dj = int(destination[1]) - int(source[1])
    penalty = clearance_weight * max(
        0.0, clearance_margin - float(grid.sdf[destination[1], destination[0]])
    )
    door_cost = (
        float(grid.door_cost[destination[1], destination[0]])
        if getattr(grid, "door_cost", None) is not None
        else 0.0
    )
    return (math.hypot(di, dj) + penalty + door_cost) * grid.cell


def _segment_cells(a, b) -> list[tuple[int, int]]:
    i0, j0 = a
    i1, j1 = b
    di = i1 - i0
    dj = j1 - j0
    steps = max(1, int(math.ceil(max(abs(di), abs(dj)) * 4)))
    cells = []
    for step in range(steps + 1):
        fraction = step / steps
        cell = (
            int(math.floor(i0 + di * fraction + 0.5)),
            int(math.floor(j0 + dj * fraction + 0.5)),
        )
        if not cells or cell != cells[-1]:
            cells.append(cell)
    return cells


def _path_cost(grid, cells, clearance_margin, clearance_weight) -> float:
    return sum(
        _cell_step_cost(
            grid,
            source,
            destination,
            clearance_margin,
            clearance_weight,
        )
        for source, destination in zip(cells, cells[1:])
    )


def _string_pull(
    grid,
    cells: list,
    clearance_margin: float = 0.35,
    clearance_weight: float = 2.0,
) -> list:
    """Collapse the A* staircase without discarding its obstacle-clearance preference."""
    if len(cells) <= 2:
        return cells
    prefix_cost = [0.0]
    for source, destination in zip(cells, cells[1:]):
        prefix_cost.append(
            prefix_cost[-1]
            + _cell_step_cost(
                grid,
                source,
                destination,
                clearance_margin,
                clearance_weight,
            )
        )
    out = [cells[0]]
    i = 0
    n = len(cells)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            shortcut = _segment_cells(cells[i], cells[j])
            if all(grid.is_free(*cell) for cell in shortcut):
                shortcut_cost = _path_cost(
                    grid, shortcut, clearance_margin, clearance_weight
                )
                if shortcut_cost <= prefix_cost[j] - prefix_cost[i] + 1e-9:
                    break
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _line_of_sight(grid, a, b) -> bool:
    """True iff every cell under the a->b segment is free. Dense-sampled (~4 samples/cell) so a diagonal
    segment cannot slip through a blocked cell corner the way thin Bresenham can (that would let string-pull
    cut a path through a wall)."""
    return all(grid.is_free(*cell) for cell in _segment_cells(a, b))


def _dedup(pts: list, min_step: float) -> list:
    out = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_step:
            out.append(p)
    if len(out) == 1 or (out[-1] != pts[-1]):
        out[-1] = pts[-1]  # always terminate exactly at the goal
    return out


def final_approach_points(
    start_xy: tuple[float, float], goal_xy: tuple[float, float]
) -> list[tuple[float, float]]:
    """Return a short, bounded translation from a free nav point to a blocked stand."""
    distance = math.dist(start_xy, goal_xy)
    if distance <= 1e-9:
        return []
    frames = max(
        _FINAL_APPROACH_MIN_FRAMES,
        min(_FINAL_APPROACH_MAX_FRAMES, math.ceil(distance / _FINAL_APPROACH_STEP_M)),
    )
    return [
        (
            float(start_xy[0]) + (float(goal_xy[0]) - float(start_xy[0])) * i / frames,
            float(start_xy[1]) + (float(goal_xy[1]) - float(start_xy[1])) * i / frames,
        )
        for i in range(1, frames + 1)
    ]


def path_starts_at(
    path, start_xy, tolerance_m: float = _PATH_OVERRIDE_START_TOL_M
) -> bool:
    if not path:
        return False
    return (
        math.dist(
            (float(path[0][0]), float(path[0][1])),
            (float(start_xy[0]), float(start_xy[1])),
        )
        <= tolerance_m
    )


# ---------------------------------------------------------------------------------------------
# Planned multi-leg walk (drop-in for produce_navigate's single straight walk)
# ---------------------------------------------------------------------------------------------
def walk_planned(rig, g, scene, st, goal, grid, *, path_override=None) -> list:
    """Walk from ``st.root_xy`` to ``goal.target_xy``, arriving on ``goal.target_yaw``. THE one walk
    producer: with a nav grid it follows a planned polyline, without one it walks the straight line.

    The body heading is steered CONTINUOUSLY by ``heading_controller`` -- there are no depart, corner
    or arrival turn BEATS any more. Those three beats are what made a 0.40 m reposition cost 141 deg
    of body rotation (turn out, shuffle, turn back) and what let a corner snap 16 deg in one frame;
    neither habitat-lab nor IsaacLab has any such decomposition. See ``heading_controller`` for the
    references and the measured before/after.
    """
    if grid is not None:
        si, sj = grid.world_to_cell(*st.root_xy)
        blocked_start = not grid.is_free(si, sj)
        if path_override is None or not path_starts_at(path_override, st.root_xy):
            try:
                path = plan_path(grid, tuple(st.root_xy), tuple(goal.target_xy))
            except PrimitiveFailure:
                # The target is isolated in the grid. Do NOT fall back to a straight walk (that would
                # drive the human THROUGH walls -- a hard penetration the render blocker flags). FAIL the
                # navigate HONESTLY: the render goal loop records the failed primitive and skips it.
                raise PrimitiveFailure(
                    PrimitiveError.PATH_BLOCKED,
                    f"nav target {tuple(round(float(v), 2) for v in goal.target_xy)} unreachable in grid",
                )
        else:
            path = [
                tuple(float(value) for value in point[:2]) for point in path_override
            ]
            path[0] = tuple(float(value) for value in st.root_xy)
        frames = []
        if blocked_start and len(path) >= 2:
            frames.extend(_initial_departure_frames(rig, scene, st, path[1]))
            path = path[1:]
        gi, gj = grid.world_to_cell(*goal.target_xy)
        if not grid.is_free(gi, gj):
            safe_xy = (
                tuple(float(value) for value in path[-2])
                if len(path) >= 2
                else tuple(float(value) for value in st.root_xy)
            )
            if len(path) >= 2:
                safe_goal = replace(goal, target_xy=safe_xy)
                frames.extend(_walk_frames(g, scene, st, safe_goal, path[:-1]))
            frames.extend(_final_approach_frames(rig, scene, st, goal, safe_xy))
            return frames
        frames.extend(_walk_frames(g, scene, st, goal, path))
        return frames
    else:
        path = [tuple(st.root_xy), tuple(goal.target_xy)]
    return _walk_frames(g, scene, st, goal, path)


def _initial_departure_frames(rig, scene, st, safe_xy) -> list:
    """Back a stable body out of a furniture-adjacent stand before starting the gait."""
    import gait_player as gait
    import humanoid_motion as wpp
    import semantic_executor as se

    held = st.held
    target_arm = (
        held["grasp_arm"]
        if held is not None
        else getattr(rig, "arm_rest", rig.arm_grasp)
    )
    target_pose = se._manip(rig, target_arm, 1.0 if held is not None else 0.0)
    previous_pose = st.last_pose156 if st.last_pose156 is not None else target_pose
    points = final_approach_points(tuple(st.root_xy), tuple(safe_xy))
    obj = se._obj_carry(st)
    frames = []
    for index, xy in enumerate(points):
        fraction = (index + 1) / len(points)
        pose = gait.blend156(previous_pose, target_pose, wpp._smoothstep(fraction))
        frames.append(
            wpp.Frame(
                pose,
                xy,
                float(st.root_yaw),
                se._pin_z(pose, st.floor_z),
                fpv_mode="look",
                obj=obj,
                seg="depart",
            )
        )

    st.root_xy = tuple(safe_xy)
    st.walk_end_yaw = float(st.root_yaw)
    st.last_pose156 = target_pose
    if held is not None:
        se._stamp_obj_path(frames, scene, held["obj_key"])
        se._stamp_carried(frames, held.get("carried"))
    return frames


def _final_approach_frames(rig, scene, st, goal, safe_xy) -> list:
    """Ease the gait into a stable body while translating into the operation stand."""
    import gait_player as gait
    import humanoid_motion as wpp
    import semantic_executor as se

    held = st.held
    target_arm = (
        _navigation_arm(held)
        if held is not None
        else getattr(rig, "arm_rest", rig.arm_grasp)
    )
    obj = se._obj_carry(st)
    target_pose = se._manip(rig, target_arm, 1.0 if held is not None else 0.0)
    previous_pose = st.last_pose156 if st.last_pose156 is not None else target_pose
    target_yaw = float(st.root_yaw if goal.target_yaw is None else goal.target_yaw)
    yaw_delta = signed_yaw_delta(st.root_yaw, target_yaw)
    points = final_approach_points(tuple(safe_xy), tuple(goal.target_xy))
    frames = []
    for index, xy in enumerate(points):
        fraction = (index + 1) / len(points)
        pose = gait.blend156(previous_pose, target_pose, wpp._smoothstep(fraction))
        frames.append(
            wpp.Frame(
                pose,
                xy,
                float(st.root_yaw) + yaw_delta * fraction,
                se._pin_z(pose, st.floor_z),
                fpv_mode="look",
                obj=obj,
                seg="arrive",
            )
        )

    st.root_xy = tuple(goal.target_xy)
    st.root_yaw = target_yaw
    st.walk_end_yaw = target_yaw
    st.last_pose156 = target_pose
    if held is not None:
        se._stamp_obj_path(frames, scene, held["obj_key"])
        se._stamp_carried(frames, held.get("carried"))
    return frames


# Frames over which the STAND pose is eased into the gait cycle at the start of a walk. This is all
# that survives of the old DEPART beat: the beat also turned the body to the travel heading while
# standing still, which the controller now does en route. Skipped entirely when already walking.
_N_WALK_IN = 4
_N_LARGE_CARRY_IN = 12


def _navigation_arm(held):
    if held is None:
        return None
    return held.get("navigation_arm", held["grasp_arm"])


def _walk_frames(g, scene, st, goal, path) -> list:
    """Turn a polyline + an arrival facing into rendered frames, and write the SchedState end-state.

    Every frame advances the gait cycle by the same nominal step, INCLUDING frames where the root is
    held (the controller halts translation while the body is crabbed past its cap). That is the
    ``max(1, ...)`` rule from habitat's ``calculate_walk_pose``: a pivot marks time on the spot. The
    old code instead froze one mid-stride pose and rotated it rigidly, which teleported a foot 703 mm
    in a single frame and swept a leg 141 mm into a wall.
    """
    import gait_player as gait
    import heading_controller as hc
    import humanoid_motion as wpp
    import semantic_executor as se

    arm_ov = _navigation_arm(st.held)
    rhp = st.held["rhp"] if st.held else None
    obj = se._obj_carry(st)

    target_yaw = None if goal.target_yaw is None else float(goal.target_yaw)
    steps = hc.walk_track(
        [(float(p[0]), float(p[1])) for p in path],
        float(st.root_yaw),
        target_yaw,
        wpp.WALK_SPEED,
        wpp.FPS,
        hc.HeadingCfg(),
    )
    if (
        not steps
    ):  # start == goal: no frames, but the state still has to land on the goal pose
        st.root_xy = tuple(goal.target_xy)
        if target_yaw is not None:
            st.root_yaw = target_yaw
            st.walk_end_yaw = target_yaw  # keep the two headings in step
        return []

    ease = 0 if st.last_seg in ("walk", "depart") else min(_N_WALK_IN, len(steps))
    carry_ease = (
        min(_N_LARGE_CARRY_IN, len(steps))
        if st.held is not None
        and "navigation_arm" in st.held
        and st.last_seg not in ("walk", "depart")
        else 0
    )
    carry_from = st.held["grasp_arm"] if carry_ease else arm_ov
    prev_pose = st.last_pose156
    dist = st.gait_cum_dist
    frames: list = []
    for k, s in enumerate(steps):
        dist += s.gait_m
        pose, mfz = gait.sample(g, dist)
        seg = "walk"
        if prev_pose is not None and k < ease:
            pose = gait.blend156(prev_pose, pose, wpp._smoothstep((k + 1) / ease))
            seg = "depart"
        frame_arm = (
            se.core.lerp_arm(
                carry_from,
                arm_ov,
                wpp._smoothstep((k + 1) / carry_ease),
            )
            if carry_ease
            else arm_ov
        )
        frames.append(
            wpp.Frame(
                pose,
                s.xy,
                s.yaw,
                st.floor_z - mfz,
                arm_override=frame_arm,
                rhp=rhp,
                fpv_mode="walk",
                obj=obj,
                seg=seg,
            )
        )

    st.root_xy = tuple(goal.target_xy)
    st.gait_cum_dist = dist
    # Read the end-state off the LAST FRAME rather than off the goal: the two agree (the controller
    # converges exactly), and taking it from the frame makes a yaw teleport at the segment seam
    # structurally impossible rather than merely unlikely.
    st.walk_end_yaw = frames[-1].yaw
    st.root_yaw = frames[-1].yaw
    st.last_pose156 = frames[-1].pose156.copy()
    if arm_ov is not None and st.held is not None and "navigation_arm" in st.held:
        for joint, value in arm_ov.items():
            st.last_pose156[joint * 3 : joint * 3 + 3] = value
        if rhp is not None:
            st.last_pose156[se.core.RHAND_LO : se.core.RHAND_LO + 45] = rhp
    if st.held is not None:
        se._stamp_obj_path(frames, scene, st.held["obj_key"])
        se._stamp_carried(frames, st.held.get("carried"))  # contents ride the walk too
    return frames


def turn_in_place(
    g,
    st,
    from_yaw: float,
    dyaw: float,
    xy,
    arm_ov=None,
    rhp=None,
    obj=("hide", None),
    deg_per_frame: float = 15.0,
    base_pose=None,
    blend_from=None,
) -> list:
    """Rotate the body ``dyaw`` degrees about its own axis at ``xy``, feet planted. THE turn producer.

    A turn is NOT a zero-length walk: ``gait.plan_walk`` sets ``yaw = 0.0`` when the leg is shorter than
    1e-6 m, so routing a turn through the walker would snap the body to world-north instead of rotating
    it.

    The COMMANDED egocentric verbs ([TurnLeft] / [TurnRight] / [TurnAround]) are the only callers left:
    a turn the operator asked for is the point of the frame, so freezing one pose and rotating it is
    the honest rendering of it. Turns that were merely a SEAM between two walks used to come through
    here too, and that is exactly what made every corner pop -- ``_walk_frames`` now steers through
    them instead.

    ``base_pose`` defaults to the gait cycle sampled at the running distance (feet ~in place).
    ``blend_from`` additionally eases the previous body pose into it, which is what a turn issued from a
    STANDING pose wants. ``dyaw`` is signed and NOT normalised, so a caller can ask for a specific
    direction through 180 degrees."""
    import gait_player as gait
    import humanoid_motion as wpp
    import semantic_executor as se

    n = max(se._N_DEPART_MIN, int(round(abs(dyaw) / max(deg_per_frame, 1e-6))))
    pose0 = base_pose if base_pose is not None else gait.sample(g, st.gait_cum_dist)[0]
    out = []
    for i in range(n):
        s = (i + 1) / n
        pose = (
            pose0
            if blend_from is None
            else gait.blend156(blend_from, pose0, wpp._smoothstep(s))
        )
        out.append(
            _hold_frame(pose, xy, from_yaw + dyaw * s, st.floor_z, arm_ov, rhp, obj)
        )
    return out


def forward_xy(yaw_deg: float) -> tuple[float, float]:
    """Unit world-XY heading of a body at ``yaw_deg`` -- the inverse of ``gait.yaw_for_dir``.

    The rig's canonical facing is -Y, so forward(yaw) = [sin(yaw), -cos(yaw)] and the walker's LEFT
    (up x forward) is +X at yaw 0, i.e. yaw = +90: turning left INCREASES yaw."""
    a = math.radians(float(yaw_deg))
    return (math.sin(a), -math.cos(a))


def _run_ahead(
    blocked, start_xy, yaw_deg: float, dist: float, step: float, escape_m: float
) -> float:
    """Longest prefix of the straight ``dist``-metre ray from ``start_xy`` along ``yaw_deg`` that no
    ``blocked(x, y)`` sample rejects. Shared core of the two forward-clearance oracles.

    ``escape_m`` forgives a blocked run at the very START: a stand that hugs the counter it is working at
    legitimately sits inside the body-inflated band, and stepping OUT of that band must stay legal. The
    budget is bounded by the inflation itself, so it can never excuse crossing real geometry."""
    fx, fy = forward_xy(yaw_deg)
    n = max(1, int(math.ceil(dist / max(step, 1e-6))))
    ts = [min(dist, (k + 1) * step) for k in range(n)]
    free = [not blocked(start_xy[0] + fx * t, start_xy[1] + fy * t) for t in ts]
    if not any(free):
        return 0.0
    i = 0
    while i < len(ts) and not free[i] and ts[i] <= escape_m:
        i += 1
    last_ok = 0.0
    for k in range(i, len(ts)):
        if not free[k]:
            return last_ok
        last_ok = ts[k]
    return float(dist)


def free_run_ahead(grid, start_xy, yaw_deg: float, dist: float) -> float:
    """How far a straight walk from ``start_xy`` along ``yaw_deg`` stays on FREE nav-grid cells (metres).

    Returns ``dist`` when the whole segment is walkable. The grid occupancy is already Minkowski-dilated
    by the human radius, so a free cell means the BODY fits there, not merely its centre point."""
    return _run_ahead(
        lambda x, y: not grid.is_free(*grid.world_to_cell(x, y)),
        start_xy,
        yaw_deg,
        dist,
        step=0.25 * grid.cell,
        escape_m=float(grid.inflate or 0.10) + grid.cell,
    )


def box_run_ahead(
    boxes, start_xy, yaw_deg: float, dist: float, radius: float = 0.25
) -> float:
    """The same measurement for a scene with NO nav grid: clearance against the furniture AABBs from
    ``scene.solid_boxes()``, each grown by the body ``radius``. Coarser than the grid (footprint pillars,
    no walls) but it is the same map the reach gate and the render nudge already trust."""

    def _blocked(x: float, y: float) -> bool:
        for _name, mn, mx in boxes:
            if (mn[0] - radius) <= x <= (mx[0] + radius) and (mn[1] - radius) <= y <= (
                mx[1] + radius
            ):
                return True
        return False

    return _run_ahead(
        _blocked, start_xy, yaw_deg, dist, step=0.05, escape_m=radius + 0.05
    )


def _hold_frame(pose, xy, yaw, floor_z, arm_ov, rhp, obj):
    import humanoid_motion as wpp
    import semantic_executor as se

    return wpp.Frame(
        pose,
        (float(xy[0]), float(xy[1])),
        float(yaw),
        se._pin_z(pose, floor_z),
        arm_override=arm_ov,
        rhp=rhp,
        fpv_mode="walk",
        obj=obj,
        seg="turn",
    )


# ---------------------------------------------------------------------------------------------
# FPV eye push-out (grid SDF clamp)
# ---------------------------------------------------------------------------------------------
def push_eye_out(
    eye, head, grid, eye_clear: float = 0.04, min_fwd_frac: float = 0.5, state=None
) -> np.ndarray:
    """Pull the forward-pushed FPV eye out of a WALL along head->eye, using the TRUE-wall SDF.

    Fires ONLY when the eye's XY is INSIDE a true wall footprint (wall_sdf < 0) -- i.e. the camera is in
    opaque geometry and the frame would render black. Merely being NEAR a wall renders fine, so we must
    NOT touch it (touching it was the black-frame regression: it moved a perfectly-good eye into the wall
    or skull). When it fires, the eye is pulled back to the farthest point clear of the wall, never closer
    than ``min_fwd_frac`` of the forward offset (else the camera enters the skull -> also black); if
    nothing in that range clears, the forward eye is KEPT (no worse than the pre-nav camera). No EMA: it
    fires rarely/briefly and the look direction is already _cap_dir-smoothed downstream."""
    eye = np.asarray(eye, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    if grid.wall_sdf is None or grid.wall_sdf_at(float(eye[0]), float(eye[1])) >= 0.0:
        return eye
    best_t = 1.0  # keep-forward fallback (never worse than the pre-nav forward eye)
    steps = 12
    for k in range(steps + 1):
        t = 1.0 - (1.0 - min_fwd_frac) * (
            k / steps
        )  # scan 1.0 (forward) down to min_fwd_frac
        p = head + t * (eye - head)
        if grid.wall_sdf_at(float(p[0]), float(p[1])) >= eye_clear:
            best_t = t
            break
    return head + best_t * (eye - head)


def _eye_ema(eye, state, alpha: float = 0.4) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    if state is None:
        return eye
    prev = state.get("eye_push_prev")
    if prev is not None:
        eye = np.asarray(prev, dtype=np.float64) * (1.0 - alpha) + eye * alpha
    state["eye_push_prev"] = eye.copy()
    return eye


# ---------------------------------------------------------------------------------------------
# Offline self-test (pure python: numpy+scipy via nav_grid, NO Isaac boot)
# ---------------------------------------------------------------------------------------------
def _selftest() -> int:
    import nav_grid

    cell = 0.15
    nx = ny = int(round(4.0 / cell))
    occ = np.zeros((ny, nx), dtype=bool)
    jmid = ny // 2
    occ[jmid, :] = True
    gi0, gi1 = int(1.5 / cell), int(2.5 / cell)
    occ[jmid, gi0:gi1] = False  # a door opening
    grid = nav_grid.NavGrid(
        0.0, 0.0, cell, nx, ny, occ, nav_grid._two_sided_sdf(occ, cell)
    )

    # SAME side (both x=0.4): a straight line hits the solid wall, so the planner MUST detour through the
    # door at x in [1.5, 2.5] -> the polyline has an intermediate waypoint near the door.
    start, goal = (0.4, 0.4), (0.4, 3.6)
    path = plan_path(grid, start, goal)

    def _dense_free(pts) -> bool:
        for a, b in zip(pts[:-1], pts[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(length / (cell * 0.4)))
            for t in range(n + 1):
                f = t / n
                x = a[0] + (b[0] - a[0]) * f
                y = a[1] + (b[1] - a[1]) * f
                if not grid.is_free(*grid.world_to_cell(x, y)):
                    return False
        return True

    ok_ends = abs(path[0][0] - start[0]) < 1e-6 and abs(path[-1][1] - goal[1]) < 1e-6
    ok_free = _dense_free(
        path
    )  # the WHOLE polyline stays in free space (never crosses the wall)
    ok_door = any(1.5 <= p[0] <= 2.5 for p in path)  # detours through the door opening
    ok = ok_ends and ok_free and ok_door and len(path) >= 3
    print(
        f"[nav_planner] selftest legs={len(path)} through_door={ok_door} ends_exact={ok_ends} "
        f"dense_free={ok_free} -> {'PASS' if ok else 'FAIL'}  "
        f"path={[tuple(round(v, 2) for v in p) for p in path]}"
    )
    # The ROUTER is only half the contract. `init_validation`'s stand search does not call plan_path on
    # the common path -- it uses a cheap connectivity PROXY (`_reachable_free_within`) as a stand-in, and
    # the two disagree in BOTH directions. That disagreement, not the router, was the mechanism behind 41
    # of the 107 `path_blocked` occurrences in the cb1/cb2 baseline, so anyone changing `_GOAL_SNAP_R`,
    # the grid inflation or `_shuffle_tol` needs to see it move. Imported lazily: `init_validation` pulls
    # `nav_planner` in itself, and this keeps the offline selftest free of the cycle.
    import init_validation as _iv

    proxy_rc = _iv._run_nav_proxy_control()
    return 0 if (ok and proxy_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

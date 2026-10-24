# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Door-crossing PRE-PASS: forward-simulate a Goal list and splice OpenDoor (and optionally CloseDoor)
whenever a NAVIGATE_TO leg must cross a CLOSED door.

Additive; runs from ``semantic_executor.execute`` ONLY when the scene has a door layer
(``scene.doors``) AND a built ``scene.nav_grid`` -- scenes without doors are unaffected (the goal
list is returned unchanged). numpy-only, Isaac-free -> importable + self-testable offline
(``bash launch/run_isaac.sh simulator/physical_world/door_planner.py``).

The crux: each NAVIGATE leg is planned on the DOORS-PASSABLE grid view
(``nav_grid.passable_view`` frees door cells) so a closed door does not make the leg unroutable; every
crossed CLOSED door then gets an OpenDoor spliced in front of the leg. The runtime ``occ`` keeps the door
BLOCKING (a hard PATH_BLOCKED backstop against a missed crossing), and ``produce_open_door`` carves it
free as it opens, so the following NAVIGATE_TO re-plans through the now-open doorway.

A VIRTUAL open-set tracked across the forward-sim opens each door AT MOST ONCE per traversal (no
double-insert on a return trip); ``close_behind`` re-closes it so a later re-crossing re-opens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from goal_schema import Goal, Primitive


# How far in FRONT of the handle the OpenDoor stand sits (a comfortable reach, clear of the thin door).
_REACH_STAND = 0.6


@dataclass
class DoorInfo:
    """One door's navigation metadata (populated by behavior_import; consumed by the pre-pass)."""

    entity_key: str  # scene_state entity name -> OPENDOOR/CLOSEDOOR obj_key
    prim_index: int  # index into scene.door_prims == the nav_grid.door_id label value
    center_xy: tuple  # door footprint center (world XY)
    normal_xy: tuple  # unit through-direction (perpendicular to the door plane)
    stand_xy: tuple  # a default near-side stand (scene.stand_pose_for; superseded per-approach below)
    stand_yaw: float  # facing at stand_xy
    handle_xy: tuple = (
        0.0,
        0.0,
    )  # the grip handle (free edge) world XY -> the OpenDoor stand faces it
    approach: float = 0.7  # how far past the door center the FAR-side stand sits


def _crossed_doors(grid, polyline) -> list[int]:
    """Door indices the polyline passes over, in first-crossing order (dense-sample of grid.door_id)."""
    if getattr(grid, "door_id", None) is None:
        return []
    seen: set[int] = set()
    order: list[int] = []
    for a, b in zip(polyline[:-1], polyline[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(length / (grid.cell * 0.4)))
        for t in range(n + 1):
            f = t / n
            x = a[0] + (b[0] - a[0]) * f
            y = a[1] + (b[1] - a[1]) * f
            i, j = grid.world_to_cell(x, y)
            if grid.in_bounds(i, j):
                d = int(grid.door_id[j, i])
                if d >= 0 and d not in seen:
                    seen.add(d)
                    order.append(d)
    return order


def _near_far(door: DoorInfo, prev_xy) -> tuple:
    """(near_xy, near_yaw, far_xy) for a crossing. near = _REACH_STAND metres in FRONT of the HANDLE on the
    walker's APPROACH side (so the handle is reachable AND the walk to it never crosses the still-closed
    door); near_yaw faces the door. far = the door center offset by ``approach`` on the OPPOSITE side, so a
    following walk / close-behind re-approach clearly reaches the far room."""
    c = np.asarray(door.center_xy, dtype=np.float64)
    nrm = np.asarray(door.normal_xy, dtype=np.float64)
    h = np.asarray(door.handle_xy, dtype=np.float64)
    prev = np.asarray(prev_xy, dtype=np.float64)
    side = (
        1.0 if float(np.dot(prev - c, nrm)) >= 0.0 else -1.0
    )  # the side the walker approaches from
    near = (
        h + nrm * side * _REACH_STAND
    )  # in front of the handle on the walker's OWN side (no crossing)
    fwd = -nrm * side  # the human faces the door (handle ahead)
    near_yaw = math.degrees(math.atan2(float(fwd[0]), -float(fwd[1])))
    far = c - nrm * side * float(door.approach)
    return (
        (float(near[0]), float(near[1])),
        near_yaw,
        (float(far[0]), float(far[1])),
    )


def insert_door_actions(goals, scene, grid, close_behind: bool = False) -> list:
    """Forward-simulate ``goals``; before any NAVIGATE_TO leg that crosses a CLOSED door, splice
    NAVIGATE_TO(door stand) + OPENDOOR (+ far NAV + CLOSEDOOR when ``close_behind``). Returns a NEW goal
    list; the input is unchanged. No-op (returns a copy of ``goals``) when the scene has no door layer."""
    doors = getattr(scene, "doors", None)
    if not doors or getattr(grid, "door_id", None) is None:
        return list(goals)

    import nav_grid
    import nav_planner

    by_index = {d.prim_index: d for d in doors.values()}
    pv = nav_grid.passable_view(grid)
    root = tuple(getattr(scene, "start_xy", (0.0, 0.0)))
    opened = set(
        getattr(scene, "open_objs", set()) or set()
    )  # skip doors already open initially
    out: list = []
    for goal in goals:
        if goal.prim is not Primitive.NAVIGATE_TO or goal.target_xy is None:
            out.append(goal)
            continue
        try:
            path = nav_planner.plan_path(pv, root, tuple(goal.target_xy))
        except Exception:  # noqa: BLE001  (unroutable even doors-open -> fall back to the straight leg)
            path = [root, tuple(goal.target_xy)]
        for d in _crossed_doors(grid, path):
            door = by_index.get(d)
            if door is None or door.entity_key in opened:
                continue
            near_xy, near_yaw, far_xy = _near_far(door, root)
            out.append(
                Goal(
                    Primitive.NAVIGATE_TO,
                    target_xy=near_xy,
                    target_yaw=near_yaw,
                )
            )
            out.append(Goal(Primitive.OPENDOOR, obj_key=door.entity_key))
            opened.add(door.entity_key)
            root = near_xy
            if close_behind:
                out.append(Goal(Primitive.NAVIGATE_TO, target_xy=far_xy))
                out.append(Goal(Primitive.CLOSEDOOR, obj_key=door.entity_key))
                opened.discard(door.entity_key)
                root = far_xy
        out.append(goal)
        root = tuple(goal.target_xy)
    return out


# =====================================================================================================
# Offline selftest (pure python: numpy+scipy via nav_grid, NO Isaac boot).
# =====================================================================================================
def _selftest() -> int:
    import nav_grid

    cell = 0.15
    nx = ny = int(round(4.0 / cell))
    occ = np.zeros((ny, nx), dtype=bool)
    jmid = ny // 2
    occ[jmid, :] = True  # a wall across x
    gi0, gi1 = int(1.5 / cell), int(2.5 / cell)
    door_id = np.full((ny, nx), -1, dtype=np.int32)
    door_cost = np.zeros((ny, nx), dtype=np.float64)
    door_id[jmid, gi0:gi1] = (
        0  # the opening is a CLOSED door (blocked in occ, labeled door 0)
    )
    door_cost[jmid, gi0:gi1] = nav_grid.DOOR_COST_M
    grid = nav_grid.NavGrid(
        0.0,
        0.0,
        cell,
        nx,
        ny,
        occ,
        nav_grid._two_sided_sdf(occ, cell),
        door_id=door_id,
        door_cost=door_cost,
    )
    cy = jmid * cell + 0.5 * cell
    door = DoorInfo(
        entity_key="door",
        prim_index=0,
        center_xy=(2.0, cy),
        normal_xy=(0.0, 1.0),
        stand_xy=(2.0, cy - 0.7),
        stand_yaw=0.0,
    )

    class _S:
        pass

    scene = _S()
    scene.start_xy = (2.0, 0.4)
    scene.doors = {"door": door}
    scene.open_objs = set()

    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and c
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    # (a) a leg that crosses the closed door -> exactly one OPENDOOR, spliced NAV,OPENDOOR before the leg.
    g1 = [Goal(Primitive.NAVIGATE_TO, target_xy=(2.0, 3.6))]
    o1 = insert_door_actions(g1, scene, grid)
    p1 = [g.prim for g in o1]
    chk(
        p1 == [Primitive.NAVIGATE_TO, Primitive.OPENDOOR, Primitive.NAVIGATE_TO],
        f"crossed-closed -> NAV,OPENDOOR,NAV: {[p.value for p in p1]}",
    )
    chk(o1[1].obj_key == "door", "OPENDOOR obj_key=door")

    # (b) a leg on the SAME side (no crossing) -> no insert.
    g2 = [Goal(Primitive.NAVIGATE_TO, target_xy=(2.0, 0.2))]
    o2 = insert_door_actions(g2, scene, grid)
    chk(all(g.prim is not Primitive.OPENDOOR for g in o2), "not-crossed -> no OPENDOOR")

    # (c) return trip (two legs across the door) -> opened ONCE (virtual open-set).
    g3 = [
        Goal(Primitive.NAVIGATE_TO, target_xy=(2.0, 3.6)),
        Goal(Primitive.NAVIGATE_TO, target_xy=(2.0, 0.4)),
    ]
    o3 = insert_door_actions(g3, scene, grid)
    chk(
        sum(1 for g in o3 if g.prim is Primitive.OPENDOOR) == 1,
        "return-trip -> opened once (no double-insert)",
    )

    # (d) close_behind -> re-opened on the second crossing + a CLOSEDOOR present.
    o4 = insert_door_actions(g3, scene, grid, close_behind=True)
    chk(
        sum(1 for g in o4 if g.prim is Primitive.OPENDOOR) == 2,
        "close_behind -> re-open on 2nd crossing",
    )
    chk(
        any(g.prim is Primitive.CLOSEDOOR for g in o4),
        "close_behind -> CLOSEDOOR present",
    )

    # (e) no door layer -> byte-identical passthrough.
    grid0 = nav_grid.NavGrid(
        0.0, 0.0, cell, nx, ny, occ, nav_grid._two_sided_sdf(occ, cell)
    )
    scene0 = _S()
    scene0.start_xy = (2.0, 0.4)
    scene0.doors = None
    scene0.open_objs = set()
    o5 = insert_door_actions(g1, scene0, grid0)
    chk([g.prim for g in o5] == [Primitive.NAVIGATE_TO], "no door layer -> passthrough")

    print(f"[door_planner] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

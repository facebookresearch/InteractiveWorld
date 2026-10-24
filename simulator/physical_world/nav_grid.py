# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-episode 2D navigation grid: occupancy + two-sided signed-distance field (SDF).

ADDITIVE. This module is the single shared collision artifact for the navigation feature
(path planning + human-wall anti-penetration + FPV eye push-out). It is built ONCE per
episode from the SAME colliders the init non-penetration gate uses, so a planned free cell
is a cell the init gate would also accept.

Design decisions:
- FURNITURE layer: reuse ``init_validation.build_occupancy_grid`` over ``collidable_footprints``
  with the wall prims EXCLUDED (their bounding-cube colliders would seal doorways).
- WALL layer: a doorway-preserving VISUAL-MESH-TRIANGLE rasterizer over ``scene.wall_prims``,
  restricted to the human pass band [floor+LO, floor+HI] so door lintels/floor trim do not
  fill the opening, then human-radius dilated.
- SDF: init_validation's ``clearance`` is ONE-SIDED (-inf inside), unusable for a gradient push,
  so we build a real two-sided SDF here (>0 outside, <0 inside).

pxr/scipy are imported lazily (pxr only imports after Isaac's SimulationApp boots; the pure-python
helpers + _selftest run under the conda python with numpy/scipy only, no Isaac boot).
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Human pass band (m, relative to floor_z): a triangle is rasterized into the wall footprint only
# if its z-extent overlaps this band. Below LO = floor trim/baseboards; above HI = door lintels /
# ceiling. Keeping HI under a standard ~2.0 m door height is what preserves doorways.
WALL_BAND_LO = 0.20
WALL_BAND_HI = 1.90

# Furniture pass band (m, rel. floor_z) for the PRECISE collision-mesh raster (scene.precise_nav): a cell
# is blocked only where furniture geometry actually sits in the standing human's column. LO skips floor
# mats/trim; HI = standing-human height (you can't walk under a table, so a 0.75 m top DOES block).
FURN_BAND_LO = 0.05
FURN_BAND_HI = 1.90

# Walkable support is the scene's real floor mesh, sampled in a narrow band around floor_z. Otherwise
# every empty cell inside the global scene AABB would count as free, including the unmodelled grey
# void outside a house.
FLOOR_BAND_BELOW = 0.12
FLOOR_BAND_ABOVE = 0.06

# Extra A* path-length penalty (m) for routing through a CLOSED door: a small nudge so an already-open
# doorway of comparable length is preferred, but not so large that a door which is the ONLY route is
# distorted in multi-goal ordering. Applied only on door cells and only when door_cost is populated.
DOOR_COST_M = 2.5

# Human-radius inflations the adaptive/task builders try, LARGEST first (the last rung is
# init_validation.NAV_INFLATE_M, the fixed default inflation). The compliance judge tests body
# capsules up to 0.140 m and the runtime nudge can recover at most 0.08 m, so a corridor planned at the
# 0.10 m floor is clipped BY CONSTRUCTION wherever furniture sits within a body radius of the route; the
# ladder takes the widest margin the scene actually admits and only steps down when a wider one would
# disconnect a target. Rungs are multiples of the 0.10 m cell so the real-valued Minkowski dilation lands
# on clean cell boundaries.
#
# The top rung is 0.20 and NOT wider, which is a measured limit rather than a taste: this grid has no
# height discrimination (see FURN_BAND_LO/HI -- a 0.5 m bin blocks its whole column), so past ~0.20 m the
# halo of a SHORT obstacle swallows the only floor a human can legally stand on next to it, and the
# runtime stand search then finds no walkable stand at all -- trading a clipped episode for an unsolvable
# one. 0.20 still clears the widest judge capsule (0.140 m pelvis) with margin, which is the whole point.
NAV_INFLATION_LADDER = (0.20, 0.15, 0.10)
EGOCENTRIC_NAV_INFLATE_M = 0.05
EGOCENTRIC_NAV_INFLATION_LADDER = (EGOCENTRIC_NAV_INFLATE_M, 0.0)
EGOCENTRIC_DOOR_PORTAL_HALF_WIDTH_M = 0.20
# A route rejected by the real 3-D body sweep must exclude a body-sized neighbourhood on retry.
# Excluding only the single 10 cm cell lets A* select the adjacent cell inside the same hollow rack,
# exhausting the bounded retry budget before it ever routes around the furniture.
TRANSIT_COLLISION_AVOID_RADIUS_M = 0.20

# Max rasterized CELLS (triangle fill) / SAMPLE POINTS (edge DDA) expanded in one vectorized chunk. Caps the
# peak temporary allocation (~a dozen float64 temporaries per element) at a few hundred MB.
_RASTER_CHUNK = 1_000_000
# Triangles buffered across meshes before a flush. BEHAVIOR convex-hull colliders average ~75 triangles, so
# flushing per mesh would pay numpy's per-call overhead thousands of times instead of amortising it.
_TRI_BATCH = 40_000


def _legacy(feature: str) -> bool:
    """Kill-switch for the vectorized fast paths: ``NAV_GRID_LEGACY=1`` (all) or
    ``NAV_GRID_LEGACY_FLOOD`` / ``NAV_GRID_LEGACY_RASTER`` (one). Read per call so an A/B can flip it in
    process. The fast paths are proven bit-identical; the switch exists to bisect a suspected divergence."""
    return (
        os.environ.get("NAV_GRID_LEGACY") == "1"
        or os.environ.get(f"NAV_GRID_LEGACY_{feature}") == "1"
    )


@dataclass
class NavGrid:
    """2D XY grid over the scene floor. ``occ[j, i]`` True = cell (i=x-col, j=y-row) is blocked.

    Index convention matches init_validation.OccupancyGrid: arrays are [row=y, col=x]; the
    ``cell_center``/``world_to_cell`` helpers take/return (i=x, j=y)."""

    x0: float
    y0: float
    cell: float
    nx: int
    ny: int
    occ: (
        np.ndarray
    )  # (ny, nx) bool: True = blocked (furniture + walls, human-radius inflated)
    sdf: (
        np.ndarray
    )  # (ny, nx) float (m): COMBINED signed distance (furniture+walls), for planning
    wall_sdf: np.ndarray | None = (
        None  # (ny, nx) TRUE-wall-only signed distance (undilated), for eye/body anti-pen
    )
    floor_support: np.ndarray | None = (
        None  # (ny, nx) bool: real modelled floor under the human; False is never walkable
    )
    # Door layer (None when the scene has no doors). A CLOSED door stays BLOCKING in
    # ``occ`` (a hard PATH_BLOCKED backstop); this SEPARATE layer marks which cells belong to which door
    # so the passable view can free them, the pre-pass can detect a crossing, and A* can add an open-cost.
    door_id: np.ndarray | None = (
        None  # (ny, nx) int32: index into scene.door_prims (-1 = not a door)
    )
    door_cost: np.ndarray | None = (
        None  # (ny, nx) float: DOOR_COST_M on a closed-door cell, 0.0 elsewhere
    )
    base_occ: np.ndarray | None = (
        None  # (ny, nx) bool: walls/furniture/floor support, excluding movable door leaves
    )
    open_leaf_count: np.ndarray | None = (
        None  # (ny, nx) int: number of currently-open door leaves covering each cell
    )
    # Body radius (m) this grid's ``occ`` was Minkowski-dilated by. Consumers that must reconcile a point
    # OUTSIDE the walkable set (a stand hugging furniture) with the walkable set need it: the shortest
    # shuffle from such a point to a free cell grows with the inflation, so a fixed tolerance silently
    # rejects every furniture-hugging stand once the ladder picks a wide rung. None = not recorded
    # (hand-built test grids); consumers fall back to init_validation.NAV_INFLATE_M.
    inflate: float | None = None
    transit_avoid: set[tuple[int, int]] = field(default_factory=set)
    raw_occ: np.ndarray | None = (
        None  # (ny, nx) bool: undilated real walls/furniture/floor bounds, excluding door leaves
    )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((x - self.x0) / self.cell)),
            int(math.floor((y - self.y0) / self.cell)),
        )

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        return (self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell)

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_free(self, i: int, j: int) -> bool:
        return (
            self.in_bounds(i, j)
            and not bool(self.occ[j, i])
            and (i, j) not in self.transit_avoid
        )

    def floor_supported_at(self, x: float, y: float) -> bool | None:
        """Whether the one authoritative floor mask supports world point ``(x, y)``."""
        if self.floor_support is None:
            return None
        i, j = self.world_to_cell(x, y)
        return bool(self.in_bounds(i, j) and self.floor_support[j, i])

    def _bilinear(self, arr: np.ndarray, x: float, y: float) -> float:
        """Bilinear sample of a (ny, nx) field at world (x, y). Cell-center aligned; clamps to the edge."""
        fx = (x - self.x0) / self.cell - 0.5
        fy = (y - self.y0) / self.cell - 0.5
        i0 = int(math.floor(fx))
        j0 = int(math.floor(fy))
        tx = fx - i0
        ty = fy - j0

        def s(i: int, j: int) -> float:
            i = min(max(i, 0), self.nx - 1)
            j = min(max(j, 0), self.ny - 1)
            return float(arr[j, i])

        return (
            s(i0, j0) * (1 - tx) * (1 - ty)
            + s(i0 + 1, j0) * tx * (1 - ty)
            + s(i0, j0 + 1) * (1 - tx) * ty
            + s(i0 + 1, j0 + 1) * tx * ty
        )

    def _grad(self, sample_fn, x: float, y: float) -> tuple[float, float]:
        h = self.cell
        gx = sample_fn(x + h, y) - sample_fn(x - h, y)
        gy = sample_fn(x, y + h) - sample_fn(x, y - h)
        n = math.hypot(gx, gy)
        if n < 1e-9:
            return (0.0, 0.0)
        return (gx / n, gy / n)

    def sdf_at(self, x: float, y: float) -> float:
        """Bilinear COMBINED SDF (furniture+walls). For planning / connectivity."""
        return self._bilinear(self.sdf, x, y)

    def sdf_grad(self, x: float, y: float) -> tuple[float, float]:
        return self._grad(self.sdf_at, x, y)

    def wall_sdf_at(self, x: float, y: float) -> float:
        """Bilinear TRUE-wall-only SDF (>0 outside the wall footprint). 10.0 (far) when the scene has no
        walls. Used for the eye push-out + body anti-pen: walls span full height, so their XY footprint is
        valid at head height, whereas the combined SDF would fire spuriously over a low counter."""
        if self.wall_sdf is None:
            return 10.0
        return self._bilinear(self.wall_sdf, x, y)

    def wall_grad(self, x: float, y: float) -> tuple[float, float]:
        return self._grad(self.wall_sdf_at, x, y)

    def nearest_free(
        self, i: int, j: int, max_r: int | None = None
    ) -> tuple[int, int] | None:
        """Nearest free cell to (i, j) by expanding-ring search (returns (i, j) or None)."""
        if self.is_free(i, j):
            return (i, j)
        big = max_r or max(self.nx, self.ny)
        for r in range(1, big + 1):
            for di in range(-r, r + 1):
                for dj in (-r, r):
                    if self.is_free(i + di, j + dj):
                        return (i + di, j + dj)
            for dj in range(-r + 1, r):
                for di in (-r, r):
                    if self.is_free(i + di, j + dj):
                        return (i + di, j + dj)
        return None


def add_transit_avoidance(
    grid: NavGrid,
    xy: tuple[float, float],
    radius_m: float,
) -> int:
    """Persist a small root-space exclusion learned from a real 3-D body collision."""
    ci, cj = grid.world_to_cell(float(xy[0]), float(xy[1]))
    radius_cells = max(0, int(math.ceil(float(radius_m) / grid.cell)))
    before = len(grid.transit_avoid)
    if grid.in_bounds(ci, cj):
        grid.transit_avoid.add((ci, cj))
    for dj in range(-radius_cells, radius_cells + 1):
        for di in range(-radius_cells, radius_cells + 1):
            i, j = ci + di, cj + dj
            if not grid.in_bounds(i, j):
                continue
            if math.dist(grid.cell_center(i, j), xy) <= radius_m + 0.5 * grid.cell:
                grid.transit_avoid.add((i, j))
    return len(grid.transit_avoid) - before


def planning_view(grid: NavGrid) -> NavGrid:
    """Return a shallow planning copy with private speculative route exclusions."""
    import dataclasses

    return dataclasses.replace(grid, transit_avoid=set(grid.transit_avoid))


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------
def _precise_occupancy_cache_owner(scene):
    """Return the cache owner when precise task navigation is available."""
    cworld = getattr(scene, "cworld", None)
    if cworld is not None:
        return cworld
    if getattr(scene, "precise_nav", False):
        return scene
    return None


def build_nav_grid(
    stage, scene, cell: float = 0.10, inflate: float | None = None
) -> NavGrid:
    """Build the per-episode NavGrid from the live stage + the grounded scene.

    Reuses init_validation for the furniture layer + region/transform (so planned cells match the
    init gate), rasterizes walls doorway-safely, unions, and builds a two-sided SDF."""
    import init_validation as iv

    if inflate is None:
        inflate = iv.NAV_INFLATE_M
    floor_z = float(getattr(scene, "floor_z", 0.0))
    wall_prims = list(getattr(scene, "wall_prims", []) or [])
    floor_prims = list(getattr(scene, "floor_prims", []) or [])
    door_prims = list(getattr(scene, "door_prims", []) or [])
    door_groups = _door_leaf_groups(scene, door_prims)
    door_leaf_paths = tuple(path for group in door_groups for path in group)

    # Region from the FULL footprint set (walls included) so the grid spans the whole floor.
    foot_full = iv.collidable_footprints(stage)
    region = iv._auto_region(
        foot_full, getattr(scene, "start_xy", (0.0, 0.0)), floor_z=floor_z
    )
    # Furniture layer: walls excluded (their bounding-cube AABBs seal doorways).
    exclude = (
        (
            "/World/HumanProxy",
            "/World/Body",
            "/World/Obj",
        )
        + tuple(wall_prims)
        + tuple(floor_prims)
        + door_leaf_paths
    )
    foot_furn = iv.collidable_footprints(stage, exclude_prefixes=exclude)
    og = iv.build_occupancy_grid(
        foot_furn, region, cell=cell, inflate=inflate, floor_z=floor_z
    )
    # FURNITURE occupancy = the SSOT PRECISE collision-mesh triangle raster (blocks only where geometry
    # really is, then Minkowski-inflated by the human radius, same as the wall layer). ONE path: no
    # per-scene precise/AABB choice -- the UNDILATED raster is OWNED + cached by the collision oracle
    # (scene.cworld) so nav AND the stand search read the SAME map. build_nav_grid_adaptive rebuilds
    # at 7 inflations but reuses this cached raster (only re-dilating). AABB occupancy remains ONLY as a
    # fallback when there is no oracle or no precise meshes (it fills under-table / L-concavities coarsely).
    precise_cache = _precise_occupancy_cache_owner(scene)
    occ_precise = (
        getattr(precise_cache, "_precise_occ", None)
        if precise_cache is not None
        else None
    )
    if precise_cache is not None and (
        occ_precise is None or occ_precise.shape != (og.ny, og.nx)
    ):
        occ_precise = _raster_furniture(
            stage, exclude, og.x0, og.y0, og.cell, og.nx, og.ny, floor_z
        )
        precise_cache._precise_occ = occ_precise
    raw_occ = (
        np.array(occ_precise, dtype=bool, copy=True)
        if occ_precise is not None
        else None
    )
    if occ_precise is not None and occ_precise.any():
        from scipy import ndimage

        # REAL-valued Minkowski inflation: a free cell is blocked iff it lies within `inflate` metres of real
        # furniture. distance_transform_edt gives each free cell's distance (in cells) to the nearest
        # obstacle; x cell -> metres. Integer-cell binary_dilation would quantise the inflation to multiples
        # of `cell` (0.20 collapsing to 0.15) and force a >=1-cell floor; here 0.10 is truly 0.10.
        dist_m = ndimage.distance_transform_edt(~occ_precise) * cell
        occ = occ_precise | (dist_m < inflate)
    else:
        occ = np.array(og.occ, dtype=bool)

    wall_sdf = None
    if wall_prims:
        occ_wall = _raster_walls(
            stage, wall_prims, og.x0, og.y0, og.cell, og.nx, og.ny, floor_z
        )
        if occ_wall.any():
            from scipy import ndimage

            # Planning uses the human-radius-DILATED walls (unioned into occ); the eye/body anti-pen uses
            # the TRUE (undilated) wall footprint so it fires only on real wall geometry. Same real-valued
            # distance-transform inflation as the furniture layer (no cell quantisation).
            dist_w = ndimage.distance_transform_edt(~occ_wall) * cell
            occ = occ | occ_wall | (dist_w < inflate)
            wall_sdf = _two_sided_sdf(occ_wall, cell)
            if raw_occ is not None:
                raw_occ |= occ_wall

    floor_support = None
    if floor_prims:
        floor_key = (
            tuple(str(path) for path in floor_prims),
            round(float(og.x0), 6),
            round(float(og.y0), 6),
            round(float(og.cell), 6),
            int(og.nx),
            int(og.ny),
            round(float(floor_z), 6),
        )
        floor_support = getattr(scene, "_floor_support", None)
        cached_key = getattr(scene, "_floor_support_key", None)
        if (
            floor_support is None
            or floor_support.shape != (og.ny, og.nx)
            or cached_key != floor_key
        ):
            floor_support = _raster_floor_support(
                stage,
                floor_prims,
                og.x0,
                og.y0,
                og.cell,
                og.nx,
                og.ny,
                floor_z,
            )
            scene._floor_support = floor_support
            scene._floor_support_key = floor_key
        if not floor_support.any():
            raise RuntimeError(
                "navigation floor mask is empty although the scene declares floor prims"
            )
        occ = occ | ~floor_support
        if raw_occ is not None:
            raw_occ |= ~floor_support

    base_occ = np.array(occ, dtype=bool, copy=True)
    # Door leaves are excluded from ``base_occ`` above and added back as their own closed-only layer.
    # Opening a door can therefore remove only leaf cells; overlapping walls, jambs, furniture and
    # unsupported floor remain blocked.
    door_id = door_cost = None
    if door_groups:
        door_id, door_cost = _raster_doors(
            stage,
            door_groups,
            og.x0,
            og.y0,
            og.cell,
            og.nx,
            og.ny,
            inflate=inflate,
        )
        occ = occ | (door_id >= 0)
    sdf = _two_sided_sdf(occ, cell)
    return NavGrid(
        og.x0,
        og.y0,
        og.cell,
        og.nx,
        og.ny,
        occ,
        sdf,
        wall_sdf=wall_sdf,
        floor_support=floor_support,
        door_id=door_id,
        door_cost=door_cost,
        base_occ=base_occ,
        raw_occ=raw_occ,
        inflate=float(inflate),
    )


def build_nav_grid_adaptive(
    stage,
    scene,
    start_xy,
    stand_xys,
    inflations=NAV_INFLATION_LADDER,
    carve_radius: float = 0.30,
    carve: bool = True,
):
    """Build the grid at the LARGEST inflation that keeps start + every stand in ONE connected free
    component (max body clearance the scene allows; the runtime nudge backstops the remainder). Coarse
    bounding-cube furniture + full human-radius inflation fragments tight scenes, so a fixed inflation is
    brittle. Returns (grid, inflate, n_connected, n_targets); if none fully connect, returns the candidate
    that reaches the most stands (caller falls back to the straight-line walk)."""
    targets = [tuple(t) for t in stand_xys]
    best = None
    scan: list[str] = []
    for infl in inflations:
        g = build_nav_grid(stage, scene, inflate=infl)
        # carve=False: do NOT free the target cells (used when the targets are object CONTACTS that sit ON
        # furniture -- carving them would free counter cells and falsely 'connect' them; instead connected()
        # snaps each contact to its nearest reachable floor, so the chosen inflation reflects REAL walkable
        # access to the objects). Always carve the validated start.
        carve_free(
            g,
            ([tuple(start_xy), *targets] if carve else [tuple(start_xy)]),
            radius=carve_radius,
        )
        # Count connectivity on the DOORS-PASSABLE view: a closed door is openable, so a stand behind it
        # must not fragment the grid (this is what lets scene.nav_grid get SET for a closed-door scene like
        # ihlen_0). The returned grid `g` still keeps doors BLOCKING in occ. Identity-preserving when the
        # scene has no doors (passable_view returns g).
        gv = passable_view(g)
        n_ok = sum(1 for t in targets if connected(gv, start_xy, t))
        scan.append(f"{infl:.2f}:{n_ok}")
        if best is None or n_ok > best[2]:
            best = (g, infl, n_ok)
        if n_ok == len(targets):
            break
    print(
        f"[nav_grid] inflation ladder over {len(targets)} stand(s): {' '.join(scan)} "
        f"-> chose {best[1]:.2f} ({best[2]}/{len(targets)} connected)"
    )
    return best[0], best[1], best[2], len(targets)


def _reachable_within(
    grid: NavGrid, reach, cx: float, cy: float, reach_m: float
) -> bool:
    """True iff some reachable-from-start FREE cell lies within ``reach_m`` (arm reach) of world (cx, cy).
    Unlike ``connected`` (which snaps a blocked point to its nearest free cell ARBITRARILY far), this asks
    'can a human stand within arm's reach and grab it', which is the real question for an object ON
    furniture."""
    ci, cj = grid.world_to_cell(cx, cy)
    r = int(math.ceil(reach_m / grid.cell)) + 1
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            i, j = ci + di, cj + dj
            if (
                0 <= i < grid.nx
                and 0 <= j < grid.ny
                and reach[j, i]
                and not grid.occ[j, i]
            ):
                wx, wy = grid.cell_center(i, j)
                if math.hypot(wx - cx, wy - cy) <= reach_m:
                    return True
    return False


def build_task_nav_grid(
    stage,
    scene,
    start_xy,
    contacts,
    inflations=NAV_INFLATION_LADDER,
    reach_m: float = 0.78,
    carve_radius: float = 0.30,
):
    """Pick the LARGEST inflation at which EVERY object CONTACT has a reachable-from-start free cell within
    ``reach_m`` (i.e. a human can stand near it and grab it). This drives the task grid inflation off REAL
    walkable access to the objects, using ``_reachable_within`` (not ``connected``, whose nearest-free snap
    makes an on-furniture contact look reachable via arbitrarily-far floor). Carves only the validated
    start. Returns (grid, inflate, n_ok, n_contacts)."""
    cs = [tuple(c) for c in contacts]
    best = None
    scan: list[str] = []
    for infl in inflations:
        g = build_nav_grid(stage, scene, inflate=infl)
        carve_free(g, [tuple(start_xy)], radius=carve_radius)
        gv = passable_view(g)
        reach = flood_free(gv, gv.world_to_cell(float(start_xy[0]), float(start_xy[1])))
        n_ok = sum(
            1 for (cx, cy) in cs if _reachable_within(gv, reach, cx, cy, reach_m)
        )
        scan.append(f"{infl:.2f}:{n_ok}")
        if best is None or n_ok > best[2]:
            best = (g, infl, n_ok)
        if n_ok == len(cs):
            break
    print(
        f"[nav_grid] inflation ladder over {len(cs)} contact(s): {' '.join(scan)} "
        f"-> chose {best[1]:.2f} ({best[2]}/{len(cs)} reachable)"
    )
    return best[0], best[1], best[2], len(cs)


def task_start_region(
    grid: NavGrid,
    contacts,
    reach_m: float = 0.78,
) -> tuple[np.ndarray, int, int]:
    """Return free cells in the component reaching the most task contacts.

    The returned mask is a start-placement constraint, not a navigation shortcut: it contains only
    genuinely free cells from the door-passable runtime grid.  A contact counts for a component only
    when that component has a free cell within the same arm-reach radius used by
    ``build_task_nav_grid``.  All equally good components are retained so the caller's existing seeded
    ranking remains the tie-breaker.
    """
    from scipy import ndimage

    view = passable_view(grid)
    free = ~np.asarray(view.occ, dtype=bool)
    for i, j in view.transit_avoid:
        if view.in_bounds(i, j):
            free[j, i] = False
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labels, count = ndimage.label(free, structure=cross)
    contact_rows = [tuple(contact) for contact in contacts]
    if count == 0 or not contact_rows:
        return free, 0, len(contact_rows)

    scores = np.zeros(count + 1, dtype=np.int32)
    radius = int(math.ceil(float(reach_m) / view.cell)) + 1
    for cx, cy in contact_rows:
        ci, cj = view.world_to_cell(float(cx), float(cy))
        nearby: set[int] = set()
        for dj in range(-radius, radius + 1):
            for di in range(-radius, radius + 1):
                i, j = ci + di, cj + dj
                if not view.in_bounds(i, j) or not free[j, i]:
                    continue
                wx, wy = view.cell_center(i, j)
                if math.hypot(wx - float(cx), wy - float(cy)) <= float(reach_m):
                    nearby.add(int(labels[j, i]))
        for label in nearby:
            if label > 0:
                scores[label] += 1

    reachable = int(scores.max(initial=0))
    if reachable == 0:
        return np.zeros(free.shape, dtype=bool), 0, len(contact_rows)
    winners = np.flatnonzero(scores == reachable)
    winners = winners[winners > 0]
    return np.isin(labels, winners), reachable, len(contact_rows)


def _door_leaf_groups(scene, door_prims) -> list[tuple[str, ...]]:
    """Movable leaf paths for each door, retaining the asset root as a legacy fallback."""
    groups = [(str(path),) for path in door_prims]
    for info in (getattr(scene, "doors", None) or {}).values():
        index = int(getattr(info, "prim_index", -1))
        if not 0 <= index < len(groups):
            continue
        entity = getattr(scene, "entities", {}).get(getattr(info, "entity_key", ""))
        leaves = tuple(str(path) for path in getattr(entity, "leaf_paths", ()) if path)
        if leaves:
            groups[index] = leaves
    return groups


def _raster_doors(stage, door_groups, x0, y0, cell, nx, ny, inflate: float = 0.0):
    """Rasterize each door's CLOSED movable-leaf AABB footprint into
    (door_id, door_cost).

    ``door_id[j,i]`` indexes ``door_groups`` (-1 = no door); ``door_cost`` adds the closed-door routing
    penalty. Only movable leaves are labelled: fixed frames, walls and jambs remain in ``base_occ`` and
    can never be erased by opening a door.
    """
    from pxr import Usd, UsdGeom

    door_id = np.full((ny, nx), -1, dtype=np.int32)
    door_cost = np.zeros((ny, nx), dtype=np.float64)
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
    )
    r = max(1, int(round(inflate / cell))) if inflate > 0 else 0
    ndimage = None
    if r > 0:
        from scipy import ndimage
    for d, paths in enumerate(door_groups):
        mask = np.zeros((ny, nx), dtype=bool)
        for wp in paths:
            prim = stage.GetPrimAtPath(wp)
            if not prim or not prim.IsValid():
                continue
            rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            i0 = max(0, int(math.floor((mn[0] - x0) / cell)))
            i1 = min(nx - 1, int(math.floor((mx[0] - x0) / cell)))
            j0 = max(0, int(math.floor((mn[1] - y0) / cell)))
            j1 = min(ny - 1, int(math.floor((mx[1] - y0) / cell)))
            if i1 < i0 or j1 < j0:
                continue
            mask[j0 : j1 + 1, i0 : i1 + 1] = True
        if r > 0:
            mask = ndimage.binary_dilation(mask, iterations=r)
        door_id[mask] = d
        door_cost[mask] = DOOR_COST_M
    return door_id, door_cost


def passable_view(grid: NavGrid) -> NavGrid:
    """Return a view that frees closed leaves while retaining real open-leaf occupancy.

    Static walls, jambs, overlapping furniture, floor bounds, and already-swung door leaves remain
    blocked. The original grid's occupancy and SDF are untouched.
    """
    if grid.door_id is None:
        return planning_view(grid)
    import dataclasses

    occ = (
        np.array(grid.base_occ, dtype=bool, copy=True)
        if grid.base_occ is not None
        else grid.occ & (grid.door_id < 0)
    )
    open_leaf_count = getattr(grid, "open_leaf_count", None)
    if open_leaf_count is not None and open_leaf_count.shape == occ.shape:
        occ |= open_leaf_count > 0
    if grid.floor_support is not None:
        occ = occ | ~grid.floor_support
    return dataclasses.replace(
        grid,
        occ=occ,
        sdf=_two_sided_sdf(occ, grid.cell),
        transit_avoid=set(grid.transit_avoid),
    )


def _registered_door_portal_mask(grid: NavGrid, scene) -> np.ndarray:
    """Central lanes of registered door leaves, used only by validated fallback routing."""
    portal = np.zeros(grid.occ.shape, dtype=bool)
    if grid.door_id is None:
        return portal
    rows, cols = np.indices(grid.occ.shape)
    xs = grid.x0 + (cols + 0.5) * grid.cell
    ys = grid.y0 + (rows + 0.5) * grid.cell
    entities = getattr(scene, "entities", None) or {}
    for info in (getattr(scene, "doors", None) or {}).values():
        nx, ny = (float(value) for value in info.normal_xy)
        tangent_x, tangent_y = -ny, nx
        lateral = np.abs(
            (xs - float(info.center_xy[0])) * tangent_x
            + (ys - float(info.center_xy[1])) * tangent_y
        )
        door_cells = grid.door_id == int(info.prim_index)
        entity = entities.get(info.entity_key)
        for i, j in (
            (getattr(entity, "meta", None) or {}).get("_nav_closed_door_cells", ())
            if entity is not None
            else ()
        ):
            if grid.in_bounds(int(i), int(j)):
                door_cells[int(j), int(i)] = True
        portal |= door_cells & (
            lateral <= EGOCENTRIC_DOOR_PORTAL_HALF_WIDTH_M + 0.5 * grid.cell
        )
    return portal


def egocentric_passable_view(
    grid: NavGrid,
    scene,
    clearance_m: float = EGOCENTRIC_NAV_INFLATE_M,
    additional_open_leaf_boxes=(),
) -> NavGrid | None:
    """Build a private narrow-clearance view from retained real geometry.

    This is only for a camera-validated fallback after the normal body-clearance grid has no route.
    Closed leaves remain passable so the door pre-pass can open them; already-open leaves are rasterized
    at the smaller clearance. Static walls, furniture, and unsupported floor are never erased.
    """
    real_occ = getattr(grid, "raw_occ", None)
    if real_occ is None or real_occ.shape != grid.occ.shape:
        return None

    from scipy import ndimage

    real_occ = np.array(real_occ, dtype=bool, copy=True)
    if real_occ.any() and clearance_m > 0.0:
        dist_m = ndimage.distance_transform_edt(~real_occ) * grid.cell
        occ = real_occ | (dist_m < float(clearance_m))
    else:
        occ = real_occ
    # A few BEHAVIOR door frames ship collision meshes whose convex pieces span the
    # visual opening.  Release only the registered closed-leaf portal here.  This
    # view is never sufficient for acceptance: its route is subsequently checked
    # against the live full-body and camera collision queries, which still include
    # the frame, jambs, walls, and furniture.
    occ[_registered_door_portal_mask(grid, scene)] = False
    if grid.floor_support is not None:
        occ |= ~grid.floor_support

    import semantic_executor

    open_leaf_boxes = list(additional_open_leaf_boxes)
    for name in getattr(scene, "open_objs", set()) or set():
        entity = getattr(scene, "entities", {}).get(name)
        if entity is None or not getattr(entity, "leaf_paths", None):
            continue
        open_leaf_boxes.extend(semantic_executor._open_leaf_boxes(scene, entity))
    for _leaf_name, mn, mx in open_leaf_boxes:
        i0, j0 = grid.world_to_cell(
            float(mn[0]) - clearance_m,
            float(mn[1]) - clearance_m,
        )
        i1, j1 = grid.world_to_cell(
            float(mx[0]) + clearance_m,
            float(mx[1]) + clearance_m,
        )
        for j in range(max(0, min(j0, j1)), min(grid.ny - 1, max(j0, j1)) + 1):
            for i in range(max(0, min(i0, i1)), min(grid.nx - 1, max(i0, i1)) + 1):
                occ[j, i] = True

    import dataclasses

    return dataclasses.replace(
        grid,
        occ=occ,
        sdf=_two_sided_sdf(occ, grid.cell),
        base_occ=np.array(occ, dtype=bool, copy=True),
        open_leaf_count=None,
        inflate=float(clearance_m),
        transit_avoid=set(),
    )


def _two_sided_sdf(occ: np.ndarray, cell: float) -> np.ndarray:
    """Signed distance (m): +dist to nearest obstacle outside, -dist to nearest free inside."""
    from scipy import ndimage

    if not occ.any():
        return np.full(occ.shape, 10.0, dtype=np.float64)
    if occ.all():
        return np.full(occ.shape, -10.0, dtype=np.float64)
    d_out = ndimage.distance_transform_edt(~occ)
    d_in = ndimage.distance_transform_edt(occ)
    return (d_out - d_in).astype(np.float64) * cell


def _raster_walls(
    stage,
    wall_prims,
    x0: float,
    y0: float,
    cell: float,
    nx: int,
    ny: int,
    floor_z: float,
) -> np.ndarray:
    """Rasterize wall VISUAL-MESH triangles (in the human pass band) into an (ny, nx) footprint.

    Only triangles whose z-extent overlaps [floor+LO, floor+HI] contribute, so a door opening
    (clear in that band) stays open even though the lintel above it does not."""
    from pxr import Usd, UsdGeom

    occ = np.zeros((ny, nx), dtype=bool)
    batch = _TriBatch(occ, x0, y0, cell, nx, ny)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    z_lo = floor_z + WALL_BAND_LO
    z_hi = floor_z + WALL_BAND_HI
    for wp in wall_prims:
        root = stage.GetPrimAtPath(wp)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            fvc = mesh.GetFaceVertexCountsAttr().Get()
            fvi = mesh.GetFaceVertexIndicesAttr().Get()
            if pts is None or fvc is None or fvi is None or len(pts) == 0:
                continue
            mtx = xc.GetLocalToWorldTransform(prim)
            m = np.array(
                [[mtx[i][j] for j in range(4)] for i in range(4)], dtype=np.float64
            )
            pl = np.asarray(pts, dtype=np.float64)
            ph = np.concatenate([pl, np.ones((len(pl), 1))], axis=1)
            wpts = (ph @ m)[:, :3]  # USD row-vector convention: world = local * M
            batch.add_mesh(
                wpts,
                np.asarray(fvc, dtype=np.int64),
                np.asarray(fvi, dtype=np.int64),
                z_lo,
                z_hi,
            )
    batch.flush()
    return occ


def _raster_floor_support(
    stage,
    floor_prims,
    x0: float,
    y0: float,
    cell: float,
    nx: int,
    ny: int,
    floor_z: float,
) -> np.ndarray:
    """Rasterize the actual floor-top triangles into the only legal walking region."""
    from pxr import Usd, UsdGeom
    from scipy import ndimage

    support = np.zeros((ny, nx), dtype=bool)
    batch = _TriBatch(support, x0, y0, cell, nx, ny)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path in floor_prims:
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if points is None or counts is None or indices is None or len(points) == 0:
                continue
            matrix = cache.GetLocalToWorldTransform(prim)
            transform = np.array(
                [[matrix[i][j] for j in range(4)] for i in range(4)],
                dtype=np.float64,
            )
            local = np.asarray(points, dtype=np.float64)
            homogeneous = np.concatenate(
                [local, np.ones((len(local), 1), dtype=np.float64)], axis=1
            )
            world = (homogeneous @ transform)[:, :3]
            batch.add_mesh(
                world,
                np.asarray(counts, dtype=np.int64),
                np.asarray(indices, dtype=np.int64),
                floor_z - FLOOR_BAND_BELOW,
                floor_z + FLOOR_BAND_ABOVE,
            )
    batch.flush()
    closed = ndimage.binary_closing(support, structure=np.ones((3, 3), dtype=bool))
    return support | closed


def _raster_mesh_tris(wpts, fvc, fvi, occ, x0, y0, cell, nx, ny, z_lo, z_hi) -> None:
    """Rasterize ONE mesh's triangles into ``occ`` (single-mesh convenience wrapper around _TriBatch)."""
    batch = _TriBatch(occ, x0, y0, cell, nx, ny)
    batch.add_mesh(wpts, fvc, fvi, z_lo, z_hi)
    batch.flush()


def _raster_mesh_tris_py(wpts, fvc, fvi, occ, x0, y0, cell, nx, ny, z_lo, z_hi) -> None:
    """LEGACY per-triangle python rasterizer (kept as the ``NAV_GRID_LEGACY_RASTER=1`` reference)."""
    off = 0
    for c in fvc:
        c = int(c)
        for k in range(1, c - 1):
            a = int(fvi[off])
            b = int(fvi[off + k])
            d = int(fvi[off + k + 1])
            va, vb, vd = wpts[a], wpts[b], wpts[d]
            zmn = min(va[2], vb[2], vd[2])
            zmx = max(va[2], vb[2], vd[2])
            if zmx < z_lo or zmn > z_hi:  # lintel above / trim below the pass band
                continue
            _fill_tri_xy(
                va[0], va[1], vb[0], vb[1], vd[0], vd[1], occ, x0, y0, cell, nx, ny
            )
            # Edge raster too, so thin walls (a sliver that misses every cell center) still block.
            _dda(va[0], va[1], vb[0], vb[1], occ, x0, y0, cell, nx, ny)
            _dda(vb[0], vb[1], vd[0], vd[1], occ, x0, y0, cell, nx, ny)
            _dda(vd[0], vd[1], va[0], va[1], occ, x0, y0, cell, nx, ny)
        off += c


def _raster_furniture(
    stage, exclude_prefixes, x0, y0, cell, nx, ny, floor_z
) -> np.ndarray:
    """Rasterize FURNITURE collision-mesh TRIANGLES (in the human column band) into an (ny,nx) footprint,
    using the REAL collision geometry instead of the per-owner AABB. This is what stops nav over-blocking
    under-table space / L-concavities / gaps between adjacent pieces / angled furniture (the AABB path
    fills all of those). Walks the SAME CollisionAPI prims collidable_footprints does (minus excluded
    walls/human/obj/doors); a Mesh collider is rasterized triangle-by-triangle (convexHull colliders are
    low-poly -> cheap), a non-Mesh collider (Cube/Sphere primitive) falls back to its world-AABB
    rectangle. UNDILATED -- the caller Minkowski-inflates by the human radius."""
    from pxr import Usd, UsdGeom, UsdPhysics

    occ = np.zeros((ny, nx), dtype=bool)
    batch = _TriBatch(occ, x0, y0, cell, nx, ny)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    z_lo = floor_z + FURN_BAND_LO
    z_hi = floor_z + FURN_BAND_HI

    def _excluded(path: str) -> bool:
        # MUST stay in sync with init_validation.collidable_footprints._excluded. A prefix matches as a
        # PATH SEGMENT (``/World/Obj`` itself or ``/World/Obj/child``) AND as a NAME prefix
        # (``/World/Obj_<key>``): graspables live at ``/World/Obj_<key>`` (an UNDERSCORE, not a slash), and
        # without the ``_`` form they leak into THIS precise furniture raster -> occ_precise -> the stand
        # picker's ``occ`` gate wrongly treats the picked object + siblings as occupancy (a graspable on the
        # FLOOR then falsely blocks nearby stands). Furniture is ``/World/b_*`` / ``/World/<name>``, so
        # ``/World/Obj_`` / ``/World/Body_`` / ``/World/HumanProxy_`` only ever match the object/human.
        return any(
            path == p
            or path.startswith(p.rstrip("/") + "/")
            or path.startswith(p.rstrip("/") + "_")
            for p in exclude_prefixes
        )

    n_mesh = n_box = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if _excluded(str(prim.GetPath())):
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            fvc = mesh.GetFaceVertexCountsAttr().Get()
            fvi = mesh.GetFaceVertexIndicesAttr().Get()
            if pts is None or fvc is None or fvi is None or len(pts) == 0:
                continue
            mtx = xc.GetLocalToWorldTransform(prim)
            m = np.array(
                [[mtx[i][j] for j in range(4)] for i in range(4)], dtype=np.float64
            )
            pl = np.asarray(pts, dtype=np.float64)
            ph = np.concatenate([pl, np.ones((len(pl), 1))], axis=1)
            wpts = (ph @ m)[:, :3]
            batch.add_mesh(
                wpts,
                np.asarray(fvc, dtype=np.int64),
                np.asarray(fvi, dtype=np.int64),
                z_lo,
                z_hi,
            )
            n_mesh += 1
        else:
            rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            if not (np.isfinite(mn[0]) and np.isfinite(mx[0])):
                continue
            if float(mx[2]) < z_lo or float(mn[2]) > z_hi:
                continue
            i0 = max(0, int(math.floor((mn[0] - x0) / cell)))
            i1 = min(nx - 1, int(math.floor((mx[0] - x0) / cell)))
            j0 = max(0, int(math.floor((mn[1] - y0) / cell)))
            j1 = min(ny - 1, int(math.floor((mx[1] - y0) / cell)))
            if i1 >= i0 and j1 >= j0:
                occ[j0 : j1 + 1, i0 : i1 + 1] = True
            n_box += 1
    batch.flush()
    print(f"[nav_grid] precise furniture raster: {n_mesh} mesh + {n_box} box colliders")
    return occ


def _fill_tri_xy(ax, ay, bx, by, cx, cy, occ, x0, y0, cell, nx, ny) -> None:
    """LEGACY scalar half-plane triangle fill (the ``_TriBatch._fill`` reference)."""
    minx, maxx = min(ax, bx, cx), max(ax, bx, cx)
    miny, maxy = min(ay, by, cy), max(ay, by, cy)
    i0 = max(0, int(math.floor((minx - x0) / cell)))
    i1 = min(nx - 1, int(math.floor((maxx - x0) / cell)))
    j0 = max(0, int(math.floor((miny - y0) / cell)))
    j1 = min(ny - 1, int(math.floor((maxy - y0) / cell)))
    if i1 < i0 or j1 < j0:
        return
    for j in range(j0, j1 + 1):
        py = y0 + (j + 0.5) * cell
        for i in range(i0, i1 + 1):
            px = x0 + (i + 0.5) * cell
            w0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            w1 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
            w2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                occ[j, i] = True


def _dda(ax, ay, bx, by, occ, x0, y0, cell, nx, ny) -> None:
    """LEGACY scalar edge line-walk (the ``_TriBatch._edges`` reference)."""
    length = math.hypot(bx - ax, by - ay)
    n = max(1, int(length / (cell * 0.5)))
    for t in range(n + 1):
        f = t / n
        i = int(math.floor((ax + (bx - ax) * f - x0) / cell))
        j = int(math.floor((ay + (by - ay) * f - y0) / cell))
        if 0 <= i < nx and 0 <= j < ny:
            occ[j, i] = True


# ---------------------------------------------------------------------------------------------
# Vectorized triangle rasterizer (bit-identical replacement for the scalar kernels above)
# ---------------------------------------------------------------------------------------------
def _fan_triangles(fvc: np.ndarray, fvi: np.ndarray) -> np.ndarray:
    """(T, 3) vertex indices from a USD face-vertex stream, fan-triangulated EXACTLY as the scalar loop
    does: face f contributes (v0, vk, vk+1) for k in 1..count-2 (a count < 3 contributes nothing)."""
    fvc = np.asarray(fvc, dtype=np.int64)
    fvi = np.asarray(fvi, dtype=np.int64)
    if fvc.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if bool(np.all(fvc == 3)):  # the overwhelmingly common case (convex-hull colliders)
        return fvi[: 3 * fvc.size].reshape(-1, 3)
    ntri = np.maximum(fvc - 2, 0)
    total = int(ntri.sum())
    if total == 0:
        return np.empty((0, 3), dtype=np.int64)
    starts = np.cumsum(fvc) - fvc
    face, k = _expand(ntri)
    o = starts[face]
    k = k + 1
    return np.stack((fvi[o], fvi[o + k], fvi[o + k + 1]), axis=1)


def _expand(cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a ragged count vector: returns (owner index, index within owner) for sum(cnt) elements."""
    total = int(cnt.sum())
    owner = np.repeat(np.arange(cnt.size, dtype=np.int64), cnt)
    base = np.repeat(np.cumsum(cnt) - cnt, cnt)
    return owner, np.arange(total, dtype=np.int64) - base


def _chunks(cnt: np.ndarray, limit: int):
    """Yield (lo, hi) slices of ``cnt`` whose element sums stay under ``limit`` (a single item larger than
    ``limit`` gets a slice of its own), so the flattened expansion never blows up memory."""
    ends = np.cumsum(cnt)
    lo, n = 0, int(cnt.size)
    while lo < n:
        base = int(ends[lo - 1]) if lo > 0 else 0
        hi = int(np.searchsorted(ends, base + limit, side="right"))
        yield lo, max(hi, lo + 1)
        lo = max(hi, lo + 1)


def _dda_steps(dx: np.ndarray, dy: np.ndarray, cell: float) -> np.ndarray:
    """``max(1, int(hypot(dx, dy) / (cell * 0.5)))`` per edge, matching ``math.hypot`` EXACTLY.

    ``np.hypot`` and ``math.hypot`` disagree in the last ulp for ~13% of inputs, and this step count sets
    every sample position on the edge, so a single differing edge would move cells. A 1-ulp difference can
    only change the truncation when the quotient sits on an integer, so any edge within 1e-9 (relative,
    ~4e6 ulps of head-room) of one is recomputed with ``math.hypot`` itself. Below 1 the ``max(1, ...)``
    makes the exact value irrelevant."""
    half = cell * 0.5
    q = np.hypot(dx, dy) / half
    n = q.astype(np.int64)
    risky = (q >= 1.0) & (np.abs(q - np.rint(q)) <= 1e-9 * q)
    if bool(risky.any()):
        idx = np.flatnonzero(risky)
        exact = np.fromiter(
            map(math.hypot, dx[idx].tolist(), dy[idx].tolist()),
            dtype=np.float64,
            count=idx.size,
        )
        n[idx] = (exact / half).astype(np.int64)
    return np.maximum(n, 1)


class _TriBatch:
    """Buffers world-space triangles across meshes and rasterizes them into ``occ`` in vectorized batches.

    BIT-IDENTICAL to the scalar path (``_raster_mesh_tris_py``): every float64 expression is evaluated in
    the same order with the same operands, so each intermediate rounds identically, and ``occ`` is only
    ever OR-ed to True -- so batching across meshes/triangles cannot change the result."""

    def __init__(self, occ, x0, y0, cell, nx, ny) -> None:
        self.occ = occ
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        self.nx, self.ny = int(nx), int(ny)
        self._buf: list[np.ndarray] = []
        self._n = 0
        self._legacy = _legacy("RASTER")

    def _scalar(self, wpts, fvc, fvi, z_lo, z_hi) -> None:
        _raster_mesh_tris_py(
            wpts,
            fvc,
            fvi,
            self.occ,
            self.x0,
            self.y0,
            self.cell,
            self.nx,
            self.ny,
            z_lo,
            z_hi,
        )

    def add_mesh(self, wpts, fvc, fvi, z_lo, z_hi) -> None:
        """Fan-triangulate ONE mesh, drop triangles outside the [z_lo, z_hi] pass band, buffer the rest."""
        if self._legacy:
            self._scalar(wpts, fvc, fvi, z_lo, z_hi)
            return
        tri = _fan_triangles(fvc, fvi)
        if tri.size == 0:
            return
        v = np.asarray(wpts, dtype=np.float64)[tri]  # (T, 3 verts, 3 coords)
        z = v[:, :, 2]
        keep = ~((z.max(axis=1) < z_lo) | (z.min(axis=1) > z_hi))
        v = v[keep]
        if v.shape[0] == 0:
            return
        if not bool(np.isfinite(v).all()):
            # A NaN/inf vertex makes the scalar path raise out of int(math.floor(...)). Hand the whole mesh
            # back to it rather than let the numpy cast turn that crash into silently garbage cells: nav
            # must fail loudly on broken geometry, not quietly block the wrong square metre.
            self._scalar(wpts, fvc, fvi, z_lo, z_hi)
            return
        v = v[:, :, :2].copy()
        self._buf.append(v)
        self._n += int(v.shape[0])
        if self._n >= _TRI_BATCH:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        v = np.concatenate(self._buf, axis=0)
        self._buf, self._n = [], 0
        ax, ay = v[:, 0, 0], v[:, 0, 1]
        bx, by = v[:, 1, 0], v[:, 1, 1]
        cx, cy = v[:, 2, 0], v[:, 2, 1]
        self._fill(ax, ay, bx, by, cx, cy)
        # Edge raster too, so thin walls (a sliver that misses every cell center) still block.
        self._edges(ax, ay, bx, by)
        self._edges(bx, by, cx, cy)
        self._edges(cx, cy, ax, ay)

    def _cell_range(self, lo_w, hi_w, origin, n_cells):
        """Clipped inclusive cell index range of a world-coordinate span, floored exactly as the scalar
        ``max(0, int(math.floor(...)))`` / ``min(n-1, int(math.floor(...)))`` pair does."""
        i0 = np.maximum(0, np.floor((lo_w - origin) / self.cell).astype(np.int64))
        i1 = np.minimum(
            n_cells - 1, np.floor((hi_w - origin) / self.cell).astype(np.int64)
        )
        return i0, i1

    def _fill(self, ax, ay, bx, by, cx, cy) -> None:
        """Half-plane test over each triangle's clipped integer bbox (vectorized ``_fill_tri_xy``)."""
        i0, i1 = self._cell_range(
            np.minimum(np.minimum(ax, bx), cx),
            np.maximum(np.maximum(ax, bx), cx),
            self.x0,
            self.nx,
        )
        j0, j1 = self._cell_range(
            np.minimum(np.minimum(ay, by), cy),
            np.maximum(np.maximum(ay, by), cy),
            self.y0,
            self.ny,
        )
        w, h = i1 - i0 + 1, j1 - j0 + 1
        sel = np.flatnonzero((w > 0) & (h > 0))
        if sel.size == 0:
            return
        i0, j0, w, h = i0[sel], j0[sel], w[sel], h[sel]
        tri = (ax[sel], ay[sel], bx[sel], by[sel], cx[sel], cy[sel])
        cnt = w * h
        for lo, hi in _chunks(cnt, _RASTER_CHUNK):
            t, loc = _expand(cnt[lo:hi])
            ii = i0[lo:hi][t] + loc % w[lo:hi][t]
            jj = j0[lo:hi][t] + loc // w[lo:hi][t]
            self._half_plane(ii, jj, [c[lo:hi][t] for c in tri])

    def _half_plane(self, ii, jj, tri) -> None:
        tax, tay, tbx, tby, tcx, tcy = tri
        px = self.x0 + (ii + 0.5) * self.cell
        py = self.y0 + (jj + 0.5) * self.cell
        w0 = (tbx - tax) * (py - tay) - (tby - tay) * (px - tax)
        w1 = (tcx - tbx) * (py - tby) - (tcy - tby) * (px - tbx)
        w2 = (tax - tcx) * (py - tcy) - (tay - tcy) * (px - tcx)
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | (
            (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
        )
        self.occ[jj[inside], ii[inside]] = True

    def _edges(self, ax, ay, bx, by) -> None:
        """Walk every edge at half-cell steps (vectorized ``_dda``)."""
        dx, dy = bx - ax, by - ay
        n = _dda_steps(dx, dy, self.cell)
        npts = n + 1
        for lo, hi in _chunks(npts, _RASTER_CHUNK):
            e, t = _expand(npts[lo:hi])
            f = t / n[lo:hi][e]
            xw = ax[lo:hi][e] + dx[lo:hi][e] * f
            yw = ay[lo:hi][e] + dy[lo:hi][e] * f
            ii = np.floor((xw - self.x0) / self.cell).astype(np.int64)
            jj = np.floor((yw - self.y0) / self.cell).astype(np.int64)
            m = (ii >= 0) & (ii < self.nx) & (jj >= 0) & (jj < self.ny)
            self.occ[jj[m], ii[m]] = True


# ---------------------------------------------------------------------------------------------
# Connectivity (doorway proof; the A* planner lives in nav_planner.py)
# ---------------------------------------------------------------------------------------------
def flood_free(grid: NavGrid, start_ij: tuple[int, int]) -> np.ndarray:
    """8-connected flood over free cells from start_ij (snapped to nearest free). Returns a mask.

    Runs as a single connected-component labelling instead of a python BFS. That is EXACT, not an
    approximation: the BFS refuses a diagonal step whose two orthogonal neighbours are not both free, and
    when they ARE both free the same two cells already form a 4-connected detour -- so every diagonal edge
    is redundant and the BFS's components are precisely the 4-connected components."""
    si, sj = start_ij
    if not grid.is_free(si, sj):
        snap = grid.nearest_free(si, sj)
        if snap is None:
            return np.zeros((grid.ny, grid.nx), dtype=bool)
        si, sj = snap
    if _legacy("FLOOD"):
        return _flood_free_bfs(grid, si, sj)
    from scipy import ndimage

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    # bool cast, not `~grid.occ`: the BFS gated on truthiness, and a non-bool occ would bit-invert instead.
    free = ~np.asarray(grid.occ, dtype=bool)
    for i, j in grid.transit_avoid:
        if grid.in_bounds(i, j):
            free[j, i] = False
    lbl, _n = ndimage.label(free, structure=cross)
    return lbl == lbl[sj, si]


def _flood_free_bfs(grid: NavGrid, si: int, sj: int) -> np.ndarray:
    """LEGACY pure-python BFS (the ``NAV_GRID_LEGACY_FLOOD=1`` reference); seed already snapped free."""
    reach = np.zeros((grid.ny, grid.nx), dtype=bool)
    reach[sj, si] = True
    dq = deque([(si, sj)])
    nbr = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while dq:
        i, j = dq.popleft()
        for di, dj in nbr:
            ii, jj = i + di, j + dj
            if not (0 <= ii < grid.nx and 0 <= jj < grid.ny):
                continue
            if not grid.is_free(ii, jj) or reach[jj, ii]:
                continue
            # No diagonal corner-cut through a blocked orthogonal cell -- MUST match the A* planner's
            # reachability, else connected() can report a 1-cell diagonal squeeze the planner then refuses
            # to route (connected=True but plan_path=PATH_BLOCKED).
            if (
                di != 0
                and dj != 0
                and (not grid.is_free(i + di, j) or not grid.is_free(i, j + dj))
            ):
                continue
            reach[jj, ii] = True
            dq.append((ii, jj))
    return reach


def connected(
    grid: NavGrid, a_xy: tuple[float, float], b_xy: tuple[float, float]
) -> bool:
    """True iff b_xy's (nearest-free) cell is reachable from a_xy's through free space."""
    reach = flood_free(grid, grid.world_to_cell(*a_xy))
    bi, bj = grid.world_to_cell(*b_xy)
    if not grid.is_free(bi, bj):
        snap = grid.nearest_free(bi, bj)
        if snap is None:
            return False
        bi, bj = snap
    return bool(reach[bj, bi])


def carve_free(grid: NavGrid, pts_xy, radius: float = 0.30) -> None:
    """Mark a disk of cells around each world point FREE, then rebuild the SDF.

    Used for the init-gate-validated human start + stand poses: those are proven non-penetrating human
    positions, but they sit at furniture edges that the coarse bounding-cube + body-radius inflation
    marks blocked. Carving reconciles the grid with the (finer) init gate so the planner can leave the
    start and reach each stand."""
    r = max(1, int(round(radius / grid.cell)))
    changed = False
    for x, y in pts_xy:
        ci, cj = grid.world_to_cell(x, y)
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if di * di + dj * dj > r * r:
                    continue
                i, j = ci + di, cj + dj
                if not grid.in_bounds(i, j):
                    continue
                supported = grid.floor_support is None or bool(grid.floor_support[j, i])
                if supported and grid.occ[j, i]:
                    grid.occ[j, i] = False
                    if grid.base_occ is not None and (
                        grid.door_id is None or int(grid.door_id[j, i]) < 0
                    ):
                        grid.base_occ[j, i] = False
                    changed = True
    if changed:
        grid.sdf = _two_sided_sdf(grid.occ, grid.cell)


def free_stats(grid: NavGrid) -> tuple[int, float]:
    """(#connected free components, largest-component fraction of all free cells), 8-connectivity.

    Doorways-open sanity: a house with open doorways is ONE dominant component (largest ~= 1.0);
    walls sealing doorways would split it into many small per-room components."""
    from scipy import ndimage

    struct = np.ones((3, 3), dtype=bool)
    lbl, n = ndimage.label(~grid.occ, structure=struct)
    if n == 0:
        return (0, 0.0)
    counts = np.bincount(lbl.ravel())[1:]  # drop background label 0
    total = float(counts.sum())
    return (int(n), float(counts.max()) / max(1.0, total))


# ---------------------------------------------------------------------------------------------
# Debug dump (PNG for visual QA + ascii fallback)
# ---------------------------------------------------------------------------------------------
def dump_grid(
    grid: NavGrid,
    out_path: str,
    start_xy=None,
    goals=None,
    path_xy=None,
    reachable=None,
) -> None:
    _dump_ascii(grid, out_path + ".txt", start_xy, goals, path_xy, reachable)
    try:
        _dump_png(grid, out_path + ".png", start_xy, goals, path_xy, reachable)
    except Exception as exc:  # noqa: BLE001  (PNG is best-effort; ascii is the guaranteed artifact)
        print(f"[nav_grid] PNG dump skipped ({exc}); ascii written to {out_path}.txt")


def _dump_png(grid, path, start_xy, goals, path_xy, reachable) -> None:
    from PIL import Image

    ny, nx = grid.occ.shape
    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[~grid.occ] = (235, 235, 235)
    img[grid.occ] = (40, 40, 40)
    if reachable is not None:
        img[reachable & (~grid.occ)] = (170, 210, 255)

    def mark(xy, color, rad=1):
        i, j = grid.world_to_cell(*xy)
        for dj in range(-rad, rad + 1):
            for di in range(-rad, rad + 1):
                if grid.in_bounds(i + di, j + dj):
                    img[j + dj, i + di] = color

    if path_xy:
        for xy in path_xy:
            mark(xy, (60, 120, 255), 0)
    if goals:
        for xy in goals:
            mark(xy, (230, 60, 60), 1)
    if start_xy is not None:
        mark(start_xy, (60, 200, 60), 1)

    scale = max(1, int(700 / max(nx, ny)))
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    img = img[::-1]  # flip so +y is up
    Image.fromarray(img, "RGB").save(path)


def _dump_ascii(grid, path, start_xy, goals, path_xy, reachable) -> None:
    ny, nx = grid.occ.shape
    step = max(1, int(math.ceil(max(nx, ny) / 110)))
    rows = list(range(ny - 1, -1, -step))
    cols = list(range(0, nx, step))
    chars = []
    for j in rows:
        line = []
        for i in cols:
            if grid.occ[j, i]:
                line.append("#")
            elif reachable is not None and reachable[j, i]:
                line.append(":")
            else:
                line.append(".")
        chars.append(line)

    def put(xy, ch):
        i, j = grid.world_to_cell(*xy)
        if not grid.in_bounds(i, j):
            return
        r = (ny - 1 - j) // step
        c = i // step
        if 0 <= r < len(chars) and 0 <= c < len(chars[0]):
            chars[r][c] = ch

    if path_xy:
        for xy in path_xy:
            put(xy, "+")
    if goals:
        for xy in goals:
            put(xy, "G")
    if start_xy is not None:
        put(start_xy, "S")

    with open(path, "w") as f:
        f.write(
            f"# NavGrid {nx}x{ny} cell={grid.cell} origin=({grid.x0:.2f},{grid.y0:.2f}) "
            f"downsample={step}\n"
            f"# '#'=blocked '.'=free ':'=reachable-from-start 'S'=start 'G'=goal '+'=path; top=+y\n"
        )
        for line in chars:
            f.write("".join(line) + "\n")


# ---------------------------------------------------------------------------------------------
# Offline self-test (pure python: numpy+scipy only, NO Isaac boot)
# ---------------------------------------------------------------------------------------------
def _selftest() -> int:
    cell = 0.15
    nx = ny = int(round(4.0 / cell))
    occ = np.zeros((ny, nx), dtype=bool)
    jmid = ny // 2
    occ[jmid, :] = True  # a wall spanning x, at mid-y
    gi0, gi1 = int(1.5 / cell), int(2.5 / cell)
    occ[jmid, gi0:gi1] = False  # a 1 m door opening
    grid = NavGrid(0.0, 0.0, cell, nx, ny, occ, _two_sided_sdf(occ, cell))
    a, b = (2.0, 0.4), (2.0, 3.6)  # opposite sides, aligned with the door
    ok_door = connected(grid, a, b)

    sealed = occ.copy()
    sealed[jmid, gi0:gi1] = True
    grid2 = NavGrid(0.0, 0.0, cell, nx, ny, sealed, _two_sided_sdf(sealed, cell))
    ok_sealed = not connected(grid2, a, b)

    s_free = grid.sdf_at(2.0, 0.4)
    s_wall = grid.sdf_at(0.2, jmid * cell + 0.5 * cell)  # on the solid part of the wall
    gx, gy = grid.sdf_grad(
        0.2, jmid * cell - 3 * cell
    )  # just below the wall -> grad points -y

    # Door layer: CLOSE the opening (block it in occ) but LABEL those cells as a door. The runtime occ
    # then blocks A*/reachability (hard backstop), while passable_view frees the doorway so it reconnects.
    docc = occ.copy()
    docc[jmid, gi0:gi1] = True  # closed door fills the opening
    did = np.full((ny, nx), -1, dtype=np.int32)
    dcost = np.zeros((ny, nx), dtype=np.float64)
    did[jmid, gi0:gi1] = 0
    dcost[jmid, gi0:gi1] = DOOR_COST_M
    dgrid = NavGrid(
        0.0,
        0.0,
        cell,
        nx,
        ny,
        docc,
        _two_sided_sdf(docc, cell),
        door_id=did,
        door_cost=dcost,
    )
    ok_closed_blocks = not connected(dgrid, a, b)
    ok_open_passable = connected(passable_view(dgrid), a, b)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fpv_outputs")
    try:
        os.makedirs(out, exist_ok=True)
        dump_grid(
            grid,
            os.path.join(out, "navgrid_selftest"),
            start_xy=a,
            goals=[b],
            reachable=flood_free(grid, grid.world_to_cell(*a)),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[nav_grid] selftest dump skipped ({exc})")
    ok = (
        ok_door
        and ok_sealed
        and ok_closed_blocks
        and ok_open_passable
        and s_free > 0
        and s_wall < 0
        and abs(gy) > abs(gx)
    )
    print(
        f"[nav_grid] selftest door_connected={ok_door} sealed_blocked={ok_sealed} "
        f"closed_door_blocks={ok_closed_blocks} passable_open={ok_open_passable} "
        f"sdf_free={s_free:.2f} sdf_wall={s_wall:.2f} grad=({gx:.2f},{gy:.2f}) -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

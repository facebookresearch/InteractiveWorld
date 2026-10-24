"""INIT human-vs-scene non-penetration validation (P0-P4): guarantee a KINEMATIC SMPL+H human is
placed at a NON-PENETRATING start pose + a non-penetrating stand pose at every action target, for
RL-scale RANDOM scenes.

This is the additive placement/validation layer that sits BETWEEN scene_state (the candidate
generator: start_xy + stand_pose_for) and the render/executor. It owns NOTHING per-scene: every
footprint comes from a prim's own world bbox and every pose from the body skeleton. It reuses
``physics_human`` (the proxy rig + the depth-bearing human-vs-scene gate) and ``object_physics``
(the reject-resample skeleton + the shared render=False SimulationContext) VERBATIM.

The strategy (from the 12-agent converge; see [[isaac-arbitrary-motion-architecture]]):

  1. GATE (repeated per-pose) = ``physics_human.human_scene_overlap`` -- a lag-free BOOLEAN scene-query
     (``overlap_sphere`` sampled along the proxy capsules straight from the joints ``J``: no
     ``sim.step`` / USD-write / reset), so it is EXACT + deterministic under repeated re-posing. The
     depth-bearing ``physics_human.human_scene_separation`` LAGS several frames when a kinematic proxy
     is re-posed, so it is kept ONLY for the P0 positive control; the bare
     ``human_scene_collision_pairs`` is a silent false-negative for a KINEMATIC human vs STATIC furniture.
  2. FREE-CELL START (``place_start``): a 2D occupancy grid from EVERY collidable prim's world-bbox
     XY footprint, Minkowski-inflated by the human footprint radius; sample the most-open free cell,
     then ONE exact confirm. Clean by construction for the footprint class + confirmed for overhangs.
  3. STAND POSES (``validate_stand_pose``): keep ``scene_state.stand_pose_for`` as the candidate
     GENERATOR, then gate + ADJUST: back off further along the same approach axis (stepped), then
     re-derive at rotated approach_yaw (+-45/90/180), each exact-gated, capped by the reach envelope.
     Confirm the ARM-EXTENDED stand keyframe separately from the body-CORE idle pose.
  4. FAILURE + RL CONTRACT: bounded caps; on exhaustion accept the LEAST-penetrating candidate AND
     set ``episode_valid=False`` with residual pairs + worst depth. NEVER accept/render a placement
     that failed the gate without ``episode_valid=False``.

Design (matching ``physics_human.py`` / ``object_physics.py``): ``from __future__ import annotations``
+ LAZY pxr/omni/isaacsim imports inside every function, so importing this module before
``launch_app()`` never pulls in Kit. torch-free at module scope (smplh_anim_core is imported lazily
inside the pose builders, and only when a live validation actually runs).

    bash hand/run_isaac.sh simulator.physical_world/init_validation.py --positive-control   # P0 gate proof (live)
    bash hand/run_isaac.sh simulator.physical_world/episode_runtime.py --scene clutter --initcheck  # V1/V2 (live)
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd  # noqa: F401
    from scene_state import SceneState

LOG = logging.getLogger("init_validation")

# ---- Human footprint + reach envelope (derive-once from the body; general, no per-scene tuning). ----
HUMAN_FOOT_R = (
    0.30  # Minkowski inflation radius (m): the human's standing footprint radius
)
# Nav-grid TRANSIT inflation (m): body clearance for WALKING only, applied as a REAL-valued distance-
# transform (not cell-quantised integer dilation). Stand-SELECTION deliberately uses NO inflation -- the
# human may stand right up against furniture to reach an object on it, and whether the body actually clips
# is decided by the precise idle/arm overlap gate, not by this radius. Only the walk PATH keeps this margin
# (the runtime nudge backstops the remainder), so "can stand here" and "can walk here" read one precise map.
NAV_INFLATE_M = 0.10
# Extra distance (m), BEYOND the nav grid's own inflation, allowed between a validated stand and the
# nearest start-reachable walkable cell -- the final "shuffle" off the planned route onto the stand. It must
# be measured relative to the grid's inflation, not fixed: a stand may hug furniture (that is the whole
# point of stand selection using NO inflation), so its distance to the walkable set is ~the inflation
# itself, and a fixed tolerance rejects every furniture-hugging stand as soon as the inflation ladder picks
# a rung above it. Keeping the slack constant also keeps the SAFETY property constant: the thickest
# obstacle this shuffle can hop is exactly the slack (the far-side cell is itself >= inflation from the
# obstacle), independent of which rung the ladder chose.
STAND_SHUFFLE_SLACK = 0.15
# The reach envelope caps how far a stand-pose back-off may go: a pose that clears the gate but puts
# the target beyond arm reach is REJECTED. ~0.75 m horizontal is a comfortable SMPL+H arm reach with
# the shared forward spine-lean; the geometry-derived stand-back already sits ~0.42-0.65 m out, so this
# leaves ~0.1-0.3 m of extra back-off headroom before a reach becomes impossible.
REACH_ENVELOPE_M = 0.78

# ---- Bounded caps (P4; mirror object_physics.plan_spawn_layout + OmniGibson last-ditch). The START
# free-cell draw IS the resample (each draw is a fresh free cell; ~12 covers the resample budget); the
# STAND ladder is the back-off x approach-side budget. The de-pen MTV push-out the debate proposed
# needs a contact NORMAL (depth-bearing gate); we use the lag-free BOOLEAN overlap gate for the
# repeated per-pose queries (determinism > depth -- see physics_human.human_scene_overlap), so a dirty
# candidate is handled by drawing the next free cell / next ladder rung rather than a normal-driven
# push, and on exhaustion the least-penetrating candidate is accepted + episode_valid=False.
MAX_START_DRAWS = 12  # free-cell start draws (== the resample budget) before the least-penetrating fallback
MAX_STAND_LADDER = 8  # back-off x approach-side rungs per stand pose
MAX_ROUTER_RETRY_CANDIDATES = 8
# Once a render-acceptable contact stand exists, inspect at most this many later candidates for a fully
# clean alternative. Each stand's reach rungs are already one GPU batch and the phase budget is still a
# hard cap, so this adds a bounded handful of batches rather than restoring an exhaustive zero-contact
# search on cramped scenes.
MAX_CLEAN_SEARCH_EXTRA_CANDIDATES = 3
PRIMARY_STAND_TRAJECTORY_PROBES = 3
MAX_COARSE_STAND_TRAJECTORY_PROBES = 6
MAX_STAND_TRAJECTORY_PROBES = 12
MAX_ROUTER_STAND_TRAJECTORY_PROBES = 8
_STAND_TRAJECTORY_SAMPLE_M = 0.45
_STAND_TRAJECTORY_MAX_SAMPLES = 24
_CAMERA_CLEAR_RADIUS_M = 0.03
_CAMERA_ROUTE_REPLAN_ATTEMPTS = 4

_GATE_TOL_M = (
    0.002  # |separation| below this is not a penetration (matches the object gate tol)
)

# Radius (m) of the intended-CONTACT sphere subtracted from the ARM-vs-furniture gate at a stand pose:
# the reaching hand + fingers legitimately enter the target furniture AT the handle/contact, so arm
# capsule samples within this sphere of the contact point are NOT violations. An arm reaching THROUGH
# the cabinet body (samples OUTSIDE this sphere overlapping furniture) IS surfaced as episode_valid=False.
# Sized to cover the hand/finger cluster around a handle (~a fist + a little slack) without masking a
# forearm/upper-arm that plows through the furniture elsewhere.
_CONTACT_SPHERE_R = 0.22
_BODY_GRAZE_TOL_M = 0.02
_CENTERED_OPENABLE_LATERAL_M = 0.30
_CENTERED_OPENABLE_MIN_OUTSIDE_M = 0.08
_CENTERED_OPENABLE_DEPTH_SLACK_M = 0.12
_NAV_DOOR_STANDOFF_M = 0.90


# =====================================================================================================
# Occupancy grid (P2): 2D free-space from EVERY collidable prim's world-bbox XY footprint.
# =====================================================================================================
@dataclass
class OccupancyGrid:
    """A 2D XY occupancy grid over the scene floor. ``occ[j, i]`` True = cell (i=x, j=y) is blocked
    (a collidable footprint, Minkowski-inflated by the human radius, overlaps it). Cell centers are
    world XY. ``clearance`` is the signed distance (m) from each cell center to the nearest blocked
    footprint edge (>0 = free, the larger the more open) -- drives "most-open free cell" sampling."""

    x0: float
    y0: float
    cell: float
    nx: int
    ny: int
    occ: np.ndarray  # (ny, nx) bool
    clearance: np.ndarray  # (ny, nx) float (m); -inf where blocked

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        return (self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell)

    @property
    def n_free(self) -> int:
        return int((~self.occ).sum())


def collidable_footprints(
    stage: "Usd.Stage",
    exclude_prefixes: tuple[str, ...] = (
        "/World/HumanProxy",
        "/World/Body",
        "/World/Obj",
    ),
    group_by_owner: bool = True,
) -> list[tuple[str, float, float, float, float, float, float]]:
    """Every collidable prim's world-bbox footprint: ``[(owner, cx, cy, hx, hy, bottom_z, top_z), ...]``.

    General (BEHAVIOR-1K-ready, no per-object magic): walk the stage, collect prims carrying a
    ``UsdPhysics.CollisionAPI``, group each collider under its nearest ANCESTOR that also carries a
    collider OR a RigidBodyAPI (so a referenced desk's child-mesh colliders roll up to one desk
    footprint), and union the child world bboxes into one XY footprint + z-range per owner. Prims under
    ``exclude_prefixes`` (the human proxy, the skinned body, the carried object) are skipped -- they are
    NOT scene obstacles. ``solid_boxes`` (scene_state) only covers foot_half entities (empty for
    graspables/seats/BEHAVIOR-1K), so this world-bbox scan is what makes the grid + gate COMPLETE. The
    z-range (``bottom_z``, ``top_z``) lets callers keep only footprints that intersect the human's
    standing column -- a CEILING/overhead lamp has a room-wide XY footprint but sits at z~4 m, so it
    must NOT block the floor grid (a standing human never reaches it)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    # ONE BBoxCache reused for ALL prims. fpv_scene.compute_world_bbox builds a FRESH cache per call, so
    # calling it once per collider is O(n) cache setups -- minutes for an instanced env's thousands of
    # colliders (Office: 3646). Reusing a single cache amortizes it to seconds.
    _bbc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        # Include 'guide': OG's per-link collision meshes (the ONLY colliders on a BEHAVIOR object once
        # behavior_collider stops tagging the visual mesh) are authored purpose='guide'. Without 'guide'
        # here they compute an EMPTY world bound and every BEHAVIOR furniture piece would silently VANISH
        # from the init non-pen gate + occupancy grid + nav. Isaac-native colliders are default/render
        # purpose, so adding 'guide' leaves their footprints byte-identical.
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )

    def _excluded(path: str) -> bool:
        # Match a prefix as a PATH SEGMENT (``/World/Obj`` itself or ``/World/Obj/child``) AND as a NAME
        # prefix (``/World/Obj_<key>``): build_behavior_task places each graspable at ``/World/Obj_<key>``
        # (an UNDERSCORE, not a slash), so the segment-only match missed them and the picked object + EVERY
        # sibling graspable were wrongly counted as floor obstacles (a stand next to them got rejected --
        # same class as the light-collider bug). Graspables are NOT scene obstacles (scene_state marks
        # is_obstacle=False), so exclude the ``_`` form too. Furniture is ``/World/b_*`` / ``/World/<name>``,
        # so ``/World/Obj_`` / ``/World/Body_`` / ``/World/HumanProxy_`` only ever match the object/human.
        return any(
            path == p
            or path.startswith(p.rstrip("/") + "/")
            or path.startswith(p.rstrip("/") + "_")
            for p in exclude_prefixes
        )

    # Owner = the highest ancestor of a collider that itself is a rigid body or a collider (else the
    # collider prim itself). This rolls child-mesh colliders up to their referenced-asset root.
    def _owner(prim) -> str:
        best = prim
        p = prim.GetParent()
        while p and p.IsValid() and p.GetPath() != p.GetPath().GetParentPath():
            if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI):
                best = p
            p = p.GetParent()
        return str(best.GetPath())

    groups: dict[str, list] = {}
    # TraverseInstanceProxies: INSTANCED envs (Office = 3413 instances) carry their colliders INSIDE
    # instanceable prototypes, which a plain PrimRange stops at -> it would see only the 1 non-instanced
    # ground plane and miss every wall/desk. Walking instance proxies exposes each instance's colliders
    # as (read-only) proxy prims, so the grid + overlap gate see the real furniture. Non-instanced envs
    # (warehouse) are unaffected (no proxies to add).
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if _excluded(path):
            continue
        rng = _bbc.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        if not (np.isfinite(mn[0]) and np.isfinite(mx[0])):
            continue
        group = _owner(prim) if group_by_owner else path
        groups.setdefault(group, []).append(
            (
                float(mn[0]),
                float(mn[1]),
                float(mn[2]),
                float(mx[0]),
                float(mx[1]),
                float(mx[2]),
            )
        )

    out: list[tuple[str, float, float, float, float, float, float]] = []
    for owner, boxes in groups.items():
        xs0 = min(b[0] for b in boxes)
        ys0 = min(b[1] for b in boxes)
        zs0 = min(b[2] for b in boxes)
        xs1 = max(b[3] for b in boxes)
        ys1 = max(b[4] for b in boxes)
        zs1 = max(b[5] for b in boxes)
        out.append(
            (
                owner,
                0.5 * (xs0 + xs1),
                0.5 * (ys0 + ys1),
                0.5 * (xs1 - xs0),
                0.5 * (ys1 - ys0),
                zs0,
                zs1,
            )
        )
    out.sort()
    return out


# A standing SMPL+H human is ~1.9 m tall. A collidable footprint blocks the floor only if its z-range
# overlaps the human's standing column [floor_z, floor_z + HUMAN_HEIGHT]; a ceiling/overhead lamp at
# z~4 m has a room-wide footprint but must NOT wall off the floor grid.
HUMAN_HEIGHT = 1.95


def _blocks_floor(
    bottom_z: float, top_z: float, floor_z: float, floor_top_max: float
) -> bool:
    """True if a footprint z-range [bottom_z, top_z] is a real floor obstacle for a standing human: it
    must rise above the floor slab (top > floor_top_max) AND its BOTTOM must be within the human column
    (bottom < floor_z + HUMAN_HEIGHT) -- else it is an overhead structure the human passes under."""
    return top_z > floor_top_max and bottom_z < floor_z + HUMAN_HEIGHT


def build_occupancy_grid(
    footprints: list[tuple[str, float, float, float, float, float, float]],
    region: tuple[float, float, float, float],
    cell: float = 0.15,
    inflate: float = HUMAN_FOOT_R,
    floor_z: float = 0.0,
    floor_top_max: float = 0.15,
) -> OccupancyGrid:
    """Rasterize ``footprints`` [(owner, cx, cy, hx, hy, bottom_z, top_z)] into an OccupancyGrid over
    ``region`` ``(xmin, xmax, ymin, ymax)``. Each floor-blocking footprint is Minkowski-inflated by
    ``inflate`` (the human radius) on each side -- a cell whose CENTER falls inside an inflated footprint
    is blocked. This guarantees a vertical cylinder of radius ``inflate`` over any free cell clears every
    obstacle footprint (the footprint class). ``clearance[j,i]`` = signed distance from the cell center
    to the nearest inflated-footprint edge (>0 free). Only footprints that ``_blocks_floor`` (rise above
    the floor slab AND whose bottom is within the human column) count -- the floor slab AND overhead
    ceilings/lamps are IGNORED (a standing human passes under a z~4 m ceiling)."""
    xmin, xmax, ymin, ymax = (float(v) for v in region)
    nx = max(1, int(math.ceil((xmax - xmin) / cell)))
    ny = max(1, int(math.ceil((ymax - ymin) / cell)))
    occ = np.zeros((ny, nx), dtype=bool)

    obstacles = [
        (cx, cy, hx, hy)
        for _owner, cx, cy, hx, hy, bot, top in footprints
        if _blocks_floor(bot, top, floor_z, floor_top_max)
    ]
    # Rasterize each Minkowski-inflated footprint by SLICE ASSIGNMENT -- O(obstacles + covered cells).
    # (The old full-grid op PER obstacle was O(obstacles * all cells) = minutes for an instanced env's
    # thousands of footprints, e.g. Office's ~3.3k.) A cell whose CENTER falls inside an inflated box is
    # blocked, so a vertical cylinder of radius ``inflate`` over any free cell clears every footprint.
    for cx, cy, hx, hy in obstacles:
        i0 = max(0, min(nx, int(math.floor((cx - hx - inflate - xmin) / cell))))
        i1 = max(0, min(nx, int(math.ceil((cx + hx + inflate - xmin) / cell))))
        j0 = max(0, min(ny, int(math.floor((cy - hy - inflate - ymin) / cell))))
        j1 = max(0, min(ny, int(math.ceil((cy + hy + inflate - ymin) / cell))))
        if i1 > i0 and j1 > j0:
            occ[j0:j1, i0:i1] = True

    # clearance[j,i] = distance (m) from a FREE cell center to the nearest inflated-obstacle cell (one
    # distance transform over the whole grid). Occupied cells -> -inf (place_start ignores them).
    from scipy import ndimage

    clearance = ndimage.distance_transform_edt(~occ).astype(np.float64) * cell
    clearance[occ] = -np.inf
    return OccupancyGrid(xmin, ymin, cell, nx, ny, occ, clearance)


# =====================================================================================================
# Proxy pose builder: world joints J at a candidate (root_xy, yaw) for a chosen arm keyframe.
# =====================================================================================================
def human_proxy_J(
    rig, arm: dict, root_xy, yaw_deg: float, stand_z: float, lean_rad: float = 0.0
):
    """World-space SMPL+H joints ``J`` (52,3) at a candidate stand pose: the body posed with ``arm``
    (idle-arms-down for the body-core check, or an arm-EXTENDED reach keyframe), leaned ``lean_rad``,
    placed at ``(root_xy, yaw_deg)`` with the feet on the floor via ``stand_z``. Fed to the overlap gate
    (``physics_human.human_scene_overlap`` samples capsule spheres straight from J) and, for the P0
    positive control, to ``build_proxy_rig`` / ``update_proxy_rig``. Lazy torch import (smplh_anim_core)
    so module import stays Kit/torch-free."""
    import humanoid_motion as wpp
    import smplh_anim_core as core

    pose = core.assemble_pose(rig, arm, (0.0, 0.0, 0.0), 1.0)
    if abs(lean_rad) > 1e-9:
        pose = wpp._add_lean(pose, lean_rad)
    from gait_player import pose_place_joints

    J = pose_place_joints(
        pose, (float(root_xy[0]), float(root_xy[1])), float(yaw_deg), float(stand_z)
    )
    return J


def _stand_z_for(rig, floor_z: float) -> float:
    """Root z that puts the idle-stand body's feet on ``floor_z`` (matches make_state's stand_z)."""
    import smplh_anim_core as core

    V0, _ = core.pose_body(rig, rig.arm_grasp, (0.0, 0.0), 0.0)
    return float(floor_z) - float(V0[:, 2].min())


# Body-CORE proxy joints (SMPL+H): pelvis/hips/spine/knees/ankles/feet(10,11)/neck/collars(13,14)/
# shoulders(16,17)/head. Includes the FOOT joints (10/11) so the ankle->foot capsules gate a thin low
# obstacle, and the COLLAR+SHOULDER joints (13/14/16/17) so the collar capsules gate the ~0.20 m shoulder
# half-width -- those are NOT the reaching arm (the r_upper_arm bone (17,19) needs joint 19, which stays
# OUT of core, so the reaching arm/forearm/hand capsules are never sampled by the body-core gate). The
# reaching ARM + HAND (18/19/20/21 + MANO) are excluded -- they are MEANT to touch furniture during a
# grasp (matching humanoid_motion._NUDGE_J). A bone is sampled only when BOTH its endpoints are in
# this set (see physics_human._bone_overlap_samples_subset). Used by the AABB pre-filter too.
_CORE_J = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
# P3: a FLOOR crouch folds the lower legs forward -- the thigh/shin/foot capsules legitimately enter the
# furniture the object rests against (the crouch analogue of the reaching ARM, already excluded because it
# is MEANT to touch the target). So the arm-extended gate, WHEN the reach pose squats, drops the
# knee/ankle/foot joints (a bone is sampled only if BOTH endpoints are in the set, so thigh(1/2->4/5),
# shin(4/5->7/8) and foot(7/8->10/11) stop being sampled) -- pelvis/spine/torso/shoulders/head stay gated
# (a real torso/head-into-furniture still rejects) and the planted feet are already gated by the idle pose
# (gate a). Non-crouch reaches keep the full _CORE_J -> byte-identical.
_CROUCH_CORE_J = tuple(j for j in _CORE_J if j not in (4, 5, 7, 8, 10, 11))
_LEG_BONE_NAMES = frozenset(
    {"l_thigh", "r_thigh", "l_shin", "r_shin", "l_foot", "r_foot"}
)


# =====================================================================================================
# Placement gate (P1): AABB pre-filter -> lag-free human-vs-scene overlap confirm (from J).
# =====================================================================================================
def _core_boxes(footprints, floor_top_max: float = 0.15, floor_z: float = 0.0):
    """Solid AABBs [(owner, mn, mx)] from collidable footprints for the AABB pre-filter. Keeps only
    footprints that ``_blocks_floor`` (floor slab + overhead ceilings dropped). Each box spans its real
    z-range so the pre-filter matches the exact overlap gate (a torso inside a table's z-band is caught,
    a ceiling at z~4 is skipped)."""
    boxes = []
    for owner, cx, cy, hx, hy, bot, top in footprints:
        if not _blocks_floor(bot, top, floor_z, floor_top_max):
            continue
        boxes.append(
            (
                owner,
                np.array([cx - hx, cy - hy, bot], dtype=np.float64),
                np.array([cx + hx, cy + hy, top], dtype=np.float64),
            )
        )
    return boxes


def validate_placement(
    ph,
    stage: "Usd.Stage",
    J,
    scene_paths: list[str],
    joint_subset: tuple[int, ...] | None = None,
    prefilter_boxes=None,
) -> list[tuple[str, str]]:
    """Return the human-vs-scene penetrating pairs for the body posed at world joints ``J`` (empty =
    clear). GATE = ``physics_human.human_scene_overlap`` -- lag-free PhysX scene-query overlaps sampled
    along the proxy capsules straight from ``J`` (no sim step, no USD write). ``joint_subset`` restricts
    to those bones (e.g. ``_CORE_J`` for the body-core idle gate; the arm/hand reach furniture legit).

    (1) AABB PRE-FILTER (``prefilter_boxes`` = solid AABBs [(name, mn, mx)]): if the sampled joints are
    clearly OUTSIDE every box inflated by the body radius, the tighter true mesh cannot overlap either
    -> return [] WITHOUT a physics query (the microsecond cull). One-sided: any ambiguous/dirty case
    falls through to the exact overlap query. ``scene_paths`` must EXCLUDE the floor (feet touch it)."""
    idx = list(joint_subset) if joint_subset is not None else list(range(J.shape[0]))
    if prefilter_boxes is not None:
        import semantic_executor as se

        inflated = [
            (name, np.asarray(mn) - HUMAN_FOOT_R, np.asarray(mx) + HUMAN_FOOT_R)
            for name, mn, mx in prefilter_boxes
        ]
        depth, _hit = se._verts_in_boxes(J[idx], inflated)
        if (
            depth <= 0.0
        ):  # every sampled joint clearly outside every inflated box -> cannot overlap
            return []
    return ph.human_scene_overlap(stage, J, scene_paths, joint_subset=joint_subset)


def _non_floor_paths(
    footprints, floor_z: float = 0.0, floor_top_max: float = 0.15
) -> list[str]:
    """The collidable OWNER paths a STANDING human can actually hit: floor slabs (feet rest on them) AND
    overhead ceilings/lamps (a standing human passes under them) are excluded via ``_blocks_floor``.
    These are the ``scene_paths`` fed to the overlap gate -- the feet + head never false-positive."""
    return [
        owner
        for owner, _cx, _cy, _hx, _hy, bot, top in footprints
        if _blocks_floor(bot, top, floor_z, floor_top_max)
    ]


# =====================================================================================================
# START placement (P2): free-cell + one exact confirm + seeded reject-resample fallback + flag.
# =====================================================================================================
def _start_free_mask(grid, allowed_cells):
    free = ~grid.occ
    if allowed_cells is None:
        return free
    allowed = np.asarray(allowed_cells, dtype=bool)
    if allowed.shape != grid.occ.shape:
        raise ValueError(
            f"allowed_cells shape {allowed.shape} != navigation grid {grid.occ.shape}"
        )
    return free & allowed


def place_start(
    ph,
    sim,  # noqa: ARG001 (kept for API symmetry; physics must be init'd by the caller before this)
    stage: "Usd.Stage",
    scene: "SceneState",
    rig,
    seed: int,
    region: tuple[float, float, float, float] | None = None,
    fallback_xy: tuple[float, float] | None = None,
    proxy=None,  # noqa: ARG001 (the overlap gate samples from J directly; no proxy USD needed)
    diverse: bool = False,
    navigation_grid=None,
    allowed_cells=None,
) -> tuple[tuple[float, float], dict]:
    """Choose a NON-PENETRATING start XY for the human. Builds the occupancy grid from every collidable
    footprint, ranks free cells NEAREST the fallback hint (else most-open), and confirms each with the
    lag-free overlap gate. Accepts the first clean cell; after ``MAX_START_DRAWS`` accepts the
    least-penetrating (fewest-overlapping) candidate and sets ``episode_valid=False``. Returns
    ``(start_xy, stats)``.

    ``fallback_xy`` (the scene's hardcoded start_xy) biases the sampler toward a SENSIBLE start near the
    action (the free-cell grid still guarantees non-penetration); ``region`` defaults to the bbox of all
    footprints inflated by ~2 m. Requires physics initialized (get_sim + one step) so the scene-query
    overlap sees the furniture colliders.

    ``allowed_cells`` optionally restricts sampling to a boolean mask on ``navigation_grid``.  It is used
    only by the task-aware repair path after the ordinary start proved unable to reach every task target;
    callers that omit it retain the original seeded ordering byte-for-byte.

    ``diverse`` (the EVAL / random-init path): DROP the near-hint distance bias entirely and rank the
    comfortably-open cells by clearance perturbed with FULL-region seeded jitter, so different seeds land
    at GENUINELY different placements (the near-hint + <=1-cell jitter used for the render biases every
    seed to ~the same cell -- fine for a sensible live start, useless as an independent-seed robustness
    proof). Non-penetration is still guaranteed by the free-cell grid + the per-cell overlap confirm."""
    footprints = collidable_footprints(stage)
    if navigation_grid is None:
        if region is None:
            region = _auto_region(footprints, fallback_xy, floor_z=scene.floor_z)
        grid = build_occupancy_grid(
            footprints, region, cell=0.15, inflate=HUMAN_FOOT_R, floor_z=scene.floor_z
        )
        clearance = grid.clearance
    else:
        grid = navigation_grid
        clearance = grid.sdf
        region = (
            float(grid.x0),
            float(grid.x0 + grid.nx * grid.cell),
            float(grid.y0),
            float(grid.y0 + grid.ny * grid.cell),
        )
    grid_free = int((~grid.occ).sum())

    stand_z = _stand_z_for(rig, scene.floor_z)
    arm_idle = getattr(rig, "arm_rest", rig.arm_grasp)
    import action_check

    camera_policy_verified = bool(
        action_check.uses_egocentric_camera_collision_policy(scene)
        and getattr(getattr(scene, "cworld", None), "sphere_overlap", None) is not None
    )
    # exclude floor + overhead structures -> feet/head don't false-positive
    scene_paths = _non_floor_paths(footprints, floor_z=scene.floor_z)
    prefilter_boxes = _core_boxes(footprints, floor_z=scene.floor_z)

    free = np.argwhere(_start_free_mask(grid, allowed_cells))
    stats = {
        "grid_free_cells": grid_free,
        "start_draws": 0,
        "confirm_pairs": 0,
        "worst_sep": 0.0,
        "resamples": 0,
        "fallback_taken": False,
        "episode_valid": True,
        "start_cell": None,  # grid (i=x, j=y) index of the accepted cell (seed-diversity proof)
    }
    if free.size == 0:
        LOG.warning("place_start: NO free cell in region %s -> fallback + flag", region)
        stats["fallback_taken"] = True
        stats["episode_valid"] = False
        return (
            fallback_xy
            or (0.5 * (region[0] + region[1]), 0.5 * (region[2] + region[3]))
        ), stats

    clr = clearance[free[:, 0], free[:, 1]]
    rng = np.random.default_rng(int(seed))
    # Rank the free cells. A cell is "comfortably open" if its clearance exceeds the human radius (so the
    # body is not just barely clear). Among comfortably-open cells, prefer the one NEAREST the fallback
    # hint (the scene's intended start) so the human starts SENSIBLY near the action rather than in the
    # far corner (which is the globally-most-open but 7 m away); seeded jitter breaks near-ties so seeds
    # differ. With no hint (a pure random scene) fall back to most-open first.
    comfy = clr >= HUMAN_FOOT_R
    cell_xy = np.stack(
        [
            grid.x0 + (free[:, 1] + 0.5) * grid.cell,
            grid.y0 + (free[:, 0] + 0.5) * grid.cell,
        ],
        axis=1,
    )
    if diverse and bool(comfy.any()):
        # EVAL / random-init: NO near-hint bias. Rank the comfortably-open cells by clearance perturbed
        # with jitter spanning the WHOLE clearance range (>> a grid cell), so distinct seeds pick distinct
        # cells across the free region -- a genuine independent-sample of the random-init distribution.
        clr_span = float(np.ptp(clr[comfy])) if int(comfy.sum()) > 1 else 1.0
        jitter = rng.uniform(0.0, max(clr_span, grid.cell), size=clr.shape)
        score = np.where(
            comfy, -clr + jitter, np.inf
        )  # low score = open + this seed's draw
        order = np.argsort(score)
    elif fallback_xy is not None and bool(comfy.any()):
        hint = np.asarray(fallback_xy, dtype=np.float64)
        dist = np.hypot(cell_xy[:, 0] - hint[0], cell_xy[:, 1] - hint[1])
        jitter = rng.uniform(0.0, grid.cell, size=dist.shape)
        score = np.where(
            comfy, dist + jitter, np.inf
        )  # nearest comfy cell first; non-comfy last
        order = np.argsort(score)
    else:
        order = np.argsort(-clr)  # most open first
        order = order[: max(MAX_START_DRAWS, 1)]
        rng.shuffle(order)
    top = order[: max(MAX_START_DRAWS, 1)]

    best_xy, best_n, best_cell = None, 1 << 30, None
    best_camera_clear = None
    for k, oi in enumerate(top[:MAX_START_DRAWS]):
        j, i = free[oi]
        cand = grid.cell_center(int(i), int(j))
        stats["start_draws"] = k + 1
        J = human_proxy_J(rig, arm_idle, cand, 0.0, stand_z)
        pairs = validate_placement(
            ph,
            stage,
            J,
            scene_paths,
            joint_subset=_CORE_J,
            prefilter_boxes=prefilter_boxes,
        )
        camera_clear = not camera_policy_verified or _camera_proxy_pose_is_clear(
            scene, J
        )
        if not pairs and camera_clear:
            stats["start_cell"] = (
                int(i),
                int(j),
            )  # grid (i=x, j=y) index of the ACCEPTED cell
            LOG.info(
                "place_start: cell (%.2f,%.2f) [grid ij=(%d,%d)] CLEAN after %d draw(s) [grid_free=%d]",
                cand[0],
                cand[1],
                int(i),
                int(j),
                k + 1,
                grid_free,
            )
            return (float(cand[0]), float(cand[1])), stats
        if camera_clear and (
            best_camera_clear is None or len(pairs) < best_camera_clear[0]
        ):
            best_camera_clear = (len(pairs), cand, (int(i), int(j)))
        if len(pairs) < best_n:
            best_xy, best_n, best_cell = cand, len(pairs), (int(i), int(j))
    if best_camera_clear is not None and camera_policy_verified:
        best_n, best_xy, best_cell = best_camera_clear
        stats["start_cell"] = best_cell
        stats["fallback_taken"] = True
        stats["confirm_pairs"] = int(best_n)
        stats["body_collision_accepted"] = True
        LOG.warning(
            "place_start: no body-clear cell in %d draws -> camera-clear body-overlap "
            "fallback (%.2f,%.2f), overlap_pairs=%d",
            MAX_START_DRAWS,
            best_xy[0],
            best_xy[1],
            best_n,
        )
        return (float(best_xy[0]), float(best_xy[1])), stats
    stats["start_cell"] = best_cell

    # Exhausted: accept least-penetrating + flag (the RL contract -- never silently accept a bad pose).
    stats["fallback_taken"] = True
    stats["episode_valid"] = False
    stats["confirm_pairs"] = int(best_n)
    LOG.warning(
        "place_start: no clean cell in %d draws -> least-penetrating (%.2f,%.2f) overlap_pairs=%d "
        "episode_valid=False",
        MAX_START_DRAWS,
        best_xy[0],
        best_xy[1],
        best_n,
    )
    return (float(best_xy[0]), float(best_xy[1])), stats


def _auto_region(footprints, fallback_xy, floor_z: float = 0.0, pad: float = 2.0):
    """Region (xmin,xmax,ymin,ymax) = the bbox of all floor-blocking footprints, padded by ``pad`` m so
    the grid has open floor around the furniture. Falls back to a box around ``fallback_xy`` when empty.
    Overhead ceilings are excluded so the region is the reachable floor area, not the ceiling extent."""
    solids = [
        (cx, cy, hx, hy)
        for _o, cx, cy, hx, hy, bot, top in footprints
        if _blocks_floor(bot, top, floor_z, 0.15)
    ]
    if not solids:
        fx, fy = fallback_xy or (0.0, 0.0)
        return (fx - pad, fx + pad, fy - pad, fy + pad)
    xmin = min(cx - hx for cx, cy, hx, hy in solids) - pad
    xmax = max(cx + hx for cx, cy, hx, hy in solids) + pad
    ymin = min(cy - hy for cx, cy, hx, hy in solids) - pad
    ymax = max(cy + hy for cx, cy, hx, hy in solids) + pad
    return (xmin, xmax, ymin, ymax)


# =====================================================================================================
# STAND-POSE validate + adjust (P3): stand_pose_for as generator, ladder-gate + reach cap + arm check.
# =====================================================================================================
@dataclass
class StandResult:
    stand_xy: tuple[float, float]
    yaw: float
    stats: dict = field(default_factory=dict)


def _arm_extended_keyframe(
    rig,
    scene,
    key,
    stand_xy,
    yaw,
    stand_z,
    lean_override=None,
    squat_override=0.0,
    contact_override=None,
    pose_override=None,
):
    """A representative ARM-EXTENDED reach keyframe for entity ``key`` at (stand_xy, yaw): warp the arm
    so the grip seat lands on the entity's contact/handle, with the shared pose. Returns the world joints J
    of that reach pose (arm/hand extended toward the furniture), or None if the entity has no reachable
    contact (a plain floor goal). ``lean_override``/``squat_override`` (rad, 0..1; from the pose-planner) are
    used verbatim so the GATED pose is EXACTLY the crouched/leaned pose the producer renders; a squat
    re-pins the root so the crouched shoulder sits at its true world height. None/0 -> the height-only lean.
    ``contact_override`` (world xyz) forces the reach target instead of the per-role contact -- used by the
    runtime DOOR stand pick so the gated pose reaches the exact HANDLE the OpenDoor producer grips (the
    per-role OPENABLE contact is the door TOP surface, not the handle)."""
    import humanoid_motion as wpp
    import semantic_executor as se
    import smplh_anim_core as core
    from scene_state import Role

    if contact_override is not None:
        contact = np.asarray(contact_override, dtype=np.float64)
    else:
        e = scene.resolve(key)
        if e.role is Role.GRASPABLE:
            contact = scene.object_pos(key)
        elif e.role is Role.CONTAINER and getattr(e, "drop_z", None) is not None:
            # reach the interior DROP point (~shoulder height), NOT the cabinet TOP -- see _entity_contact.
            cxy = e.xy if e.xy is not None else scene.object_pos(key)[:2]
            contact = np.array([cxy[0], cxy[1], float(e.drop_z)], dtype=np.float64)
        elif e.role in (Role.SURFACE, Role.CONTAINER, Role.OPENABLE):
            (sx, sy), top = scene.surface_top(key)
            contact = np.array([sx, sy, top], dtype=np.float64)
        else:
            return None
    lean_rad = (
        float(lean_override)
        if lean_override is not None
        else se._lean_for_height(float(contact[2]))[0]
    )
    base = se._reach_base_pose(rig, rig.arm_grasp, lean_rad, float(squat_override), 1.0)
    _sz = (
        se._pin_z(base, float(scene.floor_z))
        if float(squat_override) > 1e-6
        else float(stand_z)
    )
    rz, world_off = wpp._grasp_frame(stand_xy, yaw, _sz)
    if pose_override is None:
        pose, _err = core.warp_arm_to_seat(
            rig, base, rz, world_off, np.asarray(contact, dtype=np.float64)
        )
    else:
        pose = np.asarray(pose_override, dtype=np.float32)
    J = wpp.gait.pose_place_joints(
        pose, (float(stand_xy[0]), float(stand_xy[1])), float(yaw), _sz
    )
    return J


def _stand_gate_setup(stage, scene, rig, key):
    """Shared setup for the stand gate/search: the non-floor collision paths + core-AABB prefilter (feet/
    head excluded so they don't false-positive), the stand foot-z, the idle (arms-down) arm pose, and the
    reach contact for ``key``. Used by BOTH the INIT validator (``validate_stand_pose``) and the RUNTIME
    picker (``pick_stand_runtime``) so they gate against the byte-identical geometry."""
    collision_world = getattr(scene, "cworld", None)
    if collision_world is not None:
        scene_paths = list(collision_world.scene_paths)
        prefilter_boxes = collision_world.boxes()
    else:
        footprints = collidable_footprints(stage)
        scene_paths = _non_floor_paths(footprints, floor_z=scene.floor_z)
        prefilter_boxes = _core_boxes(footprints, floor_z=scene.floor_z)
    stand_z = _stand_z_for(rig, scene.floor_z)
    arm_idle = getattr(rig, "arm_rest", rig.arm_grasp)
    contact = _entity_contact(scene, key)
    return scene_paths, prefilter_boxes, stand_z, arm_idle, contact


def validate_stand_pose(
    ph,
    sim,  # noqa: ARG001 (kept for API symmetry; physics must be init'd before this)
    stage: "Usd.Stage",
    scene: "SceneState",
    rig,
    key: str,
    proxy=None,  # noqa: ARG001 (the overlap gate samples from J directly; no proxy USD needed)
    nav_grid=None,  # optional passable NavGrid: a chosen stand must also be WALKABLE from start_xy
    start_xy=None,  # the human start; connectivity is checked from here
) -> StandResult:
    """Validate + adjust the stand pose for reaching entity ``key``. Keeps ``scene.stand_pose_for`` as
    the CANDIDATE GENERATOR (near-face + travel + approach_yaw geometry unchanged), then gates + adjusts:

      * ladder: the base candidate, then FURTHER back-off along the SAME approach axis (stepped), then
        re-derive stand_pose_for at rotated approach_yaw (+-45, +-90, 180) -- each gated.
      * reach-envelope cap: a rung whose stand->contact horizontal distance exceeds REACH_ENVELOPE_M is
        REJECTED (clears the gate but out of arm reach).
      * body-CORE idle pose gated (arm/hand excluded, matching _NUDGE_J) AND the arm-EXTENDED reach
        keyframe's CORE gated SEPARATELY (a reach-LEAN drives the torso into furniture -- invisible to
        the idle gate; the reaching ARM legitimately enters the target furniture, so it is NOT gated,
        only counted for stats).
      * strict acceptance checks the short, stable final approach used for a grid-blocked stand.

    Gate = the lag-free ``physics_human.human_scene_overlap`` (sampled from J). On exhaustion accepts
    the least-penetrating rung + ``episode_valid=False``. Returns a StandResult (stand_xy, yaw, stats)."""
    scene_paths, prefilter_boxes, stand_z, arm_idle, contact = _stand_gate_setup(
        stage, scene, rig, key
    )
    import action_check

    strict = bool(getattr(scene, "strict_acceptance", False))
    camera_only_policy = action_check.uses_egocentric_camera_collision_policy(scene)
    camera_probe_available = (
        getattr(getattr(scene, "cworld", None), "sphere_overlap", None) is not None
    )
    camera_policy_verified = camera_only_policy and camera_probe_available
    base_stand, base_yaw = scene.stand_pose_for(key)
    approach = _approach_dir(base_stand, contact, base_yaw)

    # Build the ladder of (stand_xy, yaw, label) candidates: base, stepped back-off, rotated approaches.
    ladder = _stand_ladder(scene, key, base_stand, base_yaw, approach, contact)

    stats = {
        "ladder_step_used": 0,
        "confirm_pairs": 0,
        "worst_sep": 0.0,
        "resamples": 0,
        "fallback_taken": False,
        "episode_valid": True,
        "arm_pairs": 0,  # arm overlaps AT the contact region (legitimate hand-at-handle; stat only)
        "arm_out_pairs": 0,  # arm overlaps OUTSIDE the contact region (arm-through-furniture; gated)
        "arrival_pairs": 0,
    }
    best = None  # (n_pairs, stand_xy, yaw, arm_pairs_at_handle, arm_out)
    clean_islanded = (
        None  # a non-penetrating rung that is NOT walkable-from-start (nav-aware only)
    )

    # PRIMARY stand search (uniform for EVERY scene, no scene-specific gating): scan ALL free floor cells
    # AROUND the target -- not a handful of fixed ladder directions -- for a stand that is idle-clean AND
    # pose-planner-reachable AND arm-clear (AND nav-walkable when a grid exists). This is the general search
    # the fixed ladder could not be (too few directions), and it runs in pass-1 too (no longer nav-blind).
    # The fixed ladder below is kept ONLY as the least-penetrating fallback source for the honest-fail path.
    if contact is not None:
        _found = _search_reachable_stand(
            ph,
            stage,
            rig,
            scene,
            key,
            contact,
            stand_z,
            arm_idle,
            scene_paths,
            prefilter_boxes,
            nav_grid=nav_grid,
            start_xy=start_xy,
        )
        if _found is not None:
            (fx, fy), fyaw = _found
            stats["confirm_pairs"] = 0
            stats["fallback_taken"] = False
            stats["episode_valid"] = True
            stats["nav_connected"] = nav_grid is not None
            stats["reachable_search"] = True
            LOG.info(
                "validate_stand_pose[%s]: cell-scan found (%.2f,%.2f) yaw=%.0f (idle+reach+arm clean%s)",
                key,
                fx,
                fy,
                fyaw,
                ", nav-walkable" if nav_grid is not None else "",
            )
            return StandResult((fx, fy), float(fyaw), stats)

    for step, (sxy, syaw, label) in enumerate(ladder[:MAX_STAND_LADDER]):
        stats["ladder_step_used"] = step + 1
        # reach gate: reject a rung that cannot reach the contact. unified_geom -> the POSE-PLANNER's real
        # 3D IK reachability (lean forward as far as needed); the chosen lean is reused for the arm-extended
        # keyframe below so the gate poses EXACTLY what the producer renders. Off -> the flat horizontal cap
        # ignoring z (byte-identical).
        _reach_lean = None
        _reach_squat = 0.0
        if contact is not None:
            if getattr(scene, "unified_geom", False):
                import semantic_executor as _se

                _gate_tol = _se._reach_tolerance(scene)
                _rp = _plan_stand_reach(
                    _se,
                    rig,
                    scene,
                    contact,
                    sxy,
                    syaw,
                    stand_z,
                    _gate_tol,
                    strict=strict,
                )
                _reach_lean = _rp["lean_rad"]
                _reach_squat = _rp["squat"]
                if _rp.get("collision_blocked"):
                    LOG.info(
                        "validate_stand_pose[%s] rung %s (%.2f,%.2f) has no "
                        "collision-free reach pose -> skip",
                        key,
                        label,
                        sxy[0],
                        sxy[1],
                    )
                    continue
                if not _rp["reachable"]:
                    LOG.info(
                        "validate_stand_pose[%s] rung %s (%.2f,%.2f) reach_err=%.0fmm > %.0fmm "
                        "-> skip (3D IK lean %.0fdeg)",
                        key,
                        label,
                        sxy[0],
                        sxy[1],
                        _rp["err"] * 1000.0,
                        _gate_tol * 1000.0,
                        math.degrees(_rp["lean_rad"]),
                    )
                    continue
            else:
                reach = float(np.hypot(contact[0] - sxy[0], contact[1] - sxy[1]))
                if reach > REACH_ENVELOPE_M:
                    LOG.info(
                        "validate_stand_pose[%s] rung %s (%.2f,%.2f) reach=%.2fm > envelope %.2fm -> skip",
                        key,
                        label,
                        sxy[0],
                        sxy[1],
                        reach,
                        REACH_ENVELOPE_M,
                    )
                    continue
        # (a) body-CORE idle gate (arms down).
        J_idle = human_proxy_J(rig, arm_idle, sxy, syaw, stand_z)
        core_pairs = validate_placement(
            ph,
            stage,
            J_idle,
            scene_paths,
            joint_subset=_CORE_J,
            prefilter_boxes=prefilter_boxes,
        )
        if strict and not camera_policy_verified:
            core_pairs = _severe_body_hits(scene, J_idle, core_pairs)
        if camera_policy_verified and not _camera_proxy_pose_is_clear(scene, J_idle):
            continue
        # (b) arm-EXTENDED reach keyframe: gate its CORE (the lean can drive the torso in) AND the ARM,
        # but the ARM is gated against the furniture MINUS the intended CONTACT region (a small sphere
        # around the handle/grasp point). The reaching hand/fingers legitimately enter the furniture AT
        # the handle (those samples are excluded), but an arm/forearm reaching THROUGH the cabinet body
        # away from the handle IS a violation -> it joins all_pairs and drives episode_valid=False.
        Jr = _arm_extended_keyframe(
            rig,
            scene,
            key,
            sxy,
            syaw,
            stand_z,
            lean_override=_reach_lean,
            squat_override=_reach_squat,
        )
        reach_core: list = []
        arm_out: list = []
        if Jr is not None:
            _crouch = (
                _reach_squat > 1e-6
            )  # P3: drop the folding leg capsules when the reach squats
            reach_core = validate_placement(
                ph,
                stage,
                Jr,
                scene_paths,
                joint_subset=(_CROUCH_CORE_J if _crouch else _CORE_J),
                prefilter_boxes=prefilter_boxes,
            )
            core_names = {p[0] for p in reach_core}
            # (b1) ALL overlaps (arm + hand + core) minus the intended CONTACT sphere; subtract the core
            # names -> the ARM capsules overlapping furniture OUTSIDE the contact region (the arm plowing
            # THROUGH the cabinet body away from the handle). These are VIOLATIONS -> they join all_pairs.
            excl = (
                (np.asarray(contact, dtype=np.float64), _CONTACT_SPHERE_R)
                if contact is not None
                else None
            )
            arm_out = [
                p
                for p in ph.human_scene_overlap(
                    stage, Jr, scene_paths, exclude_sphere=excl
                )
                if p[0] not in core_names and not (_crouch and p[0] in _LEG_BONE_NAMES)
            ]
            # (b2) full arm overlaps (no sphere) minus core -> total arm-vs-furniture; the AT-handle count
            # (legitimate) is that total minus arm_out. Reported as a stat only (NOT gated).
            arm_full = [
                p
                for p in ph.human_scene_overlap(stage, Jr, scene_paths)
                if p[0] not in core_names and not (_crouch and p[0] in _LEG_BONE_NAMES)
            ]
            stats["arm_pairs"] = max(0, len(arm_full) - len(arm_out))
            if strict and not camera_policy_verified:
                reach_core = _severe_body_hits(scene, Jr, reach_core)
            if camera_policy_verified and not _camera_proxy_pose_is_clear(scene, Jr):
                continue

        arrival_pairs = []
        if strict and not camera_policy_verified:
            arrival_pairs = _controlled_arrival_hits(
                ph,
                stage,
                rig,
                scene,
                arm_idle,
                sxy,
                syaw,
                stand_z,
                scene_paths,
                prefilter_boxes,
                nav_grid,
                start_xy,
            )
        all_pairs = core_pairs + reach_core + arm_out + arrival_pairs
        if not all_pairs:
            stats["confirm_pairs"] = 0
            stats["arm_out_pairs"] = 0
            stats["arrival_pairs"] = 0
            # NAV-AWARE (task scenes pass nav_grid): a CLEAN rung must ALSO be walkable from the start,
            # else it is an islanded pose the human cannot reach -> stash it and keep searching for a
            # clean+connected rung. When nav_grid is None (every other caller) the FIRST clean rung wins
            # -> byte-identical to the pre-nav behaviour.
            if nav_grid is not None and start_xy is not None:
                if not _stand_connected(nav_grid, start_xy, sxy):
                    if clean_islanded is None:
                        clean_islanded = (
                            float(sxy[0]),
                            float(sxy[1]),
                            float(syaw),
                            dict(stats),
                        )
                    continue
                stats["nav_connected"] = True
            LOG.info(
                "validate_stand_pose[%s]: rung %s (%.2f,%.2f yaw=%.0f) CLEAN (idle+reach core+arm-out; "
                "arm_at_handle=%d)",
                key,
                label,
                sxy[0],
                sxy[1],
                syaw,
                stats["arm_pairs"],
            )
            return StandResult((float(sxy[0]), float(sxy[1])), float(syaw), stats)
        if best is None or len(all_pairs) < best[0]:
            best = (len(all_pairs), sxy, syaw, stats["arm_pairs"], len(arm_out))
        stats["arrival_pairs"] = len(arrival_pairs)

    # (the general cell-scan already ran as the PRIMARY search above; the fixed ladder here only supplies the
    # least-penetrating candidate for the honest-fail path below.)
    if clean_islanded is not None and not strict:
        # A non-penetrating rung existed but is not walkable (and the reachable search found nothing).
        # Accept it (matches the pre-nav 'first clean' behaviour: a valid non-penetrating stand); the
        # final grid will simply straight-line that leg.
        cx, cy, cyaw, cst = clean_islanded
        cst["nav_connected"] = False
        LOG.warning(
            "validate_stand_pose[%s]: only an ISLANDED clean stand (%.2f,%.2f) -> accepted (leg will "
            "straight-line)",
            key,
            cx,
            cy,
        )
        return StandResult((cx, cy), float(cyaw), cst)

    # Exhausted: accept least-penetrating rung + flag.
    stats["fallback_taken"] = True
    stats["episode_valid"] = False
    if best is not None and (not strict or camera_policy_verified):
        stats["confirm_pairs"] = int(best[0])
        stats["arm_pairs"] = int(best[3])
        stats["arm_out_pairs"] = int(best[4])
        stats["episode_valid"] = bool(camera_policy_verified)
        stats["body_collision_accepted"] = bool(camera_policy_verified)
        LOG.warning(
            "validate_stand_pose[%s]: no clean rung in %d -> least-penetrating (%.2f,%.2f) "
            "overlap_pairs=%d (arm_out=%d) episode_valid=%s",
            key,
            stats["ladder_step_used"],
            best[1][0],
            best[1][1],
            best[0],
            best[4],
            stats["episode_valid"],
        )
        return StandResult(
            (float(best[1][0]), float(best[1][1])), float(best[2]), stats
        )
    # No rung even survived the reach cap: fall back to the base candidate + flag.
    LOG.warning(
        "validate_stand_pose[%s]: no reachable rung -> base candidate + flag", key
    )
    return StandResult(
        (float(base_stand[0]), float(base_stand[1])), float(base_yaw), stats
    )


def _shuffle_tol(nav_grid) -> float:
    """The stand-to-walkable shuffle tolerance for ``nav_grid``: its own body inflation plus
    ``STAND_SHUFFLE_SLACK``. 0.25 m at the 0.10 m floor, i.e. unchanged from the fixed value this
    replaces."""
    infl = getattr(nav_grid, "inflate", None)
    return (NAV_INFLATE_M if infl is None else float(infl)) + STAND_SHUFFLE_SLACK


def _reachable_free_within(
    nav_grid, reach_mask, wx, wy, tol: float | None = None
) -> bool:
    """True iff a start-reachable (in ``reach_mask``) FREE cell lies within ``tol`` metres of world (wx, wy).
    This is the stand-connectivity criterion, factored out so the caller (the stand search) can flood ONCE
    and pre-cull islanded candidates cheaply, instead of re-flooding per candidate. ``tol=None`` derives it
    from the grid's inflation (see ``_shuffle_tol``)."""
    if tol is None:
        tol = _shuffle_tol(nav_grid)
    ci, cj = nav_grid.world_to_cell(float(wx), float(wy))
    r = max(1, int(round(tol / nav_grid.cell)))
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            i, j = ci + di, cj + dj
            if (
                0 <= i < nav_grid.nx
                and 0 <= j < nav_grid.ny
                and reach_mask[j, i]
                and nav_grid.is_free(i, j)
            ):
                return True
    return False


def _stand_routes(flood_grid, start_xy, wx, wy) -> bool:
    """True iff the ROUTER can actually get the body from ``start_xy`` to the stand at (wx, wy).

    ``_reachable_free_within`` (gate b0) asks a different question -- is there a start-reachable free
    cell within a shuffle of the stand -- and the two disagree whenever the shuffle bridges a gap the
    router will not walk. When they disagree the picker returns a stand and ``navigate_to`` then fails
    ``path_blocked``, which costs the command and every sub-goal downstream of it. Measured: 9 of the
    23 T5 rows, `path_blocked` 2-13 each.

    Only a genuine ``PATH_BLOCKED`` vetoes. Every other exception fails OPEN, because an unmeasurable
    route must never remove a stand that works today -- this gate may only ever reject stands whose
    command was already going to fail.
    """
    import nav_planner as _np
    from goal_schema import PrimitiveFailure

    try:
        _np.plan_path(
            flood_grid, (float(start_xy[0]), float(start_xy[1])), (float(wx), float(wy))
        )
        return True
    except PrimitiveFailure:
        return False
    except Exception as exc:  # noqa: BLE001 -- unmeasurable, so not this gate's business
        LOG.info("[stand-scan] route gate unavailable at (%.2f,%.2f): %s", wx, wy, exc)
        return True


def _stand_connected(nav_grid, start_xy, sxy) -> bool:
    """True iff stand ``sxy`` is genuinely walkable from ``start_xy``. We do NOT use nav_grid.connected():
    it snaps a blocked target to its nearest free cell ACROSS walls, so an islanded stand (grid-blocked
    but overlap-clean, sitting in a walled pocket) would falsely pass. Instead: flood free-from-start, then
    require a reachable FREE cell within a ``_shuffle_tol`` shuffle of the stand (a stand on the edge of
    walkable floor qualifies; one inside a sealed pocket does not). Fail-OPEN on error."""
    import nav_grid as _ng

    try:
        reach = _ng.flood_free(
            nav_grid, nav_grid.world_to_cell(float(start_xy[0]), float(start_xy[1]))
        )
        return _reachable_free_within(nav_grid, reach, sxy[0], sxy[1])
    except Exception:  # noqa: BLE001
        return True


def _stand_pen_depth(scene, J) -> float:
    """Worst body-vs-scene penetration DEPTH (metres) for the body posed at ``J``, or ``inf`` if it cannot
    be measured. Deliberately the SAME oracle the render grade uses (``action_check._accumulate_depth``
    bisecting against ``scene.cworld``), so a depth reported here is on the same scale as the
    ``clean``/``marginal``/``dirty`` cut points and the two can never disagree about the same pose.
    ``inf`` on any failure sorts the candidate last -- an unmeasurable fallback is never preferred."""
    try:
        import action_check as _ac

        cw = getattr(scene, "cworld", None)
        if cw is None:
            return float("inf")
        d = _ac._empty_depth()
        _ac._accumulate_depth(cw, J, 0, "stand", d)
        return float(d.get("max_pen_m") or 0.0)
    except Exception:  # noqa: BLE001
        return float("inf")


def _stand_pose_grade(
    scene,
    joints,
    exclude_owners: tuple[str, ...] = (),
    collision_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, object] | None:
    """Grade one stand pose with the exact policy used by final trajectory acceptance."""
    try:
        import action_check as _ac

        cw = getattr(scene, "cworld", None)
        if cw is None:
            return None
        return _ac.grade_human_pose(
            cw,
            np.asarray(joints, dtype=np.float64),
            exclude_owners=exclude_owners,
            collision_pairs=collision_pairs,
        )
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        LOG.info("stand pose severity unavailable: %s", exc)
        return None


def _stand_pose_is_acceptable(
    scene,
    joints,
    exclude_owners: tuple[str, ...] = (),
    collision_pairs: list[tuple[str, str]] | None = None,
) -> bool:
    """Whether a soft stand pose passes the same severity threshold as final acceptance."""
    grade = _stand_pose_grade(
        scene,
        joints,
        exclude_owners=exclude_owners,
        collision_pairs=collision_pairs,
    )
    return bool(grade is not None and grade["passes"])


def _severe_body_hits(
    scene, joints, hits: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Keep only egocentrically visible overlaps that fail the shared render grade."""
    if not hits:
        return []
    visible_hits = [pair for pair in hits if pair[0] not in _LEG_BONE_NAMES]
    if not visible_hits:
        return []
    grade = _stand_pose_grade(scene, joints, collision_pairs=hits)
    return visible_hits if grade is None or not bool(grade["passes"]) else []


def _controlled_arrival_hits(
    ph,
    stage,
    rig,
    scene,
    arm_idle,
    stand_xy,
    yaw,
    stand_z,
    scene_paths,
    prefilter_boxes,
    nav_grid,
    start_xy,
) -> list[tuple[str, str]]:
    """Reject an egregious collision on the short stable-pose approach into a blocked stand."""
    if nav_grid is None or start_xy is None:
        return []
    gi, gj = nav_grid.world_to_cell(float(stand_xy[0]), float(stand_xy[1]))
    if nav_grid.is_free(gi, gj):
        return []

    import nav_planner as _np
    from goal_schema import PrimitiveFailure

    try:
        path = _np.plan_path(nav_grid, start_xy, stand_xy)
    except PrimitiveFailure:
        return [("root", "<unroutable-final-approach>")]
    if len(path) < 2:
        return [("root", "<missing-final-approach>")]

    for xy in _np.final_approach_points(path[-2], stand_xy):
        joints = human_proxy_J(rig, arm_idle, xy, yaw, stand_z)
        hits = validate_placement(
            ph,
            stage,
            joints,
            scene_paths,
            joint_subset=_CORE_J,
            prefilter_boxes=prefilter_boxes,
        )
        if hits and not _stand_pose_is_acceptable(
            scene,
            joints,
            collision_pairs=hits,
        ):
            return hits
    return []


def _path_pose_track(
    path,
    final_yaw: float,
    nav_grid=None,
    start_yaw: float | None = None,
) -> list[tuple[tuple[float, float], float, float]]:
    """Every root pose from the same heading controller used by the rendered walk."""
    if len(path) < 2:
        return []
    if start_yaw is None:
        start, end = path[0], path[1]
        start_yaw = math.degrees(
            math.atan2(float(end[0] - start[0]), -float(end[1] - start[1]))
        )

    import heading_controller as hc
    import humanoid_motion as wpp
    import nav_planner

    route = list(path)
    samples: list[tuple[tuple[float, float], float, float]] = []
    blocked_start = nav_grid is not None and not nav_grid.is_free(
        *nav_grid.world_to_cell(*route[0])
    )
    blocked_goal = nav_grid is not None and not nav_grid.is_free(
        *nav_grid.world_to_cell(*route[-1])
    )
    if blocked_start and len(route) >= 2:
        samples.extend(
            (xy, float(start_yaw), 0.0)
            for xy in nav_planner.final_approach_points(route[0], route[1])
        )
        route = route[1:]
    walk_path = route[:-1] if blocked_goal and len(route) >= 2 else route
    samples.extend(
        (step.xy, step.yaw, step.gait_m)
        for step in hc.walk_track(
            walk_path,
            float(start_yaw),
            float(final_yaw),
            wpp.WALK_SPEED,
            wpp.FPS,
            hc.HeadingCfg(),
        )
    )
    if blocked_goal and len(route) >= 2:
        samples.extend(
            (xy, float(final_yaw), 0.0)
            for xy in nav_planner.final_approach_points(route[-2], route[-1])
        )
    return samples


def _trajectory_sample_every(n_frames: int) -> int:
    if n_frames <= _STAND_TRAJECTORY_MAX_SAMPLES:
        return 1
    return max(
        1,
        int(math.ceil((n_frames - 1) / max(1, _STAND_TRAJECTORY_MAX_SAMPLES - 1))),
    )


def _path_pose_samples(
    path,
    final_yaw: float,
    nav_grid=None,
    start_yaw: float | None = None,
) -> list[tuple[tuple[float, float], float]]:
    """Bounded samples from the same heading controller used by the rendered walk."""
    samples = _path_pose_track(
        path,
        final_yaw,
        nav_grid=nav_grid,
        start_yaw=start_yaw,
    )
    if len(samples) > _STAND_TRAJECTORY_MAX_SAMPLES:
        import action_check

        indices = action_check.sample_indices(
            len(samples), _trajectory_sample_every(len(samples))
        )
        samples = [samples[int(index)] for index in indices]
    return [(xy, yaw) for xy, yaw, _gait_m in samples]


def _route_trajectory_report(
    scene,
    rig,
    arm_idle,
    track,
    stand_z,
    excluded_roots,
    gait_profile=None,
    gait_cum_dist: float = 0.0,
    require_camera_check: bool = False,
    camera_frame_offset: int = 0,
    additional_boxes=(),
):
    """Grade a bounded route preview with the final action-compliance policy."""
    import action_check
    import gait_player
    import humanoid_motion as wpp
    import smplh_camera

    camera_required = bool(
        require_camera_check
        or action_check.uses_egocentric_camera_collision_policy(scene)
    )
    sample_every = _trajectory_sample_every(len(track))
    sampled = action_check.sample_indices(len(track), sample_every)
    joints = [None] * len(track)
    gait_distances = []
    distance = float(gait_cum_dist)
    for _xy, _yaw, gait_m in track:
        distance += float(gait_m)
        gait_distances.append(distance)
    pose_indices = range(len(track)) if camera_required else sampled
    for index in pose_indices:
        xy, yaw, gait_m = track[index]
        if gait_profile is not None and gait_m > 0.0:
            pose, min_foot_z = gait_player.sample(
                gait_profile,
                gait_distances[index],
            )
            joints[index] = gait_player.pose_place_joints(
                pose,
                xy,
                yaw,
                float(scene.floor_z) - float(min_foot_z),
            )
        else:
            joints[index] = human_proxy_J(rig, arm_idle, xy, yaw, stand_z)
    frame = SimpleNamespace(
        collision_exclude_paths=tuple(excluded_roots),
        seg="walk",
    )
    report = action_check.grade_trajectory(
        scene.cworld,
        joints,
        [frame] * len(track),
        sample_every=sample_every,
        fps=wpp.FPS,
    )
    camera_hits = []
    kinematic_leaf_boxes = (
        *_kinematically_open_leaf_boxes(scene),
        *additional_boxes,
    )
    sphere_overlap = getattr(scene.cworld, "sphere_overlap", None)
    if (
        camera_required or not bool(report["render_pass"])
    ) and sphere_overlap is not None:
        camera_indices = range(len(track)) if camera_required else sampled
        for index in camera_indices:
            _xy, yaw, _gait_m = track[index]
            eye, _aim = smplh_camera.fpv_eye_and_aim(
                joints[index],
                "walk",
                np.zeros(3, dtype=np.float64),
                frame=int(camera_frame_offset) + index,
                root_yaw_deg=yaw,
            )
            owner = sphere_overlap(
                eye,
                _CAMERA_CLEAR_RADIUS_M,
                exclude_owners=tuple(excluded_roots),
            )
            if owner is None:
                owner = _sphere_aabb_overlap_owner(
                    eye,
                    _CAMERA_CLEAR_RADIUS_M,
                    kinematic_leaf_boxes,
                )
            if owner is not None:
                camera_hits.append((index, owner))
    camera_probe_available = sphere_overlap is not None
    report["camera_proxy_clear"] = camera_probe_available and not camera_hits
    report["camera_probe_available"] = camera_probe_available
    report["camera_proxy_hits"] = camera_hits
    return report


def _kinematically_open_leaf_roots(scene) -> tuple[str, ...]:
    """Closed-pose leaf prims superseded by the live grid's open-leaf footprint."""
    entities = getattr(scene, "entities", None) or {}
    roots = []
    for key in sorted(getattr(scene, "open_objs", None) or ()):
        entity = entities.get(key)
        if entity is None or not (getattr(entity, "meta", None) or {}).get(
            "_nav_leaf_cells"
        ):
            continue
        roots.extend(str(path) for path in (getattr(entity, "leaf_paths", None) or ()))
    return tuple(dict.fromkeys(roots))


def _kinematically_open_leaf_boxes(
    scene,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """Live open poses for nav-door leaves whose USD prims remain in their closed pose."""
    import semantic_executor

    entities = getattr(scene, "entities", None) or {}
    boxes = []
    for key in sorted(getattr(scene, "open_objs", None) or ()):
        entity = entities.get(key)
        if entity is None or not (getattr(entity, "meta", None) or {}).get(
            "_nav_leaf_cells"
        ):
            continue
        boxes.extend(semantic_executor._open_leaf_boxes(scene, entity))
    return tuple(boxes)


def _sphere_aabb_overlap_owner(point, radius: float, boxes) -> str | None:
    center = np.asarray(point, dtype=np.float64)
    for owner, mn, mx in boxes:
        lower = np.asarray(mn, dtype=np.float64)
        upper = np.asarray(mx, dtype=np.float64)
        closest = np.maximum(lower, np.minimum(center, upper))
        if float(np.dot(center - closest, center - closest)) <= float(radius) ** 2:
            return str(owner)
    return None


def _avoid_camera_hit_cell(grid, track, report, start_xy, target_xy) -> bool:
    """Block one colliding route cell so the next bounded A* attempt must steer around it."""
    import nav_grid

    hits = report.get("camera_proxy_hits") or ()
    if not hits:
        return False
    index = int(hits[0][0])
    if index < 0 or index >= len(track):
        return False
    i, j = grid.world_to_cell(*track[index][0])
    protected = {
        grid.world_to_cell(*start_xy),
        grid.world_to_cell(*target_xy),
    }
    if not grid.in_bounds(i, j) or (i, j) in protected or bool(grid.occ[j, i]):
        return False
    grid.occ[j, i] = True
    grid.sdf = nav_grid._two_sided_sdf(grid.occ, grid.cell)
    return True


def camera_safe_navigation_path(
    scene,
    rig,
    start_xy,
    target_xy,
    target_yaw,
    *,
    start_yaw=None,
    gait_profile=None,
    gait_cum_dist: float = 0.0,
    arm_idle=None,
    path_override=None,
    camera_frame_offset: int = 0,
    additional_excluded_roots=(),
    additional_open_leaf_boxes=(),
    require_human_clear: bool = False,
):
    """Return a narrow-clearance path only when its real gait keeps the FPV eye clear."""
    import nav_grid
    import nav_planner
    from goal_schema import PrimitiveFailure

    if arm_idle is None:
        arm_idle = getattr(rig, "arm_rest", rig.arm_grasp)
    excluded_roots = tuple(
        dict.fromkeys(
            (
                *_kinematically_open_leaf_roots(scene),
                *(str(path) for path in additional_excluded_roots),
            )
        )
    )
    first_path = [tuple(point) for point in path_override] if path_override else None
    for clearance_m in nav_grid.EGOCENTRIC_NAV_INFLATION_LADDER:
        grid = nav_grid.egocentric_passable_view(
            scene.nav_grid,
            scene,
            clearance_m=clearance_m,
            additional_open_leaf_boxes=additional_open_leaf_boxes,
        )
        if grid is None:
            LOG.info("[camera-nav] precise fallback grid is unavailable")
            return None
        for attempt in range(_CAMERA_ROUTE_REPLAN_ATTEMPTS):
            if first_path is not None:
                path, first_path = first_path, None
            else:
                try:
                    path = nav_planner.plan_path(
                        grid, tuple(start_xy), tuple(target_xy)
                    )
                except PrimitiveFailure:
                    LOG.info(
                        "[camera-nav] no route at %.0fmm clearance",
                        clearance_m * 1000.0,
                    )
                    break
            track = _path_pose_track(
                path,
                float(target_yaw),
                nav_grid=grid,
                start_yaw=start_yaw,
            )
            if not track:
                track = [(tuple(start_xy), float(target_yaw), 0.0)]
            report = _route_trajectory_report(
                scene,
                rig,
                arm_idle,
                track,
                float(scene.floor_z),
                excluded_roots,
                gait_profile=gait_profile,
                gait_cum_dist=gait_cum_dist,
                require_camera_check=True,
                camera_frame_offset=camera_frame_offset,
                additional_boxes=additional_open_leaf_boxes,
            )
            camera_clear = bool(report["camera_proxy_clear"])
            human_clear = bool(report.get("human_clean", False))
            if camera_clear and (human_clear or not require_human_clear):
                LOG.warning(
                    "[camera-nav] accepted %.0fmm-clearance route with %d waypoint(s) "
                    "after %d repair(s)",
                    clearance_m * 1000.0,
                    len(path),
                    attempt,
                )
                return {
                    "path": [(float(x), float(y)) for x, y in path],
                    "clearance_m": float(clearance_m),
                    "report": report,
                }
            if camera_clear and require_human_clear:
                LOG.info(
                    "[camera-nav] rejected %.0fmm-clearance route: body intersects scene geometry",
                    clearance_m * 1000.0,
                )
                break
            LOG.info(
                "[camera-nav] rejected %.0fmm-clearance route: eye hits=%s",
                clearance_m * 1000.0,
                report["camera_proxy_hits"][:3],
            )
            if not _avoid_camera_hit_cell(
                grid,
                track,
                report,
                tuple(start_xy),
                tuple(target_xy),
            ):
                break
    return None


def _remember_egocentric_fallback(
    fallback,
    stand_xy,
    yaw,
    rank_distance,
    *,
    worst_frac,
    saturated,
    hit_frames,
    reason,
    force_straight: bool = False,
    path=None,
) -> None:
    if fallback is None:
        return
    score = (
        bool(saturated),
        float(worst_frac),
        int(hit_frames),
        float(rank_distance),
    )
    if fallback.get("score") is None or score < fallback["score"]:
        fallback.update(
            {
                "score": score,
                "stand": ((float(stand_xy[0]), float(stand_xy[1])), float(yaw)),
                "reason": str(reason),
                "force_straight": bool(force_straight),
                "path": (
                    [(float(x), float(y)) for x, y in path]
                    if path is not None
                    else None
                ),
            }
        )


def _stand_plan_key(stand_xy, yaw) -> tuple[float, float, float]:
    return (
        round(float(stand_xy[0]), 4),
        round(float(stand_xy[1]), 4),
        round(float(yaw), 3),
    )


def _remember_validated_path(fallback, stand_xy, yaw, path) -> None:
    if fallback is None or path is None:
        return
    fallback.setdefault("_validated_paths", {})[_stand_plan_key(stand_xy, yaw)] = [
        (float(x), float(y)) for x, y in path
    ]


def _camera_proxy_pose_is_clear(scene, joints, excluded_roots=()) -> bool:
    import smplh_camera

    collision_world = getattr(scene, "cworld", None)
    sphere_overlap = getattr(collision_world, "sphere_overlap", None)
    if sphere_overlap is None:
        return False
    eye, _aim = smplh_camera.fpv_eye_and_aim(
        joints,
        "walk",
        np.zeros(3, dtype=np.float64),
    )
    return (
        sphere_overlap(
            eye,
            _CAMERA_CLEAR_RADIUS_M,
            exclude_owners=tuple(excluded_roots),
        )
        is None
    )


def _candidate_trajectory_is_acceptable(
    ph,
    stage,
    rig,
    scene,
    arm_idle,
    start_xy,
    stand_xy,
    yaw,
    stand_z,
    scene_paths,
    prefilter_boxes,
    nav_grid,
    j_idle,
    j_reach,
    route_exclude_owners=(),
    start_yaw=None,
    gait_profile=None,
    gait_cum_dist: float = 0.0,
    egocentric_fallback=None,
    rank_distance: float = 0.0,
    force_straight: bool = False,
) -> bool:
    """Cheap bounded sweep of route, arrival turn and idle-to-reach transition for one stand."""
    if getattr(scene, "cworld", None) is None:
        return True
    import action_check

    camera_only_policy = action_check.uses_egocentric_camera_collision_policy(scene)
    camera_probe_available = (
        getattr(getattr(scene, "cworld", None), "sphere_overlap", None) is not None
    )
    excluded_roots = tuple(str(root).rstrip("/") for root in route_exclude_owners)
    acceptable = True
    fallback_grade = None
    path = None
    if nav_grid is not None and start_xy is not None:
        import nav_planner
        from goal_schema import PrimitiveFailure

        if force_straight:
            path = [tuple(start_xy), tuple(stand_xy)]
        else:
            try:
                path = nav_planner.plan_path(nav_grid, start_xy, stand_xy)
            except PrimitiveFailure:
                return False
        track = _path_pose_track(
            path,
            yaw,
            nav_grid=nav_grid,
            start_yaw=start_yaw,
        )
        report = _route_trajectory_report(
            scene,
            rig,
            arm_idle,
            track,
            stand_z,
            excluded_roots,
            gait_profile=gait_profile,
            gait_cum_dist=gait_cum_dist,
            require_camera_check=force_straight or camera_only_policy,
        )
        if (
            (camera_only_policy or force_straight)
            and report["camera_probe_available"]
            and not report["camera_proxy_clear"]
        ):
            return False
        if force_straight and not report["camera_probe_available"]:
            return False
        if force_straight or not bool(report["render_pass"]):
            index = report["human_egocentric_worst_frac_frame"]
            if index is None:
                index = report["human_worst_frac_frame"]
            index = 0 if index is None else int(index)
            xy, route_yaw, gait_m = track[index]
            cell = nav_grid.world_to_cell(*xy)
            clearance_mm = float(nav_grid.sdf[cell[1], cell[0]]) * 1000.0
            avoided = 0
            if (
                not force_straight
                and gait_m > 0.0
                and hasattr(nav_grid, "transit_avoid")
            ):
                import nav_grid as _nav_grid

                avoided = _nav_grid.add_transit_avoidance(
                    nav_grid,
                    xy,
                    _nav_grid.TRANSIT_COLLISION_AVOID_RADIUS_M,
                )
            LOG.info(
                "[stand-trajectory] route reject stand=(%.2f,%.2f) at "
                "xy=(%.2f,%.2f) yaw=%.0f clearance=%.0fmm ego_hits=%d/%d "
                "worst=%s grade=%s avoid_cells=+%d",
                stand_xy[0],
                stand_xy[1],
                xy[0],
                xy[1],
                route_yaw,
                clearance_mm,
                report["human_egocentric_hit_frames"],
                report["human_sampled"],
                report["human_egocentric_worst_frac_desc"],
                report["render_grade_reason"],
                avoided,
            )
            if not bool(report["camera_proxy_clear"]):
                return False
            acceptable = False
            fallback_grade = {
                "worst_frac": report["human_egocentric_worst_frac"],
                "saturated": report["human_egocentric_pen_saturated"],
                "hit_frames": report["human_egocentric_hit_frames"],
                "reason": report["render_grade_reason"],
                "force_straight": force_straight,
            }
    if j_reach is None or (isinstance(j_reach, (list, tuple)) and not j_reach):
        if not acceptable:
            _remember_egocentric_fallback(
                egocentric_fallback,
                stand_xy,
                yaw,
                rank_distance,
                **fallback_grade,
                path=path,
            )
        else:
            _remember_validated_path(
                egocentric_fallback,
                stand_xy,
                yaw,
                path,
            )
        return acceptable
    if (
        camera_only_policy
        and camera_probe_available
        and not _camera_proxy_pose_is_clear(
            scene,
            j_idle,
            excluded_roots,
        )
    ):
        return False
    reach_sequence = list(j_reach) if isinstance(j_reach, (list, tuple)) else [j_reach]
    previous = np.asarray(j_idle)
    for reach_index, reach_joints in enumerate(reach_sequence):
        target = np.asarray(reach_joints)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            joints = previous + (target - previous) * fraction
            if (
                camera_only_policy
                and camera_probe_available
                and not _camera_proxy_pose_is_clear(
                    scene,
                    joints,
                    excluded_roots,
                )
            ):
                return False
            hits = validate_placement(
                ph,
                stage,
                joints,
                scene_paths,
                joint_subset=_CORE_J,
                prefilter_boxes=prefilter_boxes,
            )
            if not hits:
                continue
            grade = _stand_pose_grade(scene, joints, collision_pairs=hits)
            if grade is None or not bool(grade["passes"]):
                if grade is None or not _camera_proxy_pose_is_clear(
                    scene,
                    joints,
                    excluded_roots,
                ):
                    return False
                acceptable = False
                pose_grade = {
                    "worst_frac": grade["egocentric_worst_frac"],
                    "saturated": grade["egocentric_saturated"],
                    "hit_frames": 1,
                    "reason": grade["reason"],
                    "force_straight": force_straight,
                }
                if fallback_grade is None or (
                    bool(pose_grade["saturated"]),
                    float(pose_grade["worst_frac"]),
                ) > (
                    bool(fallback_grade["saturated"]),
                    float(fallback_grade["worst_frac"]),
                ):
                    fallback_grade = pose_grade
                LOG.info(
                    "[stand-trajectory] reach reject stand=(%.2f,%.2f) "
                    "segment=%d blend=%.2f hits=%s grade=%s",
                    stand_xy[0],
                    stand_xy[1],
                    reach_index,
                    fraction,
                    hits[:3],
                    None if grade is None else grade["reason"],
                )
        previous = target
    if not acceptable:
        _remember_egocentric_fallback(
            egocentric_fallback,
            stand_xy,
            yaw,
            rank_distance,
            **fallback_grade,
            path=path,
        )
    else:
        _remember_validated_path(
            egocentric_fallback,
            stand_xy,
            yaw,
            path,
        )
    return acceptable


def _plan_stand_reach(
    semantic_executor,
    rig,
    scene,
    contact,
    stand_xy,
    yaw,
    stand_z,
    tol,
    *,
    strict: bool,
    ik_budget=None,
    action_state=None,
):
    """Run the same body-clearance reach plan used by the action producer."""
    if action_state is not None and semantic_executor._held_right_hand(action_state):
        preview_state = copy.copy(action_state)
        preview_state.root_xy = (float(stand_xy[0]), float(stand_xy[1]))
        preview_state.root_yaw = float(yaw)
        preview_state.walk_end_yaw = float(yaw)
        preview_state.stand_z = float(stand_z)
        return semantic_executor._openable_reach_plan(
            rig,
            scene,
            preview_state,
            contact,
            tol,
            ik_budget=ik_budget,
        )
    return semantic_executor.plan_reach(
        rig,
        rig.arm_grasp,
        contact,
        stand_xy,
        yaw,
        stand_z,
        tol,
        floor_z=scene.floor_z,
        enabled=True,
        clear_fn=semantic_executor.body_clear_fn(scene, stand_xy, yaw, scene.floor_z),
        allow_colliding_fallback=False,
        ik_budget=ik_budget,
    )


def _plan_stand_reach_trajectory(
    semantic_executor,
    rig,
    scene,
    contacts,
    stand_xy,
    yaw,
    stand_z,
    tol,
    *,
    ik_budget=None,
):
    """Plan every contact keyframe from one stand using the executor's reach model."""
    return semantic_executor.plan_reach_many(
        rig,
        rig.arm_grasp,
        contacts,
        stand_xy,
        yaw,
        stand_z,
        tol,
        floor_z=scene.floor_z,
        enabled=True,
        clear_fn=semantic_executor.body_clear_fn(scene, stand_xy, yaw, scene.floor_z),
        allow_colliding_fallback=False,
        ik_budget=ik_budget,
    )


_COMPLETE_TRAJECTORY_IK_SOLVES_PER_POINT = 16
_COMPLETE_TRAJECTORY_CANDIDATES_PER_PHASE = 2


def _trajectory_ik_phase_budget(contacts, *, complete: bool) -> int:
    """Reserve enough IK work to finish two relaxed multi-contact candidates."""
    if not complete or contacts is None:
        return 0
    point_count = len(contacts)
    if point_count <= 1:
        return 0
    return (
        point_count
        * _COMPLETE_TRAJECTORY_IK_SOLVES_PER_POINT
        * _COMPLETE_TRAJECTORY_CANDIDATES_PER_PHASE
    )


def _openable_phase_uses_content_geometry(
    *,
    require_openable_sweep_clearance: bool,
    openable_action_phase: bool,
    openable_action_kind: str | None,
) -> bool:
    if openable_action_kind in {"insert", "extract"}:
        return True
    return bool(
        not openable_action_phase
        and openable_action_kind is None
        and not require_openable_sweep_clearance
    )


def _stand_search_radius(scene, key, contact_xy) -> float:
    import semantic_executor as se

    radius = REACH_ENVELOPE_M + 0.15
    anchors = se.openable_stand_anchors(scene, key) if key is not None else ()
    if not anchors:
        return radius
    return max(radius, *(math.dist(anchor, contact_xy) + 0.25 for anchor in anchors))


def _search_reachable_stand(
    ph,
    stage,
    rig,
    scene,
    key,
    contact,
    stand_z,
    arm_idle,
    scene_paths,
    prefilter_boxes,
    nav_grid=None,
    start_xy=None,
    rank_from=None,
    fixed_stand_xy=None,
    seed_stand_xy=None,
    flood_grid=None,
    contact_override=None,
    contact_overrides=None,
    shelf_gap_ceiling=None,
    stand_filter=None,
    route_gate=False,
    trajectory_gate=False,
    route_exclude_owners=(),
    retry_cells=None,
    strict=None,
    ik_budget=None,
    fine_ik_budget=None,
    router_ik_budget=None,
    fine_retry=False,
    skip_cells=None,
    trajectory_probe_count=None,
    router_trajectory_probe_count=None,
    trajectory_probe_limit=None,
    start_yaw=None,
    gait_profile=None,
    gait_cum_dist: float = 0.0,
    reach_tol: float | None = None,
    require_reach: bool = True,
    require_openable_sweep_clearance: bool = True,
    openable_action_phase: bool = False,
    openable_action_kind: str | None = None,
    front_access_standoff_m: float | None = None,
    egocentric_fallback=None,
    allow_egocentric_fallback_return: bool = True,
    allow_force_straight: bool = False,
    allow_expensive_retries: bool = True,
    candidate_validator=None,
    action_state=None,
    facing_override=None,
    complete_contact_trajectory: bool = False,
):
    """Scan the free floor AROUND ``contact`` for a stand that is FULLY clean + reachable, closest first.

    Candidates: the nav grid's reachable-from-start free cells when a grid is given, else a GEOMETRIC RING
    around the contact (so this works in pass-1 with NO grid). Each candidate faces the contact and must pass
    ALL of: (a) idle body-core non-penetration, (b) the POSE-PLANNER can reach the contact (lean/squat IK),
    (c) the arm-EXTENDED reach keyframe is non-penetrating (no torso-in / arm-through-furniture), and
    (d) nav-walkable from start (only when a grid is given). Returns the CLOSEST such (fx,fy),yaw, else None.
    This is the general recovery the fixed ladder needs: a cornered object's clean stand is on the OPEN side
    the ladder never tries.

    ``flood_grid`` (optional) is the grid the (d) walkability flood + connectivity pre-cull run on, while
    the candidate CELLS still come from ``nav_grid`` (the precise occ). Defaults to ``nav_grid``. The runtime
    picker passes the DOORS-OPEN passable view here so an object behind a CLOSED door still gets a stand
    (Change 2a), while the door-approach pick passes the real grid so it must reach the handle WITHOUT first
    opening that door. ``contact_override`` (world xyz) forces the (c) arm-extended keyframe's reach target
    (the door HANDLE) so the gated pose matches what the OpenDoor producer grips.
    ``contact_overrides`` applies the same gate to every point of a contact trajectory; all points must be
    reachable from the returned stand.

    ``stand_filter(x, y) -> bool`` (optional) is a HARD pre-cull on WHERE the human may stand, applied to
    the raw candidates before any gate. It carries constraints that are not about reach at all -- a stand
    must be on the READABLE side of a sign -- which reachability ranking cannot express and would
    otherwise silently override. Culling before the gates also stops the soft last-resort fallback from
    smuggling a filtered-out stand back in.

    ``route_gate`` (default False = byte-identical to every caller that does not ask for it) adds one
    more requirement to the WINNER only: ``nav_planner.plan_path`` must actually find a route to it.
    Gate (b0) accepts a stand when a start-reachable free cell lies within ``_shuffle_tol``, which is
    the right criterion for a stand hugging the counter it works at but disagrees with the router
    whenever that shuffle bridges a gap the router will not walk; the caller then gets a stand whose
    ``navigate_to`` fails ``path_blocked``. Gating the WINNER rather than gate (b0) is deliberate: b0
    runs on every one of the hundreds of 0.20 m-stride candidates and an A* there would dominate the
    scan, whereas a winner is rejected 1-3 times per command in practice.

    ``require_reach=False`` is reserved for a navigation leg whose immediate action is visual-only. It
    keeps the floor, body, camera, arrival-trajectory, cue-facing, and route gates, but does not reject a
    viewing position merely because an arm cannot touch the observed entity.

    ``retry_cells``, ``fine_retry``, and ``skip_cells`` are internal recursion state. The fine retry scans
    the 0.10 m cells omitted by the normal 0.20 m pass with an IK tranche reserved by the top-level call.
    ``retry_cells`` restricts a pass to named stand points and swaps gate (b0)'s proxy for the real router.
    Coarse and fine share a total trajectory counter, but fine receives a higher phase limit so coarse
    candidates cannot consume the entire trajectory budget. ``allow_expensive_retries=False`` is for a
    speculative preflight that may safely return no candidate and defer to the later hard runtime gate;
    it never changes which candidates the normal runtime accepts."""
    import action_check
    import semantic_executor as _se

    strict = (
        bool(getattr(scene, "strict_acceptance", False))
        if strict is None
        else bool(strict)
    )
    camera_only_policy = action_check.uses_egocentric_camera_collision_policy(scene)
    camera_probe_available = (
        getattr(getattr(scene, "cworld", None), "sphere_overlap", None) is not None
    )
    camera_policy_verified = camera_only_policy and camera_probe_available
    if egocentric_fallback is None:
        egocentric_fallback = {}
    trajectory_contacts = None
    if contact_overrides is not None:
        raw_contacts = np.asarray(contact_overrides, dtype=np.float64).reshape(-1, 3)
        trajectory_contacts = []
        seen_contacts: set[tuple[float, float, float]] = set()
        for point in raw_contacts:
            point_key = tuple(round(float(axis), 6) for axis in point)
            if point_key in seen_contacts:
                continue
            seen_contacts.add(point_key)
            trajectory_contacts.append(point)
        if not trajectory_contacts:
            trajectory_contacts = None
    trajectory_phase_budget = _trajectory_ik_phase_budget(
        trajectory_contacts,
        complete=complete_contact_trajectory,
    )
    _can_fine_retry = (
        allow_expensive_retries
        and not strict
        and not fine_retry
        and retry_cells is None
        and nav_grid is not None
        and start_xy is not None
    )
    if ik_budget is None:
        total_ik = _se.MAX_IK_SOLVES_PER_STAND_SEARCH
        if _can_fine_retry:
            fine_limit = min(
                _se.RESERVED_IK_SOLVES_FOR_FINE_STAND_SEARCH,
                total_ik,
            )
            router_limit = min(
                _se.RESERVED_IK_SOLVES_FOR_ROUTER_STAND_RETRY,
                max(0, total_ik - fine_limit),
            )
            coarse_limit = total_ik - fine_limit - router_limit
            if trajectory_phase_budget:
                coarse_limit = max(coarse_limit, trajectory_phase_budget)
                fine_limit = max(fine_limit, trajectory_phase_budget)
                router_limit = max(router_limit, trajectory_phase_budget)
            ik_budget = _se.IKSearchBudget(coarse_limit)
            if fine_ik_budget is None:
                fine_ik_budget = _se.IKSearchBudget(fine_limit)
            if router_ik_budget is None:
                router_ik_budget = _se.IKSearchBudget(router_limit)
        else:
            ik_budget = _se.IKSearchBudget(max(total_ik, trajectory_phase_budget))
    if fine_ik_budget is None:
        fine_ik_budget = ik_budget
    if router_ik_budget is None:
        router_ik_budget = ik_budget
    if trajectory_probe_count is None:
        trajectory_probe_count = [0]
    if router_trajectory_probe_count is None:
        router_trajectory_probe_count = [0]
    if trajectory_probe_limit is None:
        trajectory_probe_limit = (
            MAX_STAND_TRAJECTORY_PROBES
            if strict or fine_retry or nav_grid is None or start_xy is None
            else MAX_COARSE_STAND_TRAJECTORY_PROBES
        )
    cx0, cy0 = float(contact[0]), float(contact[1])
    # A SHELVED container (>=2 interior spaces): judge each candidate stand by whether it can reach ANY space
    # via the SHELVED place's OWN model (best_shelf_reach = plan_reach_shelf per space), so "stand reachable"
    # == "the place can put a point into a reachable space" -- NOT plan_reach to a single mid drop_z the place
    # never targets. None -> non-shelf target keeps the single-contact plan_reach path (byte-identical).
    _shelf_spaces = None
    try:
        _ke = scene.resolve(key) if key is not None else None
        _kaff = (getattr(_ke, "target", None) or _ke) if _ke is not None else None
        _sp = getattr(_kaff, "spaces", None)
        if (
            contact_override is None
            and trajectory_contacts is None
            and _sp
            and len(_sp) >= 2
        ):
            _shelf_spaces = _sp
    except Exception:  # noqa: BLE001
        _shelf_spaces = None
    # The pose-planner's lean extends reach past the flat cap. Front-opening units may put the
    # actual object farther inside than their safe door-side stance, so include that measured span.
    _max_r = _stand_search_radius(scene, key, (cx0, cy0))
    cands: list[tuple[float, float, float]] = []
    # DIAGNOSTIC (general, scene-agnostic): tally where every candidate stand dies so a failed search is
    # attributable to occ (real furniture footprint) / dist (out of reach ring) / idle_pen (body clips at
    # rest) / navdisc (no walkable floor within the shuffle tol) / ik (arm cannot reach) / arm_pen (clips) /
    # arrival_pen (the short stable final approach clips deeply at the otherwise-clean stand).
    _dbg = {
        "cells": 0,
        "occ": 0,
        "dist": 0,
        "idle_pen": 0,
        "navdisc": 0,
        "ik": 0,
        "arm_pen": 0,
        "arrival_pen": 0,
        "trajectory_pen": 0,
        "action_pen": 0,
        "camera_pen": 0,
        "filtered": 0,
        "noroute": 0,  # only ever non-zero under route_gate
    }
    # Stand points gate (b0) turned away, kept for the ROUTER RETRY at the end of the function. Populated
    # only on the first pass; the retry pass leaves it empty so the recursion can never repeat.
    _nav_soft: list[tuple[float, float]] = []
    _soft_fallback = None
    _soft_score = None
    _soft_extra_candidates = 0
    # Flood start->free ONCE (over the inflated passable grid) so the connectivity gate can pre-cull an
    # islanded candidate BEFORE the expensive reach-IK, instead of re-flooding per candidate at the end.
    # The flood/connectivity grid may DIFFER from the candidate grid (Change 2a): the runtime picker floods
    # the DOORS-OPEN passable view (reach an object behind a closed door) while candidate cells stay on the
    # precise ``nav_grid``. Both views share x0/y0/cell/nx/ny, so the mask indices line up 1:1.
    _reach_mask = None
    _flood_grid = flood_grid if flood_grid is not None else nav_grid
    if nav_grid is not None and start_xy is not None:
        import nav_grid as _ng

        _reach_mask = _ng.flood_free(
            _flood_grid,
            _flood_grid.world_to_cell(float(start_xy[0]), float(start_xy[1])),
        )
    if nav_grid is not None and start_xy is not None:
        # STAND candidates come from the PRECISE (un-inflated) occupancy so the human may stand right up
        # against furniture (hug the table to reach an object on it); whether the body actually clips is
        # decided below by the precise idle/arm gate, NOT by the nav inflation. The INFLATED nav grid is used
        # ONLY for the (d) walkability check -- a stand at a furniture edge still connects via the
        # ``_shuffle_tol`` shuffle. Falls back to the inflated occ when no precise raster is cached.
        cw = getattr(scene, "cworld", None)
        p_occ = getattr(cw, "_precise_occ", None) if cw is not None else None
        if p_occ is None or p_occ.shape != (nav_grid.ny, nav_grid.nx):
            p_occ = nav_grid.occ
        ci, cj = nav_grid.world_to_cell(cx0, cy0)
        rad = int(math.ceil(_max_r / nav_grid.cell)) + 1
        # Sample candidates on a ~0.20 m stride (every 2nd 0.10 m cell), NOT every cell, to cut the expensive
        # reach-IK solves ~4x on hard/unreachable objects (the nav grid itself stays 0.10 m -- paths +
        # inflation unaffected). This is a deliberate completeness/speed TRADE, not lossless: a valid-stand
        # strip 0.10-0.20 m wide (a tightly cornered object reachable from only a thin sliver of floor) can
        # fall between samples -> a rare, DETERMINISTIC false-NONE. It is one-directional (never a false
        # FOUND -- every sampled cell still passes the full gate battery), acceptable under the RL soft-fail
        # contract. The knob is integer cells: at cell 0.10 the only options are stride 1 (0.10 m, complete,
        # slow) or stride 2 (0.20 m, 4x faster) -- 0.15 rounds to stride 1. If a specific reachable object is
        # ever wrongly declared NONE, force stride 1 for it rather than trusting a sub-cell value.
        _stride = (
            1 if strict or fine_retry else max(1, int(round(0.20 / nav_grid.cell)))
        )
        for dj in range(-rad, rad + 1, _stride):
            for di in range(-rad, rad + 1, _stride):
                i, j = ci + di, cj + dj
                if not (0 <= i < nav_grid.nx and 0 <= j < nav_grid.ny):
                    continue
                _dbg["cells"] += 1
                if p_occ[j, i] or (
                    nav_grid.floor_support is not None
                    and not nav_grid.floor_support[j, i]
                ):
                    _dbg["occ"] += 1
                    continue
                wx, wy = nav_grid.cell_center(i, j)
                d = math.hypot(wx - cx0, wy - cy0)
                if 0.28 <= d <= _max_r:
                    cands.append((d, wx, wy))
                else:
                    _dbg["dist"] += 1
    else:
        for r in (0.35, 0.45, 0.55, 0.65, 0.75, _max_r):
            for deg in range(0, 360, 20):
                a = math.radians(deg)
                cands.append((r, cx0 + r * math.cos(a), cy0 + r * math.sin(a)))
    fixed_stand = _finite_xy(fixed_stand_xy, exact=True)
    if fixed_stand is not None:
        fixed_x, fixed_y = float(fixed_stand[0]), float(fixed_stand[1])
        cands = [(math.hypot(fixed_x - cx0, fixed_y - cy0), fixed_x, fixed_y)]
        _dbg["cells"] = 1
        _dbg["occ"] = 0
        _dbg["dist"] = 0
    else:
        seed_stand = _finite_xy(seed_stand_xy, exact=True)
        if seed_stand is not None:
            seed_x, seed_y = float(seed_stand[0]), float(seed_stand[1])
            seed_distance = math.hypot(seed_x - cx0, seed_y - cy0)
            seed_is_free = True
            if nav_grid is not None and start_xy is not None:
                seed_i, seed_j = nav_grid.world_to_cell(seed_x, seed_y)
                seed_is_free = (
                    0 <= seed_i < nav_grid.nx
                    and 0 <= seed_j < nav_grid.ny
                    and not p_occ[seed_j, seed_i]
                    and (
                        nav_grid.floor_support is None
                        or nav_grid.floor_support[seed_j, seed_i]
                    )
                )
                _dbg["cells"] += 1
                if not seed_is_free:
                    _dbg["occ"] += 1
            if seed_is_free:
                if 0.28 <= seed_distance <= _max_r:
                    cands.append((seed_distance, seed_x, seed_y))
                else:
                    _dbg["dist"] += 1
    if stand_filter is not None:
        _n_all = len(cands)
        cands = [c for c in cands if stand_filter(c[1], c[2])]
        _dbg["filtered"] = _n_all - len(cands)
    if key is not None:
        _n_all = len(cands)
        cands = [
            candidate
            for candidate in cands
            if _se.openable_stand_is_visually_clear(
                scene,
                key,
                (candidate[1], candidate[2]),
                include_sweep=require_openable_sweep_clearance,
                action_phase=(
                    False
                    if _openable_phase_uses_content_geometry(
                        require_openable_sweep_clearance=require_openable_sweep_clearance,
                        openable_action_phase=openable_action_phase,
                        openable_action_kind=openable_action_kind,
                    )
                    else (
                        True
                        if openable_action_phase or openable_action_kind is not None
                        else None
                    )
                ),
                action_kind=openable_action_kind,
                standoff_override=front_access_standoff_m,
            )
        ]
        _dbg["filtered"] += _n_all - len(cands)
    if skip_cells:
        cands = [c for c in cands if (round(c[1], 2), round(c[2], 2)) not in skip_cells]
    if retry_cells is not None:
        # ROUTER RETRY pass: only the points gate (b0)'s proxy turned away on the first pass are worth
        # re-testing, and restricting to them is what keeps the A* count to those few rather than to every
        # candidate in the ring (which is exactly the cost the (b0) pre-cull exists to avoid).
        cands = [c for c in cands if (round(c[1], 2), round(c[2], 2)) in retry_cells]
    if rank_from is not None:
        # RUNTIME pick: rank candidates NEAREST to the human's CURRENT position (rank_from), not the
        # contact -- so the human walks to the closest valid stand from where it stands now (no detour to a
        # stand pre-picked closest to the OBJECT). c = (dist_to_contact, wx, wy).
        _rfx, _rfy = float(rank_from[0]), float(rank_from[1])
        cands.sort(key=lambda c: math.hypot(c[1] - _rfx, c[2] - _rfy))
    else:
        cands.sort(key=lambda c: c[0])  # closest first (least-effort reach)
    _tol = (
        _se._reach_tolerance(scene)
        if reach_tol is None
        else min(max(0.0, float(reach_tol)), _se._OPENABLE_REACH_TOL_M)
    )
    seen: set = set()
    for _d, wx, wy in cands:
        _candidate_force_straight = False
        if _soft_fallback is not None:
            if _soft_extra_candidates >= MAX_CLEAN_SEARCH_EXTRA_CANDIDATES:
                break
            _soft_extra_candidates += 1
        kxy = (round(wx, 2), round(wy, 2))
        if kxy in seen:
            continue
        seen.add(kxy)
        face = contact if facing_override is None else facing_override
        yaw = math.degrees(math.atan2(float(face[0]) - wx, -(float(face[1]) - wy)))
        _arm_hits: list = []
        # (b0) walkability pre-cull BEFORE the expensive reach-IK: drop a stand with no start-reachable free
        # cell within ``_shuffle_tol`` (islanded / behind furniture) now, so we never build or collision-test
        # a human pose for it. Same criterion as the trailing _stand_connected gate, but the flood is
        # precomputed once above.
        #
        # This is a PROXY, not an impossibility test, and it is DOWNRANKING here rather than vetoing. It
        # asks for a start-reachable free cell within ``_shuffle_tol`` (0.25 m at the 0.10 m inflation)
        # while the router it stands in for, ``nav_planner.plan_path``, snaps a blocked goal to any free
        # cell within ``_GOAL_SNAP_R`` (0.60 m) before routing -- so the proxy is strictly the STRICTER of
        # the two and can reject stands the router would happily walk to. When it rejects EVERY candidate
        # the whole command is lost without the router ever being asked once. Measured over the cb1/cb2
        # baseline: 20 `path_blocked` occurrences across 5 rows (`T2-HDU-01` 10, `T2-BW1-01` 4,
        # `T4-WAI0G-01` 4, `T3-ASIAN-03` 1, `T3-CHEM-01` 1) whose scan logs `navdisc` == `cands`, i.e. 100 %
        # of candidates turned away by the proxy. So: bank the rejects and re-test them against the real
        # router at the end, but ONLY on the path that would otherwise return None.
        if _reach_mask is not None:
            _b0_ok = (
                _stand_routes(_flood_grid, start_xy, wx, wy)
                if retry_cells is not None
                else _reachable_free_within(_flood_grid, _reach_mask, wx, wy)
            )
            if not _b0_ok:
                _dbg["navdisc"] += 1
                if retry_cells is None:
                    _nav_soft.append((float(wx), float(wy)))
                    continue
                if (strict and not camera_policy_verified) or not allow_force_straight:
                    continue
                _candidate_force_straight = True
        # (a) idle body-core non-penetration -- SOFT. Recorded with its depth, NOT vetoed here; a candidate
        # that clips only at rest still runs the HARD gates below and, if it passes them, is kept as a
        # last-resort fallback (see the ``_soft`` block after the loop). "The body brushes furniture while
        # standing" is a gradable picture flaw -- ``action_check.grade_render`` scores it and explicitly
        # treats even ``dirty`` as a FLAG, never a veto -- so it must not be the thing that decides an
        # object cannot be picked up AT ALL. Measured in bt_chemlab: it killed 42 of 68 candidates (62%)
        # across the three commands that failed, while the arm-reach gate rejected ZERO -- i.e. nothing was
        # actually out of reach, the body just could not stand perfectly clean anywhere near it.
        j_idle = human_proxy_J(rig, arm_idle, (wx, wy), yaw, stand_z)
        if camera_policy_verified and not _camera_proxy_pose_is_clear(
            scene,
            j_idle,
            route_exclude_owners,
        ):
            _dbg["camera_pen"] += 1
            continue
        _idle_hits = validate_placement(
            ph,
            stage,
            j_idle,
            scene_paths,
            joint_subset=_CORE_J,
            prefilter_boxes=prefilter_boxes,
        )
        if strict and not camera_policy_verified:
            _idle_hits = _severe_body_hits(scene, j_idle, _idle_hits)
        if _idle_hits:
            _dbg["idle_pen"] += 1
        if require_reach and ik_budget.exhausted:
            LOG.warning(
                "[stand-scan] %s: stopped after the %d-solve IK search budget",
                key,
                ik_budget.limit,
            )
            break
        # (b) can the arm reach from here? A SHELVED container mirrors the place EXACTLY (best_shelf_reach over
        # the interior spaces) -> a stand judged reachable is one the place can put a point INTO a reachable
        # space, NOT one where a single mid drop_z is reachable but the chosen space is not. Non-shelf targets
        # keep the single-contact plan_reach (byte-identical).
        _exact_rest_crouch = bool(
            _kaff is not None
            and (getattr(_kaff, "meta", None) or {}).get(
                "content_insert_exact_rest_crouch"
            )
        )
        if not require_reach:
            reach_points = []
        elif trajectory_contacts is not None:
            reach_plans = _plan_stand_reach_trajectory(
                _se,
                rig,
                scene,
                trajectory_contacts,
                (wx, wy),
                yaw,
                stand_z,
                _tol,
                ik_budget=ik_budget,
            )
            if any(plan.get("collision_blocked") for plan in reach_plans):
                _dbg["arm_pen"] += 1
                continue
            if any(not plan["reachable"] for plan in reach_plans):
                _dbg["ik"] += 1
                continue
            reach_points = list(zip(trajectory_contacts, reach_plans))
        elif shelf_gap_ceiling is not None:
            rp = _se.plan_reach_shelf(
                rig,
                rig.arm_grasp,
                contact,
                (wx, wy),
                yaw,
                stand_z,
                _tol,
                float(shelf_gap_ceiling),
                floor_z=scene.floor_z,
                enabled=True,
                clear_fn=_se.body_clear_fn(
                    scene,
                    (wx, wy),
                    yaw,
                    scene.floor_z,
                    require_body_clear=_exact_rest_crouch,
                ),
                allow_colliding_fallback=False,
                ik_budget=ik_budget,
                exact_rest_crouch=_exact_rest_crouch,
            )
            if rp.get("collision_blocked"):
                _dbg["arm_pen"] += 1
                continue
            if not rp["reachable"]:
                _dbg["ik"] += 1
                continue
            reach_points = [(contact, rp)]
        elif _shelf_spaces is not None:
            _low_front_shelf = bool(
                _kaff is not None and _se._low_front_openable(scene, _kaff)
            )
            _bs = _se.best_shelf_reach(
                rig,
                rig.arm_grasp,
                _shelf_spaces,
                (wx, wy),
                yaw,
                stand_z,
                _tol,
                floor_z=scene.floor_z,
                enabled=True,
                clear_fn=_se.body_clear_fn(
                    scene,
                    (wx, wy),
                    yaw,
                    scene.floor_z,
                    require_body_clear=_exact_rest_crouch,
                ),
                allow_colliding_fallback=False,
                ik_budget=ik_budget,
                minimum_squat=(
                    _se._LOW_FRONT_INSERT_MIN_SQUAT if _low_front_shelf else 0.0
                ),
                exact_rest_crouch=_exact_rest_crouch,
            )
            if _bs is None or not _bs["reachable"]:
                _dbg["ik"] += 1
                continue
            rp = {
                "reachable": True,
                "lean_rad": _bs["lean_rad"],
                "squat": _bs["squat"],
                "arm_pose": _bs.get("arm_pose"),
            }
            reach_points = [
                (_bs["contact"], rp)
            ]  # reachable space point -> arm-pen keyframe target + exclude sphere
        else:
            rp = _plan_stand_reach(
                _se,
                rig,
                scene,
                contact,
                (wx, wy),
                yaw,
                stand_z,
                _tol,
                strict=strict,
                ik_budget=ik_budget,
                action_state=action_state,
            )
            if rp.get("collision_blocked"):
                _dbg["arm_pen"] += 1
                continue
            if not rp["reachable"]:
                _dbg["ik"] += 1
                continue
            reach_points = [(contact, rp)]
        # (c) EVERY arm-extended contact keyframe must be non-penetrating. A multi-point action may use
        # a different lean/squat rung at each point, but its feet remain at this one candidate stand.
        reach_joints = []
        arm_hit_poses = []
        for reach_contact, reach_plan in reach_points:
            jr = _arm_extended_keyframe(
                rig,
                scene,
                key,
                (wx, wy),
                yaw,
                stand_z,
                lean_override=reach_plan["lean_rad"],
                squat_override=reach_plan["squat"],
                contact_override=reach_contact,
                pose_override=reach_plan.get("arm_pose"),
            )
            if jr is None:
                continue
            reach_joints.append(jr)
            if camera_policy_verified and not _camera_proxy_pose_is_clear(
                scene,
                jr,
                route_exclude_owners,
            ):
                _dbg["camera_pen"] += 1
                _arm_hits.append(("fpv_eye", "scene geometry"))
                break
            _crouch = reach_plan["squat"] > 1e-6
            _cj = _CROUCH_CORE_J if _crouch else _CORE_J
            reach_core = validate_placement(
                ph,
                stage,
                jr,
                scene_paths,
                joint_subset=_cj,
                prefilter_boxes=prefilter_boxes,
            )
            core_names = {p[0] for p in reach_core}
            excl = (
                np.asarray(reach_contact, dtype=np.float64),
                _CONTACT_SPHERE_R,
            )
            arm_out = [
                p
                for p in ph.human_scene_overlap(
                    stage, jr, scene_paths, exclude_sphere=excl
                )
                if p[0] not in core_names and not (_crouch and p[0] in _LEG_BONE_NAMES)
            ]
            if strict and not camera_policy_verified:
                reach_core = _severe_body_hits(scene, jr, reach_core)
            if reach_core or arm_out:
                _dbg["arm_pen"] += 1
                _arm_hits.extend(reach_core)
                _arm_hits.extend(arm_out)
                arm_hit_poses.append(jr)
        if any(pair[0] == "fpv_eye" for pair in _arm_hits):
            continue
        arrival_pairs = (
            []
            if _candidate_force_straight
            else _controlled_arrival_hits(
                ph,
                stage,
                rig,
                scene,
                arm_idle,
                (wx, wy),
                yaw,
                stand_z,
                scene_paths,
                prefilter_boxes,
                _flood_grid,
                start_xy,
            )
        )
        if arrival_pairs:
            _dbg["arrival_pen"] += 1
            if not camera_policy_verified:
                continue
        if trajectory_gate:
            if trajectory_probe_count[0] >= trajectory_probe_limit:
                LOG.info(
                    "[stand-trajectory] %s: phase budget %d/%d exhausted; "
                    "deferring remaining candidates to the next search phase",
                    key,
                    trajectory_probe_count[0],
                    trajectory_probe_limit,
                )
                break
            if trajectory_probe_count[0] == PRIMARY_STAND_TRAJECTORY_PROBES:
                LOG.info(
                    "[stand-trajectory] %s: first %d candidates rejected; "
                    "trying one bounded fallback batch",
                    key,
                    PRIMARY_STAND_TRAJECTORY_PROBES,
                )
            trajectory_probe_count[0] += 1
            if not _candidate_trajectory_is_acceptable(
                ph,
                stage,
                rig,
                scene,
                arm_idle,
                start_xy,
                (wx, wy),
                yaw,
                stand_z,
                scene_paths,
                prefilter_boxes,
                _flood_grid,
                j_idle,
                reach_joints,
                route_exclude_owners=route_exclude_owners,
                start_yaw=start_yaw,
                gait_profile=gait_profile,
                gait_cum_dist=gait_cum_dist,
                egocentric_fallback=egocentric_fallback,
                rank_distance=_d,
                force_straight=_candidate_force_straight,
            ):
                _dbg["trajectory_pen"] += 1
                if _reach_mask is not None:
                    _reach_mask = _ng.flood_free(
                        _flood_grid,
                        _flood_grid.world_to_cell(
                            float(start_xy[0]), float(start_xy[1])
                        ),
                    )
                continue
        if candidate_validator is not None and not candidate_validator(
            (float(wx), float(wy)), float(yaw)
        ):
            _dbg["action_pen"] += 1
            continue
        # A render-acceptable contact pose is a FALLBACK, not an immediate winner. Keep the best one and
        # spend only a bounded number of later candidate batches looking for a fully clean stand. This
        # preserves the soft-contact completeness needed by cramped scenes without paying an exhaustive
        # zero-contact search, and it prevents an early marginal candidate from hiding a nearby clean one.
        if _idle_hits or _arm_hits:
            if strict and not camera_policy_verified:
                continue
            _dirty_poses = []
            if _idle_hits:
                _dirty_poses.append(j_idle)
            _dirty_poses.extend(arm_hit_poses)
            _pose_grades = [_stand_pose_grade(scene, joints) for joints in _dirty_poses]
            if any(
                grade is None or not bool(grade["passes"]) for grade in _pose_grades
            ):
                if all(
                    grade is not None
                    and _camera_proxy_pose_is_clear(
                        scene,
                        joints,
                        route_exclude_owners,
                    )
                    for joints, grade in zip(_dirty_poses, _pose_grades)
                ):
                    worst = max(
                        _pose_grades,
                        key=lambda grade: (
                            bool(grade["egocentric_saturated"]),
                            float(grade["egocentric_worst_frac"]),
                        ),
                    )
                    _remember_egocentric_fallback(
                        egocentric_fallback,
                        (wx, wy),
                        yaw,
                        _d,
                        worst_frac=worst["egocentric_worst_frac"],
                        saturated=worst["egocentric_saturated"],
                        hit_frames=1,
                        reason=worst["reason"],
                    )
                continue
            if (
                route_gate
                and _reach_mask is not None
                and start_xy is not None
                and not _stand_routes(_flood_grid, start_xy, wx, wy)
            ):
                _dbg["noroute"] += 1
                continue
            _score = (
                max(float(grade["worst_frac"]) for grade in _pose_grades),
                len(_idle_hits) + len(_arm_hits),
                float(_d),
            )
            if _soft_score is None or _score < _soft_score:
                _soft_score = _score
                _soft_fallback = ((float(wx), float(wy)), float(yaw))
                LOG.info(
                    "[stand-scan] %s: retained render-acceptable %s-contact fallback "
                    "at (%.2f,%.2f), worst_frac=%.3f; checking up to %d later candidates for clean",
                    key,
                    (
                        "idle+arm"
                        if _idle_hits and _arm_hits
                        else ("idle" if _idle_hits else "arm")
                    ),
                    wx,
                    wy,
                    _score[0],
                    MAX_CLEAN_SEARCH_EXTRA_CANDIDATES,
                )
            continue
        # (connectivity was pre-culled at (b0) above; a candidate reaching here is walkable + reachable + clean)
        # (b1) THE ROUTER MUST AGREE, for the winner only. See ``route_gate`` in the docstring.
        if (
            route_gate
            and _reach_mask is not None
            and start_xy is not None
            and not _stand_routes(_flood_grid, start_xy, wx, wy)
        ):
            _dbg["noroute"] += 1
            LOG.info(
                "[stand-scan] %s: (%.2f,%.2f) passed every gate but the ROUTER refuses it; "
                "keeping the scan going instead of handing back a stand navigate_to will fail",
                key,
                wx,
                wy,
            )
            continue
        LOG.info(
            "[stand-scan] %s: cells=%d rej[occ=%d dist=%d filtered=%d] "
            "gate[idle_pen=%d navdisc=%d ik=%d arm_pen=%d arrival_pen=%d trajectory_pen=%d action_pen=%d camera_pen=%d noroute=%d] "
            "cands=%d -> FOUND (%.2f,%.2f)",
            key,
            _dbg["cells"],
            _dbg["occ"],
            _dbg["dist"],
            _dbg["filtered"],
            _dbg["idle_pen"],
            _dbg["navdisc"],
            _dbg["ik"],
            _dbg["arm_pen"],
            _dbg["arrival_pen"],
            _dbg["trajectory_pen"],
            _dbg["action_pen"],
            _dbg["camera_pen"],
            _dbg["noroute"],
            len(cands),
            wx,
            wy,
        )
        return (float(wx), float(wy)), float(yaw)
    if _soft_fallback is not None:
        LOG.warning(
            "[stand-scan] %s: no clean stand in the bounded lookahead; using best marginal "
            "fallback at (%.2f,%.2f), worst_frac=%.3f",
            key,
            _soft_fallback[0][0],
            _soft_fallback[0][1],
            _soft_score[0],
        )
        return _soft_fallback
    LOG.info(
        "[stand-scan] %s: cells=%d rej[occ=%d dist=%d filtered=%d] "
        "gate[idle_pen=%d navdisc=%d ik=%d arm_pen=%d arrival_pen=%d trajectory_pen=%d action_pen=%d camera_pen=%d noroute=%d] "
        "cands=%d -> NONE",
        key,
        _dbg["cells"],
        _dbg["occ"],
        _dbg["dist"],
        _dbg["filtered"],
        _dbg["idle_pen"],
        _dbg["navdisc"],
        _dbg["ik"],
        _dbg["arm_pen"],
        _dbg["arrival_pen"],
        _dbg["trajectory_pen"],
        _dbg["action_pen"],
        _dbg["camera_pen"],
        _dbg["noroute"],
        len(cands),
    )
    if (
        allow_expensive_retries
        and retry_cells is None
        and not strict
        and not fine_retry
        and fixed_stand is None
        and nav_grid is not None
        and start_xy is not None
        and _stride > 1
        and (not require_reach or not fine_ik_budget.exhausted)
    ):
        LOG.info(
            "[stand-scan] %s: coarse %.2fm scan found no stand; retrying unsampled %.2fm cells "
            "with %d IK solves left",
            key,
            _stride * nav_grid.cell,
            nav_grid.cell,
            fine_ik_budget.limit - fine_ik_budget.used,
        )
        _found = _search_reachable_stand(
            ph,
            stage,
            rig,
            scene,
            key,
            contact,
            stand_z,
            arm_idle,
            scene_paths,
            prefilter_boxes,
            nav_grid=nav_grid,
            start_xy=start_xy,
            rank_from=rank_from,
            seed_stand_xy=seed_stand_xy,
            flood_grid=flood_grid,
            contact_override=contact_override,
            contact_overrides=contact_overrides,
            shelf_gap_ceiling=shelf_gap_ceiling,
            stand_filter=stand_filter,
            route_gate=route_gate,
            trajectory_gate=trajectory_gate,
            route_exclude_owners=route_exclude_owners,
            strict=strict,
            ik_budget=fine_ik_budget,
            fine_ik_budget=fine_ik_budget,
            router_ik_budget=router_ik_budget,
            fine_retry=True,
            skip_cells=seen,
            trajectory_probe_count=trajectory_probe_count,
            router_trajectory_probe_count=[0],
            trajectory_probe_limit=MAX_STAND_TRAJECTORY_PROBES,
            start_yaw=start_yaw,
            gait_profile=gait_profile,
            gait_cum_dist=gait_cum_dist,
            reach_tol=reach_tol,
            require_reach=require_reach,
            require_openable_sweep_clearance=require_openable_sweep_clearance,
            openable_action_phase=openable_action_phase,
            openable_action_kind=openable_action_kind,
            front_access_standoff_m=front_access_standoff_m,
            egocentric_fallback=egocentric_fallback,
            allow_egocentric_fallback_return=False,
            allow_force_straight=allow_force_straight,
            allow_expensive_retries=allow_expensive_retries,
            candidate_validator=candidate_validator,
            action_state=action_state,
            facing_override=facing_override,
            complete_contact_trajectory=complete_contact_trajectory,
        )
        if _found is not None:
            return _found
    # ROUTER RETRY. Nothing survived, and gate (b0) -- a PROXY for "the router can get there" -- is the
    # only reason some candidates were never gated at all. Ask the router itself before giving up: the
    # answer it gives is the one the ``navigate_to`` that follows will act on, so a stand it accepts is by
    # construction one the walk can reach. Purely additive, in the strictest sense: this block is reached
    # only where the function returns None today, so every input that yields a stand today yields the SAME
    # stand, and the worst case here is the None it was already going to return. It adds no rejection --
    # the recursion's own gates are the ones that already ran -- so it cannot veto anything.
    if (
        allow_expensive_retries
        and retry_cells is None
        and fixed_stand is None
        and _nav_soft
        and _reach_mask is not None
        and start_xy is not None
        and (not require_reach or not router_ik_budget.exhausted)
    ):
        _retry_points = _nav_soft[:MAX_ROUTER_RETRY_CANDIDATES]
        LOG.info(
            "[stand-scan] %s: proxy gate (b0) turned away %d candidates and nothing else passed -- "
            "re-testing the first %d against the REAL router before declaring the stand unreachable",
            key,
            len(_nav_soft),
            len(_retry_points),
        )
        # SELF-REPORTING, because this block is on trial. As of 2026-08-11 it had fired 14 times across
        # the two rows where it was measured (`T2-HDU-01` 7, `T3-CHEM-01` 7) and rescued a stand ZERO
        # times -- the router agreed with the proxy every single time. The 0.35 m disagreement band it
        # exists for is STRUCTURALLY real (`init_validation.py --nav-proxy-control` reproduces both
        # directions offline) but appears UNPOPULATED in the current portfolio. Rather than leave that as
        # silent dead code or delete the knowledge, the counters ship in the manifest: if it ever rescues
        # one, nobody has to grep to find out, and if the portfolio grows and it still reads 0/N the
        # revert is trivially justified. **Do not credit this block with anything until `rescued` > 0.**
        _ctr = getattr(scene, "stand_router_retry", None)
        if _ctr is None:
            _ctr = {"fired": 0, "rescued": 0}
            try:
                scene.stand_router_retry = _ctr
            except Exception:  # noqa: BLE001 -- a scene that refuses the attribute still gets the retry
                _ctr = None
        if _ctr is not None:
            _ctr["fired"] += 1
        _found = _search_reachable_stand(
            ph,
            stage,
            rig,
            scene,
            key,
            contact,
            stand_z,
            arm_idle,
            scene_paths,
            prefilter_boxes,
            nav_grid=nav_grid,
            start_xy=start_xy,
            rank_from=rank_from,
            seed_stand_xy=seed_stand_xy,
            flood_grid=flood_grid,
            contact_override=contact_override,
            contact_overrides=contact_overrides,
            shelf_gap_ceiling=shelf_gap_ceiling,
            stand_filter=stand_filter,
            route_gate=route_gate,
            trajectory_gate=trajectory_gate,
            route_exclude_owners=route_exclude_owners,
            retry_cells={(round(x, 2), round(y, 2)) for x, y in _retry_points},
            strict=strict,
            ik_budget=router_ik_budget,
            fine_ik_budget=fine_ik_budget,
            router_ik_budget=router_ik_budget,
            fine_retry=fine_retry,
            skip_cells=skip_cells,
            trajectory_probe_count=router_trajectory_probe_count,
            router_trajectory_probe_count=router_trajectory_probe_count,
            trajectory_probe_limit=MAX_ROUTER_STAND_TRAJECTORY_PROBES,
            start_yaw=start_yaw,
            gait_profile=gait_profile,
            gait_cum_dist=gait_cum_dist,
            reach_tol=reach_tol,
            require_reach=require_reach,
            require_openable_sweep_clearance=require_openable_sweep_clearance,
            openable_action_phase=openable_action_phase,
            openable_action_kind=openable_action_kind,
            front_access_standoff_m=front_access_standoff_m,
            egocentric_fallback=egocentric_fallback,
            allow_egocentric_fallback_return=False,
            allow_force_straight=allow_force_straight,
            allow_expensive_retries=allow_expensive_retries,
            candidate_validator=candidate_validator,
            action_state=action_state,
            facing_override=facing_override,
            complete_contact_trajectory=complete_contact_trajectory,
        )
        if _found is not None and _ctr is not None:
            _ctr["rescued"] += 1
            LOG.warning(
                "[stand-scan] %s: ROUTER RETRY RESCUED a stand the proxy refused (%.2f,%.2f) -- "
                "this is the first thing that block has ever recovered; it is no longer dead code",
                key,
                _found[0][0],
                _found[0][1],
            )
        if _found is not None:
            return _found
    if (
        allow_egocentric_fallback_return
        and (not strict or camera_policy_verified)
        and egocentric_fallback.get("stand") is not None
    ):
        egocentric_fallback["selected"] = True
        _fallback = egocentric_fallback["stand"]
        if candidate_validator is not None and not candidate_validator(
            _fallback[0], _fallback[1]
        ):
            return None
        LOG.warning(
            "[stand-scan] %s: no fully clean route survived; using the least-penetrating "
            "camera-clear fallback at (%.2f,%.2f): %s",
            key,
            _fallback[0][0],
            _fallback[0][1],
            egocentric_fallback.get("reason") or "body-only route contact",
        )
        return _fallback
    return None


# How far off square-on a cue stand may be, mirroring ``task_world_builder.CUE_FACE_MIN_DOT``: cos(50 deg).
# Past this the artwork stops resolving in the FPV, and at 90 deg the reader faces the blank back board.
CUE_FACE_MIN_DOT: float = 0.64
SEAT_FRONT_MIN_DOT: float = 0.85


def cue_stand_for(scene, key):
    """``(cue_x, cue_y, face_x, face_y, declared_x, declared_y)`` when ``key`` carries READABLE artwork.

    The builder measures each cue's outward normal off the same ``front_dir`` the decal was hung on and
    publishes it as ``behavior_task['cue_stands']`` precisely because the runtime stand pick, which ranks
    on walking distance, has no way to know that one side of a board is blank. None for every entity that
    carries no cue -- which is every entity in a scene that publishes no cue stands, so the pick is
    unchanged there."""
    rec = ((getattr(scene, "behavior_task", None) or {}).get("cue_stands") or {}).get(
        key
    )
    if not rec or len(rec) < 5:
        return None
    ent = getattr(scene, "entities", {}).get(key)
    cxy = (
        (float(rec[5]), float(rec[6]))
        if len(rec) >= 7
        else (getattr(ent, "xy", None) if ent is not None else None)
    )
    if cxy is None:
        return None
    sx, sy, _yaw, fx, fy = (float(v) for v in rec[:5])
    if math.hypot(fx, fy) < 1e-6:
        return None
    return float(cxy[0]), float(cxy[1]), fx, fy, sx, sy


def _cue_face_dot(cue_x, cue_y, fx, fy, x, y) -> float:
    """cos(angle) between the artwork's outward normal and the direction from the cue to a stand."""
    dx, dy = float(x) - cue_x, float(y) - cue_y
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return -1.0
    return (dx / n) * fx + (dy / n) * fy


def _cue_face_filter(cue):
    """``stand_filter`` accepting only stands on the artwork's side of a cue (see ``cue_stand_for``)."""
    cx, cy, fx, fy = cue[0], cue[1], cue[2], cue[3]

    def _ok(x, y) -> bool:
        return _cue_face_dot(cx, cy, fx, fy, x, y) >= CUE_FACE_MIN_DOT

    return _ok


def seat_stand_for(scene, key):
    """Return the measured seat front contract when ``key`` has a seat affordance."""
    target_view = getattr(scene, "target_view", None)
    if not callable(target_view):
        return None
    from scene_state import Role

    seat = target_view(key, Role.SEAT)
    entity = getattr(scene, "entities", {}).get(key)
    if seat is None or entity is None:
        return None
    front = (getattr(entity, "meta", None) or {}).get("front_stand")
    if not isinstance(front, (list, tuple)) or len(front) < 2:
        return None
    cx, cy = float(seat.xy[0]), float(seat.xy[1])
    sx, sy = float(front[0]), float(front[1])
    dx, dy = sx - cx, sy - cy
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return cx, cy, dx / norm, dy / norm, sx, sy


def _seat_front_filter(seat):
    cx, cy, fx, fy = seat[:4]

    def _ok(x, y) -> bool:
        return _cue_face_dot(cx, cy, fx, fy, x, y) >= SEAT_FRONT_MIN_DOT

    return _ok


def _combine_stand_filters(first, second):
    if first is None:
        return second
    if second is None:
        return first

    def _ok(x, y) -> bool:
        return bool(first(x, y) and second(x, y))

    return _ok


def _either_stand_filter(first, second):
    if first is None:
        return second
    if second is None:
        return first

    def _ok(x, y) -> bool:
        return bool(first(x, y) or second(x, y))

    return _ok


def _centered_openable_stand_filter(
    center_xy,
    entry_axis,
    lateral_limit,
    standoff_m,
    *,
    lateral_cap_m: float = _CENTERED_OPENABLE_LATERAL_M,
):
    """Keep an Open/Close stance on the front-center strip of an aperture."""
    center = np.asarray(center_xy, dtype=np.float64)[:2]
    inward = np.asarray(entry_axis, dtype=np.float64)[:2]
    norm = float(np.linalg.norm(inward))
    if norm <= 1e-9:
        return None
    inward /= norm
    lateral = np.array([-inward[1], inward[0]], dtype=np.float64)
    limit = min(float(lateral_cap_m), max(0.15, float(lateral_limit)))
    minimum_outside = max(
        _CENTERED_OPENABLE_MIN_OUTSIDE_M,
        float(standoff_m) - _CENTERED_OPENABLE_DEPTH_SLACK_M,
    )

    def _ok(x, y) -> bool:
        offset = np.asarray((float(x), float(y)), dtype=np.float64) - center
        outside = -float(np.dot(offset, inward))
        sideways = abs(float(np.dot(offset, lateral)))
        return outside >= minimum_outside and sideways <= limit

    return _ok


def _finite_xy(value, *, exact=False):
    if value is None:
        return None
    try:
        xy = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if (exact and xy.size != 2) or xy.size < 2 or not np.all(np.isfinite(xy[:2])):
        return None
    return xy[:2]


def _stationary_openable_contract(
    entity,
    geometry,
    contact,
    *,
    action_kind=None,
):
    meta = getattr(entity, "meta", None) or {}
    anchor = _stationary_operator_anchor(entity, action_kind=action_kind)
    if not meta.get("stationary_operator") or anchor is None:
        return None

    facing_xy = _finite_xy(geometry[0]) if geometry is not None else None
    if facing_xy is None:
        hinge_xy = _finite_xy(getattr(entity, "hinge_xy", None))
        handle_xy = _finite_xy(getattr(entity, "handle_xy", None))
        if hinge_xy is not None and handle_xy is not None:
            facing_xy = 0.5 * (hinge_xy + handle_xy)
        else:
            for fallback in (handle_xy, getattr(entity, "xy", None), contact):
                facing_xy = _finite_xy(fallback)
                if facing_xy is not None:
                    break
    if facing_xy is None:
        return None
    return facing_xy, None, (float(anchor[0]), float(anchor[1]))


def _stationary_operator_anchor(entity, *, action_kind=None):
    meta = getattr(entity, "meta", None) or {}
    if not meta.get("stationary_operator"):
        return None
    if action_kind == "close" and meta.get("close_reposition"):
        return None
    return _finite_xy(meta.get("operator_stand_xy"), exact=True)


def _centered_openable_contract(
    scene,
    key,
    cur_xy,
    contact,
    semantic_executor,
    *,
    action_kind=None,
    opened_content: bool = False,
    standoff_override=None,
):
    """Return aperture-center facing, a center-strip filter, and preferred stand."""
    entity = semantic_executor._openable_access_entity(scene, key)
    base_geometry = (
        semantic_executor._front_access_geometry(
            entity,
            opened_content=opened_content,
            standoff_override=standoff_override,
        )
        if entity is not None
        else None
    )
    geometry = (
        semantic_executor._front_access_action_geometry(
            entity,
            opened_content=opened_content,
            action_kind=action_kind,
            standoff_override=standoff_override,
        )
        if entity is not None
        else None
    )
    stationary_contract = _stationary_openable_contract(
        entity,
        geometry,
        contact,
        action_kind=action_kind,
    )
    if stationary_contract is not None:
        return stationary_contract
    if geometry is not None:
        center, entry_axis, standoff, lateral_limit = geometry
        center_xy = np.asarray(center, dtype=np.float64)[:2]
        entry_axis = np.asarray(entry_axis, dtype=np.float64)[:2]
        meta = getattr(entity, "meta", None) or {}
        facing_xy = center_xy
        implicit_close_offset = (
            semantic_executor._implicit_close_operator_lateral_offset(entity)
            if action_kind == "close"
            else None
        )
        if action_kind == "close" and (
            meta.get("close_reposition") or implicit_close_offset is not None
        ):
            opened_contact_xy = _finite_xy(contact)
            if opened_contact_xy is not None:
                facing_xy = opened_contact_xy
        preferred = semantic_executor.preferred_openable_stand(
            scene,
            key,
            opened_content=opened_content,
            action_kind=action_kind,
            standoff_override=standoff_override,
        )
        if preferred is None:
            preferred_xy = center_xy - entry_axis * float(standoff)
            preferred = (float(preferred_xy[0]), float(preferred_xy[1]))
        action_lateral_cap = (
            float(lateral_limit)
            if "action_lateral_limit_m" in meta or implicit_close_offset is not None
            else _CENTERED_OPENABLE_LATERAL_M
        )
        stand_filter = _centered_openable_stand_filter(
            center_xy,
            entry_axis,
            lateral_limit,
            standoff,
            lateral_cap_m=action_lateral_cap,
        )
        if (
            implicit_close_offset is not None
            and base_geometry is not None
            and semantic_executor._implicit_close_allows_base_corridor(entity)
        ):
            base_center, base_entry_axis, base_standoff, base_lateral_limit = (
                base_geometry
            )
            stand_filter = _either_stand_filter(
                stand_filter,
                _centered_openable_stand_filter(
                    base_center,
                    base_entry_axis,
                    base_lateral_limit,
                    base_standoff,
                    lateral_cap_m=base_lateral_limit,
                ),
            )
        return (
            facing_xy,
            stand_filter,
            preferred,
        )
    if entity is None or getattr(entity, "hinge_xy", None) is None:
        return None
    hinge = np.asarray(entity.hinge_xy, dtype=np.float64)[:2]
    handle_xy = getattr(entity, "handle_xy", None)
    if handle_xy is None:
        handle_xy = getattr(entity, "xy", None)
    if handle_xy is None and contact is not None:
        handle_xy = np.asarray(contact, dtype=np.float64)[:2]
    if handle_xy is None:
        return None
    handle = np.asarray(handle_xy, dtype=np.float64)[:2]
    radius = handle - hinge
    panel_width = float(np.linalg.norm(radius))
    if panel_width <= 1e-6:
        return None
    center_xy = hinge + 0.5 * radius
    outward = np.array([-radius[1], radius[0]], dtype=np.float64) / panel_width
    current = np.asarray(cur_xy, dtype=np.float64)[:2]
    if float(np.dot(current - center_xy, outward)) < 0.0:
        outward *= -1.0
    entry_axis = -outward
    is_lid = (getattr(entity, "meta", None) or {}).get("openable_kind") == "lid"
    standoff = (
        min(0.70, max(0.50, float(np.linalg.norm(current - center_xy))))
        if is_lid
        else _NAV_DOOR_STANDOFF_M
    )
    filter_standoff = (
        _CENTERED_OPENABLE_MIN_OUTSIDE_M + _CENTERED_OPENABLE_DEPTH_SLACK_M
        if is_lid
        else standoff
    )
    preferred = center_xy + outward * standoff
    return (
        center_xy,
        _centered_openable_stand_filter(
            center_xy,
            entry_axis,
            min(_CENTERED_OPENABLE_LATERAL_M, 0.75 * panel_width),
            filter_standoff,
        ),
        (float(preferred[0]), float(preferred[1])),
    )


def pick_stand_runtime(
    ph,
    stage,
    rig,
    scene,
    key,
    cur_xy,
    contact_override=None,
    contact_overrides=None,
    shelf_gap_ceiling=None,
    doors_open=True,
    nearest_to_contact=False,
    cue_read=False,
    route_gate=False,
    start_yaw=None,
    gait_profile=None,
    gait_cum_dist: float = 0.0,
    reach_tol: float | None = None,
    require_reach: bool = True,
    require_openable_sweep_clearance: bool = True,
    openable_action_phase: bool = False,
    openable_action_kind: str | None = None,
    front_access_standoff_m: float | None = None,
    validated_stand_xy=None,
    transit_avoid_out=None,
    navigation_fallback_out=None,
    allow_expensive_retries: bool = True,
    candidate_validator=None,
    action_state=None,
    centered_openable: bool = False,
    complete_contact_trajectory: bool = False,
):
    """RUNTIME stand pick (Change 1): choose the stand for reaching ``key`` NOW, ranked NEAREST to the
    human's CURRENT position ``cur_xy`` -- among candidates that are render-acceptable (idle body-core) +
    arm-can-reach (pose-planner IK) + walk-reachable-from-cur (over ``scene.nav_grid`` when present).
    A shallow/marginal stand is retained as a fallback while a bounded number of later candidates are
    checked for a clean alternative; severe geometry is never returned.

    Change 2: ``doors_open`` (default True) runs the walkability flood on the DOORS-OPEN passable view of
    ``scene.nav_grid`` so a target BEHIND a closed door still gets a stand (the door is opened en route);
    pass ``doors_open=False`` for a DOOR-approach pick so the handle must be reachable WITHOUT first opening
    that door. ``contact_override`` (world xyz) reaches an explicit point (the door HANDLE) instead of the
    per-role entity contact (a door's per-role contact is its top surface, not the handle).

    ``nearest_to_contact`` (for PRECISE actions like ATTACH): rank candidate stands by CLOSEST-TO-CONTACT
    (best reach / smallest seat error) instead of closest-to-``cur_xy``. ``cur_xy`` still gates walk-
    reachability. Default False keeps the nearest-to-current runtime behavior (PUT / PLACE unchanged).

    ``require_reach=False`` keeps every navigation and FPV visibility gate while skipping arm reach. It is
    intended only for a walk immediately serving ``LOOK_AT``; interaction callers retain the True default.

    ``centered_openable`` makes every Open/Close approach face the aperture center and restricts the
    pelvis to its central strip. ``action_state`` still selects the free hand while carrying.

    ``route_gate`` (default False, i.e. byte-identical for every caller that does not ask) additionally
    requires ``nav_planner.plan_path`` to reach the WINNING stand. It exists for the ego-guided
    (blind-guidance) path, where the connectivity gate's shuffle tolerance and the router disagree
    often enough to cost 9 of 23 rows; see ``_search_reachable_stand``. Callers scope it on
    ``scene.ego_delivery``, which ``ego_route.recompile`` sets exactly when it compiled ego legs --
    a property of the episode, not a family name, so a future blind-guidance family inherits it."""
    import semantic_executor as _se

    stationary_entity = _se._openable_access_entity(scene, key)
    uses_content_geometry = _openable_phase_uses_content_geometry(
        require_openable_sweep_clearance=require_openable_sweep_clearance,
        openable_action_phase=openable_action_phase,
        openable_action_kind=openable_action_kind,
    )
    fixed_stand_xy = _stationary_operator_anchor(
        stationary_entity,
        action_kind=openable_action_kind,
    )
    explicit_lateral_anchor = _finite_xy(
        _se.explicit_openable_operator_anchor(
            scene,
            key,
            opened_content=uses_content_geometry,
            action_kind=openable_action_kind,
            standoff_override=front_access_standoff_m,
        ),
        exact=True,
    )
    content_seed_stand_xy = (
        explicit_lateral_anchor
        if uses_content_geometry and openable_action_kind in {"insert", "extract"}
        else None
    )
    close_contact_seed_stand_xy = (
        _se._implicit_close_contact_stand(
            stationary_entity,
            contact_override,
            opened_content=uses_content_geometry,
            standoff_override=front_access_standoff_m,
        )
        if stationary_entity is not None
        and contact_override is not None
        and openable_action_kind == "close"
        else None
    )
    action_seed_stand_xy = (
        content_seed_stand_xy
        if content_seed_stand_xy is not None
        else close_contact_seed_stand_xy
    )
    if fixed_stand_xy is None:
        fixed_stand_xy = _finite_xy(validated_stand_xy, exact=True)
    centered_openable = bool(centered_openable or fixed_stand_xy is not None)

    if transit_avoid_out is not None:
        transit_avoid_out.clear()
    if navigation_fallback_out is not None:
        navigation_fallback_out.clear()
    scene_paths, prefilter_boxes, stand_z, arm_idle, contact = _stand_gate_setup(
        stage, scene, rig, key
    )
    _co = None
    _cos = None
    if contact_overrides is not None:
        values = np.asarray(contact_overrides, dtype=np.float64).reshape(-1, 3)
        if len(values):
            _cos = tuple(tuple(float(axis) for axis in point) for point in values)
            contact = values.mean(axis=0)
    elif contact_override is not None:
        _co = np.asarray(contact_override, dtype=np.float64)
        contact = _co  # reach the handle: gate (b) IK + facing yaw target it, gate (c) via contact_override
    if contact is None:
        return None  # a floor / no-contact target has no object to stand at
    cur = (float(cur_xy[0]), float(cur_xy[1]))
    _ng = getattr(scene, "nav_grid", None)
    _flood = None
    if _ng is not None:
        import nav_grid as _nav_grid

        _flood = (
            _nav_grid.passable_view(_ng) if doors_open else _nav_grid.planning_view(_ng)
        )
    # CUE STANDS. Reaching an entity and READING it are different requirements, and only the first is a
    # distance. For a cue-bearing entity the nearest valid stand is very often the one BEHIND the board --
    # equally walkable, equally reachable, and showing the reader a blank back for the whole episode while
    # the goal check, which routes from the answer key, still scores 1.00. So: rank from the stand the
    # builder DECLARED (nearest-to-declared, not nearest-to-human) and hard-cull every candidate that is
    # not on the artwork's side. Reachability and non-penetration still decide WHICH facing stand is used;
    # they may no longer decide that a non-facing one is used instead.
    #
    # ``cue_read`` scopes this to the walks that exist to READ. A walk to a cue-bearing unit in order to
    # PUT something inside it is a different act: the side that carries the placard need not be the side
    # that opens, and forcing the approach onto the artwork's side there fails placements for a reason
    # that has nothing to do with what is on camera. Measured: unscoped, it cost `T3-CHEM-01` and
    # `T6-RDIN-01` several PUT commands whose stands were nav-disconnected on the placard side.
    _cue = cue_stand_for(scene, key) if cue_read else None
    _seat = seat_stand_for(scene, key)
    _filter = (
        _cue_face_filter(_cue)
        if _cue is not None
        else (_seat_front_filter(_seat) if _seat is not None else None)
    )
    _rank = _se.preferred_openable_stand(
        scene,
        key,
        opened_content=uses_content_geometry,
        action_kind=openable_action_kind,
        standoff_override=front_access_standoff_m,
    )
    if explicit_lateral_anchor is not None:
        _rank = explicit_lateral_anchor
    elif close_contact_seed_stand_xy is not None:
        _rank = close_contact_seed_stand_xy
    _facing = None
    if centered_openable:
        centered = _centered_openable_contract(
            scene,
            key,
            cur,
            contact,
            _se,
            action_kind=openable_action_kind,
            opened_content=uses_content_geometry,
            standoff_override=front_access_standoff_m,
        )
        if centered is not None:
            _facing, centered_filter, centered_rank = centered
            _filter = _combine_stand_filters(_filter, centered_filter)
            if _rank is None:
                _rank = centered_rank
    if _rank is None and not nearest_to_contact:
        _rank = cur
    if (
        _facing is None
        and action_state is not None
        and _se._held_right_hand(action_state)
    ):
        access_entity = _se._openable_access_entity(scene, key)
        access_geometry = (
            _se._front_access_geometry(access_entity)
            if access_entity is not None
            else None
        )
        if access_geometry is not None:
            _facing = access_geometry[0]
    if _cue is not None:
        _cx, _cy, _fx, _fy, _dx, _dy = _cue
        _rank = (_dx, _dy)
        print(
            f"[cue-stand] '{key}': cue @ ({_cx:.2f},{_cy:.2f}) face=({_fx:+.2f},{_fy:+.2f}); "
            f"declared stand ({_dx:.2f},{_dy:.2f}) -> only stands within "
            f"{math.degrees(math.acos(CUE_FACE_MIN_DOT)):.0f} deg of the artwork normal are eligible"
        )
    elif _seat is not None:
        _rank = (_seat[4], _seat[5])
        print(
            f"[seat-stand] '{key}': declared front ({_seat[4]:.2f},{_seat[5]:.2f}) -> "
            f"only stands within {math.degrees(math.acos(SEAT_FRONT_MIN_DOT)):.0f} deg of the "
            "seat front are eligible"
        )
    egocentric_fallback = {}
    found = _search_reachable_stand(
        ph,
        stage,
        rig,
        scene,
        key,
        contact,
        stand_z,
        arm_idle,
        scene_paths,
        prefilter_boxes,
        nav_grid=_ng,
        start_xy=cur,
        rank_from=_rank,
        fixed_stand_xy=fixed_stand_xy,
        seed_stand_xy=action_seed_stand_xy,
        flood_grid=_flood,
        contact_override=_co,
        contact_overrides=_cos,
        shelf_gap_ceiling=shelf_gap_ceiling,
        stand_filter=_filter,
        route_gate=bool(route_gate),
        trajectory_gate=True,
        route_exclude_owners=(
            tuple(str(path) for path in getattr(scene, "door_prims", ()) or ())
            if doors_open
            else ()
        ),
        start_yaw=start_yaw,
        gait_profile=gait_profile,
        gait_cum_dist=gait_cum_dist,
        reach_tol=reach_tol,
        require_reach=require_reach,
        require_openable_sweep_clearance=require_openable_sweep_clearance,
        openable_action_phase=openable_action_phase,
        openable_action_kind=openable_action_kind,
        front_access_standoff_m=front_access_standoff_m,
        egocentric_fallback=egocentric_fallback,
        allow_force_straight=navigation_fallback_out is not None,
        allow_expensive_retries=allow_expensive_retries,
        candidate_validator=candidate_validator,
        action_state=action_state,
        facing_override=_facing,
        complete_contact_trajectory=complete_contact_trajectory,
    )
    if found is None and navigation_fallback_out is not None and _ng is not None:
        relaxed_budget = _se.IKSearchBudget(
            _se.RESERVED_IK_SOLVES_FOR_ROUTER_STAND_RETRY
        )
        for clearance_m in _nav_grid.EGOCENTRIC_NAV_INFLATION_LADDER:
            relaxed_grid = _nav_grid.egocentric_passable_view(
                _ng,
                scene,
                clearance_m=clearance_m,
            )
            if relaxed_grid is None:
                break
            relaxed_fallback = {}
            found = _search_reachable_stand(
                ph,
                stage,
                rig,
                scene,
                key,
                contact,
                stand_z,
                arm_idle,
                scene_paths,
                prefilter_boxes,
                nav_grid=_ng,
                start_xy=cur,
                rank_from=_rank,
                fixed_stand_xy=fixed_stand_xy,
                seed_stand_xy=action_seed_stand_xy,
                flood_grid=relaxed_grid,
                contact_override=_co,
                contact_overrides=_cos,
                shelf_gap_ceiling=shelf_gap_ceiling,
                stand_filter=_filter,
                route_gate=True,
                trajectory_gate=False,
                strict=False,
                ik_budget=relaxed_budget,
                fine_ik_budget=relaxed_budget,
                router_ik_budget=relaxed_budget,
                start_yaw=start_yaw,
                gait_profile=gait_profile,
                gait_cum_dist=gait_cum_dist,
                reach_tol=reach_tol,
                require_reach=require_reach,
                require_openable_sweep_clearance=require_openable_sweep_clearance,
                openable_action_phase=openable_action_phase,
                openable_action_kind=openable_action_kind,
                front_access_standoff_m=front_access_standoff_m,
                egocentric_fallback=relaxed_fallback,
                allow_force_straight=False,
                candidate_validator=candidate_validator,
                action_state=action_state,
                facing_override=_facing,
                complete_contact_trajectory=complete_contact_trajectory,
            )
            if found is not None:
                egocentric_fallback.update(
                    selected=True,
                    force_straight=False,
                    relaxed_route=True,
                    stand=found,
                    reason=(
                        "the full-body nav grid was disconnected; using a camera-validated "
                        f"{clearance_m * 1000.0:.0f}mm-clearance route"
                    ),
                )
                LOG.warning(
                    "[stand-scan] %s: full-body grid has no route; retained stand "
                    "(%.2f,%.2f) on the %.0fmm-clearance grid for camera validation",
                    key,
                    found[0][0],
                    found[0][1],
                    clearance_m * 1000.0,
                )
                break
    validated_paths = egocentric_fallback.pop("_validated_paths", {})
    if navigation_fallback_out is not None and found is not None:
        validated_path = validated_paths.get(_stand_plan_key(found[0], found[1]))
        if validated_path is not None:
            navigation_fallback_out.update(
                path=validated_path,
                validated_path=True,
            )
    if navigation_fallback_out is not None and egocentric_fallback.get("selected"):
        navigation_fallback_out.update(egocentric_fallback)
    if transit_avoid_out is not None and _flood is not None:
        if egocentric_fallback.get("selected"):
            transit_avoid_out.clear()
        else:
            transit_avoid_out.update(_flood.transit_avoid)
    if found is None:
        if _cue is not None:
            # LOUD, not silent. The alternative -- dropping the facing requirement to keep the walk -- is
            # what produced an episode that scored the assistant on a sign it was never shown.
            print(
                f"[cue-stand] '{key}': NO stand is both reachable AND on the artwork's side "
                f"(cue @ ({_cue[0]:.2f},{_cue[1]:.2f}) face=({_cue[2]:+.2f},{_cue[3]:+.2f})) "
                f"-> FAIL the command; refusing to read a blank back board"
            )
        elif _seat is not None:
            print(
                f"[seat-stand] '{key}': NO reachable stand is in front of the seat -> "
                "FAIL the command; refusing to sit from the side or in open air"
            )
        return None
    (fx, fy), fyaw = found
    if _cue is not None:
        _dot = _cue_face_dot(_cue[0], _cue[1], _cue[2], _cue[3], fx, fy)
        print(
            f"[cue-stand] '{key}': stand ({fx:.2f},{fy:.2f}) yaw={fyaw:.0f} is "
            f"{math.degrees(math.acos(max(-1.0, min(1.0, _dot)))):.0f} deg off the artwork normal "
            f"({math.hypot(fx - _cue[4], fy - _cue[5]) * 1000:.0f} mm from the declared stand)"
        )
    elif _seat is not None:
        _dot = _cue_face_dot(_seat[0], _seat[1], _seat[2], _seat[3], fx, fy)
        print(
            f"[seat-stand] '{key}': stand ({fx:.2f},{fy:.2f}) yaw={fyaw:.0f} is "
            f"{math.degrees(math.acos(max(-1.0, min(1.0, _dot)))):.0f} deg off seat-front "
            f"({math.hypot(fx - _seat[4], fy - _seat[5]) * 1000:.0f} mm from declared)"
        )
    return (float(fx), float(fy)), float(fyaw)


def actor_near_entity(scene, key: str, actor_xy) -> bool:
    """Whether a failed stand search may safely resolve an explicit walk as already arrived."""
    if key not in scene.entities or actor_xy is None:
        return False
    entity = scene.entities[key]
    actor = (float(actor_xy[0]), float(actor_xy[1]))
    candidates = [tuple(entity.xy)]
    front = entity.meta.get("front_stand")
    if isinstance(front, (list, tuple)) and len(front) >= 2:
        candidates.append((float(front[0]), float(front[1])))
    clearance = max(0.65, float(getattr(entity, "clearance", 0.0) or 0.0) + 0.75)
    distances = [math.dist(actor, point) for point in candidates]
    foot_xy = getattr(entity, "foot_xy", None)
    foot_half = getattr(entity, "foot_half", None)
    if foot_xy is not None and foot_half is not None:
        dx = max(abs(actor[0] - float(foot_xy[0])) - float(foot_half[0]), 0.0)
        dy = max(abs(actor[1] - float(foot_xy[1])) - float(foot_half[1]), 0.0)
        distances.append(math.hypot(dx, dy))
    return min(distances) <= clearance


def _entity_contact(scene, key):
    """The reach contact xyz for an entity (object pos / surface top / handle), or None for a floor goal."""
    from scene_state import Role

    e = scene.resolve(key)
    if e.role is Role.GRASPABLE:
        return np.asarray(scene.object_pos(key), dtype=np.float64)
    access_meta = getattr(e, "meta", None) or {}
    if e.role is Role.CONTAINER and access_meta.get("access_mode") in {
        "front_drum",
        "front_opening",
    }:
        aperture_center = np.asarray(
            access_meta.get("aperture_center"),
            dtype=np.float64,
        )
        if aperture_center.shape == (3,):
            return aperture_center
    if e.role is Role.CONTAINER and getattr(e, "drop_z", None) is not None:
        # a CONTAINER's reach contact is where the object is DROPPED (the interior floor / front-opening
        # shelf, ~shoulder height), NOT its TOP surface: reaching the top of a HIGH wall cabinet (~2.3 m)
        # is impossible and falsely rejects every stand, whereas the object actually goes INSIDE at drop_z.
        cxy = e.xy if e.xy is not None else scene.object_pos(key)[:2]
        return np.array([cxy[0], cxy[1], float(e.drop_z)], dtype=np.float64)
    if e.role is Role.SURFACE:
        (sx, sy), top = scene.surface_top(key)
        if e.foot_half is not None:
            foot_x, foot_y = e.foot_xy if e.foot_xy is not None else (sx, sy)
            angle = math.radians(e.approach_yaw)
            forward_x, forward_y = math.sin(angle), -math.cos(angle)
            half_depth = abs(forward_x) * float(e.foot_half[0]) + abs(
                forward_y
            ) * float(e.foot_half[1])
            inset = min(0.08, 0.25 * half_depth)
            sx = float(foot_x) - forward_x * max(0.0, half_depth - inset)
            sy = float(foot_y) - forward_y * max(0.0, half_depth - inset)
        return np.array([sx, sy, top], dtype=np.float64)
    if e.role in (Role.CONTAINER, Role.OPENABLE, Role.SEAT):
        (sx, sy), top = scene.surface_top(key)
        return np.array([sx, sy, top], dtype=np.float64)
    return None


def _approach_dir(stand_xy, contact, yaw_deg):
    """Horizontal unit vector from the stand point toward the contact (the approach axis stand_pose_for
    used); falls back to the body forward at ``yaw_deg`` when degenerate."""
    if contact is not None:
        d = np.asarray(contact[:2], dtype=np.float64) - np.asarray(
            stand_xy, dtype=np.float64
        )
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            return d / n
    a = math.radians(yaw_deg)
    return np.array([math.sin(a), -math.cos(a)], dtype=np.float64)


def _stand_ladder(scene, key, base_stand, base_yaw, approach, contact):
    """The ordered ladder of stand-pose candidates: base, stepped back-off along -approach, then
    stand_pose_for re-derived at rotated approach_yaw (+-45, +-90, 180). Each rung is (stand_xy, yaw,
    label). Back-off keeps the base yaw (still facing the target); rotated rungs get their own yaw."""
    rungs = [(base_stand, base_yaw, "base")]
    # stepped back-off along the SAME approach axis (further from the near-face).
    for extra in (0.15, 0.30, 0.45):
        sxy = (base_stand[0] - approach[0] * extra, base_stand[1] - approach[1] * extra)
        rungs.append((sxy, base_yaw, f"backoff+{extra:.2f}"))
    # rotated approach_yaw: re-derive stand_pose_for's near-face geometry from a different side/back.
    e = scene.resolve(key)
    base_ay = e.approach_yaw
    for dy in (45.0, -45.0, 90.0, -90.0, 180.0):
        try:
            e.approach_yaw = base_ay + dy
            sxy, syaw = scene.stand_pose_for(key)
        finally:
            e.approach_yaw = base_ay
        rungs.append((sxy, syaw, f"rot{dy:+.0f}"))
    return rungs


# =====================================================================================================
# P0 positive control: prove the bare detector is a false-negative + the exact gate catches it.
# =====================================================================================================
def positive_control(app, stage, ph) -> tuple[bool, dict]:
    """Pose the invisible proxy rig with its TORSO embedded in a STATIC table box; CONFIRM that the bare
    ``human_scene_collision_pairs`` returns CLEAN (the false-negative bug -- kinematic-static reporting
    OFF) while the depth-bearing ``human_scene_separation`` returns the penetrating pair with a NEGATIVE
    separation. Returns (ok, info) where ok == (bare is CLEAN AND exact caught it). This is the P0 gate
    that grounds everything downstream -- without it every 'verified clean' verifies nothing."""
    import object_physics as ophys
    import smplh_anim_core as core

    ph.build_physics_scene(stage)
    ph.set_scene_timestep(stage, 60)

    # A STATIC table box centered at the origin, top at ~0.75, spanning a torso-sized volume.
    tbl_path = "/World/PC_Table"
    _make_static_box(stage, tbl_path, size=(1.2, 0.8, 0.75), center=(0.0, 0.0, 0.375))
    ph.make_static_collider(stage.GetPrimAtPath(tbl_path), approximation="none")

    rig = core.load_family("side_wrap")
    try:
        core.derive_rest_place_arms(rig)
    except Exception as exc:  # noqa: BLE001
        LOG.info("positive_control: derive_rest_place_arms skipped (%s)", exc)

    # Place the body so its TORSO is INSIDE the table box: stand at the box center, feet well below the
    # top, so pelvis/spine sit inside the 0..0.75 z-band and inside the 1.2x0.8 XY footprint.
    stand_z = _stand_z_for(rig, 0.0)
    arm_idle = getattr(rig, "arm_rest", rig.arm_grasp)
    # Deliberately place the ROOT at the box center (0,0): the standing body's pelvis (~0.9) & spine sit
    # inside the box top region, and the legs run straight through the solid box -> a gross penetration.
    J = human_proxy_J(rig, arm_idle, (0.0, 0.0), 0.0, stand_z)
    proxy = ph.build_proxy_rig(stage, rig, J)
    for _ in range(2):
        app.update()

    # BARE detector (the bug): get_initial_collider_pairs, kinematic-static reporting OFF.
    bare = ph.human_scene_collision_pairs(stage)
    bare_clean = len(bare) == 0

    # EXACT depth-bearing gate: the all_pairs_separation pattern (kinematic-static ON). Do the one-time
    # flag+reset setup FIRST, then pose the proxy (so the reset does not clobber the posed torso), query.
    sim = ophys.get_sim()
    ph.arm_kinematic_static_gate(sim, stage, [b.path for b in proxy.bones] + [tbl_path])
    ph.update_proxy_rig(proxy, J)
    exact = ph.human_scene_separation(sim, stage, [tbl_path], sep_tol=_GATE_TOL_M)
    exact_caught = len(exact) > 0
    worst = min((p[2] for p in exact), default=0.0)

    # OVERLAP gate (what the RL init path actually uses): lag-free scene-query overlap from J; must also
    # catch the torso-in-table penetration.
    overlap = ph.human_scene_overlap(stage, J, [tbl_path])
    overlap_caught = len(overlap) > 0

    ok = bare_clean and exact_caught and overlap_caught
    info = {
        "bare_pairs": len(bare),
        "bare_clean": bare_clean,
        "exact_pairs": len(exact),
        "exact_caught": exact_caught,
        "overlap_pairs": len(overlap),
        "overlap_caught": overlap_caught,
        "worst_sep_mm": worst * 1000.0,
    }
    LOG.info(
        "positive_control: bare=%d (clean=%s) exact=%d (caught=%s worst=%.0fmm) overlap=%d (caught=%s) -> %s",
        len(bare),
        bare_clean,
        len(exact),
        exact_caught,
        worst * 1000.0,
        len(overlap),
        overlap_caught,
        "PASS" if ok else "FAIL",
    )
    return ok, info


def coverage_control(app, stage, ph) -> tuple[bool, dict]:
    """Prove the NEW capsule coverage (feet + shoulders) gates via the SAME per-pose gate the RL path
    uses (``human_scene_overlap`` with ``_CORE_J``). Two sub-controls, each a small STATIC box embedding
    ONE new region while the OLD (pre-fix) capsule set would read CLEAN there:

      * FOOT: a thin LOW box at foot height under one foot -> the ankle->foot capsule must fire (the shins
        stop at the ankles, so before the fix this read clean with the foot inside the box).
      * SHOULDER: a thin box grazing one shoulder OUTSIDE the spine centerline at chest height -> the
        collar capsule must fire (the r=0.13 spine capsule does not reach the ~0.20 m shoulder half-width).

    Both ALSO assert the body FAR from the box reads clean (the gate discriminates, not a constant-fire).
    Returns (ok, info); ok == both sub-controls fire near + clean far."""
    import smplh_anim_core as core

    ph.build_physics_scene(stage)
    ph.set_scene_timestep(stage, 60)

    rig = core.load_family("side_wrap")
    try:
        core.derive_rest_place_arms(rig)
    except Exception as exc:  # noqa: BLE001
        LOG.info("coverage_control: derive_rest_place_arms skipped (%s)", exc)
    arm_idle = getattr(rig, "arm_rest", rig.arm_grasp)
    stand_z = _stand_z_for(rig, 0.0)

    # Idle body standing at the origin; read the world foot + shoulder joint positions to place the boxes.
    J0 = human_proxy_J(rig, arm_idle, (0.0, 0.0), 0.0, stand_z)
    from smplh_fpv import R_SHOULDER

    foot_j = 11  # R_foot
    foot_p = np.asarray(J0[foot_j], dtype=np.float64)
    sh_p = np.asarray(J0[R_SHOULDER], dtype=np.float64)

    # ---- FOOT sub-control: a thin LOW box straddling the foot joint (a low rail / threshold). Thin in Z
    # so it sits at foot height only (shins/knees are well above it) -- the shin capsules (ending at the
    # ankles) miss it; only the new ankle->foot capsule reaches in.
    foot_path = "/World/PC_FootRail"
    _make_static_box(
        stage,
        foot_path,
        (0.6, 0.6, 0.10),
        (float(foot_p[0]), float(foot_p[1]), float(foot_p[2])),
    )
    ph.make_static_collider(stage.GetPrimAtPath(foot_path), approximation="none")

    # ---- SHOULDER sub-control: a thin vertical slab centered ON the shoulder joint, at chest height,
    # OUTSIDE the spine centerline (the shoulder sits ~0.17-0.20 m off the spine axis). The spine capsule
    # (r=0.13 about the centerline) cannot reach it; only the new collar capsule does.
    sh_path = "/World/PC_ShoulderPost"
    _make_static_box(
        stage,
        sh_path,
        (0.12, 0.12, 0.30),
        (float(sh_p[0]), float(sh_p[1]), float(sh_p[2])),
    )
    ph.make_static_collider(stage.GetPrimAtPath(sh_path), approximation="none")
    for _ in range(3):
        app.update()

    # Near = the body at the origin (foot in the rail, shoulder in the post). Far = shifted +6 m in X.
    J_far = human_proxy_J(rig, arm_idle, (6.0, 0.0), 0.0, stand_z)
    foot_near = ph.human_scene_overlap(stage, J0, [foot_path], joint_subset=_CORE_J)
    foot_far = ph.human_scene_overlap(stage, J_far, [foot_path], joint_subset=_CORE_J)
    sh_near = ph.human_scene_overlap(stage, J0, [sh_path], joint_subset=_CORE_J)
    sh_far = ph.human_scene_overlap(stage, J_far, [sh_path], joint_subset=_CORE_J)

    foot_ok = len(foot_near) > 0 and len(foot_far) == 0
    sh_ok = len(sh_near) > 0 and len(sh_far) == 0
    ok = foot_ok and sh_ok
    info = {
        "foot_near_pairs": len(foot_near),
        "foot_far_pairs": len(foot_far),
        "foot_caps": sorted({c for c, _o in foot_near}),
        "foot_ok": foot_ok,
        "sh_near_pairs": len(sh_near),
        "sh_far_pairs": len(sh_far),
        "sh_caps": sorted({c for c, _o in sh_near}),
        "sh_ok": sh_ok,
    }
    LOG.info(
        "coverage_control: FOOT near=%d(%s) far=%d ok=%s | SHOULDER near=%d(%s) far=%d ok=%s -> %s",
        len(foot_near),
        info["foot_caps"],
        len(foot_far),
        foot_ok,
        len(sh_near),
        info["sh_caps"],
        len(sh_far),
        sh_ok,
        "PASS" if ok else "FAIL",
    )
    return ok, info


def _make_static_box(stage, path, size, center):
    """A bare UsdGeom.Cube (unit cube scaled to ``size``, centered at ``center``) -- the static table for
    the positive control. No material (collision-only test)."""
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    )
    xf.AddScaleOp().Set(Gf.Vec3d(float(size[0]), float(size[1]), float(size[2])))
    return cube.GetPrim()


# =====================================================================================================
# CLI: the P0 positive-control (the only path that must boot Isaac standalone here).
# =====================================================================================================
def _run_positive_control() -> int:
    import fpv_core
    from fpv_core import AppCfg

    app = fpv_core.launch_app(
        AppCfg(headless=True, width=640, height=480, renderer="RayTracedLighting")
    )
    import render_runtime

    fpv_core.post_launch_setup(
        AppCfg(headless=True, width=640, height=480, renderer="RayTracedLighting"),
        render_runtime._loco_cfg(),
    )
    stage = fpv_core.get_stage()
    import physics_human as ph

    ok, info = positive_control(app, stage, ph)
    print(
        f"[initval][P0] positive-control: bare human_scene_collision_pairs pairs={info['bare_pairs']} "
        f"(CLEAN={info['bare_clean']} <- the false-negative bug) | exact human_scene_separation "
        f"pairs={info['exact_pairs']} (CAUGHT={info['exact_caught']} worst={info['worst_sep_mm']:.0f}mm) "
        f"| overlap gate pairs={info['overlap_pairs']} (CAUGHT={info['overlap_caught']}) "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    # NEW coverage controls (feet + shoulders): the SAME human_scene_overlap gate must fire on a foot in
    # a low rail and a shoulder grazing a post (both would read clean before the fix), and stay clean far.
    cov_ok, cov = coverage_control(app, stage, ph)
    print(
        f"[initval][COVERAGE] FOOT overlap near={cov['foot_near_pairs']} caps={cov['foot_caps']} "
        f"far={cov['foot_far_pairs']} (FIRES={cov['foot_ok']}) | SHOULDER overlap near={cov['sh_near_pairs']} "
        f"caps={cov['sh_caps']} far={cov['sh_far_pairs']} (FIRES={cov['sh_ok']}) "
        f"-> {'PASS' if cov_ok else 'FAIL'}"
    )
    both = ok and cov_ok
    print(
        f"[initval] positive+coverage controls -> {'PASS' if both else 'FAIL'} "
        f"(P0 torso={ok}, foot={cov['foot_ok']}, shoulder={cov['sh_ok']})"
    )
    app.close()
    return 0 if both else 2


def _run_nav_proxy_control() -> int:
    """OFFLINE gate proof (no Isaac): gate (b0)'s connectivity PROXY disagrees with the router, both ways.

    ``_reachable_free_within`` -- "is a start-reachable free cell within ``_shuffle_tol``" -- stands in for
    ``nav_planner.plan_path``, but the two answer different questions and the difference is not academic:
    it accounted for 41 of the 107 ``path_blocked`` occurrences in the cb1/cb2 baseline, in BOTH
    directions. This reproduces each direction on a synthetic grid in a few milliseconds so that neither
    can regress silently, and so that anyone re-reading a ``path_blocked`` triage can see the mechanism
    without booting a scene.

    TOO STRICT (the proxy rejects what the router walks): the proxy's radius is ``_shuffle_tol`` (0.25 m at
    the 0.10 m inflation) while ``plan_path`` snaps a blocked goal to any free cell within
    ``nav_planner._GOAL_SNAP_R`` (0.60 m) before routing. A stand 0.30-0.50 m off reachable floor is
    therefore refused by the proxy and walked by the router. When it refuses EVERY candidate the command
    is lost without the router ever being consulted -- measured on ``T2-HDU-01`` (10), ``T2-BW1-01`` (4),
    ``T4-WAI0G-01`` (4), ``T3-ASIAN-03`` (1), ``T3-CHEM-01`` (1).

    TOO LOOSE (the proxy accepts what the router refuses): a stand just past a wall is within a shuffle of
    free cells on BOTH sides, so the proxy reports "connected" while standing on the unreachable side. A
    DOORWAY is this geometry by definition, which is why the runtime door-approach pick was the site that
    bled -- ``T5-CAF-02`` (both arms), ``T5-HSS-01``, ``T5-GROCA-01``, ``T1-RHOT-02``, 21 occurrences whose
    log reads ``navdisc=0 ... -> FOUND`` and then ``walk to door stand FAILED (path_blocked)``.
    """
    import nav_grid as _ng
    import nav_planner as _np

    cell, nx, ny = 0.10, 80, 80
    occ = np.zeros((ny, nx), dtype=bool)
    jw = ny // 2
    occ[jw, :] = True  # one solid wall, no opening -> two disconnected rooms
    grid = _ng.NavGrid(0.0, 0.0, cell, nx, ny, occ, _ng._two_sided_sdf(occ, cell))
    start = (2.0, 2.0)
    mask = _ng.flood_free(grid, grid.world_to_cell(*start))
    wall_y = grid.cell_center(0, jw)[1]

    # TOO LOOSE: 0.10 m past the wall -- proxy True (a reachable cell is a shuffle away, on the far side),
    # router False (there is no way around).
    x_l, y_l = 2.0, wall_y + 0.10
    loose_proxy = _reachable_free_within(grid, mask, x_l, y_l)
    loose_router = _stand_routes(grid, start, x_l, y_l)
    loose_ok = loose_proxy and not loose_router

    # TOO STRICT: a blocked blob whose nearest reachable free floor is 0.30/0.40/0.50 m away -- proxy
    # False at every distance, router True at every distance.
    strict = []
    for d in (0.30, 0.40, 0.50):
        occ2 = occ.copy()
        ci, cj = grid.world_to_cell(5.0, 2.0)
        r = int(round(d / cell))
        occ2[cj - r : cj + r + 1, ci - r : ci + r + 1] = True
        g2 = _ng.NavGrid(0.0, 0.0, cell, nx, ny, occ2, _ng._two_sided_sdf(occ2, cell))
        m2 = _ng.flood_free(g2, g2.world_to_cell(*start))
        strict.append(
            (
                d,
                _reachable_free_within(g2, m2, 5.0, 2.0),
                _stand_routes(g2, start, 5.0, 2.0),
            )
        )
    strict_ok = all((not p) and r for _d, p, r in strict)

    print(
        f"[initval][NAV-PROXY] tol={_shuffle_tol(grid):.2f} m vs router goal-snap="
        f"{_np._GOAL_SNAP_R:.2f} m -- the proxy is the STRICTER of the two by construction"
    )
    print(
        f"[initval][NAV-PROXY] TOO LOOSE  stand {y_l - wall_y:.2f} m past a solid wall: "
        f"proxy={loose_proxy} router={loose_router} -> picker accepts, the walk then fails "
        f"path_blocked (FIRES={loose_ok})"
    )
    for d, p, r in strict:
        print(
            f"[initval][NAV-PROXY] TOO STRICT reachable floor {d:.2f} m away: "
            f"proxy={p} router={r} -> command lost though the router would have walked it"
        )
    ego_ok = _nav_proxy_ego_door_case()
    both = loose_ok and strict_ok and ego_ok
    print(
        f"[initval][NAV-PROXY] all three reproduced -> {'PASS' if both else 'FAIL'} "
        f"(too_loose={loose_ok}, too_strict={strict_ok}, ego_door={ego_ok})"
    )
    return 0 if both else 2


def _nav_proxy_ego_door_case() -> bool:
    """The THIRD way a stand-picker fix can be inert: on an ego-guided episode the walk never routes.

    Measured, then reproduced here. Router-gating the runtime door-approach stand pick looked like the
    obvious repair for 21 `path_blocked` occurrences whose log reads `navdisc=0 ... -> FOUND` followed by
    `walk to door stand FAILED (path_blocked)`. It came back **byte-identical** on `T5-CAF-02.conflict`
    and `T5-GROCA-01`, because the walk to that stand is refused by ``_assert_ego_unlocked`` BEFORE any
    routing happens: ``scene.ego_delivery`` is keyed by the guidance PROGRAM's entity keys, a door is
    never one of them, so the lock falls back to comparing the body against the door stand itself, finds
    it further than ``EGO_ARRIVE_M``, and raises PATH_BLOCKED.

    That refusal is CORRECT -- it is what stops the harness walking a body the guidance failed to
    deliver, which is the whole claim a blind-guidance row tests. The gap it exposes is upstream: the ego
    compiler never plans a leg to a door. This case is here so the next person to see `path_blocked` at a
    door does not spend the night on the stand picker, as I did.
    """
    import ego_route
    import semantic_executor as _se
    from goal_schema import Goal, Primitive, PrimitiveError, PrimitiveFailure

    class _Scene:
        ego_delivery = {"unit_a": (4.9, 0.0)}  # task entities only -- never a door

    far = ego_route.EGO_ARRIVE_M + 1.5
    st = _se.SchedState(root_xy=(0.0, 0.0), root_yaw=0.0, stand_z=0.0, floor_z=0.0)
    st.ego_locked = True
    door_goal = Goal(
        Primitive.NAVIGATE_TO, target_xy=(far, 0.0), target_key="door_hzfirv_0"
    )
    refused = False
    try:
        _se._assert_ego_unlocked(_Scene(), st, door_goal, False)
    except PrimitiveFailure as exc:
        refused = exc.reason is PrimitiveError.PATH_BLOCKED
    # Control: the SAME distance to an entity the guidance DID compile a delivery for is allowed, because
    # the lock measures against the delivery point, not the stand. Without this the case above would only
    # prove "the lock rejects far things", which is not the finding.
    _se_ok = True
    near_goal = Goal(Primitive.NAVIGATE_TO, target_xy=(far, 0.0), target_key="unit_a")
    # delivered where the guidance meant to leave it
    st2 = _se.SchedState(root_xy=(4.9, 0.0), root_yaw=0.0, stand_z=0.0, floor_z=0.0)
    st2.ego_locked = True
    try:
        _se._assert_ego_unlocked(_Scene(), st2, near_goal, False)
    except PrimitiveFailure:
        _se_ok = False
    ok = refused and _se_ok
    print(
        f"[initval][NAV-PROXY] EGO DOOR   an ego-locked body {far:.2f} m from a door stand "
        f"(budget {ego_route.EGO_ARRIVE_M:.2f} m, no ego_delivery entry for a door): "
        f"refused={refused} -> the walk NEVER reaches the router, so router-gating the door "
        f"pick is inert (FIRES={ok}); delivered-entity control allowed={_se_ok}"
    )
    return ok


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nav-proxy-control",
        action="store_true",
        help="OFFLINE (no Isaac): prove that the stand search's connectivity PROXY (gate b0) and the "
        "router disagree in BOTH directions -- it rejects stands plan_path walks to, and accepts stands "
        "plan_path refuses. The mechanism behind 41 of 107 baseline `path_blocked` occurrences.",
    )
    ap.add_argument(
        "--positive-control",
        action="store_true",
        help="P0 + COVERAGE gate proofs: (1) torso in a static table -> bare collision pairs are a "
        "false-negative while human_scene_separation + human_scene_overlap catch it; (2) foot in a low "
        "rail and (3) shoulder grazing a post -> the SAME human_scene_overlap gate fires (feet + "
        "shoulder capsules) and stays clean far away. Boots Isaac.",
    )
    args = ap.parse_args()
    if args.nav_proxy_control:
        return _run_nav_proxy_control()
    if args.positive_control:
        return _run_positive_control()
    print(
        "[initval] nothing to do; pass --positive-control (P0 gate proof, boots Isaac) "
        "or --nav-proxy-control (stand-connectivity proxy vs router, offline)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

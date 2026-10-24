# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Command-driven semantic action executor.

Turns a ``list[goal_schema.Goal]`` into the SAME precomputed ``Frame`` stream the hardcoded
walk-pick-place demo produces -- but driven by ARBITRARY commands, not a fixed 9-beat script.
The runtime is a deterministic "dispatch -> produce frames -> stitch" player with NO on-the-fly
motion design and NO wall-clock-dependent control flow, so an episode is reproducible.

Reuses the proven pieces verbatim (offline-bake-once, then rigid-replay -- exactly what the demo
already does): ``gait_player`` (distance-phased walk), ``smplh_anim_core`` (pose/seat/warp), and
``humanoid_motion.{Frame, Motion, realize, solve_grasp}`` (the grasp/reach is solved ONCE and
then replayed via ``arm_override`` + ``rhp`` + ``seat_object_nosnap``). The grasp CCD warp is the
serial cost; running it once per PICK (not per frame) keeps the runtime a table-lookup + placement.

The GROUNDED verbs are NAVIGATE_TO / PICK / PLACE / RELEASE (the subset ACCAD's data actually
supports); ANCHORED verbs (Sit/Open/Close), CONTACT verbs and CAMERA verbs are produced on top of
the same machinery. Offline check (NO Isaac -- run this FIRST):

    bash launch/run_isaac.sh simulator/physical_world/semantic_executor.py --dry
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
from dataclasses import dataclass, replace

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIMULATOR_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, os.path.join(_SIMULATOR_DIR, "human_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import command_grammar  # noqa: E402

# torch-first (pre-launch): every import below pulls the torch math core at module scope.
import humanoid_motion as wpp  # noqa: E402  reuse Frame/Motion/realize/solve_grasp + constants
import gait_player as gait  # noqa: E402
import grasp_select  # noqa: E402
import hinge_gate  # noqa: E402  --step3's cover for the USD hinge path it cannot itself reach
import interaction_hints  # noqa: E402
import isaac_goal  # noqa: E402
import placement_packing  # noqa: E402
import runtime_capabilities  # noqa: E402
import smplh_anim_core as core  # noqa: E402
import smplh_camera  # noqa: E402
import surface_visual_effects  # noqa: E402
from goal_schema import (  # noqa: E402
    FORWARD_STEP_M,
    Goal,
    Primitive,
    PrimitiveError,
    PrimitiveFailure,
    TURN_DEG,
    WAIT_DEFAULT_S,
    WAIT_MAX_S,
    WAIT_MIN_S,
)
from scene_state import Role, SceneState  # noqa: E402

# Beat lengths (frames) -- identical to the proven demo choreography, so its geometry is reproduced.
_N_ARRIVE = 6
_N_REACH = 18
_N_LIFT = 10
_N_PLACE = 18
_N_RETRACT = 10
_N_HANDLE_HOLD = 6
_N_DEPART_MIN = 4  # min walk-in frames (pose ease); a turn adds ~1 frame / 15deg on top
_LEAN_RAD = math.radians(wpp.REACH_LEAN_DEG)
# warp residual above this -> OUT_OF_REACH (the demo's pick lands ~13 mm)
_REACH_TOL_M = 0.03
# Opening an articulated cabinet/drawer is visually and semantically valid when the best safe hand pose
# stops slightly short of the handle. Keep this as an explicit absolute cap: it must not grow if the
# general manipulation tolerance changes, and it remains far below obviously fabricated metre-scale reach.
_OPENABLE_REACH_TOL_M = 0.15
_FPV_EYE_CLEAR_RADIUS_M = 0.03
# how far a HIGH-target pick raises the object during LIFT (low picks lift via the lean-unwind instead)
_LIFT_RISE_M = 0.12
# furthest a drop that does not fit may be slid toward the free part of its surface. Bounded so the slide
# stays well inside the reach tolerance the stand was picked under (0.10 m for a PutInside).
_RECENTRE_CAP_M = 0.08
# A positive gap is required because visual AABBs that merely touch can still overlap PhysX contact
# skins. This matches the separation already used by the initialization-time spawn planner.
PLACEMENT_MIN_GAP_M = placement_packing.MIN_GAP_M
PLACEMENT_EDGE_MARGIN_M = placement_packing.EDGE_MARGIN_M
# LookAt-inspect of a GRABBABLE: how far in FRONT of the head to hold the object (~0.24 m past the pushed FPV
# eye = a natural label-reading distance, within arm reach), and the beat lengths (reach->raise->HOLD->lower->
# release). The HOLD is >=24 frames (>=2 s @12fps) so the reader can read the label.
_INSPECT_FWD_M = 0.40
_N_LOOK_REACH = 12
_N_LOOK_RAISE = 8
_N_LOOK_HOLD = 28
_N_LOOK_LOWER = 8
_N_LOOK_RELEASE = 6
_N_FLIP = 18
_N_PHONE_RAISE = 16
_PHONE_CLEARANCE_PROBE_M = 1.5
_PHONE_MIN_FORWARD_CLEARANCE_M = 0.9
_FRONT_APPLIANCE_SQUAT = 1.0
_FRONT_APPLIANCE_STANDOFF_M = 0.90
_FRONT_DRUM_CONTENT_STANDOFF_M = 0.55
_FRONT_DRUM_INSPECTION_STANDOFF_M = 1.30
# A closed-door front-panel press needs body clearance but no door-sweep clearance. Keep the panel
# inside the reliable arm envelope; the normal body, route, and contact-trajectory gates still apply.
_FRONT_APPLIANCE_TOGGLE_STANDOFF_M = 0.55
_FRONT_DRUM_STAND_DEPTH_SLACK_M = 0.14
_FRONT_DRUM_PREINSERT_CLEARANCE_M = 0.30
_FRONT_DRUM_CROUCH_FRAMES = 6
_FRONT_APERTURE_MARGIN_M = 0.005
_FRONT_OPENING_STANDOFF_M = 0.75
_FRONT_OPENING_ACTION_CAMERA_STANDOFF_M = 1.10
_FRONT_OPENING_ACTION_REACH_FALLBACK_M = 0.68
_FRONT_OPENING_ACTION_CAMERA_FALLBACK_M = 1.00
_FRONT_OPENING_ACTION_CAMERA_CLEARANCE_M = 0.18
_FRONT_OPENING_ACTION_MAX_LEAN_RAD = math.radians(12.0)
_FRONT_OPENING_CONTENT_STANDOFF_M = 0.50
_FRONT_OPENING_PULLBACK_M = 0.44
_FRONT_OPENING_CLOSE_PULLBACK_M = 0.44
_FRONT_OPENING_PLACE_TOL_M = 0.14
_PRE_LIFT_EXTRACT_CAMERA_PULLBACK_MAX_M = 0.85
_LOW_FRONT_INSERT_STANDOFF_M = 0.70
_LOW_FRONT_INSERT_MIN_SQUAT = 2.0
_LOW_FRONT_INSERT_EYE_CLEARANCE_M = 0.12
_LOW_FRONT_INSERT_CAMERA_PULLBACK_MAX_M = 0.06
_LOW_FRONT_INSERT_MAX_LEAN_RAD = math.radians(12.0)
_LOW_OPENABLE_TOP_MAX_M = 1.15
_LOW_OPENABLE_APERTURE_MAX_M = 0.65
_LOW_OPENABLE_HANDLE_MAX_M = 0.90
_CROUCH_CONTEXT_NAV_MAX_M = 0.75
_CONTINUOUS_FRONT_CLOSE_MAX_SHIFT_M = 0.35
_CONTINUOUS_FRONT_CLOSE_MAX_YAW_DEG = 35.0
_CONTINUOUS_FRONT_CLOSE_ROUTE_SLACK_M = 0.08
_STATIONARY_OPERATOR_HANDLE_MARGIN_M = 0.12
_STATIONARY_OPERATOR_STAND_TOL_M = 0.04
_CONFIGURED_OPERATOR_LATERAL_TOL_M = 0.16
_EXPLICIT_OPERATOR_LATERAL_TOL_M = 0.05
_EXPLICIT_OPERATOR_OUTWARD_SLACK_M = 0.15
_OPENABLE_STAND_DISTANCE_EPS_M = 1e-9
_IMPLICIT_CLOSE_GRID_SNAP_TOL_M = 0.05
_IMPLICIT_CLOSE_BASE_DEPTH_SLACK_M = 0.15
_OPENABLE_SWEEP_STANDOFF_M = 0.55
_FRONT_DRUM_SWEEP_STANDOFF_M = 0.42
_OPENABLE_SWEEP_FRACTIONS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
# The placard's outward normal, written by the importers as ``label_face``, is recorded on the
# entity -- task_world_builder uses it for front_stand and the legibility estimate. It is not a
# constraint on where the wearer stands to read one; see produce_look_at.
# EGOCENTRIC LOCOMOTION. A standalone turn eases at 10 deg/frame (12 fps -> a quarter turn in 9 frames,
# a half turn in 18) -- slower than the 15 deg/frame the A* corner beats use, because a turn commanded to
# a human who cannot see is the whole point of the frame rather than a seam between two walks.
# _FWD_BODY_R is the body radius the GRIDLESS forward-clearance fallback grows the furniture AABBs by;
# the nav-grid path needs no such constant (its occupancy is already Minkowski-dilated).
_TURN_DEG_PER_FRAME = 10.0
_FWD_BODY_R = 0.25
# How far a harness square-up may still move a body the ego channel owns: the shared budget the
# ego-lock gate checks against. Same budget as the navigate gate (``ego_route.EGO_ARRIVE_M``);
# duplicated as a literal to avoid an import, and asserted equal in ``_selftest_ego_lock``.
_EGO_SQ_M = 1.25
# Minimum height above the floor for a footprint to count as something you can walk INTO. Without it the
# floor slab itself (top == floor_z) reads as a wall and every [Forward] is blocked.
_FWD_MIN_OBSTACLE_H = 0.20

# Approach-waypoint reach (MoveIt-style pre_grasp -> grasp -> retreat): the pre-grasp standoff along
# the approach direction (clipped), how far the retreat backs off along -approach, and the graze band.
_WP_PRE_MIN = 0.10
_WP_PRE_MAX = 0.18
_WP_POST_BACK = 0.08
_WP_GRAZE_M = 0.02
_BODY_GRAZE_TOL_M = 0.02
_BATCH_IK_VERIFY_MARGIN_M = 0.008
_BATCH_IK_VERIFY_TOP_K = 3

# A broad box carried in the reach/retreat pose sits directly below the FPV eye and can fill most of
# the image for an entire walk. Large box-like props get a separate walking pose: one-handed at the
# right waist, forward enough to clear the thigh, while their grasp-time wrist orientation is retained
# so a horizontal lid does not turn upright in front of the camera. Both dimensions are required so
# long, narrow tools (for example a squeegee) keep their existing carry.
_LARGE_CONTAINER_CARRY_MIN_LONG_M = 0.32
_LARGE_CONTAINER_CARRY_MIN_SHORT_M = 0.20
_LARGE_CONTAINER_CARRY_WRIST_LOCAL = np.array(
    (-0.40, -0.70, 0.24), dtype=np.float64
)
_LARGE_CONTAINER_CARRY_MAX_IK_ERROR_M = 0.04
_LARGE_CONTAINER_CARRY_MAX_ORIENTATION_ERROR = 0.01

# Search-only budgets. A normal reach plan uses at most 14 CCD solves and usually succeeds much earlier.
# Runtime stand search splits its total between the coarse pass and this reserved fine-pass tranche, so
# an unlucky coarse ordering cannot consume every solve before the unsampled 0.10 m cells are considered.
# The final chosen action solve is intentionally outside these budgets, so a search that found a usable
# answer can always render it.
MAX_IK_SOLVES_PER_STAND_SEARCH = 128
RESERVED_IK_SOLVES_FOR_FINE_STAND_SEARCH = 48
RESERVED_IK_SOLVES_FOR_ROUTER_STAND_RETRY = 16
MAX_IK_SOLVES_PER_PLACEMENT_SEARCH = 64

# Right-arm world-sample joints for the arm-clearance check (SMPL+H joint indices, verified in
# smplh_fpv: R_shoulder=17, R_elbow=19, R_wrist=21; MANO right MCP roots=37,40,43,46,49).
_ARM_SAMPLE_J = (19, 21, 17, 37, 40, 43, 46, 49)
_LEFT_ARM_SAMPLE_J = (18, 20, 16, 22, 25, 28, 31, 34)


_FOOT_J = (7, 8, 10, 11)  # SMPL L/R ankle + L/R foot -> lowest = ground contact


@dataclass
class IKSearchBudget:
    limit: int
    used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def consume(self) -> bool:
        if self.exhausted:
            return False
        self.used += 1
        return True

    def consume_many(self, count: int) -> int:
        taken = min(max(0, int(count)), max(0, self.limit - self.used))
        self.used += taken
        return taken


def _reach_tolerance(scene: SceneState) -> float:
    """Return the IK residual accepted by both stand selection and action execution."""
    if getattr(scene, "strict_acceptance", False):
        return _REACH_TOL_M
    return 0.09 if getattr(scene, "soft_init", False) else _REACH_TOL_M


def _body_collision_is_hard(scene: SceneState) -> bool:
    """Whether avatar-body/limb geometry may veto an action in this scene."""
    import action_check

    return bool(getattr(scene, "strict_acceptance", False)) and not (
        action_check.uses_egocentric_camera_collision_policy(scene)
    )


def _goal_reach_tolerance(goal: Goal, default: float) -> float:
    fallback = (goal.meta or {}).get("_egocentric_reach_tol_m")
    if fallback is None:
        return float(default)
    return max(float(default), min(float(fallback), _OPENABLE_REACH_TOL_M))


def _placement_reach_tolerance(
    goal: Goal,
    *,
    into: bool,
    exact_rest_crouch: bool,
) -> float:
    tolerance = _goal_reach_tolerance(goal, 0.10 if into else _REACH_TOL_M)
    return (
        min(tolerance, _EXACT_REST_REACH_TOL_M)
        if exact_rest_crouch
        else tolerance
    )


def _ang_delta(a: float, b: float) -> float:
    """Shortest signed arc a->b in degrees, in (-180, 180]."""
    return (b - a + 180.0) % 360.0 - 180.0


def _pin_z(pose156: np.ndarray, floor_z: float) -> float:
    """Root z that puts the lowest foot of ``pose156`` exactly on ``floor_z`` (kills transition float)."""
    J = core.H.fwd_joints(pose156)
    return floor_z - float(J[list(_FOOT_J), 2].min())


# =====================================================================================================
# Scene + scheduler state.
# =====================================================================================================
@dataclass
class SchedState:
    """Cross-segment continuity the scheduler threads between producers (the ONLY shared state)."""

    root_xy: tuple[float, float]
    root_yaw: float  # target facing at the current stand point
    stand_z: float  # body lift so the feet sit on floor_z
    floor_z: float
    # continues the gait phase across chained walks (no cadence reset at a corner)
    gait_cum_dist: float = 0.0
    walk_end_yaw: float = 0.0  # actual heading of the last walk frame (arrival turn)
    # {otw, rhp, grasp_arm, obj_key} while carrying; None when empty-handed
    held: dict | None = None
    obj_rest_xyz: tuple[float, float, float] | None = None  # active object resting pos
    last_pose156: np.ndarray | None = None  # last body-local pose (for the blend-in)
    last_seg: str = ""  # last seg name (a walk after a walk skips re-depart)
    failed_contact_trajectory: (
        tuple[str, str | None, tuple[tuple[float, float, float], ...]] | None
    ) = None
    failed_contact_target_key: str | None = None
    seated_z: float | None = None
    # A low front-opening fixture owns the crouch until it is closed.  `stand_z`
    # remains the immutable upright baseline; these fields describe the live pose.
    crouch_owner: str | None = None
    crouch_fraction: float = 0.0
    crouch_root_z: float | None = None
    # Set by the FIRST egocentric locomotion verb and never cleared: from that point the body belongs
    # to the ego channel and a NAMED navigate may no longer carry it across the room (see
    # produce_navigate). Stays False for every program that never issues [Forward]/[Turn*].
    ego_locked: bool = False


# =====================================================================================================
# Small pose helpers (thin wrappers over the reused core).
# =====================================================================================================
def _manip(
    rig,
    arm: dict,
    hand_close: float,
    lean_frac: float = 0.0,
    squat_frac: float = 0.0,
    *,
    left_hand_close: float = 0.0,
) -> np.ndarray:
    """A standing (or crouched) manipulation pose: arm + (partial) grip + forward spine lean fraction +
    optional knee/hip/ankle squat fraction (for a low reach; the caller re-pins the root z via _pin_z).
    squat_frac defaults 0 (no squat)."""
    return _apply_squat(
        wpp._add_lean(
            core.assemble_pose(
                rig,
                arm,
                (0.0, 0.0, 0.0),
                hand_close,
                left_hand_close=left_hand_close,
            ),
            _LEAN_RAD * lean_frac,
        ),
        squat_frac,
    )


def _mirror_world_point_for_left_hand(
    point, stand_xy, yaw: float, stand_z: float
) -> np.ndarray:
    """Reflect a world target across the actor's sagittal plane."""
    rz, world_offset = wpp._grasp_frame(stand_xy, yaw, stand_z)
    local = rz.T @ (
        np.asarray(point, dtype=np.float64) - np.asarray(world_offset, dtype=np.float64)
    )
    local[0] *= -1.0
    return rz @ local + np.asarray(world_offset, dtype=np.float64)


def _merge_left_action_arm(carry_arm: dict, left_action_arm: dict) -> dict:
    combined = {joint: list(value) for joint, value in carry_arm.items()}
    combined.update({joint: list(value) for joint, value in left_action_arm.items()})
    return combined


def _held_right_hand(st: SchedState) -> bool:
    held = getattr(st, "held", None)
    return bool(held is not None and held.get("hand", "right") == "right")


def _obj_carry(st: SchedState) -> tuple:
    """The Frame.obj tuple for the CURRENT object state (carried / resting / hidden)."""
    if st.held is not None:
        return ("carry", st.held["otw"])
    if st.obj_rest_xyz is not None:
        return ("rest", st.obj_rest_xyz)
    return ("hide", None)


def _stamp_obj_path(frames: list, scene: SceneState, obj_key: str) -> None:
    """MULTI-graspable scenes: tag every object-bearing frame ("rest"/"carry") with the object's world
    prim path so the render writes the correct prim. No-op when the entity has no prim_path (single-object
    scenes -> obj_path stays None). Producers each handle ONE object, so this is safe."""
    pp = getattr(scene.resolve(obj_key), "prim_path", None)
    if not pp:
        return
    for fr in frames:
        if fr.obj[0] in ("carry", "rest", "pose"):
            fr.obj_path = pp


def _read_world_col(stage, prim_path):
    """A prim's world transform as a COLUMN-vector 4x4 (p_world = T @ p_local), matching core.set_object_world
    (which transposes to the USD row-vector). None if the prim is absent."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return None
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m = np.array(xc.GetLocalToWorldTransform(prim), dtype=np.float64).reshape(4, 4)
    return m.T


def _write_world_col(stage, prim_path, T_col) -> bool:
    """Write a COLUMN-vector 4x4 world transform to a prim -- the inverse of ``_read_world_col`` (mirrors
    ``core.set_object_world``'s transpose to the USD row-vector). WELDS an assembled part at execute-time so
    the pre-render, live-prim ATTACHED scorer reads its true mated pose (the render loop still re-drives the
    part per frame; this only makes the end-state authoritative for the goal check, mirroring how PUT's
    per-command settle moves a placed prim before scoring). No-op returning False if the prim is absent."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        return False
    T = np.asarray(T_col, dtype=np.float64)
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(
        Gf.Matrix4d(*[float(x) for x in T.T.flatten()])
    )
    return True


def _rotate_pose_about_pivot(
    pose: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    degrees: float,
) -> np.ndarray:
    """Rotate a world pose around a world-space pivot without moving that pivot."""
    unit_axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(unit_axis))
    if norm <= 1e-9:
        raise ValueError("orientation-error axis has zero length")
    x, y, z = unit_axis / norm
    angle = math.radians(float(degrees))
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    rotation = np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=np.float64,
    )
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rotation
    pivot_xyz = np.asarray(pivot, dtype=np.float64)[:3]
    delta[:3, 3] = pivot_xyz - rotation @ pivot_xyz
    return delta @ np.asarray(pose, dtype=np.float64)


def _declared_carried_parent_map(scene) -> dict[str, str]:
    parents = dict(getattr(scene, "on", {}) or {})
    parents.update(dict(getattr(scene, "inside", {}) or {}))
    parents.update(dict(getattr(scene, "attached", {}) or {}))
    return parents


def _carried_parent_map(scene) -> dict[str, str]:
    """Resolve one stable transport parent per object.

    Executor-declared relations are authoritative. Live geometry may report several
    simultaneous supports for one object, so it only fills objects that do not
    already have a declared parent.
    """
    parents = _declared_carried_parent_map(scene)
    try:
        physical = scene.relations(source="physics")
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, ValueError):
        physical = set()
    live_relations = sorted(
        (
            str(subject),
            0 if predicate is isaac_goal.Pred.INSIDE else 1,
            str(target),
        )
        for predicate, subject, target in physical
        if predicate in {isaac_goal.Pred.ON, isaac_goal.Pred.INSIDE}
        and target is not None
    )
    for subject, _priority, target in live_relations:
        parents.setdefault(subject, target)
    return parents


def _carried_descendants(parents: dict[str, str], root_key: str) -> list[str]:
    contents, frontier, seen = [], [root_key], set()
    while frontier:
        cur = frontier.pop()
        for _obj, _cont in parents.items():
            if _cont == cur and _obj != root_key and _obj not in seen:
                seen.add(_obj)
                contents.append(_obj)
                frontier.append(_obj)
    return contents


def _carried_root_inverse(scene, root_key: str, declared_contents: set[str]):
    root = scene.entities.get(root_key)
    if root is None or not root.prim_path:
        if declared_contents:
            raise PrimitiveFailure(
                PrimitiveError.NO_OBJECT,
                f"carried root {root_key!r} has no live prim for declared contents "
                f"{sorted(declared_contents)}",
            )
        return None
    root_world = _read_world_col(scene.stage, root.prim_path)
    if root_world is None:
        if declared_contents:
            raise PrimitiveFailure(
                PrimitiveError.NO_OBJECT,
                f"carried root {root_key!r} has no readable live transform for "
                f"declared contents {sorted(declared_contents)}",
            )
        return None
    try:
        return np.linalg.inv(root_world)
    except np.linalg.LinAlgError:
        if declared_contents:
            raise PrimitiveFailure(
                PrimitiveError.NO_OBJECT,
                f"carried root {root_key!r} has a singular live transform for "
                f"declared contents {sorted(declared_contents)}",
            )
        return None


def _gather_carried(scene, x_key):
    """Contents currently IN/ON ``x_key`` (TRANSITIVELY) + each one's rigid transform RELATIVE to x_key, so a
    carried container drags its contents as ONE body (real-world: pick up a bowl, its food rides along).
    rel = inv(X_world) @ content_world (column-vector). Returns ``[(content_prim_path, rel_4x4), ...]`` -- empty
    when nothing is inside/on x_key or there is no live stage."""
    stage = getattr(scene, "stage", None)
    if stage is None:
        return []
    declared_contents = set(
        _carried_descendants(_declared_carried_parent_map(scene), x_key)
    )
    contents = _carried_descendants(_carried_parent_map(scene), x_key)
    if not contents:
        return []
    Xinv = _carried_root_inverse(scene, x_key, declared_contents)
    if Xinv is None:
        return []
    out = []
    collected = set()
    for c in contents:
        ce = scene.entities.get(c)
        if ce is None or not ce.prim_path:
            continue
        C = _read_world_col(stage, ce.prim_path)
        if C is not None:
            out.append((ce.prim_path, Xinv @ C))
            collected.add(c)
    missing = sorted(declared_contents - collected)
    if missing:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT,
            f"declared contents of {x_key!r} have no readable live transform: {missing}",
        )
    return out


def sync_carried_to_root(scene, root_key: str, carried) -> None:
    """Write a carried subtree to the root's current authoritative stage pose."""
    if not carried or getattr(scene, "stage", None) is None:
        return
    root = scene.entities.get(root_key)
    if root is None or not root.prim_path:
        return
    root_world = _read_world_col(scene.stage, root.prim_path)
    if root_world is None:
        return
    by_path = {
        entity.prim_path: entity
        for entity in scene.entities.values()
        if entity.prim_path
    }
    for path, relative in carried:
        moved = root_world @ relative
        if not _write_world_col(scene.stage, path, moved):
            continue
        entity = by_path.get(path)
        if entity is not None:
            (x, y), zmin, _zmax = scene.object_bbox(entity.name)
            entity.xy, entity.z = (float(x), float(y)), float(zmin)
            rest_poses = getattr(scene, "object_rest_pose", None)
            if rest_poses is not None:
                rest_poses[entity.name] = moved.copy()
            assembly_poses = getattr(scene, "assembly_pose", None)
            if assembly_poses is not None and (
                entity.name in assembly_poses
                or entity.name in getattr(scene, "attached", {})
            ):
                assembly_poses[entity.name] = moved.copy()
    scene.invalidate_bounds()


def _stamp_carried(frames: list, carried) -> None:
    """Stamp ``fr.carried`` on every carry/rest frame so the render writes the ridden contents alongside the
    container. No-op when there is nothing to carry (single object)."""
    if not carried:
        return
    ct = tuple(carried)
    for fr in frames:
        if fr.obj[0] in ("carry", "rest"):
            fr.carried = ct


def _stamp_held_carry_frames(
    frames: list, scene: SceneState, held_at_start: dict | None
) -> None:
    """Fill identity for carry frames emitted by an action that began with a held object."""
    if held_at_start is None:
        return
    pp = getattr(scene.resolve(held_at_start["obj_key"]), "prim_path", None)
    if pp:
        for frame in frames:
            if frame.obj[0] == "carry" and not getattr(frame, "obj_path", None):
                frame.obj_path = pp
    _stamp_carried(frames, held_at_start.get("carried"))


def _solve_reach_arm(
    rig,
    arm_from: dict,
    obj_world: np.ndarray,
    stand_xy,
    yaw,
    stand_z,
    lean_override=None,
    squat_frac: float = 0.0,
) -> tuple[dict, float]:
    """Warp shoulder+elbow so the grip seat of ``arm_from`` lands the object at ``obj_world`` (CCD, once).

    Generalized ``humanoid_motion.solve_place_arm`` (that one is hard-wired to DeskB); here the
    stand point + facing come from the scheduler so a PLACE/reach works at any surface. ``lean_override``
    (rad, from the pose-planner) is used verbatim when given so the reach adopts the SAME forward lean the
    gate approved; None -> the height-only lean. ``squat_frac``
    (from the pose-planner) crouches the reach base so the shoulder drops to a LOW target -- the caller
    must pass the crouch-pinned ``stand_z`` to match; 0.0 -> upright."""
    rz, world_off = wpp._grasp_frame(stand_xy, yaw, stand_z)
    lean_rad = (
        float(lean_override)
        if lean_override is not None
        else _lean_for_height(float(obj_world[2]))[0]
    )
    base = _reach_base_pose(rig, arm_from, lean_rad, squat_frac, 1.0)
    pose, err = core.warp_arm_to_seat(
        rig, base, rz, world_off, np.asarray(obj_world, dtype=np.float64)
    )
    return wpp._arm_from_pose(pose), float(err)


def _large_container_navigation_arm(
    rig,
    scene: SceneState,
    st: SchedState,
    entity,
    contact_arm: dict,
    contact_lean_fraction: float,
    contact_squat_fraction: float,
) -> dict | None:
    """Return a camera-clear walking arm for a large box/lid, else ``None``.

    The object-to-wrist transform remains untouched. Matching the wrist orientation from the exact
    contact pose therefore keeps the object's resting orientation relative to the body while moving
    its grasp point down and to the right. The navigation layer eases into this arm only after PICK;
    manipulation producers continue to use ``grasp_arm`` for cabinet continuity and placement IK.
    """
    category = str(
        (getattr(entity, "meta", None) or {}).get("asset_category") or ""
    ).casefold().replace("-", "_")
    if category != "lid" and category != "box" and not category.endswith("_box"):
        return None
    half_x, half_y, _height = _obj_extent(scene, entity.name)
    long_span, short_span = sorted((2.0 * half_x, 2.0 * half_y), reverse=True)
    if (
        long_span < _LARGE_CONTAINER_CARRY_MIN_LONG_M
        or short_span < _LARGE_CONTAINER_CARRY_MIN_SHORT_M
    ):
        return None

    contact_pose = _manip(
        rig,
        contact_arm,
        1.0,
        contact_lean_fraction,
        contact_squat_fraction,
    )
    contact_wrist_rotation = core.H.wrist_frame(core.H.fwd_joints(contact_pose))[0]
    carry_base = _manip(rig, contact_arm, 1.0)
    rz, world_offset = wpp._grasp_frame(st.root_xy, st.root_yaw, st.stand_z)
    target_world = (
        rz @ _LARGE_CONTAINER_CARRY_WRIST_LOCAL
        + np.asarray(world_offset, dtype=np.float64)
    )
    carry_pose, position_error = core.warp_arm_to_wrist(
        rig,
        carry_base,
        rz,
        world_offset,
        target_world,
    )
    if position_error > _LARGE_CONTAINER_CARRY_MAX_IK_ERROR_M:
        print(
            f"[carry-pose] {entity.name}: skip unreachable waist carry "
            f"({position_error * 1000:.0f}mm)"
        )
        return None
    wrist_aa, orientation_error, wrist_bend = core.solve_wrist_orient(
        rig,
        carry_pose,
        contact_wrist_rotation,
    )
    if orientation_error > _LARGE_CONTAINER_CARRY_MAX_ORIENTATION_ERROR:
        print(
            f"[carry-pose] {entity.name}: skip unstable wrist orientation "
            f"(error={orientation_error:.3f})"
        )
        return None
    carry_pose = np.asarray(carry_pose, dtype=np.float32).copy()
    carry_pose[core.WRIST_J * 3 : core.WRIST_J * 3 + 3] = wrist_aa
    print(
        f"[carry-pose] {entity.name}: {long_span:.2f}x{short_span:.2f}m "
        f"waist carry err={position_error * 1000:.0f}mm wrist={wrist_bend:.0f}deg"
    )
    return wpp._arm_from_pose(carry_pose)


# =====================================================================================================
# Approach-waypoint reach: a MoveIt-style pre_grasp -> grasp -> retreat sequence, each leg
# solved by the SAME warp (_solve_reach_arm / _reach_arm_leaned), with a per-leg arm-vs-furniture
# clearance check. The approach direction is DERIVED FROM GEOMETRY (stand point -> contact), so the
# hand comes in from the open side the body already stood on -- never hardcoded world-front.
# =====================================================================================================
def _fwd_arm_points(
    pose: np.ndarray, rz=None, world_off=None, *, side: str = "right"
) -> np.ndarray:
    """Arm sample points for collision checks, optionally placed in world coordinates."""
    J = core.H.fwd_joints(np.asarray(pose))
    samples = _LEFT_ARM_SAMPLE_J if side == "left" else _ARM_SAMPLE_J
    elbow, wrist = (18, 20) if side == "left" else (19, 21)
    pts = [J[j] for j in samples]
    pts.append(0.5 * (J[elbow] + J[wrist]))
    P = np.asarray(pts, dtype=np.float64)
    if rz is not None and world_off is not None:
        P = P @ np.asarray(rz, dtype=np.float64).T + np.asarray(
            world_off, dtype=np.float64
        )
    return P


def _arm_points_from_joints(joints: np.ndarray, rz=None, world_off=None) -> np.ndarray:
    """Vectorized counterpart of ``_fwd_arm_points`` for precomputed batched joints."""
    J = np.asarray(joints, dtype=np.float64)
    single = J.ndim == 2
    if single:
        J = J[None, ...]
    points = J[:, _ARM_SAMPLE_J]
    midpoint = 0.5 * (J[:, 19] + J[:, 21])
    points = np.concatenate((points, midpoint[:, None, :]), axis=1)
    if rz is not None and world_off is not None:
        rotations = np.asarray(rz, dtype=np.float64)
        offsets = np.asarray(world_off, dtype=np.float64)
        if rotations.shape == (3, 3):
            rotations = np.broadcast_to(rotations, (J.shape[0], 3, 3))
        if offsets.shape == (3,):
            offsets = np.broadcast_to(offsets, (J.shape[0], 3))
        points = np.einsum("bkj,bij->bki", points, rotations) + offsets[:, None, :]
    return points[0] if single else points


def _verts_in_boxes(
    P: np.ndarray,
    boxes,
    exclude_sphere: tuple[np.ndarray, float] | None = None,
) -> tuple[float, str | None]:
    """Worst penetration depth (m) of points ``P`` (K,3) into any AABB in ``boxes`` [(name, mn, mx)],
    with the name of the deepest-hit box. depth of an inside point = min distance to a face; the box's
    score = the max over its inside points; the worst box wins. (0.0, None) if nothing is inside."""
    P = np.asarray(P, dtype=np.float64)
    if exclude_sphere is not None:
        center, radius = exclude_sphere
        P = P[
            np.linalg.norm(P - np.asarray(center, dtype=np.float64), axis=1)
            > float(radius)
        ]
        if not P.size:
            return 0.0, None
    worst, hit = 0.0, None
    for name, mn, mx in boxes:
        inside = np.all((P > mn) & (P < mx), axis=1)
        if not bool(inside.any()):
            continue
        pin = P[inside]
        depth = float(np.minimum(pin - mn, mx - pin).min(axis=1).max())
        if depth > worst:
            worst, hit = depth, name
    return worst, hit


# ±X/±Y/±Z outward unit face normals, indexed to match np.argmin over [(p-mn), (mx-p)] flattened as
# [-x_gap, -y_gap, -z_gap, +x_gap, +y_gap, +z_gap] -> the shallowest face's OUTWARD normal.
_FACE_NORMALS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _escape_from_boxes(P: np.ndarray, boxes):
    """Like ``_verts_in_boxes`` but ALSO returns the shallowest-face OUTWARD unit normal ``n_hat`` for
    the deepest inside point of the winning box (the direction to push that point out of the box by the
    least distance). Returns (depth, name, n_hat); n_hat is None when nothing is inside. For the deepest
    point p in the winning box, gaps = (p-mn, mx-p) over the 3 axes x 2 sides; argmin over the 6 gaps
    picks the closest face and ``_FACE_NORMALS`` maps it to the outward ±X/±Y/±Z unit vector."""
    P = np.asarray(P, dtype=np.float64)
    worst, hit, n_hat = 0.0, None, None
    for name, mn, mx in boxes:
        inside = np.all((P > mn) & (P < mx), axis=1)
        if not bool(inside.any()):
            continue
        pin = P[inside]
        face_gap = np.minimum(pin - mn, mx - pin)  # (K,3) per-axis nearest-face gap
        depth_per = face_gap.min(axis=1)  # (K,) depth of each inside point
        k = int(np.argmax(depth_per))  # deepest inside point in this box
        depth = float(depth_per[k])
        if depth > worst:
            gaps6 = np.concatenate([pin[k] - mn, mx - pin[k]])  # [-x,-y,-z,+x,+y,+z]
            worst, hit, n_hat = depth, name, _FACE_NORMALS[int(np.argmin(gaps6))].copy()
    return worst, hit, n_hat


def pack_boxes(boxes, inflate: float = 0.0):
    """Pack ``[(name, mn, mx)]`` into ``(names, MN, MX)`` -- (N,3) float64 corner arrays grown outward by
    ``inflate`` -- the form ``escape_from_boxes_packed`` consumes. A scene's solid boxes are constant for a
    whole episode, so a per-frame caller packs ONCE and reuses.

    The inflate is applied PER BOX (``np.asarray(mn) - inflate``) before the stack, so a corner stored in a
    narrower dtype rounds exactly where the scalar per-box loop rounded it."""
    names = [b[0] for b in boxes]
    if not boxes:
        return (
            names,
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )
    mn = np.array([np.asarray(b[1]) - inflate for b in boxes], dtype=np.float64)
    mx = np.array([np.asarray(b[2]) + inflate for b in boxes], dtype=np.float64)
    return names, mn, mx


def escape_from_boxes_packed(P: np.ndarray, names, MN: np.ndarray, MX: np.ndarray):
    """``_escape_from_boxes`` over PACKED boxes: the same ``(depth, name, n_hat)`` with no Python loop.

    ``MN``/``MX`` are the (N,3) already-inflated corner arrays from ``pack_boxes``; ``names[i]`` names row
    i. BIT-IDENTICAL to the scalar loop, not merely close -- every arithmetic step is the same elementwise
    float64 subtraction (broadcasting changes no per-element rounding) and every tie-break is reproduced:
    a box's deepest point is the FIRST ``argmax`` over its inside points in P order, the winning box is the
    FIRST row attaining the maximum depth (the loop's strict ``>`` keeps the earliest maximum), and the
    escape face is ``argmin`` over ``[p-mn, mx-p]``.

    The prefilter is exact, not approximate: a box disjoint from the point cloud's own AABB has ``mx <=
    min(P)`` or ``mn >= max(P)`` on some axis, so no point can satisfy the strict ``mn < p < mx`` there. It
    drops essentially all of a several-thousand-box scene before the (N,K,3) work."""
    P = np.asarray(P, dtype=np.float64)
    if MN.shape[0] == 0 or P.size == 0:
        return 0.0, None, None
    keep = np.flatnonzero(np.all((MX > P.min(axis=0)) & (MN < P.max(axis=0)), axis=1))
    if keep.size == 0:
        return 0.0, None, None
    mn, mx = MN[keep][:, None, :], MX[keep][:, None, :]  # (M,1,3)
    Q = P[None, :, :]  # (1,K,3)
    inside = np.all((Q > mn) & (Q < mx), axis=2)  # (M,K)
    if not inside.any():
        return 0.0, None, None
    depth_per = np.where(inside, np.minimum(Q - mn, mx - Q).min(axis=2), -np.inf)
    kk = np.argmax(depth_per, axis=1)  # deepest inside point of each box
    b = int(np.argmax(depth_per[np.arange(kk.size), kk]))  # first box at the max depth
    p = P[int(kk[b])]
    gaps6 = np.concatenate([p - mn[b, 0], mx[b, 0] - p])  # [-x,-y,-z,+x,+y,+z]
    return (
        float(depth_per[b, int(kk[b])]),
        names[int(keep[b])],
        _FACE_NORMALS[int(np.argmin(gaps6))].copy(),
    )


def plan_reach_waypoints(
    scene: SceneState, key: str, stand_xy, contact, entity=None
) -> dict:
    """Three reach waypoints {pre, grasp, retreat, approach} for reaching ``contact`` (world xyz) at
    entity ``key`` from ``stand_xy``. If the entity has NO footprint (a plain object on a desk) the
    waypoints COLLAPSE to a direct reach (pre==grasp==contact, retreat==contact+lift). Otherwise the
    approach is the horizontal unit from the stand point to the contact
    (respects the open side), and pre/retreat stand off / back out along it clear of the furniture."""
    e = entity or scene.resolve(key)
    contact = np.asarray(contact, dtype=np.float64)
    if e.foot_half is None:
        return {
            "pre": contact.copy(),
            "grasp": contact.copy(),
            "retreat": contact + np.array([0.0, 0.0, _LIFT_RISE_M]),
            "approach": np.array([0.0, -1.0, 0.0]),
        }
    d = contact[:2] - np.asarray(stand_xy, dtype=np.float64)
    n = float(np.linalg.norm(d))
    approach = (
        np.array([d[0] / n, d[1] / n, 0.0]) if n > 1e-6 else np.array([0.0, -1.0, 0.0])
    )
    bbox_depth = (
        abs(approach[0]) * 2.0 * e.foot_half[0]
        + abs(approach[1]) * 2.0 * e.foot_half[1]
    )
    pre_dist = float(np.clip(0.5 * bbox_depth + 0.10, _WP_PRE_MIN, _WP_PRE_MAX))
    post_up = (
        (e.top_z - float(contact[2]) + 0.10)
        if (e.top_z is not None and e.top_z > float(contact[2]))
        else _LIFT_RISE_M
    )
    post_up = float(np.clip(post_up, _LIFT_RISE_M, 0.35))
    pre = contact - approach * pre_dist
    retreat = contact + np.array([0.0, 0.0, post_up]) - approach * _WP_POST_BACK
    return {
        "pre": pre,
        "grasp": contact.copy(),
        "retreat": retreat,
        "approach": approach,
    }


def _solve_and_check(
    rig,
    arm_from,
    point,
    stand_xy,
    yaw,
    stand_z,
    boxes,
    label,
    *,
    strict: bool = False,
) -> tuple[dict, float, float, str | None]:
    """Solve one reach leg to ``point`` (via the shared warp) and check the resulting arm against the
    furniture boxes; log the leg, its residual, the worst arm-clearance depth + which box it hit."""
    arm, err = _solve_reach_arm(rig, arm_from, point, stand_xy, yaw, stand_z)
    rz, world_off = wpp._grasp_frame(stand_xy, yaw, stand_z)
    P = _fwd_arm_points(_manip(rig, arm, 1.0, 0.0), rz, world_off)
    depth, hit = _verts_in_boxes(P, boxes)
    print(
        f"[reach] {label}: err={err * 1000:.1f}mm arm-clear worst={depth * 1000:.0f}mm hit={hit or '-'}"
    )
    if strict and depth > 0.002:
        raise PrimitiveFailure(
            PrimitiveError.TRAJECTORY_COLLISION,
            f"{label} intersects {hit or 'scene geometry'} by {depth * 1000.0:.0f}mm",
        )
    return arm, float(err), float(depth), hit


def _reach_boxes(scene: SceneState):
    collision_world = getattr(scene, "cworld", None)
    if collision_world is not None:
        boxes = list(collision_world.boxes())
    else:
        boxes = list(scene.solid_boxes())
    for key in getattr(scene, "open_objs", None) or ():
        entity = scene.entities.get(key)
        if entity is None or (entity.meta or {}).get("openable_kind") != "lid":
            continue
        leaf_paths = tuple(str(path).rstrip("/") for path in entity.leaf_paths if path)
        boxes = [
            row
            for row in boxes
            if not any(
                str(row[0]).rstrip("/") == path or str(row[0]).startswith(path + "/")
                for path in leaf_paths
            )
        ]
        boxes.extend(_open_leaf_boxes(scene, entity))
    return boxes


def _require_arm_clear(
    scene: SceneState, depth: float, hit: str | None, label: str
) -> None:
    if not _body_collision_is_hard(scene) or depth <= 0.002:
        return
    raise PrimitiveFailure(
        PrimitiveError.TRAJECTORY_COLLISION,
        f"{label} intersects {hit or 'scene geometry'} by {depth * 1000.0:.0f}mm",
    )


def _assert_ego_unlocked(
    scene, st: SchedState, goal: Goal, force_straight: bool
) -> None:
    """Refuse to CARRY a body that the egocentric channel has taken over.

    A program that steers the body with [Forward]/[Turn*] is guiding someone who cannot see the room,
    and the whole claim under test is that the guidance gets them there. But every manipulation verb
    auto-inserts a navigate to its target, so without this gate a guidance run that failed -- every
    leg ``path_blocked``, zero frames, human unmoved -- would be silently rescued by the next
    ``[Grab]``, which walks the body across the room and scores 1.00, and the task would no longer
    test what it claims.

    The question the gate asks is *did the guidance do its job*, so it measures the body against the
    point the legs were compiled to DELIVER it to (``scene.ego_delivery``), not against the stand. A
    stand that no whole pace can enter -- one hugging the counter it works at, which the body-inflated
    grid calls blocked -- is geometry's doing, and charging that metre to the guidance would fail rows
    for the one thing no route could have fixed. With no compiled delivery (a navigate the guidance
    never planned) the target itself is the reference, which is the strict reading.

    Inert for every program with no ego verb, and for ``produce_forward``'s own straight leg."""
    if force_straight or not st.ego_locked or goal.target_xy is None:
        return
    import ego_route

    tgt = (float(goal.target_xy[0]), float(goal.target_xy[1]))
    plan = (getattr(scene, "ego_delivery", None) or {}).get(goal.target_key)
    ref = plan or tgt
    gap = math.hypot(ref[0] - st.root_xy[0], ref[1] - st.root_xy[1])
    if gap <= ego_route.EGO_ARRIVE_M:
        # Printed even when ALLOWED: this is the one number that says how much of the travelling the
        # harness did instead of the guidance, so it is always logged.
        carry = math.hypot(tgt[0] - st.root_xy[0], tgt[1] - st.root_xy[1])
        if carry > 0.05:
            print(
                f"[ego] square-up: harness carries {carry:.2f} m to <{goal.target_key or 'xy'}>"
            )
        return
    raise PrimitiveFailure(
        PrimitiveError.PATH_BLOCKED,
        f"ego-guided body may not be carried to <{goal.target_key or 'xy'}> "
        f"({tgt[0]:.2f},{tgt[1]:.2f}): the guidance was to leave it at "
        f"({ref[0]:.2f},{ref[1]:.2f}) and it is at "
        f"({st.root_xy[0]:.2f},{st.root_xy[1]:.2f}), {gap:.2f} m short "
        f"(budget {ego_route.EGO_ARRIVE_M:.2f} m)",
    )


# =====================================================================================================
# Producers: (rig, gait, scene, state, goal) -> list[Frame], mutating state. One per GROUNDED verb.
# Each raises PrimitiveFailure on a precondition/reachability violation (NO faked frames -> the RL
# loop sees the real failure). The choreography mirrors the proven demo beats.
# =====================================================================================================
def _navigation_preserves_crouch(
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    *,
    force_straight: bool,
) -> bool:
    owner = getattr(st, "crouch_owner", None)
    depth = float(getattr(st, "crouch_fraction", 0.0))
    if owner is None or depth <= 1e-6 or goal.target_xy is None or force_straight:
        return False
    if math.dist(tuple(st.root_xy), tuple(goal.target_xy)) > _CROUCH_CONTEXT_NAV_MAX_M:
        return False
    explicit_owner = (goal.meta or {}).get("_crouch_owner")
    if explicit_owner is not None:
        return str(explicit_owner) == str(owner)
    if goal.target_key is None:
        # Runtime reach retries omit a target key; the tight distance gate confines
        # this exception to a local reposition at the already-open fixture.
        return True
    target = str(goal.target_key)
    return _inside_owner(scene, target, str(owner)) or (
        target,
        str(owner),
    ) in (getattr(scene, "tracked_near", None) or set())


def _stand_up_from_crouch(rig, scene: SceneState, st: SchedState) -> list[wpp.Frame]:
    if (
        getattr(st, "crouch_owner", None) is None
        or float(getattr(st, "crouch_fraction", 0.0)) <= 1e-6
    ):
        return []
    held = st.held
    arm = held["grasp_arm"] if held is not None else getattr(rig, "arm_rest", rig.arm_grasp)
    frames = [
        replace(
            frame,
            rhp=held.get("rhp") if held is not None else None,
            seg="stand",
        )
        for frame in _arrive_beats(
            rig,
            st,
            arm,
            1.0 if held is not None else 0.0,
            _obj_carry(st),
        )
    ]
    if held is not None:
        _stamp_obj_path(frames, scene, held["obj_key"])
        _stamp_carried(frames, held.get("carried"))
    return frames


def _crouch_navigation_frames(
    frames: list[wpp.Frame], st: SchedState, depth: float, prior_pose: np.ndarray | None
) -> list[wpp.Frame]:
    crouched = []
    for frame in frames:
        pose = _shift_squat(frame.pose156, depth)
        crouched.append(replace(frame, pose156=pose, z=_pin_z(pose, st.floor_z)))
    if crouched:
        st.last_pose156 = crouched[-1].pose156.copy()
        st.crouch_root_z = float(crouched[-1].z)
    else:
        st.last_pose156 = prior_pose
        if prior_pose is not None:
            st.crouch_root_z = _pin_z(prior_pose, st.floor_z)
    return crouched


def produce_navigate(
    rig,
    g: gait.Gait,
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    sd_world=None,
    _force_straight=False,
) -> list[wpp.Frame]:
    """Stride from the current root to ``goal.target_xy``; carry the held object if holding.

    Begins with a DEPART walk-in (unless already walking): turn in place to the path heading while
    easing the previous pose into the gait cycle and the root height into the foot-pinned walk height.
    This kills the stand->stride pop (and any turn-to-path yaw flip) at the manip->walk seam."""
    if goal.target_xy is None:
        # An UNRESOLVED navigate carries target_key (the entity to stand at) with target_xy None.
        # The render loop resolves these at RUNTIME (init_validation.pick_stand_runtime, nearest-reachable
        # from the human's LIVE position) BEFORE dispatch, so this branch is reached only by OFFLINE callers
        # (selftests / demos via execute) with no physics handle -> fall back to the scene's declared
        # stand_pose_for (position-agnostic, a pre-picked stand). A navigate with neither xy nor key is a
        # real error.
        if goal.target_key is not None:
            stand, yaw = scene.stand_pose_for(goal.target_key)
            goal.target_xy = (float(stand[0]), float(stand[1]))
            if goal.target_yaw is None:
                goal.target_yaw = float(yaw)
        else:
            raise PrimitiveFailure(
                PrimitiveError.PATH_BLOCKED, "NAVIGATE_TO without target_xy"
            )
    if (goal.meta or {}).get("_reuse_current_stand"):
        if (
            math.dist(tuple(st.root_xy), tuple(goal.target_xy)) > 1e-6
            or goal.target_yaw is None
            or abs(_ang_delta(float(st.root_yaw), float(goal.target_yaw))) > 1e-6
        ):
            raise PrimitiveFailure(
                PrimitiveError.PATH_BLOCKED,
                "current-stand reuse target does not match the live pose",
            )
        return []
    path_override = (goal.meta or {}).get("_camera_safe_navigation_path")
    force_straight = _force_straight or bool(
        (goal.meta or {}).get("_camera_safe_navigation_fallback", False)
    )
    _assert_ego_unlocked(scene, st, goal, force_straight or bool(path_override))
    # ONE walk producer for both cases. With a per-episode nav grid (scene.nav_grid, set only when
    # scene.plan_nav and the start+stands were all reachable) it walks a planned collision-free
    # polyline; without one it walks the straight line. Either way the body heading is steered per
    # frame by ``heading_controller``, rather than by stationary turn beats or by writing target_yaw
    # into state and letting the next segment absorb a 125 deg/frame teleport.
    # _force_straight: skip planning even when a grid exists.
    if path_override:
        import nav_grid

        grid = nav_grid.egocentric_passable_view(scene.nav_grid, scene)
        if grid is None:
            raise PrimitiveFailure(
                PrimitiveError.PATH_BLOCKED,
                "camera-safe navigation path has no precise fallback grid",
            )
    else:
        grid = None if force_straight else getattr(scene, "nav_grid", None)
    preserve_crouch = _navigation_preserves_crouch(
        scene,
        st,
        goal,
        force_straight=force_straight,
    )
    snapshot = _navigation_state_snapshot(st)
    prefix = []
    prior_pose = snapshot["last_pose156"]
    crouch_depth = float(getattr(st, "crouch_fraction", 0.0))
    if preserve_crouch and prior_pose is not None:
        st.last_pose156 = _shift_squat(prior_pose, -crouch_depth)
    elif getattr(st, "crouch_owner", None) is not None:
        prefix = _stand_up_from_crouch(rig, scene, st)
    try:
        frames = _camera_safe_navigation_frames(
            rig,
            g,
            scene,
            st,
            goal,
            grid,
            path_override=path_override,
        )
    except PrimitiveFailure:
        _restore_navigation_state(st, snapshot)
        raise
    if preserve_crouch:
        frames = _crouch_navigation_frames(frames, st, crouch_depth, prior_pose)
    elif getattr(st, "crouch_owner", None) is not None:
        _clear_crouch_hold(st)
    return [*prefix, *frames]


def _arrive_beats(
    rig,
    st: SchedState,
    target_arm: dict,
    hand_close: float,
    obj_state: tuple,
    *,
    squat_frac: float = 0.0,
) -> list[wpp.Frame]:
    """Blend the last emitted pose -> a stand pose (arm+grip), turning to face st.root_yaw. Reads/updates
    st.last_pose156. Absorbs the walk->manip transition (and any arrival turn) in _N_ARRIVE frames."""
    target = _manip(rig, target_arm, hand_close, 0.0, squat_frac)
    prev = st.last_pose156 if st.last_pose156 is not None else target
    frames = []
    for i in range(_N_ARRIVE):
        s = (i + 1) / _N_ARRIVE
        pose = gait.blend156(prev, target, s)
        yaw = st.walk_end_yaw + _ang_delta(st.walk_end_yaw, st.root_yaw) * s
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                yaw,
                _pin_z(pose, st.floor_z),
                fpv_mode="look",
                obj=obj_state,
                seg="arrive",
            )
        )
    st.walk_end_yaw = st.root_yaw
    st.last_pose156 = target
    return frames


def _low_front_openable(
    scene: SceneState,
    entity,
    *,
    target_z: float | None = None,
) -> bool:
    """Whether a front-opening fixture is low enough that a person should crouch."""
    meta = getattr(entity, "meta", None) or {}
    category = str(meta.get("asset_category") or "").casefold().replace("-", "_")
    if (
        meta.get("access_mode") != "front_opening"
        or meta.get("openable_kind") == "lid"
        or "fridge" in category
        or category == "freezer"
    ):
        return False
    floor_z = float(getattr(scene, "floor_z", 0.0))
    target_is_low = (
        target_z is not None
        and math.isfinite(float(target_z))
        and float(target_z) - floor_z < _LOW_OPENABLE_APERTURE_MAX_M
    )
    top_z = getattr(entity, "top_z", None)
    known_bottom_cabinet = category in {
        "bottom_cabinet",
        "bottom_cabinet_no_top",
        "metal_bottom_cabinet",
    }
    if target_is_low:
        pass
    elif top_z is None:
        if not known_bottom_cabinet:
            return False
    elif float(top_z) - floor_z > _LOW_OPENABLE_TOP_MAX_M:
        return False
    handle_z = getattr(entity, "handle_z", None)
    aperture = meta.get("aperture_center")
    aperture_z = None
    if aperture is not None and len(aperture) >= 3:
        aperture_z = float(aperture[2])
    return bool(
        target_is_low
        or (
            handle_z is not None
            and float(handle_z) - floor_z < _LOW_OPENABLE_HANDLE_MAX_M
        )
        or (
            aperture_z is not None
            and aperture_z - floor_z < _LOW_OPENABLE_APERTURE_MAX_M
        )
    )


def _low_front_insert_camera_pullback(
    scene: SceneState,
    entity,
    *,
    target_z: float | None = None,
) -> float:
    meta = getattr(entity, "meta", None) or {}
    requested = meta.get(
        "content_insert_camera_pullback_m",
        meta.get("content_camera_pullback_m"),
    )
    if requested is None or not _low_front_openable(
        scene,
        entity,
        target_z=target_z,
    ):
        return 0.0
    return min(
        max(0.0, float(requested)),
        _LOW_FRONT_INSERT_CAMERA_PULLBACK_MAX_M,
    )


def _set_crouch_hold(
    st: SchedState,
    owner: str,
    fraction: float,
    pose: np.ndarray | None = None,
) -> None:
    st.crouch_owner = str(owner)
    st.crouch_fraction = max(0.0, float(fraction))
    st.crouch_root_z = (
        _pin_z(pose, st.floor_z)
        if pose is not None
        else getattr(st, "crouch_root_z", None)
    )


def _clear_crouch_hold(st: SchedState) -> None:
    st.crouch_owner = None
    st.crouch_fraction = 0.0
    st.crouch_root_z = None


def _inside_owner(scene: SceneState, target_key: str | None, owner: str) -> bool:
    current = None if target_key is None else str(target_key)
    parents = getattr(scene, "inside", None) or {}
    seen = set()
    while current is not None and current not in seen:
        if current == owner:
            return True
        seen.add(current)
        parent = parents.get(current)
        current = None if parent is None else str(parent)
    return False


def _crouch_owner_for_target(
    scene: SceneState, st: SchedState, target_key: str | None
) -> str | None:
    owner = getattr(st, "crouch_owner", None)
    if owner is None:
        return None
    owner = str(owner)
    if _inside_owner(scene, target_key, owner):
        return owner
    target = None if target_key is None else str(target_key)
    if target is None or (target, owner) not in (
        getattr(scene, "tracked_near", None) or set()
    ):
        return None
    try:
        target_xy = np.asarray(scene.object_origin(target), dtype=np.float64)[:2]
    except (AttributeError, KeyError, TypeError, ValueError):
        entity = (getattr(scene, "entities", None) or {}).get(target)
        target_xy = np.asarray(getattr(entity, "xy", ()), dtype=np.float64)
    if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
        return None
    return (
        owner
        if math.dist(tuple(st.root_xy), tuple(target_xy))
        <= _CROUCH_CONTEXT_NAV_MAX_M
        else None
    )


def _pick_crouch_owner(
    scene: SceneState,
    st: SchedState,
    target_key: str,
    source_container: str | None,
) -> str | None:
    owner = _crouch_owner_for_target(scene, st, target_key)
    if owner is not None or source_container is None:
        return owner
    try:
        entity = scene.resolve(source_container)
    except (KeyError, ValueError):
        return None
    return source_container if _low_front_openable(scene, entity) else None


def _active_crouch_fraction(
    scene: SceneState, st: SchedState, target_key: str | None
) -> float:
    if _crouch_owner_for_target(scene, st, target_key) is None:
        return 0.0
    return max(0.0, float(getattr(st, "crouch_fraction", 0.0)))


def _lean_for_height(z: float) -> tuple[float, float]:
    """Forward-lean (rad) + its fraction of _LEAN_RAD to reach a target at world height ``z``: lean IN
    for a LOW target (~0.75 desk -> ~40deg) but NONE for chest/overhead -- so a high reach stands tall
    instead of bending over (a fixed 40deg lean suits only a low desk). Same curve _slide
    uses, shared so pick / place / open all adapt to height identically."""
    deg = float(np.clip((1.1 - z) * 110.0, 0.0, 55.0))
    rad = math.radians(deg)
    return rad, (rad / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0)


# =====================================================================================================
# POSE-PLANNER (v1: forward-lean ladder). A single HEIGHT-driven lean (_lean_for_height) bends for a
# LOW target but NEVER leans forward to reach a FAR one at normal height, so a far object just fails
# OUT_OF_REACH (the arm alone cannot span the gap; warp_arm_to_seat moves only the arm, not the
# torso/root). This planner instead tries a LADDER of increasing FORWARD lean (the torso pitches forward,
# carrying the shoulder toward the target -> extends horizontal reach) and returns the LEAST lean whose real
# 3D IK residual actually reaches the target. "Reachable" = some rung's residual <= tol (the true IK test),
# rather than a flat REACH_ENVELOPE_M horizontal cap. Rung 0 == the height-lean, so escalation fires ONLY
# when the height-lean would miss. Squat/crouch rungs for low/floor targets (which need foot-pinned
# frames) are added on top of this. Both the init reach-gate and the producers call plan_reach, so the
# gate approves exactly the pose the render adopts.
# =====================================================================================================
_REACH_LEAN_LADDER_DEG: tuple[float, ...] = (30.0, 42.0, 55.0)
# lean+squat rungs (lean_deg, squat_frac) tried AFTER the lean-only rungs, for a LOW target a crouch can
# reach that an upright lean cannot (a floor / low-shelf object). squat_frac scales _apply_squat's flex.
# The last two rungs are a DEEP bow+crouch (near-horizontal spine bow + full knee crouch) so the shoulder
# drops to ~FLOOR height and the hand can reach an object/container ON THE FLOOR (z~0). ADDITIVE + least-
# effort-first: plan_reach returns the FIRST rung that reaches, so any mid/high target that reaches on a
# lean rung never hits these; the deep rungs fire ONLY for the low/floor targets that would otherwise be
# OUT_OF_REACH. TRADE-OFF: the deep bow can clip the legs into nearby furniture; that is not handled here.
_REACH_SQUAT_LADDER: tuple[tuple[float, float], ...] = (
    (40.0, 0.5),
    (48.0, 1.0),
    (58.0, 1.5),
    (68.0, 2.0),
    (78.0, 2.0),
    (86.0, 2.2),
)
_SQUAT_REACH_Z_MAX_M: float = (
    1.0  # crouch only helps a target at/below ~this height above the floor
)
# PUTINSIDE-only crouch ladder (lean_deg, squat_frac): favour LEG crouch over spine BOW so the shoulder
# drops INTO the target shelf gap while the TORSO + HEAD stay far more upright -- a big forward bow (the
# 58-86deg rungs above) drives the head INTO the cabinet. Milder lean here; the lowered shoulder reaches
# HORIZONTALLY into the gap. plan_reach_shelf offers these at ANY height (not z-gated) so a mid/high shelf
# whose gap needs a mild crouch can also drop the shoulder in. Deepest rung shares squat=2.2 with the
# general ladder but at 40deg (not 86deg) lean -> same leg crouch, head far more upright.
_SHELF_SQUAT_LADDER: tuple[tuple[float, float], ...] = (
    (28.0, 1.0),
    (32.0, 1.5),
    (36.0, 2.0),
    (40.0, 2.2),
)

_HEAD_UPRIGHT_CROUCH_LEAN_LADDER_DEG: tuple[float, ...] = (
    0.0,
    12.0,
    20.0,
    28.0,
)
_EXACT_REST_ROOT_DROP_LADDER_M: tuple[float, ...] = (0.0, 0.08, 0.12, 0.16, 0.20)
_EXACT_REST_CROUCH_LEAN_LADDER_DEG: tuple[float, ...] = (0.0, 6.0, 12.0)
_EXACT_REST_MAX_DROP_LEAN_LADDER_DEG: tuple[float, ...] = (18.0, 24.0)
_EXACT_REST_MAX_SQUAT: float = 2.5
_EXACT_REST_REACH_TOL_M: float = 0.10


def _apply_squat(p: np.ndarray, depth: float) -> np.ndarray:
    """Add a crouch (knee/hip/ankle flex) of ``depth`` to pose ``p`` in place (feet stay flat). depth<=0 no-op.
    Mirrors ``_squat_pose`` but ADDITIVE, so it composes on top of a leaned reach base."""
    if depth <= 1e-6:
        return p
    for j in (_HIP_L, _HIP_R):
        p[j * 3] += -0.7 * depth
    for j in (_KNEE_L, _KNEE_R):
        p[j * 3] += 1.2 * depth
    for j in (7, 8):  # ankles: mild counter-rotate so the soles stay flat
        p[j * 3] += -0.35 * depth
    return p


def _shift_squat(pose: np.ndarray, depth_delta: float) -> np.ndarray:
    """Return a copy with a signed amount of the shared squat profile applied."""
    shifted = np.asarray(pose, dtype=np.float64).copy()
    if abs(float(depth_delta)) <= 1e-6:
        return shifted
    for joint in (_HIP_L, _HIP_R):
        shifted[joint * 3] += -0.7 * float(depth_delta)
    for joint in (_KNEE_L, _KNEE_R):
        shifted[joint * 3] += 1.2 * float(depth_delta)
    for joint in (7, 8):
        shifted[joint * 3] += -0.35 * float(depth_delta)
    return shifted


def _reach_base_pose(
    rig, arm: dict, lean_rad: float, squat_frac: float, hand_close: float = 1.0
) -> np.ndarray:
    """The reach IK base: assemble + forward spine lean + (optional) knee/hip/ankle crouch."""
    return _apply_squat(
        wpp._add_lean(
            core.assemble_pose(rig, arm, (0.0, 0.0, 0.0), hand_close), lean_rad
        ),
        squat_frac,
    )


def _reach_rungs(
    z: float, floor_z: float, shelf: bool = False
) -> list[tuple[float, float]]:
    """Reach-pose ladder (lean_rad, squat_frac), least-effort first. put/grab (shelf=False): height-lean +
    progressively larger FORWARD lean (extends horizontal reach), then lean+SQUAT rungs for a LOW target
    (skipped above ~chest height). PUTINSIDE (shelf=True): height-lean + the LEG-favouring
    _SHELF_SQUAT_LADDER (mild lean, deep squat) so the shoulder drops into the shelf gap with the head/torso
    UPRIGHT -- NO big lean-only rungs (they keep the shoulder high) and the crouch is offered at ANY height
    (not z-gated), so the gap-aware picker (plan_reach_shelf) can drop the shoulder in at mid/high shelves too."""
    base = _lean_for_height(z)[0]
    rungs: list[tuple[float, float]] = [(base, 0.0)]
    if shelf:
        rungs += [(math.radians(d), s) for d, s in _SHELF_SQUAT_LADDER]
        return rungs
    rungs += [
        (math.radians(d), 0.0)
        for d in _REACH_LEAN_LADDER_DEG
        if math.radians(d) > base + 1e-6
    ]
    if z <= float(floor_z) + _SQUAT_REACH_Z_MAX_M:
        rungs += [(math.radians(d), s) for d, s in _REACH_SQUAT_LADDER]
        # HEAD-UPRIGHT FALLBACKS, last. The bow rungs above buy horizontal reach by folding the
        # torso forward, which is what drives the head into a shelf; these buy it from the LEGS and
        # keep the head up (they are PutInside's ladder, which exists for exactly this reason).
        # Appended rather than reordered, so a reach that passes on an earlier rung still picks
        # that rung -- only one that fails the clearance test falls through to these.
        rungs += [(math.radians(d), s) for d, s in _SHELF_SQUAT_LADDER]
    return rungs


def _head_upright_crouch_rungs(
    rungs: list[tuple[float, float]],
    minimum_squat: float,
) -> list[tuple[float, float]]:
    """Try leg-driven crouches before allowing a low reach to bow the head forward."""
    squat = max(0.0, float(minimum_squat))
    if squat <= 1e-6:
        return list(rungs)
    ordered = [
        (math.radians(degrees), squat)
        for degrees in _HEAD_UPRIGHT_CROUCH_LEAN_LADDER_DEG
    ]
    ordered.extend(rungs)
    unique = []
    seen: set[tuple[int, int]] = set()
    for lean_rad, depth in ordered:
        key = (int(round(lean_rad * 1e6)), int(round(depth * 1e6)))
        if key not in seen:
            seen.add(key)
            unique.append((lean_rad, depth))
    return unique


def _squat_for_pinned_root_drop(
    rig,
    arm: dict,
    lean_rad: float,
    minimum_squat: float,
    floor_z: float,
    root_drop_m: float,
) -> tuple[float, float] | None:
    """Return the least leg flex that lowers the pinned root by ``root_drop_m``."""
    minimum = max(0.0, float(minimum_squat))
    baseline = _reach_base_pose(rig, arm, lean_rad, minimum, 1.0)
    baseline_z = _pin_z(baseline, floor_z)
    desired = max(0.0, float(root_drop_m))
    if desired <= 1e-6:
        return minimum, 0.0
    upper = max(minimum, _EXACT_REST_MAX_SQUAT)
    upper_pose = _reach_base_pose(rig, arm, lean_rad, upper, 1.0)
    upper_drop = baseline_z - _pin_z(upper_pose, floor_z)
    if upper_drop + 1e-4 < desired:
        return None
    lower = minimum
    for _ in range(16):
        midpoint = 0.5 * (lower + upper)
        pose = _reach_base_pose(rig, arm, lean_rad, midpoint, 1.0)
        if baseline_z - _pin_z(pose, floor_z) >= desired:
            upper = midpoint
        else:
            lower = midpoint
    pose = _reach_base_pose(rig, arm, lean_rad, upper, 1.0)
    actual_drop = max(0.0, baseline_z - _pin_z(pose, floor_z))
    return upper, actual_drop


def _exact_rest_crouch_rungs(
    rig,
    arm: dict,
    minimum_squat: float,
    floor_z: float,
) -> list[tuple[float, float, float]]:
    """Natural low-cabinet poses ordered by the smallest foot-pinned root drop."""
    rungs = []
    for root_drop in _EXACT_REST_ROOT_DROP_LADDER_M:
        lean_degrees = _EXACT_REST_CROUCH_LEAN_LADDER_DEG
        if root_drop >= _EXACT_REST_ROOT_DROP_LADDER_M[-1] - 1e-6:
            lean_degrees += _EXACT_REST_MAX_DROP_LEAN_LADDER_DEG
        for degrees in lean_degrees:
            lean_rad = math.radians(degrees)
            solved = _squat_for_pinned_root_drop(
                rig,
                arm,
                lean_rad,
                minimum_squat,
                floor_z,
                root_drop,
            )
            if solved is not None:
                squat, actual_drop = solved
                rungs.append((lean_rad, squat, actual_drop))
    return rungs


def _placement_reach_seed_arm(rig, grasp_arm: dict, exact_rest_crouch: bool):
    return (
        getattr(rig, "arm_grasp", grasp_arm)
        if exact_rest_crouch
        else grasp_arm
    )


def body_clear_fn(
    scene,
    stand_xy,
    yaw: float,
    floor_z: float,
    *,
    require_body_clear: bool = False,
):
    """A memoised body-clearance test for ``plan_reach``.

    Returns None when the scene has no collision oracle, which keeps every offline caller (the
    gates, the probes) on the unscreened behaviour. Under the FPV benchmark contract, the exact eye-camera
    sphere is the hard gate and avatar-body overlap is telemetry. If the camera probe is unavailable, the
    scale-normalized body grade remains the conservative fallback. Reaching-arm geometry is reported and
    ranked separately by the stand search, but cannot veto a camera-clear egocentric pose.
    """
    cw = getattr(scene, "cworld", None)
    if cw is None:
        return None
    import action_check

    camera_guard = action_check.uses_egocentric_camera_collision_policy(scene) or bool(
        getattr(scene, "egocentric_collision_fallback_used", False)
        and not getattr(scene, "strict_acceptance", False)
    )
    sphere_overlap = getattr(cw, "sphere_overlap", None) if camera_guard else None
    cache: dict[tuple[int, bytes], bool] = {}

    def clear(base_pose: np.ndarray, root_z: float) -> bool:
        key = (
            int(round(float(root_z) * 1e4)),
            np.round(np.asarray(base_pose, dtype=np.float64), 6)
            .astype(np.float32)
            .tobytes(),
        )
        hit = cache.get(key)
        if hit is None:
            J = gait.pose_place_joints(
                base_pose, tuple(stand_xy), float(yaw), float(root_z)
            )
            try:
                joints = np.asarray(J, dtype=np.float64)
                exceeds_depth = getattr(cw, "human_exceeds_depth", None)
                if exceeds_depth is not None:
                    clean_bound = np.nextafter(
                        action_check.minimum_pose_dirty_depth_m(), 0.0
                    )
                    hit = not bool(exceeds_depth(joints, clean_bound))
                    if not hit and not require_body_clear:
                        hit = bool(action_check.grade_human_pose(cw, joints)["passes"])
                else:
                    hit = bool(action_check.grade_human_pose(cw, joints)["passes"])
                if sphere_overlap is not None:
                    eye, _aim = smplh_camera.fpv_eye_and_aim(
                        joints,
                        "look",
                        np.zeros(3, dtype=np.float64),
                        root_yaw_deg=float(yaw),
                    )
                    camera_clear = sphere_overlap(eye, _FPV_EYE_CLEAR_RADIUS_M) is None
                    hit = camera_clear and (hit if require_body_clear else True)
            except Exception:  # noqa: BLE001 -- never let a screen take a reach down
                hit = True
            cache[key] = hit
        return hit

    return clear


def first_fpv_eye_collision(
    scene,
    frames,
    *,
    camera_frame_offset: int = 0,
    exclude_owners: tuple[str, ...] = (),
    additional_boxes=(),
    dynamic_boxes_fn=None,
    clearance_radius_m: float = _FPV_EYE_CLEAR_RADIUS_M,
    camera_pullback_m: float = 0.0,
) -> tuple[int, str] | None:
    """Return the first exact FPV-eye collision in an already generated action."""
    cw = getattr(scene, "cworld", None)
    sphere_overlap = getattr(cw, "sphere_overlap", None)
    if sphere_overlap is None:
        return None
    for index, frame in enumerate(frames):
        joints = gait.pose_place_joints(
            frame.pose156,
            frame.xy,
            float(frame.yaw),
            float(frame.z),
            arm_override=frame.arm_override,
            rhp=frame.rhp,
        )
        eye_offset = getattr(frame, "camera_eye_offset", None)
        if eye_offset is None:
            eye, _aim = smplh_camera.fpv_eye_and_aim(
                joints,
                getattr(frame, "fpv_mode", "look"),
                np.zeros(3, dtype=np.float64),
                frame=int(camera_frame_offset) + index,
                root_yaw_deg=float(frame.yaw),
            )
        else:
            eye = smplh_camera.default_policy_fpv_eye(
                joints,
                getattr(frame, "fpv_mode", "look"),
                root_yaw_deg=float(frame.yaw),
                eye_offset_world=eye_offset,
            )
        if camera_pullback_m > 0.0 and str(getattr(frame, "seg", "")).startswith(
            "front_insert"
        ):
            eye = np.asarray(eye, dtype=np.float64) - np.asarray(
                wpp._body_forward(float(frame.yaw)), dtype=np.float64
            ) * float(camera_pullback_m)
        clearance_radius = max(
            _FPV_EYE_CLEAR_RADIUS_M,
            float(clearance_radius_m),
        )
        owner = (
            sphere_overlap(
                eye,
                clearance_radius,
                exclude_owners=exclude_owners,
            )
            if exclude_owners
            else sphere_overlap(eye, clearance_radius)
        )
        if owner is None:
            frame_boxes = additional_boxes
            if dynamic_boxes_fn is not None:
                frame_boxes = (*additional_boxes, *dynamic_boxes_fn(index, frame))
            for box_owner, mn, mx in frame_boxes:
                lower = np.asarray(mn, dtype=np.float64)
                upper = np.asarray(mx, dtype=np.float64)
                closest = np.maximum(lower, np.minimum(eye, upper))
                if float(np.dot(eye - closest, eye - closest)) <= clearance_radius**2:
                    owner = str(box_owner)
                    break
        if owner is not None:
            return index, str(owner)
    return None


def _navigation_state_snapshot(st: SchedState) -> dict[str, object]:
    last_pose = getattr(st, "last_pose156", None)
    return {
        "root_xy": tuple(st.root_xy),
        "root_yaw": float(st.root_yaw),
        "walk_end_yaw": float(st.walk_end_yaw),
        "gait_cum_dist": float(st.gait_cum_dist),
        "last_pose156": (
            None
            if last_pose is None
            else np.asarray(last_pose, dtype=np.float64).copy()
        ),
        "crouch_owner": getattr(st, "crouch_owner", None),
        "crouch_fraction": float(getattr(st, "crouch_fraction", 0.0)),
        "crouch_root_z": getattr(st, "crouch_root_z", None),
    }


def _restore_navigation_state(st: SchedState, snapshot: dict[str, object]) -> None:
    for name, value in snapshot.items():
        setattr(st, name, value)


def _navigation_fpv_collision(scene, frames, frame_offset: int):
    import init_validation

    return first_fpv_eye_collision(
        scene,
        frames,
        camera_frame_offset=frame_offset,
        exclude_owners=init_validation._kinematically_open_leaf_roots(scene),
        additional_boxes=init_validation._kinematically_open_leaf_boxes(scene),
    )


def _navigation_yaw_candidate(frames, center: int, radius: int, delta: float):
    candidate = list(frames)
    for index in range(max(0, center - radius), min(len(frames), center + radius + 1)):
        distance = abs(index - center)
        fraction = (radius + 1 - distance) / float(radius + 1)
        weight = fraction * fraction * (3.0 - 2.0 * fraction)
        candidate[index] = replace(
            frames[index],
            yaw=float(frames[index].yaw) + float(delta) * weight,
        )
    return candidate


def _camera_safe_navigation_yaw_repair(scene, frames, frame_offset: int):
    """Steer through a narrow portal when blocking its only grid cell would disconnect it."""
    repaired = list(frames)
    repairs = []
    for _attempt in range(6):
        hit = _navigation_fpv_collision(scene, repaired, frame_offset)
        if hit is None:
            return repaired, repairs
        center = int(hit[0])
        accepted = None
        for radius in (12, 18, 8):
            start = max(0, center - radius)
            stop = min(len(repaired), center + radius + 1)
            for magnitude in (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0):
                for delta in (magnitude, -magnitude):
                    candidate = _navigation_yaw_candidate(
                        repaired,
                        center,
                        radius,
                        delta,
                    )
                    candidate_hit = _navigation_fpv_collision(
                        scene,
                        candidate[start:stop],
                        frame_offset + start,
                    )
                    if candidate_hit is None or start + int(candidate_hit[0]) > center:
                        accepted = candidate
                        repairs.append(
                            {
                                "frame": center,
                                "radius": radius,
                                "delta_deg": delta,
                                "owner": hit[1],
                            }
                        )
                        break
                if accepted is not None:
                    break
            if accepted is not None:
                break
        if accepted is None:
            return None
        repaired = accepted
    return (
        (repaired, repairs)
        if _navigation_fpv_collision(scene, repaired, frame_offset) is None
        else None
    )


def _navigation_departure_candidate(
    frames,
    previous_pose,
    start_yaw: float,
    floor_z: float,
    hold_frames: int,
):
    """Keep the just-finished action pose while stepping clear, then turn into the walk."""
    import heading_controller as hc

    held_pose = np.asarray(previous_pose, dtype=np.float64)
    departure = [
        replace(
            frame,
            pose156=held_pose.copy(),
            yaw=float(start_yaw),
            z=_pin_z(held_pose, floor_z),
            fpv_mode="look",
            seg="depart",
        )
        for frame in frames[:hold_frames]
    ]
    next_frame = frames[hold_frames]
    yaw_delta = _ang_delta(float(start_yaw), float(next_frame.yaw))
    turn_frames = max(4, int(math.ceil(abs(yaw_delta) / hc.HeadingCfg().max_turn_deg)))
    turn_xy = tuple(departure[-1].xy)
    for index in range(turn_frames):
        fraction = (index + 1) / turn_frames
        pose = gait.blend156(
            held_pose,
            next_frame.pose156,
            wpp._smoothstep(fraction),
        )
        departure.append(
            replace(
                next_frame,
                pose156=pose,
                xy=turn_xy,
                yaw=float(start_yaw) + yaw_delta * fraction,
                z=_pin_z(pose, floor_z),
                fpv_mode="look" if fraction < 1.0 else next_frame.fpv_mode,
                seg="depart",
            )
        )
    return [*departure, *frames[hold_frames:]]


def _camera_safe_navigation_departure_repair(
    scene,
    st: SchedState,
    snapshot: dict[str, object],
    frames,
    frame_offset: int,
    hit,
):
    """Try a real back-away-then-turn departure for an early camera collision."""
    previous_pose = snapshot.get("last_pose156")
    if previous_pose is None or st.held is not None or not frames or int(hit[0]) > 12:
        return None
    first_hold = max(2, int(hit[0]) + 2)
    last_hold = min(len(frames) - 1, 36)
    for hold_frames in range(first_hold, last_hold + 1, 2):
        candidate = _navigation_departure_candidate(
            frames,
            previous_pose,
            float(snapshot["root_yaw"]),
            float(st.floor_z),
            hold_frames,
        )
        if _navigation_fpv_collision(scene, candidate, frame_offset) is None:
            return candidate, {
                "hold_frames": hold_frames,
                "retreat_m": math.dist(snapshot["root_xy"], frames[hold_frames - 1].xy),
                "owner": hit[1],
            }
    return None


def _navigation_camera_focus_frames(frames, goal: Goal):
    """Apply an authored gaze only to a deliberate door-approach segment."""
    meta = goal.meta or {}
    fixed = meta.get("_navigation_fixed_gaze")
    arrival = meta.get("_navigation_arrival_gaze")
    if fixed is None and arrival is None:
        return frames
    focus = np.asarray(fixed if fixed is not None else arrival, dtype=np.float64)
    if focus.shape != (3,) or not np.all(np.isfinite(focus)):
        return frames
    arrival_start = max(0, len(frames) - _N_ARRIVE)
    return [
        (
            replace(frame, fpv_mode="look", gaze=focus.copy())
            if fixed is not None or index >= arrival_start
            else frame
        )
        for index, frame in enumerate(frames)
    ]


def _continuous_front_close_navigation_frames(
    rig,
    scene,
    st: SchedState,
    goal: Goal,
    grid,
    *,
    path_override=None,
):
    """Keep a short content-to-Close reposition facing the open container."""
    owner = (goal.meta or {}).get("_continuous_front_opening_close")
    if owner is None or goal.target_xy is None or goal.target_yaw is None:
        return None
    start = tuple(float(value) for value in st.root_xy)
    target = tuple(float(value) for value in goal.target_xy)
    distance = math.dist(start, target)
    yaw_delta = abs(_ang_delta(float(st.root_yaw), float(goal.target_yaw)))
    if (
        distance > _CONTINUOUS_FRONT_CLOSE_MAX_SHIFT_M
        or yaw_delta > _CONTINUOUS_FRONT_CLOSE_MAX_YAW_DEG
    ):
        return None

    import nav_planner

    route = path_override
    if route is None and grid is not None:
        try:
            route = nav_planner.plan_path(grid, start, target)
        except PrimitiveFailure:
            return None
    if route:
        route_length = sum(
            math.dist(tuple(a[:2]), tuple(b[:2]))
            for a, b in zip(route, route[1:])
        )
        if route_length > distance + _CONTINUOUS_FRONT_CLOSE_ROUTE_SLACK_M:
            return None

    frames = nav_planner._final_approach_frames(rig, scene, st, goal, start)
    logger.info(
        "[container-close] '%s': kept aperture-facing continuity over %.2fm / %.1fdeg",
        owner,
        distance,
        yaw_delta,
    )
    return frames


def _camera_safe_navigation_frames(
    rig,
    g,
    scene,
    st: SchedState,
    goal: Goal,
    grid,
    *,
    path_override=None,
):
    """Generate the real walk, then reroute any exact FPV-eye collision."""
    import action_check
    import nav_grid
    import nav_planner

    snapshot = _navigation_state_snapshot(st)
    frames = _continuous_front_close_navigation_frames(
        rig,
        scene,
        st,
        goal,
        grid,
        path_override=path_override,
    )
    if frames is None:
        frames = nav_planner.walk_planned(
            rig,
            g,
            scene,
            st,
            goal,
            grid,
            path_override=path_override,
        )
    frames = _navigation_camera_focus_frames(frames, goal)
    candidate_runs = [
        (frames, path_override, _navigation_state_snapshot(st)),
    ]
    collision_world = getattr(scene, "cworld", None)
    if not action_check.uses_egocentric_camera_collision_policy(scene) or not callable(
        getattr(collision_world, "sphere_overlap", None)
    ):
        return frames
    frame_offset = int((goal.meta or {}).get("_camera_frame_offset", 0))
    hit = _navigation_fpv_collision(scene, frames, frame_offset)
    if hit is None:
        return frames
    repair_grid = nav_grid.egocentric_passable_view(grid, scene) if grid else None
    start_xy = snapshot["root_xy"]
    protected = set()
    if repair_grid is not None and goal.target_xy is not None:
        protected = {
            repair_grid.world_to_cell(*start_xy),
            repair_grid.world_to_cell(*goal.target_xy),
        }
    for attempt in range(4):
        if repair_grid is None or goal.target_xy is None:
            break
        i, j = repair_grid.world_to_cell(*frames[hit[0]].xy)
        if (
            not repair_grid.in_bounds(i, j)
            or (i, j) in protected
            or bool(repair_grid.occ[j, i])
        ):
            break
        repair_grid.occ[j, i] = True
        repair_grid.sdf = nav_grid._two_sided_sdf(repair_grid.occ, repair_grid.cell)
        _restore_navigation_state(st, snapshot)
        try:
            repaired_path = nav_planner.plan_path(
                repair_grid,
                start_xy,
                tuple(goal.target_xy),
            )
        except PrimitiveFailure:
            break
        frames = nav_planner.walk_planned(
            rig,
            g,
            scene,
            st,
            goal,
            repair_grid,
            path_override=repaired_path,
        )
        frames = _navigation_camera_focus_frames(frames, goal)
        candidate_runs.append((frames, repaired_path, _navigation_state_snapshot(st)))
        hit = _navigation_fpv_collision(scene, frames, frame_offset)
        if hit is None:
            logger.warning(
                "[camera-nav] exact generated walk required %d route repair(s)",
                attempt + 1,
            )
            goal.meta["_camera_safe_navigation_path"] = repaired_path
            return frames
    for candidate_frames, candidate_path, completed_state in reversed(candidate_runs):
        candidate_hit = _navigation_fpv_collision(
            scene,
            candidate_frames,
            frame_offset,
        )
        if candidate_hit is not None:
            departure_repair = _camera_safe_navigation_departure_repair(
                scene,
                st,
                snapshot,
                candidate_frames,
                frame_offset,
                candidate_hit,
            )
            if departure_repair is not None:
                frames, repair = departure_repair
                _restore_navigation_state(st, completed_state)
                if candidate_path is not None:
                    goal.meta["_camera_safe_navigation_path"] = candidate_path
                goal.meta["_camera_safe_navigation_departure"] = repair
                logger.warning(
                    "[camera-nav] furniture-adjacent start required a %.2f m "
                    "back-away before turning",
                    repair["retreat_m"],
                )
                return frames
        yaw_repair = _camera_safe_navigation_yaw_repair(
            scene,
            candidate_frames,
            frame_offset,
        )
        if yaw_repair is None:
            continue
        frames, repairs = yaw_repair
        _restore_navigation_state(st, completed_state)
        if frames:
            st.root_yaw = float(frames[-1].yaw)
            st.walk_end_yaw = float(frames[-1].yaw)
        if candidate_path is not None:
            goal.meta["_camera_safe_navigation_path"] = candidate_path
        goal.meta["_camera_safe_navigation_yaw_repairs"] = repairs
        logger.warning(
            "[camera-nav] narrow portal required %d bounded yaw repair(s)",
            len(repairs),
        )
        return frames
    _restore_navigation_state(st, snapshot)
    raise PrimitiveFailure(
        PrimitiveError.PATH_BLOCKED,
        f"generated walk intersects {hit[1]} with the FPV eye at action frame {hit[0]}",
    )


def _require_fpv_eye_clear(scene, frames, goal: Goal, action: str) -> None:
    """Reject an authored action whose exact head-mounted camera enters scene geometry."""
    import action_check

    if not action_check.uses_egocentric_camera_collision_policy(scene):
        return
    collision_kwargs = {}
    try:
        entity = scene.resolve(goal.obj_key)
    except (AttributeError, KeyError, ValueError):
        entity = None
    if (
        goal.prim
        in (
            Primitive.OPEN,
            Primitive.CLOSE,
            Primitive.OPENDOOR,
            Primitive.CLOSEDOOR,
        )
        and entity is not None
        and getattr(entity, "hinge_xy", None) is not None
    ):
        leaf_paths = tuple(
            str(path).rstrip("/")
            for path in (getattr(entity, "leaf_paths", None) or ())
            if path
        )
        if leaf_paths:
            fraction = (
                0.0
                if goal.prim in (Primitive.OPEN, Primitive.OPENDOOR)
                else 1.0
            )
            leaf_box_sources = _openable_leaf_box_sources(scene, entity)

            def _dynamic_leaf_boxes(_index, frame):
                nonlocal fraction
                anchor = getattr(frame, "contact_anchor", None)
                if anchor is not None:
                    fraction = _openable_swing_fraction(entity, anchor)
                return _open_leaf_boxes(
                    scene,
                    entity,
                    swing_fraction=fraction,
                    leaf_box_sources=leaf_box_sources,
                )

            collision_kwargs = {
                "exclude_owners": leaf_paths,
                "dynamic_boxes_fn": _dynamic_leaf_boxes,
            }
    camera_clearance = (goal.meta or {}).get("_camera_clearance_radius_m")
    if camera_clearance is None and entity is not None:
        camera_clearance = (getattr(entity, "meta", None) or {}).get(
            "action_camera_clearance_m"
        )
    if (
        camera_clearance is None
        and entity is not None
        and (getattr(entity, "meta", None) or {}).get("access_mode")
        == "front_opening"
        and not _low_front_openable(scene, entity)
    ):
        camera_clearance = _FRONT_OPENING_ACTION_CAMERA_CLEARANCE_M
    if camera_clearance is not None:
        collision_kwargs["clearance_radius_m"] = float(camera_clearance)
    hit = first_fpv_eye_collision(
        scene,
        frames,
        camera_frame_offset=int((goal.meta or {}).get("_camera_frame_offset", 0)),
        **collision_kwargs,
    )
    if hit is None:
        return
    raise PrimitiveFailure(
        PrimitiveError.TRAJECTORY_COLLISION,
        f"{action} {goal.obj_key!r} intersects {hit[1]} with the FPV eye "
        f"at action frame {hit[0]}",
    )


# Generous slack (m) added to the arm's rigid max extension for the O(1) reach pre-filter below: it covers
# the grip-seat offset beyond the knuckles plus a safety margin, so a target a warp could actually reach is
# NEVER screened out (the pre-filter only ever skips clearly-out-of-envelope rungs).
_REACH_ENVELOPE_SLACK_M: float = 0.20


def _rung_can_reach(rig, base_pose, rz, world_off, tgt, tol: float) -> bool:
    """SOUND O(1) reachability screen for ONE reach rung: could a CCD warp of this rung's arm possibly seat
    the grip at ``tgt``? The shoulder is FIXED by the base torso pose (``warp_arm_to_seat`` only rotates the
    arm joints about it, never moves it), and the arm's max straight extension shoulder->elbow->wrist->
    knuckles is rigid + pose-invariant, so a ``tgt`` farther than that (+ ``_REACH_ENVELOPE_SLACK_M``) is
    unreachable by ANY arm config of this rung. Returns False ONLY for such clearly-out-of-envelope targets
    -> the caller skips the expensive CCD; never False for a target a warp could reach (the slack exceeds the
    seat-beyond-knuckles offset). Reuses the same ``_fwd_arm_points`` sample the clearance checks use
    ([elbow, wrist, shoulder, mcp0..4, forearm_mid] in world)."""
    P = _fwd_arm_points(base_pose, rz, world_off)
    elbow_w, wrist_w, shoulder_w = P[0], P[1], P[2]
    mcp_w = P[3:8].mean(axis=0)
    reach = (
        float(np.linalg.norm(shoulder_w - elbow_w))
        + float(np.linalg.norm(elbow_w - wrist_w))
        + float(np.linalg.norm(wrist_w - mcp_w))
    )
    dist = float(np.linalg.norm(np.asarray(tgt, dtype=np.float64) - shoulder_w))
    return dist <= reach + tol + _REACH_ENVELOPE_SLACK_M


def _reach_rung_templates(
    rig,
    arm_from,
    target_z,
    stand_xy,
    yaw,
    stand_z,
    floor_z,
):
    templates = []
    for lean_rad, squat in _reach_rungs(float(target_z), floor_z):
        base = _reach_base_pose(rig, arm_from, lean_rad, squat, 1.0)
        rz_stand = _pin_z(base, floor_z) if squat > 1e-6 else float(stand_z)
        rz, world_off = wpp._grasp_frame(stand_xy, yaw, rz_stand)
        points = _fwd_arm_points(base, rz, world_off)
        elbow_w, wrist_w, shoulder_w = points[0], points[1], points[2]
        mcp_w = points[3:8].mean(axis=0)
        templates.append(
            {
                "lean_rad": float(lean_rad),
                "squat": float(squat),
                "base": base,
                "stand_z": float(rz_stand),
                "rz": rz,
                "world_off": world_off,
                "shoulder": shoulder_w,
                "max_reach": (
                    float(np.linalg.norm(shoulder_w - elbow_w))
                    + float(np.linalg.norm(elbow_w - wrist_w))
                    + float(np.linalg.norm(wrist_w - mcp_w))
                    + _REACH_ENVELOPE_SLACK_M
                ),
                "body_clear": None,
            }
        )
    return templates


def _candidates_for_target(
    templates, target, tol, allow_colliding_fallback, clear_fn=None
):
    target = np.asarray(target, dtype=np.float64)
    clear = []
    soft = []
    for template in templates:
        if (
            float(np.linalg.norm(target - template["shoulder"]))
            > template["max_reach"] + tol
        ):
            continue
        if template["body_clear"] is None:
            template["body_clear"] = clear_fn is None or bool(
                clear_fn(template["base"], template["stand_z"])
            )
        (clear if template["body_clear"] else soft).append(template)
    return clear + (soft if allow_colliding_fallback else []), bool(soft)


def _solve_reach_candidates(
    rig,
    candidates,
    target,
    ik_budget: IKSearchBudget | None,
    stop_tol: float | None = None,
):
    """Solve candidate dictionaries, using one GPU batch when enabled and scalar CCD as fallback."""
    if not candidates:
        return []
    min_batch = int(os.getenv("ISAAC_BATCH_IK_MIN_CANDIDATES", "2"))
    batch_precharged = False
    batch_enabled = _batch_reach_supported(rig)
    if batch_enabled and len(candidates) >= max(2, min_batch):
        if ik_budget is not None:
            allowed = ik_budget.consume_many(len(candidates))
            candidates = candidates[:allowed]
            if not candidates:
                return []
            batch_precharged = True
        try:
            poses, errors, joints = core.warp_arms_to_seats(
                rig,
                np.stack([candidate["base"] for candidate in candidates]),
                np.stack([candidate["rz"] for candidate in candidates]),
                np.stack([candidate["world_off"] for candidate in candidates]),
                np.broadcast_to(
                    np.asarray(target, dtype=np.float64), (len(candidates), 3)
                ),
            )
            return [
                {
                    "pose": poses[index],
                    "err": float(errors[index]),
                    "joints": joints[index],
                    "backend": "batch",
                }
                for index in range(len(candidates))
            ]
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"[warp-batch] failed ({exc}); falling back to scalar CCD")

    solved = []
    for candidate in candidates:
        if ik_budget is not None and not batch_precharged and not ik_budget.consume():
            break
        result = core.warp_arm_to_seat(
            rig,
            candidate["base"],
            candidate["rz"],
            candidate["world_off"],
            np.asarray(target, dtype=np.float64),
            return_joints=True,
        )
        pose, err = result[:2]
        joints = result[2] if len(result) > 2 else None
        solved.append(
            {
                "pose": pose,
                "err": float(err),
                "joints": (
                    joints
                    if joints is not None
                    else (None if pose is None else core.H.fwd_joints(pose))
                ),
                "backend": "scalar",
            }
        )
        if stop_tol is not None and err <= stop_tol:
            break
    return solved


def _batch_solution_needs_verification(solution, tol: float) -> bool:
    return (
        solution["backend"] == "batch"
        and solution["err"] <= tol + _BATCH_IK_VERIFY_MARGIN_M
    )


def _batch_reach_supported(rig) -> bool:
    return core.batch_reach_enabled() and all(
        hasattr(rig, name) for name in ("rhp", "obj_T_in_wrist", "_reach_pose_cache")
    )


def _refine_reach_solution(rig, candidate, target, solution):
    """Re-run one approximate batch result through the legacy CCD and replace its cache entry."""
    result = core.warp_arm_to_seat(
        rig,
        candidate["base"],
        candidate["rz"],
        candidate["world_off"],
        np.asarray(target, dtype=np.float64),
        use_cache=False,
        return_joints=True,
    )
    pose, err = result[:2]
    joints = result[2] if len(result) > 2 else None
    return {
        "pose": pose,
        "err": float(err),
        "joints": (
            joints
            if joints is not None
            else (None if pose is None else core.H.fwd_joints(pose))
        ),
        "backend": "scalar",
    }


def plan_reach(
    rig,
    arm_from: dict,
    contact,
    stand_xy,
    yaw,
    stand_z,
    tol: float,
    floor_z: float = 0.0,
    enabled: bool = True,
    clear_fn=None,
    allow_colliding_fallback: bool = True,
    ik_budget: IKSearchBudget | None = None,
    upright_first: bool = False,
    upright_crouch_first: bool = False,
    minimum_squat: float = 0.0,
    maximum_squat: float | None = None,
    maximum_lean_rad: float | None = None,
) -> dict:
    """Choose the least-effort POSE (forward-lean, else lean+squat) whose arm warp reaches ``contact`` within
    ``tol`` (real 3D IK residual) -- the replacement for the flat horizontal reach cap.

    Returns ``{lean_rad, lf, squat, stand_z, err, reachable}``. ``stand_z`` is the crouch-pinned root height
    for the chosen pose (== the input ``stand_z`` when not squatting), so the producer/gate render the
    crouched shoulder at its true world height. ``enabled=False`` -> the height-only lean, no extra warps.
    When enabled, warps each rung once and returns the first that reaches (or the
    least-residual rung + reachable=False). ``upright_first`` prepends a zero-lean pose for actions such
    as opening an appliance whose handle can be reached without moving the FPV eye toward its shell.
    ``minimum_squat`` selects the upright-torso crouch ladder for a low front-access appliance."""
    z = float(contact[2])
    base_lean = _lean_for_height(z)[0]
    base_lf = base_lean / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0
    if not enabled:
        squat = max(0.0, float(minimum_squat))
        base = _reach_base_pose(rig, arm_from, base_lean, squat, 1.0)
        return {
            "lean_rad": base_lean,
            "lf": base_lf,
            "squat": squat,
            "stand_z": (_pin_z(base, floor_z) if squat > 1e-6 else float(stand_z)),
            "err": None,
            "reachable": True,
        }
    tgt = np.asarray(contact, dtype=np.float64)
    best: dict | None = None
    clear_rungs = []
    soft_rungs = []
    if minimum_squat > 1e-6:
        rungs = [
            (math.radians(degrees), squat)
            for degrees, squat in _SHELF_SQUAT_LADDER
            if squat + 1e-6 >= float(minimum_squat)
        ]
        if not rungs:
            rungs = [(0.0, float(minimum_squat))]
    else:
        rungs = _reach_rungs(z, floor_z)
    if upright_crouch_first:
        rungs = _head_upright_crouch_rungs(rungs, minimum_squat)
    elif upright_first and rungs[0][0] > 1e-6:
        rungs.insert(0, (0.0, max(0.0, float(minimum_squat))))
    if maximum_squat is not None:
        rungs = [rung for rung in rungs if rung[1] <= float(maximum_squat) + 1e-6]
    if maximum_lean_rad is not None:
        rungs = [
            rung for rung in rungs if rung[0] <= float(maximum_lean_rad) + 1e-6
        ]
    for lean_rad, squat in rungs:
        base = _reach_base_pose(rig, arm_from, lean_rad, squat, 1.0)
        rz_stand = _pin_z(base, floor_z) if squat > 1e-6 else float(stand_z)
        rz, world_off = wpp._grasp_frame(stand_xy, yaw, rz_stand)
        if not _rung_can_reach(rig, base, rz, world_off, tgt, tol):
            continue
        candidate = {
            "lean_rad": float(lean_rad),
            "squat": float(squat),
            "base": base,
            "stand_z": float(rz_stand),
            "rz": rz,
            "world_off": world_off,
        }
        if clear_fn is not None and not clear_fn(base, rz_stand):
            soft_rungs.append(candidate)
        else:
            clear_rungs.append(candidate)
    candidates = clear_rungs + (soft_rungs if allow_colliding_fallback else [])
    has_soft_rungs = bool(soft_rungs)
    solved = _solve_reach_candidates(rig, candidates, tgt, ik_budget, stop_tol=tol)
    verified_count = 0
    for index, (candidate, solution) in enumerate(zip(candidates, solved)):
        if _batch_solution_needs_verification(solution, tol):
            if verified_count >= _BATCH_IK_VERIFY_TOP_K:
                continue
            solution = _refine_reach_solution(rig, candidate, tgt, solution)
            solved[index] = solution
            verified_count += 1
        lean_rad = candidate["lean_rad"]
        squat = candidate["squat"]
        err = solution["err"]
        lf = lean_rad / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0
        if best is None or err < best["err"]:
            best = {
                "lean_rad": float(lean_rad),
                "lf": float(lf),
                "squat": float(squat),
                "stand_z": candidate["stand_z"],
                "err": float(err),
                "arm_pose": solution["pose"],
                "arm_joints": solution["joints"],
                "collision_blocked": False,
            }
        if err <= tol:
            best["reachable"] = True
            return best
    if best is None:
        # EVERY rung was out-of-envelope -> unreachable without a single CCD. Return the height-lean default
        # (the same shape as the disabled path) flagged not-reachable, so callers behave exactly as when
        # every rung is solved and misses (which also returns reachable=False) -- just far faster.
        if minimum_squat > 1e-6 and rungs:
            base_lean, fallback_squat = rungs[0]
            fallback_pose = _reach_base_pose(
                rig,
                arm_from,
                base_lean,
                fallback_squat,
                1.0,
            )
            fallback_stand_z = _pin_z(fallback_pose, floor_z)
        else:
            base_lean = _lean_for_height(z)[0]
            fallback_squat = 0.0
            fallback_stand_z = float(stand_z)
        return {
            "lean_rad": float(base_lean),
            "lf": float(base_lean / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0),
            "squat": float(fallback_squat),
            "stand_z": float(fallback_stand_z),
            "err": float("inf"),
            "reachable": False,
            "collision_blocked": bool(has_soft_rungs and not allow_colliding_fallback),
        }
    best["reachable"] = False
    best["collision_blocked"] = bool(has_soft_rungs and not allow_colliding_fallback)
    return best


def plan_reach_many(
    rig,
    arm_from: dict,
    contacts,
    stand_xy,
    yaw,
    stand_z,
    tol: float,
    floor_z: float = 0.0,
    enabled: bool = True,
    clear_fn=None,
    allow_colliding_fallback: bool = True,
    ik_budget: IKSearchBudget | None = None,
) -> list[dict]:
    """Batch the shared reach ladder across many targets at one fixed stand pose."""
    targets = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    if not len(targets):
        return []
    if not enabled or not _batch_reach_supported(rig) or len(targets) == 1:
        return [
            plan_reach(
                rig,
                arm_from,
                target,
                stand_xy,
                yaw,
                stand_z,
                tol,
                floor_z=floor_z,
                enabled=enabled,
                clear_fn=clear_fn,
                allow_colliding_fallback=allow_colliding_fallback,
                ik_budget=ik_budget,
            )
            for target in targets
        ]
    if float(np.ptp(targets[:, 2])) > 1e-6:
        return [
            plan_reach(
                rig,
                arm_from,
                target,
                stand_xy,
                yaw,
                stand_z,
                tol,
                floor_z=floor_z,
                enabled=True,
                clear_fn=clear_fn,
                allow_colliding_fallback=allow_colliding_fallback,
                ik_budget=ik_budget,
            )
            for target in targets
        ]

    templates = _reach_rung_templates(
        rig,
        arm_from,
        float(targets[0, 2]),
        stand_xy,
        yaw,
        stand_z,
        floor_z,
    )
    has_soft = [False for _ in targets]
    candidates_by_query = []
    for query_index, target in enumerate(targets):
        candidates, query_has_soft = _candidates_for_target(
            templates, target, tol, allow_colliding_fallback, clear_fn
        )
        has_soft[query_index] = query_has_soft
        candidates_by_query.append(candidates)

    # Spend a limited search budget breadth-first over target points. This evaluates the cheapest rung for
    # every placement candidate before spending a second solve on any one point, instead of exhausting the
    # budget on the first few XYs. One budget unit is one actual batched IK candidate, matching plan_reach.
    flat_candidates = []
    flat_targets = []
    flat_queries = []
    max_rungs = max((len(candidates) for candidates in candidates_by_query), default=0)
    for rung_index in range(max_rungs):
        for query_index, candidates in enumerate(candidates_by_query):
            if rung_index >= len(candidates):
                continue
            flat_candidates.append(candidates[rung_index])
            flat_targets.append(targets[query_index])
            flat_queries.append(query_index)
    if ik_budget is not None:
        allowed = ik_budget.consume_many(len(flat_candidates))
        flat_candidates = flat_candidates[:allowed]
        flat_targets = flat_targets[:allowed]
        flat_queries = flat_queries[:allowed]
    solutions_by_query = [[] for _ in targets]
    chunk_size = max(1, int(os.getenv("ISAAC_BATCH_IK_CHUNK", "256")))
    try:
        for start in range(0, len(flat_candidates), chunk_size):
            stop = min(len(flat_candidates), start + chunk_size)
            chunk = flat_candidates[start:stop]
            poses, errors, joints = core.warp_arms_to_seats(
                rig,
                np.stack([candidate["base"] for candidate in chunk]),
                np.stack([candidate["rz"] for candidate in chunk]),
                np.stack([candidate["world_off"] for candidate in chunk]),
                np.stack(flat_targets[start:stop]),
            )
            for local_index, flat_index in enumerate(range(start, stop)):
                solutions_by_query[flat_queries[flat_index]].append(
                    (
                        flat_candidates[flat_index],
                        {
                            "pose": poses[local_index],
                            "err": float(errors[local_index]),
                            "joints": joints[local_index],
                            "backend": "batch",
                        },
                    )
                )
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            f"[warp-batch] multi-target solve failed ({exc}); falling back to scalar CCD"
        )
        solutions_by_query = [[] for _ in targets]
        for candidate, target, query_index in zip(
            flat_candidates, flat_targets, flat_queries
        ):
            result = core.warp_arm_to_seat(
                rig,
                candidate["base"],
                candidate["rz"],
                candidate["world_off"],
                target,
                return_joints=True,
            )
            pose, err = result[:2]
            joints = result[2] if len(result) > 2 else None
            solutions_by_query[query_index].append(
                (
                    candidate,
                    {
                        "pose": pose,
                        "err": float(err),
                        "joints": (
                            joints
                            if joints is not None
                            else (None if pose is None else core.H.fwd_joints(pose))
                        ),
                        "backend": "scalar",
                    },
                )
            )

    plans = []
    for query_index, target in enumerate(targets):
        best = None
        chosen = None
        verified_count = 0
        for candidate, solution in solutions_by_query[query_index]:
            if _batch_solution_needs_verification(solution, tol):
                if verified_count >= _BATCH_IK_VERIFY_TOP_K:
                    continue
                solution = _refine_reach_solution(rig, candidate, target, solution)
                verified_count += 1
            lf = candidate["lean_rad"] / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0
            current = {
                "lean_rad": candidate["lean_rad"],
                "lf": float(lf),
                "squat": candidate["squat"],
                "stand_z": candidate["stand_z"],
                "err": float(solution["err"]),
                "arm_pose": solution["pose"],
                "arm_joints": solution["joints"],
                "collision_blocked": False,
            }
            if best is None or current["err"] < best["err"]:
                best = current
            if current["err"] <= tol:
                chosen = current
                break
        if chosen is not None:
            chosen["reachable"] = True
            plans.append(chosen)
            continue
        if best is None:
            base_lean = _lean_for_height(float(target[2]))[0]
            best = {
                "lean_rad": float(base_lean),
                "lf": float(base_lean / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0),
                "squat": 0.0,
                "stand_z": float(stand_z),
                "err": float("inf"),
            }
        best["reachable"] = False
        best["collision_blocked"] = bool(
            has_soft[query_index] and not allow_colliding_fallback
        )
        plans.append(best)
    return plans


def plan_reach_shelf(
    rig,
    arm_from: dict,
    contact,
    stand_xy,
    yaw,
    stand_z,
    tol: float,
    gap_ceiling: float,
    floor_z: float = 0.0,
    enabled: bool = True,
    clear_fn=None,
    allow_colliding_fallback: bool = True,
    ik_budget: IKSearchBudget | None = None,
    minimum_squat: float = 0.0,
    exact_rest_crouch: bool = False,
) -> dict:
    """PUTINSIDE-only reach pose: like ``plan_reach`` but pick the least-effort rung whose WHOLE reaching arm
    stays WITHIN the target shelf's gap -- every arm sample point's world z <= ``gap_ceiling`` (the underside
    of the board ABOVE the target) -- AND reaches ``contact`` within ``tol``. This CROUCHES the shoulder DOWN
    into the gap so the arm enters the opening HORIZONTALLY and clears the board above, instead of plan_reach's
    least-effort reach that leaves the shoulder high and angles the arm DOWN through that board. General: the
    gap comes from the container's OWN shelves. ``gap_ceiling=inf`` (top space, no board above) -> the gap
    test is vacuous and it behaves like ``plan_reach``. put / grab / bins keep ``plan_reach`` -- this is used
    ONLY on the shelved PutInside path. ``clear_fn`` applies the same body-severity gate as
    ``plan_reach``; collision-bearing rungs are considered only when the caller explicitly permits
    a fallback. Same return dict as ``plan_reach`` (+ ``reachable``)."""
    z = float(contact[2])
    if not enabled:  # pose-planner off -> height-only lean, no gap search (matches plan_reach disabled)
        base_lean = _lean_for_height(z)[0]
        squat = max(0.0, float(minimum_squat))
        base = _reach_base_pose(rig, arm_from, base_lean, squat, 1.0)
        return {
            "lean_rad": base_lean,
            "lf": (base_lean / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0),
            "squat": squat,
            "stand_z": (_pin_z(base, floor_z) if squat > 1e-6 else float(stand_z)),
            "err": None,
            "reachable": True,
        }
    tgt = np.asarray(contact, dtype=np.float64)
    ceil = float(gap_ceiling)
    best_clear: dict | None = None
    best_soft: dict | None = None
    soft_reachable: dict | None = None
    seen: set[tuple[int, int]] = set()
    candidates = []
    # Mild, head-upright shelf poses first; deeper general poses are allowed only when they also keep the
    # whole arm below the shelf ceiling and the body within the final renderer's severity budget.
    if float(minimum_squat) >= _LOW_FRONT_INSERT_MIN_SQUAT - 1e-6:
        if exact_rest_crouch:
            upright_rungs = _exact_rest_crouch_rungs(
                rig,
                arm_from,
                minimum_squat,
                floor_z,
            )
        else:
            upright_rungs = [
                (lean_rad, squat, 0.0)
                for lean_rad, squat in _head_upright_crouch_rungs(
                    [], minimum_squat
                )
            ]
        for lean_rad, squat, root_drop_m in upright_rungs:
            base = _reach_base_pose(rig, arm_from, lean_rad, squat, 1.0)
            rz_stand = _pin_z(base, floor_z)
            rz, world_off = wpp._grasp_frame(stand_xy, yaw, rz_stand)
            candidates.append(
                {
                    "lean_rad": float(lean_rad),
                    "squat": float(squat),
                    "base": base,
                    "stand_z": float(rz_stand),
                    "rz": rz,
                    "world_off": world_off,
                    "body_clear": clear_fn is None or bool(clear_fn(base, rz_stand)),
                    "root_drop_m": float(root_drop_m),
                }
            )
            seen.add((int(round(lean_rad * 1e6)), int(round(squat * 1e6))))
    for shelf_ladder in (True, False):
        for lean_rad, squat in _reach_rungs(z, floor_z, shelf=shelf_ladder):
            if squat + 1e-6 < float(minimum_squat):
                continue
            key = (int(round(lean_rad * 1e6)), int(round(squat * 1e6)))
            if key in seen:
                continue
            seen.add(key)
            base = _reach_base_pose(rig, arm_from, lean_rad, squat, 1.0)
            rz_stand = _pin_z(base, floor_z) if squat > 1e-6 else float(stand_z)
            rz, world_off = wpp._grasp_frame(stand_xy, yaw, rz_stand)
            candidates.append(
                {
                    "lean_rad": float(lean_rad),
                    "squat": float(squat),
                    "base": base,
                    "stand_z": float(rz_stand),
                    "rz": rz,
                    "world_off": world_off,
                    "body_clear": clear_fn is None or bool(clear_fn(base, rz_stand)),
                }
            )
    solved = _solve_reach_candidates(rig, candidates, tgt, ik_budget)
    arm_tops = []
    for candidate, solution in zip(candidates, solved):
        if solution["backend"] == "batch" and solution["joints"] is not None:
            points = _arm_points_from_joints(
                solution["joints"], candidate["rz"], candidate["world_off"]
            )
        else:
            points = _fwd_arm_points(
                solution["pose"], candidate["rz"], candidate["world_off"]
            )
        arm_tops.append(float(points[:, 2].max()))
    verified_count = 0
    for index, (candidate, solution, arm_top) in enumerate(
        zip(candidates, solved, arm_tops)
    ):
        if _batch_solution_needs_verification(solution, tol):
            if verified_count >= _BATCH_IK_VERIFY_TOP_K:
                continue
            solution = _refine_reach_solution(rig, candidate, tgt, solution)
            solved[index] = solution
            verified_count += 1
            arm_top = float(
                _arm_points_from_joints(
                    solution["joints"], candidate["rz"], candidate["world_off"]
                )[:, 2].max()
            )
        rung = {
            "lean_rad": candidate["lean_rad"],
            "lf": (candidate["lean_rad"] / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0),
            "squat": candidate["squat"],
            "stand_z": candidate["stand_z"],
            "err": solution["err"],
            "arm_pose": solution["pose"],
            "arm_joints": solution["joints"],
            "root_drop_m": float(candidate.get("root_drop_m", 0.0)),
            "exact_rest_crouch": bool(
                exact_rest_crouch and "root_drop_m" in candidate
            ),
        }
        if exact_rest_crouch and "root_drop_m" in candidate:
            print(
                f"[place-exact-rest-rung] target_z={z:.3f} "
                f"lean={math.degrees(float(candidate['lean_rad'])):.0f}deg "
                f"squat={float(candidate['squat']):.2f} "
                f"root_drop={float(candidate['root_drop_m']):.2f}m "
                f"err={float(solution['err']) * 1000:.1f}mm "
                f"body_clear={bool(candidate['body_clear'])} "
                f"gap_clear={arm_top <= ceil + 1e-6}"
            )
        if arm_top > ceil + 1e-6:
            continue
        if candidate["body_clear"]:
            if best_clear is None or rung["err"] < best_clear["err"]:
                best_clear = rung
            if rung["err"] <= tol:
                rung["reachable"] = True
                rung["collision_blocked"] = False
                return rung
        else:
            if best_soft is None or rung["err"] < best_soft["err"]:
                best_soft = rung
            if soft_reachable is None and rung["err"] <= tol:
                soft_reachable = rung
    if soft_reachable is not None and allow_colliding_fallback:
        soft_reachable["reachable"] = True
        soft_reachable["collision_blocked"] = False
        return soft_reachable
    best = best_clear or best_soft
    if best is None:
        base_lean = _lean_for_height(z)[0]
        fallback_squat = max(0.0, float(minimum_squat))
        fallback_pose = _reach_base_pose(
            rig, arm_from, base_lean, fallback_squat, 1.0
        )
        return {
            "lean_rad": float(base_lean),
            "lf": float(base_lean / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0),
            "squat": fallback_squat,
            "stand_z": (
                _pin_z(fallback_pose, floor_z)
                if fallback_squat > 1e-6
                else float(stand_z)
            ),
            "err": float("inf"),
            "reachable": False,
            "collision_blocked": False,
            "root_drop_m": 0.0,
        }
    best["reachable"] = False
    best["collision_blocked"] = bool(
        soft_reachable is not None and not allow_colliding_fallback
    )
    return best


def plan_reach_shelf_many(
    rig,
    arm_from: dict,
    contacts,
    stand_xy,
    yaw,
    stand_z,
    tol: float,
    gap_ceiling: float,
    floor_z: float = 0.0,
    enabled: bool = True,
    clear_fn=None,
    allow_colliding_fallback: bool = True,
    ik_budget: IKSearchBudget | None = None,
    minimum_squat: float = 0.0,
) -> list[dict]:
    """Batch ``plan_reach_shelf`` over equal-height placement targets."""
    targets = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    if not len(targets):
        return []
    if (
        not enabled
        or not _batch_reach_supported(rig)
        or len(targets) == 1
        or float(np.ptp(targets[:, 2])) > 1e-6
    ):
        return [
            plan_reach_shelf(
                rig,
                arm_from,
                target,
                stand_xy,
                yaw,
                stand_z,
                tol,
                gap_ceiling,
                floor_z=floor_z,
                enabled=enabled,
                clear_fn=clear_fn,
                allow_colliding_fallback=allow_colliding_fallback,
                ik_budget=ik_budget,
                minimum_squat=minimum_squat,
            )
            for target in targets
        ]

    active_count = len(targets)
    if ik_budget is not None:
        active_count = ik_budget.consume_many(len(targets))
    templates = []
    seen = set()
    if float(minimum_squat) >= _LOW_FRONT_INSERT_MIN_SQUAT - 1e-6:
        for lean_rad, squat in _head_upright_crouch_rungs([], minimum_squat):
            base = _reach_base_pose(rig, arm_from, lean_rad, squat, 1.0)
            rz_stand = _pin_z(base, floor_z)
            rz, world_off = wpp._grasp_frame(stand_xy, yaw, rz_stand)
            templates.append(
                {
                    "lean_rad": float(lean_rad),
                    "squat": float(squat),
                    "base": base,
                    "stand_z": float(rz_stand),
                    "rz": rz,
                    "world_off": world_off,
                    "body_clear": clear_fn is None or bool(clear_fn(base, rz_stand)),
                }
            )
            seen.add((int(round(lean_rad * 1e6)), int(round(squat * 1e6))))
    for shelf_ladder in (True, False):
        for lean_rad, squat in _reach_rungs(
            float(targets[0, 2]), floor_z, shelf=shelf_ladder
        ):
            if squat + 1e-6 < float(minimum_squat):
                continue
            key = (int(round(lean_rad * 1e6)), int(round(squat * 1e6)))
            if key in seen:
                continue
            seen.add(key)
            base = _reach_base_pose(rig, arm_from, lean_rad, squat, 1.0)
            rz_stand = _pin_z(base, floor_z) if squat > 1e-6 else float(stand_z)
            rz, world_off = wpp._grasp_frame(stand_xy, yaw, rz_stand)
            templates.append(
                {
                    "lean_rad": float(lean_rad),
                    "squat": float(squat),
                    "base": base,
                    "stand_z": float(rz_stand),
                    "rz": rz,
                    "world_off": world_off,
                    "body_clear": clear_fn is None or bool(clear_fn(base, rz_stand)),
                }
            )

    flat_candidates = []
    flat_targets = []
    flat_queries = []
    for query_index, target in enumerate(targets[:active_count]):
        for candidate in templates:
            flat_candidates.append(candidate)
            flat_targets.append(target)
            flat_queries.append(query_index)
    solutions_by_query = [[] for _ in targets]
    chunk_size = max(1, int(os.getenv("ISAAC_BATCH_IK_CHUNK", "256")))
    try:
        for start in range(0, len(flat_candidates), chunk_size):
            stop = min(len(flat_candidates), start + chunk_size)
            chunk = flat_candidates[start:stop]
            poses, errors, joints = core.warp_arms_to_seats(
                rig,
                np.stack([candidate["base"] for candidate in chunk]),
                np.stack([candidate["rz"] for candidate in chunk]),
                np.stack([candidate["world_off"] for candidate in chunk]),
                np.stack(flat_targets[start:stop]),
            )
            arm_points = _arm_points_from_joints(
                joints,
                np.stack([candidate["rz"] for candidate in chunk]),
                np.stack([candidate["world_off"] for candidate in chunk]),
            )
            arm_tops = arm_points[:, :, 2].max(axis=1)
            for local_index, flat_index in enumerate(range(start, stop)):
                solutions_by_query[flat_queries[flat_index]].append(
                    (
                        flat_candidates[flat_index],
                        {
                            "pose": poses[local_index],
                            "err": float(errors[local_index]),
                            "joints": joints[local_index],
                            "backend": "batch",
                        },
                        float(arm_tops[local_index]),
                    )
                )
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            f"[warp-batch] shelf solve failed ({exc}); falling back to scalar CCD"
        )
        return [
            plan_reach_shelf(
                rig,
                arm_from,
                target,
                stand_xy,
                yaw,
                stand_z,
                tol,
                gap_ceiling,
                floor_z=floor_z,
                enabled=True,
                clear_fn=clear_fn,
                allow_colliding_fallback=allow_colliding_fallback,
                minimum_squat=minimum_squat,
            )
            for target in targets
        ]

    plans = []
    for query_index, target in enumerate(targets):
        best_clear = None
        best_soft = None
        soft_reachable = None
        verified_count = 0
        for candidate, solution, arm_top in solutions_by_query[query_index]:
            if _batch_solution_needs_verification(solution, tol):
                if verified_count >= _BATCH_IK_VERIFY_TOP_K:
                    continue
                solution = _refine_reach_solution(rig, candidate, target, solution)
                verified_count += 1
                arm_top = float(
                    _arm_points_from_joints(
                        solution["joints"], candidate["rz"], candidate["world_off"]
                    )[:, 2].max()
                )
            if arm_top > float(gap_ceiling) + 1e-6:
                continue
            rung = {
                "lean_rad": candidate["lean_rad"],
                "lf": (candidate["lean_rad"] / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0),
                "squat": candidate["squat"],
                "stand_z": candidate["stand_z"],
                "err": solution["err"],
                "arm_pose": solution["pose"],
                "arm_joints": solution["joints"],
            }
            if candidate["body_clear"]:
                if best_clear is None or rung["err"] < best_clear["err"]:
                    best_clear = rung
                if rung["err"] <= tol:
                    rung["reachable"] = True
                    rung["collision_blocked"] = False
                    best_clear = rung
                    break
            else:
                if best_soft is None or rung["err"] < best_soft["err"]:
                    best_soft = rung
                if soft_reachable is None and rung["err"] <= tol:
                    soft_reachable = rung
        if best_clear is not None and best_clear.get("reachable"):
            plans.append(best_clear)
            continue
        if soft_reachable is not None and allow_colliding_fallback:
            soft_reachable["reachable"] = True
            soft_reachable["collision_blocked"] = False
            plans.append(soft_reachable)
            continue
        best = best_clear or best_soft
        if best is None:
            base_lean = _lean_for_height(float(target[2]))[0]
            fallback_squat = max(0.0, float(minimum_squat))
            fallback_pose = _reach_base_pose(
                rig, arm_from, base_lean, fallback_squat, 1.0
            )
            best = {
                "lean_rad": float(base_lean),
                "lf": float(base_lean / _LEAN_RAD if _LEAN_RAD > 1e-9 else 0.0),
                "squat": fallback_squat,
                "stand_z": (
                    _pin_z(fallback_pose, floor_z)
                    if fallback_squat > 1e-6
                    else float(stand_z)
                ),
                "err": float("inf"),
            }
        best["reachable"] = False
        best["collision_blocked"] = bool(
            soft_reachable is not None and not allow_colliding_fallback
        )
        plans.append(best)
    return plans


def _apply_front_extract_camera_delta(
    frames: list[wpp.Frame],
    container_key: str,
    container,
) -> list[wpp.Frame]:
    meta = getattr(container, "meta", None) or {}
    if meta.get("access_mode") != "front_opening":
        return frames
    configured = meta.get("content_extract_camera_pullback_m")
    if configured is None:
        return frames
    delta = float(configured) - _PRE_LIFT_EXTRACT_CAMERA_PULLBACK_MAX_M
    if not math.isfinite(delta) or delta <= 0.0:
        return frames

    grasp_indices = [
        index for index, frame in enumerate(frames) if frame.seg == "grasp"
    ]
    post_contact_count = _N_REACH - max(1, int(round(_N_REACH * 0.55)))
    frame_indices = grasp_indices[-post_contact_count:]
    frame_indices.extend(
        index for index, frame in enumerate(frames) if frame.seg == "lift"
    )
    for frame_index in frame_indices:
        frame = frames[frame_index]
        frames[frame_index] = replace(
            frame,
            seg=f"front_extract_{frame.seg}",
            openable_key=container_key,
            camera_eye_offset=-np.asarray(wpp._body_forward(frame.yaw)) * delta,
        )
    return frames


def produce_pick(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Reach + grasp + lift the object ``goal.obj_key`` from where the body currently stands.

    Solves the reach+grip ONCE (CCD warp + optional live finger wrap), then replays it rigidly. Sets
    st.held so subsequent NAVIGATE carries it and PLACE can deliver it."""
    if st.held is not None:
        raise PrimitiveFailure(
            PrimitiveError.GRASP_FAILED, "PICK while already holding an object"
        )
    try:
        # Use the object's ORIGIN (its frozen rest pose), NOT object_pos's world-bbox BASE: realize drives
        # this value straight into the prim's transform translation, so sourcing the base double-applies
        # min_z -> a center-pivot object sinks by |min_z| and jumps down the moment the rest-drive begins
        # (before the hand contacts it). object_origin == object_pos for base-pivot props.
        obj_world = scene.object_origin(goal.obj_key)
        source_rest_pose = scene.object_rest_mat(goal.obj_key)
        source_surface = _carried_parent_map(scene).get(goal.obj_key)
    except KeyError:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT, f"unknown object {goal.obj_key!r}"
        )
    source_container_context = _open_front_container_context(scene, source_surface)
    source_container = (
        source_container_context[0] if source_container_context is not None else None
    )
    source_container_gaze = (
        source_container_context[1] if source_container_context is not None else None
    )
    low_front_pick = source_container is not None and _low_front_openable(
        scene,
        scene.resolve(source_container),
    )
    crouch_owner = _pick_crouch_owner(scene, st, goal.obj_key, source_container)
    keep_crouched = crouch_owner is not None
    minimum_squat = (
        max(_FRONT_APPLIANCE_SQUAT, _active_crouch_fraction(scene, st, goal.obj_key))
        if keep_crouched
        else 0.0
    )
    # NATURAL GRASP: build the live mesh before planning the reach because broad, flat objects need an
    # edge contact rather than their centre pivot.  The object's rest transform remains separate below,
    # so moving the hand target does not move or rotate the object before contact.
    entity = scene.resolve(goal.obj_key)
    grasp_hint = (getattr(entity, "meta", None) or {}).get("grasp_hint")
    live = bool(getattr(scene, "live_wrap", False))
    if sd_world is None and live and getattr(rig, "_stage", None) is not None:
        import render_runtime

        _pp = getattr(scene.resolve(goal.obj_key), "prim_path", None) or getattr(
            rig, "_obj_path", "/World/Obj"
        )
        try:
            sd_world = render_runtime._make_sd_world(
                rig._stage, _pp, cache_key=goal.obj_key
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            print(
                f"[pick] {goal.obj_key}: sd_world build failed ({exc}) -> canned grip"
            )
            sd_world = None
    grip_world = (
        grasp_select.grasp_target(
            sd_world.mesh,
            obj_world,
            st.root_xy,
            hint=grasp_hint,
        )
        if sd_world is not None
        else np.asarray(obj_world, dtype=np.float64)
    )
    # POSE-PLANNER (gated scene.unified_geom): least-effort pose (forward-lean, else lean+SQUAT) that
    # actually reaches the object (real 3D IK), so a FAR object is reached by leaning forward and a LOW/floor
    # object by CROUCHING, instead of failing OUT_OF_REACH. Off -> the height-only lean with upright legs.
    # tol == the pick accept tol below.
    _pp_on = bool(getattr(scene, "unified_geom", False))
    _reach_tol = _goal_reach_tolerance(goal, _reach_tolerance(scene))
    _egocentric_reach_fallback = "_egocentric_reach_tol_m" in (goal.meta or {})
    _plan = plan_reach(
        rig,
        rig.arm_grasp,
        grip_world,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        _reach_tol,
        floor_z=st.floor_z,
        enabled=_pp_on,
        clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
        allow_colliding_fallback=_egocentric_reach_fallback,
        upright_first=keep_crouched,
        upright_crouch_first=low_front_pick,
        minimum_squat=minimum_squat,
    )
    if _plan.get("collision_blocked"):
        raise PrimitiveFailure(
            PrimitiveError.TRAJECTORY_COLLISION,
            f"PICK {goal.obj_key!r} has no collision-free reach pose",
        )
    lean_rad, lf, squat_f, reach_z = (
        _plan["lean_rad"],
        _plan["lf"],
        _plan["squat"],
        _plan["stand_z"],
    )
    if _plan.get("err") is not None:
        print(
            f"[pick] {goal.obj_key}: pose-plan lean={math.degrees(lean_rad):.0f}deg "
            f"squat={squat_f:.1f} reach_err={_plan['err'] * 1000:.0f}mm reachable={_plan['reachable']}"
        )
    # squat lowers the crouched shoulder -> the grip must SOLVE (and render) from the pinned crouch root z.
    rz, world_off = wpp._grasp_frame(st.root_xy, st.root_yaw, reach_z)
    base = _reach_base_pose(rig, rig.arm_grasp, lean_rad, squat_f, 1.0)
    grasp = wpp.solve_grasp(
        rig,
        base,
        rz,
        world_off,
        grip_world,
        sd_world,
        family_seed=live,
        object_rest_pose=source_rest_pose,
        planned_pose=_plan.get("arm_pose"),
        planned_err=_plan.get("err"),
        grasp_hint=grasp_hint,
    )
    print(
        f"[pick] {goal.obj_key}: seat_err={grasp['seat_err'] * 1000:.1f}mm near={grasp.get('near')}"
    )
    # A dense BEHAVIOR scene (hotel/office) can only seat the canned grip ~80 mm from a small can/battery on a
    # cramped surface; soft-init demos accept that looser grip (the object is kinematically held after) rather
    # than fail the whole pick. Precise scenes keep the tight 30 mm tol (their picks are already well under
    # 30 mm, so widening the accept threshold changes nothing for them).
    _pick_tol = _reach_tol
    if grasp["seat_err"] > _pick_tol:
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"{goal.obj_key}: seat_err={grasp['seat_err'] * 1000:.0f}mm",
        )
    grasp_arm, rhp, otw = grasp["grasp_arm"], grasp["rhp"], grasp["otw"]
    contact_arm = grasp_arm
    rig.rhp = rhp  # so assemble_pose(hand_close=1) reproduces the wrapped grip

    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    # Every pre-contact frame must replay the object's current full transform.  Using
    # xyz alone silently restores identity rotation, which can expose a label that a
    # prior Flip intentionally hid just before the hand closes on the object.
    obj_rest = np.asarray(source_rest_pose, dtype=np.float64).copy()
    # World-fixed point the hand closes onto. For large flat props this is a reachable perimeter point;
    # the full object still rides from source_rest_pose through ``otw``.
    contact_pt = np.asarray(grip_world, dtype=np.float64)
    arrival_kwargs = {"squat_frac": squat_f} if keep_crouched else {}
    frames = _arrive_beats(
        rig,
        st,
        arm_rest,
        0.0,
        ("rest", obj_rest),
        **arrival_kwargs,
    )
    if source_container_gaze is not None:
        frames = [
            replace(frame, gaze=np.asarray(source_container_gaze, dtype=np.float64))
            for frame in frames
        ]

    def add(pose, obj_state, seg, contact=None):
        # SQUAT: re-pin the feet to the floor per-pose (the root lowers as the body crouches). Standing
        # frames keep st.stand_z (always the case when squat_f == 0).
        _z = _pin_z(pose, st.floor_z) if squat_f > 1e-6 else st.stand_z
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                st.root_yaw,
                _z,
                fpv_mode="look",
                obj=obj_state,
                seg=seg,
                contact_anchor=contact,
                gaze=(
                    np.asarray(source_container_gaze, dtype=np.float64)
                    if source_container_gaze is not None
                    else None
                ),
            )
        )

    e = entity
    if e.foot_half is None or squat_f > 1e-6:
        # DIRECT REACH: no furniture footprint -> the pre/retreat waypoints degenerate to the grasp
        # point, so run the plain reach/lift (cup-on-desk). The low-pick lean-unwind path is kept.
        for i in range(_N_REACH):
            s = wpp._smoothstep((i + 1) / _N_REACH)
            held = i >= _N_REACH - 1
            # The hand is closing onto the still-resting object across the whole reach -> FIXED contact.
            add(
                _manip(
                    rig,
                    core.lerp_arm(arm_rest, grasp_arm, s),
                    s,
                    lf * s,
                    squat_f if keep_crouched else squat_f * s,
                ),
                ("carry", otw) if held else ("rest", obj_rest),
                "grasp",
                contact=None if held else contact_pt,
            )
        if lf > 1e-6:
            for i in range(_N_LIFT):
                s = wpp._smoothstep((i + 1) / _N_LIFT)
                add(
                    _manip(
                        rig,
                        grasp_arm,
                        1.0,
                        lf * (1.0 - s),
                        squat_f if keep_crouched else squat_f * (1.0 - s),
                    ),
                    ("carry", otw),
                    "lift",
                )
        else:
            obj_up = np.asarray(grip_world, dtype=np.float64) + np.array(
                [0.0, 0.0, _LIFT_RISE_M], dtype=np.float64
            )
            grasp_arm_up, lift_err = _solve_reach_arm(
                rig,
                grasp_arm,
                obj_up,
                st.root_xy,
                st.root_yaw,
                reach_z if keep_crouched else st.stand_z,
                squat_frac=squat_f if keep_crouched else 0.0,
            )
            print(
                f"[pick] lift-up: rise={_LIFT_RISE_M * 1000:.0f}mm seat_err={lift_err * 1000:.1f}mm"
            )
            for i in range(_N_LIFT):
                s = wpp._smoothstep((i + 1) / _N_LIFT)
                add(
                    _manip(
                        rig,
                        core.lerp_arm(grasp_arm, grasp_arm_up, s),
                        1.0,
                        0.0,
                        squat_f if keep_crouched else 0.0,
                    ),
                    ("carry", otw),
                    "lift",
                )
            grasp_arm = grasp_arm_up  # carry the lifted pose forward (st.held below)
    else:
        # APPROACH-WAYPOINT reach: come in from the geometry-derived approach side (pre_grasp), grasp,
        # then retreat UP+back so the arm clears the furniture instead of swinging through it.
        wp = plan_reach_waypoints(scene, goal.obj_key, st.root_xy, grip_world)
        boxes = _reach_boxes(scene)
        arm_pre, _, _, _ = _solve_and_check(
            rig,
            rig.arm_grasp,
            wp["pre"],
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            boxes,
            "pick pre_grasp",
            strict=_body_collision_is_hard(scene),
        )
        # grasp leg: solved above by solve_grasp; check its arm clearance + log with the same format.
        dg, hg = _verts_in_boxes(
            _fwd_arm_points(
                _manip(rig, grasp_arm, 1.0, lf),
                *wpp._grasp_frame(st.root_xy, st.root_yaw, st.stand_z),
            ),
            boxes,
            exclude_sphere=(grip_world, 0.22),
        )
        print(
            f"[reach] pick grasp: err={grasp['seat_err'] * 1000:.1f}mm arm-clear worst={dg * 1000:.0f}mm hit={hg or '-'}"
        )
        _require_arm_clear(scene, dg, hg, "pick grasp")
        arm_ret, _, _, _ = _solve_and_check(
            rig,
            grasp_arm,
            wp["retreat"],
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            boxes,
            "pick retreat",
            strict=_body_collision_is_hard(scene),
        )
        n_pre = max(1, int(round(_N_REACH * 0.55)))
        for i in range(_N_REACH):
            if (
                i < n_pre
            ):  # rest -> pre_grasp, hand OPEN, no lean yet (approach from the front)
                s = wpp._smoothstep((i + 1) / n_pre)
                add(
                    _manip(rig, core.lerp_arm(arm_rest, arm_pre, s), 0.0, 0.0),
                    ("rest", obj_rest),
                    "grasp",
                )
            else:  # pre_grasp -> grasp, hand closing 0->1, lean in; object held on the last frame
                k = i - n_pre
                s = wpp._smoothstep((k + 1) / (_N_REACH - n_pre))
                held = i >= _N_REACH - 1
                # Closing onto the still-resting object -> FIXED contact (the pre_grasp approach frames
                # above are the hand travelling in, NOT yet on the target, so they stay unflagged).
                add(
                    _manip(rig, core.lerp_arm(arm_pre, grasp_arm, s), s, lf * s),
                    ("carry", otw) if held else ("rest", obj_rest),
                    "grasp",
                    contact=None if held else contact_pt,
                )
        for i in range(
            _N_LIFT
        ):  # LIFT: grasp -> retreat (up + back off the approach), carrying
            s = wpp._smoothstep((i + 1) / _N_LIFT)
            add(
                _manip(rig, core.lerp_arm(grasp_arm, arm_ret, s), 1.0, lf * (1.0 - s)),
                ("carry", otw),
                "lift",
            )
        grasp_arm = arm_ret  # carry the retreated pose forward (st.held below)

    if source_container is not None and not low_front_pick:
        container = scene.resolve(source_container)
        if (getattr(container, "meta", None) or {}).get(
            "access_mode"
        ) == "front_opening":
            start_root = tuple(float(value) for value in st.root_xy)
            safe_root, safe_yaw = _front_opening_post_interaction_pose(
                scene,
                source_container,
                container,
                start_root,
                st.root_yaw,
            )
            start_yaw = float(st.root_yaw)
            yaw_delta = (safe_yaw - start_yaw + 180.0) % 360.0 - 180.0
            lift_indices = [
                index for index, frame in enumerate(frames) if frame.seg == "lift"
            ]
            for offset, frame_index in enumerate(lift_indices):
                fraction = wpp._smoothstep((offset + 1) / len(lift_indices))
                frame_yaw = start_yaw + yaw_delta * fraction
                frames[frame_index] = replace(
                    frames[frame_index],
                    xy=(
                        start_root[0]
                        + (safe_root[0] - start_root[0]) * fraction,
                        start_root[1]
                        + (safe_root[1] - start_root[1]) * fraction,
                    ),
                    yaw=frame_yaw,
                    gaze=np.asarray(source_container_gaze, dtype=np.float64),
                )
            if lift_indices:
                st.root_xy = safe_root
                st.root_yaw = safe_yaw
                st.walk_end_yaw = safe_yaw

    if source_container is not None:
        frames = _apply_front_extract_camera_delta(
            frames,
            source_container,
            scene.resolve(source_container),
        )

    navigation_arm = _large_container_navigation_arm(
        rig,
        scene,
        st,
        entity,
        contact_arm,
        lf,
        squat_f,
    )
    large_front_extract = navigation_arm is not None and low_front_pick
    if large_front_extract:
        frames = [
            replace(
                frame,
                seg=(
                    frame.seg
                    if frame.seg and frame.seg.startswith("front_extract_")
                    else f"front_extract_{frame.seg}"
                    if frame.seg
                    else "front_extract"
                ),
            )
            for frame in frames
        ]
        for index in range(_N_LIFT):
            fraction = wpp._smoothstep((index + 1) / _N_LIFT)
            squat = squat_f * (1.0 - fraction)
            pose = _manip(
                rig,
                core.lerp_arm(grasp_arm, navigation_arm, fraction),
                1.0,
                0.0,
                squat,
            )
            frames.append(
                wpp.Frame(
                    pose,
                    st.root_xy,
                    st.root_yaw,
                    _pin_z(pose, st.floor_z) if squat > 1e-6 else st.stand_z,
                    fpv_mode="look",
                    obj=("carry", otw),
                    seg="front_extract_settle",
                    gaze=(
                        np.asarray(source_container_gaze, dtype=np.float64)
                        if source_container_gaze is not None
                        else None
                    ),
                )
            )
        grasp_arm = navigation_arm
    carried = _gather_carried(scene, goal.obj_key)
    st.held = {
        "otw": otw,
        "rhp": rhp,
        "grasp_arm": grasp_arm,
        "obj_key": goal.obj_key,
        "hand": "right",
        "source_rest_pose": source_rest_pose,
        "source_surface": source_surface,
        "source_open_container": source_container,
        "carried": carried,
    }
    if navigation_arm is not None:
        st.held["navigation_arm"] = navigation_arm
    # Gather what rides this object (a bowl's food) as ONE rigid body, captured at the grasp pose
    # (before it moves). Persisted on st.held so the walk/place carry frames stamp it too.
    if carried:
        print(
            f"[carry] {goal.obj_key} carries "
            f"{[c[0].rsplit('/', 1)[-1] for c in carried]} as one rigid body"
        )
    st.obj_rest_xyz = None
    st.last_pose156 = _manip(
        rig,
        grasp_arm,
        1.0,
        0.0,
        squat_f if keep_crouched and not large_front_extract else 0.0,
    )
    scene.declare_pick(
        goal.obj_key
    )  # trusted-execution state effect: object is now held
    if keep_crouched and not large_front_extract:
        _set_crouch_hold(st, crouch_owner, squat_f, st.last_pose156)
    elif large_front_extract:
        _clear_crouch_hold(st)
    _stamp_obj_path(
        frames, scene, goal.obj_key
    )  # multi-graspable: tag which prim rides the hand
    _stamp_carried(
        frames, st.held["carried"]
    )  # its contents ride the same carry frames
    return frames


def _local_x_rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.eye(4, dtype=np.float64)
    rotation[:3, :3] = ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    return rotation


def _local_z_rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.eye(4, dtype=np.float64)
    rotation[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    return rotation


def _rest_value_for_drop(
    held: dict,
    obj_drop,
    *,
    local_yaw_rad: float = 0.0,
    upright: bool = False,
) -> np.ndarray | tuple:
    pose = held.get("rest_pose_target")
    if pose is None:
        pose = held.get("source_rest_pose")
    if pose is None:
        if not upright:
            return tuple(float(value) for value in obj_drop)
        value = np.eye(4, dtype=np.float64)
        value[:3, 3] = tuple(float(component) for component in obj_drop)
        return value
    value = np.asarray(pose, dtype=np.float64).copy()
    if value.shape != (4, 4):
        raise ValueError("held rest pose must be a 4x4 transform")
    if upright:
        scale = np.linalg.norm(value[:3, :3], axis=0)
        if np.any(scale <= 1e-9):
            raise ValueError("held rest pose has a degenerate scale")
        horizontal_x = value[:2, 0]
        if float(np.linalg.norm(horizontal_x)) > 1e-9:
            yaw = math.atan2(float(horizontal_x[1]), float(horizontal_x[0]))
        else:
            horizontal_y = value[:2, 1]
            yaw = math.atan2(-float(horizontal_y[0]), float(horizontal_y[1]))
        value[:3, :3] = _local_z_rotation(yaw)[:3, :3] @ np.diag(scale)
    if abs(float(local_yaw_rad)) > 1e-9:
        value = value @ _local_z_rotation(float(local_yaw_rad))
    value[:3, 3] = tuple(float(component) for component in obj_drop)
    return value


def _carry_yaw_frames(
    st: SchedState, initial_otw, local_yaw_rad: float
) -> tuple[list[wpp.Frame], np.ndarray]:
    """Rotate a held rigid object upright about its local vertical before a tight placement."""
    initial = np.asarray(initial_otw, dtype=np.float64)
    if initial.shape != (4, 4):
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            "held transform for automatic placement rotation is not a 4x4 pose",
        )
    if abs(float(local_yaw_rad)) <= 1e-9:
        return [], initial
    if st.last_pose156 is None:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            "automatic placement rotation has no current human pose",
        )
    frames = []
    for index in range(_N_FLIP):
        fraction = wpp._smoothstep((index + 1) / _N_FLIP)
        rotated = initial @ _local_z_rotation(float(local_yaw_rad) * fraction)
        frames.append(
            wpp.Frame(
                st.last_pose156.copy(),
                st.root_xy,
                st.root_yaw,
                _pin_z(st.last_pose156, st.floor_z),
                fpv_mode="look",
                obj=("carry", rotated),
                seg="orient",
            )
        )
    return frames, initial @ _local_z_rotation(float(local_yaw_rad))


def produce_flip(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Turn the held object face-down without implicitly releasing or placing it."""
    del rig, g, sd_world
    if st.held is None:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET, "FLIP requires a held object"
        )
    held_key = str(st.held["obj_key"])
    if goal.obj_key != held_key:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"FLIP target {goal.obj_key!r} is not the held object {held_key!r}",
        )
    initial_otw = np.asarray(st.held.get("otw"), dtype=np.float64)
    if initial_otw.shape != (4, 4):
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"FLIP held transform for {held_key!r} is not a 4x4 pose",
        )
    frames = []
    for index in range(_N_FLIP):
        fraction = wpp._smoothstep((index + 1) / _N_FLIP)
        rotated_otw = initial_otw @ _local_x_rotation(math.pi * fraction)
        frames.append(
            wpp.Frame(
                st.last_pose156.copy(),
                st.root_xy,
                st.root_yaw,
                st.stand_z,
                fpv_mode="look",
                obj=("carry", rotated_otw),
                seg="flip",
            )
        )
    final_rotation = _local_x_rotation(math.pi)
    st.held["otw"] = initial_otw @ final_rotation
    rest_pose = st.held.get("rest_pose_target")
    if rest_pose is None:
        rest_pose = st.held.get("source_rest_pose")
    if rest_pose is not None:
        rest_pose = np.asarray(rest_pose, dtype=np.float64)
        if rest_pose.shape != (4, 4):
            raise PrimitiveFailure(
                PrimitiveError.PRECONDITION_UNMET,
                f"FLIP rest transform for {held_key!r} is not a 4x4 pose",
            )
        st.held["rest_pose_target"] = rest_pose @ final_rotation
    _stamp_obj_path(frames, scene, held_key)
    _stamp_carried(frames, st.held.get("carried"))
    return frames


def _space_by_request(req, by_name, spaces):
    """Resolve an EXPLICIT ``goal.meta['space']`` to one space tuple, or None to auto-distribute.

    Accepts the exact space name, a ``space{i}`` tail, or a bare 1-based index (clamped). An
    unrecognised name returns None so the caller auto-distributes -- never silently the container front,
    which would put the object on the rim."""
    if req is None:
        return None
    key = str(req).strip().lower()
    if key in by_name:
        return by_name[key]
    tail = key.rsplit("space", 1)[-1].lstrip("_") if "space" in key else key
    if tail.isdigit():
        return spaces[max(1, min(len(spaces), int(tail))) - 1]
    return None


def _narrow_pool(spaces, assigned, rejected, fits):
    """The physically usable spaces worth trying, best-first.

    Physical fit is applied BEFORE reservation: when every fitting shelf already has an object, overflow
    must reuse one of those shelves at a different XY, not choose an unused shelf whose gap is too short.
    The remaining narrowings fall back only within that physically valid pool:
      assigned  spaces already reserved for a DIFFERENT object (overflow reuses fitting spaces);
      rejected  spaces THIS goal already tried and failed from a reach-capable stand, so its retry
                re-CHOOSES instead of re-attempting the identical drop (per-goal; still free for others);
      fits      spaces the held object can physically occupy -- enough height, support, and free volume.

    An empty result is an honest ``TARGET_TOO_SMALL``/``NO_FREE_PLACEMENT`` upstream, never an invitation
    to force the object through geometry."""
    valid = [s for s in spaces if fits(s)] if fits is not None else list(spaces)
    if not valid:
        return []
    pool = [s for s in valid if s[0] not in assigned] or valid
    if rejected:
        pool = [s for s in pool if s[0] not in rejected] or pool
    return pool


def _select_space(
    spaces, goal, scene, surf_key, space_err=None, tol=_REACH_TOL_M, fits=None
):
    """Pick the interior SPACE for a PLACE_INSIDE drop, from ``spaces`` = ``((name, top_z, bottom_z, cx, cy, hx, hy),
    ...)`` ordered TOP-DOWN then LEFT-RIGHT. An explicit ``goal.meta['space']`` selects a space by exact
    name, by a ``space{i}`` tail, or by a bare 1-based index. With no space given, AUTO-DISTRIBUTE: among the
    not-yet-assigned spaces (top-down), the first one WITHIN REACH from the current stand (``space_err`` <=
    ``tol``), so different objects fill different spaces (upper levels first) via the NORMAL reach path. If NONE
    is in reach from here, the LEAST-unreachable space (min ``space_err``) -- so the place-renav walks closer
    and rescues it, NOT the top shelf no stand can reach. No predicate -> the first free space. The chosen
    space is RESERVED (later auto drops skip it) and remembered on the goal so a re-nav RETRY of the SAME goal
    reuses it (no double-consume). Returns one space tuple, or None for an empty container."""
    if not spaces:
        return None
    by_name = {s[0].lower(): s for s in spaces}
    _amap = getattr(scene, "_space_assign", None)
    if _amap is None:
        _amap = scene._space_assign = {}
    assigned = _amap.setdefault(surf_key, [])

    def _commit(sp):
        if sp[0] not in assigned:
            assigned.append(
                sp[0]
            )  # reserve it so later auto drops into this container skip it
        goal.meta["_space_chosen"] = sp[0]
        return sp

    # A re-nav RETRY of the SAME goal reuses the space it already chose (does not consume another) --
    # UNLESS the caller has since REJECTED it, which is how a retry asks for a different shelf rather than
    # a second attempt at the identical drop.
    _rej = set(goal.meta.get("_space_rejected") or ())
    chosen_name = goal.meta.get("_space_chosen")
    if (
        chosen_name is not None
        and chosen_name.lower() in by_name
        and chosen_name not in _rej
    ):
        return by_name[chosen_name.lower()]

    _req = _space_by_request(goal.meta.get("space"), by_name, spaces)
    if _req is not None:
        return _commit(_req)

    pool = _narrow_pool(spaces, assigned, _rej, fits)
    if not pool:
        return None
    if space_err is not None and pool:
        errs = [(s, space_err(s)) for s in pool]
        for s, e in errs:
            if e <= tol:
                return _commit(
                    s
                )  # first IN-REACH free space (top-down) -> normal path, no re-nav
        # None in reach from HERE: pick the LEAST-unreachable, so the place-renav walks closer and rescues it
        # -- NOT the top shelf (a tall container's top board is outside the human reach envelope from any stand).
        return _commit(min(errs, key=lambda x: x[1])[0])
    return _commit(pool[0])  # no predicate -> honest OUT_OF_REACH / re-nav


def _drop_site_probe(
    xy, top_z, obj_path, obj_half_xy, obj_h, *, probe_override=None
):
    """Ask PhysX whether the held object can rest at ``(xy, top_z)``: real surface under it, and the volume
    it will occupy free. Returns the probe dict, or None when it cannot be asked (unmeasured object, no
    prim, binding unavailable) -- and None must be read as "unknown", never as "no".

    Placement spaces are authored BEFORE any collider exists (the builder returns, THEN
    ``_apply_scene_colliders`` runs), so the authoring step physically cannot check its own claims. During
    the action the world is live, which makes this the first and only moment they can be tested."""
    if probe_override is not None:
        try:
            result = probe_override(xy, top_z, obj_path, obj_half_xy, obj_h)
        except Exception:  # noqa: BLE001
            result = None
        return result or {
            "ok": False,
            "support": False,
            "clear": False,
            "embedded": True,
            "reason": "blocked",
            "hit_z": None,
            "support_frac": 0.0,
            "support_path": None,
            "blockers": ["<content-preview-unmeasured>"],
        }
    if obj_h is None or not obj_path:
        return None
    try:
        import physics_human as _ph_probe

        return _ph_probe.probe_drop_site(
            (float(xy[0]), float(xy[1])),
            float(top_z),
            obj_half_xy,
            float(obj_h),
            exclude_prefixes=(obj_path, "/World/HumanProxy", "/World/Obj"),
        )
    except Exception:  # noqa: BLE001
        return None


def _measured_seat_z(fpr, board_z, base_off, obj_path, place_z, commanded, label):
    """Seat the object on the surface the RAY found rather than on the board top the space DECLARES.

    The two differ by a few mm -- a space's ``top_z`` comes from authored/visual geometry, while the object
    lands on whatever collider PhysX owns there -- and a few mm is enough to break the placement outright.
    One settle step is 1/12 s, in which free fall covers 34 mm, so the object crosses the gap and ends up
    INSIDE the board within a SINGLE step. PhysX then resolves that penetration along the contact normal,
    which is HORIZONTAL wherever the support mesh also carries the unit's vertical faces, and the object
    leaves sideways. Measured on bt_chemlab: declared 0.8470, real 0.8401, base seated 0.8473 -> step 1 base
    0.8369 at v=(-0.124,-0.002,+0.020). Closing the gap removes the fall, the penetration and the kick at
    once; severity scales with gap x local mesh shape, which is why the same code slid one bottle 84 mm and
    threw another 2 762 mm.

    Correcting only ever moves the seat WITHIN the probe's own tolerance band, and only when the ray landed
    on the board it was aimed at (``support``) and on something other than the held object itself. A ray
    through a hole reports no-support and keeps the declared height, i.e. no correction at all -- the
    space-rejection path already owns that case. Returns the z to use."""
    if commanded or not fpr or not fpr.get("support"):
        return place_z
    hz, on = fpr.get("hit_z"), str(fpr.get("support_path") or "")
    if hz is None or (obj_path and on.startswith(obj_path)):
        return place_z
    seat = float(hz) + base_off
    if abs(seat - float(place_z)) > 1e-4:
        print(
            f"[place-seat] {label}: board {float(board_z):.4f} -> measured {float(hz):.4f} "
            f"({(float(hz) - float(board_z)) * 1000:+.1f}mm), "
            f"place_z {float(place_z):.4f} -> {seat:.4f}"
        )
    return seat


def _lateral_free_window(xy, top_z, obj_path, obj_h, *, probe_override=None):
    """Measure the usable room around a drop on the LIVE colliders. None = could not ask, never "no room".

    The human is excluded (it walks away, and it is standing exactly where the object is being handed
    over); OTHER OBJECTS ARE NOT -- a neighbour already resting on this surface is a real limit on where
    this one can go, and the outward support march reads its top as "the surface ends here"."""
    if probe_override is not None:
        try:
            return probe_override(xy, top_z, obj_path, obj_h)
        except Exception:  # noqa: BLE001
            return None
    if obj_h is None or not obj_path:
        return None
    try:
        import physics_human as _ph_win

        return _ph_win.free_rest_window(
            (float(xy[0]), float(xy[1])),
            float(top_z),
            float(obj_h),
            exclude_prefixes=(obj_path, "/World/HumanProxy"),
        )
    except Exception:  # noqa: BLE001
        return None


def _recentre_free_xy(place_xy, win, obj_half_xy, label):
    """Slide a drop that does NOT FIT where it was chosen into the middle of the measured free room.

    A drop point is authored blind -- for an open deck it is a fixed fraction of the unit's OUTER bounding
    box, which knows nothing about what stands on the surface. Measured on ``bt_chemlab``'s fume hood that
    lands every bottle 30 mm inside the glass sash (a collider that rises 0.48 m off the deck across its
    full width) on the 172 mm ledge in FRONT of it, 142 mm from the deck's front edge. Physics resolves
    that on the first step by shoving the bottle out of the sash toward the edge; a squat bottle stops, a
    slender one tips over it and falls 0.95 m to the floor.

    Only acts when the chosen point leaves the object OVERHANGING its support or INTERSECTING something
    standing on it -- a negative margin on some axis. A drop that already fits is returned unchanged.
    Never vetoes: with no measurement, no feasible centre, or no strict improvement, the original point
    is kept and physics gets exactly the original drop."""
    if not win:
        return place_xy
    out = [float(place_xy[0]), float(place_xy[1])]
    moved = []
    for i, axis in enumerate("xy"):
        neg, pos = (win.get(axis) or (None, None))[:2]
        half = float(obj_half_xy[i])
        if neg is None or pos is None or min(neg, pos) >= half:
            continue  # already fits on this axis
        lo, hi = out[i] - float(neg), out[i] + float(pos)
        if hi - lo < 2.0 * half:
            continue  # nowhere along this axis is wide enough for the object at all
        tgt = min(
            max(0.5 * (lo + hi), out[i] - _RECENTRE_CAP_M), out[i] + _RECENTRE_CAP_M
        )
        if min(tgt - lo, hi - tgt) <= min(out[i] - lo, hi - out[i]) + 0.002:
            continue
        moved.append(
            f"{axis}{(tgt - out[i]) * 1000:+.0f}mm "
            f"(room {neg * 1000:.0f}/{pos * 1000:.0f}mm vs half {half * 1000:.0f}mm)"
        )
        out[i] = tgt
    if win:
        print(
            f"[place-recentre] {label}: "
            + (
                "; ".join(moved)
                if moved
                else "no move (room "
                + " ".join(
                    f"{a}={w[0] * 1000:.0f}/{w[1] * 1000:.0f}mm" for a, w in win.items()
                )
                + f" vs half {obj_half_xy[0] * 1000:.0f}/{obj_half_xy[1] * 1000:.0f}mm)"
            )
        )
    return (out[0], out[1])


def _space_usable(s, spaces, container_top_z, obj_h, probe):
    """None if the held object can be LEFT on space ``s``, else a short reason string.

    Three ways the answer is no, and only the first is geometry the authoring step could have known:
    TALL   the object is taller than the gap above the board;
    no-support  nothing real is under the space's drop point (an authored space over a hole);
    blocked     the volume the object would occupy is already OCCUPIED, so the release interpenetrates and
                PhysX ejects it sideways -- which looks identical to a fall in the settle log but is a
                different failure with a different fix.
    None of the three is a placement waiting for a better stand -- no amount of re-navigation fixes them --
    so they gate the CHOICE, which is what lets the one loop that re-chooses on unreachability re-choose on
    these too. Returns None whenever it cannot tell, so an unmeasurable object is never rejected."""
    if obj_h is None:
        return None
    _fc = _fit_ceiling_for(s, spaces, container_top_z)
    if math.isfinite(_fc) and float(s[1]) + obj_h > float(_fc) + 1e-3:
        return f"TALL(gap={(float(_fc) - float(s[1])) * 1000:.0f}mm)"
    # ANY point, not just the centre. A space is only unusable when the WHOLE of it is -- an open cabinet
    # door sweeps across part of its own interior, so a centre-only test passes the space, the 49-candidate
    # search inside it then finds nothing, and the drop falls through to the unchecked fallback with no
    # chance to go back and pick a different shelf (measured: bt_chemlab put a bottle inside the oxidizers
    # cabinet's own open door leaf and PhysX threw it 1.6 m). Probing the centre plus the four inset corners
    # keeps a partly-blocked shelf in play -- the xy search will find its usable side -- while rejecting one
    # that is blocked through and through.
    _worst = None
    _failed_probes = []
    for _fx, _fy in ((0.0, 0.0), (-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)):
        _pr = probe(
            float(s[3]) + _fx * float(s[5]),
            float(s[4]) + _fy * float(s[6]),
            float(s[1]),
        )
        if _pr is None or _pr["ok"]:
            return None  # unknown, or at least one usable point -> keep the space
        _failed_probes.append(_pr)
        if _worst is None:
            _worst = _pr
    if _worst is None:
        return None
    if all(
        probe_result.get("reason") == "blocked"
        and probe_result.get("blockers")
        and all(str(path).startswith("/World/Obj") for path in probe_result["blockers"])
        for probe_result in _failed_probes
    ):
        return None
    if _worst["reason"] == "no-support":
        _hz = _worst["hit_z"]
        return "no-support" + (f"(hit_z={_hz:.2f})" if _hz is not None else "(nothing)")
    return f"blocked({[b.rsplit('/', 1)[-1] for b in _worst['blockers']]})"


def _obj_extent(scene, obj_key):
    """``(half_x, half_y, height)`` of ``obj_key`` from its LIVE world bbox.

    Falls back to ``(0.05, 0.05, None)`` when the object has no prim or the bbox cannot be read -- a
    height of None means "unmeasured", which every caller treats as "do not judge", never as zero.
    (BEHAVIOR graspables do not register ``Entity.half``, so reading that would give a misleading default
    and collapse the drop-inset region -> objects pile at the centre instead of spreading.)"""
    preview_extents = getattr(scene, "_content_preview_object_extents", None) or {}
    if obj_key in preview_extents:
        return tuple(float(value) for value in preview_extents[obj_key])
    _e = scene.entities.get(obj_key) if obj_key else None
    _path = getattr(_e, "prim_path", None) if _e is not None else None
    if not _path or getattr(scene, "stage", None) is None:
        return 0.05, 0.05, None
    try:
        import fpv_scene as _fs_ext

        _bb = _fs_ext.compute_world_bbox(scene.stage.GetPrimAtPath(_path))
        _mn, _mx = _bb.GetMin(), _bb.GetMax()
        return (
            max(0.02, 0.5 * float(_mx[0] - _mn[0])),
            max(0.02, 0.5 * float(_mx[1] - _mn[1])),
            float(_mx[2] - _mn[2]),
        )
    except Exception:  # noqa: BLE001
        return 0.05, 0.05, None


def _live_footprints(scene, placed):
    """Where the already-dropped objects REALLY are now: ``[(cx, cy, hx, hy), ...]``.

    ``placed`` records where each drop was AIMED; the per-command settle then slides the object to its true
    rest (measured 111 mm in bt_chemlab), so de-conflicting against the aim point dodges a ghost and puts
    the next drop on the real body. The stacking lift downstream already reads LIVE world bboxes, so using
    aim points here would also make the two halves of one feature disagree about where things are.
    Entries with no readable prim keep their recorded aim, so this can only ever be at least as accurate."""
    out = []
    for _q in placed:
        _path = _q[4] if len(_q) > 4 else None
        _live = None
        if _path and getattr(scene, "stage", None) is not None:
            try:
                import fpv_scene as _fs_ov

                _prim = scene.stage.GetPrimAtPath(_path)
                if _prim is not None and _prim.IsValid():
                    _b = _fs_ov.compute_world_bbox(_prim)
                    _mn, _mx = _b.GetMin(), _b.GetMax()
                    _live = (
                        0.5 * float(_mn[0] + _mx[0]),
                        0.5 * float(_mn[1] + _mx[1]),
                        max(0.02, 0.5 * float(_mx[0] - _mn[0])),
                        max(0.02, 0.5 * float(_mx[1] - _mn[1])),
                    )
            except Exception:  # noqa: BLE001
                _live = None
        out.append(
            _live
            if _live is not None
            else (float(_q[0]), float(_q[1]), float(_q[2]), float(_q[3]))
        )
    return out


def _target_relation_footprints(
    scene, target_key, exclude_key, relation_name, z_range=None
):
    """Live XY footprints already ON/INSIDE ``target_key``, including preloaded objects."""
    if relation_name is None or getattr(scene, "stage", None) is None:
        return []
    try:
        import fpv_scene as _fs_rel

        relations = scene.relations(source="physics")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for pred, subject, target in relations:
        if (
            getattr(pred, "name", str(pred)).upper() != relation_name
            or str(target) != str(target_key)
            or str(subject) == str(exclude_key)
        ):
            continue
        entity = scene.entities.get(str(subject))
        path = getattr(entity, "prim_path", None) if entity is not None else None
        if not path:
            continue
        try:
            bbox = _fs_rel.compute_world_bbox(scene.stage.GetPrimAtPath(path))
            mn, mx = bbox.GetMin(), bbox.GetMax()
        except Exception:  # noqa: BLE001
            continue
        if z_range is not None:
            center_z = 0.5 * float(mn[2] + mx[2])
            if (
                center_z < float(z_range[0]) - 1e-3
                or center_z >= float(z_range[1]) - 1e-3
            ):
                continue
        out.append(
            (
                0.5 * float(mn[0] + mx[0]),
                0.5 * float(mn[1] + mx[1]),
                max(0.02, 0.5 * float(mx[0] - mn[0])),
                max(0.02, 0.5 * float(mx[1] - mn[1])),
                str(path),
            )
        )
    return out


def _occupied_footprints(
    scene, target_key, placed, exclude_key, relation_name=None, z_range=None
):
    """Tracked drops plus live relation occupants, de-duplicated by prim path."""
    tracked = _live_footprints(scene, placed)
    seen = {str(row[4]) for row in placed if len(row) > 4 and row[4]}
    for cx, cy, hx, hy, path in _target_relation_footprints(
        scene, target_key, exclude_key, relation_name, z_range
    ):
        if path in seen:
            continue
        tracked.append((cx, cy, hx, hy))
        seen.add(path)
    return tracked


def _placement_candidates(cx, cy, rx, ry, ohx, ohy, occupied, min_gap=0.0):
    """Finite candidate set complete for axis-aligned rectangle contacts and target edges."""
    return placement_packing.placement_candidates(
        (cx, cy),
        (rx, ry),
        (ohx, ohy),
        occupied,
        min_gap=min_gap,
    )


def _first_strict_drop_xy(cands, overlaps, site, reach, embedded=None):
    """Choose a supported, non-overlapping site; never degrade to piling."""
    clear = [point for point in cands if not overlaps(point)]
    if not clear:
        return None, 1
    supported = [point for point in clear if site(point)]
    if not supported:
        return None, 2
    if embedded is not None:
        supported = [point for point in supported if not embedded(point)]
        if not supported:
            return None, 3
    for point in supported:
        if reach(point):
            return point, 0
    return supported[0], 4


def _first_stackable_drop_xy(
    cands,
    overlaps,
    site,
    reach,
    clearance=None,
    embedded=None,
    defer_pile_reach=False,
):
    """Prefer a direct supported rest, then a reachable rest on an existing occupant.

    Unlike the legacy non-strict ladder, this never selects unsupported empty space. It
    only relaxes the overlap veto when the overlap itself supplies a physical support.
    A caller may defer pile reachability because the final target height is derived from
    the live occupant only after XY selection.
    """
    direct = [point for point in cands if not overlaps(point) and site(point)]
    if embedded is not None:
        clean = [point for point in direct if not embedded(point)]
        if clean:
            direct = clean
    for point in direct:
        if reach(point):
            return point, 0
    piled = [point for point in cands if overlaps(point)]
    if not piled:
        if direct:
            return direct[0], 4
        return None, 2 if any(not overlaps(point) for point in cands) else 1
    reachable_piled = [point for point in piled if reach(point)]
    if reachable_piled:
        if clearance is None:
            return reachable_piled[0], 1
        return max(reachable_piled, key=clearance), 1
    if defer_pile_reach:
        return piled[0] if clearance is None else min(piled, key=clearance), 1
    if direct:
        return direct[0], 4
    return None, 4


def _first_drop_xy(
    cands, overlaps, site, reach, fallback, clearance=None, embedded=None
):
    """``(xy, tier)`` -- the first candidate passing the strictest tier still satisfiable.

    tier 0  clear of other objects AND on real geometry AND reachable   -- a clean drop
    tier 1  on real geometry and reachable, but it PILES on something already there
    tier 2  clear of other objects and reachable, but NOT on real geometry
    tier 3  reachable only -- both compromises at once
    tier 4  nothing reachable; the centre, honest OUT_OF_REACH downstream

    Tiers 2-4 ignore the live-physics site test, so when nothing passes it the choice is exactly the
    probe-free one -- the probe may only move the pick to a better spot, never veto the placement.
    Returning the tier lets the CALLER see that it had to compromise, which is the difference between
    "this shelf is fine" and "this shelf is the least bad of the ones I have tried": only the caller knows
    whether another shelf is still untried, so only the caller can decide to go back and take one.

    SPREAD ON THE PILING TIERS (1 and 3). Those two are reached only when EVERY candidate overlaps an
    object already placed here, so the drop is going to land on top of something whatever we do -- but
    "on the far edge of the last jar" and "dead centre on the last jar" are very different outcomes, and
    taking the first candidate in the caller's order systematically picks the same xy every time (the
    deepest-first order is a fixed sort, so N objects into one container all aim at one point). With
    ``clearance`` supplied, these two tiers take the passing candidate that is FARTHEST from the live
    footprints instead of the first one. This is a change of CHOICE inside an unchanged candidate set:
    the tier a placement lands in never moves, tiers 0/2 are unaffected (they are already
    non-overlapping, so there is nothing to spread from), and with one qualifying candidate -- or no
    ``clearance`` -- the answer is the first passing candidate."""
    _spread = {1, 3} if clearance is not None else ()
    for _tier, _pred in enumerate(
        (
            lambda p: not overlaps(p) and site(p) and reach(p),
            lambda p: site(p) and reach(p),
            lambda p: not overlaps(p) and reach(p),
            reach,
        )
    ):
        if _tier in _spread:
            _ok = [p for p in cands if _pred(p)]
            if _ok:
                # Prefer a candidate that is not EMBEDDED in solid geometry, then the least-piled one.
                # ``embedded`` is a preference, never a filter: if every candidate is embedded the key is
                # constant and this is exactly the clearance-only answer.
                return max(
                    _ok,
                    key=(
                        clearance
                        if embedded is None
                        else (lambda p: (not embedded(p), clearance(p)))
                    ),
                ), _tier
            continue
        # EMBEDDED-AWARE FIRST PASS (tiers 0 and 2). ``probe_drop_site`` cannot tell an embedded drop
        # point from a perfectly seated one -- ``lift == z_tol``, so a ray starting inside a solid reports
        # a hit at its own origin and passes ``abs(hit_z - z) <= z_tol`` by exactly the tolerance. The
        # point is then released INSIDE the furniture and PhysX throws it out. Rather than reject it --
        # which would be a veto that can fail otherwise-valid placements -- take the first candidate
        # that is BOTH acceptable and not embedded, and if there is none, fall through to the first
        # acceptable candidate, exactly as with no embedded test. A candidate whose embeddedness is
        # unknown (probe unavailable) is treated as not embedded, so an unmeasurable scene is unchanged too.
        if embedded is not None:
            for p in cands:
                if _pred(p) and not embedded(p):
                    return p, _tier
        for p in cands:
            if _pred(p):
                return p, _tier
    return fallback, 4


def _selftest_drop_xy() -> None:
    """Pure-logic checks for the drop-XY chooser (no Isaac, no scene). Run with::

        python -c "import semantic_executor as s; s._selftest_drop_xy()"

    The property under test is the rule every filter here must obey: **in the worst case the result is
    identical to supplying no refinement at all**. Both refinements -- the piling-tier spread and the
    embedded-point re-choose -- are PREFERENCES inside an unchanged candidate set, never vetoes, so the
    tier a placement lands in can never move and no placement that works without them can start failing."""
    T, F = (lambda p: True), (lambda p: False)
    no_ov = lambda p: False  # noqa: E731
    cands = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)]

    # --- the four tiers, with no refinement supplied: the base rules ---
    assert _first_drop_xy(cands, no_ov, T, T, (9, 9)) == ((0.0, 0.0), 0)
    assert _first_drop_xy(cands, no_ov, F, T, (9, 9)) == ((0.0, 0.0), 2)
    assert _first_drop_xy(cands, no_ov, T, F, (9, 9)) == ((9, 9), 4)

    # --- SPREAD on the piling tiers. Reached only when EVERY candidate overlaps something already
    # placed, so the drop lands on something whatever we do -- but not on its dead centre.
    piled = [(0.0, 0.0), (0.05, 0.0), (0.09, 0.0)]
    clr = lambda p: abs(p[0]) - 0.1  # noqa: E731  separation minus summed half-extents
    assert _first_drop_xy(piled, T, T, T, (9, 9)) == ((0.0, 0.0), 1)  # no refinement: the first
    assert _first_drop_xy(piled, T, T, T, (9, 9), clearance=clr) == (
        (0.09, 0.0),
        1,
    )  # spread: the least-piled
    assert (
        _first_drop_xy(piled, T, F, T, (9, 9), clearance=clr)[1] == 3
    )  # same on tier 3
    # A first drop (nothing placed) must not prefer a corner: constant key -> the caller's own order.
    assert _first_drop_xy(piled, T, T, T, (9, 9), clearance=lambda p: float("inf")) == (
        (0.0, 0.0),
        1,
    )

    # --- EMBEDDED re-choose. probe_drop_site cannot tell an embedded drop point from a seated one
    # (lift == z_tol), so a point INSIDE the furniture passes the support test and the body is released
    # inside solid geometry. Prefer a candidate that is not embedded -- and fall through when none is.
    emb_first = lambda p: p[0] == 0.0  # noqa: E731
    emb_all = lambda p: True  # noqa: E731
    assert _first_drop_xy(cands, no_ov, T, T, (9, 9), embedded=emb_first) == (
        (0.1, 0.0),
        0,
    )  # re-chose, SAME tier
    # THE GUARANTEE: every candidate embedded -> identical to supplying no argument at all.
    for pred, site in ((emb_all, T), (emb_all, F)):
        assert _first_drop_xy(
            cands, no_ov, site, T, (9, 9), embedded=pred
        ) == _first_drop_xy(cands, no_ov, site, T, (9, 9))
    # ...and the two refinements compose: prefer non-embedded FIRST, then least-piled.
    assert _first_drop_xy(
        piled, T, T, T, (9, 9), clearance=clr, embedded=lambda p: p[0] == 0.09
    ) == ((0.05, 0.0), 1)
    assert _first_drop_xy(
        piled, T, T, T, (9, 9), clearance=clr, embedded=emb_all
    ) == _first_drop_xy(piled, T, T, T, (9, 9), clearance=clr)
    # Unreachable stays unreachable: a refinement may never rescue or condemn a candidate.
    assert _first_drop_xy(cands, no_ov, T, F, (9, 9), embedded=emb_first) == ((9, 9), 4)

    # Strict placement never asks physics to resolve an overlap or unsupported release.
    assert _first_strict_drop_xy(cands, T, T, T)[0] is None
    assert _first_strict_drop_xy(cands, no_ov, F, T)[0] is None
    assert _first_strict_drop_xy(cands, no_ov, T, F) == ((0.0, 0.0), 4)

    # Twelve same-size objects into one open target get twelve distinct, non-overlapping
    # sites. Candidate coordinates include occupant edges, so this is not tied to a fixed 7x7 grid.
    occupied = []
    for _ in range(12):
        options = _placement_candidates(
            0.0,
            0.0,
            0.55,
            0.35,
            0.05,
            0.05,
            occupied,
            min_gap=PLACEMENT_MIN_GAP_M,
        )
        options.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])

        def _hits(point):
            return any(
                abs(point[0] - qx) < 0.05 + qhx + PLACEMENT_MIN_GAP_M
                and abs(point[1] - qy) < 0.05 + qhy + PLACEMENT_MIN_GAP_M
                for qx, qy, qhx, qhy in occupied
            )

        chosen, tier = _first_strict_drop_xy(options, _hits, T, T)
        assert tier == 0 and chosen is not None
        occupied.append((chosen[0], chosen[1], 0.05, 0.05))
    assert len({(round(x, 6), round(y, 6)) for x, y, _, _ in occupied}) == 12
    for _i, _left in enumerate(occupied):
        for _right in occupied[_i + 1 :]:
            assert (
                abs(_left[0] - _right[0]) >= 0.12 - 1e-9
                or abs(_left[1] - _right[1]) >= 0.12 - 1e-9
            )

    # Overflow reuses a fitting shelf; it must never prefer an unused shelf that is physically too short.
    spaces = [("short",), ("fit_a",), ("fit_b",)]
    assert _narrow_pool(
        spaces,
        assigned=["fit_a", "fit_b"],
        rejected=set(),
        fits=lambda space: space[0] != "short",
    ) == [("fit_a",), ("fit_b",)]
    assert not _narrow_pool(spaces, [], set(), fits=lambda _space: False)

    print("[semantic_executor] _selftest_drop_xy PASS")


def _placement_target_half(surf_e0, half_override):
    half = (
        half_override
        if half_override is not None
        else (surf_e0.half if surf_e0.half is not None else (0.0, 0.0))
    )
    return float(half[0]), float(half[1])


def _stable_placement_overhang(half_extent: float) -> float:
    return placement_packing.stable_overhang(half_extent)


def _placement_fit_yaw(
    obj_half, target_half, *, allow_quarter_turn: bool
) -> float | None:
    return placement_packing.fit_yaw(
        obj_half,
        target_half,
        allow_quarter_turn=allow_quarter_turn,
    )


def _validate_placement_fit(
    obj_key,
    obj_half,
    obj_height,
    target_key,
    target_half,
    raise_too_small,
    *,
    allow_quarter_turn=False,
):
    ohx, ohy = obj_half
    hx, hy = target_half
    fit_yaw = (
        0.0
        if obj_height is None
        else _placement_fit_yaw(
            obj_half, target_half, allow_quarter_turn=allow_quarter_turn
        )
    )
    if fit_yaw is not None:
        return True
    if not raise_too_small:
        return False
    raise PrimitiveFailure(
        PrimitiveError.TARGET_TOO_SMALL,
        f"{obj_key} footprint {2 * ohx:.3f}x{2 * ohy:.3f}m exceeds "
        f"{target_key} footprint {2 * hx:.3f}x{2 * hy:.3f}m",
    )


def _placement_pool(scene, pool_key):
    pools = getattr(scene, "_into_placed", None)
    if pools is None:
        pools = scene._into_placed = {}
    return pools.setdefault(pool_key, [])


def _candidate_set(
    center,
    target_half,
    obj_half,
    occupied,
    rejected,
    edge_margin=PLACEMENT_EDGE_MARGIN_M,
    min_gap=PLACEMENT_MIN_GAP_M,
    preferred=(),
    pile_anchors=(),
):
    return placement_packing.candidate_set(
        center,
        target_half,
        obj_half,
        occupied,
        rejected,
        edge_margin=edge_margin,
        min_gap=min_gap,
        preferred=preferred,
        pile_anchors=pile_anchors,
    )


def _verified_surface_placement_sites(surface, object_key):
    meta = getattr(surface, "meta", None) or {}
    expected_path = str(meta.get("verified_placement_path") or "")
    actual_path = str(getattr(surface, "prim_path", None) or "")
    if expected_path and actual_path and expected_path != actual_path:
        return ()
    rows = (meta.get("verified_placement_sites") or {}).get(str(object_key), ())
    return tuple(row for row in rows if isinstance(row, dict) and row.get("xy"))


def _promote_verified_placement_sites(candidates, preferred):
    order = {
        (round(float(point[0]), 4), round(float(point[1]), 4)): index
        for index, point in enumerate(preferred)
    }
    if not order:
        return candidates
    candidates.sort(
        key=lambda point: (
            0,
            order[(round(float(point[0]), 4), round(float(point[1]), 4))],
        )
        if (round(float(point[0]), 4), round(float(point[1]), 4)) in order
        else (1, 0)
    )
    return candidates


def _surface_probe_profile(
    surface, object_key, point, object_half, fallback_z, placement_meta
):
    quarter_turn = abs(float(placement_meta.get("_placement_fit_yaw_rad") or 0.0)) > 1e-9
    probe_half = (
        (float(object_half[1]), float(object_half[0]))
        if quarter_turn
        else (float(object_half[0]), float(object_half[1]))
    )
    for row in _verified_surface_placement_sites(surface, object_key):
        if bool(row.get("quarter_turn")) != quarter_turn:
            continue
        row_xy = row["xy"]
        if (
            abs(float(row_xy[0]) - float(point[0])) > 1e-4
            or abs(float(row_xy[1]) - float(point[1])) > 1e-4
        ):
            continue
        try:
            return probe_half, float(row.get("support_z", fallback_z))
        except (TypeError, ValueError):
            break
    return probe_half, float(fallback_z)


def _physics_path_belongs_to_entity(path, entity) -> bool:
    candidate = str(path or "").rstrip("/")
    owner = str(getattr(entity, "prim_path", None) or "").rstrip("/")
    if not candidate or not owner:
        return False
    if (
        candidate == owner
        or candidate.startswith(owner + "/")
        or owner.startswith(candidate + "/")
    ):
        return True
    return candidate.split("/collisions/", 1)[0] == owner.split("/collisions/", 1)[0]


def _surface_site_allows_lid_bridge(
    scene,
    held_key,
    target_key,
    surface_view,
    container_view,
    probe,
) -> bool:
    """Allow a fitted lid to span a portable container's hollow opening.

    A normal surface placement still requires ``probe['ok']``. The only exception is
    a held lid placed on a graspable target that exposes both SURFACE and CONTAINER
    affordances. Its center ray may legitimately pass through the opening and hit the
    container bottom or an item inside it. Clearance and embedded-geometry failures,
    and rays that hit unrelated geometry, remain hard rejections.
    """
    if probe is None or bool(probe.get("ok")):
        return True
    held = scene.entities.get(str(held_key))
    target = scene.entities.get(str(target_key))
    held_category = str(
        (getattr(held, "meta", None) or {}).get("asset_category") or ""
    ).casefold()
    if (
        held_category != "lid"
        or target is None
        or target.role is not Role.GRASPABLE
        or surface_view is None
        or surface_view.role is not Role.SURFACE
        or container_view is None
        or container_view.role is not Role.CONTAINER
        or probe.get("support") is not False
        or probe.get("clear") is False
        or probe.get("embedded") is True
    ):
        return False
    support_path = str(probe.get("support_path") or "")
    if not support_path:
        return True
    allowed = [target]
    for child_key, parent_key in (getattr(scene, "inside", None) or {}).items():
        if str(parent_key) != str(target_key):
            continue
        child = scene.entities.get(str(child_key))
        if child is not None:
            allowed.append(child)
    return any(
        _physics_path_belongs_to_entity(support_path, entity) for entity in allowed
    )


def _order_placement_candidates(
    candidates,
    center,
    radius,
    root_xy,
    deepest,
    nearest=False,
    reserve_space=False,
):
    placement_packing.order_candidates(
        candidates,
        center,
        radius,
        root_xy,
        deepest,
        nearest,
        reserve_space=reserve_space,
    )


def _memoized_point_test(test, default):
    cache = {}

    def _check(point):
        if test is None:
            return default
        key = (round(point[0], 4), round(point[1], 4))
        if key not in cache:
            cache[key] = bool(test(point))
        return cache[key]

    return _check


def _candidate_clearance(point, obj_half, occupied, min_gap):
    if not occupied:
        return float("inf")
    ohx, ohy = obj_half
    return min(
        max(
            abs(point[0] - qx) - (ohx + qhx + min_gap),
            abs(point[1] - qy) - (ohy + qhy + min_gap),
        )
        for qx, qy, qhx, qhy in occupied
    )


def _interleave_placement_candidate_tiers(primary, secondary):
    """Keep both placement tiers inside a bounded breadth-first reach batch."""
    candidates = []
    for index in range(max(len(primary), len(secondary))):
        candidates.extend(primary[index : index + 1])
        candidates.extend(secondary[index : index + 1])
    return candidates


def _choose_placement_candidate(
    candidates,
    center,
    obj_half,
    occupied,
    reach_ok,
    site_ok,
    embedded_ok,
    strict,
    allow_stacking,
    min_gap,
    reach_many=None,
):
    ohx, ohy = obj_half

    def _overlaps(point):
        return any(
            abs(point[0] - qx) < (ohx + qhx + min_gap)
            and abs(point[1] - qy) < (ohy + qhy + min_gap)
            for qx, qy, qhx, qhy in occupied
        )

    site = _memoized_point_test(site_ok, True)
    embedded = None
    if embedded_ok is not None:
        embedded = _memoized_point_test(embedded_ok, False)

    if reach_many is None:
        reach = _memoized_point_test(reach_ok, False)
    else:
        if allow_stacking:
            direct = [
                point for point in candidates if not _overlaps(point) and site(point)
            ]
            if embedded is not None:
                clean = [point for point in direct if not embedded(point)]
                if clean:
                    direct = clean
            piled = [point for point in candidates if _overlaps(point)]
            reach_candidates = _interleave_placement_candidate_tiers(direct, piled)
        elif strict:
            reach_candidates = [
                point
                for point in candidates
                if not _overlaps(point)
                and site(point)
                and (embedded is None or not embedded(point))
            ]
        else:
            reach_candidates = candidates
        reachable = reach_many(reach_candidates)
        reach_map = {
            (round(point[0], 4), round(point[1], 4)): bool(value)
            for point, value in zip(reach_candidates, reachable)
        }

        def reach(point):
            return reach_map.get((round(point[0], 4), round(point[1], 4)), False)

    def clearance(point):
        return _candidate_clearance(point, obj_half, occupied, min_gap)

    if allow_stacking:
        return _first_stackable_drop_xy(
            candidates,
            _overlaps,
            site,
            reach,
            clearance=clearance,
            embedded=embedded,
            defer_pile_reach=True,
        )
    if strict:
        return _first_strict_drop_xy(candidates, _overlaps, site, reach, embedded)
    return _first_drop_xy(
        candidates,
        _overlaps,
        site,
        reach,
        center,
        clearance=clearance,
        embedded=embedded,
    )


def _goal_load_count(scene, relation_name, target_key) -> int:
    relation = str(relation_name or "").lower()
    target = str(target_key)
    direct = (
        sum(
            1
            for predicate, _subject, goal_target in getattr(
                scene, "goal_predicates", set()
            )
            if str(predicate).lower() == relation and str(goal_target) == target
        )
        if bool(getattr(scene, "goal_predicates_configured", False))
        else 0
    )
    aggregate = 0
    if relation == "inside":
        behavior_task = getattr(scene, "behavior_task", None) or {}
        aggregate = int(
            (behavior_task.get("aggregate_target_loads") or {}).get(target, 0)
        )
    return max(direct, aggregate)


def _select_container_xy(
    scene,
    surf_key,
    surf_e0,
    center,
    st,
    reach_ok,
    half_override=None,
    deepest=False,
    site_ok=None,
    embedded_ok=None,
    rejected=(),
    occupancy_key=None,
    occupancy_relation=None,
    occupancy_z=None,
    commit=True,
    strict=True,
    allow_stacking=False,
    raise_too_small=True,
    nearest=False,
    placement_meta=None,
    reach_many=None,
):
    """Choose a size-aware drop XY shared by containers, open bins, trays, and surfaces.

    Automatic placement is strict by default: the held footprint must fit wholly inside the target, avoid
    every tracked or physically observed occupant, and sit on real support when a site probe is available.
    An unreachable but otherwise valid site is returned as tier 4 so the existing re-navigation path can
    retry it from a better stand. Tiers 1-3 return ``None`` instead of asking PhysX to resolve overlap."""
    center = float(center[0]), float(center[1])
    target_half = _placement_target_half(surf_e0, half_override)
    obj_key = st.held["obj_key"] if st.held else None
    ohx, ohy, obj_height = _obj_extent(scene, obj_key)
    obj_half = ohx, ohy
    can_rotate = bool(
        placement_meta is not None
        and st.held
        and st.held.get("source_rest_pose") is not None
        and not st.held.get("carried")
    )
    verified_sites = _verified_surface_placement_sites(surf_e0, obj_key)
    hinted_yaw = None
    for row in verified_sites:
        quarter_turn = bool(row.get("quarter_turn"))
        candidate_half = (ohy, ohx) if quarter_turn else (ohx, ohy)
        if quarter_turn and not can_rotate:
            continue
        if _placement_fit_yaw(
            candidate_half, target_half, allow_quarter_turn=False
        ) is not None:
            hinted_yaw = math.pi / 2.0 if quarter_turn else 0.0
            break
    fit_yaw = hinted_yaw
    if fit_yaw is None:
        fit_yaw = (
            0.0
            if obj_height is None
            else _placement_fit_yaw(
                obj_half, target_half, allow_quarter_turn=can_rotate
            )
        )
    if fit_yaw is None:
        if not raise_too_small:
            return None, 5
        raise PrimitiveFailure(
            PrimitiveError.TARGET_TOO_SMALL,
            f"{obj_key} footprint {2 * ohx:.3f}x{2 * ohy:.3f}m exceeds "
            f"{surf_key} footprint {2 * target_half[0]:.3f}x{2 * target_half[1]:.3f}m",
        )
    if placement_meta is not None:
        placement_meta["_placement_fit_yaw_rad"] = fit_yaw
    if abs(fit_yaw) > 1e-9:
        obj_half = ohy, ohx
        ohx, ohy = obj_half
    preferred_points = tuple(
        tuple(row["xy"])
        for row in verified_sites
        if bool(row.get("quarter_turn")) == (abs(fit_yaw) > 1e-9)
    )
    if not _validate_placement_fit(
        obj_key, obj_half, obj_height, surf_key, target_half, raise_too_small
    ):
        return None, 5
    obj_path = (
        getattr(scene.entities.get(obj_key), "prim_path", None) if obj_key else None
    )
    pool_key = surf_key if occupancy_key is None else occupancy_key
    placed = _placement_pool(scene, pool_key)
    # Entries are (aim_x, aim_y, hx, hy, prim_path); snapshot the LIVE footprints ONCE (not per candidate
    # -- the overlap test runs 49x) so this and the stacking lift downstream read the same world.
    occupied = _occupied_footprints(
        scene,
        surf_key,
        placed,
        obj_key,
        relation_name=occupancy_relation,
        z_range=occupancy_z,
    )
    chosen = None
    tier = 1
    used_margin = PLACEMENT_EDGE_MARGIN_M
    used_gap = PLACEMENT_MIN_GAP_M
    reserve_space = (
        _goal_load_count(scene, occupancy_relation, surf_key) > len(occupied) + 1
    )
    margins = []
    if rejected:
        margins.append(PLACEMENT_EDGE_MARGIN_M + max(ohx, ohy))
    if reserve_space:
        margins.append(0.0)
    elif occupied:
        margins.extend((PLACEMENT_EDGE_MARGIN_M, 0.01, 0.004, 0.0))
    else:
        margins.append(PLACEMENT_EDGE_MARGIN_M)
    gaps = (
        (PLACEMENT_MIN_GAP_M, 0.01, 0.004, 0.0) if occupied else (PLACEMENT_MIN_GAP_M,)
    )
    for min_gap in gaps:
        for edge_margin in margins:
            candidates, rx, ry = _candidate_set(
                center,
                target_half,
                obj_half,
                occupied,
                rejected,
                edge_margin=edge_margin,
                min_gap=min_gap,
                preferred=preferred_points,
                pile_anchors=occupied if allow_stacking else (),
            )
            _order_placement_candidates(
                candidates,
                center,
                (rx, ry),
                st.root_xy,
                deepest,
                nearest,
                reserve_space=reserve_space,
            )
            _promote_verified_placement_sites(candidates, preferred_points)
            chosen, tier = _choose_placement_candidate(
                candidates,
                center,
                obj_half,
                occupied,
                reach_ok,
                site_ok,
                embedded_ok,
                strict,
                allow_stacking,
                min_gap,
                reach_many=reach_many,
            )
            used_margin = edge_margin
            used_gap = min_gap
            if chosen is not None or tier != 1:
                break
        if chosen is not None or tier != 1:
            break
    if chosen is None:
        return None, tier
    if used_margin < PLACEMENT_EDGE_MARGIN_M:
        print(
            f"[place-pack] {surf_key}: reduced edge margin to {used_margin * 1000:.0f}mm "
            f"to fit another object without overlap"
        )
    if used_gap < PLACEMENT_MIN_GAP_M:
        print(
            f"[place-pack] {surf_key}: reduced object gap to {used_gap * 1000:.0f}mm "
            f"to fit the declared load without overlap"
        )
    if commit:
        placed.append((float(chosen[0]), float(chosen[1]), ohx, ohy, obj_path))
    return (float(chosen[0]), float(chosen[1])), tier


def _strict_shelf_placement(spaces, rejected) -> bool:
    """Require an empty footprint while another interior shelf remains untried."""
    return len(rejected) + 1 < len(spaces)


def _requires_direct_goal_relation(scene, predicate, subject, target) -> bool:
    """Keep unknown/legacy scenes strict; relax only a configured auxiliary action."""
    behavior_task = getattr(scene, "behavior_task", None) or {}
    aggregate_goals = behavior_task.get("aggregate_goals") or ()
    if not bool(getattr(scene, "goal_predicates_configured", False)) and not aggregate_goals:
        return True
    relation = str(predicate).lower()
    subject_key = str(subject)
    target_key = str(target)
    if (relation, subject_key, target_key) in getattr(
        scene, "goal_predicates", set()
    ):
        return True
    return bool(
        relation == "inside"
        and any(
            str(row.get("kind") or "") == "exact_count_inside"
            and str(row.get("target") or "") == target_key
            and subject_key in {str(value) for value in row.get("candidates") or ()}
            for row in aggregate_goals
        )
    )


def mark_placement_retry(
    meta: dict, failed_xy, *, inside: bool, reject_space: bool = False
) -> int:
    """Prepare an automatic placement goal for a different geometry-backed attempt."""
    chosen_space = meta.pop("_space_chosen", None)
    if reject_space and chosen_space:
        rejected_spaces = set(meta.get("_space_rejected") or ())
        rejected_spaces.add(str(chosen_space))
        meta["_space_rejected"] = rejected_spaces
    if failed_xy is not None:
        rejected_xy = {
            (round(float(row[0]), 4), round(float(row[1]), 4))
            for row in meta.get("_placement_rejected_xy") or ()
        }
        rejected_xy.add((round(float(failed_xy[0]), 4), round(float(failed_xy[1]), 4)))
        meta["_placement_rejected_xy"] = sorted(rejected_xy)
        if inside:
            meta["_inside_rejected_xy"] = sorted(rejected_xy)
    meta.pop("_failed_drop", None)
    meta.pop("_failed_shelf_gap_ceiling", None)
    meta.pop("_inside_chosen_xy", None)
    meta.pop("_placement_chosen_xy", None)
    attempt = int(meta.get("_placement_retry_attempt", 0)) + 1
    meta["_placement_retry_attempt"] = attempt
    if inside:
        meta["_inside_retry_attempt"] = attempt
    return attempt


def placement_space_count(scene, target_key: str) -> int:
    target = (
        scene.target_view(target_key, Role.CONTAINER)
        if target_key in scene.entities
        else None
    )
    return len(getattr(target, "spaces", None) or ())


def _record_placement_footprint(scene, pool_key, xy, obj_half, obj_path) -> None:
    pools = getattr(scene, "_into_placed", None)
    if pools is None:
        pools = scene._into_placed = {}
    pools.setdefault(pool_key, []).append(
        (float(xy[0]), float(xy[1]), float(obj_half[0]), float(obj_half[1]), obj_path)
    )


def _forget_placement_footprint(scene, surf_key, xy, obj_path) -> None:
    """Remove the current failed attempt without disturbing earlier placements."""
    for key, pool in (getattr(scene, "_into_placed", None) or {}).items():
        if key != surf_key and not (
            isinstance(key, tuple) and key and key[0] == surf_key
        ):
            continue
        if not pool:
            continue
        row = pool[-1]
        if (
            (obj_path is None or (len(row) > 4 and row[4] == obj_path))
            and abs(float(row[0]) - float(xy[0])) < 1e-6
            and abs(float(row[1]) - float(xy[1])) < 1e-6
        ):
            pool.pop()
            return


def _placement_reachable(
    scene,
    rig,
    grasp_arm,
    xy,
    z,
    st,
    tol,
    shelf_gap_ceiling=None,
    ik_budget: IKSearchBudget | None = None,
    minimum_squat: float = 0.0,
    exact_rest_crouch: bool = False,
) -> bool:
    """The same arm solve used by the final PLACE gate, exposed to the XY selector."""
    enabled = bool(getattr(scene, "unified_geom", False))
    world = np.array([xy[0], xy[1], z], dtype=np.float64)
    if shelf_gap_ceiling is not None:
        plan = plan_reach_shelf(
            rig,
            grasp_arm,
            world,
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            tol,
            shelf_gap_ceiling,
            floor_z=st.floor_z,
            enabled=enabled,
            clear_fn=body_clear_fn(
                scene,
                st.root_xy,
                st.root_yaw,
                st.floor_z,
                require_body_clear=exact_rest_crouch,
            ),
            allow_colliding_fallback=False,
            ik_budget=ik_budget,
            minimum_squat=minimum_squat,
            exact_rest_crouch=exact_rest_crouch,
        )
        if plan.get("collision_blocked") or not plan["reachable"]:
            return False
        err = plan.get("err")
        if err is None:
            if ik_budget is not None and not ik_budget.consume():
                return False
            _, err = _solve_reach_arm(
                rig,
                grasp_arm,
                world,
                st.root_xy,
                st.root_yaw,
                (
                    float(plan["stand_z"])
                    if enabled or minimum_squat > 1e-6
                    else st.stand_z
                ),
                lean_override=(plan["lean_rad"] if enabled else None),
                squat_frac=(
                    float(plan["squat"])
                    if enabled or minimum_squat > 1e-6
                    else 0.0
                ),
            )
        return err <= tol
    plan = plan_reach(
        rig,
        grasp_arm,
        world,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        tol,
        floor_z=st.floor_z,
        enabled=enabled,
        clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
        allow_colliding_fallback=False,
        ik_budget=ik_budget,
        upright_first=minimum_squat > 1e-6,
        minimum_squat=minimum_squat,
    )
    if plan.get("collision_blocked"):
        return False
    err = plan.get("err")
    if err is None:
        if ik_budget is not None and not ik_budget.consume():
            return False
        _, err = _solve_reach_arm(
            rig,
            grasp_arm,
            world,
            st.root_xy,
            st.root_yaw,
            (
                float(plan["stand_z"])
                if enabled or minimum_squat > 1e-6
                else st.stand_z
            ),
            lean_override=(plan["lean_rad"] if enabled else None),
            squat_frac=(
                float(plan["squat"])
                if enabled or minimum_squat > 1e-6
                else 0.0
            ),
        )
    return err <= tol


def _placement_reachable_many(
    scene,
    rig,
    grasp_arm,
    points,
    z,
    st,
    tol,
    shelf_gap_ceiling=None,
    ik_budget: IKSearchBudget | None = None,
    minimum_squat: float = 0.0,
    exact_rest_crouch: bool = False,
) -> list[bool]:
    """Evaluate a placement candidate set in one reach batch when the target height is shared."""
    points = list(points)
    enabled = bool(getattr(scene, "unified_geom", False))
    if not points:
        return []
    if minimum_squat > 1e-6 or not enabled or not _batch_reach_supported(rig):
        return [
            _placement_reachable(
                scene,
                rig,
                grasp_arm,
                point,
                z,
                st,
                tol,
                shelf_gap_ceiling,
                ik_budget=ik_budget,
                minimum_squat=minimum_squat,
                exact_rest_crouch=exact_rest_crouch,
            )
            for point in points
        ]
    contacts = np.asarray(
        [(point[0], point[1], z) for point in points], dtype=np.float64
    )
    if shelf_gap_ceiling is None:
        plans = plan_reach_many(
            rig,
            grasp_arm,
            contacts,
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            tol,
            floor_z=st.floor_z,
            enabled=True,
            clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
            allow_colliding_fallback=False,
            ik_budget=ik_budget,
        )
    else:
        plans = plan_reach_shelf_many(
            rig,
            grasp_arm,
            contacts,
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            tol,
            shelf_gap_ceiling,
            floor_z=st.floor_z,
            enabled=True,
            clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
            allow_colliding_fallback=False,
            ik_budget=ik_budget,
            minimum_squat=minimum_squat,
        )
    return [
        bool(
            not plan.get("collision_blocked")
            and plan.get("reachable")
            and plan.get("err", float("inf")) <= tol
        )
        for plan in plans
    ]


def _prepare_placement_reach_retry(
    scene,
    goal,
    held_key,
    surf_key,
    place_xy,
    place_z,
    shelf_gap_ceiling,
) -> None:
    """Record a reachable contact so runtime can jointly re-plan stand and reach."""
    if goal.place_xy is None:
        held_entity = scene.entities.get(held_key)
        _forget_placement_footprint(
            scene,
            surf_key,
            place_xy,
            getattr(held_entity, "prim_path", None),
        )
    retry_xy = place_xy
    target = scene.entities.get(surf_key)
    access_mode = (getattr(target, "meta", None) or {}).get("access_mode")
    if (
        target is not None
        and getattr(target, "spaces", None)
        and getattr(target, "xy", None) is not None
        and access_mode not in {"front_drum", "front_opening"}
    ):
        retry_xy = target.xy
    goal.meta["_failed_drop"] = (
        float(retry_xy[0]),
        float(retry_xy[1]),
        float(place_z),
    )
    if shelf_gap_ceiling is None:
        goal.meta.pop("_failed_shelf_gap_ceiling", None)
    else:
        goal.meta["_failed_shelf_gap_ceiling"] = float(shelf_gap_ceiling)


def placement_relation_status(scene, goal) -> dict | None:
    """Read the intended post-place relation from live physics after settling."""
    intent = (goal.meta or {}).get("_placement_intent")
    if not isinstance(intent, dict):
        return None
    predicate = str(intent.get("predicate") or "").upper()
    subject = str(intent.get("subject") or "")
    target = str(intent.get("target") or "")
    if not predicate or not subject or not target:
        return None
    relations = {
        (
            getattr(pred, "name", str(pred)).upper(),
            str(left),
            "" if right is None else str(right),
        )
        for pred, left, right in scene.relations(source="physics")
    }
    ok = (predicate, subject, target) in relations
    support_mode = "direct" if ok else "missing"
    if (
        not ok
        and predicate == "ON"
        and not _requires_direct_goal_relation(scene, predicate, subject, target)
    ):
        support_edges: dict[str, set[str]] = {}
        for rel, child, parent in relations:
            if rel == "ON" and parent:
                support_edges.setdefault(child, set()).add(parent)
        frontier = [subject]
        visited = {subject}
        while frontier and not ok:
            child = frontier.pop()
            for parent in support_edges.get(child, set()):
                if parent == target:
                    ok = True
                    support_mode = "indirect"
                    break
                if parent not in visited:
                    visited.add(parent)
                    frontier.append(parent)
    result = {
        "predicate": predicate.lower(),
        "subject": subject,
        "target": target,
        "ok": bool(ok),
        "support_mode": support_mode,
        "chosen_xy": intent.get("chosen_xy"),
    }
    goal.meta["_placement_result"] = result
    return result


def _gap_ceiling_for(_s, spaces):
    """Underside z of the LOWEST board ABOVE space ``_s`` (same xy) -> the ceiling the PutInside pose crouches
    UNDER (plan_reach_shelf enters the gap horizontally). inf when nothing is above (the top space). Shared by
    the space-reachability check AND the actual place so they judge reachability IDENTICALLY. Space tuple =
    (name, top_z, bottom_z, cx, cy, hx, hy)."""
    ceil = float("inf")
    for _ob in spaces:
        if _ob is _s:
            continue
        if (
            float(_ob[2]) > float(_s[1]) + 1e-3
            and abs(float(_ob[3]) - float(_s[3])) < (float(_ob[5]) + float(_s[5]))
            and abs(float(_ob[4]) - float(_s[4])) < (float(_ob[6]) + float(_s[6]))
        ):
            ceil = min(ceil, float(_ob[2]))
    return ceil


def _fit_ceiling_for(_s, spaces, container_top_z=None):
    """Ceiling for the DOES-IT-FIT question -- deliberately NOT the same as the reach ceiling.

    ``_gap_ceiling_for`` answers "what must the arm crouch under", so it returns inf for the TOP space:
    nothing is above it to duck beneath, which is correct for the POSE. But an ENCLOSED unit's top space is
    still bounded by the unit's own lid, so inf also makes the top shelf look infinitely tall and lets an
    object be placed into a gap it cannot occupy (bt_chemlab: a 1.59 m top board under a ~1.67 m cabinet top
    is an 8 cm gap; a 23 cm bottle was placed there anyway, tipped out on release and fell to the cabinet
    floor). Falls back to the container's OWN top when there is no board above. Returns inf -- i.e. no
    constraint -- when the caller has no top to offer.

    A lidless unit must pass ``None``: its world-bbox top may be a handle or guard rail, not a ceiling."""
    _c = _gap_ceiling_for(_s, spaces)
    if math.isfinite(_c) or container_top_z is None:
        return _c
    return float(container_top_z)


def front_open_release_heights(
    rest_z: float,
    fit_ceiling: float,
    object_height: float | None,
    base_offset: float,
    handle_z: float | None,
) -> list[float]:
    """Return low-to-high, physically contained release heights above an unreachable rest."""
    if object_height is None or handle_z is None or not math.isfinite(fit_ceiling):
        return []
    top_offset = max(0.0, float(object_height) - float(base_offset))
    safe_max = min(
        float(rest_z) + 0.45,
        float(fit_ceiling) - top_offset - 0.01,
    )
    if safe_max < float(rest_z) + 0.06:
        return []
    raw = [float(rest_z) + 0.08 * step for step in range(1, 6)]
    raw.extend((float(handle_z), safe_max))
    return sorted(
        {
            round(min(safe_max, height), 6)
            for height in raw
            if float(rest_z) + 0.06 <= height <= safe_max + 1e-9
        }
    )


def _front_open_release_max_lean_rad(plan: dict) -> float:
    if bool(plan.get("exact_rest_crouch")) and float(
        plan.get("root_drop_m") or 0.0
    ) >= _EXACT_REST_ROOT_DROP_LADDER_M[-1] - 1e-6:
        return math.radians(_EXACT_REST_MAX_DROP_LEAN_LADDER_DEG[-1])
    return _LOW_FRONT_INSERT_MAX_LEAN_RAD


def _front_open_release_needed(plan: dict, *, low_front: bool) -> bool:
    if not plan.get("reachable") or plan.get("collision_blocked"):
        return True
    return bool(
        low_front
        and abs(float(plan.get("lean_rad") or 0.0))
        > _front_open_release_max_lean_rad(plan) + 1e-9
    )


def _front_open_release_plan_is_safe(
    plan: dict,
    *,
    low_front: bool,
) -> bool:
    return bool(
        plan.get("reachable")
        and not plan.get("collision_blocked")
        and (
            not low_front
            or abs(float(plan.get("lean_rad") or 0.0))
            <= _front_open_release_max_lean_rad(plan) + 1e-9
        )
    )


def _plan_front_open_release(
    rig,
    grasp_arm,
    release_world,
    st,
    reach_tolerance: float,
    reach_ceiling: float,
    *,
    low_front: bool,
    enabled: bool,
    clear_fn,
    allow_colliding_fallback: bool,
    minimum_squat: float,
) -> dict:
    """Plan a contained release without forcing the shoulder below a low cabinet shelf."""
    common = {
        "floor_z": st.floor_z,
        "enabled": enabled,
        "clear_fn": clear_fn,
        "allow_colliding_fallback": allow_colliding_fallback,
        "minimum_squat": minimum_squat,
    }
    if low_front:
        return plan_reach(
            rig,
            grasp_arm,
            release_world,
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            reach_tolerance,
            upright_crouch_first=True,
            **common,
        )
    return plan_reach_shelf(
        rig,
        grasp_arm,
        release_world,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        reach_tolerance,
        reach_ceiling,
        **common,
    )


def shelf_reach_err(
    rig,
    arm,
    s,
    spaces,
    stand_xy,
    yaw,
    stand_z,
    place_z,
    obj_half,
    tol,
    floor_z=0.0,
    enabled=True,
    clear_fn=None,
    allow_colliding_fallback=True,
    ik_budget: IKSearchBudget | None = None,
    minimum_squat: float = 0.0,
    exact_rest_crouch: bool = False,
):
    """Reach ERROR (m) to interior space ``s``'s inset-nearest point from ``stand_xy`` -- the EXACT model the
    shelved place executes: plan_reach_shelf crouched under ``s``'s OWN gap ceiling (_gap_ceiling_for) at
    ``place_z``, solved with that pose's stand_z/lean/squat. Probes the space footprint shrunk by ``obj_half``
    =(ohx,ohy) at 0.85 of the inset (the drop grid's outermost candidate), clamped to the point nearest the
    stand (easiest reach). SHARED by the drop-point selector (produce_place) AND the stand-picker
    (init_validation) so "a stand is judged reachable" == "the place can actually put a point there".
    Returns (err, contact_world_xyz, plan_dict). s=(name,top,bot,cx,cy,hx,hy)."""
    _gc = _gap_ceiling_for(s, spaces)
    _rx = 0.85 * max(0.0, float(s[5]) - float(obj_half[0]))
    _ry = 0.85 * max(0.0, float(s[6]) - float(obj_half[1]))
    _nx = min(max(float(stand_xy[0]), float(s[3]) - _rx), float(s[3]) + _rx)
    _ny = min(max(float(stand_xy[1]), float(s[4]) - _ry), float(s[4]) + _ry)
    _ow = np.array([_nx, _ny, float(place_z)], dtype=np.float64)
    _pl = plan_reach_shelf(
        rig,
        arm,
        _ow,
        stand_xy,
        yaw,
        stand_z,
        tol,
        _gc,
        floor_z=floor_z,
        enabled=enabled,
        clear_fn=clear_fn,
        allow_colliding_fallback=allow_colliding_fallback,
        ik_budget=ik_budget,
        minimum_squat=minimum_squat,
        exact_rest_crouch=exact_rest_crouch,
    )
    _e = _pl.get("err")
    if _e is None:
        if ik_budget is not None and not ik_budget.consume():
            return float("inf"), _ow, _pl
        _pa, _e = _solve_reach_arm(
            rig,
            arm,
            _ow,
            stand_xy,
            yaw,
            float(_pl["stand_z"])
            if enabled or minimum_squat > 1e-6
            else stand_z,
            lean_override=(
                _pl["lean_rad"] if enabled or minimum_squat > 1e-6 else None
            ),
            squat_frac=(
                float(_pl["squat"])
                if enabled or minimum_squat > 1e-6
                else 0.0
            ),
        )
    return _e, _ow, _pl


def best_shelf_reach(
    rig,
    arm,
    spaces,
    stand_xy,
    yaw,
    stand_z,
    tol,
    base_offset=0.0,
    obj_half=(0.05, 0.05),
    floor_z=0.0,
    enabled=True,
    clear_fn=None,
    allow_colliding_fallback=True,
    ik_budget: IKSearchBudget | None = None,
    minimum_squat: float = 0.0,
    exact_rest_crouch: bool = False,
):
    """Best (MIN-error) reachable interior space from ``stand_xy`` via shelf_reach_err over every space -- the
    SAME test the shelved place uses to pick a space. Lets the STAND-picker judge a shelved container by "can I
    reach ANY space" (what the place does), NOT "can I reach one mid drop_z point" (which the place never
    targets). Returns {reachable(err<=tol), err, contact, lean_rad, squat, stand_z, space} for the min-error
    space, or None for empty ``spaces``. ``base_offset`` lifts place_z to the object's base rest; the
    stand-picker does not know the held object, so it defaults 0 -- a feasibility estimate (the place does the
    exact selection)."""
    # MIN-error over ALL spaces (NOT first-reachable-top-down): the gate must certify a stand from which the
    # EASIEST shelf is reachable -- a comfortable chest stand -- not the first top-down one, which for a DEEP
    # tall container can be a LOW shelf only a deep crouch reaches, dragging the whole placement to the floor
    # (empirically first-top-down sent the fridge drop to the bottom shelf; min-err keeps it at the chest).
    best = None
    for s in spaces or ():
        _pz = float(s[1]) + float(base_offset)
        _e, _ow, _pl = shelf_reach_err(
            rig,
            arm,
            s,
            spaces,
            stand_xy,
            yaw,
            stand_z,
            _pz,
            obj_half,
            tol,
            floor_z=floor_z,
            enabled=enabled,
            clear_fn=clear_fn,
            allow_colliding_fallback=allow_colliding_fallback,
            ik_budget=ik_budget,
            minimum_squat=minimum_squat,
            exact_rest_crouch=exact_rest_crouch,
        )
        candidate = {
            "reachable": bool(_pl.get("reachable", False) and _e <= tol),
            "err": float(_e),
            "contact": _ow,
            "lean_rad": float(_pl["lean_rad"]) if enabled else 0.0,
            "squat": float(_pl["squat"]) if enabled else 0.0,
            "stand_z": float(_pl["stand_z"]) if enabled else float(stand_z),
            "space": s,
            "collision_blocked": bool(_pl.get("collision_blocked", False)),
            "arm_pose": _pl.get("arm_pose"),
            "arm_joints": _pl.get("arm_joints"),
        }
        if best is None or (bool(candidate["reachable"]), -float(candidate["err"])) > (
            bool(best["reachable"]),
            -float(best["err"]),
        ):
            best = candidate
        if ik_budget is not None and ik_budget.exhausted:
            break
    return best


def _rotate_point_about_axis(point, origin, axis, angle: float) -> np.ndarray:
    """Rotate one world point around an arbitrary world-space axis."""
    p = np.asarray(point, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    unit = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-9:
        raise ValueError("rotation axis must be non-zero")
    unit /= norm
    radius = p - o
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    return (
        o
        + radius * cosine
        + np.cross(unit, radius) * sine
        + unit * np.dot(unit, radius) * (1.0 - cosine)
    )


def _point_segment_distance_xy(point, start, end) -> float:
    point_xy = np.asarray(point, dtype=np.float64)[:2]
    start_xy = np.asarray(start, dtype=np.float64)[:2]
    delta = np.asarray(end, dtype=np.float64)[:2] - start_xy
    length_sq = float(np.dot(delta, delta))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point_xy - start_xy))
    fraction = float(np.dot(point_xy - start_xy, delta) / length_sq)
    closest = start_xy + np.clip(fraction, 0.0, 1.0) * delta
    return float(np.linalg.norm(point_xy - closest))


def _front_access_geometry(
    entity,
    *,
    opened_content: bool = False,
    standoff_override: float | None = None,
):
    meta = getattr(entity, "meta", None) or {}
    mode = meta.get("access_mode")
    if mode not in {"front_drum", "front_opening"}:
        return None
    center = np.asarray(meta.get("aperture_center"), dtype=np.float64)
    entry_axis = np.asarray(meta.get("entry_axis"), dtype=np.float64).reshape(-1)
    if center.shape != (3,) or entry_axis.shape != (2,):
        return None
    norm = float(np.linalg.norm(entry_axis))
    if norm <= 1e-9:
        return None
    entry_axis /= norm
    if standoff_override is not None:
        standoff = float(standoff_override)
    else:
        configured = meta.get(
            "content_standoff_m" if opened_content else "action_standoff_m"
        )
        if configured is not None:
            standoff = float(configured)
        elif mode == "front_drum":
            standoff = (
                _FRONT_DRUM_CONTENT_STANDOFF_M
                if opened_content
                else _FRONT_APPLIANCE_STANDOFF_M
            )
        else:
            standoff = (
                _FRONT_OPENING_CONTENT_STANDOFF_M
                if opened_content
                else _FRONT_OPENING_STANDOFF_M
            )
    half_width = float(meta.get("aperture_half_width") or 0.0)
    lateral_limit = (
        max(0.30, min(0.45, half_width + 0.08))
        if mode == "front_drum"
        else max(
            0.50,
            min(_FRONT_OPENING_STANDOFF_M, half_width + 0.45),
        )
    )
    if not opened_content:
        configured_action_limit = meta.get("action_lateral_limit_m")
        try:
            action_limit = float(configured_action_limit)
        except (TypeError, ValueError):
            action_limit = lateral_limit
        if math.isfinite(action_limit) and action_limit > 0.0:
            lateral_limit = min(1.50, max(lateral_limit, action_limit))
    return center, entry_axis, standoff, lateral_limit


def _finite_vector(value, size: int):
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return vector if vector.shape == (size,) and np.all(np.isfinite(vector)) else None


def _openable_swing_geometry(entity, center):
    hinge_xy = _finite_vector(getattr(entity, "hinge_xy", None), 2)
    handle_xy = _finite_vector(getattr(entity, "handle_xy", None), 2)
    if hinge_xy is None or handle_xy is None:
        return None
    try:
        angle = float(getattr(entity, "swing_sign", 1.0)) * float(
            getattr(entity, "swing_angle", 0.0)
        )
    except (TypeError, ValueError):
        return None
    if not math.isfinite(angle) or abs(angle) <= 1e-9:
        return None
    handle_z = getattr(entity, "handle_z", None)
    try:
        handle_z = float(handle_z)
    except (TypeError, ValueError):
        handle_z = float(center[2])
    if not math.isfinite(handle_z):
        handle_z = float(center[2])
    meta = getattr(entity, "meta", None) or {}
    hinge = _finite_vector(meta.get("hinge_xyz"), 3)
    if hinge is None:
        hinge = np.array((hinge_xy[0], hinge_xy[1], handle_z), dtype=np.float64)
    axis_value = getattr(entity, "hinge_axis", None)
    if axis_value is None:
        axis_value = (0.0, 0.0, 1.0)
    axis = _finite_vector(axis_value, 3)
    if axis is None:
        return None
    return hinge, np.array((handle_xy[0], handle_xy[1], handle_z)), axis, angle


def _implicit_close_operator_geometry(entity, geometry=None):
    meta = getattr(entity, "meta", None) or {}
    if (
        meta.get("access_mode") != "front_opening"
        or "close_operator_lateral_offset_m" in meta
    ):
        return None
    geometry = geometry or _front_access_geometry(entity)
    if geometry is None:
        return None
    center, entry_axis, _standoff, _lateral_limit = geometry
    swing = _openable_swing_geometry(entity, center)
    if swing is None:
        return None
    hinge, handle, axis, angle = swing
    try:
        opened_handle = _rotate_point_about_axis(
            handle,
            hinge,
            axis,
            angle,
        )
    except ValueError:
        return None
    lateral_axis = np.array([-entry_axis[1], entry_axis[0]], dtype=np.float64)
    open_offset = _operator_lateral_offset(
        entity,
        _lateral_limit,
        action_phase=True,
        action_kind="open",
    )
    open_stand = center[:2] - entry_axis * _standoff
    if open_offset is not None:
        open_stand += lateral_axis * open_offset
    close_stand = open_stand + opened_handle[:2] - handle[:2]
    close_standoff = float(np.dot(center[:2] - close_stand, entry_axis))
    close_offset = float(np.dot(close_stand - center[:2], lateral_axis))
    if (
        not math.isfinite(close_standoff)
        or close_standoff <= _OPENABLE_STAND_DISTANCE_EPS_M
        or not math.isfinite(close_offset)
    ):
        return None
    return close_offset, close_standoff


def _implicit_close_operator_lateral_offset(entity, geometry=None) -> float | None:
    derived = _implicit_close_operator_geometry(entity, geometry)
    return None if derived is None else derived[0]


def _implicit_close_allows_base_corridor(entity) -> bool:
    meta = getattr(entity, "meta", None) or {}
    return not meta.get("close_reposition") and not any(
        field in meta
        for field in (
            "close_operator_lateral_offset_m",
            "action_operator_lateral_offset_m",
            "operator_lateral_offset_m",
        )
    )


def _implicit_close_contact_stand(
    entity,
    opened_contact,
    *,
    opened_content: bool = False,
    standoff_override: float | None = None,
):
    geometry = _front_access_geometry(
        entity,
        opened_content=opened_content,
        standoff_override=standoff_override,
    )
    derived = _implicit_close_operator_geometry(entity, geometry)
    if geometry is None or derived is None:
        return None
    swing = _openable_swing_geometry(entity, geometry[0])
    contact_xy = _finite_vector(opened_contact, 3)
    if swing is None or contact_xy is None:
        return None
    hinge, handle, axis, angle = swing
    try:
        opened_handle = _rotate_point_about_axis(handle, hinge, axis, angle)
    except ValueError:
        return None
    opened_radius = opened_handle[:2] - hinge[:2]
    radius_squared = float(np.dot(opened_radius, opened_radius))
    if radius_squared <= 1e-9:
        return None
    contact_radius = contact_xy[:2] - hinge[:2]
    fraction = float(np.dot(contact_radius, opened_radius) / radius_squared)
    projected = hinge[:2] + opened_radius * fraction
    if (
        fraction < -1e-6
        or fraction > 1.0 + 1e-6
        or float(np.linalg.norm(projected - contact_xy[:2])) > 1e-3
    ):
        return None
    fraction = float(np.clip(fraction, 0.0, 1.0))
    center, entry_axis, standoff, lateral_limit = geometry
    lateral_axis = np.array([-entry_axis[1], entry_axis[0]], dtype=np.float64)
    closed_offset = _operator_lateral_offset(
        entity,
        lateral_limit,
        action_phase=True,
        action_kind="open",
    )
    closed_stand = center[:2] - entry_axis * standoff
    if closed_offset is not None:
        closed_stand += lateral_axis * closed_offset
    opened_offset, opened_standoff = derived
    opened_stand = (
        center[:2]
        - entry_axis * opened_standoff
        + lateral_axis * opened_offset
    )
    stand = closed_stand + fraction * (opened_stand - closed_stand)
    return float(stand[0]), float(stand[1])


def _front_access_action_geometry(
    entity,
    *,
    opened_content: bool = False,
    action_kind: str | None = None,
    standoff_override: float | None = None,
):
    geometry = _front_access_geometry(
        entity,
        opened_content=opened_content,
        standoff_override=standoff_override,
    )
    if geometry is None or action_kind != "close":
        return geometry
    derived = _implicit_close_operator_geometry(entity, geometry)
    if derived is None:
        return geometry
    offset, standoff = derived
    center, entry_axis, _standoff, lateral_limit = geometry
    lateral_limit = min(
        1.50,
        max(lateral_limit, abs(offset) + _CONFIGURED_OPERATOR_LATERAL_TOL_M),
    )
    return center, entry_axis, standoff, lateral_limit


def _front_access_stand_is_clear(
    entity,
    stand_xy,
    *,
    opened_content: bool = False,
    action_kind: str | None = None,
    standoff_override: float | None = None,
    enforce_content_depth: bool = True,
) -> bool:
    base_geometry = _front_access_geometry(
        entity,
        opened_content=opened_content,
        standoff_override=standoff_override,
    )
    geometry = _front_access_action_geometry(
        entity,
        opened_content=opened_content,
        action_kind=action_kind,
        standoff_override=standoff_override,
    )
    if geometry is None:
        return True
    center, entry_axis, standoff, lateral_limit = geometry
    stand = np.asarray(stand_xy, dtype=np.float64)[:2]
    stationary_anchor = _stationary_front_access_anchor(
        entity,
        geometry,
        action_kind=action_kind,
    )
    if stationary_anchor is not None:
        return (
            float(np.linalg.norm(stand - stationary_anchor))
            <= _STATIONARY_OPERATOR_STAND_TOL_M
        )
    outside = float(np.dot(center[:2] - stand, entry_axis))
    lateral_axis = np.array([-entry_axis[1], entry_axis[0]], dtype=np.float64)
    signed_lateral_offset = float(np.dot(stand - center[:2], lateral_axis))
    lateral_offset = abs(signed_lateral_offset)
    configured_lateral_offset = _operator_lateral_offset(
        entity,
        lateral_limit,
        action_phase=not opened_content,
        action_kind=action_kind,
    )
    verb_specific_lateral = _verb_specific_operator_lateral_field(
        entity, action_kind
    )
    implicit_close_geometry = (
        _implicit_close_operator_geometry(entity, base_geometry)
        if action_kind == "close" and verb_specific_lateral is None
        else None
    )
    strict_content_lateral = (
        action_kind in {"insert", "extract"} and verb_specific_lateral is not None
    )
    lateral_tolerance = (
        _EXPLICIT_OPERATOR_LATERAL_TOL_M
        if strict_content_lateral and enforce_content_depth
        else _CONFIGURED_OPERATOR_LATERAL_TOL_M
    )
    derived_lateral_is_clear = True
    if configured_lateral_offset is not None:
        if implicit_close_geometry is None:
            if (
                abs(signed_lateral_offset - configured_lateral_offset)
                > lateral_tolerance
            ):
                return False
        else:
            base_lateral_limit = float(base_geometry[3])
            closed_offset = _operator_lateral_offset(
                entity,
                base_lateral_limit,
                action_phase=True,
                action_kind="open",
            )
            if closed_offset is None:
                closed_offset = 0.0
            lower, upper = sorted((closed_offset, configured_lateral_offset))
            if not (
                lower - lateral_tolerance
                <= signed_lateral_offset
                <= upper + lateral_tolerance
            ):
                if not _implicit_close_allows_base_corridor(entity):
                    return False
                derived_lateral_is_clear = False
    mode = (getattr(entity, "meta", None) or {}).get("access_mode")
    minimum_standoff = standoff
    if implicit_close_geometry is not None:
        minimum_standoff = min(minimum_standoff, float(base_geometry[2]))
        minimum_standoff -= _IMPLICIT_CLOSE_GRID_SNAP_TOL_M
    depth_is_clear = outside >= minimum_standoff - _OPENABLE_STAND_DISTANCE_EPS_M
    if mode == "front_drum":
        depth_is_clear = depth_is_clear and (
            outside
            <= standoff
            + _FRONT_DRUM_STAND_DEPTH_SLACK_M
            + _OPENABLE_STAND_DISTANCE_EPS_M
        )
    if strict_content_lateral and enforce_content_depth:
        depth_is_clear = depth_is_clear and (
            outside < standoff + _EXPLICIT_OPERATOR_OUTWARD_SLACK_M
        )
    if (
        derived_lateral_is_clear
        and depth_is_clear
        and lateral_offset <= lateral_limit
    ):
        return True
    if (
        implicit_close_geometry is None
        or base_geometry is None
        or not _implicit_close_allows_base_corridor(entity)
    ):
        return False
    base_center, base_entry_axis, base_standoff, base_lateral_limit = base_geometry
    base_outside = float(
        np.dot(base_center[:2] - stand, np.asarray(base_entry_axis)[:2])
    )
    base_lateral_axis = np.array(
        [-base_entry_axis[1], base_entry_axis[0]], dtype=np.float64
    )
    base_lateral = abs(float(np.dot(stand - base_center[:2], base_lateral_axis)))
    return bool(
        base_outside
        >= float(base_standoff)
        - _IMPLICIT_CLOSE_BASE_DEPTH_SLACK_M
        - _OPENABLE_STAND_DISTANCE_EPS_M
        and base_lateral <= float(base_lateral_limit)
    )


def _operator_lateral_offset(
    entity,
    lateral_limit: float,
    *,
    action_phase: bool = False,
    action_kind: str | None = None,
) -> float | None:
    meta = getattr(entity, "meta", None) or {}
    content_field = {
        "insert": "content_insert_operator_lateral_offset_m",
        "extract": "content_extract_operator_lateral_offset_m",
    }.get(action_kind)
    if content_field is not None and content_field in meta:
        configured = meta.get(content_field)
    elif action_kind == "close":
        if "close_operator_lateral_offset_m" in meta:
            configured = meta.get("close_operator_lateral_offset_m")
        else:
            configured = _implicit_close_operator_lateral_offset(entity)
            if configured is None:
                configured = (
                    meta.get("action_operator_lateral_offset_m")
                    if action_phase
                    else meta.get("operator_lateral_offset_m")
                )
    elif action_phase and "action_operator_lateral_offset_m" in meta:
        configured = meta.get("action_operator_lateral_offset_m")
    elif not action_phase and "content_operator_lateral_offset_m" in meta:
        configured = meta.get("content_operator_lateral_offset_m")
    else:
        configured = meta.get("operator_lateral_offset_m")
    if configured is None:
        return None
    value = float(configured)
    if not math.isfinite(value):
        return None
    return float(np.clip(value, -lateral_limit, lateral_limit))


def _verb_specific_operator_lateral_field(entity, action_kind: str | None):
    field = {
        "insert": "content_insert_operator_lateral_offset_m",
        "extract": "content_extract_operator_lateral_offset_m",
        "close": "close_operator_lateral_offset_m",
    }.get(action_kind)
    if field is None or field not in (getattr(entity, "meta", None) or {}):
        return None
    return field


def explicit_openable_operator_anchor(
    scene,
    key: str,
    *,
    opened_content: bool = False,
    action_kind: str | None = None,
    standoff_override: float | None = None,
) -> tuple[float, float] | None:
    entity = _openable_access_entity(scene, key)
    if entity is None or _verb_specific_operator_lateral_field(entity, action_kind) is None:
        return None
    return preferred_openable_stand(
        scene,
        key,
        opened_content=opened_content,
        action_kind=action_kind,
        standoff_override=standoff_override,
    )


def _stationary_front_access_anchor(
    entity,
    geometry=None,
    *,
    action_kind: str | None = None,
):
    """Return one handle-side stance shared by open, content, and close actions."""
    meta = getattr(entity, "meta", None) or {}
    if not meta.get("stationary_operator"):
        return None
    if action_kind == "close" and meta.get("close_reposition"):
        return None
    configured = np.asarray(meta.get("operator_stand_xy", ()), dtype=np.float64)
    if configured.shape == (2,) and np.all(np.isfinite(configured)):
        return configured
    geometry = geometry or _front_access_geometry(entity)
    if geometry is None:
        return None
    center, entry_axis, standoff, lateral_limit = geometry
    anchor = center[:2] - entry_axis * standoff
    lateral_axis = np.array([-entry_axis[1], entry_axis[0]], dtype=np.float64)
    configured_lateral_offset = _operator_lateral_offset(entity, lateral_limit)
    if configured_lateral_offset is not None:
        return anchor + lateral_axis * configured_lateral_offset
    handle_xy = getattr(entity, "handle_xy", None)
    if handle_xy is None:
        return anchor
    handle_offset = float(
        np.dot(np.asarray(handle_xy, dtype=np.float64)[:2] - center[:2], lateral_axis)
    )
    if abs(handle_offset) <= 1e-9:
        return anchor
    offset = float(
        np.clip(
            handle_offset
            + math.copysign(_STATIONARY_OPERATOR_HANDLE_MARGIN_M, handle_offset),
            -lateral_limit,
            lateral_limit,
        )
    )
    return anchor + lateral_axis * offset


def _openable_access_entity(scene, key: str):
    resolve = getattr(scene, "resolve", None)
    if not callable(resolve):
        return None
    inside = getattr(scene, "inside", None) or {}
    current = key
    seen = set()
    entity = None
    while current not in seen:
        seen.add(current)
        try:
            entity = resolve(current)
        except (KeyError, ValueError):
            return None
        if _front_access_geometry(entity) is not None:
            return entity
        parent = inside.get(current)
        if parent is None:
            break
        current = str(parent)
    return entity


def _front_container_aperture(scene, key: str, entity):
    aperture = np.asarray(
        (getattr(entity, "meta", None) or {}).get("aperture_center", ()),
        dtype=np.float64,
    )
    if aperture.shape == (3,) and np.all(np.isfinite(aperture)):
        return aperture
    try:
        aperture = np.asarray(scene.object_pos(key), dtype=np.float64)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return aperture if aperture.shape == (3,) and np.all(np.isfinite(aperture)) else None


def _open_front_container_context(scene, key: str | None):
    """Return the open front-opening ancestor and its aperture focus."""
    resolve = getattr(scene, "resolve", None)
    if key is None or not callable(resolve):
        return None
    parents = getattr(scene, "inside", None) or {}
    opened = getattr(scene, "open_objs", None) or set()
    current = str(key)
    seen = set()
    while current not in seen:
        seen.add(current)
        try:
            entity = resolve(current)
        except (KeyError, ValueError):
            entity = None
        meta = getattr(entity, "meta", None) or {}
        if meta.get("access_mode") == "front_opening":
            if current not in opened:
                return None
            aperture = _front_container_aperture(scene, current, entity)
            return (current, aperture) if aperture is not None else None
        parent = parents.get(current)
        if parent is None:
            return None
        current = str(parent)
    return None


def openable_stand_anchors(
    scene,
    key: str,
    *,
    opened_content: bool = False,
    action_kind: str | None = None,
    standoff_override: float | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return the centered and lateral bounds of a front-opening stance corridor."""
    entity = _openable_access_entity(scene, key)
    geometry = (
        _front_access_action_geometry(
            entity,
            opened_content=opened_content,
            action_kind=action_kind,
            standoff_override=standoff_override,
        )
        if entity is not None
        else None
    )
    if geometry is None:
        return ()
    center, entry_axis, standoff, lateral_limit = geometry
    stationary_anchor = _stationary_front_access_anchor(
        entity,
        geometry,
        action_kind=action_kind,
    )
    if stationary_anchor is not None:
        return ((float(stationary_anchor[0]), float(stationary_anchor[1])),)
    stand = center[:2] - entry_axis * standoff
    lateral_axis = np.array([-entry_axis[1], entry_axis[0]], dtype=np.float64)
    configured_lateral_offset = _operator_lateral_offset(
        entity,
        lateral_limit,
        action_phase=not opened_content,
        action_kind=action_kind,
    )
    if configured_lateral_offset is not None:
        preferred = stand + lateral_axis * configured_lateral_offset
        return tuple(
            (float(point[0]), float(point[1]))
            for point in (
                preferred,
                stand,
                stand + lateral_axis * lateral_limit,
                stand - lateral_axis * lateral_limit,
            )
        )
    return tuple(
        (float(point[0]), float(point[1]))
        for point in (
            stand,
            stand + lateral_axis * lateral_limit,
            stand - lateral_axis * lateral_limit,
        )
    )


def preferred_openable_stand(
    scene,
    key: str,
    *,
    opened_content: bool = False,
    action_kind: str | None = None,
    standoff_override: float | None = None,
) -> tuple[float, float] | None:
    """Return the centered, outside stance for a front-opening entity or its container."""
    anchors = openable_stand_anchors(
        scene,
        key,
        opened_content=opened_content,
        action_kind=action_kind,
        standoff_override=standoff_override,
    )
    return anchors[0] if anchors else None


def openable_pullback_xy(entity, root_xy, opening: bool) -> tuple[float, float]:
    """Step away from an outward-swinging container while moving its door."""
    root = np.asarray(root_xy, dtype=np.float64).reshape(2)
    meta = getattr(entity, "meta", None) or {}
    if meta.get("access_mode") != "front_opening":
        return float(root[0]), float(root[1])
    policy_key = "opening_pullback_m" if opening else "closing_pullback_m"
    configured_pullback = meta.get(policy_key)
    if meta.get("stationary_operator") and configured_pullback is None:
        return float(root[0]), float(root[1])
    entry_axis = np.asarray(meta.get("entry_axis"), dtype=np.float64).reshape(-1)
    if entry_axis.shape != (2,):
        return float(root[0]), float(root[1])
    norm = float(np.linalg.norm(entry_axis))
    if norm <= 1e-9:
        return float(root[0]), float(root[1])
    pullback = (
        float(configured_pullback)
        if configured_pullback is not None
        else (
            _FRONT_OPENING_PULLBACK_M
            if opening
            else _FRONT_OPENING_CLOSE_PULLBACK_M
        )
    )
    end = root - entry_axis / norm * pullback
    return float(end[0]), float(end[1])


def _swing_stand_is_clear(entity, stand_xy, swing_sign: float | None) -> bool:
    meta = getattr(entity, "meta", None) or {}

    if meta.get("openable_kind") == "lid":
        return True
    hinge_xy = getattr(entity, "hinge_xy", None)
    handle_xy = getattr(entity, "handle_xy", None)
    is_navigation_door = handle_xy is None
    if handle_xy is None:
        handle_xy = getattr(entity, "xy", None)
    if hinge_xy is None or handle_xy is None:
        return True
    hinge = np.asarray(hinge_xy, dtype=np.float64)[:2]
    closed_handle = np.asarray(handle_xy, dtype=np.float64)[:2]
    radius = closed_handle - hinge
    if float(np.linalg.norm(radius)) <= 1e-6:
        return True
    authored_sign = float(getattr(entity, "swing_sign", 1.0))
    configured_sweep_standoff = meta.get("sweep_standoff_m")
    sweep_standoff = (
        float(configured_sweep_standoff)
        if configured_sweep_standoff is not None
        else (
            _FRONT_DRUM_SWEEP_STANDOFF_M
            if meta.get("access_mode") == "front_drum"
            else _OPENABLE_SWEEP_STANDOFF_M
        )
    )
    signs = (
        (float(swing_sign),)
        if swing_sign is not None
        else ((authored_sign, -authored_sign) if is_navigation_door else (authored_sign,))
    )
    for sign in signs:
        angle = sign * float(getattr(entity, "swing_angle", 0.0))
        clear = True
        for fraction in _OPENABLE_SWEEP_FRACTIONS:
            theta = angle * fraction
            cosine, sine = math.cos(theta), math.sin(theta)
            free_edge = hinge + np.array(
                [
                    radius[0] * cosine - radius[1] * sine,
                    radius[0] * sine + radius[1] * cosine,
                ],
                dtype=np.float64,
            )
            if (
                _point_segment_distance_xy(stand_xy, hinge, free_edge)
                < sweep_standoff
            ):
                clear = False
                break
        if clear:
            return True
    return False


def openable_stand_is_visually_clear(
    scene,
    key: str,
    stand_xy,
    *,
    swing_sign: float | None = None,
    include_sweep: bool = True,
    action_phase: bool | None = None,
    action_kind: str | None = None,
    standoff_override: float | None = None,
) -> bool:
    """Keep the wearer outside a front appliance and a swinging leaf's camera-filling sweep."""
    entity = _openable_access_entity(scene, key)
    if entity is None:
        return True
    content_parent = (getattr(scene, "inside", None) or {}).get(str(key))
    content_target = content_parent is not None and str(content_parent) != str(key)
    if not _front_access_stand_is_clear(
        entity,
        stand_xy,
        opened_content=(
            not include_sweep if action_phase is None else not action_phase
        ),
        action_kind=action_kind,
        standoff_override=standoff_override,
        enforce_content_depth=not content_target,
    ):
        return False
    return not include_sweep or _swing_stand_is_clear(
        entity,
        stand_xy,
        swing_sign,
    )


def _rotated_aabb_about_axis(aabb, origin, axis, angle: float):
    """Tight world AABB of an axis-aligned box after an arbitrary-axis rotation."""
    mn, mx = (np.asarray(value, dtype=np.float64) for value in aabb)
    corners = np.array(
        [
            [x, y, z]
            for x in (mn[0], mx[0])
            for y in (mn[1], mx[1])
            for z in (mn[2], mx[2])
        ],
        dtype=np.float64,
    )
    rotated = np.array(
        [_rotate_point_about_axis(point, origin, axis, angle) for point in corners]
    )
    return rotated.min(axis=0), rotated.max(axis=0)


def _openable_swing_fraction(entity, contact) -> float:
    """Recover normalized leaf progress from a point carried along the primary hinge arc."""
    handle_xy = getattr(entity, "handle_xy", None)
    hinge_xy = getattr(entity, "hinge_xy", None)
    if handle_xy is None or hinge_xy is None:
        return 0.0
    handle_z = getattr(entity, "handle_z", None)
    hinge = np.asarray(
        (getattr(entity, "meta", None) or {}).get("hinge_xyz")
        or (
            float(hinge_xy[0]),
            float(hinge_xy[1]),
            float(handle_z) if handle_z is not None else float(contact[2]),
        ),
        dtype=np.float64,
    )
    handle = np.asarray(
        (
            float(handle_xy[0]),
            float(handle_xy[1]),
            float(handle_z) if handle_z is not None else float(contact[2]),
        ),
        dtype=np.float64,
    )
    axis = np.asarray(
        getattr(entity, "hinge_axis", None) or (0.0, 0.0, 1.0),
        dtype=np.float64,
    )
    axis /= np.linalg.norm(axis) + 1e-9
    closed_radius = handle - hinge
    live_radius = np.asarray(contact, dtype=np.float64) - hinge
    closed_radius -= axis * np.dot(closed_radius, axis)
    live_radius -= axis * np.dot(live_radius, axis)
    if min(np.linalg.norm(closed_radius), np.linalg.norm(live_radius)) <= 1e-6:
        return 0.0
    angle = math.atan2(
        float(np.dot(axis, np.cross(closed_radius, live_radius))),
        float(np.dot(closed_radius, live_radius)),
    )
    open_angle = float(getattr(entity, "swing_sign", 1.0)) * float(
        getattr(entity, "swing_angle", 0.0)
    )
    if abs(open_angle) <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, angle / open_angle))


def _openable_leaf_box_sources(scene, entity) -> tuple[tuple[tuple, ...], ...]:
    """Return each moving leaf's closed, per-collider boxes when they are available."""
    leaf_paths = tuple(
        str(path).rstrip("/")
        for path in (getattr(entity, "leaf_paths", None) or ())
        if path
    )
    cworld = getattr(scene, "cworld", None)
    boxes_fn = getattr(cworld, "boxes", None)
    collision_boxes = tuple(boxes_fn()) if callable(boxes_fn) else ()
    fallback_boxes = getattr(entity, "leaf_bboxes", None) or ()
    groups = []
    for index in range(max(len(leaf_paths), len(fallback_boxes))):
        root = leaf_paths[index] if index < len(leaf_paths) else None
        matches = (
            tuple(
                box
                for box in collision_boxes
                if root is not None
                and (box[0] == root or str(box[0]).startswith(root + "/"))
            )
            if root is not None
            else ()
        )
        if matches:
            groups.append(matches)
        elif index < len(fallback_boxes):
            mn, mx = fallback_boxes[index]
            groups.append(((f"{entity.name}_leaf{index}", mn, mx),))
        else:
            groups.append(())
    return tuple(groups)


def _rotated_open_leaf_aabb(
    aabb,
    hinge,
    *,
    angle: float,
    arbitrary_axis: bool,
    axis,
    hinge_xyz,
):
    if arbitrary_axis:
        origin = hinge if len(hinge) == 3 else hinge_xyz
        if origin is None:
            return None
        return _rotated_aabb_about_axis(aabb, origin, axis, angle)

    mn, mx = aabb
    c, s = math.cos(angle), math.sin(angle)
    corners = (
        (mn[0], mn[1]),
        (mn[0], mx[1]),
        (mx[0], mn[1]),
        (mx[0], mx[1]),
    )
    rotated = [
        (
            (px - hinge[0]) * c - (py - hinge[1]) * s + hinge[0],
            (px - hinge[0]) * s + (py - hinge[1]) * c + hinge[1],
        )
        for px, py in corners
    ]
    return (
        np.array(
            [min(point[0] for point in rotated), min(point[1] for point in rotated), mn[2]],
            dtype=np.float64,
        ),
        np.array(
            [max(point[0] for point in rotated), max(point[1] for point in rotated), mx[2]],
            dtype=np.float64,
        ),
    )


def _open_leaf_boxes(
    scene,
    e,
    swing_direction_scale: float = 1.0,
    swing_fraction: float = 1.0,
    leaf_box_sources=None,
) -> list:
    """World-AABB(s) of a container's leaves at the requested swing fraction, computed KINEMATICALLY
    (the USD leaf prims are still CLOSED at frame-gen). MIRRORS the render's ``_door_box``: each leaf's CLOSED
    AABB corners are rotated about ITS OWN render-derived hinge (``e.leaf_hinges[i]``) by
    ``e.leaf_signs[i] * e.swing_angle`` about +Z. A baked leaf's PRIM ORIGIN is NOT its hinge, so we use the
    render's geometry-derived per-leaf hinge/sign (plumbed by episode_runtime._place_container_leaves), falling
    back to the primary ``e.hinge_xy`` / ``e.swing_sign`` for any leaf without a per-leaf entry and to the
    leaf's live (still-closed) bbox if ``e.leaf_bboxes`` was not plumbed. Returns [(name, mn, mx), ...]; []
    if the entity has no swing leaves / hinge. General: the container's own geometry, no per-asset constant."""
    full_angle = float(getattr(e, "swing_angle", 0.0) or 0.0)
    if full_angle <= 1e-6:
        return []
    ang = full_angle * max(0.0, min(1.0, float(swing_fraction)))
    hinges = getattr(e, "leaf_hinges", None) or ()
    signs = getattr(e, "leaf_signs", None) or ()
    bboxes = getattr(e, "leaf_bboxes", None) or ()
    if not bboxes:  # not render-plumbed -> read the (still-closed) leaf prims live
        leaf_paths = getattr(e, "leaf_paths", None) or ()
        stage = getattr(scene, "stage", None)
        if not leaf_paths or stage is None:
            return []
        import fpv_scene

        tmp = []
        for lp in leaf_paths:
            prim = stage.GetPrimAtPath(lp)
            if prim and prim.IsValid():
                bb = fpv_scene.compute_world_bbox(prim)
                mn, mx = bb.GetMin(), bb.GetMax()
                tmp.append(
                    (
                        (float(mn[0]), float(mn[1]), float(mn[2])),
                        (float(mx[0]), float(mx[1]), float(mx[2])),
                    )
                )
        bboxes = tuple(tmp)
    if leaf_box_sources is None:
        leaf_box_sources = _openable_leaf_box_sources(scene, e)
    if not leaf_box_sources:
        leaf_box_sources = tuple(
            ((f"{e.name}_leaf{index}", mn, mx),)
            for index, (mn, mx) in enumerate(bboxes)
        )
    hinge0 = getattr(e, "hinge_xy", None)
    sign0 = float(getattr(e, "swing_sign", 1.0) or 1.0)
    arbitrary_axis = (getattr(e, "meta", None) or {}).get("openable_kind") in (
        "lid",
        "flip_down",
    )
    axis = np.asarray(getattr(e, "hinge_axis", (0.0, 0.0, 1.0)), dtype=np.float64)
    hinge_xyz0 = (getattr(e, "meta", None) or {}).get("hinge_xyz")
    out = []
    for i, sources in enumerate(leaf_box_sources):
        hinge = hinges[i] if i < len(hinges) else hinge0
        sgn = (float(signs[i]) if i < len(signs) else sign0) * float(
            swing_direction_scale
        )
        if hinge is None:
            continue
        for source_index, (owner, mn, mx) in enumerate(sources):
            rotated = _rotated_open_leaf_aabb(
                (mn, mx),
                hinge,
                angle=sgn * ang,
                arbitrary_axis=arbitrary_axis,
                axis=axis,
                hinge_xyz=hinge_xyz0,
            )
            if rotated is None:
                continue
            rmn, rmx = rotated
            out.append((str(owner) or f"{e.name}_leaf{i}_{source_index}", rmn, rmx))
    return out


def _label_occluder_boxes(
    scene,
    obj_key: str,
    open_objs=None,
    *,
    ignore_target_collision: bool = False,
) -> list:
    """Return the current occluders for reading a label on ``obj_key``.

    ``CollisionWorld`` stores the authored, closed door geometry.  When the labeled object is
    inside a container that the executor has opened, replace those closed leaf boxes with the
    same swung-open boxes used by the reach planner.  Otherwise a visibly open cabinet is still
    treated as opaque at its closed pose and every label inside it is falsely marked occluded.
    """
    cw = getattr(scene, "cworld", None)
    if cw is None:
        return []
    try:
        boxes = list(cw.boxes())
    except Exception:  # noqa: BLE001 -- visibility failure must not abort a LookAt
        return []

    try:
        target_entity = scene.resolve(obj_key)
    except (AttributeError, KeyError, ValueError):
        target_entity = None
    currently_open = (
        getattr(scene, "open_objs", None) or set() if open_objs is None else open_objs
    )
    target_meta = getattr(target_entity, "meta", None) or {}
    target_prefix = str(getattr(target_entity, "prim_path", "") or "").rstrip(
        "/"
    )
    ignore_open_target = ignore_target_collision and obj_key in currently_open
    if target_prefix and (
        target_meta.get("label_pos") is not None or ignore_open_target
    ):
        boxes = [
            row
            for row in boxes
            if not (
                str(row[0]).rstrip("/") == target_prefix
                or str(row[0]).startswith(target_prefix + "/")
            )
        ]
        if ignore_open_target and target_entity is not None:
            boxes.extend(_open_leaf_boxes(scene, target_entity))

    container = (getattr(scene, "inside", None) or {}).get(obj_key)
    if not container or container not in currently_open:
        return boxes
    try:
        entity = scene.resolve(container)
    except (KeyError, ValueError):
        return boxes
    leaf_paths = tuple(str(path).rstrip("/") for path in entity.leaf_paths if path)
    if not leaf_paths:
        return boxes

    boxes = [
        row
        for row in boxes
        if not any(
            str(row[0]).rstrip("/") == path or str(row[0]).startswith(path + "/")
            for path in leaf_paths
        )
    ]
    boxes.extend(_open_leaf_boxes(scene, entity))
    return boxes


def _into_obstacle_boxes(scene, surf_key, e) -> list:
    """Obstacle AABBs for reaching INTO a shelved container: every OTHER furniture box (the container's OWN
    outer box excluded so the hand can enter its hollow interior), PLUS the container's REAL interior shelf
    boards (ALL of ``e.spaces`` as z-slabs) and its OPEN door leaf/leaves added back -- the true obstacles the
    arm must dodge. General for any container that exposes ``spaces``."""
    prefix = (e.prim_path or "").rstrip("/")
    cw = getattr(scene, "cworld", None)
    if cw is not None and getattr(scene, "unified_geom", False):
        boxes = [
            b
            for b in cw.boxes()
            if not (prefix and (b[0] == prefix or str(b[0]).startswith(prefix + "/")))
        ]
    else:
        boxes = [b for b in scene.solid_boxes() if b[0] != e.name]
    for sp in getattr(e, "spaces", None) or ():
        _, top_z, bot_z, cx, cy, hx, hy = sp
        boxes.append(
            (
                sp[0],
                np.array([cx - hx, cy - hy, bot_z], dtype=np.float64),
                np.array([cx + hx, cy + hy, top_z], dtype=np.float64),
            )
        )
    boxes.extend(_open_leaf_boxes(scene, e))
    return boxes


def _front_insertion_waypoints(entity, destination, obj_half, obj_height) -> tuple:
    meta = getattr(entity, "meta", None) or {}
    center = np.asarray(meta.get("aperture_center"), dtype=np.float64)
    axis_xy = np.asarray(meta.get("entry_axis"), dtype=np.float64).reshape(2)
    axis_norm = float(np.linalg.norm(axis_xy))
    if center.shape != (3,) or axis_norm <= 1e-9:
        raise PrimitiveFailure(
            PrimitiveError.MISSING_AFFORDANCE,
            f"{entity.name} has no measurable front aperture",
        )
    axis_xy /= axis_norm
    lateral = np.array([-axis_xy[1], axis_xy[0]], dtype=np.float64)
    object_radius = abs(lateral[0]) * float(obj_half[0]) + abs(lateral[1]) * float(
        obj_half[1]
    )
    half_width = float(meta.get("aperture_half_width") or 0.0)
    half_height = float(meta.get("aperture_half_height") or 0.0)
    object_half_height = 0.5 * float(obj_height or 0.0)
    target = np.asarray(destination, dtype=np.float64).copy()
    if object_radius > half_width or object_half_height > half_height:
        raise PrimitiveFailure(
            PrimitiveError.TARGET_TOO_SMALL,
            f"held object does not fit through {entity.name} front aperture",
        )
    lateral_limit = max(0.0, half_width - object_radius - _FRONT_APERTURE_MARGIN_M)
    lateral_offset = float(np.dot(target[:2] - center[:2], lateral))
    lateral_offset = float(np.clip(lateral_offset, -lateral_limit, lateral_limit))
    insertion_depth = max(float(np.dot(target[:2] - center[:2], axis_xy)), 0.12)
    target[:2] = center[:2] + axis_xy * insertion_depth + lateral * lateral_offset
    vertical_limit = max(
        0.0,
        half_height - object_half_height - _FRONT_APERTURE_MARGIN_M,
    )
    target[2] = float(center[2]) + float(
        np.clip(float(target[2] - center[2]), -vertical_limit, vertical_limit)
    )
    pre = center.copy()
    pre_clearance = (
        _FRONT_DRUM_PREINSERT_CLEARANCE_M
        if meta.get("access_mode") == "front_drum"
        else 0.18
    )
    pre[:2] -= axis_xy * pre_clearance
    pre[:2] += lateral * lateral_offset
    pre[2] = target[2]
    return pre, target, np.array([axis_xy[0], axis_xy[1], 0.0])


def _front_opening_post_interaction_pose(
    scene,
    key: str,
    entity,
    root_xy,
    root_yaw: float,
) -> tuple[tuple[float, float], float]:
    """Return the action-safe stand and aperture-facing yaw after an interaction."""
    preferred = preferred_openable_stand(
        scene,
        key,
        opened_content=False,
        standoff_override=_FRONT_OPENING_ACTION_CAMERA_STANDOFF_M,
    )
    meta = getattr(entity, "meta", None) or {}
    center = np.asarray(meta.get("aperture_center"), dtype=np.float64)
    entry_axis = np.asarray(meta.get("entry_axis"), dtype=np.float64).reshape(-1)
    root = np.asarray(root_xy, dtype=np.float64).reshape(2)
    axis_norm = float(np.linalg.norm(entry_axis))
    if (
        center.shape != (3,)
        or entry_axis.shape != (2,)
        or axis_norm <= 1e-9
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(root))
    ):
        target = root
    elif preferred is not None:
        target = np.asarray(preferred, dtype=np.float64)
    else:
        entry_axis /= axis_norm
        current_standoff = float(np.dot(center[:2] - root, entry_axis))
        retreat = max(
            0.0,
            _FRONT_OPENING_ACTION_CAMERA_STANDOFF_M - current_standoff,
        )
        target = root - entry_axis * retreat
    target_xy = float(target[0]), float(target[1])
    facing = center[:2] - target
    target_yaw = (
        gait.yaw_for_dir(float(facing[0]), float(facing[1]))
        if center.shape == (3,)
        and np.all(np.isfinite(center))
        and float(np.linalg.norm(facing)) > 1e-9
        else float(root_yaw)
    )
    return target_xy, float(target_yaw)


def _require_front_drum_pile_capacity(entity, place_z, object_height) -> None:
    meta = getattr(entity, "meta", None) or {}
    center = np.asarray(meta.get("aperture_center"), dtype=np.float64)
    half_height = meta.get("aperture_half_height")
    if (
        meta.get("access_mode") != "front_drum"
        or center.shape != (3,)
        or not np.all(np.isfinite(center))
        or object_height is None
        or not math.isfinite(float(object_height))
        or half_height is None
        or not math.isfinite(float(half_height))
        or float(half_height) <= 0.0
    ):
        raise PrimitiveFailure(
            PrimitiveError.NO_FREE_PLACEMENT,
            f"{entity.name} has no measured front-drum capacity for another object",
        )
    object_top = float(place_z) + 0.5 * float(object_height)
    aperture_top = (
        float(center[2]) + float(half_height) - _FRONT_APERTURE_MARGIN_M
    )
    if object_top > aperture_top + 1e-9:
        raise PrimitiveFailure(
            PrimitiveError.NO_FREE_PLACEMENT,
            f"{entity.name} cannot fit another object below its front-drum aperture",
        )


def _validated_front_drum_pile_z(
    scene,
    held_key,
    surf_key,
    surf_e,
    place_xy,
    place_z,
    object_height,
    *,
    enabled,
):
    if not enabled:
        return place_z
    piled_z = _stack_lift_z(
        scene,
        held_key,
        surf_key,
        surf_e,
        place_xy,
        place_z,
        None,
    )
    _require_front_drum_pile_capacity(surf_e, piled_z, object_height)
    return piled_z


def _held_base_offset(scene, held_key) -> float:
    """How far the held object's ORIGIN sits ABOVE its base (m), from its live world bbox. Dropping the
    origin AT a thin shelf board's top would sink the base BELOW the board -> the object lands UNDER the
    board and the de-penetration settle pushes it DOWN through the gap (it falls a shelf). Adding this offset
    rests the object's BASE on the board top instead. 0.0 if unavailable. General: the object's own geometry."""
    preview_offsets = getattr(scene, "_content_preview_object_base_offsets", None) or {}
    if held_key in preview_offsets:
        return float(preview_offsets[held_key])
    hp = getattr(scene.resolve(held_key), "prim_path", None) if held_key else None
    if not hp or getattr(scene, "stage", None) is None:
        return 0.0
    try:
        import fpv_scene
        from pxr import Usd, UsdGeom

        prim = scene.stage.GetPrimAtPath(hp)
        oz = float(
            UsdGeom.XformCache(Usd.TimeCode.Default())
            .GetLocalToWorldTransform(prim)
            .ExtractTranslation()[2]
        )
        bz = float(fpv_scene.compute_world_bbox(prim).GetMin()[2])
        return max(0.0, oz - bz)
    except Exception:  # noqa: BLE001
        return 0.0


def _stack_lift_z(
    scene, held_key, surf_key, surf_e0, place_xy, place_z, shelf_gap_ceiling
):
    """Raise ``place_z`` so the object lands ON TOP of whatever already occupies ``place_xy``, not through it.

    Without this, a forced pile onto a small surface (a 2nd book on a bookcase TOP) resolves place_z at the
    bare surface and the release shoves the new object off the edge. With live world bboxes -- the prior
    objects are already settled by the per-command physics pass -- clearance is a fraction of the HELD
    object's OWN height. No-op unless ``place_xy`` actually overlaps a prior placed object.

    SHELF-CEILING CLAMP: a stack must still FIT in the gap it is stacked into. On a MULTI-LEVEL container
    every level overlaps its neighbours in xy BY CONSTRUCTION, so the xy-only overlap test also fires on the
    object one shelf UP -- and lifting onto THAT one drives the drop through the board between them, far
    above the chosen shelf and outside the arm envelope (bt_chemlab: a 1.28 m shelf became a 1.854 m target,
    258 mm out of reach; the bottle then stayed welded to the hand and cost two more commands).
    ``shelf_gap_ceiling`` is the underside of the board ABOVE this space, already computed for the reach
    pose. It is None / inf for a bin, a plain surface and the TOP space, so the bookcase-pile case this lift
    exists for is unaffected by the clamp."""
    if getattr(scene, "stage", None) is None:
        return place_z
    import fpv_scene as _fs_stack

    _held_pp = getattr(scene.resolve(held_key), "prim_path", None)
    if not _held_pp:
        return place_z
    # Skip the DROP-TARGET container prim itself: when placing INTO a container that is ALSO a graspable (a
    # dual-role bowl/tupperware, so it IS in grasp_objs), its rim must NOT act as a stacking surface --
    # otherwise the content is lifted ABOVE the rim instead of dropped INTO the interior. Furniture
    # containers are not in grasp_objs, so this is a no-op for them.
    _surf_pp = getattr(surf_e0, "prim_path", None) if surf_e0 is not None else None
    _ohx, _ohy, _ = _obj_extent(scene, held_key)
    _tops = []
    for _go in getattr(scene, "grasp_objs", None) or []:
        _gkey = _go.get("key") if isinstance(_go, dict) else getattr(_go, "key", None)
        _gp = (
            _go.get("path")
            if isinstance(_go, dict)
            else getattr(_go, "prim_path", None)
        )
        if not _gp or _gp in (_held_pp, _surf_pp):
            continue
        if surf_key is not None and _gkey is not None:
            if (
                getattr(scene, "on", {}).get(str(_gkey)) != surf_key
                and getattr(scene, "inside", {}).get(str(_gkey)) != surf_key
            ):
                continue
        _gpr = scene.stage.GetPrimAtPath(_gp)
        if _gpr is None or not _gpr.IsValid():
            continue
        _gmn, _gmx = (lambda _b: (_b.GetMin(), _b.GetMax()))(
            _fs_stack.compute_world_bbox(_gpr)
        )
        if (
            float(place_xy[0]) + _ohx > float(_gmn[0])
            and float(place_xy[0]) - _ohx < float(_gmx[0])
            and float(place_xy[1]) + _ohy > float(_gmn[1])
            and float(place_xy[1]) - _ohy < float(_gmx[1])
        ):
            _tops.append(float(_gmx[2]))
    if not _tops:
        return place_z
    _hb = _fs_stack.compute_world_bbox(scene.stage.GetPrimAtPath(_held_pp))
    _hh = float(_hb.GetMax()[2] - _hb.GetMin()[2])
    # origin at stack_top + ~half the held height (+ a small margin) -> its base clears the top.
    _lift = max(_tops) + 0.55 * _hh
    if shelf_gap_ceiling is not None and math.isfinite(shelf_gap_ceiling):
        _lift = min(
            _lift,
            float(shelf_gap_ceiling) - _hh + _held_base_offset(scene, held_key),
        )
    return max(float(place_z), _lift)


def _front_insert_openable_key(
    scene: SceneState,
    key: str | None,
    access_mode: str | None,
) -> str | None:
    if access_mode not in {"front_drum", "front_opening"}:
        return None
    controllers = getattr(scene, "_openable_controllers", None) or {}
    return key if key in controllers else None


def _held_place_key(st: SchedState, goal: Goal) -> str:
    if st.held is None:
        raise PrimitiveFailure(
            PrimitiveError.PLACE_UNSTABLE, "PLACE with an empty hand"
        )
    held_key = str(st.held["obj_key"])
    requested_key = str(goal.meta.get("_held_object_key") or "")
    if requested_key and requested_key != held_key:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"PLACE target {requested_key!r} is not the held object {held_key!r}",
        )
    return held_key


def produce_place(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Reach out + set the held object down at ``goal.place_xy`` (z from goal.place_z or a named surface),
    release, and retract. Clears st.held; the object rests at the drop point afterwards."""
    held_key = _held_place_key(st, goal)
    grasp_arm = st.held["grasp_arm"]
    otw = st.held["otw"]
    placement_ik_budget = IKSearchBudget(MAX_IK_SOLVES_PER_PLACEMENT_SEARCH)
    _carried = st.held.get(
        "carried"
    )  # contents riding held_key (captured before st.held clears)
    goal.meta["_carried"] = tuple(_carried or ())
    goal.meta["_carried_root"] = held_key
    goal.meta.pop("_placement_fit_yaw_rad", None)
    place_xy = goal.place_xy
    place_z = goal.place_z
    surf_key = goal.meta.get("surface")
    floor_placement = bool(goal.meta.get("_floor_anchor"))
    surf_e0 = None
    surface_view = None
    container_view = None
    requested_inside = (
        bool(goal.meta.get("into")) or goal.prim is Primitive.PLACE_INSIDE
    )
    if surf_key is not None and surf_key in scene.entities:
        raw = scene.entities[surf_key]
        surface_view = scene.target_view(surf_key, Role.SURFACE)
        container_view = scene.target_view(surf_key, Role.CONTAINER)
        surf_e0 = (
            (container_view or raw)
            if requested_inside
            else (surface_view or container_view or raw)
        )
    into = (
        requested_inside
        or (
            surface_view is None
            and surf_e0 is not None
            and surf_e0.role is Role.CONTAINER
        )
    )  # [PutInside], OR a plain [Put] onto a CONTAINER (open bin): drop INTO the interior + declare INSIDE
    access_mode = (
        (getattr(surf_e0, "meta", None) or {}).get("access_mode")
        if surf_e0 is not None
        else None
    )
    front_access_into = bool(
        into and access_mode in {"front_drum", "front_opening"}
    )
    front_drum_into = bool(into and access_mode == "front_drum")
    low_front_into = bool(
        into and surf_e0 is not None and _low_front_openable(scene, surf_e0)
    )
    exact_rest_crouch = bool(
        low_front_into
        and (getattr(surf_e0, "meta", None) or {}).get(
            "content_insert_exact_rest_crouch"
        )
    )
    placement_reach_arm = _placement_reach_seed_arm(
        rig,
        grasp_arm,
        exact_rest_crouch,
    )
    keep_crouched_into = front_drum_into or low_front_into
    active_crouch = _active_crouch_fraction(scene, st, surf_key)
    required_crouch = (
        max(
            _LOW_FRONT_INSERT_MIN_SQUAT
            if low_front_into
            else _FRONT_APPLIANCE_SQUAT,
            active_crouch,
        )
        if keep_crouched_into
        else 0.0
    )
    goal.meta.pop("_placement_result", None)
    if surf_key is not None:
        goal.meta["_placement_intent"] = {
            "predicate": "INSIDE" if into else "ON",
            "subject": held_key,
            "target": surf_key,
            "chosen_xy": None,
        }
    _sp = None
    _ohz = None
    _base_off = 0.0
    _shelf_gap_ceiling = None  # set to the board-above underside (or inf) ONLY for a shelved PutInside pose
    _shelf_fit_ceiling = None
    if surf_e0 is not None:
        _drop_probe_override = getattr(scene, "_drop_site_probe_override", None)
        _window_probe_override = getattr(scene, "_lateral_free_window_override", None)
        if into and surf_e0.role.name == "CONTAINER":
            # PutInside drop-point. If the container exposes geometry-extracted interior SPACES, place ON the
            # chosen space's board (goal.meta['space'] selects one by name/index; unspecified -> auto-
            # distribute across free spaces, upper levels first) and confine the xy to THAT space's footprint.
            # Otherwise (a top-open bin with no internal boards) fall back to the single interior drop (drop_z,
            # else the interior top). z is then a MEASURED plane, never a heuristic mid-air height -> no float.
            # NO physics pre-simulation: the object is released here and the ONE post-action pass settles +
            # validates it; an unreachable spot fails honestly (OUT_OF_REACH) below. An explicit command
            # target (place_xy / place_z) is honoured.
            _spaces = getattr(surf_e0, "spaces", None) or ()
            _shalf = None
            # The board this put lands on, or None for a container with no extracted interior boards
            # (a top-open bin, or a doored unit whose interior yielded no spaces). Everything the
            # space branch derives is keyed off it, so ``None`` is what tells the code further down
            # that there is no board to recentre a drop on.
            # Held object's OWN xy half (for the inset probe) -- the SAME size shelf_reach_err insets
            # by; its HEIGHT drives the does-it-fit filter. None height = unmeasured, which disables
            # the filter rather than guessing. Bound for EVERY container put, not only a shelved one:
            # the drop-pool bookkeeping below reads them on the bin path too.
            _ohx, _ohy, _ohz = _obj_extent(scene, held_key)
            _opath = getattr(scene.entities.get(held_key), "prim_path", None)
            _pp0s = bool(getattr(scene, "unified_geom", False))

            def _free_window(_xy, _top):
                if _window_probe_override is None:
                    return _lateral_free_window(_xy, _top, _opath, _ohz)
                return _lateral_free_window(
                    _xy,
                    _top,
                    _opath,
                    _ohz,
                    probe_override=_window_probe_override,
                )

            # INTERIOR SPACE only for [PutInside] (PLACE_INSIDE). A plain [Put] onto a CONTAINER (open bin)
            # takes the interior-drop branch below (drop onto the floor), NOT the space logic.
            if goal.prim is Primitive.PLACE_INSIDE and _spaces:
                _rtols = 0.10 if into else _REACH_TOL_M

                _base_off = _held_base_offset(scene, held_key)
                _ctop_fit = (
                    None
                    if bool((getattr(surf_e0, "meta", None) or {}).get("open_top"))
                    else getattr(surf_e0, "top_z", None)
                )

                def _probe_site(_x, _y, _top):
                    if _drop_probe_override is None:
                        return _drop_site_probe(
                            (_x, _y), _top, _opath, (_ohx, _ohy), _ohz
                        )
                    return _drop_site_probe(
                        (_x, _y),
                        _top,
                        _opath,
                        (_ohx, _ohy),
                        _ohz,
                        probe_override=_drop_probe_override,
                    )

                def _space_fits(_s):
                    return (
                        _space_usable(_s, _spaces, _ctop_fit, _ohz, _probe_site) is None
                    )

                def _raise_no_fitting_space():
                    reasons = [
                        f"{space[0]}:{reason}"
                        for space in _spaces
                        if (
                            reason := _space_usable(
                                space, _spaces, _ctop_fit, _ohz, _probe_site
                            )
                        )
                        is not None
                    ]
                    raise PrimitiveFailure(
                        PrimitiveError.TARGET_TOO_SMALL,
                        f"{held_key} fits no interior space of {surf_key}: {reasons}",
                    )

                def _space_reach_err(_s):
                    # A space is reachable IFF a POINT in it is reachable -- judged with the SHARED reach model
                    # (shelf_reach_err) so the stand-picker + this selector + the final place are IDENTICAL:
                    # plan_reach_shelf crouched under _s's own gap ceiling, at place_z = board top + the held
                    # object's base offset, probing the inset nearest the stand. Returns the error metre so
                    # _select_space takes the first space within tol (top-down), else the least-unreachable.
                    _e, _contact_probe, _plan_probe = shelf_reach_err(
                        rig,
                        placement_reach_arm,
                        _s,
                        _spaces,
                        st.root_xy,
                        st.root_yaw,
                        st.stand_z,
                        float(_s[1]) + _base_off,
                        (_ohx, _ohy),
                        _rtols,
                        floor_z=st.floor_z,
                        enabled=_pp0s,
                        clear_fn=body_clear_fn(
                            scene, st.root_xy, st.root_yaw, st.floor_z
                        ),
                        allow_colliding_fallback=False,
                        ik_budget=placement_ik_budget,
                        minimum_squat=required_crouch,
                        exact_rest_crouch=exact_rest_crouch,
                    )
                    return float("inf") if _plan_probe.get("collision_blocked") else _e

                # _sp = (name, top_z, bottom_z, cx, cy, hx, hy)
                _sp = _select_space(
                    _spaces,
                    goal,
                    scene,
                    surf_key,
                    space_err=_space_reach_err,
                    tol=_rtols,
                    fits=_space_fits,
                )
                if _sp is None:
                    _raise_no_fitting_space()
                (cx, cy) = (float(_sp[3]), float(_sp[4]))
                _shalf = (float(_sp[5]), float(_sp[6]))
                # gap CEILING = underside of the LOWEST board ABOVE this space (same xy). The putinside pose
                # crouches the whole arm UNDER it (plan_reach_shelf) so the arm enters the gap HORIZONTALLY
                # and clears the board above. inf when this is the top space (no board above).
                _shelf_gap_ceiling = _gap_ceiling_for(_sp, _spaces)
                _shelf_fit_ceiling = _fit_ceiling_for(_sp, _spaces, _ctop_fit)
                if place_z is None:
                    # rest the object's BASE on the board top (placing its ORIGIN there sinks the base below
                    # the thin board -> it falls a shelf); lift by the held object's own origin-above-base.
                    place_z = float(_sp[1]) + _held_base_offset(scene, held_key)
                _why = [
                    f"{_s[0]}:{_r}"
                    for _s, _r in (
                        (_s, _space_usable(_s, _spaces, _ctop_fit, _ohz, _probe_site))
                        for _s in _spaces
                    )
                    if _r is not None
                ]
                print(
                    f"[place-inside] {surf_key}: space {_sp[0]} "
                    f"top_z={float(_sp[1]):.2f} obj_h={(_ohz or 0) * 1000:.0f}mm "
                    f"(of {[s[0] for s in _spaces]})"
                    + (f" UNUSABLE={_why}" if _why else "")
                    + (
                        f" rejected={sorted(goal.meta['_space_rejected'])}"
                        if goal.meta.get("_space_rejected")
                        else ""
                    )
                )
            else:
                (cx, cy) = (
                    surf_e0.xy
                    if surf_e0.xy is not None
                    else scene.surface_top(surf_key)[0]
                )
                _drop = getattr(surf_e0, "drop_z", None)
                if place_z is None:
                    place_z = float(
                        _drop if _drop is not None else surf_e0.z + surf_e0.height
                    )
            if place_xy is None:
                _rtol = 0.10 if into else _REACH_TOL_M

                def _reach_ok(_xy):
                    return _placement_reachable(
                        scene,
                        rig,
                        placement_reach_arm,
                        _xy,
                        place_z,
                        st,
                        _rtol,
                        _shelf_gap_ceiling,
                        ik_budget=placement_ik_budget,
                        minimum_squat=required_crouch,
                        exact_rest_crouch=exact_rest_crouch,
                    )

                def _reach_many(_points):
                    return _placement_reachable_many(
                        scene,
                        rig,
                        placement_reach_arm,
                        _points,
                        place_z,
                        st,
                        _rtol,
                        _shelf_gap_ceiling,
                        ik_budget=placement_ik_budget,
                        minimum_squat=required_crouch,
                        exact_rest_crouch=exact_rest_crouch,
                    )

                # A top-opening bin is reached at its rim, not at the deep release height. Its XY choice is
                # therefore purely a capacity decision; the one final reach solve below validates that choice
                # at the hover pose and the existing re-navigation path can move the person if needed.
                # ``handle_xy is None`` alone does NOT mean "top-opening bin": a DOOR-LESS unit with real
                # interior boards -- an open deck, an open shelf rack -- has no handle either, and treating
                # one as a bin would throw away the space branch just resolved.
                _top_bin = (
                    into
                    and _shelf_gap_ceiling is None
                    and getattr(surf_e0, "top_z", None) is not None
                    and (
                        getattr(surf_e0, "handle_xy", None) is None
                        or (getattr(surf_e0, "meta", None) or {}).get("openable_kind")
                        == "lid"
                    )
                )
                _retry_rejected_xy = (
                    goal.meta.get("_placement_rejected_xy")
                    or goal.meta.get("_inside_rejected_xy")
                    or ()
                )
                if _top_bin:
                    place_xy, _tier = _select_container_xy(
                        scene,
                        surf_key,
                        surf_e0,
                        (cx, cy),
                        st,
                        lambda _xy: True,
                        deepest=True,
                        rejected=_retry_rejected_xy,
                        occupancy_relation="INSIDE",
                        occupancy_z=(
                            float(surf_e0.z),
                            float(surf_e0.z) + float(surf_e0.height),
                        ),
                        allow_stacking=not _requires_direct_goal_relation(
                            scene, "inside", held_key, surf_key
                        ),
                        placement_meta=goal.meta,
                    )
                    if place_xy is None:
                        raise PrimitiveFailure(
                            PrimitiveError.NO_FREE_PLACEMENT,
                            f"{surf_key} has no free interior footprint for {held_key} "
                            f"(selection tier {_tier})",
                        )
                else:
                    # Live-physics site test, SHELVED drops only: a board top is a plane the object is
                    # meant to rest ON, so "is there a surface right here" is a meaningful question. A
                    # top-open bin's drop is deliberately mid-air above the interior floor (the object
                    # falls in), so the same question would reject every candidate -- hence None there,
                    # which also leaves bin drops unaffected by the site test.
                    # SHELF-FIRST, PILE LAST. _first_drop_xy reports which tier it had to settle for;
                    # tier 0 is a clean drop (clear of other objects, on real geometry, reachable). Anything
                    # worse means this board cannot take the object well -- so take ANOTHER BOARD rather
                    # than accept the compromise, and only once every board has been tried do we keep the
                    # least bad one. That ordering is what stops a PutInside from piling objects on top of
                    # each other while a perfectly good empty shelf sits one level down: the stacking lift
                    # further below is reached only on a compromised tier, i.e. only after this loop has
                    # exhausted the container. Bounded by the space count; never vetoes.
                    _tier = 4
                    _pool_key = surf_key
                    for _sp_try in range(max(1, len(_spaces))):
                        _rej = set(goal.meta.get("_space_rejected") or ())
                        _site_probe = None
                        _embed_probe = None
                        if _shelf_gap_ceiling is not None and _sp is not None:
                            # ONE probe per xy, read twice: "can it rest here" and "is this point inside
                            # solid geometry". Cached together so adding the second question costs nothing.
                            _pcache: dict = {}

                            def _probe_at(_xy, _top=float(_sp[1]), _c=_pcache):
                                k = (round(_xy[0], 4), round(_xy[1], 4))
                                if k not in _c:
                                    _c[k] = _probe_site(_xy[0], _xy[1], _top)
                                return _c[k]

                            def _probe_xy(_xy):
                                _pr = _probe_at(_xy)
                                return True if _pr is None else bool(_pr["ok"])

                            def _embed_xy(_xy):
                                _pr = _probe_at(_xy)
                                # None = unmeasured, which must read as "not embedded" so an unprobeable
                                # scene keeps exactly the probe-free choice.
                                return (
                                    False if _pr is None else bool(_pr.get("embedded"))
                                )

                            _site_probe = _probe_xy
                            _embed_probe = _embed_xy
                        _pool_key = (surf_key, _sp[0]) if _sp is not None else surf_key
                        place_xy, _tier = _select_container_xy(
                            scene,
                            surf_key,
                            surf_e0,
                            (cx, cy),
                            st,
                            _reach_ok,
                            half_override=_shalf,
                            deepest=False,
                            site_ok=_site_probe,
                            embedded_ok=_embed_probe,
                            rejected=_retry_rejected_xy,
                            occupancy_key=_pool_key,
                            occupancy_relation="INSIDE",
                            occupancy_z=(
                                float(_sp[2]),
                                float(_shelf_gap_ceiling),
                            )
                            if _sp is not None
                            else None,
                            commit=False,
                            nearest=True,
                            raise_too_small=False,
                            strict=(
                                _sp is None or _strict_shelf_placement(_spaces, _rej)
                            ),
                            allow_stacking=bool(front_drum_into and not _spaces),
                            placement_meta=goal.meta,
                            reach_many=(
                                _reach_many
                                if _pp0s and _batch_reach_supported(rig)
                                else None
                            ),
                        )
                        if (
                            _tier == 0
                            or _shelf_gap_ceiling is None
                            or _sp is None
                            or len(_rej) + 1 >= len(_spaces)
                        ):
                            break
                        _rej.add(_sp[0])
                        goal.meta["_space_rejected"] = _rej
                        print(
                            f"[place-shelf-first] {surf_key}: {_sp[0]} has no clean reachable drop "
                            f"(tier {_tier}: 1=occupied, 2=unsupported, 3=embedded, "
                            f"4=unreachable, 5=too-small) -> "
                            f"try another board ({len(_rej)}/{len(_spaces)} rejected)"
                        )
                        _sp = _select_space(
                            _spaces,
                            goal,
                            scene,
                            surf_key,
                            space_err=_space_reach_err,
                            tol=_rtols,
                            fits=_space_fits,
                        )
                        if _sp is None:
                            _raise_no_fitting_space()
                        (cx, cy) = (float(_sp[3]), float(_sp[4]))
                        _shalf = (float(_sp[5]), float(_sp[6]))
                        _shelf_gap_ceiling = _gap_ceiling_for(_sp, _spaces)
                        _shelf_fit_ceiling = _fit_ceiling_for(_sp, _spaces, _ctop_fit)
                        place_z = float(_sp[1]) + _held_base_offset(scene, held_key)
                    if place_xy is None:
                        error = (
                            PrimitiveError.TARGET_TOO_SMALL
                            if _tier == 5
                            else PrimitiveError.NO_FREE_PLACEMENT
                        )
                        raise PrimitiveFailure(
                            error,
                            f"{surf_key} has no free supported interior footprint for {held_key} "
                            f"after checking {len(_spaces)} space(s) (selection tier {_tier})",
                        )
                    place_z = _validated_front_drum_pile_z(
                        scene,
                        held_key,
                        surf_key,
                        surf_e0,
                        place_xy,
                        place_z,
                        _ohz,
                        enabled=bool(front_drum_into and not _spaces and _tier == 1),
                    )
                    # The chosen xy came from a candidate walk over an AUTHORED footprint; nothing in that
                    # walk knows how much room the object actually has THERE. Measure it now, on the live
                    # colliders, and slide a drop that does not fit into the middle of what is free.
                    # A BOARD only: with no space there is no plane to slide along, and the interior drop
                    # is a mid-air release the settle resolves.
                    if _sp is not None:
                        _placement_half = (
                            (_ohy, _ohx)
                            if abs(
                                float(goal.meta.get("_placement_fit_yaw_rad") or 0.0)
                            )
                            > 1e-9
                            else (_ohx, _ohy)
                        )
                        place_xy = _recentre_free_xy(
                            place_xy,
                            _free_window(place_xy, float(_sp[1])),
                            _placement_half,
                            f"{surf_key}/{_sp[0]}",
                        )
                    _record_placement_footprint(
                        scene,
                        _pool_key,
                        place_xy,
                        (
                            (_ohy, _ohx)
                            if abs(
                                float(goal.meta.get("_placement_fit_yaw_rad") or 0.0)
                            )
                            > 1e-9
                            else (_ohx, _ohy)
                        ),
                        _opath,
                    )
                    # What is actually under the point we are about to release over. The space-level probe
                    # above judges the space CENTRE; this one judges the CHOSEN drop, which is what physics
                    # will answer for. Logged always (not only on rejection) so a drop that later turns out
                    # to have fallen can be told apart from one that was never on anything.
                    # ONE-SHOT MEASUREMENT: does this object's physics body match the box we measured?
                    # Everything above reasons from the VISUAL bbox; PhysX moves a generated collision
                    # approximation. Logged per placement so the two can be compared directly.
                    if _opath and getattr(scene, "stage", None) is not None:
                        try:
                            import fpv_scene as _fs_cm
                            import physics_human as _ph_cm

                            _cb = _fs_cm.compute_world_bbox(
                                scene.stage.GetPrimAtPath(_opath)
                            )
                            _cmn, _cmx = _cb.GetMin(), _cb.GetMax()
                            _cm = _ph_cm.collider_margin(
                                _opath,
                                (
                                    0.5 * float(_cmn[0] + _cmx[0]),
                                    0.5 * float(_cmn[1] + _cmx[1]),
                                    0.5 * float(_cmn[2] + _cmx[2]),
                                ),
                                (_ohx, _ohy),
                                _ohz or 0.2,
                            )
                            print(
                                f"[collider-vs-visual] {held_key}: visual half="
                                f"({_ohx * 1000:.0f},{_ohy * 1000:.0f})mm h={(_ohz or 0) * 1000:.0f}mm"
                                f" | collider sticks out x={_cm['x_mm']} y={_cm['y_mm']} mm"
                                f" (dirs answering {_cm['n']}/4)"
                            )
                        except Exception as _exc:  # noqa: BLE001
                            print(
                                f"[collider-vs-visual] {held_key}: probe failed {_exc}"
                            )
                    # BOARD only, for the same reason the recentre above is: the seat measurement is
                    # "how far below the board top does this object actually come to rest", and a
                    # container with no extracted boards has no board to measure against.
                    if _sp is not None:
                        _fpr = _probe_site(place_xy[0], place_xy[1], float(_sp[1]))
                        if _fpr is not None:
                            print(
                                f"[place-site] {surf_key}/{_sp[0]} "
                                f"drop=({place_xy[0]:.3f},{place_xy[1]:.3f}) "
                                f"board_z={float(_sp[1]):.3f} -> {_fpr['reason']} "
                                f"hit_z={_fpr['hit_z']} support_frac={_fpr['support_frac']} "
                                f"on={_fpr['support_path']} blockers={_fpr['blockers']}"
                            )
                        place_z = _measured_seat_z(
                            _fpr,
                            float(_sp[1]),
                            _base_off,
                            _opath,
                            place_z,
                            goal.place_z is not None,
                            f"{surf_key}/{_sp[0]}",
                        )
        else:
            (sx, sy), stop = scene.surface_top(surf_key)
            _surface_base = _held_base_offset(scene, held_key)
            place_z = place_z if place_z is not None else stop + _surface_base
            if place_xy is None:
                _sohx, _sohy, _sohz = _obj_extent(scene, held_key)
                _sopath = getattr(scene.entities.get(held_key), "prim_path", None)
                _surface_probe_cache: dict = {}

                def _surface_probe(_xy):
                    probe_half, probe_z = _surface_probe_profile(
                        surf_e0,
                        held_key,
                        _xy,
                        (_sohx, _sohy),
                        stop,
                        goal.meta,
                    )
                    key = (
                        round(_xy[0], 4),
                        round(_xy[1], 4),
                        round(probe_z, 4),
                        round(probe_half[0], 4),
                        round(probe_half[1], 4),
                    )
                    if key not in _surface_probe_cache:
                        if _drop_probe_override is None:
                            _surface_probe_cache[key] = _drop_site_probe(
                                _xy,
                                probe_z,
                                _sopath,
                                probe_half,
                                _sohz,
                            )
                        else:
                            _surface_probe_cache[key] = _drop_site_probe(
                                _xy,
                                probe_z,
                                _sopath,
                                probe_half,
                                _sohz,
                                probe_override=_drop_probe_override,
                            )
                    return _surface_probe_cache[key]

                def _surface_site(_xy):
                    probe = _surface_probe(_xy)
                    return _surface_site_allows_lid_bridge(
                        scene,
                        held_key,
                        surf_key,
                        surface_view,
                        container_view,
                        probe,
                    )

                def _surface_embedded(_xy):
                    probe = _surface_probe(_xy)
                    return False if probe is None else bool(probe.get("embedded"))

                def _surface_reach_many(_points):
                    return _placement_reachable_many(
                        scene,
                        rig,
                        placement_reach_arm,
                        _points,
                        place_z,
                        st,
                        _REACH_TOL_M,
                        ik_budget=placement_ik_budget,
                    )

                place_xy, _surface_tier = _select_container_xy(
                    scene,
                    surf_key,
                    surf_e0,
                    (sx, sy),
                    st,
                    lambda xy: _placement_reachable(
                        scene,
                        rig,
                        placement_reach_arm,
                        xy,
                        place_z,
                        st,
                        _REACH_TOL_M,
                        ik_budget=placement_ik_budget,
                    ),
                    site_ok=_surface_site,
                    embedded_ok=_surface_embedded,
                    occupancy_relation="ON",
                    rejected=goal.meta.get("_placement_rejected_xy") or (),
                    allow_stacking=not _requires_direct_goal_relation(
                        scene, "on", held_key, surf_key
                    ),
                    placement_meta=goal.meta,
                    reach_many=(
                        _surface_reach_many
                        if bool(getattr(scene, "unified_geom", False))
                        and _batch_reach_supported(rig)
                        else None
                    ),
                )
                if place_xy is None:
                    raise PrimitiveFailure(
                        PrimitiveError.NO_FREE_PLACEMENT,
                        f"{surf_key} has no free supported surface footprint for {held_key} "
                        f"(selection tier {_surface_tier})",
                    )
                _chosen_surface_probe = _surface_probe(place_xy)
                if _chosen_surface_probe is not None and not bool(
                    _chosen_surface_probe.get("ok")
                ):
                    print(
                        f"[place-lid-bridge] {held_key} spans the opening of {surf_key}; "
                        "using the target's measured top rather than requiring center support"
                    )
                place_z = _measured_seat_z(
                    _chosen_surface_probe,
                    stop,
                    _surface_base,
                    _sopath,
                    place_z,
                    goal.place_z is not None,
                    surf_key,
                )
    elif floor_placement:
        # ``PutOnFloor`` supplies an explicit, grounded pose beside its anchor.  It
        # deliberately has no surface entity: the choreography below uses the same
        # low-reach/crouch path as every other placement, while the symbolic effect
        # remains free-standing rather than inventing ON(obj, anchor).
        pass
    else:
        (sx, sy), stop = scene.surface_top(surf_key)
        place_xy = place_xy if place_xy is not None else (sx, sy)
        place_z = place_z if place_z is not None else stop
    if place_xy is None or place_z is None:
        raise PrimitiveFailure(
            PrimitiveError.PLACE_UNSTABLE, "PLACE without a resolvable target"
        )
    goal.meta["_placement_chosen_xy"] = [float(place_xy[0]), float(place_xy[1])]
    if isinstance(goal.meta.get("_placement_intent"), dict):
        goal.meta["_placement_intent"]["chosen_xy"] = list(
            goal.meta["_placement_chosen_xy"]
        )
    if goal.prim is Primitive.PLACE_INSIDE:
        goal.meta["_inside_chosen_xy"] = [float(place_xy[0]), float(place_xy[1])]

    place_z = _stack_lift_z(
        scene,
        held_key,
        surf_key,
        surf_e0,
        place_xy,
        place_z,
        _shelf_gap_ceiling,
    )

    if (
        front_access_into
        and surf_e0 is not None
        and _low_front_openable(scene, surf_e0, target_z=place_z)
    ):
        low_front_into = True
        keep_crouched_into = True
        required_crouch = max(
            _LOW_FRONT_INSERT_MIN_SQUAT,
            active_crouch,
        )

    front_insert_geometry = None
    if front_access_into:
        front_insert_geometry = _front_insertion_waypoints(
            surf_e0,
            np.array([place_xy[0], place_xy[1], place_z], dtype=np.float64),
            (_ohx, _ohy),
            _ohz,
        )
        _pre_insert, aligned_target, _entry_axis = front_insert_geometry
        place_xy = (float(aligned_target[0]), float(aligned_target[1]))
        place_z = float(aligned_target[2])
        goal.meta["_placement_chosen_xy"] = [place_xy[0], place_xy[1]]
        if isinstance(goal.meta.get("_placement_intent"), dict):
            goal.meta["_placement_intent"]["chosen_xy"] = [place_xy[0], place_xy[1]]
        goal.meta["_inside_chosen_xy"] = [place_xy[0], place_xy[1]]

    # Physics-true placement: by DEFAULT reach the hand so the object arrives AT its rest (place_z),
    # exactly like an ontop place -- the object is then released WHERE the hand delivered it, so there is no
    # hover-then-snap teleport. Only when the rest is genuinely NOT reachable directly (a deep/low
    # TOP-opening bin with no front opening) do we fall back to hovering over the rim and dropping in. The
    # switch is the pose-planner's REACHABILITY, not a container-type marker -- so a front-open shelf whose
    # rest the arm can reach places straight into it (unifying inside with ontop).
    reach_z = place_z
    obj_world = np.array([place_xy[0], place_xy[1], reach_z], dtype=np.float64)
    # POSE-PLANNER (gated scene.unified_geom): pick the least FORWARD-LEAN that reaches the drop target
    # (real 3D IK), so a FAR place leans in instead of failing OUT_OF_REACH. The SAME lean drives the reach
    # solve AND the per-frame spine bend (lf) below, so the torso and arm stay consistent.
    _pp_on = bool(getattr(scene, "unified_geom", False))
    _place_reach_tol = _placement_reach_tolerance(
        goal,
        into=into,
        exact_rest_crouch=exact_rest_crouch,
    )
    _egocentric_reach_fallback = "_egocentric_reach_tol_m" in (goal.meta or {})
    if (
        _shelf_gap_ceiling is not None
    ):  # shelved PutInside: crouch the WHOLE arm into the target gap
        _plan = plan_reach_shelf(
            rig,
            placement_reach_arm,
            obj_world,
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            _place_reach_tol,
            _shelf_gap_ceiling,
            floor_z=st.floor_z,
            enabled=_pp_on,
            clear_fn=body_clear_fn(
                scene,
                st.root_xy,
                st.root_yaw,
                st.floor_z,
                require_body_clear=exact_rest_crouch,
            ),
            allow_colliding_fallback=_egocentric_reach_fallback,
            minimum_squat=required_crouch,
            exact_rest_crouch=exact_rest_crouch,
        )
    else:
        _plan = plan_reach(
            rig,
            placement_reach_arm,
            obj_world,
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            _place_reach_tol,
            floor_z=st.floor_z,
            enabled=_pp_on,
            clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
            allow_colliding_fallback=_egocentric_reach_fallback,
            upright_first=keep_crouched_into,
            minimum_squat=required_crouch,
        )
    if exact_rest_crouch:
        print(
            f"[place-exact-rest-crouch] {surf_key}: target_z={place_z:.3f} "
            f"lean={math.degrees(float(_plan.get('lean_rad') or 0.0)):.0f}deg "
            f"squat={float(_plan.get('squat') or 0.0):.1f} "
            f"root_drop={float(_plan.get('root_drop_m') or 0.0):.2f}m "
            f"err={float(_plan.get('err') or 0.0) * 1000:.1f}mm "
            f"reachable={bool(_plan.get('reachable'))}"
        )
    if (
        into
        and _pp_on
        and not _plan["reachable"]
        and surf_e0 is not None
        and _shelf_gap_ceiling
        is None  # a chosen BOARD is placed ON, never dropped in from above
        and (
            getattr(surf_e0, "handle_xy", None) is None
            or (getattr(surf_e0, "meta", None) or {}).get("openable_kind") == "lid"
        )
        and getattr(surf_e0, "top_z", None) is not None
    ):
        # rest unreachable + TOP-opening bin (no front opening): hover over the rim and drop in from above --
        # reaching straight down into a short, wide floor box at arm's length is a near-impossible low-forward
        # reach. max() keeps a TALL bin's own rim + a chest-height comfortable dip (unchanged bin behaviour).
        _ct = surf_e0
        reach_z = max(
            float(place_z),
            float(_ct.top_z) + 0.22,
            float(getattr(scene, "floor_z", 0.0)) + 0.82,
        )
        obj_world = np.array([place_xy[0], place_xy[1], reach_z], dtype=np.float64)
        _plan = plan_reach(
            rig,
            placement_reach_arm,
            obj_world,
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            _place_reach_tol,
            floor_z=st.floor_z,
            enabled=_pp_on,
            clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
            allow_colliding_fallback=_egocentric_reach_fallback,
        )
    if (
        into
        and _sp is not None
        and goal.place_z is None
        and surf_e0 is not None
        and _front_open_release_needed(_plan, low_front=low_front_into)
    ):
        exact_rest_requires_head_bow = bool(
            low_front_into
            and _plan.get("reachable")
            and not _plan.get("collision_blocked")
        )
        if exact_rest_requires_head_bow:
            _plan = dict(_plan, collision_blocked=True)
        reach_ceiling = (
            float(_shelf_fit_ceiling)
            if _shelf_fit_ceiling is not None
            else float("inf")
        )
        for release_z in front_open_release_heights(
            float(place_z),
            reach_ceiling,
            _ohz,
            _base_off,
            getattr(surf_e0, "handle_z", None),
        ):
            release_world = np.array(
                [place_xy[0], place_xy[1], release_z], dtype=np.float64
            )
            release_plan = _plan_front_open_release(
                rig,
                placement_reach_arm,
                release_world,
                st,
                _place_reach_tol,
                reach_ceiling,
                low_front=low_front_into,
                enabled=_pp_on,
                clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
                allow_colliding_fallback=_egocentric_reach_fallback,
                minimum_squat=required_crouch,
            )
            if _front_open_release_plan_is_safe(
                release_plan,
                low_front=low_front_into,
            ):
                print(
                    f"[place-front-drop] {surf_key}: exact rest z={place_z:.3f} "
                    + (
                        "requires an unsafe forward head bow; "
                        if exact_rest_requires_head_bow
                        else "is not safely reachable; "
                    )
                    + f"release inside the opening at z={release_z:.3f} and let physics "
                    "settle onto the measured support"
                )
                place_z = release_z
                reach_z = release_z
                obj_world = release_world
                _plan = release_plan
                goal.meta["_front_open_release_z"] = release_z
                break
    if _plan.get("collision_blocked"):
        _prepare_placement_reach_retry(
            scene,
            goal,
            held_key,
            surf_key,
            place_xy,
            place_z,
            _shelf_gap_ceiling,
        )
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"PLACE {held_key!r} needs a different stand for a safe reach",
        )
    if not _plan["reachable"]:
        _prepare_placement_reach_retry(
            scene,
            goal,
            held_key,
            surf_key,
            place_xy,
            place_z,
            _shelf_gap_ceiling,
        )
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"PLACE {held_key!r} has no valid reach pose from the current stand",
        )
    _place_lean = _plan["lean_rad"] if _pp_on else None
    # POSE-PLANNER crouch: for a LOW drop the planner returns a squat + crouch-pinned stand_z; the PLACE
    # must SOLVE + RENDER from that crouch (mirrors produce_pick) or the standing arm cannot reach DOWN into
    # a low/floor container. 0.0 / st.stand_z when the planner reached upright.
    use_planned_posture = _pp_on or keep_crouched_into
    _place_squat = float(_plan["squat"]) if use_planned_posture else 0.0
    _place_stand_z = float(_plan["stand_z"]) if use_planned_posture else st.stand_z
    if _plan.get("arm_pose") is not None and _plan.get("err") is not None:
        place_arm = wpp._arm_from_pose(_plan["arm_pose"])
        err = float(_plan["err"])
    else:
        place_arm, err = _solve_reach_arm(
            rig,
            placement_reach_arm,
            obj_world,
            st.root_xy,
            st.root_yaw,
            _place_stand_z,
            lean_override=_place_lean,
            squat_frac=_place_squat,
        )
    print(
        f"[place] seat_err={err * 1000:.1f}mm target={tuple(round(float(x), 3) for x in obj_world)}"
    )
    # A container INTERIOR is forgiving: the object is released AT the target (obj_drop) regardless of a
    # small hand residual, and a ~10 cm miss still lands inside a bin -- so relax the reach tol for into
    # drops. A surface top stays EXACT.
    place_tol = 0.10 if into else _REACH_TOL_M
    # (No "near-interior retarget" that YANKS an out-of-reach drop toward the stand -- that pushes objects
    # onto the front edge / out of the opening. The selector picks a reachable point DIRECTLY
    # (deepest-reachable), so a genuinely unreachable drop just fails honestly at the OUT_OF_REACH gate
    # below instead of being dragged to the front.)
    # A FRONT-opening openable container (handle_xy set) reaches INTO a deep shelf: its interior is even more
    # forgiving than a top bin (the object is released AT the target, which is well inside), so accept a ~14 cm
    # residual. Top-opening bins (no handle_xy) keep the 0.10 gate.
    final_tol = place_tol
    if into and surf_e0 is not None and getattr(surf_e0, "handle_xy", None) is not None:
        final_tol = _FRONT_OPENING_PLACE_TOL_M
    final_tol = _goal_reach_tolerance(goal, final_tol)
    if err > final_tol:
        _prepare_placement_reach_retry(
            scene,
            goal,
            held_key,
            surf_key,
            place_xy,
            place_z,
            _shelf_gap_ceiling,
        )
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH, f"place seat_err={err * 1000:.0f}mm"
        )

    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    obj_drop = (float(place_xy[0]), float(place_xy[1]), float(place_z))
    fit_yaw = float(goal.meta.get("_placement_fit_yaw_rad") or 0.0)
    place_upright = bool(
        into
        and surf_e0 is not None
        and (getattr(surf_e0, "meta", None) or {}).get("place_upright", False)
    )
    obj_rest_value = _rest_value_for_drop(
        st.held,
        obj_drop,
        local_yaw_rad=fit_yaw,
        upright=place_upright,
    )
    if place_upright:
        up = np.asarray(obj_rest_value, dtype=np.float64)[:3, 2]
        up_norm = float(np.linalg.norm(up))
        tilt_deg = math.degrees(
            math.acos(float(np.clip(up[2] / max(up_norm, 1e-9), -1.0, 1.0)))
        )
        print(
            f"[place-upright] {held_key}: "
            f"up={tuple(round(float(value), 4) for value in up)} "
            f"tilt={tilt_deg:.2f}deg dynamic=False"
        )
    has_rest_pose = np.asarray(obj_rest_value).shape == (4, 4)
    obj_rest_state = ("rest", obj_rest_value)
    _, lf = _lean_for_height(
        float(place_z)
    )  # lean for a low drop, stand tall for a high shelf
    if _pp_on:
        lf = _plan[
            "lf"
        ]  # POSE-PLANNER: match the reach lean so torso + arm stay consistent
    prior_walk_end_yaw = st.walk_end_yaw
    prior_last_pose156 = (
        None
        if st.last_pose156 is None
        else np.asarray(st.last_pose156, dtype=np.float64).copy()
    )
    rotation_frames, otw = _carry_yaw_frames(st, otw, fit_yaw)
    frames = [
        *rotation_frames,
        *_arrive_beats(
            rig,
            st,
            grasp_arm,
            1.0,
            ("carry", otw),
            squat_frac=_place_squat if keep_crouched_into else 0.0,
        ),
    ]

    # A transported subtree must keep one kinematic root through release: independent PhysX ownership of
    # the root would leave its carried children without an authoritative transform for those frames.
    fdp = getattr(scene, "free_dynamic_prims", None)
    held_entity = scene.resolve(held_key)
    held_path = str(held_entity.prim_path or "/World/Obj").rstrip("/")
    fdp_obj = (
        bool(fdp)
        and held_path in {str(p).rstrip("/") for p in fdp}
        and not _carried
        and not place_upright
    )

    front_insert_openable_key = _front_insert_openable_key(
        scene,
        surf_key,
        access_mode,
    )

    def add(
        pose,
        obj_state,
        seg,
        free_dynamic=False,
        root_z=None,
        gaze=None,
        root_xy=None,
        root_yaw=None,
    ):
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy if root_xy is None else root_xy,
                st.root_yaw if root_yaw is None else float(root_yaw),
                st.stand_z if root_z is None else float(root_z),
                fpv_mode="look",
                obj=obj_state,
                seg=seg,
                free_dynamic=bool(free_dynamic),
                obj_path=held_path if held_path != "/World/Obj" else None,
                gaze=(
                    np.asarray(gaze, dtype=np.float64) if gaze is not None else None
                ),
                openable_key=(
                    front_insert_openable_key
                    if str(seg).startswith("front_insert")
                    else None
                ),
            )
        )

    surf_e = surf_e0
    # STRICT: the shelf frame-gen fires iff a shelf SPACE was actually chosen (PLACE_INSIDE + spaces) -- the
    # same condition that set _shelf_gap_ceiling (and cx/cy/_shalf/place_z). A plain [Put] onto a container
    # that merely exposes spaces keeps the bin/collapse path; this also avoids a latent unbound cx/cy.
    _shelved_into = _shelf_gap_ceiling is not None
    if front_access_into:
        pre_insert, insert_target, entry_axis = front_insert_geometry
        aperture_center = np.asarray(
            (getattr(surf_e, "meta", None) or {}).get("aperture_center"),
            dtype=np.float64,
        )
        insert_gaze = (
            aperture_center
            if low_front_into
            and aperture_center.shape == (3,)
            and np.all(np.isfinite(aperture_center))
            else insert_target
        )
        if low_front_into:
            insert_gaze = np.asarray(insert_gaze, dtype=np.float64).copy()
            insert_gaze[2] += float(
                (getattr(surf_e, "meta", None) or {}).get(
                    "content_gaze_lift_m",
                    0.0,
                )
            )
        goal.meta["_front_insert_path"] = [
            [float(value) for value in pre_insert],
            [float(value) for value in insert_target],
        ]
        arm_pre, _ = _solve_reach_arm(
            rig,
            grasp_arm,
            pre_insert,
            st.root_xy,
            st.root_yaw,
            _place_stand_z,
            lean_override=_place_lean,
            squat_frac=_place_squat,
        )
        boxes = _into_obstacle_boxes(scene, surf_key, surf_e)
        points = _fwd_arm_points(
            _manip(rig, place_arm, 1.0, lf, _place_squat),
            *wpp._grasp_frame(st.root_xy, st.root_yaw, _place_stand_z),
        )
        depth, hit = _verts_in_boxes(points, boxes)
        _require_arm_clear(scene, depth, hit, "front-opening insertion")
        if keep_crouched_into:
            for index in range(_FRONT_DRUM_CROUCH_FRAMES):
                fraction = wpp._smoothstep(
                    (index + 1) / _FRONT_DRUM_CROUCH_FRAMES
                )
                squat = _place_squat
                pose = _manip(rig, grasp_arm, 1.0, 0.0, squat)
                add(
                    pose,
                    ("carry", otw),
                    "front_insert_crouch",
                    root_z=_pin_z(pose, st.floor_z),
                    gaze=insert_gaze,
                )
        n_pre = max(1, int(round(_N_PLACE * 0.45)))
        for index in range(n_pre):
            fraction = wpp._smoothstep((index + 1) / n_pre)
            squat = _place_squat if keep_crouched_into else _place_squat * fraction
            pose = _manip(
                rig,
                core.lerp_arm(grasp_arm, arm_pre, fraction),
                1.0,
                lf * fraction,
                squat,
            )
            root_z = _pin_z(pose, st.floor_z) if squat > 1e-6 else st.stand_z
            add(
                pose,
                ("carry", otw),
                "front_insert",
                root_z=root_z,
                gaze=insert_gaze,
            )
        for index in range(_N_PLACE - n_pre):
            fraction = wpp._smoothstep((index + 1) / (_N_PLACE - n_pre))
            released = fraction >= 0.9
            pose = _manip(
                rig,
                core.lerp_arm(arm_pre, place_arm, fraction),
                1.0,
                lf,
                _place_squat,
            )
            add(
                pose,
                obj_rest_state if released else ("carry", otw),
                "front_insert",
                free_dynamic=released and fdp_obj,
                root_z=_place_stand_z,
                gaze=insert_gaze,
            )
        post_insert_root = None
        post_insert_yaw = None
        if access_mode == "front_opening" and not low_front_into:
            post_insert_root, post_insert_yaw = _front_opening_post_interaction_pose(
                scene,
                surf_key,
                surf_e,
                st.root_xy,
                st.root_yaw,
            )
        yaw_delta = (
            (post_insert_yaw - st.root_yaw + 180.0) % 360.0 - 180.0
            if post_insert_yaw is not None
            else 0.0
        )
        for index in range(_N_RETRACT):
            fraction = wpp._smoothstep((index + 1) / _N_RETRACT)
            squat = _place_squat if keep_crouched_into else _place_squat * (1.0 - fraction)
            pose = _manip(
                rig,
                core.lerp_arm(place_arm, arm_rest, fraction),
                1.0 - fraction,
                lf * (1.0 - fraction),
                squat,
            )
            root_z = _pin_z(pose, st.floor_z) if squat > 1e-6 else st.stand_z
            add(
                pose,
                obj_rest_state,
                "front_insert_retract",
                free_dynamic=fdp_obj,
                root_z=root_z,
                gaze=insert_gaze,
                root_xy=(
                    (
                        st.root_xy[0]
                        + (post_insert_root[0] - st.root_xy[0]) * fraction,
                        st.root_xy[1]
                        + (post_insert_root[1] - st.root_xy[1]) * fraction,
                    )
                    if post_insert_root is not None
                    else None
                ),
                root_yaw=(
                    st.root_yaw + yaw_delta * fraction
                    if post_insert_yaw is not None
                    else None
                ),
            )
        if post_insert_root is not None:
            st.root_xy = post_insert_root
            st.root_yaw = post_insert_yaw
            st.walk_end_yaw = post_insert_yaw
        goal.meta["_front_insert_axis"] = [float(value) for value in entry_axis]
    elif _shelved_into:
        # SHELVED container (fridge/cabinet/bookcase): stage the hand at the OPENING (pre), then move it
        # HORIZONTALLY into the target gap and back out -- the arm enters THROUGH the opening at shelf height
        # instead of dipping DOWN through the boards (a straight direct reach would sweep the arm through the
        # shelves). Keeps the crouch for low shelves. Reach is checked against other furniture (this
        # container's OWN outer box excluded so the hand can enter) + ALL its shelf boards + its OPEN door
        # leaf(s); the horizontal in/out is the AVOIDANCE, _verts_in_boxes reports any residual clip.
        boxes = _into_obstacle_boxes(scene, surf_key, surf_e)
        wp = plan_reach_waypoints(scene, surf_key, st.root_xy, obj_world, entity=surf_e)
        # (b) push the PRE waypoint OUTSIDE the container's near face, so the hand DESCENDS in free space in
        # FRONT of the opening and only THEN moves horizontally in (the default pre can sit just inside the
        # opening -> the diagonal descent clips the front edge / a board). standoff = contact->near-face + margin.
        _appr = np.asarray(wp["approach"], dtype=np.float64)
        _fc = surf_e.foot_xy if surf_e.foot_xy is not None else (float(cx), float(cy))
        _fh = surf_e.foot_half if surf_e.foot_half is not None else (0.0, 0.0)
        _face = abs(_appr[0]) * float(_fh[0]) + abs(_appr[1]) * float(_fh[1])
        _c2c = (float(obj_world[0]) - float(_fc[0])) * _appr[0] + (
            float(obj_world[1]) - float(_fc[1])
        ) * _appr[1]
        _standoff = max(
            float(np.linalg.norm(np.asarray(wp["pre"])[:2] - obj_world[:2])),
            _c2c + _face + 0.12,
        )
        wp["pre"] = np.array(
            [
                float(obj_world[0]) - _appr[0] * _standoff,
                float(obj_world[1]) - _appr[1] * _standoff,
                float(obj_world[2]),
            ],
            dtype=np.float64,
        )
        arm_pre, _ = _solve_reach_arm(
            rig,
            grasp_arm,
            wp[
                "pre"
            ],  # OUTSIDE the front face, at the shelf height -> descend outside, then horizontal in
            st.root_xy,
            st.root_yaw,
            _place_stand_z,
            lean_override=_place_lean,
            squat_frac=_place_squat,
        )
        _dp, _hp = _verts_in_boxes(
            _fwd_arm_points(
                _manip(rig, place_arm, 1.0, lf, _place_squat),
                *wpp._grasp_frame(st.root_xy, st.root_yaw, _place_stand_z),
            ),
            boxes,
        )
        print(
            f"[reach] putinside place: err={err * 1000:.1f}mm "
            f"arm-clear worst={_dp * 1000:.0f}mm hit={_hp or '-'}"
        )
        n_pre = max(1, int(round(_N_PLACE * 0.4)))
        for i in range(_N_PLACE):
            if i < n_pre:  # carry -> pre (opening), still holding; crouch ramps IN
                s = wpp._smoothstep((i + 1) / n_pre)
                _sf = _place_squat * s
                _pose = _manip(
                    rig, core.lerp_arm(grasp_arm, arm_pre, s), 1.0, lf * s, _sf
                )
                _rz = _pin_z(_pose, st.floor_z) if _sf > 1e-6 else st.stand_z
                add(_pose, ("carry", otw), "place", root_z=_rz)
            else:  # pre -> place: HORIZONTAL into the gap; release near the end
                k = i - n_pre
                s = wpp._smoothstep((k + 1) / (_N_PLACE - n_pre))
                released = s >= 0.9
                _pose = _manip(
                    rig, core.lerp_arm(arm_pre, place_arm, s), 1.0, lf, _place_squat
                )
                _rz = _pin_z(_pose, st.floor_z) if _place_squat > 1e-6 else st.stand_z
                add(
                    _pose,
                    obj_rest_state if released else ("carry", otw),
                    "place",
                    free_dynamic=released and fdp_obj,
                    root_z=_rz,
                )
        n_out = max(1, int(round(_N_RETRACT * 0.5)))
        for i in range(
            n_out
        ):  # place -> pre: back the empty hand HORIZONTALLY out of the gap
            s = wpp._smoothstep((i + 1) / n_out)
            _pose = _manip(
                rig, core.lerp_arm(place_arm, arm_pre, s), 1.0 - s, lf, _place_squat
            )
            _rz = _pin_z(_pose, st.floor_z) if _place_squat > 1e-6 else st.stand_z
            add(_pose, obj_rest_state, "retract", free_dynamic=fdp_obj, root_z=_rz)
        for i in range(
            _N_RETRACT
        ):  # pre -> rest; stand back up; object stays on the board
            s = wpp._smoothstep((i + 1) / _N_RETRACT)
            _sf = _place_squat * (1.0 - s)
            _pose = _manip(
                rig, core.lerp_arm(arm_pre, arm_rest, s), 0.0, lf * (1.0 - s), _sf
            )
            _rz = _pin_z(_pose, st.floor_z) if _sf > 1e-6 else st.stand_z
            add(_pose, obj_rest_state, "retract", free_dynamic=fdp_obj, root_z=_rz)
    elif surf_e is None or surf_e.foot_half is None or into:
        # DIRECT PLACE: no surface footprint (OR an INTO open-bin drop, no shelves) -> proven direct
        # place + retract. A shelf-less bin dips straight into the opening from above (correct for a top bin).
        for i in range(_N_PLACE):
            s = wpp._smoothstep((i + 1) / _N_PLACE)
            released = s >= 0.9
            # ramp the crouch IN as the hand dips to the drop; re-pin the root so the feet stay on the floor
            # at the crouched height (matches the reach solve). _place_squat==0 -> upright.
            _sf = _place_squat * s
            _pose = _manip(
                rig, core.lerp_arm(grasp_arm, place_arm, s), 1.0, lf * s, _sf
            )
            _rz = _pin_z(_pose, st.floor_z) if _sf > 1e-6 else st.stand_z
            add(
                _pose,
                obj_rest_state if released else ("carry", otw),
                "place",
                free_dynamic=released and fdp_obj,
                root_z=_rz,
            )
        for i in range(_N_RETRACT):
            s = wpp._smoothstep((i + 1) / _N_RETRACT)
            _sf = _place_squat * (
                1.0 - s
            )  # ramp the crouch back OUT as the hand retracts + stands up
            _pose = _manip(
                rig, core.lerp_arm(place_arm, arm_rest, s), 1.0 - s, lf * (1.0 - s), _sf
            )
            _rz = _pin_z(_pose, st.floor_z) if _sf > 1e-6 else st.stand_z
            add(
                _pose,
                obj_rest_state,
                "retract",
                free_dynamic=fdp_obj,
                root_z=_rz,
            )
    else:
        # APPROACH-WAYPOINT place: bring the held object in from the geometry-derived front (pre_grasp),
        # set it down (place), then back the hand up+off along -approach (retreat) clear of the surface.
        wp = plan_reach_waypoints(
            scene, surf_key, st.root_xy, obj_world, entity=surf_e0
        )
        boxes = _reach_boxes(scene)
        arm_pre, _, _, _ = _solve_and_check(
            rig,
            grasp_arm,
            wp["pre"],
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            boxes,
            "put pre_grasp",
            strict=_body_collision_is_hard(scene),
        )
        # place leg (== place_arm solved above): check its arm clearance + log in the same format.
        dp, hp = _verts_in_boxes(
            _fwd_arm_points(
                _manip(rig, place_arm, 1.0, lf),
                *wpp._grasp_frame(st.root_xy, st.root_yaw, st.stand_z),
            ),
            boxes,
            exclude_sphere=(obj_world, 0.22),
        )
        print(
            f"[reach] put place: err={err * 1000:.1f}mm arm-clear worst={dp * 1000:.0f}mm hit={hp or '-'}"
        )
        _require_arm_clear(scene, dp, hp, "put place")
        arm_ret, _, _, _ = _solve_and_check(
            rig,
            place_arm,
            wp["retreat"],
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            boxes,
            "put retreat",
            strict=_body_collision_is_hard(scene),
        )
        n_pre = max(1, int(round(_N_PLACE * 0.4)))
        for i in range(_N_PLACE):
            if i < n_pre:  # carry -> pre_grasp above the drop, still holding
                s = wpp._smoothstep((i + 1) / n_pre)
                add(
                    _manip(rig, core.lerp_arm(grasp_arm, arm_pre, s), 1.0, lf * s),
                    ("carry", otw),
                    "place",
                )
            else:  # pre_grasp -> drop; release (rest) only near the end
                k = i - n_pre
                s = wpp._smoothstep((k + 1) / (_N_PLACE - n_pre))
                released = s >= 0.9
                add(
                    _manip(rig, core.lerp_arm(arm_pre, place_arm, s), 1.0, lf),
                    obj_rest_state if released else ("carry", otw),
                    "place",
                    free_dynamic=released and fdp_obj,
                )
        n_back = max(1, int(round(_N_RETRACT * 0.5)))
        for i in range(
            n_back
        ):  # place -> retreat: back the empty hand up + off the surface
            s = wpp._smoothstep((i + 1) / n_back)
            add(
                _manip(
                    rig, core.lerp_arm(place_arm, arm_ret, s), 1.0 - s, lf * (1.0 - s)
                ),
                obj_rest_state,
                "retract",
                free_dynamic=fdp_obj,
            )
        for i in range(_N_RETRACT):  # retreat -> rest, object stays on the surface
            s = wpp._smoothstep((i + 1) / _N_RETRACT)
            add(
                _manip(rig, core.lerp_arm(arm_ret, arm_rest, s), 0.0, 0.0),
                obj_rest_state,
                "retract",
                free_dynamic=fdp_obj,
            )

    import action_check

    insert_camera_pullback = (
        _low_front_insert_camera_pullback(scene, surf_e0, target_z=place_z)
        if low_front_into and surf_e0 is not None
        else 0.0
    )
    camera_hit = (
        first_fpv_eye_collision(
            scene,
            frames,
            camera_frame_offset=int(goal.meta.get("_camera_frame_offset", 0)),
            clearance_radius_m=(
                _FPV_EYE_CLEAR_RADIUS_M
                if insert_camera_pullback > 0.0
                else _LOW_FRONT_INSERT_EYE_CLEARANCE_M
                if low_front_into
                else _FPV_EYE_CLEAR_RADIUS_M
            ),
            camera_pullback_m=insert_camera_pullback,
        )
        if action_check.uses_egocentric_camera_collision_policy(scene)
        or (
            getattr(scene, "egocentric_collision_fallback_used", False)
            and not getattr(scene, "strict_acceptance", False)
        )
        else None
    )
    if camera_hit is not None:
        st.walk_end_yaw = prior_walk_end_yaw
        st.last_pose156 = prior_last_pose156
        _prepare_placement_reach_retry(
            scene,
            goal,
            held_key,
            surf_key,
            place_xy,
            place_z,
            _shelf_gap_ceiling,
        )
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"PLACE {held_key!r} needs a different stand because the FPV eye "
            f"intersects {camera_hit[1]} at local frame {camera_hit[0]}",
        )

    if into:
        scene.declare_place(
            held_key, None
        )  # release (clear held) WITHOUT an ON(obj, surface) relation
        scene.declare_inside(held_key, surf_key)  # state effect: INSIDE(obj, container)
    elif floor_placement:
        scene.declare_place(held_key, None)
    else:
        scene.declare_place(
            held_key, surf_key
        )  # state effect: released; ON(obj, surface) if named
    if has_rest_pose:
        scene.set_object_rest_pose(held_key, obj_rest_value)
    else:
        scene.set_object_rest(held_key, obj_drop)
    st.held = None
    st.obj_rest_xyz = obj_rest_value
    st.last_pose156 = _manip(
        rig,
        arm_rest,
        0.0,
        0.0,
        _place_squat if keep_crouched_into else 0.0,
    )
    if keep_crouched_into and surf_key is not None:
        _set_crouch_hold(st, surf_key, _place_squat, st.last_pose156)
    _stamp_obj_path(frames, scene, held_key)  # multi-graspable: tag the placed prim
    _stamp_carried(frames, _carried)  # contents settle in/on the placed container
    return frames


def _assembly_root_keys(scene: SceneState) -> set[str]:
    roots = set(getattr(scene, "assembly_roots", ()) or ())
    legacy_root = getattr(scene, "assembly_root", None)
    if legacy_root:
        roots.add(str(legacy_root))
    return roots


def _resolve_attach_target(  # noqa: C901
    scene: SceneState,
    child_key: str,
    parent_key: str,
    meta: dict,
) -> dict:
    """Resolve one connector pair and its exact child world pose.

    This is the shared geometry path for authored ``Attach`` actions and build-time
    ``world.initial_attachments``. Both therefore use the same connector lookup,
    precedence rule, indices, orientation fault injection, and 6-DOF mate math.
    """
    from scene_state import _mat_from_pos_quat

    stage = getattr(scene, "stage", None)
    child_e = scene.entities.get(child_key)
    parent_e = scene.entities.get(parent_key) if parent_key else None
    if child_e is None or parent_e is None or stage is None:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT,
            f"ATTACH child/parent missing ({child_key}->{parent_key})",
        )
    if not child_e.prim_path or not parent_e.prim_path:
        raise PrimitiveFailure(PrimitiveError.NO_OBJECT, "ATTACH part has no prim_path")

    # PRECEDENCE (honest): the parent must be the pre-placed root OR already assembled -- else its connector has no
    # meaningful world pose yet and mating onto it welds the child to the wrong spot. Refuse up front.
    if parent_key not in _assembly_root_keys(scene) and parent_key not in getattr(
        scene, "attached", {}
    ):
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"ATTACH parent {parent_key} not yet assembled",
        )

    # Resolve the mate: explicit child_point/parent_point (4-arg form) OR auto-pairing (back-compat fallback).
    child_point = meta.get("child_point")
    parent_point = meta.get("parent_point")
    cp_idx = meta.get("child_point_idx", 0)
    pp_idx = meta.get("parent_point_idx", 0)

    if child_point and parent_point:
        # Explicit 4-arg pairing. The grammar lowercases connector names but the metadata keys are mixed-case,
        # so resolve each connector case-insensitively to its actual attach key (used for the mate + scoring).
        def _ci_key(_d, _name):
            if _name in _d:
                return _name
            _ln = str(_name).lower()
            return next((k for k in _d if k.lower() == _ln), None)

        _req_c, _req_p = child_point, parent_point
        child_point = _ci_key(child_e.attach, child_point)
        parent_point = _ci_key(parent_e.attach, parent_point)
        # Name the connectors that DO exist, not only the one that did not. Reporting the failed lookup
        # alone says the request was wrong but never what would have been right, which leaves the reader
        # guessing at the naming convention. Connector names are shaped like `<part><model-id><m|f>`
        # (e.g. `shelfsideyujrmwm`, `shelftopm`, `shelfbackm`), so whether the entity is keyed on
        # something shorter, or `child_key` is pinned while the requested connectors cycle through
        # OTHER parts, is answerable in one glance from the list below.
        a_list = child_e.attach.get(child_point) if child_point else None
        if not a_list:
            raise PrimitiveFailure(
                PrimitiveError.NO_OBJECT,
                f"ATTACH child {child_key} has no connector '{_req_c}' (part-ID trap); "
                f"it has {sorted(child_e.attach)}",
            )
        b_list = parent_e.attach.get(parent_point) if parent_point else None
        if not b_list:
            raise PrimitiveFailure(
                PrimitiveError.NO_OBJECT,
                f"ATTACH parent {parent_key} has no connector '{_req_p}' (part-ID trap); "
                f"it has {sorted(parent_e.attach)}",
            )
        if not (0 <= cp_idx < len(a_list)):
            cp_idx = 0
        if not (0 <= pp_idx < len(b_list)):
            pp_idx = 0
        a_entry = a_list[cp_idx]
        b_entry = b_list[pp_idx]
    else:
        # Auto-pairing fallback (back-compat): child's single "...M" -> parent's same-base "...F".
        male = next((k for k in child_e.attach if k.endswith("M")), None)
        m_list = child_e.attach.get(male) if male else None
        if not m_list:
            raise PrimitiveFailure(
                PrimitiveError.NO_OBJECT,
                f"ATTACH child {child_key} exposes no Male attach point",
            )
        base = male[:-1]
        f_list = parent_e.attach.get(base + "F")
        if not f_list:
            raise PrimitiveFailure(
                PrimitiveError.NO_OBJECT,
                f"ATTACH parent {parent_key} has no {base}F female (part-ID trap for {child_key})",
            )
        # `point` is the Female INSTANCE INDEX (legacy 3rd arg). None / out-of-range -> 0.
        point = meta.get("point")
        try:
            idx = int(point) if point is not None else 0
        except (TypeError, ValueError):
            idx = 0
        if not (0 <= idx < len(f_list)):
            idx = 0
        a_entry = m_list[0]
        b_entry = f_list[idx]
        child_point = male
        parent_point = base + "F"
        cp_idx = 0
        pp_idx = idx

    # Mate math (column-vector 4x4; quats XYZW). C_world places the child; contact_pt = the mating point world
    # (== the parent connector's world by construction), where the hand delivers the part.
    # Parent pose: for an already-welded parent use its STORED assembled 4x4 -- its prim is not repositioned
    # until the render loop, so reading the prim at frame-gen would see the SPAWN pose and a depth-2 chain
    # (baseboard -> side_R) would weld relative to side_R's spawn. The root's prim IS pre-placed -> read live.
    p_world = scene.assembly_pose.get(parent_key)
    if p_world is None:
        p_world = _read_world_col(stage, parent_e.prim_path)
    if p_world is None:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT,
            f"ATTACH parent {parent_key} has no live transform",
        )
    a_local = _mat_from_pos_quat(a_entry["pos"], a_entry["quat"])
    b_local = _mat_from_pos_quat(b_entry["pos"], b_entry["quat"])
    c_world = p_world @ b_local @ np.linalg.inv(a_local)
    contact_world = np.asarray((p_world @ b_local)[:3, 3], dtype=np.float64)
    orientation_error_deg = float(meta.get("orientation_error_deg") or 0.0)
    if orientation_error_deg:
        orientation_error_axis = meta.get("orientation_error_axis")
        if orientation_error_axis != "connector_y":
            raise PrimitiveFailure(
                PrimitiveError.PRECONDITION_UNMET,
                f"unsupported ATTACH orientation-error axis {orientation_error_axis!r}",
            )
        # Connector-local Y flips the bookcase side panel end-for-end while keeping its connector at
        # the same point and the whole panel above the floor. Rotating this asset about local X or Z
        # would put roughly 25 cm of it below the floor.
        connector_flip_axis_world = np.asarray(
            (p_world @ b_local)[:3, 1], dtype=np.float64
        )
        c_world = _rotate_pose_about_pivot(
            c_world,
            contact_world,
            connector_flip_axis_world,
            orientation_error_deg,
        )
    return {
        "stage": stage,
        "child_entity": child_e,
        "parent_entity": parent_e,
        "child_point": child_point,
        "parent_point": parent_point,
        "child_point_idx": int(cp_idx),
        "parent_point_idx": int(pp_idx),
        "child_world": c_world,
        "contact_world": contact_world,
        "orientation_error_deg": orientation_error_deg,
    }


def apply_initial_attachments(scene: SceneState, attachments: list[dict]) -> list[dict]:
    """Physically realize and verify frozen TaskPack initial mates.

    Rows may be in any order. A child whose parent is itself initially attached is
    applied only after that parent, matching the same root/assembled-parent
    precedence enforced by interactive ``Attach``. Any cycle, missing parent chain,
    stale prim, or connector residual fails the build instead of publishing a
    symbolic-only attachment.
    """
    pending = [dict(row) for row in attachments]
    statuses: list[dict] = []
    roots = _assembly_root_keys(scene)
    root_description = (
        repr(next(iter(roots))) if len(roots) == 1 else repr(sorted(roots))
    )
    while pending:
        progressed = False
        for row in list(pending):
            child_key = str(row.get("child") or "")
            parent_key = str(row.get("parent") or "")
            if parent_key not in roots and parent_key not in scene.attached:
                continue
            target = _resolve_attach_target(scene, child_key, parent_key, row)
            child_world = target["child_world"]
            child_entity = target["child_entity"]
            if not _write_world_col(
                target["stage"], child_entity.prim_path, child_world
            ):
                raise PrimitiveFailure(
                    PrimitiveError.NO_OBJECT,
                    f"initial ATTACH child {child_key} has no live writable prim",
                )
            scene.declare_attached(
                child_key,
                parent_key,
                target["child_point"],
                target["parent_point"],
                target["child_point_idx"],
                target["parent_point_idx"],
            )
            scene.assembly_pose[child_key] = child_world
            scene.set_object_rest(
                child_key, tuple(float(v) for v in child_world[:3, 3])
            )
            status = scene.attachment_status(child_key)
            if not status["physical_attached"]:
                raise PrimitiveFailure(
                    PrimitiveError.PRECONDITION_UNMET,
                    f"initial ATTACH {child_key}->{parent_key} failed live verification: {status}",
                )
            statuses.append(status)
            pending.remove(row)
            progressed = True
        if not progressed:
            blocked = [(row.get("child"), row.get("parent")) for row in pending]
            raise PrimitiveFailure(
                PrimitiveError.PRECONDITION_UNMET,
                f"initial ATTACH graph does not reach assembly root {root_description}: {blocked}",
            )
    scene.refresh_assembly_affordances()
    return statuses


def produce_attach(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Assembly [Attach]: mate the HELD child part's connector onto the parent's connector at the DATA-
    defined 6-DOF pose ``C_world = P_world @ F_local @ inv(M_local)`` (P_world read LIVE, so a wrong-order weld
    lands at the wrong spot and fails the honest ATTACHED read). The child is driven KINEMATIC via a
    ``("pose", C_world)`` frame (smooth interpolation from carried pose) -- never settled/released to dynamic
    -- so the flush seam stays put. Explicit 4-arg pairing (child_point/parent_point from goal.meta); falls
    back to auto-pairing (child's single "...M" -> parent's same-base "...F") for back-compat.
    Honest failure modes: empty hand -> PLACE_UNSTABLE; parent not yet assembled -> PRECONDITION_UNMET; no
    matching connector (part-ID trap) -> NO_OBJECT; the mate point outside the arm envelope -> OUT_OF_REACH."""
    if st.held is None:
        raise PrimitiveFailure(
            PrimitiveError.PLACE_UNSTABLE, "ATTACH with an empty hand"
        )
    grasp_arm = st.held["grasp_arm"]
    otw = st.held["otw"]
    child_key = st.held["obj_key"]
    requested_child = goal.meta.get("child")
    if requested_child is not None and requested_child != child_key:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"ATTACH names child {requested_child}, but the hand holds {child_key}",
        )
    parent_key = goal.meta.get("parent")
    target = _resolve_attach_target(scene, child_key, parent_key, goal.meta)
    stage = target["stage"]
    child_e = target["child_entity"]
    child_point = target["child_point"]
    parent_point = target["parent_point"]
    cp_idx = target["child_point_idx"]
    pp_idx = target["parent_point_idx"]
    c_world = target["child_world"]
    contact_world = target["contact_world"]
    orientation_error_deg = target["orientation_error_deg"]

    # Reach the mating point with the crouch/lean ladder (mirrors produce_place). Unreachable -> honest fail.
    _pp_on = bool(getattr(scene, "unified_geom", False))
    _attach_reach_tol = _goal_reach_tolerance(goal, _REACH_TOL_M)
    _egocentric_reach_fallback = "_egocentric_reach_tol_m" in (goal.meta or {})
    _plan = plan_reach(
        rig,
        grasp_arm,
        contact_world,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        _attach_reach_tol,
        floor_z=st.floor_z,
        enabled=_pp_on,
        clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
        allow_colliding_fallback=_egocentric_reach_fallback,
    )
    if _plan.get("collision_blocked"):
        raise PrimitiveFailure(
            PrimitiveError.TRAJECTORY_COLLISION,
            f"ATTACH {child_key!r} has no collision-free reach pose",
        )
    _lean = _plan["lean_rad"] if _pp_on else None
    _squat = float(_plan["squat"]) if _pp_on else 0.0
    _stand_z = float(_plan["stand_z"]) if _pp_on else st.stand_z
    if _plan.get("arm_pose") is not None and _plan.get("err") is not None:
        place_arm = wpp._arm_from_pose(_plan["arm_pose"])
        err = float(_plan["err"])
    else:
        place_arm, err = _solve_reach_arm(
            rig,
            grasp_arm,
            contact_world,
            st.root_xy,
            st.root_yaw,
            _stand_z,
            lean_override=_lean,
            squat_frac=_squat,
        )
    print(
        f"[attach] {child_key}->{parent_key}[{parent_point}#{pp_idx}] child={child_point}#{cp_idx} "
        f"seat_err={err * 1000:.1f}mm orientation_error={orientation_error_deg:g}deg "
        f"target={tuple(round(float(x), 3) for x in contact_world)}"
    )
    if err > _attach_reach_tol:
        # Out of reach from the CURRENT stand -> record the connector's world point so the render loop
        # re-navigates to a stand that CAN reach it and retries ONCE (the SAME "walk to a reachable spot"
        # logic PUT uses via goal.meta['_failed_drop']). Honest OUT_OF_REACH only if no such stand exists.
        goal.meta["_failed_drop"] = tuple(float(x) for x in contact_world)
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH, f"attach seat_err={err * 1000:.0f}mm"
        )

    _, lf = _lean_for_height(float(contact_world[2]))
    if _pp_on:
        lf = _plan["lf"]
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    child_pp = child_e.prim_path
    frames = _arrive_beats(rig, st, grasp_arm, 1.0, ("carry", otw))

    def _add(pose, obj_state, seg, root_z=None, gaze=None, anchor=None):
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                st.root_yaw,
                st.stand_z if root_z is None else float(root_z),
                fpv_mode="look",
                obj=obj_state,
                seg=seg,
                gaze=(np.asarray(gaze, dtype=np.float64) if gaze is not None else None),
                contact_anchor=(
                    np.asarray(anchor, dtype=np.float64) if anchor is not None else None
                ),
                obj_path=child_pp,
            )
        )

    # Helper for quaternion SLERP (spherical linear interpolation).
    def _quat_slerp(q0, q1, t):
        # q0, q1 = (x,y,z,w) quaternions; t in [0,1]. Returns interpolated quat.
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return q0 + t * (q1 - q0)  # nearly parallel -> lerp
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        theta = theta_0 * t
        q2 = q1 - q0 * dot
        q2 = q2 / (np.linalg.norm(q2) + 1e-12)
        return q0 * np.cos(theta) + q2 * np.sin(theta)

    def _mat_to_quat(m):
        # Extract quaternion (x,y,z,w) from a 4x4 column-vector matrix.
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return (
                (m[2, 1] - m[1, 2]) * s,
                (m[0, 2] - m[2, 0]) * s,
                (m[1, 0] - m[0, 1]) * s,
                0.25 / s,
            )
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            return (
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[2, 1] - m[1, 2]) / s,
            )
        if m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            return (
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
                (m[0, 2] - m[2, 0]) / s,
            )
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        return (
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        )

    # Compute the carried start pose (T_pre) by FK'ing place_arm and seating the object (mirrors realize()).
    _p = _manip(rig, place_arm, 1.0, lf, _squat)
    _rz = _pin_z(_p, st.floor_z) if _squat > 1e-6 else st.stand_z
    try:
        _J_arr = gait.pose_place_joints(_p, st.root_xy, st.root_yaw, _rz)
        T_pre = core.seat_object_nosnap(rig, _J_arr, otw)
    except Exception:  # noqa: BLE001 -- FK/seat fallback -> start the interpolation at the mate point
        T_pre = None
    if T_pre is None:
        T_pre = np.eye(4, dtype=np.float64)
        T_pre[:3, 3] = contact_world  # fallback: start at the mate point

    # Extract position + quat from T_pre and c_world for smooth interpolation.
    pos_pre = T_pre[:3, 3]
    quat_pre = _mat_to_quat(T_pre)
    pos_final = c_world[:3, 3]
    quat_final = _mat_to_quat(c_world)

    # Carry the child in (first 60% of _N_PLACE); then smoothly interpolate from T_pre to c_world (last 40%).
    # The hand is at the mate point -> contact_anchor lets the render pitch the torso out (spine relief).
    for i in range(_N_PLACE):
        s = wpp._smoothstep((i + 1) / _N_PLACE)
        _sf = _squat * s
        pose = _manip(rig, core.lerp_arm(grasp_arm, place_arm, s), 1.0, lf * s, _sf)
        _rz = _pin_z(pose, st.floor_z) if _sf > 1e-6 else st.stand_z
        if s < 0.6:
            # Carry phase: drive object via ('carry', otw).
            _add(pose, ("carry", otw), "attach", root_z=_rz, gaze=contact_world)
        else:
            # Interpolate phase: SLERP rotation + LERP position from T_pre to c_world.
            t_interp = (s - 0.6) / 0.4  # map [0.6, 1.0] -> [0, 1]
            pos_i = pos_pre + t_interp * (pos_final - pos_pre)
            quat_i = _quat_slerp(quat_pre, quat_final, t_interp)
            T_i = np.eye(4, dtype=np.float64)
            T_i[:3, 3] = pos_i
            # Rebuild rotation from quat_i (x,y,z,w).
            x, y, z, w = quat_i
            T_i[0, 0] = 1 - 2 * (y * y + z * z)
            T_i[0, 1] = 2 * (x * y - z * w)
            T_i[0, 2] = 2 * (x * z + y * w)
            T_i[1, 0] = 2 * (x * y + z * w)
            T_i[1, 1] = 1 - 2 * (x * x + z * z)
            T_i[1, 2] = 2 * (y * z - x * w)
            T_i[2, 0] = 2 * (x * z - y * w)
            T_i[2, 1] = 2 * (y * z + x * w)
            T_i[2, 2] = 1 - 2 * (x * x + y * y)
            _add(
                pose,
                ("pose", T_i),
                "attach",
                root_z=_rz,
                gaze=contact_world,
                anchor=contact_world,
            )
    for i in range(_N_RETRACT):
        s = wpp._smoothstep((i + 1) / _N_RETRACT)
        _sf = _squat * (1.0 - s)
        pose = _manip(
            rig, core.lerp_arm(place_arm, arm_rest, s), 1.0 - s, lf * (1.0 - s), _sf
        )
        _rz = _pin_z(pose, st.floor_z) if _sf > 1e-6 else st.stand_z
        _add(pose, ("pose", c_world), "retract", root_z=_rz, gaze=contact_world)

    for fr in frames:  # multi-part scene: tag every frame with the child prim (incl. the arrive/carry beats)
        fr.obj_path = child_pp
    scene.declare_attached(
        child_key, parent_key, child_point, parent_point, cp_idx, pp_idx
    )
    scene.assembly_pose[child_key] = (
        c_world  # the assembled 4x4 -> the NEXT child's parent-pose source
    )
    # Weld the child prim to the assembled pose NOW so the pre-render live-prim ATTACHED scorer reads the
    # true mated pose (the render loop re-drives the child per frame; this makes the end-state authoritative
    # for the goal check, mirroring how PUT's per-command settle moves a placed prim before scoring).
    _write_world_col(stage, child_pp, c_world)
    _rest = tuple(float(v) for v in c_world[:3, 3])
    scene.set_object_rest(child_key, _rest)
    _refresh_attach_witnesses(scene, child_key)
    scene.refresh_assembly_affordances()
    st.held = None
    st.obj_rest_xyz = _rest
    st.last_pose156 = _manip(rig, arm_rest, 0.0, 0.0)
    return frames


def produce_detach(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Grasp one attached leaf part, then remove its declared mate for a corrective re-attach."""
    child_key = goal.obj_key or goal.meta.get("child")
    parent_key = goal.meta.get("parent")
    if child_key not in scene.entities or parent_key not in scene.entities:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT,
            f"DETACH child/parent missing ({child_key}->{parent_key})",
        )
    actual_parent = scene.attached.get(child_key)
    if actual_parent != parent_key:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"DETACH requested {child_key}->{parent_key}, actual parent is {actual_parent}",
        )
    dependents = sorted(c for c, p in scene.attached.items() if p == child_key)
    if dependents:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"DETACH {child_key} still carries attached children {dependents}",
        )
    frames = produce_pick(rig, g, scene, st, goal, sd_world=sd_world)
    try:
        scene.declare_detached(child_key, parent_key)
    except ValueError as exc:
        raise PrimitiveFailure(PrimitiveError.PRECONDITION_UNMET, str(exc)) from exc
    return frames


def produce_raise(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """[Raise]: tilt a FLAT-assembled unit upright about its base floor edge. The unit is rigid (root + every
    attached child, welded), so a SINGLE world rotation R (90 deg about the root's bottom edge) carries every
    part -- connectors stay coincident, so ATTACHED is preserved. Reuses the carried-contents render path: the
    root is driven ('pose', R_t) each frame and every attached child rides as carried content (rel =
    inv(root_flat) @ child_world), so all parts (and their parented labels) rotate as one body. The final
    upright poses are welded (assembly_pose + prim) so the scorer + any later stocking see the standing unit.
    Honest: a rigid rotation about a real floor edge -- nothing teleports. Empty assembly -> PRECONDITION_UNMET."""
    root_key = goal.meta.get("root")
    root_e = scene.entities.get(root_key) if root_key else None
    stage = getattr(scene, "stage", None)
    if root_e is None or not getattr(root_e, "prim_path", None) or stage is None:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT, f"RAISE root {root_key} missing"
        )
    children = sorted(scene.assembly_component(root_key) - {root_key})
    if not children:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"RAISE {root_key}: nothing assembled yet",
        )
    root_flat = _read_world_col(stage, root_e.prim_path)
    if root_flat is None:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT, f"RAISE root {root_key} has no live transform"
        )
    root_e.meta["raise_source_pose"] = root_flat.copy()

    # Local bbox of the root -> the tilt axis (its bottom-floor edge) + the HEIGHT axis (longest local extent,
    # the one that becomes vertical). General geometry, no scene constants.
    from pxr import Usd, UsdGeom

    _bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    _rng = _bc.ComputeUntransformedBound(
        stage.GetPrimAtPath(root_e.prim_path)
    ).ComputeAlignedRange()
    lmn, lmx = _rng.GetMin(), _rng.GetMax()
    ext = [float(lmx[i] - lmn[i]) for i in range(3)]
    hi = int(np.argmax(ext))  # HEIGHT axis (longest) -> vertical after raise
    ti = int(np.argmin(ext))  # THIN axis (thickness) -> its min face lies on the floor
    ai = ({0, 1, 2} - {hi, ti}).pop()  # remaining axis = tilt-axis direction
    p_loc = [0.5 * float(lmn[i] + lmx[i]) for i in range(3)]
    p_loc[hi] = float(lmn[hi])  # bottom end of the height axis
    p_loc[ti] = float(lmn[ti])  # floor-contact face
    d_loc = [0.0, 0.0, 0.0]
    d_loc[ai] = 1.0
    p0 = (root_flat @ np.array([*p_loc, 1.0], dtype=np.float64))[:3]
    d = root_flat[:3, :3] @ np.array(d_loc, dtype=np.float64)
    d = d / (float(np.linalg.norm(d)) + 1e-9)
    # the unit's LOCAL centre -> the FPV gaze target each frame (rises with the unit so the egocentric camera
    # WATCHES the cabinet stand up, instead of the too-close base edge which got redirected forward).
    _c_local = np.array(
        [0.5 * float(lmn[i] + lmx[i]) for i in range(3)], dtype=np.float64
    )

    def _rot_axis(theta):
        c, s = math.cos(theta), math.sin(theta)
        ux, uy, uz = float(d[0]), float(d[1]), float(d[2])
        r3 = np.array(
            [
                [
                    c + ux * ux * (1 - c),
                    ux * uy * (1 - c) - uz * s,
                    ux * uz * (1 - c) + uy * s,
                ],
                [
                    uy * ux * (1 - c) + uz * s,
                    c + uy * uy * (1 - c),
                    uy * uz * (1 - c) - ux * s,
                ],
                [
                    uz * ux * (1 - c) - uy * s,
                    uz * uy * (1 - c) + ux * s,
                    c + uz * uz * (1 - c),
                ],
            ],
            dtype=np.float64,
        )
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = r3
        m[:3, 3] = p0 - r3 @ p0
        return m

    # pick the +/-90 deg that lifts the TOP end (height-max) UP in world +Z
    top_loc = np.array([*p_loc, 1.0], dtype=np.float64)
    top_loc[hi] = float(lmx[hi])
    top_w = (root_flat @ top_loc)[:3]
    zp = float((_rot_axis(math.pi / 2) @ np.append(top_w, 1.0))[2])
    zn = float((_rot_axis(-math.pi / 2) @ np.append(top_w, 1.0))[2])
    theta_f = (math.pi / 2) if zp >= zn else (-math.pi / 2)

    # carried: every attached child rides rigidly relative to the root's FLAT pose
    carried = []
    for c in children:
        ce = scene.entities.get(c)
        cc = scene.assembly_pose.get(c)
        if ce is None or not getattr(ce, "prim_path", None) or cc is None:
            continue
        rel = np.linalg.inv(root_flat) @ cc
        carried.append((ce.prim_path, rel, c))
    carried_t = tuple((pp, rel) for (pp, rel, _c) in carried)

    root_pp = root_e.prim_path
    arm = rig.arm_grasp
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    # Step BACK from the base edge + turn to FACE the unit, so the FPV WATCHES it stand up (else the human
    # faced forward from the last attach and the unit swung up off to the side / clipped the too-close eye).
    _p0xy = np.array([float(p0[0]), float(p0[1])], dtype=np.float64)
    _cur = np.array([float(st.root_xy[0]), float(st.root_xy[1])], dtype=np.float64)
    _bd = _cur - _p0xy
    _bn = float(np.linalg.norm(_bd))
    _stand_xy = (
        tuple((_p0xy + (_bd / _bn) * 1.3).tolist())
        if _bn > 0.2
        else tuple(_cur.tolist())
    )
    _face_yaw = math.degrees(
        math.atan2(_p0xy[0] - _stand_xy[0], -(_p0xy[1] - _stand_xy[1]))
    )
    _dyaw = ((_face_yaw - st.root_yaw + 180.0) % 360.0) - 180.0
    _flat_gaze = (root_flat @ np.append(_c_local, 1.0))[:3]
    frames: list[wpp.Frame] = []
    _n_step = 8
    for i in range(_n_step):  # step back + turn to face the (still-flat) unit
        a = wpp._smoothstep((i + 1) / _n_step)
        xy = tuple((_cur + (np.array(_stand_xy) - _cur) * a).tolist())
        frames.append(
            wpp.Frame(
                _manip(rig, arm_rest, 0.0, 0.0),
                xy,
                st.root_yaw + _dyaw * a,
                st.stand_z,
                fpv_mode="look",
                obj=("pose", root_flat),
                seg="raise",
                obj_path=root_pp,
                carried=carried_t,
                gaze=_flat_gaze,
            )
        )
    _n = 24
    for i in range(
        _n
    ):  # tilt the unit upright; the human faces it, FPV tracks the rising centre
        s = wpp._smoothstep((i + 1) / _n)
        r_t = _rot_axis(theta_f * s) @ root_flat
        pose = _manip(rig, arm, 1.0, 0.5 * (1.0 - s), 0.0)
        frames.append(
            wpp.Frame(
                pose,
                _stand_xy,
                _face_yaw,
                st.stand_z,
                fpv_mode="look",
                obj=("pose", r_t),
                seg="raise",
                obj_path=root_pp,
                carried=carried_t,
                gaze=(r_t @ np.append(_c_local, 1.0))[:3],
            )
        )
    # weld the final UPRIGHT poses (root + all children) so the scorer + later stocking see the standing unit
    r_final = _rot_axis(theta_f) @ root_flat
    scene.assembly_pose[root_key] = r_final
    _write_world_col(stage, root_pp, r_final)
    for pp, rel, c in carried:
        c_new = r_final @ rel
        scene.assembly_pose[c] = c_new
        _write_world_col(stage, pp, c_new)
    scene.invalidate_bounds()
    scene.note_motion(scene.assembly_component(root_key))
    scene.refresh_assembly_affordances()
    st.last_pose156 = _manip(rig, arm, 0.0, 0.0)
    print(
        f"[raise] {root_key}: tilted {math.degrees(theta_f):.0f} deg about its base edge; "
        f"{len(carried)} attached parts carried upright"
    )
    return frames


_MOVE_TARGET_GAP_M = 0.12
_MOVE_HAND_CLEARANCE_M = 0.36
_MOVE_OBJECT_CLEARANCE_M = 0.04
_MOVE_PRECONTACT_M = 0.12
_MOVE_CONTACT_HEIGHT_M = 0.95
_N_MOVE_PRECONTACT = 8
_N_MOVE_CONTACT = 10
_N_MOVE_RETRACT = 10


@dataclass(frozen=True)
class _FixtureContact:
    point_world: np.ndarray
    point_local: np.ndarray
    outward_xy: np.ndarray
    stand_xy: tuple[float, float]
    motion_yaw: float
    body_yaw: float
    edge_distance: float


def _movement_capability(scene: SceneState, key: str, role: Role) -> None:
    if key not in scene.entities:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT, f"move target {key!r} is missing"
        )
    if scene.affordance(key, role) is None:
        raise PrimitiveFailure(
            PrimitiveError.MISSING_AFFORDANCE,
            f"{key!r} does not provide {role.value}",
        )


def _aabb_center_half(scene: SceneState, key: str) -> tuple[np.ndarray, np.ndarray]:
    mn, mx = scene.object_aabb(key)
    minimum = np.asarray(mn, dtype=np.float64)
    maximum = np.asarray(mx, dtype=np.float64)
    half = 0.5 * (maximum - minimum)
    if max(float(half[0]), float(half[1])) > 1e-4:
        return 0.5 * (minimum + maximum), half

    entity = scene.resolve(key)
    center_xy = entity.foot_xy or entity.xy or (0.0, 0.0)
    footprint = entity.foot_half or entity.half
    bottom = float(scene.floor_z)
    top = entity.top_z
    if top is None:
        top = max(bottom, float(entity.z or bottom)) + max(float(entity.height), 0.1)
    center = np.array(
        [float(center_xy[0]), float(center_xy[1]), 0.5 * (bottom + float(top))],
        dtype=np.float64,
    )
    return center, np.array(
        [float(footprint[0]), float(footprint[1]), 0.5 * (float(top) - bottom)],
        dtype=np.float64,
    )


def _destination_centers(
    scene: SceneState,
    destination: str,
    mover_half: np.ndarray,
    start: np.ndarray,
    *,
    pull: bool,
) -> list[tuple[float, float]]:
    center, half = _aabb_center_half(scene, destination)
    mover_radius = float(np.linalg.norm(mover_half[:2]))
    gap = _MOVE_TARGET_GAP_M + (_MOVE_HAND_CLEARANCE_M if pull else 0.0)
    candidates = [
        (center[0] - half[0] - mover_radius - gap, center[1]),
        (center[0] + half[0] + mover_radius + gap, center[1]),
        (center[0], center[1] - half[1] - mover_radius - gap),
        (center[0], center[1] + half[1] + mover_radius + gap),
    ]
    return sorted(candidates, key=lambda xy: math.dist(tuple(start[:2]), xy))


def _clear_moving_footprints(occ, grid, scene: SceneState, names: set[str]) -> None:
    pad = float(getattr(grid, "inflate", 0.0) or 0.0) + grid.cell
    for name in names:
        center, half = _aabb_center_half(scene, name)
        mn, mx = center - half, center + half
        i0, j0 = grid.world_to_cell(float(mn[0]) - pad, float(mn[1]) - pad)
        i1, j1 = grid.world_to_cell(float(mx[0]) + pad, float(mx[1]) + pad)
        lo_i, hi_i = max(0, min(i0, i1)), min(grid.nx - 1, max(i0, i1))
        lo_j, hi_j = max(0, min(j0, j1)), min(grid.ny - 1, max(j0, j1))
        if lo_i <= hi_i and lo_j <= hi_j:
            occ[lo_j : hi_j + 1, lo_i : hi_i + 1] = False


def _fixture_route_grid(
    scene: SceneState, mover_half: np.ndarray, moving_names: set[str]
):
    grid = getattr(scene, "nav_grid", None)
    if grid is None:
        return None
    import nav_grid
    from scipy import ndimage

    occ = np.asarray(grid.occ, dtype=bool).copy()
    _clear_moving_footprints(occ, grid, scene, moving_names)
    radius = float(np.linalg.norm(mover_half[:2])) + _MOVE_OBJECT_CLEARANCE_M
    extra = max(0.0, radius - float(grid.inflate or 0.0))
    if extra > 0.0:
        distance = ndimage.distance_transform_edt(~occ) * float(grid.cell)
        occ |= distance < extra
    if grid.floor_support is not None:
        occ |= ~grid.floor_support
    return replace(grid, occ=occ, sdf=nav_grid._two_sided_sdf(occ, grid.cell))


def _fixture_actor_grid(scene: SceneState, moving_names: set[str]):
    grid = getattr(scene, "nav_grid", None)
    if grid is None:
        return None
    import nav_grid

    occ = np.asarray(grid.occ, dtype=bool).copy()
    _clear_moving_footprints(occ, grid, scene, moving_names)
    if grid.floor_support is not None:
        occ |= ~grid.floor_support
    return replace(grid, occ=occ, sdf=nav_grid._two_sided_sdf(occ, grid.cell))


def _path_direction(path) -> np.ndarray:
    for start, end in zip(path, path[1:]):
        direction = np.asarray(end, dtype=np.float64) - np.asarray(
            start, dtype=np.float64
        )
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            return direction / norm
    raise PrimitiveFailure(
        PrimitiveError.PATH_BLOCKED, "fixture movement has zero length"
    )


def _movement_yaw(direction: np.ndarray) -> float:
    return math.degrees(math.atan2(float(direction[0]), -float(direction[1])))


def _edge_distance(half: np.ndarray, direction: np.ndarray) -> float:
    limits = [
        float(half[index]) / abs(float(direction[index]))
        for index in range(2)
        if abs(float(direction[index])) > 1e-6
    ]
    return min(limits) if limits else float(max(half[:2]))


def _fixture_stand(
    center: np.ndarray, half: np.ndarray, path, *, pull: bool
) -> tuple[tuple[float, float], float, float]:
    direction = _path_direction(path)
    outward = direction if pull else -direction
    edge_distance = _edge_distance(half, outward)
    stand = center[:2] + outward * (edge_distance + _MOVE_HAND_CLEARANCE_M)
    return (
        (float(stand[0]), float(stand[1])),
        _movement_yaw(-outward),
        edge_distance,
    )


def _actor_track_is_clear(grid, path, half: np.ndarray, *, pull: bool) -> bool:
    if grid is None:
        return True
    for start, end in zip(path, path[1:]):
        delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            continue
        direction = delta / length
        outward = direction if pull else -direction
        offset = _edge_distance(half, outward) + _MOVE_HAND_CLEARANCE_M
        count = max(1, int(math.ceil(length / max(0.5 * grid.cell, 1e-3))))
        for index in range(count + 1):
            center = np.asarray(start) + delta * (index / count)
            actor = center + outward * offset
            if not grid.is_free(*grid.world_to_cell(float(actor[0]), float(actor[1]))):
                return False
    return True


def _fixture_path(
    scene: SceneState,
    key: str,
    destination: str,
    *,
    actor_xy=None,
    pull: bool = False,
):
    import nav_planner

    start, mover_half = _aabb_center_half(scene, key)
    descendants = set(_carried_descendants(_carried_parent_map(scene), key))
    moving_names = descendants | {key}
    grid = _fixture_route_grid(scene, mover_half, moving_names)
    actor_grid = _fixture_actor_grid(scene, moving_names)
    failures: list[str] = []
    for candidate in _destination_centers(
        scene, destination, mover_half, start, pull=pull
    ):
        if grid is not None:
            ci, cj = grid.world_to_cell(*candidate)
            if not grid.is_free(ci, cj):
                continue
        try:
            path = (
                nav_planner.plan_path(grid, tuple(start[:2]), candidate)
                if grid is not None
                else [tuple(start[:2]), candidate]
            )
            stand_xy, _body_yaw, _edge = _fixture_stand(
                start, mover_half, path, pull=pull
            )
            if actor_xy is not None and getattr(scene, "nav_grid", None) is not None:
                nav_planner.plan_path(scene.nav_grid, tuple(actor_xy), stand_xy)
            if not _actor_track_is_clear(actor_grid, path, mover_half, pull=pull):
                failures.append(f"actor footprint blocked on route to {candidate}")
                continue
            return path, start, mover_half
        except PrimitiveFailure as exc:
            failures.append(exc.detail)
    raise PrimitiveFailure(
        PrimitiveError.PATH_BLOCKED,
        f"no fixture-width route from {key!r} to a free side of {destination!r}: {failures}",
    )


def _move_pose(
    root_pose: np.ndarray,
    start_center: np.ndarray,
    xy,
    yaw_delta: float = 0.0,
) -> np.ndarray:
    pose = _rotate_pose_about_pivot(
        root_pose,
        start_center,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        yaw_delta,
    )
    pose[0, 3] += float(xy[0]) - float(start_center[0])
    pose[1, 3] += float(xy[1]) - float(start_center[1])
    return pose


def _fixture_contact(
    scene: SceneState,
    key: str,
    root_pose: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    path,
    *,
    pull: bool,
) -> _FixtureContact:
    direction = _path_direction(path)
    outward = direction if pull else -direction
    edge_distance = _edge_distance(half, outward)
    lower, upper = center[2] - half[2], center[2] + half[2]
    margin = min(0.05, max(0.0, 0.2 * (upper - lower)))
    contact_z = float(
        np.clip(scene.floor_z + _MOVE_CONTACT_HEIGHT_M, lower + margin, upper - margin)
    )
    contact_world = np.array(
        [
            center[0] + outward[0] * edge_distance,
            center[1] + outward[1] * edge_distance,
            contact_z,
        ],
        dtype=np.float64,
    )
    stand = contact_world[:2] + outward * _MOVE_HAND_CLEARANCE_M
    point_local = (np.linalg.inv(root_pose) @ np.append(contact_world, 1.0))[:3]
    motion_yaw = _movement_yaw(direction)
    return _FixtureContact(
        point_world=contact_world,
        point_local=point_local,
        outward_xy=outward,
        stand_xy=(float(stand[0]), float(stand[1])),
        motion_yaw=motion_yaw,
        body_yaw=motion_yaw + (180.0 if pull else 0.0),
        edge_distance=edge_distance,
    )


def _move_exclusions(root, carried) -> tuple[str, ...]:
    paths = (root.prim_path, *(path for path, _relative in carried))
    return tuple(path for path in paths if path)


def _move_reach_boxes(scene: SceneState, key: str, carried):
    root = scene.resolve(key)
    excluded = {key, root.prim_path, *(path for path, _relative in carried)}

    def included(name: str) -> bool:
        value = str(name)
        return not any(
            token
            and (value == str(token) or value.startswith(f"{str(token).rstrip('/')}/"))
            for token in excluded
        )

    return [box for box in _reach_boxes(scene) if included(box[0])]


def _move_contact_arms(rig, scene, st, key: str, contact: _FixtureContact, carried):
    outward3 = np.array([*contact.outward_xy, 0.0], dtype=np.float64)
    precontact = contact.point_world + outward3 * _MOVE_PRECONTACT_M
    boxes = _move_reach_boxes(scene, key, carried)
    strict = _body_collision_is_hard(scene)
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    arm_pre, err_pre, _depth, _hit = _solve_and_check(
        rig,
        arm_rest,
        precontact,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        boxes,
        f"{key} pre-contact",
        strict=strict,
    )
    arm_contact, err_contact, _depth, _hit = _solve_and_check(
        rig,
        arm_pre,
        contact.point_world,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        boxes,
        f"{key} contact",
        strict=strict,
    )
    tolerance = 3.0 * _reach_tolerance(scene)
    if max(err_pre, err_contact) > tolerance:
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"{key!r} exterior contact misses by {max(err_pre, err_contact) * 1000.0:.0f}mm",
        )
    return arm_pre, arm_contact


def _move_approach_frames(
    rig,
    g: gait.Gait,
    scene: SceneState,
    st: SchedState,
    root,
    root_pose: np.ndarray,
    contact: _FixtureContact,
    carried,
) -> list[wpp.Frame]:
    goal = Goal(
        Primitive.NAVIGATE_TO,
        target_xy=contact.stand_xy,
        target_yaw=contact.body_yaw,
    )
    frames = produce_navigate(rig, g, scene, st, goal)
    for frame in frames:
        frame.obj = ("pose", root_pose)
        frame.obj_path = root.prim_path
        frame.carried = tuple(carried)
        frame.gaze = contact.point_world
    return frames


def _move_reach_frames(
    rig,
    scene: SceneState,
    st: SchedState,
    root,
    root_pose: np.ndarray,
    contact: _FixtureContact,
    carried,
    arm_pre,
    arm_contact,
    *,
    pull: bool,
) -> list[wpp.Frame]:
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    previous = st.last_pose156
    if previous is None:
        previous = _manip(rig, arm_rest, 0.0)
    pre_pose = _manip(rig, arm_pre, 0.0)
    hand_close = 1.0 if pull else 0.0
    contact_pose = _manip(rig, arm_contact, hand_close)
    exclusions = _move_exclusions(root, carried)
    frames: list[wpp.Frame] = []
    for index in range(_N_MOVE_PRECONTACT):
        alpha = wpp._smoothstep((index + 1) / _N_MOVE_PRECONTACT)
        pose = gait.blend156(previous, pre_pose, alpha)
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                st.root_yaw,
                _pin_z(pose, st.floor_z),
                fpv_mode="look",
                obj=("pose", root_pose),
                seg="pull_reach" if pull else "push_reach",
                obj_path=root.prim_path,
                carried=tuple(carried),
                gaze=contact.point_world,
                collision_exclude_paths=exclusions,
            )
        )
    for index in range(_N_MOVE_CONTACT):
        alpha = wpp._smoothstep((index + 1) / _N_MOVE_CONTACT)
        pose = gait.blend156(pre_pose, contact_pose, alpha)
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                st.root_yaw,
                _pin_z(pose, st.floor_z),
                fpv_mode="look",
                obj=("pose", root_pose),
                seg="pull_contact" if pull else "push_contact",
                obj_path=root.prim_path,
                carried=tuple(carried),
                gaze=contact.point_world,
                contact_anchor=contact.point_world if alpha >= 0.8 else None,
                collision_exclude_paths=exclusions,
            )
        )
    st.last_pose156 = contact_pose
    return frames


def _fixture_move_frame(
    rig,
    st: SchedState,
    root,
    root_pose: np.ndarray,
    start_center: np.ndarray,
    step,
    pose156: np.ndarray,
    min_foot_z: float,
    contact: _FixtureContact,
    arm_contact,
    carried,
    *,
    pull: bool,
) -> wpp.Frame:
    yaw_delta = _ang_delta(contact.motion_yaw, step.yaw)
    moved = _move_pose(root_pose, start_center, step.xy, yaw_delta)
    contact_world = (moved @ np.append(contact.point_local, 1.0))[:3]
    angle = math.radians(yaw_delta)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    outward = rotation @ contact.outward_xy
    body_xy = contact_world[:2] + outward * _MOVE_HAND_CLEARANCE_M
    body_yaw = step.yaw + (180.0 if pull else 0.0)
    return wpp.Frame(
        pose156,
        (float(body_xy[0]), float(body_xy[1])),
        body_yaw,
        st.floor_z - min_foot_z,
        arm_override=arm_contact,
        rhp=getattr(rig, "rhp", None) if pull else getattr(rig, "rhp_open", None),
        fpv_mode="look",
        obj=("pose", moved),
        seg="pull" if pull else "push",
        obj_path=root.prim_path,
        carried=tuple(carried),
        gaze=contact_world,
        contact_anchor=contact_world,
        collision_exclude_paths=_move_exclusions(root, carried),
    )


def _move_frames(
    rig,
    g: gait.Gait,
    scene: SceneState,
    st: SchedState,
    key: str,
    root_pose: np.ndarray,
    start_center: np.ndarray,
    path,
    contact: _FixtureContact,
    arm_contact,
    carried,
    *,
    pull: bool,
) -> list[wpp.Frame]:
    import heading_controller as hc

    steps = hc.walk_track(
        path,
        contact.motion_yaw,
        None,
        wpp.WALK_SPEED,
        wpp.FPS,
        hc.HeadingCfg(),
    )
    if not steps:
        raise PrimitiveFailure(
            PrimitiveError.PATH_BLOCKED, "fixture movement has zero length"
        )
    root = scene.resolve(key)
    frames: list[wpp.Frame] = []
    distance = st.gait_cum_dist
    for step in steps:
        distance += step.gait_m
        pose156, min_foot_z = gait.sample(g, distance)
        frames.append(
            _fixture_move_frame(
                rig,
                st,
                root,
                root_pose,
                start_center,
                step,
                pose156,
                min_foot_z,
                contact,
                arm_contact,
                carried,
                pull=pull,
            )
        )
    st.gait_cum_dist = distance
    st.root_xy, st.root_yaw = frames[-1].xy, frames[-1].yaw
    st.walk_end_yaw, st.last_pose156 = st.root_yaw, frames[-1].pose156.copy()
    return frames


def _move_retract_frames(
    rig,
    st: SchedState,
    root,
    final_pose: np.ndarray,
    contact: _FixtureContact,
    arm_contact,
    carried,
    *,
    pull: bool,
) -> list[wpp.Frame]:
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    hand_close = 1.0 if pull else 0.0
    contact_pose = _manip(rig, arm_contact, hand_close)
    rest_pose = _manip(rig, arm_rest, 0.0)
    contact_world = (final_pose @ np.append(contact.point_local, 1.0))[:3]
    exclusions = _move_exclusions(root, carried)
    frames: list[wpp.Frame] = []
    for index in range(_N_MOVE_RETRACT):
        alpha = wpp._smoothstep((index + 1) / _N_MOVE_RETRACT)
        pose = gait.blend156(contact_pose, rest_pose, alpha)
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                st.root_yaw,
                _pin_z(pose, st.floor_z),
                obj=("pose", final_pose),
                seg="pull_retract" if pull else "push_retract",
                obj_path=root.prim_path,
                carried=tuple(carried),
                gaze=contact_world,
                contact_anchor=contact_world if alpha <= 0.25 else None,
                collision_exclude_paths=exclusions,
            )
        )
    st.last_pose156 = rest_pose
    return frames


def _refresh_moved_world(scene: SceneState, state: SchedState) -> None:
    cworld = getattr(scene, "cworld", None)
    if cworld is not None:
        cworld.refresh(scene)
    grid = getattr(scene, "nav_grid", None)
    if grid is None or scene.stage is None:
        return
    import nav_grid

    rebuilt = nav_grid.build_nav_grid(
        scene.stage,
        scene,
        cell=float(grid.cell),
        inflate=float(grid.inflate or 0.10),
    )
    scene.nav_grid = rebuilt
    if getattr(scene, "wall_sdf_grid", None) is grid:
        scene.wall_sdf_grid = rebuilt
    for name in sorted(scene.open_objs):
        scene.carve_door_open(name)
        scene.mark_leaf_obstacle(name)
    nav_grid.carve_free(rebuilt, [state.root_xy])


def _produce_fixture_move(
    rig,
    g: gait.Gait,
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    *,
    pull: bool,
) -> list[wpp.Frame]:
    key = str(goal.obj_key or "")
    destination = str((goal.meta or {}).get("destination") or "")
    role = Role.PULLABLE if pull else Role.PUSHABLE
    _movement_capability(scene, key, role)
    if destination not in scene.entities:
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT, f"destination {destination!r} is missing"
        )
    if st.held is not None:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET, "move fixture requires an empty hand"
        )
    root = scene.resolve(key)
    root_pose = (
        _read_world_col(scene.stage, root.prim_path)
        if scene.stage is not None
        else None
    )
    if root_pose is None:
        root_pose = np.eye(4, dtype=np.float64)
        root_pose[:3, 3] = scene.object_pos(key)
    path, start, mover_half = _fixture_path(
        scene,
        key,
        destination,
        actor_xy=st.root_xy,
        pull=pull,
    )
    carried = _gather_carried(scene, key)
    contact = _fixture_contact(
        scene, key, root_pose, start, mover_half, path, pull=pull
    )
    frames = _move_approach_frames(rig, g, scene, st, root, root_pose, contact, carried)
    arm_pre, arm_contact = _move_contact_arms(rig, scene, st, key, contact, carried)
    frames.extend(
        _move_reach_frames(
            rig,
            scene,
            st,
            root,
            root_pose,
            contact,
            carried,
            arm_pre,
            arm_contact,
            pull=pull,
        )
    )
    frames.extend(
        _move_frames(
            rig,
            g,
            scene,
            st,
            key,
            root_pose,
            start,
            path,
            contact,
            arm_contact,
            carried,
            pull=pull,
        )
    )
    final_yaw = _movement_yaw(_path_direction(path[-2:]))
    final_pose = _move_pose(
        root_pose,
        start,
        path[-1],
        _ang_delta(contact.motion_yaw, final_yaw),
    )
    frames.extend(
        _move_retract_frames(
            rig,
            st,
            root,
            final_pose,
            contact,
            arm_contact,
            carried,
            pull=pull,
        )
    )
    if scene.stage is not None and not _write_world_col(
        scene.stage, root.prim_path, final_pose
    ):
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT, f"move target {key!r} has no live writable prim"
        )
    scene.set_object_rest_pose(key, final_pose)
    if key in scene.assembly_pose or scene.assembly_component(key) != {key}:
        scene.assembly_pose[key] = final_pose.copy()
    sync_carried_to_root(scene, key, carried)
    scene.declare_moved(key, destination)
    scene.invalidate_bounds()
    _refresh_moved_world(scene, st)
    goal.meta.update(
        {
            "_carried_root": key,
            "_carried": tuple(carried),
            "_contact_local": tuple(float(value) for value in contact.point_local),
        }
    )
    return frames


def produce_push(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    return _produce_fixture_move(rig, g, scene, st, goal, pull=False)


def produce_pull(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    return _produce_fixture_move(rig, g, scene, st, goal, pull=True)


def produce_place_inside(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """[PutInside]: set the held object down into a distinct interior SPACE -- ``goal.meta['space']`` selects
    one by name/index (unspecified -> auto-distributed across free spaces, upper levels first) -- and reach
    into it. A DISTINCT action from [Put] (which drops onto a surface top). Reuses produce_place's placement
    core via the interior path; the space plane + footprint are resolved there from the container's geometry-
    extracted ``e.spaces``. A top-open bin with no spaces degrades to the single interior drop (unchanged)."""
    goal.meta["into"] = True
    return produce_place(rig, g, scene, st, goal, sd_world)


def produce_release(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Open the hand where it is and let the held object rest at its current carried position."""
    if st.held is None:
        raise PrimitiveFailure(
            PrimitiveError.GRASP_FAILED, "RELEASE with an empty hand"
        )
    grasp_arm = st.held["grasp_arm"]
    otw = st.held["otw"]
    held_key = st.held["obj_key"]
    _carried = st.held.get(
        "carried"
    )  # contents riding held_key (captured before st.held clears)
    # Realize the current carried object position so it rests exactly where the hand let go.
    _, J, obj_T = wpp.realize(
        rig,
        wpp.Frame(
            _manip(rig, grasp_arm, 1.0, 0.0),
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            obj=("carry", otw),
        ),
    )
    drop = (
        tuple(float(x) for x in obj_T[:3, 3])
        if obj_T is not None
        else st.root_xy + (st.stand_z,)
    )
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    frames = []
    for i in range(_N_RETRACT):
        s = wpp._smoothstep((i + 1) / _N_RETRACT)
        released = s >= 0.5
        frames.append(
            wpp.Frame(
                _manip(rig, core.lerp_arm(grasp_arm, arm_rest, s), 1.0 - s, 0.0),
                st.root_xy,
                st.root_yaw,
                st.stand_z,
                fpv_mode="look",
                obj=("rest", drop) if released else ("carry", otw),
                seg="release",
            )
        )
    scene.declare_place(held_key, None)  # dropped in place (no named surface)
    scene.set_object_rest(held_key, drop)
    st.held = None
    st.obj_rest_xyz = drop
    st.last_pose156 = _manip(rig, arm_rest, 0.0, 0.0)
    _stamp_obj_path(frames, scene, held_key)  # multi-graspable: tag the released prim
    _stamp_carried(frames, _carried)  # contents settle in/on the placed container
    return frames


def _adjacent_floor_pose(
    scene: SceneState,
    obj_key: str,
    anchor_key: str,
    *,
    hidden_from_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    obj = scene.resolve(obj_key)
    anchor = scene.target_view(anchor_key, Role.SURFACE) or scene.resolve(anchor_key)
    if anchor.xy is None:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"fall anchor {anchor_key!r} has no grounded position",
        )
    current = scene.object_pos(obj_key)
    if hidden_from_xy is None:
        dx = float(current[0]) - float(anchor.xy[0])
        dy = float(current[1]) - float(anchor.xy[1])
    else:
        # Put an unattended fallen object just beyond the support from the wearer's
        # current viewpoint. The support then naturally occludes the floor object
        # when the wearer later approaches from the same side.
        dx = float(anchor.xy[0]) - float(hidden_from_xy[0])
        dy = float(anchor.xy[1]) - float(hidden_from_xy[1])
        if math.hypot(dx, dy) <= 1e-6:
            dx = float(current[0]) - float(anchor.xy[0])
            dy = float(current[1]) - float(anchor.xy[1])
    hx, hy = anchor.half or anchor.foot_half or (0.30, 0.30)
    score_x = abs(dx) / max(float(hx), 1e-6)
    score_y = abs(dy) / max(float(hy), 1e-6)
    obj_half = obj.half or obj.foot_half or (0.05, 0.05)
    gap = PLACEMENT_MIN_GAP_M + 0.04
    x, y = float(current[0]), float(current[1])
    if score_x >= score_y:
        sign = 1.0 if dx >= 0.0 else -1.0
        x = float(anchor.xy[0]) + sign * (float(hx) + float(obj_half[0]) + gap)
        y = min(
            float(anchor.xy[1]) + float(hy),
            max(float(anchor.xy[1]) - float(hy), y),
        )
    else:
        sign = 1.0 if dy >= 0.0 else -1.0
        y = float(anchor.xy[1]) + sign * (float(hy) + float(obj_half[1]) + gap)
        x = min(
            float(anchor.xy[0]) + float(hx),
            max(float(anchor.xy[0]) - float(hx), x),
        )
    pose = None
    if scene.stage is not None and obj.prim_path:
        pose = _read_world_col(scene.stage, obj.prim_path)
    if pose is None and obj_key in scene.object_rest_pose:
        pose = np.asarray(scene.object_rest_pose[obj_key], dtype=np.float64).copy()
    if pose is None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = scene.object_origin(obj_key)
    _xy, zmin, _zmax = scene.object_bbox(obj_key)
    origin_above_base = max(0.0, float(pose[2, 3]) - float(zmin))
    pose[:3, 3] = (x, y, float(scene.floor_z) + origin_above_base)
    return pose


def _floor_xy_in_front_of_anchor(scene: SceneState, obj_key: str, anchor_key: str):
    anchor = scene.resolve(anchor_key)
    if anchor.xy is None:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"floor anchor {anchor_key!r} has no grounded position",
        )
    stand_xy, _yaw = scene.stand_pose_for(anchor_key)
    direction = np.asarray(stand_xy, dtype=np.float64) - np.asarray(
        anchor.xy, dtype=np.float64
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        yaw = math.radians(float(anchor.approach_yaw or 0.0))
        direction = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        norm = 1.0
    direction /= norm
    # Imported fixtures keep their asset-local ``half`` extents even when the
    # instance is rotated in the scene.  Use the live world AABB here: otherwise
    # a wide cabinet rotated 90 degrees can put the object metres past its real
    # front and make the physical NEAR postcondition disagree with the action.
    anchor_min, anchor_max = scene._relation_aabb(anchor_key)
    anchor_min = np.asarray(anchor_min, dtype=np.float64)
    anchor_max = np.asarray(anchor_max, dtype=np.float64)
    anchor_center = 0.5 * (anchor_min[:2] + anchor_max[:2])
    anchor_half = 0.5 * (anchor_max[:2] - anchor_min[:2])
    obj = scene.resolve(obj_key)
    obj_half = obj.half or obj.foot_half or (0.05, 0.05)
    extent = abs(float(direction[0])) * float(anchor_half[0]) + abs(
        float(direction[1])
    ) * float(anchor_half[1])
    obj_extent = abs(float(direction[0])) * float(obj_half[0]) + abs(
        float(direction[1])
    ) * float(obj_half[1])
    distance = extent + obj_extent + PLACEMENT_MIN_GAP_M + 0.05
    target = anchor_center + direction * distance
    lateral = np.array([-direction[1], direction[0]], dtype=np.float64)
    occupied = []
    for key, entity in scene.entities.items():
        if key in {obj_key, anchor_key} or entity.xy is None or entity.z is None:
            continue
        if float(entity.z) > float(scene.floor_z) + 0.35:
            continue
        occupied.append((np.asarray(entity.xy, dtype=np.float64), entity))
    for offset in (0.0, 0.18, -0.18, 0.36, -0.36):
        candidate = target + lateral * offset
        if all(
            float(np.linalg.norm(candidate - xy))
            >= float(obj_half[0])
            + float((entity.half or entity.foot_half or (0.05, 0.05))[0])
            + PLACEMENT_MIN_GAP_M
            for xy, entity in occupied
        ):
            return float(candidate[0]), float(candidate[1])
    return float(target[0]), float(target[1])


def produce_place_on_floor(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Crouch and place the held object on the floor in front of a named anchor."""
    obj_key = str(goal.obj_key or "")
    anchor_key = str((goal.meta or {}).get("anchor") or "")
    if (
        st.held is None
        or st.held.get("obj_key") != obj_key
        or obj_key not in scene.held
    ):
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"PUT_ON_FLOOR requires held object {obj_key!r}",
        )
    place_xy = _floor_xy_in_front_of_anchor(scene, obj_key, anchor_key)
    place_z = float(scene.floor_z) + _held_base_offset(scene, obj_key)
    goal.meta["_floor_anchor"] = anchor_key
    place_goal = Goal(
        Primitive.PLACE,
        place_xy=place_xy,
        place_z=place_z,
        meta=goal.meta,
    )
    frames = produce_place(rig, g, scene, st, place_goal, sd_world)
    scene.tracked_near.add((obj_key, anchor_key))
    return frames


def produce_fall_to_floor(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Move an unheld edge object to the floor beside its named support, without camera steering."""
    del g, sd_world
    obj_key = str(goal.obj_key or "")
    anchor_key = str((goal.meta or {}).get("anchor") or "")
    if not obj_key or not anchor_key:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            "FALL_TO_FLOOR requires an object and support anchor",
        )
    if obj_key in scene.held or (
        st.held is not None and st.held.get("obj_key") == obj_key
    ):
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            "FALL_TO_FLOOR models an unattended object, not the held object",
        )
    current_anchor = (scene.on or {}).get(obj_key)
    if current_anchor is not None and current_anchor != anchor_key:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"{obj_key!r} is resting on {current_anchor!r}, not fall anchor {anchor_key!r}",
        )
    if obj_key in (scene.inside or {}):
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"{obj_key!r} is inside {(scene.inside or {})[obj_key]!r}, not on fall anchor {anchor_key!r}",
        )
    pose = _adjacent_floor_pose(
        scene,
        obj_key,
        anchor_key,
        hidden_from_xy=st.root_xy if st.held is not None else None,
    )
    if goal.place_xy is not None:
        pose[:2, 3] = np.asarray(goal.place_xy, dtype=np.float64)
    entity = scene.resolve(obj_key)
    if scene.stage is not None and entity.prim_path and not _write_world_col(
        scene.stage, entity.prim_path, pose
    ):
        raise PrimitiveFailure(
            PrimitiveError.NO_OBJECT,
            f"fall target {obj_key!r} has no live writable prim",
        )
    scene.declare_place(obj_key, None)
    scene.set_object_rest_pose(obj_key, pose)
    scene.tracked_near.add((obj_key, anchor_key))
    scene.invalidate_bounds()
    if st.held is None:
        st.obj_rest_xyz = pose
    base = st.last_pose156
    if base is None:
        base = _manip(rig, getattr(rig, "arm_rest", rig.arm_grasp), 0.0, 0.0)
    root_z = st.seated_z if st.seated_z is not None else _pin_z(base, st.floor_z)
    frames = [
        wpp.Frame(
            base,
            st.root_xy,
            st.root_yaw,
            root_z,
            rhp=st.held.get("rhp") if st.held is not None else None,
            fpv_mode="look",
            obj=("rest", pose),
            seg="fall_to_floor",
        )
    ]
    _stamp_obj_path(frames, scene, obj_key)
    st.last_pose156 = base
    if st.held is not None:
        resume = wpp.Frame(
            base,
            st.root_xy,
            st.root_yaw,
            root_z,
            rhp=st.held.get("rhp"),
            fpv_mode="look",
            obj=_obj_carry(st),
            seg="fall_to_floor",
        )
        frames.append(resume)
        _stamp_obj_path([resume], scene, st.held["obj_key"])
        _stamp_carried([resume], st.held.get("carried"))
    return frames


# =====================================================================================================
# ANCHORED producers: reach the effector to the object's GROUNDED contact (high/low adapts via
# the shared warp), move along a contact-frame trajectory, KINEMATICALLY couple the object, declare the
# state. open/close = JOINT_DRIVEN, sit/standup = BODY_ANCHORED. (grab/put = RIDES_EFFECTOR.)
# =====================================================================================================
_HIP_L, _HIP_R, _KNEE_L, _KNEE_R = (
    1,
    2,
    4,
    5,
)  # SMPL leg joints for the coarse seated pose


def _reach_arm_leaned(
    rig, target_world, stand_xy, yaw, stand_z, lean_deg, squat_frac: float = 0.0
):
    """warp_arm_to_seat the grip to a world point with a chosen forward lean (low targets lean in, high
    targets don't). Returns (arm dict, residual m, lean_rad)."""
    rz, world_off = wpp._grasp_frame(stand_xy, yaw, stand_z)
    lean_rad = math.radians(lean_deg)
    base = _reach_base_pose(
        rig,
        rig.arm_grasp,
        lean_rad,
        squat_frac,
        1.0,
    )
    pose, err = core.warp_arm_to_seat(
        rig, base, rz, world_off, np.asarray(target_world, dtype=np.float64)
    )
    return wpp._arm_from_pose(pose), float(err), lean_rad


def _left_action_arm(
    rig,
    target_world,
    stand_xy,
    yaw: float,
    stand_z: float,
    lean_rad: float,
    squat_frac: float,
) -> tuple[dict, float]:
    mirrored_target = _mirror_world_point_for_left_hand(
        target_world,
        stand_xy,
        yaw,
        stand_z,
    )
    right_arm, error = _solve_reach_arm(
        rig,
        rig.arm_grasp,
        mirrored_target,
        stand_xy,
        yaw,
        stand_z,
        lean_override=lean_rad,
        squat_frac=squat_frac,
    )
    return core.mirror_right_arm_as_left(right_arm), error


def _openable_reach_plan(
    rig,
    scene,
    st,
    contact,
    tolerance: float,
    *,
    minimum_squat: float = 0.0,
    maximum_squat: float | None = None,
    maximum_lean_rad: float | None = None,
    ik_budget=None,
) -> dict:
    clear = body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z)
    if not _held_right_hand(st):
        return plan_reach(
            rig,
            rig.arm_grasp,
            contact,
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            tolerance,
            floor_z=st.floor_z,
            enabled=True,
            clear_fn=clear,
            allow_colliding_fallback=False,
            upright_first=True,
            minimum_squat=minimum_squat,
            maximum_squat=maximum_squat,
            maximum_lean_rad=maximum_lean_rad,
            ik_budget=ik_budget,
        )

    mirrored = _mirror_world_point_for_left_hand(
        contact,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
    )
    plan = plan_reach(
        rig,
        rig.arm_grasp,
        mirrored,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        tolerance,
        floor_z=st.floor_z,
        enabled=True,
        clear_fn=None,
        allow_colliding_fallback=False,
        upright_first=True,
        minimum_squat=minimum_squat,
        maximum_squat=maximum_squat,
        maximum_lean_rad=maximum_lean_rad,
        ik_budget=ik_budget,
    )
    if plan.get("arm_pose") is None:
        return plan
    action_arm = core.mirror_right_arm_as_left(wpp._arm_from_pose(plan["arm_pose"]))
    combined = _merge_left_action_arm(st.held["grasp_arm"], action_arm)
    pose = _manip(
        rig,
        combined,
        1.0,
        float(plan["lf"]),
        float(plan["squat"]),
        left_hand_close=1.0,
    )
    plan["arm_pose"] = pose
    if clear is not None and not clear(pose, float(plan["stand_z"])):
        plan["reachable"] = False
        plan["collision_blocked"] = True
    return plan


def _solve_openable_action_arm(
    rig,
    st,
    target_world,
    stand_z: float,
    lean_rad: float,
    squat_frac: float,
    root_xy=None,
) -> tuple[dict, float]:
    action_root_xy = st.root_xy if root_xy is None else root_xy
    if not _held_right_hand(st):
        return _solve_reach_arm(
            rig,
            rig.arm_grasp,
            target_world,
            action_root_xy,
            st.root_yaw,
            stand_z,
            lean_override=lean_rad,
            squat_frac=squat_frac,
        )
    left_arm, error = _left_action_arm(
        rig,
        target_world,
        action_root_xy,
        st.root_yaw,
        stand_z,
        lean_rad,
        squat_frac,
    )
    return _merge_left_action_arm(st.held["grasp_arm"], left_arm), error


def _ordered_openable_arms(
    opening: bool,
    closed_arm,
    closed_error: float,
    other_arm,
    other_error: float,
) -> tuple[dict, float, dict, float]:
    if opening:
        return closed_arm, closed_error, other_arm, other_error
    return other_arm, other_error, closed_arm, closed_error


def _require_openable_arm_clear(
    rig,
    scene,
    st,
    arm_pre,
    arm_back,
    held_close: float,
    lean_fraction: float,
    squat_fraction: float,
    stand_z: float,
) -> str:
    action_side = "left" if _held_right_hand(st) else "right"
    boxes = _reach_boxes(scene)
    for arm, label in ((arm_pre, "pre_grasp"), (arm_back, "retreat")):
        pose = _manip(
            rig,
            arm,
            held_close if action_side == "left" else 1.0,
            lean_fraction,
            squat_fraction,
            left_hand_close=1.0 if action_side == "left" else 0.0,
        )
        points = _fwd_arm_points(
            pose,
            *wpp._grasp_frame(st.root_xy, st.root_yaw, stand_z),
            side=action_side,
        )
        depth, hit = _verts_in_boxes(points, boxes)
        _require_arm_clear(scene, depth, hit, label)
    return action_side


def _commit_slide_result(
    scene,
    st,
    goal,
    frames,
    end_pt,
    held,
    held_key,
    rest_pose,
    *,
    opening: bool,
    commit: bool,
):
    if not commit:
        return frames
    (scene.declare_open if opening else scene.declare_close)(goal.obj_key)
    if held_key is None:
        st.obj_rest_xyz = tuple(end_pt)
    else:
        _stamp_obj_path(frames, scene, held_key)
        _stamp_carried(frames, held.get("carried"))
    st.last_pose156 = rest_pose
    return frames


def _slide(
    rig,
    g,
    scene,
    st,
    goal,
    opening: bool,
    *,
    commit: bool = True,
) -> list[wpp.Frame]:
    """JOINT_DRIVEN: grip the handle (reach adapts to its height via the shared warp), translate along the
    open axis while the object slides to match, declare open/close. Same code for an upper cabinet (reach
    up) and a lower drawer (reach down) -- only the grounded handle point + axis differ."""
    e = scene.resolve(goal.obj_key)
    if (
        e.axis is None
    ):  # prismatic slide needs a translate axis; joint_name is a render-time-only field
        raise PrimitiveFailure(
            PrimitiveError.NOT_ARTICULATED, f"{goal.obj_key} is not openable"
        )
    handle = scene.object_pos(
        goal.obj_key
    )  # grounded closed-handle contact (high or low)
    if (
        e.handle_xy is not None
    ):  # openable CONTAINER drawer: grip the DRAWER FRONT, not the deep interior
        handle = np.array(
            [
                e.handle_xy[0],
                e.handle_xy[1],
                e.handle_z if e.handle_z is not None else handle[2],
            ],
            dtype=np.float64,
        )
    axis = np.asarray(e.axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    closed_pt, open_pt = handle, handle + axis * e.travel
    start_pt, end_pt = (closed_pt, open_pt) if opening else (open_pt, closed_pt)
    held = getattr(st, "held", None)
    held_key = str(held["obj_key"]) if held is not None else None
    held_rhp = held.get("rhp") if held is not None else None
    held_close = 1.0 if held_key is not None else 0.0
    closed_plan = _openable_reach_plan(
        rig,
        scene,
        st,
        closed_pt,
        _OPENABLE_REACH_TOL_M,
    )
    if not closed_plan["reachable"] or closed_plan.get("collision_blocked"):
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"{goal.obj_key} handle reach {closed_plan['err'] * 1000:.0f}mm",
        )
    lean_rad = float(closed_plan["lean_rad"])
    lf = float(closed_plan["lf"])
    squat_f = float(closed_plan["squat"])
    reach_stand_z = float(closed_plan["stand_z"])
    closed_arm = wpp._arm_from_pose(closed_plan["arm_pose"])
    other_pt = end_pt if opening else start_pt
    other_arm, other_err = _solve_openable_action_arm(
        rig,
        st,
        other_pt,
        reach_stand_z,
        lean_rad,
        squat_f,
    )
    arm_a, err_a, arm_b, err_b = _ordered_openable_arms(
        opening,
        closed_arm,
        float(closed_plan["err"]),
        other_arm,
        other_err,
    )
    lean_deg = math.degrees(lean_rad)
    print(
        f"[{'open' if opening else 'close'}] {goal.obj_key}: handle_z={handle[2]:.2f} "
        f"closed_reach={closed_plan['err'] * 1000:.0f}mm "
        f"sweep_reach={max(err_a, err_b) * 1000:.0f}mm "
        f"lean={lean_deg:.0f}deg squat={squat_f:.1f}"
    )

    # APPROACH-WAYPOINT: come to the CLOSED handle from the geometry-derived front (pre_grasp), and
    # back the hand out along -approach (+small up) after releasing so the arm clears the cabinet.
    wp = plan_reach_waypoints(scene, goal.obj_key, st.root_xy, closed_pt)
    approach = wp["approach"]
    arm_pre, _ = _solve_openable_action_arm(
        rig,
        st,
        wp["pre"],
        reach_stand_z,
        lean_rad,
        squat_f,
    )
    back_pt = end_pt - approach * _WP_POST_BACK + np.array([0.0, 0.0, 0.04])
    arm_back, _ = _solve_openable_action_arm(
        rig,
        st,
        back_pt,
        reach_stand_z,
        lean_rad,
        squat_f,
    )
    action_side = _require_openable_arm_clear(
        rig,
        scene,
        st,
        arm_pre,
        arm_back,
        held_close,
        lf,
        squat_f,
        reach_stand_z,
    )

    arm_rest = (
        held["grasp_arm"]
        if held is not None
        else getattr(rig, "arm_rest", rig.arm_grasp)
    )

    def action_pose(
        arm,
        action_close,
        lean_fraction=0.0,
        squat_fraction=0.0,
    ):
        return _manip(
            rig,
            arm,
            held_close if action_side == "left" else action_close,
            lean_fraction,
            squat_fraction,
            left_hand_close=action_close if action_side == "left" else 0.0,
        )

    frames = [
        replace(
            frame,
            rhp=held_rhp,
            gaze=np.asarray(start_pt, dtype=np.float64),
            openable_key=goal.obj_key,
        )
        for frame in _arrive_beats(
            rig,
            st,
            arm_rest,
            held_close,
            _obj_carry(st) if held_key is not None else ("rest", tuple(start_pt)),
        )
    ]

    def add(pose, obj_pt, seg, contact=None, root_z=None):
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                st.root_yaw,
                st.stand_z if root_z is None else float(root_z),
                rhp=held_rhp,
                fpv_mode="look",
                obj=(
                    _obj_carry(st) if held_key is not None else ("rest", tuple(obj_pt))
                ),
                seg=seg,
                contact_anchor=contact,
                gaze=np.asarray(obj_pt, dtype=np.float64),
                openable_key=goal.obj_key,
            )
        )

    n_pre = max(1, int(round(_N_REACH * 0.45)))
    for i in range(
        n_pre
    ):  # rest -> pre_grasp: hand OPEN, come in from the front (no lean yet)
        s = wpp._smoothstep((i + 1) / n_pre)
        add(
            action_pose(core.lerp_arm(arm_rest, arm_pre, s), 0.0, 0.0),
            start_pt,
            "reach",
        )
    for i in range(
        _N_REACH
    ):  # pre_grasp -> handle at the start position, hand closing + lean in
        s = wpp._smoothstep((i + 1) / _N_REACH)
        # Hand closing onto the handle at start_pt -> FIXED contact (on the handle).
        add(
            action_pose(
                core.lerp_arm(arm_pre, arm_a, s),
                s,
                lf * s,
                squat_f * s,
            ),
            start_pt,
            "reach",
            contact=np.asarray(start_pt, dtype=np.float64),
            root_z=st.stand_z + (reach_stand_z - st.stand_z) * s,
        )
    for i in range(_N_PLACE):  # pull: arm start->end while the object slides start->end
        s = wpp._smoothstep((i + 1) / _N_PLACE)
        # Hand gripping the handle throughout the slide -> FIXED contact at the CURRENT handle point.
        slide_pt = start_pt + (end_pt - start_pt) * s
        add(
            action_pose(core.lerp_arm(arm_a, arm_b, s), 1.0, lf, squat_f),
            slide_pt,
            "slide",
            contact=np.asarray(slide_pt, dtype=np.float64),
            root_z=reach_stand_z,
        )
    n_back = max(1, int(round(_N_RETRACT * 0.5)))
    for i in range(
        n_back
    ):  # back-out: release + back the hand off along -approach (object stays)
        s = wpp._smoothstep((i + 1) / n_back)
        # Still on the handle at end_pt while the grip opens (release) -> FIXED contact.
        add(
            action_pose(
                core.lerp_arm(arm_b, arm_back, s),
                1.0 - s,
                lf,
                squat_f,
            ),
            end_pt,
            "retract",
            contact=np.asarray(end_pt, dtype=np.float64),
            root_z=reach_stand_z,
        )
    for i in range(
        _N_RETRACT
    ):  # arm back to rest from the backed-off pose; object stays at the end position
        s = wpp._smoothstep((i + 1) / _N_RETRACT)
        # Hand OPEN, leaving the handle for rest -> NOT grip contact (free to nudge).
        add(
            action_pose(
                core.lerp_arm(arm_back, arm_rest, s),
                0.0,
                lf * (1 - s),
                squat_f * (1 - s),
            ),
            end_pt,
            "retract",
            root_z=reach_stand_z + (st.stand_z - reach_stand_z) * s,
        )

    _require_fpv_eye_clear(
        scene,
        frames,
        goal,
        "OPEN" if opening else "CLOSE",
    )
    # Do NOT move the openable's grounded handle to end_pt: an openable's grounding is its CLOSED
    # handle (the fixed reach reference), so open derives closed->open and a later close derives
    # open->closed correctly. (Moving it to the open pos would make the close slide open->further-open,
    # a geometric no-op that a check of declared state alone, such as --step3, cannot see.)
    return _commit_slide_result(
        scene,
        st,
        goal,
        frames,
        end_pt,
        held,
        held_key,
        action_pose(arm_rest, 0.0),
        opening=opening,
        commit=commit,
    )


def _require_bound_openable(scene: SceneState, key: str) -> None:
    controllers = getattr(scene, "_openable_controllers", None)
    if controllers and key not in controllers:
        raise PrimitiveFailure(
            PrimitiveError.NOT_ARTICULATED,
            f"{key} has no registered openable controller",
        )


def produce_open(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    # Route by joint TYPE so [Open] drives ANY BEHAVIOR openable, not just prismatic drawers: a revolute
    # openable (hinge_xy set -- a door / fridge / cabinet leaf) SWINGS; a prismatic one (drawer) SLIDES.
    _require_bound_openable(scene, goal.obj_key)
    e = scene.resolve(goal.obj_key)
    if getattr(e, "hinge_xy", None) is not None:
        frames = _swing(rig, g, scene, st, goal, opening=True, relax_reach=False)
        # A swung-open CONTAINER leaf sticks into the room; block its footprint in the live nav grid so a
        # later planned walk routes AROUND it instead of THROUGH the open door. Freed again on close.
        if (
            getattr(e, "leaf_paths", None)
            and (getattr(e, "meta", None) or {}).get("openable_kind") != "lid"
            and getattr(scene, "nav_grid", None) is not None
        ):
            scene.mark_leaf_obstacle(goal.obj_key)
        return frames
    return _slide(rig, g, scene, st, goal, opening=True)


def produce_close(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    _require_bound_openable(scene, goal.obj_key)
    e = scene.resolve(goal.obj_key)
    if getattr(e, "hinge_xy", None) is not None:
        frames = _swing(rig, g, scene, st, goal, opening=False, relax_reach=False)
        if (
            getattr(e, "leaf_paths", None)
            and (getattr(e, "meta", None) or {}).get("openable_kind") != "lid"
            and getattr(scene, "nav_grid", None) is not None
        ):
            scene.unmark_leaf_obstacle(
                goal.obj_key
            )  # door shut -> free the leaf's nav footprint
        return frames
    return _slide(rig, g, scene, st, goal, opening=False)


def _semantic_door_swing(
    rig,
    scene,
    st,
    goal,
    opening: bool,
    arc,
    f_start: float,
    f_end: float,
    start_pt: np.ndarray,
) -> list[wpp.Frame]:
    """Animate a navigation door without requiring hand-to-handle contact."""
    if _held_right_hand(st):
        held = st.held
        held_key = str(held["obj_key"])
        arm_rest = held["grasp_arm"]
        left_start, _ = _left_action_arm(
            rig,
            arc(f_start),
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            0.0,
            0.0,
        )
        left_end, _ = _left_action_arm(
            rig,
            arc(f_end),
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            0.0,
            0.0,
        )
        arm_start = _merge_left_action_arm(arm_rest, left_start)
        arm_end = _merge_left_action_arm(arm_rest, left_end)
        frames = [
            replace(
                frame,
                rhp=held.get("rhp"),
                gaze=np.asarray(start_pt, dtype=np.float64),
                openable_key=goal.obj_key,
            )
            for frame in _arrive_beats(
                rig,
                st,
                arm_rest,
                1.0,
                _obj_carry(st),
            )
        ]
        for index in range(_N_REACH):
            fraction = wpp._smoothstep((index + 1) / _N_REACH)
            frames.append(
                wpp.Frame(
                    _manip(
                        rig,
                        core.lerp_arm(arm_rest, arm_start, fraction),
                        1.0,
                        left_hand_close=fraction,
                    ),
                    st.root_xy,
                    st.root_yaw,
                    st.stand_z,
                    rhp=held.get("rhp"),
                    fpv_mode="look",
                    obj=_obj_carry(st),
                    seg="reach",
                    gaze=np.asarray(start_pt, dtype=np.float64),
                    openable_key=goal.obj_key,
                )
            )
        for index in range(_N_PLACE):
            fraction = wpp._smoothstep((index + 1) / _N_PLACE)
            point = np.asarray(
                arc(f_start + (f_end - f_start) * fraction), dtype=np.float64
            )
            frames.append(
                wpp.Frame(
                    _manip(
                        rig,
                        core.lerp_arm(arm_start, arm_end, fraction),
                        1.0,
                        left_hand_close=1.0,
                    ),
                    st.root_xy,
                    st.root_yaw,
                    st.stand_z,
                    rhp=held.get("rhp"),
                    fpv_mode="look",
                    obj=_obj_carry(st),
                    seg="swing",
                    gaze=point,
                    openable_key=goal.obj_key,
                )
            )
        for index in range(_N_RETRACT):
            fraction = wpp._smoothstep((index + 1) / _N_RETRACT)
            frames.append(
                wpp.Frame(
                    _manip(
                        rig,
                        core.lerp_arm(arm_end, arm_rest, fraction),
                        1.0,
                        left_hand_close=1.0 - fraction,
                    ),
                    st.root_xy,
                    st.root_yaw,
                    st.stand_z,
                    rhp=held.get("rhp"),
                    fpv_mode="look",
                    obj=_obj_carry(st),
                    seg="retract",
                    gaze=np.asarray(arc(f_end), dtype=np.float64),
                    openable_key=goal.obj_key,
                )
            )
        (scene.declare_open if opening else scene.declare_close)(goal.obj_key)
        st.last_pose156 = _manip(rig, arm_rest, 1.0, 0.0)
        _stamp_obj_path(frames, scene, held_key)
        _stamp_carried(frames, held.get("carried"))
        return frames

    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    frames = [
        replace(frame, openable_key=goal.obj_key)
        for frame in _arrive_beats(rig, st, arm_rest, 0.0, ("rest", tuple(start_pt)))
    ]
    for i in range(_N_PLACE):
        s = wpp._smoothstep((i + 1) / _N_PLACE)
        frames.append(
            wpp.Frame(
                _manip(rig, arm_rest, 0.0, 0.0),
                st.root_xy,
                st.root_yaw,
                st.stand_z,
                fpv_mode="look",
                obj=("rest", tuple(arc(f_start + (f_end - f_start) * s))),
                seg="swing",
                openable_key=goal.obj_key,
            )
        )
    (scene.declare_open if opening else scene.declare_close)(goal.obj_key)
    st.last_pose156 = _manip(rig, arm_rest, 0.0, 0.0)
    logger.info(
        "[%s] %s: semantic kinematic swing; exact handle contact skipped",
        "opendoor" if opening else "closedoor",
        goal.obj_key,
    )
    return frames


def _automatic_lid_swing(
    rig,
    scene,
    st,
    goal,
    arc,
    f_start: float,
    f_end: float,
) -> list[wpp.Frame]:
    """Open a lid while preserving the object already carried by the working hand."""
    if st.held is None:
        return []
    held_key = str(st.held["obj_key"])
    pose = _manip(rig, st.held["grasp_arm"], 1.0, 0.0)
    frames = []
    for i in range(_N_PLACE):
        fraction = wpp._smoothstep((i + 1) / _N_PLACE)
        frames.append(
            wpp.Frame(
                pose.copy(),
                st.root_xy,
                st.root_yaw,
                st.stand_z,
                fpv_mode="look",
                obj=_obj_carry(st),
                seg="lid_open",
                gaze=np.asarray(
                    arc(f_start + (f_end - f_start) * fraction), dtype=np.float64
                ),
                openable_key=goal.obj_key,
            )
        )
    scene.declare_open(goal.obj_key)
    st.last_pose156 = pose
    _stamp_obj_path(frames, scene, held_key)
    _stamp_carried(frames, st.held.get("carried"))
    return frames


def _openable_camera_focus(entity, hinge, closed_handle, axis):
    """Keep vertical-door FPV aimed at the aperture instead of chasing the leaf."""
    if abs(float(axis[2])) < 0.7:
        return None
    meta = getattr(entity, "meta", None) or {}
    if meta.get("action_camera_focus") == "handle":
        return None
    aperture = np.asarray(meta.get("aperture_center", ()), dtype=np.float64)
    if aperture.shape == (3,) and np.all(np.isfinite(aperture)):
        return aperture
    focus = (np.asarray(hinge, dtype=np.float64) + closed_handle) * 0.5
    focus[2] = float(closed_handle[2])
    return focus


def _swing(
    rig,
    g,
    scene,
    st,
    goal,
    opening: bool,
    relax_reach: bool = False,
    commit: bool = True,
) -> list[wpp.Frame]:
    """JOINT_DRIVEN (revolute): grip the door handle and sweep it along a circular ARC about the vertical
    hinge line by the door's open angle, declaring open/close. Mirrors ``_slide``'s beat structure and
    reach machinery; the ONLY difference is the handle traces an arc (rotate about ``hinge_xy``) instead
    of a straight translate. A prismatic drawer has ``hinge_xy=None`` and never reaches here."""
    e = scene.resolve(goal.obj_key)
    if (
        e.hinge_xy is None
    ):  # revolute arc needs a hinge; joint_name is a render-time-only field
        raise PrimitiveFailure(
            PrimitiveError.NOT_ARTICULATED, f"{goal.obj_key} is not a swing door"
        )
    handle = scene.object_pos(goal.obj_key)  # grounded CLOSED-handle contact
    if (
        e.handle_xy is not None
    ):  # openable CONTAINER: grip the DOOR FRONT, not the deep interior object_pos
        handle = np.array(
            [
                e.handle_xy[0],
                e.handle_xy[1],
                e.handle_z if e.handle_z is not None else handle[2],
            ],
            dtype=np.float64,
        )
    hinge = np.asarray(
        (getattr(e, "meta", None) or {}).get(
            "hinge_xyz", (e.hinge_xy[0], e.hinge_xy[1], handle[2])
        ),
        dtype=np.float64,
    )
    axis = np.asarray(getattr(e, "hinge_axis", (0.0, 0.0, 1.0)), dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        raise PrimitiveFailure(
            PrimitiveError.NOT_ARTICULATED,
            f"{goal.obj_key} has a zero-length hinge axis",
        )
    axis /= axis_norm
    closed_handle = handle.copy()
    contact_fraction = max(
        0.20,
        min(1.0, float((goal.meta or {}).get("_door_contact_fraction", 1.0))),
    )
    handle = hinge + (handle - hinge) * contact_fraction
    camera_focus = _openable_camera_focus(e, hinge, closed_handle, axis)
    theta_open = float(e.swing_sign) * float(e.swing_angle)

    def _arc(frac: float) -> np.ndarray:
        return _rotate_point_about_axis(
            handle,
            hinge,
            axis,
            theta_open * frac,
        )

    f_start, f_end = (0.0, 1.0) if opening else (1.0, 0.0)
    start_pt, end_pt = _arc(f_start), _arc(f_end)
    closed_pt = _arc(0.0)  # the CLOSED handle -> the approach + reach reference
    if (
        opening
        and (goal.meta or {}).get("_implicit_action") == "auto_open_for_placement"
        and (e.meta or {}).get("openable_kind") == "lid"
        and st.held is not None
    ):
        return _automatic_lid_swing(
            rig,
            scene,
            st,
            goal,
            _arc,
            f_start,
            f_end,
        )
    if relax_reach and bool((goal.meta or {}).get("_semantic_open", False)):
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"{goal.obj_key} cannot be opened without a reachable front-facing stand",
        )
    held = getattr(st, "held", None)
    held_key = str(held["obj_key"]) if held is not None else None
    held_rhp = held.get("rhp") if held is not None else None
    held_close = 1.0 if held_key is not None else 0.0
    access_mode = (getattr(e, "meta", None) or {}).get("access_mode")
    front_access = access_mode == "front_drum"
    low_front = _low_front_openable(scene, e)
    tall_front = access_mode == "front_opening" and not low_front
    forced_crouch = front_access or low_front
    closing_swing_retreat = False
    closing_camera_eye_offset = None
    if tall_front:
        if opening:
            entry_axis = np.asarray(
                (getattr(e, "meta", None) or {}).get("entry_axis", ()),
                dtype=np.float64,
            ).reshape(-1)
            entry_norm = float(np.linalg.norm(entry_axis))
            if entry_axis.shape == (2,) and entry_norm > 1e-9:
                default_retreat = max(
                    0.0,
                    _FRONT_OPENING_ACTION_CAMERA_STANDOFF_M
                    - _FRONT_OPENING_STANDOFF_M,
                )
                configured_retreat = (getattr(e, "meta", None) or {}).get(
                    "opening_pullback_m"
                )
                retreat = (
                    max(default_retreat, float(configured_retreat))
                    if configured_retreat is not None
                    else default_retreat
                )
                safe_root = np.asarray(st.root_xy, dtype=np.float64) - (
                    entry_axis / entry_norm
                ) * retreat
                end_root_xy = float(safe_root[0]), float(safe_root[1])
            else:
                end_root_xy = openable_pullback_xy(e, st.root_xy, opening)
        else:
            entity_meta = getattr(e, "meta", None) or {}
            swing_retreat_m = max(
                0.0, float(entity_meta.get("closing_swing_retreat_m", 0.0))
            )
            entry_axis = np.asarray(
                entity_meta.get("entry_axis", ()), dtype=np.float64
            ).reshape(-1)
            entry_norm = float(np.linalg.norm(entry_axis))
            if (
                swing_retreat_m > 0.0
                and entry_axis.shape == (2,)
                and entry_norm > 1e-9
            ):
                safe_root = np.asarray(st.root_xy, dtype=np.float64) - (
                    entry_axis / entry_norm
                ) * swing_retreat_m
                end_root_xy = float(safe_root[0]), float(safe_root[1])
                closing_swing_retreat = True
            else:
                end_root_xy = (
                    openable_pullback_xy(e, st.root_xy, opening)
                    if "closing_pullback_m" in entity_meta
                    else tuple(st.root_xy)
                )
            closing_camera_eye_offset = _closing_camera_offset_world(
                entity_meta,
                entry_axis,
            )
    else:
        end_root_xy = openable_pullback_xy(e, st.root_xy, opening)
    active_crouch = _active_crouch_fraction(scene, st, goal.obj_key)
    minimum_squat = (
        max(_FRONT_APPLIANCE_SQUAT, active_crouch) if forced_crouch else 0.0
    )
    reach_tol = _OPENABLE_REACH_TOL_M
    action_contact = start_pt if tall_front and not opening else closed_pt
    closed_plan = _openable_reach_plan(
        rig,
        scene,
        st,
        action_contact,
        reach_tol,
        minimum_squat=minimum_squat,
        maximum_squat=0.0 if tall_front else None,
        maximum_lean_rad=(
            _FRONT_OPENING_ACTION_MAX_LEAN_RAD if tall_front else None
        ),
    )
    if not closed_plan["reachable"] or closed_plan.get("collision_blocked"):
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"{goal.obj_key} door contact reach {closed_plan['err'] * 1000:.0f}mm",
        )
    lean_rad = float(closed_plan["lean_rad"])
    lf = float(closed_plan["lf"])
    squat_f = float(closed_plan["squat"])
    reach_stand_z = float(closed_plan["stand_z"])
    contact_arm = wpp._arm_from_pose(closed_plan["arm_pose"])
    if tall_front and not opening:
        arm_a = contact_arm
        err_a = float(closed_plan["err"])
        arm_b, err_b = _solve_openable_action_arm(
            rig,
            st,
            end_pt,
            st.stand_z,
            0.0,
            0.0,
            end_root_xy,
        )
    else:
        other_pt = end_pt if opening else start_pt
        if held is None and not forced_crouch:
            other_arm, other_err, _ = _reach_arm_leaned(
                rig,
                other_pt,
                end_root_xy,
                st.root_yaw,
                reach_stand_z,
                math.degrees(lean_rad),
                squat_frac=squat_f,
            )
        else:
            other_arm, other_err = _solve_openable_action_arm(
                rig,
                st,
                other_pt,
                reach_stand_z,
                lean_rad,
                squat_f,
                root_xy=end_root_xy,
            )
        arm_a, err_a, arm_b, err_b = _ordered_openable_arms(
            opening,
            contact_arm,
            float(closed_plan["err"]),
            other_arm,
            other_err,
        )
    # Gate ONLY the CLOSED-handle reach (stand_pose_for is derived for it -> guaranteed). The OPEN
    # endpoint is best-effort: a wide ~90deg arc carries the handle past the arm envelope, but the leaf
    # is driven independent of the arm (exactly as the drawer body slides regardless of arm reach), so
    # the door ALWAYS swings -- OpenDoor never fails on the far handle.
    reach_closed = err_a if opening or tall_front else err_b
    reach_sweep = max(err_a, err_b)
    if not relax_reach and reach_closed > reach_tol:
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"{goal.obj_key} door contact reach {reach_closed * 1000:.0f}mm",
        )
    if relax_reach and reach_closed > 3 * _REACH_TOL_M:
        # RELAXED (DOORS only via produce_open_door/close_door): the leaf swings
        # KINEMATICALLY, independent of the arm (per the arc-drive below), so a door opens even when the
        # single arm cannot precisely reach the wall-plane handle -- the hand does a best-effort gesture and
        # the door swings. FPV-first; naturalness down-weighted for completion. Container open/close keep the
        # strict gate (relax_reach defaults False), so non-door scenes are unaffected.
        print(
            f"[{'opendoor' if opening else 'closedoor'}] {goal.obj_key}: handle reach "
            f"{reach_closed * 1000:.0f}mm > tol -> RELAXED (kinematic swing, best-effort arm)"
        )
    print(
        f"[{'opendoor' if opening else 'closedoor'}] {goal.obj_key}: handle_z={handle[2]:.2f} "
        f"hinge=({hinge[0]:.2f},{hinge[1]:.2f}) swing={math.degrees(theta_open):.0f}deg "
        f"contact_fraction={contact_fraction:.2f} reach_closed={reach_closed * 1000:.0f}mm "
        f"reach_sweep={reach_sweep * 1000:.0f}mm"
    )

    boxes = _reach_boxes(scene)
    wp = plan_reach_waypoints(scene, goal.obj_key, st.root_xy, action_contact)
    approach = wp["approach"]
    back_pt = end_pt - approach * _WP_POST_BACK + np.array([0.0, 0.0, 0.04])
    if held is None and not forced_crouch:
        arm_pre, _, _, _ = _solve_and_check(
            rig,
            rig.arm_grasp,
            wp["pre"],
            st.root_xy,
            st.root_yaw,
            st.stand_z,
            boxes,
            f"{'opendoor' if opening else 'closedoor'} pre_grasp",
            strict=_body_collision_is_hard(scene),
        )
        arm_back, _, _, _ = _solve_and_check(
            rig,
            arm_b,
            back_pt,
            end_root_xy,
            st.root_yaw,
            st.stand_z,
            boxes,
            f"{'opendoor' if opening else 'closedoor'} retreat",
            strict=_body_collision_is_hard(scene),
        )
    else:
        arm_pre, _ = _solve_openable_action_arm(
            rig,
            st,
            wp["pre"],
            reach_stand_z,
            lean_rad,
            squat_f,
        )
        arm_back, _ = _solve_openable_action_arm(
            rig,
            st,
            back_pt,
            reach_stand_z,
            lean_rad,
            squat_f,
            root_xy=end_root_xy,
        )
    action_side = "left" if _held_right_hand(st) else "right"

    def _check_action_arm(arm, label, root_xy):
        pose = _manip(
            rig,
            arm,
            held_close if action_side == "left" else 1.0,
            lf,
            squat_f,
            left_hand_close=1.0 if action_side == "left" else 0.0,
        )
        arm_points = _fwd_arm_points(
            pose,
            *wpp._grasp_frame(root_xy, st.root_yaw, reach_stand_z),
            side=action_side,
        )
        depth, hit = _verts_in_boxes(arm_points, boxes)
        print(f"[reach] {label}: arm-clear worst={depth * 1000:.0f}mm hit={hit or '-'}")
        _require_arm_clear(scene, depth, hit, label)

    if held is not None or forced_crouch:
        _check_action_arm(
            arm_pre,
            f"{'opendoor' if opening else 'closedoor'} pre_grasp",
            st.root_xy,
        )
        _check_action_arm(
            arm_back,
            f"{'opendoor' if opening else 'closedoor'} retreat",
            end_root_xy,
        )
    arm_rest = (
        held["grasp_arm"]
        if held is not None
        else getattr(rig, "arm_rest", rig.arm_grasp)
    )
    swing_start_arm = arm_a
    if tall_front and opening:
        swing_start_arm, _ = _solve_openable_action_arm(
            rig,
            st,
            start_pt,
            st.stand_z,
            0.0,
            0.0,
            root_xy=end_root_xy,
        )

    def action_pose(arm, action_close, lean_fraction=0.0, squat_fraction=0.0):
        return _manip(
            rig,
            arm,
            held_close if action_side == "left" else action_close,
            lean_fraction,
            squat_fraction,
            left_hand_close=action_close if action_side == "left" else 0.0,
        )

    prior_last_pose = getattr(st, "last_pose156", None)
    prior_walk_end_yaw = getattr(st, "walk_end_yaw", None)
    frames = [
        replace(
            frame,
            rhp=held_rhp,
            contact_anchor=(
                np.asarray(start_pt, dtype=np.float64)
                if held_key is not None
                else getattr(frame, "contact_anchor", None)
            ),
            gaze=np.asarray(
                start_pt if camera_focus is None else camera_focus,
                dtype=np.float64,
            ),
            openable_key=goal.obj_key,
        )
        for frame in _arrive_beats(
            rig,
            st,
            arm_rest,
            held_close,
            _obj_carry(st) if held_key is not None else ("rest", tuple(start_pt)),
            squat_frac=squat_f if forced_crouch else 0.0,
        )
    ]

    def add(
        pose,
        obj_pt,
        seg,
        contact=None,
        root_z=None,
        root_xy=None,
        camera_eye_offset=None,
    ):
        gaze = obj_pt if camera_focus is None else camera_focus
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy if root_xy is None else root_xy,
                st.root_yaw,
                st.stand_z if root_z is None else float(root_z),
                rhp=held_rhp,
                fpv_mode="look",
                obj=(
                    _obj_carry(st) if held_key is not None else ("rest", tuple(obj_pt))
                ),
                seg=seg,
                camera_eye_offset=(
                    None
                    if camera_eye_offset is None
                    else np.asarray(camera_eye_offset, dtype=np.float64)
                ),
                contact_anchor=contact,
                gaze=np.asarray(gaze, dtype=np.float64),
                openable_key=goal.obj_key,
            )
        )

    def closing_camera_offset(weight: float):
        if closing_camera_eye_offset is None:
            return None
        return closing_camera_eye_offset * max(0.0, min(1.0, float(weight)))

    n_pre = max(1, int(round(_N_REACH * 0.45)))
    camera_reach_frames = n_pre + _N_REACH
    for i in range(n_pre):  # rest -> pre_grasp: hand OPEN, come in from the front
        s = wpp._smoothstep((i + 1) / n_pre)
        pre_root_xy = None
        if (
            not opening
            and not closing_swing_retreat
            and end_root_xy != st.root_xy
        ):
            pre_root_xy = (
                st.root_xy[0] + (end_root_xy[0] - st.root_xy[0]) * s,
                st.root_xy[1] + (end_root_xy[1] - st.root_xy[1]) * s,
            )
        add(
            action_pose(
                core.lerp_arm(arm_rest, arm_pre, s),
                0.0,
                0.0,
                squat_f if forced_crouch else 0.0,
            ),
            start_pt,
            "reach",
            contact=(
                np.asarray(start_pt, dtype=np.float64)
                if held_key is not None
                else None
            ),
            root_z=reach_stand_z if forced_crouch else None,
            root_xy=pre_root_xy,
            camera_eye_offset=closing_camera_offset(
                wpp._smoothstep((i + 1) / camera_reach_frames)
            ),
        )
    for i in range(
        _N_REACH
    ):  # pre_grasp -> handle at the start angle, hand closing + lean in
        s = wpp._smoothstep((i + 1) / _N_REACH)
        add(
            action_pose(
                core.lerp_arm(arm_pre, arm_a, s),
                s,
                lf * s,
                squat_f if forced_crouch else squat_f * s,
            ),
            start_pt,
            "reach",
            contact=np.asarray(start_pt, dtype=np.float64),
            root_z=(
                reach_stand_z
                if forced_crouch
                else st.stand_z + (reach_stand_z - st.stand_z) * s
            ),
            root_xy=(
                end_root_xy
                if not opening and not closing_swing_retreat
                else None
            ),
            camera_eye_offset=closing_camera_offset(
                wpp._smoothstep((n_pre + i + 1) / camera_reach_frames)
            ),
        )
    for _ in range(_N_HANDLE_HOLD):
        add(
            action_pose(arm_a, 1.0, lf, squat_f),
            start_pt,
            "grasp",
            contact=np.asarray(start_pt, dtype=np.float64),
            root_z=reach_stand_z,
            root_xy=(
                end_root_xy
                if not opening and not closing_swing_retreat
                else None
            ),
            camera_eye_offset=closing_camera_offset(1.0),
        )
    if tall_front and opening:
        for i in range(_N_ARRIVE):
            s = wpp._smoothstep((i + 1) / _N_ARRIVE)
            retreat_xy = (
                st.root_xy[0] + (end_root_xy[0] - st.root_xy[0]) * s,
                st.root_xy[1] + (end_root_xy[1] - st.root_xy[1]) * s,
            )
            add(
                action_pose(
                    core.lerp_arm(arm_a, swing_start_arm, s),
                    1.0,
                    lf * (1.0 - s),
                    0.0,
                ),
                start_pt,
                "pullback",
                contact=np.asarray(start_pt, dtype=np.float64),
                root_z=st.stand_z,
                root_xy=retreat_xy,
            )
    swing_end_arm = arm_b
    swing_retreat_frames = max(1, math.ceil(_N_PLACE * 0.5))
    for i in range(
        _N_PLACE
    ):  # swing: arm start->end while the leaf rotates along the arc
        s = wpp._smoothstep((i + 1) / _N_PLACE)
        arc_pt = _arc(f_start + (f_end - f_start) * s)
        swing_root_xy = end_root_xy
        swing_arm = core.lerp_arm(swing_start_arm, arm_b, s)
        if closing_swing_retreat:
            retreat_s = wpp._smoothstep(
                min(1.0, (i + 1) / swing_retreat_frames)
            )
            swing_root_xy = (
                st.root_xy[0]
                + (end_root_xy[0] - st.root_xy[0]) * retreat_s,
                st.root_xy[1]
                + (end_root_xy[1] - st.root_xy[1]) * retreat_s,
            )
            swing_arm, _ = _solve_openable_action_arm(
                rig,
                st,
                arc_pt,
                reach_stand_z,
                0.0,
                0.0,
                root_xy=swing_root_xy,
            )
            swing_end_arm = swing_arm
        # Hand gripping the handle throughout the swing -> FIXED contact at the CURRENT arc point; the
        # ("rest", arc_pt) coupling is what _drive_drawers' revolute branch routes to the door leaf.
        add(
            action_pose(
                swing_arm,
                1.0,
                0.0 if tall_front else lf,
                0.0 if tall_front else squat_f,
            ),
            arc_pt,
            "swing",
            contact=np.asarray(arc_pt, dtype=np.float64),
            root_z=reach_stand_z,
            root_xy=swing_root_xy,
            camera_eye_offset=closing_camera_offset(1.0),
        )
    n_back = max(1, int(round(_N_RETRACT * 0.5)))
    camera_retract_frames = n_back + _N_RETRACT
    for i in range(
        n_back
    ):  # back-out: release + back the hand off (leaf stays at the end angle)
        s = wpp._smoothstep((i + 1) / n_back)
        add(
            action_pose(
                core.lerp_arm(swing_end_arm, arm_back, s),
                1.0 - s,
                lf,
                squat_f,
            ),
            end_pt,
            "retract",
            contact=np.asarray(end_pt, dtype=np.float64),
            root_z=reach_stand_z,
            root_xy=end_root_xy,
            camera_eye_offset=closing_camera_offset(
                1.0 - wpp._smoothstep((i + 1) / camera_retract_frames)
            ),
        )
    for i in range(_N_RETRACT):  # arm back to rest; leaf stays at the end angle
        s = wpp._smoothstep((i + 1) / _N_RETRACT)
        final_squat = squat_f if forced_crouch else 0.0
        retract_squat = squat_f + (final_squat - squat_f) * s
        retract_pose = action_pose(
            core.lerp_arm(arm_back, arm_rest, s),
            0.0,
            lf * (1 - s),
            retract_squat,
        )
        add(
            retract_pose,
            end_pt,
            "retract",
            contact=(
                np.asarray(end_pt, dtype=np.float64)
                if held_key is not None
                else None
            ),
            root_z=(
                _pin_z(retract_pose, st.floor_z)
                if retract_squat > 1e-6
                else st.stand_z
            ),
            root_xy=end_root_xy,
            camera_eye_offset=closing_camera_offset(
                1.0
                - wpp._smoothstep(
                    (n_back + i + 1) / camera_retract_frames
                )
            ),
        )
    if forced_crouch and not opening:
        for i in range(_N_ARRIVE):
            s = wpp._smoothstep((i + 1) / _N_ARRIVE)
            standing_squat = squat_f * (1.0 - s)
            standing_pose = action_pose(
                arm_rest,
                0.0,
                0.0,
                standing_squat,
            )
            add(
                standing_pose,
                end_pt,
                "stand",
                contact=(
                    np.asarray(end_pt, dtype=np.float64)
                    if held_key is not None
                    else None
                ),
                root_z=(
                    _pin_z(standing_pose, st.floor_z)
                    if standing_squat > 1e-6
                    else st.stand_z
                ),
                root_xy=end_root_xy,
            )

    if held_key is not None:
        _stamp_obj_path(frames, scene, held_key)
        _stamp_carried(frames, held.get("carried"))

    try:
        _require_fpv_eye_clear(
            scene,
            frames,
            goal,
            "OPEN" if opening else "CLOSE",
        )
    except PrimitiveFailure:
        st.last_pose156 = prior_last_pose
        if prior_walk_end_yaw is not None:
            st.walk_end_yaw = prior_walk_end_yaw
        raise

    if not commit:
        return frames

    (scene.declare_open if opening else scene.declare_close)(goal.obj_key)
    # Keep the door's grounding at the CLOSED handle (the fixed reach reference), exactly as _slide does
    # for a drawer. Do NOT touch st.obj_rest_xyz: a door is not a carried graspable, and clobbering it
    # would teleport a held/rest object onto the door handle.
    st.last_pose156 = action_pose(
        arm_rest,
        0.0,
        0.0,
        squat_f if forced_crouch and opening else 0.0,
    )
    st.root_xy = end_root_xy
    if forced_crouch and opening:
        _set_crouch_hold(st, goal.obj_key, squat_f, st.last_pose156)
    elif forced_crouch:
        _clear_crouch_hold(st)
    return frames


def _closing_camera_offset_world(entity_meta, entry_axis):
    entry = np.asarray(entry_axis, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(entry))
    if entry.shape != (2,) or norm <= 1e-9:
        return None
    entry_unit = entry / norm
    local = entity_meta.get("closing_camera_door_local_offset_m")
    if local is not None:
        lateral_unit = np.array((-entry_unit[1], entry_unit[0]))
        horizontal = lateral_unit * float(local["lateral"]) + entry_unit * float(
            local["entry"]
        )
        return np.array(
            (horizontal[0], horizontal[1], float(local["vertical"])),
            dtype=np.float64,
        )
    lateral = max(
        0.0,
        float(entity_meta.get("closing_camera_door_lateral_offset_m", 0.0)),
    )
    if lateral <= 0.0:
        return None
    return np.array(
        (-entry_unit[1] * lateral, entry_unit[0] * lateral, 0.0),
        dtype=np.float64,
    )


def openable_action_is_safe_from_stand(
    rig,
    g,
    scene,
    st,
    goal,
    stand_xy,
    yaw: float,
    *,
    camera_frame_offset: int = 0,
) -> bool:
    """Preview a complete open/close action without mutating live state."""
    door_action = goal.prim in (Primitive.OPENDOOR, Primitive.CLOSEDOOR)
    if goal.prim not in (
        Primitive.OPEN,
        Primitive.CLOSE,
        Primitive.OPENDOOR,
        Primitive.CLOSEDOOR,
    ):
        return False
    last_pose = getattr(st, "last_pose156", None)
    preview_updates = {
        "root_xy": (float(stand_xy[0]), float(stand_xy[1])),
        "root_yaw": float(yaw),
        "walk_end_yaw": float(yaw),
        "last_pose156": (
            None
            if last_pose is None
            else np.asarray(last_pose, dtype=np.float64).copy()
        ),
    }
    try:
        preview_state = replace(st, **preview_updates)
    except TypeError:
        preview_state = copy.copy(st)
        for field, value in preview_updates.items():
            setattr(preview_state, field, value)
    preview_goal = replace(
        goal,
        meta={
            **(goal.meta or {}),
            "_camera_frame_offset": int(camera_frame_offset),
        },
    )
    try:
        entity = scene.resolve(goal.obj_key)
        if door_action and getattr(entity, "hinge_xy", None) is None:
            return False
        preview = _swing if getattr(entity, "hinge_xy", None) is not None else _slide
        kwargs = {
            "opening": goal.prim in (Primitive.OPEN, Primitive.OPENDOOR),
            "commit": False,
        }
        if preview is _swing:
            kwargs["relax_reach"] = door_action
        preview(rig, g, scene, preview_state, preview_goal, **kwargs)
    except Exception as error:  # noqa: BLE001 - a safety preview fails closed
        logger.info(
            "[open-preview] %s rejected stand (%.2f, %.2f, yaw=%.1f): %s",
            goal.obj_key,
            float(stand_xy[0]),
            float(stand_xy[1]),
            float(yaw),
            error,
        )
        return False
    return True


def produce_open_door(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    frames = _swing(rig, g, scene, st, goal, opening=True, relax_reach=True)
    # Free the doorway in the live nav grid so the FOLLOWING walk routes through the now-open door (the
    # runtime occ kept it blocking as a backstop). No-op without a nav grid.
    if hasattr(scene, "carve_door_open"):
        scene.carve_door_open(goal.obj_key)
    return frames


def produce_close_door(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    frames = _swing(rig, g, scene, st, goal, opening=False, relax_reach=True)
    if hasattr(scene, "restore_door"):
        scene.restore_door(goal.obj_key)
    return frames


def _toggle_keyframes(scene: SceneState, key: str) -> list[np.ndarray]:
    toggle = scene.target_view(key, Role.TOGGLEABLE)
    if toggle is None or toggle.xy is None or toggle.z is None:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"{key} has no registered toggleable control",
        )
    contact = np.array([toggle.xy[0], toggle.xy[1], toggle.z], dtype=np.float64)
    direction = np.asarray(
        (toggle.meta or {}).get("press_dir", (0.0, 0.0, -1.0)),
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"{key} toggle control has no press direction",
        )
    pressed = contact + direction / norm * 0.02
    return [contact, pressed, contact]


def _produce_toggle(
    rig,
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    *,
    turn_on: bool,
) -> list[wpp.Frame]:
    before = tuple(sorted(scene.running))
    keyframes = _toggle_keyframes(scene, goal.obj_key)
    action = "turn_on" if turn_on else "turn_off"
    declare = scene.declare_turn_on if turn_on else scene.declare_turn_off
    frames = _play_contact_traj(
        rig,
        scene,
        st,
        goal,
        kf=keyframes,
        coupling="GAZE",
        lean_z=float(keyframes[0][2]),
        seg=action,
        declare_fn=lambda: declare(goal.obj_key),
    )
    after = tuple(sorted(scene.running))
    if not frames:
        return frames
    pressed = keyframes[1]
    press_index = min(
        range(len(frames)),
        key=lambda index: (
            float(np.linalg.norm(frames[index].contact_anchor - pressed))
            if frames[index].contact_anchor is not None
            else float("inf")
        ),
    )
    return [
        replace(
            frame,
            running_devices=before if index < press_index else after,
        )
        for index, frame in enumerate(frames)
    ]


def produce_turn_on(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Press the grounded control of any registered toggleable device and mark it RUNNING."""
    return _produce_toggle(rig, scene, st, goal, turn_on=True)


def produce_turn_off(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Press the same grounded control and clear the device's RUNNING state."""
    return _produce_toggle(rig, scene, st, goal, turn_on=False)


def _seated_pose(rig, arm, hip: float = -1.5, knee: float = 1.5) -> np.ndarray:
    """A coarse hand-authored seated pose: stand pose + flexed hips & knees (thighs forward, shins down).
    Joint signs were checked against the --dry storyboard."""
    p = core.assemble_pose(rig, arm, (0.0, 0.0, 0.0), 0.0)
    for j in (_HIP_L, _HIP_R):
        p[j * 3] = hip
    for j in (_KNEE_L, _KNEE_R):
        p[j * 3] = knee
    return p


def _seated_arrive_beats(
    rig,
    st: SchedState,
    target_arm: dict,
    obj_state: tuple,
) -> list[wpp.Frame]:
    target = _seated_pose(rig, target_arm)
    previous = st.last_pose156 if st.last_pose156 is not None else target
    root_z = st.seated_z if st.seated_z is not None else st.stand_z
    rhp = st.held.get("rhp") if st.held is not None else None
    frames = []
    for index in range(_N_ARRIVE):
        fraction = (index + 1) / _N_ARRIVE
        frames.append(
            wpp.Frame(
                gait.blend156(previous, target, fraction),
                st.root_xy,
                st.root_yaw,
                root_z,
                rhp=rhp,
                fpv_mode="look",
                obj=obj_state,
                seg="arrive",
            )
        )
    st.walk_end_yaw = st.root_yaw
    st.last_pose156 = target
    return frames


def produce_sit(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """BODY_ANCHORED: lower the body onto a seat (pelvis -> seat top, slide onto it), coarse seated pose;
    declare sitting. Adapts to any seat height/pos via the grounded seat top -- same code per chair."""
    e = scene.target_view(goal.obj_key, Role.SEAT)
    if e is None:
        raise PrimitiveFailure(
            PrimitiveError.MISSING_AFFORDANCE, f"{goal.obj_key} is not a seat"
        )
    (sx, sy), seat_top = scene.surface_top(goal.obj_key)
    held = st.held
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    arm = held["grasp_arm"] if held is not None else arm_rest
    hand_close = 1.0 if held is not None else 0.0
    obj_state = _obj_carry(st) if held is not None else ("hide", None)
    rhp = held.get("rhp") if held is not None else None
    seated = _seated_pose(rig, arm)
    seated_z = max(float(seat_top), _pin_z(seated, st.floor_z))
    frames = _arrive_beats(rig, st, arm, hand_close, obj_state)
    if rhp is not None:
        for frame in frames:
            frame.rhp = rhp
    # The approach stops IN FRONT of the seat FACING it (approach_yaw). To sit, TURN 180 (back to the
    # backrest, face out) while backing the pelvis onto the cushion + lowering -- NOT walking the body
    # through the seat volume (sliding straight to the seat center from behind would penetrate it).
    prev, stand_xy = st.last_pose156, st.root_xy
    start_yaw = st.root_yaw
    authored_sit_yaw = (e.meta or {}).get("sit_yaw")
    sit_yaw = (
        float(authored_sit_yaw)
        if authored_sit_yaw is not None
        else start_yaw + 180.0
    )
    sit_back_offset = float((e.meta or {}).get("sit_back_offset_m") or 0.0)
    sit_angle = math.radians(sit_yaw)
    sx -= math.sin(sit_angle) * sit_back_offset
    sy += math.cos(sit_angle) * sit_back_offset
    for i in range(_N_REACH):
        s = wpp._smoothstep((i + 1) / _N_REACH)
        pose = gait.blend156(prev, seated, s)
        xy = (
            stand_xy[0] + (sx - stand_xy[0]) * s,
            stand_xy[1] + (sy - stand_xy[1]) * s,
        )
        yaw = start_yaw + (sit_yaw - start_yaw) * s
        z = st.stand_z + (seated_z - st.stand_z) * s
        frames.append(
            wpp.Frame(
                pose,
                xy,
                yaw,
                z,
                rhp=rhp,
                fpv_mode="look",
                obj=obj_state,
                seg="sit",
            )
        )
    scene.declare_sit(goal.obj_key)
    st.root_xy = (sx, sy)
    st.root_yaw = sit_yaw
    st.walk_end_yaw = sit_yaw  # see produce_look_at: keep the two headings in step
    st.last_pose156 = seated
    st.seated_z = seated_z
    if held is not None:
        _stamp_obj_path(frames, scene, held["obj_key"])
        _stamp_carried(frames, held.get("carried"))
    print(
        f"[sit] {goal.obj_key}: seat_top={seat_top:.2f} turn {start_yaw:.0f}->{sit_yaw:.0f} "
        f"pelvis {st.stand_z:.2f}->{seated_z:.2f} (seat={seat_top:.2f})"
    )
    return frames


def produce_standup(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Reverse of sit: rise from the current seat back to standing + step forward off it; declare standing."""
    held = st.held
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    arm = held["grasp_arm"] if held is not None else arm_rest
    hand_close = 1.0 if held is not None else 0.0
    obj_state = _obj_carry(st) if held is not None else ("hide", None)
    rhp = held.get("rhp") if held is not None else None
    seated = (
        st.last_pose156 if st.last_pose156 is not None else _seated_pose(rig, arm)
    )
    stand = _manip(rig, arm, hand_close, 0.0)
    seat = scene.sitting_on
    seat_top = scene.surface_top(seat)[1] if seat is not None else st.stand_z
    seated_z = st.seated_z if st.seated_z is not None else seat_top
    a = math.radians(st.root_yaw)
    fwd = (math.sin(a), -math.cos(a))
    (sx, sy) = st.root_xy
    end_xy = (sx + fwd[0] * 0.25, sy + fwd[1] * 0.25)  # step forward off the seat
    frames = []
    for i in range(_N_LIFT):
        s = wpp._smoothstep((i + 1) / _N_LIFT)
        pose = gait.blend156(seated, stand, s)
        xy = (sx + (end_xy[0] - sx) * s, sy + (end_xy[1] - sy) * s)
        z = seated_z + (st.stand_z - seated_z) * s
        frames.append(
            wpp.Frame(
                pose,
                xy,
                st.root_yaw,
                z,
                rhp=rhp,
                fpv_mode="look",
                obj=obj_state,
                seg="standup",
            )
        )
    scene.declare_stand()
    st.seated_z = None
    st.root_xy = end_xy
    st.last_pose156 = stand
    if held is not None:
        _stamp_obj_path(frames, scene, held["obj_key"])
        _stamp_carried(frames, held.get("carried"))
    return frames


# =====================================================================================================
# The GENERAL data-driven CONTACT-ACTION engine (separate from the grab/put/open/close producers).
#
# _play_contact_traj reproduces the _slide 5-stage envelope (arrive -> approach/pre_grasp -> ACT along a
# world keyframe path -> back-out -> retract), reusing the SAME hand-nav (warp_arm_to_seat via
# _reach_arm_leaned), the approach-waypoint clearance gate (plan_reach_waypoints/_solve_and_check), the
# arrive/retract beats, and the joint-space lerp -- but the ACT KEYFRAMES and the OBJECT COUPLING are
# parameters. So a new contact verb (cut/stir/pour/wipe/cook) is a curve + a coupling, not a ~140-line
# clone. Coupling: "JOINT" (obj rides the drawer joint via ("rest",pt) -> episode_runtime._drive_drawers),
# "CARRY" (a held tool rides the wrist via ("carry",otw)), "GAZE" (no object; hand reaches + dwells).
# The WHERE is grounded live (scene.object_pos/surface_top) and warped every call, so the arbitrary-
# position reach is inherited unchanged; completion is guaranteed exactly as _slide guarantees it (a
# reach/precondition miss raises PrimitiveFailure -> zero frames -> check_goal stays False, never faked).
# =====================================================================================================
def _kf_chop(contact, passes: int = 3, lift: float = 0.09, depth: float = 0.0) -> list:
    """Cut/chop: start above the target, then alternate a down-stroke onto it and a lift (world xyz)."""
    c = np.asarray(contact, dtype=np.float64)
    kf = [c + np.array([0.0, 0.0, lift])]
    for _ in range(max(1, passes)):
        kf.append(c + np.array([0.0, 0.0, depth]))
        kf.append(c + np.array([0.0, 0.0, lift]))
    return kf


def _kf_circle(
    center, r: float = 0.06, loops: float = 2.0, n: int = 8, up: float = 0.02
) -> list:
    """Stir/mix: a circle in the horizontal plane at ``center`` (world xyz), ``loops`` turns."""
    c = np.asarray(center, dtype=np.float64)
    total = max(2, int(round(n * loops)))
    return [
        c
        + np.array(
            [
                r * math.cos(2 * math.pi * loops * i / total),
                r * math.sin(2 * math.pi * loops * i / total),
                up,
            ]
        )
        for i in range(total + 1)
    ]


def _kf_raster(
    center, e_side, span: float = 0.20, passes: int = 3, up: float = 0.02
) -> list:
    """Wipe/saw: back-and-forth along ``e_side`` (a horizontal unit) over ``center`` (world xyz)."""
    c = np.asarray(center, dtype=np.float64) + np.array([0.0, 0.0, up])
    e = np.asarray(e_side, dtype=np.float64)
    kf = [c - e * span * 0.5]
    for _ in range(max(1, passes)):
        kf.append(c + e * span * 0.5)
        kf.append(c - e * span * 0.5)
    return kf


def _kf_pour(rim, up: float = 0.16) -> list:
    """Pour: bring the held source over the target rim, tip near it, lift back (world xyz)."""
    c = np.asarray(rim, dtype=np.float64)
    return [
        c + np.array([0.0, 0.0, up]),
        c + np.array([0.0, 0.0, 0.05]),
        c + np.array([0.0, 0.0, up]),
    ]


def _tilted_carry_otw(otw, angle_rad: float):
    """Rotate a carried object about the wrist along one of its own horizontal axes."""
    if abs(float(angle_rad)) <= 1e-12:
        return otw
    base = np.asarray(otw, dtype=np.float64)
    axis = base[:3, 0].astype(np.float64, copy=True)
    axis /= float(np.linalg.norm(axis))
    x, y, z = axis
    skew = np.array(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64
    )
    rotation = np.eye(4, dtype=np.float64)
    rotation[:3, :3] = (
        np.eye(3, dtype=np.float64)
        + math.sin(angle_rad) * skew
        + (1.0 - math.cos(angle_rad)) * (skew @ skew)
    )
    return rotation @ base


def _carry_otw_at_progress(otw, max_tilt_deg: float, action_progress):
    if action_progress is None or abs(float(max_tilt_deg)) <= 1e-12:
        return otw
    progress = min(1.0, max(0.0, float(action_progress)))
    folded = 1.0 - abs(2.0 * progress - 1.0)
    weight = wpp._smoothstep(folded)
    return _tilted_carry_otw(otw, math.radians(float(max_tilt_deg)) * weight)


def _approach_side(stand_xy, contact) -> np.ndarray:
    """Horizontal unit perpendicular to the stand->contact approach (the raster/side axis)."""
    d = np.asarray(contact, dtype=np.float64)[:2] - np.asarray(
        stand_xy, dtype=np.float64
    )
    n = float(np.linalg.norm(d))
    fwd = (
        np.array([d[0] / n, d[1] / n, 0.0]) if n > 1e-6 else np.array([0.0, -1.0, 0.0])
    )
    return np.cross(np.array([0.0, 0.0, 1.0]), fwd)


def _surface_interaction_center(
    scene: SceneState,
    key: str,
    actor_xy,
    *,
    edge_inset: float,
) -> np.ndarray:
    """Choose the nearest usable point inside a surface instead of its arbitrary center."""
    dialogue = (
        (getattr(scene, "behavior_task", None) or {}).get("episode_dialogue") or {}
    )
    surface_visual_effects.register_surface_effect_centers(scene, dialogue)
    (cx, cy), top = scene.surface_top(key)
    target_view = getattr(scene, "target_view", None)
    surface = None
    if callable(target_view):
        surface = target_view(key, Role.SURFACE) or target_view(key, Role.CONTAINER)
    if surface is None or surface.xy is None:
        return np.array([cx, cy, top], dtype=np.float64)
    effect_centers = surface.meta.get("surface_effect_centers") or {}
    if effect_centers:
        nearest = min(
            effect_centers.values(),
            key=lambda point: math.dist(
                (float(actor_xy[0]), float(actor_xy[1])),
                (float(point[0]), float(point[1])),
            ),
        )
        return np.array(
            [float(nearest[0]), float(nearest[1]), float(top)], dtype=np.float64
        )
    hx, hy = (
        max(0.0, float(value))
        for value in (getattr(surface, "half", None) or (0.0, 0.0))
    )
    usable_x = max(0.0, hx - max(0.0, float(edge_inset)))
    usable_y = max(0.0, hy - max(0.0, float(edge_inset)))
    ax, ay = float(actor_xy[0]), float(actor_xy[1])
    return np.array(
        [
            min(max(ax, float(cx) - usable_x), float(cx) + usable_x),
            min(max(ay, float(cy) - usable_y), float(cy) + usable_y),
            float(top),
        ],
        dtype=np.float64,
    )


def computer_contact_trajectory(
    scene: SceneState, key: str, root_xy
) -> list[np.ndarray]:
    """Return the world-space keystrokes used by standing computer interaction."""
    minimum, maximum = scene.object_aabb(key)
    center = 0.5 * (
        np.asarray(minimum, dtype=np.float64) + np.asarray(maximum, dtype=np.float64)
    )
    center[2] = float(maximum[2]) + 0.008
    side = _approach_side(root_xy, center)
    strokes = (-0.012, -0.004, 0.007, 0.013, 0.003, -0.009, 0.010, -0.002)
    return [
        center
        + side * offset
        + np.array([0.0, 0.0, 0.009 if i % 2 == 0 else 0.002])
        for i, offset in enumerate(strokes)
    ]


def standing_computer_reach_tolerance() -> float:
    """The reach gate shared by standing typing execution and stand selection."""
    return 3.0 * _REACH_TOL_M


def _computer_contact_plan(
    scene: SceneState, st: SchedState, key: str
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return compact keystrokes and a stable gaze on the registered monitor."""
    minimum, maximum = scene.object_aabb(key)
    center = 0.5 * (
        np.asarray(minimum, dtype=np.float64) + np.asarray(maximum, dtype=np.float64)
    )
    center[2] = float(maximum[2]) + 0.008
    keyframes = computer_contact_trajectory(scene, key, st.root_xy)
    meta = scene.resolve(key).meta or {}
    monitor_key = str(meta.get("workstation_monitor") or "")
    if monitor_key:
        monitor = scene.resolve(monitor_key)
        screen_center = (monitor.meta or {}).get("screen_center")
        if screen_center is not None:
            gaze = np.asarray(screen_center, dtype=np.float64)
        else:
            mon_min, mon_max = scene.object_aabb(monitor_key)
            gaze = 0.5 * (
                np.asarray(mon_min, dtype=np.float64)
                + np.asarray(mon_max, dtype=np.float64)
            )
        if bool(meta.get("standing_workstation")):
            gaze = 0.60 * gaze + 0.40 * center
    else:
        toward_keyboard = center[:2] - np.asarray(st.root_xy, dtype=np.float64)
        distance = float(np.linalg.norm(toward_keyboard))
        forward = np.zeros(3, dtype=np.float64)
        if distance > 1e-6:
            forward[:2] = toward_keyboard / distance
        gaze = center + forward * 0.25 + np.array([0.0, 0.0, 0.28])
    return keyframes, gaze


_TYPING_HAND_CLOSE = 0.08
_TYPING_HIP_RAD = -1.70
_TYPING_KNEE_RAD = 1.80
_TYPING_LEAN_RAD = math.radians(10.0)
_TYPING_HALF_SPAN_M = 0.055
_TYPING_PRESS_CLEARANCE_M = 0.008
_TYPING_HOVER_RISE_M = 0.018
_TYPING_SCREEN_DWELL_FRAMES = 8
_LEFT_INDEX_TIP_J = 24


def _typing_wrist_frame() -> np.ndarray:
    """Body-local wrist frame with fingers forward and the palm facing down."""
    return np.array(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        dtype=np.float64,
    )


def _typing_pose(rig, arms: dict, seated_legs: np.ndarray) -> np.ndarray:
    pose = core.assemble_pose(
        rig,
        arms,
        (0.0, 0.0, 0.0),
        _TYPING_HAND_CLOSE,
        left_hand_close=_TYPING_HAND_CLOSE,
    )
    for joint in (_HIP_L, _HIP_R, _KNEE_L, _KNEE_R):
        pose[joint * 3 : joint * 3 + 3] = seated_legs[
            joint * 3 : joint * 3 + 3
        ]
    return wpp._add_lean(pose, _TYPING_LEAN_RAD)


def _typing_fingertip_offset(rig, base_pose: np.ndarray) -> np.ndarray:
    """Body-local wrist-to-index-tip offset for the palm-down typing hand."""
    pose = np.asarray(base_pose, dtype=np.float32).copy()
    wrist, _orientation_error, _bend = core.solve_wrist_orient(
        rig, pose, _typing_wrist_frame()
    )
    pose[core.WRIST_J * 3 : core.WRIST_J * 3 + 3] = wrist
    joints = core.H.fwd_joints(pose)
    return np.asarray(
        joints[core.H.TIPISH["index"]] - joints[core.WRIST_J],
        dtype=np.float64,
    )


def _solve_typing_right_arm(
    rig,
    base_pose: np.ndarray,
    fingertip_target: np.ndarray,
    fingertip_offset: np.ndarray,
    stand_xy,
    yaw: float,
    root_z: float,
) -> tuple[dict, float, float]:
    """Place the index fingertip at a key while keeping the palm parallel to the keyboard."""
    rz, world_off = wpp._grasp_frame(stand_xy, yaw, root_z)
    wrist_target = np.asarray(fingertip_target, dtype=np.float64) - rz @ np.asarray(
        fingertip_offset, dtype=np.float64
    )
    pose, _wrist_error = core.warp_arm_to_wrist(
        rig, base_pose, rz, world_off, wrist_target
    )
    wrist, orientation_error, _bend = core.solve_wrist_orient(
        rig, pose, _typing_wrist_frame()
    )
    pose[core.WRIST_J * 3 : core.WRIST_J * 3 + 3] = wrist
    joints = core.H.fwd_joints(pose)
    fingertip_world = rz @ joints[core.H.TIPISH["index"]] + world_off
    error = float(
        np.linalg.norm(fingertip_world - np.asarray(fingertip_target, dtype=np.float64))
    )
    return wpp._arm_from_pose(pose), error, float(orientation_error)


def _solve_typing_left_arm(
    rig,
    base_pose: np.ndarray,
    fingertip_target: np.ndarray,
    fingertip_offset: np.ndarray,
    arm_rest: dict,
    typing_legs: np.ndarray,
    stand_xy,
    yaw: float,
    root_z: float,
) -> tuple[dict, float, float]:
    corrected_target = np.asarray(fingertip_target, dtype=np.float64)
    orientation_error = float("inf")
    left_arm = {}
    error = float("inf")
    for _ in range(3):
        mirrored_target = _mirror_world_point_for_left_hand(
            corrected_target, stand_xy, yaw, root_z
        )
        right_arm, _mirrored_error, orientation_error = _solve_typing_right_arm(
            rig,
            base_pose,
            mirrored_target,
            fingertip_offset,
            stand_xy,
            yaw,
            root_z,
        )
        left_arm = core.mirror_right_arm_as_left(right_arm)
        probe = _typing_pose(
            rig,
            _merge_left_action_arm(arm_rest, left_arm),
            typing_legs,
        )
        joints = gait.pose_place_joints(probe, stand_xy, yaw, root_z)
        actual = joints[_LEFT_INDEX_TIP_J]
        residual = np.asarray(fingertip_target, dtype=np.float64) - actual
        error = float(np.linalg.norm(residual))
        if error < 0.002:
            break
        corrected_target = corrected_target + residual
    return left_arm, error, orientation_error


def _typing_targets(scene, st, key, keyframes):
    side = _approach_side(st.root_xy, keyframes[0])
    _minimum, maximum = scene.object_aabb(key)
    key_center = np.mean(np.asarray(keyframes, dtype=np.float64), axis=0)
    key_center[2] = float(maximum[2]) + _TYPING_PRESS_CLEARANCE_M
    right_home = key_center - side * _TYPING_HALF_SPAN_M
    left_home = key_center + side * _TYPING_HALF_SPAN_M
    right_home[2] += _TYPING_HOVER_RISE_M
    left_home[2] += _TYPING_HOVER_RISE_M
    press_targets = []
    for point in keyframes:
        target = np.asarray(point, dtype=np.float64).copy()
        target[2] = float(maximum[2]) + _TYPING_PRESS_CLEARANCE_M
        press_targets.append(target)
    return side, right_home, left_home, press_targets


def _typing_arm_plan(
    rig,
    scene,
    st,
    key,
    keyframes,
    typing_base,
    typing_legs,
    arm_rest,
    typing_z,
):
    side, right_home, left_home, press_targets = _typing_targets(
        scene, st, key, keyframes
    )
    fingertip_offset = _typing_fingertip_offset(rig, typing_base)
    right_home_arm, right_error, right_orientation = _solve_typing_right_arm(
        rig,
        typing_base,
        right_home,
        fingertip_offset,
        st.root_xy,
        st.root_yaw,
        typing_z,
    )
    left_home_arm, left_error, left_orientation = _solve_typing_left_arm(
        rig,
        typing_base,
        left_home,
        fingertip_offset,
        arm_rest,
        typing_legs,
        st.root_xy,
        st.root_yaw,
        typing_z,
    )
    home_arms = _merge_left_action_arm(right_home_arm, left_home_arm)
    press_arms: list[dict] = []
    active_targets: list[np.ndarray] = []
    errors = [right_error, left_error]
    orientation_errors = [right_orientation, left_orientation]
    for index, target in enumerate(press_targets):
        active_target = target + side * (
            _TYPING_HALF_SPAN_M if index % 2 else -_TYPING_HALF_SPAN_M
        )
        if index % 2:
            active_arm, error, orientation = _solve_typing_left_arm(
                rig,
                typing_base,
                active_target,
                fingertip_offset,
                arm_rest,
                typing_legs,
                st.root_xy,
                st.root_yaw,
                typing_z,
            )
            arms = _merge_left_action_arm(right_home_arm, active_arm)
        else:
            active_arm, error, orientation = _solve_typing_right_arm(
                rig,
                typing_base,
                active_target,
                fingertip_offset,
                st.root_xy,
                st.root_yaw,
                typing_z,
            )
            arms = _merge_left_action_arm(active_arm, left_home_arm)
        press_arms.append(arms)
        active_targets.append(active_target)
        errors.append(error)
        orientation_errors.append(orientation)
    return home_arms, press_arms, active_targets, max(errors), max(orientation_errors)


def _seated_computer_frames(
    rig,
    scene: SceneState,
    st: SchedState,
    key: str,
    gaze_target: np.ndarray,
) -> list[wpp.Frame]:
    """Animate alternating palm-down fingertip presses from a forward seated posture."""
    keyframes, _gaze = _computer_contact_plan(scene, st, key)
    seated_z = (
        st.seated_z
        if st.seated_z is not None
        else scene.surface_top(str(scene.sitting_on))[1]
    )
    seated = (
        np.asarray(st.last_pose156, dtype=np.float32).copy()
        if st.last_pose156 is not None
        else _seated_pose(rig, getattr(rig, "arm_rest", rig.arm_grasp))
    )
    arm_rest = wpp._arm_from_pose(seated)
    typing_legs = _seated_pose(
        rig, arm_rest, hip=_TYPING_HIP_RAD, knee=_TYPING_KNEE_RAD
    )
    typing_base = _typing_pose(rig, arm_rest, typing_legs)
    chair_top = scene.surface_top(str(scene.sitting_on))[1]
    typing_z = max(float(chair_top), _pin_z(typing_base, st.floor_z))
    home_arms, press_arms, active_targets, max_error, max_orientation_error = (
        _typing_arm_plan(
            rig,
            scene,
            st,
            key,
            keyframes,
            typing_base,
            typing_legs,
            arm_rest,
            typing_z,
        )
    )
    if max_error > 4 * _REACH_TOL_M:
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"use_computer: fingertip reach {max_error * 1000:.0f}mm",
        )
    if max_orientation_error > 0.05:
        raise PrimitiveFailure(
            PrimitiveError.GRASP_FAILED,
            f"use_computer: palm orientation error {max_orientation_error:.3f}",
        )

    home_pose = _typing_pose(rig, home_arms, typing_legs)
    typing_poses = [_typing_pose(rig, arms, typing_legs) for arms in press_arms]
    frames: list[wpp.Frame] = []

    def add(pose: np.ndarray, root_z: float, contact=None) -> None:
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                st.root_yaw,
                root_z,
                fpv_mode="look",
                obj=("hide", None),
                seg="use_computer",
                contact_anchor=contact,
                gaze=np.asarray(gaze_target, dtype=np.float64),
                gaze_locked=True,
            )
        )

    for i in range(_N_REACH):
        s = wpp._smoothstep((i + 1) / _N_REACH)
        add(
            gait.blend156(seated, home_pose, s),
            seated_z + (typing_z - seated_z) * s,
        )
    for _ in range(_TYPING_SCREEN_DWELL_FRAMES):
        add(home_pose, typing_z)
    for pose, target in zip(typing_poses, active_targets):
        for frame_index in range(3):
            s = wpp._smoothstep((frame_index + 1) / 3.0)
            add(gait.blend156(home_pose, pose, s), typing_z)
        add(pose, typing_z, target)
        for frame_index in range(2):
            s = wpp._smoothstep((frame_index + 1) / 2.0)
            add(gait.blend156(pose, home_pose, s), typing_z)
    for _ in range(6):
        add(home_pose, typing_z)
    for i in range(_N_RETRACT):
        s = wpp._smoothstep((i + 1) / _N_RETRACT)
        add(
            gait.blend156(home_pose, seated, s),
            typing_z + (seated_z - typing_z) * s,
        )
    st.last_pose156 = seated
    return frames


def _orient_workstation_seat(
    scene: SceneState, seat_key: str, gaze_target: np.ndarray
) -> None:
    seat = scene.resolve(seat_key)
    seat_xy = np.asarray(scene.surface_top(seat_key)[0], dtype=np.float64)
    toward_screen = np.asarray(gaze_target[:2], dtype=np.float64) - seat_xy
    if float(np.linalg.norm(toward_screen)) <= 1e-6:
        return
    sit_yaw = gait.yaw_for_dir(
        float(toward_screen[0]), float(toward_screen[1])
    )
    seat.meta["sit_yaw"] = sit_yaw
    logger.info(
        "workstation seat %s at (%.3f, %.3f) faces screen (%.3f, %.3f, %.3f) at yaw %.1f",
        seat_key,
        seat_xy[0],
        seat_xy[1],
        gaze_target[0],
        gaze_target[1],
        gaze_target[2],
        sit_yaw,
    )


def contact_trajectory_for_goal(
    scene: SceneState, st: SchedState, goal: Goal
) -> list[np.ndarray] | None:
    """Reconstruct the world-space contact path used by a contact primitive."""
    if goal.prim in (Primitive.TURN_ON, Primitive.TURN_OFF):
        try:
            return _toggle_keyframes(scene, goal.obj_key)
        except PrimitiveFailure:
            return None
    if goal.prim is Primitive.CUT:
        return _kf_chop(scene.object_pos(goal.obj_key), passes=3)
    if goal.prim in (Primitive.STIR, Primitive.POUR):
        (sx, sy), top = scene.surface_top(goal.obj_key)
        center = np.array([sx, sy, top], dtype=np.float64)
        return (
            _kf_circle(center, r=0.06, loops=2.0)
            if goal.prim is Primitive.STIR
            else _kf_pour(center)
        )
    if goal.prim is Primitive.WIPE:
        span = 0.22
        center = _surface_interaction_center(
            scene,
            goal.obj_key,
            st.root_xy,
            edge_inset=0.5 * span + 0.03,
        )
        return _kf_raster(
            center,
            _approach_side(st.root_xy, center),
            span=span,
            passes=3,
        )
    if goal.prim is Primitive.COOK:
        appliance = _cook_appliance(scene, goal)
        if appliance is not None and _cook_is_running_closed_oven(
            scene, goal, appliance
        ):
            return None
        (sx, sy), top = scene.surface_top(goal.obj_key)
        return [np.array([sx, sy, top + 0.04], dtype=np.float64)]
    if goal.prim is Primitive.USE_COMPUTER:
        return _computer_contact_plan(scene, st, goal.obj_key)[0]
    return None


def _contact_pose_plan(rig, scene: SceneState, st: SchedState, contact, lean_z: float):
    """Choose the body pose used by every keyframe in one contact action."""
    if not getattr(scene, "unified_geom", False):
        lean_rad, lf = _lean_for_height(float(lean_z))
        return {
            "lean_rad": lean_rad,
            "lf": lf,
            "squat": 0.0,
            "stand_z": float(st.stand_z),
            "err": None,
            "reachable": True,
        }
    planner_tol = _reach_tolerance(scene)
    return plan_reach(
        rig,
        rig.arm_grasp,
        contact,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        planner_tol,
        floor_z=st.floor_z,
        enabled=True,
        clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
        allow_colliding_fallback=False,
    )


def _unique_contact_points(points) -> tuple[list[np.ndarray], list[int]]:
    """Return stable unique contact points and an index back into that compact list."""
    unique: list[np.ndarray] = []
    indices: list[int] = []
    seen: dict[tuple[float, float, float], int] = {}
    for point in points:
        value = np.asarray(point, dtype=np.float64)
        key = tuple(round(float(axis), 6) for axis in value)
        index = seen.get(key)
        if index is None:
            index = len(unique)
            seen[key] = index
            unique.append(value)
        indices.append(index)
    return unique, indices


def _contact_pose_plans(
    rig,
    scene: SceneState,
    st: SchedState,
    contacts,
    lean_z: float,
) -> list[dict]:
    """Plan every distinct contact point while keeping one fixed standing position."""
    points, indices = _unique_contact_points(contacts)
    if len(points) == 1 or not getattr(scene, "unified_geom", False):
        plan = _contact_pose_plan(rig, scene, st, points[0], lean_z)
        return [plan for _ in indices]
    planner_tol = _reach_tolerance(scene)
    plans = plan_reach_many(
        rig,
        rig.arm_grasp,
        points,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        planner_tol,
        floor_z=st.floor_z,
        enabled=True,
        clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
        allow_colliding_fallback=False,
    )
    return [plans[index] for index in indices]


def _record_failed_contact_trajectory(st: SchedState, goal: Goal, contacts) -> None:
    points, _indices = _unique_contact_points(contacts)
    values = [tuple(float(axis) for axis in point) for point in points]
    goal.meta["_failed_contacts"] = values
    st.failed_contact_trajectory = (
        goal.prim.value,
        goal.obj_key,
        tuple(values),
    )
    st.failed_contact_target_key = str(
        (goal.meta or {}).get("_contact_target_key") or goal.obj_key
    )


def _contact_arm_from_plan(rig, target, stand_xy, yaw, reach_plan):
    """Return the exact arm pose already accepted by the shared reach planner."""
    planned_pose = reach_plan.get("arm_pose")
    planned_err = reach_plan.get("err")
    if planned_pose is not None and planned_err is not None:
        return wpp._arm_from_pose(planned_pose), float(planned_err)
    return _solve_reach_arm(
        rig,
        rig.arm_grasp,
        target,
        stand_xy,
        yaw,
        float(reach_plan["stand_z"]),
        lean_override=float(reach_plan["lean_rad"]),
        squat_frac=float(reach_plan["squat"]),
    )


def _play_contact_traj(
    rig,
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    *,
    kf: list,
    coupling: str,
    lean_z: float,
    declare_fn=None,
    otw=None,
    seg: str = "act",
    gaze_on: bool = True,
    dwell: int = 0,
    contact_hand_close: float = 1.0,
    gaze_target=None,
    carry_tilt_deg: float = 0.0,
) -> list[wpp.Frame]:
    """The general contact-action player (see the section header). ``kf`` = >=1 world contact points."""
    stand_xy, yaw, stand_z = st.root_xy, st.root_yaw, st.stand_z
    kf = [np.asarray(p, dtype=np.float64) for p in kf]

    _record_failed_contact_trajectory(st, goal, kf)
    reach_plans = _contact_pose_plans(rig, scene, st, kf, lean_z)
    if any(plan.get("collision_blocked") for plan in reach_plans):
        raise PrimitiveFailure(
            PrimitiveError.TRAJECTORY_COLLISION,
            f"{seg} has no collision-free reach pose",
        )
    if any(plan.get("err") is not None for plan in reach_plans):
        worst_plan = max(
            reach_plans,
            key=lambda plan: float(plan.get("err") or 0.0),
        )
        print(
            f"[{seg}] trajectory-plan points={len(kf)} "
            f"worst_lean={math.degrees(float(worst_plan['lean_rad'])):.0f}deg "
            f"worst_squat={float(worst_plan['squat']):.1f} "
            f"reach_err={float(worst_plan['err']) * 1000:.0f}mm "
            f"reachable={all(bool(plan['reachable']) for plan in reach_plans)}"
        )

    arm_kf, max_err = [], 0.0
    for pt, reach_plan in zip(kf, reach_plans):
        arm, err = _contact_arm_from_plan(rig, pt, stand_xy, yaw, reach_plan)
        arm_kf.append(arm)
        max_err = max(max_err, err)
    reach_tolerance = (
        standing_computer_reach_tolerance()
        if goal.prim is Primitive.USE_COMPUTER
        else 3.0 * _REACH_TOL_M
    )
    if max_err > reach_tolerance:
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH, f"{seg}: reach {max_err * 1000:.0f}mm"
        )

    boxes = _reach_boxes(scene)
    contact_key = str((goal.meta or {}).get("_contact_target_key") or goal.obj_key)
    wp = plan_reach_waypoints(scene, contact_key, stand_xy, kf[0])
    approach = wp["approach"]
    arm_pre, _, _, _ = _solve_and_check(
        rig,
        rig.arm_grasp,
        wp["pre"],
        stand_xy,
        yaw,
        stand_z,
        boxes,
        f"{seg} pre",
        strict=_body_collision_is_hard(scene),
    )
    back_pt = kf[-1] - approach * _WP_POST_BACK + np.array([0.0, 0.0, 0.04])
    arm_back, _, _, _ = _solve_and_check(
        rig,
        arm_kf[-1],
        back_pt,
        stand_xy,
        yaw,
        stand_z,
        boxes,
        f"{seg} retreat",
        strict=_body_collision_is_hard(scene),
    )
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    hold = coupling == "CARRY"  # a held tool stays gripped through the whole action

    def _objstate(pt, action_progress=None):
        if coupling == "CARRY":
            return (
                "carry",
                _carry_otw_at_progress(otw, carry_tilt_deg, action_progress),
            )
        if coupling == "JOINT":
            return ("rest", tuple(np.asarray(pt, dtype=np.float64)))
        return ("hide", None)

    contact_hand_close = max(0.0, min(1.0, float(contact_hand_close)))

    def _gaze(pt):
        return (
            np.asarray(gaze_target if gaze_target is not None else pt, dtype=np.float64)
            if (gaze_on and coupling != "JOINT")
            else None
        )

    frames = _arrive_beats(rig, st, arm_rest, 1.0 if hold else 0.0, _objstate(kf[0]))

    uses_squat = any(float(plan["squat"]) > 1e-6 for plan in reach_plans)

    def add(pose, pt, seg_name, contact=None, action_progress=None):
        frame_z = _pin_z(pose, st.floor_z) if uses_squat else stand_z
        frames.append(
            wpp.Frame(
                pose,
                stand_xy,
                yaw,
                frame_z,
                fpv_mode="look",
                obj=_objstate(pt, action_progress),
                seg=seg_name,
                contact_anchor=contact,
                gaze=_gaze(pt),
            )
        )

    # Phase B: approach -- rest -> pre_grasp, then pre_grasp -> first contact (hand closing + lean in).
    n_pre = max(1, int(round(_N_REACH * 0.45)))
    for i in range(n_pre):
        s = wpp._smoothstep((i + 1) / n_pre)
        add(
            _manip(rig, core.lerp_arm(arm_rest, arm_pre, s), 1.0 if hold else 0.0, 0.0),
            kf[0],
            seg,
        )
    first_plan = reach_plans[0]
    first_lf = float(first_plan["lf"])
    first_squat = float(first_plan["squat"])
    for i in range(_N_REACH):
        s = wpp._smoothstep((i + 1) / _N_REACH)
        add(
            _manip(
                rig,
                core.lerp_arm(arm_pre, arm_kf[0], s),
                1.0 if hold else contact_hand_close * s,
                first_lf * s,
                first_squat * s,
            ),
            kf[0],
            seg,
            contact=kf[0],
        )

    # Phase C: ACT along the keyframe path (hand closed, leaning), the object coupled to the effector.
    act_seg = "slide" if coupling == "JOINT" else seg
    if len(kf) == 1:
        act_frames = max(1, dwell)
        for i in range(act_frames):
            add(
                _manip(
                    rig,
                    arm_kf[0],
                    1.0 if hold else contact_hand_close,
                    first_lf,
                    first_squat,
                ),
                kf[0],
                act_seg,
                contact=kf[0],
                action_progress=i / max(1, act_frames - 1),
            )
    else:
        n_seg = max(2, int(round(_N_PLACE / (len(kf) - 1))))
        act_frames = n_seg * (len(kf) - 1)
        for a in range(len(kf) - 1):
            from_plan = reach_plans[a]
            to_plan = reach_plans[a + 1]
            for i in range(n_seg):
                s = wpp._smoothstep((i + 1) / n_seg)
                pt = kf[a] + (kf[a + 1] - kf[a]) * s
                lf = (
                    float(from_plan["lf"])
                    + (float(to_plan["lf"]) - float(from_plan["lf"])) * s
                )
                squat = (
                    float(from_plan["squat"])
                    + (float(to_plan["squat"]) - float(from_plan["squat"])) * s
                )
                add(
                    _manip(
                        rig,
                        core.lerp_arm(arm_kf[a], arm_kf[a + 1], s),
                        1.0 if hold else contact_hand_close,
                        lf,
                        squat,
                    ),
                    pt,
                    act_seg,
                    contact=pt,
                    action_progress=(a * n_seg + i) / max(1, act_frames - 1),
                )

    # Phase D: back-out + retract. CARRY keeps the tool gripped; JOINT releases the handle.
    end_pt = kf[-1]
    last_plan = reach_plans[-1]
    last_lf = float(last_plan["lf"])
    last_squat = float(last_plan["squat"])
    n_back = max(1, int(round(_N_RETRACT * 0.5)))
    for i in range(n_back):
        s = wpp._smoothstep((i + 1) / n_back)
        add(
            _manip(
                rig,
                core.lerp_arm(arm_kf[-1], arm_back, s),
                1.0 if hold else contact_hand_close * (1.0 - s),
                last_lf,
                last_squat,
            ),
            end_pt,
            "retract",
            contact=None if hold else end_pt,
        )
    for i in range(_N_RETRACT):
        s = wpp._smoothstep((i + 1) / _N_RETRACT)
        add(
            _manip(
                rig,
                core.lerp_arm(arm_back, arm_rest, s),
                1.0 if hold else 0.0,
                last_lf * (1.0 - s),
                last_squat * (1.0 - s),
            ),
            end_pt,
            "retract",
        )

    if hold and st.held is not None:
        _stamp_obj_path(frames, scene, st.held["obj_key"])
        _stamp_carried(frames, st.held.get("carried"))
    _require_fpv_eye_clear(scene, frames, goal, seg)
    if declare_fn is not None:
        declare_fn()
    goal.meta.pop("_failed_contacts", None)
    st.failed_contact_trajectory = None
    st.failed_contact_target_key = None
    st.obj_rest_xyz = None if hold else tuple(end_pt)
    st.last_pose156 = _manip(rig, arm_rest, 1.0 if hold else 0.0, 0.0)
    return frames


def _require_tool(st: SchedState, goal: Goal, verb: str):
    """A CONTACT tool verb precondition: the tool named in goal.meta['tool'] must be currently held."""
    tool = goal.meta.get("tool")
    if st.held is None or (tool is not None and st.held.get("obj_key") != tool):
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"{verb} needs tool {tool!r} held (holding {None if st.held is None else st.held.get('obj_key')!r})",
        )
    rig_rhp = st.held.get("rhp")
    return st.held["otw"], rig_rhp


def produce_cut(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """CONTACT cut: chop the held tool over the target with a raster of down-strokes; declare CUT."""
    otw, rhp = _require_tool(st, goal, "cut")
    if rhp is not None:
        rig.rhp = rhp
    contact = scene.object_pos(goal.obj_key)
    kf = _kf_chop(contact, passes=3)
    return _play_contact_traj(
        rig,
        scene,
        st,
        goal,
        kf=kf,
        coupling="CARRY",
        lean_z=float(contact[2]),
        otw=otw,
        seg="cut",
        declare_fn=lambda: scene.declare_cut(goal.obj_key),
    )


def produce_stir(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """CONTACT stir: circle the held tool in the target container (rendered motion; no state flip)."""
    otw, rhp = _require_tool(st, goal, "stir")
    if rhp is not None:
        rig.rhp = rhp
    (sx, sy), top = scene.surface_top(goal.obj_key)
    kf = _kf_circle(np.array([sx, sy, top]), r=0.06, loops=2.0)
    return _play_contact_traj(
        rig,
        scene,
        st,
        goal,
        kf=kf,
        coupling="CARRY",
        lean_z=float(top),
        otw=otw,
        seg="stir",
    )


def produce_pour(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """CONTACT pour: bring the held source over the target rim; declare INSIDE(source, target)."""
    otw, rhp = _require_tool(st, goal, "pour")
    if rhp is not None:
        rig.rhp = rhp
    src = st.held.get("obj_key")
    (sx, sy), top = scene.surface_top(goal.obj_key)
    kf = _kf_pour(np.array([sx, sy, top]))
    return _play_contact_traj(
        rig,
        scene,
        st,
        goal,
        kf=kf,
        coupling="CARRY",
        lean_z=float(top),
        otw=otw,
        seg="pour",
        carry_tilt_deg=60.0,
        declare_fn=lambda: scene.declare_inside(src, goal.obj_key),
    )


def produce_wipe(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Pick up the named tool when needed, then visibly raster it across the target."""
    pickup_frames = []
    tool = (goal.meta or {}).get("tool")
    if st.held is None and tool is not None:
        pickup_frames = produce_pick(
            rig,
            g,
            scene,
            st,
            Goal(
                Primitive.PICK,
                obj_key=str(tool),
                meta={"_implicit_action": "wipe_tool_pick"},
            ),
            sd_world=None,
        )
    otw, rhp = _require_tool(st, goal, "wipe")
    if rhp is not None:
        rig.rhp = rhp
    span = 0.22
    center = _surface_interaction_center(
        scene,
        goal.obj_key,
        st.root_xy,
        edge_inset=0.5 * span + 0.03,
    )
    kf = _kf_raster(
        center,
        _approach_side(st.root_xy, center),
        span=span,
        passes=3,
    )
    wipe_frames = _play_contact_traj(
        rig,
        scene,
        st,
        goal,
        kf=kf,
        coupling="CARRY",
        lean_z=float(center[2]),
        otw=otw,
        seg="wipe",
    )
    return [*pickup_frames, *wipe_frames]


def _cook_appliance_candidates(scene: SceneState, goal: Goal) -> list[str]:
    """Return legacy cook targets followed by their declared parent chains."""
    inside = getattr(scene, "inside", {}) or {}
    on = getattr(scene, "on", {}) or {}
    meta = goal.meta or {}
    legacy = (
        meta.get("appliance"),
        meta.get("container"),
        inside.get(goal.obj_key),
        on.get(goal.obj_key),
        goal.obj_key,
    )
    candidates: list[str] = []
    seen: set[str] = set()
    for raw_key in legacy:
        if raw_key is None:
            continue
        key = str(raw_key)
        if key and key not in seen:
            candidates.append(key)
            seen.add(key)

    cursor = 0
    while cursor < len(candidates):
        key = candidates[cursor]
        cursor += 1
        for raw_parent in (inside.get(key), on.get(key)):
            if raw_parent is None:
                continue
            parent = str(raw_parent)
            if parent and parent not in seen:
                candidates.append(parent)
                seen.add(parent)
    return candidates


def _nearby_running_cook_appliance(scene: SceneState, goal: Goal) -> str | None:
    """Recover a missing support edge when a pan visibly rests on a running hob."""
    inside = getattr(scene, "inside", {}) or {}
    on = getattr(scene, "on", {}) or {}
    parent_key = inside.get(goal.obj_key) or on.get(goal.obj_key)
    if parent_key is None or str(parent_key) not in scene.entities:
        return None
    parent = scene.resolve(str(parent_key))
    parent_xy = tuple(float(value) for value in parent.xy)
    nearby: list[tuple[float, str]] = []
    for raw_key in getattr(scene, "running", set()) or set():
        key = str(raw_key)
        if key not in scene.entities:
            continue
        if scene.affordance(key, Role.TOGGLEABLE) is None:
            continue
        appliance = scene.resolve(key)
        distance = math.hypot(
            float(appliance.xy[0]) - parent_xy[0],
            float(appliance.xy[1]) - parent_xy[1],
        )
        if distance <= 0.55:
            nearby.append((distance, key))
    return min(nearby)[1] if nearby else None


def _cook_appliance(scene: SceneState, goal: Goal) -> str | None:
    declared = next(
        (
            str(key)
            for key in _cook_appliance_candidates(scene, goal)
            if key in scene.entities
            and scene.affordance(str(key), Role.TOGGLEABLE) is not None
        ),
        None,
    )
    return declared or _nearby_running_cook_appliance(scene, goal)


def _placement_ancestor(scene: SceneState, child_key: str, ancestor_key: str) -> bool:
    parents = dict(getattr(scene, "on", {}) or {})
    parents.update(dict(getattr(scene, "inside", {}) or {}))
    cursor = str(child_key)
    seen: set[str] = set()
    while cursor not in seen:
        seen.add(cursor)
        parent = parents.get(cursor)
        if parent is None:
            return False
        cursor = str(parent)
        if cursor == ancestor_key:
            return True
    return False


def _cook_is_running_closed_oven(
    scene: SceneState,
    goal: Goal,
    appliance_key: str,
) -> bool:
    appliance = scene.resolve(appliance_key)
    meta = getattr(appliance, "meta", None) or {}
    category = str(meta.get("asset_category") or "").casefold().replace("-", "_")
    return bool(
        meta.get("access_mode") == "front_opening"
        and "oven" in category
        and appliance_key in (getattr(scene, "running", set()) or set())
        and appliance_key not in (getattr(scene, "open_objs", set()) or set())
        and _placement_ancestor(scene, goal.obj_key, appliance_key)
    )


def _observe_cooking_appliance(
    rig,
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    appliance_key: str,
) -> list[wpp.Frame]:
    gaze = np.asarray(_toggle_keyframes(scene, appliance_key)[0], dtype=np.float64)
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    frames = _arrive_beats(rig, st, arm_rest, 0.0, _obj_carry(st))
    frames = [replace(frame, seg="cook", gaze=gaze) for frame in frames]
    base = st.last_pose156
    root_z = _pin_z(base, st.floor_z)
    frames.extend(
        wpp.Frame(
            base,
            st.root_xy,
            st.root_yaw,
            root_z,
            fpv_mode="look",
            obj=_obj_carry(st),
            seg="cook",
            gaze=gaze,
        )
        for _ in range(_N_PLACE)
    )
    scene.declare_cook(goal.obj_key)
    return frames


def produce_cook(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Cook food at its active appliance, without re-pressing a closed running oven."""
    appliance = _cook_appliance(scene, goal)
    if appliance is None:
        raise PrimitiveFailure(
            PrimitiveError.MISSING_AFFORDANCE,
            f"cook target {goal.obj_key!r} is not in or on a toggleable appliance",
        )
    goal.meta["_contact_target_key"] = appliance
    if _cook_is_running_closed_oven(scene, goal, appliance):
        return _observe_cooking_appliance(rig, scene, st, goal, appliance)
    kf = _toggle_keyframes(scene, appliance)
    return _play_contact_traj(
        rig,
        scene,
        st,
        goal,
        kf=kf,
        coupling="GAZE",
        lean_z=float(kf[0][2]),
        seg="cook",
        dwell=_N_PLACE,
        declare_fn=lambda: scene.declare_cook(goal.obj_key),
    )


def produce_use_computer(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Sit at a registered workstation, type with both hands, and watch its live monitor."""
    keyframes, gaze_target = _computer_contact_plan(scene, st, goal.obj_key)
    meta = scene.resolve(goal.obj_key).meta or {}
    seat_key = str(meta.get("workstation_seat") or "")
    monitor_key = str(meta.get("workstation_monitor") or "")
    if seat_key:
        _orient_workstation_seat(scene, seat_key, gaze_target)
        if scene.sitting_on not in (None, seat_key):
            raise PrimitiveFailure(
                PrimitiveError.PRECONDITION_UNMET,
                f"use_computer needs seat {seat_key!r}, already sitting on {scene.sitting_on!r}",
            )
        before = tuple(sorted(scene.running))
        frames = []
        auto_sat = scene.sitting_on is None
        if auto_sat:
            frames.extend(
                replace(frame, running_devices=before)
                for frame in produce_sit(
                    rig,
                    g,
                    scene,
                    st,
                    Goal(Primitive.SIT, obj_key=seat_key),
                    sd_world=sd_world,
                )
            )
        if monitor_key:
            scene.running.add(monitor_key)
        running = tuple(sorted(scene.running))
        frames.extend(
            replace(frame, running_devices=running)
            for frame in _seated_computer_frames(
                rig, scene, st, goal.obj_key, gaze_target
            )
        )
        if auto_sat:
            frames.extend(
                replace(frame, running_devices=running)
                for frame in produce_standup(
                    rig,
                    g,
                    scene,
                    st,
                    Goal(Primitive.STANDUP),
                    sd_world=sd_world,
                )
            )
        return frames
    previous_rest = st.obj_rest_xyz
    if monitor_key:
        scene.running.add(monitor_key)
    running = tuple(sorted(scene.running))
    frames = _play_contact_traj(
        rig,
        scene,
        st,
        goal,
        kf=keyframes,
        coupling="GAZE",
        lean_z=float(keyframes[0][2]),
        seg="use_computer",
        contact_hand_close=0.15,
        gaze_target=gaze_target,
    )
    st.obj_rest_xyz = previous_rest
    return [replace(frame, running_devices=running) for frame in frames]


def _answer_phone_turn_degrees(scene, st: SchedState) -> float:
    current_clearance = _forward_clearance(
        scene,
        st.root_xy,
        st.root_yaw,
        _PHONE_CLEARANCE_PROBE_M,
    )
    if current_clearance >= _PHONE_MIN_FORWARD_CLEARANCE_M:
        return 0.0
    candidates = (90.0, -90.0, 135.0, -135.0, 180.0)
    scored = [
        (
            _forward_clearance(
                scene,
                st.root_xy,
                st.root_yaw + delta,
                _PHONE_CLEARANCE_PROBE_M,
            ),
            -abs(delta),
            -index,
            delta,
        )
        for index, delta in enumerate(candidates)
    ]
    best_clearance, _turn_cost, _order, best_delta = max(scored)
    if best_clearance < current_clearance + 0.20:
        return 0.0
    return float(best_delta)


def produce_answer_phone(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Raise the held mobile phone to the right ear and leave it there for a following Wait."""
    if st.held is None or str(st.held.get("obj_key")) != str(goal.obj_key):
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"ANSWER_PHONE requires {goal.obj_key!r} to be held",
        )
    entity = scene.resolve(goal.obj_key)
    category = str((getattr(entity, "meta", None) or {}).get("asset_category") or "")
    if category not in {"cell_phone", "mobile_phone", "smartphone"}:
        raise PrimitiveFailure(
            PrimitiveError.MISSING_AFFORDANCE,
            f"{goal.obj_key!r} is {category or 'an untyped object'}, not a mobile phone",
        )

    held = st.held
    carry_arm = held["grasp_arm"]
    is_seated = getattr(scene, "sitting_on", None) is not None and st.seated_z is not None
    root_z = st.seated_z if is_seated else st.stand_z
    turn_frames = []
    turn_degrees = 0.0 if is_seated else _answer_phone_turn_degrees(scene, st)
    if abs(turn_degrees) > 1e-6:
        turn_frames = produce_turn(
            rig,
            g,
            scene,
            st,
            Goal(Primitive.TURN, meta={"turn_deg": turn_degrees}),
            sd_world=sd_world,
        )
        st.last_pose156 = _manip(rig, carry_arm, 1.0, 0.0)
    carry_pose = (
        _seated_pose(rig, carry_arm)
        if is_seated
        else _manip(rig, carry_arm, 1.0, 0.0)
    )
    joints = core.H.fwd_joints(carry_pose)
    ear_local = np.asarray(joints[core.HEAD_J], dtype=np.float64) + np.array(
        [0.13, -0.015, -0.035], dtype=np.float64
    )
    rz, world_offset = wpp._grasp_frame(st.root_xy, st.root_yaw, root_z)
    phone_target = rz @ ear_local + np.asarray(world_offset, dtype=np.float64)
    phone_arm, error = _solve_reach_arm(
        rig,
        carry_arm,
        phone_target,
        st.root_xy,
        st.root_yaw,
        root_z,
        lean_override=0.0,
    )
    if error > 0.10:
        raise PrimitiveFailure(
            PrimitiveError.OUT_OF_REACH,
            f"phone-to-ear reach residual is {error * 1000:.0f}mm",
        )
    forward = np.array(
        [
            math.sin(math.radians(st.root_yaw)),
            -math.cos(math.radians(st.root_yaw)),
            0.0,
        ],
        dtype=np.float64,
    )
    gaze = phone_target + 2.0 * forward
    arrive_frames = (
        _seated_arrive_beats(rig, st, carry_arm, _obj_carry(st))
        if is_seated
        else _arrive_beats(rig, st, carry_arm, 1.0, _obj_carry(st))
    )
    frames = [*turn_frames, *arrive_frames]
    for index in range(_N_PHONE_RAISE):
        fraction = wpp._smoothstep((index + 1) / _N_PHONE_RAISE)
        frames.append(
            wpp.Frame(
                _manip(
                    rig,
                    core.lerp_arm(carry_arm, phone_arm, fraction),
                    1.0,
                ),
                st.root_xy,
                st.root_yaw,
                root_z,
                rhp=held.get("rhp"),
                fpv_mode="look",
                obj=_obj_carry(st),
                seg="answer_phone",
                gaze=gaze,
            )
        )
    ear_pose = (
        _seated_pose(rig, phone_arm)
        if is_seated
        else _manip(rig, phone_arm, 1.0, 0.0)
    )
    for _ in range(_N_HANDLE_HOLD):
        frames.append(
            wpp.Frame(
                ear_pose.copy(),
                st.root_xy,
                st.root_yaw,
                root_z,
                rhp=held.get("rhp"),
                fpv_mode="look",
                obj=_obj_carry(st),
                seg="answer_phone_hold",
                gaze=gaze,
            )
        )
    st.last_pose156 = ear_pose
    _stamp_obj_path(frames, scene, str(goal.obj_key))
    _stamp_carried(frames, held.get("carried"))
    return frames


# --- CAMERA / body verbs (coupling NONE; they exist to MOVE the egocentric camera) -------------------
def _squat_pose(rig, arm: dict, depth: float = 1.0) -> np.ndarray:
    """A crouch: deep knee flex + hips over the ankles + upright torso (NOT the sit-on-chair pose)."""
    p = core.assemble_pose(rig, arm, (0.0, 0.0, 0.0), 0.0)
    for j in (_HIP_L, _HIP_R):
        p[j * 3] = -0.7 * depth
    for j in (_KNEE_L, _KNEE_R):
        p[j * 3] = 1.2 * depth
    for j in (
        7,
        8,
    ):  # ankles: mild counter-rotate so the feet stay flat (large tilt sinks the soles)
        p[j * 3] = -0.35 * depth
    return p


def produce_look_around(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Turn one full circle in place so the egocentric camera surveys the room."""
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    base = _manip(rig, arm_rest, 0.0, 0.0)
    obj = _obj_carry(st)
    frames = []
    n = max(24, int(math.ceil(360.0 / _TURN_DEG_PER_FRAME)))
    for i in range(n):
        s = (i + 1) / n
        yaw = st.root_yaw + 360.0 * s
        frames.append(
            wpp.Frame(
                base,
                st.root_xy,
                yaw,
                _pin_z(base, st.floor_z),
                fpv_mode="walk",
                obj=obj,
                seg="look",
            )
        )
    st.last_pose156 = base
    return frames


def produce_look_scan(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Sweep in place like LOOK_AROUND, but with the gaze pinned to the FLOOR a couple of metres out.

    LOOK_AROUND is an eye-level pan: rendering it in the trash scene returned cabinets, a sink and a
    range hood, and not one of the three cans lying on the floor -- its pitch is the gentle walk clamp,
    which frames what is ahead rather than what is underfoot. Small objects on the ground fall below the
    bottom edge. This verb exists so a person can do what a person does when looking for something they
    dropped: turn on the spot with their eyes down.

    Implemented with the per-frame ``gaze`` override, which the ego camera aims at UNCLAMPED, so the
    pitch comes out of the geometry rather than a second magic angle.
    """
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    base = _manip(rig, arm_rest, 0.0, 0.0)
    obj = _obj_carry(st)
    reach = float(
        os.getenv("FPV_SCAN_REACH_M", "2.0")
    )  # how far out on the floor to look
    frames = []
    n = 24
    for i in range(n):
        s = (i + 1) / n
        # SAME +-45 deg swing as LOOK_AROUND. A full 360 turn can yield as little as one usable
        # frame out of 24: the collision nudge rejects most poses of an in-place spin, while
        # this amplitude is already proven to pass. Cover the room by issuing the verb more than
        # once from different headings rather than by spinning.
        yaw = st.root_yaw + 45.0 * math.sin(2 * math.pi * s)
        gz = np.array(
            [
                st.root_xy[0] + reach * math.cos(math.radians(yaw)),
                st.root_xy[1] + reach * math.sin(math.radians(yaw)),
                float(st.floor_z),
            ],
            dtype=np.float64,
        )
        frames.append(
            wpp.Frame(
                base,
                st.root_xy,
                yaw,
                _pin_z(base, st.floor_z),
                fpv_mode="look",  # NOT "walk": that branch hardcodes aim = head + fwd*3 - 0.25z and never reads gaze
                obj=obj,
                seg="look",  # the renderer keys on known seg names; "scan" would be filtered out
                gaze=gz,
            )
        )
    st.last_pose156 = base
    return frames


def _grip_clear_of_label(ent, rest_mat, obj_world, bbox_zmin, bbox_zmax) -> np.ndarray:
    """Where to take hold of something you are about to READ: away from its writing.

    Seating the inspect grasp at the object's ORIGIN puts it in the middle of a bottle's body, which
    is exactly where a bottle wears its label. The fingers close over it, and the raised hold
    delivers a hand with a strip of label above it: in one measured frame, of the 353 px of card on
    screen only the top 70 px, its header band, cleared the fingers. Nothing is wrong with the
    artwork, the aim, the lighting or the distance -- the hand is in front.

    A person shifts their grip to the neck. So does this: the hold moves along the object's own up
    axis to the MIDDLE of whichever clear run -- above the label or below it -- is longer. There is
    no margin to tune, because the midpoint of the free run is already as far from both the writing
    and the end of the object as that run allows. An object with no recorded label, one lying on
    its side, or one whose label covers it end to end keeps the origin grip it has always had.
    """
    meta = getattr(ent, "meta", None) or {}
    explicit_grip = meta.get("inspect_grip_local")
    if explicit_grip is not None:
        local = np.asarray(explicit_grip, dtype=np.float64)
        if local.shape == (3,):
            return rest_mat[:3, :3] @ local + np.asarray(obj_world, dtype=np.float64)
    loc, size = meta.get("label_local"), meta.get("label_size")
    if not loc or not size:
        return obj_world
    if meta.get("label_mount") == "top":
        face = np.asarray(
            meta.get("label_face_local", (0.0, 0.0, 1.0)), dtype=np.float64
        )
        label_up = np.asarray(
            meta.get("label_up_local", (0.0, 1.0, 0.0)), dtype=np.float64
        )
        label_right = np.cross(label_up, face)
        norm = float(np.linalg.norm(label_right))
        if norm > 1e-6:
            label_right /= norm
            label_center = np.asarray(loc, dtype=np.float64)
            edge_outset = min(0.025, 0.20 * float(size[0]))
            grip_local = label_center + label_right * (
                0.5 * float(size[0]) + edge_outset
            )
            return (
                rest_mat[:3, :3] @ grip_local
                + np.asarray(obj_world, dtype=np.float64)
            )
    up = rest_mat[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(up[2])) < 0.5:  # toppled: "along its up axis" is no longer a height
        return obj_world
    lab = rest_mat[:3, :3] @ np.array(
        [float(loc[0]), float(loc[1]), float(loc[2])], dtype=np.float64
    ) + np.asarray(obj_world, dtype=np.float64)
    half = 0.5 * float(size[1])
    top, bot = float(lab[2]) + half, float(lab[2]) - half
    up_room, dn_room = float(bbox_zmax) - top, bot - float(bbox_zmin)
    if max(up_room, dn_room) <= 1e-3:
        return obj_world
    out = np.array(obj_world, dtype=np.float64, copy=True)
    out[2] = (
        0.5 * (top + float(bbox_zmax))
        if up_room >= dn_room
        else 0.5 * (bot + float(bbox_zmin))
    )
    return out


def _look_grabbable_label(scene, key, ent, bbox, label):
    import label_visibility as lv

    meta = ent.meta or {}
    texture_path = _look_texture_path(scene, key, meta)
    local_position = meta.get("label_local")
    if local_position is not None:
        try:
            rest_transform = np.asarray(scene.object_rest_mat(key), dtype=np.float64)
        except (AttributeError, KeyError, TypeError, ValueError):
            rest_transform = None
        if rest_transform is not None and rest_transform.shape == (4, 4):
            local_meta = dict(meta)
            local_meta["label_pos"] = local_position
            for local_name, world_name, axis in (
                ("label_face_local", "label_face3", getattr(label, "face", None)),
                ("label_up_local", "label_up3", getattr(label, "up", None)),
            ):
                local_axis = meta.get(local_name)
                if local_axis is None and axis is not None:
                    local_axis = rest_transform[:3, :3].T @ np.asarray(axis)
                local_meta[world_name] = local_axis
            local_label = lv.label_from_meta(local_meta, texture_path=texture_path)
            if local_label is not None:
                return lv.transform_label(local_label, rest_transform)
    if label is not None:
        return label
    return lv.label_from_meta(meta, bbox, texture_path=texture_path)


def _look_hold_rotation(rest_rotation, label, meta, toward_camera):
    """Return the held orientation, preferring the complete authored label frame."""
    import label_visibility as lv

    if label is not None:
        aligned = lv.front_facing_object_rotation(
            rest_rotation,
            label.face,
            label.up,
            toward_camera,
        )
        if aligned is not None:
            return aligned

    label_face = meta.get("label_face")
    delta = 0.0
    if label_face and (abs(float(label_face[0])) + abs(float(label_face[1])) > 1e-6):
        face_world = rest_rotation @ np.array(
            [float(label_face[0]), float(label_face[1]), 0.0], dtype=np.float64
        )
        if abs(face_world[0]) + abs(face_world[1]) > 1e-6:
            delta = math.atan2(toward_camera[1], toward_camera[0]) - math.atan2(
                float(face_world[1]), float(face_world[0])
            )
    cosine, sine = math.cos(delta), math.sin(delta)
    yaw_rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return yaw_rotation @ rest_rotation


def _fit_look_inspect_center(
    rig,
    st,
    key,
    stand,
    grasp_arm,
    inspect_center,
    grip_to_label,
    reach_tolerance,
    meta=None,
    *,
    label_shift=None,
):
    """Move the held label just outside near-read range when the arm can reach it."""
    import label_visibility as lv

    actual_grip_to_label = np.asarray(grip_to_label, dtype=np.float64)
    if label_shift is not None:
        actual_grip_to_label = actual_grip_to_label + np.asarray(
            label_shift, dtype=np.float64
        )
    label_at = inspect_center + actual_grip_to_label
    eye = _held_policy_eye_looking_at(stand, st, label_at)
    direction = label_at - eye
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-6:
        return inspect_center
    direction /= distance
    label_meta = meta or {}
    native_package = label_meta.get("label_source") == "native_asset_texture"
    native_default_distance = lv.HELD_NEAR_M + 0.02
    requested_distance = label_meta.get("look_held_distance_m")
    if requested_distance is not None:
        read_distance = max(lv.HELD_NEAR_M, float(requested_distance))
    elif native_package:
        read_distance = native_default_distance
    else:
        read_distance = lv.NEAR_M + 0.05
    for _ in range(3):
        eye = _held_policy_eye_looking_at(stand, st, label_at)
        label_at = eye + direction * read_distance
    wanted_center = label_at - actual_grip_to_label
    floor_z = float(getattr(st, "floor_z", 0.0))
    squat_frac = max(0.0, float(getattr(st, "crouch_fraction", 0.0)))
    reach_kwargs = {"squat_frac": squat_frac} if squat_frac > 1e-6 else {}
    pose_root_z = _pin_z(stand, floor_z)
    _arm, reach_error = _solve_reach_arm(
        rig,
        grasp_arm,
        wanted_center,
        st.root_xy,
        st.root_yaw,
        pose_root_z,
        **reach_kwargs,
    )
    error_limit = max(reach_tolerance, _REACH_TOL_M)
    if reach_error <= error_limit:
        return wanted_center

    # The exact reading-distance target can sit just outside the arm workspace. Do not jump all
    # the way back to the original face-close hold: find the furthest collision-independent point
    # along the same geometric correction that the arm can actually reach.
    reachable_center = np.asarray(inspect_center, dtype=np.float64)
    low, high = 0.0, 1.0
    for _ in range(6):
        fraction = 0.5 * (low + high)
        candidate = inspect_center + fraction * (wanted_center - inspect_center)
        _arm, candidate_error = _solve_reach_arm(
            rig,
            grasp_arm,
            candidate,
            st.root_xy,
            st.root_yaw,
            pose_root_z,
            **reach_kwargs,
        )
        if candidate_error <= error_limit:
            low = fraction
            reachable_center = candidate
        else:
            high = fraction
    print(
        f"[lookat] {key}: reading distance {read_distance:.2f} m out of reach "
        f"({reach_error * 1000:.0f}mm); using {low * 100:.0f}% of the reachable correction"
    )
    return reachable_center


def _held_policy_eye_looking_at(stand, st, target):
    """Predict the exact virtual FPV eye after the body's gaze solve."""
    import egocentric_camera as ego_camera
    import gaze_stabilizer as gaze
    import head_mount

    root_z = _pin_z(stand, float(getattr(st, "floor_z", 0.0)))
    mounted_eye, head_rotation = gaze.eye_looking_at(
        stand, st.root_yaw, st.root_xy, root_z, target
    )
    head = mounted_eye - head_rotation @ head_mount.EYE_LOCAL
    config = ego_camera.DEFAULT_CAMERA_CONFIG
    return (
        head
        + np.asarray(wpp._body_forward(st.root_yaw), dtype=np.float64)
        * config.manip_eye_forward
        + np.array([0.0, 0.0, config.manip_eye_lift], dtype=np.float64)
    )


def _camera_facing_hold_rotation(
    rest_rotation,
    label,
    meta,
    stand,
    st,
    inspect_center,
    grip_to_label_local,
    initial_toward_camera,
):
    """Face a label at the eye while keeping its intended reading point fixed."""
    if meta.get("label_source") != "native_asset_texture":
        rotation = _look_hold_rotation(
            rest_rotation, label, meta, initial_toward_camera
        )
        for _ in range(2):
            label_at = inspect_center + rotation @ grip_to_label_local
            eye = _held_policy_eye_looking_at(stand, st, label_at)
            rotation = _look_hold_rotation(rest_rotation, label, meta, eye - label_at)
        return rotation
    label_at = inspect_center + rest_rotation @ grip_to_label_local
    eye = _held_policy_eye_looking_at(stand, st, label_at)
    toward_camera = eye - label_at
    if float(np.linalg.norm(toward_camera)) <= 1e-6:
        toward_camera = initial_toward_camera
    return _look_hold_rotation(rest_rotation, label, meta, toward_camera)


def _label_anchored_inspect_center(
    inspect_center,
    rest_rotation,
    held_rotation,
    grip_to_label_local,
):
    label_at = inspect_center + rest_rotation @ grip_to_label_local
    return label_at - held_rotation @ grip_to_label_local


def _assess_held_label(
    stand,
    st,
    label,
    held_transform,
    rest_transform,
    fallback_position,
    fallback_span,
    min_detail_px,
    meta,
):
    """Measure the exact held pose at the resolution delivered to the model."""
    import head_mount
    import label_visibility as lv

    if label is None:
        held_label = lv.Label(fallback_position, fallback_span)
    else:
        held_label = lv.transform_label(
            label, held_transform @ np.linalg.inv(rest_transform)
        )
    eye = _held_policy_eye_looking_at(stand, st, held_label.pos)
    zoom_target = int(
        min_detail_px
        if min_detail_px is not None
        else _look_planning_detail_px(held_label, meta)
    )
    required_detail = _look_required_detail_px(held_label, meta)
    zoom_span = _look_zoom_span(held_label, zoom_target)
    focal_mm = lv.zoom_focal_mm(zoom_span, float(np.linalg.norm(held_label.pos - eye)))
    text_metric = held_label.glyph_height_m is not None
    assessment = lv.assess(
        eye,
        head_mount.spine3_frame(
            stand,
            st.root_yaw,
            tuple(st.root_xy),
            _pin_z(stand, float(getattr(st, "floor_z", 0.0))),
        )[0],
        held_label,
        image_w=lv.DELIVERED_IMAGE_W if text_metric else 960,
        image_h=lv.DELIVERED_IMAGE_H if text_metric else 760,
        min_px=required_detail,
        min_glyph_px=required_detail,
        focal_mm=focal_mm,
        camera_look=held_label.pos - eye,
        enforce_torso_limits=False,
        min_distance_m=(
            lv.HELD_NEAR_M
            if meta.get("label_source") == "native_asset_texture"
            or meta.get("look_held_distance_m") is not None
            else lv.NEAR_M
        ),
    )
    return assessment, required_detail, zoom_span


def _held_object_world_transform(rig, st) -> np.ndarray:
    """Reconstruct the carried object's transform at the scheduler's current pose."""
    if st.held is None or st.last_pose156 is None:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            "held-label inspection requires a carried object and a current pose",
        )
    pose = np.asarray(st.last_pose156, dtype=np.float32)
    return _held_wrist_world_transform(
        rig, st, pose, st.held.get("grasp_arm")
    ) @ np.asarray(st.held["otw"], dtype=np.float64)


def _held_wrist_world_transform(rig, st, pose, arm) -> np.ndarray:
    """Return the live wrist frame for a carried-object arm pose."""
    pose = np.asarray(pose, dtype=np.float32)
    root_z = _pin_z(pose, st.floor_z)
    joints = gait.pose_place_joints(
        pose,
        st.root_xy,
        st.root_yaw,
        root_z,
        arm_override=arm,
        rhp=st.held.get("rhp"),
    )
    return np.asarray(core.wrist_world_T(joints), dtype=np.float64)


def _held_label_clearance_shift(
    rest_transform: np.ndarray,
    rest_label,
    held_rotation: np.ndarray,
) -> np.ndarray:
    """Return a small in-plane shift that keeps fingers above printed text."""
    if rest_label.up is None:
        return np.zeros(3, dtype=np.float64)
    local_up = rest_transform[:3, :3].T @ np.asarray(rest_label.up, dtype=np.float64)
    held_up = np.asarray(held_rotation, dtype=np.float64) @ local_up
    norm = float(np.linalg.norm(held_up))
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    clearance_m = 0.025
    lateral_clearance_m = 0.0
    if rest_label.size_m is not None:
        label_width, label_height = (abs(float(value)) for value in rest_label.size_m)
        aspect = label_width / max(label_height, 1e-9)
        if aspect >= 0.8:
            clearance_m = min(0.060, max(clearance_m, 0.5 * label_height))
        elif label_height <= 0.10:
            clearance_m = min(0.045, max(clearance_m, 0.5 * label_height))
            lateral_clearance_m = min(0.015, 0.45 * label_width)
    shift = -held_up / norm * clearance_m
    if lateral_clearance_m > 0.0 and rest_label.face is not None:
        local_face = rest_transform[:3, :3].T @ np.asarray(
            rest_label.face, dtype=np.float64
        )
        held_face = np.asarray(held_rotation, dtype=np.float64) @ local_face
        held_right = np.cross(held_up, held_face)
        right_norm = float(np.linalg.norm(held_right))
        if right_norm > 1e-9:
            shift -= held_right / right_norm * lateral_clearance_m
    return shift


def _lower_held_label_below_grip(
    desired_transform: np.ndarray,
    rest_transform: np.ndarray,
    rest_label,
):
    """Slide the carton down in the grip so fingers do not cover its printed text."""
    import label_visibility as lv

    desired_label = lv.transform_label(
        rest_label, desired_transform @ np.linalg.inv(rest_transform)
    )
    shift = _held_label_clearance_shift(
        rest_transform, rest_label, desired_transform[:3, :3]
    )
    if not np.any(shift):
        return desired_transform, desired_label
    shifted = desired_transform.copy()
    shifted[:3, 3] += shift
    return shifted, lv.transform_label(
        rest_label, shifted @ np.linalg.inv(rest_transform)
    )


def _held_label_rest_pose(scene, key, ent):
    try:
        bbox = scene.object_bbox(key)
        rest_transform = np.asarray(scene.object_rest_mat(key), dtype=np.float64)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if rest_transform.shape != (4, 4):
        return None
    rest_label = _look_grabbable_label(scene, key, ent, bbox, None)
    return None if rest_label is None else (rest_transform, rest_label)


def _solve_held_wrist_arm(rig, st, base, initial_arm, desired_wrist):
    solve_pose = np.asarray(base, dtype=np.float32).copy()
    for joint, value in (initial_arm or {}).items():
        solve_pose[joint * 3 : joint * 3 + 3] = np.asarray(value, dtype=np.float32)
    rhp = st.held.get("rhp")
    if rhp is not None:
        solve_pose[core.H.RHAND_LO : core.H.RHAND_LO + 45] = np.asarray(
            rhp, dtype=np.float32
        )
    root_z = _pin_z(solve_pose, st.floor_z)
    root_rotation, world_offset = wpp._grasp_frame(
        st.root_xy, st.root_yaw, root_z
    )
    wrist_rotation_local = root_rotation.T @ desired_wrist[:3, :3]
    reach_error = float("inf")
    orientation_error = float("inf")
    for _ in range(2):
        solve_pose, reach_error = core.warp_arm_to_wrist(
            rig,
            solve_pose,
            root_rotation,
            world_offset,
            desired_wrist[:3, 3],
        )
        wrist, orientation_error, _bend = core.solve_wrist_orient(
            rig, solve_pose, wrist_rotation_local
        )
        solve_pose[core.WRIST_J * 3 : core.WRIST_J * 3 + 3] = wrist
    return wpp._arm_from_pose(solve_pose), reach_error, orientation_error


def _held_label_target(
    rig,
    st,
    key,
    meta,
    rest_transform,
    rest_label,
    initial_transform,
    reach_tolerance,
):
    import label_visibility as lv

    initial_label = lv.transform_label(
        rest_label, initial_transform @ np.linalg.inv(rest_transform)
    )
    base = np.asarray(st.last_pose156, dtype=np.float32)
    initial_arm = st.held.get("grasp_arm")
    initial_otw = np.asarray(st.held["otw"], dtype=np.float64)
    joints = gait.pose_place_joints(
        base,
        st.root_xy,
        st.root_yaw,
        _pin_z(base, st.floor_z),
        arm_override=initial_arm,
        rhp=st.held.get("rhp"),
    )
    head = np.asarray(joints[core.HEAD_J], dtype=np.float64)
    forward = np.asarray(wpp._body_forward(st.root_yaw), dtype=np.float64)
    inspect_label = np.array(
        [
            head[0] + forward[0] * _INSPECT_FWD_M,
            head[1] + forward[1] * _INSPECT_FWD_M,
            head[2],
        ],
        dtype=np.float64,
    )
    eye = _held_policy_eye_looking_at(base, st, initial_label.pos)
    local_label = (
        np.linalg.inv(rest_transform)
        @ np.append(np.asarray(rest_label.pos, dtype=np.float64), 1.0)
    )[:3]
    inspect_origin = inspect_label - rest_transform[:3, :3] @ local_label
    desired_rotation = _camera_facing_hold_rotation(
        rest_transform[:3, :3],
        rest_label,
        meta,
        base,
        st,
        inspect_origin,
        local_label,
        eye - initial_label.pos,
    )
    label_from_origin = desired_rotation @ local_label
    inspect_origin = _fit_look_inspect_center(
        rig,
        st,
        key,
        base,
        initial_arm,
        inspect_label - label_from_origin,
        label_from_origin,
        reach_tolerance,
        meta,
    )
    desired_transform = np.eye(4, dtype=np.float64)
    desired_transform[:3, :3] = desired_rotation
    desired_transform[:3, 3] = inspect_origin
    desired_wrist = desired_transform @ np.linalg.inv(initial_otw)
    desired_arm, _reach_error, _orientation_error = _solve_held_wrist_arm(
        rig, st, base, initial_arm, desired_wrist
    )
    achieved_wrist = _held_wrist_world_transform(rig, st, base, desired_arm)
    desired_transform = achieved_wrist @ initial_otw
    desired_label = lv.transform_label(
        rest_label, desired_transform @ np.linalg.inv(rest_transform)
    )
    return base, initial_arm, desired_arm, desired_transform, desired_label


def _held_look_phase(index: int, n_in: int, n_hold: int, n_out: int):
    if index < n_in:
        fraction = wpp._smoothstep((index + 1) / n_in)
        return fraction, fraction, "look"
    if index < n_in + n_hold:
        return 1.0, 1.0, "look_hold"
    return_fraction = wpp._smoothstep((index - n_in - n_hold + 1) / n_out)
    fraction = 1.0 - return_fraction
    return fraction, fraction, "look"


def _held_look_frames(
    rig,
    st,
    ent,
    base,
    rest_transform,
    rest_label,
    initial_arm,
    desired_arm,
    initial_otw,
    zoom_span,
):
    import label_visibility as lv

    frames = []
    n_in, n_hold, n_out = _N_LOOK_RAISE, _N_LOOK_HOLD, _N_LOOK_LOWER
    for index in range(n_in + n_hold + n_out):
        fraction, zoom, segment = _held_look_phase(index, n_in, n_hold, n_out)
        frame_arm = core.lerp_arm(initial_arm, desired_arm, fraction)
        frame_transform = (
            _held_wrist_world_transform(rig, st, base, frame_arm) @ initial_otw
        )
        frame_label = lv.transform_label(
            rest_label, frame_transform @ np.linalg.inv(rest_transform)
        )
        frames.append(
            wpp.Frame(
                base.copy(),
                st.root_xy,
                st.root_yaw,
                _pin_z(base, st.floor_z),
                arm_override=frame_arm,
                rhp=st.held.get("rhp"),
                fpv_mode="look",
                obj=("pose", frame_transform),
                obj_path=getattr(ent, "prim_path", None),
                seg=segment,
                gaze=frame_label.pos.copy(),
                zoom=zoom,
                zoom_span=zoom_span,
                gaze_locked=True,
            )
        )
    return frames


def _look_at_held_grabbable(rig, scene, st, goal, ent):
    """Bring a carried label to the eye, hold it readable, then resume carrying."""
    if st.held is None or str(st.held.get("obj_key")) != str(goal.obj_key):
        return None
    rest_pose = _held_label_rest_pose(scene, goal.obj_key, ent)
    if rest_pose is None:
        return None
    rest_transform, rest_label = rest_pose
    meta = ent.meta or {}
    initial_otw = np.asarray(st.held["otw"], dtype=np.float64)
    initial_transform = _held_object_world_transform(rig, st)
    base, initial_arm, desired_arm, desired_transform, desired_label = (
        _held_label_target(
            rig,
            st,
            goal.obj_key,
            meta,
            rest_transform,
            rest_label,
            initial_transform,
            _reach_tolerance(scene),
        )
    )
    assessment, detail_target, zoom_span = _assess_held_label(
        base,
        st,
        rest_label,
        desired_transform,
        rest_transform,
        desired_label.pos,
        desired_label.span_m,
        _look_planning_detail_px(rest_label, meta),
        meta,
    )
    frames = _held_look_frames(
        rig,
        st,
        ent,
        base,
        rest_transform,
        rest_label,
        initial_arm,
        desired_arm,
        initial_otw,
        zoom_span,
    )
    _stamp_obj_path(frames, scene, str(goal.obj_key))
    _stamp_carried(frames, st.held.get("carried"))
    outcome = "readable" if assessment.ok else f"best-effort {assessment.verdict}"
    print(
        f"[lookat] {goal.obj_key}: held label {outcome} at {assessment.dist_m:.2f}m "
        f"({(assessment.glyph_px or assessment.px):.1f}px, target {detail_target}px)"
    )
    return frames


def _look_at_inspect_grabbable(
    rig, scene, st, goal, ent, sd_world, *, label=None, min_detail_px=None
):
    """LookAt a GRABBABLE: BRING THE OBJECT UP TO THE FACE. Grasp it at rest (reusing produce_pick's
    solve_grasp), RAISE it to ~arm's length in front of the FPV eye at eye height with its label turned to the
    camera, HOLD it still >=24 frames (>=2 s @12fps), then LOWER + RELEASE back to rest. The FPV eye stays at
    the head and tracks the object. Self-contained: st.held stays None and the object ends at its rest pose (a
    following [Grab] re-picks it), so it never leaves the scene mid-inspect. The lift interpolates from the exact
    rest orientation to a full 3-D label-facing orientation, then restores the exact rest transform."""
    key = goal.obj_key
    crouch_owner = _crouch_owner_for_target(scene, st, key)
    keep_crouched = crouch_owner is not None
    minimum_squat = (
        max(_FRONT_APPLIANCE_SQUAT, _active_crouch_fraction(scene, st, key))
        if keep_crouched
        else 0.0
    )
    obj_world = np.asarray(scene.object_origin(key), dtype=np.float64)  # rest origin
    rest_mat = scene.object_rest_mat(
        key
    )  # full rest 4x4 (upright orientation preserved)
    (mnx, mny, mnz), (mxx, mxy, mxz) = scene.object_aabb(key)
    import label_visibility as _lv

    label = _look_grabbable_label(
        scene,
        key,
        ent,
        ((0.5 * (mnx + mxx), 0.5 * (mny + mxy)), mnz, mxz),
        label,
    )
    center_off = (
        np.array(
            [0.5 * (mnx + mxx), 0.5 * (mny + mxy), 0.5 * (mnz + mxz)], dtype=np.float64
        )
        - obj_world
    )  # origin -> bbox centre at rest (world)
    span = 0.5 * max(
        float(mxz - mnz), float(min(mxx - mnx, mxy - mny))
    )  # label size -> focal zoom
    grip_world = _grip_clear_of_label(ent, rest_mat, obj_world, mnz, mxz)

    # --- grasp at rest (reuse produce_pick's pose-plan + solve_grasp: warp the arm + wrap the fingers) ---
    _pp_on = bool(getattr(scene, "unified_geom", False))
    _tol = _reach_tolerance(scene)
    _plan = plan_reach(
        rig,
        rig.arm_grasp,
        grip_world,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        _tol,
        floor_z=st.floor_z,
        enabled=_pp_on,
        clear_fn=body_clear_fn(scene, st.root_xy, st.root_yaw, st.floor_z),
        allow_colliding_fallback=False,
        upright_first=keep_crouched,
        minimum_squat=minimum_squat,
    )
    if _plan.get("collision_blocked"):
        raise PrimitiveFailure(
            PrimitiveError.TRAJECTORY_COLLISION,
            f"LOOKAT {key!r} has no collision-free reach pose",
        )
    lean_rad, lf, squat_f, reach_z = (
        _plan["lean_rad"],
        _plan["lf"],
        _plan["squat"],
        _plan["stand_z"],
    )
    rz, world_off = wpp._grasp_frame(st.root_xy, st.root_yaw, reach_z)
    base_reach = _reach_base_pose(rig, rig.arm_grasp, lean_rad, squat_f, 1.0)
    grasp = wpp.solve_grasp(
        rig,
        base_reach,
        rz,
        world_off,
        grip_world,
        sd_world,
        family_seed=False,
        planned_pose=_plan.get("arm_pose"),
        planned_err=_plan.get("err"),
    )
    grasp_arm, rhp = grasp["grasp_arm"], grasp["rhp"]
    rig.rhp = rhp  # so assemble_pose(hand_close=1) reproduces the wrapped grip
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)

    # --- inspect pose: object CENTRE ~arm's length in front of the FPV eye, at eye (head) height ---
    _stand = _manip(
        rig,
        arm_rest,
        0.0,
        0.0,
        squat_f if keep_crouched else 0.0,
    )
    inspect_root_z = _pin_z(_stand, st.floor_z)
    _J0 = gait.pose_place_joints(_stand, st.root_xy, st.root_yaw, inspect_root_z)
    head = np.asarray(_J0[core.HEAD_J], dtype=np.float64)
    fwd = wpp._body_forward(st.root_yaw)  # [sin,-cos,0] horizontal unit
    inspect_center = np.array(
        [head[0] + fwd[0] * _INSPECT_FWD_M, head[1] + fwd[1] * _INSPECT_FWD_M, head[2]],
        dtype=np.float64,
    )
    # Offsets are carried in the object's OWN frame from here on. ``center_off`` is measured in
    # WORLD at rest, so composing it with the raised rotation would apply the rest orientation
    # TWICE -- invisible while every graspable settles upright, wrong the moment one does not.
    _rest_R = rest_mat[:3, :3]
    center_off_local = _rest_R.T @ center_off
    # The point the HAND holds, in the object's own frame. The raise carries THIS to the inspect
    # point, because it is the hand the arm IK drives; pinning the bounding-box centre there
    # instead leaves the grip floating off any object whose origin is not its centre.
    grip_off_local = _rest_R.T @ (grip_world - obj_world)
    # WHAT THE CAMERA IS AIMED AT: the label, when the mount recorded where it put one. Aiming at
    # the bounding-box centre puts the writing wherever it happens to sit relative to it -- on a
    # typical bottle the band's centre is 57 mm below the bbox centre, a fifth of the frame at inspect
    # range.
    _llocal = ent.meta.get("label_local")
    aim_off_local = (
        np.array(
            [float(_llocal[0]), float(_llocal[1]), float(_llocal[2])], dtype=np.float64
        )
        if _llocal
        else center_off_local
    )
    # Turn the complete label frame toward the eye. A yaw-only turn works for a side label but
    # leaves a top label horizontal; mapping both its face and up axes makes either mount square and
    # upright in the image.
    #
    # HOLD IT AT A READABLE DISTANCE FROM THE EYE THAT WILL EXIST, not the one that does now.
    #
    # ``inspect_center`` above is measured from the HEAD JOINT of a wearer looking straight ahead.
    # They are about to look DOWN at the thing, and that moves the eye: it rides 0.11 m in front of
    # the joint and 0.09 m above it, so a 40 deg pitch swings it ~0.10 m forward and down. Without
    # this, a placement intended at 0.31 m eye-to-label can render at 0.202 m -- 40% inside
    # ``NEAR_M``, the closest range this codebase calls readable, with the object filling the
    # frame. Solving against ``gaze_stabilizer.eye_looking_at`` closes the loop:
    # place, aim, see where the eye went, place again. The direction is left exactly as it was, so
    # only the distance changes.
    _label_from_grip_local = aim_off_local - grip_off_local
    _R_hold = _camera_facing_hold_rotation(
        _rest_R,
        label,
        ent.meta or {},
        _stand,
        st,
        inspect_center,
        _label_from_grip_local,
        -fwd,
    )
    if (ent.meta or {}).get("label_source") == "native_asset_texture":
        inspect_center = _label_anchored_inspect_center(
            inspect_center,
            _rest_R,
            _R_hold,
            _label_from_grip_local,
        )
    _grip_to_label = _R_hold @ _label_from_grip_local
    _grip_clearance_shift = _held_label_clearance_shift(rest_mat, label, _R_hold)
    inspect_center = _fit_look_inspect_center(
        rig,
        st,
        key,
        _stand,
        grasp_arm,
        inspect_center,
        _grip_to_label,
        _tol,
        ent.meta or {},
        label_shift=(
            _grip_clearance_shift
            if (ent.meta or {}).get("look_held_distance_m") is not None
            else None
        ),
    )

    def _obj_T(s):
        # Object 4x4 at raise fraction s in [0,1]. Interpolating the complete orientation is what
        # lets a horizontal top label rotate toward the eye without snapping during lift/lower.
        R = _lv.interpolate_rotation(_rest_R, _R_hold, s)
        origin_at_inspect = inspect_center - R @ grip_off_local
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = (
            obj_world * (1.0 - s) + origin_at_inspect * s + _grip_clearance_shift * s
        )
        return T

    def _center_of(T):
        return (T[:3, :3] @ aim_off_local) + T[:3, 3]  # the FPV gaze point

    # raised hold arm: IK the grip up to the inspect centre so the hand appears to hold the object at the face.
    raised_arm, _rerr = _solve_reach_arm(
        rig,
        grasp_arm,
        inspect_center,
        st.root_xy,
        st.root_yaw,
        inspect_root_z,
        squat_frac=squat_f if keep_crouched else 0.0,
    )
    _held_transform = _obj_T(1.0)
    _lab = _center_of(_held_transform)
    _rest_transform = np.eye(4, dtype=np.float64)
    _rest_transform[:3, :3] = _rest_R
    _rest_transform[:3, 3] = obj_world
    _held, _detail_target, _zoom_span = _assess_held_label(
        _stand,
        st,
        label,
        _held_transform,
        _rest_transform,
        _lab,
        float(ent.meta.get("label_span") or 2.0 * span),
        min_detail_px,
        ent.meta or {},
    )
    _relative_rotation = _R_hold @ _rest_R.T
    _turn_deg = math.degrees(
        math.acos(
            min(1.0, max(-1.0, 0.5 * (float(np.trace(_relative_rotation)) - 1.0)))
        )
    )
    print(
        f"[lookat] {key}: grasp_err={grasp['seat_err'] * 1000:.0f}mm "
        f"raise_err={_rerr * 1000:.0f}mm label_turn={_turn_deg:.0f}deg "
        f"held={_held.verdict} {_held.dist_m:.2f}m "
        f"{(_held.glyph_px or _held.px):.1f}px pitch={_held.pitch_deg:+.0f}deg "
        f"roll={_held.text_roll_deg:+.0f}deg"
    )
    if not _held.ok:
        raise PrimitiveFailure(
            PrimitiveError.UNREADABLE_LABEL,
            f"{key}: held_{_held.verdict}; detail={(_held.glyph_px or _held.px):.1f}px/"
            f"{_detail_target}px, pitch={_held.pitch_deg:+.1f}deg, "
            f"distance={_held.dist_m:.2f}m, roll={_held.text_roll_deg:+.1f}deg",
        )

    pp = ent.prim_path
    frames: list = []

    def _add(pose, T, seg, zoom=None):
        frames.append(
            wpp.Frame(
                pose,
                st.root_xy,
                st.root_yaw,
                _pin_z(pose, st.floor_z),
                fpv_mode="look",
                obj=("pose", T),
                obj_path=pp,
                seg=seg,
                gaze=_center_of(T),
                zoom=zoom,
                zoom_span=(_zoom_span if zoom is not None else None),
                gaze_locked=True,  # object is in front of the face -> keep the cam on it (no redirect)
            )
        )

    T_rest, T_up = _obj_T(0.0), _obj_T(1.0)
    for i in range(
        _N_LOOK_REACH
    ):  # REACH: crouch/lean in + close the hand onto the still-resting object
        s = wpp._smoothstep((i + 1) / _N_LOOK_REACH)
        _add(
            _manip(
                rig,
                core.lerp_arm(arm_rest, grasp_arm, s),
                s,
                lf * s,
                squat_f if keep_crouched else squat_f * s,
            ),
            T_rest,
            "look_reach",
        )
    for i in range(
        _N_LOOK_RAISE
    ):  # RAISE: stand up + bring the object to the face, turning its label to cam
        s = wpp._smoothstep((i + 1) / _N_LOOK_RAISE)
        _add(
            _manip(
                rig,
                core.lerp_arm(grasp_arm, raised_arm, s),
                1.0,
                lf * (1.0 - s),
                squat_f if keep_crouched else squat_f * (1.0 - s),
            ),
            _obj_T(s),
            "look_raise",
            zoom=s,
        )
    for _i in range(
        _N_LOOK_HOLD
    ):  # HOLD: object still at the face, label to the camera, >=2 s
        _add(
            _manip(
                rig,
                raised_arm,
                1.0,
                0.0,
                squat_f if keep_crouched else 0.0,
            ),
            T_up,
            "look_hold",
            zoom=1.0,
        )
    for i in range(
        _N_LOOK_LOWER
    ):  # LOWER: crouch/lean back + return the object to the bench
        s = wpp._smoothstep((i + 1) / _N_LOOK_LOWER)
        _add(
            _manip(
                rig,
                core.lerp_arm(raised_arm, grasp_arm, s),
                1.0,
                lf * s,
                squat_f if keep_crouched else squat_f * s,
            ),
            _obj_T(1.0 - s),
            "look_lower",
            zoom=(1.0 - s),
        )
    for i in range(
        _N_LOOK_RELEASE
    ):  # RELEASE: open the hand + stand up; the object stays at rest
        s = wpp._smoothstep((i + 1) / _N_LOOK_RELEASE)
        _add(
            _manip(
                rig,
                core.lerp_arm(grasp_arm, arm_rest, s),
                1.0 - s,
                lf * (1.0 - s),
                squat_f if keep_crouched else squat_f * (1.0 - s),
            ),
            T_rest,
            "look_release",
        )
    import action_check

    if action_check.uses_egocentric_camera_collision_policy(scene):
        camera_hit = first_fpv_eye_collision(scene, frames)
        if camera_hit is not None:
            raise PrimitiveFailure(
                PrimitiveError.TRAJECTORY_COLLISION,
                f"LOOKAT {key!r} intersects {camera_hit[1]} with the FPV eye "
                f"at frame {camera_hit[0]}",
            )
    st.last_pose156 = _stand
    if keep_crouched:
        _set_crouch_hold(st, crouch_owner, squat_f, st.last_pose156)
    return frames


# No on-normal reading stand.
#
# "Where must the wearer stand to read this placard head-on" is a question neither habitat-lab nor
# IsaacLab asks: both are satisfied by facing the OBJECT, not its surface normal. See
# produce_look_at for the cost of squaring up (75 deg of body rotation to improve a 23.5 deg
# oblique read that is already legible). scene.cue_stand_misses stays on SceneState -- it is read
# by the interaction episode loop and is always empty, which is the honest answer: nothing can fail
# to reach a stand that is never sought.


def _expected_attach_label_pose(scene: SceneState, meta: dict):
    expected_child = meta.get("expected_attach_child")
    expected_parent = meta.get("expected_attach_parent")
    if not expected_child or not expected_parent:
        return None
    target = _resolve_attach_target(
        scene,
        str(expected_child),
        str(expected_parent),
        {
            "child_point": meta.get("expected_child_point"),
            "parent_point": meta.get("expected_parent_point"),
            "child_point_idx": int(meta.get("expected_child_point_idx") or 0),
            "parent_point_idx": int(meta.get("expected_parent_point_idx") or 0),
        },
    )
    expected_world = target["child_world"]
    child_e = target["child_entity"]
    label_path = meta.get("expected_label_path")
    label_world = (
        scene._live_world_col(str(label_path))
        if label_path and getattr(scene, "stage", None) is not None
        else None
    )
    child_world = (
        scene._live_world_col(child_e.prim_path)
        if getattr(child_e, "prim_path", None)
        else None
    )
    expected_label_world = expected_world
    if label_world is not None and child_world is not None:
        label_local = np.linalg.inv(child_world) @ label_world
        expected_label_world = expected_world @ label_local
    target_xyz = expected_label_world[:3, 3]
    normal = expected_label_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
    normal_xy = np.asarray(normal[:2], dtype=np.float64)
    n = float(np.linalg.norm(normal_xy))
    if n < 1e-6:
        parent_e = target["parent_entity"]
        parent_xy = (
            scene.object_origin(str(expected_parent))[:2]
            if getattr(parent_e, "prim_path", None)
            else np.asarray(parent_e.xy or (0.0, 0.0), dtype=np.float64)
        )
        normal_xy = np.asarray(target_xyz[:2], dtype=np.float64) - parent_xy
        n = float(np.linalg.norm(normal_xy))
    if n < 1e-6:
        normal_xy = np.array([0.0, -1.0], dtype=np.float64)
    else:
        normal_xy = normal_xy / n
    return np.asarray(target_xyz, dtype=np.float64), normal_xy


def _dynamic_look_target(scene: SceneState, ent):
    if ent is None:
        return None
    meta = ent.meta or {}

    def _pack(target, span):
        return (
            np.asarray(target, dtype=np.float64),
            float(meta.get("label_span") or span),
            bool(meta.get("gaze_locked", True)),
            bool(meta.get("force_look_zoom", False)),
            bool(meta.get("fixed_look_frames", False)),
        )

    look_region = meta.get("look_region")
    if look_region is not None:
        try:
            target, span = interaction_hints.look_region_target(
                scene,
                str(ent.name),
                ent,
                look_region,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise PrimitiveFailure(
                PrimitiveError.NO_OBJECT,
                f"LOOK_AT region for {ent.name!r} cannot be resolved: {exc}",
            ) from exc
        return (
            np.asarray(target, dtype=np.float64),
            float(span),
            True,
            True,
            bool(meta.get("fixed_look_frames", False)),
        )

    try:
        expected = _expected_attach_label_pose(scene, meta)
    except Exception:  # noqa: BLE001
        expected = None
    if expected is not None:
        target_xyz, _normal_xy = expected
        return _pack(target_xyz, 0.0)

    label_path = meta.get("live_label_path")
    if label_path and getattr(scene, "stage", None) is not None:
        bbox = scene._live_bbox(str(label_path))
        if bbox is not None:
            (x, y), zmin, zmax = bbox
            return _pack([x, y, 0.5 * (zmin + zmax)], max(zmax - zmin, 0.0))
    target_key = meta.get("live_target_key")
    if target_key:
        (x, y), zmin, zmax = scene.object_bbox(str(target_key))
        return _pack([x, y, 0.5 * (zmin + zmax)], max(zmax - zmin, 0.0))
    return None


def _refresh_dynamic_look_entity(scene: SceneState, key: str) -> None:
    ent = scene.entities.get(key)
    if ent is None:
        return
    meta = ent.meta or {}
    try:
        expected = _expected_attach_label_pose(scene, meta)
    except Exception as exc:  # noqa: BLE001
        print(f"[lookat] witness {key}: cannot refresh expected stand ({exc})")
        return
    if expected is None:
        return
    target_xyz, normal_xy = expected
    stand_gap = float(meta.get("witness_standoff_m") or 1.0)
    ent.xy = (float(target_xyz[0]), float(target_xyz[1]))
    ent.z = float(target_xyz[2])
    ent.stand_gap = stand_gap
    ent.approach_yaw = gait.yaw_for_dir(-normal_xy[0], -normal_xy[1])
    stand_xy = (
        float(target_xyz[0] + normal_xy[0] * stand_gap),
        float(target_xyz[1] + normal_xy[1] * stand_gap),
    )
    task = getattr(scene, "behavior_task", None) or {}
    task.setdefault("cue_stands", {})[key] = (
        stand_xy[0],
        stand_xy[1],
        float(ent.approach_yaw),
        float(normal_xy[0]),
        float(normal_xy[1]),
    )


def _refresh_attach_witnesses(scene: SceneState, child_key: str) -> None:
    for witness_key in (getattr(scene, "attach_witness_keys", None) or {}).get(
        child_key, []
    ):
        _refresh_dynamic_look_entity(scene, str(witness_key))


# Adjustments a wearer makes to read something, one per way of failing to read it. Each is a
# CANDIDATE evaluated with the same predicate that judged the original pose, so nothing is applied
# on faith: the verdict names the degree of freedom, the search over that ONE degree of freedom
# confirms it. Bounded and small, because a person does not survey a room to read a sign.
_LOOK_SQUATS = (0.5, 1.0, 1.5, 2.0)  # crouch depths, shallowest first
_LOOK_LEAN_FRACS = (
    0.2,
    0.4,
    0.6,
    0.8,
)  # 8, 16, 24, 32 degrees with the standard 40-degree lean
_LOOK_STEPS_M = (0.35, 0.7, 1.05)  # how far to shuffle, nearest first
_LOOK_COMFORTABLE_LABEL_PX = 110
_LOOK_MAX_OPTIONS = 4
_LOOK_GEOMETRIC_OPTIONS = 24
_LOOK_POSE_COLLISION_CHECKS = 8
_LOOK_VISIBILITY_CANDIDATES = 12
_LOOK_DETAIL_ZOOM_HEADROOM = 1.1
_LOOK_AZIMUTH_OFFSETS_DEG = (
    0.0,
    -30.0,
    30.0,
    -60.0,
    60.0,
    -90.0,
    90.0,
    -120.0,
    120.0,
    -150.0,
    150.0,
    180.0,
)


def _look_required_detail_px(label, meta):
    import label_visibility as lv

    if label.glyph_height_m is not None:
        return int(meta.get("look_min_glyph_px") or lv.MIN_GLYPH_PX)
    if meta.get("label_pos") is not None:
        return int(meta.get("look_target_px") or _LOOK_COMFORTABLE_LABEL_PX)
    return lv.MIN_LABEL_PX


def _look_zoom_target_px(label, meta):
    import label_visibility as lv

    if label.glyph_height_m is not None:
        return int(meta.get("look_target_glyph_px") or lv.TARGET_GLYPH_PX)
    return _look_required_detail_px(label, meta)


def _look_planning_detail_px(label, meta):
    """Keep headroom between an isolated candidate and the replayed camera trajectory."""
    return max(
        _look_required_detail_px(label, meta),
        _look_zoom_target_px(label, meta),
    )


def _look_zoom_span(label, target_px):
    """Return a text-readable zoom span without cropping the document."""
    import egocentric_camera as ego_camera
    import label_visibility as lv

    glyph_height = label.glyph_height_m
    if glyph_height is None or target_px <= 0:
        return label.span_m
    config = ego_camera.DEFAULT_CAMERA_CONFIG
    obliquity_scale = math.cos(math.radians(lv.MAX_OFF_NORMAL_DEG))
    text_span = float(
        glyph_height
        * config.zoom_target_frame_fraction
        * lv.DELIVERED_IMAGE_W
        * obliquity_scale
        / (float(target_px) * _LOOK_DETAIL_ZOOM_HEADROOM)
    )
    if label.size_m is None:
        return max(text_span, float(label.span_m))
    width, height = label.size_m
    u0, v0, u1, v1 = label.content_box_uv or (0.0, 0.0, 1.0, 1.0)
    visible_width = float(width) * (float(u1) - float(u0))
    visible_height = float(height) * (float(v1) - float(v0))
    frame_fill = 0.85
    width_span = visible_width * config.zoom_target_frame_fraction / frame_fill
    height_span = (
        visible_height
        * config.zoom_target_frame_fraction
        * lv.DELIVERED_IMAGE_W
        / (frame_fill * lv.DELIVERED_IMAGE_H)
    )
    return max(text_span, width_span, height_span)


def _look_pose_components(rig, st, squat, lean_frac, cache):
    import head_mount

    key = (float(squat), float(lean_frac))
    cached = cache.get(key)
    if cached is not None:
        return cached
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    pose = _manip(rig, arm_rest, 0.0, lean_frac, squat)
    z = _pin_z(pose, st.floor_z)
    _head_rot, head = head_mount.head_pose(pose, 0.0, (0.0, 0.0), z)
    spine = head_mount.spine3_frame(pose, 0.0, (0.0, 0.0), z)[0]
    cached = (pose, z, head, spine)
    cache[key] = cached
    return cached


def _look_pose_at(
    rig,
    scene,
    st,
    label,
    xy,
    yaw,
    squat,
    boxes,
    lean_frac=0.0,
    min_px=None,
    pose_cache=None,
):
    """Assess a candidate stand+posture without moving anything. Returns (assessment, pose156)."""
    import egocentric_camera as ego_camera
    import head_mount
    import label_visibility as lv

    cache = {} if pose_cache is None else pose_cache
    pose, _z, local_head, local_spine = _look_pose_components(
        rig, st, squat, lean_frac, cache
    )
    _stand, _stand_z, local_stand_head, _stand_spine = _look_pose_components(
        rig, st, 0.0, 0.0, cache
    )
    yaw_rotation = head_mount.yaw_matrix(float(yaw))
    offset = np.array([float(xy[0]), float(xy[1]), 0.0])
    head = yaw_rotation @ local_head + offset
    spine = yaw_rotation @ local_spine
    stand_head = yaw_rotation @ local_stand_head + offset
    camera_state = ego_camera.CameraState(standing_head_z=float(stand_head[2]))
    required_px = lv.MIN_GLYPH_PX if min_px is None else int(min_px)
    zoom_target_px = (
        max(required_px, lv.TARGET_GLYPH_PX)
        if label.glyph_height_m is not None
        else required_px
    )
    zoom_span = _look_zoom_span(label, zoom_target_px)
    base_frame = ego_camera.CameraFrame(
        head_world=head,
        body_forward=wpp._body_forward(float(yaw)),
        mode="gaze",
        dt_seconds=1.0 / 12.0,
        focus_world=label.pos,
        focus_visible=False,
        requested_zoom=0.0,
        zoom_span=zoom_span,
    )
    framing = ego_camera.SceneFraming(floor_z=float(st.floor_z))
    base_output = ego_camera.step_camera(
        base_frame, camera_state, framing=framing
    ).output
    probe = lv.assess(
        np.asarray(base_output.eye),
        spine,
        label,
        boxes,
        image_w=(lv.DELIVERED_IMAGE_W if label.glyph_height_m is not None else 960),
        image_h=(lv.DELIVERED_IMAGE_H if label.glyph_height_m is not None else 760),
        min_px=lv.MIN_LABEL_PX if min_px is None else int(min_px),
        min_glyph_px=lv.MIN_GLYPH_PX if min_px is None else int(min_px),
        focal_mm=float(base_output.focal_mm),
        camera_look=np.asarray(base_output.look),
        enforce_torso_limits=False,
    )
    zoom_on = (
        label.glyph_height_m is not None and os.environ.get("ISAAC_HEAD_MOUNT") != "1"
    )
    camera_frame = replace(
        base_frame,
        focus_visible=not bool(probe.occluder),
        requested_zoom=1.0 if zoom_on else 0.0,
    )
    camera_output = ego_camera.step_camera(
        camera_frame, camera_state, framing=framing
    ).output
    return lv.assess(
        np.asarray(camera_output.eye),
        spine,
        label,
        boxes,
        image_w=(lv.DELIVERED_IMAGE_W if label.glyph_height_m is not None else 960),
        image_h=(lv.DELIVERED_IMAGE_H if label.glyph_height_m is not None else 760),
        min_px=lv.MIN_LABEL_PX if min_px is None else int(min_px),
        min_glyph_px=lv.MIN_GLYPH_PX if min_px is None else int(min_px),
        focal_mm=float(camera_output.focal_mm),
        camera_look=np.asarray(camera_output.look),
        enforce_torso_limits=False,
    ), pose


def _readable_distance_band(
    dz: float,
    span_m: float,
    min_px: int,
    image_w: int = 960,
    pitch_margin_deg: float | None = None,
    focal_mm: float | None = None,
) -> tuple[float, float]:
    """The horizontal distances from which a label ``dz`` below the eye is BOTH inside the neck's
    pitch cone and still big enough to read. Returns (nearest, furthest); empty when lo > hi.

    Solved, not searched. Pitch sets the near end -- at ``d`` the label sits ``atan(dz/d)`` below the
    eye line, so the cone is entered at ``d >= dz / tan(limit)``. Apparent size sets the far end --
    the label subtends ``min_px`` at ``d <= span / (2 tan(px/2 * rad_per_px))``. A person reading a
    low shelf does exactly this: crouch (shrinks dz, pulling the near end in) and stand back a bit.
    """
    import label_visibility as lv

    limit = lv.MAX_PITCH_DOWN_DEG if dz >= 0.0 else lv.MAX_PITCH_UP_DEG
    margin = limit * 0.1 if pitch_margin_deg is None else pitch_margin_deg
    lo = abs(dz) / math.tan(math.radians(max(1.0, limit - margin)))
    hfov_deg = lv.HFOV_DEG
    if focal_mm is not None:
        hfov_deg = math.degrees(
            2.0 * math.atan(lv.HORIZONTAL_APERTURE_MM / (2.0 * focal_mm))
        )
    rad_per_px = math.radians(hfov_deg) / float(image_w)
    hi = (
        span_m / (2.0 * math.tan(0.5 * min_px * rad_per_px))
        if span_m > 1e-6
        else lv.FAR_M
    )
    return lo, min(hi, lv.FAR_M)


def _look_ring_candidates(label_xy, outward, distance, squat, lean_frac=0.0):
    lx, ly = label_xy
    ux, uy = outward
    out = []
    for offset_deg in _LOOK_AZIMUTH_OFFSETS_DEG:
        angle = math.radians(offset_deg)
        ca, sa = math.cos(angle), math.sin(angle)
        dx, dy = ux * ca - uy * sa, ux * sa + uy * ca
        out.append(((lx + dx * distance, ly + dy * distance), squat, lean_frac))
    return out


def _look_distance_candidates(
    d_cur, dz, label, min_px=None, focal_mm=None, eye_forward_m=0.0
):
    import label_visibility as lv

    min_px = lv.MIN_LABEL_PX if min_px is None else int(min_px)
    values = [d_cur]
    face = label.face
    top_label = face is not None and float(np.asarray(face)[2]) > 0.25 and dz > 0.0
    image_w = lv.DELIVERED_IMAGE_W if label.glyph_height_m is not None else 960
    lo, hi = _readable_distance_band(
        dz,
        label.readable_span_m,
        min_px,
        image_w=image_w,
        pitch_margin_deg=1.0 if top_label else None,
        focal_mm=focal_mm,
    )
    if top_label:
        hi = min(
            hi,
            abs(dz) * math.tan(math.radians(lv.MAX_OFF_NORMAL_DEG - 1.0)),
        )
    if lo <= hi:
        span_d = hi - lo
        values.extend(
            [min(max(d_cur - eye_forward_m, lo), hi) + eye_forward_m]
            + [
                lo + span_d * fraction + eye_forward_m
                for fraction in (0.08, 0.38, 0.68, 0.95)
            ]
        )
    return tuple(
        dict.fromkeys(distance for distance in values if 0.05 < distance <= 3.0)
    )


def _look_candidates(
    verdict,
    st,
    label,
    face_yaw,
    eye_z_for,
    min_px=None,
    focal_mm=None,
    eye_forward_m=0.0,
):
    """The adjustments that fix THIS deficiency, best first. ``eye_z_for(squat)`` gives the eye
    height at a crouch depth, which is the only thing the caller knows and this does not.

    Pure geometry off the verdict -- no verb, object or scene is consulted, so a new scene gets the
    same behaviour without anyone adding a case for it.
    """
    import label_visibility as lv
    import nav_planner

    x, y = float(st.root_xy[0]), float(st.root_xy[1])
    lx, ly, lz = (float(v) for v in label.pos)
    d_cur = math.hypot(lx - x, ly - y)
    # Unit vector from the label out towards the wearer: standing off means moving along it.
    ux, uy = (
        ((x - lx) / d_cur, (y - ly) / d_cur)
        if d_cur > 1e-6
        else nav_planner.forward_xy(face_yaw)
    )
    out: list[tuple[tuple[float, float], float, float]] = []

    if verdict in (
        lv.TOO_LOW,
        lv.TOO_HIGH,
        lv.TOO_NEAR,
        lv.TOO_SMALL,
        lv.OFF_NORMAL,
        lv.ROTATED,
        lv.OUT_OF_FRAME,
    ):
        postures = (0.0,) + _LOOK_SQUATS
        top_label = label.face is not None and float(np.asarray(label.face)[2]) > 0.25
        if top_label:
            out.extend(((x, y), 0.0, lean) for lean in _LOOK_LEAN_FRACS)
        out.extend(((x, y), squat, 0.0) for squat in postures)
        if top_label:
            out.extend(
                ((x, y), squat, lean)
                for squat in _LOOK_SQUATS[:2]
                for lean in _LOOK_LEAN_FRACS
            )
        for squat in postures:
            out.extend(_look_ring_candidates((lx, ly), (ux, uy), d_cur, squat))
        for squat in postures:
            distances = _look_distance_candidates(
                d_cur,
                eye_z_for(squat) - lz,
                label,
                min_px,
                focal_mm,
                eye_forward_m,
            )
            for distance in distances:
                if abs(distance - d_cur) <= 1e-6:
                    continue
                out.extend(_look_ring_candidates((lx, ly), (ux, uy), distance, squat))
                if top_label:
                    for lean_frac in (0.4, 0.8):
                        out.extend(
                            _look_ring_candidates(
                                (lx, ly),
                                (ux, uy),
                                distance,
                                squat,
                                lean_frac=lean_frac,
                            )
                        )
    if verdict == lv.OCCLUDED:
        # Nothing about height or distance clears a shelf edge; stepping out from behind it does.
        px, py = -uy, ux
        out += [
            ((x + px * s * d, y + py * s * d), 0.0, 0.0)
            for d in _LOOK_STEPS_M
            for s in (1.0, -1.0)
        ]
        for d in (0.6, 0.9, 1.2):
            out.extend(_look_ring_candidates((lx, ly), (ux, uy), d, 0.0))
    # OFF_TO_SIDE needs no candidate: the look beat turns the body to face the target anyway, and
    # the assessment below is taken at that final facing.
    return out


def _look_facing(label, xy, fallback_yaw):
    direction = np.asarray(label.pos[:2], dtype=np.float64) - np.asarray(
        xy, dtype=np.float64
    )
    if float(np.linalg.norm(direction)) <= 1e-6:
        return fallback_yaw
    return gait.yaw_for_dir(direction[0], direction[1])


def _readable_view_options(
    rig,
    scene,
    st,
    label,
    boxes,
    verdict,
    eye_z_for,
    min_px,
    pose_cache=None,
    minimum_squat: float = 0.0,
):
    grid = getattr(scene, "nav_grid", None)
    reachable = None
    if grid is not None:
        import nav_grid

        reachable = nav_grid.flood_free(grid, grid.world_to_cell(*st.root_xy))
    geometric_options = []
    face_yaw = _look_facing(label, st.root_xy, st.root_yaw)
    import egocentric_camera as ego_camera

    config = ego_camera.DEFAULT_CAMERA_CONFIG
    max_focal_mm = None
    eye_forward_m = config.manip_eye_forward
    if label.glyph_height_m is not None and os.environ.get("ISAAC_HEAD_MOUNT") != "1":
        max_focal_mm = config.base_focal_mm * config.max_focal_multiple
    for order, (xy, squat, lean_frac) in enumerate(
        _look_candidates(
            verdict,
            st,
            label,
            face_yaw,
            eye_z_for,
            min_px,
            max_focal_mm,
            eye_forward_m,
        )
    ):
        squat = max(float(squat), float(minimum_squat))
        yaw = _look_facing(label, xy, st.root_yaw)
        moved = math.hypot(xy[0] - st.root_xy[0], xy[1] - st.root_xy[1])
        if moved > 1e-3 and grid is not None:
            i, j = grid.world_to_cell(*xy)
            if not grid.is_free(i, j) or not bool(reachable[j, i]):
                continue
        cand, pose = _look_pose_at(
            rig,
            scene,
            st,
            label,
            xy,
            yaw,
            squat,
            (),
            lean_frac,
            min_px,
            pose_cache,
        )
        if not cand.ok:
            continue
        detail_px = cand.glyph_px if label.glyph_height_m is not None else cand.px
        geometric_options.append(
            (order, detail_px, xy, yaw, squat, lean_frac, moved, pose, cand)
        )
        if len(geometric_options) >= _LOOK_GEOMETRIC_OPTIONS:
            break
    ranked = sorted(
        geometric_options,
        key=lambda option: (
            abs(option[8].text_roll_deg),
            -option[1],
            option[6],
            option[4] + option[5],
            option[0],
        ),
    )
    options = []
    pose_checks = 0
    visibility_checks = 0
    for option in ranked:
        _order, _detail_px, xy, yaw, _squat, _lean, _moved, pose, _cand = option
        if pose_checks < _LOOK_POSE_COLLISION_CHECKS:
            pose_checks += 1
            clear = body_clear_fn(scene, xy, yaw, st.floor_z)
            body_clear = clear is None or clear(pose, _pin_z(pose, st.floor_z))
        else:
            body_clear = False
        if not body_clear:
            continue
        if visibility_checks >= _LOOK_VISIBILITY_CANDIDATES:
            break
        visibility_checks += 1
        visible, _pose = _look_pose_at(
            rig,
            scene,
            st,
            label,
            xy,
            yaw,
            _squat,
            boxes,
            _lean,
            min_px,
            pose_cache,
        )
        if visible.ok:
            options.append(
                (
                    _order,
                    _detail_px,
                    xy,
                    yaw,
                    _squat,
                    _lean,
                    _moved,
                    pose,
                    visible,
                )
            )
        if len(options) >= _LOOK_MAX_OPTIONS:
            break
    return options


def _apply_readable_view_option(
    rig,
    g,
    scene,
    st,
    goal,
    sd_world,
    option,
    here,
    min_px,
    *,
    preserve_crouch: bool = False,
):
    _order, detail_px, xy, yaw, squat, lean_frac, moved, pose, assessment = option
    if moved > 1e-3:
        try:
            frames = produce_navigate(
                rig,
                g,
                scene,
                st,
                Goal(
                    Primitive.NAVIGATE_TO,
                    target_xy=tuple(xy),
                    target_yaw=yaw,
                    meta=(
                        {"_crouch_owner": st.crouch_owner}
                        if preserve_crouch
                        else {}
                    ),
                ),
                sd_world,
            )
        except PrimitiveFailure:
            return None
    else:
        frames = []
    if squat > 1e-6 or lean_frac > 1e-6:
        frames += _look_crouch(rig, st, pose, yaw)
        if preserve_crouch:
            _set_crouch_hold(st, st.crouch_owner, squat, pose)
    posture = []
    if lean_frac > 1e-6:
        posture.append(f"lean {math.degrees(_LEAN_RAD * lean_frac):.0f}deg")
    if squat > 1e-6:
        posture.append(f"crouch {squat:.1f}")
    print(
        f"[lookat] {goal.obj_key}: {here.verdict} "
        f"({(here.glyph_px or here.px):.1f}px detail, pitch {here.pitch_deg:+.0f}deg"
        f", {here.dist_m:.2f} m) -> "
        + (" + ".join(posture) if posture else f"move {moved:.2f} m")
        + f" -> readable ({detail_px:.1f}px, target {min_px}px, "
        + f"roll {assessment.text_roll_deg:+.0f}deg)"
    )
    return frames, (pose if squat > 1e-6 or lean_frac > 1e-6 else None)


def _look_texture_path(scene, key: str | None, meta: dict) -> str | None:
    for field in ("label_texture", "label_png", "decal_png"):
        value = meta.get(field)
        if value:
            return str(value)
    if not key:
        return None
    for field in ("board_labels", "label_pngs"):
        mapping = getattr(scene, field, None) or {}
        value = mapping.get(key)
        if value:
            return str(value)
    return None


def _look_label_at_rest(scene, key, ent, target, span, bbox, meta):
    import label_visibility as lv

    if meta.get("look_region") is not None:
        return lv.Label(np.asarray(target, dtype=np.float64), float(span))
    label = lv.label_from_meta(
        meta,
        bbox,
        texture_path=_look_texture_path(scene, key, meta),
    )
    if label is None:
        return lv.Label(np.asarray(target, dtype=np.float64), float(span))
    if ent is not None and ent.role is Role.GRASPABLE:
        return _look_grabbable_label(scene, key, ent, bbox, label)
    return label


def _look_rest_frame_object(scene, key, ent, st):
    """Pin a labelled graspable to the same rest transform used for its gaze target."""
    obj = _obj_carry(st)
    meta = getattr(ent, "meta", None) or {}
    if (
        ent is None
        or ent.role is not Role.GRASPABLE
        or meta.get("label_local") is None
        or not getattr(ent, "prim_path", None)
    ):
        return obj, None
    try:
        rest_transform = np.asarray(scene.object_rest_mat(key), dtype=np.float64)
    except (AttributeError, KeyError, TypeError, ValueError):
        return obj, None
    if rest_transform.shape != (4, 4):
        return obj, None
    return ("pose", rest_transform), ent.prim_path


def _apply_legacy_ask12802_report_reading_distance(scene, key, ent) -> None:
    """Restore the reviewed close-held report view for the ASK-12802 replay."""
    report_mode = (
        os.environ.get(
            "ISAAC_LEGACY_ASK12802_REPORT_READING_DISTANCE", "inherit"
        )
        or "inherit"
    ).strip()
    if report_mode == "off":
        return
    if report_mode == "on":
        enabled = True
    elif report_mode == "inherit":
        enabled = os.environ.get("ISAAC_LEGACY_RUNTIME_DOOR_APPROACH") == "1"
    else:
        enabled = False
    if not enabled:
        return
    if str(key) != "item_1":
        return
    task = (getattr(scene, "behavior_task", None) or {}).get("task")
    if task != "ASK-12802" or ent is None:
        return
    meta = getattr(ent, "meta", None)
    if not isinstance(meta, dict) or meta.get("look_held_distance_m") is not None:
        return
    import label_visibility as lv

    meta["look_held_distance_m"] = lv.HELD_NEAR_M


def _seek_readable_view(
    rig,
    g,
    scene,
    st,
    goal,
    ent,
    target,
    span,
    sd_world,
    *,
    fixed_observer: bool = False,
):
    """Make the label readable, or say why it could not be.

    Returns ``(frames, look_pose_or_None, inspection_complete)``. A readable current view needs no
    remediation unless the command immediately follows an explicit Grab of a labelled object; that
    sequence always performs the promised label-facing lift. Otherwise a free-handed wearer lifts
    an unreadable graspable label, while fixed or unreachable labels use the crouch/move search.

    THE SHAPE IS DELIBERATE: assess -> the verdict names the deficiency -> try only the adjustments
    that fix THAT deficiency -> re-assess before committing. It is the geometric cousin of
    habitat-lab's ``ObjectViewLocation``, where a viewpoint is kept because the target was measured
    to be visible from it, not because a formula said it should be.

    Everything is decided from ``label_pos``/``label_span``/``label_face`` (or the bounding box when
    the scene records none), so this is one code path for every scene and every target.
    """
    import label_visibility as lv

    label_meta = getattr(ent, "meta", None) or {}
    look_region = label_meta.get("look_region") or {}
    fixed_interior = (
        fixed_observer and str(look_region.get("kind") or "").lower() == "interior"
    )
    if fixed_interior and not _interior_region_observer_is_authorized(
        ent, st.root_xy
    ):
        raise PrimitiveFailure(
            PrimitiveError.UNREADABLE_LABEL,
            f"{goal.obj_key}: did not reach its authored interior observer stand",
        )
    boxes = _label_occluder_boxes(
        scene,
        goal.obj_key,
        ignore_target_collision=fixed_interior,
    )
    bbox = None
    try:
        bbox = scene.object_bbox(goal.obj_key) if goal.obj_key else None
    except Exception:  # noqa: BLE001
        bbox = None
    label = _look_label_at_rest(
        scene, goal.obj_key, ent, target, span, bbox, label_meta
    )
    min_px = _look_planning_detail_px(label, label_meta)
    preserve_crouch = _crouch_owner_for_target(scene, st, goal.obj_key) is not None
    minimum_squat = (
        _active_crouch_fraction(scene, st, goal.obj_key)
        if preserve_crouch
        else 0.0
    )

    pose_cache = {}
    here, _pose = _look_pose_at(
        rig,
        scene,
        st,
        label,
        st.root_xy,
        _look_facing(label, st.root_xy, st.root_yaw),
        minimum_squat,
        boxes,
        min_px=min_px,
        pose_cache=pose_cache,
    )
    readable_at_rest = here.ok or here.verdict == lv.NO_LABEL
    inspect_after_release = bool((goal.meta or {}).get("_placed_before_look_at"))
    if readable_at_rest and not inspect_after_release:
        return [], None, False

    if fixed_observer:
        detail_px = here.glyph_px if label.glyph_height_m is not None else here.px
        raise PrimitiveFailure(
            PrimitiveError.UNREADABLE_LABEL,
            f"{goal.obj_key}: explicit region observer is {here.verdict}; "
            f"detail={detail_px:.1f}px/{min_px}px, pitch={here.pitch_deg:+.1f}deg, "
            f"distance={here.dist_m:.2f}m, roll={here.text_roll_deg:+.1f}deg"
            + (f", occluder={here.occluder}" if here.occluder else ""),
        )

    can_lift = (
        _dynamic_look_target(scene, ent) is None
        and ent is not None
        and ent.role is Role.GRASPABLE
        and getattr(ent, "prim_path", None)
        and label_meta.get("label_pos") is not None
        and not bool(label_meta.get("inspect_in_place"))
        and st.held is None
    )
    if readable_at_rest and not can_lift:
        return [], None, False

    if can_lift:
        try:
            return (
                _look_at_inspect_grabbable(
                    rig,
                    scene,
                    st,
                    goal,
                    ent,
                    sd_world,
                    label=label,
                    min_detail_px=min_px,
                ),
                None,
                True,
            )
        except PrimitiveFailure as exc:
            print(
                f"[lookat] {goal.obj_key}: cannot bring its unreadable label to the face "
                f"({exc.detail or exc.reason}); trying a crouched view"
            )

    import egocentric_camera as ego_camera

    _standing_pose, _standing_z, standing_head, _standing_spine = _look_pose_components(
        rig, st, 0.0, 0.0, pose_cache
    )
    standing_head_z = float(standing_head[2])

    def eye_z_for(squat: float) -> float:
        _pose, _z, head, _spine = _look_pose_components(rig, st, squat, 0.0, pose_cache)
        raw_eye_z = float(head[2]) + ego_camera.DEFAULT_CAMERA_CONFIG.manip_eye_lift
        floor_eye_z = float(st.floor_z) + (
            ego_camera.DEFAULT_CAMERA_CONFIG.crouch_eye_fraction
            * (standing_head_z - float(st.floor_z))
        )
        return max(raw_eye_z, floor_eye_z)

    options = _readable_view_options(
        rig,
        scene,
        st,
        label,
        boxes,
        here.verdict,
        eye_z_for,
        min_px,
        pose_cache,
        minimum_squat=minimum_squat,
    )
    for option in options:
        result = _apply_readable_view_option(
            rig,
            g,
            scene,
            st,
            goal,
            sd_world,
            option,
            here,
            min_px,
            preserve_crouch=preserve_crouch,
        )
        if result is not None:
            frames, pose = result
            return frames, pose, False

    detail_px = here.glyph_px if label.glyph_height_m is not None else here.px
    detail = (
        f"{goal.obj_key}: {here.verdict}; detail={detail_px:.1f}px/"
        f"{min_px}px, pitch={here.pitch_deg:+.1f}deg, distance={here.dist_m:.2f}m, "
        f"roll={here.text_roll_deg:+.1f}deg"
        + (f", occluder={here.occluder}" if here.occluder else "")
    )
    if label_meta.get("label_pos") is not None:
        raise PrimitiveFailure(PrimitiveError.UNREADABLE_LABEL, detail)
    print(f"[lookat][UNREADABLE] {detail}")
    return [], None, False


def _look_crouch(rig, st, crouched, yaw) -> list:
    """Ease from the current pose into the readable posture, feet planted."""
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    stand = st.last_pose156
    if stand is None:
        stand = _manip(rig, arm_rest, 0.0, 0.0)
    out = []
    for i in range(_N_ARRIVE):
        pose = gait.blend156(stand, crouched, wpp._smoothstep((i + 1) / _N_ARRIVE))
        out.append(
            wpp.Frame(
                pose,
                st.root_xy,
                float(yaw),
                _pin_z(pose, st.floor_z),
                fpv_mode="look",
                obj=_obj_carry(st),
                seg="look",
            )
        )
    st.last_pose156 = crouched
    st.root_yaw = float(yaw)
    st.walk_end_yaw = float(yaw)
    return out


def _put_down_held_target_before_look_at(
    rig,
    g: gait.Gait,
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    sd_world=None,
) -> list[wpp.Frame]:
    held = st.held
    if held is None or str(held.get("obj_key")) != str(goal.obj_key):
        return []
    surface = held.get("source_surface")
    if not surface or surface not in scene.entities:
        raise PrimitiveFailure(
            PrimitiveError.PLACE_UNSTABLE,
            f"LOOKAT {goal.obj_key!r} cannot put down the held object without a known support",
        )
    print(f"[lookat] {goal.obj_key}: put down on {surface} before reading")
    return produce_place(
        rig,
        g,
        scene,
        st,
        Goal(
            Primitive.PLACE,
            obj_key=goal.obj_key,
            meta={"surface": str(surface), "_automatic_look_at_release": True},
        ),
        sd_world=sd_world,
    )


def _front_drum_inspection_goal(
    scene: SceneState, st: SchedState, target_key: str | None
) -> Goal | None:
    owner = _crouch_owner_for_target(scene, st, target_key)
    if owner is None:
        return None
    try:
        entity = scene.resolve(owner)
    except (KeyError, ValueError):
        return None
    meta = getattr(entity, "meta", None) or {}
    if meta.get("access_mode") != "front_drum":
        return None
    geometry = _front_access_geometry(
        entity,
        standoff_override=_FRONT_DRUM_INSPECTION_STANDOFF_M,
    )
    stand = preferred_openable_stand(
        scene,
        owner,
        standoff_override=_FRONT_DRUM_INSPECTION_STANDOFF_M,
    )
    if geometry is None or stand is None:
        return None
    center = geometry[0]
    delta = center[:2] - np.asarray(stand, dtype=np.float64)
    yaw = (
        gait.yaw_for_dir(float(delta[0]), float(delta[1]))
        if float(np.linalg.norm(delta)) > 1e-9
        else float(st.root_yaw)
    )
    return Goal(
        Primitive.NAVIGATE_TO,
        target_xy=(float(stand[0]), float(stand[1])),
        target_yaw=float(yaw),
        target_key=target_key,
        meta={
            "_crouch_owner": owner,
            "_front_drum_inspection": True,
        },
    )


def _unlabeled_object_look_target(scene: SceneState, key: str, entity):
    (x, y), z_min, z_max = scene.object_bbox(key)
    span = max(float(z_max) - float(z_min), 0.0)
    foot_half = getattr(entity, "foot_half", None) if entity is not None else None
    if foot_half is not None and len(foot_half) >= 2:
        span = max(span, 2.0 * abs(float(foot_half[0])), 2.0 * abs(float(foot_half[1])))
    return (
        np.array([float(x), float(y), 0.5 * (float(z_min) + float(z_max))]),
        span,
    )


def _front_drum_inspection_frames(
    rig,
    g: gait.Gait,
    scene: SceneState,
    st: SchedState,
    target_key: str | None,
    sd_world=None,
) -> list[wpp.Frame]:
    inspection_goal = _front_drum_inspection_goal(scene, st, target_key)
    if inspection_goal is None or math.dist(
        tuple(st.root_xy), tuple(inspection_goal.target_xy)
    ) <= 0.04:
        return []
    try:
        return produce_navigate(
            rig,
            g,
            scene,
            st,
            inspection_goal,
            sd_world=sd_world,
        )
    except PrimitiveFailure as exc:
        print(
            f"[lookat] {target_key}: front-drum inspection stand unavailable "
            f"({exc.detail or exc.reason}); retaining the open stance"
        )
        return []


def _interior_region_observer_is_authorized(ent, actual_xy) -> bool:
    region = (getattr(ent, "meta", None) or {}).get("look_region") or {}
    authored = np.asarray(region.get("observer_stand_xy"), dtype=np.float64).reshape(
        -1
    )
    actual = np.asarray(actual_xy, dtype=np.float64).reshape(-1)
    return (
        authored.shape == (2,)
        and actual.shape == (2,)
        and np.all(np.isfinite(authored))
        and np.all(np.isfinite(actual))
        and float(np.linalg.norm(actual - authored)) <= 0.15 + 1e-9
    )


def _look_region_observer_frames(
    rig,
    g: gait.Gait,
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    ent,
    target,
    sd_world=None,
) -> list[wpp.Frame]:
    region = (getattr(ent, "meta", None) or {}).get("look_region") or {}
    observer_stand = region.get("observer_stand_xy")
    if observer_stand is None:
        return []
    stand = np.asarray(observer_stand, dtype=np.float64)
    if stand.shape != (2,) or not np.all(np.isfinite(stand)):
        raise PrimitiveFailure(
            PrimitiveError.PATH_BLOCKED,
            f"LOOK_AT region for {goal.obj_key!r} has an invalid observer stand",
        )
    if str(region.get("kind") or "").lower() == "interior":
        if goal.obj_key not in (getattr(scene, "open_objs", None) or set()):
            raise PrimitiveFailure(
                PrimitiveError.UNREADABLE_LABEL,
                f"LOOK_AT interior region for {goal.obj_key!r} requires an open owner",
            )
    target_delta = np.asarray(target, dtype=np.float64)[:2] - stand
    yaw = (
        gait.yaw_for_dir(float(target_delta[0]), float(target_delta[1]))
        if float(np.linalg.norm(target_delta)) > 1e-9
        else float(st.root_yaw)
    )
    frames = produce_navigate(
        rig,
        g,
        scene,
        st,
        Goal(
            Primitive.NAVIGATE_TO,
            target_key=goal.obj_key,
            target_xy=(float(stand[0]), float(stand[1])),
            target_yaw=float(yaw),
            meta={
                "_look_region_observer": goal.obj_key,
                "_navigation_fixed_gaze": tuple(
                    float(value) for value in np.asarray(target, dtype=np.float64)[:3]
                ),
            },
        ),
        sd_world=sd_world,
    )
    print(
        f"[lookat] {goal.obj_key}: explicit region observer "
        f"({stand[0]:.3f}, {stand[1]:.3f}), yaw {yaw:.1f}deg"
    )
    return frames


def produce_look_at(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Look at an entity, remediating a measured unreadable label only when necessary.

    A held target is first put back on its known supporting surface, clearing the ordinary carry
    grip that can cover its label. It is then handled exactly like any other resting object: a
    readable view stays in place, while an unreadable graspable uses the dedicated label-facing
    inspection pickup and returns to rest afterward. Fixed or unreachable labels use the existing
    crouch/move search.

    ``scene.look_zoom`` controls the final camera hold; it does not decide whether physical
    remediation is required."""
    crouch_owner = _crouch_owner_for_target(scene, st, goal.obj_key)
    keep_crouched = crouch_owner is not None
    zoom_on = bool(getattr(scene, "look_zoom", False))
    _ent = scene.entities.get(goal.obj_key)
    _apply_legacy_ask12802_report_reading_distance(scene, goal.obj_key, _ent)
    placed_before_look_at = st.held is not None and str(st.held.get("obj_key")) == str(
        goal.obj_key
    )
    prefix_frames = _put_down_held_target_before_look_at(
        rig, g, scene, st, goal, sd_world=sd_world
    )
    if placed_before_look_at:
        goal = replace(
            goal,
            meta={**(goal.meta or {}), "_placed_before_look_at": True},
        )
    prefix_frames.extend(
        _front_drum_inspection_frames(
            rig,
            g,
            scene,
            st,
            goal.obj_key,
            sd_world=sd_world,
        )
    )
    _dynamic_target = _dynamic_look_target(scene, _ent)
    _look_meta = getattr(_ent, "meta", None) or {}

    # NON-GRABBABLE sign / look_zoom off: the eye stays at the head; aim (and, under look_zoom, HOLD) on the
    # target from the front stand.
    _gaze_locked = False
    force_zoom = False
    fixed_look_frames = False
    if _dynamic_target is not None:
        target, span, _gaze_locked, force_zoom, fixed_look_frames = _dynamic_target
        zoom_on = zoom_on or force_zoom
    else:
        _meta = _ent.meta if _ent is not None else {}
        _lpos = _meta.get("label_pos")
        if _lpos is not None:
            target = np.array(
                [float(_lpos[0]), float(_lpos[1]), float(_lpos[2])], dtype=np.float64
            )
            span = float(_meta.get("label_span") or 0.0)
            _gaze_locked = True
        elif zoom_on:
            target, span = _unlabeled_object_look_target(
                scene, goal.obj_key, _ent
            )
        else:
            target = scene.object_pos(goal.obj_key).astype(np.float64)
            span = 0.0
    fixed_observer = bool(
        _dynamic_target is not None
        and (_look_meta.get("look_region") or {}).get("observer_stand_xy") is not None
    )
    if fixed_observer:
        prefix_frames.extend(
            _look_region_observer_frames(
                rig,
                g,
                scene,
                st,
                goal,
                _ent,
                target,
                sd_world=sd_world,
            )
        )
    # READ IT FROM WHERE WE STAND. There is no "square-on reposition" that walks to a point 0.90 m
    # out on the placard's outward normal (``label_face``) so the sign is viewed HEAD-ON.
    #
    # Neither reference implementation has such a constraint.
    # habitat-lab's ``place_agent_at_dist_from_pos`` takes ANY navigable point within
    # ``spawn_max_dist_to_obj`` (2.0 m) and its arrival test is ``angle_to_obj < turn_thresh`` --
    # the angle to the OBJECT, never to a surface normal. IsaacLab's ``UniformPose2dCommand``
    # (``simple_heading``) will even accept arriving BACKWARDS to save a turn.
    #
    # Squaring up is expensive. In one measured case the wearer was already 1.03 m from the
    # placard, facing it to within 5.4 deg, reading it 23.5 deg off-normal -- inside a +/-40 deg
    # legibility fan. To square that up it walked 0.41 m along a heading 84 deg away from where it
    # had to end up, which pointed straight at a wall it stopped 31 cm from, and spent 75 deg of body
    # rotation (23 deg out, 52 deg back) getting there. That out-and-back is what a viewer sees as
    # "the head turns to the wall and then turns back".
    #
    # An oblique read is cheap: 23.5 deg foreshortens the text by cos = 0.92.
    #
    # What DOES happen instead: the wearer checks whether the label is actually readable from here
    # and, if it is not, makes the ONE adjustment the deficiency calls for. See _seek_readable_view.
    seek_kwargs = {"fixed_observer": True} if fixed_observer else {}
    frames, base, inspection_complete = _seek_readable_view(
        rig,
        g,
        scene,
        st,
        goal,
        _ent,
        target,
        span,
        sd_world,
        **seek_kwargs,
    )
    frames = prefix_frames + frames
    if inspection_complete:
        return frames
    frame_zoom_span = span
    focus_visibility_verified = False
    if _ent is not None:
        try:
            _bbox = scene.object_bbox(goal.obj_key)
        except (AttributeError, KeyError, TypeError, ValueError):
            _bbox = None
        label = _look_label_at_rest(
            scene,
            goal.obj_key,
            _ent,
            target,
            span,
            _bbox,
            _look_meta,
        )
        if label is not None:
            target = np.asarray(label.pos, dtype=np.float64)
            span = float(label.span_m)
            zoom_on = zoom_on or label.glyph_height_m is not None
            focus_visibility_verified = _gaze_locked
            frame_zoom_span = _look_zoom_span(
                label, _look_zoom_target_px(label, _look_meta)
            )
    d = target[:2] - np.asarray(st.root_xy, dtype=np.float64)
    face = (
        gait.yaw_for_dir(d[0], d[1]) if float(np.linalg.norm(d)) > 1e-6 else st.root_yaw
    )
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    if base is None:
        base = (
            st.last_pose156.copy()
            if keep_crouched and st.last_pose156 is not None
            else _manip(rig, arm_rest, 0.0, 0.0)
        )
    obj, obj_path = _look_rest_frame_object(scene, goal.obj_key, _ent, st)
    # look_zoom off -> the plain 16-frame turn. look_zoom on -> a short turn+zoom-in, then a
    # HOLD >=24 frames STILL on the sign (>=2 s @12fps to read it), then a zoom-out.
    #
    # The turn length is a floor, not a fixed count: a fixed 8 or 16 frames WHATEVER the angle would
    # make a big turn run faster (90 deg in 8 frames is 135 deg/s, 180 deg is 270). Because the
    # wearer reads from where it stands, the residual turn can be anything, so it is rate-limited to
    # the same ``_TURN_DEG_PER_FRAME`` the standalone [TurnLeft]/[TurnRight] verbs use.
    hold_on = zoom_on and span > 1e-3 and not fixed_look_frames
    n_turn, n_hold, n_out = (8, _N_LOOK_HOLD, 6) if hold_on else (16, 0, 0)
    n_turn = max(
        n_turn, int(math.ceil(abs(_ang_delta(st.root_yaw, face)) / _TURN_DEG_PER_FRAME))
    )
    n = n_turn + n_hold + n_out
    for i in range(n):
        # ``seg`` NAMES the phase: only the HOLD frames have the sign square-on and at full zoom, and a
        # downstream frame sampler that cannot tell them from the zoom-out tail delivers the tail. With
        # look_zoom off n_hold is 0, so every frame keeps the plain ``look`` tag.
        _seg = "look"
        if i < n_turn:
            s = wpp._smoothstep((i + 1) / n_turn)
            if hold_on:
                zoom = wpp._smoothstep((i + 1) / n_turn)
            else:
                zoom = 1.0 if force_zoom else None
        elif i < n_turn + n_hold:
            s, zoom, _seg = 1.0, 1.0, "look_hold"
        else:
            s, zoom = 1.0, wpp._smoothstep((n - i) / n_out)
        yaw = st.root_yaw + _ang_delta(st.root_yaw, face) * s
        frames.append(
            wpp.Frame(
                base,
                st.root_xy,
                yaw,
                _pin_z(base, st.floor_z),
                fpv_mode="look",
                obj=obj,
                obj_path=obj_path,
                seg=_seg,
                gaze=target,
                zoom=zoom,
                zoom_span=(frame_zoom_span if zoom_on else None),
                gaze_locked=_gaze_locked,
                focus_visibility_verified=focus_visibility_verified,
            )
        )
    st.root_yaw = face
    # Keep the two headings in step. ``_arrive_beats`` sweeps walk_end_yaw -> root_yaw, so leaving
    # the stale WALK heading here makes the next manipulation re-turn the body through an arc it
    # already covered during the look. ``produce_turn`` and sit do the same.
    st.walk_end_yaw = face
    st.last_pose156 = base
    if keep_crouched:
        _set_crouch_hold(
            st,
            crouch_owner,
            max(_FRONT_APPLIANCE_SQUAT, st.crouch_fraction),
            base,
        )
    return frames


def produce_squat(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    """Lower the body (and the egocentric eye) into a crouch and rise; feet stay planted."""
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    stand = _manip(rig, arm_rest, 0.0, 0.0)
    squat = _squat_pose(rig, arm_rest)
    obj = _obj_carry(st)
    frames = []
    n = 12
    for seq in ((stand, squat), (squat, stand)):
        for i in range(n):
            s = wpp._smoothstep((i + 1) / n)
            pose = gait.blend156(seq[0], seq[1], s)
            frames.append(
                wpp.Frame(
                    pose,
                    st.root_xy,
                    st.root_yaw,
                    _pin_z(pose, st.floor_z),
                    fpv_mode="look",
                    obj=obj,
                    seg="squat",
                )
            )
    st.last_pose156 = stand
    return frames


def produce_policy_halt(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
):
    return []


def produce_wait(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """Hold a visible idle beat so a task request or reply has time to be delivered."""
    del g, sd_world
    base = st.last_pose156
    if base is None:
        base = _manip(rig, getattr(rig, "arm_rest", rig.arm_grasp), 0.0, 0.0)
    root_z = st.seated_z if st.seated_z is not None else _pin_z(base, st.floor_z)
    rhp = st.held.get("rhp") if st.held is not None else None
    duration_s = float((goal.meta or {}).get("duration_s", WAIT_DEFAULT_S))
    if not math.isfinite(duration_s) or not WAIT_MIN_S <= duration_s <= WAIT_MAX_S:
        raise PrimitiveFailure(
            PrimitiveError.PRECONDITION_UNMET,
            f"wait duration must be in [{WAIT_MIN_S:g}, {WAIT_MAX_S:g}] seconds",
        )
    n_frames = max(1, int(round(duration_s * 12.0)))
    frames = [
        wpp.Frame(
            base,
            st.root_xy,
            st.root_yaw,
            root_z,
            rhp=rhp,
            fpv_mode="look",
            obj=_obj_carry(st),
            seg="wait",
        )
        for _ in range(n_frames)
    ]
    st.last_pose156 = base
    if st.held is not None:
        _stamp_obj_path(frames, scene, st.held["obj_key"])
        _stamp_carried(frames, st.held.get("carried"))
    return frames


# =====================================================================================================
# EGOCENTRIC LOCOMOTION: the only producers with no target entity. Every other verb names a thing and
# lets the planner find a path to it; these two are expressed in the BODY frame, so they are the verbs
# you can give a human who cannot see the room and therefore cannot name anything in it.
# =====================================================================================================
def _walk_obstacle_boxes(scene) -> list:
    """The AABBs a WALKING body has to miss: everything with a footprint that also stands up off the floor.

    Both sources are consulted because either can be empty depending on how the scene was built --
    ``solid_boxes()`` returns the shared collision map when the scene runs unified geometry and the
    per-entity footprint pillars otherwise, and a scene can have entities registered before the oracle
    exists. The height filter is what keeps a floor slab (top == floor_z) from reading as a wall."""
    boxes = list(scene.solid_boxes()) if hasattr(scene, "solid_boxes") else []
    seen = {b[0] for b in boxes}
    for e in getattr(scene, "entities", {}).values():
        if e.foot_half is None or e.name in seen:
            continue
        cx, cy = e.foot_xy if e.foot_xy is not None else e.xy
        top = e.top_z if e.top_z is not None else e.z
        boxes.append(
            (
                e.name,
                np.array([cx - e.foot_half[0], cy - e.foot_half[1], scene.floor_z]),
                np.array([cx + e.foot_half[0], cy + e.foot_half[1], top]),
            )
        )
    lo = float(getattr(scene, "floor_z", 0.0)) + _FWD_MIN_OBSTACLE_H
    return [b for b in boxes if float(b[2][2]) > lo]


def _forward_clearance(scene, xy, yaw: float, dist: float) -> float:
    """Metres of the straight forward ray that are actually walkable, from the best oracle this scene has.

    Nav grid first (precise triangle raster + walls, human-radius inflated); else the furniture AABBs the
    reach gate and the render nudge already use; else -- a scene with no geometry registered at all -- the
    full distance, which is the same contract every other verb has in a gridless scene."""
    import nav_planner

    grid = getattr(scene, "nav_grid", None)
    if grid is not None:
        clear = nav_planner.free_run_ahead(grid, xy, yaw, dist)
        src = "nav_grid"
    else:
        boxes = _walk_obstacle_boxes(scene)
        clear = (
            nav_planner.box_run_ahead(boxes, xy, yaw, dist, radius=_FWD_BODY_R)
            if boxes
            else float(dist)
        )
        src = f"{len(boxes)} box(es)" if boxes else "NO oracle (nothing registered)"
    print(f"[forward] want {dist:.2f} m, {clear:.2f} m clear ahead (oracle: {src})")
    return clear


def produce_forward(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """[Forward]: stride ``meta['dist_m']`` metres along the CURRENT facing. No target, no path search.

    This is the one verb that can walk into something, because nothing chose its destination for it. The
    segment is measured against the scene's collision oracle BEFORE a single frame exists, and a segment
    that does not clear raises PATH_BLOCKED carrying how far WAS clear. No frames are emitted: the human
    does not move, the episode scores a real failure, and no geometry is ever crossed. The walk itself is
    produce_navigate's proven straight leg (depart beat, distance-phased gait, carry threading)."""
    st.ego_locked = (
        True  # BEFORE the clearance check: a FAILED leg must not leave a teleport open
    )
    dist = float((goal.meta or {}).get("dist_m", FORWARD_STEP_M))
    if dist <= 0.0:
        raise PrimitiveFailure(
            PrimitiveError.PATH_BLOCKED,
            f"forward distance must be > 0 m, got {dist:.2f}",
        )
    clear = _forward_clearance(scene, st.root_xy, st.root_yaw, dist)
    if clear < dist - 1e-3:
        raise PrimitiveFailure(
            PrimitiveError.PATH_BLOCKED,
            f"forward {dist:.2f} m from ({st.root_xy[0]:.2f},{st.root_xy[1]:.2f}) "
            f"yaw={st.root_yaw:.0f}: only {clear:.2f} m clear ahead (wall/furniture)",
        )
    import nav_planner

    fx, fy = nav_planner.forward_xy(st.root_yaw)
    leg = Goal(
        Primitive.NAVIGATE_TO,
        target_xy=(st.root_xy[0] + fx * dist, st.root_xy[1] + fy * dist),
        target_yaw=st.root_yaw,
    )
    return produce_navigate(
        rig, g, scene, st, leg, sd_world=sd_world, _force_straight=True
    )


def produce_turn(
    rig, g: gait.Gait, scene: SceneState, st: SchedState, goal: Goal, sd_world=None
) -> list[wpp.Frame]:
    """[TurnLeft] / [TurnRight] / [TurnAround]: rotate ``meta['turn_deg']`` degrees in place (+ = left).

    The feet stay planted -- ``st.root_xy`` is never touched -- so the only thing that changes is the
    heading, and with it the egocentric camera. Delegates to ``nav_planner.turn_in_place`` (the same
    producer the A* polyline corners use) because a turn CANNOT be a zero-length NAVIGATE_TO:
    ``gait.plan_walk`` sets yaw=0.0 for a leg shorter than 1e-6 m, which would snap the body to
    world-north instead of rotating it."""
    import nav_planner

    st.ego_locked = True
    dyaw = float((goal.meta or {}).get("turn_deg", TURN_DEG))
    base = _manip(rig, getattr(rig, "arm_rest", rig.arm_grasp), 0.0, 0.0)
    frames = nav_planner.turn_in_place(
        g,
        st,
        st.root_yaw,
        dyaw,
        st.root_xy,
        arm_ov=st.held["grasp_arm"] if st.held else None,
        rhp=st.held["rhp"] if st.held else None,
        obj=_obj_carry(st),
        deg_per_frame=_TURN_DEG_PER_FRAME,
        base_pose=base,
        blend_from=st.last_pose156,
    )
    st.root_yaw = (st.root_yaw + dyaw + 180.0) % 360.0 - 180.0
    st.walk_end_yaw = st.root_yaw
    st.last_pose156 = base
    if st.held is not None:
        _stamp_obj_path(frames, scene, st.held["obj_key"])
        _stamp_carried(frames, st.held.get("carried"))
    return frames


_DISPATCH = {
    Primitive.NAVIGATE_TO: produce_navigate,
    Primitive.FORWARD: produce_forward,
    Primitive.TURN: produce_turn,
    Primitive.PICK: produce_pick,
    Primitive.FLIP: produce_flip,
    Primitive.PLACE: produce_place,
    Primitive.PLACE_INSIDE: produce_place_inside,
    Primitive.ATTACH: produce_attach,
    Primitive.DETACH: produce_detach,
    Primitive.RAISE: produce_raise,
    Primitive.PUSH: produce_push,
    Primitive.PULL: produce_pull,
    Primitive.RELEASE: produce_release,
    Primitive.FALL_TO_FLOOR: produce_fall_to_floor,
    Primitive.PLACE_ON_FLOOR: produce_place_on_floor,
    Primitive.OPEN: produce_open,
    Primitive.CLOSE: produce_close,
    Primitive.OPENDOOR: produce_open_door,
    Primitive.CLOSEDOOR: produce_close_door,
    Primitive.TURN_ON: produce_turn_on,
    Primitive.TURN_OFF: produce_turn_off,
    Primitive.SIT: produce_sit,
    Primitive.STANDUP: produce_standup,
    Primitive.CUT: produce_cut,
    Primitive.STIR: produce_stir,
    Primitive.POUR: produce_pour,
    Primitive.WIPE: produce_wipe,
    Primitive.COOK: produce_cook,
    Primitive.USE_COMPUTER: produce_use_computer,
    Primitive.ANSWER_PHONE: produce_answer_phone,
    Primitive.LOOK_AROUND: produce_look_around,
    Primitive.LOOK_SCAN: produce_look_scan,
    Primitive.LOOK_AT: produce_look_at,
    Primitive.SQUAT: produce_squat,
    Primitive.CONFIRM: produce_policy_halt,
    Primitive.WAIT: produce_wait,
}


def validate_capability_registry() -> None:
    """Fail loudly if the declared action inventory drifts from executable dispatch."""
    declared = runtime_capabilities.action_by_primitive(include_unavailable=True)
    dispatched = {
        primitive.value: producer.__name__ for primitive, producer in _DISPATCH.items()
    }
    if set(declared) != set(dispatched):
        raise RuntimeError(
            f"capability registry/dispatch mismatch: declared={sorted(declared)} "
            f"dispatched={sorted(dispatched)}"
        )
    wrong = {
        name: (declared[name].executor, dispatched[name])
        for name in declared
        if declared[name].executor != dispatched[name]
    }
    if wrong:
        raise RuntimeError(f"capability executor mismatch: {wrong}")


# =====================================================================================================
# The executor: dispatch a command program to producers, concatenating frames + per-goal reports.
# =====================================================================================================
def make_state(
    rig, scene: SceneState, start_xy: tuple[float, float] | None = None
) -> SchedState:
    """Initial scheduler state: body standing at the start XY with feet on the floor, object at rest.

    ``start_xy`` defaults to ``scene.start_xy``. The INIT
    non-penetration layer (``init_validation.place_start``) passes a VALIDATED, free-cell start here so a
    randomly-generated RL scene begins the human at a guaranteed non-penetrating pose instead of the
    hardcoded literal -- the literal becomes a fallback hint, used when the arg is omitted."""
    V0, _ = core.pose_body(rig, rig.arm_grasp, (0.0, 0.0), 0.0)
    stand_z = scene.floor_z - float(V0[:, 2].min())
    obj_rest = None
    graspables = [e for e in scene.entities.values() if e.role is Role.GRASPABLE]
    if graspables:
        # ORIGIN (frozen rest), not object_pos's bbox base -- see produce_pick: the first graspable is driven
        # during the initial walk, so sourcing the base would sink it from frame 0 (center-pivot objects).
        obj_rest = tuple(float(x) for x in scene.object_origin(graspables[0].name))
    return SchedState(
        root_xy=tuple(start_xy) if start_xy is not None else scene.start_xy,
        root_yaw=0.0,
        stand_z=stand_z,
        floor_z=scene.floor_z,
        obj_rest_xyz=obj_rest,
    )


def step_one(
    rig,
    g: gait.Gait,
    scene: SceneState,
    st: SchedState,
    goal: Goal,
    sd_world=None,
) -> tuple[list, PrimitiveError, str]:
    """Run ONE goal through its producer -> (frames, reason, detail).

    The single-command building block shared by ``execute`` (batch) and the online ``SemanticEnv``. On an
    unsupported primitive returns ([], NOT_ARTICULATED, ""); on a ``PrimitiveFailure`` returns
    ([], reason, detail); on success returns (frames, OK, "") and advances ``st.last_seg`` -- exactly the
    per-goal work of the ``execute`` loop, so the CALLER owns frame accumulation + logging
    (``execute`` keeps its own producer-None pre-check and its own log lines)."""
    producer = _DISPATCH.get(goal.prim)
    if producer is None:
        return [], PrimitiveError.NOT_ARTICULATED, ""
    try:
        held_at_start = st.held
        frames = producer(rig, g, scene, st, goal, sd_world=sd_world)
        _stamp_held_carry_frames(frames, scene, held_at_start)
        running = tuple(sorted(scene.running))
        frames = [
            frame
            if getattr(frame, "running_devices", None) is not None
            else replace(frame, running_devices=running)
            for frame in frames
        ]
        if frames:
            st.last_seg = frames[-1].seg
        if hasattr(scene, "note_body_state"):
            scene.note_body_state(st)
        if hasattr(scene, "note_motion"):
            scene.note_motion()
        return frames, PrimitiveError.OK, ""
    except PrimitiveFailure as e:
        return [], e.reason, e.detail


def execute(
    rig,
    g: gait.Gait,
    scene: SceneState,
    goals: list[Goal],
    sd_world=None,
    st: SchedState | None = None,
) -> tuple[wpp.Motion, list[tuple[Goal, PrimitiveError]]]:
    """Run a command program -> (Motion of concatenated frames, per-goal (goal, PrimitiveError) reports).

    A failing primitive reports its structured reason and emits NO frames (the assistant/RL loop reads
    the failure and recovers), then execution continues with the next command."""
    st = st or make_state(rig, scene)
    # Naturalness (gated per-scene): tell warp_arm_to_seat to apply the reach-exact elbow-down swivel so
    # every reach (pick/place/open/close/contact) reads human. Default (no scene.natural_arm) leaves
    # rig.natural_arm False -> the side-grip pose returned as-is. Reach is unaffected.
    if getattr(scene, "natural_arm", False):
        rig.natural_arm = True
        print(
            "[exec] natural_arm ON: reach-exact elbow-down swivel enabled in warp_arm_to_seat"
        )
    motion = wpp.Motion()
    reports: list[tuple[Goal, PrimitiveError]] = []
    # Door handling is a RUNTIME concern in the render goal loop (episode_runtime.run_render), CLOSED-first,
    # keyed off each navigate's runtime-resolved stand -- NOT a compile-time pre-pass: the navigates are
    # UNRESOLVED (target_xy=None) at this point, so a pre-pass could not detect a door crossing. execute()
    # stays the offline/batch path (no physics handle -> no runtime stand pick); door scenes render through
    # run_render.
    for goal in goals:
        producer = _DISPATCH.get(goal.prim)
        if producer is None:
            reports.append((goal, PrimitiveError.NOT_ARTICULATED))
            print(f"[exec] {goal.prim.value}: UNSUPPORTED -> skipped (no frames)")
            continue
        frames, reason, detail = step_one(rig, g, scene, st, goal, sd_world=sd_world)
        if reason is PrimitiveError.OK:
            motion.frames.extend(frames)
            reports.append((goal, PrimitiveError.OK))
            print(
                f"[exec] {goal.prim.value}: OK (+{len(frames)} frames, total {len(motion.frames)})"
            )
        else:
            reports.append((goal, reason))
            print(
                f"[exec] {goal.prim.value}: FAILED {reason.value} ({detail}) -> no frames"
            )
    return motion, reports


# =====================================================================================================
# Example scene + program (mirrors humanoid_motion, but command-driven).
# =====================================================================================================
def demo_scene() -> SceneState:
    ss = SceneState(floor_z=0.0, start_xy=(0.0, 1.05))
    ss.register("cup", Role.GRASPABLE, xy=(0.0, -1.36), z=0.75, approach_yaw=0.0)
    ss.register("deskA", Role.SURFACE, xy=(0.0, -1.36), z=0.75, half=(1.4, 0.4))
    ss.register(
        "deskB",
        Role.SURFACE,
        xy=(0.0, 1.36),
        z=0.75,
        approach_yaw=180.0,
        half=(1.4, 0.4),
    )
    ss.declare_place("cup", "deskA")  # initial world state: cup rests on DeskA
    return ss


# The demo program as a VirtualHome command STRING (the command grammar drives the executor).
_DEMO_PROGRAM = "[Walk] <cup> (1), [Grab] <cup> (1), [Walk] <deskB> (1), [Put] <cup> (1) <deskB> (1)"


def demo_program(scene: SceneState) -> list[Goal]:
    """Compile the VirtualHome command string into a Goal sequence (names grounded via the scene, the
    implicit navigate auto-inserted). Exercises the grammar layer end-to-end into the executor."""
    return command_grammar.compile_program(_DEMO_PROGRAM, scene)


# =====================================================================================================
# ``--step3`` scene + offline check: open/close (upper cabinet + lower drawer) and sit (chair + sofa) --
# each action on TWO different-position/height objects, showing the reach + coupling generalize.
# =====================================================================================================
def step3_scene() -> SceneState:
    """Two openables at different heights (reach up vs down) + two seats at different heights/positions."""
    ss = SceneState(floor_z=0.0, start_xy=(0.0, 1.4))
    ss.register(
        "cabinet",
        Role.OPENABLE,
        xy=(-0.55, -0.5),
        z=1.15,
        approach_yaw=0.0,
        axis=(0.0, -1.0, 0.0),
        travel=0.32,
    )  # HIGH handle -> reach up
    ss.register(
        "drawer",
        Role.OPENABLE,
        xy=(0.55, -0.5),
        z=0.78,
        approach_yaw=0.0,
        axis=(0.0, -1.0, 0.0),
        travel=0.28,
    )  # LOWER handle -> reach down/forward
    # approach_yaw=0: walk up FACING the seat; produce_sit turns 180 to sit (back to backrest).
    ss.register(
        "chair", Role.SEAT, xy=(-0.55, 0.55), z=0.45, approach_yaw=0.0, stand_gap=0.5
    )
    ss.register(
        "sofa", Role.SEAT, xy=(0.55, 0.55), z=0.38, approach_yaw=0.0, stand_gap=0.6
    )
    return ss


def run_dry3(rig, g: gait.Gait) -> int:
    from isaac_goal import check_goal, is_open, is_sitting, Pred

    ok = True

    def _chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    # open/close two different-height openables in ONE episode (st reused -> continuous).
    ss = step3_scene()
    st = make_state(rig, ss)
    m1, r1 = execute(
        rig,
        g,
        ss,
        command_grammar.compile_program(
            "[Walk] <cabinet> (1), [Open] <cabinet> (1), [Walk] <drawer> (1), [Open] <drawer> (1)",
            ss,
        ),
        st=st,
    )
    _chk(
        all(x is PrimitiveError.OK for _, x in r1),
        "open cabinet(up)+drawer(down): all primitives OK",
    )
    _chk(
        check_goal([is_open("cabinet"), is_open("drawer")], ss.relations()),
        "both OPEN after open",
    )
    m2, r2 = execute(
        rig,
        g,
        ss,
        command_grammar.compile_program(
            "[Close] <cabinet> (1), [Close] <drawer> (1)", ss
        ),
        st=st,
    )
    _chk(all(x is PrimitiveError.OK for _, x in r2), "close both: all primitives OK")
    _chk(not any(p[0] is Pred.OPEN for p in ss.relations()), "neither OPEN after close")

    # sit on two different seats (chair then sofa), standing up between.
    ss2 = step3_scene()
    st2 = make_state(rig, ss2)
    ms, rs = execute(
        rig,
        g,
        ss2,
        command_grammar.compile_program("[Walk] <chair> (1), [Sit] <chair> (1)", ss2),
        st=st2,
    )
    _chk(all(x is PrimitiveError.OK for _, x in rs), "sit chair: all primitives OK")
    _chk(check_goal([is_sitting("chair")], ss2.relations()), "SITTING chair")
    ms2, rs2 = execute(
        rig,
        g,
        ss2,
        command_grammar.compile_program(
            "[StandUp], [Walk] <sofa> (1), [Sit] <sofa> (1)", ss2
        ),
        st=st2,
    )
    _chk(
        all(x is PrimitiveError.OK for _, x in rs2),
        "standup + sit sofa: all primitives OK",
    )
    _chk(
        check_goal([is_sitting("sofa")], ss2.relations())
        and not check_goal([is_sitting("chair")], ss2.relations()),
        "SITTING sofa, not chair",
    )

    # feet (standing frames only) + continuity WITHIN each episode (skip the cross-episode teleport).
    ep1, ep2 = m1.frames + m2.frames, ms.frames + ms2.frames
    J1 = np.asarray([wpp.realize(rig, fr)[1] for fr in ep1])
    J2 = np.asarray([wpp.realize(rig, fr)[1] for fr in ep2])

    def _foot_range(J, frames):
        idx = [i for i, fr in enumerate(frames) if fr.seg not in ("sit", "standup")]
        zmin = J[idx][:, list(_FOOT_J), 2].min(
            axis=1
        )  # lowest (planted) foot per frame
        return float(zmin.min()), float(zmin.max())

    f1, f2 = _foot_range(J1, ep1), _foot_range(J2, ep2)
    fmin, fmax = min(f1[0], f2[0]), max(f1[1], f2[1])
    jmax = 0.0
    for J, frames, tag in (
        (J1, ep1, "ep1"),
        (J2, ep2, "ep2"),
    ):  # diagnostics: worst foot + worst jump seg
        idx = [i for i, fr in enumerate(frames) if fr.seg not in ("sit", "standup")]
        z = J[idx][:, list(_FOOT_J), 2].min(axis=1)
        wf = idx[int(np.argmax(np.abs(z)))]
        jj = np.linalg.norm(np.diff(J[:, core.HEAD_J], axis=0), axis=1)
        ji = int(np.argmax(jj)) if len(jj) else 0
        jmax = max(jmax, float(jj.max()) if len(jj) else 0.0)
        print(
            f"    [{tag}] worst-foot {z[np.argmax(np.abs(z))] * 1000:.0f}mm @{frames[wf].seg}; "
            f"worst-jump {jj[ji] * 1000:.0f}mm @{frames[ji].seg}->{frames[min(ji + 1, len(frames) - 1)].seg}"
        )
    _chk(
        abs(fmin) < 0.06 and abs(fmax) < 0.10,
        f"feet on floor (standing) min={fmin * 1000:.0f} max={fmax * 1000:.0f}mm",
    )
    _chk(jmax < 0.35, f"continuity max head-jump within-episode={jmax * 1000:.0f}mm")
    try:  # sagittal skeleton storyboard: confirm the coarse poses (reach up/down + seated) visually
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        parents = core.H.PARENTS
        reach_i = [i for i, fr in enumerate(ep1) if fr.seg == "reach"]
        sit_i = [i for i, fr in enumerate(ep2) if fr.seg == "sit"]
        picks = []
        if reach_i:
            picks += [
                (J1[reach_i[len(reach_i) // 4]], "open cabinet (reach up)"),
                (J1[reach_i[3 * len(reach_i) // 4]], "open drawer (reach down)"),
            ]
        if sit_i:
            picks += [
                (J2[sit_i[len(sit_i) // 2 - 1]], "sit chair"),
                (J2[sit_i[-1]], "sit sofa"),
            ]
        fig, axes = plt.subplots(
            1, len(picks), figsize=(3 * len(picks), 4.6), squeeze=False
        )
        for ax, (J, title) in zip(axes[0], picks):
            for j in range(len(parents)):
                p = int(parents[j])
                if p >= 0:
                    ax.plot(
                        [-J[j, 1], -J[p, 1]],
                        [J[j, 2], J[p, 2]],
                        "-",
                        color="#1f5fbf",
                        lw=1.4,
                    )
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_title(title, fontsize=9)
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-0.1, 1.9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            "Step 3: open (up/down) + sit (chair/sofa) — sagittal, forward=right",
            fontsize=10,
        )
        fig.tight_layout()
        out = os.path.abspath(
            os.path.join(_SIMULATOR_DIR, "fpv_outputs", "sem_step3_storyboard.png")
        )
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(f"[sem][dry3] storyboard -> {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[sem][dry3] storyboard skipped ({exc})")

    # HINGE WIRING. This gate is OFFLINE -- a synthetic SceneState through the kinematic executor,
    # no USD stage, and ``pxr`` is not even importable in this process -- so ``episode_runtime``'s
    # ``_place_door`` / ``_place_container_leaves`` / ``_wire_hinge_ops`` are unreachable from here by
    # construction, and no openable added to ``step3_scene()`` would change that: a symbolic openable
    # is not a hinge. A bug in that code can kill every episode in ``setup_episode`` with NO manifest
    # and still pass this gate green. ``hinge_gate`` covers it from the other side: a static check
    # that every module-level function binds the pxr symbols it uses, and a runtime check that wires
    # one leaf TWICE, which reproduces the glazed-pane double-wire case. It adds no frames.
    try:
        hinge_ok, hinge_lines = hinge_gate.check()
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must FAIL, never skip quietly
        hinge_ok, hinge_lines = (
            False,
            [f"  [FAIL] hinge gate raised {type(exc).__name__}: {exc}"],
        )
    print("\n".join(hinge_lines))
    ok = ok and hinge_ok

    n_frames = len(ep1) + len(ep2)
    print(
        f"[sem][dry3] Step-3 open/close/sit = {'PASS' if ok else 'FAIL'} ({n_frames} frames)"
    )
    # Say what was COVERED, not just that it was green. A ``--step3`` PASS is not proof that any
    # change is safe; it is proof about locomotion, sitting, prismatic openables and the hinge op
    # wiring, and about nothing else. The hinge checks are structural rather than kinematic, so they
    # do not change the frame count.
    print(
        "[sem][dry3] COVERED: locomotion + sit/stand + open/close (SYMBOLIC, prismatic) "
        "+ hinge xformOp wiring (episode_runtime._wire_hinge_ops, single AND double wire, "
        "via simulator/physical_world/hinge_gate.py)"
    )
    print(
        "[sem][dry3] NOT COVERED: USD scene build, colliders, physics settle, rendering, "
        "navigation mesh -- this gate never opens a stage"
    )
    return 0 if ok else 2


# =====================================================================================================
# ``--step4`` offline check: the CONTACT verbs (cut/stir/pour/wipe/cook) + CAMERA verbs (look/squat).
# Verifies each executes, the declared state flips (CUT/COOKED/INSIDE), a missing tool fails honestly
# (PRECONDITION_UNMET, zero frames), and feet stay planted. No Isaac.
# =====================================================================================================
def step4_scene() -> SceneState:
    """A counter with a knife (graspable tool), a veg on a board (cut target), and a pot (stir/pour/cook)."""
    ss = SceneState(floor_z=0.0, start_xy=(0.0, 1.4))
    ss.register("knife", Role.GRASPABLE, xy=(0.45, -0.6), z=0.80, approach_yaw=0.0)
    ss.register(
        "board",
        Role.SURFACE,
        xy=(-0.25, -0.6),
        z=0.80,
        approach_yaw=0.0,
        half=(0.3, 0.3),
        foot_xy=(-0.25, -0.6),
        foot_half=(0.3, 0.3),
        top_z=0.80,
        clearance=0.40,
    )
    ss.register(
        "veg",
        Role.GRASPABLE,
        xy=(-0.25, -0.6),
        z=0.84,
        approach_yaw=0.0,
        foot_xy=(-0.25, -0.6),
        foot_half=(0.3, 0.3),
        top_z=0.80,
        clearance=0.40,
    )
    ss.register(
        "pot",
        Role.SURFACE,
        xy=(0.5, -0.6),
        z=0.82,
        approach_yaw=0.0,
        half=(0.15, 0.15),
        foot_xy=(0.5, -0.6),
        foot_half=(0.15, 0.15),
        top_z=0.82,
        clearance=0.40,
    )
    ss.declare_place("knife", "board")
    return ss


def run_dry4(rig, g: gait.Gait) -> int:
    from isaac_goal import check_goal, is_cooked, is_cut, is_off, is_running, Pred

    ok = True

    def _chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    # CUT: grab the knife, walk to the veg, chop -> CUT(veg).
    ss = step4_scene()
    st = make_state(rig, ss)
    m, r = execute(
        rig,
        g,
        ss,
        command_grammar.compile_program(
            "[Walk] <knife> (1), [Grab] <knife> (1), [Walk] <veg> (1), [Cut] <veg> (1) <knife> (1)",
            ss,
        ),
        st=st,
    )
    _chk(
        all(x is PrimitiveError.OK for _, x in r),
        "grab knife + cut veg: all primitives OK",
    )
    _chk(check_goal([is_cut("veg")], ss.relations()), "CUT(veg) declared after cut")

    # COOK (no tool): dwell at the pot -> COOKED(pot).
    ss2 = step4_scene()
    st2 = make_state(rig, ss2)
    _m2, r2 = execute(
        rig,
        g,
        ss2,
        command_grammar.compile_program("[Walk] <pot> (1), [Cook] <pot> (1)", ss2),
        st=st2,
    )
    _chk(all(x is PrimitiveError.OK for _, x in r2), "cook pot: all primitives OK")
    _chk(
        check_goal([is_cooked("pot")], ss2.relations()),
        "COOKED(pot) declared after cook",
    )

    # PRECONDITION: cut with NO tool held -> PRECONDITION_UNMET, no faked success, no frames.
    ss3 = step4_scene()
    st3 = make_state(rig, ss3)
    _m3, r3 = execute(
        rig,
        g,
        ss3,
        command_grammar.compile_program(
            "[Walk] <veg> (1), [Cut] <veg> (1) <knife> (1)", ss3
        ),
        st=st3,
    )
    _chk(
        any(x is PrimitiveError.PRECONDITION_UNMET for _, x in r3),
        "cut without a held tool -> PRECONDITION_UNMET",
    )
    _chk(
        not check_goal([is_cut("veg")], ss3.relations()),
        "CUT(veg) NOT set on unmet precondition",
    )

    # STIR + POUR (tool held): both run; POUR declares INSIDE(source, target).
    ss4 = step4_scene()
    st4 = make_state(rig, ss4)
    m4, r4 = execute(
        rig,
        g,
        ss4,
        command_grammar.compile_program(
            "[Grab] <knife> (1), [Walk] <pot> (1), [Stir] <pot> (1) <knife> (1), [Pour] <pot> (1) <knife> (1)",
            ss4,
        ),
        st=st4,
    )
    _chk(
        all(x is PrimitiveError.OK for _, x in r4),
        "grab + stir + pour: all primitives OK",
    )
    _chk(
        (Pred.INSIDE, "knife", "pot") in ss4.relations(), "INSIDE(knife,pot) after pour"
    )

    # CAMERA verbs: execute + emit frames.
    ss5 = step4_scene()
    st5 = make_state(rig, ss5)
    m5, r5 = execute(
        rig,
        g,
        ss5,
        command_grammar.compile_program(
            "[LookAround], [LookAt] <pot> (1), [Squat]", ss5
        ),
        st=st5,
    )
    _chk(
        all(x is PrimitiveError.OK for _, x in r5),
        "look_around + look_at + squat: all primitives OK",
    )
    _chk(len(m5.frames) > 0, f"camera verbs emitted {len(m5.frames)} frames")

    # TOGGLE: a container can independently expose a press control; the action reaches that control
    # and changes RUNNING without replacing the container's primary role.
    ss6 = SceneState(floor_z=0.0, start_xy=(0.0, 1.4))
    ss6.register(
        "device",
        Role.CONTAINER,
        xy=(0.0, -0.6),
        z=0.0,
        half=(0.3, 0.3),
        height=0.85,
        approach_yaw=0.0,
        foot_xy=(0.0, -0.6),
        foot_half=(0.3, 0.3),
        top_z=0.85,
        clearance=0.40,
    )
    ss6.register_affordance(
        "device",
        Role.TOGGLEABLE,
        xy=(0.0, -0.29),
        z=0.72,
        meta={"press_dir": (0.0, -1.0, 0.0)},
    )
    st6 = make_state(rig, ss6)
    m6, r6 = execute(
        rig,
        g,
        ss6,
        command_grammar.compile_program("[TurnOn] <device> (1)", ss6),
        st=st6,
    )
    _chk(
        all(x is PrimitiveError.OK for _, x in r6), "turn_on device: all primitives OK"
    )
    _chk(
        check_goal([is_running("device")], ss6.relations()),
        "RUNNING(device) after turn_on",
    )
    _chk(
        m6.frames[0].running_devices == ()
        and m6.frames[-1].running_devices == ("device",),
        "visible device state changes on the press frame",
    )
    _m7, r7 = execute(
        rig,
        g,
        ss6,
        command_grammar.compile_program("[TurnOff] <device> (1)", ss6),
        st=st6,
    )
    _chk(
        all(x is PrimitiveError.OK for _, x in r7), "turn_off device: all primitives OK"
    )
    _chk(
        not check_goal([is_running("device")], ss6.relations()),
        "RUNNING(device) cleared after turn_off",
    )
    _chk(
        check_goal([is_off("device")], ss6.relations()),
        "OFF(device) derived after turn_off",
    )

    # Feet planted across the cut episode (lowest foot ~ floor every frame).
    Jc = np.asarray([wpp.realize(rig, fr)[1] for fr in m.frames])
    zlow = Jc[:, list(_FOOT_J), 2].min(axis=1)
    _chk(
        abs(float(zlow.min())) < 0.06 and float(zlow.max()) < 0.15,
        f"cut feet planted low={zlow.min() * 1000:.0f}..{zlow.max() * 1000:.0f}mm",
    )

    print(
        f"[sem][dry4] Step-4 contact/camera verbs = {'PASS' if ok else 'FAIL'} "
        f"(cut {len(m.frames)}f, stir+pour {len(m4.frames)}f, camera {len(m5.frames)}f, "
        f"toggle {len(m6.frames)}f)"
    )
    return 0 if ok else 2


def locomotion_scene() -> SceneState:
    """An empty 6x6 m room with a solid wall 2 m in front of the start, and a nav grid over it.

    Deliberately has NO entities: the point of the four egocentric verbs is that they work in a room the
    human cannot name anything in. The grid is hand-built (as in nav_planner._selftest) so the check runs
    with no Isaac stage -- the wall it contains is the same kind of obstacle build_nav_grid rasters."""
    import nav_grid

    ss = SceneState(floor_z=0.0, start_xy=(0.0, 0.0))
    cell = 0.10
    x0 = y0 = -3.0
    nx = ny = int(round(6.0 / cell))
    occ = np.zeros((ny, nx), dtype=bool)
    jwall = int((-2.0 - y0) / cell)  # a wall spanning the room at y = -2.0
    occ[jwall : jwall + 2, :] = True
    occ[0, :] = occ[-1, :] = (
        True  # room bounds, so a runaway [Forward] cannot leave the grid
    )
    occ[:, 0] = occ[:, -1] = True
    ss.nav_grid = nav_grid.NavGrid(
        x0, y0, cell, nx, ny, occ, nav_grid._two_sided_sdf(occ, cell), inflate=0.10
    )
    return ss


def _loco_probe(rig, g: gait.Gait, scene: SceneState, program: str):
    """Compile + execute a locomotion program, returning (motion, reports, per-goal end pose)."""
    goals = command_grammar.compile_program(program, scene)
    st = make_state(rig, scene)
    motion = wpp.Motion()
    reports, poses = [], []
    for goal in goals:
        frames, reason, detail = step_one(rig, g, scene, st, goal)
        motion.frames.extend(frames)
        reports.append((goal, reason, detail, len(frames)))
        poses.append((tuple(st.root_xy), float(st.root_yaw)))
    return motion, reports, poses


def run_dry_locomotion(rig, g: gait.Gait) -> int:
    """OFFLINE proof of the four egocentric verbs: a turn rotates in place, a forward translates along
    the facing, and a forward into the wall is a NAMED failure that emits nothing."""
    ok = True

    def _chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    scene = locomotion_scene()
    prog = "[Forward] 2, [TurnLeft], [Forward] 2, [TurnRight], [Forward], [TurnAround], [Forward] 3"
    motion, reports, poses = _loco_probe(rig, g, scene, prog)
    for (goal, reason, detail, n), (xy, yaw) in zip(reports, poses):
        tag = (goal.meta or {}).get("verb", goal.prim.value)
        print(
            f"  [{tag:<11}] {reason.value:<12} +{n:>3}f -> xy=({xy[0]:+.2f},{xy[1]:+.2f}) "
            f"yaw={yaw:+7.1f}{('  ' + detail) if detail else ''}"
        )
    _chk(
        all(r[1] is PrimitiveError.OK for r in reports),
        f"all 7 clear locomotion commands OK ({len(motion.frames)} frames total)",
    )
    # [Forward] 2 from the start (yaw 0, facing -Y) translates 2 paces along the facing, nothing else.
    _chk(
        abs(poses[0][0][0]) < 1e-6
        and abs(poses[0][0][1] - (-2 * FORWARD_STEP_M)) < 1e-6
        and abs(poses[0][1]) < 1e-6,
        f"[Forward] 2 translates along the facing -> {poses[0][0]} yaw={poses[0][1]:.1f}",
    )
    # [TurnLeft] rotates IN PLACE: xy identical to the previous command's end, yaw +90 (left = +X at yaw 0).
    _chk(
        poses[1][0] == poses[0][0] and abs(poses[1][1] - 90.0) < 1e-6,
        f"[TurnLeft] in place: xy unchanged {poses[1][0]}, yaw {poses[0][1]:.0f} -> {poses[1][1]:.0f}",
    )
    turn_frames = [fr for fr in motion.frames if fr.seg == "turn"]
    _chk(
        all(abs(fr.xy[0] - poses[0][0][0]) < 1e-9 for fr in turn_frames[:9]),
        f"every turn frame holds the root xy ({len(turn_frames)} turn frames)",
    )
    # After TurnLeft the facing is +X, so [Forward] 2 must move in +X only.
    _chk(
        abs(poses[2][0][0] - 2 * FORWARD_STEP_M) < 1e-6
        and abs(poses[2][0][1] - poses[1][0][1]) < 1e-6,
        f"[Forward] after [TurnLeft] moves along the NEW facing -> {poses[2][0]}",
    )
    _chk(
        abs(poses[3][1]) < 1e-6 and abs(poses[5][1] - (-180.0)) < 1e-6,
        f"[TurnRight] -> {poses[3][1]:.0f} deg, then [TurnAround] -> {poses[5][1]:.0f} deg",
    )
    _chk(
        _turn_frame_counts(motion) == (9, 9, 18),
        f"turn frame counts (90/90/180 deg @ {_TURN_DEG_PER_FRAME:.0f} deg/frame) = "
        f"{_turn_frame_counts(motion)}",
    )
    ok = _loco_blocked_probe(rig, g, scene, _chk) and ok
    _loco_storyboard(motion, reports)
    _loco_body_storyboard(rig, motion)
    print(f"[sem][loco] egocentric locomotion verbs = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


def _turn_frame_counts(motion) -> tuple:
    """Length of each contiguous run of seg=='turn' frames, in order."""
    runs, cur = [], 0
    for fr in motion.frames:
        if fr.seg == "turn":
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return tuple(runs)


def _loco_blocked_probe(rig, g: gait.Gait, scene: SceneState, _chk) -> bool:
    """A [Forward] that would cross the wall must FAIL BY NAME and emit nothing (no teleport, no nudge)."""
    ok = True
    m, reports, poses = _loco_probe(rig, g, scene, "[Forward] 6m")
    goal, reason, detail, n = reports[0]
    print(f"  [blocked   ] {reason.value:<12} +{n:>3}f  {detail}")
    ok = (reason is PrimitiveError.PATH_BLOCKED) and ok
    _chk(
        reason is PrimitiveError.PATH_BLOCKED and n == 0 and not m.frames,
        f"[Forward] 6m into the wall -> {reason.value}, 0 frames emitted",
    )
    _chk(poses[0][0] == (0.0, 0.0), f"the blocked human did not move: {poses[0][0]}")
    _chk(
        "clear ahead" in detail and detail.split("only ")[1].startswith("1."),
        f"the failure NAMES how far was clear: {detail!r}",
    )
    # ...but the legal part of the same approach still works: 1.2 m stops short of the wall at y=-2.
    m2, r2, p2 = _loco_probe(rig, g, scene, "[Forward] 2")
    _chk(
        r2[0][1] is PrimitiveError.OK and len(m2.frames) > 0,
        f"the same heading is still walkable up to the wall ({len(m2.frames)} frames)",
    )
    return ok


def _loco_body_storyboard(rig, motion) -> None:
    """Six TOP-DOWN skeleton snapshots through the program. Seen from above, a [TurnLeft] is the body
    swinging round a fixed pelvis and a [Forward] is the whole skeleton sliding along its own facing --
    the two things the verbs claim to do, in a picture rather than in numbers."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(motion.frames)
        picks = [
            (0, "start"),
            (18, "after [Forward] 2"),
            (23, "mid [TurnLeft]"),
            (27, "after [TurnLeft]"),
            (46, "after [Forward] 2"),
            (85, "after [TurnAround]"),
        ]
        picks = [(min(i, n - 1), t) for i, t in picks]
        parents = core.H.PARENTS
        fig, axes = plt.subplots(
            1, len(picks), figsize=(2.6 * len(picks), 3.2), squeeze=False
        )
        for ax, (i, title) in zip(axes[0], picks):
            fr = motion.frames[i]
            J = wpp.realize(rig, fr)[1]
            for j in range(len(parents)):
                p = int(parents[j])
                if p >= 0:
                    ax.plot(
                        [J[j, 0], J[p, 0]],
                        [J[j, 1], J[p, 1]],
                        "-",
                        color="#1f5fbf",
                        lw=1.2,
                    )
            a = math.radians(fr.yaw)
            ax.arrow(
                fr.xy[0],
                fr.xy[1],
                0.45 * math.sin(a),
                -0.45 * math.cos(a),
                head_width=0.09,
                color="#d24a3a",
                length_includes_head=True,
            )
            ax.set_title(
                f"f{i}: {title}\nxy=({fr.xy[0]:+.2f},{fr.xy[1]:+.2f}) yaw={fr.yaw:+.0f}",
                fontsize=8,
            )
            ax.set_xlim(-0.9, 2.1)
            ax.set_ylim(-2.4, 0.6)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            "egocentric locomotion, TOP-DOWN (red = facing): turns pivot in place, forwards slide along it",
            fontsize=10,
        )
        fig.tight_layout()
        out = os.path.abspath(
            os.path.join(_SIMULATOR_DIR, "fpv_outputs", "locomotion_body_storyboard.png")
        )
        fig.savefig(out, dpi=95)
        plt.close(fig)
        print(f"[sem][loco] body storyboard -> {out}")
    except Exception as exc:  # noqa: BLE001 - a review aid, never a gate
        print(f"[sem][loco] body storyboard skipped ({exc})")


def _loco_storyboard(motion, reports) -> None:
    """Top-down track with heading arrows + the yaw trace, so a person can see the turns and the steps."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xy = np.array([fr.xy for fr in motion.frames])
        yaw = np.array([fr.yaw for fr in motion.frames])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        ax1.plot(xy[:, 0], xy[:, 1], "-", color="#888", lw=1)
        for k in range(0, len(xy), 3):
            a = math.radians(yaw[k])
            ax1.arrow(
                xy[k, 0],
                xy[k, 1],
                0.18 * math.sin(a),
                -0.18 * math.cos(a),
                head_width=0.06,
                color="#1f6fd2",
                length_includes_head=True,
            )
        ax1.axhline(-2.0, color="#b03a2e", lw=3)
        ax1.text(-1.2, -1.92, "wall (blocks [Forward])", color="#b03a2e", fontsize=8)
        ax1.set_title("top-down: body track + heading")
        ax1.set_aspect("equal")
        ax2.plot(yaw, color="#333")
        ax2.set_title("yaw (deg) per frame")
        fig.tight_layout()
        out = os.path.abspath(
            os.path.join(_THIS_DIR, "..", "fpv_outputs", "locomotion_check.png")
        )
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(f"[sem][loco] storyboard -> {out}  ({len(motion.frames)} frames)")
    except Exception as exc:  # noqa: BLE001 - the storyboard is a review aid, never a gate
        print(f"[sem][loco] storyboard skipped ({exc})")


# =====================================================================================================
# Offline dry run: execute the program, realize every frame, verify feet + reach + continuity. No Isaac.
# =====================================================================================================
def run_dry(rig, g: gait.Gait) -> int:
    scene = demo_scene()
    goals = demo_program(scene)
    motion, reports = execute(rig, g, scene, goals, sd_world=None)
    n_ok = sum(1 for _, r in reports if r is PrimitiveError.OK)
    print(
        f"[sem][dry] program: {n_ok}/{len(reports)} primitives OK, motion = {len(motion.frames)} frames"
    )
    if not motion.frames:
        print("[sem][dry] CHECK = FAIL (no frames)")
        return 2

    Js, heads, objs = [], [], []
    for fr in motion.frames:
        _, J, objT = wpp.realize(rig, fr)
        Js.append(J)
        heads.append(J[core.HEAD_J])
        objs.append(objT[:3, 3] if objT is not None else None)
    Js = np.asarray(Js)
    heads = np.asarray(heads)

    foot_min = Js[:, list(_FOOT_J), 2].min(axis=1)
    head_jump = np.linalg.norm(
        np.diff(heads, axis=0), axis=1
    )  # per-frame ego-camera motion
    print(
        f"[sem][dry] foot-floor z: min={foot_min.min() * 1000:.0f}mm max={foot_min.max() * 1000:.0f}mm (want ~0)"
    )
    print(
        f"[sem][dry] head-jump per frame: max={head_jump.max() * 1000:.0f}mm mean={head_jump.mean() * 1000:.0f}mm"
    )
    for ji in np.argsort(head_jump)[-3:][::-1]:
        print(
            f"[sem][dry]   jump {head_jump[ji] * 1000:.0f}mm at f{ji}->{ji + 1} "
            f"seg {motion.frames[ji].seg}->{motion.frames[ji + 1].seg}"
        )

    feet_ok = abs(float(foot_min.min())) < 0.05 and abs(float(foot_min.max())) < 0.08
    # Continuity: no single-frame ego jump larger than a stride step + sway margin (a pop would spike).
    cont_ok = float(head_jump.max()) < 0.22
    prog_ok = n_ok == len(reports)
    # B: symbolic relations + goal check -- the RL/benchmark success signal (physics-agnostic here; the
    # live render/RL path derives `relations()` from PhysX contacts/overlap instead of geometry).
    goal = [isaac_goal.on("cup", "deskB")]
    rel = scene.relations()
    goal_ok = isaac_goal.check_goal(goal, rel)
    print(
        f"[sem][dry] goal ON(cup,deskB)={goal_ok} progress={isaac_goal.goal_progress(goal, rel):.2f} "
        f"relations={sorted(f'{p.value}({a},{b})' for p, a, b in rel)}"
    )
    ok = feet_ok and cont_ok and prog_ok and goal_ok
    print(
        f"[sem][dry] CHECK program={'PASS' if prog_ok else 'FAIL'} feet={'PASS' if feet_ok else 'FAIL'} "
        f"continuity={'PASS' if cont_ok else 'FAIL'} goal={'PASS' if goal_ok else 'FAIL'} -> {'PASS' if ok else 'FAIL'}"
    )
    _dry_storyboard(rig, motion, Js, objs)
    return 0 if ok else 2


def _dry_storyboard(rig, motion, Js, objs) -> None:
    """Sagittal skeleton storyboard at each segment's representative frame (parity with the demo's)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[sem][dry] storyboard skipped ({exc})")
        return
    segs = [fr.seg for fr in motion.frames]
    want = ["walk", "grasp", "lift", "walk", "place", "retract"]
    picks, cursor = [], 0
    # monotonic scan so the 2nd 'walk' comes from the carrying navigate, not the 1st
    for w in want:
        idx = next((k for k in range(cursor, len(segs)) if segs[k] == w), None)
        if idx is None:
            idx = next((k for k, s in enumerate(segs) if s == w), 0)
        picks.append(idx)
        cursor = idx + 1
    parents = core.H.PARENTS
    fig, axes = plt.subplots(
        1, len(picks), figsize=(2.6 * len(picks), 4.4), squeeze=False
    )
    for ax, k in zip(axes[0], picks):
        J = Js[k]
        for j in range(len(parents)):
            p = int(parents[j])
            if p >= 0:
                ax.plot(
                    [-J[j, 1], -J[p, 1]],
                    [J[j, 2], J[p, 2]],
                    "-",
                    color="#1f5fbf",
                    lw=1.4,
                )
        if objs[k] is not None:
            ax.scatter([-objs[k][1]], [objs[k][2]], s=70, marker="s", color="#e0952b")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{segs[k]} f{k}", fontsize=8)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.1, 1.95)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "command-driven: [navigate][pick][navigate][place] (obj=orange, forward=right)",
        fontsize=9,
    )
    fig.tight_layout()
    out = os.path.abspath(
        os.path.join(_SIMULATOR_DIR, "fpv_outputs", "sem_dry_storyboard.png")
    )
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f"[sem][dry] storyboard -> {out}")


# Pickup position on DeskA per object: a DIFFERENT world XY per object so a swap also relocates it.
# The human navigates to a stand _STAND_GAP in front (facing DeskA) and the reach adapts to the position.
_OBJ_PICK_XY = {
    "cup": (0.0, -1.36),  # matches the original demo
    "box": (0.45, -1.42),  # cracker box (WRAP), shifted right + deeper
    "ball": (-0.42, -1.36),  # sphere (round wrap), shifted left
    "pen": (0.30, -1.34),  # large marker -> PINCH family
    "plate": (0.0, -1.40),  # bowl (flat-support)
    "tool": (0.35, -1.40),  # power drill (WRAP)
}
_STAND_GAP = 0.46  # human stands this far in front of the object (matches the cup demo)


def run_render(rig, g: gait.Gait, args, obj_a_xy: tuple[float, float]) -> int:
    """Render a command-driven program by injecting execute() into the demo's Isaac render path.

    Reuses humanoid_motion.run_render (office scene, physics presettle, FPV+TPV cameras, mp4 encode);
    only the motion source is swapped to the executor, and the pickup object + its DeskA position vary."""

    def _build(rig, g, stand_z, obj_a_z, obj_b_z, sd_world, obj_a_xy):
        scene = SceneState(floor_z=0.0, start_xy=wpp.START_XY)
        scene.register(
            "obj",
            Role.GRASPABLE,
            xy=(obj_a_xy[0], obj_a_xy[1]),
            z=obj_a_z,
            approach_yaw=wpp.YAW_A,
        )
        scene.register(
            "deskA",
            Role.SURFACE,
            xy=(obj_a_xy[0], obj_a_xy[1]),
            z=obj_a_z,
            half=(1.4, 0.4),
        )
        scene.register(
            "deskB",
            Role.SURFACE,
            xy=wpp.OBJ_B_XY,
            z=obj_b_z,
            approach_yaw=wpp.YAW_B,
            half=(1.4, 0.4),
        )
        scene.declare_place(
            "obj", "deskA"
        )  # initial world state: object rests on DeskA
        goals = command_grammar.compile_program(
            "[Walk] <obj> (1), [Grab] <obj> (1), [Walk] <deskB> (1), [Put] <obj> (1) <deskB> (1)",
            scene,
        )
        motion, reports = execute(rig, g, scene, goals, sd_world=sd_world)
        n_ok = sum(1 for _, r in reports if r is PrimitiveError.OK)
        print(
            f"[sem] render motion: {n_ok}/{len(reports)} primitives OK, {len(motion.frames)} frames"
        )
        return motion

    return wpp.run_render(rig, g, args, build_fn=_build, obj_a_xy=obj_a_xy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry", action="store_true", help="offline verify (no Isaac / render)"
    )
    ap.add_argument(
        "--obj", default="cup", help="object to pick: cup/box/ball/pen/plate/tool"
    )
    ap.add_argument(
        "--tag", default=None, help="output filename tag (default: sem[_<obj>])"
    )
    ap.add_argument("--frames", type=int, default=0, help="cap total frames (0 = all)")
    ap.add_argument(
        "--step3", action="store_true", help="offline Step-3 check (open/close/sit)"
    )
    ap.add_argument(
        "--step4",
        action="store_true",
        help="offline Step-4 check (cut/stir/pour/wipe/cook/look/squat)",
    )
    ap.add_argument(
        "--locomotion",
        action="store_true",
        help="offline egocentric-locomotion check (Forward/TurnLeft/TurnRight/TurnAround + the "
        "walk-into-a-wall failure)",
    )
    args = ap.parse_args()
    if args.tag is None:
        args.tag = "sem" if args.obj == "cup" else f"sem_{args.obj}"

    rig = core.load_family("side_wrap")
    build, otw = core.load_object_def(args.obj)
    core.set_object(rig, build, otw)
    g = gait.build_gait()

    if args.step3:
        return run_dry3(rig, g)
    if args.step4:
        return run_dry4(rig, g)
    if args.locomotion:
        return run_dry_locomotion(rig, g)
    if args.dry:
        return run_dry(rig, g)
    obj_a_xy = _OBJ_PICK_XY.get(args.obj, (0.0, -1.36))
    return run_render(rig, g, args, obj_a_xy)


if __name__ == "__main__":
    raise SystemExit(main())

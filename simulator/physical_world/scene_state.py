"""Isaac-native scene registry: GROUNDING (command name -> live object position + stand pose) and
symbolic RELATION extraction (ON / INSIDE / HELD / OPEN) for the benchmark / RL loop.

Why this exists: the VirtualHome harness (``check_goal`` / ``resolve_ids``) reads a VH
``EnvironmentGraph`` that Isaac does not have. Isaac gives us the physical truth instead -- PhysX
contact reports, scene-query overlaps, articulation joint state -- so we build a thin symbolic layer
on top. See [[isaac-semantic-executor-plan]].

Two use modes (so it works for RANDOM RL scenes AND offline tests):
  * SPEC (offline / --dry / tests): register entities with explicit world positions.
  * LIVE Isaac (render / RL): register with prim_paths; object positions are read from the live USD
    world bbox, and relations from PhysX (reusing ``physics_human`` contact reports + scene-query
    overlap + articulation DOF).

GROUNDING replaces the executor's hardcoded scene: an object's position is read live, so any command
can reference any object at any (random) position. RELATIONS feed ``isaac_goal.check_goal``.

Import-safe: numpy only at module scope; pxr/omni are imported lazily inside the live paths.

    bash hand/run_isaac.sh simulator.physical_world/scene_state.py   # offline selftest (grounding + geometric relations)
"""

from __future__ import annotations

import copy
import enum
import math
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from isaac_goal import Pred  # noqa: E402  (same dir)


# --- ATTACH / assembly geometry (Task 2). OmniGibson AttachedTo tolerance, copied verbatim. ---
_ATTACH_POS_TOL_M: float = 0.05  # DEFAULT_POSITION_THRESHOLD (m)
_ATTACH_ANG_TOL_RAD: float = math.radians(15.0)  # DEFAULT_ORIENTATION_THRESHOLD (rad)

# --- LIVE ON (``_relations_on_live``). How far the underside of a resting body may sit from the top
# it rests on. The flat term is the one ``_relations_geom`` has used for the offline guard since it
# was written; the fractional term is the tilt slack that ``_relations_inside_live`` learned the hard
# way -- ComputeWorldBound reports a ROTATED BOX CORNER, which dives below the true lowest vertex as
# the body leans (measured: a can tilted 39 deg read 14.1 mm low against a 13.9 mm band and scored
# NOT placed, while the two cans stacked on top of it both scored). Scaling with the body's own
# height keeps a tall bottle and a flat book on the same footing without a per-object constant.
_ON_BAND_M: float = 0.06
_ON_TILT_FRAC: float = 0.25
# A shelf-less open-top receptacle releases just inside its opening. Its base must
# settle below that plane. The registered floor follows visible geometry, while the
# collision hull can hold the visible base above it; accepting only the lower half
# of the registered interior keeps that stable floor rest far below the rim.
_INSIDE_DROP_PLANE_TOL_M: float = 0.002


def _mat_from_pos_quat(pos, quat_xyzw) -> np.ndarray:
    """Column-vector 4x4 from a translation + a quaternion in XYZW order (scipy/Isaac convention)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4, dtype=np.float64)
    T[0, 0], T[0, 1], T[0, 2] = (
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
    )
    T[1, 0], T[1, 1], T[1, 2] = (
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
    )
    T[2, 0], T[2, 1], T[2, 2] = (
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    )
    T[:3, 3] = (float(pos[0]), float(pos[1]), float(pos[2]))
    return T


def _pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return connector position error in metres and orientation error in radians."""
    position_error = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    ra, rb = a[:3, :3].astype(np.float64).copy(), b[:3, :3].astype(np.float64).copy()
    for m in (ra, rb):
        for c in range(3):
            n = float(np.linalg.norm(m[:, c])) or 1.0
            m[:, c] /= n
    cos = (float(np.trace(ra.T @ rb)) - 1.0) * 0.5
    orientation_error = math.acos(max(-1.0, min(1.0, cos)))
    return position_error, orientation_error


def _poses_coincide(a: np.ndarray, b: np.ndarray) -> bool:
    """Two connector poses coincide within OmniGibson's 5 cm / 15 degree tolerance."""
    position_error, orientation_error = _pose_error(a, b)
    return (
        position_error <= _ATTACH_POS_TOL_M and orientation_error <= _ATTACH_ANG_TOL_RAD
    )


class Role(enum.Enum):
    GRASPABLE = (
        "graspable"  # a pickable object (one rendered /World/Obj at a time in P0)
    )
    SURFACE = "surface"  # a support top a PLACE targets / an object rests ON
    CONTAINER = "container"  # an interior an object goes INSIDE
    OPENABLE = "openable"  # a drawer/door with a drivable joint (OPEN/CLOSED)
    TOGGLEABLE = "toggleable"  # a device control that supports TURN_ON / TURN_OFF
    SEAT = "seat"  # a sittable surface (chair/sofa) for SIT
    PUSHABLE = "pushable"  # a floor fixture that can be translated by PUSH
    PULLABLE = "pullable"  # a floor fixture that can be translated by PULL


@dataclass
class Entity:
    """One scene entity. In SPEC mode ``xy``/``z`` are the authored world pose; in LIVE mode they are a
    fallback and the live USD bbox is read from ``prim_path``."""

    name: str
    role: Role
    prim_path: str | None = None
    xy: tuple[float, float] | None = (
        None  # world XY (object resting pos / surface-top center)
    )
    z: float | None = None  # object rest-z OR surface top-z
    approach_yaw: float = (
        0.0  # heading to FACE when reaching it (deg, pose_body convention)
    )
    half: tuple[float, float] = (
        0.6,
        0.4,
    )  # surface/container XY half-extent (footprint test)
    height: float = 0.30  # container interior height (INSIDE test)
    joint_name: str | None = None  # OPENABLE drivable dof
    open_thresh: float = 0.10  # dof value above which it counts as OPEN
    joint_val: float = 0.0  # SPEC-mode dof value (LIVE reads the articulation)
    axis: tuple[float, float, float] = (
        0.0,
        -1.0,
        0.0,
    )  # OPENABLE: world open direction (unit)
    travel: float = 0.25  # OPENABLE: how far the drawer slides / door opens (m)
    stand_gap: float = (
        0.46  # FALLBACK flat stand distance (used only when foot_half is None)
    )
    # Footprint of the furniture the BODY must clear when reaching this entity's contact. When
    # foot_half is set, stand_pose_for derives the stand distance from the furniture near-face +
    # clearance (+ travel for openables) instead of the flat stand_gap -> "object at any position,
    # human stands clear by construction". foot_xy defaults to the contact xy (object/handle sits
    # over its own footprint); set it to the SUPPORT footprint for an object resting ON a pillar.
    foot_xy: tuple[float, float] | None = (
        None  # obstacle footprint center (None -> contact xy)
    )
    foot_half: tuple[float, float] | None = (
        None  # obstacle footprint half-extent (X,Y); None -> stand_gap
    )
    top_z: float | None = (
        None  # obstacle top height (used by the reach/collision guard)
    )
    clearance: float = (
        0.45  # body clearance to stand IN FRONT of the furniture near-face (m)
    )
    meta: dict = field(default_factory=dict)
    # --- revolute DOOR swing (v1). hinge_xy is None for a prismatic drawer -> the swing path is never
    # taken and _slide runs byte-identical. Populated by behavior_import for category=='door' with a
    # revolute-Z hinge; the door's grounded object_pos stays the CLOSED handle (reach reference). ---
    hinge_xy: tuple[float, float] | None = None  # world XY of the vertical hinge line
    hinge_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)  # hinge axis (unit, +Z)
    swing_angle: float = 0.0  # open sweep magnitude (rad, ~pi/2 = joint upper limit)
    swing_sign: float = 1.0  # sweep direction about +Z (+1 / -1, from the joint frame)
    leaf_paths: tuple[str, ...] = ()  # USD sub-prim path(s) of the swinging leaf/leaves
    # Per-leaf swung-open geometry aligned with leaf_paths. Importers seed authored joint origins; the
    # renderer validates or replaces them and then publishes the final pivots used by navigation/reach.
    leaf_hinges: tuple = ()  # per-leaf world hinge XY (x, y)
    leaf_signs: tuple = ()  # per-leaf swing sign (+1 / -1)
    leaf_bboxes: tuple = ()  # per-leaf CLOSED world AABB ((mn3), (mx3))
    # OPEN/CLOSE grip override: an openable CONTAINER (fridge/cabinet/drawer) registers its interior xy for
    # PutInside, but the open/close hand must grip the DOOR/DRAWER FRONT (not the deep interior). When set,
    # _swing/_slide reach here instead of object_pos. None -> a plain door grips its own grounded object_pos.
    handle_xy: tuple[float, float] | None = None  # world XY of the door/drawer grip
    handle_z: float | None = None  # grip height (m); None -> use the object_pos z
    # PutInside drop height for a TALL front-opening container: the INSIDE band stays [z, z+height] (full
    # interior, for the goal check) but the object is set down at drop_z (a reachable shelf ~chest height)
    # instead of the interior ceiling z+height. None -> place at z+height (byte-identical top-opening bin).
    drop_z: float | None = None
    # Interior placement SPACES, geometry-extracted: ((name, top_z, bottom_z, cx, cy, hx, hy), ...) ordered
    # TOP-DOWN then LEFT-RIGHT, named "<container>_space{i}" (1-based). Distinct spaces are split by shelf
    # LEVEL and by LATERAL dividers, so a fridge's left-shelf and right-shelf are two separate spaces. Each
    # carries its board's z-slab (bottom_z..top_z) so the PutInside reach can treat ALL boards as obstacles.
    # Empty for a top-open bin (no internal boards). produce_place_inside drops the held object onto a chosen
    # space (auto-distributed across free spaces, upper levels first, when unspecified); _relations_inside_live
    # reads the interior z-band for a real INSIDE test.
    spaces: tuple = ()
    # DUAL-ROLE affordance: a GRASPABLE object that is ALSO a valid put/putinside TARGET (a bowl / tupperware:
    # pick it up AND place things into/onto it, exactly like a table or a trash can) carries its container/
    # surface drop geometry HERE, in a nested Entity (role CONTAINER/SURFACE + interior z/height/spaces, or a
    # surface top). The primary Entity keeps role=GRASPABLE with its OWN pick pose (xy/z) -- a single `role`
    # cannot be both and `register` is last-write-wins. Placement / verb-selection / scoring read
    # ``target or self`` so ANY object can be a target like furniture. None -> a plain graspable, or a furniture
    # target that already carries its affordance on itself.
    target: "Entity | None" = None
    # Per-part ATTACH points (assembly, Task 2): full point name ("<base>M" / "<base>F") -> [{"pos": (x,y,z)
    # part-local, "quat": (x,y,z,w) XYZW}, ...]. A Male plugs into the same-based Female on another part.
    # Populated by a task builder from BEHAVIOR metadata; empty for every existing entity -> the ATTACHED
    # emitter is a no-op unless a scene registers attach parts.
    attach: dict = field(default_factory=dict)
    affordances: dict[Role, "Entity"] = field(default_factory=dict)


@dataclass
class SceneState:
    """The scene registry + grounding + relation extraction. ``held`` is maintained by the executor
    (authoritative for our kinematic carry). ``stage`` != None selects the LIVE Isaac paths."""

    floor_z: float = 0.0
    start_xy: tuple[float, float] = (0.0, 0.0)
    stage: object | None = None
    entities: dict[str, Entity] = field(default_factory=dict)
    # Declared symbolic state = the single source of truth under trusted execution (VH-style): objects
    # held, each object's supporting surface, and which openables are open. The executor declares these
    # as actions complete; relations() reports them. Geometric/physics paths stay as optional guards.
    held: set[str] = field(default_factory=set)
    # Dead ``prim_path``s seen by ``_live_bbox``: {path: [entity keys that hold it]}. Diagnostic only --
    # nothing reads it to make a decision; it exists so a stale path is nameable from the manifest
    # instead of only from a USD traceback that identifies neither the path nor its owner.
    _stale_paths: dict[str, list] = field(default_factory=dict)
    # Signs a [LookAt] could not find ANY reachable reading stand for, so it read them from wherever the
    # body already was (semantic_executor._square_on_stand). Diagnostic only -- nothing branches on it --
    # but it is the difference between "the assistant chose wrong" and "we never showed it the placard",
    # and those two must never share a label.
    cue_stand_misses: list = field(default_factory=list)
    on: dict[str, str] = field(default_factory=dict)  # obj -> the surface it rests on
    open_objs: set[str] = field(default_factory=set)  # openables currently OPEN
    _openable_controllers: dict[str, dict] = field(
        default_factory=dict,
        repr=False,
    )
    running: set[str] = field(default_factory=set)  # toggleable devices currently ON
    sitting_on: str | None = None  # the seat the agent is sitting on (None = standing)
    # Step-4 CONTACT-verb declared effects (symbolic, trusted-execution -- same pattern as on/open).
    cut: set[str] = field(default_factory=set)  # objects a CUT action completed on
    cooked: set[str] = field(default_factory=set)  # objects a COOK action completed on
    inside: dict[str, str] = field(default_factory=dict)  # contents -> container (POUR)
    # --- navigation door layer (all defaulted -> byte-identical when the scene has no doors). Populated
    # by behavior_import under the door-swing build flag; consumed by the nav grid door layer and the
    # execute() door pre-pass (door_planner). ---
    door_prims: list = field(
        default_factory=list
    )  # USD prim paths of door assets (parallel to wall_prims)
    doors: dict | None = (
        None  # name -> DoorInfo (footprint/hinge/near-far approach); None = no doors
    )
    close_doors_behind: bool = (
        False  # opt-in: auto-insert CloseDoor after passing through
    )
    # --- assembly layer (all defaulted -> byte-identical when a scene registers no attach parts). Populated by
    # a task builder + produce_attach; consumed by _relations_attached_live (the honest ATTACHED relation). ---
    attached: dict[str, str] = field(
        default_factory=dict
    )  # child key -> parent key (declared mates)
    attach_point: dict = field(
        default_factory=dict
    )  # child key -> (point_base, female_instance_idx)
    assembly_root: str | None = (
        None  # the pre-placed root part (a mate onto it needs no prior assembly)
    )
    assembly_roots: tuple[str, ...] = ()
    assembly_pose: dict = field(
        default_factory=dict
    )  # child key -> welded world 4x4 (frame-gen parent-pose source; the render then drives each prim from its ("pose",T) frames)
    assembly_required: dict[str, str] = field(default_factory=dict)
    movement_targets: dict[str, str] = field(default_factory=dict)
    actor_xy: tuple[float, float] | None = None
    actor_seated: bool = False
    tracked_near: set[tuple[str, str]] = field(default_factory=set)
    tracked_raised: set[str] = field(default_factory=set)
    tracked_unmoved: set[str] = field(default_factory=set)
    goal_predicates: set[tuple[str, str, str]] = field(default_factory=set)
    goal_predicates_configured: bool = False
    motion_max_m: dict[str, float] = field(default_factory=dict)
    motion_max_deg: dict[str, float] = field(default_factory=dict)
    # FPV benchmark contract: avatar-body contact remains telemetry; only a blocked/intersecting
    # eye camera can make human-vs-scene geometry fail.  Set False only for non-egocentric diagnostics.
    egocentric_camera_only_collision: bool = True
    egocentric_collision_fallback_used: bool = False
    # Runtime-checkable staging alternatives keyed by the logical surface name. The builder records only
    # same-room, same-category scene fixtures; the live PhysX preflight may replace an unusable primary
    # without changing the task's semantic binding.
    initial_support_options: dict[str, list[dict]] = field(default_factory=dict)
    object_rest_pose: dict = field(
        default_factory=dict
    )  # graspable key -> explicit 6-DoF rest pose, used when an action intentionally changes orientation
    # Per-episode world-AABB memo for entities that CANNOT move within an episode (see entity_aabbs).
    # repr/compare off: a memo is not scene state -- it must not print alongside the entity registry, and
    # two scenes are not unequal because one has been read and the other has not.
    _static_aabb: dict = field(default_factory=dict, repr=False, compare=False)
    _episode_baseline: dict | None = field(default=None, repr=False, compare=False)
    _episode_baseline_relations: set[tuple] = field(
        default_factory=set, repr=False, compare=False
    )

    # ---- registration + grounding -----------------------------------------------------------------
    def register(self, name: str, role: Role, **kw) -> Entity:
        """Register (or replace) an entity. kw = any Entity field (prim_path/xy/z/approach_yaw/...)."""
        self.entities[name] = Entity(name=name, role=role, **kw)
        self._static_aabb.pop(name, None)  # a re-registration can repoint prim_path
        return self.entities[name]

    def register_affordance(self, name: str, role: Role, **kw) -> Entity:
        """Attach another typed capability without replacing the entity's primary role."""
        owner = self.resolve(name)
        if owner.role is role:
            for key, value in kw.items():
                setattr(owner, key, value)
            return owner
        kw.setdefault("prim_path", owner.prim_path)
        kw.setdefault("approach_yaw", owner.approach_yaw)
        affordance = Entity(name=name, role=role, **kw)
        owner.affordances[role] = affordance
        return affordance

    def register_from_prim(
        self,
        name: str,
        role: Role,
        prim_path: str,
        *,
        is_obstacle: bool | None = None,
        **kw,
    ) -> Entity:
        """ADDITIVE grounding: register an entity whose footprint is read LIVE from ``prim_path``'s
        world AABB (never edits ``register``/``Entity`` defaults). Derives xy + z + XY half-extent +
        top_z from the bbox; a GRASPABLE gets z = bbox base (object origin), any other role gets
        z = bbox top (surface). ``is_obstacle`` (default: True for non-GRASPABLE) also sets
        foot_xy/foot_half/top_z so the body stands clear + the reach/nudge treat it as solid;
        graspable props leave foot_half=None so they stay OUT of solid_boxes()/the nudge. Any explicit
        kw (approach_yaw/joint_name/axis/travel/clearance/...) overrides the derived value. LIVE-only
        (needs an attached stage)."""
        prim = self.stage.GetPrimAtPath(prim_path) if self.stage is not None else None
        if prim is None or not prim.IsValid():
            # This function's ENTIRE job is to derive geometry from the prim, so unlike the readers
            # there is no spec pose to fall back to -- registering the entity with an empty bbox would
            # silently give it a zero footprint at the origin. Fail, but fail NAMED: the raw USD error
            # ("Invalid prim: null") identifies neither the entity being registered nor the dead path.
            self._stale_prim(prim_path)
            raise ValueError(
                f"register_from_prim({name!r}, {role}): prim_path {prim_path!r} does not resolve on "
                f"this stage -- cannot read a world AABB from a prim that is not there. Either the "
                f"path is stale or registration ran before the prim was built."
            )
        from pxr import Usd, UsdGeom

        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        hx = max(0.5 * (mx[0] - mn[0]), 0.02)
        hy = max(0.5 * (mx[1] - mn[1]), 0.02)
        zmax = float(mx[2])
        if is_obstacle is None:
            is_obstacle = role is not Role.GRASPABLE
        fields: dict = {"prim_path": prim_path, "xy": (float(cx), float(cy))}
        if role is Role.GRASPABLE:
            fields["z"] = float(mn[2])  # object origin ~ its bbox base on the support
        else:
            fields["z"] = zmax  # surface / seat / openable top
            fields["half"] = (float(hx), float(hy))
        if is_obstacle:
            fields["foot_xy"] = (float(cx), float(cy))
            fields["foot_half"] = (float(hx), float(hy))
            fields["top_z"] = zmax
        fields.update(kw)  # explicit caller values win over the derived ones
        return self.register(name, role, **fields)

    def resolve(self, name: str) -> Entity:
        e = self.entities.get(name)
        if e is None:
            raise KeyError(
                f"scene_state: unknown entity {name!r} (have {sorted(self.entities)})"
            )
        return e

    def affordance(self, name: str, role: Role) -> Entity | None:
        """Return one typed affordance without changing the entity's primary role."""
        entity = self.resolve(name)
        if entity.role is role:
            return entity
        target = entity.affordances.get(role)
        if target is not None:
            return target
        if entity.target is not None and entity.target.role is role:
            return entity.target
        return None

    def is_movable(self, name: str) -> bool:
        """Whether reset/motion tracking must treat this entity as pose-bearing."""
        entity = self.resolve(name)
        return entity.role is Role.GRASPABLE or any(
            self.affordance(name, role) is not None
            for role in (Role.PUSHABLE, Role.PULLABLE)
        )

    def add_mobility(self, name: str, *, push: bool, pull: bool) -> None:
        """Register explicit floor-movement capabilities without replacing the primary role."""
        owner = self.resolve(name)
        common = {
            "xy": owner.xy,
            "z": owner.z,
            "half": owner.half,
            "height": owner.height,
            "foot_xy": owner.foot_xy,
            "foot_half": owner.foot_half,
            "top_z": owner.top_z,
        }
        if push:
            self.register_affordance(name, Role.PUSHABLE, **common)
        if pull:
            self.register_affordance(name, Role.PULLABLE, **common)

    def target_view(self, name: str, role: Role) -> Entity | None:
        """Typed target geometry at the entity's current live pose."""
        entity = self.resolve(name)
        target = self.affordance(name, role)
        drawer_offset = (
            (target.meta or {}).get("drawer_open_offset_xy") if target else None
        )
        if (
            target is entity
            and getattr(role, "value", None) == Role.CONTAINER.value
            and name in self.open_objs
            and drawer_offset is not None
        ):
            dx, dy = (float(value) for value in drawer_offset)
            return replace(
                target,
                xy=(float(target.xy[0]) + dx, float(target.xy[1]) + dy),
                spaces=self._translated_spaces(target.spaces, dx, dy, 0.0),
            )
        if (
            target is None
            or target is entity
            or self.stage is None
            or not entity.prim_path
        ):
            return target
        prim = self.stage.GetPrimAtPath(entity.prim_path)
        if prim is None or not prim.IsValid():
            return target
        return self._live_target_view(entity, target, self.object_aabb(name))

    @staticmethod
    def _translated_spaces(spaces, dx: float, dy: float, dz: float) -> tuple:
        return tuple(
            (
                space[0],
                float(space[1]) + dz,
                float(space[2]) + dz,
                float(space[3]) + dx,
                float(space[4]) + dy,
                *space[5:],
            )
            for space in spaces
        )

    def _live_target_view(self, entity: Entity, target: Entity, bounds) -> Entity:
        mn, mx = bounds
        cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        hx, hy = 0.5 * (mx[0] - mn[0]), 0.5 * (mx[1] - mn[1])
        meta = dict(target.meta or {})
        original = meta.get("affordance_bbox")
        if original:
            old_mn, old_mx = original
            old_cx, old_cy = (
                0.5 * (old_mn[0] + old_mx[0]),
                0.5 * (old_mn[1] + old_mx[1]),
            )
            dx, dy, dz = cx - old_cx, cy - old_cy, mn[2] - old_mn[2]
        else:
            old_xy = target.xy or (cx, cy)
            dx, dy, dz = cx - old_xy[0], cy - old_xy[1], 0.0
        half_scale = meta.get("half_scale", (1.0, 1.0))
        common = {
            "prim_path": entity.prim_path,
            "xy": (float(cx), float(cy)),
            "half": (
                max(0.03, float(hx) * float(half_scale[0])),
                max(0.03, float(hy) * float(half_scale[1])),
            ),
            "foot_xy": (float(cx), float(cy)),
            "foot_half": (float(hx), float(hy)),
            "approach_yaw": entity.approach_yaw,
            "spaces": self._translated_spaces(target.spaces, dx, dy, dz),
        }
        if target.role in (Role.SURFACE, Role.SEAT):
            return replace(target, z=float(mx[2]), top_z=float(mx[2]), **common)
        if target.role is Role.TOGGLEABLE:
            if original:
                anchor_xy = target.xy or (old_cx, old_cy)
                anchor_z = float(target.z) if target.z is not None else float(old_mx[2])
                common["xy"] = (
                    float(anchor_xy[0]) + dx,
                    float(anchor_xy[1]) + dy,
                )
                return replace(
                    target,
                    z=anchor_z + dz,
                    top_z=float(mx[2]),
                    **common,
                )
            return replace(target, z=float(mx[2]), top_z=float(mx[2]), **common)
        if target.role is not Role.CONTAINER:
            return replace(target, **common)
        target_z = float(target.z) if target.z is not None else float(mn[2])
        bottom = float(meta.get("bottom_inset", max(0.0, target_z - mn[2])))
        top = float(meta.get("top_inset", 0.05))
        z = float(mn[2]) + bottom
        drop = float(mn[2]) + float(meta.get("drop_offset", bottom))
        return replace(
            target,
            z=z,
            height=max(0.05, float(mx[2]) - top - z),
            top_z=float(mx[2]),
            drop_z=drop,
            **common,
        )

    def object_pos(self, name: str) -> np.ndarray:
        """Current world (x,y,z) of a GRASPABLE object: live USD bbox if a stage is attached AND the prim
        exists, else the registered spec pose. The prim may not be built yet during INIT validation (which
        runs BEFORE the object-settle build), so fall back to the spec xy/z instead of crashing on a null
        prim (byte-identical for existing scenes: a built prim is valid -> the live path is taken)."""
        e = self.resolve(name)
        if self.stage is not None and e.prim_path:
            prim = self.stage.GetPrimAtPath(e.prim_path)
            if prim and prim.IsValid():
                (x, y), zmin, _zmax = self._live_bbox(e.prim_path)
                return np.array(
                    [x, y, zmin]
                )  # object origin sits ~at its bbox base on the surface
        return np.array([e.xy[0], e.xy[1], e.z], dtype=np.float64)

    def object_bbox(self, name: str):
        """Current world AABB of an object -> ((cx,cy), zmin, zmax). Live prim if built, else a degenerate
        spec-based fallback (byte-identical for built scenes: a built prim takes the live path). Used by the
        look-zoom framing so the object's OWN size drives the FOV -- never a magic metre."""
        e = self.resolve(name)
        if self.stage is not None and e.prim_path:
            prim = self.stage.GetPrimAtPath(e.prim_path)
            # KEPT deliberately, do not route this through _live_bbox's naming path. Like
            # ``object_xyz`` above, this can legitimately run before the prim is BUILT (INIT
            # validation), where a null prim is expected and benign -- not a stale path. Reporting it
            # as STALE would fire on every object of every scene during INIT and bury the real ones.
            if prim and prim.IsValid():
                return self._live_bbox(e.prim_path)
        z = float(e.z)
        return ((float(e.xy[0]), float(e.xy[1])), z, z)

    def object_aabb(self, name: str):
        """Full world AABB of an object -> ((mnx,mny,mnz),(mxx,mxy,mxz)). Live prim if built, else a degenerate
        spec-based point (byte-identical for built scenes). Unlike object_bbox this keeps the XY extents, so the
        look-zoom label framing can size a label from the object's OWN width/depth -- never a magic metre."""
        e = self.resolve(name)
        if self.stage is not None and e.prim_path:
            prim = self.stage.GetPrimAtPath(e.prim_path)
            if prim and prim.IsValid():
                from pxr import Usd, UsdGeom

                bc = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                )
                rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
                mn, mx = rng.GetMin(), rng.GetMax()
                return (
                    (float(mn[0]), float(mn[1]), float(mn[2])),
                    (float(mx[0]), float(mx[1]), float(mx[2])),
                )
        z = float(e.z)
        return (
            (float(e.xy[0]), float(e.xy[1]), z),
            (float(e.xy[0]), float(e.xy[1]), z),
        )

    def entity_aabbs(self, names) -> dict:
        """World AABBs ``{name: ((mnx,mny,mnz),(mxx,mxy,mxz))}`` for MANY entities in ONE pass.

        Deliberately not a loop over ``object_aabb``: that builds a fresh ``UsdGeom.BBoxCache`` per call
        (~1 ms of prim traversal each, so ~0.35 s across a 290-entity BEHAVIOR registry), and almost every
        entity in that registry has the SAME bound on every step of the episode. So one BBoxCache serves the
        whole sweep, and each ``_bounds_static`` entity is read from USD exactly once per episode -- after
        the first call only the graspables (a handful) touch USD at all. Callers that sweep the registry
        every step (the visibility pass) are what this exists for; ``object_aabb`` stays the single-entity,
        always-live reader.

        Unknown names are skipped rather than raising: a caller sweeping the whole registry must not die on
        one stale key."""
        out: dict = {}
        bcache: list = [None]  # lazily built, shared across the sweep
        for name in names:
            e = self.entities.get(name)
            if e is None:
                continue
            box = self._static_aabb.get(name)
            if box is None:
                box = self._read_aabb(e, bcache)
                if self._bounds_static(e):
                    self._static_aabb[name] = box
            out[name] = box
        return out

    def invalidate_bounds(self) -> None:
        """Drop the static world-AABB memo (see ``entity_aabbs``). MANDATORY at any episode boundary that
        can move a cached prim -- a reset rewrites poses, a scene switch rebuilds every prim."""
        self._static_aabb.clear()

    def capture_episode_baseline(self) -> None:
        """Snapshot movable prim poses and symbolic relations after scene construction."""
        poses: dict[str, np.ndarray] = {}
        named_poses: dict[str, np.ndarray] = {}
        for name, entity in self.entities.items():
            if not self.is_movable(name):
                continue
            pose = self._entity_world_pose(entity)
            if pose is None:
                continue
            named_poses[name] = pose.copy()
            if entity.prim_path:
                poses[entity.prim_path] = pose.copy()
        self._episode_baseline = {
            "held": set(self.held),
            "on": dict(self.on),
            "inside": dict(self.inside),
            "open_objs": set(self.open_objs),
            "running": set(self.running),
            "cut": set(self.cut),
            "cooked": set(self.cooked),
            "attached": dict(self.attached),
            "attach_point": copy.deepcopy(self.attach_point),
            "assembly_pose": {
                name: np.asarray(pose, dtype=np.float64).copy()
                for name, pose in self.assembly_pose.items()
            },
            "sitting_on": self.sitting_on,
            "actor_xy": self.actor_xy,
            "actor_seated": self.actor_seated,
            "movement_targets": dict(self.movement_targets),
            "motion_max_m": dict.fromkeys(named_poses, 0.0),
            "motion_max_deg": dict.fromkeys(named_poses, 0.0),
            "object_rest_pose": {
                name: np.asarray(pose, dtype=np.float64).copy()
                for name, pose in self.object_rest_pose.items()
            },
            "entity_pose": {
                name: (entity.xy, entity.z) for name, entity in self.entities.items()
            },
            "prim_pose": poses,
            "named_pose": named_poses,
            "affordances": {
                name: copy.deepcopy(entity.affordances)
                for name, entity in self.entities.items()
            },
            "assembly_active": {
                name: list(
                    (entity.meta or {}).get("_active_assembly_affordances") or ()
                )
                for name, entity in self.entities.items()
            },
        }
        self.motion_max_m = dict.fromkeys(named_poses, 0.0)
        self.motion_max_deg = dict.fromkeys(named_poses, 0.0)
        self._episode_baseline_relations = set(self.relations(source="physics"))

    def episode_baseline_relations(self) -> set[tuple]:
        """Return the physical relations captured before any episode action."""
        return set(self._episode_baseline_relations)

    def restore_episode_baseline(self) -> None:
        """Restore the scene snapshot captured after all deferred objects were realized."""
        baseline = self._episode_baseline
        if baseline is None:
            return
        self.held = set(baseline["held"])
        self.on = dict(baseline["on"])
        self.inside = dict(baseline["inside"])
        self.open_objs = set(baseline["open_objs"])
        self.running = set(baseline["running"])
        self.cut = set(baseline.get("cut", ()))
        self.cooked = set(baseline.get("cooked", ()))
        self.attached = dict(baseline.get("attached", {}))
        self.attach_point = copy.deepcopy(baseline.get("attach_point", {}))
        self.assembly_pose = {
            name: np.asarray(pose, dtype=np.float64).copy()
            for name, pose in baseline.get("assembly_pose", {}).items()
        }
        self.sitting_on = baseline.get("sitting_on")
        self.actor_xy = baseline.get("actor_xy")
        self.actor_seated = bool(baseline.get("actor_seated", False))
        self.movement_targets = dict(baseline.get("movement_targets", {}))
        self.motion_max_m = dict(baseline.get("motion_max_m", {}))
        self.motion_max_deg = dict(baseline.get("motion_max_deg", {}))
        self.object_rest_pose = {
            name: np.asarray(pose, dtype=np.float64).copy()
            for name, pose in baseline.get("object_rest_pose", {}).items()
        }
        for name, (xy, z) in baseline["entity_pose"].items():
            entity = self.entities.get(name)
            if entity is not None:
                entity.xy, entity.z = xy, z
                entity.affordances = copy.deepcopy(
                    baseline.get("affordances", {}).get(name, {})
                )
                entity.meta["_active_assembly_affordances"] = list(
                    baseline.get("assembly_active", {}).get(name, ())
                )
        for path, pose in baseline["prim_pose"].items():
            self._write_live_world_col(path, pose)
        for name in ("_space_assign", "_into_placed"):
            if hasattr(self, name):
                delattr(self, name)
        self.invalidate_bounds()

    def _entity_world_pose(self, entity: Entity) -> np.ndarray | None:
        if self.stage is not None and entity.prim_path:
            live = self._live_world_col(entity.prim_path)
            if live is not None:
                return live
        if entity.name in self.object_rest_pose:
            return np.asarray(
                self.object_rest_pose[entity.name], dtype=np.float64
            ).copy()
        if entity.xy is None or entity.z is None:
            return None
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = (float(entity.xy[0]), float(entity.xy[1]), float(entity.z))
        return pose

    def _write_live_world_col(self, prim_path: str, pose) -> None:
        if self.stage is None:
            return
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return
        from pxr import Gf, UsdGeom

        matrix = np.asarray(pose, dtype=np.float64)
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(
            Gf.Matrix4d(*[float(value) for value in matrix.T.flatten()])
        )

    def _bounds_static(self, e: Entity) -> bool:
        """Can ``e``'s world AABB be read ONCE per episode? True only for a fixture with no moving part.

            Movement is excluded by construction, not by role alone: a GRASPABLE is carried and re-settled; an
            entity with a drivable joint, a door row or a swinging leaf grows its OWN world bound the moment it
            opens (``ComputeWorldBound`` on the cabinet includes the leaf -- the exact staleness the footprint
        memo in the environment layer warns about); an attached/assembled part is welded to a new pose
            mid-episode."""
        if e.role in (Role.GRASPABLE, Role.OPENABLE):
            return False
        if any(role in e.affordances for role in (Role.PUSHABLE, Role.PULLABLE)):
            return False
        if e.joint_name or e.leaf_paths:
            return False
        if e.name in (self.doors or {}) or e.name in self.attached:
            return False
        return e.name not in self.assembly_pose and e.name not in self.held

    def _read_aabb(self, e: Entity, bcache: list):
        """One entity's LIVE world AABB through the shared BBoxCache in ``bcache[0]`` (built on first use),
        or the degenerate spec-pose point when the prim is not built yet -- the same fallback
        ``object_aabb`` takes, so both readers agree on an entity registered before its prim exists."""
        prim = (
            self.stage.GetPrimAtPath(e.prim_path)
            if (self.stage is not None and e.prim_path)
            else None
        )
        if prim is None or not prim.IsValid():
            return self._spec_aabb(e)
        if bcache[0] is None:
            from pxr import Usd, UsdGeom

            bcache[0] = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            )
        rng = bcache[0].ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        return (
            (float(mn[0]), float(mn[1]), float(mn[2])),
            (float(mx[0]), float(mx[1]), float(mx[2])),
        )

    @staticmethod
    def _spec_aabb(e: Entity):
        """Degenerate AABB at the entity's registered spec pose. Tolerates an unset xy/z (a registry sweep
        sees entities ``object_aabb``'s callers never ask for)."""
        x, y = (float(e.xy[0]), float(e.xy[1])) if e.xy else (0.0, 0.0)
        z = float(e.z) if e.z is not None else 0.0
        return ((x, y, z), (x, y, z))

    def object_rest_mat(self, name: str) -> np.ndarray:
        """Object's current rest transform as a 4x4 in the COLUMN-vector convention (rotation in [:3,:3],
        translation in [:3,3]) -- the same convention ``smplh_anim_core.set_object_world`` writes. Live prim if
        built (its true settled orientation, whatever keeps it upright), else an upright identity at the spec
        pose. Used to RAISE an object to an inspect pose while preserving that orientation (only adding a yaw),
        so it never tips."""
        e = self.resolve(name)
        if name in self.object_rest_pose:
            return np.asarray(self.object_rest_pose[name], dtype=np.float64).copy()
        T = np.eye(4, dtype=np.float64)
        if self.stage is not None and e.prim_path:
            prim = self.stage.GetPrimAtPath(e.prim_path)
            if prim and prim.IsValid():
                from pxr import Usd, UsdGeom

                xc = UsdGeom.XformCache(Usd.TimeCode.Default())
                m = np.array(xc.GetLocalToWorldTransform(prim)).reshape(4, 4)
                T[:3, :3] = m[
                    :3, :3
                ].T  # USD row-vector matrix -> column-vector rotation
                T[:3, 3] = m[3, :3]  # translation (USD stores it in the last row)
                return T
        T[:3, 3] = np.array([e.xy[0], e.xy[1], e.z], dtype=np.float64)
        return T

    def object_origin(self, name: str) -> np.ndarray:
        """Current world (x,y,z) of a GRASPABLE object's ORIGIN (its prim's world-transform translation) --
        the pose the settle froze and ``set_object_rest`` stored. Use THIS (not ``object_pos``, which returns
        the world-bbox BASE) to DRIVE the object's kinematic rest and to target the grasp: ``realize`` writes
        the value straight into the prim's transform translation (= the origin), so driving with the bbox base
        double-applies ``min_z`` and sinks a CENTER-pivot object by ``|min_z|`` (and makes it jump down the
        moment the rest-drive begins, before the hand contacts it). Falls back to the spec pose when the prim
        is not built yet. For a BASE-pivot object (origin == bbox base) this equals ``object_pos`` exactly, so
        base-pivot scenes stay byte-identical."""
        e = self.resolve(name)
        if name in self.object_rest_pose:
            return np.asarray(self.object_rest_pose[name], dtype=np.float64)[
                :3, 3
            ].copy()
        if self.stage is not None and e.prim_path:
            prim = self.stage.GetPrimAtPath(e.prim_path)
            if prim and prim.IsValid():
                from pxr import Usd, UsdGeom

                xc = UsdGeom.XformCache(Usd.TimeCode.Default())
                m = np.array(xc.GetLocalToWorldTransform(prim)).reshape(4, 4)
                return m[3, :3].astype(
                    np.float64
                )  # last row = world translation = the prim ORIGIN
        return np.array([e.xy[0], e.xy[1], e.z], dtype=np.float64)

    def surface_top(self, name: str) -> tuple[tuple[float, float], float]:
        """((x,y) center, top_z) of a SURFACE: live world-bbox top if a stage is attached, else spec."""
        primary = self.resolve(name)
        e = (
            self.target_view(name, Role.SURFACE)
            or self.target_view(name, Role.CONTAINER)
            or primary
        )
        if e is not primary:
            return (e.xy, float(e.top_z) if e.top_z is not None else float(e.z))
        if self.stage is not None and e.prim_path:
            # None = the path is stale (named by _live_bbox). Fall through to the spec pose below --
            # the SAME branch this already takes when there is no stage, i.e. "no live reading".
            _bb = self._live_bbox(e.prim_path)
            if _bb is not None:
                (x, y), _zmin, zmax = _bb
                seat_contact = (e.meta or {}).get("seat_contact_xy")
                if isinstance(seat_contact, (list, tuple)) and len(seat_contact) >= 2:
                    x, y = float(seat_contact[0]), float(seat_contact[1])
                # The MEASURED worktop wins over the raw bbox top when the builder recorded one. A
                # lab bench carries a reagent rack, so its bbox tops out 168 mm above the surface
                # things actually rest on -- and reading the bbox made every `on` goal on such a
                # bench false while the video showed the object placed correctly (NARRATE-3).
                return ((x, y), float(e.top_z) if e.top_z is not None else zmax)
        return (e.xy, e.z)

    def place_xy_from_spot(
        self,
        name: str,
        spot: tuple[float, float],
        *,
        object_name: str | None = None,
        space: str | None = None,
    ) -> tuple[float, float]:
        """Map a normalized surface-local ``(u, v)`` to a safe world-XY center.

        ``u`` is left-to-right in the target's front-facing frame and spans
        ``[-1, 1]``. ``v`` is front-to-back and spans ``[0, 1]``. When the
        placed object's footprint is measurable, the usable rectangle is
        inset so an extreme hint keeps the whole object on the support.
        """
        u, v = float(spot[0]), float(spot[1])
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError("place spot coordinates must be finite")
        if not -1.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
            raise ValueError("place spot requires -1 <= u <= 1 and 0 <= v <= 1")
        target = self._placement_target(name)
        center, half = self._placement_region(name, target, space)
        entry_axis = (target.meta or {}).get("entry_axis")
        try:
            measured_axis = np.asarray(entry_axis, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            measured_axis = np.empty(0, dtype=np.float64)
        axis_norm = (
            float(np.linalg.norm(measured_axis))
            if measured_axis.shape == (2,) and np.all(np.isfinite(measured_axis))
            else 0.0
        )
        if axis_norm > 1e-9:
            # The measured aperture axis is authoritative for a front-opening target;
            # approach_yaw can instead describe the furniture's navigation pose.
            forward = (
                float(measured_axis[0] / axis_norm),
                float(measured_axis[1] / axis_norm),
            )
        else:
            angle = math.radians(float(target.approach_yaw))
            forward = (math.sin(angle), -math.cos(angle))
        across = (-forward[1], forward[0])
        if axis_norm > 1e-9:
            try:
                aperture_center = np.asarray(
                    (target.meta or {}).get("aperture_center"), dtype=np.float64
                ).reshape(-1)
            except (TypeError, ValueError):
                aperture_center = np.empty(0, dtype=np.float64)
            if aperture_center.shape == (3,) and np.all(np.isfinite(aperture_center)):
                lateral_shift = (float(aperture_center[0]) - float(center[0])) * across[
                    0
                ] + (float(aperture_center[1]) - float(center[1])) * across[1]
                center = (
                    float(center[0]) + across[0] * lateral_shift,
                    float(center[1]) + across[1] * lateral_shift,
                )
        span = min(
            half[0] / max(abs(across[0]), 1e-6),
            half[1] / max(abs(across[1]), 1e-6),
        )
        depth = min(
            half[0] / max(abs(forward[0]), 1e-6),
            half[1] / max(abs(forward[1]), 1e-6),
        )
        object_half = self._placement_object_half(object_name, across, forward)
        across_offset = u * max(0.0, span - object_half[0])
        depth_offset = (2.0 * v - 1.0) * max(0.0, depth - object_half[1])
        return (
            float(center[0] + across[0] * across_offset + forward[0] * depth_offset),
            float(center[1] + across[1] * across_offset + forward[1] * depth_offset),
        )

    def _placement_target(self, name: str) -> Entity:
        primary = self.resolve(name)
        return (
            self.target_view(name, Role.SURFACE)
            or self.target_view(name, Role.CONTAINER)
            or primary
        )

    def _placement_region(
        self, name: str, target: Entity, space: str | None
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        spaces = tuple(target.spaces or ())
        selected = SceneState._placement_space(spaces, space)
        if selected is not None:
            return (
                (float(selected[3]), float(selected[4])),
                (float(selected[5]), float(selected[6])),
            )
        center, _top = self.surface_top(name)
        if center is None:
            raise ValueError(f"placement target {target.name!r} has no XY center")
        return (
            (float(center[0]), float(center[1])),
            (float(target.half[0]), float(target.half[1])),
        )

    @staticmethod
    def _placement_space(spaces: tuple, request: str | None):
        if not spaces or request is None:
            return None
        by_name = {str(candidate[0]).lower(): candidate for candidate in spaces}
        key = str(request).strip().lower()
        if key in by_name:
            return by_name[key]
        tail = key.rsplit("space", 1)[-1].lstrip("_") if "space" in key else key
        if not tail.isdigit():
            return None
        return spaces[max(1, min(len(spaces), int(tail))) - 1]

    def _placement_object_half(
        self,
        object_name: str | None,
        across: tuple[float, float],
        forward: tuple[float, float],
    ) -> tuple[float, float]:
        if object_name is None:
            return 0.0, 0.0
        minimum, maximum = self.object_aabb(object_name)
        hx = 0.5 * max(0.0, float(maximum[0]) - float(minimum[0]))
        hy = 0.5 * max(0.0, float(maximum[1]) - float(minimum[1]))
        if hx <= 1e-6 and hy <= 1e-6:
            fallback = self.resolve(object_name).foot_half
            hx, hy = fallback if fallback is not None else (0.0, 0.0)
        return (
            abs(across[0]) * hx + abs(across[1]) * hy,
            abs(forward[0]) * hx + abs(forward[1]) * hy,
        )

    def calibrate_surface_top(self, name: str, top_z: float) -> None:
        """Use a verified live collision plane as the named surface's top."""
        value = float(top_z)
        if not math.isfinite(value):
            raise ValueError("surface top must be finite")
        primary = self.resolve(name)
        surface = (
            self.target_view(name, Role.SURFACE)
            or self.target_view(name, Role.CONTAINER)
            or primary
        )
        if surface.role not in (Role.SURFACE, Role.CONTAINER):
            raise ValueError(f"{name!r} is not a surface or container")
        surface.z = value
        surface.top_z = value

    def stand_pose_for(
        self, name: str, gap: float | None = None
    ) -> tuple[tuple[float, float], float]:
        """Where to stand + face to reach ``name``: ``gap`` metres in front along its approach heading
        (defaults to the entity's per-action ``stand_gap`` -- grab reaches over a table ~0.46, an
        openable stands back > its travel, a seat stands clear of its front edge).

        Native body forward at heading yaw is Rz(yaw)@[0,-1,0] = [sin, -cos]; standing IN FRONT means
        stepping back from the target along that forward, so the target ends up ahead of the body."""
        e = self.resolve(name)
        geometry = e
        if e.role is Role.GRASPABLE:
            geometry = (
                self.target_view(name, Role.CONTAINER)
                or self.target_view(name, Role.SURFACE)
                or e
            )
        explicit_stand = (geometry.meta or {}).get("operator_stand_xy")
        if (
            isinstance(explicit_stand, (list, tuple))
            and len(explicit_stand) == 2
            and not any(isinstance(value, bool) for value in explicit_stand)
        ):
            try:
                stand = tuple(float(value) for value in explicit_stand)
            except (TypeError, ValueError):
                stand = ()
            if len(stand) == 2 and all(math.isfinite(value) for value in stand):
                return (stand[0], stand[1]), geometry.approach_yaw
        pos = (
            self.object_pos(name)
            if e.role is Role.GRASPABLE
            else np.array([*self.surface_top(name)[0], 0.0])
        )
        a = math.radians(geometry.approach_yaw)
        fwd = np.array([math.sin(a), -math.cos(a)])  # body forward = toward the target
        if gap is None and geometry.foot_half is not None:
            # GEOMETRY-DERIVED (general): stand clear of the furniture NEAR-FACE (+ travel for an
            # openable's out-swing), not a flat gap from the contact. half_depth = the footprint
            # extent projected onto the approach axis; measured from foot_xy (the obstacle), which
            # may differ from the contact (an object resting ON TOP of a pillar).
            foot_c = (
                np.array(geometry.foot_xy, dtype=np.float64)
                if geometry.foot_xy is not None
                else pos[:2].astype(np.float64)
            )
            half_depth = (
                abs(fwd[0]) * geometry.foot_half[0]
                + abs(fwd[1]) * geometry.foot_half[1]
            )
            travel = geometry.travel if geometry.role is Role.OPENABLE else 0.0
            back = float(half_depth + geometry.clearance + travel)
            stand = (float(foot_c[0] - fwd[0] * back), float(foot_c[1] - fwd[1] * back))
            return stand, geometry.approach_yaw
        if gap is None:
            gap = e.stand_gap
            # A LOW graspable (floor / low shelf) needs the human RIGHT over it so a crouch+lean can reach
            # DOWN to it -- the default ~0.46 m over-a-table gap leaves a floor object beyond even a
            # crouched arm. Stand close so the pose-planner's squat+lean reaches. Gated on unified_geom
            # (the pose-planner path) so every other scene is byte-identical.
            if (
                getattr(self, "unified_geom", False)
                and e.role is Role.GRASPABLE
                and float(pos[2]) < self.floor_z + 0.5
            ):
                gap = min(gap, 0.25)
        stand = (float(pos[0] - fwd[0] * gap), float(pos[1] - fwd[1] * gap))
        return stand, geometry.approach_yaw

    def solid_boxes(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """AABBs (name, mn, mx) of every entity with a footprint (foot_half set) -- the furniture the
        reach must not push the arm through. center = foot_xy or xy; the box spans floor_z..top_z (or
        the entity z if top_z is None). Used by the approach-waypoint reach's arm-clearance check.

        HARMONY (gated on ``unified_geom`` + a built oracle): return the SHARED collision map's boxes
        instead -- REAL per-collider z-range (so the space UNDER a wall cabinet is free, not a solid
        floor->top pillar) sourced from the SAME geometry the gate/settle/validator use. Off -> the
        legacy per-entity visual-AABB pillars (byte-identical)."""
        cw = getattr(self, "cworld", None)
        if cw is not None and getattr(self, "unified_geom", False):
            return cw.boxes()
        boxes: list[tuple[str, np.ndarray, np.ndarray]] = []
        for e in self.entities.values():
            if e.foot_half is None:
                continue
            cx, cy = e.foot_xy if e.foot_xy is not None else e.xy
            hx, hy = e.foot_half
            top = e.top_z if e.top_z is not None else e.z
            mn = np.array([cx - hx, cy - hy, self.floor_z], dtype=np.float64)
            mx = np.array([cx + hx, cy + hy, top], dtype=np.float64)
            boxes.append((e.name, mn, mx))
        return boxes

    def set_held(self, name: str, is_held: bool = True) -> None:
        (self.held.add if is_held else self.held.discard)(name)

    def set_object_rest(self, name: str, xyz) -> None:
        """Update a SPEC-mode object's resting pose (after a PLACE) so grounding + the geometric guard stay
        correct for chained commands. In LIVE mode the USD bbox already reflects the moved object."""
        e = self.resolve(name)
        old_xy, old_z = e.xy, e.z
        e.xy = (float(xyz[0]), float(xyz[1]))
        e.z = float(xyz[2])
        if name in self.object_rest_pose:
            self.object_rest_pose[name] = np.asarray(
                self.object_rest_pose[name], dtype=np.float64
            ).copy()
            self.object_rest_pose[name][:3, 3] = (e.xy[0], e.xy[1], e.z)
        if self.stage is not None or old_xy is None or old_z is None:
            return
        dx, dy, dz = e.xy[0] - old_xy[0], e.xy[1] - old_xy[1], e.z - old_z
        targets = list(e.affordances.values())
        if e.target is not None and all(e.target is not target for target in targets):
            targets.append(e.target)
        for target in targets:
            target.xy = (target.xy[0] + dx, target.xy[1] + dy)
            target.z = float(target.z) + dz
            if target.foot_xy is not None:
                target.foot_xy = (target.foot_xy[0] + dx, target.foot_xy[1] + dy)
            if target.top_z is not None:
                target.top_z = float(target.top_z) + dz
            if target.drop_z is not None:
                target.drop_z = float(target.drop_z) + dz
            target.spaces = self._translated_spaces(target.spaces, dx, dy, dz)

    def set_object_rest_pose(self, name: str, pose) -> None:
        """Record an intentional full 6-DoF rest pose and keep grounding in sync."""
        value = np.asarray(pose, dtype=np.float64)
        if value.shape != (4, 4) or not bool(np.all(np.isfinite(value))):
            raise ValueError("object rest pose must be a finite 4x4 transform")
        self.object_rest_pose[name] = value.copy()
        self.set_object_rest(name, value[:3, 3])

    # ---- declarative effects (the executor calls these as actions complete; trusted-execution state) ---
    def declare_pick(self, obj: str) -> None:
        """Object is now held: add to held, drop any supporting-surface relation."""
        self.held.add(obj)
        self.on.pop(obj, None)
        self.inside.pop(obj, None)

    def declare_place(self, obj: str, surface: str | None) -> None:
        """Object released: clear held; if placed on a named surface, record ON(obj, surface)."""
        self.held.discard(obj)
        self.on.pop(obj, None)
        self.inside.pop(obj, None)
        if surface is not None:
            self.on[obj] = surface

    def declare_open(self, name: str) -> None:
        self.open_objs.add(name)

    def declare_close(self, name: str) -> None:
        self.open_objs.discard(name)

    def bind_openable_controllers(self, controllers: dict[str, dict]) -> None:
        self._openable_controllers = dict(controllers)

    def declare_turn_on(self, name: str) -> None:
        if self.affordance(name, Role.TOGGLEABLE) is None:
            raise ValueError(f"{name!r} is not toggleable")
        self.running.add(name)

    def declare_turn_off(self, name: str) -> None:
        if self.affordance(name, Role.TOGGLEABLE) is None:
            raise ValueError(f"{name!r} is not toggleable")
        self.running.discard(name)

    def carve_door_open(self, name: str) -> None:
        """Replace a closed door's nav footprint with its swung-open leaf footprint."""
        grid = getattr(self, "nav_grid", None)
        di = (self.doors or {}).get(name)
        if grid is None or di is None or getattr(grid, "door_id", None) is None:
            return
        mask = grid.door_id == int(di.prim_index)
        if not mask.any():
            return
        entity = self.entities.get(name)
        if entity is not None:
            entity.meta["_nav_closed_door_cells"] = [
                (int(i), int(j)) for j, i in np.argwhere(mask)
            ]
        if getattr(grid, "base_occ", None) is not None:
            grid.occ[mask] = grid.base_occ[mask]
        else:
            grid.occ[mask] = False
        grid.door_cost[mask] = 0.0
        grid.door_id[mask] = -1
        import nav_grid

        grid.sdf = nav_grid._two_sided_sdf(grid.occ, grid.cell)
        self.mark_leaf_obstacle(name)

    def restore_door(self, name: str) -> None:
        """Remove the swung-open leaf obstacle and restore the exact closed-door cells."""
        grid = getattr(self, "nav_grid", None)
        di = (self.doors or {}).get(name)
        if grid is None or di is None:
            return
        import nav_grid

        self.unmark_leaf_obstacle(name)
        entity = self.entities.get(name)
        cells = (
            list((entity.meta or {}).get("_nav_closed_door_cells") or ())
            if entity
            else []
        )
        if cells:
            for i, j in cells:
                if not grid.in_bounds(i, j):
                    continue
                grid.occ[j, i] = True
                if grid.door_id is not None:
                    grid.door_id[j, i] = int(di.prim_index)
                if grid.door_cost is not None:
                    grid.door_cost[j, i] = nav_grid.DOOR_COST_M
            entity.meta["_nav_closed_door_cells"] = []
        else:
            cx, cy = di.center_xy
            ci, cj = grid.world_to_cell(cx, cy)
            r = max(1, int(round(0.5 / grid.cell)))
            for dj in range(-r, r + 1):
                for dcol in range(-r, r + 1):
                    i, j = ci + dcol, cj + dj
                    if grid.in_bounds(i, j):
                        grid.occ[j, i] = True
        grid.sdf = nav_grid._two_sided_sdf(grid.occ, grid.cell)

    def mark_leaf_obstacle(self, name: str) -> None:
        """After a swing CONTAINER (fridge/cabinet) opens, BLOCK its OPEN door leaf/leaves' footprint in the
        live nav grid so a later planned walk routes AROUND the open door instead of THROUGH it. The swung
        footprint is computed KINEMATICALLY (semantic_executor._open_leaf_boxes) -- the leaf prims are still
        closed at frame-gen. Records the cells it NEWLY blocks so unmark_leaf_obstacle frees exactly those on
        close. No-op without a nav grid / leaves (byte-identical). Rebuilds the SDF."""
        grid = getattr(self, "nav_grid", None)
        e = self.entities.get(name)
        if grid is None or e is None or not getattr(e, "leaf_paths", None):
            return
        if (e.meta or {}).get("_nav_leaf_cells"):
            return  # already marked (a repeated [Open]) -> keep the recorded cells for a correct unmark
        import semantic_executor  # the swung-leaf geometry lives with the swing math

        boxes = semantic_executor._open_leaf_boxes(self, e)
        if not boxes:
            return
        import nav_grid
        from init_validation import NAV_INFLATE_M

        pad = float(
            NAV_INFLATE_M if getattr(grid, "inflate", None) is None else grid.inflate
        )
        counts = getattr(grid, "open_leaf_count", None)
        if counts is None or counts.shape != grid.occ.shape:
            counts = np.zeros(grid.occ.shape, dtype=np.int16)
            grid.open_leaf_count = counts
        covered = set()
        for _nm, mn, mx in boxes:
            i0, j0 = grid.world_to_cell(float(mn[0]) - pad, float(mn[1]) - pad)
            i1, j1 = grid.world_to_cell(float(mx[0]) + pad, float(mx[1]) + pad)
            for j in range(min(j0, j1), max(j0, j1) + 1):
                for i in range(min(i0, i1), max(i0, i1) + 1):
                    if not grid.in_bounds(i, j):
                        continue
                    if grid.base_occ is None and grid.occ[j, i] and counts[j, i] == 0:
                        continue
                    covered.add((i, j))
        for i, j in covered:
            counts[j, i] += 1
            grid.occ[j, i] = True
        e.meta["_nav_leaf_cells"] = sorted(covered)
        grid.sdf = nav_grid._two_sided_sdf(grid.occ, grid.cell)

    def unmark_leaf_obstacle(self, name: str) -> None:
        """Undo mark_leaf_obstacle after CloseDoor: free exactly the cells the open leaf had blocked so the
        grid returns to its pre-open state. No-op if the container was never marked."""
        grid = getattr(self, "nav_grid", None)
        e = self.entities.get(name)
        if grid is None or e is None:
            return
        cells = (e.meta or {}).get("_nav_leaf_cells") or []
        if not cells:
            return
        counts = getattr(grid, "open_leaf_count", None)
        for i, j in cells:
            if not grid.in_bounds(i, j):
                continue
            if counts is not None and counts[j, i] > 0:
                counts[j, i] -= 1
            if counts is not None and counts[j, i] > 0:
                continue
            base_blocked = bool(grid.base_occ is not None and grid.base_occ[j, i])
            closed_door = bool(
                grid.door_id is not None and int(grid.door_id[j, i]) >= 0
            )
            grid.occ[j, i] = base_blocked or closed_door
        e.meta["_nav_leaf_cells"] = []
        import nav_grid

        grid.sdf = nav_grid._two_sided_sdf(grid.occ, grid.cell)

    def declare_sit(self, seat: str) -> None:
        self.sitting_on = seat

    def declare_stand(self) -> None:
        self.sitting_on = None

    def declare_cut(self, obj: str) -> None:
        """A CONTACT cut action completed on ``obj`` (symbolic, trusted-execution)."""
        self.cut.add(obj)

    def declare_cook(self, obj: str) -> None:
        """A COOK action completed on ``obj`` (symbolic, trusted-execution)."""
        self.cooked.add(obj)

    def declare_inside(self, obj: str, container: str) -> None:
        """POUR: ``obj`` (contents) is now INSIDE ``container`` (symbolic, trusted-execution)."""
        self.held.discard(obj)
        self.on.pop(obj, None)
        self.inside.pop(obj, None)
        self.inside[obj] = container

    def declare_attached(
        self,
        child: str,
        parent: str,
        child_point: str,
        parent_point: str,
        cp_idx: int = 0,
        pp_idx: int = 0,
    ) -> None:
        """A part MATED (assembly): record child->parent + explicit connector points (child_point, cp_idx,
        parent_point, pp_idx). ATTACHED is NOT trusted here -- _relations_attached_live re-reads the LIVE
        world poses, so a wrong-order weld that later decouples (the parent moved, the fixed-pose child did
        not follow) fails honestly."""
        self.held.discard(child)
        self.on.pop(child, None)
        self.attached[child] = parent
        self.attach_point[child] = (child_point, int(cp_idx), parent_point, int(pp_idx))

    def declare_detached(self, child: str, parent: str) -> None:
        """Remove one leaf mate after the part has been physically grasped for repair."""
        actual_parent = self.attached.get(child)
        if actual_parent != parent:
            raise ValueError(
                f"{child!r} is attached to {actual_parent!r}, not requested parent {parent!r}"
            )
        dependents = sorted(c for c, p in self.attached.items() if p == child)
        if dependents:
            raise ValueError(
                f"cannot detach {child!r} while it carries attached children {dependents}"
            )
        self.attached.pop(child, None)
        self.attach_point.pop(child, None)
        self.assembly_pose.pop(child, None)
        self.refresh_assembly_affordances()

    def configure_goal_predicates(self, rows) -> None:
        """Register only the physical state pairs that the current task will query."""
        self.goal_predicates_configured = True
        for row in rows or ():
            pred = str(row.get("pred") or "").lower()
            subject = str(row.get("subject") or "")
            target = str(row.get("target") or "")
            if pred and subject:
                self.goal_predicates.add((pred, subject, target))
        self.configure_observation_predicates(rows)

    def configure_observation_predicates(self, rows) -> None:
        """Enable physical readers without changing the task's scored goal set."""
        for row in rows or ():
            pred = str(row.get("pred") or "").lower()
            subject = str(row.get("subject") or "")
            target = str(row.get("target") or "")
            if pred == "near" and subject and target:
                self.tracked_near.add((subject, target))
            elif pred in {"raised", "is_raised"} and subject:
                self.tracked_raised.add(subject)
            elif pred in {"unmoved", "is_unmoved", "remain_unmoved"} and subject:
                self.tracked_unmoved.add(subject)

    def note_body_state(self, state) -> None:
        """Publish the produced body's physical endpoint for the sitting reader."""
        self.actor_xy = tuple(float(value) for value in state.root_xy)
        self.actor_seated = self.sitting_on is not None

    def note_motion(self, names=None) -> None:
        """Accumulate maximum pose deviation so UNMOVED cannot pass after move-and-return."""
        baseline = (self._episode_baseline or {}).get("named_pose", {})
        selected = set(names or baseline)
        for name in selected:
            entity = self.entities.get(name)
            reference = baseline.get(name)
            current = self._entity_world_pose(entity) if entity is not None else None
            if reference is None or current is None:
                continue
            distance, angle = self._transform_delta(reference, current)
            self.motion_max_m[name] = max(self.motion_max_m.get(name, 0.0), distance)
            self.motion_max_deg[name] = max(self.motion_max_deg.get(name, 0.0), angle)

    @staticmethod
    def _transform_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        distance = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
        relative = a[:3, :3].T @ b[:3, :3]
        cosine = (float(np.trace(relative)) - 1.0) * 0.5
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        return distance, angle

    def declare_moved(self, subject: str, destination: str) -> None:
        self.movement_targets[subject] = destination
        self.tracked_near.add((subject, destination))
        self.note_motion()

    def assembly_component(self, name: str) -> set[str]:
        edges = {**self.assembly_required, **self.attached}
        component, frontier = {name}, [name]
        while frontier:
            node = frontier.pop()
            neighbors = {
                other
                for child, parent in edges.items()
                for other in (
                    parent if child == node else None,
                    child if parent == node else None,
                )
                if other
            }
            for other in neighbors - component:
                component.add(other)
                frontier.append(other)
        return component

    def _assembly_complete(self, name: str) -> bool:
        component = self.assembly_component(name)
        required = {
            child: parent
            for child, parent in self.assembly_required.items()
            if child in component or parent in component
        }
        return bool(required) and all(
            self.attached.get(c) == p for c, p in required.items()
        )

    def refresh_assembly_affordances(self) -> None:
        """Activate declared post-assembly roles only on a complete physical assembly."""
        for name, entity in self.entities.items():
            specs = tuple((entity.meta or {}).get("post_assembly_affordances") or ())
            active = set((entity.meta or {}).get("_active_assembly_affordances") or ())
            wanted = set(specs) if self._assembly_complete(name) else set()
            if (entity.meta or {}).get("assembly_affordance_requires_raised", True):
                if not self.is_raised(name):
                    wanted.clear()
            for value in sorted(active - wanted):
                role = Role(value)
                entity.affordances.pop(role, None)
            for value in sorted(wanted - active):
                self._activate_assembly_affordance(name, Role(value))
            entity.meta["_active_assembly_affordances"] = sorted(wanted)

    def _activate_assembly_affordance(self, name: str, role: Role) -> None:
        if role in (Role.PUSHABLE, Role.PULLABLE):
            self.add_mobility(
                name,
                push=role is Role.PUSHABLE,
                pull=role is Role.PULLABLE,
            )
            return
        if role not in (Role.SURFACE, Role.CONTAINER, Role.TOGGLEABLE, Role.SEAT):
            return
        mn, mx = self.object_aabb(name)
        cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        common = {
            "xy": (cx, cy),
            "half": (
                max(0.03, 0.5 * (mx[0] - mn[0])),
                max(0.03, 0.5 * (mx[1] - mn[1])),
            ),
            "foot_xy": (cx, cy),
            "foot_half": (0.5 * (mx[0] - mn[0]), 0.5 * (mx[1] - mn[1])),
            "top_z": float(mx[2]),
        }
        if role is Role.SURFACE:
            self.register_affordance(name, role, z=float(mx[2]), **common)
            return
        if role is Role.TOGGLEABLE:
            self.register_affordance(
                name,
                role,
                z=float(mx[2]),
                meta={"press_dir": (0.0, 0.0, -1.0)},
                **common,
            )
            return
        if role is Role.SEAT:
            self.register_affordance(name, role, z=float(mx[2]), **common)
            return
        bottom = float(mn[2]) + 0.05
        self.register_affordance(
            name,
            role,
            z=bottom,
            height=max(0.05, float(mx[2]) - bottom - 0.05),
            drop_z=bottom,
            **common,
        )

    def is_raised(self, name: str) -> bool:
        entity = self.entities.get(name)
        if entity is None:
            return False
        current = self._entity_world_pose(entity)
        baseline = (self._episode_baseline or {}).get("named_pose", {}).get(name)
        baseline = (
            baseline if baseline is not None else entity.meta.get("raise_source_pose")
        )
        if current is None or baseline is None:
            return False
        _distance, angle = self._transform_delta(np.asarray(baseline), current)
        return angle >= float(entity.meta.get("raised_angle_threshold_deg", 45.0))

    def is_unmoved(self, name: str) -> bool:
        self.note_motion((name,))
        return (
            self.motion_max_m.get(name, float("inf")) <= 0.02
            and self.motion_max_deg.get(name, float("inf")) <= 3.0
        )

    def attachment_status(self, child: str) -> dict[str, object]:
        """JSON-safe live residual for one declared mate, used by error/recovery manifests."""
        parent = self.attached.get(child)
        status: dict[str, object] = {
            "child": child,
            "declared": parent is not None,
            "parent": parent,
            "physical_attached": False,
            "position_error_m": None,
            "orientation_error_deg": None,
        }
        if parent is None or self.stage is None:
            return status
        ec = self.entities.get(child)
        ep = self.entities.get(parent)
        if ec is None or ep is None or not ec.prim_path or not ep.prim_path:
            return status
        cp, cpi, pp, ppi = self.attach_point.get(child, (None, 0, None, 0))
        a_list = ec.attach.get(cp) if cp else None
        b_list = ep.attach.get(pp) if pp else None
        if (
            not a_list
            or not b_list
            or not (0 <= cpi < len(a_list))
            or not (0 <= ppi < len(b_list))
        ):
            return status
        wc = self._live_world_col(ec.prim_path)
        wp = self._live_world_col(ep.prim_path)
        if wc is None or wp is None:
            return status
        a_world = wc @ _mat_from_pos_quat(a_list[cpi]["pos"], a_list[cpi]["quat"])
        b_world = wp @ _mat_from_pos_quat(b_list[ppi]["pos"], b_list[ppi]["quat"])
        position_error, orientation_error = _pose_error(a_world, b_world)
        status.update(
            {
                "child_point": cp,
                "child_point_idx": int(cpi),
                "parent_point": pp,
                "parent_point_idx": int(ppi),
                "position_error_m": round(position_error, 6),
                "orientation_error_deg": round(math.degrees(orientation_error), 3),
                "physical_attached": _poses_coincide(a_world, b_world),
            }
        )
        return status

    def _near_holds(self, subject: str, target: str) -> bool:
        if not self._has_relation_extent(subject) or not self._has_relation_extent(
            target
        ):
            return False
        smn, smx = self._relation_aabb(subject)
        tmn, tmx = self._relation_aabb(target)
        dx = max(float(tmn[0]) - float(smx[0]), float(smn[0]) - float(tmx[0]), 0.0)
        dy = max(float(tmn[1]) - float(smx[1]), float(smn[1]) - float(tmx[1]), 0.0)
        entity = self.entities.get(subject)
        threshold = float(
            (entity.meta if entity is not None else {}).get("near_threshold_m", 0.75)
        )
        return math.hypot(dx, dy) <= threshold

    def _has_relation_extent(self, name: str) -> bool:
        return name in self.entities or (name == "wearer" and self.actor_xy is not None)

    def _relation_aabb(self, name: str):
        if name == "wearer" and self.actor_xy is not None:
            x, y = self.actor_xy
            return ((x, y, self.floor_z), (x, y, self.floor_z + 1.8))
        entity = self.resolve(name)
        if self.stage is not None and entity.prim_path:
            return self.object_aabb(name)
        x, y = entity.xy or (0.0, 0.0)
        z = float(entity.z or 0.0)
        hx, hy = entity.half
        return ((x - hx, y - hy, z), (x + hx, y + hy, z + entity.height))

    def _physically_sitting(self) -> bool:
        if not self.actor_seated or self.sitting_on is None or self.actor_xy is None:
            return False
        seat = self.target_view(self.sitting_on, Role.SEAT)
        if seat is None or seat.xy is None:
            return False
        return (
            abs(self.actor_xy[0] - seat.xy[0]) <= seat.half[0] + 0.15
            and abs(self.actor_xy[1] - seat.xy[1]) <= seat.half[1] + 0.15
        )

    def _runtime_state_relations(self, source: str) -> set:
        relations: set = set()
        near_pairs = self.tracked_near | set(self.movement_targets.items())
        relations |= {
            (Pred.NEAR, subject, target)
            for subject, target in near_pairs
            if self._near_holds(subject, target)
        }
        relations |= {
            (Pred.RAISED, name, None)
            for name in self.tracked_raised
            if self.is_raised(name)
        }
        relations |= {
            (Pred.UNMOVED, name, None)
            for name in self.tracked_unmoved
            if self.is_unmoved(name)
        }
        if source == "physics" and self._physically_sitting():
            relations.add((Pred.SITTING, self.sitting_on, None))
        return relations

    # ---- relations --------------------------------------------------------------------------------
    def relations(self, source: str = "declared", contact_report=None) -> set:
        """Symbolic relation set {(Pred,a,b)} the goal checker reads. Three sources:
          "declared" (default, trusted execution): the executor-declared held/on/open -- VH-style, exact.
          "geometric": ON/INSIDE from registry positions + OPEN from the joint value (offline guard).
          "physics": ON from the live stage (``_relations_on_live``; also from PhysX contact pairs when
                     a ``contact_report`` is supplied, which no scored call site does) + INSIDE from
                     scene-query overlap + ATTACHED + OPEN from the DOF (needs a live stage).
        HELD is always the held-set (exact for our kinematic carry)."""
        rel: set = {(Pred.HELD, o, None) for o in self.held}
        rel |= {(Pred.RUNNING, device, None) for device in self.running}
        rel |= {(Pred.CUT, o, None) for o in self.cut}
        rel |= {(Pred.COOKED, o, None) for o in self.cooked}
        if source == "declared":
            rel |= {(Pred.ON, o, s) for o, s in self.on.items()}
            rel |= {(Pred.OPEN, x, None) for x in self.open_objs}
            if self.sitting_on is not None:
                rel.add((Pred.SITTING, self.sitting_on, None))
            rel |= {(Pred.INSIDE, o, c) for o, c in self.inside.items()}
        elif source == "geometric":
            rel |= self._relations_geom()
            rel |= self._relations_open()
        elif source == "physics":
            if contact_report is not None:
                rel |= self._relations_contacts(contact_report)
            rel |= self._relations_on_live()
            rel |= self._relations_inside_live()
            rel |= self._relations_attached_live()
            rel |= self._relations_open()
        else:
            raise ValueError(f"relations: unknown source {source!r}")
        rel |= self._runtime_state_relations(source)
        if self.held:
            rel = {
                row
                for row in rel
                if not (row[0] in {Pred.ON, Pred.INSIDE} and row[1] in self.held)
            }
        return rel

    def _role_targets(self, role: Role) -> list[tuple[Entity, Entity]]:
        targets = []
        for entity in self.entities.values():
            target = self.target_view(entity.name, role)
            if target is not None:
                targets.append((entity, target))
        return targets

    def _relations_geom(self) -> set:
        """Geometric ON/INSIDE from the registry positions (offline/tests + a no-PhysX fallback)."""
        rel: set = set()
        objs = [
            e
            for e in self.entities.values()
            if e.role is Role.GRASPABLE and e.name not in self.held
        ]
        surfs = self._role_targets(Role.SURFACE)
        conts = self._role_targets(Role.CONTAINER)
        for o in objs:
            op = self.object_pos(o.name)
            for owner, s in surfs:
                if o.name == owner.name:
                    continue
                (sx, sy), stop = self.surface_top(owner.name)
                if (
                    abs(op[2] - stop) < 0.06
                    and abs(op[0] - sx) <= s.half[0]
                    and abs(op[1] - sy) <= s.half[1]
                ):
                    rel.add((Pred.ON, o.name, owner.name))
            for owner, c in conts:
                if o.name == owner.name:
                    continue
                cx, cy = c.xy
                if (
                    abs(op[0] - cx) <= c.half[0]
                    and abs(op[1] - cy) <= c.half[1]
                    and c.z <= op[2] <= c.z + c.height
                ):
                    rel.add((Pred.INSIDE, o.name, owner.name))
        return rel

    def _relations_open(self) -> set:
        """OPEN(openable) from the articulation DOF (live) or the spec joint_val, past open_thresh."""
        rel: set = set()
        for e in self.entities.values():
            controller_bound = e.name in self._openable_controllers
            if (
                not controller_bound and e.role is not Role.OPENABLE
            ) or e.joint_name is None:
                continue
            val = self._live_dof(e)
            if val is not None and abs(val) > e.open_thresh:
                rel.add((Pred.OPEN, e.name, None))
        return rel

    def _prim_to_entity(self, path: str) -> Entity | None:
        """Prefix-match a collider prim path to a registered entity (colliders live on child meshes)."""
        for e in self.entities.values():
            if e.prim_path and (
                path == e.prim_path or path.startswith(e.prim_path.rstrip("/") + "/")
            ):
                return e
        return None

    def _relations_contacts(self, contact_report) -> set:
        """LIVE ON (obj<->surface contact pair) from a PhysX contact report -- same decode as
        physics_human.classify_contacts, generalized to the registry's roles. (HELD comes from the
        held-set; obj<->hand contact could also feed it but the executor's carry is exact.)"""
        from pxr import PhysicsSchemaTools

        headers, _data = contact_report
        rel: set = set()
        for h in headers:
            ea = self._prim_to_entity(str(PhysicsSchemaTools.intToSdfPath(h.collider0)))
            eb = self._prim_to_entity(str(PhysicsSchemaTools.intToSdfPath(h.collider1)))
            if ea is None or eb is None:
                continue
            for obj, other in ((ea, eb), (eb, ea)):
                if (
                    obj.role is Role.GRASPABLE
                    and obj.name != other.name
                    and self.affordance(other.name, Role.SURFACE) is not None
                ):
                    rel.add((Pred.ON, obj.name, other.name))
        return rel

    def _relations_inside_live(self) -> set:
        """LIVE placement success: a graspable counts as placed ON/IN a container when its settled xy lies
        within the container's xy footprint. The post-action physics pass separately guarantees a valid,
        non-penetrating rest, so xy-over-the-target + a settled rest = a successful put. General and
        surface-friendly -- no 3D containment box and no interior/shelf assumption: an object dropped in the
        middle of a bin falls through the opening to the bottom, one on a surface top rests there, and either
        way its xy stays over the target. (A specific multi-height nested 'inside' -- a chosen cabinet shelf
        -- is a later addition.)"""
        conts = [row for row in self._role_targets(Role.CONTAINER) if row[0].prim_path]
        objs = [
            e
            for e in self.entities.values()
            if e.role is Role.GRASPABLE and e.prim_path
        ]
        rel: set = set()
        if not conts or not objs:
            return rel
        from pxr import Usd, UsdGeom

        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        for owner, _target in conts:
            # The CONTAINER-side twin of the object guard below: this loop reads both sides of the
            # INSIDE relation from a stored prim_path, and only the object half was ever checked. A
            # container whose prim is gone cannot hold anything, so withhold its relations rather than
            # score against an empty bbox at the origin (which would read as a container at (0,0,0)
            # and could match a fallen object).
            _cprim = (
                self.stage.GetPrimAtPath(owner.prim_path)
                if self.stage is not None
                else None
            )
            if _cprim is None or not _cprim.IsValid():
                self._stale_prim(owner.prim_path)
                continue
            _rng = bc.ComputeWorldBound(_cprim).ComputeAlignedRange()
            _mn, _mx = _rng.GetMin(), _rng.GetMax()
            ccx, ccy = 0.5 * (_mn[0] + _mx[0]), 0.5 * (_mn[1] + _mx[1])
            chx, chy = 0.5 * (_mx[0] - _mn[0]), 0.5 * (_mx[1] - _mn[1])
            # Interior z-band from the container's OUTER bbox: an object counts INSIDE only when its base
            # sits BELOW the outer top (it is NOT resting on the lid / outer surface) and not far below the
            # outer bottom -> kills the "floating on the outer top" false pass. Use the OUTER bottom (_mn[2]),
            # NOT the registered interior floor c.z: c.z sits a few cm above the true bottom, so a bin object
            # that settles onto the real floor would be wrongly excluded. General for bins + shelved cabinets.
            c_floor = float(_mn[2])
            c_top = float(_mx[2])
            _band = 0.05 * max(0.1, c_top - c_floor)
            open_top_drop_z = (
                float(_target.drop_z)
                if (getattr(_target, "meta", None) or {}).get("open_top")
                and not (getattr(_target, "spaces", None) or ())
                and _target.drop_z is not None
                else None
            )
            for o in objs:
                if (
                    o.name == owner.name
                ):  # a dual-role bowl is both a container AND an object -> not "in itself"
                    continue
                _obb = self._live_bbox(o.prim_path)
                if _obb is None:
                    # A body whose prim is gone cannot be measured INSIDE anything. Skipping it is the
                    # conservative reading -- it withholds a relation rather than inventing one from the
                    # spec pose, which would be scoring geometry that is not on the stage.
                    continue
                (ox, oy), oz0, oz1 = _obb
                xy_in = abs(ox - ccx) <= chx and abs(oy - ccy) <= chy
                # LOWER bound, tilt-robust. oz0 comes from ComputeWorldBound, which rotates the object's
                # axis-aligned BOX rather than re-fitting its vertices, so the reported min-z is a rotated
                # box CORNER that dives as the object tilts -- with no relation to where the body actually
                # is. Measured on bt_picking_up_trash: a can resting on the bin's own floor, 139 mm below
                # the rim and 6 mm off the bin's axis, tilts 39 deg; its true lowest vertex is -0.0146 but
                # oz0 reads -0.0287, i.e. 14.1 mm low against a 13.9 mm band -> scored NOT inside, while
                # the two cans STACKED ON TOP OF IT (tilt 12.8/24.9 deg, error 2.8/9.8 mm) both scored.
                # The AABB CENTRE is rotation-invariant, so accept that as an alternative: it measures
                # where the body is, not where a corner points. Written as a DISJUNCTION so nothing that
                # passes today can start failing. The UPPER bound is untouched -- it is the guard against
                # an object resting on the lid/rim, and widening _band (the other obvious fix) would have
                # LOWERED that upper threshold and made rim false-positives MORE likely.
                z_in = (oz0 <= c_top - _band) and (
                    oz0 >= c_floor - _band or 0.5 * (oz0 + oz1) >= c_floor
                )
                if z_in and open_top_drop_z is not None:
                    pose = self._live_world_col(o.prim_path)
                    if pose is None:
                        z_in = False
                    else:
                        base_offset = max(0.0, float(pose[2, 3]) - float(oz0))
                        release_base_z = open_top_drop_z - base_offset
                        floor_rest_slack = 0.5 * max(0.0, float(_target.height))
                        settled_base_ceiling = max(
                            float(_target.z)
                            + floor_rest_slack
                            + _INSIDE_DROP_PLANE_TOL_M,
                            release_base_z - _INSIDE_DROP_PLANE_TOL_M,
                        )
                        z_in = oz0 <= settled_base_ceiling
                if xy_in and z_in:
                    rel.add((Pred.INSIDE, o.name, owner.name))
        return rel

    def _relations_on_live(self) -> set:
        """LIVE ON: a graspable rests on a SURFACE when its settled xy is over the surface footprint and
        its underside is at the surface top.

        THIS IS WHY ``on`` WAS UNSCORABLE. ``relations(source="physics")`` is what every scored path
        calls -- ``episode_runtime`` twice, ``isaac_env`` six times -- and the only ON producer that branch
        had was ``_relations_contacts``, which needs a PhysX ``contact_report`` that all eight call sites
        pass as ``None``. So the relation was never emitted anywhere, and ``spec_validate.SCORED_PREDS``
        had ``on`` excluded to stop specs from silently scoring 0.00 forever. Nothing was missing except
        the reading: the surfaces have been registered with a footprint and a top since
        ``task_world_builder.register_supports``, and ``_relations_geom`` has computed exactly this from
        registry poses all along -- it was just never wired into the branch that scores.

        Measured off the live stage rather than the registry, and the twin of ``_relations_inside_live``
        in every respect that matters: no contact report, no 3D containment box, no per-scene tuning.
        The vertical test is tilt-robust the same way -- ``oz0`` from ``ComputeWorldBound`` is a rotated
        box CORNER that dives as the body leans, so the band grows with the body's own height instead of
        being a flat constant that a tilted bottle would fall out of."""
        surfs = [row for row in self._role_targets(Role.SURFACE) if row[1].half]
        objs = [
            e
            for e in self.entities.values()
            if e.role is Role.GRASPABLE and e.prim_path and e.name not in self.held
        ]
        rel: set = set()
        for owner, s in surfs:
            (sx, sy), top = self.surface_top(owner.name)
            for o in objs:
                if o.name == owner.name:  # a dual-role tray is not ON itself
                    continue
                obb = self._live_bbox(o.prim_path)
                if obb is None:
                    # No live geometry -> withhold the relation rather than invent one from the spec
                    # pose. Same conservative reading as the INSIDE scorer.
                    if os.environ.get("ON_DEBUG"):
                        print(
                            f"[on-dbg] {o.name}: NO LIVE BBOX at {o.prim_path!r} -- cannot read ON"
                        )
                    continue
                (ox, oy), oz0, oz1 = obb
                band = max(_ON_BAND_M, _ON_TILT_FRAC * (oz1 - oz0))
                dx, dy, dz = abs(ox - sx), abs(oy - sy), abs(oz0 - top)
                ok = dx <= s.half[0] and dy <= s.half[1] and dz <= band
                if ok:
                    rel.add((Pred.ON, o.name, owner.name))
                elif os.environ.get("ON_DEBUG"):
                    # Why a placed object did not read ON. Three numbers, one line: the relation is
                    # an AND of three windows and the silent version of this cost a whole build.
                    print(
                        f"[on-dbg] {o.name} vs {owner.name}: dx={dx:.3f}/{s.half[0]:.3f} "
                        f"dy={dy:.3f}/{s.half[1]:.3f} dz={dz:.3f}/{band:.3f} "
                        f"(obj base z={oz0:.3f}, top z={top:.3f})"
                    )
        return rel

    def _relations_attached_live(self) -> set:
        """LIVE ATTACHED (assembly): for each declared (child -> parent) mate, transform the child's and parent's
        attach points by the parts' LIVE world matrices and emit (ATTACHED, child, parent) IFF the two world
        poses coincide within OmniGibson's AttachedTo tolerance (5 cm / 15 deg). Explicit connector pairing:
        attach_point[child] = (child_point, cp_idx, parent_point, pp_idx). Honest: a wrong-order weld (parent
        later moved, the fixed-pose child did not follow) decouples -> no emit. Empty self.attached (every
        existing scene) -> returns set() -> relations() stays byte-identical."""
        if not self.attached or self.stage is None:
            return set()
        rel: set = set()
        for child, parent in self.attached.items():
            if self.attachment_status(child)["physical_attached"]:
                rel.add((Pred.ATTACHED, child, parent))
        return rel

    def _live_world_col(self, prim_path: str):
        """A prim's local-to-world matrix as a COLUMN-vector 4x4 numpy (USD is row-vector -> transpose)."""
        from pxr import Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return None
        m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        return np.asarray(m, dtype=np.float64).T

    def _stale_prim(self, prim_path: str) -> None:
        """Record + announce a prim_path that no longer resolves, NAMING the entities that hold it.

        ``ComputeWorldBound`` on a null prim raises ``Invalid prim: null`` -- a message that names
        neither the path nor the entity, so the only way to find out which of a scene's ~40 entities
        carries the dead path was to reproduce the build under a debugger. Measured cost: it took down
        `T2-HSF-01` entirely and burned several episodes of another agent's experiment before it was
        handed on with only a traceback. The path and the owning key are both in hand right here."""
        keys = sorted(
            k
            for k, e in (self.entities or {}).items()
            if getattr(e, "prim_path", None) == prim_path
        )
        first = prim_path not in self._stale_paths
        self._stale_paths.setdefault(prim_path, keys)
        if first:
            # Say WHAT is wrong and WHO holds it; each caller announces its own recovery, because they
            # differ (the readers fall back to the spec pose, the INSIDE scorer withholds the relation,
            # and register_from_prim refuses outright).
            print(
                f"[scene-state] STALE prim_path {prim_path!r} does not resolve on this stage "
                f"-- held by {keys or ['<no entity, path passed in directly>']}. No live geometry "
                f"can be read for it, so anything derived from it is authored, not measured."
            )

    def stale_prim_paths(self) -> dict:
        """``{prim_path: [entity keys]}`` for every dead path seen. Empty = every path resolved."""
        return dict(self._stale_paths)

    # ---- live Isaac readers (lazy pxr/omni) -------------------------------------------------------
    def _live_bbox(self, prim_path: str):
        """World AABB of a prim -> ((cx,cy), zmin, zmax), or **None** when the path does not resolve.

        Uses UsdGeom.BBoxCache (authored + simulated via Fabric is read elsewhere; for kinematic-driven
        objects the USD xform is current).

        ``None`` is the same contract ``_live_xform`` above already uses, and for the same reason: a
        stale ``prim_path`` is not an emergency, it is the *absence of a live reading*, and every caller
        already has a spec-based fallback for exactly that (it is the branch they take when there is no
        stage at all). Raising instead -- which is what happened before, from deep inside USD -- turned a
        missing measurement into a dead build. ⚠️ Callers MUST test for None: two of the four used to
        unpack the result blind."""
        # The validity check needs no pxr -- GetPrimAtPath/IsValid are stage methods -- so it runs
        # BEFORE the import. That keeps the stale path free of any USD dependency and, usefully, makes
        # this branch testable without an Isaac runtime.
        prim = self.stage.GetPrimAtPath(prim_path) if self.stage is not None else None
        if prim is None or not prim.IsValid():
            self._stale_prim(prim_path)
            return None
        from pxr import Usd, UsdGeom

        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        return (
            (0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])),
            float(mn[2]),
            float(mx[2]),
        )

    def _live_dof(self, e: Entity) -> float | None:
        """Current kinematic controller value for an OPENABLE (drawer/door)."""
        controller = self._openable_controllers.get(e.name)
        if controller is None:
            return e.joint_val
        field_name = "cur_angle" if controller.get("kind") == "revolute" else "cur_disp"
        value = controller.get(field_name)
        return float(value) if value is not None else e.joint_val


# =====================================================================================================
# Offline selftest: grounding + geometric relations (no Isaac).
# =====================================================================================================
def _selftest() -> int:
    ok = True

    def _chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

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

    from isaac_goal import check_goal, on

    # Grounding: object pos + stand pose in front (0.46 m, facing yaw=0 -> stand at +Y of the object).
    p = ss.object_pos("cup")
    _chk(np.allclose(p, [0.0, -1.36, 0.75]), f"object_pos(cup)={p.round(3).tolist()}")
    stand, yaw = ss.stand_pose_for("cup", gap=0.46)
    _chk(
        abs(stand[1] - (-0.90)) < 1e-6 and abs(stand[0]) < 1e-6 and yaw == 0.0,
        f"stand_pose_for(cup)={tuple(round(v, 3) for v in stand)} yaw={yaw}",
    )

    # Geometry-derived stand-back (general): an object resting ON a 0.4 m-deep pillar -> stand clear
    # of the pillar NEAR-FACE + clearance, NOT a flat gap from the object center (this is the
    # highreach body-in-pillar fix; foot_c=(0.6,-0.6), half_depth=0.2, clearance=0.5 -> gap 0.7).
    ss.register(
        "hi_obj",
        Role.GRASPABLE,
        xy=(0.6, -0.6),
        z=1.34,
        approach_yaw=0.0,
        foot_xy=(0.6, -0.6),
        foot_half=(0.225, 0.2),
        top_z=1.3,
        clearance=0.5,
    )
    gstand, _ = ss.stand_pose_for("hi_obj")
    _chk(
        abs(gstand[1] - 0.10) < 1e-6 and abs(gstand[0] - 0.6) < 1e-6,
        f"geom stand_pose_for(hi_obj)={tuple(round(v, 3) for v in gstand)} (expect (0.6, 0.1))",
    )
    ss.entities.pop(
        "hi_obj"
    )  # keep the rest of the selftest's declarative checks unchanged

    # Declarative state (default source): cup starts ON deskA; goal ON(cup,deskB) unmet.
    ss.declare_place("cup", "deskA")
    _chk(
        (Pred.ON, "cup", "deskA") in ss.relations(), "declared: cup ON deskA initially"
    )
    _chk(
        not check_goal([on("cup", "deskB")], ss.relations()),
        "goal ON(cup,deskB) unmet initially",
    )

    # Pick -> HELD, not ON anything.
    ss.declare_pick("cup")
    rel = ss.relations()
    _chk(
        (Pred.HELD, "cup", None) in rel and not any(r[0] is Pred.ON for r in rel),
        "picked -> HELD, no ON",
    )

    # Place on deskB -> ON deskB, goal met (declared); grounding position also updated.
    ss.declare_place("cup", "deskB")
    ss.set_object_rest("cup", (0.0, 1.36, 0.75))
    _chk(
        check_goal([on("cup", "deskB")], ss.relations()),
        "after place: goal ON(cup,deskB) MET (declared)",
    )
    # Geometric guard agrees (position-derived ON), independent of the declared state.
    _chk(
        (Pred.ON, "cup", "deskB") in ss.relations(source="geometric"),
        "geometric guard: cup ON deskB",
    )

    # OPEN: declared + spec-DOF (geometric) both report it.
    ss.register(
        "drawer", Role.OPENABLE, joint_name="d", open_thresh=0.1, joint_val=0.25
    )
    ss.declare_open("drawer")
    _chk((Pred.OPEN, "drawer", None) in ss.relations(), "declared: drawer OPEN")
    _chk(
        (Pred.OPEN, "drawer", None) in ss.relations(source="geometric"),
        "geometric: drawer OPEN via DOF",
    )

    dual = ss.register("tray", Role.GRASPABLE, xy=(2.0, 0.0), z=0.5)
    container = Entity(
        "tray", Role.CONTAINER, xy=(2.0, 0.0), z=0.5, height=0.15, half=(0.4, 0.3)
    )
    surface = Entity("tray", Role.SURFACE, xy=(2.0, 0.0), z=0.7, half=(0.4, 0.3))
    dual.affordances = {Role.CONTAINER: container, Role.SURFACE: surface}
    dual.target = container
    ss.register("nested", Role.GRASPABLE, xy=(2.0, 0.0), z=0.55)
    ss.register("on_tray", Role.GRASPABLE, xy=(2.0, 0.0), z=0.7)
    dual_rel = ss.relations(source="geometric")
    _chk(
        (Pred.INSIDE, "nested", "tray") in dual_rel
        and (Pred.ON, "on_tray", "tray") in dual_rel,
        "dual-role target emits typed INSIDE and ON relations",
    )
    ss.set_object_rest("tray", (3.0, 1.0, 0.5))
    _chk(
        ss.target_view("tray", Role.CONTAINER).xy == (3.0, 1.0)
        and ss.surface_top("tray") == ((3.0, 1.0), 0.7),
        "dual-role target geometry follows the moved graspable",
    )
    ss.declare_inside("nested", "tray")
    ss.declare_pick("nested")
    _chk(
        "nested" not in ss.inside and "nested" in ss.held,
        "picking an item clears its stale INSIDE parent",
    )
    ss.declare_place("nested", "deskB")
    _chk(
        ss.on.get("nested") == "deskB" and "nested" not in ss.inside,
        "placing an item leaves exactly one symbolic parent relation",
    )
    ss.declare_inside("nested", "tray")
    ss.capture_episode_baseline()
    ss.declare_pick("nested")
    ss.restore_episode_baseline()
    _chk(
        ss.inside.get("nested") == "tray" and "nested" not in ss.held,
        "episode reset restores the initial containment relation",
    )

    # ---- STALE prim_path: a dead path must be NAMED and survivable, never a USD traceback ----------
    # Regression for the defect that took down `T2-HSF-01`: ComputeWorldBound on a null prim raised
    # "Invalid prim: null", naming neither the path nor the entity holding it.
    class _DeadPrim:
        def IsValid(self):  # noqa: N802 - mirrors the pxr API
            return False

        def __bool__(self):
            return True  # a pxr null prim is truthy; only IsValid() tells the truth

    class _Stage:
        def GetPrimAtPath(self, _p):  # noqa: N802 - mirrors the pxr API
            return _DeadPrim()

    ss2 = SceneState(stage=_Stage())
    ss2.register("bench", Role.SURFACE, xy=(1.5, 2.5), z=0.9, prim_path="/World/gone")
    _chk(
        ss2._live_bbox("/World/gone") is None,
        "stale: _live_bbox returns None, does not raise",
    )
    _chk(
        ss2.stale_prim_paths() == {"/World/gone": ["bench"]},
        f"stale: the path AND its entity are named -> {ss2.stale_prim_paths()}",
    )
    # surface_top is the accessor that actually crashed the build; it must fall back to the spec pose.
    _chk(
        ss2.surface_top("bench") == ((1.5, 2.5), 0.9),
        "stale: surface_top falls back to the spec pose instead of dying",
    )
    # register_from_prim has no spec to fall back to, so it must fail -- but NAMED, not from inside USD.
    try:
        ss2.register_from_prim("ghost", Role.SURFACE, "/World/also_gone")
        _chk(False, "stale: register_from_prim must refuse a dead path")
    except ValueError as _e:
        _chk(
            "/World/also_gone" in str(_e) and "ghost" in str(_e),
            "stale: register_from_prim names both the entity and the dead path",
        )
    # A VALID prim must be completely unaffected -- no warning, no ledger entry.
    ss3 = SceneState()
    _chk(ss3.stale_prim_paths() == {}, "stale: ledger is empty on a clean scene")

    print(f"[scene_state] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())

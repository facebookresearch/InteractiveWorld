"""Restore a persistent InteractiveWorld episode to its validated initial state."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from simulator.physical_world.scene_state import Role

if TYPE_CHECKING:
    from simulator.physical_world.scene_state import Entity, SceneState

logger: logging.Logger = logging.getLogger(__name__)

# Symbolic containers the executor mutates during an episode (all live on the SceneState object). Listed
# EXPLICITLY (no reflection) so adding a new SceneState field is a deliberate change here, not a silent
# miss. Kept in sync with scene_state.SceneState's declared-effect fields.
_SYMBOLIC_ATTRS: tuple[str, ...] = (
    "held",
    "on",
    "open_objs",
    "running",
    "sitting_on",
    "actor_xy",
    "actor_seated",
    "cut",
    "cooked",
    "inside",
    "attached",
    "attach_point",
    "assembly_root",
    "assembly_roots",
    "assembly_pose",
    "movement_targets",
    "tracked_near",
    "tracked_raised",
    "tracked_unmoved",
    "motion_max_m",
    "motion_max_deg",
)
# Dynamic attrs created lazily by the placement producers (may be absent until first use); snapshot +
# restore only when present (set at build time in episode_runtime, and lazily in semantic_executor).
_DYNAMIC_ATTRS: tuple[str, ...] = ("_space_assign", "_into_placed")

# Keep the complete planning contract together. Some arrays are not normally mutated in place, but the
# live grid can be replaced after an object or openable moves; restoring only ``occ`` then combines the
# pristine occupancy with end-of-episode floor, door, and precise-geometry layers.
_NAV_ARRAYS: tuple[str, ...] = (
    "occ",
    "sdf",
    "door_id",
    "door_cost",
    "wall_sdf",
    "floor_support",
    "base_occ",
    "open_leaf_count",
    "raw_occ",
)


@dataclass
class _EntitySnap:
    """Pristine per-entity state. ``pose`` is captured only for GRASPABLEs (the only prims that move);
    furniture keeps ``pose=None`` (its USD prim is static, so there is nothing to restore)."""

    xy: tuple[float, float] | None
    z: float | None
    joint_val: float
    meta: dict
    affordances: dict
    pose: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float],
        ]
        | None
    )


@dataclass
class EpisodeSnapshot:
    """Everything a per-episode reset must restore. Produced once at env ``open()``."""

    symbolic: dict[str, object]
    dynamic: dict[str, object]
    entities: dict[str, _EntitySnap]
    free_dynamic: dict[str, object]
    nav: dict[str, object] | None
    precise_occ: object | None
    start_xy: tuple[float, float]


# ---------------------------------------------------------------------------------------------
# Snapshot (call ONCE at open(), after build + validate + settle, before any episode).
# ---------------------------------------------------------------------------------------------
def snapshot(scene: "SceneState") -> EpisodeSnapshot:
    """Capture the pristine task-start state of ``scene`` for later ``restore``."""
    symbolic = {a: copy.deepcopy(getattr(scene, a)) for a in _SYMBOLIC_ATTRS}
    dynamic = {
        a: copy.deepcopy(getattr(scene, a)) for a in _DYNAMIC_ATTRS if hasattr(scene, a)
    }

    stage = getattr(scene, "stage", None)
    xc = _xform_cache(stage)
    entities = {name: _snap_entity(stage, xc, e) for name, e in scene.entities.items()}
    free_dynamic = _snap_free_dynamic(scene, stage, xc)

    nav = _snap_nav(getattr(scene, "nav_grid", None))
    precise_occ = getattr(getattr(scene, "cworld", None), "_precise_occ", None)
    start_xy = tuple(float(v) for v in getattr(scene, "start_xy", (0.0, 0.0)))
    logger.info(
        "reset.snapshot: %d entities (%d graspable poses), nav=%s",
        len(entities),
        sum(1 for s in entities.values() if s.pose is not None),
        "yes" if nav is not None else "none",
    )
    return EpisodeSnapshot(
        symbolic=symbolic,
        dynamic=dynamic,
        entities=entities,
        free_dynamic=free_dynamic,
        nav=nav,
        precise_occ=precise_occ.copy() if precise_occ is not None else None,
        start_xy=(start_xy[0], start_xy[1]),
    )


def _snap_entity(stage, xc, e: "Entity") -> _EntitySnap:
    pose = None
    if stage is not None and xc is not None and _is_movable_entity(e) and e.prim_path:
        pose = _read_prim_pose(stage, xc, e.prim_path)
    return _EntitySnap(
        xy=e.xy,
        z=e.z,
        joint_val=e.joint_val,
        meta=copy.deepcopy(e.meta),
        affordances=copy.deepcopy(e.affordances),
        pose=pose,
    )


def _is_movable_entity(entity: "Entity") -> bool:
    return entity.role is Role.GRASPABLE or any(
        role in entity.affordances for role in (Role.PUSHABLE, Role.PULLABLE)
    )


def _snap_free_dynamic(scene: "SceneState", stage, xc) -> dict[str, object]:
    if stage is None or xc is None:
        return {}
    from pxr import UsdPhysics

    out = {}
    for raw_path in getattr(scene, "free_dynamic_prims", None) or ():
        path = str(raw_path).rstrip("/")
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            continue
        pose = _read_prim_pose(stage, xc, path)
        rigid = UsdPhysics.RigidBodyAPI(prim)
        attr = rigid.GetKinematicEnabledAttr()
        was_kinematic = bool(attr.Get()) if attr and attr.HasAuthoredValue() else False
        velocity, angular_velocity = _read_rigid_motion(rigid)
        out[path] = (was_kinematic, pose, velocity, angular_velocity)
    return out


def _read_rigid_motion(rigid) -> tuple[object | None, object | None]:
    velocity = rigid.GetVelocityAttr().Get()
    angular_velocity = rigid.GetAngularVelocityAttr().Get()
    return velocity, angular_velocity


def _snap_nav(grid) -> dict[str, object] | None:
    if grid is None:
        return None
    out: dict[str, object] = {}
    for a in _NAV_ARRAYS:
        v = getattr(grid, a, None)
        out[a] = v.copy() if v is not None else None
    return out


# ---------------------------------------------------------------------------------------------
# Restore (call at the START of every episode, before compiling/running goals).
# ---------------------------------------------------------------------------------------------
def restore(
    scene: "SceneState",
    snap: EpisodeSnapshot,
    start_xy: tuple[float, float] | None = None,
) -> None:
    """Reset ``scene`` to ``snap``'s pristine task-start state (symbolic + USD poses + nav grid)."""
    # 1. Symbolic containers -- fresh deep copies so the snapshot is never aliased into the live scene.
    for a in _SYMBOLIC_ATTRS:
        setattr(scene, a, copy.deepcopy(snap.symbolic[a]))
    # Dynamic attrs: restore the snapshot value if it was present, else DELETE any the episode created --
    # pristine == absent, so an episode-created _space_assign/_into_placed must not leak to the next reset.
    for a in _DYNAMIC_ATTRS:
        if a in snap.dynamic:
            setattr(scene, a, copy.deepcopy(snap.dynamic[a]))
        elif hasattr(scene, a):
            delattr(scene, a)

    # 2. Per-entity grounding + meta + (graspable) USD pose.
    stage = getattr(scene, "stage", None)
    for name, es in snap.entities.items():
        e = scene.entities.get(name)
        if e is None:
            continue
        e.xy, e.z, e.joint_val = es.xy, es.z, es.joint_val
        e.meta = copy.deepcopy(es.meta)
        e.affordances = copy.deepcopy(es.affordances)
        if stage is not None and es.pose is not None and e.prim_path:
            _write_prim_pose(stage, e.prim_path, es.pose)
    _restore_free_dynamic(stage, snap.free_dynamic)
    cworld = getattr(scene, "cworld", None)
    if cworld is not None:
        cworld.refresh(scene)
        if snap.precise_occ is not None:
            cworld._precise_occ = snap.precise_occ.copy()

    # 3. Nav-grid arrays, then re-carve the (possibly re-sampled) start so the planner can leave it.
    _restore_nav(scene, snap, start_xy)


def _restore_free_dynamic(stage, saved: dict[str, object]) -> None:
    if stage is None or not saved:
        return
    import physics_human as ph
    from pxr import UsdPhysics

    for path, (was_kinematic, pose, velocity, angular_velocity) in saved.items():
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            continue
        ph.set_rigid_kinematic(prim, True)
        if pose is not None:
            _write_prim_pose(stage, path, pose)
        ph.set_rigid_kinematic(prim, bool(was_kinematic))
        if velocity is not None:
            UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr().Set(velocity)
        if angular_velocity is not None:
            UsdPhysics.RigidBodyAPI(prim).GetAngularVelocityAttr().Set(angular_velocity)


def _restore_nav(
    scene: "SceneState",
    snap: EpisodeSnapshot,
    start_xy: tuple[float, float] | None,
) -> None:
    grid = getattr(scene, "nav_grid", None)
    if grid is None or snap.nav is None:
        return
    for a in _NAV_ARRAYS:
        v = snap.nav.get(a)
        setattr(grid, a, v.copy() if v is not None else None)
    import nav_grid  # simulator.physical_world (needs numpy/scipy; safe pre-Kit)

    sx = tuple(start_xy) if start_xy is not None else snap.start_xy
    # carve_free rebuilds the SDF only if a cell actually changes; with the same start the start cells
    # are already free in the snapshot occ, so this is a no-op -> a same-seed reset is byte-identical.
    nav_grid.carve_free(grid, [(float(sx[0]), float(sx[1]))])


# ---------------------------------------------------------------------------------------------
# USD pose helpers (lazy pxr / simulator.physical_world imports -- safe to import this module before launch_app).
# ---------------------------------------------------------------------------------------------
def _xform_cache(stage):
    """A fresh XformCache at the default time code, or None when there is no live stage (offline)."""
    if stage is None:
        return None
    from pxr import Usd, UsdGeom

    return UsdGeom.XformCache(Usd.TimeCode.Default())


def _read_prim_pose(stage, xc, prim_path: str):
    """(pos, quat_wxyz, scale) of a prim, or None if it is not built.

    Translate + orient come from object_physics._read_pose (world matrix, kinematic prims -> current).
    Scale is read from the xformOp directly: it is static, but set_world_pose REWRITES the scale op, so
    we must pass the real scale back or a non-uniform BEHAVIOR asset would be reset to scale 1.
    """
    import object_physics  # simulator.physical_world
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return None
    xc.Clear()  # poses are current at snapshot time; a stale cache would return the build-time pose
    pos, quat = object_physics._read_pose(xc, prim)
    return (pos, quat, _read_scale(prim, UsdGeom))


def _read_scale(prim, UsdGeom) -> tuple[float, float, float]:
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:scale":
            v = op.Get()
            if v is not None:
                return (float(v[0]), float(v[1]), float(v[2]))
    return (1.0, 1.0, 1.0)


def _write_prim_pose(stage, prim_path: str, pose) -> None:
    """Kinematic restore: rewrite the prim's translate/orient/scale to the snapshot pose (pure USD)."""
    import fpv_scene  # simulator.physical_world

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return
    pos, quat, scale = pose
    fpv_scene.set_world_pose(prim, pos, quat, scale)


def snapshot_graspable_poses(scene: "SceneState") -> dict[str, object]:
    """Read every built graspable's kinematic flag and pose without changing it."""
    stage = getattr(scene, "stage", None)
    if stage is None:
        return {}
    from pxr import UsdPhysics

    xc = _xform_cache(stage)
    out = {}
    for entity in scene.entities.values():
        if entity.role is not Role.GRASPABLE or not entity.prim_path:
            continue
        prim = stage.GetPrimAtPath(entity.prim_path)
        pose = _read_prim_pose(stage, xc, entity.prim_path)
        if prim is None or not prim.IsValid() or pose is None:
            continue
        rigid = UsdPhysics.RigidBodyAPI(prim)
        attr = rigid.GetKinematicEnabledAttr()
        was_kinematic = bool(attr.Get()) if attr and attr.HasAuthoredValue() else False
        velocity, angular_velocity = _read_rigid_motion(rigid)
        out[entity.prim_path] = (
            was_kinematic,
            pose,
            velocity,
            angular_velocity,
        )
    return out


def restore_graspable_poses(scene: "SceneState", states: dict[str, object]) -> None:
    """Restore graspable poses and the kinematic flags captured with them."""
    stage = getattr(scene, "stage", None)
    if stage is None:
        return
    import physics_human as ph
    from pxr import UsdPhysics

    for prim_path, (
        was_kinematic,
        pose,
        velocity,
        angular_velocity,
    ) in states.items():
        prim = stage.GetPrimAtPath(prim_path)
        if prim is not None and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            ph.set_rigid_kinematic(prim, True)
        _write_prim_pose(stage, prim_path, pose)
        if prim is not None and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            ph.set_rigid_kinematic(prim, bool(was_kinematic))
            rigid = UsdPhysics.RigidBodyAPI(prim)
            if velocity is not None:
                rigid.GetVelocityAttr().Set(velocity)
            if angular_velocity is not None:
                rigid.GetAngularVelocityAttr().Set(angular_velocity)


# ---------------------------------------------------------------------------------------------
# Render-time object isolation: freeze the dynamic graspables KINEMATIC + snapshot their poses so a
# MULTI-frame FPV render (which re-poses the human rig + ticks the RTX/physics clock several times per
# command) cannot let PhysX drift / knock a just-settled object out of its scored rest.
# ---------------------------------------------------------------------------------------------
def freeze_graspables(
    scene: "SceneState", exclude_paths: tuple[str, ...] = ()
) -> dict | None:
    """Freeze every GRASPABLE rigid body KINEMATIC and snapshot its live USD pose; return the restore
    token (``{prim_path: (was_kinematic, pose)}``) or ``None`` when there is no live stage.

    WHY a plain USD pose snapshot/restore alone is NOT enough (and this freeze is): a settled object
    left DYNAMIC in PhysX fabric ignores an external USD xform write, so restoring its pose is a no-op
    while the render's physics ticks slide it out of the container's INSIDE z-band -> score undercount.
    Setting ``kinematicEnabled`` makes the body immune to gravity/contacts AND makes the pose write
    stick; the render loop's own ``advance_once`` resyncs the flip into fabric before the captures. Pure
    USD attr writes + XformCache reads -- no physics step, cheap enough to bracket every render."""
    stage = getattr(scene, "stage", None)
    if stage is None:
        return None
    import physics_human as ph
    from pxr import UsdPhysics

    xc = _xform_cache(stage)
    saved: dict[str, object] = {}
    excluded = {str(path).rstrip("/") for path in exclude_paths}
    for e in scene.entities.values():
        if e.role is not Role.GRASPABLE or not e.prim_path:
            continue
        if str(e.prim_path).rstrip("/") in excluded:
            continue
        prim = stage.GetPrimAtPath(e.prim_path)
        if (
            prim is None
            or not prim.IsValid()
            or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            continue
        kin_attr = UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr()
        was_kin = (
            bool(kin_attr.Get()) if kin_attr and kin_attr.HasAuthoredValue() else False
        )
        saved[e.prim_path] = (was_kin, _read_prim_pose(stage, xc, e.prim_path))
        ph.set_rigid_kinematic(prim, True)
    return saved


def restore_graspables(scene: "SceneState", saved: dict | None) -> None:
    """Undo ``freeze_graspables``: rewrite each graspable to its snapshot pose (a kinematic USD write
    sticks, correcting any transient drift), then restore its prior kinematic flag, so the object state
    -- and thus the terminal score read off the USD poses -- is byte-identical to before the render."""
    if not saved:
        return
    stage = getattr(scene, "stage", None)
    if stage is None:
        return
    import physics_human as ph

    for prim_path, (was_kin, pose) in saved.items():
        prim = stage.GetPrimAtPath(prim_path)
        if prim is not None and prim.IsValid():
            ph.set_rigid_kinematic(prim, True)
        if pose is not None:
            _write_prim_pose(stage, prim_path, pose)
        if prim is not None and prim.IsValid():
            ph.set_rigid_kinematic(prim, bool(was_kin))


# =====================================================================================================
# Offline selftest: symbolic + grounding round-trip WITHOUT Isaac (stage=None skips the USD pose path).
# =====================================================================================================
def _selftest() -> int:
    from scene_state import SceneState

    ok = True

    def _chk(cond: bool, msg: str) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    scene = SceneState(floor_z=0.0, start_xy=(0.0, 0.0))
    scene.register("can1", Role.GRASPABLE, xy=(1.0, 1.0), z=0.1, approach_yaw=0.0)
    scene.register("can2", Role.GRASPABLE, xy=(1.5, 1.0), z=0.1, approach_yaw=0.0)
    scene.register("bin", Role.CONTAINER, xy=(2.0, 0.0), z=0.2, half=(0.3, 0.3))
    scene.register("fridge", Role.OPENABLE, xy=(3.0, 0.0), z=1.0, joint_name="d")

    snap = snapshot(scene)

    # Mutate every per-episode surface the way an episode would.
    scene.declare_pick("can1")
    scene.declare_inside("can2", "bin")
    scene.set_object_rest("can1", (9.0, 9.0, 9.0))
    scene.open_objs.add("fridge")
    scene.entities["fridge"].joint_val = 0.75
    scene.actor_xy = (4.0, 5.0)
    scene.actor_seated = True
    scene.cut.add("can1")
    scene.entities["fridge"].meta["_nav_leaf_cells"] = [(1, 2), (3, 4)]
    scene._space_assign = {"bin": "bin_space1"}  # noqa: SLF001 (dynamic attr, mirrors the producers)

    _chk(scene.held == {"can1"}, "mutation applied: can1 held")
    _chk(
        scene.inside == {"can2": "bin"},
        "mutation applied: can2 inside bin without releasing can1",
    )
    _chk(
        scene.entities["can1"].xy == (9.0, 9.0),
        "mutation applied: can1 grounding moved",
    )
    _chk(
        scene.open_objs == {"fridge"} and scene.entities["fridge"].joint_val == 0.75,
        "mutation applied: fridge opened",
    )
    _chk(
        scene.actor_xy == (4.0, 5.0) and scene.actor_seated,
        "mutation applied: actor moved and seated",
    )

    restore(scene, snap)

    _chk(scene.held == set(), "restore: held cleared")
    _chk(scene.inside == {}, "restore: inside cleared")
    _chk(scene.open_objs == set(), "restore: open_objs cleared")
    _chk(
        scene.entities["fridge"].joint_val == 0.0,
        "restore: fridge joint returned to task-start",
    )
    _chk(
        scene.actor_xy is None and not scene.actor_seated,
        "restore: actor returned to task-start",
    )
    _chk(scene.cut == set(), "restore: cut cleared")
    _chk(
        scene.entities["can1"].xy == (1.0, 1.0) and scene.entities["can1"].z == 0.1,
        "restore: can1 grounding back to task-start",
    )
    _chk(
        scene.entities["fridge"].meta.get("_nav_leaf_cells") in (None, []),
        "restore: fridge leaf-cells cleared",
    )
    _chk(getattr(scene, "_space_assign", {}) == {}, "restore: _space_assign cleared")

    # Snapshot isolation: mutating the restored scene must NOT corrupt the pristine snapshot (deep copy).
    scene.declare_pick("can1")
    scene.entities["can1"].xy = (7.0, 7.0)
    restore(scene, snap)
    _chk(scene.held == set(), "snapshot isolation: 2nd restore still clears held")
    _chk(
        scene.entities["can1"].xy == (1.0, 1.0),
        "snapshot isolation: 2nd restore still returns grounding",
    )

    print(f"[reset] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())

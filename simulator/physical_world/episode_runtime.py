"""Isaac setup and rendering runtime for TaskPack worlds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
from enum import Enum
from types import SimpleNamespace

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRY_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, os.path.join(_TRY_DIR, "human_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# torch-first (pre-launch): the motion core + gait + executor import torch at module scope.
import clutter_render_replay  # noqa: E402
import command_grammar  # noqa: E402
import humanoid_motion as wpp  # noqa: E402
import egocentric_camera as ego_camera  # noqa: E402
import gait_player as gait  # noqa: E402
import placement_slots  # noqa: E402
import semantic_executor as se  # noqa: E402
import smplh_anim_core as core  # noqa: E402
from simulator.runtime_paths import autoexposure_cache_path  # noqa: E402
from goal_schema import (  # noqa: E402
    Goal,
    Primitive,
    PrimitiveError,
    WAIT_DEFAULT_S,
)
from scene_state import Role, SceneState  # noqa: E402

logger: logging.Logger = logging.getLogger(__name__)

FPS = 12
PLACEMENT_SETTLE_ATTEMPTS = 3
INITIAL_SUPPORT_SEARCH_M = 0.50
INITIAL_SUPPORT_PLANE_TOL_M = 0.03
INITIAL_SUPPORT_CLEARANCE_EPS_M = 0.003
# A live collision plane this far from the fixture's measured worktop is a different structural
# surface (typically the plinth or floor under an open bar), not a valid correction of its height.
INITIAL_SUPPORT_SEMANTIC_PLANE_TOL_M = 0.25
INITIAL_SUPPORT_OPTION_LIMIT = 64
KINEMATIC_SUPPORT_FRACTION = 0.60
INITIAL_PROFILE_SCALE = 1.15
INITIAL_PROFILE_MARGIN_M = 0.015
OPEN_ACTION_STAND_PROBES = 12
CONTENT_ACTION_STAND_PROBES = 4
OPENABLE_PANEL_CONTACT_FRACTIONS = (1.0, 0.75, 0.50, 0.25)
_OPENABLE_ACTION_KIND_UNSET = object()
NAV_DOOR_CAMERA_STANDOFF_M = 1.15
NAV_DOOR_ACTION_CAMERA_CLEARANCE_M = 0.16
FRONT_OPENING_CAMERA_PULLBACK_M = 0.12
FRONT_INSERT_CAMERA_PULLBACK_M = 0.12
FRONT_INSERT_CARRIED_CAMERA_CLEARANCE_M = 0.18
LOW_FRONT_INSERT_CAMERA_MIN_FORWARD_OF_HEAD_M = 0.18
LARGE_FRONT_CONTAINER_CAMERA_PULLBACK_M = 0.12
LARGE_FRONT_CONTAINER_CAMERA_MIN_EYE_HEIGHT_M = 1.10
LARGE_FRONT_CONTAINER_CAMERA_MIN_FORWARD_OF_HEAD_M = 0.06
LARGE_FRONT_CONTAINER_CAMERA_MAX_FORWARD_OF_HEAD_M = 0.14
LARGE_FRONT_CONTAINER_CAMERA_MAX_LATERAL_FROM_HEAD_M = 0.08
LARGE_FRONT_CONTAINER_CAMERA_MAX_ABOVE_HEAD_M = 0.12
PROACTIVE_PHASE_CAPTURE_VERSION = "wearable-proactive-phase-sparse-capture/1.0"


_DYNAMIC_SETTLE_PRIMS = frozenset(
    {
        Primitive.PICK,
        Primitive.FLIP,
        Primitive.ATTACH,
        Primitive.DETACH,
        Primitive.RAISE,
        Primitive.PUSH,
        Primitive.PULL,
    }
)


def objects_requiring_dynamic_settle(program: str, scene) -> set[str]:
    """Objects whose program semantics require a physically discovered loose-body rest."""
    result = set(getattr(scene, "held", set()))
    for goal in command_grammar.compile_program(program, scene):
        if goal.prim in _DYNAMIC_SETTLE_PRIMS and goal.obj_key:
            result.add(str(goal.obj_key))
        if goal.prim is Primitive.ATTACH and goal.meta.get("child"):
            result.add(str(goal.meta["child"]))
        if goal.prim is Primitive.RAISE and goal.meta.get("root"):
            result.add(str(goal.meta["root"]))
        if goal.meta.get("tool"):
            result.add(str(goal.meta["tool"]))
    return result


def _record_support_loads(occupants, snapshots) -> None:
    for support_key, rows in occupants.items():
        if not rows:
            continue
        load = tuple(
            sorted(
                (object_key, rotatable, preferred_xy)
                for object_key, (rotatable, preferred_xy) in rows.items()
            )
        )
        if load not in snapshots[support_key]:
            snapshots[support_key].append(load)


def support_planning_specs(scene, specs) -> dict[str, dict]:
    """Include graspables built outside the loose-object staging path."""
    result = {str(spec["key"]): spec for spec in specs}
    for key, entity in scene.entities.items():
        object_key = str(key)
        path = str(getattr(entity, "prim_path", None) or "")
        if (
            object_key not in result
            and getattr(entity, "role", None) is Role.GRASPABLE
            and path
        ):
            result[object_key] = {"key": object_key, "path": path}
    return result


def planned_support_loads(program: str, scene, specs, support_keys):
    """Return every distinct object load a surface must carry during the program.

    Each load row is ``(object_key, may_rotate_quarter_turn, preferred_xy)``.
    Initial occupants keep their authored orientation; objects placed later may use
    the runtime's existing in-hand quarter-turn placement. An authored ``place_spot``
    is preserved as the preferred world-space point for the collision preflight.
    """
    planning_specs = support_planning_specs(scene, specs)
    known_objects = set(planning_specs)
    support_keys = {str(key) for key in support_keys}
    occupants = {key: {} for key in support_keys}
    locations = {}
    for spec in planning_specs.values():
        object_key = str(spec["key"])
        initial_place = str(spec.get("initial_place") or "")
        support_key = initial_place[3:] if initial_place.startswith("on:") else ""
        if support_key in occupants:
            occupants[support_key][object_key] = (False, None)
            locations[object_key] = support_key
    snapshots = {key: [] for key in support_keys}
    _record_support_loads(occupants, snapshots)
    held = None
    for goal in command_grammar.compile_program(program, scene):
        if goal.prim is Primitive.PICK and str(goal.obj_key) in known_objects:
            held = str(goal.obj_key)
            previous = locations.pop(held, None)
            if previous in occupants:
                occupants[previous].pop(held, None)
        elif goal.prim in (Primitive.PLACE, Primitive.PLACE_INSIDE) and held:
            target = str(goal.meta.get("surface") or "")
            if goal.prim is Primitive.PLACE and target in occupants:
                preferred_xy = (
                    None
                    if goal.place_xy is None
                    else (float(goal.place_xy[0]), float(goal.place_xy[1]))
                )
                occupants[target][held] = (True, preferred_xy)
                locations[held] = target
            held = None
            _record_support_loads(occupants, snapshots)
        elif goal.prim in (Primitive.RELEASE, Primitive.ATTACH) and held:
            held = None
    return {key: tuple(rows) for key, rows in snapshots.items()}


def _kinematic_place_slot(scene, goal) -> tuple[str, str] | None:
    behavior_task = getattr(scene, "behavior_task", None) or {}
    dialogue = behavior_task.get("episode_dialogue") or {}
    meta = getattr(goal, "meta", None) or {}
    subject = str(meta.get("_held_object_key") or "")
    parent = str(meta.get("surface") or "")
    return placement_slots.declared_slot(dialogue, subject, parent)


def _settle_or_lock_placement(scene, goal, lock_slot, settle_physics):
    behavior_task = getattr(scene, "behavior_task", None) or {}
    dialogue = behavior_task.get("episode_dialogue") or {}
    meta = getattr(goal, "meta", None) or {}
    return placement_slots.settle_or_lock(
        dialogue,
        str(meta.get("_held_object_key") or ""),
        str(meta.get("surface") or ""),
        lock_slot,
        settle_physics,
    )


def _capture_placement_retry(env, scene, goal, frame_start: int) -> dict | None:
    held = getattr(env.st, "held", None)
    target = str((goal.meta or {}).get("surface") or "")
    subject = str((held or {}).get("obj_key") or "")
    if not subject or not target:
        return None
    import object_physics

    paths = [
        entity.prim_path
        for entity in scene.entities.values()
        if entity.role is Role.GRASPABLE and entity.prim_path
    ]
    dynamic = {}
    for name in ("_space_assign", "_into_placed"):
        dynamic[name] = (
            (True, copy.deepcopy(getattr(scene, name)))
            if hasattr(scene, name)
            else (False, None)
        )
    entity = scene.entities.get(subject)
    return {
        "subject": subject,
        "target": target,
        "frame_start": frame_start,
        "state": copy.deepcopy(env.st),
        "rigid_states": object_physics.snapshot_rigid_states(scene.stage, paths),
        "held": copy.deepcopy(scene.held),
        "on": copy.deepcopy(scene.on),
        "inside": copy.deepcopy(scene.inside),
        "open_objs": copy.deepcopy(getattr(scene, "open_objs", set())),
        "dynamic": dynamic,
        "entity_grounding": copy.deepcopy((entity.xy, entity.z)) if entity else None,
        "reports_len": len(env.reports),
        "report_detail": copy.deepcopy(getattr(env, "report_detail", {})),
        "report_frame_span": copy.deepcopy(getattr(env, "report_frame_span", {})),
        "report_state": copy.deepcopy(getattr(env, "report_state", {})),
    }


def _restore_placement_retry(env, scene, retry) -> None:
    import object_physics

    del env.motion.frames[int(retry["frame_start"]) :]
    env.st = copy.deepcopy(retry["state"])
    object_physics.restore_rigid_states(scene.stage, retry["rigid_states"])
    scene.held = copy.deepcopy(retry["held"])
    scene.on = copy.deepcopy(retry["on"])
    scene.inside = copy.deepcopy(retry["inside"])
    scene.open_objs = copy.deepcopy(retry["open_objs"])
    for name, raw in retry["dynamic"].items():
        existed, value = raw
        if existed:
            setattr(scene, name, copy.deepcopy(value))
        elif hasattr(scene, name):
            delattr(scene, name)
    entity = scene.entities.get(str(retry["subject"]))
    if entity is not None and retry["entity_grounding"] is not None:
        entity.xy, entity.z = copy.deepcopy(retry["entity_grounding"])
    del env.reports[int(retry["reports_len"]) :]
    env.report_detail = copy.deepcopy(retry["report_detail"])
    env.report_frame_span = copy.deepcopy(retry["report_frame_span"])
    env.report_state = copy.deepcopy(retry["report_state"])


def _failed_placement_xy(scene, retry, goal):
    chosen = (goal.meta or {}).get("_placement_chosen_xy") or (goal.meta or {}).get(
        "_inside_chosen_xy"
    )
    if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
        return float(chosen[0]), float(chosen[1])
    placed = getattr(scene, "_into_placed", {}).get(retry["target"]) or []
    return (float(placed[-1][0]), float(placed[-1][1])) if placed else None


def _placement_space_count(scene, target_key: str) -> int:
    return se.placement_space_count(scene, target_key)


def _stand_pose_changed(state, stand, *, xy_tol: float = 0.05, yaw_tol: float = 1.0):
    current_xy = tuple(state.root_xy)
    current_yaw = float(getattr(state, "root_yaw", 0.0))
    target_xy, target_yaw = stand
    yaw_delta = abs((float(target_yaw) - current_yaw + 180.0) % 360.0 - 180.0)
    return (
        abs(float(target_xy[0]) - float(current_xy[0])) > xy_tol
        or abs(float(target_xy[1]) - float(current_xy[1])) > xy_tol
        or yaw_delta > yaw_tol
    )


def _content_action_parent_key(scene, goal, reach_key: str) -> str:
    if getattr(goal, "prim", None) is command_grammar.Primitive.PLACE_INSIDE:
        return str((getattr(goal, "meta", None) or {}).get("surface") or reach_key)
    current = str(getattr(goal, "obj_key", None) or reach_key)
    inside = getattr(scene, "inside", None) or {}
    seen = set()
    while current not in seen:
        seen.add(current)
        try:
            entity = scene.resolve(current)
        except (KeyError, ValueError):
            break
        if (getattr(entity, "meta", None) or {}).get("access_mode") in {
            "front_drum",
            "front_opening",
        }:
            return current
        parent = inside.get(current)
        if parent is None:
            break
        current = str(parent)
    return str(reach_key)


def _frozen_content_collision_world(scene, goal, reach_key: str):
    """Snapshot collider AABBs into a query-only oracle with no USD/PhysX handles."""
    source = getattr(scene, "cworld", None)
    if source is None:
        return None
    try:
        source_boxes = getattr(source, "boxes", None)
        if not callable(source_boxes):
            return None
        boxes = tuple(
            (
                str(owner),
                np.asarray(mn, dtype=np.float64).copy(),
                np.asarray(mx, dtype=np.float64).copy(),
            )
            for owner, mn, mx in source_boxes()
        )
        if not boxes:
            return None
        frozen = SimpleNamespace(_ph=None, _stage=None, _precise_occ=None)
        parent_key = _content_action_parent_key(scene, goal, reach_key)
        parent = scene.resolve(parent_key)
        if parent_key in (getattr(scene, "open_objs", None) or set()):
            leaf_roots = tuple(
                str(path).rstrip("/")
                for path in (getattr(parent, "leaf_paths", None) or ())
                if path
            )
            if leaf_roots:
                if not (getattr(parent, "leaf_bboxes", None) or ()):
                    return None
                boxes = tuple(
                    row
                    for row in boxes
                    if not any(
                        row[0] == root or row[0].startswith(root + "/")
                        for root in leaf_roots
                    )
                )
                geometry_scene = copy.copy(scene)
                geometry_scene.stage = None
                boxes += tuple(
                    (str(owner), np.asarray(mn).copy(), np.asarray(mx).copy())
                    for owner, mn, mx in se._open_leaf_boxes(geometry_scene, parent)
                )
    except Exception as error:
        logger.info("[content-preview] collider snapshot unavailable: %s", error)
        return None

    def filtered(exclude_owners=()):
        roots = tuple(str(owner).rstrip("/") for owner in (exclude_owners or ()))
        return tuple(
            row
            for row in boxes
            if not any(
                row[0] == root or row[0].startswith(root + "/") for root in roots
            )
        )

    def frozen_boxes(z_lo=None, z_hi=None, exclude_owners=()):
        return [
            (owner, mn.copy(), mx.copy())
            for owner, mn, mx in filtered(exclude_owners)
            if (z_hi is None or float(mn[2]) <= float(z_hi))
            and (z_lo is None or float(mx[2]) >= float(z_lo))
        ]

    def sphere_hits(center, radius: float, exclude_owners=()):
        point = np.asarray(center, dtype=np.float64)
        for owner, mn, mx in filtered(exclude_owners):
            closest = np.maximum(mn, np.minimum(point, mx))
            if float(np.dot(point - closest, point - closest)) <= float(radius) ** 2:
                return owner
        return None

    def human_samples(joints, joint_subset=None, exclude_owners=()):
        import physics_human

        hits = []
        for center, radius, capsule in physics_human._bone_overlap_samples_subset(
            np.asarray(joints, dtype=np.float64), joint_subset
        ):
            point = np.asarray(center, dtype=np.float64)
            for owner, mn, mx in filtered(exclude_owners):
                closest = np.maximum(mn, np.minimum(point, mx))
                distance = float(np.linalg.norm(point - closest))
                penetration = float(radius) - distance
                if penetration > 0.0:
                    hits.append((penetration, str(capsule), owner, distance <= 1e-9))
        return hits

    def human_overlap(joints, joint_subset=None, exclude_owners=(), **_kwargs):
        return sorted(
            {
                (capsule, owner)
                for _depth, capsule, owner, _saturated in human_samples(
                    joints, joint_subset, exclude_owners
                )
            }
        )

    def human_exceeds_depth(joints, allowed_m, joint_subset=None, exclude_owners=()):
        return any(
            depth > float(allowed_m)
            for depth, _capsule, _owner, _saturated in human_samples(
                joints, joint_subset, exclude_owners
            )
        )

    def human_depth(joints, joint_subset=None, exclude_owners=()):
        hits = human_samples(joints, joint_subset, exclude_owners)
        if not hits:
            return 0.0, "", False
        depth, capsule, owner, saturated = max(hits, key=lambda row: row[0])
        return float(depth), f"{capsule} vs {owner}", bool(saturated)

    def object_extent(obj_path):
        if not obj_path:
            return None
        root = str(obj_path).rstrip("/")
        rows = [
            (mn, mx)
            for owner, mn, mx in boxes
            if owner == root or owner.startswith(root + "/")
        ]
        if not rows:
            return None
        lower = np.min(np.asarray([row[0] for row in rows]), axis=0)
        upper = np.max(np.asarray([row[1] for row in rows]), axis=0)
        return (
            max(0.02, 0.5 * float(upper[0] - lower[0])),
            max(0.02, 0.5 * float(upper[1] - lower[1])),
            float(upper[2] - lower[2]),
        )

    def drop_site_probe(xy, top_z, obj_path, obj_half_xy, obj_h):
        if obj_h is None or not obj_path:
            return None
        x, y = float(xy[0]), float(xy[1])
        hx, hy = float(obj_half_xy[0]), float(obj_half_xy[1])
        excluded = (str(obj_path).rstrip("/"), "/World/HumanProxy", "/World/Obj")
        candidates = [
            row
            for row in filtered(excluded)
            if float(row[1][0]) <= x <= float(row[2][0])
            and float(row[1][1]) <= y <= float(row[2][1])
            and float(row[2][2]) <= float(top_z) + 0.05
            and float(row[2][2]) >= float(top_z) - 0.08
        ]
        support = max(candidates, key=lambda row: float(row[2][2]), default=None)
        if support is None:
            return {
                "ok": False,
                "support": False,
                "clear": False,
                "embedded": False,
                "reason": "no-support",
                "hit_z": None,
                "support_frac": 0.0,
                "support_path": None,
                "blockers": [],
            }
        seat_z = float(support[2][2])
        blockers = []
        for owner, mn, mx in filtered(excluded):
            if owner == support[0]:
                continue
            overlaps_xy = (
                float(mn[0]) < x + hx
                and float(mx[0]) > x - hx
                and float(mn[1]) < y + hy
                and float(mx[1]) > y - hy
            )
            overlaps_z = float(mn[2]) < seat_z + float(obj_h) and float(mx[2]) > (
                seat_z + 0.002
            )
            if overlaps_xy and overlaps_z:
                blockers.append(owner)
        return {
            "ok": not blockers,
            "support": True,
            "clear": not blockers,
            "embedded": bool(blockers),
            "reason": "ok" if not blockers else "blocked",
            "hit_z": seat_z,
            "support_frac": 1.0,
            "support_path": support[0],
            "blockers": blockers,
        }

    def free_rest_window(xy, top_z, _obj_path, _obj_h):
        x, y = float(xy[0]), float(xy[1])
        supports = [
            row
            for row in boxes
            if float(row[1][0]) <= x <= float(row[2][0])
            and float(row[1][1]) <= y <= float(row[2][1])
            and abs(float(row[2][2]) - float(top_z)) <= 0.08
        ]
        support = max(supports, key=lambda row: float(row[2][2]), default=None)
        if support is None:
            return None
        return {
            "x": (x - float(support[1][0]), float(support[2][0]) - x),
            "y": (y - float(support[1][1]), float(support[2][1]) - y),
        }

    frozen.boxes = frozen_boxes
    frozen.sphere_overlap = sphere_hits
    frozen.human_overlap = human_overlap
    frozen.human_exceeds_depth = human_exceeds_depth
    frozen.human_depth = human_depth
    frozen.object_extent = object_extent
    frozen.drop_site_probe = drop_site_probe
    frozen.free_rest_window = free_rest_window
    return frozen


def _detached_content_preview_container(value, memo):
    value_id = id(value)
    if isinstance(value, dict):
        result = {}
        memo[value_id] = result
        for key, item in value.items():
            result[_detached_content_preview_value(key, memo)] = (
                _detached_content_preview_value(item, memo)
            )
        return result
    if isinstance(value, list):
        result = []
        memo[value_id] = result
        result.extend(_detached_content_preview_value(item, memo) for item in value)
        return result
    if isinstance(value, set):
        result = set()
        memo[value_id] = result
        result.update(_detached_content_preview_value(item, memo) for item in value)
        return result
    if isinstance(value, tuple):
        result = tuple(_detached_content_preview_value(item, memo) for item in value)
        memo[value_id] = result
        return result
    if isinstance(value, frozenset):
        result = frozenset(
            _detached_content_preview_value(item, memo) for item in value
        )
        memo[value_id] = result
        return result
    raise TypeError(f"unsupported preview container {type(value).__name__}")


def _detached_content_preview_object(value, memo):
    value_id = id(value)
    if not hasattr(value, "__dict__"):
        try:
            result = copy.deepcopy(value)
        except Exception:
            return None
        memo[value_id] = result
        return result
    try:
        result = copy.copy(value)
    except Exception:
        return None
    memo[value_id] = result
    for name, item in vars(value).items():
        setattr(result, name, _detached_content_preview_value(item, memo))
    return result


def _detached_content_preview_value(value, memo=None):
    if memo is None:
        memo = {}
    if value is None or isinstance(
        value, (bool, int, float, complex, str, bytes, Enum)
    ):
        return value
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    module = type(value).__module__
    if module == "pxr" or module.startswith(
        ("pxr.", "omni.", "isaacsim.", "carb.")
    ):
        return None
    if isinstance(value, np.ndarray):
        result = value.copy()
        memo[value_id] = result
        return result
    if isinstance(value, (dict, list, set, tuple, frozenset)):
        return _detached_content_preview_container(value, memo)
    return _detached_content_preview_object(value, memo)


def _content_action_is_safe_from_stand(
    env,
    rig,
    scene,
    goal,
    reach_key: str,
    stand_xy,
    yaw: float,
    frozen_collision_world,
) -> bool:
    action_kind = _openable_action_kind(scene, goal, reach_key)
    producer = {
        "insert": se.produce_place_inside,
        "extract": se.produce_pick,
    }.get(action_kind)
    if producer is None:
        return False
    try:
        preview_rig = copy.copy(rig)
        rig_memo = {id(rig): preview_rig}
        for name, value in vars(rig).items():
            if name in {"_stage", "_body_mesh", "_normals_pv", "_obj_xf"}:
                setattr(preview_rig, name, None)
            else:
                setattr(
                    preview_rig,
                    name,
                    _detached_content_preview_value(value, rig_memo),
                )
        if hasattr(preview_rig, "_refetch_handles"):
            preview_rig._refetch_handles = False

        preview_scene = copy.copy(scene)
        scene_memo = {id(scene): preview_scene}
        for name, value in vars(scene).items():
            if name in {"stage", "cworld", "nav_grid"}:
                setattr(preview_scene, name, None)
            elif name == "_openable_controllers":
                setattr(preview_scene, name, {})
            elif name == "resolve":
                continue
            else:
                setattr(
                    preview_scene,
                    name,
                    _detached_content_preview_value(value, scene_memo),
                )
        if "resolve" in vars(scene):
            preview_scene.resolve = preview_scene.entities.__getitem__
        preview_scene.stage = None
        preview_scene.cworld = None
        preview_scene.nav_grid = None
        preview_scene._openable_controllers = {}
        preview_scene.live_wrap = False
        if frozen_collision_world is None:
            return False
        preview_scene.cworld = frozen_collision_world
        preview_scene._drop_site_probe_override = frozen_collision_world.drop_site_probe
        preview_scene._lateral_free_window_override = (
            frozen_collision_world.free_rest_window
        )
        preview_state = _detached_content_preview_value(env.st)
        preview_scene._content_preview_object_extents = {}
        for key, entity in preview_scene.entities.items():
            extent = frozen_collision_world.object_extent(
                getattr(entity, "prim_path", None)
            )
            if extent is not None:
                preview_scene._content_preview_object_extents[str(key)] = extent
        preview_scene._content_preview_object_base_offsets = {}
        preview_held = getattr(preview_state, "held", None)
        preview_held_key = (
            str(preview_held.get("obj_key"))
            if isinstance(preview_held, dict) and preview_held.get("obj_key")
            else None
        )
        if preview_held_key is None and action_kind == "insert":
            preview_held_key = str(
                (getattr(goal, "meta", None) or {}).get("_held_object_key") or ""
            ) or None
            if preview_held_key is not None:
                preview_state.held = {
                    "obj_key": preview_held_key,
                    "grasp_arm": _detached_content_preview_value(
                        getattr(preview_rig, "arm_grasp", {})
                    ),
                    "otw": np.eye(4, dtype=np.float64),
                    "rhp": _detached_content_preview_value(
                        getattr(preview_rig, "rhp", None)
                    ),
                    "hand": "right",
                    "source_rest_pose": None,
                    "source_surface": None,
                    "carried": (),
                }
                if hasattr(preview_scene, "held"):
                    preview_scene.held.add(preview_held_key)
        if preview_held_key is not None:
            held_extent = se._obj_extent(scene, preview_held_key)
            if held_extent[2] is not None:
                preview_scene._content_preview_object_extents[preview_held_key] = (
                    tuple(float(value) for value in held_extent)
                )
            preview_scene._content_preview_object_base_offsets[preview_held_key] = (
                se._held_base_offset(scene, preview_held_key)
            )
        preview_state.root_xy = (float(stand_xy[0]), float(stand_xy[1]))
        preview_state.root_yaw = float(yaw)
        preview_state.walk_end_yaw = float(yaw)
        preview_goal = _detached_content_preview_value(goal)
        tolerance = _openable_content_action_reach_tolerance(
            preview_scene,
            preview_goal,
            reach_key,
            action_kind=action_kind,
        )
        if tolerance is not None:
            preview_goal.meta["_egocentric_reach_tol_m"] = float(tolerance)
        frames = producer(
            preview_rig,
            _detached_content_preview_value(getattr(env, "g", None)),
            preview_scene,
            preview_state,
            preview_goal,
        )
        camera_pullback_m = max(
            (
                _camera_pullback_for_frame(preview_scene, frame)
                for frame in frames
                if str(getattr(frame, "seg", "")).startswith("front_insert")
            ),
            default=0.0,
        )
        if se.first_fpv_eye_collision(
            preview_scene,
            frames,
            camera_frame_offset=int(
                (getattr(preview_goal, "meta", None) or {}).get(
                    "_camera_frame_offset", 0
                )
            ),
            camera_pullback_m=camera_pullback_m,
        ) is not None:
            return False
    except Exception as error:
        logger.info(
            "[content-preview] %s rejected stand (%.2f, %.2f, yaw=%.1f): %s",
            goal.obj_key,
            float(stand_xy[0]),
            float(stand_xy[1]),
            float(yaw),
            error,
        )
        return False
    return True


def _openable_candidate_validator(
    env,
    rig,
    scene,
    goal,
    reach_key=None,
    *,
    action_kind=_OPENABLE_ACTION_KIND_UNSET,
):
    """Return a bounded full-action preview for openable stand search."""
    import action_check

    door_action = goal.prim in (
        command_grammar.Primitive.OPEN,
        command_grammar.Primitive.CLOSE,
        command_grammar.Primitive.OPENDOOR,
        command_grammar.Primitive.CLOSEDOOR,
    )
    resolved_key = str(reach_key or getattr(goal, "obj_key", ""))
    if action_kind is _OPENABLE_ACTION_KIND_UNSET:
        action_kind = _openable_action_kind(scene, goal, resolved_key)
    content_action = action_kind in ("insert", "extract")
    if not content_action and (
        not door_action
        or not action_check.uses_egocentric_camera_collision_policy(scene)
    ):
        return None
    frozen_collision_world = (
        _frozen_content_collision_world(scene, goal, resolved_key)
        if content_action
        else None
    )
    attempts = 0
    results = {}
    probe_limit = (
        CONTENT_ACTION_STAND_PROBES if content_action else OPEN_ACTION_STAND_PROBES
    )

    def _validate(stand_xy, yaw: float) -> bool:
        nonlocal attempts
        stand_key = (
            round(float(stand_xy[0]), 6),
            round(float(stand_xy[1]), 6),
            round(float(yaw), 6),
        )
        if stand_key in results:
            return results[stand_key]
        if attempts >= probe_limit:
            return False
        attempts += 1
        if content_action:
            result = _content_action_is_safe_from_stand(
                env,
                rig,
                scene,
                goal,
                resolved_key,
                stand_xy,
                yaw,
                frozen_collision_world,
            )
        else:
            result = se.openable_action_is_safe_from_stand(
                rig,
                getattr(env, "g", None),
                scene,
                env.st,
                goal,
                stand_xy,
                yaw,
                camera_frame_offset=len(env.motion.frames),
            )
        results[stand_key] = bool(result)
        return bool(result)

    if content_action:
        tolerance = _openable_content_action_reach_tolerance(
            scene,
            goal,
            resolved_key,
            action_kind=action_kind,
        )
        _validate._standpick_cache_key = (
            "content-action-preview-v1",
            action_kind,
            resolved_key,
            str(getattr(goal, "obj_key", "") or ""),
            str((getattr(goal, "meta", None) or {}).get("surface") or ""),
            _cache_key_value(getattr(goal, "place_xy", None)),
            _cache_key_value(getattr(goal, "place_z", None)),
            *(
                (field, _cache_key_value((getattr(goal, "meta", None) or {}).get(field)))
                for field in (
                    "space",
                    "place_spot",
                    "_failed_drop",
                    "_space_chosen",
                    "_space_rejected",
                    "_failed_shelf_gap_ceiling",
                )
            ),
            None if tolerance is None else repr(float(tolerance)),
        )
    return _validate


def _openable_contact_attempts(
    scene, key: str, handle
) -> tuple[tuple[np.ndarray, float], ...]:
    """Try the handle first, then progressively more reachable points on the same door leaf."""
    handle_point = np.asarray(handle, dtype=np.float64)
    entity = scene.resolve(key)
    hinge_xy = getattr(entity, "hinge_xy", None)
    if hinge_xy is None:
        return ((handle_point, 1.0),)
    return tuple(
        (
            _openable_contact_at_fraction(scene, key, handle_point, fraction),
            fraction,
        )
        for fraction in OPENABLE_PANEL_CONTACT_FRACTIONS
    )


def _openable_contact_at_fraction(scene, key: str, handle, fraction: float):
    handle_point = np.asarray(handle, dtype=np.float64)
    entity = scene.resolve(key)
    hinge_xy = getattr(entity, "hinge_xy", None)
    if hinge_xy is None:
        return handle_point
    return np.array(
        (
            float(hinge_xy[0])
            + (float(handle_point[0]) - float(hinge_xy[0])) * float(fraction),
            float(hinge_xy[1])
            + (float(handle_point[1]) - float(hinge_xy[1])) * float(fraction),
            float(handle_point[2]),
        ),
        dtype=np.float64,
    )


def _action_stand_contact_attempts(scene, key: str, goal, contact):
    if goal.prim not in (
        command_grammar.Primitive.OPEN,
        command_grammar.Primitive.CLOSE,
    ) or contact is None:
        return ((contact, None),)
    selected_fraction = (goal.meta or {}).get("_door_contact_fraction")
    if selected_fraction is not None:
        fraction = float(selected_fraction)
        return (
            (
                _openable_contact_at_fraction(scene, key, contact, fraction),
                fraction,
            ),
        )
    return _openable_contact_attempts(scene, key, contact)


def _apply_action_contact_fraction(goal, contact_fraction) -> None:
    if contact_fraction is not None:
        goal.meta["_door_contact_fraction"] = float(contact_fraction)


def _cache_key_value(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return tuple(
            (str(key), _cache_key_value(item))
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_cache_key_value(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_cache_key_value(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return value


def _validated_open_stand_key(scene, goal, reach_key: str | None = None):
    if getattr(goal, "prim", None) not in (
        command_grammar.Primitive.OPEN,
        command_grammar.Primitive.OPENDOOR,
    ):
        return None
    key = str(reach_key or getattr(goal, "obj_key", None) or "")
    if not key or key in (getattr(scene, "open_objs", None) or set()):
        return None
    entity = se._openable_access_entity(scene, key)
    meta = getattr(entity, "meta", None) or {}
    if meta.get("access_mode") != "front_opening":
        return None
    return (
        key,
        "open",
        "closed",
        *(
            (field, _cache_key_value(meta.get(field)))
            for field in (
                "open_standoff_m",
                "action_operator_lateral_offset_m",
                "action_lateral_limit_m",
                "sweep_standoff_m",
                "opening_pullback_m",
                "operator_stand_xy",
            )
        ),
        ("hinge_xy", _cache_key_value(getattr(entity, "hinge_xy", None))),
        ("handle_xy", _cache_key_value(getattr(entity, "handle_xy", None))),
        ("swing_angle", _cache_key_value(getattr(entity, "swing_angle", None))),
        ("swing_sign", _cache_key_value(getattr(entity, "swing_sign", None))),
    )


def _validated_open_stand(scene, goal, reach_key: str | None = None):
    key = _validated_open_stand_key(scene, goal, reach_key)
    cache = getattr(scene, "_validated_open_action_stands", None)
    return cache.get(key) if key is not None and isinstance(cache, dict) else None


def _remember_validated_open_stand(scene, goal, reach_key, stand) -> None:
    key = _validated_open_stand_key(scene, goal, reach_key)
    if key is None or stand is None:
        return
    cache = getattr(scene, "_validated_open_action_stands", None)
    if not isinstance(cache, dict):
        cache = {}
        scene._validated_open_action_stands = cache
    cache[key] = (
        (float(stand[0][0]), float(stand[0][1])),
        float(stand[1]),
    )


def _openable_action_kind(scene, goal, reach_key: str | None = None) -> str | None:
    primitive = getattr(goal, "prim", None)
    if primitive is command_grammar.Primitive.CLOSE:
        return "close"
    if primitive in (
        command_grammar.Primitive.OPEN,
        command_grammar.Primitive.OPENDOOR,
    ):
        return "open"
    content_kind = {
        command_grammar.Primitive.PLACE_INSIDE: "insert",
        command_grammar.Primitive.PICK: "extract",
    }.get(primitive)
    if content_kind is None or scene is None or reach_key is None:
        return None
    entity = se._openable_access_entity(scene, str(reach_key))
    meta = getattr(entity, "meta", None) or {}
    if meta.get("access_mode") not in {"front_drum", "front_opening"}:
        return None
    if any(
        field in meta
        for field in (
            f"content_{content_kind}_operator_lateral_offset_m",
            f"content_{content_kind}_standoff_m",
        )
    ):
        return content_kind
    return None


def _pick_action_aware_stand(
    picker,
    ph,
    stage,
    rig,
    scene,
    env,
    goal,
    reach_key,
    current,
    search_kwargs,
    navigation_fallback,
    *,
    current_yaw,
    gait_profile,
    gait_cum_dist,
):
    base_search_kwargs = dict(search_kwargs)
    action_kind_supplied = "openable_action_kind" in base_search_kwargs
    openable_action_kind = base_search_kwargs.pop("openable_action_kind", None)
    if not action_kind_supplied:
        openable_action_kind = _openable_action_kind(scene, goal, reach_key)
    _apply_openable_content_action_reach_tolerance(
        scene,
        goal,
        reach_key,
        action_kind=openable_action_kind,
    )
    original_fraction = (goal.meta or {}).get("_door_contact_fraction")
    contact_attempts = _action_stand_contact_attempts(
        scene,
        reach_key,
        goal,
        search_kwargs.get("contact_override"),
    )
    cached_stand = _validated_open_stand(scene, goal, reach_key)
    fixed_attempts = (cached_stand[0], None) if cached_stand is not None else (None,)
    content_preview = openable_action_kind in ("insert", "extract")
    if content_preview:
        base_search_kwargs["require_reach"] = False
        base_search_kwargs["action_state"] = env.st
    for validated_stand_xy in fixed_attempts:
        for front_standoff in _front_access_standoffs_for_goal(goal, scene):
            for contact, contact_fraction in contact_attempts:
                effective_search_kwargs = dict(base_search_kwargs)
                effective_search_kwargs["front_access_standoff_m"] = front_standoff
                effective_search_kwargs["contact_override"] = contact
                _apply_action_contact_fraction(goal, contact_fraction)
                stand = picker(
                    ph,
                    stage,
                    rig,
                    scene,
                    reach_key,
                    current,
                    **effective_search_kwargs,
                    route_gate=_ego_route_gate(scene),
                    start_yaw=current_yaw,
                    gait_profile=gait_profile,
                    gait_cum_dist=gait_cum_dist,
                    openable_action_phase=goal.prim
                    in (
                        command_grammar.Primitive.OPEN,
                        command_grammar.Primitive.CLOSE,
                    ),
                    openable_action_kind=openable_action_kind,
                    validated_stand_xy=validated_stand_xy,
                    navigation_fallback_out=(
                        navigation_fallback
                        if getattr(scene, "egocentric_camera_only_collision", True)
                        or not getattr(scene, "strict_acceptance", False)
                        else None
                    ),
                    candidate_validator=_openable_candidate_validator(
                        env,
                        rig,
                        scene,
                        goal,
                        reach_key,
                        action_kind=openable_action_kind,
                    ),
                )
                if stand is not None:
                    if validated_stand_xy is not None:
                        print(
                            f"[open-stand] '{reach_key}': revalidated cached closed-door "
                            f"stand ({stand[0][0]:.2f},{stand[0][1]:.2f})"
                        )
                    return stand
    if original_fraction is None:
        goal.meta.pop("_door_contact_fraction", None)
    else:
        goal.meta["_door_contact_fraction"] = original_fraction
    return None


def _navigation_stand_key(scene, navigation_goal, followup_goal) -> str:
    """Resolve an explicit fixture walk to the concrete item it is preparing to use."""
    navigation_key = str(navigation_goal.target_key)
    if followup_goal is None or followup_goal.obj_key is None:
        return navigation_key
    if followup_goal.obj_key == navigation_key:
        return navigation_key
    if followup_goal.prim not in (
        command_grammar.Primitive.PICK,
        command_grammar.Primitive.LOOK_AT,
        command_grammar.Primitive.FLIP,
        command_grammar.Primitive.USE_COMPUTER,
    ):
        return navigation_key
    parent = (getattr(scene, "on", None) or {}).get(followup_goal.obj_key) or (
        getattr(scene, "inside", None) or {}
    ).get(followup_goal.obj_key)
    return str(followup_goal.obj_key) if parent == navigation_key else navigation_key


def _navigation_serves_read(scene, navigation_goal, followup_goal) -> bool:
    """Return whether this navigation leg prepares a same-target visual observation."""
    return bool(
        followup_goal is not None
        and followup_goal.prim is command_grammar.Primitive.LOOK_AT
        and _navigation_stand_key(scene, navigation_goal, followup_goal)
        == getattr(followup_goal, "obj_key", None)
    )


def _navigation_previews_openable_action(
    scene,
    followup_goal,
    stand_key: str,
    *,
    serves_openable: bool,
) -> bool:
    return bool(
        serves_openable
        or _openable_action_kind(scene, followup_goal, stand_key)
        in ("insert", "extract")
    )


def _openable_content_action_reach_tolerance(
    scene,
    goal,
    reach_key: str,
    *,
    action_kind=_OPENABLE_ACTION_KIND_UNSET,
) -> float | None:
    if action_kind is _OPENABLE_ACTION_KIND_UNSET:
        action_kind = _openable_action_kind(scene, goal, reach_key)
    if action_kind == "insert":
        return se._FRONT_OPENING_PLACE_TOL_M
    if action_kind == "extract":
        return se._OPENABLE_REACH_TOL_M
    return None


def _apply_openable_content_action_reach_tolerance(
    scene,
    goal,
    reach_key: str,
    *,
    action_kind=_OPENABLE_ACTION_KIND_UNSET,
) -> float | None:
    tolerance = _openable_content_action_reach_tolerance(
        scene,
        goal,
        reach_key,
        action_kind=action_kind,
    )
    if tolerance is not None:
        goal.meta["_egocentric_reach_tol_m"] = float(tolerance)
    return tolerance


def _has_stationary_operator_anchor(entity) -> bool:
    meta = getattr(entity, "meta", None) or {}
    if not meta.get("stationary_operator"):
        return False
    try:
        anchor = np.asarray(meta.get("operator_stand_xy"), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(anchor.shape == (2,) and np.all(np.isfinite(anchor)))


def navigation_stand_request(
    scene, navigation_goal, followup_goal, *, actor_xy=None
) -> dict:
    """Build the action-aware stand constraints shared by preflight and runtime."""
    stand_key = _navigation_stand_key(scene, navigation_goal, followup_goal)
    serves_read = _navigation_serves_read(scene, navigation_goal, followup_goal)
    serves_openable = bool(
        followup_goal is not None
        and followup_goal.prim
        in (
            command_grammar.Primitive.OPEN,
            command_grammar.Primitive.CLOSE,
        )
        and getattr(followup_goal, "obj_key", None) == navigation_goal.target_key
    )
    serves_toggle = bool(
        followup_goal is not None
        and followup_goal.prim
        in (
            command_grammar.Primitive.TURN_ON,
            command_grammar.Primitive.TURN_OFF,
        )
        and getattr(followup_goal, "obj_key", None) == navigation_goal.target_key
    )
    serves_computer = bool(
        followup_goal is not None
        and followup_goal.prim is command_grammar.Primitive.USE_COMPUTER
        and stand_key == getattr(followup_goal, "obj_key", None)
    )
    is_nav_door = navigation_goal.target_key in (getattr(scene, "doors", None) or {})
    is_lidded = _is_lidded_openable(scene, stand_key)
    contact = None
    entity = scene.resolve(stand_key)
    stationary_operator = _has_stationary_operator_anchor(entity)
    if is_nav_door or (
        (serves_openable or is_lidded)
        and getattr(entity, "handle_xy", None) is not None
    ):
        contact = _door_handle_contact(
            scene,
            stand_key,
            opened=bool(
                serves_openable
                and followup_goal.prim is command_grammar.Primitive.CLOSE
            ),
        )
    contact_attempts = ((contact, 1.0),)
    if serves_openable and contact is not None:
        contact_attempts = _openable_contact_attempts(scene, stand_key, contact)
    if serves_toggle:
        contact_overrides = se.contact_trajectory_for_goal(scene, None, followup_goal)
    elif serves_computer:
        origin = actor_xy if actor_xy is not None else entity.xy
        contact_overrides = se.computer_contact_trajectory(
            scene, str(followup_goal.obj_key), origin
        )
    else:
        contact_overrides = None
    return {
        "stand_key": stand_key,
        "serves_openable": serves_openable,
        "centered_openable": bool(
            serves_openable or is_nav_door or stationary_operator
        ),
        "serves_read": serves_read,
        "serves_computer": serves_computer,
        "is_nav_door": is_nav_door,
        "is_lidded": is_lidded,
        "contact_attempts": contact_attempts,
        "contact_overrides": contact_overrides,
        "complete_contact_trajectory": serves_computer,
        "require_reach": followup_goal is not None and not serves_read,
        "front_access_standoff_m": _front_access_standoff_for_goal(
            followup_goal, scene
        ),
    }


def _navigation_stand_reach_tolerance(
    scene, stand_request, followup_goal=None
) -> float | None:
    """Use the tolerance that the follow-up producer will enforce."""
    if stand_request["is_nav_door"]:
        return se._OPENABLE_REACH_TOL_M
    if stand_request["serves_openable"] or stand_request["is_lidded"]:
        return se._OPENABLE_REACH_TOL_M
    content_tolerance = _openable_content_action_reach_tolerance(
        scene, followup_goal, str(stand_request["stand_key"])
    )
    if content_tolerance is not None:
        return content_tolerance
    if stand_request.get("serves_computer"):
        return se.standing_computer_reach_tolerance()
    return None


def _allow_actor_near_stand_fallback(
    stand_request,
    *,
    content_preview: bool = False,
) -> bool:
    """An Open/Close must re-position instead of inheriting an arbitrary nearby side stance."""
    return not content_preview and not bool(stand_request.get("centered_openable"))


def _active_low_container_crouch_owner(scene, state, close_goal) -> str | None:
    if close_goal is None or close_goal.prim is not command_grammar.Primitive.CLOSE:
        return None
    owner = getattr(state, "crouch_owner", None)
    if (
        owner is None
        or float(getattr(state, "crouch_fraction", 0.0)) <= 1e-6
        or str(owner) != str(close_goal.obj_key)
        or str(owner) not in (getattr(scene, "open_objs", None) or set())
    ):
        return None
    try:
        entity = scene.resolve(str(owner))
    except (KeyError, ValueError):
        return None
    if (getattr(entity, "meta", None) or {}).get("access_mode") != "front_opening":
        return None
    return str(owner) if se._low_front_openable(scene, entity) else None


def _preceding_container_interaction_owner(
    scene,
    state,
    goals,
    goal_index: int,
    followup_goal,
) -> str | None:
    """Identify a continuous interaction -> close handoff for one front container."""
    if (
        followup_goal is None
        or followup_goal.prim is not command_grammar.Primitive.CLOSE
    ):
        return None
    if goal_index <= 0:
        return _active_low_container_crouch_owner(scene, state, followup_goal)
    previous_index = goal_index - 1
    previous = goals[previous_index]
    if previous.prim is command_grammar.Primitive.NAVIGATE_TO:
        if str(previous.target_key) != str(followup_goal.obj_key):
            return None
        return _active_low_container_crouch_owner(scene, state, followup_goal)
    if previous.prim is command_grammar.Primitive.WAIT:
        try:
            duration_s = float((previous.meta or {}).get("duration_s", WAIT_DEFAULT_S))
        except (TypeError, ValueError):
            return None
        if duration_s > WAIT_DEFAULT_S or previous_index <= 0:
            return None
        previous = goals[previous_index - 1]
    held = getattr(state, "held", None)
    if previous.prim is command_grammar.Primitive.PICK:
        held = held or {}
        if held.get("obj_key") != previous.obj_key:
            return None
        owner = held.get("source_open_container")
    elif previous.prim is command_grammar.Primitive.PLACE_INSIDE:
        if held is not None:
            return None
        owner = (previous.meta or {}).get("surface")
    elif previous.prim is command_grammar.Primitive.OPEN:
        owner = previous.obj_key
    else:
        return None
    if owner is None or str(owner) != str(followup_goal.obj_key):
        return None
    if str(owner) not in (getattr(scene, "open_objs", None) or set()):
        return None
    try:
        entity = scene.resolve(str(owner))
    except (KeyError, ValueError):
        return None
    if (getattr(entity, "meta", None) or {}).get("access_mode") != "front_opening":
        return None
    return str(owner)


def _continuous_container_close_owner(
    scene,
    state,
    goals,
    goal_index: int,
    navigation_goal,
    followup_goal,
) -> str | None:
    if navigation_goal.target_key != getattr(followup_goal, "obj_key", None):
        return None
    owner = _preceding_container_interaction_owner(
        scene,
        state,
        goals,
        goal_index,
        followup_goal,
    )
    if owner is None:
        return None
    navigation_goal.meta["_continuous_front_opening_close"] = owner
    entity = scene.resolve(owner)
    aperture = np.asarray(
        (getattr(entity, "meta", None) or {}).get("aperture_center", ()),
        dtype=np.float64,
    )
    if aperture.shape == (3,) and np.all(np.isfinite(aperture)):
        navigation_goal.meta["_navigation_fixed_gaze"] = [
            float(value) for value in aperture
        ]
    return str(owner)


def _mark_direct_continuous_container_close(
    scene,
    state,
    goals,
    goal_index: int,
    close_goal,
) -> str | None:
    owner = _preceding_container_interaction_owner(
        scene,
        state,
        goals,
        goal_index,
        close_goal,
    )
    if owner is not None:
        close_goal.meta["_reuse_current_stand"] = owner
    return owner


def _continuous_container_close_stand(
    scene,
    env,
    rig,
    goals,
    goal_index: int,
    navigation_goal,
    followup_goal,
    stand_request,
):
    owner = _continuous_container_close_owner(
        scene,
        env.st,
        goals,
        goal_index,
        navigation_goal,
        followup_goal,
    )
    inferred_standalone_close = owner is None
    if owner is None:
        candidate_owner = str(getattr(followup_goal, "obj_key", "") or "")
        if (
            followup_goal is None
            or goal_index != 0
            or followup_goal.prim is not command_grammar.Primitive.CLOSE
            or str(navigation_goal.target_key) != candidate_owner
            or candidate_owner
            not in (getattr(scene, "open_objs", None) or set())
        ):
            return None
        owner = candidate_owner
    entity = scene.resolve(owner)
    entity_meta = getattr(entity, "meta", None) or {}
    if inferred_standalone_close and (
        entity_meta.get("access_mode") != "front_opening"
        or entity_meta.get("close_reposition")
        or se._low_front_openable(scene, entity)
    ):
        return None
    if (
        entity_meta.get("close_reposition")
        and (navigation_goal.meta or {}).get("_close_reposition_walk") == owner
    ):
        return None
    current = tuple(float(value) for value in env.st.root_xy)
    yaw = float(env.st.root_yaw)
    original_fraction = (followup_goal.meta or {}).get("_door_contact_fraction")
    for _contact, fraction in stand_request["contact_attempts"]:
        followup_goal.meta["_door_contact_fraction"] = float(fraction)
        if se.openable_action_is_safe_from_stand(
            rig,
            getattr(env, "g", None),
            scene,
            env.st,
            followup_goal,
            current,
            yaw,
            camera_frame_offset=len(env.motion.frames),
        ):
            if inferred_standalone_close:
                navigation_goal.meta["_continuous_front_opening_close"] = owner
                aperture = np.asarray(
                    entity_meta.get("aperture_center", ()), dtype=np.float64
                )
                if aperture.shape == (3,) and np.all(np.isfinite(aperture)):
                    navigation_goal.meta["_navigation_fixed_gaze"] = [
                        float(value) for value in aperture
                    ]
            navigation_goal.meta["_reuse_current_stand"] = owner
            followup_goal.meta["_reuse_current_stand"] = owner
            return current, yaw
    if original_fraction is None:
        followup_goal.meta.pop("_door_contact_fraction", None)
    else:
        followup_goal.meta["_door_contact_fraction"] = original_fraction
    return None


def _open_goal_blocked_by_failed_navigation(
    failed_navigation_commands: set[int], command_index: int, goal
) -> bool:
    """Do not execute an open action after its source command's navigation failed."""
    return int(command_index) in failed_navigation_commands and goal.prim in (
        command_grammar.Primitive.OPEN,
        command_grammar.Primitive.OPENDOOR,
    )


def _preflight_navigation_route_is_valid(
    scene,
    rig,
    env,
    current_xy,
    stand,
    *,
    gait_profile,
    gait_cum_dist: float,
    navigation_fallback: dict,
    transit_avoid=(),
) -> bool:
    """Check a stand using the same temporary route exclusions as runtime."""
    grid = scene.nav_grid
    prior_transit_avoid = set(grid.transit_avoid)
    grid.transit_avoid.clear()
    grid.transit_avoid.update(transit_avoid)
    try:
        return _preflight_navigation_route_on_grid(
            scene,
            rig,
            env,
            current_xy,
            stand,
            gait_profile=gait_profile,
            gait_cum_dist=gait_cum_dist,
            navigation_fallback=navigation_fallback,
        )
    finally:
        grid.transit_avoid.clear()
        grid.transit_avoid.update(prior_transit_avoid)


def _preflight_relaxed_route(scene, grid, current, target):
    import door_planner
    import nav_grid
    import nav_planner
    from goal_schema import PrimitiveFailure

    relaxed = nav_grid.egocentric_passable_view(grid, scene)
    relaxed_path = None
    if relaxed is not None:
        try:
            relaxed_path = nav_planner.plan_path(relaxed, current, target)
        except PrimitiveFailure:
            pass
    by_index = {
        door.prim_index: door for door in (getattr(scene, "doors", None) or {}).values()
    }
    crossed = (
        door_planner._crossed_doors(grid, relaxed_path or [current, target])
        if getattr(grid, "door_id", None) is not None
        else ()
    )
    opened = set(getattr(scene, "open_objs", None) or ())
    unopened_doors = tuple(
        by_index[index]
        for index in crossed
        if by_index.get(index) is not None and by_index[index].entity_key not in opened
    )
    return relaxed, relaxed_path, unopened_doors


def _preflight_camera_route(
    scene,
    rig,
    env,
    current,
    target,
    target_yaw,
    gait_profile,
    gait_cum_dist,
    navigation_fallback,
    *,
    additional_excluded_roots=(),
    additional_open_leaf_boxes=(),
) -> bool:
    import init_validation as iv

    state = getattr(env, "st", None)
    held = getattr(state, "held", None)
    arm_idle = held.get("grasp_arm") if hasattr(held, "get") else None
    if arm_idle is None:
        arm_idle = getattr(rig, "arm_rest", getattr(rig, "arm_grasp", None))
    path_overrides = []
    accepted = navigation_fallback.get("path")
    if accepted:
        path_overrides.append(accepted)
    path_overrides.append(None)
    if navigation_fallback.get("force_straight"):
        path_overrides.append((current, target))
    return any(
        iv.camera_safe_navigation_path(
            scene,
            rig,
            current,
            target,
            target_yaw,
            start_yaw=float(getattr(state, "root_yaw", 0.0)),
            gait_profile=gait_profile,
            gait_cum_dist=gait_cum_dist,
            arm_idle=arm_idle,
            path_override=path_override,
            camera_frame_offset=len(
                getattr(getattr(env, "motion", None), "frames", ())
            ),
            additional_excluded_roots=additional_excluded_roots,
            additional_open_leaf_boxes=additional_open_leaf_boxes,
        )
        is not None
        for path_override in path_overrides
    )


def _preflight_open_door_route(
    scene,
    rig,
    env,
    current,
    target,
    target_yaw,
    gait_profile,
    gait_cum_dist,
    navigation_fallback,
    relaxed_grid,
    unopened_doors,
) -> bool | None:
    """Preview crossed doors at the same swing choices used by runtime."""
    import action_check

    if not action_check.uses_egocentric_camera_collision_policy(scene):
        return None
    controllers = getattr(scene, "_openable_controllers", None) or {}
    excluded_roots = []
    open_leaf_boxes = []
    for door in unopened_doors:
        ctrl = controllers.get(door.entity_key)
        entity = scene.resolve(door.entity_key)
        if (
            ctrl is None
            or ctrl.get("kind") != "revolute"
            or ctrl.get("openable_kind") == "lid"
            or not getattr(entity, "leaf_paths", None)
        ):
            return None

        def _camera_route_for_sign(sign: float, _door=door):
            return _camera_safe_door_swing_route(
                env,
                rig,
                scene,
                controllers,
                _door,
                current,
                target,
                target_yaw,
                sign,
            )

        current_sign, selected_sign, _camera_route = _choose_runtime_door_swing(
            scene,
            ctrl,
            current,
            target,
            relaxed_grid,
            _camera_route_for_sign,
        )
        sign = current_sign if selected_sign is None else selected_sign
        boxes = se._open_leaf_boxes(
            scene,
            entity,
            swing_direction_scale=float(sign) / current_sign,
        )
        if not boxes:
            return None
        excluded_roots.extend(entity.leaf_paths)
        open_leaf_boxes.extend(boxes)

    return _preflight_camera_route(
        scene,
        rig,
        env,
        current,
        target,
        target_yaw,
        gait_profile,
        gait_cum_dist,
        navigation_fallback,
        additional_excluded_roots=tuple(excluded_roots),
        additional_open_leaf_boxes=tuple(open_leaf_boxes),
    )


def _preflight_navigation_route_on_grid(
    scene,
    rig,
    env,
    current_xy,
    stand,
    *,
    gait_profile,
    gait_cum_dist: float,
    navigation_fallback: dict,
) -> bool:
    """Reject a stand only when no direct, door-mediated, or narrow path can exist."""
    import nav_planner
    from goal_schema import PrimitiveFailure

    grid = scene.nav_grid
    current = (float(current_xy[0]), float(current_xy[1]))
    target = (float(stand[0][0]), float(stand[0][1]))
    if not navigation_fallback.get("force_straight"):
        try:
            nav_planner.plan_path(grid, current, target)
            return True
        except PrimitiveFailure:
            pass
    relaxed, relaxed_path, unopened_doors = _preflight_relaxed_route(
        scene, grid, current, target
    )
    if unopened_doors:
        open_door_route = _preflight_open_door_route(
            scene,
            rig,
            env,
            current,
            target,
            float(stand[1]),
            gait_profile,
            gait_cum_dist,
            navigation_fallback,
            relaxed,
            unopened_doors,
        )
        if open_door_route is None:
            return True
        return open_door_route or relaxed_path is not None
    return (
        _preflight_camera_route(
            scene,
            rig,
            env,
            current,
            target,
            float(stand[1]),
            gait_profile,
            gait_cum_dist,
            navigation_fallback,
        )
        or relaxed_path is not None
    )


def _current_front_opening_close_stand(
    rig,
    scene,
    env,
    goal,
    reach_key,
    current,
    current_yaw,
    contact,
    *,
    require_open_state: bool = True,
):
    """Return the live stance only when a complete Close preview accepts it."""
    entity = se._openable_access_entity(scene, str(reach_key))
    meta = getattr(entity, "meta", None) or {}
    if (
        entity is None
        or goal.prim is not command_grammar.Primitive.CLOSE
        or str(reach_key) != str(goal.obj_key)
        or (
            require_open_state
            and str(goal.obj_key)
            not in (getattr(scene, "open_objs", None) or set())
        )
        or meta.get("access_mode") != "front_opening"
        or meta.get("close_reposition")
    ):
        return None
    original_fraction = (goal.meta or {}).get("_door_contact_fraction")
    for _contact, fraction in _action_stand_contact_attempts(
        scene, str(reach_key), goal, contact
    ):
        _apply_action_contact_fraction(goal, fraction)
        if se.openable_action_is_safe_from_stand(
            rig,
            getattr(env, "g", None),
            scene,
            env.st,
            goal,
            current,
            current_yaw,
            camera_frame_offset=len(env.motion.frames),
        ):
            return (tuple(float(value) for value in current), float(current_yaw))
    if original_fraction is None:
        goal.meta.pop("_door_contact_fraction", None)
    else:
        goal.meta["_door_contact_fraction"] = original_fraction
    return None


def _preflight_continuous_close_stand(
    rig,
    scene,
    env,
    previous_goal,
    goal,
    reach_key,
    current,
    current_yaw,
    contact,
):
    if (
        previous_goal is None
        or previous_goal.prim is not command_grammar.Primitive.PLACE_INSIDE
        or str((previous_goal.meta or {}).get("surface")) != str(goal.obj_key)
    ):
        return None
    return _current_front_opening_close_stand(
        rig,
        scene,
        env,
        goal,
        reach_key,
        current,
        current_yaw,
        contact,
        require_open_state=False,
    )


def _preflight_action_stand(
    ph,
    stage,
    rig,
    scene,
    env,
    goal,
    current,
    current_yaw,
    gait_profile,
    gait_cum_dist,
    *,
    previous_goal=None,
):
    """Return an action's reachable stand and any definite preflight failure phase."""
    import init_validation as iv

    entity = (getattr(scene, "entities", None) or {}).get(goal.obj_key)
    if (
        goal.prim is command_grammar.Primitive.USE_COMPUTER
        and entity is not None
        and (getattr(entity, "meta", None) or {}).get("workstation_seat")
    ):
        # The compiler already inserts and validates navigation to the workstation seat. The
        # seated typing trajectory is intentionally not reachable from a standing hand pose.
        return None

    request = _retry_stand_search_request(env, scene, goal)
    if request is None:
        return None
    reach_key, search_kwargs = request
    navigation_fallback = {}
    stand = _preflight_continuous_close_stand(
        rig,
        scene,
        env,
        previous_goal,
        goal,
        reach_key,
        current,
        current_yaw,
        search_kwargs.get("contact_override"),
    )
    if stand is None:
        stand = _pick_action_aware_stand(
            iv.pick_stand_runtime,
            ph,
            stage,
            rig,
            scene,
            env,
            goal,
            reach_key,
            current,
            search_kwargs,
            navigation_fallback,
            current_yaw=current_yaw,
            gait_profile=gait_profile,
            gait_cum_dist=gait_cum_dist,
        )
    if stand is None:
        return reach_key, None, "action"
    if not _preflight_navigation_route_is_valid(
        scene,
        rig,
        env,
        current,
        stand,
        gait_profile=gait_profile,
        gait_cum_dist=gait_cum_dist,
        navigation_fallback=navigation_fallback,
    ):
        return reach_key, stand, "action_route"
    _remember_validated_open_stand(scene, goal, reach_key, stand)
    return reach_key, stand, None


def _allow_legacy_close_navigation_preview(scene, goal, followup) -> bool:
    behavior_task = getattr(scene, "behavior_task", None) or {}
    return bool(
        behavior_task.get("allow_legacy_close_navigation_preview") is True
        and goal.prim is command_grammar.Primitive.NAVIGATE_TO
        and goal.target_key is not None
        and followup is not None
        and followup.prim is command_grammar.Primitive.CLOSE
        and followup.obj_key == goal.target_key
    )


def _preflight_navigation_stand(
    ph,
    stage,
    rig,
    scene,
    env,
    goal,
    followup,
    current,
    current_yaw,
    gait_profile,
    gait_cum_dist,
    *,
    preview_contact_fractions=True,
):
    """Return a navigation stand and any definite stand/route failure phase."""
    import init_validation as iv

    request = navigation_stand_request(scene, goal, followup, actor_xy=current)
    openable_action_kind = _openable_action_kind(
        scene, followup, request["stand_key"]
    )
    previews_openable_action = _navigation_previews_openable_action(
        scene,
        followup,
        request["stand_key"],
        serves_openable=request["serves_openable"],
    )
    _apply_openable_content_action_reach_tolerance(
        scene, followup, request["stand_key"]
    )
    content_preview = openable_action_kind in ("insert", "extract")
    stand = None
    transit_avoid = set()
    navigation_fallback = {}
    missing_contact_fraction = object()
    followup_meta = getattr(followup, "meta", None)
    tracks_contact_fraction = bool(
        request["serves_openable"] and isinstance(followup_meta, dict)
    )
    original_contact_fraction = (
        followup_meta.get("_door_contact_fraction", missing_contact_fraction)
        if tracks_contact_fraction
        else missing_contact_fraction
    )
    legacy_close_fallback = bool(
        not preview_contact_fractions
        and _allow_legacy_close_navigation_preview(scene, goal, followup)
    )

    def restore_contact_fraction() -> None:
        if not tracks_contact_fraction:
            return
        if original_contact_fraction is missing_contact_fraction:
            followup_meta.pop("_door_contact_fraction", None)
        else:
            followup_meta["_door_contact_fraction"] = original_contact_fraction

    cached_stand = _validated_open_stand(scene, followup, request["stand_key"])
    fixed_attempts = (cached_stand[0], None) if cached_stand is not None else (None,)
    for validated_stand_xy in fixed_attempts:
        for front_standoff in _front_access_standoffs_for_goal(followup, scene):
            for contact, fraction in request["contact_attempts"]:
                transit_avoid = set()
                navigation_fallback = {}
                if tracks_contact_fraction:
                    if preview_contact_fractions:
                        _apply_action_contact_fraction(followup, fraction)
                    else:
                        followup_meta.pop("_door_contact_fraction", None)
                keep_contact_fraction = False
                try:
                    stand = iv.pick_stand_runtime(
                        ph,
                        stage,
                        rig,
                        scene,
                        request["stand_key"],
                        current,
                        contact_override=contact,
                        contact_overrides=request["contact_overrides"],
                        complete_contact_trajectory=request[
                            "complete_contact_trajectory"
                        ],
                        nearest_to_contact=request["serves_openable"],
                        cue_read=request["serves_read"],
                        require_reach=(
                            False if content_preview else request["require_reach"]
                        ),
                        require_openable_sweep_clearance=bool(
                            request["is_nav_door"]
                            or (
                                request["serves_openable"]
                                and _requires_openable_sweep_clearance(scene, followup)
                            )
                        ),
                        openable_action_phase=bool(request["serves_openable"]),
                        openable_action_kind=openable_action_kind,
                        front_access_standoff_m=front_standoff,
                        validated_stand_xy=validated_stand_xy,
                        route_gate=_ego_route_gate(scene),
                        start_yaw=current_yaw,
                        gait_profile=gait_profile,
                        gait_cum_dist=gait_cum_dist,
                        reach_tol=_navigation_stand_reach_tolerance(
                            scene, request, followup
                        ),
                        transit_avoid_out=transit_avoid,
                        navigation_fallback_out=(
                            navigation_fallback
                            if getattr(scene, "egocentric_camera_only_collision", True)
                            or not getattr(scene, "strict_acceptance", False)
                            else None
                        ),
                        candidate_validator=(
                            _openable_candidate_validator(
                                env,
                                rig,
                                scene,
                                followup,
                                request["stand_key"],
                            )
                            if env is not None
                            and previews_openable_action
                            and not legacy_close_fallback
                            else None
                        ),
                        action_state=(
                            env.st
                            if env is not None
                            and (request["serves_openable"] or content_preview)
                            else None
                        ),
                        centered_openable=request["centered_openable"],
                    )
                    keep_contact_fraction = stand is not None
                finally:
                    if not keep_contact_fraction:
                        restore_contact_fraction()
                if stand is not None:
                    break
            if stand is not None:
                break
        if stand is None:
            continue
        route_valid = False
        try:
            route_valid = _preflight_navigation_route_is_valid(
                scene,
                rig,
                env,
                current,
                stand,
                gait_profile=gait_profile,
                gait_cum_dist=gait_cum_dist,
                navigation_fallback=navigation_fallback,
                transit_avoid=transit_avoid,
            )
        finally:
            if not route_valid:
                restore_contact_fraction()
        if not route_valid:
            if validated_stand_xy is None:
                return str(request["stand_key"]), stand, "navigation_route"
            stand = None
            continue
        _remember_validated_open_stand(scene, followup, request["stand_key"], stand)
        if validated_stand_xy is not None:
            print(
                f"[open-stand] '{request['stand_key']}': revalidated cached closed-door "
                f"stand ({stand[0][0]:.2f},{stand[0][1]:.2f})"
            )
        return str(request["stand_key"]), stand, None
    if (
        stand is None
        and preview_contact_fractions
        and _allow_legacy_close_navigation_preview(scene, goal, followup)
    ):
        restore_contact_fraction()
        print(
            f"[open-stand] '{request['stand_key']}': retrying navigation without "
            "the coupled close preview; the Close goal remains independently validated"
        )
        return _preflight_navigation_stand(
            ph,
            stage,
            rig,
            scene,
            env,
            goal,
            followup,
            current,
            current_yaw,
            gait_profile,
            gait_cum_dist,
            preview_contact_fractions=False,
        )
    if not preview_contact_fractions:
        restore_contact_fraction()
        return str(request["stand_key"]), None, "navigation"
    if (
        stand is None
        and _allow_actor_near_stand_fallback(
            request,
            content_preview=content_preview,
        )
        and iv.actor_near_entity(scene, request["stand_key"], current)
    ):
        stand = (current, current_yaw)
    if stand is None:
        return str(request["stand_key"]), None, "navigation"
    if not _preflight_navigation_route_is_valid(
        scene,
        rig,
        env,
        current,
        stand,
        gait_profile=gait_profile,
        gait_cum_dist=gait_cum_dist,
        navigation_fallback=navigation_fallback,
        transit_avoid=transit_avoid,
    ):
        return str(request["stand_key"]), stand, "navigation_route"
    _remember_validated_open_stand(scene, followup, request["stand_key"], stand)
    return str(request["stand_key"]), stand, None


def program_reachability_preflight(
    ph,
    stage,
    rig,
    scene,
    goals,
    compiled_commands,
    start_xy,
    *,
    env=None,
    gait_profile=None,
) -> dict:
    """Run runtime navigation and action stand selection without mutating its grid."""
    import nav_grid

    runtime_grid = getattr(scene, "nav_grid", None)
    if runtime_grid is None:
        return {
            "ok": True,
            "checked": 0,
            "end_xy": tuple(start_xy),
            "navigation_stands": {},
        }
    scene.nav_grid = nav_grid.planning_view(runtime_grid)
    current = (float(start_xy[0]), float(start_xy[1]))
    current_yaw = 0.0
    gait_cum_dist = 0.0
    checked = 0
    navigation_stands = {}
    try:
        for index, goal in enumerate(goals):
            is_unresolved_navigation = (
                goal.prim is command_grammar.Primitive.NAVIGATE_TO
                and goal.target_xy is None
                and goal.target_key is not None
            )
            if not is_unresolved_navigation:
                if env is None:
                    continue
                action_result = _preflight_action_stand(
                    ph,
                    stage,
                    rig,
                    scene,
                    env,
                    goal,
                    current,
                    current_yaw,
                    gait_profile,
                    gait_cum_dist,
                    previous_goal=goals[index - 1] if index > 0 else None,
                )
                if action_result is None:
                    continue
                reach_key, stand, failure_phase = action_result
                checked += 1
                if failure_phase is not None:
                    return {
                        "ok": False,
                        "phase": failure_phase,
                        "checked": checked,
                        "goal_index": index,
                        "target_key": reach_key,
                        "stand_key": reach_key,
                        "from_xy": current,
                        "navigation_stands": navigation_stands,
                    }
                gait_cum_dist += math.dist(current, stand[0])
                current = (float(stand[0][0]), float(stand[0][1]))
                current_yaw = float(stand[1])
                continue
            followup = command_grammar.navigation_action_followup(
                goals, compiled_commands, index, scene
            )
            stand_key, stand, failure_phase = _preflight_navigation_stand(
                ph,
                stage,
                rig,
                scene,
                env,
                goal,
                followup,
                current,
                current_yaw,
                gait_profile,
                gait_cum_dist,
            )
            checked += 1
            if failure_phase is not None:
                return {
                    "ok": False,
                    "phase": failure_phase,
                    "checked": checked,
                    "goal_index": index,
                    "target_key": str(goal.target_key),
                    "stand_key": stand_key,
                    "from_xy": current,
                    "navigation_stands": navigation_stands,
                }
            navigation_stands[index] = {
                "target_key": str(goal.target_key),
                "stand_key": str(stand_key),
                "xy": (float(stand[0][0]), float(stand[0][1])),
            }
            gait_cum_dist += math.dist(current, stand[0])
            current = (float(stand[0][0]), float(stand[0][1]))
            current_yaw = float(stand[1])
    finally:
        scene.nav_grid = runtime_grid
    return {
        "ok": True,
        "checked": checked,
        "end_xy": current,
        "navigation_stands": navigation_stands,
    }


def _retry_reach_tolerance(
    scene,
    goal,
    reach_key,
    *,
    placement: bool,
    openable: bool,
    contact_motion: bool,
    action_kind=_OPENABLE_ACTION_KIND_UNSET,
) -> float | None:
    content_tolerance = _openable_content_action_reach_tolerance(
        scene,
        goal,
        str(reach_key),
        action_kind=action_kind,
    )
    if content_tolerance is not None:
        return content_tolerance
    reach_entity = se._openable_access_entity(scene, reach_key)
    access_mode = (getattr(reach_entity, "meta", None) or {}).get("access_mode")
    placement_intent = (goal.meta or {}).get("_placement_intent") or {}
    places_inside = (
        goal.prim is command_grammar.Primitive.PLACE_INSIDE
        or bool((goal.meta or {}).get("into"))
        or placement_intent.get("predicate") == "INSIDE"
    )
    front_opening = access_mode in {"front_drum", "front_opening"} or (
        bool(getattr(reach_entity, "leaf_paths", ()))
        and not bool((getattr(reach_entity, "meta", None) or {}).get("open_top"))
    )
    if placement and places_inside and front_opening:
        return se._FRONT_OPENING_PLACE_TOL_M
    if placement and places_inside:
        return 0.10
    if contact_motion:
        if goal.prim is command_grammar.Primitive.USE_COMPUTER:
            return se.standing_computer_reach_tolerance()
        return se._reach_tolerance(scene)
    if openable or not getattr(scene, "strict_acceptance", False):
        return se._OPENABLE_REACH_TOL_M
    return None


def _front_insert_standoff(scene, entity) -> float | None:
    if entity is None:
        return None
    meta = getattr(entity, "meta", None) or {}
    configured = meta.get("content_insert_standoff_m")
    if (
        meta.get("access_mode") in {"front_drum", "front_opening"}
        and configured is not None
    ):
        return float(configured)
    if not se._low_front_openable(scene, entity):
        return None
    legacy = meta.get("content_standoff_m")
    return float(legacy) if legacy is not None else se._LOW_FRONT_INSERT_STANDOFF_M


def _front_access_standoff_for_goal(goal, scene=None) -> float | None:
    if goal is not None and goal.prim in (
        command_grammar.Primitive.TURN_ON,
        command_grammar.Primitive.TURN_OFF,
    ):
        target_key = str(goal.obj_key or "")
        entity = (
            se._openable_access_entity(scene, target_key)
            if scene is not None and target_key
            else None
        )
        meta = getattr(entity, "meta", None) or {}
        configured_standoff = meta.get("action_standoff_m")
        if (
            meta.get("access_mode") in {"front_drum", "front_opening"}
            and configured_standoff is not None
        ):
            return float(configured_standoff)
        return se._FRONT_APPLIANCE_TOGGLE_STANDOFF_M
    if goal is not None and goal.prim is command_grammar.Primitive.PLACE_INSIDE:
        target_key = str((goal.meta or {}).get("surface") or "")
        entity = (getattr(scene, "entities", None) or {}).get(target_key)
        return _front_insert_standoff(scene, entity)
    if goal is not None and goal.prim is command_grammar.Primitive.PICK:
        target_key = str(goal.obj_key or "")
        entity = (
            se._openable_access_entity(scene, target_key)
            if scene is not None and target_key
            else None
        )
        meta = getattr(entity, "meta", None) or {}
        configured_standoff = meta.get("content_extract_standoff_m")
        if (
            meta.get("access_mode") in {"front_drum", "front_opening"}
            and configured_standoff is not None
        ):
            return float(configured_standoff)
    if goal is not None and goal.prim in (
        command_grammar.Primitive.OPEN,
        command_grammar.Primitive.CLOSE,
    ):
        target_key = str(goal.obj_key or "")
        entity = (
            se._openable_access_entity(scene, target_key)
            if scene is not None and target_key
            else None
        )
        if (
            getattr(entity, "meta", None) or {}
        ).get("access_mode") == "front_opening":
            meta = getattr(entity, "meta", None) or {}
            if goal.prim is command_grammar.Primitive.OPEN:
                configured_open_standoff = meta.get("open_standoff_m")
                if configured_open_standoff is not None:
                    return float(configured_open_standoff)
            if goal.prim is command_grammar.Primitive.CLOSE:
                configured_close_standoff = meta.get("close_standoff_m")
                if configured_close_standoff is not None:
                    return float(configured_close_standoff)
            if meta.get("stationary_operator") or se._low_front_openable(
                scene, entity
            ):
                configured_standoff = meta.get("action_standoff_m")
                if configured_standoff is not None:
                    return float(configured_standoff)
            return (
                se._FRONT_OPENING_ACTION_CAMERA_STANDOFF_M
                if goal.prim is command_grammar.Primitive.CLOSE
                else se._FRONT_OPENING_STANDOFF_M
            )
    return None


def _front_access_standoffs_for_goal(goal, scene=None) -> tuple[float | None, ...]:
    primary = _front_access_standoff_for_goal(goal, scene)
    if primary == se._LOW_FRONT_INSERT_STANDOFF_M:
        return (primary, se._FRONT_OPENING_CONTENT_STANDOFF_M)
    if primary == se._FRONT_OPENING_STANDOFF_M:
        return (primary, se._FRONT_OPENING_ACTION_REACH_FALLBACK_M)
    if primary == se._FRONT_OPENING_ACTION_CAMERA_STANDOFF_M:
        return (primary, se._FRONT_OPENING_ACTION_CAMERA_FALLBACK_M)
    return (primary,)


def _retry_stand_search_request(env, scene, goal) -> tuple[str, dict] | None:
    """Return the shared action-aware stand-search target and constraints."""
    placement = goal.prim in (
        command_grammar.Primitive.PLACE,
        command_grammar.Primitive.PLACE_INSIDE,
        command_grammar.Primitive.PLACE_ON_FLOOR,
        command_grammar.Primitive.ATTACH,
    )
    toggle = goal.prim in (
        command_grammar.Primitive.TURN_ON,
        command_grammar.Primitive.TURN_OFF,
    )
    openable = goal.prim in (
        command_grammar.Primitive.OPEN,
        command_grammar.Primitive.CLOSE,
    )
    pick = goal.prim is command_grammar.Primitive.PICK
    failed_contacts = (goal.meta or {}).get("_failed_contacts")
    remembered_contacts = getattr(env.st, "failed_contact_trajectory", None)
    if (
        not failed_contacts
        and remembered_contacts is not None
        and remembered_contacts[0] == goal.prim.value
        and remembered_contacts[1] == goal.obj_key
    ):
        failed_contacts = remembered_contacts[2]
    if not failed_contacts:
        failed_contacts = se.contact_trajectory_for_goal(scene, env.st, goal)
    contact_motion = goal.prim in (
        command_grammar.Primitive.CUT,
        command_grammar.Primitive.STIR,
        command_grammar.Primitive.POUR,
        command_grammar.Primitive.WIPE,
        command_grammar.Primitive.COOK,
        command_grammar.Primitive.USE_COMPUTER,
        command_grammar.Primitive.TURN_ON,
        command_grammar.Primitive.TURN_OFF,
    ) and bool(failed_contacts)
    if placement:
        reach_key = (
            (goal.meta or {}).get("surface")
            or (goal.meta or {}).get("parent")
            or (goal.meta or {}).get("_floor_anchor")
            or (goal.meta or {}).get("anchor")
        )
    else:
        reach_key = (
            (goal.meta or {}).get("_contact_target_key")
            or getattr(env.st, "failed_contact_target_key", None)
            or goal.obj_key
        )
    if (
        not (placement or toggle or openable or pick or contact_motion)
        or reach_key not in scene.entities
        or (
            not (pick or toggle or openable or contact_motion)
            and (env.st.held is None or (goal.meta or {}).get("_failed_drop") is None)
        )
    ):
        return None
    contact_override = (goal.meta or {}).get("_failed_drop") if placement else None
    contact_overrides = failed_contacts if contact_motion else None
    if toggle and not contact_motion:
        affordance = scene.affordance(reach_key, Role.TOGGLEABLE)
        if affordance is None or affordance.xy is None or affordance.z is None:
            return None
        contact_override = (
            float(affordance.xy[0]),
            float(affordance.xy[1]),
            float(affordance.z),
        )
    elif openable:
        contact_override = _door_handle_contact(
            scene,
            reach_key,
            opened=goal.prim is command_grammar.Primitive.CLOSE,
        )
    action_kind = _openable_action_kind(scene, goal, str(reach_key))
    front_standoff = _front_access_standoff_for_goal(goal, scene)
    explicit_lateral_anchor = se.explicit_openable_operator_anchor(
        scene,
        str(reach_key),
        opened_content=action_kind in {"insert", "extract"},
        action_kind=action_kind,
        standoff_override=front_standoff,
    )
    return str(reach_key), {
        "contact_override": contact_override,
        "contact_overrides": contact_overrides,
        "shelf_gap_ceiling": (goal.meta or {}).get("_failed_shelf_gap_ceiling"),
        "require_openable_sweep_clearance": (
            openable and _requires_openable_sweep_clearance(scene, goal)
        ),
        "front_access_standoff_m": front_standoff,
        "nearest_to_contact": (
            goal.prim is command_grammar.Primitive.ATTACH or openable
        ),
        "reach_tol": _retry_reach_tolerance(
            scene,
            goal,
            reach_key,
            placement=placement,
            openable=openable,
            contact_motion=contact_motion,
            action_kind=action_kind,
        ),
        "complete_contact_trajectory": (
            goal.prim is command_grammar.Primitive.USE_COMPUTER
        ),
        "action_state": env.st if openable else None,
        "centered_openable": openable or explicit_lateral_anchor is not None,
        "openable_action_kind": action_kind,
    }


def _needs_front_opening_action_stand(scene, goal) -> bool:
    if goal.prim not in (
        command_grammar.Primitive.OPEN,
        command_grammar.Primitive.CLOSE,
    ):
        return False
    entity = se._openable_access_entity(scene, str(goal.obj_key))
    if entity is None:
        return False
    meta = getattr(entity, "meta", None) or {}
    return bool(
        meta.get("access_mode") == "front_opening"
        and not se._low_front_openable(scene, entity)
    )


def _requires_openable_sweep_clearance(scene, goal) -> bool:
    if not _needs_front_opening_action_stand(scene, goal):
        return True
    if goal is None or goal.prim is not command_grammar.Primitive.CLOSE:
        return False
    entity = se._openable_access_entity(scene, str(goal.obj_key))
    return bool((getattr(entity, "meta", None) or {}).get("close_reposition"))


def _prepare_front_opening_action_stand(
    env,
    ph,
    stage,
    rig,
    scene,
    goal,
    iv_module,
    *,
    controllers=None,
    open_doors_fn=None,
    run_goal_fn=None,
) -> tuple[bool, str | None]:
    """Move a tall front-opening action to its camera-safe sweep stance first."""
    reuse_owner = (goal.meta or {}).get("_reuse_current_stand")
    if (
        goal.prim is command_grammar.Primitive.CLOSE
        and str(reuse_owner) == str(goal.obj_key)
        and str(goal.obj_key) in (getattr(scene, "open_objs", None) or set())
    ):
        current = tuple(float(value) for value in env.st.root_xy)
        yaw = float(env.st.root_yaw)
        if se.openable_action_is_safe_from_stand(
            rig,
            getattr(env, "g", None),
            scene,
            env.st,
            goal,
            current,
            yaw,
            camera_frame_offset=len(env.motion.frames),
        ):
            print(
                f"[open-stand] '{goal.obj_key}': preserving the previewed "
                "continuous close stand"
            )
            return True, None
        goal.meta.pop("_reuse_current_stand", None)
    if not _needs_front_opening_action_stand(scene, goal):
        return True, None
    request = _retry_stand_search_request(env, scene, goal)
    if request is None:
        return False, f"no action-aware stand contract for '{goal.obj_key}'"
    reach_key, search_kwargs = request
    current = tuple(env.st.root_xy)
    current_yaw = float(getattr(env.st, "root_yaw", 0.0))
    current_stand = _current_front_opening_close_stand(
        rig,
        scene,
        env,
        goal,
        reach_key,
        current,
        current_yaw,
        search_kwargs.get("contact_override"),
    )
    if current_stand is not None:
        print(
            f"[open-stand] '{goal.obj_key}': preserving the fully previewed "
            "current close stand"
        )
        return True, None
    navigation_fallback = {}
    stand = _pick_action_aware_stand(
        iv_module.pick_stand_runtime,
        ph,
        stage,
        rig,
        scene,
        env,
        goal,
        reach_key,
        current,
        search_kwargs,
        navigation_fallback,
        current_yaw=current_yaw,
        gait_profile=getattr(env, "g", None),
        gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
    )
    if stand is None:
        return False, f"no camera-safe sweep stand for '{goal.obj_key}'"
    if not _stand_pose_changed(env.st, stand):
        return True, None
    open_doors = open_doors_fn or _runtime_open_doors
    if not open_doors(
        env,
        ph,
        stage,
        rig,
        scene,
        current,
        stand[0],
        controllers=controllers,
        target_yaw=stand[1],
        gait_profile=getattr(env, "g", None),
        gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
        allow_camera_safe_fallback=bool(navigation_fallback.get("force_straight")),
        navigation_fallback_out=(
            navigation_fallback
            if getattr(scene, "egocentric_camera_only_collision", True)
            or not getattr(scene, "strict_acceptance", False)
            else None
        ),
    ):
        return False, f"camera-safe sweep stand for '{goal.obj_key}' is unreachable"
    navigation_meta = {}
    if navigation_fallback.get("force_straight"):
        navigation_meta["_camera_safe_navigation_fallback"] = True
    if navigation_fallback.get("path"):
        navigation_meta["_camera_safe_navigation_path"] = navigation_fallback["path"]
    run_goal = run_goal_fn or env.run_goal
    _, _, navigation_info = run_goal(
        Goal(
            command_grammar.Primitive.NAVIGATE_TO,
            target_xy=stand[0],
            target_yaw=stand[1],
            meta=navigation_meta,
        )
    )
    if not navigation_info["ok"]:
        return False, (
            f"walk to camera-safe sweep stand for '{goal.obj_key}' failed: "
            f"{navigation_info.get('detail') or navigation_info['reason'].value}"
        )
    print(
        f"[open-stand] '{goal.obj_key}': normalized to camera-safe sweep stand "
        f"({stand[0][0]:.2f},{stand[0][1]:.2f}) yaw={stand[1]:.0f}"
    )
    return True, None


def _run_reachable_stand_retry(
    env,
    ph,
    stage,
    rig,
    scene,
    goal,
    info,
    placement_retry,
    reach_key,
    stand,
    navigation_fallback,
    controllers=None,
):
    cur = tuple(env.st.root_xy)
    needs_move = _stand_pose_changed(env.st, stand)
    if needs_move:
        if not _runtime_open_doors(
            env,
            ph,
            stage,
            rig,
            scene,
            cur,
            stand[0],
            controllers=controllers,
            target_yaw=stand[1],
            gait_profile=getattr(env, "g", None),
            gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
            allow_camera_safe_fallback=bool(
                navigation_fallback.get("force_straight")
            ),
            navigation_fallback_out=(
                navigation_fallback
                if getattr(scene, "egocentric_camera_only_collision", True)
                or not getattr(scene, "strict_acceptance", False)
                else None
            ),
        ):
            return info, placement_retry, False
        print(
            f"[reach-renav] '{reach_key}': target unreachable from "
            f"({cur[0]:.2f},{cur[1]:.2f}) -> move/turn to reach-capable pose "
            f"({stand[0][0]:.2f},{stand[0][1]:.2f}, yaw={stand[1]:.1f}) + retry"
        )
    else:
        print(
            f"[reach-renav] '{reach_key}': retrying at the current stand with the "
            "bounded egocentric reach fallback"
        )
    if not getattr(scene, "strict_acceptance", False):
        scene.egocentric_collision_fallback_used = True
        goal.meta["_egocentric_reach_tol_m"] = se._OPENABLE_REACH_TOL_M
    nav_meta = {}
    if navigation_fallback.get("force_straight"):
        nav_meta["_camera_safe_navigation_fallback"] = True
    if navigation_fallback.get("path"):
        nav_meta["_camera_safe_navigation_path"] = navigation_fallback["path"]
    if needs_move:
        _, _, navigation_info = env.run_goal(
            Goal(
                command_grammar.Primitive.NAVIGATE_TO,
                target_xy=stand[0],
                target_yaw=stand[1],
                meta=nav_meta,
            )
        )
        if not navigation_info["ok"]:
            return navigation_info, placement_retry, False
    if placement_retry is not None:
        placement_retry = _capture_placement_retry(
            env, scene, goal, len(env.motion.frames)
        )
    _obs, _done, info = env.run_goal(goal)
    return info, placement_retry, True


def _retry_from_reachable_stand(
    env,
    ph,
    stage,
    rig,
    scene,
    goal,
    info,
    placement_retry,
    iv_module,
    controllers=None,
):
    retryable = (
        PrimitiveError.OUT_OF_REACH,
        PrimitiveError.TRAJECTORY_COLLISION,
    )
    if info["reason"] not in retryable or getattr(scene, "nav_grid", None) is None:
        return info, placement_retry
    request = _retry_stand_search_request(env, scene, goal)
    if request is None:
        return info, placement_retry
    reach_key, search_kwargs = request
    base_search_kwargs = dict(search_kwargs)
    openable_action_kind = base_search_kwargs.pop("openable_action_kind", None)
    _apply_openable_content_action_reach_tolerance(
        scene,
        goal,
        reach_key,
        action_kind=openable_action_kind,
    )
    content_preview = openable_action_kind in ("insert", "extract")
    if content_preview:
        base_search_kwargs["require_reach"] = False
        base_search_kwargs["action_state"] = env.st
    for standoff_index, front_standoff in enumerate(
        _front_access_standoffs_for_goal(goal, scene)
    ):
        navigation_fallback = {}
        effective_search_kwargs = dict(base_search_kwargs)
        effective_search_kwargs["front_access_standoff_m"] = front_standoff
        stand = iv_module.pick_stand_runtime(
            ph,
            stage,
            rig,
            scene,
            reach_key,
            tuple(env.st.root_xy),
            **effective_search_kwargs,
            route_gate=_ego_route_gate(scene),
            start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
            gait_profile=getattr(env, "g", None),
            gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
            openable_action_phase=goal.prim
            in (
                command_grammar.Primitive.OPEN,
                command_grammar.Primitive.CLOSE,
            ),
            openable_action_kind=openable_action_kind,
            navigation_fallback_out=(
                navigation_fallback
                if getattr(scene, "egocentric_camera_only_collision", True)
                or not getattr(scene, "strict_acceptance", False)
                else None
            ),
            candidate_validator=_openable_candidate_validator(
                env,
                rig,
                scene,
                goal,
                reach_key,
                action_kind=openable_action_kind,
            ),
        )
        if stand is None:
            continue
        if standoff_index:
            print(
                f"[reach-renav] '{reach_key}': nominal front-opening stand remained "
                f"unreachable; trying validated {front_standoff:.2f}m standoff"
            )
        info, placement_retry, action_ran = _run_reachable_stand_retry(
            env,
            ph,
            stage,
            rig,
            scene,
            goal,
            info,
            placement_retry,
            reach_key,
            stand,
            navigation_fallback,
            controllers,
        )
        if not action_ran or info["reason"] not in retryable:
            break
    return info, placement_retry


def _retry_from_alternate_space(env, scene, goal, info, placement_retry):
    target_key = (goal.meta or {}).get("surface")
    target = (
        scene.target_view(target_key, Role.CONTAINER)
        if target_key in scene.entities
        else None
    )
    spaces = getattr(target, "spaces", None) or ()
    tries = 0
    while (
        info["reason"] is PrimitiveError.OUT_OF_REACH
        and goal.prim is command_grammar.Primitive.PLACE_INSIDE
        and env.st.held is not None
        and len(spaces) > 1
        and goal.meta.get("_space_chosen") is not None
        and tries < len(spaces) - 1
    ):
        tries += 1
        rejected = set(goal.meta.get("_space_rejected") or ())
        rejected.add(goal.meta["_space_chosen"])
        goal.meta["_space_rejected"] = rejected
        if len(rejected) >= len(spaces):
            break
        print(
            f"[place-rechoose] '{target_key}': {goal.meta['_space_chosen']} unreachable even from a "
            f"reach-capable stand -> reject it and try another interior space "
            f"({len(rejected)}/{len(spaces)} rejected)"
        )
        if placement_retry is not None:
            placement_retry = _capture_placement_retry(
                env, scene, goal, len(env.motion.frames)
            )
        _obs, _done, info = env.run_goal(goal)
    return info, placement_retry


def _recover_out_of_reach_action(
    env,
    ph,
    stage,
    rig,
    scene,
    goal,
    info,
    placement_retry,
    iv_module,
    controllers=None,
):
    info, placement_retry = _retry_from_reachable_stand(
        env,
        ph,
        stage,
        rig,
        scene,
        goal,
        info,
        placement_retry,
        iv_module,
        controllers=controllers,
    )
    return _retry_from_alternate_space(env, scene, goal, info, placement_retry)


# ---- Asset relative paths (under fpv_core.ASSET_ROOT) --------------------------------------------
_CABINET = "/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
_CHAIR = "/Environments/Office/Props/SM_Chair.usd"
_SOFA = "/Environments/Office/Props/SM_Sofa.usd"

# ---- Clutter scene: SIX diverse stock props spawned by spawn_prop, then dropped from a MODEST float
# onto a static tabletop. XY is NO LONGER hand-picked -- P2(A): build_clutter rejection-samples every
# prop's XY with object_physics.plan_spawn_layout so ANY draw is NON-OVERLAPPING (physically valid at
# spawn), which is what a RANDOMLY-generated RL scene (BEHAVIOR-1K next) needs. The env-init multi-body
# settle then drops each prop the modest float onto its physics-correct rest. General: each is a stock
# USD by path, footprint half-extent read from its world bbox, collider auto-classified from geometry,
# XY from the seeded planner, rest z from PhysX. Only (name, usd_rel, yaw_deg) is authored per prop; the
# yaw is about +Z only so every prop spawns UPRIGHT and a modest float drops it straight down.
#
# _CLUTTER_SPAWN_Z = the modest drop float (m) above the tabletop -- big enough that the settle visibly
# plays (the recorded-trajectory drop), small enough that a prop only falls a few cm and stays standing
# (a tall float tips a cylinder/box onto its side / adds roll on impact).
# _CLUTTER_SEED / _CLUTTER_MIN_GAP / _CLUTTER_EDGE_MARGIN drive the reproducible reject-sampler.
_CLUTTER = (
    ("Can", "Props/YCB/Axis_Aligned/002_master_chef_can.usd", 0.0),  # upright cylinder
    ("Cracker", "Props/YCB/Axis_Aligned/003_cracker_box.usd", 0.0),  # flat box
    ("Sugar", "Props/YCB/Axis_Aligned/004_sugar_box.usd", 0.0),  # flat box
    ("Bowl", "Props/YCB/Axis_Aligned/024_bowl.usd", 0.0),  # rounded bowl
    ("Banana", "Props/YCB/Axis_Aligned/011_banana.usd", 30.0),  # long, rests on side
    ("Brick", "Props/YCB/Axis_Aligned/061_foam_brick.usd", 10.0),  # low foam brick
)
_CLUTTER_SPAWN_Z = (
    0.04  # modest drop float above the tabletop (visible settle, stays upright)
)
_CLUTTER_SEED = 7  # fixed seed -> reproducible non-overlapping layout
_CLUTTER_MIN_GAP = 0.02  # min free space (m) the planner leaves between prop footprints
# Shrink the tabletop by this before sampling (a straight-drop prop stays on). The plank is SHALLOW in Y
# (0.45 m) so a symmetric 10 cm margin removes ~44% of the depth -> 6 props + the reserved front-center
# mug lane cannot pack without overlap; 6 cm still generously keeps a prop on (the planner pre-separates
# them so the settle drops each ~straight down with no lateral shove, unlike the old mis-stack layout).
_CLUTTER_EDGE_MARGIN = 0.06
_CLUTTER_MAX_TRIES = 400  # rejection-sampler tries per object (>default so the tight strip packs cleanly)

# ---- Scene layout (meters, Z-up). Human starts at +Y, furniture in front (-Y). ------------------
# Two cabinets: "cabinet" raised on a base so its top-drawer handle reaches UP (~1.15); "drawer"
# on the floor -> reach mid/down. Different XY so the per-frame drawer router separates them.
# Both drawers open TOWARD the human, so the human stands back > travel to clear it -> the reach is
# far, and the analytic warp reaches HIGH targets cleanly but low ones poorly (dry3: z=1.15->7mm,
# z=0.78->76mm, z=0.71->91mm > the 90mm gate). So place both handles at reach-friendly heights (a
# high cabinet + a chest-height drawer); a true low reach-down needs deeper IK (Step-4).
_CAB_XY = (-0.6, -0.55)
_CAB_BASE_H = 0.44  # reach-up cabinet: top-drawer handle 0.71 -> ~1.15 (reach_err ~1mm)
_DRW_XY = (0.6, -0.55)
_DRW_BASE_H = 0.31  # chest-height drawer: handle -> ~1.02 (clean reach)
_OPEN_TRAVEL = (
    0.28  # drawer slide distance (< the 0.4 joint limit; < stand gap so it clears)
)
_START_OC = (0.0, 1.5)

# Two seats. Seat-cushion z is tuned here (refined by the critic render).
_CHAIR_XY = (-0.7, -0.2)
_CHAIR_SEAT_Z = 0.47
_SOFA_XY = (1.05, -0.2)  # long axis runs along X after Rz+90; keep clear of the chair
_SOFA_SEAT_Z = 0.43
_START_SIT = (0.0, 1.5)


# =================================================================================================
# USD helpers (post-launch: pxr available).
# =================================================================================================
def _deinstance(prim) -> None:
    """Flip every instanceable prim in the subtree to non-instanceable so sub-prims are editable.
    PrimRange stops at instance boundaries, so iterate until none remain."""
    from pxr import Usd

    for _ in range(8):
        found = [p for p in Usd.PrimRange(prim) if p.IsInstanceable()]
        if not found:
            return
        for p in found:
            p.SetInstanceable(False)


def _find_by_name(root, name: str):
    """First prim in the subtree whose leaf name == ``name`` (paths shift under add_reference)."""
    from pxr import Usd

    for p in Usd.PrimRange(root):
        if p.GetName() == name:
            return p
    return None


def _read_prim_l2w(stage, prim_path: str):
    """Local-to-world 4x4 (row-major flat list) of a prim -- used by the clutter --check to read a
    settled prop's full pose (for the upright/tilt diagnostic)."""
    from pxr import Usd, UsdGeom

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    return xc.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path))


def _world_center(prim) -> np.ndarray:
    import fpv_scene

    rng = fpv_scene.compute_world_bbox(prim)
    mn, mx = rng.GetMin(), rng.GetMax()
    return np.array(
        [0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), 0.5 * (mn[2] + mx[2])]
    )


def _camera_object_local_bounds(
    stage, scene, rig
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Intrinsic visual bounds for every graspable the FPV may carry."""
    from pxr import Usd, UsdGeom

    paths = {
        str(entity.prim_path)
        for entity in scene.entities.values()
        if entity.role is Role.GRASPABLE and entity.prim_path
    }
    paths.add(str(getattr(rig, "_obj_path", "/World/Obj")))
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bounds = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not (prim and prim.IsValid()):
            continue
        rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        minimum, maximum = rng.GetMin(), rng.GetMax()
        bounds[path] = (
            np.array(minimum, dtype=np.float64),
            np.array(maximum, dtype=np.float64),
        )
    return bounds


def _transformed_camera_aabb(
    transform: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    minimum, maximum = bounds
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    world = (np.asarray(transform, dtype=np.float64) @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _camera_object_aabbs(fr, obj_T, rig, local_bounds):
    if obj_T is None or fr.obj[0] not in ("carry", "pose"):
        return []
    transforms = [(getattr(fr, "obj_path", None) or rig._obj_path, obj_T)]
    transforms.extend(
        (path, np.asarray(obj_T, dtype=np.float64) @ np.asarray(relative))
        for path, relative in (getattr(fr, "carried", ()) or ())
    )
    return [
        _transformed_camera_aabb(transform, local_bounds[path])
        for path, transform in transforms
        if path in local_bounds
    ]


def _large_container_frame(scene, frame, local_bounds) -> bool:
    path = str(getattr(frame, "obj_path", None) or "").rstrip("/")
    if not path or not local_bounds:
        return False
    entity = next(
        (
            candidate
            for candidate in (getattr(scene, "entities", None) or {}).values()
            if str(getattr(candidate, "prim_path", None) or "").rstrip("/") == path
        ),
        None,
    )
    if entity is None:
        return False
    category = str(
        (getattr(entity, "meta", None) or {}).get("asset_category") or ""
    ).casefold().replace("-", "_")
    if category != "lid" and category != "box" and not category.endswith("_box"):
        return False
    bounds = local_bounds.get(path) or local_bounds.get(path + "/")
    if bounds is None:
        return False
    minimum, maximum = (np.asarray(value, dtype=np.float64) for value in bounds)
    long_span, short_span = sorted(
        (float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1])),
        reverse=True,
    )
    return bool(
        long_span >= se._LARGE_CONTAINER_CARRY_MIN_LONG_M
        and short_span >= se._LARGE_CONTAINER_CARRY_MIN_SHORT_M
    )


def _bounded_front_content_camera_pullback(
    scene,
    target,
    segment: str,
    requested_pullback: float,
) -> float:
    if segment.startswith("front_insert") and se._low_front_openable(scene, target):
        max_pullback = max(
            0.0,
            ego_camera.DEFAULT_CAMERA_CONFIG.manip_eye_forward
            - LOW_FRONT_INSERT_CAMERA_MIN_FORWARD_OF_HEAD_M,
        )
        return min(requested_pullback, max_pullback)
    return requested_pullback


def _configured_front_content_camera_pullback(scene, frame, segment: str):
    target_key = getattr(frame, "openable_key", None)
    if target_key is None:
        return None
    try:
        target = scene.resolve(str(target_key))
    except (KeyError, ValueError):
        return None
    target_meta = getattr(target, "meta", None) or {}
    phase_field = (
        "content_extract_camera_pullback_m"
        if segment.startswith("front_extract")
        else "content_insert_camera_pullback_m"
    )
    requested_pullback = target_meta.get(
        phase_field,
        target_meta.get("content_camera_pullback_m"),
    )
    if requested_pullback is None:
        return None
    return _bounded_front_content_camera_pullback(
        scene,
        target,
        segment,
        float(requested_pullback),
    )


def _camera_pullback_for_frame(scene, frame, local_bounds=None) -> float:
    segment = str(getattr(frame, "seg", ""))
    if segment.startswith("front_extract") and getattr(
        frame, "camera_eye_offset", None
    ) is not None:
        return 0.0
    if segment.startswith(("front_extract", "front_insert")):
        configured_pullback = _configured_front_content_camera_pullback(
            scene,
            frame,
            segment,
        )
        if configured_pullback is not None:
            return configured_pullback
    if segment.startswith("front_extract"):
        return LARGE_FRONT_CONTAINER_CAMERA_PULLBACK_M
    if segment.startswith("front_insert"):
        return (
            LARGE_FRONT_CONTAINER_CAMERA_PULLBACK_M
            if _large_container_frame(scene, frame, local_bounds)
            else FRONT_INSERT_CAMERA_PULLBACK_M
        )
    key = getattr(frame, "openable_key", None)
    if key is None:
        return 0.0
    try:
        entity = se._openable_access_entity(scene, str(key))
    except (KeyError, ValueError):
        return 0.0
    meta = getattr(entity, "meta", None) or {}
    if (
        meta.get("access_mode") == "front_opening"
        and not se._low_front_openable(scene, entity)
    ):
        return FRONT_OPENING_CAMERA_PULLBACK_M
    return 0.0


def _camera_forward_for_frame(scene, frame) -> float:
    segment = str(getattr(frame, "seg", ""))
    if not segment.startswith("front_insert"):
        return 0.0
    target_key = getattr(frame, "openable_key", None)
    if target_key is None:
        return 0.0
    try:
        target = scene.resolve(str(target_key))
    except (KeyError, ValueError):
        return 0.0
    return float(
        (getattr(target, "meta", None) or {}).get(
            "content_insert_camera_forward_m",
            0.0,
        )
    )


def _camera_lateral_for_frame(scene, frame) -> float:
    segment = str(getattr(frame, "seg", ""))
    if not segment.startswith("front_insert"):
        return 0.0
    target_key = getattr(frame, "openable_key", None)
    if target_key is None:
        return 0.0
    try:
        target = scene.resolve(str(target_key))
    except (KeyError, ValueError):
        return 0.0
    return float(
        (getattr(target, "meta", None) or {}).get(
            "content_insert_camera_lateral_m",
            0.0,
        )
    )


def _pulled_back_camera_pose(eye, look, yaw: float, pullback_m: float):
    """Move the FPV behind a close carried object or an outward-swinging leaf."""
    original_eye = np.asarray(eye, dtype=np.float64)
    original_look = np.asarray(look, dtype=np.float64)
    if pullback_m <= 0.0:
        return tuple(original_eye), tuple(original_look)
    aim = original_eye + original_look
    adjusted_eye = original_eye - np.asarray(wpp._body_forward(yaw)) * pullback_m
    adjusted_look = aim - adjusted_eye
    adjusted_look /= np.linalg.norm(adjusted_look) + 1e-9
    return tuple(adjusted_eye), tuple(adjusted_look)


def _moved_forward_camera_pose(eye, look, forward_m: float):
    original_eye = np.asarray(eye, dtype=np.float64)
    original_look = np.asarray(look, dtype=np.float64)
    if forward_m <= 0.0:
        return tuple(original_eye), tuple(original_look)
    direction = original_look / (np.linalg.norm(original_look) + 1e-9)
    return tuple(original_eye + direction * forward_m), tuple(direction)


def _shifted_lateral_camera_pose(eye, look, yaw: float, lateral_m: float):
    original_eye = np.asarray(eye, dtype=np.float64)
    original_look = np.asarray(look, dtype=np.float64)
    if lateral_m == 0.0:
        return tuple(original_eye), tuple(original_look)
    aim = original_eye + original_look
    heading = math.radians(float(yaw))
    camera_left = np.asarray(
        (-math.cos(heading), -math.sin(heading), 0.0),
        dtype=np.float64,
    )
    adjusted_eye = original_eye + camera_left * lateral_m
    adjusted_look = aim - adjusted_eye
    adjusted_look /= np.linalg.norm(adjusted_look) + 1e-9
    return tuple(adjusted_eye), tuple(adjusted_look)


def _head_mount_frame_camera_pose(pose, frame):
    """Apply a recorded camera-rig translation to the rigid head mount once."""
    camera_eye_offset = getattr(frame, "camera_eye_offset", None)
    if camera_eye_offset is None:
        return pose
    import fpv_render

    eye = np.asarray(pose.eye, dtype=np.float64) + np.asarray(
        camera_eye_offset,
        dtype=np.float64,
    )
    return fpv_render.CameraRenderPose(
        tuple(float(value) for value in eye),
        pose.look,
        pose.focal_mm,
        pose.up,
    )


def _stabilize_large_front_container_camera_pose(
    scene,
    frame,
    eye,
    look,
    local_bounds,
):
    segment = str(getattr(frame, "seg", ""))
    if not (
        segment.startswith("front_insert")
        or segment.startswith("front_extract")
    ) or not _large_container_frame(scene, frame, local_bounds):
        return tuple(eye), tuple(look)
    floor_z = float(getattr(scene, "floor_z", 0.0))
    minimum_eye_z = floor_z + LARGE_FRONT_CONTAINER_CAMERA_MIN_EYE_HEIGHT_M
    original_eye = np.asarray(eye, dtype=np.float64)
    if float(original_eye[2]) >= minimum_eye_z:
        return tuple(original_eye), tuple(look)
    aim = original_eye + np.asarray(look, dtype=np.float64)
    adjusted_eye = original_eye.copy()
    adjusted_eye[2] = minimum_eye_z
    adjusted_look = aim - adjusted_eye
    adjusted_look /= np.linalg.norm(adjusted_look) + 1e-9
    return tuple(adjusted_eye), tuple(adjusted_look)


def _preserve_large_front_container_egocentric_pose(
    scene,
    frame,
    eye,
    look,
    yaw,
    head_world,
    local_bounds,
):
    segment = str(getattr(frame, "seg", ""))
    if not (
        segment.startswith("front_insert")
        or segment.startswith("front_extract")
    ) or not _large_container_frame(scene, frame, local_bounds):
        return tuple(eye), tuple(look)

    original_eye = np.asarray(eye, dtype=np.float64)
    original_look = np.asarray(look, dtype=np.float64)
    head = np.asarray(head_world, dtype=np.float64)
    forward = np.asarray(wpp._body_forward(yaw), dtype=np.float64)
    forward[2] = 0.0
    forward /= np.linalg.norm(forward) + 1e-9
    lateral = np.array((-forward[1], forward[0], 0.0), dtype=np.float64)
    delta = original_eye - head
    forward_offset = float(np.dot(delta, forward))
    lateral_offset = float(np.dot(delta, lateral))

    adjusted_eye = head.copy()
    adjusted_eye += forward * float(
        np.clip(
            forward_offset,
            LARGE_FRONT_CONTAINER_CAMERA_MIN_FORWARD_OF_HEAD_M,
            LARGE_FRONT_CONTAINER_CAMERA_MAX_FORWARD_OF_HEAD_M,
        )
    )
    adjusted_eye += lateral * float(
        np.clip(
            lateral_offset,
            -LARGE_FRONT_CONTAINER_CAMERA_MAX_LATERAL_FROM_HEAD_M,
            LARGE_FRONT_CONTAINER_CAMERA_MAX_LATERAL_FROM_HEAD_M,
        )
    )
    adjusted_eye[2] = min(
        float(original_eye[2]),
        float(head[2]) + LARGE_FRONT_CONTAINER_CAMERA_MAX_ABOVE_HEAD_M,
    )
    aim = original_eye + original_look
    adjusted_look = aim - adjusted_eye
    adjusted_look /= np.linalg.norm(adjusted_look) + 1e-9
    return tuple(adjusted_eye), tuple(adjusted_look)


def _camera_object_clearance_for_frame(frame) -> float:
    segment = str(getattr(frame, "seg", ""))
    if segment.startswith("front_extract"):
        return LARGE_FRONT_CONTAINER_CAMERA_PULLBACK_M
    if segment.startswith("front_insert"):
        return FRONT_INSERT_CARRIED_CAMERA_CLEARANCE_M
    obj = getattr(frame, "obj", ("rest",))
    return 0.18 if obj and obj[0] == "pose" else 0.04


def _require_multi_object_carry_identity(scene, fr) -> None:
    """Reject an ambiguous carried pose before replay can silently freeze its prim."""
    if (
        getattr(scene, "grasp_objs", None)
        and fr.obj[0] == "carry"
        and not getattr(fr, "obj_path", None)
    ):
        raise RuntimeError("multi-object carry frame is missing obj_path")


def _find_drawer_joint(prim, drawer: str):
    """The drawer's PhysX PrismaticJoint under the referenced Sektion cabinet (after de-instancing).
    Fetch by the known relative path first, then fall back to a type+name scan."""
    from pxr import Usd, UsdPhysics

    stage = prim.GetStage()
    jp = stage.GetPrimAtPath(f"{prim.GetPath()}/sektion/{drawer}_joint")
    if jp and jp.IsValid() and jp.IsA(UsdPhysics.PrismaticJoint):
        return jp
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdPhysics.PrismaticJoint) and drawer in p.GetName():
            return p
    return None


def _find_prismatic_joint(
    prim,
    joint_name: str | None,
    leaf_name: str | None = None,
):
    from pxr import Usd, UsdPhysics

    candidates = [
        candidate
        for candidate in Usd.PrimRange(prim)
        if candidate.IsA(UsdPhysics.PrismaticJoint)
    ]
    if joint_name:
        exact = [
            candidate
            for candidate in candidates
            if candidate.GetName() == joint_name
            or str(candidate.GetPath()).rstrip("/").endswith(f"/{joint_name}")
        ]
        if exact:
            return exact[0]
    if leaf_name:
        for candidate in candidates:
            targets = UsdPhysics.PrismaticJoint(candidate).GetBody1Rel().GetTargets()
            if any(
                str(target).rstrip("/").endswith(f"/{leaf_name}") for target in targets
            ):
                return candidate
    return candidates[0] if len(candidates) == 1 else None


def _make_joint_setter(joint_prim):
    """Stiffen the drawer's linear drive and return a setter for its target position (meters along the
    joint-local axis; Rz+90 maps it to world +Y). Driving the JOINT is what actually moves the drawer
    under the playing timeline -- PhysX owns the articulated body's pose and ignores kinematic xforms.
    The handle is welded to the drawer body by a FixedJoint, so it slides along for free.
    """
    from pxr import UsdPhysics

    drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear") or UsdPhysics.DriveAPI.Apply(
        joint_prim, "linear"
    )
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(
        2.0e5
    )  # shipped 10 is far too soft to move/hold the drawer
    drive.CreateDampingAttr().Set(2.0e4)
    drive.CreateMaxForceAttr().Set(1.0e6)
    tgt = drive.GetTargetPositionAttr()

    def _set(t: float) -> None:
        tgt.Set(float(t))

    _set(0.0)
    return _set


_SLIDE_OP_NAME = "xformOp:translate:openableSlide"


def _make_leaf_slide_setter(leaf, axis_world):
    """Drive a stripped prismatic leaf kinematically along a world-space axis.

    The shared BEHAVIOR cache intentionally removes articulation joints from its
    geometry-only USDs. Openable fixtures still retain the joint/leaf facts in
    their specs, so use an authored leaf translation when no live joint remains.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    parent_to_world = UsdGeom.XformCache().GetLocalToWorldTransform(leaf.GetParent())
    world_to_parent = parent_to_world.GetInverse()
    origin = world_to_parent.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    tip = world_to_parent.Transform(Gf.Vec3d(*(float(value) for value in axis_world)))
    local_delta = tip - origin

    xformable = UsdGeom.Xformable(leaf)
    prior = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    slide = prior.get(_SLIDE_OP_NAME) or xformable.AddTranslateOp(
        opSuffix="openableSlide"
    )
    other_ops = [
        op for op in xformable.GetOrderedXformOps() if op.GetOpName() != _SLIDE_OP_NAME
    ]
    xformable.SetXformOpOrder([slide] + other_ops)

    rigid = UsdPhysics.RigidBodyAPI.Apply(leaf)
    rigid.CreateKinematicEnabledAttr(True)

    def _set(distance: float) -> None:
        slide.Set(
            Gf.Vec3d(*(float(local_delta[i]) * float(distance) for i in range(3)))
        )

    _set(0.0)
    return _set


def _make_prismatic_setter(joint, leaf, axis_world):
    if joint is not None:
        return _make_joint_setter(joint), "joint"
    return _make_leaf_slide_setter(leaf, axis_world), "kinematic-leaf"


def _make_box(stage, path: str, size, center, color=(0.55, 0.45, 0.35)):
    """A static matte box (UsdGeom.Cube unit-cube scaled to ``size``, centered at ``center``)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    )
    xf.AddScaleOp().Set(Gf.Vec3d(float(size[0]), float(size[1]), float(size[2])))
    mtl = UsdShade.Material.Define(stage, path + "/Mat")
    sh = UsdShade.Shader.Define(stage, path + "/Mat/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
        mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    return cube.GetPrim()


def _place_toggle_indicators(stage, scene) -> dict:
    """Add one small, non-colliding status lamp to every registered toggleable device."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Xform.Define(stage, "/World/ToggleIndicators")
    indicators = {}
    for key in sorted(scene.entities):
        toggle = scene.affordance(key, Role.TOGGLEABLE)
        if not _uses_synthetic_toggle_indicator(scene, key, toggle):
            continue
        press = np.asarray(
            (toggle.meta or {}).get("press_dir", (0.0, 0.0, -1.0)),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(press)) or 1.0
        outward = -press / norm
        center = np.array([toggle.xy[0], toggle.xy[1], toggle.z], dtype=np.float64)
        center += outward * 0.025
        safe = "".join(char if char.isalnum() or char == "_" else "_" for char in key)
        root = f"/World/ToggleIndicators/{safe}"
        housing = _make_box(
            stage,
            root + "/Housing",
            (0.06, 0.06, 0.035),
            center,
            color=(0.025, 0.03, 0.025),
        )
        lamp = _make_box(
            stage,
            root + "/On",
            (0.042, 0.042, 0.038),
            center + outward * 0.003,
            color=(0.03, 1.0, 0.08),
        )
        shader = UsdShade.Shader.Get(stage, root + "/On/Mat/S")
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.03, 1.0, 0.08)
        )
        for prim in (housing, lamp):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
        imageable = UsdGeom.Imageable(lamp)
        if key in scene.running:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        indicators[key] = imageable
        scene.entities[key].meta["toggle_indicator_path"] = str(lamp.GetPath())
    scene.toggle_indicators = indicators
    return indicators


def _uses_synthetic_toggle_indicator(scene, key: str, toggle) -> bool:
    if toggle is None or toggle.xy is None or toggle.z is None:
        return False
    entity = scene.entities[key]
    metadata = {**(entity.meta or {}), **(toggle.meta or {})}
    category = str(metadata.get("asset_category") or "").lower().replace("-", "_")
    graspable_phone = any(
        token in category
        for token in (
            "cell_phone",
            "desk_phone",
            "mobile_phone",
            "smartphone",
            "telephone",
        )
    )
    if graspable_phone and scene.affordance(key, Role.GRASPABLE) is not None:
        return False
    return not _is_water_fixture(entity, toggle)


def _drive_toggle_indicators(scene, frame) -> None:
    """Apply the frame's running-device snapshot to the visible status lamps."""
    running = getattr(frame, "running_devices", None)
    if running is None:
        return
    active = set(running)
    for key, imageable in (getattr(scene, "toggle_indicators", None) or {}).items():
        if key in active:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def _is_water_fixture_category(entity, toggle) -> bool:
    metadata = {
        **(getattr(entity, "meta", None) or {}),
        **(getattr(toggle, "meta", None) or {}),
    }
    category = str(metadata.get("asset_category") or "").lower().replace("-", "_")
    return any(token in category for token in ("sink", "faucet", "water_tap"))


def _is_water_fixture(entity, toggle) -> bool:
    """Return whether a toggleable asset represents a water outlet."""
    metadata = {
        **(getattr(entity, "meta", None) or {}),
        **(getattr(toggle, "meta", None) or {}),
    }
    explicit = metadata.get("water_stream_visual")
    if explicit is not None:
        return bool(explicit)
    return _is_water_fixture_category(entity, toggle)


def _water_stream_segment(scene, key: str, toggle):
    """Resolve a stable world-space stream from an authored override or fixture bounds."""
    entity = scene.resolve(key)
    metadata = {
        **(getattr(entity, "meta", None) or {}),
        **(getattr(toggle, "meta", None) or {}),
    }
    authored = metadata.get("water_stream_segment")
    if authored is not None:
        values = tuple(float(value) for value in authored)
        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError(
                f"water_stream_segment for {key!r} must be four finite numbers"
            )
        x, y, lower, upper = values
        if upper <= lower:
            raise ValueError(
                f"water_stream_segment for {key!r} must have upper > lower"
            )
        return x, y, lower, upper

    minimum, maximum = scene.object_aabb(key)
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    height = float(maximum[2] - minimum[2])
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or height < 0.10:
        return None
    stream_xy = metadata.get("water_stream_xy")
    if stream_xy is None:
        x = 0.5 * float(minimum[0] + maximum[0])
        y = 0.5 * float(minimum[1] + maximum[1])
    else:
        x, y = (float(value) for value in stream_xy)
    lower = float(metadata.get("water_basin_z", minimum[2] + 0.55 * height))
    upper = float(metadata.get("water_outlet_z", minimum[2] + 0.84 * height))
    if not np.isfinite((x, y, lower, upper)).all() or upper <= lower:
        return None
    return x, y, lower, upper


def _liquid_segment_pose(segment):
    if segment is None:
        return None
    x, y, lower, upper = (float(value) for value in segment)
    if not np.isfinite((x, y, lower, upper)).all() or upper - lower < 0.002:
        return None
    return (x, y, 0.5 * (lower + upper)), upper - lower


def _make_liquid_stream(stage, path: str, *, radius: float):
    """Create one translucent, kinematic visual stream and return its frame setter."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(float(radius))
    height = cylinder.CreateHeightAttr(0.10)
    translate = UsdGeom.Xformable(cylinder).AddTranslateOp()
    material = UsdShade.Material.Define(stage, path + "/Mat")
    shader = UsdShade.Shader.Define(stage, path + "/Mat/S")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.30, 0.72, 0.95)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.12)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.48)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim()).CreateCollisionEnabledAttr(False)
    imageable = UsdGeom.Imageable(cylinder.GetPrim())

    def _set(segment) -> None:
        pose = _liquid_segment_pose(segment)
        if pose is None:
            imageable.MakeInvisible()
            return
        center, stream_height = pose
        translate.Set(Gf.Vec3d(*center))
        height.Set(float(stream_height))
        imageable.MakeVisible()

    _set(None)
    return _set


def _place_liquid_visuals(stage, scene) -> None:
    """Install generic running-fixture streams plus a shared transient pour stream."""
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, "/World/LiquidEffects")
    fixtures = {}
    for key in sorted(scene.entities):
        toggle = scene.affordance(key, Role.TOGGLEABLE)
        entity = scene.entities[key]
        if toggle is None or not _is_water_fixture(entity, toggle):
            continue
        segment = _water_stream_segment(scene, key, toggle)
        if segment is None:
            continue
        safe = "".join(char if char.isalnum() or char == "_" else "_" for char in key)
        setter = _make_liquid_stream(
            stage,
            f"/World/LiquidEffects/Fixture_{safe}",
            radius=0.006,
        )
        setter(segment if key in scene.running else None)
        fixtures[key] = (segment, setter)
    scene.fixture_water_streams = fixtures
    scene.pour_stream_setter = _make_liquid_stream(
        stage,
        "/World/LiquidEffects/Pour",
        radius=0.007,
    )


def _drive_liquid_visuals(scene, frame, pour_segment=None) -> None:
    running = getattr(frame, "running_devices", None)
    if running is not None:
        active = set(running)
        for key, (segment, setter) in (
            getattr(scene, "fixture_water_streams", None) or {}
        ).items():
            setter(segment if key in active else None)
    setter = getattr(scene, "pour_stream_setter", None)
    if setter is not None:
        setter(pour_segment)


def _pour_stream_schedule(frames) -> tuple:
    """Infer visible liquid only around the low point of each Pour trajectory."""
    schedule = [None] * len(frames)
    run: list[int] = []

    def _flush() -> None:
        if not run:
            return
        anchored = [
            (index, np.asarray(getattr(frames[index], "contact_anchor", None)))
            for index in run
            if getattr(frames[index], "contact_anchor", None) is not None
        ]
        if not anchored:
            return
        minimum_z = min(float(anchor[2]) for _, anchor in anchored)
        lower = minimum_z - 0.04
        for index, anchor in anchored:
            if float(anchor[2]) <= minimum_z + 0.075:
                schedule[index] = (
                    float(anchor[0]),
                    float(anchor[1]),
                    lower,
                    max(float(anchor[2]), lower + 0.025),
                )

    for index, frame in enumerate(frames):
        if getattr(frame, "seg", "") == "pour":
            run.append(index)
            continue
        _flush()
        run = []
    _flush()
    return tuple(schedule)


def _is_cut_scoreable(entity) -> bool:
    metadata = getattr(entity, "meta", None) or {}
    explicit = metadata.get("cut_score_visual")
    if explicit is not None:
        return bool(explicit)
    category = str(metadata.get("asset_category") or "").lower().replace("-", "_")
    return any(
        token in category for token in ("dough", "pastry", "bread", "loaf", "cake")
    )


def _cut_score_schedule(frames, goals, compiled_commands, command_spans) -> tuple:
    """Return the cumulative cut targets visible after each physical down-stroke."""
    events: dict[int, set[str]] = {}
    frame_count = len(frames)
    for command, span in zip(compiled_commands, command_spans):
        targets = {
            str(goal.obj_key)
            for goal in goals[command.goal_start : command.goal_end]
            if goal.prim is command_grammar.Primitive.CUT and goal.obj_key
        }
        if not targets:
            continue
        start = max(0, min(frame_count, int(span[0])))
        end = max(start, min(frame_count, int(span[1])))
        cut_indices = [
            index
            for index in range(start, end)
            if getattr(frames[index], "seg", "") == "cut"
        ]
        if not cut_indices:
            continue
        anchored = [
            (index, float(frames[index].contact_anchor[2]))
            for index in cut_indices
            if getattr(frames[index], "contact_anchor", None) is not None
        ]
        if anchored:
            minimum_z = min(value for _, value in anchored)
            transition = next(
                index for index, value in anchored if value <= minimum_z + 0.012
            )
        else:
            transition = cut_indices[-1]
        events.setdefault(transition, set()).update(targets)

    active: set[str] = set()
    schedule = []
    for index in range(frame_count):
        active.update(events.get(index, ()))
        schedule.append(tuple(sorted(active)))
    return tuple(schedule)


def _place_cut_score_marks(stage, scene) -> dict:
    """Create non-colliding score marks that remain attached to cuttable objects."""
    import fpv_scene
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    UsdGeom.Xform.Define(stage, "/World/CutScoreMarks")
    setters = {}
    for key in sorted(scene.entities):
        entity = scene.entities[key]
        if not entity.prim_path or not _is_cut_scoreable(entity):
            continue
        target = stage.GetPrimAtPath(entity.prim_path)
        if target is None or not target.IsValid():
            continue
        minimum, maximum = scene.object_aabb(key)
        minimum = np.asarray(minimum, dtype=np.float64)
        maximum = np.asarray(maximum, dtype=np.float64)
        extent = maximum - minimum
        if not np.isfinite(extent).all() or max(float(extent[0]), float(extent[1])) <= 0.0:
            continue
        length, width, height, surface_offset = _cut_score_geometry(extent)
        world_point = Gf.Vec3d(
            0.5 * float(minimum[0] + maximum[0]),
            0.5 * float(minimum[1] + maximum[1]),
            float(maximum[2]) + surface_offset,
        )
        target_world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            target
        )
        local_point = target_world.GetInverse().Transform(world_point)
        safe = "".join(char if char.isalnum() or char == "_" else "_" for char in key)
        mark_root = f"/World/CutScoreMarks/{safe}"
        mark = UsdGeom.Xform.Define(stage, mark_root).GetPrim()
        stroke_a = _make_box(
            stage,
            mark_root + "/StrokeA",
            (length, width, height),
            (0.0, 0.0, 0.0),
            color=(0.055, 0.018, 0.006),
        )
        stroke_b = _make_box(
            stage,
            mark_root + "/StrokeB",
            (width, length, height),
            (0.0, 0.0, 0.0),
            color=(0.055, 0.018, 0.006),
        )
        for stroke in (stroke_a, stroke_b):
            UsdPhysics.CollisionAPI.Apply(stroke).CreateCollisionEnabledAttr(False)
        imageable = UsdGeom.Imageable(mark)
        imageable.MakeInvisible()
        score_rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), 45.0).GetQuat()

        def _set(
            visible: bool,
            *,
            _target=target,
            _mark=mark,
            _point=local_point,
            _imageable=imageable,
            _score_rotation=score_rotation,
        ) -> None:
            if not visible:
                _imageable.MakeInvisible()
                return
            transform = UsdGeom.XformCache(
                Usd.TimeCode.Default()
            ).GetLocalToWorldTransform(_target)
            position = transform.Transform(_point)
            rotation = transform.ExtractRotationQuat() * _score_rotation
            imaginary = rotation.GetImaginary()
            fpv_scene.set_world_pose(
                _mark,
                tuple(float(value) for value in position),
                (
                    float(rotation.GetReal()),
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                ),
                scale=1.0,
            )
            _imageable.MakeVisible()

        setters[key] = _set
    scene.cut_score_mark_setters = setters
    return setters


def _cut_score_geometry(extent) -> tuple[float, float, float, float]:
    span = max(float(extent[0]), float(extent[1]))
    return max(0.025, min(0.12, 0.45 * span)), 0.004, 0.0008, 0.0005


def _drive_cut_score_marks(scene, active_targets) -> None:
    if active_targets is None:
        return
    active = set(active_targets)
    for key, setter in (getattr(scene, "cut_score_mark_setters", None) or {}).items():
        setter(key in active)


def _make_workstation_panel(
    stage,
    path: str,
    center,
    right,
    width: float,
    height: float,
    color,
    *,
    emissive: bool = False,
):
    """A non-colliding vertical quad in the screen plane."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    c = np.asarray(center, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    r /= max(float(np.linalg.norm(r)), 1e-6)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    points = [
        c - r * width * 0.5 - up * height * 0.5,
        c + r * width * 0.5 - up * height * 0.5,
        c + r * width * 0.5 + up * height * 0.5,
        c - r * width * 0.5 + up * height * 0.5,
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*(float(v) for v in point)) for point in points])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    material = UsdShade.Material.Define(stage, path + "/Mat")
    shader = UsdShade.Shader.Define(stage, path + "/Mat/S")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    if emissive:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
    return mesh.GetPrim()


def _workstation_screen_axes(scene, workstation, monitor, center):
    seat_key = str(workstation.get("seat") or "")
    if seat_key:
        seat_xy = np.asarray(scene.surface_top(seat_key)[0], dtype=np.float64)
        toward_seat = seat_xy - np.asarray(center[:2], dtype=np.float64)
        distance = float(np.linalg.norm(toward_seat))
        if distance > 1e-6:
            normal = np.array(
                [toward_seat[0] / distance, toward_seat[1] / distance, 0.0],
                dtype=np.float64,
            )
            right = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
            monitor.meta["screen_normal"] = tuple(float(value) for value in normal)
            monitor.meta["screen_right"] = tuple(float(value) for value in right)
            return normal, right
    normal = np.asarray(monitor.meta["screen_normal"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-6)
    right = np.asarray(monitor.meta["screen_right"], dtype=np.float64)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    return normal, right


def _workstation_screen_z(monitor, physical_top: float, screen_height: float) -> float:
    screen_center = monitor.meta.get("screen_center")
    if screen_center is not None and len(screen_center) >= 3:
        authored_z = float(screen_center[2])
        if math.isfinite(authored_z):
            return authored_z
    return float(physical_top) - 0.58 * float(screen_height)


def _workstation_screen_layout(style: str):
    """Return normalized panels for the authored workstation application."""
    if style != "shift_queue":
        return (
            ("Desktop", 0.0, 0.0, 0.91, 0.84, (0.08, 0.28, 0.62), True),
            ("Row0", 0.0, 0.25, 0.72, 0.055, (0.78, 0.90, 1.0), True),
            ("Row1", 0.0, 0.03, 0.58, 0.055, (0.78, 0.90, 1.0), True),
            ("Row2", 0.0, -0.18, 0.64, 0.055, (0.78, 0.90, 1.0), True),
        )
    return (
        ("Canvas", 0.0, 0.0, 0.91, 0.84, (0.82, 0.84, 0.86), True),
        ("Header", 0.0, 0.355, 0.91, 0.09, (0.035, 0.075, 0.12), True),
        ("Sidebar", -0.37, -0.035, 0.15, 0.68, (0.16, 0.19, 0.22), True),
        ("QueueTitle", -0.17, 0.265, 0.22, 0.035, (0.09, 0.12, 0.15), True),
        ("QueueRow0", -0.17, 0.145, 0.23, 0.11, (0.96, 0.97, 0.97), True),
        ("QueueRow1", -0.17, 0.005, 0.23, 0.11, (0.70, 0.83, 0.92), True),
        ("QueueRow2", -0.17, -0.135, 0.23, 0.11, (0.96, 0.97, 0.97), True),
        ("QueueRow3", -0.17, -0.275, 0.23, 0.11, (0.93, 0.94, 0.95), True),
        ("Preview", 0.18, -0.015, 0.34, 0.57, (0.96, 0.97, 0.97), True),
        ("PreviewTitle", 0.18, 0.215, 0.23, 0.035, (0.08, 0.11, 0.14), True),
        ("PreviewLine0", 0.15, 0.125, 0.22, 0.022, (0.34, 0.38, 0.42), True),
        ("PreviewLine1", 0.18, 0.065, 0.27, 0.018, (0.48, 0.51, 0.54), True),
        ("PreviewLine2", 0.16, 0.01, 0.23, 0.018, (0.48, 0.51, 0.54), True),
        ("StatusOpen", 0.12, -0.18, 0.10, 0.045, (0.16, 0.48, 0.35), True),
        ("Assignee", 0.25, -0.18, 0.12, 0.045, (0.79, 0.61, 0.20), True),
    )


def _workstation_mouse_pose(scene, keyboard_key: str, workstation):
    stored = np.asarray(workstation["mouse_center"], dtype=np.float64)
    seat_key = str(workstation.get("seat") or "")
    if not seat_key:
        return stored, float(workstation.get("yaw", 0.0))
    minimum, maximum = (
        np.asarray(value, dtype=np.float64) for value in scene.object_aabb(keyboard_key)
    )
    center = 0.5 * (minimum + maximum)
    seat_xy = np.asarray(scene.surface_top(seat_key)[0], dtype=np.float64)
    forward = center[:2] - seat_xy
    distance = float(np.linalg.norm(forward))
    if distance <= 1e-6:
        return stored, float(workstation.get("yaw", 0.0))
    forward /= distance
    right = np.array([forward[1], -forward[0]], dtype=np.float64)
    half_xy = 0.5 * (maximum[:2] - minimum[:2])
    lateral_half = abs(float(right[0])) * half_xy[0] + abs(float(right[1])) * half_xy[1]
    center[:2] += right * (lateral_half + 0.065)
    center[2] = float(stored[2])
    yaw = math.degrees(math.atan2(float(forward[0]), -float(forward[1])))
    return center, yaw


def _align_workstation_keyboard(stage, scene, keyboard_key, workstation, screen_center):
    import fpv_scene
    import object_physics as ophys
    from pxr import Usd, UsdGeom

    seat_key = str(workstation.get("seat") or "")
    keyboard = scene.resolve(keyboard_key)
    prim = stage.GetPrimAtPath(str(keyboard.prim_path or ""))
    if not seat_key or not prim or not prim.IsValid():
        return
    minimum, maximum = (
        np.asarray(value, dtype=np.float64)
        for value in ophys.physical_world_bounds(fpv_scene, prim)
    )
    current_center = 0.5 * (minimum + maximum)
    seat_xy = np.asarray(scene.surface_top(seat_key)[0], dtype=np.float64)
    toward_screen = np.asarray(screen_center[:2], dtype=np.float64) - seat_xy
    screen_distance = float(np.linalg.norm(toward_screen))
    if screen_distance <= 0.30:
        return
    forward = toward_screen / screen_distance
    keyboard_distance = min(0.42, max(0.30, screen_distance - 0.18))
    desired_xy = seat_xy + forward * keyboard_distance
    delta_xy = desired_xy - current_center[:2]
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        prim
    )
    origin = transform.ExtractTranslation()
    fpv_scene.set_world_pose(
        prim,
        (
            float(origin[0]) + float(delta_xy[0]),
            float(origin[1]) + float(delta_xy[1]),
            float(origin[2]),
        ),
        None,
        None,
    )
    keyboard.xy = (float(desired_xy[0]), float(desired_xy[1]))
    keyboard.foot_xy = keyboard.xy
    getattr(scene, "_static_aabb", {}).pop(keyboard_key, None)
    scene.set_object_rest_pose(
        keyboard_key, ophys.read_prim_transform(stage, str(keyboard.prim_path))
    )


def _place_workstation_details(stage, scene) -> dict:
    """Add a visible live screen and correctly placed mouse to registered workstations."""
    import fpv_scene
    import object_physics as ophys
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    workstations = getattr(scene, "workstations", None) or {}
    if not workstations:
        return {}
    UsdGeom.Xform.Define(stage, "/World/WorkstationDetails")
    screens = {}
    for keyboard_key, workstation in sorted(workstations.items()):
        monitor_key = str(workstation["monitor"])
        monitor = scene.resolve(monitor_key)
        procedural_monitor = bool(workstation.get("procedural_monitor"))
        if procedural_monitor:
            center = np.asarray(workstation["screen_center"], dtype=np.float64)
            normal, right = _workstation_screen_axes(
                scene, workstation, monitor, center
            )
            screen_height = 0.23
            screen_width = 0.38
            screen_center = center.copy()
        else:
            monitor_prim = stage.GetPrimAtPath(str(monitor.prim_path or ""))
            mn, mx = ophys.physical_world_bounds(fpv_scene, monitor_prim)
            minimum = np.asarray(mn, dtype=np.float64)
            maximum = np.asarray(mx, dtype=np.float64)
            center = 0.5 * (minimum + maximum)
            UsdGeom.Imageable(monitor_prim).MakeInvisible()
            normal, right = _workstation_screen_axes(
                scene, workstation, monitor, center
            )
            half_xy = 0.5 * (maximum[:2] - minimum[:2])
            face_depth = (
                abs(float(normal[0])) * half_xy[0]
                + abs(float(normal[1])) * half_xy[1]
            )
            screen_height = min(
                0.23, max(0.15, 0.55 * float(maximum[2] - minimum[2]))
            )
            screen_width = min(
                0.42,
                max(
                    0.27,
                    1.65
                    * (
                        abs(float(right[0])) * half_xy[0]
                        + abs(float(right[1])) * half_xy[1]
                    ),
                ),
            )
            screen_center = center.copy()
            screen_center[:2] += normal[:2] * (face_depth + 0.004)
            screen_center[2] = _workstation_screen_z(
                monitor, float(maximum[2]), screen_height
            )
        monitor.meta["screen_center"] = tuple(float(value) for value in screen_center)
        _align_workstation_keyboard(
            stage, scene, keyboard_key, workstation, screen_center
        )
        safe = "".join(
            char if char.isalnum() or char == "_" else "_" for char in monitor_key
        )
        root = f"/World/WorkstationDetails/{safe}"
        _make_workstation_panel(
            stage,
            root + "/Bezel",
            screen_center,
            right,
            screen_width,
            screen_height,
            (0.012, 0.015, 0.020),
        )
        desk_z = float(workstation["mouse_center"][2]) - 0.022
        screen_bottom = float(screen_center[2]) - 0.5 * screen_height
        stem_height = max(0.04, screen_bottom - desk_z)
        stem_center = screen_center.copy()
        stem_center[2] = desk_z + 0.5 * stem_height
        _make_workstation_panel(
            stage,
            root + "/Stand",
            stem_center,
            right,
            0.035,
            stem_height,
            (0.025, 0.028, 0.032),
        )
        on = UsdGeom.Xform.Define(stage, root + "/On")
        lit_center = screen_center + normal * 0.002
        style = str(
            workstation.get("screen_style")
            or monitor.meta.get("workstation_screen_style")
            or ""
        )
        for index, (
            name,
            horizontal,
            vertical,
            width,
            height,
            color,
            emissive,
        ) in enumerate(_workstation_screen_layout(style)):
            _make_workstation_panel(
                stage,
                f"{root}/On/{name}",
                lit_center
                + right * (horizontal * screen_width)
                + np.array([0.0, 0.0, vertical * screen_height])
                + normal * (0.001 * index),
                right,
                screen_width * width,
                screen_height * height,
                color,
                emissive=emissive,
            )
        imageable = UsdGeom.Imageable(on.GetPrim())
        (
            imageable.MakeVisible()
            if monitor_key in scene.running
            else imageable.MakeInvisible()
        )
        screens[monitor_key] = imageable

        mouse_center, mouse_yaw = _workstation_mouse_pose(
            scene, keyboard_key, workstation
        )
        if mouse_center is not None:
            mouse_path = root + "/Mouse"
            mouse = UsdGeom.Sphere.Define(stage, mouse_path)
            mouse.CreateRadiusAttr(1.0)
            xform = UsdGeom.Xformable(mouse)
            xform.AddTranslateOp().Set(
                Gf.Vec3d(*(float(value) for value in mouse_center))
            )
            yaw = math.radians(mouse_yaw)
            xform.AddOrientOp().Set(
                Gf.Quatf(math.cos(yaw * 0.5), Gf.Vec3f(0.0, 0.0, math.sin(yaw * 0.5)))
            )
            xform.AddScaleOp().Set(Gf.Vec3f(0.032, 0.052, 0.018))
            material = UsdShade.Material.Define(stage, mouse_path + "/Mat")
            shader = UsdShade.Shader.Define(stage, mouse_path + "/Mat/S")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.055, 0.060, 0.070)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI.Apply(mouse.GetPrim()).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            scene.resolve(keyboard_key).meta["mouse_path"] = mouse_path
    scene.workstation_screens = screens
    return screens


def _drive_workstation_screens(scene, frame) -> None:
    running = getattr(frame, "running_devices", None)
    if running is None:
        return
    active = set(running)
    for key, imageable in (getattr(scene, "workstation_screens", None) or {}).items():
        imageable.MakeVisible() if key in active else imageable.MakeInvisible()


def _recolor_prim(stage, prim_path: str, color) -> None:
    """Bind a fresh matte UsdPreviewSurface (``color``) to every Gprim/Mesh under ``prim_path``, using
    ``strongerThanDescendants`` so it OVERRIDES a referenced asset's own material. Same proven binding
    ``_make_box`` uses (RTX-visible). Used to make the carried mug visually DISTINCT from the clutter.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return
    mtl = UsdShade.Material.Define(stage, prim_path + "/RecolorMat")
    sh = UsdShade.Shader.Define(stage, prim_path + "/RecolorMat/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    for p in Usd.PrimRange(root):
        # NEVER recolor a cosmetic label decal: Label_/RLabel_ meshes carry their own textured (opaque+unlit)
        # material, so a BODY recolor (e.g. the amber reagent bottle) must skip the sign/label subtree or it
        # clobbers it. Byte-identical for objects without such decals (no path segment matches).
        if any(
            seg.startswith(("Label_", "RLabel_")) for seg in str(p.GetPath()).split("/")
        ):
            continue
        if p.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )


def _clean_room(stage, app):
    """Reference Simple_Room, deactivate its built-in table_low stool prop, and shift the whole room so
    its VISIBLE floor top sits at z=0 (the asset ships the floor at ~z=-0.82, which floated our scene).
    """
    import fpv_scene
    from pxr import Usd

    room = fpv_scene.build_room(stage)
    for p in Usd.PrimRange(room):
        if "table_low" in p.GetName():
            p.SetActive(False)
    app.update()
    floor = stage.GetPrimAtPath(str(room.GetPath()) + "/Floor")
    top = float(fpv_scene.compute_world_bbox(floor).GetMax()[2])
    fpv_scene.set_world_pose(room, (0.0, 0.0, -top))
    app.update()
    print(f"[step3r] room floor shifted by {-top:.3f} -> floor top at z=0")
    return room


def _place_cabinet(stage, name: str, xy, base_h: float, drawer: str):
    """Reference + seat a Sektion cabinet (front +X -> +Y, toward the human), de-instance, add slide
    ops to the chosen drawer body + its handle. Returns a controller dict for per-frame driving.
    """
    import fpv_scene

    path = f"/World/{name}"
    prim = fpv_scene.add_reference(stage, path, fpv_scene.ASSET_ROOT + _CABINET)
    yaw = (
        0.70710678,
        0.0,
        0.0,
        0.70710678,
    )  # Rz(+90): local +X (front/handles) -> world +Y
    # seat the cabinet bottom on top of its base (z=base_h): pose at 0 to read local min-z, then lift.
    fpv_scene.set_world_pose(prim, (xy[0], xy[1], 0.0), yaw)
    min_z = fpv_scene.local_bbox_min_z(prim)
    fpv_scene.set_world_pose(prim, (xy[0], xy[1], base_h - min_z), yaw)
    if base_h > 1e-3:
        # A cabinet-matching base (footprint ~ the Rz90 cabinet extent, light gray) so cabinet+base
        # reads as ONE tall cabinet unit rather than a cabinet on a pedestal.
        _make_box(
            stage,
            path + "_Base",
            (0.74, 0.65, base_h),
            (xy[0], xy[1], base_h * 0.5),
            color=(0.80, 0.79, 0.76),
        )
    _deinstance(prim)

    handle = _find_by_name(prim, "drawer_handle_" + drawer.split("_")[1])
    joint = _find_drawer_joint(prim, drawer)
    if joint is None:
        raise RuntimeError(
            f"_place_cabinet: no PrismaticJoint for {drawer!r} under {path}"
        )
    moving_prim = None
    body1 = joint.GetRelationship("physics:body1")
    targets = body1.GetTargets() if body1 else ()
    if targets:
        candidate = stage.GetPrimAtPath(targets[0])
        if candidate is not None and candidate.IsValid():
            moving_prim = candidate
    if moving_prim is None:
        moving_prim = _find_by_name(prim, drawer)
    if moving_prim is None:
        raise RuntimeError(
            f"_place_cabinet: no moving drawer prim for {drawer!r} under {path}"
        )
    moving_range = fpv_scene.compute_world_bbox(moving_prim)
    moving_minimum, moving_maximum = moving_range.GetMin(), moving_range.GetMax()
    return {
        "name": name,
        "prim": prim,
        "handle": handle,
        "joint_set": _make_joint_setter(
            joint
        ),  # drive the PhysX joint target (not a kinematic xform)
        "world_axis": np.array([0.0, 1.0, 0.0]),  # R_cab @ (1,0,0) for Rz(+90)
        "travel": _OPEN_TRAVEL,
        "moving_paths": (str(moving_prim.GetPath()),),
        "moving_aabb": (
            np.array(
                [moving_minimum[0], moving_minimum[1], moving_minimum[2]],
                dtype=np.float64,
            ),
            np.array(
                [moving_maximum[0], moving_maximum[1], moving_maximum[2]],
                dtype=np.float64,
            ),
        ),
        "yaw_wxyz": yaw,
        "foot_xy": (
            float(xy[0]),
            float(xy[1]),
        ),  # cabinet+base footprint center (for stand-back)
        "foot_half": (
            0.37,
            0.325,
        ),  # base box (0.74,0.65) half-extent = the Rz90 cabinet footprint
    }


# =================================================================================================
# Scene builders -> (SceneState, controllers, start_xy, program string, tpv_look).
# =================================================================================================
def build_openclose(stage, app):
    scene = SceneState(floor_z=0.0, start_xy=_START_OC)
    _clean_room(stage, app)

    cab = _place_cabinet(stage, "Cabinet", _CAB_XY, _CAB_BASE_H, "drawer_top")
    drw = _place_cabinet(stage, "Drawer", _DRW_XY, _DRW_BASE_H, "drawer_top")
    for _ in range(3):
        app.update()  # commit placement so the live handle bbox is current

    import fpv_scene

    for ctrl, ekey in ((cab, "cabinet"), (drw, "drawer")):
        h = _world_center(ctrl["handle"])
        ctrl["closed_handle"] = h
        top_z = float(fpv_scene.compute_world_bbox(ctrl["prim"]).GetMax()[2])
        scene.register(
            ekey,
            Role.OPENABLE,
            xy=(float(h[0]), float(h[1])),
            z=float(h[2]),
            approach_yaw=0.0,  # face -Y toward the cabinet front
            axis=tuple(ctrl["world_axis"]),
            travel=ctrl["travel"],
            foot_xy=ctrl[
                "foot_xy"
            ],  # geometry-derived stand-back: clear the cabinet footprint + travel
            foot_half=ctrl["foot_half"],
            top_z=top_z,
            clearance=0.36,  # body clearance in front of the near-face (travel added on top)
            stand_gap=0.56,  # fallback (unused while foot_half is set)
            joint_name=ekey,
        )
        print(
            f"[step3r] {ekey}: handle world={h.round(3).tolist()} top_z={top_z:.2f} axis={ctrl['world_axis'].tolist()}"
        )

    program = (
        "[Walk] <cabinet> (1), [Open] <cabinet> (1), "
        "[Walk] <drawer> (1), [Open] <drawer> (1), [Close] <drawer> (1), "
        "[Walk] <cabinet> (1), [Close] <cabinet> (1)"
    )
    controllers = {"cabinet": cab, "drawer": drw}
    tpv_look = (0.0, -0.55, 0.8)  # aim the side cam at the furniture band
    return (
        scene,
        controllers,
        program,
        tpv_look,
        None,
    )  # tpv_eye None -> side-follow cam


def build_sit(stage, app):
    import fpv_scene

    scene = SceneState(floor_z=0.0, start_xy=_START_SIT)
    _clean_room(stage, app)

    # Seats ship with front on +X / backrest on -X. Rz(+90) turns the FRONT to +Y so the human (faces
    # +Y at approach_yaw=180) sits back-to-backrest with legs off the front edge (probe: seat_orient_probe).
    for name, rel, xy in (("Chair", _CHAIR, _CHAIR_XY), ("Sofa", _SOFA, _SOFA_XY)):
        p = fpv_scene.add_reference(stage, f"/World/{name}", fpv_scene.ASSET_ROOT + rel)
        yaw = (
            0.70710678,
            0.0,
            0.0,
            0.70710678,
        )  # Rz(+90): front +X -> +Y (toward the human)
        fpv_scene.set_world_pose(p, (xy[0], xy[1], 0.0), yaw)
        mnz = fpv_scene.local_bbox_min_z(p)
        fpv_scene.set_world_pose(p, (xy[0], xy[1], -mnz), yaw)
    for _ in range(3):
        app.update()

    # approach_yaw=0: walk up FACING the seat + stop clear of its front edge; produce_sit turns 180
    # and backs the pelvis onto the cushion (no walking through the seat).
    scene.register(
        "chair",
        Role.SEAT,
        xy=_CHAIR_XY,
        z=_CHAIR_SEAT_Z,
        approach_yaw=0.0,
        stand_gap=0.5,
    )
    scene.register(
        "sofa", Role.SEAT, xy=_SOFA_XY, z=_SOFA_SEAT_Z, approach_yaw=0.0, stand_gap=0.62
    )
    print(f"[step3r] chair seat_z={_CHAIR_SEAT_Z} sofa seat_z={_SOFA_SEAT_Z}")

    program = (
        "[Walk] <chair> (1), [Sit] <chair> (1), [StandUp], "
        "[Walk] <sofa> (1), [Sit] <sofa> (1)"
    )
    # Seats face +Y (backrest at -Y); a FIXED front-elevated cam at +Y sees the seated body's front
    # (lap + bent legs) unobstructed -- a side cam is blocked by the other seat.
    tpv_look = (0.2, -0.25, 0.55)
    tpv_eye = (0.2, 3.0, 1.5)
    return scene, {}, program, tpv_look, tpv_eye


# =================================================================================================
# Per-frame drawer driver: route each frame's moving handle point to the nearest openable.
# =================================================================================================
def build_highreach(stage, app):
    """Problem-3 probe: a cabinet whose handle is ABOVE the head (~1.6) to OPEN, and a graspable object
    on a tall shelf (~1.35) to GRAB -- shows how the shared warp_arm_to_seat reach behaves overhead.
    """
    scene = SceneState(floor_z=0.0, start_xy=(0.0, 1.6))
    _clean_room(stage, app)

    import fpv_scene

    # handle ~1.39 (above shoulder = a genuine reach-up) but BELOW eye level, so the opening drawer
    # sits under the FPV eyeline and the head-cam frames it instead of clipping into it (was 0.88 ->
    # 1.59 m, face-level, which put the sliding drawer in the camera and blacked out the FPV).
    cab = _place_cabinet(stage, "HiCab", (-0.55, -0.6), 0.68, "drawer_top")
    for _ in range(3):
        app.update()
    h = _world_center(cab["handle"])
    cab["closed_handle"] = h
    cab_top = float(fpv_scene.compute_world_bbox(cab["prim"]).GetMax()[2])
    scene.register(
        "cabinet",
        Role.OPENABLE,
        xy=(float(h[0]), float(h[1])),
        z=float(h[2]),
        approach_yaw=0.0,
        axis=tuple(cab["world_axis"]),
        travel=cab["travel"],
        foot_xy=cab["foot_xy"],
        foot_half=cab["foot_half"],
        top_z=cab_top,
        clearance=0.36,
        stand_gap=0.56,  # fallback (unused while foot_half is set)
        joint_name="cabinet",
    )
    print(
        f"[step3r] HIGH cabinet handle world={h.round(3).tolist()} top_z={cab_top:.2f} (reach UP to open)"
    )

    # A high OPEN shelf (thin top surface on back legs, NOT a solid pillar) with a graspable object at
    # its FRONT edge. A full-height solid pillar forces the chest/arm INTO it at chest height (measured:
    # 800+ body verts inside), which no stand-back can fix -- an open shelf lets the human stand close
    # and reach UP to the front edge through free space below the shelf.
    shelf_c, shelf_top = (0.6, -0.6), 1.3
    _make_box(
        stage,
        "/World/Shelf",  # thin top plank (z spans ~[1.275, 1.325])
        (0.5, 0.35, 0.05),
        (shelf_c[0], shelf_c[1], shelf_top),
        color=(0.55, 0.5, 0.45),
    )
    for lx in (-0.2, 0.2):  # thin back legs (at the far edge, out of the reach path)
        _make_box(
            stage,
            f"/World/ShelfLeg{'L' if lx < 0 else 'R'}",
            (0.05, 0.05, shelf_top),
            (shelf_c[0] + lx, shelf_c[1] - 0.14, shelf_top * 0.5),
            color=(0.5, 0.45, 0.4),
        )
    obj_xy = (shelf_c[0], shelf_c[1] + 0.14)  # front edge of the shelf top
    # z is now a pre-drop HINT only: run_render's env-init settle (object_physics) rests the cup on the
    # shelf top (top_z below = the real support surface) and overwrites this via scene.set_object_rest,
    # so the ~3.5cm float here becomes the PHYSICS-DERIVED rest z.
    scene.register(
        "obj",
        Role.GRASPABLE,
        xy=obj_xy,
        z=shelf_top + 0.06,
        approach_yaw=0.0,
        foot_xy=shelf_c,  # stand clear of the shelf footprint
        foot_half=(0.25, 0.175),  # thin-top half-extent
        top_z=shelf_top + 0.025,
        clearance=0.42,
    )
    print(
        f"[step3r] HIGH grasp object at {obj_xy} z={shelf_top + 0.06:.2f} (open shelf front edge)"
    )

    program = (
        "[Walk] <cabinet> (1), [Open] <cabinet> (1), [Walk] <obj> (1), [Grab] <obj> (1)"
    )
    tpv_look = (0.0, -0.6, 1.2)
    return scene, {"cabinet": cab}, program, tpv_look, None


def build_walkby(stage, app):
    """Whole-timeline nudge probe: the human WALKS PAST a solid low counter that its straight path
    clips (partial overlap ~10-15 cm). WITHOUT the render-time nudge the body/hip passes through the
    box edge; WITH it the body sidesteps out. A NON-manipulation case no other fix (reach waypoints /
    stand-back) covers. Registers the counter as a footprint Entity so scene.solid_boxes() sees it, and
    a far goal to navigate to; the program is a single Walk to the goal (no manipulation).
    """
    scene = SceneState(floor_z=0.0, start_xy=(0.0, 1.6))
    _clean_room(stage, app)

    box_xy = (0.35, 0.0)
    box_size = (0.5, 0.5, 0.95)
    _make_box(
        stage,
        "/World/Counter",
        box_size,
        (box_xy[0], box_xy[1], box_size[2] * 0.5),
        color=(0.60, 0.50, 0.42),
    )
    for _ in range(3):
        app.update()
    # The counter as a footprint obstacle: the STRAIGHT path X=0 (start +Y -> goal -Y) clips its near
    # (X=0.10) edge, so the body proxy penetrates the body-inflated box -> the render-time nudge fires.
    scene.register(
        "counter",
        Role.SURFACE,
        xy=box_xy,
        z=0.0,
        foot_xy=box_xy,
        foot_half=(0.25, 0.25),
        top_z=box_size[2],
    )
    scene.register(
        "goal",
        Role.SURFACE,
        xy=(0.0, -1.4),
        z=0.75,
        foot_half=(0.3, 0.3),
        top_z=0.75,
        approach_yaw=0.0,
    )
    print(
        f"[step3r] walkby: counter at {box_xy} size {box_size}; walk (0.0,1.6)->(0.0,-1.4) clips it"
    )

    program = "[Walk] <goal> (1)"
    tpv_look = (
        0.0,
        0.1,
        0.9,
    )  # aim the side-follow cam at the walk band past the counter
    return scene, {}, program, tpv_look, None


def _loco_room(stage, app):
    """A synthetic room (floor slab + four walls + a sun and two lamps) built from primitives only.

    Deliberately does NOT reference Simple_Room: the locomotion demo has to be runnable on a machine that
    has the BEHAVIOR data but not the Isaac stock asset tree, which is exactly the state this one is in.
    """
    import bench_lighting

    _make_box(
        stage,
        "/World/LocoFloor",
        (9.0, 9.0, 0.10),
        (0.0, 0.0, -0.05),
        color=(0.34, 0.31, 0.28),
    )
    for name, size, ctr, col in (
        ("N", (9.0, 0.2, 2.6), (0.0, 3.0, 1.3), (0.62, 0.58, 0.52)),
        ("S", (9.0, 0.2, 2.6), (0.0, -3.0, 1.3), (0.50, 0.55, 0.60)),
        ("E", (0.2, 6.2, 2.6), (3.0, 0.0, 1.3), (0.58, 0.52, 0.46)),
        ("W", (0.2, 6.2, 2.6), (-3.0, 0.0, 1.3), (0.46, 0.52, 0.46)),
    ):
        _make_box(stage, f"/World/LocoWall{name}", size, ctr, color=col)
    # Four corner pillars in distinct colours: they are what makes a TURN legible in the EGOCENTRIC
    # view, where a featureless room looks the same at every heading.
    for i, (px, py, col) in enumerate(
        (
            (-2.4, 2.4, (0.72, 0.24, 0.20)),
            (2.4, 2.4, (0.20, 0.45, 0.72)),
            (2.4, -2.4, (0.85, 0.70, 0.20)),
            (-2.4, -2.4, (0.25, 0.60, 0.35)),
        )
    ):
        _make_box(
            stage, f"/World/LocoPillar{i}", (0.3, 0.3, 2.2), (px, py, 1.1), color=col
        )
    for i, (lx, ly) in enumerate(((-1.4, 1.2), (1.4, 1.2), (0.0, -1.6))):
        bench_lighting._author_lamp(
            stage, f"/World/LocoLamp{i}", (lx, ly, 2.35), 0.30, 9000.0
        )
    for _ in range(3):
        app.update()
    print(
        "[step3r] locomotion room: 6x6 m primitive floor + 4 walls + 4 colour pillars + 3 lamps"
    )


def build_locomotion(stage, app):
    """EGOCENTRIC LOCOMOTION demo: an empty room, one solid counter, and a program that names NOTHING.

    Every other scene's program refers to entities ("[Grab] <cup>"); this one is built only from
    [Forward] / [TurnLeft] / [TurnRight] / [TurnAround], which is what you can say to a human who is
    being guided without being shown the room. The route is a rectangle so the FPV shows the heading
    changing at each turn while the body stays put, and the LAST command deliberately walks into the
    counter -- it must come back path_blocked with zero frames, not a nudge through the geometry.
    """
    scene = SceneState(floor_z=0.0, start_xy=(0.0, 1.5))
    _loco_room(stage, app)

    box_xy = (0.0, -1.3)
    box_size = (1.6, 0.5, 0.95)
    _make_box(
        stage,
        "/World/Counter",
        box_size,
        (box_xy[0], box_xy[1], box_size[2] * 0.5),
        color=(0.60, 0.50, 0.42),
    )
    for _ in range(3):
        app.update()
    # The ONLY registered entity, and only so scene.solid_boxes() can answer "is the way ahead clear?".
    # The program never names it.
    scene.register(
        "counter",
        Role.SURFACE,
        xy=box_xy,
        z=0.0,
        foot_xy=box_xy,
        foot_half=(0.8, 0.25),
        top_z=box_size[2],
    )
    # yaw 0 faces -Y, and turning LEFT increases yaw: 0 -> 90 -> 0 -> -180 -> -90 -> 0, walking a
    # rectangle and ending back on the heading that points at the counter.
    program = (
        "[Forward] 2, [TurnLeft], [Forward] 2, [TurnRight], [Forward], [TurnAround], "
        "[Forward] 2, [TurnLeft], [Forward] 2, [TurnLeft], [Forward] 6m"
    )
    print(
        f"[step3r] locomotion: counter at {box_xy} half=(0.80,0.25); rectangle from (0.0,1.5); "
        f"the final [Forward] 6m must report path_blocked"
    )
    # Explicit TPV eye: the auto side-follow camera lands OUTSIDE a room this small (6x6 m) and renders
    # the back of a wall. This corner is clear of both the pillars and the route.
    return scene, {}, program, (0.3, -0.2, 0.9), (2.60, 1.85, 1.95)


def build_clutter(stage, app):
    """CDS object-physics probe: SIX diverse stock props on a static table in deliberately-WRONG poses
    (floating / interpenetrating / mis-stacked) that the env-init multi-body settle corrects into a
    physically-correct rest, THEN the human walks up and places a carried mug onto the settled tabletop.
    Proves the settle generalizes to arbitrary USD prim sets (zero per-object tuning) and that runtime
    placement is physics-corrected. The six clutter prims are separate USD refs (NOT the rig sink); the
    carried mug is the single rig /World/Obj. Clutter prims get NO foot_half -> they stay OUT of
    solid_boxes() so the reach gate / nudge ignore them (only the table is an obstacle).
    """
    import fpv_scene

    scene = SceneState(floor_z=0.0, start_xy=(0.0, 1.6))
    _clean_room(stage, app)

    # Static table in front of the human (-Y): a THIN top plank on legs (the proven build_highreach shelf
    # pattern). A thin plank is a safe settle support at the 1/60 settle step (a prop drops only a few cm,
    # ~1.7cm/substep near impact, far less than the 6cm plank -> no tunneling) AND its collision AABB is
    # only the plank's thin z-band, so a human reaching OVER the tabletop grazes just the top (like the
    # shelf), not a full-height solid pedestal its leg/torso would clip. The NEAR edge (toward the human)
    # is fixed at -0.395 so the reach + geometry-derived stand-back are unchanged; the plank extends
    # DEEPER in Y (0.55) than the old 0.45 so the reject-sampler has room to fit all six props toward the
    # BACK/sides, clear of the front-center mug lane (a shallow 0.45 plank left the six + the reserved mug
    # lane over-packed, so a rolling bowl ended up in the mug's drop spot; deepening the table -- with the
    # near edge fixed -- keeps the front-center mug target clear while stand-back stays identical).
    table_top = 0.75
    table_cx = 0.0
    near_edge_y = (
        -0.395
    )  # toward the human (+Y): FIXED front edge of the tabletop (stand-back anchor)
    plank = (
        1.0,
        0.55,
        0.05,
    )  # thin top; center at table_top-half so the TOP surface is exactly 0.75
    top_half_y = plank[1] * 0.5
    table_cy = (
        near_edge_y - top_half_y
    )  # push the plank back so the near edge stays fixed
    _make_box(
        stage,
        "/World/Table",
        plank,
        (table_cx, table_cy, table_top - plank[2] * 0.5),
        color=(0.52, 0.40, 0.30),
    )
    for lx in (
        -0.42,
        0.42,
    ):  # four legs so it reads as a table; at the corners, out of the reach path
        for ly in (
            -0.24,
            0.24,
        ):  # near the deeper plank's Y-edges so it reads as a table
            _make_box(
                stage,
                f"/World/TableLeg{'L' if lx < 0 else 'R'}{'B' if ly < 0 else 'F'}",
                (0.05, 0.05, table_top),
                (table_cx + lx, table_cy + ly, table_top * 0.5),
                color=(0.42, 0.32, 0.24),
            )
    for _ in range(3):
        app.update()

    # P2(A): spawn the six props, read each XY footprint half-extent from its WORLD bbox, then
    # rejection-sample a NON-OVERLAPPING XY layout (object_physics.plan_spawn_layout) over the tabletop
    # MINUS a generous edge margin (so a rolling bowl/banana stays on). This replaces the old hand-picked
    # layout so ANY (seeded) random draw is physically valid at spawn -- what a randomly-generated RL
    # scene (BEHAVIOR-1K next) needs. The carried mug's front-center grab/place spot is passed as a
    # RESERVED footprint so no clutter lands on it (clutter stays toward the BACK/sides, mug front-center).
    # Then each prop is placed at (planned_xy, table_top + a MODEST float) so the env-init settle still
    # drops it (the visible-drop still plays). Collect (prim_path, support_top_z=table_top, xy) for
    # run_render's settle_clutter.
    import object_physics as _ophys

    # The mug grab/place lane (front-center) is reserved so the planner keeps clutter off it. cup_xy /
    # place_xy below are (table_cx, near_edge_y - 0.06/-0.08); reserve a GENEROUS footprint spanning both
    # plus mug-radius (~5 cm) clearance AND roll headroom, so a settling prop cannot roll into the mug's
    # drop spot (the shallow-table failure). Clutter is pushed to the BACK/sides, mug front-center.
    mug_lane = (table_cx, near_edge_y - 0.09, 0.10, 0.13)  # (x, y, hx, hy)

    # (prim_path, (hx, hy) world-bbox XY footprint half-extent, yaw_deg, seated_origin_z), aligned to
    # _CLUTTER. seated_origin_z = the ORIGIN z at which spawn_prop put the bbox-bottom on the tabletop
    # (read back after seating -- robust vs recomputing local_bbox_min_z on the already-lifted prim, which
    # folds the seat translate in and mis-places the drop). Adding the modest float to THIS origin lifts
    # the bbox-bottom exactly _CLUTTER_SPAWN_Z above the tabletop; the seat z is xy-independent.
    spawned: list[tuple[str, tuple[float, float], float, float]] = []
    for name, rel, yaw in _CLUTTER:
        prim = fpv_scene.spawn_prop(
            stage, name, rel, table_top, (table_cx, table_cy), yaw
        )
        app.update()  # commit the (temporary) seat so the world bbox + seated origin read current
        rng = fpv_scene.compute_world_bbox(prim)
        mn, mx = rng.GetMin(), rng.GetMax()
        half_xy = (
            float(mx[0] - mn[0]) * 0.5,
            float(mx[1] - mn[1]) * 0.5,
        )  # XY footprint half-extent
        seat_z = float(_ophys.read_prim_xyz(stage, str(prim.GetPath()))[2])
        spawned.append((str(prim.GetPath()), half_xy, float(yaw), seat_z))

    # Rejection-sample the non-overlapping XY over the tabletop shrunk by the edge margin.
    region = (
        table_cx - plank[0] * 0.5 + _CLUTTER_EDGE_MARGIN,
        table_cx + plank[0] * 0.5 - _CLUTTER_EDGE_MARGIN,
        table_cy - top_half_y + _CLUTTER_EDGE_MARGIN,
        table_cy + top_half_y - _CLUTTER_EDGE_MARGIN,
    )
    planned_xy, plan_stats = _ophys.plan_spawn_layout(
        region,
        [h for _p, h, _y, _z in spawned],
        seed=_CLUTTER_SEED,
        min_gap=_CLUTTER_MIN_GAP,
        max_tries=_CLUTTER_MAX_TRIES,
        reserved=[mug_lane],
    )
    scene.clutter_plan_stats = (
        plan_stats  # for the --check print (additive dynamic attr)
    )

    clutter: list[tuple[str, float, tuple[float, float]]] = []
    for (path, _half, yaw, seat_z), (px, py) in zip(spawned, planned_xy):
        prim = stage.GetPrimAtPath(path)
        # Place at the planned XY with a MODEST float above the tabletop so the env-init settle visibly
        # drops it: origin = seated_origin_z + _CLUTTER_SPAWN_Z (bbox-bottom = tabletop + the float); the
        # orient re-authors the same +Z yaw so the prop drops straight down and stays upright.
        drop_z = float(seat_z) + _CLUTTER_SPAWN_Z
        fpv_scene.set_world_pose(
            prim, (px, py, drop_z), fpv_scene._to_wxyz(fpv_scene._yaw_quat(yaw))
        )
        clutter.append((path, float(table_top), (px, py)))
    # Expose the planned clutter prim list for run_render's settle_clutter. Stored as a dynamic
    # attribute on the SceneState instance (a plain dataclass, so this is additive -- no scene_state
    # edit); the six clutter prims are SEPARATE USD refs, NOT the single-object rig sink.
    scene.clutter_prims = clutter
    for _ in range(2):
        app.update()
    print(
        f"[step3r] clutter planner: placed={plan_stats['placed']} "
        f"rejected={plan_stats['rejected']} attempts={plan_stats['attempts']} "
        f"min_pairwise_gap="
        + (
            "inf"
            if not np.isfinite(plan_stats["min_pairwise_gap"])
            else f"{plan_stats['min_pairwise_gap'] * 1000:.0f}mm"
        )
    )

    # The table as a footprint SURFACE (the thin-top shelf pattern). Its PLACE target xy = the reachable
    # FRONT-center of the tabletop (produce_place drops the mug at the surface xy). foot_xy/foot_half =
    # the tabletop footprint + top_z = the plank top, so stand-back / the thin-top reach gate / the nudge
    # treat it exactly like build_highreach's shelf (a hand-graze over the top, not a solid pedestal).
    # clearance=0.42: stand the body's CORE clearly IN FRONT of the near edge. With the geometry-derived
    # stand-back, stand_y = table_cy + (half_depth + clearance) = -0.62 + (0.225 + 0.42) = +0.025, i.e.
    # ~0.42 m in front of near_edge_y (-0.395). This keeps the pelvis/torso ENTIRELY outside the tabletop
    # Y-footprint even when the spine leans forward to reach over -- the D3 BLOCKER was the body standing
    # UNDER the thin plank (a 0.30 clearance left it only ~0.30 m clear, so a leaning torso bisected the
    # top). The reach is distance-general (~0.5 m forward), so warp_arm_to_seat still lands the place.
    place_xy = (table_cx, near_edge_y - 0.08)
    scene.register(
        "table",
        Role.SURFACE,
        xy=place_xy,
        z=table_top,
        foot_xy=(table_cx, table_cy),
        foot_half=(plank[0] * 0.5, top_half_y),
        top_z=table_top,
        approach_yaw=0.0,
        clearance=0.42,
    )
    # The CARRIED mug -> the single rig /World/Obj. z is a small pre-drop float; top_z = the TABLE top
    # (the real support surface run_render's settle drops it onto -- NOT the object's own top), so
    # settle_object_on_support seats it correctly. NO foot_half (stays out of solid_boxes). Its GRAB xy
    # is a reachable FRONT-center spot, distinct from the pile clusters spread over the tabletop.
    cup_xy = (table_cx, near_edge_y - 0.06)
    scene.register(
        "cup",
        Role.GRASPABLE,
        xy=cup_xy,
        z=table_top + 0.05,
        approach_yaw=0.0,
        foot_xy=(table_cx, table_cy),
        top_z=table_top,
        clearance=0.42,
    )
    # Stash the tabletop geometry so the --check branch can ASSERT (a) the human's core stands clear IN
    # FRONT of the near edge and (b) every settled clutter XY is inside the tabletop-minus-margin. Plain
    # dynamic attribute on the SceneState (additive; no scene_state edit), like clutter_prims above.
    scene.clutter_table = {
        "cx": table_cx,
        "cy": table_cy,
        "half_x": plank[0] * 0.5,
        "half_y": top_half_y,
        "near_edge_y": near_edge_y,
        "top_z": table_top,
    }
    print(
        f"[step3r] clutter: table top z={table_top:.2f}, tabletop X[{table_cx - plank[0] * 0.5:.2f},"
        f"{table_cx + plank[0] * 0.5:.2f}] Y[{table_cy - top_half_y:.2f},{near_edge_y:.2f}]; "
        f"{len(clutter)} props planned+dropped; carried mug grab at {cup_xy} z={table_top + 0.05:.2f}"
    )

    program = "[Walk] <table> (1), [Grab] <cup> (1), [Put] <cup> (1) <table> (1)"
    tpv_look = (0.0, -0.7, 0.78)  # fixed front-elevated cam framing the whole tabletop
    tpv_eye = (0.0, 1.8, 1.4)
    return scene, {}, program, tpv_look, tpv_eye


def build_put_penetrate(stage, app):
    """P3 during-render de-penetration demo (fork of build_clutter): the human carries a mug and PUTS it
    at a target that overlaps a settled DYNAMIC blocker on the table. On release the mug flips
    kinematic->dynamic and PhysX de-penetrates it off the blocker (bounded by maxDepenetrationVelocity)
    + settles it ON CAMERA, read back per frame. Unlike build_clutter this sets scene.free_dynamic_prims
    (NOT clutter_prims) -> the P3 render path (proxy + substepping + dynamic readback) engages and NO
    kinematic-replay runs on the DYNAMIC prims.

    Staging so the de-penetration is a VISIBLE, valid correction of the PLACED object (v1 knocked the
    light can blocker off the table + barely moved the mug):
      * BLOCKER = a HEAVY (density ~10x the mug), WIDE, STABLE box (003_cracker_box, ~164x214mm base,
        ~72mm tall) sat CENTRAL on the tabletop, well clear of every edge. Heavy + central so PhysX moves
        the MUG, not the blocker, and no nudge can push it off. It stays a DYNAMIC free object.
      * PUT target overlaps the blocker's +X face by ``overlap`` (~6 cm, clearly visible) on the side
        facing WIDE-OPEN +X tabletop, so PhysX SLIDES the released mug sideways into free table space
        (never toward an edge) and it rests ON the table, clearly displaced from where it was placed.
        The overlap is kept < the box height so PhysX resolves it HORIZONTALLY (slide off), not
        VERTICALLY (climb over). +X is the RIGHT-hand side of the -Y-facing human, so the side-grip
        reach lands cleanly; the mug is GRABBED offset in +X (clear of the -X blocker), then carried
        front-center and placed over the blocker.
      * TPV is a SIDE-ANGLED, elevated cam offset in +X: the centered human no longer occludes the table
        center, so the blocker + sliding mug are seen PAST the body. (Its own tpv_eye/tpv_look, so the
        shared camera code + other scenes are untouched.) The FPV free-object focus follows the released
        mug via the existing focus_of / free_focus path."""
    import fpv_scene

    scene = SceneState(floor_z=0.0, start_xy=(0.0, 1.6))
    _clean_room(stage, app)

    # Thin static table in front of the human (-Y) -- the same safe thin-plank support build_clutter uses
    # (its collision AABB is only the thin top z-band, so a reaching torso grazes just the top). near_edge
    # FIXED at -0.395 so the geometry-derived stand-back matches build_clutter (a proven reach envelope).
    table_top = 0.75
    table_cx = 0.0
    near_edge_y = -0.395
    plank = (1.0, 0.55, 0.05)
    top_half_y = plank[1] * 0.5
    table_cy = near_edge_y - top_half_y
    _make_box(
        stage,
        "/World/Table",
        plank,
        (table_cx, table_cy, table_top - plank[2] * 0.5),
        color=(0.52, 0.40, 0.30),
    )
    for lx in (-0.42, 0.42):
        for ly in (-0.24, 0.24):
            _make_box(
                stage,
                f"/World/TableLeg{'L' if lx < 0 else 'R'}{'B' if ly < 0 else 'F'}",
                (0.05, 0.05, table_top),
                (table_cx + lx, table_cy + ly, table_top * 0.5),
                color=(0.42, 0.32, 0.24),
            )
    for _ in range(3):
        app.update()

    # ONE blocker prop -- a WIDE, STABLE box (003_cracker_box: lying flat ~164x214mm base, ~72mm tall),
    # NOT the tippy master_chef can v1 used. Spawn yaw=90 so its SHORTER (~164 mm) footprint axis runs
    # along Y (near face stays well inside the shallow table) and its LONGER (~214 mm) axis along X (the
    # slide axis, so the mug overlaps a wide face). Left DYNAMIC via free_dynamic_prims; run_render makes
    # it HEAVY (density 4000 = ~10x the mug) so the mug's de-penetration moves the MUG, not the blocker.
    # A tallish (~72 mm) box is deliberate: the mug de-penetrates along the SMALLER overlap axis, so the
    # box height must EXCEED the horizontal overlap or PhysX lifts the mug UP over the box instead of
    # sliding it sideways (a low ~46mm box did exactly that). Placed CENTRAL, clear of every edge.
    blocker_yaw = 90.0
    blocker = fpv_scene.spawn_prop(
        stage,
        "Blocker",
        "Props/YCB/Axis_Aligned/003_cracker_box.usd",
        table_top,
        (table_cx, -0.53),  # temporary xy; re-seated below once its footprint is known
        blocker_yaw,
    )
    app.update()
    import object_physics as _ophys_b

    b_rng = fpv_scene.compute_world_bbox(blocker)
    b_half_x = float(b_rng.GetMax()[0] - b_rng.GetMin()[0]) * 0.5
    b_half_y = float(b_rng.GetMax()[1] - b_rng.GetMin()[1]) * 0.5
    b_height = float(b_rng.GetMax()[2] - b_rng.GetMin()[2])
    blocker_path = str(blocker.GetPath())
    # spawn_prop already seated the bbox-bottom on the tabletop; capture that ORIGIN z (invariant to XY
    # at a fixed yaw) so the re-seat below keeps the base on the table. (local_bbox_min_z is only correct
    # when the prim sits at z=0 -- spawn_prop probes it there -- so we reuse the seated z instead.)
    b_seat_z = float(_ophys_b.read_prim_xyz(stage, blocker_path)[2])

    # The mug slides in +X, so the overlap uses the X footprints. cup_half_x ~= the mug's X half-extent
    # (025_mug ~58 mm). overlap = a clearly-visible ~6.5 cm horizontal embed (in the task's 5-8 cm band),
    # capped a hair BELOW the box height so PhysX resolves the overlap HORIZONTALLY (slide off) not
    # VERTICALLY (climb over), bounded by max_depen_vel to a gentle SLIDE. The realized slide comes out a
    # touch under the embed (the hulls separate on contact); the mug ends resting on the table ~overlap
    # from where it was placed, ~cup_half_x + b_half_x from the blocker center.
    cup_half_x = 0.058
    overlap = min(0.065, max(0.0, b_height - 0.008))
    # Blocker Y = a reachable depth, clear of the near edge (near face = blocker_y + b_half_y stays >~5 cm
    # inside near_edge_y). Blocker X placed so the PUT target (on its +X face) lands at table_cx (center =
    # best reach); the mug then slides in +X to rest ~cup_half_x + b_half_x from the blocker center. Re-seat
    # (set_world_pose, NOT a 2nd spawn_prop) at the final XY; the seated rest z from the first spawn is
    # yaw-only and unchanged, so re-seating at the same yaw keeps the base on the tabletop.
    blocker_y = -0.53
    blocker_x = table_cx - (b_half_x + cup_half_x - overlap)
    fpv_scene.set_world_pose(
        blocker,
        (blocker_x, blocker_y, b_seat_z),
        fpv_scene._to_wxyz(fpv_scene._yaw_quat(blocker_yaw)),
    )
    app.update()

    # PUT target = table_cx on the blocker's +X face (overlap into the blocker); the mug placed there
    # penetrates the blocker by ``overlap`` and PhysX slides it +X into open table.
    place_xy = (table_cx, blocker_y)

    # The table as a footprint SURFACE (thin-top shelf pattern), with its PLACE xy = the penetrating
    # target over the blocker. foot_xy/foot_half = the tabletop footprint (stand-back / nudge treat it
    # like build_clutter's/highreach's shelf: a hand-graze over the top, not a solid pedestal).
    scene.register(
        "table",
        Role.SURFACE,
        xy=place_xy,
        z=table_top,
        foot_xy=(table_cx, table_cy),
        foot_half=(plank[0] * 0.5, top_half_y),
        top_z=table_top,
        approach_yaw=0.0,
        clearance=0.42,
    )
    # The CARRIED mug -> the single rig /World/Obj, grabbed NEAR the human but OFFSET to +X, clear of the
    # blocker (which extends in -X): the env-init settle drops both, so their rest footprints must not
    # overlap. +X is the right-hand side of the -Y-facing human (natural side-grip reach). It is then
    # carried to the front-center PUT target over the blocker. cup_y kept shallow (near the human) for a
    # clean grab reach, with ~3 cm of near-edge margin. top_z = the TABLE top.
    cup_x = max(table_cx + 0.13, blocker_x + b_half_x + cup_half_x + 0.03)
    cup_xy = (cup_x, near_edge_y - 0.075)
    scene.register(
        "cup",
        Role.GRASPABLE,
        xy=cup_xy,
        z=table_top + 0.05,
        approach_yaw=0.0,
        foot_xy=(table_cx, table_cy),
        top_z=table_top,
        clearance=0.42,
    )

    # Opt into the P3 during-render dynamics: BOTH the blocker (free from init) and the released mug
    # (/World/Obj, freed at the [Put] release frame) are PhysX-owned. Additive dynamic attribute on the
    # SceneState (a plain dataclass), like clutter_prims -- NO scene_state edit. Do NOT set clutter_prims
    # / clutter_traj (no kinematic replay on DYNAMIC prims).
    scene.free_dynamic_prims = [blocker_path, "/World/Obj"]
    # Stash the penetrating target + the blocker path so the --check pp_ok assertion can place the
    # released mug at the penetrating target and prove PhysX de-penetrates it to a clean on-table rest.
    scene.put_penetrate = {
        "obj_path": "/World/Obj",
        "blocker_path": blocker_path,
        "target": (float(place_xy[0]), float(place_xy[1]), float(table_top)),
        "top_z": table_top,
        "blocker_density": 4000.0,  # ~10x the mug (400) -> de-pen moves the MUG, not the blocker
    }
    for _ in range(2):
        app.update()
    print(
        f"[step3r] put_penetrate: table top z={table_top:.2f}; blocker '{blocker_path.rsplit('/', 1)[-1]}' "
        f"at ({blocker_x:.3f},{blocker_y:.3f}) (half_x={b_half_x * 1000:.0f}mm half_y={b_half_y * 1000:.0f}mm "
        f"height={b_height * 1000:.0f}mm, HEAVY+DYNAMIC); mug grab at {cup_xy}; PUT target "
        f"{tuple(round(v, 3) for v in place_xy)} overlaps blocker +X face by ~{overlap * 1000:.0f}mm "
        f"(< box height -> mug slides +X into open table, not up)"
    )

    program = "[Walk] <table> (1), [Grab] <cup> (1), [Put] <cup> (1) <table> (1)"
    # SIDE-ANGLED, elevated cam offset in +X so the centered human does not occlude the table center;
    # the blocker + the mug sliding into +X are seen past the body. Aim at the tabletop interaction band.
    tpv_look = (0.1, -0.55, 0.78)
    tpv_eye = (2.2, 1.0, 1.55)
    return scene, {}, program, tpv_look, tpv_eye


# =================================================================================================
# Full-environment scenes (warehouse / hospital): a REAL Isaac environment as the backdrop + real
# obstacles (its baked per-mesh colliders), our OWN thin-plank pick + place tables in an OPEN floor
# area, and a walk -> grab -> walk -> put program. The env is referenced at /World/Room (skipped by
# _NON_FURNITURE / _apply_scene_colliders) so it never becomes ONE giant collider, yet its baked
# colliders still feed the init non-pen gate as genuine obstacles (racks / walls). The human starts
# NEAR the action (scene.near_start) so the straight-line walk stays inside the validated open area --
# the runtime nudge only avoids OUR registered furniture, not the env geometry. Both envs ship their
# floor at z=0 (probed: warehouse decals z=0.000, hospital floors z=0.001) so NO floor shift is needed.
# =================================================================================================
_WAREHOUSE_USD = "/Environments/Simple_Warehouse/warehouse.usd"
_HOSPITAL_USD = "/Environments/Hospital/hospital.usd"


def _sphere_light(stage, path, xy, z=3.5, intensity=18000.0, radius=1.5):
    """A soft, high, broad local sphere light (env interiors are dim headless -> the action would be
    black). Kept dim + high + broad so the REFLECTIVE env floor is lit, not blown to pure white -- a hot
    low point light over the bright warehouse/hospital floor makes a specular hotspot that fills the FPV
    the instant the head pitches DOWN to reach the 0.75 m table (the pick/put frames washed out at 35k).
    """
    import fpv_scene
    from pxr import UsdLux

    lt = UsdLux.SphereLight.Define(stage, path)
    lt.CreateRadiusAttr(radius)
    lt.CreateIntensityAttr(intensity)
    lt.CreateColorAttr((1.0, 1.0, 1.0))
    fpv_scene.set_world_pose(lt.GetPrim(), (float(xy[0]), float(xy[1]), float(z)))


def _plank_table(
    stage,
    name,
    cx,
    near_edge_y,
    top=0.75,
    plank=(1.0, 0.55, 0.05),
    color=(0.52, 0.40, 0.30),
):
    """Build a thin-top plank table (the proven clutter/highreach support: a safe settle surface AND a
    reaching torso grazes only the thin top, not a solid pedestal) whose +Y NEAR edge (toward the human)
    sits at ``near_edge_y``. Returns (plank_path, cx, cy_center, half_x, top_half_y, top).
    """
    top_half_y = plank[1] * 0.5
    cy = (
        near_edge_y - top_half_y
    )  # push the plank back so its +Y edge is at near_edge_y
    plank_path = f"/World/{name}"
    _make_box(stage, plank_path, plank, (cx, cy, top - plank[2] * 0.5), color=color)
    leg = (color[0] * 0.8, color[1] * 0.8, color[2] * 0.8)
    for lx in (-plank[0] * 0.42, plank[0] * 0.42):
        for ly in (-top_half_y * 0.8, top_half_y * 0.8):
            _make_box(
                stage,
                f"/World/{name}Leg{'L' if lx < 0 else 'R'}{'B' if ly < 0 else 'F'}",
                (0.05, 0.05, top),
                (cx + lx, cy + ly, top * 0.5),
                color=leg,
            )
    return plank_path, cx, cy, plank[0] * 0.5, top_half_y, top


def _build_open_area_demo(
    stage,
    app,
    usd_rel,
    start_xy,
    bench,
    pallet,
    tpv_eye,
    tpv_look,
    mat_color=None,
    light_intensity=18000.0,
    dome_intensity=250.0,
    table_colors=((0.52, 0.40, 0.30), (0.34, 0.38, 0.44)),
):
    """Shared builder for the warehouse / hospital walk -> grab -> walk -> put demos. ``bench`` / ``pallet``
    are (cx, near_edge_y) for the pick table (holds the graspable) and the place table. SPEC grounding
    (explicit poses) + scene.near_start so the seeded init gate starts the human near the action.
    ``mat_color`` optionally draws a matte work-mat over the action; None keeps the env floor.
    ``light_intensity`` tunes the local action lights per env (the warehouse's yellow walls + glossy
    floor bounce a lot -> it needs about half the neutral hospital's light or the downward FPV blows out).
    """
    import fpv_scene
    from pxr import UsdLux

    scene = SceneState(floor_z=0.0, start_xy=start_xy)
    env = fpv_scene.add_reference(stage, "/World/Room", fpv_scene.ASSET_ROOT + usd_rel)
    fpv_scene.set_world_pose(
        env, (0.0, 0.0, 0.0)
    )  # floor already at z=0 (probed) -> no shift
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights_Dome")
    dome.CreateIntensityAttr(dome_intensity)
    for _ in range(3):
        app.update()

    # Ground under the action (Cube gprim -> exact 'none' collider; its top < 0.15 -> excluded from the
    # occupancy grid + overlap gate, so feet never false-fire). With mat_color it is a VISIBLE matte
    # work-mat whose top sits just above the env floor (z=0.006): the downward-looking FPV grab/put
    # frames then see a matte surface, not the glossy white warehouse floor blown to pure white. Without
    # it (hospital: the speckled clinical floor already reads well) a hidden slab (top z=0) just catches
    # a stray drop at floor level.
    mid_xy = (0.5 * (bench[0] + pallet[0]), 0.5 * (bench[1] + pallet[1]))
    if mat_color is not None:
        _make_box(
            stage,
            "/World/GroundCol",
            (12.0, 5.0, 0.012),
            (mid_xy[0], mid_xy[1], 0.0),
            color=mat_color,
        )
    else:
        _make_box(
            stage,
            "/World/GroundCol",
            (18.0, 12.0, 0.10),
            (mid_xy[0], mid_xy[1], -0.05),
            color=(0.5, 0.5, 0.5),
        )

    b_path, bcx, bcy, b_hx, b_thy, b_top = _plank_table(
        stage, "Bench", bench[0], bench[1], color=table_colors[0]
    )
    p_path, pcx, pcy, p_hx, p_thy, p_top = _plank_table(
        stage, "Pallet", pallet[0], pallet[1], color=table_colors[1]
    )
    _sphere_light(stage, "/World/Lights_Bench", (bcx, bcy), intensity=light_intensity)
    _sphere_light(stage, "/World/Lights_Pallet", (pcx, pcy), intensity=light_intensity)
    for _ in range(2):
        app.update()

    # Pick table = a footprint SURFACE (thin-top shelf pattern); the graspable is seated at its near edge.
    scene.register(
        "bench",
        Role.SURFACE,
        xy=(bcx, bench[1] - 0.08),
        z=b_top,
        foot_xy=(bcx, bcy),
        foot_half=(b_hx, b_thy),
        top_z=b_top,
        approach_yaw=0.0,
        clearance=0.42,
    )
    # Place table = the [Put] target surface (the "another place" the object is carried to).
    scene.register(
        "pallet",
        Role.SURFACE,
        xy=(pcx, pallet[1] - 0.08),
        z=p_top,
        foot_xy=(pcx, pcy),
        foot_half=(p_hx, p_thy),
        top_z=p_top,
        approach_yaw=0.0,
        clearance=0.42,
    )
    # The carried/graspable object -> the single rig /World/Obj, grabbed at the bench near-edge front.
    scene.register(
        "cup",
        Role.GRASPABLE,
        xy=(bcx, bench[1] - 0.06),
        z=b_top + 0.05,
        foot_xy=(bcx, bcy),
        top_z=b_top,
        approach_yaw=0.0,
        clearance=0.42,
    )
    # The bench / pallet planks are exact static settle supports (the graspable settles onto the bench).
    scene.behavior_support_prims = [b_path, p_path]
    scene.near_start = True  # seeded init gate: start NEAR the action (short walk stays in the open area)
    scene.obj_color = (
        0.85,
        0.15,
        0.12,
    )  # strong red so the cup reads against the bright env floor (FPV)

    program = "[Walk] <bench> (1), [Grab] <cup> (1), [Walk] <pallet> (1), [Put] <cup> (1) <pallet> (1)"
    print(
        f"[step3r] open-area demo env={usd_rel}: start={start_xy} bench@({bcx:.1f},{bench[1]:.1f}) "
        f"pallet@({pcx:.1f},{pallet[1]:.1f}) top={b_top:.2f}"
    )
    return scene, {}, program, tpv_look, tpv_eye


def build_warehouse(stage, app):
    """Walk -> pick -> walk -> place inside the REAL Isaac WAREHOUSE, in the central open aisle (shelving
    racks frame the background). The env's baked rack / wall colliders feed the init non-pen gate as real
    obstacles; our two plank tables are the interactables. See [[isaac-real-furniture-assets]].
    """
    return _build_open_area_demo(
        stage,
        app,
        _WAREHOUSE_USD,
        start_xy=(-4.5, 1.6),
        bench=(-3.0, -0.10),
        pallet=(3.0, -0.10),
        tpv_eye=(-5.5, 2.2, 2.4),
        tpv_look=(0.5, -0.4, 0.75),
        # The warehouse's GLOSSY white epoxy floor throws a specular hotspot that blows the downward FPV
        # grab/put to pure white (dimming alone did not fix it). Cover the action with a MATTE dark
        # charcoal work-mat + matte dark workbench tops (our _make_box surfaces are matte -> no specular),
        # so the straight-down FPV sees a non-blown surface and the red cup reads. Half light for margin.
        mat_color=(0.16, 0.16, 0.18),
        table_colors=((0.30, 0.26, 0.22), (0.24, 0.26, 0.30)),
        light_intensity=3000.0,
        dome_intensity=40.0,  # low + warm sphere lights dominate -> neutral (not bluish) + not washed out
    )


def build_hospital(stage, app):
    """Walk -> pick -> walk -> place inside the REAL Isaac HOSPITAL, in the open atrium (rooms / walls
    frame the background). Same additive mechanism as build_warehouse."""
    return _build_open_area_demo(
        stage,
        app,
        _HOSPITAL_USD,
        start_xy=(5.5, 8.2),
        bench=(7.0, 6.9),
        pallet=(12.5, 6.9),
        # NW-corner diagonal (the atrium is shallow in Y, so a front cam can't stand far enough back to
        # frame both x=7 and x=12.5 tables at 24mm -> the human was cut off the left edge). The diagonal
        # gives ~7 m standoff to the pallet so the whole walk + both tables + the human stay in frame.
        tpv_eye=(6.0, 8.7, 2.8),
        tpv_look=(11.5, 6.4, 0.6),
    )


# =================================================================================================
# AUTO env: "just pass a USD path". Auto-derives the action region from the occupancy grid (no
# hand-picked coordinates) + auto-exposes the local lights (no hand-tuned intensity). Same additive
# machinery as build_warehouse/hospital -- env at /World/Room (backdrop + real colliders), our own two
# plank tables in the largest open on-floor spot, walk->grab->walk->put, --seed init gate enforced.
# =================================================================================================
def _detect_floor_z(footprints) -> float:
    """Floor z = the TOP of the largest THIN, near-ground collidable footprint (an env's floor slab).
    Defaults to 0.0 if nothing floor-like is found (most Isaac envs ship the floor at z=0).
    """
    best_area, best_top = 0.0, None
    for _o, _cx, _cy, hx, hy, bot, top in footprints:
        if (top - bot) < 0.4 and -1.0 <= top <= 1.0:  # thin slab near the ground
            area = (2.0 * hx) * (2.0 * hy)
            if area > best_area:
                best_area, best_top = area, top
    return float(best_top) if best_top is not None else 0.0


def _auto_action_layout(stage) -> dict:
    """Pick a walk->grab->walk->put layout in the LARGEST open, on-floor area of ANY loaded env, with NO
    hand-picked coordinates. Reuses the init-validation occupancy grid (the same one the init gate uses)
    + searches the 4 axis-aligned orientations for a placement whose tables/stands/walk-lane/start are all
    on-floor and clear of obstacles. Returns a layout dict; raises if the env has no room for the action.
    """
    import time as _time

    import init_validation as iv

    _t0 = _time.time()
    fps = iv.collidable_footprints(stage)
    if not fps:
        raise RuntimeError(
            "_auto_action_layout: no collidable footprints (env not loaded?)"
        )
    floor_z = _detect_floor_z(fps)
    # Drop tiny footprints (desk clutter: pencils/markers etc.): they sit within a desk's own footprint,
    # so they add nothing to FLOOR navigation but bloat the O(obstacles*cells) grid build. Keep floor
    # patches + any obstacle >= 8 cm. Cuts an instanced office's ~3.6k footprints to the ~hundreds that
    # actually bound the walkable floor.
    fps = [
        f
        for f in fps
        if not iv._blocks_floor(f[5], f[6], floor_z, 0.15) or max(f[3], f[4]) >= 0.08
    ]
    print(
        f"[autoregion] {len(fps)} significant footprints, floor_z={floor_z:.2f} "
        f"({_time.time() - _t0:.1f}s)"
    )
    # Obstacles that rise into the standing column (walls + furniture). The INTERIOR is bounded by their
    # XY extent (the outermost are the walls); a cell is "interior" if inside that box shrunk by 0.6 m.
    # This replaces an on-floor-mesh test: an instanced env's visible floor carries NO collider (only a
    # single degenerate ground PLANE with no XY extent), so the walkable interior is bounded by the walls.
    blocking = np.array(
        [
            (cx, cy, hx, hy)
            for _o, cx, cy, hx, hy, bot, top in fps
            if iv._blocks_floor(bot, top, floor_z, 0.15)
        ],
        dtype=np.float64,
    )
    if blocking.size == 0:
        raise RuntimeError(
            "_auto_action_layout: no floor-blocking obstacles (env has no walls/furniture?)"
        )
    # ROBUST interior bounds via 2/98 percentiles of the obstacle cluster: some envs park a stray
    # collider (skybox/far light) at +-hundreds of metres, which a plain min/max bbox would turn into a
    # kilometre-wide grid. Percentiles clip to the DENSE building. Then a 1 m pad for the grid region.
    lo_x = float(np.percentile(blocking[:, 0] - blocking[:, 2], 2))
    hi_x = float(np.percentile(blocking[:, 0] + blocking[:, 2], 98))
    lo_y = float(np.percentile(blocking[:, 1] - blocking[:, 3], 2))
    hi_y = float(np.percentile(blocking[:, 1] + blocking[:, 3], 98))
    ix0, ix1, iy0, iy1 = lo_x + 0.6, hi_x - 0.6, lo_y + 0.6, hi_y - 0.6
    region = (lo_x - 1.0, hi_x + 1.0, lo_y - 1.0, hi_y + 1.0)
    grid = iv.build_occupancy_grid(
        fps, region, cell=0.25, inflate=iv.HUMAN_FOOT_R, floor_z=floor_z
    )
    ny, nx = grid.clearance.shape
    print(
        f"[autoregion] grid {nx}x{ny} over {tuple(round(v, 1) for v in region)} ({_time.time() - _t0:.1f}s)"
    )

    def interior(x, y) -> bool:
        return ix0 <= x <= ix1 and iy0 <= y <= iy1

    def clr(x, y) -> float:
        i = int((x - grid.x0) / grid.cell)
        j = int((y - grid.y0) / grid.cell)
        if 0 <= i < nx and 0 <= j < ny:
            return float(grid.clearance[j, i])
        return -1e9

    # Candidate anchors: cells that are OPEN ENOUGH for the action but SURROUNDED by the env (so the
    # backdrop reads as a real room, not an empty exterior slab -- office.usd ships parking/surroundings
    # where a "most-open" pick would land at 60 m clearance). Clearance band [2.4, 8] m = room enough for
    # a ~5 m walk yet still enclosed; rank by CENTRALITY to the obstacle-cluster median (the building
    # core). VECTORIZED. Relaxes the upper cap, then the lower, if a tight env has nothing in-band.
    xs = grid.x0 + (np.arange(nx) + 0.5) * grid.cell
    ys = grid.y0 + (np.arange(ny) + 0.5) * grid.cell
    gx, gy = np.meshgrid(xs, ys)
    in_box = (gx >= ix0) & (gx <= ix1) & (gy >= iy0) & (gy <= iy1)
    mx0, my0 = float(np.median(blocking[:, 0])), float(np.median(blocking[:, 1]))
    for lo, hi in ((2.4, 8.0), (2.4, 1e9), (0.6, 1e9)):
        mask = in_box & (grid.clearance >= lo) & (grid.clearance <= hi)
        jj, ii = np.nonzero(mask)
        if jj.size:
            break
    if jj.size == 0:
        raise RuntimeError("_auto_action_layout: no open interior cell found")
    cxs, cys = xs[ii], ys[jj]
    order = np.argsort(np.hypot(cxs - mx0, cys - my0))[
        :80
    ]  # most CENTRAL (inside the building) first
    cand = [
        (float(grid.clearance[jj[k], ii[k]]), float(cxs[k]), float(cys[k]))
        for k in order
    ]
    print(
        f"[autoregion] {len(cand)} interior candidates near cluster median ({mx0:.1f},{my0:.1f}); "
        f"nearest clr={cand[0][0]:.2f} at ({cand[0][1]:.1f},{cand[0][2]:.1f})"
    )

    # (face_dir the human looks toward the table, walk_dir tables separate along, approach_yaw).
    orients = [
        ((0.0, -1.0), (1.0, 0.0), 0.0),
        ((0.0, 1.0), (1.0, 0.0), 180.0),
        ((1.0, 0.0), (0.0, 1.0), 90.0),
        ((-1.0, 0.0), (0.0, 1.0), 270.0),
    ]
    half_depth, stand_clear, start_back = 0.275, 0.42, 1.3
    for walk_len in (5.0, 4.0, 3.0, 2.4):
        for _c, ax, ay in cand:
            C = np.array([ax, ay], dtype=np.float64)
            for face_t, u_t, yaw in orients:
                face = np.array(face_t)
                u = np.array(u_t)
                bench_c = C - u * (walk_len / 2.0)
                pallet_c = C + u * (walk_len / 2.0)
                b_stand = bench_c - face * (half_depth + stand_clear)
                p_stand = pallet_c - face * (half_depth + stand_clear)
                start = b_stand - face * start_back
                mids = [b_stand + u * (walk_len * f) for f in (0.25, 0.5, 0.75)]
                beyond = [bench_c + face * 0.6, pallet_c + face * 0.6]
                ok = all(
                    interior(p[0], p[1]) and clr(p[0], p[1]) >= 0.55
                    for p in (bench_c, pallet_c)
                )
                ok = ok and all(
                    interior(p[0], p[1]) and clr(p[0], p[1]) >= iv.HUMAN_FOOT_R
                    for p in ([b_stand, p_stand, start] + mids)
                )
                ok = ok and all(clr(p[0], p[1]) >= 0.15 for p in beyond)
                if ok:
                    print(
                        f"[autoregion] floor_z={floor_z:.2f} center=({ax:.2f},{ay:.2f}) yaw={yaw:.0f} "
                        f"walk_len={walk_len:.1f} bench=({bench_c[0]:.2f},{bench_c[1]:.2f}) "
                        f"pallet=({pallet_c[0]:.2f},{pallet_c[1]:.2f})"
                    )
                    return {
                        "floor_z": floor_z,
                        "center": (float(ax), float(ay)),
                        "face": (float(face[0]), float(face[1])),
                        "walk_dir": (float(u[0]), float(u[1])),
                        "yaw": float(yaw),
                        "walk_len": float(walk_len),
                        "bench_c": (float(bench_c[0]), float(bench_c[1])),
                        "pallet_c": (float(pallet_c[0]), float(pallet_c[1])),
                        "start_xy": (float(start[0]), float(start[1])),
                    }
    raise RuntimeError(
        "_auto_action_layout: no open region fits the walk->grab->walk->put action (env too cluttered)"
    )


def _oriented_plank_table(
    stage, name, center, u, face, top, dp=0.55, w=1.0, color=(0.30, 0.26, 0.22)
):
    """A thin-top plank table oriented so its WIDTH is along ``u`` (walk axis) and DEPTH along ``face``
    (the human reaches over the -face near edge). Axis-aligned (u, face are axis unit vectors) -> its
    world size is just the swapped dims. Returns (path, cx, cy, half_x, half_y)."""
    ux, uy = abs(u[0]), abs(u[1])
    fx, fy = abs(face[0]), abs(face[1])
    sx = ux * w + fx * dp
    sy = uy * w + fy * dp
    cx, cy = float(center[0]), float(center[1])
    path = f"/World/{name}"
    _make_box(stage, path, (sx, sy, 0.05), (cx, cy, top - 0.025), color=color)
    leg = (color[0] * 0.8, color[1] * 0.8, color[2] * 0.8)
    for su in (-1.0, 1.0):
        for sf in (-1.0, 1.0):
            lx = cx + su * (w * 0.42) * u[0] + sf * (dp * 0.35) * face[0]
            ly = cy + su * (w * 0.42) * u[1] + sf * (dp * 0.35) * face[1]
            _make_box(
                stage,
                f"/World/{name}Leg{'A' if su < 0 else 'B'}{'C' if sf < 0 else 'D'}",
                (0.05, 0.05, top),
                (lx, ly, top * 0.5),
                color=leg,
            )
    return path, cx, cy, sx * 0.5, sy * 0.5


def _build_auto_env(stage, app, usd_rel, tag=""):
    """Generic builder: reference ANY full Isaac env at /World/Room, AUTO-pick the action region from the
    occupancy grid, AUTO-expose the local lights (run_render), and run walk->grab->walk->put. "Just pass a
    USD path." Strictly additive; --seed init gate enforced (validates vs the env's real colliders).
    """
    import time as _time

    import fpv_scene
    from pxr import UsdLux

    _t0 = _time.time()
    env = fpv_scene.add_reference(stage, "/World/Room", fpv_scene.ASSET_ROOT + usd_rel)
    fpv_scene.set_world_pose(env, (0.0, 0.0, 0.0))
    for _ in range(4):
        app.update()  # compose the reference so the collidable bboxes read current
    print(
        f"[autoenv] {usd_rel} referenced + composed ({_time.time() - _t0:.1f}s); computing layout..."
    )

    layout = _auto_action_layout(stage)
    floor_z = layout["floor_z"]
    face = np.array(layout["face"])
    u = np.array(layout["walk_dir"])
    yaw = layout["yaw"]
    top = floor_z + 0.75
    dp = 0.55

    scene = SceneState(floor_z=floor_z, start_xy=layout["start_xy"])
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights_Dome")
    dome.CreateIntensityAttr(120.0)  # nominal; auto-exposure calibrates the final level

    # Hidden ground collider at the detected floor so a stray drop rests at floor level.
    cx0, cy0 = layout["center"]
    _make_box(
        stage,
        "/World/GroundCol",
        (20.0, 20.0, 0.10),
        (cx0, cy0, floor_z - 0.05),
        color=(0.5, 0.5, 0.5),
    )

    b_path, bcx, bcy, b_hx, b_hy = _oriented_plank_table(
        stage, "Bench", layout["bench_c"], u, face, top, dp=dp, color=(0.30, 0.26, 0.22)
    )
    p_path, pcx, pcy, p_hx, p_hy = _oriented_plank_table(
        stage,
        "Pallet",
        layout["pallet_c"],
        u,
        face,
        top,
        dp=dp,
        color=(0.24, 0.26, 0.30),
    )
    _sphere_light(
        stage, "/World/Lights_Bench", (bcx, bcy), z=floor_z + 3.0, intensity=9000.0
    )
    _sphere_light(
        stage, "/World/Lights_Pallet", (pcx, pcy), z=floor_z + 3.0, intensity=9000.0
    )
    for _ in range(2):
        app.update()

    near = (
        dp / 2.0
    )  # center -> near (-face) edge; seat objects/targets just inside it (toward the human)
    b_cup = (bcx - face[0] * (near - 0.06), bcy - face[1] * (near - 0.06))
    b_surf = (bcx - face[0] * (near - 0.08), bcy - face[1] * (near - 0.08))
    p_surf = (pcx - face[0] * (near - 0.08), pcy - face[1] * (near - 0.08))
    scene.register(
        "bench",
        Role.SURFACE,
        xy=b_surf,
        z=top,
        foot_xy=(bcx, bcy),
        foot_half=(b_hx, b_hy),
        top_z=top,
        approach_yaw=yaw,
        clearance=0.42,
    )
    scene.register(
        "pallet",
        Role.SURFACE,
        xy=p_surf,
        z=top,
        foot_xy=(pcx, pcy),
        foot_half=(p_hx, p_hy),
        top_z=top,
        approach_yaw=yaw,
        clearance=0.42,
    )
    scene.register(
        "cup",
        Role.GRASPABLE,
        xy=b_cup,
        z=top + 0.05,
        foot_xy=(bcx, bcy),
        top_z=top,
        approach_yaw=yaw,
        clearance=0.42,
    )
    scene.behavior_support_prims = [b_path, p_path]
    scene.near_start = True
    scene.obj_color = (0.85, 0.15, 0.12)
    scene.auto_expose = (
        True  # run_render calibrates the light intensity to a target (no hand-tuning)
    )
    scene.expose_xy = (
        b_cup  # calibrate exposure on the reach surface (the downward-FPV grab spot)
    )
    scene.expose_z = top + 0.1

    program = "[Walk] <bench> (1), [Grab] <cup> (1), [Walk] <pallet> (1), [Put] <cup> (1) <pallet> (1)"
    C = np.array(layout["center"])
    wl = layout["walk_len"]
    tpv_eye = (
        float(C[0] - face[0] * 2.6 - u[0] * (wl * 0.5 + 1.6)),
        float(C[1] - face[1] * 2.6 - u[1] * (wl * 0.5 + 1.6)),
        floor_z + 2.8,
    )
    tpv_look = (float(C[0]), float(C[1]), floor_z + 0.7)
    print(
        f"[autoenv] {usd_rel}: start={layout['start_xy']} yaw={yaw:.0f} walk_len={wl:.1f} floor_z={floor_z:.2f}"
    )
    return scene, {}, program, tpv_look, tpv_eye


def build_office(stage, app):
    """Walk -> pick -> walk -> place inside the REAL Isaac OFFICE -- built by the GENERIC auto-env path
    (auto region + auto exposure). Demonstrates "just pass a USD path"."""
    return _build_auto_env(stage, app, "/Environments/Office/office.usd")


# Single-slot memo behind _nudge_memo: (static_boxes, removed_names, base). Module-level because
# a process hosts ONE live scene (the RL pool runs one SimulationApp per subprocess); a second scene just
# misses the identity guard and rebuilds.
_NB_MEMO: tuple | None = None


def _nudge_memo(static_boxes, removed_names):
    """Frame-invariant static rows minus collider owners replaced by live openables.

    The slot keeps a STRONG reference to the exact list it was built from and hits only on ``is``:
    holding that list alive means its address can never be recycled under a stale entry, which an
    ``id()`` key would silently read as a hit. ``solid_boxes()`` returns a FRESH list of FRESH corner
    arrays on every call and no caller mutates either, so identity here also implies equal contents.
    """
    global _NB_MEMO
    memo = _NB_MEMO
    if memo is not None and memo[0] is static_boxes and memo[1] == removed_names:
        return memo[2]
    base = [(n, mn, mx) for (n, mn, mx) in static_boxes if n not in removed_names]
    _NB_MEMO = (static_boxes, set(removed_names), base)
    return base


def _path_is_in_subtree(path: str, root: str) -> bool:
    normalized_path = str(path).rstrip("/")
    normalized_root = str(root).rstrip("/")
    return normalized_path == normalized_root or normalized_path.startswith(
        normalized_root + "/"
    )


def _controller_moving_paths(ctrl) -> tuple[str, ...]:
    paths = tuple(
        str(path).rstrip("/") for path in ctrl.get("moving_paths", ()) if path
    )
    if paths:
        return paths
    leaf_specs = ctrl.get("leaf_specs") or ()
    return tuple(str(spec[0]).rstrip("/") for spec in leaf_specs if spec and spec[0])


def _controller_owner_rows(ctrl, static_boxes):
    paths = _controller_moving_paths(ctrl)
    if paths:
        rows = [
            row
            for row in static_boxes
            if any(_path_is_in_subtree(row[0], path) for path in paths)
        ]
        if rows:
            return rows
    names = {str(name) for name in ctrl.get("leaf_only_static_names", ()) if name}
    return [row for row in static_boxes if str(row[0]) in names]


def _rotated_leaf_box(aabb, hinge, angle: float, axis=None):
    if axis is not None:
        return se._rotated_aabb_about_axis(aabb, hinge, axis, angle)
    mn, mx = aabb
    c, s = math.cos(angle), math.sin(angle)
    xs, ys = [], []
    for px, py in ((mn[0], mn[1]), (mn[0], mx[1]), (mx[0], mn[1]), (mx[0], mx[1])):
        vx, vy = px - hinge[0], py - hinge[1]
        xs.append(vx * c - vy * s + hinge[0])
        ys.append(vx * s + vy * c + hinge[1])
    return (
        np.array([min(xs), min(ys), mn[2]], dtype=np.float64),
        np.array([max(xs), max(ys), mx[2]], dtype=np.float64),
    )


def _exclude_frame_contact_boxes(static_boxes, fr):
    roots = tuple(getattr(fr, "collision_exclude_paths", ()) or ())
    if not roots:
        return static_boxes
    return [
        row
        for row in static_boxes
        if not any(_path_is_in_subtree(str(row[0]), root) for root in roots)
    ]


def _nudge_boxes_for(scene, controllers, static_boxes, fr):
    """Per-frame nudge box set = the STATIC solid boxes + each openable drawer's CURRENT world extent, so
    the nudge sees a drawer that has slid OPEN toward the human (which solid_boxes() -- static closed
    footprints -- cannot). NUDGE-ONLY: the reach gate stays on scene.solid_boxes(). Moving collider rows
    are replaced by their live translated/rotated AABBs using exact prim-path ancestry. No controllers
    (walkby/sit/AMASS) -> just the static boxes (unchanged).

    Only the drawer/door extents below are frame-VARYING; the whole-scene map + base list come from
    ``_nudge_memo``. ``ISAAC_NUDGEBOX_MEMO=0`` rebuilds them every frame instead (the two paths are
    bit-identical, so it exists only to let an A/B flip them in-process)."""
    static_boxes = _exclude_frame_contact_boxes(static_boxes, fr)
    if not controllers:
        return static_boxes
    # A revolute door's CLOSED static box (from its Entity foot_half) spans the opening -> it would nudge
    # the human away from the doorway they must walk through once it is open. Drop it and replace with the
    # LIVE swept leaf box (cur_angle): closed -> spans the opening (human approaches from the front); open
    # -> the leaf is against the wall, leaving the opening clear.
    owner_rows = {
        str(row[0])
        for ctrl in controllers.values()
        for row in _controller_owner_rows(ctrl, static_boxes)
    }
    if os.environ.get("ISAAC_NUDGEBOX_MEMO") == "0":
        boxes = [(n, mn, mx) for (n, mn, mx) in static_boxes if n not in owner_rows]
    else:
        base = _nudge_memo(static_boxes, owner_rows)
        # Copy: the appends below must never reach the memoized list, and every caller so far has been
        # handed a fresh list per frame (humanoid_motion._packed_nudge_boxes keys its pack cache on
        # that list's identity).
        boxes = list(base)
    for name, ctrl in controllers.items():
        if ctrl.get("kind") == "revolute":
            value = float(ctrl.get("cur_angle", 0.0))
            specs = ctrl.get("leaf_specs") or (
                (
                    str(ctrl["leaf"].GetPath()),
                    ctrl["hinge_xy"],
                    ctrl["swing_sign"],
                    ctrl["leaf_aabb"],
                ),
            )
            for index, (path, hinge, sign, aabb) in enumerate(specs):
                rows = [
                    row for row in static_boxes if _path_is_in_subtree(row[0], path)
                ]
                sources = [(row[1], row[2]) for row in rows] or [aabb]
                for collider_index, source in enumerate(sources):
                    mn, mx = _rotated_leaf_box(
                        source,
                        hinge,
                        _door_leaf_angle(ctrl, sign, value),
                        axis=ctrl.get("rotation_axis_world"),
                    )
                    boxes.append((f"{name}_leaf_{index}_{collider_index}", mn, mx))
            for index, companion in enumerate(ctrl.get("rigid_companions", ())):
                path = str(companion["path"])
                rows = [
                    row for row in static_boxes if _path_is_in_subtree(row[0], path)
                ]
                sources = [(row[1], row[2]) for row in rows] or [companion["aabb"]]
                for collider_index, source in enumerate(sources):
                    mn, mx = _rotated_leaf_box(
                        source,
                        companion["hinge"],
                        _door_leaf_angle(ctrl, companion["sign"], value),
                        axis=ctrl.get("rotation_axis_world"),
                    )
                    boxes.append(
                        (f"{name}_companion_{index}_{collider_index}", mn, mx)
                    )
            continue
        rows = _controller_owner_rows(ctrl, static_boxes)
        sources = [(row[1], row[2]) for row in rows]
        if not sources and ctrl.get("moving_aabb") is not None:
            sources = [ctrl["moving_aabb"]]
        if not sources:
            continue
        t = float(ctrl.get("cur_disp", 0.0))
        axis = np.asarray(ctrl["world_axis"], dtype=np.float64)
        offset = axis * t
        for index, (closed_minimum, closed_maximum) in enumerate(sources):
            boxes.append(
                (
                    f"{name}_live_{index}",
                    np.asarray(closed_minimum, dtype=np.float64) + offset,
                    np.asarray(closed_maximum, dtype=np.float64) + offset,
                )
            )
    return boxes


def _body_nudge_boxes(scene, boxes):
    """Use box nudging only when no precise navigation plan owns body clearance.

    Task scenes with ``precise_nav`` already route on collision-mesh occupancy and validate
    manipulation stands against the exact collision oracle. Re-applying an AABB-based whole-body
    translation can push a valid route into neighbouring geometry, especially around dense cabinets.
    The wall-SDF backstop remains active through ``nudge_state``.
    """
    if (
        getattr(scene, "precise_nav", False)
        and getattr(scene, "nav_grid", None) is not None
    ):
        return []
    return boxes


def _drive_drawers(controllers, fr) -> bool:
    """If the Frame carries ("rest", pt) NEAR an openable handle, slide that drawer to match and return
    True (this frame belongs to the openable, NOT the graspable). Else return False (a graspable rest /
    carry / hide frame -- so /World/Obj should be driven instead, not left on the drawer handle).
    """
    state = _current_openable_state(controllers)
    state, consumed = _step_openable_state(controllers, fr, state)
    if state != _current_openable_state(controllers):
        _apply_openable_state(controllers, state)
    return consumed


def _initial_openable_state(controllers) -> dict[str, float]:
    return {
        str(name): float(
            ctrl.get(
                "initial_angle" if ctrl.get("kind") == "revolute" else "initial_disp",
                0.0,
            )
        )
        for name, ctrl in (controllers or {}).items()
    }


def _current_openable_state(controllers) -> dict[str, float]:
    return {
        str(name): float(
            ctrl.get("cur_angle", 0.0)
            if ctrl.get("kind") == "revolute"
            else ctrl.get("cur_disp", 0.0)
        )
        for name, ctrl in (controllers or {}).items()
    }


def _controllers_at_openable_state(controllers, state: dict[str, float]) -> dict:
    """Shallow controller view with replayed scalar state and no USD writes."""
    replay = {}
    for name, ctrl in (controllers or {}).items():
        value = float(state.get(str(name), 0.0))
        copy_ctrl = dict(ctrl)
        if ctrl.get("kind") == "revolute":
            copy_ctrl["cur_angle"] = value
        else:
            copy_ctrl["cur_disp"] = value
        replay[name] = copy_ctrl
    return replay


def _step_openable_state(controllers, fr, state: dict[str, float]):
    """Pure chronological openable-state update for one motion frame."""
    next_state = dict(state)
    segment = str(getattr(fr, "seg", ""))
    if segment.startswith(("front_insert", "front_extract")):
        return next_state, False
    kind, value = fr.obj
    openable_key = getattr(fr, "openable_key", None)
    if not controllers or openable_key is None:
        return next_state, False
    if openable_key not in controllers:
        raise RuntimeError(f"frame references unknown openable: {openable_key}")
    ctrl = controllers[openable_key]
    consumed = kind == "rest"
    if consumed:
        point = wpp.object_rest_position(value)
    elif kind == "carry":
        contact = getattr(fr, "contact_anchor", None)
        if contact is not None:
            point = np.asarray(contact, dtype=np.float64)
        elif getattr(fr, "gaze", None) is not None:
            point = np.asarray(fr.gaze, dtype=np.float64)
        else:
            return next_state, False
    else:
        return next_state, False
    if ctrl.get("kind") == "revolute":
        matched, angle = _door_hit(ctrl, point)
        if not matched:
            raise RuntimeError(
                f"openable frame is outside declared door arc: {openable_key}"
            )
        next_state[str(openable_key)] = float(angle)
        return next_state, consumed
    displacement = float(np.dot(point - ctrl["closed_handle"], ctrl["world_axis"]))
    next_state[str(openable_key)] = max(0.0, min(float(ctrl["travel"]), displacement))
    return next_state, consumed


def _openable_state_after_frames(controllers, frames) -> dict[str, float]:
    """Replay frame-authored controls without touching USD and return the final state."""
    state = _initial_openable_state(controllers)
    for frame in frames:
        state, _consumed = _step_openable_state(controllers, frame, state)
    return state


def _apply_openable_state(controllers, state: dict[str, float]) -> None:
    """Author one replayed openable state to USD/PhysX and controller metadata."""
    for name, ctrl in (controllers or {}).items():
        value = float(state.get(str(name), 0.0))
        if ctrl.get("kind") == "revolute":
            if ctrl.get("angle_set") is not None:
                ctrl["angle_set"](math.degrees(value))
            else:
                ctrl["rot"].Set(math.degrees(value))
            ctrl["cur_angle"] = value
            for extra_rot, extra_sign in ctrl.get("extra_leaves", ()):
                extra_rot.Set(math.degrees(_door_leaf_angle(ctrl, extra_sign, value)))
            for companion in ctrl.get("rigid_companions", ()):
                companion["rot"].Set(
                    math.degrees(
                        _door_leaf_angle(ctrl, companion["sign"], value)
                    )
                )
        else:
            ctrl["joint_set"](value)
            ctrl["cur_disp"] = value


def sync_openable_motion_state(controllers, frames) -> bool:
    """Apply the final openable state authored by newly executed motion frames."""
    if not controllers or not frames:
        return False
    state = _current_openable_state(controllers)
    for frame in frames:
        state, _consumed = _step_openable_state(controllers, frame, state)
    if state == _current_openable_state(controllers):
        return False
    _apply_openable_state(controllers, state)
    return True


def _authored_initial_room_door_names(scene) -> tuple[str, ...]:
    behavior_task = getattr(scene, "behavior_task", None) or {}
    scene_doors = tuple(
        sorted(str(name) for name in (getattr(scene, "doors", None) or {}))
    )
    if behavior_task.get("initially_open_room_doors") is True:
        return scene_doors
    requested = tuple(
        dict.fromkeys(
            str(name)
            for name in behavior_task.get("initially_open_room_door_names", ())
        )
    )
    unknown = tuple(name for name in requested if name not in scene_doors)
    if unknown:
        raise RuntimeError(
            f"authored initially-open room doors are absent from the scene: {unknown}"
        )
    return requested


def apply_authored_initial_openables(scene, controllers) -> tuple[str, ...]:
    behavior_task = getattr(scene, "behavior_task", None) or {}
    explicit = tuple(
        dict.fromkeys(str(name) for name in behavior_task.get("initially_open", ()))
    )
    room_doors = _authored_initial_room_door_names(scene)
    requested = tuple(dict.fromkeys((*explicit, *room_doors)))
    if not requested:
        return ()
    missing = tuple(name for name in requested if name not in controllers)
    if missing:
        raise RuntimeError(
            f"authored initially-open entities have no controller: {missing}"
        )
    reverse_room_doors = tuple(
        dict.fromkeys(
            str(name)
            for name in behavior_task.get(
                "initially_open_room_door_reverse_swing_names", ()
            )
        )
    )
    invalid_reverse_room_doors = tuple(
        name for name in reverse_room_doors if name not in room_doors
    )
    if invalid_reverse_room_doors:
        raise RuntimeError(
            "authored reverse-swing room doors must also be initially open: "
            f"{invalid_reverse_room_doors}"
        )
    applied = set(getattr(scene, "_authored_initial_openables_applied", ()))
    pending = tuple(name for name in requested if name not in applied)
    if pending:
        state = _initial_openable_state(controllers)
        for name in requested:
            controller = controllers[name]
            if controller.get("kind") == "revolute":
                if name in pending and name in reverse_room_doors:
                    controller["swing_sign"] = -float(
                        controller.get("swing_sign", 1.0)
                    )
                value = float(controller.get("swing_sign", 1.0)) * abs(
                    float(controller["swing_angle"])
                )
                controller["initial_angle"] = value
            else:
                value = float(controller["travel"])
                controller["initial_disp"] = value
            state[name] = value
        _apply_openable_state(controllers, state)
        scene._authored_initial_openables_applied = applied | set(pending)
    for name in requested:
        scene.declare_open(name)
    if hasattr(scene, "carve_door_open"):
        for name in room_doors:
            scene.carve_door_open(name)
    return pending


def prepare_authored_initial_room_door_navigation(scene) -> tuple[str, ...]:
    room_doors = _authored_initial_room_door_names(scene)
    if hasattr(scene, "carve_door_open"):
        for name in room_doors:
            scene.carve_door_open(name)
    return room_doors


def _snapshot_dynamic_prims(stage, paths) -> dict[str, object]:
    """Capture pose, ownership and velocity for render-time dynamic replay."""
    import dynamic_render_replay

    return dynamic_render_replay.snapshot_dynamic_prims(stage, paths or ())


def _freeze_dynamic_prims(stage, paths) -> dict[str, object]:
    """Pause free dynamics while non-visual placement settling advances PhysX."""
    import dynamic_render_replay

    return dynamic_render_replay.freeze_dynamic_prims(stage, paths or ())


def _restore_dynamic_prims(stage, saved: dict[str, object]) -> None:
    """Restore a dynamic replay start without leaking program-execution physics into frame zero."""
    import dynamic_render_replay

    dynamic_render_replay.restore_dynamic_prims(stage, saved)


# =================================================================================================
# Revolute DOORS (kinematic swing): mirror the drawer drive/nudge, but rotate a leaf about its hinge.
# =================================================================================================
def _door_leaf_angle(ctrl, leaf_sign: float, primary_angle: float) -> float:
    """Map the primary leaf's signed replay angle onto one authored leaf."""
    specs = ctrl.get("leaf_specs") or ()
    authored_primary_sign = (
        float(specs[0][2]) if specs else float(ctrl.get("swing_sign", 1.0))
    )
    if abs(authored_primary_sign) < 1e-9:
        authored_primary_sign = 1.0
    return float(primary_angle) * float(leaf_sign) / authored_primary_sign


def _door_closed_handle(free_xy, height: float):
    """Represent the door handle at the primary leaf's measured free edge."""
    return np.array(
        [float(free_xy[0]), float(free_xy[1]), float(height)], dtype=np.float64
    )


def _door_hit(ctrl, pt) -> tuple[bool, float]:
    """(matched, signed angle rad) for a revolute door controller and a world point ``pt``: matched iff
    pt is on the movable panel (between the hinge and free edge) AND within either signed swing range.
    Frames carry an explicit ``openable_key``, so replay must preserve the direction encoded by each frame
    even if a later action chooses the opposite swing; the radial guards still reject unrelated points.
    """
    axis = ctrl.get("rotation_axis_world")
    if axis is not None:
        hinge = np.asarray(ctrl["hinge_xyz"], dtype=np.float64)
        unit = np.asarray(axis, dtype=np.float64)
        unit /= np.linalg.norm(unit) + 1e-9
        v0 = np.asarray(ctrl["closed_handle"], dtype=np.float64) - hinge
        v1 = np.asarray(pt, dtype=np.float64) - hinge
        v0 -= unit * np.dot(v0, unit)
        v1 -= unit * np.dot(v1, unit)
        r = float(np.linalg.norm(v1))
        min_radius = min(0.15, max(0.05, float(ctrl["panel_w"]) * 0.15))
        if r < min_radius or r > ctrl["panel_w"] + 0.25:
            return False, 0.0
        ang = math.atan2(float(np.dot(unit, np.cross(v0, v1))), float(np.dot(v0, v1)))
        limit = abs(float(ctrl["swing_angle"]))
        if ang < -limit - 0.2 or ang > limit + 0.2:
            return False, 0.0
        return True, max(-limit, min(limit, ang))
    hinge = ctrl["hinge_xy"]
    v1 = np.asarray(pt, dtype=np.float64)[:2] - hinge
    r = float(np.linalg.norm(v1))
    min_radius = min(0.15, max(0.05, float(ctrl["panel_w"]) * 0.15))
    if r < min_radius or r > ctrl["panel_w"] + 0.25:
        return False, 0.0
    v0 = ctrl["closed_handle"][:2] - hinge
    ang = math.atan2(
        float(v0[0] * v1[1] - v0[1] * v1[0]), float(v0[0] * v1[0] + v0[1] * v1[1])
    )
    limit = abs(float(ctrl["swing_angle"]))
    if ang < -limit - 0.2 or ang > limit + 0.2:
        return False, 0.0
    return True, max(-limit, min(limit, ang))


def _door_angle(ctrl, fr) -> float:
    """Signed swing angle from a Frame's ("rest", arc_pt); 0 if the frame is not this door's arc."""
    kind, val = fr.obj
    if kind != "rest":
        return 0.0
    ok, ang = _door_hit(ctrl, val)
    return ang if ok else 0.0


def _door_box(ctrl, ang: float):
    """Leaf world-AABB footprint rotated about the hinge by ``ang`` (the closed leaf AABB corners swept
    through the swing) -> (mn, mx). ang=0 gives the closed footprint (spans the opening).
    """
    return _rotated_leaf_box(
        ctrl["leaf_aabb"],
        ctrl.get("hinge_xyz", ctrl["hinge_xy"]),
        ang,
        axis=ctrl.get("rotation_axis_world"),
    )


def _world_xy_to_parent_local(leaf, hx, hy, z_ref):
    """A world hinge point (hx,hy) -> the swinging leaf's PARENT-local xy, so a prepended
    ``T(+H)*Rz*T(-H)`` pivots the leaf about the TRUE world-VERTICAL hinge axis whatever world pose the
    door/container ROOT carries. Writing world coords straight into local ops was the 'door swings about a
    displaced point / flies off' bug; and a baked leaf's origin is not reliably the hinge (often the panel
    CENTRE -> swinging about its middle). Upright furniture => the root's rotation is yaw only => local Z =
    world up and the hinge xy is z-invariant, so this one transform makes EVERY revolute leaf (fridge/
    cabinet/oven leaf AND a plain nav door) swing about its real edge. General; no per-scene constants.
    """
    from pxr import Gf, UsdGeom

    p_w2l = UsdGeom.XformCache().GetLocalToWorldTransform(leaf.GetParent()).GetInverse()
    hl = p_w2l.Transform(Gf.Vec3d(float(hx), float(hy), float(z_ref)))
    return float(hl[0]), float(hl[1])


_HINGE_OP_NAMES = (
    "xformOp:translate:hingeTo",
    "xformOp:rotateZ:doorswing",
    "xformOp:translate:hingeBack",
)


def _wire_hinge_ops(xf, hlx: float, hly: float):
    """Wire a leaf's hinge pivot, tolerating a second call on the same leaf.

    `AddXformOp` RAISES when the op is already in `xformOpOrder`. A glazed cabinet is
    authored as sibling `door<N>` + `glass<N>` where only the door carries the joint, so
    the pane is wired once as itself and again riding its frame -- which aborted the whole
    episode in `setup_episode`, leaving no manifest and no `eval.json`.
    """
    from pxr import (  # this module imports pxr per-function; there is no module-level Gf
        Gf,
    )

    prior = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    others = [
        op for op in xf.GetOrderedXformOps() if op.GetOpName() not in _HINGE_OP_NAMES
    ]
    t_to = prior.get(_HINGE_OP_NAMES[0]) or xf.AddTranslateOp(opSuffix="hingeTo")
    t_to.Set(Gf.Vec3d(hlx, hly, 0.0))
    rot = prior.get(_HINGE_OP_NAMES[1]) or xf.AddRotateZOp(opSuffix="doorswing")
    rot.Set(0.0)
    t_back = prior.get(_HINGE_OP_NAMES[2]) or xf.AddTranslateOp(opSuffix="hingeBack")
    t_back.Set(Gf.Vec3d(-hlx, -hly, 0.0))
    # pivot about the TRUE world hinge edge (parent-local)
    xf.SetXformOpOrder([t_to, rot, t_back] + others)
    return t_to, rot, t_back


_DOOR_RIGID_COMPANION_TOKENS = ("handle", "knob", "pull", "grip")
_DOOR_COMPANION_PLANE_GAP_M = 0.18
_DOOR_COMPANION_EDGE_GAP_M = 0.10


def _aabb_interval_gap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> float:
    return max(float(lo_a) - float(hi_b), float(lo_b) - float(hi_a), 0.0)


def _door_rigid_companion_leaf_index(name: str, candidate_aabb, leaf_aabbs) -> int | None:
    """Match a named hardware prim to the closed leaf whose face it touches.

    Handles are occasionally authored as siblings of the articulated leaf. Name alone is unsafe in a
    cabinet containing drawer pulls, while proximity alone could capture trim. Requiring both keeps the
    fallback narrow; hardware already below a leaf is excluded by the USD traversal caller.
    """
    lowered = str(name).lower()
    if not any(token in lowered for token in _DOOR_RIGID_COMPANION_TOKENS):
        return None
    candidate_min, candidate_max = (
        np.asarray(value, dtype=np.float64) for value in candidate_aabb
    )
    if (
        candidate_min.shape != (3,)
        or candidate_max.shape != (3,)
        or not np.all(np.isfinite(candidate_min))
        or not np.all(np.isfinite(candidate_max))
        or np.any(candidate_max <= candidate_min)
    ):
        return None
    candidate_extent = candidate_max - candidate_min
    candidate_center = 0.5 * (candidate_min + candidate_max)
    matches = []
    for index, leaf_aabb in enumerate(leaf_aabbs):
        leaf_min, leaf_max = (
            np.asarray(value, dtype=np.float64) for value in leaf_aabb
        )
        leaf_extent = leaf_max - leaf_min
        if np.any(leaf_extent <= 1e-6):
            continue
        normal_axis = int(np.argmin(leaf_extent[:2]))
        lateral_axis = 1 - normal_axis
        plane_gap = _aabb_interval_gap(
            candidate_min[normal_axis],
            candidate_max[normal_axis],
            leaf_min[normal_axis],
            leaf_max[normal_axis],
        )
        lateral_gap = _aabb_interval_gap(
            candidate_min[lateral_axis],
            candidate_max[lateral_axis],
            leaf_min[lateral_axis],
            leaf_max[lateral_axis],
        )
        vertical_gap = _aabb_interval_gap(
            candidate_min[2], candidate_max[2], leaf_min[2], leaf_max[2]
        )
        leaf_face_area = leaf_extent[lateral_axis] * leaf_extent[2]
        candidate_face_area = candidate_extent[lateral_axis] * candidate_extent[2]
        if (
            plane_gap > _DOOR_COMPANION_PLANE_GAP_M
            or lateral_gap > _DOOR_COMPANION_EDGE_GAP_M
            or vertical_gap > _DOOR_COMPANION_EDGE_GAP_M
            or candidate_face_area > 0.40 * leaf_face_area
            or candidate_extent[normal_axis]
            > max(0.25, 6.0 * leaf_extent[normal_axis])
        ):
            continue
        leaf_center = 0.5 * (leaf_min + leaf_max)
        score = (
            4.0 * plane_gap
            + 2.0 * lateral_gap
            + 2.0 * vertical_gap
            + abs(candidate_center[lateral_axis] - leaf_center[lateral_axis])
            / max(leaf_extent[lateral_axis], 1e-6)
            + abs(candidate_center[2] - leaf_center[2])
            / max(leaf_extent[2], 1e-6)
        )
        matches.append((score, index))
    return min(matches)[1] if matches else None


def _door_rigid_companion_prims(droot, leaf_setups):
    """Return sibling handle-like prims paired with the leaf they must rigidly follow."""
    import fpv_scene
    from pxr import Usd, UsdGeom

    leaf_paths = tuple(str(setup["leaf"].GetPath()).rstrip("/") for setup in leaf_setups)
    leaf_aabbs = tuple(setup["aabb"] for setup in leaf_setups)
    root_path = str(droot.GetPath()).rstrip("/")
    selected_paths: list[str] = []
    companions = []
    for prim in Usd.PrimRange(droot):
        path = str(prim.GetPath()).rstrip("/")
        relative_path = path[len(root_path) :].lower()
        if not path or not any(
            token in relative_path for token in _DOOR_RIGID_COMPANION_TOKENS
        ):
            continue
        if any(
            _path_is_in_subtree(path, leaf_path)
            or _path_is_in_subtree(leaf_path, path)
            for leaf_path in leaf_paths
        ):
            continue
        if any(_path_is_in_subtree(path, selected) for selected in selected_paths):
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable or not xformable.GetPrim().IsValid():
            continue
        try:
            bounds = fpv_scene.compute_world_bbox(prim)
            minimum, maximum = bounds.GetMin(), bounds.GetMax()
            aabb = (
                np.array(minimum, dtype=np.float64),
                np.array(maximum, dtype=np.float64),
            )
        except Exception:  # noqa: BLE001 - a non-imageable named scope is not hardware
            continue
        leaf_index = _door_rigid_companion_leaf_index(path, aabb, leaf_aabbs)
        if leaf_index is None:
            continue
        selected_paths.append(path)
        companions.append((prim, leaf_index, aabb))
    return tuple(companions)


def _wire_door_rigid_companions(stage, droot, leaf_setups, name: str):
    """Give sibling door hardware the exact hinge transform of its owning leaf."""
    from pxr import UsdGeom

    wired = []
    for prim, leaf_index, aabb in _door_rigid_companion_prims(droot, leaf_setups):
        xformable = UsdGeom.Xformable(prim)
        if not xformable or not xformable.GetPrim().IsValid():
            continue
        leaf = leaf_setups[leaf_index]
        hinge = np.asarray(leaf["H"], dtype=np.float64)
        z_ref = 0.5 * (float(aabb[0][2]) + float(aabb[1][2]))
        hlx, hly = _world_xy_to_parent_local(prim, hinge[0], hinge[1], z_ref)
        _t_to, rot, _t_back = _wire_hinge_ops(xformable, hlx, hly)
        try:
            import physics_human as _ph

            if hasattr(_ph, "_apply_kinematic_collider"):
                _ph._apply_kinematic_collider(stage, str(prim.GetPath()))
        except Exception:  # noqa: BLE001
            pass
        wired.append(
            {
                "path": str(prim.GetPath()),
                "rot": rot,
                "sign": float(leaf["sign"]),
                "hinge": hinge,
                "aabb": aabb,
            }
        )
        logger.info(
            "[door] %s: rigid companion %s follows leaf %s",
            name,
            prim.GetName(),
            leaf["leaf"].GetName(),
        )
    return tuple(wired)


_LID_HINGE_OP_NAMES = (
    "xformOp:translate:lidHingeTo",
    "xformOp:orient:lidswing",
    "xformOp:translate:lidHingeBack",
)


def _world_hinge_axis_to_parent_local(leaf, hinge_xyz, axis_world):
    """Convert a world hinge point and direction into the leaf parent's coordinates."""
    from pxr import Gf, UsdGeom

    world_to_parent = (
        UsdGeom.XformCache().GetLocalToWorldTransform(leaf.GetParent()).GetInverse()
    )
    hinge_world = Gf.Vec3d(*(float(value) for value in hinge_xyz))
    axis_tip_world = Gf.Vec3d(
        *(float(hinge_xyz[i]) + float(axis_world[i]) for i in range(3))
    )
    hinge_local = world_to_parent.Transform(hinge_world)
    axis_tip_local = world_to_parent.Transform(axis_tip_world)
    axis_local = axis_tip_local - hinge_local
    norm = math.sqrt(sum(float(axis_local[i]) ** 2 for i in range(3)))
    if norm <= 1e-9:
        raise RuntimeError("lid hinge axis collapsed under the parent transform")
    return (
        tuple(float(hinge_local[i]) for i in range(3)),
        tuple(float(axis_local[i]) / norm for i in range(3)),
    )


def _wire_lid_hinge_ops(xf, hinge_local, axis_local):
    """Wire a pivoted arbitrary-axis orient op and return its degree setter."""
    from pxr import Gf, UsdGeom

    prior = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    others = [
        op
        for op in xf.GetOrderedXformOps()
        if op.GetOpName() not in _LID_HINGE_OP_NAMES
    ]
    t_to = prior.get(_LID_HINGE_OP_NAMES[0]) or xf.AddTranslateOp(opSuffix="lidHingeTo")
    t_to.Set(Gf.Vec3d(*hinge_local))
    orient = prior.get(_LID_HINGE_OP_NAMES[1]) or xf.AddOrientOp(
        UsdGeom.XformOp.PrecisionDouble, opSuffix="lidswing"
    )
    t_back = prior.get(_LID_HINGE_OP_NAMES[2]) or xf.AddTranslateOp(
        opSuffix="lidHingeBack"
    )
    t_back.Set(Gf.Vec3d(*(-float(value) for value in hinge_local)))
    xf.SetXformOpOrder([t_to, orient, t_back] + others)

    def _set(degrees: float) -> None:
        radians = math.radians(float(degrees))
        sine = math.sin(0.5 * radians)
        orient.Set(
            Gf.Quatd(
                math.cos(0.5 * radians),
                Gf.Vec3d(*(float(value) * sine for value in axis_local)),
            )
        )

    _set(0.0)
    return orient, _set


def _place_container_lid(stage, scene, e, name, droot):
    """Build a 3-D hinge controller for a top lid or front flip-down panel."""
    import fpv_scene
    from pxr import UsdGeom

    leaf = _find_by_name(droot, e.leaf_paths[0].rstrip("/").split("/")[-1])
    if leaf is None:
        return None
    rng = fpv_scene.compute_world_bbox(leaf)
    mn, mx = rng.GetMin(), rng.GetMax()
    aabb = (
        np.array([mn[0], mn[1], mn[2]], dtype=np.float64),
        np.array([mx[0], mx[1], mx[2]], dtype=np.float64),
    )
    closed_handle = np.array(
        [e.handle_xy[0], e.handle_xy[1], e.handle_z], dtype=np.float64
    )
    hinge_xyz = np.asarray((e.meta or {}).get("hinge_xyz"), dtype=np.float64)
    axis_world = np.asarray(e.hinge_axis, dtype=np.float64)
    axis_world /= np.linalg.norm(axis_world) + 1e-9
    hinge_local, axis_local = _world_hinge_axis_to_parent_local(
        leaf, hinge_xyz, axis_world
    )
    orient, angle_set = _wire_lid_hinge_ops(
        UsdGeom.Xformable(leaf), hinge_local, axis_local
    )
    try:
        import physics_human as _ph

        if hasattr(_ph, "_apply_kinematic_collider"):
            _ph._apply_kinematic_collider(stage, str(leaf.GetPath()))
    except Exception:  # noqa: BLE001
        pass
    e.hinge_xy = (float(hinge_xyz[0]), float(hinge_xyz[1]))
    e.leaf_hinges = (tuple(float(value) for value in hinge_xyz),)
    e.leaf_signs = (1.0,)
    e.leaf_bboxes = (
        (
            tuple(float(value) for value in aabb[0]),
            tuple(float(value) for value in aabb[1]),
        ),
    )
    openable_kind = str((e.meta or {}).get("openable_kind") or "lid")
    print(
        f"[{openable_kind}] {name}: hinge=({hinge_xyz[0]:.2f},{hinge_xyz[1]:.2f},{hinge_xyz[2]:.2f}) "
        f"axis=({axis_world[0]:.2f},{axis_world[1]:.2f},{axis_world[2]:.2f}) "
        f"swing={math.degrees(e.swing_angle):.0f}deg leaf={leaf.GetName()}"
    )
    return {
        "kind": "revolute",
        "openable_kind": openable_kind,
        "name": name,
        "leaf": leaf,
        "rot": orient,
        "angle_set": angle_set,
        "hinge_xy": hinge_xyz[:2],
        "hinge_xyz": hinge_xyz,
        "rotation_axis_world": axis_world,
        "closed_handle": closed_handle,
        "panel_w": float(np.linalg.norm(closed_handle - hinge_xyz)),
        "swing_angle": float(e.swing_angle),
        "swing_sign": 1.0,
        "leaf_aabb": aabb,
        "leaf_specs": ((str(leaf.GetPath()), hinge_xyz, 1.0, aabb),),
        "cur_angle": 0.0,
    }


def _place_slide_openable(stage, scene, name):
    """Wire a registered BEHAVIOR prismatic leaf into the shared openable replay."""
    import fpv_scene

    entity = scene.resolve(name)
    root = stage.GetPrimAtPath(entity.prim_path)
    if root is None or not root.IsValid():
        logger.warning("[slide] %s: invalid root %s", name, entity.prim_path)
        return None
    _deinstance(root)
    if not entity.leaf_paths:
        logger.warning("[slide] %s: no registered moving leaf", name)
        return None
    leaf_name = entity.leaf_paths[0].rstrip("/").split("/")[-1]
    leaf = stage.GetPrimAtPath(entity.leaf_paths[0])
    if leaf is None or not leaf.IsValid():
        leaf = _find_by_name(root, leaf_name)
    if leaf is None:
        logger.warning("[slide] %s: moving leaf %s not found", name, leaf_name)
        return None
    bounds = fpv_scene.compute_world_bbox(leaf)
    minimum, maximum = bounds.GetMin(), bounds.GetMax()
    handle = np.array(
        [
            entity.handle_xy[0],
            entity.handle_xy[1],
            entity.handle_z,
        ],
        dtype=np.float64,
    )
    axis = np.asarray(entity.axis, dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-9
    joint = _find_prismatic_joint(root, entity.joint_name, leaf_name)
    joint_set, driver = _make_prismatic_setter(joint, leaf, axis)
    logger.info(
        "[slide] %s: driver=%s joint=%s leaf=%s travel=%.3fm axis=%s",
        name,
        driver,
        joint.GetName() if joint is not None else entity.joint_name,
        leaf.GetName(),
        float(entity.travel),
        tuple(round(float(value), 3) for value in axis),
    )
    return {
        "kind": "prismatic",
        "name": name,
        "joint_set": joint_set,
        "world_axis": axis,
        "travel": float(entity.travel),
        "moving_paths": (str(leaf.GetPath()),),
        "moving_aabb": (
            np.array(minimum, dtype=np.float64),
            np.array(maximum, dtype=np.float64),
        ),
        "closed_handle": handle,
        "cur_disp": 0.0,
    }


def _place_door(stage, scene, name):
    """De-instance a BEHAVIOR door + prepend a local rotateZ op to its swinging leaf (leaf origin = the
    hinge, local Z = world up), resolve the swing SIGN to the free side (nav-grid wall clearance) and write
    it back to the door Entity so the _swing producer arcs the SAME way, then return a revolute controller
    for the per-frame drive/nudge. Returns None if the leaf cannot be located."""
    import fpv_scene
    from pxr import UsdGeom

    e = scene.resolve(name)
    # A nav door lives in scene.doors/door_prims; an openable CONTAINER (fridge/cabinet leaf) is a normal
    # entity with hinge fields -> its prim is e.prim_path.
    di = (scene.doors or {}).get(name)
    pp = scene.door_prims[di.prim_index] if di is not None else e.prim_path
    droot = stage.GetPrimAtPath(pp)
    _deinstance(droot)
    if not e.leaf_paths:
        return None
    if (e.meta or {}).get("openable_kind") in ("lid", "flip_down"):
        return _place_container_lid(stage, scene, e, name, droot)
    # OPENABLE CONTAINER (handle_xy set): drive EVERY revolute leaf about its OWN true hinge edge via a
    # world-pivot, opening toward the room -> a 2-door fridge/cabinet opens BOTH doors in its natural shape.
    # A plain nav door keeps the byte-identical single-leaf origin-rotate below (its leaf origin IS hinge).
    if e.handle_xy is not None:
        return _place_container_leaves(stage, scene, e, name, droot)

    leaf = _find_by_name(droot, e.leaf_paths[0].rstrip("/").split("/")[-1])
    if leaf is None:
        return None
    # A plain nav door has no handle_xy (openable CONTAINERS already returned above) -> it grips its own
    # grounded object_pos e.xy (the free edge). closed_handle IS the _swing producer's grip, so _door_hit's
    # radius/angle reference agrees with the emitted arc.
    closed_handle = np.array([e.xy[0], e.xy[1], e.z], dtype=np.float64)
    # Navigation doors retain the authored revolute-joint origin registered by behavior_import. Replacing
    # it with a visual AABB edge pulls panels away from jambs when trim or handles widen the leaf bounds.
    # Flattened appliance/container leaves take the separate geometry-derived path above.
    _lr = fpv_scene.compute_world_bbox(leaf)
    _lmn, _lmx = _lr.GetMin(), _lr.GetMax()
    hinge_xy = np.asarray(e.hinge_xy, dtype=np.float64)
    # Resolve swing_sign: open toward the side with MORE wall clearance (into a room, not into a wall).
    sign = 1.0
    grid = getattr(scene, "nav_grid", None)
    if grid is not None and getattr(grid, "wall_sdf", None) is not None:

        def _swept(s):
            th = s * float(e.swing_angle)
            c, sn = math.cos(th), math.sin(th)
            v = closed_handle[:2] - hinge_xy
            return (
                float(hinge_xy[0] + v[0] * c - v[1] * sn),
                float(hinge_xy[1] + v[0] * sn + v[1] * c),
            )

        sign = (
            1.0
            if grid.wall_sdf_at(*_swept(1.0)) >= grid.wall_sdf_at(*_swept(-1.0))
            else -1.0
        )
    e.swing_sign = float(sign)

    _hlx, _hly = _world_xy_to_parent_local(
        leaf, hinge_xy[0], hinge_xy[1], 0.5 * (float(_lmn[2]) + float(_lmx[2]))
    )
    t_to, rot, t_back = _wire_hinge_ops(UsdGeom.Xformable(leaf), _hlx, _hly)

    try:  # kinematic-rigid collider on the leaf so OBJECT physics reacts (NOT the human anti-pen path)
        import physics_human as _ph

        if hasattr(_ph, "_apply_kinematic_collider"):
            _ph._apply_kinematic_collider(stage, str(leaf.GetPath()))
    except Exception:  # noqa: BLE001
        pass

    rng = fpv_scene.compute_world_bbox(leaf)
    lmn, lmx = rng.GetMin(), rng.GetMax()
    leaf_aabb = (
        np.array([lmn[0], lmn[1], lmn[2]], dtype=np.float64),
        np.array([lmx[0], lmx[1], lmx[2]], dtype=np.float64),
    )
    rigid_companions = _wire_door_rigid_companions(
        stage,
        droot,
        (
            {
                "leaf": leaf,
                "H": hinge_xy,
                "sign": float(sign),
                "aabb": leaf_aabb,
            },
        ),
        name,
    )
    print(
        f"[door] {name}: hinge=({hinge_xy[0]:.2f},{hinge_xy[1]:.2f}) "
        f"handle=({closed_handle[0]:.2f},{closed_handle[1]:.2f}) "
        f"swing={math.degrees(e.swing_angle):.0f}deg sign={sign:+.0f} leaf={leaf.GetName()}"
    )
    return {
        "kind": "revolute",
        "name": name,
        "leaf": leaf,
        "rot": rot,
        "hinge_xy": hinge_xy,
        "closed_handle": closed_handle,
        "panel_w": float(np.linalg.norm(closed_handle[:2] - hinge_xy)),
        "swing_angle": float(e.swing_angle),
        "swing_sign": float(sign),
        "leaf_aabb": leaf_aabb,
        "leaf_specs": (
            (
                str(leaf.GetPath()),
                hinge_xy,
                float(sign),
                leaf_aabb,
            ),
        ),
        "rigid_companions": rigid_companions,
        "moving_paths": (
            str(leaf.GetPath()),
            *(companion["path"] for companion in rigid_companions),
        ),
        "leaf_only_static_names": (name,),
        "cur_angle": 0.0,
    }


def _container_authored_hinges(e):
    if (getattr(e, "meta", None) or {}).get("hinge_source") == "geometry":
        return ()
    hinges = tuple(getattr(e, "leaf_hinges", ()) or ())
    if not hinges and e.hinge_xy is not None:
        hinges = (e.hinge_xy,)
    return hinges


def _place_container_leaves(stage, scene, e, name, droot):
    """Swing controller for an openable CONTAINER (fridge/cabinet/oven...) with 1+ revolute leaves.

    The importer records each articulated link's authored joint origin before de-instancing. Use that
    origin when it lies on a panel edge; otherwise fall back to the panel-edge inference used by older
    assets whose link origin was flattened to the asset root. Every leaf rotates through a world-pivot
    ``T(H)*Rz*T(-H)*existing``. The primary leaf drives the hand arc, while extra leaves ride the same
    magnitude with their own hinge and sign."""
    import fpv_scene
    from pxr import UsdGeom

    arad = math.radians(e.approach_yaw)
    room_dir = -np.array(
        [math.sin(arad), -math.cos(arad)]
    )  # container -> room (doors open this way)
    foot = np.array(
        e.foot_xy if e.foot_xy is not None else e.xy,
        dtype=np.float64,
    )
    ang = float(e.swing_angle)
    authored_hinges = _container_authored_hinges(e)

    def _setup(leaf_path, H_override=None, sgn_override=None):
        # ``H_override`` / ``sgn_override``: pin this leaf to ANOTHER leaf's hinge and swing direction.
        # Used for a glass pane riding its own door frame -- a pane's bbox would otherwise derive its own
        # (very slightly different) hinge, and a pane that swings about a hinge 10 mm off its frame's
        # tears out of the frame over an 88 deg arc.
        leaf = _find_by_name(droot, leaf_path.rstrip("/").split("/")[-1])
        if leaf is None:
            return None
        r = fpv_scene.compute_world_bbox(leaf)
        mn, mx = r.GetMin(), r.GetMax()
        cx, cy = (
            0.5 * (float(mn[0]) + float(mx[0])),
            0.5 * (float(mn[1]) + float(mx[1])),
        )
        if (float(mx[0]) - float(mn[0])) >= (
            float(mx[1]) - float(mn[1])
        ):  # long axis X
            e0, e1 = np.array([float(mn[0]), cy]), np.array([float(mx[0]), cy])
        else:  # long axis Y
            e0, e1 = np.array([cx, float(mn[1])]), np.array([cx, float(mx[1])])
        H, free = (
            (e0, e1)
            if np.linalg.norm(e0 - foot) >= np.linalg.norm(e1 - foot)
            else (e1, e0)
        )
        hinge_source = "geometry"
        if H_override is not None:
            authored = np.asarray(H_override, dtype=np.float64)[:2]
            endpoint_distances = (
                float(np.linalg.norm(authored - e0)),
                float(np.linalg.norm(authored - e1)),
            )
            panel_width = float(np.linalg.norm(e1 - e0))
            if np.isfinite(authored).all() and min(endpoint_distances) <= max(
                0.08, 0.25 * panel_width
            ):
                H = authored
                free = e1 if endpoint_distances[0] <= endpoint_distances[1] else e0
                hinge_source = "joint"

        def _swept(s):  # free edge swept about H by s*ang
            th = s * ang
            c, sn = math.cos(th), math.sin(th)
            v = free - H
            return H + np.array([v[0] * c - v[1] * sn, v[0] * sn + v[1] * c])

        sgn = (
            1.0
            if np.dot(_swept(1.0) - H, room_dir) >= np.dot(_swept(-1.0) - H, room_dir)
            else -1.0
        )
        if sgn_override is not None:
            sgn = float(sgn_override)
        hlx, hly = _world_xy_to_parent_local(
            leaf, H[0], H[1], 0.5 * (float(mn[2]) + float(mx[2]))
        )  # world hinge -> parent-local so the pivot is world-correct (was: world coords as local ops)
        t_to, rot, t_back = _wire_hinge_ops(UsdGeom.Xformable(leaf), hlx, hly)
        try:
            import physics_human as _ph

            if hasattr(_ph, "_apply_kinematic_collider"):
                _ph._apply_kinematic_collider(stage, str(leaf.GetPath()))
        except Exception:  # noqa: BLE001
            pass
        aabb = (
            np.array([mn[0], mn[1], mn[2]], dtype=np.float64),
            np.array([mx[0], mx[1], mx[2]], dtype=np.float64),
        )
        return {
            "leaf": leaf,
            "rot": rot,
            "H": H,
            "free": free,
            "sign": float(sgn),
            "aabb": aabb,
            "hinge_source": hinge_source,
        }

    setups = []
    for index, leaf_path in enumerate(e.leaf_paths):
        hinge = authored_hinges[index] if index < len(authored_hinges) else None
        setup = _setup(leaf_path, H_override=hinge)
        if setup is not None:
            setups.append(setup)
    articulated_setups = tuple(setups)
    # THE PANE RIDES ITS FRAME. OG models a glazed cabinet as SIBLING links ``door<N>`` + ``glass<N>`` and
    # gives a revolute joint to ``door<N>`` ONLY (read straight off the spec: rmmvcd has j_door1..j_door4
    # and ZERO joints for glass1..glass4; the fume hood has no joints at all). ``leaf_paths`` comes from
    # those joints, so opening the unit swung four empty FRAMES and left four panes hanging across the
    # opening -- a glass wall the placement engine, which only ever rays DOWNWARD for support, cannot see.
    # Measured in bt_chemlab: the three reagents put into the fume hood are the three that failed, and the
    # depth they were jammed into a pane predicted how far they were ejected (74 mm -> 764 mm, 19 mm ->
    # 106 mm), along -X, perpendicular to the pane, exactly as a de-penetration impulse must be.
    # Matched by NAME/index so a pane can only ever ride its own frame; a unit with no glass sibling (every
    # other scene: plain doors, `left_door`, `reddoor1`, `leaf`) adds nothing and is byte-identical.
    for _s in list(setups):
        _nm = _s["leaf"].GetName()
        if not _nm.startswith("door"):
            continue
        _idx = _nm[4:]
        if _idx and not _idx.isdigit():
            continue
        _pane = _find_by_name(droot, f"glass{_idx}")
        if _pane is None:
            continue
        _ps = _setup(str(_pane.GetPath()), H_override=_s["H"], sgn_override=_s["sign"])
        if _ps is not None:
            setups.append(_ps)
            print(f"[door] {name}: pane {_pane.GetName()} now rides frame {_nm}")
    if not setups:
        return None
    rigid_companions = _wire_door_rigid_companions(
        stage, droot, articulated_setups, name
    )
    prim = setups[0]  # primary: drives the hand arc + _door_hit
    closed_handle = _door_closed_handle(
        prim["free"], e.handle_z if e.handle_z is not None else e.z
    )
    e.hinge_xy = (float(prim["H"][0]), float(prim["H"][1]))
    e.handle_xy = (float(closed_handle[0]), float(closed_handle[1]))
    e.swing_sign = prim["sign"]
    # Per-leaf hinge/sign/closed-AABB for the swung-OPEN footprint (nav mark + PutInside reach): a baked
    # leaf's prim origin is NOT its hinge, so the correct swing is about THIS render-derived H by sign*ang.
    e.leaf_hinges = tuple((float(s["H"][0]), float(s["H"][1])) for s in setups)
    e.leaf_signs = tuple(float(s["sign"]) for s in setups)
    e.leaf_bboxes = tuple(
        (
            (float(s["aabb"][0][0]), float(s["aabb"][0][1]), float(s["aabb"][0][2])),
            (float(s["aabb"][1][0]), float(s["aabb"][1][1]), float(s["aabb"][1][2])),
        )
        for s in setups
    )
    print(
        f"[door] {name}: {len(setups)} leaf/leaves primary hinge=({prim['H'][0]:.2f},{prim['H'][1]:.2f}) "
        f"handle=({closed_handle[0]:.2f},{closed_handle[1]:.2f}) swing={math.degrees(ang):.0f}deg "
        f"signs={[round(s['sign']) for s in setups]} hinge_source={prim['hinge_source']}"
    )
    return {
        "kind": "revolute",
        "name": name,
        "leaf": prim["leaf"],
        "rot": prim["rot"],
        "hinge_xy": prim["H"],
        "closed_handle": closed_handle,
        "panel_w": float(np.linalg.norm(closed_handle[:2] - prim["H"])),
        "swing_angle": ang,
        "swing_sign": prim["sign"],
        "leaf_aabb": prim["aabb"],
        "leaf_specs": tuple(
            (
                str(setup["leaf"].GetPath()),
                setup["H"],
                setup["sign"],
                setup["aabb"],
            )
            for setup in setups
        ),
        "extra_leaves": [(s["rot"], s["sign"]) for s in setups[1:]],
        "rigid_companions": rigid_companions,
        "moving_paths": (
            *(str(setup["leaf"].GetPath()) for setup in setups),
            *(companion["path"] for companion in rigid_companions),
        ),
        "cur_angle": 0.0,
    }


def _check_clutter_geometry(stage, scene, clutter_rest_xy) -> tuple[bool, bool]:
    """CLUTTER-ONLY offline asserts for the D3 fixes (returns (stand_ok, margin_ok); (True, True) for
    every other scene, which has no ``clutter_table``):
      A) the human's CORE stands clearly IN FRONT (+Y) of the tabletop near edge by >= 0.35 m -- the D3
         BLOCKER was the body standing UNDER the thin plank when the stand-back was too small.
      B) every settled clutter XY is inside the tabletop MINUS a 5 cm margin, so nothing rests
         cantilevered over an edge (the D3 can-off-the-edge fail). A per-prim UPRIGHT tilt is logged as
         a diagnostic (the can/boxes must not have toppled onto their side)."""
    ct = getattr(scene, "clutter_table", None)
    if ct is None:
        return True, True
    (stand_x, stand_y), _yaw = scene.stand_pose_for("table")
    front_margin = stand_y - ct["near_edge_y"]  # +Y => in front of the near edge
    stand_ok = front_margin >= 0.35
    print(
        f"[step3r][STANDBACK] stand=({stand_x:.3f},{stand_y:.3f}) "
        f"near_edge_y={ct['near_edge_y']:.3f} -> pelvis {front_margin * 1000:.0f}mm IN FRONT "
        f"(need >=350mm) {'OK' if stand_ok else 'FAIL'}"
    )
    inner = 0.05  # settled CoM must be this far inside every tabletop edge
    lo_x, hi_x = ct["cx"] - ct["half_x"] + inner, ct["cx"] + ct["half_x"] - inner
    lo_y, hi_y = ct["cy"] - ct["half_y"] + inner, ct["cy"] + ct["half_y"] - inner
    margin_ok = True
    for cpath, (rx, ry) in clutter_rest_xy.items():
        inside = (lo_x <= rx <= hi_x) and (lo_y <= ry <= hi_y)
        margin_ok = margin_ok and inside
        m = np.array(_read_prim_l2w(stage, cpath)).reshape(4, 4)
        zaxis = m[2, :3] / (np.linalg.norm(m[2, :3]) + 1e-9)
        tilt = math.degrees(math.acos(float(np.clip(zaxis[2], -1.0, 1.0))))
        print(
            f"[step3r][EDGE]   {cpath.rsplit('/', 1)[-1]}: settled XY=({rx:.3f},{ry:.3f}) "
            f"tilt={tilt:.0f}deg in tabletop-minus-{inner * 100:.0f}cm X[{lo_x:.2f},{hi_x:.2f}] "
            f"Y[{lo_y:.2f},{hi_y:.2f}] {'OK' if inside else 'FAIL (cantilevered)'}"
        )
    return stand_ok, margin_ok


def _collision_gate(stage, rig, frames, floor_z, solid_box_paths, app, sample_every=6):
    """OFFLINE geometric penetration PRE-FILTER / regression assert: body mesh vertices vs the SOLID
    procedural boxes (pillar / cabinet bases) + lowest-vertex-vs-floor, over the MOTION frames.
    Deterministic + reports real depth in mm. DEMOTED (P1): this AABB test is now a cheap pre-filter /
    regression guard on the rendered motion; the AUTHORITATIVE non-penetration gate for INIT placement
    is the depth-bearing, kinematic-static PhysX query ``physics_human.human_scene_separation`` (run by
    ``--initcheck``), which sees thin-top/concave/referenced-mesh geometry this AABB cannot. (The Sektion
    cabinet is not a solid box -- arm-vs-cabinet is judged from the render; here we verify the solid
    obstacles + the floor.) DETECT ONLY. Returns (n_vert_hits, n_floor_frames, worst_floor_m).
    """
    import fpv_scene

    boxes = []
    for p in solid_box_paths:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            rng = fpv_scene.compute_world_bbox(prim)
            mn, mx = rng.GetMin(), rng.GetMax()
            boxes.append(
                (p, np.array([mn[0], mn[1], mn[2]]), np.array([mx[0], mx[1], mx[2]]))
            )

    idxs = sorted(set(range(0, len(frames), sample_every)) | {len(frames) - 1})
    n_pen, worst_pen, n_floor, worst_floor = 0, 0.0, 0, 0.0
    for i in idxs:
        V, _J, _ = wpp.realize(rig, frames[i])
        for name, mn, mx in boxes:
            inside = np.all((V > mn) & (V < mx), axis=1)
            k = int(inside.sum())
            if k:
                vin = V[inside]
                depth = float(np.minimum(vin - mn, mx - vin).min(axis=1).max())
                n_pen += k
                worst_pen = max(worst_pen, depth)
                fr = frames[i]
                print(
                    f"[step3r][GATE]   frame {i}/{len(frames)} seg={fr.seg}: "
                    f"{k} vert(s) in {name} (depth {depth * 1000:.0f}mm) "
                    f"root={tuple(round(v, 2) for v in fr.xy)} "
                    f"vinY=[{vin[:, 1].min():.2f},{vin[:, 1].max():.2f}] "
                    f"vinZ=[{vin[:, 2].min():.2f},{vin[:, 2].max():.2f}]"
                )
        d = floor_z - float(V[:, 2].min())
        if d > 0.005:
            n_floor += 1
            worst_floor = max(worst_floor, d)
    print(
        f"[step3r][GATE] {len(idxs)} frames: {n_pen} body-vert-in-solid-box hit(s) (worst "
        f"{worst_pen * 1000:.0f}mm), {n_floor} feet-below-floor frame(s) (worst {worst_floor * 1000:.0f}mm)"
    )
    return n_pen, worst_pen, n_floor, worst_floor


def _prenudge_diagnostic(rig, frames, scene):
    """PRE-NUDGE planner telemetry: grade the RAW motion, the pose BEFORE the runtime nudge.

    REPORT ONLY -- this is deliberately NOT a gate. The renderer never draws this pose, so a verdict
    taken from it is a verdict about a video that does not exist; that is exactly what made ``--check``
    disagree with the render path. What it IS good for is showing how much work the nudge is doing: a
    raw grade far worse than the as-rendered one means the planner is emitting poses the nudge has to
    rescue, which is worth fixing at the source.

    Runs the SAME ``action_check.grade_trajectory`` sweep + cut points as the verdict, on the SAME
    ``scene.cworld`` oracle, so the two numbers are directly comparable and differ only by the nudge.
    (It replaces a second, independent gate that queried a different obstacle set via PhysX and reported
    "clean" whenever that set failed to build.) Returns the grade record, or None when the oracle is
    absent -- an absent measurement, never a clean one."""
    import action_check

    cworld = getattr(scene, "cworld", None)
    if cworld is None or not frames:
        print("[step3r][RAW] no collision oracle; pre-nudge diagnostic unmeasured")
        return None
    # Pose ONLY the frames the sweep reads (its own schedule): posing all of them costs ~40s on a
    # 229-frame episode, and --check is the fast loop. The raw realize carries no nudge state, so a
    # sparse pass is identical to a full one at these indices.
    stride = action_check.DEFAULT_SAMPLE_EVERY
    raw_traj: list = [None] * len(frames)
    for _i in action_check.sample_indices(len(frames), stride):
        raw_traj[_i] = wpp.realize(rig, frames[_i])[1]
    rec = action_check.grade_trajectory(
        cworld, raw_traj, frames, sample_every=stride, fps=FPS
    )
    print(
        f"[step3r][RAW] pre-nudge (NOT a gate): grade={rec['render_grade']} "
        f"worst_frac={rec['human_worst_frac']:.3f} [{rec['human_worst_frac_desc']}] "
        f"hit_frames={rec['human_hit_frames']}/{rec['human_sampled']} "
        f"max_pen={rec['human_max_pen_mm']:.0f}mm segs={rec['human_by_seg']}"
    )
    return rec


def _ensure_cworld(scene, stage, ph):
    """Build the SSOT collision oracle ONCE per episode (gated) and cache it on ``scene.cworld``.

    The oracle (``collision_world.CollisionWorld``) is the ONE precise-geometry map that the settle,
    the reach/nudge boxes, and the post-action validator all read, so they can no longer disagree.
    Built only when the scene opts in (``scene.unified_geom`` / ``scene.post_action_check`` = the
    BEHAVIOR render path); every other scene keeps ``scene.cworld`` None and each consumer falls back
    to its legacy geometry -> byte-identical. Safe to call repeatedly (cached)."""
    if getattr(scene, "cworld", None) is not None:
        return scene.cworld
    if not (
        getattr(scene, "unified_geom", False)
        or getattr(scene, "post_action_check", False)
    ):
        return None
    try:
        import collision_world

        scene.cworld = collision_world.CollisionWorld.build(stage, scene, ph)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[step3r] collision_world build failed ({exc}); unified_geom OFF this episode"
        )
        scene.cworld = None
    return scene.cworld


def engine_code_id() -> str:
    """A fingerprint of the ENGINE SOURCE that actually produced this episode.

    The manifest has always carried a ``code_id`` slot, but it read ``os.environ["CODE_ID"]`` and
    nothing ever set it, so every manifest on disk records ``null``. That single null is what makes
    run-to-run comparison unsound: two manifests for the same spec and seed are indistinguishable even
    when the engine changed underneath them, and a frame-count difference caused by an edit is then
    indistinguishable from physics noise. Several agents edit these modules concurrently, so a run can
    even import a file mid-edit.

    Hashes the source of every already-imported module under this directory -- i.e. exactly the code
    this process is running, not a VCS revision that says nothing about uncommitted edits.
    """
    import hashlib

    h = hashlib.sha256()
    for name, mod in sorted(sys.modules.items()):
        path = getattr(mod, "__file__", None) or ""
        if not path.startswith(_THIS_DIR) or not path.endswith(".py"):
            continue
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            continue
        h.update(name.encode())
        h.update(hashlib.sha256(body).digest())
    return h.hexdigest()[:16]


def _resolve_container_drops(stage, scene) -> None:
    """Set each CONTAINER's release height ``drop_z`` to its OWN true interior RIM (world-bbox top), so
    produce_place releases the object just inside the opening and the post-action PHYSICS settles it to its
    real rest (bin bottom, or the shelf under the drop xy). This REPLACES the earlier PhysX interior
    shelf-raycast: the ONE precise geometry (the container's own bbox) is the sole source, and the release
    point stays DECOUPLED from the containment band -- ``_relations_inside_live`` reads the container's true
    bbox independently, so the release retarget can never corrupt "is it inside" (the bin-bottom false-miss).
    General: derived from the container's live bbox, no second geometry query, no magic numbers. Gated on the
    unified/BEHAVIOR path (no container -> no-op). Front-opening openable containers (handle_xy set) are
    SKIPPED -- they place onto an interior shelf via their own drop_z, not this outer rim (their top is a
    closed lid; that is the deferred multi-height 'inside' case)."""
    if not (
        getattr(scene, "unified_geom", False)
        or getattr(scene, "behavior_task", None)
        or getattr(scene, "behavior_grab", None)
    ):
        return
    import fpv_scene
    from scene_state import Role

    scene._into_placed = (
        {}
    )  # fresh per-episode placement footprints (multi-object de-confliction)
    scene._space_assign = (
        {}
    )  # fresh per-episode PutInside space assignments (reset for the RL reset loop)
    for key, e in list(scene.entities.items()):
        if e.role is not Role.CONTAINER or not e.prim_path:
            continue
        # FRONT-OPENING openable containers (handle_xy set) open at the FRONT onto an interior shelf, not the
        # top -- the outer-bbox top is a closed lid, so a rim release would land the object ON the lid. Leave
        # their existing interior drop_z untouched (the deferred multi-height 'inside' path owns them).
        if getattr(e, "handle_xy", None) is not None:
            continue
        prim = stage.GetPrimAtPath(e.prim_path)
        if not (prim and prim.IsValid()):
            continue
        rng = fpv_scene.compute_world_bbox(prim)
        cz_top = float(rng.GetMax()[2])
        # NEVER RAISE A RELEASE POINT. This function exists to LOWER a blindly-guessed drop onto the real
        # rim; raising one above what the builder already derived can only move the release further from the
        # interior, and on a unit with a solid roof it moves it ONTO that roof. Measured on
        # ``bookcase/qpylqf``: the builder registers the interior board at drop_z=1.37 and this line used to
        # overwrite it with the world-bbox top 1.47, so the object was released 100 mm above the roof, settled
        # on it, and never read back INSIDE -- a placement that reports success into thin air, which is the
        # same silent-failure shape as the ejection guard's. It is the whole of T6-MER-01's 1.00 -> 0.00 and
        # the residue on T6-RURB-01. ``min`` keeps every case this function was written for byte-identical (a
        # top-open bin's authored drop sits ABOVE its low rim, so the rim still wins) and removes only the
        # case where the "rim" is a lid.
        prev = getattr(e, "drop_z", None)
        e.drop_z = cz_top if prev is None else min(float(prev), cz_top)
        if prev is not None and float(prev) < cz_top:
            print(
                f"[step3r] container '{key}': KEEP builder drop_z={float(prev):.3f} "
                f"(world-bbox top {cz_top:.3f} is a ROOF, not a rim -- not raising the release)"
            )
        else:
            print(
                f"[step3r] container '{key}': release rim drop_z={e.drop_z:.3f} (world-bbox top); "
                f"containment = true bbox (decoupled)"
            )


def _declared_support_path(scene, initial_place="", support_path=None) -> str | None:
    """Resolve one authored object placement to its concrete fixture prim."""
    place = str(initial_place or "")
    if place.startswith("floor:"):
        return None
    explicit = str(support_path or "").rstrip("/")
    relation, separator, support_key = place.partition(":")
    if not separator or relation not in {"on", "in"}:
        return explicit or None
    if not support_key:
        raise RuntimeError(f"declared initial placement {place!r} has no support key")
    try:
        owner = scene.resolve(support_key)
        role = Role.SURFACE if relation == "on" else Role.CONTAINER
        affordance = getattr(scene, "affordance", None)
        entity = affordance(support_key, role) if callable(affordance) else None
        entity = entity or owner
    except KeyError as error:
        raise RuntimeError(
            f"declared initial support {support_key!r} is not registered in the scene"
        ) from error
    bound = str(getattr(entity, "prim_path", None) or "").rstrip("/")
    if not bound:
        raise RuntimeError(
            f"declared initial support {support_key!r} has no concrete prim binding"
        )
    if explicit and not (
        _support_path_matches(explicit, (bound,))
        or _support_path_matches(bound, (explicit,))
    ):
        raise RuntimeError(
            f"declared initial support {support_key!r} is bound to {bound!r}, "
            f"not {explicit!r}"
        )
    return explicit or bound


def _support_top_for(
    scene,
    xy,
    ge,
    exclude_path,
    *,
    declared=False,
    initial_place="",
    support_path=None,
):
    """SSOT support top under ``xy`` (the real SUPPORT collider top, never the moved object's own top).

    A declared floor placement is pinned to ``floor_z``; a declared surface limits the query to that
    support's prim. Legacy callers without placement metadata retain the registered ``top_z``/``z``
    fallback. The moved object's own top is never treated as the floor beneath it."""
    floor_staged = declared and str(initial_place).startswith("floor:")
    if floor_staged:
        return float(scene.floor_z)
    declared_support_path = (
        _declared_support_path(scene, initial_place, support_path) if declared else None
    )
    cw = getattr(scene, "cworld", None)
    if cw is not None:
        excluded = {str(exclude_path)}
        excluded.update(
            str(entity.prim_path)
            for entity in scene.entities.values()
            if entity.role is Role.GRASPABLE and entity.prim_path
        )
        included = ()
        if declared_support_path:
            included = (declared_support_path,)
        elif declared and getattr(scene, "behavior_support_prims", None):
            included = tuple(str(path) for path in scene.behavior_support_prims)
        top_z = float(
            cw.support_top_z(
                xy,
                exclude_owners=tuple(sorted(excluded)),
                include_owners=included,
                max_z=float(ge.z) + 0.30,
            )
        )
        if declared and top_z <= cw.floor_z + 1e-6 and ge.top_z is not None:
            return float(ge.top_z)
        return top_z
    return ge.top_z if ge.top_z is not None else ge.z


def _support_path_matches(actual: str | None, expected: tuple[str, ...]) -> bool:
    if not expected:
        return True
    path = str(actual or "").rstrip("/")
    return any(
        path == root or path.startswith(root + "/")
        for root in (value.rstrip("/") for value in expected if value)
    )


def _support_path_matches_or_is_coplanar_overlay(
    actual: str | None,
    expected: tuple[str, ...],
    support_z: float,
    declared_z: float,
) -> bool:
    """Accept another collider owner only when it caps the declared plane within probe tolerance."""
    return _support_path_matches(actual, expected) or abs(
        float(support_z) - float(declared_z)
    ) <= (INITIAL_SUPPORT_PLANE_TOL_M + 1e-3)


def probe_declared_support(
    ph,
    xy: tuple[float, float],
    support_hint_z: float,
    object_half_xy: tuple[float, float],
    object_height: float,
    *,
    exclude_path: str,
    support_paths: tuple[str, ...] = (),
    min_support_fraction: float = 0.0,
):
    """Find the declared support broadly, then validate its exact live plane."""
    import object_physics as ophys

    discovery = ph.probe_drop_site(
        xy,
        float(support_hint_z),
        object_half_xy,
        float(object_height),
        exclude_prefixes=(str(exclude_path),),
        z_tol=INITIAL_SUPPORT_SEARCH_M,
    )
    support = discovery.get("support")
    if support is None:
        return ophys.SupportProbe(
            verified=False,
            ok=True,
            support_z=float(support_hint_z),
            reason=str(discovery.get("reason") or "probe-unavailable"),
        )
    hit_z = discovery.get("hit_z")
    support_z = float(hit_z) if hit_z is not None else float(support_hint_z)
    path = str(discovery.get("support_path") or "") or None
    if not support or hit_z is None:
        return ophys.SupportProbe(
            verified=True,
            ok=False,
            support_z=support_z,
            support_path=path,
            reason="no exact support near the declared plane",
        )
    if not _support_path_matches(path, support_paths):
        # A broad search can hit a different surface stacked above the declared one.
        # Probe the authored plane narrowly before deciding that its support is absent.
        support_z = float(support_hint_z)

    raw = ph.probe_drop_site(
        xy,
        support_z,
        object_half_xy,
        float(object_height),
        exclude_prefixes=(str(exclude_path),),
        z_tol=INITIAL_SUPPORT_PLANE_TOL_M,
    )
    hit_z = raw.get("seat_z")
    if hit_z is None:
        hit_z = raw.get("hit_z")
    support_z = float(hit_z) if hit_z is not None else support_z
    fraction = raw.get("support_frac")
    support_fraction = float(fraction) if fraction is not None else None
    support = raw.get("support")
    clear = raw.get("clear")
    if support is None or clear is None:
        return ophys.SupportProbe(
            verified=False,
            ok=True,
            support_z=float(support_hint_z),
            reason=str(raw.get("reason") or "probe-unavailable"),
        )
    path = str(raw.get("support_path") or "") or path
    reason = "verified"
    ok = bool(support and clear)
    if not support:
        reason = "no exact support near the declared plane"
    elif not clear:
        blockers = [str(value) for value in raw.get("blockers") or ()]
        detail = ", ".join(value.rsplit("/", 1)[-1] for value in blockers)
        reason = (
            f"verified support site is blocked by {detail}"
            if detail
            else "verified support site is blocked"
        )
    elif raw.get("embedded") is True:
        ok, reason = False, "declared support site is embedded in geometry"
    elif not _support_path_matches_or_is_coplanar_overlay(
        path, support_paths, support_z, support_hint_z
    ):
        ok, reason = False, "exact support belongs to a different object"
    elif support_fraction is not None and support_fraction + 1e-9 < float(
        min_support_fraction
    ):
        ok, reason = False, "too little of the object footprint is supported"
    return ophys.SupportProbe(
        verified=True,
        ok=ok,
        support_z=support_z,
        support_path=path,
        support_fraction=support_fraction,
        reason=reason,
    )


def _support_paths(scene, support_path=None, initial_place="") -> tuple[str, ...]:
    if str(initial_place).startswith("floor:"):
        return ()
    declared = _declared_support_path(scene, initial_place, support_path)
    if declared:
        return (declared,)
    return tuple(
        str(path) for path in getattr(scene, "behavior_support_prims", None) or ()
    )


def _support_candidate_bounds(stage, support_prims) -> list[tuple[float, ...]]:
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    result = []
    seen = set()
    for path in support_prims:
        root = stage.GetPrimAtPath(path)
        if not (root and root.IsValid()):
            continue
        colliders = [
            prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        for prim in [*colliders, root]:
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum, maximum = bounds.GetMin(), bounds.GetMax()
            row = tuple(
                float(value)
                for value in (minimum[0], minimum[1], maximum[0], maximum[1])
            )
            key = tuple(round(value, 5) for value in row)
            if key in seen or not all(math.isfinite(value) for value in row):
                continue
            seen.add(key)
            result.append(row)
    return result


def _support_object_half_xy(prim) -> tuple[float, float]:
    import fpv_scene
    import object_physics as ophys

    minimum, maximum = ophys.physical_world_bounds(fpv_scene, prim)
    return (
        max(0.005, 0.5 * float(maximum[0] - minimum[0])),
        max(0.005, 0.5 * float(maximum[1] - minimum[1])),
    )


def _support_grid_xy(
    stage,
    support_prims,
    xy,
    object_half_xy=(0.0, 0.0),
    max_candidates=160,
) -> list[tuple[float, float]]:
    if not hasattr(stage, "GetPrimAtPath"):
        return []
    import object_physics as ophys

    return ophys.support_placement_candidates(
        _support_candidate_bounds(stage, support_prims),
        xy,
        object_half_xy,
        max_candidates=max_candidates,
    )


def _calibrate_initial_support(scene, key: str, settled) -> None:
    surface = scene.on.get(str(key))
    if not surface or not settled.support_verified:
        return
    _xy, old_top = scene.surface_top(surface)
    scene.calibrate_surface_top(surface, settled.actual_support_z)
    if abs(float(old_top) - float(settled.actual_support_z)) > 1e-4:
        print(
            f"[step3r] calibrated support {surface}: registered z={float(old_top):.3f} "
            f"-> PhysX z={float(settled.actual_support_z):.3f}"
        )


def _support_retry_xy(
    stage,
    scene,
    xy,
    *,
    initial_place="",
    support_path=None,
    object_half_xy=(0.0, 0.0),
) -> list[tuple[float, float]]:
    """Deterministic retry points constrained to the object's declared support."""
    if str(initial_place).startswith("floor:"):
        return []
    support_prims = list(
        _support_paths(
            scene,
            support_path=support_path,
            initial_place=initial_place,
        )
    )
    if not support_prims:
        return []
    cx, cy, _ = _nearest_support_centroid(stage, support_prims, xy)
    if cx is None or cy is None:
        return []
    origin_x, origin_y = float(xy[0]), float(xy[1])
    dx, dy = float(cx) - origin_x, float(cy) - origin_y
    distance = math.hypot(dx, dy)
    if distance > 1e-6:
        inward_x, inward_y = dx / distance, dy / distance
    else:
        inward_x, inward_y = 1.0, 0.0
    side_x, side_y = -inward_y, inward_x

    result: list[tuple[float, float]] = []
    for step in (0.08, 0.16, 0.24, 0.34, 0.45):
        travel = min(step, distance) if distance > 1e-6 else 0.0
        base_x = origin_x + inward_x * travel
        base_y = origin_y + inward_y * travel
        if travel > 1e-6:
            result.append((base_x, base_y))

        lateral = step if distance <= 1e-6 else min(0.12, step * 0.5)
        result.extend(
            [
                (base_x + side_x * lateral, base_y + side_y * lateral),
                (base_x - side_x * lateral, base_y - side_y * lateral),
            ]
        )

        if distance <= 1e-6:
            result.extend(
                [
                    (base_x + inward_x * step, base_y + inward_y * step),
                    (base_x - inward_x * step, base_y - inward_y * step),
                ]
            )
    result.extend(_support_grid_xy(stage, support_prims, xy, object_half_xy))
    return list(dict.fromkeys(result))


def _oriented_box_size(size, orientation) -> tuple[float, float, float]:
    """Axis-aligned world size of a local box under a WXYZ quaternion."""
    sx, sy, sz = (float(value) for value in size[:3])
    w, x, y, z = (float(value) for value in orientation)
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotation = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    local = (sx, sy, sz)
    return tuple(sum(abs(row[j]) * local[j] for j in range(3)) for row in rotation)


def _inflate_initial_profile(
    world_size: tuple[float, float, float],
) -> tuple[tuple[float, float], float]:
    return (
        (
            max(
                0.005,
                INITIAL_PROFILE_SCALE * 0.5 * world_size[0] + INITIAL_PROFILE_MARGIN_M,
            ),
            max(
                0.005,
                INITIAL_PROFILE_SCALE * 0.5 * world_size[1] + INITIAL_PROFILE_MARGIN_M,
            ),
        ),
        max(
            0.01,
            INITIAL_PROFILE_SCALE * world_size[2] + INITIAL_PROFILE_MARGIN_M,
        ),
    )


def _initial_object_profile(
    spec, stage=None
) -> tuple[tuple[float, float], float] | None:
    """Return a conservative oriented collision footprint, with an asset-spec fallback."""
    if stage is not None:
        prim = stage.GetPrimAtPath(str(spec.get("path") or ""))
        if prim and prim.IsValid():
            import fpv_scene
            import object_physics as ophys

            minimum, maximum = ophys.physical_world_bounds(fpv_scene, prim)
            world_size = tuple(
                float(maximum[axis] - minimum[axis]) for axis in range(3)
            )
            if all(math.isfinite(value) and value > 0.0 for value in world_size):
                return _inflate_initial_profile(world_size)

    build = spec.get("build") or {}
    ref = build.get("behavior")
    if not ref:
        return None
    import behavior_import

    size = behavior_import.read_spec(str(ref)).get("bbox_size") or ()
    if len(size) < 3:
        return None
    raw_scale = build.get("scale", 1.0)
    if isinstance(raw_scale, (list, tuple)):
        if len(raw_scale) != 3:
            raise ValueError(
                f"asset scale must be a scalar or [x, y, z], got {raw_scale!r}"
            )
        scale_xyz = tuple(float(value) for value in raw_scale)
    else:
        scale = float(raw_scale)
        scale_xyz = (scale, scale, scale)
    world_size = _oriented_box_size(
        [scale_xyz[axis] * float(value) for axis, value in enumerate(size[:3])],
        spec.get("orientation", (1.0, 0.0, 0.0, 0.0)),
    )
    return _inflate_initial_profile(world_size)


def _footprints_overlap(xy, half_xy, occupied, gap=0.02) -> bool:
    x, y = float(xy[0]), float(xy[1])
    hx, hy = float(half_xy[0]), float(half_xy[1])
    return any(
        abs(x - ox) < hx + ohx + float(gap) and abs(y - oy) < hy + ohy + float(gap)
        for ox, oy, ohx, ohy in occupied
    )


def _collision_world_support_blocker(
    scene,
    xy: tuple[float, float],
    half_xy: tuple[float, float],
    support_z: float,
    object_height: float,
    *,
    support_paths: tuple[str, ...] = (),
) -> str | None:
    """Return geometry occupying the object's conservative volume above a support plane."""
    cworld = getattr(scene, "cworld", None)
    if cworld is None:
        return None
    excluded_roots = tuple(
        str(entity.prim_path).rstrip("/")
        for entity in scene.entities.values()
        if entity.role is Role.GRASPABLE and entity.prim_path
    ) + tuple(str(value).rstrip("/") for value in support_paths if value)
    x, y = float(xy[0]), float(xy[1])
    hx, hy = float(half_xy[0]), float(half_xy[1])
    base_z = float(support_z)
    top_z = base_z + float(object_height)
    for owner, cx, cy, ohx, ohy, bottom_z, obstacle_top_z in cworld.footprints:
        path = str(owner).rstrip("/")
        if any(path == root or path.startswith(root + "/") for root in excluded_roots):
            continue
        if float(obstacle_top_z) <= base_z + INITIAL_SUPPORT_CLEARANCE_EPS_M:
            continue
        if float(bottom_z) >= top_z - INITIAL_SUPPORT_CLEARANCE_EPS_M:
            continue
        if abs(x - float(cx)) >= hx + float(ohx):
            continue
        if abs(y - float(cy)) >= hy + float(ohy):
            continue
        return path
    return None


def _rank_initial_support_points(origin, candidates, center, yaw):
    """Keep a valid authored point first, then try the fixture's reachable edge inward."""
    unique = list(dict.fromkeys([origin, *candidates]))
    angle = math.radians(float(yaw))
    front = (-math.sin(angle), math.cos(angle))
    return [
        unique[0],
        *sorted(
            unique[1:],
            key=lambda point: -(
                (float(point[0]) - float(center[0])) * front[0]
                + (float(point[1]) - float(center[1])) * front[1]
            ),
        ),
    ]


def _initial_support_point_reachable(
    stage, scene, ph, rig, spec, point, support_z, object_height
) -> bool:
    """Run the runtime body's navigation, IK, and collision gates on a proposed object point."""
    if rig is None or getattr(scene, "nav_grid", None) is None:
        return True
    import init_validation
    import nav_grid

    contact = (
        float(point[0]),
        float(point[1]),
        float(support_z) + max(0.05, 0.5 * float(object_height)),
    )
    runtime_grid = scene.nav_grid
    behavior_task = getattr(scene, "behavior_task", None) or {}
    navigation_fallback = (
        {}
        if behavior_task.get("allow_narrow_support_preflight_routes") is True
        else None
    )
    scene.nav_grid = nav_grid.planning_view(runtime_grid)
    try:
        return (
            init_validation.pick_stand_runtime(
                ph,
                stage,
                rig,
                scene,
                str(spec["key"]),
                tuple(scene.start_xy),
                contact_override=contact,
                nearest_to_contact=True,
                navigation_fallback_out=navigation_fallback,
                allow_expensive_retries=False,
            )
            is not None
        )
    finally:
        scene.nav_grid = runtime_grid


def _probe_initial_support_point(
    stage,
    scene,
    ph,
    rig,
    spec,
    point,
    half_xy,
    height,
    path,
    fallback_hint_z,
    occupied,
    support_key,
    contact_key=None,
):
    if _footprints_overlap(point, half_xy, occupied):
        return None
    hint_z = fallback_hint_z
    if getattr(scene, "cworld", None) is not None:
        support_entity = scene.resolve(support_key)
        hint_z = _support_top_for(
            scene,
            point,
            support_entity,
            str(spec["path"]),
            declared=True,
            initial_place=f"on:{support_key}",
            support_path=path,
        )
        if hint_z <= float(scene.floor_z) + 1e-6:
            return None
        if abs(float(hint_z) - fallback_hint_z) > INITIAL_SUPPORT_SEMANTIC_PLANE_TOL_M:
            return None
        if _collision_world_support_blocker(
            scene,
            point,
            half_xy,
            hint_z,
            height,
            support_paths=(path,),
        ):
            return None
    probe = probe_declared_support(
        ph,
        point,
        hint_z,
        half_xy,
        height,
        exclude_path=str(spec["path"]),
        support_paths=(path,),
        min_support_fraction=KINEMATIC_SUPPORT_FRACTION,
    )
    if not (probe.verified and probe.ok):
        return None
    if not _initial_support_point_reachable(
        stage,
        scene,
        ph,
        rig,
        {**spec, "key": contact_key or spec["key"]},
        point,
        probe.support_z,
        height,
    ):
        return None
    return probe


def _empty_support_option_layout(stage, scene, ph, option, rig):
    """Validate a destination support that has no object resting on it at frame zero."""
    path = str(option.get("path") or "")
    support_key = str(option.get("support_key") or "")
    if not path or not support_key:
        return None
    surface = scene.resolve(support_key)
    cx, cy, bbox_top = _nearest_support_centroid(stage, [path], surface.xy)
    if cx is None or cy is None:
        return None
    fallback_hint_z = float(option.get("top_z", bbox_top))
    points = _rank_initial_support_points(
        (float(cx), float(cy)),
        _support_grid_xy(
            stage,
            [path],
            (float(cx), float(cy)),
            object_half_xy=(0.05, 0.05),
            max_candidates=64,
        ),
        (float(cx), float(cy)),
        float(option.get("yaw", 0.0)),
    )
    for point in points:
        hint_z = _support_top_for(
            scene,
            point,
            surface,
            path,
            declared=True,
            initial_place=f"on:{support_key}",
            support_path=path,
        )
        if hint_z <= float(scene.floor_z) + 1e-6:
            continue
        if abs(float(hint_z) - fallback_hint_z) > INITIAL_SUPPORT_SEMANTIC_PLANE_TOL_M:
            continue
        if _collision_world_support_blocker(
            scene, point, (0.05, 0.05), hint_z, 0.10, support_paths=(path,)
        ):
            continue
        probe = probe_declared_support(
            ph,
            point,
            hint_z,
            (0.05, 0.05),
            0.10,
            exclude_path="",
            support_paths=(path,),
            min_support_fraction=KINEMATIC_SUPPORT_FRACTION,
        )
        if not (probe.verified and probe.ok):
            continue
        if _initial_support_point_reachable(
            stage,
            scene,
            ph,
            rig,
            {"key": support_key},
            point,
            probe.support_z,
            0.10,
        ):
            return {"": (point, probe)}
    return None


def _support_profile_orientations(profile, allow_quarter_turn):
    half_xy, height = profile
    result = [(half_xy, height, False)]
    if allow_quarter_turn and abs(float(half_xy[0]) - float(half_xy[1])) > 1e-6:
        result.append(((float(half_xy[1]), float(half_xy[0])), height, True))
    return result


def _support_option_layout(
    stage,
    scene,
    ph,
    specs,
    option,
    rig=None,
    rotatable_keys=frozenset(),
    use_object_reach_keys: bool = False,
    preferred_xy_by_key=None,
):
    """Find a collision-verified, non-overlapping layout on one support option.

    Initial objects may use their own normal runtime stand search. Planned placements
    continue to use the support's authored operator stand. Explicit placement points
    lead each candidate search; if the hinted layout reaches a dead end, the whole
    layout retries automatic packing so a hint cannot make a valid support fail.
    """
    preferred_xy_by_key = preferred_xy_by_key or {}
    layout = _try_support_option_layout(
        stage,
        scene,
        ph,
        specs,
        option,
        rig=rig,
        rotatable_keys=rotatable_keys,
        use_object_reach_keys=use_object_reach_keys,
        preferred_xy_by_key=preferred_xy_by_key,
    )
    if layout is not None or not preferred_xy_by_key:
        return layout
    return _try_support_option_layout(
        stage,
        scene,
        ph,
        specs,
        option,
        rig=rig,
        rotatable_keys=rotatable_keys,
        use_object_reach_keys=use_object_reach_keys,
        preferred_xy_by_key={},
    )


def _try_support_option_layout(
    stage,
    scene,
    ph,
    specs,
    option,
    *,
    rig,
    rotatable_keys,
    use_object_reach_keys,
    preferred_xy_by_key,
):
    path = str(option.get("path") or "")
    support_key = str(option.get("support_key") or "")
    if not path or not support_key:
        return None
    if not specs:
        return _empty_support_option_layout(stage, scene, ph, option, rig)
    profiles = [(spec, _initial_object_profile(spec, stage)) for spec in specs]
    if any(profile is None for _spec, profile in profiles):
        return None
    profiles.sort(
        key=lambda row: row[1][0][0] * row[1][0][1],
        reverse=True,
    )
    first_entity = scene.resolve(str(profiles[0][0]["key"]))
    cx, cy, bbox_top = _nearest_support_centroid(stage, [path], first_entity.xy)
    if cx is None or cy is None:
        return None
    fallback_hint_z = float(option.get("top_z", bbox_top))
    occupied = []
    layout = {}
    for spec, profile in profiles:
        entity = scene.resolve(str(spec["key"]))
        original_path = str(spec.get("support_path") or "")
        preferred_xy = (preferred_xy_by_key or {}).get(str(spec["key"]))
        origin = (
            preferred_xy
            if preferred_xy is not None
            else entity.xy
            if original_path == path
            else (float(cx), float(cy))
        )
        selected = None
        for half_xy, height, quarter_turn in _support_profile_orientations(
            profile, str(spec["key"]) in rotatable_keys
        ):
            points = _rank_initial_support_points(
                origin,
                _support_grid_xy(
                    stage, [path], origin, object_half_xy=half_xy, max_candidates=64
                ),
                (cx, cy),
                float(option.get("yaw", 0.0)),
            )
            for point in points:
                probe = _probe_initial_support_point(
                    stage,
                    scene,
                    ph,
                    rig,
                    spec,
                    point,
                    half_xy,
                    height,
                    path,
                    fallback_hint_z,
                    occupied,
                    support_key,
                    contact_key=(
                        str(spec["key"])
                        if use_object_reach_keys
                        else str(option.get("support_key") or "") or None
                    ),
                )
                if probe is not None:
                    selected = (point, probe, quarter_turn, half_xy)
                    break
            if selected is not None:
                break
        if selected is None:
            return None
        point, probe, quarter_turn, half_xy = selected
        occupied.append((float(point[0]), float(point[1]), *half_xy))
        layout[str(spec["key"])] = (point, probe, quarter_turn)
    return layout


def _rebind_initial_support(stage, scene, key, option, layout, specs) -> None:
    """Commit one PhysX-validated alternative while preserving the logical surface name."""
    import behavior_import as bi
    import fpv_scene
    from scene_state import Role

    path = str(option["path"])
    yaw = float(option.get("yaw", 0.0))
    owner = scene.resolve(key)
    surface = scene.affordance(key, Role.SURFACE) or owner
    measured = bi._collision_bbox(stage, path)
    if measured is None:
        bounds = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(path))
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
    else:
        minimum, maximum = measured
    cx = 0.5 * float(minimum[0] + maximum[0])
    cy = 0.5 * float(minimum[1] + maximum[1])
    hx = 0.5 * float(maximum[0] - minimum[0])
    hy = 0.5 * float(maximum[1] - minimum[1])
    support_z = float(next(iter(layout.values()))[1].support_z)
    surface.prim_path = path
    surface.xy = (cx, cy)
    surface.z = support_z
    surface.top_z = support_z
    surface.half = (hx, hy)
    surface.foot_xy = (cx, cy)
    surface.foot_half = (hx, hy)
    surface.approach_yaw = yaw
    angle = math.radians(yaw)
    forward = (-math.sin(angle), math.cos(angle))
    surface.meta["front_stand"] = (
        cx + forward[0] * (abs(forward[0]) * hx + abs(forward[1]) * hy + 0.55),
        cy + forward[1] * (abs(forward[0]) * hx + abs(forward[1]) * hy + 0.55),
    )
    getattr(scene, "_static_aabb", {}).pop(key, None)
    for spec in specs:
        item_key = str(spec["key"])
        point, probe, _quarter_turn = layout[item_key]
        spec["support_path"] = path
        entity = scene.resolve(item_key)
        entity.xy = (float(point[0]), float(point[1]))
        entity.foot_xy = entity.xy
        entity.z = float(probe.support_z) + 0.05
        entity.top_z = float(probe.support_z)
        getattr(scene, "_static_aabb", {}).pop(item_key, None)
    scene.behavior_support_prims = sorted(
        {*getattr(scene, "behavior_support_prims", []), path}
    )


def _support_load_specs(load, specs_by_key):
    return [specs_by_key[row[0]] for row in load if row[0] in specs_by_key]


def _support_load_rotation_keys(load) -> frozenset[str]:
    return frozenset(row[0] for row in load if row[1])


def _support_load_preferred_xy(load) -> dict[str, tuple[float, float]]:
    return {row[0]: row[2] for row in load if row[2] is not None}


def _publish_verified_placement_sites(scene, key, path, planned_layouts) -> None:
    sites = {}
    for _load, layout in planned_layouts:
        for object_key, (point, probe, quarter_turn) in layout.items():
            if not object_key:
                continue
            row = {
                "xy": (float(point[0]), float(point[1])),
                "quarter_turn": bool(quarter_turn),
                "support_z": float(probe.support_z),
            }
            bucket = sites.setdefault(object_key, [])
            if row not in bucket:
                bucket.append(row)
    surface = scene.resolve(key)
    surface.meta["verified_placement_path"] = str(path)
    surface.meta["verified_placement_sites"] = sites


def _support_option_plan(
    stage,
    scene,
    ph,
    initial_group,
    planned_loads,
    option,
    specs_by_key,
    rig,
):
    if (
        not initial_group
        and not planned_loads
        and not bool(option.get("requires_worktop", True))
    ):
        return {}, []
    initial_layout = None
    behavior_task = getattr(scene, "behavior_task", None) or {}
    use_initial_object_stands = (
        behavior_task.get("allow_initial_support_object_stands") is True
    )
    if initial_group:
        if use_initial_object_stands:
            initial_layout = _support_option_layout(
                stage,
                scene,
                ph,
                initial_group,
                option,
                rig=rig,
                use_object_reach_keys=True,
            )
        else:
            initial_layout = _support_option_layout(
                stage, scene, ph, initial_group, option, rig=rig
            )
        if initial_layout is None:
            return None
    planned_layouts = []
    initial_keys = tuple(sorted(str(spec["key"]) for spec in initial_group))
    for load in planned_loads:
        load_keys = tuple(row[0] for row in load)
        # Removing objects preserves every physical and reachability property already
        # established for the initial layout; only a newly placed/rotated object needs
        # another support-stand validation.
        unchanged_initial_load = not any(row[1] for row in load) and set(
            load_keys
        ).issubset(initial_keys)
        if unchanged_initial_load and (
            load_keys == initial_keys or use_initial_object_stands
        ):
            layout = {
                object_key: initial_layout[object_key]
                for object_key in load_keys
                if object_key in initial_layout
            }
        else:
            layout = _support_option_layout(
                stage,
                scene,
                ph,
                _support_load_specs(load, specs_by_key),
                option,
                rig=rig,
                rotatable_keys=_support_load_rotation_keys(load),
                preferred_xy_by_key=_support_load_preferred_xy(load),
            )
        if layout is None:
            return None
        planned_layouts.append((load, layout))
    if initial_layout is None:
        if planned_layouts:
            initial_layout = planned_layouts[0][1]
        else:
            initial_layout = _support_option_layout(
                stage, scene, ph, (), option, rig=rig
            )
    return None if initial_layout is None else (initial_layout, planned_layouts)


def _support_group_difficulty(stage, group):
    profiles = [
        _initial_object_profile(spec, stage) or ((0.0, 0.0), 0.0) for spec in group
    ]
    return len(group), sum(half[0] * half[1] for half, _height in profiles)


def _support_preflight_difficulty(stage, key, group, planned_by_key, specs_by_key):
    loads = planned_by_key.get(key) or ()
    planned_specs = [_support_load_specs(load, specs_by_key) for load in loads] or [
        group
    ]
    return max(
        _support_group_difficulty(stage, candidate) for candidate in planned_specs
    )


def _select_planned_support_option(
    stage,
    scene,
    ph,
    key,
    group,
    planned_loads,
    options,
    specs_by_key,
    claimed,
    rig,
):
    for raw_option in options[:INITIAL_SUPPORT_OPTION_LIMIT]:
        option = {**raw_option, "support_key": key}
        path = str(option.get("path") or "")
        if not path or path in claimed:
            continue
        plan = _support_option_plan(
            stage,
            scene,
            ph,
            group,
            planned_loads,
            option,
            specs_by_key,
            rig,
        )
        if plan is not None:
            return option, *plan
    return None


def _prepare_initial_support_bindings(
    stage, scene, ph, specs, rig=None, program=""
) -> None:
    """Validate layouts on each logical fixture's already-bound concrete prim."""
    options_by_key = getattr(scene, "initial_support_options", None) or {}
    specs_by_key = support_planning_specs(scene, specs)
    planned_by_key = planned_support_loads(program, scene, specs, options_by_key)
    groups = {}
    for spec in specs:
        initial_place = str(spec.get("initial_place") or "")
        if initial_place.startswith("on:"):
            groups.setdefault(initial_place[3:], []).append(spec)
    for key in options_by_key:
        groups.setdefault(str(key), [])
    if not groups:
        return
    claimed = {
        str(entity.prim_path)
        for key, entity in scene.entities.items()
        if key in options_by_key and key not in groups and entity.prim_path
    }

    ordered_groups = sorted(
        groups.items(),
        key=lambda row: _support_preflight_difficulty(
            stage, row[0], row[1], planned_by_key, specs_by_key
        ),
        reverse=True,
    )
    for key, group in ordered_groups:
        movable_group = bool(group) and all(
            str(spec.get("movable_support") or "") == key for spec in group
        )
        if movable_group and key not in options_by_key:
            owner = scene.resolve(key)
            if owner.role is not Role.GRASPABLE or not owner.prim_path:
                raise RuntimeError(
                    f"declared movable initial support {key!r} is not a registered graspable"
                )
            continue
        planned_loads = sorted(
            planned_by_key.get(key) or (),
            key=lambda load: _support_group_difficulty(
                stage, _support_load_specs(load, specs_by_key)
            ),
            reverse=True,
        )
        bound_path = _declared_support_path(scene, f"on:{key}")
        exact_options = [
            option
            for option in options_by_key.get(key, ())
            if str(option.get("path") or "").rstrip("/") == bound_path
        ]
        if not exact_options:
            raise RuntimeError(
                f"declared initial support {key!r} has no preflight option for its "
                f"bound prim {bound_path!r}"
            )
        selected = _select_planned_support_option(
            stage,
            scene,
            ph,
            key,
            group,
            planned_loads,
            exact_options,
            specs_by_key,
            claimed,
            rig,
        )
        if selected is None:
            raise RuntimeError(
                f"declared initial support {key!r} at {bound_path!r} cannot fit its "
                "initial or planned object load"
            )
        option, layout, planned_layouts = selected
        if layout:
            _rebind_initial_support(stage, scene, key, option, layout, group)
        _publish_verified_placement_sites(
            scene, key, str(option["path"]), planned_layouts
        )
        claimed.add(str(option["path"]))
        print(
            f"[step3r] support preflight {key}: retained {bound_path} "
            f"({len(group)} initial object(s), {len(planned_loads)} planned load(s), "
            "collision-backed placement sites verified)"
        )


def _build_and_park_initial_objects(stage, scene, core, rig, specs) -> None:
    """Load every initial object before support planning and keep it outside the probe volume."""
    import fpv_scene

    park_z = float(scene.floor_z) + 5.0
    for index, spec in enumerate(specs):
        if spec.get("prebuilt"):
            prim = stage.GetPrimAtPath(spec["path"])
            if not prim or not prim.IsValid():
                raise RuntimeError(
                    f"prebuilt graspable {spec['key']!r} is missing at {spec['path']}"
                )
            continue
        if "def" in spec:
            build, offset = core.load_object_def(spec["def"])
        else:
            build, offset = spec["build"], None
        core.set_object(rig, build, offset)
        core.build_object(stage, rig, spec["path"])
        if spec.get("color"):
            _recolor_prim(stage, spec["path"], spec["color"])
        entity = scene.resolve(spec["key"])
        orientation = tuple(
            float(value) for value in spec.get("orientation", (1.0, 0.0, 0.0, 0.0))
        )
        fpv_scene.set_world_pose(
            stage.GetPrimAtPath(spec["path"]),
            (float(entity.xy[0]), float(entity.xy[1]), park_z + 2.0 * index),
            orientation,
        )


def _refresh_movable_initial_support_pose(stage, scene, spec) -> None:
    """Resolve a deferred child's pose after its movable support has settled."""
    support_key = str(spec.get("movable_support") or "")
    if not support_key:
        return
    import fpv_scene
    import object_physics as ophys

    support = scene.resolve(support_key)
    support_path = str(support.prim_path or "")
    if not support_path or support_path != str(spec.get("support_path") or ""):
        raise RuntimeError(
            f"movable initial support {support_key!r} has no matching concrete prim"
        )
    support_prim = stage.GetPrimAtPath(support_path)
    if not support_prim or not support_prim.IsValid():
        raise RuntimeError(
            f"movable initial support {support_key!r} was not built before its contents"
        )
    minimum, maximum = ophys.physical_world_bounds(fpv_scene, support_prim)
    offset = spec.get("movable_support_offset_xy") or (0.0, 0.0)
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        raise RuntimeError(
            f"movable initial support {support_key!r} has an invalid child offset"
        )
    entity = scene.resolve(str(spec["key"]))
    support_center = (
        0.5 * float(minimum[0] + maximum[0]),
        0.5 * float(minimum[1] + maximum[1]),
    )
    entity.xy = (
        support_center[0] + float(offset[0]),
        support_center[1] + float(offset[1]),
    )
    entity.foot_xy = entity.xy
    support_top = float(maximum[2])
    entity.z = support_top + 0.05
    entity.top_z = support_top
    spec["support_z"] = support_top
    getattr(scene, "_static_aabb", {}).pop(str(spec["key"]), None)


def _declare_movable_initial_support_relations(scene, specs) -> None:
    """Publish movable support relations only after both live bodies have settled."""
    for spec in specs or ():
        support_key = str(spec.get("movable_support") or "")
        if support_key:
            scene.declare_place(str(spec["key"]), support_key)


def _validate_initial_object_relations(scene) -> None:
    """Require every declared physical ON/INSIDE relation after deferred realization."""
    import object_physics as ophys
    from isaac_goal import Pred

    physical = {
        name
        for name, entity in scene.entities.items()
        if getattr(entity, "prim_path", None)
    }
    expected = {
        (Pred.ON, subject, target)
        for subject, target in (getattr(scene, "on", None) or {}).items()
        if subject in physical and target in physical
    }
    expected.update(
        (Pred.INSIDE, subject, target)
        for subject, target in (getattr(scene, "inside", None) or {}).items()
        if subject in physical and target in physical
    )
    ophys.require_initial_relations(expected, scene.relations(source="physics"))


def _exit_process(code: int) -> None:
    """Terminate this process with ``code``, INSTEAD of ``app.close()``.

    ``SimulationApp.close()`` ends in ``shutdown_and_release_framework()``, which tears Kit down
    natively and NEVER RETURNS -- so every ``return code`` after it is unreachable and the shell sees 0
    whatever happened. Hand the status to the OS directly instead.

    Safe because nothing here is still pending: PNGs are written synchronously by ``fpv_render._save_png``
    (PIL) inside the capture loop, ``fpv_render.teardown`` drains replicator via ``wait_until_complete``
    BEFORE the encode, ``encode_mp4`` shells out through a synchronous ``subprocess.run``, the manifest and
    replay dump are written under ``with open(...)``, and any artifact upload happens after the
    process exits. ``episode_runtime`` registers no ``atexit`` handler, and its capture cleanup finishes
    before reaching this success exit.
    """
    import logging as _logging

    sys.stdout.flush()
    sys.stderr.flush()
    _logging.shutdown()
    os._exit(int(code))


def _out_and_tag(args) -> tuple[str, str]:
    """The output directory + artifact tag, resolved identically on EVERY exit path.

    The directory is created unconditionally: ``--check`` returns long before the render path's
    ``makedirs``, and a fresh MAST node has no ``fpv_outputs/`` at all (``_write_episode_manifest``
    would die ``[Errno 2]``).

    ``--check`` gets its own ``_chk`` tag: it ships no video, and ``mast_isaac_render_node.sh`` hands
    the SAME ``--tag`` to its ``CHECK=1`` pre-flight and to the render that follows, so a shared tag
    lets the pre-flight overwrite the manifest + replay dump the render published. The suffix is on the
    TAG, not on each filename, so ONE chokepoint separates every artifact the check path writes and
    ``<tag>_chk_manifest.json`` stays derivable from ``<tag>``."""
    out = os.path.join(_TRY_DIR, "fpv_outputs")
    if getattr(args, "subdir", None):
        out = os.path.join(out, args.subdir)
    os.makedirs(out, exist_ok=True)
    tag = args.tag
    if not tag:
        # A seeded run must not overwrite another seed's mp4 / frames / manifest.
        seed = getattr(args, "seed", None)
        tag = f"step3_{args.scene}" if seed is None else f"step3_{args.scene}_s{seed}"
    if getattr(args, "check", False):
        tag = f"{tag}_chk"
    return out, tag


_GRAB_PRIMS: frozenset = frozenset({"pick", "detach"})
_PUT_PRIMS: frozenset = frozenset({"place_inside", "place", "attach"})


def _staging_offtarget() -> list[dict]:
    """Objects that did not stay on the support they were staged on. ``[]`` when the ledger is
    unavailable, never an exception -- a manifest field must not be able to fail an episode.
    """
    try:
        import object_physics as _ophys

        return list(_ophys.offtarget_events())
    except Exception:  # noqa: BLE001
        return []


def _attributed_histogram(cmds: list[dict]) -> tuple[dict, int]:
    """``(histogram, n_cascade)`` -- the same failures as ``failure_histogram``, charged to the failure
    that CAUSED them rather than to the command that reported them.

    ``PrimitiveError.PLACE_UNSTABLE`` is emitted from five sites and only one is physics; the other four
    include **"PLACE with an empty hand"**. So one upstream failure that leaves the hand empty makes
    *every remaining* ``[PutInside]`` in the program report ``place_unstable``, and a single real failure
    manufactures an arbitrarily long run of fake ones. Measured over the `cb1` baseline: **52 % of all
    `place_unstable` occurrences were an empty hand, and 79 % of those trace to an upstream
    `path_blocked`** -- re-attributed, the class falls from 38 % of decided rows to **27 %**, and 23 of
    its 66 occurrences disappear. That is why the raw label read as the portfolio's largest defect and
    as stubbornly flat: most of it was a shadow of `path_blocked`.

    Attribution prefers the DETAIL string, which says outright that the hand was empty, and falls back to
    reconstructing hand state from the command stream for records written before details were carried.
    A cascade failure is re-keyed to ``"<cause> (cascade)"``, never dropped, so the total always equals
    ``n_fail``.

    ADDITIVE ONLY. ``failure_histogram`` and every ``reason`` are left exactly as they were, so no
    verdict, score or historical comparison moves; this is a second reading of the same rows.
    """
    import collections

    out: collections.Counter = collections.Counter()
    held, last_fail, n_cascade = False, None, 0
    for c in cmds:
        prim, reason, ok = c.get("prim"), c.get("reason"), bool(c.get("ok"))
        detail = str(c.get("detail") or "")
        if prim in _GRAB_PRIMS and ok:
            held = True
        elif prim in _PUT_PRIMS and reason == "place_unstable":
            # "empty hand" from the detail is authoritative; the reconstruction is the fallback.
            if "empty hand" in detail or (not detail and not held):
                out[f"{last_fail or 'unknown'} (cascade)"] += 1
                n_cascade += 1
            else:
                out[reason] += 1
                held = False
            continue
        elif prim in _PUT_PRIMS and ok:
            held = False
        if not ok and reason:
            out[reason] += 1
            last_fail = reason
    return dict(out), n_cascade


def _selftest_attributed_histogram() -> None:
    """The cascade must be re-charged, never invented and never dropped."""
    C = lambda p, r, d=None: {  # noqa: E731
        "prim": p,
        "reason": r,
        "ok": r == "ok",
        "detail": d,
    }
    # One blocked move -> a failed pick -> two puts on an empty hand: ONE real failure, not three.
    h, n = _attributed_histogram(
        [
            C("navigate_to", "path_blocked"),
            C("pick", "out_of_reach"),
            C("place_inside", "place_unstable", "PLACE with an empty hand"),
            C("place_inside", "place_unstable", "PLACE with an empty hand"),
        ]
    )
    assert n == 2 and h.get("place_unstable") is None, h
    assert h == {"path_blocked": 1, "out_of_reach": 1, "out_of_reach (cascade)": 2}, h
    # A REAL ejection (hand was full, detail is the settle message) stays `place_unstable`.
    h2, n2 = _attributed_histogram(
        [
            C("pick", "ok"),
            C(
                "place_inside",
                "place_unstable",
                "the settle LAUNCHED ['Obj_a'] (EJECTED:...)",
            ),
        ]
    )
    assert h2 == {"place_unstable": 1} and n2 == 0, h2
    # Fallback path: no detail recorded, hand state decides. Full hand -> real; empty -> cascade.
    assert _attributed_histogram([C("pick", "ok"), C("place", "place_unstable")])[
        0
    ] == {"place_unstable": 1}
    assert _attributed_histogram(
        [C("pick", "grasp_failed"), C("place", "place_unstable")]
    )[0] == {
        "grasp_failed": 1,
        "grasp_failed (cascade)": 1,
    }
    # TOTALS ARE CONSERVED: the attributed histogram always sums to the raw one.
    for rows in (
        [
            C("navigate_to", "path_blocked"),
            C("pick", "out_of_reach"),
            C("place", "place_unstable", "PLACE with an empty hand"),
        ],
        [
            C("pick", "ok"),
            C("place_inside", "ok"),
            C("pick", "ok"),
            C("place_inside", "place_unstable", "EJECTED"),
        ],
    ):
        raw = sum(1 for r in rows if not r["ok"])
        assert sum(_attributed_histogram(rows)[0].values()) == raw
    assert _attributed_histogram([]) == ({}, 0)
    print("[episode_runtime] _selftest_attributed_histogram PASS")


def _live_label(ent, fr):
    """The label where it IS on this frame, for a target the wearer is HOLDING. None otherwise.

    ``label_from_meta`` reads the pose the entity was REGISTERED at, i.e. where the object rests.
    A raised inspect carries the object to the face, so for every graspable that reader was
    measuring an empty shelf: on ``BOTTLE0816`` it reported the bottle 1.59 m away and 78 deg below
    the eye while the frame it was judging has it 0.3 m in front of the face. Numbers that describe
    a different place than the frame are worse than no numbers, because they read as measurements.
    """
    import label_visibility as lv
    import numpy as np

    meta = getattr(ent, "meta", None) or {}
    off = meta.get("label_local")
    obj = getattr(fr, "obj", None)
    if not off or not (isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "pose"):
        return None
    T = np.asarray(obj[1], dtype=np.float64)
    if T.shape != (4, 4):
        return None
    local_meta = dict(meta)
    local_meta["label_pos"] = off
    local_meta["label_face3"] = meta.get("label_face_local")
    local_meta["label_up3"] = meta.get("label_up_local")
    label = lv.label_from_meta(local_meta)
    return None if label is None else lv.transform_label(label, T)


def _glass_shader(mtl):
    """The transmissive shader behind ``mtl``, or None when the material is opaque.

    Two ways a BEHAVIOR body ends up see-through: an MDL whose name says glass (this bottle binds
    ``OmniGlass`` to its wall), or a preview surface whose opacity is authored below one. Anything
    else is left completely alone -- an opaque body has no problem to fix and repainting it would
    only throw away the texture it came with.
    """
    from pxr import Usd, UsdShade

    if not mtl:
        return None
    for prim in Usd.PrimRange(mtl.GetPrim()):
        sh = UsdShade.Shader(prim)
        if not sh:
            continue
        name = " ".join(
            str(v or "")
            for v in (
                sh.GetSourceAssetSubIdentifier("mdl"),
                sh.GetSourceAsset("mdl"),
                sh.GetShaderId(),
            )
        ).lower()
        if "glass" in name:
            return sh
        op = sh.GetInput("opacity")
        if op and op.Get() is not None and float(op.Get()) < 1.0:
            return sh
    return None


def _asset_diffuse_texture(root, mesh_prim):
    """The body's OWN diffuse map, so making it opaque does not also make it grey.

    A BEHAVIOR bake keeps the original VRay material next to the glass override -- the bottle
    carries ``qsdqik__base_link__diffuse.png`` even though its wall is bound to OmniGlass -- and
    the bake names those maps ``<model>__<link>__diffuse.png``. Matching on the mesh's own name is
    what lets the repainted body keep the colour and print it shipped with.
    """
    from pxr import Usd, UsdShade

    want = mesh_prim.GetName().lower()
    for prim in Usd.PrimRange(root):
        sh = UsdShade.Shader(prim)
        if not sh:
            continue
        for inp in sh.GetInputs():
            if "diffuse" not in inp.GetBaseName().lower():
                continue
            val = inp.Get()
            path = getattr(val, "resolvedPath", "") or getattr(val, "path", "") or ""
            if path and want in str(path).lower():
                return str(path)
    return ""


def _opaque_graspables(stage, scene) -> None:
    """Make a LABELLED graspable's body opaque IF it is transparent. ``ISAAC_LABEL_OPAQUE_BODY=0`` off.

    ⛔ MEASURED OVER FOURTEEN RENDERS, and it is the whole reason a raised [LookAt] showed no label.
    The decal was provably carried by the object, upright, dead centre in frame, textured, and
    5-38 mm IN FRONT of the bottle -- and it did not draw. It drew in exactly the frames where it
    did NOT overlap its own bottle on screen (held out at 120 mm, poking past the shoulder, lying
    beside a toppled bottle) and in none where it did. Repainting only the BODY -- the decal's own
    material untouched -- makes it read perfectly. A clear-glass body composites over what is in
    front of it, and no standoff, thickness, aspect or aim gets around that.

    A printed label and a transparent body do not go together, which is also true of the real
    thing: nobody prints a best-before date on clear glass and expects it read. Scoped twice over,
    because this is a repaint and repaints are how scenes quietly lose their look:

      * only objects that CARRY a decal -- unlabelled glassware keeps its material;
      * only meshes that are actually transmissive (``_glass_shader``) -- an opaque box is not
        touched at all;
      * and the replacement re-uses the body's OWN diffuse map (``_asset_diffuse_texture``), so it
        keeps its colour and print and only stops being see-through.
    """
    import os

    if os.getenv("ISAAC_LABEL_OPAQUE_BODY", "1") != "1":
        return
    labelled = {
        str(getattr(e, "prim_path", "") or "")
        for e in (getattr(scene, "entities", None) or {}).values()
        if (getattr(e, "meta", None) or {}).get("label_local")
    }
    if not labelled:
        return
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    n_seen = n_fixed = 0
    for spec in getattr(scene, "grasp_objs", None) or []:
        path = str(spec.get("path") or "")
        root = stage.GetPrimAtPath(path)
        if path not in labelled or not (root and root.IsValid()):
            continue
        body = stage.GetPrimAtPath(f"{path}/m")
        for prim in Usd.PrimRange(body if (body and body.IsValid()) else root):
            if not UsdGeom.Mesh(prim) or "Cue_" in str(prim.GetPath()):
                continue
            n_seen += 1
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if _glass_shader(bound) is None:
                continue  # already opaque: leave the asset exactly as it shipped
            base = f"{path}/OpaqueLabelBody_{prim.GetName()}"
            mtl = UsdShade.Material.Define(stage, base)
            sh = UsdShade.Shader.Define(stage, base + "/S")
            sh.CreateIdAttr("UsdPreviewSurface")
            tex_path = _asset_diffuse_texture(root, prim)
            if tex_path:
                st = UsdShade.Shader.Define(stage, base + "/St")
                st.CreateIdAttr("UsdPrimvarReader_float2")
                st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                tex = UsdShade.Shader.Define(stage, base + "/T")
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                    "sRGB"
                )
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    st.ConnectableAPI(), "result"
                )
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                sh.CreateInput(
                    "diffuseColor", Sdf.ValueTypeNames.Color3f
                ).ConnectToSource(tex.ConnectableAPI(), "rgb")
            else:
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(0.72, 0.74, 0.70)
                )
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            n_fixed += 1
    print(
        f"[bench] labelled bodies: {n_fixed}/{n_seen} mesh(es) were transmissive and were made "
        f"opaque (a decal does not render over clear glass); the rest kept their material"
    )


_DECAL_POSE_DEBUG_DONE: list = []


def _decal_pose_debug(stage, rig, fr, obj_T) -> None:
    """ONE dump, on the first ``look_hold`` frame: who is actually driving the decal.

    The picture has been ambiguous four times running -- a card that renders vertical and legible
    while the object rests, and as a horizontal sliver at its base while it is held, means the BODY
    and the DECAL are not on the same transform. Reading both world matrices out of the live stage
    at the frame in question is the only statement about that which cannot be misread.
    """
    if _DECAL_POSE_DEBUG_DONE or getattr(fr, "seg", "") != "look_hold":
        return
    _DECAL_POSE_DEBUG_DONE.append(1)
    try:
        import numpy as np
        from pxr import Usd, UsdGeom

        xc = UsdGeom.XformCache(Usd.TimeCode.Default())

        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )

        def _row(path):
            prim = stage.GetPrimAtPath(path)
            if not (prim and prim.IsValid()):
                return f"{path}: NO PRIM"
            m = np.array(xc.GetLocalToWorldTransform(prim)).reshape(4, 4)
            # USD is row-vector: the basis rows are m[0..2, :3], the translation is m[3, :3].
            up = m[:3, :3].T @ np.array([0.0, 0.0, 1.0])
            # The TRANSFORM only says which frame the prim hangs off. WHERE ITS GEOMETRY IS is the
            # world bbox -- a decal whose corner points were authored in some other frame rides the
            # object's transform and still sits where the object used to be. The two have to be
            # printed together or the transform alone reads as proof of something it does not show.
            rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            vis = UsdGeom.Imageable(prim).ComputeVisibility()
            mat = ""
            try:
                from pxr import UsdShade

                b = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                mat = str(b.GetPath()) if b else "NONE"
            except Exception:  # noqa: BLE001
                mat = "?"
            return (
                f"{path}: t=({m[3, 0]:.3f},{m[3, 1]:.3f},{m[3, 2]:.3f}) "
                f"localZ=({up[0]:+.2f},{up[1]:+.2f},{up[2]:+.2f}) "
                f"bbox x[{mn[0]:.3f},{mx[0]:.3f}] y[{mn[1]:.3f},{mx[1]:.3f}] "
                f"z[{mn[2]:.3f},{mx[2]:.3f}] vis={vis} mat={mat.rsplit('/', 1)[-1]}"
            )

        root = str(getattr(rig, "_obj_path", "") or "")
        print(f"[posedbg] seg={fr.seg} written_to={root}")
        print(
            f"[posedbg] obj_T t=({obj_T[0, 3]:.3f},{obj_T[1, 3]:.3f},{obj_T[2, 3]:.3f})"
        )
        for p in (root, f"{root}/m", f"{root}/Cue_side", f"{root}/Cue_top"):
            print(f"[posedbg]   {_row(p)}")
        prim = stage.GetPrimAtPath(root)
        if prim and prim.IsValid():
            print(
                "[posedbg]   children="
                + ",".join(c.GetName() for c in prim.GetChildren())
            )
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must never take a render down
        print(f"[posedbg] unavailable: {exc}")


def _decal_probe(scene, ent, label) -> None:
    """Where the decal PRIM actually is, beside where the metadata says the label is.

    A row built from ``label_local`` describes a point, not a prim. If the two ever part company --
    a decal never created, parented under the wrong anchor, or left behind when the object moved --
    every number in the row stays plausible and every one of them is about empty space.
    """
    if getattr(scene, "stage", None) is None:
        return
    roots = [
        p
        for p in (
            getattr(ent, "prim_path", None),
            f"/World/Obj_{getattr(ent, 'name', '')}",
        )
        if p
    ]
    # The BODY's own live bounds, because the decal is offset from an axis taken from the BAKED
    # spec's ``base_link_offset``. If the composed body does not sit on that axis, a card mounted
    # "4 mm proud of the wall" is inside the glass -- which is what a card that renders on a narrow
    # neck and vanishes on a wide body looks like.
    try:
        (amnx, amny, amnz), (amxx, amxy, amxz) = scene.object_aabb(ent.name)
        print(
            f"[decalprobe] {ent.name} body: centre=({0.5 * (amnx + amxx):.3f},"
            f"{0.5 * (amny + amxy):.3f},{0.5 * (amnz + amxz):.3f}) "
            f"size=({amxx - amnx:.3f},{amxy - amny:.3f},{amxz - amnz:.3f})"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[decalprobe] {getattr(ent, 'name', '?')} body bounds unreadable ({exc})"
        )
    for path, name in (
        (p, n) for p in dict.fromkeys(roots) for n in ("Cue_side", "Cue_top", "Cue_lid")
    ):
        full = f"{path}/{name}"
        prim = scene.stage.GetPrimAtPath(full)
        if not (prim and prim.IsValid()):
            print(f"[decalprobe] {full}: NO SUCH PRIM")
            continue
        try:
            (bx, by), zmin, zmax = scene._live_bbox(full)
        except Exception as exc:  # noqa: BLE001 -- a diagnostic must not take a render down
            print(f"[decalprobe] {full}: unreadable ({exc})")
            continue
        d = float(((bx - label.pos[0]) ** 2 + (by - label.pos[1]) ** 2) ** 0.5)
        print(
            f"[decalprobe] {full}: centre=({bx:.3f},{by:.3f},{0.5 * (zmin + zmax):.3f}) "
            f"z={zmin:.3f}..{zmax:.3f} visible={prim.IsActive()} "
            f"xy-gap-to-label={d * 1000:.0f}mm"
        )


def _label_min_distance(ent, frame) -> float:
    import label_visibility as lv

    meta = getattr(ent, "meta", None) or {}
    if (
        getattr(frame, "seg", "") == "look_hold"
        and (
            meta.get("label_source") == "native_asset_texture"
            or meta.get("look_held_distance_m") is not None
        )
    ):
        return lv.HELD_NEAR_M
    return lv.NEAR_M


def _label_frame_sample(
    ent, bbox, frame, index: int, boxes, camera_output=None
) -> dict | None:
    import head_mount
    import label_visibility as lv

    label = _live_label(ent, frame) or lv.label_from_meta(
        (getattr(ent, "meta", None) or {}),
        bbox,
    )
    if label is None:
        return None
    rot, pos = head_mount.head_pose(frame.pose156, frame.yaw, frame.xy, frame.z)
    spine = head_mount.spine3_frame(frame.pose156, frame.yaw, frame.xy, frame.z)[0]
    if camera_output is None:
        eye = pos + rot @ head_mount.EYE_LOCAL
        gaze = getattr(frame, "gaze", None)
        camera_look = (
            rot @ head_mount.FACE_FORWARD_LOCAL
            if gaze is None
            else np.asarray(gaze, dtype=np.float64) - eye
        )
        requested_zoom = float(getattr(frame, "zoom", None) or 0.0)
        zoom_span = float(getattr(frame, "zoom_span", None) or label.span_m)
        focal_mm = (
            lv.zoom_focal_mm(zoom_span, float(np.linalg.norm(label.pos - eye)))
            if requested_zoom > 0.0
            else head_mount.FOCAL_MM
        )
        effective_zoom = requested_zoom
        zoom_suppressed_reasons = ()
    else:
        eye = np.asarray(camera_output.eye, dtype=np.float64)
        camera_look = np.asarray(camera_output.look, dtype=np.float64)
        focal_mm = float(camera_output.focal_mm)
        effective_zoom = float(camera_output.effective_zoom)
        zoom_suppressed_reasons = tuple(
            camera_output.diagnostics.zoom_suppressed_reasons
        )
    assessment = lv.assess(
        eye,
        spine,
        label,
        boxes,
        image_w=lv.DELIVERED_IMAGE_W,
        image_h=lv.DELIVERED_IMAGE_H,
        focal_mm=focal_mm,
        camera_look=camera_look,
        enforce_torso_limits=False,
        min_distance_m=_label_min_distance(ent, frame),
    )
    cam_yaw, cam_pitch = lv.camera_relative_angles(label.pos - eye, camera_look)
    gaze = getattr(frame, "gaze", None)
    authored_focus = bool(
        gaze is not None
        and float(np.linalg.norm(np.asarray(gaze) - np.asarray(label.pos))) < 0.08
    )
    return {
        "index": index,
        "label": label,
        "assessment": assessment,
        "cam_yaw": cam_yaw,
        "cam_pitch": cam_pitch,
        "in_frame": assessment.in_frame,
        "authored_focus": authored_focus,
        "focal_mm": focal_mm,
        "effective_zoom": effective_zoom,
        "zoom_suppressed_reasons": zoom_suppressed_reasons,
        "readable": assessment.ok,
    }


def _longest_consecutive(indices) -> int:
    longest = run = 0
    previous = None
    for index in sorted(indices):
        run = run + 1 if previous is not None and index == previous + 1 else 1
        longest = max(longest, run)
        previous = index
    return longest


def _label_look_row(
    scene, frames, command, boxes, min_run: int, camera_outputs=None
) -> dict:
    import label_visibility as lv

    lo = int(command["frame_start"])
    hi = min(int(command["frame_end"]), len(frames))
    holds = [
        index
        for index in range(lo, hi)
        if getattr(frames[index], "seg", "") == "look_hold"
    ]
    ent = (getattr(scene, "entities", None) or {}).get(command["obj_key"])
    try:
        bbox = scene.object_bbox(command["obj_key"]) if command["obj_key"] else None
    except Exception:  # noqa: BLE001
        bbox = None
    samples = [
        sample
        for index in (holds or list(range(lo, hi)))
        if 0 <= index < len(frames)
        for sample in [
            _label_frame_sample(
                ent,
                bbox,
                frames[index],
                index,
                boxes,
                (
                    camera_outputs[index]
                    if camera_outputs is not None and index < len(camera_outputs)
                    else None
                ),
            )
        ]
        if sample is not None
    ]
    readable_indices = [sample["index"] for sample in samples if sample["readable"]]
    longest = _longest_consecutive(readable_indices)
    if not samples:
        return {
            "i": command["i"],
            "obj": command["obj_key"],
            "verdict": lv.NO_LABEL,
            "hold_frames": len(holds),
            "readable_frames": 0,
            "longest_readable_run": 0,
            "readable": False,
        }
    sample = samples[len(samples) // 2]
    label = sample["label"]
    assessment = sample["assessment"]
    _decal_probe(scene, ent, label)
    return {
        "i": command["i"],
        "obj": command["obj_key"],
        "frame": sample["index"],
        "verdict": assessment.verdict,
        "dist_m": round(assessment.dist_m, 3),
        "px": round(assessment.px, 1),
        "glyph_px": round(assessment.glyph_px, 1),
        "required_px": round(assessment.required_px, 1),
        "off_normal_deg": round(assessment.off_normal_deg, 1),
        "text_roll_deg": round(assessment.text_roll_deg, 1),
        "yaw_deg": round(assessment.yaw_deg, 1),
        "pitch_deg": round(assessment.pitch_deg, 1),
        "cam_yaw_deg": round(sample["cam_yaw"], 1),
        "cam_pitch_deg": round(sample["cam_pitch"], 1),
        "in_frame": bool(sample["in_frame"]),
        "authored_focus": bool(sample["authored_focus"]),
        "focal_mm": round(float(sample["focal_mm"]), 2),
        "effective_zoom": round(float(sample["effective_zoom"]), 3),
        "zoom_suppressed_reasons": list(sample["zoom_suppressed_reasons"]),
        "occluder": assessment.occluder or None,
        "hold_frames": len(holds),
        "readable_frames": len(readable_indices),
        "longest_readable_run": longest,
        "readable": longest >= min_run,
    }


def _label_visibility_report(scene, frames, cmds, camera_outputs=None) -> dict:
    """Measure every label-reading hold and enforce the required observation contract."""
    required = [
        str(key)
        for key in (
            (getattr(scene, "behavior_task", None) or {}).get("required_label_looks")
            or []
        )
    ]
    label_keys = set(required) | {
        str(key) for key in (getattr(scene, "label_prim_paths", None) or {}).keys()
    }
    looks = [
        command
        for command in cmds
        if command["prim"] == "look_at"
        and command["frame_start"] is not None
        and str(command.get("obj_key") or "") in label_keys
    ]
    if not looks:
        return {
            "n_looks": 0,
            "n_readable": 0,
            "required": required,
            "missing": required,
            "contract_ok": not required,
            "min_consecutive_readable_frames": 12,
            "looks": [],
        }
    min_run = 12
    initial = getattr(scene, "_episode_baseline", None) or {}
    open_objs = set(initial.get("open_objs") or ())
    rows = []
    for command in cmds:
        prim = str(command.get("prim") or "")
        key = command.get("obj_key")
        if command.get("ok", True) and prim in ("open", "open_door") and key:
            open_objs.add(str(key))
        elif command.get("ok", True) and prim in ("close", "close_door") and key:
            open_objs.discard(str(key))
        if (
            prim == "look_at"
            and command.get("frame_start") is not None
            and str(key or "") in label_keys
        ):
            boxes = se._label_occluder_boxes(scene, str(key or ""), open_objs=open_objs)
            rows.append(
                _label_look_row(
                    scene,
                    frames,
                    command,
                    boxes,
                    min_run,
                    camera_outputs,
                )
            )
    n_ok = sum(1 for r in rows if r.get("readable"))
    missing, unreadable = _label_contract_failures(required, rows)
    contract_ok = not missing and not unreadable
    print(
        f"[step3r] label-visibility: {n_ok}/{len(rows)} look(s) READABLE  "
        + " ".join(
            f"#{r['i']}:{r['obj']}={r['verdict']}"
            + (
                f"({r.get('glyph_px', 0):.1f}px glyph)"
                if r.get("glyph_px")
                else f"({r.get('px', 0):.0f}px)"
            )
            + f"[run {r.get('longest_readable_run', 0)}]"
            + (
                f"[cam {r['cam_yaw_deg']:+.0f},{r['cam_pitch_deg']:+.0f}"
                + ("" if r.get("in_frame") else " OFF-FRAME")
                + "]"
                if "cam_yaw_deg" in r
                else ""
            )
            for r in rows
        )
    )
    return {
        "n_looks": len(rows),
        "n_readable": n_ok,
        "required": required,
        "missing": missing,
        "unreadable": unreadable,
        "contract_ok": contract_ok,
        "min_consecutive_readable_frames": min_run,
        "looks": rows,
    }


def _label_contract_failures(
    required: list[str], rows: list[dict]
) -> tuple[list[str], list[str]]:
    by_object: dict[str, list[dict]] = {}
    for row in rows:
        by_object.setdefault(str(row.get("obj")), []).append(row)
    missing = [key for key in required if key not in by_object]
    unreadable = [
        key
        for key in required
        if key in by_object
        and not any(bool(row.get("readable")) for row in by_object[key])
    ]
    return missing, unreadable


def _compile_visual_state_events(scene, frames) -> tuple:
    """Resolve non-human visual changes and prove requested cuts happen off camera."""
    import label_visibility as lv
    import surface_visual_effects as sve
    import visual_state_events as vse

    dialogue = dict(
        ((getattr(scene, "behavior_task", None) or {}).get("episode_dialogue") or {})
    )
    entity_paths = {
        str(key): str(entity.prim_path)
        for key, entity in (getattr(scene, "entities", None) or {}).items()
        if getattr(entity, "prim_path", None)
    }
    effect_paths = sve.materialize_surface_effects(scene, dialogue)
    events = vse.compile_visual_state_events(
        dialogue,
        list(getattr(scene, "source_command_timeline", None) or []),
        dict(getattr(scene, "label_prim_paths", None) or {}),
        entity_paths,
        len(frames),
        effect_paths=effect_paths,
    )
    report = []
    for event in events:
        visible = []
        applied_in_clip = event.frame < len(frames)
        if event.require_offscreen and applied_in_clip:
            sample_index = max(0, event.frame - 1)
            for key in event.target_keys:
                ent = (getattr(scene, "entities", None) or {}).get(key)
                if ent is None:
                    continue
                try:
                    bbox = scene.object_bbox(key)
                except Exception:  # noqa: BLE001
                    bbox = None
                boxes = se._label_occluder_boxes(scene, key)
                sample = _label_frame_sample(
                    ent, bbox, frames[sample_index], sample_index, boxes
                )
                if (
                    sample is not None
                    and sample["in_frame"]
                    and sample["assessment"].verdict != lv.OCCLUDED
                ):
                    visible.append(key)
        row = event.manifest_row(visible_objects=visible)
        row["applied_in_clip"] = applied_in_clip
        report.append(row)
        if event.require_offscreen and applied_in_clip and visible:
            raise RuntimeError(
                "visual state event would change a visible target in the FPV: "
                f"frame={event.frame} visible={visible}"
            )
        value = (
            f"tint={event.tint}"
            if isinstance(event, vse.EntityTintEvent)
            else f"visibility={event.visible}"
        )
        print(
            f"[visual-event] frame={event.frame} type={event.event_type} "
            f"{value} targets={list(event.target_keys)} offscreen_ok={not visible}"
        )
    scene.visual_state_event_report = report
    return events


def _execution_pose_transfer_schedule(scene, compiled_commands) -> dict[int, tuple]:
    """Resolve variant pose transfers to source-command indices before motion planning."""
    import visual_state_events as vse

    dialogue = dict(
        ((getattr(scene, "behavior_task", None) or {}).get("episode_dialogue") or {})
    )
    rows = [
        dict(row)
        for row in dialogue.get("visual_state_events") or ()
        if row.get("pose_from")
    ]
    if not rows:
        return {}
    entity_paths = {
        str(key): str(entity.prim_path)
        for key, entity in (getattr(scene, "entities", None) or {}).items()
        if getattr(entity, "prim_path", None)
    }
    timeline = [
        {
            "source": command.source,
            "frame_start": command.index,
            "frame_end": command.index + 1,
        }
        for command in compiled_commands
    ]
    events = vse.compile_visual_state_events(
        {"visual_state_events": rows},
        timeline,
        {},
        entity_paths,
        len(compiled_commands),
    )
    schedule: dict[int, list] = {}
    for event in events:
        schedule.setdefault(event.frame - 1, []).append(event)
    return {index: tuple(values) for index, values in schedule.items()}


def _place_variant_at_source(stage, scene, event) -> None:
    """Align replacement centers and bases while preserving each target's own orientation."""
    for target_key, target_path, source_key, source_path in zip(
        event.entities,
        event.paths,
        event.pose_from_entities,
        event.pose_from_paths,
        strict=True,
    ):
        source_pose = se._read_world_col(stage, source_path)
        target_pose = se._read_world_col(stage, target_path)
        if source_pose is None or target_pose is None:
            raise RuntimeError(
                "visual state pose transfer references a missing prim: "
                f"{source_path} -> {target_path}"
            )
        source_base = scene.object_pos(source_key)
        target_base = scene.object_pos(target_key)
        target_pose[:3, 3] += source_base - target_base
        if not se._write_world_col(stage, target_path, target_pose):
            raise RuntimeError(f"could not move visual state variant {target_path}")
        scene.set_object_rest_pose(target_key, target_pose)


def _apply_visual_state_events(stage, events, frame_index: int, state: dict) -> None:
    """Apply declared visibility and material state for one rendered frame."""
    if not events:
        return
    import visual_state_events as vse
    from pxr import UsdGeom

    visibility = (
        events.visibility_at(frame_index)
        if isinstance(events, vse.OnlineVisualStateEvents)
        else vse.visibility_at(events, frame_index)
    )
    if visibility != state.get("visibility"):
        for path, visible in visibility.items():
            prim = stage.GetPrimAtPath(path)
            if not (prim and prim.IsValid()):
                raise RuntimeError(f"visual state event references missing prim {path}")
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
            )
        state["visibility"] = visibility

    tints = (
        events.tints_at(frame_index)
        if isinstance(events, vse.OnlineVisualStateEvents)
        else vse.tints_at(events, frame_index)
    )
    if tints == state.get("tints"):
        return
    import behavior_import as bi

    tokens = state.setdefault("tint_restore_tokens", {})
    previous = dict(state.get("tints") or {})
    for path in reversed(tuple(previous)):
        if path in tints and tints[path] == previous[path]:
            continue
        token = tokens.pop(path, None)
        if token is not None:
            bi.restore_entity_tint(stage, token)
    for path, tint in tints.items():
        if previous.get(path) == tint:
            continue
        tokens[path] = bi.apply_entity_tint(stage, path, tint)
    state["tints"] = tints


def _reset_visual_state_events(stage, state: dict) -> None:
    """Undo reversible material transitions and discard replay state."""
    if not state:
        return
    import behavior_import as bi

    for token in reversed(tuple(state.get("tint_restore_tokens", {}).values())):
        bi.restore_entity_tint(stage, token)
    state.clear()


def _apply_visual_state_pose_transfers(
    stage, scene, events, frame_index: int, state: dict
) -> None:
    """Move a newly revealed replacement entity onto its source entity exactly once."""
    if not events:
        return
    import visual_state_events as vse

    applied = state.setdefault("pose_transfers", set())
    for index, event in enumerate(events):
        if index in applied or event.frame > frame_index:
            continue
        if not isinstance(event, vse.EntityVisibilityEvent) or not event.pose_from_entities:
            continue
        applied.add(index)
        _place_variant_at_source(stage, scene, event)


def _digest_json(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _proactive_queries_by_frame(sparse_capture) -> dict[int, tuple[dict, ...]]:
    if not sparse_capture or (
        sparse_capture.get("contract_version") != PROACTIVE_PHASE_CAPTURE_VERSION
    ):
        return {}
    windows = {
        str(row.get("decision_id")): row
        for row in sparse_capture.get("windows", ())
        if isinstance(row, dict)
    }
    result: dict[int, list[dict]] = {}
    for raw_query in sparse_capture.get("observation_queries", ()):
        query = dict(raw_query)
        decision_id = str(query.get("decision_id") or "")
        window = windows.get(decision_id)
        if window is None:
            raise ValueError(f"observation query {decision_id!r} has no capture window")
        if any(
            query.get(key) != window.get(key)
            for key in ("observation_phase", "capture_position")
        ):
            raise ValueError(
                f"observation query {decision_id!r} disagrees with its window"
            )
        query["frame_indices"] = list(window.get("frame_indices", ()))
        for index in query["frame_indices"]:
            result.setdefault(int(index), []).append(query)
    return {index: tuple(rows) for index, rows in result.items()}


def _entity_visibility(scene, entity_id: str, pose, boxes, width: int, height: int):
    if entity_id == "wearer":
        return {
            "entity_id": entity_id,
            "visible": None,
            "visibility_status": "egocentric_camera_source",
        }
    try:
        box = scene.entity_aabbs((entity_id,)).get(entity_id)
    except Exception:  # noqa: BLE001
        box = None
    if box is None or pose is None:
        return {
            "entity_id": entity_id,
            "visible": False,
            "visibility_status": "geometry_unavailable",
        }
    import label_visibility as lv

    minimum, maximum = (np.asarray(value, dtype=np.float64) for value in box)
    center = 0.5 * (minimum + maximum)
    span = max(float(np.max(maximum - minimum)), 0.01)
    assessment = lv.assess(
        np.asarray(pose.eye),
        np.eye(3),
        lv.Label(center, span),
        boxes,
        image_w=width,
        image_h=height,
        min_px=1,
        focal_mm=float(pose.focal_mm),
        camera_look=np.asarray(pose.look),
        camera_up=np.asarray(pose.up if pose.up is not None else (0.0, 0.0, 1.0)),
        self_radius_m=0.02,
        enforce_torso_limits=False,
        min_distance_m=0.0,
    )
    visible = bool(assessment.in_frame and not assessment.occluder)
    return {
        "entity_id": entity_id,
        "visible": visible,
        "visibility_status": "visible" if visible else assessment.verdict,
        "occluder": assessment.occluder or None,
    }


def _record_proactive_visibility(
    queries_by_frame,
    samples,
    frame_index: int,
    scene,
    pose,
    boxes,
    width: int,
    height: int,
) -> None:
    for query in queries_by_frame.get(frame_index, ()):
        predicate = query.get("predicate") or {}
        entities = [
            _entity_visibility(scene, str(entity_id), pose, boxes, width, height)
            for entity_id in predicate.get("entity_ids", ())
        ]
        measurable = [row for row in entities if row["entity_id"] != "wearer"]
        samples.setdefault(str(query["query_digest"]), []).append(
            {
                "frame_index": frame_index,
                "related_entities": entities,
                "all_related_visible": bool(measurable)
                and all(row["visible"] is True for row in measurable),
            }
        )


def _proactive_frame_observations(sparse_capture, samples) -> list[dict]:
    observations = []
    windows = {
        str(row.get("decision_id")): row
        for row in sparse_capture.get("windows", ())
        if isinstance(row, dict)
    }
    for query in sparse_capture.get("observation_queries", ()):
        query_digest = str(query.get("query_digest") or "")
        rows = list(samples.get(query_digest, ()))
        window = windows.get(str(query.get("decision_id") or "")) or {}
        frame_indices = [int(value) for value in window.get("frame_indices", ())]
        if [row["frame_index"] for row in rows] != frame_indices:
            raise ValueError(
                f"query {query_digest!r} is missing delivered frame evidence"
            )
        entity_ids = list((query.get("predicate") or {}).get("entity_ids", ()))
        entity_visibility = _aggregate_entity_visibility(entity_ids, rows)
        visible_indices = [
            row["frame_index"] for row in rows if row["all_related_visible"]
        ]
        minimum = int(query.get("minimum_visible_frames", 1))
        body = {
            "proof_kind": "entity_visibility_only",
            "decision_id": query.get("decision_id"),
            "query_digest": query_digest,
            "predicate": copy.deepcopy(query.get("predicate")),
            "frame_indices": frame_indices,
            "visible_frame_indices": visible_indices,
            "minimum_visible_frames": minimum,
            "visible": len(visible_indices) >= minimum,
            "related_entity_visibility": entity_visibility,
            "visibility_basis": "live_entity_aabb_frustum_occlusion",
        }
        body["observation_digest"] = _digest_json(body)
        observations.append(body)
    return observations


def _aggregate_entity_visibility(entity_ids, rows) -> list[dict]:
    result = []
    for entity_id in entity_ids:
        samples = [
            entity
            for row in rows
            for entity in row["related_entities"]
            if entity["entity_id"] == entity_id
        ]
        indices = [
            row["frame_index"]
            for row, sample in zip(rows, samples)
            if sample["visible"] is True
        ]
        result.append(
            {
                "entity_id": entity_id,
                "visible_frame_indices": indices,
                "visible_frame_count": len(indices),
                "visibility_statuses": [
                    sample["visibility_status"] for sample in samples
                ],
            }
        )
    return result


def _episode_extra(
    args,
    scene,
    env,
    goals,
    frames,
    n_fail,
    t_start,
    mode,
    exit_path,
    camera_outputs=None,
) -> dict:
    """The execution half of the episode row: counts, the per-command log, timings.

    Field names follow ``simulator.physical_world/bench_episode.py``'s consumer contract. Every value is read
    defensively, and anything this process cannot produce is ``None`` rather than a default -- so a
    reader never mistakes "not measured" for "clean". ``env.reports`` holds ONE final verdict per
    ``Goal`` (``semantic_env._record`` supersedes a retried command), so ``commands`` and
    ``failure_histogram`` are already de-duplicated and agree with ``n_fail``."""
    import collections
    import time as _time

    cmds: list[dict] = []
    hist: collections.Counter = collections.Counter()
    # WHY, not just WHAT. ``reason`` is a CATEGORY and several unrelated mechanisms share each one --
    # `path_blocked` alone is emitted by the stand scan, the door pre-pass, the ego lock and the router,
    # and telling them apart previously meant fetching a 250-600 kB ``render.log`` per row.
    # The detail string is already computed by ``step_one``; carrying it here is what makes "judge from
    # the MANIFEST" self-sufficient. Read defensively so an env without the map still produces rows.
    _details = getattr(env, "report_detail", None) or {}
    _spans = getattr(env, "report_frame_span", None) or {}
    _states = getattr(env, "report_state", None) or {}
    for i, (goal, reason) in enumerate(getattr(env, "reports", None) or []):
        name = getattr(reason, "value", str(reason))
        meta = getattr(goal, "meta", None) or {}
        span = _spans.get(id(goal))
        cmds.append(
            {
                "i": i,
                "prim": getattr(goal.prim, "value", str(goal.prim)),
                "target_key": getattr(goal, "target_key", None),
                "obj_key": getattr(goal, "obj_key", None),
                "child": meta.get("child"),
                "parent": meta.get("parent"),
                "child_point": meta.get("child_point"),
                "child_point_idx": meta.get("child_point_idx"),
                "parent_point": meta.get("parent_point"),
                "parent_point_idx": meta.get("parent_point_idx"),
                "orientation_error_deg": meta.get("orientation_error_deg"),
                "orientation_error_axis": meta.get("orientation_error_axis"),
                "injected_error_kind": meta.get("injected_error_kind"),
                "implicit_action": meta.get("_implicit_action"),
                "surface": meta.get("surface"),
                "placement": meta.get("_placement_result")
                or meta.get("_placement_intent"),
                "failure_phase": meta.get("_placement_failure_phase"),
                "frame_start": None if span is None else int(span[0]),
                "frame_end": None if span is None else int(span[1]),
                "state": _states.get(id(goal)),
                "reason": name,
                "detail": (_details.get(id(goal)) or None) if name != "ok" else None,
                "ok": name == "ok",
            }
        )
        if name != "ok":
            hist[name] += 1
    attributed, n_cascade = _attributed_histogram(cmds)
    missing = getattr(scene, "assets_missing", None)
    return {
        "mode": mode,
        "exit_path": exit_path,
        "oracle_steps": len(goals),
        "n_frames": len(frames),
        "n_fail": int(n_fail),
        "commands": cmds,
        "commands_total": len(cmds),
        "label_visibility": _label_visibility_report(
            scene, frames, cmds, camera_outputs
        ),
        "source_command_timeline": list(
            getattr(scene, "source_command_timeline", None) or []
        ),
        "episode_dialogue": dict(
            (
                (getattr(scene, "behavior_task", None) or {}).get("episode_dialogue")
                or {}
            )
        ),
        "visual_state_events": list(
            getattr(scene, "visual_state_event_report", None) or []
        ),
        "failure_histogram": dict(hist),
        # THE SAME FAILURES, CHARGED TO WHAT CAUSED THEM. Purely additive: ``failure_histogram`` above
        # is untouched, so every historical comparison still holds and no verdict moves. See
        # ``_attributed_histogram`` for why the raw one systematically over-counts `place_unstable`.
        "failure_histogram_attributed": attributed,
        "n_cascade_failures": n_cascade,
        "build_reachability": getattr(scene, "build_reachability", None),
        # STAGING VERIFICATION -- the measurement the engine already made and always discarded. Objects
        # that did not stay on the support they were staged on, worst fall first. REPORTED ONLY: it is
        # NOT wired to ``init_episode_valid``, so nothing that passes today starts failing.
        "staging_offtarget": _staging_offtarget(),
        "n_staging_offtarget": len(_staging_offtarget()),
        "n_assets_skipped": None if missing is None else len(missing),
        "wall_time": round(_time.perf_counter() - t_start, 1),
        "rtx_warmed": bool(getattr(make_fpv_products, "_rtx_warmed", False)),
        "argv": list(sys.argv),
        # Which stand-picking rule this episode ran under. Recorded because "the flag was off" is
        # otherwise unfalsifiable from the artifacts, and a default-off flag that is accidentally ON is
        # a routine way to lose an A/B: an out-of-scope control that silently ran the treatment reads
        # as a null result. False on every episode with no ego verb.
        "ego_route_gate": _ego_route_gate(scene),
        # Gate (b0)'s router retry, on trial and reporting itself. `fired` = times the connectivity
        # PROXY refused every candidate and the real router was asked instead; `rescued` = times the
        # router found a stand the proxy had refused. As of 2026-08-11: 14 fired, 0 rescued, so the
        # block has never yet changed an outcome. Shipped as an instrument rather than deleted because
        # the disagreement band is structurally real (`init_validation.py --nav-proxy-control`) and only
        # unpopulated in THIS portfolio -- a growing portfolio either populates it or justifies the
        # revert. **Never credit the retry with a recovered row while `rescued` is 0.**
        "stand_router_retry": getattr(scene, "stand_router_retry", None)
        or {"fired": 0, "rescued": 0},
        "intervention_contract": getattr(scene, "intervention_contract", None),
    }


def _task_goal_record(
    bt: dict,
    goal_preds: list,
    relations: set,
    baseline_relations: set | None = None,
) -> dict:
    """Score one ``behavior_task`` into a manifest-ready dict carrying a PER-SUB-GOAL boolean.

    ``sub_goals`` is the canonical scoreable array: the positive goals in goal order, then the negative
    constraints in constraint order, each stamped with its position. A caller can therefore attribute a
    pass/fail to the individual unit that earned it (per-capability scoring, per-item diagnosis) instead
    of only seeing the conjunction. Positives read ``{idx, pred, subject, target, ok}``, negatives
    ``{idx, kind, label, ok, offenders}``.

    TWO denominators are published rather than one overloaded number: ``progress``/``n`` count earned
    positive goals only, excluding predicates already true at the episode baseline, and
    ``progress_denominator`` says so out loud; ``n_sub_goals``/``n_sub_goals_ok`` count every unit.
    ``met`` is the full conjunction (earned goals AND baseline constraints AND negatives).

    A PROHIBITION IS A GATE, NEVER A NUMERATOR. ``not_inside`` / ``no_share_group`` / ``capacity`` are
    true at t = 0 by construction, so counting them in ``progress_sub_goals`` paid an agent that did
    nothing at all a measured 33-50 % (mean 37.6 %) of the score. ``progress_sub_goals`` is therefore
    the ACHIEVEMENT ratio (positives plus ``all_placed``, which must be earned), and holding every
    prohibition is a precondition for receiving any of it -- breaking one withholds the lot.
    ``n_sub_goals``/``n_sub_goals_ok`` stay the RAW counts of every unit, for display; do not divide
    them, that ratio is the defect. The two classes are also published separately so a reader sees both.
    """
    import task_score

    baseline_goal_indices = (
        task_score.satisfied_goal_indices(goal_preds, baseline_relations)
        if baseline_relations is not None
        else ()
    )
    score = task_score.score_task(
        goal_preds,
        list(bt.get("neg_constraints") or []),
        relations,
        aggregate_goals=list(bt.get("aggregate_goals") or []),
        baseline_goal_indices=baseline_goal_indices,
    )
    return {
        "task": bt.get("task"),
        "container": bt.get("container"),
        **score.to_manifest_dict(),
    }


def _assistant_intervention_command_index(
    compiled_commands, behavior_task
) -> int | None:
    """Resolve the earliest source-command index following an authored intervention."""
    dialogue = dict((behavior_task or {}).get("episode_dialogue") or {})
    turns = [
        row for row in dialogue.get("assistant_turns") or () if isinstance(row, dict)
    ]
    if not turns:
        return None
    resolved = []
    for turn in turns:
        command = str(turn.get("command") or "")
        occurrence = int(turn.get("occurrence") or 1)
        seen = 0
        for compiled in compiled_commands:
            if str(compiled.source) != command:
                continue
            seen += 1
            if seen == occurrence:
                resolved.append(int(compiled.index))
                break
    return min(resolved) if resolved else None


def _write_episode_manifest(
    out_dir,
    tag,
    args,
    init_valid,
    compliance,
    task_goal=None,
    extra: dict | None = None,
) -> None:
    """Write the honest per-episode verdict to ``<tag>_manifest.json`` as TWO INDEPENDENT SCORES.

    ``task_goal`` -- did the assistant drive the world into the commanded state. ALWAYS computed and
    ALWAYS recorded; no physical measurement suppresses it.

    ``render_grade`` -- the graded picture quality (``clean`` / ``marginal`` / ``dirty``), produced by
    ``action_check.grade_render`` from the human penetration depth as a fraction of the offending
    capsule's radius. It is a LABEL, never a veto: a ``dirty`` episode is still rendered and still
    scored (this benchmark grades reasoning, not physical fidelity).

    ``episode_valid`` -- is the recorded WORLD STATE trustworthy enough to score ``task_goal`` against.
    That is ``init_episode_valid`` (the start pose was certified, so the task was not made unfairly
    impossible) AND ``objects_clean`` (no two placed objects interpenetrate, so the settled rests the
    goal check reads are physically realizable). It is deliberately NO LONGER ``human_clean``: welding
    those together made one sampled frame of a body capsule brushing furniture condemn a whole
    correctly-solved episode (measured: ``episode_valid == human_clean`` in 93/93 stored manifests, 24
    of which passed). Human-vs-furniture contact is now reported by ``render_grade`` alone.

    A real serializer (json) handles quoting -- never hand-format structured output."""
    import json

    comply = None
    if compliance is not None:
        _sampled = int(compliance.get("human_sampled", 0))
        _run_f = int(compliance.get("human_longest_run_frames", 0))
        comply = {
            "ok": bool(compliance["ok"]),
            "human_clean": bool(compliance["human_clean"]),
            "human_hit_frames": int(compliance["human_hit_frames"]),
            "human_by_seg": {
                str(k): int(v) for k, v in compliance["human_by_seg"].items()
            },
            "human_sampled": _sampled,
            "human_dirty_frac": round(
                float(compliance.get("human_dirty_frac", 0.0)), 6
            ),
            "human_longest_run_frames": _run_f,
            "human_longest_run_s": round(_run_f / float(FPS), 4),
            # (capsule, furniture_owner) identities of the worst frame -- NOT a depth.
            "human_worst": [
                [str(cap), str(owner)]
                for cap, owner in compliance.get("human_worst", [])
            ],
            "human_hit_idx": [int(i) for i in compliance.get("human_hit_idx", [])],
            "sample_every": int(compliance.get("sample_every", 0)),
            "joint_subset": str(compliance.get("joint_subset", "")),
            # Bisected against the SAME convex hulls the boolean gate queries. Saturates at the capsule
            # radius, so when human_pen_saturated is true this is a LOWER BOUND, never a maximum. It is
            # NOT comparable with replay_compliance's world-AABB depth (a different, unbounded quantity).
            "human_max_pen_mm": round(
                float(compliance.get("human_max_pen_mm", 0.0)), 1
            ),
            "human_pen_saturated": bool(compliance.get("human_pen_saturated", False)),
            "human_pen_mm_by_seg": {
                str(k): round(float(v), 1)
                for k, v in (compliance.get("human_pen_mm_by_seg") or {}).items()
            },
            "human_worst_pen_desc": str(compliance.get("human_worst_pen_desc", "")),
            "human_worst_pen_frame": compliance.get("human_worst_pen_frame"),
            # The SCALE-FREE depth the grade keys on (millimetres alone cannot be thresholded: a
            # 140 mm pelvis and a 52 mm forearm do not share a budget).
            "human_worst_frac": round(
                float(compliance.get("human_worst_frac", 0.0)), 4
            ),
            "human_worst_frac_capsule": str(
                compliance.get("human_worst_frac_capsule", "")
            ),
            "human_worst_frac_desc": str(compliance.get("human_worst_frac_desc", "")),
            "human_worst_frac_frame": compliance.get("human_worst_frac_frame"),
            "render_grade": str(compliance.get("render_grade", "")),
            "render_pass": bool(compliance.get("render_pass", False)),
            "render_grade_reason": str(compliance.get("render_grade_reason", "")),
            "objects_clean": bool(compliance["objects_clean"]),
            "objects": {str(k): str(v) for k, v in compliance["objects"].items()},
            "obj_pen": int(len(compliance["obj_pen"])),
        }
    ex = extra or {}
    _nf = ex.get("n_fail")
    goal_scored = task_goal is not None and int(task_goal.get("n", 0) or 0) > 0
    # Pointer to the episode's DIALOGUE sidecar (``simulator.physical_world/dialogue_log.py``), when a harness recorded
    # one. Two capabilities -- should the assistant have spoken, and could the person act on what it said
    # -- leave no trace in the final world state, so they are unmeasurable from this file alone. ``null``
    # is the honest reading "no dialogue was recorded here", which the scorer reports as UNCOMPUTABLE; it
    # is never a zero. A scripted render with no assistant legitimately has none.
    _dialogue = os.path.join(out_dir, f"{tag}_dialogue.json")
    _dialogue_rel = os.path.basename(_dialogue) if os.path.exists(_dialogue) else None
    # World-state trust ONLY -- see this function's docstring. Human penetration is graded, not vetoed.
    physics_valid = bool(init_valid) and (comply is None or comply["objects_clean"])
    render_grade = None if comply is None else comply["render_grade"]
    rec = {
        "schema_version": 2,
        "episode_id": f"{args.scene}__{(task_goal or {}).get('task') or 'none'}"
        f"__{getattr(args, 'seed', None)}",
        "tag": tag,
        "task": (task_goal or {}).get("task"),
        # Set by the batch node script; absent on a local run -> null, never a fabricated id.
        "task_id": os.environ.get("TASK_ID") or None,
        "code_id": os.environ.get("CODE_ID") or engine_code_id(),
        "scene": args.scene,
        "seed": getattr(args, "seed", None),
        "fps": FPS,
        "init_episode_valid": bool(init_valid),
        "compliance": comply,
        "task_goal": task_goal,
        "goal_scored": goal_scored,
        "dialogue_log": _dialogue_rel,
        "episode_valid": physics_valid,
        # None ONLY when no compliance pass ran (the scene opts out) -- never "clean" by default.
        "render_grade": render_grade,
        "verdict": {
            "physics_valid": physics_valid,
            "execution_complete": None if _nf is None else int(_nf) == 0,
            "goal_scorable": goal_scored,
            "render_grade": render_grade,
            # Owned by bench_episode._verdict, which also weighs the shipped products; this process
            # cannot see them, so it reports the inputs and leaves the aggregate unset.
            "benchmark_usable": None,
        },
    }
    rec.update(ex)
    if ex.get("intervention_contract") is not None:
        import bookcase_intervention

        rec["intervention"] = bookcase_intervention.evaluate_orientation_recovery(
            list(ex.get("commands") or []),
            task_goal,
            ex["intervention_contract"],
        )
        rec["verdict"]["intervention_valid"] = bool(rec["intervention"]["valid"])
    path = os.path.join(out_dir, f"{tag}_manifest.json")
    try:
        with open(path, "w") as f:
            f.write(json.dumps(rec, indent=2) + "\n")
        print(
            f"[step3r][MANIFEST] episode_valid={rec['episode_valid']} "
            f"render_grade={render_grade} goal_scored={goal_scored} -> {path}"
        )
    except OSError as exc:
        print(f"[step3r][MANIFEST] write failed ({exc})")


def _merge_manifest_extra(out_dir, tag, extra: dict) -> None:
    """ADD keys to an already-written ``<tag>_manifest.json``.

    The physics verdict is published before the render products exist, so anything measured on a
    rendered frame (auto-exposure) can only be recorded afterwards. This merges rather than rewrites so
    the published verdict can never be contradicted by a later pass."""
    import json

    path = os.path.join(out_dir, f"{tag}_manifest.json")
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[step3r][MANIFEST] merge skipped ({exc})")
        return
    rec.update(extra)
    try:
        with open(path, "w") as f:
            f.write(json.dumps(rec, indent=2) + "\n")
        print(f"[step3r][MANIFEST] merged {sorted(extra)} -> {path}")
    except OSError as exc:
        print(f"[step3r][MANIFEST] merge write failed ({exc})")


# The SMPL+H body block ``J[0:22]``. ``init_validation._CORE_J`` (the judged subset) is 0..17; 18..21 are
# the elbows/wrists ``physics_human._bone_endpoints`` needs to synthesise the hand tips. A subset wider
# than the body block would additionally need ``_RHAND_MID_MCP`` (J[40]) -- not dumped.
_REPLAY_NJOINT: int = 22
_REPLAY_SCHEMA: int = 1


def _write_replay_dump(out_dir, tag, args, scene, frames, jtraj) -> None:
    """Persist the three inputs of the compliance judgement to ``<tag>_replay.json.gz``.

    ``simulator.physical_world/replay_compliance.py replay`` re-judges a stored episode offline (no Isaac boot), which
    is the only affordable regression gate for the clipping/scoring stack -- ``--step3`` is blind to it
    (``run_dry3`` passes neither ``scene_boxes`` nor ``nudge_state``). Everything written here is already
    resident in memory: no query is issued and no verdict is recomputed.

    ``jtraj`` MUST be the trajectory the gate judged, never a re-realize -- the point is to reproduce the
    verdict. Joints are in metres rounded to mm (three orders below any threshold under discussion, and
    it gzips well)."""
    import gzip
    import json

    cw = getattr(scene, "cworld", None)
    if cw is None or jtraj is None or not frames:
        return
    keep = set(cw.scene_paths)  # the non-floor owners human_overlap passes to PhysX
    segs: list[list] = []
    for fr in frames:
        s = getattr(fr, "seg", "") or "?"
        if segs and segs[-1][0] == s:
            segs[-1][1] += 1
        else:
            segs.append([s, 1])
    dump = {
        "schema": _REPLAY_SCHEMA,
        "tag": tag,
        "scene": args.scene,
        "seed": getattr(args, "seed", None),
        "fps": FPS,
        "joints": [
            [[round(float(c), 3) for c in J[j]] for j in range(_REPLAY_NJOINT)]
            for J in jtraj
        ],
        "segs": segs,
        "obstacles": [
            [
                o,
                round(float(cx), 3),
                round(float(cy), 3),
                round(float(hx), 3),
                round(float(hy), 3),
                round(float(bz), 3),
                round(float(tz), 3),
            ]
            for (o, cx, cy, hx, hy, bz, tz) in cw.footprints
            if o in keep
        ],
    }
    path = os.path.join(out_dir, f"{tag}_replay.json.gz")
    try:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(dump))
        print(
            f"[step3r][REPLAY] {len(jtraj)} poses x {len(dump['obstacles'])} obstacles -> {path}"
        )
    except OSError as exc:
        print(f"[step3r][REPLAY] write failed ({exc})")


# =================================================================================================
# INIT non-penetration validation (P1-P4): give ALL furniture static colliders, then place the human
# at a NON-PENETRATING start + validate every action stand pose via the exact human-vs-scene gate.
# =================================================================================================
# Top-level /World children that are NEVER scene obstacles (the human, the carried object, the room
# handled separately for its floor, the physics/camera/light infra). Everything else placed under
# /World is furniture the human must not penetrate -> gets a static collider.
_NON_FURNITURE = {
    "Room",
    "HumanProxy",
    "Obj",
    "Body",
    "FpvCam",
    "TpvCam",
    "PhysicsScene",
    "PhysicsMaterials",
    "OfficeLights",
    "Lights",
    "DomeLight",
    "GraspJoint",
}


def _apply_scene_colliders(stage, ph) -> list[str]:
    """GENERALIZE make_static_collider beyond the hardcoded prim list to ALL placed furniture/surfaces.

    Walk the top-level ``/World`` children; every prim that is not human/object/infra (see
    ``_NON_FURNITURE``) becomes a STATIC collider, choosing the approximation PER CLASS (the open_risks
    tradeoff): a SURFACE-like bare box (Cube gprim: shelf/table/counter/cabinet-base) uses exact
    triangle collision (approximation ignored for a gprim -- a box is exact, no concavity to fill); a
    referenced-mesh SOLID (the Sektion cabinet, a chair/sofa) uses 'convexDecomposition' so a body
    standing into it is caught (accepting that decomposition FILLS an open cabinet's interior -> a
    conservative false-POSITIVE, the safe direction). The room FLOOR is added as an exact collider so a
    body sinking through it would be caught, but a body standing ON it is not blocked (top_z ~= 0 keeps
    it out of the occupancy grid). Idempotent (CollisionAPI.Apply is get-or-create). Returns the paths
    that were tagged (for logging)."""
    from pxr import UsdGeom, UsdLux

    tagged: list[str] = []
    world = stage.GetPrimAtPath("/World")
    if not (world and world.IsValid()):
        return tagged
    for child in world.GetChildren():
        name = child.GetName()
        if name in _NON_FURNITURE:
            continue
        # A UsdLux light (illumination source) is NEVER a physical obstacle -- skip it so an added
        # SphereLight/DomeLight (e.g. BehaviorLightWork hovering over the object cluster) is not turned
        # into a phantom floor-blocking collider. General/type-based, no scene names: real lamp/fixture
        # FURNITURE is a Mesh/Xform asset (b_*_lamp/b_room_light with collision meshes), NOT a UsdLux-typed
        # prim, so it is unaffected and still becomes a collider.
        if child.HasAPI(UsdLux.LightAPI) or "Light" in str(child.GetTypeName()):
            continue
        # Native procedural boxes are exact. Referenced meshes retain the conservative boundingCube
        # fallback, while ``make_static_collider`` recognizes BEHAVIOR assets and uses their authored
        # low-poly collision meshes as exact static triangles. Exactness matters for open cabinets,
        # L-shaped worktops and cubicles: convex approximations fill their usable void with phantom solid.
        approx = "none" if child.IsA(UsdGeom.Cube) else "boundingCube"
        ph.make_static_collider(child, approximation=approx)
        tagged.append(str(child.GetPath()))
    # Room floor: exact collider (a body must not sink through it), kept out of the occupancy grid by
    # its ~z=0 top. Guarded by IsValid so a scene without the room is unaffected.
    floor = stage.GetPrimAtPath("/World/Room/Floor")
    if floor and floor.IsValid():
        ph.make_static_collider(floor, approximation="none")
        tagged.append("/World/Room/Floor")
    return tagged


def _nearest_support_centroid(stage, support_prims, xy):
    """Collision-inclusive ``(cx, cy, top_z)`` of the support prim nearest ``xy`` (None-triple if there is
    none). Used by the graspable settle: if an object's placed XY overhangs the surface's real solid it
    would fall to the floor, so it is stepped toward this centroid until it rests ON the surface. The
    BBoxCache includes 'guide' so OG's per-link collision hulls (purpose=guide) contribute.
    """
    from pxr import Usd, UsdGeom

    best = (None, None, None)
    best_d = 1e30
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    for sp in support_prims or []:
        prim = stage.GetPrimAtPath(sp)
        if not (prim and prim.IsValid()):
            continue
        rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        if not (mx[0] >= mn[0]):
            continue
        cx, cy, top = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), float(mx[2])
        d = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2
        if d < best_d:
            best_d, best = d, (float(cx), float(cy), top)
    return best


def _support_visual_top(stage, support_prims, xy):
    """VISUAL world-bbox top z (purposes default+render, so the guide-purpose collision meshes are excluded)
    of the support prim nearest ``xy`` -- the height a graspable's VISIBLE base should rest on. Returns None
    if there is no support prim. The physics settle grounds the object's convexHull collider on the support's
    COLLISION top, which can sit above/below the visible desktop; seating the render z against THIS visual top
    puts the visible object flush on the visible surface. Reads the STATIC, visible support prim (reliable),
    not the dynamic graspable."""
    from pxr import Usd, UsdGeom

    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    best_top, best_d = None, 1e30
    for sp in support_prims or []:
        prim = stage.GetPrimAtPath(sp)
        if not (prim and prim.IsValid()):
            continue
        rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        if not (mx[0] >= mn[0]):
            continue
        cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        d = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2
        if d < best_d:
            best_d, best_top = d, float(mx[2])
    return best_top


def _build_behavior_scene(stage, app, scene_name):
    """Wrapper so a BEHAVIOR scene builder matches the (stage, app) -> 5-tuple builder signature."""
    import behavior_import

    return behavior_import.build_behavior(stage, app, scene_name)


# BEHAVIOR-1K scenes rendered with the walk->grab->walk->put loop (behavior_import.build_behavior_wgp):
# residential interiors with real work surfaces. Registered in _scene_builders as --scene bwgp_<lower>.
# MANDATE --seed for the init non-pen gate. Bake a scene's models first (behavior_bake.py --scene NAME).
_BEHAVIOR_WGP_SCENES = (
    "Ihlen_0_int",
    "Ihlen_1_int",
    "Rs_int",
    "Benevolence_0_int",
    "Beechwood_0_int",
    "Beechwood_1_int",
    "Merom_0_int",
    "Merom_1_int",
    "Pomaria_1_int",
    "Wainscott_0_int",
)


def _build_behavior_wgp_scene(stage, app, scene_name):
    """Wrapper: a BEHAVIOR walk-grab-put scene as a (stage, app) -> 5-tuple builder."""
    import behavior_import

    return behavior_import.build_behavior_wgp(stage, app, scene_name)


def _build_behavior_task_scene(stage, app, scene_name, task_name):
    """Wrapper: a 2026 BEHAVIOR-1K CHALLENGE task instance as a (stage, app) -> 5-tuple builder."""
    import behavior_import

    return behavior_import.build_behavior_task(stage, app, scene_name, task_name)


def _build_behavior_generic_scene(stage, app, scene_name, *, seed=0):
    """Wrapper: an AUTO-GENERATED pick-and-place task on any admitted scene, as a builder.

    Unlike ``bt_*`` this needs no challenge task instance -- build_behavior_generic picks a work surface
    and a container out of the scene's own geometry and injects the graspables. That is what lifts the
    ceiling from the 7 scenes with BDDL goals to every scene the offline manifest admits.
    """
    import behavior_import

    return behavior_import.build_behavior_generic(
        stage,
        app,
        scene_name,
        seed=seed,
    )


# Remaining Tier-A challenge tasks (pick-place: graspables -> a surface / open-top or portable container),
# wired so each is reachability-checkable (--navcheck), gate-able (--check) and renderable as --scene
# bt_<task>. Task names are globally unique -> these keys are DISTINCT from the 5 explicit bt_* in
# _scene_builders, so those 5 stay byte-identical. CAVEAT on current build_behavior_task coverage (see
# behavior_import.build_behavior_task): it needs an inside(obj,container) goal. Tasks tagged (ontop-only)
# have NO container -> the builder RAISES (surfaces as a build error in --navcheck) until an ontop-place
# branch lands; tasks tagged (mixed) additionally have ontop objects the v1 builder routes INTO the
# container (needs per-object routing to render correctly). All still register for the reachability probe.
_BEHAVIOR_TASK_SCENES = (
    ("house_double_floor_upper", "boxing_books_up_for_storage"),
    ("house_double_floor_upper", "organizing_art_supplies"),  # mixed
    ("house_double_floor_upper", "re_shelving_library_books"),
    ("house_double_floor_upper", "tidying_living_room"),  # mixed
    ("house_double_floor_lower", "bringing_in_wood"),  # ontop-only
    ("house_double_floor_lower", "cleaning_up_branches_and_twigs"),  # mixed
    ("house_double_floor_lower", "make_rose_centerpieces"),  # mixed
    ("house_double_floor_lower", "picking_up_trash"),
    ("house_single_floor", "carrying_out_garden_furniture"),  # ontop-only
    ("house_single_floor", "collecting_childrens_toys"),
    ("house_single_floor", "organizing_school_stuff"),  # mixed
    ("house_single_floor", "picking_up_toys"),
    ("house_single_floor", "sorting_vegetables"),
    ("hotel_suite_large", "dispose_of_glass"),
    ("hotel_suite_large", "tidying_bathroom"),  # mixed
    ("restaurant_diner", "putting_dirty_dishes_in_sink"),
    # Tier-B door-container tasks (object goes INSIDE / needs an openable fridge/cabinet/drawer opened
    # then closed). Registered so they are testable once the generic openable open/close driver lands.
    ("house_single_floor", "boxing_food_after_dinner"),
    ("house_single_floor", "bringing_water"),
    ("house_single_floor", "clean_up_your_desk"),
    ("house_single_floor", "outfit_a_basic_toolbox"),
    ("house_single_floor", "preparing_lunch_box"),
    ("house_single_floor", "putting_dishes_away_after_cleaning"),
    ("house_single_floor", "storing_food"),
    ("house_double_floor_lower", "bringing_paper_to_recycling"),
    ("house_double_floor_lower", "carrying_in_groceries"),
    ("house_double_floor_lower", "clearing_food_from_table_into_fridge"),
    ("house_double_floor_lower", "loading_the_car"),
    ("house_double_floor_lower", "put_together_a_basic_pruning_kit"),
    ("house_double_floor_lower", "rearranging_kitchen_furniture"),
    ("Rs_int", "store_batteries"),
    ("Rs_int", "store_honey"),
    ("restaurant_diner", "store_produce"),
)


def build_cut(stage, app):
    """Step-4 CONTACT demo: walk to a knife on a counter, GRAB it, walk to a vegetable, CUT it (a raster
    of chop strokes) -> declares CUT(veg). The TOOL (/World/Obj) is a marker proxy (scene.obj_def='pen');
    the vegetable is a static box the ego camera frames via Frame.gaze. Symbolic-state (no visible split);
    reuses the FROZEN grab + the shared warp hand-nav. Mirrors build_highreach's room+shelf pattern.
    """
    scene = SceneState(floor_z=0.0, start_xy=(0.0, 1.6))
    _clean_room(stage, app)
    # Deep counter whose NEAR edge is at y=-0.45; the veg + knife sit AT the front edge so the human
    # stands CLEAR of the plank (foot_half stand-back, like highreach) and reaches forward-down to the
    # front lip -- the torso stays BEHIND the edge and the arm passes ABOVE the thin top (no AABB graze).
    cpath, cx, cy, chx, cthy, ctop = _plank_table(
        stage,
        "Counter",
        0.0,
        -0.45,
        top=0.82,
        plank=(0.7, 0.40, 0.05),
        color=(0.50, 0.42, 0.34),
    )
    # vegetable: a small green box resting at the counter front edge (the CUT target the FPV frames).
    veg_xy, veg_h = (-0.10, -0.45), 0.08
    _make_box(
        stage,
        "/World/Veg",
        (0.10, 0.10, veg_h),
        (veg_xy[0], veg_xy[1], ctop + veg_h * 0.5),
        color=(0.25, 0.65, 0.25),
    )
    foot = {
        "foot_half": (0.25, cthy),
        "top_z": ctop,
        "clearance": 0.38,
    }  # stand clear of the counter near-face
    scene.register(
        "veg",
        Role.SURFACE,
        xy=veg_xy,
        z=ctop + veg_h,
        approach_yaw=0.0,
        half=(0.08, 0.08),
        foot_xy=(veg_xy[0], cy),
        **foot,
    )
    # The knife (tool) sits on a TALL thin stand (~shoulder height, BACK legs) so the GRAB is a clean
    # reach-UP with free space below -- the proven highreach pattern. (Grabbing off the LOW counter leans
    # the FROZEN grab body into the plank, which the strict AABB gate flags; the veg CUT stays on the low
    # counter and is already clean.)
    ks_top, ks_x = (
        1.14,
        0.95,
    )  # clear of the (now 0.7 m-wide) counter's x<=0.35 right edge
    _make_box(
        stage,
        "/World/KnifeStand",
        (0.35, 0.30, 0.05),
        (ks_x, -0.62, ks_top),
        color=(0.50, 0.42, 0.34),
    )
    for lx in (-0.13, 0.13):  # thin BACK legs, out of the reach path
        _make_box(
            stage,
            f"/World/KnifeStandLeg{'L' if lx < 0 else 'R'}",
            (0.05, 0.05, ks_top),
            (ks_x + lx, -0.62 - 0.12, ks_top * 0.5),
            color=(0.42, 0.36, 0.30),
        )
    scene.register(
        "knife",
        Role.GRASPABLE,
        xy=(ks_x, -0.50),
        z=ks_top + 0.03,
        approach_yaw=0.0,
        foot_xy=(ks_x, -0.62),
        foot_half=(0.20, 0.15),
        top_z=ks_top,
        clearance=0.42,
    )
    scene.obj_def = "cup"  # radial-wrap tool proxy (no knife USD in load_object_def; a cup grabs cleanly)
    scene.behavior_support_prims = [
        cpath,
        "/World/KnifeStand",
    ]  # settle colliders (counter + knife stand)
    program = "[Walk] <knife> (1), [Grab] <knife> (1), [Walk] <veg> (1), [Cut] <veg> (1) <knife> (1)"
    tpv_look = (0.0, -0.45, 0.95)
    return scene, {}, program, tpv_look, None


def build_inspect(stage, app):
    """Step-4 CAMERA demo: walk up to a target on a pedestal, LOOK_AROUND, LOOK_AT it, then SQUAT to
    inspect it low. No object interaction -- proves the new camera/body verbs drive the egocentric camera
    (gaze aim + eye height) and pass the non-penetration gate (the human stands clear, never reaches in).
    """
    scene = SceneState(floor_z=0.0, start_xy=(0.0, 2.4))
    _clean_room(stage, app)
    ped_top = 0.60
    _make_box(
        stage,
        "/World/Pedestal",
        (0.40, 0.40, ped_top),
        (0.0, -0.6, ped_top * 0.5),
        color=(0.45, 0.40, 0.35),
    )
    _make_box(
        stage,
        "/World/Target",
        (0.16, 0.16, 0.16),
        (0.0, -0.6, ped_top + 0.08),
        color=(0.80, 0.30, 0.20),
    )
    scene.register(
        "target",
        Role.SURFACE,
        xy=(0.0, -0.6),
        z=ped_top + 0.16,
        approach_yaw=0.0,
        half=(0.10, 0.10),
        foot_xy=(0.0, -0.6),
        foot_half=(0.20, 0.20),
        top_z=ped_top + 0.16,
        clearance=0.55,
    )
    program = "[Walk] <target> (1), [LookAround], [LookAt] <target> (1), [Squat]"
    tpv_look = (0.0, -0.6, 0.7)
    return scene, {}, program, tpv_look, None


def build_grabnat(stage, app):
    """Step-4 NATURAL grasp demo: a SOURCE table holds three differently-shaped objects -- a tall CYLINDER
    (side-wrap grip), a round BALL (cage grip) and a thin MARKER (pinch) -- and the human picks each up with
    a SHAPE-APPROPRIATE grip and places it on a DESTINATION table a short walk away, all driven by one
    action string. ``scene.live_wrap=True`` turns on the per-shape live wrap (grasp_select.classify_shape
    seeds the contact wrap); every other scene stays canned/byte-identical. Normal (0.75 m) table height.

    Objects are separate prims at /World/Obj_<key> (scene.grasp_objs) so several can be grabbed in one run;
    each rides rig._obj_path, swapped per frame via Frame.obj_path. Table planks feed the object settle via
    scene.behavior_support_prims (the general static-collider hook) + the reach gate via the entities'
    foot_half."""
    scene = SceneState(floor_z=0.0, start_xy=(0.0, 1.6))
    scene.live_wrap = True  # opt IN to the per-shape natural grasp (produce_pick builds the SDF + seeds)
    scene.natural_arm = True  # naturalness P1: seed the reach CCD from a relaxed elbow-down arm (see execute)
    # FPV: aim CLOSER + more DOWN during the walk so the low tables/hands stay framed (not the far wall).
    scene.fpv_walk_ahead = 1.6
    scene.fpv_walk_downtilt = (
        0.95  # aim the walk FPV further DOWN (more of the floor/workspace)
    )
    # FPV: LOWER the eye + bias the grasp aim down so the hand + object fill the frame (see the detail),
    # not sit at the bottom edge under a head-height eye.
    scene.fpv_eye_drop = 0.32
    scene.fpv_manip_aim_drop = (
        0.22  # aim the grasp FPV further DOWN so it looks down at the hands
    )
    _clean_room(stage, app)

    # SOURCE + DEST plank tables (normal 0.75 m height), a short walk apart along X. near_edge_y=-0.395 is
    # the proven clutter/grab stand-back anchor (the human's torso clears the front edge on a low reach).
    src_path, scx, scy, shx, sthy, stop = _plank_table(
        stage,
        "SrcTable",
        cx=-0.6,
        near_edge_y=-0.395,
        top=0.75,
        color=(0.50, 0.38, 0.28),
    )
    dst_path, dcx, dcy, dhx, dthy, dtop = _plank_table(
        stage,
        "DstTable",
        cx=0.7,
        near_edge_y=-0.395,
        top=0.75,
        color=(0.30, 0.34, 0.42),
    )
    for _ in range(3):
        app.update()

    obj_y = (
        -0.455
    )  # 6 cm back from the front edge (like clutter's carried mug), reachable
    # (key, source_x, dest_x, build_params, color). Shapes chosen to hit CYL / ROUND / PINCH distinctly.
    specs = [
        (
            "can",
            -0.88,
            0.42,
            {"shape": "cylinder", "radius": 0.036, "height": 0.13},
            (0.85, 0.20, 0.18),
        ),
        ("ball", -0.60, 0.70, {"shape": "sphere", "radius": 0.045}, (0.93, 0.53, 0.12)),
        (
            "marker",
            -0.32,
            0.98,
            {"shape": "cylinder", "radius": 0.014, "height": 0.10},
            (0.15, 0.35, 0.85),
        ),
    ]
    scene.grasp_objs = [
        {"key": k, "path": f"/World/Obj_{k}", "build": b, "color": c}
        for (k, _sx, _dx, b, c) in specs
    ]
    scene.behavior_support_prims = [
        src_path,
        dst_path,
    ]  # static colliders so the objects settle on top
    # foot_xy is centered on EACH object's own X (a full-depth column), so stand_pose_for stands the human
    # directly IN FRONT of the object it is grabbing (a straight forward reach) rather than at the shared
    # table center -- otherwise the right arm reaches sideways to the off-center objects and misses.
    for k, sx, _dx, _b, _c in specs:
        scene.register(
            k,
            Role.GRASPABLE,
            xy=(sx, obj_y),
            z=stop
            + 0.06,  # pre-drop hint; the env-init settle rests it on the table top
            approach_yaw=0.0,
            prim_path=f"/World/Obj_{k}",
            foot_xy=(sx, scy),
            foot_half=(0.18, sthy),
            top_z=stop,
            clearance=0.42,
        )
    for (
        k,
        _sx,
        dx,
        _b,
        _c,
    ) in specs:  # destination drop spots (distinct X so the objects don't stack)
        scene.register(
            f"d_{k}",
            Role.SURFACE,
            xy=(dx, obj_y),
            z=dtop,
            approach_yaw=0.0,
            foot_xy=(dx, dcy),
            foot_half=(0.18, dthy),
            top_z=dtop,
            clearance=0.42,
        )
    program = ", ".join(
        f"[Walk] <{k}> (1), [Grab] <{k}> (1), [Walk] <d_{k}> (1), [Put] <{k}> (1) <d_{k}> (1)"
        for (k, _sx, _dx, _b, _c) in specs
    )
    print(
        "[step3r] grabnat: 3 shapes (can=CYL, ball=ROUND, marker=PINCH); live_wrap=ON"
    )
    # TPV in FRONT of the human (they face -Y toward the tables, so -Y is the front) + elevated 3/4, so
    # the arm motion is clearly visible instead of a back view.
    tpv_look = (0.15, -0.45, 0.9)
    tpv_eye = (-0.8, -2.9, 1.55)
    return scene, {}, program, tpv_look, tpv_eye


def _scene_builders(seed: int = 0) -> dict:
    """Return every admitted raw-world builder plus the frozen TaskPack builders.

    Raw worlds deliberately use their exact manifest names as public registry
    keys.  Their generic builder is deterministic in ``(scene, seed)``; binding
    both values here prevents the caller's episode seed from being silently
    replaced by the generic builder's default.
    """
    from simulator.physical_world import task_world_builder
    from simulator.world_catalog.scene_manifest import admitted_scenes

    builders = {
        scene_name: (
            lambda name: lambda stage, app: _build_behavior_generic_scene(
                stage,
                app,
                name,
                seed=seed,
            )
        )(scene_name)
        for scene_name in admitted_scenes()
    }
    builders.update(task_world_builder.spec_scene_builders())
    return builders


def _run_initcheck(app, stage, ph, args) -> int:
    """OFFLINE INIT non-penetration harness (V0/V1/V2; no RTX): for SEEDS, randomly place the human
    (place_start) + (for object scenes) the objects, then assert the EXACT human-vs-scene gate reads 0
    penetrating pairs at the START pose AND at EVERY action stand pose. A scene the sampler cannot place
    cleanly is honestly reported episode_valid=False with the reason. Returns 0 iff every seed is either
    clean or correctly flagged.

    This is the sole place the P1-P4 pieces run end-to-end on a real (random) scene: build furniture
    colliders (P1) -> free-cell start (P2) -> per-target stand-pose validate+adjust (P3) -> bounded
    caps + episode_valid contract (P4). It boots the same scene builders --check/render use, so it
    exercises the ACTUAL RL path, not a toy."""
    import init_validation as iv
    import object_physics as ophys

    rig = core.load_family("side_wrap")
    _build, _otw = core.load_object_def("cup")
    core.set_object(rig, _build, _otw)
    try:
        core.derive_rest_place_arms(rig)
    except Exception as exc:  # noqa: BLE001
        print(f"[initcheck] derive_rest_place_arms failed ({exc}); using canonical arm")

    builders = _scene_builders(seed=getattr(args, "seed", 0))
    scene, controllers, program, _tpv_look, _tpv_eye = builders[args.scene](stage, app)

    ph.build_physics_scene(stage)
    ph.set_scene_timestep(stage, 60)
    tagged = _apply_scene_colliders(stage, ph)
    for _ in range(3):
        app.update()
    print(
        f"[initcheck] scene={args.scene}: {len(tagged)} furniture/floor colliders applied"
    )

    # The action targets whose stand pose must be validated = the WALK targets of the program (every
    # entity the human stands at to act). Derived from the program so it is program-general.
    stand_keys = _stand_targets(program, scene)
    print(f"[initcheck] scene={args.scene}: stand targets = {stand_keys}")

    seeds = [int(s) for s in str(args.seeds).split(",")] if args.seeds else [0, 1, 2]
    # Init physics + ONE step so the furniture colliders are registered for the lag-free scene-query
    # overlap gate (place_start / validate_stand_pose sample overlaps from J -- no proxy USD, no stepping
    # per query, so re-posing a kinematic body never lags the gate).
    sim = ophys.get_sim()
    sim.step(render=False)

    # DISCRIMINATION self-check (scene-level positive control): the gate must not be a no-op. Pose the
    # idle body standing AT the center of a floor-blocking furniture footprint (a guaranteed torso/leg
    # penetration) and ASSERT the overlap gate reports pairs; then confirm the same body FAR away reads
    # clean. Without this, an "all clean" run could be silently vacuous.
    disc_ok = _initcheck_discrimination(ph, stage, scene, rig, iv)
    print(
        f"[initcheck] scene={args.scene}: gate discrimination self-check {'OK' if disc_ok else 'FAIL'}"
    )

    all_ok = disc_ok
    seed_cells: list = []
    for seed in seeds:
        ok, cell, summary = _initcheck_one_seed(
            ph, sim, stage, scene, rig, seed, stand_keys, iv
        )
        all_ok = all_ok and ok
        seed_cells.append(cell)
        print(summary)
    # Independent-seed diversity: with >=2 seeds the accepted start cells should be DISTINCT (the eval
    # path samples the random-init distribution, not one deterministic near-hint cell). Reported, and (for
    # a scene with room for >=2 free cells) REQUIRED, so "3 seeds" is a genuine 3-sample robustness proof.
    distinct = {c for c in seed_cells if c is not None}
    diversity_ok = len(seeds) < 2 or len(distinct) >= min(len(seeds), 2)
    print(
        f"[initcheck] scene={args.scene}: seed start cells={seed_cells} distinct={len(distinct)}/"
        f"{len(seeds)} -> diversity {'OK' if diversity_ok else 'FAIL (seeds converged)'}"
    )
    all_ok = all_ok and diversity_ok
    print(
        f"[initcheck] scene={args.scene}: {'PASS' if all_ok else 'FAIL'} over seeds {seeds}"
    )
    _exit_process(0 if all_ok else 2)
    return 0 if all_ok else 2


def _run_navcheck(app, stage, ph, args) -> int:
    """OFFLINE NAV-GRID doorway/connectivity probe (no RTX): build the scene + colliders, place the human
    with the SAME place_start + validate_stand_pose the seeded render uses, build the nav grid (furniture
    occupancy + doorway-preserving wall-triangle raster + two-sided SDF), carve the init-gate-validated
    start + stand cells free (proven non-penetrating human positions that sit at furniture edges), dump
    occ/sdf PNG+ascii to fpv_outputs/, and ASSERT the start reaches every stand through free space
    (doorways open). Returns 0 iff every stand is reachable. Additive: runs only under --navcheck.
    """
    import os

    import init_validation as iv
    import nav_grid
    import nav_planner
    import object_physics as ophys

    builders = _scene_builders(seed=getattr(args, "seed", 0))
    scene, _controllers, program, _tpv_look, _tpv_eye = builders[args.scene](stage, app)

    ph.build_physics_scene(stage)
    ph.set_scene_timestep(stage, 60)
    tagged = _apply_scene_colliders(stage, ph)
    for _ in range(3):
        app.update()
    sim = ophys.get_sim()
    sim.step(render=False)

    rig = core.load_family("side_wrap")
    _b, _o = core.load_object_def("cup")
    core.set_object(rig, _b, _o)
    try:
        core.derive_rest_place_arms(rig)
    except Exception as exc:  # noqa: BLE001
        print(f"[navcheck] derive_rest_place_arms failed ({exc})")

    # Same validated init positions the seeded render plans on, so the probe matches the render path.
    # diverse=False: rank free cells by proximity to the interior near-hint (else the emptiest cell is
    # the sealed-off exterior void, which is correctly unreachable from the interior stands).
    start_xy, _sstats = iv.place_start(
        ph, sim, stage, scene, rig, seed=0, fallback_xy=scene.start_xy, diverse=False
    )
    start_xy = (float(start_xy[0]), float(start_xy[1]))
    _stand_keys = _stand_targets(program, scene)
    stands = {}
    for k in _stand_keys:
        res = iv.validate_stand_pose(ph, sim, stage, scene, rig, k)
        stands[k] = (float(res.stand_xy[0]), float(res.stand_xy[1]))

    wall_n = len(getattr(scene, "wall_prims", []) or [])
    stands_str = ", ".join(f"{k}=({v[0]:.2f},{v[1]:.2f})" for k, v in stands.items())
    print(
        f"[navcheck] scene={args.scene}: {len(tagged)} colliders; wall_prims={wall_n}; "
        f"start=({start_xy[0]:.2f},{start_xy[1]:.2f}); stands=[{stands_str}]"
    )

    # Use the SAME adaptive grid build the seeded render uses (largest inflation keeping start + every
    # stand connected), then plan the actual A* polyline for each leg so the dump shows the real route.
    # Task scenes (precise_nav): ONE fixed-inflation grid for BOTH stand re-pick AND the verdict (see
    # _validate_init_for_render). Re-pick only stands islanded on it; connected ones untouched. Non-task
    # scenes keep the adaptive build (byte-identical).
    grid, infl, nav_ok, nav_tot = nav_grid.build_nav_grid_adaptive(
        stage, scene, start_xy, list(stands.values())
    )
    if getattr(scene, "precise_nav", False) and nav_ok < nav_tot:
        # Re-pick only islanded stands on a reachable-within-reach grid; ADOPT only if strictly more connect
        # (regression-safe; mirrors _validate_init_for_render so navcheck reflects the render).
        _contacts = []
        for k in _stand_keys:
            try:
                _c = iv._entity_contact(scene, k)
            except Exception:  # noqa: BLE001
                _c = None
            if _c is not None:
                _contacts.append((float(_c[0]), float(_c[1])))
        g2, infl2, _o2, _t2 = nav_grid.build_task_nav_grid(
            stage, scene, start_xy, _contacts or list(stands.values())
        )
        gv2 = nav_grid.passable_view(g2)
        st2 = dict(stands)
        for k in _stand_keys:
            if iv._stand_connected(gv2, start_xy, st2[k]):
                continue
            res = iv.validate_stand_pose(
                ph, sim, stage, scene, rig, k, nav_grid=gv2, start_xy=start_xy
            )
            st2[k] = (float(res.stand_xy[0]), float(res.stand_xy[1]))
        nav_grid.carve_free(g2, [start_xy, *st2.values()], radius=0.30)
        gvv2 = nav_grid.passable_view(g2)
        ok2 = sum(1 for v in st2.values() if nav_grid.connected(gvv2, start_xy, v))
        if ok2 > nav_ok:
            stands, grid, infl, nav_ok, nav_tot = st2, g2, infl2, ok2, len(st2)
            _ss = ", ".join(f"{k}=({v[0]:.2f},{v[1]:.2f})" for k, v in stands.items())
            print(
                f"[navcheck] nav-aware re-pick improved -> [{_ss}] reachable={nav_ok}/{nav_tot}"
            )
    free = int((~grid.occ).sum())
    ncomp, big = nav_grid.free_stats(grid)
    print(
        f"[navcheck] grid {grid.nx}x{grid.ny} cell={grid.cell} free={free}/{grid.occ.size} "
        f"({100.0 * free / max(1, grid.occ.size):.0f}%) components={ncomp} largest={big:.0%} "
        f"chosen_inflate={infl:.2f} reachable={nav_ok}/{nav_tot}"
    )

    # --door-open PROOF: probe the DOORS-PASSABLE view (door cells freed). A closed-door goal that is
    # BLOCKED on the raw occ (today's failure) becomes CONNECTED once the door can be opened. Without
    # --door-open the probe is the raw occ (doors blocking) -> reproduces the current behaviour.
    door_open = bool(getattr(args, "door_open", False))
    n_doors = (
        int((grid.door_id >= 0).sum())
        if getattr(grid, "door_id", None) is not None
        else 0
    )
    probe = nav_grid.passable_view(grid) if door_open else grid
    if door_open:
        print(
            f"[navcheck] --door-open: probing DOORS-PASSABLE view ({n_doors} door cells freed)"
        )
    reach = nav_grid.flood_free(probe, probe.world_to_cell(*start_xy))
    all_ok = True
    path_all: list = []
    prev = start_xy
    for k, gxy in stands.items():
        conn = nav_grid.connected(probe, start_xy, gxy)
        leg = ""
        if conn:
            try:
                pth = nav_planner.plan_path(probe, prev, gxy)
                path_all.extend(pth)
                leg = f"-> planned {len(pth)} legs"
                prev = gxy
            except Exception as exc:  # noqa: BLE001
                leg = f"-> PLAN FAILED ({exc})"
                all_ok = False
        else:
            all_ok = False
        print(
            f"[navcheck]   start -> {k}({gxy[0]:.2f},{gxy[1]:.2f}): "
            f"{'CONNECTED' if conn else 'BLOCKED'} {leg}"
        )

    outdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fpv_outputs"
    )
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(
        outdir, f"navgrid_{args.scene}" + ("_dooropen" if door_open else "")
    )
    nav_grid.dump_grid(
        probe,
        out,
        start_xy=start_xy,
        goals=list(stands.values()),
        path_xy=path_all,
        reachable=reach,
    )
    print(f"[navcheck] dumped {out}.png / .txt")
    print(
        f"[navcheck] scene={args.scene}: {'PASS' if all_ok else 'FAIL (a stand is unreachable)'}"
    )
    _exit_process(0 if all_ok else 2)
    return 0 if all_ok else 2


def _initcheck_discrimination(ph, stage, scene, rig, iv) -> bool:
    """Prove the overlap gate DISCRIMINATES (not a no-op): the idle body standing at the center of a
    floor-blocking furniture footprint must read PENETRATING, and far from all furniture must read
    CLEAN. Returns True iff both hold (or there is no furniture to test, in which case it is vacuous).
    """
    footprints = iv.collidable_footprints(stage)
    scene_paths = iv._non_floor_paths(footprints, floor_z=scene.floor_z)
    blocking = [
        (cx, cy)
        for _o, cx, cy, hx, hy, bot, top in footprints
        if iv._blocks_floor(bot, top, scene.floor_z, 0.15)
    ]
    if not blocking or not scene_paths:
        return True  # no furniture -> nothing to discriminate (vacuously ok)
    stand_z = iv._stand_z_for(rig, scene.floor_z)
    arm_idle = getattr(rig, "arm_rest", rig.arm_grasp)
    # Stand at the first blocking footprint center -> torso/legs inside it.
    cx, cy = blocking[0]
    J_in = iv.human_proxy_J(rig, arm_idle, (cx, cy), 0.0, stand_z)
    pen = ph.human_scene_overlap(stage, J_in, scene_paths)
    # Stand far outside every footprint.
    fx = max(cx for cx, cy in blocking) + 5.0
    J_out = iv.human_proxy_J(rig, arm_idle, (fx, cy), 0.0, stand_z)
    clr = ph.human_scene_overlap(stage, J_out, scene_paths)
    print(
        f"[initcheck][disc] body AT furniture ({cx:.2f},{cy:.2f}) overlap_pairs={len(pen)} (want>0); "
        f"body FAR ({fx:.2f},{cy:.2f}) overlap_pairs={len(clr)} (want 0)"
    )

    # ARM-vs-furniture discrimination: stand AT the furniture center with an arm-EXTENDED reach keyframe
    # (torso + arm inside the furniture) and gate the FULL body minus a small contact sphere at a POINT
    # FAR from the arm. The arm plows through the furniture body away from that sphere -> the arm-out gate
    # must FIRE. Then exclude a HUGE sphere covering the whole body -> it must DROP to 0. Proves the arm
    # gate both catches arm-through-furniture AND respects the intended contact region (not a no-op / not
    # constant-fire). Uses a GRASPABLE/openable key if the program has one; else vacuously ok.
    arm_ok = True
    key = next(
        (
            k
            for k in scene.entities
            if scene.entities[k].role
            in (Role.GRASPABLE, Role.SURFACE, Role.CONTAINER, Role.OPENABLE, Role.SEAT)
        ),
        None,
    )
    if key is not None:
        Jr = iv._arm_extended_keyframe(rig, scene, key, (cx, cy), 0.0, stand_z)
        if Jr is not None:
            far_pt = np.array(
                [cx + 100.0, cy, 0.0], dtype=np.float64
            )  # sphere far from the arm
            arm_hits = ph.human_scene_overlap(
                stage, Jr, scene_paths, exclude_sphere=(far_pt, iv._CONTACT_SPHERE_R)
            )
            arm_masked = ph.human_scene_overlap(
                stage,
                Jr,
                scene_paths,
                exclude_sphere=(np.array([cx, cy, stand_z + 0.9]), 100.0),
            )
            arm_ok = len(arm_hits) > 0 and len(arm_masked) == 0
            print(
                f"[initcheck][disc] ARM extended AT furniture: overlap w/ far contact-sphere="
                f"{len(arm_hits)} (want>0, arm-through-furniture caught); w/ all-body sphere="
                f"{len(arm_masked)} (want 0, contact region respected) -> {'OK' if arm_ok else 'FAIL'}"
            )
    return len(pen) > 0 and len(clr) == 0 and arm_ok


def _stand_targets(program: str, scene) -> list[str]:
    """The entity keys the human STANDS at during the program (walk-to targets) -- their stand poses are
    what INIT must validate. Read from the compiled goals: every 1-arg manipulation's obj_key + every
    [Put] surface + the resolved walk targets (a plain [Walk] <goal> counts too)."""
    keys: list[str] = []
    for line in command_grammar.split_program(program):
        parsed = command_grammar.parse_line(line)
        if parsed is None:
            continue
        verb, params = parsed
        prim = command_grammar._VERB.get(verb)
        # The egocentric-locomotion verbs name no entity at all -- their optional argument is a scalar
        # (a count of paces / degrees), never something _resolve could ground.
        if prim is None or not params or prim in command_grammar.LOCOMOTION_PRIMS:
            continue
        if prim is command_grammar.Primitive.PLACE and len(params) >= 2:
            key = command_grammar._resolve(scene, *params[1])
        else:
            key = command_grammar._resolve(scene, *params[0])
        if key not in keys:
            keys.append(key)
    return keys


def _initcheck_one_seed(
    ph, sim, stage, scene, rig, seed, stand_keys, iv
) -> tuple[bool, str]:
    """Run place_start + validate_stand_pose for one seed; a placement is ACCEPTABLE iff it is exact-
    clean (``confirm_pairs == 0``) OR it was honestly flagged ``episode_valid=False`` -- the RL
    contract (never a silently-accepted penetrating placement). Returns (seed_ok, log_text).
    """

    def _accept(stats: dict) -> bool:
        return stats["confirm_pairs"] == 0 or not stats["episode_valid"]

    # diverse=True: drop the near-hint bias so the 3 seeds sample GENUINELY different start cells (the
    # independent-seed robustness proof the critic asked for), still non-penetrating by construction.
    start_xy, sstats = iv.place_start(
        ph, sim, stage, scene, rig, seed=seed, fallback_xy=scene.start_xy, diverse=True
    )
    seed_ok = _accept(sstats)
    lines = [
        f"[initcheck][seed {seed}] START xy=({start_xy[0]:.2f},{start_xy[1]:.2f}) "
        f"cell={sstats['start_cell']} "
        f"grid_free={sstats['grid_free_cells']} draws={sstats['start_draws']} "
        f"confirm_pairs={sstats['confirm_pairs']} "
        f"fallback={sstats['fallback_taken']} episode_valid={sstats['episode_valid']} "
        f"{'OK' if _accept(sstats) else 'VIOLATION'}"
    ]
    for key in stand_keys:
        res = iv.validate_stand_pose(ph, sim, stage, scene, rig, key)
        st = res.stats
        seed_ok = seed_ok and _accept(st)
        lines.append(
            f"[initcheck][seed {seed}] STAND '{key}' xy=({res.stand_xy[0]:.2f},{res.stand_xy[1]:.2f}) "
            f"yaw={res.yaw:.0f} rung={st['ladder_step_used']} confirm_pairs={st['confirm_pairs']} "
            f"arm_pairs={st['arm_pairs']} arm_out={st.get('arm_out_pairs', 0)} "
            f"fallback={st['fallback_taken']} episode_valid={st['episode_valid']} "
            f"{'OK' if _accept(st) else 'VIOLATION'}"
        )
    lines.append(
        f"[initcheck][seed {seed}] start_cell={sstats['start_cell']} -> {'PASS' if seed_ok else 'FAIL'}"
    )
    return seed_ok, sstats.get("start_cell"), "\n".join(lines)


def _init_gate_rig():
    """The rig the INIT GATE poses the human with: side-wrap family + cup object def + derived rest arms.

    ``derive_rest_place_arms`` runs two batched-torch ``_ik_arm(n=6000)`` solves (seconds), so
    ``setup_episode`` -- which builds exactly this rig before it calls the gate -- hands its copy down and
    the gate only falls back to this builder when called without one."""
    rig = core.load_family("side_wrap")
    # Define a grip seat (obj_T_in_wrist) so warp_arm_to_seat can land the hand on a handle.
    _build, _otw = core.load_object_def("cup")
    core.set_object(rig, _build, _otw)
    try:
        core.derive_rest_place_arms(rig)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[step3r][seed-init] derive_rest_place_arms failed ({exc}); canonical arm"
        )
    return rig


def _rig_field_diff(a, b) -> list[str]:
    """Attribute names where two ``SmplhRig`` objects differ (arrays via ``np.array_equal``)."""

    def eq(x, y) -> bool:
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.array_equal(np.asarray(x), np.asarray(y))
        if isinstance(x, dict) and isinstance(y, dict):
            return sorted(x) == sorted(y) and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return len(x) == len(y) and all(eq(u, v) for u, v in zip(x, y))
        try:
            return bool(x == y)
        except Exception:  # noqa: BLE001 (USD/mesh handles need not be comparable)
            return x is y

    names = sorted(set(vars(a)) | set(vars(b)))
    return [n for n in names if not eq(getattr(a, n, None), getattr(b, n, None))]


def _check_init_gate_rig(rig) -> None:
    """``ISAAC_RIG_EQUIV_CHECK=1``: rebuild the gate rig and PROVE the handed-down one is identical to it,
    field by field, instead of asserting it by reading the code. Also reports what the skipped rebuild
    costs. Off by default (it re-does exactly the work the hand-down exists to avoid).
    """
    import time as _time

    _t0 = _time.perf_counter()
    fresh = _init_gate_rig()
    dt = _time.perf_counter() - _t0
    diff = _rig_field_diff(rig, fresh)
    n_fields = len(set(vars(rig)) | set(vars(fresh)))
    verdict = "IDENTICAL" if not diff else "DIFFERS in " + ",".join(diff)
    print(
        f"[step3r][seed-init] RIG-EQUIV: handed-down rig vs fresh build -> {verdict} "
        f"({n_fields} fields compared; the skipped rebuild took {dt:.2f}s)"
    )


def _repair_start_for_task_contacts(
    ph,
    sim,
    stage,
    scene,
    rig,
    seed,
    authored_start,
    contacts_xy,
    diverse,
):
    """Find a clean start in a component that reaches every static task contact."""
    import init_validation as iv
    import nav_grid

    for inflation in nav_grid.NAV_INFLATION_LADDER:
        candidate_grid = nav_grid.build_nav_grid(stage, scene, inflate=inflation)
        allowed, reachable, total = nav_grid.task_start_region(
            candidate_grid, contacts_xy
        )
        print(
            f"[step3r][seed-init] START repair inflate={inflation:.2f}: "
            f"best component reaches {reachable}/{total} contacts"
        )
        if total == 0 or reachable != total or not bool(allowed.any()):
            continue
        candidate_xy, candidate_stats = iv.place_start(
            ph,
            sim,
            stage,
            scene,
            rig,
            seed=int(seed),
            fallback_xy=authored_start,
            diverse=diverse,
            navigation_grid=candidate_grid,
            allowed_cells=allowed,
        )
        clean = (
            candidate_stats["confirm_pairs"] == 0
            and not candidate_stats["fallback_taken"]
        )
        if not clean:
            continue
        repaired_grid, chosen_inflation, nav_ok, nav_total = (
            nav_grid.build_task_nav_grid(
                stage,
                scene,
                candidate_xy,
                contacts_xy,
                inflations=(inflation,),
            )
        )
        if nav_ok == nav_total:
            return (
                candidate_xy,
                candidate_stats,
                repaired_grid,
                chosen_inflation,
                nav_ok,
                nav_total,
            )
    return None


def _activate_task_nav_grid(scene, grid, inflation, reachable, total) -> None:
    scene.wall_sdf_grid = grid if getattr(grid, "wall_sdf", None) is not None else None
    soft = getattr(scene, "soft_init", False)
    floor_bounded = getattr(grid, "floor_support", None) is not None
    if floor_bounded or (
        total > 0 and (reachable == total or (soft and reachable >= 1))
    ):
        scene.nav_grid = grid
        extra = (
            "; floor-bounded: unreachable legs fail instead of leaving the modelled floor"
            if floor_bounded and reachable < total
            else (
                "; soft_init: unreachable legs fail per-leg"
                if reachable < total
                else ""
            )
        )
        print(
            f"[step3r][seed-init] NAV planning ON (inflate={inflation:.2f}, "
            f"{reachable}/{total} object contacts reachable{extra})"
        )
        return
    scene.nav_grid = None
    print(
        f"[step3r][seed-init] NAV planning OFF: {reachable}/{total} object contacts reachable "
        f"(inflate={inflation:.2f}); straight-line fallback (byte-identical)"
    )


def refresh_task_navigation(stage, scene, ph, program, start_xy) -> dict:
    """Rebuild the shared geometry and task grid after final support collider setup."""
    collision_world = _ensure_cworld(scene, stage, ph)
    refresh = getattr(collision_world, "refresh", None)
    if callable(refresh):
        refresh(scene)
    if not getattr(scene, "plan_nav", False):
        return {"reachable": 0, "total": 0}
    import init_validation as iv
    import nav_grid

    contacts = []
    for key in _stand_targets(program, scene):
        contact = iv._entity_contact(scene, key)
        if contact is not None:
            contacts.append((float(contact[0]), float(contact[1])))
    if not contacts:
        return {"reachable": 0, "total": 0}
    grid, inflation, reachable, total = nav_grid.build_task_nav_grid(
        stage, scene, start_xy, contacts
    )
    _activate_task_nav_grid(scene, grid, inflation, reachable, total)
    return {
        "reachable": int(reachable),
        "total": int(total),
        "inflation": float(inflation),
    }


def _validate_init_for_render(app, stage, ph, scene, program, seed, rig=None) -> dict:
    """ENFORCE-AT-RENDER init validation for the seeded random-init render path (the BLOCKER fix).

    Builds the furniture static colliders, steps physics once so the lag-free scene-query overlap gate
    sees them, free-cell places the human (``place_start``, diverse), and validates the START pose against
    the EXACT human-vs-scene gate (body-core + feet + shoulders). Returns {start_xy, episode_valid,
    start_cell, reason}. Change 1: stands are NO LONGER pre-picked here -- each is chosen at RUNTIME in
    run_render's goal loop (``init_validation.pick_stand_runtime``, nearest-reachable from the human's live
    position), so this returns no ``stand_poses``. ``episode_valid`` is True ONLY if the START cleared the
    gate (confirm_pairs==0) -- if the start had to fall back to the least-penetrating candidate, it is False
    and the caller ABORTS (never renders a penetrating init). This runs on the SAME scene builders + the
    SAME colliders the render uses, so the validation and the render share ONE collision world.

    ``rig`` is the init-gate rig (``_init_gate_rig``); ``setup_episode`` has already built exactly that rig
    so it hands its copy down rather than pay for a second identical ``derive_rest_place_arms``. ``None``
    (any standalone caller) rebuilds it here, unchanged."""
    import init_validation as iv
    import object_physics as ophys

    ph.build_physics_scene(stage)
    ph.set_scene_timestep(stage, 60)
    tagged = _apply_scene_colliders(stage, ph)
    for _ in range(3):
        app.update()
    sim = ophys.get_sim()
    sim.step(
        render=False
    )  # register the furniture colliders for the scene-query overlap gate
    print(f"[step3r][seed-init] {len(tagged)} furniture/floor colliders applied")
    # P1a: build the SSOT collision oracle HERE (once, gated on scene.unified_geom) so scene.cworld exists
    # BEFORE nav / stand geometry is consumed. Idempotent -> the later run_render _ensure_cworld calls are
    # cache hits. ADDITIVE in P1a: no consumer reads scene.cworld any earlier than before -> byte-identical
    # (nav/stand are rewired to read it in P1b/P1c).
    _ensure_cworld(scene, stage, ph)
    # P4 container-inside: retarget each CONTAINER's drop onto its REAL interior shelf (raycast) BEFORE the
    # stand gate, so the gate reach-tests + the render both target the real shelf (no more mid-air drop).
    _resolve_container_drops(stage, scene)

    if rig is None:
        rig = _init_gate_rig()
    elif os.getenv("ISAAC_RIG_EQUIV_CHECK"):
        _check_init_gate_rig(rig)

    import nav_grid

    stand_keys = _stand_targets(program, scene)
    contacts_xy: list[tuple[float, float]] = []
    for key in stand_keys:
        try:
            contact = iv._entity_contact(scene, key)
        except Exception:  # noqa: BLE001
            contact = None
        if contact is not None:
            contacts_xy.append((float(contact[0]), float(contact[1])))

    authored_start = tuple(scene.start_xy)
    start_grid = None
    if getattr(scene, "plan_nav", False) or getattr(scene, "floor_prims", None):
        start_grid = nav_grid.build_nav_grid(
            stage, scene, inflate=nav_grid.NAV_INFLATION_LADDER[0]
        )
        scene.nav_grid = start_grid
    diverse_start = getattr(scene, "behavior_grab", None) is None and not getattr(
        scene, "near_start", False
    )
    start_xy, sstats = iv.place_start(
        ph,
        sim,
        stage,
        scene,
        rig,
        seed=int(seed),
        fallback_xy=authored_start,
        # BEHAVIOR + full-env (warehouse/hospital) scenes start NEAR the action (near-hint -> short walk
        # within the lit/open area; a long cross-building walk clips the env geometry the runtime nudge
        # does not avoid, and hits unlit areas). Existing scenes keep diverse random-init (byte-identical:
        # they set neither behavior_grab nor near_start).
        diverse=diverse_start,
        navigation_grid=start_grid,
    )
    scene.start_xy = (float(start_xy[0]), float(start_xy[1]))
    start_clean = sstats["confirm_pairs"] == 0 and not sstats["fallback_taken"]
    episode_valid = bool(start_clean)
    reason = (
        ""
        if start_clean
        else f"start fallback (confirm_pairs={sstats['confirm_pairs']})"
    )
    print(
        f"[step3r][seed-init] START xy=({start_xy[0]:.2f},{start_xy[1]:.2f}) cell={sstats['start_cell']} "
        f"confirm_pairs={sstats['confirm_pairs']} fallback={sstats['fallback_taken']} "
        f"episode_valid={sstats['episode_valid']}"
    )

    # ADDITIVE navigation grid: when the scene opts in (scene.plan_nav), build the per-episode nav grid on
    # the SAME colliders + the validated start over the object CONTACTS (build_task_nav_grid: the LARGEST
    # inflation keeping a within-arm's-reach free cell of EVERY contact reachable from start). This is the
    # ONE map both the runtime stand picker (its reachability gate) and the walker (A* + wall nudge + eye
    # push-out) consume. Enable planning only when the start reaches every contact's neighbourhood; else
    # leave scene.nav_grid unset -> the byte-identical straight-line fallback. episode_valid is NOT gated on
    # nav here (a penetrating START already set it False above).
    if getattr(scene, "plan_nav", False) and contacts_xy:
        n_grid, nav_infl, nav_ok, nav_tot = nav_grid.build_task_nav_grid(
            stage, scene, start_xy, contacts_xy
        )
        if nav_ok < nav_tot:
            repaired = _repair_start_for_task_contacts(
                ph,
                sim,
                stage,
                scene,
                rig,
                seed,
                authored_start,
                contacts_xy,
                diverse_start,
            )
            if repaired is not None:
                start_xy, sstats, n_grid, nav_infl, nav_ok, nav_tot = repaired
                scene.start_xy = (float(start_xy[0]), float(start_xy[1]))
                start_clean = (
                    sstats["confirm_pairs"] == 0 and not sstats["fallback_taken"]
                )
                episode_valid = bool(start_clean)
                reason = "" if start_clean else "start repair was not collision-free"
                print(
                    f"[step3r][seed-init] START repaired to ({start_xy[0]:.2f},"
                    f"{start_xy[1]:.2f}); {nav_ok}/{nav_tot} contacts reachable"
                )
        _activate_task_nav_grid(scene, n_grid, nav_infl, nav_ok, nav_tot)
    return {
        "start_xy": start_xy,
        "episode_valid": episode_valid,
        "start_cell": sstats.get("start_cell"),
        "reason": reason or "clean",
    }


def _run_fallbackcheck(app, stage, ph, args) -> int:
    """P4 FALLBACK proof (no RTX): force an OVER-CLUTTERED / unplaceable init and assert it DEGRADES to
    episode_valid=False (least-penetrating accepted + flagged), NOT a crash or a silently-accepted
    penetrating frame. Builds a real scene, then RESTRICTS place_start's region to a tiny box CENTERED ON
    a furniture footprint (so every cell is blocked/penetrating) -- exhausting MAX_START_DRAWS -- and
    checks the RL contract holds: fallback_taken=True, episode_valid=False, and a start_xy is still
    returned (no exception). Returns 0 iff the fallback fired correctly."""
    import init_validation as iv
    import object_physics as ophys

    rig = core.load_family("side_wrap")
    _build, _otw = core.load_object_def("cup")
    core.set_object(rig, _build, _otw)
    try:
        core.derive_rest_place_arms(rig)
    except Exception as exc:  # noqa: BLE001
        print(f"[fallbackcheck] derive_rest_place_arms failed ({exc}); canonical arm")

    builders = _scene_builders(seed=getattr(args, "seed", 0))
    scene, _controllers, _program, _tl, _te = builders[args.scene](stage, app)
    ph.build_physics_scene(stage)
    ph.set_scene_timestep(stage, 60)
    _apply_scene_colliders(stage, ph)
    for _ in range(3):
        app.update()
    sim = ophys.get_sim()
    sim.step(render=False)

    # A TINY region centered ON the first floor-blocking furniture footprint -> every grid cell overlaps
    # the inflated footprint, so place_start finds NO clean cell and MUST hit the least-penetrating
    # fallback. This is the "over-cluttered -> flagged" degradation the blueprint asked for.
    footprints = iv.collidable_footprints(stage)
    blocking = [
        (cx, cy)
        for _o, cx, cy, hx, hy, bot, top in footprints
        if iv._blocks_floor(bot, top, scene.floor_z, 0.15)
    ]
    if not blocking:
        print(
            "[fallbackcheck] scene has no blocking furniture -> cannot force fallback; SKIP"
        )
        _exit_process(0)
        return 0
    cx, cy = blocking[0]
    tiny = (
        cx - 0.2,
        cx + 0.2,
        cy - 0.2,
        cy + 0.2,
    )  # ~0.4 m box on the furniture center
    print(
        f"[fallbackcheck] scene={args.scene}: forcing tiny region {tuple(round(v, 2) for v in tiny)} on furniture ({cx:.2f},{cy:.2f})"
    )
    crashed = False
    try:
        start_xy, sstats = iv.place_start(
            ph, sim, stage, scene, rig, seed=0, region=tiny, fallback_xy=(cx, cy)
        )
    except Exception as exc:  # noqa: BLE001
        crashed = True
        start_xy, sstats = None, {}
        print(f"[fallbackcheck] place_start CRASHED: {exc}")

    got_xy = start_xy is not None
    flagged = bool(sstats.get("fallback_taken")) and not bool(
        sstats.get("episode_valid", True)
    )
    ok = (not crashed) and got_xy and flagged
    branch = (
        "no-free-cell"
        if sstats.get("grid_free_cells", -1) == 0
        else "least-penetrating"
    )
    print(
        f"[fallbackcheck] result: crashed={crashed} start_returned={got_xy} "
        f"grid_free={sstats.get('grid_free_cells')} branch={branch} "
        f"fallback_taken={sstats.get('fallback_taken')} episode_valid={sstats.get('episode_valid')} "
        f"confirm_pairs={sstats.get('confirm_pairs')} -> "
        f"{'PASS (degraded to episode_valid=False, penetrating fallback flagged, no crash)' if ok else 'FAIL'}"
    )
    _exit_process(0 if ok else 2)
    return 0 if ok else 2


# =================================================================================================
# Substep-delivery GATE (offline, no RTX): drop a YCB prop 0.5m and measure fall vs analytic.
# =================================================================================================
def _run_physcheck(app, stage, ph, rcfg) -> int:
    """Free-fall probe proving N substeps ACTUALLY deliver under the render loop's single time-advance.

    Spawns ONE YCB prop as a DYNAMIC rigid HIGH above the floor (so 1.0 s of fall never lands it),
    raises the scene rate to ``set_scene_timestep(60)`` (= FPS*5 substeps + the determinism flags),
    starts the timeline, and calls ``fpv_render.advance_once`` per frame reading the prop's z. The very
    first advance after ``timeline.play()`` does not integrate physics (a known warmup step), so we do a
    short WARMUP of advance_once calls (uncounted) and only take z0 once the body is actually moving;
    then K=12 measured frames (1.0 s at FPS=12) of PURE free-fall (no floor to hit). Asserts the total
    measured fall is within +-25% of analytic 0.5*g*t^2 from that start AND the per-frame drop tracks
    analytic (NOT ~1/5 = slow-motion, NOT ~0 = no advance). This gates the whole free-dynamic P3 path --
    if it fails, the fallback (delta_time=0.0 + app.update-advanced timeline) is needed. Returns 0/2.
    """
    import fpv_core
    import fpv_render
    import fpv_scene
    import object_physics as ophys

    g_accel = 9.81
    drop_h = 8.0  # high enough that 1.0s free-fall (~4.9m) never reaches the floor -> pure free-fall
    warmup = (
        3  # uncounted advance_once calls to absorb the play()/physics startup step(s)
    )
    k_frames = 12
    dt = 1.0 / float(rcfg.capture_fps)

    path = "/World/PhysProbe"
    prim = fpv_scene.add_reference(
        stage,
        path,
        fpv_scene.ASSET_ROOT + "/Props/YCB/Axis_Aligned/002_master_chef_can.usd",
    )
    min_z = fpv_scene.local_bbox_min_z(prim)
    fpv_scene.set_world_pose(prim, (0.0, 0.0, drop_h - min_z))
    ph.make_dynamic_rigid(
        prim, density=400.0, approximation=ophys.classify_collider(prim)
    )
    ph.set_scene_timestep(
        stage, float(rcfg.capture_fps), substeps=5
    )  # 60Hz = 5 substeps/frame
    for _ in range(3):
        app.update()  # commit the spawn pose before the clock starts

    fpv_core.start_timeline(rcfg.playback_fps)
    # WARMUP: the first advance_once after play() does not integrate physics (a known startup stall), so
    # run a few (uncounted) advances to reach a clean falling regime. We then measure the object's initial
    # velocity v0 at the measurement start and predict the fall with the FULL kinematic law
    # fall(t)=v0*t + 0.5*g*t^2 (v0 != 0 because warmup already accelerated it). This is the rigorous
    # gravity-integration proof: SLOW-MOTION would show BOTH a too-small v0 AND a too-small per-frame
    # acceleration; here the per-frame drop must grow by ~g*dt^2 each frame (the acceleration signature).
    for _ in range(warmup):
        fpv_render.advance_once(rcfg)
    z0 = float(ophys.read_prim_xyz(stage, path)[2])
    fpv_render.advance_once(rcfg)  # one more to measure v0 over a single dt
    z1 = float(ophys.read_prim_xyz(stage, path)[2])
    v0 = (z0 - z1) / dt  # downward speed at the measurement start (m/s)
    analytic_accel_step = (
        g_accel * dt * dt
    )  # ideal frame-to-frame increase in per-frame drop
    print(
        f"[physcheck] free-fall probe: start z={z0:.4f} (from {drop_h - min_z:.4f}, {warmup} warmup), "
        f"v0={v0:.3f} m/s, K={k_frames} frames @ {rcfg.capture_fps}fps, physx=60Hz(5 substeps); "
        f"expect per-frame drop to grow by ~g*dt^2={analytic_accel_step * 1000:.1f}mm/frame"
    )
    prev = z1
    prev_step = z0 - z1
    ok_per_frame = True
    for i in range(1, k_frames + 1):
        fpv_render.advance_once(rcfg)
        z = float(ophys.read_prim_xyz(stage, path)[2])
        t = i * dt
        # analytic from the measurement start (z1), using measured v0: v0*t + 0.5*g*t^2.
        analytic_total = v0 * t + 0.5 * g_accel * t * t
        drop_total = z1 - z
        drop_step = prev - z
        accel_step = (
            drop_step - prev_step
        )  # frame-to-frame change in the drop (== g*dt^2 ideally)
        # ACCELERATION signature: each frame's drop grows by ~g*dt^2. Slow-motion at 1/5 rate would grow
        # by ~1/25 of this; no-advance by ~0. Require within 40% of the ideal increment.
        frame_ok = abs(accel_step - analytic_accel_step) <= 0.4 * analytic_accel_step
        ok_per_frame = ok_per_frame and frame_ok
        print(
            f"[physcheck]   f{i:02d} t={t:.3f}s: z={z:.4f} drop_total={drop_total * 1000:7.1f}mm "
            f"(analytic {analytic_total * 1000:7.1f}mm) step={drop_step * 1000:6.1f}mm "
            f"d_step={accel_step * 1000:5.1f}mm (ideal {analytic_accel_step * 1000:.1f}mm) "
            f"{'ok' if frame_ok else 'SLOW/STALL'}"
        )
        prev, prev_step = z, drop_step

    t_end = k_frames * dt
    analytic_end = v0 * t_end + 0.5 * g_accel * t_end * t_end
    measured = z1 - prev
    rel_err = abs(measured - analytic_end) / analytic_end
    within = rel_err <= 0.25
    ok = bool(within and ok_per_frame)
    print(
        f"[physcheck] RESULT: measured fall={measured * 1000:.1f}mm vs analytic (v0*t+0.5gt^2) "
        f"{analytic_end * 1000:.1f}mm (rel_err={rel_err * 100:.1f}%, need <=25%); per-frame accel "
        f"tracks g*dt^2 = {ok_per_frame} -> {'PASS' if ok else 'FAIL (slow-motion or no-advance)'}"
    )
    _exit_process(0 if ok else 2)
    return 0 if ok else 2


# ---- AUTO-EXPOSURE tuning (see _calibrate_exposure) ----------------------------------------------
# Zero-time RTX refreshes between re-aiming the probe camera and reading it. A camera teleport resets
# temporal accumulation, so one refresh is not enough; none of these refreshes may move physics.
_EXPOSE_PROBE_FRAMES: int = 16
# Standing viewpoints spread across the SCENE's navigable floor, and look directions per viewpoint.
# The probe budget is N_CELLS x N_DIRS x PROBE_FRAMES advances per iteration, paid once per scene
# (the result is cached).
_EXPOSE_N_CELLS: int = 6
_EXPOSE_N_DIRS: int = 2
# Open floor required around a probe viewpoint -- roughly the room a standing person occupies. Keeps the
# eye out of wall cavities and closets, where it renders black and measures nothing about the scene.
_EXPOSE_MIN_CLEARANCE_M: float = 0.4
# A probe frame flatter than this is not a view of the scene (an empty dome, or the inside of a mesh):
# it carries no information about the lighting and must not be median-ed in as if it did.
_EXPOSE_MIN_CONTRAST: float = 0.02
# Bump when anything that changes a cached scale changes shape.
_EXPOSE_CACHE_SCHEMA: int = 4
# Acceptance window for the probe's 92nd-percentile luminance.
_EXPOSE_WINDOW: tuple[float, float] = (0.5, 0.8)
# Per-iteration clip on the light scale.
_EXPOSE_STEP_LO: float = 0.2
_EXPOSE_STEP_HI: float = 3.5
# CUMULATIVE clip, deliberately ASYMMETRIC. p_hi is an LDR percentile in [0,1], so however blown a
# scene looks the honest dim it asks for is bounded -- the floor is what caps the damage a still-
# undetected bad probe can do. A near-black scene, by contrast, legitimately needs a large boost, so
# the ceiling is two full-strength corrections. Both are far tighter than the old unbounded compound
# (the per-step clip taken 4 times spanned 0.0016x..150x).
_EXPOSE_CUM_LO: float = 0.25
_EXPOSE_CUM_HI: float = 12.0
# FIND-28003's task route works over the black kitchen bar in house_double_floor_lower. The scene-wide
# median is dominated by brighter rooms and measuredly dims that work area to near-black. Keep this
# narrowly scoped until a lower-tail exposure policy has been validated across the full corpus.
_EXPOSE_MIN_SCALE_BY_SCENE_ID: dict[str, float] = {
    "spec:FIND-28003": 1.0,
    "spec:GUIDE-40003": 1.35,
    "spec:GUIDE-40005": 1.35,
    "spec:GUIDE-109009": 1.35,
}
_DEFAULT_HEADLAMP_INTENSITY: float = 2500.0
_HEADLAMP_INTENSITY_BY_SCENE_ID: dict[str, float] = {
    "spec:GUIDE-40003": 12000.0,
    "spec:GUIDE-40005": 12000.0,
}
_TASK_FILL_LIGHTS_BY_SCENE_ID: dict[
    str, tuple[tuple[str, float], ...]
] = {
    "spec:GUIDE-40003": (
        ("breakfast_bar", 60000.0),
        ("burner", 60000.0),
    ),
    "spec:GUIDE-40005": (
        ("work_counter", 60000.0),
        ("washer", 90000.0),
    ),
}
_TASK_FILL_LIGHT_HEIGHT_M: float = 1.25
_TASK_FILL_LIGHT_RADIUS_M: float = 0.6
# A scaling of at least DEADBAND relative size must move the metric by at least MIN_FRAC of what it
# should, in the right direction. MIN_FRAC is deliberately an order of magnitude loose: the ACES
# highlight rolloff really does compress a dim of a blown-out scene, but a response 10x below the
# applied factor is not a tonemap, it is a measurement that is not reading the lights.
_EXPOSE_RESPONSE_DEADBAND: float = 0.10
_EXPOSE_RESPONSE_MIN_FRAC: float = 0.10


def _all_light_intensity_attrs(stage):
    """Every UsdLux light intensity attr under /World (our scene lights + add_lighting's dome/distant +
    _boost_lights), so auto-exposure can scale the WHOLE lighting rig by one factor."""
    from pxr import Usd, UsdLux

    out = []
    root = stage.GetPrimAtPath("/World")
    if not (root and root.IsValid()):
        return out
    for p in Usd.PrimRange(root):
        # the FPV headlamp is a FIXED fill -> never scaled by auto-exposure
        if "Headlamp" in p.GetName():
            continue
        tn = p.GetTypeName()
        if tn and "Light" in str(tn):
            attr = UsdLux.LightAPI(p).GetIntensityAttr()
            if attr:
                out.append(attr)
    return out


def _expose_measure(rcfg, annot) -> dict | None:
    """Refresh RTX ``_EXPOSE_PROBE_FRAMES`` times at fixed simulation time, then measure.

    Returns None when the annotator never populates. ``trace`` is the per-refresh p92 so an
    unconverged probe shows up in the log instead of being silently calibrated against.
    """
    import fpv_render

    trace: list[float] = []
    stats: dict | None = None
    for _ in range(_EXPOSE_PROBE_FRAMES):
        fpv_render.refresh_once(rcfg)
        data = annot.get_data()
        if data is None or getattr(data, "size", 0) == 0:
            continue
        rgb = np.asarray(data)[..., :3].astype(np.float32) / 255.0
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        stats = {
            "p92": float(np.percentile(lum, 92)),
            "p50": float(np.percentile(lum, 50)),
            "mean": float(lum.mean()),
            "lo": float(lum.min()),
            "hi": float(lum.max()),
        }
        trace.append(stats["p92"])
    if stats is None:
        return None
    stats["trace"] = trace
    return stats


def _farthest_points(xy: np.ndarray, k: int) -> list[int]:
    """Deterministic farthest-point sampling: seed at the point nearest the centroid, then repeatedly
    take the point furthest from everything already chosen. Ties break to the lowest index, so the
    result depends only on the point set."""
    if len(xy) <= k:
        return list(range(len(xy)))
    first = int(np.argmin(((xy - xy.mean(axis=0)) ** 2).sum(axis=1)))
    out = [first]
    d = ((xy - xy[first]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        out.append(nxt)
        d = np.minimum(d, ((xy - xy[nxt]) ** 2).sum(axis=1))
    return out


def _expose_array_digest(values: np.ndarray, *, sdf_mm: bool = False) -> str:
    """Deterministic digest of a grid field, independent of NumPy strides and host endianness."""
    import hashlib
    import json

    array = np.asarray(values)
    if sdf_mm:
        source = np.asarray(array, dtype=np.float64)
        payload = np.empty(source.shape, dtype="<i4")
        finite = np.isfinite(source)
        clipped = np.clip(source[finite], -2_000_000.0, 2_000_000.0)
        payload[finite] = np.rint(clipped * 1000.0).astype("<i4")
        payload[np.isnan(source)] = np.iinfo(np.int32).min
        payload[np.isposinf(source)] = np.iinfo(np.int32).max
        payload[np.isneginf(source)] = np.iinfo(np.int32).min + 1
        encoding = "signed_mm_i32_le"
    else:
        payload = np.asarray(array, dtype=np.bool_).astype(np.uint8)
        encoding = "bool_u8"
    header = json.dumps(
        {"encoding": encoding, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(header)
    digest.update(np.ascontiguousarray(payload).tobytes(order="C"))
    return digest.hexdigest()


def _expose_pose_fingerprint(poses: list) -> list[dict[str, list[float]]]:
    return [
        {
            "eye": [round(float(value), 6) for value in eye],
            "look": [round(float(value), 6) for value in look],
        }
        for eye, look in poses
    ]


def _expose_furnished_mask(stage, scene, grid) -> np.ndarray | None:
    """Cells within the extent of the scene's own FURNITURE, or None if it has none.

    The grid's region is an AABB over every footprint, so the open ground OUTSIDE the building is "free"
    too -- and being the extreme points it is the first thing farthest-point sampling grabs. Metering
    there reads the empty dome, not the scene. Reachability does not exclude it either: the doors-OPEN
    view the flood needs (so a shut door cannot amputate a room) is exactly what lets it walk out of the
    front door. Furniture, on the other hand, only exists indoors, so its extent is a floor-plan-agnostic
    stand-in for "inside" that needs no assumption about what the scene calls a wall."""
    import init_validation as iv

    exclude = ("/World/HumanProxy", "/World/Body", "/World/Obj") + tuple(
        getattr(scene, "wall_prims", []) or []
    )
    foot = iv.collidable_footprints(stage, exclude_prefixes=exclude)
    if not foot:
        return None
    xs = [(cx - hx, cx + hx) for _o, cx, _cy, hx, _hy, _b, _t in foot]
    ys = [(cy - hy, cy + hy) for _o, _cx, cy, _hx, hy, _b, _t in foot]
    lo_x, hi_x = min(a for a, _ in xs), max(b for _, b in xs)
    lo_y, hi_y = min(a for a, _ in ys), max(b for _, b in ys)
    cx = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.cell
    cy = grid.y0 + (np.arange(grid.ny) + 0.5) * grid.cell
    return ((cy >= lo_y) & (cy <= hi_y))[:, None] & ((cx >= lo_x) & (cx <= hi_x))[
        None, :
    ]


def _expose_scene_poses(stage, scene, eye_z: float) -> tuple[list, dict]:
    """Standing probe poses spread over the SCENE's navigable floor -- NOT over the episode's route.

    At eval time the route is whatever the assistant under test drives the human to walk, so metering it
    would hand two models a different exposure on the same task and make the render depend on the episode
    it is about to render. Navigable free space is a property of the scene alone (``build_nav_grid`` reads
    only stage geometry and the scene's declared walls, never the rendered route),
    so it is identical for every model, seed, task and route -- and, being the floor a human can stand on,
    it spans every room the episode can reach.

    A cell must be free on the scene-wide grid, WITHIN the furniture extent (see
    _expose_furnished_mask), and have
    ``_EXPOSE_MIN_CLEARANCE_M`` of open floor around it -- a bare free cell can be a wall cavity or a
    closet, where the eye renders black and measures nothing about the scene.

    Cells come from FARTHEST-POINT sampling rather than a grid stride: it maximises spread whatever shape
    the free area has, where a fixed stride happily drops every sample into the one big room. Directions
    walk the golden angle so they decorrelate from the grid axes; with two per cell they are opposite, so
    each cell contributes one likely wall-facing and one room-facing sample.

    Returns (poses, free-space fingerprint). The fingerprint goes in the cache key -- move the floor plan
    and the cached exposure is invalidated."""
    import nav_grid as _ng

    grid = _ng.build_nav_grid(stage, scene)
    ok = ~np.asarray(grid.occ, dtype=bool)
    for extra in (
        _expose_furnished_mask(stage, scene, grid),
        np.asarray(grid.sdf) >= _EXPOSE_MIN_CLEARANCE_M,
    ):
        # Each filter is a preference, not a requirement: one that would leave nothing to sample is
        # dropped rather than allowed to collapse the metering to a single arbitrary cell.
        if extra is not None and (ok & extra).any():
            ok = ok & extra
    free = np.argwhere(ok)
    if free.size == 0:
        return [], {}
    xy = np.stack(
        [
            grid.x0 + (free[:, 1] + 0.5) * grid.cell,
            grid.y0 + (free[:, 0] + 0.5) * grid.cell,
        ],
        axis=1,
    )
    # Quantised so a millimetre of jitter in the measured stature cannot thrash the per-scene cache.
    ez = round(float(eye_z), 1)
    ahead = 3.0
    drop = 0.25
    poses = []
    for k, idx in enumerate(_farthest_points(xy, _EXPOSE_N_CELLS)):
        eye = np.array([xy[idx, 0], xy[idx, 1], ez], dtype=np.float64)
        for d in range(_EXPOSE_N_DIRS):
            a = math.radians(137.507 * k + 360.0 * d / _EXPOSE_N_DIRS)
            look = np.array([math.cos(a) * ahead, math.sin(a) * ahead, -drop])
            poses.append((eye, look / np.linalg.norm(look)))
    fingerprint = {
        "cell": round(float(grid.cell), 4),
        "nx": int(grid.nx),
        "ny": int(grid.ny),
        "x0": round(float(grid.x0), 3),
        "y0": round(float(grid.y0), 3),
        "n_free": int(len(xy)),
        "eye_z": ez,
        "occupancy_digest": _expose_array_digest(grid.occ),
        "sdf_mm_digest": _expose_array_digest(grid.sdf, sdf_mm=True),
        "eligible_digest": _expose_array_digest(ok),
        "probe_poses": _expose_pose_fingerprint(poses),
    }
    return poses, fingerprint


def _expose_cache_path() -> str:
    return str(autoexposure_cache_path())


def _expose_json_value(value):
    """Convert common USD/Gf/Vt values to stable JSON primitives."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "+inf" if value > 0.0 else "-inf"
        rounded = round(value, 8)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, np.ndarray):
        return [_expose_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            str(key): _expose_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_expose_json_value(item) for item in value]
    asset_path = getattr(value, "path", None)
    if isinstance(asset_path, str):
        return asset_path
    try:
        return [_expose_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _expose_attr_fingerprint(attr) -> dict:
    record = {"value": _expose_json_value(attr.Get())}
    get_connections = getattr(attr, "GetConnections", None)
    if callable(get_connections):
        connections = sorted(str(path) for path in (get_connections() or ()))
        if connections:
            record["connections"] = connections
    get_times = getattr(attr, "GetTimeSamples", None)
    if callable(get_times):
        times = list(get_times() or ())
        if times:
            record["time_samples"] = [
                {
                    "time": _expose_json_value(float(time)),
                    "value": _expose_json_value(attr.Get(time)),
                }
                for time in times
            ]
    return record


def _expose_prim_attributes(prim, *, transforms_only: bool) -> dict[str, dict]:
    get_attributes = getattr(prim, "GetAttributes", None)
    if not callable(get_attributes):
        return {}
    out = {}
    for attr in get_attributes() or ():
        name = str(attr.GetName())
        if transforms_only:
            keep = (
                name == "visibility"
                or name == "xformOpOrder"
                or name.startswith("xformOp:")
            )
        else:
            keep = True
        if keep:
            out[name] = _expose_attr_fingerprint(attr)
    return {name: out[name] for name in sorted(out)}


def _expose_transform_chain(prim) -> list[dict]:
    chain = []
    seen = set()
    current = prim
    while current is not None:
        is_valid = getattr(current, "IsValid", None)
        if callable(is_valid) and not is_valid():
            break
        path = str(current.GetPath())
        if path in seen:
            break
        seen.add(path)
        row = {
            "path": path,
            "attributes": _expose_prim_attributes(current, transforms_only=True),
        }
        is_active = getattr(current, "IsActive", None)
        if callable(is_active):
            row["active"] = bool(is_active())
        chain.append(row)
        get_parent = getattr(current, "GetParent", None)
        current = get_parent() if callable(get_parent) else None
    chain.reverse()
    return chain


def _expose_light_fingerprint(intensity_attr, base_intensity: float) -> dict:
    prim = intensity_attr.GetPrim()
    get_schemas = getattr(prim, "GetAppliedSchemas", None)
    schemas = (
        sorted(str(schema) for schema in (get_schemas() or ()))
        if callable(get_schemas)
        else []
    )
    relationships = {}
    get_relationships = getattr(prim, "GetRelationships", None)
    if callable(get_relationships):
        for relationship in get_relationships() or ():
            relationships[str(relationship.GetName())] = sorted(
                str(path) for path in (relationship.GetTargets() or ())
            )
    return {
        "path": str(prim.GetPath()),
        "type": str(prim.GetTypeName() or ""),
        "applied_schemas": schemas,
        "base_intensity": _expose_json_value(float(base_intensity)),
        "attributes": _expose_prim_attributes(prim, transforms_only=False),
        "relationships": {name: relationships[name] for name in sorted(relationships)},
        "transform_chain": _expose_transform_chain(prim),
    }


def _expose_cache_inputs(attrs, base, free_fp: dict, target: float, tries: int) -> dict:
    lights = [
        _expose_light_fingerprint(attr, intensity)
        for attr, intensity in zip(attrs, base)
    ]
    lights.sort(key=lambda light: light["path"])
    return {
        "schema": _EXPOSE_CACHE_SCHEMA,
        "target": round(float(target), 4),
        "tries": int(tries),
        "probe_frames": _EXPOSE_PROBE_FRAMES,
        "calibration_body": "neutral_v1",
        "render_aa_op": 3,
        "zero_dt_refresh": True,
        "n_cells": _EXPOSE_N_CELLS,
        "n_dirs": _EXPOSE_N_DIRS,
        "min_clearance_m": _EXPOSE_MIN_CLEARANCE_M,
        "min_contrast": _EXPOSE_MIN_CONTRAST,
        "window": [_EXPOSE_WINDOW[0], _EXPOSE_WINDOW[1]],
        "cum_bounds": [_EXPOSE_CUM_LO, _EXPOSE_CUM_HI],
        "lights": lights,
        "free_space": free_fp,
    }


def _expose_cache_key(inputs: dict) -> str:
    import hashlib
    import json

    return hashlib.blake2b(
        json.dumps(
            inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode(),
        digest_size=8,
    ).hexdigest()


def _expose_cache_get(inputs: dict) -> dict | None:
    """The cached scale for exactly these inputs, or None.

    The FULL recorded inputs are re-compared, not just their hash: a stale entry must never silently
    apply an exposure that was computed for different lighting or a different floor plan.
    """
    import json

    if os.environ.get("ISAAC_AUTOEXP_NOCACHE"):
        return None
    try:
        with open(_expose_cache_path()) as f:
            db = json.load(f)
    except (OSError, ValueError):
        return None
    ent = db.get(_expose_cache_key(inputs))
    if not isinstance(ent, dict) or ent.get("inputs") != inputs:
        return None
    res = ent.get("result")
    return res if isinstance(res, dict) else None


def _expose_cache_put(inputs: dict, result: dict) -> None:
    import json

    path = _expose_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path) as f:
                db = json.load(f)
        except (OSError, ValueError):
            db = {}
        db[_expose_cache_key(inputs)] = {"inputs": inputs, "result": result}
        with open(path, "w") as f:
            f.write(json.dumps(db, indent=2, sort_keys=True) + "\n")
        print(f"[autoexp] cached -> {path}")
    except OSError as exc:
        print(f"[autoexp] cache write failed ({exc})")


def _expose_combine(reads: list[dict], rejected: list[dict] | None = None) -> dict:
    """Median-combine the per-pose readings into the ONE number the calibration acts on.

    MEDIAN, not mean: a single blown-out or pitch-black viewpoint must not drag the whole episode's
    exposure -- that robustness is the entire point of metering more than one pose. The spread is
    carried alongside because a wide one means the scene's lighting is genuinely uneven and NO single
    fixed exposure can be right everywhere, which is a finding, not a bug to be averaged away.

    ``rejected`` carries the poses that were DROPPED, each with its pose id, why, and what it did
    measure. A dropped pose is exactly how a darkness gate goes blind: the one viewpoint that renders
    black is also the one the FLAT test discards, so a scene can meter perfectly while half its
    viewpoints are unusable. Dropping is still right -- a black frame carries no information about the
    lighting -- but it must be REPORTED, never silent."""
    p92 = [float(r["p92"]) for r in reads]
    rej = list(rejected or ())
    return {
        "p92": float(np.median(p92)),
        "p50": float(np.median([r["p50"] for r in reads])),
        "mean": float(np.median([r["mean"] for r in reads])),
        "lo": float(min(r["lo"] for r in reads)),
        "hi": float(max(r["hi"] for r in reads)),
        "p92_poses": p92,
        "pose_ids": [int(r["pose"]) for r in reads],
        "n_rejected": len(rej),
        "rejected": rej,
        "trace": reads[0]["trace"],
    }


def _probe_poses(cop, rcfg, annot, poses, label: str) -> dict | None:
    """Converge + measure the probe at EVERY pose, then median-combine. One log line per pose carries
    its whole per-advance trace, so an unconverged probe is visible rather than calibrated against.
    """
    import smplh_camera

    reads: list[dict] = []
    rejected: list[dict] = []
    for i, (peye, plook) in enumerate(poses):
        smplh_camera.drive_fpv_gaze(
            cop, peye, plook, peye + plook, eye_push=0.0, aim_blend=1.0
        )
        st = _expose_measure(rcfg, annot)
        if st is None:
            print(f"[autoexp] {label} pose {i}: empty frame; REJECTED")
            rejected.append(
                {"pose": i, "why": "empty", "eye": [round(float(v), 3) for v in peye]}
            )
            continue
        st["pose"] = i
        flat = st["hi"] - st["lo"] < _EXPOSE_MIN_CONTRAST
        print(
            f"[autoexp] {label} pose {i}: p92={st['p92']:.3f} p50={st['p50']:.3f} "
            f"mean={st['mean']:.3f} range=[{st['lo']:.3f},{st['hi']:.3f}]"
            f"{' FLAT -> REJECTED' if flat else ''} | p92/refresh: "
            + ",".join(f"{v:.3f}" for v in st["trace"])
        )
        if flat:
            rejected.append(
                {
                    "pose": i,
                    "why": "flat",
                    "eye": [round(float(v), 3) for v in peye],
                    "p92": round(st["p92"], 4),
                    "mean": round(st["mean"], 4),
                    "contrast": round(st["hi"] - st["lo"], 4),
                }
            )
            continue
        reads.append(st)
    if not reads:
        print(f"[autoexp] {label}: no pose returned a view of the scene")
        return None
    if rejected:
        print(
            f"[autoexp] {label}: {len(rejected)}/{len(poses)} pose(s) REJECTED and excluded from the "
            f"median -> " + ", ".join(f"p{r['pose']}({r['why']})" for r in rejected)
        )
    return _expose_combine(reads, rejected)


def _expose_log(label: str, st: dict, target: float) -> None:
    """One honest summary per round: the median the loop acts on, and the spread it hides."""
    per = st["p92_poses"]
    print(
        f"[autoexp] {label}: MEDIAN p92={st['p92']:.3f} p50={st['p50']:.3f} mean={st['mean']:.3f} "
        f"(target {target:.2f}) over {len(per)} pose(s) p92 min/max={min(per):.3f}/{max(per):.3f} "
        f"spread={max(per) - min(per):.3f} range=[{st['lo']:.3f},{st['hi']:.3f}]"
    )


def _expose_search(
    probe,
    set_scale,
    first,
    target,
    tries,
    *,
    min_scale: float = _EXPOSE_CUM_LO,
) -> tuple[float, dict, str]:
    """Iterate the cumulative light scale toward ``target``; return (best_scale, its measurement, stop).

    Never worse than doing nothing: scale 1.0 (``first``, measured before any light is touched) starts as
    the best and is replaced only by a STRICTLY closer measured p92, and the caller applies the returned
    best. A scaling whose measured response does not track the factor applied -- too small, or the wrong
    direction -- is proof the measurement is not reading the lights, so it is abandoned rather than
    compounded (the failure mode that dimmed whole scenes 4.6x)."""
    cum, cur, stop = 1.0, first, "max_tries"
    best_cum, best = 1.0, first
    if min_scale > 1.0:
        set_scale(min_scale)
        floor_measurement = probe()
        cum = min_scale
        best_cum = min_scale
        if floor_measurement is None:
            return best_cum, best, "empty_probe_at_minimum"
        cur = floor_measurement
        best = floor_measurement
    for _ in range(int(tries)):
        if _EXPOSE_WINDOW[0] <= cur["p92"] <= _EXPOSE_WINDOW[1]:
            stop = "in_window"
            break
        step = float(
            np.clip(target / max(cur["p92"], 0.04), _EXPOSE_STEP_LO, _EXPOSE_STEP_HI)
        )
        new_cum = float(np.clip(cum * step, min_scale, _EXPOSE_CUM_HI))
        applied = new_cum / cum
        if abs(applied - 1.0) < 1e-3:
            stop = "cum_bound"
            break
        set_scale(new_cum)
        nxt = probe()
        if nxt is None:
            stop = "empty_probe"
            break
        want = applied - 1.0
        got = nxt["p92"] / max(cur["p92"], 1e-6) - 1.0
        if abs(want) >= _EXPOSE_RESPONSE_DEADBAND and (
            got * want <= 0.0 or abs(got) < _EXPOSE_RESPONSE_MIN_FRAC * abs(want)
        ):
            print(
                f"[autoexp] x{applied:.3f} moved p92 {cur['p92']:.3f}->{nxt['p92']:.3f} "
                f"({got:+.1%}, expected ~{want:+.1%}): the probe does not track the lights, "
                f"so the reading is invalid -> revert and stop"
            )
            stop = "no_response"
            break
        cum, cur = new_cum, nxt
        _expose_log(f"x{cum:.3f}", cur, target)
        if abs(cur["p92"] - target) < abs(best["p92"] - target):
            best_cum, best = cum, cur
    return best_cum, best, stop


def _expose_record(
    rec: dict, first: dict, best: dict, best_cum: float, stop: str
) -> dict:
    """The manifest row for one calibration: what the frame measured before, what it measures after, and
    the single light scale in between -- so a dark corpus is auditable from the manifests alone.
    """
    rec.update(
        {
            "light_scale": round(best_cum, 4),
            "stop": stop,
            "in_window": bool(_EXPOSE_WINDOW[0] <= best["p92"] <= _EXPOSE_WINDOW[1]),
            "n_poses": len(first["p92_poses"]),
            "p92_before": round(first["p92"], 4),
            "p50_before": round(first["p50"], 4),
            "mean_before": round(first["mean"], 4),
            "p92_poses_before": [round(v, 4) for v in first["p92_poses"]],
            "p92_spread_before": round(
                max(first["p92_poses"]) - min(first["p92_poses"]), 4
            ),
            "p92_after": round(best["p92"], 4),
            "p50_after": round(best["p50"], 4),
            "mean_after": round(best["mean"], 4),
            "p92_poses_after": [round(v, 4) for v in best["p92_poses"]],
            "p92_spread_after": round(
                max(best["p92_poses"]) - min(best["p92_poses"]), 4
            ),
            # The poses the median was NOT taken over. Without these a scene whose dark half is
            # rejected as FLAT looks perfectly exposed in the manifest (see _expose_combine).
            "n_rejected_before": int(first.get("n_rejected", 0)),
            "rejected_before": first.get("rejected", []),
            "n_rejected_after": int(best.get("n_rejected", 0)),
            "rejected_after": best.get("rejected", []),
            "probe_trace_before": [round(v, 4) for v in first["trace"]],
        }
    )
    return rec


def _calibrate_exposure(
    app, stage, rcfg, cop, annot, scene, target=0.66, tries=4, eye_z=None, scene_id=None
) -> dict:
    """AUTO-EXPOSURE (matches the codebase's fixed-exposure philosophy -- RTX auto-exposure is kept OFF
    because it flickers/over-brightens headless). Meter CONVERGED probe frames at standing viewpoints
    spread over the SCENE's navigable floor, take the MEDIAN high-percentile luminance, and scale ALL
    lights toward ``target`` -- iterating a few times to absorb the ACES tonemap non-linearity.

    Two independence properties are deliberate and load-bearing for a benchmark. The result is ONE fixed
    intensity for the whole episode: re-metering mid-render would flicker (the reason RTX auto-exposure is
    off here in the first place), would be scaling the world's lights rather than a camera parameter, and
    would make the frames path-dependent. And it is metered over the SCENE, not the episode, so every
    model, seed and route gets the same exposure on the same scene -- which also makes it computable once
    and cacheable. ``eye_z`` (the human's standing eye height) is what turns the floor plan into
    viewpoints; if no nav-grid probe is available, the uncached fallback uses the scene geometry centroid.

    Returns the diagnostics the episode manifest records, so a dark corpus is never invisible.
    """
    attrs = _all_light_intensity_attrs(stage)
    base = [float(a.Get() or 0.0) for a in attrs]
    probe_poses, free_fp, source = [], {}, "scene_geometry_centroid"
    if eye_z is not None:
        try:
            probe_poses, free_fp = _expose_scene_poses(stage, scene, float(eye_z))
        except Exception as exc:  # noqa: BLE001 - degrade to geometry, never lose the render
            print(
                f"[autoexp] scene free-space sampling failed ({exc}); geometry centroid only"
            )
    if probe_poses:
        source = "scene_navgrid"
    else:
        boxes = tuple(getattr(scene, "solid_boxes", lambda: ())())
        if boxes:
            centers = np.asarray(
                [
                    (
                        0.5 * (float(mn[0]) + float(mx[0])),
                        0.5 * (float(mn[1]) + float(mx[1])),
                    )
                    for _name, mn, mx in boxes
                ],
                dtype=np.float64,
            )
            ex, ey = (float(value) for value in centers.mean(axis=0))
            source = "scene_geometry_centroid"
        else:
            ex, ey = 0.0, 0.0
            source = "world_origin"
        ez = float(getattr(scene, "floor_z", 0.0))
        eye = np.array([ex, ey + 0.02, ez + 0.85], dtype=np.float64)
        look = np.array([ex, ey, ez], dtype=np.float64) - eye
        probe_poses = [(eye, look / (np.linalg.norm(look) + 1e-9))]
    rec = {
        "enabled": True,
        "target": float(target),
        "n_lights": len(attrs),
        "probe_frames": _EXPOSE_PROBE_FRAMES,
        "pose_source": source,
        "n_poses_probed": len(probe_poses),
        "light_scale": 1.0,
        "cached": False,
    }
    min_scale_override = _EXPOSE_MIN_SCALE_BY_SCENE_ID.get(str(scene_id))
    min_scale = (
        float(min_scale_override)
        if min_scale_override is not None
        else _EXPOSE_CUM_LO
    )
    if min_scale_override is not None:
        rec.update(
            {
                "min_light_scale": min_scale,
                "min_light_scale_source": f"task_override:{scene_id}",
            }
        )
        print(
            f"[autoexp] task override for {scene_id}: minimum light scale x{min_scale:.3f}; "
            "bypassing shared exposure cache"
        )

    live = {"cum": 1.0, "round": 0}

    def set_scale(cum: float) -> None:
        # Absolute (base * cum), never a compounding in-place multiply, so any scale is reachable
        # from any other without drift -- including the revert back to 1.0.
        for a, b in zip(attrs, base):
            a.Set(b * cum)
        live["cum"] = cum

    def probe() -> dict | None:
        live["round"] += 1
        return _probe_poses(cop, rcfg, annot, probe_poses, f"r{live['round']}")

    if not attrs:
        print("[autoexp] no scalable lights found; skipping calibration")
        return {**rec, "stop": "no_lights"}
    # Cache only the nav-grid probe whose complete geometry fingerprint is part of the key.
    inputs = (
        _expose_cache_inputs(attrs, base, free_fp, target, tries)
        if source == "scene_navgrid" and min_scale_override is None
        else None
    )
    hit = _expose_cache_get(inputs) if inputs is not None else None
    if hit is not None:
        if abs(float(hit["light_scale"]) - 1.0) > 1e-9:
            set_scale(float(hit["light_scale"]))
        print(
            f"[autoexp] CACHED for scene {scene_id}: {len(attrs)} light(s) "
            f"x{float(hit['light_scale']):.3f} (median p92 {hit.get('p92_before')}->"
            f"{hit.get('p92_after')}, spread {hit.get('p92_spread_after')})"
        )
        return {**rec, **hit, "cached": True}

    first = probe()
    if first is None:
        print("[autoexp] nothing measurable; skipping calibration (lights unchanged)")
        return {**rec, "stop": "no_valid_probe"}
    _expose_log("before", first, target)

    best_cum, best, stop = _expose_search(
        probe,
        set_scale,
        first,
        target,
        tries,
        min_scale=min_scale,
    )
    if abs(live["cum"] - best_cum) > 1e-9:
        set_scale(best_cum)
    _expose_log("after", best, target)
    print(
        f"[autoexp] {len(attrs)} light(s) x{best_cum:.3f} over {len(probe_poses)} "
        f"{source} pose(s) (median p92 {first['p92']:.3f}->{best['p92']:.3f}, stop={stop})"
    )
    out = _expose_record(rec, first, best, best_cum, stop)
    if inputs is not None:
        _expose_cache_put(inputs, {k: v for k, v in out.items() if k != "cached"})
    return out


def configure_fpv_render_settings() -> dict[str, int]:
    """Apply the renderer policy shared by dense and model-facing FPV products."""
    import carb

    aa_op = 3
    carb.settings.get_settings().set("/rtx/post/aa/op", aa_op)
    return {"aa_op": aa_op}


def _fpv_headlamp_intensity(scene_id: str | None) -> float:
    return _HEADLAMP_INTENSITY_BY_SCENE_ID.get(
        str(scene_id),
        _DEFAULT_HEADLAMP_INTENSITY,
    )


def _task_fill_light_plan(scene, scene_id: str | None) -> list[dict[str, object]]:
    plan = []
    for key, intensity in _TASK_FILL_LIGHTS_BY_SCENE_ID.get(str(scene_id), ()):
        entity = scene.resolve(key)
        plan.append(
            {
                "entity": key,
                "xyz": (
                    float(entity.xy[0]),
                    float(entity.xy[1]),
                    float(entity.top_z) + _TASK_FILL_LIGHT_HEIGHT_M,
                ),
                "intensity": float(intensity),
            }
        )
    return plan


def _author_task_fill_lights(stage, scene, scene_id: str | None) -> list[dict[str, object]]:
    plan = _task_fill_light_plan(scene, scene_id)
    if not plan:
        return []

    import fpv_scene
    from pxr import Sdf, UsdLux

    for index, item in enumerate(plan):
        lamp = UsdLux.SphereLight.Define(stage, f"/World/TaskFill{index:02d}")
        lamp.CreateRadiusAttr(_TASK_FILL_LIGHT_RADIUS_M)
        lamp.CreateIntensityAttr(float(item["intensity"]))
        lamp.CreateColorAttr((1.0, 0.98, 0.95))
        lamp.GetPrim().CreateAttribute(
            "visibleInPrimaryRay", Sdf.ValueTypeNames.Bool
        ).Set(False)
        fpv_scene.set_world_pose(lamp.GetPrim(), item["xyz"])
    return plan


def configure_fpv_lighting(
    app,
    stage,
    rcfg,
    cop,
    annot,
    scene,
    *,
    eye_z: float,
    scene_id: str | None = None,
) -> dict | None:
    """Apply the episode-level FPV lighting policy shared by every renderer.

    Base scene lights are authored before render products are created. This second phase runs after
    the product is warm: first meter one fixed exposure for the scene, then add the camera-rigged fill
    light so it cannot bias that meter. The order is part of the rendering contract.
    """
    autoexp = _calibrate_exposure(
        app,
        stage,
        rcfg,
        cop,
        annot,
        scene,
        eye_z=float(eye_z),
        scene_id=scene_id,
    )
    from pxr import UsdLux

    headlamp = UsdLux.DistantLight.Define(stage, "/World/FpvCam/Headlamp")
    headlamp_intensity = _fpv_headlamp_intensity(scene_id)
    headlamp.CreateIntensityAttr(headlamp_intensity)
    headlamp.CreateColorAttr((1.0, 1.0, 1.0))
    headlamp.CreateAngleAttr(1.5)
    task_fill_lights = _author_task_fill_lights(stage, scene, scene_id)
    if isinstance(autoexp, dict):
        autoexp["headlamp_intensity"] = headlamp_intensity
        autoexp["task_fill_lights"] = task_fill_lights
    return autoexp


# =================================================================================================
# Change 2: runtime door handling (CLOSED-first). Replaces the compile-time door_planner pre-pass, which
# went dead once navigates became runtime-resolved (Change 1). Keyed off each navigate's already-resolved
# stand: reach it on the CLOSED grid if possible, else open the crossed door(s) and re-plan.
# =================================================================================================
def _door_handle_contact(scene, entity_key, *, opened: bool = False):
    """The world xyz the OpenDoor producer grips (matches semantic_executor._swing): the door's grounded
    object_pos (which IS the handle for a plain nav door -- behavior_import registers it with xy=handle),
    overridden to the front-face handle for an openable CONTAINER that carries handle_xy/handle_z.
    """
    e = scene.resolve(entity_key)
    handle = np.asarray(scene.object_pos(entity_key), dtype=np.float64)
    if getattr(e, "handle_xy", None) is not None:
        handle = np.array(
            [
                e.handle_xy[0],
                e.handle_xy[1],
                e.handle_z if getattr(e, "handle_z", None) is not None else handle[2],
            ],
            dtype=np.float64,
        )
    if opened and getattr(e, "hinge_xy", None) is not None:
        hinge = np.asarray(
            (getattr(e, "meta", None) or {}).get(
                "hinge_xyz",
                (float(e.hinge_xy[0]), float(e.hinge_xy[1]), float(handle[2])),
            ),
            dtype=np.float64,
        )
        axis = np.asarray(
            getattr(e, "hinge_axis", None) or (0.0, 0.0, 1.0),
            dtype=np.float64,
        )
        handle = se._rotate_point_about_axis(
            handle,
            hinge,
            axis,
            float(getattr(e, "swing_sign", 1.0))
            * float(getattr(e, "swing_angle", 0.0)),
        )
    return handle


def _is_lidded_openable(scene, entity_key) -> bool:
    """Whether walking to this entity should target its lid's free edge."""
    resolve = getattr(scene, "resolve", None)
    if not callable(resolve):
        return False
    e = resolve(entity_key)
    return bool(
        getattr(e, "handle_xy", None) is not None
        and (getattr(e, "meta", None) or {}).get("openable_kind") == "lid"
    )


def _door_swing_route_score(
    scene,
    ctrl,
    sign,
    from_xy,
    target_xy=None,
    nav_grid_override=None,
) -> int | None:
    """Reachable floor area for one safe swing, or None when it blocks the requested route."""
    import init_validation as iv
    import nav_grid as _ng
    import nav_planner
    from goal_schema import PrimitiveFailure

    if not se.openable_stand_is_visually_clear(
        scene,
        str(ctrl.get("name") or ""),
        from_xy,
        swing_sign=float(sign),
    ):
        return None
    grid = scene.nav_grid if nav_grid_override is None else nav_grid_override
    pad = max(
        float(getattr(grid, "inflate", None) or iv.NAV_INFLATE_M),
        float(iv.HUMAN_FOOT_R),
    )
    angle = float(sign) * float(ctrl["swing_angle"])
    hinge = np.asarray(ctrl["hinge_xy"], dtype=np.float64)
    radius = np.asarray(ctrl["closed_handle"], dtype=np.float64)[:2] - hinge
    for fraction in (0.25, 0.5, 0.75, 1.0):
        theta = angle * fraction
        cosine, sine = math.cos(theta), math.sin(theta)
        rotated = np.array(
            [
                radius[0] * cosine - radius[1] * sine,
                radius[0] * sine + radius[1] * cosine,
            ],
            dtype=np.float64,
        )
        if getattr(grid, "wall_sdf", None) is not None and any(
            grid.wall_sdf_at(*(hinge + rotated * radial)) < -0.02
            for radial in (0.25, 0.5, 0.75, 1.0)
        ):
            return None
        mn, mx = _door_box(ctrl, angle * fraction)
        if (
            float(mn[0]) - pad <= float(from_xy[0]) <= float(mx[0]) + pad
            and float(mn[1]) - pad <= float(from_xy[1]) <= float(mx[1]) + pad
        ):
            return None

    view = _ng.passable_view(grid)
    occ = np.array(view.occ, dtype=bool, copy=True)
    mn, mx = _door_box(ctrl, angle)
    i0, j0 = view.world_to_cell(float(mn[0]) - pad, float(mn[1]) - pad)
    i1, j1 = view.world_to_cell(float(mx[0]) + pad, float(mx[1]) + pad)
    i0, i1 = max(0, min(i0, i1)), min(view.nx - 1, max(i0, i1))
    j0, j1 = max(0, min(j0, j1)), min(view.ny - 1, max(j0, j1))
    if i0 <= i1 and j0 <= j1:
        occ[j0 : j1 + 1, i0 : i1 + 1] = True
    view.occ = occ
    view.sdf = _ng._two_sided_sdf(occ, view.cell)
    if target_xy is not None:
        try:
            nav_planner.plan_path(view, from_xy, target_xy)
        except PrimitiveFailure:
            return None
    reachable = _ng.flood_free(view, view.world_to_cell(*from_xy))
    return int(np.count_nonzero(reachable))


def _door_swing_route_is_clear(
    scene,
    ctrl,
    sign,
    from_xy,
    target_xy,
    nav_grid_override=None,
) -> bool:
    """Whether one fully-open nav-door direction leaves the actor and onward route clear."""
    return (
        _door_swing_route_score(
            scene,
            ctrl,
            sign,
            from_xy,
            target_xy,
            nav_grid_override,
        )
        is not None
    )


def _door_swing_point_clearance_sq(ctrl, sign: float, point_xy) -> float:
    """Squared XY clearance from a point to the leaf in one fully-open pose."""
    if not {"swing_angle", "leaf_aabb", "hinge_xy", "closed_handle"}.issubset(ctrl):
        return 0.0
    angle = float(sign) * float(ctrl["swing_angle"])
    mn, mx = _door_box(ctrl, angle)
    point = np.asarray(point_xy, dtype=np.float64)[:2]
    lower = np.asarray(mn, dtype=np.float64)[:2]
    upper = np.asarray(mx, dtype=np.float64)[:2]
    delta = np.maximum(lower - point, np.maximum(point - upper, 0.0))
    return float(np.dot(delta, delta))


def _camera_safe_swing_choice(
    scene,
    ctrl,
    current,
    from_xy,
    target_xy,
    camera_route_for_sign,
):
    if camera_route_for_sign is None:
        return None, None
    candidates = []
    for sign in (current, -current):
        if not se.openable_stand_is_visually_clear(
            scene,
            str(ctrl.get("name") or ""),
            from_xy,
            swing_sign=float(sign),
        ):
            continue
        route = camera_route_for_sign(sign)
        if route is not None:
            candidates.append((sign, route))
    return max(
        candidates,
        default=(None, None),
        key=lambda item: (
            _door_swing_point_clearance_sq(ctrl, item[0], target_xy),
            item[0] == current,
        ),
    )


def _choose_runtime_door_swing(
    scene,
    ctrl,
    from_xy,
    target_xy=None,
    nav_grid_override=None,
    camera_route_for_sign=None,
):
    """Return the runtime swing decision without changing the scene or controller."""
    current = 1.0 if float(ctrl.get("swing_sign", 1.0)) >= 0.0 else -1.0
    selected_camera_route = None
    if target_xy is not None and camera_route_for_sign is not None:
        selected, selected_camera_route = _camera_safe_swing_choice(
            scene,
            ctrl,
            current,
            from_xy,
            target_xy,
            camera_route_for_sign,
        )
    elif target_xy is not None:
        candidates = [
            sign
            for sign in (current, -current)
            if _door_swing_route_is_clear(
                scene,
                ctrl,
                sign,
                from_xy,
                target_xy,
                nav_grid_override,
            )
        ]
        selected = max(
            candidates,
            default=None,
            key=lambda sign: (
                _door_swing_point_clearance_sq(ctrl, sign, target_xy),
                sign == current,
            ),
        )
    else:
        scored = [
            (
                score,
                _door_swing_point_clearance_sq(ctrl, sign, from_xy),
                sign == current,
                sign,
            )
            for sign in (current, -current)
            if (
                score := _door_swing_route_score(
                    scene,
                    ctrl,
                    sign,
                    from_xy,
                    nav_grid_override=nav_grid_override,
                )
            )
            is not None
        ]
        selected = max(scored, default=(None, None, None, None))[3]
    return current, selected, selected_camera_route


def _select_runtime_door_swing(
    scene,
    controllers,
    door,
    from_xy,
    target_xy=None,
    nav_grid_override=None,
    camera_route_for_sign=None,
    camera_route_out=None,
) -> bool:
    """Choose a safe swing that preserves the requested route or the most reachable floor."""
    entity_key = door if isinstance(door, str) else door.entity_key
    registered_doors = getattr(scene, "doors", None)
    if (
        isinstance(door, str)
        and registered_doors is not None
        and entity_key not in registered_doors
    ):
        return True
    ctrl = (controllers or {}).get(entity_key)
    if ctrl is None or ctrl.get("kind") != "revolute":
        return True
    if ctrl.get("openable_kind") == "lid":
        return True
    current, selected, selected_camera_route = _choose_runtime_door_swing(
        scene,
        ctrl,
        from_xy,
        target_xy,
        nav_grid_override,
        camera_route_for_sign,
    )
    if selected is None:
        return False
    if camera_route_out is not None and selected_camera_route is not None:
        camera_route_out.update(
            selected=True,
            force_straight=False,
            path=selected_camera_route["path"],
            reason="selected door swing by its camera-safe post-open route",
        )
    if selected == current:
        return True

    entity = scene.resolve(entity_key)
    direction_scale = float(selected) / current
    entity.swing_sign = float(selected)
    if getattr(entity, "leaf_signs", None):
        entity.leaf_signs = tuple(
            float(sign) * direction_scale for sign in entity.leaf_signs
        )
    ctrl["swing_sign"] = float(selected)
    logger.info(
        "[runtime-door] '%s': switched swing sign from %+.0f to %+.0f to %s",
        entity_key,
        current,
        selected,
        (
            "preserve a camera-validated post-open route"
            if selected_camera_route is not None
            else "keep the actor and onward route clear"
        ),
    )
    return True


def _runtime_door_staging_pose(scene, door_key, handle, reach_stand):
    """Return a centered camera-safe pause pose outside a closed room door."""
    entity = scene.resolve(door_key)
    handle = np.asarray(handle, dtype=np.float64)
    reach = np.asarray(reach_stand, dtype=np.float64)[:2]
    meta = getattr(entity, "meta", None) or {}
    aperture = np.asarray(meta.get("aperture_center", ()), dtype=np.float64)
    entry_axis = np.asarray(meta.get("entry_axis", ()), dtype=np.float64)
    if aperture.shape == (3,) and entry_axis.shape == (2,):
        center = aperture[:2]
        inward_norm = float(np.linalg.norm(entry_axis))
        if inward_norm <= 1e-9:
            return None
        outward = -entry_axis / inward_norm
        focus = aperture
    else:
        hinge_xy = getattr(entity, "hinge_xy", None)
        if hinge_xy is None or handle.shape != (3,):
            return None
        hinge = np.asarray(hinge_xy, dtype=np.float64)[:2]
        radius = handle[:2] - hinge
        panel_width = float(np.linalg.norm(radius))
        if panel_width <= 1e-9:
            return None
        center = hinge + 0.5 * radius
        outward = np.array([-radius[1], radius[0]], dtype=np.float64) / panel_width
        focus = np.array((center[0], center[1], handle[2]), dtype=np.float64)
    if float(np.dot(reach - center, outward)) < 0.0:
        outward *= -1.0
    reach_standoff = max(0.0, float(np.dot(reach - center, outward)))
    staging_standoff = max(NAV_DOOR_CAMERA_STANDOFF_M, reach_standoff)
    staging = center + outward * staging_standoff
    yaw = math.degrees(
        math.atan2(float(center[0] - staging[0]), -float(center[1] - staging[1]))
    )
    return (
        (float(staging[0]), float(staging[1])),
        float(yaw),
        np.asarray(focus, dtype=np.float64),
    )


def _camera_safe_door_approach_path(
    env,
    rig,
    scene,
    start_xy,
    stand_xy,
    yaw,
    *,
    path_override=None,
):
    """Validate a door-approach walk at its real global gait/camera phase."""
    import action_check
    import init_validation as iv

    collision_world = getattr(scene, "cworld", None)
    if not action_check.uses_egocentric_camera_collision_policy(scene) or not callable(
        getattr(collision_world, "sphere_overlap", None)
    ):
        return False, None
    held = getattr(env.st, "held", None)
    arm_idle = held.get("grasp_arm") if hasattr(held, "get") else None
    if arm_idle is None:
        arm_idle = getattr(rig, "arm_rest", rig.arm_grasp)
    safe = iv.camera_safe_navigation_path(
        scene,
        rig,
        start_xy,
        stand_xy,
        yaw,
        start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
        gait_profile=getattr(env, "g", None),
        gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
        arm_idle=arm_idle,
        path_override=path_override,
        camera_frame_offset=len(env.motion.frames),
        require_human_clear=True,
    )
    return True, None if safe is None else safe["path"]


def _runtime_door_action_is_camera_safe(
    env,
    rig,
    scene,
    door,
    stand_xy,
    yaw: float,
    sign: float,
    contact_fraction: float,
) -> bool:
    """Preview a room-door swing with the exact candidate leaf direction."""
    entity_key = door if isinstance(door, str) else door.entity_key
    entity = scene.resolve(entity_key)
    original_sign = float(getattr(entity, "swing_sign", 1.0))
    original_leaf_signs = getattr(entity, "leaf_signs", None)
    normalized = 1.0 if float(sign) >= 0.0 else -1.0
    current = 1.0 if original_sign >= 0.0 else -1.0
    scale = normalized / current
    entity.swing_sign = normalized
    if original_leaf_signs:
        entity.leaf_signs = tuple(float(value) * scale for value in original_leaf_signs)
    try:
        return se.openable_action_is_safe_from_stand(
            rig,
            getattr(env, "g", None),
            scene,
            env.st,
            Goal(
                Primitive.OPENDOOR,
                obj_key=entity_key,
                meta={
                    "_door_contact_fraction": float(contact_fraction),
                    "_camera_clearance_radius_m": NAV_DOOR_ACTION_CAMERA_CLEARANCE_M,
                },
            ),
            stand_xy,
            yaw,
            camera_frame_offset=len(env.motion.frames),
        )
    finally:
        entity.swing_sign = original_sign
        if original_leaf_signs is not None:
            entity.leaf_signs = original_leaf_signs


def _camera_safe_door_swing_route(
    env,
    rig,
    scene,
    controllers,
    door,
    start_xy,
    target_xy,
    target_yaw,
    sign: float,
):
    """Preview one nav-door direction together with the walk after it opens."""
    import action_check
    import init_validation as iv

    if not action_check.uses_egocentric_camera_collision_policy(scene):
        return None
    entity_key = door if isinstance(door, str) else door.entity_key
    entity = scene.resolve(entity_key)
    ctrl = (controllers or {}).get(entity_key)
    if ctrl is None or not getattr(entity, "leaf_paths", None):
        return None
    current = 1.0 if float(ctrl.get("swing_sign", 1.0)) >= 0.0 else -1.0
    boxes = se._open_leaf_boxes(
        scene,
        entity,
        swing_direction_scale=float(sign) / current,
    )
    if not boxes:
        return None
    held = getattr(env.st, "held", None)
    arm_idle = held.get("grasp_arm") if hasattr(held, "get") else None
    if arm_idle is None:
        arm_idle = getattr(rig, "arm_rest", rig.arm_grasp)
    yaw = float(getattr(env.st, "root_yaw", 0.0) if target_yaw is None else target_yaw)
    return iv.camera_safe_navigation_path(
        scene,
        rig,
        start_xy,
        target_xy,
        yaw,
        start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
        gait_profile=getattr(env, "g", None),
        gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
        arm_idle=arm_idle,
        camera_frame_offset=len(env.motion.frames),
        additional_excluded_roots=tuple(entity.leaf_paths),
        additional_open_leaf_boxes=boxes,
        require_human_clear=True,
    )


def _alternate_closed_door(
    grid,
    cur_xy,
    target_xy,
    by_index,
    opened,
    rejected_door_keys,
):
    """Choose the first unopened door on a route that avoids unusable entrances."""
    import door_planner
    import nav_grid as _ng
    import nav_planner
    from goal_schema import PrimitiveFailure

    door_ids = getattr(grid, "door_id", None)
    if door_ids is None:
        return None
    view = _ng.passable_view(grid)
    blocked = np.zeros(view.occ.shape, dtype=bool)
    for info in by_index.values():
        if info.entity_key in rejected_door_keys:
            blocked |= door_ids == int(info.prim_index)
    if not bool(np.any(blocked)):
        return None
    view.occ = np.asarray(view.occ, dtype=bool).copy()
    view.occ[blocked] = True
    view.sdf = _ng._two_sided_sdf(view.occ, view.cell)
    try:
        path = nav_planner.plan_path(view, tuple(cur_xy), tuple(target_xy))
    except PrimitiveFailure:
        return None
    return next(
        (
            by_index.get(index)
            for index in door_planner._crossed_doors(grid, path)
            if by_index.get(index) is not None
            and by_index[index].entity_key not in opened
            and by_index[index].entity_key not in rejected_door_keys
        ),
        None,
    )


def _legacy_runtime_door_approach_enabled() -> bool:
    """Restore the pre-staging door approach only for an explicit replay request."""
    return (
        os.environ.get("ISAAC_LEGACY_RUNTIME_DOOR_APPROACH") or ""
    ).strip() == "1"


def _pick_legacy_runtime_door_stand(
    env,
    ph,
    stage,
    rig,
    scene,
    door,
    current_xy,
    handle,
    iv_module,
):
    """Reproduce the handle-first stand search used by the archived ASK-12802 run."""
    transit_avoid = set()

    def _pick(contact):
        return iv_module.pick_stand_runtime(
            ph,
            stage,
            rig,
            scene,
            door.entity_key,
            current_xy,
            contact_override=contact,
            doors_open=False,
            start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
            gait_profile=getattr(env, "g", None),
            gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
            reach_tol=se._OPENABLE_REACH_TOL_M,
            transit_avoid_out=transit_avoid,
        )

    contact_fraction = 1.0
    door_stand = _pick(handle)
    if door_stand is not None:
        return door_stand, contact_fraction, transit_avoid
    entity = scene.resolve(door.entity_key)
    hinge_xy = getattr(entity, "hinge_xy", None)
    if hinge_xy is None:
        return None, contact_fraction, transit_avoid
    contact_fraction = 0.25
    panel_contact = np.array(
        [
            float(hinge_xy[0])
            + (float(handle[0]) - float(hinge_xy[0])) * contact_fraction,
            float(hinge_xy[1])
            + (float(handle[1]) - float(hinge_xy[1])) * contact_fraction,
            float(handle[2]),
        ],
        dtype=np.float64,
    )
    door_stand = _pick(panel_contact)
    if door_stand is not None:
        logger.info(
            "[runtime-door] '%s': handle stand unavailable -> using a reachable "
            "quarter-panel push",
            door.entity_key,
        )
    return door_stand, contact_fraction, transit_avoid


def _runtime_open_doors(
    env,
    ph,
    stage,
    rig,
    scene,
    cur_xy,
    target_xy,
    why=None,
    controllers=None,
    allow_camera_safe_fallback: bool = False,
    navigation_fallback_out=None,
    target_yaw=None,
    gait_profile=None,
    gait_cum_dist: float = 0.0,
) -> bool:
    """Ensure ``target_xy`` is reachable from ``cur_xy``, opening any CLOSED door in between (CLOSED-first).

    Returns True if reachable -- either directly on the live grid, or after opening one crossed door and
    replanning against the updated leaf occupancy until the target becomes reachable. Returns False = HONEST
    FAIL (blocked even with every door open, or a door handle / its OpenDoor is itself unreachable) -- the
    caller records PATH_BLOCKED and skips the leg (no fabricated stand, no straight-line penetration).
    """
    import door_planner
    import init_validation as iv
    import nav_grid as _ng
    import nav_planner
    from goal_schema import Goal, Primitive, PrimitiveFailure

    def _no(msg: str) -> bool:
        """Record WHY the door pre-pass gave up, then fail. ``False`` alone conflates FIVE unrelated
        causes -- no door layer, blocked even with every door open, no stand at the handle, the walk to
        that stand refused, the OpenDoor itself failing -- and the caller turns all five into one
        `path_blocked`. ``why`` is an optional out-list so a caller that reports the failure can put the
        real cause in the manifest; callers using this in a boolean expression simply omit it.
        """
        if why is not None:
            why.append(msg)
        return False

    grid = scene.nav_grid
    cur = (float(cur_xy[0]), float(cur_xy[1]))
    tgt = (float(target_xy[0]), float(target_xy[1]))
    doors = getattr(scene, "doors", None)
    by_index = {d.prim_index: d for d in (doors or {}).values()}
    opened = set(getattr(scene, "open_objs", set()) or set())
    queued_door = None
    deferred_pairs = set()
    rejected_door_keys = set()
    alternate_door_reroute_used = False

    def _validate_open_door_route(path) -> bool:
        if navigation_fallback_out is None or not iv._kinematically_open_leaf_roots(
            scene
        ):
            return True
        held = getattr(env.st, "held", None)
        arm_idle = held.get("grasp_arm") if hasattr(held, "get") else None
        if arm_idle is None:
            arm_idle = getattr(rig, "arm_rest", getattr(rig, "arm_grasp", None))
        fallback_yaw = float(
            getattr(env.st, "root_yaw", 0.0) if target_yaw is None else target_yaw
        )
        cached_path = navigation_fallback_out.get("path")
        preferred_path = (
            cached_path
            if cached_path is not None and nav_planner.path_starts_at(cached_path, cur)
            else path
        )
        safe = iv.camera_safe_navigation_path(
            scene,
            rig,
            cur,
            tgt,
            fallback_yaw,
            start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
            gait_profile=gait_profile,
            gait_cum_dist=gait_cum_dist,
            arm_idle=arm_idle,
            path_override=preferred_path,
            camera_frame_offset=len(env.motion.frames),
        )
        if safe is None:
            navigation_fallback_out.update(
                selected=True,
                force_straight=False,
                path=path,
                reason=(
                    "the open-door preview was conservative; deferred to the exact "
                    "generated-walk camera guard"
                ),
            )
            scene.egocentric_collision_fallback_used = True
            logger.warning(
                "[runtime-door] open-door route preview rejected %d waypoint(s); "
                "deferring to the exact generated-walk camera guard",
                len(path),
            )
            return True
        navigation_fallback_out.update(
            selected=True,
            force_straight=False,
            path=safe["path"],
            reason="validated route around a live open door",
        )
        return True

    def _camera_swing_selector(
        door, start_xy, contact_fraction, *, validate_action=True
    ):
        if navigation_fallback_out is None:
            return None

        def _validate(sign: float):
            if validate_action and not _runtime_door_action_is_camera_safe(
                env,
                rig,
                scene,
                door,
                start_xy,
                float(getattr(env.st, "root_yaw", 0.0)),
                sign,
                contact_fraction,
            ):
                return None
            return _camera_safe_door_swing_route(
                env,
                rig,
                scene,
                controllers,
                door,
                start_xy,
                tgt,
                target_yaw,
                sign,
            )

        return _validate

    def _fallback_door_candidates(required_doors=()):
        relaxed = _ng.egocentric_passable_view(grid, scene)
        relaxed_path = None
        try:
            if relaxed is not None:
                relaxed_path = nav_planner.plan_path(relaxed, cur, tgt)
        except PrimitiveFailure:
            pass
        crossed = (
            door_planner._crossed_doors(grid, relaxed_path or [cur, tgt])
            if getattr(grid, "door_id", None) is not None
            else tuple(by_index)
        )
        fallback_doors = []
        seen_doors = set()
        candidates = [*required_doors, *(by_index.get(index) for index in crossed)]
        for candidate in candidates:
            if (
                candidate is None
                or candidate.entity_key in opened
                or candidate.entity_key in seen_doors
            ):
                continue
            seen_doors.add(candidate.entity_key)
            fallback_doors.append(candidate)
        return relaxed_path, fallback_doors

    def _direct_fallback(reason: str, required_doors=()) -> bool:
        if navigation_fallback_out is None:
            return _no(reason)
        relaxed_path, fallback_doors = _fallback_door_candidates(required_doors)
        if fallback_doors:
            names = ", ".join(door.entity_key for door in fallback_doors)
            return _no(
                f"{reason}; refusing to open [{names}] without a reachable "
                "front-facing handle or panel stand"
            )
        try:
            restored_path = nav_planner.plan_path(grid, cur, tgt)
        except PrimitiveFailure:
            pass
        else:
            if _validate_open_door_route(restored_path):
                logger.warning(
                    "[runtime-door] %s; semantic opening restored the live nav route",
                    reason,
                )
                return True

        held = getattr(env.st, "held", None)
        arm_idle = held.get("grasp_arm") if hasattr(held, "get") else None
        if arm_idle is None:
            arm_idle = getattr(rig, "arm_rest", getattr(rig, "arm_grasp", None))
        fallback_yaw = float(
            getattr(env.st, "root_yaw", 0.0) if target_yaw is None else target_yaw
        )
        accepted_path = (
            navigation_fallback_out.get("path")
            if navigation_fallback_out is not None
            else None
        )
        safe = None
        if accepted_path:
            safe = iv.camera_safe_navigation_path(
                scene,
                rig,
                cur,
                tgt,
                fallback_yaw,
                start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
                gait_profile=gait_profile,
                gait_cum_dist=gait_cum_dist,
                arm_idle=arm_idle,
                path_override=accepted_path,
                camera_frame_offset=len(env.motion.frames),
            )
        if safe is None:
            safe = iv.camera_safe_navigation_path(
                scene,
                rig,
                cur,
                tgt,
                fallback_yaw,
                start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
                gait_profile=gait_profile,
                gait_cum_dist=gait_cum_dist,
                arm_idle=arm_idle,
                camera_frame_offset=len(env.motion.frames),
            )
        if safe is None and allow_camera_safe_fallback:
            safe = iv.camera_safe_navigation_path(
                scene,
                rig,
                cur,
                tgt,
                fallback_yaw,
                start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
                gait_profile=gait_profile,
                gait_cum_dist=gait_cum_dist,
                arm_idle=arm_idle,
                path_override=(cur, tgt),
                camera_frame_offset=len(env.motion.frames),
            )
        if safe is None and relaxed_path is not None:
            navigation_fallback_out.update(
                selected=True,
                force_straight=False,
                path=relaxed_path,
                reason=(f"{reason}; deferred to the exact generated-walk camera guard"),
            )
            scene.egocentric_collision_fallback_used = True
            logger.warning(
                "[runtime-door] %s; the preview was conservative, deferring a %d-waypoint "
                "narrow route to the exact generated-walk camera guard",
                reason,
                len(relaxed_path),
            )
            return True
        if safe is None:
            return _no(
                f"{reason}; no full-body or camera-safe narrow-clearance route remained "
                "after opening the crossed doors"
            )
        navigation_fallback_out.update(
            selected=True,
            force_straight=False,
            path=safe["path"],
            reason=reason,
        )
        scene.egocentric_collision_fallback_used = True
        logger.warning(
            "[runtime-door] %s; using a camera-validated narrow-clearance route with %d "
            "waypoint(s) after opening %d crossed door(s) semantically",
            reason,
            len(safe["path"]),
            len(fallback_doors),
        )
        return True

    if allow_camera_safe_fallback:
        _relaxed_path, fallback_doors = _fallback_door_candidates()
        if fallback_doors and callable(getattr(scene, "resolve", None)):
            queued_door = fallback_doors[0]
        else:
            return _direct_fallback(
                "the stand picker prevalidated a camera-clear direct route"
            )
    for _attempt in range(2 * len(by_index) + 1):
        try:
            live_path = nav_planner.plan_path(grid, cur, tgt)
        except PrimitiveFailure:
            pass
        else:
            if _validate_open_door_route(live_path):
                return True
            return _no(
                "the live route exists, but no camera-clear route remains around the "
                "kinematically open door"
            )
        if not by_index or getattr(grid, "door_id", None) is None:
            return _direct_fallback(
                "no door layer on this grid, and the target is unroutable on the live grid"
            )
        if queued_door is None:
            try:
                path = nav_planner.plan_path(_ng.passable_view(grid), cur, tgt)
            except PrimitiveFailure:
                reason = (
                    "unroutable after preserving the currently-open door leaves "
                    "in the passable view"
                )
                _relaxed_path, fallback_doors = _fallback_door_candidates()
                if not fallback_doors or not callable(getattr(scene, "resolve", None)):
                    return _direct_fallback(reason)
                door = fallback_doors[0]
                logger.warning(
                    "[runtime-door] %s; trying a physically reachable crossed door '%s'",
                    reason,
                    door.entity_key,
                )
            else:
                door = next(
                    (
                        by_index.get(index)
                        for index in door_planner._crossed_doors(grid, path)
                        if by_index.get(index) is not None
                        and by_index[index].entity_key not in opened
                    ),
                    None,
                )
        else:
            door = queued_door
            queued_door = None
        if door is None:
            return _direct_fallback(
                "the optimistic route crosses no unopened door, but the live grid remains blocked"
            )
        # VALIDATED door-approach stand: reachable + non-penetrating, facing the HANDLE the producer grips
        # (NOT the old blind handle+0.6m point that could land inside furniture). doors_open=False: the handle
        # must be reachable WITHOUT first opening THIS door (this door is still closed in the live grid).
        handle = _door_handle_contact(scene, door.entity_key)
        # Require a stand that actually reaches the handle. A prior doorway-front fallback could place the
        # actor cleanly near the door but still feed a metres-away best-effort arm pose into OpenDoor; the
        # leaf opened kinematically while the body crossed the wall. Failing this pre-pass is cheaper and
        # physically honest.
        #
        # Do not "fix" a `path_blocked` here by router-gating this pick. That was tried and MEASURED
        # inert (`the design record` C1): `T5-CAF-02.conflict` and `T5-GROCA-01` came back with
        # byte-identical failure histograms, because on an ego-guided episode the walk below never reaches
        # the router at all. `semantic_executor._assert_ego_unlocked` refuses it first -- `scene.ego_delivery`
        # is keyed by the PROGRAM's entity keys and a door is never one of them, so the lock compares the
        # body against the door stand itself, finds it further than `EGO_ARRIVE_M`, and raises PATH_BLOCKED
        # to stop the harness teleporting a body the guidance failed to deliver. That refusal is CORRECT.
        # The real gap is that the ego compiler never plans a leg to a door.
        if _legacy_runtime_door_approach_enabled():
            door_stand, contact_fraction, door_transit_avoid = (
                _pick_legacy_runtime_door_stand(
                    env,
                    ph,
                    stage,
                    rig,
                    scene,
                    door,
                    cur,
                    handle,
                    iv,
                )
            )
            semantic_open = door_stand is None
            if semantic_open:
                logger.info(
                    "[runtime-door] '%s': no safe handle/panel stand from "
                    "cur=(%.2f,%.2f) -> semantic door open",
                    door.entity_key,
                    cur[0],
                    cur[1],
                )
            else:
                (dsx, dsy), dsyaw = door_stand
                logger.info(
                    "[runtime-door] '%s': legacy stand (%.2f,%.2f) yaw=%.0f -> "
                    "walk + OpenDoor (CLOSED-first)",
                    door.entity_key,
                    dsx,
                    dsy,
                    dsyaw,
                )
                grid_transit_avoid = getattr(grid, "transit_avoid", None)
                prior_transit_avoid = (
                    set(grid_transit_avoid)
                    if grid_transit_avoid is not None
                    else set()
                )
                if grid_transit_avoid is not None:
                    grid_transit_avoid.clear()
                    grid_transit_avoid.update(door_transit_avoid)
                try:
                    _, _, navigation_info = env.run_goal(
                        Goal(
                            Primitive.NAVIGATE_TO,
                            target_xy=(dsx, dsy),
                            target_yaw=dsyaw,
                        )
                    )
                finally:
                    if grid_transit_avoid is not None:
                        grid_transit_avoid.clear()
                        grid_transit_avoid.update(prior_transit_avoid)
                if not navigation_info["ok"]:
                    return _direct_fallback(
                        f"walk to '{door.entity_key}' door stand failed: "
                        f"{navigation_info.get('detail') or navigation_info['reason'].value}",
                        required_doors=(door,),
                    )
            swing_from = cur if semantic_open else (dsx, dsy)
            swing_selected = _select_runtime_door_swing(
                scene,
                controllers,
                door,
                swing_from,
                tgt,
            )
            if not swing_selected:
                swing_selected = _select_runtime_door_swing(
                    scene,
                    controllers,
                    door,
                    swing_from,
                    tgt,
                    camera_route_for_sign=_camera_swing_selector(
                        door,
                        swing_from,
                        contact_fraction,
                        validate_action=False,
                    ),
                    camera_route_out=navigation_fallback_out,
                )
            if not swing_selected:
                return _direct_fallback(
                    f"neither swing direction for '{door.entity_key}' leaves a "
                    "live route to the target",
                    required_doors=(door,),
                )
            _, _, open_info = env.run_goal(
                Goal(
                    Primitive.OPENDOOR,
                    obj_key=door.entity_key,
                    meta={
                        "_door_contact_fraction": contact_fraction,
                        "_semantic_open": semantic_open,
                    },
                )
            )
            if not open_info["ok"]:
                return _direct_fallback(
                    f"OpenDoor on '{door.entity_key}' failed: "
                    f"{open_info.get('detail') or open_info['reason'].value}",
                    required_doors=(door,),
                )
            opened.add(door.entity_key)
            if not semantic_open:
                cur = (dsx, dsy)
            continue

        door_transit_avoid = set()
        door_approach_fallback = {}
        door_key = str(door.entity_key)
        door_current_xy = cur

        def _pick_door_stand(
            contact_point,
            fraction,
            *,
            centered,
            door_key=door_key,
            current_xy=door_current_xy,
            transit_avoid=door_transit_avoid,
            approach_out=door_approach_fallback,
        ):
            preview_goal = Goal(
                Primitive.OPENDOOR,
                obj_key=door_key,
                meta={"_door_contact_fraction": fraction},
            )
            candidate_navigation = {}
            found = iv.pick_stand_runtime(
                ph,
                stage,
                rig,
                scene,
                door_key,
                current_xy,
                contact_override=contact_point,
                nearest_to_contact=True,
                doors_open=False,
                start_yaw=float(getattr(env.st, "root_yaw", 0.0)),
                gait_profile=getattr(env, "g", None),
                gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
                reach_tol=se._OPENABLE_REACH_TOL_M,
                transit_avoid_out=transit_avoid,
                navigation_fallback_out=candidate_navigation,
                candidate_validator=_openable_candidate_validator(
                    env,
                    rig,
                    scene,
                    preview_goal,
                ),
                action_state=env.st,
                centered_openable=centered,
                openable_action_phase=True,
                openable_action_kind="open",
            )
            if found is not None:
                approach_out.clear()
                approach_out.update(candidate_navigation)
            return found

        contact_attempts = _openable_contact_attempts(
            scene,
            door.entity_key,
            handle,
        )
        door_stand = None
        door_stand_centered = True
        door_direct_path = None
        contact_fraction = 1.0
        for centered in (True, False):
            for contact_point, fraction in contact_attempts:
                door_stand = _pick_door_stand(
                    contact_point,
                    fraction,
                    centered=centered,
                )
                if door_stand is None:
                    continue
                if not centered:
                    picker_path = door_approach_fallback.get("path")
                    if picker_path is not None and not nav_planner.path_starts_at(
                        picker_path, cur
                    ):
                        logger.warning(
                            "[runtime-door] '%s': ignoring a stand-picker route whose "
                            "start no longer matches the actor",
                            door.entity_key,
                        )
                        picker_path = None
                    route_checked, candidate_path = _camera_safe_door_approach_path(
                        env,
                        rig,
                        scene,
                        cur,
                        door_stand[0],
                        door_stand[1],
                        path_override=picker_path,
                    )
                    if route_checked and candidate_path is None:
                        logger.warning(
                            "[runtime-door] '%s': rejecting non-centered %.2f-span "
                            "stand because its actor-to-action route is not camera/body safe",
                            door.entity_key,
                            fraction,
                        )
                        door_stand = None
                        continue
                    door_direct_path = candidate_path or picker_path
                door_stand_centered = centered
                contact_fraction = fraction
                if fraction < 1.0:
                    print(
                        f"[runtime-door] '{door.entity_key}': handle stand unavailable -> "
                        f"using a reachable {fraction:.2f}-span panel push"
                    )
                if not centered:
                    logger.warning(
                        "[runtime-door] '%s': centered stand unavailable; using a "
                        "non-centered stand that passed the full door-action safety preview",
                        door.entity_key,
                    )
                break
            if door_stand is not None:
                break
        if door_stand is None:
            rejected_door_keys.add(door.entity_key)
            alternate_door = (
                None
                if alternate_door_reroute_used
                else _alternate_closed_door(
                    grid,
                    cur,
                    tgt,
                    by_index,
                    opened,
                    rejected_door_keys,
                )
            )
            if alternate_door is not None:
                alternate_door_reroute_used = True
                queued_door = alternate_door
                logger.warning(
                    "[runtime-door] '%s': no safe physical stand; routing through "
                    "alternate entrance '%s'",
                    door.entity_key,
                    alternate_door.entity_key,
                )
                continue
            return _no(
                f"no reachable front-facing handle or panel stand for '{door.entity_key}'"
            )
        (dsx, dsy), dsyaw = door_stand
        staging_pose = _runtime_door_staging_pose(
            scene,
            door.entity_key,
            handle,
            (dsx, dsy),
        )
        if staging_pose is None:
            return _no(f"no centered camera staging pose for '{door.entity_key}'")
        (csx, csy), csyaw, door_focus = staging_pose
        print(
            f"[runtime-door] '{door.entity_key}': camera stand ({csx:.2f},{csy:.2f}) "
            f"yaw={csyaw:.0f} -> reach stand ({dsx:.2f},{dsy:.2f}) yaw={dsyaw:.0f} "
            "-> OpenDoor (CLOSED-first)"
        )
        staging_unreachable = False
        staging_target = (csx, csy)
        if not door_stand_centered:
            (csx, csy), csyaw = (dsx, dsy), dsyaw
            camera_staging_path = door_direct_path
            logger.warning(
                "[runtime-door] '%s': using the non-centered action stand directly "
                "from the actor's current pose",
                door.entity_key,
            )
        else:
            picker_path = door_approach_fallback.get("path")
            if picker_path is not None and not nav_planner.path_starts_at(
                picker_path, cur
            ):
                picker_path = None
            camera_checked, camera_staging_path = _camera_safe_door_approach_path(
                env,
                rig,
                scene,
                cur,
                (csx, csy),
                csyaw,
            )
            if camera_checked and camera_staging_path is None:
                reach_checked, direct_reach_path = _camera_safe_door_approach_path(
                    env,
                    rig,
                    scene,
                    cur,
                    (dsx, dsy),
                    dsyaw,
                    path_override=picker_path,
                )
                if reach_checked and direct_reach_path is not None:
                    (csx, csy), csyaw = (dsx, dsy), dsyaw
                    camera_staging_path = direct_reach_path
                    logger.warning(
                        "[runtime-door] '%s': centered camera stand is blocked; using the "
                        "camera-safe action stand directly",
                        door.entity_key,
                    )
                else:
                    staging_unreachable = True
        if staging_unreachable:
            try:
                staging_path = nav_planner.plan_path(
                    _ng.passable_view(grid),
                    cur,
                    staging_target,
                )
            except PrimitiveFailure:
                staging_path = None
            blocking_door = next(
                (
                    by_index.get(index)
                    for index in (
                        door_planner._crossed_doors(grid, staging_path)
                        if staging_path is not None
                        else ()
                    )
                    if by_index.get(index) is not None
                    and by_index[index].entity_key not in opened
                    and by_index[index].entity_key != door.entity_key
                ),
                None,
            )
            if blocking_door is not None:
                pair = (door.entity_key, blocking_door.entity_key)
                if pair in deferred_pairs:
                    return _no(
                        f"cyclic closed-door dependency while approaching '{door.entity_key}'"
                    )
                deferred_pairs.add(pair)
                queued_door = blocking_door
                print(
                    f"[runtime-door] '{door.entity_key}': staging route is blocked by "
                    f"closed '{blocking_door.entity_key}'; opening that nearer blocker first"
                )
                continue
            rejected_door_keys.add(door.entity_key)
            alternate_door = (
                None
                if alternate_door_reroute_used
                else _alternate_closed_door(
                    grid,
                    cur,
                    tgt,
                    by_index,
                    opened,
                    rejected_door_keys,
                )
            )
            if alternate_door is not None:
                alternate_door_reroute_used = True
                queued_door = alternate_door
                logger.warning(
                    "[runtime-door] '%s': no camera-clear approach; routing through "
                    "alternate entrance '%s'",
                    door.entity_key,
                    alternate_door.entity_key,
                )
                continue
            return _no(
                f"no camera-clear route reaches the staging stand for '{door.entity_key}'"
            )
        grid_transit_avoid = getattr(grid, "transit_avoid", None)
        prior_transit_avoid = (
            set(grid_transit_avoid) if grid_transit_avoid is not None else set()
        )
        if grid_transit_avoid is not None:
            grid_transit_avoid.clear()
            grid_transit_avoid.update(door_transit_avoid)
        try:
            staging_meta = {
                "_navigation_arrival_gaze": tuple(float(v) for v in door_focus),
            }
            if camera_staging_path is not None:
                staging_meta["_camera_safe_navigation_path"] = camera_staging_path
            _, _, _in = env.run_goal(
                Goal(
                    Primitive.NAVIGATE_TO,
                    target_xy=(csx, csy),
                    target_yaw=csyaw,
                    meta=staging_meta,
                )
            )
            if not _in["ok"]:
                return _no(
                    f"walk to '{door.entity_key}' camera staging stand failed: "
                    f"{_in.get('detail') or _in['reason'].value}"
                )
            if math.dist((csx, csy), (dsx, dsy)) > 0.02:
                camera_checked, door_approach_path = _camera_safe_door_approach_path(
                    env,
                    rig,
                    scene,
                    (csx, csy),
                    (dsx, dsy),
                    dsyaw,
                )
                if camera_checked and door_approach_path is None:
                    return _no(
                        f"no camera-clear centered reach approach for '{door.entity_key}'"
                    )
                reach_meta = {
                    "_navigation_fixed_gaze": tuple(float(v) for v in door_focus),
                }
                if door_approach_path is not None:
                    reach_meta["_camera_safe_navigation_path"] = door_approach_path
                _, _, _in = env.run_goal(
                    Goal(
                        Primitive.NAVIGATE_TO,
                        target_xy=(dsx, dsy),
                        target_yaw=dsyaw,
                        meta=reach_meta,
                    )
                )
        finally:
            if grid_transit_avoid is not None:
                grid_transit_avoid.clear()
                grid_transit_avoid.update(prior_transit_avoid)
        if not _in["ok"]:
            print(
                f"[runtime-door] '{door.entity_key}': centered reach approach FAILED "
                f"({_in['reason'].value}: {_in.get('detail') or 'no detail reported'}) -> FAIL"
            )
            return _no(
                f"centered reach approach to '{door.entity_key}' failed: "
                f"{_in.get('detail') or _in['reason'].value}"
            )
        swing_from = (dsx, dsy)
        swing_selected = _select_runtime_door_swing(
            scene,
            controllers,
            door,
            swing_from,
            tgt,
            camera_route_for_sign=_camera_swing_selector(
                door,
                swing_from,
                contact_fraction,
            ),
            camera_route_out=navigation_fallback_out,
        )
        if not swing_selected and navigation_fallback_out is not None:
            def _locally_safe_swing(
                sign: float,
                *,
                door=door,
                swing_from=swing_from,
                contact_fraction=contact_fraction,
            ):
                if not _runtime_door_action_is_camera_safe(
                    env,
                    rig,
                    scene,
                    door,
                    swing_from,
                    float(getattr(env.st, "root_yaw", 0.0)),
                    sign,
                    contact_fraction,
                ):
                    return None
                return {"path": ()}

            swing_selected = _select_runtime_door_swing(
                scene,
                controllers,
                door,
                swing_from,
                tgt,
                camera_route_for_sign=_locally_safe_swing,
            )
            if swing_selected:
                logger.warning(
                    "[runtime-door] '%s': post-open route preview was conservative; "
                    "using a locally camera-safe swing and validating the live route after opening",
                    door.entity_key,
                )
        if not swing_selected:
            return _no(
                f"neither swing direction for '{door.entity_key}' leaves a live route to the target"
            )
        _, _, _io = env.run_goal(
            Goal(
                Primitive.OPENDOOR,
                obj_key=door.entity_key,
                meta={
                    "_door_contact_fraction": contact_fraction,
                    "_camera_clearance_radius_m": NAV_DOOR_ACTION_CAMERA_CLEARANCE_M,
                },
            )
        )
        if not _io["ok"]:
            print(
                f"[runtime-door] '{door.entity_key}': OpenDoor FAILED "
                f"({_io['reason'].value}: {_io.get('detail') or 'no detail reported'}) -> FAIL"
            )
            return _no(
                f"OpenDoor on '{door.entity_key}' failed: "
                f"{_io.get('detail') or _io['reason'].value}"
            )
        opened.add(door.entity_key)
        cur = (
            dsx,
            dsy,
        )
    return _no(
        "door-opening replan budget exhausted while the live route remained blocked"
    )


def _ego_route_gate(scene) -> bool:
    """Is this episode ego-guided? ``ego_route.recompile`` sets ``scene.ego_delivery`` exactly when it
    compiled ego legs, so this is a property of the EPISODE, not a family name: a future family that
    guides a body it cannot let see the room inherits it, and every episode with no ego verb reads
    False and takes the byte-identical pre-existing path.

    It turns on ``init_validation.pick_stand_runtime(route_gate=...)``. The connectivity gate accepts a
    stand when a start-reachable free cell lies within a shuffle of it -- correct for a stand hugging
    the counter it works at -- but where that disagrees with the router the picker hands back a stand
    whose ``navigate_to`` then fails ``path_blocked``, losing the command and every sub-goal below it.
    Measured over the 23 blind-guidance rows: 9 of them, 2-13 blocked commands each.

    Module level on purpose: ``setup_episode`` (the leg compiler's stand prediction) and ``run_render``
    (the runtime pick) are different functions and MUST ask the same question, or the compiler's model
    of where the body ends up stops being a fixed point.

    ``BENCH_STAND_ROUTE_GATE=0|1`` forces it either way. That exists for ONE reason: the honest A/B.
    Comparing an ego row before and after this change compares two different code snapshots and
    therefore attributes to the gate whatever else landed in between; forcing it OFF gives a baseline
    at the IDENTICAL code, which is the only comparison that isolates the gate. It doubles as the
    owner's off-switch, so disabling the fix never needs an edit."""
    override = (os.environ.get("BENCH_STAND_ROUTE_GATE") or "").strip().lower()
    if override in ("0", "off", "false", "no"):
        return False
    if override in ("1", "on", "true", "yes"):
        return True
    return getattr(scene, "ego_delivery", None) is not None


# =================================================================================================
# Render.
# =================================================================================================
def setup_episode(args):
    """Boot + build + settle + validate ONE episode's scene, returning a context dict the render loop
    (and the persistent IsaacEnv) both consume.

    This is the former FIRST HALF of ``run_render``, extracted VERBATIM so the one-shot render
    (``run_render`` below) and a reset-many RL env share ONE setup path -- no duplication, no drift.
    Returns an int early-exit status for the preview / --check-family / seed-abort paths, else a dict of
    the built app/stage/ph/rig/scene/env/goals + the settle artifacts the render half needs.
    Behavior-preserving: ``run_render`` unpacks the dict back into the same local names, so the render
    half is byte-identical."""
    from fpv_core import AppCfg, RenderCfg

    W, Hh = 960, 760
    # active_gpu: the RL sim-server pool pins each server to its own GPU (multi-sim on one node) via
    # AppCfg.active_gpu/physics_gpu -- NOT CUDA_VISIBLE_DEVICES (that breaks Isaac's renderer<->CUDA
    # interop). Default 0 (getattr) -> byte-identical for the benchmark render (no args.active_gpu).
    _gpu = int(getattr(args, "active_gpu", 0))
    app_cfg = AppCfg(
        headless=True,
        width=W,
        height=Hh,
        renderer="RayTracedLighting",
        active_gpu=_gpu,
        physics_gpu=_gpu,
        disable_viewport_updates=not bool(getattr(args, "render_obs", True)),
    )
    rcfg = RenderCfg(capture_fps=FPS, playback_fps=FPS, max_frames=600, rt_subframes=6)

    import fpv_core

    app = fpv_core.launch_app(
        app_cfg
    )  # sets the OmniGibson MDL search path before RTX init (global fix)
    import render_runtime  # proven _loco_cfg / _boost_lights / _clear_frame_pngs

    fpv_core.post_launch_setup(app_cfg, render_runtime._loco_cfg())
    import fpv_render
    import smplh_camera

    stage = fpv_core.get_stage()
    import physics_human as ph

    ph.build_physics_scene(
        stage
    )  # the cabinet articulation needs a sim owner to drive the drawer joint
    ph.set_scene_timestep(
        stage, FPS
    )  # one PhysX substep per captured frame (FPS=12; default 60 -> lag)

    # Door-swing feature: ON by default for BEHAVIOR scenes (the pre-pass only inserts an OpenDoor when a
    # walk crosses a CLOSED door, so a doorless path is unchanged; no [OpenDoor] needed in the action
    # string). --no-doors disables it (closed door -> plain obstacle -> straight-line). Set before any
    # builder runs so navcheck AND render see it. Non-BEHAVIOR scenes never read it.
    import behavior_import as _bimp

    _bimp.ENABLE_DOORS = not bool(getattr(args, "no_doors", False))

    if getattr(args, "physcheck", False):
        # GATE: prove N substeps deliver under the render clock BEFORE building any scene or wiring P3.
        return _run_physcheck(app, stage, ph, rcfg)

    if getattr(args, "initcheck", False):
        # INIT non-penetration harness (V0/V1/V2): free-cell start + per-target stand-pose validate for
        # N seeds against the EXACT human-vs-scene gate. Builds its own scene (own rig) + returns.
        return _run_initcheck(app, stage, ph, args)

    if getattr(args, "navcheck", False):
        # NAV-GRID doorway/connectivity probe: build the nav grid on the same colliders + assert the
        # start reaches every stand pose through free space (doorways open). Builds its own scene.
        return _run_navcheck(app, stage, ph, args)

    if getattr(args, "fallback_check", False):
        # P4 FALLBACK proof: force an unplaceable init + assert it degrades to episode_valid=False
        # (least-penetrating accepted + flagged), no crash / no silent accept. Builds its own scene.
        return _run_fallbackcheck(app, stage, ph, args)

    rig = core.load_family("side_wrap")
    # Define a grip seat (obj_T_in_wrist) so warp_arm_to_seat can land the hand on a handle -- same
    # setup semantic_executor.main does before run_dry3. No object is rendered (no /World/Obj built).
    _build, _otw = core.load_object_def("cup")
    core.set_object(rig, _build, _otw)
    # Derive a relaxed arms-at-sides rest pose (rig.arm_rest); else _seated_pose + the non-working arm
    # fall back to the grip template's outstretched (T-pose-ish) arms. Same call wpp.run_render uses.
    try:
        core.derive_rest_place_arms(rig)
    except Exception as exc:  # noqa: BLE001
        print(f"[step3r] derive_rest_place_arms failed ({exc}); using canonical arm")
    # The init gate (_validate_init_for_render) poses the human with EXACTLY this rig, so hand it down
    # instead of letting it build a second identical one. Snapshotted HERE, before the behavior_grab /
    # obj_def rig rebuild and the live_wrap rhp_open assignment below, so the gate keeps receiving the
    # plain side-wrap+cup rig it used to build for itself. A SHALLOW copy is what makes that exact: those
    # later steps REBIND attributes (rig.rhp_open = ..., set_object) on the rig they are given, and when
    # that is this same object the alias would leak the edit into the gate. No array is ever mutated in
    # place, so sharing them is safe. ISAAC_RIG_EQUIV_CHECK=1 proves the equality field by field.
    init_gate_rig = copy.copy(rig)
    g = gait.build_gait()

    builders = _scene_builders(seed=getattr(args, "seed", 0))
    scene, controllers, program, tpv_look, tpv_eye = builders[args.scene](stage, app)
    scene.strict_acceptance = bool(getattr(args, "strict_acceptance", False))
    # DE-SPECIALIZATION: the unified pipeline -- SSOT collision oracle + pose-planner cell-search stand +
    # post-action physics validate/correct + reject-if-broken -- is the DEFAULT for EVERY scene, with NO
    # per-scene opt-in flags. Any video / scene / task runs through the SAME latest system; where colliders
    # or the oracle are absent the code degrades gracefully. Set uniformly HERE so no scene builder ever
    # special-cases behaviour (the builders no longer set these).
    scene.unified_geom = True
    scene.post_action_check = True
    scene.soft_init = True
    _place_toggle_indicators(stage, scene)
    _place_liquid_visuals(stage, scene)
    # Opt-in: after passing through an auto-opened door, walk to the far side + CloseDoor it (default: leave
    # it open -- harmless, already through). Only meaningful for a scene with doors.
    if getattr(scene, "doors", None) and getattr(args, "close_doors_behind", False):
        scene.close_doors_behind = True
    has_obj = any(e.role is Role.GRASPABLE for e in scene.entities.values())
    # NATURAL GRASP opt-in: a scene may set scene.live_wrap itself; --natural-grasp forces it on for any
    # scene. When on, produce_pick builds the object SDF + seeds the per-shape grip (else canned, unchanged).
    if bool(getattr(args, "natural_grasp", False)):
        scene.live_wrap = True

    if getattr(args, "preview", False):
        # CHEAP PREVIEW: on the freshly-BUILT scene, set up the camera/lights/render-products, snapshot each
        # labeled container from a fixed camera, and EXIT -- BEFORE the (slow ~20 min) motion planning +
        # episode. Lets a scene change be visually checked without the full render. Gated on --preview.
        import fpv_render as _fr
        import smplh_camera as _sc

        cop = _sc.make_fpv_camera(stage, "/World/FpvCam", W, Hh)
        _sc.make_tpv_camera(stage, "/World/TpvCam", W, Hh)
        fpv_core.add_lighting(stage, app_cfg)
        _rpf, _rpt, a_f, a_t = _fr.setup_products(
            "/World/FpvCam", "/World/TpvCam", rcfg, W, Hh
        )
        _tl = fpv_core.start_timeline(rcfg.playback_fps)
        _fr.settle_and_validate(app, _tl, a_f, app_cfg, annot_tpv=a_t)
        configure_fpv_lighting(
            app,
            stage,
            rcfg,
            cop,
            a_f,
            scene,
            eye_z=float(getattr(scene, "floor_z", 0.0)) + 1.6,
            scene_id=str(args.scene),
        )
        _tag = args.tag or f"step3_{args.scene}"
        _out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fpv_outputs"
        )
        _pvdir = os.path.join(_out, f"{_tag}_preview")
        render_runtime._clear_frame_pngs(_pvdir, _pvdir)
        _shots = getattr(scene, "preview_shots", None) or []
        print(f"[preview] {len(_shots)} shot(s) -> {_pvdir}")
        for _i, (_lab, _eye, _look) in enumerate(_shots):
            _sc.drive_tpv(
                cop,
                tuple(float(v) for v in _eye),
                tuple(float(v) for v in _look),
            )
            for _ in range(12):  # let RTX denoise/accumulate on the new pose
                _fr.advance_once(rcfg)
            _fr.read_and_save(a_f, _pvdir, _i, W, Hh, rcfg)
            print(f"[preview] shot {_i:02d} '{_lab}'")
        print(f"[preview] DONE -> {_pvdir}")
        return 0

    # BEHAVIOR graspable (gated on scene.behavior_grab; existing scenes never set it -> unchanged):
    # rebuild the rig with a shape-matched atlas family + proven seat (round apple -> round_wrap+ball),
    # then the object def points at the baked BEHAVIOR USD; the live SDF wrap conforms the fingers.
    _bg = getattr(scene, "behavior_grab", None)
    if _bg:
        import behavior_import

        rig = core.load_family(_bg.get("family", "round_wrap"))
        _bs_build, _bs_otw = core.load_object_def(_bg.get("seat", "ball"))
        core.set_object(rig, behavior_import.behavior_build_params(_bg["obj"]), _bs_otw)
        try:
            core.derive_rest_place_arms(rig)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[step3r] behavior derive_rest_place_arms failed ({exc}); using canonical arm"
            )
        print(f"[step3r] BEHAVIOR grab object {_bg}")

    # Step-4 (additive): a scene may pick a different graspable proxy for the /World/Obj TOOL via
    # scene.obj_def (e.g. a marker as a "knife"). Existing scenes never set it -> byte-identical (cup).
    _od = getattr(scene, "obj_def", None)
    if _od and not _bg:
        _b2, _o2 = core.load_object_def(_od)
        core.set_object(rig, _b2, _o2)
        try:
            core.derive_rest_place_arms(rig)
        except Exception as exc:  # noqa: BLE001
            print(f"[step3r] obj_def={_od} derive_rest_place_arms failed ({exc})")
        print(f"[step3r] Step-4 obj_def -> {_od}")

    # NATURAL GRASP: give the rig a relaxed OPEN hand (mild finger curl) so reach/release ramps don't
    # splay into a flat "starfish"; the grasp end (hand_close=1) is unchanged. Gated on live_wrap + set
    # AFTER any behavior/obj_def rig rebuild so it survives. Other scenes leave rhp_open None (unchanged).
    if getattr(scene, "live_wrap", False):
        rig.rhp_open = core.build_rhand_pose(
            {"index": 20.0, "middle": 20.0, "ring": 22.0, "pinky": 22.0},
            thumb=(14.0, 12.0),
        )

    # ---- INIT non-penetration validation, ENFORCED AT RENDER (the seeded random-init path). ----------
    # Default (no --seed): render_start_xy stays None -> make_state uses the scene's hardcoded start_xy.
    # With --seed N: build furniture colliders, free-cell place the human, GATE on episode_valid (a
    # penetrating START is NEVER rendered -- abort with a clear flag), and build the per-episode nav grid
    # over the object contacts. The validated start_xy is threaded into make_state via env.reset(). Stands
    # are NO LONGER pre-picked here (Change 1): each is chosen at RUNTIME in the goal loop below,
    # nearest-reachable from the human's live position. This runs BEFORE the object settle (reuses the sim).
    render_start_xy = None
    render_episode_valid = (
        True  # P0: fed into --check ok; stays True for non-seeded hand-tuned scenes
    )
    if getattr(args, "seed", None) is not None:
        rc = _validate_init_for_render(
            app, stage, ph, scene, program, int(args.seed), rig=init_gate_rig
        )
        # P1 (RL contract): bounded seed RE-SAMPLE. Only the START is seed-dependent (stand poses are
        # pure geometry), so re-sample ONLY while the failure is the start placement; each retry draws a
        # fresh free-cell start. Exhausting the budget keeps the last (least-penetrating) result, which is
        # then honestly flagged below. A stand-fallback never re-samples (a new seed cannot move a stand).
        _resample_k = int(getattr(args, "resample", 4))
        _cand_seed = int(args.seed)
        while (
            not rc["episode_valid"]
            and str(rc.get("reason", "")).startswith("start")
            and _cand_seed < int(args.seed) + _resample_k
        ):
            _cand_seed += 1
            print(
                f"[step3r] P1 re-sample: seed {int(args.seed)} start invalid ({rc['reason']}) "
                f"-> retry seed {_cand_seed}"
            )
            rc = _validate_init_for_render(
                app, stage, ph, scene, program, _cand_seed, rig=init_gate_rig
            )
        render_start_xy = rc["start_xy"]
        render_episode_valid = bool(rc["episode_valid"])
        # P1: soft_init now KEEPS the init-gate's VALIDATED (backed-off, non-penetrating) stands instead of
        # discarding them for the raw stand_pose_for. The old discard existed only because the boundingCube
        # back-off ladder pushed stands out of reach; with behavior_collider's precise per-link geometry a
        # validated stand both reaches the object AND stays clear of the furniture, so the human no longer
        # stands into the counter/desk (the "arm through table" cause). START + nav grid are kept as before.
        if not rc["episode_valid"]:
            # SOFT-INIT (gated on scene.soft_init): a tight scene (e.g. a BEHAVIOR kitchen where reaching a
            # counter object forces the body near the boundingCube-approximated counter/cabinets) fails the
            # conservative stand gate even though the runtime nudge keeps the ACTUAL motion clear. When the
            # scene opts in, PROCEED with the built nav grid + runtime nudge so navigation (path-planning +
            # wall anti-pen + eye push-out) engages, instead of aborting into a nav-less straight line.
            # Default -> hard abort (never render a penetrating init); byte-identical for every existing scene.
            if getattr(scene, "soft_init", False):
                _navon = getattr(scene, "nav_grid", None) is not None
                # P1: do NOT silently accept a penetrating init. RENDER it (so it can be inspected) but the
                # AUTHORITATIVE verdict is the post-action compliance pass below, HARD-LABELLED into the
                # per-episode manifest -> the RL loop DISCARDS it; a penetrating episode is never counted clean.
                print(
                    f"[step3r] SEED {args.seed}: init episode_valid=FALSE ({rc['reason']}) after re-sample; "
                    f"scene.soft_init -> RENDER for inspection, HARD-LABELLED invalid in the manifest "
                    f"(RL must discard; post-action compliance is the true verdict; "
                    f"nav planning {'ON' if _navon else 'OFF -> straight-line'})"
                )
            else:
                print(
                    f"[step3r] SEED {args.seed}: init validation episode_valid=FALSE "
                    f"({rc['reason']}) -> ABORT (never render a penetrating init). "
                    f"Re-run with a different --seed or fix the scene."
                )
                _exit_process(3)
                return 3
        else:
            print(
                f"[step3r] SEED {args.seed}: init validation ENFORCED -- start_xy="
                f"({render_start_xy[0]:.2f},{render_start_xy[1]:.2f}) cell={rc['start_cell']} "
                f"episode_valid=True -> rendering (stands picked at runtime, nearest-reachable)"
            )

    # Support placement validates a route from the initial actor pose before runtime door controllers
    # are wired. Carve only scene room-door portals in this planning snapshot; the later authored-open
    # application performs the physical rotation and declares semantic state once controllers exist.
    planning_open = prepare_authored_initial_room_door_navigation(scene)
    if planning_open:
        print(
            "[step3r] authored initially-open room-door navigation: "
            + ", ".join(planning_open)
        )

    # P3 gate: a scene opts into DURING-RENDER object dynamics by setting scene.free_dynamic_prims (a
    # list of prim paths), exactly mirroring scene.clutter_prims. fdp None/empty (openclose/highreach/
    # sit/walkby AND the existing frozen-kinematic clutter demo) -> the ENTIRE P3 path below is skipped:
    # no proxy in the render loop, no substepping, no on_post_step, every settled body still frozen
    # KINEMATIC. ONLY build_put_penetrate (and future continuous-dynamics scenes) set it.
    fdp = getattr(scene, "free_dynamic_prims", None)
    fdp_paths = [str(p).rstrip("/") for p in fdp] if fdp else []
    graspable_paths = {
        str(entity.prim_path).rstrip("/")
        for entity in scene.entities.values()
        if entity.role is Role.GRASPABLE and entity.prim_path
    }
    if has_obj and not getattr(scene, "grasp_objs", None):
        graspable_paths.add("/World/Obj")
    fdp_object_paths = set(fdp_paths) & graspable_paths
    fdp_obj = "/World/Obj" in fdp_object_paths
    fdp_blockers = [path for path in fdp_paths if path not in fdp_object_paths]
    reg: dict = (
        {}
    )  # SINK 2: read-only per-frame free-object registry (PhysX owns these prims)

    # ---- CDS object physics (D1/D3): a ONE-TIME render=False settle so object REST poses are
    # PHYSICS-DERIVED, not hardcoded literals. Runs in BOTH --check and render (both boot the app); the
    # crux invariant that keeps this additive: settle once here, then flip every body back to KINEMATIC
    # BEFORE the render loop, so the per-frame flow (realize -> set_object_world guarded by not
    # drove_drawer, the nudge, the drawer joint drive, the camera) stays byte-identical -- only a rest
    # pose changed. GATED on has_obj / a clutter set, so openclose/sit/walkby skip it entirely.
    scene_clutter = getattr(scene, "clutter_prims", None)
    settle_report: list[tuple[str, float, float, float, bool]] = (
        []
    )  # (label, init_z, rested_z, mm, ok)
    pen_pairs: list = (
        []
    )  # penetrating clutter pairs (all_pairs_separation), empty = physically clean
    clutter_rest_xy: dict[str, tuple[float, float]] = (
        {}
    )  # settled XY per prim (edge-margin assert)
    clutter_traj: list | None = (
        None  # FIX C: recorded drop trajectory for the visible-settle replay
    )
    if has_obj or scene_clutter:
        import object_physics as ophys

        ophys.reset_offtarget()
        # Settle at a FINE 1/60 physics step matching the SimulationContext (object_physics.get_sim).
        # run_render set the scene TimeStepsPerSecond=FPS(12) for the drawer drive; at that coarse dt a
        # dropped prop moves ~7cm/substep and TUNNELS through the support. Bracket the settle at 60, then
        # RESTORE FPS below so the render-loop drawer drive is byte-identical.
        ph.set_scene_timestep(stage, 60)
        # /World/Obj is the single carried/graspable rig object; build it once here (BEFORE the reused
        # later set_object_world). build_object is NOT idempotent (re-adds the ref), so it is called ONCE.
        # MULTI-graspable scenes (scene.grasp_objs) build/settle their own prims below instead.
        _multi_obj = getattr(scene, "grasp_objs", None)
        _dynamic_settle_keys = objects_requiring_dynamic_settle(program, scene)
        if has_obj and not _multi_obj:
            core.build_object(stage, rig)
            core.enable_physics_refetch(rig, stage, "/World/Obj", "/World/Body")
            # FIX D (clutter only): recolor the carried mug a distinct blue so it reads as the object
            # being placed, not a pre-existing green mug among the clutter. Other scenes keep the mug
            # unchanged (the recolor is gated on the clutter set).
            if scene_clutter:
                _recolor_prim(stage, "/World/Obj", (0.12, 0.35, 0.85))
            elif getattr(scene, "obj_color", None):
                # Full-env scenes: recolor the grasped object a strong hue so it READS against the bright
                # reflective warehouse/hospital floor in the downward-looking FPV grab/put frames.
                _recolor_prim(stage, "/World/Obj", scene.obj_color)
        # Static support colliders (exact-triangle mesh: convexHull/decomposition would leak -> tunnel).
        # Each guarded by IsValid so scenes without a given surface are unaffected.
        for spath in (
            "/World/Room/Floor",
            "/World/Shelf",
            "/World/ShelfLegL",
            "/World/ShelfLegR",
            "/World/Table",
        ):
            sp = stage.GetPrimAtPath(spath)
            if sp and sp.IsValid():
                ph.make_static_collider(sp, approximation="none")
        # BEHAVIOR pick/place tables: exact static colliders so the graspable settles ON them (the
        # hardcoded tuple above does not know BEHAVIOR prim paths). Gated -> other scenes unaffected.
        _support_collider_paths = list(
            getattr(scene, "behavior_support_prims", None) or []
        )
        for _options in (
            getattr(scene, "initial_support_options", None) or {}
        ).values():
            _support_collider_paths.extend(
                str(option.get("path") or "") for option in _options
            )
        for spath in dict.fromkeys(path for path in _support_collider_paths if path):
            sp = stage.GetPrimAtPath(spath)
            if sp and sp.IsValid():
                # BEHAVIOR surfaces are often INSTANCED prims. make_static_collider walks Usd.PrimRange,
                # which does NOT descend into an instance, so it finds ZERO editable meshes and attaches NO
                # collider -> the settled graspable falls THROUGH the surface and the fallback FLOATS it at
                # the bbox-top. De-instance the support subtree first (bounded + nested-safe) so the real
                # meshes become collidable and the object PHYSICALLY rests on it. Non-instanced supports
                # (glass/cans tables) have no instances here -> unchanged.
                from pxr import Usd as _Usd

                for _ in range(4):
                    _inst = [p for p in _Usd.PrimRange(sp) if p.IsInstance()]
                    if not _inst:
                        break
                    for _p in _inst:
                        _p.SetInstanceable(False)
                ph.make_static_collider(sp, approximation="none")

        if _multi_obj:
            _build_and_park_initial_objects(stage, scene, core, rig, _multi_obj)
            for _ in range(2):
                app.update()
            ophys.get_sim().step(render=False)
            if getattr(args, "seed", None) is not None:
                refresh_task_navigation(
                    stage,
                    scene,
                    ph,
                    program,
                    render_start_xy or scene.start_xy,
                )
                prepare_authored_initial_room_door_navigation(scene)
            if getattr(scene, "initial_support_options", None):
                _prepare_initial_support_bindings(
                    stage, scene, ph, _multi_obj, rig=rig, program=program
                )
            _place_workstation_details(stage, scene)

        # P3 (free-dynamic blockers): a blocker prop that is free from INIT (the put-into-penetration
        # target obstacle). Make it a dynamic rigid with a per-tier material + a bounded de-penetration
        # velocity, settle it onto the table, and LEAVE IT DYNAMIC (skip the kinematic freeze) so PhysX
        # owns it during the render and it gets shoved when the released mug de-penetrates against it.
        # The blocker is deliberately HEAVY (scene-provided density; put_penetrate uses ~10x the mug) so
        # the mug's de-penetration slides the MUG, not the blocker -- the blocker barely moves + stays on
        # the table. Defaults to 400 (mug-equal) for any future free-dynamic-blocker scene.
        _pp_cfg = getattr(scene, "put_penetrate", None)
        blocker_density = (
            float(_pp_cfg["blocker_density"])
            if _pp_cfg and "blocker_density" in _pp_cfg
            else 400.0
        )
        for bpath in fdp_blockers:
            bp = stage.GetPrimAtPath(bpath)
            if not (bp and bp.IsValid()):
                continue
            ph.make_dynamic_rigid(
                bp,
                density=blocker_density,
                approximation=ophys.classify_collider(bp),
                tier=ophys.classify_collider(bp),
                max_depen_vel=0.6,
            )
            b_init_z = float(ophys.read_prim_xyz(stage, bpath)[2])
            b_res = ophys.settle_bodies(ph, stage, [bpath])
            b_xyz, b_mm, _b_ok = b_res[bpath]
            # ok = it did not fall THROUGH the table or fly off (same tolerant band as settle_clutter).
            import fpv_scene as _fs_b

            b_base_z = float(_fs_b.compute_world_bbox(bp).GetMin()[2])
            settle_report.append(
                (bpath.rsplit("/", 1)[-1], b_init_z, float(b_xyz[2]), float(b_mm), True)
            )
            clutter_rest_xy[bpath] = (float(b_xyz[0]), float(b_xyz[1]))
            print(
                f"[step3r] blocker {bpath.rsplit('/', 1)[-1]}: settled base z={b_base_z:.3f} "
                f"(moved {b_mm:.0f}mm) -> left DYNAMIC (PhysX-owned)"
            )

        # D3: settle the clutter (deliberately-wrong poses) TOGETHER while dynamic, run the all-pairs
        # separation gate (needs the bodies still dynamic), THEN freeze each kinematic BEFORE the object
        # settle (else the graspable's settle would re-simulate the clutter).
        if scene_clutter:
            cpaths = [it[0] for it in scene_clutter]
            init_z = {p: float(ophys.read_prim_xyz(stage, p)[2]) for p in cpaths}
            clutter_traj = (
                []
            )  # FIX C: recorded per-step (translate,orient) of every clutter prim
            cres = ophys.settle_clutter(ph, stage, scene_clutter, traj_out=clutter_traj)
            pen_pairs = ophys.all_pairs_separation(ph, stage, cpaths)
            for cpath, (xyz, mm, cok) in cres.items():
                settle_report.append(
                    (
                        cpath.rsplit("/", 1)[-1],
                        init_z[cpath],
                        float(xyz[2]),
                        float(mm),
                        bool(cok),
                    )
                )
                clutter_rest_xy[cpath] = (float(xyz[0]), float(xyz[1]))
                ph.set_rigid_kinematic(stage.GetPrimAtPath(cpath), True)

        # SSOT: build the collision oracle now that ALL colliders (furniture from init + the
        # de-instanced support tables above) are applied, so the settle's support_top_z and the
        # post-action validator read ONE precise map. Gated -> None (and byte-identical) otherwise.
        _ensure_cworld(scene, stage, ph)

        # D1/D3: settle the carried/graspable object onto its support, then freeze kinematic + re-ground.
        if has_obj and not _multi_obj:
            gkey = next(
                e.name for e in scene.entities.values() if e.role is Role.GRASPABLE
            )
            ge = scene.resolve(gkey)
            obj_xy = ge.xy
            support_top_z = _support_top_for(
                scene, obj_xy, ge, "/World/Obj", declared=True
            )
            _obj_prim = stage.GetPrimAtPath("/World/Obj")
            _obj_approx = ophys.classify_collider(_obj_prim)
            # When /World/Obj is a free-dynamic prim (put_penetrate: it is released mid-render), give it a
            # physics material + a bounded de-penetration velocity so on release it SLIDES off the blocker
            # (not launches). fdp_obj False -> tier/max_depen_vel default None -> byte-identical.
            ph.make_dynamic_rigid(
                _obj_prim,
                density=400.0,
                approximation=_obj_approx,
                tier=_obj_approx if fdp_obj else None,
                max_depen_vel=0.6 if fdp_obj else None,
            )
            declared_z = float(ge.z)
            _dynamic_settle = gkey in _dynamic_settle_keys
            _declared_support_paths = _support_paths(scene)
            # BEHAVIOR objects may be CENTER-pivot (e.g. apple/agveuv), so the base-pivot assumption in
            # core.local_obj_min_z (returns 0 for usd_ref) would seat the ORIGIN -- not the base -- on the
            # support and the object floats by ~its lower half. Use the object's REAL local bbox bottom so
            # the settle seats the actual base. (Isaac mugs are base-pivot -> local_bbox_min_z ~= 0 ->
            # unchanged; gated on behavior_grab so existing scenes stay byte-identical.)
            import fpv_scene as _fs_settle

            _bmz = (
                (lambda p: _fs_settle.local_bbox_min_z(p))
                if getattr(scene, "behavior_grab", None)
                else None
            )
            settled = ophys.settle_object_on_support(
                ph,
                stage,
                rig,
                core,
                "/World/Obj",
                float(support_top_z),
                obj_xy,
                local_min_z_fn=_bmz,
                # P0-B: BEHAVIOR objects keep the REAL physics rest -- accept_below>0 returns the settled
                # pose on OG's convexHull collider even when it lands below the world-bbox top (rim/backsplash),
                # instead of the analytic bbox-top seat that FLOATS it. Isaac cups (behavior_grab None) keep 0.0.
                accept_below=(0.20 if getattr(scene, "behavior_grab", None) else 0.0),
                retry_xy=_support_retry_xy(
                    stage,
                    scene,
                    obj_xy,
                    object_half_xy=_support_object_half_xy(_obj_prim),
                ),
                support_top_fn=lambda final_xy: _support_top_for(
                    scene, final_xy, ge, "/World/Obj"
                ),
                support_probe_fn=lambda candidate_xy, half_xy, object_height: probe_declared_support(
                    ph,
                    candidate_xy,
                    float(support_top_z),
                    half_xy,
                    object_height,
                    exclude_path="/World/Obj",
                    support_paths=_declared_support_paths,
                    min_support_fraction=KINEMATIC_SUPPORT_FRACTION,
                ),
                kinematic_seat=not _dynamic_settle,
                require_upright=bool((ge.meta or {}).get("require_upright")),
            )
            if not settled.ok:
                raise ophys.InitialSettleError(gkey, "/World/Obj", settled)
            _calibrate_initial_support(scene, gkey, settled)
            rested_z = float(settled.position[2])
            obj_xy = (float(settled.position[0]), float(settled.position[1]))
            ph.set_rigid_kinematic(stage.GetPrimAtPath("/World/Obj"), True)
            scene.set_object_rest_pose(
                gkey, ophys.read_prim_transform(stage, "/World/Obj")
            )
            settle_report.append(
                (
                    gkey,
                    declared_z,
                    float(rested_z),
                    abs(rested_z - declared_z) * 1000.0,
                    True,
                )
            )
            print(
                f"[step3r] settle {gkey}: declared z={declared_z:.3f} -> rested z={rested_z:.3f} "
                f"(moved {abs(rested_z - declared_z) * 1000:.0f}mm) "
                f"attempts={settled.attempts} ok=True"
            )
        # MULTI-graspable: build + settle EACH object at its OWN prim path so one program can pick several
        # different-shaped objects. Each spec = {key, path, def|build, color?}; xy/support come from the
        # registered entity. At render each object rides rig._obj_path, swapped per frame via Frame.obj_path.
        if _multi_obj:
            import fpv_scene as _fs_mo

            core.enable_physics_refetch(
                rig, stage, _multi_obj[0]["path"], "/World/Body"
            )
            for spec in _multi_obj:
                if "def" in spec:
                    _b, _o = core.load_object_def(spec["def"])
                else:
                    _b, _o = spec["build"], None
                core.set_object(rig, _b, _o)
                ge = scene.resolve(spec["key"])
                _refresh_movable_initial_support_pose(stage, scene, spec)
                obj_xy = ge.xy
                initial_place = str(spec.get("initial_place") or "")
                support_path = spec.get("support_path")
                support_top_z = (
                    float(spec["support_z"])
                    if spec.get("support_z") is not None
                    else _support_top_for(
                        scene,
                        obj_xy,
                        ge,
                        spec["path"],
                        declared=True,
                        initial_place=initial_place,
                        support_path=support_path,
                    )
                )
                op = stage.GetPrimAtPath(spec["path"])
                orientation = tuple(
                    float(value)
                    for value in spec.get("orientation", (1.0, 0.0, 0.0, 0.0))
                )
                approximation = ophys.classify_collider(op)
                is_free_dynamic = str(spec["path"]).rstrip("/") in fdp_object_paths
                ph.make_dynamic_rigid(
                    op,
                    density=400.0,
                    approximation=approximation,
                    tier=approximation if is_free_dynamic else None,
                    max_depen_vel=0.6 if is_free_dynamic else None,
                )
                declared_z = float(ge.z)
                if spec.get("prebuilt"):
                    # Fixed-container preloads were already packed against measured interior
                    # shelves. Re-seating them as ordinary ON-support objects can lift them onto
                    # a rim and destroy their verified initial INSIDE relation.
                    ph.set_rigid_kinematic(op, True)
                    scene.set_object_rest_pose(
                        spec["key"], ophys.read_prim_transform(stage, spec["path"])
                    )
                    settle_report.append(
                        (spec["key"], declared_z, declared_z, 0.0, True)
                    )
                    print(
                        f"[step3r] retain prebuilt {spec['key']} @ {spec['path']}: "
                        f"interior pose z={declared_z:.3f}"
                    )
                    continue
                # A BEHAVIOR graspable (OG collisions scope) rests ON the surface via REAL physics -- never
                # float, never fall to the floor: accept_below>0 keeps the real settled pose (on OG's precise
                # per-link convexHull collider) instead of the analytic bbox-top seat that FLOATED it, and if
                # the placed XY overhangs the real solid the object is stepped toward the support's collision
                # centroid (reach-bounded) and re-settled. Isaac-native multi-object scenes use their
                # existing 0.06 m below-top tolerance, but a failed physical rest is never fabricated.
                _is_beh = ph._has_collisions_scope(op)
                _dynamic_settle = str(spec["key"]) in _dynamic_settle_keys
                _declared_support_paths = _support_paths(
                    scene,
                    support_path=support_path,
                    initial_place=initial_place,
                )
                settled = ophys.settle_object_on_support(
                    ph,
                    stage,
                    rig,
                    core,
                    spec["path"],
                    float(support_top_z),
                    obj_xy,
                    local_min_z_fn=lambda p: _fs_mo.local_bbox_min_z(p),
                    accept_below=(0.20 if _is_beh else 0.06),
                    orientation=orientation,
                    retry_xy=_support_retry_xy(
                        stage,
                        scene,
                        obj_xy,
                        initial_place=initial_place,
                        support_path=support_path,
                        object_half_xy=_support_object_half_xy(op),
                    ),
                    support_top_fn=lambda final_xy, _ge=ge, _path=spec[
                        "path"
                    ], _place=initial_place, _support=support_path: (
                        _support_top_for(
                            scene,
                            final_xy,
                            _ge,
                            _path,
                            declared=True,
                            initial_place=_place,
                            support_path=_support,
                        )
                    ),
                    support_probe_fn=lambda candidate_xy, half_xy, object_height, _hint=float(
                        support_top_z
                    ), _path=str(
                        spec["path"]
                    ), _support_paths=_declared_support_paths: probe_declared_support(
                        ph,
                        candidate_xy,
                        _hint,
                        half_xy,
                        object_height,
                        exclude_path=_path,
                        support_paths=_support_paths,
                        min_support_fraction=KINEMATIC_SUPPORT_FRACTION,
                    ),
                    kinematic_seat=not _dynamic_settle,
                    require_upright=bool(
                        spec.get("require_upright")
                        or (ge.meta or {}).get("require_upright")
                    ),
                )
                if not settled.ok:
                    raise ophys.InitialSettleError(
                        str(spec["key"]), str(spec["path"]), settled
                    )
                _calibrate_initial_support(scene, str(spec["key"]), settled)
                rested_z = float(settled.position[2])
                settled_xy = (
                    float(settled.position[0]),
                    float(settled.position[1]),
                )
                if settled_xy != (float(obj_xy[0]), float(obj_xy[1])):
                    print(
                        f"[step3r] settle {spec['key']}: placed XY overhung the solid -> stepped "
                        f"{math.hypot(settled_xy[0] - obj_xy[0], settled_xy[1] - obj_xy[1]) * 1000:.0f}mm "
                        f"toward surface centre -> rests ON surface z={rested_z:.3f}"
                    )
                ph.set_rigid_kinematic(op, True)
                scene.set_object_rest_pose(
                    spec["key"], ophys.read_prim_transform(stage, spec["path"])
                )
                settle_report.append(
                    (
                        spec["key"],
                        declared_z,
                        float(rested_z),
                        abs(rested_z - declared_z) * 1000.0,
                        True,
                    )
                )
                print(
                    f"[step3r] settle {spec['key']} @ {spec['path']}: declared z={declared_z:.3f} "
                    f"-> rested z={rested_z:.3f} attempts={settled.attempts} ok=True"
                )
        # RESTORE the render-loop timestep. Default (fdp None) -> FPS(12) substeps=1, byte-identical to
        # pre-change (the drawer joint drive integration is unchanged). Free-dynamic scenes step at
        # FPS*5 (>=60Hz + determinism flags) so a released object de-penetrates + settles smoothly on
        # camera (proven by --physcheck). The free-object registry (SINK 2, read-only) is built here.
        if fdp:
            ph.set_scene_timestep(stage, FPS, substeps=5)
            reg = {p: stage.GetPrimAtPath(p) for p in fdp_blockers}
        else:
            ph.set_scene_timestep(stage, FPS)

    _initial_attachment_rows = list(
        (getattr(scene, "behavior_task", None) or {}).get("initial_attachments") or []
    )
    initial_attachment_status = se.apply_initial_attachments(
        scene, _initial_attachment_rows
    )
    if initial_attachment_status:
        from isaac_goal import Pred as _Pred_initial

        expected_initial = {
            (_Pred_initial.ATTACHED, row["child"], row["parent"])
            for row in _initial_attachment_rows
        }
        actual_initial = scene._relations_attached_live()
        if not expected_initial.issubset(actual_initial):
            raise RuntimeError(
                "initial attachment live predicate mismatch: "
                f"missing={sorted(expected_initial - actual_initial, key=str)} "
                f"status={initial_attachment_status}"
            )
        print(
            f"[step3r] initial ATTACHED: {len(initial_attachment_status)} "
            f"physically verified mate(s) {initial_attachment_status}"
        )

    # Build revolute-door controllers (de-instance + leaf rotate op + resolve swing sign from the nav-grid
    # wall clearance) AFTER the grid exists and BEFORE execute (the _swing producer reads the resolved
    # Entity.swing_sign; the execute() door pre-pass reads scene.doors). Merged into `controllers` for the
    # per-frame drive/nudge. Gated on scene.doors (None for every non-door scene -> unchanged).
    if getattr(scene, "doors", None):
        for _dname in list(scene.doors.keys()):
            _dc = _place_door(stage, scene, _dname)
            if _dc is not None:
                controllers[_dname] = _dc
        _ndoors = sum(1 for c in controllers.values() if c.get("kind") == "revolute")
        print(f"[step3r] doors: {_ndoors} swing controllers wired")
    # Openable CONTAINERS (fridge/cabinet with a revolute leaf) are normal entities carrying hinge fields
    # (NOT nav doors -> they stay solid obstacles). Build the SAME swing controller so [Open]/[Close] drive
    # the leaf. Keyed by hinge_xy set + a prim_path + not already wired as a door. Gated -> non-openable
    # scenes are unchanged.
    _n_openc = 0
    for _e in list(scene.entities.values()):
        if (
            getattr(_e, "hinge_xy", None) is not None
            and getattr(_e, "prim_path", None)
            and _e.name not in controllers
            and _e.name not in (getattr(scene, "doors", None) or {})
        ):
            _oc = _place_door(stage, scene, _e.name)
            if _oc is not None:
                controllers[_e.name] = _oc
                _n_openc += 1
    if _n_openc:
        print(f"[step3r] openable containers: {_n_openc} swing controllers wired")
    _n_slides = 0
    for _entity in list(scene.entities.values()):
        if (
            (_entity.meta or {}).get("openable_kind") == "slide"
            and _entity.prim_path
            and _entity.name not in controllers
        ):
            _slide_controller = _place_slide_openable(stage, scene, _entity.name)
            if _slide_controller is not None:
                controllers[_entity.name] = _slide_controller
                _n_slides += 1
    if _n_slides:
        logger.info("[step3r] openable supports: %d slide controllers wired", _n_slides)

    # EGO LEGS. A program that guides a body it cannot let see the room ([Forward]/[Turn*]) has its legs
    # RECOMPILED here, from the nav grid the init gate just measured, because an author dead-reckoning
    # against an imagined layout produces legs the built scene refuses (measured: 0 of 5 walked, blocked
    # at 0.00 m clear). Must run AFTER the grid exists and BEFORE compile_program. Returns the program
    # byte-identical for every program with no ego verb, so no other family is touched.
    import ego_route
    import init_validation as _iv_ego

    def _ego_stand(key, cur):
        """The stand the goal loop below WILL pick for ``key`` from ``cur`` -- same picker, same
        inputs, INCLUDING the route gate, so the compiler's model of where the body ends up is a fixed
        point rather than a guess. None (unregistered / no clean stand) falls back to the declared
        stand."""
        if key not in scene.entities:
            return None
        try:
            _entity = scene.resolve(key)
            _reach_tol = (
                se._OPENABLE_REACH_TOL_M
                if getattr(_entity, "handle_xy", None) is not None
                and (
                    getattr(_entity, "hinge_xy", None) is not None
                    or getattr(_entity, "axis", None) is not None
                )
                else None
            )
            return _iv_ego.pick_stand_runtime(
                ph,
                stage,
                rig,
                scene,
                key,
                cur,
                route_gate=_ego_route_gate(scene),
                reach_tol=_reach_tol,
            )
        except Exception as _exc:  # noqa: BLE001 -- prediction is best-effort; the fallback is sound
            print(f"[ego] stand prediction for <{key}> unavailable ({_exc})")
            return None

    program = ego_route.recompile(
        program,
        scene,
        render_start_xy if render_start_xy is not None else scene.start_xy,
        stand_fn=_ego_stand if getattr(scene, "nav_grid", None) is not None else None,
    )
    # Dual-role targets can only be measured after their graspable prim exists. Deferred preloads
    # are realized before the shared collision oracle is built, so later physics sees them too.
    import behavior_import as _bimp_aff

    _bimp_aff.attach_graspable_target_affordances(scene, stage)
    _bimp_aff.realize_deferred_preloads(scene, stage)
    _bimp_aff.validate_aggregate_target_capacity(scene, stage)
    _declare_movable_initial_support_relations(scene, _multi_obj)
    scene.bind_openable_controllers(controllers)
    initially_open = apply_authored_initial_openables(scene, controllers)
    if initially_open:
        print(
            "[step3r] authored initially open: " + ", ".join(initially_open)
        )
    _validate_initial_object_relations(scene)
    scene.capture_episode_baseline()
    # Change 1: navigates are compiled UNRESOLVED (target_key only); the stand pose is picked at RUNTIME in
    # the goal loop below (init_validation.pick_stand_runtime, nearest-reachable from the human's live
    # position). The VALIDATED start_xy is still threaded into make_state via env.reset(start_xy=...).
    goals, compiled_commands = command_grammar.compile_program_with_commands(
        program, scene
    )
    # P2: drive the program through the ONLINE env (SemanticEnv) one command at a time instead of the batch
    # se.execute(all goals). reset() builds the persistent SchedState (== execute's make_state) and state
    # carries across run_goal() calls; frames still accumulate + render batched here (per-command render +
    # FPV obs = P5). The DOOR pre-pass stays a pre-loop compile step until P3 moves door handling into walk.
    # Byte-identical to se.execute (same step_one, producers, order, per-goal logging).
    import semantic_env

    env = semantic_env.SemanticEnv(rig, g, scene)
    env.reset(start_xy=render_start_xy)
    if getattr(args, "seed", None) is not None:
        reachability = program_reachability_preflight(
            ph,
            stage,
            rig,
            scene,
            goals,
            compiled_commands,
            render_start_xy or scene.start_xy,
            env=env,
            gait_profile=g,
        )
        scene.build_reachability = reachability
        if reachability["ok"]:
            print(
                f"[step3r][build-reach] exact runtime contract PASS: "
                f"{reachability['checked']} navigation/action stand check(s)"
            )
        else:
            render_episode_valid = False
            print(
                f"[step3r][build-reach] exact runtime contract FAIL during "
                f"{reachability['phase']} at goal "
                f"{reachability['goal_index']} <{reachability['target_key']}> from "
                f"({reachability['from_xy'][0]:.2f},{reachability['from_xy'][1]:.2f}); "
                "episode marked invalid before execution"
            )
        env.reset(start_xy=render_start_xy)
    if getattr(scene, "natural_arm", False):
        print(
            "[exec] natural_arm ON: reach-exact elbow-down swivel enabled in warp_arm_to_seat"
        )
    # Change 2: the compile-time door pre-pass (door_planner.insert_door_actions) is REMOVED. It needed the
    # navigate's target_xy to detect a door crossing, but Change 1 makes navigates UNRESOLVED (target_xy=None)
    # here -> it would skip them all. Door handling now runs at RUNTIME in the goal loop below, CLOSED-first,
    # keyed off each navigate's runtime-resolved stand (_runtime_open_doors).
    # P5 (per-command physics): build the ONE oracle up front + a LAZY physics-clock bracket (started on the
    # first placement). After each command that placed objects, settle them RIGHT AWAY so the NEXT command's
    # producer reads their REAL settled poses -- physics then stacks/spreads multiple objects into one target
    # correctly (vs the old end-of-episode batch that released everyone at once and let co-located drops
    # collide mid-fall). Gated exactly like the post-action pass; offline/no-cworld scenes skip it (unchanged).
    _ensure_cworld(scene, stage, ph)
    _percmd_settle = (
        getattr(scene, "post_action_check", False)
        and getattr(scene, "cworld", None) is not None
    )
    _settle_started = [False]

    def _ensure_settle_clock() -> None:
        if _settle_started[0]:
            return
        ph.set_scene_timestep(
            stage, FPS, substeps=5
        )  # >=60Hz: the settle drop cannot tunnel a shelf
        fpv_core.start_timeline(rcfg.playback_fps)
        for _ in range(
            4
        ):  # WARMUP: the first steps after play() do not integrate physics
            fpv_render.advance_once(rcfg)
        _settle_started[0] = True

    def _settle_step() -> None:
        fpv_render.advance_once(rcfg)

    # Option A (multi-object render fix): snapshot each graspable's SPAWN world pose NOW, before the
    # per-command settle below moves the prims to their final resting pose. The render loop animates only
    # the ONE active object per frame (rig._obj_path); idle objects keep whatever pose is left in the
    # stage. We restore these spawn poses just before the render loop, so a not-yet-grabbed object sits at
    # its start (not "already in the container") and each object moves only during its own grab/put frames.
    import object_physics as _ophys_spawn
    from pxr import Usd as _Usd_s, UsdGeom as _UsdGeom_s

    _obj_spawn: dict = {}
    _xc_spawn = _UsdGeom_s.XformCache(_Usd_s.TimeCode.Default())
    _spawn_paths = {
        str(entity.prim_path)
        for entity in scene.entities.values()
        if entity.role is Role.GRASPABLE and entity.prim_path
    }
    for _sp in sorted(_spawn_paths):
        _sprim = stage.GetPrimAtPath(_sp) if _sp else None
        if _sprim is not None and _sprim.IsValid():
            _obj_spawn[_sp] = _ophys_spawn._read_pose(_xc_spawn, _sprim)

    # Feature 1 (dual-role): a graspable that is ALSO a put/putinside target (a bowl/tupperware) gets its
    # container/surface affordance attached NOW that its rig prim exists (built above) -- deferred from
    # build_behavior_task, where the prim did not exist yet. No-op for scenes with no graspable targets.
    import init_validation as iv

    # Bookcase board letter labels: stamp each built board with its part-letter placard so the FPV/TPV read
    # which board is which. Gated on scene.board_labels -> no-op + byte-identical for every non-bookcase scene.
    _bimp_aff.apply_board_labels(scene, stage, viewer_xy=env.st.root_xy)
    # Flat colour on the bodies a spec asked to be coloured. The library holds one model per category
    # -- there is no `red_box`, only `box` -- so "the red bin, not the green one" is unanswerable
    # until the paint goes on. Gated on scene.obj_tints -> no-op for every scene that authors none.
    _bimp_aff.apply_object_tints(scene, stage)

    # EXISTENTIAL witness re-pick (faithful, general): an `exists` inside/ontop goal ("food inside SOME
    # tupperware") is satisfied by ANY instance of the type -- the compile-time witness is arbitrary (BDDL dict
    # order), so a food whose default container is boxed-in never gets placed. Here, BEFORE running the program,
    # re-route each such food to a REACHABLE candidate: keep the default if it is reachable, else the first
    # candidate that has a reachable stand. Only EXISTENTIAL subjects (scene.behavior_task['existential_witness'])
    # are re-picked -- a fully-named goal is never touched. Gated on nav_grid (hand-tuned demos unaffected).
    _exw = (getattr(scene, "behavior_task", None) or {}).get(
        "existential_witness"
    ) or {}
    if _exw and getattr(scene, "nav_grid", None) is not None:
        _cur0 = tuple(scene.start_xy)
        _reach_cache: dict = {}

        def _witness_reachable(_key):
            if _key not in _reach_cache:
                _reach_cache[_key] = (
                    iv.pick_stand_runtime(ph, stage, rig, scene, _key, _cur0)
                    is not None
                )
            return _reach_cache[_key]

        for _fi, _fg in enumerate(goals):
            if _fg.prim is not command_grammar.Primitive.PICK:
                continue
            _cands = _exw.get(getattr(_fg, "obj_key", None))
            if not _cands:
                continue
            # Only REGISTERED candidates are reachability-testable (pick_stand_runtime->resolve KeyErrors on an
            # unregistered name -- a candidate whose category was skipped / USD unbaked is never composed).
            _cands = [c for c in _cands if c in scene.entities]
            if not _cands:
                continue
            # this food's put = the first PLACE/PLACE_INSIDE after its PICK; its container = meta['surface'].
            _place_g = _w = None
            for _g in goals[_fi + 1 :]:
                if _g.prim in (
                    command_grammar.Primitive.PLACE,
                    command_grammar.Primitive.PLACE_INSIDE,
                ):
                    _place_g, _w = _g, (_g.meta or {}).get("surface")
                    break
            if _w is None or _w not in scene.entities or _witness_reachable(_w):
                continue  # no put found, or the default witness is reachable -> keep it (byte-identical)
            _new = next((c for c in _cands if c != _w and _witness_reachable(c)), None)
            if _new is None:
                continue  # no reachable candidate -> the command honest-fails downstream (no fake)
            _place_g.meta["surface"] = _new
            for _g in goals[
                _fi + 1 :
            ]:  # its walk-to-container = first NAVIGATE_TO(target_key==old)
                if (
                    _g.prim is command_grammar.Primitive.NAVIGATE_TO
                    and _g.target_key == _w
                ):
                    _g.target_key = _new
                    break
            print(
                f"[existential] {_fg.obj_key}: witness {_w} unreachable -> re-route to reachable {_new}"
            )

    fdp_initial_state = _snapshot_dynamic_prims(stage, fdp_paths)

    # END of setup: hand the built + settled + validated episode to the caller. run_render unpacks this
    # into the same local names and continues; IsaacEnv reuses it for reset-many episodes.
    return {
        "app": app,
        "stage": stage,
        "ph": ph,
        "rig": rig,
        "g": g,
        "scene": scene,
        "scene_id": str(args.scene),
        "controllers": controllers,
        "env": env,
        "goals": goals,
        "compiled_commands": compiled_commands,
        "program": program,
        "render_start_xy": render_start_xy,
        "render_episode_valid": render_episode_valid,
        "has_obj": has_obj,
        "fdp": fdp,
        "fdp_paths": fdp_paths,
        "fdp_obj": fdp_obj,
        "fdp_initial_state": fdp_initial_state,
        "reg": reg,
        "scene_clutter": scene_clutter,
        "settle_report": settle_report,
        "initial_attachment_status": initial_attachment_status,
        "pen_pairs": pen_pairs,
        "clutter_rest_xy": clutter_rest_xy,
        "clutter_traj": clutter_traj,
        "obj_spawn": _obj_spawn,
        "W": W,
        "Hh": Hh,
        "app_cfg": app_cfg,
        "rcfg": rcfg,
        "tpv_look": tpv_look,
        "tpv_eye": tpv_eye,
        "percmd_settle": _percmd_settle,
        "ensure_settle_clock": _ensure_settle_clock,
        "settle_step": _settle_step,
        "settle_started": _settle_started,
    }


def make_start_frame(ctx: dict):
    """Build the neutral standing frame used at an episode boundary by every renderer."""
    st = getattr(ctx.get("env"), "st", None)
    rig = ctx.get("rig")
    if st is None or rig is None:
        return None
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)
    pose = se._manip(rig, arm_rest, 0.0, 0.0)
    return wpp.Frame(
        pose,
        st.root_xy,
        st.root_yaw,
        st.stand_z,
        fpv_mode="look",
        obj=("hide", None),
        seg="stand",
    )


def author_fpv_calibration_body(ctx: dict, frame):
    """Author the same neutral body geometry before either renderer meters exposure."""
    vertices, joints, _object_transform = wpp.realize(ctx["rig"], frame)
    if getattr(ctx["rig"], "_body_mesh", None) is None:
        core.author_body_once(ctx["stage"], ctx["rig"], vertices)
    else:
        core.update_body(ctx["rig"], vertices)
    ctx["body_authored"] = True
    return joints


def _start_head_bootstrap_pose(ctx: dict, frame, joints):
    """Return the real first-observation camera pose from the shared FPV policy."""
    import egocentric_camera
    import egocentric_camera_isaac

    scene = ctx["scene"]
    try:
        solid_boxes = scene.solid_boxes()
    except Exception:  # noqa: BLE001 -- the head pose itself does not need visibility boxes
        solid_boxes = ()
    adapter = egocentric_camera_isaac.IsaacCameraAdapter.from_scene(
        scene, solid_boxes
    )
    camera_frame = adapter.camera_frame(
        frame,
        joints,
        None,
        dt_seconds=1.0 / FPS,
    )
    output = egocentric_camera.step_camera(
        camera_frame,
        framing=adapter.framing,
    ).output
    eye = tuple(float(value) for value in output.eye)
    target = tuple(
        float(value)
        for value in np.asarray(output.eye, dtype=np.float64)
        + np.asarray(output.look, dtype=np.float64)
    )
    return eye, target


def _nav_safe_bootstrap_pose(scene, start_xy):
    """Return a low near-overhead view rooted in the validated start's nav cell."""
    sx, sy = (float(start_xy[0]), float(start_xy[1]))
    floor_z = float(getattr(scene, "floor_z", 0.0))
    grid = getattr(scene, "nav_grid", None)
    if grid is None:
        grid = getattr(scene, "wall_sdf_grid", None)
    offsets = (
        (0.2, 0.2),
        (0.2, -0.2),
        (-0.2, 0.2),
        (-0.2, -0.2),
        (0.2, 0.0),
        (-0.2, 0.0),
        (0.0, 0.2),
        (0.0, -0.2),
    )
    selected = offsets[0]
    if grid is not None:
        for dx, dy in offsets:
            try:
                ci, cj = grid.world_to_cell(sx + dx, sy + dy)
                if grid.is_free(ci, cj):
                    selected = (dx, dy)
                    break
            except (AttributeError, IndexError, TypeError, ValueError):
                break
    eye = (sx + selected[0], sy + selected[1], floor_z + 2.2)
    target = (sx, sy, floor_z + 1.0)
    return eye, target


def _bootstrap_camera_candidates(ctx: dict, start_frame, start_joints) -> list[dict]:
    """Build bounded, scene-agnostic poses for renderer initialization."""
    candidates: list[dict] = []
    try:
        eye, look = _start_head_bootstrap_pose(ctx, start_frame, start_joints)
        candidates.append({"candidate": "start_head_fpv", "eye": eye, "look": look})
    except Exception as exc:  # noqa: BLE001 -- later safe candidates remain usable
        logger.warning("[fpv-bootstrap] start-head pose unavailable: %s", exc)

    eye, look = _nav_safe_bootstrap_pose(ctx["scene"], start_frame.xy)
    candidates.append({"candidate": "start_nav_overhead", "eye": eye, "look": look})

    legacy_eye, legacy_look = ctx.get("tpv_eye"), ctx.get("tpv_look")
    if legacy_eye is not None and legacy_look is not None:
        candidates.append(
            {
                "candidate": "legacy_tpv_overview",
                "eye": tuple(float(value) for value in legacy_eye),
                "look": tuple(float(value) for value in legacy_look),
            }
        )

    unique: list[dict] = []
    seen = set()
    for candidate in candidates:
        identity = tuple(round(value, 6) for value in candidate["eye"] + candidate["look"])
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(candidate)
    return unique


def _settle_fpv_bootstrap(
    app,
    timeline,
    annot_fpv,
    annot_tpv,
    app_cfg,
    cop,
    tcop,
    candidates,
    fpv_render_module,
    camera_module,
) -> dict:
    """Try safe bootstrap poses in one process and accept only a non-black FPV."""
    try:
        budget_s = float(os.getenv("FPV_BOOTSTRAP_CANDIDATE_BUDGET_S", "20"))
    except ValueError as exc:
        raise ValueError("FPV_BOOTSTRAP_CANDIDATE_BUDGET_S must be numeric") from exc
    if budget_s <= 0.0:
        raise ValueError("FPV_BOOTSTRAP_CANDIDATE_BUDGET_S must be positive")
    failures = []
    for attempt, candidate in enumerate(candidates, start=1):
        context = {
            "attempt": attempt,
            "candidate": str(candidate["candidate"]),
            "eye": tuple(float(value) for value in candidate["eye"]),
            "look": tuple(float(value) for value in candidate["look"]),
        }
        logger.info(
            "[fpv-bootstrap] attempt=%d/%d candidate=%s eye=%s look=%s budget=%.1fs",
            attempt,
            len(candidates),
            context["candidate"],
            context["eye"],
            context["look"],
            budget_s,
        )
        camera_module.drive_tpv(cop, context["eye"], context["look"])
        camera_module.drive_tpv(tcop, context["eye"], context["look"])
        try:
            diagnostics = fpv_render_module.settle_and_validate(
                app,
                timeline,
                annot_fpv,
                app_cfg,
                annot_tpv=annot_tpv,
                warmup_frames=(
                    int(getattr(app_cfg, "warmup_frames", 0)) if attempt == 1 else 0
                ),
                warm_budget_s=budget_s,
                context=context,
            )
        except fpv_render_module.BlackOrEmptyFrameError as exc:
            failures.append(dict(exc.diagnostics))
            logger.warning(
                "[fpv-bootstrap] rejected attempt=%d candidate=%s diagnostics=%s",
                attempt,
                context["candidate"],
                exc.diagnostics,
            )
            continue
        selected = {**context, "diagnostics": dict(diagnostics)}
        logger.info("[fpv-bootstrap] selected %s", selected)
        return selected
    raise fpv_render_module.BlackOrEmptyFrameError(
        "BLACK/EMPTY FRAME after every bootstrap camera candidate; "
        f"attempts={failures}",
        diagnostics={"attempts": failures},
    )


def make_fpv_products(ctx: dict) -> dict:
    """RL-rollout counterpart of run_render's FPV/TPV camera+products setup (see run_render ~4995-5303):
    given an already-open episode ``ctx`` (from setup_episode), create the FPV (+TPV) camera, the rgb
    render products/annotators, and the timeline, PRIME the readback once (settle_and_validate), and stash
    them in ctx so ``IsaacEnv.render_fpv()`` can render ONE egocentric frame per observation. ADDITIVE and
    only for the RL path (the sim-server process); run_render/--step3 create their OWN identically, so the
    benchmark render is byte-untouched. Post-launch imports (need the SimulationApp booted), like run_render.
    """
    import render_runtime
    import fpv_core
    import fpv_render
    import smplh_camera

    stage, app = ctx["stage"], ctx["app"]
    W, Hh, rcfg, app_cfg = ctx["W"], ctx["Hh"], ctx["rcfg"], ctx["app_cfg"]
    render_settings = configure_fpv_render_settings()
    # Scene switches can introduce uncached shaders and materials even in a warm process. Use the same
    # full convergence budget as offline rendering; brightness alone cannot detect a bright gray shader
    # placeholder that has not finished compiling.
    try:
        _warm = max(int(getattr(app_cfg, "warmup_frames", 0)), 400)
    except (TypeError, ValueError):
        _warm = 400
    cop = smplh_camera.make_fpv_camera(stage, "/World/FpvCam", W, Hh)
    tcop = smplh_camera.make_tpv_camera(stage, "/World/TpvCam", W, Hh)
    fpv_core.add_lighting(stage, app_cfg)  # must precede setup_products
    render_runtime._boost_lights(stage)
    rp_f, rp_t, a_f, a_t = fpv_render.setup_products(
        "/World/FpvCam", "/World/TpvCam", rcfg, W, Hh
    )
    tl = fpv_core.start_timeline(rcfg.playback_fps)
    start_frame = make_start_frame(ctx)
    if start_frame is None:
        raise RuntimeError(
            "cannot configure FPV lighting without an episode start frame"
        )
    start_joints = author_fpv_calibration_body(ctx, start_frame)
    app_cfg.warmup_frames = _warm
    bootstrap_camera = _settle_fpv_bootstrap(
        app,
        tl,
        a_f,
        a_t,
        app_cfg,
        cop,
        tcop,
        _bootstrap_camera_candidates(ctx, start_frame, start_joints),
        fpv_render,
        smplh_camera,
    )
    autoexp = configure_fpv_lighting(
        app,
        stage,
        rcfg,
        cop,
        a_f,
        ctx["scene"],
        eye_z=float(start_joints[wpp.H.HEAD_J][2]),
        scene_id=str(ctx.get("scene_id") or "") or None,
    )
    ctx.update(
        {
            "cop": cop,
            "tcop": tcop,
            "a_f": a_f,
            "a_t": a_t,
            "rp_f": rp_f,
            "rp_t": rp_t,
            "timeline": tl,
            "autoexp": autoexp,
            "render_settings": render_settings,
            "bootstrap_camera": bootstrap_camera,
        }
    )
    return ctx


def run_render(args) -> int:
    """One-shot render: set up the episode scene, run the program, render FPV+TPV, encode + close.

    The setup (boot/build/settle/validate/compile) now lives in ``setup_episode``; everything below
    (the goal loop + post-action compliance + goal-check + render + encode) is UNCHANGED -- it just
    unpacks the setup context first, so this path is byte-identical to before the extraction.
    """
    import time as _time

    _t_start = _time.perf_counter()
    _c = setup_episode(args)
    if not isinstance(_c, dict):
        return _c  # preview / --check-family / seed-abort early exit (an int status)
    import action_check
    import render_runtime
    import fpv_core
    import fpv_render
    import init_validation as iv
    import smplh_camera
    from command_frame_spans import CommandFrameSpanTracker
    from fpv_segments import (
        materialize_compatibility_sequence,
        resolve_capture_plan,
        run_segment,
        SegmentCtx,
    )

    app, stage, ph, rig, scene = (
        _c["app"],
        _c["stage"],
        _c["ph"],
        _c["rig"],
        _c["scene"],
    )
    controllers, env, goals = _c["controllers"], _c["env"], _c["goals"]
    compiled_commands = _c["compiled_commands"]
    render_episode_valid, has_obj = _c["render_episode_valid"], _c["has_obj"]
    fdp, fdp_paths, reg = _c["fdp"], _c["fdp_paths"], _c["reg"]
    fdp_initial_state = _c["fdp_initial_state"]
    settle_report, pen_pairs = _c["settle_report"], _c["pen_pairs"]
    clutter_rest_xy = _c["clutter_rest_xy"]
    _obj_spawn = _c["obj_spawn"]
    W, Hh, app_cfg, rcfg = _c["W"], _c["Hh"], _c["app_cfg"], _c["rcfg"]
    tpv_look, tpv_eye = _c["tpv_look"], _c["tpv_eye"]
    _percmd_settle = _c["percmd_settle"]
    _ensure_settle_clock, _settle_step = _c["ensure_settle_clock"], _c["settle_step"]
    _settle_started = _c["settle_started"]

    trajectory_artifact = None
    execution_goals = goals
    command_spans = []
    if getattr(args, "trajectory_in", ""):
        import frame_trajectory_artifact

        trajectory_artifact = frame_trajectory_artifact.read_frame_trajectory(
            args.trajectory_in
        )
        execution_goals = ()
        print(
            "[step3r] exact trajectory replay-only -> skip task execution; "
            f"{len(trajectory_artifact.frames)} frames "
            f"render_input_sha256={trajectory_artifact.render_input_hash or 'legacy-v1'}"
        )
        if not trajectory_artifact.has_explicit_command_spans:
            print(
                "[step3r] legacy trajectory has no authoritative command boundaries; "
                "temporal camera lookahead will remain disabled"
            )

    span_tracker = (
        CommandFrameSpanTracker(compiled_commands)
        if trajectory_artifact is None
        else None
    )
    execution_pose_transfers = _execution_pose_transfer_schedule(
        scene, compiled_commands
    )
    scoring_baseline_relations = scene.episode_baseline_relations()
    intervention_command_index = _assistant_intervention_command_index(
        compiled_commands,
        getattr(scene, "behavior_task", None),
    )
    intervention_baseline_captured = False
    failed_navigation_commands: set[int] = set()

    def _settle_placement_attempt(goal, frame_start):
        if not _percmd_settle or not action_check._placed_objects(
            env.motion.frames[frame_start:]
        ):
            return None, None, {}
        if controllers:
            _apply_openable_state(
                controllers,
                _openable_state_after_frames(controllers, env.motion.frames),
            )

        def _lock_slot() -> bool:
            slot = _kinematic_place_slot(scene, goal)
            if slot is None:
                return False
            subject, _parent = slot
            entity = scene.entities.get(subject)
            path = str(getattr(entity, "prim_path", None) or "")
            placed = action_check._placed_objects(env.motion.frames[frame_start:])
            row = placed.get(path)
            if row is None:
                return False
            rest = np.asarray(row[1], dtype=np.float64)
            if rest.shape != (4, 4):
                current = se._read_world_col(stage, path)
                if current is None or rest.shape != (3,):
                    return False
                current[:3, 3] = rest
                rest = current
            if not se._write_world_col(stage, path, rest):
                return False
            prim = stage.GetPrimAtPath(path)
            if prim is not None and prim.IsValid():
                ph.set_rigid_kinematic(prim, True)
            return True

        def _settle_physics():
            paused = _freeze_dynamic_prims(stage, fdp_paths)
            try:
                _ensure_settle_clock()
                return action_check.settle_placements(
                    ph,
                    stage,
                    rig,
                    core,
                    scene.cworld,
                    env.motion.frames,
                    frame_start,
                    step_fn=_settle_step,
                )
            finally:
                _restore_dynamic_prims(stage, paused)

        notes, slot_locked = _settle_or_lock_placement(
            scene, goal, _lock_slot, _settle_physics
        )
        if slot_locked:
            child, parent = _kinematic_place_slot(scene, goal) or ("", "")
            print(
                f"[place-slot] {child}->{parent}: retained the authored rack-slot pose"
            )
        se.sync_carried_to_root(
            scene,
            str((goal.meta or {}).get("_carried_root") or ""),
            (goal.meta or {}).get("_carried") or (),
        )
        ejected = action_check.ejected_paths(notes)
        detail = None
        if ejected:
            detail = (
                f"settle launched {[p.rsplit('/', 1)[-1] for p in ejected]} "
                f"instead of producing a rest ({notes[ejected[0]]})"
            )
        status = se.placement_relation_status(scene, goal)
        if detail is None and status is not None and not status["ok"]:
            detail = (
                f"live {status['predicate'].upper()} relation is false after settle: "
                f"{status['subject']} -> {status['target']} "
                f"at chosen_xy={status['chosen_xy']}"
            )
        return detail, status, notes

    for _gi, goal in enumerate(execution_goals):
        assert span_tracker is not None
        command_index = span_tracker.begin_goal(
            _gi,
            len(env.motion.frames),
            frame_limit=int(args.frames or 0),
        )
        if command_index is None:
            print(
                f"[step3r] frame budget {int(args.frames)} reached before the "
                "next source command; "
                "stopping program execution for this partial render"
            )
            break
        if _open_goal_blocked_by_failed_navigation(
            failed_navigation_commands, command_index, goal
        ):
            detail = (
                "skipped because the preceding navigation in the same source command failed"
            )
            print(f"[exec] {goal.prim.value}: FAILED path_blocked ({detail}) -> no frames")
            env.reports.append((goal, PrimitiveError.PATH_BLOCKED))
            env.report_detail[id(goal)] = detail
            continue
        if (
            not intervention_baseline_captured
            and intervention_command_index is not None
            and command_index == intervention_command_index
        ):
            scoring_baseline_relations = set(scene.relations(source="physics"))
            intervention_baseline_captured = True
        _prev = len(env.motion.frames)
        _nav_transit_restore = None
        # Change 1: resolve an UNRESOLVED navigate (target_key set, target_xy None) HERE, at runtime -- pick
        # the stand nearest-reachable from the human's LIVE position (env.st.root_xy), among candidates that
        # are non-penetrating + arm-can-reach + walk-reachable from there. None -> FAIL the command honestly
        # (no fabricated stand), recording the specific stance failure in strict mode. Producers never see
        # an unresolved nav.
        if (
            goal.prim is command_grammar.Primitive.NAVIGATE_TO
            and goal.target_xy is None
            and goal.target_key is not None
        ):
            if getattr(scene, "nav_grid", None) is not None:
                _cur = tuple(env.st.root_xy)
                _followup_goal = command_grammar.navigation_action_followup(
                    goals,
                    compiled_commands,
                    _gi,
                    scene,
                )
                _stand_request = navigation_stand_request(
                    scene, goal, _followup_goal, actor_xy=_cur
                )
                _stand_key = _stand_request["stand_key"]
                _serves_openable = _stand_request["serves_openable"]
                _openable_kind = _openable_action_kind(
                    scene, _followup_goal, _stand_key
                )
                _previews_openable_action = _navigation_previews_openable_action(
                    scene,
                    _followup_goal,
                    _stand_key,
                    serves_openable=_serves_openable,
                )
                _apply_openable_content_action_reach_tolerance(
                    scene, _followup_goal, _stand_key
                )
                _content_preview = _openable_kind in ("insert", "extract")
                _serves_read = _stand_request["serves_read"]
                _requires_reach = bool(
                    _stand_request["require_reach"] and not _content_preview
                )
                _is_nav_door = _stand_request["is_nav_door"]
                _target_transit_avoid = set()
                _navigation_fallback = {}
                _stand = _continuous_container_close_stand(
                    scene,
                    env,
                    rig,
                    goals,
                    _gi,
                    goal,
                    _followup_goal,
                    _stand_request,
                )
                if _stand is not None:
                    print(
                        f"[runtime-stand] '{_stand_key}': reusing the safe pick stand for "
                        "the immediate container close"
                    )
                else:
                    _cached_open_stand = _validated_open_stand(
                        scene, _followup_goal, _stand_key
                    )
                    _fixed_open_attempts = (
                        (_cached_open_stand[0], None)
                        if _cached_open_stand is not None
                        else (None,)
                    )
                    for _validated_stand_xy in _fixed_open_attempts:
                        for _front_standoff in _front_access_standoffs_for_goal(
                            _followup_goal, scene
                        ):
                            for _stand_contact, _contact_fraction in _stand_request[
                                "contact_attempts"
                            ]:
                                if _serves_openable:
                                    _followup_goal.meta["_door_contact_fraction"] = (
                                        _contact_fraction
                                    )
                                _stand = iv.pick_stand_runtime(
                                    ph,
                                    stage,
                                    rig,
                                    scene,
                                    _stand_key,
                                    _cur,
                                    contact_override=_stand_contact,
                                    contact_overrides=_stand_request[
                                        "contact_overrides"
                                    ],
                                    complete_contact_trajectory=_stand_request[
                                        "complete_contact_trajectory"
                                    ],
                                    nearest_to_contact=_serves_openable,
                                    cue_read=_serves_read,
                                    require_reach=_requires_reach,
                                    require_openable_sweep_clearance=bool(
                                        _is_nav_door
                                        or (
                                            _serves_openable
                                            and _requires_openable_sweep_clearance(
                                                scene,
                                                _followup_goal,
                                            )
                                        )
                                    ),
                                    openable_action_phase=bool(_serves_openable),
                                    openable_action_kind=_openable_kind,
                                    front_access_standoff_m=_front_standoff,
                                    validated_stand_xy=_validated_stand_xy,
                                    route_gate=_ego_route_gate(scene),
                                    start_yaw=float(
                                        getattr(env.st, "root_yaw", 0.0)
                                    ),
                                    gait_profile=getattr(env, "g", None),
                                    gait_cum_dist=float(
                                        getattr(env.st, "gait_cum_dist", 0.0)
                                    ),
                                    reach_tol=_navigation_stand_reach_tolerance(
                                        scene, _stand_request, _followup_goal
                                    ),
                                    transit_avoid_out=_target_transit_avoid,
                                    navigation_fallback_out=(
                                        _navigation_fallback
                                        if getattr(
                                            scene,
                                            "egocentric_camera_only_collision",
                                            True,
                                        )
                                        or not getattr(scene, "strict_acceptance", False)
                                        else None
                                    ),
                                    candidate_validator=(
                                        _openable_candidate_validator(
                                            env,
                                            rig,
                                            scene,
                                            _followup_goal,
                                            _stand_key,
                                        )
                                        if _previews_openable_action
                                        else None
                                    ),
                                    action_state=(
                                        env.st
                                        if _serves_openable or _content_preview
                                        else None
                                    ),
                                    centered_openable=_stand_request[
                                        "centered_openable"
                                    ],
                                )
                                if _stand is not None:
                                    if (
                                        _serves_openable
                                        and _contact_fraction < 1.0
                                    ):
                                        print(
                                            f"[runtime-stand] '{_stand_key}': handle stand unavailable -> "
                                            f"using the {_contact_fraction:.2f}-span door contact"
                                        )
                                    if _validated_stand_xy is not None:
                                        print(
                                            f"[open-stand] '{_stand_key}': revalidated cached "
                                            f"closed-door stand ({_stand[0][0]:.2f},"
                                            f"{_stand[0][1]:.2f})"
                                        )
                                    break
                            if _stand is not None:
                                break
                        if _stand is not None:
                            break
                _stationary_openable = se._openable_access_entity(scene, _stand_key)
                _stationary_operator = bool(
                    (
                        getattr(_stationary_openable, "meta", None)
                        if _stationary_openable is not None
                        else None
                    )
                    and _stationary_openable.meta.get("stationary_operator")
                )
                if (
                    _stand is None
                    and not _stationary_operator
                    and _allow_actor_near_stand_fallback(
                        _stand_request,
                        content_preview=_content_preview,
                    )
                    and iv.actor_near_entity(scene, _stand_key, _cur)
                ):
                    _stand = ((_cur[0], _cur[1]), float(env.st.root_yaw))
                    _target_transit_avoid.clear()
                    print(
                        f"[runtime-stand] '{goal.target_key}': no better stand, but already "
                        "within interaction range -> explicit walk is a no-op"
                    )
                if _stand is None:
                    _stand_reason = (
                        PrimitiveError.NO_COLLISION_FREE_STANCE
                        if getattr(scene, "strict_acceptance", False)
                        else PrimitiveError.PATH_BLOCKED
                    )
                    print(
                        f"[runtime-stand] '{goal.target_key}': NO reachable non-penetrating stand from "
                        f"cur=({_cur[0]:.2f},{_cur[1]:.2f}) -> FAIL command "
                        f"({_stand_reason.value}; no fabricated stand)"
                    )
                    env.reports.append((goal, _stand_reason))
                    env.report_detail[id(goal)] = (
                        f"runtime-stand: no reachable non-penetrating stand for "
                        f"'{goal.target_key}' from cur=({_cur[0]:.2f},{_cur[1]:.2f})"
                    )
                    failed_navigation_commands.add(int(command_index))
                    continue
                _nav_transit_restore = set(scene.nav_grid.transit_avoid)
                scene.nav_grid.transit_avoid.clear()
                scene.nav_grid.transit_avoid.update(_target_transit_avoid)
                goal.target_xy, goal.target_yaw = _stand
                if _navigation_fallback.get("selected"):
                    scene.egocentric_collision_fallback_used = True
                    goal.meta["_egocentric_navigation_fallback"] = True
                    if _navigation_fallback.get("force_straight"):
                        goal.meta["_camera_safe_navigation_fallback"] = True
                    print(
                        f"[runtime-stand] '{goal.target_key}': using least-penetrating "
                        "camera-clear fallback"
                    )
                print(
                    f"[runtime-stand] '{goal.target_key}': picked "
                    f"({_stand[0][0]:.2f},{_stand[0][1]:.2f}) yaw={_stand[1]:.0f} "
                    f"(nearest cur=({_cur[0]:.2f},{_cur[1]:.2f}))"
                )
                # Change 2 (CLOSED-first door handling): open any CLOSED door between the human and the resolved
                # stand BEFORE walking there. Runs the door NAVIGATE + OpenDoor via env.run_goal, then the
                # NAVIGATE below re-plans through the now-open doorway. False -> honest fail (no penetration).
                _door_why: list[str] = []
                if not _runtime_open_doors(
                    env,
                    ph,
                    stage,
                    rig,
                    scene,
                    _cur,
                    goal.target_xy,
                    why=_door_why,
                    controllers=controllers,
                    target_yaw=goal.target_yaw,
                    gait_profile=getattr(env, "g", None),
                    gait_cum_dist=float(getattr(env.st, "gait_cum_dist", 0.0)),
                    allow_camera_safe_fallback=bool(
                        _navigation_fallback.get("force_straight")
                    ),
                    navigation_fallback_out=(
                        _navigation_fallback
                        if getattr(scene, "egocentric_camera_only_collision", True)
                        or not getattr(scene, "strict_acceptance", False)
                        else None
                    ),
                ):
                    # "unreachable even with doors open" was the message for all five door-pre-pass
                    # causes; ``_door_why`` now names the one that actually fired.
                    _why = _door_why[-1] if _door_why else "no cause reported"
                    print(
                        f"[runtime-stand] '{goal.target_key}': stand "
                        f"({goal.target_xy[0]:.2f},{goal.target_xy[1]:.2f}) -> FAIL command "
                        f"(path_blocked; {_why})"
                    )
                    env.reports.append((goal, PrimitiveError.PATH_BLOCKED))
                    env.report_detail[id(goal)] = (
                        f"runtime-door: stand ({goal.target_xy[0]:.2f},{goal.target_xy[1]:.2f}) for "
                        f"'{goal.target_key}': {_why}"
                    )
                    scene.nav_grid.transit_avoid.clear()
                    scene.nav_grid.transit_avoid.update(_nav_transit_restore)
                    failed_navigation_commands.add(int(command_index))
                    continue
                if _navigation_fallback.get("force_straight"):
                    scene.egocentric_collision_fallback_used = True
                    goal.meta["_egocentric_navigation_fallback"] = True
                    goal.meta["_camera_safe_navigation_fallback"] = True
                if _navigation_fallback.get("path"):
                    scene.egocentric_collision_fallback_used = True
                    goal.meta["_egocentric_navigation_fallback"] = True
                    goal.meta["_camera_safe_navigation_path"] = _navigation_fallback[
                        "path"
                    ]
            else:
                # #3: hand-tuned non-nav_grid demos (openclose/sit) keep the DECLARED stand pose -- no runtime
                # pick, no door handling -> byte-identical to before Change 1.
                _stand2, _yaw2 = scene.stand_pose_for(goal.target_key)
                goal.target_xy = (float(_stand2[0]), float(_stand2[1]))
                goal.target_yaw = float(_yaw2)
        _placement_retry = (
            _capture_placement_retry(env, scene, goal, len(env.motion.frames))
            if goal.prim
            in (
                command_grammar.Primitive.PLACE,
                command_grammar.Primitive.PLACE_INSIDE,
            )
            and getattr(goal, "place_xy", None) is None
            else None
        )
        if (
            goal.prim is command_grammar.Primitive.OPEN
            and goal.obj_key not in (getattr(scene, "open_objs", None) or set())
            and not _select_runtime_door_swing(
                scene,
                controllers,
                goal.obj_key,
                tuple(env.st.root_xy),
            )
        ):
            logger.warning(
                "[runtime-door] '%s': neither swing direction passed the local clearance "
                "preview; preserving the authored direction",
                goal.obj_key,
            )
        if goal.prim is command_grammar.Primitive.CLOSE:
            _mark_direct_continuous_container_close(
                scene,
                env.st,
                goals,
                _gi,
                goal,
            )
        _front_ready, _front_detail = _prepare_front_opening_action_stand(
            env,
            ph,
            stage,
            rig,
            scene,
            goal,
            iv,
            controllers=controllers,
        )
        if not _front_ready:
            _front_detail = _front_detail or "camera-safe sweep stand unavailable"
            print(
                f"[exec] {goal.prim.value}: FAILED no_collision_free_stance "
                f"({_front_detail}) -> no frames"
            )
            env.reports.append((goal, PrimitiveError.NO_COLLISION_FREE_STANCE))
            env.report_detail[id(goal)] = _front_detail
            continue
        try:
            _obs, _done, _info = env.run_goal(goal)
        finally:
            if _nav_transit_restore is not None:
                scene.nav_grid.transit_avoid.clear()
                scene.nav_grid.transit_avoid.update(_nav_transit_restore)
        _info, _placement_retry = _recover_out_of_reach_action(
            env,
            ph,
            stage,
            rig,
            scene,
            goal,
            _info,
            _placement_retry,
            iv,
            controllers=controllers,
        )
        _prim, _reason, _detail = _info["primitive"], _info["reason"], _info["detail"]
        if (
            goal.prim is command_grammar.Primitive.NAVIGATE_TO
            and _reason is not PrimitiveError.OK
        ):
            failed_navigation_commands.add(int(command_index))
        while True:
            if _reason is PrimitiveError.OK:
                print(
                    f"[exec] {_prim}: OK (+{_info['n_new_frames']} frames, total {_info['n_frames']})"
                )
            elif se._DISPATCH.get(goal.prim) is None:
                print(f"[exec] {_prim}: UNSUPPORTED in P0 -> skipped (no frames)")
            else:
                print(
                    f"[exec] {_prim}: FAILED {_reason.value} ({_detail}) -> no frames"
                )
            _settle_from = (
                int(_placement_retry["frame_start"])
                if _placement_retry is not None
                else _prev
            )
            _failure, _placement_status, _notes = _settle_placement_attempt(
                goal, _settle_from
            )
            for _dp, _ddz, _ddxy in action_check.displaced_paths(_notes):
                print(
                    f"[place-verify] {_dp.rsplit('/', 1)[-1]}: settled {_ddz * 1000:+.0f}mm / "
                    f"{_ddxy * 1000:.0f}mm from where it was released -- the command reported "
                    f"{'FAILED' if _failure else 'OK'}"
                )
            _attempt = int((goal.meta or {}).get("_placement_retry_attempt", 0))
            if (
                _failure is not None
                and _placement_retry is not None
                and _reason is PrimitiveError.OK
                and _attempt + 1 < PLACEMENT_SETTLE_ATTEMPTS
            ):
                _failed_xy = _failed_placement_xy(scene, _placement_retry, goal)
                _inside = bool(
                    _placement_status is not None
                    and _placement_status["predicate"] == "inside"
                )
                _reject_space = (
                    _attempt > 0
                    and _placement_space_count(scene, str(_placement_retry["target"]))
                    > 1
                )
                _restore_placement_retry(env, scene, _placement_retry)
                _next_attempt = se.mark_placement_retry(
                    goal.meta,
                    _failed_xy,
                    inside=_inside,
                    reject_space=_reject_space,
                )
                print(
                    f"[place-settle-retry] {_next_attempt + 1}/{PLACEMENT_SETTLE_ATTEMPTS}: "
                    f"{_failure}; restoring the held object and choosing another site"
                )
                _placement_retry = _capture_placement_retry(
                    env, scene, goal, len(env.motion.frames)
                )
                _obs, _done, _info = env.run_goal(goal)
                _info, _placement_retry = _recover_out_of_reach_action(
                    env,
                    ph,
                    stage,
                    rig,
                    scene,
                    goal,
                    _info,
                    _placement_retry,
                    iv,
                )
                _prim, _reason, _detail = (
                    _info["primitive"],
                    _info["reason"],
                    _info["detail"],
                )
                continue
            if _failure is not None:
                goal.meta["_placement_failure_phase"] = "post_settle"
                print(f"[exec] {_prim}: FAILED unstable_after_release -- {_failure}")
                env._record(goal, PrimitiveError.UNSTABLE_AFTER_RELEASE, _failure)
                _reason = PrimitiveError.UNSTABLE_AFTER_RELEASE
            break
        if _reason is PrimitiveError.OK:
            sync_openable_motion_state(controllers, env.motion.frames[_prev:])
            command = compiled_commands[command_index]
            if _gi + 1 == command.goal_end:
                for event in execution_pose_transfers.get(command_index, ()):
                    _place_variant_at_source(stage, scene, event)
    if span_tracker is not None:
        command_spans = list(
            span_tracker.finish(
                len(env.motion.frames), frame_limit=int(args.frames or 0)
            )
        )
    if _settle_started[
        0
    ]:  # restore the render-loop timestep after the per-command settles
        (
            ph.set_scene_timestep(stage, FPS, substeps=5)
            if fdp
            else ph.set_scene_timestep(stage, FPS)
        )
    motion, reports = env.motion, env.reports
    if trajectory_artifact is not None:
        source_frames = list(trajectory_artifact.frames)
        source_command_spans = list(trajectory_artifact.command_spans)
        authoritative_command_spans = bool(
            trajectory_artifact.has_explicit_command_spans
        )
    else:
        source_frames = list(motion.frames)
        source_command_spans = list(command_spans)
        authoritative_command_spans = True
    import frame_trajectory_artifact as _frame_artifact

    if source_frames:
        frame_limit = (
            min(len(source_frames), int(args.frames))
            if args.frames
            else len(source_frames)
        )
        frames = list(source_frames[:frame_limit])
        command_spans = list(
            _frame_artifact.clip_command_spans(source_command_spans, frame_limit)
        )
    else:
        frames = []
        command_spans = list(source_command_spans)
    scene.source_command_timeline = [
        {
            "index": int(command.index),
            "source": str(command.source),
            "frame_start": int(span[0]),
            "frame_end": int(span[1]),
        }
        for command, span in zip(compiled_commands, command_spans)
    ]
    n_fail = sum(1 for _, r in reports if r is not PrimitiveError.OK)
    print(f"[step3r] program -> {len(frames)} frames; {n_fail} failed primitive(s)")

    # ISAAC_GAZE_STABILIZE=1 restores the vestibulo-ocular reflex the retargeted clip lacks: the
    # neck counter-rotates so the gaze holds through the gait's pelvis sway. Authored into the
    # MOTION here -- before physics validation and before any render -- so the wearer's neck
    # really turns and compliance, TPV and FPV all see the same body. Stabilising the camera
    # instead would put the lens back somewhere the head is not.
    if os.environ.get("ISAAC_GAZE_STABILIZE") == "1" and frames:
        import gaze_stabilizer

        # Order matters: damping the root height moves the head, and the gaze solve fixates
        # from wherever the head ends up.
        _bob = gaze_stabilizer.damp_vertical_bob(frames, fps=float(FPS))
        frames = _bob.frames
        print(
            f"[step3r] bob-damp: vertical travel {_bob.peak_to_peak_before_mm:.0f} -> "
            f"{_bob.peak_to_peak_after_mm:.0f} mm peak-to-peak; root left exact foot contact "
            f"by at most {_bob.max_root_shift_mm:.1f} mm"
        )
        _vor = gaze_stabilizer.stabilize_gaze(frames, fps=float(FPS))
        frames = _vor.frames
        print(
            f"[step3r] gaze-stabilise: neck correction mean={_vor.mean_correction_deg:.2f}deg "
            f"max={_vor.max_correction_deg:.2f}deg, "
            f"clamped {_vor.clamped_frames}/{len(frames)} frames, "
            f"rate-limited {_vor.rate_limited_frames}/{len(frames)} frames "
            f"(cap {gaze_stabilizer.MAX_GAZE_DEG_PER_FRAME:.0f} deg/frame), "
            f"{_vor.gaze_frames} frames aimed at an authored gaze target"
        )

    visual_state_event_schedule = _compile_visual_state_events(scene, frames)

    # ============================ STEP 2: UNIVERSAL POST-ACTION PHYSICS ============================
    # After EVERY action (grab/put/putinside/open/close/...), validate + correct physics on the ONE
    # shared oracle map (scene.cworld). OBJECTS (real PhysX): each released object is re-settled on the
    # real colliders (release -> settle -> de-penetrate -> read true rest -> re-freeze), fixing float /
    # object-vs-furniture penetration; object-vs-object separation is then checked. The corrected rests
    # are written back into `frames` so the render shows the physics-valid pose. HUMAN (kinematic, cannot
    # be pushed): per-action penetration is DETECTED on the as-rendered joints -> a dirty episode is
    # honestly flagged (P1), never silently rendered. Gated on scene.post_action_check -> non-BEHAVIOR
    # scenes skip it entirely and are byte-identical.
    compliance = None
    # Assigned only inside the gated block below, but read on both exit paths (replay dump).
    _jtraj = None
    _ensure_cworld(scene, stage, ph)
    if (
        trajectory_artifact is None
        and getattr(scene, "post_action_check", False)
        and getattr(scene, "cworld", None) is not None
        and frames
    ):
        import action_check

        _nb_static = scene.solid_boxes()
        _ns_c = {
            "nav_grid": getattr(scene, "nav_grid", None)
            or getattr(scene, "wall_sdf_grid", None)
        }
        _openable_state = _initial_openable_state(controllers)
        _openable_post = []
        _jtraj = []
        for _frame_index, _fr in enumerate(frames):
            _replay_controllers = _controllers_at_openable_state(
                controllers, _openable_state
            )
            _all_nudge_boxes = _nudge_boxes_for(
                scene, _replay_controllers, _nb_static, _fr
            )
            _ns_c["camera_boxes"] = _all_nudge_boxes
            _ns_c["frame_index"] = _frame_index
            _jtraj.append(
                wpp.realize(
                    rig,
                    _fr,
                    scene_boxes=_body_nudge_boxes(
                        scene,
                        _all_nudge_boxes,
                    ),
                    nudge_state=_ns_c,
                )[1]
            )
            _openable_state, _consumed = _step_openable_state(
                controllers, _fr, _openable_state
            )
            _openable_post.append(dict(_openable_state))
        # OBJECTS were already settled PER-COMMAND in the loop above (P5), each left at its real rest + frozen
        # kinematic. So this end-of-episode pass only RE-CHECKS: object-vs-object penetration over the final
        # rests + per-frame human-vs-scene penetration. No object re-settle here -> no clock bracket needed
        # (the human check is pure FK + oracle queries).
        _last_applied = None

        def _apply_compliance_openable(frame_index):
            nonlocal _last_applied
            _state = _openable_post[frame_index]
            if _state == _last_applied:
                return
            _apply_openable_state(controllers, _state)
            _last_applied = _state

        try:
            compliance, frames = action_check.run_action_compliance(
                ph,
                stage,
                rig,
                core,
                scene,
                scene.cworld,
                frames,
                joints_traj=_jtraj,
                settle_objects=False,
                fps=FPS,
                sample_state_fn=(_apply_compliance_openable if controllers else None),
            )
        finally:
            if controllers and _openable_post:
                _apply_openable_state(controllers, _openable_post[-1])
        print(
            f"[step3r][COMPLY] render_grade={compliance['render_grade']} "
            f"(pass={compliance['render_pass']}: {compliance['render_grade_reason']}) "
            f"worst_frac={compliance['human_worst_frac']:.3f} "
            f"[{compliance['human_worst_frac_desc']}] @frame {compliance['human_worst_frac_frame']} | "
            f"ok={compliance['ok']} human_clean={compliance['human_clean']} "
            f"(hit_frames={compliance['human_hit_frames']}/{compliance['human_sampled']} = "
            f"{100.0 * compliance['human_dirty_frac']:.1f}% dirty, longest run "
            f"{compliance['human_longest_run_frames'] / FPS:.2f}s, stride "
            f"{compliance['sample_every']}, joints {compliance['joint_subset']}) "
            f"max_pen={compliance['human_max_pen_mm']:.0f}mm"
            f"{' LOWER-BOUND (saturated)' if compliance['human_pen_saturated'] else ''} "
            f"[{compliance['human_worst_pen_desc']}] @frame {compliance['human_worst_pen_frame']} "
            f"pen_by_seg={ {k: round(v, 1) for k, v in compliance['human_pen_mm_by_seg'].items()} } "
            f"segs={compliance['human_by_seg']} worst_pairs={compliance['human_worst']} "
            f"objects_clean={compliance['objects_clean']} ({compliance['objects']}) "
            f"obj_pen={len(compliance['obj_pen'])}"
        )
    # BEHAVIOR CHALLENGE task goal-check (gated on scene.behavior_task): score the task's OWN BDDL :goal
    # against the PHYSICAL relations (source="physics" -> _relations_inside_live: the settled object's xy is
    # within the target's footprint), NOT the executor's declared effects -- so a [Put] counts as done only
    # when the object PHYSICALLY ended up ON/IN its target. Read-only + post-hoc: this never moves an
    # object or gates the render (a physically-clean video that missed the target is still rendered; the goal
    # verdict is recorded honestly in the manifest). Additive: no-op for every non-BEHAVIOR scene.
    task_goal = None
    _gp = None
    _bt = getattr(scene, "behavior_task", None)
    if _bt and trajectory_artifact is None:
        try:
            import isaac_goal

            # Score exactly what the demo does: each (capped) graspable INSIDE the container. For a
            # many-instance task this is the demoed subset; the derivation itself came from the BDDL goal.
            # GATED per-item + NEGATIVE branch: a task that sets _bt["goals"] scores each item INSIDE its OWN
            # container + rule negatives (restaurant closing); absent -> the original single-container path,
            # byte-identical for every existing bt_* task (they set "container", never "goals"/"neg_constraints").
            _goals = _bt.get("goals")
            _carcass = _bt.get(
                "carcass"
            )  # bookcase assembly (Task 2): score each edge via ATTACHED
            # PREDICATE-CARRYING form first. ``goals`` is a list of (subject, target) with no
            # predicate, so this line used to score every edge as INSIDE -- which was right while
            # INSIDE was the only thing the physics branch could emit, and became a WRONG NUMBER
            # rather than a missing one once ``scene_state._relations_on_live`` landed. A task that
            # does not publish ``goal_preds`` falls through to exactly what it did before.
            _preds = _bt.get("goal_preds")
            if _preds:
                _gp = isaac_goal.from_rows(_preds)
                scene.configure_goal_predicates(_preds)
            elif _carcass:
                _gp = [isaac_goal.attached(c, p) for (c, p) in _carcass]
            elif _goals:
                _gp = [isaac_goal.inside(o, c) for (o, c) in _goals]
            else:
                _gp = [
                    isaac_goal.inside(k, _bt["container"]) for k in _bt.get("grasp", [])
                ]
            _rel = scene.relations(source="physics")
            if os.environ.get("ON_DEBUG"):
                # What the scorer actually had in its hand. A goal that reads false while the object
                # is visibly on the table is either a missing relation or a mismatched key, and
                # those two look identical from the outside.
                print(f"[goal-dbg] held={sorted(scene.held)}")
                for _p in sorted(
                    str(r)
                    for r in _rel
                    if "ON" in str(r) or "INSIDE" in str(r) or "RUNNING" in str(r)
                ):
                    print(f"[goal-dbg] rel {_p}")
                for _g in _gp:
                    print(f"[goal-dbg] want {_g.as_tuple()}")
            task_goal = _task_goal_record(
                _bt,
                _gp,
                _rel,
                scoring_baseline_relations,
            )
            _gmet = bool(task_goal["met"])
            _gprog = float(task_goal["progress"])
            # DIAGNOSTIC: name the units that were PLACED but did not register INSIDE/ATTACHED. Reads the
            # per-sub-goal booleans, so it now covers the carcass + legacy branches too, not just _goals.
            _unmet = [
                f"{r['subject']}->{r['target']}"
                for r in task_goal["goals"]
                if not r["ok"]
            ]
            if _unmet:
                print(f"[step3r][GOAL] UNMET (placed but not scored INSIDE): {_unmet}")
            for _r in task_goal.get("neg", []):
                print(
                    f"[step3r][GOAL]   negative {_r['label']}: "
                    f"{'OK' if _r['ok'] else 'VIOLATED'}"
                    + (f" offenders={_r['offenders']}" if _r["offenders"] else "")
                )
            _tgt = _bt.get("container") or "their rule-correct target"
            print(
                f"[step3r][GOAL] {_bt['task']}: check_goal={_gmet} progress={_gprog:.2f} "
                f"({round(_gprog * len(_gp))}/{len(_gp)} object(s) physically INSIDE {_tgt}) "
                f"sub_goals={task_goal['n_sub_goals_ok']}/{task_goal['n_sub_goals']}"
            )
        except Exception as _ge:  # noqa: BLE001
            print(f"[step3r][GOAL] {_bt['task']}: goal-check error {_ge}")
    if (
        not frames
    ):  # render partial motion if SOME primitives succeeded (e.g. a too-high grab fails)
        print("[step3r] ABORT: no frames produced -> not rendering")
        _nf_out, _nf_tag = _out_and_tag(args)
        _write_episode_manifest(
            _nf_out,
            _nf_tag,
            args,
            render_episode_valid,
            compliance,
            task_goal,
            extra=_episode_extra(
                args,
                scene,
                env,
                goals,
                frames,
                n_fail,
                _t_start,
                "render",
                "no_frames",
            ),
        )
        _exit_process(2)
        return 2
    if n_fail:
        print(
            f"[step3r] WARN: {n_fail} primitive(s) failed -> rendering the successful part only"
        )
    # Fast pre-render gate (no RTX): reach + primitives + settle + init, and the SAME picture verdict
    # the render publishes.
    if args.check:
        box_paths = [f"/World/{c['name']}_Base" for c in controllers.values()]
        box_paths.append(
            "/World/Shelf"
        )  # highreach pillar; guarded by IsValid when absent
        box_paths.append(
            "/World/Counter"
        )  # walkby obstacle; guarded by IsValid when absent
        box_paths.append(
            "/World/Table"
        )  # clutter table; guarded by IsValid when absent
        n_pen, worst_pen, n_floor, worst_floor = _collision_gate(
            stage, rig, frames, scene.floor_z, box_paths, app
        )
        # Planner telemetry on the RAW pre-nudge pose. Reported, never gated -- the human-vs-furniture
        # VERDICT below comes from the compliance pass, which judged the as-rendered (nudged) motion.
        raw_grade = _prenudge_diagnostic(rig, frames, scene)
        raw_s = "n/a" if raw_grade is None else raw_grade["render_grade"]
        # CDS physical-correctness gate (offline D1/D3 proof): the env-init settle already ran above, so
        # here we ASSERT it produced a physically-correct rest -- every settled object's bbox-bottom is
        # within ~1cm of its support (no float/sink) and all_pairs_separation found NO penetrating pair
        # (e.g. cracker/sugar separated). Empty settle_report (openclose/sit/walkby) -> trivial pass.
        settle_ok_all = all(row[4] for row in settle_report)
        no_penetration = len(pen_pairs) == 0
        for label, iz, rz, mm, sok in settle_report:
            print(
                f"[step3r][SETTLE]   {label}: initial_z={iz:.3f} -> rested_z={rz:.3f} "
                f"(moved {mm:.0f}mm) ok={sok}"
            )
        if pen_pairs:
            for a, b, sep in pen_pairs:
                print(
                    f"[step3r][SETTLE]   PENETRATION {a.rsplit('/', 1)[-1]} <-> "
                    f"{b.rsplit('/', 1)[-1]}: min-sep {sep * 1000:.0f}mm"
                )
        elif settle_report:
            print("[step3r][SETTLE]   all-pairs separation: NO penetrating pair")

        # P2(A) SPAWN-TIME non-overlap assert: BEFORE any settle can de-penetrate, the reject-sampled
        # layout must already be physically valid (no clutter-vs-clutter AABB overlap at spawn), so a
        # randomly-generated RL scene never starts with an interpenetrating init. Proven from the planner
        # stats (min_pairwise_gap = the smallest signed AABB gap over every clutter pair, computed from the
        # exact world-bbox spawn footprints): >= 0 means every pair is separated at spawn. Also logs the
        # reject-sampler activity (placed/rejected/attempts) so it is evident in the output. Trivial pass
        # on the other scenes (no clutter_plan_stats). NOTE: an axis-aligned bbox gap is CONSERVATIVE (the
        # rotated banana's real geometry is even more separated), so >= 0 here is a sound spawn-time proof.
        plan_stats = getattr(scene, "clutter_plan_stats", None)
        spawn_nonoverlap = True
        if plan_stats is not None:
            gap = float(plan_stats["min_pairwise_gap"])
            spawn_nonoverlap = (
                not np.isfinite(gap)
            ) or gap >= 0.0  # inf = <2 prims (trivial)
            gap_s = "inf" if not np.isfinite(gap) else f"{gap * 1000:.0f}mm"
            print(
                f"[step3r][PLAN]   reject-sampler: placed={plan_stats['placed']} "
                f"rejected={plan_stats['rejected']} attempts={plan_stats['attempts']} "
                f"min_pairwise_gap={gap_s}"
            )
            print(
                f"[step3r][PLAN]   SPAWN non-overlap (before settle): {'OK' if spawn_nonoverlap else 'FAIL (interpenetrating layout)'} "
                f"(min_pairwise_gap={gap_s} >= 0 required)"
            )
            assert (
                spawn_nonoverlap
            ), f"clutter spawn layout interpenetrates: min_pairwise_gap={gap_s} < 0"

        # CLUTTER-ONLY geometry asserts (D3 fixes A + B); trivially skipped on the other scenes.
        stand_ok, margin_ok = _check_clutter_geometry(stage, scene, clutter_rest_xy)

        # P3 DE-PENETRATION proof (put_penetrate only; trivially True elsewhere): reproduce the release
        # offline -- move /World/Obj to the penetrating [Put] target, flip it DYNAMIC, run a short
        # post-put settle, and ASSERT the WHOLE user requirement (the placed object auto-resolves to a
        # valid on-table rest WITHOUT knocking the blocker off):
        #   (a) blocker base_z within ~5 cm of table_top -- it STAYED ON THE TABLE (catches the v1
        #       knock-off, where the light can fell to the floor z 0.80 -> 0.10);
        #   (b) mug base_z within ~5 cm of table_top -- the released mug also RESTS ON THE TABLE;
        #   (c) all_pairs_separation(mug, blocker) EMPTY -- PhysX de-penetrated the overlap (no residual);
        #   (d) the mug's rested_xy differs from its placed target_xy by > 3 cm -- the correction happened
        #       to the PLACED object (it visibly SLID), not to the blocker (v1's bug).
        # Prints placed_xy -> rested_xy (the slide distance) + the blocker base_z. fdp None -> no
        # put_penetrate dict -> pp_ok True (every other scene trivially passes).
        pp_ok = True
        pp = getattr(scene, "put_penetrate", None)
        if pp is not None:
            import fpv_render as _fr_d
            import fpv_scene as _fs_d
            import object_physics as ophys2

            obj_p, blk_p = pp["obj_path"], pp["blocker_path"]
            tgt = pp["target"]
            top_z = float(pp["top_z"])
            obj_prim2 = stage.GetPrimAtPath(obj_p)
            obj_min_z = float(core.local_obj_min_z(rig))
            # Place the mug ORIGIN so its bbox-bottom sits on the tabletop AT the penetrating xy (this is
            # the last carried/held pose == the release pose, so the flip introduces no jump).
            core.place_obj_resting(rig, np.array([tgt[0], tgt[1], top_z - obj_min_z]))
            ph.set_scene_timestep(stage, FPS, substeps=5)
            fpv_core.start_timeline(rcfg.playback_fps)
            # Release the mug through the ONE lifecycle chokepoint (physics_human.release_to_dynamic): flip
            # kinematic->dynamic + drive the synced resync (advance_once with the timeline PLAYING) that makes
            # the flip LIVE -- a bare SimulationContext.step cannot re-parse it. The resync doubles as the
            # post-play startup warmup (the first steps after play() do not integrate physics), so no separate
            # warmup loop is needed. This reproduces the render-loop release offline (physics only, no capture).
            ph.release_to_dynamic([obj_prim2], step_fn=lambda: _fr_d.advance_once(rcfg))
            # POST-PUT settle: PhysX de-penetrates + rests the released mug on camera.
            for _ in range(40):
                _fr_d.advance_once(rcfg)
            pen2 = ophys2.all_pairs_separation(ph, stage, [obj_p, blk_p])
            no_residual = len(pen2) == 0
            final_xyz = ophys2.read_prim_xyz(stage, obj_p)
            mug_base_z = float(_fs_d.compute_world_bbox(obj_prim2).GetMin()[2])
            blk_base_z = float(
                _fs_d.compute_world_bbox(stage.GetPrimAtPath(blk_p)).GetMin()[2]
            )
            slide = float(np.hypot(final_xyz[0] - tgt[0], final_xyz[1] - tgt[1]))
            blocker_on_table = abs(blk_base_z - top_z) <= 0.05
            mug_on_table = abs(mug_base_z - top_z) <= 0.05
            mug_moved = slide > 0.03
            pp_ok = no_residual and blocker_on_table and mug_on_table and mug_moved
            print(
                f"[step3r][DEPEN] released mug at penetrating target "
                f"placed_xy=({tgt[0]:.3f},{tgt[1]:.3f}) -> rested_xy=({final_xyz[0]:.3f},"
                f"{final_xyz[1]:.3f}) slide={slide * 1000:.0f}mm (need >30mm) mug_moved={mug_moved}"
            )
            print(
                f"[step3r][DEPEN] blocker '{blk_p.rsplit('/', 1)[-1]}' base_z={blk_base_z:.3f} "
                f"(table_top {top_z:.3f}, on_table={blocker_on_table}); mug base_z={mug_base_z:.3f} "
                f"(on_table={mug_on_table}); obj<->blocker separation pairs={len(pen2)} "
                f"no_residual={no_residual} -> pp_ok={pp_ok}"
            )
            for a, b, sep in pen2:
                print(
                    f"[step3r][DEPEN]   RESIDUAL PENETRATION {a.rsplit('/', 1)[-1]} <-> "
                    f"{b.rsplit('/', 1)[-1]}: min-sep {sep * 1000:.0f}mm"
                )

        # The human-vs-furniture term is the SAME graded verdict the render path publishes, read back
        # through the one accessor -- --check can no longer FAIL an episode the render calls clean. It
        # stays strict on everything else: a failed primitive (an unreachable stand lands here), a body
        # vert driven through a solid box, a bad settle, interpenetrating objects, a broken init.
        human_pass, human_grade, human_reason = action_check.render_verdict(compliance)
        # tolerate a shallow hand-on-surface graze (<40 mm) while grasping; flag real body-through-solid
        ok = (
            n_fail == 0
            and worst_pen < 0.04
            and settle_ok_all
            and no_penetration
            and stand_ok
            and margin_ok
            and spawn_nonoverlap
            and pp_ok
            and human_pass
            and (compliance is None or compliance["objects_clean"])
            and render_episode_valid  # P0: never call a soft_init-flagged penetrating init "OK"
        )
        print(
            f"[step3r] --check {'OK' if ok else 'FAIL'}: {len(frames)} frames, {n_fail} failed | "
            f"RENDER_GRADE {human_grade or 'n/a'} pass={human_pass} ({human_reason}) | "
            f"GATE worst-penetration {worst_pen * 1000:.0f}mm ({n_pen} verts), "
            f"{n_floor} feet-below-floor (worst {worst_floor * 1000:.0f}mm) | "
            f"RAW pre-nudge grade={raw_s} (diagnostic) "
            f"episode_valid={render_episode_valid} | "
            f"SETTLE ok_all={settle_ok_all} no_pen={no_penetration} "
            f"standback={stand_ok} edge_margin={margin_ok} spawn_nonoverlap={spawn_nonoverlap} "
            f"pp_ok={pp_ok}"
        )
        _chk_out, _chk_tag = _out_and_tag(args)
        _chk_extra = _episode_extra(
            args, scene, env, goals, frames, n_fail, _t_start, "check", "check"
        )
        _chk_extra.update(
            {
                "check_ok": bool(ok),
                # _collision_gate runs on the --check branch only, so these are null on a render.
                "floor_below_frames": int(n_floor),
                "floor_below_worst_mm": round(float(worst_floor) * 1000.0, 1),
                # PRE-NUDGE planner telemetry, not the picture score: the same sweep as
                # compliance.render_grade run on the RAW pose the renderer never draws. Compare the two
                # to see how much the runtime nudge is rescuing. Null when the oracle is absent. These
                # REPLACE swept_overlap_frames, whose number came from a different obstacle set -- a
                # rename rather than a silent redefinition, so a stale reader gets None, not a wrong value.
                "prenudge_grade": (
                    None if raw_grade is None else str(raw_grade["render_grade"])
                ),
                "prenudge_worst_frac": (
                    None
                    if raw_grade is None
                    else round(float(raw_grade["human_worst_frac"]), 4)
                ),
                "prenudge_hit_frames": (
                    None if raw_grade is None else int(raw_grade["human_hit_frames"])
                ),
            }
        )
        _write_episode_manifest(
            _chk_out,
            _chk_tag,
            args,
            render_episode_valid,
            compliance,
            task_goal,
            extra=_chk_extra,
        )
        _write_replay_dump(_chk_out, _chk_tag, args, scene, frames, _jtraj)
        _exit_process(0 if ok else 2)
        return 0 if ok else 2

    # Render-time collision nudge: the RAW motion may clip a solid box (the walkby obstacle, a base);
    # shift the posed body/object out of it per-frame (smoothed via nudge_state). solid_boxes() covers
    # every entity with a footprint; empty -> a safe no-op. Bound at the render entry points ONLY. The
    # per-frame set ADDS each openable's live open-drawer extent (nudge-only; the reach gate above stays
    # on the static solid_boxes()).
    nudge_static = scene.solid_boxes()
    nudge_state: dict = {}
    # P3: thread the per-episode nav grid into the runtime nudge so a core body joint that sways into a
    # WALL (walls are absent from solid_boxes) is pushed out via the grid SDF. Prefer the routing grid; if
    # nav did not fully connect (scene.nav_grid None) fall back to the wall-only carrier (P0-D) so the
    # wall-push still fires on straight-line legs. None for wall-less scenes -> the nudge stays byte-identical.
    nudge_state["nav_grid"] = getattr(scene, "nav_grid", None) or getattr(
        scene, "wall_sdf_grid", None
    )

    init_nudge_state = {"nav_grid": nudge_state["nav_grid"]}
    init_openable_state = _initial_openable_state(controllers)
    init_controllers = _controllers_at_openable_state(controllers, init_openable_state)
    init_boxes = _nudge_boxes_for(scene, init_controllers, nudge_static, frames[0])
    init_nudge_state["camera_boxes"] = init_boxes
    init_nudge_state["frame_index"] = 0
    V_init, J_init, obj_init = wpp.realize(
        rig,
        frames[0],
        scene_boxes=_body_nudge_boxes(scene, init_boxes),
        nudge_state=init_nudge_state,
    )
    core.author_body_once(stage, rig, V_init)
    # /World/Obj was already built + settled + frozen KINEMATIC above; just write its first-frame pose.
    if has_obj and obj_init is not None:
        core.set_object_world(rig, obj_init)
    _opaque_graspables(stage, scene)

    # P3: build the INVISIBLE kinematic human proxy ONLY on the free-dynamic path, so the moving human +
    # the carried object PUSH the DYNAMIC free objects via PhysX (reportKinematicStaticPairs). Runs AFTER
    # J_init exists and AFTER the settle set the substepped scene rate. fdp None -> proxy None (no proxy
    # in the render loop for openclose/sit/highreach/walkby + the frozen clutter demo -> byte-identical).
    proxy = None
    dynamic_replay = None
    if fdp:
        import dynamic_render_replay
        import object_physics

        proxy = ph.build_proxy_rig(stage, rig, J_init)
        ph.enable_contact_reports(
            stage, [b.path for b in proxy.bones] + list(fdp_paths) + ["/World/Obj"]
        )
        dynamic_replay = dynamic_render_replay.DynamicRenderReplay(
            stage,
            ph,
            object_physics,
            proxy,
            fdp_paths,
            reg,
        )
    import egocentric_camera as _ego_camera
    import egocentric_camera_isaac as _ego_isaac
    import head_mount

    # ISAAC_HEAD_MOUNT=1 replaces the synthesised gaze policy with IsaacLab's CameraCfg
    # contract: the FPV rides the head link on a fixed offset with a fixed lens, and nothing
    # else may move it. OFF by default, because the motion producer does not yet author head
    # orientation (`semantic_executor` writes no neck/head rotation) -- so under a rigid mount
    # the wearer looks wherever the torso points and never turns to face a cue.
    # ISAAC_HEAD_MOUNT_ROLL=0 additionally levels the horizon, to A/B head roll on its own.
    _head_mount = os.environ.get("ISAAC_HEAD_MOUNT") == "1"
    _head_mount_level = os.environ.get("ISAAC_HEAD_MOUNT_ROLL") == "0"
    cop = smplh_camera.make_fpv_camera(
        stage,
        "/World/FpvCam",
        W,
        Hh,
        near_clip=head_mount.NEAR_CLIP_M,
    )
    tcop = smplh_camera.make_tpv_camera(stage, "/World/TpvCam", W, Hh)
    fpv_core.add_lighting(stage, app_cfg)
    render_runtime._boost_lights(stage)

    ego_adapter = _ego_isaac.IsaacCameraAdapter.from_scene(scene, nudge_static)
    camera_local_bounds = _camera_object_local_bounds(stage, scene, rig)
    cam = {
        "ego_state": _ego_camera.INITIAL_CAMERA_STATE,
        "ego_yaw_clamps": 0,
        "ego_pitch_clamps": 0,
        "clearance_frames": 0,
        "clearance_max_shift_m": 0.0,
    }

    def _apply_fpv_pose(pose: fpv_render.CameraRenderPose) -> None:
        focal_attr = cam.get("fpv_focal")
        if focal_attr is None:
            from pxr import UsdGeom

            focal_attr = cam["fpv_focal"] = UsdGeom.Camera(
                cop.GetAttr().GetPrim()
            ).GetFocalLengthAttr()
        focal_attr.Set(pose.focal_mm)
        if pose.up is not None:
            smplh_camera.drive_fpv_rigid(cop, pose.eye, pose.look, pose.up)
            return
        smplh_camera.drive_fpv_gaze(
            cop,
            pose.eye,
            pose.look,
            np.asarray(pose.eye) + np.asarray(pose.look),
            eye_push=0.0,
            aim_blend=1.0,
        )

    def _policy_fpv_pose(fr, J, obj_T, solid_boxes, physx_owned, camera_output):
        """Today's synthesised-gaze camera: an aim point plus a rate-limited slew."""
        frame_adapter = ego_adapter.with_solid_boxes(solid_boxes)
        camera_frame = frame_adapter.camera_frame(
            fr,
            J,
            obj_T,
            dt_seconds=1.0 / FPS,
            free_focus=cam.get("free_focus"),
            physx_owned=physx_owned,
        )
        if camera_output is None:
            camera_step = _ego_camera.step_camera(
                camera_frame,
                cam["ego_state"],
                framing=frame_adapter.framing,
            )
            cam["ego_state"] = camera_step.state
            camera_output = camera_step.output
        cam["last_output"] = camera_output
        cam["ego_yaw_clamps"] += int(
            camera_output.diagnostics.physiological_yaw_clamped
        )
        cam["ego_pitch_clamps"] += int(
            camera_output.diagnostics.physiological_pitch_clamped
        )
        return fpv_render.CameraRenderPose(
            tuple(float(value) for value in camera_output.eye),
            tuple(float(value) for value in camera_output.look),
            float(camera_output.focal_mm),
        )

    def _rigid_fpv_pose(fr, J):
        """The head link's own pose plus a fixed mount offset and a fixed lens.

        Reads ``fr.pose156`` rather than the pose ``realize`` actually FK'd. Those differ only
        on the rare frames where the collision nudge applies its capped spine lean
        (``humanoid_motion.py:461``, <=10 deg), which tilts the head slightly without the
        camera following. Known residual, orders smaller than the defect this removes.
        """
        eye, forward, up = head_mount.mount(
            fr.pose156,
            fr.yaw,
            J[wpp.H.HEAD_J],
            level_horizon=_head_mount_level,
        )
        return _head_mount_frame_camera_pose(
            fpv_render.CameraRenderPose(
                tuple(float(value) for value in eye),
                tuple(float(value) for value in forward),
                head_mount.FOCAL_MM,
                tuple(float(value) for value in up),
            ),
            fr,
        )

    def drive_cams(
        fr,
        J,
        obj_T,
        solid_boxes,
        *,
        physx_owned: bool = False,
        camera_output=None,
    ):
        pose = (
            _rigid_fpv_pose(fr, J)
            if _head_mount
            else _policy_fpv_pose(fr, J, obj_T, solid_boxes, physx_owned, camera_output)
        )
        if not _head_mount:
            adjusted_eye, adjusted_look = _pulled_back_camera_pose(
                pose.eye,
                pose.look,
                fr.yaw,
                _camera_pullback_for_frame(scene, fr, camera_local_bounds),
            )
            adjusted_eye, adjusted_look = _moved_forward_camera_pose(
                adjusted_eye,
                adjusted_look,
                _camera_forward_for_frame(scene, fr),
            )
            adjusted_eye, adjusted_look = _shifted_lateral_camera_pose(
                adjusted_eye,
                adjusted_look,
                fr.yaw,
                _camera_lateral_for_frame(scene, fr),
            )
            adjusted_eye, adjusted_look = (
                _stabilize_large_front_container_camera_pose(
                    scene,
                    fr,
                    adjusted_eye,
                    adjusted_look,
                    camera_local_bounds,
                )
            )
            adjusted_eye, adjusted_look = (
                _preserve_large_front_container_egocentric_pose(
                    scene,
                    fr,
                    adjusted_eye,
                    adjusted_look,
                    fr.yaw,
                    J[wpp.H.HEAD_J],
                    camera_local_bounds,
                )
            )
            pose = fpv_render.CameraRenderPose(
                adjusted_eye,
                adjusted_look,
                pose.focal_mm,
                pose.up,
            )
            clearance = _ego_camera.enforce_camera_clearance(
                pose.eye,
                pose.look,
                head_world=J[wpp.H.HEAD_J],
                occluders=_camera_object_aabbs(fr, obj_T, rig, camera_local_bounds),
                clearance_m=_camera_object_clearance_for_frame(fr),
            )
            if clearance.adjusted:
                cam["clearance_frames"] += 1
                cam["clearance_max_shift_m"] = max(
                    cam["clearance_max_shift_m"], clearance.shift_m
                )
                pose = fpv_render.CameraRenderPose(
                    clearance.eye,
                    clearance.look,
                    pose.focal_mm,
                )
            if not clearance.adjusted:
                adjusted_eye, adjusted_look = (
                    _preserve_large_front_container_egocentric_pose(
                        scene,
                        fr,
                        pose.eye,
                        pose.look,
                        fr.yaw,
                        J[wpp.H.HEAD_J],
                        camera_local_bounds,
                    )
                )
                pose = fpv_render.CameraRenderPose(
                    adjusted_eye,
                    adjusted_look,
                    pose.focal_mm,
                    pose.up,
                )
        cam["last_frame"] = fr
        cam["last_head"] = np.asarray(J[wpp.H.HEAD_J], dtype=np.float64).copy()
        cam["last_render_pose"] = pose
        cam["last_delivered_pose"] = pose
        _apply_fpv_pose(pose)
        if getattr(scene, "doors", None) or getattr(scene, "tpv_chase", False):
            # DOOR demo / behavior-task (tpv_chase): an elevated 3/4 chase that TRACKS the human so the walk
            # + interaction stay in frame across rooms. The chase base is offset toward OPEN floor: a FIXED
            # diagonal (hx+2,hy+2) exits the building in a big scene -> the cam lands in/behind an exterior
            # wall = a fully BLACK TPV. Try compass offsets against the nav grid, keep the first whose ground
            # cell is FREE (sticky via cam["tpv_off"], to avoid per-frame jitter), and fall back to directly
            # overhead (always over the open cell the human stands on). Ceilings are not loaded, so 3.4 m
            # looks down into the open-top rooms and clears interior walls between the cam and the human.
            hx, hy = float(fr.xy[0]), float(fr.xy[1])
            # NOTE: TPV chase uses ONLY scene.nav_grid (the connected routing grid), NOT the P0-D wall-only
            # carrier -- feeding the carrier here regressed the chase cam on non-connected scenes (composting
            # TPV went black, toys TPV dropped through the floor). The FPV eye push-out + the runtime nudge DO
            # use the wall carrier (that is the P0-D deliverable); the TPV is QA-only and keeps its prior grid.
            _ng = getattr(scene, "nav_grid", None)
            # NEAR-OVERHEAD chase, driven by two hard constraints learned the hard way:
            #  * an elevated eye (3.4 m+) renders BLACK under the roof a CLOSED scene keeps as colliders, and
            #  * a far horizontal offset (2 m) buries the cam in kitchen counters/cabinets (no line-of-sight).
            # So keep it LOW (2.2 m, below any roof/soffit) with a SMALL offset (0.9 m): nearly overhead, so
            # the sight line to the human is near-vertical and only an obstacle DIRECTLY above them (rare) can
            # block it. Offset toward OPEN floor via the nav grid, then push out of any wall. Robust across an
            # open living room AND a tight galley kitchen.
            # A nicer 3/4 offset while WALKING (open aisles); a PURE top-down while STATIONARY placing. When
            # the human works beside a counter/cabinet, ANY horizontal offset makes the eye->human sight line
            # graze into that furniture (the "buried in dark wood" placement frames). A vertical sight line
            # (offset 0, straight down over the human on open floor) can only be blocked by something DIRECTLY
            # overhead -- rare -- so it stays clear.
            _R = 0.9 if fr.fpv_mode == "walk" else 0.2
            ox, oy = _R, _R
            if _ng is not None:
                prev = cam.get("tpv_off")
                cands = ([prev] if prev is not None else []) + [
                    (_R, _R),
                    (_R, -_R),
                    (-_R, _R),
                    (-_R, -_R),
                    (_R, 0.0),
                    (-_R, 0.0),
                    (0.0, _R),
                    (0.0, -_R),
                    (0.0, 0.0),
                ]
                ox, oy = 0.0, 0.0
                for dx, dy in cands:
                    ci, cj = _ng.world_to_cell(hx + dx, hy + dy)
                    if _ng.is_free(ci, cj):
                        ox, oy = dx, dy
                        break
                cam["tpv_off"] = (ox, oy)
            _base = np.array([hx, hy, 2.2])
            _eye = np.array([hx + ox, hy + oy, 2.2])
            if _ng is not None:
                import nav_planner

                _eye = nav_planner.push_eye_out(_eye, _base, _ng, min_fwd_frac=0.2)
            # The near-overhead eye can still land INSIDE tall furniture (a kitchen island/counter/cabinet
            # reaching ~2.2 m) whose overhang the FLOOR nav grid does not mark -- the "buried in dark wood"
            # TPV frames during a placement off to the side of a galley. Test the eye against the real 3D
            # furniture AABBs and LIFT it just above the tallest box it is inside (capped below the ~3 m roof
            # so it does not go black). The human is directly below, so lifting preserves the line of sight.
            _bx = cam.get("_solidboxes")
            if _bx is None:
                try:
                    _bx = cam["_solidboxes"] = scene.solid_boxes()
                except Exception:  # noqa: BLE001
                    _bx = cam["_solidboxes"] = []
            _lift = None
            for _nm, _mn, _mx in _bx:
                if (
                    float(_mn[0]) - 0.05 <= float(_eye[0]) <= float(_mx[0]) + 0.05
                    and float(_mn[1]) - 0.05 <= float(_eye[1]) <= float(_mx[1]) + 0.05
                    and float(_eye[2]) <= float(_mx[2]) + 0.10
                ):
                    _lift = (
                        float(_mx[2]) if _lift is None else max(_lift, float(_mx[2]))
                    )
            if _lift is not None:
                _eye[2] = min(_lift + 0.30, 2.9)
            smplh_camera.drive_tpv(
                tcop, (float(_eye[0]), float(_eye[1]), float(_eye[2])), (hx, hy, 1.0)
            )
        elif (
            tpv_eye is not None
        ):  # fixed cam (sit): a front-elevated view of the seated body
            smplh_camera.drive_tpv(tcop, tpv_eye, tpv_look)
        else:  # side-follow cam (openclose): tracks the human's XY
            root = np.array([fr.xy[0], fr.xy[1], 1.2])
            smplh_camera.drive_tpv(tcop, (root[0] + 3.4, root[1] + 0.6, 1.7), tpv_look)

    offline_frame_cache = []
    if not fdp:
        cache_nudge_state = {"nav_grid": nudge_state["nav_grid"]}
        cache_openable_state = init_openable_state
        cache_camera_state = _ego_camera.INITIAL_CAMERA_STATE
        for frame_index, frame in enumerate(frames):
            replay_controllers = _controllers_at_openable_state(
                controllers, cache_openable_state
            )
            nudge_boxes = _nudge_boxes_for(
                scene, replay_controllers, nudge_static, frame
            )
            cache_nudge_state["camera_boxes"] = nudge_boxes
            cache_nudge_state["frame_index"] = frame_index
            vertices, joints, object_transform = wpp.realize(
                rig,
                frame,
                scene_boxes=_body_nudge_boxes(scene, nudge_boxes),
                nudge_state=cache_nudge_state,
            )
            next_openable_state, consumed_openable_object = _step_openable_state(
                controllers, frame, cache_openable_state
            )
            if next_openable_state != cache_openable_state:
                render_controllers = _controllers_at_openable_state(
                    controllers, next_openable_state
                )
                render_boxes = _nudge_boxes_for(
                    scene, render_controllers, nudge_static, frame
                )
            else:
                render_boxes = nudge_boxes
            frame_adapter = ego_adapter.with_solid_boxes(render_boxes)
            camera_frame = frame_adapter.camera_frame(
                frame,
                joints,
                object_transform,
                dt_seconds=1.0 / FPS,
            )
            camera_step = _ego_camera.step_camera(
                camera_frame,
                cache_camera_state,
                framing=frame_adapter.framing,
            )
            cache_camera_state = camera_step.state
            cache_openable_state = next_openable_state
            offline_frame_cache.append(
                {
                    "vertices": vertices,
                    "joints": joints,
                    "object_transform": object_transform,
                    "render_boxes": render_boxes,
                    "openable_state": dict(next_openable_state),
                    "consumed_openable_object": bool(consumed_openable_object),
                    "camera_output": camera_step.output,
                }
            )
    drive_cams(frames[0], J_init, obj_init, init_boxes)

    # group a set of demos under fpv_outputs/<subdir>/ for review
    out, tag = _out_and_tag(args)
    # Write the honest per-episode verdict manifest. Gated on --seed (the validated path); compliance is
    # None when post_action_check is off -> verdict = init only.
    if getattr(args, "seed", None) is not None:
        camera_outputs = (
            [row["camera_output"] for row in offline_frame_cache]
            if len(offline_frame_cache) == len(frames)
            else None
        )
        episode_extra = _episode_extra(
            args,
            scene,
            env,
            goals,
            frames,
            n_fail,
            _t_start,
            "render",
            "render",
            camera_outputs,
        )
        _write_episode_manifest(
            out,
            tag,
            args,
            render_episode_valid,
            compliance,
            task_goal,
            extra=episode_extra,
        )
        label_contract = episode_extra.get("label_visibility") or {}
        if label_contract.get("required") and not label_contract.get("contract_ok"):
            print(
                "[step3r] LABEL CONTRACT FAIL: "
                f"missing={label_contract.get('missing')} "
                f"unreadable={label_contract.get('unreadable')}"
            )
            _exit_process(2)
            return 2
    sparse_capture = None
    capture_indices = None
    capture_plan_path = str(getattr(args, "capture_plan", "") or "")
    if capture_plan_path:
        with open(os.path.abspath(capture_plan_path), encoding="utf-8") as handle:
            raw_capture_plan = json.load(handle)
        if not isinstance(raw_capture_plan, dict):
            raise ValueError("--capture-plan must contain a JSON object")
        expected_task_id = str(args.scene).removeprefix("spec:")
        if str(raw_capture_plan.get("runtime_task_id") or "") != expected_task_id:
            raise ValueError("--capture-plan runtime task does not match --scene")
        sparse_capture = resolve_capture_plan(
            raw_capture_plan,
            command_spans,
            total_frames=len(frames),
        )
        capture_indices = frozenset(sparse_capture["capture_indices"])
        print(
            f"[sparse] selected {len(capture_indices)}/{len(frames)} unique FPV frames "
            f"for {len(sparse_capture['windows'])} HoloAssist decisions"
        )
    elif getattr(args, "sparse_stills_only", False):
        raise ValueError("--sparse-stills-only requires --capture-plan")
    proactive_queries_by_frame = _proactive_queries_by_frame(sparse_capture)
    proactive_visibility_samples: dict[str, list[dict]] = {}
    # Not gated on --seed: unlike the manifest this is a diagnostic artifact, and a non-seeded episode
    # is exactly as worth re-judging offline.
    _write_replay_dump(out, tag, args, scene, frames, _jtraj)
    # NO PHYSICS VETO. The render used to refuse any episode the compliance pass did not certify perfectly
    # clean, so a single sampled frame of a leg brushing furniture produced NO VIDEO AT ALL for an episode
    # that had solved its task. This benchmark grades reasoning: the picture quality is now the GRADED
    # render_grade in the manifest, and every episode is rendered and scored whatever that grade says.
    # Read through the SAME accessor --check gates on, so the two paths quote one verdict.
    _rp, _rg, _rr = action_check.render_verdict(compliance)
    if compliance is not None and not _rp:
        print(
            f"[step3r] RENDER GRADE {_rg.upper()} (rendering anyway): {_rr} -- "
            f"{compliance['human_worst_frac_desc']} "
            f"(hit {compliance['human_hit_frames']}/{compliance['human_sampled']} sampled frames "
            f"{compliance['human_by_seg']}). task_goal is scored independently."
        )
    # Option A (multi-object render fix): restore every graspable to its captured SPAWN pose (taken before
    # the per-command settle). Every final-pose consumer has already run (post-action compliance, the task-
    # goal read, the manifest), so this changes ONLY what the render draws: each object sits at its start
    # until its own grab/put frames drive it -- no "already placed" idle object, no flicker-back-to-start.
    import fpv_scene as _fs_restore

    _restore_dynamic_prims(stage, fdp_initial_state)
    for _sp, (_spos, _squat) in _obj_spawn.items():
        _sprim = stage.GetPrimAtPath(_sp)
        if _sprim is not None and _sprim.IsValid():
            _fs_restore.set_world_pose(_sprim, _spos, _squat)
    _place_cut_score_marks(stage, scene)
    openable_replay_state = init_openable_state
    _apply_openable_state(controllers, openable_replay_state)
    fpv_dir = os.path.join(out, f"{tag}_frames")
    tpv_dir = os.path.join(out, f"{tag}_tpv_frames")
    render_runtime._clear_frame_pngs(fpv_dir, tpv_dir)
    render_settings = configure_fpv_render_settings()
    rp_f, rp_t, a_f, a_t = fpv_render.setup_products(
        "/World/FpvCam", "/World/TpvCam", rcfg, W, Hh
    )
    timeline = fpv_core.start_timeline(rcfg.playback_fps)
    app_cfg.warmup_frames = max(int(app_cfg.warmup_frames), 400)
    fpv_render.settle_and_validate(app, timeline, a_f, app_cfg, annot_tpv=a_t)
    calibration_frame = make_start_frame(_c)
    if calibration_frame is None:
        raise RuntimeError("cannot author FPV calibration body without a start frame")
    calibration_joints = author_fpv_calibration_body(_c, calibration_frame)
    _autoexp = configure_fpv_lighting(
        app,
        stage,
        rcfg,
        cop,
        a_f,
        scene,
        eye_z=float(calibration_joints[wpp.H.HEAD_J][2]),
        scene_id=str(args.scene),
    )
    render_metadata = None
    if getattr(args, "seed", None) is not None:
        render_metadata = {"render_settings": render_settings}
        if _autoexp is not None:
            render_metadata["autoexp"] = _autoexp
        _merge_manifest_extra(out, tag, render_metadata)

    ctx = SegmentCtx(
        a_f,
        a_t,
        fpv_dir,
        tpv_dir,
        rcfg,
        1.0 / FPS,
        W,
        Hh,
    )

    n = len(frames)
    pour_stream_schedule = _pour_stream_schedule(frames)
    cut_score_schedule = _cut_score_schedule(
        frames,
        goals,
        compiled_commands,
        command_spans,
    )
    trace_module = None
    trace_session = None
    trace_frame_state = {}
    trace_aperture = None
    if os.environ.get("ISAAC_RENDER_TRACE_PATH"):
        from simulator.physical_world import render_parity_integration as trace_module

        if not authoritative_command_spans:
            raise RuntimeError(
                "renderer parity tracing requires authoritative source-command spans"
            )
        if len(frames) != len(source_frames):
            raise RuntimeError(
                "renderer parity tracing requires the full source trajectory; "
                "a frame-limited prefix can depend on an untraced command suffix"
            )
        trace_session = trace_module.maybe_create_trace_session(
            frames,
            renderer="offline",
            start_index=0,
            command_spans=command_spans,
        )
        trace_aperture = trace_module.camera_horizontal_aperture(cop)

    # Render the FULL program: lift the render frame budget to the actual program length so a long
    # multi-object / open-close task is never truncated mid-render. `max_frames` was a fixed 600 ceiling
    # that cut long programs off "halfway" (run_segment stops at gidx >= max_frames). Derived from n ->
    # general, no magic length; encode_mp4 encodes every rendered frame, so the mp4 gets the whole program.
    if n > rcfg.max_frames:
        rcfg.max_frames = n

    # FRAME-SHARDING (MAST multi-GPU): T = n is the FULL deterministic frame count. This process
    # captures ONLY the contiguous global range [shard_lo, shard_hi); the render loop still ADVANCES
    # every frame (single clock + physics identical -> captured frames byte-identical to a full render),
    # it just skips the PNG write outside the range (see run_segment cap_lo/cap_hi). FLOOR-TILING:
    # shard i's hi == shard i+1's lo, so N shards tile [0,T) with no gap/overlap, and each PNG's GLOBAL
    # index name merges the shards into one contiguous sequence in one dir. N=1 => [0,T) => capture all.
    shard_n = max(1, int(getattr(args, "num_shards", 1)))
    shard_i = int(getattr(args, "shard_idx", 0))
    if not (0 <= shard_i < shard_n):
        raise SystemExit(
            f"[shard] --shard-idx {shard_i} out of range for --num-shards {shard_n} (need 0 <= idx < N)"
        )
    shard_lo = (shard_i * n) // shard_n
    shard_hi = ((shard_i + 1) * n) // shard_n
    print(f"[shard] {shard_i}/{shard_n} -> frames [{shard_lo},{shard_hi}) of {n}")

    clutter_replay = clutter_render_replay.make_clutter_render_replay(_c)
    visual_state_replay = {}
    proactive_visibility_state = {"solid_boxes": ()}
    trace_persistent_paths = (
        trace_module.persistent_scene_object_paths(
            scene,
            clutter_replay.active_paths,
            trace_module.openable_controller_paths(controllers),
        )
        if trace_module is not None
        else ()
    )

    def on_frame(t, gidx):
        nonlocal openable_replay_state
        frame_index = min(int(round(t * (n - 1))), n - 1)
        fr = frames[frame_index]
        cached_frame = offline_frame_cache[gidx] if offline_frame_cache else None
        if cached_frame is None:
            replay_controllers = _controllers_at_openable_state(
                controllers, openable_replay_state
            )
            nudge_boxes = _nudge_boxes_for(scene, replay_controllers, nudge_static, fr)
            nudge_state["camera_boxes"] = nudge_boxes
            nudge_state["frame_index"] = gidx
            V, J, obj_T = wpp.realize(
                rig,
                fr,
                scene_boxes=_body_nudge_boxes(scene, nudge_boxes),
                nudge_state=nudge_state,
            )
            next_openable_state, consumed_openable_object = _step_openable_state(
                controllers, fr, openable_replay_state
            )
            if next_openable_state != openable_replay_state:
                render_controllers = _controllers_at_openable_state(
                    controllers, next_openable_state
                )
                render_boxes = _nudge_boxes_for(
                    scene, render_controllers, nudge_static, fr
                )
            else:
                render_boxes = nudge_boxes
            camera_output = None
        else:
            V = cached_frame["vertices"]
            J = cached_frame["joints"]
            obj_T = cached_frame["object_transform"]
            next_openable_state = cached_frame["openable_state"]
            consumed_openable_object = cached_frame["consumed_openable_object"]
            render_boxes = cached_frame["render_boxes"]
            camera_output = cached_frame["camera_output"]
        _drive_toggle_indicators(scene, fr)
        _drive_liquid_visuals(scene, fr, pour_stream_schedule[frame_index])
        _drive_workstation_screens(scene, fr)
        core.update_body(rig, V)
        released_path = (
            dynamic_replay.before_frame(
                fr, J, object_frame=not consumed_openable_object
            )
            if dynamic_replay is not None
            else None
        )
        physx_owned = bool(
            dynamic_replay is not None
            and dynamic_replay.owns_frame_object(
                fr, object_frame=not consumed_openable_object
            )
        )
        if (
            dynamic_replay is not None
            and cam.get("free_focus_path") != dynamic_replay.focus_path
        ):
            cam["free_focus"] = None
            cam["free_focus_path"] = dynamic_replay.focus_path
        clutter_replay.apply(gidx)
        _apply_visual_state_events(
            stage, visual_state_event_schedule, gidx, visual_state_replay
        )
        # P3 RELEASE: the FIRST free_dynamic frame for the carried object un-freezes /World/Obj (the exact
        # reverse of render_runtime's grasp flip) and hands it to PhysX. realize already returns obj_T=None
        # for this frame, so set_object_world below naturally stops writing it; it enters the read-only
        # free registry. Because a kinematic body carries no velocity and its last carried pose == the
        # penetrating target, the flip introduces no jump -- PhysX then de-penetrates + settles it.
        if released_path is not None:
            print(
                f"[step3r] RELEASE {released_path} -> DYNAMIC at gidx={gidx} "
                "(PhysX owns it now)"
            )
        openable_replay_state = next_openable_state
        _apply_openable_state(controllers, openable_replay_state)
        # Drive the graspable /World/Obj ONLY on frames NOT consumed by a drawer -- else the cup gets
        # written onto the drawer handle during the open (the "cup floats/teleports" bug). obj_T is None
        # once the object is released (free_dynamic), so this naturally stops writing it post-release.
        _require_multi_object_carry_identity(scene, fr)
        if (
            has_obj
            and obj_T is not None
            and not consumed_openable_object
            and not physx_owned
        ):
            _op = getattr(fr, "obj_path", None)
            if _op:
                # MULTI-graspable: point the write at THIS frame's object prim (refetch re-acquires it).
                rig._obj_path = _op
                core.set_object_world(rig, obj_T)
            elif not getattr(scene, "grasp_objs", None):
                # Single-object scene: obj_path is always None and /World/Obj is the only graspable -> write it
                # every frame (rig._obj_path stays /World/Obj), byte-identical to before.
                core.set_object_world(rig, obj_T)
            # else: a MULTI-object frame with NO obj_path is an EMPTY-HANDED walk between objects. Its obj is a
            # stale ("rest", prev_drop) that would be written to the STALE rig._obj_path (the previously-placed
            # object), teleporting it onto the last drop point -- a piled object lands exactly on the one below
            # it and visually vanishes. Skip the write: every object stays kinematically at its own last
            # correct pose (its placement frames already wrote it; idle ones sit at their Option-A spawn).
            # Feature 2: contents riding the carried container (a bowl's food) move as ONE rigid body -- write
            # each at obj_T @ rel (obj_T = the container's world this frame), then restore rig._obj_path so the
            # next frame's main write is unaffected.
            _decal_pose_debug(stage, rig, fr, obj_T)
            _carried = getattr(fr, "carried", ()) or ()
            for _cp, _rel in _carried:
                rig._obj_path = _cp
                core.set_object_world(rig, obj_T @ _rel)
            if _carried and _op:
                rig._obj_path = _op
        _apply_visual_state_pose_transfers(
            stage,
            scene,
            visual_state_event_schedule,
            gidx,
            visual_state_replay,
        )
        _drive_cut_score_marks(scene, cut_score_schedule[frame_index])
        drive_cams(
            fr,
            J,
            obj_T,
            render_boxes,
            physx_owned=physx_owned,
            camera_output=camera_output,
        )
        proactive_visibility_state["solid_boxes"] = render_boxes
        if trace_session is not None:
            trace_frame_state.update(
                {
                    "frame": fr,
                    "vertices": V,
                    "joints": J,
                    "object_transform": obj_T,
                    "solid_boxes": render_boxes,
                }
            )

    def on_post_step(t, gidx):
        # P3: read back each PhysX-owned free object's SIMULATED pose (valid only after the single
        # advance_once, which run_segment has just taken). Feeds the camera focus so the FPV frames the
        # object as it de-penetrates + settles, and traces min-separation to the blocker for the CSV log.
        fr = frames[min(int(round(t * (n - 1))), n - 1)]
        positions = dynamic_replay.after_step()
        focus = dynamic_replay.focus_from(positions)
        if focus is not None:
            cam["free_focus"] = np.asarray(focus, dtype=np.float64)
            cam["free_focus_path"] = dynamic_replay.focus_path
        print(
            f"[step3r][POST] gidx={gidx} seg={fr.seg} free={{"
            + ", ".join(
                f"{path.rsplit('/', 1)[-1]}:z={float(position[2]):.3f}"
                for path, position in positions.items()
            )
            + "}"
        )

    def _trace_snapshot(*, delivered: bool):
        if trace_module is None or trace_session is None:
            raise RuntimeError("renderer parity trace is not enabled")
        fr = trace_frame_state["frame"]
        paths = trace_module.object_paths_for_frame(
            fr,
            rig,
            fdp_paths,
            trace_persistent_paths,
        )
        owned_paths = (
            dynamic_replay.released_paths if dynamic_replay is not None else ()
        )
        pose = (
            cam.get("last_delivered_pose") if delivered else cam.get("last_render_pose")
        )
        return trace_module.build_render_state_snapshot(
            frame=fr,
            camera_pose=pose,
            width=W,
            height=Hh,
            horizontal_aperture=trace_aperture,
            body_vertices=trace_frame_state["vertices"],
            joints=trace_frame_state["joints"],
            object_transform=trace_frame_state["object_transform"],
            object_world_transforms=trace_module.snapshot_world_transforms(
                stage, paths
            ),
            openable_state=dict(openable_replay_state),
            solid_boxes=trace_frame_state["solid_boxes"],
            dynamic_state=trace_module.snapshot_dynamic_state(
                stage, fdp_paths, owned_paths
            ),
            renderer="offline",
        )

    if trace_session is not None:
        ctx.on_authored_pre_step = lambda gidx: trace_session.authored_pre_step(
            gidx,
            trace_frame_state["frame"],
            _trace_snapshot(delivered=False),
        )
        ctx.on_post_positive_step = (
            lambda gidx, zero_dt: trace_session.post_positive_step(
                gidx,
                trace_frame_state["frame"],
                _trace_snapshot(delivered=False),
                zero_dt=zero_dt,
            )
        )
        ctx.on_delivered_readback = (
            lambda gidx, rgb, readback, zero_dt: trace_session.delivered_readback(
                gidx,
                trace_frame_state["frame"],
                _trace_snapshot(delivered=True),
                readback=readback,
                rgb=rgb,
                zero_dt=zero_dt,
            )
        )
    if proactive_queries_by_frame:
        trace_readback = ctx.on_delivered_readback

        def _record_readback(gidx, rgb, readback, zero_dt):
            if trace_readback is not None:
                trace_readback(gidx, rgb, readback, zero_dt)
            _record_proactive_visibility(
                proactive_queries_by_frame,
                proactive_visibility_samples,
                gidx,
                scene,
                cam.get("last_delivered_pose"),
                proactive_visibility_state["solid_boxes"],
                W,
                Hh,
            )

        ctx.on_delivered_readback = _record_readback

    # Reset before the loop: the first loop frame seeds the smoother (un-smoothed), then it caps.
    cam["ego_state"] = _ego_camera.INITIAL_CAMERA_STATE
    cam["ego_yaw_clamps"] = 0
    cam["ego_pitch_clamps"] = 0
    cam["last_delivered_pose"] = None
    # FRAME-SHARDING: pass the capture window. N==1 -> cap_hi=-1 (no bound) + cap_tpv=True == capture
    # EVERY frame + BOTH products == byte-identical to today. N>1 -> capture only [shard_lo, shard_hi),
    # optionally dropping the TPV product (--shard-no-tpv).
    _shard_active = shard_n > 1
    if capture_indices is not None and _shard_active:
        raise ValueError("--capture-plan cannot be combined with frame sharding")
    # SHARD-FAST (opt-in REAL speedup): by default every shard runs advance_once (rep.orchestrator.step
    # == the RTX raytracing) for ALL T frames just to stay byte-identical, so N shards cost ~N x the
    # raytracing and sharding buys NO wall-clock win (README blocker #4). --shard-fast skips advance_once
    # OUTSIDE this shard's window so each shard raytraces only ~T/N frames -> the real speedup. It is
    # SOUND ONLY when there is no per-frame physics coupling: with live free-dynamic PhysX (fdp truthy)
    # each frame's object pose integrates from the prior advance_once, so skipping would DESYNC the sim
    # -> REFUSE (warn + fall back to the full advance, no speedup). Also requires N>1 (nothing to skip at
    # N==1). on_frame still runs for every frame, so the cheap kinematic/camera/nudge state stays
    # deterministic at the window start; ONLY the RTX step is elided.
    # *** BYTE-IDENTICAL boundary-frame verification REQUIRED before trusting --shard-fast (GPU was busy
    #     at impl time): the RTX temporal denoiser carries cross-frame history, so the FIRST captured
    #     frames of a fast shard may differ from a full render. Diff a shard's boundary PNGs against a
    #     full render before use. This path was NOT runtime-verified. ***
    _shard_fast = bool(_shard_active and getattr(args, "shard_fast", False))
    if _shard_fast and trace_session is not None:
        print(
            "[shard] --shard-fast REFUSED: renderer parity tracing requires the complete "
            "authored -> positive-step -> readback lifecycle for every dense frame."
        )
        _shard_fast = False
    if _shard_fast and fdp:
        print(
            "[shard] --shard-fast REFUSED: scene has LIVE per-frame free-dynamic physics "
            f"(free_dynamic_prims={fdp_paths}); skipping advance_once outside the window would desync "
            "the simulation. Falling back to the FULL advance (every frame raytraced -> NO speedup, "
            "byte-identical to a normal shard)."
        )
        _shard_fast = False
    if _shard_fast:
        print(
            f"[shard] --shard-fast ACTIVE: raytracing ONLY frames [{shard_lo},{shard_hi}) of {n}; "
            "advance_once SKIPPED elsewhere. *** BYTE-IDENTICAL boundary-frame verification REQUIRED "
            "before trusting these frames (RTX denoiser temporal history differs at the window start). ***"
        )
    _sparse_fast = capture_indices is not None
    if _sparse_fast and trace_session is not None:
        print(
            "[sparse] fast skipping disabled because renderer parity tracing requires every frame"
        )
        _sparse_fast = False
    if _sparse_fast and fdp:
        print(
            "[sparse] live free-dynamic physics detected; advancing every frame for correctness "
            "while writing only selected FPV frames"
        )
        _sparse_fast = False
    if _sparse_fast:
        print(
            f"[sparse] fast capture active: raytracing {len(capture_indices)}/{n} "
            "selected frames with temporal refresh after skipped gaps"
        )
    _cap_tpv = not getattr(args, "shard_no_tpv", False)
    if sparse_capture is not None:
        sparse_capture["execution_mode"] = (
            "sparse_raytrace" if _sparse_fast else "dense_safe_sparse_write"
        )
        sparse_capture["raytraced_frame_count"] = (
            len(capture_indices) if _sparse_fast else n
        )
        if render_metadata is not None:
            render_metadata["sparse_capture"] = sparse_capture
            _merge_manifest_extra(out, tag, render_metadata)
    trace_finished = False
    try:
        run_segment(
            ctx,
            f"step3_{args.scene}",
            n,
            on_frame,
            0,
            rcfg.max_frames,
            on_post_step=(on_post_step if fdp else None),
            cap_lo=shard_lo,
            cap_hi=(shard_hi if _shard_active else -1),
            cap_tpv=_cap_tpv,
            fast_skip=(_shard_fast or _sparse_fast),
            capture_indices=capture_indices,
            sparse_refresh_frames=(2 if _sparse_fast else 0),
        )
        if trace_session is not None:
            trace_session.finish()
        trace_finished = True
    finally:
        if trace_session is not None and not trace_finished:
            trace_session.abort()
        fpv_render.teardown([rp_f, rp_t], [a_f, a_t])
    if fdp and _bt and _gp:
        try:
            task_goal = _task_goal_record(
                _bt,
                _gp,
                scene.relations(source="physics"),
                scoring_baseline_relations,
            )
            print(
                f"[step3r][GOAL][post-render] {_bt['task']}: "
                f"check_goal={bool(task_goal['met'])} "
                f"progress={float(task_goal['progress']):.2f}"
            )
            if getattr(args, "seed", None) is not None:
                _write_episode_manifest(
                    out,
                    tag,
                    args,
                    render_episode_valid,
                    compliance,
                    task_goal,
                    extra=_episode_extra(
                        args,
                        scene,
                        env,
                        goals,
                        frames,
                        n_fail,
                        _t_start,
                        "render",
                        "render",
                    ),
                )
                if render_metadata is not None:
                    _merge_manifest_extra(out, tag, render_metadata)
        except Exception as _ge:  # noqa: BLE001
            print(f"[step3r][GOAL][post-render] {_bt['task']}: goal-check error {_ge}")
    print(
        f"[camera] frames={n} physiological_yaw_clamps={cam['ego_yaw_clamps']} "
        f"physiological_pitch_clamps={cam['ego_pitch_clamps']} "
        f"clearance_frames={cam['clearance_frames']} "
        f"clearance_max_shift_mm={cam['clearance_max_shift_m'] * 1000.0:.0f} "
        f"fpv_readback_retries={ctx.fpv_retries}"
    )

    if sparse_capture is not None:
        if proactive_queries_by_frame:
            frame_observations = _proactive_frame_observations(
                sparse_capture, proactive_visibility_samples
            )
            sparse_capture["frame_observation_count"] = len(frame_observations)
            if render_metadata is None:
                raise ValueError(
                    "phase-aware sparse capture requires a seeded manifest"
                )
            render_metadata["frame_observations"] = frame_observations
        if getattr(args, "sparse_stills_only", False):
            sparse_capture["compatibility_timeline"] = None
        else:
            compatibility = materialize_compatibility_sequence(
                fpv_dir,
                total_frames=n,
                capture_indices=sparse_capture["capture_indices"],
            )
            sparse_capture["compatibility_timeline"] = compatibility
        if render_metadata is not None:
            render_metadata["sparse_capture"] = sparse_capture
            _merge_manifest_extra(out, tag, render_metadata)
        if not getattr(args, "sparse_stills_only", False):
            print(
                f"[sparse] materialized {n}-frame compatibility timeline "
                f"({compatibility['linked']} hard links, {compatibility['copied']} copies)"
            )

    if _shard_active:
        # SHARD MODE: this process wrote only PNGs [shard_lo, shard_hi) (GLOBAL-indexed). Do NOT encode an
        # mp4 here -- a shard holds a from-nonzero subset, so ffmpeg's frame_%05d.png demux (which starts at
        # 00000) would be wrong. The merge step (all shards' PNGs collected into one dir) runs the final
        # ffmpeg concat/encode. The frames dir is this shard's artifact to sync.
        print(
            f"[step3r] DONE (shard {shard_i}/{shard_n}) -> {fpv_dir} frames [{shard_lo},{shard_hi}) of {n}; "
            f"mp4 encode deferred to the shard-merge step"
        )
    elif getattr(args, "sparse_stills_only", False):
        print(
            f"[step3r] DONE -> {fpv_dir} "
            f"({len(capture_indices or ())} sparse FPV stills; video encode skipped)"
        )
    else:
        fpv_render.encode_mp4(fpv_dir, os.path.join(out, f"{tag}_fpv.mp4"), rcfg)
        if _cap_tpv:
            fpv_render.encode_mp4(tpv_dir, os.path.join(out, f"{tag}_tpv.mp4"), rcfg)
        products = f"{tag}_fpv.mp4" + (f" + {tag}_tpv.mp4" if _cap_tpv else "")
        print(f"[step3r] DONE -> {out}/{products} ({n} timeline frames)")
    _exit_process(0)
    return 0


def nonnegative_frame_limit(value: str) -> int:
    frame_limit = int(value)
    if frame_limit < 0:
        raise argparse.ArgumentTypeError("--frames must be non-negative")
    return frame_limit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene",
        default="openclose",
        choices=sorted(
            _scene_builders()
        ),  # single source of truth: the registry keys (incl bwgp_*)
    )
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--subdir",
        default=None,
        help="group outputs (mp4s + manifest + frames) under fpv_outputs/<subdir>/ for review",
    )
    ap.add_argument(
        "--allow-dirty",
        dest="allow_dirty",
        action="store_true",
        help="NO-OP, accepted for compatibility with the MAST node scripts that still pass it. There is "
        "no physics veto any more: every episode renders and the manifest carries the graded "
        "render_grade (clean/marginal/dirty) plus the independent task_goal.",
    )
    ap.add_argument(
        "--frames",
        type=nonnegative_frame_limit,
        default=0,
        help="cap total frames (0 = all)",
    )
    ap.add_argument(
        "--trajectory-in",
        default="",
        help="strict Frame trajectory artifact to render instead of the locally planned motion. "
        "Used for same-trajectory offline/model-facing parity validation.",
    )
    ap.add_argument(
        "--capture-plan",
        default="",
        help="HoloAssist sparse-capture plan; renders the union of evaluator frame indices and "
        "materializes a timeline-compatible FPV video.",
    )
    ap.add_argument(
        "--sparse-stills-only",
        action="store_true",
        help="with --capture-plan, write only selected FPV PNGs and skip compatibility-video encoding",
    )
    ap.add_argument(
        "--active-gpu",
        type=int,
        default=0,
        dest="active_gpu",
        help="physical GPU index for the renderer + PhysX (AppCfg.active_gpu/physics_gpu). Lets N "
        "workers share one multi-GPU node. Leave CUDA_VISIBLE_DEVICES UNSET -- remapping it breaks "
        "Isaac's renderer<->CUDA interop. Default 0 == today's behaviour.",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        dest="num_shards",
        help="FRAME-SHARDING (MAST multi-GPU): split the deterministic frame list (length T, computed "
        "AFTER --frames) into N contiguous shards. This process renders ONLY shard --shard-idx: every "
        "frame is still ADVANCED (single render clock + physics stay identical -> captured frames are "
        "byte-identical to a full render), but frames outside this shard's global range are NOT captured "
        "(no PNG). Captured PNGs keep their GLOBAL index (frame_<global>.png) so N shards merge into one "
        "contiguous sequence in one dir. Under sharding the per-shard mp4 encode is DEFERRED to the merge "
        "step (a shard holds a from-nonzero subset). N=1 (default) = byte-identical to today.",
    )
    ap.add_argument(
        "--shard-idx",
        type=int,
        default=0,
        dest="shard_idx",
        help="which shard in [0, --num-shards) this process renders (see --num-shards).",
    )
    ap.add_argument(
        "--shard-no-tpv",
        action="store_true",
        dest="shard_no_tpv",
        help="skip the TPV output (used by frame sharding and sparse HoloAssist capture); the FPV "
        "product remains the model-facing artifact.",
    )
    ap.add_argument(
        "--shard-fast",
        action="store_true",
        dest="shard_fast",
        help="REAL sharding speedup: SKIP advance_once (rep.orchestrator.step == the RTX raytracing) for "
        "frames OUTSIDE this shard's [shard_lo,shard_hi) window, so a shard raytraces only ~T/N frames "
        "instead of all T. Without this every shard runs the full raytracing for all T frames (README "
        "blocker #4: sharding gives NO wall-clock win). REFUSED (falls back to the full advance) when the "
        "scene has live per-frame free-dynamic physics (free_dynamic_prims set) -- skipping would desync "
        "the sim. Ignored when --num-shards==1. *** BYTE-IDENTICAL boundary-frame verification REQUIRED "
        "before trusting this: the RTX temporal denoiser has cross-frame history, so a fast shard's first "
        "captured frames may differ from a full render (GPU was busy at impl time -> NOT verified). ***",
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="snapshot the built scene from a few fixed cameras (labeled containers) + exit (no full episode)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="build scene + execute + gate on reach/primitives/settle/init and the SAME graded picture "
        "verdict the render publishes (action_check.render_verdict), then skip the RTX render. Fast "
        "pre-render pre-flight; it cannot FAIL an episode the render path calls clean.",
    )
    ap.add_argument(
        "--physcheck",
        action="store_true",
        help="GATE: headless free-fall probe (no RTX) -- drops a YCB prop 0.5m under substepped "
        "orchestrator.step and asserts the fall matches analytic 0.5*g*t^2 (proves N substeps deliver "
        "so the during-render free-dynamic path is sound). Runs BEFORE any scene build.",
    )
    ap.add_argument(
        "--initcheck",
        action="store_true",
        help="INIT non-penetration harness (no RTX): for --seeds, free-cell place the human + validate "
        "every action stand pose against the EXACT human-vs-scene gate; assert 0 penetrating pairs (or "
        "episode_valid=False). Runs BEFORE any render.",
    )
    ap.add_argument(
        "--seeds",
        default="0,1,2",
        help="comma-separated seeds for --initcheck (default 0,1,2)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RANDOM-INIT RENDER: build furniture colliders + free-cell place the human + validate every "
        "stand pose against the human-vs-scene gate, GATE on episode_valid (a penetrating init is never "
        "rendered -- aborts with exit 3), then render the VALIDATED init. Omit to render the hand-tuned "
        "scene at its hardcoded start (byte-identical to before).",
    )
    ap.add_argument(
        "--fallback-check",
        action="store_true",
        help="P4 FALLBACK proof (no RTX): force an over-cluttered/unplaceable init (tiny region on a "
        "furniture footprint) + assert place_start degrades to episode_valid=False (least-penetrating "
        "accepted + flagged), not a crash or a silently-accepted penetrating frame.",
    )
    ap.add_argument(
        "--natural-grasp",
        action="store_true",
        help="NATURAL GRASP: force scene.live_wrap=True for ANY scene so PICK builds the object SDF and "
        "the per-shape grip prototype (CYL/ROUND/PINCH) seeds the live finger wrap (instead of the canned "
        "atlas pose). Scenes that set live_wrap themselves are on by default; omit to keep others canned.",
    )
    ap.add_argument(
        "--navcheck",
        action="store_true",
        help="NAV-GRID doorway/connectivity probe (no RTX): build the occupancy + two-sided-SDF nav grid "
        "(furniture occupancy + doorway-preserving wall-triangle raster), dump PNG/ascii to fpv_outputs/, "
        "and assert the human start -> every action stand pose is connected through free space (doorways "
        "open). Runs BEFORE any render.",
    )
    ap.add_argument(
        "--no-doors",
        action="store_true",
        help="DISABLE the door-swing feature for a BEHAVIOR scene (default: ON). With it, a closed 'door' "
        "asset is treated as a plain nav obstacle -> the goal behind it is unroutable -> straight-line "
        "fallback (the pre-door behaviour). Default (feature ON) auto-opens + traverses closed doors.",
    )
    ap.add_argument(
        "--door-open",
        action="store_true",
        help="NAVCHECK proof mode: probe connectivity on the DOORS-PASSABLE grid view (door cells freed) "
        "to show a closed-door goal that is BLOCKED on the raw occ becomes CONNECTED once the door can be "
        "opened. Pair with --navcheck.",
    )
    ap.add_argument(
        "--close-doors-behind",
        action="store_true",
        help="opt-in: after passing through an auto-opened door, walk to the far side and CloseDoor it "
        "(default: leave it open -- harmless, already through).",
    )
    args = ap.parse_args()
    return run_render(args)

"""Load baked BEHAVIOR-1K assets into the InteractiveWorld physical layer.

Asset locations are configured by ``INTERACTIVEWORLD_BEHAVIOR_CACHE`` and
``INTERACTIVEWORLD_BEHAVIOR_DATA``. Importing this module does not initialize
OmniGibson or Isaac Sim.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass

from simulator.taskpack_support.bundled_assets import generated_label_cache

CACHE_ROOT = os.environ.get(
    "INTERACTIVEWORLD_BEHAVIOR_CACHE", "/opt/interactiveworld/behavior-cache"
)
CACHE_USD = os.path.join(CACHE_ROOT, "usd")
CACHE_SPEC = os.path.join(CACHE_ROOT, "spec")
GENERATED_LABEL_DIR = str(generated_label_cache())


def _split(ref: str) -> tuple[str, str]:
    """ "category/model" -> (category, model)."""
    if "/" not in ref:
        raise ValueError(f"BEHAVIOR ref must be 'category/model', got {ref!r}")
    cat, model = ref.split("/", 1)
    return cat, model


def baked_usd(ref: str) -> str:
    """Absolute path to a baked plain USD; raise a clear error if it was not baked yet (run P1)."""
    cat, model = _split(ref)
    p = os.path.join(CACHE_USD, cat, f"{model}.usd")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"BEHAVIOR asset not present in INTERACTIVEWORLD_BEHAVIOR_CACHE: {ref}"
        )
    return p


def read_spec(ref: str) -> dict:
    """The P1 spec.json (role_hint, bbox_size, mass, openable_joint_ids, joints, ...)."""
    cat, model = _split(ref)
    p = os.path.join(CACHE_SPEC, cat, f"{model}.json")
    with open(p) as f:
        return json.load(f)


def _is_baked(ref: str) -> bool:
    """True when ``category/model`` has a baked USD in the cache (no exception, no spec read)."""
    cat, model = _split(ref)
    return os.path.exists(os.path.join(CACHE_USD, cat, f"{model}.usd"))


class BehaviorBuildError(RuntimeError):
    """Base for a NAMED scene-build failure, so an offline sweep can record WHY a scene was rejected
    instead of a bare RuntimeError string."""


class MissingStructuralAsset(BehaviorBuildError):
    """A scene referenced a ``walls``/``floors`` model that is not in the bake cache."""


class NoWorkSurface(BehaviorBuildError):
    """A scene has no comfortable-height work surface to stage graspables on."""


class NoContainerFound(BehaviorBuildError):
    """A scene has no prim with a usable interior to drop an object INTO."""


class NoGraspableAsset(BehaviorBuildError):
    """None of the requested library graspables is baked in this cache."""


class ContainerPredicateUnavailable(BehaviorBuildError):
    """The world-catalog scene manifest would not import, so the shared container family predicate
    is unreachable and no drop target can be identified.

    Unlike the room lookup this cannot degrade: a local fallback copy is exactly the divergence this
    import exists to end, and returning "no containers" would make every scene fail as
    ``NoContainerFound`` -- a named lie."""


# Categories whose absence is FATAL rather than degrading. Everything else missing costs one object; these
# two cost the world: with no floors there is no dz reference, and with no walls the scene has no nav
# geometry and no occluder at all (walls are the environment's only modeled occluder), so the
# visibility gate silently passes on everything and the episode scores 0 with nothing in the log.
_STRUCTURAL_CATS = ("walls", "floors")


def _resolve_baked(ref: str, cat: str, missing: list) -> str | None:
    """``baked_usd(ref)``, or None after recording ``ref`` in ``missing`` when the bake is absent.

    A missing STRUCTURAL bake raises: a scene that renders furniture floating in a void is worse than a
    scene that fails to build, because the void renders, scores 0 and never appears in any log."""
    try:
        return baked_usd(ref)
    except FileNotFoundError as exc:
        if cat in _STRUCTURAL_CATS:
            raise MissingStructuralAsset(
                f"structural asset not baked: {ref} (category {cat!r}). A scene missing its {cat} "
                "renders as a void -- no nav geometry and no occluder. Install the frozen "
                "asset bundle before running this task."
            ) from exc
        missing.append(ref)
        return None


def _record_missing_bakes(scene, missing: list, where: str) -> None:
    """Expose the per-build unbaked-model tally on the scene and WARN once, loudly.

    ``scene.behavior_missing_bakes`` (int) is set even when nothing was skipped, so a caller can always
    assert on it; ``scene.behavior_missing_refs`` carries the distinct ``cat/model`` ids."""
    refs = tuple(sorted(set(missing)))
    scene.behavior_missing_bakes = len(missing)
    scene.behavior_missing_refs = refs
    if not refs:
        return
    head = ", ".join(refs[:6]) + (" ..." if len(refs) > 6 else "")
    print(
        f"[behavior-import] WARNING {where}: SKIPPED {len(missing)} unbaked model(s) "
        f"({len(refs)} distinct): {head}. Install the frozen asset bundle before evaluation."
    )


BEHAVIOR_DATA_ROOT = os.environ.get(
    "INTERACTIVEWORLD_BEHAVIOR_DATA", "/opt/interactiveworld/behavior-data"
)
SCENE_ROOT = os.path.join(BEHAVIOR_DATA_ROOT, "behavior-1k-assets", "scenes")

# Categories to SKIP when loading a scene for a clean FPV grab demo: ceilings (occlude the dome light ->
# black void), thin/decorative clutter, and glass we don't interact with. Everything else (floors, walls,
# furniture) loads. This keeps a real BEHAVIOR layout without the tiny-object / occluder noise.
_SKIP_CATS = {
    "ceilings",
    "electric_switch",
    "picture",
    "photograph",
    "mirror",
    "wall_clock",
    "openable_window",
    "window",
    "curtain",
    "blind",
    "shade",
    "lighting",
    "downlight",
}


def _skip_scene_object(o, skip_names: set[str], include_categories: set[str]) -> bool:
    return o["name"] in skip_names or (
        o["cat"] in _SKIP_CATS and o["cat"] not in include_categories
    )


# Categories that have a horizontal top but are NOT valid pick/place work surfaces (closed storage whose
# lid is not a real staging shelf, plumbing fixtures, appliances). Keeps the D1 support pick off a closed
# cabinet lid / toilet / stove. Matched as substrings of the category name.
_SURF_EXCLUDE = (
    "toilet",
    "sink",
    "bathtub",
    "shower",
    "stove",
    "oven",
    "washer",
    "dryer",
    "microwave",
    "fridge",
    "refrigerator",
    "dishwasher",
    "hamper",
    "urinal",
    "bidet",
    "range_hood",
)


# Gate for the door-swing feature (OpenDoor/CloseDoor + nav auto-insert). Default ON for BEHAVIOR scenes:
# the execute() pre-pass only inserts an OpenDoor when a walk actually crosses a CLOSED door, so a doorless
# path is unchanged, and a closed door on the path is opened + traversed automatically (no [OpenDoor] in the
# action string). episode_runtime sets it from --no-doors (disable -> closed door becomes a plain obstacle ->
# straight-line fallback). When ON, _compose_behavior_scene records scene.door_prims (nav door layer) and
# registers each revolute 'door' as a Role.OPENABLE with hinge fields + a DoorInfo. v1 scope = category
# 'door' (revolute-Z hinge). Other families (sliding_door prismatic / garage_door revolute-Y / elevator_door
# no-joint) are deferred: they keep hinge_xy=None -> obstacle -> straight-line fallback (no regression).
# Non-BEHAVIOR scenes never call _compose_behavior_scene, so this flag never affects them.
ENABLE_DOORS = True

# Door categories handled by the swing feature (v1 = 'door'). Recorded into scene.door_prims when ENABLE.
_DOOR_CATS = ("door",)

# USE_TASK_LAYOUT: place the challenge task's objects + human start EXACTLY at the pre-sampled instance
# positions (NO heuristic re-staging). This fixes the "objects re-staged onto a globally-clearest surface
# that lands in a nav component disconnected from the robot start" failure (the #1 cause of unreachable
# tasks). The init physics-validation (place_start/validate_stand_pose snap the human to the nearest valid
# pose; object settle snaps each object onto its real support) is the safety net for slightly-off samples.
# True = real-layout (default, desired); False = legacy approachability heuristic (rollback / byte-identical
# comparison for the original 5 demos). Gated so flipping it back restores the exact prior behaviour.
USE_TASK_LAYOUT = True


def _safe(name: str) -> str:
    """USD-safe prim name."""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


def _quat_wxyz(ori_xyzw):
    x, y, z, w = ori_xyzw
    return (float(w), float(x), float(y), float(z))


def _in_rooms(args: dict) -> list[str]:
    """The dataset's room annotation for one object, e.g. ``["kitchen_0"]``.

    Normalised to a list of str: a few scenes store a bare string, and an unannotated object has the
    key absent or null. Never None, so callers can filter with a plain ``in`` test.
    """
    rooms = args.get("in_rooms") or []
    if isinstance(rooms, str):
        rooms = [rooms]
    return [str(r) for r in rooms]


def scene_objects(scene_name: str) -> list:
    """Parse a scene *_best.json -> list of dicts (name/cat/model/pos/ori/scale/fixed_base/visual_only/
    in_rooms) for every DatasetObject (skips the embedded robot). pos/ori come from state.registry.root_link.

    ``in_rooms`` is the dataset's own room annotation (e.g. ``["kitchen_0"]``); it is normalised to a list
    of strings because a few scenes store a bare string. Empty list when the object carries no annotation.
    """
    with open(
        os.path.join(SCENE_ROOT, scene_name, "json", f"{scene_name}_best.json")
    ) as f:
        d = json.load(f)
    init = d["objects_info"]["init_info"]
    reg = d["state"]["registry"]["object_registry"]
    out = []
    for name, info in init.items():
        if not isinstance(info, dict) or info.get("class_name") != "DatasetObject":
            continue
        a = info.get("args", {})
        cat, model = a.get("category"), a.get("model")
        rl = reg.get(name, {}).get("root_link", {})
        pos, ori = rl.get("pos"), rl.get("ori")
        if not (cat and model and pos and ori):
            continue
        sc = a.get("scale", [1.0, 1.0, 1.0])
        if isinstance(sc, (int, float)):
            sc = [sc, sc, sc]
        out.append(
            {
                "name": name,
                "cat": cat,
                "model": model,
                "pos": pos,
                "ori": ori,
                "scale": sc,
                "fixed_base": a.get("fixed_base", True),
                "visual_only": a.get("visual_only", False),
                "in_rooms": _in_rooms(a),
            }
        )
    return out


# 2026 BEHAVIOR challenge pre-sampled task instances (a NORMAL filesystem). Each task ships a scene JSON
# in the SAME objects_info + state.registry schema as <scene>_best.json, PLUS a metadata.task block
# (inst_to_name: BDDL synset-instance -> concrete object name; robot_poses: the human start).
CHALLENGE_ROOT = os.path.join(
    BEHAVIOR_DATA_ROOT, "2026-challenge-task-instances", "scenes"
)

# Per-task placement overrides (world x,y; the template z is KEPT) that relocate task objects into the
# nav-CONNECTED open region, so the human navigates through the doorway to reach them instead of straight-
# lining through a wall to a stand walled-off inside tight kitchen furniture. Each entry is verified
# reachable via `--navcheck`. Applied BEFORE the scene compose (so the container prim is placed at the
# override and the graspables register/settle there). A task with no entry uses the sampled positions.
_TASK_POS_OVERRIDE = {
    # (empty) build_behavior_task now places task objects via the PROVEN build_behavior_wgp approachability
    # heuristic (clearest reachable work surface + distinct reachable spot), which is nav-clean by
    # construction, so per-task manual position overrides are no longer needed. Kept as a hook for a future
    # task that needs a manual nudge.
}

# Generic interior fill for any layout whose retained roof blocks the dome above declared room ceilings.
_INTERIOR_FILL_LIGHT = {
    "spacing_m": 3.5,
    "radius": 0.25,
    "intensity": 60000.0,
    "below_ceiling_m": 0.20,
    "color": (1.0, 0.97, 0.92),
}


def _interior_fill_config(by_name) -> dict | None:
    categories = {
        str(obj.get("cat", "")).strip().lower().rstrip("s") for obj in by_name.values()
    }
    if "roof" not in categories or "ceiling" not in categories:
        return None
    return dict(_INTERIOR_FILL_LIGHT)


def scene_objects_task(scene_name: str, task_name: str) -> tuple:
    """Parse a 2026-challenge task-instance template -> (objects, inst_to_name, robot_start, joints).

    ``objects``: the SAME per-object dicts scene_objects returns (name/cat/model/pos/ori/scale/...),
    parsed from the identical objects_info.init_info + state.registry.object_registry schema (objects are
    already placed to satisfy the task :init). ``inst_to_name``: BDDL synset-instance -> concrete object
    name (metadata.task.inst_to_name). ``robot_start``: {"pos","ori"} from metadata.task.robot_poses (the
    task human start; the object_registry robot entry is parked off-scene and must NOT be used).
    ``joints``: concrete name -> joint_pos list (for articulated objects, e.g. a trash-can lid)."""
    p = os.path.join(
        CHALLENGE_ROOT,
        scene_name,
        "json",
        f"{scene_name}_task_{task_name}_0_0_template.json",
    )
    with open(p) as f:
        d = json.load(f)
    init = d["objects_info"]["init_info"]
    reg = d["state"]["registry"]["object_registry"]
    out = []
    joints: dict = {}
    for name, info in init.items():
        if not isinstance(info, dict) or info.get("class_name") != "DatasetObject":
            continue
        a = info.get("args", {})
        cat, model = a.get("category"), a.get("model")
        rl = reg.get(name, {}).get("root_link", {})
        pos, ori = rl.get("pos"), rl.get("ori")
        if not (cat and model and pos and ori):
            continue
        sc = a.get("scale", [1.0, 1.0, 1.0])
        if isinstance(sc, (int, float)):
            sc = [sc, sc, sc]
        out.append(
            {
                "name": name,
                "cat": cat,
                "model": model,
                "pos": pos,
                "ori": ori,
                "scale": sc,
                "fixed_base": a.get("fixed_base", True),
                "visual_only": a.get("visual_only", False),
                "in_rooms": _in_rooms(a),
            }
        )
        jp = reg.get(name, {}).get("joint_pos")
        if jp:
            joints[name] = jp
    task = d.get("metadata", {}).get("task", {})
    i2n = task.get("inst_to_name", {}) or {}
    rp = (task.get("robot_poses", {}) or {}).get("robot", [{}])
    robot_start = rp[0] if rp else {}
    return out, i2n, robot_start, joints


def build_behavior(
    stage,
    app,
    scene_name,
    grab="apple/agveuv",
    grab_family="round_wrap",
    grab_seat="ball",
):
    """Deserialize a curated BEHAVIOR scene into the EXISTING step3 harness and return its 5-tuple
    (scene, controllers, program, tpv_look, tpv_eye). Real BEHAVIOR furniture at real scene poses,
    shifted so the floor top = z 0 (the gait/reach floor convention); ceilings + tiny clutter skipped.
    The grabbed object is the rig /World/Obj (a baked BEHAVIOR graspable), placed on the largest table;
    a second surface is the [Put] target. The human start + every stand pose are validated by the
    existing --seed init gate (MANDATORY for BEHAVIOR). NEVER edits existing modules."""
    import fpv_scene
    import numpy as np
    from scene_state import Role, SceneState

    objs = scene_objects(scene_name)
    # PASS 1: reference every curated object as a DIRECT /World/<name> child (so init_validation's
    # per-object footprint walk sees each with its own collider owner) at its raw scene pose.
    placed = {}  # name -> (prim_path, dict)
    missing: list = []  # unbaked refs: counted + warned; structural ones raise
    for o in objs:
        if o["cat"] in _SKIP_CATS:
            continue
        usd = _resolve_baked(f"{o['cat']}/{o['model']}", o["cat"], missing)
        if usd is None:
            continue
        pp = f"/World/b_{_safe(o['name'])}"
        prim = fpv_scene.add_reference(stage, pp, usd)
        fpv_scene.set_world_pose(
            prim, tuple(o["pos"]), _quat_wxyz(o["ori"]), tuple(o["scale"])
        )
        placed[o["name"]] = (pp, o)

    # floor top -> shift everything so the walkable floor is at z=0 (matches gait foot-pin / reach).
    floor_top = None
    for name, (pp, o) in placed.items():
        if o["cat"] == "floors":
            rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(pp))
            ft = float(rng.GetMax()[2])
            floor_top = ft if floor_top is None else max(floor_top, ft)
    dz = -float(floor_top) if floor_top is not None else 0.0

    # PASS 2: re-pose every object shifted by dz (floor top -> 0).
    for name, (pp, o) in placed.items():
        p = o["pos"]
        fpv_scene.set_world_pose(
            stage.GetPrimAtPath(pp),
            (float(p[0]), float(p[1]), float(p[2]) + dz),
            _quat_wxyz(o["ori"]),
            tuple(o["scale"]),
        )

    # stage=stage -> LIVE grounding (register_from_prim + object_pos/surface_top read the real USD bbox).
    scene = SceneState(floor_z=0.0, start_xy=(0.0, 0.0), stage=stage)
    _record_missing_bakes(scene, missing, f"build_behavior({scene_name})")
    scene.behavior_grab = {"obj": grab, "family": grab_family, "seat": grab_seat}

    # Register furniture as obstacles (surfaces/seats); skip floors/walls from solid_boxes but keep them
    # as colliders (walls bound the room; floor is the ground). register_from_prim reads the live bbox.
    scene.floor_prims = []
    scene.wall_prims = []
    tables = []  # (name, prim_path, top_z, (cx,cy), footprint area)
    for name, (pp, o) in placed.items():
        cat = o["cat"]
        if cat in ("floors", "walls"):
            if cat == "floors":
                scene.floor_prims.append(pp)
            else:
                scene.wall_prims.append(pp)
            continue  # colliders applied by _apply_scene_colliders; not scene_state obstacles
        role = (
            Role.SEAT
            if any(
                k in cat
                for k in ("chair", "sofa", "couch", "bench", "stool", "armchair")
            )
            else Role.SURFACE
        )
        e = scene.register_from_prim(name, role, pp, is_obstacle=True)
        # A pickable/placeable surface = ANY furniture top at reachable table height with a real
        # horizontal footprint (category-agnostic -> robust across scene types). Excludes thin
        # wall fixtures, the floor, and tiny objects.
        if role is Role.SURFACE:
            (cx, cy), top = scene.surface_top(name)
            hx, hy = e.half
            # A pick/place surface at COMFORTABLE reach height (low coffee tables ~0.4m fail the reach-DOWN;
            # too-high fails too) with a real top. Table DEPTH is handled by seating the object at the NEAR
            # EDGE below, so a deep table is fine -- no footprint cap needed.
            if 0.62 <= top <= 1.15 and (hx * hy) >= 0.06:
                tables.append((name, pp, top, (cx, cy), (hx, hy), hx * hy))

    tables.sort(
        key=lambda t: -t[5]
    )  # the largest (most prominent/stable) comfortable-height surface
    if not tables:
        raise RuntimeError(
            f"build_behavior({scene_name}): no comfortable-height (0.62-1.05m) table found"
        )
    pick = tables[0]
    (pcx, pcy), ptop, (phx, phy) = pick[3], pick[2], pick[4]
    # Seat the apple at the table NEAR EDGE along the approach so BOTH the grab and the put reach are
    # ~clearance (well inside the reach envelope) regardless of table depth. foot_xy/foot_half = the FULL
    # table footprint so the human stands CLEAR of the near face. GRASPABLE stays SPEC-grounded (no
    # prim_path): /World/Obj is not built until AFTER init validation.
    # Approach the table from the ROOM INTERIOR (toward the scene centroid) so the human stands on the
    # OPEN side, not against a wall. yaw from the centroid->table direction; the near edge is on that side.
    pts = np.array(
        [[o["pos"][0], o["pos"][1]] for (_pp, o) in placed.values()], dtype=np.float64
    )
    centroid = pts.mean(axis=0) if len(pts) else np.array([pcx, pcy])
    d = np.array([float(pcx), float(pcy)]) - centroid
    fwd_v = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([0.0, -1.0])
    approach_yaw = math.degrees(math.atan2(float(fwd_v[0]), -float(fwd_v[1])))
    fwd = (float(fwd_v[0]), float(fwd_v[1]))
    half_depth = abs(fwd[0]) * phx + abs(fwd[1]) * phy
    # Seat the apple just inside the table's NEAR EDGE along the approach, using the RAY-BOX edge
    # distance min(hx/|fwdx|, hy/|fwdy|) -- the true center->edge distance along fwd. (The old
    # half_depth = |fwd.foot_half| is the PROJECTED extent, which OVERSHOOTS a long/diagonal table and
    # lands the apple off it -> floating; the table CENTER is unreachable from outside a big table ->
    # init penetration.) foot anchored at this near-edge point + a small foot_half so stand_pose_for
    # stands the human ~0.55 m in front of the near edge (clear of the table, reach inside the envelope);
    # the full-footprint TABLE entity (registered above) still supplies the solid_box the nudge avoids.
    edge_dist = min(phx / max(abs(fwd[0]), 1e-6), phy / max(abs(fwd[1]), 1e-6))
    d_in = max(0.05, edge_dist - 0.10)  # just inside the near edge (on the tabletop)
    ax = float(pcx) - fwd[0] * d_in
    ay = float(pcy) - fwd[1] * d_in
    scene.register(
        "apple",
        Role.GRASPABLE,
        xy=(ax, ay),
        z=float(ptop) + 0.05,
        approach_yaw=approach_yaw,
        foot_xy=(ax, ay),
        foot_half=(0.14, 0.14),
        top_z=float(ptop),
        clearance=0.40,
    )
    # PUT target = a SPEC "putspot" at the SAME near-edge (put the apple back down there). SPEC (no
    # prim_path) so surface_top returns this near-edge xy -- NOT the far table-center (which is out of
    # reach). foot = the table footprint so the put stand pose also clears the table.
    scene.register(
        "putspot",
        Role.SURFACE,
        xy=(ax, ay),
        z=float(ptop),
        half=(0.16, 0.16),
        approach_yaw=approach_yaw,
        foot_xy=(float(pcx), float(pcy)),
        foot_half=(float(phx), float(phy)),
        top_z=float(ptop),
        clearance=0.40,
    )
    scene.behavior_pick_xy = (ax, ay, float(ptop))
    scene.behavior_place = "putspot"
    # The pick table needs an EXACT static collider so the graspable settles ON it (the hardcoded
    # settle-support tuple does not know BEHAVIOR prim paths).
    scene.behavior_support_prims = [pick[1]]

    # Start the human NEAR the table (short walk within the lit room) instead of the most-open cell across
    # the whole house (a long cross-house walk clips walls + hits dark/unlit areas -> black FPV frames).
    scene.start_xy = (
        float(pcx) - fwd[0] * (half_depth + 2.3),
        float(pcy) - fwd[1] * (half_depth + 2.3),
    )

    # BEHAVIOR house interiors are DIM (walls occlude the dome light) -> add a bright local light over the
    # action table so the pick/carry/place reads (the enclosed-ceiling relight pattern from the GRAB office).
    from pxr import UsdLux

    _lt = UsdLux.SphereLight.Define(stage, "/World/BehaviorActionLight")
    _lt.CreateRadiusAttr(0.6)
    _lt.CreateIntensityAttr(120000.0)
    _lt.CreateColorAttr((1.0, 1.0, 1.0))
    fpv_scene.set_world_pose(_lt.GetPrim(), (float(pcx), float(pcy), float(ptop) + 1.9))
    # Register the OmniGibson MDL material dir on the RTX MDL search path so the BEHAVIOR VRay materials
    # resolve (else surfaces render as a flat red fallback -- "Unable to find SdrShaderNode OmniGibsonVRayMtl").
    try:
        import carb.settings

        _root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )  # -> isaacsim_try
        _mdl_dir = os.path.join(
            _root,
            "migration_eval",
            "OmniGibson",
            "OmniGibson",
            "omnigibson",
            "materials",
        )
        _s = carb.settings.get_settings()
        for _key in (
            "/rtx/mdltranslator/mdlPaths",
            "/renderer/mdl/searchPaths/templates",
        ):
            _cur = _s.get(_key) or []
            if isinstance(_cur, str):
                _cur = [_cur]
            if _mdl_dir not in _cur:
                _s.set(_key, list(_cur) + [_mdl_dir])
        print(f"[behavior] registered MDL search path: {_mdl_dir}")
    except Exception as _exc:  # noqa: BLE001
        print(
            f"[behavior] MDL path registration failed ({_exc}); materials may render red"
        )

    # v1 demo = walk -> grab -> lift (a clean "人拿东西", like highreach). The [Put]-back-on-table is a
    # refinement (the released kinematic object currently rests on the floor, not the table top -- a
    # produce_place rest-pose issue to fix); grab+carry are critic-clean, so ship grab-only first.
    program = "[Grab] <apple> (1)"
    print(
        f"[behavior] {scene_name}: placed {len(placed)} objs, floor_top={floor_top} dz={dz:.3f}; "
        f"pick={pick[0]}@({pcx:.2f},{pcy:.2f},{ptop:.2f}) apple@edge=({ax:.2f},{ay:.2f}) put=putspot; grab={grab}"
    )
    # TPV (QA) = a fixed elevated 3/4 view of the action table (tpv_look must be non-None). The eye is
    # offset on the interior/approach side + up so the human walks into frame and the pick/place is seen.
    tpv_look = (float(pcx), float(pcy), float(ptop) + 0.10)
    tpv_eye = (float(pcx) - fwd[0] * 3.2, float(pcy) - fwd[1] * 3.2, 2.2)
    return scene, {}, program, tpv_look, tpv_eye


def behavior_build_params(
    ref: str,
    scale: float | tuple[float, float, float] | list[float] = 1.0,
) -> dict:
    """``build_params`` for ``core.build_object`` -- reference the baked BEHAVIOR USD as ``/World/Obj/m``.

    BEHAVIOR object USDs are authored in real-world METERS (a mug bbox is ~0.09-0.12 m), so scale=1.0
    is correct; ``scale`` is passed through for the rare resized asset and may be a scalar or an
    explicit ``(x, y, z)``. The seat (obj_T_in_wrist) is supplied by the CALLER (reuse a proven
    family seat for a similar object; the live SDF wrap then conforms the fingers to this real
    mesh)."""
    if isinstance(scale, (list, tuple)):
        if len(scale) != 3:
            raise ValueError(
                f"asset scale must be a scalar or [x, y, z], got {scale!r}"
            )
        serialized_scale: float | tuple[float, float, float] = tuple(
            float(value) for value in scale
        )
    else:
        serialized_scale = float(scale)
    return {
        "shape": "usd_ref",
        "usd": baked_usd(ref),
        "behavior": ref,
        "scale": serialized_scale,
    }


def _register_mdl() -> None:
    """Register the OmniGibson VRay MDL dir on the RTX search path (else BEHAVIOR surfaces render flat
    red). Best-effort; the same block build_behavior runs inline."""
    try:
        import carb.settings

        _root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )  # -> isaacsim_try
        _mdl_dir = os.path.join(
            _root,
            "migration_eval",
            "OmniGibson",
            "OmniGibson",
            "omnigibson",
            "materials",
        )
        _s = carb.settings.get_settings()
        for _key in (
            "/rtx/mdltranslator/mdlPaths",
            "/renderer/mdl/searchPaths/templates",
        ):
            _cur = _s.get(_key) or []
            if isinstance(_cur, str):
                _cur = [_cur]
            if _mdl_dir not in _cur:
                _s.set(_key, list(_cur) + [_mdl_dir])
        print(f"[behavior] registered MDL search path: {_mdl_dir}")
    except Exception as _exc:  # noqa: BLE001
        print(
            f"[behavior] MDL path registration failed ({_exc}); materials may render red"
        )


def _register_door(scene, name, pp, o, prim_index) -> bool:
    """Register a revolute 'door' as a Role.OPENABLE swing door + a door_planner.DoorInfo. Returns True on
    success, False if the model has no revolute-Z hinge (a deferred family -> caller registers it as a plain
    obstacle). Geometry is read LIVE from the placed door prim: the swinging leaf's sub-prim ORIGIN is the
    hinge and its local Z is world-up (verified by door_probe), so the render just rotates the leaf about
    its own local Z. The handle (grip contact) is the free edge of the leaf; the door stays an OBSTACLE
    (foot_half set) so occ blocks the closed leaf."""
    import numpy as np
    from pxr import Usd, UsdGeom
    from scene_state import Role

    try:
        spec = read_spec(f"{o['cat']}/{o['model']}")
    except Exception:  # noqa: BLE001
        return False
    hinge = None
    for j in spec.get("joints", []):
        if (
            j.get("type") == "revolute"
            and j.get("axis") == "Z"
            and str(j.get("body0", "")).rstrip("/").endswith("base_link")
        ):
            hinge = j
            break
    if hinge is None:
        return False  # sliding_door (prismatic) / garage_door (axis Y) / no joint -> deferred
    leaf_name = str(hinge["body1"]).rstrip("/").split("/")[-1]
    droot = scene.stage.GetPrimAtPath(pp)
    leaf = None
    for p in Usd.PrimRange(droot, Usd.TraverseInstanceProxies()):
        if p.GetName() == leaf_name:
            leaf = p
            break
    if leaf is None:
        return False

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m = xc.GetLocalToWorldTransform(leaf)
    hinge_w = m.ExtractTranslation()
    hinge_xy = (float(hinge_w[0]), float(hinge_w[1]))
    # leaf local +Y in world = the free-edge (handle) direction; local +X = the through/normal direction.
    yax = np.array([m[1][0], m[1][1]], dtype=np.float64)
    yn = float(np.linalg.norm(yax))
    yax = yax / yn if yn > 1e-6 else np.array([1.0, 0.0])
    xax = np.array([m[0][0], m[0][1]], dtype=np.float64)
    xn = float(np.linalg.norm(xax))
    normal_xy = xax / xn if xn > 1e-6 else np.array([0.0, 1.0])
    bb = spec.get("bbox_size") or [0.15, 0.8, 1.9]
    scl = o.get("scale", [1.0, 1.0, 1.0])
    sy = float(scl[1] if isinstance(scl, (list, tuple)) else scl)
    panel_w = float(bb[1]) * sy
    grip_z = scene.floor_z + 1.0
    handle_xy = (
        hinge_xy[0] + float(yax[0]) * panel_w * 0.9,
        hinge_xy[1] + float(yax[1]) * panel_w * 0.9,
    )
    swing_angle = math.radians(min(float(hinge.get("upper", 90.0)), 88.0))

    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    rng = bc.ComputeWorldBound(droot).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
    hx = max(0.5 * (mx[0] - mn[0]), 0.02)
    hy = max(0.5 * (mx[1] - mn[1]), 0.02)
    top_z = float(mx[2])
    # Face the door from the near side (approach along -normal). pose_body fwd = [sin(yaw), -cos(yaw)].
    approach_yaw = math.degrees(math.atan2(float(normal_xy[0]), -float(normal_xy[1])))

    scene.register(
        name,
        Role.OPENABLE,
        xy=handle_xy,  # object_pos(door) -> the grip handle (SPEC; no prim_path so it is NOT the bbox center)
        z=grip_z,
        approach_yaw=approach_yaw,
        hinge_xy=hinge_xy,
        hinge_axis=(0.0, 0.0, 1.0),
        swing_angle=swing_angle,
        swing_sign=1.0,  # resolved to the free side at render (_place_door), then written back
        leaf_paths=(str(leaf.GetPath()),),
        joint_name=hinge["name"],
        open_thresh=0.10,  # radians
        joint_val=0.0,
        foot_xy=(float(cx), float(cy)),
        foot_half=(float(hx), float(hy)),
        top_z=top_z,
        clearance=0.45,
        travel=0.0,  # not a prismatic out-swing; stand-back is footprint + clearance
    )
    stand_xy, stand_yaw = scene.stand_pose_for(name)
    from door_planner import DoorInfo

    scene.doors[name] = DoorInfo(
        entity_key=name,
        prim_index=int(prim_index),
        center_xy=(float(cx), float(cy)),
        normal_xy=(float(normal_xy[0]), float(normal_xy[1])),
        stand_xy=(float(stand_xy[0]), float(stand_xy[1])),
        stand_yaw=float(stand_yaw),
        handle_xy=(float(handle_xy[0]), float(handle_xy[1])),
    )
    return True


_HORIZONTAL_FRONT_DOOR_CATEGORIES = frozenset({"oven"})


def _container_revolute_joints(
    spec: dict, *, accept_blank_vertical: bool = False
) -> tuple[dict, ...]:
    """Return only revolute joints that describe a supported container door."""
    joints = tuple(spec.get("joints") or ())
    vertical_axes = {"Z", ""} if accept_blank_vertical else {"Z"}
    vertical = tuple(
        joint
        for joint in joints
        if joint.get("type") == "revolute"
        and joint.get("axis") in vertical_axes
        and str(joint.get("body0", "")).rstrip("/").endswith("base_link")
    )
    if vertical:
        return vertical
    if str(spec.get("category") or "") not in _HORIZONTAL_FRONT_DOOR_CATEGORIES:
        return ()
    declared = {
        str(row[-1])
        for row in (spec.get("openable_joint_ids") or ())
        if isinstance(row, (list, tuple)) and row
    }
    return tuple(
        joint
        for joint in joints
        if joint.get("type") == "revolute"
        and joint.get("axis") in ("X", "Y")
        and str(joint.get("name") or "") in declared
        and str(joint.get("body0", "")).rstrip("/").endswith("base_link")
    )


def _openable_leaf_fields(scene, pp, o) -> dict | None:
    """Openable-leaf fields for a placed BEHAVIOR container with a real drivable leaf, or None if it has
    none (bin/basket -> caller keeps the lid-hide fallback). Returns a dict a CONTAINER entity carries so
    the SAME open/close driver acts on its leaf when [Open]/[Close] target it:
      revolute-Z door  -> {"kind":"swing", hinge_xy, swing_angle, leaf_paths, joint_name, leaf_bbox}
      prismatic drawer -> {"kind":"slide", travel, leaf_paths, joint_name, leaf_bbox}
    (a container stays a solid nav obstacle; it is NOT a nav door). ``leaf_bbox`` is the primary leaf's
    world (mn,mx), while ``leaf_bboxes`` retains every measured leaf for deriving the complete front
    aperture. The world open ``axis`` for a drawer is resolved in the caller (toward the human stand)."""
    from pxr import Usd, UsdGeom

    if not o or not o.get("cat") or not o.get("model"):
        return None
    try:
        spec = read_spec(f"{o['cat']}/{o['model']}")
    except Exception:  # noqa: BLE001
        return None
    joints = spec.get("joints", []) or []
    revs = list(_container_revolute_joints(spec))
    pri = (
        None
        if revs
        else next((j for j in joints if j.get("type") == "prismatic"), None)
    )
    if not revs and pri is None:
        return None
    droot = scene.stage.GetPrimAtPath(pp)

    def _leaf_prim(jt):
        nm = str(jt["body1"]).rstrip("/").split("/")[-1]
        return next(
            (
                p
                for p in Usd.PrimRange(droot, Usd.TraverseInstanceProxies())
                if p.GetName() == nm
            ),
            None,
        )

    def _bbox(prim):
        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        lmn, lmx = rng.GetMin(), rng.GetMax()
        return (
            (float(lmn[0]), float(lmn[1]), float(lmn[2])),
            (float(lmx[0]), float(lmx[1]), float(lmx[2])),
        )

    if revs:
        # ALL revolute-Z leaves (a fridge/cabinet has 2 doors); the render driver swings EVERY leaf about
        # its own hinge so the unit opens in its natural shape, not just one door.
        leaves = [(_leaf_prim(j), j) for j in revs]
        leaves = [(lf, j) for lf, j in leaves if lf is not None]
        if not leaves:
            return None
        j0 = leaves[0][1]
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        leaf_origins = tuple(
            xform_cache.GetLocalToWorldTransform(leaf).ExtractTranslation()
            for leaf, _joint in leaves
        )
        front_flip = all(joint.get("axis") in ("X", "Y") for _, joint in leaves)
        leaf_hinges = tuple(
            (
                (float(origin[0]), float(origin[1]), float(origin[2]))
                if front_flip
                else (float(origin[0]), float(origin[1]))
            )
            for origin in leaf_origins
        )
        leaf_bboxes = tuple(_bbox(lf) for lf, _ in leaves)
        fields = {
            "kind": "swing",
            "hinge_xy": leaf_hinges[0][:2],
            "leaf_hinges": leaf_hinges,
            "swing_angle": math.radians(min(float(j0.get("upper", 90.0)), 88.0)),
            "leaf_paths": tuple(str(lf.GetPath()) for lf, _ in leaves),
            "joint_name": j0["name"],
            "leaf_bbox": leaf_bboxes[0],
            "leaf_bboxes": leaf_bboxes,
        }
        if front_flip:
            fields.update(front_flip=True, hinge_xyz=leaf_hinges[0])
        return fields
    leaf = _leaf_prim(pri)
    if leaf is None:
        return None
    return {
        "kind": "slide",
        "travel": float(min(max(float(pri.get("upper", 0.30)), 0.15), 0.45)),
        "leaf_paths": (str(leaf.GetPath()),),
        "joint_name": pri["name"],
        "leaf_bbox": _bbox(leaf),
    }


FRONT_DRUM_CATEGORIES = frozenset({"clothes_dryer", "washer"})


def is_front_drum_category(category: str) -> bool:
    return category in FRONT_DRUM_CATEGORIES


def front_opening_access_metadata(
    category: str,
    model: str,
    leaf_fields: dict,
    unit_center_xy: tuple[float, float],
) -> dict[str, object]:
    """Describe the measurable front aperture and its inward access direction."""
    meta: dict[str, object] = {
        "asset_category": category,
        "asset_model": model,
    }
    if leaf_fields.get("kind") != "swing" or not leaf_fields.get("leaf_bbox"):
        return meta
    boxes = tuple(leaf_fields.get("leaf_bboxes") or (leaf_fields["leaf_bbox"],))
    leaf_min = tuple(min(box[0][axis] for box in boxes) for axis in range(3))
    leaf_max = tuple(max(box[1][axis] for box in boxes) for axis in range(3))
    leaf_center = tuple(
        0.5 * (float(leaf_min[axis]) + float(leaf_max[axis])) for axis in range(3)
    )
    outward_x = leaf_center[0] - float(unit_center_xy[0])
    outward_y = leaf_center[1] - float(unit_center_xy[1])
    outward_norm = math.hypot(outward_x, outward_y)
    if outward_norm <= 1e-9:
        return meta
    entry_axis = (-outward_x / outward_norm, -outward_y / outward_norm)
    lateral_axis = (-entry_axis[1], entry_axis[0])
    lateral_values = [
        x * lateral_axis[0] + y * lateral_axis[1]
        for box in boxes
        for x in (float(box[0][0]), float(box[1][0]))
        for y in (float(box[0][1]), float(box[1][1]))
    ]
    aperture_width = max(lateral_values) - min(lateral_values)
    aperture_height = float(leaf_max[2]) - float(leaf_min[2])
    meta.update(
        {
            "access_mode": (
                "front_drum" if is_front_drum_category(category) else "front_opening"
            ),
            "entry_axis": entry_axis,
            "aperture_center": (
                leaf_center[0] + entry_axis[0] * 0.05,
                leaf_center[1] + entry_axis[1] * 0.05,
                leaf_center[2],
            ),
            "aperture_half_width": max(0.12, 0.45 * aperture_width),
            "aperture_half_height": max(0.12, 0.45 * aperture_height),
        }
    )
    return meta


def _compose_behavior_scene(
    stage,
    app,
    scene_name,
    grab,
    grab_family,
    grab_seat,
    objects=None,
    skip_names=None,
    include_categories=None,
):
    """Load a scene's baked BEHAVIOR furniture into /World, shift so the floor top = z 0, register every
    furniture prim as an obstacle, and collect the comfortable-height WORK SURFACES (excluding closed
    storage / plumbing / appliances / seats). Returns (scene, placed, surfaces, centroid, floor_top, dz).
    Additive: does NOT modify build_behavior, so the 3 legacy behavior scenes stay byte-identical."""
    import fpv_scene
    import numpy as np
    from scene_state import Role, SceneState

    placed = {}
    _objs = objects if objects is not None else scene_objects(scene_name)
    _skip = set(
        skip_names or ()
    )  # concrete names to EXCLUDE (task graspables built as rig objects)
    # Scene composition omits glass/decorative categories by default, but a TaskPack may bind one
    # of them as an interaction fixture.  In that case the explicitly required category must win
    # over the visual-cleanup skip list or the builder can never register the authored entity.
    _include_categories = {str(category) for category in (include_categories or ())}
    missing: list = []  # unbaked refs: counted + warned; structural ones raise
    for o in _objs:
        if _skip_scene_object(o, _skip, _include_categories):
            continue
        usd = _resolve_baked(f"{o['cat']}/{o['model']}", o["cat"], missing)
        if usd is None:
            continue
        pp = f"/World/b_{_safe(o['name'])}"
        prim = fpv_scene.add_reference(stage, pp, usd)
        fpv_scene.set_world_pose(
            prim, tuple(o["pos"]), _quat_wxyz(o["ori"]), tuple(o["scale"])
        )
        placed[o["name"]] = (pp, o)

    floor_top = None
    for _name, (pp, o) in placed.items():
        if o["cat"] == "floors":
            ft = float(
                fpv_scene.compute_world_bbox(stage.GetPrimAtPath(pp)).GetMax()[2]
            )
            floor_top = ft if floor_top is None else max(floor_top, ft)
    dz = -float(floor_top) if floor_top is not None else 0.0
    for _name, (pp, o) in placed.items():
        p = o["pos"]
        fpv_scene.set_world_pose(
            stage.GetPrimAtPath(pp),
            (float(p[0]), float(p[1]), float(p[2]) + dz),
            _quat_wxyz(o["ori"]),
            tuple(o["scale"]),
        )

    scene = SceneState(floor_z=0.0, start_xy=(0.0, 0.0), stage=stage)
    _record_missing_bakes(scene, missing, f"_compose_behavior_scene({scene_name})")
    # Enable the shape-adaptive live finger-wrap for ALL BEHAVIOR objects. Without this, produce_pick takes
    # the sd_world=None arm-only-reach path -> the fingers never close -> the hand stays in the canned open
    # pose and the object appears to float in an open hand. With it, produce_pick builds the object's SDF on
    # demand (render_runtime._make_sd_world on the real prim mesh) and solve_grasp(family_seed=True) closes
    # the MANO fingers onto the surface (grasp_select.classify_shape -> CYL/ROUND/PINCH). Safe: produce_pick
    # auto-falls back to the canned grip if the SDF build fails.
    scene.live_wrap = True
    if (
        grab is not None
    ):  # single-grab (build_behavior_wgp) sets it; a multi-graspable TASK leaves it unset
        scene.behavior_grab = {"obj": grab, "family": grab_family, "seat": grab_seat}
    # Wall prim paths for the navigation grid (doorway-preserving wall raster in nav_grid.py). Recorded
    # here but deliberately NOT registered as scene_state entities, so solid_boxes() + the reach gate stay
    # byte-identical; consumed only when scene.plan_nav is set.
    scene.floor_prims = []
    scene.wall_prims = []
    # Door layer (gated on ENABLE_DOORS; else empty/None -> byte-identical). door_prims feeds the nav-grid
    # door_id raster (nav_grid.build_nav_grid); scene.doors (name->DoorInfo) is populated in P4 for the
    # execute() pre-pass + the swing render. A closed door still registers as an OBSTACLE below, so occ
    # blocks it (hard backstop); the door layer is a SEPARATE channel.
    scene.door_prims = []
    scene.doors = {} if ENABLE_DOORS else None

    surfaces = []  # (name, prim_path, top_z, (cx,cy), (hx,hy), area)
    for name, (pp, o) in placed.items():
        cat = o["cat"]
        if cat in ("floors", "walls"):
            if cat == "floors":
                scene.floor_prims.append(pp)
            else:
                scene.wall_prims.append(pp)
            continue  # colliders applied by _apply_scene_colliders; not scene_state obstacles
        if ENABLE_DOORS and cat in _DOOR_CATS:
            # Register the revolute door as a swing OPENABLE (hinge fields + DoorInfo) and record its prim
            # for the nav door layer. The door stays an obstacle (foot_half) so occ blocks the closed leaf;
            # the door layer (door_id) + the execute() pre-pass open it on demand. A model with no
            # revolute-Z hinge (deferred family) falls through to the generic obstacle registration.
            if _register_door(scene, name, pp, o, len(scene.door_prims)):
                scene.door_prims.append(pp)
                continue
        is_seat = any(
            k in cat for k in ("chair", "sofa", "couch", "bench", "stool", "armchair")
        )
        role = Role.SEAT if is_seat else Role.SURFACE
        e = scene.register_from_prim(name, role, pp, is_obstacle=True)
        if (
            role is Role.SURFACE
            and cat != "bed"
            and not any(k in cat for k in _SURF_EXCLUDE)
        ):
            (cx, cy), top = scene.surface_top(name)
            hx, hy = e.half
            if 0.62 <= top <= 1.15 and (hx * hy) >= 0.06:
                surfaces.append(
                    (
                        name,
                        pp,
                        float(top),
                        (float(cx), float(cy)),
                        (float(hx), float(hy)),
                        float(hx * hy),
                    )
                )
    surfaces.sort(key=lambda t: -t[5])
    pts = np.array(
        [[o["pos"][0], o["pos"][1]] for (_pp, o) in placed.values()], dtype=np.float64
    )
    centroid = pts.mean(axis=0) if len(pts) else np.array([0.0, 0.0])
    return scene, placed, surfaces, centroid, floor_top, dz


def build_behavior_wgp(
    stage,
    app,
    scene_name,
    grab="apple/agveuv",
    grab_family="round_wrap",
    grab_seat="ball",
):
    """WALK -> GRAB a real BEHAVIOR object off one real table -> WALK to a SECOND real table -> PUT it
    down, inside a real BEHAVIOR scene. Additive sibling of build_behavior (which is grab-only); leaves
    it + the 3 legacy scenes byte-identical. MANDATE --seed (init non-pen gate). Returns the step3 5-tuple."""
    import math

    import fpv_scene
    import numpy as np
    from pxr import UsdLux
    from scene_state import Role

    scene, placed, surfaces, centroid, floor_top, dz = _compose_behavior_scene(
        stage, app, scene_name, grab, grab_family, grab_seat
    )
    if not surfaces:
        raise RuntimeError(
            f"build_behavior_wgp({scene_name}): no comfortable work surface found"
        )

    def _edge(table):
        _n, _pp, top, (cx, cy), (hx, hy), _a = table
        d = np.array([cx, cy], dtype=np.float64) - centroid
        nn = float(np.linalg.norm(d))
        fwd = d / nn if nn > 1e-6 else np.array([0.0, -1.0])
        yaw = math.degrees(math.atan2(float(fwd[0]), -float(fwd[1])))
        edge = min(
            hx / max(abs(float(fwd[0])), 1e-6), hy / max(abs(float(fwd[1])), 1e-6)
        )
        d_in = max(0.05, edge - 0.10)
        ex, ey = float(cx) - float(fwd[0]) * d_in, float(cy) - float(fwd[1]) * d_in
        return (
            yaw,
            (ex, ey),
            (float(fwd[0]), float(fwd[1])),
            (float(hx), float(hy)),
            float(top),
            (float(cx), float(cy)),
        )

    # Choose PICK + PUT for APPROACHABILITY: a table ringed by chairs or jammed against a wall has no
    # clean stand -> the init gate aborts the episode. Score each surface by how CLEAR its near-edge
    # stand cell is (fewest OTHER furniture boxes within 0.55 m). pick = clearest (tiebreak larger).
    boxes = (
        scene.solid_boxes()
    )  # [(name, mn, mx)] furniture footprints (graspable not yet registered)

    def _stand_xy(table):
        _y, edge, fwd, _hh, _t, _c = _edge(table)
        return (edge[0] - fwd[0] * 0.55, edge[1] - fwd[1] * 0.55)

    def _clutter(stand, self_name):
        c = 0
        for nm, mn, mx in boxes:
            if nm == self_name:
                continue
            dx = max(float(mn[0]) - stand[0], 0.0, stand[0] - float(mx[0]))
            dy = max(float(mn[1]) - stand[1], 0.0, stand[1] - float(mx[1]))
            if dx * dx + dy * dy < 0.55 * 0.55:
                c += 1
        return c

    scored = sorted(surfaces, key=lambda t: (_clutter(_stand_xy(t), t[0]), -t[5]))
    pick = scored[0]
    p_cx, p_cy = pick[3]
    # PUT = the clearest OTHER surface 0.8-3.5 m away (a short walk stays in one lit area -> far less FPV
    # wall-clipping than a cross-house trek); fallback = put back on the pick surface (single=True).
    cand = [s for s in scored[1:] if math.hypot(s[3][0] - p_cx, s[3][1] - p_cy) >= 0.8]
    if cand:
        put = min(
            cand,
            key=lambda s: (
                _clutter(_stand_xy(s), s[0]),
                math.hypot(s[3][0] - p_cx, s[3][1] - p_cy),
            ),
        )
        best = math.hypot(put[3][0] - p_cx, put[3][1] - p_cy)
        single = False
    else:
        put, best, single = pick, 0.0, True

    p_yaw, p_edge, p_fwd, (p_hx, p_hy), p_top, p_ctr = _edge(pick)
    q_yaw, q_edge, q_fwd, (q_hx, q_hy), q_top, q_ctr = _edge(put)

    # BEHAVIOR objects are CENTER-pivot: place the origin so the object BASE (not center) rests on the
    # put surface (else the released object sinks ~half its height into the table). min_z = local bbox
    # bottom from the spec (<= 0 for a center pivot); the put origin z = surface top - min_z.
    sp = read_spec(grab)
    bb = sp.get("bbox_size") or [0.1, 0.1, 0.1]
    blo = sp.get("base_link_offset") or [0.0, 0.0, 0.0]
    min_z = float(blo[2]) - 0.5 * float(bb[2])
    put_origin_z = float(q_top) - min_z

    scene.register(
        "apple",
        Role.GRASPABLE,
        xy=p_edge,
        z=float(p_top) + 0.05,
        approach_yaw=p_yaw,
        foot_xy=p_edge,
        foot_half=(0.14, 0.14),
        top_z=float(p_top),
        clearance=0.40,
    )
    scene.register(
        "putspot",
        Role.SURFACE,
        xy=q_edge,
        z=put_origin_z,
        half=(0.16, 0.16),
        approach_yaw=q_yaw,
        foot_xy=q_edge,
        foot_half=(0.14, 0.14),
        top_z=put_origin_z,
        clearance=0.40,
    )
    scene.behavior_support_prims = [pick[1]] if single else [pick[1], put[1]]

    # start near the PICK table (short first walk); the main walk is pick -> put.
    scene.near_start = True
    scene.start_xy = (p_edge[0] - p_fwd[0] * 2.2, p_edge[1] - p_fwd[1] * 2.2)
    # opt into navigation (path planning + wall anti-pen + FPV eye push-out). The init gate builds
    # scene.nav_grid ONLY if the validated start reaches every stand through free space; otherwise it
    # leaves it unset and the executor/camera fall back to the byte-identical straight-line behaviour.
    scene.plan_nav = True
    # HARMONY + VALIDATOR (Step 1 + Step 2): all systems read ONE precise collision map (SSOT oracle),
    # and every action is followed by a universal physics validate+correct on that same map.
    scene.unified_geom = True
    scene.post_action_check = True

    for _tag, ctr, top in (("Pick", p_ctr, p_top), ("Put", q_ctr, q_top)):
        lt = UsdLux.SphereLight.Define(stage, f"/World/BehaviorLight{_tag}")
        lt.CreateRadiusAttr(0.6)
        lt.CreateIntensityAttr(60000.0)
        lt.CreateColorAttr((1.0, 1.0, 1.0))
        fpv_scene.set_world_pose(
            lt.GetPrim(), (float(ctr[0]), float(ctr[1]), float(top) + 1.9)
        )

    # A dim dome fills the flat-gray voids where ceilings/windows are NOT loaded (they occlude the
    # FPV so we skip them). Auto-exposure then calibrates ALL /World lights to a target luminance on
    # the reach surface -> fixes the downward-FPV washout the bright sphere lights caused (office §10.3).
    _dome = UsdLux.DomeLight.Define(stage, "/World/BehaviorDome")
    _dome.CreateIntensityAttr(400.0)
    _dome.CreateColorAttr((0.72, 0.80, 0.92))
    scene.auto_expose = True
    scene.expose_xy = p_edge
    scene.expose_z = float(p_top) + 0.1

    _register_mdl()

    program = "[Grab] <apple> (1), [Put] <apple> (1) <putspot> (1)"
    mid = (0.5 * (p_ctr[0] + q_ctr[0]), 0.5 * (p_ctr[1] + q_ctr[1]))
    top_z = float(max(p_top, q_top))
    # TPV (QA) = an angled OVERHEAD view: ceilings are NOT loaded, so a high eye looks down into the room
    # without landing inside a wall (the earlier in-room eye framed a blank wall corner).
    tpv_look = (float(mid[0]), float(mid[1]), top_z)
    tpv_eye = (
        float(mid[0]) - p_fwd[0] * 1.8,
        float(mid[1]) - p_fwd[1] * 1.8,
        top_z + 3.2,
    )
    print(
        f"[behavior-wgp] {scene_name}: {len(placed)} objs dz={dz:.3f} "
        f"pick={pick[0]}@({p_cx:.2f},{p_cy:.2f},{p_top:.2f}) "
        f"put={put[0]}@({put[3][0]:.2f},{put[3][1]:.2f},{q_top:.2f}) single={single} walk={best:.1f}m grab={grab}"
    )
    return scene, {}, program, tpv_look, tpv_eye


def _static_open_container(stage, cpp, cat, model) -> None:
    """Make a lidded container read OPEN for an INSIDE drop by HIDING its lid leaf (the recorded revolute
    body1). A static HIDE is robust + occlusion-free: a kinematic rotate about a sub-prim whose local
    origin is NOT the true hinge renders as a giant slab that walls off the egocentric view (observed on
    trash_can/ifzxzj). The ACTED, correctly-hinged human-open is a later demo. Best-effort: no-op if there
    is no revolute joint / the leaf sub-prim cannot be located (caller wraps in try/except)."""
    from pxr import Usd, UsdGeom

    if not cat or not model:
        return
    sp = read_spec(f"{cat}/{model}")
    jspec = next((j for j in sp.get("joints", []) if j.get("type") == "revolute"), None)
    if jspec is None:
        return
    leaf_name = str(jspec.get("body1", "")).rstrip("/").split("/")[-1]
    if not leaf_name:
        return
    root = stage.GetPrimAtPath(cpp)

    def _deinst(prim):  # de-instance top-down so the leaf sub-prim becomes editable
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
        for c in prim.GetChildren():
            _deinst(c)

    _deinst(root)
    leaf = next((p for p in Usd.PrimRange(root) if p.GetName() == leaf_name), None)
    if leaf is None:
        return
    UsdGeom.Imageable(leaf).MakeInvisible()
    print(
        f"[behavior-task] open container: HID lid leaf {cat}/{model}:{leaf_name} "
        f"(static-open via hide -> INSIDE drop unblocked, no occluding slab)"
    )


def _collision_bbox(stage, prim_path):
    """World AABB ``(mn, mx)`` over ``prim_path``'s OG per-link collision meshes (``/collisions/*``,
    purpose='guide'), or None if it has none. This is the REAL solid extent -- tighter than the visual
    world bbox, which can overhang open floor / an adjacent chair -- so graspables can be placed over the
    actual surface they will settle onto. BBoxCache includes 'guide' (the collision meshes' purpose)."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not (prim and prim.IsValid()):
        return None
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    mn = [1e30, 1e30, 1e30]
    mx = [-1e30, -1e30, -1e30]
    found = False
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() == "Mesh" and "/collisions/" in str(p.GetPath()):
            rng = bc.ComputeWorldBound(p).ComputeAlignedRange()
            a, b = rng.GetMin(), rng.GetMax()
            if not (b[0] >= a[0]):
                continue
            for i in range(3):
                mn[i] = min(mn[i], a[i])
                mx[i] = max(mx[i], b[i])
            found = True
    return (mn, mx) if found else None


def _connect_boards(boards, gap_x, gap_y):
    """Group same-level boards ``(top_z, xmin, xmax, ymin, ymax)`` into connected components by xy adjacency:
    two boards join when their x-intervals AND their y-intervals are within ``gap_x`` / ``gap_y`` of each
    other (overlap counts as gap 0). Overlapping / hairline-adjacent boards (one shelf built from several
    meshes, or slats) merge into ONE space; boards separated by a real lateral gap (an L/R divider) stay
    DISTINCT spaces. Returns a list of board-lists."""
    n = len(boards)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _igap(lo1, hi1, lo2, hi2):
        return max(
            0.0, max(lo1, lo2) - min(hi1, hi2)
        )  # gap between two 1D intervals (0 if overlapping)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boards[i], boards[j]
            if (
                _igap(a[1], a[2], b[1], b[2]) <= gap_x
                and _igap(a[3], a[4], b[3], b[4]) <= gap_y
            ):
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boards[i])
    return list(groups.values())


def _name_spaces_top_down(spaces, name_prefix):
    ordered = sorted(spaces, key=lambda space: (-space[0], space[2]))
    return tuple(
        (f"{name_prefix}_space{index + 1}",) + tuple(float(value) for value in space)
        for index, space in enumerate(ordered)
    )


def _extract_spaces(stage, prim_path, name_prefix):
    """Distinct interior placement SPACES of a container, from its per-link ``/collisions/*`` guide meshes.
    A space = one connected group of near-HORIZONTAL, THIN boards at the same height that spans a real
    interior region and is NOT the body's own outer TOP face (the surface an object would falsely 'float'
    on). Boards are clustered by z into LEVELS, then each level is split by xy-connectivity: boards that
    overlap / nearly touch merge into ONE space (a shelf built from several meshes); boards at the same z
    separated by a real lateral gap (an L/R divider) stay SEPARATE spaces -- so a fridge's left-shelf and
    right-shelf become two spaces. Returns ``((name, top_z, bottom_z, cx, cy, hx, hy), ...)`` TOP-DOWN then
    LEFT-RIGHT, named ``<name_prefix>_space{i}`` (1-based); empty for a single-box / top-open bin (no
    internal boards) so the caller falls back to the single interior drop. General geometry -- every
    threshold is a fraction of the container's OWN size, no per-asset / per-scene constant."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not (prim and prim.IsValid()):
        return ()
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    meshes = []
    amn = [1e30, 1e30, 1e30]
    amx = [-1e30, -1e30, -1e30]
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() == "Mesh" and "/collisions/" in str(p.GetPath()):
            rng = bc.ComputeWorldBound(p).ComputeAlignedRange()
            a, b = rng.GetMin(), rng.GetMax()
            if not (b[0] >= a[0]):
                continue
            meshes.append((a, b))
            for i in range(3):
                amn[i] = min(amn[i], a[i])
                amx[i] = max(amx[i], b[i])
    if len(meshes) < 2:
        return ()  # a single-box bin has no internal boards
    ww, dd, hh = amx[0] - amn[0], amx[1] - amn[1], amx[2] - amn[2]
    if min(ww, dd, hh) <= 1e-3:
        return ()
    z_thin = 0.15 * hh  # a board is THIN in z relative to the body height
    z_edge = 0.06 * hh  # ignore the body's own outer TOP face (the float-bug lid)
    boards = []  # (top_z, xmin, xmax, ymin, ymax, bottom_z) -- FULL xy extent for connectivity + z-slab
    for a, b in meshes:
        w, d, h = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        if h > z_thin or w < 0.4 * ww or d < 0.4 * dd:
            continue  # not a wide, thin, horizontal board (a wall / the body / a bracket)
        top = float(b[2])
        if top >= amx[2] - z_edge:
            continue  # the body's own TOP face -> an object there is ON the lid, not INSIDE
        boards.append(
            (top, float(a[0]), float(b[0]), float(a[1]), float(b[1]), float(a[2]))
        )
    if not boards:
        return ()
    boards.sort(key=lambda c: -c[0])  # TOP-DOWN
    z_merge = 0.05 * hh  # boards within this z-tolerance are the same LEVEL
    levels = []
    for c in boards:
        if levels and abs(levels[-1][-1][0] - c[0]) < z_merge:
            levels[-1].append(c)
        else:
            levels.append([c])
    gap_x = (
        0.08 * ww
    )  # a real lateral divider gap (fraction of the container's OWN width/depth)
    gap_y = 0.08 * dd
    spaces = []
    for lvl in levels:
        for g in _connect_boards(lvl, gap_x, gap_y):
            xmn = min(x[1] for x in g)
            xmx = max(x[2] for x in g)
            ymn = min(x[3] for x in g)
            ymx = max(x[4] for x in g)
            spaces.append(
                (
                    max(x[0] for x in g),  # top_z
                    min(
                        x[5] for x in g
                    ),  # bottom_z (board underside -> z-slab for the arm obstacle)
                    0.5 * (xmn + xmx),  # cx
                    0.5 * (ymn + ymx),  # cy
                    max(0.03, 0.5 * (xmx - xmn)),  # hx
                    max(0.03, 0.5 * (ymx - ymn)),  # hy
                )
            )
    spaces.sort(key=lambda s: (-s[0], s[2]))  # TOP-DOWN (top_z), then LEFT-RIGHT (cx)
    if os.environ.get("SPACES_DEBUG"):
        print(
            f"[spaces-dbg] {name_prefix}: {len(boards)} boards -> {len(levels)} levels "
            f"-> {len(spaces)} spaces (ww={ww:.3f} dd={dd:.3f} hh={hh:.3f})"
        )
        for lv in levels:
            print(f"[spaces-dbg]  level z~{lv[0][0]:.3f}: {len(lv)} board(s)")
            for bd in lv:
                print(
                    f"[spaces-dbg]    board top={bd[0]:.3f} bot={bd[5]:.3f} "
                    f"x[{bd[1]:.3f},{bd[2]:.3f}] y[{bd[3]:.3f},{bd[4]:.3f}]"
                )
        for sp in spaces:
            print(
                f"[spaces-dbg]  SPACE top={sp[0]:.3f} bot={sp[1]:.3f} "
                f"c=({sp[2]:.3f},{sp[3]:.3f}) h=({sp[4]:.3f},{sp[5]:.3f})"
            )
    return _name_spaces_top_down(spaces, name_prefix)


# ---- generic (no-BDDL) task builder -------------------------------------------------------------------
# Library graspables build_behavior_generic INJECTS into a scene that ships none of its own. Selected from
# the bake cache under four constraints: spec role_hint GRASPABLE (mass <= 3 kg, max dim <= 0.35 m), zero
# articulation joints, max bbox dim in 0.06-0.16 m (drops into any cavity that passes _container_interior
# and still reads in an FPV frame), and a compact convex-ish shape the live SDF wrap closes on -- flat
# plates and long screwdrivers are pinch grasps and are deliberately absent. The first five are the exact
# models the four MAST training tasks already grab, so they are proven end-to-end.
# GROWING THIS LIST: a palette ref belongs to no scene, so a scene-scoped asset tar will NOT carry it --
# every entry must also be present in the external asset cache or the build raises here instead
# of silently rendering an empty table.
_GRASP_PALETTE = (
    "apple/agveuv",
    "can_of_soda/itolcg",
    "toy_figure/acymin",
    "half_banana/xytkre",
    "half_pomegranate/vqybii",
    "bar_soap/feqemg",
    "tennis_ball/rgekxe",
    "bell_pepper/wszvwc",
)

# The container VOCABULARY -- open-top tokens, shelf tokens, the never-a-drop-target exclusions and the
# revolute joint rule deliberately lives in the shared world-catalog scene manifest, which
# the offline sweep gates on too; see _container_family.
_SCENE_MANIFEST: list = []  # 1-slot lazy cache: [module] or [None] once resolution has been tried


def _scene_manifest():
    """Resolve the pure-Python world-catalog scene manifest once, or return None.

    The import is lazy and guarded. Announce a failure once: a scene_manifest that stops importing
    (a syntax error, a renamed
    symbol) would otherwise turn every task into a cross-house walk with nothing in any log to
    distinguish that from a house with one room. What the two callers do next differs -- ``_room_of``
    degrades to "no room information" because co-location is a PREFERENCE, ``_container_family`` raises
    because the family gate is a REQUIREMENT -- but they share this one bridge and this one warning."""
    if _SCENE_MANIFEST:
        return _SCENE_MANIFEST[0]
    mod = None
    try:
        from simulator.world_catalog import scene_manifest

        mod = scene_manifest
    except Exception as exc:  # noqa: BLE001  -- absent / broken manifest module
        print(
            f"[behavior-import] scene_manifest unavailable ({type(exc).__name__}: {exc}) -- "
            f"co-location preference OFF, containers rank by distance alone; the generic "
            f"builder cannot run at all"
        )
    _SCENE_MANIFEST.append(mod)
    return mod


def _room_of(scene_name: str, x: float, y: float):
    """``scene_manifest.room_of(scene, x, y)`` if that module is importable, else None.

    The manifest json may not exist at all, so this degrades to "no room information" instead of failing
    a build: co-location is a PREFERENCE and must never be able to veto a scene."""
    sm = _scene_manifest()
    if sm is None:
        return None
    try:
        return sm.room_of(scene_name, float(x), float(y))
    except Exception:  # noqa: BLE001
        return None


def _container_family(cat: str, model) -> tuple:
    """``(kind, shelf_family)`` for one placed prim, from the SHARED predicate in scene_manifest.

    ``kind`` is "openable" / "open_top" / None (None = not a container family). ``shelf_family`` says the
    interior is BOARDS, so it is real only once ``_extract_spaces`` confirms it -- the half of the gate
    the offline sweep cannot run.

    The spec is read HERE, by ref, so the builder's own cache-miss behaviour applies; an unreadable spec
    becomes ``{}`` and the model then qualifies on its category alone."""
    sm = _scene_manifest()
    if sm is None:
        raise ContainerPredicateUnavailable(
            "the world-catalog scene manifest did not import (see the warning above), so the shared "
            "container family predicate is unreachable. It is pure stdlib and lives beside this file "
            "-- install the complete simulator package rather than reintroducing a local copy."
        )
    try:
        spec = read_spec(f"{cat}/{model}") if model else {}
    except (OSError, ValueError):
        spec = {}
    return sm.container_kind(cat, spec), sm.container_shelf_family(cat)


def _surface_edge(surface, centroid):
    """(yaw_deg, edge_xy, fwd, half, top_z, center_xy) for one ``_compose_behavior_scene`` surface tuple.

    Same construction as build_behavior_wgp's nested ``_edge``: the approach runs from the scene centroid
    OUTWARD through the surface center, so the human stands on the open-room side and reaches the near
    edge. Module-level and numpy-free so the generic builder can score surfaces without duplicating it."""
    _n, _pp, top, (cx, cy), (hx, hy), _a = surface
    dx, dy = float(cx) - float(centroid[0]), float(cy) - float(centroid[1])
    nn = math.hypot(dx, dy)
    fwd = (dx / nn, dy / nn) if nn > 1e-6 else (0.0, -1.0)
    yaw = math.degrees(math.atan2(fwd[0], -fwd[1]))
    edge = min(hx / max(abs(fwd[0]), 1e-6), hy / max(abs(fwd[1]), 1e-6))
    d_in = max(0.05, edge - 0.10)
    return (
        yaw,
        (float(cx) - fwd[0] * d_in, float(cy) - fwd[1] * d_in),
        fwd,
        (float(hx), float(hy)),
        float(top),
        (float(cx), float(cy)),
    )


def _stand_clutter(boxes, stand, self_name, radius=0.55):
    """How many OTHER furniture footprints fall within ``radius`` of a stand cell -- build_behavior_wgp's
    approachability score. A table ringed by chairs or jammed into a corner has no clean stand, so the
    init gate aborts the episode; the builder prefers whichever target has the fewest neighbours."""
    n = 0
    for nm, mn, mx in boxes:
        if nm == self_name:
            continue
        dx = max(float(mn[0]) - stand[0], 0.0, stand[0] - float(mx[0]))
        dy = max(float(mn[1]) - stand[1], 0.0, stand[1] - float(mx[1]))
        if dx * dx + dy * dy < radius * radius:
            n += 1
    return n


def _approach_from(src_xy, tgt_xy):
    """(yaw_deg, unit fwd) for a human at ``src_xy`` facing ``tgt_xy`` -- pose_body's yaw convention
    (atan2(fx, -fy)), the same construction build_behavior_task uses to face its container."""
    dx = float(tgt_xy[0]) - float(src_xy[0])
    dy = float(tgt_xy[1]) - float(src_xy[1])
    nn = math.hypot(dx, dy) or 1.0
    fwd = (dx / nn, dy / nn)
    return math.degrees(math.atan2(fwd[0], -fwd[1])), fwd


def _pick_palette_refs(palette, n_grasp: int, seed: int, scene_name: str) -> list:
    """``n_grasp`` DISTINCT baked palette refs, deterministic in (scene_name, seed).

    Unbaked refs are dropped HERE rather than at build_object time, so a partially staged asset tar
    yields fewer objects instead of a mid-render crash; an entirely unbaked palette raises."""
    import random

    baked = [r for r in dict.fromkeys(palette or ()) if _is_baked(r)]
    if not baked:
        raise NoGraspableAsset(
            f"build_behavior_generic({scene_name}): none of the {len(tuple(palette or ()))} palette "
            f"refs is baked under {CACHE_USD}. A palette item belongs to no scene, so a scene-scoped "
            f"asset tar does not carry it -- stage the palette explicitly."
        )
    rng = random.Random(f"{scene_name}:{int(seed)}")
    rng.shuffle(baked)
    n = max(1, min(int(n_grasp), len(baked)))
    if n < int(n_grasp):
        print(
            f"[behavior-generic] {scene_name}: injecting {n} graspables, not the {int(n_grasp)} asked "
            f"for -- the palette holds only {len(baked)} baked refs"
        )
    return baked[:n]


def _grasp_keys(refs) -> list:
    """Executor keys for injected graspables: the CATEGORY, suffixed only when repeated (apple, apple_2).
    The key is what the assistant and the goal synthesizer see (IsaacEnv.vocab), so it has to read as the
    object's name rather than as a model hash."""
    keys: list = []
    used: dict = {}
    for r in refs:
        cat = _split(r)[0]
        used[cat] = used.get(cat, 0) + 1
        keys.append(cat if used[cat] == 1 else f"{cat}_{used[cat]}")
    return keys


def _container_interior(mn, mx, floor_z):
    """Drop geometry for a cavity whose world AABB is ``(mn, mx)``, or None when it cannot take a palette
    object. Interior derivation matches build_behavior_task (bottom + 0.06 up to top - 0.05, footprint
    shrunk to 80%). Rejected: a cavity narrower than the widest palette item, a lip at ankle height (a rug
    reads as a 2 cm interior), a lip above the reach ceiling (the drop would be a blind overhead toss and
    the init reach gate would abort the episode), and a footprint wider than any real compartment -- a car
    and a 4 m lab bench are both articulated, so their AABB "interior" is a mid-air point under the body.

    The interior HEIGHT is then capped so the PutInside release (Entity.drop_z is None here, so the drop
    happens at z + height) stays at the reachable ~0.90 m _build_task_real_layout uses. The cap only
    lowers the TOP of the INSIDE band; the object comes to rest near ``inner_z``, so the band still
    contains it.

    The ACCEPT half of this test is mirrored offline by ``scene_manifest.container_cavity_ok`` (over the
    looser visual AABB, floor at z=0), which is how the sweep predicts what this will reject. Move a
    threshold here and move it there."""
    hx = 0.5 * (float(mx[0]) - float(mn[0]))
    hy = 0.5 * (float(mx[1]) - float(mn[1]))
    top_z = float(mx[2])
    inner_z = float(mn[2]) + 0.06
    inner_h = (top_z - 0.05) - inner_z
    if min(hx, hy) < 0.09 or max(hx, hy) > 1.00 or inner_h < 0.10:
        return None
    if top_z < float(floor_z) + 0.15 or inner_z > float(floor_z) + 1.35:
        return None
    inner_h = min(inner_h, max(0.10, (float(floor_z) + 0.90) - inner_z))
    cx, cy = 0.5 * (float(mn[0]) + float(mx[0])), 0.5 * (float(mn[1]) + float(mx[1]))
    return {
        "xy": (cx, cy),
        "inner_z": inner_z,
        "inner_h": inner_h,
        "half": (max(0.05, hx * 0.80), max(0.05, hy * 0.80)),
        "foot_half": (hx, hy),
        "top_z": top_z,
        "area": 4.0 * hx * hy,
    }


def _container_candidates(stage, placed, floor_z) -> list:
    """Every placed furniture prim that can take an object INSIDE, with its drop geometry.

    TWO gates, both required. FAMILY: ``scene_manifest.container_kind`` -- THE shared category+joint
    predicate, imported rather than copied so the offline sweep admits exactly the scenes this builder
    can build (the private twin that used to live here disagreed with it on 1222 of 13516 scene
    instances). CAVITY: ``_container_interior`` over the REAL collision solid, which stays HERE because
    the manifest has no stage -- a category token cannot tell a crate from a chair, and a shelf family
    has no interior at all until ``_extract_spaces`` finds its boards. Each dict carries what
    scene.register needs."""
    import fpv_scene

    out: list = []
    for name, (pp, o) in placed.items():
        cat = str(o.get("cat") or "")
        kind, shelf_family = _container_family(cat, o.get("model"))
        if kind is None:
            continue
        # _extract_spaces walks the whole USD subtree, so it runs ONLY for the families whose interior
        # is boards -- a residential scene carries several hundred prims and the walk is not free.
        spaces = (
            _extract_spaces(stage, pp, name)
            if (kind == "openable" or shelf_family)
            else ()
        )
        if shelf_family and not (kind == "openable" or spaces):
            continue  # a 'shelf' category with no boards is a solid block, not a container
        box = _collision_bbox(stage, pp)
        if box is None:
            rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(pp))
            box = (tuple(rng.GetMin()), tuple(rng.GetMax()))
        geo = _container_interior(box[0], box[1], floor_z)
        if geo is None:
            continue
        geo.update(
            {
                "name": name,
                "prim": pp,
                "cat": cat,
                "model": o.get("model"),
                "kind": kind,
                "spaces": spaces,
                "room": None,
            }
        )
        out.append(geo)
    return out


def _container_rank(cand, from_xy, room_pref, boxes):
    """Sort key (lower is better) for the drop container: the START ROOM first, then a walk that is
    neither degenerate nor a cross-house trek, then a clean approach stand, then the roomiest interior,
    then the nearest, then the name for determinism.

    Co-location leads but is only ever a PREFERENCE: with no container in the start room every candidate
    ties on the first key and the ordering falls straight through to "the nearest usable one". A hard
    same-room rule would reject 3 of the 4 scenes that train today (TASKGEN_PLAN_0809)."""
    d = math.hypot(cand["xy"][0] - from_xy[0], cand["xy"][1] - from_xy[1])
    _yaw, fwd = _approach_from(from_xy, cand["xy"])
    back = max(cand["foot_half"]) + 0.45
    stand = (cand["xy"][0] - fwd[0] * back, cand["xy"][1] - fwd[1] * back)
    return (
        0 if (room_pref is not None and cand["room"] == room_pref) else 1,
        0 if 0.8 <= d <= 6.0 else 1,
        _stand_clutter(boxes, stand, cand["name"]),
        -cand["area"],
        d,
        cand["name"],
    )


def _pick_work_surface(scene_name, surfaces, centroid, boxes, cont_rooms):
    """The staging surface: build_behavior_wgp's approachability score (fewest furniture boxes around the
    stand cell, larger wins ties) with a CO-LOCATION preference in front of it -- a surface in a room that
    also holds a container makes a single-room task. Preference only: when no room is annotated, or none
    holds a container, every surface ties on that key and the wgp ordering is recovered exactly."""

    def _key(s):
        _yaw, edge, fwd, _half, _top, ctr = _surface_edge(s, centroid)
        stand = (edge[0] - fwd[0] * 0.55, edge[1] - fwd[1] * 0.55)
        room = _room_of(scene_name, ctr[0], ctr[1])
        return (
            0 if (room is not None and room in cont_rooms) else 1,
            _stand_clutter(boxes, stand, s[0]),
            -s[5],
            s[0],
        )

    return min(surfaces, key=_key)


def _graspable_row(stage, surface_prim, edge_xy, fwd, top_z, n: int) -> list:
    """``n`` evenly spaced ``(x, y, top_z)`` staging slots along a work surface's near edge, measured on
    its REAL collision solid. Same derivation as build_behavior_task: the VISUAL near-edge can overhang
    open floor or an adjacent chair, so an object staged there settles onto the wrong support and is then
    out of reach. Falls back to the visual edge for a surface with no collision meshes."""
    perp = (-fwd[1], fwd[0])
    cb = _collision_bbox(stage, surface_prim)
    if cb is None:
        ex, ey = float(edge_xy[0]), float(edge_xy[1])
        top, perp_half = float(top_z), 0.30
    else:
        (cmn, cmx) = cb
        ccx, ccy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
        chx, chy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
        cedge = min(chx / max(abs(fwd[0]), 1e-6), chy / max(abs(fwd[1]), 1e-6))
        cin = max(0.06, cedge - 0.10)  # inset: sit ON the solid, not over its lip
        ex, ey = ccx - fwd[0] * cin, ccy - fwd[1] * cin
        top = float(cmx[2])
        perp_half = abs(chx * perp[0]) + abs(chy * perp[1])
    spacing = 0.13
    if (n - 1) * spacing > 1.4 * perp_half:  # keep the row within the solid
        spacing = 1.4 * perp_half / max(n - 1, 1)
    out = []
    for i in range(n):
        off = (i - (n - 1) / 2.0) * spacing
        out.append((ex + perp[0] * off, ey + perp[1] * off, top))
    return out


def build_behavior_generic(stage, app, scene_name, *, seed=0, n_grasp=2, palette=None):
    """GENERIC walk -> grab -> walk -> put-INSIDE task on ANY baked BEHAVIOR scene: no pre-sampled
    challenge instance, no BDDL, no per-scene hand-tuning.

    build_behavior_wgp is already scene-agnostic and already injects a library object, but its drop target
    is an ONTOP ``putspot`` on a second table and it never publishes a task -- so IsaacEnv.vocab()
    (the environment layer, which reads scene.behavior_task and nothing else) returns an empty menu and
    the online goal synthesizer has nothing to draw from. This builder keeps wgp's proven surface
    approachability placement, swaps the putspot for a REAL container found by geometry, generalises the
    single hardcoded apple to a palette, and publishes ``scene.behavior_task``.

    Deterministic in (scene_name, seed). Every failure is a NAMED BehaviorBuildError subclass --
    NoGraspableAsset / NoWorkSurface / NoContainerFound / MissingStructuralAsset -- so an offline sweep can
    record WHY a scene is inadmissible instead of a shrug. Additive: build_behavior_wgp and every bt_*
    builder are untouched. Returns the step3 5-tuple (scene, controllers, program, tpv_look, tpv_eye)."""
    import fpv_scene
    from pxr import UsdLux
    from scene_state import Role

    refs = _pick_palette_refs(palette or _GRASP_PALETTE, n_grasp, seed, scene_name)
    keys = _grasp_keys(refs)
    # grab=None -> no scene.behavior_grab: this is the MULTI-graspable render path (scene.grasp_objs),
    # exactly as build_behavior_task drives it.
    scene, placed, surfaces, _centroid, _floor_top, dz = _compose_behavior_scene(
        stage, app, scene_name, None, None, None
    )
    centroid = (float(_centroid[0]), float(_centroid[1]))
    if not surfaces:
        raise NoWorkSurface(
            f"build_behavior_generic({scene_name}): no comfortable work surface among {len(placed)} "
            f"prims (needs a top in [0.62, 1.15] m with area >= 0.06 m2, not a seat/bed/appliance)"
        )
    cands = _container_candidates(stage, placed, scene.floor_z)
    if not cands:
        raise NoContainerFound(
            f"build_behavior_generic({scene_name}): none of {len(placed)} prims is a usable drop "
            f"container (open-top / lidded / shelved family AND a reachable cavity >= 0.18 m wide)"
        )
    boxes = scene.solid_boxes()
    for c in cands:
        c["room"] = _room_of(scene_name, c["xy"][0], c["xy"][1])
    cont_rooms = {c["room"] for c in cands if c["room"] is not None}

    pick = _pick_work_surface(scene_name, surfaces, centroid, boxes, cont_rooms)
    p_yaw, p_edge, p_fwd, _p_half, p_top, p_ctr = _surface_edge(pick, centroid)
    p_stand = (p_edge[0] - p_fwd[0] * 0.55, p_edge[1] - p_fwd[1] * 0.55)
    scene.start_xy = (p_edge[0] - p_fwd[0] * 2.2, p_edge[1] - p_fwd[1] * 2.2)
    start_room = _room_of(scene_name, scene.start_xy[0], scene.start_xy[1])

    scene.grasp_objs = []
    for key, ref, (sx, sy, s_top) in zip(
        keys, refs, _graspable_row(stage, pick[1], p_edge, p_fwd, p_top, len(keys))
    ):
        path = f"/World/Obj_{_safe(key)}"
        scene.grasp_objs.append(
            {"key": key, "path": path, "build": behavior_build_params(ref)}
        )
        scene.register(
            key,
            Role.GRASPABLE,
            xy=(float(sx), float(sy)),
            z=float(s_top) + 0.05,
            prim_path=path,
            approach_yaw=p_yaw,
            foot_xy=(float(sx), float(sy)),  # stand IN FRONT of each object
            foot_half=(0.12, 0.12),
            top_z=float(s_top),
            clearance=0.34,
        )

    # A bottom cabinet passes BOTH filters, and staging the objects on the very unit they go into is a
    # zero-walk task. Prefer any other candidate; keep it only when it is the scene's only container.
    elsewhere = [c for c in cands if c["prim"] != pick[1]] or cands
    cont = min(elsewhere, key=lambda c: _container_rank(c, p_stand, start_room, boxes))
    c_yaw, _c_fwd = _approach_from(p_stand, cont["xy"])
    # Replaces the Role.SURFACE _compose_behavior_scene registered for this prim (register is
    # last-write-wins), keeping its obstacle fields -- the same swap _build_task_real_layout does.
    scene.register(
        cont["name"],
        Role.CONTAINER,
        xy=cont["xy"],
        z=cont["inner_z"],
        height=cont["inner_h"],
        half=cont["half"],
        prim_path=cont["prim"],
        approach_yaw=c_yaw,
        foot_xy=cont["xy"],
        foot_half=cont["foot_half"],
        top_z=cont["top_z"],
        clearance=0.30,
        spaces=cont["spaces"],
    )
    if cont["kind"] == "openable":
        try:
            _static_open_container(stage, cont["prim"], cont["cat"], cont["model"])
        except Exception as _exc:  # noqa: BLE001  -- best-effort, as in every other caller
            print(
                f"[behavior-generic] static-open lid skipped for {cont['name']}: {_exc}"
            )

    # Scene flags: read off build_behavior_wgp (near_start/start_xy/plan_nav/unified_geom/
    # post_action_check/auto_expose/behavior_support_prims), plus the container prim as a settle support
    # so a dropped object rests on the interior floor instead of tunnelling through it.
    scene.behavior_support_prims = sorted({pick[1], cont["prim"]})
    scene.near_start = True
    scene.plan_nav = True
    scene.unified_geom = True
    scene.post_action_check = True
    for _tag, (_lx, _ly), _lz in (
        ("Pick", p_ctr, float(p_top) + 1.9),
        ("Cont", cont["xy"], float(cont["top_z"]) + 1.9),
    ):
        lt = UsdLux.SphereLight.Define(stage, f"/World/BehaviorLight{_tag}")
        lt.CreateRadiusAttr(0.6)
        lt.CreateIntensityAttr(60000.0)
        lt.CreateColorAttr((1.0, 1.0, 1.0))
        fpv_scene.set_world_pose(lt.GetPrim(), (float(_lx), float(_ly), float(_lz)))
    _dome = UsdLux.DomeLight.Define(stage, "/World/BehaviorDome")
    _dome.CreateIntensityAttr(400.0)
    _dome.CreateColorAttr((0.72, 0.80, 0.92))
    scene.auto_expose = True
    scene.expose_xy = p_edge
    scene.expose_z = float(p_top) + 0.1
    _register_mdl()

    # PutInside only for a MULTI-space container (choose a shelf + reach into it); a single-cavity bin
    # achieves the same `inside` with a plain [Put] onto its interior floor -- _build_task_real_layout's rule.
    verb = "PutInside" if len(cont["spaces"]) >= 2 else "Put"
    cname = cont["name"]
    program = ", ".join(
        f"[Walk] <{k}> (1), [Grab] <{k}> (1), [Walk] <{cname}> (1), "
        f"[{verb}] <{k}> (1) <{cname}> (1)"
        for k in keys
    )
    scene.behavior_task = {
        "scene": scene_name,
        # episode_runtime's goal-check subscripts bt["task"] unconditionally -- in the report line AND in its
        # own except handler -- so a task-less dict turns any render into a KeyError there.
        "task": f"generic_{scene_name}",
        "grasp": list(keys),
        "container": cname,
        "existential_witness": {},
        # (object, container) pairs: isaac_env._derive_goal and episode_runtime both compile this as
        # isaac_goal.inside(o, c), so element 0 MUST be the object key, never the predicate name.
        "goals": [(k, cname) for k in keys],
    }

    mid = (0.5 * (p_ctr[0] + cont["xy"][0]), 0.5 * (p_ctr[1] + cont["xy"][1]))
    top_z = max(float(p_top), float(cont["top_z"]))
    tpv_look = (float(mid[0]), float(mid[1]), top_z)
    tpv_eye = (
        float(mid[0]) - p_fwd[0] * 1.8,
        float(mid[1]) - p_fwd[1] * 1.8,
        top_z + 3.2,
    )
    print(
        f"[behavior-generic] {scene_name}: {len(placed)} objs dz={dz:.3f} seed={seed} "
        f"pick={pick[0]}@({p_ctr[0]:.2f},{p_ctr[1]:.2f},{p_top:.2f}) room={start_room} "
        f"grasp={keys} refs={refs} container={cname}[{cont['kind']}] room={cont['room']} "
        f"spaces={len(cont['spaces'])} verb={verb} "
        f"cands={len(cands)} missing_bakes={getattr(scene, 'behavior_missing_bakes', 0)}"
    )
    return scene, {}, program, tpv_look, tpv_eye


_AVAIL_TASKS_YAML = os.path.join(
    os.path.dirname(CHALLENGE_ROOT), "metadata", "available_tasks.yaml"
)


def _challenge_start(task_name: str):
    """(x, y) human start from ``available_tasks.yaml`` -- the UNIFORM start source (present for all 100
    tasks). Minimal stdlib parse (no pyyaml). We deliberately do NOT use the per-template
    ``metadata.task.robot_poses`` for the start: it ships in 3 inconsistent schemas ('robot' list vs
    per-robot-model keys R1Pro/Fetch/... vs empty), and the missing-'robot' case silently defaulted the
    start to (0,0) (a #1 unreachable cause). Returns None if not found."""
    try:
        lines = open(_AVAIL_TASKS_YAML).read().splitlines()
    except OSError:
        return None
    in_task = False
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Za-z0-9_]+):\s*$", line)
        if m:
            in_task = m.group(1) == task_name
            continue
        if in_task and line.strip() == "robot_start_position:":
            vals = []
            for j in range(i + 1, min(i + 4, len(lines))):
                s = lines[j].strip()
                if not s.startswith("-"):
                    break
                try:
                    vals.append(float(s[1:].strip()))
                except ValueError:
                    break
            if len(vals) >= 2:
                return (vals[0], vals[1])
    return None


def _ceiling_footprint(o) -> tuple[float, float]:
    """World-axis-aligned (w, h) of one ceiling slab: its baked bbox, scaled, then yawed.

    (0, 0) when the slab has no baked spec. Ceilings are in _SKIP_CATS, so nothing else in the build
    reads one and a bake that never wrote it is not otherwise an error -- losing one room's fill light
    is the right cost, killing the episode is not."""
    try:
        bb = read_spec(f"{o['cat']}/{o['model']}").get("bbox_size")
    except OSError:
        return 0.0, 0.0
    if not bb:
        return 0.0, 0.0
    sc = o.get("scale", [1.0, 1.0, 1.0])
    w, h = float(bb[0]) * float(sc[0]), float(bb[1]) * float(sc[1])
    qx, qy, qz, qw = (float(v) for v in o["ori"])
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return w * c + h * s, w * s + h * c


def _interior_fill_cells(
    by_name, spacing, dz, drop
) -> list[tuple[float, float, float]]:
    """Lamp positions covering every ceilinged room, no more than ``spacing`` apart.

    The scene's OWN ``ceilings`` slabs are the room partition: one exists per enclosed room and their
    union is exactly the floor a roof takes the dome away from. The floor slabs are not a substitute --
    they run on under the eaves and out onto the patio, which is open to the sky and already lit."""
    out: list[tuple[float, float, float]] = []
    for o in by_name.values():
        if o["cat"] != "ceilings":
            continue
        w, h = _ceiling_footprint(o)
        if w <= 0.0 or h <= 0.0:
            continue
        cx, cy, cz = (float(v) for v in o["pos"])
        nx, ny = max(1, math.ceil(w / spacing)), max(1, math.ceil(h / spacing))
        for i in range(nx):
            for j in range(ny):
                out.append(
                    (
                        cx - w * 0.5 + (i + 0.5) * w / nx,
                        cy - h * 0.5 + (j + 0.5) * h / ny,
                        cz + dz - drop,
                    )
                )
    return out


def _add_interior_fill_lights(stage, by_name, placed, dz, cfg) -> int:
    """Author the per-room fill lamps and return how many were placed.

    Type-wise these are ordinary ``/World`` UsdLux sphere lights, so the collider pass skips them and
    auto-exposure scales them along with every other light in the scene."""
    import fpv_scene
    from pxr import UsdLux

    boxes = []
    for pp, _o in placed.values():
        pr = stage.GetPrimAtPath(pp)
        if pr and pr.IsValid():
            rng = fpv_scene.compute_world_bbox(pr)
            boxes.append((rng.GetMin(), rng.GetMax()))
    n = 0
    cells = _interior_fill_cells(
        by_name, float(cfg["spacing_m"]), dz, float(cfg["below_ceiling_m"])
    )
    for x, y, z in cells:
        # A lamp that lands inside a wall cabinet or a wall cavity emits into solid geometry and lights
        # nothing at all; the neighbouring cells already cover the floor beneath it.
        if any(
            mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]
            for mn, mx in boxes
        ):
            continue
        lt = UsdLux.SphereLight.Define(stage, f"/World/BehaviorFill{n:02d}")
        lt.CreateRadiusAttr(float(cfg["radius"]))
        lt.CreateIntensityAttr(float(cfg["intensity"]))
        lt.CreateColorAttr(tuple(cfg["color"]))
        fpv_scene.set_world_pose(lt.GetPrim(), (float(x), float(y), float(z)))
        n += 1
    print(
        f"[behavior-task] interior fill: {n}/{len(cells)} lamp(s) @ "
        f"intensity={float(cfg['intensity']):.0f} r={float(cfg['radius']):.2f}"
    )
    return n


def _build_task_real_layout(
    stage,
    app,
    scene,
    placed,
    centroid,
    dz,
    by_name,
    grasp_names,
    goal_preds,
    scene_name,
    task_name,
    i2n,
    robot_start,
    bddl_path,
    existential=None,
):
    """USE_TASK_LAYOUT path: register the task's graspables + targets + human start EXACTLY at the
    challenge instance's sampled positions (no heuristic re-staging), and route each object per its own
    goal (inside->PutInside a container, ontop->Put on a surface). The init non-pen gate + object settle
    snap any slightly-off pose to the nearest valid one. Returns the step3 5-tuple. Additive: only reached
    when the module flag USE_TASK_LAYOUT is True."""
    import math

    import fpv_scene
    import numpy as np
    from pxr import UsdLux
    from scene_state import Role

    cx0, cy0 = float(centroid[0]), float(centroid[1])

    def _ayaw(
        x, y
    ):  # seed approach from the room interior (open side); the init ladder refines the side
        dx, dy = float(x) - cx0, float(y) - cy0
        n = math.hypot(dx, dy) or 1.0
        return math.degrees(math.atan2(dx / n, -dy / n))

    # per-object target from the (floor-filtered) goal: obj -> ('inside'|'ontop', target_name)
    target_of: dict = {}
    for p in goal_preds:
        nm = p.pred.name
        if nm == "INSIDE" and p.a in by_name and p.b in by_name:
            target_of.setdefault(p.a, ("inside", p.b))
        elif nm == "ON" and p.a in by_name and p.b in by_name:
            target_of.setdefault(p.a, ("ontop", p.b))
    grasp_names = [k for k in grasp_names if k in target_of]
    if not grasp_names:
        raise RuntimeError(
            f"build_behavior_task({scene_name}/{task_name}): no routable inside/ontop goal object "
            f"(targets must be concrete scene objects, not floor/room)"
        )

    def _support_prim(
        x, y, z
    ):  # highest placed-furniture top under (x,y) below z -> the object's support
        best, best_top = None, -1e30
        for _nm, (pp, _o) in placed.items():
            pr = stage.GetPrimAtPath(pp)
            if not (pr and pr.IsValid()):
                continue
            rng = fpv_scene.compute_world_bbox(pr)
            mn, mx = rng.GetMin(), rng.GetMax()
            top = float(mx[2])
            if (
                top <= z + 0.08
                and float(mn[0]) - 0.10 <= x <= float(mx[0]) + 0.10
                and float(mn[1]) - 0.10 <= y <= float(mx[1]) + 0.10
                and top > best_top
            ):
                best_top, best = top, pp
        return best

    # --- graspables at REAL positions (rig objects; settle refines onto the real support) --------------
    scene.grasp_objs = []
    supports: set = set()
    for k in grasp_names:
        o = by_name[k]
        sc = o.get("scale", [1.0, 1.0, 1.0])
        sca = float(sum(sc) / len(sc)) if isinstance(sc, (list, tuple)) else float(sc)
        scene.grasp_objs.append(
            {
                "key": k,
                "path": f"/World/Obj_{_safe(k)}",
                "build": behavior_build_params(f"{o['cat']}/{o['model']}", sca),
            }
        )
    for k in grasp_names:
        o = by_name[k]
        px, py = float(o["pos"][0]), float(o["pos"][1])
        pz = float(o["pos"][2]) + dz
        sp = _support_prim(px, py, pz)
        if sp:
            supports.add(sp)
        scene.register(
            k,
            Role.GRASPABLE,
            xy=(px, py),
            z=pz,
            prim_path=f"/World/Obj_{_safe(k)}",
            approach_yaw=_ayaw(px, py),
            foot_xy=(px, py),
            foot_half=(0.12, 0.12),
            top_z=pz,
            clearance=0.34,
            meta={"asset_category": o["cat"], "asset_model": o["model"]},
        )

    # --- targets (containers + surfaces) at REAL positions --------------------------------------------
    openable_conts: set = (
        set()
    )  # containers with a revolute leaf -> [Open]/[Close] instead of lid-hide
    for ttype, tgt in sorted(set(target_of.values())):
        tpp = placed.get(tgt, (None,))[0]
        if tpp is None:
            # GRASPABLE-as-target (a bowl / tupperware): NOT a placed furniture prim. Its rig prim
            # /World/Obj_<key> is built LATER at render time, so it does not exist here -- RECORD it and
            # attach the container/surface affordance after the objects are built (dual-role, via
            # attach_graspable_target_affordances). A missing entity is just skipped.
            _ge = scene.entities.get(tgt)
            if _ge is not None and _ge.role is Role.GRASPABLE and _ge.prim_path:
                if not hasattr(scene, "_graspable_targets"):
                    scene._graspable_targets = []
                scene._graspable_targets.append((tgt, ttype))
            continue
        rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(tpp))
        mn, mx = rng.GetMin(), rng.GetMax()
        tcx, tcy = (
            0.5 * (float(mn[0]) + float(mx[0])),
            0.5 * (float(mn[1]) + float(mx[1])),
        )
        thx = max(0.03, 0.5 * (float(mx[0]) - float(mn[0])))
        thy = max(0.03, 0.5 * (float(mx[1]) - float(mn[1])))
        supports.add(tpp)
        if ttype == "inside":
            inner_z = float(mn[2]) + 0.06
            inner_h = max(0.12, (float(mx[2]) - 0.05) - inner_z)
            # If the container has a drivable leaf (fridge/cabinet door = revolute, drawer chest = prismatic),
            # carry its openable fields so [Open]/[Close] drive it NATURALLY (like step3_openclose) instead of
            # the lid-hide fallback (bins). A FRONT-opening appliance needs its grip + drop keyed off the FRONT
            # face (reachable), not the deep interior center: derive the approach forward, the front face, and a
            # reachable just-inside drop.
            _ayaw_c = _ayaw(tcx, tcy)
            _ar = math.radians(_ayaw_c)
            _fwd = (
                math.sin(_ar),
                -math.cos(_ar),
            )  # room -> container (INTO); front face is the -_fwd side
            _hdepth = abs(_fwd[0]) * thx + abs(_fwd[1]) * thy
            _inset = min(0.22, max(0.0, _hdepth - 0.05))
            _front_int = (
                tcx - _fwd[0] * (_hdepth - _inset),
                tcy - _fwd[1] * (_hdepth - _inset),
            )
            _front_face = (tcx - _fwd[0] * _hdepth, tcy - _fwd[1] * _hdepth)
            _lo, _hi = float(mn[2]) + 0.08, float(mx[2]) - 0.12
            # Drop INTO the real interior floor -- do NOT cap below it at a heuristic "reach ceiling". A HIGH
            # wall cabinet's interior floor (e.g. 1.51 m) sits above ~1.35 m, and capping the drop there put
            # the object BELOW the cabinet in mid-air (the floating bug). Clamp only to the interior z-range
            # [lo, hi]; if that height is genuinely unreachable, the reach gate fails HONESTLY (the object is
            # never placed below the cabinet in mid-air, and the pose-planner leans/reaches up to try).
            _clampz = lambda v: min(max(v, min(_lo, _hi)), max(_lo, _hi))  # noqa: E731
            _shelf_z = _clampz(
                scene.floor_z + 0.90
            )  # -> the interior floor for a high cabinet
            _handle_z = _clampz(
                scene.floor_z + 1.00
            )  # door/drawer grip at the interior height
            _target_object = by_name.get(tgt, {})
            _target_category = str(_target_object.get("cat") or "")
            _of = _openable_leaf_fields(scene, tpp, _target_object)
            _target_model = str(_target_object.get("model") or "")
            _access_meta = {
                "asset_category": _target_category,
                "asset_model": _target_model,
            }
            if _of is not None and _of["kind"] == "swing":
                _access_meta = front_opening_access_metadata(
                    _target_category,
                    _target_model,
                    _of,
                    (tcx, tcy),
                )
                if "entry_axis" in _access_meta:
                    _fwd = tuple(float(value) for value in _access_meta["entry_axis"])
                    _aperture_center = _access_meta["aperture_center"]
                    _front_face = (
                        float(_aperture_center[0]) - _fwd[0] * 0.05,
                        float(_aperture_center[1]) - _fwd[1] * 0.05,
                    )
                    _hdepth = abs(_fwd[0]) * thx + abs(_fwd[1]) * thy
                    _inset = min(0.22, max(0.0, _hdepth - 0.05))
                    _front_int = (
                        _front_face[0] + _fwd[0] * _inset,
                        _front_face[1] + _fwd[1] * _inset,
                    )
                    _ayaw_c = math.degrees(math.atan2(_fwd[0], -_fwd[1]))
                if (
                    is_front_drum_category(_target_category)
                    and "aperture_center" in _access_meta
                ):
                    _aperture_z = float(_access_meta["aperture_center"][2])
                    _shelf_z = _clampz(_aperture_z)
                    _handle_z = _clampz(_aperture_z + 0.15)
            # v1 drives the NATURAL swing (revolute door: fridge / cabinet / oven) about its own hinge, like
            # step3_openclose. A prismatic DRAWER slides along its axis, but its contents must RIDE the drawer
            # out+back (not yet modelled) so a visible slide would float the object -> drawers keep the cleaner
            # lid-hide (front hidden, object drops into the visible interior). _of["kind"]=="slide" -> lid-hide.
            if _of is not None and _of["kind"] == "swing":
                _okw = dict(
                    hinge_xy=_of["hinge_xy"],
                    hinge_axis=(0.0, 0.0, 1.0),
                    swing_angle=_of["swing_angle"],
                    swing_sign=1.0,
                    leaf_paths=_of["leaf_paths"],
                    joint_name=_of["joint_name"],
                    open_thresh=0.10,
                    handle_xy=_front_face,
                    handle_z=_handle_z,
                    drop_z=_shelf_z,
                )
            else:
                _okw = {}
            _shv = (
                ()
                if is_front_drum_category(_target_category)
                else _extract_spaces(scene.stage, tpp, tgt)
            )  # distinct interior placement spaces (top-down, L/R split); () for a drum or top-open bin
            _ckw = dict(
                xy=_front_int,  # reachable just-inside-the-front drop + INSIDE-goal footprint center
                z=inner_z,
                height=inner_h,
                half=(max(0.05, thx * 0.80), max(0.05, thy * 0.80)),
                prim_path=tpp,
                approach_yaw=_ayaw_c,
                # nav obstacle + stand back-off use the TRUE footprint center
                foot_xy=(tcx, tcy),
                foot_half=(thx, thy),
                top_z=float(mx[2]),
                clearance=0.30,
                spaces=_shv,
                meta=_access_meta,
                **_okw,
            )
            scene.register(tgt, Role.CONTAINER, **_ckw)
            if _shv:
                print(
                    f"[behavior-task] container {tgt}: {len(_shv)} interior space(s) "
                    f"{[s[0] for s in _shv]} top_z={[round(s[1], 2) for s in _shv]}"
                )
            if _okw:  # a swing door was wired -> [Open]/[Close] act on it
                openable_conts.add(tgt)
                print(
                    f"[behavior-task] openable container {tgt}: swing leaf -> [Open]/[Close] wired"
                )
            else:  # bin / basket / drawer chest -> reveal the interior by hiding the lid/front
                if _of is not None and _of["kind"] == "slide":
                    print(
                        f"[behavior-task] container {tgt}: prismatic drawer -> lid-hide (interior drop)"
                    )
                try:
                    _static_open_container(
                        stage,
                        tpp,
                        by_name.get(tgt, {}).get("cat"),
                        by_name.get(tgt, {}).get("model"),
                    )
                except Exception as _exc:  # noqa: BLE001
                    print(f"[behavior-task] static-open lid skipped for {tgt}: {_exc}")
        else:
            top = float(mx[2])
            _skw = dict(
                xy=(tcx, tcy),
                z=top,
                half=(thx, thy),
                prim_path=tpp,
                approach_yaw=_ayaw(tcx, tcy),
                foot_xy=(tcx, tcy),
                foot_half=(thx, thy),
                top_z=top,
                clearance=0.40,
            )
            scene.register(tgt, Role.SURFACE, **_skw)
    # Attach the container affordance to EVERY existential candidate, not just the arbitrary compile-time
    # witness: a runtime reroute may move a food into a SIBLING tupperware, which must also be a ready
    # container. Record each graspable candidate (dedup) so attach_graspable_target_affordances builds them all.
    for _subj, _cands in (existential or {}).items():
        _tt = target_of.get(_subj, ("inside", None))[0]
        for _c in _cands:
            _ce = scene.entities.get(_c)
            if _ce is not None and _ce.role is Role.GRASPABLE and _ce.prim_path:
                if not hasattr(scene, "_graspable_targets"):
                    scene._graspable_targets = []
                if (_c, _tt) not in scene._graspable_targets:
                    scene._graspable_targets.append((_c, _tt))
    scene.behavior_support_prims = sorted(supports)

    # --- human start EXACTLY at the challenge start (uniform available_tasks.yaml; template fallback) --
    st = _challenge_start(task_name)
    if st is None and robot_start:
        rs = robot_start.get("position") or robot_start.get("pos")
        if rs:
            st = (float(rs[0]), float(rs[1]))
    if st:
        scene.start_xy = (float(st[0]), float(st[1]))
    scene.near_start = True
    scene.plan_nav = True
    scene.precise_nav = (
        True  # rasterize furniture COLLISION-MESH triangles (not per-owner AABB) in nav
    )
    scene.soft_init = True
    # HARMONY + VALIDATOR (Step 1 + Step 2): all systems read ONE precise collision map (SSOT oracle),
    # and every action is followed by a universal physics validate+correct on that same map.
    scene.unified_geom = True
    scene.post_action_check = True
    # S1 CAMERA: FPV uses the ORIGINAL tuned drive_cams -- do NOT override the FPV aim knobs
    # (fpv_walk_ahead/downtilt/eye_drop/manip_aim_drop). The aggressive down-aim overrides made the FPV
    # pitch JUMP on every walk<->manip transition (the _cap_dir smoother renders the big pitch delta as a
    # multi-frame up/down pan); the gentle native defaults transition smoothly. KEEP tpv_chase for the TPV:
    # in an ENCLOSED BEHAVIOR room a FIXED eye gets buried in furniture -> black TPV, so the tracking chase
    # (offset-probed + furniture-lifted + overhead fallback) is the robust TPV here.
    scene.tpv_chase = True

    # --- lights + dome + auto-expose over the object cluster -------------------------------------------
    ox = float(np.mean([by_name[k]["pos"][0] for k in grasp_names]))
    oy = float(np.mean([by_name[k]["pos"][1] for k in grasp_names]))
    lt = UsdLux.SphereLight.Define(stage, "/World/BehaviorLightWork")
    lt.CreateRadiusAttr(0.6)
    lt.CreateIntensityAttr(60000.0)
    lt.CreateColorAttr((1.0, 1.0, 1.0))
    fpv_scene.set_world_pose(lt.GetPrim(), (ox, oy, 1.7))
    _dome = UsdLux.DomeLight.Define(stage, "/World/BehaviorDome")
    _dome.CreateIntensityAttr(400.0)
    _dome.CreateColorAttr((0.72, 0.80, 0.92))
    _fill_cfg = _interior_fill_config(by_name)
    if _fill_cfg is not None:
        _add_interior_fill_lights(stage, by_name, placed, dz, _fill_cfg)
    scene.auto_expose = True
    scene.expose_xy = (ox, oy)
    scene.expose_z = 1.0
    scene.fpv_headlamp = True  # enclosed BEHAVIOR house: rig a fixed FPV fill so the egocentric view isn't black
    _register_mdl()

    # --- program: open any revolute container first, per-object routing, then close it --------------------
    parts = []
    for _cont in sorted(
        openable_conts
    ):  # walk to + OPEN each fridge/cabinet (empty-handed) up front
        parts.append(f"[Walk] <{_cont}> (1), [Open] <{_cont}> (1)")
    for k in grasp_names:
        ttype, tgt = target_of[k]
        # PutInside ONLY for a MULTI-space container (>=2 distinct interior spaces to choose from -- pick a
        # space + reach into it). A single-cavity bin (0 or 1 space = just its bottom) achieves its `inside`
        # goal with a plain [Put] that drops the object onto the interior floor (same logic as a surface).
        _te = scene.entities.get(tgt)
        _taff = scene.target_view(tgt, Role.CONTAINER) or _te
        _n_spaces = len(getattr(_taff, "spaces", None) or ())
        verb = "PutInside" if (ttype == "inside" and _n_spaces >= 2) else "Put"
        parts.append(
            f"[Walk] <{k}> (1), [Grab] <{k}> (1), [Walk] <{tgt}> (1), [{verb}] <{k}> (1) <{tgt}> (1)"
        )
    for _cont in sorted(
        openable_conts
    ):  # CLOSE each container at the end (goal often wants it shut)
        parts.append(f"[Walk] <{_cont}> (1), [Close] <{_cont}> (1)")
    program = ", ".join(parts)
    scene.behavior_task = {
        "scene": scene_name,
        "task": task_name,
        "bddl": bddl_path,
        "inst_to_name": i2n,
        "grasp": list(grasp_names),
        "container": next((t for tt, t in target_of.values() if tt == "inside"), None),
        # {food: [candidate tupperwares]} for EXISTENTIAL inside/ontop goals -> the runtime re-picks a
        # REACHABLE witness (any instance satisfies `exists`). Empty for fully-named goals.
        "existential_witness": existential or {},
    }
    tpv_look = (ox, oy, 1.0)
    tpv_eye = (ox - 2.6, oy - 2.6, 3.0)
    print(
        f"[behavior-task][real-layout] {scene_name}/{task_name}: grasp={grasp_names} "
        f"targets={sorted(set(target_of.values()))} start={scene.start_xy}"
    )
    return scene, {}, program, tpv_look, tpv_eye


def attach_graspable_target_affordances(scene, stage):
    """Attach the container/surface DROP affordance to each GRASPABLE that is ALSO a put/putinside TARGET
    (a bowl / tupperware). DEFERRED to render time: a graspable's rig prim ``/World/Obj_<key>`` is built AFTER
    build_behavior_task, so its geometry can only be read now. The affordance is a nested Entity on
    ``entity.affordances`` (CONTAINER for ``inside``, SURFACE for ``ontop``), read LIVE from the prim;
    ``entity.target`` remains a compatibility alias for older readers. The primary entity stays GRASPABLE.
    General -- no scene names, no magic metres (only fractions of the object's OWN bbox)."""
    import fpv_scene

    for tgt, ttype in dict.fromkeys(getattr(scene, "_graspable_targets", None) or []):
        e = scene.entities.get(tgt)
        if e is None or not e.prim_path:
            continue
        prim = stage.GetPrimAtPath(e.prim_path)
        if prim is None or not prim.IsValid():
            continue
        rng = fpv_scene.compute_world_bbox(prim)
        mn, mx = rng.GetMin(), rng.GetMax()
        affordance = _graspable_target_entity(e, tgt, ttype, mn, mx)
        e.affordances[affordance.role] = affordance
        if e.target is None:
            e.target = affordance
        print(
            f"[behavior-task] graspable target {tgt} ({ttype}) -> "
            f"{affordance.role.name} affordance attached"
        )


def validate_aggregate_target_capacity(scene, stage) -> None:
    """Require aggregate targets to pack every eligible live object shape."""
    import fpv_scene
    import placement_packing
    from scene_state import Role

    task = getattr(scene, "behavior_task", None) or {}
    goals = tuple(task.get("aggregate_goals") or ())
    loads = task.get("aggregate_target_loads") or {}
    for target_key, raw_count in loads.items():
        count = int(raw_count)
        if count <= 0:
            continue
        target = scene.target_view(str(target_key), Role.CONTAINER)
        if target is None:
            raise ValueError(
                f"aggregate target {target_key} has no live container affordance"
            )
        candidates = tuple(
            dict.fromkeys(
                str(candidate)
                for goal in goals
                if str(goal.get("target") or "") == str(target_key)
                for candidate in goal.get("candidates") or ()
            )
        )
        extents = []
        for candidate in candidates:
            entity = scene.entities.get(candidate)
            prim_path = getattr(entity, "prim_path", None)
            prim = stage.GetPrimAtPath(prim_path) if prim_path else None
            if prim is None or not prim.IsValid():
                raise ValueError(
                    f"aggregate candidate {candidate} has no live geometry"
                )
            bounds = fpv_scene.compute_world_bbox(prim)
            minimum, maximum = bounds.GetMin(), bounds.GetMax()
            extents.append(
                (
                    max(0.02, 0.5 * float(maximum[0] - minimum[0])),
                    max(0.02, 0.5 * float(maximum[1] - minimum[1])),
                )
            )
        if not extents:
            raise ValueError(f"aggregate target {target_key} has no candidate geometry")
        largest = (
            max(extent[0] for extent in extents),
            max(extent[1] for extent in extents),
        )
        if getattr(target, "spaces", None):
            available = len(target.spaces)
        elif (getattr(target, "meta", None) or {}).get("open_top"):
            center = (float(target.xy[0]), float(target.xy[1]))
            sites = placement_packing.strict_pack_sites(
                center,
                (float(target.half[0]), float(target.half[1])),
                largest,
                count,
                center,
            )
            available = len(sites)
        else:
            available = 1
        if available < count:
            raise ValueError(
                f"aggregate target {target_key} needs {count} non-overlapping live "
                f"placement sites for candidate half-extents {largest}, but has {available}"
            )
        print(
            f"[behavior-task] aggregate target {target_key}: {available}/{count} "
            f"live packing sites across {len(extents)} candidate instance(s)"
        )


_GRASPABLE_CONTAINER_HALF_SCALE = 0.92


def _graspable_target_entity(e, tgt, ttype, mn, mx):
    from scene_state import Entity, Role

    cx = 0.5 * (float(mn[0]) + float(mx[0]))
    cy = 0.5 * (float(mn[1]) + float(mx[1]))
    thx = max(0.03, 0.5 * (float(mx[0]) - float(mn[0])))
    thy = max(0.03, 0.5 * (float(mx[1]) - float(mn[1])))
    bounds = (
        tuple(float(value) for value in mn),
        tuple(float(value) for value in mx),
    )
    if ttype == "inside":
        inner_z = float(mn[2]) + 0.06
        top_z = float(mx[2])
        inner_h = max(0.12, (top_z - 0.05) - inner_z)
        height = top_z - float(mn[2])
        # ``z`` is the interior floor used after settling; ``drop_z`` is the
        # object's origin release plane and therefore belongs just under the rim.
        drop_z = top_z - 0.05 * max(0.1, height) - 0.03
        return Entity(
            name=tgt,
            role=Role.CONTAINER,
            xy=(cx, cy),
            z=inner_z,
            height=inner_h,
            half=(
                max(0.05, thx * _GRASPABLE_CONTAINER_HALF_SCALE),
                max(0.05, thy * _GRASPABLE_CONTAINER_HALF_SCALE),
            ),
            prim_path=e.prim_path,
            approach_yaw=e.approach_yaw,
            foot_xy=(cx, cy),
            foot_half=(thx, thy),
            top_z=top_z,
            clearance=0.30,
            drop_z=drop_z,
            meta={
                "affordance_bbox": bounds,
                "half_scale": (
                    _GRASPABLE_CONTAINER_HALF_SCALE,
                    _GRASPABLE_CONTAINER_HALF_SCALE,
                ),
                "bottom_inset": inner_z - float(mn[2]),
                "top_inset": 0.05,
                "drop_offset": drop_z - float(mn[2]),
                "open_top": True,
            },
        )
    top = float(mx[2])
    return Entity(
        name=tgt,
        role=Role.SURFACE,
        xy=(cx, cy),
        z=top,
        half=(thx, thy),
        prim_path=e.prim_path,
        approach_yaw=e.approach_yaw,
        foot_xy=(cx, cy),
        foot_half=(thx, thy),
        top_z=top,
        clearance=0.40,
        meta={"affordance_bbox": bounds, "half_scale": (1.0, 1.0)},
    )


def realize_deferred_preloads(scene, stage) -> None:
    rows = list(getattr(scene, "deferred_preloads", None) or [])
    if not rows:
        return
    for row in rows:
        _instantiate_deferred_preload(scene, stage, row)
    attach_graspable_target_affordances(scene, stage)
    pending = list(rows)
    while pending:
        pending_keys = {str(row["key"]) for row in pending}
        ready = [row for row in pending if str(row["target"]) not in pending_keys]
        if not ready:
            raise ValueError(
                f"cyclic deferred preload graph: "
                f"{[(row['key'], row['target']) for row in pending]}"
            )
        by_target: dict[str, list[dict]] = {}
        for row in ready:
            by_target.setdefault(str(row["target"]), []).append(row)
        for target, target_rows in by_target.items():
            _seat_deferred_preloads(scene, stage, target, target_rows)
        ready_ids = {id(row) for row in ready}
        pending = [row for row in pending if id(row) not in ready_ids]
        attach_graspable_target_affordances(scene, stage)
    scene.invalidate_bounds()


def _instantiate_deferred_preload(scene, stage, row: dict) -> None:
    import fpv_scene
    from scene_state import Role

    key = str(row["key"])
    path = f"/World/Obj_{_safe(key)}"
    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid():
        prim = fpv_scene.add_reference(
            stage,
            path,
            baked_usd(f"{row['cat']}/{row['model']}"),
            isolate_xform=True,
        )
    fpv_scene.set_world_pose(
        prim,
        (0.0, 0.0, float(getattr(scene, "floor_z", 0.0)) + 0.5),
        (1.0, 0.0, 0.0, 0.0),
    )
    half = float(row["half"])
    scene.register(
        key,
        Role.GRASPABLE,
        xy=(0.0, 0.0),
        z=float(getattr(scene, "floor_z", 0.0)) + 0.5,
        prim_path=path,
        foot_half=(half, half),
        clearance=0.20,
    )
    grasp_objs = list(getattr(scene, "grasp_objs", None) or [])
    if not any(
        str(item.get("path")) == path for item in grasp_objs if isinstance(item, dict)
    ):
        grasp_objs.append(
            {
                "key": key,
                "path": path,
                "build": behavior_build_params(f"{row['cat']}/{row['model']}", 1.0),
            }
        )
    scene.grasp_objs = grasp_objs


def _deferred_preload_slots(target, rows: list[dict]) -> list[tuple[float, float]]:
    count = len(rows)
    if count == 0:
        return []
    half = max(float(row["half"]) for row in rows)
    edge_margin = 0.01
    min_gap = 0.02
    available_x = 2.0 * (float(target.half[0]) - edge_margin)
    available_y = 2.0 * (float(target.half[1]) - edge_margin)
    step = 2.0 * half + min_gap
    layouts: list[tuple[tuple[float, int, int, int], int, int]] = []
    for cols in range(1, count + 1):
        nrows = int(math.ceil(count / cols))
        width = 2.0 * half + (cols - 1) * step
        height = 2.0 * half + (nrows - 1) * step
        if width <= available_x + 1e-9 and height <= available_y + 1e-9:
            utilization = max(width / available_x, height / available_y)
            layouts.append(
                (
                    (utilization, cols * nrows - count, abs(cols - nrows), cols),
                    cols,
                    nrows,
                )
            )
    if not layouts:
        raise ValueError(
            f"{target.name} is too small for {count} deferred preload(s) "
            f"with {min_gap:.2f}m clearance"
        )
    _, cols, nrows = min(layouts)
    x0 = float(target.xy[0]) - 0.5 * (cols - 1) * step
    y0 = float(target.xy[1]) - 0.5 * (nrows - 1) * step

    return [
        (
            x0 + (index % cols) * step,
            y0 + (index // cols) * step,
        )
        for index in range(count)
    ]


def _seat_deferred_preloads(scene, stage, target_key: str, rows: list[dict]) -> None:
    import fpv_scene
    from scene_state import Role

    target = scene.target_view(target_key, Role.CONTAINER)
    if target is None:
        raise ValueError(
            f"deferred preload target {target_key!r} has no CONTAINER affordance"
        )
    for row, (x, y) in zip(rows, _deferred_preload_slots(target, rows)):
        key = str(row["key"])
        entity = scene.resolve(key)
        prim = stage.GetPrimAtPath(entity.prim_path)
        base_z = float(target.z) + 0.01
        fpv_scene.set_world_pose(prim, (x, y, base_z), (1.0, 0.0, 0.0, 0.0))
        bbox = fpv_scene.compute_world_bbox(prim)
        correction = base_z - float(bbox.GetMin()[2])
        if abs(correction) > 0.002:
            fpv_scene.set_world_pose(
                prim, (x, y, base_z + correction), (1.0, 0.0, 0.0, 0.0)
            )
            bbox = fpv_scene.compute_world_bbox(prim)
        mn, mx = bbox.GetMin(), bbox.GetMax()
        entity.xy = (0.5 * float(mn[0] + mx[0]), 0.5 * float(mn[1] + mx[1]))
        entity.z = float(mn[2])
        entity.foot_xy = entity.xy
        entity.top_z = float(mx[2])
        scene.declare_inside(key, target_key)
        print(f"[behavior-task] deferred preload {key} INSIDE {target_key}")


def build_door_demo(stage, app):
    """MINIMAL standalone open/close demo (no task / no grab): compose a real kitchen, pick up to 2
    FLOOR-level revolute openables (a fridge + a cabinet), register them as openable containers, and walk
    the human up to OPEN then CLOSE each. Shows the GENERAL container-door fix -- every revolute leaf swings
    about its OWN true hinge edge (a 2-door fridge/cabinet opens both doors), not a broken far-pivot arc."""
    import math

    import fpv_scene
    import numpy as np
    from pxr import UsdLux
    from scene_state import Role

    scene, placed, _surfaces, centroid, _floor_top, _dz = _compose_behavior_scene(
        stage, app, "house_double_floor_lower", None, None, None
    )
    cx0, cy0 = float(centroid[0]), float(centroid[1])

    def _ayaw(x, y):
        dx, dy = float(x) - cx0, float(y) - cy0
        n = math.hypot(dx, dy) or 1.0
        return math.degrees(math.atan2(dx / n, -dy / n))

    cands = []
    for nm, (pp, o) in placed.items():
        if o["cat"] not in (
            "fridge",
            "bottom_cabinet_no_top",
            "bottom_cabinet",
            "oven",
            "dishwasher",
            "washer",
        ):
            continue
        of = _openable_leaf_fields(scene, pp, o)
        if not of or of["kind"] != "swing":
            continue
        rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(pp))
        mn, mx = rng.GetMin(), rng.GetMax()
        if float(mn[2]) > scene.floor_z + 0.4:  # wall-mounted -> door too high to reach
            continue
        cands.append((nm, pp, o, of, mn, mx, o["cat"]))
    fridges = [c for c in cands if c[6] == "fridge"]
    cabs = [c for c in cands if c[6] != "fridge"]
    chosen = (fridges[:1] + cabs[:1]) or cands[:2]
    if not chosen:
        raise RuntimeError(
            "door_demo: no floor-level revolute openable found in the scene"
        )

    names = []
    for nm, pp, _o, of, mn, mx, _cat in chosen:
        tcx, tcy = (
            0.5 * (float(mn[0]) + float(mx[0])),
            0.5 * (float(mn[1]) + float(mx[1])),
        )
        thx = max(0.05, 0.5 * (float(mx[0]) - float(mn[0])))
        thy = max(0.05, 0.5 * (float(mx[1]) - float(mn[1])))
        ayaw = _ayaw(tcx, tcy)
        ar = math.radians(ayaw)
        fwd = (
            math.sin(ar),
            -math.cos(ar),
        )  # room -> container; front face is the -fwd side
        hdepth = abs(fwd[0]) * thx + abs(fwd[1]) * thy
        front_face = (tcx - fwd[0] * hdepth, tcy - fwd[1] * hdepth)
        handle_z = min(
            max(scene.floor_z + 1.0, float(mn[2]) + 0.2),
            float(mx[2]) - 0.1,
            scene.floor_z + 1.25,
        )
        scene.register(
            nm,
            Role.CONTAINER,
            xy=front_face,
            z=float(mn[2]) + 0.06,
            height=max(0.15, float(mx[2]) - 0.1 - (float(mn[2]) + 0.06)),
            half=(max(0.05, thx * 0.8), max(0.05, thy * 0.8)),
            prim_path=pp,
            approach_yaw=ayaw,
            foot_xy=(tcx, tcy),
            foot_half=(thx, thy),
            top_z=float(mx[2]),
            clearance=0.35,
            hinge_xy=of["hinge_xy"],
            hinge_axis=(0.0, 0.0, 1.0),
            swing_angle=of["swing_angle"],
            swing_sign=1.0,
            leaf_paths=of["leaf_paths"],
            leaf_hinges=of.get("leaf_hinges", ()),
            joint_name=of["joint_name"],
            open_thresh=0.10,
            handle_xy=front_face,
            handle_z=handle_z,
        )
        names.append(nm)
        print(
            f"[door-demo] openable {nm} ({_cat}) leaves={len(of['leaf_paths'])} handle_z={handle_z:.2f}"
        )

    fnm, fpp, _fo, _fof, f_mn, f_mx, _fc = chosen[0]
    fcx, fcy = (
        0.5 * (float(f_mn[0]) + float(f_mx[0])),
        0.5 * (float(f_mn[1]) + float(f_mx[1])),
    )
    ar = math.radians(_ayaw(fcx, fcy))
    fwd = (math.sin(ar), -math.cos(ar))
    scene.start_xy = (
        fcx - fwd[0] * 1.5,
        fcy - fwd[1] * 1.5,
    )  # 1.5 m in front of the first door
    scene.near_start = True
    scene.tpv_chase = True
    scene.fpv_walk_ahead = 1.6
    scene.fpv_eye_drop = 0.30

    program = ", ".join(f"[Walk] <{n}> (1), [Open] <{n}> (1)" for n in names)
    program += ", " + ", ".join(f"[Walk] <{n}> (1), [Close] <{n}> (1)" for n in names)

    lt = UsdLux.SphereLight.Define(stage, "/World/DoorDemoLight")
    lt.CreateRadiusAttr(0.6)
    lt.CreateIntensityAttr(60000.0)
    fpv_scene.set_world_pose(lt.GetPrim(), (fcx, fcy, 1.9))
    _dome = UsdLux.DomeLight.Define(stage, "/World/DoorDemoDome")
    _dome.CreateIntensityAttr(500.0)
    _dome.CreateColorAttr((0.72, 0.80, 0.92))
    scene.auto_expose = True
    scene.expose_xy = (fcx, fcy)
    scene.expose_z = 1.0
    _register_mdl()

    tpv_look = (fcx, fcy, 1.0)
    tpv_eye = (fcx - fwd[0] * 3.0, fcy - fwd[1] * 3.0, 2.2)
    print(f"[door-demo] containers={names} start={scene.start_xy} program={program}")
    return scene, {}, program, tpv_look, tpv_eye


def _is_horizontal_top_lid(container_bbox, leaf_bbox) -> bool:
    """Whether one revolute leaf is a thin panel covering the container top."""
    cmn, cmx = container_bbox
    lmn, lmx = leaf_bbox
    cext = [max(1e-6, float(cmx[i]) - float(cmn[i])) for i in range(3)]
    lext = [max(0.0, float(lmx[i]) - float(lmn[i])) for i in range(3)]
    leaf_center_z = 0.5 * (float(lmn[2]) + float(lmx[2]))
    near_top = leaf_center_z >= float(cmx[2]) - max(0.12, 0.20 * cext[2])
    covers_top = lext[0] >= 0.55 * cext[0] and lext[1] >= 0.55 * cext[1]
    thin = lext[2] <= max(0.12, 0.25 * min(lext[0], lext[1]))
    return bool(near_top and covers_top and thin)


def _closing_leaf_fields(scene, pp, o) -> dict | None:
    """Task-local LAX mirror of ``_openable_leaf_fields`` for the restaurant-closing builder: identical
    logic + return shape, but a revolute base_link hinge is accepted when its spec ``axis`` is ``"Z"`` OR
    ``""`` (blank). The hotel kitchen fridges (fridge/jtqazu) record a revolute base_link->link_0 joint
    with a BLANK axis, which the shared strict reader (axis=="Z" only) rejects; a fridge/cabinet door hinge
    IS a vertical Z line, so treat a blank axis as a vertical Z hinge. Everything else -- leaf lookup,
    2-door collection, bbox, prismatic fallback -- is the same, so ``_register_swing`` consumes the dict
    unchanged. Additive: the shared strict reader is untouched."""
    from pxr import Usd, UsdGeom

    if not o or not o.get("cat") or not o.get("model"):
        return None
    try:
        spec = read_spec(f"{o['cat']}/{o['model']}")
    except Exception:  # noqa: BLE001
        return None
    joints = spec.get("joints", []) or []
    revs = list(_container_revolute_joints(spec, accept_blank_vertical=True))
    pri = (
        None
        if revs
        else next((j for j in joints if j.get("type") == "prismatic"), None)
    )
    if not revs and pri is None:
        return None
    droot = scene.stage.GetPrimAtPath(pp)

    def _leaf_prim(jt):
        nm = str(jt["body1"]).rstrip("/").split("/")[-1]
        return next(
            (
                p
                for p in Usd.PrimRange(droot, Usd.TraverseInstanceProxies())
                if p.GetName() == nm
            ),
            None,
        )

    def _bbox(prim):
        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        lmn, lmx = rng.GetMin(), rng.GetMax()
        return (
            (float(lmn[0]), float(lmn[1]), float(lmn[2])),
            (float(lmx[0]), float(lmx[1]), float(lmx[2])),
        )

    if revs:
        leaves = [(_leaf_prim(j), j) for j in revs]
        leaves = [(lf, j) for lf, j in leaves if lf is not None]
        if not leaves:
            return None
        j0 = leaves[0][1]
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        leaf_origins = tuple(
            xform_cache.GetLocalToWorldTransform(leaf).ExtractTranslation()
            for leaf, _joint in leaves
        )
        front_flip = all(joint.get("axis") in ("X", "Y") for _, joint in leaves)
        leaf_hinges = tuple(
            (
                (float(origin[0]), float(origin[1]), float(origin[2]))
                if front_flip
                else (float(origin[0]), float(origin[1]))
            )
            for origin in leaf_origins
        )
        leaf_bboxes = tuple(_bbox(lf) for lf, _ in leaves)
        leaf_bbox = leaf_bboxes[0]
        fields = {
            "kind": "swing",
            "hinge_xy": leaf_hinges[0][:2],
            "leaf_hinges": leaf_hinges,
            "swing_angle": math.radians(min(float(j0.get("upper", 90.0)), 88.0)),
            "leaf_paths": tuple(str(lf.GetPath()) for lf, _ in leaves),
            "leaf_bboxes": leaf_bboxes,
            "joint_name": j0["name"],
            "leaf_bbox": leaf_bbox,
            "horizontal_top_lid": len(leaves) == 1
            and _is_horizontal_top_lid(_bbox(droot), leaf_bbox),
        }
        if front_flip:
            fields.update(front_flip=True, hinge_xyz=leaf_hinges[0])
        return fields
    leaf = _leaf_prim(pri)
    if leaf is None:
        return None
    return {
        "kind": "slide",
        "travel": float(min(max(float(pri.get("upper", 0.30)), 0.15), 0.45)),
        "leaf_paths": (str(leaf.GetPath()),),
        "joint_name": pri["name"],
        "leaf_bbox": _bbox(leaf),
    }


def _declared_openable_joints(spec: dict) -> tuple[dict, ...]:
    """Return only joints explicitly identified by the asset as openable."""
    names = {
        str(row[-1])
        for row in (spec.get("openable_joint_ids") or ())
        if isinstance(row, (list, tuple)) and row
    }
    return tuple(
        joint
        for joint in (spec.get("joints") or ())
        if str(joint.get("name") or "") in names and joint.get("body1")
    )


def _static_open_leaf_fields(scene, pp, o) -> dict | None:
    """Geometry of leaves an asset explicitly marks openable, independent of hinge axis.

    A dishwasher door rotates about a horizontal axis, while its racks translate.  The generic
    closing-door reader intentionally accepts only vertical hinges and therefore identifies the
    first rack as its prismatic fallback.  Static-open containers do not animate a hinge, so their
    authoritative leaf is instead the one named by ``openable_joint_ids``.
    """
    from pxr import Usd, UsdGeom

    if not o or not o.get("cat") or not o.get("model"):
        return None
    try:
        spec = read_spec(f"{o['cat']}/{o['model']}")
    except Exception:  # noqa: BLE001
        return None
    joints = _declared_openable_joints(spec)
    if not joints:
        return None
    root = scene.stage.GetPrimAtPath(pp)

    def _leaf_prim(joint):
        name = str(joint["body1"]).rstrip("/").split("/")[-1]
        return next(
            (
                prim
                for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if prim.GetName() == name
            ),
            None,
        )

    def _bbox(prim):
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
        return (
            tuple(float(minimum[axis]) for axis in range(3)),
            tuple(float(maximum[axis]) for axis in range(3)),
        )

    leaves = [(_leaf_prim(joint), joint) for joint in joints]
    leaves = [(leaf, joint) for leaf, joint in leaves if leaf is not None]
    if not leaves:
        return None
    boxes = tuple(_bbox(leaf) for leaf, _joint in leaves)
    return {
        "kind": "static",
        "leaf_paths": tuple(str(leaf.GetPath()) for leaf, _joint in leaves),
        "joint_name": str(leaves[0][1]["name"]),
        "leaf_bbox": boxes[0],
        "leaf_bboxes": boxes,
    }


def build_behavior_closing(stage, app, scene_name="restaurant_hotel", task_mod=None):
    """Benchmark Task 1 -- restaurant "closing: break down the line safely" as a MULTI-ROOM scene.

    Consumes the single-source-of-truth data tables in ``constraint_evaluator`` (ITEMS / CONTAINERS /
    ANSWER_KEY / build_task_fields) and stages them across the three connected zones of ``restaurant_hotel``:
      * DINING   -- bus the tables: dishes/glass -> bus tub; scraps/napkin bussed on to the kitchen.
      * KITCHEN  -- put the cold-chain away in the real openable fridges + sort waste / used-oil.
      * CORRIDOR -- return cleaning chemicals to the cabinet, in a different room from the food.
    Items are seated on the COLLISION solid of real breakfast / commercial-kitchen tables per their
    zone/place; imported bins/cabinet are sited on open floor in their own room via a clearance slot
    search; the two cold units reuse the kitchen's real revolute fridges (the jtqazu doors have a BLANK
    spec axis that only ``_closing_leaf_fields`` accepts). Per-item + NEGATIVE scoring comes from
    ``constraint_evaluator.build_task_fields()`` via ``scene.behavior_task`` (goals / neg_constraints), read by
    episode_runtime's gated scorer. Additive: a NEW multi-room builder; existing scenes byte-identical. Returns
    the step3 5-tuple (scene, {}, program, tpv_look, tpv_eye). Data table: simulator.physical_world/constraint_evaluator.py."""
    import math
    import random

    import constraint_evaluator
    import fpv_scene
    import numpy as np
    from pxr import UsdLux
    from scene_state import Role

    # The data tables are a PARAMETER, not a hard import: a sibling task (waste_task) drives this whole
    # builder with a different ITEMS/CONTAINERS set. None = Task 1, byte-identical to before.
    rt = task_mod or constraint_evaluator

    # deterministic scatter/dressing RNG: a fixed seed -> the "messy" layout is identical every build
    # (reproducible --check/--step3/render), it just doesn't look like a tidy grid.
    _rng = random.Random(20240720)

    # This task is multi-room and fixed to rt.SCENE (restaurant_hotel). The episode_runtime registry entry
    # still passes the legacy single-room "restaurant_diner" positionally (leftover from the diner version);
    # redirect it so --scene bt_restaurant_closing composes the correct multi-room scene WITHOUT editing
    # episode_runtime. A caller that passes a real multi-room scene is honoured unchanged.
    if scene_name in (None, "restaurant_diner"):
        scene_name = rt.SCENE

    scene, placed, _surfaces, centroid, _floor_top, _dz = _compose_behavior_scene(
        stage, app, scene_name, None, None, None
    )
    cx0, cy0 = float(centroid[0]), float(centroid[1])

    # Room anchors (world xy; x,y are unaffected by the floor-top dz shift). Used to filter furniture by
    # room and to site the imported containers in their own room. From the verified restaurant_hotel layout.
    ROOM = {"dining": (-1.5, -10.6), "kitchen": (6.7, -0.1), "corridor": (3.5, -3.8)}

    def _wbbox(pp):
        rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(pp))
        mn, mx = rng.GetMin(), rng.GetMax()
        return (
            (float(mn[0]), float(mn[1]), float(mn[2])),
            (float(mx[0]), float(mx[1]), float(mx[2])),
        )

    def _ctr(pp):
        (mnx, mny, _z), (mxx, mxy, _z2) = _wbbox(pp)
        return (0.5 * (mnx + mxx), 0.5 * (mny + mxy))

    def _yaw_from(anchor, x, y):
        # Face a target from the ROOM interior (open side): body fwd = (sin,-cos) points anchor -> target,
        # so the human stands on the anchor side and reaches across the near edge.
        dx, dy = float(x) - anchor[0], float(y) - anchor[1]
        n = math.hypot(dx, dy) or 1.0
        return math.degrees(math.atan2(dx / n, -dy / n))

    # ---- inter-room doors stay ON: the human OPENS each doorway on its path (option A) -----------------
    # Doors remain registered by _compose_behavior_scene (scene.doors / door_prims), so the runtime auto-open
    # pre-pass opens each doorway the planned path crosses. Door-OPEN was RELAXED at the engine level
    # (semantic_executor._swing relax_reach for doors + episode_runtime._runtime_open_doors aims the door stand
    # at the doorway CENTER-FRONT, not the wall-plane handle), so a door opens from a clean, reachable doorway
    # stand + swings kinematically even when its handle is jammed in a wall corner (e.g. door_ubesbe_0's
    # over-wide leaf). An over-wide leaf may clip the wall on swing -- accepted (FPV-first; clipping ignored).
    print(
        f"[closing] inter-room doors ON: {len(scene.doors or {})} auto-open doorway(s)"
    )

    # ---- helpers to pick a clear, connected approach side -------------------------------------------
    CORRIDOR_ENTRY = (
        3.5,
        -4.4,
    )  # where the corridor opens into the dining room (the nav connection)
    _boxes0 = (
        scene.solid_boxes()
    )  # scene furniture footprints (items/imports not registered yet)

    def _clear_at(x, y, exclude):
        best = 1e9
        for nm, mn, mx in _boxes0:
            if nm == exclude:
                continue
            dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
            dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
            best = min(best, math.hypot(dx, dy))
        return best

    def _best_side(table_pp, self_name, toward):
        # Approach side whose front-STAND is clear (>=0.5 m open floor) and, among those, nearest `toward`
        # (the connected corridor entry). Dining tables are chair-boxed on 2-3 sides, so the naive
        # room-centroid side lands the stand in an islanded pocket. Returns (approach_yaw, clearance, dist).
        cb = _collision_bbox(stage, table_pp)
        (cmn, cmx) = cb if cb is not None else _wbbox(table_pp)
        cx, cy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
        hx, hy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
        best = None
        for ang in (0.0, 90.0, 180.0, 270.0):
            a = math.radians(ang)
            fwd = (
                math.sin(a),
                -math.cos(a),
            )  # approach dir INTO the table; human stands on the -fwd side
            hd = abs(fwd[0]) * hx + abs(fwd[1]) * hy
            sx, sy = cx - fwd[0] * (hd + 0.55), cy - fwd[1] * (hd + 0.55)
            cl = _clear_at(sx, sy, self_name)
            dist = math.hypot(sx - toward[0], sy - toward[1])
            rank = (1 if cl >= 0.5 else 0, -dist if cl >= 0.5 else cl)
            if best is None or rank > best[0]:
                best = (rank, float(ang), float(cl), float(dist))
        return best[1], best[2], best[3]

    def _tables_near(cat, anchor, n, sep):
        cands = [(nm, pp) for nm, (pp, o) in placed.items() if o["cat"] == cat]
        cands.sort(
            key=lambda t: math.hypot(
                _ctr(t[1])[0] - anchor[0], _ctr(t[1])[1] - anchor[1]
            )
        )
        picked: list = []
        for nm, pp in cands:
            c = _ctr(pp)
            if all(
                math.hypot(c[0] - _ctr(q[1])[0], c[1] - _ctr(q[1])[1]) >= sep
                for q in picked
            ):
                picked.append((nm, pp))
            if len(picked) >= n:
                break
        return picked

    # DINING tables: rank breakfast_tables by their best-side front-stand CLEARANCE (most open floor first,
    # so the runtime stand is non-penetrating), tie-broken by nearest the corridor entry. The dense wnrior
    # grid is chair-boxed on 2-3 sides (~0.7 m best stand -> the body penetrates a chair), while the isolated
    # qjiray table is ~1 m clear. Take the 2 clearest, well separated.
    _dcands = []
    for nm, (pp, o) in placed.items():
        if o["cat"] != "breakfast_table":
            continue
        yaw, cl, dist = _best_side(pp, nm, CORRIDOR_ENTRY)
        _dcands.append((cl, -dist, nm, pp, yaw))
    _dcands.sort(
        key=lambda t: (t[0], t[1]), reverse=True
    )  # highest clearance first, then nearest entry
    dining_tables = []  # (name, pp, approach_yaw)
    for cl, _nd, nm, pp, yaw in _dcands:
        c = _ctr(pp)
        if all(
            math.hypot(c[0] - _ctr(q[1])[0], c[1] - _ctr(q[1])[1]) >= 1.6
            for q in dining_tables
        ):
            dining_tables.append((nm, pp, yaw))
            print(
                f"[closing] dining table {nm}: best-side clearance>={cl:.2f}m approach_yaw={yaw:.0f}"
            )
        if len(dining_tables) >= 2:
            break
    if not dining_tables:
        raise RuntimeError("closing: no breakfast_table in the dining room")

    # KITCHEN: prep line by the fridges; counter = a distinct kitchen table >=2.5 m away (all connect).
    kt = _tables_near("commercial_kitchen_table", (4.5, -0.6), 1, 0.0)
    if not kt:
        raise RuntimeError(
            "closing: no commercial_kitchen_table (prep line) in the kitchen"
        )
    prep_nm, prep_pp = kt[0]
    counter_cands = [
        (nm, pp)
        for nm, (pp, o) in placed.items()
        if o["cat"] == "commercial_kitchen_table" and nm != prep_nm
    ]
    counter_cands.sort(
        key=lambda t: (
            math.hypot(
                _ctr(t[1])[0] - ROOM["kitchen"][0], _ctr(t[1])[1] - ROOM["kitchen"][1]
            )
            if math.hypot(
                _ctr(t[1])[0] - _ctr(prep_pp)[0], _ctr(t[1])[1] - _ctr(prep_pp)[1]
            )
            >= 2.5
            else 1e9
        )
    )
    counter_pp = counter_cands[0][1] if counter_cands else prep_pp

    # ---- seat the graspable ITEMS on their table's COLLISION solid, rowed along the approach edge -------
    dining_items = [k for k in rt.ITEMS if rt.ITEMS[k]["place"] == "table"]
    prep_items = [k for k in rt.ITEMS if rt.ITEMS[k]["place"] == "prepline"]
    counter_items = [k for k in rt.ITEMS if rt.ITEMS[k]["place"] == "counter"]

    scene.grasp_objs = []
    for k in rt.ITEMS:
        cat, model = rt.ITEMS[k]["cat"], rt.ITEMS[k]["model"]
        scene.grasp_objs.append(
            {
                "key": k,
                "path": f"/World/Obj_{_safe(k)}",
                "build": behavior_build_params(f"{cat}/{model}", 1.0),
            }
        )

    support_prims: list = []

    def _seat_on(table_pp, keys, yaw):
        # Place items over the table's COLLISION solid (not the visual edge, which can overhang open floor
        # -> the object falls through at settle), rowed along the near edge on the approach (yaw) side so
        # each is a short straight reach from a stand IN FRONT of it.
        cb = _collision_bbox(stage, table_pp)
        (cmn, cmx) = cb if cb is not None else _wbbox(table_pp)
        s_cx, s_cy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
        s_hx, s_hy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
        s_top = float(cmx[2])
        ar = math.radians(yaw)
        fwd = (
            math.sin(ar),
            -math.cos(ar),
        )  # approach dir INTO the table; near edge is on the -fwd side
        perp = (-fwd[1], fwd[0])
        edge_depth = min(s_hx / max(abs(fwd[0]), 1e-6), s_hy / max(abs(fwd[1]), 1e-6))
        inset = max(
            0.08, edge_depth * 0.45
        )  # mid-way to the near edge -> stay ON the solid
        ex, ey = s_cx - fwd[0] * inset, s_cy - fwd[1] * inset
        n_i = len(keys)
        perp_half = abs(s_hx * perp[0]) + abs(s_hy * perp[1])
        # SIZE-AWARE uniform pitch: at least the original 0.20 m, but WIDEN to the largest neighbour pair so
        # big proteins never interpenetrate (the old fixed-0.20 cap made chicken/steak overlap + settle off).
        # Same centered formula + inset line as the known-good v1 -> dining (small items -> 0.20) is
        # placement-identical/known-reachable; only crowded big items get more room. No positional jitter:
        # the dining tables are a cramped chair grid where a shifted stand penetrates a chair. The messy
        # just-closed look comes from the (static, reach-free) DECOR, not from scattering the task items.
        halves = []
        for k in keys:
            try:
                _bb = read_spec(f"{rt.ITEMS[k]['cat']}/{rt.ITEMS[k]['model']}").get(
                    "bbox_size"
                ) or [0.12, 0.12, 0.1]
            except Exception:  # noqa: BLE001
                _bb = [0.12, 0.12, 0.1]
            halves.append(0.5 * max(float(_bb[0]), float(_bb[1])))
        need_pitch = (
            max(halves[i] + halves[i + 1] for i in range(n_i - 1)) if n_i > 1 else 0.0
        ) + 0.04
        spacing = max(0.20, need_pitch)
        for i, k in enumerate(keys):
            off = (i - (n_i - 1) / 2.0) * spacing
            sx, sy = ex + perp[0] * off, ey + perp[1] * off
            scene.register(
                k,
                Role.GRASPABLE,
                xy=(float(sx), float(sy)),
                z=float(s_top) + 0.05,
                prim_path=f"/World/Obj_{_safe(k)}",
                approach_yaw=float(yaw),
                foot_xy=(float(sx), float(sy)),
                foot_half=(0.12, 0.12),
                top_z=float(s_top),
                clearance=0.34,
            )
        if table_pp not in support_prims:
            support_prims.append(table_pp)

    def _kyaw(pp):
        c = _ctr(pp)
        return _yaw_from(ROOM["kitchen"], c[0], c[1])

    # distribute the dining items across the chosen dining tables (round-robin -> no table overloaded)
    dt_groups: dict = {i: [] for i in range(len(dining_tables))}
    for i, k in enumerate(dining_items):
        dt_groups[i % len(dining_tables)].append(k)
    for i, (_nm, tpp, tyaw) in enumerate(dining_tables):
        if dt_groups[i]:
            _seat_on(tpp, dt_groups[i], tyaw)
    if counter_pp != prep_pp:
        _seat_on(prep_pp, prep_items, _kyaw(prep_pp))
        _seat_on(counter_pp, counter_items, _kyaw(counter_pp))
    else:  # only one kitchen table -> row prep + counter items together on it
        _seat_on(prep_pp, prep_items + counter_items, _kyaw(prep_pp))
    scene.behavior_support_prims = list(support_prims)

    # Start the human in front of the prep line (kitchen hub -> fridges, bins, the corridor door and,
    # through the now-openable corridor, the dining room). The re-aimed corridor door opens from either
    # side, so the first cross-room walk crosses it. place_start snaps this hint to the nearest free cell.
    _pmn, _pmx = _collision_bbox(stage, prep_pp) or _wbbox(prep_pp)
    _pcx, _pcy = 0.5 * (_pmn[0] + _pmx[0]), 0.5 * (_pmn[1] + _pmx[1])
    _phx, _phy = 0.5 * (_pmx[0] - _pmn[0]), 0.5 * (_pmx[1] - _pmn[1])
    _pyaw = _yaw_from(ROOM["kitchen"], _pcx, _pcy)
    _pfwd = (math.sin(math.radians(_pyaw)), -math.cos(math.radians(_pyaw)))
    _phd = abs(_pfwd[0]) * _phx + abs(_pfwd[1]) * _phy
    scene.start_xy = (
        float(_pcx - _pfwd[0] * (_phd + 0.75)),
        float(_pcy - _pfwd[1] * (_phd + 0.75)),
    )
    scene.near_start = True

    # ---- container registration helpers ---------------------------------------------------------------
    def _best_yaw(ux, uy, boxes):
        # openest cardinal side (max clearance ~1 m out): the door-facing for an imported swing unit / the
        # approach for an open bin.
        def _cd(x, y):
            best = 1e9
            for _n, mn, mx in boxes:
                dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
                dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
                best = min(best, math.hypot(dx, dy))
            return best

        best_a, best_c = 0.0, -1.0
        for a_deg in (0.0, 90.0, 180.0, 270.0):
            a = math.radians(a_deg)
            c = _cd(ux - math.sin(a) * 1.0, uy + math.cos(a) * 1.0)
            if c > best_c:
                best_a, best_c = a_deg, c
        return best_a

    _LABEL_DIR = GENERATED_LABEL_DIR
    front_dir: dict = {}  # reg_name -> unit-centre->door/approach unit vector (where a label sign faces)
    reg_pp: dict = {}  # reg_name -> prim path (for the --preview snapshot cameras)

    def _register_swing(reg_name, pp, obj):
        of = _closing_leaf_fields(scene, pp, obj)
        (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(pp)
        tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        thx, thy = max(0.05, 0.5 * (mxx - mnx)), max(0.05, 0.5 * (mxy - mny))
        if not of or of.get("kind") != "swing":
            return False
        # door side from the ACTUAL swinging leaf: front_dir = unit center -> leaf center; the human stands
        # on the door side facing INTO the unit (fwd = -front_dir).
        if of.get("leaf_bbox"):
            (_lmn, _lmx) = of["leaf_bbox"]
            fdx, fdy = 0.5 * (_lmn[0] + _lmx[0]) - tcx, 0.5 * (_lmn[1] + _lmx[1]) - tcy
            _fn = math.hypot(fdx, fdy) or 1.0
            fdir = (fdx / _fn, fdy / _fn)
        else:
            _a0 = math.radians(_best_yaw(tcx, tcy, scene.solid_boxes()))
            fdir = (-math.sin(_a0), math.cos(_a0))
        ayaw = math.degrees(math.atan2(-fdir[0], fdir[1]))
        front_dir[reg_name] = fdir  # door side -> where the label sign is mounted
        hd = abs(fdir[0]) * thx + abs(fdir[1]) * thy
        front_face = (tcx + fdir[0] * hd, tcy + fdir[1] * hd)
        inset = min(0.22, max(0.0, hd - 0.05))
        front_int = (tcx + fdir[0] * (hd - inset), tcy + fdir[1] * (hd - inset))
        _lo, _hi = float(mnz) + 0.08, float(mxz) - 0.12
        _clampz = lambda v: min(max(v, min(_lo, _hi)), max(_lo, _hi))  # noqa: E731
        shelf_z = _clampz(
            scene.floor_z + 0.90
        )  # reachable interior drop height (tall fridge / cabinet)
        handle_z = _clampz(scene.floor_z + 1.00)
        inner_z = float(mnz) + 0.06
        inner_h = max(0.15, (float(mxz) - 0.1) - inner_z)
        shv = _extract_spaces(
            scene.stage, pp, reg_name
        )  # distinct interior shelves (top-down, L/R split)
        scene.register(
            reg_name,
            Role.CONTAINER,
            xy=front_int,
            z=inner_z,
            height=inner_h,
            half=(max(0.05, thx * 0.8), max(0.05, thy * 0.8)),
            prim_path=pp,
            approach_yaw=ayaw,
            foot_xy=(tcx, tcy),
            foot_half=(thx, thy),
            top_z=float(mxz),
            clearance=0.35,
            hinge_xy=of["hinge_xy"],
            hinge_axis=(0.0, 0.0, 1.0),
            swing_angle=of["swing_angle"],
            swing_sign=1.0,
            leaf_paths=of["leaf_paths"],
            leaf_hinges=of.get("leaf_hinges", ()),
            joint_name=of["joint_name"],
            open_thresh=0.10,
            handle_xy=front_face,
            handle_z=handle_z,
            drop_z=shelf_z,
            spaces=shv,
        )
        if shv:
            print(
                f"[closing] swing container {reg_name}: {len(shv)} interior space(s) "
                f"{[s[0] for s in shv]} leaves={len(of['leaf_paths'])}"
            )
        return True

    def _register_open(reg_name, pp, cat, model):
        (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(pp)
        hx, hy = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
        tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        band = 0.05 * max(
            0.1, float(mxz) - float(mnz)
        )  # match _relations_inside_live's z-band
        inner_z = float(mnz) + 0.04
        inner_h = max(0.10, (float(mxz) - 0.03) - inner_z)
        # release just below the rim: reachable (near the top edge) AND the object's base ends up inside the
        # outer bbox z-band so _relations_inside_live scores INSIDE. Clamp just off the floor for a flat tray.
        drop_z = max(float(scene.floor_z) + 0.02, float(mxz) - band - 0.03)
        _ay = _best_yaw(tcx, tcy, scene.solid_boxes())
        _aar = math.radians(_ay)
        front_dir[reg_name] = (
            -math.sin(_aar),
            math.cos(_aar),
        )  # openest/person side -> label faces there
        scene.register(
            reg_name,
            Role.CONTAINER,
            xy=(tcx, tcy),
            z=inner_z,
            height=inner_h,
            half=(max(0.05, hx * 0.85), max(0.05, hy * 0.85)),
            prim_path=pp,
            approach_yaw=_ay,
            foot_xy=(tcx, tcy),
            foot_half=(float(hx), float(hy)),
            top_z=float(mxz),
            clearance=0.30,
            drop_z=float(drop_z),
        )
        try:
            _static_open_container(stage, pp, cat, model)
        except Exception as _exc:  # noqa: BLE001
            print(f"[closing] static-open lid skipped for {reg_name}: {_exc}")

    def _label_decal(reg_name, pp, sign, tall):
        # mount a REAL 2D LABEL SIGN (a PNG under _LABEL_DIR) FLAT on the unit's door/front -- NOT a floating
        # block. A textured quad facing the approach/door side, self-lit (emissive) so it reads in dim light.
        # Swap the PNG (raw/dairy/compost/recycle/oil/chem) to restyle; a real photo works too.
        if not sign:
            return
        png = f"{_LABEL_DIR}/{sign}.png"
        if not os.path.exists(png):
            print(f"[closing] label sign missing, skipped: {png}")
            return
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        try:
            (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(pp)
        except Exception:  # noqa: BLE001
            return
        cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        hx, hy = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
        fd = front_dir.get(reg_name)
        if not fd or (abs(fd[0]) + abs(fd[1]) < 1e-6):
            a = math.radians(_best_yaw(cx, cy, scene.solid_boxes()))
            fd = (-math.sin(a), math.cos(a))
        if tall:
            # fridge: mount on the DOOR (approach) side
            nrm, sw, sh_ = fd, 0.46, 0.30
            zc = float(scene.floor_z) + 1.32
        else:
            _sf = station_front.get(reg_name)
            if _sf:  # a tucked-against-a-unit bin: label on its OUT-facing side (toward the aisle)
                if abs(_sf[0]) >= abs(_sf[1]):
                    nrm, face_w = (1.0 if _sf[0] >= 0.0 else -1.0, 0.0), 2.0 * hy
                else:
                    nrm, face_w = (0.0, 1.0 if _sf[1] >= 0.0 else -1.0), 2.0 * hx
            elif (2.0 * hx) >= (2.0 * hy):  # else: broadest face turned toward the room
                nrm, face_w = (0.0, 1.0 if fd[1] >= 0.0 else -1.0), 2.0 * hx
            else:
                nrm, face_w = (1.0 if fd[0] >= 0.0 else -1.0, 0.0), 2.0 * hy
            sw = min(0.85 * face_w, 0.5)
            sh_ = sw * 0.62
            zc = float(mnz) + 0.58 * (float(mxz) - float(mnz))
        reach = abs(nrm[0]) * hx + abs(nrm[1]) * hy
        px, py = cx + nrm[0] * (reach + 0.02), cy + nrm[1] * (reach + 0.02)
        ux, uy = -nrm[1], nrm[0]  # in-plane horizontal across the (wider) face
        hw = sw * 0.5
        corners = [
            Gf.Vec3f(float(px - ux * hw), float(py - uy * hw), float(zc - sh_ * 0.5)),
            Gf.Vec3f(float(px + ux * hw), float(py + uy * hw), float(zc - sh_ * 0.5)),
            Gf.Vec3f(float(px + ux * hw), float(py + uy * hw), float(zc + sh_ * 0.5)),
            Gf.Vec3f(float(px - ux * hw), float(py - uy * hw), float(zc + sh_ * 0.5)),
        ]
        nrm3 = Gf.Vec3f(float(fd[0]), float(fd[1]), 0.0)
        # Record the sign's WORLD placement on the container entity so the non-grabbable front-read look_at aims
        # SQUARELY at it -- outward normal (xy) + world CENTRE + size. The face is `nrm`, NOT `fd`: the quad's
        # own plane normal is (ux,uy,0) x (0,0,1) == nrm, and px/py are pushed OUT along +nrm, so +nrm is the
        # side a reader stands on. (`fd` is only the shading normal above; on an open bin the broadest-face pick
        # can put it up to 90deg off the panel, which would send the reader to an edge-on stand.) Recorded HERE,
        # BEFORE the fridge branch below rewrites `corners` into the door leaf's LOCAL frame -- the consumer
        # wants world metres. The leaf is CLOSED at build time, so for a swing unit this is the sign's pose on
        # the SHUT door; the decal then rides the leaf, so the recorded pose is stale while the door is open.
        _ce = scene.entities.get(reg_name)
        if _ce is not None:
            _ce.meta["label_face"] = (float(nrm[0]), float(nrm[1]))
            _ce.meta["label_pos"] = (float(px), float(py), float(zc))
            # zoom target = the sign's WIDEST dimension, so the look-at frames the WHOLE sign and the unit stays
            # visible around it (reads as ON the unit); the small dimension over-zooms until the sign overflows.
            _ce.meta["label_span"] = float(max(sw, sh_))
        # A fridge label RIDES its door leaf: parented UNDER the leaf it swings open WITH the door (so it
        # never sits in the door OPENING), and -- riding a BEHAVIOR asset that gets its collider only from
        # its own /collisions/ shells -- it is never rebuilt into a phantom boundingCube blocking the doorway
        # (the bug that made the reaching arm clip in from the side). Falls back to a /World decal if no leaf.
        leaf_path = None
        if tall:
            _e = scene.entities.get(reg_name)
            _lp = _e.leaf_paths[0] if (_e and getattr(_e, "leaf_paths", None)) else None
            _root = stage.GetPrimAtPath(pp) if _lp else None
            if _root and _root.IsValid():

                def _deinst(pm):  # de-instance so the leaf sub-prim is authorable
                    if pm.IsInstanceable():
                        pm.SetInstanceable(False)
                    for _c in pm.GetChildren():
                        _deinst(_c)

                _deinst(_root)
                _lname = _lp.rstrip("/").split("/")[-1]
                _leaf = next(
                    (p for p in Usd.PrimRange(_root) if p.GetName() == _lname), None
                )
                if _leaf is not None:
                    leaf_path = str(_leaf.GetPath())
                    _linv = (
                        UsdGeom.XformCache(Usd.TimeCode.Default())
                        .GetLocalToWorldTransform(_leaf)
                        .GetInverse()
                    )
                    corners = [Gf.Vec3f(_linv.Transform(Gf.Vec3d(c))) for c in corners]
                    nrm3 = Gf.Vec3f(
                        _linv.TransformDir(Gf.Vec3d(float(fd[0]), float(fd[1]), 0.0))
                    )
        path = (
            f"{leaf_path}/Label_{_safe(reg_name)}"
            if leaf_path
            else f"/World/Label_{_safe(reg_name)}"
        )
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(corners)
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([nrm3] * 4)
        mesh.SetNormalsInterpolation("faceVarying")
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        mtl = UsdShade.Material.Define(stage, path + "/M")
        rdr = UsdShade.Shader.Define(stage, path + "/M/St")
        rdr.CreateIdAttr("UsdPrimvarReader_float2")
        rdr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        rdr.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, path + "/M/T")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            rdr.ConnectableAPI(), "result"
        )
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh = UsdShade.Shader.Define(stage, path + "/M/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        # A decal is a pure visual sign, never a physical obstacle -> disable collision so the init
        # collider pass cannot rebuild it into a boundingCube (this is the guard for the /World-level bin
        # labels, and a belt-and-suspenders for the leaf-mounted fridge labels).
        from pxr import UsdPhysics

        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)

    def _recolor_body(pp, color):
        # bind a flat matte color to every gprim under an imported container (overrides its own material), so
        # a generic tote reads unmistakably as the RED chemical caddy and not another gray bus tub.
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        root = stage.GetPrimAtPath(pp)
        if not root or not root.IsValid():
            return
        mtl = UsdShade.Material.Define(stage, pp + "/BodyMat")
        sh = UsdShade.Shader.Define(stage, pp + "/BodyMat/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(
                    mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )

    # ---- scene containers: 2 real openable fridges (cold units) + the sink (negative-only) ------------
    # Both are SCENE-sourced, i.e. they exist only because the task spec claims them. A task that declares
    # no ranked fridge and no sink (waste_task) skips this whole block rather than failing on furniture it
    # never asked for; Task 1 declares both, so nothing changes for it.
    _wants_cold = len(rt.pick_order("fridge")) == 2
    _wants_sink = "sink" in rt.CONTAINERS
    fridge_cands = []
    for nm, (pp, o) in placed.items():
        if o.get("cat") != "fridge":
            continue
        of = _closing_leaf_fields(scene, pp, o)
        if of and of.get("kind") == "swing":
            c = _ctr(pp)
            fridge_cands.append(
                (
                    nm,
                    pp,
                    math.hypot(c[0] - ROOM["kitchen"][0], c[1] - ROOM["kitchen"][1]),
                )
            )
    # TOTAL order: distance first, then the scene instance name. Without the name tie-break, two equidistant
    # fridges are separated only by `placed` insertion order, which no spec states and no rebuild guarantees.
    fridge_cands.sort(key=lambda t: (t[2], t[0]))
    chosen_fridges: list = []
    for nm, pp, _d in fridge_cands:
        c = _ctr(pp)
        if all(
            math.hypot(c[0] - _ctr(q[1])[0], c[1] - _ctr(q[1])[1]) >= 1.2
            for q in chosen_fridges
        ):
            chosen_fridges.append((nm, pp))
        if len(chosen_fridges) >= 2:
            break
    if _wants_cold and len(chosen_fridges) < 2:
        raise RuntimeError(
            f"closing: need 2 openable kitchen fridges, found {len(chosen_fridges)} "
            f"(candidates={[c[0] for c in fridge_cands]})"
        )
    # Rank -> appliance mapping comes from the SPEC (rt.CONTAINERS[...]["pick"]["rank"]), so the fridge that
    # carries the RAW answer key is stated in the task data rather than implied by this loop's literal order.
    _cold_by_rank = rt.pick_order("fridge") if _wants_cold else []
    for reg_name, (snm, spp) in zip(_cold_by_rank, chosen_fridges):
        scene.entities.pop(
            snm, None
        )  # drop the SURFACE dup; re-register the real fridge as CONTAINER
        if not _register_swing(reg_name, spp, placed[snm][1]):
            raise RuntimeError(
                f"closing: fridge {snm} has no drivable swing leaf for {reg_name}"
            )
        _label_decal(reg_name, spp, rt.CONTAINERS[reg_name].get("sign"), True)
        reg_pp[reg_name] = spp
        print(f"[closing] cold unit {reg_name} = {snm} @ {_ctr(spp)}")

    # sink: register the kitchen sink nearest the work area as an open CONTAINER (so "no oil in the sink"
    # is scorable). No program action targets it, so it needs no reachable stand.
    sink_cands = [
        (nm, pp)
        for nm, (pp, o) in placed.items()
        if o.get("cat") == "commercial_kitchen_sink"
    ]
    sink_cands.sort(
        key=lambda t: (
            math.hypot(
                _ctr(t[1])[0] - ROOM["kitchen"][0], _ctr(t[1])[1] - ROOM["kitchen"][1]
            ),
            t[0],
        )
    )
    if _wants_sink and not sink_cands:
        raise RuntimeError(
            "closing: no commercial_kitchen_sink for the 'sink' container"
        )
    if _wants_sink:
        _snm, _spp = sink_cands[0]
        _so = placed[_snm][1]
        scene.entities.pop(_snm, None)
        _register_open("sink", _spp, _so.get("cat"), _so.get("model"))
        print(f"[closing] sink (negative-only) = {_snm} @ {_ctr(_spp)}")

    # ---- imported containers: site each on open floor IN ITS OWN ROOM (clearance slot search) ----------
    def _grid_and_bounds():
        boxes = scene.solid_boxes()
        ext_mnx = min(float(mn[0]) for _n, mn, _mx in boxes)
        ext_mny = min(float(mn[1]) for _n, mn, _mx in boxes)
        ext_mxx = max(float(mx[0]) for _n, _mn, mx in boxes)
        ext_mxy = max(float(mx[1]) for _n, _mn, mx in boxes)

        def _cd(x, y):
            best = 1e9
            for _n, mn, mx in boxes:
                dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
                dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
                best = min(best, math.hypot(dx, dy))
            return best

        grid = [
            (float(x), float(y), _cd(float(x), float(y)))
            for x in np.arange(ext_mnx + 0.4, ext_mxx - 0.4, 0.25)
            for y in np.arange(ext_mny + 0.4, ext_mxy - 0.4, 0.25)
        ]
        return grid, boxes

    # protect the narrow corridor_1 passage (x~3.06-4.00, the ONLY dining<->kitchen link) so no imported
    # unit walls it off (a 1.4 m cabinet at the mouth disconnects the whole dining room).
    used_slots: list = [
        (3.5, -4.0, 1.7),
        (3.5, -3.2, 1.3),
    ]  # (x, y, keep-out radius); also import-vs-import

    # Walls aren't in solid_boxes() (only in scene.wall_prims), so the raw furniture-clearance metric can't
    # tell a corner from an open walkway -> imported bins used to land mid-aisle. A wall-distance term makes
    # each imported unit HUG a wall/corner (like a real bussing/waste station) instead of the walkway centre.
    _wall_boxes = []
    for _wp in scene.wall_prims or []:
        try:
            _wall_boxes.append(_wbbox(_wp))
        except Exception:  # noqa: BLE001
            pass

    def _wall_dist(x, y):
        if not _wall_boxes:
            return 0.0
        b = 1e9
        for mn, mx in _wall_boxes:
            dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
            dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
            b = min(b, math.hypot(dx, dy))
        return b

    def _nearest_wall_axis(x, y):
        # the long (along-wall) axis of the nearest wall to (x,y): a row of bins runs ALONG this.
        best_u, bestd = (1.0, 0.0), 1e18
        for mn, mx in _wall_boxes:
            dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
            dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
            d = math.hypot(dx, dy)
            if d < bestd:
                bestd = d
                best_u = (
                    (1.0, 0.0)
                    if (float(mx[0]) - float(mn[0])) >= (float(mx[1]) - float(mn[1]))
                    else (0.0, 1.0)
                )
        return best_u

    def _find_slot(grid, anchor, need, fh, dcap_max=1e9):
        wall_gap = fh + 0.15  # target: sit ~0.15 m off the wall face
        for cf, dcap0 in (
            (need, 3.0),
            (need, 5.0),
            (need * 0.85, 7.0),
            (need * 0.72, 10.0),
            (need * 0.6, 16.0),
        ):
            dcap = min(
                dcap0, dcap_max
            )  # cap the search radius (keeps a unit in the reachable zone)
            best, bestscore = None, 1e18
            for x, y, c in grid:
                if c < cf:
                    continue
                d = math.hypot(x - anchor[0], y - anchor[1])
                if d > dcap:
                    continue
                if any(math.hypot(x - ux, y - uy) < ur for (ux, uy, ur) in used_slots):
                    continue
                wd = _wall_dist(x, y)
                if _wall_boxes and wd < fh + 0.08:  # would clip into the wall
                    continue
                # minimise: distance to the room anchor PLUS how far the cell floats off a wall
                score = d + 1.3 * max(0.0, wd - wall_gap)
                if score < bestscore:
                    bestscore, best = score, (x, y)
            if best is not None:
                return best, cf
        return None, 0.0

    def _drop_to_floor(prim, pp, x, y, quat):
        fpv_scene.set_world_pose(
            prim, (float(x), float(y), float(scene.floor_z)), quat, (1.0, 1.0, 1.0)
        )
        (bx, by, base_z), _mx = _wbbox(pp)
        if abs(float(base_z) - float(scene.floor_z)) > 0.02:
            fpv_scene.set_world_pose(
                prim,
                (
                    float(x),
                    float(y),
                    float(scene.floor_z) - (float(base_z) - float(scene.floor_z)),
                ),
                quat,
                (1.0, 1.0, 1.0),
            )

    def _orient_door_open(pp, cat, model, cell, grid_boxes):
        # rotate an imported SWING unit about z so its baked door faces the openest cardinal direction ->
        # the leaf-derived stand lands on open, reachable floor (not against a wall).
        of = _closing_leaf_fields(scene, pp, {"cat": cat, "model": model})
        if not of or not of.get("leaf_bbox"):
            return (1.0, 0.0, 0.0, 0.0)
        (mn, mx) = _wbbox(pp)
        ucx, ucy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        (lmn, lmx) = of["leaf_bbox"]
        d0x, d0y = 0.5 * (lmn[0] + lmx[0]) - ucx, 0.5 * (lmn[1] + lmx[1]) - ucy
        d0n = math.hypot(d0x, d0y) or 1.0
        d0 = (d0x / d0n, d0y / d0n)  # current door normal
        a = math.radians(_best_yaw(cell[0], cell[1], grid_boxes))
        d1 = (-math.sin(a), math.cos(a))  # openest direction
        off = math.atan2(d0[0] * d1[1] - d0[1] * d1[0], d0[0] * d1[0] + d0[1] * d1[1])
        return (math.cos(off / 2.0), 0.0, 0.0, math.sin(off / 2.0))

    # ---- waste bins tucked AGAINST a steel counter across the aisle from a fridge (not mid-aisle), the
    # LABELLED face turned OUT toward the walkway. compost (green) + recycling (blue) share the counter across
    # from the RAW fridge; the grease/oil bucket (yellow) sits against a counter across from the DAIRY fridge
    # (kept separate). front_dir is recorded as the aisle-facing normal so the label sign faces the walkway. -
    station_pos: dict = {}
    station_front: dict = {}

    def _against_counter(target, ref, names):
        # place `names` in a row tucked against the nearest COUNTER to `target`, on the counter face turned
        # toward `ref` (the aisle side / the fridge), recording front_dir = that face-normal (label faces out).
        best, bd = None, 1e9
        for _nm, _mn, _mx in scene.solid_boxes():
            if (
                _nm.startswith("b_") or "fridge" in _nm
            ):  # skip imported bins + the fridges themselves
                continue
            if float(_mx[0] - _mn[0]) < 0.35 or float(_mx[1] - _mn[1]) < 0.35:
                continue  # skip thin/small props; keep real counters / prep units
            _cx, _cy = 0.5 * (_mn[0] + _mx[0]), 0.5 * (_mn[1] + _mx[1])
            _dd = math.hypot(_cx - target[0], _cy - target[1])
            if _dd < bd:
                bd, best = _dd, (_nm, _mn, _mx)
        if best is None:
            return {}
        _nm, _mn, _mx = best
        _cx, _cy = 0.5 * (_mn[0] + _mx[0]), 0.5 * (_mn[1] + _mx[1])
        _hxb, _hyb = 0.5 * (_mx[0] - _mn[0]), 0.5 * (_mx[1] - _mn[1])
        _tdx, _tdy = ref[0] - _cx, ref[1] - _cy  # counter centre -> the aisle / fridge
        if abs(_tdx) >= abs(_tdy):
            _nrm, _fe, _al = (
                (1.0 if _tdx > 0 else -1.0, 0.0),
                None,
                (0.0, 1.0),
            )
            _fe = (_cx + _nrm[0] * _hxb, _cy)
        else:
            _nrm, _al = (0.0, 1.0 if _tdy > 0 else -1.0), (1.0, 0.0)
            _fe = (_cx, _cy + _nrm[1] * _hyb)
        _hf = {}
        for n in names:
            _bb = read_spec(
                f"{rt.CONTAINERS[n]['cat']}/{rt.CONTAINERS[n]['model']}"
            ).get("bbox_size") or [0.4, 0.4, 0.4]
            _hf[n] = 0.5 * max(float(_bb[0]), float(_bb[1]))
        _gap = 0.08
        _row = sum(2 * _hf[n] for n in names) + _gap * (len(names) - 1)
        _cur = -0.5 * _row
        _pos = {}
        for n in names:
            _o = _cur + _hf[n]
            _cur += 2 * _hf[n] + _gap
            _pos[n] = (
                _fe[0] + _nrm[0] * (_hf[n] + 0.06) + _al[0] * _o,
                _fe[1] + _nrm[1] * (_hf[n] + 0.06) + _al[1] * _o,
            )
        # REACHABILITY proxy: every bin needs a CLEAR front stand (aisle side). A blocked front (e.g. a
        # cramped grill aisle) -> the human can't reach it, so abort the whole group -> the bins fall back to
        # the normal reachable slot search instead of being tucked somewhere unreachable.
        _fb = scene.solid_boxes()

        def _clr(x, y):
            b = 1e9
            for _n2, _q0, _q1 in _fb:
                dx = max(float(_q0[0]) - x, 0.0, x - float(_q1[0]))
                dy = max(float(_q0[1]) - y, 0.0, y - float(_q1[1]))
                b = min(b, math.hypot(dx, dy))
            return b

        for n in names:
            _sx = _pos[n][0] + _nrm[0] * (_hf[n] + 0.55)
            _sy = _pos[n][1] + _nrm[1] * (_hf[n] + 0.55)
            if _clr(_sx, _sy) < 0.30:
                print(
                    f"[closing] waste {names} front-stand blocked at {_nm} -> fall back to slot search"
                )
                return {}
        for n in names:
            station_front[n] = _nrm  # label faces the aisle
            used_slots.append((_pos[n][0], _pos[n][1], _hf[n] + 0.40))
        print(
            f"[closing] waste bins vs counter {_nm} face_n={_nrm}: {list(_pos.items())}"
        )
        return _pos

    _rf, _re, _kc = (
        _ctr(chosen_fridges[0][1]),
        _ctr(chosen_fridges[1][1]),
        ROOM["kitchen"],
    )

    def _acrosspt(
        fxy,
    ):  # a point ~2 m across the aisle from a fridge (toward the kitchen interior)
        _dx, _dy = _kc[0] - fxy[0], _kc[1] - fxy[1]
        _n = math.hypot(_dx, _dy) or 1.0
        return (fxy[0] + _dx / _n * 2.0, fxy[1] + _dy / _n * 2.0)

    for _dfi, _dfc in (
        (0, _rf),
        (1, _re),
    ):  # DIAG: verify the counters chosen opposite each fridge
        print(
            f"[diag] fridge{_dfi} @ ({_dfc[0]:.2f},{_dfc[1]:.2f}) across={_acrosspt(_dfc)}"
        )
    _pair = [
        n
        for n in ("compost_bin", "recycling_bin")
        if n in rt.CONTAINERS and rt.CONTAINERS[n]["source"] == "import"
    ]
    if _pair:
        station_pos.update(_against_counter(_acrosspt(_rf), _rf, _pair))
    # grease/oil bucket: kept SEPARATE and placed by the normal reachable slot search (tucking it against the
    # grill made it a nav-unreachable pocket -- see v13). It still gets its amber body + USED OIL label.

    # bus_tub: tuck against the nearest DINING-ROOM wall (off the walkway), label turned toward the room.
    if "bus_tub" in rt.CONTAINERS and rt.CONTAINERS["bus_tub"]["source"] == "import":
        _dctr = (
            float(np.mean([_ctr(t[1])[0] for t in dining_tables])),
            float(np.mean([_ctr(t[1])[1] for t in dining_tables])),
        )
        _bb = read_spec("tote/bnkjle").get("bbox_size") or [0.34, 0.24, 0.47]
        _bh = 0.5 * max(float(_bb[0]), float(_bb[1]))
        _furn = scene.solid_boxes()

        def _cd_furn(x, y):
            b = 1e9
            for _n, _m0, _m1 in _furn:
                dx = max(float(_m0[0]) - x, 0.0, x - float(_m1[0]))
                dy = max(float(_m0[1]) - y, 0.0, y - float(_m1[1]))
                b = min(b, math.hypot(dx, dy))
            return b

        _bwall, _bwd = None, 1e9
        for _m0, _m1 in _wall_boxes:
            dx = max(float(_m0[0]) - _dctr[0], 0.0, _dctr[0] - float(_m1[0]))
            dy = max(float(_m0[1]) - _dctr[1], 0.0, _dctr[1] - float(_m1[1]))
            d = math.hypot(dx, dy)
            if d < _bwd:
                _bwd, _bwall = d, (_m0, _m1)
        _bbest, _bnrm = None, (0.0, 1.0)
        if _bwall is not None:
            _m0, _m1 = _bwall
            _wcx = 0.5 * (float(_m0[0]) + float(_m1[0]))
            _wcy = 0.5 * (float(_m0[1]) + float(_m1[1]))
            _whx = 0.5 * (float(_m1[0]) - float(_m0[0]))
            _why = 0.5 * (float(_m1[1]) - float(_m0[1]))
            if _whx <= _why:  # wall runs along y -> its broad faces look +/-x
                _bnrm = (1.0 if _dctr[0] > _wcx else -1.0, 0.0)
                _ax = _wcx + _bnrm[0] * (_whx + _bh + 0.06)
                for _d in (0.0, 0.7, -0.7, 1.4, -1.4):
                    _ay = min(
                        max(_dctr[1] + _d, float(_m0[1]) + _bh + 0.1),
                        float(_m1[1]) - _bh - 0.1,
                    )
                    if _cd_furn(_ax, _ay) >= _bh + 0.05:
                        _bbest = (_ax, _ay)
                        break
            else:  # wall runs along x -> broad faces +/-y
                _bnrm = (0.0, 1.0 if _dctr[1] > _wcy else -1.0)
                _ay = _wcy + _bnrm[1] * (_why + _bh + 0.06)
                for _d in (0.0, 0.7, -0.7, 1.4, -1.4):
                    _ax = min(
                        max(_dctr[0] + _d, float(_m0[0]) + _bh + 0.1),
                        float(_m1[0]) - _bh - 0.1,
                    )
                    if _cd_furn(_ax, _ay) >= _bh + 0.05:
                        _bbest = (_ax, _ay)
                        break
        if _bbest is not None:
            station_pos["bus_tub"] = _bbest
            station_front["bus_tub"] = _bnrm
            used_slots.append((_bbest[0], _bbest[1], _bh + 0.45))
            print(
                f"[closing] bus_tub against dining wall @ ({_bbest[0]:.2f},{_bbest[1]:.2f}) face_n={_bnrm}"
            )
        else:
            print("[closing] bus_tub: no clear dining-wall spot -> default slot search")

    for name, spec in rt.CONTAINERS.items():
        if spec["source"] != "import":
            continue
        cat, model = spec["cat"], spec["model"]
        room, kind = spec["room"], spec["kind"]
        # the tiny corridor_1 cannot hold the wide chem cabinet without blocking the dining<->kitchen
        # passage, so site it in the adjacent DINING room (a different room from the kitchen food storage,
        # so still "away from food") OFF the corridor through-axis (x~3.5).
        anchor = (2.0, -6.0) if room == "corridor" else ROOM.get(room, (cx0, cy0))
        bb = read_spec(f"{cat}/{model}").get("bbox_size") or [0.4, 0.4, 0.4]
        fh = 0.5 * max(float(bb[0]), float(bb[1]))
        need = fh + (0.50 if kind == "swing" else 0.28)
        grid, boxes = _grid_and_bounds()
        if (
            name in station_pos
        ):  # part of the grouped waste station (row against a wall)
            slot, got = station_pos[name], 0.0
        else:
            slot, got = _find_slot(grid, anchor, need, fh)
        if slot is None:
            raise RuntimeError(
                f"closing: no open floor for imported {name} near {room} {anchor}"
            )
        pp = f"/World/b_{_safe(name)}"
        prim = fpv_scene.add_reference(stage, pp, baked_usd(f"{cat}/{model}"))
        _drop_to_floor(prim, pp, slot[0], slot[1], (1.0, 0.0, 0.0, 0.0))
        quat = (1.0, 0.0, 0.0, 0.0)
        if kind == "swing":
            quat = _orient_door_open(pp, cat, model, slot, boxes)
            _drop_to_floor(prim, pp, slot[0], slot[1], quat)
        placed[name] = (pp, {"cat": cat, "model": model})
        if spec.get("recolor_body"):
            _recolor_body(pp, spec.get("color") or (0.8, 0.1, 0.1))
        used_slots.append((slot[0], slot[1], fh + 0.85))
        if kind == "swing":
            if not _register_swing(name, pp, {"cat": cat, "model": model}):
                print(f"[closing] {name}: no swing leaf -> registering top-open")
                _register_open(name, pp, cat, model)
        else:
            _register_open(name, pp, cat, model)
        if (
            name in station_front
        ):  # tucked-against-a-counter bin: turn the label OUT toward the aisle
            front_dir[name] = station_front[name]
        _label_decal(name, pp, spec.get("sign"), kind == "swing")
        reg_pp[name] = pp
        print(
            f"[closing] import {name} ({cat}/{model}, {room}) -> "
            f"({slot[0]:.2f},{slot[1]:.2f}) clearance>={got:.2f}m"
        )

    # ---- preview shots: (eye, look) camera poses for --preview, so the bins/fridges can be checked before
    # the expensive full render. Each frames a labeled unit from its person-facing side. -------------------
    scene.preview_shots = []

    def _pv_shot(label, reg_name, wide=False):
        pp = reg_pp.get(reg_name)
        fd = front_dir.get(reg_name)
        if not pp or not fd:
            return
        (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(pp)
        cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        half = abs(fd[0]) * 0.5 * (mxx - mnx) + abs(fd[1]) * 0.5 * (mxy - mny)
        tall = (float(mxz) - float(mnz)) >= 1.0
        dist = (2.2 if wide else 1.6) if tall else (2.0 if wide else 1.25)
        eye = (
            float(cx + fd[0] * (half + dist)),
            float(cy + fd[1] * (half + dist)),
            float(scene.floor_z) + 1.5,
        )
        look = (float(cx), float(cy), float(mnz) + 0.5 * (float(mxz) - float(mnz)))
        scene.preview_shots.append((label, eye, look))

    for _lab, _rn in (
        ("raw_fridge", "raw_fridge"),
        ("rte_fridge", "rte_fridge"),
        ("compost_bin", "compost_bin"),
        ("recycling_bin", "recycling_bin"),
        ("grease_caddy", "grease_caddy"),
        ("bus_tub", "bus_tub"),
        ("chem_bin", "chem_cabinet"),
    ):
        _pv_shot(_lab, _rn)
    _pair_pos = [
        station_pos[n] for n in ("compost_bin", "recycling_bin") if n in station_pos
    ]
    if (
        _pair_pos
    ):  # one combined shot framing the compost+recycling PAIR (grease is separate)
        _scx = sum(p[0] for p in _pair_pos) / len(_pair_pos)
        _scy = sum(p[1] for p in _pair_pos) / len(_pair_pos)
        _fd = (
            front_dir.get("compost_bin")
            or front_dir.get("recycling_bin")
            or (0.0, -1.0)
        )
        scene.preview_shots.append(
            (
                "waste_pair",
                (
                    float(_scx + _fd[0] * 2.2),
                    float(_scy + _fd[1] * 2.2),
                    float(scene.floor_z) + 1.6,
                ),
                (float(_scx), float(_scy), float(scene.floor_z) + 0.45),
            )
        )
    print(f"[closing] preview shots: {[s[0] for s in scene.preview_shots]}")

    # ---- set-dressing: unregistered realism props (NOT graspable / scored / obstacles). Silently skipped
    # if a model isn't baked, so dressing can never break the build. Two roles: table items a shift would
    # NOT bus (condiments/glassware that just STAY), and end-of-shift clutter (dirty pans, a bussing cart).
    _dn = [0]

    def _place_decor(cat, model, x, y, base_z, quat=(1.0, 0.0, 0.0, 0.0)):
        try:
            usd = baked_usd(f"{cat}/{model}")
        except Exception:  # noqa: BLE001
            return False
        path = f"/World/decor_{_dn[0]}"
        _dn[0] += 1
        prim = fpv_scene.add_reference(stage, path, usd)
        fpv_scene.set_world_pose(
            prim, (float(x), float(y), float(base_z)), quat, (1.0, 1.0, 1.0)
        )
        try:  # seat its base on the target surface (its local origin may not be at its base)
            (bx, by, bz), _mx = _wbbox(path)
            if abs(float(bz) - float(base_z)) > 0.01:
                fpv_scene.set_world_pose(
                    prim,
                    (float(x), float(y), float(base_z) - (float(bz) - float(base_z))),
                    quat,
                    (1.0, 1.0, 1.0),
                )
        except Exception:  # noqa: BLE001
            pass
        return True

    def _seat_decor(surface_pp, cat, model, yaw):
        cb = _collision_bbox(stage, surface_pp) or _wbbox(surface_pp)
        (cmn, cmx) = cb
        cx, cy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
        hx, hy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
        a = math.radians(yaw)
        fwd = (
            math.sin(a),
            -math.cos(a),
        )  # into the table; bias dressing to the FAR half (off the bus edge)
        bx = cx + fwd[0] * hx * 0.35 + _rng.uniform(-hx * 0.5, hx * 0.5)
        by = cy + fwd[1] * hy * 0.35 + _rng.uniform(-hy * 0.5, hy * 0.5)
        bx = min(max(bx, cx - hx + 0.05), cx + hx - 0.05)
        by = min(max(by, cy - hy + 0.05), cy + hy - 0.05)
        _place_decor(cat, model, bx, by, float(cmx[2]))

    def _floor_decor(anchor, cat, model):
        boxes = scene.solid_boxes()

        def _cd(x, y):
            b = 1e9
            for _n, mn, mx in boxes:
                dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
                dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
                b = min(b, math.hypot(dx, dy))
            return b

        best, bestc = None, -1.0
        for _ in range(
            14
        ):  # a clear-ish floor cell near the room, so it doesn't interpenetrate furniture
            ang = _rng.uniform(0.0, 2.0 * math.pi)
            r = _rng.uniform(0.8, 2.4)
            x, y = anchor[0] + math.cos(ang) * r, anchor[1] + math.sin(ang) * r
            c = _cd(x, y)
            if c > bestc:
                bestc, best = c, (x, y)
        if best is not None:
            _place_decor(cat, model, best[0], best[1], float(scene.floor_z))

    for _d in getattr(rt, "DECOR", []):
        _on = _d.get("on", "")
        try:
            if _on == "dining":
                _dt = _rng.choice(dining_tables)
                _seat_decor(_dt[1], _d["cat"], _d["model"], _dt[2])
            elif _on == "prep":
                _seat_decor(prep_pp, _d["cat"], _d["model"], _kyaw(prep_pp))
            elif _on == "counter":
                _seat_decor(counter_pp, _d["cat"], _d["model"], _kyaw(counter_pp))
            elif _on.startswith("floor_"):
                _floor_decor(
                    ROOM.get(_on.split("_", 1)[1], (cx0, cy0)), _d["cat"], _d["model"]
                )
        except Exception as _exc:  # noqa: BLE001
            print(f"[closing] decor skipped {_d.get('cat')}/{_d.get('model')}: {_exc}")
    print(f"[closing] set-dressing: placed {_dn[0]} decor prop(s)")

    # ---- program: DINING first, then KITCHEN, then CORRIDOR; grouped by target container --------------
    goals, negs = rt.build_task_fields()
    # Only reachable BUILD-side gate for `neg_constraints`: at score time episode_runtime wraps the whole goal
    # block in a blanket except, which would turn an unimplemented kind into task_goal:null + a silent pass.
    rt.validate_neg_kinds(negs)
    by_cont: dict = {}
    for k in rt.ITEMS:
        by_cont.setdefault(rt.ANSWER_KEY[k], []).append(k)
    segs: list = []
    for room in ("dining", "kitchen", "corridor"):
        for cont, cspec in rt.CONTAINERS.items():
            if cspec["room"] != room:
                continue
            items = by_cont.get(cont, [])
            if not items:  # e.g. the sink (negative-only)
                continue
            swing = cspec["kind"] == "swing"
            if swing:
                segs.append(f"[Walk] <{cont}> (1), [Open] <{cont}> (1)")
            for k in items:
                segs.append(
                    f"[Walk] <{k}> (1), [Grab] <{k}> (1), [Walk] <{cont}> (1), "
                    f"[PutInside] <{k}> (1) <{cont}> (1)"
                )
            if swing:
                segs.append(f"[Walk] <{cont}> (1), [Close] <{cont}> (1)")
    program = ", ".join(segs)
    scene.behavior_task = {
        "scene": scene_name,
        "task": "closing_line_hotel",
        "grasp": list(rt.ITEMS),
        "goals": goals,
        "neg_constraints": negs,
        "container": None,
    }

    # ---- flags + lights + auto-exposure ---------------------------------------------------------------
    scene.plan_nav = True
    scene.soft_init = True
    # precise_nav: rasterize furniture COLLISION-MESH triangles (not per-owner AABB) so the dense dining
    # room's chair/table aisles stay open + connected (AABBs over-block them into disjoint components) and
    # islanded stands get nav-aware re-picked onto a connected cell. unified_geom + post_action_check: one
    # precise collision map for reach/gate + a physics validate/settle after each action so every PutInside
    # object comes to a real rest INSIDE its container. (Same combo as the challenge-task path.)
    scene.precise_nav = True
    scene.unified_geom = True
    scene.post_action_check = True
    scene.tpv_chase = True
    scene.auto_expose = True
    scene.expose_z = 1.0
    scene.fpv_headlamp = True
    scene.look_zoom = True  # FPV look-at adaptive zoom (produce_look_at/drive_cams) -> reads labels legibly
    _dining_ctr = (
        float(np.mean([_ctr(tpp)[0] for _n, tpp, _y in dining_tables])),
        float(np.mean([_ctr(tpp)[1] for _n, tpp, _y in dining_tables])),
    )
    # Calibrate exposure on the DINING room, NOT the kitchen: the kitchen sat directly under a 60000 sphere,
    # so calibrating there drove the global light scale down and crushed every other room to black. The
    # dining centre is a representative mid-brightness point -> the front-of-house is exposed correctly.
    scene.expose_xy = (float(_dining_ctr[0]), float(_dining_ctr[1]))
    _chem_ctr = (
        2.0,
        -6.0,
    )  # the chem-cabinet corner (dining side, off the corridor axis) -- was UNLIT
    # Per-room key lights with EVEN coverage so no zone is dark. Two are NEW (Corridor, Chem) -- those zones
    # had no local light at all. Cold is much LOWER than the old 60000 so the black-glass fridge specular no
    # longer clips to white (~frame 2150 blowout).
    for _tag, (wx0, wy0), wz0, _inten in (
        ("Kitchen", ROOM["kitchen"], 2.4, 42000.0),
        ("Dining", _dining_ctr, 2.4, 44000.0),
        ("Cold", _ctr(chosen_fridges[0][1]), 2.2, 20000.0),
        (
            "Cold2",
            _ctr(chosen_fridges[1][1]),
            2.2,
            20000.0,
        ),  # the DAIRY fridge -- its interior was unlit
        ("Corridor", ROOM["corridor"], 2.3, 40000.0),
        ("Chem", _chem_ctr, 2.2, 34000.0),
        (
            "Waste",
            (7.3, 1.0),
            2.3,
            18000.0,
        ),  # over the recycling/grease cluster (deep bins read too dark)
    ):
        lt = UsdLux.SphereLight.Define(stage, f"/World/ClosingLight{_tag}")
        lt.CreateRadiusAttr(0.7)
        lt.CreateIntensityAttr(float(_inten))
        lt.CreateColorAttr((1.0, 1.0, 1.0))
        fpv_scene.set_world_pose(lt.GetPrim(), (float(wx0), float(wy0), float(wz0)))
    _dome = UsdLux.DomeLight.Define(stage, "/World/ClosingDome")
    _dome.CreateIntensityAttr(650.0)
    _dome.CreateColorAttr((0.80, 0.85, 0.95))
    _register_mdl()

    tpv_look = (float(ROOM["kitchen"][0]), float(ROOM["kitchen"][1]), 1.0)
    tpv_eye = (float(ROOM["kitchen"][0]) - 3.0, float(ROOM["kitchen"][1]) - 3.0, 3.0)
    print(
        f"[closing] {scene_name}: {len(rt.ITEMS)} items across "
        f"{len(dining_tables)} dining table(s)+prep+counter, {len(rt.CONTAINERS)} containers; "
        f"start=({scene.start_xy[0]:.2f},{scene.start_xy[1]:.2f})"
    )
    print(f"[closing] program: {program}")
    return scene, {}, program, tpv_look, tpv_eye


def _read_attach_points(cat: str, model: str) -> dict:
    """Read attach-point metadata from the ORIGINAL BEHAVIOR metadata.json (the bake strips meta__
    attach Xforms). Returns {pointname: [{"pos": (x,y,z), "quat": (x,y,z,w)}, ...]}.

    The metadata structure is meta_links.base_link.attachment.<pointname> = a single dict or a LIST
    of dicts, each with {"position": [x,y,z], "orientation": [x,y,z,w]} (XYZW quaternion). We
    normalize both cases to a list of entries with `pos`/`quat` keys (tuples for consistency).
    """
    meta_path = os.path.join(
        BEHAVIOR_DATA_ROOT,
        "behavior-1k-assets",
        "objects",
        cat,
        model,
        "misc",
        "metadata.json",
    )
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    attach_raw = meta.get("meta_links", {}).get("base_link", {}).get("attachment", {})
    attach_out = {}
    for point_name, raw_entries in attach_raw.items():
        # Normalize: single dict -> [dict]; list of dicts stays list.
        entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
        attach_out[point_name] = [
            {
                "pos": tuple(e.get("position", [0.0, 0.0, 0.0])),
                "quat": tuple(e.get("orientation", [0.0, 0.0, 0.0, 1.0])),
            }
            for e in entries
        ]
    return attach_out


def _bind_label_png(stage, mesh_prim, path, png, *, self_lit: bool = True):
    """Bind a textured decal, preserving PNG alpha and optional emissive ink."""
    from pxr import Sdf, UsdShade

    mtl = UsdShade.Material.Define(stage, path + "/M")
    rdr = UsdShade.Shader.Define(stage, path + "/M/St")
    rdr.CreateIdAttr("UsdPrimvarReader_float2")
    rdr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    rdr.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, path + "/M/T")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        rdr.ConnectableAPI(), "result"
    )
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    tex.CreateOutput("a", Sdf.ValueTypeNames.Float)
    sh = UsdShade.Shader.Define(stage, path + "/M/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    if self_lit:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
        tex.ConnectableAPI(), "a"
    )
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.01)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6 if self_lit else 0.9)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
        mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )


_TINT_RGB: dict[str, tuple[float, float, float]] = {
    "black": (0.05, 0.05, 0.05),
    "white": (0.90, 0.90, 0.90),
    "grey": (0.45, 0.45, 0.45),
    "gray": (0.45, 0.45, 0.45),
    "red": (0.70, 0.06, 0.06),
    "orange": (0.85, 0.35, 0.02),
    "yellow": (0.90, 0.75, 0.05),
    "green": (0.10, 0.45, 0.12),
    "blue": (0.07, 0.20, 0.65),
    "purple": (0.35, 0.10, 0.55),
    "pink": (0.90, 0.45, 0.60),
    "brown": (0.32, 0.20, 0.10),
    "golden brown": (0.50, 0.24, 0.07),
    "tan": (0.72, 0.60, 0.42),
    "silver": (0.72, 0.73, 0.75),
}


def tint_rgb(name: str) -> tuple[float, float, float] | None:
    """A plain colour word -> linear RGB, or None when the word is not one we can paint.

    A NAMED colour is a physical fact about the paint, not a rule about task design, which is why the
    table lives in code. Returning None rather than a default is the point: the library has no
    ``red_box``, only ``box``, so a row that says red and silently gets the untinted body is a row
    whose picture no longer answers its own question -- the caller must be able to say so."""
    key = str(name or "").strip().lower().replace("_", " ").replace("-", " ")
    return _TINT_RGB.get(key) or _TINT_RGB.get(key.split()[-1] if key else "")


def _define_tint_material(stage, path, rgb):
    """Flat-colour override for a whole body, so "the red bin" and "the green bin" are one baked
    category wearing two coats.

    The same ``UsdPreviewSurface`` + ``strongerThanDescendants`` binding ``_bind_label_png`` uses, and
    for the same reason: that strength is what overrides the MDL material BEHAVIOR ships on the mesh.
    Two deliberate differences from the label path -- ``diffuseColor`` is SET rather than wired to a
    texture reader, and ``emissiveColor`` is left alone. A label is self-lit on purpose so a placard
    stays crisp in any light; a tinted object must NOT be, or a red bin reads as a glowing red lamp
    and stops looking like a bin at all."""
    from pxr import Gf, Sdf, UsdShade

    mtl = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    )
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mtl


def _bind_material_override(mesh_prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )


def _bind_tint(stage, mesh_prim, path, rgb) -> None:
    """Bind a permanent flat-colour material used by authored initial tints."""
    material = _define_tint_material(stage, path + "/MT", rgb)
    _bind_material_override(mesh_prim, material)


@dataclass(frozen=True, slots=True)
class DirectMaterialBindingSnapshot:
    prim_path: str
    relationship_existed: bool
    targets: tuple[object, ...]
    binding_strength: object | None
    api_schema_existed: bool


@dataclass(frozen=True, slots=True)
class EntityTintRestoreToken:
    entity_path: str
    material_path: str
    bindings: tuple[DirectMaterialBindingSnapshot, ...]


def _has_material_binding_api(prim) -> bool:
    from pxr import UsdShade

    return bool(prim.HasAPI(UsdShade.MaterialBindingAPI))


def _remove_material_binding_api(prim) -> None:
    from pxr import UsdShade

    prim.RemoveAPI(UsdShade.MaterialBindingAPI)


def _snapshot_direct_material_binding(prim) -> DirectMaterialBindingSnapshot:
    relationship = prim.GetRelationship("material:binding")
    exists = bool(relationship and relationship.IsValid())
    targets = tuple(relationship.GetTargets()) if exists else ()
    strength = relationship.GetMetadata("bindMaterialAs") if exists else None
    return DirectMaterialBindingSnapshot(
        prim_path=str(prim.GetPath()),
        relationship_existed=exists,
        targets=targets,
        binding_strength=strength,
        api_schema_existed=_has_material_binding_api(prim),
    )


def _restore_direct_material_binding(prim, snapshot) -> None:
    if snapshot.relationship_existed:
        relationship = prim.CreateRelationship("material:binding")
        relationship.SetTargets(list(snapshot.targets))
        if snapshot.binding_strength is None:
            relationship.ClearMetadata("bindMaterialAs")
        else:
            relationship.SetMetadata("bindMaterialAs", snapshot.binding_strength)
    else:
        prim.RemoveProperty("material:binding")
    if not snapshot.api_schema_existed:
        _remove_material_binding_api(prim)


def _entity_tint_targets(entity_prim) -> list:
    from pxr import Usd, UsdGeom

    root_path = str(entity_prim.GetPath()).rstrip("/")
    gprims = [child for child in Usd.PrimRange(entity_prim) if UsdGeom.Gprim(child)]
    by_path = {str(child.GetPath()): child for child in gprims}
    return [by_path[path] for path in object_tint_gprim_paths(root_path, list(by_path))]


def apply_entity_tint(stage, entity_path: str, tint: str) -> EntityTintRestoreToken:
    """Apply a reversible visual-only tint to one existing scene entity."""
    rgb = tint_rgb(tint)
    if rgb is None:
        raise ValueError(f"unknown entity tint {tint!r}")
    entity_prim = stage.GetPrimAtPath(entity_path)
    if not (entity_prim and entity_prim.IsValid()):
        raise RuntimeError(f"entity tint references missing prim {entity_path}")
    targets = _entity_tint_targets(entity_prim)
    if not targets:
        raise RuntimeError(f"entity tint target {entity_path} has no body Gprim")
    material_path = str(entity_prim.GetPath()).rstrip("/") + "/VisualStateTint"
    existing = stage.GetPrimAtPath(material_path)
    if existing and existing.IsValid():
        raise RuntimeError(f"entity tint material path already exists: {material_path}")
    bindings = tuple(_snapshot_direct_material_binding(prim) for prim in targets)
    try:
        material = _define_tint_material(stage, material_path, rgb)
        for target in targets:
            _bind_material_override(target, material)
    except Exception:
        for snapshot in bindings:
            prim = stage.GetPrimAtPath(snapshot.prim_path)
            if prim and prim.IsValid():
                _restore_direct_material_binding(prim, snapshot)
        stage.RemovePrim(material_path)
        raise
    return EntityTintRestoreToken(
        entity_path=str(entity_prim.GetPath()),
        material_path=material_path,
        bindings=bindings,
    )


def restore_entity_tint(stage, token: EntityTintRestoreToken) -> None:
    """Restore every direct material binding captured by ``apply_entity_tint``."""
    for snapshot in token.bindings:
        prim = stage.GetPrimAtPath(snapshot.prim_path)
        if not (prim and prim.IsValid()):
            raise RuntimeError(
                f"cannot restore entity tint; missing prim {snapshot.prim_path}"
            )
        _restore_direct_material_binding(prim, snapshot)
    stage.RemovePrim(token.material_path)


def object_tint_gprim_paths(root_path: str, gprim_paths: list[str]) -> list[str]:
    """Return body Gprims that may be tinted without replacing authored cue artwork."""
    root = str(root_path).rstrip("/")
    selected = []
    for path in gprim_paths:
        relative = str(path)[len(root) :].lstrip("/")
        first = relative.split("/", 1)[0]
        if first.startswith(("Cue_", "Placard_", "BoardLabel_", "Label_")):
            continue
        selected.append(str(path))
    return selected


def apply_object_tints(scene, stage) -> int:
    """Paint every entity named in ``scene.obj_tints`` = {entity_key: colour word}.

    Runs at render time after the prims exist -- the same slot and the same shape as
    ``apply_board_labels`` below, including the no-op when the channel is unset, so a scene that
    authors no colour is byte-identical. A BEHAVIOR body is a hierarchy of meshes with per-mesh MDL
    bindings, so every body Gprim is painted. Authored cue descendants are excluded; binding tint on
    the root or directly on a cue would replace its texture with a solid colour.

    Returns the number of entities painted; a colour word with no RGB and an entity with no live prim
    are both counted as misses by the caller, never silently swallowed -- an unpainted body in a row
    whose question IS the colour makes that row unanswerable."""
    wanted = getattr(scene, "obj_tints", None)
    if not wanted:
        return 0
    from pxr import Usd, UsdGeom

    painted = 0
    for key, colour in wanted.items():
        rgb = tint_rgb(colour)
        entity = (getattr(scene, "entities", None) or {}).get(key)
        path = getattr(entity, "prim_path", None) if entity is not None else None
        prim = stage.GetPrimAtPath(path) if path else None
        if rgb is None or prim is None or not prim.IsValid():
            print(
                f"[behavior] TINT MISS {key}: colour={colour!r} rgb={rgb} "
                f"prim={path!r} -- this body keeps its stock appearance"
            )
            continue
        root_path = str(prim.GetPath()).rstrip("/")
        gprims = [child for child in Usd.PrimRange(prim) if UsdGeom.Gprim(child)]
        by_path = {str(child.GetPath()): child for child in gprims}
        target_paths = object_tint_gprim_paths(root_path, list(by_path))
        targets = [by_path[target_path] for target_path in target_paths]
        if len(target_paths) == len(gprims):
            targets.insert(0, prim)
        if not targets:
            print(f"[behavior] TINT MISS {key}: prim={path!r} has no non-cue Gprim")
            continue
        for target in targets:
            _bind_tint(stage, target, str(target.GetPath()), rgb)
        painted += 1
        print(f"[behavior] tint {key} = {colour} rgb={tuple(round(v, 3) for v in rgb)}")
    return painted


_LABEL_READ_MIN_M = 0.55
_LABEL_READ_MAX_M = 1.35


def _label_reachability(scene, viewer_xy):
    grid = getattr(scene, "nav_grid", None)
    if grid is None or viewer_xy is None:
        return None, None
    import nav_grid

    return grid, nav_grid.flood_free(grid, grid.world_to_cell(*viewer_xy))


def _label_read_ray_clear(grid, center_xy, stand_xy) -> bool:
    cx, cy = center_xy
    sx, sy = stand_xy
    distance = math.hypot(sx - cx, sy - cy)
    samples = max(1, int(math.ceil(distance / (0.5 * grid.cell))))
    for sample in range(1, samples + 1):
        radius = distance * sample / samples
        if radius < 0.40:
            continue
        fraction = radius / distance
        i, j = grid.world_to_cell(
            cx + (sx - cx) * fraction,
            cy + (sy - cy) * fraction,
        )
        if not grid.is_free(i, j):
            return False
    return True


def _reachable_label_up(grid, reachable, viewer_xy, center_xy, aligned_up=(0.0, 1.0)):
    """Choose an object-aligned text direction that is not inverted for a reachable reader."""
    axis_length = math.hypot(aligned_up[0], aligned_up[1])
    axis = (
        (aligned_up[0] / axis_length, aligned_up[1] / axis_length)
        if axis_length > 1e-9
        else (0.0, 1.0)
    )
    if grid is None or reachable is None:
        return axis, None
    cx, cy = center_xy
    vi, vj = grid.world_to_cell(cx, cy)
    radius_cells = int(math.ceil(_LABEL_READ_MAX_M / grid.cell))
    best = None
    for j in range(max(0, vj - radius_cells), min(grid.ny, vj + radius_cells + 1)):
        for i in range(max(0, vi - radius_cells), min(grid.nx, vi + radius_cells + 1)):
            if not grid.is_free(i, j) or not bool(reachable[j, i]):
                continue
            sx, sy = grid.cell_center(i, j)
            distance = math.hypot(cx - sx, cy - sy)
            if not _LABEL_READ_MIN_M <= distance <= _LABEL_READ_MAX_M:
                continue
            if not _label_read_ray_clear(grid, center_xy, (sx, sy)):
                continue
            score = (
                -distance,
                math.hypot(sx - viewer_xy[0], sy - viewer_xy[1]),
                j,
                i,
            )
            if best is None or score < best[0]:
                best = (score, sx, sy, distance)
    if best is None:
        return axis, None
    _score, sx, sy, distance = best
    desired = ((cx - sx) / distance, (cy - sy) / distance)
    sign = 1.0 if axis[0] * desired[0] + axis[1] * desired[1] >= 0.0 else -1.0
    return (axis[0] * sign, axis[1] * sign), (sx, sy)


def apply_board_labels(scene, stage, *, viewer_xy=None) -> int:
    """Stamp each registered board with its letter placard (a self-lit textured quad on the board's TOP face)
    so the FPV/TPV read WHICH board is which. Consumes ``scene.board_labels`` = {entity_key: png_abs_path}; the
    quad is placed FLAT (normal +Z) at the built+settled board's world top face, sized to its short side. Runs
    at render time AFTER the board prims exist. No-op (returns 0) if ``scene.board_labels`` is unset -> every
    non-bookcase scene byte-identical."""
    labels = getattr(scene, "board_labels", None)
    if not labels:
        return 0
    import os

    import fpv_scene
    import label_visibility as lv
    from pxr import Gf, Sdf, Usd, UsdGeom

    if viewer_xy is None:
        viewer_xy = getattr(scene, "start_xy", None)
    read_grid, reachable = _label_reachability(scene, viewer_xy)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    n = 0
    for key, png in labels.items():
        if not png or not os.path.exists(png):
            continue
        e = scene.entities.get(key)
        pp = getattr(e, "prim_path", None) if e is not None else None
        prim = stage.GetPrimAtPath(pp) if pp else None
        if prim is None or not prim.IsValid():
            continue
        rng = fpv_scene.compute_world_bbox(prim)
        mn, mx = rng.GetMin(), rng.GetMax()
        cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        topz = (
            float(mx[2]) + 0.004
        )  # a hair above the top face so it never z-fights the board
        side = 0.98 * min(
            float(mx[0] - mn[0]), float(mx[1] - mn[1])
        )  # square placard fits the short side
        # RIDE the board: parent the placard UNDER the board prim; set its LOCAL matrix so a unit quad lands on
        # the board's world top face NOW, so when the board is grabbed/attached (its prim matrix changes) the
        # placard moves rigidly WITH it (like the Task-1 fridge label riding its door leaf). USD row-vector:
        # label_localToWorld = L * P_world; want it == D_world (scale->translate the unit quad) -> L = D * inv(P).
        Pw = xc.GetLocalToWorldTransform(prim)
        board_axis = Pw.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
        board_axis_xy = (float(board_axis[0]), float(board_axis[1]))
        if math.hypot(*board_axis_xy) <= 1e-9:
            board_axis = Pw.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            board_axis_xy = (float(board_axis[0]), float(board_axis[1]))
        label_up_xy, _read_stand = _reachable_label_up(
            read_grid,
            reachable,
            viewer_xy,
            (float(cx), float(cy)),
            board_axis_xy,
        )
        label_up = Gf.Vec3d(label_up_xy[0], label_up_xy[1], 0.0)
        S = Gf.Matrix4d(1.0)
        S.SetScale(Gf.Vec3d(side, side, 1.0))
        R = Gf.Matrix4d(1.0)
        R.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), label_up))
        T = Gf.Matrix4d(1.0)
        T.SetTranslate(Gf.Vec3d(float(cx), float(cy), topz))
        Lm = (S * R * T) * Pw.GetInverse()
        path = pp + "/letter_label"
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(
            [
                Gf.Vec3f(-0.5, -0.5, 0.0),
                Gf.Vec3f(0.5, -0.5, 0.0),
                Gf.Vec3f(0.5, 0.5, 0.0),
                Gf.Vec3f(-0.5, 0.5, 0.0),
            ]
        )
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        m.SetNormalsInterpolation("faceVarying")
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        UsdGeom.Xformable(m).MakeMatrixXform().Set(Lm)
        _bind_label_png(stage, m.GetPrim(), path, png)
        local_center = Pw.GetInverse().Transform(Gf.Vec3d(float(cx), float(cy), topz))
        local_normal = Pw.GetInverse().TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
        normal_length = local_normal.GetLength()
        if normal_length > 1e-9:
            local_normal /= normal_length
        local_up = Pw.GetInverse().TransformDir(label_up)
        up_length = local_up.GetLength()
        if up_length > 1e-9:
            local_up /= up_length
        readability = lv.texture_readability(str(png))
        if not readability:
            try:
                import label_image_metrics

                ink = label_image_metrics.ink_metrics(str(png))
                readability = {
                    "glyph_height_ratio": float(ink.get("cap_frac") or 0.0),
                    "line_count": int(ink.get("n_lines") or 0),
                }
            except (ModuleNotFoundError, OSError, ValueError):
                readability = {}
        e.meta.update(
            {
                "label_pos": (float(cx), float(cy), topz),
                "label_local": tuple(float(value) for value in local_center),
                "label_face": (0.0, 0.0),
                "label_face3": (0.0, 0.0, 1.0),
                "label_face_local": tuple(float(value) for value in local_normal),
                "label_up3": (float(label_up[0]), float(label_up[1]), 0.0),
                "label_up_local": tuple(float(value) for value in local_up),
                "label_span": float(side),
                "label_size": (float(side), float(side)),
                "label_texture": str(png),
                "label_primary_glyph_height_ratio": float(
                    readability.get("glyph_height_ratio") or 0.0
                ),
                "label_primary_line_count": int(readability.get("line_count") or 0),
                "label_primary_text_bbox_norm_xyxy": readability.get("content_box_uv"),
                "label_curved": False,
                "label_mount": "top",
                "label_read_in_place": True,
            }
        )
        label_paths = dict(getattr(scene, "label_prim_paths", None) or {})
        label_paths[key] = path
        scene.label_prim_paths = label_paths
        n += 1
    print(f"[bookcase] applied {n} board labels")
    return n


def _bookcase_side_r_attach_tokens(
    child: str,
    parent: str,
    canonical_attach: str,
    side_r_witness_walk: str,
    side_r_witness: str,
    orientation_recovery: bool,
) -> list[str]:
    if child != "side_R":
        return [canonical_attach]
    if orientation_recovery:
        return [
            canonical_attach + " <orientation_error_deg>(180)",
            side_r_witness_walk,
            side_r_witness,
            f"[Detach] <{child}>(1) <{parent}>(1)",
            canonical_attach,
            side_r_witness_walk,
            side_r_witness,
        ]
    return [canonical_attach, side_r_witness_walk, side_r_witness]


def build_behavior_bookcase(
    stage,
    app,
    scene_name="hall_arch_wood",
    program_variant="canonical",
):
    """Benchmark Task 2 -- flat-pack bookcase carcass assembly.

    Consumes the data tables in `bookcase_task` (PARTS / CARCASS_EDGES / SHELF_SLOTS_LOCAL_Y /
    REACHABLE_SLOTS / build_task_fields) and stages them in an open floor patch of `hall_arch_wood`
    (a bigger open scene). The BACK panel is pre-placed flat as the assembly root; the other 5 carcass
    boards + N shelves are laid out on the floor AROUND it using the closing builder's clearance-based
    slot search. MINIMAL program (v1 first render): lay out ALL boards but attach ONLY side_L -> back
    to confirm the [Attach] action works. Additive: a NEW multi-room builder; existing scenes
    byte-identical. Returns the step3 5-tuple (scene, {}, program, tpv_look, tpv_eye). Data table:
    simulator.physical_world/bookcase_task.py.

    v1 SCOPE = ONE attach (side_L -> back) to validate the attach API + motion.
    """
    import math

    import bookcase_task as bt
    import constraint_evaluator as rt
    import fpv_scene
    import numpy as np
    from pxr import UsdLux
    from scene_state import Role

    # The assembly needs an open hall, so this task is fixed to bt.SCENE. A caller passing the stale
    # single-room key the spec used to name is redirected rather than silently composing the wrong scene.
    if scene_name in (None, "Rs_int"):
        scene_name = bt.SCENE

    scene, placed, _surfaces, centroid, _floor_top, _dz = _compose_behavior_scene(
        stage, app, scene_name, None, None, None
    )
    cx0, cy0 = float(centroid[0]), float(centroid[1])

    def _wbbox(pp):
        rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(pp))
        mn, mx = rng.GetMin(), rng.GetMax()
        return (
            (float(mn[0]), float(mn[1]), float(mn[2])),
            (float(mx[0]), float(mx[1]), float(mx[2])),
        )

    # ---- Build clearance grid + slot-search helpers (mirrors closing builder's pattern) -----------
    # Grid = (x, y, clearance_to_nearest_furniture) on a 0.25m lattice; _find_slot picks the nearest
    # cell with clearance >= need. used_slots tracks placed parts to avoid overlap.
    boxes = scene.solid_boxes()

    def _cd(x, y):
        # clearance = distance to the nearest furniture bbox edge
        best = 1e9
        for _n, mn, mx in boxes:
            dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
            dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
            best = min(best, math.hypot(dx, dy))
        return best

    # Compute scene extent from furniture bboxes (the grid covers the furniture bounding hull + margin).
    ext_mnx = min((float(mn[0]) for (_n, mn, _mx) in boxes), default=cx0 - 10.0)
    ext_mxx = max((float(mx[0]) for (_n, _mn, mx) in boxes), default=cx0 + 10.0)
    ext_mny = min((float(mn[1]) for (_n, mn, _mx) in boxes), default=cy0 - 10.0)
    ext_mxy = max((float(mx[1]) for (_n, _mn, mx) in boxes), default=cy0 + 10.0)

    grid = [
        (float(x), float(y), _cd(float(x), float(y)))
        for x in np.arange(ext_mnx + 0.4, ext_mxx - 0.4, 0.25)
        for y in np.arange(ext_mny + 0.4, ext_mxy - 0.4, 0.25)
    ]

    used_slots: list = []  # (x, y, keep-out_radius) for part-vs-part spacing

    def _find_slot(anchor, need, dcap_max=1e9):
        # Search for a grid cell (x,y) with clearance >= need, nearest to anchor, within dcap_max.
        # Relaxes need downward if no cell qualifies (same relaxation ladder as closing builder).
        for cf, dcap0 in (
            (need, 3.0),
            (need, 5.0),
            (need * 0.85, 7.0),
            (need * 0.72, 10.0),
            (need * 0.6, 16.0),
        ):
            dcap = min(dcap0, dcap_max)
            best, bestscore = None, 1e18
            for x, y, c in grid:
                if c < cf:
                    continue
                d = math.hypot(x - anchor[0], y - anchor[1])
                if d > dcap:
                    continue
                if any(math.hypot(x - ux, y - uy) < ur for (ux, uy, ur) in used_slots):
                    continue
                if d < bestscore:
                    bestscore, best = d, (x, y)
            if best is not None:
                return best, cf
        return None, 0.0

    # ---- Place the BACK panel FLAT on the floor (the assembly root) --------------------------------
    # The back lies FLAT on the floor; the human reaches DOWN to its connectors. [Attach] RE-NAVIGATES to a
    # stand that can actually reach each connector (same "walk to a reachable spot" logic as PUT), so no
    # workbench is needed -- and a workbench would only push a TALL bookcase's UPPER connectors out of reach.
    # The back is a 2cm-THIN flat panel -> foot_half=None (NOT a standing obstacle) so a stand can sit adjacent.
    ASSEMBLY_ANCHOR = (cx0, cy0)  # search from the scene centroid
    back_slot, back_clr = _find_slot(ASSEMBLY_ANCHOR, 1.3)
    if back_slot is None:
        raise RuntimeError(
            f"bookcase: no open floor for the back panel near {ASSEMBLY_ANCHOR}"
        )
    back_xy = back_slot
    back_z = float(scene.floor_z) + 0.05  # lay flat on the floor
    used_slots.append((back_xy[0], back_xy[1], 1.3))  # reserve the back's spot

    back_cat = bt.PARTS["back"]["cat"]
    back_model = bt.PARTS["back"]["model"]
    back_attach = _read_attach_points(back_cat, back_model)

    scene.grasp_objs = []
    scene.grasp_objs.append(
        {
            "key": "back",
            "path": "/World/Obj_back",
            "build": behavior_build_params(f"{back_cat}/{back_model}", 1.0),
        }
    )
    # Register the back with foot_half=None (no obstacle footprint) so stands can be placed adjacent.
    scene.register(
        "back",
        Role.GRASPABLE,
        xy=back_xy,
        z=back_z,
        prim_path="/World/Obj_back",
        approach_yaw=0.0,
        clearance=0.34,
        attach=back_attach,
    )
    scene.assembly_root = "back"
    # Do NOT pre-seed scene.assembly_pose["back"] with an ASSUMED identity orientation: produce_attach falls
    # back to _read_world_col(back) for the root, reading the back prim's ACTUAL placed pose (whatever
    # orientation the render gives a flat panel at frame-gen), so children mate relative to the real root.

    print(
        f"[bookcase] back panel @ ({back_xy[0]:.2f},{back_xy[1]:.2f}) z={back_z:.2f}m clearance={back_clr:.2f}m"
    )

    # ---- Place the OTHER 5 carcass boards + N shelves on open floor AROUND the back ---------------
    # Each is a Role.GRASPABLE with foot_half=None (thin flat boards are NOT standing obstacles). Use
    # _find_slot to place each part in a verified-clear cell near the back, with part-vs-part spacing
    # so 2m boards don't overlap. All parts are flat on the floor (z = floor_z + 0.05).
    other_parts = ["side_L", "side_R", "top", "baseboard"]
    n_shelves = len(bt.REACHABLE_SLOTS)
    all_keys = other_parts + [f"shelf_{i}" for i in range(n_shelves)]

    for key in all_keys:
        # Resolve the part data: shelf instances use the shelf model; carcass parts use their own.
        if key.startswith("shelf_"):
            part_key_data = "shelf"
        else:
            part_key_data = key
        part_info = bt.PARTS[part_key_data]
        part_cat = part_info["cat"]
        part_model = part_info["model"]
        part_attach = _read_attach_points(part_cat, part_model)

        # Read the part's bbox from the spec to estimate clearance need (half the max(X,Y) dimension + margin).
        try:
            bb = read_spec(f"{part_cat}/{part_model}").get("bbox_size") or [
                0.3,
                0.3,
                0.1,
            ]
        except Exception:  # noqa: BLE001
            bb = [0.3, 0.3, 0.1]
        part_half = 0.5 * max(float(bb[0]), float(bb[1]))
        need = part_half + 0.25  # clearance = part half-extent + 0.25m margin

        # Search for an open slot near the back panel (dcap_max=4m so parts stay nearby).
        slot, got = _find_slot(back_xy, need, dcap_max=4.0)
        if slot is None:
            raise RuntimeError(
                f"bookcase: no open floor for part {key} (need={need:.2f}m) near back"
            )
        px, py = slot
        pz = float(scene.floor_z) + 0.05
        used_slots.append(
            (px, py, part_half + 0.30)
        )  # reserve this slot (part + spacing margin)

        prim_path = f"/World/Obj_{_safe(key)}"
        scene.grasp_objs.append(
            {
                "key": key,
                "path": prim_path,
                "build": behavior_build_params(f"{part_cat}/{part_model}", 1.0),
            }
        )
        # Register with foot_half=None (no obstacle) so stands can be placed adjacent (flat boards).
        scene.register(
            key,
            Role.GRASPABLE,
            xy=(px, py),
            z=pz,
            prim_path=prim_path,
            approach_yaw=0.0,
            clearance=0.34,
            attach=part_attach,
        )
        print(
            f"[bookcase] {key:12s} @ ({px:.2f},{py:.2f}) clearance={got:.2f}m half={part_half:.2f}m"
        )

    # Start the human a SHORT walk from the assembly: hint = the assembly centre, so the seed-init's place_start
    # picks the NEAREST clean (non-penetrating) cell to it, instead of the scene's default free-cell start ~40 m
    # across the open hall. Tighter video + far fewer walk frames. place_start still validates/relocates it.
    _start_cands = [
        (x, y, c)
        for (x, y, c) in grid
        if 2.5 <= math.hypot(x - back_xy[0], y - back_xy[1]) <= 6.0
    ]
    _start_cands.sort(
        key=lambda t: -t[2]
    )  # most-open floor cell a short walk from the assembly
    if _start_cands:
        scene.start_xy = (float(_start_cands[0][0]), float(_start_cands[0][1]))

    # ---- board letter LABELS (read via LookAt; the manual maps letters -> roles) ------------------
    # Stamp each board with its part-letter placard (A-F) so the FPV/TPV read which board is which. Applied at
    # render time (apply_board_labels, after the prims are built) via scene.board_labels {key: png}. The placard
    # shows ONLY the letter -- the assistant must read the MANUAL + LookAt to map letter -> role (part-ID task).
    _LBL_DIR = os.path.join(CACHE_ROOT, "labels")

    def _lbl_png(k):
        pk = "shelf" if k.startswith("shelf_") else k
        base = bt.LABELS.get(pk)
        return f"{_LBL_DIR}/{base}.png" if base else None

    scene.board_labels = {k: _lbl_png(k) for k in (["back"] + all_keys) if _lbl_png(k)}
    side_r_witness_key = "side_R_witness"
    scene.register(
        side_r_witness_key,
        Role.SURFACE,
        xy=back_xy,
        z=back_z,
        meta={
            "expected_attach_child": "side_R",
            "expected_attach_parent": "back",
            "expected_child_point": "shelfsidebxfkjjM",
            "expected_parent_point": "shelfsidebxfkjjF",
            "expected_label_path": "/World/Obj_side_R/letter_label",
            "live_target_key": "side_R",
            "label_span": 0.20,
            "force_look_zoom": True,
            "fixed_look_frames": True,
            "gaze_locked": True,
            "state_child": "side_R",
            "witness_standoff_m": 1.0,
        },
    )
    scene.attach_witness_keys = {"side_R": [side_r_witness_key]}
    side_r_witness_walk = f"[Walk] <{side_r_witness_key}>(1)"
    side_r_witness = f"[LookAt] <{side_r_witness_key}>(1)"

    # ---- PROGRAM (VirtualHome tokens): FULL CARCASS -- assemble all 4 edges in dependency order -----
    # For each CARCASS_EDGE (child, parent, base): LookAt the board (read its letter), walk to it, grab it, walk
    # to the parent, and Attach child.<base>M onto parent.<base>F (explicit 4-arg pairing). The edge order
    # respects precedence (side_R before baseboard, since the baseboard mates side_R's female). The [Attach]
    # reach-renav walks to a connector-reachable stand per edge; depth-2 (baseboard -> side_R) mates side_R's
    # stored assembled pose. P2 appends shelves; P3 stand-up; P4 stocking.
    if program_variant not in ("canonical", "orientation_recovery"):
        raise ValueError(f"unknown bookcase program_variant {program_variant!r}")
    program_tokens: list[str] = []
    for _child, _parent, _base in bt.CARCASS_EDGES:
        prefix = [
            f"[LookAt] <{_child}>(1)",
            f"[Walk] <{_child}>(1)",
            f"[Grab] <{_child}>(1)",
            f"[Walk] <{_parent}>(1)",
        ]
        canonical_attach = (
            f"[Attach] <{_child}>(1) <{_base}M>(0) <{_parent}>(1) <{_base}F>(0)"
        )
        program_tokens += prefix
        program_tokens += _bookcase_side_r_attach_tokens(
            _child,
            _parent,
            canonical_attach,
            side_r_witness_walk,
            side_r_witness,
            program_variant == "orientation_recovery",
        )
    # P2 shelves: install each shelf instance into a back shelfbackF slot (same attach mechanic; the PARENT
    # connector INDEX selects the height slot). v1 installs one shelf per REACHABLE_SLOT.
    for _i, _slot in enumerate(bt.REACHABLE_SLOTS):
        _sk = f"shelf_{_i}"
        program_tokens += [
            f"[LookAt] <{_sk}>(1)",
            f"[Walk] <{_sk}>(1)",
            f"[Grab] <{_sk}>(1)",
            f"[Walk] <back>(1)",
            f"[Attach] <{_sk}>(1) <{bt.SHELF_EDGE_POINT}M>(0) <back>(1) <{bt.SHELF_EDGE_POINT}F>({_slot})",
        ]
    # P3 stand-up: tilt the fully-assembled unit upright about the back's base edge (rigid; carries all parts).
    program_tokens.append("[Raise] <back>(1)")
    program = ", ".join(program_tokens)

    # ---- scene.behavior_task (consumed by episode_runtime's gated scorer) ----------------------------
    # Sourced from bookcase_task.build_task_fields(), so the spec module is the single source of truth for
    # what is scored; only the shelf edges are added here, since only the builder knows how many shelf
    # instances it laid out. The stocking phase is DEFERRED (bookcase_task.DEFERRED_SCORING) -> no
    # negatives. validate_neg_kinds is the build-time gate on that: an unimplemented kind evaluates
    # ok=False for every world and the scorer ANDs neg_ok into met, so one reaching here would make the
    # task unpassable -- caught in milliseconds now instead of after a 40-minute render.
    _tf = bt.build_task_fields()
    rt.validate_neg_kinds(_tf["neg_constraints"])
    _shelf_keys = [f"shelf_{i}" for i in range(len(bt.REACHABLE_SLOTS))]
    scene.behavior_task = {
        "scene": scene_name,
        "task": "bookcase_assembly",
        "carcass": list(_tf["carcass"])  # 4 carcass edges + 4 shelves, all ATTACHED
        + [(k, _tf["shelf_parent"]) for k in _shelf_keys],
        "neg_constraints": _tf["neg_constraints"],
        "grasp": [c for (c, _p, _b) in bt.CARCASS_EDGES]
        + _shelf_keys,  # all boards + shelves grabbed + attached
        "container": None,
        "cue_stands": {
            side_r_witness_key: (
                back_xy[0],
                back_xy[1] - 1.0,
                0.0,
                0.0,
                -1.0,
            )
        },
    }
    if program_variant == "orientation_recovery":
        import bookcase_intervention

        scene.intervention_contract = dict(
            bookcase_intervention.ORIENTATION_RECOVERY_CONTRACT
        )

    # ---- scene flags (copy closing builder's set) -------------------------------------------------
    scene.plan_nav = True
    scene.soft_init = True
    scene.precise_nav = True
    scene.unified_geom = True
    scene.post_action_check = True
    scene.tpv_chase = False  # bookcase: a FIXED 3/4 cam framing the ASSEMBLY (the human-chase zoomed on the back, hiding the carcass + stand-up)
    scene.doors = {}  # bookcase: assembly is open-hall (the hall door is far off any path); clear it so the shared TPV gate (doors OR chase) uses our FIXED cam, not the door/chase TPV
    scene.auto_expose = True
    scene.expose_z = 1.8
    scene.expose_xy = back_xy
    scene.fpv_headlamp = True

    # ---- lighting: 1-2 SphereLights over the assembly area + a DomeLight ---------------------------
    _register_mdl()
    light = UsdLux.SphereLight.Define(stage, "/World/AssemblyLight1")
    light.CreateRadiusAttr(0.15)
    light.CreateIntensityAttr(8000.0)
    light.CreateColorAttr((1.0, 0.98, 0.95))
    fpv_scene.set_world_pose(
        light.GetPrim(),
        (back_xy[0] + 0.5, back_xy[1], back_z + 2.5),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    )
    light2 = UsdLux.SphereLight.Define(stage, "/World/AssemblyLight2")
    light2.CreateRadiusAttr(0.15)
    light2.CreateIntensityAttr(8000.0)
    light2.CreateColorAttr((1.0, 0.98, 0.95))
    fpv_scene.set_world_pose(
        light2.GetPrim(),
        (back_xy[0] - 0.5, back_xy[1], back_z + 2.5),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    )
    dome = UsdLux.DomeLight.Define(stage, "/World/AssemblyDome")
    dome.CreateIntensityAttr(500.0)

    # ---- TPV camera framing the assembly spot -----------------------------------------------------
    # FIXED 3/4 cam pulled WIDE + high so the whole flat layout AND the ~2 m unit (once stood upright, it
    # rises + shifts to one side) both stay fully in frame.
    tpv_look = (back_xy[0], back_xy[1], 1.0)
    tpv_eye = (back_xy[0] - 6.8, back_xy[1] - 6.8, 4.8)

    print(
        f"[bookcase] {scene_name}: {len(bt.CARCASS_EDGES)} carcass edges + {n_shelves} shelves; "
        f"assembly=({back_xy[0]:.2f},{back_xy[1]:.2f})"
    )
    print(f"[bookcase] program: {program}")
    return scene, {}, program, tpv_look, tpv_eye


def build_behavior_bookcase_orientation_recovery(
    stage, app, scene_name="hall_arch_wood"
):
    """Bookcase assembly with one HoloAssist-aligned orientation error and full repair."""
    return build_behavior_bookcase(
        stage,
        app,
        scene_name,
        program_variant="orientation_recovery",
    )


def build_behavior_task(stage, app, scene_name, task_name):
    """Run a 2026 BEHAVIOR-1K CHALLENGE task as a human-FPV demo. Loads the pre-sampled task instance
    (objects already placed for :init), derives the graspables + goal CONTAINER from the task BDDL :goal
    (via metadata.task.inst_to_name), builds each graspable as a rig object (the grabnat multi-object render
    path), registers the container with INTERIOR geometry (a real INSIDE drop), STATICALLY opens a lidded
    container so the drop reads cleanly, starts the human at the task robot pose, and emits the hand-authored
    PutInside program. The task goal is scored from its own BDDL (bddl_goal, in run_render). Additive: reuses
    _compose_behavior_scene + the grabnat multi-graspable path; existing scenes byte-identical.

    v1 verified target: composting_waste@Rs_int. Returns the step3 5-tuple (scene, {}, program, tpv, eye)."""
    import math

    import bddl_goal
    import fpv_scene
    import numpy as np
    from pxr import UsdLux
    from scene_state import Role

    _here = os.path.dirname(os.path.abspath(__file__))
    bddl_path = os.path.join(
        _here,
        "..",
        "BEHAVIOR-1K",
        "bddl3",
        "bddl",
        "activity_definitions",
        task_name,
        "problem0.bddl",
    )
    objects, i2n, robot_start, _joints = scene_objects_task(scene_name, task_name)
    by_name = {o["name"]: o for o in objects}
    # Relocate task objects into the nav-connected open region (see _TASK_POS_OVERRIDE) BEFORE compose, so
    # the container prim is placed at the override and the graspables register/settle there. x,y only.
    for _nm, _xy in _TASK_POS_OVERRIDE.get(task_name, {}).items():
        if _nm in by_name:
            _z = by_name[_nm]["pos"][2]
            by_name[_nm]["pos"] = [float(_xy[0]), float(_xy[1]), float(_z)]
            print(f"[behavior-task] pos override {_nm} -> ({_xy[0]:.2f},{_xy[1]:.2f})")

    # --- derive graspables + containers from the BDDL :goal (EXPANDED over forall/exists instances) -----
    # existential_out collects {food: [candidate tupperwares]} so the runtime can faithfully pick a REACHABLE
    # witness (an `exists` goal is satisfied by ANY instance) instead of the arbitrary first one.
    _existential: dict = {}
    _goal_preds = bddl_goal.bddl_to_goal(bddl_path, i2n, existential_out=_existential)
    grasp_names, container_names, surface_names = [], [], []
    for p in _goal_preds:
        if p.pred.name == "INSIDE":
            if p.a in by_name and p.a not in grasp_names:
                grasp_names.append(p.a)
            if p.b in by_name and p.b not in container_names:
                container_names.append(p.b)
        elif p.pred.name == "ON":
            if p.a in by_name and p.a not in grasp_names:
                grasp_names.append(p.a)
            if p.b in by_name and p.b not in surface_names:
                surface_names.append(p.b)
    if not grasp_names:
        raise RuntimeError(
            f"build_behavior_task({scene_name}/{task_name}): goal has no graspable object (grasp={grasp_names})"
        )
    if not USE_TASK_LAYOUT and not container_names:
        raise RuntimeError(
            f"build_behavior_task({scene_name}/{task_name}): no inside(obj,container) in the goal "
            f"(grasp={grasp_names} cont={container_names}); the legacy heuristic targets inside-a-container tasks"
        )
    _MAX_GRASP = 64  # safety ceiling only (was 4): render the FULL object set so a forall-goal can reach
    # full BDDL credit; no task in the challenge exceeds this, so the truncation below never fires in practice.
    if len(grasp_names) > _MAX_GRASP:
        print(
            f"[behavior-task] {task_name}: {len(grasp_names)} objects -> demo a subset of {_MAX_GRASP}"
        )
        grasp_names = grasp_names[:_MAX_GRASP]

    # --- compose the BEHAVIOR furniture (SKIP graspables -> they become rig objects) ------------------
    scene, placed, surfaces, centroid, floor_top, dz = _compose_behavior_scene(
        stage,
        app,
        scene_name,
        None,
        None,
        None,
        objects=objects,
        skip_names=set(grasp_names),
    )

    if USE_TASK_LAYOUT:
        # Use the challenge instance's OWN object placements + human start (no heuristic re-staging).
        return _build_task_real_layout(
            stage,
            app,
            scene,
            placed,
            centroid,
            dz,
            by_name,
            grasp_names,
            _goal_preds,
            scene_name,
            task_name,
            i2n,
            robot_start,
            bddl_path,
            _existential,
        )

    print(
        f"[behavior-task][diag] {scene_name}/{task_name}: placed={len(placed)} "
        f"surfaces={len(surfaces)} floor_top={floor_top} dz={dz:.3f} "
        f"skip={sorted(grasp_names)} tops={sorted(round(s[2], 2) for s in surfaces)[:14]}"
    )
    if not surfaces:
        raise RuntimeError(
            f"build_behavior_task({scene_name}/{task_name}): no comfortable work surface found"
        )

    # Reuse build_behavior_wgp's PROVEN approachability placement (nav-clean on Rs_int + others): scraps on
    # the CLEAREST reachable work surface, the goal CONTAINER on the floor by the CLEAREST DISTINCT reachable
    # spot -> both stands are nav-connected, so the human walks the planned path (through doorways, around
    # walls) instead of straight-lining. The task's exact sampled positions are NOT used (they can be walled-
    # off in tight kitchens); the task GOAL (scraps inside the can) still holds.
    def _edge(table):
        _n, _pp, top, (cx, cy), (hx, hy), _a = table
        d = np.array([cx, cy], dtype=np.float64) - centroid
        nn = float(np.linalg.norm(d))
        fwd = d / nn if nn > 1e-6 else np.array([0.0, -1.0])
        yaw = math.degrees(math.atan2(float(fwd[0]), -float(fwd[1])))
        edge = min(
            hx / max(abs(float(fwd[0])), 1e-6), hy / max(abs(float(fwd[1])), 1e-6)
        )
        d_in = max(0.05, edge - 0.10)
        ex, ey = float(cx) - float(fwd[0]) * d_in, float(cy) - float(fwd[1]) * d_in
        return (
            yaw,
            (ex, ey),
            (float(fwd[0]), float(fwd[1])),
            (float(hx), float(hy)),
            float(top),
            (float(cx), float(cy)),
        )

    boxes = scene.solid_boxes()

    def _stand_xy(table):
        _y, edge, fwd, _hh, _t, _c = _edge(table)
        return (edge[0] - fwd[0] * 0.55, edge[1] - fwd[1] * 0.55)

    def _clutter(stand, self_name):
        c = 0
        for nm, mn, mx in boxes:
            if nm == self_name:
                continue
            dx = max(float(mn[0]) - stand[0], 0.0, stand[0] - float(mx[0]))
            dy = max(float(mn[1]) - stand[1], 0.0, stand[1] - float(mx[1]))
            if dx * dx + dy * dy < 0.55 * 0.55:
                c += 1
        return c

    scored = sorted(surfaces, key=lambda t: (_clutter(_stand_xy(t), t[0]), -t[5]))
    pick = scored[0]
    p_yaw, p_edge, p_fwd, (p_hx, p_hy), p_top, p_ctr = _edge(pick)
    cand = [
        s
        for s in scored[1:]
        if math.hypot(s[3][0] - pick[3][0], s[3][1] - pick[3][1]) >= 0.8
    ]
    put = (
        min(
            cand,
            key=lambda s: (
                _clutter(_stand_xy(s), s[0]),
                math.hypot(s[3][0] - pick[3][0], s[3][1] - pick[3][1]),
            ),
        )
        if cand
        else pick
    )
    q_yaw, q_edge, q_fwd, (q_hx, q_hy), q_top, q_ctr = _edge(put)

    # scraps: rig objects along the PICK surface near-edge (all grabbed from one clear, reachable stand).
    perp = (-p_fwd[1], p_fwd[0])
    scene.grasp_objs = []
    for k in grasp_names:
        o = by_name[k]
        sc = o.get("scale", [1.0, 1.0, 1.0])
        sca = float(sum(sc) / len(sc)) if isinstance(sc, (list, tuple)) else float(sc)
        scene.grasp_objs.append(
            {
                "key": k,
                "path": f"/World/Obj_{_safe(k)}",
                "build": behavior_build_params(f"{o['cat']}/{o['model']}", sca),
            }
        )
    n_s = len(grasp_names)
    # Place graspables over the pick surface's REAL collision solid, not the visual near-edge (which can
    # overhang open floor or an adjacent chair -> the object falls / settles onto the wrong surface, then is
    # out of reach). Derive the near-edge + width from the collision bbox; the stand stays IN FRONT of each
    # object (foot_xy=object xy) so wherever on the solid it sits, the reach is short. Fall back to the
    # visual edge if the surface has no collision meshes.
    _pcb = _collision_bbox(stage, pick[1])
    if _pcb is not None:
        (_cmn, _cmx) = _pcb
        _ccx, _ccy = 0.5 * (_cmn[0] + _cmx[0]), 0.5 * (_cmn[1] + _cmx[1])
        _chx, _chy = 0.5 * (_cmx[0] - _cmn[0]), 0.5 * (_cmx[1] - _cmn[1])
        _cedge = min(_chx / max(abs(p_fwd[0]), 1e-6), _chy / max(abs(p_fwd[1]), 1e-6))
        _cin = max(
            0.06, _cedge - 0.10
        )  # inset from the SOLID near-edge so the object sits ON the solid
        g_ex, g_ey = _ccx - p_fwd[0] * _cin, _ccy - p_fwd[1] * _cin
        g_top = float(_cmx[2])
        perp_half = abs(_chx * perp[0]) + abs(_chy * perp[1])
    else:
        g_ex, g_ey, g_top, perp_half = p_edge[0], p_edge[1], float(p_top), 0.30
    spacing = 0.13
    if (
        n_s - 1
    ) * spacing > 1.4 * perp_half:  # keep the whole row within the solid's width
        spacing = 1.4 * perp_half / max(n_s - 1, 1)
    for i, k in enumerate(grasp_names):
        off = (i - (n_s - 1) / 2.0) * spacing
        sx, sy = g_ex + perp[0] * off, g_ey + perp[1] * off
        scene.register(
            k,
            Role.GRASPABLE,
            xy=(float(sx), float(sy)),
            z=float(g_top) + 0.05,
            prim_path=f"/World/Obj_{_safe(k)}",
            approach_yaw=p_yaw,
            foot_xy=(
                float(sx),
                float(sy),
            ),  # stand IN FRONT of each object (straight fwd reach, not sideways)
            foot_half=(0.12, 0.12),
            top_z=float(g_top),
            clearance=0.34,  # stand close -> the small-scrap grab is a short, reliable reach
        )

    # container: MOVE the trash-can prim onto the floor at the PUT near-edge (clear reachable spot a short
    # walk from the pick), register it as a CONTAINER with interior geometry, and hide any lid.
    cont = container_names[0]
    cpp = placed.get(cont, (None,))[0]
    co = by_name.get(cont, {})
    # Place the can on CLEAR FLOOR near the pick stand (same nav-connected region as the scraps) so disposal
    # is reachable by construction AND not on top of a chair/table. Search candidate offsets around the pick
    # stand and take the first that is not inside any furniture footprint (a far auto-'put' surface can be
    # clutter-clear yet nav-isolated; a naive side offset can land on a dining chair).
    _pstand = (p_edge[0] - p_fwd[0] * 0.55, p_edge[1] - p_fwd[1] * 0.55)

    def _in_furniture(x, y):
        for nm, mn, mx in boxes:
            if nm == pick[0]:
                continue
            if (
                float(mn[0]) - 0.12 <= x <= float(mx[0]) + 0.12
                and float(mn[1]) - 0.12 <= y <= float(mx[1]) + 0.12
            ):
                return True
        return False

    _cands = [
        (perp[0] * 0.95, perp[1] * 0.95),
        (-perp[0] * 0.95, -perp[1] * 0.95),
        (-p_fwd[0] * 0.75, -p_fwd[1] * 0.75),
        (perp[0] * 0.7 - p_fwd[0] * 0.6, perp[1] * 0.7 - p_fwd[1] * 0.6),
        (-perp[0] * 0.7 - p_fwd[0] * 0.6, -perp[1] * 0.7 - p_fwd[1] * 0.6),
    ]
    _sx0, _sy0 = scene.start_xy
    can_xy = None
    for _dx, _dy in _cands:
        _cx2, _cy2 = _pstand[0] + _dx, _pstand[1] + _dy
        if _in_furniture(_cx2, _cy2):
            continue
        if (
            math.hypot(_cx2 - _sx0, _cy2 - _sy0) < 0.7
        ):  # keep off the start cell (would break nav connectivity)
            continue
        can_xy = (float(_cx2), float(_cy2))
        break
    if can_xy is None:
        can_xy = (
            float(_pstand[0] + perp[0] * 0.95),
            float(_pstand[1] + perp[1] * 0.95),
        )
    _cv = (can_xy[0] - _pstand[0], can_xy[1] - _pstand[1])
    _cn = math.hypot(_cv[0], _cv[1]) or 1.0
    can_yaw = math.degrees(math.atan2(_cv[0] / _cn, -_cv[1] / _cn))
    if cpp is not None and co:
        _cr = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(cpp))
        _cur_cz = 0.5 * (float(_cr.GetMin()[2]) + float(_cr.GetMax()[2]))
        fpv_scene.set_world_pose(
            stage.GetPrimAtPath(cpp),
            (can_xy[0], can_xy[1], _cur_cz),
            _quat_wxyz(co["ori"]),
            tuple(co["scale"]),
        )
        rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(cpp))
        mn, mx = rng.GetMin(), rng.GetMax()
        # If the container was SAMPLED on a surface (bottom well above the floor), moving it to a floor XY
        # leaves it FLOATING at chest height -> the human reaches UP into thin air and the box clips the body.
        # Drop it so its bottom rests ON the floor. A floor-standing bin (bottom already ~floor) is left
        # untouched (byte-identical for demos whose container was floor-sampled, e.g. the trash can / bin).
        if float(mn[2]) > float(scene.floor_z) + 0.10:
            _drop = float(mn[2]) - float(scene.floor_z)
            fpv_scene.set_world_pose(
                stage.GetPrimAtPath(cpp),
                (can_xy[0], can_xy[1], _cur_cz - _drop),
                _quat_wxyz(co["ori"]),
                tuple(co["scale"]),
            )
            rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(cpp))
            mn, mx = rng.GetMin(), rng.GetMax()
        hx, hy = 0.5 * (mx[0] - mn[0]), 0.5 * (mx[1] - mn[1])
        inner_z = float(mn[2]) + 0.06
        drop_z = float(mx[2]) - 0.05
        inner_h = max(0.12, drop_z - inner_z)
        scene.register(
            cont,
            Role.CONTAINER,
            xy=can_xy,
            z=inner_z,
            height=inner_h,
            half=(max(0.05, hx * 0.80), max(0.05, hy * 0.80)),
            prim_path=cpp,
            approach_yaw=can_yaw,
            foot_xy=can_xy,
            foot_half=(float(hx), float(hy)),
            top_z=float(mx[2]),
            clearance=0.30,
        )
        try:
            _static_open_container(stage, cpp, co.get("cat"), co.get("model"))
        except Exception as _exc:  # noqa: BLE001
            print(f"[behavior-task] static-open lid skipped for {cont}: {_exc}")
    scene.behavior_support_prims = [pick[1]]  # settle the scraps on the pick surface
    work_xy = (float(p_ctr[0]), float(p_ctr[1]))

    # --- human start at the task's robot pose (near-hint; the init gate places near it) ----------------
    rs = robot_start.get("position") or robot_start.get(
        "pos"
    )  # challenge templates use "position"
    if rs:
        scene.start_xy = (float(rs[0]), float(rs[1]))
    scene.near_start = True
    scene.plan_nav = True
    # This BEHAVIOR kitchen is tight: reaching a counter object forces the body near the (boundingCube-
    # approximated) counter/cabinets, so the conservative init stand-gate flags it even though the runtime
    # nudge keeps the actual motion clear (--check: 0 body-vert-in-box). Opt into SOFT-INIT so the --seed
    # render PROCEEDS with the built nav grid + nudge (path-planning around walls) instead of aborting.
    scene.soft_init = True
    # S1 CAMERA: FPV uses the ORIGINAL tuned drive_cams (no FPV aim overrides); KEEP tpv_chase for the TPV
    # (robust in enclosed rooms). Consistent with _build_task_real_layout. (Non-USE_TASK_LAYOUT fallback,
    # currently inactive.)
    scene.tpv_chase = True

    # --- lights + dome + auto-exposure over the work area (pick surface + container) -------------------
    cont0 = scene.entities[container_names[0]]
    for _tag, (wx0, wy0), wz0 in (
        ("Work", work_xy, 1.6),
        ("Cont", cont0.xy, float(cont0.top_z) + 1.0),
    ):
        lt = UsdLux.SphereLight.Define(stage, f"/World/BehaviorLight{_tag}")
        lt.CreateRadiusAttr(0.6)
        lt.CreateIntensityAttr(60000.0)
        lt.CreateColorAttr((1.0, 1.0, 1.0))
        fpv_scene.set_world_pose(lt.GetPrim(), (float(wx0), float(wy0), float(wz0)))
    _dome = UsdLux.DomeLight.Define(stage, "/World/BehaviorDome")
    _dome.CreateIntensityAttr(400.0)
    _dome.CreateColorAttr((0.72, 0.80, 0.92))
    scene.auto_expose = True
    scene.expose_xy = work_xy
    scene.expose_z = 1.0
    scene.fpv_headlamp = True  # enclosed BEHAVIOR house: rig a fixed FPV fill so the egocentric view isn't black

    _register_mdl()

    # --- the hand-authored PutInside program + the task goal for scoring (bddl_goal in run_render) -----
    cont = container_names[0]
    program = ", ".join(
        f"[Walk] <{k}> (1), [Grab] <{k}> (1), [Walk] <{cont}> (1), [PutInside] <{k}> (1) <{cont}> (1)"
        for k in grasp_names
    )
    scene.behavior_task = {
        "scene": scene_name,
        "task": task_name,
        "bddl": bddl_path,
        "inst_to_name": i2n,
        "grasp": list(grasp_names),
        "container": container_names[0],
    }

    wx, wy = work_xy
    tpv_look = (wx, wy, 1.0)
    tpv_eye = (wx - 2.6, wy - 2.6, 3.0)
    print(
        f"[behavior-task] {scene_name}/{task_name}: {len(placed)} furniture, "
        f"grasp={grasp_names} -> container={cont} dz={dz:.3f} "
        f"start=({scene.start_xy[0]:.2f},{scene.start_xy[1]:.2f})"
    )
    return scene, {}, program, tpv_look, tpv_eye


def build_behavior_chemlab(stage, app, scene_name="school_chemistry"):
    """Benchmark Task 2 -- chemistry-lab "close-down: safe reagent segregation" as a MULTI-ROOM scene.

    Consumes the single-source-of-truth data tables in ``chemlab_task`` (ITEMS / CONTAINERS /
    build_task_fields) and stages them across ``school_chemistry``'s three connected zones:
      * CHEMISTRY_LAB -- reagent bottles left out on the lab benches; return each to a COMPATIBLE
        storage unit (3 cabinets/bins + the vented fume hood for volatiles), reading each GHS decal.
      * CORRIDOR      -- a broken-glass spill isolated to the sharps bin near the lab exit.
      * BATHROOM      -- the prep sink + regular trash (negative-only: never legal destinations).
    Reagents are seated on the COLLISION benchtop board of the two lab_tables farthest from the hood
    (a NEW benchtop-finder skips the OVERHEAD reagent shelf baked into base_link so ``_seat_on``'s
    plain collision-bbox top would mis-seat onto the shelf). The fume hood has NO door joint -- its
    open DECK is registered via a local ``_register_deck`` (drop just above the glass sash) and
    scored INSIDE HONESTLY by ``_relations_inside_live`` on the outer-bbox z-band. ``toxics`` is the
    scene's revolute cabinet; the hazard-class bins (flammables/oxidizers/acids/bases) + the corridor
    sharps bin are imported open bins sited by the clearance slot search. One reagent is pre-placed
    INSIDE ``toxics`` (a mis-store) and the reference program relocates it. Each reagent also carries a
    GHS decal (``reagent_<key>.png``) parented under its own graspable prim so the label rides the bottle.
    Per-item + NEGATIVE scoring comes from ``chemlab_task.build_task_fields()``
    via ``scene.behavior_task``, read by episode_runtime's gated scorer. Additive: a NEW multi-room
    builder cloning the seams of ``build_behavior_closing``; existing scenes byte-identical. Returns
    the step3 5-tuple (scene, {}, program, tpv_look, tpv_eye). Data table: simulator.physical_world/chemlab_task.py."""
    import math

    import chemlab_task as ct
    import constraint_evaluator as rt
    import fpv_scene
    import numpy as np
    from pxr import Usd, UsdGeom, UsdLux
    from scene_state import Role

    # This task is multi-room and fixed to ct.SCENE (school_chemistry). A caller passing the legacy
    # single-room hint is redirected so --scene bt_chemlab composes the correct multi-room scene.
    if scene_name in (None, "restaurant_diner"):
        scene_name = ct.SCENE

    scene, placed, _surfaces, centroid, _floor_top, _dz = _compose_behavior_scene(
        stage, app, scene_name, None, None, None
    )
    cx0, cy0 = float(centroid[0]), float(centroid[1])

    # Room anchors (world xy; x,y unaffected by the floor-top dz shift). Used to filter furniture by
    # room and to site the imported containers in their own room. From the verified school_chemistry
    # layout (chemistry_lab_0 ~(-5.4,12.3), corridor_1 ~(0.2,2.6), bathroom_0 ~(-5.2,-1.2)); prims
    # are still resolved GEOMETRICALLY (nearest-to-anchor / farthest-from-hood), never hardcoded.
    ROOM = {
        "chemistry_lab": (-4.5, 12.5),
        "corridor": (1.0, 5.5),
        "bathroom": (-5.2, -1.2),
    }

    def _wbbox(pp):
        rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(pp))
        mn, mx = rng.GetMin(), rng.GetMax()
        return (
            (float(mn[0]), float(mn[1]), float(mn[2])),
            (float(mx[0]), float(mx[1]), float(mx[2])),
        )

    def _ctr(pp):
        (mnx, mny, _z), (mxx, mxy, _z2) = _wbbox(pp)
        return (0.5 * (mnx + mxx), 0.5 * (mny + mxy))

    def _yaw_from(anchor, x, y):
        # Face a target from the ROOM interior (open side): body fwd = (sin,-cos) points anchor -> target.
        dx, dy = float(x) - anchor[0], float(y) - anchor[1]
        n = math.hypot(dx, dy) or 1.0
        return math.degrees(math.atan2(dx / n, -dy / n))

    def _best_yaw(ux, uy, boxes):
        # openest cardinal side (max furniture-clearance ~1 m out): the approach for a bin / the deck.
        def _cd(x, y):
            best = 1e9
            for _n, mn, mx in boxes:
                dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
                dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
                best = min(best, math.hypot(dx, dy))
            return best

        best_a, best_c = 0.0, -1.0
        for a_deg in (0.0, 90.0, 180.0, 270.0):
            a = math.radians(a_deg)
            c = _cd(ux - math.sin(a) * 1.0, uy + math.cos(a) * 1.0)
            if c > best_c:
                best_a, best_c = a_deg, c
        return best_a

    _boxes0 = (
        scene.solid_boxes()
    )  # furniture footprints (items/imports not registered yet)

    def _clear_at(x, y, exclude):
        best = 1e9
        for nm, mn, mx in _boxes0:
            if nm == exclude:
                continue
            dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
            dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
            best = min(best, math.hypot(dx, dy))
        return best

    # ---- fume hood + the two reagent benches farthest from it ------------------------------------
    hood_cands = [
        (nm, pp) for nm, (pp, o) in placed.items() if o.get("cat") == "lab_exhaust_hood"
    ]
    if not hood_cands:
        raise RuntimeError("chemlab: no lab_exhaust_hood (fume hood) in the scene")
    hood_cands.sort(
        key=lambda t: math.hypot(
            _ctr(t[1])[0] - ROOM["chemistry_lab"][0],
            _ctr(t[1])[1] - ROOM["chemistry_lab"][1],
        )
    )
    hood_nm, hood_pp = hood_cands[0]
    (_hmn, _hmx) = _collision_bbox(stage, hood_pp) or _wbbox(hood_pp)
    _hfp = ((float(_hmn[0]), float(_hmn[1])), (float(_hmx[0]), float(_hmx[1])))
    _hdim = max(float(_hmx[0] - _hmn[0]), float(_hmx[1] - _hmn[1]))

    def _fp_gap_to_hood(
        pp,
    ):  # nearest-point footprint distance from a table to the hood footprint
        (mn, mx) = _collision_bbox(stage, pp) or _wbbox(pp)
        (hmn, hmx) = _hfp
        dx = max(hmn[0] - float(mx[0]), 0.0, float(mn[0]) - hmx[0])
        dy = max(hmn[1] - float(mx[1]), 0.0, float(mn[1]) - hmx[1])
        return math.hypot(dx, dy)

    lab_tables = [
        (nm, pp) for nm, (pp, o) in placed.items() if o.get("cat") == "lab_table"
    ]
    # hood-safe: a bench whose footprint sits >= 0.5*hood-size clear of the hood cannot false-positive
    # INSIDE the hood (a reagent seated on its near edge stays out of the hood's xy footprint). Among
    # those, the two NEAREST the lab work anchor are the reachable reagent benches -- the central bench
    # run, clear of the hood at the room's far end. Every threshold is a fraction of the hood's OWN size.
    _gap_thresh = 0.5 * _hdim
    safe = [(nm, pp) for nm, pp in lab_tables if _fp_gap_to_hood(pp) >= _gap_thresh]
    if len(safe) < 2:  # relax (never crash): take the two farthest from the hood
        safe = sorted(lab_tables, key=lambda t: -_fp_gap_to_hood(t[1]))[:2]
    safe.sort(
        key=lambda t: math.hypot(
            _ctr(t[1])[0] - ROOM["chemistry_lab"][0],
            _ctr(t[1])[1] - ROOM["chemistry_lab"][1],
        )
    )
    benches = safe[:2]
    if len(benches) < 2:
        raise RuntimeError(
            f"chemlab: need 2 reagent benches, found {len(benches)} (lab_tables={[t[0] for t in lab_tables]})"
        )
    # order the benches so bench[0] is the ENTRY bench (nearest the corridor exit) -> start there.
    benches.sort(
        key=lambda t: math.hypot(
            _ctr(t[1])[0] - ROOM["corridor"][0], _ctr(t[1])[1] - ROOM["corridor"][1]
        )
    )
    print(
        f"[chemlab] hood={hood_nm}@{_ctr(hood_pp)}; benches="
        f"{[(nm, _ctr(pp), round(_fp_gap_to_hood(pp), 2)) for nm, pp in benches]}"
    )

    def _bench_top(table_pp):
        # (top_z, cx, cy, hx, hy) of the reachable BENCHTOP board: the largest thin horizontal collision
        # board whose top is BELOW ~0.75*table_height above the floor, so the OVERHEAD reagent shelf baked
        # into base_link (which drives the plain collision-bbox top ~1.4 m) is skipped and reagents seat on
        # the ~0.9 m benchtop. Every threshold is a fraction of the table's OWN bbox (no magic metres).
        cb = _collision_bbox(stage, table_pp)
        (cmn, cmx) = cb if cb is not None else _wbbox(table_pp)
        W, D, H = (
            float(cmx[0] - cmn[0]),
            float(cmx[1] - cmn[1]),
            float(cmx[2] - cmn[2]),
        )
        floor = float(scene.floor_z)
        band_cap = floor + 0.75 * H
        root = stage.GetPrimAtPath(table_pp)
        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=True,
        )
        boards = []
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if p.GetTypeName() != "Mesh" or "/collisions/" not in str(p.GetPath()):
                continue
            rng = bc.ComputeWorldBound(p).ComputeAlignedRange()
            a, b = rng.GetMin(), rng.GetMax()
            if not (b[0] >= a[0]):
                continue
            w, d, h = float(b[0] - a[0]), float(b[1] - a[1]), float(b[2] - a[2])
            if h > 0.15 * H or w < 0.4 * W or d < 0.4 * D:
                continue  # not a wide, thin, horizontal board (a wall / the body / a bracket)
            top = float(b[2])
            if top > band_cap:
                continue  # the overhead reagent shelf -> above the reachable band
            boards.append(
                (
                    w * d,
                    top,
                    0.5 * (a[0] + b[0]),
                    0.5 * (a[1] + b[1]),
                    0.5 * (b[0] - a[0]),
                    0.5 * (b[1] - a[1]),
                )
            )
        if boards:
            # The benchtop is the TOPMOST wide work surface in the reachable band; lower wide boards are
            # base-cabinet panels/shelves and the (shallow) overhead reagent shelf is already excluded by
            # the depth filter -- so pick HIGHEST top_z (area as tiebreak), not largest area.
            boards.sort(
                key=lambda t: (t[1], t[0]), reverse=True
            )  # highest board, then largest area
            _ar, top, bx, by, bhx, bhy = boards[0]
            return float(top), float(bx), float(by), float(bhx), float(bhy)
        # fallback (no board found): collision-bbox top clamped into the reachable band
        return (
            min(float(cmx[2]), band_cap),
            0.5 * (cmn[0] + cmx[0]),
            0.5 * (cmn[1] + cmx[1]),
            0.5 * (cmx[0] - cmn[0]),
            0.5 * (cmx[1] - cmn[1]),
        )

    def _bench_side(table_pp, self_name):
        # clearest cardinal approach side (>=0.5 m open front-stand), tie-broken toward the lab interior
        # (the open aisle), so the reagent stand lands on open floor -- not against the wall / top cabinets.
        cb = _collision_bbox(stage, table_pp)
        (cmn, cmx) = cb if cb is not None else _wbbox(table_pp)
        cx, cy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
        hx, hy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
        best = None
        for ang in (0.0, 90.0, 180.0, 270.0):
            a = math.radians(ang)
            fwd = (math.sin(a), -math.cos(a))
            hd = abs(fwd[0]) * hx + abs(fwd[1]) * hy
            sx, sy = cx - fwd[0] * (hd + 0.55), cy - fwd[1] * (hd + 0.55)
            cl = _clear_at(sx, sy, self_name)
            dist = math.hypot(
                sx - ROOM["chemistry_lab"][0], sy - ROOM["chemistry_lab"][1]
            )
            rank = (1 if cl >= 0.5 else 0, -dist if cl >= 0.5 else cl)
            if best is None or rank > best[0]:
                best = (rank, float(ang))
        return best[1]

    support_prims: list = []

    def _seat_reagents(table_pp, keys, yaw, bench):
        # Row the reagents over the benchtop board on the approach (near) edge -- each a short straight
        # reach from a stand IN FRONT. Clones _seat_on's size-aware pitch, but seats on the reachable
        # benchtop (bench), NOT _seat_on's plain collision-bbox top (which is the OVERHEAD shelf here).
        b_top, s_cx, s_cy, s_hx, s_hy = bench
        ar = math.radians(yaw)
        fwd = (
            math.sin(ar),
            -math.cos(ar),
        )  # into the table; near edge on the -fwd (approach) side
        perp = (-fwd[1], fwd[0])
        edge_depth = min(s_hx / max(abs(fwd[0]), 1e-6), s_hy / max(abs(fwd[1]), 1e-6))
        inset = max(
            0.08, edge_depth * 0.45
        )  # toward the near edge, staying ON the board
        ex, ey = s_cx - fwd[0] * inset, s_cy - fwd[1] * inset
        n_i = len(keys)
        halves = []
        for k in keys:
            try:
                _bb = read_spec(f"{ct.ITEMS[k]['cat']}/{ct.ITEMS[k]['model']}").get(
                    "bbox_size"
                ) or [0.12, 0.12, 0.1]
            except Exception:  # noqa: BLE001
                _bb = [0.12, 0.12, 0.1]
            halves.append(0.5 * max(float(_bb[0]), float(_bb[1])))
        need_pitch = (
            max(halves[i] + halves[i + 1] for i in range(n_i - 1)) if n_i > 1 else 0.0
        ) + 0.04
        spacing = max(0.18, need_pitch)
        for i, k in enumerate(keys):
            off = (i - (n_i - 1) / 2.0) * spacing
            sx, sy = ex + perp[0] * off, ey + perp[1] * off
            scene.register(
                k,
                Role.GRASPABLE,
                xy=(float(sx), float(sy)),
                z=float(b_top) + 0.05,
                prim_path=f"/World/Obj_{_safe(k)}",
                approach_yaw=float(yaw),
                foot_xy=(float(sx), float(sy)),
                foot_half=(0.10, 0.10),
                top_z=float(b_top),
                clearance=0.34,
            )
        if table_pp not in support_prims:
            support_prims.append(table_pp)

    # build EACH graspable as a rig object at its own /World/Obj_<key> prim (the multi-object render path).
    # Recolor every REAGENT body a uniform OPAQUE amber (a real amber reagent bottle): the generic bottles are
    # CLEAR GLASS, whose transparency otherwise occludes/darkens the curved label -> unreadable. An opaque
    # amber body lets the opaque+unlit label read crisply (as it already does on the opaque named bottles), and
    # is UNIFORM across reagents so it leaks NO answer (the hazard is on the LABEL, not the bottle colour). The
    # non-reagent shard (group None) keeps its own glass look. The recolor rides the grasp-obj "color" hook,
    # applied where the geometry is actually composed (episode_runtime's build loop), and episode_runtime._recolor_prim
    # skips the Label_/RLabel_ decals so the sign/label material stays crisp.
    _AMBER = (0.55, 0.34, 0.12)
    scene.grasp_objs = []
    for k in ct.ITEMS:
        cat, model = ct.ITEMS[k]["cat"], ct.ITEMS[k]["model"]
        _go = {
            "key": k,
            "path": f"/World/Obj_{_safe(k)}",
            "build": behavior_build_params(f"{cat}/{model}", 1.0),
        }
        if ct.ITEMS[k].get("group") is not None:  # a reagent -> opaque amber body
            _go["color"] = _AMBER
        scene.grasp_objs.append(_go)

    # seat the BENCH reagents (place=="bench"; the mis-stored solvent is pre-placed in toxics below),
    # distributed round-robin across the two benches so neither is overcrowded.
    bench_items = [k for k in ct.ITEMS if ct.ITEMS[k]["place"] == "bench"]
    bt_groups: dict = {i: [] for i in range(len(benches))}
    for i, k in enumerate(bench_items):
        bt_groups[i % len(benches)].append(k)
    bench_tops = []
    for i, (bnm, bpp) in enumerate(benches):
        _yaw = _bench_side(bpp, bnm)
        _bench = _bench_top(bpp)
        bench_tops.append((bnm, bpp, _yaw, _bench))
        print(
            f"[chemlab] bench {bnm}: approach_yaw={_yaw:.0f} benchtop_z={_bench[0]:.3f} "
            f"(collbbox_top={(_collision_bbox(stage, bpp) or _wbbox(bpp))[1][2]:.3f})"
        )
        if bt_groups[i]:
            _seat_reagents(bpp, bt_groups[i], _yaw, _bench)
    scene.behavior_support_prims = list(support_prims)

    # start the human in front of the ENTRY bench (bench[0], nearest the corridor exit); place_start
    # snaps this to the nearest free cell.
    _b0nm, _b0pp, _b0yaw, _b0bench = bench_tops[0]
    _b0top, _b0cx, _b0cy, _b0hx, _b0hy = _b0bench
    _b0fwd = (math.sin(math.radians(_b0yaw)), -math.cos(math.radians(_b0yaw)))
    _b0hd = abs(_b0fwd[0]) * _b0hx + abs(_b0fwd[1]) * _b0hy
    scene.start_xy = (
        float(_b0cx - _b0fwd[0] * (_b0hd + 0.75)),
        float(_b0cy - _b0fwd[1] * (_b0hd + 0.75)),
    )
    scene.near_start = True

    # ---- container registration helpers (cloned from build_behavior_closing) --------------------
    _LABEL_DIR = GENERATED_LABEL_DIR
    # BUILD-side gate for the pixel-observation contract: mounting a decal is a per-item skip, so a
    # `needs_lookat` item whose PNG was never authored would render an episode whose declared observation
    # channel does not exist -- unsolvable by design, discovered only after the render. Fail here instead.
    ct.validate_lookat_decals(_LABEL_DIR)
    front_dir: dict = {}  # reg_name -> unit-centre->door/approach unit vector (label facing)
    station_front: dict = {}  # (unused here; kept so _label_decal is a verbatim clone)
    reg_pp: dict = {}  # reg_name -> prim path
    used_slots: list = []  # (x, y, keep-out radius) for imported units

    def _register_swing(reg_name, pp, obj):
        of = _closing_leaf_fields(scene, pp, obj)
        (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(pp)
        tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        thx, thy = max(0.05, 0.5 * (mxx - mnx)), max(0.05, 0.5 * (mxy - mny))
        if not of or of.get("kind") != "swing":
            return False
        if of.get("leaf_bbox"):
            (_lmn, _lmx) = of["leaf_bbox"]
            fdx, fdy = 0.5 * (_lmn[0] + _lmx[0]) - tcx, 0.5 * (_lmn[1] + _lmx[1]) - tcy
            _fn = math.hypot(fdx, fdy) or 1.0
            fdir = (fdx / _fn, fdy / _fn)
        else:
            _a0 = math.radians(_best_yaw(tcx, tcy, scene.solid_boxes()))
            fdir = (-math.sin(_a0), math.cos(_a0))
        ayaw = math.degrees(math.atan2(-fdir[0], fdir[1]))
        front_dir[reg_name] = fdir
        hd = abs(fdir[0]) * thx + abs(fdir[1]) * thy
        front_face = (tcx + fdir[0] * hd, tcy + fdir[1] * hd)
        inset = min(0.22, max(0.0, hd - 0.05))
        front_int = (tcx + fdir[0] * (hd - inset), tcy + fdir[1] * (hd - inset))
        _lo, _hi = float(mnz) + 0.08, float(mxz) - 0.12
        _clampz = lambda v: min(max(v, min(_lo, _hi)), max(_lo, _hi))  # noqa: E731
        shelf_z = _clampz(scene.floor_z + 0.90)  # reachable interior drop height
        handle_z = _clampz(scene.floor_z + 1.00)
        inner_z = float(mnz) + 0.06
        inner_h = max(0.15, (float(mxz) - 0.1) - inner_z)
        shv = _extract_spaces(scene.stage, pp, reg_name)
        scene.register(
            reg_name,
            Role.CONTAINER,
            xy=front_int,
            z=inner_z,
            height=inner_h,
            half=(max(0.05, thx * 0.8), max(0.05, thy * 0.8)),
            prim_path=pp,
            approach_yaw=ayaw,
            foot_xy=(tcx, tcy),
            foot_half=(thx, thy),
            top_z=float(mxz),
            clearance=0.35,
            hinge_xy=of["hinge_xy"],
            hinge_axis=(0.0, 0.0, 1.0),
            swing_angle=of["swing_angle"],
            swing_sign=1.0,
            leaf_paths=of["leaf_paths"],
            leaf_hinges=of.get("leaf_hinges", ()),
            joint_name=of["joint_name"],
            open_thresh=0.10,
            handle_xy=front_face,
            handle_z=handle_z,
            drop_z=shelf_z,
            spaces=shv,
        )
        if shv:
            print(
                f"[chemlab] swing container {reg_name}: {len(shv)} interior space(s) "
                f"{[s[0] for s in shv]} leaves={len(of['leaf_paths'])}"
            )
        return True

    def _orient_door_open(pp, cat, model, cell, grid_boxes):
        # Rotate an imported SWING unit about z so its baked door faces the openest cardinal direction -> the
        # leaf-derived stand + the door-mounted sign land on open, reachable floor (not against a wall). No leaf
        # -> identity quat. Verbatim of build_behavior_closing._orient_door_open (chemlab _wbbox/_best_yaw).
        of = _closing_leaf_fields(scene, pp, {"cat": cat, "model": model})
        if not of or not of.get("leaf_bbox"):
            return (1.0, 0.0, 0.0, 0.0)
        (mn, mx) = _wbbox(pp)
        ucx, ucy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        (lmn, lmx) = of["leaf_bbox"]
        d0x, d0y = 0.5 * (lmn[0] + lmx[0]) - ucx, 0.5 * (lmn[1] + lmx[1]) - ucy
        d0n = math.hypot(d0x, d0y) or 1.0
        d0 = (d0x / d0n, d0y / d0n)  # current baked door normal
        a = math.radians(_best_yaw(cell[0], cell[1], grid_boxes))
        d1 = (-math.sin(a), math.cos(a))  # openest cardinal direction
        off = math.atan2(d0[0] * d1[1] - d0[1] * d1[0], d0[0] * d1[0] + d0[1] * d1[1])
        return (math.cos(off / 2.0), 0.0, 0.0, math.sin(off / 2.0))

    def _register_open(reg_name, pp, cat, model):
        (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(pp)
        hx, hy = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
        tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        band = 0.05 * max(0.1, float(mxz) - float(mnz))
        inner_z = float(mnz) + 0.04
        inner_h = max(0.10, (float(mxz) - 0.03) - inner_z)
        drop_z = max(float(scene.floor_z) + 0.02, float(mxz) - band - 0.03)
        _ay = _best_yaw(tcx, tcy, scene.solid_boxes())
        _aar = math.radians(_ay)
        front_dir[reg_name] = (-math.sin(_aar), math.cos(_aar))
        scene.register(
            reg_name,
            Role.CONTAINER,
            xy=(tcx, tcy),
            z=inner_z,
            height=inner_h,
            half=(max(0.05, hx * 0.85), max(0.05, hy * 0.85)),
            prim_path=pp,
            approach_yaw=_ay,
            foot_xy=(tcx, tcy),
            foot_half=(float(hx), float(hy)),
            top_z=float(mxz),
            clearance=0.30,
            drop_z=float(drop_z),
        )
        try:
            _static_open_container(stage, pp, cat, model)
        except Exception as _exc:  # noqa: BLE001
            print(f"[chemlab] static-open lid skipped for {reg_name}: {_exc}")

    def _register_deck(reg_name, pp):
        # The fume hood has NO door joint -- register its OPEN DECK (work surface): clone _register_open
        # but drop just ABOVE the glass sash (the reachable bench-height deck), never a lid. INSIDE is
        # scored HONESTLY by _relations_inside_live on the OUTER-bbox z-band (deck items in-band, roof out).
        (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(pp)
        hx, hy = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
        tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        floor = float(scene.floor_z)
        Hh = float(mxz) - float(mnz)
        root = stage.GetPrimAtPath(pp)

        def _deinst(pm):  # de-instance so the sash sub-prim is traversable by name
            if pm.IsInstanceable():
                pm.SetInstanceable(False)
            for c in pm.GetChildren():
                _deinst(c)

        if root and root.IsValid():
            _deinst(root)
        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
            useExtentsHint=True,
        )
        deck_top, how = None, "frac"
        # (a) the glass sash sub-prim world min-z (the sash sits ON the deck; its bottom edge = surface)
        if root and root.IsValid():
            for p in Usd.PrimRange(root):
                nm = p.GetName().lower()
                if p.GetTypeName() not in ("Mesh", "Xform"):
                    continue
                if not any(k in nm for k in ("glass", "sash", "window", "pane")):
                    continue
                rng = bc.ComputeWorldBound(p).ComputeAlignedRange()
                a, b = rng.GetMin(), rng.GetMax()
                if not (b[0] >= a[0]):
                    continue
                z0 = float(a[2])
                if floor + 0.2 < z0 < floor + 0.9:  # a reachable-height sash bottom
                    deck_top = z0 if deck_top is None else min(deck_top, z0)
                    how = "glass"
        # (b) largest thin horizontal collision board with top in the reachable band
        if deck_top is None:
            shv = _extract_spaces(scene.stage, pp, "deck")
            cand = [s for s in shv if float(s[1]) < floor + 0.9]
            if cand:
                cand.sort(key=lambda s: float(s[5]) * float(s[6]), reverse=True)
                deck_top, how = float(cand[0][1]), "spaces"
        # (c) fraction of the hood's own height
        if deck_top is None:
            deck_top = float(mnz) + 0.46 * Hh
        inner_z = float(mnz) + 0.04
        inner_h = max(0.10, (float(mxz) - 0.03) - inner_z)
        drop_z = (
            float(deck_top) + 0.005
        )  # just above the deck -> minimal drop bounce/roll
        # The deck CENTRE is too deep to reach when the hood's near face is boxed in (a bench 0.35 m off its
        # front) -- so put the deck CONTACT on the clearest WALL-AWARE open side's near edge. It stays INSIDE
        # the hood's xy footprint (INSIDE is scored on the prim's OUTER bbox, not this contact), so scoring is
        # unchanged, but a stand on the open side can now reach it. _best_yaw is furniture-only (would pick a
        # wall side); include the wall footprints so a wall-facing side loses.
        _allb = list(scene.solid_boxes())
        for _wp in scene.wall_prims or []:
            try:
                (_wmn, _wmx) = _wbbox(_wp)
                _allb.append(("wall", _wmn, _wmx))
            except Exception:  # noqa: BLE001
                pass

        def _clr_pt(x, y):
            best = 1e9
            for _n, mn, mx in _allb:
                dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
                dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
                best = min(best, math.hypot(dx, dy))
            return best

        _sides = []
        for _out in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
            _ha = abs(_out[0]) * hx + abs(_out[1]) * hy
            _px, _py = tcx + _out[0] * (_ha + 0.55), tcy + _out[1] * (_ha + 0.55)
            _sides.append((_clr_pt(_px, _py), _out, _ha))
        _sides.sort(key=lambda s: -s[0])
        _best_cl, _out, _ha = _sides[0]
        cxy = (tcx + _out[0] * (0.55 * _ha), tcy + _out[1] * (0.55 * _ha))
        _ay = math.degrees(math.atan2(-_out[0], _out[1]))
        front_dir[reg_name] = (float(_out[0]), float(_out[1]))
        # An OPEN deck has no walls to contain a pile -- multiple items dropped at ONE point roll off. So
        # spread synthetic drop SPACES along the deck's reachable WIDTH (perp to the approach), centred on
        # the reachable contact, so several items land side-by-side + all stay within the deck footprint
        # (INSIDE). General: fractions of the deck's OWN size; spaces sit at the reachable inset, never the
        # deep centre. produce_place_inside auto-distributes; each is reach-validated by pick_stand_runtime.
        _perp = (-_out[1], _out[0])
        _perp_half = abs(_perp[0]) * hx + abs(_perp[1]) * hy
        _n_sp = int(
            ct.CONTAINERS.get(reg_name, {}).get("cap") or 3
        )  # one drop space per capacity slot
        deck_spaces = []
        for _i in range(_n_sp):
            _frac = (_i - (_n_sp - 1) / 2.0) / max(1, _n_sp - 1)  # -0.5 .. 0.5
            _off = _frac * (
                2.0 * _perp_half * 0.25
            )  # tight to the centerline -> stay well inside the deck
            _scx = cxy[0] + _perp[0] * _off
            _scy = cxy[1] + _perp[1] * _off
            deck_spaces.append(
                (
                    f"{reg_name}_deck{_i + 1}",
                    float(deck_top),
                    float(deck_top) - 0.02,
                    float(_scx),
                    float(_scy),
                    0.09,
                    0.09,
                )
            )
        scene.register(
            reg_name,
            Role.CONTAINER,
            xy=(float(cxy[0]), float(cxy[1])),
            z=inner_z,
            height=inner_h,
            half=(max(0.05, hx * 0.85), max(0.05, hy * 0.85)),
            prim_path=pp,
            approach_yaw=_ay,
            foot_xy=(tcx, tcy),
            foot_half=(float(hx), float(hy)),
            top_z=float(mxz),
            clearance=0.30,
            drop_z=float(drop_z),
            spaces=tuple(deck_spaces),
        )
        print(
            f"[chemlab] deck container {reg_name}: deck_top={deck_top:.3f} ({how}) "
            f"drop_z={drop_z:.3f} outer_z=[{mnz:.2f},{mxz:.2f}] "
            f"open_side={_out} contact=({cxy[0]:.2f},{cxy[1]:.2f}) clr={_best_cl:.2f} "
            f"deck_spaces={[(round(s[3], 2), round(s[4], 2)) for s in deck_spaces]}"
        )

    def _bind_label_material(mesh_prim, path, png, lit=False):
        # Label material driven by the PNG. Two modes:
        #  * lit=True (CONTAINER placards) -- diffuse AND emissive BOTH from the PNG (verbatim Task-1 fridge-label
        #    material): the sign RECEIVES the cabinet's lighting + shading, so it reads as PART OF the door, not a
        #    uniformly-glowing panel pasted on top (which looked "not stuck on the cabinet").
        #  * lit=False (REAGENT bottle decals) -- diffuse 0, pure emissive: a lit label on a bottle rendered faint
        #    / see-through, so it stays SELF-LIT + opacity-pinned to read crisply regardless of the bottle material.
        from pxr import Gf, Sdf, UsdShade

        mtl = UsdShade.Material.Define(stage, path + "/M")
        rdr = UsdShade.Shader.Define(stage, path + "/M/St")
        rdr.CreateIdAttr("UsdPrimvarReader_float2")
        rdr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        rdr.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, path + "/M/T")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            rdr.ConnectableAPI(), "result"
        )
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh = UsdShade.Shader.Define(stage, path + "/M/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        if lit:  # Task-1 fridge-label material: diffuse driven by the PNG -> receives cabinet lighting/shadows
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb"
            )
        else:  # reagent bottle decal: no lit diffuse term -> pure self-lit
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.0, 0.0, 0.0)
            )
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5 if lit else 1.0)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
            mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

    def _label_decal(reg_name, pp, sign, tall):
        # mount a REAL 2D LABEL SIGN (a PNG under _LABEL_DIR) on the unit (verbatim clone of
        # build_behavior_closing._label_decal, then adapted: open bins get a RAISED eye-level placard). If the
        # PNG is absent it is silently skipped (no crash).
        if not sign:
            return
        png = f"{_LABEL_DIR}/{sign}.png"
        if not os.path.exists(png):
            print(f"[chemlab] label sign missing, skipped: {png}")
            return
        from pxr import Gf, Sdf

        try:
            (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(pp)
        except Exception:  # noqa: BLE001
            return
        cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        hx, hy = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
        fd = front_dir.get(reg_name)
        if not fd or (abs(fd[0]) + abs(fd[1]) < 1e-6):
            a = math.radians(_best_yaw(cx, cy, scene.solid_boxes()))
            fd = (-math.sin(a), math.cos(a))
        # Locate the swinging door LEAF prim UP-FRONT (swing units): the sign mounts FLUSH on the actual door
        # panel, not on the whole-cabinet bbox (which offset + angled it). De-instance so the leaf is authorable
        # + its world bbox readable. Open bins / deck have no leaf -> _leaf_prim stays None.
        _root = stage.GetPrimAtPath(pp)
        _leaf_prim = None
        if _root and _root.IsValid():

            def _deinst(pm):
                if pm.IsInstanceable():
                    pm.SetInstanceable(False)
                for _c in pm.GetChildren():
                    _deinst(_c)

            _deinst(_root)
            _e0 = scene.entities.get(reg_name)
            _lp = (
                _e0.leaf_paths[0]
                if (_e0 and getattr(_e0, "leaf_paths", None))
                else None
            )
            if _lp:
                _lname = _lp.rstrip("/").split("/")[-1]
                _leaf_prim = next(
                    (p for p in Usd.PrimRange(_root) if p.GetName() == _lname), None
                )
        _sign_face = fd  # entity-recorded outward normal (open bin / fallback: fd; flush door: dn below)
        if tall and _leaf_prim is not None:
            # FLUSH DOOR SIGN, COPLANAR WITH THE ACTUAL DOOR PANEL. The prior version faced `front_dir`
            # (cabinet-centre -> leaf-centre), which for a MULTI-LEAF cabinet (rmmvcd) is ~20-30deg off the real
            # door-panel normal -> the sign sat on a DIFFERENT plane than the door (looked tilted, tracked the
            # camera not the door). Here we derive the panel's TRUE normal + front face from the leaf's OWN mesh
            # points (2D PCA in xy: the minor axis = thinnest spread = the panel outward normal; the max
            # projection along it = the front face), so the sign is coplanar with the door and the look-at
            # (label_face = this normal) views it head-on. A single-door fridge is unaffected (its PCA normal
            # equals front_dir), so Task-1 stays byte-identical in spirit.
            import numpy as _np

            _xc = UsdGeom.XformCache(Usd.TimeCode.Default())
            _P = []
            for _p in Usd.PrimRange(_leaf_prim):
                if _p.IsA(UsdGeom.Mesh):
                    _pts = UsdGeom.Mesh(_p).GetPointsAttr().Get()
                    if not _pts:
                        continue
                    _l2w = _xc.GetLocalToWorldTransform(_p)
                    for _v in _pts:
                        _w = _l2w.Transform(Gf.Vec3d(_v))
                        _P.append((float(_w[0]), float(_w[1]), float(_w[2])))
            if len(_P) >= 4:
                _A = _np.array(_P)
                _xy = _A[:, :2] - _A[:, :2].mean(axis=0)
                _ev, _evec = _np.linalg.eigh(_xy.T @ _xy)
                _n = _evec[
                    :, 0
                ]  # minor axis = thinnest xy spread = panel outward normal
                _n = _n / (float(_np.linalg.norm(_n)) + 1e-9)
                if (
                    float(_n[0] * fd[0] + _n[1] * fd[1]) < 0.0
                ):  # orient toward the approach side
                    _n = -_n
                _t = _np.array([-_n[1], _n[0]])  # in-plane horizontal axis OF THE PANEL
                _pn = _A[:, :2] @ _n
                _pt = _A[:, :2] @ _t
                nrm = _sign_face = (float(_n[0]), float(_n[1]))
                leaf_w = float(_pt.max() - _pt.min())
                _tc = 0.5 * (
                    float(_pt.min()) + float(_pt.max())
                )  # panel horizontal centre
                _front = float(_pn.max())  # panel front face along the outward normal
                sw = min(0.8 * leaf_w, 0.46)
                sh_ = min(0.26, 0.6 * sw)
                zc = min(
                    float(scene.floor_z) + 1.20,
                    float(_A[:, 2].max()) - sh_ * 0.5 - 0.05,
                )
                zc = max(zc, float(_A[:, 2].min()) + sh_ * 0.5 + 0.05)
                px = float(
                    (_front + 0.006) * _n[0] + _tc * _t[0]
                )  # ON the panel front + 6 mm, coplanar
                py = float((_front + 0.006) * _n[1] + _tc * _t[1])
                print(
                    f"[chemlab] door sign {reg_name}: PANEL_normal=({nrm[0]:.3f},{nrm[1]:.3f}) "
                    f"front_dir=({fd[0]:.2f},{fd[1]:.2f}) leaf_w={leaf_w:.2f} "
                    f"-> coplanar sign @({px:.2f},{py:.2f},{zc:.2f}) sw={sw:.2f}"
                )
            else:  # leaf had no readable mesh points -> fall back to the cabinet-front estimate
                nrm = _sign_face = fd
                face_w = (2.0 * hy) if abs(nrm[0]) >= abs(nrm[1]) else (2.0 * hx)
                sw = min(0.8 * float(face_w), 0.46)
                sh_ = min(0.26, 0.6 * sw)
                zc = min(float(scene.floor_z) + 1.20, float(mxz) - sh_ * 0.5 - 0.05)
                zc = max(zc, float(mnz) + sh_ * 0.5 + 0.05)
                reach = abs(nrm[0]) * hx + abs(nrm[1]) * hy
                px, py = cx + nrm[0] * (reach + 0.005), cy + nrm[1] * (reach + 0.005)
        elif tall:  # swing unit but NO leaf prim found -> mount flat on the cabinet FRONT face (degraded)
            nrm, _sign_face = fd, fd
            face_w = (2.0 * hy) if abs(nrm[0]) >= abs(nrm[1]) else (2.0 * hx)
            sw = min(0.8 * float(face_w), 0.46)
            sh_ = min(0.26, 0.6 * sw)
            zc = min(float(scene.floor_z) + 1.20, float(mxz) - sh_ * 0.5 - 0.05)
            zc = max(zc, float(mnz) + sh_ * 0.5 + 0.05)
            reach = abs(nrm[0]) * hx + abs(nrm[1]) * hy
            px, py = cx + nrm[0] * (reach + 0.005), cy + nrm[1] * (reach + 0.005)
        else:
            # OPEN unit (waste bin / deck, no door): a sign FLAT ON THE FRONT FACE (dominant face toward fd),
            # sized to that face, mid-height. NOT a floating placard.
            if (2.0 * hx) >= (2.0 * hy):
                nrm, face_w = (0.0, 1.0 if fd[1] >= 0.0 else -1.0), 2.0 * hx
            else:
                nrm, face_w = (1.0 if fd[0] >= 0.0 else -1.0, 0.0), 2.0 * hy
            sw = min(0.85 * face_w, 0.5)
            sh_ = sw * 0.62
            zc = float(mnz) + 0.58 * (float(mxz) - float(mnz))
            reach = abs(nrm[0]) * hx + abs(nrm[1]) * hy
            px, py = cx + nrm[0] * (reach + 0.02), cy + nrm[1] * (reach + 0.02)
        ux, uy = -nrm[1], nrm[0]
        hw = sw * 0.5
        corners = [
            Gf.Vec3f(float(px - ux * hw), float(py - uy * hw), float(zc - sh_ * 0.5)),
            Gf.Vec3f(float(px + ux * hw), float(py + uy * hw), float(zc - sh_ * 0.5)),
            Gf.Vec3f(float(px + ux * hw), float(py + uy * hw), float(zc + sh_ * 0.5)),
            Gf.Vec3f(float(px - ux * hw), float(py - uy * hw), float(zc + sh_ * 0.5)),
        ]
        nrm3 = Gf.Vec3f(float(_sign_face[0]), float(_sign_face[1]), 0.0)
        # Record the sign's WORLD placement on the container entity so the non-grabbable front-read look_at aims
        # SQUARELY at it -- its outward normal (xy) + world CENTRE + height. For a swing cabinet this is the sign
        # centre ON THE CLOSED DOOR FACE (label_face = the door's outward normal); the sign rides the leaf, so it
        # turns WITH the door. Containers stay put during a look-at, so a recorded world pose is exact.
        _ce = scene.entities.get(reg_name)
        if _ce is not None:
            _ce.meta["label_face"] = (float(_sign_face[0]), float(_sign_face[1]))
            _ce.meta["label_pos"] = (float(px), float(py), float(zc))
            # zoom target = the sign's WIDEST dimension, so the look-at frames the WHOLE sign at ~45% and the
            # cabinet stays visible around it (reads as ON the unit); using the small height instead cranked the
            # zoom until the 3:1-wide sign overflowed the frame, erasing all context -> looked like a floating板.
            _ce.meta["label_span"] = float(max(sw, sh_))
        # Mount the sign UNDER the leaf (swing -> it turns WITH the door) or the unit prim (open bin/deck),
        # reusing the leaf prim found above. Convert the world corners into that prim's LOCAL frame so the world
        # placement is exact + it rides the prim; a BEHAVIOR asset's collider comes only from its own
        # /collisions/ shells, so a child decal is never rebuilt into a phantom boundingCube.
        mount_path = None
        if _root and _root.IsValid():
            _mount_prim = _leaf_prim if _leaf_prim is not None else _root
            mount_path = str(_mount_prim.GetPath())
            _minv = (
                UsdGeom.XformCache(Usd.TimeCode.Default())
                .GetLocalToWorldTransform(_mount_prim)
                .GetInverse()
            )
            corners = [Gf.Vec3f(_minv.Transform(Gf.Vec3d(c))) for c in corners]
            nrm3 = Gf.Vec3f(
                _minv.TransformDir(
                    Gf.Vec3d(float(_sign_face[0]), float(_sign_face[1]), 0.0)
                )
            )
        path = (
            f"{mount_path}/Label_{_safe(reg_name)}"
            if mount_path
            else f"/World/Label_{_safe(reg_name)}"
        )
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(corners)
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([nrm3] * 4)
        mesh.SetNormalsInterpolation("faceVarying")
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        # LIT (Task-1 fridge-label material): the container sign receives the cabinet lighting/shadows so it reads
        # as PART OF the door, not a uniformly-glowing pasted-on panel.
        _bind_label_material(mesh.GetPrim(), path, png, lit=True)
        # A flush sticker must not cast a drop shadow onto the door behind it (that reads as a floating board);
        # Omniverse RTX honours primvars:doNotCastShadows on the mesh prim.
        mesh.GetPrim().CreateAttribute(
            "primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool
        ).Set(True)
        from pxr import UsdPhysics

        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)

    def _recolor_body(pp, color):
        from pxr import UsdShade

        root = stage.GetPrimAtPath(pp)
        if not root or not root.IsValid():
            return
        mtl = UsdShade.Material.Define(stage, pp + "/BodyMat")
        sh = UsdShade.Shader.Define(stage, pp + "/BodyMat/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        from pxr import Gf

        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(
                    mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )

    from pxr import Sdf  # noqa: E402 (used by the label/recolor closures above)

    def _reagent_decal(key, bottle_pp, approach_yaw):
        # Mount reagent_<key>.png as a SMALL flat textured quad on the bottle's FRONT face (toward
        # approach_yaw, the side the human stands on) at mid-height, sized to a fraction of the bottle's OWN
        # bbox. Authored in the graspable anchor's LOCAL (base_link) frame and parented UNDER the anchor
        # (/World/Obj_<key>) so it RIDES the bottle when grabbed/moved -- build_object later composes the
        # bottle geometry under a /m SUB-prim of this same anchor + poses the anchor per frame, so this child
        # tracks it. Idle bottles render upright (identity orient) => anchor-local == world, so the outward
        # face is placed along the WORLD approach direction. Collision OFF (cosmetic). Records the outward
        # face normal on the entity (meta["label_face"]) for produce_look_at's decal-facing gaze.
        # A REQUIRED decal cannot reach this skip -- validate_lookat_decals already raised -- so the only
        # reachable case is an item the spec says is identifiable WITHOUT reading, i.e. cosmetic.
        png = f"{_LABEL_DIR}/reagent_{_safe(key)}.png"
        if not os.path.exists(png):
            print(
                f"[chemlab][WARN] optional decal absent for {key!r} (needs_lookat=False), "
                f"rendering it unlabelled: {png}"
            )
            return
        from pxr import Gf, UsdPhysics

        try:
            _rs = read_spec(f"{ct.ITEMS[key]['cat']}/{ct.ITEMS[key]['model']}")
        except Exception:  # noqa: BLE001
            return
        bb = _rs.get("bbox_size") or [0.1, 0.1, 0.18]
        blo = _rs.get("base_link_offset") or [0.0, 0.0, 0.0]
        a = math.radians(float(approach_yaw))
        nrm = (
            -math.sin(a),
            math.cos(a),
        )  # arc-CENTRE outward normal: bottle-centre -> human (FRONT face)
        cxl, cyl, czl = (
            float(blo[0]),
            float(blo[1]),
            float(blo[2]),
        )  # bbox centre in the anchor-LOCAL frame
        # CURVED label: a cylinder-section strip WRAPPING the bottle like a real bottle label. Radius = the
        # bottle's OWN radius (a hair proud of the wall to avoid z-fight), spanning ~110deg centred on the FRONT
        # (approach) side, ~half the bottle's height, UV-mapped u across the arc / v up. Readable head-on AND
        # across a range of angles (a flat card went edge-on / looked like a floating card off-axis). Every size
        # is a fraction of the bottle's own bbox -- no magic metres.
        radius = 0.5 * min(float(bb[0]), float(bb[1])) + 0.03 * max(
            float(bb[0]), float(bb[1])
        )
        sh_ = 0.5 * float(bb[2])  # label height ~ half the bottle's own height
        c_ang = math.atan2(
            nrm[1], nrm[0]
        )  # arc-centre angle (identity orient -> local == world)
        half_arc = math.radians(55.0)  # 110deg total wrap
        n_seg = 16
        z_bot, z_top = czl - sh_ * 0.5, czl + sh_ * 0.5
        _ring, _ring_n = [], []
        for _i in range(n_seg + 1):
            _th = c_ang - half_arc + (2.0 * half_arc) * (_i / float(n_seg))
            _ox, _oy = math.cos(_th), math.sin(_th)
            _ring.append((cxl + radius * _ox, cyl + radius * _oy))
            _ring_n.append((_ox, _oy))
        corners = [Gf.Vec3f(float(x), float(y), float(z_bot)) for (x, y) in _ring] + [
            Gf.Vec3f(float(x), float(y), float(z_top)) for (x, y) in _ring
        ]
        counts, indices, sts, norms = [], [], [], []
        for _i in range(n_seg):
            _t0, _t1 = (n_seg + 1) + _i, (n_seg + 1) + _i + 1
            counts.append(4)
            indices += [
                _i,
                _i + 1,
                _t1,
                _t0,
            ]  # bottom-left, bottom-right, top-right, top-left (CCW from front)
            _u0, _u1 = _i / float(n_seg), (_i + 1) / float(n_seg)
            sts += [
                Gf.Vec2f(_u0, 0.0),
                Gf.Vec2f(_u1, 0.0),
                Gf.Vec2f(_u1, 1.0),
                Gf.Vec2f(_u0, 1.0),
            ]
            _n0 = Gf.Vec3f(float(_ring_n[_i][0]), float(_ring_n[_i][1]), 0.0)
            _n1 = Gf.Vec3f(float(_ring_n[_i + 1][0]), float(_ring_n[_i + 1][1]), 0.0)
            norms += [_n0, _n1, _n1, _n0]
        anchor = stage.DefinePrim(
            bottle_pp, "Xform"
        )  # build_object composes /m + a matrix xform onto this
        # Pose the anchor at the reagent's registered rest so the seed-init nav-grid contact (object_pos, read
        # BEFORE build_object exists) lands at the bench, not the world origin. build_object's MakeMatrixXform
        # + the render settle re-pose it later; the decal (child) rides whatever pose the anchor is given.
        _e = scene.entities.get(key)
        if _e is not None and _e.xy is not None:
            _z = float(_e.z) if _e.z is not None else float(scene.floor_z)
            fpv_scene.set_world_pose(anchor, (float(_e.xy[0]), float(_e.xy[1]), _z))
        path = f"{bottle_pp}/RLabel_{_safe(key)}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(corners)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateNormalsAttr(norms)
        mesh.SetNormalsInterpolation("faceVarying")
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set(sts)
        _bind_label_material(
            mesh.GetPrim(), path, png
        )  # OPAQUE + UNLIT so the label reads on CLEAR-GLASS bottles
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
        if _e is not None:
            _e.meta["label_face"] = (float(nrm[0]), float(nrm[1]))
        print(
            f"[chemlab] reagent decal {key}: face=({nrm[0]:.2f},{nrm[1]:.2f}) "
            f"curved arc=110deg r={radius:.3f} h={sh_:.3f} under {path}"
        )

    # ---- scene containers: fume hood (open deck), toxics (revolute cabinet), sink + trash (neg-only) ----
    scene.entities.pop(
        hood_nm, None
    )  # drop the SURFACE dup; re-register the hood as a deck CONTAINER
    _register_deck("fume_hood", hood_pp)
    _label_decal("fume_hood", hood_pp, ct.CONTAINERS["fume_hood"].get("sign"), False)
    reg_pp["fume_hood"] = hood_pp

    cab_cands = [
        (nm, pp) for nm, (pp, o) in placed.items() if o.get("cat") == "bottom_cabinet"
    ]
    cab_cands.sort(
        key=lambda t: math.hypot(
            _ctr(t[1])[0] - ROOM["chemistry_lab"][0],
            _ctr(t[1])[1] - ROOM["chemistry_lab"][1],
        )
    )
    if not cab_cands:
        raise RuntimeError("chemlab: no bottom_cabinet for the toxics cabinet")
    ca_nm, ca_pp = cab_cands[0]
    scene.entities.pop(ca_nm, None)
    if not _register_swing("toxics", ca_pp, placed[ca_nm][1]):
        print("[chemlab] toxics: no swing leaf -> registering top-open")
        _register_open(
            "toxics", ca_pp, placed[ca_nm][1].get("cat"), placed[ca_nm][1].get("model")
        )
    _label_decal("toxics", ca_pp, ct.CONTAINERS["toxics"].get("sign"), True)
    reg_pp["toxics"] = ca_pp
    print(f"[chemlab] toxics cabinet = {ca_nm} @ {_ctr(ca_pp)}")

    for _cnm, _cat, _room in (
        ("sink", "multi_station_furniture_sink", "bathroom"),
        ("trash_regular", "trash_can", "bathroom"),
    ):
        _cands = [(nm, pp) for nm, (pp, o) in placed.items() if o.get("cat") == _cat]
        _cands.sort(
            key=lambda t: math.hypot(
                _ctr(t[1])[0] - ROOM[_room][0], _ctr(t[1])[1] - ROOM[_room][1]
            )
        )
        if not _cands:
            raise RuntimeError(
                f"chemlab: no {_cat} for the '{_cnm}' negative-only container"
            )
        _nm2, _pp2 = _cands[0]
        _o2 = placed[_nm2][1]
        scene.entities.pop(_nm2, None)
        _register_open(_cnm, _pp2, _o2.get("cat"), _o2.get("model"))
        reg_pp[_cnm] = _pp2
        print(f"[chemlab] {_cnm} (negative-only) = {_nm2} @ {_ctr(_pp2)}")

    # ---- imported containers: hazard bins (flammables/oxidizers/acids/bases near benches) + sharps (corridor) ----
    def _grid_and_bounds():
        boxes = scene.solid_boxes()
        ext_mnx = min(float(mn[0]) for _n, mn, _mx in boxes)
        ext_mny = min(float(mn[1]) for _n, mn, _mx in boxes)
        ext_mxx = max(float(mx[0]) for _n, _mn, mx in boxes)
        ext_mxy = max(float(mx[1]) for _n, _mn, mx in boxes)

        def _cd(x, y):
            best = 1e9
            for _n, mn, mx in boxes:
                dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
                dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
                best = min(best, math.hypot(dx, dy))
            return best

        grid = [
            (float(x), float(y), _cd(float(x), float(y)))
            for x in np.arange(ext_mnx + 0.4, ext_mxx - 0.4, 0.25)
            for y in np.arange(ext_mny + 0.4, ext_mxy - 0.4, 0.25)
        ]
        return grid, boxes

    _wall_boxes = []
    for _wp in scene.wall_prims or []:
        try:
            _wall_boxes.append(_wbbox(_wp))
        except Exception:  # noqa: BLE001
            pass

    def _wall_dist(x, y):
        if not _wall_boxes:
            return 0.0
        b = 1e9
        for mn, mx in _wall_boxes:
            dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
            dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
            b = min(b, math.hypot(dx, dy))
        return b

    def _find_slot(grid, anchor, need, fh, dcap_max=1e9):
        wall_gap = fh + 0.15
        for cf, dcap0 in (
            (need, 3.0),
            (need, 5.0),
            (need * 0.85, 7.0),
            (need * 0.72, 10.0),
            (need * 0.6, 16.0),
        ):
            dcap = min(dcap0, dcap_max)
            best, bestscore = None, 1e18
            for x, y, c in grid:
                if c < cf:
                    continue
                d = math.hypot(x - anchor[0], y - anchor[1])
                if d > dcap:
                    continue
                if any(math.hypot(x - ux, y - uy) < ur for (ux, uy, ur) in used_slots):
                    continue
                wd = _wall_dist(x, y)
                if _wall_boxes and wd < fh + 0.08:
                    continue
                score = d + 1.3 * max(0.0, wd - wall_gap)
                if score < bestscore:
                    bestscore, best = score, (x, y)
            if best is not None:
                return best, cf
        return None, 0.0

    def _drop_to_floor(prim, pp, x, y, quat):
        fpv_scene.set_world_pose(
            prim, (float(x), float(y), float(scene.floor_z)), quat, (1.0, 1.0, 1.0)
        )
        (bx, by, base_z), _mx = _wbbox(pp)
        if abs(float(base_z) - float(scene.floor_z)) > 0.02:
            fpv_scene.set_world_pose(
                prim,
                (
                    float(x),
                    float(y),
                    float(scene.floor_z) - (float(base_z) - float(scene.floor_z)),
                ),
                quat,
                (1.0, 1.0, 1.0),
            )

    # ---- derive the lab<->corridor doorway GEOMETRICALLY (no hardcoded world coords) so the sharps bin
    # sits JUST INSIDE the corridor on the doorway's OPEN side, and the through-lane (doorway -> corridor
    # interior, the ONLY lab<->bathroom link) is kept clear. The doorway = the 'door' prim nearest the
    # segment from the lab anchor to the corridor anchor. Offsets are fractions of the DOOR's own width.
    def _seg_dist(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        _l2 = vx * vx + vy * vy
        t = (
            0.0
            if _l2 <= 1e-9
            else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / _l2))
        )
        return math.hypot(px - (ax + t * vx), py - (ay + t * vy))

    def _clr_furn(x, y):
        b = 1e9
        for _n, mn, mx in scene.solid_boxes():
            dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
            dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
            b = min(b, math.hypot(dx, dy))
        return b

    _lab_a, _cor_a = ROOM["chemistry_lab"], ROOM["corridor"]
    corridor_anchor = _cor_a  # fallback if no door prim is found
    _door_cands = [(nm, pp) for nm, (pp, o) in placed.items() if o.get("cat") == "door"]
    if _door_cands:
        _door_cands.sort(
            key=lambda t: _seg_dist(
                _ctr(t[1])[0], _ctr(t[1])[1], _lab_a[0], _lab_a[1], _cor_a[0], _cor_a[1]
            )
        )
        _dnm, _dpp = _door_cands[0]
        _dcx, _dcy = _ctr(_dpp)
        (_dmn, _dmx) = _wbbox(_dpp)
        _door_w = max(0.4, max(float(_dmx[0] - _dmn[0]), float(_dmx[1] - _dmn[1])))
        _dvec = (_cor_a[0] - _dcx, _cor_a[1] - _dcy)  # doorway -> corridor interior
        _dn = math.hypot(_dvec[0], _dvec[1]) or 1.0
        _dir = (_dvec[0] / _dn, _dvec[1] / _dn)
        _perp = (-_dir[1], _dir[0])
        _off = 0.6 * _door_w
        _pp_plus = (
            _dcx + _dir[0] * _off + _perp[0] * _off,
            _dcy + _dir[1] * _off + _perp[1] * _off,
        )
        _pp_minus = (
            _dcx + _dir[0] * _off - _perp[0] * _off,
            _dcy + _dir[1] * _off - _perp[1] * _off,
        )
        _side = 1.0 if _clr_furn(*_pp_plus) >= _clr_furn(*_pp_minus) else -1.0
        corridor_anchor = (
            _dcx + _dir[0] * _off + _perp[0] * _side * _off,
            _dcy + _dir[1] * _off + _perp[1] * _side * _off,
        )
        # keep-outs marching from the doorway INTO the corridor -> the bin never lands in the through-lane
        for _k in range(4):
            _kk = _k * (0.7 * _door_w)
            used_slots.append(
                (_dcx + _dir[0] * _kk, _dcy + _dir[1] * _kk, 0.7 * _door_w)
            )
        # ... and marching from the doorway INTO the LAB -> a lab hazard bin never lands crammed against the
        # doorway with its sign facing the corridor (the oxidizers bin did, so its look-at framed the corridor
        # lockers). Bins then settle in the lab interior (nearer the lab room anchor) with open lab floor for the
        # human to stand on + read the sign squarely. Reserve depth is a multiple of the door's OWN width.
        for _k in range(1, 5):
            _kk = _k * (0.7 * _door_w)
            used_slots.append(
                (_dcx - _dir[0] * _kk, _dcy - _dir[1] * _kk, 0.7 * _door_w)
            )
        print(
            f"[chemlab] lab<->corridor door {_dnm} @ ({_dcx:.2f},{_dcy:.2f}) w={_door_w:.2f} "
            f"dir=({_dir[0]:.2f},{_dir[1]:.2f}) side={_side:+.0f} -> sharps anchor "
            f"({corridor_anchor[0]:.2f},{corridor_anchor[1]:.2f})"
        )
    else:
        print(
            "[chemlab] no 'door' prim found -> sharps anchor falls back to the corridor room anchor"
        )

    for name, spec in ct.CONTAINERS.items():
        if spec["source"] != "import":
            continue
        cat, model = spec["cat"], spec["model"]
        room, kind = spec["room"], spec["kind"]
        # sharps bin: just inside the corridor on the doorway's open side (geometric, off the through-lane).
        anchor = corridor_anchor if room == "corridor" else ROOM.get(room, (cx0, cy0))
        bb = read_spec(f"{cat}/{model}").get("bbox_size") or [0.4, 0.4, 0.4]
        fh = 0.5 * max(float(bb[0]), float(bb[1]))
        need = fh + (
            0.50 if kind == "swing" else 0.28
        )  # a swing door needs out-swing + reach clearance
        grid, boxes = _grid_and_bounds()
        slot, got = _find_slot(grid, anchor, need, fh)
        if slot is None:
            raise RuntimeError(
                f"chemlab: no open floor for imported {name} near {room} {anchor}"
            )
        pp = f"/World/b_{_safe(name)}"
        prim = fpv_scene.add_reference(stage, pp, baked_usd(f"{cat}/{model}"))
        _drop_to_floor(prim, pp, slot[0], slot[1], (1.0, 0.0, 0.0, 0.0))
        if (
            kind == "swing"
        ):  # rotate so the baked door faces the open lab, then re-drop at that yaw
            quat = _orient_door_open(pp, cat, model, slot, boxes)
            _drop_to_floor(prim, pp, slot[0], slot[1], quat)
        placed[name] = (pp, {"cat": cat, "model": model})
        if spec.get("recolor_body"):
            _recolor_body(pp, spec.get("color") or (0.6, 0.63, 0.68))
        used_slots.append((slot[0], slot[1], fh + 0.85))
        # DOORED cabinet (kind=="swing"): register the revolute leaf + interior shelf (drop_z) via the swing
        # path, and mount the sign FLAT ON THE DOOR LEAF (tall=True -> leaf-parented, turns with the door). Fall
        # back to an open bin only if the asset has no swing leaf. Non-swing imports keep the open-bin path.
        _swing = kind == "swing" and _register_swing(
            name, pp, {"cat": cat, "model": model}
        )
        if not _swing:
            if kind == "swing":
                print(f"[chemlab] {name}: no swing leaf -> registering open bin")
            _register_open(name, pp, cat, model)
        if _swing:
            # Reserve CLEAR FLOOR in front of the door (along front_dir) so no LATER unit is dropped in this
            # cabinet's doorway -- the human needs ~1 m of open floor to stand SQUARE-ON and read the sign.
            # Without it, oxidizers faced -Y into then-empty floor and bases was placed 0.44 m in front of it,
            # forcing an oblique read that made the flat door sign look tilted / off-plane.
            _fdir = front_dir.get(name)
            if _fdir and (abs(_fdir[0]) + abs(_fdir[1]) > 1e-6):
                for _fd_d in (0.55, 1.05, 1.55):
                    used_slots.append(
                        (slot[0] + _fdir[0] * _fd_d, slot[1] + _fdir[1] * _fd_d, 0.6)
                    )
        _label_decal(name, pp, spec.get("sign"), _swing)
        reg_pp[name] = pp
        print(
            f"[chemlab] import {name} ({cat}/{model}, {room}, {'swing' if _swing else 'open'}) -> "
            f"({slot[0]:.2f},{slot[1]:.2f}) clearance>={got:.2f}m"
        )

    # ---- the MIS-STORED solvent: pre-place its graspable INSIDE toxics (on its reachable shelf) ------
    for k in ct.ITEMS:
        if ct.ITEMS[k].get("place") != "in:toxics":
            continue
        _cae = scene.resolve("toxics")
        _sxy = _cae.xy  # toxics cabinet's interior (front-inset) xy
        _sz = float(_cae.drop_z) if _cae.drop_z is not None else float(_cae.z)
        scene.register(
            k,
            Role.GRASPABLE,
            xy=(float(_sxy[0]), float(_sxy[1])),
            z=float(_sz),
            prim_path=f"/World/Obj_{_safe(k)}",
            approach_yaw=float(_cae.approach_yaw),
            foot_xy=(float(_sxy[0]), float(_sxy[1])),
            foot_half=(0.08, 0.08),
            top_z=float(_sz),
            clearance=0.34,
        )
        print(
            f"[chemlab] mis-stored {k} pre-placed INSIDE toxics @ "
            f"({_sxy[0]:.2f},{_sxy[1]:.2f},{_sz:.2f})"
        )

    # ---- REAGENT DECALS: mount each bottle's own GHS label (reagent_<key>.png) on its FRONT face (toward its
    # approach) so the human can READ it on [LookAt]; parented under the graspable's anchor so the label RIDES
    # the bottle when grabbed/moved. Covers every seated/registered reagent (bench + the mis-stored solvent);
    # the closure silently skips any item with no PNG (e.g. the non-reagent shard).
    for k in ct.ITEMS:
        _re = scene.entities.get(k)
        if _re is None:
            continue
        _reagent_decal(k, f"/World/Obj_{_safe(k)}", float(_re.approach_yaw))

    # ---- reference program: CHEMISTRY_LAB first (grouped by ref_target), then the CORRIDOR shard trip ---
    # Opt-in FAST LABEL TOUR: `CHEMLAB_LABEL_TOUR=1` -> author a minimal walk+[LookAt] pass (no grabs/places)
    # so the human can render JUST the label shots; unset -> the full segregation task (default, unchanged).
    _tour_env = os.environ.get("CHEMLAB_LABEL_TOUR", "")
    scene.label_tour = bool(_tour_env)
    # a comma-list of keys (not "1"/"all") -> tour ONLY those (fast iteration); "1"/"all"/other -> full tour.
    _tour_keys = [
        s.strip()
        for s in _tour_env.split(",")
        if s.strip() and s.strip() not in ("1", "all")
    ]
    goals, negs = ct.build_task_fields()
    # Only reachable BUILD-side gate for `neg_constraints`: at score time episode_runtime wraps the whole goal
    # block in a blanket except, which would turn an unimplemented kind into task_goal:null + a silent pass.
    rt.validate_neg_kinds(negs)

    def _bench_targets(cont):
        return [
            k
            for k in ct.ITEMS
            if ct.ITEMS[k]["ref_target"] == cont and ct.ITEMS[k].get("place") == "bench"
        ]

    if getattr(scene, "label_tour", False):
        # LABEL TOUR: render ONLY the look-at shots. Empty _tour_keys ("1"/"all") -> every hazard unit + every
        # reagent (bench + the in-toxics solvent), grouped; a comma-list -> ONLY those keys, IN THE REQUESTED
        # ORDER. An in:toxics key (the mis-stored solvent) needs the cabinet OPENED to be seen, so those keys
        # are deferred into an [Open]/[Close] wrapper at the END (a closed-door 'toxics' sign look-at stays
        # readable). NO grabs/places -> a fast pass to inspect label legibility.
        if _tour_keys:
            _ordered = list(_tour_keys)  # honour the caller's order
        else:
            _ordered = (
                list(ct.STORAGE_UNITS)
                + [k for k in ct.ITEMS if ct.ITEMS[k].get("place") == "bench"]
                + [k for k in ct.ITEMS if ct.ITEMS[k].get("place") == "in:toxics"]
            )
        _known = set(ct.CONTAINERS) | set(ct.ITEMS)
        _bad = [k for k in _ordered if k not in _known]
        if _bad:
            print(f"[chemlab] LABEL TOUR: skipping unknown key(s) {_bad}")
        _ordered = [k for k in _ordered if k in _known]
        _tox_keys = [
            k for k in _ordered if ct.ITEMS.get(k, {}).get("place") == "in:toxics"
        ]
        _normal_keys = [k for k in _ordered if k not in _tox_keys]
        segs: list = []
        for k in (
            _normal_keys
        ):  # closed-cabinet-safe keys first (a swing-door sign reads only when shut)
            segs.append(f"[Walk] <{k}> (1)")
            segs.append(f"[LookAt] <{k}> (1)")
        if _tox_keys:  # reveal the in-toxics solvent: open, look, close
            segs.append("[Open] <toxics> (1)")
            for k in _tox_keys:
                segs.append(f"[Walk] <{k}> (1)")
                segs.append(f"[LookAt] <{k}> (1)")
            segs.append("[Close] <toxics> (1)")
        program = ", ".join(segs)
        _mode = f"subset {_tour_keys}" if _tour_keys else "full"
        print(f"[chemlab] LABEL TOUR ({_mode}, {len(segs)} steps): {program}")
    else:
        segs = []
        # OPENING SURVEY: walk ONE smooth loop past every labeled hazard unit and LOOK at its door sign FIRST, so
        # the assistant (and the viewer) sees WHAT each cabinet is and WHERE before any sorting. Order the units by
        # angle around the lab centroid so the walk reads as a loop, not a zig-zag. Cabinets stay shut (the door
        # signs read when closed); reuses the same [Walk]+[LookAt] the label tour uses.
        _survey = [u for u in ct.HAZARD_UNITS if u in scene.entities]
        if _survey:
            _spos = {
                u: (getattr(scene.entities[u], "xy", None) or (0.0, 0.0))
                for u in _survey
            }
            _scx = sum(p[0] for p in _spos.values()) / len(_spos)
            _scy = sum(p[1] for p in _spos.values()) / len(_spos)
            _survey.sort(
                key=lambda u: math.atan2(_spos[u][1] - _scy, _spos[u][0] - _scx)
            )
            for _u in _survey:
                segs.append(f"[Walk] <{_u}> (1)")
                segs.append(f"[LookAt] <{_u}> (1)")
            print(f"[chemlab] opening survey loop: {[u for u in _survey]}")

        def _relocate(k, cont):
            # WALK up FIRST (LookAt gazes from where we stand + does NOT auto-navigate), then read the GHS
            # label CLOSE + facing it (the reagent decal faces the approach), grab, put. The Grab's own
            # auto-navigate to k is de-duped by the preceding [Walk], so this is one trip.
            segs.append(f"[Walk] <{k}> (1)")
            segs.append(f"[LookAt] <{k}> (1)")
            segs.append(f"[Grab] <{k}> (1)")
            segs.append(f"[PutInside] <{k}> (1) <{cont}> (1)")

        # volatiles -> the fume hood (open deck; NO open/close)
        for k in _bench_targets("fume_hood"):
            _relocate(k, "fume_hood")
        # toxics (scene revolute cabinet): OPEN, stow its toxic-class bench reagent(s), then CLOSE. The
        # mis-stored solvent that lives INSIDE toxics is diagnosed on [LookAt] and moved to its rule-correct
        # cabinet in that cabinet's own OPEN/CLOSE block below (toxics is re-opened there to reach it).
        segs.append("[Open] <toxics> (1)")
        for k in _bench_targets("toxics"):
            _relocate(k, "toxics")
        segs.append("[Close] <toxics> (1)")
        # hazard-class storage CABINETS (revolute doors, same model as toxics): OPEN -> PutInside each class
        # reagent on a reachable shelf -> CLOSE, per unit. A mis-stored reagent sitting in toxics but destined
        # here needs BOTH toxics (to grab it) and this cabinet (to place it) open during its move.
        for cont in ("flammables", "oxidizers", "acids", "bases"):
            _tgts = _bench_targets(cont)
            _instored = [
                k
                for k in ct.ITEMS
                if ct.ITEMS[k].get("place") == "in:toxics"
                and ct.ITEMS[k]["ref_target"] == cont
            ]
            if not _tgts and not _instored:
                continue
            segs.append(f"[Open] <{cont}> (1)")
            if (
                _instored
            ):  # move the mis-stored solvent out of toxics into its correct cabinet
                segs.append("[Open] <toxics> (1)")
                for k in _instored:
                    _relocate(k, cont)
                segs.append("[Close] <toxics> (1)")
            for k in _tgts:
                _relocate(k, cont)
            segs.append(f"[Close] <{cont}> (1)")
        # CORRIDOR: the broken-glass spill -> the sharps bin (crosses the lab->corridor door, auto-opens)
        for k in _bench_targets("sharps_bin"):
            _relocate(k, "sharps_bin")
        program = ", ".join(segs)

    scene.behavior_task = {
        "scene": scene_name,
        "task": "chemlab_segregation",
        "grasp": list(ct.ITEMS),
        "goals": goals,
        "neg_constraints": negs,
        "container": None,
    }

    # ---- flags + lights + auto-exposure (cloned from build_behavior_closing) --------------------------
    scene.plan_nav = True
    scene.soft_init = True
    scene.precise_nav = True
    scene.unified_geom = True
    scene.post_action_check = True
    scene.tpv_chase = True
    scene.auto_expose = True
    scene.fpv_headlamp = True
    scene.look_zoom = True  # FPV look-at adaptive zoom (produce_look_at/drive_cams) -> reads labels legibly
    scene.expose_z = 1.0
    _lab_ctr = (
        float(np.mean([_ctr(bpp)[0] for _n, bpp, _y, _b in bench_tops])),
        float(np.mean([_ctr(bpp)[1] for _n, bpp, _y, _b in bench_tops])),
    )
    scene.expose_xy = (float(_lab_ctr[0]), float(_lab_ctr[1]))
    for _tag, (wx0, wy0), wz0, _inten in (
        ("Lab", _lab_ctr, 2.6, 44000.0),
        ("Hood", _ctr(hood_pp), 2.6, 38000.0),
        ("Corridor", ROOM["corridor"], 2.4, 40000.0),
        ("CabA", _ctr(ca_pp), 2.4, 30000.0),
    ):
        lt = UsdLux.SphereLight.Define(stage, f"/World/ChemLight{_tag}")
        lt.CreateRadiusAttr(0.7)
        lt.CreateIntensityAttr(float(_inten))
        lt.CreateColorAttr((1.0, 1.0, 1.0))
        fpv_scene.set_world_pose(lt.GetPrim(), (float(wx0), float(wy0), float(wz0)))
    _dome = UsdLux.DomeLight.Define(stage, "/World/ChemDome")
    _dome.CreateIntensityAttr(650.0)
    _dome.CreateColorAttr((0.82, 0.86, 0.95))
    _register_mdl()

    tpv_look = (float(ROOM["chemistry_lab"][0]), float(ROOM["chemistry_lab"][1]), 1.0)
    tpv_eye = (
        float(ROOM["chemistry_lab"][0]) - 3.0,
        float(ROOM["chemistry_lab"][1]) - 3.0,
        3.0,
    )
    print(
        f"[chemlab] {scene_name}: {len(ct.ITEMS)} items ({len(bench_items)} on "
        f"{len(benches)} bench(es) + 1 mis-stored), {len(ct.CONTAINERS)} containers; "
        f"start=({scene.start_xy[0]:.2f},{scene.start_xy[1]:.2f})"
    )
    print(f"[chemlab] program: {program}")
    return scene, {}, program, tpv_look, tpv_eye

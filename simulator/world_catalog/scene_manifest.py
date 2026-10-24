#!/usr/bin/env python3
"""Offline BEHAVIOR-1K scene manifest -- the static half of the task-generation validator.

Answers, without booting Isaac and without rendering, "which scenes may training draw from, and
what is in them". Everything it needs is already on disk as data: the scene graph
(``scenes/<S>/json/<S>_best.json``) gives pos/ori/scale/in_rooms, and the bake cache
(``behavior-cache/spec/<cat>/<model>.json``) gives the canonical bbox + base-link offset +
openable joints. From those three the live stage's floor shift, every world AABB, and the
builder's own work-surface filter are all reconstructible in a couple of seconds.

Four gates, cheapest first (rejection is DATA, not an error -- the CLI still exits 0):

  V0 bake       every referenced (cat, model) has a baked USD over the 4 KB floor AND a spec.json
  V1 structure  >=1 baked ``floors`` prim (this is where ``dz`` comes from) and >=1 baked ``walls``
                prim (nav_grid's wall raster + the only occluder isaac_env models)
  V2 surface    >=1 work surface passing the BUILDER's own filter (behavior_import.py:827-844)
  V3 pair       >=1 work surface AND >=1 real container, anywhere in the scene

Co-location is deliberately NOT a veto anywhere in here: a strict "person and objects in the same
room" rule rejects 3 of the 4 scenes that train today, and an earlier room-scoped V3 also rejected
office_cubicles_right -- 16 usable containers and 8 work surfaces that simply never put one of each
in the same ANNOTATED room. Rooms are labels, not walls (9 scenes have rooms no floor prim covers).
It is reported as the ``colocation`` score and used by the builder as a placement preference.
V3 does insist on a real CONTAINER rather than "a second work surface", because
build_behavior_generic raises NoContainerFound without one -- admitting a container-less scene
would be the manifest blessing a scene the builder cannot build, which is the lie this file exists
to remove.

Pure stdlib on purpose -- the trainer venv has neither Isaac nor third-party packages, and
``load()`` / ``room_of()`` are imported from there. Importing this module never raises when the
json is absent; it returns an empty manifest.

It is also the home of the container family predicate (``container_kind`` /
``container_shelf_family``), shared by the physical-world importer so scene admission and
world construction cannot diverge.

Usage:
    python3 scene_manifest.py --all
    python3 scene_manifest.py --all --quiet --out /tmp/m.json
    python3 scene_manifest.py --scene Rs_int
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterable
from datetime import date

DATA_ROOT = os.path.join(
    os.environ.get(
        "INTERACTIVEWORLD_BEHAVIOR_DATA", "/opt/interactiveworld/behavior-data"
    ),
    "behavior-1k-assets",
)
SCENES_ROOT = os.path.join(DATA_ROOT, "scenes")
_CACHE_ROOT = os.environ.get(
    "INTERACTIVEWORLD_BEHAVIOR_CACHE", "/opt/interactiveworld/behavior-cache"
)
CACHE_USD = os.path.join(_CACHE_ROOT, "usd")
CACHE_SPEC = os.path.join(_CACHE_ROOT, "spec")

# A truncated bake writes a valid but empty USD. 0 of the 4058 baked models are under 4 KB today
# (walls: 12 KB .. 2.98 MB), so plain existence plus this floor is a sound completeness proxy.
USD_MIN_BYTES = 4096

# --- mirrored from the physical-world importer ------------------------------------------------
# Duplicated, not imported: behavior_import pulls in fpv_scene/pxr/scene_state and lives outside
# the physical-world layer, so importing it would drag Isaac into non-simulator environments. Keep in sync with the
# lines; the gates below are only as faithful as these three constants.
_SKIP_CATS = frozenset(  # behavior_import.py:56-70 -- never loaded into /World at all
    {
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
)
_SURF_EXCLUDE = (  # behavior_import.py:75-92 -- has a top, is not a work surface
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
# -> Role.SEAT instead of Role.SURFACE, behavior_import.py:822-825
_SEAT_KW = ("chair", "sofa", "couch", "bench", "stool", "armchair")
# registered as a swing OPENABLE and skipped, behavior_import.py:107 + :814-821
_DOOR_CATS = ("door",)
_SURF_TOP_MIN, _SURF_TOP_MAX = 0.62, 1.15  # behavior_import.py:834
_SURF_MIN_AREA = 0.06  # hx*hy, behavior_import.py:834
# ----------------------------------------------------------------------------------------------

_STRUCTURAL_CATS = frozenset(
    {"floors", "walls", "ceilings", "roof", "lawn", "driveway"}
)

# --- THE container family predicate -----------------------------------------------------------
# Single definition, shared: the physical-world importer's _container_candidates imports
# container_kind + container_shelf_family from here instead of keeping a private twin. The twins
# disagreed on 1222 of the 13516 spec-bearing scene instances -- locker x515 / bookcase x125 /
# shelf x116 / grocery_shelf x92 were builder-only, drawer-only bottom_cabinet x146 was
# manifest-only, and trash_can x18 came out 'open_top' here and 'openable' there -- so the scenes
# this manifest admitted were not the scenes the builder could build.
#
# Fixtures that own an interior or an articulated leaf but are never a place to put a task object:
# structural openings, plumbing (a wet volume, and the task reads as vandalism), fire, cooking, and
# light fixtures -- a table lamp's arm is metadata-identical to a lidded bin (both are one untagged
# revolute 'j_link_N') and a scaled-up one clears the cavity test. Substring-matched, same
# convention as _SURF_EXCLUDE, and checked against all 482 baked categories for collateral: 'door'
# catches only the 5 real door families (there is no 'outdoor_*'), 'light' catches no vessel.
# _SURF_EXCLUDE is deliberately NOT reused: a fridge / oven / dishwasher is a bad WORK SURFACE and
# an excellent INSIDE container.
_CONTAINER_EXCLUDE = (
    "door",
    "window",
    "laptop",
    "toilet",
    "urinal",
    "bidet",
    "sink",
    "bathtub",
    "shower",
    "fireplace",
    "stove",
    "range_hood",
    "lamp",
    "light",
)

# Rigid mouth-up vessels, matched on whole '_'-separated category TOKENS (substring matching would
# read 'bin' out of 'ca-bin-et'). Geometry alone cannot see a MISSING lid, so it cannot tell a bin
# from a solid block: 3 of the 4 containers the live training tasks use (trash_can, ice_bucket,
# toy_box) have neither a joint nor an interior board.
# 'pot' is deliberately ABSENT: it matches pot_plant, and a planter's cavity is full of soil and
# plant (13 false candidates in house_single_floor alone). No pot_plant model carries a revolute
# joint either, so the token set is the whole of that fix -- do not add 'pot'.
_OPEN_TOP_TOKENS = frozenset(
    {
        "basket",
        "bin",
        "box",
        "bucket",
        "can",
        "carton",
        "crate",
        "hamper",
        "pail",
        "sack",
        "tote",
        "tray",
        "tub",
        "wastebasket",
    }
)

# DOORLESS shelved families, same whole-token matching. A doored unit (fridge / cabinet / oven /
# wardrobe) is already found by its revolute leaf, but an open bookcase has neither a joint nor an
# open-top token and only a live interior-board walk (behavior_import._extract_spaces) can see it.
# Kept small on purpose: it exists to bound where that subtree walk runs, not to enumerate
# containers. Disjoint from _OPEN_TOP_TOKENS over all 482 baked categories, which is what lets
# container_shelf_family be a clean either/or.
_SHELF_TOKENS = frozenset(
    {"bookcase", "bookshelf", "cubby", "hutch", "rack", "shelf", "shelves", "shelving"}
)


# ---------------------------------------------------------------------------- library API (C-2)


def default_manifest_path() -> str:
    """The json this module writes and reads by default: ``scene_manifest.json`` beside it."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scene_manifest.json"
    )


def empty_manifest() -> dict:
    """A well-formed manifest with no scenes -- what ``load`` returns when the json is missing."""
    return {
        "generated": "",
        "asset_version": "",
        "n_scenes": 0,
        "admitted_scenes": [],
        "scenes": {},
    }


_CACHE: dict[str, dict] = {}


def load(path: str | None = None) -> dict:
    """Read the manifest json. Never raises: a missing or unreadable file yields an empty manifest.

    Results are memoised per path, so ``room_of`` in a hot loop costs one dict lookup.
    """
    p = path or default_manifest_path()
    if p in _CACHE:
        return _CACHE[p]
    try:
        with open(p) as f:
            m = json.load(f)
    except (OSError, ValueError):
        m = empty_manifest()
    _CACHE[p] = m
    return m


def admitted_scenes(manifest: dict | None = None) -> list[str]:
    """Scene names that passed every gate -- the pool training may draw from."""
    m = manifest if manifest is not None else load()
    return list(m.get("admitted_scenes") or [])


def scene_record(scene: str, manifest: dict | None = None) -> dict | None:
    """The per-scene record, or None when the scene is not in the manifest."""
    m = manifest if manifest is not None else load()
    return (m.get("scenes") or {}).get(scene)


def room_of(scene: str, x: float, y: float, manifest: dict | None = None) -> str | None:
    """Which annotated room contains world (x, y), or None outside every floor rectangle.

    SMALLEST-AREA containing rectangle wins. That tiebreak is load-bearing, not cosmetic: a
    garden slab encloses whole interior rooms, and rects overlap at every doorway. Measured over
    the 1404 objects that fall in more than one rect, smallest-area agrees with the dataset's own
    ``in_rooms`` annotation 92.5% of the time and largest-area 13.8%."""
    rec = scene_record(scene, manifest)
    if not rec:
        return None
    return _smallest_containing(rec.get("rooms") or {}, x, y)


# ------------------------------------------------------------------------------- the predicates


def is_work_surface(cat: str, top_z: float, hx: float, hy: float) -> bool:
    """The BUILDER's own comfortable-work-surface filter, reimplemented from
    ``behavior_import.py:827-844``: a Role.SURFACE (not a seat, not a bed, not closed storage /
    plumbing / an appliance) whose top sits in the reach window with a real horizontal footprint.

    ``top_z`` is post-``dz`` (floor top at 0) and ``hx``/``hy`` are the world-AABB half-extents,
    exactly what ``register_from_prim`` derives and ``surface_top`` returns."""
    if cat in _SKIP_CATS or cat in _STRUCTURAL_CATS or cat in _DOOR_CATS:
        return False
    if any(k in cat for k in _SEAT_KW):
        return False
    if cat == "bed" or any(k in cat for k in _SURF_EXCLUDE):
        return False
    return _SURF_TOP_MIN <= top_z <= _SURF_TOP_MAX and (hx * hy) >= _SURF_MIN_AREA


def container_kind(cat: str, spec: dict) -> str | None:
    """THE container FAMILY predicate -- ``"openable"``, ``"open_top"`` or None (not a container
    family at all). The offline sweep and ``behavior_import.build_behavior_generic`` both gate on
    THIS function, so the scenes the manifest admits are the scenes the builder can build.

    Inputs are the two things BOTH callers already hold: ``cat`` is the snake_case BEHAVIOR
    category and ``spec`` is that model's bake spec, the very same
    ``behavior-cache/spec/<cat>/<model>.json`` dict the sweep reads with ``read_spec`` and the
    builder reads with ``behavior_import.read_spec``. Pass ``{}`` when the spec is unreadable: the
    model then qualifies on its category alone. No stage, no geometry, no third-party import.

    ``openable``  the model ships a REVOLUTE joint -- a lid or a door.
    ``open_top``  a rigid mouth-up vessel (``_OPEN_TOP_TOKENS``) or a doorless shelved unit
                  (``_SHELF_TOKENS``): drop straight in, nothing to swing out of the way.

    THE JOINT RULE IS REVOLUTE, not the dataset's ``openable_joint_ids`` annotation, for two
    reasons. Coverage: only 72 of the 4058 baked models carry ``openable_joint_ids`` at all,
    against 225 with a revolute joint, and the 166 revolute-but-unannotated models include every
    locker, oven, carton, top_cabinet, display_fridge and (all 71 of) door -- an annotation absent
    from 71/71 doors is not a systematic one. Semantics: "openable" is a promise the builder has to
    keep, and ``behavior_import._static_open_container`` keeps it by HIDING the revolute joint's
    ``body1`` leaf; it is a silent no-op when there is no revolute joint. The 13 prismatic-only
    annotated models (drawer-only ``bottom_cabinet``) are therefore rejected on purpose -- nothing
    in the codebase can pull a drawer, so its interior is sealed and a drop into it is unachievable.

    This gate is FAMILY ONLY. It deliberately says nothing about whether a particular instance has
    a reachable cavity: that is ``behavior_import._container_interior`` over the real collision
    solid live, and ``container_cavity_ok`` over the world AABB offline. ``can_of_soda`` (66 mm)
    matches the 'can' token and is meant to be dropped by the cavity gate, not by this one."""
    if cat in _SKIP_CATS or cat in _STRUCTURAL_CATS:
        return None
    if any(k in cat for k in _CONTAINER_EXCLUDE):
        return None
    if any(str(j.get("type")) == "revolute" for j in (spec.get("joints") or ())):
        return "openable"
    words = set(cat.split("_"))
    if words & _OPEN_TOP_TOKENS or words & _SHELF_TOKENS:
        return "open_top"
    return None


def container_shelf_family(cat: str) -> bool:
    """True when this category's interior is BOARDS rather than a cavity, so a caller with a stage
    must confirm it with an interior-board walk before believing it.

    Only ``behavior_import._container_candidates`` can act on this (it owns the stage); the offline
    sweep records such a candidate unconfirmed, which is why the manifest's container list is a
    superset of the builder's on shelf families -- a ``towel_rack`` is a rack with no boards.

    A category that is ALSO an open-top vessel is NOT a shelf family: the vessel wins and needs no
    board walk. The two token sets are disjoint over all 482 baked categories today, so that clause
    only guards against a future edit."""
    words = set(cat.split("_"))
    return bool(words & _SHELF_TOKENS) and not (words & _OPEN_TOP_TOKENS)


def container_cavity_ok(hx: float, hy: float, z_lo: float, z_hi: float) -> bool:
    """OFFLINE PROXY for the live cavity gate: can this world AABB take a palette object?

    Mirrors ``behavior_import._container_interior``'s accept test with the floor top at z = 0 --
    the frame ``_compose_behavior_scene`` builds and this module reconstructs through ``dz``
    (``SceneState.floor_z`` is 0.0 for every BEHAVIOR scene). It is a PROXY and not the truth for
    one reason: the builder measures the per-link ``/collisions/*`` solid and this measures the
    visual bbox from ``bbox_size * scale``, which is the looser of the two.

    ``hx``/``hy`` are half-extents, ``z_lo``/``z_hi`` the AABB's floor-relative bottom and top.
    The ``max(hx, hy) <= 1.00`` cap is load-bearing and is not a size preference: a car and a 4 m
    lab bench are both articulated, so their AABB "interior" is a mid-air point under the body."""
    inner_z = z_lo + 0.06
    inner_h = (z_hi - 0.05) - inner_z
    if min(hx, hy) < 0.09 or max(hx, hy) > 1.00 or inner_h < 0.10:
        return False
    return not (z_hi < 0.15 or inner_z > 1.35)


# ------------------------------------------------------------------------------ offline geometry


def scene_json_path(scene: str) -> str:
    return os.path.join(SCENES_ROOT, scene, "json", f"{scene}_best.json")


def scene_names() -> list[str]:
    """Every scene shipping a ``<scene>_best.json``."""
    if not os.path.isdir(SCENES_ROOT):
        return []
    return sorted(
        d for d in os.listdir(SCENES_ROOT) if os.path.exists(scene_json_path(d))
    )


def asset_version() -> str:
    try:
        with open(os.path.join(DATA_ROOT, "VERSION")) as f:
            return f.read().strip()
    except OSError:
        return ""


def scene_instances(scene: str) -> list[dict]:
    """One dict per DatasetObject instance -- the stdlib twin of ``behavior_import.scene_objects``
    (same objects_info.init_info + state.registry.object_registry schema, same in_rooms
    normalisation, same skip of the embedded robot)."""
    with open(scene_json_path(scene)) as f:
        doc = json.load(f)
    init = doc["objects_info"]["init_info"]
    reg = doc["state"]["registry"]["object_registry"]
    out = []
    for name, info in init.items():
        if not isinstance(info, dict) or info.get("class_name") != "DatasetObject":
            continue
        args = info.get("args") or {}
        cat, model = args.get("category"), args.get("model")
        link = (reg.get(name) or {}).get("root_link") or {}
        pos, ori = link.get("pos"), link.get("ori")
        if not (cat and model and pos and ori):
            continue
        out.append(
            {
                "name": name,
                "cat": cat,
                "model": model,
                "pos": [float(v) for v in pos],
                "ori": [float(v) for v in ori],
                "scale": _scale3(args.get("scale")),
                "in_rooms": _in_rooms(args),
            }
        )
    return out


def _scale3(scale) -> list[float]:
    if scale is None:
        return [1.0, 1.0, 1.0]
    if isinstance(scale, (int, float)):
        return [float(scale)] * 3
    return [float(v) for v in scale]


def _in_rooms(args: dict) -> list[str]:
    """behavior_import.py:129 -- a few scenes store a bare string; never None."""
    rooms = args.get("in_rooms") or []
    if isinstance(rooms, str):
        rooms = [rooms]
    return [str(r) for r in rooms]


_SPEC_CACHE: dict[tuple[str, str], dict | None] = {}


def read_spec(cat: str, model: str) -> dict | None:
    """The bake spec (bbox_size / base_link_offset / openable_joint_ids / joints); None if unbaked."""
    key = (cat, model)
    if key not in _SPEC_CACHE:
        try:
            with open(os.path.join(CACHE_SPEC, cat, f"{model}.json")) as f:
                _SPEC_CACHE[key] = json.load(f)
        except (OSError, ValueError):
            _SPEC_CACHE[key] = None
    return _SPEC_CACHE[key]


def usd_baked(cat: str, model: str) -> bool:
    try:
        return (
            os.path.getsize(os.path.join(CACHE_USD, cat, f"{model}.usd"))
            >= USD_MIN_BYTES
        )
    except OSError:
        return False


def _quat_matrix(ori_xyzw: Iterable[float]) -> tuple[tuple[float, ...], ...]:
    x, y, z, w = (float(v) for v in ori_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def world_aabb(obj: dict, spec: dict) -> tuple[list[float], list[float]]:
    """(center, half_extent) of an instance's world AABB, exactly what ``BBoxCache
    .ComputeWorldBound(prim).ComputeAlignedRange()`` returns for the composed prim.

    The scene stores an object as (pos, quaternion, per-axis scale) over the model's CANONICAL
    bbox; the spec's ``bbox_size`` is that canonical extent at scale 1 and ``base_link_offset``
    is the bbox centre in the unscaled root-link frame. Scale multiplies in the LOCAL frame, then
    the OBB is rotated and re-bounded -- which is what swaps hx/hy on a yaw-turned table. Reading
    ``bbox_size`` without applying scale is the documented ``--dry`` canonical-scale trap and
    reports ~0.90 m for a table whose live top is 0.699 m."""
    bb = spec.get("bbox_size") or (0.1, 0.1, 0.1)
    blo = spec.get("base_link_offset") or (0.0, 0.0, 0.0)
    sc, pos = obj["scale"], obj["pos"]
    rot = _quat_matrix(obj["ori"])
    c_local = [float(blo[i]) * sc[i] for i in range(3)]
    h_local = [0.5 * abs(float(bb[i]) * sc[i]) for i in range(3)]
    center = [pos[i] + sum(rot[i][j] * c_local[j] for j in range(3)) for i in range(3)]
    half = [sum(abs(rot[i][j]) * h_local[j] for j in range(3)) for i in range(3)]
    return center, half


# ------------------------------------------------------------------------------- the sweep


def _bake_report(instances: list[dict]) -> tuple[int, list[str]]:
    """(unique models referenced, sorted 'cat/model' of those not usably baked)."""
    uniq = sorted({(o["cat"], o["model"]) for o in instances})
    missing = [
        f"{c}/{m}" for c, m in uniq if not usd_baked(c, m) or read_spec(c, m) is None
    ]
    return len(uniq), missing


def _floor_shift(boxes: list[tuple[dict, list[float], list[float]]]) -> float:
    """``dz`` = -(highest baked floor top). behavior_import.py:767-774 shifts z ONLY, so every
    XY here is already in the builder's frame and needs no correction."""
    tops = [c[2] + h[2] for o, c, h in boxes if o["cat"] == "floors"]
    return -max(tops) if tops else 0.0


def _room_rects(
    boxes: list[tuple[dict, list[float], list[float]]], annotated: set[str]
) -> dict[str, dict]:
    """Per-room floor rectangles from the ``floors`` prims. A room annotated on some object but
    covered by no floor prim is KEPT with an empty rect list -- 9 of 51 scenes have one, and
    silently dropping it would hide the reason its objects get charged to a neighbour.
    """
    rooms: dict[str, dict] = {r: {"rects": [], "area": 0.0} for r in sorted(annotated)}
    for o, c, h in boxes:
        if o["cat"] != "floors":
            continue
        rect = [c[0] - h[0], c[1] - h[1], c[0] + h[0], c[1] + h[1]]
        for r in o["in_rooms"]:
            rooms.setdefault(r, {"rects": [], "area": 0.0})
            rooms[r]["rects"].append([round(v, 4) for v in rect])
            rooms[r]["area"] += (2 * h[0]) * (2 * h[1])
    for r in rooms.values():
        r["area"] = round(r["area"], 3)
    return rooms


def _smallest_containing(rooms: dict[str, dict], x: float, y: float) -> str | None:
    """The room lookup itself -- ``room_of`` is this over a manifest record, and ``_classify``
    is this over the rooms it just built. One implementation so the two cannot drift."""
    best, best_area = None, 0.0
    for name, room in rooms.items():
        for x0, y0, x1, y1 in room.get("rects") or ():
            if x0 <= x <= x1 and y0 <= y <= y1:
                area = (x1 - x0) * (y1 - y0)
                if best is None or area < best_area:
                    best, best_area = name, area
    return best


def _classify(
    boxes: list[tuple[dict, list[float], list[float]]],
    dz: float,
    rooms: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Collect the scene's work surfaces and its container candidates.

    The two lists deliberately OVERLAP: a counter-height ``bottom_cabinet`` is a legitimate place
    to stage an object on and a legitimate place to put one in, and which role it plays is the
    builder's choice, not the manifest's. V3 counts distinct NAMES so one dual-role prim can
    never satisfy the pair on its own.

    Room = the dataset's own ``in_rooms`` annotation when present (it is ground truth), else
    ``room_of`` on the footprint centre, else null."""
    surfaces: list[dict] = []
    ranked: list[tuple[bool, float, dict]] = []
    for o, c, h in boxes:
        if o["cat"] in _SKIP_CATS:
            continue
        spec = read_spec(o["cat"], o["model"])
        if spec is None:
            continue
        top = c[2] + h[2] + dz
        hx, hy = max(h[0], 0.02), max(h[1], 0.02)
        room = (o["in_rooms"] or [None])[0] or _smallest_containing(rooms, c[0], c[1])
        if is_work_surface(o["cat"], top, hx, hy):
            surfaces.append(
                {
                    "name": o["name"],
                    "room": room,
                    "top_z": round(top, 4),
                    "area": round(hx * hy, 4),
                }
            )
        kind = container_kind(o["cat"], spec)
        if kind and container_cavity_ok(h[0], h[1], c[2] - h[2] + dz, top):
            rec = {
                "name": o["name"],
                "room": room,
                "kind": kind,
                # A shelf family is a container only if a live board walk finds boards, which needs
                # a stage. Flagged, not dropped: the sweep must not veto on what it cannot see.
                "unconfirmed": container_shelf_family(o["cat"]),
            }
            ranked.append((kind != "open_top", abs(top - 0.9), rec))
    # work_surfaces in the builder's own order (largest footprint first, behavior_import.py:845);
    # containers most-usable first: no [Open] needed, then rim nearest a comfortable 0.9 m.
    surfaces.sort(key=lambda s: -s["area"])
    ranked.sort(key=lambda t: (t[0], t[1], t[2]["name"]))
    return surfaces, [rec for _o, _d, rec in ranked]


def _door_inventory(instances: list[dict]) -> list[dict]:
    return sorted(
        (
            {
                "name": str(instance["name"]),
                "rooms": [str(room) for room in instance.get("in_rooms") or ()],
            }
            for instance in instances
            if str(instance.get("cat") or "") == "door"
        ),
        key=lambda row: row["name"],
    )


def _room_pair(surfaces: list[dict], containers: list[dict]) -> bool:
    """V3: the scene has a work surface AND a real container. Anywhere in the scene.

    Two deliberate departures from the obvious version, both learned the hard way:

    * **A container is required, a second work surface will not do.** ``build_behavior_generic`` raises
      ``NoContainerFound`` without one, so admitting a container-less scene means the manifest blesses a
      scene the builder cannot build -- the exact class of lie this file exists to remove. It costs
      ``hotel_suite_large`` and ``office_bike``, which have zero container candidates.
    * **Same-room is NOT required.** It is scored (``colocation``) and used as a placement preference by
      the builder, never as a veto. A strict same-room rule rejects 3 of the 4 scenes that train today,
      and it removed ``office_cubicles_right`` -- 16 usable containers and 8 work surfaces, which simply
      never puts one of each in the same ANNOTATED room (rooms are labels, not walls; 9 scenes have rooms
      no floor prim covers at all). A proxy with veto power already cost this project a third of its
      score once; the runtime settle guard therefore treats authored floor geometry as authoritative.
    """
    return bool(surfaces) and bool(containers)


def _colocation(surfaces: list[dict], containers: list[dict]) -> float:
    """SCORE in [0,1]: the best single room's share of (surface, container) pairs. Never a gate.

    Feeds task ranking and the builder's placement preference -- put the injected graspables and the
    container in the room the person starts in when the scene allows it, and say so in the record when
    it does not."""
    per_room: dict[str, list[int]] = {}
    for rec, idx in [(s, 0) for s in surfaces] + [(c, 1) for c in containers]:
        if rec["room"]:
            per_room.setdefault(rec["room"], [0, 0])[idx] += 1
    if not per_room:
        return 0.0
    best = max(min(s, c) for s, c in per_room.values())
    return round(best / max(1, min(len(surfaces), len(containers))), 3)


def _verdict(gates: dict[str, bool], n_missing: int, has_floors: bool) -> str:
    """Short human phrase naming the FIRST failed gate; '' when admitted."""
    if not gates["V0_bake"]:
        return f"V0 bake: {n_missing} model(s) not baked"
    if not gates["V1_structure"]:
        return "V1 structure: no baked " + ("walls" if has_floors else "floors")
    if not gates["V2_surface"]:
        return (
            f"V2 surface: no work surface in [{_SURF_TOP_MIN}, {_SURF_TOP_MAX}] m "
            f"with area >= {_SURF_MIN_AREA} m2"
        )
    if not gates["V3_room_pair"]:
        return "V3: the scene has no work surface + real container pair"
    return ""


def analyse_scene(scene: str) -> dict:
    """Run all four gates over one scene and return its manifest record (C-1 schema)."""
    instances = scene_instances(scene)
    n_models, missing = _bake_report(instances)
    boxes = []
    for o in instances:
        spec = read_spec(o["cat"], o["model"])
        if spec is None:
            continue
        c, h = world_aabb(o, spec)
        boxes.append((o, c, h))
    has_floors = any(o["cat"] == "floors" for o, _c, _h in boxes)
    has_walls = any(o["cat"] == "walls" for o, _c, _h in boxes)
    dz = _floor_shift(boxes)
    rooms = _room_rects(boxes, {r for o in instances for r in o["in_rooms"]})
    surfaces, containers = _classify(boxes, dz, rooms)
    doors = _door_inventory(instances)
    gates = {
        "V0_bake": not missing,
        "V1_structure": has_floors and has_walls,
        "V2_surface": bool(surfaces),
        "V3_room_pair": _room_pair(surfaces, containers),
    }
    admitted = all(gates.values())
    return {
        "bake_ok": not missing,
        "n_models": n_models,
        "n_missing": len(missing),
        "missing": missing,
        "has_walls": has_walls,
        "has_floors": has_floors,
        "dz": round(dz, 4),
        "rooms": rooms,
        "doors": doors,
        "work_surfaces": surfaces,
        "containers": containers,
        "gates": gates,
        # SCORE, never a gate -- the share of (surface, container) pairs the best single room holds.
        # Ranks tasks and steers the builder's placement; 0.0 does not disqualify anything.
        "colocation": _colocation(surfaces, containers),
        "admitted": admitted,
        "reason": _verdict(gates, len(missing), has_floors),
    }


def build_manifest(scenes: list[str]) -> dict:
    records = {s: analyse_scene(s) for s in scenes}
    return {
        "generated": date.today().isoformat(),
        "asset_version": asset_version(),
        "n_scenes": len(records),
        "admitted_scenes": [s for s in scenes if records[s]["admitted"]],
        "scenes": records,
    }


# ------------------------------------------------------------------------------------- CLI


def _format_scene(scene: str, rec: dict) -> str:
    flags = "".join(
        g[1] if rec["gates"][g] else "-"
        for g in ("V0_bake", "V1_structure", "V2_surface", "V3_room_pair")
    )
    head = (
        f"  {scene:38s} [{flags}] models={rec['n_models']:4d}/-{rec['n_missing']:<3d} "
        f"dz={rec['dz']:+.3f} rooms={len(rec['rooms']):3d} "
        f"surf={len(rec['work_surfaces']):4d} cont={len(rec['containers']):4d}"
    )
    return head if rec["admitted"] else f"{head}  REJECT {rec['reason']}"


def _format_summary(manifest: dict) -> str:
    recs = manifest["scenes"]
    lines = ["", "=" * 100, "GATES", "=" * 100]
    for gate in ("V0_bake", "V1_structure", "V2_surface", "V3_room_pair"):
        bad = sorted(s for s, r in recs.items() if not r["gates"][gate])
        lines.append(
            f"  {gate:14s} {len(recs) - len(bad):3d}/{len(recs):<3d} pass"
            + (f"   FAIL: {', '.join(bad)}" if bad else "")
        )
    n_adm = len(manifest["admitted_scenes"])
    lines.append("")
    lines.append(f"  ADMITTED       {n_adm}/{len(recs)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--all", action="store_true", help="sweep every scene on disk")
    ap.add_argument(
        "--scene", action="append", default=[], help="one scene (repeatable)"
    )
    ap.add_argument(
        "--out",
        default=None,
        help=f"write the manifest here (default {default_manifest_path()}; "
        "a --scene subset is only written when --out is explicit, so it cannot "
        "clobber the full sweep)",
    )
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args(argv)

    scenes = scene_names() if args.all else list(args.scene)
    if not scenes:
        ap.error("give --all or --scene <name>")
    unknown = [s for s in scenes if not os.path.exists(scene_json_path(s))]
    if unknown:
        print(f"no _best.json for: {', '.join(unknown)}", file=sys.stderr)
        return 2

    manifest = build_manifest(scenes)
    if not args.quiet:
        for scene in scenes:
            print(_format_scene(scene, manifest["scenes"][scene]))
    print(_format_summary(manifest))

    out = args.out or (default_manifest_path() if args.all else None)
    if out:
        with open(out, "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

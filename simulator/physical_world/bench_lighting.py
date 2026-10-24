"""ONE derived lighting rule for all 51 BEHAVIOR scenes -- no per-scene hand-authored intensities.

WHY THIS EXISTS
    Every builder today authors its own literal light table (``behavior_import.py`` L3250-3267 is the
    canonical one: 42000/44000/20000/20000/40000/34000/18000 W/sr at seven hand-picked room centres).
    That is tractable for six worlds and not for fifty-one, and an under-lit scene is not cosmetic
    here: the benchmark is scored on whether the assistant can READ a printed cue in the FPV frame.

THE RULE (v1, deliberately simple)
    1. POSITIONS ARE DERIVED, never authored. The scene's own ``ceilings`` slabs are its room
       partition -- one per enclosed room -- so their union is exactly the indoor floor. Lamps sit on
       ONE GLOBAL grid of pitch ``spacing`` over that union, which dedups the slab overlap that a
       per-slab grid double-counts (``hall_arch_wood`` ships 10532 m2 of ceiling over 4994 m2 of
       floor; per-slab gives 911 lamps, the global grid gives 103).
    2. ONE NOMINAL INTENSITY for every scene. ``NOMINAL`` is not a new guess: it is the rig that was
       MEASURED good on house_double_floor_lower/picking_up_trash (``_TASK_FILL_LIGHTS``), the one
       roofed scene anybody had tuned. When a huge hall forces a coarser grid (see ``MAX_LAMPS``) the
       intensity rises as spacing**2, so the areal luminous density -- the thing that actually sets
       floor illuminance -- stays the SAME number in every scene.
    3. HEIGHT IS CLAMPED, not copied. A lamp hung under a 20 m train-station ceiling delivers ~1/25
       of the illuminance of one under a 4 m ceiling, so "at the ceiling" is not a scene-independent
       height. ``LAMP_Z_MAX`` caps it; ``LAMP_Z_MIN`` keeps the emitter above head height, because an
       FPV eye that walks INTO a sphere light renders a full-white frame.
    4. THE EXISTING GLOBAL EXPOSURE METER does the last mile. Every lamp here is a plain ``/World``
       UsdLux prim, so ``episode_runtime._all_light_intensity_attrs`` finds it and the 12-viewpoint
       median-p92 solve scales the WHOLE rig by one factor. This module sets the SHAPE of the light
       field; the meter sets its LEVEL.
    5. A SMALL HAND BUDGET. ``HAND_FILL`` is the escape hatch for a scene that is measured to fail.
       It starts EMPTY on purpose: an entry may be added only with a probe measurement behind it.

WHAT THIS MODULE DOES NOT DO
    It does not touch the roof. Six scenes ship a ``roof`` prim (house_double_floor_lower,
    house_single_floor, Beechwood_0_garden, Merom_0_garden, Pomaria_0_garden, Rs_garden) and keep it;
    the other 45 have their ``ceilings`` dropped by ``behavior_import._SKIP_CATS`` and are flooded by
    the dome from above. The rig is applied to ALL 51 anyway: a dome pours straight DOWN, which leaves
    every vertical cabinet front -- where the printed cues live -- lit only at a grazing angle.
"""

from __future__ import annotations

import math

# The one nominal rig. Cloned from behavior_import._TASK_FILL_LIGHTS, the only fill rig in the
# codebase that was measured rather than guessed.
NOMINAL: dict[str, float] = {
    "spacing_m": 3.5,  # lamp pitch on the global grid
    "radius": 0.25,  # sphere radius; the emitter's own size
    "intensity": 60000.0,  # W/sr AT spacing_m -- scaled as spacing**2 when the grid is coarsened
    "below_ceiling_m": 0.20,  # hang this far under the slab it was derived from
}
LAMP_COLOR: tuple[float, float, float] = (1.0, 0.97, 0.92)

# One nominal dome for every scene: the sky an un-roofed interior sees once its ceilings are dropped.
DOME_INTENSITY: float = 400.0
DOME_COLOR: tuple[float, float, float] = (0.78, 0.84, 0.95)

# OUTDOOR KEY. Seven scenes ship ground that is outside the building (``lawn``/``driveway``): the four
# gardens, Wainscott_0_garden, and the two houses. The ceiling grid by construction never lights any of
# it -- there is no ceiling slab over a lawn -- and measurement showed the consequence: with the dome
# (400) and add_lighting's distant (1500) alone, Rs_garden's covered patio came out 89% black pixels and
# the four gardens all missed the visibility floor. This is the derived answer: a scene that ships
# outdoor ground gets a SUN, at one nominal intensity, tilted like add_lighting's distant. Scenes with
# no outdoor ground never see it, so 44 of 51 are untouched.
SUN_INTENSITY: float = 12000.0
SUN_COLOR: tuple[float, float, float] = (1.0, 0.97, 0.90)
SUN_ROTATE_XYZ: tuple[float, float, float] = (-50.0, 0.0, 25.0)
_OUTDOOR_CATS: frozenset[str] = frozenset({"lawn", "driveway", "pavement"})

# Emitter height band, in metres above the composed floor (which _compose_behavior_scene puts at z=0).
# MIN: the bottom of the sphere must clear a standing head, or an FPV eye walks into the emitter and
# the frame goes full white. MAX: inverse-square makes a cathedral-ceiling lamp useless at the floor,
# so the rig hangs at a task-relevant height instead of at the geometry.
LAMP_Z_MIN: float = 2.05
LAMP_Z_MAX: float = 4.20

# Ceiling on the lamp COUNT (RTX cost is roughly linear in lit lights). Hit only by the four halls and
# the two biggest offices; the grid is coarsened until it fits and the intensity compensates.
MAX_LAMPS: int = 120
_SPACING_GROWTH: float = 1.25
_SPACING_MAX: float = 20.0

# The emitter sphere is a light, not set dressing. Hiding it from the primary ray keeps a bare bulb out
# of the FPV frame AND keeps the exposure meter from metering its own lamps: a probe pose that happens
# to face one reads a blown p92 and dims the whole scene (the failure that crushed a restaurant to
# black when a single close-up was metered).
LAMP_VISIBLE_IN_PRIMARY_RAY: bool = False

# MEASURED failures only. scene_id -> tuple of (tag, (x, y), z, intensity) in COMPOSED world coords
# (floor at z=0). Empty is the correct state until lighting_probe.py proves a scene needs one.
HAND_FILL: dict[str, tuple[tuple[str, tuple[float, float], float, float], ...]] = {}


def _yawed_footprint(obj: dict) -> tuple[float, float]:
    """World-axis-aligned (w, h) of one slab: its baked bbox, scaled, then yawed.

    (0, 0) when the slab has no baked spec -- losing one room's lamps is the right cost of a partial
    bake, killing the build is not.
    """
    import behavior_import as bi

    try:
        bbox = bi.read_spec(f"{obj['cat']}/{obj['model']}").get("bbox_size")
    except (OSError, ValueError):
        return 0.0, 0.0
    if not bbox:
        return 0.0, 0.0
    scale = obj.get("scale", (1.0, 1.0, 1.0))
    w = float(bbox[0]) * float(scale[0])
    h = float(bbox[1]) * float(scale[1])
    qx, qy, qz, qw = (float(v) for v in obj["ori"])
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return w * c + h * s, w * s + h * c


def _grid_cells(
    ceilings: list[dict], spacing: float, dz: float
) -> list[tuple[float, float, float]]:
    """One lamp per GLOBAL grid bucket that any ceiling slab covers, at the LOWEST covering slab.

    The global bucket (rather than a per-slab grid) is what dedups overlapping slabs and what keeps the
    pitch uniform ACROSS a room boundary -- a per-slab grid crowds lamps at every internal wall. The
    lowest covering slab wins because that is the ceiling the room under it actually has.
    """
    cells: dict[tuple[int, int], tuple[float, float, float]] = {}
    for obj in ceilings:
        w, h = _yawed_footprint(obj)
        if w <= 0.0 or h <= 0.0:
            continue
        cx, cy, cz = (float(v) for v in obj["pos"])
        x0, y0 = cx - w * 0.5, cy - h * 0.5
        z = min(
            max(cz + dz - float(NOMINAL["below_ceiling_m"]), LAMP_Z_MIN), LAMP_Z_MAX
        )
        for i in range(math.floor(x0 / spacing), math.ceil((x0 + w) / spacing)):
            for j in range(math.floor(y0 / spacing), math.ceil((y0 + h) / spacing)):
                px, py = (i + 0.5) * spacing, (j + 0.5) * spacing
                if not (x0 <= px <= x0 + w and y0 <= py <= y0 + h):
                    continue
                prev = cells.get((i, j))
                if prev is None or z < prev[2]:
                    cells[(i, j)] = (px, py, z)
    return [cells[k] for k in sorted(cells)]


def plan_scene_lights(
    scene_name: str, dz: float = 0.0, objects: list | None = None
) -> dict:
    """The derived plan for one scene, computed WITHOUT Isaac (pure JSON + baked specs).

    Returned so the plan is auditable and dry-runnable across all 51 scenes in a second, instead of
    only being observable as a side effect on a live stage.
    """
    import behavior_import as bi

    objs = objects if objects is not None else bi.scene_objects(scene_name)
    ceilings = [o for o in objs if o["cat"] == "ceilings"]
    outdoor = any(o["cat"] in _OUTDOOR_CATS for o in objs)
    spacing = float(NOMINAL["spacing_m"])
    cells = _grid_cells(ceilings, spacing, dz)
    while len(cells) > MAX_LAMPS and spacing < _SPACING_MAX:
        spacing *= _SPACING_GROWTH
        cells = _grid_cells(ceilings, spacing, dz)
    # Constant areal luminous density: coarsening the grid by k dims the floor by k**2 unless the
    # lamps get k**2 brighter. This is what lets ONE nominal number cover a bedroom and a train station.
    intensity = (
        float(NOMINAL["intensity"]) * (spacing / float(NOMINAL["spacing_m"])) ** 2
    )
    return {
        "scene": scene_name,
        "n_ceiling_slabs": len(ceilings),
        "spacing_m": round(spacing, 3),
        "intensity": round(intensity, 1),
        "radius": float(NOMINAL["radius"]),
        "cells": cells,
        "n_cells": len(cells),
        "z_min": round(min((c[2] for c in cells), default=float("nan")), 3),
        "z_max": round(max((c[2] for c in cells), default=float("nan")), 3),
        "coarsened": spacing > float(NOMINAL["spacing_m"]) + 1e-9,
        "outdoor": outdoor,
    }


def _dz_from_placed(stage, placed: dict) -> float:
    """The floor shift ``_compose_behavior_scene`` applied, recovered from the stage.

    Compose re-poses each prim to ``pos + dz`` but leaves ``o["pos"]`` at the authored value, so the
    translation delta on ANY placed prim IS dz. Recovered rather than threaded through the caller: the
    builders discard it, and a lighting rig that needs a new argument at every call site is a rig that
    does not get adopted.
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim_path, obj in placed.values():
        prim = stage.GetPrimAtPath(prim_path)
        if not (prim and prim.IsValid()):
            continue
        world_z = float(cache.GetLocalToWorldTransform(prim).ExtractTranslation()[2])
        return world_z - float(obj["pos"][2])
    return 0.0


# The building SHELL is not an obstacle a lamp can be "inside" of. Its AABB is the whole storey: a
# roof slab spans every room from ~2.2 m up, so testing a 2.35 m lamp against it rejects EVERY cell.
# Measured: this vetoed all 3 of Rs_garden's lamps and left the scene lit by the dome alone.
_SHELL_CATS: frozenset[str] = frozenset(
    {"roof", "ceilings", "floors", "walls", "lawn", "driveway", "pavement", "fence"}
)
# Backstop for a shell category we have not named: nothing a lamp can hide inside has a 25 m2 footprint.
_MAX_OBSTACLE_AREA_M2: float = 25.0


def _occupied_boxes(stage, placed: dict) -> list[tuple]:
    """World AABBs of the FURNITURE compose placed, so a lamp is never authored inside solid geometry.

    Furniture only -- see ``_SHELL_CATS``.
    """
    import fpv_scene

    boxes = []
    for prim_path, obj in placed.values():
        if obj.get("cat") in _SHELL_CATS:
            continue
        prim = stage.GetPrimAtPath(prim_path)
        if not (prim and prim.IsValid()):
            continue
        rng = fpv_scene.compute_world_bbox(prim)
        mn, mx = rng.GetMin(), rng.GetMax()
        if (mx[0] - mn[0]) * (mx[1] - mn[1]) > _MAX_OBSTACLE_AREA_M2:
            continue
        boxes.append((mn, mx))
    return boxes


def _author_lamp(
    stage, path: str, xyz: tuple[float, float, float], radius: float, intensity: float
) -> None:
    import fpv_scene
    from pxr import Sdf, UsdLux

    lamp = UsdLux.SphereLight.Define(stage, path)
    lamp.CreateRadiusAttr(float(radius))
    lamp.CreateIntensityAttr(float(intensity))
    lamp.CreateColorAttr(LAMP_COLOR)
    prim = lamp.GetPrim()
    prim.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(
        bool(LAMP_VISIBLE_IN_PRIMARY_RAY)
    )
    fpv_scene.set_world_pose(prim, (float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _author_sun(stage) -> None:
    """The outdoor key for a scene that ships ground outside the building. See ``SUN_INTENSITY``."""
    import fpv_core
    from pxr import Gf, UsdLux

    sun = UsdLux.DistantLight.Define(stage, "/World/BenchSun")
    sun.CreateIntensityAttr(float(SUN_INTENSITY))
    sun.CreateColorAttr(SUN_COLOR)
    sun.CreateAngleAttr(0.8)
    fpv_core._set_xform_rotate_xyz(sun.GetPrim(), Gf.Vec3f(*SUN_ROTATE_XYZ))


def rig_scene_lights(
    stage,
    scene_name: str,
    placed: dict,
    *,
    dz: float | None = None,
    objects: list | None = None,
    dome_intensity: float | None = None,
) -> dict:
    """Author the whole lighting rig for ``scene_name`` and return the manifest record.

    ONE call replaces a builder's hand-authored light table. ``placed`` is the dict
    ``_compose_behavior_scene`` returns; ``dz`` and ``objects`` are recovered from it and from the
    scene JSON when not supplied, so the only thing a caller must already have is the composed scene.

    Every prim it authors lives at ``/World/BenchFill*`` / ``/World/BenchDome`` -- plain UsdLux, so the
    collider pass skips it (lights are type-skipped) and ``_all_light_intensity_attrs`` scales it.
    """
    from pxr import UsdLux

    shift = _dz_from_placed(stage, placed) if dz is None else float(dz)
    plan = plan_scene_lights(scene_name, shift, objects)
    boxes = _occupied_boxes(stage, placed)
    n_inside = 0
    n = 0
    for x, y, z in plan["cells"]:
        # A lamp inside a wall cabinet or a wall cavity emits into solid geometry and lights nothing;
        # the neighbouring cells already cover the floor beneath it.
        if any(
            mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]
            for mn, mx in boxes
        ):
            n_inside += 1
            continue
        _author_lamp(
            stage,
            f"/World/BenchFill{n:03d}",
            (x, y, z),
            plan["radius"],
            plan["intensity"],
        )
        n += 1

    hand = HAND_FILL.get(scene_name, ())
    for tag, (hx, hy), hz, hi in hand:
        _author_lamp(
            stage, f"/World/BenchHandFill_{tag}", (hx, hy, hz), plan["radius"], hi
        )

    dome = UsdLux.DomeLight.Define(stage, "/World/BenchDome")
    dome.CreateIntensityAttr(
        float(DOME_INTENSITY if dome_intensity is None else dome_intensity)
    )
    dome.CreateColorAttr(DOME_COLOR)
    if plan["outdoor"]:
        _author_sun(stage)

    rec = {
        "rule": "derived_ceiling_grid_v1",
        "scene": scene_name,
        "dz": round(shift, 4),
        "spacing_m": plan["spacing_m"],
        "intensity": plan["intensity"],
        "radius": plan["radius"],
        "coarsened": plan["coarsened"],
        "n_cells": plan["n_cells"],
        "n_lamps": n,
        "n_dropped_inside_geometry": n_inside,
        "n_hand_fill": len(hand),
        "z_min": plan["z_min"],
        "z_max": plan["z_max"],
        "dome_intensity": float(
            DOME_INTENSITY if dome_intensity is None else dome_intensity
        ),
        "sun_intensity": float(SUN_INTENSITY) if plan["outdoor"] else 0.0,
    }
    print(
        f"[bench-light] {scene_name}: {n} lamp(s) @ {plan['intensity']:.0f} W/sr r={plan['radius']:.2f} "
        f"pitch={plan['spacing_m']:.2f}m z=[{plan['z_min']:.2f},{plan['z_max']:.2f}] "
        f"({n_inside} cell(s) dropped inside geometry, {len(hand)} hand fill) + dome "
        f"{rec['dome_intensity']:.0f}"
        + (f" + sun {SUN_INTENSITY:.0f}" if plan["outdoor"] else "")
    )
    return rec

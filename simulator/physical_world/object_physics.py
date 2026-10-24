# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reusable object-physics layer (CDS): settle arbitrary USD prim sets to a PHYSICS-DERIVED rest.

This is the general, additive core that makes the FPV benchmark's OBJECTS obey physics without
touching the kinematic-during-render human path. The proven invariant (from ``render_runtime`` /
``humanoid_motion``): PhysX runs ONCE at env-init as a ``render=False`` side-channel settle on a
``SimulationContext`` at 1/60, then every free body is flipped back to KINEMATIC before the render
loop, so only an object's REST pose is now physics-derived instead of a hardcoded literal.

It owns NOTHING per-object or per-scene. Every collider is auto-classified from geometry and every
rest pose comes from PhysX, so it generalizes over arbitrary USD prim path lists (e.g. BEHAVIOR-1K
open-vocabulary assets). It reuses ``physics_human`` / ``smplh_anim_core`` / ``fpv_scene`` VERBATIM and
only adds the multi-body settle + geometry collider classifier + all-pairs separation gate.

  * ``classify_collider(prim)``            -- geometry -> PhysX approximation ('convexHull' /
                                              'convexDecomposition'); the generalization of
                                              ``make_dynamic_rigid``'s hardcoded 'convexHull'.
  * ``get_sim()``                          -- the shared render=False SimulationContext (1/60).
  * ``read_prim_xyz(stage, path)``         -- world-space origin of any prim (XformCache).
  * ``settle_bodies(...)``                 -- step PhysX until all bodies rest; read each pose back.
  * ``settle_object_on_support(...)``      -- the general single-object drop+settle+reject (the
                                              generalization of ``_presettle_cup_on_desk``).
  * ``settle_clutter(...)``                -- the multi-body drop+settle+readback+per-object reject.
  * ``all_pairs_separation(...)``          -- all-pairs penetration gate (contact reports).

Design (matching ``physics_human.py``): ``from __future__ import annotations`` + LAZY pxr/isaacsim
imports inside every function, so importing this module before ``launch_app()`` never pulls in Kit.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd  # noqa: F401

LOG = logging.getLogger("object_physics")
_MAX_REASONABLE_WORLD_COORD_M = 1e6


def _valid_world_values(values) -> bool:
    return all(
        math.isfinite(float(value))
        and abs(float(value)) <= _MAX_REASONABLE_WORLD_COORD_M
        for value in values
    )


@dataclass(frozen=True)
class SupportSettleResult:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    moved_mm: float
    base_z: float
    expected_support_z: float
    actual_support_z: float
    xy_roll_m: float
    tilt_deg: float
    attempts: int
    selected_attempt: int
    support_verified: bool
    support_path: str | None
    support_fraction: float | None
    ok: bool
    reason: str


@dataclass(frozen=True)
class SupportProbe:
    verified: bool
    ok: bool
    support_z: float
    support_path: str | None = None
    support_fraction: float | None = None
    reason: str = "not-probed"


class InitialSettleError(RuntimeError):
    """An object could not reach a stable rest on its declared initial support."""

    code = "INITIAL_SETTLE_UNSTABLE"

    def __init__(self, key: str, prim_path: str, result: SupportSettleResult) -> None:
        super().__init__(
            f"{key} failed initial settle on its declared support: {result.reason} "
            f"(base={result.base_z:.3f}, expected_support={result.expected_support_z:.3f}, "
            f"actual_support={result.actual_support_z:.3f}, "
            f"xy_roll={result.xy_roll_m * 1000.0:.0f}mm, "
            f"tilt={result.tilt_deg:.1f}deg, attempts={result.attempts}, "
            f"selected_attempt={result.selected_attempt})"
        )
        self.context = {
            "entity": key,
            "prim_path": prim_path,
            "position": list(result.position),
            "orientation": list(result.orientation),
            "base_z": result.base_z,
            "expected_support_z": result.expected_support_z,
            "actual_support_z": result.actual_support_z,
            "xy_roll_m": result.xy_roll_m,
            "tilt_deg": result.tilt_deg,
            "attempts": result.attempts,
            "selected_attempt": result.selected_attempt,
            "support_verified": result.support_verified,
            "support_path": result.support_path,
            "support_fraction": result.support_fraction,
            "reason": result.reason,
        }


class InitialRelationError(RuntimeError):
    """Physical state no longer satisfies a declared initial object relation."""

    code = "INITIAL_RELATION_DIVERGENCE"

    def __init__(self, missing: set[tuple[object, str, str]]) -> None:
        rows = [
            {
                "predicate": getattr(predicate, "value", str(predicate)),
                "subject": subject,
                "target": target,
            }
            for predicate, subject, target in sorted(
                missing,
                key=lambda row: (
                    str(getattr(row[0], "value", row[0])),
                    row[1],
                    row[2],
                ),
            )
        ]
        super().__init__(f"physical initial relations are missing: {rows}")
        self.context = {"missing_relations": rows}


def require_initial_relations(
    expected: set[tuple[object, str, str]],
    actual: set[tuple[object, str, str]],
) -> None:
    missing = expected - actual
    if missing:
        raise InitialRelationError(missing)


def _aabb_overlap_depth(
    ax: float,
    ay: float,
    ahx: float,
    ahy: float,
    b: tuple[float, float, float, float],
    pad: float,
) -> float:
    """Signed AABB overlap of footprint A (expanded by ``pad`` on each side) vs footprint B (x,y,hx,hy).
    Two AABBs overlap iff they overlap on BOTH axes. Returns > 0 = penetration depth (overlapping both
    axes -> the smaller-axis penetration), <= 0 = separated (the LESS clearance == closer, so a
    candidate deeper into free space ranks better in the fallback)."""
    bx, by, bhx, bhy = b
    gx = abs(ax - bx) - (ahx + pad + bhx)  # <0 => overlap along X
    gy = abs(ay - by) - (ahy + pad + bhy)  # <0 => overlap along Y
    if gx >= 0.0 or gy >= 0.0:
        return -max(
            gx, gy
        )  # separated -> <= 0 (more clearance == more negative == better)
    return min(-gx, -gy)  # overlapping on both axes -> penetration depth (> 0)


def _sample_footprint(
    rng,
    region: tuple[float, float, float, float],
    hx: float,
    hy: float,
    obstacles: list[tuple[float, float, float, float]],
    min_gap: float,
    max_tries: int,
) -> tuple[tuple[float, float], int]:
    """Rejection-sample ONE non-overlapping (x,y) for a (hx,hy) footprint within ``region`` (inset by the
    half-extent so the footprint stays inside), avoiding every ``obstacles`` footprint by ``min_gap``.
    Returns ((x,y), rejected) -- the accepted position (or the least-overlapping fallback after
    ``max_tries``) and how many draws were rejected."""
    xmin, xmax, ymin, ymax = region
    lo_x, hi_x = xmin + hx, xmax - hx
    lo_y, hi_y = ymin + hy, ymax - hy
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    if (
        hi_x < lo_x
    ):  # footprint wider than the region on X -> clamp to center (degenerate)
        lo_x = hi_x = cx
    if hi_y < lo_y:
        lo_y = hi_y = cy

    best_xy: tuple[float, float] = (cx, cy)
    best_depth = float("inf")
    rejected = 0
    for _ in range(max(1, int(max_tries))):
        x = float(rng.uniform(lo_x, hi_x)) if hi_x > lo_x else lo_x
        y = float(rng.uniform(lo_y, hi_y)) if hi_y > lo_y else lo_y
        worst = 0.0
        for b in obstacles:
            worst = max(worst, _aabb_overlap_depth(x, y, hx, hy, b, min_gap))
        if worst <= 0.0:  # a fully-separated draw: accept immediately
            return (x, y), rejected
        rejected += 1
        if worst < best_depth:  # remember the least-overlapping fallback
            best_xy, best_depth = (x, y), worst
    return best_xy, rejected


def _min_pairwise_gap(placed: list[tuple[float, float, float, float]]) -> float:
    """Smallest signed BARE-AABB gap over every placed-vs-placed pair (>=0 = min free space between the
    closest pair; <0 = a penetrating pair). ``inf`` when <2 footprints (no pair -> trivially separated)."""
    gap = float("inf")
    for i in range(len(placed)):
        ax, ay, ahx, ahy = placed[i]
        for j in range(i + 1, len(placed)):
            bx, by, bhx, bhy = placed[j]
            gx = abs(ax - bx) - (ahx + bhx)
            gy = abs(ay - by) - (ahy + bhy)
            gap = min(
                gap, max(gx, gy)
            )  # AABBs overlap iff BOTH axes overlap -> the LARGER gap decides
    return gap


def plan_spawn_layout(
    region_bounds: tuple[float, float, float, float],
    half_extents: list[tuple[float, float]],
    seed: int,
    min_gap: float = 0.02,
    max_tries: int = 200,
    reserved: list[tuple[float, float, float, float]] | None = None,
) -> tuple[list[tuple[float, float]], dict]:
    """GENERAL seeded rejection-sampler for NON-OVERLAPPING spawn XY (BEHAVIOR-1K-ready, zero per-object magic).

    Random scene generation can spawn objects OVERLAPPING/inside each other, so the init starts
    physically invalid (>=1 penetrating frame; PhysX only de-penetrates gradually). This guarantees a
    non-overlapping initial layout by rejection-sampling each object's XY.

    ``region_bounds`` = ``(xmin, xmax, ymin, ymax)`` (the tabletop footprint, already margin-shrunk by
    the caller). ``half_extents`` = per-object ``(hx, hy)`` XY footprint half-sizes (from each prim's
    world bbox). ``reserved`` = optional pre-occupied ``(x, y, hx, hy)`` footprints to avoid (e.g. the
    carried mug's grab/place spot). ``seed`` seeds ``np.random.default_rng`` for reproducibility.

    For each object (via ``_sample_footprint``): sample ``(x, y)`` uniformly within the region INSET by
    its ``(hx, hy)`` so its footprint stays inside; REJECT + resample if its AABB (each side expanded by
    ``min_gap``) overlaps any already-placed or reserved footprint; accept the first valid draw, or --
    after ``max_tries`` -- fall back to the least-overlapping candidate (so a caller always gets N
    positions, even in an over-packed region). Pure geometry, reusable for arbitrary asset sets.

    Returns ``(positions, stats)`` where ``positions`` is a list of ``(x, y)`` aligned to
    ``half_extents`` and ``stats`` = ``{placed, rejected, attempts, min_pairwise_gap}``
    (``min_pairwise_gap`` = the smallest AABB gap over every placed-vs-placed pair; >= 0 means no
    overlap, < 0 means the fallback had to accept a penetrating pair -- a signed metric, in meters).
    """
    rng = np.random.default_rng(int(seed))
    region = tuple(float(v) for v in region_bounds)
    reserved = list(reserved or [])

    placed: list[
        tuple[float, float, float, float]
    ] = []  # (x, y, hx, hy) accepted footprints
    positions: list[tuple[float, float]] = []
    total_rejected = 0
    for hx, hy in half_extents:
        hx, hy = float(hx), float(hy)
        (x, y), rej = _sample_footprint(
            rng, region, hx, hy, placed + reserved, min_gap, max_tries
        )
        total_rejected += rej
        placed.append((x, y, hx, hy))
        positions.append((x, y))

    # min_pairwise_gap over placed-vs-placed pairs only (the physically-meaningful spawn separation;
    # reserved footprints are external constraints, excluded here).
    min_gap_seen = _min_pairwise_gap(placed)
    stats = {
        "placed": len(placed),
        "rejected": total_rejected,
        "attempts": total_rejected
        + len(placed),  # rejected draws + one accepted draw per object
        "min_pairwise_gap": min_gap_seen,
    }
    LOG.info(
        "plan_spawn_layout: placed=%d rejected=%d attempts=%d min_pairwise_gap=%s "
        "(region x[%.2f,%.2f] y[%.2f,%.2f], %d reserved, min_gap=%.3f, seed=%d)",
        stats["placed"],
        stats["rejected"],
        stats["attempts"],
        ("inf" if not np.isfinite(min_gap_seen) else f"{min_gap_seen * 1000:.0f}mm"),
        region[0],
        region[1],
        region[2],
        region[3],
        len(reserved),
        min_gap,
        int(seed),
    )
    return positions, stats


def classify_collider(prim: "Usd.Prim") -> str:
    """Pick a PhysX collision approximation for ``prim`` purely from its GEOMETRY (no per-object magic).

    Rule (geometry only): a bare Gprim (Cube/Sphere/Capsule/Cylinder/Cone with no child Mesh) ->
    'convexHull' (``make_dynamic_rigid``'s gprim fallback tags it via CollisionAPI only and ignores
    the approximation anyway, so any string is safe); a single convex-ish mesh part -> 'convexHull';
    concave / multi-part meshes (the DEFAULT) -> 'convexDecomposition' (a hollow bowl / a mug handle /
    a banana need the decomposition or their concavity is lost). The returned string is passed
    straight to ``make_dynamic_rigid(approximation=...)``; callers passing nothing keep its default.
    """
    from pxr import Usd

    tname = prim.GetTypeName()
    if tname in ("Cube", "Sphere", "Capsule", "Cylinder", "Cone"):
        return "convexHull"
    meshes = [p for p in Usd.PrimRange(prim) if p.GetTypeName() == "Mesh"]
    if len(meshes) <= 1:
        return "convexHull"
    return "convexDecomposition"


def get_sim() -> object:
    """The shared render=False SimulationContext (1/60, numpy backend), created-or-reused.

    Verbatim lift of ``render_runtime._settle_physics``'s context setup: ``SimulationContext``'s are
    singletons, so ``.instance()`` returns the one the demo already made (or we create it), then
    ``initialize_physics()`` binds PhysX. This is the render=False SIDE CHANNEL -- it coexists with the
    RTX render stepping exactly as the proven office/walk-pick-place paths prove.
    """
    from isaacsim.core.api.simulation_context import SimulationContext

    sim = SimulationContext.instance()
    if sim is None:
        sim = SimulationContext(
            physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, backend="numpy"
        )
    sim.initialize_physics()
    return sim


def read_prim_xyz(stage: "Usd.Stage", prim_path: str) -> np.ndarray:
    """World-space translation of ``prim_path``'s origin (path-parametrized ``_read_cup_pos``).

    USD is row-vector, so the translation is the LAST ROW of the local-to-world matrix.
    """
    from pxr import Usd, UsdGeom

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    xc.Clear()
    m = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path))).reshape(
        4, 4
    )
    return m[3, :3].astype(np.float64)


def read_prim_transform(stage: "Usd.Stage", prim_path: str) -> np.ndarray:
    """Column-vector 4x4 world transform of ``prim_path`` at the current physics pose."""
    from pxr import Usd, UsdGeom

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    xc.Clear()
    matrix = np.array(
        xc.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path)),
        dtype=np.float64,
    ).reshape(4, 4)
    return matrix.T.copy()


def _read_pose(xc, prim) -> tuple:
    """(translate xyz, orient wxyz) of a prim from an XformCache matrix (USD row-vector: translate is
    the last row; the rotation quaternion is extracted from the orthonormalized 3x3). Used to RECORD a
    full settle trajectory (translate + orientation) so the drop can be replayed kinematically."""
    m = xc.GetLocalToWorldTransform(prim)
    t = m.ExtractTranslation()
    q = m.ExtractRotation().GetQuat()  # matches physics_human's proven extraction
    img = q.GetImaginary()
    return (
        (float(t[0]), float(t[1]), float(t[2])),
        (float(q.GetReal()), float(img[0]), float(img[1]), float(img[2])),
    )


def _read_scale(prim, UsdGeom) -> tuple[float, float, float]:
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:scale":
            value = op.Get()
            if value is not None:
                return float(value[0]), float(value[1]), float(value[2])
    return 1.0, 1.0, 1.0


def snapshot_rigid_states(stage: "Usd.Stage", prim_paths) -> dict[str, object]:
    """Capture live rigid-body poses and motion for a recoverable action attempt."""
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    states = {}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            continue
        cache.Clear()
        position, orientation = _read_pose(cache, prim)
        rigid = UsdPhysics.RigidBodyAPI(prim)
        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        kinematic = rigid.GetKinematicEnabledAttr()
        states[str(prim_path)] = (
            has_rigid,
            bool(kinematic.Get())
            if kinematic and kinematic.HasAuthoredValue()
            else False,
            position,
            orientation,
            _read_scale(prim, UsdGeom),
            rigid.GetVelocityAttr().Get() if has_rigid else None,
            rigid.GetAngularVelocityAttr().Get() if has_rigid else None,
        )
    return states


def restore_rigid_states(stage: "Usd.Stage", states: dict[str, object]) -> None:
    """Restore states produced by :func:`snapshot_rigid_states`."""
    import fpv_scene
    import physics_human as ph
    from pxr import UsdPhysics

    for prim_path, state in states.items():
        has_rigid, kinematic, position, orientation, scale, velocity, angular = state
        prim = stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            continue
        if has_rigid and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            ph.set_rigid_kinematic(prim, True)
        fpv_scene.set_world_pose(prim, position, orientation, scale)
        if has_rigid and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            ph.set_rigid_kinematic(prim, bool(kinematic))
            rigid = UsdPhysics.RigidBodyAPI(prim)
            if velocity is not None:
                rigid.GetVelocityAttr().Set(velocity)
            if angular is not None:
                rigid.GetAngularVelocityAttr().Set(angular)


def settle_bodies(
    ph,
    stage: "Usd.Stage",
    prim_paths: list[str],
    max_steps: int | None = None,
    traj_out: list | None = None,
    step_fn=None,
) -> dict:
    """Drop-in MULTI-body settle: step PhysX (render=False) until EVERY body rests, read each back.

    Generalizes ``render_runtime._settle_physics`` from one cup to N prims: record each prim's INITIAL
    world xyz, keep a per-prim previous position, and use a velocity function that returns the MAX |v|
    over all bodies (finite-differenced from the USD pose over the 1/60 s dt) so ``ph.settle`` only
    stops once the SLOWEST body is at rest. Reuses ``ph.settle`` UNCHANGED. Returns
    ``{path: (rested_xyz, moved_mm, ok_placeholder)}`` where ``ok_placeholder`` is always True here --
    the physical-correctness reject is decided by the callers (settle_object_on_support/settle_clutter).

    ``traj_out`` (optional): if a list is passed, each PhysX step appends a snapshot
    ``{path: ((tx,ty,tz), (qw,qx,qy,qz))}`` (the FULL transform of every body that step), and the final
    rested pose is appended last -- so a caller can KINEMATICALLY replay the drop over render frames
    (the visible settle). None -> no recording.
    """
    from pxr import Usd, UsdGeom

    sim = get_sim()
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    def _pos(path: str) -> np.ndarray:
        m = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).reshape(
            4, 4
        )
        return m[3, :3].astype(np.float64)

    def _snapshot() -> dict:
        return {p: _read_pose(xc, stage.GetPrimAtPath(p)) for p in prim_paths}

    initial = {p: _pos(p) for p in prim_paths}
    prev = {p: initial[p].copy() for p in prim_paths}
    if traj_out is not None:
        traj_out.append(_snapshot())  # frame 0 = the deliberately-wrong spawn pose

    def _max_vel():
        xc.Clear()
        vmax = 0.0
        for p in prim_paths:
            cur = _pos(p)
            v = float(np.linalg.norm((cur - prev[p]) * 60.0))
            prev[p] = cur
            vmax = max(vmax, v)
        if traj_out is not None:
            traj_out.append(
                _snapshot()
            )  # one snapshot per settle step (post-step pose)
        return vmax

    # NOTE: the kinematic->dynamic flip is made LIVE by physics_human.release_to_dynamic BEFORE this
    # call (the compliance resettle releases the whole batch through that chokepoint), so settle_bodies
    # needs no per-object flip-resync warmup -- ph.settle just steps (synced via step_fn when given, bare
    # otherwise). Init / clutter / blocker settles (step_fn None, FRESH dynamic bodies) take the bare path.
    # Pin the scene rate to the 1/60 this context was BUILT for. A guard only -- in practice it does not
    # fire (the render clock is already at 60 by the first settle); see ph.pinned_scene_timestep. Skipped
    # on the step_fn path: there the render orchestrator drives, and it set >=60 Hz WITH the determinism
    # flags that make a multi-substep advance actually deliver (``_ensure_settle_clock``); re-pinning
    # without them buys slow motion.
    # ⚠️ KNOWN LIMITATION: ``_max_vel``'s ``* 60.0`` assumes one settle step is 1/60 s. Under
    # ``substeps=5`` one step_fn() is 1/12 s, so it over-reads every speed 5x and the vel_thresh rest test
    # is 5x stricter than intended -- conservative (settles run longer), never wrong in the unsafe
    # direction.
    _pin = (
        ph.pinned_scene_timestep(stage) if step_fn is None else contextlib.nullcontext()
    )
    with _pin:
        steps = ph.settle(
            sim,
            _max_vel,
            max_steps=max_steps or (40 + 40 * len(prim_paths)),
            vel_thresh=1e-3,
            step_fn=step_fn,
        )
    xc.Clear()
    out: dict = {}
    for p in prim_paths:
        rested = _pos(p)
        moved_mm = float(np.linalg.norm(rested - initial[p]) * 1000.0)
        out[p] = (rested, moved_mm, True)
    if traj_out is not None:
        traj_out.append(_snapshot())  # exact final rest pose (held after playback)
    LOG.info(
        "settle_bodies: %d prims, %d physics steps (render=False)",
        len(prim_paths),
        steps,
    )
    _ = Usd  # keep the import explicit (matches physics_human style)
    return out


# ---- OFF-TARGET LEDGER (staging half of the post-place verification) ----------------------------
# ``settle_object_on_support`` MEASURES that a staged object failed to stay on its support -- it logs
# `landed off-target (base 0.004 vs support 1.116)` -- but ``init_episode_valid`` does not consume that
# measurement, so an episode whose target objects rest BELOW THE FLOOR (falls of 1.10-1.26 m) can still
# certify its start state as valid. The failure then resurfaces downstream wearing an unrelated label,
# e.g. `path_blocked`, because you cannot navigate to an object that is under the floor.
#
# ⚠️ REPORTING ONLY. This ledger is read into the manifest and is NOT wired to ``init_episode_valid`` or
# to any score, so it never turns a pass into a failure. It keeps the number the engine already computes
# instead of throwing it away.
_OFFTARGET: list[dict] | None = None


def _record_offtarget(
    prim_path: str, base_z: float, support_z: float, xy_roll: float
) -> None:
    global _OFFTARGET
    if _OFFTARGET is None:
        _OFFTARGET = []
    if (
        len(_OFFTARGET) < 200
    ):  # bounded: a pathological scene must not grow the manifest without limit
        _OFFTARGET.append(
            {
                "path": str(prim_path),
                "base_z": round(float(base_z), 4),
                "support_z": round(float(support_z), 4),
                "dz_mm": round((float(base_z) - float(support_z)) * 1000.0, 1),
                "xy_roll_mm": round(float(xy_roll) * 1000.0, 1),
            }
        )


def offtarget_events() -> list[dict]:
    """Every staged object that did not stay on its support, worst fall first. Empty = clean staging."""
    return sorted(_OFFTARGET or [], key=lambda e: e["dz_mm"])


def reset_offtarget() -> None:
    """Clear the ledger (per-episode boundary)."""
    global _OFFTARGET
    _OFFTARGET = None


def requires_upright_stability(
    extents: Sequence[float], isotropic_ratio: float = 0.70
) -> bool:
    """Whether rotating away from the authored up axis is evidence of toppling."""
    dimensions = sorted(abs(float(value)) for value in extents)
    if len(dimensions) != 3 or dimensions[-1] <= 1e-6:
        return True
    return dimensions[0] / dimensions[-1] < float(isotropic_ratio)


def initial_tilt_limit(
    extents: Sequence[float], *, require_upright: bool
) -> float | None:
    """Return an initial-rest tilt limit only when upright is part of the contract."""
    if not require_upright or not requires_upright_stability(extents):
        return None
    return 12.0


def _eroded_support_rectangles(
    bounds_xy: Sequence[Sequence[float]],
    object_half_xy: Sequence[float],
    clearance_m: float,
) -> list[tuple[float, float, float, float]]:
    half_x = max(0.0, float(object_half_xy[0])) + float(clearance_m)
    half_y = max(0.0, float(object_half_xy[1])) + float(clearance_m)
    rectangles = []
    for raw in bounds_xy:
        if len(raw) < 4 or not all(math.isfinite(float(value)) for value in raw[:4]):
            continue
        min_x, min_y, max_x, max_y = (float(value) for value in raw[:4])
        eroded = (min_x + half_x, min_y + half_y, max_x - half_x, max_y - half_y)
        if eroded[0] <= eroded[2] and eroded[1] <= eroded[3]:
            rectangles.append(eroded)
    return rectangles


def _distance_to_rectangle_sq(
    point: tuple[float, float], rectangle: tuple[float, float, float, float]
) -> float:
    x, y = point
    min_x, min_y, max_x, max_y = rectangle
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    return dx * dx + dy * dy


def support_placement_candidates(
    bounds_xy: Sequence[Sequence[float]],
    origin_xy: Sequence[float],
    object_half_xy: Sequence[float],
    *,
    clearance_m: float = 0.012,
    max_distance_m: float = 1.25,
    max_candidates: int = 160,
) -> list[tuple[float, float]]:
    """Sample footprint-safe points from live support-collider bounds."""
    origin = (float(origin_xy[0]), float(origin_xy[1]))
    rectangles = _eroded_support_rectangles(bounds_xy, object_half_xy, clearance_m)
    rectangles.sort(
        key=lambda rect: (
            _distance_to_rectangle_sq(origin, rect),
            (rect[2] - rect[0]) * (rect[3] - rect[1]),
        )
    )
    points = []
    for min_x, min_y, max_x, max_y in rectangles:
        near_x = min(max(origin[0], min_x), max_x)
        near_y = min(max(origin[1], min_y), max_y)
        points.extend(((near_x, near_y), ((min_x + max_x) / 2, (min_y + max_y) / 2)))
    for radius in (0.08, 0.16, 0.24, 0.34, 0.46, 0.60, 0.80, 1.05):
        for index in range(12):
            angle = 2.0 * math.pi * index / 12.0
            point = (
                origin[0] + radius * math.cos(angle),
                origin[1] + radius * math.sin(angle),
            )
            if any(
                rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]
                for rect in rectangles
            ):
                points.append(point)
    for min_x, min_y, max_x, max_y in rectangles:
        for fx in (0.0, 0.25, 0.5, 0.75, 1.0):
            for fy in (0.0, 0.25, 0.5, 0.75, 1.0):
                points.append(
                    (min_x + fx * (max_x - min_x), min_y + fy * (max_y - min_y))
                )
    unique = []
    seen = set()
    for point in points:
        if math.dist(point, origin) > float(max_distance_m):
            continue
        key = (round(point[0], 5), round(point[1], 5))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
        if len(unique) >= int(max_candidates):
            break
    return unique


def _set_support_pose(
    core, rig, fpv_scene, prim, prim_path, xy, z, orientation
) -> None:
    if prim_path == "/World/Obj":
        core.place_obj_resting(rig, np.array([float(xy[0]), float(xy[1]), float(z)]))
        return
    fpv_scene.set_world_pose(
        prim,
        (float(xy[0]), float(xy[1]), float(z)),
        orientation or (1.0, 0.0, 0.0, 0.0),
    )


def _sync_probe_pose(step_fn) -> None:
    if step_fn is not None:
        step_fn()
        return
    get_sim().step(render=False)


def physical_world_bounds(
    fpv_scene, prim
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """World bounds of the physical collision body, excluding labels and render-only geometry."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    found = False
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if child.GetTypeName() != "Mesh" or "/collisions/" not in str(child.GetPath()):
            continue
        bounds = cache.ComputeWorldBound(child).ComputeAlignedRange()
        lower, upper = bounds.GetMin(), bounds.GetMax()
        values = [
            *(float(lower[axis]) for axis in range(3)),
            *(float(upper[axis]) for axis in range(3)),
        ]
        if not _valid_world_values(values) or any(
            float(upper[axis]) < float(lower[axis]) for axis in range(3)
        ):
            continue
        for axis in range(3):
            minimum[axis] = min(minimum[axis], float(lower[axis]))
            maximum[axis] = max(maximum[axis], float(upper[axis]))
        found = True
    if found:
        return tuple(minimum), tuple(maximum)
    bounds = fpv_scene.compute_world_bbox(prim)
    lower, upper = bounds.GetMin(), bounds.GetMax()
    values = [
        *(float(lower[axis]) for axis in range(3)),
        *(float(upper[axis]) for axis in range(3)),
    ]
    if not _valid_world_values(values) or any(
        float(upper[axis]) < float(lower[axis]) for axis in range(3)
    ):
        raise ValueError(
            "physical bounds are non-finite or outside the supported world range"
        )
    return (
        tuple(float(lower[axis]) for axis in range(3)),
        tuple(float(upper[axis]) for axis in range(3)),
    )


def _bbox_metrics(
    fpv_scene, prim, origin_z: float
) -> tuple[float, tuple[float, float], float]:
    minimum, maximum = physical_world_bounds(fpv_scene, prim)
    half_xy = (
        max(0.005, 0.5 * float(maximum[0] - minimum[0])),
        max(0.005, 0.5 * float(maximum[1] - minimum[1])),
    )
    height = max(0.01, float(maximum[2] - minimum[2]))
    return float(minimum[2]) - float(origin_z), half_xy, height


def _object_local_min_z(
    fpv_scene, prim, prim_path: str, rig, core, local_min_z_fn
) -> float:
    if local_min_z_fn is not None:
        return float(local_min_z_fn(prim))
    if prim_path in (getattr(rig, "_obj_path", "/World/Obj"), "/World/Obj"):
        return float(core.local_obj_min_z(rig))
    return float(fpv_scene.local_bbox_min_z(prim))


def _probe_failure_result(
    prim,
    fpv_scene,
    probe: SupportProbe,
    expected_support_z: float,
    attempt: int,
) -> SupportSettleResult:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    position, final_orientation = _read_pose(cache, prim)
    base_z = float(physical_world_bounds(fpv_scene, prim)[0][2])
    return SupportSettleResult(
        position=position,
        orientation=final_orientation,
        moved_mm=0.0,
        base_z=base_z,
        expected_support_z=float(expected_support_z),
        actual_support_z=float(probe.support_z),
        xy_roll_m=0.0,
        tilt_deg=0.0,
        attempts=attempt,
        selected_attempt=attempt,
        support_verified=probe.verified,
        support_path=probe.support_path,
        support_fraction=probe.support_fraction,
        ok=False,
        reason=probe.reason,
    )


def _kinematic_support_result(
    ph,
    prim,
    prim_path: str,
    core,
    rig,
    fpv_scene,
    seat_xy: tuple[float, float],
    support_z: float,
    posed_min_z: float,
    orientation,
    original_position: np.ndarray,
    expected_support_z: float,
    probe: SupportProbe,
    attempt: int,
) -> SupportSettleResult:
    from pxr import Usd, UsdGeom

    ph.set_rigid_kinematic(prim, True)
    origin_z = float(support_z) - float(posed_min_z)
    _set_support_pose(
        core, rig, fpv_scene, prim, prim_path, seat_xy, origin_z, orientation
    )
    measured_base_z = float(physical_world_bounds(fpv_scene, prim)[0][2])
    correction = float(support_z) - measured_base_z
    if abs(correction) > 1e-6:
        origin_z += correction
        _set_support_pose(
            core, rig, fpv_scene, prim, prim_path, seat_xy, origin_z, orientation
        )
        measured_base_z = float(physical_world_bounds(fpv_scene, prim)[0][2])
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    position, final_orientation = _read_pose(cache, prim)
    ok, reason, base_z, xy_roll = _settle_verdict(
        position,
        min_z=posed_min_z,
        seat_xy=seat_xy,
        expected_support_z=float(expected_support_z),
        actual_support_z=float(support_z),
        accept_below=0.0,
        measured_base_z=measured_base_z,
        support_verified=True,
        allow_supported_slide=True,
        max_tilt_deg=None,
    )
    return SupportSettleResult(
        position=position,
        orientation=final_orientation,
        moved_mm=float(
            np.linalg.norm(np.asarray(position, dtype=np.float64) - original_position)
            * 1000.0
        ),
        base_z=base_z,
        expected_support_z=float(expected_support_z),
        actual_support_z=float(support_z),
        xy_roll_m=xy_roll,
        tilt_deg=0.0,
        attempts=attempt,
        selected_attempt=attempt,
        support_verified=True,
        support_path=probe.support_path,
        support_fraction=probe.support_fraction,
        ok=ok,
        reason=("kinematically seated on declared support" if ok else reason),
    )


def settle_object_on_support(
    ph,
    stage: "Usd.Stage",
    rig,
    core,
    prim_path: str,
    support_top_z: float,
    xy,
    z_offset: float = 0.03,
    local_min_z_fn=None,
    accept_below: float = 0.0,
    step_fn=None,
    orientation=None,
    retry_xy: Sequence[tuple[float, float]] = (),
    support_top_fn: Callable[[tuple[float, float]], float] | None = None,
    support_probe_fn: Callable[
        [tuple[float, float], tuple[float, float], float], SupportProbe
    ]
    | None = None,
    kinematic_seat: bool = False,
    require_upright: bool = True,
) -> SupportSettleResult:
    """General single-object form of ``render_runtime._presettle_cup_on_desk`` (any prim, not /World/Obj).

    Same contract: drop the dynamic body ``z_offset`` above ``support_top_z`` at ``xy``, run the
    render=False settle, read where it ACTUALLY rested. ``min_z`` is the object's local bbox bottom:
    ``local_min_z_fn(prim)`` if given, else ``core.local_obj_min_z(rig)`` for the rig's /World/Obj,
    else ``fpv_scene.local_bbox_min_z`` of the prim. The returned record contains the full physical
    pose and an explicit verdict. ``retry_xy`` is a deterministic list of alternate points on the same
    declared support. When ``support_probe_fn`` verifies a candidate, its measured support plane is used
    for that candidate's drop. ``kinematic_seat`` is for objects the program never transports: their
    authored orientation is seated on a verified support instead of being treated as a loose body.
    ``require_upright`` must reflect an explicit task contract; otherwise a dynamically discovered stable
    pose remains valid even when it differs from the asset's authored up axis.

    ``step_fn`` (optional) is forwarded to ``settle_bodies`` -> ``ph.settle``: pass the render orchestrator's
    per-step advance when settling a body that was just flipped kinematic->dynamic (a bare
    ``SimulationContext.step`` cannot re-parse that flip). None -> bare step (byte-identical default).
    """
    import fpv_scene
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    min_z = _object_local_min_z(fpv_scene, prim, prim_path, rig, core, local_min_z_fn)

    original_position = read_prim_xyz(stage, prim_path)
    candidates = [(float(xy[0]), float(xy[1]))]
    candidates.extend((float(px), float(py)) for px, py in retry_xy)
    unique_candidates = list(dict.fromkeys(candidates))
    best: SupportSettleResult | None = None
    for attempt, seat_xy in enumerate(unique_candidates, start=1):
        probe = SupportProbe(
            verified=False,
            ok=True,
            support_z=float(support_top_z),
        )
        probe_origin_z = float(support_top_z) - min_z + 0.60
        _set_support_pose(
            core,
            rig,
            fpv_scene,
            prim,
            prim_path,
            seat_xy,
            probe_origin_z,
            orientation,
        )
        _sync_probe_pose(step_fn)
        probe_current_z = float(read_prim_xyz(stage, prim_path)[2])
        posed_min_z, half_xy, object_height = _bbox_metrics(
            fpv_scene, prim, probe_current_z
        )
        if attempt == 1:
            LOG.info(
                "settle_object_on_support %s physical profile: half_xy=(%.4f,%.4f) "
                "height=%.4f posed_min_z=%.4f",
                prim_path,
                half_xy[0],
                half_xy[1],
                object_height,
                posed_min_z,
            )
        if support_probe_fn is not None:
            probe = support_probe_fn(seat_xy, half_xy, object_height)
        if probe.verified and not probe.ok:
            result = _probe_failure_result(
                prim,
                fpv_scene,
                probe,
                float(support_top_z),
                attempt,
            )
            LOG.info(
                "settle_object_on_support %s attempt=%d rejected before drop: %s",
                prim_path,
                attempt,
                probe.reason,
            )
            if best is None or _settle_error(result) < _settle_error(best):
                best = result
            continue

        candidate_support_z = (
            float(probe.support_z) if probe.verified else float(support_top_z)
        )
        if kinematic_seat and probe.verified:
            result = _kinematic_support_result(
                ph,
                prim,
                prim_path,
                core,
                rig,
                fpv_scene,
                seat_xy,
                candidate_support_z,
                posed_min_z,
                orientation,
                original_position,
                float(support_top_z),
                probe,
                attempt,
            )
            if result.ok:
                return result
            if best is None or _settle_error(result) < _settle_error(best):
                best = result
            continue

        drop_origin_z = candidate_support_z - posed_min_z + float(z_offset)
        _set_support_pose(
            core,
            rig,
            fpv_scene,
            prim,
            prim_path,
            seat_xy,
            drop_origin_z,
            orientation,
        )

        initial_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        _, initial_orientation = _read_pose(initial_cache, prim)
        settled, moved_mm, _ = settle_bodies(ph, stage, [prim_path], step_fn=step_fn)[
            prim_path
        ]
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        position, final_orientation = _read_pose(cache, prim)
        measured_base_z = float(physical_world_bounds(fpv_scene, prim)[0][2])
        final_xy = (float(settled[0]), float(settled[1]))
        actual_support_z = candidate_support_z
        if not probe.verified and support_top_fn is not None:
            actual_support_z = float(support_top_fn(final_xy))
        tilt_deg = _orientation_tilt_deg(initial_orientation, final_orientation)
        extents = (half_xy[0] * 2.0, half_xy[1] * 2.0, object_height)
        max_tilt_deg = initial_tilt_limit(
            extents, require_upright=bool(require_upright)
        )
        ok, reason, base_z, xy_roll = _settle_verdict(
            position,
            min_z=min_z,
            seat_xy=seat_xy,
            expected_support_z=float(support_top_z),
            actual_support_z=actual_support_z,
            accept_below=float(accept_below),
            tilt_deg=tilt_deg,
            max_tilt_deg=max_tilt_deg,
            measured_base_z=measured_base_z,
            support_verified=probe.verified,
            allow_supported_slide=probe.verified,
        )
        result = SupportSettleResult(
            position=position,
            orientation=final_orientation,
            moved_mm=float(moved_mm),
            base_z=base_z,
            expected_support_z=float(support_top_z),
            actual_support_z=actual_support_z,
            xy_roll_m=xy_roll,
            tilt_deg=tilt_deg,
            attempts=attempt,
            selected_attempt=attempt,
            support_verified=probe.verified,
            support_path=probe.support_path,
            support_fraction=probe.support_fraction,
            ok=ok,
            reason=reason,
        )
        LOG.info(
            "settle_object_on_support %s attempt=%d: origin z=%.4f base=%.4f "
            "expected_support=%.4f actual_support=%.4f xy-roll=%.1fmm "
            "tilt=%.1fdeg ok=%s reason=%s",
            prim_path,
            attempt,
            position[2],
            base_z,
            float(support_top_z),
            actual_support_z,
            xy_roll * 1000.0,
            tilt_deg,
            ok,
            reason,
        )
        if ok:
            return result
        if best is None or _settle_error(result) < _settle_error(best):
            best = result

    assert best is not None
    best = replace(best, attempts=len(unique_candidates))
    LOG.warning("settle_object_on_support %s failed: %s", prim_path, best.reason)
    _record_offtarget(prim_path, best.base_z, best.expected_support_z, best.xy_roll_m)
    return best


def _settle_verdict(
    position: tuple[float, float, float],
    *,
    min_z: float,
    seat_xy: tuple[float, float],
    expected_support_z: float,
    actual_support_z: float,
    accept_below: float,
    tilt_deg: float = 0.0,
    max_tilt_deg: float | None = 12.0,
    measured_base_z: float | None = None,
    support_verified: bool = False,
    allow_supported_slide: bool = False,
) -> tuple[bool, str, float, float]:
    values = [
        *position,
        min_z,
        *seat_xy,
        expected_support_z,
        actual_support_z,
        tilt_deg,
    ]
    if measured_base_z is not None:
        values.append(measured_base_z)
    if not _valid_world_values(values):
        return (
            False,
            "physics returned an invalid object pose",
            float("inf"),
            float("inf"),
        )
    base_z = (
        float(position[2]) + float(min_z)
        if measured_base_z is None
        else float(measured_base_z)
    )
    xy_roll = float(np.hypot(position[0] - seat_xy[0], position[1] - seat_xy[1]))
    # The authored/registered top can underestimate a live mesh's local ledge or raised
    # surface. A stable rest on that higher collider is valid; only a lower support means
    # the object fell off the surface it was placed on.
    if (
        not support_verified
        and float(actual_support_z) < float(expected_support_z) - 0.02
    ):
        return False, "settled on a different support height", base_z, xy_roll
    below_tol = max(0.01, float(accept_below))
    if base_z < float(actual_support_z) - below_tol:
        return False, "object base fell below the support", base_z, xy_roll
    # Imported assets often have a collision hull that sits slightly above the
    # visible support.  Treat that small, stable collider gap as supported; a
    # materially floating object is still rejected.
    if base_z > float(actual_support_z) + 0.03:
        return False, "object base remained above the support", base_z, xy_roll
    if xy_roll > 0.03 and not allow_supported_slide:
        return (
            False,
            "object rolled away from the attempted support point",
            base_z,
            xy_roll,
        )
    if max_tilt_deg is not None and float(tilt_deg) > float(max_tilt_deg):
        return (
            False,
            "object tipped away from its declared orientation",
            base_z,
            xy_roll,
        )
    reason = (
        "stable after sliding on declared support"
        if xy_roll > 0.03
        else "stable on declared support"
    )
    return True, reason, base_z, xy_roll


def _orientation_tilt_deg(
    reference: tuple[float, float, float, float],
    actual: tuple[float, float, float, float],
) -> float:
    def local_z(quaternion: tuple[float, float, float, float]) -> np.ndarray:
        q = np.asarray(quaternion, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        w, x, y, z = q / norm
        return np.array(
            [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
            dtype=np.float64,
        )

    cosine = float(np.clip(np.dot(local_z(reference), local_z(actual)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _settle_error(result: SupportSettleResult) -> float:
    return (
        abs(result.base_z - result.actual_support_z)
        + abs(result.actual_support_z - result.expected_support_z)
        + result.xy_roll_m
        + math.radians(result.tilt_deg)
    )


def settle_clutter(
    ph, stage: "Usd.Stage", items: list, traj_out: list | None = None
) -> dict:
    """Drop + settle N clutter prims TOGETHER (deliberately-wrong initial poses) -> physics-correct rest.

    ``items`` = list of ``(prim_path, support_top_z, xy)``. Does NOT re-seat the objects -- the builder
    authored the wrong pose on purpose; the settle is what fixes it. Each prim is made a dynamic rigid
    (``make_dynamic_rigid(density=400, approximation=classify_collider(prim))``), then ALL are settled
    together via ``settle_bodies``, then each rested pose is read back. Per-object off-target reject:
    the settled bbox-bottom must land within ~1 cm of its ``support_top_z`` (a stacked object that
    rests on ANOTHER object legitimately lands higher -> its support is that object, so the reject uses
    a generous 1.5 cm band and only flags a body that FELL THROUGH / did not converge). Returns
    ``{path: (rested_xyz, moved_mm, ok)}``; ok=False bodies are left frozen on their settled pose.

    ``traj_out`` (optional) is forwarded to ``settle_bodies`` to RECORD the full drop trajectory
    (translate + orientation per step) for a kinematic visible-settle replay.
    """
    import fpv_scene

    paths = [it[0] for it in items]
    support = {it[0]: float(it[1]) for it in items}
    for p in paths:
        prim = stage.GetPrimAtPath(p)
        ph.make_dynamic_rigid(
            prim, density=400.0, approximation=classify_collider(prim)
        )

    res = settle_bodies(ph, stage, paths, traj_out=traj_out)

    out: dict = {}
    for p in paths:
        rested, moved_mm, _ok = res[p]
        # base_z = the object's WORLD bbox bottom after settling. Read it directly from the world bbox
        # (robust vs origin+local_min_z: for a prim with a non-zero translate, ComputeLocalBound folds
        # the translate in, so origin+local_min_z double-counts it). BBoxCache reads the PhysX-updated USD
        # pose the settle already relied on.
        base_z = float(fpv_scene.compute_world_bbox(stage.GetPrimAtPath(p)).GetMin()[2])
        dz = base_z - support[p]
        # ok = it did not fall THROUGH the support (base far below the surface) or fly off; a body that
        # settled ON another object rests ABOVE its own support_top -> still ok (dz > 0). A small negative
        # dz is a benign convex-hull base gap (the hull can sit a cm or two off the visual bottom); we
        # only reject a body that clearly sank below the surface or is still floating high after converge.
        ok = (dz > -0.025) and (dz < 0.20)
        out[p] = (rested, moved_mm, bool(ok))
        LOG.info(
            "settle_clutter %s: base z=%.4f vs support %.4f (dz=%.1fmm) moved=%.1fmm ok=%s",
            p,
            base_z,
            support[p],
            dz * 1000.0,
            moved_mm,
            ok,
        )
    return out


def all_pairs_separation(
    ph, stage: "Usd.Stage", prim_paths: list[str], sep_tol: float = 0.002
) -> list:
    """All-pairs penetration gate: every prim-prim pair whose min contact separation is < -sep_tol.

    Reuses ``ph.enable_contact_reports`` + one settle step + ``ph.read_contacts()`` (does NOT touch
    ``physics_human.classify_contacts`` -- render_runtime depends on its 3-bucket contract). Each
    contact collider int is decoded to its SdfPath (``PhysicsSchemaTools.intToSdfPath``), matched to
    the owning prim by prefix (colliders live on child meshes), and the most-negative separation over
    the contact range is kept. Returns ``[(a, b, min_sep_m), ...]`` for penetrating pairs only.
    """
    from pxr import PhysicsSchemaTools

    ph.enable_contact_reports(stage, list(prim_paths))
    sim = get_sim()
    sim.step(render=False)
    headers, data = ph.read_contacts()

    roots = [p.rstrip("/") for p in prim_paths]

    def _owner(path: str) -> str | None:
        for r in roots:
            if path == r or path.startswith(r + "/"):
                return r
        return None

    worst: dict = {}
    for h in headers:
        a = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
        b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        oa, ob = _owner(a), _owner(b)
        if oa is None or ob is None or oa == ob:
            continue
        off, num = int(h.contact_data_offset), int(h.num_contact_data)
        seps = [float(data[k].separation) for k in range(off, off + num)]
        if not seps:
            continue
        key = tuple(sorted((oa, ob)))
        min_sep = min(seps)
        worst[key] = min(worst.get(key, 0.0), min_sep)

    out = [(a, b, sep) for (a, b), sep in worst.items() if sep < -float(sep_tol)]
    out.sort(key=lambda t: t[2])
    return out

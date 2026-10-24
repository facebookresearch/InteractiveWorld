# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Scene construction: room, tables, props, and the bbox helpers that seat them.

Uses RAW pxr USD (``DefinePrim`` + ``GetReferences().AddReference``) rather than
``isaacsim.core.experimental`` to minimize namespace drift across Isaac versions —
the experimental API churns, the USD reference API does not.

All vertical placement (table-top z, prop rest-z) is COMPUTED at runtime via
``UsdGeom.BBoxCache`` so nothing is hardcoded. The numbers in the spec's scene_layout
express intent, not magic constants used here.

Coordinate frame (whole project): meters, Z-up. +X = forward/walk, +Y = left, +Z = up.
Floor at z=0.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fpv_core import ASSET_ROOT

if TYPE_CHECKING:
    from pxr import Gf, Usd  # noqa: F401

LOG = logging.getLogger("fpv_scene")


def add_reference(
    stage: "Usd.Stage",
    prim_path: str,
    abs_usd_path: str,
    *,
    isolate_xform: bool = False,
) -> "Usd.Prim":
    """Define an Xform at ``prim_path`` and add ``abs_usd_path`` as a reference.

    References the asset's default prim (robust against sub-prim renames). Returns the
    outer prim so the caller can pose it. ``isolate_xform`` places the reference below
    an ``m`` child, leaving a clean outer transform for a rigid body. This is required
    for movable BEHAVIOR assets: their root ``quatd`` xform op conflicts with PhysX's
    live pose write after an animated object switches to a matrix transform.
    """
    prim = stage.DefinePrim(prim_path, "Xform")
    referenced = (
        stage.DefinePrim(f"{prim_path.rstrip('/')}/m", "Xform")
        if isolate_xform
        else prim
    )
    refs = referenced.GetReferences()
    ok = refs.AddReference(abs_usd_path)
    # AddReference returns bool in most USD builds; log but don't hard-fail on the
    # return value (composition errors surface at traversal time with clearer info).
    LOG.info(
        "add_reference %s%s -> %s (ok=%s)",
        prim_path,
        "/m" if isolate_xform else "",
        abs_usd_path,
        ok,
    )
    if not prim.IsValid():
        raise RuntimeError(
            f"Referenced prim {prim_path} is invalid after AddReference({abs_usd_path})"
        )
    _log_layer_meters_per_unit(prim_path, abs_usd_path)
    return prim


def _log_layer_meters_per_unit(prim_path: str, abs_usd_path: str) -> None:
    """Log a referenced layer's metersPerUnit for fast scale-mismatch diagnosis.

    Up-axis/metersPerUnit are NOT inherited across references, so the root stage being
    meter-scale does not auto-correct a cm-scale referenced asset. The root stage has no
    global guard for that (it surfaces as a wrong bbox in spawn_prop/add_table); logging the
    per-layer value here makes a scale mismatch obvious at reference time. Best-effort.
    """
    from pxr import Usd, UsdGeom

    try:
        side = Usd.Stage.Open(abs_usd_path)
        if side is None:
            return
        mpu = UsdGeom.GetStageMetersPerUnit(side)
        up = UsdGeom.GetStageUpAxis(side)
        flag = (
            ""
            if abs(mpu - 1.0) <= 1e-6
            else "  <-- NOT meter-scale; expect bbox mismatch"
        )
        LOG.info(
            "  layer %s: metersPerUnit=%.4f up-axis=%s%s", prim_path, mpu, up, flag
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic only; never block referencing
        LOG.debug("Could not probe metersPerUnit for %s (%s)", abs_usd_path, exc)


def set_world_pose(
    prim: "Usd.Prim",
    pos: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float] | None = (1.0, 0.0, 0.0, 0.0),
    scale: "float | tuple[float, float, float] | None" = 1.0,
) -> None:
    """Rewrite the prim's xformOpOrder with translate, orient (w,x,y,z), and scale (uniform float OR per-axis 3-tuple), preserving any pre-authored extra xformOps.

    Idempotent: reuses existing ops if present, otherwise creates them, and pins the
    op order so repeated calls don't stack duplicate ops.

    PRECISION-AWARE: a referenced asset (e.g. Simple_Room) can ship its own ``xformOp:orient``
    authored as ``GfQuatf`` (and translate/scale as float). ``.Set(GfQuatd)`` on a single-precision
    op raises ``Tf.ErrorException`` ("Type mismatch ... expected 'GfQuatf', got 'GfQuatd'"). So we
    Set each op with the precision its existing typeName declares (double otherwise).
    """
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)

    # Find-or-create each op.  A previous MakeMatrixXform() can leave the core
    # attributes authored but remove them from xformOpOrder, so ordered ops alone
    # are not enough to decide whether Add*Op is safe.
    existing = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    for name in ("xformOp:translate", "xformOp:orient", "xformOp:scale"):
        if name in existing:
            continue
        attr = prim.GetAttribute(name)
        if attr.IsValid():
            existing[name] = UsdGeom.XformOp(attr)

    t_op = existing.get("xformOp:translate") or xform.AddTranslateOp()
    o_op = existing.get("xformOp:orient") or xform.AddOrientOp()
    s_op = existing.get("xformOp:scale") or xform.AddScaleOp()

    _set_vec3(t_op, (float(pos[0]), float(pos[1]), float(pos[2])), Gf)
    # ``None`` = LEAVE THIS OP ALONE. The defaults write identity/1.0, so a caller that only wanted to MOVE
    # something silently also straightened and re-sized it -- which is a placement bug when the thing being
    # moved is an object whose orientation was earned (a bottle carried at the angle the hand gripped it).
    # Explicit None lets a caller express "move only".
    if quat_wxyz is not None:
        _set_quat(
            o_op,
            (
                float(quat_wxyz[0]),
                float(quat_wxyz[1]),
                float(quat_wxyz[2]),
                float(quat_wxyz[3]),
            ),
            Gf,
        )
    if scale is not None:
        if isinstance(scale, (int, float)):
            sx = sy = sz = float(scale)
        else:  # per-axis (BEHAVIOR assets ship non-uniform scale)
            sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
        _set_vec3(s_op, (sx, sy, sz), Gf)

    # Preserve any pre-authored xformOps (e.g. a referenced asset's pivot) instead of clobbering
    # the order down to exactly [t,o,s]. A fresh Xform has no extras, so the resulting order is
    # exactly [t,o,s].
    #
    # EXCEPT xformOp:transform. That op is a COMPLETE transform, not a pivot-style extra, so ordering
    # it after [t,o,s] COMPOSES the two and the prim lands at pose + stale_matrix. It appears whenever
    # smplh_anim_core.set_object_world() has run on this prim: MakeMatrixXform() rewrites the order to
    # exactly ['xformOp:transform'], which drops t/o/s out of the ORDER (their attributes survive), so
    # the find-or-create above re-adds them and the matrix ends up last. Only the render path calls
    # set_object_world, and only a multi-episode env writes a pose afterwards -- without this, every
    # episode after the first would place objects at start+end of the previous episode, compounding
    # to metres off within a few episodes, while a one-shot render that never restores a pose is
    # unaffected. Dropping it from the order leaves the attribute inert; the next
    # set_object_world call re-establishes it via MakeMatrixXform as usual.
    _core = {"xformOp:translate", "xformOp:orient", "xformOp:scale"}
    _stale_matrix = {"xformOp:transform"}
    _extras = [
        op for nm, op in existing.items() if nm not in _core and nm not in _stale_matrix
    ]
    xform.SetXformOpOrder([t_op, o_op, s_op] + _extras)


def _is_single_precision(op) -> bool:
    """True if the xform op's value typeName is single-precision (float), so we must Set float types."""
    try:
        tn = op.GetTypeName()  # pxr.Sdf.ValueTypeName
        name = str(tn)
    except Exception:  # noqa: BLE001 - if we cannot read it, assume double (the safe AddOp default)
        return False
    # Single-precision USD types render as e.g. 'float3', 'quatf'; double as 'double3', 'quatd'.
    return ("float" in name) or ("quatf" in name) or name.endswith("f")


def _set_vec3(op, xyz: tuple[float, float, float], Gf) -> None:
    """Set a translate/scale op, matching its declared float/double precision."""
    if _is_single_precision(op):
        op.Set(Gf.Vec3f(xyz[0], xyz[1], xyz[2]))
    else:
        op.Set(Gf.Vec3d(xyz[0], xyz[1], xyz[2]))


def _set_quat(op, wxyz: tuple[float, float, float, float], Gf) -> None:
    """Set an orient op, matching its declared GfQuatf/GfQuatd precision."""
    if _is_single_precision(op):
        op.Set(Gf.Quatf(wxyz[0], wxyz[1], wxyz[2], wxyz[3]))
    else:
        op.Set(Gf.Quatd(wxyz[0], wxyz[1], wxyz[2], wxyz[3]))


def _yaw_quat(yaw_deg: float) -> "Gf.Quatd":
    """Quaternion for a yaw (rotation about +Z) in degrees, returned as Gf.Quatd."""
    from pxr import Gf

    rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(yaw_deg))
    return rot.GetQuat()


def compute_world_bbox(prim: "Usd.Prim") -> "Gf.Range3d":
    """World-aligned bbox of ``prim`` (default + render purposes), aligned range."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bound = cache.ComputeWorldBound(prim)
    rng = bound.ComputeAlignedRange()
    return rng


def local_bbox_min_z(prim: "Usd.Prim") -> float:
    """Geometry-base offset (min-z) relative to the prim's OWN ORIGIN, independent of where the prim is
    currently posed.

    Uses ``BBoxCache.ComputeUntransformedBound``, which EXCLUDES the transform authored on the prim
    itself (its translate + rotation) while still including its descendants' transforms -- so it returns
    the intrinsic geometry offset (e.g. ~-0.0156 for a book) whether the prim sits at z=0 or has been
    moved onto a table. NOTE ``ComputeLocalBound`` does the opposite: it FOLDS the prim's own transform
    in, so for a top-level prim under identity /World it equals the WORLD bound and returns the posed z,
    which would make a caller probing a posed prim double-count the pose.
    Used by ``spawn_prop`` etc. so the bbox BOTTOM seats on the support top, correct for both base-pivot
    SM_ props (offset ~ 0) and center-pivot YCB props (offset < 0).
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    # ComputeUntransformedBound omits the prim's own xform (translate/rotate) -> the pure geometry offset
    # from the prim origin. At z=0/identity this equals ComputeLocalBound; on a posed prim it stays the
    # true offset instead of the world z.
    local_bound = cache.ComputeUntransformedBound(prim)
    rng = local_bound.ComputeAlignedRange()
    return float(rng.GetMin()[2])


def build_room(stage: "Usd.Stage") -> "Usd.Prim":
    """Reference the Simple_Room (floor + skysphere; ships NO table) at identity."""
    room = add_reference(
        stage, "/World/Room", ASSET_ROOT + "/Environments/Simple_Room/simple_room.usd"
    )
    set_world_pose(room, (0.0, 0.0, 0.0))
    LOG.info("Room built at identity; floor z=0")
    return room


def add_table(
    stage: "Usd.Stage", name: str, pos: tuple[float, float, float]
) -> tuple["Usd.Prim", float]:
    """Reference packing_table.usd at /World/<name>; return (prim, table_top_z).

    ``table_top_z`` = world bbox max[2], computed (never hardcoded).
    """
    prim = add_reference(
        stage, f"/World/{name}", ASSET_ROOT + "/Props/PackingTable/packing_table.usd"
    )
    set_world_pose(prim, pos)
    rng = compute_world_bbox(prim)
    table_top_z = float(rng.GetMax()[2])
    LOG.info("Table %s at %s -> table_top_z=%.4f (computed)", name, pos, table_top_z)
    return prim, table_top_z


def spawn_prop(
    stage: "Usd.Stage",
    name: str,
    usd_rel: str,
    table_top_z: float,
    xy: tuple[float, float],
    yaw_deg: float = 0.0,
    scale: float = 1.0,
) -> "Usd.Prim":
    """Reference a prop at /World/<name> and seat its bbox BOTTOM on the table top.

    rest_z = table_top_z - local_bbox_min_z. This is correct for base-pivot SM props
    (min_z ~ 0 => rest_z ~ table_top) and center-pivot YCB props (min_z < 0 => the prop
    is lifted so its bottom touches the table), with NO +half_height fudge.
    """
    prim = add_reference(stage, f"/World/{name}", ASSET_ROOT + "/" + usd_rel)
    # Pose at z=0 first so the local-min-z probe sees the asset's intrinsic geometry
    # offset (scale applied) without an arbitrary z already baked in.
    set_world_pose(
        prim, (float(xy[0]), float(xy[1]), 0.0), _to_wxyz(_yaw_quat(yaw_deg)), scale
    )

    min_z = local_bbox_min_z(prim)
    rest_z = float(table_top_z) - min_z
    set_world_pose(
        prim, (float(xy[0]), float(xy[1]), rest_z), _to_wxyz(_yaw_quat(yaw_deg)), scale
    )
    LOG.info(
        "Prop %s (%s) seated at xy=%s rest_z=%.4f (table_top=%.4f, local_min_z=%.4f, scale=%.3f)",
        name,
        usd_rel,
        xy,
        rest_z,
        table_top_z,
        min_z,
        scale,
    )
    return prim


def _to_wxyz(quat: "Gf.Quatd") -> tuple[float, float, float, float]:
    """Convert a Gf.Quatd to a (w,x,y,z) tuple for set_world_pose."""
    img = quat.GetImaginary()
    return (float(quat.GetReal()), float(img[0]), float(img[1]), float(img[2]))


def add_table_top_collider(table_prim: "Usd.Prim") -> None:
    """Apply UsdPhysics.CollisionAPI to the table (ONLY used in physics-weld mode).

    The default keyframed path never touches physics, so this is a no-op there.
    """
    from pxr import UsdPhysics

    UsdPhysics.CollisionAPI.Apply(table_prim)
    LOG.info("CollisionAPI applied to %s (physics-weld mode)", table_prim.GetPath())

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Biped human rig: load, joint resolution, ANALYTIC head/wrist xforms, body hiding.

Key design decision: the UsdSkel animation-evaluating runtime (omni.anim.skelJoint /
omni.anim.graph.core) is NOT loaded by the default standalone SimulationApp experience, so
a bound ``skel:animationSource`` + ``timeline.play()`` likely leaves the Biped in REST pose
— which would also FREEZE a head-joint-driven camera. Therefore the head and wrist
transforms used by the camera and grasp are ANALYTIC by construction: derived from the
/World/Human root xform (set by the locomotion module) composed with fixed/scripted local
offsets. They are deterministic regardless of whether UsdSkel evaluates.

The richer UsdSkel-eval path is opt-in behind ``LocoCfg.drive_skel_anim`` and is gated by
``assert_skel_animates`` (which proves the head actually moves) — if it can't move, the
check raises and tells the caller to fall back to analytic mode.

Because the Biped is a SINGLE skinned Body_Mesh, per-prim torso hiding is impossible;
``set_body_visible_for_camera`` hides the WHOLE body in the FPV render product only, and
``make_proxy_forearm`` adds an in-frame skin-colored capsule so the egocentric view has a
hand without the chest-blob.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fpv_core import ASSET_ROOT
from fpv_scene import add_reference, set_world_pose

if TYPE_CHECKING:
    from fpv_core import CameraCfg, LocoCfg
    from isaacsim import SimulationApp  # noqa: F401
    from pxr import Gf, Usd, UsdSkel  # noqa: F401

LOG = logging.getLogger("fpv_human")

# Analytic rig constants. Eye height is a 1.62 m rig-local value; the rest are plausible
# bind-pose offsets for the hold/reach poses. Re-derive them from the rig's real Head/Wrist
# rest world positions if a different Biped asset is used.
EYE_HEIGHT_M: float = 1.62
SHOULDER_HEIGHT_M: float = 1.42
SHOULDER_HALF_WIDTH_M: float = 0.20  # +Y for left shoulder, -Y for right
ELBOW_DROP_M: float = 0.28  # below shoulder at rest
FOREARM_LEN_M: float = 0.27
UPPER_ARM_LEN_M: float = 0.28


@dataclass
class BipedRig:
    """Handle to the loaded Biped + resolved joint indices + scripted per-hand state."""

    skel_prim: "Usd.Prim"
    skeleton: "UsdSkel.Skeleton"
    root_path: str = "/World/Human"
    joint_tokens: list[str] = field(default_factory=list)
    head_idx: int = -1
    l_wrist_idx: int = -1
    r_wrist_idx: int = -1

    # When False (DEFAULT) head_world_xform/wrist_world_xform are PURELY ANALYTIC: the skel
    # path is never consulted. This is load-bearing -- UsdSkelSkeletonQuery
    # .ComputeJointWorldTransforms falls back to the skeleton REST transforms when no
    # animationSource is bound (pure pxr math, no Kit anim runtime needed), so an ungated
    # skel read would silently return BIND-POSE world transforms and the analytic
    # EYE_HEIGHT_M/forward design + set_wrist_target reaches would be ignored. Only flip True
    # in the opt-in drive_skel_anim path AFTER assert_skel_animates proves the rig moves.
    drive_skel_anim: bool = False

    # Runtime mutable state set by locomotion + grasp/demo loop:
    #   _root_xy / _root_yaw_deg : current root pose (set by step_root / hold_root)
    #   _wrist_world['L'/'R']    : world-space target the analytic wrist should sit at
    #   _skel_cache              : one cached UsdSkel.Cache + SkelQuery (skel path only)
    _root_xy: tuple[float, float] = (0.0, 0.0)
    _root_yaw_deg: float = 0.0
    _wrist_world: dict = field(default_factory=dict)
    _skel_cache: object = None
    _skel_query: object = None


def load_biped(
    stage: "Usd.Stage",
    pos: tuple[float, float, float] = (-0.85, 0.0, 0.0),
    yaw_deg: float = 0.0,
    drive_skel_anim: bool = False,
) -> BipedRig:
    """Reference Biped_Setup.usd at /World/Human; force meter-scale + Z-up; return rig.

    * Meter-scale guard: if the referenced layer's metersPerUnit != 1.0, author
      xformOp:scale=0.01 on /World/Human so the human stands ~1.75 m in the meter stage.
    * Up-axis guard: if the rig is authored Y-up while the stage is Z-up, prepend a +90deg
      rotate about X on /World/Human.
    * ``drive_skel_anim`` (default False) is stored on the rig and is the ONLY switch that
      lets head_world_xform/wrist_world_xform consult the UsdSkel path; pass LocoCfg
      .drive_skel_anim so the camera/grasp math is analytic by default.
    """
    from pxr import Usd, UsdGeom

    abs_path = ASSET_ROOT + "/People/Characters/Biped_Setup.usd"
    prim = add_reference(stage, "/World/Human", abs_path)

    # Inspect the referenced layer's stage metadata (units + up-axis) WITHOUT mounting it
    # into the live stage's composition decisions — open it read-only on the side.
    rig_meters = 1.0
    rig_up = UsdGeom.Tokens.z
    try:
        side = Usd.Stage.Open(abs_path)
        if side is not None:
            rig_meters = UsdGeom.GetStageMetersPerUnit(side)
            rig_up = UsdGeom.GetStageUpAxis(side)
    except Exception as exc:  # noqa: BLE001 - if probe fails, assume meter/Z-up and log
        LOG.warning(
            "Could not open Biped layer to probe units/up-axis (%s); assuming meters/Z-up",
            exc,
        )

    scale = 1.0
    if abs(rig_meters - 1.0) > 1e-6:
        # Asset is cm-scale (metersPerUnit=0.01) => shrink by 0.01 into the meter stage.
        scale = (
            rig_meters  # metersPerUnit == meters-per-unit; 0.01 cm-rig => scale 0.01
        )
        LOG.info(
            "Biped metersPerUnit=%.4f != 1.0 => forcing xformOp:scale=%.4f",
            rig_meters,
            scale,
        )

    quat = _yaw_quat_wxyz(yaw_deg)
    set_world_pose(prim, pos, quat, scale)

    if str(rig_up) == str(UsdGeom.Tokens.y):
        _prepend_x_rotate(prim, 90.0)
        LOG.info(
            "Biped authored Y-up in Z-up stage => prepended +90deg X rotate on /World/Human"
        )

    # Find the SkelRoot/Skeleton beneath /World/Human.
    skeleton = _find_skeleton(stage, prim)
    rig = BipedRig(
        skel_prim=prim,
        skeleton=skeleton,
        root_path="/World/Human",
        drive_skel_anim=bool(drive_skel_anim),
    )
    rig._root_xy = (float(pos[0]), float(pos[1]))
    rig._root_yaw_deg = float(yaw_deg)
    LOG.info(
        "Biped loaded at %s yaw=%.1f (skeleton=%s, drive_skel_anim=%s)",
        pos,
        yaw_deg,
        skeleton.GetPath() if skeleton else "NONE",
        rig.drive_skel_anim,
    )
    return rig


def _find_skeleton(stage: "Usd.Stage", root_prim: "Usd.Prim") -> "UsdSkel.Skeleton":
    """Depth-first search beneath ``root_prim`` for the first UsdSkel.Skeleton."""
    from pxr import Usd, UsdSkel

    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdSkel.Skeleton):
            return UsdSkel.Skeleton(prim)
    raise RuntimeError(
        f"No UsdSkel.Skeleton found beneath {root_prim.GetPath()}; Biped_Setup.usd structure "
        "differs from expectation. Run verify_assets.py to dump the SkelRoot/Skeleton path."
    )


def resolve_joint_indices(rig: BipedRig) -> None:
    """Fill head/l_wrist/r_wrist indices by case-insensitive substring match.

    Reads the live ``skel:joints`` token list. Raises with the full token list if any of
    Head / R_Wrist / L_Wrist cannot be located (so the orchestrator can bake the exact
    spelling). Joint convention is the biped L_/R_ form (mixamorig ruled out).
    """
    joints_attr = rig.skeleton.GetJointsAttr()
    tokens = list(joints_attr.Get() or [])
    rig.joint_tokens = [str(t) for t in tokens]
    if not rig.joint_tokens:
        raise RuntimeError(f"Skeleton {rig.skeleton.GetPath()} has empty skel:joints")

    rig.head_idx = _find_joint(rig.joint_tokens, ["head"])
    rig.r_wrist_idx = _find_joint(
        rig.joint_tokens, ["r_wrist", "rightwrist", "righthand", "r_hand"]
    )
    rig.l_wrist_idx = _find_joint(
        rig.joint_tokens, ["l_wrist", "leftwrist", "lefthand", "l_hand"]
    )

    missing = []
    if rig.head_idx < 0:
        missing.append("Head")
    if rig.r_wrist_idx < 0:
        missing.append("R_Wrist")
    if rig.l_wrist_idx < 0:
        missing.append("L_Wrist")
    if missing:
        raise RuntimeError(
            f"Could not resolve joints {missing} by substring. Full skel:joints token list "
            f"({len(rig.joint_tokens)}):\n  " + "\n  ".join(rig.joint_tokens)
        )
    LOG.info(
        "Resolved joints: head_idx=%d (%s), r_wrist_idx=%d (%s), l_wrist_idx=%d (%s)",
        rig.head_idx,
        rig.joint_tokens[rig.head_idx],
        rig.r_wrist_idx,
        rig.joint_tokens[rig.r_wrist_idx],
        rig.l_wrist_idx,
        rig.joint_tokens[rig.l_wrist_idx],
    )


def _find_joint(tokens: list[str], needles: list[str]) -> int:
    """Return the index of the LAST path segment matching any needle (case-insensitive).

    UsdSkel joint tokens are usually full paths like 'Root/Spine/.../R_Wrist'; match on
    the leaf segment so e.g. 'head' does not match 'forehead_attach' upstream.
    """
    for i, tok in enumerate(tokens):
        leaf = tok.split("/")[-1].lower()
        for needle in needles:
            if needle in leaf:
                return i
    return -1


def root_xform(
    rig: BipedRig, frame: int, cfg: "CameraCfg", loco: "LocoCfg"
) -> "Gf.Matrix4d":
    """World xform of /World/Human at this frame (translation from _root_xy, yaw about Z).

    Single source of truth for analytic joint placement. The actual /World/Human prim is
    posed by the locomotion module; this recomputes the same matrix for math use.
    """
    from pxr import Gf

    m = Gf.Matrix4d(1.0)
    rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rig._root_yaw_deg))
    trans = Gf.Matrix4d().SetTranslate(Gf.Vec3d(rig._root_xy[0], rig._root_xy[1], 0.0))
    m = rot * trans
    return m


def head_world_xform(rig: BipedRig, frame: int) -> "Gf.Matrix4d":
    """World head transform.

    DEFAULT (analytic): root xform composed with a FIXED head offset (eye height in
    rig-local, forward = +X), deterministic regardless of UsdSkel eval.
    If skel-eval is active and resolvable, use the real head joint world transform.
    """
    from pxr import Gf

    # Consult the skel-eval path ONLY when the caller opted in (rig.drive_skel_anim). The
    # analytic fallback is the contract and is always correct for the FPV (legs out of
    # frame). Gating here is critical: an ungated skel read returns BIND-POSE world
    # transforms (ComputeJointWorldTransforms falls back to skeleton rest with no anim
    # source) and would silently override the analytic EYE_HEIGHT_M/forward camera.
    if rig.drive_skel_anim:
        skel_xf = _try_skel_joint_world(rig, rig.head_idx, frame)
        if skel_xf is not None:
            return skel_xf

    root = root_xform(rig, frame, None, None)  # type: ignore[arg-type]
    head_local = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, EYE_HEIGHT_M))
    return head_local * root


def wrist_world_xform(rig: BipedRig, side: str, frame: int) -> "Gf.Matrix4d":
    """World wrist transform for hand ``side`` in {'L','R'}.

    DEFAULT (analytic): if the demo/grasp loop stored a scripted world wrist target on the
    rig (``rig._wrist_world[side]``), use it; otherwise fall back to a neutral rest pose
    (arm down at the side). Skel-eval path picks the real l/r wrist joint when active.
    """
    from pxr import Gf

    side = side.upper()
    # Skel path is opt-in (see head_world_xform): otherwise the stored scripted target / rest
    # pose is authoritative, never the bind-pose wrist joint.
    if rig.drive_skel_anim:
        skel_idx = rig.r_wrist_idx if side == "R" else rig.l_wrist_idx
        skel_xf = _try_skel_joint_world(rig, skel_idx, frame)
        if skel_xf is not None:
            return skel_xf

    stored = rig._wrist_world.get(side)
    if stored is not None:
        # Stored as a world-space Gf.Vec3d target; build a translation-only frame.
        return Gf.Matrix4d().SetTranslate(stored)

    # Neutral rest: arm hanging at the side.
    root = root_xform(rig, frame, None, None)  # type: ignore[arg-type]
    sign = 1.0 if side == "L" else -1.0
    rest_local = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(
            0.05,
            sign * SHOULDER_HALF_WIDTH_M,
            SHOULDER_HEIGHT_M - ELBOW_DROP_M - FOREARM_LEN_M,
        )
    )
    return rest_local * root


def set_wrist_target(rig: BipedRig, side: str, world_pos: "Gf.Vec3d") -> None:
    """Store the scripted world wrist position for a hand (used by reach/carry/place)."""
    rig._wrist_world[side.upper()] = world_pos


def _try_skel_joint_world(
    rig: BipedRig, joint_idx: int, frame: int
) -> "Gf.Matrix4d | None":
    """Best-effort UsdSkel world transform for one joint; None if eval is unavailable.

    Never raises — the analytic path is the contract; this is an optional enhancement.
    """
    if joint_idx is None or joint_idx < 0:
        return None
    try:
        from pxr import Usd, UsdGeom, UsdSkel

        # Cache the Cache + SkelQuery ONCE on the rig (re-deriving them every joint/frame is
        # wasteful and the cache is what makes XformCache reuse effective).
        skel_query = rig._skel_query
        if skel_query is None:
            cache = UsdSkel.Cache()
            skel_query = cache.GetSkelQuery(rig.skeleton)
            if not skel_query:
                return None
            rig._skel_cache = cache
            rig._skel_query = skel_query
        xf_cache = UsdGeom.XformCache(Usd.TimeCode(float(frame)))
        xforms = skel_query.ComputeJointWorldTransforms(xf_cache)
        if not xforms or joint_idx >= len(xforms):
            return None
        return xforms[joint_idx]
    except Exception:  # noqa: BLE001 - any failure => fall back to analytic
        return None


def bind_walk_animation(rig: BipedRig, loco: "LocoCfg") -> None:
    """Bind the in-place walk clip as the skel animation source (ONLY drive_skel_anim).

    No-op in default analytic mode: a non-deforming body translated by the root reads as
    walking egocentrically because the legs are out of the FPV frame.
    """
    if not loco.drive_skel_anim:
        LOG.info("bind_walk_animation: analytic mode (drive_skel_anim=False) => no-op")
        return

    from pxr import UsdSkel

    stage = rig.skel_prim.GetStage()
    anim_path = "/World/Human/WalkAnim"
    anim_prim = add_reference(stage, anim_path, ASSET_ROOT + "/" + loco.walk_clip_rel)
    binding = UsdSkel.BindingAPI.Apply(rig.skeleton.GetPrim())
    binding.CreateAnimationSourceRel().SetTargets([anim_prim.GetPath()])
    LOG.info(
        "Bound walk clip %s as animationSource on %s",
        loco.walk_clip_rel,
        rig.skeleton.GetPath(),
    )


def assert_skel_animates(rig: BipedRig, app: "SimulationApp", timeline) -> None:
    """Prove the rig actually animates (ONLY drive_skel_anim). Static dump is insufficient.

    Reads the REAL skel head-joint world translation DIRECTLY (``_try_skel_joint_world``, not
    ``head_world_xform`` -- the latter returns the analytic offset and would compare two
    identical rest reads and falsely raise / pass), steps 10 frames, and asserts the
    translation delta exceeds 1e-4. Raises an actionable message if the rig is frozen.

    NOTE: this CONSUMES ~10 warmup ``app.update()`` frames before capture starts; call it
    BEFORE the capture loop so those frames are never expected in the PNG sequence. Only
    meaningful when ``rig.drive_skel_anim`` is True (callers gate the call on
    ``LocoCfg.drive_skel_anim``).
    """
    from pxr import Gf

    def _head_skel() -> "Gf.Vec3d | None":
        xf = _try_skel_joint_world(rig, rig.head_idx, 0)
        return Gf.Vec3d(xf.ExtractTranslation()) if xf is not None else None

    before = _head_skel()
    if before is None:
        raise RuntimeError(
            "assert_skel_animates: UsdSkel head-joint world transform is unavailable "
            "(no SkelQuery). Cannot drive the skel path; use analytic mode "
            "(drive_skel_anim=False)."
        )
    for _ in range(10):
        app.update()
    after = _head_skel()
    if after is None:
        raise RuntimeError(
            "assert_skel_animates: skel head transform vanished after stepping"
        )
    delta = (after - before).GetLength()
    LOG.info(
        "assert_skel_animates: skel head translation delta over 10 frames = %.6f", delta
    )
    if delta <= 1e-4:
        raise RuntimeError(
            "UsdSkel not evaluating: enable omni.anim.skelJoint/omni.anim.graph.core or use "
            "analytic mode (drive_skel_anim=False). Skel head world delta over 10 frames was "
            f"{delta:.6e} (<=1e-4). The default standalone experience does not load the anim "
            "runtime; the analytic FPV path does not need it."
        )


def set_body_visible_for_camera(
    rig: BipedRig, cam_path: str, visible: bool, allow_global_hide: bool = False
) -> None:
    """GLOBAL visibility of the single Body_Mesh -- there is NO per-render-product scoping.

    Isaac Sim 5.1 has no ``hide_render_products`` / per-camera visibility /
    render-product-scoped invisibility API. ``UsdGeom`` visibility is a single stage
    attribute, so ``MakeInvisible()`` here hides the body in BOTH the FPV product AND the TPV
    product. The TPV is a third-person verification video that must show the human
    performing grab/carry/place, so a global hide would produce a TPV with only a floating
    object + proxy capsule and no human. The default therefore keeps the body visible.

    Therefore:
      * ``visible=True``  -> ``MakeVisible()`` (always allowed).
      * ``visible=False`` is a GLOBAL hide and is REFUSED unless ``allow_global_hide=True``
        is passed explicitly. The default refuses + logs WARNING and leaves the body VISIBLE
        (accept-some-chest, realistic egocentric -- relies on camera near-clip + aim).

    To get a body-hidden FPV without breaking the TPV, render the two products in SEPARATE
    passes (FPV pass: call this with ``visible=False, allow_global_hide=True``; TPV pass:
    body shown). A single shared-stage run cannot have it both ways.
    """
    from pxr import UsdGeom

    body = _find_body_mesh(rig)
    if body is None:
        LOG.warning(
            "set_body_visible_for_camera: no Body_Mesh found; cannot change body visibility"
        )
        return

    imageable = UsdGeom.Imageable(body)
    if visible:
        imageable.MakeVisible()
        LOG.info(
            "set_body_visible_for_camera(%s): body VISIBLE on %s",
            cam_path,
            body.GetPath(),
        )
        return

    if not allow_global_hide:
        LOG.warning(
            "set_body_visible_for_camera(%s, visible=False) REFUSED: visibility is GLOBAL in "
            "5.1 (no per-render-product scoping), so hiding would also blank the TPV "
            "verification video. Body left VISIBLE. Pass allow_global_hide=True ONLY in a "
            "dedicated FPV-only render pass.",
            cam_path,
        )
        return

    # Explicit opt-in: caller accepts that this hides the body in ALL render products.
    imageable.MakeInvisible()
    body.SetCustomDataByKey("fpv:hidden_globally_for", cam_path)
    LOG.warning(
        "set_body_visible_for_camera(%s, visible=False, allow_global_hide=True): body hidden "
        "GLOBALLY on %s -- the TPV product (if any) will NOT show the human this pass.",
        cam_path,
        body.GetPath(),
    )


def _find_body_mesh(rig: BipedRig) -> "Usd.Prim | None":
    """Find the single skinned body Mesh under the Biped (largest Mesh by bbox)."""
    from pxr import Gf, Usd, UsdGeom

    best = None
    best_diag = -1.0
    for prim in Usd.PrimRange(rig.skel_prim):
        if prim.IsA(UsdGeom.Mesh):
            try:
                rng = (
                    UsdGeom.BBoxCache(
                        Usd.TimeCode.Default(),
                        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                    )
                    .ComputeWorldBound(prim)
                    .ComputeAlignedRange()
                )
                diag = (Gf.Vec3d(rng.GetMax()) - Gf.Vec3d(rng.GetMin())).GetLength()
            except Exception:  # noqa: BLE001
                diag = 0.0
            if diag > best_diag:
                best_diag = diag
                best = prim
    return best


def make_proxy_forearm(stage: "Usd.Stage", side: str) -> "Usd.Prim":
    """A skin-colored capsule /World/ProxyForearm_<side> visible ONLY in the FPV product.

    Gives an in-frame hand/forearm without the chest-blob. Driven each frame from
    ``wrist_world_xform`` back toward an elbow estimate by ``drive_proxy_forearm``.
    """
    from pxr import Gf, UsdGeom, Vt

    side = side.upper()
    path = f"/World/ProxyForearm_{side}"
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateRadiusAttr(0.045)
    capsule.CreateHeightAttr(float(FOREARM_LEN_M))
    capsule.CreateAxisAttr(UsdGeom.Tokens.z)
    # Skin tone via displayColor (cheap; no material network needed for a proxy).
    capsule.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.85, 0.67, 0.55)]))
    set_world_pose(capsule.GetPrim(), (0.0, 0.0, -10.0))  # park off-screen until driven
    LOG.info("Proxy forearm %s created at %s", side, path)
    return capsule.GetPrim()


def drive_proxy_forearm(prim: "Usd.Prim", rig: BipedRig, side: str, frame: int) -> None:
    """Orient the capsule from an elbow estimate to the wrist position each frame.

    Elbow estimate = wrist pulled back toward the shoulder by FOREARM_LEN; the capsule is
    placed at the midpoint and rotated so its +Z axis points elbow->wrist.
    """
    from pxr import Gf

    side = side.upper()
    wrist = Gf.Vec3d(wrist_world_xform(rig, side, frame).ExtractTranslation())

    # Shoulder world position from the analytic root, to bias the elbow estimate inward.
    root = root_xform(rig, frame, None, None)  # type: ignore[arg-type]
    sign = 1.0 if side == "L" else -1.0
    shoulder_local = Gf.Vec3d(0.05, sign * SHOULDER_HALF_WIDTH_M, SHOULDER_HEIGHT_M)
    shoulder = root.Transform(shoulder_local)

    to_shoulder = shoulder - wrist
    if to_shoulder.GetLength() < 1e-6:
        to_shoulder = Gf.Vec3d(0.0, 0.0, 1.0)
    elbow = wrist + to_shoulder.GetNormalized() * float(FOREARM_LEN_M)

    midpoint = (wrist + elbow) * 0.5
    axis_dir = (wrist - elbow).GetNormalized()  # capsule +Z should point toward wrist

    # Rotation that takes +Z onto axis_dir.
    z = Gf.Vec3d(0.0, 0.0, 1.0)
    rot = Gf.Rotation(z, axis_dir)
    quat = rot.GetQuat()
    img = quat.GetImaginary()
    set_world_pose(
        prim,
        (float(midpoint[0]), float(midpoint[1]), float(midpoint[2])),
        (float(quat.GetReal()), float(img[0]), float(img[1]), float(img[2])),
    )


# --------------------------------------------------------------------------------------
# Small local helpers.
# --------------------------------------------------------------------------------------
def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    """(w,x,y,z) quaternion for a yaw about +Z."""
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _prepend_x_rotate(prim: "Usd.Prim", deg: float) -> None:
    """Prepend a rotateX op ahead of the existing xform ops (up-axis correction)."""
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(prim)
    ops = list(xform.GetOrderedXformOps())
    rx = xform.AddRotateXOp(opSuffix="upaxis")
    rx.Set(float(deg))
    xform.SetXformOpOrder([rx] + ops)

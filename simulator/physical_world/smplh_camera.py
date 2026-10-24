# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Thin head-mounted FPV camera for the SMPL+H dynamic demo (zero-roll, object-weighted aim).

The static side-grip framing of the FPV is: eye = J[HEAD_J] + [0,-0.12,0.05] (in
front of the face along world -Y, OUTSIDE the skull), aim = 0.25*wrist + 0.75*obj_ctr, up=+Z,
focal=16 (wide egocentric look). We reuse that verbatim for the idle/reach/grasp/place regimes,
and for the walk regime aim forward (along body-forward) with a gentle downward tilt so the
chest-carried cup sits in the lower-center.

Camera authoring per frame is two steps: SetLookAt(eye, tgt, +Z), then write the camera's
local-to-world inverse onto its MakeMatrixXform op. We do NOT reuse the Biped camera driver (it is
Biped-coupled: it reads head_world_xform + rig._root_yaw_deg). SetLookAt with up=+Z keeps the
horizon level (zero roll).

pxr is imported lazily so importing this module before launch_app() never pulls in Kit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd  # noqa: F401

# Eye offset from the head joint: in front of the face along world -Y, a touch up, OUTSIDE the skull.
# This is the offset used for an upright object such as the cup.
EYE_OFFSET = np.array([0.0, -0.12, 0.05])
FOCAL_MM = 16.0
H_APERTURE = 36.0

# SMPL+H joint layout (same as ``smplh_fpv``): head joint 15, wrist joint 21. Hardcoded as
# documented constants (not magic) to keep this module import-cheap (no module-scope torch import).
H_HEAD = 15
H_WRIST = 21


def make_fpv_camera(
    stage: "Usd.Stage",
    path: str,
    width: int,
    height: int,
    near_clip: float = 0.003,
) -> object:
    """Define the head-mounted FPV camera (focal 16, vertical aperture from the real aspect).

    Returns the camera's MakeMatrixXform op so drive_fpv can write the world matrix each frame.
    ``near_clip`` defaults to the historical 3 mm; a rigid head mount raises it past the
    measured skin clearance so the wearer's own face can never be drawn.
    """
    from pxr import Gf, UsdGeom

    aspect = float(height) / float(width) if width else 760.0 / 960.0
    cam = UsdGeom.Camera.Define(stage, path)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(float(near_clip), 100.0))
    cam.CreateHorizontalApertureAttr(H_APERTURE)
    cam.CreateVerticalApertureAttr(H_APERTURE * aspect)
    cam.CreateFocalLengthAttr(FOCAL_MM)
    cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    cop = UsdGeom.Xformable(cam).MakeMatrixXform()
    hfov = 2.0 * math.degrees(math.atan(0.5 * H_APERTURE / FOCAL_MM))
    print(
        f"[fpv_cam] {path}: focal={FOCAL_MM}mm aperture={H_APERTURE} aspect={aspect:.4f} HFOV~{hfov:.1f}deg"
    )
    return cop


def make_tpv_camera(stage: "Usd.Stage", path: str, width: int, height: int) -> object:
    """Define an external 3/4 verification camera (wider 24mm lens) for the TPV product.

    Used to unambiguously check the whole-body action (cup on the desk, open-hand reach, side-grip,
    lift) that the egocentric FPV can't show. Returns its MakeMatrixXform op."""
    from pxr import Gf, UsdGeom

    aspect = float(height) / float(width) if width else 760.0 / 960.0
    cam = UsdGeom.Camera.Define(stage, path)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 500.0))
    cam.CreateHorizontalApertureAttr(H_APERTURE)
    cam.CreateVerticalApertureAttr(H_APERTURE * aspect)
    cam.CreateFocalLengthAttr(24.0)  # wider than the FPV so the body + desk both frame
    cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    return UsdGeom.Xformable(cam).MakeMatrixXform()


def drive_tpv(cop: object, eye, target) -> None:
    """Author a static external camera looking at the action center (up=+Z, zero roll)."""
    from pxr import Gf

    v = Gf.Matrix4d(1.0)
    v.SetLookAt(
        Gf.Vec3d(*[float(x) for x in eye]),
        Gf.Vec3d(*[float(x) for x in target]),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    cop.Set(v.GetInverse())


def _walk_bob(frame: int, amp: float = 0.012, hz: float = 2.0) -> np.ndarray:
    """Small vertical + lateral head bob for the walk regime (applied to the EYE, not the body)."""
    t = float(frame) / 12.0
    vert = math.sin(2.0 * math.pi * hz * t) * amp
    lat = math.sin(2.0 * math.pi * (hz * 0.5) * t) * (amp * 0.7)
    return np.array([lat, 0.0, vert])


def fpv_eye_and_aim(
    J: np.ndarray,
    mode: str,
    obj_ctr: np.ndarray,
    frame: int = 0,
    root_yaw_deg: float = 0.0,
    eye_offset_world: np.ndarray | tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (eye, target) for the FPV this frame.

    idle/reach/grasp/place: eye in front of the head; aim = 0.25*wrist + 0.75*obj_ctr (peer at the
    cup) -- the proven static framing. walk: aim forward along body-forward with a gentle down-tilt
    so the chest-held cup sits lower-center, plus a small eye bob.
    """
    head_w = J[H_HEAD]
    wrist_w = J[H_WRIST]
    eye_offset = (
        np.zeros(3, dtype=np.float64)
        if eye_offset_world is None
        else np.asarray(eye_offset_world, dtype=np.float64)
    )
    if eye_offset.shape != (3,) or not bool(np.all(np.isfinite(eye_offset))):
        raise ValueError("eye_offset_world must contain three finite numbers")
    eye = head_w + EYE_OFFSET + eye_offset
    if mode == "walk":
        eye = eye + _walk_bob(frame)
        a = math.radians(root_yaw_deg)
        # Native forward is -Y (yaw=0); rotate it by the root yaw about +Z.
        fwd = np.array([math.sin(a) * 1.0, -math.cos(a), 0.0])  # Rz(yaw) @ [0,-1,0]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        # Gentle downward tilt so the carried cup is in the lower-center of the frame.
        aim_dir = fwd + np.array([0.0, 0.0, -0.28])
        aim_dir = aim_dir / (np.linalg.norm(aim_dir) + 1e-9)
        tgt = eye + aim_dir * 0.6
        return eye, tgt
    fpv_aim = 0.25 * wrist_w + 0.75 * np.asarray(obj_ctr)
    look = fpv_aim - eye
    look = look / (np.linalg.norm(look) + 1e-9)
    tgt = eye + look * 0.6
    return eye, tgt


def default_policy_fpv_eye(
    J: np.ndarray,
    mode: str,
    *,
    root_yaw_deg: float,
    eye_offset_world: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Return the default-renderer eye for task-authored camera-offset frames."""
    import egocentric_camera

    yaw = math.radians(float(root_yaw_deg))
    body_forward = (math.sin(yaw), -math.cos(yaw), 0.0)
    camera_mode = "walk" if mode == "walk" else "manip"
    return np.asarray(
        egocentric_camera.default_policy_eye(
            J[H_HEAD],
            body_forward,
            camera_mode,
            eye_offset_world=eye_offset_world,
        ),
        dtype=np.float64,
    )


def drive_fpv(
    cop: object,
    J: np.ndarray,
    mode: str,
    obj_ctr: np.ndarray,
    frame: int = 0,
    root_yaw_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Author the FPV camera world transform for this frame via SetLookAt(eye, tgt, +Z) (zero roll).

    Returns (eye, tgt) for logging / frustum checks.
    """
    from pxr import Gf

    eye, tgt = fpv_eye_and_aim(J, mode, obj_ctr, frame, root_yaw_deg)
    v = Gf.Matrix4d(1.0)
    v.SetLookAt(
        Gf.Vec3d(*[float(x) for x in eye]),
        Gf.Vec3d(*[float(x) for x in tgt]),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    cop.Set(v.GetInverse())
    return eye, tgt


def drive_fpv_gaze(
    cop: object,
    head_pos: np.ndarray,
    gaze_fwd: np.ndarray,
    obj_ctr: np.ndarray,
    eye_push: float = 0.22,
    aim_blend: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Head-GAZE FPV for retargeted mocap (GRAB): eye in front of the face along the head's own
    forward axis, aim blended between the gaze point and the object.

    The fixed world-``-Y`` eye offset (``drive_fpv``) only works for the upright side-grip stand; a
    retargeted body PITCHES its head down to look at the object, so a fixed offset sinks the eye into
    the skull (the camera then renders the inside of the body -> black frames). Here the eye is pushed
    out along the head's actual gaze direction (``gaze_fwd`` = the head's local +Z in world, supplied
    by the caller which has the SMPL+H FK), clearing the mesh at every head pitch. ``eye_push`` (0.22 m
    verified to keep the eye >=~0.12 m outside the body mesh across the whole clip) sets the standoff;
    the aim is ``aim_blend`` toward the object + (1-aim_blend) along the gaze, so the object stays
    framed while the look direction tracks where the person is actually looking. up=+Z (zero roll).
    Returns (eye, tgt).
    """
    from pxr import Gf

    g = np.asarray(gaze_fwd, dtype=np.float64)
    g = g / (np.linalg.norm(g) + 1e-9)
    eye = np.asarray(head_pos, dtype=np.float64) + g * float(eye_push)
    to_obj = np.asarray(obj_ctr, dtype=np.float64) - eye
    to_obj = to_obj / (np.linalg.norm(to_obj) + 1e-9)
    look = aim_blend * to_obj + (1.0 - aim_blend) * g
    look = look / (np.linalg.norm(look) + 1e-9)
    tgt = eye + look * 0.6
    v = Gf.Matrix4d(1.0)
    v.SetLookAt(
        Gf.Vec3d(*[float(x) for x in eye]),
        Gf.Vec3d(*[float(x) for x in tgt]),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    cop.Set(v.GetInverse())
    return eye, tgt


def drive_fpv_rigid(cop: object, eye, forward, up) -> None:
    """Author a RIGIDLY MOUNTED FPV camera: the head link's pose, written verbatim.

    Unlike ``drive_fpv_gaze`` there is no push-out, no aim blend and no world-up levelling --
    ``eye``/``forward``/``up`` are produced by ``head_mount.mount`` from the head link and are
    written as-is. That is IsaacLab's ``CameraCfg`` contract (sensor pose == body pose composed
    with a fixed offset, and nothing else may move it), evaluated per frame because our wearer
    is a point-baked mesh with no USD parent to attach a camera prim to.
    """
    from pxr import Gf

    origin = [float(value) for value in eye]
    target = [float(o) + float(f) for o, f in zip(origin, forward)]
    view = Gf.Matrix4d(1.0)
    view.SetLookAt(
        Gf.Vec3d(*origin),
        Gf.Vec3d(*target),
        Gf.Vec3d(*[float(value) for value in up]),
    )
    cop.Set(view.GetInverse())


def point_in_fpv_frustum(
    eye: np.ndarray,
    tgt: np.ndarray,
    world_pt: np.ndarray,
    width: int,
    height: int,
    focal_mm: float = FOCAL_MM,
) -> tuple[bool, float, float]:
    """Project world_pt into the FPV frustum; return (inside, ndc_x, ndc_y).

    Builds the view basis from (eye, tgt, +Z) -- the same basis SetLookAt uses (camera looks down
    -Z, +Y up) -- and a perspective from focal/aperture. inside = in front AND |ndc|<=1 on both axes.
    Used by --print-geom and the optional in-frame warning.
    """
    f = np.asarray(tgt) - np.asarray(eye)
    f = f / (np.linalg.norm(f) + 1e-9)  # forward (camera looks down -Z, so -Z == f)
    up_world = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(f, up_world))) > 0.999:
        up_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(f, up_world)
    right = right / (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, f)
    p = np.asarray(world_pt) - np.asarray(eye)
    x_cam = float(np.dot(p, right))
    y_cam = float(np.dot(p, true_up))
    z_cam = float(np.dot(p, -f))  # camera -Z is forward -> point in front has z_cam<0
    if z_cam >= 0.0:
        return False, float("nan"), float("nan")
    aspect = float(height) / float(width) if width else 760.0 / 960.0
    tan_h = (H_APERTURE * 0.5) / float(focal_mm)
    tan_v = (H_APERTURE * aspect * 0.5) / float(focal_mm)
    ndc_x = (x_cam / -z_cam) / tan_h
    ndc_y = (y_cam / -z_cam) / tan_v
    inside = abs(ndc_x) <= 1.0 and abs(ndc_y) <= 1.0
    return inside, ndc_x, ndc_y

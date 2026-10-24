"""Reusable SMPL+H motion, grasp, carry, and realization primitives."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRY_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, os.path.join(_TRY_DIR, "human_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# torch-first (pre-launch): the math core + gait + grasp all import smplh_fpv (torch) at module scope.
import accad_retarget  # noqa: E402
import gait_player as gait  # noqa: E402
import grasp_select  # noqa: E402
import smplh_anim_core as core  # noqa: E402
from smplh_anim_core import (  # noqa: E402
    H,
    L_COLLAR,
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    R_COLLAR,
    R_ELBOW,
    R_SHOULDER,
    WRIST_J,
)

# ---- Scene layout (meters, Z-up, native forward = -Y). Two tables front/back; walk between them. ----
OBJ_MESH = os.path.join(
    os.environ.get("INTERACTIVEWORLD_HUMAN_ASSET_ROOT", ""),
    "atlas",
    "objects",
    "grabcup_m.obj",
)
START_XY = (
    0.0,
    1.05,
)  # in the corridor between the two desks; clear of DeskB's front edge (~1.28)
TABLE_A_XY = (
    0.0,
    -1.68,
)  # pickup desk CENTER (rotated 90° in run_render -> wide X, 0.8m deep Y, front edge ~-1.28)
OBJ_A_XY = (0.0, -1.36)  # cup rests here ON DeskA (just behind its front edge)
STAND_A_XY = (
    0.0,
    -0.90,
)  # human stops IN FRONT of DeskA (~0.38 m clear) and leans over to reach
TABLE_B_XY = (0.0, 1.68)  # drop desk CENTER (rotated 90°; front edge ~+1.28)
OBJ_B_XY = (0.0, 1.36)  # cup placed here ON DeskB
STAND_B_XY = (0.0, 0.90)  # human stops IN FRONT of DeskB and leans over to place
YAW_A, YAW_B = 0.0, 180.0
REACH_LEAN_DEG = 40.0  # forward spine bend to reach a ~0.75 m table (feet stay planted; straighten to lift)
WALK_SPEED = 1.1
FPS = 12

_ARM_JOINTS = (
    R_COLLAR,
    R_SHOULDER,
    R_ELBOW,
    WRIST_J,
    L_COLLAR,
    L_SHOULDER,
    L_ELBOW,
    L_WRIST,
)
_SPINE_J = (3, 6, 9)  # SMPL spine1/2/3; forward flexion about local X
_SPINE_W = (0.5, 0.3, 0.2)  # bend distribution (most at the lower spine)


@dataclass
class Frame:
    """One precomputed motion frame (fully offline-computable)."""

    pose156: np.ndarray
    xy: tuple[float, float]
    yaw: float
    z: float
    arm_override: dict | None = None  # override the right arm on a walk pose (to carry)
    rhp: np.ndarray | None = None  # override the 45-D fingers
    fpv_mode: str = "look"  # "walk" (aim ahead) | "look" (aim at object)
    obj: tuple = (
        "hide",
        None,
    )  # ("rest", xyz|T_4x4) | ("carry", otw) | ("pose", T_4x4) | ("hide", None)
    # World-FIXED contact point (xyz) the hand is ON this frame (grasp seat / handle). Set ONLY by the
    # producers on frames where the hand is pinned to a world target -> the render nudge treats the frame
    # as FIXED contact (no root push; spine-lean relief instead). None (default) -> the generic
    # carry/hide/rest rule (so unflagged clips, e.g. AMASS, stay free). Zero added runtime cost.
    contact_anchor: "np.ndarray | None" = None
    # P3 (during-render object dynamics): True ONLY on the released-rest frames of a FREE-DYNAMIC
    # producer, telling realize/nudge that PhysX -- not a literal pose -- owns this object. Defaults
    # False so every existing frame (drawer handle rests, carry, hide, literal-obj rests) is identical;
    # a free-dynamic scene is opted in via scene.free_dynamic_prims + this flag on its release frames.
    free_dynamic: bool = False
    # Egocentric aim override (world xyz): when set, the FPV camera aims at THIS point (the worked
    # target / gaze focus) instead of the carried-object centroid. Read by episode_runtime.focus_of via
    # getattr; None (default) -> every existing frame is byte-identical (generic obj-based aim).
    gaze: "np.ndarray | None" = None
    # World-space camera-eye offset for a task-authored head lean. The body, root, and contact arm remain
    # unchanged; simulator camera clearance and renderer replay both consume this exact per-frame value.
    camera_eye_offset: "np.ndarray | None" = None
    seg: str = ""
    # World prim path of the object this frame's obj-state ("rest"/"carry") refers to, for MULTI-graspable
    # scenes (the render sets rig._obj_path to it before set_object_world so the right prim is written).
    # None (default) -> the render writes rig._obj_path (/World/Obj), so single-object scenes are unchanged.
    obj_path: "str | None" = None
    # Identity of the articulated controller this frame interacts with. Open/close producers set it on
    # every frame so replay never guesses from spatial proximity to a handle; front-insert frames use it
    # to resolve task-local camera policy from the receiving container.
    openable_key: "str | None" = None
    # CONTENTS that RIDE this frame's carried object as one rigid body (a bowl carried with its food): a tuple
    # of ``(content_prim_path, rel_4x4)`` where rel = inv(container_world) @ content_world captured at grab.
    # The render writes each content at ``obj_T @ rel`` (obj_T = the container's world this frame). Empty
    # (default) -> single object, byte-identical.
    carried: tuple = ()
    # Collider roots intentionally contacted by this frame (for example a rack being pushed). The
    # compliance oracle still checks every other scene collider; only the manipulated body is excluded.
    collision_exclude_paths: tuple[str, ...] = ()
    # Adaptive "lean-in" zoom for a LookAt (perception aid + demo legibility; NOT scored). zoom = per-frame ease
    # weight 0..1 (None -> no zoom); zoom_span = the target's OWN bbox vertical extent (m) that drives the FOV.
    # Both default None -> every existing frame is byte-identical; the FPV focal is only narrowed when set (gated
    # in produce_look_at on scene.look_zoom).
    zoom: "float | None" = None
    zoom_span: "float | None" = None
    # Snapshot of toggleable devices that should LOOK active on this frame. ``None`` means the frame
    # predates visual-state support; an empty tuple explicitly means every registered indicator is off.
    # The semantic executor stamps every generated frame, and marks the exact press frame where a
    # TURN_ON/TURN_OFF transition becomes visible. The renderer reads this without consulting final
    # scene state, so frame zero cannot accidentally show the episode's final state.
    running_devices: "tuple[str, ...] | None" = None
    # LookAt at a FIXED sign target (a container's hazard placard): trust the recorded sign point and do NOT
    # let the FPV camera's anti-backward-gaze / doorway redirect override it (the human has turned to face the
    # sign). Set ONLY by produce_look_at on a label_pos look-at under scene.look_zoom; default False -> every
    # existing frame is byte-identical (the redirects run as before).
    gaze_locked: bool = False
    # True only when the semantic LookAt planner has already measured this exact label view as readable.
    # The replay adapter may then trust that precise check instead of rejecting zoom with its conservative
    # axis-aligned scene-box ray test.
    focus_visibility_verified: bool = False


@dataclass
class Motion:
    frames: list[Frame] = field(default_factory=list)
    grasp_info: dict = field(default_factory=dict)


def _arm_from_pose(pose: np.ndarray) -> dict:
    """Extract the right+left arm joint dict from a full 156-D pose (for lerp / carry override)."""
    return {
        j: list(np.asarray(pose[j * 3 : j * 3 + 3], dtype=np.float64))
        for j in _ARM_JOINTS
    }


def _add_lean(pose: np.ndarray, lean_rad: float) -> np.ndarray:
    """Bend the torso forward by ``lean_rad`` (distributed over the spine), feet planted. Returns a copy."""
    p = np.asarray(pose, dtype=np.float32).copy()
    for j, w in zip(_SPINE_J, _SPINE_W):
        p[j * 3] += float(lean_rad) * w  # local X = forward flexion
    return p


def object_rest_position(value) -> np.ndarray:
    """Extract xyz from either a legacy rest point or a full 4x4 rest transform."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        return array[:3, 3].copy()
    flat = array.reshape(-1)
    if flat.size < 3:
        raise ValueError("object rest value must contain xyz or a 4x4 transform")
    return flat[:3].copy()


def _obj_rest_T(value) -> np.ndarray:
    """Full resting transform, accepting legacy xyz and physics-derived 6-DoF poses."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        return array.copy()
    T = np.eye(4)
    T[:3, 3] = object_rest_position(value)
    return T


_OBJTRACE_N = 0


def _objtrace(fr, kind, obj_T) -> None:
    """DEBUG (env STEP3R_OBJTRACE=1): log the driven object's world xyz per frame, to measure any real
    per-frame object oscillation during carry/grab. No-op unless the env var is set (byte-identical).
    """
    import os

    if not os.environ.get("STEP3R_OBJTRACE"):
        return
    global _OBJTRACE_N
    _OBJTRACE_N += 1
    print(
        f"[objtrace] n={_OBJTRACE_N} seg={getattr(fr, 'seg', None)} kind={kind} "
        f"xyz=({float(obj_T[0, 3]):.4f},{float(obj_T[1, 3]):.4f},{float(obj_T[2, 3]):.4f})",
        flush=True,
    )


def _pose_place_cache_key(rig, fr: Frame) -> tuple:
    def array_key(value) -> tuple | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        return (array.shape, array.tobytes())

    arm_override = tuple(
        (int(joint), array_key(value))
        for joint, value in sorted((fr.arm_override or {}).items())
    )
    root_offset = np.array([fr.xy[0], fr.xy[1], fr.z])
    return (
        id(rig),
        id(gait.pose_place),
        id(gait.H.fwd),
        array_key(fr.pose156),
        (root_offset.dtype.str, root_offset.shape, root_offset.tobytes()),
        float(fr.yaw).hex(),
        arm_override,
        array_key(fr.rhp),
    )


def realize(
    rig, fr: Frame, scene_boxes=None, nudge_state=None, pose_place_cache=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """FK + place a frame -> (V, J, obj_T). Pure; used by both the dry storyboard and the Isaac render.

    ``scene_boxes`` (a list of solid AABBs (name, mn, mx)) + ``nudge_state`` (a mutable dict) enable the
    generic render-time collision nudge (``nudge_frame``): when passed, the posed body/object are shifted
    out of any solid furniture it egregiously penetrates. Both default None -> byte-identical legacy
    behaviour (the non-render callers pass nothing).

    ``pose_place_cache`` may be shared only within one stable FK-backend scope. It memoizes the complete
    pure body-FK input; collision nudging and every other stateful per-frame operation still run on fresh
    arrays.
    """
    if pose_place_cache is None:
        V, J = gait.pose_place(
            fr.pose156,
            fr.xy,
            fr.yaw,
            fr.z,
            arm_override=fr.arm_override,
            rhp=fr.rhp,
        )
    else:
        cache_key = _pose_place_cache_key(rig, fr)
        cached = pose_place_cache.get(cache_key)
        if cached is None:
            base_vertices, base_joints = gait.pose_place(
                fr.pose156,
                fr.xy,
                fr.yaw,
                fr.z,
                arm_override=fr.arm_override,
                rhp=fr.rhp,
            )
            pose_place_cache[cache_key] = (base_vertices.copy(), base_joints.copy())
        else:
            base_vertices, base_joints = cached
        V, J = base_vertices.copy(), base_joints.copy()
    kind, val = fr.obj
    if getattr(fr, "free_dynamic", False):
        # A->a: PhysX owns this object's pose (released, de-penetrating + settling on camera). Emit
        # obj_T=None so nothing writes a literal pose to it and _apply_nudge (line ~267) leaves it be;
        # the moving kinematic human proxy pushes it via PhysX contact instead.
        obj_T = None
    elif kind == "rest":
        obj_T = _obj_rest_T(val)
    elif kind == "carry":
        obj_T = core.seat_object_nosnap(rig, J, val)
    elif kind == "pose":
        # A part welded at an explicit full 6-DOF world transform (assembly [Attach]). Kinematic + not a
        # "carry", so _apply_nudge leaves it fixed and action_check's resettle (keys off "rest") skips it.
        obj_T = np.asarray(val, dtype=np.float64)
    else:
        obj_T = None
    if scene_boxes is not None:
        V, J, obj_T = nudge_frame(rig, fr, V, J, obj_T, scene_boxes, nudge_state)
    if obj_T is not None:
        _objtrace(fr, kind, obj_T)
    return V, J, obj_T


# =====================================================================================================
# Generic render-time collision nudge (P0): a post-pose filter that shifts the WHOLE posed body (+ any
# held/rested object) horizontally out of a solid-furniture box it egregiously penetrates. Runs only at
# the render entry points; smoothed + rate-/mag-capped so the sidestep eases in/out over a few frames.
# =====================================================================================================
_NUDGE_EGREGIOUS_M = (
    0.03  # penetration below this depth is left alone (a graze -> no push)
)
_NUDGE_MARGIN_M = 0.01  # push slightly past the face so it clears
_NUDGE_BODY_R = (
    0.10  # inflate each solid AABB by this (approx body radius) before testing
)
_NUDGE_ALPHA = 0.2  # exponential smoothing toward the raw target displacement
_NUDGE_RATE_M = 0.012  # per-frame cap on how fast the displacement may change
_NUDGE_MAG_M = 0.08  # absolute cap on the total displacement
_NUDGE_CAMERA_R_M = 0.03
_NUDGE_R_CONTACT = (
    0.13  # a resting-object frame with the hand within this of the target is a contact
)
# P2a spine-lean relief: on a FIXED-contact frame (contact_anchor set) whose body CORE is egregiously
# inside a box, a root translate would drag the gripping hand off, so instead we lean the SPINE back to
# pitch the torso out while the arm keeps reaching. Lean angle ~ depth (capped small), smoothed via
# nudge_state like the translate. Fires only on the rare fixed-contact-AND-torso-penetrating frame.
_NUDGE_LEAN_K = (
    30.0  # rad-per-metre depth -> lean; depth 0.03..0.33 -> ~0.9..10deg (clipped)
)
_NUDGE_LEAN_MAX = math.radians(10.0)  # absolute cap on the spine-back lean
_NUDGE_LEAN_RATE = math.radians(2.0)  # per-frame cap on how fast the lean may change
# Body-CORE proxy joints (SMPL+H 0-15): pelvis/hips/spine/knees/ankles/feet/neck/collars/head. The
# reaching ARM + HAND (17/19/21 + MANO) are DELIBERATELY excluded: they are meant to touch the object/
# furniture during a task (grasp/carry/open), so including them false-triggered the nudge on every
# grasp-near-furniture; arm-vs-furniture clearance is the proactive waypoint reach's job, not the nudge's.
_NUDGE_J = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# ``state`` slot holding the packed+inflated box arrays (see _packed_nudge_boxes).
_NUDGE_PACK_KEY = "_boxpack"


def _repack_boxes(boxes, cache):
    """Pack ``boxes`` inflated by ``_NUDGE_BODY_R``, reusing the longest identity-matching PREFIX of the
    previously packed set. ``episode_runtime._nudge_boxes_for`` rebuilds the list every frame on a scene with
    openables, but it passes the static rows' corner ARRAYS straight through and only appends each live
    drawer/door extent, so the prefix covers ~everything and only the changed tail is re-packed.
    """
    import semantic_executor as se  # lazy: se imports wpp at module scope (avoid the import cycle)

    if cache is None:
        return se.pack_boxes(boxes, _NUDGE_BODY_R)
    prev, (pnames, pmn, pmx) = cache
    k, lim = 0, min(len(prev), len(boxes))
    while (
        k < lim
        and prev[k][1] is boxes[k][1]
        and prev[k][2] is boxes[k][2]
        and prev[k][0] == boxes[k][0]
    ):
        k += 1
    if k == len(prev) == len(boxes):
        return pnames, pmn, pmx
    tnames, tmn, tmx = se.pack_boxes(boxes[k:], _NUDGE_BODY_R)
    return (
        pnames[:k] + tnames,
        np.concatenate([pmn[:k], tmn]),
        np.concatenate([pmx[:k], tmx]),
    )


def _packed_nudge_boxes(boxes, state):
    """``(names, MN, MX)`` for ``boxes`` inflated by ``_NUDGE_BODY_R``, packed once and cached in ``state``.

    Hoists BOTH per-frame costs out of the nudge: the whole-scene inflate comprehension and the pack. The
    cache validates by object IDENTITY (never by value), and it keeps the previous list alive, so a corner
    array's address can never be recycled under it."""
    cache = state.get(_NUDGE_PACK_KEY)
    if cache is not None and cache[0] is boxes:
        return cache[1]
    packed = _repack_boxes(boxes, cache)
    state[_NUDGE_PACK_KEY] = (boxes, packed)
    return packed


def _escape_for_frame(se, P, boxes, state):
    """Worst (depth, box name, outward face normal) for the body proxy ``P`` against the solid boxes.

    Default: inflate + pack the boxes ONCE per episode and search them with array ops. The legacy path
    inflated all ~4000 boxes of a BEHAVIOR scene into fresh tuples and looped over them in Python on EVERY
    motion frame -- together ~87% of the compliance FK pass. ``ISAAC_NUDGE_VEC=0`` restores that loop; the
    two are bit-identical, so it exists only to let an A/B flip them in-process."""
    if os.environ.get("ISAAC_NUDGE_VEC") == "0":
        inflated = [
            (name, np.asarray(mn) - _NUDGE_BODY_R, np.asarray(mx) + _NUDGE_BODY_R)
            for name, mn, mx in boxes
        ]
        return se._escape_from_boxes(P, inflated)
    return se.escape_from_boxes_packed(P, *_packed_nudge_boxes(boxes, state))


def _camera_clear_after_nudge(fr: Frame, J, displacement, state) -> bool:
    """Whether a proposed body nudge keeps the exact FPV eye outside scene geometry."""
    import smplh_camera

    shifted = np.asarray(J, dtype=np.float64) + np.asarray(
        displacement, dtype=np.float64
    )
    eye_offset = getattr(fr, "camera_eye_offset", None)
    if eye_offset is None:
        eye, _aim = smplh_camera.fpv_eye_and_aim(
            shifted,
            getattr(fr, "fpv_mode", "look"),
            np.zeros(3, dtype=np.float64),
            frame=int(state.get("frame_index", 0)),
            root_yaw_deg=float(getattr(fr, "yaw", 0.0)),
        )
    else:
        eye = smplh_camera.default_policy_fpv_eye(
            shifted,
            getattr(fr, "fpv_mode", "look"),
            root_yaw_deg=float(getattr(fr, "yaw", 0.0)),
            eye_offset_world=eye_offset,
        )
    grid = state.get("nav_grid")
    if (
        grid is not None
        and getattr(grid, "wall_sdf", None) is not None
        and float(grid.wall_sdf_at(float(eye[0]), float(eye[1]))) < _NUDGE_CAMERA_R_M
    ):
        return False
    radius_sq = _NUDGE_CAMERA_R_M * _NUDGE_CAMERA_R_M
    for _name, mn, mx in state.get("camera_boxes", ()):
        nearest = np.maximum(np.asarray(mn), np.minimum(eye, np.asarray(mx)))
        if float(np.dot(eye - nearest, eye - nearest)) < radius_sq:
            return False
    return True


def _camera_safe_nudge(fr: Frame, J, displacement, state):
    """Clip only a nudge-created eye collision, preserving an already-colliding escape."""
    displacement = np.asarray(displacement, dtype=np.float64)
    if _camera_clear_after_nudge(fr, J, displacement, state):
        return displacement
    zero = np.zeros(3, dtype=np.float64)
    if not _camera_clear_after_nudge(fr, J, zero, state):
        return displacement
    lo, hi = 0.0, 1.0
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        if _camera_clear_after_nudge(fr, J, displacement * mid, state):
            lo = mid
        else:
            hi = mid
    return displacement * lo


def nudge_frame(rig, fr: Frame, V, J, obj_T, boxes, state):
    """Shift the posed (V, J, obj_T) out of any solid furniture box the body egregiously penetrates.

    GENERIC (no per-scene knowledge): builds a coarse body proxy from ``_NUDGE_J`` (+ the forearm
    midpoint), inflates each box by a body radius, finds the worst escape face normal, and eases a
    horizontal displacement in (smoothed, rate-/mag-capped) via ``state["d"]``. Skips the push while the
    hand is manipulating the active object (a rest target within reach, or a carry/hide frame that is
    mid-grasp) so a reach/place is never yanked sideways -- but always decays + applies a prior nudge so
    it eases out. Returns (V, J, obj_T) (obj_T copied only if shifted)."""
    import semantic_executor as se  # lazy: se imports wpp at module scope (avoid the import cycle)

    if state is None:
        state = {}
    d_prev = state.get("d", np.zeros(3))

    # Body proxy (already world-frame from realize): the sampled joints + the forearm midpoint.
    P = J[list(_NUDGE_J)]

    # Inflate each solid box by the body radius, then find the worst escape (see _escape_for_frame).
    depth, hit, n_hat = _escape_for_frame(se, P, boxes, state)

    # FIXED-CONTACT RULE. Prefer an EXPLICIT anchor: a producer that pinned the hand to a world target
    # (grasp seat / handle) sets fr.contact_anchor, so this frame is UNAMBIGUOUSLY fixed contact -- never
    # root-nudge it (that would drag the gripping hand off). Absent the flag, fall back to the generic
    # guess (carry/hide free; a "rest" frame with the wrist near the target is a contact). Unflagged clips
    # (e.g. AMASS) hit the generic rule, so they stay free.
    if getattr(fr, "contact_anchor", None) is not None:
        free = False
    elif getattr(fr, "free_dynamic", False):
        # A released free-dynamic rest frame: the object is PhysX-owned (obj_T is None), so the hand is
        # NOT pinned to it -- treat the frame as free so the human still sidesteps around furniture (the
        # body root nudge runs; _apply_nudge no-ops the None object).
        free = True
    else:
        kind = fr.obj[0]
        free = kind in ("carry", "hide")
        if kind == "rest":
            wrist_o = J[WRIST_J]
            free = (
                float(np.linalg.norm(wrist_o[:2] - object_rest_position(fr.obj[1])[:2]))
                >= _NUDGE_R_CONTACT
            )
    if not free:
        # Decay any prior root nudge (a fixed-contact frame is never root-pushed) so it eases out.
        d = d_prev + (np.zeros(3) - d_prev) * _NUDGE_ALPHA
        state["d"] = d
        # P2a: on an EXPLICIT fixed-contact frame whose CORE penetrates egregiously, lean the spine back
        # to pitch the torso out (arm/grip untouched) instead of translating. Decays the lean to 0 when
        # the depth is below threshold. Everything else (guessed contacts) keeps today's skip-only path.
        if getattr(fr, "contact_anchor", None) is not None:
            V, J, obj_T = _spine_relief(rig, fr, V, J, obj_T, depth, hit, state)
        return _apply_nudge(V, J, obj_T, d, fr, hit)

    # Raw target displacement: push out to just past the face, horizontal-only unless a pure top face.
    if depth <= _NUDGE_EGREGIOUS_M or n_hat is None:
        d_raw = np.zeros(3)
    else:
        d_raw = n_hat * (depth - _NUDGE_EGREGIOUS_M + _NUDGE_MARGIN_M)
        if not (
            abs(n_hat[2]) > 0.999
        ):  # keep XY; zero Z unless it is a pure ±Z top face
            d_raw = d_raw.copy()
            d_raw[2] = 0.0

    # Wall anti-penetration backstop (additive, P3): the planner routes the ROOT through free space, but a
    # swaying core joint (arms/hands excluded via _NUDGE_J) can still intrude a WALL, which solid_boxes does
    # NOT cover. Push out along the nav-grid SDF gradient of the WORST-penetrating core joint (doorway-safe:
    # the gradient runs ALONG a passage, not across it). Folds into the SAME eased + rate/mag-capped d.
    _grid = state.get("nav_grid")
    if _grid is not None and getattr(_grid, "wall_sdf", None) is not None:
        _worst = None
        for _p in J[list(_NUDGE_J)]:
            _sdf = _grid.wall_sdf_at(float(_p[0]), float(_p[1]))
            if _sdf < 0.0 and (_worst is None or _sdf < _worst[0]):
                _worst = (_sdf, float(_p[0]), float(_p[1]))
        if _worst is not None:
            _gx, _gy = _grid.wall_grad(_worst[1], _worst[2])
            if _gx != 0.0 or _gy != 0.0:
                d_raw = d_raw + np.array([_gx, _gy, 0.0]) * (-_worst[0])

    # Smooth -> rate-cap the step -> mag-cap the total.
    d = d_prev + (d_raw - d_prev) * _NUDGE_ALPHA
    step = d - d_prev
    step_n = float(np.linalg.norm(step))
    if step_n > _NUDGE_RATE_M:
        d = d_prev + step * (_NUDGE_RATE_M / step_n)
    d_n = float(np.linalg.norm(d))
    if d_n > _NUDGE_MAG_M:
        d = d * (_NUDGE_MAG_M / d_n)
    d = _camera_safe_nudge(fr, J, d, state)
    state["d"] = d
    return _apply_nudge(V, J, obj_T, d, fr, hit)


def _apply_nudge(V, J, obj_T, d, fr: Frame, hit):
    """Add displacement ``d`` to the posed verts/joints (+ a CARRIED object), log a real push, and return.

    Only a CARRIED object (kind=="carry", i.e. held in the hand) follows the body's collision sidestep. A
    RESTING object (kind=="rest") sits on a surface and is NOT attached to the body, so it must stay at its
    true world pose while the body/hand nudge past furniture. Previously ``d`` was added to the object
    unconditionally, which DRAGGED the resting grab TARGET up/down + fore/aft before contact (the nudge
    builds as the human nears the surface, then decays over the contact frames) -- and only the currently
    driven target moved (the other frozen objects are never re-posed). Gating on "carry" fixes that.
    """
    if float(np.linalg.norm(d)) <= 1e-4:
        return V, J, obj_T
    V = V + d
    J = J + d
    if obj_T is not None and fr.obj[0] == "carry":
        obj_T = obj_T.copy()
        obj_T[:3, 3] += d
    print(
        f"[nudge] {fr.seg} obj={fr.obj[0]}: push={float(np.linalg.norm(d)) * 1000:.0f}mm out of {hit}"
    )
    return V, J, obj_T


def _spine_relief(rig, fr: Frame, V, J, obj_T, depth, hit, state):
    """P2a: on a FIXED-contact frame whose body CORE is egregiously inside a box, lean the SPINE back to
    pitch the torso out of the box while the reaching arm/grip stays put -- a root translate would drag
    the gripping hand off the target. Lean ~ depth (capped small), smoothed + rate-capped via
    ``state["lean"]`` (decays to 0 when depth is below threshold). Only the spine joints move (via
    ``_add_lean`` with a NEGATIVE angle = lean back); the shoulder/elbow/wrist + fingers are untouched, so
    the grip is preserved. Re-FKs the leaned pose through the SAME path realize uses (gait.pose_place) and
    re-derives a carry object's seat. Returns the (re-FK'd) (V, J, obj_T) -- unchanged if no lean applies.
    """
    lean_prev = float(state.get("lean", 0.0))
    lean_tgt = (
        float(np.clip(depth * _NUDGE_LEAN_K, 0.0, _NUDGE_LEAN_MAX))
        if depth > _NUDGE_EGREGIOUS_M
        else 0.0
    )
    lean = lean_prev + (lean_tgt - lean_prev) * _NUDGE_ALPHA
    step = lean - lean_prev
    if abs(step) > _NUDGE_LEAN_RATE:
        lean = lean_prev + math.copysign(_NUDGE_LEAN_RATE, step)
    state["lean"] = lean
    if lean <= 1e-4:
        return V, J, obj_T
    # NEGATIVE lean = lean the torso BACK (the reach lean uses +, so - pitches away from the box).
    leaned = _add_lean(fr.pose156.copy(), -lean)
    Vn, Jn = gait.pose_place(
        leaned, fr.xy, fr.yaw, fr.z, arm_override=fr.arm_override, rhp=fr.rhp
    )
    if fr.obj[0] == "carry":
        obj_Tn = core.seat_object_nosnap(rig, Jn, fr.obj[1])
    else:  # "rest" with a contact anchor: the object stays at its declared world pose
        obj_Tn = obj_T
    print(
        f"[nudge] {fr.seg} obj={fr.obj[0]}: spine-lean {math.degrees(lean):.1f}deg "
        f"(depth {depth * 1000:.0f}mm) off {hit}"
    )
    return Vn, Jn, obj_Tn


def _body_forward(yaw_deg: float) -> np.ndarray:
    """World forward for a body at heading yaw (native -Y): Rz(yaw)@[0,-1,0] = [sin,-cos,0]."""
    a = math.radians(yaw_deg)
    return np.array([math.sin(a), -math.cos(a), 0.0])


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


# =====================================================================================================
# Grasp solve (object-general): lean + warp the arm to the resting object + live-wrap the fingers.
# =====================================================================================================
def solve_grasp(
    rig,
    base_pose,
    rz,
    world_off,
    obj_place_world,
    sd_world,
    family_seed=False,
    *,
    object_rest_pose=None,
    planned_pose=None,
    planned_err=None,
    grasp_hint=None,
):
    """Solve reach+grip on a (leaned) base pose. sd_world None -> arm-only reach (dry, fast). Returns dict.

    ``family_seed`` (opt-in): when the live wrap runs, seed it from the per-shape grip prototype
    (grasp_select.classify_shape -> CYL/ROUND/PINCH) so different shapes get different grips. Default
    False -> the flat-open live wrap (existing callers byte-identical)."""
    if (
        sd_world is None
    ):  # dry: fast arm-only reach (verifies the seat lands on the object)
        if planned_pose is not None and planned_err is not None:
            grasp_pose = np.asarray(planned_pose, dtype=np.float32).copy()
            seat_err = float(planned_err)
        else:
            grasp_pose, seat_err = core.warp_arm_to_seat(
                rig, base_pose, rz, world_off, obj_place_world
            )
        wrapped_rhp, near = rig.rhp.copy(), None
    else:  # full: warp + wrist-orient + live SDF finger wrap (+ bounded fine-tune)
        sel = grasp_select.select_and_grasp(
            core,
            rig,
            base_pose,
            rz,
            world_off,
            obj_place_world,
            sd_world,
            core.right_hand_masks(),
            seed_grasp=False,
            seed_wrist=False,
            auto_family=family_seed,
            initial_pose=planned_pose,
            initial_seat_err=planned_err,
            grasp_hint=grasp_hint,
        )
        grasp_pose, wrapped_rhp, seat_err, near = (
            sel["pose"],
            sel["rhp"],
            sel["seat_err"],
            sel["near"],
        )
    grasp_arm = _arm_from_pose(grasp_pose)
    Jg = (rz @ H.fwd_joints(grasp_pose).T).T + world_off
    rest_pose = obj_place_world if object_rest_pose is None else object_rest_pose
    otw = np.linalg.inv(core.wrist_world_T(Jg)) @ _obj_rest_T(rest_pose)
    return {
        "grasp_arm": grasp_arm,
        "rhp": np.asarray(wrapped_rhp, dtype=np.float32),
        "otw": otw,
        "seat_err": float(seat_err),
        "near": near,
    }


def solve_place_arm(rig, grasp_arm, obj_place_world, stand_z):
    """Warp the (already-wrapped) carry arm so the object SEAT lands at the drop point on DeskB.

    Symmetric to the pick reach: this makes the hand DELIVER the cup to obj_place_world, so releasing it
    there leaves it exactly where the hand is (no teleport/vanish). Returns (place_arm, seat_err_m).
    """
    rz, world_off = _grasp_frame(STAND_B_XY, YAW_B, stand_z)
    base = _add_lean(
        core.assemble_pose(rig, grasp_arm, (0.0, 0.0, 0.0), 1.0),
        math.radians(REACH_LEAN_DEG),
    )
    pose, err = core.warp_arm_to_seat(
        rig, base, rz, world_off, np.asarray(obj_place_world, dtype=np.float64)
    )
    return _arm_from_pose(pose), float(err)


# =====================================================================================================
# Choreography -> a flat list of precomputed frames.
# =====================================================================================================
def build_motion(
    rig, g: gait.Gait, stand_z: float, obj_a_z: float, obj_b_z: float, grasp, place_arm
) -> Motion:
    """Sequence walk1 -> lean/grasp -> lift -> turn -> walk2(carry) -> place, as precomputed frames."""
    m = Motion(grasp_info=grasp)
    grasp_arm, rhp, otw = grasp["grasp_arm"], grasp["rhp"], grasp["otw"]
    rig.rhp = rhp  # so assemble_pose(hand_close) reproduces the wrapped grip

    obj_a = (OBJ_A_XY[0], OBJ_A_XY[1], obj_a_z)
    obj_b = (OBJ_B_XY[0], OBJ_B_XY[1], obj_b_z)
    lean = math.radians(REACH_LEAN_DEG)
    arm_rest = getattr(rig, "arm_rest", rig.arm_grasp)

    def manip(arm, hc, lean_frac=0.0):
        return _add_lean(
            core.assemble_pose(rig, arm, (0.0, 0.0, 0.0), hc), lean * lean_frac
        )

    def add(pose, xy, yaw, mode, obj, seg):
        m.frames.append(Frame(pose, xy, yaw, stand_z, fpv_mode=mode, obj=obj, seg=seg))

    # 1) WALK1: START -> STAND_A (empty-handed natural gait), object resting on table A.
    wf1, _ = gait.plan_walk(
        g, START_XY, STAND_A_XY, 0.0, WALK_SPEED, FPS, start_dist=0.0
    )
    for wf in wf1:
        m.frames.append(
            Frame(
                wf.pose156,
                wf.root_xy,
                wf.yaw_deg,
                wf.root_z,
                fpv_mode="walk",
                obj=("rest", obj_a),
                seg="walk1",
            )
        )

    # 2) ARRIVE1: blend the last walk pose -> upright manip stand (arm down, open hand) at STAND_A.
    stand_open = manip(arm_rest, 0.0)
    for i in range(6):
        add(
            gait.blend156(wf1[-1].pose156, stand_open, (i + 1) / 6.0),
            STAND_A_XY,
            YAW_A,
            "look",
            ("rest", obj_a),
            "arrive1",
        )

    # 3) REACH+GRASP: lean forward + lerp arm_rest -> grasp_arm + open -> wrapped hand; object held at end.
    n_reach = 18
    for i in range(n_reach):
        s = _smoothstep((i + 1) / n_reach)
        held = i >= n_reach - 1
        add(
            manip(core.lerp_arm(arm_rest, grasp_arm, s), s, lean_frac=s),
            STAND_A_XY,
            YAW_A,
            "look",
            ("carry", otw) if held else ("rest", obj_a),
            "grasp",
        )

    # 3b) LIFT: straighten up from the lean, carrying the object (a natural pick-up).
    for i in range(10):
        s = _smoothstep((i + 1) / 10.0)
        add(
            manip(grasp_arm, 1.0, lean_frac=1.0 - s),
            STAND_A_XY,
            YAW_A,
            "look",
            ("carry", otw),
            "lift",
        )

    # 4) TURN: rotate in place YAW_A -> YAW_B holding the object.
    for i in range(14):
        s = _smoothstep((i + 1) / 14.0)
        add(
            manip(grasp_arm, 1.0),
            STAND_A_XY,
            YAW_A + (YAW_B - YAW_A) * s,
            "look",
            ("carry", otw),
            "turn",
        )

    # 5) WALK2: STAND_A -> STAND_B carrying (legs/spine walk; right arm overridden to hold).
    wf2, _ = gait.plan_walk(
        g, STAND_A_XY, STAND_B_XY, 0.0, WALK_SPEED, FPS, start_dist=0.0
    )
    for wf in wf2:
        m.frames.append(
            Frame(
                wf.pose156,
                wf.root_xy,
                YAW_B,
                wf.root_z,
                arm_override=grasp_arm,
                rhp=rhp,
                fpv_mode="walk",
                obj=("carry", otw),
                seg="walk2",
            )
        )

    # 6) ARRIVE2: blend walk -> upright manip stand (still holding) at STAND_B.
    carry_stand = manip(grasp_arm, 1.0)
    walk2_last = wf2[-1].pose156.copy()
    for j, v in grasp_arm.items():
        walk2_last[j * 3 : j * 3 + 3] = v
    walk2_last[H.RHAND_LO : H.RHAND_LO + 45] = rhp
    for i in range(6):
        add(
            gait.blend156(walk2_last, carry_stand, (i + 1) / 6.0),
            STAND_B_XY,
            YAW_B,
            "look",
            ("carry", otw),
            "arrive2",
        )

    # 7) PLACE: lean + warp the arm so the HAND delivers the cup to obj_b on DeskB, still gripping;
    # release (cup -> rest upright at obj_b, where the hand is) only near the end.
    n_place = 18
    for i in range(n_place):
        s = _smoothstep((i + 1) / n_place)
        released = s >= 0.9
        add(
            manip(core.lerp_arm(grasp_arm, place_arm, s), 1.0, lean_frac=s),
            STAND_B_XY,
            YAW_B,
            "look",
            ("rest", obj_b) if released else ("carry", otw),
            "place",
        )

    # 8) RETRACT: straighten + open hand + arm back to rest; the cup stays resting on DeskB.
    for i in range(10):
        s = _smoothstep((i + 1) / 10.0)
        add(
            manip(core.lerp_arm(place_arm, arm_rest, s), 1.0 - s, lean_frac=1.0 - s),
            STAND_B_XY,
            YAW_B,
            "look",
            ("rest", obj_b),
            "retract",
        )

    # 9) STEP AWAY: turn from DeskB and walk into the corridor, so the cup is CLEARLY left on DeskB
    # (the human otherwise stands right at the desk edge, making the placed cup look attached to the body).
    for i in range(12):
        s = _smoothstep((i + 1) / 12.0)
        add(
            manip(arm_rest, 0.0),
            STAND_B_XY,
            YAW_B + 180.0 * s,
            "walk",
            ("rest", obj_b),
            "away_turn",
        )
    wf3, _ = gait.plan_walk(
        g, STAND_B_XY, (0.0, -0.2), 0.0, WALK_SPEED, FPS, start_dist=0.0
    )
    for wf in wf3:
        m.frames.append(
            Frame(
                wf.pose156,
                wf.root_xy,
                wf.yaw_deg,
                wf.root_z,
                fpv_mode="walk",
                obj=("rest", obj_b),
                seg="away",
            )
        )
    return m


def _grasp_frame(stand_xy, yaw, stand_z):
    """The (rz, world_off) body frame + leaned base pose used to solve the grasp at a stand point."""
    a = math.radians(yaw)
    c, s = math.cos(a), math.sin(a)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world_off = np.array([stand_xy[0], stand_xy[1], stand_z])
    return rz, world_off


# =====================================================================================================
# Offline dry run: verify the whole choreography + reach geometry + a storyboard, NO Isaac.
# =====================================================================================================
def run_dry(rig, g: gait.Gait) -> int:
    V0, _ = core.pose_body(rig, rig.arm_grasp, (0.0, 0.0), 0.0)
    stand_z = 0.0 - float(V0[:, 2].min())  # feet on floor_z=0
    obj_a_z = obj_b_z = 0.75  # office desk top (approx; render reads it exactly)
    obj_a = (OBJ_A_XY[0], OBJ_A_XY[1], obj_a_z)
    rz, world_off = _grasp_frame(STAND_A_XY, YAW_A, stand_z)
    base = _add_lean(
        core.assemble_pose(rig, rig.arm_grasp, (0.0, 0.0, 0.0), 1.0),
        math.radians(REACH_LEAN_DEG),
    )
    grasp = solve_grasp(rig, base, rz, world_off, np.array(obj_a), sd_world=None)
    place_arm, place_err = solve_place_arm(
        rig, grasp["grasp_arm"], np.array([OBJ_B_XY[0], OBJ_B_XY[1], obj_b_z]), stand_z
    )
    print(
        f"[wpp][dry] stand_z={stand_z:.3f} pick seat_err={grasp['seat_err'] * 1000:.1f}mm "
        f"place seat_err={place_err * 1000:.1f}mm lean={REACH_LEAN_DEG}deg"
    )

    m = build_motion(rig, g, stand_z, obj_a_z, obj_b_z, grasp, place_arm)
    print(f"[wpp][dry] motion = {len(m.frames)} frames")

    Js, objs, segs = [], [], []
    for fr in m.frames:
        _, J, objT = realize(rig, fr)
        Js.append(J)
        objs.append(objT[:3, 3] if objT is not None else None)
        segs.append(fr.seg)
    Js = np.array(Js)
    foot_min = Js[:, list(accad_retarget._FOOT_J), 2].min(axis=1)
    print(
        f"[wpp][dry] foot-floor z: min={foot_min.min() * 1000:.0f}mm max={foot_min.max() * 1000:.0f}mm (want ~0)"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        want = ["walk1", "grasp", "lift", "turn", "walk2", "place"]
        picks = [next((k for k, s in enumerate(segs) if s == w), 0) for w in want]
        # grasp: use the LAST grasp frame (deepest lean) for a clearer reach picture.
        picks[1] = max(k for k, s in enumerate(segs) if s == "grasp")
        fig, axes = plt.subplots(
            1, len(picks), figsize=(2.6 * len(picks), 4.4), squeeze=False
        )
        parents = H.PARENTS
        for ax, k, name in zip(axes[0], picks, want):
            J = Js[k]
            for j in range(len(parents)):
                p = int(parents[j])
                if p >= 0:
                    ax.plot(
                        [-J[j, 1], -J[p, 1]],
                        [J[j, 2], J[p, 2]],
                        "-",
                        color="#1f5fbf",
                        lw=1.4,
                    )
            if objs[k] is not None:
                ax.scatter(
                    [-objs[k][1]], [objs[k][2]], s=70, marker="s", color="#e0952b"
                )
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_title(f"{name} f{k}", fontsize=8)
            ax.set_xlim(-2.4, 2.4)
            ax.set_ylim(-0.1, 1.95)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            "walk->lean/pick->lift->turn->walk->place (obj=orange square, forward=right)",
            fontsize=9,
        )
        fig.tight_layout()
        out = os.path.abspath(
            os.path.join(_TRY_DIR, "fpv_outputs", "wpp_dry_storyboard.png")
        )
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(f"[wpp][dry] storyboard -> {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[wpp][dry] montage skipped ({exc})")

    ok = abs(float(foot_min.min())) < 0.05 and grasp["seat_err"] < 0.02
    print(f"[wpp][dry] CHECK reach+feet = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


# =====================================================================================================
# Main.
# =====================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry", action="store_true", help="offline verify (no Isaac / render)"
    )
    ap.add_argument("--tag", default="wpp")
    ap.add_argument("--obj", default=OBJ_MESH)
    ap.add_argument("--frames", type=int, default=0, help="cap total frames (0 = all)")
    ap.add_argument(
        "--behavior-obj",
        default=None,
        help="grab a BAKED BEHAVIOR object 'category/model' (real USD geometry + a shape-matched proven seat)",
    )
    ap.add_argument(
        "--behavior-family",
        default="side_wrap",
        help="atlas grip family for a BEHAVIOR object: side_wrap (cup/box/tall) or round_wrap (apple/ball)",
    )
    ap.add_argument(
        "--behavior-seat",
        default="cup",
        help="atlas object whose PROVEN seat to reuse for a BEHAVIOR object (cup/ball/box/plate/pen/tool)",
    )
    args = ap.parse_args()

    # A BEHAVIOR object reuses a proven family+seat MATCHED to its shape (round->round_wrap+ball,
    # cup/tall->side_wrap+cup); the live SDF wrap then conforms the fingers to the real baked mesh. With
    # no --behavior-obj this is byte-identical to before (side_wrap + the solved cup def).
    rig = core.load_family(args.behavior_family if args.behavior_obj else "side_wrap")
    build, otw = core.load_object_def(
        args.behavior_seat if args.behavior_obj else "cup"
    )
    if args.behavior_obj:
        import behavior_import

        build = behavior_import.behavior_build_params(args.behavior_obj)
        print(
            f"[wpp] BEHAVIOR grab {args.behavior_obj} family={args.behavior_family} "
            f"seat={args.behavior_seat} -> {build['usd']}"
        )
    core.set_object(rig, build, otw)
    g = gait.build_gait()

    if args.dry:
        return run_dry(rig, g)
    return run_render(rig, g, args)


def run_render(rig, g: gait.Gait, args, build_fn=None, obj_a_xy=None) -> int:
    """Boot Isaac, build the scene, precompute the motion, and render FPV + TPV mp4s.

    ``obj_a_xy`` overrides where the pickup object rests on DeskA (default OBJ_A_XY), so a different
    object can sit at a different position. ``build_fn(rig, g, stand_z, obj_a_z, obj_b_z, sd_world,
    obj_a_xy) -> Motion`` (optional) overrides the hardcoded choreography with an externally-built motion
    (e.g. the command-driven semantic_executor), reusing this exact scene/physics/camera setup.
    """
    obj_a_xy = OBJ_A_XY if obj_a_xy is None else tuple(obj_a_xy)
    from fpv_core import AppCfg, RenderCfg

    W, Hh = 960, 760
    app_cfg = AppCfg(headless=True, width=W, height=Hh, renderer="RayTracedLighting")
    rcfg = RenderCfg(capture_fps=FPS, playback_fps=FPS, max_frames=400, rt_subframes=6)

    import fpv_core

    app = fpv_core.launch_app(app_cfg)
    import render_runtime  # _loco_cfg + _make_sd_world + _boost_lights + _clear_frame_pngs

    fpv_core.post_launch_setup(app_cfg, render_runtime._loco_cfg())
    import fpv_render
    import fpv_scene
    import office_scene
    import smplh_camera
    from fpv_segments import run_segment, SegmentCtx

    stage = fpv_core.get_stage()

    layout = office_scene.build_office(stage, TABLE_A_XY, TABLE_B_XY)
    floor_z = float(layout.floor_z)
    print(f"[wpp] office floor_z={floor_z:.3f}")

    # SM_TableB's long axis is DEPTH (Y, 2.8 m). The default placement puts the human INSIDE the desk
    # footprint (standing under the top slab) -> reads as clipping even though the exact-mesh detector
    # shows 0 real penetration. Rotate each desk 90° about Z so it is WIDE (X) and SHALLOW (0.8 m in Y)
    # with a near front edge the human stands IN FRONT of and reaches over (§12.6: desk yaw 90°, body
    # stands clear + walks up to it).
    _YAW90 = (0.70710678, 0.0, 0.0, 0.70710678)  # wxyz, Rz(90°)

    def _reorient_desk(path, xy):
        prim = stage.GetPrimAtPath(path)
        fpv_scene.set_world_pose(prim, (xy[0], xy[1], floor_z), _YAW90)
        mz = float(
            fpv_scene.local_bbox_min_z(prim)
        )  # local bbox min-z is invariant to a Z rotation
        fpv_scene.set_world_pose(prim, (xy[0], xy[1], floor_z - mz), _YAW90)
        rng = fpv_scene.compute_world_bbox(prim)
        return float(rng.GetMax()[2]), (float(rng.GetMin()[1]), float(rng.GetMax()[1]))

    top_a, ya = _reorient_desk("/World/DeskA", TABLE_A_XY)
    top_b, yb = _reorient_desk("/World/DeskB", TABLE_B_XY)
    print(
        f"[wpp] desks rotated 90°: DeskA y[{ya[0]:.2f},{ya[1]:.2f}] top={top_a:.3f} | "
        f"DeskB y[{yb[0]:.2f},{yb[1]:.2f}] top={top_b:.3f} (STAND_A={STAND_A_XY[1]} in front of {ya[1]:.2f})"
    )

    # The default office RectLights (11000) blow the GLOSSY desk top to pure white in the down-looking
    # FPV (specular), hiding the object. Dim them AND rebind the desks to a MATTE material so the desk
    # top reads as a surface (no specular blow-out) and the cup is clearly visible resting on it.
    from pxr import Sdf, UsdLux, UsdShade

    for _i in range(4):
        _lp = stage.GetPrimAtPath(f"/World/OfficeLights/L{_i}")
        if _lp and _lp.IsValid():
            UsdLux.RectLight(_lp).GetIntensityAttr().Set(5000.0)

    def _matte(prim_path, color):
        prim = stage.GetPrimAtPath(prim_path)
        if not (prim and prim.IsValid()):
            return
        mtl = UsdShade.Material.Define(stage, prim_path + "/MatteMat")
        sh = UsdShade.Shader.Define(stage, prim_path + "/MatteMat/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

    _matte("/World/DeskA", (0.62, 0.45, 0.30))
    _matte("/World/DeskB", (0.62, 0.45, 0.30))

    V0, _ = core.pose_body(rig, rig.arm_grasp, (0.0, 0.0), 0.0)
    stand_z = floor_z - float(V0[:, 2].min())

    core.build_object(stage, rig)
    obj_min_z = float(fpv_scene.local_bbox_min_z(stage.GetPrimAtPath("/World/Obj")))
    obj_a_z, obj_b_z = top_a - obj_min_z, top_b - obj_min_z
    obj_a = np.array([obj_a_xy[0], obj_a_xy[1], obj_a_z])
    core.place_obj_resting(rig, obj_a)
    for _ in range(2):
        app.update()

    # Cup as a rigid body, physics-VALIDATED in its free state (§12.5/§13): make it dynamic, drop it on
    # DeskA and let PhysX rest it, read back the true resting z (falls back + flags if the drop is
    # unreliable). Both desks are exact-mesh static colliders. Then keep the cup KINEMATIC so the render
    # drives it (rests at the validated z / rides the wrist when held), never letting gravity pull it off.
    import physics_human as ph

    ph.build_physics_scene(stage)
    ph.make_static_collider(stage.GetPrimAtPath("/World/DeskA"), approximation="none")
    ph.make_static_collider(stage.GetPrimAtPath("/World/DeskB"), approximation="none")
    ph.make_dynamic_rigid(stage.GetPrimAtPath("/World/Obj"), density=400.0)
    # PhysX + the settle stale cached USD handles ("Accessed schema on invalid prim") -> make the per-
    # frame object/body writers re-acquire fresh handles each call.
    core.enable_physics_refetch(rig, stage, "/World/Obj", "/World/Body")
    rested_z, settle_ok = render_runtime._presettle_cup_on_desk(
        ph, stage, rig, core, top_a, obj_a_xy
    )
    ph.set_rigid_kinematic(stage.GetPrimAtPath("/World/Obj"), True)
    obj_a_z = float(rested_z)
    obj_a = np.array([obj_a_xy[0], obj_a_xy[1], obj_a_z])
    core.place_obj_resting(rig, obj_a)
    print(f"[wpp] cup presettle DeskA: rested_z={rested_z:.3f} settle_ok={settle_ok}")

    sd_world = render_runtime._make_sd_world(stage, "/World/Obj", cache_key="wpp_novel")
    try:
        core.derive_rest_place_arms(rig)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[wpp] derive_rest_place_arms failed ({exc}); using canonical arm as rest/place"
        )
    if build_fn is not None:
        motion = build_fn(rig, g, stand_z, obj_a_z, obj_b_z, sd_world, obj_a_xy)
    else:
        rz, world_off = _grasp_frame(STAND_A_XY, YAW_A, stand_z)
        base = _add_lean(
            core.assemble_pose(rig, rig.arm_grasp, (0.0, 0.0, 0.0), 1.0),
            math.radians(REACH_LEAN_DEG),
        )
        grasp = solve_grasp(rig, base, rz, world_off, obj_a, sd_world)
        obj_b = np.array([OBJ_B_XY[0], OBJ_B_XY[1], obj_b_z])
        place_arm, place_err = solve_place_arm(rig, grasp["grasp_arm"], obj_b, stand_z)
        print(
            f"[wpp] pick seat_err={grasp['seat_err'] * 1000:.1f}mm near={grasp['near']}/5 "
            f"place seat_err={place_err * 1000:.1f}mm lean={REACH_LEAN_DEG}"
        )
        motion = build_motion(rig, g, stand_z, obj_a_z, obj_b_z, grasp, place_arm)
    frames = motion.frames if not args.frames else motion.frames[: args.frames]
    print(f"[wpp] motion = {len(frames)} frames")

    # Render-time collision nudge (bound at the render entry point only). The office/cup scene has no
    # footprint furniture, so nudge_boxes is empty -> a safe no-op; the executor render path routes
    # through here, so the hook is in place for any footprint furniture a build_fn adds.
    nudge_boxes: list = []
    nudge_state: dict = {}

    V_init, J_init, obj_init = realize(
        rig, frames[0], scene_boxes=nudge_boxes, nudge_state=nudge_state
    )
    core.author_body_once(stage, rig, V_init)
    cop = smplh_camera.make_fpv_camera(stage, "/World/FpvCam", W, Hh)
    tcop = smplh_camera.make_tpv_camera(stage, "/World/TpvCam", W, Hh)
    fpv_core.add_lighting(stage, app_cfg)
    render_runtime._boost_lights(stage)

    def drive_cams(fr, J, obj_T):
        """Position FPV + TPV for a frame.

        FPV: push the eye mostly FORWARD (near head height) so it clears the torso, and get the
        downward look from AIMING at the object (not by tilting the eye down into the chest -- that
        was the 'body wall'). TPV: follow from the +X side.
        """
        head = J[H.HEAD_J]
        fwd = _body_forward(fr.yaw)
        if fr.fpv_mode == "walk":
            eye_head = head
            gaze = fwd + np.array([0.0, 0.0, -0.10])
            obj_ctr = head + fwd * 3.0  # aim far ahead down the path
            blend, push = 0.40, 0.20
        else:
            # Look FORWARD at the workspace with a gentle downward tilt (NOT steeply down at the low
            # object). Aiming above the cup keeps a moderate pitch so the desk+object+hand sit in the
            # lower-center and the body stays at the bottom edge instead of a steep-down look that fills
            # the frame with the leaned-over torso.
            eye_head = head + np.array([0.0, 0.0, 0.05])
            gaze = fwd + np.array([0.0, 0.0, -0.18])
            obj_ctr = (
                (obj_T[:3, 3] + np.array([0.0, 0.0, 0.30]))
                if obj_T is not None
                else head + fwd
            )
            blend, push = 0.55, 0.16
        smplh_camera.drive_fpv_gaze(
            cop, eye_head, gaze, obj_ctr, eye_push=push, aim_blend=blend
        )
        root = np.array([fr.xy[0], fr.xy[1], 0.0])
        # Pure side view (+X) so the reach + BOTH desks stay in frame through walk, pick, turn, place.
        smplh_camera.drive_tpv(
            tcop, (root[0] + 3.3, root[1], 1.6), (root[0], root[1], 1.0)
        )

    # Position the cameras at frame 0 BEFORE settle_and_validate (else the first render is black).
    drive_cams(frames[0], J_init, obj_init)

    out = os.path.join(_TRY_DIR, "fpv_outputs")
    fpv_dir = os.path.join(out, f"{args.tag}_frames")
    tpv_dir = os.path.join(out, f"{args.tag}_tpv_frames")
    render_runtime._clear_frame_pngs(fpv_dir, tpv_dir)
    rp_f, rp_t, a_f, a_t = fpv_render.setup_products(
        "/World/FpvCam", "/World/TpvCam", rcfg, W, Hh
    )
    timeline = fpv_core.start_timeline(rcfg.playback_fps)
    fpv_render.settle_and_validate(app, timeline, a_f, app_cfg, annot_tpv=a_t)
    ctx = SegmentCtx(a_f, a_t, fpv_dir, tpv_dir, rcfg, 1.0 / FPS, W, Hh)

    n = len(frames)

    def on_frame(t, gidx):
        fr = frames[min(int(round(t * (n - 1))), n - 1)]
        V, J, obj_T = realize(rig, fr, scene_boxes=nudge_boxes, nudge_state=nudge_state)
        core.update_body(rig, V)
        if obj_T is not None:
            core.set_object_world(rig, obj_T)
        drive_cams(fr, J, obj_T)

    run_segment(ctx, "walk_pick_place", n, on_frame, 0, rcfg.max_frames)

    fpv_render.teardown([rp_f, rp_t], [a_f, a_t])
    fpv_render.encode_mp4(fpv_dir, os.path.join(out, f"{args.tag}_fpv.mp4"), rcfg)
    fpv_render.encode_mp4(tpv_dir, os.path.join(out, f"{args.tag}_tpv.mp4"), rcfg)
    print(f"[wpp] DONE -> {out}/{args.tag}_fpv.mp4 + {args.tag}_tpv.mp4")
    app.close()
    return 0

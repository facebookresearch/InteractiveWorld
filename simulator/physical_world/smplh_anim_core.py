# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SMPL+H rig + side-grip grasp core for dynamic FPV rendering (no SimulationApp here).

This is the reusable bridge between the STATIC side-grip pipeline (human_model/smplh_fpv.py) and the
dynamic single-clock render framework (simulator/physical_world/fpv_*). It owns:

  * loading the SMPL+H model + the LOCKED side-grip grasp (grasp_<obj>_side.npz),
  * the FROZEN canonical side arm (from smplh_fpv.py) + the verbatim 45-D finger pose,
  * per-frame body posing with an analytic locomotion root (Rz(yaw)@V + [x,y,0]),
  * the kinematic cup attach (T_wrist_world(J) @ OBJ_T_IN_WRIST_baked, transposed for USD),
  * a ONE-TIME side penetration backoff folded into OBJ_T_IN_WRIST_baked,
  * authoring + per-frame updating the /World/Body mesh, and
  * per-joint axis-angle LERP of the arm for the reach / retract / place segments.

HARD CONSTRAINTS honored here (do not change):
  * The finger pose (RHP) is applied VERBATIM every frame; the wrist joint is 0 every frame.
  * The canonical (collar, shoulder, elbow) is FROZEN for every HELD frame; we never re-IK
    while the cup is attached (re-IK re-rolls the forearm and moves the grip off the validated
    side configuration). _ik_arm runs at most ONCE per derived rest/place pose, seeded from
    canonical, for the cup-DETACHED reach/place endpoints only.
  * Cup attach is a pure kinematic USD matrix write; no physics, no FixedJoint.

pxr/omni are imported lazily inside the functions that need them, so importing this module
before launch_app() never pulls in Kit. The hand math (human_model/smplh_fpv.py) imports torch at its
module scope; import THIS module before launch_app so torch loads first (the known-good order).
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pxr import Usd  # noqa: F401


# The hand pipeline lives in ../human_model; add it to sys.path so `import smplh_fpv` resolves
# regardless of cwd, then import the reusable math + constants. smplh_fpv is import-safe (its
# bootstrap / grasp solve / render are guarded behind __main__).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HUMAN_CODE_DIR = os.path.join(os.path.dirname(_THIS_DIR), "human_model")
_HAND_DIR = os.environ.get("INTERACTIVEWORLD_HUMAN_ASSET_ROOT", "") or os.path.join(
    _HUMAN_CODE_DIR, "assets"
)
_ISAAC_ASSET_ROOT = os.environ.get(
    "INTERACTIVEWORLD_ISAAC_ASSET_ROOT", "/opt/isaac-sim/Assets/Isaac/5.1/Isaac"
)
if _HUMAN_CODE_DIR not in sys.path:
    sys.path.insert(0, _HUMAN_CODE_DIR)

import smplh_fpv as H  # noqa: E402  (after sys.path insert)

# Canonical SIDE arm (cup-derived, shared by every side object) -- FROZEN for all held frames.
# Verbatim from smplh_fpv.py; reused, never re-tuned.
R_COLLAR_AA = [0.35, 0.55, 0.31]
R_SHOULDER_AA = [0.38, 1.77, -0.42]
R_ELBOW_AA = [-1.18, -0.93, -1.3]

# Joint indices + the relaxed-left-arm + collar constants (mirror smplh_fpv).
WRIST_J = H.WRIST_J
R_COLLAR, R_SHOULDER, R_ELBOW = H.R_COLLAR, H.R_SHOULDER, H.R_ELBOW
HEAD_J = H.HEAD_J
L_COLLAR, L_SHOULDER, L_ELBOW, L_WRIST = 13, 16, 18, 20
COLLAR = [0.0, 0.15, 0.20]
L_ARM = {L_SHOULDER: [0.60, -0.20, -1.40], L_ELBOW: [0.0, 0.40, 0.0]}
NJ = H.NJ
LHAND_LO = 22 * 3
RHAND_LO = H.RHAND_LO


@dataclass
class SmplhRig:
    """All state for the SMPL+H body + the locked side grasp, with the baked attach transform."""

    rhp: np.ndarray  # (45,) finger pose, applied VERBATIM (side path -> no reshaping)
    obj_T_in_wrist: np.ndarray  # (4,4) object pose in the wrist frame (from the npz)
    build: dict  # build_params -> the mug USD to reconstruct
    arm_grasp: dict = field(
        default_factory=dict
    )  # FROZEN canonical side arm (collar/sh/el + L arm)
    arm_rest: dict = field(
        default_factory=dict
    )  # relaxed arm-down (cup detached); derived once
    arm_place: dict = field(
        default_factory=dict
    )  # arm lowered to the table-B drop (cup detached); once
    obj_T_in_wrist_baked: np.ndarray | None = (
        None  # obj_T_in_wrist + the folded one-time side backoff
    )
    # Relaxed OPEN-hand pose (45,) for the hand_close=0 end of a reach/release ramp. When None (default)
    # the open hand is the flat zero pose (existing behaviour, byte-identical). Set it (gated) to a mild
    # finger curl so the hand does not splay into a "starfish" while reaching / releasing.
    rhp_open: np.ndarray | None = None
    # Naturalness (gated per-scene via scene.natural_arm). When False (default) warp_arm_to_seat solves
    # from the frozen side-grip arm and returns it as-is -> existing behaviour, byte-identical. When True,
    # warp applies a reach-EXACT null-space elbow swivel after the position CCD (rotate the arm about the
    # shoulder->seat axis so the elbow points DOWN instead of the side seed's "winged-up" wrench). The
    # seat lies on that axis, so the reach target does NOT move -> arbitrary-target reach is preserved.
    natural_arm: bool = False
    # Cached USD handles (set by author_body_once / build the object anchor).
    _body_mesh: object | None = None
    _normals_pv: object | None = None
    _obj_xf: object | None = None
    faces: np.ndarray | None = None
    # Physics path only: when set, the per-frame USD writers re-acquire fresh xform-op / mesh handles
    # each call instead of reusing the cached ones. PhysX simulation + the detector's stage attach /
    # detach + session-layer edits STALE cached schema handles ('Accessed schema on invalid prim');
    # a freshly-fetched handle always works, so the physics demo flips this on after build.
    _stage: object | None = None
    _obj_path: str = "/World/Obj"
    _body_path: str = "/World/Body"
    _refetch_handles: bool = False
    _reach_pose_cache: dict = field(default_factory=dict, repr=False)


# ----------------------------------------------------------------------------------------------
# Load + bake.
# ----------------------------------------------------------------------------------------------
def load_rig(grasp_npz_path: str) -> SmplhRig:
    """Load the SMPL+H model + the side-grip grasp npz into an SmplhRig (no Isaac needed yet).

    The 45-D finger pose is taken VERBATIM (the side path applies it directly -- no FLAT/WRAP
    reshaping). The canonical side arm is frozen. obj_T_in_wrist_baked starts as a copy of the
    npz transform; bake_grasp() folds the one-time penetration backoff into it.
    """
    import json

    g = np.load(grasp_npz_path, allow_pickle=True)
    grip = str(g["grip_type"])
    if grip != "side-wrap":
        raise ValueError(
            f"{grasp_npz_path}: grip_type={grip!r}, expected 'side-wrap'. The dynamic demo only "
            "supports the LOCKED side grip; load a grasp_<obj>_side.npz."
        )
    rhp = g["right_hand_pose"].astype(np.float32)  # VERBATIM -- do not mutate
    obj_T_in_wrist = g["obj_T_in_wrist"].astype(np.float64)
    build = json.loads(str(g["build_params"]))
    arm_grasp = {
        R_COLLAR: list(R_COLLAR_AA),
        R_SHOULDER: list(R_SHOULDER_AA),
        R_ELBOW: list(R_ELBOW_AA),
        WRIST_J: [
            0.0,
            0.0,
            0.0,
        ],  # canonical carry: wrist joint 0 (the reference side-grip)
        **{k: list(v) for k, v in L_ARM.items()},
    }
    rig = SmplhRig(
        rhp=rhp,
        obj_T_in_wrist=obj_T_in_wrist,
        build=build,
        arm_grasp=arm_grasp,
        obj_T_in_wrist_baked=obj_T_in_wrist.copy(),
        faces=H.FACES,
    )
    print(
        f"[anim_core] loaded grasp: grip={grip} build={build} "
        f"rhp[:3]={rhp[:3].round(3).tolist()} obj_T_in_wrist_t={obj_T_in_wrist[:3, 3].round(4).tolist()}"
    )
    return rig


# ----------------------------------------------------------------------------------------------
# Family templates + object defs (decouple the per-object grasp file into a SHARED family grip +
# a per-object GEOMETRY/seat). Grasp = family (rhp seed + grip_type, object-independent). Object =
# build_params + obj_T_in_wrist (geometry + where it seats in the hand). Known objects keep their
# solved seat (render unchanged); arbitrary meshes get a geometry-derived seat.
# ----------------------------------------------------------------------------------------------
def load_family(family_id: str) -> SmplhRig:
    """Load a SHARED family grip template (atlas/<family>_family.npz) -- grip only, no object.

    Carries the seed finger pose (rhp, engaged via the live contact-wrap), grip_type, and a
    representative seat_template that derive_seat() adapts per object. The frozen canonical side arm is
    the same for every object. Attach the object's geometry + seat afterwards with set_object()."""
    path = os.path.join(_HAND_DIR, "atlas", f"{family_id}_family.npz")
    g = np.load(path, allow_pickle=True)
    rhp = g["right_hand_pose"].astype(np.float32)
    arm_grasp = {
        R_COLLAR: list(R_COLLAR_AA),
        R_SHOULDER: list(R_SHOULDER_AA),
        R_ELBOW: list(R_ELBOW_AA),
        WRIST_J: [0.0, 0.0, 0.0],
        **{k: list(v) for k, v in L_ARM.items()},
    }
    rig = SmplhRig(
        rhp=rhp,
        obj_T_in_wrist=None,
        build=None,
        arm_grasp=arm_grasp,
        obj_T_in_wrist_baked=None,
        faces=H.FACES,
    )
    rig.family_id = str(g["family_id"]) if "family_id" in g.files else family_id
    rig.grip_type = str(g["grip_type"]) if "grip_type" in g.files else "side-wrap"
    rig.seat_template = (
        g["seat_template"].astype(np.float64) if "seat_template" in g.files else None
    )
    rig.wrist_R_world = (
        g["wrist_R_world"].astype(np.float64) if "wrist_R_world" in g.files else None
    )  # family approach orientation (world<-wrist); the reach targets it (or r_canon for side-wrap)
    rig.thumb_opp = float(g["thumb_opp"]) if "thumb_opp" in g.files else None
    print(f"[anim_core] loaded FAMILY {family_id}: grip_type={rig.grip_type}")
    return rig


def load_object_def(obj_name: str) -> tuple[dict, "np.ndarray | None"]:
    """Load an object def (atlas/objects/<obj>.npz) -> (build_params, obj_T_in_wrist or None).

    An object def is GEOMETRY + how the object seats in the hand -- NOT a grasp (the grasp is the
    family). Known objects (e.g. the cup) keep their exact solved seat here so the render is unchanged."""
    import json

    path = os.path.join(_HAND_DIR, "atlas", "objects", f"{obj_name}.npz")
    g = np.load(path, allow_pickle=True)
    build = json.loads(str(g["build_params"]))
    # Bundled defs name Isaac props relative to the Isaac Sim asset root.
    if build.get("shape") == "usd_ref" and not os.path.isabs(build["usd"]):
        build["usd"] = os.path.join(_ISAAC_ASSET_ROOT, build["usd"])
    otw = (
        g["obj_T_in_wrist"].astype(np.float64) if "obj_T_in_wrist" in g.files else None
    )
    return build, otw


def derive_seat(rig: SmplhRig) -> np.ndarray:
    """Geometry-driven seat (obj_T_in_wrist) for an arbitrary object with NO stored def. v1 heuristic:
    keep the family seat_template's orientation (the side-wrap upright-in-wrist frame) and rescale the
    palm-normal (wrist +Z) offset by the object's cross-section radius so the wall meets the palm; the
    runtime backoff + the live contact-wrap then refine the exact fit. Dims come from build_params
    (sphere/cylinder) or the mesh bbox. (Not exercised by known objects, which use their stored seat.)"""
    tmpl = (
        rig.seat_template
        if getattr(rig, "seat_template", None) is not None
        else np.eye(4)
    )
    seat = np.asarray(tmpl, dtype=np.float64).copy()
    b = rig.build or {}
    shape = b.get("shape")
    if shape in ("sphere", "cylinder"):
        r = float(b["radius"])
    elif shape == "mesh":
        import trimesh

        src = b["mesh"]
        m = (
            trimesh.load(src, force="mesh")
            if isinstance(src, str)
            else trimesh.Trimesh(vertices=src["vertices"], faces=src["faces"])
        )
        r = 0.5 * float(
            sorted(m.bounding_box.extents)[1]
        )  # mid extent ~ cross-section radius
    else:
        r = 0.04  # fallback ~ a typical grip radius
    r_template = 0.04  # ~ the radius the side-wrap template was solved for (the mug)
    seat[2, 3] = float(tmpl[2, 3]) * (
        r / r_template
    )  # palm-normal offset scales with object radius
    return seat


def set_object(rig: SmplhRig, build_params: dict, obj_T_in_wrist=None) -> None:
    """Attach an object's GEOMETRY (build_params) + seat to a family rig. A provided obj_T_in_wrist
    (known object) is used verbatim (render unchanged); otherwise the seat is DERIVED from geometry.
    Sets obj_T_in_wrist_baked = a copy (bake_side_backoff folds the runtime backoff in later)."""
    rig.build = build_params
    if obj_T_in_wrist is None:
        obj_T_in_wrist = derive_seat(rig)
    rig.obj_T_in_wrist = np.asarray(obj_T_in_wrist, dtype=np.float64)
    rig.obj_T_in_wrist_baked = rig.obj_T_in_wrist.copy()
    print(
        f"[anim_core] set_object: shape={build_params.get('shape')} "
        f"seat_t={rig.obj_T_in_wrist[:3, 3].round(4).tolist()}"
    )


def make_body_only_rig(build_npz_path: str) -> SmplhRig:
    """An SmplhRig carrying ONLY the body faces + the object's build_params (no side-grip arm/grip).

    For the GRAB-retarget path the body pose comes from real mocap (V,J are supplied directly to
    author_body_once / update_body), so the locked finger pose + canonical arm + obj_T_in_wrist are
    irrelevant. We still reuse the proven USD authoring (author_body_once / update_body / build_object
    / set_object_world / enable_physics_refetch), which only needs rig.faces + rig.build, so we load
    those two from a grasp_<obj>_side.npz (its build_params point at the SAME object USD the demo uses)
    and leave the grasp-specific fields as harmless identity defaults. NO grip math is run on this rig.
    """
    import json

    g = np.load(build_npz_path, allow_pickle=True)
    build = json.loads(str(g["build_params"]))
    rig = SmplhRig(
        rhp=np.zeros(45, dtype=np.float32),
        obj_T_in_wrist=np.eye(4),
        build=build,
        obj_T_in_wrist_baked=np.eye(4),
        faces=H.FACES,
    )
    print(f"[anim_core] body-only rig: faces={len(H.FACES)} build={build}")
    return rig


def mirror_axis_angle_across_x(axis_angle) -> np.ndarray:
    """Mirror an axial rotation vector across the body's left/right plane."""
    value = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    return value * np.array([1.0, -1.0, -1.0], dtype=np.float64)


def mirror_right_arm_as_left(arm: dict) -> dict:
    """Return the left-arm counterpart of the right-arm joints in ``arm``."""
    joint_pairs = (
        (R_COLLAR, L_COLLAR),
        (R_SHOULDER, L_SHOULDER),
        (R_ELBOW, L_ELBOW),
        (WRIST_J, L_WRIST),
    )
    return {
        left: list(mirror_axis_angle_across_x(arm[right]))
        for right, left in joint_pairs
        if right in arm
    }


def mirror_right_hand_as_left(right_hand_pose) -> np.ndarray:
    """Mirror the 15-joint MANO right-hand pose into the SMPL+H left-hand block."""
    pose = np.asarray(right_hand_pose, dtype=np.float64).reshape(15, 3)
    return (pose * np.array([1.0, -1.0, -1.0], dtype=np.float64)).reshape(45)


def assemble_pose(
    rig: SmplhRig,
    arm: dict,
    extra_wrist=(0.0, 0.0, 0.0),
    hand_close: float = 1.0,
    *,
    left_hand_close: float = 0.0,
) -> np.ndarray:
    """Build the full 156-D SMPL+H pose: upright global_orient + arm + wrist + (partially-closed) fingers.

    `hand_close` scales the locked finger curl: 0.0 = the hand is EXTENDED/open (a natural reach), 1.0
    = the full locked side-grip. The side-grip only fully engages at contact, so the approach uses an
    open hand (hand_close=0) and the grasp animates hand_close 0->1. The wrist joint may be carried IN
    the arm dict (the grasp pose rotates it so the geometric wrist frame == the canonical side-grip);
    if absent, fall back to extra_wrist (default 0 = a straight wrist).
    """
    pose = np.zeros(NJ * 3, dtype=np.float32)
    pose[0:3] = [np.pi / 2.0, 0.0, 0.0]  # SMPL +Y-up -> world +Z-up; body faces -Y
    for j, v in arm.items():
        pose[j * 3 : j * 3 + 3] = v
    if WRIST_J not in arm:
        pose[WRIST_J * 3 : WRIST_J * 3 + 3] = extra_wrist
    hc = float(hand_close)
    if getattr(rig, "rhp_open", None) is not None:
        # relaxed-open -> grip blend: at hc=0 a mild finger curl (not a flat splay), at hc=1 the grip.
        pose[RHAND_LO : RHAND_LO + 45] = rig.rhp_open * (1.0 - hc) + rig.rhp * hc
    else:
        pose[RHAND_LO : RHAND_LO + 45] = (
            rig.rhp * hc
        )  # 0=extended ... 1=locked side-grip
    left_closed = mirror_right_hand_as_left(rig.rhp)
    left_open = (
        np.zeros(45, dtype=np.float64)
        if getattr(rig, "rhp_open", None) is None
        else mirror_right_hand_as_left(rig.rhp_open)
    )
    lhc = float(left_hand_close)
    pose[LHAND_LO : LHAND_LO + 45] = left_open * (1.0 - lhc) + left_closed * lhc
    return pose


def _upright_snap(t_nat: np.ndarray) -> np.ndarray:
    """Snap an object transform EXACTLY upright (local +Z -> world +Z) keeping position + heading.

    Verbatim port of solve_orient_side()'s snap so an open vessel never looks tilted.
    """
    z_cup = np.array([0.0, 0.0, 1.0])
    x_cup = t_nat[:3, 0] - np.dot(t_nat[:3, 0], z_cup) * z_cup
    if np.linalg.norm(x_cup) < 1e-6:
        x_cup = t_nat[:3, 1] - np.dot(t_nat[:3, 1], z_cup) * z_cup
    x_cup = x_cup / (np.linalg.norm(x_cup) + 1e-9)
    y_cup = np.cross(z_cup, x_cup)
    t_obj = np.eye(4)
    t_obj[:3, 0] = x_cup
    t_obj[:3, 1] = y_cup
    t_obj[:3, 2] = z_cup
    t_obj[:3, 3] = t_nat[:3, 3]
    return t_obj


def seat_object(rig: SmplhRig, J: np.ndarray, baked: bool = True) -> np.ndarray:
    """World transform of the cup for the posed joints J: T_wrist_world @ obj_T_in_wrist + upright snap.

    Identical placement math to solve_orient_side(): rebuild the wrist frame from the posed finger
    joints, multiply by obj_T_in_wrist (baked -> includes the one-time side backoff), then snap the
    cup exactly upright about its realized position. Returns a column-vector 4x4.
    """
    R_w, wrist_o = H.wrist_frame(J)
    T_wrist_world = np.eye(4)
    T_wrist_world[:3, :3] = R_w
    T_wrist_world[:3, 3] = wrist_o
    otw = (
        rig.obj_T_in_wrist_baked
        if (baked and rig.obj_T_in_wrist_baked is not None)
        else rig.obj_T_in_wrist
    )
    t_nat = T_wrist_world @ otw
    return _upright_snap(t_nat)


def pose_body(
    rig: SmplhRig,
    arm: dict,
    root_xy: tuple[float, float] = (0.0, 0.0),
    root_yaw_deg: float = 0.0,
    extra_wrist=(0.0, 0.0, 0.0),
    root_z: float = 0.0,
    hand_close: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose the body with the given arm, then place it at world (x,y,z) with heading yaw via the root rule.

    The SMPL+H body has NO root Xform -- fwd() returns V,J posed about the origin facing -Y. To
    move it to world (x,y,z,yaw) we apply, AFTER fwd: Vw = (Rz(yaw)@V.T).T + [x,y,z] (and the same
    for J). yaw=0 keeps the native -Y forward (the straight A->B walk). root_z is the vertical lift:
    a stand offset puts the feet on the floor, and a transient DIP (root_z below stand) lowers the
    whole upright body so the FROZEN-arm grasp seat descends to a cup resting on the table -- a natural
    "lean down to the surface" in FPV (legs are out of frame), with the grip never re-IK'd. Returns
    world-space (V,J).
    """
    pose = assemble_pose(rig, arm, extra_wrist, hand_close)
    V, J = H.fwd(pose)
    if abs(root_yaw_deg) > 1e-9:
        a = np.deg2rad(root_yaw_deg)
        c, s = np.cos(a), np.sin(a)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        V = V @ Rz.T
        J = J @ Rz.T
    off = np.array([root_xy[0], root_xy[1], root_z])
    V = V + off
    J = J + off
    return V, J


def lerp_arm(a: dict, b: dict, s: float) -> dict:
    """Per-joint axis-angle linear interpolation of two arm dicts (reach / retract / place).

    s is expected pre-eased (pass smoothstep(t)). Linear interpolation of the axis-angle vectors
    is adequate for these small upper-arm sweeps and avoids any quaternion/USD convention issues.
    """
    s = max(0.0, min(1.0, float(s)))
    out: dict = {}
    keys = set(a) | set(b)
    for k in keys:
        va = np.asarray(a.get(k, b.get(k)), dtype=np.float64)
        vb = np.asarray(b.get(k, a.get(k)), dtype=np.float64)
        out[k] = list(va + (vb - va) * s)
    return out


# ----------------------------------------------------------------------------------------------
# Bake the grasp: derive rest/place arms + the one-time side penetration backoff. Needs the live
# stage + object (so it can measure penetration via trimesh), so it runs AFTER the driver launches.
# ----------------------------------------------------------------------------------------------
# The two rest/place IK targets + solver settings. Module constants (not function literals) so the
# cache key below can cover them: if a future change makes a target or seed vary, the key changes
# with it instead of silently serving the old solve.
# REST: at the right hip, low + back -- clearly away from the cup so the reach reads as a distinct
# beat. PLACE: out front, lowered toward a desk top ~0.75 m in front.
REST_IK_TARGET = (0.12, -0.05, -0.30)
PLACE_IK_TARGET = (0.02, -0.42, 0.18)
REST_PLACE_IK_N = 6000
REST_IK_SEED, PLACE_IK_SEED = 3, 5

# derive_rest_place_arms is a CONSTANT function: it reads nothing off `rig` (it only writes
# arm_rest/arm_place) and every input is a module constant, so its two batched-torch _ik_arm(n=6000)
# solves (~5 s) produce the same answer on every scene build/switch. Cache the canonical result for
# the process and hand out a FRESH copy each call. ISAAC_RESTARM_CACHE=0 disables.
_REST_PLACE_CACHE: dict[tuple, tuple[dict, dict, float, float]] = {}


def _rest_place_cache_key() -> tuple:
    """Everything the derived arms depend on, by VALUE -- a mutated constant can never hit a stale entry.

    _ik_arm reads smplh_fpv's OWN COLLAR/L_ARM/joint indices at call time (not ours), and we build the
    output dicts from ours, so both copies are keyed.
    """

    def frz(d: dict) -> tuple:
        return tuple(sorted((int(k), tuple(float(x) for x in v)) for k, v in d.items()))

    return (
        REST_IK_TARGET,
        PLACE_IK_TARGET,
        REST_PLACE_IK_N,
        REST_IK_SEED,
        PLACE_IK_SEED,
        tuple(float(x) for x in COLLAR),
        frz(L_ARM),
        (int(R_COLLAR), int(R_SHOULDER), int(R_ELBOW), int(WRIST_J)),
        tuple(float(x) for x in H.COLLAR),
        frz(H.L_ARM),
        (int(H.R_COLLAR), int(H.R_SHOULDER), int(H.R_ELBOW), int(H.WRIST_J), int(H.NJ)),
    )


def _solve_rest_place_arms() -> tuple[dict, dict, float, float]:
    """The two one-shot IK solves -> (arm_rest, arm_place, err_rest, err_place). No rig, no state."""
    # _ik_arm reads smplh_fpv.COLLAR + smplh_fpv.L_ARM at call time; they already match ours
    # (same literals). Solve a relaxed-down wrist target and a table-drop target, seeded for variety.
    # ARM_REST: wrist low and at the side (relaxed). Target chosen below chest, right of center.
    sh_r, el_r, err_r = H._ik_arm(
        np.array(REST_IK_TARGET),
        n=REST_PLACE_IK_N,
        seed=REST_IK_SEED,
        horizontal=False,
        arm_dir=None,
    )
    arm_rest = {
        R_COLLAR: list(COLLAR),
        R_SHOULDER: list(sh_r),
        R_ELBOW: list(el_r),
        WRIST_J: [0.0, 0.0, 0.0],
        **{k: list(v) for k, v in L_ARM.items()},
    }
    # ARM_PLACE: wrist forward + low, as if setting the cup down on a table ~0.75 m in front.
    sh_p, el_p, err_p = H._ik_arm(
        np.array(PLACE_IK_TARGET),
        n=REST_PLACE_IK_N,
        seed=PLACE_IK_SEED,
        horizontal=False,
        arm_dir=None,
    )
    arm_place = {
        R_COLLAR: list(COLLAR),
        R_SHOULDER: list(sh_p),
        R_ELBOW: list(el_p),
        WRIST_J: [0.0, 0.0, 0.0],
        **{k: list(v) for k, v in L_ARM.items()},
    }
    return arm_rest, arm_place, float(err_r), float(err_p)


def derive_rest_place_arms(rig: SmplhRig) -> None:
    """Derive ARM_REST (relaxed arm-down) + ARM_PLACE (lowered to a table drop) via ONE _ik_arm each.

    Seeded from the canonical arm region; this is the only place _ik_arm is used in the dynamic
    pipeline, and only for cup-DETACHED endpoints (reach start, place end). The held frames never
    re-IK. _ik_arm needs the module ARM/RHP context of smplh_fpv only for COLLAR/L_ARM, which we
    pass through its own globals -- it reads COLLAR + L_ARM from smplh_fpv at call time, so we mirror
    our constants onto it first.

    The solve is constant (see _REST_PLACE_CACHE), so it runs once per process and every later call
    is served from the cache -- worth ~5 s on every scene build/switch. Callers are free to mutate
    rig.arm_rest / rig.arm_place (and the float lists inside them), so the cache keeps the canonical
    copy private and rebuilds fresh dicts + fresh lists here, exactly as the uncached solve did.
    """
    use_cache = os.getenv("ISAAC_RESTARM_CACHE", "1") != "0"
    key = _rest_place_cache_key() if use_cache else None
    hit = _REST_PLACE_CACHE.get(key) if use_cache else None
    if hit is None:
        t0 = time.perf_counter()
        arm_rest, arm_place, err_r, err_p = _solve_rest_place_arms()
        how = f"one-shot IK in {time.perf_counter() - t0:.2f}s"
        if use_cache:
            _REST_PLACE_CACHE[key] = (arm_rest, arm_place, err_r, err_p)
    else:
        arm_rest, arm_place, err_r, err_p = hit
        how = "CACHE (constant solve; fresh copy, 2x _ik_arm skipped)"
    rig.arm_rest = {k: list(v) for k, v in arm_rest.items()}
    rig.arm_place = {k: list(v) for k, v in arm_place.items()}
    print(
        f"[anim_core] derived ARM_REST (err={err_r:.3f}) + ARM_PLACE (err={err_p:.3f}) via {how} "
        "(cup-detached endpoints only; held frames stay frozen on the canonical arm)"
    )


def swivel_elbow_down(
    rig: SmplhRig, pose: np.ndarray, obj_local: np.ndarray, deg_clip: float = 150.0
) -> np.ndarray:
    """Naturalness: reach-EXACT null-space elbow swivel (rotate the arm about the shoulder->seat axis).

    The side-grip seed reaches the target tightly but leaves the elbow "winged up" and the forearm
    wrenched (its ~1.77 rad internal shoulder rotation is ~invisible to a position-only CCD, so it
    survives). The one redundant DOF of the arm is the ELBOW SWIVEL: with the shoulder and the seat
    fixed, the elbow rides a circle about the shoulder->seat line. Rotating the whole arm about THAT
    line moves the elbow to a human "down (+ slightly out/back)" position WITHOUT moving the seat --
    both the shoulder joint and the seat lie on the axis -- so the reach target is preserved exactly.
    Returns the swiveled 156-D pose (only R_SHOULDER's local rotation changes; the distal chain rides
    it rigidly). ``obj_local`` is the body-local seat target (== warp's converged seat position).
    """
    p = np.asarray(pose, dtype=np.float32).copy()
    J = H.fwd_joints(p)
    sh = J[R_SHOULDER].astype(np.float64)
    axis = (
        np.asarray(obj_local, dtype=np.float64) - sh
    )  # shoulder -> seat (== reach target)
    an = float(np.linalg.norm(axis))
    if an < 1e-6:
        return p
    axis = axis / an
    ev = J[R_ELBOW].astype(np.float64) - sh
    v_cur = (
        ev - float(np.dot(ev, axis)) * axis
    )  # elbow offset in the plane perpendicular to the axis
    vn = float(np.linalg.norm(v_cur))
    if (
        vn < 1e-4
    ):  # elbow ~on the shoulder->seat line (straight arm): swivel is ill-defined
        return p
    v_cur = v_cur / vn
    # Target elbow direction: mostly DOWN (world -Z), slightly OUT (+x = body-right) and BACK (+y; the
    # body faces -y), projected into the plane perpendicular to the swivel axis.
    d = np.array([0.25, 0.20, -1.0])
    d = d - float(np.dot(d, axis)) * axis
    dn = float(np.linalg.norm(d))
    if dn < 1e-4:
        return p
    d = d / dn
    phi_raw = float(
        np.degrees(
            np.arctan2(float(np.dot(np.cross(v_cur, d), axis)), float(np.dot(v_cur, d)))
        )
    )
    phi = float(np.clip(phi_raw, -deg_clip, deg_clip))
    print(
        f"[swivel] phi_raw={phi_raw:+.0f}deg -> phi={phi:+.0f}deg (shortest arc to elbow-down)"
    )
    r_l = H.axis_rot(
        axis, phi
    )  # world rotation about the shoulder->seat line (through the shoulder)
    g_col = H.global_rot(p, R_COLLAR)  # shoulder's parent global frame
    r_sh = _rodrigues(p[R_SHOULDER * 3 : R_SHOULDER * 3 + 3])
    new_sh = (
        g_col.T @ r_l @ g_col
    ) @ r_sh  # compose the world swivel onto the shoulder's local rot
    p[R_SHOULDER * 3 : R_SHOULDER * 3 + 3] = H.logmap(new_sh).astype(np.float32)
    return p


def bake_side_backoff(
    rig: SmplhRig,
    place_and_measure,
    pen_tol_mm: float = 3.0,
    max_steps: int = 16,
) -> float:
    """Run the SIDE penetration backoff ONCE and FOLD the constant offset into obj_T_in_wrist_baked.

    Replicates smplh_fpv.py's SIDE seat: pose the body at root (0,0,0) with the frozen canonical arm,
    seat the cup, then back it OUTWARD along the palm normal (wrist-frame +Z) in 0.5 mm steps while
    that REALLY reduces the arm+hand penetration, accepting up to pen_tol_mm of (invisible) contact
    and hard-capping the backoff so a tight wrap never opens a gap. The resulting scalar is folded
    into obj_T_in_wrist_baked along wrist-frame +Z, so per-frame seating is invariant (never re-run).

    ``place_and_measure(T_world_colvec, Vq)`` is provided by the driver (it owns the USD object +
    sim + trimesh); it sets the object's USD matrix and returns (mesh, max_pen_mm of Vq into it).
    Returns the baked backoff in meters.
    """
    # Pose at the origin with the frozen arm; build the hand + arm+hand vertex masks (as smplh_fpv).
    V, J = pose_body(rig, rig.arm_grasp, (0.0, 0.0), 0.0)
    W = H.TPL[5].numpy()
    hand_mask = W[:, [WRIST_J] + list(range(37, 52))].sum(1) > 0.5
    armhand_mask = W[:, [R_ELBOW, WRIST_J] + list(range(37, 52))].sum(1) > 0.5
    Vhand, Varmhand = V[hand_mask], V[armhand_mask]

    R_w, _ = H.wrist_frame(J)
    pn_world = R_w[:, 2] / (
        np.linalg.norm(R_w[:, 2]) + 1e-9
    )  # palm normal = wrist-frame +Z in world

    # Start from the un-baked seat; iterate the SAME backoff loop as smplh_fpv's SIDE branch.
    base = rig.obj_T_in_wrist.copy()
    t_obj = seat_object_from(rig, J, base)
    obj0 = t_obj[:3, 3].copy()
    _, pen_hand0 = place_and_measure(t_obj, Vhand)
    _, pen_arm = place_and_measure(t_obj, Varmhand)
    pen_arm0 = pen_arm
    best_pen, best_t = pen_arm, t_obj[:3, 3].copy()
    for _ in range(max_steps):  # absolute cap (max_steps * 0.5 mm)
        if pen_arm <= pen_tol_mm:
            break
        t_obj[:3, 3] = t_obj[:3, 3] + pn_world * 0.0005
        _, pen_arm = place_and_measure(t_obj, Varmhand)
        if pen_arm < best_pen - 0.05:
            best_pen, best_t = pen_arm, t_obj[:3, 3].copy()
        else:
            break
    backoff_m = float(np.linalg.norm(best_t - obj0))

    # FOLD the backoff into obj_T_in_wrist_baked along wrist-frame +Z. Because the upright snap
    # preserves the (post-fold) position and T_wrist_world rotates wrist +Z -> world palm normal,
    # folding +backoff*[0,0,1] into obj_T_in_wrist reproduces best_t at root and rides rigidly per frame.
    baked = rig.obj_T_in_wrist.copy()
    baked[:3, 3] = baked[:3, 3] + backoff_m * np.array([0.0, 0.0, 1.0])
    rig.obj_T_in_wrist_baked = baked
    print(
        f"[anim_core] side backoff baked: {backoff_m * 1000:.1f}mm  init pen hand={pen_hand0:.1f}mm "
        f"armhand={pen_arm0:.1f}mm -> {best_pen:.2f}mm (folded into obj_T_in_wrist_baked)"
    )
    return backoff_m


def seat_object_from(rig: SmplhRig, J: np.ndarray, otw: np.ndarray) -> np.ndarray:
    """Seat helper used by the bake loop: same as seat_object but with an explicit obj_T_in_wrist."""
    R_w, wrist_o = H.wrist_frame(J)
    T_wrist_world = np.eye(4)
    T_wrist_world[:3, :3] = R_w
    T_wrist_world[:3, 3] = wrist_o
    return _upright_snap(T_wrist_world @ otw)


# ----------------------------------------------------------------------------------------------
# USD authoring (pxr imported lazily; called only after launch_app).
# ----------------------------------------------------------------------------------------------
def author_body_once(
    stage: "Usd.Stage", rig: SmplhRig, V: np.ndarray, body_path: str = "/World/Body"
) -> None:
    """Define the /World/Body mesh + a skin material ONCE (never re-Define; update points per frame).

    Mirrors smplh_fpv.author_body but caches the mesh + the normals primvar on the rig so update_body
    can rewrite points + normals each frame without re-creating the prim (or re-binding the material).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    faces = rig.faces
    Vd = np.asarray(V, dtype=np.float64)
    mesh = UsdGeom.Mesh.Define(stage, body_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in Vd])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    nrm = H.vertex_normals(Vd, faces)
    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.vertex
    )
    pv.Set([Gf.Vec3f(float(x[0]), float(x[1]), float(x[2])) for x in nrm])
    mesh.CreateNormalsAttr(
        [Gf.Vec3f(float(x[0]), float(x[1]), float(x[2])) for x in nrm]
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mtl = UsdShade.Material.Define(stage, body_path + "/Mat")
    sh = UsdShade.Shader.Define(stage, body_path + "/Mat/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.86, 0.68, 0.57))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mtl)
    rig._body_mesh = mesh
    rig._normals_pv = pv


def update_body(rig: SmplhRig, V: np.ndarray) -> None:
    """Per-frame: rewrite the /World/Body points + recompute the vertex-normals primvar (LBS changes them).

    In the physics path (``_refetch_handles``) the mesh + normals primvar are re-acquired fresh each
    call (cached handles stale under PhysX); otherwise the cached handles are reused (the fast path).
    """
    from pxr import Gf, UsdGeom

    if rig._refetch_handles and rig._stage is not None:
        mesh = UsdGeom.Mesh(rig._stage.GetPrimAtPath(rig._body_path))
        rig._body_mesh = mesh
        rig._normals_pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("normals")
    if rig._body_mesh is None:
        raise RuntimeError("update_body called before author_body_once")
    Vd = np.asarray(V, dtype=np.float64)
    rig._body_mesh.GetPointsAttr().Set(
        [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in Vd]
    )
    nrm = H.vertex_normals(Vd, rig.faces)
    nl = [Gf.Vec3f(float(x[0]), float(x[1]), float(x[2])) for x in nrm]
    rig._normals_pv.Set(nl)
    rig._body_mesh.GetNormalsAttr().Set(nl)


def build_object(
    stage: "Usd.Stage", rig: SmplhRig, obj_path: str = "/World/Obj"
) -> object:
    """Build the cup geometry from build_params (the EXACT mug the grasp was solved against).

    Returns the MakeMatrixXform op (cached on the rig) so attach_obj/place_obj_resting can write the
    object's world matrix each frame. For the cup this references build['usd'] (SM_Mug_A2.usd).
    """
    from pxr import UsdGeom

    anchor = stage.DefinePrim(obj_path, "Xform")
    axf = UsdGeom.Xformable(anchor).MakeMatrixXform()
    shape = rig.build["shape"]
    if shape == "usd_ref":
        model = stage.GetPrimAtPath(obj_path + "/m")
        if model is None or not model.IsValid():
            model = stage.DefinePrim(obj_path + "/m", "Xform")
            model.GetReferences().AddReference(rig.build["usd"])
        raw_scale = rig.build.get("scale", 1.0)
        if isinstance(raw_scale, (list, tuple)):
            if len(raw_scale) != 3:
                raise ValueError(
                    f"asset scale must be a scalar or [x, y, z], got {raw_scale!r}"
                )
            scale_xyz = tuple(float(value) for value in raw_scale)
            if any(abs(value - 1.0) > 1e-9 for value in scale_xyz):
                UsdGeom.Xformable(model).AddScaleOp(opSuffix="taskScaleXYZ").Set(
                    scale_xyz
                )
        else:
            scale = float(raw_scale)
            if abs(scale - 1.0) > 1e-9:
                UsdGeom.Xformable(model).AddScaleOp(opSuffix="taskScale").Set(
                    (scale, scale, scale)
                )
    elif shape == "sphere":
        sp = UsdGeom.Sphere.Define(stage, obj_path + "/m")
        sp.GetRadiusAttr().Set(float(rig.build["radius"]))
        sp.GetDisplayColorAttr().Set([(0.78, 0.10, 0.10)])
    elif shape == "cylinder":
        import trimesh
        from pxr import Gf

        pm = trimesh.creation.cylinder(
            radius=float(rig.build["radius"]),
            height=float(rig.build["height"]),
            sections=int(rig.build.get("sections", 96)),
        )
        ms = UsdGeom.Mesh.Define(stage, obj_path + "/m")
        ms.CreatePointsAttr([Gf.Vec3f(*v) for v in pm.vertices])
        ms.CreateFaceVertexCountsAttr([3] * len(pm.faces))
        ms.CreateFaceVertexIndicesAttr(pm.faces.flatten().tolist())
        ms.CreateDisplayColorAttr([(0.20, 0.45, 0.85)])
    elif shape == "mesh":
        # Any-object: load an ARBITRARY triangle mesh (.obj/.ply/.stl path, a loaded USD, or an
        # explicit {vertices, faces}) as a UsdGeom.Mesh -- which _make_sd_world's extraction already
        # handles. Watertight repair (fill_holes) helps the SDF voxelization on scanned/exported meshes.
        import trimesh
        from pxr import Gf

        src = rig.build["mesh"]
        if isinstance(src, str) and src.lower().endswith(
            (".usd", ".usda", ".usdc", ".usdz")
        ):
            stage.DefinePrim(obj_path + "/m", "Xform").GetReferences().AddReference(src)
        else:
            pm = (
                trimesh.load(src, force="mesh")
                if isinstance(src, str)
                else trimesh.Trimesh(vertices=src["vertices"], faces=src["faces"])
            )
            if not pm.is_watertight:
                pm.fill_holes()
            ms = UsdGeom.Mesh.Define(stage, obj_path + "/m")
            ms.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in pm.vertices])
            ms.CreateFaceVertexCountsAttr([3] * len(pm.faces))
            ms.CreateFaceVertexIndicesAttr(np.asarray(pm.faces).flatten().tolist())
            ms.CreateDisplayColorAttr([(0.30, 0.55, 0.85)])
    else:
        raise ValueError("unknown build shape " + shape)
    rig._obj_xf = axf
    return axf


def set_object_world(rig: SmplhRig, T_obj_world: np.ndarray) -> None:
    """Write the cup's world matrix (column-vector T -> USD row-vector transpose). The TRANSPOSE is load-bearing.

    In the physics path (``_refetch_handles``) the op is re-acquired fresh each call (cached handles
    stale under PhysX); otherwise the cached op from build_object is reused (the proven fast path).
    """
    from pxr import Gf, UsdGeom

    if rig._refetch_handles and rig._stage is not None:
        # RE-ACQUIRE the op, but do not RE-AUTHOR it. ``MakeMatrixXform`` clears the prim's whole
        # xformOpOrder and adds a fresh op -- every frame, on a prim that has other children. Reuse
        # the transform op when one is already there and only build one when it is missing, so the
        # per-frame write is a value change rather than a re-authoring of the transform stack.
        xf = UsdGeom.Xformable(rig._stage.GetPrimAtPath(rig._obj_path))
        ops = [
            op
            for op in xf.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTransform
        ]
        rig._obj_xf = ops[0] if len(ops) == 1 else xf.MakeMatrixXform()
    if rig._obj_xf is None:
        raise RuntimeError("set_object_world called before build_object")
    rig._obj_xf.Set(Gf.Matrix4d(*[float(x) for x in T_obj_world.T.flatten()]))


def enable_physics_refetch(
    rig: SmplhRig,
    stage: "Usd.Stage",
    obj_path: str = "/World/Obj",
    body_path: str = "/World/Body",
) -> None:
    """Turn ON per-frame handle re-acquisition for the physics path (idempotent).

    PhysX simulation + the collision detector's stage attach/detach + its session-layer edits STALE
    previously-cached USD schema handles, so the next ``.Set`` raises 'Accessed schema on invalid
    prim'. Freshly-fetched handles always work; flipping this flag makes set_object_world / update_body
    re-acquire their handles each call. The non-physics path never sets it, so its fast cached path is
    untouched."""
    rig._stage = stage
    rig._obj_path = obj_path
    rig._body_path = body_path
    rig._refetch_handles = True


def attach_obj(rig: SmplhRig, J: np.ndarray) -> np.ndarray:
    """Attach the cup to the wrist this frame: seat via the baked transform + write the USD matrix.

    Returns the world transform (so the caller can read obj_ctr for the camera aim).
    """
    T = seat_object(rig, J, baked=True)
    set_object_world(rig, T)
    return T


def place_obj_resting(rig: SmplhRig, pos_xyz, x_heading=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Place the cup UPRIGHT at a resting world position (on a table); detached from the hand.

    Used for the idle (resting on table A) + release (resting on table B) states. Heading is a
    cosmetic horizontal x-axis; the cup stands exactly upright (+Z).
    """
    z = np.array([0.0, 0.0, 1.0])
    x = np.asarray(x_heading, dtype=np.float64)
    x = x - np.dot(x, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    set_object_world(rig, T)
    return T


def local_obj_min_z(rig: SmplhRig) -> float:
    """Local min-z of the cup geometry (so its base seats on a table top). Sphere/cylinder analytic.

    For a usd_ref mug the base pivot is ~0; we return the bbox min-z of the reconstructed object via
    its build params. The mugs are base-pivot (min_z~=0), so a cup at z = table_top rests on the top;
    we expose a tiny helper for symmetry with fpv_scene.local_bbox_min_z.
    """
    shape = rig.build["shape"]
    if shape == "sphere":
        return -float(rig.build["radius"])
    if shape == "cylinder":
        return -0.5 * float(rig.build["height"])
    return 0.0  # usd_ref mugs are base-pivot (min_z ~ 0)


# ----------------------------------------------------------------------------------------------
# Natural reach-DOWN grasp pose: the arm articulates to a cup on the table, but with the wrist
# rotated so the grip + upright cup match the LOCKED side-grip at the moment of pickup. (The arm
# should move like a real person's; only the pickup/carry grip must match the side-grip.)
# ----------------------------------------------------------------------------------------------
def _rodrigues(aa) -> np.ndarray:
    """Axis-angle (3,) -> 3x3 rotation matrix (numpy)."""
    aa = np.asarray(aa, dtype=np.float64)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    return H.axis_rot(aa / th, np.rad2deg(th))


def canonical_wrist_orient(rig: SmplhRig) -> np.ndarray:
    """R_canon: the geometric wrist frame of the FROZEN canonical side arm (the reference palm)."""
    _, Jc = pose_body(rig, rig.arm_grasp)
    R_w, _ = H.wrist_frame(Jc)
    return R_w


def canonical_seat_local(rig: SmplhRig) -> np.ndarray:
    """Cup seat (baked, upright) for the canonical arm at the origin -> world [dx, dy, seat_z]."""
    _, Jc = pose_body(rig, rig.arm_grasp)
    return seat_object(rig, Jc, baked=True)[:3, 3].copy()


def _wrist_frame_batch(Jb: np.ndarray) -> np.ndarray:
    """Vectorized H.wrist_frame over a batch of posed joints Jb (n,NJ,3) -> R_w (n,3,3) (world<-wrist).

    Identical construction to H.wrist_frame (palm dir, index-pinky axis, palm normal), batched so the
    reach IK can score each candidate's natural hand orientation against the canonical side-grip."""

    def _nz(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    wrist = Jb[:, WRIST_J]
    palm = np.stack([Jb[:, H.MCP[f]] for f in H.FINGERS], axis=1).mean(1)
    pd = _nz(palm - wrist)
    ac = _nz(Jb[:, H.MCP["index"]] - Jb[:, H.MCP["pinky"]])
    pn = _nz(np.cross(pd, ac))
    tip = np.stack([Jb[:, H.TIPISH[f]] for f in H.FINGERS], axis=1).mean(1)
    pn[np.sum((tip - palm) * pn, axis=1) < 0] *= -1.0
    ex = ac.copy()
    ey = _nz(pd - np.sum(pd * ex, axis=1, keepdims=True) * ex)
    ez = np.cross(ex, ey)
    ex[np.sum(ez * pn, axis=1) < 0] *= -1.0
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez], axis=2)


def _ik_reach(
    rig: SmplhRig,
    wrist_target: np.ndarray,
    r_canon: np.ndarray,
    n: int = 18000,
    seed: int = 11,
    spread: float = 1.3,
) -> tuple[list, list, float]:
    """Batched shoulder+elbow search (PERTURBED AROUND the canonical arm) that drives J[WRIST_J] ->
    wrist_target AND prefers a natural hand orientation already near r_canon, so the wrist needs only a
    SMALL bend to land the exact side-grip. Two-stage coarse->fine. Returns (shoulder_aa, elbow_aa,
    position_err_m)."""
    import torch

    def _search(sh0, el0, spr, nn, sd):
        rng = np.random.default_rng(sd)
        sh = np.asarray(sh0) + rng.uniform(-spr, spr, size=(nn, 3))
        el = np.asarray(el0) + rng.uniform(-spr, spr, size=(nn, 3))
        pb = np.zeros((nn, NJ * 3), dtype=np.float32)
        pb[:, 0:3] = [np.pi / 2.0, 0.0, 0.0]
        for j, v in {R_COLLAR: COLLAR, **L_ARM}.items():
            pb[:, j * 3 : j * 3 + 3] = v
        pb[:, R_SHOULDER * 3 : R_SHOULDER * 3 + 3] = sh
        pb[:, R_ELBOW * 3 : R_ELBOW * 3 + 3] = el
        pb[:, RHAND_LO : RHAND_LO + 45] = (
            rig.rhp
        )  # fingers: needed for the wrist-frame orientation
        _, Jb = H.lbs(torch.zeros(nn, 10), torch.tensor(pb), *H.TPL, pose2rot=True)
        Jb = Jb.detach().numpy()
        pos = np.linalg.norm(Jb[:, WRIST_J] - wrist_target, axis=1)
        orient = np.linalg.norm(_wrist_frame_batch(Jb) - r_canon[None], axis=(1, 2))
        elbow_pen = np.maximum(
            0.0, Jb[:, R_ELBOW, 0] - 0.05
        )  # don't cross the body center (+x)
        # Position FIRST: keep the candidates within a tight position band, then among those minimize
        # the natural-orientation gap (-> small wrist bend) plus a light elbow penalty.
        thr = max(0.012, float(np.percentile(pos, 2)))
        idx = np.where(pos <= thr)[0]
        if idx.size == 0:
            idx = np.argsort(pos)[:200]
        r = idx[int(np.argmin(orient[idx] + 0.3 * elbow_pen[idx]))]
        return list(sh[r]), list(el[r]), float(pos[r])

    sh, el, err = _search(R_SHOULDER_AA, R_ELBOW_AA, spread, n, seed)
    sh, el, err = _search(
        sh, el, 0.35, max(8000, n // 2), seed + 1
    )  # fine refine around the best
    return sh, el, err


def _solve_arm_to(
    rig: SmplhRig,
    wrist_target: np.ndarray,
    r_canon: np.ndarray,
    n: int = 16000,
    seed: int = 11,
    iters: int = 4,
) -> tuple[dict, float, float, float]:
    """Solve an arm that puts J[WRIST_J] at wrist_target (body-local) with the wrist frame == r_canon.

    Stage 1: orientation-aware shoulder+elbow IK for the wrist position. Stage 2: rotate the wrist joint
    so the geometric wrist frame == r_canon (closed-form via the elbow global frame, refined against the
    geometric wrist_frame). Returns (arm_dict_with_wrist, ik_err_m, orient_err_frob, wrist_bend_deg)."""
    sh, el, ik_err = _ik_reach(rig, wrist_target, r_canon, n=n, seed=seed)
    arm = {
        R_COLLAR: list(COLLAR),
        R_SHOULDER: list(sh),
        R_ELBOW: list(el),
        WRIST_J: [0.0, 0.0, 0.0],
        **{k: list(v) for k, v in L_ARM.items()},
    }
    rw = np.eye(3)  # current wrist-joint rotation matrix (starts at identity)
    orient_err = float("nan")
    for _ in range(iters):
        arm[WRIST_J] = list(H.logmap(rw))
        pose = assemble_pose(rig, arm)
        J = H.fwd_joints(pose)
        r_w, _ = H.wrist_frame(J)
        orient_err = float(np.linalg.norm(r_w - r_canon))
        if orient_err < 1e-3:
            break
        g_el = H.global_rot(
            pose, R_ELBOW
        )  # elbow global frame (the wrist joint's parent)
        e_world = r_canon @ r_w.T  # world-space correction to apply to the hand
        rw = (
            g_el.T @ e_world @ g_el
        ) @ rw  # express in the elbow-local frame, compose onto wrist
    arm[WRIST_J] = list(H.logmap(rw))
    wrist_bend = float(np.rad2deg(np.linalg.norm(H.logmap(rw))))
    return arm, float(ik_err), orient_err, wrist_bend


def solve_wrist_orient(
    rig: SmplhRig,
    base_pose: np.ndarray,
    r_canon: np.ndarray,
    iters: int = 4,
) -> tuple[np.ndarray, float, float]:
    """Solve ONLY the wrist joint (WRIST_J=21) of a full 156-D ``base_pose`` so the geometric wrist
    frame == ``r_canon``, leaving the GRAB shoulder/elbow/spine/global UNTOUCHED.

    This is the wrist-orient loop EXTRACTED from ``_solve_arm_to``, but it keeps the rest of an
    already-posed body (the retargeted GRAB arm) fixed and rotates only the wrist joint to land the
    object's side-grip wrist frame. Closed-form in the wrist's parent (elbow) global frame, refined
    against the geometric ``H.wrist_frame``. The GRAB fingers in ``base_pose`` do not change the
    wrist-frame construction enough to matter (the caller overwrites the fingers separately), but we
    use the SAME pose for the FK so the frame is consistent. Returns
    (wrist_aa (3,), orient_err_frob, wrist_bend_deg). ``wrist_bend_deg`` is the magnitude of the wrist
    correction relative to the GRAB wrist (a large value -- see the >70deg note -- means the GRAB
    forearm roll is far from the side-grip and the wrist must twist hard to land it)."""
    pose = np.asarray(base_pose, dtype=np.float32).copy()
    # The wrist joint already carries the GRAB local rotation; we COMPOSE an extra rotation `rw` onto
    # it (world-correction expressed in the elbow-local frame) so r_w -> r_canon, starting from the
    # GRAB wrist (rw = identity == leave the GRAB wrist as-is).
    base_wrist_aa = pose[WRIST_J * 3 : WRIST_J * 3 + 3].astype(np.float64).copy()
    base_wrist_R = _rodrigues(base_wrist_aa)
    rw = np.eye(3)
    orient_err = float("nan")
    for _ in range(iters):
        wrist_R = rw @ base_wrist_R
        pose[WRIST_J * 3 : WRIST_J * 3 + 3] = H.logmap(wrist_R)
        J = H.fwd_joints(pose)
        r_w, _ = H.wrist_frame(J)
        orient_err = float(np.linalg.norm(r_w - r_canon))
        if orient_err < 1e-3:
            break
        g_el = H.global_rot(pose, R_ELBOW)  # the wrist joint's parent global frame
        e_world = r_canon @ r_w.T  # world-space correction to apply to the hand
        rw = (g_el.T @ e_world @ g_el) @ rw  # express in the elbow-local frame, compose
    wrist_R = rw @ base_wrist_R
    wrist_aa = H.logmap(wrist_R)
    pose[WRIST_J * 3 : WRIST_J * 3 + 3] = wrist_aa
    J = H.fwd_joints(pose)
    orient_err = float(np.linalg.norm(H.wrist_frame(J)[0] - r_canon))
    # wrist_bend = the size of the correction RELATIVE to the GRAB wrist (how hard the wrist twisted).
    wrist_bend = float(np.rad2deg(np.linalg.norm(H.logmap(rw))))
    return wrist_aa.astype(np.float64), orient_err, wrist_bend


def solve_grasp_arm(
    rig: SmplhRig, cup_local_pos, n: int = 16000, seed: int = 11, iters: int = 4
) -> tuple[dict, float, float, float]:
    """Solve a reach-DOWN arm whose grip seat lands the cup at cup_local_pos (body-local) with the
    CANONICAL wrist orientation, so the side-grip + upright cup match the reference AT PICKUP."""
    r_canon = canonical_wrist_orient(rig)
    otw = (
        rig.obj_T_in_wrist_baked
        if rig.obj_T_in_wrist_baked is not None
        else rig.obj_T_in_wrist
    )
    wrist_target = np.asarray(cup_local_pos, dtype=np.float64) - r_canon @ otw[:3, 3]
    return _solve_arm_to(rig, wrist_target, r_canon, n=n, seed=seed, iters=iters)


def solve_reach_path(
    rig: SmplhRig, rest_arm: dict, grasp_arm: dict, n_steps: int = 6
) -> list[dict]:
    """Monotonic OPEN-HAND reach: the wrist travels in a straight body-local line from the relaxed rest
    pose to the grasp wrist position, with a STRAIGHT wrist (joint 0) at EVERY waypoint -- the side-grip
    wrist rotation + finger curl engage only at contact (the grasp segment), so the approach shows a
    naturally extended hand, not a pre-curled claw. Replaying it (lerp adjacent poses, render with
    hand_close=0) avoids the up-and-over swing a single big axis-angle lerp produces. Returns
    [rest, p1, ..., open_grasp] (length n_steps); the last == grasp_arm's shoulder/elbow with wrist 0."""
    r_canon = canonical_wrist_orient(rig)
    w_rest = pose_body(rig, rest_arm)[1][WRIST_J].astype(np.float64).copy()
    w_grasp = pose_body(rig, grasp_arm)[1][WRIST_J].astype(np.float64).copy()
    open_grasp = {**grasp_arm, WRIST_J: [0.0, 0.0, 0.0]}
    path = [{**rest_arm, WRIST_J: [0.0, 0.0, 0.0]}]
    for i in range(1, n_steps):
        if i == n_steps - 1:
            path.append(open_grasp)
            break
        wt = w_rest + (w_grasp - w_rest) * (i / (n_steps - 1))
        arm, _, _, _ = _solve_arm_to(rig, wt, r_canon, n=10000, seed=20 + i)
        arm[WRIST_J] = [
            0.0,
            0.0,
            0.0,
        ]  # OPEN reach: straight wrist (grip wrist engages at contact)
        path.append(arm)
    return path


# ----------------------------------------------------------------------------------------------
# cup-first reach WARP: rotate ONLY the right shoulder+elbow of a GRAB pose (closed-form CCD, no
# search) so the cup SEAT lands at an INDEPENDENTLY-chosen world position. This is what makes the
# pickup object-driven ("place the cup, the hand goes to it") instead of hand-driven. Cheap enough
# to run per frame; here we run it ONCE at the grasp frame and ramp the joint-space delta in.
# ----------------------------------------------------------------------------------------------
def _perp(a: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to ``a`` (for the 180deg degenerate case of _rot_between)."""
    t = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(a[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    p = np.cross(a, t)
    return p / (np.linalg.norm(p) + 1e-9)


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 shortest-arc rotation taking unit(a) -> unit(b). Identity if already aligned."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if c > 0 else H.axis_rot(_perp(a), 180.0)
    return H.axis_rot(v / s, float(np.rad2deg(np.arctan2(s, c))))


def batch_reach_enabled() -> bool:
    """Whether planner call sites should use the batched reach backend."""
    return os.getenv("ISAAC_BATCH_IK", "1") != "0"


def _batch_ik_device() -> torch.device:
    requested = os.getenv("ISAAC_BATCH_IK_DEVICE")
    if not requested:
        assigned_gpu = (os.getenv("ISAAC_GPUS") or "").split(",", 1)[0].strip()
        requested = (
            f"cuda:{assigned_gpu}"
            if assigned_gpu.isdigit() and torch.cuda.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _reach_cache_key(
    rig: SmplhRig,
    side_pose: np.ndarray,
    obj_local: np.ndarray,
    chain,
) -> tuple:
    otw = (
        rig.obj_T_in_wrist_baked
        if rig.obj_T_in_wrist_baked is not None
        else rig.obj_T_in_wrist
    )

    def packed(value) -> bytes:
        return (
            np.round(np.asarray(value, dtype=np.float64), 6)
            .astype(np.float32)
            .tobytes()
        )

    return (
        packed(side_pose),
        packed(obj_local),
        packed(rig.rhp),
        packed(otw),
        tuple((int(j), int(parent)) for j, parent in chain),
        bool(getattr(rig, "natural_arm", False)),
    )


def _cache_reach(
    rig: SmplhRig, key: tuple, pose: np.ndarray, err: float, joints: np.ndarray
) -> None:
    cache = rig._reach_pose_cache
    cache[key] = (pose.copy(), float(err), joints.copy())
    limit = int(os.getenv("ISAAC_BATCH_IK_CACHE_SIZE", "4096"))
    while len(cache) > max(1, limit):
        cache.pop(next(iter(cache)))


def _normalize_batch(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value / norm.clamp_min(1e-9), norm.squeeze(-1)


def _seat_positions_batch(rig: SmplhRig, joints: torch.Tensor) -> torch.Tensor:
    wrist = joints[:, WRIST_J]
    mcp_ids = [H.MCP[finger] for finger in H.FINGERS]
    tip_ids = [H.TIPISH[finger] for finger in H.FINGERS]
    palm = joints[:, mcp_ids].mean(dim=1)
    pd, _ = _normalize_batch(palm - wrist)
    ac, _ = _normalize_batch(joints[:, H.MCP["index"]] - joints[:, H.MCP["pinky"]])
    pn, _ = _normalize_batch(torch.linalg.cross(pd, ac, dim=-1))
    fingertip = joints[:, tip_ids].mean(dim=1)
    pn_sign = torch.where(
        ((fingertip - palm) * pn).sum(dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(pn[:, :1]),
        torch.ones_like(pn[:, :1]),
    )
    pn = pn * pn_sign
    ex = ac
    ey, _ = _normalize_batch(pd - (pd * ex).sum(dim=-1, keepdim=True) * ex)
    ez = torch.linalg.cross(ex, ey, dim=-1)
    frame_sign = torch.where(
        (ez * pn).sum(dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(ez[:, :1]),
        torch.ones_like(ez[:, :1]),
    )
    ex = ex * frame_sign
    ez = ez * frame_sign
    otw = (
        rig.obj_T_in_wrist_baked
        if rig.obj_T_in_wrist_baked is not None
        else rig.obj_T_in_wrist
    )
    offset = torch.as_tensor(otw[:3, 3], dtype=joints.dtype, device=joints.device)
    return wrist + ex * offset[0] + ey * offset[1] + ez * offset[2]


def _point_rotation_jacobian(point: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
    x, y, z = (point - joint).unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-1,
    )


def _rot_between_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_unit, a_norm = _normalize_batch(a)
    b_unit, b_norm = _normalize_batch(b)
    cross = torch.linalg.cross(a_unit, b_unit, dim=-1)
    axis, sine = _normalize_batch(cross)
    cosine = (a_unit * b_unit).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.atan2(sine, cosine)
    antiparallel = (sine < 1e-7) & (cosine < 0.0)
    fallback = torch.zeros_like(axis)
    fallback[:, 0] = 1.0
    use_y = a_unit[:, 0].abs() >= 0.9
    fallback[use_y, 0] = 0.0
    fallback[use_y, 1] = 1.0
    fallback, _ = _normalize_batch(torch.linalg.cross(a_unit, fallback, dim=-1))
    axis = torch.where(antiparallel[:, None], fallback, axis)
    angle = torch.where(antiparallel, torch.full_like(angle, torch.pi), angle)
    invalid = (a_norm < 1e-9) | (b_norm < 1e-9) | ((sine < 1e-7) & (cosine >= 0.0))
    angle = torch.where(invalid, torch.zeros_like(angle), angle)
    return H.batch_rodrigues(axis * angle[:, None])


def _apply_world_rotation_updates(
    rotations: torch.Tensor,
    global_rotations: torch.Tensor,
    updates: torch.Tensor,
    chain,
    active: torch.Tensor,
) -> None:
    delta_world = H.batch_rodrigues(updates.reshape(-1, 3)).reshape(
        rotations.shape[0], len(chain), 3, 3
    )
    for index, (joint, parent) in enumerate(chain):
        parent_global = (
            global_rotations[parent]
            if isinstance(global_rotations, dict)
            else global_rotations[:, parent]
        )
        delta_local = (
            parent_global.transpose(1, 2) @ delta_world[:, index] @ parent_global
        )
        proposed = delta_local @ rotations[:, joint]
        rotations[:, joint] = torch.where(
            active[:, None, None], proposed, rotations[:, joint]
        )


def _swivel_elbow_down_batch(
    rotations: torch.Tensor,
    joints: torch.Tensor,
    global_rotations: torch.Tensor,
    obj_local: torch.Tensor,
    deg_clip: float = 150.0,
) -> None:
    shoulder = joints[:, R_SHOULDER]
    axis, axis_norm = _normalize_batch(obj_local - shoulder)
    elbow_vec = joints[:, R_ELBOW] - shoulder
    elbow_perp, elbow_norm = _normalize_batch(
        elbow_vec - (elbow_vec * axis).sum(dim=-1, keepdim=True) * axis
    )
    desired = torch.tensor([0.25, 0.20, -1.0], dtype=joints.dtype, device=joints.device)
    desired = desired.expand_as(axis)
    desired, desired_norm = _normalize_batch(
        desired - (desired * axis).sum(dim=-1, keepdim=True) * axis
    )
    sine = (torch.linalg.cross(elbow_perp, desired, dim=-1) * axis).sum(dim=-1)
    cosine = (elbow_perp * desired).sum(dim=-1)
    phi = torch.atan2(sine, cosine).clamp(
        -math.radians(deg_clip), math.radians(deg_clip)
    )
    valid = (axis_norm >= 1e-6) & (elbow_norm >= 1e-4) & (desired_norm >= 1e-4)
    delta_world = H.batch_rodrigues(axis * phi[:, None])
    parent_global = global_rotations[:, R_COLLAR]
    delta_local = parent_global.transpose(1, 2) @ delta_world @ parent_global
    proposed = delta_local @ rotations[:, R_SHOULDER]
    rotations[:, R_SHOULDER] = torch.where(
        valid[:, None, None], proposed, rotations[:, R_SHOULDER]
    )


def warp_arms_to_seats(
    rig: SmplhRig,
    base_poses: np.ndarray,
    rotations_world: np.ndarray,
    world_offsets: np.ndarray,
    object_targets_world: np.ndarray,
    *,
    dls_iters: int = 12,
    ccd_cycles: int = 6,
    damping: float = 0.04,
    max_delta_rad: float = 0.35,
    chain: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve many SMPL+H reach targets together and return poses, residuals, and local joints."""
    poses = np.asarray(base_poses, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    rzs = np.asarray(rotations_world, dtype=np.float64)
    offsets = np.asarray(world_offsets, dtype=np.float64)
    targets = np.asarray(object_targets_world, dtype=np.float64)
    batch = poses.shape[0]
    if rzs.shape == (3, 3):
        rzs = np.broadcast_to(rzs, (batch, 3, 3))
    if offsets.shape == (3,):
        offsets = np.broadcast_to(offsets, (batch, 3))
    if targets.shape == (3,):
        targets = np.broadcast_to(targets, (batch, 3))
    if poses.shape != (batch, NJ * 3):
        raise ValueError(f"expected base poses shape (B, {NJ * 3}), got {poses.shape}")
    if (
        rzs.shape != (batch, 3, 3)
        or offsets.shape != (batch, 3)
        or targets.shape != (batch, 3)
    ):
        raise ValueError(
            "batched reach transforms must have shapes (B,3,3), (B,3), and (B,3)"
        )
    if chain is None:
        chain = [(R_ELBOW, R_SHOULDER), (R_SHOULDER, R_COLLAR)]

    side_poses = poses.copy()
    side_poses[:, RHAND_LO : RHAND_LO + 45] = np.asarray(rig.rhp, dtype=np.float32)
    obj_local = np.einsum("bij,bj->bi", np.transpose(rzs, (0, 2, 1)), targets - offsets)
    output_poses = np.empty_like(side_poses)
    output_errors = np.empty(batch, dtype=np.float64)
    output_joints = np.empty((batch, NJ, 3), dtype=np.float32)
    misses = []
    keys = []
    for index in range(batch):
        key = _reach_cache_key(rig, side_poses[index], obj_local[index], chain)
        keys.append(key)
        cached = rig._reach_pose_cache.get(key)
        if cached is None:
            misses.append(index)
            continue
        output_poses[index], output_errors[index], output_joints[index] = cached
    if not misses:
        return output_poses, output_errors, output_joints

    device = _batch_ik_device()
    miss_poses = side_poses[misses]
    miss_targets = obj_local[misses]
    t0 = time.perf_counter()
    with torch.no_grad():
        # Materialize solver inputs directly on the selected device.
        pose_tensor = torch.as_tensor(miss_poses, dtype=torch.float32, device=device)
        target_tensor = torch.as_tensor(
            miss_targets, dtype=torch.float32, device=device
        )
        local_rotations = H.batch_rodrigues(pose_tensor.reshape(-1, 3)).reshape(
            len(misses), NJ, 3, 3
        )
        eye = torch.eye(3, dtype=pose_tensor.dtype, device=device).expand(
            len(misses), -1, -1
        )
        initial_joints, initial_global_rotations = H.fwd_joints_from_rot_mats_batch(
            local_rotations, return_global_rot=True
        )
        initial_seat = _seat_positions_batch(rig, initial_joints)
        seat_offset_local = (
            initial_global_rotations[:, WRIST_J].transpose(1, 2)
            @ (initial_seat - initial_joints[:, WRIST_J]).unsqueeze(-1)
        ).squeeze(-1)
        selected_joints = [WRIST_J]
        selected_joints.extend(joint for joint, _parent in chain)
        selected_joints.extend(parent for _joint, parent in chain)

        def reach_fk():
            positions, rotations = H.fwd_selected_joints_from_rot_mats_batch(
                local_rotations, selected_joints
            )
            seat_position = positions[WRIST_J] + (
                rotations[WRIST_J] @ seat_offset_local.unsqueeze(-1)
            ).squeeze(-1)
            return positions, rotations, seat_position

        # Keep the iterative solve device-side; fixed iterations avoid a
        # host synchronization for a tensor-valued convergence check on every step.
        for _ in range(int(dls_iters)):
            joint_positions, global_rotations, seat = reach_fk()
            error = target_tensor - seat
            active = torch.linalg.vector_norm(error, dim=-1) >= 1e-3
            jacobian = torch.cat(
                [
                    _point_rotation_jacobian(seat, joint_positions[joint])
                    for joint, _ in chain
                ],
                dim=2,
            )
            lhs = jacobian @ jacobian.transpose(1, 2) + (damping**2) * eye
            task_step = torch.linalg.solve(lhs, error.unsqueeze(-1))
            updates = (jacobian.transpose(1, 2) @ task_step).squeeze(-1)
            updates = updates.reshape(len(misses), len(chain), 3)
            update_norm = torch.linalg.vector_norm(updates, dim=-1, keepdim=True)
            updates = updates * torch.clamp(
                max_delta_rad / update_norm.clamp_min(1e-9), max=1.0
            )
            _apply_world_rotation_updates(
                local_rotations, global_rotations, updates, chain, active
            )

        for _ in range(int(ccd_cycles)):
            for joint, parent in chain:
                joint_positions, global_rotations, seat = reach_fk()
                error_norm = torch.linalg.vector_norm(target_tensor - seat, dim=-1)
                active = error_norm >= 1e-3
                delta_world = _rot_between_batch(
                    seat - joint_positions[joint],
                    target_tensor - joint_positions[joint],
                )
                parent_global = global_rotations[parent]
                delta_local = (
                    parent_global.transpose(1, 2) @ delta_world @ parent_global
                )
                proposed = delta_local @ local_rotations[:, joint]
                local_rotations[:, joint] = torch.where(
                    active[:, None, None], proposed, local_rotations[:, joint]
                )

        joints, global_rotations = H.fwd_joints_from_rot_mats_batch(
            local_rotations, return_global_rot=True
        )
        if getattr(rig, "natural_arm", False):
            _swivel_elbow_down_batch(
                local_rotations, joints, global_rotations, target_tensor
            )
            joints = H.fwd_joints_from_rot_mats_batch(local_rotations)
        seat = _seat_positions_batch(rig, joints)
        errors = torch.linalg.vector_norm(seat - target_tensor, dim=-1)
        rotation_np = local_rotations[:, [joint for joint, _ in chain]].cpu().numpy()
        joint_np = joints.cpu().numpy()
        error_np = errors.cpu().numpy().astype(np.float64)

    solved_poses = miss_poses.copy()
    for local_index in range(len(misses)):
        for chain_index, (joint, _parent) in enumerate(chain):
            solved_poses[local_index, joint * 3 : joint * 3 + 3] = H.logmap(
                rotation_np[local_index, chain_index]
            ).astype(np.float32)
    for local_index, output_index in enumerate(misses):
        output_poses[output_index] = solved_poses[local_index]
        output_errors[output_index] = error_np[local_index]
        output_joints[output_index] = joint_np[local_index]
        _cache_reach(
            rig,
            keys[output_index],
            solved_poses[local_index],
            error_np[local_index],
            joint_np[local_index],
        )
    elapsed = time.perf_counter() - t0
    logger.info(
        f"[warp-batch] solved={len(misses)} cached={batch - len(misses)} device={device} "
        f"time={elapsed:.3f}s median_err={float(np.median(error_np)) * 1000:.1f}mm"
    )
    return output_poses, output_errors, output_joints


def warp_arm_to_seat(
    rig: SmplhRig,
    base_pose: np.ndarray,
    rz: np.ndarray,
    world_off: np.ndarray,
    obj_place_world: np.ndarray,
    cycles: int = 28,
    chain: list | None = None,
    use_cache: bool = True,
    return_joints: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    """Closed-form CCD: rotate ONLY the right shoulder+elbow (+optional collar) of ``base_pose`` so the
    cup SEAT (``seat_object``) lands at ``obj_place_world``, leaving everything else (global/spine/wrist
    joint/other arm) UNTOUCHED. The fingers are set to the contact grip ``rig.rhp`` first so the wrist
    frame matches the real grasp (``seat_object`` is built from the finger joints). No random search --
    each CCD step is the analytic shortest-arc that swings the distal chain's cup toward the target,
    expressed in the joint's parent global frame and composed onto its local rotation (same trick as
    ``solve_wrist_orient``). Works in body-local space (where ``H.fwd`` lives); the world target is
    pulled back via ``rz``/``world_off``. Returns (warped 156-D pose, final cup position error in m).
    """
    side_pose = np.asarray(base_pose, dtype=np.float32).copy()
    side_pose[RHAND_LO : RHAND_LO + 45] = rig.rhp
    t0 = time.perf_counter()
    obj_local = np.asarray(rz, dtype=np.float64).T @ (
        np.asarray(obj_place_world, dtype=np.float64)
        - np.asarray(world_off, dtype=np.float64)
    )
    if chain is None:
        chain = [(R_ELBOW, R_SHOULDER), (R_SHOULDER, R_COLLAR)]
    cache_key = _reach_cache_key(rig, side_pose, obj_local, chain)
    cached = rig._reach_pose_cache.get(cache_key) if use_cache else None
    if cached is not None:
        if return_joints:
            return cached[0].copy(), float(cached[1]), cached[2].copy()
        return cached[0].copy(), float(cached[1])

    def _ccd(p: np.ndarray, ncyc: int) -> tuple[np.ndarray, float]:
        p = p.copy()
        e_fin = float("nan")
        for _ in range(int(ncyc)):
            for j, parent in chain:
                Jl = H.fwd_joints(p)
                cup = seat_object(rig, Jl, baked=True)[:3, 3]
                jp = Jl[j].astype(np.float64)
                e = _rot_between(cup - jp, obj_local - jp)
                g_p = H.global_rot(p, parent)
                R_j = _rodrigues(p[j * 3 : j * 3 + 3])
                new_R = (g_p.T @ e @ g_p) @ R_j
                p[j * 3 : j * 3 + 3] = H.logmap(new_R).astype(np.float32)
            Jl = H.fwd_joints(p)
            e_fin = float(
                np.linalg.norm(seat_object(rig, Jl, baked=True)[:3, 3] - obj_local)
            )
            if e_fin < 1e-3:
                break
        return p, e_fin

    # Reach: the proven side seed drives the seat onto the target for ANY height/side (unchanged default).
    pose, err = _ccd(side_pose, int(cycles))
    seed_tag = "side"
    # Naturalness (gated, reach-EXACT): the side solution reaches tightly but leaves the elbow "winged up"
    # and the forearm wrenched. Swivel the arm about the shoulder->seat axis so the elbow drops to a human
    # pose; the seat is on that axis so it does NOT move -> the reach is preserved exactly (re-measured
    # below to confirm). Default (natural_arm False) skips this entirely -> byte-identical.
    if getattr(rig, "natural_arm", False):
        pose = swivel_elbow_down(rig, pose, obj_local)
        err = float(
            np.linalg.norm(
                seat_object(rig, H.fwd_joints(pose), baked=True)[:3, 3] - obj_local
            )
        )
        seed_tag = "swivel"
    # Naturalness probe: signed elbow elevation vs the shoulder->wrist line (deg). <0 == elbow BELOW
    # the line == human "elbow down"; the twisted side seed drives it positive (elbow winged up/out).
    Je = H.fwd_joints(pose)
    _sw = Je[WRIST_J] - Je[R_SHOULDER]
    _swn = float(np.linalg.norm(_sw))
    _elev = float("nan")
    if _swn > 1e-6:
        _sw = _sw / _swn
        _ev = Je[R_ELBOW] - Je[R_SHOULDER]
        _perp = _ev - float(np.dot(_ev, _sw)) * _sw
        _elev = float(
            np.degrees(np.arctan2(_perp[2], float(np.linalg.norm(_perp[:2])) + 1e-9))
        )
    print(
        f"[warp] reach err={err * 1000:.1f}mm in {time.perf_counter() - t0:.3f}s "
        f"({int(cycles)} cyc) seed={seed_tag} elbow_elev={_elev:+.0f}deg"
    )
    _cache_reach(rig, cache_key, pose, err, Je)
    if return_joints:
        return pose, err, Je
    return pose, err


def warp_arm_to_wrist(
    rig: SmplhRig,
    base_pose: np.ndarray,
    rz: np.ndarray,
    world_off: np.ndarray,
    wrist_target_world: np.ndarray,
    cycles: int = 28,
    chain: list | None = None,
) -> tuple[np.ndarray, float]:
    """Rotate the right shoulder/elbow until the wrist reaches a world-space target."""
    pose = np.asarray(base_pose, dtype=np.float32).copy()
    target_local = np.asarray(rz, dtype=np.float64).T @ (
        np.asarray(wrist_target_world, dtype=np.float64)
        - np.asarray(world_off, dtype=np.float64)
    )
    arm_chain = (
        [(R_ELBOW, R_SHOULDER), (R_SHOULDER, R_COLLAR)] if chain is None else chain
    )
    for _ in range(int(cycles)):
        for joint, parent in arm_chain:
            joints = H.fwd_joints(pose)
            wrist = joints[WRIST_J].astype(np.float64)
            pivot = joints[joint].astype(np.float64)
            correction = _rot_between(wrist - pivot, target_local - pivot)
            parent_rotation = H.global_rot(pose, parent)
            local_rotation = _rodrigues(pose[joint * 3 : joint * 3 + 3])
            pose[joint * 3 : joint * 3 + 3] = H.logmap(
                (parent_rotation.T @ correction @ parent_rotation) @ local_rotation
            ).astype(np.float32)
        error = float(np.linalg.norm(H.fwd_joints(pose)[WRIST_J] - target_local))
        if error < 1e-3:
            break
    if getattr(rig, "natural_arm", False):
        pose = swivel_elbow_down(rig, pose, target_local)
        error = float(np.linalg.norm(H.fwd_joints(pose)[WRIST_J] - target_local))
    return pose, error


def wrist_world_T(J: np.ndarray) -> np.ndarray:
    """4x4 world transform of the geometric wrist frame for posed (world) joints J."""
    R_w, wrist_o = H.wrist_frame(J)
    T = np.eye(4)
    T[:3, :3] = R_w
    T[:3, 3] = wrist_o
    return T


def seat_object_nosnap(rig: SmplhRig, J: np.ndarray, otw: np.ndarray) -> np.ndarray:
    """Cup world pose = wrist_world_T(J) @ otw, with NO upright snap (the cup rides the wrist exactly).

    Used by the live-wrap path: the cup keeps the orientation it was grasped at (its real resting pose),
    so it follows the hand naturally during the lift instead of being forced upright (which rotated it
    into the fingers). ``otw`` is the cup-in-wrist transform captured live at the grasp frame."""
    return wrist_world_T(J) @ np.asarray(otw, dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# RUNTIME contact-wrap: close the SMPL+H right-hand fingers to the LIVE object surface (the
# object-general grasp). Mirrors the offline side-grasp generator's close-to-contact curl (Zc flex,
# PIP/DIP ratios, contact gate) but runs on OUR SMPL+H hand verts (LBS weights) against the live object
# mesh at runtime -- no per-object offline grip, no obj_T_in_wrist, no upright snap.
# ----------------------------------------------------------------------------------------------
_RH_SLOT = {
    "index": (0, 1, 2),
    "middle": (3, 4, 5),
    "pinky": (6, 7, 8),
    "ring": (9, 10, 11),
    "thumb": (12, 13, 14),
}
_RH_FINGERS = ("index", "middle", "ring", "pinky")
_RH_Zc = np.array(
    [0.0, 0.0, 1.0]
)  # 4-finger flexion axis (curl into palm) -- mirrors the offline side-grasp generator
_RH_THX = np.array([0.0, -1.0, 0.0])  # thumb flexion axis
_RH_TOPP = np.array([0.0, 0.0, 1.0])  # thumb opposition axis
_RH_PIP, _RH_DIP = 0.85, 0.30
# right-hand global joint indices (the 45-D block maps to SMPL+H joints 37..51): block i -> joint 37+i.
_RH_FINGER_JOINTS = {
    "index": [37, 38, 39],
    "middle": [40, 41, 42],
    "pinky": [43, 44, 45],
    "ring": [46, 47, 48],
    "thumb": [49, 50, 51],
}


def _set_rh(p45: np.ndarray, finger: str, which: int, aa) -> None:
    j = _RH_SLOT[finger][which]
    p45[j * 3 : j * 3 + 3] = np.asarray(aa, dtype=np.float32)


def build_rhand_pose(mcp_deg: dict, thumb=(25.0, 5.0)) -> np.ndarray:
    """45-D right-hand MANO pose from per-finger MCP flex (deg) + thumb (opposition, flex) deg.

    Same convention as the offline side-grasp generator: MCP = mcp*Zc, PIP = 0.85*mcp, DIP = 0.30*mcp;
    thumb slot0 = opp*TOPP + flex*THX, slot1/2 = 0.7/0.55 * flex * THX."""
    p = np.zeros(45, dtype=np.float32)
    for f in _RH_FINGERS:
        m = float(mcp_deg[f])
        _set_rh(p, f, 0, np.deg2rad(m) * _RH_Zc)
        _set_rh(p, f, 1, np.deg2rad(_RH_PIP * m) * _RH_Zc)
        _set_rh(p, f, 2, np.deg2rad(_RH_DIP * m) * _RH_Zc)
    opp, t = float(thumb[0]), float(thumb[1])
    _set_rh(p, "thumb", 0, np.deg2rad(opp) * _RH_TOPP + np.deg2rad(t) * _RH_THX)
    _set_rh(p, "thumb", 1, np.deg2rad(0.7 * t) * _RH_THX)
    _set_rh(p, "thumb", 2, np.deg2rad(0.55 * t) * _RH_THX)
    return p


def right_hand_masks(thresh: float = 0.5) -> dict:
    """Per-finger SMPL+H vertex masks (full finger + distal 'pad') via the LBS skinning weights.

    pad = the PIP+DIP verts only (exclude the proximal/MCP, which sit near the palm and would false-
    trigger contact). Returns {finger: (full_mask(Nv,), pad_mask(Nv,))} (numpy bool)."""
    W = H.TPL[5].numpy()  # (Nv, NJ) linear-blend-skinning weights
    masks = {}
    for f, js in _RH_FINGER_JOINTS.items():
        full = W[:, js].sum(1) > thresh
        pad = W[:, js[1:]].sum(1) > thresh  # distal two joints = the contact pad
        masks[f] = (full, pad)
    return masks


def _runtime_wrap_seq(
    rig: SmplhRig,
    base_pose: np.ndarray,
    rz: np.ndarray,
    world_off: np.ndarray,
    sd_world,
    masks: dict,
    flex_max: float = 110.0,
    step: float = 6.0,
    contact_band: float = 0.006,
    pen_tol: float = 0.004,
    thumb_opp: float = 22.0,
) -> tuple[np.ndarray, dict]:
    """Reference SEQUENTIAL close-to-contact wrap: one full-body ``H.fwd`` per curl step, per-
    finger signed-distance gate. Kept verbatim as the parity + timing baseline for ``runtime_wrap``."""
    pose = np.asarray(base_pose, dtype=np.float32).copy()
    mcp = {f: 5.0 for f in _RH_FINGERS}
    thumb_flex = 5.0

    def _fk_world():
        V = H.fwd(pose)[0]
        return (np.asarray(rz, dtype=np.float64) @ V.T).T + np.asarray(
            world_off, dtype=np.float64
        )

    def _apply():
        pose[RHAND_LO : RHAND_LO + 45] = build_rhand_pose(
            mcp, thumb=(thumb_opp, thumb_flex)
        )

    stats = {}
    for f in list(_RH_FINGERS) + ["thumb"]:
        full, pad = masks[f]
        is_thumb = f == "thumb"
        d = 5.0
        stopped = d
        while d <= flex_max + 1e-6:
            if is_thumb:
                thumb_flex = d
            else:
                mcp[f] = d
            _apply()
            Vw = _fk_world()
            if (
                float(sd_world(Vw[full]).max()) > pen_tol
            ):  # penetrating -> back off a step
                d = max(5.0, d - step)
                break
            stopped = d
            if (
                float(sd_world(Vw[pad]).max()) >= -contact_band
            ):  # pad reached the surface
                break
            d += step
        if is_thumb:
            thumb_flex = stopped
        else:
            mcp[f] = stopped
        _apply()
        Vw = _fk_world()
        stats[f] = {
            "deg": round(float(stopped), 1),
            "pad_sd_mm": round(float(sd_world(Vw[pad]).max()) * 1000.0, 1),
            "pen_mm": round(float(sd_world(Vw[full]).max()) * 1000.0, 1),
        }
    return pose[RHAND_LO : RHAND_LO + 45].copy(), stats


def runtime_wrap(
    rig: SmplhRig,
    base_pose: np.ndarray,
    rz: np.ndarray,
    world_off: np.ndarray,
    sd_world,
    masks: dict,
    flex_max: float = 110.0,
    step: float = 6.0,
    contact_band: float = 0.006,
    pen_tol: float = 0.004,
    thumb_opp: float = 22.0,
    seed_rhp: "np.ndarray | None" = None,
) -> tuple[np.ndarray, dict]:
    """Close-to-contact wrap of the SMPL+H right hand against the LIVE object (object-general).

    ``seed_rhp`` (OPT-IN): when given a 45-D right-hand MANO pose (e.g. a retargeted GRAB grasp),
    the wrap does NOT open from a flat hand and re-synthesize; it STARTS from that grasp shape and
    only curls each finger a little further to meet the real surface (or relaxes it slightly if it
    over-penetrates), preserving the natural grasp's spread/thumb while contact tightens the loose
    fingers. When ``seed_rhp is None`` the ORIGINAL flat-open + rebuild path below runs verbatim
    (so existing callers + the WRAP_BENCH parity are unchanged). See _seeded branch just below.

    Batched -- BEHAVIOUR-PRESERVING vs ``_runtime_wrap_seq`` but ~60x faster. For each finger the whole
    MCP angle sweep (5..flex_max) is forward-kinematiced in ONE batched ``H.lbs`` call (instead of a full-
    body ``H.fwd`` per 6deg step), and the object gate is a precomputed voxel SDF (``sd_world``) instead of
    a per-call ``trimesh.signed_distance``. Each finger curls a kinematically independent sub-chain off the
    wrist, so its stop angle depends only on its OWN verts vs the object; enumerating the sweep (with the
    already-finalized fingers held, exactly as the sequential loop) and applying the identical stop rule
    (stop one step before penetration, else at first pad contact) reproduces the sequential result.

    Convention unchanged: ``sd_world(Vw_subset)`` POSITIVE = inside; stop when the PAD reaches the band
    (sd >= -contact_band), back off if any finger vert penetrates (sd > pen_tol). No fixed grip, no
    obj_T_in_wrist, no snap. Set env ``WRAP_BENCH=1`` to also run the sequential+exact baseline once and
    print a parity/speedup report. Returns (45-D right-hand pose, stats).
    """
    rz64 = np.asarray(rz, dtype=np.float64)
    off64 = np.asarray(world_off, dtype=np.float64)
    angles = np.arange(5.0, flex_max + 1e-6, step, dtype=np.float32)
    n_cand = int(len(angles))

    def _batch_fk_world(pose_batch):
        device = _batch_ik_device()
        model = H.lbs_tensors(device)
        with torch.no_grad():
            # Construct the batch directly on its execution device.
            Vb, _ = H.lbs(
                torch.zeros((len(pose_batch), 10), dtype=torch.float32, device=device),
                torch.as_tensor(pose_batch, dtype=torch.float32, device=device),
                *model,
                pose2rot=True,
            )
        Vb = Vb.detach().cpu().numpy().astype(np.float64)  # (B, Nv, 3) body-local
        return (
            np.einsum("ij,bnj->bni", rz64, Vb) + off64
        )  # (rz @ V.T).T + world_off, batched

    # ---- OPT-IN seeded wrap: start from the GRAB grasp shape, tighten only the loose fingers. ----
    if seed_rhp is not None:
        base = np.asarray(base_pose, dtype=np.float32).copy()
        committed = np.asarray(seed_rhp, dtype=np.float64).reshape(45).copy()
        # per-finger delta about its OWN flex axis, on TOP of the seed: a little open (-) so an
        # oversized proxy relaxes the grip, then close (+) to contact. Small range keeps the seed shape.
        deltas = np.arange(-12.0, 48.0 + 1e-6, step, dtype=np.float32)
        n_seed = int(len(deltas))
        zero_i = int(
            np.argmin(np.abs(deltas))
        )  # delta==0 == the untouched seed (kept if no contact)
        t0 = time.perf_counter()
        stats = {}
        for f in list(_RH_FINGERS) + ["thumb"]:
            full, pad = masks[f]
            slots = _RH_SLOT[f]
            axis = _RH_THX if f == "thumb" else _RH_Zc
            ratios = (1.0, 0.7, 0.55) if f == "thumb" else (1.0, _RH_PIP, _RH_DIP)
            pose_batch = np.tile(base.reshape(1, -1), (n_seed, 1))
            for i in range(n_seed):
                cand = committed.copy()
                for slot, r in zip(slots, ratios):
                    cand[slot * 3 : slot * 3 + 3] += (
                        np.deg2rad(r * float(deltas[i])) * axis
                    )
                pose_batch[i, RHAND_LO : RHAND_LO + 45] = cand
            Vw_batch = _batch_fk_world(pose_batch)
            sel = None
            for i in range(
                n_seed
            ):  # scan open -> closed, stop one before penetration / at first contact
                Vw = Vw_batch[i]
                if float(sd_world(Vw[full]).max()) > pen_tol:
                    sel = max(0, i - 1)
                    break
                if float(sd_world(Vw[pad]).max()) >= -contact_band:
                    sel = i
                    break
            if (
                sel is None
            ):  # never contacted and never penetrated -> keep the natural seed shape
                sel = zero_i
            dl = float(deltas[sel])
            for slot, r in zip(
                slots, ratios
            ):  # commit finger f so later fingers see it (as in the seq loop)
                committed[slot * 3 : slot * 3 + 3] += np.deg2rad(r * dl) * axis
            Vwf = Vw_batch[sel]
            stats[f] = {
                "deg": round(
                    dl, 1
                ),  # NOTE: delta from the GRAB seed (not an absolute MCP angle)
                "pad_sd_mm": round(float(sd_world(Vwf[pad]).max()) * 1000.0, 1),
                "pen_mm": round(float(sd_world(Vwf[full]).max()) * 1000.0, 1),
            }
        print(
            f"[wrap] SEEDED {time.perf_counter() - t0:.3f}s ({n_seed} cand/finger) "
            "finger delta-from-GRAB-seed(deg): "
            + ", ".join(f"{k}={v['deg']:+.0f}" for k, v in stats.items()),
            flush=True,
        )
        return np.asarray(committed, dtype=np.float32).copy(), stats

    t0 = time.perf_counter()
    base = np.asarray(base_pose, dtype=np.float32)
    mcp = {f: 5.0 for f in _RH_FINGERS}
    thumb_flex = 5.0
    stats = {}
    for f in list(_RH_FINGERS) + ["thumb"]:
        full, pad = masks[f]
        is_thumb = f == "thumb"
        # One batch: sweep finger f across every candidate angle; the OTHER fingers stay at their
        # already-finalized state (mcp / thumb_flex), matching the sequential loop's context exactly.
        pose_batch = np.tile(base.reshape(1, -1), (n_cand, 1))
        for i in range(n_cand):
            if is_thumb:
                rhp_i = build_rhand_pose(mcp, thumb=(thumb_opp, float(angles[i])))
            else:
                mcp_i = dict(mcp)
                mcp_i[f] = float(angles[i])
                rhp_i = build_rhand_pose(mcp_i, thumb=(thumb_opp, thumb_flex))
            pose_batch[i, RHAND_LO : RHAND_LO + 45] = rhp_i
        Vw_batch = _batch_fk_world(pose_batch)
        # Replicate the sequential stop rule over the (now precomputed) sweep.
        stopped = 5.0
        chosen = 0
        for i in range(n_cand):
            Vw = Vw_batch[i]
            if (
                float(sd_world(Vw[full]).max()) > pen_tol
            ):  # penetrating -> stop at previous angle
                break
            stopped = float(angles[i])
            chosen = i
            if (
                float(sd_world(Vw[pad]).max()) >= -contact_band
            ):  # pad reached the surface
                break
        if is_thumb:
            thumb_flex = stopped
        else:
            mcp[f] = stopped
        Vwf = Vw_batch[
            chosen
        ]  # the finalized pose for finger f (== sequential's post-commit FK)
        stats[f] = {
            "deg": round(float(stopped), 1),
            "pad_sd_mm": round(float(sd_world(Vwf[pad]).max()) * 1000.0, 1),
            "pen_mm": round(float(sd_world(Vwf[full]).max()) * 1000.0, 1),
        }
    rhp_out = build_rhand_pose(mcp, thumb=(thumb_opp, thumb_flex))
    t_new = time.perf_counter() - t0

    _wrap_bench(
        rig,
        base_pose,
        rz,
        world_off,
        sd_world,
        masks,
        stats,
        t_new,
        n_cand,
        flex_max=flex_max,
        step=step,
        contact_band=contact_band,
        pen_tol=pen_tol,
        thumb_opp=thumb_opp,
    )
    return np.asarray(rhp_out, dtype=np.float32).copy(), stats


def _wrap_bench(
    rig, base_pose, rz, world_off, sd_world, masks, new_stats, t_new, n_cand, **kw
) -> None:
    """Print a parity + speedup report for the batched wrap. Always reports the new timing and the
    FINAL wrapped-hand penetration measured with the EXACT mesh query (proves no penetration regression);
    when env WRAP_BENCH=1 and ``sd_world`` exposes an exact backend, also runs the sequential+exact
    baseline once for a same-input speedup + per-finger stop-angle parity."""
    kind = getattr(sd_world, "kind", "sd")
    exact = getattr(sd_world, "exact", None)
    pen_exact_mm = float("nan")
    if exact is not None:
        pose = np.asarray(base_pose, dtype=np.float32).copy()
        pose[RHAND_LO : RHAND_LO + 45] = build_rhand_pose(
            {f: new_stats[f]["deg"] for f in _RH_FINGERS},
            thumb=(kw["thumb_opp"], new_stats["thumb"]["deg"]),
        )
        V = H.fwd(pose)[0]
        Vw = (np.asarray(rz, np.float64) @ V.T).T + np.asarray(world_off, np.float64)
        allf = np.zeros(Vw.shape[0], dtype=bool)
        for f in list(_RH_FINGERS) + ["thumb"]:
            allf |= masks[f][0]
        pen_exact_mm = float(exact(Vw[allf]).max()) * 1000.0
    if os.environ.get("WRAP_BENCH") == "1" and exact is not None:
        t1 = time.perf_counter()
        _, seq_stats = _runtime_wrap_seq(
            rig,
            base_pose,
            rz,
            world_off,
            exact,
            masks,
            flex_max=kw["flex_max"],
            step=kw["step"],
            contact_band=kw["contact_band"],
            pen_tol=kw["pen_tol"],
            thumb_opp=kw["thumb_opp"],
        )
        t_old = time.perf_counter() - t1
        dmax = max(
            abs(new_stats[f]["deg"] - seq_stats[f]["deg"])
            for f in list(_RH_FINGERS) + ["thumb"]
        )
        print(
            f"[wrap-bench] NEW(batched+{kind}) {t_new:.3f}s  vs  OLD(seq+trimesh) {t_old:.3f}s "
            f"-> {t_old / max(t_new, 1e-6):.1f}x | per-finger stop-angle max_delta={dmax:.1f}deg "
            f"| final EXACT pen={pen_exact_mm:.2f}mm | {n_cand} cand/finger",
            flush=True,
        )
    else:
        print(
            f"[wrap] batched+{kind} {t_new:.3f}s ({n_cand} cand/finger) "
            f"final EXACT pen={pen_exact_mm:.2f}mm",
            flush=True,
        )

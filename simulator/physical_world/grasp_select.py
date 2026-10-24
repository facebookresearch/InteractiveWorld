"""Unified geometric grasp for the FPV grasping demo: NATURAL reach wrist as the initial, then a small
bounded fine-tune to improve the grip. Object-general, no per-object tuning, no VLM.

Flow (select_and_grasp):
  1. object_geom  -- OBB descriptors off the world mesh.
  2. pick_family  -- PINCH (thin/small) else WRAP; this only picks the thumb opposition for the wrap.
  3. warp_arm_to_seat  -- rotate shoulder+elbow so the object seat lands at obj_place, KEEPING the natural
     GRAB wrist (the hand extends along the forearm). This natural wrist is the INITIAL orientation.
  4. wrap the fingers to the real surface. If the natural grip already contacts well (>= _ACCEPT_NEAR of
     the 5 fingers), keep it -- fast, and with no twist.
  5. FINE-TUNE (only if the natural grip is loose): try a few SMALL bounded perturbations of the natural
     wrist (yaw about vertical + roll about the palm normal, all <= ~25 deg so it never looks twisted),
     wrap each, and keep the one that contacts the most fingers (tie-break: closest to natural).

Why anchor the search on the NATURAL wrist (not a canonical frame): forcing an absolute canonical frame
(r_canon) bends the wrist off the forearm (contorted) and, being object-independent, makes every grasp
identical. Anchoring on the natural reach wrist and only perturbing it a little keeps every grasp natural
and object-varied, while the scoring still fine-tunes for a better grip. Anti-claw comes from the thumb
opposition in the wrap, not from the wrist frame.

All heavy lifting is delegated to ``core`` (smplh_anim_core) + ``core.H`` (smplh_fpv). numpy only. The
caller supplies ``sd_world`` (``sd_world(pts) -> signed dist``, POSITIVE = inside) with attribute
``sd_world.mesh`` (a world-space trimesh.Trimesh). All frames are BODY-LOCAL (the space of core.H.fwd /
warp_arm_to_seat / solve_wrist_orient).
"""

from __future__ import annotations

import os

import interaction_hints
import numpy as np

# Per-family thumb OPPOSITION angle (deg) for the live wrap -- this, not a forced wrist frame, is what
# swings the thumb to the opposite wall from the four fingers (prevents a same-side claw).
_THUMB_OPP = {"WRAP": 55.0, "PINCH": 35.0}

# P0 per-shape family prototypes: shape class -> atlas family whose right_hand_pose SEEDS the live wrap
# (the wrap then only contact-tightens). CYL/ROUND reuse the existing curated families; PINCH is authored.
_PROTO_FAMILY = {"CYL": "side_wrap", "ROUND": "round_wrap", "PINCH": "pinch"}
_PROTO_CACHE: "dict[str, np.ndarray] | None" = None

# Fine-tune search (run only when the natural grip is loose): SMALL perturbations of the natural wrist,
# in degrees. Kept small so the wrist stays natural. yaw = about vertical (repoint the palm at the object);
# roll = about the palm normal (tilt the fingers). Worst-case combined bend ~25 deg (a mild, natural wrist
# adjustment -- not the ~90 deg fold that forcing r_canon produced).
_FT_YAW = (-20.0, -10.0, 10.0, 20.0)
_FT_ROLL = (0.0, -15.0, 15.0)
# A natural grip contacting at least this many of the 5 fingers is kept as-is (no fine-tune search).
_ACCEPT_NEAR = 4

# A broad, paper-thin object is not naturally grasped at its centre: doing so puts the palm through
# the sheet. Keep small thin props (phones/cards) on the existing compact pinch, and move only objects
# wider than a hand to a perimeter grip. The hand target sits just outside the nearest edge and just
# above the accessible face; the live finger wrap closes back onto the material.
_LARGE_FLAT_MIN_SPAN_M = 0.18
_FLAT_EDGE_CLEARANCE_M = 0.015
_FLAT_FACE_CLEARANCE_M = 0.010


def object_geom(mesh) -> dict:
    """OBB descriptors of a world-space trimesh: center, extents, cross radius, height, size flag.

    e0>=e1>=e2 are the oriented-bbox extents; r = 0.5*median(width_x, width_y) is the horizontal
    cross-section radius; is_small marks a thin/small (pinchable) object.
    """
    c = np.asarray(mesh.centroid, dtype=np.float64)
    ext = np.asarray(mesh.bounding_box_oriented.primitive.extents, dtype=np.float64)
    e0, e1, e2 = sorted([float(x) for x in ext], reverse=True)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    width_x = float(bounds[1][0] - bounds[0][0])
    width_y = float(bounds[1][1] - bounds[0][1])
    r = 0.5 * float(np.median([width_x, width_y]))
    h = float(bounds[1][2] - bounds[0][2])
    is_flat = bool(e2 <= 0.05 and e1 >= 3.0 * max(e2, 1e-6))
    return {
        "c": c,
        "r": r,
        "h": h,
        "e0": e0,
        "e1": e1,
        "e2": e2,
        "min_ext": e2,
        "max_ext": e0,
        "is_small": bool(e0 < 0.14 and e2 < 0.05),
        "is_flat": is_flat,
    }


def _body_target(bounds, stand_xy, height_fraction: float, clearance_m: float):
    minimum, maximum = np.asarray(bounds, dtype=np.float64)
    center = 0.5 * (minimum + maximum)
    direction = np.asarray(stand_xy, dtype=np.float64).reshape(2) - center[:2]
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9:
        direction = np.array([0.0, -1.0], dtype=np.float64)
    else:
        direction /= distance
    half = 0.5 * (maximum[:2] - minimum[:2])
    limits = [
        float(half[axis] / abs(direction[axis]))
        for axis in range(2)
        if abs(float(direction[axis])) > 1e-9
    ]
    radial = min(limits) if limits else 0.0
    target = center.copy()
    target[:2] += direction * (radial + clearance_m)
    target[2] = float(minimum[2]) + height_fraction * float(maximum[2] - minimum[2])
    return target


def grasp_target(mesh, object_origin, stand_xy, hint=None) -> np.ndarray:
    """Return a geometry-derived hand target, optionally constrained by an authored hint."""
    fallback = np.asarray(object_origin, dtype=np.float64).reshape(3)
    hint = interaction_hints.normalize_grasp_hint(hint)
    geom = object_geom(mesh)
    target_mode = hint.get("target", "auto")
    if target_mode == "center":
        return fallback.copy()
    if target_mode == "body":
        return _body_target(
            mesh.bounds,
            stand_xy,
            hint["height_fraction"],
            hint["clearance_m"],
        )
    use_edge = target_mode == "edge" or (
        target_mode == "auto"
        and geom["is_flat"]
        and geom["max_ext"] >= _LARGE_FLAT_MIN_SPAN_M
    )
    if not use_edge:
        return fallback.copy()

    primitive = mesh.bounding_box_oriented.primitive
    extents = np.asarray(primitive.extents, dtype=np.float64).reshape(-1)
    transform = np.asarray(getattr(primitive, "transform", None), dtype=np.float64)
    if extents.shape != (3,) or transform.shape != (4, 4):
        return fallback.copy()

    thin_axis = int(np.argmin(extents))
    plane_axes = [axis for axis in range(3) if axis != thin_axis]
    rotation = transform[:3, :3]
    thin_normal = rotation[:, thin_axis]
    if abs(float(thin_normal[2])) >= 0.6:
        face_sign = 1.0 if float(thin_normal[2]) >= 0.0 else -1.0
    else:
        stand = np.asarray(stand_xy, dtype=np.float64).reshape(2)
        positive_face = transform[:3, 3] + thin_normal * 0.5 * extents[thin_axis]
        negative_face = transform[:3, 3] - thin_normal * 0.5 * extents[thin_axis]
        face_sign = (
            1.0
            if np.linalg.norm(positive_face[:2] - stand)
            <= np.linalg.norm(negative_face[:2] - stand)
            else -1.0
        )
    edge_mode = hint.get("edge", "nearest")
    if edge_mode == "long":
        candidate_axes = [min(plane_axes, key=lambda axis: float(extents[axis]))]
    elif edge_mode == "short":
        candidate_axes = [max(plane_axes, key=lambda axis: float(extents[axis]))]
    else:
        candidate_axes = plane_axes
    candidates = []
    for edge_axis in candidate_axes:
        edge = 0.5 * float(extents[edge_axis]) + float(
            hint.get("clearance_m", _FLAT_EDGE_CLEARANCE_M)
        )
        for sign in (-1.0, 1.0):
            local = np.zeros(3, dtype=np.float64)
            local[edge_axis] = sign * edge
            local[thin_axis] = face_sign * (
                0.5 * float(extents[thin_axis]) + _FLAT_FACE_CLEARANCE_M
            )
            world = rotation @ local + transform[:3, 3]
            candidates.append(world)

    stand = np.asarray(stand_xy, dtype=np.float64).reshape(2)
    return min(
        candidates,
        key=lambda point: (
            float(np.linalg.norm(point[:2] - stand)),
            float(np.linalg.norm(point - fallback)),
            tuple(float(value) for value in point),
        ),
    ).copy()


def pick_family(geom: dict, hint=None) -> str:
    """ONE grip family from pure shape: thin/small -> PINCH, else WRAP. Selects the thumb opposition only;
    the wrist starts natural and the fingers adapt to the real surface, so no per-object wrist tuning."""
    family = interaction_hints.normalize_grasp_hint(hint).get("family", "auto")
    if family != "auto":
        return family.upper()
    return "PINCH" if geom["is_small"] or geom.get("is_flat", False) else "WRAP"


def classify_shape(geom: dict, hint=None) -> str:
    """Coarse graspable-shape class used to pick a per-shape grip PROTOTYPE (P0: CYL / ROUND / PINCH).

    Reuses object_geom's OBB descriptors: thin/small -> PINCH; clearly taller-than-wide -> CYL (side
    wrap of a mug/can); otherwise ROUND (cage of a ball/low object). Deliberately few classes; extend
    as more prototypes are added (BOX/FLAT are P1). Distinct from pick_family (which only sets thumb_opp)."""
    forced_family = interaction_hints.normalize_grasp_hint(hint).get("family", "auto")
    if forced_family == "pinch":
        return "PINCH"
    if forced_family != "wrap" and (geom["is_small"] or geom.get("is_flat", False)):
        return "PINCH"
    cross_diam = 2.0 * geom["r"]
    if geom["h"] > 1.15 * max(cross_diam, 1e-6):
        return "CYL"
    return "ROUND"


def _proto_seed(core, cls: str) -> "np.ndarray | None":
    """Lazy-load (cached) the 45-D right-hand SEED pose for a shape class from its atlas family npz.

    CYL/ROUND reuse the curated side_wrap/round_wrap families; PINCH uses the authored pinch prototype.
    Returns None if the family file is missing -> caller falls back to the flat-open live wrap."""
    global _PROTO_CACHE
    if _PROTO_CACHE is None:
        _PROTO_CACHE = {}
        atlas = os.path.join(core._HAND_DIR, "atlas")
        for k, fam in _PROTO_FAMILY.items():
            path = os.path.join(atlas, f"{fam}_family.npz")
            try:
                with np.load(path) as g:
                    _PROTO_CACHE[k] = (
                        np.asarray(g["right_hand_pose"], dtype=np.float64)
                        .reshape(45)
                        .copy()
                    )
            except (OSError, KeyError):
                pass
    return _PROTO_CACHE.get(cls)


def _family_name(fam: str, geom: dict) -> str:
    """Human-facing grip label from the family + shape (for logs only)."""
    if fam == "PINCH":
        return "FLAT-PINCH" if geom.get("is_flat", False) else "SMALL-PINCH"
    cross_diam = 2.0 * geom["r"]
    tall = geom["h"] > 1.15 * max(cross_diam, 1e-6)
    aspect = geom["e0"] / max(geom["e2"], 1e-6)
    if tall:
        return "CYL-WRAP-SIDE"
    if aspect < 1.35:
        return "ROUND-WRAP"
    if aspect >= 1.6:
        return "BOX-WRAP"
    return "CAGE-FALLBACK"


def _wrap(
    core, rig, arm_pose, rz, world_off, sd_world, masks, thumb_opp, seed_rhp=None
):
    """Close the fingers to the live surface on ``arm_pose``. Returns (rhp, stats, near), where ``near``
    is how many of the 5 fingers landed within 15 mm of the surface.

    ``seed_rhp`` (opt-in): a 45-D GRAB grasp shape to START from (only tighten loose fingers), instead
    of opening flat and re-synthesizing -- keeps the natural grasp's thumb/spread. See runtime_wrap."""
    open_pose = arm_pose.copy()
    if seed_rhp is None:
        open_pose[core.RHAND_LO : core.RHAND_LO + 45] = 0.0
    rhp, stats = core.runtime_wrap(
        rig,
        open_pose,
        rz,
        world_off,
        sd_world,
        masks,
        thumb_opp=thumb_opp,
        seed_rhp=seed_rhp,
    )
    near = sum(1 for st in stats.values() if abs(st["pad_sd_mm"]) < 15.0)
    return rhp, stats, near


# 45-D right-hand slots for the NON-pinch fingers (middle 3-5, pinky 6-8, ring 9-11) -> flat indices 9..36.
# For a PINCH we keep these FOLDED at the seed so the grip is a thumb+index pinch, not a full wrap.
_PINCH_FOLD_LO, _PINCH_FOLD_HI = 9, 36


def _apply_pinch_fold(best: dict, shape_cls: str, seed_rhp, core) -> dict:
    """PINCH only: overwrite the middle/ring/pinky finger slots in the result with the folded seed, so the
    contact wrap (which curls EVERY finger onto the object) does not turn the pinch into a full-hand wrap.
    Index + thumb stay as the live-wrapped (contact-fitted) pinch. No-op for non-pinch / unseeded grasps."""
    if shape_cls != "PINCH" or seed_rhp is None:
        return best
    seed = np.asarray(seed_rhp, dtype=np.float64).reshape(45)
    rhp = np.asarray(best["rhp"], dtype=np.float64).copy()
    rhp[_PINCH_FOLD_LO:_PINCH_FOLD_HI] = seed[_PINCH_FOLD_LO:_PINCH_FOLD_HI]
    pose = np.asarray(best["pose"], dtype=np.float32).copy()
    pose[core.RHAND_LO : core.RHAND_LO + 45] = rhp
    best["rhp"] = rhp
    best["pose"] = pose
    return best


def select_and_grasp(
    core,
    rig,
    base_pose: np.ndarray,
    rz: np.ndarray,
    world_off: np.ndarray,
    obj_place_world: np.ndarray,
    sd_world,
    masks: dict,
    weights: dict | None = None,
    seed_grasp: bool = False,
    seed_wrist: bool = False,
    family_rhp: "np.ndarray | None" = None,
    family_wrist_R: "np.ndarray | None" = None,
    auto_family: bool = False,
    initial_pose: "np.ndarray | None" = None,
    initial_seat_err: "float | None" = None,
    grasp_hint=None,
) -> dict:
    """Natural reach wrist as the initial, then a small bounded fine-tune if the grip is loose.

    ``weights`` is accepted for call-site compatibility and ignored. Returns a dict: pose (156-D, fingers
    wrapped), rhp (45-D), wrap_stats, family, R (None), wrist_bend (0 for the natural pick, small for a
    fine-tuned one), seat_err, near, n_tried.
    """
    geom = object_geom(sd_world.mesh)
    family = pick_family(geom, grasp_hint)
    thumb_opp = _THUMB_OPP[family]
    shape_cls = classify_shape(geom, grasp_hint)
    # P0 per-shape grip: when enabled and no explicit family seed was passed, seed the wrap from the
    # prototype for this shape class (CYL/ROUND/PINCH) so different shapes get visibly different grips;
    # the live wrap below then contact-tightens onto the real mesh. Falls back to flat-open if absent.
    if family_rhp is None and auto_family:
        family_rhp = _proto_seed(core, shape_cls)
    # Seed the wrap from a GRAB grasp's finger pose (then contact only tightens the loose fingers). Prefer
    # ``family_rhp`` (the FAMILY clip's grasp fingers, decoupled from base_pose's body) when given; else fall
    # back to base_pose's own right hand (seed_grasp); else None -> the flat-open live-wrap.
    if family_rhp is not None:
        seed_rhp = np.asarray(family_rhp, dtype=np.float64).reshape(45).copy()
    elif seed_grasp:
        seed_rhp = np.asarray(
            base_pose[core.RHAND_LO : core.RHAND_LO + 45], dtype=np.float64
        ).copy()
    else:
        seed_rhp = None

    # 1) NATURAL initial wrist: warp shoulder+elbow so the seat lands at obj_place; keep the GRAB wrist
    # (the hand extends along the forearm). r_nat is that natural wrist frame (body-local).
    if initial_pose is not None and initial_seat_err is not None:
        nat_pose = np.asarray(initial_pose, dtype=np.float32).copy()
        seat_err = float(initial_seat_err)
    else:
        nat_pose, seat_err = core.warp_arm_to_seat(
            rig, base_pose, rz, world_off, obj_place_world
        )
    r_nat = core.H.wrist_frame(core.H.fwd_joints(nat_pose))[0]

    # Align the initial wrist to the FAMILY grasp orientation. Prefer ``family_wrist_R`` -- the target wrist
    # frame computed OBJECT-RELATIVE (family wrist-vs-object re-expressed against our object; body-independent,
    # so it transfers cleanly onto the fixed mug_lift body). Else fall back to base_pose's own wrist frame
    # (seed_wrist). warp_arm_to_seat kept the wrist JOINT but rotated shoulder+elbow (tilting the world wrist);
    # solve_wrist_orient re-solves the joint so the wrist frame matches the target.
    if family_wrist_R is not None or seed_wrist:
        r_fam = (
            np.asarray(family_wrist_R, dtype=np.float64)
            if family_wrist_R is not None
            else core.H.wrist_frame(core.H.fwd_joints(base_pose))[0]
        )
        wrist_aa, _oerr, fam_bend = core.solve_wrist_orient(rig, nat_pose, r_fam)
        nat_pose = nat_pose.copy()
        nat_pose[core.WRIST_J * 3 : core.WRIST_J * 3 + 3] = np.asarray(
            wrist_aa, dtype=np.float32
        )
        r_nat = r_fam
        _src = "object-relative family" if family_wrist_R is not None else "base-pose"
        print(
            f"[grasp_select] seed-wrist ({_src}): aligned initial wrist to family grasp "
            f"(bend {fam_bend:.1f}deg vs warped forearm)"
        )

    def _result(arm_pose, rhp, stats, near, bend, n_tried):
        final = arm_pose.copy()
        final[core.RHAND_LO : core.RHAND_LO + 45] = rhp
        return {
            "pose": final,
            "rhp": rhp,
            "wrap_stats": stats,
            "family": _family_name(family, geom),
            "shape": shape_cls,
            "R": None,
            "wrist_bend": float(bend),
            "seat_err": float(seat_err),
            "near": int(near),
            "cheap_score": 0.0,
            "n_tried": int(n_tried),
        }

    rhp, stats, near = _wrap(
        core, rig, nat_pose, rz, world_off, sd_world, masks, thumb_opp, seed_rhp
    )
    print(
        f"[grasp_select] shape={shape_cls} family={family} ({_family_name(family, geom)}) "
        f"seeded={'Y' if seed_rhp is not None else 'N'} natural wrist: "
        f"seat_err={seat_err * 1000:.1f}mm near={near}/5 thumb_opp={thumb_opp:.0f}"
    )
    best = _result(nat_pose, rhp, stats, near, 0.0, 1)
    if near >= _ACCEPT_NEAR:
        print("[grasp_select] natural grip good -> keep (no fine-tune)")
        return _apply_pinch_fold(best, shape_cls, seed_rhp, core)

    # 2) FINE-TUNE: the natural grip is loose -> search a few SMALL bounded perturbations of the natural
    # wrist (never far from natural, so never twisted) and keep the best-contacting.
    n = 1
    for yaw in _FT_YAW:
        for roll in _FT_ROLL:
            R = (
                core.H.axis_rot(np.array([0.0, 0.0, 1.0]), yaw)
                @ core.H.axis_rot(r_nat[:, 2], roll)
                @ r_nat
            )
            wrist_aa, _oerr, bend = core.solve_wrist_orient(rig, nat_pose, R)
            pose = nat_pose.copy()
            pose[core.WRIST_J * 3 : core.WRIST_J * 3 + 3] = np.asarray(
                wrist_aa, dtype=np.float32
            )
            rhp_i, stats_i, near_i = _wrap(
                core, rig, pose, rz, world_off, sd_world, masks, thumb_opp, seed_rhp
            )
            n += 1
            better = near_i > best["near"] or (
                near_i == best["near"] and bend < best["wrist_bend"]
            )
            if better:
                best = _result(pose, rhp_i, stats_i, near_i, bend, n)
    best["n_tried"] = n
    print(
        f"[grasp_select] fine-tuned {n - 1} perturbations around the natural wrist -> "
        f"best near={best['near']}/5 wrist_bend={best['wrist_bend']:.1f}deg"
    )
    return _apply_pinch_fold(best, shape_cls, seed_rhp, core)


# ----------------------------------------------------------------------------------------------
# Self-test: geometry + family classification only (no Isaac; the grasp itself needs core + a stage).
# Run: run_isaac.sh grasp_select.py
# ----------------------------------------------------------------------------------------------
def _self_test() -> int:
    import trimesh

    def _mk(name, mesh, expect_family):
        mesh.apply_translation(-mesh.centroid + np.array([0.0, -0.45, 0.05]))
        geom = object_geom(mesh)
        fam = pick_family(geom)
        cls = classify_shape(geom)
        ok = fam == expect_family
        print(
            f"  {name}: shape={cls} family={fam} (expected {expect_family}) [{'PASS' if ok else 'WARN'}]  "
            f"r={geom['r']:.3f} h={geom['h']:.3f} e=({geom['e0']:.3f},{geom['e1']:.3f},{geom['e2']:.3f}) "
            f"is_small={geom['is_small']}"
        )
        return ok

    print("grasp_select self-test (geometry + family classification; no Isaac)")
    results = [
        _mk(
            "cylinder r0.04 h0.09",
            trimesh.creation.cylinder(radius=0.04, height=0.09, sections=64),
            "WRAP",
        ),
        _mk("sphere r0.04", trimesh.creation.icosphere(radius=0.04), "WRAP"),
        _mk(
            "thin disk r0.08 h0.02",
            trimesh.creation.cylinder(radius=0.08, height=0.02, sections=64),
            "WRAP",
        ),
        _mk(
            "box 0.06x0.05x0.05",
            trimesh.creation.box(extents=(0.06, 0.05, 0.05)),
            "WRAP",
        ),
        _mk(
            "thin marker r0.008 h0.10",
            trimesh.creation.cylinder(radius=0.008, height=0.10, sections=32),
            "PINCH",
        ),
    ]
    print(f"self-test: {sum(results)}/{len(results)} shapes matched expected family")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())

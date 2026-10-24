"""action_check.py -- the UNIVERSAL post-action physics validator + corrector.

Runs after the executor produces the motion and BEFORE the RTX capture, so it can both
DETECT physical violations and CORRECT the objects to a physics-valid rest -- for EVERY
action (grab / put / putinside / open / close / ...), not one special demo scene.

Two guarantees, BOTH read from the ONE shared geometry (``scene.cworld`` =
``collision_world.CollisionWorld``), so the validator can never disagree with the
systems that placed things:

  * HUMAN is kinematic -> it cannot be pushed by physics, so we only DETECT: sample the
    (as-rendered) joints, run the oracle's human-vs-scene overlap, and report which action
    SEGMENT penetrates. A penetrating frame => the episode is human-dirty and the caller
    re-samples the seed or discards it. We NEVER "correct" the human by shoving it (that
    would push it into another obstacle -- the rejected per-frame-collider approach).

  * OBJECTS are real PhysX bodies -> we CORRECT them: every object a producer released is
    re-settled on the oracle's real colliders (release-to-dynamic -> settle -> de-penetrate
    -> read the true rest -> re-freeze kinematic), which fixes floating and object-vs-
    furniture penetration; an all-pairs separation pass then catches object-vs-object
    interpenetration. The corrected rest is written back into the motion frames so the
    render shows the physics-valid pose.

Gated on ``scene.post_action_check``, which the render sets for EVERY scene (the de-specialized
unified pipeline), so this runs after every action on both the BEHAVIOR and native Isaac paths.

This pass is PURE PHYSICS: it moves each object to its nearest physically-valid rest (settle +
de-penetrate) and reports interpenetration. It carries NO placement-quality / task heuristic --
"did the object end up where the command intended" is a separate, read-only, post-hoc judgement
(the goal check), which never moves an object or gates this pass.

TWO INDEPENDENT SCORES (the picture score is NOT the task score).
``human_clean`` (zero flagged frames) is retained verbatim for continuity but is NO LONGER a
verdict: this benchmark grades REASONING, so a body capsule brushing furniture must never suppress
the task read. Human-body overlap is retained as telemetry rather than blocking an egocentric
recording; only an FPV eye-camera intersection is a hard human-geometry failure. The picture quality
is the GRADED ``render_grade``
(``clean`` / ``marginal`` / ``dirty``), keyed on ``human_worst_frac`` = penetration depth as a
FRACTION OF THE OFFENDING CAPSULE'S OWN RADIUS -- scale-free, so a 140 mm pelvis and a 52 mm
forearm are not judged against one absolute millimetre budget. A body-derived ``dirty`` grade is
telemetry, not a veto; an FPV eye-camera intersection remains a hard failure.

ONE VERDICT, ONE PATH. ``grade_trajectory`` is the only sweep-and-grade, ``grade_render`` the only
cut points, and ``render_verdict`` the only way to read the answer back -- so the RTX render path and
the fast ``--check`` pre-flight cannot reach different conclusions about the same motion. They did:
``--check`` used to AND its own zero-tolerance boolean over a pre-nudge pose the renderer never draws,
so one episode could be ``clean`` on the render path and FAIL the gate.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import replace

import numpy as np

LOG: logging.Logger = logging.getLogger("action_check")

_DEFAULT_OBJ: str = "/World/Obj"

# A recorded rest is computed from visible geometry, while PhysX resolves the authored collision hull.
# Starting the dynamic actor exactly at that pose can therefore begin with a millimetre-scale overlap.
# Give every released prop the same small free-fall gap so contact starts from separation instead.
_RELEASE_CLEARANCE_M: float = 0.01

# --- render grade (the PICTURE score) --------------------------------------------------------
# Cut points are fractions of the offending capsule's radius. ``pen = radius - clearance``, so
# ``frac`` is where the collision hull cuts the limb: frac=1.0 puts the bone AXIS on the hull
# (half the limb buried), frac=0.90 buries ~44% of the limb cross-section, frac=0.50 ~20%. The
# owner's bar ("only fail when half a body part is inside") therefore lands at 0.90, and 0.50 is
# where the clipping first becomes visible on close inspection.
_MARGINAL_FRAC: float = 0.50
_DIRTY_FRAC: float = 0.90
# The two big trunk capsules. A half-buried TRUNK reads far worse on camera than a half-buried
# limb, so inside the marginal band it escalates. Deliberately NOT "any torso contact": a measured
# episode grazes the pelvis by 3.5 mm (2.5% of its radius) during a place, which is invisible.
_TORSO_CAPSULES: frozenset[str] = frozenset({"pelvis", "spine"})
# Lower-body clipping is retained in telemetry but does not by itself invalidate an egocentric
# recording: these capsules are outside the camera view and do not carry the manipulated object.
_LOWER_BODY_CAPSULES: frozenset[str] = frozenset(
    {"l_thigh", "r_thigh", "l_shin", "r_shin", "l_foot", "r_foot"}
)
_EGOCENTRIC_CORE_JOINTS: tuple[int, ...] = (0, 3, 12, 13, 14, 15, 16, 17)
_SUSTAINED_S: float = 1.0
_DEFAULT_FPS: float = 12.0
_CAMERA_CLEAR_RADIUS_M: float = 0.03
# Final-state object contacts are persistent, but millimetre-scale mesh overlap is common when
# authored visual and collision hulls differ. Keep it as dirty telemetry unless the penetration is
# plainly visible; five centimetres also keeps the measured WATCH-12 shelf failure (128 mm) fatal.
_OBJECT_EGREGIOUS_PEN_M: float = 0.05
# Stride of every penetration sweep. Public so a caller that has to pose the skeleton itself samples the
# SAME frames the sweep will read (see ``sample_indices``) instead of guessing the stride.
DEFAULT_SAMPLE_EVERY: int = 6

GRADE_CLEAN: str = "clean"
GRADE_MARGINAL: str = "marginal"
GRADE_DIRTY: str = "dirty"


def uses_egocentric_camera_collision_policy(scene) -> bool:
    """Whether avatar-body clipping is telemetry and the FPV eye is the hard geometry gate."""
    return bool(getattr(scene, "egocentric_camera_only_collision", True))


def _obj_path(fr) -> str:
    p = getattr(fr, "obj_path", None)
    return p if p else _DEFAULT_OBJ


# ------------------------------------------------------------- human (detect)


def _longest_dirty_run(hit_idx: list[int], sample_every: int) -> int:
    """Longest contiguous run of FLAGGED samples, expressed in MOTION FRAMES.

    Consecutive samples differ by ``sample_every``; the appended final frame can sit closer, so any gap
    at or below the stride continues the run. Each flagged sample stands for its whole stride window, so
    the run is ``n_consecutive * sample_every`` -- the same convention
    ``replay_compliance._replay_record`` uses for ``longest_run_s``. Frames, not seconds: this module
    does not know the render FPS, so the caller divides."""
    if not hit_idx:
        return 0
    step = max(1, int(sample_every))
    best = run = 1
    for a, b in zip(hit_idx, hit_idx[1:]):
        run = run + 1 if b - a <= step else 1
        best = max(best, run)
    return best * step


def sample_indices(n: int, sample_every: int) -> list[int]:
    """The frame indices a penetration sweep judges: every ``sample_every``-th, plus the last.

    The ONE definition of the sampling schedule. ``human_violations`` reads ``joints_traj`` at exactly
    these indices and nowhere else, so a caller that has to pose the skeleton itself may materialise
    ONLY these frames (leaving the rest of the list ``None``) instead of the whole trajectory -- which is
    what keeps the pre-nudge diagnostic from doubling the cost of ``--check``. Callers must take the
    schedule from here rather than re-deriving it; a private copy would silently sample other frames."""
    if n <= 0:
        return []
    return sorted(set(range(0, n, max(1, sample_every))) | {n - 1})


def _empty_depth() -> dict:
    return {
        "max_pen_m": 0.0,
        "by_seg_m": {},
        "desc": "",
        "frame": None,
        "saturated": False,
        "worst_frac": 0.0,
        "worst_frac_desc": "",
        "worst_frac_frame": None,
        "worst_frac_capsule": "",
        "egocentric_hit_idx": [],
        "egocentric_max_pen_m": 0.0,
        "egocentric_desc": "",
        "egocentric_frame": None,
        "egocentric_saturated": False,
        "egocentric_worst_frac": 0.0,
        "egocentric_worst_frac_desc": "",
        "egocentric_worst_frac_frame": None,
        "egocentric_worst_frac_capsule": "",
        "camera_probe_available": False,
        "camera_hit_idx": [],
        "camera_hits": [],
        "camera_hit_details": [],
    }


def _capsule_radius_m(cap_name: str) -> float:
    """Radius of the named proxy capsule, read from the SAME bone table the depth oracle samples.

    Returns 0.0 for an unknown name, which makes the fraction undefined and leaves the grade to fall
    back on absolute depth -- never to a silently-invented radius."""
    import physics_human as phh

    if cap_name == "head":
        return float(phh._HEAD_RADIUS)
    for spec in phh._BONES:
        if spec.name == cap_name:
            return float(spec.radius)
    return 0.0


def minimum_pose_dirty_depth_m() -> float:
    """Smallest absolute penetration that could make a single core-body pose dirty."""
    import init_validation as iv
    import physics_human as phh

    subset = set(iv._CORE_J)
    thresholds = [
        float(spec.radius)
        * (_MARGINAL_FRAC if spec.name in _TORSO_CAPSULES else _DIRTY_FRAC)
        for spec in phh._BONES
        if {spec.a, spec.b} <= subset
    ]
    if phh._HEAD_JOINT in subset:
        thresholds.append(float(phh._HEAD_RADIUS) * _DIRTY_FRAC)
    if not thresholds:
        raise RuntimeError("core-body collision sample set is empty")
    return min(thresholds)


def _accumulate_depth(
    cworld,
    J: np.ndarray,
    i: int,
    seg: str,
    depth: dict,
    exclude_owners: tuple[str, ...] = (),
    pairs: list[tuple[str, str]] | None = None,
) -> None:
    """Bisect the REAL penetration depth of an ALREADY-FLAGGED frame into ``depth`` (in place).

    Only flagged frames are measured, so a clean episode pays nothing. ``saturated`` is ORed over every
    measured frame: once any sample bottoms out at its capsule radius, ``max_pen_m`` is a LOWER BOUND
    for the episode and must never be quoted as a maximum.

    ``worst_frac`` tracks the depth as a fraction of the offending capsule's OWN radius, which is what
    the render grade keys on. The oracle reports each frame's deepest sample in metres, so this is the
    worst fraction AMONG THOSE per-frame winners: a shallower-but-relatively-deeper small capsule in the
    same frame is not seen here, and is covered instead by the ``saturated`` escalator."""
    import init_validation as iv

    if exclude_owners:
        pen_m, desc, sat = cworld.human_depth(
            J,
            joint_subset=iv._CORE_J,
            exclude_owners=exclude_owners,
        )
    else:
        pen_m, desc, sat = cworld.human_depth(J, joint_subset=iv._CORE_J)
    depth["by_seg_m"][seg] = max(depth["by_seg_m"].get(seg, 0.0), pen_m)
    depth["saturated"] = depth["saturated"] or sat
    if pen_m > depth["max_pen_m"]:
        depth["max_pen_m"], depth["desc"], depth["frame"] = pen_m, desc, i
    cap = desc.split(" ", 1)[0] if desc else ""
    radius_m = _capsule_radius_m(cap)
    frac = (pen_m / radius_m) if radius_m > 0.0 else 0.0
    if frac > depth["worst_frac"]:
        depth["worst_frac"] = frac
        depth["worst_frac_desc"] = desc
        depth["worst_frac_frame"] = i
        depth["worst_frac_capsule"] = cap
    if pairs is None:
        return
    egocentric_capsules = {name for name, _owner in pairs} - _LOWER_BODY_CAPSULES
    if not egocentric_capsules:
        return
    depth["egocentric_hit_idx"].append(i)
    if cap in egocentric_capsules:
        ego_pen_m, ego_desc, ego_sat = pen_m, desc, sat
    else:
        if exclude_owners:
            ego_pen_m, ego_desc, ego_sat = cworld.human_depth(
                J,
                joint_subset=_EGOCENTRIC_CORE_JOINTS,
                exclude_owners=exclude_owners,
            )
        else:
            ego_pen_m, ego_desc, ego_sat = cworld.human_depth(
                J,
                joint_subset=_EGOCENTRIC_CORE_JOINTS,
            )
    depth["egocentric_saturated"] = depth["egocentric_saturated"] or ego_sat
    if ego_pen_m > depth["egocentric_max_pen_m"]:
        depth["egocentric_max_pen_m"] = ego_pen_m
        depth["egocentric_desc"] = ego_desc
        depth["egocentric_frame"] = i
    ego_cap = ego_desc.split(" ", 1)[0] if ego_desc else ""
    ego_radius_m = _capsule_radius_m(ego_cap)
    ego_frac = ego_pen_m / ego_radius_m if ego_radius_m > 0.0 else 0.0
    if ego_frac > depth["egocentric_worst_frac"]:
        depth["egocentric_worst_frac"] = ego_frac
        depth["egocentric_worst_frac_desc"] = ego_desc
        depth["egocentric_worst_frac_frame"] = i
        depth["egocentric_worst_frac_capsule"] = ego_cap


def grade_render(
    worst_frac: float,
    saturated: bool,
    capsule: str,
    longest_run_s: float,
) -> tuple[str, bool, str]:
    """The PICTURE score: ``(grade, passes, reason)``. Pure -- no Isaac, no scene, unit-testable.

    ``grade`` is ``clean`` / ``marginal`` / ``dirty`` and ``passes`` is ``grade != "dirty"``. NEITHER is
    a veto: the caller renders and scores a dirty episode exactly like a clean one, it is only labelled.

    ``worst_frac`` is penetration / capsule radius (see ``_accumulate_depth``). The two escalators
    (trunk, duration) only act INSIDE the marginal band, so they can promote marginal -> dirty but can
    never condemn a shallow graze however long it lasts. ``saturated`` is its own dirty trigger because
    a bisection that bottoms out means some capsule is buried to its own axis -- literally half that
    body part inside the geometry, which is the bar the benchmark owner set."""
    if saturated:
        return (
            GRADE_DIRTY,
            False,
            "a capsule is buried to its own axis (bisection saturated)",
        )
    if worst_frac >= _DIRTY_FRAC:
        return (
            GRADE_DIRTY,
            False,
            f"penetration {worst_frac:.2f} of the {capsule or '?'} capsule radius "
            f">= {_DIRTY_FRAC:.2f}",
        )
    if worst_frac < _MARGINAL_FRAC:
        return (
            GRADE_CLEAN,
            True,
            f"deepest penetration {worst_frac:.2f} of the capsule radius < {_MARGINAL_FRAC:.2f}",
        )
    if capsule in _TORSO_CAPSULES:
        return (
            GRADE_DIRTY,
            False,
            f"trunk capsule {capsule} penetrated {worst_frac:.2f} of its radius "
            f">= {_MARGINAL_FRAC:.2f}",
        )
    if longest_run_s >= _SUSTAINED_S:
        return (
            GRADE_DIRTY,
            False,
            f"penetration {worst_frac:.2f} of the capsule radius sustained "
            f"{longest_run_s:.2f}s >= {_SUSTAINED_S:.2f}s",
        )
    return (
        GRADE_MARGINAL,
        True,
        f"penetration {worst_frac:.2f} of the {capsule or '?'} capsule radius, "
        f"{longest_run_s:.2f}s -- visible on close inspection only",
    )


def grade_egocentric_render(
    any_collision: bool,
    egocentric_collision: bool,
    worst_frac: float,
    saturated: bool,
    capsule: str,
    longest_run_s: float,
) -> tuple[str, bool, str]:
    """Grade what can affect the first-person recording while retaining all-body telemetry."""
    if any_collision and not egocentric_collision:
        return (
            GRADE_MARGINAL,
            True,
            "penetration is confined to lower-body capsules outside the egocentric view",
        )
    duration = longest_run_s if capsule in _TORSO_CAPSULES else 0.0
    grade, passes, reason = grade_render(
        worst_frac,
        saturated,
        capsule,
        duration,
    )
    if passes and grade == GRADE_MARGINAL and longest_run_s > 0.0 and duration == 0.0:
        reason = (
            f"penetration {worst_frac:.2f} of the {capsule or '?'} capsule radius; "
            f"contact lasted {longest_run_s:.2f}s but remained below the deep-penetration cutoff"
        )
    return grade, passes, reason


def grade_human_pose(
    cworld,
    joints: np.ndarray,
    exclude_owners: tuple[str, ...] = (),
    collision_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Grade one candidate pose with the same depth scale as the final render.

    Motion planning uses this before committing to a reach pose.  A shallow graze remains
    usable, while a capsule buried deeply enough to fail the rendered episode is rejected.
    Duration is intentionally zero for a single pose; the trajectory sweep still owns the
    separate sustained-contact rule once the complete motion exists.
    """
    pairs = collision_pairs
    if pairs is None:
        pairs = cworld.human_overlap(
            joints,
            exclude_owners=exclude_owners,
        )
    depth = _empty_depth()
    if pairs:
        _accumulate_depth(
            cworld,
            joints,
            0,
            "pose",
            depth,
            exclude_owners=exclude_owners,
            pairs=pairs,
        )
    egocentric_hits = depth["egocentric_hit_idx"]
    grade, passes, reason = grade_egocentric_render(
        bool(pairs),
        bool(egocentric_hits),
        float(depth["egocentric_worst_frac"]),
        bool(depth["egocentric_saturated"]),
        str(depth["egocentric_worst_frac_capsule"]),
        0.0,
    )
    return {
        "passes": passes,
        "grade": grade,
        "reason": reason,
        "max_pen_m": float(depth["max_pen_m"]),
        "worst_frac": float(depth["worst_frac"]),
        "capsule": str(depth["worst_frac_capsule"]),
        "saturated": bool(depth["saturated"]),
        "lower_body_only": bool(pairs) and not bool(egocentric_hits),
        "egocentric_max_pen_m": float(depth["egocentric_max_pen_m"]),
        "egocentric_worst_frac": float(depth["egocentric_worst_frac"]),
        "egocentric_capsule": str(depth["egocentric_worst_frac_capsule"]),
        "egocentric_saturated": bool(depth["egocentric_saturated"]),
    }


def render_verdict(report: dict | None) -> tuple[bool, str, str]:
    """The ONE human-vs-furniture verdict, read by BOTH the render path and the ``--check`` pre-flight.

    Returns ``(passes, grade, reason)`` straight off a ``run_action_compliance`` report, so the fast gate
    and the render cannot hold two opinions about the same trajectory. Consumers must call this instead of
    re-deriving a verdict from the raw depths -- a second derivation is how the two paths drifted apart.

    ``report is None`` means no compliance pass ran (no collision oracle): the picture is UNMEASURED, which
    is not evidence of a dirty one, so it passes with an EMPTY grade rather than a fabricated ``clean``."""
    if report is None:
        return (
            True,
            "",
            "no compliance pass ran (no collision oracle) -- human penetration unmeasured",
        )
    return (
        bool(report["render_pass"]),
        str(report["render_grade"]),
        str(report["render_grade_reason"]),
    )


def object_penetration_is_egregious(
    penetrations: list[tuple[str, str, float]],
) -> bool:
    """Whether a final object overlap is too deep to ship rather than merely dirty."""
    return any(
        -float(separation) >= _OBJECT_EGREGIOUS_PEN_M
        for _, _, separation in penetrations
    )


def grade_trajectory(
    cworld,
    joints_traj: list[np.ndarray] | None,
    frames: list,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    fps: float = _DEFAULT_FPS,
    sample_state_fn: Callable[[int], None] | None = None,
    check_camera: bool = False,
) -> dict:
    """Sweep ONE trajectory for human-vs-furniture penetration and GRADE it -- the SINGLE implementation
    of the picture score, so every caller measures the same quantity against the same cut points.

    Returns the ``human_*`` + ``render_*`` half of a ``run_action_compliance`` report. ``joints_traj``
    None skips the sweep and yields the unmeasured record (zero hits, ``clean``); absence of a measurement
    is reported as such by ``render_verdict``, never quoted as a clean measurement. Entries outside
    ``sample_indices`` are never read, so they may be ``None``.

    Grading two trajectories of ONE episode is only meaningful when they are genuinely different motions
    (as-rendered vs raw pre-nudge). Only the as-rendered one describes the video, so only it may gate.

    ``sample_state_fn`` runs immediately before each sampled frame's overlap and depth queries. The
    no-render path uses it to replay doors and drawers at their historical state instead of grading every
    body pose against one terminal USD state."""
    human_hit, human_by_seg, human_worst = 0, {}, []
    human_hit_idx: list[int] = []
    human_sampled = 0
    depth = _empty_depth()
    if joints_traj is not None:
        (
            human_hit,
            human_by_seg,
            human_worst,
            human_hit_idx,
            human_sampled,
            depth,
        ) = human_violations(
            cworld,
            joints_traj,
            frames,
            sample_every=sample_every,
            sample_state_fn=sample_state_fn,
            check_camera=check_camera,
        )
    run_frames = _longest_dirty_run(human_hit_idx, sample_every)
    egocentric_hit_idx = depth["egocentric_hit_idx"]
    egocentric_run_frames = _longest_dirty_run(egocentric_hit_idx, sample_every)
    grade, render_pass, grade_reason = grade_egocentric_render(
        human_hit > 0,
        bool(egocentric_hit_idx),
        depth["egocentric_worst_frac"],
        depth["egocentric_saturated"],
        depth["egocentric_worst_frac_capsule"],
        egocentric_run_frames / float(fps) if fps else 0.0,
    )
    return {
        "human_clean": human_hit == 0,
        "human_hit_frames": human_hit,
        "human_by_seg": human_by_seg,
        "human_worst": human_worst,
        "human_hit_idx": human_hit_idx,
        "human_sampled": human_sampled,
        "human_dirty_frac": (human_hit / human_sampled) if human_sampled else 0.0,
        "human_longest_run_frames": run_frames,
        "human_egocentric_hit_frames": len(egocentric_hit_idx),
        "human_egocentric_hit_idx": egocentric_hit_idx,
        "human_egocentric_longest_run_frames": egocentric_run_frames,
        "human_lower_body_only": human_hit > 0 and not egocentric_hit_idx,
        "sample_every": int(sample_every),
        "joint_subset": "_CORE_J",
        "human_max_pen_mm": depth["max_pen_m"] * 1000.0,
        "human_pen_mm_by_seg": {s: m * 1000.0 for s, m in depth["by_seg_m"].items()},
        "human_worst_pen_desc": depth["desc"],
        "human_worst_pen_frame": depth["frame"],
        "human_pen_saturated": depth["saturated"],
        # The SCALE-FREE depth the grade keys on: penetration / the offending capsule's own radius.
        "human_worst_frac": depth["worst_frac"],
        "human_worst_frac_capsule": depth["worst_frac_capsule"],
        "human_worst_frac_desc": depth["worst_frac_desc"],
        "human_worst_frac_frame": depth["worst_frac_frame"],
        "human_egocentric_max_pen_mm": depth["egocentric_max_pen_m"] * 1000.0,
        "human_egocentric_worst_pen_desc": depth["egocentric_desc"],
        "human_egocentric_worst_pen_frame": depth["egocentric_frame"],
        "human_egocentric_pen_saturated": depth["egocentric_saturated"],
        "human_egocentric_worst_frac": depth["egocentric_worst_frac"],
        "human_egocentric_worst_frac_capsule": depth["egocentric_worst_frac_capsule"],
        "human_egocentric_worst_frac_desc": depth["egocentric_worst_frac_desc"],
        "human_egocentric_worst_frac_frame": depth["egocentric_worst_frac_frame"],
        "camera_probe_available": depth["camera_probe_available"],
        "camera_hit_frames": len(depth["camera_hit_idx"]),
        "camera_hit_idx": depth["camera_hit_idx"],
        "camera_hits": depth["camera_hits"],
        "camera_hit_details": depth["camera_hit_details"],
        "render_grade": grade,
        "render_pass": render_pass,
        "render_grade_reason": grade_reason,
    }


def human_violations(
    cworld,
    joints_traj: list[np.ndarray],
    frames: list,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    sample_state_fn: Callable[[int], None] | None = None,
    check_camera: bool = False,
) -> tuple[int, dict[str, int], list[tuple[str, str]], list[int], int, dict]:
    """Per-action-segment human-vs-scene penetration on the SHARED oracle.

    ``joints_traj[i]`` is the AS-RENDERED joints of frame ``i`` (the caller realizes the
    frames through the SAME nudge the render applies, so this reflects the actual video).
    Returns ``(n_hit_frames, {seg: worst_pair_count}, worst_pairs_example, hit_indices,
    n_sampled, depth)``. ``worst_pairs_example`` is the ``(capsule, furniture_owner)`` pair list of the
    frame with the MOST pairs -- pair identities, NOT a depth (the query is boolean).
    ``hit_indices`` are FRAME indices into ``frames`` (not sample ordinals), so a caller can report the
    dirty fraction and the longest contiguous dirty run without re-running the sweep. ``depth`` carries
    the bisected millimetres (see ``_accumulate_depth``). ``sample_state_fn`` is called once before each
    sampled frame, keeping the boolean overlap and subsequent depth measurement on the same scene state.
    """
    import init_validation as iv
    import smplh_camera

    n = len(frames)
    if n == 0:
        return 0, {}, [], [], 0, _empty_depth()
    idxs = sample_indices(n, sample_every)
    by_seg: dict[str, int] = {}
    worst: list[tuple[str, str]] = []
    hits: list[int] = []
    depth = _empty_depth()
    sphere_overlap = getattr(cworld, "sphere_overlap", None) if check_camera else None
    depth["camera_probe_available"] = sphere_overlap is not None
    human_idx_set = set(idxs)
    query_indices = range(n) if sphere_overlap is not None else idxs
    for i in query_indices:
        if sample_state_fn is not None:
            sample_state_fn(i)
        excluded = tuple(getattr(frames[i], "collision_exclude_paths", ()) or ())
        if sphere_overlap is not None:
            eye_offset = getattr(frames[i], "camera_eye_offset", None)
            if eye_offset is None:
                eye, _aim = smplh_camera.fpv_eye_and_aim(
                    joints_traj[i],
                    getattr(frames[i], "fpv_mode", "look"),
                    np.zeros(3, dtype=np.float64),
                    frame=i,
                    root_yaw_deg=float(getattr(frames[i], "yaw", 0.0)),
                )
            else:
                eye = smplh_camera.default_policy_fpv_eye(
                    joints_traj[i],
                    getattr(frames[i], "fpv_mode", "look"),
                    root_yaw_deg=float(getattr(frames[i], "yaw", 0.0)),
                    eye_offset_world=eye_offset,
                )
            camera_owner = sphere_overlap(eye, _CAMERA_CLEAR_RADIUS_M)
            if camera_owner is not None:
                depth["camera_hit_idx"].append(i)
                depth["camera_hits"].append((i, camera_owner))
                depth["camera_hit_details"].append(
                    {
                        "frame": i,
                        "owner": camera_owner,
                        "segment": getattr(frames[i], "seg", "") or "?",
                        "fpv_mode": getattr(frames[i], "fpv_mode", "look"),
                        "openable_key": getattr(frames[i], "openable_key", None),
                        "obj_path": getattr(frames[i], "obj_path", None),
                        "root_xy": [
                            float(value)
                            for value in getattr(frames[i], "xy", (0.0, 0.0))
                        ],
                        "root_yaw": float(getattr(frames[i], "yaw", 0.0)),
                        "eye": [float(value) for value in eye],
                    }
                )
        if i not in human_idx_set:
            continue
        pairs = (
            cworld.human_overlap(
                joints_traj[i],
                joint_subset=iv._CORE_J,
                exclude_owners=excluded,
            )
            if excluded
            else cworld.human_overlap(joints_traj[i], joint_subset=iv._CORE_J)
        )
        if not pairs:
            continue
        hits.append(i)
        seg = getattr(frames[i], "seg", "") or "?"
        by_seg[seg] = max(by_seg.get(seg, 0), len(pairs))
        if len(pairs) > len(worst):
            worst = pairs
        _accumulate_depth(
            cworld,
            joints_traj[i],
            i,
            seg,
            depth,
            excluded,
            pairs,
        )
    return len(hits), by_seg, worst, hits, len(idxs), depth


# ------------------------------------------------------- objects (correct)


def _placed_objects(frames: list) -> dict[str, tuple[int, object]]:
    """Per object path, the LAST released rest = the placement to validate.

    A placement is a ``("rest", xyz)`` frame that comes AFTER the object's last ``carry``
    frame (i.e. the human put it down). Objects that are never carried (spawned decor,
    already handled by the scene-init settle) are skipped. Returns
    ``{path: (first_place_frame_index, rest_xyz)}``.
    """
    import humanoid_motion as wpp

    last_carry: dict[str, int] = {}
    for i, fr in enumerate(frames):
        if fr.obj[0] == "carry":
            last_carry[_obj_path(fr)] = i
    placed: dict[str, tuple[int, object]] = {}
    for i, fr in enumerate(frames):
        path = _obj_path(fr)
        if path not in last_carry or i <= last_carry[path]:
            continue
        if fr.obj[0] != "rest" or getattr(fr, "free_dynamic", False):
            continue
        if path not in placed:
            value = np.asarray(fr.obj[1], dtype=np.float64)
            rest = (
                value.copy()
                if value.shape == (4, 4)
                else tuple(float(v) for v in wpp.object_rest_position(value))
            )
            placed[path] = (i, rest)
    return placed


# --- full-state tracer -------------------------------------------------------
# Set TRACE_OBJ=<substring> to dump one object's ENTIRE journey through the settle: where the body is
# before we touch it, what its xform op order looks like (a stale xformOp:transform composes with the
# pose we write and lands the prim somewhere else entirely), whether the seat actually moved it, where
# its BASE ends up relative to the board it is supposed to rest on, and then the per-step trajectory of
# the settle itself -- which separates "launched at step 0" from "toppled over" from "pushed out slowly".
# Off by default; costs nothing when unset.
_TRACE_OBJ: str = os.environ.get("TRACE_OBJ", "")


def _tr(path: str) -> bool:
    return bool(_TRACE_OBJ) and _TRACE_OBJ in str(path)


def _dump_pose(stage, prim, path: str, when: str) -> None:
    """World translation + world bbox + xform op ORDER of ``prim``, tagged ``when``."""
    try:
        import fpv_scene
        from pxr import Usd, UsdGeom

        xf = UsdGeom.Xformable(prim)
        ops = [op.GetOpName() for op in xf.GetOrderedXformOps()]
        m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        bb = fpv_scene.compute_world_bbox(prim)
        mn, mx = bb.GetMin(), bb.GetMax()
        from pxr import PhysxSchema, UsdPhysics as _UP

        _rb = _UP.RigidBodyAPI(prim)
        _kin = _rb.GetKinematicEnabledAttr().Get() if _rb else "no-RigidBodyAPI"
        _grav = None
        try:
            _px = PhysxSchema.PhysxRigidBodyAPI(prim)
            _grav = _px.GetDisableGravityAttr().Get() if _px else None
        except Exception:  # noqa: BLE001
            pass
        # Did the zero written at body creation survive? ``physics:velocity`` is also what the sim can
        # write BACK to the stage, so a non-zero read here means something re-authored it after creation
        # -- which would be exactly why clearing it once at build time had no effect while clearing it
        # inside the kinematic window at placement time did.
        _av0 = _rb.GetVelocityAttr().Get() if _rb else None
        _aw0 = _rb.GetAngularVelocityAttr().Get() if _rb else None
        LOG.info(
            "TRACE[%s] %s: kinematic=%s disableGravity=%s AUTHORED v=%s w=%s",
            when,
            path,
            _kin,
            _grav,
            None if _av0 is None else tuple(round(float(x), 5) for x in _av0),
            None if _aw0 is None else tuple(round(float(x), 4) for x in _aw0),
        )
        # WHO ELSE IS HERE? Overlap the object's own volume excluding ONLY itself -- in particular NOT the
        # human. Every probe so far excluded the human proxy, so a hand still posed inside the drop point
        # would be invisible to all of them while PhysX pushes the object out of it at a constant rate.
        try:
            import carb
            from omni.physx import get_physx_scene_query_interface

            _sq = get_physx_scene_query_interface()
            _cx, _cy = 0.5 * float(mn[0] + mx[0]), 0.5 * float(mn[1] + mx[1])
            _hits: set = set()
            for _f in (0.15, 0.5, 0.85):
                _z = float(mn[2]) + _f * float(mx[2] - mn[2])
                _acc: list = []

                def _rp(h, _a=_acc):
                    _a.append(str(h.rigid_body) or str(h.collision))
                    return True

                _sq.overlap_sphere(
                    0.35 * min(float(mx[0] - mn[0]), float(mx[1] - mn[1])),
                    carb.Float3(_cx, _cy, _z),
                    _rp,
                    False,
                )
                for _h in _acc:
                    if _h and not _h.startswith(path):
                        _hits.add(_h)
            LOG.info(
                "TRACE[%s] %s: OVERLAPS (nothing excluded) = %s",
                when,
                path,
                sorted(_hits)[:6],
            )
        except Exception as _e:  # noqa: BLE001
            LOG.info("TRACE[%s] %s: overlap probe failed %s", when, path, _e)
        LOG.info(
            "TRACE[%s] %s: origin=(%.4f,%.4f,%.4f) bbox_z=[%.4f..%.4f] "
            "bbox_xy=[%.3f..%.3f]x[%.3f..%.3f] ops=%s",
            when,
            path,
            float(t[0]),
            float(t[1]),
            float(t[2]),
            float(mn[2]),
            float(mx[2]),
            float(mn[0]),
            float(mx[0]),
            float(mn[1]),
            float(mx[1]),
            ops,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.info("TRACE[%s] %s: dump failed (%s)", when, path, exc)


# ------------------------------------------------------------ ejection guard
# A settle that ends with the body still moving at speed, metres from where it was released, or under the
# floor did not PLACE anything: PhysX resolved an impossible release pose by LAUNCHING the body. Measured
# on a benchmark spec, an object left its shelf at 312 m/s and came to rest 200 m below the floor while
# the primitive still reported ``n_fail=0`` -- so the run read as an unsolved task instead of a broken
# engine, which is the one failure mode a benchmark must never have. ``_resettle_object`` deliberately
# owns no placement-QUALITY heuristic ("did it land in the intended container" is a task question, scored
# separately); this is the strictly different question of whether the physics step produced a rest at all.
#
# The thresholds sit an order of magnitude clear of every real placement outcome in the corpus: a settled
# body reads |v| ~ 1e-3 m/s (``settle_bodies`` stops at 1e-3), the worst honest failure on record is a
# 1.09 m fall off a fume-hood ledge, and the largest recorded slide is 2.76 m... on a launch. Nothing that
# merely slides, tips, or falls one shelf can trip these -- only a geometry-resolution blow-up can.
EJECT_SPEED_MPS: float = 1.0
EJECT_FALL_M: float = 3.0
EJECT_DRIFT_M: float = 5.0
EJECT_BELOW_FLOOR_M: float = 0.5
EJECTED = "EJECTED"

# ---- DISPLACEMENT (post-place verification, the owner's 放完验收) -------------------------------
# The ejection guard above asks "did the body stop moving". A body that falls off its shelf and lands on
# the floor HAS stopped moving, so that guard is silent for it -- measured: an object released at
# z=1.490 settling at z=0.075, dz=-1415 mm, with n_fail=0 and the primitive reporting OK. That signature
# (goal < 1.00 with n_fail == 0) is 13 of 133 rows across four families, and ten of them are PARTIAL
# scores, which is exactly why it reads as "the task half-worked" rather than "something broke".
#
# ⚠️ These thresholds are DETECTION ONLY. Nothing below them is used to fail a primitive, reject a site,
# or change a score -- ``displaced_paths`` has no caller that can veto. Its entire job is to make a point
# that is ALREADY lost legible in the log and the manifest. Deliberately well inside the ejection
# thresholds (0.15 m vs a 3 m fall) because the failures this is for are one-shelf falls, not blow-ups.
DISPLACED_FALL_M: float = 0.15
DISPLACED_DRIFT_M: float = 0.25


class SettleNote(str):
    """The per-path settle note, unchanged as a STRING but carrying the numbers alongside it.

    ``settle_placements`` has always returned ``dict[str, str]`` and ``ejected_paths`` has always tested
    ``startswith(EJECTED)``; subclassing ``str`` keeps every existing consumer byte-identical (the value
    still prints, compares and slices exactly as before) while giving new readers the measured
    displacement without re-deriving it from a log line or re-parsing the text."""

    dz: float = 0.0  # signed metres, settled_z - released_z (negative = it fell)
    dxy: float = 0.0  # metres travelled horizontally from the release point
    ejected: bool = False

    def __new__(
        cls, text: str, dz: float = 0.0, dxy: float = 0.0, ejected: bool = False
    ):
        self = super().__new__(cls, text)
        self.dz, self.dxy, self.ejected = float(dz), float(dxy), bool(ejected)
        return self


def displaced_paths(
    notes: dict,
    fall_m: float = DISPLACED_FALL_M,
    drift_m: float = DISPLACED_DRIFT_M,
) -> list[tuple[str, float, float]]:
    """``[(path, dz_m, dxy_m), ...]`` for placements that came to REST far from where they were released.

    The complement of ``ejected_paths``: those are the bodies that never stopped, these are the ones that
    stopped somewhere else. Ejected paths are excluded so a single failure is never reported twice.
    Notes produced before this landed (plain ``str``) carry no numbers and are simply skipped, so this can
    never invent a finding. Pure measurement -- no caller may use it to fail anything."""
    out = []
    for p, n in sorted((notes or {}).items()):
        if not isinstance(n, SettleNote) or n.ejected:
            continue
        if -float(n.dz) >= float(fall_m) or float(n.dxy) >= float(drift_m):
            out.append((p, float(n.dz), float(n.dxy)))
    return out


def _body_speed(prim) -> float | None:
    """|v| of a dynamic body, m/s. ``physics:velocity`` is a LIVE read (the sim writes it back). None when
    the body carries no rigid-body API / the attribute is unauthored -- unknown, never zero."""
    try:
        from pxr import UsdPhysics

        v = UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr().Get()
    except Exception:  # noqa: BLE001
        return None
    if v is None:
        return None
    return float((float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2) ** 0.5)


def _ejection_reason(prim, released, rested, floor_z) -> str | None:
    """Name the way this settle FAILED to produce a rest, or None if it produced one.

    Pure post-hoc measurement on the body's own numbers -- no scene, task or container knowledge -- so it
    applies to every placement the engine makes, in any scene."""
    speed = _body_speed(prim)
    if speed is not None and speed > EJECT_SPEED_MPS:
        return f"{EJECTED}:still-moving(|v|={speed:.1f}m/s after settle)"
    fall = float(released[2]) - float(rested[2])
    if fall > EJECT_FALL_M:
        return f"{EJECTED}:fell({fall:.1f}m below the release)"
    drift = (
        (float(rested[0]) - float(released[0])) ** 2
        + (float(rested[1]) - float(released[1])) ** 2
    ) ** 0.5
    if drift > EJECT_DRIFT_M:
        return f"{EJECTED}:flew({drift:.1f}m from the release)"
    if floor_z is not None and float(rested[2]) < float(floor_z) - EJECT_BELOW_FLOOR_M:
        return f"{EJECTED}:below-floor(z={float(rested[2]):.1f} vs floor {float(floor_z):.2f})"
    return None


def _resettle_object(
    ph,
    stage,
    rig,
    core,
    cworld,
    path: str,
    rest_value,
    step_fn=None,
) -> tuple[tuple[float, float, float], str]:
    """Drop ``path`` just above its recorded ORIGIN rest, settle on the REAL colliders, read the true rest.
    Returns (corrected_origin_xyz, note). ``note`` is informational only (logging) -- there is no
    pass/fail here: the settle always yields a physically-valid rest.

    The recorded rest is an ORIGIN pose (the render drives the ``("rest", xyz)`` frame as the prim
    origin). Keep the body kinematic while moving it a small fixed clearance above that pose, then release
    it through ``ph.release_to_dynamic``. PhysX therefore sees the first support contact while falling,
    instead of recreating a dynamic actor inside a collision hull whose bottom differs slightly from the
    visible bbox. There is still NO bbox-min-z / support-top round-trip.

    A dynamic settle can never leave the object floating (gravity) and de-penetrates it against the real
    colliders, so it MOVES the object to its nearest physically-valid rest -- that is the whole job here.
    NO placement-quality heuristic lives in this physics pass ("did it fall / land in the intended target"
    is a task-success question, judged separately + read-only, post-hoc). ``cworld`` is kept for API
    symmetry (the settle runs on the SAME PhysX colliders it wraps)."""
    import fpv_scene
    import humanoid_motion as wpp
    import object_physics as ophys

    rest_array = np.asarray(rest_value, dtype=np.float64)
    rest_xyz = wpp.object_rest_position(rest_array)
    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid():
        return tuple(float(value) for value in rest_xyz), "missing-prim"
    xy = (float(rest_xyz[0]), float(rest_xyz[1]))
    intended_z = float(rest_xyz[2])
    release_xyz = (xy[0], xy[1], intended_z + _RELEASE_CLEARANCE_M)
    if _tr(path):
        LOG.info(
            "TRACE[0-target] %s: recorded rest = (%.4f,%.4f,%.4f)",
            path,
            xy[0],
            xy[1],
            intended_z,
        )
        _dump_pose(stage, prim, path, "1-before-seat")
    ph.set_rigid_kinematic(prim, True)
    if rest_array.shape == (4, 4):
        release_pose = rest_array.copy()
        release_pose[:3, 3] = release_xyz
        previous_path = getattr(rig, "_obj_path", _DEFAULT_OBJ)
        previous_xform = getattr(rig, "_obj_xf", None)
        try:
            rig._obj_path = path
            core.set_object_world(rig, release_pose)
        finally:
            rig._obj_path = previous_path
            rig._obj_xf = previous_xform
    elif path == _DEFAULT_OBJ:
        core.place_obj_resting(rig, np.asarray(release_xyz, dtype=np.float64))
    else:
        # MOVE ONLY -- keep the orientation and scale the object already has.
        fpv_scene.set_world_pose(prim, release_xyz, quat_wxyz=None, scale=None)
    ph.zero_body_velocity(prim, step_fn=step_fn)
    if _tr(path):
        _dump_pose(stage, prim, path, "2-after-seat")
    _release_before = release_xyz
    ph.release_to_dynamic([prim], step_fn=step_fn)
    if _tr(path):
        try:
            from pxr import Usd, UsdGeom

            _t = (
                UsdGeom.XformCache(Usd.TimeCode.Default())
                .GetLocalToWorldTransform(prim)
                .ExtractTranslation()
            )
            _release_after = (float(_t[0]), float(_t[1]), float(_t[2]))
            _release_move = (
                sum((a - b) ** 2 for a, b in zip(_release_after, _release_before))
                ** 0.5
                * 1000.0
            )
            LOG.info(
                "TRACE[0b-release-window] %s: before=%s after=%s moved=%.1f mm",
                path,
                tuple(round(v, 4) for v in _release_before),
                tuple(round(v, 4) for v in _release_after),
                _release_move,
            )
        except Exception as _e:  # noqa: BLE001
            LOG.info("TRACE[0b-release-window] %s: failed %s", path, _e)
    # PER-STEP FORENSICS. ``physics:velocity`` turned out to be a LIVE read of the body's velocity (the
    # sim writes it back), which is the instrument this investigation was missing: the pose deltas alone
    # could not say whether the object arrives moving or is accelerated on the way. Step manually for a
    # few frames first, reading velocity, height and -- critically -- WHO IT IS TOUCHING each step, then
    # hand over to the normal settle. Same step_fn, so the trajectory is unchanged; it is only observed.
    if _tr(path):
        try:
            import carb
            from omni.physx import get_physx_scene_query_interface
            from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

            # --check never passes a step_fn (there is no render orchestrator to sync to), which silently
            # disabled the one instrument that reports CONTACTS. Fall back to the same bare step ph.settle
            # itself uses, so the forensics run wherever TRACE_OBJ is set.
            _sim_dbg = ophys.get_sim()

            def _bare_step_at_settle_hz(_s=_sim_dbg) -> None:
                # These 16 steps ARE the first 16 steps of the real settle, so they must integrate at the
                # SAME rate settle_bodies pins -- otherwise the instrument reports physics the production
                # run never does (the exact trap that cost this investigation 8 runs). Pinned per step
                # rather than around the loop so the block below stays a byte-for-byte observer.
                with ph.pinned_scene_timestep(stage, quiet=True):
                    _s.step(render=False)

            _step_dbg = step_fn or _bare_step_at_settle_hz
            _sq = get_physx_scene_query_interface()
            _rbv = UsdPhysics.RigidBodyAPI(prim)
            _xc = UsdGeom.XformCache(Usd.TimeCode.Default())
            # SETUP READOUT. A body that is KINEMATIC (or pose-driven) moves at a commanded constant speed
            # and ignores friction -- from outside that is indistinguishable from "the friction material
            # is not reaching PhysX". Print both so the next trace cannot be misread as either one.
            _cms = ph._collision_meshes(prim) or []
            _mats = []
            for _cm in _cms[:3]:
                _rel = UsdShade.MaterialBindingAPI(_cm).GetDirectBinding("physics")
                _mp = _rel.GetMaterialPath() if _rel else None
                _mu = (
                    UsdPhysics.MaterialAPI(prim.GetStage().GetPrimAtPath(_mp))
                    .GetDynamicFrictionAttr()
                    .Get()
                    if _mp
                    else None
                )
                _mats.append(f"{_mp or 'NONE'}:mu_d={_mu}")
            LOG.info(
                "FORENSIC %s SETUP: kinematic=%s driven=%s colliders=%d mats=%s",
                path,
                prim.GetAttribute("physics:kinematicEnabled").Get(),
                "step_fn/orchestrator" if step_fn is not None else "bare sim.step",
                len(_cms),
                _mats,
            )
            for _k in range(16):
                _xc.Clear()  # XformCache MEMOISES: without this every step reads step 0
                _t = _xc.GetLocalToWorldTransform(prim).ExtractTranslation()
                _v = _rbv.GetVelocityAttr().Get()
                _w = _rbv.GetAngularVelocityAttr().Get()
                _bb = fpv_scene.compute_world_bbox(prim)
                _bmn, _bmx = _bb.GetMin(), _bb.GetMax()
                _touch: set = set()
                for _f in (0.05, 0.5, 0.95):
                    _z = float(_bmn[2]) + _f * float(_bmx[2] - _bmn[2])
                    _acc: list = []

                    def _rp(h, _a=_acc):
                        _a.append(str(h.rigid_body) or str(h.collision))
                        return True

                    _sq.overlap_sphere(
                        0.5 * min(float(_bmx[0] - _bmn[0]), float(_bmx[1] - _bmn[1])),
                        carb.Float3(
                            0.5 * float(_bmn[0] + _bmx[0]),
                            0.5 * float(_bmn[1] + _bmx[1]),
                            _z,
                        ),
                        _rp,
                        False,
                    )
                    for _h in _acc:
                        if _h and not _h.startswith(path):
                            _touch.add("/".join(_h.split("/")[2:]))
                LOG.info(
                    "FORENSIC %s k=%02d kin=%s pos=(%.4f,%.4f,%.4f) base_z=%.4f |v|=%.4f "
                    "v=(%.3f,%.3f,%.3f) |w|=%.2f touching=%s",
                    path,
                    _k,
                    prim.GetAttribute("physics:kinematicEnabled").Get(),
                    float(_t[0]),
                    float(_t[1]),
                    float(_t[2]),
                    float(_bmn[2]),
                    (float(_v[0]) ** 2 + float(_v[1]) ** 2 + float(_v[2]) ** 2) ** 0.5
                    if _v
                    else -1,
                    float(_v[0]) if _v else 0,
                    float(_v[1]) if _v else 0,
                    float(_v[2]) if _v else 0,
                    (float(_w[0]) ** 2 + float(_w[1]) ** 2 + float(_w[2]) ** 2) ** 0.5
                    if _w
                    else -1,
                    sorted(_touch)[:4],
                )
                _step_dbg()
        except Exception as _e:  # noqa: BLE001
            LOG.info("FORENSIC %s: failed %s", path, _e)
    _traj: list | None = [] if _tr(path) else None
    _pos = ophys.settle_bodies(ph, stage, [path], traj_out=_traj, step_fn=step_fn)[
        path
    ][0]
    if _tr(path) and _traj:
        LOG.info("TRACE[3-settle] %s: %d physics steps recorded", path, len(_traj))
        _p0 = None
        for _i, _snap in enumerate(_traj):
            _e = _snap.get(path)
            if _e is None:
                continue
            (_tx, _ty, _tz), _q = _e
            _d = (
                0.0
                if _p0 is None
                else ((_tx - _p0[0]) ** 2 + (_ty - _p0[1]) ** 2 + (_tz - _p0[2]) ** 2)
                ** 0.5
            )
            if _i < 12 or _i % 20 == 0 or _i == len(_traj) - 1:
                LOG.info(
                    "TRACE[3-settle] %s  step %3d: (%.4f,%.4f,%.4f) step_move=%.1fmm quat=(%.2f,%.2f,%.2f,%.2f)",
                    path,
                    _i,
                    _tx,
                    _ty,
                    _tz,
                    _d * 1000.0,
                    float(_q[0]),
                    float(_q[1]),
                    float(_q[2]),
                    float(_q[3]),
                )
            _p0 = (_tx, _ty, _tz)
    if _tr(path):
        _dump_pose(stage, prim, path, "4-after-settle")
    corrected = (float(_pos[0]), float(_pos[1]), float(_pos[2]))
    _dz = corrected[2] - float(intended_z)
    _dxy = ((corrected[0] - xy[0]) ** 2 + (corrected[1] - xy[1]) ** 2) ** 0.5
    LOG.info(
        "resettle %s: origin (%.3f,%.3f,%.3f) -> settled (%.3f,%.3f,%.3f) dz=%+.0fmm dxy=%.0fmm",
        path,
        xy[0],
        xy[1],
        intended_z,
        corrected[0],
        corrected[1],
        corrected[2],
        (corrected[2] - intended_z) * 1000.0,
        ((corrected[0] - xy[0]) ** 2 + (corrected[1] - xy[1]) ** 2) ** 0.5 * 1000.0,
    )
    ejected = _ejection_reason(
        prim,
        (xy[0], xy[1], intended_z),
        corrected,
        getattr(cworld, "floor_z", None),
    )
    if ejected is not None:
        LOG.error(
            "resettle %s: %s -- released at (%.3f,%.3f,%.3f), ended (%.1f,%.1f,%.1f). "
            "This is an ENGINE failure, not a task failure; the primitive must not report success.",
            path,
            ejected,
            xy[0],
            xy[1],
            intended_z,
            corrected[0],
            corrected[1],
            corrected[2],
        )
        print(f"[eject-guard] {path}: {ejected}")
        return corrected, SettleNote(ejected, _dz, _dxy, True)
    return corrected, SettleNote("settled", _dz, _dxy, False)


def _rewrite_rests(frames: list, path: str, after_idx: int, corrected) -> int:
    """Replace post-release rest frames with one physics-derived xyz or 6-DoF transform."""
    n = 0
    for i in range(len(frames)):
        fr = frames[i]
        if _obj_path(fr) != path or i < after_idx or fr.obj[0] != "rest":
            continue
        if getattr(fr, "free_dynamic", False):
            continue
        frames[i] = replace(fr, obj=("rest", np.asarray(corrected, dtype=np.float64)))
        n += 1
    return n


# ---------------------------------------------------------------- orchestrate


def settle_placements(
    ph, stage, rig, core, cworld, all_frames, start_idx, step_fn=None
) -> dict:
    """Physically settle every object PLACED by the frames at ``all_frames[start_idx:]`` (one command's
    worth), rewrite the corrected rest back into the whole accumulated ``all_frames``, then re-freeze
    kinematic. Called PER-COMMAND from the render loop so the NEXT command's producer reads the REAL settled
    pose -- physics then stacks/spreads multiple objects into one target correctly, instead of a single
    end-of-episode batch that releases everyone at once and lets co-located drops collide mid-fall.
    ``start_idx=0`` settles every placement in ``all_frames`` (the whole-episode call). Returns per-path
    notes. Each object follows one complete lifecycle before the next begins: kinematic clearance seat ->
    release/resync -> settle -> read the true rest -> re-freeze."""
    new_frames = all_frames[start_idx:]
    placed = _placed_objects(new_frames)
    notes: dict[str, str] = {}

    import object_physics

    for path, (first_idx, rest_xyz) in placed.items():
        corrected, note = _resettle_object(
            ph, stage, rig, core, cworld, path, rest_xyz, step_fn=step_fn
        )
        corrected_transform = object_physics.read_prim_transform(stage, path)
        corrected_transform[:3, 3] = np.asarray(corrected, dtype=np.float64)
        _rewrite_rests(all_frames, path, start_idx + first_idx, corrected_transform)
        notes[path] = note
        prim = stage.GetPrimAtPath(path)
        if prim is not None and prim.IsValid():
            ph.set_rigid_kinematic(
                prim, True
            )  # re-freeze so the render drives it kinematically
    return notes


def ejected_paths(notes: dict) -> list[str]:
    """The prims this settle LAUNCHED rather than placed (see ``_ejection_reason``). Empty = every
    placement produced a real rest. The single read-back, so no caller re-derives the string test."""
    return sorted(p for p, n in (notes or {}).items() if str(n).startswith(EJECTED))


def run_action_compliance(
    ph,
    stage,
    rig,
    core,
    scene,
    cworld,
    frames: list,
    joints_traj: list[np.ndarray] | None = None,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    step_fn=None,
    settle_objects: bool = True,
    fps: float = _DEFAULT_FPS,
    sample_state_fn: Callable[[int], None] | None = None,
) -> tuple[dict, list]:
    """The universal post-action physics pass. Corrects placed objects (in-place on a COPY of
    ``frames``) and detects human penetration. Returns ``(report, frames_out)``.

    report keys: ``human_clean`` / ``human_hit_frames`` / ``human_by_seg`` /
    ``objects`` (per-path note) / ``objects_clean`` / ``obj_pen`` (object-object pairs) /
    ``render_grade`` + ``render_pass`` + ``render_grade_reason`` (the graded PICTURE score) /
    ``ok``.

    ``ok`` is the LENIENT physical verdict: neither the human nor an object has egregious geometry.
    Ordinary human contact and shallow object-object overlap remain dirty telemetry and do not veto a
    reasoning sample. Both zero-tolerance terms are still reported verbatim, so a consumer that wants
    the old strict value computes ``human_clean and objects_clean`` itself.

    The human half comes from ``grade_trajectory``; consumers read the verdict back through
    ``render_verdict(report)``. Those two functions are the ONLY place a picture verdict is decided --
    a caller that ANDs its own human-vs-furniture boolean on top has forked the verdict again.

    ``fps`` converts the longest contiguous flagged run into seconds for the duration escalator; the
    default matches ``episode_runtime.FPS``. Callers using ``sample_state_fn`` restore their desired terminal
    scene state after this function returns.
    """
    frames_out = list(frames)

    # --- OBJECTS: settle every released object on the real colliders, rewrite its rest. ---
    placed = _placed_objects(frames_out)
    obj_notes: dict[str, str] = {}
    objects_clean = True
    objects_egregious = False
    # settle_objects=False when the caller already settled each placement PER-COMMAND (P5) -- the objects are
    # already at their real rests, frozen kinematic, so this pass only RE-CHECKS penetration below.
    if settle_objects and placed:
        obj_notes = settle_placements(
            ph, stage, rig, core, cworld, frames_out, 0, step_fn=step_fn
        )
        # An EJECTED body is the one case where the settle did NOT yield a valid rest, so the sentence
        # below ("the settle always yields a physically-valid rest") does not hold and the episode is not
        # physically clean. Reported here as well as failing the primitive, so a whole-episode caller that
        # never runs the per-command path still sees it.
        if ejected_paths(obj_notes):
            objects_clean = False
            objects_egregious = True

    # objects_clean is PURE PHYSICS: the settle moved every object to a de-penetrated, non-floating rest,
    # so the only remaining physical-compliance failure is interpenetration -> object-vs-object below.
    # object-vs-object interpenetration over EVERY placed object (all read from the oracle).
    obj_pen: list[tuple[str, str, float]] = []
    if len(placed) >= 2:
        obj_pen = cworld.object_penetration(list(placed.keys()))
        if obj_pen:
            objects_clean = False
            objects_egregious = objects_egregious or object_penetration_is_egregious(
                obj_pen
            )

    # --- HUMAN: per-segment penetration on the (as-rendered) joints. ---
    legacy_fallback_requested = bool(
        getattr(scene, "egocentric_collision_fallback_used", False)
        and not getattr(scene, "strict_acceptance", False)
    )
    camera_only_policy = uses_egocentric_camera_collision_policy(scene)
    camera_check_requested = camera_only_policy or legacy_fallback_requested
    human = grade_trajectory(
        cworld,
        joints_traj,
        frames_out,
        sample_every=sample_every,
        fps=fps,
        sample_state_fn=sample_state_fn,
        check_camera=camera_check_requested,
    )
    fpv_eye_clear = bool(
        human["camera_probe_available"] and not human["camera_hit_frames"]
    )
    body_collision_waived = bool(
        camera_check_requested and fpv_eye_clear and not human["render_pass"]
    )
    if human["camera_probe_available"] and human["camera_hit_frames"]:
        human["render_grade"] = GRADE_DIRTY
        human["render_pass"] = False
        human["render_grade_reason"] = (
            f"FPV eye sphere intersected scene geometry in "
            f"{human['camera_hit_frames']} frame(s)"
        )
    elif body_collision_waived:
        human["render_grade_original"] = human["render_grade"]
        human["render_grade_reason_original"] = human["render_grade_reason"]
        human["render_grade"] = GRADE_MARGINAL
        human["render_pass"] = True
        human["render_grade_reason"] = (
            "body-only collision accepted by the egocentric camera policy; the exact FPV eye "
            "sphere stayed clear"
        )
    human["egocentric_camera_only_collision"] = camera_only_policy
    human["human_body_collision_waived"] = body_collision_waived
    human["egocentric_fallback_requested"] = legacy_fallback_requested
    human["egocentric_fallback_accepted"] = bool(
        legacy_fallback_requested and fpv_eye_clear
    )
    human["fpv_eye_clear"] = fpv_eye_clear
    run_frames = human["human_longest_run_frames"]
    run_s = run_frames / float(fps) if fps else 0.0
    report = {
        **human,
        "objects": obj_notes,
        "objects_clean": objects_clean,
        "objects_egregious": objects_egregious,
        "obj_pen": obj_pen,
        "ok": human["render_pass"] and not objects_egregious,
    }
    LOG.info(
        "action_compliance: ok=%s render_grade=%s (%s) human_clean=%s (%d/%d hit frames = %.1f%%, "
        "longest run %d frames = %.2fs, ego_hits=%d, max_pen %.0fmm%s [%s], "
        "worst_frac %.2f [%s], segs=%s, "
        "worst pairs=%s) objects_clean=%s (%s) obj_pen=%d",
        report["ok"],
        report["render_grade"],
        report["render_grade_reason"],
        report["human_clean"],
        report["human_hit_frames"],
        report["human_sampled"],
        100.0 * report["human_dirty_frac"],
        run_frames,
        run_s,
        report["human_egocentric_hit_frames"],
        report["human_max_pen_mm"],
        " (SATURATED: lower bound)" if report["human_pen_saturated"] else "",
        report["human_worst_pen_desc"],
        report["human_worst_frac"],
        report["human_worst_frac_desc"],
        report["human_by_seg"],
        report["human_worst"],
        objects_clean,
        obj_notes,
        len(obj_pen),
    )
    return report, frames_out


# ------------------------------------------------------------------ selftest


class _StubFrame:
    """Minimal Frame stand-in for the pure grouping/rewrite selftest (no Isaac / dataclass)."""

    def __init__(self, obj, seg="", obj_path=None, free_dynamic=False):
        self.obj = obj
        self.seg = seg
        self.obj_path = obj_path
        self.free_dynamic = free_dynamic


def _selftest_grade() -> None:
    """The picture score, calibrated against the MEASURED corpus (fpv_outputs/**/*_manifest.json)."""
    # A clean episode: nothing penetrates -> frac 0.
    assert grade_render(0.0, False, "", 0.0)[:2] == (GRADE_CLEAN, True)
    # bt_composting_waste s0: l_thigh r=105mm clearance=1.4mm -> frac 0.987, half the thigh in a wall.
    g, p, _ = grade_render(103.6 / 105.0, False, "l_thigh", 6 / 12.0)
    assert (g, p) == (GRADE_DIRTY, False), (g, p)
    # bt_composting_waste s0 --check: same thigh, bisection bottomed out elsewhere in the episode.
    assert grade_render(103.5 / 105.0, True, "l_thigh", 1.0)[0] == GRADE_DIRTY
    # bt_chemlab s0: pelvis r=140mm clearance=0.0mm for 6.0s -> the trunk is inside a bookcase.
    assert grade_render(1.0, True, "pelvis", 6.0)[0] == GRADE_DIRTY
    # bt_picking_up_trash --check: l_shin r=75mm clearance=6.1mm -> frac 0.919, just over the bar.
    assert grade_render(68.9 / 75.0, False, "l_shin", 18 / 12.0)[0] == GRADE_DIRTY
    # autoexp_scene_after: l_thigh r=105mm clearance=23.1mm -> frac 0.780 for 0.5s. Visible only on
    # close inspection and it does not touch the reasoning under test -> MARGINAL, and it PASSES.
    g, p, _ = grade_render(81.9 / 105.0, False, "l_thigh", 6 / 12.0)
    assert (g, p) == (GRADE_MARGINAL, True), (g, p)
    # The same 0.780 thigh held for 1.0s escalates; the same depth on the TRUNK escalates immediately.
    assert grade_render(81.9 / 105.0, False, "l_thigh", 1.0)[0] == GRADE_DIRTY
    assert grade_render(81.9 / 105.0, False, "pelvis", 0.5)[0] == GRADE_DIRTY
    # A shallow trunk graze is NOT egregious however long it lasts: bt_chemlab's `place` segment
    # brushes the pelvis by 3.5mm = 0.025 of its radius, which no viewer would notice.
    assert grade_render(3.5 / 140.0, False, "pelvis", 6.0)[0] == GRADE_CLEAN
    # Scale-free: the same 50mm penetration is marginal on a 105mm thigh and dirty on a 52mm forearm.
    assert grade_render(50.0 / 105.0, False, "l_thigh", 0.5)[0] == GRADE_CLEAN
    assert grade_render(50.0 / 52.0, False, "r_forearm", 0.5)[0] == GRADE_DIRTY


def _selftest_verdict() -> None:
    """``render_verdict`` is the single read-back, and an UNMEASURED picture is not a clean one."""
    unmeasured = render_verdict(None)
    assert unmeasured[0] is True and unmeasured[1] == "", unmeasured
    assert "unmeasured" in unmeasured[2], unmeasured

    # No trajectory -> the unmeasured record: zero hits, and the grade reads back through the SAME
    # accessor the render path uses, so the fast gate cannot invent a stricter answer.
    empty = grade_trajectory(None, None, [])
    assert empty["human_clean"] and empty["human_hit_frames"] == 0, empty
    assert render_verdict(empty) == (True, GRADE_CLEAN, empty["render_grade_reason"])

    # The measured bt_composting_waste thigh: 21mm on a 105mm capsule for one frame -> the render path
    # says clean, so --check reading the same report MUST also pass.
    graded = dict(empty)
    graded.update(
        zip(
            ("render_grade", "render_pass", "render_grade_reason"),
            grade_render(21.0 / 105.0, False, "l_thigh", 6 / 12.0),
        )
    )
    assert render_verdict(graded)[:2] == (True, GRADE_CLEAN), graded


def _selftest_eject() -> None:
    """The ejection guard: every launch is named, and nothing a real placement does trips it."""

    class _P:  # a prim with no rigid-body API -> speed unknown, so only the pose tests can fire
        pass

    def _r(released, rested, floor=0.0, speed=None):
        _saved = globals()["_body_speed"]
        globals()["_body_speed"] = lambda _prim: speed
        try:
            return _ejection_reason(_P(), released, rested, floor)
        finally:
            globals()["_body_speed"] = _saved

    rest = (1.0, 2.0, 0.9)
    # The measured failure: 312 m/s, 200 m under the floor. Every clause names itself.
    assert _r(rest, (1.0, 2.0, 0.9), speed=312.0).startswith(f"{EJECTED}:still-moving")
    assert _r(rest, (1.0, 2.0, -200.0)).startswith(f"{EJECTED}:fell")
    assert _r(rest, (40.0, 2.0, 0.9)).startswith(f"{EJECTED}:flew")
    assert _r(rest, (1.0, 2.0, -0.9)).startswith(f"{EJECTED}:below-floor")
    # ...and every real placement outcome on record does NOT: at rest, a 106 mm slide, a 1.09 m fall off
    # a fume-hood ledge to the floor, and a 2.76 m sideways slide -- all task questions, not engine ones.
    assert _r(rest, (1.0, 2.0, 0.9), speed=0.0009) is None
    assert _r(rest, (1.106, 2.0, 0.9), speed=0.0) is None
    assert _r((1.0, 2.0, 1.14), (1.2, 2.1, 0.05), speed=0.0) is None
    assert _r(rest, (3.76, 2.0, 0.05), speed=0.0) is None
    # Unknown speed is not zero speed, and an unknown floor is not z=0: neither may invent a verdict.
    assert _r(rest, (1.0, 2.0, 0.9), speed=None) is None
    assert _r(rest, (1.0, 2.0, -0.9), floor=None) is None
    # The read-back is the only string test any caller performs.
    assert ejected_paths({"/a": "settled", "/b": f"{EJECTED}:flew(9.0m)"}) == ["/b"]
    assert ejected_paths({}) == [] and ejected_paths(None) == []


def _selftest_displaced() -> None:
    """The displacement guard: it names what the ejection guard cannot see, and changes nothing else."""
    # A SettleNote IS its string -- this is what keeps every existing consumer byte-identical.
    n = SettleNote("settled", -1.415, 0.203, False)
    assert n == "settled" and str(n) == "settled" and not n.startswith(EJECTED)
    assert (n.dz, n.dxy, n.ejected) == (-1.415, 0.203, False)
    e = SettleNote(f"{EJECTED}:flew(9.0m)", -0.1, 9.0, True)
    assert ejected_paths({"/a": n, "/b": e}) == [
        "/b"
    ]  # the old read-back still works verbatim

    # THE CASE THE EJECTION GUARD IS BLIND TO: released at 1.490, came to REST on the floor at 0.075.
    # At rest -> _ejection_reason says nothing; the fall is 1.4 m, well under EJECT_FALL_M = 3 m.
    assert displaced_paths({"/a": n}) == [("/a", -1.415, 0.203)]
    # An ejected body is reported ONCE, by the ejection guard, never also here.
    assert displaced_paths({"/b": e}) == []
    # Nothing a good placement does trips it: a 1 mm settle and a 106 mm slide are both silent.
    assert displaced_paths({"/c": SettleNote("settled", -0.001, 0.001)}) == []
    assert displaced_paths({"/d": SettleNote("settled", -0.002, 0.106)}) == []
    # Either axis alone is enough: a straight drop, and a pure sideways departure.
    assert len(displaced_paths({"/e": SettleNote("settled", -0.30, 0.0)})) == 1
    assert len(displaced_paths({"/f": SettleNote("settled", 0.0, 0.40)})) == 1
    # Rising is not falling -- a body pushed UP off a shelf is the ejection guard's business.
    assert displaced_paths({"/g": SettleNote("settled", +0.30, 0.0)}) == []
    # A note from before this landed carries no numbers, so it can never invent a finding.
    assert displaced_paths({"/h": "settled"}) == []
    assert displaced_paths({}) == [] and displaced_paths(None) == []


def _selftest() -> None:
    """Pure-logic checks (NO Isaac): placement detection + rest rewrite + the grade + the verdict."""
    logging.basicConfig(level=logging.INFO)
    from dataclasses import dataclass

    @dataclass
    class F:
        obj: tuple
        seg: str = ""
        obj_path: str | None = None
        free_dynamic: bool = False

    # A single object: rest(origin) -> carry -> rest(dest) x3. Only the dest run is a placement.
    frames = [
        F(("rest", (0.0, 0.0, 0.7)), "arrive"),
        F(("carry", None), "grasp"),
        F(("carry", None), "walk"),
        F(("rest", (1.0, 2.0, 0.9)), "place"),
        F(("rest", (1.0, 2.0, 0.9)), "retract"),
        F(("rest", (1.0, 2.0, 0.9)), "away"),
    ]
    placed = _placed_objects(frames)
    assert set(placed) == {_DEFAULT_OBJ}, placed
    first_idx, xyz = placed[_DEFAULT_OBJ]
    assert first_idx == 3 and xyz == (1.0, 2.0, 0.9), (first_idx, xyz)

    # rewrite the dest rests to a corrected z; the ORIGIN rest (idx 0) must be untouched.
    n = _rewrite_rests(frames, _DEFAULT_OBJ, first_idx, (1.0, 2.0, 0.82))
    assert n == 3, n
    assert tuple(frames[0].obj[1]) == (0.0, 0.0, 0.7), frames[0].obj
    assert abs(float(frames[3].obj[1][2]) - 0.82) < 1e-9, frames[3].obj

    # Two objects distinguished by obj_path; a free_dynamic rest is skipped (PhysX owns it).
    frames2 = [
        F(("carry", None), "g", "/World/Obj_a"),
        F(("rest", (1.0, 0.0, 0.9)), "p", "/World/Obj_a"),
        F(("carry", None), "g", "/World/Obj_b"),
        F(("rest", (2.0, 0.0, 0.9)), "p", "/World/Obj_b", free_dynamic=True),
    ]
    placed2 = _placed_objects(frames2)
    assert set(placed2) == {"/World/Obj_a"}, (
        placed2
    )  # b is free_dynamic -> PhysX-owned, skipped

    _selftest_grade()
    _selftest_verdict()
    _selftest_eject()
    _selftest_displaced()

    print("[action_check] _selftest PASS")


if __name__ == "__main__":
    _selftest()

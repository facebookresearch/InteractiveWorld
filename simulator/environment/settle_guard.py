# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reject a per-command settle result that is not a physically-valid REST of the intended placement.

WHY. ``action_check.settle_placements`` -> ``action_check._resettle_object`` seats a just-placed object
at the rest the producer recorded, runs ``object_physics.settle_bodies`` -> ``physics_human.settle``, and
then writes whatever pose comes back into the world AND into the motion frames
(``action_check._rewrite_rests``). That write-back is UNCONDITIONAL: ``physics_human.settle`` returns after
``max_steps`` whether or not the body converged, ``object_physics.settle_bodies`` hard-codes its per-body
``ok`` to ``True`` ("ok_placeholder"), and ``_resettle_object`` asserts in its own docstring that "the
settle always yields a physically-valid rest" -- which is false on the ``max_steps`` exit.

On some headless GPU nodes that assumption breaks and the consequence is unrecoverable. A typical
log::

    settle: 80 physics steps (render=False), final |v|=13.08000 m/s     <- ran out of steps, still falling
    resettle /World/Obj_can_of_soda_113: origin (6.470,1.260,0.265)
                                      -> settled (17.424,5.583,-4.255)  <- 11.8 m away, 4.3 m BELOW the floor

The corrupted pose becomes the entity's rest, so ``init_validation._entity_contact`` centres the next
stand search on a point under the floor / outside the map and ``_search_reachable_stand`` cannot return
anything -- ``[stand-scan] ... gate[ik=52] -> NONE`` (every candidate fails the reach IK) or
``cells=0 -> NONE`` (the whole candidate ring is off-grid). Every later
``[Grab]`` of that object is then a permanent no-op and the episode can never score.

WHAT THIS DOES. Wrap the two physical-world functions from the environment side (the same module-attribute
boundary technique ``IsaacEnv.switch_scene`` uses for ``fpv_core.launch_app`` -- no simulator.physical_world edit) and
veto a correction the settle cannot justify, keeping the producer's intended rest instead. Three
rejection tests, all derived from existing scene state (no per-scene tuning):

  * ``off-map``        -- the corrected XY falls outside ``scene.nav_grid``. Such an object can NEVER be
    picked up again by construction: the stand scan generates candidates from grid cells only.
  * ``sunk``           -- the rest is below the floor the human walks on (``scene.floor_z``) by more than
    the object's own size, i.e. it fell out of the world.
  * ``risen``          -- the rest is ABOVE where the object was released by more than its own size.
    A settle only ever moves a body DOWN; the one legitimate upward motion is a contact de-penetration,
    which cannot exceed the body's own extent. (Measured: legitimate settles rise about +14 mm;
    corrupted settles rise +549 to +1869 mm.)

Deliberately NOT a "did it converge" test either: hitting ``max_steps`` means SLOW, not impossible. Such a
test rejects legitimate settles (e.g. two cans still rolling gently in a bin) and lowers the score of every
later episode, while the corruption above trips ``sunk`` and ``off-map`` on its own. Non-convergence is
logged and counted (``stats()["not_converged"]``) but never vetoes. See ``_reject_reason``.

Deliberately NOT a distance test: a legitimate settle can slide an object well over half a metre (a
banana rolls off a counter onto the floor -- dxy 544 mm, dz -657 mm), so any drift threshold tight enough
to catch a 969 mm corruption also vetoes real physics and would float the object in mid-air, which the
INSIDE goal check would then wrongly score. Every test above is one-directional: it only fires on a pose
that is physically impossible for a drop-and-settle.

A veto restores the prim to the intended rest and freezes it kinematic immediately, so a later object's
settle in the same batch cannot drop it again, and returns the intended rest so ``_rewrite_rests`` bakes
the sane pose into the motion frames.

This is CONTAINMENT, not a physics fix: it bounds the damage of a bad settle to "the put landed where it
was aimed" instead of "the object left the world". ``ISAAC_SETTLE_GUARD=0`` disables it.

DISPLACEMENT IS COUNTED, NEVER VETOED (``_note_displacement``). The three tests above are all about a pose
that is physically IMPOSSIBLE. They are silent on the commonest real failure: a put released in mid-air,
the object falls, lands on the floor and stops -- a perfectly valid rest that is simply nowhere near where
it was aimed (measured: ``origin (...,1.490) -> settled (...,0.075)``, 1415 mm down). ``sunk`` does not
fire (it is above the floor), ``risen`` does not fire (it went down), ``off-map`` does not fire. So the
episode reads as "settled fine" and the only surviving symptom is a partial score with zero errors,
which looks like a partial success. The counters below give that signature a name
(``goal < 1.00 and n_fail == 0 and displaced > 0``) WITHOUT touching the accept/reject decision: a
distance veto is exactly what this file already refuses to add, because a banana legitimately rolls off a
counter (dxy 544 mm, dz -657 mm) and vetoing that would float it in mid-air for the INSIDE check to
misread.
"""

from __future__ import annotations

import logging
import math
import os

logger: logging.Logger = logging.getLogger(__name__)

# Filled by the physics_human.settle wrapper, read by the _resettle_object wrapper: (steps, max_steps) of
# the settle that ran inside the current _resettle_object call. Single-threaded per sim process (one
# IsaacEnv drives one stage), so a module-level slot is sufficient.
_LAST_SETTLE: dict[str, tuple[int, int] | None] = {"stat": None}
_STATS: dict[str, int] = {"checked": 0, "rejected": 0}


def stats() -> dict[str, int]:
    """``{checked, rejected, displaced, grounded, not_converged}`` over the life of the process.

    Only ``rejected`` reflects a decision; the rest are telemetry for the rollout logs."""
    return dict(_STATS)


def _rest_xyz(value) -> tuple[float, float, float]:
    """Canonical XYZ for either a legacy point or a full 4x4 rest transform."""
    import humanoid_motion as wpp

    xyz = wpp.object_rest_position(value)
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _grid_holds(scene, xy) -> bool:
    """True when ``xy`` maps to a cell inside ``scene.nav_grid`` -- the same bounds test
    ``init_validation._search_reachable_stand`` uses to generate stand candidates. No grid -> vacuously
    true (a scene without navigation cannot make this judgement)."""
    grid = getattr(scene, "nav_grid", None)
    if grid is None:
        return True
    try:
        i, j = grid.world_to_cell(float(xy[0]), float(xy[1]))
        return 0 <= i < grid.nx and 0 <= j < grid.ny
    except (AttributeError, TypeError, ValueError):
        return True


def _obj_size(stage, path: str) -> float:
    """The object's own largest world-bbox extent -- the budget for the two Z tests. Largest (not the Z
    extent) so a long object that settles on its side is never mistaken for a corruption. 0.0 when the
    bbox is unavailable/degenerate, which makes both Z tests strictly tighter but never wrong-signed."""
    import fpv_scene

    try:
        prim = stage.GetPrimAtPath(path)
        rng = fpv_scene.compute_world_bbox(prim)
        mn, mx = rng.GetMin(), rng.GetMax()
        ext = [float(mx[k]) - float(mn[k]) for k in range(3)]
        return max(e for e in ext if e == e and abs(e) != float("inf"))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0.0


def _note_displacement(scene, size: float, path: str, intended, corrected) -> None:
    """Count -- never veto -- a settle that came to rest somewhere other than where it was aimed.

    Two counters, both keyed off the object's OWN size so there is no metre constant and no per-scene
    tuning, and both skipped when the bbox is unmeasurable (size 0.0) rather than firing on everything:

      * ``displaced`` -- the rest is more than one body-length from the intended rest, in Z or in XY.
      * ``grounded``  -- the stricter, named signature: aimed at least two body-lengths above the floor
        and came to rest within one of it, i.e. it fell all the way down and stopped. This is the case
        that reads as a partial success.

    A veto here is forbidden: ``_reject_reason``'s docstring records that any drift threshold tight
    enough to catch a corruption also rejects a banana rolling off a counter, and a rejected settle
    leaves the body floating where the INSIDE check would wrongly score it."""
    if size <= 0.0:
        return
    try:
        intended_xyz = _rest_xyz(intended)
        corrected_xyz = _rest_xyz(corrected)
    except (IndexError, TypeError, ValueError) as exc:
        logger.warning(
            "[settle-guard] displacement telemetry skipped for %s: "
            "intended=%s shape=%s corrected=%s shape=%s (%s)",
            path,
            type(intended).__name__,
            getattr(intended, "shape", None),
            type(corrected).__name__,
            getattr(corrected, "shape", None),
            exc,
        )
        return
    drop = intended_xyz[2] - corrected_xyz[2]
    dxy = math.hypot(
        corrected_xyz[0] - intended_xyz[0],
        corrected_xyz[1] - intended_xyz[1],
    )
    if drop <= size and dxy <= size:
        return
    _STATS["displaced"] = _STATS.get("displaced", 0) + 1
    floor_z = float(getattr(scene, "floor_z", 0.0))
    grounded = (
        intended_xyz[2] - floor_z > 2.0 * size and corrected_xyz[2] - floor_z <= size
    )
    if grounded:
        _STATS["grounded"] = _STATS.get("grounded", 0) + 1
    logger.warning(
        "[settle-guard] DISPLACED%s %s: aimed (%.3f,%.3f,%.3f) -> rested (%.3f,%.3f,%.3f), "
        "fell %.0fmm and slid %.0fmm against its own %.0fmm size -- COUNTED ONLY, this note never "
        "vetoes (the verdict is the three tests below); 'goal<1.00 with n_fail==0' is its signature",
        " ON THE FLOOR" if grounded else "",
        path,
        intended_xyz[0],
        intended_xyz[1],
        intended_xyz[2],
        corrected_xyz[0],
        corrected_xyz[1],
        corrected_xyz[2],
        drop * 1000.0,
        dxy * 1000.0,
        size * 1000.0,
    )


def _reject_reason(scene, stage, path, intended, corrected) -> str | None:
    """Why this correction must not be written back, or None to accept it.

    NOT-CONVERGED IS DELIBERATELY NOT A REJECTION -- it is counted and logged only. As a rejection it
    fires on legitimate slow settles, e.g. two cans still rolling gently inside a bin at step 80
    (``not-converged (80/80 steps)``), throwing away their real settled pose and lowering the score of
    every later episode, while the three tests below do not fire on them.

    Hitting ``max_steps`` means the settle was SLOW, which is not evidence that the pose is impossible.
    A corrupted settle -- settled (17.424, 5.583, -4.255) from origin (6.470, 1.260, 0.265) -- trips
    ``sunk`` (4.3 m under the floor) AND ``off-map`` (11.8 m away, off the nav grid) on its own, so a
    convergence proxy adds no recall and only false positives. Keep only the three DIRECT,
    one-directional tests below.
    """
    stat = _LAST_SETTLE["stat"]
    if stat is not None and stat[0] >= stat[1]:
        _STATS["not_converged"] = _STATS.get("not_converged", 0) + 1
        logger.info(
            "[settle-guard] %s settle did not converge (%d/%d steps) -- ACCEPTING the settled pose "
            "anyway; only a physically impossible rest is vetoed",
            path,
            stat[0],
            stat[1],
        )
    size = _obj_size(stage, path)
    # Counted before the vetoes so a displaced rest is on the record whatever the verdict turns out to
    # be. Pure telemetry: it reads nothing the tests below write and writes nothing they read.
    _note_displacement(scene, size, path, intended, corrected)
    if not _grid_holds(scene, corrected):
        return "off-map (outside scene.nav_grid -> no stand can ever be found)"
    floor_z = float(getattr(scene, "floor_z", 0.0))
    if float(corrected[2]) < floor_z - size:
        return f"sunk (z={corrected[2]:.3f} below floor_z={floor_z:.3f} by more than its {size:.3f}m size)"
    rise = float(corrected[2]) - float(intended[2])
    if rise > size:
        return f"risen +{rise * 1000:.0f}mm, more than its own {size * 1000:.0f}mm size"
    return None


def _restore(ph, stage, rig, core, path: str, rest_value, rest_xyz) -> None:
    """Restore the producer's intended point or full pose and freeze the prim."""
    import fpv_scene
    import numpy as np

    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid():
        return
    rest_array = np.asarray(rest_value, dtype=np.float64)
    if rest_array.shape == (4, 4):
        previous_path = getattr(rig, "_obj_path", "/World/Obj")
        previous_xform = getattr(rig, "_obj_xf", None)
        try:
            rig._obj_path = path
            core.set_object_world(rig, rest_array.copy())
        finally:
            rig._obj_path = previous_path
            rig._obj_xf = previous_xform
    elif path == "/World/Obj":
        core.place_obj_resting(rig, np.asarray(rest_xyz, dtype=np.float64))
    else:
        fpv_scene.set_world_pose(prim, rest_xyz, quat_wxyz=None, scale=None)
    # Freeze NOW: settle_placements settles the rest of the batch after this call, and a still-dynamic
    # restored body would just fall again during those steps (it re-freezes everything at the end anyway).
    ph.set_rigid_kinematic(prim, True)


def install(scene_getter) -> bool:
    """Idempotently wrap ``physics_human.settle`` and ``action_check._resettle_object``.

    ``scene_getter`` is a zero-arg callable returning the LIVE scene (``IsaacEnv.scene_obj``), read per
    call so a ``switch_scene`` is picked up without re-installing. Returns True when the guard is active.
    """
    if os.getenv("ISAAC_SETTLE_GUARD", "1") == "0":
        return False
    import action_check
    import physics_human

    if not getattr(physics_human.settle, "_settle_guard", False):
        _orig_settle = physics_human.settle

        def settle(
            sim, get_obj_lin_vel_fn, max_steps=120, vel_thresh=1e-3, step_fn=None
        ):
            steps = _orig_settle(
                sim,
                get_obj_lin_vel_fn,
                max_steps=max_steps,
                vel_thresh=vel_thresh,
                step_fn=step_fn,
            )
            _LAST_SETTLE["stat"] = (int(steps), int(max_steps))
            return steps

        settle._settle_guard = True  # noqa: SLF001
        physics_human.settle = settle

    if getattr(action_check._resettle_object, "_settle_guard", False):
        return True
    _orig_resettle = action_check._resettle_object

    def _resettle_object(ph, stage, rig, core, cworld, path, rest_value, step_fn=None):
        _LAST_SETTLE["stat"] = None
        corrected, note = _orig_resettle(
            ph, stage, rig, core, cworld, path, rest_value, step_fn=step_fn
        )
        _STATS["checked"] += 1
        try:
            intended_xyz = _rest_xyz(rest_value)
            corrected_xyz = _rest_xyz(corrected)
        except (IndexError, TypeError, ValueError) as exc:
            logger.warning(
                "[settle-guard] checks skipped for %s: intended=%s shape=%s "
                "corrected=%s shape=%s (%s)",
                path,
                type(rest_value).__name__,
                getattr(rest_value, "shape", None),
                type(corrected).__name__,
                getattr(corrected, "shape", None),
                exc,
            )
            return corrected, note
        why = _reject_reason(scene_getter(), stage, path, intended_xyz, corrected_xyz)
        if why is None:
            return corrected_xyz, note
        _STATS["rejected"] += 1
        logger.error(
            "[settle-guard] REJECTED %s: settled (%.3f,%.3f,%.3f) is not a rest of the intended "
            "(%.3f,%.3f,%.3f) -- %s; keeping the intended rest",
            path,
            corrected_xyz[0],
            corrected_xyz[1],
            corrected_xyz[2],
            intended_xyz[0],
            intended_xyz[1],
            intended_xyz[2],
            why,
        )
        _restore(ph, stage, rig, core, path, rest_value, intended_xyz)
        return intended_xyz, f"settle-rejected: {why}"

    _resettle_object._settle_guard = True  # noqa: SLF001
    action_check._resettle_object = _resettle_object
    return True

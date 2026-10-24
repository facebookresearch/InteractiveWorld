# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""settle_trace.py -- measure what the post-placement physics settle ACTUALLY did, step by step.

WHY. A settle after a ``[Put]`` (``action_check._resettle_object`` -> ``object_physics.settle_bodies``
-> ``physics_human.settle``) can leave an object metres underground or far from its intended rest,
which permanently breaks the object's stand search and with it the episode. The regular logs only carry
the settle's OUTPUT (``resettle ... -> settled (...)``), which cannot distinguish the three candidate
causes:

  A. the integrator blew up (an effective ``dt`` bigger than the one ``set_scene_timestep`` asked for --
     a playing RTX timeline is a known way to coarsen the settle clock);
  B. the body was seated INSIDE geometry and got de-penetrated explosively;
  C. the seat never took effect, so the body settled from somewhere else entirely and the whole
     "displacement" is a bookkeeping artefact, not physics.

They look identical in the output and completely different in the trace, so this probe records, per
resettled object:

  * ``dt_requested`` (read back off ``/World/PhysicsScene``'s ``timeStepsPerSecond``, NOT the value the
    setter was handed) vs ``dt_effective`` (measured: how far ``SimulationContext.current_time``
    actually advanced, divided by the number of steps) -- discriminates A;
  * ``seat_err_m`` = |world pose immediately AFTER the seat write - the intended rest|, plus the prim's
    ``xformOpOrder`` before/after. A non-zero seat error means the settle started somewhere we never
    asked for -- discriminates C. (``xformOp:transform`` surviving in the order after the write is the
    known composition bug: ``fpv_scene.set_world_pose`` writes translate/orient/scale, and a stale
    matrix op ordered AFTER them composes, landing the prim at ``intended + stale_translation``.)
  * ``v_traj`` -- |v| every step, not just the final value. A diverging integrator, a body converging
    slowly, and a body in unobstructed free fall have three completely different shapes; ``freefall``
    reports how well the trajectory matches ``g*n*dt`` (a match == the body touched NOTHING for the
    whole settle);
  * ``pos_head`` -- the first few post-release world positions (an explosive de-penetration shows up in
    step 1) -- discriminates B, together with ``support_top_z`` at the seat XY (what, if anything, the
    body was dropped onto) and the settle's ``exit`` reason / step count.

Pure logging, installed by the environment layer (no physical-world file is touched), gated by
``ISAAC_SETTLE_TRACE=1`` -- default OFF leaves behaviour byte-identical. Emits one
``[settle_env]`` line (the physics clock + timeline, once per process) and one ``[settle_trace]`` JSON
line per resettled object.

Install it BEFORE ``settle_guard`` so the trace sees the RAW settle output rather than the guard's
post-veto value; both are idempotent, so the order is stable across commands.
"""

from __future__ import annotations

import json
import logging
import math
import os

logger: logging.Logger = logging.getLogger(__name__)

_G: float = 9.81
_MAX_TRAJ: int = 24  # samples kept per settle (downsampled) -- keeps the log line small
_INSTALLED: bool = False

# Per-resettle scratch shared by the three wrappers. One IsaacEnv drives one stage in one process, and
# the settle is synchronous, so a module-level slot is sufficient (same technique as settle_guard).
_CUR: dict = {}


def enabled() -> bool:
    return os.getenv("ISAAC_SETTLE_TRACE", "0") != "0"


def _rest_xyz(value) -> tuple[float, float, float]:
    """Canonical XYZ for either a legacy point or a full 4x4 rest transform."""
    import humanoid_motion as wpp

    xyz = wpp.object_rest_position(value)
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


# --------------------------------------------------------------------------- readback helpers


def _world_xyz(stage, path: str) -> list[float] | None:
    """World-space origin of ``path`` read fresh (no cached XformCache), or None."""
    try:
        from pxr import Usd, UsdGeom

        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        return [float(m[3][0]), float(m[3][1]), float(m[3][2])]
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _xform_ops(stage, path: str) -> list[str]:
    """The prim's authored ``xformOpOrder``. A surviving ``xformOp:transform`` after a
    ``set_world_pose`` is the stale-matrix composition that silently offsets the seat."""
    try:
        from pxr import UsdGeom

        order = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetXformOpOrderAttr().Get()
        return [str(o) for o in (order or ())]
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return []


def _scene_dt(stage) -> dict:
    """The PhysX clock as AUTHORED on the scene prim -- what PhysX will actually integrate at, read back
    rather than trusting whatever ``physics_human.set_scene_timestep`` was handed."""
    out: dict = {}
    try:
        from pxr import PhysxSchema, UsdPhysics

        prim = stage.GetPrimAtPath("/World/PhysicsScene")
        tsps = PhysxSchema.PhysxSceneAPI(prim).GetTimeStepsPerSecondAttr().Get()
        if tsps:
            out["tsps_authored"] = float(tsps)
            out["dt_substep_authored"] = 1.0 / float(tsps)
        g = UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Get()
        if g is not None:
            out["gravity"] = float(g)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        out["scene_err"] = f"{type(exc).__name__}: {exc}"
    return out


def _sim_clock() -> dict:
    """The SimulationContext's own dt + wall position. ``sim.step()`` advances by ``physics_dt``, which
    can DISAGREE with the scene's authored substep rate (PhysX then runs several substeps per call)."""
    out: dict = {}
    try:
        import object_physics

        sim = object_physics.get_sim()
        for attr, key in (
            ("get_physics_dt", "physics_dt"),
            ("get_rendering_dt", "rendering_dt"),
        ):
            fn = getattr(sim, attr, None)
            if callable(fn):
                out[key] = float(fn())
        for attr in ("current_time", "current_time_step_index"):
            val = getattr(sim, attr, None)
            if isinstance(val, (int, float)):
                out[attr] = float(val)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        out["sim_err"] = f"{type(exc).__name__}: {exc}"
    return out


def _timeline() -> dict:
    """Timeline state at settle time: a PLAYING timeline is the known way the fine settle dt gets
    coarsened."""
    out: dict = {}
    try:
        import omni.timeline  # type: ignore

        tl = omni.timeline.get_timeline_interface()
        out["playing"] = bool(tl.is_playing())
        out["tcps"] = float(tl.get_time_codes_per_seconds())
        out["time"] = float(tl.get_current_time())
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        out["timeline_err"] = f"{type(exc).__name__}: {exc}"
    return out


def _sim_time() -> float | None:
    try:
        import object_physics

        t = getattr(object_physics.get_sim(), "current_time", None)
        return float(t) if isinstance(t, (int, float)) else None
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- trajectory analysis


def _downsample(vals: list[float], n: int = _MAX_TRAJ) -> list[float]:
    if len(vals) <= n:
        return [round(v, 5) for v in vals]
    idx = sorted({round(i * (len(vals) - 1) / (n - 1)) for i in range(n)})
    return [round(vals[i], 5) for i in idx]


def _freefall_fit(vals: list[float], dt: float) -> dict:
    """How well the measured speed sequence matches UNOBSTRUCTED free fall.

    ``object_physics.settle_bodies`` reports speed as a USD-pose finite difference scaled by a hardcoded
    60, so a body touching nothing reads ``60 * g * dt^2 * n`` on step ``n``. A close match means the
    settle never made contact -- the body was released into empty space; a poor match means it was
    interacting with something. Returns the implied dt too: solving the last sample for ``dt`` recovers
    the EFFECTIVE integration step independently of any API readback."""
    if not vals or dt <= 0.0:
        return {}
    n = len(vals)
    pred = [60.0 * _G * dt * dt * (i + 1) for i in range(n)]
    denom = max(pred[-1], 1e-9)
    err = max(abs(vals[i] - pred[i]) for i in range(n)) / denom
    out = {"freefall_rel_err": round(err, 4), "is_freefall": bool(err < 0.05)}
    if vals[-1] > 0.0:
        out["dt_implied_by_v"] = round(math.sqrt(vals[-1] / (60.0 * _G * n)), 6)
    return out


# --------------------------------------------------------------------------- wrappers


def _wrap_settle(physics_human) -> None:
    """Record the per-step speed + position trajectory and the true elapsed sim time of one settle."""
    if getattr(physics_human.settle, "_settle_trace", False):
        return
    orig = physics_human.settle

    def settle(sim, get_obj_lin_vel_fn, max_steps=120, vel_thresh=1e-3, step_fn=None):
        vs: list[float] = []
        ps: list[list[float]] = []
        stage, path = _CUR.get("stage"), _CUR.get("path")

        def probe():
            v = get_obj_lin_vel_fn()
            try:
                vs.append(
                    float(v)
                    if not hasattr(v, "__len__")
                    else float(sum(x * x for x in v) ** 0.5)
                )
            except (TypeError, ValueError):
                vs.append(float("nan"))
            if stage is not None and path and len(ps) < 6:
                xyz = _world_xyz(stage, path)
                if xyz is not None:
                    ps.append([round(c, 4) for c in xyz])
            return v

        t0 = _sim_time()
        steps = orig(
            sim, probe, max_steps=max_steps, vel_thresh=vel_thresh, step_fn=step_fn
        )
        t1 = _sim_time()
        _CUR["settle"] = {
            "steps": int(steps),
            "max_steps": int(max_steps),
            "vel_thresh": float(vel_thresh),
            "exit": "max_steps" if int(steps) >= int(max_steps) else "converged",
            "v_final": round(vs[-1], 5) if vs else None,
            "v_max": round(max(vs), 5) if vs else None,
            "v_traj": _downsample(vs),
            "pos_head": ps,
            "sim_time": [t0, t1],
            "dt_effective": (
                round((t1 - t0) / steps, 6)
                if (t0 is not None and t1 is not None and steps)
                else None
            ),
            "stepped_via": "step_fn" if step_fn is not None else "sim.step",
        }
        return steps

    settle._settle_trace = True  # noqa: SLF001
    physics_human.settle = settle


def _wrap_settle_bodies(object_physics) -> None:
    """Capture the world pose the body ACTUALLY has once the seat has been written -- the measurement
    that separates "physics moved it" from "it was never seated where we asked"."""
    if getattr(object_physics.settle_bodies, "_settle_trace", False):
        return
    orig = object_physics.settle_bodies

    def settle_bodies(ph, stage, prim_paths, **kw):
        if prim_paths:
            _CUR["stage"] = stage
            _CUR.setdefault("path", prim_paths[0])
            _CUR["seat"] = _world_xyz(stage, prim_paths[0])
            _CUR["ops_after_seat"] = _xform_ops(stage, prim_paths[0])
        return orig(ph, stage, prim_paths, **kw)

    settle_bodies._settle_trace = True  # noqa: SLF001
    object_physics.settle_bodies = settle_bodies


def _support(cworld, xy) -> float | None:
    """Top of whatever the oracle says is under ``xy`` -- 'was there anything to land on'."""
    try:
        return float(cworld.support_top_z((float(xy[0]), float(xy[1]))))
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return None


def _record(path, intended, corrected, note, pre, ops_pre, cworld, stage) -> dict:
    """Assemble one settle record from the scratch slots the wrappers filled."""
    intended_xyz = _rest_xyz(intended)
    corrected_xyz = None if corrected is None else _rest_xyz(corrected)
    seat = _CUR.get("seat")
    rec: dict = {
        "path": path,
        "intended": [round(v, 4) for v in intended_xyz],
        "pre_seat": pre,
        "seat": seat,
        "corrected": [round(v, 4) for v in corrected_xyz]
        if corrected_xyz is not None
        else None,
        "note": str(note)[:60],
        "ops_before": ops_pre,
        "ops_after_seat": _CUR.get("ops_after_seat", []),
    }
    if seat is not None:
        d = [seat[i] - intended_xyz[i] for i in range(3)]
        rec["seat_err_m"] = round(math.sqrt(sum(c * c for c in d)), 4)
        rec["seat_err_xyz"] = [round(c, 4) for c in d]
        rec["SEAT_MISSED"] = bool(rec["seat_err_m"] > 0.001)
        if cworld is not None:
            top = _support(cworld, seat)
            rec["support_top_z"] = None if top is None else round(top, 4)
            if top is not None:
                rec["seat_above_support_m"] = round(seat[2] - top, 4)
    st = _CUR.get("settle") or {}
    rec.update(st)
    dt = st.get("dt_effective")
    if dt:
        rec.update(_freefall_fit(st.get("v_traj") or [], dt))
    if corrected_xyz is not None:
        d = [corrected_xyz[i] - intended_xyz[i] for i in range(3)]
        rec["dxy_m"] = round(math.hypot(d[0], d[1]), 4)
        rec["dz_m"] = round(d[2], 4)
        rec["DIVERGED"] = bool(rec["dxy_m"] > 1.0 or abs(rec["dz_m"]) > 1.0)
    if stage is not None:
        rec["ops_final"] = _xform_ops(stage, path)
    return rec


def install() -> bool:
    """Idempotently wrap the three settle layers. Returns True when the trace is active."""
    global _INSTALLED
    if _INSTALLED or not enabled():
        return _INSTALLED
    try:
        import action_check
        import object_physics
        import physics_human
    except ImportError as exc:
        logger.warning(
            "settle_trace: simulator.physical_world not importable (%s); not installed",
            exc,
        )
        return False
    _wrap_settle(physics_human)
    _wrap_settle_bodies(object_physics)

    orig = action_check._resettle_object
    if getattr(orig, "_settle_trace", False):
        _INSTALLED = True
        return True
    env_logged = {"done": False}

    def _resettle_object(ph, stage, rig, core, cworld, path, rest_xyz, step_fn=None):
        _CUR.clear()
        _CUR["stage"], _CUR["path"] = stage, path
        pre, ops_pre = _world_xyz(stage, path), _xform_ops(stage, path)
        corrected, note = orig(
            ph, stage, rig, core, cworld, path, rest_xyz, step_fn=step_fn
        )
        try:
            if not env_logged["done"]:
                env = {
                    "clock": _sim_clock(),
                    "scene": _scene_dt(stage),
                    "timeline": _timeline(),
                }
                logger.info("[settle_env] %s", json.dumps(env, sort_keys=True))
                env_logged["done"] = True
            rec = _record(path, rest_xyz, corrected, note, pre, ops_pre, cworld, stage)
            logger.info("[settle_trace] %s", json.dumps(rec, sort_keys=True))
        except (TypeError, ValueError) as exc:  # a probe must never kill a rollout
            logger.warning("settle_trace: logging failed (%s)", exc)
        return corrected, note

    _resettle_object._settle_trace = True  # noqa: SLF001
    action_check._resettle_object = _resettle_object
    _INSTALLED = True
    logger.info("[settle_trace] installed (ISAAC_SETTLE_TRACE=1)")
    return True

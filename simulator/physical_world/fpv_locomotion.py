"""Locomotion: analytic root glide along a path + the omni.anim.people runtime probe.

CHOSEN approach (frozen): self-driven analytic root glide. ``step_root`` translates
/World/Human along straight-line waypoints keeping yaw = path direction, and stores the
root pose on the rig so the analytic head/wrist transforms stay valid. In the FPV the legs
are out of frame, so a translated (possibly non-deforming) body + head bob reads as walking.

omni.anim.people (high-level GoTo) is only RUNTIME-PROBED and is never the critical path —
it is absent from the default standalone experience (verified), so the analytic glide is
primary.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from fpv_scene import set_world_pose

if TYPE_CHECKING:
    from fpv_core import LocoCfg
    from fpv_human import BipedRig

LOG = logging.getLogger("fpv_locomotion")


def plan_path(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    speed_mps: float,
    fps: int,
) -> list[tuple[float, float]]:
    """Straight-line per-frame waypoints. n = ceil(dist / speed * fps); includes the end."""
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return [(float(start_xy[0]), float(start_xy[1]))]

    n = max(1, int(math.ceil(dist / max(speed_mps, 1e-6) * fps)))
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        pts.append((start_xy[0] + dx * t, start_xy[1] + dy * t))
    LOG.info(
        "plan_path %s -> %s : %.3f m @ %.2f m/s, %d frames",
        start_xy,
        end_xy,
        dist,
        speed_mps,
        len(pts),
    )
    return pts


def step_root(rig: "BipedRig", waypoints: list[tuple[float, float]], idx: int) -> bool:
    """Set /World/Human to waypoints[clamp(idx)] with yaw = path direction. Returns walking.

    Stores the root pose on the rig (``_root_xy`` / ``_root_yaw_deg``) so the analytic head
    and wrist transforms remain the single source of truth. Returns True while idx is within
    the waypoint list (still walking).
    """
    last = len(waypoints) - 1
    clamped = min(idx, last)
    xy = waypoints[clamped]

    # Yaw from the local path tangent (segment to the next point, or previous if at end).
    if clamped < last:
        nx, ny = waypoints[clamped + 1]
        yaw = math.degrees(math.atan2(ny - xy[1], nx - xy[0]))
    elif last > 0:
        px, py = waypoints[clamped - 1]
        yaw = math.degrees(math.atan2(xy[1] - py, xy[0] - px))
    else:
        yaw = rig._root_yaw_deg

    rig._root_xy = (float(xy[0]), float(xy[1]))
    rig._root_yaw_deg = float(yaw)
    _author_root(rig)
    return idx < len(waypoints)


def hold_root(rig: "BipedRig", xy: tuple[float, float], yaw_deg: float) -> None:
    """Pin the root stationary (idle/grab/place). Keeps root pose current for analytic math."""
    rig._root_xy = (float(xy[0]), float(xy[1]))
    rig._root_yaw_deg = float(yaw_deg)
    _author_root(rig)


def maybe_enable_anim_people(loco: "LocoCfg") -> bool:
    """Runtime probe for omni.anim.people; return True if present (else analytic glide).

    Default LocoCfg keeps this off the critical path. The extension is absent from the
    default standalone tree (verified), so this normally returns False.
    """
    try:
        from isaacsim.core.utils.extensions import enable_extension

        ok = enable_extension("omni.anim.people")
        if not ok:
            LOG.info(
                "maybe_enable_anim_people: extension not enabled => analytic glide"
            )
            return False
        import omni.anim.people  # noqa: F401

        LOG.info("maybe_enable_anim_people: omni.anim.people present")
        return True
    except Exception as exc:  # noqa: BLE001 - absence is the expected/safe case
        LOG.info("maybe_enable_anim_people: unavailable (%s) => analytic glide", exc)
        return False


def _author_root(rig: "BipedRig") -> None:
    """Write the current root pose onto the /World/Human prim (translation + yaw)."""
    half = math.radians(rig._root_yaw_deg) * 0.5
    quat_wxyz = (math.cos(half), 0.0, 0.0, math.sin(half))
    set_world_pose(
        rig.skel_prim,
        (rig._root_xy[0], rig._root_xy[1], 0.0),
        quat_wxyz,
        _current_scale(rig),
    )


def _current_scale(rig: "BipedRig") -> float:
    """Read the existing uniform scale authored on /World/Human (preserve meter-scale guard)."""
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(rig.skel_prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:scale":
            val = op.Get()
            if val is not None:
                return float(val[0])
    return 1.0

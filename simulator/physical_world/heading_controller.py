# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-frame heading control for a walking body: turn WHILE walking, never as a separate beat.

This is a port of what two mature repos already do, not a scheme tuned here.

* **habitat-lab** ``HumanoidRearrangeController.calculate_walk_pose``
  (``habitat/articulated_agent_controllers/humanoid_rearrange_controller.py:168``) re-aims at the
  NEXT path waypoint every frame, slews the body heading toward it by at most
  ``turning_step_amount`` degrees, and -- the part that matters most -- keeps advancing the mocap
  cycle even on a frame where the root does not translate (``step_size = max(1, ...)``), so a pivot
  is a body MARCHING round rather than a frozen mid-stride pose rotated rigidly.
* **IsaacLab** ``UniformVelocityCommand._update_command``
  (``isaaclab/envs/mdp/commands/velocity_command.py``) computes
  ``ang_vel_z = clip(k * wrap_to_pi(heading_error), +/-w_max)`` every step while ``lin_vel_x/y``
  keep running: heading is a proportional controller running CONCURRENTLY with translation.

Neither reference has any notion of "walk the polyline, then pivot on the spot onto the goal
heading". That decomposition is what this module exists to replace: in a recorded episode, a
0.40 m reposition cost 141.5 deg of body rotation and a 0.68 m one cost 249.9 deg, because the
depart beat turned one way and the arrival beat turned back the other.

TWO DELIBERATE DEPARTURES FROM HABITAT, both because our body is a kinematic mocap replay rather
than a physics agent stepped at 120 Hz:

1. ``arrival_blend_m`` -- habitat only switches its heading target onto the goal facing once inside
   ``dist_thresh`` (0.20 m), which at its control rate is a handful of frames. At 12 fps that would
   still leave a visible terminal pivot, so the target is instead eased from "travel direction" to
   "arrival facing" over the last ``arrival_blend_m`` metres. Set it to 0 to get habitat's exact
   behaviour.
2. ``Step.gait_m`` is never zero. habitat gets this from ``max(1, ...)`` on the mocap step; here the
   gait phase is indexed by distance, so the phase always advances at the nominal walking cadence
   while ``Step.moved_m`` may be zero. A pivot therefore marks time on the spot instead of rigidly
   rotating one frozen stride, which can bury a leg in a nearby wall.

Pure ``math`` (no numpy, no Isaac) so it runs in any interpreter, including offline tools.
Angle convention matches ``gait_player`` / ``nav_planner``: Z-up, body-forward is -Y, so
``forward(yaw) = [sin(yaw), -cos(yaw))]`` and turning LEFT increases yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wrap180(deg: float) -> float:
    """Shortest signed representation of an angle, in (-180, 180]."""
    return (float(deg) + 180.0) % 360.0 - 180.0


def yaw_for_dir(dx: float, dy: float) -> float:
    """Root yaw (deg) whose body-forward points along (dx, dy). Mirrors ``gait_player.yaw_for_dir``."""
    return math.degrees(math.atan2(float(dx), -float(dy)))


def forward_xy(yaw_deg: float) -> tuple[float, float]:
    """Unit world-XY heading of a body at ``yaw_deg``. Mirrors ``nav_planner.forward_xy``."""
    a = math.radians(float(yaw_deg))
    return (math.sin(a), -math.cos(a))


def _smoothstep(s: float) -> float:
    s = max(0.0, min(1.0, float(s)))
    return s * s * (3.0 - 2.0 * s)


@dataclass(frozen=True)
class HeadingCfg:
    """Every number here is a rate or a tolerance -- none of them is per-scene."""

    # Rate limit on body yaw. 10 deg/frame at 12 fps = 120 deg/s; the codebase already uses this
    # rate for its calm turn and treats ~300 deg/s as the human ceiling.
    max_turn_deg: float = 10.0
    # habitat's MIN_ANGLE_TURN: below this the heading is taken exactly, with no slew, so the body
    # does not jitter around a target it has effectively reached.
    snap_deg: float = 5.0
    # Ease the heading target from "travel direction" onto "arrival facing" over the last N metres.
    # 0.0 reproduces habitat exactly (switch only at the goal).
    arrival_blend_m: float = 1.5
    # HARD CAP on how far the body may face off its own direction of travel while translating, and
    # therefore the ONLY thing that can halt the root -- this replaces habitat's
    # THRESHOLD_ROTATE_NOT_MOVE, which keys on the heading error instead. Ours keys on the crab
    # because that is the actual physical constraint here: the gait clip is a FORWARD walk, so every
    # degree of crab is a degree of foot skate. A person carries maybe 30-40 deg before it reads as
    # sidestepping. Whatever the arrival lead-in cannot spend inside this cap is taken as a
    # rate-limited turn at the goal -- with the legs still stepping.
    max_crab_deg: float = 35.0
    # Pure-pursuit lookahead for the HEADING ONLY: aim at the point this far ahead ALONG the
    # polyline, so the travel direction sweeps smoothly through a corner instead of jumping when the
    # body reaches it. Measured on a right-angle corner, sliding the aim point rather than hopping
    # waypoint-to-waypoint is the difference between 0 halted frames and 3. The ROOT still walks the
    # polyline exactly, so a long lookahead can never cut a corner into the furniture -- it only
    # lets the facing anticipate.
    lookahead_m: float = 1.2
    # Runaway guard for the integration loop; a 400-frame episode never approaches this.
    max_frames: int = 4000


@dataclass
class Step:
    """One rendered frame of the walk."""

    xy: tuple[float, float]
    yaw: float
    moved_m: float  # root translation this frame; 0 on a pivot frame
    gait_m: float  # gait-cycle advance this frame; NEVER 0, so the legs keep stepping
    travel_yaw: float  # heading the body is actually translating along
    turning: bool


def _polyline_remaining(
    path: list[tuple[float, float]], wp: int, pos: tuple[float, float]
) -> float:
    """Distance from ``pos`` to the end of the polyline, travelling along it from waypoint ``wp``."""
    if wp >= len(path):
        return 0.0
    total = math.dist(pos, path[wp])
    for a, b in zip(path[wp:-1], path[wp + 1 :]):
        total += math.dist(a, b)
    return total


def _aim_point(
    path: list[tuple[float, float]], wp: int, pos: tuple[float, float], lookahead: float
) -> tuple[float, float]:
    """The point exactly ``lookahead`` metres ahead ALONG the polyline, or the goal if that is nearer.

    It has to be an interpolated point, not "the next waypoint past the lookahead": a waypoint
    lookahead jumps as the body crosses a corner, which spikes the travel direction and trips the
    crab halt. Sliding the aim point continuously is what makes the facing sweep through the corner.
    """
    if wp >= len(path):
        return path[-1]
    budget = lookahead
    cur = pos
    idx = wp
    while idx < len(path):
        d = math.dist(cur, path[idx])
        if d >= budget:
            if d <= 1e-12:
                return path[idx]
            f = budget / d
            return (
                cur[0] + (path[idx][0] - cur[0]) * f,
                cur[1] + (path[idx][1] - cur[1]) * f,
            )
        budget -= d
        cur = path[idx]
        idx += 1
    return path[-1]


def _advance(
    path: list[tuple[float, float]], wp: int, pos: tuple[float, float], budget: float
) -> tuple[int, tuple[float, float], float]:
    """Slide ``budget`` metres along the polyline. Returns (waypoint, position, distance covered)."""
    covered = 0.0
    while budget > 1e-12 and wp < len(path):
        d = math.dist(pos, path[wp])
        if d <= budget:
            pos = path[wp]
            covered += d
            budget -= d
            wp += 1
        else:
            f = budget / d
            pos = (
                pos[0] + (path[wp][0] - pos[0]) * f,
                pos[1] + (path[wp][1] - pos[1]) * f,
            )
            covered += budget
            budget = 0.0
    return wp, pos, covered


def walk_track(
    path: list[tuple[float, float]],
    start_yaw: float,
    arrival_yaw: float | None,
    speed: float,
    fps: float,
    cfg: HeadingCfg | None = None,
) -> list[Step]:
    """Walk ``path`` while continuously steering the body heading. THE replacement for
    "depart beat + constant yaw per leg + corner turn beats + arrival pivot".

    ``path`` is [start, ..., goal] in world XY. ``arrival_yaw`` is the facing the body must hold at
    the goal, or None to finish along the last leg. Returns one ``Step`` per rendered frame; the
    caller maps ``Step.gait_m`` onto the gait cycle and ``Step.xy`` / ``Step.yaw`` onto the root.

    The loop is deliberately identical in shape to habitat's: aim, slew, decide whether the slew is
    big enough to halt translation, translate, repeat.
    """
    cfg = cfg or HeadingCfg()
    if len(path) < 2:
        return []
    nominal = float(speed) / float(fps)
    pos = (float(path[0][0]), float(path[0][1]))
    yaw = float(start_yaw)
    wp = 1
    out: list[Step] = []

    while len(out) < cfg.max_frames:
        remaining = _polyline_remaining(path, wp, pos)
        arrival_err = (
            0.0 if arrival_yaw is None else abs(wrap180(float(arrival_yaw) - yaw))
        )
        if remaining <= 1e-6 and arrival_err <= 1e-6:
            break

        aim = _aim_point(path, wp, pos, cfg.lookahead_m)
        dx, dy = aim[0] - pos[0], aim[1] - pos[1]
        travel_yaw = yaw_for_dir(dx, dy) if math.hypot(dx, dy) > 1e-9 else yaw

        # Heading target: the travel direction, eased onto the arrival facing over the last stretch.
        desired = travel_yaw
        if arrival_yaw is not None:
            if remaining <= 1e-6:
                desired = float(arrival_yaw)
            elif cfg.arrival_blend_m > 1e-9 and remaining < cfg.arrival_blend_m:
                t = _smoothstep(1.0 - remaining / cfg.arrival_blend_m)
                desired = travel_yaw + wrap180(float(arrival_yaw) - travel_yaw) * t

        # While still translating the body may not face further than max_crab_deg off its own
        # direction of travel: the gait clip only walks FORWARD, so crab == foot skate.
        if remaining > 1e-6:
            crab = wrap180(desired - travel_yaw)
            if abs(crab) > cfg.max_crab_deg:
                desired = travel_yaw + math.copysign(cfg.max_crab_deg, crab)

        residual = wrap180(desired - yaw)
        turning = abs(residual) > cfg.snap_deg
        if turning:
            yaw = wrap180(
                yaw + math.copysign(min(cfg.max_turn_deg, abs(residual)), residual)
            )
        else:
            yaw = wrap180(desired)

        # The root may only advance while the body is facing near enough to where it is going. The
        # test is on the POST-slew yaw, so ``max_crab_deg`` is an invariant of the emitted track and
        # not merely of the target -- a corner the lookahead failed to anticipate halts the body
        # rather than skating it sideways.
        # The epsilon is load-bearing: ``desired`` is clamped to exactly ``travel +/- max_crab_deg``
        # and ``yaw`` then lands on it, so an exact-tie comparison flips on wrap-around rounding --
        # which halts the root on a frame that also has zero heading error, and the walk terminates
        # silently half way to the goal.
        crab_now = abs(wrap180(yaw - travel_yaw))
        scale = 0.0 if crab_now > cfg.max_crab_deg + 1e-6 else 1.0
        wp, pos, moved = _advance(path, wp, pos, nominal * scale)

        if moved <= 1e-12 and abs(residual) <= 1e-9:
            break  # neither translating nor rotating: fully converged

        out.append(
            Step(
                xy=pos,
                yaw=yaw,
                moved_m=moved,
                gait_m=nominal,  # cadence is constant: a pivot marks time, it does not freeze
                travel_yaw=travel_yaw,
                turning=turning,
            )
        )
    return out


@dataclass
class TrackStats:
    """The numbers a walk is judged on. ``crab`` is the foot-skate risk: how far the body faces off
    its own direction of travel while actually translating."""

    frames: int
    total_turn_deg: float
    max_turn_deg_per_frame: float
    arrival_residual_deg: float
    path_m: float
    max_crab_deg: float
    p95_crab_deg: float
    pivot_frames: int


def track_stats(
    steps: list[Step], start_yaw: float, arrival_yaw: float | None
) -> TrackStats:
    """Summarise a track (also usable on a track reconstructed from a recorded episode)."""
    if not steps:
        return TrackStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    yaws = [float(start_yaw)] + [s.yaw for s in steps]
    deltas = [abs(wrap180(b - a)) for a, b in zip(yaws[:-1], yaws[1:])]
    crab = sorted(abs(wrap180(s.yaw - s.travel_yaw)) for s in steps if s.moved_m > 1e-9)
    residual = (
        0.0 if arrival_yaw is None else abs(wrap180(float(arrival_yaw) - steps[-1].yaw))
    )
    return TrackStats(
        frames=len(steps),
        total_turn_deg=sum(deltas),
        max_turn_deg_per_frame=max(deltas),
        arrival_residual_deg=residual,
        path_m=sum(s.moved_m for s in steps),
        max_crab_deg=(crab[-1] if crab else 0.0),
        p95_crab_deg=(crab[min(len(crab) - 1, int(0.95 * len(crab)))] if crab else 0.0),
        pivot_frames=sum(1 for s in steps if s.moved_m <= 1e-9),
    )


# =====================================================================================================
# Offline self-test (pure python, ~5 ms): representative path shapes an episode walks.
# =====================================================================================================
def _selftest() -> int:
    cfg = HeadingCfg()
    fails: list[str] = []
    shapes = {
        # A straight run that must END facing 90 deg off the travel direction. Under a walk-then-pivot
        # scheme this is "walk, then pivot 90 deg on the spot"; here most of it is spent en route.
        "straight+90": ([(0.0, 0.0), (0.0, -4.0)], 0.0, 90.0),
        # A right-angle corner: the corner turn must be SPREAD, never a single-frame snap.
        "corner": ([(0.0, 0.0), (0.0, -3.0), (3.0, -3.0)], 0.0, 90.0),
        # A sub-metre reposition with a large start-to-arrival facing change.
        "hop0.4": ([(0.0, 0.0), (-0.40, -0.02)], -32.6, 2.4),
    }
    for name, (path, y0, y1) in shapes.items():
        steps = walk_track(
            path, start_yaw=y0, arrival_yaw=y1, speed=1.1, fps=12, cfg=cfg
        )
        st = track_stats(steps, y0, y1)
        want_m = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))
        if st.arrival_residual_deg > 1e-6:
            fails.append(f"{name}: arrived {st.arrival_residual_deg:.1f} deg off")
        if abs(st.path_m - want_m) > 1e-6:
            fails.append(
                f"{name}: walked {st.path_m:.3f} m of {want_m:.3f} m (truncated)"
            )
        if st.max_turn_deg_per_frame > cfg.max_turn_deg + 1e-6:
            fails.append(
                f"{name}: {st.max_turn_deg_per_frame:.1f} deg/frame over the rate limit"
            )
        if st.max_crab_deg > cfg.max_crab_deg + 1e-6:
            fails.append(f"{name}: crabbed {st.max_crab_deg:.1f} deg while translating")
        if any(s.gait_m <= 0.0 for s in steps):
            fails.append(f"{name}: a frame advanced the gait cycle by zero")
        print(
            f"[heading] {name:<12} frames={st.frames:>3} turn={st.total_turn_deg:6.1f} deg "
            f"max={st.max_turn_deg_per_frame:5.1f}/fr crab(max/p95)={st.max_crab_deg:4.1f}/"
            f"{st.p95_crab_deg:4.1f} pivot={st.pivot_frames:>2} residual={st.arrival_residual_deg:.2f}"
        )
    print(f"[heading] selftest -> {'PASS' if not fails else 'FAIL'}")
    for f in fails:
        print(f"  FAIL {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

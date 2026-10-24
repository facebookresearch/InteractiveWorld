"""Phase-blended striding gait: turn a retargeted ACCAD walk into a body that strides along any path.

Consumes ``accad_retarget`` (Z-up, canonical -Y-facing, in-place poses). Extracts ONE loopable gait
cycle, then advances the cycle phase by DISTANCE TRAVELLED (not by frame index) so the planted foot does
not slide on straight paths regardless of render fps or chosen speed -- the key anti-foot-skate trick
(see [[isaac-arbitrary-motion-architecture]]). ``pose_place`` mirrors ``smplh_anim_core.pose_body``'s
placement (Rz(yaw)@V + [x,y,z]) but for a FULL body pose (legs+spine move), and can OVERRIDE the right arm
(+fingers) so the human can carry a held object while the legs walk.

numpy + torch FK only -- NO Isaac. Run directly for an OFFLINE self-test (top-down footprint track + a
sagittal montage) with no SimulationApp:

    bash hand/run_isaac.sh simulator.physical_world/gait_player.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HAND_DIR = os.path.join(os.path.dirname(_THIS_DIR), "human_model")
for _p in (_THIS_DIR, _HAND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import accad_retarget as A  # noqa: E402
import smplh_fpv as H  # noqa: E402

_L_ANKLE, _R_ANKLE = 7, 8


@dataclass
class Gait:
    """One loopable gait cycle in our Z-up world, in-place, canonical body facing -Y."""

    cycle: np.ndarray  # (P, 156) SMPL+H poses for one full stride cycle
    min_foot_z: (
        np.ndarray
    )  # (P,) lowest foot-joint z per frame (negative; for floor pinning)
    stride_len: float  # meters the body advances over one full cycle
    fps: float  # source fps of the cycle
    src: str  # source clip name


@dataclass
class WalkFrame:
    """One placed walk frame: full body pose + world root (xy, yaw, z)."""

    pose156: np.ndarray
    root_xy: tuple[float, float]
    yaw_deg: float
    root_z: float


def _detect_period(
    s: np.ndarray, fps: float, lo_s: float = 0.7, hi_s: float = 1.7
) -> int:
    """Dominant period (frames) of a 1-D gait signal via autocorrelation in [lo_s, hi_s] seconds."""
    s = np.asarray(s, dtype=np.float64) - float(np.mean(s))
    n = len(s)
    lo = max(2, int(lo_s * fps))
    hi = min(n - 1, int(hi_s * fps))
    best = (-1e18, lo)
    for lag in range(lo, hi):
        c = float(np.dot(s[:-lag], s[lag:]) / (n - lag))
        if c > best[0]:
            best = (c, lag)
    return best[1]


def _first_up_cross(s: np.ndarray) -> int:
    """First upward zero-crossing of a mean-subtracted signal (a stable cycle start phase)."""
    s = np.asarray(s, dtype=np.float64) - float(np.mean(s))
    for i in range(len(s) - 1):
        if s[i] <= 0.0 < s[i + 1]:
            return i
    return 0


def build_gait(npz_path: str | None = None, min_seconds: float = 1.6) -> Gait:
    """Retarget a walk clip and extract one loopable stride cycle + its ground-advance (stride_len)."""
    src = npz_path or A.default_walk_clip()
    clip = A.load_clip(src)
    up_R, _ = A.detect_up_R(clip)
    lo, hi = A.pick_straight_window(clip, up_R, min_seconds=min_seconds)
    yaw = A.canonical_yaw_deg(clip, lo, hi, up_R)
    poses = A.to_smplh_zup(clip, up_R, extra_yaw_deg=yaw)[lo : hi + 1]
    J = A.fk_joints(poses)

    # Cycle boundaries from the antisymmetric fore-aft ankle signal (one full period == one gait cycle).
    s = J[:, _R_ANKLE, 1] - J[:, _L_ANKLE, 1]
    period = _detect_period(s, clip.fps)
    i0 = _first_up_cross(s)
    if i0 + period >= len(poses):
        i0 = 0
    cyc = poses[i0 : i0 + period].copy()
    minfz = A.foot_floor_z(J[i0 : i0 + period]).copy()

    # Ground advance over the cycle = horizontal displacement of the pelvis track over the same frames.
    tz = A.trans_zup(clip, up_R, extra_yaw_deg=yaw)[:, :2]
    stride = (
        float(np.linalg.norm(tz[lo + i0 + period] - tz[lo + i0]))
        if (lo + i0 + period) < len(tz)
        else 0.7
    )
    stride = max(0.35, stride)  # guard against a degenerate near-zero measurement
    print(
        f"[gait] {os.path.basename(src)}: cycle P={period} ({period / clip.fps:.2f}s) "
        f"stride_len={stride:.2f} m/cycle up=I yaw={yaw:+.0f}"
    )
    return Gait(
        cycle=cyc,
        min_foot_z=minfz,
        stride_len=stride,
        fps=clip.fps,
        src=os.path.basename(src),
    )


def sample(gait: Gait, dist: float) -> tuple[np.ndarray, float]:
    """Cycle pose + lowest-foot-z at ground distance ``dist`` (phase advances with distance)."""
    p = len(gait.cycle)
    fidx = (float(dist) / gait.stride_len) * p
    idx = int(round(fidx)) % p
    return gait.cycle[idx], float(gait.min_foot_z[idx])


def yaw_for_dir(dx: float, dy: float) -> float:
    """Root yaw (deg) so the body's native forward (-Y) points along (dx, dy). forward(yaw)=[sin,-cos]."""
    return math.degrees(math.atan2(dx, -dy))


def plan_walk(
    gait: Gait,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    floor_z: float,
    speed: float = 1.1,
    fps: int = 12,
    start_dist: float = 0.0,
) -> tuple[list[WalkFrame], float]:
    """Per-frame striding walk from start to end. Returns (frames, cumulative_distance_at_end).

    ``start_dist`` continues the gait phase across chained segments (no cadence reset at a corner).
    root_z pins the lowest foot to ``floor_z`` each frame (kills vertical float from the betas mismatch).
    """
    s = np.asarray(start_xy, dtype=np.float64)
    e = np.asarray(end_xy, dtype=np.float64)
    d = e - s
    L = float(np.linalg.norm(d))
    yaw = yaw_for_dir(d[0], d[1]) if L > 1e-6 else 0.0
    n = max(2, int(math.ceil(L / max(speed, 1e-6) * fps)))
    frames: list[WalkFrame] = []
    for i in range(n + 1):
        f = i / n
        xy = s + d * f
        pose, mfz = sample(gait, start_dist + L * f)
        frames.append(WalkFrame(pose, (float(xy[0]), float(xy[1])), yaw, floor_z - mfz))
    return frames, start_dist + L


def pose_place(
    pose156: np.ndarray,
    root_xy: tuple[float, float],
    yaw_deg: float,
    root_z: float,
    arm_override: dict | None = None,
    rhp: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """FK a FULL 156-D pose and place it at world (x,y,z) with heading yaw (== pose_body's placement rule).

    ``arm_override`` (joint index -> aa) overrides the right arm so the body can carry a held object while
    legs/spine/other-arm follow the gait; ``rhp`` overrides the 45-D right-hand fingers (the grasp shape).
    """
    p = np.asarray(pose156, dtype=np.float32).copy()
    if arm_override:
        for j, v in arm_override.items():
            p[j * 3 : j * 3 + 3] = np.asarray(v, dtype=np.float32)
    if rhp is not None:
        p[H.RHAND_LO : H.RHAND_LO + 45] = np.asarray(rhp, dtype=np.float32)
    V, J = H.fwd(p)
    if abs(yaw_deg) > 1e-9:
        a = math.radians(yaw_deg)
        c, s = math.cos(a), math.sin(a)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        V = V @ Rz.T
        J = J @ Rz.T
    off = np.array([root_xy[0], root_xy[1], root_z])
    return V + off, J + off


def pose_place_joints(
    pose156: np.ndarray,
    root_xy: tuple[float, float],
    yaw_deg: float,
    root_z: float,
    arm_override: dict | None = None,
    rhp: np.ndarray | None = None,
) -> np.ndarray:
    """Joints-only variant of ``pose_place`` for planning and collision queries."""
    p = np.asarray(pose156, dtype=np.float32).copy()
    if p.size != H.NJ * 3:
        return pose_place(
            pose156,
            root_xy,
            yaw_deg,
            root_z,
            arm_override=arm_override,
            rhp=rhp,
        )[1]
    if arm_override:
        for joint, value in arm_override.items():
            p[joint * 3 : joint * 3 + 3] = np.asarray(value, dtype=np.float32)
    if rhp is not None:
        p[H.RHAND_LO : H.RHAND_LO + 45] = np.asarray(rhp, dtype=np.float32)
    joints = H.fwd_joints(p)
    if abs(yaw_deg) > 1e-9:
        angle = math.radians(yaw_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        joints = joints @ rotation.T
    return joints + np.array([root_xy[0], root_xy[1], root_z])


def blend156(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """Axis-angle LERP of two full poses (adequate for a short walk<->stand transition)."""
    s = max(0.0, min(1.0, float(s)))
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a + (b - a) * s).astype(np.float32)


def stand_pose_from_cycle(gait: Gait) -> np.ndarray:
    """A near-standing frame from the cycle (feet closest together) -- a natural walk<->manip bridge pose."""
    J = A.fk_joints(gait.cycle)
    sep = np.abs(J[:, _R_ANKLE, 1] - J[:, _L_ANKLE, 1])
    return gait.cycle[int(np.argmin(sep))].copy()


# =====================================================================================================
# OFFLINE self-test: plan an L-shaped walk, place frames, verify feet advance without back-slide.
# =====================================================================================================
def _selftest() -> int:
    gait = build_gait()
    floor_z = 0.0
    legs = [((0.0, 0.6), (0.0, -1.6)), ((0.0, -1.6), (1.4, -1.6))]
    frames: list[WalkFrame] = []
    dist = 0.0
    for a, b in legs:
        fr, dist = plan_walk(gait, a, b, floor_z, speed=1.1, fps=12, start_dist=dist)
        frames.extend(fr)
    print(f"[gait] planned {len(frames)} walk frames over {dist:.2f} m")

    # Place every frame; collect pelvis + ankle world positions.
    pel, la, ra, foot_res = [], [], [], []
    for wf in frames:
        J = pose_place_joints(wf.pose156, wf.root_xy, wf.yaw_deg, wf.root_z)
        pel.append(J[0])
        la.append(J[_L_ANKLE])
        ra.append(J[_R_ANKLE])
        foot_res.append(float(J[list(A._FOOT_J), 2].min()))
    pel = np.array(pel)
    la = np.array(la)
    ra = np.array(ra)
    foot_res = np.array(foot_res)
    # Pelvis should advance monotonically along the path; lowest foot should sit ~on the floor.
    fwd_prog = float(np.linalg.norm(pel[-1, :2] - pel[0, :2]))
    max_floor_pen = float(foot_res.min())
    print(
        f"[gait] pelvis advance={fwd_prog:.2f} m  lowest-foot z: min={foot_res.min() * 1000:.0f}mm "
        f"max={foot_res.max() * 1000:.0f}mm (want ~0)"
    )
    ok = fwd_prog > 2.0 and abs(max_floor_pen) < 0.05
    print(f"[gait] CHECK path-follow={'PASS' if ok else 'FAIL'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        ax1.plot(pel[:, 0], pel[:, 1], "-", color="#888", lw=1, label="pelvis")
        ax1.scatter(la[:, 0], la[:, 1], s=10, color="#d24a3a", label="L foot")
        ax1.scatter(ra[:, 0], ra[:, 1], s=10, color="#1f8f3a", label="R foot")
        for a, b in legs:
            ax1.plot([a[0], b[0]], [a[1], b[1]], "b--", lw=0.8)
        ax1.set_title("top-down: footprints along path")
        ax1.set_aspect("equal")
        ax1.legend(fontsize=7)
        ax2.plot(foot_res * 1000.0, color="#333")
        ax2.axhline(0, color="r", lw=0.8, ls="--")
        ax2.set_title("lowest foot z (mm) per frame")
        fig.tight_layout()
        out = os.path.abspath(
            os.path.join(_THIS_DIR, "..", "fpv_outputs", "gait_player_check.png")
        )
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(f"[gait] montage -> {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[gait] montage skipped ({exc})")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())

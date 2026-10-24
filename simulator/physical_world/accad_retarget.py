"""Retarget an ACCAD / AMASS SMPL-X walk clip onto our SMPL+H body as Z-up, pelvis-origin poses.

This REPLACES GRAB replay as the locomotion motion source (see [[isaac-arbitrary-motion-architecture]]).
The ACCAD "stageii" data in this tree is ALREADY Z-up (verified: raw global gives head-foot_z=+1.56 m),
so the up-conversion is auto-detected among {I, Rx+90, Rx-90} (identity wins here) and defaults to identity.
We then CANONICALIZE the mean walking heading to -Y (our native forward) so ``gait_player`` can re-place
the cycle along an arbitrary path via ``pose_body``'s root_yaw. Body joints 1..21 are shared SMPL-X<->SMPL+H;
hands are zeroed (a relaxed walking hand).

numpy + torch FK only (``smplh_fpv``) -- NO Isaac. Run directly for an OFFLINE self-test (upright / foot-plant
checks + a sagittal skeleton montage) with no SimulationApp:

    bash hand/run_isaac.sh simulator.physical_world/accad_retarget.py
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HAND_DIR = os.path.join(os.path.dirname(_THIS_DIR), "human_model")
if _HAND_DIR not in sys.path:
    sys.path.insert(0, _HAND_DIR)

import smplh_fpv as H  # noqa: E402  torch FK; import-safe offline (no omni)

_WALK_CLIP = os.environ.get("INTERACTIVEWORLD_WALK_CLIP", "")
ACCAD_ROOT = os.path.dirname(_WALK_CLIP) if _WALK_CLIP else ""

# SMPL-X poses(165): global 0:3 | body 3:66 | jaw/eyes 66:75 | lhand 75:120 | rhand 120:165.
# Our SMPL+H pose(156): global 0:3 | body 3:66 | lhand 66:111 | rhand 111:156.
_AMASS_ROOT = (0, 3)
_AMASS_BODY = (3, 66)

# Foot joints (SMPL tree): L_ankle 7, R_ankle 8, L_foot 10, R_foot 11 -> lowest = ground contact.
_FOOT_J = (7, 8, 10, 11)


@dataclass
class WalkClip:
    """A loaded AMASS/ACCAD clip: raw SMPL-X params + fps + source path."""

    poses: np.ndarray  # (T, 165) SMPL-X full pose
    trans: np.ndarray  # (T, 3) pelvis world translation
    fps: float
    src_path: str


def _rx(deg: float) -> np.ndarray:
    """Rotation about world +X by ``deg`` degrees."""
    return H.axis_rot(np.array([1.0, 0.0, 0.0]), float(deg))


def _rz(deg: float) -> np.ndarray:
    """Rotation about world +Z by ``deg`` degrees."""
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _aa_to_R(aa) -> np.ndarray:
    """Axis-angle (3,) -> 3x3 rotation (Rodrigues via H.axis_rot)."""
    aa = np.asarray(aa, dtype=np.float64)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    return H.axis_rot(aa / th, np.rad2deg(th))


# Up-conversion candidates (the correct one makes the body upright = head far above feet in +Z).
_UP_CANDIDATES = {"I": np.eye(3), "Rx+90": _rx(90.0), "Rx-90": _rx(-90.0)}


def load_clip(npz_path: str) -> WalkClip:
    """Load an AMASS/ACCAD ``*_stageii.npz`` (poses(T,165), trans(T,3), mocap_frame_rate)."""
    d = np.load(npz_path, allow_pickle=True)
    poses = np.asarray(d["poses"], dtype=np.float64)
    trans = np.asarray(d["trans"], dtype=np.float64)
    fps = (
        float(np.asarray(d["mocap_frame_rate"]).reshape(-1)[0])
        if "mocap_frame_rate" in d
        else 120.0
    )
    return WalkClip(poses=poses, trans=trans, fps=fps, src_path=npz_path)


def fk_joints(poses156: np.ndarray) -> np.ndarray:
    """Batched SMPL+H FK (betas=0) for (T,156) poses -> joints (T, NJ, 3), pelvis at origin."""
    import torch

    P = np.asarray(poses156, dtype=np.float32)
    with torch.no_grad():
        _, J = H.lbs(torch.zeros(len(P), 10), torch.tensor(P), *H.TPL, pose2rot=True)
    return J.detach().numpy().astype(np.float64)


def detect_up_R(clip: WalkClip, n_probe: int = 40) -> tuple[np.ndarray, str]:
    """Pick the up-conversion (I / Rx+90 / Rx-90) that makes the body upright (max mean head-foot_z)."""
    T = min(n_probe, len(clip.poses))
    body = clip.poses[:T, _AMASS_BODY[0] : _AMASS_BODY[1]]
    best_R, best_name, best_up = np.eye(3), "I", -1e9
    for name, up_R in _UP_CANDIDATES.items():
        P = np.zeros((T, 156), dtype=np.float32)
        for f in range(T):
            P[f, 0:3] = H.logmap(
                up_R @ _aa_to_R(clip.poses[f, _AMASS_ROOT[0] : _AMASS_ROOT[1]])
            )
        P[:, 3:66] = body
        J = fk_joints(P)
        up = float((J[:, H.HEAD_J, 2] - J[:, _FOOT_J, 2].min(axis=1)).mean())
        if up > best_up:
            best_up, best_R, best_name = up, up_R, name
    return best_R, best_name


def to_smplh_zup(
    clip: WalkClip, up_R: np.ndarray, extra_yaw_deg: float = 0.0
) -> np.ndarray:
    """Convert the clip's SMPL-X params to (T,156) SMPL+H poses in our Z-up world (hands zeroed).

    global' = Rz(extra_yaw) @ up_R @ R_amass_root ; body copied; hands = 0. ``extra_yaw_deg`` canonicalizes
    the walking heading to native -Y.
    """
    T = len(clip.poses)
    out = np.zeros((T, 156), dtype=np.float32)
    Rworld = _rz(extra_yaw_deg) @ np.asarray(up_R, dtype=np.float64)
    for f in range(T):
        out[f, 0:3] = H.logmap(
            Rworld @ _aa_to_R(clip.poses[f, _AMASS_ROOT[0] : _AMASS_ROOT[1]])
        )
        out[f, 3:66] = clip.poses[f, _AMASS_BODY[0] : _AMASS_BODY[1]]
    return out


def trans_zup(
    clip: WalkClip, up_R: np.ndarray, extra_yaw_deg: float = 0.0
) -> np.ndarray:
    """The pelvis translation rotated into our Z-up world (same rotation as ``to_smplh_zup``)."""
    Rworld = _rz(extra_yaw_deg) @ np.asarray(up_R, dtype=np.float64)
    return (Rworld @ np.asarray(clip.trans, dtype=np.float64).T).T


def foot_floor_z(J: np.ndarray) -> np.ndarray:
    """Per-frame lowest foot-joint z (T,) -> the ground-contact height to pin to the floor."""
    return J[:, _FOOT_J, 2].min(axis=1)


def pick_straight_window(
    clip: WalkClip, up_R: np.ndarray, min_seconds: float = 1.6, max_seconds: float = 3.5
) -> tuple[int, int]:
    """Longest ~straight forward-walking window (steady heading, real displacement) on the Z-up track."""
    tz = trans_zup(clip, up_R)[:, :2]
    T = len(tz)
    lo_n = max(4, int(min_seconds * clip.fps))
    hi_n = max(lo_n + 1, int(max_seconds * clip.fps))
    seg = max(1, int(clip.fps / 24.0))
    best = (-1.0, 0, min(T - 1, hi_n))
    for a in range(0, max(1, T - lo_n), seg):
        for wn in (lo_n, (lo_n + hi_n) // 2, hi_n):
            b = a + wn
            if b >= T:
                continue
            disp = float(np.linalg.norm(tz[b] - tz[a]))
            path = (
                float(np.sum(np.linalg.norm(np.diff(tz[a : b + 1], axis=0), axis=1)))
                + 1e-6
            )
            score = (disp / path) * disp  # straightness * displacement
            if disp > 0.5 and score > best[0]:
                best = (score, a, b)
    return best[1], best[2]


def canonical_yaw_deg(clip: WalkClip, lo: int, hi: int, up_R: np.ndarray) -> float:
    """Extra Z-yaw (deg) rotating the window's mean walking direction to native -Y ([0,-1,0])."""
    tz = trans_zup(clip, up_R)[:, :2]
    d = tz[hi] - tz[lo]
    if np.linalg.norm(d) < 1e-6:
        return 0.0
    cur = np.arctan2(d[1], d[0])
    tgt = np.arctan2(-1.0, 0.0)
    return float(np.rad2deg(tgt - cur))


# =====================================================================================================
# Clip selection: prefer a clean, continuous FORWARD walk.
# =====================================================================================================
_BAD_WORDS = (
    "turn",
    "side",
    "hop",
    "leap",
    "run",
    "back",
    "crouch",
    "skip",
    "pickup",
    "pick_up",
    "box",
    "stand",
    "neutral",
    "change",
)


def default_walk_clip() -> str:
    """Return the explicitly installed no-distribution ACCAD/AMASS walk clip."""
    if not _WALK_CLIP or not os.path.isfile(_WALK_CLIP):
        raise FileNotFoundError(
            "Set INTERACTIVEWORLD_WALK_CLIP to a licensed ACCAD/AMASS stageii NPZ"
        )
    return _WALK_CLIP


def _list_walk_npz() -> list[str]:
    """All ACCAD walk clips, ranked: plain 'walk' first, turn/side/hop/... last."""
    fs: list[str] = []
    for d in ("Female1Walking_c3d", "Male1Walking_c3d", "Male2Walking_c3d"):
        fs.extend(sorted(glob.glob(os.path.join(ACCAD_ROOT, d, "*.npz"))))

    def _rank(path: str) -> tuple[int, int, str]:
        n = os.path.basename(path).lower()
        has_walk = "walk" in n
        bad = sum(w in n for w in _BAD_WORDS)
        return (0 if ("walk1" in n) else 1, bad if has_walk else 99, n)

    return sorted(fs, key=_rank)


# =====================================================================================================
# OFFLINE self-test: prove AMASS -> Z-up upright striding, no Isaac. Saves a sagittal skeleton montage.
# =====================================================================================================
def _skeleton_montage(
    J_seq: np.ndarray, out_png: str, title: str, n_cols: int = 7
) -> bool:
    """Sagittal (walk-plane) stick-figure montage: plot (-Y, Z) so native forward (-Y) points right."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[accad] matplotlib unavailable ({exc}); skipping montage")
        return False

    parents = H.PARENTS
    n = len(J_seq)
    idxs = np.linspace(0, n - 1, min(n_cols, n)).astype(int)
    fig, axes = plt.subplots(
        1, len(idxs), figsize=(2.4 * len(idxs), 4.2), squeeze=False
    )
    ymin = float((-J_seq[:, :, 1]).min()) - 0.2
    ymax = float((-J_seq[:, :, 1]).max()) + 0.2
    zmax = float(J_seq[:, :, 2].max()) + 0.15
    for ax, fi in zip(axes[0], idxs):
        J = J_seq[fi]
        for j in range(len(parents)):
            p = int(parents[j])
            if p < 0:
                continue
            ax.plot(
                [-J[j, 1], -J[p, 1]], [J[j, 2], J[p, 2]], "-", color="#1f5fbf", lw=1.6
            )
        ax.scatter(-J[:, 1], J[:, 2], s=6, color="#d24a3a")
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xlim(ymin, ymax)
        ax.set_ylim(-0.1, zmax)
        ax.set_aspect("equal")
        ax.set_title(f"f{fi}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"[accad] montage -> {out_png}")
    return True


def _self_test() -> int:
    src = default_walk_clip()
    if not src:
        print(f"[accad] NO walk npz under {ACCAD_ROOT}")
        return 1
    clip = load_clip(src)
    print(
        f"[accad] clip={os.path.relpath(src, ACCAD_ROOT)} frames={len(clip.poses)} fps={clip.fps:.0f}"
    )

    up_R, up_name = detect_up_R(clip)
    print(f"[accad] up-conversion = {up_name} (auto)")

    lo, hi = pick_straight_window(clip, up_R)
    yaw = canonical_yaw_deg(clip, lo, hi, up_R)
    tz = trans_zup(clip, up_R)[:, :2]
    disp = float(np.linalg.norm(tz[hi] - tz[lo]))
    print(
        f"[accad] straight window=[{lo},{hi}] ({(hi - lo) / clip.fps:.2f}s) disp={disp:.2f}m canon_yaw={yaw:+.1f}deg"
    )

    p156 = to_smplh_zup(clip, up_R, extra_yaw_deg=yaw)
    win = p156[lo : hi + 1]
    J = fk_joints(win)
    fz = foot_floor_z(J)
    Jg = J.copy()
    Jg[:, :, 2] -= fz[
        :, None
    ]  # pin lowest foot to floor each frame (kills vertical float)

    head_z = float(Jg[:, H.HEAD_J, 2].mean())
    pelvis_z = float(Jg[:, 0, 2].mean())
    # Stride: fore-aft (Y, the walk axis after canonicalization) swing between the two ankles.
    stride = float(np.abs(Jg[:, 7, 1] - Jg[:, 8, 1]).max())
    print(
        f"[accad] GROUNDED window: pelvis_z~{pelvis_z:.2f} head_z~{head_z:.2f} "
        f"max fore-aft foot swing(stride)={stride:.2f}m"
    )
    ok_upright = head_z > 1.3 and pelvis_z > 0.75
    ok_stride = 0.2 < stride < 1.2
    print(
        f"[accad] CHECK upright={'PASS' if ok_upright else 'FAIL'} striding={'PASS' if ok_stride else 'FAIL'}"
    )

    out_png = os.path.abspath(
        os.path.join(_THIS_DIR, "..", "fpv_outputs", "accad_retarget_check.png")
    )
    _skeleton_montage(
        Jg, out_png, f"{os.path.basename(src)} [{lo}:{hi}] {up_name} yaw{yaw:+.0f}"
    )
    return 0 if (ok_upright and ok_stride) else 2


if __name__ == "__main__":
    raise SystemExit(_self_test())

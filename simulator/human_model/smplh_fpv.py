"""SMPL+H forward kinematics used by the InteractiveWorld human runtime.

No licensed model data is bundled. Configure the MANO and SMPL+H environment
variables before importing this module.
"""

import os
import pickle
import sys

MANO_SRC = os.environ.get("INTERACTIVEWORLD_MANO_PYTHON", "")
SMPLH_PKL = os.environ.get("INTERACTIVEWORLD_SMPLH_MODEL", "")


def _require_external_human_assets() -> None:
    missing = []
    if not MANO_SRC or not os.path.isdir(os.path.join(MANO_SRC, "mano")):
        missing.append("INTERACTIVEWORLD_MANO_PYTHON (directory containing mano/)")
    if not SMPLH_PKL or not os.path.isfile(SMPLH_PKL):
        missing.append("INTERACTIVEWORLD_SMPLH_MODEL (licensed SMPLH_male.pkl)")
    if missing:
        raise RuntimeError(
            "Required no-distribution human assets are not installed: "
            + "; ".join(missing)
        )


_require_external_human_assets()

sys.path.insert(0, MANO_SRC)
import numpy as np
import torch
from mano.lbs import batch_rigid_transform, batch_rodrigues, lbs, vertices2joints

# ---------------------------------------------------------------- SMPL+H joint layout (VERIFIED)
WRIST_J = 21
R_COLLAR, R_SHOULDER, R_ELBOW = 14, 17, 19
HEAD_J, NECK_J = 15, 12
# right-hand chain-roots off the wrist (MANO finger order index,middle,pinky,ring,thumb) + distal tips
MCP = {"index": 37, "middle": 40, "pinky": 43, "ring": 46, "thumb": 49}
TIPISH = {"index": 39, "middle": 42, "pinky": 45, "ring": 48, "thumb": 51}
FINGERS = ("index", "middle", "ring", "pinky")
RHAND_LO = 37 * 3  # 111: first index of the 45-D right-hand block (joints 37..51)


# ================================================================ load + forward (CPU)
def _arr(x):
    if "scipy.sparse" in str(type(x)):
        return np.asarray(x.todense())
    return np.asarray(x)


def load_smplh():
    d = pickle.load(open(SMPLH_PKL, "rb"))
    v_template = _arr(d["v_template"]).astype(np.float32)
    faces = _arr(d["f"]).astype(np.int64)
    shapedirs = _arr(d["shapedirs"]).astype(np.float32)[:, :, :10]
    posedirs = _arr(d["posedirs"]).astype(np.float32)
    posedirs = posedirs.reshape(-1, posedirs.shape[-1]).T
    Jreg = _arr(d["J_regressor"]).astype(np.float32)
    kintree = _arr(d["kintree_table"]).astype(np.int64)
    weights = _arr(d["weights"]).astype(np.float32)
    parents = kintree[0].copy()
    parents[0] = -1
    t = lambda x: torch.tensor(x)
    tpl = (t(v_template), t(shapedirs), t(posedirs), t(Jreg), t(parents), t(weights))
    return tpl, faces, parents, Jreg.shape[0]


TPL, FACES, PARENTS, NJ = load_smplh()


def fwd(pose_np):
    V, J = lbs(
        torch.zeros(1, 10), torch.tensor(pose_np.reshape(1, -1)), *TPL, pose2rot=True
    )
    return V[0].detach().numpy(), J[0].detach().numpy()


_J0_REST = (
    None  # betas=0 rest joints (J_regressor @ v_template); constant, computed once
)
_JOINT_FK_CACHE = {}
_LBS_TENSOR_CACHE = {}


def lbs_tensors(device: str | torch.device):
    """Return the immutable SMPL+H model tensors resident on ``device``."""
    target = torch.device(device)
    key = str(target)
    cached = _LBS_TENSOR_CACHE.get(key)
    if cached is None:
        cached = tuple(tensor.to(device=target) for tensor in TPL)
        _LBS_TENSOR_CACHE[key] = cached
    return cached


def fwd_joints(pose_np):
    """Fast joints-only FK (betas=0): returns the SAME joints as ``fwd(pose)[1]`` but skips the
    6890-vertex pose-blendshapes + skinning that the arm-reach / IK never uses. Because betas=0 makes
    the rest joints constant, only batch_rodrigues + batch_rigid_transform run -> ~50-100x cheaper per
    call. Use anywhere the reach/IK needs joints (not vertices)."""
    global _J0_REST
    if _J0_REST is None:
        _J0_REST = vertices2joints(
            TPL[3], TPL[0].unsqueeze(0)
        )  # (1, NJ, 3), betas=0 -> constant
    with torch.no_grad():
        rot = batch_rodrigues(torch.tensor(pose_np.reshape(-1, 3), dtype=torch.float32))
        posed, _ = batch_rigid_transform(rot.unsqueeze(0), _J0_REST, TPL[4])
    return posed[0].detach().numpy()


def _joint_fk_constants(device: torch.device, dtype: torch.dtype):
    """Return device-resident rest joints and parent-relative offsets for batched FK."""
    global _J0_REST
    if _J0_REST is None:
        _J0_REST = vertices2joints(TPL[3], TPL[0].unsqueeze(0))
    key = (str(device), dtype)
    cached = _JOINT_FK_CACHE.get(key)
    if cached is None:
        rest = _J0_REST.to(device=device, dtype=dtype)
        parent_ids = torch.as_tensor(PARENTS[1:], dtype=torch.long, device=device)
        rel = rest.clone()
        rel[:, 1:] -= rest[:, parent_ids]
        cached = rest, rel
        _JOINT_FK_CACHE[key] = cached
    return cached


def fwd_joints_from_rot_mats_batch(
    rot_mats: torch.Tensor, return_global_rot: bool = False
):
    """Batched joints-only SMPL+H FK without vertex skinning or host/device transfers.

    ``rot_mats`` is ``(B, NJ, 3, 3)`` and contains local joint rotations. The returned joints are
    ``(B, NJ, 3)`` on the same device. ``return_global_rot`` also returns the corresponding global
    rotations, which lets the batched reach solver update local rotations without recomputing parent
    transforms in NumPy.
    """
    if rot_mats.ndim != 4 or rot_mats.shape[1:] != (NJ, 3, 3):
        raise ValueError(
            f"expected rot_mats shape (B, {NJ}, 3, 3), got {tuple(rot_mats.shape)}"
        )
    batch = rot_mats.shape[0]
    rest, rel = _joint_fk_constants(rot_mats.device, rot_mats.dtype)
    positions = [rest[:, 0].expand(batch, -1)]
    rotations = [rot_mats[:, 0]]
    for joint in range(1, NJ):
        parent = int(PARENTS[joint])
        offset = rel[:, joint].expand(batch, -1)
        positions.append(
            positions[parent]
            + torch.matmul(rotations[parent], offset.unsqueeze(-1)).squeeze(-1)
        )
        rotations.append(torch.matmul(rotations[parent], rot_mats[:, joint]))
    joints = torch.stack(positions, dim=1)
    if return_global_rot:
        return joints, torch.stack(rotations, dim=1)
    return joints


def fwd_selected_joints_from_rot_mats_batch(
    rot_mats: torch.Tensor, joint_ids
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """FK only the requested joints and their ancestors, retaining device-resident transforms."""
    required = set()
    for joint in joint_ids:
        current = int(joint)
        while current >= 0 and current not in required:
            required.add(current)
            current = int(PARENTS[current])
    order = [joint for joint in range(NJ) if joint in required]
    batch = rot_mats.shape[0]
    rest, rel = _joint_fk_constants(rot_mats.device, rot_mats.dtype)
    positions = {}
    rotations = {}
    for joint in order:
        parent = int(PARENTS[joint])
        if parent < 0:
            positions[joint] = rest[:, joint].expand(batch, -1)
            rotations[joint] = rot_mats[:, joint]
            continue
        offset = rel[:, joint].expand(batch, -1)
        positions[joint] = positions[parent] + torch.matmul(
            rotations[parent], offset.unsqueeze(-1)
        ).squeeze(-1)
        rotations[joint] = torch.matmul(rotations[parent], rot_mats[:, joint])
    return positions, rotations


def fwd_joints_batch(
    pose, device: str | torch.device | None = None, return_global_rot: bool = False
):
    """Convert one or more 156-D axis-angle poses to joints with batch-first Torch FK."""
    if isinstance(pose, torch.Tensor):
        poses = (
            pose.to(device=device, dtype=torch.float32)
            if device is not None
            else pose.to(dtype=torch.float32)
        )
    else:
        poses = torch.as_tensor(pose, dtype=torch.float32, device=device)
    if poses.ndim == 1:
        poses = poses.unsqueeze(0)
    if poses.ndim != 2 or poses.shape[1] != NJ * 3:
        raise ValueError(f"expected pose shape (B, {NJ * 3}), got {tuple(poses.shape)}")
    rot_mats = batch_rodrigues(poses.reshape(-1, 3)).reshape(-1, NJ, 3, 3)
    return fwd_joints_from_rot_mats_batch(rot_mats, return_global_rot=return_global_rot)


def global_rot(pose_np, j):
    """Global rotation of joint j = product of local axis-angle rot-mats up the parent chain (rest
    relative rotations are identity, so translations are irrelevant to the orientation)."""
    aa = pose_np.reshape(-1, 3)
    R = batch_rodrigues(torch.tensor(aa)).numpy()
    chain = []
    k = j
    while k != -1:
        chain.append(k)
        k = int(PARENTS[k])
    M = np.eye(3)
    for k in reversed(chain):
        M = M @ R[k]
    return M


# ================================================================ wrist frame + small rotation utils
def wrist_frame(J):
    """Rebuild the MANO wrist frame from the posed SMPL+H finger joints -- IDENTICAL construction to
    mano_grasp_general's dump (pd=wrist->mean MCP, ac=index-pinky MCP, pn=pd x ac, orthonormalized to
    ex=ac, ey=pd_perp, ez).  Because both the grasp and SMPL+H produce the same finger geometry for the
    same pose, this reproduces the exact frame the dump used, so obj_T_in_wrist reseats the object 1:1.
    Returns (R_w[3x3] world<-wrist, origin = wrist joint)."""
    wrist = J[WRIST_J]
    palm_ctr = np.array([J[MCP[f]] for f in FINGERS]).mean(0)
    pd = palm_ctr - wrist
    pd /= np.linalg.norm(pd)
    ac = J[MCP["index"]] - J[MCP["pinky"]]
    ac /= np.linalg.norm(ac)
    pn = np.cross(pd, ac)
    pn /= np.linalg.norm(pn)
    ftip = np.array([J[TIPISH[f]] for f in FINGERS]).mean(0)
    if np.dot(ftip - palm_ctr, pn) < 0:
        pn = -pn
    ex = ac
    ey = pd - np.dot(pd, ex) * ex
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    if np.dot(ez, pn) < 0:
        ex = -ex
        ez = np.cross(ex, ey)
    R_w = np.column_stack([ex, ey, ez])
    return R_w, wrist.copy()


def rot_align(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else (-np.eye(3) + 2 * np.outer(b, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def axis_rot(axis, deg):
    """Rotation matrix about an arbitrary unit axis by `deg` degrees."""
    axis = axis / np.linalg.norm(axis)
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def logmap(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-8:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2 * np.sin(ang)
    )
    return ax * ang


# Reusable arm constants and inverse-kinematics solver.
L_SHOULDER, L_ELBOW = 16, 18
COLLAR = [0.0, 0.15, 0.20]
L_ARM = {L_SHOULDER: [0.60, -0.20, -1.40], L_ELBOW: [0.0, 0.40, 0.0]}


def _ik_arm(target, n=8000, seed=1, horizontal=False, arm_dir=None):
    rng = np.random.default_rng(seed)
    sh = rng.uniform([0.5, -0.4, -0.7], [2.1, 1.3, 0.5], size=(n, 3))
    el = rng.uniform([-0.4, -2.5, -0.9], [1.0, -0.5, 0.4], size=(n, 3))
    pb = np.zeros((n, NJ * 3), dtype=np.float32)
    pb[:, 0:3] = [np.pi / 2.0, 0.0, 0.0]
    for j, v in {R_COLLAR: COLLAR, **L_ARM}.items():
        pb[:, j * 3 : j * 3 + 3] = v
    pb[:, R_SHOULDER * 3 : R_SHOULDER * 3 + 3] = sh
    pb[:, R_ELBOW * 3 : R_ELBOW * 3 + 3] = el
    _, Jb = lbs(torch.zeros(n, 10), torch.tensor(pb), *TPL, pose2rot=True)
    Jb = Jb.detach().numpy()
    w, e = Jb[:, WRIST_J], Jb[:, R_ELBOW]
    err = np.linalg.norm(w - target, axis=1)
    if horizontal:
        # HORIZONTAL forearm along the grasp's arm_dir_world: the wrist->elbow segment must run PARALLEL to
        # arm_dir (which is horizontal +X = from the person's right), so the hand reaches the object from the
        # SIDE and the object stands BESIDE the wrist, never under it.  Penalize the forearm's component
        # PERPENDICULAR to arm_dir, and reward a POSITIVE projection onto arm_dir (elbow on the +arm_dir side
        # of the wrist) so the forearm points the right way (not folded back).
        d = arm_dir / (np.linalg.norm(arm_dir) + 1e-9)
        fa = e - w  # forearm vector wrist->elbow
        proj = fa @ d
        perp = fa - np.outer(proj, d)
        pen = (
            6.0 * np.linalg.norm(perp, axis=1)
            - 1.0 * proj
            + np.maximum(0, e[:, 2] - w[:, 2]) * 0.3
        )
    else:
        pen = (
            np.maximum(0, e[:, 2] - w[:, 2]) * 0.5 + np.maximum(0, e[:, 0] - 0.05) * 0.6
        )
    r = int(np.argmin((0.4 if horizontal else 1.0) * err + pen))
    return list(sh[r]), list(el[r]), float(err[r])


def vertex_normals(verts, faces):
    """Smooth per-VERTEX normals (area-weighted) so the body is not faceted."""
    n = np.zeros_like(verts)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    return n / ln


__all__ = [
    "COLLAR",
    "FACES",
    "FINGERS",
    "HEAD_J",
    "L_ARM",
    "MCP",
    "NJ",
    "PARENTS",
    "RHAND_LO",
    "R_COLLAR",
    "R_ELBOW",
    "R_SHOULDER",
    "TIPISH",
    "TPL",
    "WRIST_J",
    "_ik_arm",
    "axis_rot",
    "batch_rigid_transform",
    "batch_rodrigues",
    "fwd",
    "fwd_joints",
    "fwd_joints_batch",
    "fwd_joints_from_rot_mats_batch",
    "fwd_selected_joints_from_rot_mats_batch",
    "global_rot",
    "lbs",
    "lbs_tensors",
    "logmap",
    "rot_align",
    "vertex_normals",
    "wrist_frame",
]

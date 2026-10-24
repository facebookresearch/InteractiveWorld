# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Rigidly head-mounted egocentric camera -- IsaacLab's ``CameraCfg`` contract.

IsaacLab mounts a camera as a USD *child* of a rigid body prim with a fixed
``OffsetCfg(pos, rot)`` and fixed ``PinholeCameraCfg`` intrinsics, so the sensor pose is
``body_pose @ offset`` on every frame and nothing else is allowed to move it
(``IsaacLab/scripts/tutorials/04_sensors/add_sensors_on_robot.py:72-82``; unitree's
``d435_link`` mount is the same shape with a zero offset,
``unitree_sim_isaaclab/tasks/common_config/camera_configs.py:57-74``).

Our wearer has no USD parent to attach to -- the SMPL+H body is a point-baked ``/World/Body``
mesh, not an articulation -- so the head link's pose is reconstructed from the pose vector and
the same product is evaluated per frame. Pure numpy on purpose (no ``pxr``, no Isaac, no
scene) so both renderers can share it and it stays testable offline.
"""

from __future__ import annotations

import math

import numpy as np

HEAD_JOINT = 15
# SMPL+H chain from the root to the head, root-first:
# pelvis(0) -> spine1(3) -> spine2(6) -> spine3(9) -> neck(12) -> head(15).
HEAD_CHAIN: tuple[int, ...] = (0, 3, 6, 9, 12, 15)

# Head-LOCAL axes. At rest every local rotation is identity, so the head frame IS the raw SMPL
# template frame: Y-up with the face along +Z. The pipeline reaches world Z-up by baking
# Rx(+90 deg) into the root axis-angle (``smplh_anim_core.py:309``), which is why a mount
# offset written in WORLD axes stops being correct the moment the head pitches.
FACE_FORWARD_LOCAL = np.array([0.0, 0.0, 1.0])
HEAD_UP_LOCAL = np.array([0.0, 1.0, 0.0])
WORLD_UP = np.array([0.0, 0.0, 1.0])

# Mount point, head-local metres: eye height, pushed out past the face. Measured against the
# SMPL+H template mesh (betas=0), which puts the anatomical eye midpoint (template vertices
# 2800/6260) at up 0.091 / fwd 0.065 and the nose tip (vertex 332) at up 0.057 / fwd 0.108.
#
# The eyeball itself cannot hold a camera -- it is 10 mm INSIDE the skin, because the centreline
# face surface is at fwd 0.098. Real glasses sit on the frame front a couple of cm ahead of the
# cornea, which is what this is: eye height, 15.6 mm clear of the skin. The predecessor constant
# (``smplh_camera.EYE_OFFSET``, up 0.05 / fwd 0.12) was 4 cm low -- nose-tip height, not eye.
EYE_LOCAL = np.array([0.0, 0.09, 0.11])
SKIN_CLEARANCE_M = 0.0156

# Near clip has to exceed the skin clearance so the wearer's own face can never be drawn, and
# stay well below the ~0.24 m at which a held object is inspected.
NEAR_CLIP_M = 0.02
FAR_CLIP_M = 100.0
FOCAL_MM = 16.0

# Rest geometry of the root->head chain, baked from the SMPL+H template at betas=0 (the shape the
# whole pipeline uses; ``smplh_fpv._J0_REST`` documents it as constant there). Lets the head link
# be placed without the full 6890-vertex FK -- verified against ``smplh_fpv.fwd`` to 1.7e-7 m over
# 200 random poses.
PELVIS_REST = np.array([-0.00217, -0.240775, 0.028592])
CHAIN_BONES: tuple[tuple[int, int, tuple[float, float, float]], ...] = (
    (0, 3, (0.004439, 0.124404, -0.038385)),
    (3, 6, (0.004488, 0.137956, 0.026820)),
    (6, 9, (-0.002265, 0.056032, 0.002855)),
    (9, 12, (-0.013390, 0.211636, -0.033468)),
    (12, 15, (0.010117, 0.088949, 0.050407)),
)
SPINE3_JOINT = 9
NECK_BONE = np.array(CHAIN_BONES[3][2])
HEAD_BONE = np.array(CHAIN_BONES[4][2])


def _walk_chain(pose: np.ndarray, stop_after: int) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate (global rotation, position) down the chain, stopping at ``stop_after``."""
    rotation = rodrigues(pose[HEAD_CHAIN[0]])
    position = PELVIS_REST.copy()
    for parent, child, bone in CHAIN_BONES:
        if parent == stop_after:
            break
        position = position + rotation @ np.asarray(bone)
        rotation = rotation @ rodrigues(pose[child])
    return rotation, position


def _place(
    rotation: np.ndarray,
    position: np.ndarray,
    yaw_deg: float,
    root_xy,
    root_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the world placement ``pose_place`` performs after the FK."""
    yaw = yaw_matrix(yaw_deg)
    offset = np.array([float(root_xy[0]), float(root_xy[1]), float(root_z)])
    return yaw @ rotation, yaw @ position + offset


def spine3_frame(
    pose156: np.ndarray, yaw_deg: float, root_xy, root_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """World (rotation, position) of spine3 -- the neck's parent, unaffected by neck/head."""
    pose = np.asarray(pose156, dtype=np.float64).reshape(-1, 3)
    rotation, position = _walk_chain(pose, stop_after=SPINE3_JOINT)
    return _place(rotation, position, yaw_deg, root_xy, root_z)


def head_pose(
    pose156: np.ndarray, yaw_deg: float, root_xy, root_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """World (rotation, position) of the head link, without running the full mesh FK."""
    pose = np.asarray(pose156, dtype=np.float64).reshape(-1, 3)
    rotation, position = _walk_chain(pose, stop_after=-1)
    return _place(rotation, position, yaw_deg, root_xy, root_z)


# Below this |sin(angle)| an up-reference is too close to the view axis to define roll: the cross
# product is dominated by noise and the horizon snaps. Guard against NEAR-degeneracy, not just
# exact -- an exact-zero test lets a 1-degree separation through and it flips frame to frame.
_MIN_UP_SEPARATION = 0.2


def basis_from_forward(forward: np.ndarray, up_reference: np.ndarray) -> np.ndarray:
    """Head rotation whose face-forward is ``forward`` and whose roll follows ``up_reference``."""
    axis_z = _unit(np.asarray(forward, dtype=np.float64))
    for candidate in (np.asarray(up_reference, dtype=np.float64), WORLD_UP):
        side = np.cross(candidate, axis_z)
        if float(np.linalg.norm(side)) >= _MIN_UP_SEPARATION:
            axis_x = _unit(side)
            break
    else:
        # Looking near-straight up or down: roll is genuinely arbitrary, so pick a stable axis
        # rather than let numerical noise choose one.
        axis_x = _unit(np.cross(np.array([0.0, 1.0, 0.0]), axis_z), fallback=WORLD_UP)
    return np.column_stack([axis_x, np.cross(axis_z, axis_x), axis_z])


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Rotation matrix for one axis-angle triple."""
    vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = vector / theta
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * (cross @ cross)
    )


def yaw_matrix(yaw_deg: float) -> np.ndarray:
    """World Rz applied by ``gait_player.pose_place`` after the FK."""
    angle = math.radians(float(yaw_deg))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])


def head_rotation(pose156: np.ndarray, yaw_deg: float) -> np.ndarray:
    """World rotation of the head link for one motion frame.

    ``pose_place`` FKs the pose and only then applies Rz(yaw) to the result
    (``gait_player.py:172-178``), so the world head rotation is that Rz composed with the
    product of the local rotations down the root->head chain.
    """
    pose = np.asarray(pose156, dtype=np.float64).reshape(-1, 3)
    rotation = np.eye(3)
    for joint in HEAD_CHAIN:
        rotation = rotation @ rodrigues(pose[joint])
    return yaw_matrix(yaw_deg) @ rotation


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        if fallback is None:
            raise ValueError("cannot normalise a zero-length vector")
        return _unit(fallback)
    return vector / length


def mount(
    pose156: np.ndarray,
    yaw_deg: float,
    head_world: np.ndarray,
    *,
    level_horizon: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(eye, forward, up)`` in world metres for the head-mounted camera.

    ``level_horizon`` drops the wearer's head roll and re-derives up from world +Z. That is a
    deliberate departure from the physical mount; it exists only so the roll can be A/B'd
    separately from the mount itself.
    """
    rotation = head_rotation(pose156, yaw_deg)
    eye = np.asarray(head_world, dtype=np.float64).reshape(3) + rotation @ EYE_LOCAL
    forward = _unit(rotation @ FACE_FORWARD_LOCAL)
    head_up = _unit(rotation @ HEAD_UP_LOCAL)
    if not level_horizon:
        return eye, forward, head_up
    # Gram-Schmidt world +Z against the view axis; fall back to the true head up when the
    # wearer is looking near-straight up or down and world +Z degenerates.
    levelled = WORLD_UP - float(np.dot(WORLD_UP, forward)) * forward
    return eye, forward, _unit(levelled, fallback=head_up)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vestibulo-ocular gaze stabilisation, authored into the neck instead of into the camera.

Neither IsaacLab nor unitree stabilises a camera -- ``Camera._update_poses`` just reads the
prim's world pose, and a library-wide search for smoothing/filtering on the camera turns up
nothing. Their images look steady because the *body* the sensor is bolted to is held steady by
the locomotion controller. A retargeted mocap wearer has no such controller, and the clip
carries the raw pelvis sway and head bob of the walk cycle.

Real people do have one: the vestibulo-ocular and cervicocollic reflexes counter-rotate the
neck so the gaze direction stays put while the torso bobs. Without it a head-mounted camera
rocks about 11 deg/s during a walk, which is far shakier than any real egocentric recording.

So this restores the missing reflex in the producer, where it physically belongs: the neck
really turns, the rendered body shows it, and the camera stays a dumb rider. The alternative --
smoothing the camera -- would put the lens back somewhere the wearer's head is not.

Pure numpy, no ``pxr``, no Isaac, no scene.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, NamedTuple

import head_mount
import numpy as np

NECK_JOINT = 12

# The measured gait bob is 1.3 Hz. A Gaussian only rolls off as exp(-(2*pi*f*sigma/fps)^2/2), so
# a cutoff needs real separation from it: at 0.8 Hz a quarter of the bob still gets through
# (measured: 16.9% residual), at 0.4 Hz it is 1.8%, and below that the curve flattens while the
# neck has to work harder. Filtering this low is safe because ``frame.yaw`` is NOT filtered -- a
# deliberate 50 deg course change still comes through at 51 deg.
CUTOFF_HZ = 0.4
# Where the wearer is taken to be looking. Room scale; gaze fixation is insensitive to the exact
# value because what it cancels is the *change* in bearing to the point, not the point itself.
FIXATION_DISTANCE_M = 2.5
# Turning the neck moves the eye the fixation is measured from, so the solve is iterated.
# Measured convergence of the residual shake: 0.783 (2 passes) -> 0.776 (3) -> 0.7764 (4) ->
# fixed point. Three is within 0.0004 deg/frame of converged.
FIXATION_ITERATIONS = 3
# Below the 1.3 Hz bob, above the slow drift of walking onto a ramp or a step.
BOB_CUTOFF_HZ = 0.5
# 58-63 mm measured -> ~45 mm, the middle of the human range at this walking speed.
BOB_GAIN = 0.78
# HOW FAST THE VIEW MAY SWING, degrees per frame, measured on the world-space aim direction.
#
# This is the backstop, and it is the one guard that does not care WHY the aim moved. The body
# already has one (``heading_controller.HeadingCfg.max_turn_deg``, 10 deg/frame); the neck had
# none at all, so whatever the fixation point did in one frame the head did too. Measured on a
# recorded walk: the walk->look seam swung the camera 30.5 deg horizontally in a single frame
# (366 deg/s) and a look->walk seam swung it 35.1 deg vertically (421 deg/s), against a median of
# 2.8 deg for an ordinary walking frame.
#
# 15 deg/frame at 12 fps is 180 deg/s. Higher than the body's cap because a neck genuinely does
# outrun a torso, and comfortably under the ~300 deg/s a person can manage.
#
# NOTE this caps the TOTAL swing, body rotation included, because that is what the viewer sees.
# When the body is already turning at its own cap the neck simply gets the remainder.
MAX_GAZE_DEG_PER_FRAME = 15.0
# How early the gaze arrives on a target the producer has authored, and how late it leaves.
#
# Extending backwards until the target sits more than 40 deg off the walking direction does not
# work: that test is evaluated FIRST on the frame immediately before the look -- the one frame
# where the wearer is closest to the target and the angle is therefore at its worst -- and a
# single failure there sets the lead to zero. So the mechanism that exists to prevent the
# whip-round switches itself off in exactly the geometry that produces one.
#
# The length is instead derived from the thing that actually matters: how many frames the rate cap
# needs to cover the angle, with margin for the smoothstep's peak slope. Short swings get a short
# lead, big ones get a longer one, and nothing gets a 4 s stare at a wall. Whether the head can
# physically hold the target is left to ``clamp_head_on_torso``, which is what it is for.
GAZE_EASE_MIN_FRAMES = 3
GAZE_EASE_MAX_SECONDS = 1.5
# A step this large in the EYE track is a teleport, not motion; the filters restart across it.
# It must be measured on the eye and never on the fixation point: a point 2.5 m away moves
# 2.5*dtheta, so an ordinary 15 deg/frame turn shifts it 0.65 m and would be misread as a cut,
# shattering the filter at every corner (observed as 38 deg roll spikes).
CUT_TELEPORT_M = 0.5
# The cervical spine contributes most of the correction; the atlanto-occipital joint the rest.
NECK_SHARE = 0.6
# Comfortable sustained head-on-torso range, not the anatomical maximum.
MAX_YAW_DEG = 70.0
MAX_PITCH_DOWN_DEG = 55.0
MAX_PITCH_UP_DEG = 45.0
MAX_ROLL_DEG = 35.0
# An inter-frame step this large is a cut, not motion; the filter restarts across it.
CUT_ANGLE_DEG = 45.0


def log_rotation(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (the inverse of ``head_mount.rodrigues``)."""
    cos_theta = min(1.0, max(-1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    if math.pi - theta < 1e-6:
        # Near pi the skew part vanishes; recover the axis from the symmetric part.
        symmetric = (rotation + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] > 1e-9:
            axis = symmetric[:, largest] / axis[largest]
        return _unit_or_zero(axis) * theta
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(theta))
    return axis * theta


def _unit_or_zero(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros(3)


def _to_quaternion(rotation: np.ndarray) -> np.ndarray:
    vector = log_rotation(rotation)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = vector / theta
    return np.concatenate([[math.cos(theta / 2.0)], math.sin(theta / 2.0) * axis])


def _from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    angle = 2.0 * math.acos(min(1.0, max(-1.0, float(quaternion[0]))))
    axis = _unit_or_zero(quaternion[1:])
    return head_mount.rodrigues(axis * angle)


def gaussian_weights(sigma_frames: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma_frames)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / max(sigma_frames, 1e-6)) ** 2)
    return weights / weights.sum()


def _segment_bounds(quaternions: np.ndarray, cut_cos: float) -> list[tuple[int, int]]:
    """Split at cuts so the filter never smooths a discontinuity into its neighbours."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(quaternions)):
        if abs(float(np.dot(quaternions[index], quaternions[index - 1]))) < cut_cos:
            bounds.append((start, index))
            start = index
    bounds.append((start, len(quaternions)))
    return bounds


def smooth_rotations(
    rotations: list[np.ndarray],
    sigma_frames: float,
    bounds: list[int] | None = None,
) -> list[np.ndarray]:
    """Zero-phase Gaussian smoothing of a rotation sequence, restarted across cuts.

    Zero-phase (symmetric, non-causal) is available because the whole trajectory is authored
    before anything renders it, and it matters: a causal filter would make the head lag every
    deliberate turn.
    """
    if not rotations:
        return []
    quaternions = np.stack([_to_quaternion(rotation) for rotation in rotations])
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index], quaternions[index - 1])) < 0.0:
            quaternions[index] = -quaternions[index]
    weights = gaussian_weights(sigma_frames)
    radius = len(weights) // 2
    cut_cos = math.cos(math.radians(CUT_ANGLE_DEG) / 2.0)
    edges = (
        list(zip(bounds, bounds[1:]))
        if bounds is not None
        else _segment_bounds(quaternions, cut_cos)
    )
    smoothed = np.empty_like(quaternions)
    for lo, hi in edges:
        block = quaternions[lo:hi]
        padded = np.pad(block, ((radius, radius), (0, 0)), mode="edge")
        for index in range(len(block)):
            window = padded[index : index + len(weights)]
            smoothed[lo + index] = weights @ window
    return [_from_quaternion(quaternion) for quaternion in smoothed]


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    vector = np.zeros(3)
    vector[axis] = angle
    return head_mount.rodrigues(vector)


def clamp_head_on_torso(
    rotation: np.ndarray,
    *,
    max_yaw_deg: float = MAX_YAW_DEG,
    max_pitch_down_deg: float = MAX_PITCH_DOWN_DEG,
    max_pitch_up_deg: float = MAX_PITCH_UP_DEG,
    max_roll_deg: float = MAX_ROLL_DEG,
) -> tuple[np.ndarray, bool]:
    """Clamp a head-relative-to-torso rotation into the comfortable cervical range.

    Decomposed as Ry(yaw) Rx(pitch) Rz(roll) in head-local axes (+Y up, +Z face-forward), so
    positive pitch is looking down. A safety net rather than a shaper: VOR corrections are a
    few degrees, so this should almost never bind.
    """
    pitch = math.asin(min(1.0, max(-1.0, -float(rotation[1, 2]))))
    roll = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
    limited_pitch = min(
        math.radians(max_pitch_down_deg), max(-math.radians(max_pitch_up_deg), pitch)
    )
    limited_yaw = min(math.radians(max_yaw_deg), max(-math.radians(max_yaw_deg), yaw))
    limited_roll = min(
        math.radians(max_roll_deg), max(-math.radians(max_roll_deg), roll)
    )
    if (
        abs(limited_pitch - pitch) < 1e-12
        and abs(limited_yaw - yaw) < 1e-12
        and abs(limited_roll - roll) < 1e-12
    ):
        return rotation, False
    limited = (
        _axis_rotation(1, limited_yaw)
        @ _axis_rotation(0, limited_pitch)
        @ _axis_rotation(2, limited_roll)
    )
    return limited, True


def _smooth_array(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    weights = gaussian_weights(sigma_frames)
    radius = len(weights) // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [weights @ padded[index : index + len(weights)] for index in range(len(values))]
    )


def teleport_bounds(
    track: np.ndarray, threshold_m: float = CUT_TELEPORT_M
) -> list[int]:
    """Split indices where a world-space track jumps further than a body plausibly can."""
    track = np.asarray(track, dtype=np.float64)
    if len(track) == 0:
        return [0]
    jumps = np.linalg.norm(np.diff(track, axis=0), axis=1)
    return [0, *(int(i) + 1 for i in np.where(jumps > threshold_m)[0]), len(track)]


def smooth_points(
    points: np.ndarray, sigma_frames: float, bounds: list[int] | None = None
) -> np.ndarray:
    """Zero-phase Gaussian smoothing of a world-space point track.

    ``bounds`` must come from a track that only jumps on a real cut -- see ``CUT_TELEPORT_M``.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points
    edges = bounds if bounds is not None else teleport_bounds(points)
    smoothed = np.empty_like(points)
    for lo, hi in zip(edges, edges[1:]):
        smoothed[lo:hi] = _smooth_array(points[lo:hi], sigma_frames)
    return smoothed


class BobDampingResult(NamedTuple):
    frames: list[Any]
    peak_to_peak_before_mm: float
    peak_to_peak_after_mm: float
    max_root_shift_mm: float


def damp_vertical_bob(
    frames: list[Any],
    *,
    fps: float = 12.0,
    cutoff_hz: float = BOB_CUTOFF_HZ,
    gain: float = BOB_GAIN,
) -> BobDampingResult:
    """Shrink the gait's vertical excursion toward the human range.

    ``gait_player.plan_walk`` pins the root so the lowest foot sits on the floor every frame, so
    the clip's full vertical travel is transmitted to the head. It measures ~58-63 mm peak-to-peak
    here against 40-50 mm for a real walk at this speed, and it cannot be removed by rotating
    anything -- only the legs can flatten a CoM trajectory.

    Scaling the oscillation about its own slow baseline is the cheap approximation: the root
    deviates from exact foot contact by at most ``(1-gain)/2`` of the amplitude, a few mm, well
    inside the 30 mm the collision nudge already treats as a graze.
    """
    if not frames:
        return BobDampingResult([], 0.0, 0.0, 0.0)
    if not 0.0 < gain <= 1.0:
        raise ValueError("gain must be in (0, 1]")
    heights = np.array([float(frame.z) for frame in frames])
    baseline = smooth_points(heights[:, None], fps / (2.0 * math.pi * cutoff_hz))[:, 0]
    damped = baseline + (heights - baseline) * gain
    return BobDampingResult(
        [
            dataclasses.replace(frame, z=float(height))
            for frame, height in zip(frames, damped)
        ],
        float((heights - baseline).ptp() * 1000.0),
        float((damped - baseline).ptp() * 1000.0),
        float(np.abs(damped - heights).max() * 1000.0),
    )


def _smoothstep(fraction: float) -> float:
    value = min(1.0, max(0.0, fraction))
    return value * value * (3.0 - 2.0 * value)


def _run_slices(flags: list[bool]) -> list[tuple[int, int]]:
    """Contiguous ``[lo, hi)`` runs where ``flags`` is True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _ease_frames(
    swing_deg: float,
    max_frames: int,
    rate_deg_per_frame: float = MAX_GAZE_DEG_PER_FRAME,
) -> int:
    """How many frames an ease of ``swing_deg`` needs to stay inside the rate cap.

    The 2x is the smoothstep's peak slope (1.5x the mean) plus margin, so the eased ramp does not
    itself trip the rate limiter and get flattened into a lag.
    """
    if rate_deg_per_frame <= 0.0:
        return max_frames
    need = int(math.ceil(2.0 * max(swing_deg, 0.0) / rate_deg_per_frame))
    return max(GAZE_EASE_MIN_FRAMES, min(max_frames, need))


def apply_gaze_intent(
    frames: list[Any],
    fixations: np.ndarray,
    eyes: np.ndarray,
    sigma_frames: float,
    bounds: list[int],
    *,
    max_lead_frames: int,
) -> tuple[np.ndarray, int]:
    """Aim at whatever a frame says is being worked on, easing BOTH into and out of the beat.

    ``Frame.gaze`` is the producer-agnostic "what the wearer is working on" channel: attach,
    retract, place and look all write it, so nothing here is keyed to a verb, an object type or
    a scene.

    Two failure modes this avoids, both measured on a recorded walk:

    * A lead that stops at the first frame where the target sits more than 40 deg off the
      walking direction, scanning BACKWARDS from the look, tests the frame where the wearer
      stands closest to the target and the angle is worst, so one failure there disables the
      lead entirely. The entry then swung the camera 30.5 deg in a single frame.
    * Without an ease-OUT (writing only indices below ``lo``), every beat ends by snapping the
      fixation back to the default forward point. Measured at 24 deg and 35 deg of vertical swing
      in one frame.

    Both eases are sized by the swing they have to cover, so a small change gets a short ease
    and a big one gets a longer one. Whether the head can physically hold the target is left to
    ``clamp_head_on_torso``, which already clamps and counts it.
    """
    targets = np.array(fixations, dtype=np.float64, copy=True)
    authored = [getattr(frame, "gaze", None) is not None for frame in frames]
    if not any(authored):
        return targets, 0
    cuts = set(bounds[1:-1])
    total = len(frames)
    for lo, hi in _run_slices(authored):
        run = np.stack(
            [np.asarray(frames[i].gaze, dtype=np.float64) for i in range(lo, hi)]
        )
        # A hand-held or carried target jitters with the hand; steady it the same way.
        targets[lo:hi] = _smooth_array(run, sigma_frames) if hi - lo > 1 else run

        # -- ease IN: how far back may we go, and how far back do we need to go?
        room = 0
        for step in range(1, max_lead_frames + 1):
            index = lo - step
            if index < 0 or authored[index] or (index + 1) in cuts:
                break
            room = step
        if room:
            swing = math.degrees(
                _angle_between(
                    targets[lo] - eyes[lo - 1], fixations[lo - 1] - eyes[lo - 1]
                )
            )
            reach = min(room, _ease_frames(swing, max_lead_frames))
            for step in range(1, reach + 1):
                index = lo - step
                blend = _smoothstep(1.0 - step / (reach + 1.0))
                targets[index] = (1.0 - blend) * fixations[index] + blend * targets[lo]

        # -- ease OUT, the mirror image. Without it the beat ends on a hard cut back to "ahead".
        room = 0
        for step in range(1, max_lead_frames + 1):
            index = hi - 1 + step
            if index >= total or authored[index] or index in cuts:
                break
            room = step
        if room:
            swing = math.degrees(
                _angle_between(targets[hi - 1] - eyes[hi], fixations[hi] - eyes[hi])
            )
            reach = min(room, _ease_frames(swing, max_lead_frames))
            for step in range(1, reach + 1):
                index = hi - 1 + step
                blend = _smoothstep(1.0 - step / (reach + 1.0))
                targets[index] = (1.0 - blend) * fixations[index] + blend * targets[
                    hi - 1
                ]
    return targets, sum(authored)


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    left, right = _unit_or_zero(first), _unit_or_zero(second)
    if not np.any(left) or not np.any(right):
        return 0.0
    return math.acos(min(1.0, max(-1.0, float(np.dot(left, right)))))


def _slerp_unit(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Rotate ``first`` ``fraction`` of the way toward ``second`` along the great circle.

    Falls back to normalised lerp when the two are near-parallel or near-opposite, where the
    great circle is either unnecessary or undefined.
    """
    left, right = _unit_or_zero(first), _unit_or_zero(second)
    dot = min(1.0, max(-1.0, float(np.dot(left, right))))
    theta = math.acos(dot)
    if theta < 1e-6 or theta > math.pi - 1e-6:
        blended = _unit_or_zero(left + (right - left) * fraction)
        return blended if np.any(blended) else right
    sin_theta = math.sin(theta)
    return left * (math.sin((1.0 - fraction) * theta) / sin_theta) + right * (
        math.sin(fraction * theta) / sin_theta
    )


class StabilizationResult(NamedTuple):
    frames: list[Any]
    clamped_frames: int
    mean_correction_deg: float
    max_correction_deg: float
    gaze_frames: int = 0
    # Frames where the aim wanted to move faster than a neck can. Should be RARE: the eases are
    # meant to absorb the swings, and a run that trips this often is reporting that some producer
    # is authoring a gaze the eases cannot reach, not that the cap is wrong.
    rate_limited_frames: int = 0


def eye_looking_at(
    pose156, yaw: float, xy, z: float, target, *, neck_share: float = NECK_SHARE
) -> tuple[np.ndarray, np.ndarray]:
    """Where the EYE ends up once the wearer turns to look at ``target``. Returns (eye, rotation).

    WHY A PRODUCER NEEDS THIS. Looking at something MOVES the thing you look with: the eye sits
    0.11 m in front of the head joint and 0.09 m above it, so pitching the head down 40 deg swings
    it ~0.10 m forward and down. A producer that positions an object "0.40 m in front of the head"
    and then aims the camera at it has placed it against a head pose its own aim destroys. In one
    measured case a raised inspect intended 0.31 m eye-to-label and delivered 0.202 m, i.e. 40%
    inside the ``label_visibility.NEAR_M`` this codebase calls the closest readable range.

    Same clamp and same neck/head share as ``stabilize_gaze``'s inner loop, and iterated for the
    same reason -- the neck that turns the head is what moved the eye in the first place.
    """
    spine_rotation, spine_position = head_mount.spine3_frame(pose156, yaw, xy, z)
    raw_rotation, _raw_position = head_mount.head_pose(pose156, yaw, xy, z)
    neck_position = spine_position + spine_rotation @ head_mount.NECK_BONE
    up_reference = raw_rotation @ head_mount.HEAD_UP_LOCAL
    eye = _raw_position + raw_rotation @ head_mount.EYE_LOCAL
    neck_and_head = np.eye(3)
    for _ in range(FIXATION_ITERATIONS):
        bearing = _unit_or_zero(np.asarray(target, dtype=np.float64) - eye)
        if not np.any(bearing):
            bearing = raw_rotation @ head_mount.FACE_FORWARD_LOCAL
        desired = head_mount.basis_from_forward(bearing, up_reference)
        neck_and_head, _clamped = clamp_head_on_torso(spine_rotation.T @ desired)
        neck = head_mount.rodrigues(neck_share * log_rotation(neck_and_head))
        head_position = neck_position + (spine_rotation @ neck) @ head_mount.HEAD_BONE
        eye = head_position + (spine_rotation @ neck_and_head) @ head_mount.EYE_LOCAL
    return eye, spine_rotation @ neck_and_head


def stabilize_gaze(
    frames: list[Any],
    *,
    fps: float = 12.0,
    cutoff_hz: float = CUTOFF_HZ,
    neck_share: float = NECK_SHARE,
    fixation_m: float = FIXATION_DISTANCE_M,
    iterations: int = FIXATION_ITERATIONS,
    gaze_lead_seconds: float = GAZE_EASE_MAX_SECONDS,
    max_gaze_deg_per_frame: float = MAX_GAZE_DEG_PER_FRAME,
) -> StabilizationResult:
    """Re-author each frame's neck + head so the gaze holds a steady point in the world.

    Fixating a *point* rather than smoothing a *direction* is what makes this cancel the head's
    translation as well as its rotation. Holding a world point 2.5 m away means a 12 mm bob of
    the eye is absorbed by a 0.3 deg counter-rotation of the neck, so the image stops swinging --
    exactly the compensation a look-at camera performs implicitly, but performed here by the
    body, so the lens stays bolted to the head.

    The authored heading ``frame.yaw`` is never filtered: it is the wearer's intended course. It
    still moves the fixation point, so a deliberate turn passes through undamped.

    Moving the neck moves the head, which moves the eye the fixation is measured from, so the
    solve is iterated to its fixed point (see ``FIXATION_ITERATIONS``).
    """
    if not frames:
        return StabilizationResult([], 0, 0.0, 0.0, 0)
    if not 0.0 <= neck_share <= 1.0:
        raise ValueError("neck_share must be in [0, 1]")
    if cutoff_hz <= 0.0 or fps <= 0.0:
        raise ValueError("fps and cutoff_hz must be positive")
    if fixation_m <= 0.0 or iterations < 1:
        raise ValueError("fixation_m must be positive and iterations at least 1")
    sigma = fps / (2.0 * math.pi * cutoff_hz)
    spines = [
        head_mount.spine3_frame(frame.pose156, frame.yaw, frame.xy, frame.z)
        for frame in frames
    ]
    raw = [
        head_mount.head_pose(frame.pose156, frame.yaw, frame.xy, frame.z)
        for frame in frames
    ]
    raw_eyes = np.stack(
        [position + rotation @ head_mount.EYE_LOCAL for rotation, position in raw]
    )
    # Cuts are decided on the EYE, then imposed on both filters. See CUT_TELEPORT_M.
    bounds = teleport_bounds(raw_eyes)
    # Two compensations, separated so that neither eats the wearer's intent.
    #
    # Orientation is filtered in the HEADING frame -- ``Rz(-yaw)`` out, ``Rz(+yaw)`` back in --
    # so the authored course passes through untouched while the gait's sway is rejected. Filtering
    # it in world space instead makes the head lag every corner (measured: the neck correction
    # went to 38.6 deg, because the body turned and the smoothed gaze did not follow).
    #
    # Position is filtered in world space, and that is what buys translation compensation: the
    # fixation point is built on the DE-BOBBED eye, then aimed at from the real one.
    ideal_rotations = [
        head_mount.yaw_matrix(frame.yaw) @ local
        for frame, local in zip(
            frames,
            smooth_rotations(
                [
                    head_mount.yaw_matrix(-frame.yaw) @ rotation
                    for frame, (rotation, _) in zip(frames, raw)
                ],
                sigma,
                bounds,
            ),
        )
    ]
    up_references = [
        rotation @ head_mount.HEAD_UP_LOCAL for rotation in ideal_rotations
    ]
    ideal_eyes = smooth_points(raw_eyes, sigma, bounds)
    fixations = (
        ideal_eyes
        + np.stack(
            [rotation @ head_mount.FACE_FORWARD_LOCAL for rotation in ideal_rotations]
        )
        * fixation_m
    )
    fixations, gaze_frames = apply_gaze_intent(
        frames,
        fixations,
        ideal_eyes,
        sigma,
        bounds,
        max_lead_frames=max(1, int(round(gaze_lead_seconds * fps))),
    )
    out: list[Any] = []
    corrections: list[float] = []
    clamped_frames = 0
    rate_limited_frames = 0
    cut_at = set(bounds[1:-1])
    previous_bearing: np.ndarray | None = None
    for index, (
        frame,
        spine,
        (raw_rotation, _),
        eye,
        target,
        up_reference,
    ) in enumerate(zip(frames, spines, raw, raw_eyes, fixations, up_references)):
        spine_rotation, spine_position = spine
        neck_position = spine_position + spine_rotation @ head_mount.NECK_BONE
        neck_and_head = np.eye(3)
        clamped = False
        limited = False
        if index in cut_at:
            previous_bearing = None  # a real teleport: do not drag the aim across it
        for _ in range(iterations):
            bearing = _unit_or_zero(target - eye)
            if not np.any(bearing):
                bearing = raw_rotation @ head_mount.FACE_FORWARD_LOCAL
            # RATE LIMIT. The eases above are what make the aim move smoothly; this is the guard
            # that holds when something else moves it anyway -- a producer that writes a gaze with
            # no run-up, a target that jumps, a beat too short to ease across. Applied to the
            # world-space aim, so it bounds what the viewer actually sees.
            if previous_bearing is not None:
                swing = math.degrees(_angle_between(previous_bearing, bearing))
                if swing > max_gaze_deg_per_frame:
                    bearing = _slerp_unit(
                        previous_bearing, bearing, max_gaze_deg_per_frame / swing
                    )
                    limited = True
            desired = head_mount.basis_from_forward(bearing, up_reference)
            neck_and_head, clamped = clamp_head_on_torso(spine_rotation.T @ desired)
            neck = head_mount.rodrigues(neck_share * log_rotation(neck_and_head))
            head_position = (
                neck_position + (spine_rotation @ neck) @ head_mount.HEAD_BONE
            )
            eye = (
                head_position + (spine_rotation @ neck_and_head) @ head_mount.EYE_LOCAL
            )
        # Carry the REALISED aim, not the requested one, so the cap bounds what the lens does
        # even on a frame the cervical clamp also had a say in.
        previous_bearing = _unit_or_zero(
            spine_rotation @ neck_and_head @ head_mount.FACE_FORWARD_LOCAL
        )
        clamped_frames += int(clamped)
        rate_limited_frames += int(limited)
        corrections.append(
            math.degrees(
                float(
                    np.linalg.norm(
                        log_rotation(raw_rotation.T @ spine_rotation @ neck_and_head)
                    )
                )
            )
        )
        rotation_vector = log_rotation(neck_and_head)
        pose = np.array(frame.pose156, copy=True)
        pose[NECK_JOINT * 3 : NECK_JOINT * 3 + 3] = neck_share * rotation_vector
        head_lo = head_mount.HEAD_JOINT * 3
        pose[head_lo : head_lo + 3] = (1.0 - neck_share) * rotation_vector
        out.append(dataclasses.replace(frame, pose156=pose))
    return StabilizationResult(
        out,
        clamped_frames,
        float(np.mean(corrections)),
        float(np.max(corrections)),
        gaze_frames,
        rate_limited_frames,
    )

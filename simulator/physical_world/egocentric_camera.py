# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pure egocentric camera policy shared by dense and model-facing renderers.

The policy consumes every motion frame in order. Rendering and observation sampling
happen only after the trajectory is evaluated, so camera motion does not depend on how
many frames a caller chooses to deliver.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

Vec3 = tuple[float, float, float]
CameraMode = Literal["walk", "manip", "gaze"]


@dataclass(frozen=True)
class CameraConfig:
    walk_eye_forward: float = 0.18
    manip_eye_forward: float = 0.24
    manip_eye_lift: float = 0.05
    manip_fallback_ahead: float = 0.6
    crouch_eye_fraction: float = 0.6
    anti_backward_min_distance: float = 0.25
    anti_backward_ahead: float = 0.5
    anti_backward_max_down: float = 0.6
    anti_backward_max_up: float = 0.05
    doorway_activation_radius: float = 1.8
    doorway_aim_drop: float = 0.1
    turn_rate_deg_s: float = 144.0
    gaze_acquisition_rate_deg_s: float = 180.0
    zoom_tracking_slowdown: float = 5.0
    zoom_heading_threshold_deg: float = 8.0
    max_head_yaw_deg: float = 60.0
    max_head_pitch_down_deg: float = 55.0
    max_head_pitch_up_deg: float = 55.0
    base_focal_mm: float = 16.0
    horizontal_aperture_mm: float = 36.0
    zoom_target_frame_fraction: float = 0.45
    max_focal_multiple: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.crouch_eye_fraction <= 1.0:
            raise ValueError("crouch_eye_fraction must be in [0, 1]")
        if self.turn_rate_deg_s <= 0.0 or self.gaze_acquisition_rate_deg_s <= 0.0:
            raise ValueError("camera turn rates must be positive")
        if not 0.0 < self.max_head_yaw_deg < 180.0:
            raise ValueError("max_head_yaw_deg must be in (0, 180)")
        if not 0.0 < self.max_head_pitch_down_deg < 90.0:
            raise ValueError("max_head_pitch_down_deg must be in (0, 90)")
        if not 0.0 < self.max_head_pitch_up_deg < 90.0:
            raise ValueError("max_head_pitch_up_deg must be in (0, 90)")
        if self.base_focal_mm <= 0.0 or self.horizontal_aperture_mm <= 0.0:
            raise ValueError("camera intrinsics must be positive")
        if not 0.0 < self.zoom_target_frame_fraction <= 1.0:
            raise ValueError("zoom_target_frame_fraction must be in (0, 1]")


@dataclass(frozen=True)
class SceneFraming:
    floor_z: float = 0.0
    eye_drop: float = 0.0
    walk_ahead: float = 3.0
    walk_downtilt: float = 0.25
    manip_aim_drop: float = 0.0


@dataclass(frozen=True)
class Doorway:
    center_xy: tuple[float, float]


@dataclass(frozen=True)
class CameraFrame:
    head_world: Sequence[float]
    body_forward: Sequence[float]
    mode: CameraMode
    dt_seconds: float
    focus_world: Sequence[float] | None = None
    focus_visible: bool = False
    requested_zoom: float = 0.0
    zoom_span: float | None = None
    doorways: tuple[Doorway, ...] = ()
    eye_offset_world: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraState:
    look_direction: Vec3 | None = None
    body_forward_h: Vec3 | None = None
    standing_head_z: float | None = None
    processed_frames: int = 0


@dataclass(frozen=True)
class CameraDiagnostics:
    raw_eye: Vec3
    doorway_override_applied: bool
    anti_backward_applied: bool
    physiological_yaw_clamped: bool
    physiological_pitch_clamped: bool
    crouch_clamped: bool
    requested_turn_deg: float
    applied_turn_deg: float
    turn_rate_deg_s: float
    heading_error_deg: float
    zoom_suppressed_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CameraOutput:
    eye: Vec3
    aim: Vec3
    look: Vec3
    focal_mm: float
    effective_zoom: float
    diagnostics: CameraDiagnostics


@dataclass(frozen=True)
class CameraStep:
    output: CameraOutput
    state: CameraState


@dataclass(frozen=True)
class CameraTrajectory:
    outputs: tuple[CameraOutput, ...]
    final_state: CameraState


@dataclass(frozen=True)
class SampledCameraTrajectory:
    global_indices: tuple[int, ...]
    outputs: tuple[CameraOutput, ...]
    final_state: CameraState


@dataclass(frozen=True)
class CameraClearance:
    eye: Vec3
    look: Vec3
    shift_m: float
    adjusted: bool


@dataclass(frozen=True)
class _PoseChoice:
    eye: Vec3
    aim: Vec3
    doorway_override_applied: bool
    anti_backward_applied: bool


@dataclass(frozen=True)
class _FrameGeometry:
    head: Vec3
    body_forward: Vec3
    body_forward_h: Vec3
    standing_head_z: float
    pose: _PoseChoice
    eye: Vec3
    crouch_clamped: bool


@dataclass(frozen=True)
class _TurnResult:
    look: Vec3
    requested_deg: float
    applied_deg: float
    rate_deg_s: float
    physiological_yaw_clamped: bool
    physiological_pitch_clamped: bool


@dataclass(frozen=True)
class _ZoomResult:
    value: float
    suppressed_reasons: tuple[str, ...]


DEFAULT_CAMERA_CONFIG = CameraConfig()
DEFAULT_SCENE_FRAMING = SceneFraming()
INITIAL_CAMERA_STATE = CameraState()


def step_camera(
    frame: CameraFrame,
    state: CameraState = INITIAL_CAMERA_STATE,
    *,
    framing: SceneFraming = DEFAULT_SCENE_FRAMING,
    config: CameraConfig = DEFAULT_CAMERA_CONFIG,
) -> CameraStep:
    """Advance the camera policy by one chronological motion frame."""
    geometry = _prepare_geometry(frame, state, framing, config)
    target_look = _normalize(
        _sub(geometry.pose.aim, geometry.eye), fallback=geometry.body_forward
    )
    desired_look, physiological_yaw_clamped = _clamp_head_yaw(
        target_look, geometry.body_forward_h, config.max_head_yaw_deg
    )
    desired_look, physiological_pitch_clamped = _clamp_head_pitch(
        desired_look,
        config.max_head_pitch_down_deg,
        config.max_head_pitch_up_deg,
    )
    turn = _turn_toward(state, desired_look, geometry.body_forward_h, frame, config)
    heading_error = _angle_deg(turn.look, target_look)
    zoom = _effective_zoom(frame, heading_error, config)
    output = _camera_output(
        geometry,
        turn,
        zoom,
        heading_error,
        physiological_yaw_clamped or turn.physiological_yaw_clamped,
        physiological_pitch_clamped or turn.physiological_pitch_clamped,
        _focal_mm(frame, geometry.eye, zoom.value, config),
    )
    return CameraStep(output=output, state=_next_state(state, geometry, turn))


def _camera_output(
    geometry: _FrameGeometry,
    turn: _TurnResult,
    zoom: _ZoomResult,
    heading_error: float,
    physiological_yaw_clamped: bool,
    physiological_pitch_clamped: bool,
    focal_mm: float,
) -> CameraOutput:
    diagnostics = CameraDiagnostics(
        raw_eye=geometry.pose.eye,
        doorway_override_applied=geometry.pose.doorway_override_applied,
        anti_backward_applied=geometry.pose.anti_backward_applied,
        physiological_yaw_clamped=physiological_yaw_clamped,
        physiological_pitch_clamped=physiological_pitch_clamped,
        crouch_clamped=geometry.crouch_clamped,
        requested_turn_deg=turn.requested_deg,
        applied_turn_deg=turn.applied_deg,
        turn_rate_deg_s=turn.rate_deg_s,
        heading_error_deg=heading_error,
        zoom_suppressed_reasons=zoom.suppressed_reasons,
    )
    return CameraOutput(
        eye=geometry.eye,
        aim=geometry.pose.aim,
        look=turn.look,
        focal_mm=focal_mm,
        effective_zoom=zoom.value,
        diagnostics=diagnostics,
    )


def _next_state(
    state: CameraState,
    geometry: _FrameGeometry,
    turn: _TurnResult,
) -> CameraState:
    return CameraState(
        look_direction=turn.look,
        body_forward_h=geometry.body_forward_h,
        standing_head_z=geometry.standing_head_z,
        processed_frames=state.processed_frames + 1,
    )


def evaluate_camera_trajectory(
    frames: Iterable[CameraFrame],
    *,
    initial_state: CameraState = INITIAL_CAMERA_STATE,
    framing: SceneFraming = DEFAULT_SCENE_FRAMING,
    config: CameraConfig = DEFAULT_CAMERA_CONFIG,
) -> CameraTrajectory:
    """Evaluate all motion frames in order and retain every camera result."""
    state = initial_state
    outputs: list[CameraOutput] = []
    for frame in frames:
        result = step_camera(
            frame,
            state,
            framing=framing,
            config=config,
        )
        outputs.append(result.output)
        state = result.state
    return CameraTrajectory(tuple(outputs), state)


def sample_camera_trajectory(
    trajectory: CameraTrajectory, sample_count: int | None
) -> SampledCameraTrajectory:
    """Sample an already-evaluated trajectory without changing camera dynamics."""
    indices = uniform_sample_indices(len(trajectory.outputs), sample_count)
    return SampledCameraTrajectory(
        global_indices=indices,
        outputs=tuple(trajectory.outputs[index] for index in indices),
        final_state=trajectory.final_state,
    )


def evaluate_and_sample_camera_trajectory(
    frames: Iterable[CameraFrame],
    sample_count: int | None,
    *,
    initial_state: CameraState = INITIAL_CAMERA_STATE,
    framing: SceneFraming = DEFAULT_SCENE_FRAMING,
    config: CameraConfig = DEFAULT_CAMERA_CONFIG,
) -> SampledCameraTrajectory:
    """Evaluate densely first, then select observations by global frame index."""
    trajectory = evaluate_camera_trajectory(
        frames,
        initial_state=initial_state,
        framing=framing,
        config=config,
    )
    return sample_camera_trajectory(trajectory, sample_count)


def enforce_camera_clearance(
    eye: Sequence[float],
    look: Sequence[float],
    *,
    head_world: Sequence[float],
    occluders: Iterable[tuple[Sequence[float], Sequence[float]]],
    clearance_m: float = 0.04,
    min_head_distance_m: float = 0.12,
) -> CameraClearance:
    """Keep the FPV origin outside carried geometry while preserving its aim point.

    The desired eye remains authoritative until it enters an occluder's clearance-expanded AABB.
    When that happens, retreat along the head-to-eye segment to the near face. This keeps the camera
    on the wearer's side of a carried object instead of letting the object cross the lens and fill the
    frame. The function is pure so dense and sampled renderers can share the same decision.
    """
    if clearance_m < 0.0 or min_head_distance_m <= 0.0:
        raise ValueError("camera clearances must be non-negative and non-zero")
    original_eye = _vec3(eye, "eye")
    current_eye = original_eye
    head = _vec3(head_world, "head_world")
    original_look = _normalize(_vec3(look, "look"))
    aim = _add(original_eye, original_look)
    expanded_boxes = tuple(
        _expanded_box(minimum, maximum, clearance_m) for minimum, maximum in occluders
    )
    for _ in range(max(1, 2 * len(expanded_boxes))):
        containing_box = next(
            (
                (minimum, maximum)
                for minimum, maximum in expanded_boxes
                if _point_in_box(current_eye, minimum, maximum)
            ),
            None,
        )
        if containing_box is None:
            break
        next_eye = _retreat_eye(
            head,
            current_eye,
            containing_box[0],
            containing_box[1],
            min_head_distance_m,
        )
        if _length(_sub(next_eye, current_eye)) <= 1e-9:
            break
        current_eye = next_eye
    if any(
        _point_in_box(current_eye, minimum, maximum)
        for minimum, maximum in expanded_boxes
    ):
        current_eye = _safe_eye_outside_all_boxes(
            head,
            current_eye,
            expanded_boxes,
            min_head_distance_m,
        )
    shift = _length(_sub(current_eye, original_eye))
    return CameraClearance(
        eye=current_eye,
        look=_normalize(_sub(aim, current_eye), fallback=original_look),
        shift_m=shift,
        adjusted=shift > 1e-9,
    )


def _safe_eye_outside_all_boxes(
    head: Vec3,
    eye: Vec3,
    boxes: Sequence[tuple[Vec3, Vec3]],
    min_head_distance: float,
) -> Vec3:
    epsilon = 1e-3
    candidates: list[Vec3] = []
    for axis in range(3):
        axis_min = min(minimum[axis] for minimum, _ in boxes)
        axis_max = max(maximum[axis] for _, maximum in boxes)
        for face in (axis_min - epsilon, axis_max + epsilon):
            candidate = list(eye)
            candidate[axis] = face
            value = tuple(candidate)
            if _length(_sub(value, head)) < min_head_distance:
                continue
            if any(
                _point_in_box(value, minimum, maximum) for minimum, maximum in boxes
            ):
                continue
            candidates.append(value)
    if not candidates:
        raise RuntimeError("cannot place camera outside occluder clearance boxes")
    return min(candidates, key=lambda candidate: _length(_sub(candidate, eye)))


def _expanded_box(
    minimum: Sequence[float], maximum: Sequence[float], clearance: float
) -> tuple[Vec3, Vec3]:
    mn = _vec3(minimum, "occluder minimum")
    mx = _vec3(maximum, "occluder maximum")
    if any(mn[axis] > mx[axis] for axis in range(3)):
        raise ValueError("occluder minimum must not exceed maximum")
    pad = (clearance, clearance, clearance)
    return _sub(mn, pad), _add(mx, pad)


def _point_in_box(point: Vec3, minimum: Vec3, maximum: Vec3) -> bool:
    return all(minimum[axis] <= point[axis] <= maximum[axis] for axis in range(3))


def _retreat_eye(
    head: Vec3,
    eye: Vec3,
    minimum: Vec3,
    maximum: Vec3,
    min_head_distance: float,
) -> Vec3:
    ray = _sub(eye, head)
    ray_length = _length(ray)
    if ray_length <= 1e-9 or _point_in_box(head, minimum, maximum):
        return _nearest_safe_face(head, eye, minimum, maximum, min_head_distance)
    entry = _segment_box_entry(head, eye, minimum, maximum)
    if entry is None:
        return _nearest_safe_face(head, eye, minimum, maximum, min_head_distance)
    retreat = max(0.0, entry - min(1e-3 / ray_length, 0.01))
    candidate = _add(head, _scale(ray, retreat))
    if _length(_sub(candidate, head)) >= min_head_distance:
        return candidate
    return _nearest_safe_face(head, eye, minimum, maximum, min_head_distance)


def _segment_box_entry(
    start: Vec3, end: Vec3, minimum: Vec3, maximum: Vec3
) -> float | None:
    direction = _sub(end, start)
    enter, leave = 0.0, 1.0
    for axis in range(3):
        delta = direction[axis]
        if abs(delta) <= 1e-12:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return None
            continue
        near = (minimum[axis] - start[axis]) / delta
        far = (maximum[axis] - start[axis]) / delta
        if near > far:
            near, far = far, near
        enter, leave = max(enter, near), min(leave, far)
        if enter > leave:
            return None
    return enter


def _nearest_safe_face(
    head: Vec3,
    eye: Vec3,
    minimum: Vec3,
    maximum: Vec3,
    min_head_distance: float,
) -> Vec3:
    epsilon = 1e-3
    candidates: list[Vec3] = []
    for axis in range(3):
        for face in (minimum[axis] - epsilon, maximum[axis] + epsilon):
            candidate = list(eye)
            candidate[axis] = face
            value = tuple(candidate)
            if _length(_sub(value, head)) >= min_head_distance:
                candidates.append(value)
    if not candidates:
        direction = _normalize(_sub(eye, head), fallback=(1.0, 0.0, 0.0))
        return _add(head, _scale(direction, min_head_distance))
    return min(
        candidates,
        key=lambda candidate: (
            _length(_sub(candidate, head)),
            _length(_sub(candidate, eye)),
        ),
    )


def uniform_sample_indices(length: int, sample_count: int | None) -> tuple[int, ...]:
    if length <= 0:
        return ()
    if sample_count is None or sample_count >= length:
        return tuple(range(length))
    if sample_count <= 0:
        return ()
    if sample_count == 1:
        return (length - 1,)
    return tuple(
        sorted(
            {
                round(index * (length - 1) / (sample_count - 1))
                for index in range(sample_count)
            }
        )
    )


def _prepare_geometry(
    frame: CameraFrame,
    state: CameraState,
    framing: SceneFraming,
    config: CameraConfig,
) -> _FrameGeometry:
    if frame.dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    head = _vec3(frame.head_world, "head_world")
    body_forward = _normalize(_vec3(frame.body_forward, "body_forward"))
    body_forward_h = _horizontal_direction(body_forward)
    standing_head_z = _standing_head_z(state, head)
    pose = _choose_pose(frame, head, body_forward, body_forward_h, framing, config)
    eye_offset = _vec3(frame.eye_offset_world, "eye_offset_world")
    unoffset_eye = _choose_eye(
        head,
        body_forward,
        body_forward_h,
        frame.mode,
        framing,
        config,
    )
    raw_eye = _add(unoffset_eye, eye_offset)
    eye = default_policy_eye(
        head,
        body_forward,
        frame.mode,
        eye_offset_world=eye_offset,
        standing_head_z=standing_head_z,
        framing=framing,
        config=config,
    )
    pose = _PoseChoice(
        eye=raw_eye,
        aim=pose.aim,
        doorway_override_applied=pose.doorway_override_applied,
        anti_backward_applied=pose.anti_backward_applied,
    )
    crouch_clamped = eye != raw_eye
    return _FrameGeometry(
        head=head,
        body_forward=body_forward,
        body_forward_h=body_forward_h,
        standing_head_z=standing_head_z,
        pose=pose,
        eye=eye,
        crouch_clamped=crouch_clamped,
    )


def default_policy_eye(
    head_world: Sequence[float],
    body_forward: Sequence[float],
    mode: CameraMode,
    *,
    eye_offset_world: Sequence[float] = (0.0, 0.0, 0.0),
    standing_head_z: float | None = None,
    framing: SceneFraming = DEFAULT_SCENE_FRAMING,
    config: CameraConfig = DEFAULT_CAMERA_CONFIG,
) -> Vec3:
    """Return the raw default-renderer eye used before scene-occluder clearance."""
    head = _vec3(head_world, "head_world")
    forward = _normalize(_vec3(body_forward, "body_forward"))
    forward_h = _horizontal_direction(forward)
    eye = _add(
        _choose_eye(head, forward, forward_h, mode, framing, config),
        _vec3(eye_offset_world, "eye_offset_world"),
    )
    standing_z = head[2] if standing_head_z is None else float(standing_head_z)
    return _apply_crouch_floor(eye, standing_z, framing, config)[0]


def _choose_eye(
    head: Vec3,
    body_forward: Vec3,
    body_forward_h: Vec3,
    mode: CameraMode,
    framing: SceneFraming,
    config: CameraConfig,
) -> Vec3:
    if mode == "walk":
        return _walk_pose(head, body_forward, framing, config).eye
    if mode in ("manip", "gaze"):
        return _add(
            _add(head, _scale(body_forward_h, config.manip_eye_forward)),
            (0.0, 0.0, config.manip_eye_lift - framing.eye_drop),
        )
    raise ValueError(f"unsupported camera mode: {mode}")


def _choose_pose(
    frame: CameraFrame,
    head: Vec3,
    body_forward: Vec3,
    body_forward_h: Vec3,
    framing: SceneFraming,
    config: CameraConfig,
) -> _PoseChoice:
    if frame.mode == "walk":
        return _walk_pose(head, body_forward, framing, config)
    if frame.mode not in ("manip", "gaze"):
        raise ValueError(f"unsupported camera mode: {frame.mode}")
    focus = _optional_vec3(frame.focus_world, "focus_world")
    if frame.mode == "gaze" and focus is None:
        raise ValueError("gaze mode requires focus_world")
    return _manip_pose(frame, head, body_forward_h, focus, framing, config)


def _walk_pose(
    head: Vec3,
    body_forward: Vec3,
    framing: SceneFraming,
    config: CameraConfig,
) -> _PoseChoice:
    eye = _add(
        _add(head, _scale(body_forward, config.walk_eye_forward)),
        (0.0, 0.0, -framing.eye_drop),
    )
    aim = _add(
        _add(head, _scale(body_forward, framing.walk_ahead)),
        (0.0, 0.0, -framing.walk_downtilt),
    )
    return _PoseChoice(eye, aim, False, False)


def _manip_pose(
    frame: CameraFrame,
    head: Vec3,
    body_forward_h: Vec3,
    focus: Vec3 | None,
    framing: SceneFraming,
    config: CameraConfig,
) -> _PoseChoice:
    eye = _add(
        _add(head, _scale(body_forward_h, config.manip_eye_forward)),
        (0.0, 0.0, config.manip_eye_lift - framing.eye_drop),
    )
    aim = _manip_aim(eye, head, body_forward_h, focus, framing, config)
    doorway_applied = False
    if frame.mode == "manip" and focus is None:
        doorway = _nearest_doorway(head, frame.doorways, config)
        if doorway is not None:
            aim = (
                doorway.center_xy[0],
                doorway.center_xy[1],
                head[2] - config.doorway_aim_drop,
            )
            doorway_applied = True
    anti_backward = False
    if frame.mode == "manip" and _needs_forward_fallback(
        eye, aim, body_forward_h, config
    ):
        aim = _forward_fallback(eye, aim, body_forward_h, config)
        anti_backward = True
    return _PoseChoice(eye, aim, doorway_applied, anti_backward)


def _manip_aim(
    eye: Vec3,
    head: Vec3,
    body_forward_h: Vec3,
    focus: Vec3 | None,
    framing: SceneFraming,
    config: CameraConfig,
) -> Vec3:
    if focus is None:
        return _add(head, _scale(body_forward_h, config.manip_fallback_ahead))
    return _add(focus, (0.0, 0.0, -framing.manip_aim_drop))


def _nearest_doorway(
    head: Vec3, doorways: tuple[Doorway, ...], config: CameraConfig
) -> Doorway | None:
    eligible = [
        doorway
        for doorway in doorways
        if math.hypot(head[0] - doorway.center_xy[0], head[1] - doorway.center_xy[1])
        < config.doorway_activation_radius
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda doorway: math.hypot(
            head[0] - doorway.center_xy[0], head[1] - doorway.center_xy[1]
        ),
    )


def _needs_forward_fallback(
    eye: Vec3, aim: Vec3, body_forward_h: Vec3, config: CameraConfig
) -> bool:
    offset = _sub(aim, eye)
    horizontal = (offset[0], offset[1], 0.0)
    return (
        _length(horizontal) < config.anti_backward_min_distance
        or _dot(horizontal, body_forward_h) <= 0.0
    )


def _forward_fallback(
    eye: Vec3, aim: Vec3, body_forward_h: Vec3, config: CameraConfig
) -> Vec3:
    dz = min(
        config.anti_backward_max_up,
        max(-config.anti_backward_max_down, aim[2] - eye[2]),
    )
    return _add(
        _add(eye, _scale(body_forward_h, config.anti_backward_ahead)),
        (0.0, 0.0, dz),
    )


def _standing_head_z(state: CameraState, head: Vec3) -> float:
    if state.standing_head_z is None:
        return head[2]
    return max(state.standing_head_z, head[2])


def _apply_crouch_floor(
    eye: Vec3,
    standing_head_z: float,
    framing: SceneFraming,
    config: CameraConfig,
) -> tuple[Vec3, bool]:
    eye_floor = framing.floor_z + config.crouch_eye_fraction * (
        standing_head_z - framing.floor_z
    )
    if eye[2] >= eye_floor:
        return eye, False
    return (eye[0], eye[1], eye_floor), True


def _clamp_head_yaw(
    target: Vec3, body_forward_h: Vec3, max_yaw_deg: float
) -> tuple[Vec3, bool]:
    horizontal_length = math.hypot(target[0], target[1])
    if horizontal_length <= 1e-12:
        return target, False
    body_yaw = math.atan2(body_forward_h[1], body_forward_h[0])
    target_yaw = math.atan2(target[1], target[0])
    yaw_delta = _wrap_radians(target_yaw - body_yaw)
    max_yaw = math.radians(max_yaw_deg)
    clamped_delta = min(max_yaw, max(-max_yaw, yaw_delta))
    if abs(clamped_delta - yaw_delta) <= 1e-12:
        return target, False
    yaw = body_yaw + clamped_delta
    clamped = (
        math.cos(yaw) * horizontal_length,
        math.sin(yaw) * horizontal_length,
        target[2],
    )
    return _normalize(clamped), True


def _clamp_head_pitch(
    target: Vec3, max_down_deg: float, max_up_deg: float
) -> tuple[Vec3, bool]:
    horizontal = math.hypot(target[0], target[1])
    pitch = math.atan2(target[2], horizontal)
    clamped_pitch = min(
        math.radians(max_up_deg),
        max(-math.radians(max_down_deg), pitch),
    )
    if abs(clamped_pitch - pitch) <= 1e-12:
        return target, False
    yaw = math.atan2(target[1], target[0])
    horizontal_scale = math.cos(clamped_pitch)
    return (
        horizontal_scale * math.cos(yaw),
        horizontal_scale * math.sin(yaw),
        math.sin(clamped_pitch),
    ), True


def _wrap_radians(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _turn_toward(
    state: CameraState,
    desired: Vec3,
    body_forward_h: Vec3,
    frame: CameraFrame,
    config: CameraConfig,
) -> _TurnResult:
    if state.look_direction is None:
        return _TurnResult(
            desired,
            0.0,
            0.0,
            config.turn_rate_deg_s,
            False,
            False,
        )
    previous = _transport_with_body(
        _normalize(state.look_direction), state.body_forward_h, body_forward_h
    )
    previous, previous_clamped = _clamp_head_yaw(
        previous, body_forward_h, config.max_head_yaw_deg
    )
    previous, previous_pitch_clamped = _clamp_head_pitch(
        previous,
        config.max_head_pitch_down_deg,
        config.max_head_pitch_up_deg,
    )
    requested_turn = _angle_deg(previous, desired)
    rate = _turn_rate(frame, requested_turn, config)
    max_turn = rate * frame.dt_seconds
    look = _slerp_capped(previous, desired, math.radians(max_turn))
    look, look_clamped = _clamp_head_yaw(look, body_forward_h, config.max_head_yaw_deg)
    look, look_pitch_clamped = _clamp_head_pitch(
        look,
        config.max_head_pitch_down_deg,
        config.max_head_pitch_up_deg,
    )
    return _TurnResult(
        look,
        requested_turn,
        _angle_deg(previous, look),
        rate,
        previous_clamped or look_clamped,
        previous_pitch_clamped or look_pitch_clamped,
    )


def _transport_with_body(
    look: Vec3,
    previous_body_forward_h: Vec3 | None,
    body_forward_h: Vec3,
) -> Vec3:
    if previous_body_forward_h is None:
        return look
    previous_yaw = math.atan2(previous_body_forward_h[1], previous_body_forward_h[0])
    body_yaw = math.atan2(body_forward_h[1], body_forward_h[0])
    yaw_delta = _wrap_radians(body_yaw - previous_yaw)
    cos_yaw, sin_yaw = math.cos(yaw_delta), math.sin(yaw_delta)
    return _normalize(
        (
            cos_yaw * look[0] - sin_yaw * look[1],
            sin_yaw * look[0] + cos_yaw * look[1],
            look[2],
        )
    )


def _turn_rate(
    frame: CameraFrame, requested_turn_deg: float, config: CameraConfig
) -> float:
    zoom = min(1.0, max(0.0, frame.requested_zoom))
    if zoom <= 0.0:
        return config.turn_rate_deg_s
    if requested_turn_deg > config.zoom_heading_threshold_deg:
        return config.gaze_acquisition_rate_deg_s
    return config.turn_rate_deg_s / (1.0 + config.zoom_tracking_slowdown * zoom)


def _effective_zoom(
    frame: CameraFrame, heading_error_deg: float, config: CameraConfig
) -> _ZoomResult:
    requested = min(1.0, max(0.0, frame.requested_zoom))
    if requested <= 0.0:
        return _ZoomResult(0.0, ())
    reasons: list[str] = []
    if frame.focus_world is None:
        reasons.append("missing_focus")
    if frame.zoom_span is None or frame.zoom_span <= 0.0:
        reasons.append("missing_zoom_span")
    if not frame.focus_visible:
        reasons.append("focus_not_visible")
    if heading_error_deg > config.zoom_heading_threshold_deg:
        reasons.append("heading_error")
    if reasons:
        return _ZoomResult(0.0, tuple(reasons))
    return _ZoomResult(requested, ())


def _focal_mm(
    frame: CameraFrame, eye: Vec3, zoom: float, config: CameraConfig
) -> float:
    if zoom <= 0.0 or frame.focus_world is None or frame.zoom_span is None:
        return config.base_focal_mm
    focus = _vec3(frame.focus_world, "focus_world")
    distance = max(_length(_sub(focus, eye)), 1e-3)
    base_hfov = 2.0 * math.atan(
        0.5 * config.horizontal_aperture_mm / config.base_focal_mm
    )
    subtend = 2.0 * math.atan(0.5 * frame.zoom_span / distance)
    target_hfov = min(base_hfov, subtend / config.zoom_target_frame_fraction)
    target_hfov = max(target_hfov, 1e-6)
    target_focal = 0.5 * config.horizontal_aperture_mm / math.tan(target_hfov / 2.0)
    target_focal = min(target_focal, config.base_focal_mm * config.max_focal_multiple)
    return config.base_focal_mm + (target_focal - config.base_focal_mm) * zoom


def _horizontal_direction(direction: Vec3) -> Vec3:
    horizontal = (direction[0], direction[1], 0.0)
    if _length(horizontal) <= 1e-12:
        raise ValueError("body_forward must have a horizontal component")
    return _normalize(horizontal)


def _slerp_capped(previous: Vec3, target: Vec3, max_angle_rad: float) -> Vec3:
    angle = math.radians(_angle_deg(previous, target))
    if angle <= max_angle_rad or angle <= 1e-12:
        return target
    if math.pi - angle <= 1e-7:
        tangent = _orthogonal(previous)
    else:
        tangent = _normalize(
            _sub(target, _scale(previous, math.cos(angle))),
            fallback=_orthogonal(previous),
        )
    return _normalize(
        _add(
            _scale(previous, math.cos(max_angle_rad)),
            _scale(tangent, math.sin(max_angle_rad)),
        )
    )


def _orthogonal(vector: Vec3) -> Vec3:
    basis = min(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        key=lambda candidate: abs(_dot(vector, candidate)),
    )
    return _normalize(_cross(vector, basis))


def _angle_deg(first: Vec3, second: Vec3) -> float:
    dot = min(1.0, max(-1.0, _dot(_normalize(first), _normalize(second))))
    return math.degrees(math.acos(dot))


def _optional_vec3(value: Sequence[float] | None, name: str) -> Vec3 | None:
    return None if value is None else _vec3(value, name)


def _vec3(value: Sequence[float], name: str) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    vector = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain finite values")
    return vector


def _normalize(vector: Vec3, fallback: Vec3 | None = None) -> Vec3:
    length = _length(vector)
    if length <= 1e-12:
        if fallback is None:
            raise ValueError("cannot normalize a zero-length vector")
        return _normalize(fallback)
    return _scale(vector, 1.0 / length)


def _length(vector: Vec3) -> float:
    return math.sqrt(_dot(vector, vector))


def _add(first: Vec3, second: Vec3) -> Vec3:
    return (first[0] + second[0], first[1] + second[1], first[2] + second[2])


def _sub(first: Vec3, second: Vec3) -> Vec3:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _scale(vector: Vec3, factor: float) -> Vec3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(first: Vec3, second: Vec3) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(first: Vec3, second: Vec3) -> Vec3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )

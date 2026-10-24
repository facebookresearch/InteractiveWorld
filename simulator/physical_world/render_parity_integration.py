# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Opt-in state capture shared by offline and model-facing render loops."""

from __future__ import annotations

import dataclasses
import enum
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from simulator.physical_world.render_parity_trace import (
    AUTHORED_PRE_STEP,
    ClockDelta,
    compare_trace_files,
    DELIVERED_READBACK,
    dynamic_prim_snapshot,
    normalize_trace_value,
    ParityConfig,
    ParityReport,
    pose_snapshot,
    POST_POSITIVE_STEP,
    RenderInputIdentity,
    RenderTraceWriter,
    rgb_snapshot,
    stable_trace_hash,
    TraceEvent,
    TraceFormatError,
)


TRACE_PATH_ENV = "ISAAC_RENDER_TRACE_PATH"


@dataclasses.dataclass(frozen=True)
class RenderStateSnapshot:
    """One live renderer-state sample ready for a lifecycle trace event."""

    poses: dict[str, object]
    intrinsics: dict[str, object]
    static_state: dict[str, object]
    dynamic_state: dict[str, object]
    metadata: dict[str, object]


def _normalized_dict(value: Mapping[str, object]) -> dict[str, object]:
    normalized = normalize_trace_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("renderer trace state must normalize to a mapping")
    return normalized


def _camera_orientation(look: object) -> tuple[float, float, float, float]:
    direction = np.asarray(look, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("camera look direction must be finite and non-zero")
    direction /= norm
    yaw = float(np.arctan2(direction[1], direction[0]))
    pitch = float(np.arctan2(direction[2], np.hypot(*direction[:2])))
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    return (float(cy * cp), float(sy * sp), float(-cy * sp), float(sy * cp))


def camera_pose_snapshot(pose: object) -> dict[str, object]:
    """Capture the delivered eye, look vector, focal length, and view orientation."""
    eye = tuple(float(value) for value in pose.eye)
    look = tuple(float(value) for value in pose.look)
    return pose_snapshot(
        position=eye,
        orientation=_camera_orientation(look),
        eye=eye,
        look=look,
        focal_mm=float(pose.focal_mm),
    )


def build_render_state_snapshot(
    *,
    frame: object,
    camera_pose: object,
    width: int,
    height: int,
    horizontal_aperture: float,
    body_vertices: object,
    joints: object,
    object_transform: object,
    object_world_transforms: Mapping[str, object],
    openable_state: object,
    solid_boxes: object,
    dynamic_state: Mapping[str, object],
    renderer: str,
) -> RenderStateSnapshot:
    """Build the common offline/online renderer snapshot schema."""
    focal_mm = float(camera_pose.focal_mm)
    poses = {"camera": camera_pose_snapshot(camera_pose)}
    intrinsics = {
        "camera": {
            "focal_length": focal_mm,
            "horizontal_aperture": float(horizontal_aperture),
            "resolution": [int(width), int(height)],
        }
    }
    static_state = {
        "frame_hash": stable_trace_hash(frame),
        "body_hash": stable_trace_hash(np.asarray(body_vertices)),
        "joint_hash": stable_trace_hash(np.asarray(joints)),
        "object_transform": object_transform,
        "object_world_transforms": dict(object_world_transforms),
        "post_openable_state": openable_state,
        "live_solid_boxes_hash": stable_trace_hash(solid_boxes),
    }
    return RenderStateSnapshot(
        poses=_normalized_dict(poses),
        intrinsics=_normalized_dict(intrinsics),
        static_state=_normalized_dict(static_state),
        dynamic_state=_normalized_dict(dict(dynamic_state)),
        metadata={"renderer": str(renderer)},
    )


def camera_horizontal_aperture(camera_op: object) -> float:
    """Read the live USD camera aperture used by the render product."""
    from pxr import UsdGeom

    camera = UsdGeom.Camera(camera_op.GetAttr().GetPrim())
    value = camera.GetHorizontalApertureAttr().Get()
    if value is None:
        raise RuntimeError("FPV camera has no horizontal aperture")
    return float(value)


def object_paths_for_frame(
    frame: object,
    rig: object,
    dynamic_paths: Iterable[str] = (),
    extra_paths: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return every manipulated or dynamic prim relevant to the current frame."""
    paths = {
        str(path).rstrip("/")
        for path in (*tuple(dynamic_paths), *tuple(extra_paths))
        if path
    }
    object_path = getattr(frame, "obj_path", None) or getattr(rig, "_obj_path", None)
    if object_path:
        paths.add(str(object_path).rstrip("/"))
    for path, _relative_transform in getattr(frame, "carried", ()) or ():
        paths.add(str(path).rstrip("/"))
    return tuple(sorted(paths))


def persistent_scene_object_paths(
    scene: object,
    clutter_paths: Iterable[str] = (),
    openable_paths: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return every idle graspable and render-replayed clutter prim in the scene."""
    paths = {
        str(path).rstrip("/")
        for path in (*tuple(clutter_paths), *tuple(openable_paths))
        if path
    }
    for entity in (getattr(scene, "entities", None) or {}).values():
        role = getattr(entity, "role", None)
        role_value = str(getattr(role, "value", role) or "").lower()
        path = getattr(entity, "prim_path", None)
        if path and role_value in {"graspable", "role.graspable"}:
            paths.add(str(path).rstrip("/"))
    for item in getattr(scene, "grasp_objs", None) or ():
        path = (
            item.get("path")
            if isinstance(item, Mapping)
            else getattr(item, "prim_path", None)
        )
        if path:
            paths.add(str(path).rstrip("/"))
    return tuple(sorted(paths))


def openable_controller_paths(controllers: Mapping[object, object]) -> tuple[str, ...]:
    """Return every actual moving leaf/drawer prim authored by openable replay."""
    paths = set()
    for controller in (controllers or {}).values():
        for path in controller.get("moving_paths", ()) or ():
            if path:
                paths.add(str(path).rstrip("/"))
        for spec in controller.get("leaf_specs", ()) or ():
            if spec and spec[0]:
                paths.add(str(spec[0]).rstrip("/"))
    return tuple(sorted(paths))


def snapshot_world_transforms(stage: object, paths: Iterable[str]) -> dict[str, object]:
    """Read actual world matrices after renderer authoring."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    transforms = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            continue
        cache.Clear()
        matrix = cache.GetLocalToWorldTransform(prim)
        transforms[path] = [
            [float(matrix[row][column]) for column in range(4)] for row in range(4)
        ]
    return transforms


def _vec3(value: object | None) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    return tuple(float(value[index]) for index in range(3))


def snapshot_dynamic_state(
    stage: object,
    paths: Iterable[str],
    owned_paths: Iterable[str] = (),
) -> dict[str, object]:
    """Read pose, velocity, kinematic state, and renderer ownership."""
    from simulator.physical_world import dynamic_render_replay

    normalized_paths = tuple(sorted({str(path).rstrip("/") for path in paths if path}))
    owned = {str(path).rstrip("/") for path in owned_paths if path}
    states = dynamic_render_replay.snapshot_dynamic_prims(stage, normalized_paths)
    return {
        path: dynamic_prim_snapshot(
            position=pose[0],
            orientation=pose[1],
            linear_velocity=_vec3(velocity),
            angular_velocity=_vec3(angular_velocity),
            kinematic=bool(was_kinematic),
            owned=path in owned,
        )
        for path, (
            was_kinematic,
            pose,
            velocity,
            angular_velocity,
        ) in states.items()
    }


def _diagnostic_value(value: object) -> object:
    return value.value if hasattr(value, "value") else value


def _readback_metadata(readback: object) -> dict[str, object]:
    diagnostics = readback.diagnostics
    return {
        "quality": _diagnostic_value(diagnostics.quality),
        "mean": diagnostics.mean,
        "max": diagnostics.max,
        "reason": diagnostics.reason,
    }


class RenderParityTraceSession:
    """Own one complete dense renderer trace and its RGB sidecars."""

    def __init__(
        self,
        destination: str | os.PathLike[str],
        frames: Iterable[object],
        *,
        renderer: str,
        start_index: int,
        command_spans: Iterable[object] | None = None,
    ) -> None:
        if int(start_index) != 0:
            raise TraceFormatError(
                f"renderer parity tracing must start at global frame 0, got {start_index}"
            )
        materialized = tuple(frames)
        if command_spans is None:
            raise TraceFormatError(
                "offline temporal-plan parity requires authoritative command spans"
            )
        identity = RenderInputIdentity.from_frames(materialized, command_spans)
        self._path = Path(destination)
        self._writer = RenderTraceWriter(self._path, identity)
        self._renderer = str(renderer)
        self._rgb_dir = self._path.parent / f"{self._path.name}.rgb"

    def _event_kwargs(
        self, snapshot: RenderStateSnapshot, clock: ClockDelta
    ) -> dict[str, object]:
        if snapshot.metadata.get("renderer") != self._renderer:
            raise TraceFormatError("renderer trace snapshot identity changed")
        return {
            "poses": snapshot.poses,
            "intrinsics": snapshot.intrinsics,
            "static_state": snapshot.static_state,
            "dynamic_state": snapshot.dynamic_state,
            "clock": clock,
            "metadata": snapshot.metadata,
        }

    def authored_pre_step(
        self, global_index: int, frame: object, snapshot: RenderStateSnapshot
    ) -> TraceEvent:
        return self._writer.authored_pre_step(
            global_index,
            frame,
            **self._event_kwargs(snapshot, ClockDelta()),
        )

    def post_positive_step(
        self,
        global_index: int,
        frame: object,
        snapshot: RenderStateSnapshot,
        *,
        zero_dt: int = 0,
    ) -> TraceEvent:
        return self._writer.post_positive_step(
            global_index,
            frame,
            **self._event_kwargs(
                snapshot,
                ClockDelta(positive_dt=1, zero_dt=int(zero_dt)),
            ),
        )

    def _save_rgb(self, global_index: int, rgb: object) -> Path:
        self._rgb_dir.mkdir(parents=True, exist_ok=True)
        path = self._rgb_dir / f"frame_{global_index:05d}.npy"
        np.save(path, np.asarray(rgb), allow_pickle=False)
        return path.resolve()

    def delivered_readback(
        self,
        global_index: int,
        frame: object,
        snapshot: RenderStateSnapshot,
        *,
        readback: object,
        rgb: object,
        zero_dt: int = 0,
    ) -> TraceEvent:
        retries = int(readback.diagnostics.retries)
        quality = {
            "valid": _diagnostic_value(readback.diagnostics.quality) == "valid",
            "fallback": False,
            "retries": retries,
            "classification": _diagnostic_value(readback.diagnostics.quality),
        }
        metadata = dict(snapshot.metadata)
        metadata["readback"] = _readback_metadata(readback)
        path = self._save_rgb(global_index, rgb)
        kwargs = self._event_kwargs(
            dataclasses.replace(snapshot, metadata=metadata),
            ClockDelta(zero_dt=int(zero_dt), readbacks=retries + 1),
        )
        return self._writer.delivered_readback(
            global_index,
            frame,
            quality=quality,
            rgb=rgb_snapshot(rgb, path=path),
            **kwargs,
        )

    def finish(self) -> None:
        self._writer.finish()

    def abort(self) -> None:
        self._writer.abort()

    def __enter__(self) -> RenderParityTraceSession:
        return self

    def __exit__(self, exc_type, _exc_value, _traceback) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()


@dataclasses.dataclass(frozen=True)
class _BufferedTraceEvent:
    phase: str
    global_index: int
    frame: object
    frame_hash: str
    snapshot: RenderStateSnapshot
    clock: ClockDelta
    quality: dict[str, object] | None = None
    rgb: dict[str, object] | None = None


class OnlineTraceStatus(str, enum.Enum):
    ACTIVE = "active"
    ABORTED = "aborted"
    FINISHED = "finished"


class OnlineEpisodeTraceSession:
    """Buffer consecutive command segments until the full episode identity is known."""

    def __init__(
        self,
        destination: str | os.PathLike[str],
        *,
        renderer: str,
        start_index: int,
    ) -> None:
        self._path = Path(destination)
        self._renderer = str(renderer)
        self._rgb_dir = self._path.parent / f"{self._path.name}.rgb"
        self._events: list[_BufferedTraceEvent] = []
        self._segment_starts: list[int] = []
        self._aborted = False
        self._finished = False
        self.begin_segment(start_index)

    @property
    def next_global_index(self) -> int:
        return len(self._events) // 3

    @property
    def status(self) -> OnlineTraceStatus:
        if self._finished:
            return OnlineTraceStatus.FINISHED
        if self._aborted:
            return OnlineTraceStatus.ABORTED
        return OnlineTraceStatus.ACTIVE

    def begin_segment(self, start_index: int) -> None:
        if self._aborted or self._finished:
            raise TraceFormatError("cannot append to a closed online parity trace")
        if len(self._events) % 3:
            raise TraceFormatError(
                "previous online trace frame has an incomplete lifecycle"
            )
        expected = self.next_global_index
        if int(start_index) != expected:
            raise TraceFormatError(
                f"online parity trace expected global frame {expected}, got {start_index}"
            )
        self._segment_starts.append(int(start_index))

    def _append(
        self,
        phase: str,
        global_index: int,
        frame: object,
        snapshot: RenderStateSnapshot,
        clock: ClockDelta,
        *,
        quality: dict[str, object] | None = None,
        rgb: dict[str, object] | None = None,
    ) -> None:
        if self._aborted or self._finished:
            raise TraceFormatError("cannot append to a closed online parity trace")
        if snapshot.metadata.get("renderer") != self._renderer:
            raise TraceFormatError("online trace snapshot renderer identity changed")
        expected_index, phase_offset = divmod(len(self._events), 3)
        expected_phase = (
            AUTHORED_PRE_STEP,
            POST_POSITIVE_STEP,
            DELIVERED_READBACK,
        )[phase_offset]
        if global_index != expected_index or phase != expected_phase:
            raise TraceFormatError(
                f"expected {expected_phase} for global frame {expected_index}, "
                f"got {phase} for {global_index}"
            )
        self._events.append(
            _BufferedTraceEvent(
                phase,
                global_index,
                frame,
                stable_trace_hash(frame),
                snapshot,
                clock,
                quality,
                rgb,
            )
        )

    def authored_pre_step(
        self, global_index: int, frame: object, snapshot: RenderStateSnapshot
    ) -> None:
        self._append(
            AUTHORED_PRE_STEP,
            global_index,
            frame,
            snapshot,
            ClockDelta(),
        )

    def post_positive_step(
        self,
        global_index: int,
        frame: object,
        snapshot: RenderStateSnapshot,
        *,
        zero_dt: int = 0,
    ) -> None:
        self._append(
            POST_POSITIVE_STEP,
            global_index,
            frame,
            snapshot,
            ClockDelta(positive_dt=1, zero_dt=int(zero_dt)),
        )

    def _save_rgb(self, global_index: int, rgb: object) -> Path:
        self._rgb_dir.mkdir(parents=True, exist_ok=True)
        path = self._rgb_dir / f"frame_{global_index:05d}.npy"
        np.save(path, np.asarray(rgb), allow_pickle=False)
        return path.resolve()

    def delivered_readback(
        self,
        global_index: int,
        frame: object,
        snapshot: RenderStateSnapshot,
        *,
        readback: object,
        rgb: object,
        zero_dt: int = 0,
    ) -> None:
        retries = int(readback.diagnostics.retries)
        quality = {
            "valid": _diagnostic_value(readback.diagnostics.quality) == "valid",
            "fallback": False,
            "retries": retries,
            "classification": _diagnostic_value(readback.diagnostics.quality),
        }
        metadata = dict(snapshot.metadata)
        metadata["readback"] = _readback_metadata(readback)
        self._append(
            DELIVERED_READBACK,
            global_index,
            frame,
            dataclasses.replace(snapshot, metadata=metadata),
            ClockDelta(zero_dt=int(zero_dt), readbacks=retries + 1),
            quality=quality,
            rgb=rgb_snapshot(rgb, path=self._save_rgb(global_index, rgb)),
        )

    def finish(self, frames: Iterable[object]) -> OnlineTraceStatus:
        if self._aborted:
            return OnlineTraceStatus.ABORTED
        if self._finished:
            return OnlineTraceStatus.FINISHED
        materialized = tuple(frames)
        frame_count = len(materialized)
        command_spans = tuple(
            (start, end)
            for start, end in zip(
                self._segment_starts,
                (*self._segment_starts[1:], frame_count),
            )
        )
        identity = RenderInputIdentity.from_frames(materialized, command_spans)
        expected = identity.frame_count * 3
        if len(self._events) != expected:
            raise TraceFormatError(
                f"online trace ended after {len(self._events)} events; expected {expected}"
            )
        writer = RenderTraceWriter(self._path, identity)
        try:
            for event in self._events:
                if stable_trace_hash(event.frame) != event.frame_hash:
                    raise TraceFormatError(
                        f"global frame {event.global_index} changed after it was rendered"
                    )
                writer.record(
                    event.phase,
                    event.global_index,
                    event.frame,
                    poses=event.snapshot.poses,
                    intrinsics=event.snapshot.intrinsics,
                    static_state=event.snapshot.static_state,
                    dynamic_state=event.snapshot.dynamic_state,
                    clock=event.clock,
                    quality=event.quality,
                    rgb=event.rgb,
                    metadata=event.snapshot.metadata,
                )
            writer.finish()
        except Exception:
            writer.abort()
            raise
        self._finished = True
        return OnlineTraceStatus.FINISHED

    def abort(self) -> OnlineTraceStatus:
        if not self._finished:
            self._aborted = True
        return self.status

    def __enter__(self) -> OnlineEpisodeTraceSession:
        return self

    def __exit__(self, exc_type, _exc_value, _traceback) -> None:
        if exc_type is not None:
            self.abort()


def maybe_create_trace_session(
    frames: Iterable[object],
    *,
    renderer: str,
    start_index: int,
    command_spans: Iterable[object] | None = None,
) -> RenderParityTraceSession | None:
    """Create a trace only when explicitly requested by the environment."""
    destination = os.environ.get(TRACE_PATH_ENV)
    if not destination:
        return None
    return RenderParityTraceSession(
        destination,
        frames,
        renderer=renderer,
        start_index=start_index,
        command_spans=command_spans,
    )


def maybe_create_online_episode_trace(
    *,
    renderer: str,
    start_index: int,
) -> OnlineEpisodeTraceSession | None:
    """Create the deferred online trace at the first global dense frame."""
    destination = os.environ.get(TRACE_PATH_ENV)
    if not destination:
        return None
    return OnlineEpisodeTraceSession(
        destination,
        renderer=renderer,
        start_index=start_index,
    )


def load_trace_rgb(event: TraceEvent) -> np.ndarray:
    """Load an RGB sidecar emitted by ``RenderParityTraceSession``."""
    if event.rgb is None or not event.rgb.get("path"):
        raise ValueError("trace event does not name an RGB sidecar")
    return np.load(event.rgb["path"], allow_pickle=False)


def compare_saved_trace_files(
    baseline_path: str | os.PathLike[str],
    candidate_path: str | os.PathLike[str],
    *,
    config: ParityConfig | None = None,
) -> ParityReport:
    """Compare two real renderer traces, loading their saved RGB sidecars."""
    return compare_trace_files(
        baseline_path,
        candidate_path,
        config=config,
        rgb_loader=load_trace_rgb,
    )

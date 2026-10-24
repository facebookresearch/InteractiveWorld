# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic traces and comparison gates for renderer parity."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

import numpy as np


SCHEMA_VERSION = 2
AUTHORED_PRE_STEP = "authored_pre_step"
POST_POSITIVE_STEP = "post_positive_step"
DELIVERED_READBACK = "delivered_readback"
TRACE_PHASES = (AUTHORED_PRE_STEP, POST_POSITIVE_STEP, DELIVERED_READBACK)

_DYNAMIC_FIELDS = frozenset(
    {
        "position",
        "orientation",
        "linear_velocity",
        "angular_velocity",
        "kinematic",
        "owned",
    }
)
_POSE_FIELDS = frozenset({"position", "orientation"})


class TraceFormatError(ValueError):
    """Raised when a parity trace is malformed or internally inconsistent."""


def _normalized_mapping_key(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def _normalize_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("trace values must not contain NaN or infinity")
    return 0.0 if result == 0.0 else result


def _normalize_mapping(value: Mapping[object, object]) -> dict[str, object]:
    normalized = {}
    for raw_key, raw_value in value.items():
        key = _normalized_mapping_key(raw_key)
        if key in normalized:
            raise ValueError(f"mapping keys collide after normalization: {key!r}")
        normalized[key] = normalize_trace_value(raw_value)
    return {key: normalized[key] for key in sorted(normalized)}


def _normalize_object(value: object) -> dict[str, object]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: normalize_trace_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if hasattr(value, "_asdict"):
        return _normalize_mapping(value._asdict())
    try:
        attributes = vars(value)
    except TypeError as error:
        raise TypeError(
            f"unsupported trace value type: {type(value).__qualname__}"
        ) from error
    public = {
        key: attribute
        for key, attribute in attributes.items()
        if not key.startswith("_") and not callable(attribute)
    }
    if not public:
        raise TypeError(
            f"trace object has no serializable fields: {type(value).__qualname__}"
        )
    return _normalize_mapping(public)


def normalize_trace_value(value: object) -> object:
    """Convert Frame-like or numeric state into deterministic JSON data."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, enum.Enum):
        return normalize_trace_value(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        return normalize_trace_value(value.tolist())
    if isinstance(value, np.generic):
        return normalize_trace_value(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _normalize_float(value)
    if isinstance(value, bytes):
        return {
            "sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }
    if isinstance(value, Mapping):
        return _normalize_mapping(value)
    if isinstance(value, (set, frozenset)):
        normalized = [normalize_trace_value(item) for item in value]
        return sorted(normalized, key=canonical_trace_json)
    if isinstance(value, Sequence):
        return [normalize_trace_value(item) for item in value]
    return _normalize_object(value)


def canonical_trace_json(value: object) -> str:
    """Serialize a value using the canonical representation used by trace hashes."""
    return json.dumps(
        normalize_trace_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_trace_hash(value: object) -> str:
    """Return the SHA-256 identity of a normalized Frame or state snapshot."""
    return hashlib.sha256(canonical_trace_json(value).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class TrajectoryIdentity:
    """Content identity for the exact dense Frame artifact consumed by a renderer."""

    trajectory_hash: str
    frame_hashes: tuple[str, ...]

    @classmethod
    def from_frames(cls, frames: Iterable[object]) -> TrajectoryIdentity:
        materialized = tuple(frames)
        if not materialized:
            raise ValueError("a parity trajectory must contain at least one frame")
        return cls(
            trajectory_hash=stable_trace_hash(materialized),
            frame_hashes=tuple(stable_trace_hash(frame) for frame in materialized),
        )

    @property
    def frame_count(self) -> int:
        return len(self.frame_hashes)

    def verify_frame(self, global_index: int, frame: object) -> str:
        if global_index < 0 or global_index >= self.frame_count:
            raise IndexError(
                f"global frame {global_index} is outside trajectory of "
                f"{self.frame_count} frames"
            )
        frame_hash = stable_trace_hash(frame)
        expected = self.frame_hashes[global_index]
        if frame_hash != expected:
            raise TraceFormatError(
                f"global frame {global_index} does not match the trajectory artifact"
            )
        return frame_hash


def _command_span_pair(value: object, index: int) -> tuple[int, int]:
    if isinstance(value, Mapping):
        if set(value) != {"start", "end"}:
            raise ValueError(f"command span {index} must contain exactly start and end")
        raw_start, raw_end = value["start"], value["end"]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"command span {index} must contain start and end")
        raw_start, raw_end = value
    else:
        raise ValueError(f"command span {index} must be a pair or mapping")
    if (
        isinstance(raw_start, bool)
        or not isinstance(raw_start, int)
        or isinstance(raw_end, bool)
        or not isinstance(raw_end, int)
    ):
        raise ValueError(f"command span {index} bounds must be integers")
    return int(raw_start), int(raw_end)


def normalize_command_spans(
    command_spans: Iterable[object], frame_count: int
) -> tuple[tuple[int, int], ...]:
    """Validate a contiguous half-open command partition of the dense frames."""

    spans = tuple(
        _command_span_pair(value, index) for index, value in enumerate(command_spans)
    )
    if not spans:
        raise ValueError("authoritative command spans must not be empty")
    expected_start = 0
    for index, (start, end) in enumerate(spans):
        if start != expected_start or end < start or end > frame_count:
            raise ValueError(
                f"command span {index} must be contiguous and within the trajectory"
            )
        expected_start = end
    if expected_start != frame_count:
        raise ValueError(
            f"command spans end at {expected_start}, expected frame count {frame_count}"
        )
    return spans


def command_spans_hash(command_spans: Sequence[tuple[int, int]]) -> str:
    """Hash normalized command boundaries exactly as trajectory artifacts do."""

    return stable_trace_hash([[start, end] for start, end in command_spans])


def render_input_hash(
    identity: TrajectoryIdentity, authoritative_command_spans_hash: str
) -> str:
    """Hash the dense trajectory and command partition consumed by a renderer."""

    if not _is_sha256(authoritative_command_spans_hash):
        raise ValueError("command_spans_hash must be a lowercase SHA-256 digest")
    return _render_input_hash_values(
        identity.trajectory_hash, authoritative_command_spans_hash
    )


def _render_input_hash_values(
    trajectory_hash: str, authoritative_command_spans_hash: str
) -> str:
    return stable_trace_hash(
        {
            "command_spans_hash": authoritative_command_spans_hash,
            "trajectory_hash": trajectory_hash,
        }
    )


@dataclasses.dataclass(frozen=True)
class RenderInputIdentity:
    """Trajectory plus authoritative command partition used by temporal planning."""

    trajectory: TrajectoryIdentity
    command_spans: tuple[tuple[int, int], ...]
    command_spans_hash: str
    render_input_hash: str

    def __post_init__(self) -> None:
        normalized = normalize_command_spans(
            self.command_spans, self.trajectory.frame_count
        )
        if normalized != self.command_spans:
            raise ValueError("command spans must use their normalized representation")
        expected_spans_hash = command_spans_hash(normalized)
        if self.command_spans_hash != expected_spans_hash:
            raise ValueError("command spans do not match their identity hash")
        expected_render_hash = render_input_hash(self.trajectory, expected_spans_hash)
        if self.render_input_hash != expected_render_hash:
            raise ValueError("render input does not match its identity hash")

    @classmethod
    def from_frames(
        cls,
        frames: Iterable[object],
        command_spans: Iterable[object],
    ) -> RenderInputIdentity:
        materialized = tuple(frames)
        trajectory = TrajectoryIdentity.from_frames(materialized)
        spans = normalize_command_spans(command_spans, trajectory.frame_count)
        authoritative_spans_hash = command_spans_hash(spans)
        authoritative_render_input_hash = render_input_hash(
            trajectory, authoritative_spans_hash
        )
        return cls(
            trajectory,
            spans,
            authoritative_spans_hash,
            authoritative_render_input_hash,
        )

    @property
    def trajectory_hash(self) -> str:
        return self.trajectory.trajectory_hash

    @property
    def frame_hashes(self) -> tuple[str, ...]:
        return self.trajectory.frame_hashes

    @property
    def frame_count(self) -> int:
        return self.trajectory.frame_count

    def verify_frame(self, global_index: int, frame: object) -> str:
        return self.trajectory.verify_frame(global_index, frame)


@dataclasses.dataclass(frozen=True)
class ClockDelta:
    """Simulator work performed between this event and its preceding trace event."""

    positive_dt: int = 0
    zero_dt: int = 0
    readbacks: int = 0

    def to_dict(self) -> dict[str, int]:
        values = dataclasses.asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"clock {name} must be a non-negative integer")
        return values


def dynamic_prim_snapshot(
    *,
    position: object,
    orientation: object,
    linear_velocity: object,
    angular_velocity: object,
    kinematic: bool,
    owned: bool,
    **extra: object,
) -> dict[str, object]:
    """Build the canonical state for one potentially PhysX-owned prim."""
    if not isinstance(kinematic, bool) or not isinstance(owned, bool):
        raise ValueError("dynamic ownership fields must be booleans")
    snapshot = {
        "position": _numeric_vector(position, 3, "position"),
        "orientation": _quaternion(orientation),
        "linear_velocity": _numeric_vector(linear_velocity, 3, "linear_velocity"),
        "angular_velocity": _numeric_vector(angular_velocity, 3, "angular_velocity"),
        "kinematic": kinematic,
        "owned": owned,
    }
    snapshot.update(_normalize_mapping(extra))
    return snapshot


def pose_snapshot(
    *, position: object, orientation: object, **extra: object
) -> dict[str, object]:
    """Build a camera or rigid-pose snapshot with a sign-invariant quaternion."""
    snapshot = {
        "position": _numeric_vector(position, 3, "position"),
        "orientation": _quaternion(orientation),
    }
    snapshot.update(_normalize_mapping(extra))
    return snapshot


def _numeric_vector(value: object, size: int, name: str) -> list[float]:
    normalized = normalize_trace_value(value)
    if not isinstance(normalized, list) or len(normalized) != size:
        raise ValueError(f"{name} must be a numeric vector of length {size}")
    vector = []
    for component in normalized:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f"{name} must contain only finite numbers")
        vector.append(_normalize_float(float(component)))
    return vector


def _quaternion(value: object) -> list[float]:
    quaternion = _numeric_vector(value, 4, "orientation")
    if math.sqrt(sum(component * component for component in quaternion)) <= 1e-12:
        raise ValueError("orientation quaternion must have non-zero norm")
    return quaternion


def rgb_snapshot(
    pixels: object,
    *,
    path: str | os.PathLike[str] | None = None,
    channel_order: str = "RGB",
) -> dict[str, object]:
    """Hash an RGB readback without embedding its pixels in the JSONL trace."""
    if not isinstance(channel_order, str) or not channel_order:
        raise ValueError("RGB channel order must be a non-empty string")
    array = np.asarray(pixels)
    if array.ndim != 3 or array.size == 0 or array.shape[-1] != len(channel_order):
        raise ValueError("RGB readback must be a non-empty HWC array")
    is_integer = np.issubdtype(array.dtype, np.integer)
    is_floating = np.issubdtype(array.dtype, np.floating)
    if not is_integer and not is_floating:
        raise ValueError("RGB readback must have an integer or floating dtype")
    if is_floating and not bool(np.all(np.isfinite(array))):
        raise ValueError("RGB readback must contain only finite values")
    canonical = np.ascontiguousarray(array)
    header = canonical_trace_json(
        {
            "channel_order": channel_order,
            "dtype": canonical.dtype.name,
            "shape": canonical.shape,
        }
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\0" + canonical.tobytes()).hexdigest()
    result: dict[str, object] = {
        "channel_order": str(channel_order),
        "dtype": canonical.dtype.name,
        "shape": list(canonical.shape),
        "sha256": digest,
    }
    if path is not None:
        result["path"] = os.fspath(path)
    return result


def _normalize_rgb_snapshot(value: Mapping[str, object]) -> dict[str, object]:
    normalized = _normalize_mapping(value)
    required = {"channel_order", "dtype", "shape", "sha256"}
    if not required.issubset(normalized):
        raise ValueError(
            f"RGB snapshot is missing fields: {sorted(required - set(normalized))}"
        )
    if (
        not isinstance(normalized["channel_order"], str)
        or not normalized["channel_order"]
        or not isinstance(normalized["dtype"], str)
        or not normalized["dtype"]
    ):
        raise ValueError("RGB channel order and dtype must be non-empty strings")
    shape = normalized["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in shape
        )
    ):
        raise ValueError("RGB shape must contain three positive integer dimensions")
    if shape[-1] != len(normalized["channel_order"]):
        raise ValueError("RGB channel order does not match the channel dimension")
    if not isinstance(normalized["sha256"], str) or not _is_sha256(
        normalized["sha256"]
    ):
        raise ValueError("RGB sha256 is not a SHA-256 digest")
    if "path" in normalized and not isinstance(normalized["path"], str):
        raise ValueError("RGB path must be a string")
    return normalized


def _normalize_clock(clock: ClockDelta | Mapping[str, object]) -> dict[str, int]:
    raw = clock.to_dict() if isinstance(clock, ClockDelta) else dict(clock)
    if set(raw) != {"positive_dt", "zero_dt", "readbacks"}:
        raise ValueError(
            "clock must contain exactly positive_dt, zero_dt, and readbacks"
        )
    return ClockDelta(
        positive_dt=raw["positive_dt"],
        zero_dt=raw["zero_dt"],
        readbacks=raw["readbacks"],
    ).to_dict()


def _normalize_dynamic_state(value: Mapping[str, object]) -> dict[str, object]:
    normalized = _normalize_mapping(value)
    for path, state in normalized.items():
        if not isinstance(state, dict) or not _DYNAMIC_FIELDS.issubset(state):
            missing = sorted(
                _DYNAMIC_FIELDS - set(state if isinstance(state, dict) else ())
            )
            raise ValueError(f"dynamic state {path!r} is missing fields: {missing}")
        state["position"] = _numeric_vector(state["position"], 3, "position")
        state["orientation"] = _quaternion(state["orientation"])
        state["linear_velocity"] = _numeric_vector(
            state["linear_velocity"], 3, "linear_velocity"
        )
        state["angular_velocity"] = _numeric_vector(
            state["angular_velocity"], 3, "angular_velocity"
        )
        if not isinstance(state["kinematic"], bool) or not isinstance(
            state["owned"], bool
        ):
            raise ValueError("dynamic ownership fields must be booleans")
    return normalized


def _normalize_pose_state(value: Mapping[str, object]) -> dict[str, object]:
    normalized = _normalize_mapping(value)
    for name, state in normalized.items():
        if not isinstance(state, dict) or not _POSE_FIELDS.issubset(state):
            missing = sorted(
                _POSE_FIELDS - set(state if isinstance(state, dict) else ())
            )
            raise ValueError(f"pose {name!r} is missing fields: {missing}")
        state["position"] = _numeric_vector(state["position"], 3, "position")
        state["orientation"] = _quaternion(state["orientation"])
    return normalized


def _rgb_state_identity(value: Mapping[str, object] | None) -> object:
    if value is None:
        return None
    return {key: item for key, item in value.items() if key != "path"}


def _event_state_hash(
    *,
    poses: Mapping[str, object],
    intrinsics: Mapping[str, object],
    static_state: object,
    dynamic_state: Mapping[str, object],
    clock: Mapping[str, object],
    quality: Mapping[str, object],
    rgb: Mapping[str, object] | None,
) -> str:
    return stable_trace_hash(
        {
            "poses": poses,
            "intrinsics": intrinsics,
            "static_state": static_state,
            "dynamic_state": dynamic_state,
            "clock": clock,
            "quality": quality,
            "rgb": _rgb_state_identity(rgb),
        }
    )


@dataclasses.dataclass(frozen=True)
class _EventState:
    poses: dict[str, object]
    intrinsics: dict[str, object]
    static_state: object
    dynamic_state: dict[str, object]
    clock: dict[str, int]
    quality: dict[str, object]
    rgb: dict[str, object] | None
    state_hash: str


def _normalize_event_state(
    *,
    poses: Mapping[str, object],
    intrinsics: Mapping[str, object],
    static_state: object,
    dynamic_state: Mapping[str, object] | None,
    clock: ClockDelta | Mapping[str, object],
    quality: Mapping[str, object] | None,
    rgb: Mapping[str, object] | None,
) -> _EventState:
    normalized_poses = _normalize_pose_state(poses)
    normalized_intrinsics = _normalize_mapping(intrinsics)
    if not normalized_poses or not normalized_intrinsics:
        raise ValueError("every trace event requires camera poses and intrinsics")
    normalized_static = normalize_trace_value(static_state)
    normalized_dynamic = _normalize_dynamic_state(dynamic_state or {})
    normalized_clock = _normalize_clock(clock)
    normalized_quality = _normalize_mapping(quality or {})
    normalized_rgb = _normalize_rgb_snapshot(rgb) if rgb is not None else None
    state_hash = _event_state_hash(
        poses=normalized_poses,
        intrinsics=normalized_intrinsics,
        static_state=normalized_static,
        dynamic_state=normalized_dynamic,
        clock=normalized_clock,
        quality=normalized_quality,
        rgb=normalized_rgb,
    )
    return _EventState(
        normalized_poses,
        normalized_intrinsics,
        normalized_static,
        normalized_dynamic,
        normalized_clock,
        normalized_quality,
        normalized_rgb,
        state_hash,
    )


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    """One renderer observation at a required dense-frame lifecycle boundary."""

    phase: str
    global_index: int
    trajectory_hash: str
    trajectory_frame_count: int
    command_spans_hash: str
    render_input_hash: str
    frame_hash: str
    state_hash: str
    poses: dict[str, object]
    intrinsics: dict[str, object]
    static_state: object
    dynamic_state: dict[str, object]
    clock: dict[str, int]
    quality: dict[str, object]
    rgb: dict[str, object] | None
    metadata: dict[str, object]
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        identity: RenderInputIdentity,
        phase: str,
        global_index: int,
        frame: object,
        *,
        poses: Mapping[str, object],
        intrinsics: Mapping[str, object],
        static_state: object = None,
        dynamic_state: Mapping[str, object] | None = None,
        clock: ClockDelta | Mapping[str, object],
        quality: Mapping[str, object] | None = None,
        rgb: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TraceEvent:
        if not isinstance(identity, RenderInputIdentity):
            raise TraceFormatError(
                "temporal-plan parity requires authoritative command spans"
            )
        if phase not in TRACE_PHASES:
            raise ValueError(f"unknown trace phase: {phase!r}")
        state = _normalize_event_state(
            poses=poses,
            intrinsics=intrinsics,
            static_state=static_state,
            dynamic_state=dynamic_state,
            clock=clock,
            quality=quality,
            rgb=rgb,
        )
        return cls(
            phase=phase,
            global_index=int(global_index),
            trajectory_hash=identity.trajectory_hash,
            trajectory_frame_count=identity.frame_count,
            command_spans_hash=identity.command_spans_hash,
            render_input_hash=identity.render_input_hash,
            frame_hash=identity.verify_frame(global_index, frame),
            state_hash=state.state_hash,
            poses=state.poses,
            intrinsics=state.intrinsics,
            static_state=state.static_state,
            dynamic_state=state.dynamic_state,
            clock=state.clock,
            quality=state.quality,
            rgb=state.rgb,
            metadata=_normalize_mapping(metadata or {}),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TraceEvent:
        schema_version = value.get("schema_version")
        if schema_version == 1:
            raise TraceFormatError(
                "trace schema v1 has no authoritative command spans and cannot "
                "establish temporal-plan parity"
            )
        if schema_version != SCHEMA_VERSION:
            raise TraceFormatError(
                f"unsupported trace schema version: {schema_version!r}"
            )
        required = {field.name for field in dataclasses.fields(cls)}
        if set(value) != required:
            missing_identity = {
                "command_spans_hash",
                "render_input_hash",
            } - set(value)
            if missing_identity:
                raise TraceFormatError(
                    "trace lacks authoritative temporal render-input identity: "
                    f"{sorted(missing_identity)}"
                )
            raise TraceFormatError(
                f"trace event keys differ: missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        event = cls(
            phase=str(value["phase"]),
            global_index=_strict_int(value["global_index"], "global_index"),
            trajectory_hash=str(value["trajectory_hash"]),
            trajectory_frame_count=_strict_int(
                value["trajectory_frame_count"], "trajectory_frame_count"
            ),
            command_spans_hash=str(value["command_spans_hash"]),
            render_input_hash=str(value["render_input_hash"]),
            frame_hash=str(value["frame_hash"]),
            state_hash=str(value["state_hash"]),
            poses=_normalize_pose_state(_require_mapping(value["poses"], "poses")),
            intrinsics=_normalize_mapping(
                _require_mapping(value["intrinsics"], "intrinsics")
            ),
            static_state=normalize_trace_value(value["static_state"]),
            dynamic_state=_normalize_dynamic_state(
                _require_mapping(value["dynamic_state"], "dynamic_state")
            ),
            clock=_normalize_clock(_require_mapping(value["clock"], "clock")),
            quality=_normalize_mapping(_require_mapping(value["quality"], "quality")),
            rgb=(
                _normalize_rgb_snapshot(_require_mapping(value["rgb"], "rgb"))
                if value["rgb"] is not None
                else None
            ),
            metadata=_normalize_mapping(
                _require_mapping(value["metadata"], "metadata")
            ),
            schema_version=SCHEMA_VERSION,
        )
        event._validate_integrity()
        return event

    def _validate_integrity(self) -> None:
        if self.phase not in TRACE_PHASES:
            raise TraceFormatError(f"unknown trace phase: {self.phase!r}")
        if self.global_index < 0 or self.trajectory_frame_count < 1:
            raise TraceFormatError("trace frame indices and counts must be positive")
        if self.global_index >= self.trajectory_frame_count:
            raise TraceFormatError("trace global index exceeds trajectory frame count")
        if not self.poses or not self.intrinsics:
            raise TraceFormatError("trace event is missing camera poses or intrinsics")
        try:
            _normalize_clock(self.clock)
            _normalize_pose_state(self.poses)
            _normalize_dynamic_state(self.dynamic_state)
            if self.rgb is not None:
                _normalize_rgb_snapshot(self.rgb)
        except ValueError as error:
            raise TraceFormatError(str(error)) from error
        for name, digest in (
            ("trajectory_hash", self.trajectory_hash),
            ("command_spans_hash", self.command_spans_hash),
            ("render_input_hash", self.render_input_hash),
            ("frame_hash", self.frame_hash),
            ("state_hash", self.state_hash),
        ):
            if not _is_sha256(digest):
                raise TraceFormatError(f"{name} is not a SHA-256 digest")
        expected_render_input_hash = _render_input_hash_values(
            self.trajectory_hash, self.command_spans_hash
        )
        if self.render_input_hash != expected_render_input_hash:
            raise TraceFormatError(
                "render_input_hash does not match trajectory and command spans"
            )
        expected = _event_state_hash(
            poses=self.poses,
            intrinsics=self.intrinsics,
            static_state=self.static_state,
            dynamic_state=self.dynamic_state,
            clock=self.clock,
            quality=self.quality,
            rgb=self.rgb,
        )
        if self.state_hash != expected:
            raise TraceFormatError("trace event state hash does not match its payload")

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TraceFormatError(f"{name} must be a mapping")
    return value


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceFormatError(f"{name} must be an integer")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def event_json(event: TraceEvent) -> str:
    """Serialize one event as a canonical JSONL record."""
    event._validate_integrity()
    return canonical_trace_json(event.to_dict())


class RenderTraceWriter:
    """Write a complete, ordered trace for one exact dense trajectory."""

    def __init__(
        self,
        destination: str | os.PathLike[str] | TextIO,
        identity: RenderInputIdentity,
    ) -> None:
        if not isinstance(identity, RenderInputIdentity):
            raise TraceFormatError(
                "temporal-plan parity requires authoritative command spans"
            )
        self._identity = identity
        self._event_count = 0
        self._closed = False
        self._owns_stream = not hasattr(destination, "write")
        if self._owns_stream:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("w", encoding="utf-8")
        else:
            self._stream = destination

    def record(
        self,
        phase: str,
        global_index: int,
        frame: object,
        **snapshot: object,
    ) -> TraceEvent:
        if self._closed:
            raise TraceFormatError("cannot append to a closed parity trace")
        expected_index, expected_phase = divmod(self._event_count, len(TRACE_PHASES))
        if global_index != expected_index or phase != TRACE_PHASES[expected_phase]:
            raise TraceFormatError(
                f"expected {TRACE_PHASES[expected_phase]} for global frame "
                f"{expected_index}, got {phase} for {global_index}"
            )
        event = TraceEvent.create(
            self._identity,
            phase,
            global_index,
            frame,
            **snapshot,
        )
        self._stream.write(event_json(event) + "\n")
        self._stream.flush()
        self._event_count += 1
        return event

    def authored_pre_step(
        self, global_index: int, frame: object, **snapshot: object
    ) -> TraceEvent:
        return self.record(AUTHORED_PRE_STEP, global_index, frame, **snapshot)

    def post_positive_step(
        self, global_index: int, frame: object, **snapshot: object
    ) -> TraceEvent:
        return self.record(POST_POSITIVE_STEP, global_index, frame, **snapshot)

    def delivered_readback(
        self, global_index: int, frame: object, **snapshot: object
    ) -> TraceEvent:
        return self.record(DELIVERED_READBACK, global_index, frame, **snapshot)

    def finish(self) -> None:
        if self._closed:
            return
        expected = self._identity.frame_count * len(TRACE_PHASES)
        if self._event_count != expected:
            raise TraceFormatError(
                f"trace ended after {self._event_count} events; expected {expected}"
            )
        self._closed = True
        if self._owns_stream:
            self._stream.close()

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_stream:
                self._stream.close()

    def __enter__(self) -> RenderTraceWriter:
        return self

    def __exit__(self, exc_type, _exc_value, _traceback) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()


def read_trace(source: str | os.PathLike[str] | TextIO) -> tuple[TraceEvent, ...]:
    """Read and integrity-check a JSONL parity trace."""
    owns_stream = not hasattr(source, "read")
    stream = Path(source).open("r", encoding="utf-8") if owns_stream else source
    events = []
    try:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise TraceFormatError(f"blank JSONL record at line {line_number}")
            try:
                raw = json.loads(line)
                events.append(
                    TraceEvent.from_dict(_require_mapping(raw, "trace event"))
                )
            except (TypeError, ValueError) as error:
                raise TraceFormatError(
                    f"invalid trace event at line {line_number}: {error}"
                ) from error
    finally:
        if owns_stream:
            stream.close()
    if not events:
        raise TraceFormatError("parity trace is empty")
    return tuple(events)


@dataclasses.dataclass(frozen=True)
class ParityConfig:
    """Numerical gates for strict state and tolerant dynamic/RGB comparison."""

    pose_atol: float = 1e-6
    intrinsics_atol: float = 1e-6
    static_atol: float = 1e-6
    dynamic_position_atol: float = 1e-4
    dynamic_rotation_atol_rad: float = 1e-3
    dynamic_velocity_atol: float = 1e-3
    require_exact_rgb: bool = True
    min_rgb_psnr: float | None = None
    min_rgb_ssim: float | None = None
    rgb_data_range: float = 255.0
    max_issues: int = 100

    def __post_init__(self) -> None:
        tolerances = (
            self.pose_atol,
            self.intrinsics_atol,
            self.static_atol,
            self.dynamic_position_atol,
            self.dynamic_rotation_atol_rad,
            self.dynamic_velocity_atol,
        )
        if any(
            isinstance(value, bool) or value < 0 or not math.isfinite(value)
            for value in tolerances
        ):
            raise ValueError("parity tolerances must be finite and non-negative")
        if not isinstance(self.require_exact_rgb, bool):
            raise ValueError("require_exact_rgb must be a boolean")
        if (
            isinstance(self.rgb_data_range, bool)
            or self.rgb_data_range <= 0
            or not math.isfinite(self.rgb_data_range)
        ):
            raise ValueError("rgb_data_range must be finite and positive")
        if isinstance(self.max_issues, bool) or self.max_issues < 1:
            raise ValueError("max_issues must be positive")
        for name, value in (
            ("min_rgb_psnr", self.min_rgb_psnr),
            ("min_rgb_ssim", self.min_rgb_ssim),
        ):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite when configured")
        if self.min_rgb_ssim is not None and not -1.0 <= self.min_rgb_ssim <= 1.0:
            raise ValueError("min_rgb_ssim must be between -1 and 1")
        if not self.require_exact_rgb and (
            self.min_rgb_psnr is None and self.min_rgb_ssim is None
        ):
            raise ValueError("tolerant RGB comparison requires a PSNR or SSIM gate")


@dataclasses.dataclass(frozen=True)
class ParityIssue:
    code: str
    message: str
    global_index: int | None = None
    phase: str | None = None
    field: str | None = None
    baseline: object = None
    candidate: object = None

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: normalize_trace_value(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


@dataclasses.dataclass(frozen=True)
class RgbMetrics:
    """Exact identity or channel-averaged global SSIM/PSNR for one readback."""

    global_index: int
    phase: str
    exact: bool
    psnr: float | None = None
    ssim: float | None = None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ParityReport:
    passed: bool
    trajectory_hash: str | None
    command_spans_hash: str | None
    render_input_hash: str | None
    frame_count: int
    compared_events: int
    issues: tuple[ParityIssue, ...]
    rgb_metrics: tuple[RgbMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "trajectory_hash": self.trajectory_hash,
            "command_spans_hash": self.command_spans_hash,
            "render_input_hash": self.render_input_hash,
            "frame_count": self.frame_count,
            "compared_events": self.compared_events,
            "issues": [issue.to_dict() for issue in self.issues],
            "rgb_metrics": [metric.to_dict() for metric in self.rgb_metrics],
        }


class _ReportBuilder:
    def __init__(self, config: ParityConfig) -> None:
        self.config = config
        self.issues: list[ParityIssue] = []
        self.rgb_metrics: list[RgbMetrics] = []

    def add(self, code: str, message: str, **context: object) -> None:
        if len(self.issues) < self.config.max_issues:
            self.issues.append(ParityIssue(code=code, message=message, **context))


def _event_key(event: TraceEvent) -> tuple[int, str]:
    return (event.global_index, event.phase)


def _integrity_error(event: TraceEvent) -> str | None:
    try:
        event._validate_integrity()
    except TraceFormatError as error:
        return str(error)
    return None


def _validate_clock(event: TraceEvent, report: _ReportBuilder, side: str) -> None:
    positive_dt = event.clock["positive_dt"]
    readbacks = event.clock["readbacks"]
    if event.phase == AUTHORED_PRE_STEP and (positive_dt != 0 or readbacks != 0):
        report.add(
            f"{side}.clock_phase",
            "authored_pre_step must precede positive time and readback work",
            global_index=event.global_index,
            phase=event.phase,
            field="clock",
            candidate=event.clock,
        )
    if event.phase == POST_POSITIVE_STEP and (positive_dt != 1 or readbacks != 0):
        report.add(
            f"{side}.clock_phase",
            "post_positive_step must contain exactly one positive-dt step",
            global_index=event.global_index,
            phase=event.phase,
            field="clock",
            candidate=event.clock,
        )
    if event.phase == DELIVERED_READBACK and (positive_dt != 0 or readbacks < 1):
        report.add(
            f"{side}.clock_phase",
            "delivered_readback must contain at least one readback and no positive step",
            global_index=event.global_index,
            phase=event.phase,
            field="clock",
            candidate=event.clock,
        )


def _validate_quality(event: TraceEvent, report: _ReportBuilder, side: str) -> None:
    if event.phase != DELIVERED_READBACK:
        if event.quality or event.rgb is not None:
            report.add(
                f"{side}.premature_delivery",
                "quality and RGB are only valid on delivered_readback",
                global_index=event.global_index,
                phase=event.phase,
            )
        return
    _validate_delivery_quality_fields(event, report, side)
    _validate_readback_accounting(event, report, side)
    if event.rgb is None:
        report.add(
            f"{side}.missing_rgb",
            "delivered_readback is missing an RGB snapshot",
            global_index=event.global_index,
            phase=event.phase,
            field="rgb",
        )


def _validate_delivery_quality_fields(
    event: TraceEvent, report: _ReportBuilder, side: str
) -> None:
    if not isinstance(event.quality.get("valid"), bool) or not isinstance(
        event.quality.get("fallback"), bool
    ):
        report.add(
            f"{side}.quality_schema",
            "delivered quality requires boolean valid and fallback fields",
            global_index=event.global_index,
            phase=event.phase,
            field="quality",
            candidate=event.quality,
        )
    elif not event.quality["valid"] or event.quality["fallback"]:
        report.add(
            f"{side}.quality_gate",
            "delivered frame must be valid and must not be a fallback",
            global_index=event.global_index,
            phase=event.phase,
            field="quality",
            candidate=event.quality,
        )


def _validate_readback_accounting(
    event: TraceEvent, report: _ReportBuilder, side: str
) -> None:
    retries = event.quality.get("retries")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        report.add(
            f"{side}.quality_schema",
            "delivered quality requires a non-negative retry count",
            global_index=event.global_index,
            phase=event.phase,
            field="quality.retries",
            candidate=retries,
        )
    elif event.clock["readbacks"] != retries + 1:
        report.add(
            f"{side}.readback_accounting",
            "readback attempts must equal one initial read plus retries",
            global_index=event.global_index,
            phase=event.phase,
            field="clock.readbacks",
            baseline=retries + 1,
            candidate=event.clock["readbacks"],
        )


def _validate_trace(
    events: Sequence[TraceEvent], report: _ReportBuilder, side: str
) -> None:
    if not events:
        report.add(f"{side}.empty", "trace is empty")
        return
    first = events[0]
    expected_count = first.trajectory_frame_count * len(TRACE_PHASES)
    if len(events) != expected_count:
        report.add(
            f"{side}.event_count",
            "trace does not contain all required lifecycle events",
            baseline=expected_count,
            candidate=len(events),
        )
    for ordinal, event in enumerate(events):
        expected_index, phase_offset = divmod(ordinal, len(TRACE_PHASES))
        expected_phase = TRACE_PHASES[phase_offset]
        if _event_key(event) != (expected_index, expected_phase):
            report.add(
                f"{side}.event_order",
                "trace events are not contiguous lifecycle triples",
                global_index=event.global_index,
                phase=event.phase,
                baseline=[expected_index, expected_phase],
                candidate=[event.global_index, event.phase],
            )
        if (
            event.trajectory_hash != first.trajectory_hash
            or event.trajectory_frame_count != first.trajectory_frame_count
        ):
            report.add(
                f"{side}.trajectory_identity",
                "trace changes trajectory identity between events",
                global_index=event.global_index,
                phase=event.phase,
            )
        if (
            event.command_spans_hash != first.command_spans_hash
            or event.render_input_hash != first.render_input_hash
        ):
            report.add(
                f"{side}.render_input_identity",
                "trace changes authoritative render-input identity between events",
                global_index=event.global_index,
                phase=event.phase,
            )
        integrity_error = _integrity_error(event)
        if integrity_error is not None:
            report.add(
                f"{side}.integrity",
                integrity_error,
                global_index=event.global_index,
                phase=event.phase,
            )
            continue
        _validate_clock(event, report, side)
        _validate_quality(event, report, side)
    _validate_frame_hash_groups(events, report, side)


def _validate_frame_hash_groups(
    events: Sequence[TraceEvent], report: _ReportBuilder, side: str
) -> None:
    frame_hashes: dict[int, str] = {}
    for event in events:
        previous = frame_hashes.setdefault(event.global_index, event.frame_hash)
        if event.frame_hash != previous:
            report.add(
                f"{side}.frame_hash_changed",
                "lifecycle events for one dense frame use different Frame records",
                global_index=event.global_index,
                phase=event.phase,
                baseline=previous,
                candidate=event.frame_hash,
            )


def _tree_compare(
    baseline: object,
    candidate: object,
    *,
    atol: float,
    field: str,
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        _compare_mapping_tree(
            baseline, candidate, atol=atol, field=field, event=event, report=report
        )
        return
    if isinstance(baseline, list) and isinstance(candidate, list):
        _compare_sequence_tree(
            baseline, candidate, atol=atol, field=field, event=event, report=report
        )
        return
    if _is_number(baseline) and _is_number(candidate):
        _compare_number(
            baseline, candidate, atol=atol, field=field, event=event, report=report
        )
        return
    if baseline != candidate:
        report.add(
            "state.value",
            "state values differ",
            global_index=event.global_index,
            phase=event.phase,
            field=field,
            baseline=baseline,
            candidate=candidate,
        )


def _compare_mapping_tree(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    atol: float,
    field: str,
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    if set(baseline) != set(candidate):
        report.add(
            "state.keys",
            "state mappings have different keys",
            global_index=event.global_index,
            phase=event.phase,
            field=field,
            baseline=sorted(baseline),
            candidate=sorted(candidate),
        )
    for key in sorted(set(baseline) & set(candidate)):
        _tree_compare(
            baseline[key],
            candidate[key],
            atol=atol,
            field=f"{field}.{key}",
            event=event,
            report=report,
        )


def _compare_sequence_tree(
    baseline: list[object],
    candidate: list[object],
    *,
    atol: float,
    field: str,
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    if len(baseline) != len(candidate):
        report.add(
            "state.length",
            "state sequences have different lengths",
            global_index=event.global_index,
            phase=event.phase,
            field=field,
            baseline=len(baseline),
            candidate=len(candidate),
        )
    for index, (left_item, right_item) in enumerate(zip(baseline, candidate)):
        _tree_compare(
            left_item,
            right_item,
            atol=atol,
            field=f"{field}[{index}]",
            event=event,
            report=report,
        )


def _compare_number(
    baseline: int | float,
    candidate: int | float,
    *,
    atol: float,
    field: str,
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    if abs(float(baseline) - float(candidate)) > atol:
        report.add(
            "state.numeric",
            f"numeric state differs by more than {atol}",
            global_index=event.global_index,
            phase=event.phase,
            field=field,
            baseline=baseline,
            candidate=candidate,
        )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    dot = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return 2.0 * math.acos(min(1.0, max(-1.0, abs(dot))))


def _compare_poses(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    if set(baseline) != set(candidate):
        report.add(
            "pose.names",
            "named pose sets differ",
            global_index=event.global_index,
            phase=event.phase,
            field="poses",
            baseline=sorted(baseline),
            candidate=sorted(candidate),
        )
    for name in sorted(set(baseline) & set(candidate)):
        _compare_pose(baseline[name], candidate[name], name, event, report)


def _compare_pose(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    name: str,
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    tolerance = report.config.pose_atol
    _tree_compare(
        baseline["position"],
        candidate["position"],
        atol=tolerance,
        field=f"poses.{name}.position",
        event=event,
        report=report,
    )
    rotation = _quaternion_distance(baseline["orientation"], candidate["orientation"])
    if rotation > tolerance:
        report.add(
            "pose.rotation",
            "pose orientation exceeds the angular tolerance",
            global_index=event.global_index,
            phase=event.phase,
            field=f"poses.{name}.orientation",
            baseline=0.0,
            candidate=rotation,
        )
    extras = (set(baseline) | set(candidate)) - _POSE_FIELDS
    _tree_compare(
        {key: baseline.get(key) for key in extras},
        {key: candidate.get(key) for key in extras},
        atol=tolerance,
        field=f"poses.{name}.extra",
        event=event,
        report=report,
    )


def _compare_dynamic_prim(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    path: str,
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    config = report.config
    for field_name, tolerance in (
        ("position", config.dynamic_position_atol),
        ("linear_velocity", config.dynamic_velocity_atol),
        ("angular_velocity", config.dynamic_velocity_atol),
    ):
        _tree_compare(
            baseline[field_name],
            candidate[field_name],
            atol=tolerance,
            field=f"dynamic_state.{path}.{field_name}",
            event=event,
            report=report,
        )
    rotation = _quaternion_distance(baseline["orientation"], candidate["orientation"])
    if rotation > config.dynamic_rotation_atol_rad:
        report.add(
            "dynamic.rotation",
            "dynamic orientation exceeds the angular tolerance",
            global_index=event.global_index,
            phase=event.phase,
            field=f"dynamic_state.{path}.orientation",
            baseline=0.0,
            candidate=rotation,
        )
    _compare_dynamic_ownership(baseline, candidate, path, event, report)
    extras = (set(baseline) | set(candidate)) - _DYNAMIC_FIELDS
    _tree_compare(
        {key: baseline.get(key) for key in extras},
        {key: candidate.get(key) for key in extras},
        atol=config.static_atol,
        field=f"dynamic_state.{path}.extra",
        event=event,
        report=report,
    )


def _compare_dynamic_ownership(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    path: str,
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    for field_name in ("kinematic", "owned"):
        if baseline[field_name] != candidate[field_name]:
            report.add(
                "dynamic.ownership",
                "dynamic ownership state differs",
                global_index=event.global_index,
                phase=event.phase,
                field=f"dynamic_state.{path}.{field_name}",
                baseline=baseline[field_name],
                candidate=candidate[field_name],
            )


def _compare_dynamic(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    event: TraceEvent,
    report: _ReportBuilder,
) -> None:
    if set(baseline) != set(candidate):
        report.add(
            "dynamic.paths",
            "dynamic prim sets differ",
            global_index=event.global_index,
            phase=event.phase,
            field="dynamic_state",
            baseline=sorted(baseline),
            candidate=sorted(candidate),
        )
    for path in sorted(set(baseline) & set(candidate)):
        _compare_dynamic_prim(
            baseline[path],
            candidate[path],
            path=path,
            event=event,
            report=report,
        )


def _rgb_array_metrics(
    baseline: object, candidate: object, data_range: float
) -> tuple[float, float]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("loaded RGB arrays must have the same non-empty shape")
    if not bool(np.all(np.isfinite(left))) or not bool(np.all(np.isfinite(right))):
        raise ValueError("loaded RGB arrays must contain only finite values")
    mse = float(np.mean(np.square(left - right)))
    psnr = math.inf if mse == 0.0 else 20.0 * math.log10(data_range / math.sqrt(mse))
    samples_left = (
        left.reshape(-1, left.shape[-1]) if left.ndim == 3 else left.reshape(-1, 1)
    )
    samples_right = (
        right.reshape(-1, right.shape[-1]) if right.ndim == 3 else right.reshape(-1, 1)
    )
    ssim = float(
        np.mean(
            [
                _channel_ssim(
                    samples_left[:, index], samples_right[:, index], data_range
                )
                for index in range(samples_left.shape[1])
            ]
        )
    )
    return psnr, ssim


def _channel_ssim(left: np.ndarray, right: np.ndarray, data_range: float) -> float:
    mean_left = float(np.mean(left))
    mean_right = float(np.mean(right))
    variance_left = float(np.mean(np.square(left - mean_left)))
    variance_right = float(np.mean(np.square(right - mean_right)))
    covariance = float(np.mean((left - mean_left) * (right - mean_right)))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mean_left * mean_right + c1) * (2.0 * covariance + c2)
    denominator = (mean_left * mean_left + mean_right * mean_right + c1) * (
        variance_left + variance_right + c2
    )
    return numerator / denominator


def _compare_rgb(
    baseline: TraceEvent,
    candidate: TraceEvent,
    report: _ReportBuilder,
    rgb_loader: Callable[[TraceEvent], object] | None,
) -> None:
    if baseline.rgb is None or candidate.rgb is None:
        return
    descriptor_fields = ("shape", "dtype", "channel_order")
    if not all(
        baseline.rgb.get(field) == candidate.rgb.get(field)
        for field in descriptor_fields
    ):
        report.add(
            "rgb.descriptor",
            "RGB shape, dtype, or channel order differs",
            global_index=baseline.global_index,
            phase=baseline.phase,
            field="rgb",
            baseline={field: baseline.rgb.get(field) for field in descriptor_fields},
            candidate={field: candidate.rgb.get(field) for field in descriptor_fields},
        )
        return
    exact = baseline.rgb.get("sha256") == candidate.rgb.get("sha256")
    if exact:
        report.rgb_metrics.append(
            RgbMetrics(baseline.global_index, baseline.phase, exact=True)
        )
        return
    if report.config.require_exact_rgb:
        report.add(
            "rgb.hash",
            "RGB hashes differ under the exact pixel gate",
            global_index=baseline.global_index,
            phase=baseline.phase,
            field="rgb.sha256",
            baseline=baseline.rgb.get("sha256"),
            candidate=candidate.rgb.get("sha256"),
        )
        return
    _compare_tolerant_rgb(baseline, candidate, report, rgb_loader)


def _compare_tolerant_rgb(
    baseline: TraceEvent,
    candidate: TraceEvent,
    report: _ReportBuilder,
    rgb_loader: Callable[[TraceEvent], object] | None,
) -> None:
    if rgb_loader is None:
        report.add(
            "rgb.loader",
            "tolerant RGB comparison needs a pixel loader when hashes differ",
            global_index=baseline.global_index,
            phase=baseline.phase,
            field="rgb",
        )
        return
    metrics = _load_rgb_metrics(baseline, candidate, report, rgb_loader)
    if metrics is None:
        return
    psnr, ssim = metrics
    report.rgb_metrics.append(
        RgbMetrics(baseline.global_index, baseline.phase, False, psnr, ssim)
    )
    _apply_rgb_metric_gates(baseline, psnr, ssim, report)


def _load_rgb_metrics(
    baseline: TraceEvent,
    candidate: TraceEvent,
    report: _ReportBuilder,
    rgb_loader: Callable[[TraceEvent], object],
) -> tuple[float, float] | None:
    try:
        baseline_pixels = _load_rgb_pixels(baseline, rgb_loader)
        candidate_pixels = _load_rgb_pixels(candidate, rgb_loader)
        return _rgb_array_metrics(
            baseline_pixels,
            candidate_pixels,
            report.config.rgb_data_range,
        )
    except (TypeError, ValueError) as error:
        report.add(
            "rgb.load",
            f"could not compare RGB pixels: {error}",
            global_index=baseline.global_index,
            phase=baseline.phase,
            field="rgb",
        )
        return None


def _load_rgb_pixels(
    event: TraceEvent, rgb_loader: Callable[[TraceEvent], object]
) -> np.ndarray:
    if event.rgb is None:
        raise ValueError("trace event has no RGB descriptor")
    pixels = np.asarray(rgb_loader(event))
    if list(pixels.shape) != event.rgb["shape"]:
        raise ValueError("loaded RGB shape does not match its trace descriptor")
    if pixels.dtype.name != event.rgb["dtype"]:
        raise ValueError("loaded RGB dtype does not match its trace descriptor")
    observed = rgb_snapshot(
        pixels,
        channel_order=event.rgb["channel_order"],
    )
    if observed["sha256"] != event.rgb["sha256"]:
        raise ValueError("loaded RGB pixels do not match their trace hash")
    return pixels


def _apply_rgb_metric_gates(
    baseline: TraceEvent,
    psnr: float,
    ssim: float,
    report: _ReportBuilder,
) -> None:
    if report.config.min_rgb_psnr is not None and psnr < report.config.min_rgb_psnr:
        report.add(
            "rgb.psnr",
            "RGB PSNR is below the configured gate",
            global_index=baseline.global_index,
            phase=baseline.phase,
            field="rgb.psnr",
            baseline=report.config.min_rgb_psnr,
            candidate=psnr,
        )
    if report.config.min_rgb_ssim is not None and ssim < report.config.min_rgb_ssim:
        report.add(
            "rgb.ssim",
            "RGB SSIM is below the configured gate",
            global_index=baseline.global_index,
            phase=baseline.phase,
            field="rgb.ssim",
            baseline=report.config.min_rgb_ssim,
            candidate=ssim,
        )


def _compare_event(
    baseline: TraceEvent,
    candidate: TraceEvent,
    report: _ReportBuilder,
    rgb_loader: Callable[[TraceEvent], object] | None,
) -> None:
    if _integrity_error(baseline) or _integrity_error(candidate):
        return
    if baseline.frame_hash != candidate.frame_hash:
        report.add(
            "frame_hash",
            "renderers consumed different Frame records at this dense index",
            global_index=baseline.global_index,
            phase=baseline.phase,
            field="frame_hash",
            baseline=baseline.frame_hash,
            candidate=candidate.frame_hash,
        )
    _compare_poses(baseline.poses, candidate.poses, baseline, report)
    for field_name, tolerance in (
        ("intrinsics", report.config.intrinsics_atol),
        ("static_state", report.config.static_atol),
    ):
        _tree_compare(
            getattr(baseline, field_name),
            getattr(candidate, field_name),
            atol=tolerance,
            field=field_name,
            event=baseline,
            report=report,
        )
    _compare_dynamic(baseline.dynamic_state, candidate.dynamic_state, baseline, report)
    for field_name in ("clock", "quality"):
        _tree_compare(
            getattr(baseline, field_name),
            getattr(candidate, field_name),
            atol=0.0,
            field=field_name,
            event=baseline,
            report=report,
        )
    if baseline.phase == DELIVERED_READBACK:
        _compare_rgb(baseline, candidate, report, rgb_loader)


def _compare_trajectory_identity(
    baseline: Sequence[TraceEvent],
    candidate: Sequence[TraceEvent],
    report: _ReportBuilder,
) -> str | None:
    trajectory_hash = baseline[0].trajectory_hash if baseline else None
    if not baseline or not candidate:
        return trajectory_hash
    same_identity = (
        baseline[0].trajectory_hash == candidate[0].trajectory_hash
        and baseline[0].trajectory_frame_count == candidate[0].trajectory_frame_count
    )
    if same_identity:
        return trajectory_hash
    report.add(
        "trajectory_identity",
        "renderers did not consume the same trajectory artifact",
        baseline={
            "hash": baseline[0].trajectory_hash,
            "frames": baseline[0].trajectory_frame_count,
        },
        candidate={
            "hash": candidate[0].trajectory_hash,
            "frames": candidate[0].trajectory_frame_count,
        },
    )
    return None


def _compare_render_input_identity(
    baseline: Sequence[TraceEvent],
    candidate: Sequence[TraceEvent],
    report: _ReportBuilder,
) -> tuple[str | None, str | None, bool]:
    if not baseline or not candidate:
        return None, None, False
    baseline_event = baseline[0]
    candidate_event = candidate[0]
    command_spans_match = (
        baseline_event.command_spans_hash == candidate_event.command_spans_hash
    )
    render_input_match = (
        baseline_event.render_input_hash == candidate_event.render_input_hash
    )
    if not command_spans_match:
        report.add(
            "command_spans_identity",
            "renderers did not use the same authoritative command partition",
            baseline=baseline_event.command_spans_hash,
            candidate=candidate_event.command_spans_hash,
        )
    if not render_input_match:
        report.add(
            "render_input_identity",
            "renderers did not use the same temporal render input",
            baseline=baseline_event.render_input_hash,
            candidate=candidate_event.render_input_hash,
        )
    return (
        baseline_event.command_spans_hash if command_spans_match else None,
        baseline_event.render_input_hash if render_input_match else None,
        command_spans_match and render_input_match,
    )


def _common_event_keys(
    baseline: Sequence[TraceEvent],
    candidate: Sequence[TraceEvent],
    report: _ReportBuilder,
) -> tuple[
    dict[tuple[int, str], TraceEvent],
    dict[tuple[int, str], TraceEvent],
    list[tuple[int, str]],
]:
    baseline_by_key = {_event_key(event): event for event in baseline}
    candidate_by_key = {_event_key(event): event for event in candidate}
    if set(baseline_by_key) != set(candidate_by_key):
        report.add(
            "event_keys",
            "traces contain different global-index/phase events",
            baseline=sorted(baseline_by_key),
            candidate=sorted(candidate_by_key),
        )
    common = sorted(
        set(baseline_by_key) & set(candidate_by_key),
        key=lambda key: (key[0], _phase_sort_index(key[1])),
    )
    return baseline_by_key, candidate_by_key, common


def _phase_sort_index(phase: str) -> int:
    try:
        return TRACE_PHASES.index(phase)
    except ValueError:
        return len(TRACE_PHASES)


def compare_traces(
    baseline: Sequence[TraceEvent],
    candidate: Sequence[TraceEvent],
    *,
    config: ParityConfig | None = None,
    rgb_loader: Callable[[TraceEvent], object] | None = None,
) -> ParityReport:
    """Compare two complete traces and return every parity-gate failure."""
    resolved_config = config or ParityConfig()
    report = _ReportBuilder(resolved_config)
    _validate_trace(baseline, report, "baseline")
    _validate_trace(candidate, report, "candidate")
    trajectory_hash = _compare_trajectory_identity(baseline, candidate, report)
    if baseline and candidate and trajectory_hash is None:
        return _finish_report(report, None, None, None, baseline, 0)
    command_spans_hash, render_input_hash, same_render_input = (
        _compare_render_input_identity(baseline, candidate, report)
    )
    if baseline and candidate and not same_render_input:
        return _finish_report(
            report,
            trajectory_hash,
            command_spans_hash,
            render_input_hash,
            baseline,
            0,
        )
    baseline_by_key, candidate_by_key, common_keys = _common_event_keys(
        baseline, candidate, report
    )
    for key in common_keys:
        _compare_event(baseline_by_key[key], candidate_by_key[key], report, rgb_loader)
    return _finish_report(
        report,
        trajectory_hash,
        command_spans_hash,
        render_input_hash,
        baseline,
        len(common_keys),
    )


def _finish_report(
    report: _ReportBuilder,
    trajectory_hash: str | None,
    command_spans_hash: str | None,
    render_input_hash: str | None,
    baseline: Sequence[TraceEvent],
    compared_events: int,
) -> ParityReport:
    frame_count = baseline[0].trajectory_frame_count if baseline else 0
    return ParityReport(
        passed=not report.issues,
        trajectory_hash=trajectory_hash,
        command_spans_hash=command_spans_hash,
        render_input_hash=render_input_hash,
        frame_count=frame_count,
        compared_events=compared_events,
        issues=tuple(report.issues),
        rgb_metrics=tuple(report.rgb_metrics),
    )


def compare_trace_files(
    baseline_path: str | os.PathLike[str],
    candidate_path: str | os.PathLike[str],
    *,
    config: ParityConfig | None = None,
    rgb_loader: Callable[[TraceEvent], object] | None = None,
) -> ParityReport:
    """Read two JSONL traces and apply the same strict parity gates."""
    return compare_traces(
        read_trace(baseline_path),
        read_trace(candidate_path),
        config=config,
        rgb_loader=rgb_loader,
    )

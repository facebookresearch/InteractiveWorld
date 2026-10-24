# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Strict JSON artifacts for replaying an exact dense Frame trajectory."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Never

import numpy as np

if __package__:
    from simulator.physical_world.humanoid_motion import Frame
    from simulator.physical_world.render_parity_trace import (
        command_spans_hash,
        normalize_command_spans,
        render_input_hash,
        TraceFormatError,
        TrajectoryIdentity,
    )
else:
    from humanoid_motion import Frame
    from render_parity_trace import (
        command_spans_hash,
        normalize_command_spans,
        render_input_hash,
        TraceFormatError,
        TrajectoryIdentity,
    )


SCHEMA_NAME = "isaacsim_try.frame_trajectory"
SCHEMA_VERSION = 3

_ROOT_FIELDS_V1 = frozenset(
    {"schema", "schema_version", "trajectory_hash", "frame_hashes", "frames"}
)
_ROOT_FIELDS = frozenset((*_ROOT_FIELDS_V1, "command_spans", "command_spans_hash"))
_LEGACY_FRAME_FIELDS = (
    "pose156",
    "xy",
    "yaw",
    "z",
    "arm_override",
    "rhp",
    "fpv_mode",
    "obj",
    "contact_anchor",
    "free_dynamic",
    "gaze",
    "seg",
    "obj_path",
    "openable_key",
    "carried",
    "zoom",
    "zoom_span",
    "gaze_locked",
)
_FRAME_FIELDS_V2 = (
    "pose156",
    "xy",
    "yaw",
    "z",
    "arm_override",
    "rhp",
    "fpv_mode",
    "obj",
    "contact_anchor",
    "free_dynamic",
    "gaze",
    "seg",
    "obj_path",
    "openable_key",
    "carried",
    "collision_exclude_paths",
    "zoom",
    "zoom_span",
    "running_devices",
    "gaze_locked",
)
_FRAME_FIELDS = (
    "pose156",
    "xy",
    "yaw",
    "z",
    "arm_override",
    "rhp",
    "fpv_mode",
    "obj",
    "contact_anchor",
    "free_dynamic",
    "gaze",
    "camera_eye_offset",
    "seg",
    "obj_path",
    "openable_key",
    "carried",
    "collision_exclude_paths",
    "zoom",
    "zoom_span",
    "running_devices",
    "gaze_locked",
)
_ARRAY_FIELDS = frozenset({"dtype", "shape", "data"})
_ARM_FIELDS = frozenset({"joint", "rotation"})
_OBJ_FIELDS = frozenset({"kind", "value"})
_CARRIED_FIELDS = frozenset({"path", "relative_transform"})
_FLOAT_ITEM_SIZES = frozenset({2, 4, 8})


class FrameTrajectoryFormatError(TraceFormatError):
    """Raised when a Frame trajectory artifact violates its strict schema."""


@dataclasses.dataclass(frozen=True)
class FrameTrajectoryArtifact:
    """Validated Frames and their renderer-parity content identity."""

    frames: tuple[Frame, ...]
    identity: TrajectoryIdentity
    command_spans: tuple[tuple[int, int], ...]
    command_spans_hash: str | None
    render_input_hash: str | None
    has_explicit_command_spans: bool


def _fail(message: str) -> Never:
    raise FrameTrajectoryFormatError(message)


def _exact_mapping(
    value: object, expected: frozenset[str] | tuple[str, ...], context: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{context} must be an object")
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        _fail(f"{context} fields differ: missing={missing}, extra={extra}")
    return value


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        _fail(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{context} must be finite")
    return result


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{context} must be a boolean")
    return value


def _string_tuple(
    value: object,
    context: str,
    *,
    optional: bool = False,
) -> tuple[str, ...] | None:
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)):
        _fail(f"{context} must be a sequence of strings")
    result = tuple(
        _string(item, f"{context}[{index}]") for index, item in enumerate(value)
    )
    if any(not item for item in result):
        _fail(f"{context} entries must be non-empty")
    if len(set(result)) != len(result):
        _fail(f"{context} entries must be unique")
    return result


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        _fail(f"{context} must be a string")
    return value


def _optional_string(value: object, context: str) -> str | None:
    return None if value is None else _string(value, context)


def _optional_float(value: object, context: str) -> float | None:
    return None if value is None else _finite_float(value, context)


def _shape(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        _fail(f"{context} must be a list")
    dimensions = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            _fail(f"{context} must contain non-negative integers")
        if dimension < 0:
            _fail(f"{context} must contain non-negative integers")
        dimensions.append(dimension)
    return tuple(dimensions)


def _array_dtype(value: object, context: str) -> np.dtype:
    if not isinstance(value, str):
        _fail(f"{context} dtype must be a string")
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as error:
        raise FrameTrajectoryFormatError(f"{context} has an invalid dtype") from error
    if dtype.kind != "f" or dtype.itemsize not in _FLOAT_ITEM_SIZES:
        _fail(f"{context} dtype must be a 16-, 32-, or 64-bit float")
    if dtype.str != value:
        _fail(f"{context} dtype must use numpy's canonical dtype string")
    return dtype


def _encode_array(
    value: object, context: str, allowed_shapes: tuple[tuple[int, ...], ...]
) -> dict[str, object]:
    if not isinstance(value, np.ndarray):
        _fail(f"{context} must be a numpy array")
    if value.shape not in allowed_shapes:
        _fail(f"{context} has invalid shape {value.shape}")
    _array_dtype(value.dtype.str, context)
    if not bool(np.all(np.isfinite(value))):
        _fail(f"{context} must contain only finite values")
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "data": [float(component) for component in value.reshape(-1)],
    }


def _decode_array(
    value: object, context: str, allowed_shapes: tuple[tuple[int, ...], ...]
) -> np.ndarray:
    encoded = _exact_mapping(value, _ARRAY_FIELDS, context)
    dtype = _array_dtype(encoded["dtype"], context)
    shape = _shape(encoded["shape"], f"{context}.shape")
    if shape not in allowed_shapes:
        _fail(f"{context} has invalid shape {shape}")
    data = encoded["data"]
    if not isinstance(data, list) or len(data) != math.prod(shape):
        _fail(f"{context}.data does not match shape {shape}")
    components = [
        _finite_float(component, f"{context}.data[{index}]")
        for index, component in enumerate(data)
    ]
    limit = float(np.finfo(dtype).max)
    if any(abs(component) > limit for component in components):
        _fail(f"{context} values overflow its declared dtype")
    array = np.asarray(components, dtype=dtype).reshape(shape)
    if not bool(np.all(np.isfinite(array))):
        _fail(f"{context} values overflow its declared dtype")
    decoded = [float(component) for component in array.reshape(-1)]
    if any(
        source != restored
        or (
            source == 0.0 and math.copysign(1.0, source) != math.copysign(1.0, restored)
        )
        for source, restored in zip(components, decoded)
    ):
        _fail(f"{context} data is not lossless for its declared dtype")
    return array


def _encode_optional_array(
    value: object,
    context: str,
    allowed_shapes: tuple[tuple[int, ...], ...],
) -> dict[str, object] | None:
    if value is None:
        return None
    return _encode_array(value, context, allowed_shapes)


def _decode_optional_array(
    value: object,
    context: str,
    allowed_shapes: tuple[tuple[int, ...], ...],
) -> np.ndarray | None:
    if value is None:
        return None
    return _decode_array(value, context, allowed_shapes)


def _numeric_vector(value: object, context: str, size: int) -> list[float]:
    if isinstance(value, np.ndarray):
        sequence: object = value.tolist()
    else:
        sequence = value
    if not isinstance(sequence, (list, tuple)) or len(sequence) != size:
        _fail(f"{context} must contain {size} numbers")
    return [
        _finite_float(component, f"{context}[{index}]")
        for index, component in enumerate(sequence)
    ]


def _encode_arm_override(value: object) -> list[dict[str, object]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _fail("frame.arm_override must be an object")
    encoded = []
    joints = set()
    for raw_joint, rotation in value.items():
        if isinstance(raw_joint, bool) or not isinstance(raw_joint, (int, np.integer)):
            _fail("frame.arm_override keys must be integers")
        joint = int(raw_joint)
        if joint in joints:
            _fail(f"frame.arm_override contains duplicate joint {joint}")
        joints.add(joint)
        encoded.append(
            {
                "joint": joint,
                "rotation": _numeric_vector(
                    rotation, f"frame.arm_override[{joint}]", 3
                ),
            }
        )
    return sorted(encoded, key=lambda entry: int(entry["joint"]))


def _decode_arm_override(value: object) -> dict[int, list[float]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        _fail("frame.arm_override must be a list")
    result = {}
    for index, item in enumerate(value):
        entry = _exact_mapping(item, _ARM_FIELDS, f"frame.arm_override[{index}]")
        joint = entry["joint"]
        if isinstance(joint, bool) or not isinstance(joint, int):
            _fail(f"frame.arm_override[{index}].joint must be an integer")
        if joint in result:
            _fail(f"frame.arm_override contains duplicate joint {joint}")
        result[joint] = _numeric_vector(
            entry["rotation"], f"frame.arm_override[{index}].rotation", 3
        )
    return result


def _obj_parts(value: object) -> tuple[str, object]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail("frame.obj must be a (kind, value) pair")
    return _string(value[0], "frame.obj kind"), value[1]


def _encode_obj(value: object) -> dict[str, object]:
    kind, state = _obj_parts(value)
    if kind == "hide":
        if state is not None:
            _fail("frame.obj hide value must be null")
        encoded_state = None
    elif kind == "rest":
        array = np.asarray(state)
        if array.shape == (3,):
            if isinstance(state, np.ndarray):
                encoded_state = _encode_array(state, "frame.obj rest value", ((3,),))
            else:
                encoded_state = _numeric_vector(state, "frame.obj rest value", 3)
        else:
            encoded_state = _encode_array(state, "frame.obj rest value", ((4, 4),))
    elif kind in {"carry", "pose"}:
        encoded_state = _encode_array(state, f"frame.obj {kind} value", ((4, 4),))
    else:
        _fail(f"frame.obj has unsupported kind {kind!r}")
    return {"kind": kind, "value": encoded_state}


def _decode_obj(value: object) -> tuple[str, object]:
    encoded = _exact_mapping(value, _OBJ_FIELDS, "frame.obj")
    kind = _string(encoded["kind"], "frame.obj.kind")
    state = encoded["value"]
    if kind == "hide":
        if state is not None:
            _fail("frame.obj hide value must be null")
        return kind, None
    if kind == "rest" and isinstance(state, list):
        return kind, tuple(_numeric_vector(state, "frame.obj.value", 3))
    if kind in {"rest", "carry", "pose"}:
        shapes = ((3,), (4, 4)) if kind == "rest" else ((4, 4),)
        return kind, _decode_array(state, "frame.obj.value", shapes)
    _fail(f"frame.obj has unsupported kind {kind!r}")


def _encode_carried(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        _fail("frame.carried must be a sequence")
    encoded = []
    paths = set()
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            _fail(f"frame.carried[{index}] must be a (path, transform) pair")
        path = _string(item[0], f"frame.carried[{index}].path")
        if not path or path in paths:
            _fail("frame.carried paths must be non-empty and unique")
        paths.add(path)
        encoded.append(
            {
                "path": path,
                "relative_transform": _encode_array(
                    item[1], f"frame.carried[{index}].relative_transform", ((4, 4),)
                ),
            }
        )
    return encoded


def _decode_carried(value: object) -> tuple[tuple[str, np.ndarray], ...]:
    if not isinstance(value, list):
        _fail("frame.carried must be a list")
    result = []
    paths = set()
    for index, item in enumerate(value):
        entry = _exact_mapping(item, _CARRIED_FIELDS, f"frame.carried[{index}]")
        path = _string(entry["path"], f"frame.carried[{index}].path")
        if not path or path in paths:
            _fail("frame.carried paths must be non-empty and unique")
        paths.add(path)
        transform = _decode_array(
            entry["relative_transform"],
            f"frame.carried[{index}].relative_transform",
            ((4, 4),),
        )
        result.append((path, transform))
    return tuple(result)


def _check_frame_contract(frame: object) -> None:
    try:
        fields = tuple(field.name for field in dataclasses.fields(frame))
    except TypeError as error:
        raise FrameTrajectoryFormatError(
            "trajectory values must be Frame dataclasses"
        ) from error
    if fields != _FRAME_FIELDS:
        _fail(f"Frame schema changed: expected {_FRAME_FIELDS}, got {fields}")


def _encoded_frame_fields(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        _fail("frame must be an object")
    keys = set(value)
    if keys == set(_FRAME_FIELDS):
        return _FRAME_FIELDS
    if keys == set(_FRAME_FIELDS_V2):
        return _FRAME_FIELDS_V2
    if keys == set(_LEGACY_FRAME_FIELDS):
        return _LEGACY_FRAME_FIELDS
    _exact_mapping(value, _FRAME_FIELDS, "frame")
    raise AssertionError("unreachable")


def _encode_frame(frame: Frame) -> dict[str, object]:
    _check_frame_contract(frame)
    xy = _numeric_vector(frame.xy, "frame.xy", 2)
    return {
        "pose156": _encode_array(frame.pose156, "frame.pose156", ((156,),)),
        "xy": xy,
        "yaw": _finite_float(frame.yaw, "frame.yaw"),
        "z": _finite_float(frame.z, "frame.z"),
        "arm_override": _encode_arm_override(frame.arm_override),
        "rhp": _encode_optional_array(frame.rhp, "frame.rhp", ((45,),)),
        "fpv_mode": _string(frame.fpv_mode, "frame.fpv_mode"),
        "obj": _encode_obj(frame.obj),
        "contact_anchor": _encode_optional_array(
            frame.contact_anchor, "frame.contact_anchor", ((3,),)
        ),
        "free_dynamic": _boolean(frame.free_dynamic, "frame.free_dynamic"),
        "gaze": _encode_optional_array(frame.gaze, "frame.gaze", ((3,),)),
        "camera_eye_offset": _encode_optional_array(
            frame.camera_eye_offset,
            "frame.camera_eye_offset",
            ((3,),),
        ),
        "seg": _string(frame.seg, "frame.seg"),
        "obj_path": _optional_string(frame.obj_path, "frame.obj_path"),
        "openable_key": _optional_string(frame.openable_key, "frame.openable_key"),
        "carried": _encode_carried(frame.carried),
        "collision_exclude_paths": list(
            _string_tuple(
                frame.collision_exclude_paths,
                "frame.collision_exclude_paths",
            )
            or ()
        ),
        "zoom": _optional_float(frame.zoom, "frame.zoom"),
        "zoom_span": _optional_float(frame.zoom_span, "frame.zoom_span"),
        "running_devices": (
            None
            if frame.running_devices is None
            else list(
                _string_tuple(
                    frame.running_devices,
                    "frame.running_devices",
                )
                or ()
            )
        ),
        "gaze_locked": _boolean(frame.gaze_locked, "frame.gaze_locked"),
    }


def _decode_frame(value: object) -> Frame:
    fields = _encoded_frame_fields(value)
    encoded = _exact_mapping(value, fields, "frame")
    legacy = fields == _LEGACY_FRAME_FIELDS
    has_camera_eye_offset = fields == _FRAME_FIELDS
    return Frame(
        pose156=_decode_array(encoded["pose156"], "frame.pose156", ((156,),)),
        xy=tuple(_numeric_vector(encoded["xy"], "frame.xy", 2)),
        yaw=_finite_float(encoded["yaw"], "frame.yaw"),
        z=_finite_float(encoded["z"], "frame.z"),
        arm_override=_decode_arm_override(encoded["arm_override"]),
        rhp=_decode_optional_array(encoded["rhp"], "frame.rhp", ((45,),)),
        fpv_mode=_string(encoded["fpv_mode"], "frame.fpv_mode"),
        obj=_decode_obj(encoded["obj"]),
        contact_anchor=_decode_optional_array(
            encoded["contact_anchor"],
            "frame.contact_anchor",
            ((3,),),
        ),
        free_dynamic=_boolean(encoded["free_dynamic"], "frame.free_dynamic"),
        gaze=_decode_optional_array(encoded["gaze"], "frame.gaze", ((3,),)),
        camera_eye_offset=(
            _decode_optional_array(
                encoded["camera_eye_offset"],
                "frame.camera_eye_offset",
                ((3,),),
            )
            if has_camera_eye_offset
            else None
        ),
        seg=_string(encoded["seg"], "frame.seg"),
        obj_path=_optional_string(encoded["obj_path"], "frame.obj_path"),
        openable_key=_optional_string(encoded["openable_key"], "frame.openable_key"),
        carried=_decode_carried(encoded["carried"]),
        collision_exclude_paths=(
            ()
            if legacy
            else _string_tuple(
                encoded["collision_exclude_paths"],
                "frame.collision_exclude_paths",
            )
        ),
        zoom=_optional_float(encoded["zoom"], "frame.zoom"),
        zoom_span=_optional_float(encoded["zoom_span"], "frame.zoom_span"),
        running_devices=(
            None
            if legacy
            else _string_tuple(
                encoded["running_devices"],
                "frame.running_devices",
                optional=True,
            )
        ),
        gaze_locked=_boolean(encoded["gaze_locked"], "frame.gaze_locked"),
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_from_payload(
    encoded: Mapping[str, object], frame_count: int
) -> TrajectoryIdentity:
    trajectory_hash = encoded["trajectory_hash"]
    frame_hashes = encoded["frame_hashes"]
    if not _is_sha256(trajectory_hash):
        _fail("trajectory_hash must be a lowercase SHA-256 digest")
    if not isinstance(frame_hashes, list) or len(frame_hashes) != frame_count:
        _fail("frame_hashes must contain one digest per frame")
    if not all(_is_sha256(frame_hash) for frame_hash in frame_hashes):
        _fail("frame_hashes must contain lowercase SHA-256 digests")
    return TrajectoryIdentity(
        str(trajectory_hash), tuple(str(frame_hash) for frame_hash in frame_hashes)
    )


def _normalize_command_spans(
    command_spans: Iterable[Sequence[int]] | None,
    frame_count: int,
) -> tuple[tuple[int, int], ...]:
    if command_spans is None:
        _fail("explicit command_spans are required for schema version 2")
    try:
        return normalize_command_spans(command_spans, frame_count)
    except (TypeError, ValueError) as error:
        raise FrameTrajectoryFormatError(str(error)) from error


def clip_command_spans(
    command_spans: Iterable[Sequence[int]], frame_count: int
) -> tuple[tuple[int, int], ...]:
    """Clip an authoritative command timeline to a non-empty Frame prefix."""

    materialized, source_frame_count = _materialize_command_spans(command_spans)
    normalized = _normalize_command_spans(materialized, source_frame_count)
    if frame_count <= 0 or frame_count > source_frame_count:
        _fail("frame_count must select a non-empty prefix within command_spans")
    if frame_count == source_frame_count:
        return normalized
    clipped = []
    for start, end in normalized:
        if start >= frame_count:
            break
        clipped.append((start, min(end, frame_count)))
        if end >= frame_count:
            break
    return tuple(clipped)


def command_horizon_end(
    command_spans: Iterable[Sequence[int]], frame_count: int
) -> int:
    """Return the end of the source command containing a rendered Frame prefix."""

    materialized, source_frame_count = _materialize_command_spans(command_spans)
    normalized = _normalize_command_spans(materialized, source_frame_count)
    if frame_count <= 0 or frame_count > source_frame_count:
        _fail("frame_count must select a non-empty prefix within command_spans")
    for start, end in normalized:
        if start < frame_count <= end:
            return end
    return frame_count


def _materialize_command_spans(
    command_spans: Iterable[Sequence[int]],
) -> tuple[tuple[Sequence[int], ...], int]:
    try:
        materialized = tuple(command_spans)
    except TypeError:
        _fail("command_spans must be iterable")
    if not materialized:
        _fail("command_spans must contain at least one command")
    last = materialized[-1]
    if not isinstance(last, (list, tuple)) or len(last) != 2:
        _fail(f"command_spans[{len(materialized) - 1}] must be a [start, end] pair")
    source_frame_count = last[1]
    if isinstance(source_frame_count, bool) or not isinstance(source_frame_count, int):
        _fail("the final command span end must be an integer")
    return materialized, source_frame_count


def _command_spans_hash(command_spans: Sequence[tuple[int, int]]) -> str:
    return command_spans_hash(command_spans)


def _render_input_hash(identity: TrajectoryIdentity, command_spans_hash: str) -> str:
    return render_input_hash(identity, command_spans_hash)


def _build_payload(
    frames: Iterable[Frame],
    command_spans: Iterable[Sequence[int]] | None = None,
) -> dict[str, object]:
    materialized = tuple(frames)
    if not materialized:
        _fail("a trajectory artifact must contain at least one Frame")
    encoded_frames = [_encode_frame(frame) for frame in materialized]
    source_identity = TrajectoryIdentity.from_frames(materialized)
    decoded_identity = TrajectoryIdentity.from_frames(
        _decode_frame(frame) for frame in encoded_frames
    )
    if decoded_identity != source_identity:
        _fail("Frame encoding would change the renderer-parity trajectory identity")
    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": 1 if command_spans is None else SCHEMA_VERSION,
        "trajectory_hash": source_identity.trajectory_hash,
        "frame_hashes": list(source_identity.frame_hashes),
        "frames": encoded_frames,
    }
    if command_spans is not None:
        normalized_spans = _normalize_command_spans(command_spans, len(materialized))
        payload["command_spans"] = [[start, end] for start, end in normalized_spans]
        payload["command_spans_hash"] = _command_spans_hash(normalized_spans)
    return payload


def serialize_frame_trajectory(
    frames: Iterable[Frame],
    command_spans: Iterable[Sequence[int]] | None = None,
) -> str:
    """Return a deterministic, self-verifying JSON representation of Frames."""
    return (
        json.dumps(
            _build_payload(frames, command_spans),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    _fail(f"JSON contains non-finite constant {value}")


def _parse_json(payload: str | bytes) -> object:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise FrameTrajectoryFormatError("artifact is not valid UTF-8") from error
    if not isinstance(payload, str):
        _fail("artifact payload must be text or UTF-8 bytes")
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise FrameTrajectoryFormatError("artifact is not valid JSON") from error


def deserialize_frame_trajectory(payload: str | bytes) -> FrameTrajectoryArtifact:
    """Validate JSON and reconstruct the exact Frame trajectory without code execution."""
    parsed = _parse_json(payload)
    if not isinstance(parsed, Mapping):
        _fail("artifact must be an object")
    version = parsed.get("schema_version")
    root_fields = _ROOT_FIELDS_V1 if version == 1 else _ROOT_FIELDS
    encoded = _exact_mapping(parsed, root_fields, "artifact")
    if encoded["schema"] != SCHEMA_NAME:
        _fail(f"artifact schema must be {SCHEMA_NAME!r}")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in (1, 2, SCHEMA_VERSION)
    ):
        _fail(f"unsupported artifact schema version {version!r}")
    encoded_frames = encoded["frames"]
    if not isinstance(encoded_frames, list) or not encoded_frames:
        _fail("artifact frames must be a non-empty list")
    frame_schemas = {_encoded_frame_fields(frame) for frame in encoded_frames}
    if len(frame_schemas) != 1:
        _fail("artifact frames must use one consistent frame schema")
    frame_schema = next(iter(frame_schemas))
    frames = tuple(_decode_frame(frame) for frame in encoded_frames)
    declared_identity = _identity_from_payload(encoded, len(frames))
    verification_values = (
        frames
        if frame_schema == _FRAME_FIELDS
        else tuple(
            {name: getattr(frame, name) for name in frame_schema} for frame in frames
        )
    )
    verified_identity = TrajectoryIdentity.from_frames(verification_values)
    for index, (declared, actual) in enumerate(
        zip(declared_identity.frame_hashes, verified_identity.frame_hashes)
    ):
        if declared != actual:
            _fail(f"frame {index} does not match its declared SHA-256 identity")
    if declared_identity.trajectory_hash != verified_identity.trajectory_hash:
        _fail("trajectory does not match its declared SHA-256 identity")
    actual_identity = TrajectoryIdentity.from_frames(frames)
    command_spans = (
        _normalize_command_spans(encoded["command_spans"], len(frames))
        if version >= 2
        else ((0, len(frames)),)
    )
    command_spans_hash = None
    render_input_hash = None
    if version >= 2:
        declared_command_spans_hash = encoded["command_spans_hash"]
        if not _is_sha256(declared_command_spans_hash):
            _fail("command_spans_hash must be a lowercase SHA-256 digest")
        command_spans_hash = _command_spans_hash(command_spans)
        if declared_command_spans_hash != command_spans_hash:
            _fail("command_spans do not match their declared SHA-256 identity")
        render_input_hash = _render_input_hash(actual_identity, command_spans_hash)
    return FrameTrajectoryArtifact(
        frames=frames,
        identity=actual_identity,
        command_spans=command_spans,
        command_spans_hash=command_spans_hash,
        render_input_hash=render_input_hash,
        has_explicit_command_spans=version >= 2,
    )


def write_frame_trajectory(
    path: str | os.PathLike[str],
    frames: Iterable[Frame],
    command_spans: Iterable[Sequence[int]] | None = None,
) -> TrajectoryIdentity:
    """Write a canonical UTF-8 trajectory artifact and return its identity."""
    payload = serialize_frame_trajectory(frames, command_spans)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return deserialize_frame_trajectory(payload).identity


def read_frame_trajectory(path: str | os.PathLike[str]) -> FrameTrajectoryArtifact:
    """Load and validate a trajectory artifact from disk."""
    return deserialize_frame_trajectory(Path(path).read_bytes())

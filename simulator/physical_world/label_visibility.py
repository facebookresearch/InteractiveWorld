"""Cheap geometric validation for whether an egocentric camera can read a label.

The evaluator uses label layout metrics written beside the texture and the same camera geometry used
by rendering. It checks the primary glyph height at model-delivery resolution, full-label framing,
occlusion, viewing angle, text roll, and the wearer's neck limits. No image is rendered and no OCR or
model call runs while candidates are searched.

``Assessment.verdict`` names the first failed constraint so the caller can change posture or viewpoint
and evaluate the new candidate with the same predicate before committing to it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import head_mount
import numpy as np
from gaze_stabilizer import MAX_PITCH_DOWN_DEG, MAX_PITCH_UP_DEG, MAX_YAW_DEG

# Horizontal field of view of the head-mounted lens: 16 mm focal on a 36 mm aperture.
HFOV_DEG = math.degrees(2.0 * math.atan(36.0 / (2.0 * head_mount.FOCAL_MM)))
# A label smaller than this in the image cannot be read (the design record).
MIN_LABEL_PX = 40
# Text is judged at the resolution delivered to the model, not the larger render target.
DELIVERED_IMAGE_W = 512
DELIVERED_IMAGE_H = 405
MIN_GLYPH_PX = 10
TARGET_GLYPH_PX = 12
MAX_TEXT_ROLL_DEG = 60.0
HORIZONTAL_APERTURE_MM = 36.0
# Beyond this obliquity the glyphs shear (the retired _SIGN_FAN_DEG's own measurement).
MAX_OFF_NORMAL_DEG = 40.0
# Closer than this and the wearer's own forearm and the near clip get in the way; the eye also
# cannot converge. Further than this and even a legible label is a smear at this resolution.
NEAR_M = 0.35
# A deliberately held label has its grip moved away from the writing and may be brought to a
# normal near-reading distance without the body/self-occlusion risk of an in-place label.
HELD_NEAR_M = 0.30
FAR_M = 2.5

OK = "ok"
OCCLUDED = "occluded"
TOO_LOW = "too_low"
TOO_HIGH = "too_high"
OFF_TO_SIDE = "off_to_side"
TOO_NEAR = "too_near"
TOO_SMALL = "too_small"
OFF_NORMAL = "off_normal"
ROTATED = "rotated"
OUT_OF_FRAME = "out_of_frame"
NO_LABEL = "no_label"


@dataclass(frozen=True)
class Label:
    """What is to be read. ``face`` may be None for a target with no recorded orientation (then the
    obliquity term is skipped rather than guessed)."""

    pos: np.ndarray
    span_m: float
    face: np.ndarray | None = None
    size_m: tuple[float, float] | None = None
    up: np.ndarray | None = None
    glyph_height_ratio: float | None = None
    content_box_uv: tuple[float, float, float, float] | None = None

    @property
    def glyph_height_m(self) -> float | None:
        if self.glyph_height_ratio is None or self.glyph_height_ratio <= 0.0:
            return None
        height = self.size_m[1] if self.size_m is not None else self.span_m
        return float(height) * float(self.glyph_height_ratio)

    @property
    def readable_span_m(self) -> float:
        return self.glyph_height_m or self.span_m


@dataclass(frozen=True)
class Assessment:
    verdict: str
    dist_m: float
    px: float
    off_normal_deg: float
    yaw_deg: float  # of the label in the TORSO frame; + is the wearer's left
    pitch_deg: float  # + is below the wearer's eye line
    occluder: str = ""
    glyph_px: float = 0.0
    text_roll_deg: float = 0.0
    in_frame: bool = True
    required_px: float = 0.0
    focal_mm: float = head_mount.FOCAL_MM

    @property
    def ok(self) -> bool:
        return self.verdict == OK


def torso_relative_angles(
    bearing_world: np.ndarray, spine_rotation: np.ndarray
) -> tuple[float, float]:
    """(yaw, pitch) in degrees of a world direction expressed in the TORSO frame.

    Uses ``gaze_stabilizer.clamp_head_on_torso``'s decomposition exactly (head-local +Y up, +Z
    face-forward, positive pitch = looking down), so "outside the comfort cone" here and "the neck
    clamp binds" there are the same statement about the same numbers.
    """
    local = spine_rotation.T @ _unit(bearing_world)
    yaw = math.degrees(math.atan2(float(local[0]), float(local[2])))
    pitch = math.degrees(math.asin(min(1.0, max(-1.0, -float(local[1])))))
    return yaw, pitch


def camera_relative_angles(
    bearing_world: np.ndarray,
    camera_look: np.ndarray,
    camera_up: np.ndarray = head_mount.WORLD_UP,
) -> tuple[float, float]:
    """Return image-relative yaw and pitch for a world-space bearing."""
    basis = _camera_basis(camera_look, camera_up)
    if basis is None:
        return 0.0, 0.0
    right, up, forward = basis
    bearing = _unit(bearing_world)
    depth = float(np.dot(bearing, forward))
    yaw = math.degrees(math.atan2(float(np.dot(bearing, right)), depth))
    pitch = math.degrees(math.atan2(-float(np.dot(bearing, up)), depth))
    return yaw, pitch


def subtended_px(
    span_m: float,
    dist_m: float,
    image_w: int,
    hfov_deg: float = HFOV_DEG,
    *,
    focal_mm: float | None = None,
) -> float:
    """How many pixels across the label is, at this distance, through this lens."""
    if dist_m <= 1e-6 or span_m <= 0.0:
        return 0.0
    if focal_mm is not None:
        hfov_deg = math.degrees(
            2.0 * math.atan(HORIZONTAL_APERTURE_MM / (2.0 * float(focal_mm)))
        )
    theta = 2.0 * math.atan(0.5 * span_m / dist_m)
    return float(theta / math.radians(hfov_deg) * image_w)


def zoom_focal_mm(
    span_m: float,
    dist_m: float,
    *,
    base_focal_mm: float = head_mount.FOCAL_MM,
    target_frame_fraction: float = 0.45,
    max_focal_multiple: float = 6.0,
) -> float:
    """Match the egocentric camera's label zoom using only scalar geometry."""
    if span_m <= 0.0 or dist_m <= 1e-6:
        return float(base_focal_mm)
    base_hfov = 2.0 * math.atan(0.5 * HORIZONTAL_APERTURE_MM / float(base_focal_mm))
    subtend = 2.0 * math.atan(0.5 * span_m / dist_m)
    target_hfov = max(1e-6, min(base_hfov, subtend / target_frame_fraction))
    target_focal = 0.5 * HORIZONTAL_APERTURE_MM / math.tan(target_hfov / 2.0)
    return min(float(base_focal_mm) * max_focal_multiple, target_focal)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3)


def front_facing_object_rotation(
    rest_rotation: np.ndarray,
    label_face_world: np.ndarray | None,
    label_up_world: np.ndarray | None,
    toward_camera_world: np.ndarray,
    camera_up_world: np.ndarray | None = None,
) -> np.ndarray | None:
    """Rotate a held object so its label faces the camera with upright text.

    Label axes are expressed in the object's resting world pose. The returned matrix remains an
    object-local-to-world rotation, so it can replace the rotation block of a held object transform.
    """
    if label_face_world is None or label_up_world is None:
        return None
    rest = np.asarray(rest_rotation, dtype=np.float64)
    if rest.shape != (3, 3):
        return None
    source_face = _unit(np.asarray(label_face_world, dtype=np.float64))
    source_up = np.asarray(label_up_world, dtype=np.float64)
    source_up = _unit(source_up - source_face * float(np.dot(source_up, source_face)))
    target_face = _unit(np.asarray(toward_camera_world, dtype=np.float64))
    up_reference = np.asarray(
        head_mount.WORLD_UP if camera_up_world is None else camera_up_world,
        dtype=np.float64,
    )
    target_up = _unit(
        up_reference - target_face * float(np.dot(up_reference, target_face))
    )
    if not np.any(source_face) or not np.any(source_up) or not np.any(target_face):
        return None
    if not np.any(target_up):
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        target_up = _unit(fallback - target_face * float(np.dot(fallback, target_face)))
    if not np.any(target_up):
        return None
    source_right = _unit(np.cross(source_up, source_face))
    target_right = _unit(np.cross(target_up, target_face))
    if not np.any(source_right) or not np.any(target_right):
        return None
    source_up = _unit(np.cross(source_face, source_right))
    target_up = _unit(np.cross(target_face, target_right))
    source_basis = np.column_stack((source_right, source_up, source_face))
    target_basis = np.column_stack((target_right, target_up, target_face))
    return target_basis @ source_basis.T @ rest


def interpolate_rotation(
    start_rotation: np.ndarray, end_rotation: np.ndarray, fraction: float
) -> np.ndarray:
    """Interpolate two rotation matrices and project the result back onto SO(3)."""
    start = np.asarray(start_rotation, dtype=np.float64)
    end = np.asarray(end_rotation, dtype=np.float64)
    amount = min(1.0, max(0.0, float(fraction)))
    if amount <= 0.0:
        return start.copy()
    if amount >= 1.0:
        return end.copy()
    left, _singular, right = np.linalg.svd((1.0 - amount) * start + amount * end)
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _camera_basis(
    look: np.ndarray, up_reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    forward = _unit(look)
    if not np.any(forward):
        return None
    right = _unit(np.cross(forward, _unit(up_reference)))
    if not np.any(right):
        right = _unit(np.cross(forward, np.array([0.0, 1.0, 0.0])))
    if not np.any(right):
        return None
    return right, _unit(np.cross(right, forward)), forward


def _project_point_px(
    point: np.ndarray,
    eye: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    focal_px: float,
) -> tuple[float, float] | None:
    right, up, forward = basis
    relative = point - eye
    depth = float(np.dot(relative, forward))
    if depth <= 1e-6:
        return None
    return (
        float(np.dot(relative, right)) * focal_px / depth,
        float(np.dot(relative, up)) * focal_px / depth,
    )


def _label_axes(label: Label) -> tuple[np.ndarray, np.ndarray] | None:
    if label.up is None:
        return None
    up = _unit(np.asarray(label.up, dtype=np.float64))
    if not np.any(up):
        return None
    if label.face is None:
        return None
    normal = _unit(np.asarray(label.face, dtype=np.float64))
    up = _unit(up - normal * float(np.dot(up, normal)))
    if not np.any(up):
        return None
    right = _unit(np.cross(up, normal))
    return (right, up) if np.any(right) else None


def _projected_glyph_px(
    eye: np.ndarray,
    label: Label,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    focal_px: float,
) -> float:
    glyph_height = label.glyph_height_m
    if glyph_height is None:
        return 0.0
    axes = _label_axes(label)
    if axes is None or basis is None:
        dist = float(np.linalg.norm(np.asarray(label.pos) - eye))
        return glyph_height * focal_px / max(dist, 1e-6)
    up = axes[1]
    half = 0.5 * glyph_height
    first = _project_point_px(label.pos - up * half, eye, basis, focal_px)
    second = _project_point_px(label.pos + up * half, eye, basis, focal_px)
    if first is None or second is None:
        return 0.0
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _text_roll_deg(
    label: Label, basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None
) -> float:
    axes = _label_axes(label)
    if axes is None or basis is None:
        return 0.0
    label_up = axes[1]
    camera_right, camera_up, _forward = basis
    return math.degrees(
        math.atan2(
            float(np.dot(label_up, camera_right)),
            float(np.dot(label_up, camera_up)),
        )
    )


def _label_in_frame(
    eye: np.ndarray,
    label: Label,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    focal_px: float,
    image_w: int,
    image_h: int,
) -> bool:
    if basis is None:
        return True
    axes = _label_axes(label)
    size = label.size_m
    if axes is None or size is None:
        projected = _project_point_px(label.pos, eye, basis, focal_px)
        return bool(
            projected is not None
            and abs(projected[0]) <= 0.5 * image_w
            and abs(projected[1]) <= 0.5 * image_h
        )
    right, up = axes
    width, height = float(size[0]), float(size[1])
    u0, v0, u1, v1 = label.content_box_uv or (0.0, 0.0, 1.0, 1.0)
    corners = [
        _project_point_px(
            label.pos + right * (u - 0.5) * width + up * (0.5 - v) * height,
            eye,
            basis,
            focal_px,
        )
        for u in (u0, u1)
        for v in (v0, v1)
    ]
    return all(
        point is not None
        and abs(point[0]) <= 0.5 * image_w
        and abs(point[1]) <= 0.5 * image_h
        for point in corners
    )


def _segment_hits_box(
    a: np.ndarray, b: np.ndarray, mn: np.ndarray, mx: np.ndarray
) -> bool:
    """Slab test: does the segment a->b enter the AABB? Endpoints inside count as a hit."""
    d = b - a
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if a[axis] < mn[axis] or a[axis] > mx[axis]:
                return False
            continue
        lo = (mn[axis] - a[axis]) / d[axis]
        hi = (mx[axis] - a[axis]) / d[axis]
        if lo > hi:
            lo, hi = hi, lo
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            return False
    return True


def _off_normal_deg(to_label: np.ndarray, face: np.ndarray | None) -> float:
    if face is None:
        return 0.0
    normal = _unit(np.asarray(face, dtype=np.float64))
    if not np.any(normal):
        return 0.0
    view = _unit(-to_label)
    if not np.any(view) or not np.any(normal):
        return 0.0
    cosine = float(np.dot(view, normal))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _first_occluder(
    eye: np.ndarray,
    pos: np.ndarray,
    occluders,
    self_radius_m: float,
) -> str:
    for name, mn, mx in occluders:
        if any("glass" in part for part in str(name).lower().split("/")[-2:]):
            continue
        mn = np.asarray(mn, dtype=np.float64)
        mx = np.asarray(mx, dtype=np.float64)
        if np.all(pos >= mn - 1e-6) and np.all(pos <= mx + 1e-6):
            continue
        if np.all(eye >= mn - self_radius_m) and np.all(eye <= mx + self_radius_m):
            continue
        if _segment_hits_box(eye, pos, mn, mx):
            return name
    return ""


def _visibility_verdict(
    *,
    occluder: str,
    pitch: float,
    yaw: float,
    dist: float,
    detail_px: float,
    required_px: float,
    off_normal: float,
    in_frame: bool,
    text_roll: float,
    enforce_torso_limits: bool,
    min_distance_m: float,
) -> str:
    if occluder:
        return OCCLUDED
    if enforce_torso_limits and pitch > MAX_PITCH_DOWN_DEG:
        return TOO_LOW
    if enforce_torso_limits and pitch < -MAX_PITCH_UP_DEG:
        return TOO_HIGH
    if enforce_torso_limits and abs(yaw) > MAX_YAW_DEG:
        return OFF_TO_SIDE
    if dist < min_distance_m:
        return TOO_NEAR
    if not in_frame:
        return OUT_OF_FRAME
    if detail_px < required_px or dist > FAR_M:
        return TOO_SMALL
    if off_normal > MAX_OFF_NORMAL_DEG:
        return OFF_NORMAL
    if abs(text_roll) > MAX_TEXT_ROLL_DEG:
        return ROTATED
    return OK


def assess(
    eye: np.ndarray,
    spine_rotation: np.ndarray,
    label: Label,
    occluders=(),
    *,
    image_w: int = 960,
    image_h: int | None = None,
    min_px: int = MIN_LABEL_PX,
    min_glyph_px: int = MIN_GLYPH_PX,
    focal_mm: float = head_mount.FOCAL_MM,
    camera_look: np.ndarray | None = None,
    camera_up: np.ndarray = head_mount.WORLD_UP,
    self_radius_m: float = 0.25,
    enforce_torso_limits: bool = True,
    min_distance_m: float = NEAR_M,
) -> Assessment:
    """Judge the CURRENT viewing position, and name the one thing wrong with it.

    ``occluders`` is an iterable of (name, min_xyz, max_xyz) -- ``CollisionWorld.boxes()`` shape.
    Boxes whose own volume contains the label are skipped: a placard is mounted ON the cabinet, so
    the cabinet's own AABB always swallows it and would report every label as occluded.
    """
    eye = np.asarray(eye, dtype=np.float64)
    pos = np.asarray(label.pos, dtype=np.float64)
    to_label = pos - eye
    dist = float(np.linalg.norm(to_label))
    if dist <= 1e-6 or label.span_m <= 0.0:
        return Assessment(NO_LABEL, dist, 0.0, 0.0, 0.0, 0.0)

    yaw, pitch = torso_relative_angles(to_label, spine_rotation)
    px = subtended_px(label.span_m, dist, image_w, focal_mm=focal_mm)
    off_normal = _off_normal_deg(to_label, label.face)
    look = (
        to_label if camera_look is None else np.asarray(camera_look, dtype=np.float64)
    )
    basis = _camera_basis(look, np.asarray(camera_up, dtype=np.float64))
    focal_px = float(focal_mm) / HORIZONTAL_APERTURE_MM * image_w
    glyph_px = _projected_glyph_px(eye, label, basis, focal_px)
    text_roll = _text_roll_deg(label, basis)
    in_frame = _label_in_frame(
        eye,
        label,
        basis,
        focal_px,
        image_w,
        (
            image_h
            if image_h is not None
            else round(image_w * DELIVERED_IMAGE_H / DELIVERED_IMAGE_W)
        ),
    )
    detail_px = glyph_px if label.glyph_height_m is not None else px
    required_px = float(min_glyph_px if label.glyph_height_m is not None else min_px)

    occluder = _first_occluder(eye, pos, occluders, self_radius_m)
    verdict = _visibility_verdict(
        occluder=occluder,
        pitch=pitch,
        yaw=yaw,
        dist=dist,
        detail_px=detail_px,
        required_px=required_px,
        off_normal=off_normal,
        in_frame=in_frame,
        text_roll=text_roll,
        enforce_torso_limits=enforce_torso_limits,
        min_distance_m=float(min_distance_m),
    )
    return Assessment(
        verdict,
        dist,
        px,
        off_normal,
        yaw,
        pitch,
        occluder,
        glyph_px,
        text_roll,
        in_frame,
        required_px,
        float(focal_mm),
    )


@lru_cache(maxsize=2048)
def texture_readability(texture_path: str) -> dict[str, object]:
    """Read cached label-layout metrics without loading the texture or running OCR."""
    try:
        record = json.loads(Path(f"{texture_path}.recipe.json").read_text())
    except (OSError, ValueError, TypeError):
        return {}
    readability = record.get("readability") or {}
    ratio = readability.get("primary_glyph_height_ratio")
    if ratio is None and record.get("kind") == "short_string":
        ratio = 0.05
    try:
        ratio_value = float(ratio)
    except (TypeError, ValueError):
        return {}
    if not 0.0 < ratio_value <= 1.0:
        return {}
    result: dict[str, object] = {
        "kind": str(readability.get("kind") or "text"),
        "glyph_height_ratio": ratio_value,
        "line_count": int(readability.get("primary_line_count") or 0),
    }
    raw_box = readability.get("primary_text_bbox_norm_xyxy")
    if isinstance(raw_box, list) and len(raw_box) == 4:
        try:
            box = tuple(float(value) for value in raw_box)
        except (TypeError, ValueError):
            box = ()
        if (
            len(box) == 4
            and 0.0 <= box[0] < box[2] <= 1.0
            and 0.0 <= box[1] < box[3] <= 1.0
        ):
            result["content_box_uv"] = box
    return result


def _vector3(value) -> np.ndarray | None:
    if value is None:
        return None
    values = [float(item) for item in value]
    if len(values) == 2:
        values.append(0.0)
    if len(values) < 3 or sum(abs(item) for item in values[:3]) <= 1e-6:
        return None
    return np.array(values[:3], dtype=np.float64)


def _meta_value(meta: dict, *keys: str):
    for key in keys:
        value = meta.get(key)
        if value is not None:
            return value
    return None


def label_from_meta(
    meta: dict, fallback_bbox=None, *, texture_path: str | None = None
) -> Label | None:
    """Build a ``Label`` from the entity metadata the importers already write.

    ``label_pos`` / ``label_span`` / ``label_face`` are written by ``task_world_builder`` and
    ``behavior_import``; a target with none of them falls back to its bounding box, which is the
    honest reading of "look at the thing" when nothing says where its writing is.
    """
    texture_path = texture_path or meta.get("label_texture")
    pos = meta.get("label_pos")
    if pos is not None:
        face3 = _vector3(_meta_value(meta, "label_face3", "label_face"))
        up3 = _vector3(_meta_value(meta, "label_up3", "label_up"))
        raw_size = meta.get("label_size")
        size = None
        if raw_size is not None and len(raw_size) >= 2:
            size = (float(raw_size[0]), float(raw_size[1]))
        glyph_ratio = _meta_value(
            meta,
            "label_primary_glyph_height_ratio",
            "label_glyph_height_ratio",
        )
        if glyph_ratio is None and texture_path:
            glyph_ratio = texture_readability(texture_path).get("glyph_height_ratio")
        content_box = meta.get("label_primary_text_bbox_norm_xyxy")
        if content_box is None and texture_path:
            content_box = texture_readability(texture_path).get("content_box_uv")
        content_box_uv = None
        if content_box is not None and len(content_box) == 4:
            content_box_uv = tuple(float(value) for value in content_box)
        return Label(
            np.array([float(pos[0]), float(pos[1]), float(pos[2])]),
            float(meta.get("label_span") or 0.0),
            face3,
            size,
            up3,
            None if glyph_ratio is None else float(glyph_ratio),
            content_box_uv,
        )
    if fallback_bbox is None:
        return None
    (bx, by), zmin, zmax = fallback_bbox
    return Label(
        np.array([float(bx), float(by), 0.5 * (zmin + zmax)]),
        float(max(zmax - zmin, 0.0)),
        None,
    )


def transform_label(label: Label, transform: np.ndarray) -> Label:
    """Move a label rigidly with its parent object."""
    matrix = np.asarray(transform, dtype=np.float64)
    rotation = matrix[:3, :3]
    return Label(
        rotation @ np.asarray(label.pos, dtype=np.float64) + matrix[:3, 3],
        label.span_m,
        None if label.face is None else rotation @ label.face,
        label.size_m,
        None if label.up is None else rotation @ label.up,
        label.glyph_height_ratio,
        label.content_box_uv,
    )


def _selftest() -> int:
    """A placard 1.2 m ahead at eye height, then each way of getting it wrong.

    The world is Z-UP (``head_mount.WORLD_UP``) while head-local is +Y up / +Z forward, so the spine
    rotation below is the one a standing body actually has. An identity spine would put the test in
    a Y-up world and quietly break the horizontal terms.
    """
    spine = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )  # local +Y->+Z, +Z->+X
    eye = np.zeros(3)
    fails = []

    def check(name, want, pos, span=0.20, face=None, occ=()):
        got = assess(
            eye,
            spine,
            Label(
                np.array(pos, dtype=np.float64),
                span,
                None if face is None else np.array(face, dtype=np.float64),
            ),
            occ,
        )
        if got.verdict != want:
            fails.append(f"{name}: wanted {want}, got {got.verdict} ({got})")
        return got

    good = check("head-on", OK, [1.2, 0.0, 0.0], face=[-1.0, 0.0, 0.0])
    check("far away", TOO_SMALL, [6.0, 0.0, 0.0])
    check("tiny", TOO_SMALL, [1.2, 0.0, 0.0], span=0.004)
    check("nose on it", TOO_NEAR, [0.2, 0.0, 0.0])
    check("at the feet", TOO_LOW, [0.6, 0.0, -1.4])
    check("overhead", TOO_HIGH, [0.6, 0.0, 1.4])
    check("behind the shoulder", OFF_TO_SIDE, [-0.3, 1.2, 0.0])
    check("edge on", OFF_NORMAL, [1.2, 0.0, 0.0], face=[0.0, 1.0, 0.0])
    # Standing back from a LOW label must not be called off-normal just because the eye is above it:
    # label_face is a 2-D room facing and the elevation is the pitch term's business.
    check("low but square on", OK, [1.4, 0.0, -0.9], span=0.16, face=[-1.0, 0.0, 0.0])
    check(
        "blocked",
        OCCLUDED,
        [1.2, 0.0, 0.0],
        occ=[("crate", np.array([0.5, -0.3, -0.3]), np.array([0.7, 0.3, 0.3]))],
    )
    check(
        "its own mount is not an occluder",
        OK,
        [1.2, 0.0, 0.0],
        face=[-1.0, 0.0, 0.0],
        occ=[("cabinet", np.array([1.1, -0.5, -0.5]), np.array([1.6, 0.5, 0.5]))],
    )
    print(
        f"[labelvis] hfov={HFOV_DEG:.1f} deg, min={MIN_LABEL_PX}px  |  head-on placard: "
        f"{good.px:.0f}px at {good.dist_m:.2f} m -> {good.verdict}"
    )
    print(f"[labelvis] selftest -> {'PASS' if not fails else 'FAIL'}")
    for f in fails:
        print(f"  FAIL {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

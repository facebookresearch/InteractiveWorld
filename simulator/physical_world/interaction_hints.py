# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Validated, declarative interaction hints shared by task-built scenes."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


_GRASP_TARGETS = frozenset({"auto", "center", "edge", "body"})
_GRASP_FAMILIES = frozenset({"auto", "pinch", "wrap"})
_GRASP_EDGES = frozenset({"nearest", "long", "short"})
_GRASP_FIELDS = frozenset(
    {"target", "family", "edge", "height_fraction", "clearance_m"}
)
_LOOK_REGION_KINDS = frozenset({"surface", "interior"})
_LOOK_REGION_FIELDS = frozenset({"kind", "space", "members", "observer_stand_xy"})


def _mapping(value, field: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def normalize_grasp_hint(value) -> dict:
    """Validate an optional per-object grasp override and return canonical values."""
    if value in (None, {}):
        return {}
    hint = _mapping(value, "grasp_hint")
    unknown = sorted(set(hint) - _GRASP_FIELDS)
    if unknown:
        raise ValueError(f"grasp_hint has unknown field(s): {unknown}")
    target = str(hint.get("target") or "auto").lower()
    family = str(hint.get("family") or "auto").lower()
    edge = str(hint.get("edge") or "nearest").lower()
    if target not in _GRASP_TARGETS:
        raise ValueError(f"grasp_hint.target must be one of {sorted(_GRASP_TARGETS)}")
    if family not in _GRASP_FAMILIES:
        raise ValueError(f"grasp_hint.family must be one of {sorted(_GRASP_FAMILIES)}")
    if edge not in _GRASP_EDGES:
        raise ValueError(f"grasp_hint.edge must be one of {sorted(_GRASP_EDGES)}")
    try:
        height_fraction = float(hint.get("height_fraction", 0.5))
        clearance_m = float(hint.get("clearance_m", 0.012))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "grasp_hint.height_fraction and clearance_m must be numeric"
        ) from exc
    if not math.isfinite(height_fraction) or not 0.0 <= height_fraction <= 1.0:
        raise ValueError("grasp_hint.height_fraction must be between 0 and 1")
    if not math.isfinite(clearance_m) or not 0.0 <= clearance_m <= 0.10:
        raise ValueError("grasp_hint.clearance_m must be between 0 and 0.10 metres")
    return {
        "target": target,
        "family": family,
        "edge": edge,
        "height_fraction": height_fraction,
        "clearance_m": clearance_m,
    }


def normalize_look_region(value, *, entity_key: str = "") -> dict:
    """Validate an entity-relative surface/interior camera target."""
    region = _mapping(value, "look_region")
    unknown = sorted(set(region) - _LOOK_REGION_FIELDS)
    if unknown:
        raise ValueError(f"look_region has unknown field(s): {unknown}")
    kind = str(region.get("kind") or "").lower()
    if kind not in _LOOK_REGION_KINDS:
        raise ValueError(
            f"look_region.kind must be one of {sorted(_LOOK_REGION_KINDS)}"
        )
    raw_space = region.get("space", "all")
    space = str(raw_space or "all")
    raw_members = region.get("members")
    members: tuple[str, ...] = ()
    if raw_members is not None:
        if (
            not isinstance(raw_members, (list, tuple))
            or not raw_members
            or any(not isinstance(member, str) or not member for member in raw_members)
        ):
            raise ValueError(
                "look_region.members must be a non-empty list of entity keys"
            )
        members = tuple(raw_members)
        if len(set(members)) != len(members):
            raise ValueError(
                "look_region.members must not contain duplicate entity keys"
            )
    if kind == "surface" and "space" in region:
        raise ValueError("look_region.space is only valid for an interior region")
    if kind == "interior" and members:
        raise ValueError("look_region.members is only valid for a surface region")
    if kind == "interior" and space != "all":
        prefix = f"{entity_key}_space" if entity_key else ""
        suffix = space[len(prefix) :] if prefix and space.startswith(prefix) else ""
        if not prefix or not suffix.isdigit() or int(suffix) < 1:
            raise ValueError(
                "look_region.space must be 'all' or a numbered "
                f"{entity_key or '<container>'}_spaceN"
            )
    observer_stand_xy = None
    if "observer_stand_xy" in region:
        raw_observer_stand = region["observer_stand_xy"]
        if (
            not isinstance(raw_observer_stand, (list, tuple))
            or len(raw_observer_stand) != 2
        ):
            raise ValueError(
                "look_region.observer_stand_xy must be a two-coordinate list"
            )
        try:
            observer_stand_xy = [float(value) for value in raw_observer_stand]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "look_region.observer_stand_xy coordinates must be numeric"
            ) from exc
        if not all(math.isfinite(value) for value in observer_stand_xy):
            raise ValueError("look_region.observer_stand_xy coordinates must be finite")
    return {
        "kind": kind,
        **({"space": space} if kind == "interior" else {}),
        **({"members": list(members)} if members else {}),
        **(
            {"observer_stand_xy": observer_stand_xy}
            if observer_stand_xy is not None
            else {}
        ),
    }


def _bbox_region(scene, key: str) -> tuple[np.ndarray, float]:
    minimum, maximum = scene.object_aabb(key)
    lo = np.asarray(minimum, dtype=np.float64)
    hi = np.asarray(maximum, dtype=np.float64)
    return 0.5 * (lo + hi), float(max(hi - lo))


def look_region_target(scene, key: str, entity, value) -> tuple[np.ndarray, float]:
    """Resolve a validated region to a world-space gaze centre and framing span."""
    region = normalize_look_region(value, entity_key=key)
    if region["kind"] == "surface":
        members = tuple(region.get("members") or ())
        if members:
            bounds = []
            for member in members:
                if member not in scene.entities:
                    raise ValueError(
                        f"look_region member {member!r} does not exist for {key!r}"
                    )
                minimum, maximum = scene.object_aabb(member)
                bounds.append(
                    (
                        np.asarray(minimum, dtype=np.float64),
                        np.asarray(maximum, dtype=np.float64),
                    )
                )
            centers = [0.5 * (minimum + maximum) for minimum, maximum in bounds]
            union_min = np.min([minimum for minimum, _ in bounds], axis=0)
            union_max = np.max([maximum for _, maximum in bounds], axis=0)
            return np.mean(centers, axis=0), float(max(union_max - union_min))
        foot_xy = getattr(entity, "foot_xy", None) or getattr(entity, "xy", None)
        foot_half = getattr(entity, "foot_half", None) or getattr(entity, "half", None)
        top_z = getattr(entity, "top_z", None)
        if foot_xy is None or foot_half is None or top_z is None:
            center, span = _bbox_region(scene, key)
            center[2] = float(scene.object_aabb(key)[1][2])
            return center, span
        center = np.array(
            [float(foot_xy[0]), float(foot_xy[1]), float(top_z)],
            dtype=np.float64,
        )
        return center, 2.0 * max(abs(float(foot_half[0])), abs(float(foot_half[1])))

    spaces = tuple(getattr(entity, "spaces", None) or ())
    selected = spaces
    requested = region["space"]
    if requested != "all":
        selected = tuple(space for space in spaces if str(space[0]) == requested)
        if not selected:
            raise ValueError(
                f"look_region.space {requested!r} does not exist on {key!r}"
            )
    if not selected:
        xy = getattr(entity, "xy", None)
        z = getattr(entity, "z", None)
        half = getattr(entity, "half", None)
        height = getattr(entity, "height", None)
        if xy is None or z is None or half is None or height is None:
            return _bbox_region(scene, key)
        minimum = np.array(
            [float(xy[0]) - float(half[0]), float(xy[1]) - float(half[1]), float(z)],
            dtype=np.float64,
        )
        maximum = np.array(
            [
                float(xy[0]) + float(half[0]),
                float(xy[1]) + float(half[1]),
                float(z) + float(height),
            ],
            dtype=np.float64,
        )
    else:
        minimum = np.array(
            [
                min(float(space[3]) - float(space[5]) for space in selected),
                min(float(space[4]) - float(space[6]) for space in selected),
                min(float(space[2]) for space in selected),
            ],
            dtype=np.float64,
        )
        maximum = np.array(
            [
                max(float(space[3]) + float(space[5]) for space in selected),
                max(float(space[4]) + float(space[6]) for space in selected),
                max(float(space[1]) for space in selected),
            ],
            dtype=np.float64,
        )
    return 0.5 * (minimum + maximum), float(max(maximum - minimum))

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Materialize non-colliding visual effects on declared scene surfaces."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulator.taskpack_support.bundled_assets import resolve_label_asset


@dataclass(frozen=True, slots=True)
class SurfaceEffectSpec:
    key: str
    surface: str
    kind: str
    spot: tuple[float, float]
    size: tuple[float, float]
    color: tuple[float, float, float]
    opacity: float
    roughness: float
    irregularity: float
    vertices: int
    seed: int
    z_offset: float
    texture: str | None


def _bounded_number(
    value: object,
    path: str,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be a number") from error
    below = number < minimum if minimum_inclusive else number <= minimum
    if not math.isfinite(number) or below or number > maximum:
        lower = "at least" if minimum_inclusive else "greater than"
        raise ValueError(f"{path} must be {lower} {minimum} and at most {maximum}")
    return number


def _number_tuple(
    value: object,
    path: str,
    length: int,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{path} must contain exactly {length} numbers")
    return tuple(
        _bounded_number(
            item,
            f"{path}[{index}]",
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=minimum_inclusive,
        )
        for index, item in enumerate(value)
    )


def _effect_scalar_fields(row: Mapping[str, Any], path: str) -> dict[str, Any]:
    vertices = int(row.get("vertices", 18))
    if not 8 <= vertices <= 64:
        raise ValueError(f"{path}.vertices must be between 8 and 64")
    texture = str(row.get("texture") or "").strip()
    return {
        "opacity": _bounded_number(
            row.get("opacity", 0.24),
            f"{path}.opacity",
            minimum=0.02,
            maximum=0.85,
        ),
        "roughness": _bounded_number(
            row.get("roughness", 0.06),
            f"{path}.roughness",
            minimum=0.0,
            maximum=1.0,
        ),
        "irregularity": _bounded_number(
            row.get("irregularity", 0.20),
            f"{path}.irregularity",
            minimum=0.0,
            maximum=0.45,
        ),
        "vertices": vertices,
        "seed": int(row.get("seed", 0)),
        "z_offset": _bounded_number(
            row.get("z_offset", 0.002),
            f"{path}.z_offset",
            minimum=0.0005,
            maximum=0.01,
        ),
        "texture": texture or None,
    }


def _parse_surface_effect(
    raw_key: object, raw_row: object, surface_keys: Collection[str]
) -> SurfaceEffectSpec:
    key = str(raw_key)
    path = f"episode_dialogue.surface_effects.{key}"
    if not key or not isinstance(raw_row, Mapping):
        raise ValueError(f"{path} must be a named object")
    surface = str(raw_row.get("surface") or "")
    if surface not in surface_keys:
        raise ValueError(f"{path}.surface references missing surface {surface!r}")
    kind = str(raw_row.get("kind") or "")
    if kind != "water_film":
        raise ValueError(f"{path}.kind {kind!r} is unsupported")
    spot = _number_tuple(
        raw_row.get("spot", (0.0, 0.0)),
        f"{path}.spot",
        2,
        minimum=-1.0,
        maximum=1.0,
    )
    size = _number_tuple(
        raw_row.get("size"), f"{path}.size", 2, minimum=0.02, maximum=2.0
    )
    color = _number_tuple(
        raw_row.get("color", (0.72, 0.84, 0.90)),
        f"{path}.color",
        3,
        minimum=0.0,
        maximum=1.0,
    )
    return SurfaceEffectSpec(
        key=key,
        surface=surface,
        kind=kind,
        spot=(spot[0], spot[1]),
        size=(size[0], size[1]),
        color=(color[0], color[1], color[2]),
        **_effect_scalar_fields(raw_row, path),
    )


def parse_surface_effects(
    dialogue: Mapping[str, Any], surface_keys: Collection[str]
) -> tuple[SurfaceEffectSpec, ...]:
    """Validate renderer-neutral surface-effect declarations."""
    raw = dialogue.get("surface_effects") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("episode_dialogue.surface_effects must be an object")
    available = set(surface_keys)
    return tuple(_parse_surface_effect(key, row, available) for key, row in raw.items())


def irregular_film_points(
    spec: SurfaceEffectSpec,
    center: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """Return a deterministic, smooth-edged puddle polygon in world coordinates."""
    phase = float(spec.seed % 997) * 0.017
    points = []
    for index in range(spec.vertices):
        angle = 2.0 * math.pi * index / spec.vertices
        wave = 0.62 * math.sin(3.0 * angle + phase) + 0.38 * math.sin(
            5.0 * angle - 0.7 * phase
        )
        radius = 1.0 + spec.irregularity * wave
        points.append(
            (
                center[0] + 0.5 * spec.size[0] * radius * math.cos(angle),
                center[1] + 0.5 * spec.size[1] * radius * math.sin(angle),
                center[2],
            )
        )
    return tuple(points)


def _surface_center(spec: SurfaceEffectSpec, entity: Any) -> tuple[float, float, float]:
    foot_xy = getattr(entity, "foot_xy", None) or getattr(entity, "xy", None)
    foot_half = getattr(entity, "foot_half", None)
    top_z = getattr(entity, "top_z", None)
    if foot_xy is None or foot_half is None or top_z is None:
        raise ValueError(
            f"surface effect {spec.key!r} requires measured footprint and top height "
            f"for {spec.surface!r}"
        )
    half_x, half_y = abs(float(foot_half[0])), abs(float(foot_half[1]))
    margin = 0.02
    available_x = half_x - 0.5 * spec.size[0] - margin
    available_y = half_y - 0.5 * spec.size[1] - margin
    if available_x < 0.0 or available_y < 0.0:
        raise ValueError(
            f"surface effect {spec.key!r} size {spec.size} does not fit "
            f"surface {spec.surface!r} footprint {foot_half}"
        )
    return (
        float(foot_xy[0]) + spec.spot[0] * available_x,
        float(foot_xy[1]) + spec.spot[1] * available_y,
        float(top_z) + spec.z_offset,
    )


def register_surface_effect_centers(
    scene: Any, dialogue: Mapping[str, Any]
) -> dict[str, tuple[float, float, float]]:
    """Publish renderer-neutral effect centers before interaction planning."""
    entities = getattr(scene, "entities", None) or {}
    centers = {}
    for spec in parse_surface_effects(dialogue, entities.keys()):
        entity = entities[spec.surface]
        center = _surface_center(spec, entity)
        entity.meta.setdefault("surface_effect_centers", {})[spec.key] = center
        centers[spec.key] = center
    return centers


def _safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def resolve_effect_texture(source: str) -> Path:
    """Resolve an authored effect texture inside the benchmark asset root."""
    return resolve_label_asset(source)


def _film_geometry(
    spec: SurfaceEffectSpec, center: tuple[float, float, float]
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float], ...]]:
    if spec.texture is None:
        points = irregular_film_points(spec, center)
        uvs = tuple(
            (
                0.5 + (point[0] - center[0]) / spec.size[0],
                0.5 + (point[1] - center[1]) / spec.size[1],
            )
            for point in points
        )
        return points, uvs
    half_x, half_y = 0.5 * spec.size[0], 0.5 * spec.size[1]
    return (
        (
            (center[0] - half_x, center[1] - half_y, center[2]),
            (center[0] + half_x, center[1] - half_y, center[2]),
            (center[0] + half_x, center[1] + half_y, center[2]),
            (center[0] - half_x, center[1] + half_y, center[2]),
        ),
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    )


def _bind_water_material(
    stage: Any, mesh: Any, path: str, spec: SurfaceEffectSpec
) -> None:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, f"{path}/WaterMaterial")
    shader = UsdShade.Shader.Define(stage, f"{path}/WaterMaterial/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    if spec.texture is None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*spec.color)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(spec.opacity)
    else:
        reader = UsdShade.Shader.Define(stage, f"{path}/WaterMaterial/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, f"{path}/WaterMaterial/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            str(resolve_effect_texture(spec.texture))
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        texture.CreateOutput("a", Sdf.ValueTypeNames.Float)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            texture.ConnectableAPI(), "a"
        )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(spec.roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(
        min(spec.roughness, 0.03)
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )


def _materialize_water_film(
    stage: Any, path: str, spec: SurfaceEffectSpec, entity: Any
) -> tuple[float, float, float]:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    center = _surface_center(spec, entity)
    points, uvs = _film_geometry(spec, center)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/Film")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(points)])
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * len(points))
    mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*spec.color)])
    mesh.CreateDisplayOpacityAttr([1.0 if spec.texture is not None else spec.opacity])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set([Gf.Vec2f(*uv) for uv in uvs])
    _bind_water_material(stage, mesh, path, spec)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
    return center


def materialize_surface_effects(
    scene: Any, dialogue: Mapping[str, Any]
) -> dict[str, str]:
    """Create declared effects and return their stable root prim paths."""
    entities = getattr(scene, "entities", None) or {}
    specs = parse_surface_effects(dialogue, entities.keys())
    if not specs:
        return {}
    register_surface_effect_centers(scene, dialogue)
    from pxr import UsdGeom

    stage = scene.stage
    root = "/World/SurfaceEffects"
    UsdGeom.Xform.Define(stage, root)
    paths = {}
    for spec in specs:
        path = f"{root}/{_safe_key(spec.key)}"
        UsdGeom.Xform.Define(stage, path)
        if spec.kind == "water_film":
            entity = entities[spec.surface]
            center = _materialize_water_film(stage, path, spec, entity)
            entity.meta.setdefault("surface_effect_centers", {})[spec.key] = center
        paths[spec.key] = path
    scene.surface_effect_prim_paths = paths
    return paths

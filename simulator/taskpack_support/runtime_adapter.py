#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Lossless runtime view of a canonical ``bts/1.0`` executable spec.

The existing Isaac builder still consumes the older flat goal/program fields.
This adapter exposes those fields without inventing a second world or answer.
The complete canonical spec remains attached as ``bts_source`` and every
materialized field is covered by a source-to-view consumption ledger.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn


ADAPTER_VERSION = "bts-runtime-view/1.7"
BTS_VERSION = "bts/1.0"
RUNTIME_EXTENSION_KEYS = frozenset({"episode_dialogue"})
RUNTIME_WORLD_EXTENSION_KEYS = frozenset(
    {
        "allow_initial_support_object_stands",
        "allow_legacy_close_navigation_preview",
        "allow_legacy_multi_shelf_preload_spots",
        "allow_narrow_support_preflight_routes",
    }
)


class BTSAdapterBlocked(ValueError):
    """The canonical spec cannot be represented without changing its meaning."""

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(f"{code} at {path}: {message}")
        self.code = code
        self.path = path
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {
            "status": "BLOCKED",
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


def canonical_digest(value: object) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def adapt_taskpack(pack: Mapping[str, Any]) -> dict[str, Any]:
    spec = _mapping(pack.get("executable_spec"), "$.executable_spec")
    task_id = _text(pack.get("task_id"), "$.task_id")
    if spec.get("task_id") != task_id:
        _blocked(
            "TASK_ID_DIVERGENCE",
            "$.executable_spec.task_id",
            "must equal TaskPack task_id",
        )
    view = adapt_bts_spec(spec)
    runtime = _mapping(pack.get("runtime"), "$.runtime")
    compatibility = runtime.get("runtime_compatibility")
    if compatibility is not None:
        extensions = _mapping(compatibility, "$.runtime.runtime_compatibility")
        unknown = set(extensions) - RUNTIME_WORLD_EXTENSION_KEYS
        if unknown:
            _blocked(
                "RUNTIME_EXTENSION_UNSUPPORTED",
                "$.runtime.runtime_compatibility",
                f"unsupported fields: {sorted(unknown)!r}",
            )
        for key, value in extensions.items():
            if value is not True:
                _blocked(
                    "RUNTIME_EXTENSION_INVALID",
                    f"$.runtime.runtime_compatibility.{key}",
                    "must be literal true",
                )
            view["world"][str(key)] = True
    slots = _taskpack_kinematic_place_slots(runtime, spec)
    visual_runtime = _taskpack_visual_runtime(runtime, spec)
    if slots or visual_runtime:
        dialogue = dict(
            _mapping(view.get("episode_dialogue", {}), "$.episode_dialogue")
        )
        if slots:
            existing = dialogue.get("kinematic_place_slots")
            if existing is not None and existing != slots:
                _blocked(
                    "KINEMATIC_PLACE_SLOT_DIVERGENCE",
                    "$.runtime.kinematic_place_slots",
                    "conflicts with the existing runtime-view extension",
                )
            dialogue["kinematic_place_slots"] = slots
        for key, value in visual_runtime.items():
            existing = dialogue.get(key)
            if existing is not None and existing != value:
                _blocked(
                    "VISUAL_RUNTIME_DIVERGENCE",
                    f"$.runtime.{key}",
                    "conflicts with the existing runtime-view extension",
                )
            dialogue[key] = value
        view["episode_dialogue"] = dialogue
    return view


def _taskpack_visual_runtime(
    runtime: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, object]:
    rows_value = runtime.get("visual_state_events")
    effects_value = runtime.get("surface_effects")
    if rows_value is None and effects_value is None:
        return {}
    objects, entities = _visual_runtime_entities(spec)
    effects = _visual_runtime_effects(effects_value)
    rows = (
        ()
        if rows_value is None
        else _sequence(rows_value, "$.runtime.visual_state_events")
    )
    declared_by_type = {
        "label_visibility": objects,
        "entity_visibility": entities,
        "surface_effect_visibility": set(effects),
        "entity_tint": entities,
    }
    for index, value in enumerate(rows):
        _validate_visual_runtime_event(value, index, declared_by_type)
    result: dict[str, object] = {}
    if rows_value is not None:
        result["visual_state_events"] = copy.deepcopy(list(rows))
    if effects_value is not None:
        result["surface_effects"] = copy.deepcopy(dict(effects))
    return result


def _visual_runtime_entities(
    spec: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    world = _mapping(spec.get("world"), "$.executable_spec.world")
    objects = set(_mapping(world.get("objects", {}), "$.executable_spec.world.objects"))
    entities = set(objects)
    for key in ("containers", "fixtures"):
        entities.update(_mapping(world.get(key, {}), f"$.executable_spec.world.{key}"))
    return objects, entities


def _visual_runtime_effects(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    effects = _mapping(value, "$.runtime.surface_effects")
    for key, effect in effects.items():
        _text(key, "$.runtime.surface_effects.<key>")
        _mapping(effect, f"$.runtime.surface_effects.{key}")
    return effects


def _validate_visual_runtime_event(
    value: object,
    index: int,
    declared_by_type: Mapping[str, set[str]],
) -> None:
    path = f"$.runtime.visual_state_events[{index}]"
    row = _mapping(value, path)
    event_type = _text(row.get("type"), f"{path}.type")
    if event_type not in declared_by_type:
        _blocked(
            "VISUAL_STATE_EVENT_INVALID",
            f"{path}.type",
            f"unsupported type {event_type!r}",
        )
    _text(row.get("after_command"), f"{path}.after_command")
    occurrence = row.get("occurrence", 1)
    if (
        isinstance(occurrence, bool)
        or not isinstance(occurrence, int)
        or occurrence < 1
    ):
        _blocked(
            "VISUAL_STATE_EVENT_INVALID",
            f"{path}.occurrence",
            "must be a positive integer",
        )
    field = {
        "label_visibility": "objects",
        "entity_visibility": "entities",
        "surface_effect_visibility": "effects",
        "entity_tint": "entities",
    }[event_type]
    target_value = row.get(field)
    if target_value is None and event_type == "entity_visibility":
        target_value = row.get("objects")
    targets = _sequence(target_value, f"{path}.{field}")
    if not targets:
        _blocked(
            "VISUAL_STATE_EVENT_INVALID",
            f"{path}.{field}",
            "must not be empty",
        )
    target_keys = [_text(target, f"{path}.{field}") for target in targets]
    unknown = sorted(set(target_keys) - declared_by_type[event_type])
    if unknown:
        _blocked(
            "VISUAL_STATE_EVENT_INVALID",
            f"{path}.{field}",
            f"references undeclared targets: {unknown!r}",
        )
    if event_type == "entity_tint":
        _validate_entity_tint_event(row, path)


def _validate_entity_tint_event(row: Mapping[str, Any], path: str) -> None:
    forbidden = {"visible", "initially_visible", "pose_from"}.intersection(row)
    if forbidden:
        _blocked(
            "VISUAL_STATE_EVENT_INVALID",
            path,
            f"entity_tint forbids fields: {sorted(forbidden)!r}",
        )
    _text(row.get("tint"), f"{path}.tint")


def _taskpack_kinematic_place_slots(
    runtime: Mapping[str, Any], spec: Mapping[str, Any]
) -> list[dict[str, str]]:
    value = runtime.get("kinematic_place_slots")
    if value is None:
        return []
    path = "$.runtime.kinematic_place_slots"
    if not isinstance(value, (list, tuple)):
        _blocked("KINEMATIC_PLACE_SLOT_INVALID", path, "must be an array")
    world = _mapping(spec.get("world"), "$.executable_spec.world")
    objects = _mapping(world.get("objects", {}), "$.executable_spec.world.objects")
    declared = set(objects)
    for key in ("containers", "fixtures"):
        declared.update(_mapping(world.get(key, {}), f"$.executable_spec.world.{key}"))
    slots: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(value):
        row_path = f"{path}[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != {"child", "parent"}:
            _blocked(
                "KINEMATIC_PLACE_SLOT_INVALID",
                row_path,
                "must contain exactly child and parent",
            )
        child = _text(raw.get("child"), f"{row_path}.child")
        parent = _text(raw.get("parent"), f"{row_path}.parent")
        pair = (child, parent)
        if child not in objects or parent not in declared or pair in seen:
            _blocked(
                "KINEMATIC_PLACE_SLOT_INVALID",
                row_path,
                "must be one unique exact pair of declared entities",
            )
        seen.add(pair)
        slots.append({"child": child, "parent": parent})
    return slots


def adapt_bts_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the flat fields read by the current builder and scorer."""
    if spec.get("spec_version") != BTS_VERSION:
        _blocked(
            "SPEC_VERSION_UNSUPPORTED",
            "$.spec_version",
            f"expected {BTS_VERSION}",
        )
    task_id = _text(spec.get("task_id"), "$.task_id")
    meta = _mapping(spec.get("meta"), "$.meta")
    world = _mapping(spec.get("world"), "$.world")
    goal = _mapping(spec.get("goal"), "$.goal")
    program = _mapping(spec.get("program"), "$.program")
    report = _mapping(spec.get("report"), "$.report")
    commands = _commands(program.get("commands"))
    human_program = _program_text(commands)
    roster = copy.deepcopy(_mapping(goal.get("roster"), "$.goal.roster"))
    goals = copy.deepcopy(_sequence(goal.get("goals"), "$.goal.goals"))
    aggregate_goals = copy.deepcopy(
        _sequence(goal.get("aggregate_goals", []), "$.goal.aggregate_goals")
    )
    answer_key = copy.deepcopy(_sequence(goal.get("answer_key"), "$.goal.answer_key"))
    negatives = copy.deepcopy(_sequence(goal.get("negatives"), "$.goal.negatives"))
    trace_outcomes = copy.deepcopy(
        _sequence(goal.get("trace_outcomes", []), "$.goal.trace_outcomes")
    )
    scene = _text(world.get("scene"), "$.world.scene")
    rooms = [str(room) for room in _sequence(world.get("rooms", []), "$.world.rooms")]
    budget = _mapping(program.get("budget"), "$.program.budget")
    objects = _mapping(world.get("objects", {}), "$.world.objects")
    containers = _mapping(world.get("containers", {}), "$.world.containers")
    fixtures = _mapping(world.get("fixtures", {}), "$.world.fixtures")
    printed = _printed_view(world.get("labels", {}), objects, containers, fixtures)
    if world.get("auto_support_labels") is False:
        printed["auto_support_labels"] = False
    view: dict[str, Any] = {
        "spec_version": "g0.1",
        "task_id": task_id,
        "source_task_id": task_id,
        "variant": spec.get("variant"),
        "family": meta.get("family"),
        "template": meta.get("template"),
        "scene": scene,
        "room": rooms[0] if rooms else None,
        "rooms": rooms,
        "wave": meta.get("wave"),
        "difficulty": meta.get("difficulty"),
        "goals": goals,
        "aggregate_goals": aggregate_goals,
        "negs": negatives,
        "trace_outcomes": trace_outcomes,
        "roster": roster,
        "answer_key": answer_key,
        "answer_key_raw": [
            [row.get("subject"), row.get("target")]
            + ([row.get("connector")] if row.get("connector") else [])
            for row in answer_key
            if isinstance(row, Mapping)
        ],
        "human_program": human_program,
        "oracle_cmds": list(human_program.split(", ")) if human_program else [],
        "cmds": _command_budget_view(budget, len(commands)),
        "object_kinds": sorted(
            {
                str(row.get("cat"))
                for row in objects.values()
                if isinstance(row, Mapping) and row.get("cat")
            }
        ),
        "unit_roles": {
            str(key): str(row.get("role"))
            for key, row in containers.items()
            if isinstance(row, Mapping) and row.get("role")
        },
        "sub_goals": copy.deepcopy(report.get("sub_goals") or []),
        "caps_by_sub_goal": copy.deepcopy(report.get("caps_by_sub_step") or []),
        "world": copy.deepcopy(world),
        "bts_source": copy.deepcopy(spec),
    }
    if printed:
        view["printed"] = printed
    view["runtime_adapter"] = {
        "version": ADAPTER_VERSION,
        "source_digest": canonical_digest(spec),
        "view_digest": canonical_digest(
            {key: value for key, value in view.items() if key != "bts_source"}
        ),
        "consumption_ledger": _consumption_ledger(
            bool(printed), world.get("auto_support_labels") is False
        ),
    }
    return view


def verify_runtime_view(view: Mapping[str, Any]) -> None:
    source = _mapping(view.get("bts_source"), "$.bts_source")
    rebuilt = adapt_bts_spec(source)
    comparable = _upgrade_legacy_single_room_view(
        _without_runtime_extensions(view, rebuilt), rebuilt
    )
    if canonical_digest(rebuilt) != canonical_digest(comparable):
        _blocked(
            "RUNTIME_VIEW_DRIFT",
            "$",
            "runtime view is not the deterministic projection of bts_source",
        )


def _without_runtime_extensions(
    view: Mapping[str, Any], rebuilt: Mapping[str, Any]
) -> Mapping[str, Any]:
    world = view.get("world")
    rebuilt_world = rebuilt.get("world")
    world_extension_keys = {
        key
        for key in RUNTIME_WORLD_EXTENSION_KEYS
        if isinstance(world, Mapping)
        and key in world
        and (not isinstance(rebuilt_world, Mapping) or key not in rebuilt_world)
    }
    if (
        not any(key in view for key in RUNTIME_EXTENSION_KEYS)
        and not world_extension_keys
    ):
        return view
    comparable = copy.deepcopy(dict(view))
    for key in RUNTIME_EXTENSION_KEYS:
        comparable.pop(key, None)
    comparable_world = dict(_mapping(comparable.get("world"), "$.world"))
    for key in world_extension_keys:
        if comparable_world.get(key) is not True:
            _blocked(
                "RUNTIME_EXTENSION_INVALID",
                f"$.world.{key}",
                "must be literal true when supplied as a runtime extension",
            )
        comparable_world.pop(key)
    comparable["world"] = comparable_world
    return comparable


def _upgrade_legacy_single_room_view(
    view: Mapping[str, Any], rebuilt: Mapping[str, Any]
) -> Mapping[str, Any]:
    if "rooms" in view or view.get("room") != rebuilt.get("room"):
        return view
    source = _mapping(view.get("bts_source"), "$.bts_source")
    world = _mapping(source.get("world"), "$.bts_source.world")
    rooms = _sequence(world.get("rooms", []), "$.bts_source.world.rooms")
    if len(rooms) > 1:
        return view
    metadata = _mapping(view.get("runtime_adapter"), "$.runtime_adapter")
    expected_payload_digest = canonical_digest(
        {
            key: value
            for key, value in view.items()
            if key not in {"bts_source", "runtime_adapter"}
        }
    )
    if (
        metadata.get("version") != ADAPTER_VERSION
        or metadata.get("source_digest") != canonical_digest(source)
        or metadata.get("view_digest") != expected_payload_digest
        or not _is_legacy_single_room_ledger(
            metadata.get("consumption_ledger"), "printed" in rebuilt
        )
    ):
        return view
    upgraded = copy.deepcopy(dict(view))
    upgraded["rooms"] = copy.deepcopy(rebuilt["rooms"])
    upgraded["runtime_adapter"] = copy.deepcopy(rebuilt["runtime_adapter"])
    return upgraded


def _is_legacy_single_room_ledger(value: object, printed: bool) -> bool:
    if not isinstance(value, list):
        return False
    pairs: list[tuple[object, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            return False
        pairs.append((row.get("source"), row.get("view")))
    expected = [
        (row["source"], row["view"])
        for row in _consumption_ledger(printed)
        if row["view"] not in {"rooms", "room"}
    ]
    expected.insert(1, ("world.rooms", "room"))
    return pairs == expected


def _commands(value: object) -> list[Mapping[str, Any]]:
    rows = _sequence(value, "$.program.commands")
    commands: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        commands.append(_mapping(row, f"$.program.commands[{index}]"))
    if not commands:
        _blocked(
            "PROGRAM_EMPTY",
            "$.program.commands",
            "an explicit corrected human branch needs at least one command",
        )
    return commands


def _program_text(commands: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for index, command in enumerate(commands):
        verb = _text(command.get("verb"), f"$.program.commands[{index}].verb")
        args = [
            _text(arg, f"$.program.commands[{index}].args[{arg_index}]")
            for arg_index, arg in enumerate(
                _sequence(command.get("args"), f"$.program.commands[{index}].args")
            )
        ]
        attach = verb.upper() in {"ASSEMBLE", "ATTACH", "INSTALL"}
        raw_indices = command.get("indices")
        if raw_indices is None:
            indices = [
                0 if attach and arg_index in {1, 3} else 1
                for arg_index in range(len(args))
            ]
        else:
            values = _sequence(raw_indices, f"$.program.commands[{index}].indices")
            if len(values) != len(args) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            ):
                _blocked(
                    "COMMAND_INDICES_INVALID",
                    f"$.program.commands[{index}].indices",
                    "must contain one non-negative integer per command argument",
                )
            indices = [int(value) for value in values]
        rendered = f"[{verb}]" + "".join(
            f" <{argument}> ({indices[arg_index]})"
            for arg_index, argument in enumerate(args)
        )
        scalar_arg = command.get("scalar_arg")
        if scalar_arg is not None:
            if args:
                _blocked(
                    "COMMAND_SCALAR_INVALID",
                    f"$.program.commands[{index}].scalar_arg",
                    "cannot be combined with entity arguments",
                )
            rendered += " " + _text(
                scalar_arg, f"$.program.commands[{index}].scalar_arg"
            )
        place_spot = _normalized_place_spot(command, f"$.program.commands[{index}]")
        if place_spot is not None:
            rendered += f" @place_spot={place_spot[0]!r}:{place_spot[1]!r}"
        lines.append(rendered)
    return ", ".join(lines)


def _normalized_place_spot(
    command: Mapping[str, Any], path: str
) -> tuple[float, float] | None:
    raw = command.get("place_spot")
    if raw is None:
        return None
    verb = str(command.get("verb") or "").upper()
    if verb not in {
        "PUT",
        "PUTBACK",
        "PUTON",
        "PUTIN",
        "PUTINSIDE",
        "INSERT",
        "FALLTOFLOOR",
    }:
        _blocked(
            "COMMAND_PLACE_SPOT_INVALID",
            f"{path}.place_spot",
            "is allowed only on placement and FallToFloor commands",
        )
    values = _sequence(raw, f"{path}.place_spot")
    if len(values) != 2 or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in values
    ):
        _blocked(
            "COMMAND_PLACE_SPOT_INVALID",
            f"{path}.place_spot",
            "must be a numeric [u, v] pair",
        )
    u, v = float(values[0]), float(values[1])
    if not math.isfinite(u) or not math.isfinite(v):
        _blocked(
            "COMMAND_PLACE_SPOT_INVALID",
            f"{path}.place_spot",
            "coordinates must be finite",
        )
    if not -1.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
        _blocked(
            "COMMAND_PLACE_SPOT_INVALID",
            f"{path}.place_spot",
            "requires -1 <= u <= 1 and 0 <= v <= 1",
        )
    return u, v


def _command_budget_view(budget: Mapping[str, Any], count: int) -> dict[str, Any]:
    grammar = int(budget.get("grammar_cmds", count))
    expanded = int(budget.get("expanded_cmds", count))
    return {
        "grammar_cmds": grammar,
        "expanded_cmds": expanded,
        "grammar_source": budget.get("grammar_source", "from_program"),
        "expanded_source": budget.get("expanded_source", "oracle_cmds"),
        "grammar_cmds_from_program": count,
        "program_tokenisable": True,
        "budget_k": budget.get("k"),
        "budget_source": budget.get("k_source"),
        "orig_oracle_cmds": count,
    }


def _printed_view(
    labels_value: object,
    objects: Mapping[str, Any],
    containers: Mapping[str, Any],
    fixtures: Mapping[str, Any],
) -> dict[str, Any]:
    labels = _mapping(labels_value, "$.world.labels")
    if not labels:
        return {}
    units: dict[str, object] = {}
    items: dict[str, object] = {}
    item_mount_faces: dict[str, str] = {}
    walls: dict[str, object] = {}
    wall_targets: set[str] = set()
    for key, value in labels.items():
        row = _mapping(value, f"$.world.labels.{key}")
        mount = _mapping(row.get("mount"), f"$.world.labels.{key}.mount")
        target = _text(mount.get("on"), f"$.world.labels.{key}.mount.on")
        content_path = f"$.world.labels.{key}.content"
        content = _mapping(row.get("content"), content_path)
        value = _label_runtime_value(
            row,
            content,
            f"$.world.labels.{key}",
        )
        surface = mount.get("surface")
        if surface == "container_sign" and target in containers:
            units[target] = value
        elif surface == "item_decal" and target in objects:
            items[target] = value
            face = mount.get("face")
            if face is not None:
                face = _text(face, f"$.world.labels.{key}.mount.face")
                if face not in {"front", "top"}:
                    _blocked(
                        "LABEL_MOUNT_FACE_UNSUPPORTED",
                        f"$.world.labels.{key}.mount.face",
                        "item decals support only front or top faces",
                    )
                item_mount_faces[target] = face
        elif surface == "wall" and target in {*fixtures, *objects, *containers}:
            if target in wall_targets:
                _blocked(
                    "LABEL_MOUNT_AMBIGUOUS",
                    f"$.world.labels.{key}.mount.on",
                    f"wall anchor {target!r} already has a wall-mounted label",
                )
            wall_targets.add(target)
            walls[str(key)] = {
                "label": value,
                "on": target,
                "side": str(mount.get("side") or "above"),
            }
        else:
            _blocked(
                "LABEL_MOUNT_UNSUPPORTED",
                f"$.world.labels.{key}.mount",
                "runtime view supports container_sign, item_decal, and wall mounts on "
                "declared fixtures, objects, or containers",
            )
    printed: dict[str, object] = {"units": units, "items": items, "sparse": True}
    if item_mount_faces:
        printed["item_mount_faces"] = item_mount_faces
    if walls:
        printed["walls"] = walls
    return printed


def _label_runtime_value(
    row: Mapping[str, Any], content: Mapping[str, Any], path: str
) -> object:
    """Keep renderer-relevant label structure while retaining the legacy string shape."""
    kind = _text(row.get("kind"), f"{path}.kind")
    args = content.get("args")
    if kind == "short_string" and not args:
        return _label_text(content, f"{path}.content")
    return {
        "kind": kind,
        "png": _text(row.get("png"), f"{path}.png"),
        "content": copy.deepcopy(dict(content)),
    }


def _label_text(content: Mapping[str, Any], path: str) -> str:
    if content.get("text") is not None:
        return _text(content.get("text"), f"{path}.text")
    if content.get("number") is not None:
        return str(content["number"])
    lines = content.get("lines")
    if isinstance(lines, list) and lines:
        return " / ".join(str(line) for line in lines)
    _blocked(
        "LABEL_CONTENT_UNSUPPORTED",
        path,
        "text, number, or non-empty lines is required for the legacy renderer",
    )


def _consumption_ledger(
    printed: bool, explicit_support_label_policy: bool = False
) -> list[dict[str, str]]:
    rows = [
        ("world.scene", "scene", "scene manifest and builder"),
        ("world.rooms", "rooms", "multi-room layout planner"),
        ("world.rooms[0]", "room", "legacy single-room consumers"),
        ("world", "world", "bts-aware world realization"),
        ("goal.goals", "goals", "shared scorer"),
        ("goal.aggregate_goals", "aggregate_goals", "shared scorer"),
        ("goal.negatives", "negs", "shared scorer"),
        ("goal.trace_outcomes", "trace_outcomes", "shared scorer"),
        ("goal.roster", "roster", "builder and scorer"),
        ("goal.answer_key", "answer_key", "oracle and audit"),
        ("program.commands", "human_program", "command compiler"),
        ("program.budget", "cmds", "budget gate"),
        ("report.sub_goals", "sub_goals", "training/eval reporting"),
    ]
    if printed:
        rows.append(("world.labels", "printed", "generic label renderer"))
    if explicit_support_label_policy:
        rows.append(
            (
                "world.auto_support_labels",
                "printed.auto_support_labels",
                "generic label renderer",
            )
        )
    return [
        {"source": source, "view": view, "consumer": consumer}
        for source, view, consumer in rows
    ]


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _blocked("EXPECTED_OBJECT", path, "must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        _blocked("EXPECTED_ARRAY", path, "must be an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _blocked("EXPECTED_TEXT", path, "must be a non-empty string")
    return value.strip()


def _blocked(code: str, path: str, message: str) -> NoReturn:
    raise BTSAdapterBlocked(code, path, message)

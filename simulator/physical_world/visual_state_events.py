# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Compile command-aligned visual state changes that are not human actions."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class LabelVisibilityEvent:
    frame: int
    objects: tuple[str, ...]
    paths: tuple[str, ...]
    visible: bool
    after_command: str
    occurrence: int
    require_offscreen: bool
    initially_visible: bool | None = None

    @property
    def event_type(self) -> str:
        return "label_visibility"

    @property
    def target_keys(self) -> tuple[str, ...]:
        return self.objects

    def manifest_row(self, *, visible_objects: Sequence[str] = ()) -> dict:
        row = {
            "type": self.event_type,
            "frame": self.frame,
            "objects": list(self.objects),
            "visible": self.visible,
            "after_command": self.after_command,
            "occurrence": self.occurrence,
            "require_offscreen": self.require_offscreen,
            "offscreen_ok": not visible_objects,
            "visible_objects_at_transition": list(visible_objects),
        }
        if self.initially_visible is not None:
            row["initially_visible"] = self.initially_visible
        return row


@dataclass(frozen=True, slots=True)
class EntityVisibilityEvent:
    frame: int
    entities: tuple[str, ...]
    paths: tuple[str, ...]
    visible: bool
    after_command: str
    occurrence: int
    require_offscreen: bool
    initially_visible: bool | None = None
    pose_from_entities: tuple[str, ...] = ()
    pose_from_paths: tuple[str, ...] = ()

    @property
    def event_type(self) -> str:
        return "entity_visibility"

    @property
    def target_keys(self) -> tuple[str, ...]:
        return self.entities

    def manifest_row(self, *, visible_objects: Sequence[str] = ()) -> dict:
        row = {
            "type": self.event_type,
            "frame": self.frame,
            "entities": list(self.entities),
            "visible": self.visible,
            "after_command": self.after_command,
            "occurrence": self.occurrence,
            "require_offscreen": self.require_offscreen,
            "offscreen_ok": not visible_objects,
            "visible_entities_at_transition": list(visible_objects),
        }
        if self.initially_visible is not None:
            row["initially_visible"] = self.initially_visible
        if self.pose_from_entities:
            row["pose_from"] = list(self.pose_from_entities)
        return row


@dataclass(frozen=True, slots=True)
class SurfaceEffectVisibilityEvent:
    frame: int
    effects: tuple[str, ...]
    paths: tuple[str, ...]
    visible: bool
    after_command: str
    occurrence: int
    require_offscreen: bool
    initially_visible: bool | None = None

    @property
    def event_type(self) -> str:
        return "surface_effect_visibility"

    @property
    def target_keys(self) -> tuple[str, ...]:
        return self.effects

    def manifest_row(self, *, visible_objects: Sequence[str] = ()) -> dict:
        row = {
            "type": self.event_type,
            "frame": self.frame,
            "effects": list(self.effects),
            "visible": self.visible,
            "after_command": self.after_command,
            "occurrence": self.occurrence,
            "require_offscreen": self.require_offscreen,
            "offscreen_ok": not visible_objects,
            "visible_effects_at_transition": list(visible_objects),
        }
        if self.initially_visible is not None:
            row["initially_visible"] = self.initially_visible
        return row


@dataclass(frozen=True, slots=True)
class EntityTintEvent:
    frame: int
    entities: tuple[str, ...]
    paths: tuple[str, ...]
    tint: str
    after_command: str
    occurrence: int
    require_offscreen: bool

    @property
    def event_type(self) -> str:
        return "entity_tint"

    @property
    def target_keys(self) -> tuple[str, ...]:
        return self.entities

    def manifest_row(self, *, visible_objects: Sequence[str] = ()) -> dict:
        return {
            "type": self.event_type,
            "frame": self.frame,
            "entities": list(self.entities),
            "tint": self.tint,
            "after_command": self.after_command,
            "occurrence": self.occurrence,
            "require_offscreen": self.require_offscreen,
            "offscreen_ok": not visible_objects,
            "visible_entities_at_transition": list(visible_objects),
        }


VisibilityEvent = (
    LabelVisibilityEvent | EntityVisibilityEvent | SurfaceEffectVisibilityEvent
)
VisualStateEvent = VisibilityEvent | EntityTintEvent


_ACTION_RE = re.compile(r"^\s*\[([^\]]+)\](.*)$")
_ARG_RE = re.compile(r"\s*<([^<>]+)>\s*\((-?\d+)\)")
_PLACE_SPOT_RE = re.compile(
    r"\s+@place_spot="
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?:"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)
_VERB_ALIASES = {
    "ANSWER_PHONE": "ANSWERPHONE",
    "ASSEMBLE": "ATTACH",
    "CLOSEDOOR": "CLOSE",
    "CLOSE_DOOR": "CLOSE",
    "DISASSEMBLE": "DETACH",
    "DROP": "RELEASE",
    "ERECT": "RAISE",
    "FALL_TO_FLOOR": "FALLTOFLOOR",
    "INSERT": "PUTINSIDE",
    "INSTALL": "ATTACH",
    "LOOKAROUND": "LOOK_AROUND",
    "LOOKAT": "LOOK_AT",
    "LOOKSCAN": "LOOK_SCAN",
    "OPEN": "OPEN",
    "OPENDOOR": "OPEN",
    "OPEN_DOOR": "OPEN",
    "PUT": "PUT",
    "PUTBACK": "PUT",
    "PUTIN": "PUTINSIDE",
    "PUTINSIDE": "PUTINSIDE",
    "PUTOBJBACK": "PUT",
    "PUTON": "PUT",
    "PUT_ON_FLOOR": "PUTONFLOOR",
    "PULLRACK": "PULL",
    "PULL_RACK": "PULL",
    "PUSHRACK": "PUSH",
    "PUSH_RACK": "PUSH",
    "REMOVE": "DETACH",
    "RUN": "WALK",
    "SQUATDOWN": "SQUAT",
    "STANDUPRIGHT": "RAISE",
    "SWITCHOFF": "TURNOFF",
    "SWITCHON": "TURNON",
    "SWITCH_OFF": "TURNOFF",
    "SWITCH_ON": "TURNON",
    "TURN_OFF": "TURNOFF",
    "TURN_ON": "TURNON",
    "TURNAROUND": "TURNAROUND",
    "TURNLEFT": "TURNLEFT",
    "TURNOVER": "FLIP",
    "TURNRIGHT": "TURNRIGHT",
    "TURN_AROUND": "TURNAROUND",
    "TURN_LEFT": "TURNLEFT",
    "TURN_OVER": "FLIP",
    "TURN_RIGHT": "TURNRIGHT",
    "TYPE": "USECOMPUTER",
    "USE_COMPUTER": "USECOMPUTER",
    "WALKFORWARD": "FORWARD",
    "WALKTOWARDS": "WALK",
    "WALK_FORWARD": "FORWARD",
}


def command_identity(command: object) -> str:
    """Canonical trigger identity, excluding executor-only placement details."""
    text = str(command or "").strip()
    match = _ACTION_RE.fullmatch(text)
    if match is None:
        return " ".join(text.split())
    verb = match.group(1).strip().upper()
    verb = _VERB_ALIASES.get(verb, verb)
    tail = _PLACE_SPOT_RE.sub("", match.group(2))

    def normalize_arg(arg_match: re.Match[str]) -> str:
        key = "_".join(arg_match.group(1).strip().lower().split())
        instance = int(arg_match.group(2))
        is_putinside_space = (
            verb == "PUTINSIDE"
            and instance == 0
            and re.search(r"(?:^|_)space\d*$", key) is not None
        )
        return "" if is_putinside_space else f" <{key}> ({instance})"

    tail = _ARG_RE.sub(normalize_arg, tail)
    tail = " ".join(tail.split())
    return f"[{verb}]" + (f" {tail}" if tail else "")


def _command_occurrences(timeline: Sequence[Mapping]) -> dict[tuple[str, int], Mapping]:
    occurrences: dict[str, int] = {}
    indexed: dict[tuple[str, int], Mapping] = {}
    for entry in timeline:
        source = command_identity(entry.get("source"))
        occurrences[source] = occurrences.get(source, 0) + 1
        indexed[(source, occurrences[source])] = entry
    return indexed


def _trigger_frame(
    row: Mapping,
    index: int,
    timeline: Mapping[tuple[str, int], Mapping],
    frame_count: int,
) -> tuple[int, str, int]:
    command = str(row.get("after_command") or "").strip()
    occurrence = int(row.get("occurrence") or 1)
    if not command or occurrence < 1:
        raise ValueError(f"visual state event {index} has an invalid trigger")
    trigger = timeline.get((command_identity(command), occurrence))
    if trigger is None:
        raise ValueError(
            f"visual state event {index} trigger was not executed: "
            f"{command!r} occurrence {occurrence}"
        )
    frame = min(max(0, int(trigger.get("frame_end") or 0)), int(frame_count))
    return frame, command, occurrence


def _target_keys(row: Mapping, index: int, event_type: str) -> tuple[str, ...]:
    fields = {
        "label_visibility": "objects",
        "entity_visibility": "entities",
        "surface_effect_visibility": "effects",
        "entity_tint": "entities",
    }
    field = fields[event_type]
    values = row.get(field)
    if values is None and event_type == "entity_visibility":
        values = row.get("objects")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"visual state event {index} {field} must be an array")
    keys = tuple(str(value) for value in values or ())
    if not keys:
        raise ValueError(f"visual state event {index} has no {field}")
    return keys


def _target_paths(
    keys: Sequence[str],
    paths_by_key: Mapping[str, str],
    index: int,
    event_type: str,
) -> tuple[str, ...]:
    missing = [key for key in keys if not paths_by_key.get(key)]
    if missing:
        target_kind = {
            "label_visibility": "decals",
            "entity_visibility": "scene prims",
            "surface_effect_visibility": "surface effects",
            "entity_tint": "scene prims",
        }[event_type]
        raise ValueError(
            f"visual state event {index} references targets without {target_kind}: "
            f"{missing}"
        )
    return tuple(str(paths_by_key[key]) for key in keys)


def _compile_event(
    row: Mapping,
    index: int,
    timeline: Mapping[tuple[str, int], Mapping],
    label_paths: Mapping[str, str],
    entity_paths: Mapping[str, str],
    effect_paths: Mapping[str, str],
    frame_count: int,
) -> VisualStateEvent:
    event_type = str(row.get("type") or "")
    if event_type not in {
        "label_visibility",
        "entity_visibility",
        "surface_effect_visibility",
        "entity_tint",
    }:
        raise ValueError(f"visual state event {index} has unsupported type")
    frame, command, occurrence = _trigger_frame(row, index, timeline, frame_count)
    keys = _target_keys(row, index, event_type)
    path_lookup = {
        "label_visibility": label_paths,
        "entity_visibility": entity_paths,
        "surface_effect_visibility": effect_paths,
        "entity_tint": entity_paths,
    }[event_type]
    paths = _target_paths(keys, path_lookup, index, event_type)
    common = {
        "frame": frame,
        "paths": paths,
        "visible": bool(row.get("visible", False)),
        "after_command": command,
        "occurrence": occurrence,
        "require_offscreen": bool(row.get("require_offscreen", False)),
        "initially_visible": (
            bool(row["initially_visible"]) if "initially_visible" in row else None
        ),
    }
    if event_type == "label_visibility":
        return LabelVisibilityEvent(objects=keys, **common)
    if event_type == "entity_tint":
        forbidden = {"visible", "initially_visible", "pose_from"}.intersection(row)
        if forbidden:
            raise ValueError(
                f"visual state event {index} entity_tint forbids fields: "
                f"{sorted(forbidden)!r}"
            )
        tint = str(row.get("tint") or "").strip()
        if not tint:
            raise ValueError(f"visual state event {index} entity_tint has no tint")
        return EntityTintEvent(
            frame=frame,
            entities=keys,
            paths=paths,
            tint=tint,
            after_command=command,
            occurrence=occurrence,
            require_offscreen=bool(row.get("require_offscreen", False)),
        )
    if event_type == "entity_visibility":
        raw_pose_from = row.get("pose_from")
        if raw_pose_from is None:
            pose_from_entities: tuple[str, ...] = ()
        elif isinstance(raw_pose_from, str):
            pose_from_entities = (raw_pose_from,)
        else:
            pose_from_entities = tuple(str(value) for value in raw_pose_from)
        if pose_from_entities and not common["visible"]:
            raise ValueError(
                f"visual state event {index} can transfer a pose only while revealing an entity"
            )
        if pose_from_entities and len(pose_from_entities) != len(keys):
            raise ValueError(
                f"visual state event {index} needs one pose_from entity per target"
            )
        pose_from_paths = _target_paths(
            pose_from_entities,
            entity_paths,
            index,
            event_type,
        )
        return EntityVisibilityEvent(
            entities=keys,
            pose_from_entities=pose_from_entities,
            pose_from_paths=pose_from_paths,
            **common,
        )
    return SurfaceEffectVisibilityEvent(effects=keys, **common)


def _validate_initial_visibility(events: Sequence[VisualStateEvent]) -> None:
    by_path: dict[str, bool] = {}
    for event in events:
        if isinstance(event, EntityTintEvent) or event.initially_visible is None:
            continue
        for path in event.paths:
            previous = by_path.setdefault(path, event.initially_visible)
            if previous != event.initially_visible:
                raise ValueError(
                    f"visual state events disagree on initial visibility for {path}"
                )


def compile_visual_state_events(
    dialogue: Mapping,
    timeline: Sequence[Mapping],
    label_paths: Mapping[str, str],
    entity_paths: Mapping[str, str],
    frame_count: int,
    *,
    effect_paths: Mapping[str, str] | None = None,
) -> tuple[VisualStateEvent, ...]:
    """Resolve label and entity events to exclusive command-end frame indices."""
    rows = list(dialogue.get("visual_state_events") or [])
    if not rows:
        return ()
    indexed = _command_occurrences(timeline)
    events = tuple(
        _compile_event(
            row,
            index,
            indexed,
            label_paths,
            entity_paths,
            effect_paths or {},
            frame_count,
        )
        for index, row in enumerate(rows)
    )
    _validate_initial_visibility(events)
    return tuple(sorted(events, key=lambda event: event.frame))


def compile_label_visibility_events(
    dialogue: Mapping,
    timeline: Sequence[Mapping],
    label_paths: Mapping[str, str],
    frame_count: int,
) -> tuple[LabelVisibilityEvent, ...]:
    """Resolve legacy label-only events without changing the public API."""
    events = compile_visual_state_events(
        dialogue, timeline, label_paths, {}, frame_count
    )
    if any(not isinstance(event, LabelVisibilityEvent) for event in events):
        raise ValueError("compile_label_visibility_events only accepts label events")
    return tuple(event for event in events if isinstance(event, LabelVisibilityEvent))


def visibility_at(events: Sequence[VisualStateEvent], frame: int) -> dict[str, bool]:
    """Return the declared visibility state for every affected USD prim path."""
    state: dict[str, bool] = {}
    for event in events:
        if isinstance(event, EntityTintEvent) or event.initially_visible is None:
            continue
        for path in event.paths:
            state.setdefault(path, event.initially_visible)
    for event in events:
        if event.frame > frame:
            break
        if isinstance(event, EntityTintEvent):
            continue
        for path in event.paths:
            state[path] = event.visible
    return state


def tints_at(events: Sequence[VisualStateEvent], frame: int) -> dict[str, str]:
    """Return the most recent declared tint for each affected entity path."""
    state: dict[str, str] = {}
    for event in events:
        if event.frame > frame:
            break
        if not isinstance(event, EntityTintEvent):
            continue
        for path in event.paths:
            state[path] = event.tint
    return state


class OnlineVisualStateEvents:
    """Advance visual state from successful online commands, without an oracle trace."""

    def __init__(
        self,
        dialogue: Mapping,
        label_paths: Mapping[str, str],
        entity_paths: Mapping[str, str],
        *,
        effect_paths: Mapping[str, str] | None = None,
    ) -> None:
        rows = list(dialogue.get("visual_state_events") or [])
        templates: list[VisualStateEvent] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"visual state event {index} must be an object")
            command = str(row.get("after_command") or "").strip()
            occurrence = int(row.get("occurrence") or 1)
            if not command or occurrence < 1:
                raise ValueError(f"visual state event {index} has an invalid trigger")
            timeline = {(command_identity(command), occurrence): {"frame_end": 0}}
            templates.append(
                _compile_event(
                    row,
                    index,
                    timeline,
                    label_paths,
                    entity_paths,
                    effect_paths or {},
                    0,
                )
            )
        _validate_initial_visibility(templates)
        self._templates = tuple(templates)
        self.reset()

    @property
    def templates(self) -> tuple[VisualStateEvent, ...]:
        return self._templates

    def __bool__(self) -> bool:
        return bool(self._templates)

    @property
    def events(self) -> tuple[VisualStateEvent, ...]:
        return tuple(self._events)

    def reset(self) -> None:
        self._occurrences: dict[str, int] = {}
        self._events: list[VisualStateEvent] = []

    def note_command(
        self, command: object, *, succeeded: bool, frame: int
    ) -> tuple[VisualStateEvent, ...]:
        if not succeeded:
            return ()
        identity = command_identity(command)
        occurrence = self._occurrences.get(identity, 0) + 1
        self._occurrences[identity] = occurrence
        triggered = tuple(
            replace(event, frame=max(0, int(frame)))
            for event in self._templates
            if command_identity(event.after_command) == identity
            and event.occurrence == occurrence
        )
        self._events.extend(triggered)
        self._events.sort(key=lambda event: event.frame)
        return triggered

    def visibility_at(self, frame: int) -> dict[str, bool]:
        initial = visibility_at(self._templates, -1)
        initial.update(visibility_at(self._events, frame))
        return initial

    def tints_at(self, frame: int) -> dict[str, str]:
        return tints_at(self._events, frame)

    def hidden_entity_keys(self) -> set[str]:
        state: dict[str, bool] = {}
        for event in self._templates:
            if not isinstance(event, EntityVisibilityEvent):
                continue
            if event.initially_visible is not None:
                for key in event.entities:
                    state.setdefault(key, event.initially_visible)
        for event in self._events:
            if not isinstance(event, EntityVisibilityEvent):
                continue
            for key in event.entities:
                state[key] = event.visible
        return {key for key, visible in state.items() if not visible}


def label_visibility_at(
    events: Sequence[LabelVisibilityEvent], frame: int
) -> dict[str, bool]:
    """Return label visibility while preserving the legacy public API."""
    return visibility_at(events, frame)

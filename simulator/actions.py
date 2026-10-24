# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Typed semantic actions for driving the InteractiveWorld human avatar."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import SimulatorConfigError
from .physical_world.runtime_capabilities import ACTIONS, IMPLEMENTED, normalize


_COMMAND = re.compile(r"^\s*\[([^\[\]]+)\]\s*(.*?)\s*$")
_REFERENCE = re.compile(r"<([^<>]+)>\s*\(([0-9]+)\)")
# Egocentric locomotion and Wait take an optional number with a unit instead of objects.
_SCALAR = re.compile(
    r"^[-+]?(?:\d+\.?\d*|\.\d+)\s*"
    r"(?:m|metres?|meters?|steps?|deg|degrees?|s|secs?|seconds?)?$",
    re.IGNORECASE,
)
_SCALAR_PRIMITIVES = frozenset({"forward", "turn", "wait"})


@dataclass(frozen=True)
class ActionSpec:
    """One action family supported by the semantic executor."""

    primitive: str
    verbs: tuple[str, ...]
    arities: tuple[int, ...]
    affordances: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class Action:
    """A high-level human action with stable object references.

    ``Action`` is the preferred public representation. Existing VirtualHome-style
    command strings remain accepted through :meth:`parse` and :meth:`to_command`.
    """

    verb: str
    arguments: tuple[str, ...] = ()
    instances: tuple[int, ...] = ()
    scalar: str = ""

    def __post_init__(self) -> None:
        verb = self.verb.strip()
        scalar = str(self.scalar).strip()
        arguments = tuple(value.strip() for value in self.arguments)
        instances = self.instances or tuple(1 for _ in arguments)
        if not verb or any(character in verb for character in "[]<>"):
            raise SimulatorConfigError("action verb is empty or contains delimiters")
        if any(not value or "<" in value or ">" in value for value in arguments):
            raise SimulatorConfigError("action arguments must be non-empty object keys")
        if len(instances) != len(arguments):
            raise SimulatorConfigError(
                "action instances must have the same length as arguments"
            )
        if any(isinstance(value, bool) or value < 1 for value in instances):
            raise SimulatorConfigError("action instance indices must be positive")
        capabilities = {
            normalize(alias): capability
            for capability in ACTIONS
            if capability.status == IMPLEMENTED
            for alias in capability.verbs
        }
        capability = capabilities.get(normalize(verb))
        if capability is None:
            raise SimulatorConfigError(f"unsupported action verb: {verb!r}")
        if len(arguments) not in capability.arities:
            raise SimulatorConfigError(
                f"action {verb!r} expects {capability.arities} object "
                f"reference(s), got {len(arguments)}"
            )
        if scalar and (
            capability.primitive not in _SCALAR_PRIMITIVES
            or normalize(verb) == "turnaround"
            or not _SCALAR.fullmatch(scalar)
        ):
            raise SimulatorConfigError(
                f"action {verb!r} does not take the scalar argument {scalar!r}"
            )
        object.__setattr__(self, "verb", verb)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "instances", tuple(int(value) for value in instances))
        object.__setattr__(self, "scalar", scalar)

    @classmethod
    def parse(cls, command: str) -> Action:
        """Parse ``[Verb] <object> (instance) ...`` or a scalar form such as ``[Wait] 3s``."""
        match = _COMMAND.fullmatch(command)
        if match is None:
            raise SimulatorConfigError(f"invalid semantic action: {command!r}")
        tail = match.group(2)
        references = list(_REFERENCE.finditer(tail))
        residue = _REFERENCE.sub("", tail).strip()
        if residue and (references or not _SCALAR.fullmatch(residue)):
            raise SimulatorConfigError(
                f"invalid semantic action arguments: {residue!r}"
            )
        return cls(
            match.group(1),
            tuple(reference.group(1) for reference in references),
            tuple(int(reference.group(2)) for reference in references),
            residue,
        )

    def to_command(self) -> str:
        parts = [f"[{self.verb}]"]
        parts.extend(
            f"<{argument}> ({instance})"
            for argument, instance in zip(self.arguments, self.instances)
        )
        if self.scalar:
            parts.append(self.scalar)
        return " ".join(parts)

    @classmethod
    def walk(cls, target: str) -> Action:
        return cls("Walk", (target,))

    @classmethod
    def walk_forward(cls, meters: float | None = None) -> Action:
        return cls("WalkForward", scalar=f"{meters:g}m" if meters is not None else "")

    @classmethod
    def turn_left(cls, degrees: float | None = None) -> Action:
        return cls("TurnLeft", scalar=f"{degrees:g}" if degrees is not None else "")

    @classmethod
    def turn_right(cls, degrees: float | None = None) -> Action:
        return cls("TurnRight", scalar=f"{degrees:g}" if degrees is not None else "")

    @classmethod
    def look_around(cls) -> Action:
        return cls("LookAround")

    @classmethod
    def look_at(cls, target: str) -> Action:
        return cls("LookAt", (target,))

    @classmethod
    def grab(cls, target: str) -> Action:
        return cls("Grab", (target,))

    @classmethod
    def put_on(cls, item: str, target: str) -> Action:
        return cls("PutOn", (item, target))

    @classmethod
    def put_inside(cls, item: str, target: str) -> Action:
        return cls("PutInside", (item, target))

    @classmethod
    def open(cls, target: str) -> Action:
        return cls("Open", (target,))

    @classmethod
    def close(cls, target: str) -> Action:
        return cls("Close", (target,))

    @classmethod
    def turn_on(cls, target: str) -> Action:
        return cls("TurnOn", (target,))

    @classmethod
    def turn_off(cls, target: str) -> Action:
        return cls("TurnOff", (target,))

    @classmethod
    def wait(cls, seconds: float | None = None) -> Action:
        return cls("Wait", scalar=f"{seconds:g}s" if seconds is not None else "")


def action_capabilities(*, include_unavailable: bool = False) -> tuple[ActionSpec, ...]:
    """Return the executor's canonical action catalog without importing Isaac."""
    return tuple(
        ActionSpec(
            primitive=capability.primitive,
            verbs=capability.verbs,
            arities=capability.arities,
            affordances=capability.affordances,
            status=capability.status,
        )
        for capability in ACTIONS
        if include_unavailable or capability.status == IMPLEMENTED
    )


__all__ = ["Action", "ActionSpec", "action_capabilities"]

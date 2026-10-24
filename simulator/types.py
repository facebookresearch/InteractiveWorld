# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Immutable public result types for the InteractiveWorld simulator."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import SimulatorProtocolError


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _decode_frame(value: object) -> bytes | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise SimulatorProtocolError("observation frame must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise SimulatorProtocolError("observation frame is not valid base64") from error


@dataclass(frozen=True)
class Relation:
    predicate: str
    subject: str
    target: str | None


@dataclass(frozen=True)
class ActionOutcome:
    primitive: str
    ok: bool
    reason: str
    detail: str

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ActionOutcome:
        return cls(
            primitive=str(value.get("primitive") or ""),
            ok=bool(value.get("ok")),
            reason=str(value.get("reason") or ""),
            detail=str(value.get("detail") or ""),
        )


@dataclass(frozen=True)
class Score:
    success: bool
    scoreable: bool
    n_done: int
    n_total: int
    progress: float
    prohibitions_ok: bool
    details: Mapping[str, Any]

    @classmethod
    def from_payload(cls, value: object) -> Score:
        payload = _mapping(value)
        return cls(
            success=bool(payload.get("success")),
            scoreable=bool(payload.get("scoreable")),
            n_done=int(payload.get("n_done") or 0),
            n_total=int(payload.get("n_total") or 0),
            progress=float(payload.get("progress") or 0.0),
            prohibitions_ok=payload.get("prohibitions_ok") is not False,
            details=dict(payload),
        )


@dataclass(frozen=True)
class Observation:
    """Agent-safe observation by default; privileged state is opt-in."""

    task_id: str
    scene: str
    taskpack_sha256: str | None
    taskpack_digest: str | None
    task: str
    task_completed: bool
    held: tuple[str, ...]
    at: tuple[str, ...]
    objects: tuple[str, ...]
    objects_are_visible_only: bool
    put_verb: str
    frame_png: bytes | None
    frames_png: tuple[bytes, ...]
    frame_segments: tuple[str, ...]
    action_trace: tuple[Mapping[str, Any], ...]
    visibility: Mapping[str, Any] | None
    relations: tuple[Relation, ...]
    score: Score | None

    @classmethod
    def from_payload(
        cls, value: Mapping[str, Any], *, profile: str = "agent"
    ) -> Observation:
        if profile not in {"agent", "debug"}:
            raise SimulatorProtocolError(
                "observation profile must be 'agent' or 'debug'"
            )
        frames = tuple(
            frame
            for frame in (
                _decode_frame(item) for item in _sequence(value.get("frames"))
            )
            if frame is not None
        )
        relations: tuple[Relation, ...] = ()
        score: Score | None = None
        if profile == "debug":
            relation_rows: list[Relation] = []
            for row in _sequence(value.get("relations")):
                parts = _sequence(row)
                if len(parts) != 3:
                    raise SimulatorProtocolError(
                        "observation relation must have three fields"
                    )
                relation_rows.append(
                    Relation(
                        predicate=str(parts[0]),
                        subject=str(parts[1]),
                        target=None if parts[2] is None else str(parts[2]),
                    )
                )
            relations = tuple(relation_rows)
            score = Score.from_payload(value.get("score"))
        visibility_value = value.get("vischeck")
        visibility = (
            dict(visibility_value) if isinstance(visibility_value, Mapping) else None
        )
        return cls(
            task_id=str(value.get("task_id") or ""),
            scene=str(value.get("scene") or ""),
            taskpack_sha256=(
                str(value["taskpack_sha256"]) if value.get("taskpack_sha256") else None
            ),
            taskpack_digest=(
                str(value["taskpack_digest"]) if value.get("taskpack_digest") else None
            ),
            task=str(value.get("task") or ""),
            task_completed=bool(value.get("task_completed")),
            held=tuple(str(item) for item in _sequence(value.get("held"))),
            at=tuple(str(item) for item in _sequence(value.get("at"))),
            objects=tuple(str(item) for item in _sequence(value.get("objects"))),
            objects_are_visible_only=bool(value.get("objects_are_visible_only")),
            put_verb=str(value.get("put_verb") or "PutInside"),
            frame_png=_decode_frame(value.get("frame")),
            frames_png=frames,
            frame_segments=tuple(
                str(item) for item in _sequence(value.get("frame_segs"))
            ),
            action_trace=tuple(
                dict(item)
                for item in _sequence(value.get("trace_events"))
                if isinstance(item, Mapping)
            ),
            visibility=visibility,
            relations=relations,
            score=score,
        )


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    outcomes: tuple[ActionOutcome, ...]
    terminated: bool
    truncated: bool
    score: Score | None
    timings: Mapping[str, Any]

    @property
    def success(self) -> bool:
        return bool(self.outcomes) and all(outcome.ok for outcome in self.outcomes)

    @property
    def error_code(self) -> str | None:
        return next(
            (outcome.reason for outcome in self.outcomes if not outcome.ok), None
        )

    @property
    def error_message(self) -> str | None:
        return next(
            (outcome.detail for outcome in self.outcomes if not outcome.ok), None
        )

    @classmethod
    def from_payload(
        cls,
        value: Mapping[str, Any],
        *,
        profile: str = "agent",
        meta: Mapping[str, Any] | None = None,
        truncated: bool = False,
    ) -> StepResult:
        raw_score = Score.from_payload(value.get("score"))
        outcomes = tuple(
            ActionOutcome.from_payload(item)
            for item in _sequence(value.get("outcomes"))
            if isinstance(item, Mapping)
        )
        return cls(
            observation=Observation.from_payload(value, profile=profile),
            outcomes=outcomes,
            terminated=bool(value.get("task_completed")) or raw_score.success,
            truncated=truncated,
            score=raw_score if profile == "debug" else None,
            timings=dict(_mapping(meta).get("phases") or {}),
        )


@dataclass(frozen=True)
class SimulatorInfo:
    protocol_version: str
    simulator_version: str
    taskpack_versions: tuple[str, ...]
    runtime_contract_versions: tuple[str, ...]
    operations: tuple[str, ...]
    action_primitives: tuple[str, ...]
    ready: bool
    render_observations: bool
    gpu: int
    observation_profile: str

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> SimulatorInfo:
        return cls(
            protocol_version=str(value.get("protocol_version") or ""),
            simulator_version=str(value.get("simulator_version") or ""),
            taskpack_versions=tuple(
                str(item) for item in _sequence(value.get("taskpack_versions"))
            ),
            runtime_contract_versions=tuple(
                str(item) for item in _sequence(value.get("runtime_contract_versions"))
            ),
            operations=tuple(str(item) for item in _sequence(value.get("operations"))),
            action_primitives=tuple(
                str(item) for item in _sequence(value.get("action_primitives"))
            ),
            ready=bool(value.get("ready")),
            render_observations=bool(value.get("render_observations")),
            gpu=int(value.get("gpu") or 0),
            observation_profile=str(value.get("observation_profile") or ""),
        )


__all__ = [
    "ActionOutcome",
    "Observation",
    "Relation",
    "Score",
    "SimulatorInfo",
    "StepResult",
]

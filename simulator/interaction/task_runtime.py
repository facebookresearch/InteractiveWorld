# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Validated per-task runtime inputs shared by generated training episodes."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from simulator.taskpack_support.runtime_adapter import (
    adapt_taskpack,
    BTSAdapterBlocked,
    canonical_digest,
    verify_runtime_view,
)


class TaskRuntimeError(ValueError):
    """A generated task cannot be executed without changing its authored meaning."""


def decode_extra_info(value: Any) -> dict[str, Any]:
    """Decode the logical row carried either directly or through Parquet JSON."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise TaskRuntimeError(f"extra_info is not valid JSON: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise TaskRuntimeError(
                f"extra_info JSON must contain an object, got {type(decoded).__name__}"
            )
        return dict(decoded)
    raise TaskRuntimeError(
        f"extra_info must be an object or JSON string, got {type(value).__name__}"
    )


@dataclass(frozen=True)
class TaskRuntime:
    """The runtime projection consumed by the training agent loop."""

    assistant_goal: str
    prologue_actions: tuple[str, ...]
    prologue_say: str
    tick_budget: int
    human_policy: Mapping[str, Any]
    frozen_taskpack: Mapping[str, Any] | None = None

    @classmethod
    def from_extra_info(
        cls, extra_info: Mapping[str, Any], default_tick_budget: int
    ) -> TaskRuntime:
        """Parse the projector's training view without re-interpreting the story."""
        frozen = _mapping(extra_info.get("frozen_taskpack"), "frozen_taskpack")
        if frozen:
            return cls._from_frozen(extra_info, frozen, default_tick_budget)
        authored_goal = str(extra_info.get("authored_goal") or "").strip()
        prologue = _mapping(extra_info.get("prologue"), "prologue")
        actions = _actions(prologue.get("actions"))
        prologue_say = str(prologue.get("say") or "").strip()
        human_policy = _mapping(extra_info.get("human_policy"), "human_policy")
        tick_budget = _positive_int(
            extra_info.get("tick_budget", default_tick_budget), "tick_budget"
        )
        return cls(
            assistant_goal=authored_goal,
            prologue_actions=actions,
            prologue_say=prologue_say,
            tick_budget=tick_budget,
            human_policy=human_policy,
        )

    @classmethod
    def _from_frozen(
        cls,
        extra_info: Mapping[str, Any],
        frozen: Mapping[str, Any],
        default_tick_budget: int,
    ) -> TaskRuntime:
        pack = _source_taskpack(frozen)
        expected_digest = digest_json(pack)
        _same(
            expected_digest,
            frozen.get("taskpack_digest"),
            "frozen_taskpack.taskpack_digest",
        )
        if extra_info.get("taskpack_digest") is not None:
            _same(expected_digest, extra_info.get("taskpack_digest"), "taskpack_digest")
        _, expected_spec_digest = builder_spec_from_taskpack(pack)
        _same(
            expected_spec_digest,
            frozen.get("builder_spec_digest"),
            "frozen_taskpack.builder_spec_digest",
        )
        if extra_info.get("builder_spec_digest") is not None:
            _same(
                expected_spec_digest,
                extra_info.get("builder_spec_digest"),
                "builder_spec_digest",
            )
        story = _mapping(pack["story"], "frozen_taskpack.story")
        runtime = _mapping(pack["runtime"], "frozen_taskpack.runtime")
        goal = str(story.get("assistant_goal") or "").strip()
        prologue = _mapping(story.get("prologue"), "frozen_taskpack.story.prologue")
        human_policy = _mapping(
            runtime.get("human_policy"), "frozen_taskpack.runtime.human_policy"
        )
        ticks = _positive_int(
            runtime.get("tick_budget", default_tick_budget),
            "frozen_taskpack.runtime.tick_budget",
        )
        _same(goal, extra_info.get("authored_goal"), "authored_goal")
        _same(prologue, extra_info.get("prologue"), "prologue")
        _same(human_policy, extra_info.get("human_policy"), "human_policy")
        _same(ticks, extra_info.get("tick_budget"), "tick_budget")
        scene = f"spec:{pack['task_id']}"
        _same(scene, extra_info.get("scene"), "scene")
        frozen_spec = _mapping(extra_info.get("frozen_spec"), "frozen_spec")
        _same(scene, frozen_spec.get("scene"), "frozen_spec.scene")
        return cls(
            assistant_goal=goal,
            prologue_actions=_actions(prologue.get("actions")),
            prologue_say=str(prologue.get("say") or "").strip(),
            tick_budget=ticks,
            human_policy=human_policy,
            frozen_taskpack=copy.deepcopy(pack),
        )

    def assistant_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return the assistant view; the answer-bearing checklist stays hidden."""
        if not self.assistant_goal:
            return dict(observation)
        return {**observation, "task": self.assistant_goal, "goal": []}

    def prologue(self, fallback: Sequence[str]) -> tuple[str, ...]:
        """Use the authored opening verbatim, or preserve the legacy generic opening."""
        if self.frozen_taskpack is not None:
            return self.prologue_actions
        return self.prologue_actions or tuple(fallback)


def assert_unscored_state(observation: Mapping[str, Any], phase: str) -> None:
    """Reject a start or opening beat that already earns task credit."""
    score = _mapping(observation.get("score"), "score")
    if not score:
        return
    if bool(score.get("success")) or int(score.get("n_done") or 0) > 0:
        raise TaskRuntimeError(
            f"{phase} already satisfies {int(score.get('n_done') or 0)} scored "
            "achievement(s); the assistant would receive free reward"
        )
    if score.get("prohibitions_ok") is False:
        raise TaskRuntimeError(
            f"{phase} violates a task prohibition before the assistant can act"
        )


def digest_json(value: Any) -> str:
    """Projector-compatible canonical digest."""
    try:
        body = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TaskRuntimeError(f"frozen_taskpack is not canonical JSON: {exc}") from exc
    return f"sha256:{hashlib.sha256(body.encode('utf-8')).hexdigest()}"


def builder_spec_from_taskpack(
    frozen_taskpack: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Adapt and verify the exact builder payload carried by a frozen TaskPack."""
    try:
        builder_spec = adapt_taskpack(frozen_taskpack)
        verify_runtime_view(builder_spec)
    except BTSAdapterBlocked as exc:
        raise TaskRuntimeError(
            f"frozen_taskpack executable_spec is not runtime-buildable: {exc}"
        ) from exc
    return builder_spec, canonical_digest(builder_spec)


def _source_taskpack(frozen: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "taskpack_version",
        "task_id",
        "executable_spec",
        "story",
        "runtime",
        "lineage",
    )
    missing = [field for field in fields if field not in frozen]
    if missing:
        raise TaskRuntimeError(f"frozen_taskpack is missing {missing}")
    return {field: frozen[field] for field in fields}


def _same(expected: Any, actual: Any, field: str) -> None:
    if actual != expected:
        raise TaskRuntimeError(
            f"{field} drifted from frozen_taskpack: expected {expected!r}, got {actual!r}"
        )


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TaskRuntimeError(f"{field} must be an object, got {type(value).__name__}")
    return dict(value)


def _actions(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskRuntimeError("prologue.actions must be an array of command strings")
    actions = tuple(str(action).strip() for action in value)
    if any(not action for action in actions):
        raise TaskRuntimeError("prologue.actions contains an empty command")
    return actions


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TaskRuntimeError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TaskRuntimeError(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise TaskRuntimeError(f"{field} must be a positive integer")
    return result

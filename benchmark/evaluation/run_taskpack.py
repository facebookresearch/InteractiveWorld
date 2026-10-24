#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run an InteractiveWorld TaskPack through the released two-agent simulator.

The TaskPack owns the human-policy branch. A configured Human model realizes every
non-trivial human action, while the evaluated assistant sees only egocentric frames
and human speech through the released interaction protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RELEASE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)

from simulator.environment.runtime_builder_spec import (  # noqa: E402
    register_runtime_builder_spec,
)
from simulator.interaction.assistant_backend import (  # noqa: E402
    AssistantBackend,
    AssistantBackendError,
    default_provider_base_url,
    make_backend,
    make_human_gateway,
    normalize_provider,
    resolve_api_key_env,
)
from simulator.interaction.human_policy_runtime import HumanPolicyBlocked  # noqa: E402
from simulator.interaction.task_runtime import (  # noqa: E402
    assert_unscored_state,
    builder_spec_from_taskpack,
    digest_json,
    TaskRuntime,
)
from simulator.interaction.taskpack_human_runtime import (  # noqa: E402
    AuthoredHumanBeat,
    HumanModelSemanticVerifier,
    observation_world_predicate,
    TaskPackHumanRuntime,
)
from simulator.interaction.taskpack_interaction import (  # noqa: E402
    adapt_taskpack,
    TaskPackInteractionSession,
    TaskPackInteractionSpec,
)

from ..tasks.validate import (  # noqa: E402
    validate as validate_release_task_manifest,
    ValidationError as TaskManifestValidationError,
)
from .artifact_bundle import (  # noqa: E402
    compute_run_evidence_sha256,
    reset_task_artifacts,
    validate_artifact_tag,
    verify_run_evidence_sha256,
)
from .rubric import load_rubric, task_index, validate_task_manifest  # noqa: E402

_HUMAN_VISUAL_MEMORY_INTERVALS: int = 2
_ORACLE_INTERNAL_KEY_PATTERN = r"[A-Za-z][A-Za-z0-9_-]*"
_ORACLE_COMMAND_PATTERN = (
    r"`?\[(?P<verb>[A-Za-z]+)\]"
    r"(?P<args>(?:\s*<[A-Za-z][A-Za-z0-9_-]*>\s*\(-?\d+\))*)`?"
)
_ORACLE_COMMAND_RE = re.compile(_ORACLE_COMMAND_PATTERN)
_ORACLE_COMMAND_ARG_RE = re.compile(
    rf"<(?P<key>{_ORACLE_INTERNAL_KEY_PATTERN})>\s*\(-?\d+\)"
)
_ORACLE_USE_COMMAND_RE = re.compile(
    rf"\b(?P<lead>Use|use)\s+(?P<command>{_ORACLE_COMMAND_PATTERN})"
)
_ORACLE_PARENTHETICAL_KEY_RE = re.compile(
    rf"\s*\(\s*`(?P<key>{_ORACLE_INTERNAL_KEY_PATTERN})`\s*\)"
)
_ORACLE_BACKTICKED_KEY_RE = re.compile(rf"`(?P<key>{_ORACLE_INTERNAL_KEY_PATTERN})`")
_ORACLE_CONTROL_KEY_CLAUSE_RE = re.compile(
    rf",\s*(?:and\s+)?(?:its|whose)\s+(?:simulator\s+)?control key is\s+"
    rf"`(?P<key>{_ORACLE_INTERNAL_KEY_PATTERN})`",
    re.IGNORECASE,
)
_ORACLE_PRIVATE_SENTENCE_RE = re.compile(
    r"(?:(?<=^)|(?<=[.!?]))\s*(?:Treat\b|Its\b)[^.!?]*"
    r"(?:control key|execution handle)[^.!?]*[.!?]",
    re.IGNORECASE,
)
_ORACLE_USING_KEY_RE = re.compile(
    rf"\s+using\s+`(?P<key>{_ORACLE_INTERNAL_KEY_PATTERN})`",
    re.IGNORECASE,
)
_ORACLE_USING_WALK_RE = re.compile(
    r"\s+using\s+(`?\[Walk\](?:\s*<[^>]+>\s*\(-?\d+\))*`?)",
    re.IGNORECASE,
)
_ORACLE_WITH_LOOKSCAN_RE = re.compile(
    r"\s+with\s+(`?\[LookScan\]`?)",
    re.IGNORECASE,
)
_ORACLE_PRIVATE_INTERFACE_RE = re.compile(
    r"\b(?:command key|simulator control key|execution handle|motor legend)\b",
    re.IGNORECASE,
)


class TaskPackEvalError(RuntimeError):
    """The episode cannot be attributed to the evaluated assistant."""


def _assistant_output_protocol_metadata(
    args: argparse.Namespace,
) -> dict[str, str | None]:
    del args
    return {
        "name": "direct",
        "version": None,
        "prompt_sha256": None,
    }


def _oracle_public_referent_before(prefix: str) -> str:
    segment = re.split(r"[.!?;:,—()]", prefix)[-1].strip()
    determiners = list(
        re.finditer(
            r"\b(?:the|The|that|That|this|This|these|These|those|Those|an|An|a)\b",
            segment,
        )
    )
    if determiners:
        referent = segment[determiners[0].start() :].strip()
    else:
        labels = list(
            re.finditer(
                r"\b(?:ORDER|UNIT)\s+[A-Z](?:\s+\d+\s*mL)?\b",
                segment,
                re.IGNORECASE,
            )
        )
        if not labels:
            return ""
        referent = labels[-1].group(0)
    referent = re.sub(r"^(?:the|The)\s+same\s+side\s+of\s+", "the ", referent)
    referent = re.sub(
        r"^(?:the|The|that|That)\s+same\s+viewpoint\s+at\s+",
        "the ",
        referent,
    )
    if referent.upper().startswith("ORDER "):
        referent = f"the {referent} box"
    if referent.startswith("The "):
        referent = f"the {referent[4:]}"
    return referent


def _oracle_default_public_referent(key: str, descriptions: Mapping[str, str]) -> str:
    description = descriptions.get(key)
    if not description:
        raise TaskPackEvalError(
            f"opaque oracle dialogue references unknown internal key {key!r}"
        )
    description = re.sub(r",\s*in\s+[^,]+\s+\d+$", "", description)
    description = description.replace(", labelled ", " labeled ")
    if re.match(r"(?i)^(?:the|a|an|unit|order)\b", description):
        return description
    return f"the {description}"


def _oracle_command_public_text(
    verb: str,
    raw_args: str,
    referents: Mapping[str, str],
    descriptions: Mapping[str, str],
) -> str:
    keys = tuple(
        match.group("key") for match in _ORACLE_COMMAND_ARG_RE.finditer(raw_args)
    )
    args = tuple(
        referents.get(key) or _oracle_default_public_referent(key, descriptions)
        for key in keys
    )
    normalized = verb.upper()
    if normalized == "COOK" and args:
        return f"cook {args[0]}"
    if normalized == "STIR" and len(args) >= 2:
        return f"stir {args[0]} with {args[1]}"
    if normalized == "WALK" and args:
        return f"walk to {args[0]}"
    if normalized == "LOOKSCAN":
        return "look around"
    return " ".join((normalized.lower(), *args))


def _oracle_collect_public_referents(speech: str, referents: dict[str, str]) -> None:
    for match in _ORACLE_PARENTHETICAL_KEY_RE.finditer(speech):
        public_referent = _oracle_public_referent_before(speech[: match.start()])
        if public_referent:
            referents.setdefault(match.group("key"), public_referent)
    for match in _ORACLE_COMMAND_RE.finditer(speech):
        if match.group("verb").upper() != "WALK":
            continue
        public_referent = _oracle_public_referent_before(
            speech[: match.start()].removesuffix("using ").rstrip()
        )
        keys = tuple(
            arg.group("key")
            for arg in _ORACLE_COMMAND_ARG_RE.finditer(match.group("args"))
        )
        if public_referent and keys:
            referents.setdefault(keys[0], public_referent)
    for match in re.finditer(
        rf"(?P<label>UNIT\s+[A-Z0-9]+)\b[^.!?]{{0,250}}?"
        rf"(?:simulator\s+)?control key is\s+"
        rf"`(?P<key>{_ORACLE_INTERNAL_KEY_PATTERN})`",
        speech,
        re.IGNORECASE,
    ):
        referents[match.group("key")] = match.group("label").upper()
    for match in re.finditer(
        rf"Its\s+(?:simulator\s+)?control key is\s+"
        rf"`(?P<key>{_ORACLE_INTERNAL_KEY_PATTERN})`",
        speech,
        re.IGNORECASE,
    ):
        selections = list(
            re.finditer(r"\bUse the\s+([A-Za-z-]+)", speech[: match.start()])
        )
        if selections:
            referents[match.group("key")] = (
                f"the selected {selections[-1].group(1).lower()}"
            )


def _project_guide_oracle_utterance(
    speech: str,
    referents: dict[str, str],
    descriptions: Mapping[str, str],
) -> str:
    _oracle_collect_public_referents(speech, referents)
    projected = _ORACLE_CONTROL_KEY_CLAUSE_RE.sub("", speech)
    projected = _ORACLE_PRIVATE_SENTENCE_RE.sub(" ", projected)
    projected = _ORACLE_USING_KEY_RE.sub("", projected)
    projected = _ORACLE_PARENTHETICAL_KEY_RE.sub("", projected)
    projected = _ORACLE_USING_WALK_RE.sub("", projected)
    projected = _ORACLE_WITH_LOOKSCAN_RE.sub("", projected)

    def replace_use(match: re.Match[str]) -> str:
        replacement = _oracle_command_public_text(
            match.group("verb"), match.group("args"), referents, descriptions
        )
        if match.group("lead") == "Use":
            return f"{replacement[:1].upper()}{replacement[1:]}"
        return replacement

    projected = _ORACLE_USE_COMMAND_RE.sub(replace_use, projected)
    projected = _ORACLE_COMMAND_RE.sub(
        lambda match: _oracle_command_public_text(
            match.group("verb"), match.group("args"), referents, descriptions
        ),
        projected,
    )
    projected = _ORACLE_BACKTICKED_KEY_RE.sub(
        lambda match: referents.get(match.group("key"))
        or _oracle_default_public_referent(match.group("key"), descriptions),
        projected,
    )
    projected = re.sub(
        r"\bcook (the [A-Za-z -]+) (?:on|for) \1 itself\b",
        r"cook \1 itself",
        projected,
        flags=re.IGNORECASE,
    )
    projected = re.sub(
        r"\bcook (the [A-Za-z -]+) for \1\b",
        r"cook \1",
        projected,
        flags=re.IGNORECASE,
    )
    projected = re.sub(r"\s+", " ", projected).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", projected)


def _project_guide_oracle_dialogue(
    spec: TaskPackInteractionSpec,
) -> tuple[str, ...]:
    referents: dict[str, str] = {}
    dialogue = tuple(
        _project_guide_oracle_utterance(speech, referents, spec.object_descriptions)
        for speech in spec.reference_dialogue
    )
    joined = " ".join(dialogue)
    private_references = []
    if "`" in joined or _ORACLE_COMMAND_RE.search(joined):
        private_references.append("private command syntax")
    if _ORACLE_PRIVATE_INTERFACE_RE.search(joined):
        private_references.append("private interface wording")
    private_references.extend(
        key
        for key in spec.object_descriptions
        if "_" in key
        and re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", joined)
    )
    if private_references:
        raise TaskPackEvalError(
            f"{spec.task_id}: opaque oracle dialogue retains private references "
            f"{sorted(set(private_references))!r}"
        )
    return dialogue


@dataclass(frozen=True)
class FrozenTaskInput:
    taskpack_path: str
    pack: dict[str, Any]
    runtime: TaskRuntime
    builder_spec: dict[str, Any]
    taskpack_file_sha256: str
    taskpack_digest: str
    builder_spec_digest: str
    interaction: TaskPackInteractionSpec | None


@dataclass
class EpisodeStepState:
    observation: dict[str, Any]
    fresh_frames: bool = True
    pending_beat_id: str | None = None


class TaskPackControlAssistant(AssistantBackend):
    """Correct-answer and always-silent controls for one TaskPack."""

    name = "taskpack-control"

    def __init__(
        self,
        arm: str,
        spec: TaskPackInteractionSpec,
        *,
        human_object_interface: str = "full_ids",
    ) -> None:
        super().__init__()
        self.arm = arm
        self.spec = spec
        self.dialogue = (
            _project_guide_oracle_dialogue(spec)
            if arm == "oracle-verbal"
            and spec.family == "GUIDE"
            and human_object_interface == "opaque_handles"
            else spec.reference_dialogue
        )
        self.human_object_interface = human_object_interface
        self.interaction_state: dict[str, object] = {}
        self.delivered = 0

    def reset(self) -> None:
        self.interaction_state = {}
        self.delivered = 0

    def set_interaction_state(self, state: Mapping[str, object]) -> None:
        self.interaction_state = dict(state)

    def generate(self, msgs: list[dict]) -> str:
        del msgs
        self._bump("calls")
        if self.arm != "oracle-verbal":
            return "$silent$"
        if self.interaction_state.get("goal_success") and (
            self.spec.family != "NARRATE"
            or self._action_progress() >= len(self.spec.oracle_commands)
        ):
            return "$done$"
        if self.spec.family == "WATCH":
            dialogue_index = self._watch_dialogue_index()
            if dialogue_index is None:
                return "$silent$"
        else:
            if (
                not self.interaction_state.get("waiting_for_assistant")
                and not self._next_anchor_reached_without_handoff()
            ):
                return "$silent$"
            dialogue_index = self._next_dialogue_index()
            if dialogue_index is None:
                return "$silent$"
        answer = self.dialogue[dialogue_index]
        if dialogue_index == self.delivered:
            self.delivered += 1
        return f"$interrupt${answer}"

    def _next_anchor_reached_without_handoff(self) -> bool:
        if self.spec.family not in {"NARRATE", "ORCHESTRATE"} or self.delivered >= len(
            self.dialogue
        ):
            return False
        anchor = self.spec.reference_dialogue_action_indices[self.delivered]
        return (
            isinstance(anchor, int)
            and not isinstance(anchor, bool)
            and self._action_progress() >= anchor
        )

    def _next_dialogue_index(self) -> int | None:
        if self.delivered >= len(self.dialogue):
            if (
                self.spec.family in {"GUIDE", "NARRATE", "ORCHESTRATE"}
                and self.delivered
            ):
                return self.delivered - 1
            return None
        anchor = self.spec.reference_dialogue_action_indices[self.delivered]
        if anchor is None or self._action_progress() >= anchor:
            return self.delivered
        if self.spec.family in {"GUIDE", "NARRATE", "ORCHESTRATE"} and self.delivered:
            return self.delivered - 1
        return None

    def _action_progress(self) -> int:
        if self.spec.family in {
            "ASK",
            "FIND",
            "GUIDE",
            "NARRATE",
            "ORCHESTRATE",
        }:
            semantic_progress = self.interaction_state.get("oracle_action_progress")
            if isinstance(semantic_progress, int) and not isinstance(
                semantic_progress, bool
            ):
                return semantic_progress
        successful_actions = self.interaction_state.get("successful_actions")
        if isinstance(successful_actions, int) and not isinstance(
            successful_actions, bool
        ):
            return successful_actions
        return 0

    def _watch_dialogue_index(self) -> int | None:
        due = self.interaction_state.get("watch_due_intervention")
        if not isinstance(due, Mapping):
            return None
        dialogue_index = due.get("dialogue_index")
        if isinstance(dialogue_index, bool) or not isinstance(dialogue_index, int):
            return None
        if not 0 <= dialogue_index < len(self.dialogue):
            return None
        if dialogue_index == self.delivered:
            return dialogue_index
        if dialogue_index == self.delivered - 1 and self.interaction_state.get(
            "waiting_for_assistant"
        ):
            return dialogue_index
        return None

    def stats(self) -> dict:
        return {
            **super().stats(),
            "arm": self.arm,
            "scripted": True,
            "answers_delivered": self.delivered,
            "human_object_interface": self.human_object_interface,
            "oracle_dialogue_projected": self.dialogue != self.spec.reference_dialogue,
        }


def _human_policy(args: argparse.Namespace) -> str:
    if args.human_policy != "auto":
        return str(args.human_policy)
    return "scripted" if args.assistant == "oracle-verbal" else "model"


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_taskpack(path: str, expected_sha256: str) -> dict[str, Any]:
    actual = _file_sha256(path)
    expected = expected_sha256.strip()
    if expected and not expected.startswith("sha256:"):
        expected = f"sha256:{expected}"
    if expected and actual != expected:
        raise TaskPackEvalError(
            f"TaskPack file digest mismatch: expected {expected}, got {actual}"
        )
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TaskPackEvalError("TaskPack must be a JSON object")
    return value


def _expected_taskpack_sha256(
    release_tasks: Mapping[str, Mapping[str, Any]], task_id: str, supplied: str
) -> str:
    release_task = release_tasks.get(task_id)
    if release_task is None:
        return supplied
    expected = str(release_task["expected_taskpack_sha256"])
    if supplied != expected:
        raise TaskPackEvalError(
            f"{task_id}: supplied hash does not match the release rubric; "
            "give a modified TaskPack its own task_id"
        )
    return expected


def _freeze_taskpack(
    path: str,
    expected_sha256: str,
    opening_catalog: Mapping[str, Mapping[str, object]],
) -> FrozenTaskInput:
    pack = _load_taskpack(path, expected_sha256)
    builder_spec, builder_digest = builder_spec_from_taskpack(pack)
    taskpack_digest = digest_json(pack)
    frozen = {
        **copy.deepcopy(pack),
        "taskpack_digest": taskpack_digest,
        "builder_spec_digest": builder_digest,
    }
    story = dict(pack.get("story") or {})
    runtime_blob = dict(pack.get("runtime") or {})
    task_id = str(pack.get("task_id") or "")
    extra_info = {
        "scene": f"spec:{task_id}",
        "frozen_spec": {"scene": f"spec:{task_id}"},
        "frozen_taskpack": frozen,
        "taskpack_digest": taskpack_digest,
        "builder_spec_digest": builder_digest,
        "authored_goal": story.get("assistant_goal"),
        "prologue": copy.deepcopy(story.get("prologue")),
        "human_policy": copy.deepcopy(runtime_blob.get("human_policy")),
        "tick_budget": runtime_blob.get("tick_budget"),
        "dialogue_budget": runtime_blob.get("dialogue_budget"),
    }
    runtime = TaskRuntime.from_extra_info(extra_info, 1)
    return FrozenTaskInput(
        taskpack_path=path,
        pack=pack,
        runtime=runtime,
        builder_spec=builder_spec,
        taskpack_file_sha256=_file_sha256(path),
        taskpack_digest=taskpack_digest,
        builder_spec_digest=builder_digest,
        interaction=adapt_taskpack(pack, opening_catalog),
    )


def _reference_dialogue_text(pack: Mapping[str, object]) -> str:
    runtime = pack.get("runtime")
    rows = runtime.get("dialogue") if isinstance(runtime, Mapping) else None
    utterances = [
        str(row.get("say") or "").strip()
        for row in rows or ()
        if isinstance(row, Mapping) and str(row.get("say") or "").strip()
    ]
    if not utterances:
        story = pack.get("story")
        beats = story.get("beats") if isinstance(story, Mapping) else None
        utterances = [
            str(beat.get("say") or "").strip()
            for beat in beats or ()
            if isinstance(beat, Mapping)
            and beat.get("actor") == "assistant"
            and str(beat.get("say") or "").strip()
        ]
    if not utterances:
        raise TaskPackEvalError("TaskPack has no authored assistant answer")
    return " ".join(utterances)


def _intervention_after(pack: Mapping[str, object]) -> str:
    runtime = pack.get("runtime")
    policy = runtime.get("human_policy") if isinstance(runtime, Mapping) else None
    window = policy.get("intervention_window") if isinstance(policy, Mapping) else None
    value = window.get("after_beat") if isinstance(window, Mapping) else None
    if not isinstance(value, str) or not value:
        raise TaskPackEvalError("TaskPack has no intervention_window.after_beat")
    return value


def _command_key(command: str) -> tuple[str, tuple[tuple[str, int], ...]]:
    import command_grammar

    verb, args = command_grammar.parse_line(command)
    return str(verb).upper(), tuple((str(name), int(index)) for name, index in args)


def _is_wait(command: str) -> bool:
    return _command_key(command)[0] == "WAIT"


def _is_preparatory(command: str) -> bool:
    return _command_key(command)[0] in {
        "BACKWARD",
        "FORWARD",
        "LOOKAROUND",
        "LOOKAT",
        "LOOKSCAN",
        "SQUAT",
        "STANDUP",
        "TURNLEFT",
        "TURNRIGHT",
        "WAIT",
        "WALK",
    }


def _should_keep_prologue_frame(command: str, index: int, total: int) -> bool:
    verb = _command_key(command)[0]
    return index == total - 1 or verb in {
        "CLOSE",
        "GRAB",
        "LOOKAT",
        "LOOKSCAN",
        "OPEN",
        "PUT",
        "PUTINSIDE",
        "PUTON",
    }


def _trace_event(beat: AuthoredHumanBeat) -> dict[str, object]:
    definition = beat.definition
    return {
        "beat_id": beat.beat_id,
        "action": dict(definition.get("action") or {}),
        "effects": [
            dict(row)
            for row in definition.get("effects", ())
            if isinstance(row, Mapping)
        ],
        "required_fact_ids": [
            str(value)
            for value in definition.get("requires", ())
            if isinstance(value, str)
        ],
    }


def _step_failures(
    observation: Mapping[str, object], reports: Sequence[object] = ()
) -> list[str]:
    outcomes = [
        row
        for row in (
            observation.get("gate_outcomes")
            or observation.get("outcomes")
            or reports
            or ()
        )
        if isinstance(row, Mapping)
    ]
    failures = [
        str(row.get("detail") or "").strip() or str(row.get("reason") or "")
        for row in outcomes
        if not row.get("ok", True)
    ]
    if failures:
        return [value for value in failures if value and value != "OK"]
    return [
        str(value)
        for value in observation.get("reasons", ())
        if value and str(value) != "OK"
    ]


def _policy_observation(observation: Mapping[str, object]) -> dict[str, object]:
    """Convert live Isaac sets and predicate enums to the TaskPack JSON view."""
    relations = []
    for row in observation.get("relations") or ():
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            parts = list(row)
            if len(parts) == 3:
                predicate = getattr(parts[0], "name", str(parts[0]))
                relations.append([predicate, str(parts[1]), str(parts[2])])
    return {
        "relations": sorted(relations),
        "held": sorted(str(value) for value in (observation.get("held") or ())),
        "at": sorted(str(value) for value in (observation.get("at") or ())),
    }


def _effect_status(
    beat: AuthoredHumanBeat, observation: Mapping[str, object]
) -> tuple[bool, list[str]]:
    """Verify the authored postconditions against the live TaskPack view."""
    view = _policy_observation(observation)
    unmet: list[str] = []
    for row in beat.definition.get("effects", ()):
        if not isinstance(row, Mapping):
            continue
        op = str(row.get("op") or "").lower()
        subject = str(row.get("subject") or "")
        target = str(row.get("target") or "")
        if op in {"not_held", "released"}:
            ok = not observation_world_predicate(view, "held", subject, None)
        elif op in {
            "at",
            "held",
            "inside",
            "attached",
            "not_inside",
            "not_attached",
        }:
            ok = observation_world_predicate(
                view, op, subject, None if not target else target
            )
        elif op in {"assert_reset_from_executable_spec", "policy_halted"}:
            ok = True
        else:
            raise TaskPackEvalError(f"unsupported authored effect {op!r}")
        if not ok:
            unmet.append(f"{op}({subject},{target})")
    return not unmet, unmet


def _realize_human_action(
    proto: Any,
    observation: dict,
    beat: AuthoredHumanBeat,
    frames: list,
    assistant_say: str,
) -> tuple[str | None, bool]:
    expected = str(beat.command or "")
    if _is_wait(expected):
        return expected, True
    directive = (
        "Private scenario direction: perform exactly this one action now and no "
        "other action: "
        f"{expected}"
    )
    before = len(proto._transcript)
    proto._pending_cmds = []
    actual = proto._human_command(
        {**observation, "task": "follow the current scenario direction"},
        directive,
        frames,
    )
    proto._n["human_calls"] += 1
    model_row = proto._transcript[-1] if len(proto._transcript) > before else None
    if model_row is not None:
        model_row.update(
            {
                "assistant_say": assistant_say,
                "human_mode": "taskpack_32b_realizer",
                "human_beat_id": beat.beat_id,
                "human_action_authored": expected,
                "human_private_instruction": directive,
            }
        )
    queued = list(proto._pending_cmds)
    proto._pending_cmds = []
    proto._last_human_say = ""
    if actual is None:
        proto._n["human_idle_retries"] = proto._n.get("human_idle_retries", 0) + 1
        proto._n["human_action_fallbacks"] = (
            proto._n.get("human_action_fallbacks", 0) + 1
        )
        if model_row is not None:
            model_row.update(
                {
                    "human_action_selected": expected,
                    "human_action_selected_index": None,
                    "human_action_is_authored": False,
                    "human_action_fallback_to_authored": True,
                    "human_actions_discarded": [],
                }
            )
        return expected, False
    candidates = [actual, *queued]
    expected_key = _command_key(expected)
    selected_index = next(
        (
            index
            for index, candidate in enumerate(candidates)
            if _command_key(candidate) == expected_key
        ),
        None,
    )
    if selected_index is None:
        proto._n["human_action_mismatches"] = (
            proto._n.get("human_action_mismatches", 0) + 1
        )
        if _is_preparatory(actual):
            proto._n["human_preparatory_proposals"] = (
                proto._n.get("human_preparatory_proposals", 0) + 1
            )
        proto._n["human_action_fallbacks"] = (
            proto._n.get("human_action_fallbacks", 0) + 1
        )
    authored = selected_index is not None
    action_index = selected_index if authored else None
    selected = candidates[selected_index] if authored else expected
    discarded = [
        candidate
        for index, candidate in enumerate(candidates)
        if not authored or index != action_index
    ]
    # The generic executor may batch a short plan, but the TaskPack FSM owns
    # future beats. Preserve unexecuted output for audit and then discard it.
    if discarded:
        proto._n["human_extra_actions_discarded"] = proto._n.get(
            "human_extra_actions_discarded", 0
        ) + len(discarded)
    if model_row is not None:
        model_row.update(
            {
                "human_action_selected": selected,
                "human_action_selected_index": action_index,
                "human_action_is_authored": authored,
                "human_action_fallback_to_authored": not authored,
                "human_actions_discarded": discarded,
            }
        )
    return selected, authored


def _task_goal(score: Mapping[str, object], task_id: str) -> dict[str, object]:
    sub_goals = [
        dict(row)
        for row in (score.get("sub_goals") or score.get("goals") or ())
        if isinstance(row, Mapping)
    ]
    if sub_goals:
        n_total = len(sub_goals)
        n_done = sum(bool(row.get("ok")) for row in sub_goals)
        progress = n_done / n_total
    else:
        n_total = int(score.get("n_total") or 0)
        n_done = int(score.get("n_done") or 0)
        raw_progress = score.get("progress")
        progress = float(
            raw_progress
            if raw_progress is not None
            else (n_done / n_total if n_total else 0.0)
        )
    return {
        "task": task_id,
        "n": n_total,
        "met": bool(score.get("success")),
        "progress": float(progress),
        "n_sub_goals": n_total,
        "n_sub_goals_ok": n_done,
        "sub_goals": sub_goals,
        "neg": list(score.get("neg") or ()),
        "shared_score": dict(score),
    }


def _eval_valid(
    counters: Mapping[str, object], backend_stats: Mapping[str, object]
) -> bool:
    return (
        int(counters.get("human_http_fail", 0)) == 0
        and int(backend_stats.get("http_fail", 0)) == 0
        and bool(counters.get("score_baseline_captured"))
        and int(counters.get("watch_opening_mismatches", 0)) == 0
        and int(counters.get("watch_default_mismatches", 0)) == 0
        and int(counters.get("watch_planned_speech_mismatches", 0)) == 0
        and int(counters.get("watch_planned_speech_repeats", 0)) == 0
        and int(counters.get("human_action_fallbacks", 0)) == 0
        and int(counters.get("human_invalid_attempt_limit_exhausted", 0)) == 0
        and int(counters.get("human_unseen_grab_blocks", 0)) == 0
        and int(counters.get("human_ask_instruction_violations", 0)) == 0
        and int(counters.get("watch_action_retry_attempts", 0)) == 0
        and int(counters.get("watch_preassistant_recovery_unrecovered_chains", 0)) == 0
        and int(counters.get("watch_preassistant_recovery_allowance_exceeded", 0)) == 0
        and int(counters.get("watch_preassistant_recovery_interrupted_by_assistant", 0))
        == 0
        and int(counters.get("watch_preassistant_recovery_human_deviations", 0)) == 0
        and int(counters.get("watch_preassistant_recovery_human_speech", 0)) == 0
        and int(counters.get("watch_preassistant_recovery_stateful_detours", 0)) == 0
        and int(
            counters.get(
                "watch_preassistant_recovery_observation_context_unsupported", 0
            )
        )
        == 0
        and int(counters.get("watch_preassistant_recovery_chains_started", 0)) == 0
        and int(counters.get("action_retry_invalid", 0)) == 0
    )


def _release_validity_evidence(
    *,
    proto: Any,
    counters: Mapping[str, object],
    backend_stats: Mapping[str, object],
    render_episode_valid: bool,
    goal_scored: bool,
) -> dict[str, bool]:
    """Project runner facts into the release scorer's explicit validity checks."""
    human_failure_counters = (
        "human_http_fail",
        "watch_opening_mismatches",
        "watch_default_mismatches",
        "watch_planned_speech_mismatches",
        "watch_planned_speech_repeats",
        "human_action_fallbacks",
        "human_invalid_attempt_limit_exhausted",
        "watch_action_retry_attempts",
        "watch_preassistant_recovery_unrecovered_chains",
        "watch_preassistant_recovery_allowance_exceeded",
        "watch_preassistant_recovery_interrupted_by_assistant",
        "watch_preassistant_recovery_human_deviations",
        "watch_preassistant_recovery_human_speech",
        "watch_preassistant_recovery_stateful_detours",
        "watch_preassistant_recovery_observation_context_unsupported",
        "watch_preassistant_recovery_chains_started",
    )
    traces_complete = all(
        isinstance(getattr(proto, name, None), list)
        for name in ("_assistant_turn_trace", "_human_action_trace", "_transcript")
    )
    simulator_ok = (
        render_episode_valid
        and goal_scored
        and bool(counters.get("score_baseline_captured"))
        and int(counters.get("action_retry_invalid", 0)) == 0
    )
    return {
        "scene_setup_ok": render_episode_valid,
        "runtime_complete": True,
        "trace_complete": traces_complete,
        "frames_usable": render_episode_valid,
        "assistant_transport_ok": int(backend_stats.get("http_fail", 0)) == 0,
        "human_executor_ok": all(
            int(counters.get(name, 0)) == 0 for name in human_failure_counters
        ),
        "simulator_ok": simulator_ok,
        "no_material_human_leakage": (
            int(counters.get("human_unseen_grab_blocks", 0)) == 0
            and int(counters.get("human_ask_instruction_violations", 0)) == 0
        ),
    }


def _human_command_qualifier_manifest(
    interaction: TaskPackInteractionSession,
) -> list[dict[str, object]]:
    return [
        {
            "kind": (
                "authored_putinside_qualifier"
                if qualifier.space is not None
                else "authored_put_qualifier"
            ),
            "matched_command": qualifier.match_command,
            "qualified_command": qualifier.qualified_command,
            "space": qualifier.space,
            "place_spot": list(qualifier.place_spot),
        }
        for qualifier in getattr(interaction.spec, "human_command_qualifiers", ())
    ]


def _write_artifacts(
    env: Any,
    proto: Any,
    backend: AssistantBackend,
    dlog: Any,
    args: argparse.Namespace,
    frozen: FrozenTaskInput,
    out_dir: str,
    tag: str,
    started: float,
    backend_stats: Mapping[str, object],
    interaction: TaskPackInteractionSession,
    final_observation: Mapping[str, object],
) -> dict[str, object]:
    from simulator.interaction import episode_loop as bel
    from simulator.physical_world import episode_runtime as s3

    score = env.score()
    task_goal = _task_goal(score, str(frozen.pack["task_id"]))
    physical_goal_met = bool(task_goal["met"])
    assistant_declared_done = proto._stop_reason == "assistant_done"
    process_requirements_met = interaction.process_requirements_met()
    verification_points = interaction.verification_results(score, final_observation)
    verification_points_met = all(
        bool(point.get("ok")) for point in verification_points
    )
    verification_points_define_success = bool(
        interaction.spec.completion_contract.get(
            "verification_points_define_success", False
        )
    )
    contract_world_success = (
        verification_points_met
        if verification_points_define_success
        else physical_goal_met
    )
    task_goal["physical_met"] = physical_goal_met
    task_goal["contract_world_success"] = contract_world_success
    task_goal["verification_points_define_success"] = verification_points_define_success
    task_goal["assistant_declared_done"] = assistant_declared_done
    task_goal["process_requirements_met"] = process_requirements_met
    task_goal["verification_points_met"] = verification_points_met
    task_goal["verification_points"] = verification_points
    requires_done = interaction.score_requires_assistant_done
    task_goal["met"] = (
        contract_world_success
        and process_requirements_met
        and verification_points_met
        and (assistant_declared_done or not requires_done)
    )
    ctx = env.ctx or {}
    if "render_episode_valid" not in ctx:
        raise TaskPackEvalError("setup_episode did not report render_episode_valid")
    manifest_args = argparse.Namespace(
        scene=f"spec:{frozen.pack['task_id']}",
        seed=int(args.seed),
        subdir=os.path.basename(os.path.normpath(out_dir)),
        tag=tag,
        check=False,
    )
    counters = {
        **bel.eval_counters(proto, backend),
        "prologue_steps": int(proto._n.get("prologue_steps", 0)),
        "human_policy_beats": int(proto._n.get("human_policy_beats", 0)),
        "human_policy_transitions": int(proto._n.get("human_policy_transitions", 0)),
        "human_policy_waits": int(proto._n.get("human_policy_waits", 0)),
        "semantic_verifier_calls": int(proto._n.get("semantic_verifier_calls", 0)),
        "human_policy_status": proto._human_policy_state.get("status", ""),
        "human_action_mismatches": int(proto._n.get("human_action_mismatches", 0)),
        "human_extra_actions_discarded": int(
            proto._n.get("human_extra_actions_discarded", 0)
        ),
        "human_preparatory_actions": int(proto._n.get("human_preparatory_actions", 0)),
        "human_idle_retries": int(proto._n.get("human_idle_retries", 0)),
        "human_preparatory_proposals": int(
            proto._n.get("human_preparatory_proposals", 0)
        ),
        "human_action_fallbacks": int(proto._n.get("human_action_fallbacks", 0)),
        "human_invalid_attempts": int(proto._n.get("human_invalid_attempts", 0)),
        "human_invalid_attempts_recovered": int(
            proto._n.get("human_invalid_attempts_recovered", 0)
        ),
        "human_invalid_attempt_limit_exhausted": int(
            proto._n.get("human_invalid_attempt_limit_exhausted", 0)
        ),
        "human_http_attempt_failures": int(
            proto._n.get("human_http_attempt_failures", 0)
        ),
        "human_http_retries": int(proto._n.get("human_http_retries", 0)),
        "human_provider_truncated": int(proto._n.get("human_provider_truncated", 0)),
        "human_unseen_grab_blocks": int(
            proto._n.get("human_policy_guard_ask_unseen_grab_blocked", 0)
        ),
        "watch_planned_speech_repeats": int(
            proto._n.get("watch_planned_speech_repeats", 0)
        ),
        "human_ask_instruction_violations": sum(
            int(proto._n.get(f"human_policy_guard_{reason}", 0))
            for reason in (
                "ask_unseen_grab_blocked",
                "ask_unseen_target_blocked",
                "ask_unrequested_action_blocked",
                "ask_multiple_commands_blocked",
                "ask_premature_completion_speech_blocked",
                "ask_redundant_walk_blocked",
                "ask_hands_full_grab_blocked",
                "ask_internal_interface_speech_blocked",
                "ask_handle_map_missing",
                "ask_unknown_human_handle_blocked",
            )
        ),
        "human_find_unseen_target_audits": int(
            proto._n.get("human_policy_audit_find_unseen_target_blocked", 0)
        ),
        "human_find_ungrounded_instruction_audits": int(
            proto._n.get("human_policy_audit_find_instruction_grounding_blocked", 0)
        ),
        "action_effect_retries": int(proto._n.get("action_effect_retries", 0)),
        "action_retry_recovered": int(proto._n.get("action_retry_recovered", 0)),
        "action_retry_exhausted": int(proto._n.get("action_retry_exhausted", 0)),
        "action_retry_unsafe_partial_state": int(
            proto._n.get("action_retry_unsafe_partial_state", 0)
        ),
        "action_retry_unattributed_failure": int(
            proto._n.get("action_retry_unattributed_failure", 0)
        ),
        "action_retry_transient_failure": int(
            proto._n.get("action_retry_transient_failure", 0)
        ),
        "action_retry_human_command_rejection": int(
            proto._n.get("action_retry_human_command_rejection", 0)
        ),
        "action_retry_ambiguous_exception": int(
            proto._n.get("action_retry_ambiguous_exception", 0)
        ),
        "action_retry_invalid": int(proto._n.get("action_retry_invalid", 0)),
        "assistant_interventions_consumed": int(
            proto._n.get("assistant_interventions_consumed", 0)
        ),
        "semantic_rejection_short_circuits": int(
            proto._n.get("semantic_rejection_short_circuits", 0)
        ),
        **interaction.counters,
    }
    counters["assistant_backend_stats"] = dict(backend_stats)
    counters["eval_valid"] = _eval_valid(counters, backend_stats)
    validity = bel._benchmark_validity_fields(
        eval_valid=counters["eval_valid"],
        episode_valid=bool(ctx["render_episode_valid"]),
        goal_scored=int(task_goal.get("n") or 0) > 0,
    )
    validity_evidence = _release_validity_evidence(
        proto=proto,
        counters=counters,
        backend_stats=backend_stats,
        render_episode_valid=bool(ctx["render_episode_valid"]),
        goal_scored=int(task_goal.get("n") or 0) > 0,
    )
    s3._write_episode_manifest(
        out_dir,
        tag,
        manifest_args,
        bool(ctx["render_episode_valid"]),
        None,
        task_goal,
        extra={
            "n_fail": int(proto._n.get("bad_command", 0)),
            "wall_time": time.time() - started,
            "taskpack_digest": frozen.taskpack_digest,
            "builder_spec_digest": frozen.builder_spec_digest,
            "taskpack_file_sha256": frozen.taskpack_file_sha256,
            "physical_goal_met": physical_goal_met,
            "contract_world_success": contract_world_success,
            "verification_points_define_success": (verification_points_define_success),
            "assistant_declared_done": assistant_declared_done,
            "process_requirements_met": process_requirements_met,
            "verification_points_met": verification_points_met,
            "score_requires_assistant_done": requires_done,
            "assistant_output_protocol": _assistant_output_protocol_metadata(args),
            **validity,
        },
    )
    dlog.close(counters=counters, stop_reason=proto._stop_reason)
    process_metrics = _process_metrics(proto, frozen, interaction)
    release_task = task_index(load_rubric()).get(str(frozen.pack["task_id"]))
    native_taskpack_outcome = {
        "contract_world_success": contract_world_success,
        "process_requirements_met": process_requirements_met,
        "verification_points_met": verification_points_met,
        "assistant_declared_done": assistant_declared_done,
        "requires_assistant_done": requires_done,
        "strict_success": bool(task_goal["met"]),
    }
    payload = {
        "version": "taskpack-egocentric-eval/2.3",
        "task_id": frozen.pack["task_id"],
        "arm": args.assistant,
        "assistant_backend": args.assistant_backend,
        "assistant_model": args.assistant_model,
        "assistant_output_protocol": _assistant_output_protocol_metadata(args),
        "assistant_control": args.assistant,
        "human_url": args.human_base_url,
        "human_backend": args.human_backend,
        "human_model": args.human_model,
        "human_policy": _human_policy(args),
        "human_object_interface": args.human_object_interface,
        "find_human_control": args.find_human_control,
        "ask_human_control": args.ask_human_control,
        "watch_human_turn_guard": args.watch_human_turn_guard,
        "seed": int(args.seed),
        "tag": tag,
        **validity,
        "validity_evidence": validity_evidence,
        "input_policy": {
            "human_object_interface": args.human_object_interface,
            "find_human_control": args.find_human_control,
            "ask_human_control": args.ask_human_control,
            "watch_human_turn_guard": args.watch_human_turn_guard,
            "assistant_control": args.assistant,
            "assistant_output_protocol": _assistant_output_protocol_metadata(args),
        },
        "frame_config": {
            "frames_per_tick": int(args.per_interval),
            "render_selected_only": True,
            "history_image_cap": int(args.max_frames),
            "assistant_image_cap": int(args.assistant_frame_cap),
            "human_frame_policy": args.human_frame_policy,
            "find_human_control": args.find_human_control,
            "ask_human_control": args.ask_human_control,
            "watch_human_turn_guard": args.watch_human_turn_guard,
            "human_image_cap": int(args.human_frame_cap),
            "human_timeout_s": float(args.human_timeout_s),
            "human_http_attempt_limit": int(args.human_http_attempt_limit),
            "human_http_retry_backoff_s": float(args.human_http_retry_backoff_s),
            "human_invalid_attempt_limit": int(args.human_invalid_attempt_limit),
            "watch_preassistant_recovery_allowance": int(
                args.watch_preassistant_recovery_allowance
            ),
            "progress_line": args.progress_line,
            "assistant_token_cap": (
                int(args.assistant_tokens) if args.assistant_tokens > 0 else None
            ),
            "human_token_cap": (
                int(args.human_tokens) if args.human_tokens > 0 else None
            ),
        },
        "taskpack_file": os.path.basename(frozen.taskpack_path),
        "taskpack_file_sha256": frozen.taskpack_file_sha256,
        "taskpack_digest": frozen.taskpack_digest,
        "builder_spec_digest": frozen.builder_spec_digest,
        "reference_dialogue_text": _reference_dialogue_text(frozen.pack),
        "interaction_spec": {
            "family": interaction.spec.family,
            "public_task_context": interaction.spec.public_task_context,
            "human_task_context": interaction.spec.human_task_context,
            "authored_tick_budget": interaction.spec.authored_tick_budget,
            "opening": {
                "prelude_say": interaction.spec.opening.prelude_say,
                "actions": list(interaction.spec.opening.actions),
                "handoff_say": interaction.spec.opening.handoff_say,
                "utterances": [
                    {
                        "before_action_index": utterance.before_action_index,
                        "say": utterance.say,
                    }
                    for utterance in interaction.spec.opening.utterances
                ],
                "watch_utterances": [
                    {
                        "before_action_index": utterance.before_action_index,
                        "say": utterance.say,
                    }
                    for utterance in interaction.spec.opening.watch_utterances
                ],
                "frame_policy": interaction.spec.opening.frame_policy,
                "recent_intervals": interaction.spec.opening.recent_intervals,
            },
            "public_information_contract": {
                "spoken_cues": [
                    {
                        "id": cue.cue_id,
                        "required_phrases": list(cue.required_phrases),
                    }
                    for cue in interaction.spec.required_spoken_cues
                ],
                "opening_visual_evidence_fact_ids": list(
                    interaction.spec.opening_evidence_fact_ids
                ),
                "assistant_input_sources": [
                    "human opening speech",
                    "retained opening video",
                    "live human speech",
                    "live egocentric video",
                ],
            },
            "reference_dialogue": list(interaction.spec.reference_dialogue),
            "reference_dialogue_action_indices": list(
                interaction.spec.reference_dialogue_action_indices
            ),
            "scripted_human_replies": list(interaction.spec.scripted_human_replies),
            "opening_evidence_action_indices": list(
                interaction.spec.opening_evidence_action_indices
            ),
            "human_command_qualifiers": _human_command_qualifier_manifest(interaction),
        },
        "interaction": interaction.metrics(),
        "completion_contract": {
            "physical_goal_met": physical_goal_met,
            "contract_world_success": contract_world_success,
            "verification_points_define_success": (verification_points_define_success),
            "assistant_declared_done": assistant_declared_done,
            "process_requirements_met": process_requirements_met,
            "requires_assistant_done": requires_done,
            "passed": (
                contract_world_success
                and process_requirements_met
                and verification_points_met
                and (assistant_declared_done or not requires_done)
            ),
            "verification_points": verification_points,
        },
        "native_taskpack_outcome": native_taskpack_outcome,
        "release_rubric_outcome": (
            {
                "status": "UNSCORABLE",
                "rubric_version": release_task["rubric_version"],
                "strict_pass": None,
                "completion_score": None,
                "reason": (
                    "The live runner records native TaskPack outcome and raw state/trace "
                    "evidence. Revised release scoring requires all protocol-specific "
                    "criterion results; it is never inferred from native success."
                ),
            }
            if release_task is not None
            else {
                "status": "NOT_APPLICABLE",
                "rubric_version": None,
                "strict_pass": None,
                "completion_score": None,
                "reason": (
                    "Custom TaskPacks have no release rubric; "
                    "native_taskpack_outcome is their score."
                ),
            }
        ),
        "counters": counters,
        "task_goal": task_goal,
        "process_metrics": process_metrics,
        "assistant_turns": list(getattr(proto, "_assistant_turn_trace", ())),
        "human_action_trace": list(getattr(proto, "_human_action_trace", ())),
        "transcript": proto._transcript,
        "timing_s": {key: round(value, 2) for key, value in proto._t.items() if value},
    }
    _write_evaluation_artifacts(out_dir, tag, payload)
    return payload


def _write_evaluation_artifacts(
    out_dir: str, tag: str, payload: Mapping[str, object]
) -> None:
    body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    for suffix in ("eval", "taskpack_eval"):
        path = os.path.join(out_dir, f"{tag}_{suffix}.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)


def _backend_snapshot(backend: AssistantBackend) -> tuple[dict[str, int], int]:
    return dict(backend._n), len(backend._latency_s)


def _backend_stats_since(
    backend: AssistantBackend, snapshot: tuple[dict[str, int], int]
) -> dict[str, object]:
    counts, latency_index = snapshot
    stats = dict(backend.stats())
    for key in (
        "calls",
        "http_fail",
        "empty",
        "gateway_retries",
    ):
        if key not in stats:
            continue
        stats[key] = int(backend._n.get(key, 0)) - int(counts.get(key, 0))
    latency = sorted(backend._latency_s[latency_index:])
    stats["latency_p50_s"] = round(latency[len(latency) // 2], 2) if latency else None
    stats["latency_total_s"] = round(sum(latency), 1)
    return stats


def _process_metrics(
    proto: Any,
    frozen: FrozenTaskInput,
    interaction: TaskPackInteractionSession | None = None,
) -> dict[str, object]:
    turns = list(getattr(proto, "_assistant_turn_trace", ()))
    actions = list(getattr(proto, "_human_action_trace", ()))
    if interaction is not None:
        spoken = [row for row in turns if str(row.get("say") or "").strip()]
        credited_recovery_ticks = int(
            proto._n.get("watch_preassistant_recovery_credited_ticks", 0)
        )
        first_spoken_tick = spoken[0].get("tick") if spoken else None
        declared_at_tick = proto._n.get("declared_at_tick")
        return {
            "family": interaction.spec.family,
            "human_policy": getattr(proto, "_interaction_human_policy", "model"),
            "find_human_control": getattr(
                proto, "_find_human_control", "dynamic-guarded"
            ),
            "ask_human_control": getattr(proto, "_ask_human_control", "strict"),
            "watch_human_turn_guard": bool(
                getattr(proto, "_interaction_watch_human_turn_guard", True)
            ),
            "assistant_turns": len(turns),
            "assistant_spoken_turns": len(spoken),
            "assistant_silent_turns": len(turns) - len(spoken),
            "assistant_first_spoken_tick": first_spoken_tick,
            "raw_assistant_first_spoken_tick": first_spoken_tick,
            "semantic_assistant_first_spoken_tick": (
                int(first_spoken_tick) - credited_recovery_ticks
                if first_spoken_tick is not None
                else None
            ),
            "assistant_declared_done": proto._stop_reason == "assistant_done",
            "declared_at_tick": declared_at_tick,
            "raw_declared_at_tick": declared_at_tick,
            "semantic_declared_at_tick": (
                int(declared_at_tick) - credited_recovery_ticks
                if declared_at_tick is not None
                else None
            ),
            "ticks_to_terminal": int(proto._n.get("ticks", 0)),
            "raw_ticks_to_terminal": int(
                proto._n.get("n_raw_ticks", proto._n.get("ticks", 0))
            ),
            "semantic_ticks_to_terminal": int(
                proto._n.get("n_semantic_ticks", proto._n.get("ticks", 0))
            ),
            "raw_assistant_calls": int(
                proto._n.get("n_raw_assistant_calls", len(turns))
            ),
            "semantic_assistant_calls": int(
                proto._n.get("n_semantic_assistant_calls", len(turns))
            ),
            "raw_human_actions": int(
                proto._n.get("n_raw_human_actions", len(proto._human_action_trace))
            ),
            "semantic_human_actions": int(
                proto._n.get("n_semantic_human_actions", len(proto._human_action_trace))
            ),
            "prologue_steps": int(proto._n.get("prologue_steps", 0)),
            "environment_steps": int(proto._n.get("env_steps", 0)),
            "human_model_calls": int(proto._n.get("human_calls", 0)),
            "human_actions": len(actions),
            "human_action_fallbacks": int(proto._n.get("human_action_fallbacks", 0)),
            "human_invalid_attempts": int(proto._n.get("human_invalid_attempts", 0)),
            "human_invalid_attempts_recovered": int(
                proto._n.get("human_invalid_attempts_recovered", 0)
            ),
            "human_invalid_attempt_limit_exhausted": int(
                proto._n.get("human_invalid_attempt_limit_exhausted", 0)
            ),
            "human_http_attempt_failures": int(
                proto._n.get("human_http_attempt_failures", 0)
            ),
            "human_http_retries": int(proto._n.get("human_http_retries", 0)),
            "human_provider_truncated": int(
                proto._n.get("human_provider_truncated", 0)
            ),
            "human_unseen_grab_blocks": int(
                proto._n.get("human_policy_guard_ask_unseen_grab_blocked", 0)
            ),
            "human_ask_instruction_violations": sum(
                int(proto._n.get(f"human_policy_guard_{reason}", 0))
                for reason in (
                    "ask_unseen_grab_blocked",
                    "ask_unseen_target_blocked",
                    "ask_unrequested_action_blocked",
                    "ask_multiple_commands_blocked",
                    "ask_premature_completion_speech_blocked",
                    "ask_redundant_walk_blocked",
                    "ask_hands_full_grab_blocked",
                    "ask_internal_interface_speech_blocked",
                    "ask_handle_map_missing",
                    "ask_unknown_human_handle_blocked",
                )
            ),
            "human_find_unseen_target_audits": int(
                proto._n.get("human_policy_audit_find_unseen_target_blocked", 0)
            ),
            "human_find_ungrounded_instruction_audits": int(
                proto._n.get("human_policy_audit_find_instruction_grounding_blocked", 0)
            ),
            "semantic_verifier_calls": int(proto._n.get("semantic_verifier_calls", 0)),
            "action_effect_retries": int(proto._n.get("action_effect_retries", 0)),
            "action_retry_recovered": int(proto._n.get("action_retry_recovered", 0)),
            "action_retry_exhausted": int(proto._n.get("action_retry_exhausted", 0)),
            "action_retry_transient_failure": int(
                proto._n.get("action_retry_transient_failure", 0)
            ),
            "action_retry_invalid": int(proto._n.get("action_retry_invalid", 0)),
            **interaction.metrics(),
            "watch_planned_speech_repeats": int(
                proto._n.get("watch_planned_speech_repeats", 0)
            ),
        }
    intervention_after = _intervention_after(frozen.pack)
    spoken = [row for row in turns if str(row.get("say") or "").strip()]
    in_window = [
        row for row in turns if row.get("pending_beat_id") == intervention_after
    ]
    action_beats = [row for row in actions if row.get("authored_command")]
    matches = sum(bool(row.get("human_model_matched_oracle")) for row in action_beats)
    return {
        "intervention_after_beat": intervention_after,
        "assistant_turns": len(turns),
        "assistant_spoken_turns": len(spoken),
        "assistant_silent_turns": len(turns) - len(spoken),
        "assistant_spoke_in_intervention_window": any(
            str(row.get("say") or "").strip() for row in in_window
        ),
        "assistant_spoke_before_intervention_window": (
            any(
                str(row.get("say") or "").strip()
                and row.get("pending_beat_id") != intervention_after
                for row in turns[: max(0, turns.index(in_window[0]))]
            )
            if in_window
            else bool(spoken)
        ),
        "assistant_first_spoken_tick": spoken[0].get("tick") if spoken else None,
        "assistant_intervention_consumed": bool(
            proto._n.get("assistant_interventions_consumed", 0)
        ),
        "ticks_to_terminal": int(proto._n.get("ticks", 0)),
        "prologue_steps": int(proto._n.get("prologue_steps", 0)),
        "environment_steps": int(proto._n.get("env_steps", 0)),
        "human_model_calls": int(proto._n.get("human_calls", 0)),
        "human_oracle_action_matches": matches,
        "human_oracle_action_total": len(action_beats),
        "human_oracle_action_match_rate": (
            round(matches / len(action_beats), 4) if action_beats else 1.0
        ),
        "human_action_fallbacks": int(proto._n.get("human_action_fallbacks", 0)),
        "action_effect_retries": int(proto._n.get("action_effect_retries", 0)),
        "action_retry_recovered": int(proto._n.get("action_retry_recovered", 0)),
        "action_retry_exhausted": int(proto._n.get("action_retry_exhausted", 0)),
        "action_retry_transient_failure": int(
            proto._n.get("action_retry_transient_failure", 0)
        ),
        "action_retry_invalid": int(proto._n.get("action_retry_invalid", 0)),
        "semantic_verifier_calls": int(proto._n.get("semantic_verifier_calls", 0)),
        "semantic_rejection_short_circuit": bool(
            proto._n.get("semantic_rejection_short_circuits", 0)
        ),
    }


def _assistant_turn(
    proto: Any,
    backend: AssistantBackend,
    dlog: Any,
    runtime: TaskRuntime,
    observation: dict,
    frame_buffer: Any,
    tick: int,
    fresh_frames: bool,
    pending_beat_id: str | None,
) -> str:
    if hasattr(backend, "set_pending_beat"):
        backend.set_pending_beat(pending_beat_id)
    assistant_frames = (
        frame_buffer.fed_frames()
        if tick == 0
        else (frame_buffer.delta() if fresh_frames else [])
    )
    messages = proto._assistant_messages(
        runtime.assistant_observation(observation),
        first=tick == 0,
        frames=assistant_frames,
        heard=proto._last_human_say,
    )
    proto._last_human_say = ""
    started = time.perf_counter()
    raw = backend.generate(messages)
    proto._tk["t_gen_s"].append(round(proto._tadd("t_gen_s", started), 3))
    decision_raw = backend.decision_text(raw)
    turn_metadata = backend.turn_metadata()
    assistant = proto._parse_assistant(decision_raw)
    assistant_say = str(assistant.get("say") or "")
    proto._last_assistant_mode = str(assistant.get("mode") or "silence")
    proto._assistant_turn_trace.append(
        {
            "tick": tick,
            "pending_beat_id": pending_beat_id,
            "mode": proto._last_assistant_mode,
            "say": assistant_say,
            "raw": raw,
            "decision_raw": decision_raw,
            "n_frames": len(assistant_frames),
            **turn_metadata,
        }
    )
    if assistant_say:
        dlog.utterance(
            "assistant",
            assistant_say,
            raw_text=raw[:600],
            mode=proto._last_assistant_mode,
        )
    else:
        dlog.silence("assistant", prompted=True, reason="chose_silence")
    return assistant_say


def _apply_assistant_turn(
    proto: Any,
    human: TaskPackHumanRuntime,
    dlog: Any,
    observation: dict,
    frame_buffer: Any,
    assistant_say: str,
) -> tuple[str, str | None]:
    calls_before = human.semantic_verifier_calls
    started = time.perf_counter()
    transition = human.apply_assistant(
        assistant_text=assistant_say,
        assistant_mode=proto._last_assistant_mode,
        tick=proto._n["ticks"],
        observation=_policy_observation(observation),
        frames=(
            frame_buffer.recent_frames(_HUMAN_VISUAL_MEMORY_INTERVALS)
            if human.has_pending_beat
            else ()
        ),
    )
    proto._tk["t_human_s"].append(round(proto._tadd("t_human_s", started), 3))
    proto._n["semantic_verifier_calls"] += human.semantic_verifier_calls - calls_before
    if transition is None:
        proto._assistant_turn_trace[-1]["human_policy_status"] = "READY"
        return "READY", None
    if transition.status == "WAITING":
        proto._n["human_policy_waits"] += 1
        proto._assistant_turn_trace[-1]["human_policy_status"] = "WAITING"
        return "WAITING", transition.completed_beat_id
    proto._n["human_policy_transitions"] += 1
    proto._assistant_turn_trace[-1]["human_policy_status"] = transition.status
    proto._assistant_turn_trace[-1]["human_policy_matched_condition"] = (
        transition.matched_condition
    )
    proto._assistant_turn_trace[-1]["human_policy_next_beat_id"] = (
        transition.next_beat_id
    )
    if transition.consumed_intervention_beat is not None:
        proto._n["assistant_interventions_consumed"] = (
            proto._n.get("assistant_interventions_consumed", 0) + 1
        )
        proto._assistant_turn_trace[-1]["intervention_consumed"] = True
        dlog.marker(
            "taskpack_intervention",
            text=str(transition.consumed_intervention_beat.get("say") or ""),
            ref=str(transition.consumed_intervention_beat.get("id") or ""),
        )
    return transition.status, None


def _should_short_circuit_rejection(
    enabled: bool,
    pending_beat_id: str | None,
    intervention_after: str,
    assistant_turn: Mapping[str, object],
) -> bool:
    return (
        enabled
        and pending_beat_id == intervention_after
        and assistant_turn.get("human_policy_status") != "WAITING"
        and not assistant_turn.get("intervention_consumed")
    )


def _complete_speech_beat(
    proto: Any,
    human: TaskPackHumanRuntime,
    beat: AuthoredHumanBeat,
    observation: dict,
    assistant_say: str,
    tick: int,
) -> bool:
    proto._last_human_say = beat.say
    proto._transcript.append(
        {
            "turn": tick,
            "assistant_mode": proto._last_assistant_mode,
            "assistant_say": assistant_say,
            "assistant_raw": proto._last_assistant_raw,
            "human_mode": "taskpack_authored_speech",
            "human_model_call": False,
            "human_action": "",
            "human_say": beat.say,
            "human_beat_id": beat.beat_id,
        }
    )
    completed = human.complete_beat(
        beat.beat_id,
        tick=proto._n["ticks"],
        observation=_policy_observation(observation),
    )
    return completed is not None and completed.status == "COMPLETE"


def _execute_action_beat(
    env: Any,
    proto: Any,
    human: TaskPackHumanRuntime,
    dlog: Any,
    frame_buffer: Any,
    observation: dict,
    beat: AuthoredHumanBeat,
    assistant_say: str,
) -> tuple[dict, bool, bool]:
    started = time.perf_counter()
    command, authored = _realize_human_action(
        proto,
        observation,
        beat,
        frame_buffer.fed_frames(),
        assistant_say,
    )
    proto._tk["t_human_s"].append(round(proto._tadd("t_human_s", started), 3))
    if command is None:
        return observation, False, False
    proto._last_human_say = beat.say
    trace = {
        "beat_id": beat.beat_id,
        "authored_command": str(beat.command or ""),
        "executed_command": command,
        "human_model_matched_oracle": authored,
        "attempts": [],
    }
    proto._human_action_trace.append(trace)
    max_attempts = 1 + int(getattr(proto, "_taskpack_action_retries", 0))
    effects_ok = False
    failures: list[str] = []
    unmet: list[str] = []
    for attempt in range(1, max_attempts + 1):
        dlog.command(
            "human",
            command,
            command,
            channel="action",
            beat_id=beat.beat_id,
            attempt=attempt,
            human_model_matched_oracle=authored,
        )
        started = time.perf_counter()
        observation = env.step(command, trace_event=_trace_event(beat))
        proto._tk["t_step_s"].append(round(proto._tadd("t_step_s", started), 3))
        proto._n["env_steps"] += 1
        proto._vis_absorb(observation)
        frame_buffer.add_interval(
            proto._obs_frames(observation), observation.get("frame_segs")
        )
        failures = _step_failures(observation, env.last_step_reports())
        effects_ok, unmet = _effect_status(beat, observation)
        trace["attempts"].append(
            {
                "attempt": attempt,
                "failures": failures,
                "effects_ok": effects_ok,
                "unmet_effects": unmet,
                "score": dict(observation.get("score") or {}),
            }
        )
        if not failures and effects_ok:
            break
        if attempt < max_attempts:
            proto._n["action_effect_retries"] = (
                proto._n.get("action_effect_retries", 0) + 1
            )
    if failures or not effects_ok:
        detail = "; ".join(failures or [f"unmet effect {value}" for value in unmet])
        raise TaskPackEvalError(
            f"authored beat {beat.beat_id!r} failed after {max_attempts} "
            f"attempt(s): {detail}"
        )
    completed = human.complete_beat(
        beat.beat_id,
        tick=proto._n["ticks"],
        observation=_policy_observation(observation),
    )
    return observation, completed is not None and completed.status == "COMPLETE", True


def _run_prologue(
    env: Any,
    proto: Any,
    dlog: Any,
    runtime: TaskRuntime,
    frame_buffer: Any,
    observation: dict[str, Any],
) -> dict[str, Any]:
    prologue = list(runtime.prologue(()))
    for index, command in enumerate(prologue):
        observation = env.step(command)
        failures = _step_failures(observation, env.last_step_reports())
        if failures:
            raise TaskPackEvalError(
                f"authored prologue command {command!r} failed: {'; '.join(failures)}"
            )
        assert_unscored_state(observation, f"prologue command {command!r}")
        if _should_keep_prologue_frame(command, index, len(prologue)):
            frame_buffer.add_interval(
                proto._obs_frames(observation), observation.get("frame_segs")
            )
        proto._n["prologue_steps"] = proto._n.get("prologue_steps", 0) + 1
        dlog.marker("prologue_step", text=command)
    return observation


def _finish_human_policy(proto: Any) -> None:
    proto._human_policy_state["status"] = "COMPLETE"
    proto._stop_reason = "human_policy_terminal"


def _run_assistant_phase(
    proto: Any,
    backend: AssistantBackend,
    dlog: Any,
    runtime: TaskRuntime,
    human: TaskPackHumanRuntime,
    frame_buffer: Any,
    state: EpisodeStepState,
    tick: int,
    intervention_after: str,
) -> tuple[str, str, bool]:
    assistant_say = _assistant_turn(
        proto,
        backend,
        dlog,
        runtime,
        state.observation,
        frame_buffer,
        tick,
        state.fresh_frames,
        state.pending_beat_id,
    )
    state.fresh_frames = False
    completed_beat_id = state.pending_beat_id
    status, state.pending_beat_id = _apply_assistant_turn(
        proto, human, dlog, state.observation, frame_buffer, assistant_say
    )
    if _should_short_circuit_rejection(
        bool(getattr(proto, "_taskpack_stop_on_rejected_intervention", False)),
        completed_beat_id,
        intervention_after,
        proto._assistant_turn_trace[-1],
    ):
        proto._n["semantic_rejection_short_circuits"] = (
            proto._n.get("semantic_rejection_short_circuits", 0) + 1
        )
        proto._human_policy_state["status"] = "REJECTED_INTERVENTION"
        proto._stop_reason = "semantic_rejection"
        return status, assistant_say, True
    if status == "COMPLETE":
        _finish_human_policy(proto)
        return status, assistant_say, True
    return status, assistant_say, False


def _run_human_phase(
    env: Any,
    proto: Any,
    human: TaskPackHumanRuntime,
    dlog: Any,
    frame_buffer: Any,
    state: EpisodeStepState,
    assistant_say: str,
    tick: int,
) -> bool:
    beat = human.next_beat()
    proto._n["human_policy_beats"] += 1
    if beat.say:
        dlog.utterance("human", beat.say, channel="speech", beat_id=beat.beat_id)
    if beat.command is None:
        state.pending_beat_id = beat.beat_id
        if _complete_speech_beat(
            proto, human, beat, state.observation, assistant_say, tick
        ):
            _finish_human_policy(proto)
            return True
        return False
    state.observation, complete, authored = _execute_action_beat(
        env,
        proto,
        human,
        dlog,
        frame_buffer,
        state.observation,
        beat,
        assistant_say,
    )
    state.fresh_frames = True
    state.pending_beat_id = beat.beat_id if authored else None
    if complete:
        _finish_human_policy(proto)
    return complete


def _run_episode(
    env: Any,
    proto: Any,
    backend: AssistantBackend,
    dlog: Any,
    frozen: FrozenTaskInput,
    observation: dict,
) -> dict:
    from simulator.interaction.ego_frames import EgoFrameBuffer

    runtime = frozen.runtime
    human = TaskPackHumanRuntime(
        runtime.frozen_taskpack or {},
        semantic_verifier=HumanModelSemanticVerifier(
            proto.human_server,
            runtime.frozen_taskpack or {},
            timeout_s=proto.human_timeout_s,
        ),
    )
    proto._human_policy_state["status"] = "ACTIVE"
    proto._episode_max_ticks = proto.max_ticks
    frame_buffer = EgoFrameBuffer(
        per_interval=proto.per_interval, max_frames=proto.max_frames
    )
    frame_buffer.add_interval(
        proto._obs_frames(observation), observation.get("frame_segs")
    )
    assert_unscored_state(observation, "episode start")
    observation = _run_prologue(env, proto, dlog, runtime, frame_buffer, observation)
    frame_buffer.pinned = len(frame_buffer.intervals)
    proto._last_human_say = runtime.prologue_say
    proto._stop_reason = "max_ticks"
    state = EpisodeStepState(observation)
    intervention_after = _intervention_after(frozen.pack)
    for tick in range(proto.max_ticks):
        proto._n["ticks"] = tick + 1
        dlog.set_tick(tick)
        status, assistant_say, terminal = _run_assistant_phase(
            proto,
            backend,
            dlog,
            runtime,
            human,
            frame_buffer,
            state,
            tick,
            intervention_after,
        )
        if terminal:
            break
        if status == "WAITING":
            continue
        if _run_human_phase(
            env, proto, human, dlog, frame_buffer, state, assistant_say, tick
        ):
            break

    if not human.is_complete and proto._stop_reason != "semantic_rejection":
        raise HumanPolicyBlocked(
            "TICK_BUDGET_EXHAUSTED",
            "$.runtime.tick_budget",
            f"human policy did not reach a terminal within {proto.max_ticks} ticks",
        )
    return state.observation


def _make_assistant(
    args: argparse.Namespace, frozen: FrozenTaskInput
) -> AssistantBackend:
    if args.assistant == "model":
        return make_backend(
            args.assistant_provider,
            base_url=args.assistant_base_url,
            model=args.assistant_model,
            api_key_env=args.assistant_api_key_env,
            max_new_tokens=int(args.assistant_tokens),
            temperature=0.0,
            timeout_s=float(args.assistant_timeout_s),
            frame_cap=int(args.assistant_frame_cap),
        )
    if frozen.interaction is None:
        raise TaskPackEvalError("frozen TaskPack has no interaction specification")
    return TaskPackControlAssistant(
        args.assistant,
        frozen.interaction,
        human_object_interface=args.human_object_interface,
    )


def _validate_human_retry_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.human_invalid_attempt_limit < 1:
        parser.error("--human-invalid-attempt-limit must be positive")
    if args.human_http_attempt_limit < 1:
        parser.error("--human-http-attempt-limit must be positive")
    if args.human_http_retry_backoff_s < 0:
        parser.error("--human-http-retry-backoff-s must be non-negative")


def _validate_frame_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.frame_budget < 1:
        parser.error("--frame-budget must be positive")
    if args.per_interval < 1:
        parser.error("--per-interval must be positive")
    if args.max_frames < args.per_interval:
        parser.error("--max-frames must hold at least one complete tick")
    if args.assistant_frame_cap < args.per_interval:
        parser.error("--assistant-frame-cap must hold at least one complete tick")
    if args.human_frame_cap < args.per_interval:
        parser.error("--human-frame-cap must hold at least one complete tick")


def _validate_parsed_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    _validate_frame_args(parser, args)
    if args.assistant_tokens < 0:
        parser.error("--assistant-tokens must be non-negative")
    if args.human_tokens < 0:
        parser.error("--human-tokens must be non-negative")
    if args.assistant_timeout_s <= 0:
        parser.error("--assistant-timeout-s must be positive")
    if args.human_timeout_s <= 0:
        parser.error("--human-timeout-s must be positive")
    _validate_human_retry_args(parser, args)
    if not 0 <= args.watch_preassistant_recovery_allowance <= 8:
        parser.error("--watch-preassistant-recovery-allowance must be between 0 and 8")


def _provider_base_url(provider: str, role: str, configured: str) -> str:
    if configured.strip():
        return configured.strip()
    default = default_provider_base_url(provider)
    if default:
        return default
    port = 8802 if role == "assistant" else 8801
    return f"http://127.0.0.1:{port}/generate"


def _configure_provider_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    raw_assistant_provider = str(args.assistant_provider).strip().lower()
    raw_human_provider = str(args.human_provider).strip().lower()
    if (
        raw_assistant_provider == "openai-compatible"
        and not str(args.assistant_base_url).strip()
    ):
        parser.error(
            "--assistant-base-url is required for openai-compatible; "
            "use provider 'openai' for the hosted OpenAI endpoint"
        )
    if (
        raw_human_provider == "openai-compatible"
        and not str(args.human_base_url).strip()
    ):
        parser.error(
            "--human-base-url is required for openai-compatible; "
            "use provider 'openai' for the hosted OpenAI endpoint"
        )
    try:
        args.assistant_provider = normalize_provider(args.assistant_provider)
        args.human_provider = normalize_provider(args.human_provider)
        args.assistant_base_url = _provider_base_url(
            args.assistant_provider, "assistant", args.assistant_base_url
        )
        args.human_base_url = _provider_base_url(
            args.human_provider, "human", args.human_base_url
        )
        if args.assistant == "model":
            args.assistant_api_key_env = resolve_api_key_env(
                args.assistant_provider,
                "assistant",
                args.assistant_api_key_env,
                base_url=args.assistant_base_url,
            )
        if _human_policy(args) == "model":
            args.human_api_key_env = resolve_api_key_env(
                args.human_provider,
                "human",
                args.human_api_key_env,
                base_url=args.human_base_url,
            )
    except AssistantBackendError as exc:
        parser.error(str(exc))
    if args.assistant == "model" and not args.assistant_model:
        if args.assistant_provider != "http-json":
            parser.error("--assistant-model is required for this provider")
        args.assistant_model = args.assistant_base_url
    if _human_policy(args) == "model" and not args.human_model:
        if args.human_provider != "http-json":
            parser.error("--human-model is required for this provider")
        args.human_model = args.human_base_url

    # Compatibility attributes consumed by the frozen episode protocol.  The Human
    # URL is replaced with a loopback gateway immediately before protocol creation.
    args.assistant_url = args.assistant_base_url
    args.human_url = args.human_base_url
    args.assistant_backend = (
        args.assistant_provider if args.assistant == "model" else "scripted"
    )
    args.human_backend = (
        args.human_provider if _human_policy(args) == "model" else "scripted"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more frozen InteractiveWorld TaskPacks."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--task-id",
        help="run a task from the bundled Balanced120 manifest",
    )
    source.add_argument(
        "--taskpack",
        help=(
            "run a bundled or custom TaskPack file; --taskpack-sha256 is required"
        ),
    )
    source.add_argument(
        "--taskpack-batch",
        help=(
            "manifest with tasks[] entries containing task_id, taskpack_path and "
            "taskpack_sha256"
        ),
    )
    parser.add_argument("--taskpack-sha256", default="")
    parser.add_argument(
        "--task-manifest",
        default=os.path.join(_RELEASE_ROOT, "benchmark", "tasks", "manifest.json"),
        help="frozen release manifest used by --task-id",
    )
    parser.add_argument(
        "--assistant",
        choices=("oracle-verbal", "silent", "model"),
        required=True,
    )
    parser.add_argument(
        "--assistant-provider",
        "--assistant-backend",
        dest="assistant_provider",
        choices=("http-json", "openai-compatible", "openai", "anthropic", "gemini"),
        default=os.environ.get("INTERACTIVEWORLD_ASSISTANT_PROVIDER", "http-json"),
        help=(
            "Assistant API: generic http-json, OpenAI-compatible "
            "(OpenAI/vLLM/Ollama), Anthropic Messages, or Gemini REST"
        ),
    )
    parser.add_argument(
        "--assistant-base-url",
        "--assistant-url",
        dest="assistant_base_url",
        default=os.environ.get(
            "INTERACTIVEWORLD_ASSISTANT_BASE_URL",
            os.environ.get("INTERACTIVEWORLD_ASSISTANT_URL", ""),
        ),
        help="Assistant API base URL; --assistant-url is a compatibility alias",
    )
    parser.add_argument(
        "--assistant-model",
        default=os.environ.get("INTERACTIVEWORLD_ASSISTANT_MODEL", ""),
        help="Assistant model ID (optional only for generic http-json)",
    )
    parser.add_argument(
        "--assistant-api-key-env",
        default=os.environ.get("INTERACTIVEWORLD_ASSISTANT_API_KEY_ENV", ""),
        metavar="ENV_NAME",
        help=(
            "name of the environment variable holding the Assistant API key; "
            "the key is never accepted as a command-line value or written to artifacts"
        ),
    )
    parser.add_argument(
        "--assistant-tokens",
        type=int,
        default=0,
        help=(
            "Assistant maximum output tokens; 0 uses the provider default "
            "(Anthropic requires and receives 1024)"
        ),
    )
    parser.add_argument("--assistant-timeout-s", type=float, default=180.0)
    parser.add_argument("--assistant-frame-cap", type=int, default=32)
    parser.add_argument(
        "--human-provider",
        choices=("http-json", "openai-compatible", "openai", "anthropic", "gemini"),
        default=os.environ.get("INTERACTIVEWORLD_HUMAN_PROVIDER", "http-json"),
        help=(
            "Human API: generic http-json, OpenAI-compatible "
            "(OpenAI/vLLM/Ollama), Anthropic Messages, or Gemini REST"
        ),
    )
    parser.add_argument(
        "--human-base-url",
        "--human-url",
        dest="human_base_url",
        default=os.environ.get(
            "INTERACTIVEWORLD_HUMAN_BASE_URL",
            os.environ.get("INTERACTIVEWORLD_HUMAN_URL", ""),
        ),
        help=("Human API base URL; --human-url is a compatibility alias"),
    )
    parser.add_argument(
        "--human-timeout-s",
        type=float,
        default=360.0,
        help="timeout for Human readiness and action requests",
    )
    parser.add_argument(
        "--human-invalid-attempt-limit",
        type=int,
        default=8,
        help="maximum Human generations for one ASK turn before marking it invalid",
    )
    parser.add_argument("--human-http-attempt-limit", type=int, default=3)
    parser.add_argument("--human-http-retry-backoff-s", type=float, default=0.25)
    parser.add_argument(
        "--human-model",
        default=os.environ.get("INTERACTIVEWORLD_HUMAN_MODEL", ""),
        help="Human model ID (optional only for generic http-json)",
    )
    parser.add_argument(
        "--human-api-key-env",
        default=os.environ.get("INTERACTIVEWORLD_HUMAN_API_KEY_ENV", ""),
        metavar="ENV_NAME",
        help=(
            "name of the environment variable holding the Human API key; "
            "resolved independently from Assistant credentials"
        ),
    )
    parser.add_argument(
        "--human-tokens",
        type=int,
        default=int(os.environ.get("INTERACTIVEWORLD_HUMAN_TOKENS", "0")),
        help="Human maximum output tokens (0 keeps the simulator default)",
    )
    parser.add_argument("--human-frame-cap", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=0)
    parser.add_argument(
        "--frame-budget",
        type=int,
        default=24,
        help="total egocentric-frame budget distributed across an episode",
    )
    parser.add_argument("--per-interval", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--frame-max-side", type=int, default=512)
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "INTERACTIVEWORLD_OUTPUT_DIR", "/tmp/interactiveworld-output"
        ),
        help="artifact directory (default: INTERACTIVEWORLD_OUTPUT_DIR or /tmp)",
    )
    parser.add_argument("--tag-prefix", default="TP")
    parser.add_argument("--results", default="")
    parser.add_argument("--io-capture", default="auto", choices=("auto", "off"))
    parser.add_argument(
        "--graceful-exit",
        dest="hard_exit",
        action="store_false",
        help="close Isaac normally instead of bypassing Kit teardown",
    )
    parser.set_defaults(
        human_object_interface="full_ids",
        find_human_control="dynamic-guarded",
        ask_human_control="strict",
        guide_human_turn_guard="on",
        watch_human_turn_guard="on",
        orchestrate_human_turn_guard="on",
        human_frame_policy="recent",
        human_policy="auto",
        progress_line="off",
        action_retries=2,
        watch_preassistant_recovery_allowance=8,
        preflight_stand_reuse="on",
        stop_on_rejected_intervention=False,
        hard_exit=True,
    )
    args = parser.parse_args(argv)
    args.arm = args.assistant
    _validate_parsed_args(parser, args)
    _configure_provider_args(parser, args)
    if args.assistant != "model":
        args.assistant_model = f"taskpack-control/{args.assistant}"
    if args.human_backend == "scripted":
        args.human_model = "taskpack-control/scripted-human"
    return args


def _configure_render_policy(args: argparse.Namespace) -> None:
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.environ["INTERACTIVEWORLD_OUTPUT_DIR"] = args.output_dir
    os.environ["ISAAC_FRAME_BUDGET"] = str(int(args.frame_budget))
    os.environ["ISAAC_FRAMES_PER_SEG"] = str(int(args.per_interval))
    os.environ["ISAAC_RENDER_SELECTED_ONLY"] = "1"
    os.environ["ISAAC_PREFLIGHT_STAND_REUSE"] = (
        "1" if args.preflight_stand_reuse == "on" else "0"
    )


def load_frozen_tasks(  # noqa: C901
    args: argparse.Namespace,
) -> list[FrozenTaskInput]:
    manifest_path = Path(args.task_manifest).resolve()
    try:
        validate_release_task_manifest(manifest_path)
    except TaskManifestValidationError as exc:
        raise TaskPackEvalError(f"invalid release task manifest: {exc}") from exc
    rubric = load_rubric()
    validate_task_manifest(rubric, manifest_path)
    release_tasks = task_index(rubric)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    openings_entry = manifest["interaction_openings"]
    openings_path = manifest_path.parent / openings_entry["path"]
    openings_payload = json.loads(openings_path.read_text(encoding="utf-8"))
    opening_catalog = openings_payload["tasks"]
    if not isinstance(opening_catalog, dict):
        raise TaskPackEvalError("interaction opening catalog tasks must be an object")
    if args.task_id:
        row = next(
            (item for item in manifest["tasks"] if item["task_id"] == args.task_id),
            None,
        )
        if row is None:
            raise TaskPackEvalError(f"unknown release task_id: {args.task_id}")
        path = manifest_path.parent / row["taskpack_path"]
        return [_freeze_taskpack(str(path), row["taskpack_sha256"], opening_catalog)]
    if args.taskpack:
        if not args.taskpack_sha256:
            raise TaskPackEvalError(
                "--taskpack-sha256 is required for an explicit TaskPack"
            )
        with open(args.taskpack, encoding="utf-8") as handle:
            candidate = json.load(handle)
        task_id = str(candidate.get("task_id") or "")
        supplied = args.taskpack_sha256
        if not supplied.startswith("sha256:"):
            supplied = f"sha256:{supplied}"
        # A Balanced120 ID must be the frozen file; any other ID is a custom TaskPack
        # pinned only by the caller's digest.
        expected = _expected_taskpack_sha256(release_tasks, task_id, supplied)
        frozen = _freeze_taskpack(args.taskpack, expected, opening_catalog)
        return [frozen]
    batch_path = Path(args.taskpack_batch).resolve()
    with batch_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("tasks") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TaskPackEvalError(
            "--taskpack-batch must contain a non-empty tasks[] list"
        )
    frozen_tasks: list[FrozenTaskInput] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TaskPackEvalError(f"taskpack batch row {index} is not an object")
        raw_path = str(row.get("taskpack_path") or row.get("taskpack") or "")
        path = Path(raw_path)
        if not path.is_absolute():
            path = batch_path.parent / path
        expected = str(row.get("taskpack_sha256") or row.get("sha256") or "")
        if not raw_path or not expected:
            raise TaskPackEvalError(
                f"taskpack batch row {index} needs taskpack_path and taskpack_sha256"
            )
        if not expected.startswith("sha256:"):
            expected = f"sha256:{expected}"
        requested = str(row.get("task_id") or "")
        if not requested:
            raise TaskPackEvalError(f"taskpack batch row {index} needs task_id")
        expected = _expected_taskpack_sha256(release_tasks, requested, expected)
        frozen = _freeze_taskpack(str(path), expected, opening_catalog)
        actual = str(frozen.pack["task_id"])
        if requested and requested != actual:
            raise TaskPackEvalError(
                f"taskpack batch row {index} requested {requested!r}, "
                f"pack has {actual!r}"
            )
        frozen_tasks.append(frozen)
    return frozen_tasks


def _write_error_artifacts(
    out_dir: str, tag: str, record: Mapping[str, object]
) -> None:
    body = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    for suffix in ("eval", "taskpack_eval"):
        with open(
            os.path.join(out_dir, f"{tag}_{suffix}.json"), "w", encoding="utf-8"
        ) as handle:
            handle.write(body)


def _open_taskpack_scene(
    env: Any,
    scene: str,
    args: argparse.Namespace,
    frozen: FrozenTaskInput,
) -> tuple[Any, dict]:
    if env is None:
        from simulator.interaction import episode_loop as bel

        register_runtime_builder_spec(
            scene, frozen.builder_spec, frozen.builder_spec_digest
        )
        return bel._open_or_switch(env, scene, args)
    observation = env.switch_scene(
        scene,
        int(args.seed),
        builder_spec=frozen.builder_spec,
        builder_spec_digest=frozen.builder_spec_digest,
    )
    return env, observation


def _error_record(
    record: dict[str, object],
    exc: Exception,
    proto: Any,
    backend: AssistantBackend,
    backend_snapshot: tuple[dict[str, int], int],
    frozen: FrozenTaskInput,
    started: float,
    interaction: TaskPackInteractionSession | None = None,
) -> None:
    record.pop("run_evidence_sha256", None)
    backend_stats = _backend_stats_since(backend, backend_snapshot)
    record.update(
        {
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "wall_s": round(time.time() - started, 1),
            "taskpack_file_sha256": frozen.taskpack_file_sha256,
            "taskpack_digest": frozen.taskpack_digest,
            "builder_spec_digest": frozen.builder_spec_digest,
            "assistant_backend": backend.name,
            "find_human_control": getattr(
                proto, "_find_human_control", "dynamic-guarded"
            ),
            "ask_human_control": getattr(proto, "_ask_human_control", "strict"),
            "watch_human_turn_guard": bool(
                getattr(proto, "_interaction_watch_human_turn_guard", True)
            ),
            "assistant_turns": list(getattr(proto, "_assistant_turn_trace", ())),
            "human_action_trace": list(getattr(proto, "_human_action_trace", ())),
            "transcript": list(getattr(proto, "_transcript", ())),
            "process_metrics": _process_metrics(proto, frozen, interaction),
            "counters": {
                "assistant_backend_stats": backend_stats,
                "eval_valid": False,
            },
            "validity_evidence": {
                "scene_setup_ok": None,
                "runtime_complete": False,
                "trace_complete": None,
                "frames_usable": None,
                "assistant_transport_ok": None,
                "human_executor_ok": None,
                "simulator_ok": None,
                "no_material_human_leakage": None,
            },
        }
    )


def _run_one_task(
    args: argparse.Namespace,
    frozen: FrozenTaskInput,
    proto: Any,
    assistant: AssistantBackend,
    env: Any,
    out_dir: str,
) -> tuple[Any, dict[str, object]]:
    from simulator.interaction import episode_loop as bel

    task_id = str(frozen.pack["task_id"])
    scene = f"spec:{task_id}"
    interaction_tick_budget = (
        frozen.interaction.authored_tick_budget
        if frozen.interaction is not None
        else frozen.runtime.tick_budget
    )
    proto.max_ticks = int(args.max_ticks or interaction_tick_budget)
    proto._taskpack_action_retries = max(0, int(args.action_retries))
    proto._taskpack_stop_on_rejected_intervention = bool(
        args.stop_on_rejected_intervention
    )
    bel.reset_protocol(proto)
    proto._interaction_human_policy = _human_policy(args)
    proto._find_human_control = args.find_human_control
    proto._ask_human_control = args.ask_human_control
    proto._scripted_human_started = False
    if (
        proto._interaction_human_policy == "scripted"
        and frozen.interaction is not None
        and not frozen.interaction.oracle_commands
    ):
        raise TaskPackEvalError(
            f"{task_id}: scripted Human policy requires runtime.oracle_commands"
        )
    proto._assistant_turn_trace = []
    proto._human_action_trace = []
    assistant.reset()
    tag = f"{args.tag_prefix}_{task_id}_{args.assistant}_s{args.seed}"
    validate_artifact_tag(tag)
    record: dict[str, object] = {
        "task_id": task_id,
        "taskpack_file_sha256": frozen.taskpack_file_sha256,
        "arm": args.assistant,
        "assistant_control": args.assistant,
        "assistant_backend": args.assistant_backend,
        "assistant_model": args.assistant_model,
        "assistant_output_protocol": _assistant_output_protocol_metadata(args),
        "human_backend": args.human_backend,
        "human_model": args.human_model,
        "human_policy": _human_policy(args),
        "human_object_interface": args.human_object_interface,
        "find_human_control": args.find_human_control,
        "ask_human_control": args.ask_human_control,
        "watch_human_turn_guard": args.watch_human_turn_guard,
        "seed": int(args.seed),
        "tag": tag,
    }
    started = time.time()
    capture = None
    dlog = None
    interaction = None
    backend_snapshot = _backend_snapshot(assistant)
    try:
        reset_task_artifacts(out_dir, tag)
        env, observation = _open_taskpack_scene(env, scene, args, frozen)
        from simulator.physical_world import dialogue_log

        dlog = dialogue_log.DialogueLog.beside_manifest(
            out_dir,
            tag,
            task_id=task_id,
            scene=scene,
            seed=int(args.seed),
            arm=args.assistant,
            participants={
                "assistant": args.assistant_model,
                "assistant_backend": args.assistant_backend,
                "assistant_output_protocol": _assistant_output_protocol_metadata(args),
                "human": (
                    args.human_model
                    if _human_policy(args) == "model"
                    else "taskpack-control/scripted-human"
                ),
                "human_backend": (
                    args.human_backend if _human_policy(args) == "model" else "scripted"
                ),
                "human_object_interface": args.human_object_interface,
                "find_human_control": args.find_human_control,
                "ask_human_control": args.ask_human_control,
                "guide_human_turn_guard": args.guide_human_turn_guard,
                "watch_human_turn_guard": args.watch_human_turn_guard,
                "orchestrate_human_turn_guard": (args.orchestrate_human_turn_guard),
                "human_url": (
                    args.human_base_url if _human_policy(args) == "model" else ""
                ),
                "human_role": (
                    "model-driven family-conditioned interactive person"
                    if _human_policy(args) == "model"
                    else "deterministic oracle demonstration"
                ),
                "protocol": "simulator/interaction/episode_loop.py",
                "taskpack_digest": frozen.taskpack_digest,
                "interaction_family": (
                    frozen.interaction.family if frozen.interaction is not None else ""
                ),
            },
        )
        if _human_policy(args) == "model" and args.io_capture != "off":
            from simulator.interaction import io_capture

            capture = io_capture.attach(
                assistant,
                out_dir,
                tag,
                {
                    "task_id": task_id,
                    "arm": args.assistant,
                    "assistant_model": args.assistant_model,
                    "assistant_output_protocol": _assistant_output_protocol_metadata(
                        args
                    ),
                    "human_object_interface": args.human_object_interface,
                    "find_human_control": args.find_human_control,
                    "ask_human_control": args.ask_human_control,
                    "watch_human_turn_guard": args.watch_human_turn_guard,
                    "taskpack_digest": frozen.taskpack_digest,
                    "builder_spec_digest": frozen.builder_spec_digest,
                },
            )
        if frozen.interaction is None:
            raise TaskPackEvalError("frozen TaskPack has no interaction specification")
        interaction = TaskPackInteractionSession(
            frozen.interaction,
            human_object_interface=args.human_object_interface,
        )
        final_observation = bel.run_episode(
            env,
            proto,
            assistant,
            dlog,
            args,
            observation,
            task_id=task_id,
            interaction=interaction,
        )
        payload = _write_artifacts(
            env,
            proto,
            assistant,
            dlog,
            args,
            frozen,
            out_dir,
            tag,
            started,
            _backend_stats_since(assistant, backend_snapshot),
            interaction,
            final_observation,
        )
        record.update(bel.verdict_from_manifest(out_dir, tag))
        record["manifest"] = f"{tag}_manifest.json"
        record["dialogue_log"] = f"{tag}_dialogue.json"
        record["counters"] = payload["counters"]
        record["taskpack_digest"] = payload["taskpack_digest"]
        record["builder_spec_digest"] = payload["builder_spec_digest"]
        record["process_metrics"] = payload["process_metrics"]
        record["validity_evidence"] = payload["validity_evidence"]
        record["native_taskpack_outcome"] = payload["native_taskpack_outcome"]
        record["release_rubric_outcome"] = payload["release_rubric_outcome"]
        if capture is not None:
            record["io_capture"] = capture.close(record)
            assistant.capture = None
            capture = None
        run_evidence_sha256 = compute_run_evidence_sha256(out_dir, tag)
        record["run_evidence_sha256"] = run_evidence_sha256
        payload["run_evidence_sha256"] = run_evidence_sha256
        _write_evaluation_artifacts(out_dir, tag, payload)
        verify_run_evidence_sha256(out_dir, tag, run_evidence_sha256)
    # Preserve one invalid row and continue the shard.
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _error_record(
            record,
            exc,
            proto,
            assistant,
            backend_snapshot,
            frozen,
            started,
            interaction,
        )
        if capture is not None:
            try:
                record["io_capture"] = capture.close(record)
            finally:
                assistant.capture = None
        if dlog is not None:
            try:
                dlog.note(f"episode aborted: {record['error']}")
                dlog.close(stop_reason="error")
            except Exception:  # noqa: BLE001 - failure artifact is best effort
                pass
        _write_error_artifacts(out_dir, tag, record)
    record["wall_s"] = round(time.time() - started, 1)
    return env, record


def _selftest() -> int:
    effect_beat = AuthoredHumanBeat(
        beat_id="o000",
        command="[Grab] <item_1> (1)",
        say="",
        definition={
            "effects": [
                {"op": "held", "subject": "item_1"},
                {"op": "inside", "subject": "item_2", "target": "unit_a"},
            ]
        },
    )

    pack = {
        "story": {
            "beats": [{"actor": "assistant", "say": "fallback"}],
        },
        "runtime": {
            "dialogue": [{"say": "first"}, {"say": "second"}],
            "human_policy": {"intervention_window": {"after_beat": "turn0"}},
        },
    }
    manifest_goal = _task_goal(
        {"success": False, "n_done": 0, "n_total": 2, "sub_goals": []},
        "TASK-1",
    )

    class _SwitchEnv:
        def __init__(self) -> None:
            self.call: tuple[object, ...] = ()

        def switch_scene(self, *values: object, **named: object) -> dict[str, bool]:
            self.call = (*values, named)
            return {"switched": True}

    switch_env = _SwitchEnv()
    frozen = FrozenTaskInput(
        taskpack_path="/tmp/taskpack.json",
        pack={"task_id": "TASK-2"},
        runtime=None,  # pyre-ignore[6] - switch forwarding does not inspect runtime
        builder_spec={"task_id": "TASK-2"},
        taskpack_file_sha256="sha256:file",
        taskpack_digest="sha256:pack",
        builder_spec_digest="sha256:builder",
        interaction=None,
    )
    switched_env, switched_observation = _open_taskpack_scene(
        switch_env, "spec:TASK-2", argparse.Namespace(seed=7), frozen
    )
    checks = {
        "reference dialogue summary": _reference_dialogue_text(pack) == "first second",
        "intervention anchor": _intervention_after(pack) == "turn0",
        "command normalization": _command_key("[walk] <unit_a> (1)")
        == _command_key("[Walk] <unit_a> (1)"),
        "wait detection": _is_wait("[Wait]"),
        "preparatory action detection": _is_preparatory("[LookAround]"),
        "prologue evidence frame": _should_keep_prologue_frame(
            "[PutInside] <item_1> (1) <unit_a> (1)", 1, 3
        ),
        "live observation normalization": _policy_observation(
            {
                "relations": {("Pred.INSIDE", "item", "bin")},
                "held": {"item"},
                "at": {"table"},
            }
        )
        == {
            "relations": [["Pred.INSIDE", "item", "bin"]],
            "held": ["item"],
            "at": ["table"],
        },
        "manifest task score shape": manifest_goal["task"] == "TASK-1"
        and manifest_goal["n"] == 2,
        "local step reports": _step_failures(
            {}, [{"ok": False, "reason": "OUT_OF_REACH", "detail": ""}]
        )
        == ["OUT_OF_REACH"],
        "authored effects": _effect_status(
            effect_beat,
            {
                "relations": {("inside", "item_2", "unit_a")},
                "held": {"item_1"},
                "at": set(),
            },
        )
        == (True, []),
        "warm switch receives dynamic TaskPack": switched_env is switch_env
        and switched_observation == {"switched": True}
        and switch_env.call
        == (
            "spec:TASK-2",
            7,
            {
                "builder_spec": {"task_id": "TASK-2"},
                "builder_spec_digest": "sha256:builder",
            },
        ),
        "semantic rejection short circuit": _should_short_circuit_rejection(
            True,
            "turn0",
            "turn0",
            {
                "human_policy_status": "READY",
                "human_policy_matched_condition": None,
            },
        )
        and not _should_short_circuit_rejection(
            True,
            "turn0",
            "turn0",
            {"human_policy_status": "READY", "intervention_consumed": True},
        ),
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


def _finish_process(env: Any, rc: int, hard_exit: bool) -> int:
    if env is None or not hard_exit:
        return rc
    # The episode artifacts and results row are already closed before this point.
    # Preserve the render-parity trace, but bypass Kit's app.close/plugin teardown:
    # it can hang after printing "Simulation App Shutting Down" and prevent the lane
    # scheduler from observing this status.
    env.finalize_render_parity_trace()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def main(argv: list[str] | None = None) -> int:
    if argv is None and "--selftest" in sys.argv[1:]:
        return _selftest()

    args = _parse_args(argv)
    _configure_render_policy(args)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    from simulator.interaction import episode_loop as bel

    bel.assert_frame_budget()
    frozen_tasks = load_frozen_tasks(args)
    human_gateway = None
    assistant = None
    env = None
    rc = 0
    try:
        if _human_policy(args) == "model":
            human_gateway = make_human_gateway(
                args.human_provider,
                model=args.human_model,
                base_url=args.human_base_url,
                api_key_env=args.human_api_key_env,
                timeout_s=float(args.human_timeout_s),
                max_new_tokens=int(args.human_tokens),
            ).start()
            args.human_url = human_gateway.url
            print(
                "[taskpack_eval] human probe -> "
                f"{bel.probe_human(args.human_url, timeout_s=args.human_timeout_s)[:60]!r}"
            )
        else:
            print("[taskpack_eval] human policy -> scripted oracle trace")
        assistant = _make_assistant(args, frozen_tasks[0])
        if args.assistant == "model":
            print(f"[taskpack_eval] assistant probe -> {assistant.probe()[:60]!r}")
        proto = bel.build_protocol(args)
        args.human_url = args.human_base_url
        results = args.results or os.path.join(out_dir, "results.jsonl")
        for index, frozen in enumerate(frozen_tasks):
            if index and args.assistant != "model":
                assistant.close()
                assistant = _make_assistant(args, frozen)
            env, record = _run_one_task(args, frozen, proto, assistant, env, out_dir)
            with open(results, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "[taskpack_eval] " + json.dumps(record, ensure_ascii=False),
                flush=True,
            )
            rc = max(rc, 1 if record.get("error") else 0)
    finally:
        if assistant is not None:
            assistant.close()
        if human_gateway is not None:
            human_gateway.close()
    return _finish_process(env, rc, args.hard_exit)


if __name__ == "__main__":
    raise SystemExit(main())

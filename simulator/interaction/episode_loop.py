# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Final synchronous two-agent episode protocol shared by TaskPack evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_AL_DIR: str = os.path.dirname(os.path.abspath(__file__))
_RELEASE_ROOT: str = os.path.dirname(os.path.dirname(_AL_DIR))

from .agent_protocol import (
    _arrival_clause,
    human_reprompt_guard_enabled,
    IsaacAgentLoop,
)
from .assistant_backend import select_temporal_image_batches
from .task_runtime import assert_unscored_state

# A step rejection the tick can absorb: the engine refused this command while leaving the environment
# unchanged, so the episode can continue on the next tick. Match both the exception type and message;
# some physical-world wrappers preserve only the underlying error name in their message.
_RECOVERABLE_STEP_ERRORS: tuple[str, ...] = (
    "CommandParseError",
    "GrammarError",
    "NotVisible",
    "VisibilityError",
    "unknown object",
    "not visible",
    "VISIBILITY",
)

# These are executor/planner failures for which replaying the exact accepted Human command can
# recover a transient simulation miss.  Semantic refusals (for example PRECONDITION_UNMET or
# NOT_ARTICULATED) are deliberately absent: retrying those would conceal an Assistant instruction
# error.  GRASP_FAILED, placement failures, and POSTCONDITION_FAILED are also absent because they can
# follow physical contact or release; held/score equality cannot prove that an unscored object pose
# stayed unchanged, so replaying them could duplicate a partially successful manipulation.
_MODEL_HUMAN_RETRYABLE_FAILURES: frozenset[str] = frozenset(
    {
        "NO_COLLISION_FREE_STANCE",
        "NO_FREE_PLACEMENT",
        "OUT_OF_REACH",
        "PATH_BLOCKED",
        "TRAJECTORY_COLLISION",
        "UNREADABLE_LABEL",
    }
)
_MODEL_HUMAN_COMMAND_FAILURES: frozenset[str] = frozenset(
    {
        "MISSING_AFFORDANCE",
        "NOT_ARTICULATED",
        "NO_OBJECT",
        "PRECONDITION_UNMET",
        "TARGET_TOO_SMALL",
    }
)
_MODEL_HUMAN_RETRY_SAFE_PREFIX_PRIMITIVES: frozenset[str] = frozenset()

_HUMAN_VISUAL_MEMORY_INTERVALS = 2
# Cross-turn recovery remains bounded so an invalid episode can finish for diagnosis without
# consuming its semantic tick budget. Any such chain is excluded from official scoring.
_DEFAULT_WATCH_PREASSISTANT_RECOVERY_ALLOWANCE = 8
_DEFAULT_HUMAN_INVALID_ATTEMPT_LIMIT = 8
_HUMAN_INVALID_ATTEMPT_STOP_REASON = "human_invalid_attempt_limit_exhausted"

_ASK_HUMAN_REPROMPT_TEXT: Mapping[str, str] = {
    "human_action_format_invalid": (
        "Your non-empty <action> block did not contain an executable simulator command. "
        "Re-output the same intended action using exactly one line from PRIVATE ACTION SYNTAX; "
        "do not change the object, destination, or instruction you are following."
    ),
    "human_action_unsupported_verb": (
        "The bracketed action name in your previous response is not supported. Re-output the same "
        "intended action using the matching verb from PRIVATE ACTION SYNTAX; do not change the "
        "object, destination, or instruction you are following."
    ),
    "human_action_wrong_arity": (
        "Your previous action used the wrong number of arguments. Re-output the same intended "
        "action by copying that verb's exact form from PRIVATE ACTION SYNTAX; do not change the "
        "object, destination, or instruction you are following."
    ),
    "human_motor_claim_without_action": (
        "Do not claim that a physical action succeeded unless you emit its simulator command in "
        "this response. Follow the Assistant's latest instruction with the required command, or "
        "ask a natural clarification question."
    ),
    "ask_handle_map_missing": (
        "Use only a currently listed motor name. If none matches, leave <action> empty and ask "
        "for an ordinary visible description or location."
    ),
    "ask_flip_requires_held": (
        "Flip can be executed only on the one object currently held, and the proposed target is "
        "not that held object. Do not Flip and do not invent a Grab or cleanup action. Leave "
        "<action> empty and ask naturally what physical step to do first."
    ),
    "ask_hands_full_grab_blocked": (
        "You are already holding an object, so do not grab another one. Follow the Assistant's "
        "latest instruction without inventing a place to set the held object, or ask what to do "
        "with it first."
    ),
    "ask_inspection_grab_blocked": (
        "The Assistant requested visual inspection, not a Grab. If the requested object is "
        "currently visible, use the appropriate viewing command; otherwise leave <action> empty "
        "and ask naturally where to look. Do not name an internal ID or infer the answer."
    ),
    "ask_internal_interface_speech_blocked": (
        "Speak naturally without mentioning any legend, catalog, ID, key, simulator interface, "
        "targetability, or list of available destinations. "
        "If the instruction is unclear, leave <action> empty and ask for a visible description "
        "or location."
    ),
    "ask_unobservable_attribute_speech_blocked": (
        "Do not claim an attribute that cannot be established from the current view. Briefly say "
        "that you cannot confirm it visually and ask the Assistant what observable step to take."
    ),
    "ask_multiple_commands_blocked": (
        "Emit at most one command. Choose only the first immediate action explicitly requested by "
        "the Assistant; do not queue later actions."
    ),
    "ask_premature_completion_speech_blocked": (
        "Do not decide or announce that the whole task or episode is complete. Follow the "
        "Assistant's latest explicit instruction, or ask naturally what to do next. You may still "
        "report a specific observable local action result."
    ),
    "ask_redundant_walk_blocked": (
        "You are already at that destination. Do not walk there again and do not invent a scan "
        "or Grab. If the Assistant's requested object is not visible there, leave <action> empty, "
        "say naturally that you cannot see it, and ask what to do next. Otherwise execute only "
        "the Assistant's currently requested action."
    ),
    "ask_unknown_human_handle_blocked": (
        "Use only a currently listed motor name. If none matches, leave <action> empty and ask "
        "for an ordinary visible description or location."
    ),
    "ask_unrequested_action_blocked": (
        "Your proposed action did not match the Assistant's latest explicit action, object, or "
        "destination. Do not choose a substitute from the catalog. Try again, or leave <action> "
        "empty and ask a natural clarification question."
    ),
    "ask_unseen_grab_blocked": (
        "The object is not currently visible, so do not grab it. Follow only the Assistant's "
        "requested landmark navigation or viewing action, or ask where to look."
    ),
    "ask_unseen_target_blocked": (
        "Do not target an unseen movable object. Follow only the Assistant's requested landmark "
        "navigation or viewing action, or ask where to look."
    ),
    "guide_required_inspection_missing": (
        "The Assistant explicitly requested an inspection before closing. Opening a container or "
        "seeing its contents does not perform that inspection. Use LookAt on the requested visible "
        "item before Close, and do not claim either action succeeded before it executes."
    ),
    "guide_active_instruction_mute_blocked": (
        "The Assistant still has an active instruction. Do not silently idle: execute exactly its "
        "first unfinished action, ask one concrete grounding question, or report an actual action "
        "failure."
    ),
    "guide_ambiguous_reference_blocked": (
        "The Assistant's description matches more than one current object or place. Do not pick a "
        "private handle by guessing; leave <action> empty and ask for a distinguishing visible "
        "description or location."
    ),
    "guide_unknown_human_handle_blocked": (
        "Use only a currently listed motor name. A handle from an earlier view may no longer name "
        "the same visible object; ask for a visible description or location if no current handle "
        "matches the Assistant's instruction."
    ),
    "guide_movable_source_conflict_blocked": (
        "The selected movable object's tracked current location conflicts with the source named by "
        "the Assistant. Do not reuse a previously seen handle merely because its description looks "
        "similar. Choose only an item currently tracked at the named source; if none is identifiable, "
        "leave <action> empty and ask for a distinguishing visible description."
    ),
    "guide_instruction_mismatch_blocked": (
        "Your proposed action, object, or destination is not the next action authorized by the "
        "Assistant's current instruction. Keep completed steps completed and retry only the next "
        "explicit step, or a Grab, Walk, or Open strictly needed for its current placement."
    ),
    "guide_incomplete_instruction_next_step_blocked": (
        "Do not ask for the next step yet. The Assistant's current multi-step instruction is still "
        "active. Continue with its first unfinished action and report completion only after every "
        "requested action has actually succeeded."
    ),
    "guide_incomplete_instruction_completion_claim_blocked": (
        "Do not say that the current step or instruction is finished until every requested action "
        "has succeeded. Continue with its first unfinished action, and report only actions already "
        "completed."
    ),
    "guide_inference_conclusion_speech_blocked": (
        "Report only what is directly visible or whether the requested local action succeeded. "
        "Do not infer, classify, or explain what a landmark is for; leave that reasoning to the "
        "Assistant."
    ),
    "guide_failed_placement_claim_blocked": (
        "The last placement failed. Do not say that the object is in place. State the failure "
        "briefly and ask the Assistant how to recover, without issuing an unrequested action."
    ),
    "guide_unseen_movable_blocked": (
        "Do not use an unseen movable object's handle to navigate or act. Follow a concrete "
        "location or visible target given by the Assistant; otherwise ask where to look."
    ),
    "guide_unverified_progress_speech_blocked": (
        "The current instruction is not complete. Keep <speak> empty while executing it unless you "
        "need a concrete clarification or must report an actual failed action. Do not announce an "
        "open, closed, inspected, placed, or completed state before matching actions succeed."
    ),
    "guide_visual_evidence_speech_blocked": (
        "The Assistant receives the same camera view. Do not describe an inspected object's identity, "
        "label, category, or contents. Execute the requested inspection silently; after the whole local "
        "instruction succeeds, report only that the check is done and ask for the next step."
    ),
    "narrate_internal_interface_speech_blocked": (
        "Your previous spoken sentence referred to private implementation details. Do not discuss "
        "those details with the Assistant. If you cannot safely identify what the Assistant means, "
        "leave <action> empty and ask one ordinary question such as 'Where is it relative to me?'"
    ),
    "narrate_handle_map_missing": (
        "The private action references needed to execute your previous command were unavailable. "
        "Do not invent a reference. Re-output the same intended action only if its listed reference "
        "is now available; otherwise leave <action> empty and ask a natural clarification question."
    ),
    "narrate_unknown_human_handle_blocked": (
        "Your previous command used a private action reference that is not currently listed. "
        "Re-output the same intended action with a currently listed reference, or leave <action> "
        "empty and ask a natural clarification question."
    ),
    "narrate_hidden_entity_speech_blocked": (
        "Your previous reply introduced an object or place that the Assistant did not state. Do not "
        "offer that possible answer or route. Mention only the fixed request, the Assistant's current "
        "instruction, or an object you are already holding; otherwise ask naturally where to go or look."
    ),
    "narrate_multiple_commands_blocked": (
        "Emit only the first immediate physical action explicitly requested by the Assistant. Use "
        "at most one command in this response, and do not claim that any later action happened."
    ),
    "narrate_completed_action_repeat_blocked": (
        "That exact action already succeeded under the Assistant's still-active instruction. Do not "
        "restart or repeat completed steps. Continue only with an explicit unfinished step; if none "
        "remains, leave <action> empty and ask the Assistant what to do next."
    ),
    "narrate_ungrounded_movable_blocked": (
        "Do not travel to or operate an unseen movable object just because a private action name "
        "is available. Leave <action> empty and ask naturally where the object is relative to you."
    ),
    "narrate_unrequested_action_blocked": (
        "Your proposed physical action, object, or destination was not requested by the Assistant's "
        "current instruction. Follow exactly one requested action without substituting a private "
        "option. Do not add a preparatory Walk before Grab or Open: when the Assistant gave one concrete "
        "location, emit the requested Grab or Open directly because the executor handles the approach. "
        "If the location is insufficient, leave <action> empty and ask one short natural question."
    ),
    "orchestrate_action_target_mismatch_blocked": (
        "One proposed action targeted a different object from the Assistant's current instruction. "
        "Retry the same instruction using only the explicitly named object for each Cook, Cut, Pour, "
        "TurnOn, or TurnOff action. Do not add or substitute a task step."
    ),
    "orchestrate_completed_action_repeat_blocked": (
        "One proposed action repeats a clause that already succeeded under this same active "
        "instruction. Keep every successful clause completed and continue with the first unfinished "
        "clause. A repeated use is allowed only when the Assistant explicitly requested it again."
    ),
    "orchestrate_unrequested_action_blocked": (
        "One proposed action was not explicitly requested by the Assistant and was not a local Grab, "
        "Walk, or Open needed to perform its requested placement. Remove that action and follow only "
        "the current instruction; never add an appliance state change or cooking step yourself."
    ),
    "watch_assistant_state_override_blocked": (
        "Your previous reply tried to correct or bypass the Assistant using current-state or "
        "private-plan inference. Execute the latest Assistant instruction literally even if you "
        "believe it is redundant or wrong. Ask only if its object or destination cannot be "
        "uniquely grounded, or an object already in your hand makes its first action impossible."
    ),
    "watch_assistant_instruction_ignored_blocked": (
        "Your previous reply did not execute the latest actionable Assistant instruction. "
        "Execute that instruction literally now, even if you believe it is redundant or wrong. "
        "Ask only if its object or destination cannot be uniquely grounded, or an object already "
        "in your hand makes its first action impossible."
    ),
}


def is_recoverable_step_error(exc: BaseException) -> bool:
    """Can this tick absorb ``exc`` and carry on, or must the episode end?

    The exception's type chain is consulted first, then its text for wrapped physical-world errors.
    """
    if {t.__name__ for t in type(exc).__mro__} & set(_RECOVERABLE_STEP_ERRORS):
        return True
    return any(k in str(exc) for k in _RECOVERABLE_STEP_ERRORS)


def _step_outcomes_and_failures(
    observation: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    outcomes = [
        dict(value)
        for value in (observation.get("outcomes") or ())
        if isinstance(value, Mapping)
    ]
    reasons = [str(value) for value in (observation.get("reasons") or ())]
    failures = [
        str(value.get("detail") or "").strip() or str(value.get("reason") or "").strip()
        for value in outcomes
        if value.get("ok") is not True
    ] or [value for value in reasons if value and value != "OK"]
    return outcomes, [value for value in failures if value and value != "OK"]


def _model_human_action_retry_decision(
    observation_before: Mapping[str, object],
    observation_after: Mapping[str, object],
) -> dict[str, object]:
    """Classify a failed model-Human step without guessing whether it partially committed."""
    outcomes, failures = _step_outcomes_and_failures(observation_after)
    failed_outcomes = [value for value in outcomes if value.get("ok") is not True]
    if not failures:
        return {"kind": "success", "retry": False, "reasons": []}

    reason_names = {
        str(value.get("reason") or "").strip().upper() for value in failed_outcomes
    }
    reason_names.discard("")
    if reason_names and reason_names <= _MODEL_HUMAN_COMMAND_FAILURES:
        return {
            "kind": "command_failure",
            "retry": False,
            "reasons": sorted(reason_names),
        }
    if not reason_names or not reason_names <= _MODEL_HUMAN_RETRYABLE_FAILURES:
        return {
            "kind": "unattributed_failure",
            "retry": False,
            "reasons": sorted(reason_names),
        }

    unsafe: list[str] = []
    if "held" not in observation_before or "held" not in observation_after:
        unsafe.append("held_state_unavailable")
    elif set(observation_before.get("held") or ()) != set(
        observation_after.get("held") or ()
    ):
        unsafe.append("held_state_changed")
    if "score" not in observation_before or "score" not in observation_after:
        unsafe.append("score_state_unavailable")
    elif observation_before.get("score") != observation_after.get("score"):
        unsafe.append("score_state_changed")
    successful_primitives = {
        str(value.get("primitive") or "").strip().lower()
        for value in outcomes
        if value.get("ok") is True
    }
    unsafe_successes = successful_primitives - _MODEL_HUMAN_RETRY_SAFE_PREFIX_PRIMITIVES
    if unsafe_successes:
        unsafe.append(
            "successful_non_idempotent_prefix=" + ",".join(sorted(unsafe_successes))
        )
    if unsafe:
        return {
            "kind": "unsafe_partial_state",
            "retry": False,
            "reasons": sorted(reason_names),
            "unsafe": unsafe,
        }
    return {
        "kind": "retryable_transient",
        "retry": True,
        "reasons": sorted(reason_names),
    }


def _execute_model_human_action_attempt(
    env: Any,
    proto: Any,
    command: str,
    observation_before: dict,
    *,
    attempt: int,
    classify_failure: bool,
    can_retry: bool,
) -> tuple[dict, dict[str, object], Exception | None]:
    started = time.perf_counter()
    try:
        observation_after = env.step(command)
    except Exception as exc:  # noqa: BLE001 - caller classifies the simulator boundary error
        proto._tadd("t_step_s", started)
        return (
            observation_before,
            {
                "attempt": attempt,
                "command": command,
                "result": "rejected",
                "error": str(exc)[:300],
                "retry_decision": "exception",
                "retry_reasons": [],
                "retry_unsafe": [],
                "will_retry": False,
            },
            exc,
        )
    proto._n["env_steps"] += 1
    proto._tk["t_step_s"].append(round(proto._tadd("t_step_s", started), 3))
    outcomes, failures = _step_outcomes_and_failures(observation_after)
    decision = (
        _model_human_action_retry_decision(observation_before, observation_after)
        if classify_failure
        else {
            "kind": "success" if not failures else "retry_disabled",
            "retry": False,
            "reasons": [],
        }
    )
    return (
        observation_after,
        {
            "attempt": attempt,
            "command": command,
            "result": "failed" if failures else "succeeded",
            "failures": failures,
            "outcomes": outcomes,
            "retry_decision": str(decision.get("kind") or ""),
            "retry_reasons": list(decision.get("reasons") or ()),
            "retry_unsafe": list(decision.get("unsafe") or ()),
            "will_retry": bool(decision.get("retry")) and can_retry,
            "_observation": observation_after,
        },
        None,
    )


def _note_model_human_retry_terminal(
    proto: Any,
    attempt: int,
    decision: str,
    *,
    retry_enabled: bool,
    invalidate_failure: bool,
) -> None:
    if decision == "command_failure" and attempt == 1:
        return
    if retry_enabled and (attempt > 1 or decision == "retryable_transient"):
        proto._n["action_retry_exhausted"] = (
            proto._n.get("action_retry_exhausted", 0) + 1
        )
    elif decision == "retryable_transient":
        proto._n["action_retry_transient_failure"] = (
            proto._n.get("action_retry_transient_failure", 0) + 1
        )
    elif decision == "unsafe_partial_state":
        proto._n["action_retry_unsafe_partial_state"] = (
            proto._n.get("action_retry_unsafe_partial_state", 0) + 1
        )
    elif decision == "unattributed_failure":
        proto._n["action_retry_unattributed_failure"] = (
            proto._n.get("action_retry_unattributed_failure", 0) + 1
        )
    else:
        return
    if not invalidate_failure:
        return
    if retry_enabled and attempt > 1 and decision == "unsafe_partial_state":
        proto._n["action_retry_unsafe_partial_state"] = (
            proto._n.get("action_retry_unsafe_partial_state", 0) + 1
        )
    if retry_enabled and attempt > 1 and decision == "unattributed_failure":
        proto._n["action_retry_unattributed_failure"] = (
            proto._n.get("action_retry_unattributed_failure", 0) + 1
        )
    proto._n["action_retry_invalid"] = proto._n.get("action_retry_invalid", 0) + 1


def _run_model_human_action_attempts(
    env: Any,
    proto: Any,
    command: str,
    observation: dict,
    *,
    retry_enabled: bool,
    classify_failures: bool | None = None,
    invalidate_classified_failures: bool | None = None,
) -> tuple[dict, list[dict[str, object]], Exception | None]:
    """Execute one logical Human action, replaying only proven-safe transient failures."""
    classify_failures = (
        retry_enabled if classify_failures is None else bool(classify_failures)
    )
    invalidate_classified_failures = (
        classify_failures
        if invalidate_classified_failures is None
        else bool(invalidate_classified_failures)
    )
    retry_limit = (
        max(0, int(getattr(proto, "_taskpack_action_retries", 0)))
        if retry_enabled
        else 0
    )
    attempts: list[dict[str, object]] = []
    for attempt in range(1, retry_limit + 2):
        observation, record, error = _execute_model_human_action_attempt(
            env,
            proto,
            command,
            observation,
            attempt=attempt,
            classify_failure=classify_failures,
            can_retry=attempt <= retry_limit,
        )
        attempts.append(record)
        if error is not None:
            if classify_failures:
                counter = (
                    "action_retry_human_command_rejection"
                    if is_recoverable_step_error(error)
                    else "action_retry_ambiguous_exception"
                )
                proto._n[counter] = proto._n.get(counter, 0) + 1
                proto._n["action_retry_invalid"] = (
                    proto._n.get("action_retry_invalid", 0) + 1
                )
            return observation, attempts, error
        if record["result"] == "succeeded":
            if attempt > 1:
                proto._n["action_retry_recovered"] = (
                    proto._n.get("action_retry_recovered", 0) + 1
                )
            return observation, attempts, None
        if record["will_retry"]:
            proto._n["action_effect_retries"] = (
                proto._n.get("action_effect_retries", 0) + 1
            )
            continue
        if classify_failures:
            _note_model_human_retry_terminal(
                proto,
                attempt,
                str(record["retry_decision"]),
                retry_enabled=retry_enabled,
                invalidate_failure=invalidate_classified_failures,
            )
        return observation, attempts, None
    raise AssertionError("model-Human action attempt loop did not terminate")


class EvalConfigError(RuntimeError):
    """A startup condition that would silently produce a full-length, empty episode."""


class HumanInvalidAttemptLimitError(RuntimeError):
    """A Human policy failure that must end the episode before another logical tick."""

    def __init__(self, guard: str, row: dict[str, object] | None) -> None:
        super().__init__(
            f"Human policy guard {guard!r} exhausted its same-tick retries"
        )
        self.guard = guard
        self.row = row


def _parse_provenance(
    proto: Any,
    cmd: str,
    *,
    scripted: bool = False,
    qualified_from: str | None = None,
) -> dict:
    """``dialogue_log.parse_record`` for one command, carrying whether the LENIENT path rewrote it."""
    import dialogue_log  # noqa: PLC0415 - sibling on the simulator.physical_world path added above

    source_command = qualified_from or cmd
    if scripted:
        record = dialogue_log.override_record(
            source_command, cmd, reason="scripted_oracle_human"
        )
    else:
        prov = proto.repair_provenance(source_command)
        record = dialogue_log.parse_record(
            prov["repaired_from"] or source_command,
            strict=prov["strict_parse"],
            command=source_command,
            repair_kind=prov["repair_kind"],
        )
    if qualified_from is not None:
        prior_kind = str(record.get("repair_kind") or "")
        record.update(
            {
                "strict_parse": False,
                "command": cmd,
                "repaired": True,
                "repaired_from": record.get("repaired_from") or qualified_from,
                "repaired_to": cmd,
                "repair_kind": "+".join(
                    filter(None, (prior_kind, "authored_putinside_qualifier"))
                ),
            }
        )
    return record


# ---------------------------------------------------------------------------------------------------
#  what each agent is told the task IS
# ---------------------------------------------------------------------------------------------------
def _authored_row(task_id: str) -> dict:
    """Legacy fallback; frozen TaskPacks carry their own goal and opening."""
    del task_id
    return {}


def _authored_prologue(task_id: str) -> dict:
    """The row's SCRIPTED OPENING: actions the person performs and the question they then ask.

    ⭐ This is better than deleting the premise. ``design.premise`` describes a
    situation -- *"the human clears a bottle off the bench, stops in front of the cabinet wall and asks
    'which one does this go in?'"*. Rather than discard it as a stage direction, MAKE IT TRUE: play the
    opening deterministically and seed the question, so the episode starts where the premise says it
    starts. Five things follow, and only the first is about the goal text:

      1. the assistant has something to ANSWER -- the person asks on turn 0, ending the deadlock in
         which the assistant waits for a question the person never asks
      2. no ticks are burned on navigation and search, which is not what these rows measure
      3. the containers are already in frame, so the visibility wall is out of the opening
      4. every model faces a byte-identical start state -- no model variance in the setup
      5. the premise stops being a lie

    ⚠️ CONSTRAINTS, checked rather than trusted:
      * **no scored action.** Grabbing the item is fine; placing it is not. Only ``Walk``/``Grab`` and
        the look verbs are permitted here, and ``_assert_prologue_safe`` refuses anything else.
      * **no container key and no placard**, in the actions or the seeded speech. The person may ask
        "which one does this go in?"; it may not ask "is it LINE CHARLIE?".
    """
    ent = _authored_row(task_id)
    pro = ent.get("prologue") or {}
    return {
        "actions": list(pro.get("actions") or []),
        "say": str(pro.get("say") or "").strip(),
    }


def _bank_anchor(env: Any, task_id: str) -> str:
    """A DECOR entity standing at the container bank -- "stops in front of the cabinet wall".

    ⛔ MEASURED, and it is why this exists. With the prologue at ``Walk item / Grab item / LookAround``,
    the delivered turn-0 frames for an office row whose item sits in a bathroom are sinks, mirrors and
    a hand dryer: the person sweeps a bathroom and **not one answer container is in view**. The
    assistant would be asked to name a container it has never seen, and with everything else looking
    correct, that would be misread as a model failure.

    ⚠️ Walking to the WALL is context; walking to ``unit_c`` is the answer. So the anchor is chosen
    ANSWER-INDEPENDENTLY by construction: take the centroid of **every** unit the row declares, answers
    and decoys alike, and return the nearest entity that is neither a unit nor a task object. The
    answer key cannot move it -- swapping which unit is correct returns the identical anchor.
    """
    import task_world_builder

    spec = task_world_builder.load_spec(task_id)
    ents = getattr(getattr(env, "scene_obj", None), "entities", None) or {}
    units = [k for k in ents if _PSEUDO_UNIT_RE.match(str(k))]
    if not units:
        return ""
    pts = [_xy(ents[k]) for k in units]
    pts = [q for q in pts if q]
    if not pts:
        return ""
    cx = sum(q[0] for q in pts) / len(pts)
    cy = sum(q[1] for q in pts) / len(pts)
    skip = set(units) | {
        str(g.get("subject"))
        for g in (spec.get("answer_key") or [])
        if isinstance(g, dict)
    }
    best, bestd = "", 1e9
    for k, e in ents.items():
        if str(k) in skip or _PSEUDO_ANY_RE.match(str(k)):
            continue
        # ⛔ Ceiling and wall fixtures are the nearest thing to a unit bank's centroid far more often
        # than furniture is, and they are useless as an anchor: e.g. a centroid's nearest entity can be
        # a ceiling light (`room_light_*`); walking to it leaves the person under a lamp with the bins
        # still occluded, and `answers_visible` stays 0. An anchor has to be something you can
        # stand in front of.
        if _FIXTURE_RE.search(str(k)):
            continue
        q = _xy(e)
        if not q:
            continue
        d = ((q[0] - cx) ** 2 + (q[1] - cy) ** 2) ** 0.5
        if d < bestd:
            best, bestd = str(k), d
    return best if bestd <= _BANK_ANCHOR_MAX_M else ""


_BANK_ANCHOR_MAX_M: float = 6.0
_FIXTURE_RE: re.Pattern = re.compile(
    r"(light|lamp|sprinkler|ceiling|vent|duct|smoke|alarm|sensor|speaker|camera|wire|pipe|"
    r"outlet|switch|thermostat|clock|painting|picture|poster|window|curtain|rug|mat|floor|wall_)",
    re.I,
)
_PSEUDO_UNIT_RE: re.Pattern = re.compile(r"^unit_[a-z0-9]+$", re.I)
_PSEUDO_ANY_RE: re.Pattern = re.compile(r"^(?:unit|item|obj|tool)_[a-z0-9]+$", re.I)


def _xy(ent: Any) -> tuple | None:
    for attr in ("xy", "pos", "position", "center", "centre"):
        v = getattr(ent, attr, None)
        if v is not None and len(v) >= 2:
            return (float(v[0]), float(v[1]))
    if isinstance(ent, dict):
        for attr in ("xy", "pos", "position"):
            v = ent.get(attr)
            if v is not None and len(v) >= 2:
                return (float(v[0]), float(v[1]))
    return None


def _answers_visible(obs: dict, task_id: str) -> dict:
    """How many of the row's ANSWER containers are in a frame the assistant was shown.

    ⭐ The frame count that decides whether a row can be interpreted at all. Uses
    the engine's own measurement (``obs["vischeck"]``), not a distance proxy: an entity counts as
    visible when a sample of its AABB lands inside the frustum of a frame that was actually shown,
    within the visibility horizon, with no wall between.
    """
    import task_world_builder

    want = {
        str(g.get("target"))
        for g in (task_world_builder.load_spec(task_id).get("answer_key") or [])
        if isinstance(g, dict)
    }
    vis = obs.get("vischeck") or {}
    if not vis or vis.get("error"):
        return {"decidable": False, "detail": (vis or {}).get("error", "no vischeck")}
    seen = set(vis.get("visible") or [])
    return {
        "decidable": True,
        "n_answers": len(want),
        "n_visible": len(want & seen),
        "visible": sorted(want & seen),
        "missing": sorted(want - seen),
        "truncated": bool(vis.get("truncated")),
    }


def _answer_keys_visible(obs: dict, answers: set[str]) -> dict:
    vis = obs.get("vischeck") or {}
    if not vis or vis.get("error"):
        return {"decidable": False, "detail": vis.get("error", "no vischeck")}
    seen = set(vis.get("visible") or [])
    return {
        "decidable": True,
        "n_answers": len(answers),
        "n_visible": len(answers & seen),
        "visible": sorted(answers & seen),
        "missing": sorted(answers - seen),
        "truncated": bool(vis.get("truncated")),
    }


_PROLOGUE_VERBS: frozenset[str] = frozenset(
    {"walk", "grab", "lookaround", "lookscan", "lookat", "stand", "sit"}
)


def _assert_prologue_safe(task_id: str, pro: dict) -> None:
    """A prologue that scores a sub-goal, or that SINGLES OUT the answer, invalidates the row silently.

    ⭐ THE RULE IS ABOUT SELECTION, NOT ABOUT MENTION. A rule that forbids ever naming an answer
    container makes a whole class of rows unanswerable. Consider a row that asks the assistant to
    pick one of FIVE battery bins whose only discriminator is a printed placard. A prologue forbidden
    from ever naming an answer container can only sweep the room from a distance, and at the delivered
    512 x 405 a 0.46 m placard three metres away is a grey smudge -- so the row would be scored on
    information the assistant was never given. Measured, not argued: the same five placards are
    plainly legible once the person walks up and looks.

    **An EXHAUSTIVE survey is answer-independent by construction.** Visiting every container the row
    declares singles out nothing: permute which one is correct and the prologue is byte-identical, so
    it cannot carry one bit about the answer. Visiting only ``unit_c`` IS the answer. So the check
    becomes: an action may name an answer container only when the prologue names EVERY container in
    the row's roster. A partial tour that happens to include the answer is still refused, and that is
    the case a "don't name the answer" rule is really aimed at.

    ⚠️ This is deliberately stricter than "don't name the answer" in one direction and looser in the
    other. It still refuses: a scored verb, a partial tour touching an answer, an answer key in the
    seeded speech, and a quoted placard in the seeded speech.
    """
    import task_world_builder

    spec = task_world_builder.load_spec(task_id)
    banned = {
        str(g.get("target"))
        for g in (spec.get("answer_key") or [])
        if isinstance(g, dict)
    }
    roster = {str(c) for c in ((spec.get("roster") or {}).get("containers") or [])}
    acts = list(pro.get("actions") or [])
    named = {k for k in roster if any(f"<{k}>" in a for a in acts)}
    exhaustive = bool(roster) and named >= roster
    for act in acts:
        verb = (act.split("]")[0].lstrip("[") or "").strip().lower()
        if verb not in _PROLOGUE_VERBS:
            raise EvalConfigError(
                f"{task_id}: prologue verb {verb!r} is not one of {sorted(_PROLOGUE_VERBS)} -- a "
                f"prologue must never perform a scored action"
            )
        for key in banned:
            if key and f"<{key}>" in act and not exhaustive:
                raise EvalConfigError(
                    f"{task_id}: prologue action {act!r} names the answer container {key!r} in a "
                    f"PARTIAL tour (names {sorted(named)} of {sorted(roster)}). Either drop it or "
                    f"visit every container -- a full survey is answer-independent, a partial one "
                    f"that includes the answer is a leak."
                )
    say = pro.get("say") or ""
    for key in banned:
        if key and key in say:
            raise EvalConfigError(
                f"{task_id}: seeded speech names the answer container {key!r}"
            )
    if re.search(r"[\"“][^\"”]+[\"”]", say):
        raise EvalConfigError(f"{task_id}: seeded speech quotes a placard: {say!r}")


def _authored_goal(task_id: str) -> str:
    """An AUTHORED assistant goal for this row, or "" to fall back to the premise.

    ⛔ WHY. ``design.premise`` is a stage direction written for a human reading the task-design
    document, and a model handed one RECITES it instead of directing the person -- e.g. an
    assistant's whole first turn on a chemistry row can be *"Clear a bottle off the bench."*,
    which is the premise verbatim. Almost no premises open with an imperative verb.

    A row with no entry keeps the premise untouched, so this can only affect rows that were
    deliberately authored. Absent or malformed file -> "" -> the premise is used unchanged.
    """
    return _authored_row(task_id).get("goal", "")


def spec_goal_text(task_id: str) -> tuple[str, str]:
    """Return the assistant goal sentence and human activity for a TaskPack row.

    ⚠️ PROTOCOL DEVIATION, forced by the specs and declared here rather than buried.
    ``prompts.assistant_goal`` derives the assistant's sentence from the SCORING CHECKLIST -- "put all
    the nitric acids into the oxidizers" -- which is correct for RL, where the checklist is the
    person's own request. On these rows it is the answer: every spec marks ``goals`` and
    ``answer_key`` HIDDEN and lists the container keys in ``redact_in_visible_text``, because working
    out which container an item belongs in from its label plus a posted rule table IS the thing being
    measured. Handing that over would make every task free.

    So the sentence is taken from the one natural-language field the spec marks VISIBLE --
    ``design.premise`` -- and the checklist is withheld. The STRUCTURE the protocol calls for is
    unchanged: the assistant still gets a specific statement of the task, the human still gets only the
    vague activity (the scene). Redacted keys are stripped from both.

    Falls back to the humanised scene key when a spec has no premise, never to the checklist.
    """
    import task_world_builder

    spec = task_world_builder.load_spec(task_id)
    scene = str(spec.get("scene") or "")
    activity = scene.replace("_", " ").strip() or "a task at home"
    premise = str(((spec.get("design") or {}).get("premise") or "")).strip()
    goal = _authored_goal(task_id) or premise or activity
    for key in spec.get("redact_in_visible_text") or []:
        goal = goal.replace(str(key), "one of the labelled units")
    return goal, activity


_TICK_BASE: int = 8
_TICK_PER_ITEM: int = 6


def tick_budget(task_id: str, override: int = 0) -> int:
    """Dialogue ticks this row gets: ``8 + 6 x (things that have to be moved)``.

    ⛔ WHY NOT A CONSTANT. A flat budget sized for a ONE-item row is too small for a three-item one,
    where even the SCRIPTED-PERFECT oracle runs out before it can declare. Then "the model did not
    finish the task" and "nobody was given room to finish" score the same. A budget that does not
    grow with the work ends up measuring the budget.

    Counts DISTINCT OBJECTS, not goals -- the union of every ``goals`` subject with every object an
    ``all_placed`` negative names, which is exactly the set the person must carry somewhere, one trip
    each because they have one hand. An explicit ``--max-ticks`` still wins so a probe can pin it, and
    an unreadable spec falls back to the base rather than to zero ticks.
    """
    if override:
        return int(override)
    try:
        import task_world_builder

        spec = task_world_builder.load_spec(task_id)
    except Exception:  # noqa: BLE001 - the budget must never be what fails an episode
        return _TICK_BASE
    objs = {
        str(g.get("subject")) for g in (spec.get("goals") or []) if g.get("subject")
    }
    for neg in spec.get("negs") or []:
        if str(neg.get("kind")) != "all_placed":
            continue
        lit = (((neg.get("objs_expr") or {}).get("of") or {}).get("lit")) or []
        objs.update(str(o) for o in lit)
        objs.update(str(o) for o in (neg.get("objs") or []))
    return _TICK_BASE + _TICK_PER_ITEM * max(1, len(objs))


def assistant_obs(obs: dict, goal_text: str) -> dict:
    """A view of the observation for the ASSISTANT turn only.

    ``_assistant_messages`` composes its system prompt as ``assistant_system(assistant_goal(task,
    goal))``; with an EMPTY checklist ``assistant_goal`` returns the activity verbatim, so overriding
    these two keys is enough to install ``goal_text`` without mutating shared protocol state. Frames,
    speech and everything else are the protocol's own.
    """
    return {**obs, "task": goal_text, "goal": []}


def human_obs(obs: dict, activity: str) -> dict:
    """A view for the HUMAN turn: the vague activity in place of the raw ``spec:<id>`` scene key.

    Only the title changes. The OBJECTS list, ``held``, ``put_verb`` and the visibility flag are the
    env's own values, untouched.
    """
    return {**obs, "task": activity}


# ---------------------------------------------------------------------------------------------------
#  the protocol object
# ---------------------------------------------------------------------------------------------------
def build_protocol(args: argparse.Namespace) -> IsaacAgentLoop:
    """Build an ``IsaacAgentLoop`` with only release dialogue state.

    The protocol class contains reusable dialogue behavior without a constructor, so this factory
    initializes its release runtime state explicitly. Nothing here overrides a prompt.
    """
    proto = object.__new__(IsaacAgentLoop)
    proto.max_ticks = int(
        args.max_ticks or _TICK_BASE
    )  # main() re-sets this PER ROW (tick_budget)
    proto.human_server = args.human_url
    proto.human_timeout_s = float(getattr(args, "human_timeout_s", 120.0))
    proto.human_http_attempt_limit = int(getattr(args, "human_http_attempt_limit", 3))
    proto.human_http_retry_backoff_s = float(
        getattr(args, "human_http_retry_backoff_s", 0.25)
    )
    proto.human_invalid_attempt_limit = int(
        getattr(
            args, "human_invalid_attempt_limit", _DEFAULT_HUMAN_INVALID_ATTEMPT_LIMIT
        )
    )
    proto._interaction_guide_human_turn_guard = (
        str(getattr(args, "guide_human_turn_guard", "on")) == "on"
    )
    proto._interaction_watch_human_turn_guard = (
        str(getattr(args, "watch_human_turn_guard", "on")) == "on"
    )
    proto._interaction_orchestrate_human_turn_guard = (
        str(getattr(args, "orchestrate_human_turn_guard", "on")) == "on"
    )
    proto.frame_max_side = int(args.frame_max_side)
    proto.per_interval = int(args.per_interval)
    proto.max_frames = int(args.max_frames)
    proto.max_turn_tokens = int(args.assistant_tokens)
    proto.scene = ""
    proto.response_length = 1 << 30  # no token budget off the trainer
    proto._timing_reset()
    proto._transcript = []
    proto._last_cmd_error = ""
    proto._last_successful_human_command = ""
    proto._last_assistant_raw = ""
    proto._last_assistant_mode = ""
    proto._last_human_say = ""
    proto._pending_human_say = ""
    proto._last_human_mode = "normal"
    proto._last_human_reasoning = ""
    proto._pending_cmds = []
    proto._carry_say = []
    proto._error_turn = None
    proto._error_mode = "normal"
    return proto


def reset_protocol(proto: IsaacAgentLoop) -> None:
    """Per-episode state, mirroring the block at the top of ``_run_episode``."""
    proto._timing_reset()
    proto._transcript = []
    proto._last_cmd_error = ""
    proto._last_successful_human_command = ""
    proto._last_assistant_raw = ""
    proto._last_assistant_mode = ""
    proto._last_human_say = ""
    proto._pending_human_say = ""
    proto._last_human_mode = "normal"
    proto._last_human_reasoning = ""
    proto._pending_cmds = []
    proto._carry_say = []
    proto._error_turn = None
    proto._error_mode = "normal"


# ---------------------------------------------------------------------------------------------------
#  startup gates -- the two failures that look like a healthy run
# ---------------------------------------------------------------------------------------------------
def assert_frame_budget() -> int:
    """``ISAAC_FRAME_BUDGET`` must be positive, and the run's own log must show the value used."""
    budget = int(os.getenv("ISAAC_FRAME_BUDGET", "0") or 0)
    if budget <= 0:
        raise EvalConfigError(
            "ISAAC_FRAME_BUDGET is 0 (or unset). With no schedule the 32B is handed every frame of the "
            "episode (~96 images, ~69k tokens) and starts returning HTTP 400 around tick 10, which is "
            "swallowed into an idled tick -- the episode runs to full length and does nothing. "
            "Set ISAAC_FRAME_BUDGET (24 is the recommended default) and re-run."
        )
    return budget


# The clause that turns a silent no-op walk into a fact the person can act on. Imported, not restated:
# this file composes no sentence of its own, and the trainer must say the identical thing.
#
# ⚠️ A ``[Walk] <x>`` issued from where you already stand moves nothing, renders ONE frame, and is
# reported as a success. The person is told it worked, is shown a frame of nothing, and cannot tell
# ARRIVED from STILL WALKING -- so at temperature 0 it re-issues the identical command until the tick
# budget is gone. This has been observed while standing at the CORRECT answer container, so it reads
# as a binding failure when binding had in fact succeeded. It penalises exactly the runs that are
# working.
#
# Worded as ARRIVAL rather than as an error, because that is what it is: the person is where it asked to
# be, and the next move is to act, not to walk again.
arrival_text = _arrival_clause


def outcome_text(
    cmd: str, bad: list, sc: dict, progress_line: str, no_motion: dict | None = None
) -> str:
    """What the person is told about the action their own hands just took.

    ⛔ EVAL PATH ONLY. ``progress_line="off"`` drops the ``Progress: n/N done.`` clause. That clause
    reports how much of the HIDDEN scoring checklist is complete -- a channel that does not exist in the
    world we claim to be simulating (nobody doing a chore has a live counter against a rubric someone
    else is holding), and one that plausibly decides when the person believes they are finished. The
    ``RESULT of your last action`` half stays either way: that is the person feeling what happened to
    their own hands, which is legitimate and which a real person has.

    With the default ``"on"`` this function includes the progress clause; ``"off"`` preserves only
    feedback that the person could directly observe.
    """
    show = str(progress_line or "on") != "off"
    prog = (
        f" Progress: {sc.get('n_done')}/{sc.get('n_total')} done."
        if (show and sc.get("n_total"))
        else ""
    )
    arrived = arrival_text(no_motion)
    if bad:
        return (
            f"Your last command {cmd!r} did NOT fully succeed "
            f"({', '.join(sorted(set(bad)))}).{arrived}{prog}"
        )
    if prog or arrived:
        return f"Your last command {cmd!r} succeeded.{arrived}{prog}"
    # No checklist clause to carry. With the oracle suppressed the person must still be told the plain
    # outcome, or turning the progress line off would also delete the honest half of the feedback.
    return f"Your last command {cmd!r} succeeded." if not show else ""


def _effective_progress_line(args: argparse.Namespace, interaction: Any | None) -> str:
    """TaskPack Humans never receive hidden rubric progress; legacy runs keep their flag."""
    if interaction is not None:
        return "off"
    return str(getattr(args, "progress_line", "on"))


def _last_command_failed(feedback: str) -> bool:
    return "did NOT fully succeed" in feedback or "was REJECTED" in feedback


class _WatchPreAssistantRecoveryBudget:
    """Track bounded cross-turn WATCH recovery for invalid-but-auditable episodes."""

    POLICY_VERSION = 2

    def __init__(
        self,
        semantic_limit: int,
        allowance: int,
        enabled: bool,
        observation_context_supported: bool = True,
    ) -> None:
        if allowance < 0 or allowance > _DEFAULT_WATCH_PREASSISTANT_RECOVERY_ALLOWANCE:
            raise ValueError(
                "WATCH pre-Assistant recovery allowance must be between 0 and "
                f"{_DEFAULT_WATCH_PREASSISTANT_RECOVERY_ALLOWANCE}, got {allowance}"
            )
        self.semantic_limit = int(semantic_limit)
        self.allowance = int(allowance) if enabled else 0
        self.enabled = bool(enabled)
        self.observation_context_supported = bool(observation_context_supported)
        self.raw_ticks = 0
        self.semantic_ticks = 0
        self.raw_actions = 0
        self.semantic_actions = 0
        self.credited_ticks = 0
        self.pending_ticks = 0
        self.chains_started = 0
        self.chains_recovered = 0
        self.unrecovered_chains = 0
        self.allowance_exceeded = 0
        self.interrupted_by_assistant = 0
        self.human_deviations = 0
        self.human_speech = 0
        self.stateful_detours = 0
        self.observation_context_unsupported = 0
        self.context_restores = 0
        self.assistant_has_spoken = False
        self.active_source = ""
        self.active_expected = ""

    def can_start_turn(self) -> bool:
        if self.active_expected:
            return self.credited_ticks + self.pending_ticks < self.allowance
        return self.semantic_ticks < self.semantic_limit

    def begin_turn(self) -> None:
        self.raw_ticks += 1
        self.semantic_ticks += 1

    def note_assistant(self, *, mode: str, say: str, done: bool) -> None:
        if mode == "silence" and not say.strip() and not done:
            return
        self.assistant_has_spoken = True
        if self.active_expected:
            self.interrupted_by_assistant += 1
            self._invalidate_active()

    def note_no_action(self) -> str:
        if not self.active_expected:
            return "semantic"
        self.human_deviations += 1
        self._invalidate_active()
        return "invalid_human_deviation"

    def note_action(
        self,
        *,
        source: str,
        expected: str,
        expected_failure: bool,
        expected_success: bool,
        motor_recovery: bool,
        human_say: str = "",
    ) -> str:
        self.raw_actions += 1
        self.semantic_actions += 1
        eligible_source = source in {"watch_opening", "watch_default"}
        if not self.active_expected:
            if (
                self.enabled
                and not self.assistant_has_spoken
                and eligible_source
                and expected
                and expected_failure
            ):
                if not self._start(source, expected):
                    return "invalid_allowance_exhausted"
                if human_say.strip():
                    self.human_speech += 1
                    self._invalidate_active()
                    return "invalid_human_speech"
                if not self.observation_context_supported:
                    self.observation_context_unsupported += 1
                    self._invalidate_active()
                    return "invalid_observation_context"
                return "recovery_chain_start"
            return "semantic"
        if (
            self.assistant_has_spoken
            or source != self.active_source
            or expected != self.active_expected
        ):
            self.human_deviations += 1
            self._invalidate_active()
            return "invalid_human_deviation"
        if human_say.strip():
            self.human_speech += 1
            self._invalidate_active()
            return "invalid_human_speech"
        if expected_success:
            self._consume_recovery_turn()
            self.credited_ticks += self.pending_ticks
            self.pending_ticks = 0
            self.chains_recovered += 1
            self._clear_active()
            return "recovery_chain_closed"
        if expected_failure:
            self._consume_recovery_turn()
            return self._continue_or_exhaust()
        if motor_recovery:
            self.stateful_detours += 1
            self._invalidate_active()
            return "invalid_stateful_detour"
        self.human_deviations += 1
        self._invalidate_active()
        return "invalid_human_deviation"

    def finish_episode(self) -> None:
        if not self.active_expected:
            return
        if self.credited_ticks + self.pending_ticks >= self.allowance:
            self.allowance_exceeded += 1
        self._invalidate_active()

    def publish(self, counters: dict) -> None:
        counters.update(
            {
                "n_raw_ticks": self.raw_ticks,
                "n_semantic_ticks": self.semantic_ticks,
                "n_raw_assistant_calls": self.raw_ticks,
                "n_semantic_assistant_calls": self.semantic_ticks,
                "n_raw_human_actions": self.raw_actions,
                "n_semantic_human_actions": self.semantic_actions,
                "watch_preassistant_recovery_policy_version": self.POLICY_VERSION,
                "watch_preassistant_recovery_allowance": self.allowance,
                "watch_preassistant_recovery_chains_started": self.chains_started,
                "watch_preassistant_recovery_chains_recovered": self.chains_recovered,
                "watch_preassistant_recovery_credited_ticks": self.credited_ticks,
                "watch_preassistant_recovery_unrecovered_chains": self.unrecovered_chains,
                "watch_preassistant_recovery_allowance_exceeded": self.allowance_exceeded,
                "watch_preassistant_recovery_interrupted_by_assistant": (
                    self.interrupted_by_assistant
                ),
                "watch_preassistant_recovery_human_deviations": self.human_deviations,
                "watch_preassistant_recovery_human_speech": self.human_speech,
                "watch_preassistant_recovery_stateful_detours": self.stateful_detours,
                "watch_preassistant_recovery_observation_context_unsupported": (
                    self.observation_context_unsupported
                ),
                "watch_preassistant_recovery_context_restores": self.context_restores,
            }
        )

    def note_context_restored(self) -> None:
        self.context_restores += 1

    def _start(self, source: str, expected: str) -> bool:
        if self.credited_ticks >= self.allowance:
            self.allowance_exceeded += 1
            self.unrecovered_chains += 1
            return False
        self.chains_started += 1
        self.active_source = source
        self.active_expected = expected
        return True

    def _consume_recovery_turn(self) -> None:
        self.semantic_ticks -= 1
        self.semantic_actions -= 1
        self.pending_ticks += 1

    def _continue_or_exhaust(self) -> str:
        if self.credited_ticks + self.pending_ticks < self.allowance:
            return "recovery_turn"
        self.allowance_exceeded += 1
        self._invalidate_active()
        return "invalid_allowance_exhausted"

    def _invalidate_active(self) -> None:
        self.semantic_ticks += self.pending_ticks
        self.semantic_actions += self.pending_ticks
        self.pending_ticks = 0
        self.unrecovered_chains += 1
        self._clear_active()

    def _clear_active(self) -> None:
        self.active_source = ""
        self.active_expected = ""


def _snapshot_watch_recovery_observation_context(
    backend: Any,
    assistant_frame_batches: Sequence[list],
    human_frame_batches: Sequence[list],
    ledger: Any,
    last_human_say: str,
) -> dict[str, object]:
    return {
        "backend": backend.snapshot_observational_state(),
        "assistant_frame_batches": list(assistant_frame_batches),
        "human_frame_batches": list(human_frame_batches),
        "ledger_cum": list(ledger.cum),
        "last_human_say": last_human_say,
    }


def _restore_watch_recovery_observation_context(
    backend: Any,
    assistant_frame_batches: list[list],
    human_frame_batches: list[list],
    ledger: Any,
    snapshot: Mapping[str, object],
) -> str:
    backend_state = snapshot.get("backend")
    if not isinstance(backend_state, Mapping):
        raise EvalConfigError("WATCH recovery snapshot has no backend state")
    backend.restore_observational_state(backend_state)
    assistant_frame_batches[:] = list(snapshot.get("assistant_frame_batches") or ())
    human_frame_batches[:] = list(snapshot.get("human_frame_batches") or ())
    ledger.cum[:] = list(snapshot.get("ledger_cum") or ())
    return str(snapshot.get("last_human_say") or "")


def _model_human_action_retry_enabled(
    proto: Any,
    interaction: Any | None,
    interaction_turn: Any | None,
    assistant: Mapping[str, object],
    command: str,
) -> bool:
    """Model Humans may retry safe ASK/ORCHESTRATE transient simulator misses."""
    _ = interaction_turn, assistant, command
    if (
        interaction is None
        or getattr(proto, "_last_human_scripted", False)
        or getattr(proto, "_interaction_human_policy", "") != "model"
    ):
        return False
    return str(getattr(interaction.spec, "family", "")).upper() in {
        "ASK",
        "ORCHESTRATE",
    }


def _watch_action_failure_audit_enabled(interaction: Any | None) -> bool:
    return bool(
        interaction is not None
        and str(getattr(interaction.spec, "family", "")).upper() == "WATCH"
    )


def _watch_recovery_counter_snapshot(interaction: Any | None) -> dict[str, int]:
    if interaction is None:
        return {}
    names = (
        "watch_opening_matches",
        "watch_opening_execution_failures",
        "watch_opening_mismatches",
        "watch_default_matches",
        "watch_default_execution_failures",
        "watch_default_mismatches",
        "watch_motor_recovery_actions",
    )
    counters = getattr(interaction, "counters", {})
    return {name: int(counters.get(name, 0)) for name in names}


def _watch_recovery_step_flags(
    interaction: Any | None,
    before: Mapping[str, int],
    source: str,
) -> dict[str, bool]:
    after = _watch_recovery_counter_snapshot(interaction)

    def increased(name: str) -> bool:
        return int(after.get(name, 0)) > int(before.get(name, 0))

    prefix = "watch_opening" if source == "watch_opening" else "watch_default"
    motor_recovery = increased("watch_motor_recovery_actions")
    return {
        "expected_failure": increased(f"{prefix}_execution_failures"),
        "expected_success": increased(f"{prefix}_matches"),
        "motor_recovery": motor_recovery,
    }


# ===================================================================================================
#  ⭐ DECLARE-DONE. The assistant says when it has finished; the system never stops on goal-met.
# ===================================================================================================
# Without this a blind assistant can score 1.00 in two commands by guessing a bin: the person puts the
# item there, it guesses a second bin -- and a loop watching object state HALTS AND SCORES the instant
# the second guess happens to be right. Such a system does not merely fail to prevent a
# guess-until-it-sticks strategy; it chooses the guesser's luckiest moment to score on its behalf,
# even on a row whose declared chance level is 0.20.
#
# The fix is not to ban retries -- an assistant that notices its own mistake, corrects it and THEN
# declares should get full marks, and that is exactly the self-correction the benchmark wants. The
# fix is that stopping is the ASSISTANT'S decision and it must commit: the world is scored when its
# declaration is accepted, and never declaring is its own recorded failure. A TaskPack declaration
# waits only for an already-accepted Human action sequence to finish. Under this rule a blind agent's
# expected score falls to chance.
#
# The declaration clause is appended to the composed system message here rather than mutating the
# shared prompt constant.
_DONE_TOKEN = "$done$"
_DONE_DECLARATION_PHRASES: frozenset[str] = frozenset(
    {"", "all set", "complete", "completed", "done", "finished"}
)
_MALFORMED_SILENT_RE = re.compile(r"^\$silent[.!]?$", re.IGNORECASE)
_MALFORMED_SPEECH_RE = re.compile(
    r"^\$(interrupt|error)(?!\$)(?:\s*:\s*|\s+)(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_SPEECH_TOKEN_RE = re.compile(
    r"^(.+?)\s*\$(interrupt|error)\$\s*$",
    re.IGNORECASE | re.DOTALL,
)
_SPEECH_CONTROL_BOUNDARY_RE = re.compile(
    r"(?:^|(?<=\$))[ \t]*\$(interrupt|error)\$", re.IGNORECASE | re.MULTILINE
)
_MARKDOWN_CONTROL_RE = re.compile(
    r"^(?P<fence>`{1,3})\s*(?P<token>\$silent\$|\$done\$|<done/>)\s*(?P=fence)$",
    re.IGNORECASE,
)
_DECLARE_ADDENDUM = """
Nothing here stops on its own, and the job is not over until you say so.
- When you judge the task complete, reply with EXACTLY: $done$
The world is scored when that declaration is accepted. A declaration made while the person is still
carrying out an already-planned action sequence waits until that sequence finishes, so inspect the
result and declare afterwards. Changing your mind before declaring costs you nothing: if you see them
put something in the wrong place, correct them and declare afterwards. If you never declare, that is
recorded as a failure to finish."""


def _with_declare_clause(msgs: list[dict]) -> list[dict]:
    """Append the declare-done clause to a composed SYSTEM message. No-op if there is not one."""
    out = []
    for m in msgs:
        if (
            m.get("role") == "system"
            and isinstance(m.get("content"), str)
            and _DONE_TOKEN not in m["content"]
        ):
            m = {**m, "content": m["content"] + "\n" + _DECLARE_ADDENDUM}
        out.append(m)
    return out


def _deduplicate_leading_speech_controls(text: str) -> tuple[str, bool]:
    controls = list(_SPEECH_CONTROL_BOUNDARY_RE.finditer(text))
    if len(controls) < 2 or controls[0].start() != 0:
        return text, False
    segments = [
        (
            control.group(1).lower(),
            text[
                control.end() : (
                    controls[index + 1].start()
                    if index + 1 < len(controls)
                    else len(text)
                )
            ].strip(),
        )
        for index, control in enumerate(controls)
    ]
    speeches = (
        [segments[0][1]]
        if all(segment == segments[0] for segment in segments[1:])
        else [speech for _, speech in segments]
    )
    combined = "\n".join(speech for speech in speeches if speech)
    return f"${segments[0][0]}${combined}", True


def parse_assistant_eval(proto: Any, raw: str) -> dict:
    """``proto._parse_assistant`` plus the eval-only ``$done$`` declaration.

    Guarded the same way ``$silent$`` is: the token counts as a declaration only when it is
    essentially the whole reply, so an assistant that MENTIONS finishing inside an instruction does
    not accidentally end its own episode. ``<done/>`` is accepted too -- ``agent_protocol``'s own
    assistant contract already promises that spelling, and honouring both costs nothing. A malformed
    declaration carrying substantive speech remains non-terminal, but its control token is removed
    before the speech reaches the person.
    """
    t = (raw or "").strip()
    markdown_control = _MARKDOWN_CONTROL_RE.fullmatch(t)
    if markdown_control:
        proto._n["malformed_control_token"] = (
            proto._n.get("malformed_control_token", 0) + 1
        )
        t = markdown_control.group("token")
    low = t.lower()
    if _MALFORMED_SILENT_RE.fullmatch(t):
        proto._n["malformed_control_token"] = (
            proto._n.get("malformed_control_token", 0) + 1
        )
        parsed = proto._parse_assistant("$silent$")
        proto._last_assistant_raw = t[:200]
        return parsed
    normalized_control, had_duplicate_control = _deduplicate_leading_speech_controls(t)
    if had_duplicate_control:
        proto._n["malformed_control_token"] = (
            proto._n.get("malformed_control_token", 0) + 1
        )
        parsed = proto._parse_assistant(normalized_control)
        proto._last_assistant_raw = t[:200]
        return parsed
    malformed_speech = _MALFORMED_SPEECH_RE.match(t)
    if malformed_speech:
        proto._n["malformed_control_token"] = (
            proto._n.get("malformed_control_token", 0) + 1
        )
        token, speech = malformed_speech.groups()
        parsed = proto._parse_assistant(f"${token.lower()}${speech.strip()}")
        proto._last_assistant_raw = t[:200]
        return parsed
    trailing_speech_token = _TRAILING_SPEECH_TOKEN_RE.match(t)
    if trailing_speech_token:
        proto._n["malformed_control_token"] = (
            proto._n.get("malformed_control_token", 0) + 1
        )
        speech, token = trailing_speech_token.groups()
        parsed = proto._parse_assistant(f"${token.lower()}${speech.strip()}")
        proto._last_assistant_raw = t[:200]
        return parsed
    if "$silent$" in low and low.replace("$silent$", "").strip():
        proto._n["malformed_control_token"] = (
            proto._n.get("malformed_control_token", 0) + 1
        )
        speech = re.sub(r"\$silent\$", "", t, flags=re.IGNORECASE).strip()
        parsed = proto._parse_assistant(f"$interrupt${speech}")
        proto._last_assistant_raw = t[:200]
        return parsed
    sanitized = t
    sanitized_done_token = False
    for tok in (_DONE_TOKEN, "<done/>"):
        k = low.find(tok)
        if k < 0:
            continue
        rest = (low[:k] + low[k + len(tok) :]).strip()
        declaration = re.sub(r"[^a-z]+", " ", rest).strip()
        if declaration in _DONE_DECLARATION_PHRASES:
            proto._n["mode_done"] = proto._n.get("mode_done", 0) + 1
            proto._last_assistant_raw = t[:200]
            return {"mode": "done", "say": "", "done": True}
        sanitized = sanitized[:k] + ("done" if k else "") + sanitized[k + len(tok) :]
        low = sanitized.lower()
        sanitized_done_token = True
    if sanitized_done_token:
        proto._n["done_token_sanitized"] = proto._n.get("done_token_sanitized", 0) + 1
    return proto._parse_assistant(sanitized.strip())


def _should_defer_assistant_done(
    assistant: Mapping[str, object],
    interaction: object | None,
    pending_commands: Sequence[str],
) -> bool:
    """Keep an accepted TaskPack action sequence atomic across per-action polling."""
    return bool(assistant.get("done") and interaction is not None and pending_commands)


def _assert_interaction_termination(proto: Any, interaction: Any) -> None:
    if interaction is None:
        return
    stop_reason = str(getattr(proto, "_stop_reason", ""))
    if stop_reason not in {"assistant_done", "max_ticks"}:
        raise EvalConfigError(
            "TaskPack interaction ended through an unsupported path: "
            f"{stop_reason!r}; expected Assistant $done$ or the action limit"
        )


def _find_model_guard_enabled(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "find_human_control", "dynamic-guarded")) == "dynamic-guarded"
    )


def _ask_instruction_guard_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "ask_human_control", "strict")) == "strict"


def _prompt_for_audit(messages: list[dict]) -> list[dict[str, object]]:
    """JSON-safe prompt text with image positions preserved but pixels kept in IO capture."""
    audited: list[dict[str, object]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            audited.append({"role": message.get("role"), "content": content})
            continue
        image_index = 0
        parts: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"type": "unknown", "value": str(part)})
                continue
            if part.get("type") == "image":
                parts.append({"type": "image", "request_index": image_index})
                image_index += 1
            else:
                parts.append(dict(part))
        audited.append({"role": message.get("role"), "content": parts})
    return audited


def _obs_frame_segs(obs: dict) -> list:
    """The per-frame motion-segment tags parallel to ``obs["frames"]``, or ``[]``.

    ``isaac_env.observe`` emits the key only when it is exactly parallel to the pixels, so an empty
    list here means "not measured" and the frame buffer falls back to its historical selection --
    it never means "no frame held a cue".
    """
    return list(obs.get("frame_segs") or [])


def _assistant_frame_batch(
    frame_buffer: Any,
    tick: int,
    fresh_frames: bool,
    interaction: Any | None,
    frame_cap: int = 0,
    recent_intervals: int = 0,
) -> tuple[list, int]:
    """Select this assistant turn's images and report omitted setup-history frames."""
    if tick != 0:
        return (frame_buffer.delta() if fresh_frames else []), 0
    if recent_intervals > 0:
        opening_frames = frame_buffer.fed_frames()
        selected = frame_buffer.recent_frames(recent_intervals)
        if frame_cap > 0:
            selected = selected[-frame_cap:]
        return selected, max(0, len(opening_frames) - len(selected))
    if interaction is not None and hasattr(interaction, "assistant_opening_frames"):
        return interaction.assistant_opening_frames(frame_buffer, frame_cap)
    opening_frames = frame_buffer.fed_frames()
    if interaction is None or interaction.assistant_receives_opening_history:
        return opening_frames, 0
    current_frames = opening_frames[-1:]
    return current_frames, max(0, len(opening_frames) - len(current_frames))


def select_human_frames(
    frame_buffer: Any,
    frame_batches: list[list],
    policy: str,
    frame_cap: int,
) -> list:
    """Select the stateless Human model's images under the requested memory policy."""
    if policy == "recent":
        return frame_buffer.recent_frames(_HUMAN_VISUAL_MEMORY_INTERVALS)
    if policy == "assistant-history":
        selected = select_temporal_image_batches(
            frame_batches,
            frame_cap,
            recent_batch_count=_HUMAN_VISUAL_MEMORY_INTERVALS,
        )
        return [frame for batch in selected for frame in batch]
    raise EvalConfigError(f"unsupported human frame policy: {policy!r}")


def _assert_initial_interaction_frame_cap(
    args: argparse.Namespace, frames: list, tick: int, interaction: Any | None
) -> None:
    if tick != 0 or interaction is None:
        return
    frame_cap = int(getattr(args, "assistant_frame_cap", 0) or 0)
    if frame_cap > 0 and len(frames) > frame_cap:
        raise EvalConfigError(
            f"TaskPack opening selected {len(frames)} assistant frames, exceeding "
            f"--assistant-frame-cap={frame_cap}; refusing an evaluation that would "
            "silently discard authored opening evidence"
        )


def probe_human(url: str, timeout_s: float = 120.0) -> str:
    """Request a real completion from the configured Human HTTP JSON endpoint."""
    import urllib.request

    req = {
        "system": "Reply with the single word OK.",
        "messages": [{"role": "user", "text": "Say OK."}],
        "max_new_tokens": 8,
        "temperature": 0.0,
        "do_sample": False,
    }
    data = json.dumps(req).encode("utf-8")
    r = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=timeout_s) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if not out.get("ok"):
        raise EvalConfigError(
            f"human endpoint answered NOT-OK: {str(out.get('error'))[:200]}"
        )
    text = str(out.get("text") or "")
    if not text.strip():
        raise EvalConfigError(
            "human endpoint returned an EMPTY completion -- the model is not serving"
        )
    return text


def _ask_human_reprompt_enabled(
    observation: Mapping[str, object],
    guard: str,
    *,
    guide_human_turn_guard: bool = True,
) -> bool:
    family = str(observation.get("interaction_family") or "").strip().upper()
    if family == "GUIDE" and not guide_human_turn_guard:
        return False
    if (
        family == "ASK"
        and str(observation.get("_interaction_ask_human_control") or "strict")
        == "prompt-only"
    ):
        return False
    return human_reprompt_guard_enabled(
        family,
        guard,
        human_object_interface=str(
            observation.get("interaction_human_object_interface") or "full_ids"
        ),
    )


def _mark_human_retry_attempt(row: dict[str, object], guard: str) -> None:
    row["human_retry_only"] = True
    row["human_retry_guard"] = guard
    row["human_policy_guard"] = ""
    row["human_say"] = ""
    row["human_say_deferred"] = ""


def _record_exhausted_human_guard(proto: Any, guard: str) -> None:
    key = f"human_policy_guard_{guard}"
    proto._n[key] = proto._n.get(key, 0) + 1
    proto._n["human_invalid_attempt_limit_exhausted"] = (
        proto._n.get("human_invalid_attempt_limit_exhausted", 0) + 1
    )


def _model_interaction_human_command(
    proto: Any,
    human_observation: dict,
    assistant_say: str,
    human_frames: list,
) -> tuple[str | None, dict[str, object] | None]:
    invalid_rows: list[dict[str, object]] = []
    attempt_limit = max(
        1,
        int(
            getattr(
                proto,
                "human_invalid_attempt_limit",
                _DEFAULT_HUMAN_INVALID_ATTEMPT_LIMIT,
            )
        ),
    )
    for attempt in range(attempt_limit):
        prompt_observation = dict(human_observation)
        if invalid_rows:
            guard = str(invalid_rows[-1].get("human_retry_guard") or "")
            prompt_observation["_interaction_human_retry_feedback"] = (
                _ASK_HUMAN_REPROMPT_TEXT[guard]
            )
        started = time.perf_counter()
        transcript_start = len(proto._transcript)
        proto._defer_ask_human_guard_accounting = True
        try:
            command = proto._human_command(
                prompt_observation, assistant_say, human_frames
            )
        finally:
            proto._defer_ask_human_guard_accounting = False
        proto._n["human_calls"] += 1
        proto._tk["t_human_s"].append(round(proto._tadd("t_human_s", started), 3))
        row = (
            proto._transcript[-1] if len(proto._transcript) > transcript_start else None
        )
        if row is not None:
            row.update(
                {
                    "human_model_call": True,
                    "human_received_frames": len(human_frames),
                }
            )
        guard = str((row or {}).get("human_policy_guard") or "")
        if not _ask_human_reprompt_enabled(
            human_observation,
            guard,
            guide_human_turn_guard=bool(
                getattr(proto, "_interaction_guide_human_turn_guard", True)
            ),
        ):
            if row is not None and invalid_rows:
                row["human_invalid_attempt_count"] = len(invalid_rows)
            if invalid_rows:
                proto._n["human_invalid_attempts_recovered"] = proto._n.get(
                    "human_invalid_attempts_recovered", 0
                ) + len(invalid_rows)
            return command, row
        if row is not None:
            _mark_human_retry_attempt(row, guard)
            invalid_rows.append(row)
        proto._pending_cmds = []
        proto._pending_human_say = ""
        proto._last_human_say = ""
        if attempt + 1 < attempt_limit:
            continue
        _record_exhausted_human_guard(proto, guard)
        if row is not None:
            row["human_retry_only"] = False
            row["human_policy_guard"] = guard
            row["human_invalid_attempt_count"] = len(invalid_rows)
            row["human_invalid_attempt_limit_exhausted"] = True
        raise HumanInvalidAttemptLimitError(guard, row)
    raise AssertionError("positive Human attempt limit exhausted without returning")


def _annotate_interaction_human_row(
    row: dict[str, object] | None,
    turn: Any,
    human_frame_count: int,
) -> None:
    if row is None:
        return
    row.update(
        {
            "interaction_source": turn.source,
            "interaction_active_instruction": turn.active_instruction,
            "interaction_private_plan": turn.private_plan,
            "interaction_planned_speech": turn.planned_speech,
            "human_model_call": True,
            "human_received_frames": human_frame_count,
        }
    )


def _record_interaction_human_inputs(
    cap: Any,
    ledger: Any,
    rows: Sequence[object],
    human_frames: list,
    tick: int,
) -> None:
    for value in rows:
        if not isinstance(value, dict) or value.get("human_model_call") is not True:
            continue
        frame_count = value.get("human_received_frames")
        frames_sent = (
            human_frames[:frame_count]
            if isinstance(frame_count, int) and not isinstance(frame_count, bool)
            else []
        )
        frame_records = cap.record_human_input(
            frames_sent,
            turn=tick,
            provenance=ledger.trace(frames_sent),
            system_prompt=str(value.get("human_system_prompt") or ""),
            user_prompt=str(value.get("human_user_prompt") or ""),
            raw_response=str(value.get("human_raw") or ""),
            parsed_action=str(value.get("human_action") or ""),
            spoken=str(value.get("human_say") or ""),
        )
        value["human_frame_refs"] = [record.get("file") for record in frame_records]
        value["human_frame_provenance"] = [
            record.get("provenance") for record in frame_records
        ]


def _interaction_human_command(
    proto: Any,
    interaction: Any,
    observation: dict,
    assistant: Mapping[str, object],
    frames: list,
) -> tuple[str | None, bool, Any]:
    """Get one free-human action while preserving per-action assistant polling."""
    turn = interaction.prepare_human_turn(
        str(assistant.get("say") or ""), proto._pending_cmds
    )
    if turn.preempt_queue:
        proto._n["queue_interrupted"] += 1
        proto._n["queue_preempted_actions"] = proto._n.get(
            "queue_preempted_actions", 0
        ) + len(proto._pending_cmds)
        proto._pending_cmds = []
        proto._pending_human_say = ""
        proto._carry_say = []
    if (
        getattr(proto, "_interaction_human_policy", "model") == "scripted"
        and turn.source != "watch_speech"
    ):
        return _scripted_interaction_human_command(
            proto, interaction, observation, assistant, turn
        )
    if turn.source == "queued":
        queue_origin = proto._transcript[-1] if proto._transcript else {}
        held = sorted(observation.get("held", []))
        command_before_repair = proto._pending_cmds.pop(0)
        command = proto._repair_one_arg_put(command_before_repair, held)
        command_repair = IsaacAgentLoop._one_arg_put_repair_audit(
            command_before_repair, command, held
        )
        proto._n["queue_played"] += 1
        proto._last_human_say = ""
        deferred_say = proto._pending_human_say if not proto._pending_cmds else ""
        human_observation = interaction.human_observation(
            {
                **observation,
                "_interaction_find_human_control": getattr(
                    proto, "_find_human_control", "dynamic-guarded"
                ),
                "_interaction_ask_human_control": getattr(
                    proto, "_ask_human_control", "strict"
                ),
                "_interaction_orchestrate_human_turn_guard": bool(
                    getattr(proto, "_interaction_orchestrate_human_turn_guard", True)
                ),
            },
            turn,
        )
        guarded_commands, human_policy_guard = proto._apply_interaction_human_policy(
            human_observation,
            str(human_observation.get("interaction_family") or ""),
            [command] if command else [],
        )
        command = guarded_commands[0] if guarded_commands else None
        if human_policy_guard in {
            "find_instruction_grounding_blocked",
            "find_unseen_target_blocked",
        }:
            proto._pending_cmds = []
            proto._pending_human_say = ""
        public_command = (
            proto._public_find_text(human_observation, command_before_repair)
            if command_before_repair and hasattr(proto, "_public_find_text")
            else command_before_repair
        )
        proto._transcript.append(
            {
                "turn": proto._n["ticks"],
                "assistant_mode": assistant.get("mode", ""),
                "assistant_say": assistant.get("say", ""),
                "assistant_raw": proto._last_assistant_raw,
                "human_mode": "interaction_queue",
                "human_model_call": False,
                "human_reasoning": "(continuing the active local instruction)",
                "human_parsed_handle_actions": (
                    [public_command] if public_command else []
                ),
                "human_translated_actions": [command_before_repair],
                "human_action": command or "",
                "human_queued": list(proto._pending_cmds),
                "human_say": proto._last_human_say,
                "human_say_deferred": deferred_say,
                "human_raw": "",
                "human_system_prompt": "",
                "human_user_prompt": "",
                "human_policy_guard": human_policy_guard,
                "human_policy_audit": getattr(
                    proto, "_last_human_policy_audit", "pass"
                ),
                "human_policy_audit_reasons": list(
                    getattr(proto, "_last_human_policy_audit_reasons", ())
                ),
                "find_human_control": str(
                    human_observation.get("interaction_find_human_control")
                    or "dynamic-guarded"
                ),
                "ask_human_control": str(
                    getattr(proto, "_ask_human_control", "strict")
                ),
                "visible_object_keys_internal": list(
                    human_observation.get("interaction_visible_object_keys") or ()
                ),
                "interaction_source": turn.source,
                "held": list(human_observation.get("held") or ()),
                "held_internal": list(held),
                "human_handle_map": dict(
                    human_observation.get("_interaction_human_handle_map") or {}
                ),
                "human_available_handles": list(
                    human_observation.get("_interaction_human_available_handles") or ()
                ),
                "human_object_interface": str(
                    human_observation.get("interaction_human_object_interface")
                    or "full_ids"
                ),
                "objects": list(human_observation.get("objects") or ()),
                "objects_visible_only": bool(
                    human_observation.get("objects_are_visible_only")
                ),
                "human_received_frames": int(
                    queue_origin.get("human_received_frames") or 0
                ),
                "human_frame_refs": list(queue_origin.get("human_frame_refs") or ()),
                "human_frame_provenance": list(
                    queue_origin.get("human_frame_provenance") or ()
                ),
                "vischeck": proto._last_vischeck,
                **(
                    {
                        "human_action_before_repair": command_before_repair,
                        "human_command_repair": command_repair,
                    }
                    if command_repair is not None
                    else {}
                ),
            }
        )
        return command, True, turn

    human_frames = frames if turn.receives_frames else []
    human_observation = interaction.human_observation(
        {
            **observation,
            "_interaction_find_human_control": getattr(
                proto, "_find_human_control", "dynamic-guarded"
            ),
            "_interaction_ask_human_control": getattr(
                proto, "_ask_human_control", "strict"
            ),
            "_interaction_orchestrate_human_turn_guard": bool(
                getattr(proto, "_interaction_orchestrate_human_turn_guard", True)
            ),
        },
        turn,
    )
    model_transcript_start = len(proto._transcript)
    try:
        command, row = _model_interaction_human_command(
            proto,
            human_observation,
            turn.new_assistant_speech,
            human_frames,
        )
    finally:
        for value in proto._transcript[model_transcript_start:]:
            if isinstance(value, dict):
                _annotate_interaction_human_row(value, turn, len(human_frames))
    if turn.source in {"watch_opening", "watch_default"} and proto._pending_cmds:
        discarded = list(proto._pending_cmds)
        proto._pending_cmds = []
        proto._n["watch_extra_actions_discarded"] = proto._n.get(
            "watch_extra_actions_discarded", 0
        ) + len(discarded)
        if row is not None:
            row["human_actions_discarded"] = discarded
            row["human_queued"] = []
    if command is None:
        interaction.note_model_idle(turn, proto._last_human_say)
    return command, False, turn


def _release_interaction_human_speech_after_action(
    proto: Any, *, succeeded: bool
) -> str:
    """Expose action-coupled speech only after its final command succeeds."""
    pending_say = str(getattr(proto, "_pending_human_say", "") or "").strip()
    if not pending_say:
        return ""
    if succeeded and proto._pending_cmds:
        return ""
    proto._pending_human_say = ""
    row = proto._transcript[-1] if proto._transcript else None
    if not succeeded:
        if isinstance(row, dict):
            row["human_say_discarded_after_failed_action"] = pending_say
        return ""
    proto._last_human_say = pending_say
    if isinstance(row, dict):
        row["human_say"] = pending_say
        row["human_say_released_after_success"] = True
    return pending_say


def _qualify_interaction_human_command(
    proto: Any, interaction: Any, command: str | None
) -> tuple[str | None, Mapping[str, object] | None]:
    """Apply one TaskPack-authored Human command qualifier and retain its audit."""
    if command is None:
        return None, None
    qualified, audit = interaction.qualify_human_command(command)
    if audit is None:
        return command, None
    if proto._transcript:
        proto._transcript[-1].update(
            {
                "human_action_before_qualification": command,
                "human_action": qualified,
                "human_command_qualification": dict(audit),
            }
        )
    return qualified, audit


def _scripted_interaction_human_command(
    proto: Any,
    interaction: Any,
    observation: Mapping[str, object],
    assistant: Mapping[str, object],
    turn: Any,
) -> tuple[str | None, bool, Any]:
    """Execute the frozen oracle action trace without calling the Human model."""
    if str(assistant.get("say") or "").strip() and not proto._pending_cmds:
        commands, completion_say = interaction.scripted_oracle_chunk()
        proto._pending_cmds = list(commands)
        proto._pending_human_say = completion_say
        proto._scripted_human_started = True
    if not proto._pending_cmds:
        proto._last_human_say = ""
        return None, False, turn
    from_queue = turn.source == "queued"
    command = proto._pending_cmds.pop(0)
    proto._last_human_say = ""
    deferred_say = proto._pending_human_say if not proto._pending_cmds else ""
    proto._last_human_scripted = True
    proto._transcript.append(
        {
            "turn": proto._n["ticks"],
            "assistant_mode": assistant.get("mode", ""),
            "assistant_say": assistant.get("say", ""),
            "assistant_raw": proto._last_assistant_raw,
            "human_mode": "scripted_oracle",
            "human_model_call": False,
            "human_reasoning": "(deterministic oracle demonstration)",
            "human_action": command,
            "human_queued": list(proto._pending_cmds),
            "human_say": proto._last_human_say,
            "human_say_deferred": deferred_say,
            "human_raw": command,
            "human_system_prompt": "",
            "human_user_prompt": "",
            "interaction_source": turn.source,
            "held": sorted(observation.get("held") or ()),
            "vischeck": proto._last_vischeck,
        }
    )
    return command, from_queue, turn


_PLACEMENT_PREDICATES: frozenset[str] = frozenset({"INSIDE", "ON"})


def _predicate_name(value: object) -> str:
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


def _placement_expectations(
    answer_key: Iterable[Mapping[str, object]],
) -> dict[str, set[tuple[str, str]]]:
    expected: dict[str, set[tuple[str, str]]] = {}
    for goal in answer_key:
        predicate = _predicate_name(goal.get("pred"))
        subject_value, target_value = goal.get("subject"), goal.get("target")
        if (
            predicate not in _PLACEMENT_PREDICATES
            or not subject_value
            or not target_value
        ):
            continue
        subject, target = str(subject_value), str(target_value)
        expected.setdefault(subject, set()).add((predicate, target))
    return expected


def _scored_targets(
    answer_key: Iterable[Mapping[str, object]],
) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    for goal in answer_key:
        subject_value, target_value = goal.get("subject"), goal.get("target")
        if not subject_value or not target_value:
            continue
        expected.setdefault(str(subject_value), set()).add(str(target_value))
    return expected


def _relation_triple(value: object) -> tuple[str, str, str] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    parts = list(value)
    if len(parts) != 3 or parts[1] is None or parts[2] is None:
        return None
    return _predicate_name(parts[0]), str(parts[1]), str(parts[2])


def _has_inside_ancestor(
    inner: str,
    outer: str,
    inside_parents: Mapping[str, set[str]],
) -> bool:
    if inner == outer:
        return True
    pending = [inner]
    seen = {inner}
    while pending:
        current = pending.pop()
        for parent in inside_parents.get(current, set()):
            if parent == outer:
                return True
            if parent not in seen:
                seen.add(parent)
                pending.append(parent)
    return False


def _wrong_placement_pairs(
    answer_key: Iterable[Mapping[str, object]],
    relations: Iterable[object],
    baseline_relations: Iterable[object] = (),
) -> set[str]:
    """Return wrong placements without mistaking a correct carrier for its outer container.

    Unary TaskPack goals serialize ``target == subject`` rather than naming a container. For those
    goals, only an ``ON``/``INSIDE`` carrier present at the authored handoff is trusted. An outer
    containment is legitimate only while that exact carrier relation still holds and the carrier
    reaches the outer container through the current ``INSIDE`` ancestry.
    """
    answer_key = list(answer_key)
    scored_targets = _scored_targets(answer_key)
    placement_expectations = _placement_expectations(answer_key)
    normalized = {
        triple
        for relation in relations
        if (triple := _relation_triple(relation)) is not None
    }
    baseline_carriers: dict[str, set[tuple[str, str]]] = {}
    for relation in baseline_relations:
        triple = _relation_triple(relation)
        if triple is None:
            continue
        predicate, subject, target = triple
        if predicate in _PLACEMENT_PREDICATES and subject in scored_targets:
            baseline_carriers.setdefault(subject, set()).add((predicate, target))

    inside_parents: dict[str, set[str]] = {}
    for predicate, subject, target in normalized:
        if predicate == "INSIDE":
            inside_parents.setdefault(subject, set()).add(target)

    wrong: set[str] = set()
    for predicate, subject, target in normalized:
        if predicate != "INSIDE" or subject not in scored_targets:
            continue
        if target in scored_targets[subject]:
            continue
        allowed_carriers = placement_expectations.get(
            subject, baseline_carriers.get(subject, set())
        )
        nested_through_correct_carrier = any(
            (expected_predicate, subject, expected_target) in normalized
            and _has_inside_ancestor(expected_target, target, inside_parents)
            for expected_predicate, expected_target in allowed_carriers
        )
        if not nested_through_correct_carrier:
            wrong.add(f"{subject}->{target}")
    return wrong


# ---------------------------------------------------------------------------------------------------
#  one episode
# ---------------------------------------------------------------------------------------------------
def run_episode(  # noqa: C901
    env: Any,
    proto: IsaacAgentLoop,
    backend: Any,
    dlog: Any,
    args: argparse.Namespace,
    obs: dict,
    goal_text: str = "",
    activity: str = "",
    task_id: str = "",
    interaction: Any | None = None,
) -> dict:
    """The tick loop of ``IsaacAgentLoop._run_episode``, synchronous and against a local ``IsaacEnv``."""
    import task_world_builder
    from ego_frames import EgoFrameBuffer
    from simulator.interaction.io_capture import FrameLedger

    budget = int(os.getenv("ISAAC_FRAME_BUDGET", "0") or 0)
    proto._find_human_control = str(
        getattr(args, "find_human_control", "dynamic-guarded")
    )
    proto._interaction_find_model_guard = _find_model_guard_enabled(args)
    proto._interaction_ask_instruction_guard = _ask_instruction_guard_enabled(args)
    proto._ask_human_control = str(getattr(args, "ask_human_control", "strict"))
    proto._interaction_guide_human_turn_guard = (
        str(getattr(args, "guide_human_turn_guard", "on")) == "on"
    )
    proto._interaction_watch_human_turn_guard = (
        str(getattr(args, "watch_human_turn_guard", "on")) == "on"
    )
    proto._interaction_orchestrate_human_turn_guard = (
        str(getattr(args, "orchestrate_human_turn_guard", "on")) == "on"
    )
    if interaction is not None:
        interaction.assert_tick_budget(proto.max_ticks)
    # TaskPack interaction evaluates exactly one assistant decision per physical action. Its configured
    # per-interval sample therefore stays fixed; max_frames handles the cumulative image ceiling. The
    # legacy benchmark keeps its historical per-tick budget schedule.
    sched = (
        []
        if interaction is not None
        else EgoFrameBuffer.budget_schedule(proto.max_ticks, budget, proto.per_interval)
    )
    clamp = getattr(args, "frame_clamp", "on") != "off"
    fb = EgoFrameBuffer(per_interval=proto.per_interval, max_frames=proto.max_frames)
    if sched and not clamp:
        fb.per_interval = sched[0]
    # The IO trail, when a driver attached one. `ledger` is kept unconditionally because it costs a
    # dict per interval and it is what makes "was the answer container in the delivered frames"
    # answerable per turn rather than once per episode.
    cap = getattr(backend, "io_capture", None)
    if interaction is not None:
        _akey = [dict(row) for row in interaction.answer_key]
    else:
        _akey = (
            [
                g
                for g in (task_world_builder.load_spec(task_id).get("answer_key") or [])
                if isinstance(g, dict)
            ]
            if task_id
            else []
        )
    answers = {str(g.get("target")) for g in _akey}
    _wrong: dict = {}
    _wrong_placement_baseline: tuple[object, ...] = ()
    ledger = FrameLedger(answers)
    if interaction is not None:
        proto._assistant_turn_trace = []
        proto._human_action_trace = []

    def _note_wrong_placements(o: dict) -> None:
        """Count when a scored object comes to rest in a non-answer container.

        Scoring is one conjunction on the FINAL state, ``[Grab]`` lifts an item back out of a
        container, and ``not_inside`` is read at the end -- so a wrong answer costs nothing and a
        BLIND assistant can simply enumerate the alphabet until the score latches (e.g. from a
        declared 0.20 chance to 1.00 in two guesses). Detection-only, by design: it changes no
        score and can fail no row. It exists so
        that "answered correctly" and "answered every option until one stuck" stop sharing a number.
        """
        for pair in _wrong_placement_pairs(
            _akey,
            o.get("relations") or (),
            _wrong_placement_baseline,
        ):
            _wrong[pair] = _wrong.get(pair, 0) + 1

    # ⛔ ``budget_schedule`` returns ZEROS for the early ticks whenever the budget is tight, and
    # assigning one straight onto ``fb.per_interval`` bypasses ``__init__``'s ``max(1, …)`` clamp
    # (``ego_frames``). At 0, ``add_interval`` stores NOTHING. For example, max_ticks 12,
    # per_interval 2, ISAAC_FRAME_BUDGET 12 give ``[0,0,0,0,1,1,1,1,2,2,2,2]``, so the start frame and
    # all three prologue intervals would be discarded, ``fb.pinned`` would become ``min(2, 0) == 0``,
    # and tick 0 would feed BOTH agents an empty image list: the scan circle runs and its pixels are
    # thrown away, so the episode is scored against a blind assistant and a blind person.
    # Two halves to the fix, both gated by --frame-clamp so the unclamped behaviour stays reproducible:
    #   * the PROLOGUE is not budgeted at all. It is the pinned look-around, the assistant's only survey
    #     of the room, and ``budgeted_frames``/``fed_frames`` already exempt it from every later trim --
    #     spending the schedule's thinnest allowance on it contradicts the pinning it then receives.
    #   * inside the tick loop the schedule is clamped to >= 1, so a tick can be thin but never blind.
    if interaction is not None:
        assert_unscored_state(obs, "TaskPack episode start")
    initial_frames = proto._obs_frames(obs)
    dlog.set_frame(len(initial_frames))
    if interaction is None or interaction.keep_opening_start_frame():
        fb.add_interval(initial_frames, _obs_frame_segs(obs))
        ledger.note(fb, "start_state", obs)

    # PROLOGUE. Nothing here is scored and a refused step is skipped -- a prologue must never be able
    # to fail an episode.
    #
    # An AUTHORED prologue REPLACES the generic look-around rather than adding to it: its first action
    # is already a walk to the item, so keeping both would double the walk and spend twice the frame
    # budget on the opening for no extra information.
    continuous_watch = bool(
        interaction is not None and interaction.opening.continuous_watch
    )
    if interaction is not None:
        pro = {
            "actions": list(interaction.opening.actions),
            "say": interaction.opening.prelude_say,
            "handoff_say": interaction.opening.handoff_say,
        }
        if pro["say"]:
            dlog.utterance(
                "human",
                pro["say"],
                channel="opening",
                phase="before_setup",
            )
    else:
        pro = _authored_prologue(task_id) if task_id else {"actions": [], "say": ""}
    if pro["actions"] and interaction is None:
        _assert_prologue_safe(task_id, pro)
    cmds = (
        []
        if continuous_watch
        else list(pro["actions"]) or proto._prologue_cmds(obs.get("goal") or [])
    )
    # "...stops in front of the cabinet wall". The item is not necessarily near the containers -- it
    # may be in a bathroom -- so a sweep from wherever the item was leaves the answer
    # containers out of every delivered frame, and the row measures nothing. `@BANK` is resolved at
    # runtime, answer-independently: see _bank_anchor.
    if "@BANK" in " ".join(cmds):
        anchor = _bank_anchor(env, task_id)
        print(f"[bench_eval][prologue] @BANK -> {anchor!r}", flush=True)
        cmds = [
            c.replace("@BANK", anchor)
            for c in cmds
            if not ("@BANK" in c and not anchor)
        ]
    n_rendered = 0
    for index, pc in enumerate(cmds):
        try:
            opening_speech: tuple[str, ...] = ()
            if interaction is not None:
                opening_speech = interaction.opening_utterances_before(index)
                for speech in opening_speech:
                    dlog.utterance(
                        "human",
                        speech,
                        channel="opening",
                        phase="before_setup_action",
                        opening_action_index=index,
                    )
            obs = env.step(pc)
            if interaction is not None:
                failures = [
                    str(value)
                    for value in (obs.get("reasons") or ())
                    if value and str(value) != "OK"
                ]
                if failures:
                    raise EvalConfigError(
                        f"TaskPack setup command {pc!r} failed: {'; '.join(failures)}"
                    )
            raw = proto._obs_frames(obs)
            n_rendered += len(raw)
            dlog.advance_frames(len(raw))
            if interaction is None or interaction.keep_opening_frame(
                pc, index, len(cmds)
            ):
                fb.add_interval(raw, _obs_frame_segs(obs))
                ledger.note(fb, f"prologue:{pc}", obs)
            proto._n["prologue_steps"] = proto._n.get("prologue_steps", 0) + 1
            dlog.marker("prologue_step", text=pc)
        except Exception as exc:  # noqa: BLE001 - see docstring
            if interaction is not None:
                raise
            proto._n["prologue_skipped"] = proto._n.get("prologue_skipped", 0) + 1
            print(
                f"[bench_eval][prologue] skipped {pc!r}: {str(exc)[:120]}", flush=True
            )
    # A temporal-memory opening may intentionally reach and later undo a scored
    # predicate before asking the model (for example, refrigerate a bottle and then
    # leave it in another room). It may also turn an initially true absence-derived
    # goal such as CLOSED into unfinished work. The evaluated policy's score baseline
    # therefore begins at the authored handoff, after all setup actions have run.
    score_baseline_captured = interaction is None
    if interaction is not None and not continuous_watch:
        obs = {**obs, "score": env.capture_score_baseline()}
        assert_unscored_state(obs, "TaskPack setup handoff")
        score_baseline_captured = True
    _wrong_placement_baseline = tuple(obs.get("relations") or ())
    if interaction is not None:
        proto._n["score_baseline_captured"] = int(score_baseline_captured)
    # ⛔ PIN THE WHOLE PROLOGUE, not the first two intervals. A fixed ``min(2, ...)`` only suits an
    # opening that is a single look-around; with a scripted opening it would leave everything after
    # the second interval -- the grab, the turn to the
    # containers -- competing with the live ticks and being restrided away. The opening is the most
    # information-dense part of the episode and it is exactly what must not be dropped.
    fb.pinned = (
        len(fb.intervals)
        if interaction is None or interaction.pin_opening_intervals()
        else 0
    )
    # The person has already spoken, deterministically, before any model was asked anything. This is
    # the channel ``_assistant_messages`` reads as ``heard`` on turn 0.
    if pro["say"] and interaction is None:
        proto._last_human_say = pro["say"]
        dlog.marker("prologue_say", text=pro["say"])
    timed_speech = (
        {utterance.say for utterance in interaction.opening.utterances}
        if interaction is not None
        else set()
    )
    if (
        interaction is not None
        and pro.get("handoff_say")
        and pro["handoff_say"] not in timed_speech
    ):
        dlog.utterance(
            "human",
            pro["handoff_say"],
            channel="opening",
            phase="after_setup",
        )
    proto._n["prologue_frames_rendered"] = n_rendered
    proto._n["prologue_intervals"] = len(fb.intervals)
    proto._n["prologue_pinned"] = fb.pinned
    # ⭐ "an inspect happened" and "its held-up frame was delivered" are different facts. Record both,
    # so a prologue whose cue was sampled away is visible in the manifest.
    proto._n["prologue_cue_intervals"] = fb.n_intervals_with_cue
    proto._n["prologue_cue_holds_pinned"] = fb.n_cue_holds_pinned
    # ⭐ AND: a [LookAt] that could not reach any reading stand and read the sign from across the room
    # (semantic_executor._square_on_stand -> scene.cue_stand_misses). One print in a long render
    # log is not a gate; this is the same fact where the manifest can see it.
    lost = list(getattr(getattr(env, "scene_obj", None), "cue_stand_misses", []) or [])
    proto._n["cue_stands_lost"] = len(lost)
    if lost:
        print(
            f"[bench_eval][prologue] ⛔ CUE-LOST x{len(lost)}: no reading stand was reachable for "
            f"{[m.get('sign_xy') for m in lost]} -- read from "
            f"{[m.get('dist_m') for m in lost]} m away. Those placards are probably NOT legible in "
            f"the delivered frames, so any option they carry is unscoreable.",
            flush=True,
        )
    av = (
        _answer_keys_visible(obs, answers)
        if interaction is not None
        else (_answers_visible(obs, task_id) if task_id else {"decidable": False})
    )
    proto._n["answers_visible"] = av.get("n_visible")
    proto._n["answers_total"] = av.get("n_answers")
    proto._vis_after_prologue = av
    print(
        f"[bench_eval][prologue] answer containers visible after the prologue: "
        f"{av.get('n_visible')}/{av.get('n_answers')} seen={av.get('visible')} "
        f"missing={av.get('missing')} decidable={av.get('decidable')}",
        flush=True,
    )
    if interaction is None:
        proto._pick_error_turn(proto.max_ticks)
    proto._stop_reason = "max_ticks"
    print(
        f"[bench_eval] FRAMES sched={sched} clamp={'on' if clamp else 'off'} "
        f"prologue_intervals={len(fb.intervals)} pinned={fb.pinned} "
        f"prologue_frames_rendered={n_rendered} "
        f"cue_intervals={fb.n_intervals_with_cue} cue_holds_pinned={fb.n_cue_holds_pinned} "
        f"answers_visible={proto._n.get('answers_visible')}/{proto._n.get('answers_total')} "
        f"tick0_frames={len(fb.fed_frames())} frames_per_seg="
        f"{os.getenv('ISAAC_FRAMES_PER_SEG', '1')} "
        f"human_frame_policy={getattr(args, 'human_frame_policy', 'recent')} "
        f"human_image_cap={int(getattr(args, 'human_frame_cap', 32) or 0)} "
        f"seeded_say={pro['say']!r}",
        flush=True,
    )
    if clamp and not fb.fed_frames():
        raise EvalConfigError(
            "the frame buffer is EMPTY after the prologue -- the assistant and the person would both "
            "open the episode blind, which is the defect --frame-clamp exists to prevent"
        )

    recovery_budget = _WatchPreAssistantRecoveryBudget(
        proto.max_ticks,
        int(
            getattr(
                args,
                "watch_preassistant_recovery_allowance",
                _DEFAULT_WATCH_PREASSISTANT_RECOVERY_ALLOWANCE,
            )
        ),
        bool(
            interaction is not None
            and str(getattr(interaction.spec, "family", "")) == "WATCH"
            and bool(getattr(proto, "_interaction_watch_human_turn_guard", True))
        ),
    )
    fresh_frames = True
    assistant_frame_batches: list[list] = []
    human_frame_batches: list[list] = []
    recovery_observation_snapshot: dict[str, object] | None = None

    def update_recovery_observation_context(accounting: str) -> None:
        nonlocal recovery_observation_snapshot
        if accounting == "recovery_chain_start":
            recovery_observation_snapshot = (
                _snapshot_watch_recovery_observation_context(
                    backend,
                    assistant_frame_batches,
                    human_frame_batches,
                    ledger,
                    proto._last_human_say,
                )
            )
            return
        if accounting == "recovery_chain_closed":
            if recovery_observation_snapshot is None:
                raise EvalConfigError(
                    "WATCH recovery chain closed without an observation snapshot"
                )
            proto._last_human_say = _restore_watch_recovery_observation_context(
                backend,
                assistant_frame_batches,
                human_frame_batches,
                ledger,
                recovery_observation_snapshot,
            )
            recovery_observation_snapshot = None
            recovery_budget.note_context_restored()
            return
        if (
            recovery_observation_snapshot is not None
            and not recovery_budget.active_expected
        ):
            recovery_observation_snapshot = None

    for tick in range(proto.max_ticks + recovery_budget.allowance):
        if not recovery_budget.can_start_turn():
            break
        recovery_budget.begin_turn()
        proto._n["ticks"] = tick + 1
        dlog.set_tick(tick)
        if (
            interaction is not None
            and not score_baseline_captured
            and interaction.score_baseline_ready()
        ):
            obs = {**obs, "score": env.capture_score_baseline()}
            assert_unscored_state(obs, "TaskPack continuous WATCH handoff")
            score_baseline_captured = True
            proto._n["score_baseline_captured"] = 1
            proto._n["score_baseline_tick"] = tick
        # A backend that is not a model needs the live state to decide what to say; a served model must
        # never get it. Only backends that opt in by defining the hook are handed anything.
        if interaction is not None and hasattr(backend, "set_interaction_state"):
            control_state = interaction.assistant_state(obs)
            if bool(getattr(backend, "accepts_private_control_state", False)):
                visible = obs.get("_visible_object_keys_full")
                if not isinstance(visible, (list, tuple, set)):
                    vischeck = obs.get("vischeck")
                    visible = (
                        vischeck.get("visible") if isinstance(vischeck, Mapping) else ()
                    )
                control_state = {
                    **control_state,
                    "visible_object_keys": [str(name) for name in visible or ()],
                    "successful_commands": list(interaction.successful_commands),
                }
            backend.set_interaction_state(control_state)
        if hasattr(backend, "on_tick"):
            backend.on_tick(obs, proto, tick)
        # (a) ASSISTANT -- pixels + the person's speech, nothing symbolic.
        handed, omitted_opening_frames = _assistant_frame_batch(
            fb,
            tick,
            fresh_frames,
            interaction,
            int(getattr(args, "assistant_frame_cap", 0) or 0),
        )
        _assert_initial_interaction_frame_cap(args, handed, tick, interaction)
        assistant_frame_batches.append(list(handed))
        human_handed, _ = _assistant_frame_batch(
            fb,
            tick,
            fresh_frames,
            interaction,
            int(getattr(args, "human_frame_cap", 32) or 0),
        )
        human_frame_batches.append(list(human_handed))
        if interaction is not None and tick == 0:
            proto._n["assistant_opening_frames"] = len(handed)
        if omitted_opening_frames:
            proto._n["assistant_opening_history_frames_omitted"] = (
                proto._n.get("assistant_opening_history_frames_omitted", 0)
                + omitted_opening_frames
            )
        new_prov = ledger.deliver(handed)
        first_prompt = tick == 0
        assistant_view = (
            interaction.assistant_observation(obs, first=first_prompt)
            if interaction is not None
            else (assistant_obs(obs, goal_text) if goal_text else obs)
        )
        heard = proto._last_human_say
        if interaction is not None and tick == 0:
            heard = " ".join(
                value
                for value in (
                    interaction.first_handoff_say(),
                    proto._last_human_say,
                )
                if value
            )
        msgs = _with_declare_clause(
            proto._assistant_messages(
                assistant_view,
                first=first_prompt,
                frames=handed,
                heard=heard,
            )
        )
        if not fresh_frames:
            proto._n["idle_textonly_turns"] += 1
        fresh_frames = False
        heard_this_tick = heard
        proto._last_human_say = ""
        if cap is not None:
            cap.begin_turn(tick)
            cap.set_frame_provenance(new_prov)
            cap.record(
                heard=heard_this_tick,
                n_frames_new=len(handed),
                frames_new=new_prov,
                answers_visible_cumulative=ledger.answers_seen(),
                answers_visible_last_step=(
                    _answer_keys_visible(obs, answers)
                    if interaction is not None
                    else (_answers_visible(obs, task_id) if task_id else None)
                ),
                held=sorted(obs.get("held") or []),
            )
        t0 = time.perf_counter()
        raw = backend.generate(msgs)
        proto._tk["t_gen_s"].append(round(proto._tadd("t_gen_s", t0), 3))
        decision_fn = getattr(backend, "decision_text", None)
        decision_raw = decision_fn(raw) if decision_fn is not None else raw
        metadata_fn = getattr(backend, "turn_metadata", None)
        turn_metadata = metadata_fn() if metadata_fn is not None else {}
        a_out = parse_assistant_eval(proto, decision_raw)
        recovery_budget.note_assistant(
            mode=str(a_out.get("mode") or "silence"),
            say=str(a_out.get("say") or ""),
            done=bool(a_out.get("done")),
        )
        update_recovery_observation_context("assistant_turn")
        deferred_done = _should_defer_assistant_done(
            a_out, interaction, proto._pending_cmds
        )
        if deferred_done:
            proto._n["assistant_done_deferred"] = (
                proto._n.get("assistant_done_deferred", 0) + 1
            )
            print(
                f"[bench_eval] deferring Assistant $done$ at tick {tick}: "
                f"the Human still has {len(proto._pending_cmds)} queued action(s)",
                flush=True,
            )
            dlog.marker(
                "assistant_done_deferred",
                text=f"tick={tick}",
                ref=f"{len(proto._pending_cmds)} queued action(s)",
            )
            a_out = {**a_out, "mode": "silence", "say": "", "done": False}
        proto._last_assistant_mode = a_out.get("mode", "silence")
        if interaction is not None:
            assistant_state = interaction.assistant_state(obs)
            interaction.note_assistant_turn(
                mode=str(a_out.get("mode") or "silence"),
                say=str(a_out.get("say") or ""),
                done=bool(a_out.get("done")),
                goal_success=bool(assistant_state.get("goal_success")),
            )
            proto._assistant_turn_trace.append(
                {
                    "tick": tick,
                    "mode": a_out.get("mode"),
                    "say": a_out.get("say", ""),
                    "raw": raw,
                    "decision_raw": decision_raw,
                    "heard": heard_this_tick,
                    "n_frames": len(handed),
                    "prompt": _prompt_for_audit(
                        backend.audit_messages(msgs)
                        if hasattr(backend, "audit_messages")
                        else msgs
                    ),
                    "interaction_state": assistant_state,
                    "assistant_done_deferred": deferred_done,
                    **turn_metadata,
                }
            )
        if cap is not None:
            cap.record(
                parsed_mode=a_out.get("mode"),
                parsed_say=a_out.get("say"),
                parsed_done=bool(a_out.get("done")),
                **turn_metadata,
            )
        if a_out.get("mode") == "silence":
            dlog.silence(
                "assistant",
                prompted=True,
                reason=(
                    "deferred_done"
                    if deferred_done
                    else (
                        "empty_generation"
                        if not (raw or "").strip()
                        else "chose_silence"
                    )
                ),
                mode=a_out.get("mode"),
            )
        else:
            dlog.utterance(
                "assistant",
                a_out.get("say", "") or "(declares the task finished)",
                raw_text=raw[:600],
                mode=a_out.get("mode"),
            )
        if a_out.get("done"):
            # Scoring happens before the person's turn only after any already-accepted TaskPack
            # action sequence has drained.
            proto._stop_reason = "assistant_done"
            proto._n["declared_at_tick"] = tick
            print(
                f"[bench_eval] ⭐ ASSISTANT DECLARED DONE at tick {tick} -- the world is scored as "
                f"it stands now; nothing after this counts.",
                flush=True,
            )
            dlog.marker("assistant_done", text=f"tick={tick}")
            break

        # (b) HUMAN. TaskPack interaction preempts an unfinished motor queue whenever the assistant
        # speaks, then asks the configured Human backend to realize the new instruction.
        interaction_turn = None
        proto._last_human_scripted = False
        human_frames = select_human_frames(
            fb,
            human_frame_batches,
            str(getattr(args, "human_frame_policy", "recent")),
            int(getattr(args, "human_frame_cap", 32) or 0),
        )
        human_transcript_start = len(proto._transcript)
        if interaction is not None:
            try:
                cmd, from_queue, interaction_turn = _interaction_human_command(
                    proto, interaction, obs, a_out, human_frames
                )
            except HumanInvalidAttemptLimitError as exc:
                if cap is not None:
                    _record_interaction_human_inputs(
                        cap,
                        ledger,
                        proto._transcript[human_transcript_start:],
                        human_frames,
                        tick,
                    )
                proto._stop_reason = _HUMAN_INVALID_ATTEMPT_STOP_REASON
                if proto._assistant_turn_trace:
                    proto._assistant_turn_trace[-1][
                        "human_invalid_attempt_limit_exhausted"
                    ] = True
                    proto._assistant_turn_trace[-1]["human_policy_guard"] = exc.guard
                dlog.marker(
                    _HUMAN_INVALID_ATTEMPT_STOP_REASON,
                    text=f"tick={tick}",
                    ref=exc.guard,
                )
                if cap is not None:
                    cap.record(
                        human_invalid_attempt_limit_exhausted=True,
                        human_policy_guard=exc.guard,
                        env_result="(episode aborted before any Human action)",
                    )
                break
        else:
            if a_out.get("mode") != "silence" and proto._pending_cmds:
                proto._n["queue_interrupted"] += 1
            from_queue = bool(proto._pending_cmds)
            if proto._pending_cmds:
                held_now = sorted(obs.get("held", []))
                cmd = proto._repair_one_arg_put(proto._pending_cmds.pop(0), held_now)
                proto._n["queue_played"] += 1
                if a_out.get("say"):
                    proto._carry_say.append(a_out["say"].strip())
                proto._transcript.append(
                    {
                        "turn": proto._n["human_calls"],
                        "assistant_mode": a_out.get("mode", ""),
                        "assistant_say": a_out.get("say", ""),
                        "human_mode": "queued",
                        "human_model_call": False,
                        "human_reasoning": "(continuing the sequence planned last turn)",
                        "human_action": cmd or "",
                        "human_queued": list(proto._pending_cmds),
                        "human_say": "",
                        "held": list(held_now),
                        "assistant_raw": proto._last_assistant_raw,
                        "human_raw": "",
                        "vischeck": proto._last_vischeck,
                    }
                )
            else:
                t0 = time.perf_counter()
                cmd = proto._human_command(
                    human_obs(obs, activity) if activity else obs,
                    proto._merge_carried_say(a_out.get("say", "")),
                    human_frames,
                )
                proto._n["human_calls"] += 1
                proto._tk["t_human_s"].append(round(proto._tadd("t_human_s", t0), 3))
        command_before_qualification = cmd
        command_qualification = None
        if interaction is not None:
            cmd, command_qualification = _qualify_interaction_human_command(
                proto, interaction, cmd
            )
        if cap is not None and interaction is not None:
            _record_interaction_human_inputs(
                cap,
                ledger,
                proto._transcript[human_transcript_start:],
                human_frames,
                tick,
            )
        if proto._last_human_say:
            proto._n["human_spoke"] = proto._n.get("human_spoke", 0) + 1
            dlog.utterance(
                "human", proto._last_human_say, channel="speech", acted=bool(cmd)
            )
        if cap is not None:
            cap.record(
                human_source=(
                    "deterministic oracle demonstration"
                    if proto._last_human_scripted
                    else (
                        "queue (planned on an earlier turn, the 32B was not called)"
                        if from_queue
                        else "the frozen 32B, called this turn"
                    )
                ),
                human_command=cmd or "",
                human_say=proto._last_human_say,
                human_reasoning=(
                    proto._transcript[-1].get("human_reasoning")
                    if proto._transcript
                    else ""
                ),
                human_raw=(
                    proto._transcript[-1].get("human_raw") if proto._transcript else ""
                ),
            )
        if cmd:
            if interaction is not None:
                human_row = proto._transcript[-1] if proto._transcript else {}
                action_trace = {
                    "tick": tick,
                    "source": interaction_turn.source,
                    "command": cmd,
                    "held_before": sorted(obs.get("held") or []),
                    "assistant_say": a_out.get("say", ""),
                    "expected_watch_command": interaction_turn.expected_watch_command,
                    "human_mode": human_row.get("human_mode", ""),
                    "human_raw": human_row.get("human_raw", ""),
                    "human_policy_guard": human_row.get("human_policy_guard", ""),
                    "human_policy_audit": human_row.get("human_policy_audit", "pass"),
                    "human_policy_audit_reasons": list(
                        human_row.get("human_policy_audit_reasons") or ()
                    ),
                    "find_human_control": human_row.get(
                        "find_human_control", "dynamic-guarded"
                    ),
                    "ask_human_control": human_row.get("ask_human_control", "strict"),
                    "result": "pending",
                    "attempts": [],
                }
                if command_qualification is not None:
                    action_trace.update(
                        {
                            "command_before_qualification": (
                                command_before_qualification
                            ),
                            "command_qualification": dict(command_qualification),
                        }
                    )
                if "human_command_repair" in human_row:
                    action_trace.update(
                        {
                            "command_before_repair": human_row.get(
                                "human_action_before_repair"
                            ),
                            "command_repair": dict(
                                human_row.get("human_command_repair") or {}
                            ),
                        }
                    )
                proto._human_action_trace.append(action_trace)
            # The provenance, not a blanket strict=False: a command the lenient path REWROTE
            # (`[LookAt] item_1 (1)` -> `[LookAt] <item_1> (1)`) must be distinguishable in the log
            # from one the model wrote to the grammar, or the repair rate cannot be measured at all.
            dlog.command(
                "human",
                (proto._transcript[-1].get("human_raw") if proto._transcript else "")
                or cmd,
                cmd,
                channel="action",
                parse=_parse_provenance(
                    proto,
                    cmd,
                    scripted=proto._last_human_scripted,
                    qualified_from=(
                        command_before_qualification
                        if command_qualification is not None
                        else None
                    ),
                ),
                attempt=1,
                logical_tick=tick,
            )
        if not cmd:
            proto._n["human_none"] += 1
            tick_accounting = recovery_budget.note_no_action()
            update_recovery_observation_context(tick_accounting)
            if interaction is not None and proto._assistant_turn_trace:
                proto._assistant_turn_trace[-1]["watch_tick_accounting"] = (
                    tick_accounting
                )
            if cap is not None:
                cap.record(env_result="(the person issued no command this tick)")
            continue

        # (c) STEP.
        watch_motor_recovery = bool(
            interaction is not None
            and getattr(interaction_turn, "source", "")
            in {"watch_opening", "watch_default"}
            and _last_command_failed(proto._last_cmd_error)
        )
        watch_counters_before = _watch_recovery_counter_snapshot(interaction)
        model_human_retry = _model_human_action_retry_enabled(
            proto,
            interaction,
            interaction_turn,
            a_out,
            cmd,
        )
        watch_failure_audit = _watch_action_failure_audit_enabled(interaction)
        action_retry_invalid_before = int(proto._n.get("action_retry_invalid", 0))
        obs, step_attempts, step_exc = _run_model_human_action_attempts(
            env,
            proto,
            cmd,
            obs,
            retry_enabled=model_human_retry,
            classify_failures=model_human_retry or watch_failure_audit,
            invalidate_classified_failures=(
                model_human_retry
                or (watch_failure_audit and not recovery_budget.assistant_has_spoken)
            ),
        )
        if watch_failure_audit and len(step_attempts) > 1:
            proto._n["watch_action_retry_attempts"] = (
                proto._n.get("watch_action_retry_attempts", 0) + len(step_attempts) - 1
            )
        if sched:
            allot = sched[min(tick, len(sched) - 1)]
            fb.per_interval = max(1, allot) if clamp else allot
        visible_attempt = next(
            (
                attempt
                for attempt in reversed(step_attempts)
                if isinstance(attempt.get("_observation"), dict)
            ),
            None,
        )
        public_attempts: list[dict[str, object]] = []
        for attempt in step_attempts:
            attempt_observation = attempt.get("_observation")
            public_attempt = {
                key: value for key, value in attempt.items() if not key.startswith("_")
            }
            public_attempt["model_visible"] = attempt is visible_attempt
            if not isinstance(attempt_observation, dict):
                public_attempts.append(public_attempt)
                continue
            attempt_index = int(attempt.get("attempt") or 1)
            attempt_score = attempt_observation.get("score", {}) or {}
            attempt_bad = list(attempt.get("failures") or ())
            rendered_frames = proto._obs_frames(attempt_observation)
            public_attempt["frame_count"] = len(rendered_frames)
            public_attempts.append(public_attempt)
            if attempt_index > 1:
                dlog.marker(
                    "action_retry_attempt",
                    text=cmd,
                    attempt=attempt_index,
                    same_logical_tick=True,
                )
            if attempt is visible_attempt:
                proto._vis_absorb(attempt_observation)
                dlog.advance_frames(len(rendered_frames))
                fb.add_interval(rendered_frames, _obs_frame_segs(attempt_observation))
                ledger.note(
                    fb,
                    f"tick{tick}:attempt{attempt_index}:{cmd}",
                    attempt_observation,
                )
            dlog.marker(
                "env_step",
                text=cmd,
                ref=(f"{attempt_score.get('n_done')}/{attempt_score.get('n_total')}"),
                refused=bool(attempt_bad),
                attempt=attempt_index,
                same_logical_tick=True,
                model_visible=attempt is visible_attempt,
                frame_count=len(rendered_frames),
            )
            _note_wrong_placements(attempt_observation)
            if attempt_observation.get("walk_no_motion"):
                proto._n["walk_no_motion"] = proto._n.get("walk_no_motion", 0) + 1
                dlog.marker(
                    "walk_no_motion",
                    text=cmd,
                    ref=",".join(
                        attempt_observation["walk_no_motion"].get("keys") or []
                    ),
                    attempt=attempt_index,
                )
            if attempt.get("will_retry"):
                dlog.marker(
                    "action_retry_scheduled",
                    text=cmd,
                    ref=",".join(attempt.get("retry_reasons") or ()),
                    attempt=attempt_index,
                    next_attempt=attempt_index + 1,
                    same_logical_tick=True,
                )
            if attempt_score.get("success") and "first_goal_tick" not in proto._n:
                proto._n["first_goal_tick"] = tick
        if interaction is not None:
            proto._human_action_trace[-1]["attempts"] = public_attempts
            proto._human_action_trace[-1]["attempt_count"] = len(public_attempts)
        if step_exc is not None:
            exc = step_exc
            if is_recoverable_step_error(exc):
                proto._last_cmd_error = (
                    f"Your last command {cmd!r} was REJECTED: "
                    f"{str(exc).splitlines()[0][:160]}. Re-issue it correctly this turn."
                )
                proto._last_successful_human_command = ""
                proto._n["bad_command"] = proto._n.get("bad_command", 0) + 1
                proto._pending_cmds = []
                if interaction is not None:
                    _release_interaction_human_speech_after_action(
                        proto, succeeded=False
                    )
                    interaction.note_step(
                        interaction_turn,
                        cmd,
                        succeeded=False,
                        human_say=proto._last_human_say,
                        action_sequence_complete=not proto._pending_cmds,
                        motor_recovery=watch_motor_recovery,
                        human_command=command_before_qualification,
                    )
                    proto._human_action_trace[-1]["result"] = "rejected"
                    proto._human_action_trace[-1]["error"] = str(exc)[:300]
                    proto._human_action_trace[-1]["retry_disposition"] = (
                        "human_command_rejection"
                        if is_recoverable_step_error(exc)
                        else "ambiguous_exception"
                    )
                    proto._human_action_trace[-1]["retry_invalid"] = (
                        int(proto._n.get("action_retry_invalid", 0))
                        > action_retry_invalid_before
                    )
                    flags = _watch_recovery_step_flags(
                        interaction,
                        watch_counters_before,
                        str(getattr(interaction_turn, "source", "")),
                    )
                    tick_accounting = recovery_budget.note_action(
                        source=str(getattr(interaction_turn, "source", "")),
                        expected=str(
                            getattr(interaction_turn, "expected_watch_command", "")
                            or ""
                        ),
                        human_say=proto._last_human_say,
                        **flags,
                    )
                    update_recovery_observation_context(tick_accounting)
                    proto._human_action_trace[-1]["watch_tick_accounting"] = (
                        tick_accounting
                    )
                dlog.marker(
                    "command_rejected",
                    text=cmd,
                    ref=str(exc)[:160],
                    attempt=len(step_attempts),
                )
                print(
                    f"[bench_eval] command rejected, skipping tick: {cmd!r}", flush=True
                )
                if cap is not None:
                    cap.record(
                        env_result=f"REJECTED by the engine: {str(exc).splitlines()[0][:200]}",
                        env_bad=["command_rejected"],
                        env_action_attempts=public_attempts,
                    )
                continue
            raise
        sc = obs.get("score", {}) or {}
        _outs, bad = _step_outcomes_and_failures(obs)
        if interaction is not None:
            released_human_say = _release_interaction_human_speech_after_action(
                proto, succeeded=not bad
            )
            if released_human_say:
                proto._n["human_spoke"] = proto._n.get("human_spoke", 0) + 1
                dlog.utterance(
                    "human",
                    released_human_say,
                    channel="speech",
                    acted=True,
                    after_action_success=True,
                )
                if cap is not None:
                    cap.record(human_say=released_human_say)
            interaction.note_step(
                interaction_turn,
                cmd,
                succeeded=not bad,
                human_say=proto._last_human_say,
                action_sequence_complete=not proto._pending_cmds,
                motor_recovery=watch_motor_recovery,
                observation=obs,
                human_command=command_before_qualification,
            )
            proto._human_action_trace[-1].update(
                {
                    "result": "failed" if bad else "succeeded",
                    "failures": list(bad),
                    "score": dict(sc),
                    "retry_disposition": (
                        "recovered"
                        if len(public_attempts) > 1 and not bad
                        else (
                            public_attempts[-1].get("retry_decision")
                            if public_attempts
                            else ""
                        )
                    ),
                    "retry_invalid": bool(
                        int(proto._n.get("action_retry_invalid", 0))
                        > action_retry_invalid_before
                    ),
                }
            )
            flags = _watch_recovery_step_flags(
                interaction,
                watch_counters_before,
                str(getattr(interaction_turn, "source", "")),
            )
            tick_accounting = recovery_budget.note_action(
                source=str(getattr(interaction_turn, "source", "")),
                expected=str(
                    getattr(interaction_turn, "expected_watch_command", "") or ""
                ),
                human_say=proto._last_human_say,
                **flags,
            )
            update_recovery_observation_context(tick_accounting)
            proto._human_action_trace[-1]["watch_tick_accounting"] = tick_accounting
        proto._last_cmd_error = outcome_text(
            cmd,
            bad,
            sc,
            _effective_progress_line(args, interaction),
            obs.get("walk_no_motion"),
        )
        proto._last_successful_human_command = "" if bad else cmd
        if bad:
            proto._n["step_refused"] = proto._n.get("step_refused", 0) + 1
        if cap is not None:
            cap.record(
                env_result=proto._last_cmd_error or "(succeeded, nothing to report)",
                env_bad=bad,
                score={
                    "n_done": sc.get("n_done"),
                    "n_total": sc.get("n_total"),
                    "success": bool(sc.get("success")),
                },
                walk_no_motion=bool(obs.get("walk_no_motion")),
                held_after=sorted(obs.get("held") or []),
                env_action_attempts=public_attempts,
            )
        fresh_frames = True
        # ⛔ NO HALT-ON-GOAL. Stopping the instant object state happens to satisfy the key would freeze
        # the episode at a guesser's luckiest moment and credit it -- see the DECLARE-DONE note above.
        # The tick is recorded because "when did it first become true" is still worth knowing; it is
        # not a stop condition and it is not the score.
        _ = heard_this_tick
    recovery_budget.finish_episode()
    recovery_budget.publish(proto._n)
    hard_invalid_human_abort = proto._stop_reason == _HUMAN_INVALID_ATTEMPT_STOP_REASON
    if not hard_invalid_human_abort:
        _assert_interaction_termination(proto, interaction)
    if hard_invalid_human_abort:
        proto._n["never_declared"] = 1
        proto._n["declared_at_tick"] = None
        print(
            "[bench_eval] ⛔ HARD INVALID: the Human exhausted its same-tick policy "
            "retries; the episode stopped before any further Assistant turn or env action.",
            flush=True,
        )
    elif proto._stop_reason != "assistant_done":
        proto._n["never_declared"] = 1
        proto._n["declared_at_tick"] = None
        print(
            f"[bench_eval] ⚠️ NEVER DECLARED: the tick budget ran out with no $done$ "
            f"(stop_reason={proto._stop_reason!r}). Scored at timeout; this is an assistant "
            f"failure to commit, not a neutral end.",
            flush=True,
        )
    else:
        proto._n["never_declared"] = 0
    proto._n["wrong_placements"] = len(_wrong)
    proto._n["wrong_placement_pairs"] = sorted(_wrong)
    if _wrong:
        print(
            f"[bench_eval] ⚠️ WRONG-PLACEMENT x{len(_wrong)}: a scored object was INSIDE a container "
            f"that is not its answer at some point in this episode -- {sorted(_wrong)}. Final-state "
            f"scoring cannot see this, so a score of 1.00 here is 'found it eventually', not "
            f"'answered correctly'.",
            flush=True,
        )
    if cap is not None:
        cap.record_episode_frames(ledger)
    return obs


# ---------------------------------------------------------------------------------------------------
#  scoring + artifacts
# ---------------------------------------------------------------------------------------------------
def _benchmark_validity_fields(
    *, eval_valid: object, episode_valid: object, goal_scored: object
) -> dict[str, bool]:
    evaluation_valid = bool(eval_valid)
    return {
        "eval_valid": evaluation_valid,
        "benchmark_usable": bool(evaluation_valid and episode_valid and goal_scored),
    }


def _manifest_task_goal(bt: Mapping[str, object], score: Mapping[str, object]) -> dict:
    """Adapt the environment's canonical score to the render-manifest field names."""
    out = {
        "task": bt.get("task"),
        "container": bt.get("container"),
        "met": bool(score.get("success")),
        "progress": float(score.get("progress_positive") or 0.0),
        "n": int(score.get("n_positive_total") or 0),
        "progress_denominator": "positive_goals",
        "n_sub_goals": int(score.get("n_sub_goals") or 0),
        "n_sub_goals_ok": int(score.get("n_sub_goals_ok") or 0),
        "sub_goal_scoring": score.get("sub_goal_scoring"),
        "n_achievements": int(score.get("n_achievements") or 0),
        "n_achievements_ok": int(score.get("n_achievements_ok") or 0),
        "progress_achievements": float(score.get("progress_achievements") or 0.0),
        "n_prohibitions": int(score.get("n_prohibitions") or 0),
        "n_prohibitions_ok": int(score.get("n_prohibitions_ok") or 0),
        "prohibitions_ok": bool(score.get("prohibitions_ok")),
        "progress_sub_goals": float(score.get("progress_sub_goals") or 0.0),
        "goals": [dict(row) for row in score.get("goals") or ()],
        "aggregates": [dict(row) for row in score.get("aggregates") or ()],
        "outcomes": [dict(row) for row in score.get("outcomes") or ()],
        "sub_goals": [dict(row) for row in score.get("sub_goals") or ()],
    }
    negative = [dict(row) for row in score.get("neg") or ()]
    if negative:
        out["neg"] = negative
        out["neg_ok"] = bool(score.get("prohibitions_ok")) and all(
            bool(row.get("ok")) for row in negative
        )
    return out


def score_and_write(
    env: Any,
    proto: IsaacAgentLoop,
    backend: Any,
    dlog: Any,
    args: argparse.Namespace,
    task_id: str,
    out_dir: str,
    tag: str,
    t_start: float,
) -> dict:
    """Write the canonical environment score and the interaction sidecars."""
    import episode_runtime as s3

    scene = env.scene_obj
    bt = getattr(scene, "behavior_task", None) or {}
    task_goal = _manifest_task_goal(bt, env.score()) if bt else None

    ns = argparse.Namespace(
        scene=f"spec:{task_id}",
        seed=int(args.seed),
        subdir=args.subdir,
        tag=tag,
        check=False,
    )
    # ``setup_episode`` names it render_episode_valid (``episode_runtime``) -- the certified start
    # pose. Absent would mean the context changed shape, which must not read as a silent True.
    ctx = env.ctx or {}
    if "render_episode_valid" not in ctx:
        raise EvalConfigError(
            "setup_episode context has no render_episode_valid -- refusing to guess"
        )
    init_valid = bool(ctx["render_episode_valid"])
    counters = eval_counters(proto, backend)
    requires_causal_review = bool(counters.get("requires_causal_review"))
    validity = _benchmark_validity_fields(
        eval_valid=counters.get("eval_valid"),
        episode_valid=init_valid,
        goal_scored=task_goal is not None and int(task_goal.get("n", 0) or 0) > 0,
    )
    # compliance=None -> no picture grade was produced (this driver renders obs frames, not a video);
    # the manifest records render_grade null rather than a fabricated "clean".
    s3._write_episode_manifest(
        out_dir,
        tag,
        ns,
        init_valid,
        None,
        task_goal,
        extra={
            "n_fail": proto._n.get("bad_command", 0),
            "wall_time": time.time() - t_start,
            "human_backend": args.human_backend,
            "human_model": args.human_model,
            "requires_causal_review": requires_causal_review,
            **validity,
        },
    )
    dlog.close(counters=counters, stop_reason=proto._stop_reason)
    with open(os.path.join(out_dir, f"{tag}_eval.json"), "w") as fh:
        json.dump(
            {
                "task_id": task_id,
                "arm": args.arm,
                "assistant_backend": args.assistant_backend,
                "assistant_model": args.assistant_model or args.assistant_url,
                "human_backend": args.human_backend,
                "human_model": args.human_model,
                "human_url": args.human_url,
                "seed": int(args.seed),
                "tag": tag,
                **validity,
                "requires_causal_review": requires_causal_review,
                "counters": counters,
                "transcript": proto._transcript,
                "timing_s": {k: round(v, 2) for k, v in proto._t.items() if v},
            },
            fh,
            indent=1,
        )
        fh.write("\n")
    return counters


def eval_counters(proto: IsaacAgentLoop, backend: Any) -> dict:
    """The episode validity and interaction counters, plus the assistant-side failure counts.

    ``human_http_fail`` is the ONLY observable for the frame-budget trap, so it is promoted here and
    ``eval_valid`` is false whenever it is non-zero -- an episode that lost human turns to HTTP is not
    a measurement of anything.
    """
    n = proto._n
    http_fail = int(n.get("human_http_fail", 0)) + int(n.get("human_not_ok", 0))
    bstats = backend.stats()
    recovery_invalid = sum(
        int(n.get(name, 0))
        for name in (
            "watch_action_retry_attempts",
            "watch_preassistant_recovery_chains_started",
            "watch_preassistant_recovery_unrecovered_chains",
            "watch_preassistant_recovery_allowance_exceeded",
            "watch_preassistant_recovery_interrupted_by_assistant",
            "watch_preassistant_recovery_human_deviations",
            "watch_preassistant_recovery_human_speech",
            "watch_preassistant_recovery_stateful_detours",
            "watch_preassistant_recovery_observation_context_unsupported",
        )
    )
    human_adherence_invalid = int(n.get("human_invalid_attempt_limit_exhausted", 0))
    causal_review_count = sum(
        int(n.get(name, 0))
        for name in (
            "human_policy_audit_orchestrate_action_order_unverified",
            "human_policy_audit_watch_assistant_command_mismatch_blocked",
            "human_policy_audit_watch_assistant_command_alignment_needs_causal_review",
            "human_policy_audit_watch_assistant_command_alignment_unverified",
        )
    )
    return {
        "n_ticks": int(n.get("ticks", 0)),
        "n_raw_ticks": int(n.get("n_raw_ticks", n.get("ticks", 0))),
        "n_semantic_ticks": int(n.get("n_semantic_ticks", n.get("ticks", 0))),
        "n_env_steps": int(n.get("env_steps", 0)),
        "n_human_calls": int(n.get("human_calls", 0)),
        "n_human_invalid_attempts": int(n.get("human_invalid_attempts", 0)),
        "n_human_invalid_attempts_recovered": int(
            n.get("human_invalid_attempts_recovered", 0)
        ),
        "n_human_invalid_attempt_limit_exhausted": int(
            n.get("human_invalid_attempt_limit_exhausted", 0)
        ),
        "human_policy_audit_watch_assistant_command_alignment_unverified": int(
            n.get("human_policy_audit_watch_assistant_command_alignment_unverified", 0)
        ),
        "human_policy_audit_watch_assistant_command_mismatch_blocked": int(
            n.get("human_policy_audit_watch_assistant_command_mismatch_blocked", 0)
        ),
        "human_policy_audit_watch_assistant_command_alignment_needs_causal_review": int(
            n.get(
                "human_policy_audit_watch_assistant_command_alignment_needs_causal_review",
                0,
            )
        ),
        "human_policy_audit_orchestrate_action_order_unverified": int(
            n.get("human_policy_audit_orchestrate_action_order_unverified", 0)
        ),
        "requires_causal_review": causal_review_count > 0,
        "n_human_http_attempt_failures": int(n.get("human_http_attempt_failures", 0)),
        "n_human_http_retries": int(n.get("human_http_retries", 0)),
        "n_human_provider_truncated": int(n.get("human_provider_truncated", 0)),
        "n_human_asked": int(n.get("human_asked", 0)),
        "n_human_said_generic": int(n.get("human_said_generic", 0)),
        "n_say_spatial": int(n.get("say_spatial", 0)),
        "n_say_redactions": int(n.get("say_redactions", 0)),
        "n_human_multi_action": int(n.get("human_multi_action", 0)),
        "n_queue_played": int(n.get("queue_played", 0)),
        "n_queue_interrupted": int(n.get("queue_interrupted", 0)),
        "n_idle_textonly_turns": int(n.get("idle_textonly_turns", 0)),
        "n_human_mute_idle": int(n.get("human_mute_idle", 0)),
        "n_human_none": int(n.get("human_none", 0)),
        "n_bad_command": int(n.get("bad_command", 0)),
        "n_step_refused": int(n.get("step_refused", 0)),
        "action_effect_retries": int(n.get("action_effect_retries", 0)),
        "action_retry_recovered": int(n.get("action_retry_recovered", 0)),
        "action_retry_exhausted": int(n.get("action_retry_exhausted", 0)),
        "action_retry_unsafe_partial_state": int(
            n.get("action_retry_unsafe_partial_state", 0)
        ),
        "action_retry_unattributed_failure": int(
            n.get("action_retry_unattributed_failure", 0)
        ),
        "action_retry_transient_failure": int(
            n.get("action_retry_transient_failure", 0)
        ),
        "action_retry_human_command_rejection": int(
            n.get("action_retry_human_command_rejection", 0)
        ),
        "action_retry_ambiguous_exception": int(
            n.get("action_retry_ambiguous_exception", 0)
        ),
        "action_retry_invalid": int(n.get("action_retry_invalid", 0)),
        # a [Walk] that moved nothing because the person was already there (see arrival_text). NOT a
        # failure -- but a run with a high count is a run losing ticks to a loop, so it must be visible.
        "n_walk_no_motion": int(n.get("walk_no_motion", 0)),
        "assistant_empty": int(n.get("assistant_empty", 0)),
        "mode_silence": int(n.get("mode_silence", 0)),
        "mode_interrupt": int(n.get("mode_interrupt", 0)),
        "mode_errordetection": int(n.get("mode_errordetection", 0)),
        "mode_untokenised": int(n.get("mode_untokenised", 0)),
        "assistant_done_deferred": int(n.get("assistant_done_deferred", 0)),
        # ⭐ DECLARE-DONE. `n_ticks` is a signal in its own right -- how long the assistant took to
        # commit -- so the three facts that make it readable travel with it.
        "mode_done": int(n.get("mode_done", 0)),
        "declared_at_tick": n.get("declared_at_tick"),
        "never_declared": bool(n.get("never_declared", 0)),
        "first_goal_tick": n.get("first_goal_tick"),
        # Detection-only: a scored object was inside a container that is not its answer at some tick.
        "wrong_placements": int(n.get("wrong_placements", 0)),
        "wrong_placement_pairs": list(n.get("wrong_placement_pairs") or []),
        "cue_stands_lost": int(n.get("cue_stands_lost", 0)),
        "score_baseline_captured": bool(n.get("score_baseline_captured", 0)),
        "score_baseline_tick": n.get("score_baseline_tick"),
        "prologue_cue_intervals": int(n.get("prologue_cue_intervals", 0)),
        "prologue_cue_holds_pinned": int(n.get("prologue_cue_holds_pinned", 0)),
        "watch_preassistant_recovery_policy_version": int(
            n.get("watch_preassistant_recovery_policy_version", 0)
        ),
        "watch_action_retry_attempts": int(n.get("watch_action_retry_attempts", 0)),
        "watch_preassistant_recovery_allowance": int(
            n.get("watch_preassistant_recovery_allowance", 0)
        ),
        "watch_preassistant_recovery_chains_started": int(
            n.get("watch_preassistant_recovery_chains_started", 0)
        ),
        "watch_preassistant_recovery_chains_recovered": int(
            n.get("watch_preassistant_recovery_chains_recovered", 0)
        ),
        "watch_preassistant_recovery_credited_ticks": int(
            n.get("watch_preassistant_recovery_credited_ticks", 0)
        ),
        "watch_preassistant_recovery_unrecovered_chains": int(
            n.get("watch_preassistant_recovery_unrecovered_chains", 0)
        ),
        "watch_preassistant_recovery_allowance_exceeded": int(
            n.get("watch_preassistant_recovery_allowance_exceeded", 0)
        ),
        "watch_preassistant_recovery_interrupted_by_assistant": int(
            n.get("watch_preassistant_recovery_interrupted_by_assistant", 0)
        ),
        "watch_preassistant_recovery_human_deviations": int(
            n.get("watch_preassistant_recovery_human_deviations", 0)
        ),
        "watch_preassistant_recovery_human_speech": int(
            n.get("watch_preassistant_recovery_human_speech", 0)
        ),
        "watch_preassistant_recovery_stateful_detours": int(
            n.get("watch_preassistant_recovery_stateful_detours", 0)
        ),
        "watch_preassistant_recovery_observation_context_unsupported": int(
            n.get("watch_preassistant_recovery_observation_context_unsupported", 0)
        ),
        "watch_preassistant_recovery_context_restores": int(
            n.get("watch_preassistant_recovery_context_restores", 0)
        ),
        "human_http_fail": http_fail,
        "assistant_backend_stats": bstats,
        "stop_reason": proto._stop_reason,
        "eval_valid": (
            http_fail == 0
            and int(bstats.get("http_fail", 0)) == 0
            and recovery_invalid == 0
            and human_adherence_invalid == 0
            and int(n.get("action_retry_invalid", 0)) == 0
        ),
    }


def verdict_from_manifest(out_dir: str, tag: str) -> dict:
    """⛔ The episode is judged HERE, from the manifest on disk -- never from an exit code."""
    path = os.path.join(out_dir, f"{tag}_manifest.json")
    try:
        with open(path) as fh:
            man = json.load(fh)
    except (OSError, ValueError) as exc:
        return {"build": "NO_MANIFEST", "why": f"{type(exc).__name__}: {exc}"}
    tg = man.get("task_goal") or {}
    negs = list(tg.get("neg") or [])
    return {
        "build": "OK",
        "code_id": man.get("code_id"),
        "episode_valid": bool(man.get("episode_valid")),
        "goal_scored": bool(man.get("goal_scored")),
        "eval_valid": bool(man.get("eval_valid")),
        "benchmark_usable": bool(man.get("benchmark_usable")),
        "goal_met": bool(tg.get("met")),
        "progress": tg.get("progress"),
        "n_sub_goals": tg.get("n_sub_goals"),
        "n_sub_goals_ok": tg.get("n_sub_goals_ok"),
        "fired_negs": [n.get("label") for n in negs if not n.get("ok")],
        "dialogue_log": man.get("dialogue_log"),
        "manifest": path,
    }


# ---------------------------------------------------------------------------------------------------
#  driver
# ---------------------------------------------------------------------------------------------------


def _open_or_switch(env: Any, scene: str, args: argparse.Namespace) -> tuple[Any, dict]:
    """First task boots the app; every later task reuses it (``switch_scene``, ~25-73 s)."""
    from simulator.environment.isaac_env import IsaacEnv

    if env is None:
        env = IsaacEnv(
            scene, seed=int(args.seed), render_obs=True, active_gpu=int(args.active_gpu)
        )
        return env, env.open()
    return env, env.switch_scene(scene, int(args.seed))


# ---------------------------------------------------------------------------------------------------
#  regression: a malformed command from the person must cost ONE TICK, never the episode
# ---------------------------------------------------------------------------------------------------

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fail-closed executor for the authored ``human-policy/1.0`` beat FSM."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn


SemanticVerifier = Callable[[object], Mapping[str, object] | None]
WorldPredicate = Callable[[str, str, str | None], bool]

_POLICY_FIELDS = frozenset(
    {
        "version",
        "initial_state",
        "default_plan",
        "intervention_window",
        "transitions",
        "terminal_beats",
    }
)
_INITIAL_STATE_FIELDS = frozenset({"facing", "held", "posture"})
_INTERVENTION_WINDOW_FIELDS = frozenset({"after_beat", "before_beat"})
_TRANSITION_FIELDS = frozenset({"after_beat", "condition", "next_beat"})
_SEMANTIC_TRANSITION_FIELDS = _TRANSITION_FIELDS | {"intervention_beat"}
_WORLD_CHECK_FIELDS = frozenset({"pred", "subject", "target", "truth"})
_SEMANTIC_FIELDS = frozenset(
    {
        "kind",
        "expected_act",
        "expected_referents",
        "required_fact_ids",
        "world_check",
        "verifier",
    }
)
_CONDITION_FIELDS = {
    "timeout": frozenset({"kind", "after_ticks"}),
    "world_state": frozenset({"kind", *_WORLD_CHECK_FIELDS}),
    "semantic_and_world": _SEMANTIC_FIELDS,
}
_SEMANTIC_RESULT_FIELDS = frozenset({"intent", "referents", "facts"})
_SEMANTIC_ACTS = frozenset({"confirm", "correct", "halt", "redirect"})


@dataclass(frozen=True)
class BlockedError:
    """Machine-readable reason why the FSM refused to proceed."""

    code: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "status": "BLOCKED",
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


class HumanPolicyBlocked(ValueError):
    """Raised before any state change when authored policy cannot be trusted."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.error = BlockedError(code=code, path=path, message=message)
        super().__init__(f"{code}@{path}: {message}")

    def as_dict(self) -> dict[str, str]:
        return self.error.as_dict()


@dataclass(frozen=True)
class AdvanceResult:
    """Result of consuming one authored beat."""

    status: str
    completed_beat_id: str
    next_beat_id: str | None
    matched_condition: str | None
    consumed_intervention_beat: Mapping[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "completed_beat_id": self.completed_beat_id,
            "next_beat_id": self.next_beat_id,
            "matched_condition": self.matched_condition,
            "consumed_intervention_beat": _clone(self.consumed_intervention_beat),
        }


@dataclass(frozen=True)
class _Transition:
    after_beat: str
    condition: Mapping[str, object]
    next_beat: str | None
    intervention_beat: str | None
    path: str


@dataclass(frozen=True)
class _SemanticResult:
    intent: str
    referents: frozenset[str]
    facts: frozenset[str]


class HumanPolicyRuntime:
    """Deterministic beat FSM backed only by frozen TaskPack story and policy data."""

    def __init__(
        self,
        frozen_taskpack: Mapping[str, object],
        *,
        semantic_verifier: SemanticVerifier | None = None,
        world_predicate: WorldPredicate | None = None,
    ) -> None:
        story = _mapping(frozen_taskpack.get("story"), "$.story")
        runtime = _mapping(frozen_taskpack.get("runtime"), "$.runtime")
        self._beats, fact_ids = _parse_beats(story.get("beats"))
        policy = _mapping(runtime.get("human_policy"), "$.runtime.human_policy")
        _exact_fields(policy, _POLICY_FIELDS, "$.runtime.human_policy")
        _validate_policy_header(policy, self._beats)

        self._default_plan = _beat_id_list(
            policy.get("default_plan"),
            "$.runtime.human_policy.default_plan",
            self._beats,
        )
        self._terminal_beats = frozenset(
            _beat_id_list(
                policy.get("terminal_beats"),
                "$.runtime.human_policy.terminal_beats",
                self._beats,
            )
        )
        self._transitions = _parse_transitions(
            policy.get("transitions"), self._beats, fact_ids
        )
        _validate_human_beat_refs(
            policy,
            self._beats,
            self._default_plan,
            self._terminal_beats,
            self._transitions,
        )
        self._by_after = _group_transitions(self._transitions)
        self._default_successor = dict(zip(self._default_plan, self._default_plan[1:]))
        _validate_graph(
            self._beats,
            self._default_plan,
            self._default_successor,
            self._by_after,
            self._terminal_beats,
        )
        _validate_dependencies(self._transitions, semantic_verifier, world_predicate)

        self._semantic_verifier = semantic_verifier
        self._world_predicate = world_predicate
        self._next_beat_id: str | None = self._default_plan[0]
        self._complete = False
        self._waiting = False

    @property
    def next_beat_id(self) -> str | None:
        return self._next_beat_id

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def is_waiting(self) -> bool:
        return self._waiting

    def is_terminal_beat(self, beat_id: str) -> bool:
        return beat_id in self._terminal_beats

    def next_beat(self) -> dict[str, object] | None:
        """Return an isolated copy of the next authored beat, without interpreting it."""
        if self._next_beat_id is None or self._waiting:
            return None
        return _clone(self._beats[self._next_beat_id])

    def advance(
        self,
        completed_beat_id: str,
        *,
        elapsed_ticks: int = 0,
        semantic_input: object | None = None,
    ) -> AdvanceResult:
        """Consume the expected beat and select exactly one safe successor."""
        self._validate_advance(completed_beat_id, elapsed_ticks)
        if completed_beat_id in self._terminal_beats:
            return self._commit(completed_beat_id, None, None, None)

        transitions = self._by_after.get(completed_beat_id, ())
        semantic = self._semantic_snapshot(transitions, semantic_input)
        world_cache: dict[tuple[str, str, str | None], bool] = {}
        matches = [
            transition
            for transition in transitions
            if self._matches(transition.condition, elapsed_ticks, semantic, world_cache)
        ]
        if len(matches) > 1:
            _blocked(
                "AMBIGUOUS_TRANSITION",
                f"$.runtime.human_policy.transitions[after={completed_beat_id!r}]",
                "more than one condition matched the same completed beat",
            )
        if matches:
            selected = matches[0]
            kind = str(selected.condition["kind"])
            return self._commit(
                completed_beat_id,
                selected.next_beat,
                kind,
                selected.intervention_beat,
            )

        waiting_on_timeout = any(
            transition.condition["kind"] == "timeout"
            and elapsed_ticks < int(transition.condition["after_ticks"])
            for transition in transitions
        )
        if waiting_on_timeout:
            self._waiting = True
            return AdvanceResult(
                status="WAITING",
                completed_beat_id=completed_beat_id,
                next_beat_id=None,
                matched_condition=None,
                consumed_intervention_beat=None,
            )

        successor = self._default_successor.get(completed_beat_id)
        if successor is None:
            _blocked(
                "NO_TRANSITION",
                f"$.story.beats[id={completed_beat_id!r}]",
                "no condition matched and the default plan has no successor",
            )
        return self._commit(completed_beat_id, successor, None, None)

    def _validate_advance(self, completed_beat_id: str, elapsed_ticks: int) -> None:
        if self._complete or self._next_beat_id is None:
            _blocked("FSM_COMPLETE", "$", "the human policy already completed")
        if completed_beat_id != self._next_beat_id:
            _blocked(
                "UNEXPECTED_BEAT",
                "$.story.beats",
                f"expected {self._next_beat_id!r}, got {completed_beat_id!r}",
            )
        if (
            not isinstance(elapsed_ticks, int)
            or isinstance(elapsed_ticks, bool)
            or elapsed_ticks < 0
        ):
            _blocked(
                "INVALID_TICK_COUNT",
                "$.runtime.elapsed_ticks",
                "must be a non-negative integer",
            )

    def _semantic_snapshot(
        self,
        transitions: Sequence[_Transition],
        semantic_input: object | None,
    ) -> _SemanticResult | None:
        needs_semantics = any(
            row.condition["kind"] == "semantic_and_world" for row in transitions
        )
        if not needs_semantics or semantic_input is None:
            return None
        verifier = self._semantic_verifier
        if verifier is None:
            _blocked(
                "SEMANTIC_VERIFIER_MISSING",
                "$.runtime.human_policy",
                "semantic_and_world requires a semantic verifier",
            )
        try:
            raw = verifier(semantic_input)
        except HumanPolicyBlocked:
            raise
        except Exception as exc:
            _blocked(
                "SEMANTIC_VERIFIER_FAILED",
                "$.runtime.semantic_verifier",
                f"{type(exc).__name__}: {exc}",
            )
        if raw is None:
            return None
        return _parse_semantic_result(raw)

    def _matches(
        self,
        condition: Mapping[str, object],
        elapsed_ticks: int,
        semantic: _SemanticResult | None,
        world_cache: dict[tuple[str, str, str | None], bool],
    ) -> bool:
        kind = condition["kind"]
        if kind == "timeout":
            return elapsed_ticks >= int(condition["after_ticks"])
        if kind == "world_state":
            return self._world_matches(condition, world_cache)
        if semantic is None or not _semantic_matches(condition, semantic):
            return False
        world_check = _mapping(condition["world_check"], "world_check")
        return self._world_matches(world_check, world_cache)

    def _world_matches(
        self,
        check: Mapping[str, object],
        cache: dict[tuple[str, str, str | None], bool],
    ) -> bool:
        pred = str(check["pred"])
        subject = str(check["subject"])
        target_value = check["target"]
        target = None if target_value is None else str(target_value)
        key = (pred, subject, target)
        if key not in cache:
            callback = self._world_predicate
            if callback is None:
                _blocked(
                    "WORLD_PREDICATE_MISSING",
                    "$.runtime.human_policy",
                    "world conditions require a world predicate callback",
                )
            try:
                result = callback(pred, subject, target)
            except HumanPolicyBlocked:
                raise
            except Exception as exc:
                _blocked(
                    "WORLD_PREDICATE_FAILED",
                    "$.runtime.world_predicate",
                    f"{type(exc).__name__}: {exc}",
                )
            if type(result) is not bool:
                _blocked(
                    "WORLD_PREDICATE_INVALID",
                    "$.runtime.world_predicate",
                    f"expected bool, got {type(result).__name__}",
                )
            cache[key] = result
        return cache[key] is check["truth"]

    def _commit(
        self,
        completed_beat_id: str,
        next_beat_id: str | None,
        matched_condition: str | None,
        intervention_beat_id: str | None,
    ) -> AdvanceResult:
        self._next_beat_id = next_beat_id
        self._complete = next_beat_id is None
        self._waiting = False
        return AdvanceResult(
            status="COMPLETE" if self._complete else "READY",
            completed_beat_id=completed_beat_id,
            next_beat_id=next_beat_id,
            matched_condition=matched_condition,
            consumed_intervention_beat=(
                _clone(self._beats[intervention_beat_id])
                if intervention_beat_id is not None
                else None
            ),
        )


def _validate_policy_header(
    policy: Mapping[str, object], beats: Mapping[str, Mapping[str, object]]
) -> None:
    if policy.get("version") != "human-policy/1.0":
        _blocked(
            "UNSUPPORTED_VERSION",
            "$.runtime.human_policy.version",
            "expected 'human-policy/1.0'",
        )
    initial = _mapping(
        policy.get("initial_state"), "$.runtime.human_policy.initial_state"
    )
    _exact_fields(
        initial, _INITIAL_STATE_FIELDS, "$.runtime.human_policy.initial_state"
    )
    _optional_text(initial.get("facing"), "$.runtime.human_policy.initial_state.facing")
    _text_list(initial.get("held"), "$.runtime.human_policy.initial_state.held")
    if initial.get("posture") not in {"sitting", "squatting", "standing"}:
        _blocked(
            "INVALID_POSTURE",
            "$.runtime.human_policy.initial_state.posture",
            "expected sitting, squatting, or standing",
        )
    window = _mapping(
        policy.get("intervention_window"),
        "$.runtime.human_policy.intervention_window",
    )
    _exact_fields(
        window,
        _INTERVENTION_WINDOW_FIELDS,
        "$.runtime.human_policy.intervention_window",
    )
    for field in sorted(_INTERVENTION_WINDOW_FIELDS):
        _beat_ref(
            window.get(field),
            f"$.runtime.human_policy.intervention_window.{field}",
            beats,
            nullable=True,
        )


def _parse_beats(
    value: object,
) -> tuple[dict[str, Mapping[str, object]], frozenset[str]]:
    rows = _sequence(value, "$.story.beats")
    if not rows:
        _blocked("EMPTY_BEATS", "$.story.beats", "must contain at least one beat")
    beats: dict[str, Mapping[str, object]] = {}
    facts: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"$.story.beats[{index}]"
        beat = _mapping(raw, path)
        beat_id = _text(beat.get("id"), f"{path}.id")
        if beat_id in beats:
            _blocked("DUPLICATE_BEAT_ID", f"{path}.id", beat_id)
        beats[beat_id] = _clone(beat)
        facts.update(_established_fact_ids(beat.get("establishes"), path))
    return beats, frozenset(facts)


def _established_fact_ids(value: object, beat_path: str) -> set[str]:
    if value is None:
        return set()
    facts: set[str] = set()
    for index, raw in enumerate(_sequence(value, f"{beat_path}.establishes")):
        path = f"{beat_path}.establishes[{index}]"
        if isinstance(raw, str):
            facts.add(_text(raw, path))
            continue
        row = _mapping(raw, path)
        facts.add(_text(row.get("id"), f"{path}.id"))
    return facts


def _parse_transitions(
    value: object,
    beats: Mapping[str, Mapping[str, object]],
    fact_ids: frozenset[str],
) -> tuple[_Transition, ...]:
    rows = _sequence(value, "$.runtime.human_policy.transitions")
    if not rows:
        _blocked(
            "EMPTY_TRANSITIONS",
            "$.runtime.human_policy.transitions",
            "must contain at least one transition",
        )
    result: list[_Transition] = []
    for index, raw in enumerate(rows):
        path = f"$.runtime.human_policy.transitions[{index}]"
        row = _mapping(raw, path)
        condition = _parse_condition(
            row.get("condition"), f"{path}.condition", fact_ids
        )
        semantic = condition["kind"] == "semantic_and_world"
        _exact_fields(
            row,
            frozenset(_SEMANTIC_TRANSITION_FIELDS if semantic else _TRANSITION_FIELDS),
            path,
        )
        after = _beat_ref(row.get("after_beat"), f"{path}.after_beat", beats)
        next_beat = _beat_ref(
            row.get("next_beat"), f"{path}.next_beat", beats, nullable=True
        )
        intervention_beat = _beat_ref(
            row.get("intervention_beat"),
            f"{path}.intervention_beat",
            beats,
            nullable=not semantic,
        )
        result.append(_Transition(after, condition, next_beat, intervention_beat, path))
    return tuple(result)


def _parse_condition(
    value: object, path: str, fact_ids: frozenset[str]
) -> Mapping[str, object]:
    condition = _mapping(value, path)
    kind = condition.get("kind")
    if kind not in _CONDITION_FIELDS:
        _blocked(
            "UNSUPPORTED_CONDITION",
            f"{path}.kind",
            "expected timeout, world_state, or semantic_and_world",
        )
    _exact_fields(condition, _CONDITION_FIELDS[str(kind)], path)
    if kind == "timeout":
        ticks = condition.get("after_ticks")
        if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 1:
            _blocked(
                "INVALID_TIMEOUT", f"{path}.after_ticks", "must be a positive integer"
            )
    elif kind == "world_state":
        _validate_world_check(condition, path, has_kind=True)
    else:
        _validate_semantic_condition(condition, path, fact_ids)
    return _clone(condition)


def _validate_semantic_condition(
    condition: Mapping[str, object], path: str, fact_ids: frozenset[str]
) -> None:
    act = _text(condition.get("expected_act"), f"{path}.expected_act")
    if act not in _SEMANTIC_ACTS:
        _blocked(
            "INVALID_SEMANTIC_ACT",
            f"{path}.expected_act",
            f"expected one of {sorted(_SEMANTIC_ACTS)}",
        )
    referents = _text_list(
        condition.get("expected_referents"), f"{path}.expected_referents"
    )
    if not referents:
        _blocked(
            "EMPTY_EXPECTED_REFERENTS",
            f"{path}.expected_referents",
            "semantic intervention must bind at least one referent",
        )
    required = _text_list(
        condition.get("required_fact_ids"), f"{path}.required_fact_ids"
    )
    if not required:
        _blocked(
            "EMPTY_REQUIRED_FACTS",
            f"{path}.required_fact_ids",
            "semantic verification must cite at least one established fact",
        )
    for index, fact_id in enumerate(required):
        if fact_id not in fact_ids:
            _blocked(
                "DANGLING_FACT_REFERENCE",
                f"{path}.required_fact_ids[{index}]",
                fact_id,
            )
    if condition.get("verifier") != "human_model_then_world":
        _blocked(
            "INVALID_VERIFIER",
            f"{path}.verifier",
            "expected 'human_model_then_world'",
        )
    world = _mapping(condition.get("world_check"), f"{path}.world_check")
    _validate_world_check(world, f"{path}.world_check")


def _validate_world_check(
    check: Mapping[str, object], path: str, *, has_kind: bool = False
) -> None:
    fields = _WORLD_CHECK_FIELDS | ({"kind"} if has_kind else set())
    _exact_fields(check, frozenset(fields), path)
    _text(check.get("pred"), f"{path}.pred")
    _text(check.get("subject"), f"{path}.subject")
    _optional_text(check.get("target"), f"{path}.target")
    if type(check.get("truth")) is not bool:
        _blocked("INVALID_WORLD_CHECK", f"{path}.truth", "must be a boolean")


def _validate_graph(
    beats: Mapping[str, Mapping[str, object]],
    default_plan: tuple[str, ...],
    default_successor: Mapping[str, str],
    by_after: Mapping[str, Sequence[_Transition]],
    terminals: frozenset[str],
) -> None:
    if not terminals:
        _blocked(
            "EMPTY_TERMINALS",
            "$.runtime.human_policy.terminal_beats",
            "must contain at least one beat",
        )
    edges = _graph_edges(beats, default_successor, by_after, terminals)
    _reject_cycles(edges)
    reachable = _reachable(default_plan[0], edges)
    for terminal in sorted(terminals - reachable):
        _blocked(
            "UNREACHABLE_TERMINAL",
            "$.runtime.human_policy.terminal_beats",
            terminal,
        )
    for beat_id in sorted(reachable - terminals):
        has_exit = beat_id in default_successor or bool(by_after.get(beat_id))
        if not has_exit:
            _blocked(
                "DEAD_END",
                f"$.story.beats[id={beat_id!r}]",
                "reachable non-terminal beat has no successor",
            )


def _validate_human_beat_refs(
    policy: Mapping[str, object],
    beats: Mapping[str, Mapping[str, object]],
    default_plan: tuple[str, ...],
    terminals: frozenset[str],
    transitions: Sequence[_Transition],
) -> None:
    window = _mapping(
        policy["intervention_window"],
        "$.runtime.human_policy.intervention_window",
    )
    refs = set(default_plan) | set(terminals)
    refs.update(row.after_beat for row in transitions)
    refs.update(row.next_beat for row in transitions if row.next_beat is not None)
    refs.update(
        str(window[field])
        for field in _INTERVENTION_WINDOW_FIELDS
        if window[field] is not None
    )
    for beat_id in sorted(refs):
        beat = beats[beat_id]
        if beat.get("actor") != "human":
            _blocked(
                "NON_HUMAN_POLICY_BEAT",
                f"$.story.beats[id={beat_id!r}].actor",
                "human policy may reference only human beats",
            )
        if beat.get("phase") not in {"during", "after"}:
            _blocked(
                "NON_RUNTIME_POLICY_BEAT",
                f"$.story.beats[id={beat_id!r}].phase",
                "human policy beats must be in the during or after phase",
            )
    for transition in transitions:
        if transition.intervention_beat is None:
            continue
        beat = beats[transition.intervention_beat]
        path = f"{transition.path}.intervention_beat"
        if beat.get("actor") != "assistant":
            _blocked(
                "INVALID_INTERVENTION_BEAT",
                path,
                "must reference an assistant beat",
            )
        say = beat.get("say")
        if not isinstance(say, str) or not say.strip():
            _blocked(
                "INVALID_INTERVENTION_BEAT",
                path,
                "assistant intervention beat must contain non-empty say",
            )
        if beat.get("action") or beat.get("effects"):
            _blocked(
                "INVALID_INTERVENTION_BEAT",
                path,
                "assistant intervention beat cannot contain action or physical effects",
            )
        if beat.get("phase") not in {"during", "after"}:
            _blocked(
                "INVALID_INTERVENTION_BEAT",
                path,
                "assistant intervention beat must be in the during or after phase",
            )


def _graph_edges(
    beats: Mapping[str, Mapping[str, object]],
    default_successor: Mapping[str, str],
    by_after: Mapping[str, Sequence[_Transition]],
    terminals: frozenset[str],
) -> dict[str, set[str]]:
    edges = {beat_id: set() for beat_id in beats}
    for beat_id in beats:
        if beat_id in terminals:
            if beat_id in default_successor or by_after.get(beat_id):
                _blocked(
                    "TERMINAL_HAS_SUCCESSOR",
                    f"$.story.beats[id={beat_id!r}]",
                    "terminal beats cannot have outgoing policy edges",
                )
            continue
        successor = default_successor.get(beat_id)
        if successor is not None:
            edges[beat_id].add(successor)
        edges[beat_id].update(
            row.next_beat for row in by_after.get(beat_id, ()) if row.next_beat
        )
    return edges


def _reject_cycles(edges: Mapping[str, set[str]]) -> None:
    state: dict[str, int] = {}

    def visit(node: str) -> None:
        if state.get(node) == 1:
            _blocked("FSM_CYCLE", "$.runtime.human_policy", f"cycle includes {node!r}")
        if state.get(node) == 2:
            return
        state[node] = 1
        for child in sorted(edges[node]):
            visit(child)
        state[node] = 2

    for node in sorted(edges):
        visit(node)


def _reachable(start: str, edges: Mapping[str, set[str]]) -> frozenset[str]:
    seen: set[str] = set()
    pending = deque([start])
    while pending:
        node = pending.popleft()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(sorted(edges[node] - seen))
    return frozenset(seen)


def _validate_dependencies(
    transitions: Sequence[_Transition],
    semantic_verifier: SemanticVerifier | None,
    world_predicate: WorldPredicate | None,
) -> None:
    kinds = {str(row.condition["kind"]) for row in transitions}
    if "semantic_and_world" in kinds and semantic_verifier is None:
        _blocked(
            "SEMANTIC_VERIFIER_MISSING",
            "$.runtime.human_policy.transitions",
            "semantic_and_world requires an injected semantic verifier",
        )
    if kinds & {"semantic_and_world", "world_state"} and world_predicate is None:
        _blocked(
            "WORLD_PREDICATE_MISSING",
            "$.runtime.human_policy.transitions",
            "world conditions require an injected predicate callback",
        )


def _parse_semantic_result(raw: Mapping[str, object]) -> _SemanticResult:
    path = "$.runtime.semantic_verifier.result"
    result = _mapping(raw, path)
    _exact_fields(result, _SEMANTIC_RESULT_FIELDS, path)
    return _SemanticResult(
        intent=_text(result.get("intent"), f"{path}.intent"),
        referents=frozenset(_text_list(result.get("referents"), f"{path}.referents")),
        facts=frozenset(_text_list(result.get("facts"), f"{path}.facts")),
    )


def _semantic_matches(
    condition: Mapping[str, object], semantic: _SemanticResult
) -> bool:
    expected_referents = frozenset(
        str(value) for value in _sequence(condition["expected_referents"], "referents")
    )
    required_facts = frozenset(
        str(value) for value in _sequence(condition["required_fact_ids"], "facts")
    )
    return (
        semantic.intent == condition["expected_act"]
        and semantic.referents == expected_referents
        and required_facts <= semantic.facts
    )


def _group_transitions(
    transitions: Sequence[_Transition],
) -> dict[str, tuple[_Transition, ...]]:
    grouped: dict[str, list[_Transition]] = defaultdict(list)
    for transition in transitions:
        grouped[transition.after_beat].append(transition)
    return {key: tuple(value) for key, value in grouped.items()}


def _beat_id_list(
    value: object,
    path: str,
    beats: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    rows = _sequence(value, path)
    if not rows:
        _blocked("EMPTY_BEAT_LIST", path, "must contain at least one beat id")
    result = tuple(
        _beat_ref(row, f"{path}[{index}]", beats) for index, row in enumerate(rows)
    )
    if len(set(result)) != len(result):
        _blocked("DUPLICATE_BEAT_REFERENCE", path, "beat ids must be unique")
    return result


def _beat_ref(
    value: object,
    path: str,
    beats: Mapping[str, Mapping[str, object]],
    *,
    nullable: bool = False,
) -> str | None:
    if value is None and nullable:
        return None
    beat_id = _text(value, path)
    if beat_id not in beats:
        _blocked("DANGLING_BEAT_REFERENCE", path, beat_id)
    return beat_id


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _blocked("INVALID_TYPE", path, "expected an object")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _blocked("INVALID_TYPE", path, "expected an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _blocked("INVALID_TEXT", path, "expected a non-empty string")
    return value


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _text_list(value: object, path: str) -> tuple[str, ...]:
    rows = _sequence(value, path)
    result = tuple(_text(row, f"{path}[{index}]") for index, row in enumerate(rows))
    if len(set(result)) != len(result):
        _blocked("DUPLICATE_VALUE", path, "values must be unique")
    return result


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], path: str
) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    if unknown:
        _blocked("UNKNOWN_FIELD", f"{path}.{unknown[0]}", "field is not allowed")
    missing = sorted(expected - actual)
    if missing:
        _blocked("MISSING_FIELD", f"{path}.{missing[0]}", "field is required")


def _clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(child) for child in value)
    if isinstance(value, list):
        return [_clone(child) for child in value]
    return value


def _blocked(code: str, path: str, message: str) -> NoReturn:
    raise HumanPolicyBlocked(code, path, message)

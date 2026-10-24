# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Single capability registry shared by authoring checks and the simulator runtime."""

from __future__ import annotations

from dataclasses import dataclass


IMPLEMENTED = "IMPLEMENTED"
SIM_DEBT = "SIM_DEBT"
UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ActionCapability:
    primitive: str
    verbs: tuple[str, ...]
    arities: tuple[int, ...]
    executor: str
    affordances: tuple[str, ...] = ()
    status: str = IMPLEMENTED


@dataclass(frozen=True)
class PredicateCapability:
    predicate: str
    aliases: tuple[str, ...]
    arity: int
    reader: str
    runtime_scorable: bool = True


@dataclass(frozen=True)
class EventCapability:
    operator: str
    action_verb: str | None
    arity: int
    placement: str = "append_after"
    status: str = IMPLEMENTED


ACTIONS: tuple[ActionCapability, ...] = (
    ActionCapability(
        "navigate_to", ("WALK", "RUN", "WALKTOWARDS"), (1,), "produce_navigate"
    ),
    ActionCapability("pick", ("GRAB",), (1,), "produce_pick", ("graspable",)),
    ActionCapability(
        "flip", ("FLIP", "TURNOVER", "TURN_OVER"), (1,), "produce_flip", ("graspable",)
    ),
    ActionCapability(
        "place",
        ("PUT", "PUTBACK", "PUTOBJBACK", "PUTON"),
        (2,),
        "produce_place",
        ("surface",),
    ),
    ActionCapability(
        "place_inside",
        ("PUTIN", "PUTINSIDE", "INSERT"),
        (2, 3),
        "produce_place_inside",
        ("container",),
    ),
    ActionCapability(
        "attach",
        ("ATTACH", "ASSEMBLE", "INSTALL"),
        (4, 5),
        "produce_attach",
        ("assembly_part",),
    ),
    ActionCapability(
        "detach",
        ("DETACH", "DISASSEMBLE", "REMOVE"),
        (2,),
        "produce_detach",
        ("assembly_part",),
    ),
    ActionCapability(
        "raise",
        ("RAISE", "ERECT", "STANDUPRIGHT"),
        (1,),
        "produce_raise",
        ("assembly_part",),
    ),
    ActionCapability(
        "push",
        ("PUSH", "PUSHRACK", "PUSH_RACK"),
        (2,),
        "produce_push",
        ("pushable",),
    ),
    ActionCapability(
        "pull",
        ("PULL", "PULLRACK", "PULL_RACK"),
        (2,),
        "produce_pull",
        ("pullable",),
    ),
    ActionCapability("release", ("DROP", "RELEASE"), (0,), "produce_release"),
    ActionCapability(
        "fall_to_floor",
        ("FALLTOFLOOR", "FALL_TO_FLOOR"),
        (2,),
        "produce_fall_to_floor",
    ),
    ActionCapability(
        "place_on_floor",
        ("PUTONFLOOR", "PUT_ON_FLOOR"),
        (2,),
        "produce_place_on_floor",
    ),
    ActionCapability("open", ("OPEN",), (1,), "produce_open", ("openable",)),
    ActionCapability("close", ("CLOSE",), (1,), "produce_close", ("openable",)),
    ActionCapability(
        "open_door", ("OPENDOOR", "OPEN_DOOR"), (1,), "produce_open_door", ("openable",)
    ),
    ActionCapability(
        "close_door",
        ("CLOSEDOOR", "CLOSE_DOOR"),
        (1,),
        "produce_close_door",
        ("openable",),
    ),
    ActionCapability(
        "turn_on",
        ("TURNON", "TURN_ON", "SWITCHON", "SWITCH_ON"),
        (1,),
        "produce_turn_on",
        ("toggleable",),
    ),
    ActionCapability(
        "turn_off",
        ("TURNOFF", "TURN_OFF", "SWITCHOFF", "SWITCH_OFF"),
        (1,),
        "produce_turn_off",
        ("toggleable",),
    ),
    ActionCapability("sit", ("SIT",), (1,), "produce_sit", ("seat",)),
    ActionCapability("standup", ("STANDUP",), (0,), "produce_standup"),
    ActionCapability("cut", ("CUT",), (2,), "produce_cut"),
    ActionCapability("stir", ("STIR",), (2,), "produce_stir"),
    ActionCapability("pour", ("POUR",), (2,), "produce_pour"),
    ActionCapability("wipe", ("WIPE",), (2,), "produce_wipe"),
    ActionCapability("cook", ("COOK",), (1,), "produce_cook"),
    ActionCapability(
        "use_computer",
        ("USECOMPUTER", "USE_COMPUTER", "TYPE"),
        (1,),
        "produce_use_computer",
        ("graspable",),
    ),
    ActionCapability(
        "answer_phone",
        ("ANSWERPHONE", "ANSWER_PHONE"),
        (1,),
        "produce_answer_phone",
        ("graspable",),
    ),
    ActionCapability(
        "look_around", ("LOOK_AROUND", "LOOKAROUND"), (0,), "produce_look_around"
    ),
    ActionCapability("look_scan", ("LOOK_SCAN", "LOOKSCAN"), (0,), "produce_look_scan"),
    ActionCapability("look_at", ("LOOK_AT", "LOOKAT"), (1,), "produce_look_at"),
    ActionCapability("squat", ("SQUAT", "SQUATDOWN"), (0,), "produce_squat"),
    ActionCapability("confirm", ("CONFIRM",), (0,), "produce_policy_halt"),
    ActionCapability("wait", ("WAIT",), (0,), "produce_wait"),
    ActionCapability(
        "forward", ("FORWARD", "WALKFORWARD", "WALK_FORWARD"), (0,), "produce_forward"
    ),
    ActionCapability(
        "turn",
        (
            "TURNLEFT",
            "TURN_LEFT",
            "TURNRIGHT",
            "TURN_RIGHT",
            "TURNAROUND",
            "TURN_AROUND",
        ),
        (0,),
        "produce_turn",
    ),
)


PREDICATES: tuple[PredicateCapability, ...] = (
    PredicateCapability("on", ("on", "ontop"), 2, "relations"),
    PredicateCapability("inside", ("inside",), 2, "relations"),
    PredicateCapability("held", ("held",), 1, "relations"),
    PredicateCapability("open", ("open",), 1, "relations"),
    PredicateCapability("closed", ("closed",), 1, "absence_of_open"),
    PredicateCapability("off", ("off",), 1, "absence_of_running"),
    PredicateCapability("near", ("near",), 2, "relations"),
    PredicateCapability("sitting", ("sitting", "is_sat_on"), 1, "relations"),
    PredicateCapability("cut", ("cut",), 1, "relations"),
    PredicateCapability("cooked", ("cooked",), 1, "relations"),
    PredicateCapability("running", ("running",), 1, "relations"),
    PredicateCapability("attached", ("attached",), 2, "relations"),
    PredicateCapability("raised", ("raised", "is_raised"), 1, "relations"),
    PredicateCapability(
        "unmoved", ("unmoved", "is_unmoved", "remain_unmoved"), 1, "relations"
    ),
)


EVENTS: tuple[EventCapability, ...] = (
    EventCapability("turnon", "TurnOn", 1),
    EventCapability("turnoff", "TurnOff", 1),
    EventCapability("pushrack", "Push", 2, placement="replace_trigger"),
    EventCapability("dissolvetext", None, 1, status=SIM_DEBT),
    EventCapability("occlude", None, 1, status=SIM_DEBT),
    EventCapability("turnfacedown", None, 1, status=SIM_DEBT),
    EventCapability("turnblack", None, 1, status=SIM_DEBT),
    EventCapability("smearink", None, 1, status=SIM_DEBT),
    EventCapability("showhardwareerror", None, 1, status=SIM_DEBT),
    EventCapability("knockoff", None, 1, status=SIM_DEBT),
    EventCapability("shiftobjects", None, 1, status=SIM_DEBT),
    EventCapability("deploypartition", None, 1, status=SIM_DEBT),
    EventCapability("navmeshupdate", None, 0, status=SIM_DEBT),
    EventCapability("physicsbounceandscramble", None, 1, status=SIM_DEBT),
    EventCapability("lockdoor", None, 1, status=SIM_DEBT),
    EventCapability("jamdoor", None, 1, status=SIM_DEBT),
    EventCapability("ring", None, 1, status=SIM_DEBT),
    EventCapability("becomeoccupied", None, 1, status=SIM_DEBT),
    EventCapability("enforceactionlimit", None, 1, status=SIM_DEBT),
)


def normalize(value: object) -> str:
    return "".join(char for char in str(value or "").lower() if char.isalnum())


def action_by_verb(*, include_unavailable: bool = False) -> dict[str, ActionCapability]:
    return {
        verb: capability
        for capability in ACTIONS
        if include_unavailable or capability.status == IMPLEMENTED
        for verb in capability.verbs
    }


def action_by_primitive(
    *, include_unavailable: bool = False
) -> dict[str, ActionCapability]:
    return {
        capability.primitive: capability
        for capability in ACTIONS
        if include_unavailable or capability.status == IMPLEMENTED
    }


def predicate_by_name() -> dict[str, PredicateCapability]:
    return {
        normalize(alias): capability
        for capability in PREDICATES
        for alias in capability.aliases
    }


def event_by_operator() -> dict[str, EventCapability]:
    return {normalize(capability.operator): capability for capability in EVENTS}


def canonical_predicate(value: object) -> str | None:
    capability = predicate_by_name().get(normalize(value))
    return capability.predicate if capability is not None else None

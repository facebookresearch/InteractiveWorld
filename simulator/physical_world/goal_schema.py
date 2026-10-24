"""Structured-goal vocabulary + error taxonomy for the arbitrary-human-motion controller.

The RL action / benchmark command is a STRUCTURED GOAL (a primitive + args), never a torque and
never a replayed clip. The primitive vocabulary is COPIED (not imported) from OmniGibson's
BaseActionPrimitiveSet so the schema stays aligned with BEHAVIOR while the training/motor core keeps
ZERO dependency on the (uninstalled, non-commercial) OmniGibson/BDDL stack. See
[[isaac-arbitrary-motion-architecture]].

numpy-only, no Isaac / torch at module scope -> importable anywhere (pre-launch, offline tests).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Primitive(enum.Enum):
    """The motor primitives the HumanController can execute. Vocabulary mirrors OmniGibson."""

    NAVIGATE_TO = (
        "navigate_to"  # walk to an arbitrary world (x, y[, yaw]) with a striding gait
    )
    PICK = "pick"  # reach + grasp an object prim, weld it to the hand
    FLIP = "flip"  # rotate the held object face-down while keeping it in the hand
    PLACE = "place"  # reach to a target pose, release, settle the object on a surface
    PLACE_INSIDE = (
        "place_inside"  # reach INTO a container + release the object into its interior
    )
    ATTACH = "attach"  # mate a held part's Male attach-point onto a target part's Female at the data-defined 6-DOF pose
    DETACH = "detach"  # grasp an attached leaf part and remove its declared mate so it can be re-attached
    RAISE = "raise"  # tilt a FLAT-assembled unit upright about its base edge (rigid; carries all attached parts + labels)
    PUSH = "push"  # move a pushable fixture to a named destination while its contents ride with it
    PULL = "pull"  # move a pullable fixture to a named destination while its contents ride with it
    OPEN = "open"  # drive an articulated object's joint toward its open limit (phase 3)
    CLOSE = (
        "close"  # drive an articulated object's joint toward its closed limit (phase 3)
    )
    OPENDOOR = (
        "open_door"  # swing a revolute door leaf toward its open limit (about a hinge)
    )
    CLOSEDOOR = "close_door"  # swing a revolute door leaf back to its closed limit
    TURN_ON = "turn_on"  # press a toggleable device's control and enter RUNNING state
    TURN_OFF = "turn_off"  # press the control again and leave RUNNING state
    RELEASE = "release"  # let go of the held object in place
    FALL_TO_FLOOR = "fall_to_floor"  # deterministic off-camera transition from a support to its adjacent floor
    PLACE_ON_FLOOR = (
        "place_on_floor"  # crouch and place the held object beside a named anchor
    )
    SIT = "sit"  # lower the body onto a seat surface (BODY_ANCHORED coupling)
    STANDUP = "standup"  # rise from a seat back to standing
    # Step-4 CONTACT verbs (tool-mediated, symbolic-state effect) -- see the contact-action engine.
    # 2-arg grammar [Verb] <target> <tool>: stand at <target>, act with the already-held <tool>.
    CUT = "cut"  # oscillating strokes over a target -> CUT(target)
    STIR = (
        "stir"  # circular strokes in a container -> (rendered motion; optional state)
    )
    POUR = "pour"  # tilt the held source over a target -> INSIDE(contents, target)
    WIPE = "wipe"  # raster strokes over a surface -> (clean; rendered motion)
    COOK = "cook"  # 1-arg: dwell at an appliance -> COOKED(target)
    USE_COMPUTER = "use_computer"  # 1-arg: type at a keyboard; no symbolic-state effect
    ANSWER_PHONE = "answer_phone"  # 1-arg: raise a held phone to the ear
    # Step-4 CAMERA / body verbs (coupling NONE; they exist to MOVE the egocentric camera).
    LOOK_AROUND = "look_around"  # 0-arg: sweep the gaze/heading in place
    LOOK_SCAN = "look_scan"  # 0-arg: same sweep, gaze pitched DOWN to take in the floor
    LOOK_AT = "look_at"  # 1-arg: aim the gaze at an entity
    SQUAT = "squat"  # 0-arg: lower the body (and the eye) in a crouch
    CONFIRM = "confirm"
    WAIT = "wait"
    # EGOCENTRIC LOCOMOTION (the only verbs with NO target entity). Everything else names a thing and
    # lets the planner find the path; these two are expressed purely in the BODY frame, so they can be
    # issued to a human who cannot see the room and therefore cannot name anything in it.
    FORWARD = "forward"  # stride along the CURRENT facing by meta['dist_m'] metres
    TURN = "turn"  # rotate in place by meta['turn_deg'] degrees (+ = left/CCW)


# Locomotion defaults, shared by the grammar (token -> Goal) and the producers (Goal -> frames) so a
# hand-built Goal and a parsed command mean the same thing.
FORWARD_STEP_M = 0.6  # one bare [Forward] == one human pace
TURN_DEG = 90.0  # a bare [TurnLeft] / [TurnRight] is a quarter turn
TURN_AROUND_DEG = 180.0  # [TurnAround] is fixed; it takes no argument
WAIT_DEFAULT_S = 3.0
WAIT_MIN_S = 0.1
WAIT_MAX_S = 600.0


class PrimitiveError(enum.Enum):
    """Failure modes a primitive reports instead of silently faking success (mirrors OmniGibson)."""

    OK = "ok"
    OUT_OF_REACH = (
        "out_of_reach"  # warp residual too large -> target outside the arm envelope
    )
    GRASP_FAILED = "grasp_failed"  # too few fingers contacted / hand-object penetration
    PLACE_UNSTABLE = "place_unstable"  # object did not settle at rest on a surface
    NO_FREE_PLACEMENT = "no_free_placement"  # target has no collision-free supported site for the held object
    TARGET_TOO_SMALL = "target_too_small"  # held object's measured footprint exceeds the target footprint
    UNSTABLE_AFTER_RELEASE = "unstable_after_release"  # settled, but the intended live ON/INSIDE relation is false
    PATH_BLOCKED = "path_blocked"  # navigation could not reach the goal
    NO_COLLISION_FREE_STANCE = (
        "no_collision_free_stance"  # no reachable stand avoids body/arm collision
    )
    TRAJECTORY_COLLISION = "trajectory_collision"  # a manipulation approach or retreat crosses solid geometry
    NO_OBJECT = "no_object"  # referenced object prim not found
    UNREADABLE_LABEL = (
        "unreadable_label"  # no safe camera/body pose resolves the required text
    )
    NOT_ARTICULATED = "not_articulated"  # OPEN/CLOSE target has no drivable joint
    PRECONDITION_UNMET = (
        "precondition_unmet"  # e.g. a CONTACT verb whose required tool is not held
    )
    POSTCONDITION_FAILED = "postcondition_failed"  # action returned success but its required runtime state is absent
    MISSING_AFFORDANCE = (
        "missing_affordance"  # target does not expose the action's required capability
    )


class PrimitiveFailure(Exception):
    """Raised by a primitive when it cannot complete its goal (carries the structured reason)."""

    def __init__(self, reason: PrimitiveError, detail: str = "") -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class Goal:
    """One structured command to the controller.

    prim: which primitive. The rest are the args each primitive reads (only the relevant ones are set):
      target_xy   -> NAVIGATE_TO destination (meters, world XY).
      target_yaw  -> optional final facing (deg, pose_body convention); None = face the path tangent.
      target_key  -> NAVIGATE_TO UNRESOLVED target: the scene entity to stand-at, resolved to (target_xy,
                     target_yaw) at RUNTIME (nearest-reachable from the human's live position) rather than
                     at compile/init time. When set with target_xy=None, the runtime layer picks the stand.
      obj_key     -> PICK / OPEN / CLOSE / RELEASE object handle (a key into the scene's object table).
      place_xy    -> PLACE destination XY on the drop surface; place_z resolved from the surface top.
      place_z     -> optional explicit drop-surface top z (else taken from the scene surface).
      amount      -> OPEN/CLOSE fraction of the joint range (default fully open/close).
      meta        -> per-primitive extras; FORWARD reads meta['dist_m'] (metres, > 0) and TURN reads
                     meta['turn_deg'] (signed degrees, + = left/CCW). Both fall back to the module
                     defaults when absent, so a bare Goal(Primitive.FORWARD) is one default pace.
    """

    prim: Primitive
    target_xy: tuple[float, float] | None = None
    target_yaw: float | None = None
    target_key: str | None = None
    obj_key: str | None = None
    place_xy: tuple[float, float] | None = None
    place_z: float | None = None
    amount: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


def navigate_to(x: float, y: float, yaw: float | None = None) -> Goal:
    """Convenience builder: walk to world (x, y) with an optional final facing."""
    return Goal(Primitive.NAVIGATE_TO, target_xy=(float(x), float(y)), target_yaw=yaw)


def pick(obj_key: str) -> Goal:
    """Convenience builder: pick up the object registered under obj_key."""
    return Goal(Primitive.PICK, obj_key=obj_key)


def place(x: float, y: float, z: float | None = None) -> Goal:
    """Convenience builder: place the held object on the surface at (x, y[, z])."""
    return Goal(Primitive.PLACE, place_xy=(float(x), float(y)), place_z=z)


def forward(dist_m: float = FORWARD_STEP_M) -> Goal:
    """Convenience builder: stride ``dist_m`` metres along the body's current facing."""
    return Goal(Primitive.FORWARD, meta={"dist_m": float(dist_m)})


def turn(deg: float = TURN_DEG) -> Goal:
    """Convenience builder: rotate in place by ``deg`` degrees (positive = left / counter-clockwise)."""
    return Goal(Primitive.TURN, meta={"turn_deg": float(deg)})

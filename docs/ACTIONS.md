# Human actions

The human in InteractiveWorld is driven by short commands. A command names an action
and the objects it acts on; the simulator plans the rest: where to stand, a
collision-aware route there (opening closed doors on the way), the reach, the grip and
the first-person camera.

```text
[Verb] <object_key> (instance) <object_key> (instance) ...
```

* **Verb**: case-insensitive; `[PutInside]`, `[PUTINSIDE]` and `[Put_Inside]` are the same.
  Aliases are listed below.
* **Object key**: an entry of `observation.objects`. In a raw scene these are BEHAVIOR
  instance names such as `fridge_xyejdx_0`; in a TaskPack they are the task's own keys
  such as `item_2` or `surf_call`.
* **Instance**: `(1)` for the first object with that key. Keys are already unique, so it is
  almost always `(1)`.
* A **scalar** replaces the objects for egocentric movement and waiting:
  `[Forward] 1.5m`, `[Forward] 3 steps`, `[TurnLeft] 45`, `[Wait] 30s`.

Each step returns one outcome per primitive the command expands to (a manipulation first
walks to its target when needed). An outcome has `ok`, a `reason` code and a `detail`
message; `StepResult.success` is true when every outcome is `ok`.

```bash
./launch/simulator.sh capabilities          # this catalog as JSON
./launch/simulator.sh capabilities --all    # including documented but unavailable actions
```

## Catalog

The *Requires* column is the affordance the target must have; the simulator rejects a
command whose target lacks it.

### Moving

| Action | Command | Aliases | What happens |
| :--- | :--- | :--- | :--- |
| navigate_to | `[Walk] <target> (1)` | `Run`, `WalkTowards` | Plans an A* route to a reachable standing spot by the target and walks it; opens closed doors on the way; carries whatever is held |
| forward | `[Forward]` · `[Forward] 1.5m` · `[Forward] 3 steps` | `WalkForward`, `Walk_Forward` | Steps straight ahead along the current heading (default one 0.6 m pace); fails with `PATH_BLOCKED` instead of walking into an obstacle |
| turn | `[TurnLeft]` · `[TurnRight] 45` · `[TurnAround]` | `Turn_Left`, `Turn_Right`, `Turn_Around` | Turns in place (default 90°; `TurnAround` is always 180°) |
| squat | `[Squat]` | `SquatDown` | Crouches with the feet planted and rises again |
| sit | `[Sit] <seat> (1)` | | Walks to the seat and sits on it (*Requires*: seat) |
| standup | `[StandUp]` | | Rises from the current seat and steps forward |

### Looking

| Action | Command | Aliases | What happens |
| :--- | :--- | :--- | :--- |
| look_at | `[LookAt] <object> (1)` | `Look_At` | Looks at an object; a small object with an unreadable label is picked up, turned to the camera and put back |
| look_around | `[LookAround]` | `Look_Around` | Turns one full circle at eye level |
| look_scan | `[LookScan]` | `Look_Scan` | Sweeps a full circle with the gaze on the floor a few metres ahead |
| wait | `[Wait]` · `[Wait] 30s` | | Holds still (default 3 s, 0.1–600 s) |

### Hands

| Action | Command | Aliases | What happens |
| :--- | :--- | :--- | :--- |
| pick | `[Grab] <object> (1)` | | Walks to the object, chooses a grip from its shape (pinch, cylindrical wrap, round grip, handle) and lifts it (*Requires*: graspable) |
| place | `[Put] <object> (1) <surface> (1)` | `PutOn`, `PutBack`, `PutObjBack` | Sets the held object on a free spot of the surface (*Requires*: surface) |
| place_inside | `[PutInside] <object> (1) <container> (1)` | `PutIn`, `Insert` | Puts the held object into a container interior, opening it first if needed (*Requires*: container) |
| release | `[Drop]` | `Release` | Opens the hand where it is |
| flip | `[Flip] <object> (1)` | `TurnOver`, `Turn_Over` | Turns the held object face down without releasing it |
| place_on_floor | `[PutOnFloor] <object> (1) <anchor> (1)` | `Put_On_Floor` | Crouches and places the held object on the floor in front of the anchor |
| fall_to_floor | `[FallToFloor] <object> (1) <support> (1)` | `Fall_To_Floor` | Moves an object resting on an edge to the floor beside its support (used by task prologues) |
| push | `[Push] <object> (1) <target> (1)` | `PushRack`, `Push_Rack` | Pushes a movable object, such as a rack, toward a target (*Requires*: pushable) |
| pull | `[Pull] <object> (1) <target> (1)` | `PullRack`, `Pull_Rack` | Pulls a movable object toward a target (*Requires*: pullable) |

### Articulated objects and devices

| Action | Command | Aliases | What happens |
| :--- | :--- | :--- | :--- |
| open | `[Open] <object> (1)` | | Opens a cabinet, drawer, fridge or oven by its handle (*Requires*: openable) |
| close | `[Close] <object> (1)` | | Closes it again (*Requires*: openable) |
| open_door | `[OpenDoor] <door> (1)` | `Open_Door` | Opens a room door (*Requires*: openable) |
| close_door | `[CloseDoor] <door> (1)` | `Close_Door` | Closes a room door (*Requires*: openable) |
| turn_on | `[TurnOn] <device> (1)` | `Turn_On`, `SwitchOn`, `Switch_On` | Presses the device's control and marks it running (*Requires*: toggleable) |
| turn_off | `[TurnOff] <device> (1)` | `Turn_Off`, `SwitchOff`, `Switch_Off` | Presses the control again and clears running (*Requires*: toggleable) |

### Using objects

| Action | Command | Aliases | What happens |
| :--- | :--- | :--- | :--- |
| cut | `[Cut] <target> (1) <tool> (1)` | | Chops over the target with the held tool; the target becomes `cut` |
| stir | `[Stir] <target> (1) <tool> (1)` | | Circles the held tool inside the target container |
| pour | `[Pour] <target> (1) <tool> (1)` | | Brings the held source over the target's rim and pours; the source contents end up inside the target |
| wipe | `[Wipe] <target> (1) <tool> (1)` | | Picks up the tool if needed and wipes it across the target |
| cook | `[Cook] <food> (1)` | | Cooks the food at its active appliance; the food becomes `cooked` |
| use_computer | `[UseComputer] <device> (1)` | `Use_Computer`, `Type` | Sits at the workstation, types with both hands and watches the monitor (*Requires*: graspable device) |
| answer_phone | `[AnswerPhone] <phone> (1)` | `Answer_Phone` | Raises the held phone to the ear; follow with `[Wait]` for the call |
| confirm | `[Confirm]` | | Produces no motion; a human policy issues it to confirm and stop |

### Assembly

| Action | Command | Aliases | What happens |
| :--- | :--- | :--- | :--- |
| attach | `[Attach] <child> (1) <child_point> (0) <parent> (1) <parent_point> (0)` | `Assemble`, `Install` | Mates a connector of the held part onto a connector of the parent; 4 or 5 arguments (*Requires*: assembly_part) |
| detach | `[Detach] <child> (1) <parent> (1)` | `Disassemble`, `Remove` | Grasps an attached part and removes its mate |
| raise | `[Raise] <assembly_root> (1)` | `Erect`, `StandUpright` | Tilts a flat-assembled unit upright about its bottom edge |

## Outcome codes

A failed outcome carries one of these `reason` codes (`PrimitiveError` in
`simulator/physical_world/goal_schema.py`); `detail` explains the specific case.

| Reason | Meaning |
| :--- | :--- |
| `NO_OBJECT` | The referenced object is not in the world |
| `MISSING_AFFORDANCE` | The target lacks the capability the action requires |
| `PRECONDITION_UNMET` | A requirement is not met, for example a tool action without the tool in hand |
| `PATH_BLOCKED` | Navigation cannot reach the goal, or `[Forward]` would walk into an obstacle |
| `NO_COLLISION_FREE_STANCE` | No reachable place to stand near the target avoids collisions |
| `TRAJECTORY_COLLISION` | The reach or retreat would pass through solid geometry |
| `OUT_OF_REACH` | The target is outside the arm's reach from the chosen stance |
| `GRASP_FAILED` | Too few fingers make contact, or the hand would penetrate the object |
| `NO_FREE_PLACEMENT` / `TARGET_TOO_SMALL` | No collision-free spot on the surface fits the held object |
| `PLACE_UNSTABLE` / `UNSTABLE_AFTER_RELEASE` | The object did not settle in the requested `on`/`inside` relation |
| `NOT_ARTICULATED` | The object has no drivable joint in this world |
| `UNREADABLE_LABEL` | No camera pose makes the required text readable |
| `POSTCONDITION_FAILED` | The motion finished but the required state change is absent |

## State predicates

Task goals and `observation.relations` (debug profile) use these predicates over the
live world state:

| Predicate | Arity | Meaning |
| :--- | :---: | :--- |
| `on` (`ontop`) | 2 | Object rests on a surface |
| `inside` | 2 | Object is inside a container |
| `held` | 1 | Object is in the human's hand |
| `open` / `closed` | 1 | Articulated object is open / closed |
| `running` / `off` | 1 | Device is running / not running |
| `near` | 2 | Human or object is near a target |
| `sitting` (`is_sat_on`) | 1 | Seat is occupied |
| `cut`, `cooked` | 1 | Food was cut / cooked |
| `attached` | 2 | Parts are mated |
| `raised` | 1 | Assembly stands upright |
| `unmoved` | 1 | Object was not displaced |

The registry behind these tables is `simulator/physical_world/runtime_capabilities.py`.

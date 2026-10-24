# TaskPacks: the task format

Every InteractiveBench task is one **TaskPack**: a JSON file that fully determines the
world, the person's story, how the simulated Human reacts and what counts as success.
The 120 bundled TaskPacks live in [`benchmark/tasks/taskpacks/`](../benchmark/tasks/taskpacks)
and are pinned by SHA256 in [`benchmark/tasks/manifest.json`](../benchmark/tasks/manifest.json).
This guide explains the format and how to build, check and evaluate your own task.

## Workflow

1. **Start from a bundled task** whose scene and interaction pattern are close to yours
   (`./launch/simulator.sh tasks list`, `tasks show ID`).
2. **Edit** a copy outside the repository: give it a new `task_id`, then change the
   objects, goal, story and Human policy together (see the field reference below).
3. **Pin and validate** it: `sha256sum my_task.taskpack.json`, then load it with
   `TaskSpec.load(path, expected_sha256="sha256:…")`. Validation errors name the field.
4. **Replay the oracle** in the simulator; the score must be `success: true`.
5. **Evaluate models** on it with `launch/run_taskpack.sh`.

[`examples/05_derive_taskpack.py`](../examples/05_derive_taskpack.py) automates steps 1–3
for one concrete change: it takes `ASK-12802` (choose the snack that fits a calorie
budget, then carry it to the kitchen breakfast table) and moves the destination to the
living-room table, editing the goal, oracle, Human policy and story text consistently.

```bash
python3 examples/05_derive_taskpack.py /tmp/my_task.taskpack.json
# task_id  MY-ASK-12802-LIVINGROOM   sha256:<digest>
# scene    Beechwood_0_int   preparation prologue
# oracle   --action '[Grab] <item_2> (1)' --action '[Walk] <surf_call> (1)' --action '[Put] <item_2> (1) <surf_call> (1)'
```

### Replay the oracle

```bash
./launch/simulator.sh run --taskpack /tmp/my_task.taskpack.json \
    --taskpack-sha256 sha256:<digest> --prepare-task --gpu 0 \
    --action '[Grab] <item_2> (1)' --action '[Walk] <surf_call> (1)' \
    --action '[Put] <item_2> (1) <surf_call> (1)' > oracle.json
```

`--prepare-task` replays `story.prologue`; the final `"score"` must report
`"success": true`. While opening, the simulator validates the authored world (every object
placed, nothing overlapping, no goal satisfied in advance) and names the offending entry
when a check fails. Add `--strict-acceptance` for stricter authoring checks; leave it off
when you compare with benchmark behavior, which runs without it.

### Evaluate models on it

```bash
./launch/run_taskpack.sh --taskpack /tmp/my_task.taskpack.json \
    --taskpack-sha256 sha256:<digest> \
    --assistant model --assistant-provider openai-compatible \
    --assistant-model Qwen/Qwen3-VL-8B-Instruct --assistant-base-url http://127.0.0.1:8000/v1 \
    --human-provider openai-compatible \
    --human-model Qwen/Qwen3-VL-8B-Instruct --human-base-url http://127.0.0.1:8000/v1 \
    --active-gpu 0 --output-dir runs/my-task
```

`--assistant oracle-verbal` replaces the model with the task's scripted guidance, which is
a good check that the Human can finish your task at all. For several tasks, pass
`--taskpack-batch tasks.json` with

```json
{"tasks": [{"task_id": "MY-ASK-12802-LIVINGROOM",
            "taskpack_path": "/tmp/my_task.taskpack.json",
            "taskpack_sha256": "sha256:<digest>"}]}
```

The result JSON reports `native_taskpack_outcome.strict_success`; custom tasks have no
release rubric (`release_rubric_outcome.status` is `NOT_APPLICABLE`). A TaskPack that
reuses a Balanced120 `task_id` must be byte-identical to the frozen file.

Keep custom TaskPacks outside the repository: the benchmark runner verifies the release
tree before it runs.

## Top level

```json
{
  "taskpack_version": "data-generate-taskpack/1.0",
  "task_id": "MY-ASK-12802-LIVINGROOM",
  "executable_spec": { ... },   // world, goal, reference program    (bts/1.0)
  "story":           { ... },   // what the person did, says and needs
  "runtime":         { ... },   // how the episode runs and the Human reacts
  "lineage":         { ... }    // free-form provenance
}
```

`taskpack_version` and `runtime.contract_version` (`compiled-runtime-contract/1.8`) must
match the release exactly; `executable_spec.task_id` must equal `task_id`.

## `executable_spec`: world, goal and program

`spec_version` is `bts/1.0`. The simulator compiles this block into the physical world.

### `world`

| Field | Meaning |
| :--- | :--- |
| `scene` | One of `./launch/simulator.sh worlds`, e.g. `Beechwood_0_int` |
| `rooms` | Rooms that are part of the task, e.g. `["kitchen_0", "living_room_0"]` |
| `layout`, `instance`, `authoring_seed`, `seed_scope` | Determinism controls; keep the template's values |
| `fixtures` | Surfaces objects stand on (see below) |
| `containers` | Things objects go into: fridges, cabinets, bins (see below) |
| `objects` | Movable objects the task places (see below) |
| `labels` | Printed documents and labels rendered onto objects |
| `start` | Where the human starts: `{"policy": "seeded_grid", "forbid_rooms": [], "forbid_near": [], "resample_ladder": true}` |
| `initially_open_room_doors` | Open all room doors at reset |
| `assertions` | Authoring checks the world must pass (`no_pre_satisfied`, `labels_present`, …); keep `true` |
| `decor`, `parts`, `preload`, `roles`, `permutations`, `object_groups`, `initial_attachments` | Advanced features (assembly, randomisation); usually empty |

**Fixtures and containers** bind to furniture of the scene:

```json
"surf_call": {"bind": {"cat": "breakfast_table", "source": "scene"},
              "room": "living_room_0", "seatable": false}

"fridge": {"bind": {"cat": "fridge", "source": "scene",
                    "pick": {"by": "scene_instance", "name": "fridge_dszchb_0",
                             "stage_prim_path": "/World/b_fridge_dszchb_0", "rank": 0,
                             "asset_sha256": "sha256:…", "instance_digest": "sha256:…"}},
           "kind": "front_opening", "door": {"required": true},
           "role": "scored", "room": "kitchen_0", "alt_cats": []}
```

* `bind.source: "scene"` uses an existing piece of furniture of category `cat` in `room`.
  Without `pick`, the simulator chooses one deterministically; with `pick`, it must be
  exactly that instance. Copy `pick` blocks from a bundled TaskPack of the same scene:
  their digests are checked against your baked assets.
* `kind` is `front_opening` (has a door: fridge, cabinet, oven) or `open_bin` (open top).

**Objects** are BEHAVIOR models placed at reset:

```json
"item_2": {"cat": "bottle_of_chocolate_sauce", "model": "tsyldw",
           "place": "on:surf_pantry", "room": "kitchen_0", "spot": [-0.62, 0.15],
           "scored": true, "needs_lookat": true, "cue": "native_nutrition"}
```

| Field | Meaning |
| :--- | :--- |
| `cat`, `model` | A BEHAVIOR category and model; it must be in the baked cache (`$INTERACTIVEWORLD_BEHAVIOR_CACHE/usd/<cat>/<model>.usd`) |
| `place` | `on:<fixture or object>`, `in:<container>` or `floor:<room>` |
| `spot` | Optional `[x, y]` position on the support, in metres from its centre |
| `room` | Room of the support |
| `scored` | Whether the object takes part in the goal |
| `needs_lookat` | The object carries information that must be read (a label, a document) |
| `decal`, `cue` | Text printed on the object; which label renderer supplies its facts |

### `goal`

```json
"goal": {
  "goals":      [{"pred": "on", "subject": "item_2", "target": "surf_call"}],
  "answer_key": [{"pred": "on", "subject": "item_2", "target": "surf_call"}],
  "negatives":  [],
  "roster":     {"objects": ["item_1", "item_2", "item_3", "item_4", "item_5"],
                 "containers": [], "nested_containers": []},
  "aggregate_goals": [], "trace_outcomes": [], "arms": null, "rule_table": null
}
```

`goals` must all hold at the end for `strict_success`. Predicates: `on`, `inside`,
`held`, `open`, `closed`, `running`, `off`, `near`, `cut`, `cooked`, `attached`,
`raised`, `unmoved` ([ACTIONS.md](ACTIONS.md#state-predicates)). `negatives` are states
that must *not* hold (for example `{"pred": "on", "subject": "item_4", "target": "surf_call"}`).
`roster` lists every object and container the goal may mention.

### `program`, `meta`, `report`, `provenance`

* `program.commands`: the full reference program as `{"verb": "Put", "args": ["item_2", "surf_call"]}`
  rows: the prologue followed by the oracle solution. The simulator executes it while
  validating the world; `program.budget` records its length.
* `meta`: scene, family, difficulty and the authoring rationale (`design.premise`,
  `design.assistant_must`, `design.anti_shortcut`).
* `report`: per-goal bookkeeping (`sub_goals`, `sub_steps`) used in reports.
* `provenance`: free-form authoring metadata.

## `story`: what the person experienced

| Field | Meaning |
| :--- | :--- |
| `premise` | The situation in plain language |
| `assistant_goal` | The person's opening request to the Assistant |
| `prologue.actions` | Commands the human performs before asking; the Assistant sees their frames |
| `prologue.say` | Optional words spoken during the prologue |
| `beats` | The authored timeline: each beat has an `id`, `actor` (`human`, `world`, `assistant`), `phase` (`reset`, `before_turn0`, `during`), an `action`, its state `effects` and the facts it `establishes` |
| `evidence` | Facts the Assistant can only know from what it saw or heard (`fact_001`, …) and where they come from |
| `interaction` | `public_task_context` (what the Assistant is told), `human_task_context` (the Human's brief: when to act and what not to do), `opening` (the opening utterances and actions), `information_contract`, `completion_contract` (e.g. phrases the guidance must contain), `motor_aliases` |
| `assessment_contract` | What the task tests: `core_question`, `success_condition`, `required_final_predicates`, `required_evidence`, `forbidden_changes` |

The Human and Assistant models only see what these fields expose, so keep the text
consistent with the world: if you move the destination, rename it in
`assistant_goal`, `human_task_context` and the completion phrases too.

## `runtime`: how the episode runs

| Field | Meaning |
| :--- | :--- |
| `contract_version` | `compiled-runtime-contract/1.8` |
| `tick_budget` | Maximum Human/Assistant turns |
| `dialogue_budget` | Maximum scripted Assistant interventions in `oracle-verbal` mode |
| `dialogue` | Scripted guidance for `oracle-verbal`: `{"id", "say", "anchor_beat_id", ...}` |
| `oracle_commands` | The correct Human actions after guidance (used by the oracle and in validation) |
| `default_commands` | What the Human does without guidance |
| `human_policy` | The Human's state machine (below) |
| `constraints` | Runtime guards such as `human_must_remain_on_scene_floor` |
| `public_preparation` | `{"mode": "continuous_watch"}` for tasks where the person keeps acting while the Assistant watches; omitted for prologue tasks |

### `human_policy`

```json
"human_policy": {
  "version": "human-policy/1.0",
  "initial_state": {"facing": "item_4", "held": [], "posture": "standing"},
  "default_plan": ["turn0", "h000", "h001", "h002"],
  "intervention_window": {"after_beat": "turn0", "before_beat": "h000"},
  "transitions": [
    {"after_beat": "turn0", "next_beat": "o000", "intervention_beat": "assistant_intervention",
     "condition": {"kind": "semantic_and_world", "expected_act": "correct",
                   "expected_referents": ["item_2"],
                   "required_fact_ids": ["fact_003", "fact_007", "fact_011"],
                   "verifier": "human_model_then_world",
                   "world_check": {"pred": "held", "subject": "item_2", "truth": false}}},
    {"after_beat": "o000", "next_beat": "o001",
     "condition": {"kind": "world_state", "pred": "held", "subject": "item_2", "truth": true}},
    {"after_beat": "o001", "next_beat": "o002",
     "condition": {"kind": "world_state", "pred": "at", "subject": "human", "target": "surf_call", "truth": true}}
  ],
  "terminal_beats": ["h002", "o002"]
}
```

The Human follows `default_plan` (its own, unguided course of action) unless a transition
fires. Transition conditions are:

* `world_state`: a predicate on the live world becomes true or false;
* `semantic_and_world`: the Human model judges that the Assistant's guidance is correct and
  refers to `expected_referents` using the facts in `required_fact_ids`, and the
  `world_check` holds;
* `timeout`: a number of turns passes.

The episode ends at a `terminal_beat`. The `o…` beats follow the oracle path and the `h…`
beats the unguided default.

## Rules a valid TaskPack satisfies

`TaskSpec.load` and the simulator reject a TaskPack unless:

* the file matches the SHA256 you pass, and it is plain JSON without duplicate keys or
  non-finite numbers;
* `taskpack_version`, `runtime.contract_version` and `human_policy.version` are the
  supported versions, and `executable_spec.task_id == task_id`;
* `story.premise`, `story.assistant_goal`, `story.prologue.actions` and
  `story.prologue.say` exist; `runtime.tick_budget` is a positive integer;
  `human_policy` has `initial_state.facing`, a non-empty `default_plan`, an
  `intervention_window`, `transitions` and non-empty `terminal_beats`;
* the scene is admitted, every `cat`/`model` is baked, every `place` resolves, scene-bound
  `pick` digests match, objects do not overlap, and no goal holds before the task starts;
* the reference program executes and reaches the goal; `--strict-acceptance` adds stricter
  authoring checks while you develop a task.

The validators live in `simulator/catalog.py`,
`simulator/taskpack_support/runtime_adapter.py` and
`simulator/physical_world/spec_validator.py`.

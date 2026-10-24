# Using the simulator

InteractiveWorld has three layers: a **physical layer** (BEHAVIOR-1K scenes in Isaac Sim,
object physics and state), a **human-action layer** (an SMPL+H human that executes
commands with planned stances, routes, reaches and grips) and a **semantic layer**
(free-form Human/Assistant interaction grounded in the live scene). This guide covers the
first two; the benchmark layer is in [BENCHMARK.md](BENCHMARK.md).

Install first ([INSTALL.md](INSTALL.md)) and `source ~/interactiveworld-assets/env.sh`.

## Command line

`./launch/simulator.sh` runs with any Python 3.10+; only `run` starts Isaac Sim.

| Command | Purpose |
| :--- | :--- |
| `worlds` | The 44 scenes that open without a TaskPack |
| `capabilities [--all]` | The action catalog ([ACTIONS.md](ACTIONS.md)) |
| `tasks list [--category C] [--source-family F]` | The 120 bundled tasks |
| `tasks show TASK_ID` | One TaskPack and its compiled runtime spec |
| `doctor` | Checks the Isaac Python and every asset; exit status 1 on failure |
| `run` | Opens a scene or task, executes `--action` commands and prints JSON |

### `run`

Exactly one source: `--scene NAME`, `--task TASK_ID` or `--taskpack FILE --taskpack-sha256 sha256:…`.

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--action COMMAND` | | A command; repeat for a sequence. Execution stops early when a task's goal is met |
| `--prepare-task` | off | Apply the task's prologue (what the human did before asking) and start scoring from there |
| `--gpu N` | 0 | Physical GPU index |
| `--seed N` | 0 | World seed |
| `--frames-per-action N` | 4 | First-person frames rendered across each action |
| `--frames-dir DIR` | | Write every frame as a numbered PNG |
| `--video FILE` | | Encode the frames as `.mp4` or `.gif` (needs `ffmpeg`), at `--video-fps` (default 8) |
| `--no-render` | | Skip rendering (symbolic run, faster) |
| `--observation-profile agent\|debug` | agent | `debug` adds relations and the native score to every observation |
| `--output-dir DIR` | system temp | Runtime scratch directory (sockets, exposure cache) |
| `--visibility-gating` | off | Restrict the actionable objects to those the human can see |
| `--strict-acceptance` | off | Stricter authoring checks: finer stance search and no navigation fallbacks. The benchmark runs without them, so leave it off when you compare with benchmark behavior |

The JSON printed on stdout has `scene`, `task_id`, `initial_observation`, one
`{action, result}` per step, the final `score` (tasks only) and the saved `frames`/`video`.
Isaac Sim logs go to stderr, so `> result.json` captures clean JSON.

Opening takes a few minutes: about 2–4 minutes for a raw scene and 5–10 minutes for a task,
whose world is validated while it opens. Each action then takes seconds to a minute,
mostly for stance and route planning, which is CPU-bound.

## Python API

`import simulator` loads only the standard library. `Simulator.launch` starts Isaac Sim in
a child process through `ISAAC_PYTHON` and talks to it over a local socket, so your
program can use any Python and ML stack.

```python
from simulator import Action, Simulator, SimulatorConfig

config = SimulatorConfig(gpu=0, seed=0, frames_per_action=8)
with Simulator.launch(scene="Rs_int", config=config) as world:
    observation = world.observe()
    result = world.step(Action.grab("can_of_soda"))     # or world.step("[Grab] <can_of_soda> (1)")
    results = world.run(["[Walk] <coffee_table_fqluyq_0> (1)",
                         "[Put] <can_of_soda> (1) <coffee_table_fqluyq_0> (1)"])
    world.switch_world("Beechwood_0_int")               # reuse the warm process
```

### Lifecycle

| Call | Meaning |
| :--- | :--- |
| `Simulator.launch(scene=…)` / `launch(task_id=…)` / `launch(task=TaskSpec)` | Start an owned simulator and open a world or task; the context manager shuts it down |
| `Simulator.connect_unix(path)` / `connect_tcp(host, port)` | Attach to a running server without owning it |
| `open_world(scene)`, `open_task(task_id=… or task=…)` | Open after attaching |
| `switch_world(scene)`, `switch_task(…)` | Rebuild in the same process |
| `reset(start_xy=None)` | Rebuild the current world from its seed |
| `close()` / `shutdown()` / `detach()` | Close the world / stop the server / leave it running |

### Acting and observing

| Call | Returns |
| :--- | :--- |
| `observe()` | `Observation` of the current state |
| `step(action)` | `StepResult` for one `Action` or command string |
| `run(actions, stop_on_termination=True)` | A tuple of `StepResult` |
| `render()` | The latest PNG frame as bytes, or `None` without rendering |
| `prepare_task()` | `TaskPreparation(steps, observation, score_baseline)` after applying the prologue |
| `score()` | `Score(success, progress, n_done, n_total, prohibitions_ok, scoreable, details)` |
| `reward()` | The raw reward payload |
| `actions()`, `list_worlds()`, `list_tasks()`, `describe()` | Catalogs and server information |

`Action` builds commands without string formatting: `Action.walk(t)`, `Action.grab(o)`,
`Action.put_on(o, s)`, `Action.put_inside(o, c)`, `Action.open(o)`, `Action.close(o)`,
`Action.turn_on(o)`, `Action.turn_off(o)`, `Action.look_at(o)`, `Action.look_around()`,
`Action.walk_forward(meters=None)`, `Action.turn_left(degrees=None)`,
`Action.turn_right(degrees=None)`, `Action.wait(seconds=None)`, or generically
`Action("Cut", ("tomato", "knife"))` and `Action.parse("[Wait] 5s")`.

### `Observation`

| Field | Content |
| :--- | :--- |
| `scene`, `task_id`, `task` | What is open |
| `objects` | Object keys the human can act on (`objects_are_visible_only` tells whether this is the visible subset) |
| `held`, `at` | Objects in hand; what the human stands at |
| `frame_png`, `frames_png` | The latest first-person frame and the frames rendered across the last action |
| `frame_segments` | The command each frame belongs to |
| `action_trace` | Structured events of the last step |
| `relations`, `score` | Live predicates and the native score (`debug` profile only) |
| `task_completed` | Whether the task goal holds |

`StepResult` adds `outcomes` (one `ActionOutcome(primitive, ok, reason, detail)` per
primitive), `success`, `error_code`, `error_message`, `terminated`, `truncated` and
`timings`. Reason codes are listed in [ACTIONS.md](ACTIONS.md#outcome-codes).

### `SimulatorConfig`

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `gpu`, `seed` | 0, 0 | Physical GPU, world seed |
| `render_observations` | True | Render first-person frames |
| `frames_per_action` | 4 | Frames sampled across each action |
| `observation_profile` | `"agent"` | `"debug"` exposes relations and score |
| `visibility_gating`, `visibility_checks` | False, True | Visible-only object lists; visibility diagnostics |
| `strict_acceptance` | False | Stricter authoring checks (see `--strict-acceptance`) |
| `output_dir` | system temp | Runtime directory outside the repository |
| `isaac_python`, `isaac_env_dir` | from environment | Override `ISAAC_PYTHON` / `ISAAC_ENV_DIR` |
| `open_timeout_s`, `step_timeout_s`, `connect_timeout_s` | 2400, 600, 120 | Timeouts |
| `assets` | from environment | `AssetConfig(behavior_cache=…, behavior_data=…, smplh_model=…, walk_clip=…)` |

Errors derive from `SimulatorError`: `SimulatorConfigError`, `TaskNotFoundError`,
`TaskValidationError`, `SimulatorConnectionError`, `SimulatorTimeoutError`,
`SimulatorProtocolError` and `SimulatorRemoteError` (raised for server-side failures).

## Tasks inside the simulator

A bundled or custom TaskPack opens like a scene. `prepare_task()` replays the task's
prologue (for example, the human reading a report and taking a phone call) and captures
the score baseline; your actions then continue from the hand-off point:

```python
from simulator import Simulator, SimulatorConfig, TaskCatalog

task = TaskCatalog.bundled().get("ASK-12802")
with Simulator.launch(task=task, config=SimulatorConfig(gpu=0)) as world:
    preparation = world.prepare_task()
    world.run(["[Grab] <item_2> (1)", "[Walk] <surf_3> (1)", "[Put] <item_2> (1) <surf_3> (1)"])
    print(world.score().success)
```

The 22 continuous-watch tasks (all of Prevention, plus one Decision and one Finding task)
keep the human acting while the assistant watches; they only run through the benchmark runner, which interleaves the
Human and Assistant turns.

## Persistent server

Opening a world is the slow part. Keep one simulator alive and attach to it repeatedly:

```bash
./launch/run_isaac.sh launch/run_sim_server.py --sock /tmp/interactiveworld.sock \
    --scene Rs_int --preopen --gpu 0
```

```python
from simulator import Simulator

world = Simulator.connect_unix("/tmp/interactiveworld.sock")
print(world.observe().objects)
world.step("[LookAround]")
world.detach()           # leave the server running; world.shutdown() stops it
```

`--port 5555` serves TCP instead (loopback only unless `--bind ADDRESS --allow-remote`;
the protocol has no authentication). `--no-render-obs` skips rendering,
`--observation-profile debug` exposes evaluator state. The protocol is newline-delimited
JSON (`simulator/protocol.py`, `simulator/client.py`).

## Examples

| File | Shows |
| :--- | :--- |
| [`examples/01_list_tasks.py`](../examples/01_list_tasks.py) | Reading the task catalog without Isaac Sim |
| [`examples/02_run_actions.py`](../examples/02_run_actions.py) | Opening a scene and stepping typed actions |
| [`examples/03_custom_taskpack.py`](../examples/03_custom_taskpack.py) | Loading a hash-pinned custom TaskPack, applying its prologue, scoring |
| [`examples/04_prepare_task.py`](../examples/04_prepare_task.py) | The state before and after a task prologue |
| [`examples/05_derive_taskpack.py`](../examples/05_derive_taskpack.py) | Deriving a new TaskPack from a bundled one ([TASKPACK.md](TASKPACK.md)) |

## Where the code lives

| Path | Responsibility |
| :--- | :--- |
| `simulator/api.py`, `client.py`, `protocol.py`, `types.py`, `actions.py` | Public API, process isolation and wire protocol |
| `simulator/catalog.py` | TaskPack loading and validation |
| `simulator/transport/isaac_sim_server.py` | The server inside Isaac Sim |
| `simulator/environment/isaac_env.py` | Reset, step, observe and score of one world |
| `simulator/physical_world/episode_runtime.py` | Building a world from a TaskPack or scene |
| `simulator/physical_world/semantic_executor.py` | One producer per action family |
| `simulator/physical_world/command_grammar.py` | Command parsing |
| `simulator/physical_world/nav_planner.py`, `door_planner.py` | Routes and doors |
| `simulator/physical_world/grasp_select.py`, `smplh_anim_core.py` | Grips and body motion |
| `simulator/physical_world/behavior_import.py`, `behavior_bake.py` | BEHAVIOR scenes and the asset bake |
| `simulator/interaction/` | Human/Assistant episode loop and prompts |

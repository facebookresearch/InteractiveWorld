<div align="center">

# InteractiveWorld

### A Verifiable Human-Centric Environment for Building Better Interactive MLLM Assistants

**Website** *(coming soon)* &nbsp;|&nbsp; **Paper** *(coming soon)* &nbsp;|&nbsp; [**Code**](https://github.com/facebookresearch/InteractiveWorld)

<!-- Replace the Website and Paper labels above with public links when available. -->

</div>

InteractiveWorld is a closed-loop simulator for studying how an AI assistant helps a human in the physical world. The assistant sees through the human's egocentric camera and communicates with the human; the human decides what to do; the simulator executes semantic actions and updates the world.

InteractiveBench is built on top of the simulator as a collection of structured, hash-pinned TaskPack JSON documents. It contains 120 multi-turn tasks across six categories and evaluates both the assistant model and the human model against verifiable world state.

## Highlights

- **Simulator first.** Open a physical world, observe it, move the human, manipulate objects, or load a structured task from either the CLI or Python.
- **120 verifiable tasks.** Guidance, Planning, Accessibility, Decision, Prevention, and Finding each contain 20 tasks.
- **Human and assistant are independently configurable.** Each role can use a hosted model, an OpenAI-compatible open-source server, or the provider-neutral HTTP interface.
- **Reproducible evaluation.** Deterministic task selection, physical-GPU assignment, resumable runs, dialogue reports, state-based scoring, and optional videos are part of one runner.
- **Extensible TaskPacks.** New tasks use the same structured JSON contract as the bundled benchmark and can be loaded with a required SHA-256 digest.

## Repository layout

| Path | Purpose |
| --- | --- |
| `simulator/` | Public Python API, semantic actions, process-isolated Isaac runtime, physical worlds, and TaskPack adapter |
| `benchmark/tasks/` | Frozen manifest, 120 TaskPack JSON files, and bundled task label assets |
| `benchmark/runner/` | One-command model runner, multi-GPU orchestration, reports, and video generation |
| `benchmark/evaluation/` | State-based rubric, evidence preparation, and scoring |
| `launch/` | User-facing simulator and benchmark commands |
| `examples/` | Small Python examples for discovery, actions, custom TaskPacks, and task preparation |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/facebookresearch/InteractiveWorld.git
cd InteractiveWorld
```

### 2. Install the runtime

InteractiveWorld targets Linux, NVIDIA GPUs, **Isaac Sim 5.1**, and the Python 3.11 environment provided with Isaac Sim. Point `ISAAC_PYTHON` at an absolute executable path:

```bash
export ISAAC_PYTHON=/absolute/path/to/isaac-sim/python.sh

# Optional: prepend this environment's lib/ directory at launch time.
export ISAAC_ENV_DIR=/absolute/path/to/isaac-sim-environment
```

The full Python-package and runtime contract is recorded in [`simulator/external_dependencies.json`](simulator/external_dependencies.json). `ffmpeg` is optional and is needed only when benchmark video generation is enabled.

### 3. Configure external assets

Large BEHAVIOR-1K assets and licensed human models are deliberately not redistributed in this repository. Obtain them from their respective licensors and configure absolute paths:

```bash
export INTERACTIVEWORLD_BEHAVIOR_CACHE=/absolute/path/to/behavior-cache
export INTERACTIVEWORLD_BEHAVIOR_DATA=/absolute/path/to/behavior-data
export INTERACTIVEWORLD_MANO_PYTHON=/absolute/path/to/mano-python
export INTERACTIVEWORLD_SMPLH_MODEL=/absolute/path/to/SMPLH_male.pkl
export INTERACTIVEWORLD_WALK_CLIP=/absolute/path/to/walk_cycle.npz
export INTERACTIVEWORLD_HUMAN_ASSET_ROOT=/absolute/path/to/human-assets
```

Check the code release and your local installation before starting Isaac Sim:

```bash
python3 -B validate_release.py
./launch/simulator.sh doctor
```

`doctor` reports every missing runtime or asset path. Exact required subdirectories and file hashes are defined in [`simulator/external_dependencies.json`](simulator/external_dependencies.json).

## Use the simulator

The discovery commands are lightweight and do not start Isaac Sim:

```bash
# List the 44 available physical worlds.
./launch/simulator.sh worlds

# List executable semantic action families.
./launch/simulator.sh capabilities

# List or inspect the bundled structured tasks.
./launch/simulator.sh tasks list
./launch/simulator.sh tasks show ASK-13000
```

### Drive a world from the command line

This opens a task-free world, observes it, and executes three semantic actions:

```bash
./launch/simulator.sh run \
  --scene Rs_int \
  --gpu 0 \
  --action '[LookAround]' \
  --action '[WalkForward]' \
  --action '[TurnLeft]'
```

Use `--no-render` when RGB observations are unnecessary. Use `--observation-profile debug` for additional simulator state. Object actions use stable references such as `[Open] <fridge> (1)`; run `capabilities` to inspect all supported action families.

To initialize a bundled task, apply its structured prologue, execute actions, and return its native state score:

```bash
./launch/simulator.sh run \
  --task NARRATE-104000 \
  --prepare-task \
  --gpu 0 \
  --action '[LookAround]'
```

Continuous-watch tasks should be run with the benchmark runner so that Human/Assistant turns and time-dependent checks remain active.

### Drive a world from Python

The public Python package is import-safe: Isaac Sim is started only by `Simulator.launch`, in a separate process.

```python
from simulator import Action, Simulator, SimulatorConfig

config = SimulatorConfig(
    gpu=0,
    seed=0,
    render_observations=True,
)

with Simulator.launch(scene="Rs_int", config=config) as simulator:
    observation = simulator.observe()
    print(observation.scene, observation.at, observation.objects)

    for action in (
        Action.look_around(),
        Action.walk_forward(),
        Action.turn_left(),
    ):
        result = simulator.step(action)
        print(action.to_command(), result.success, result.error_code)
```

For a bundled task, replace `scene="Rs_int"` with `task_id="NARRATE-104000"`, call `simulator.prepare_task()`, and use `simulator.score()` to read the verifiable state score. See [`examples/`](examples) for complete scripts.

### Load a custom TaskPack

TaskPacks are JSON documents using the same contract as the files under [`benchmark/tasks/taskpacks/`](benchmark/tasks/taskpacks). Start from a bundled example, change the task specification, and compute its digest:

```bash
sha256sum my_task.taskpack.json
```

The digest is mandatory when loading an external task:

```bash
./launch/simulator.sh run \
  --taskpack /absolute/path/to/my_task.taskpack.json \
  --taskpack-sha256 sha256:YOUR_SHA256_HERE \
  --prepare-task \
  --gpu 0 \
  --action '[LookAround]'
```

The equivalent typed Python entry point is `TaskSpec.load(path, expected_sha256=...)`; [`examples/03_custom_taskpack.py`](examples/03_custom_taskpack.py) shows the full flow. This separation is intentional: the simulator supplies the world and action interface, while TaskPack documents supply task initialization, interaction rules, and verifiable goals.

## Run InteractiveBench

### Supported model endpoints

Human and Assistant endpoints are configured independently.

| Provider | Intended use |
| --- | --- |
| `openai-compatible` | Open-source models served by vLLM, Ollama, or another Chat Completions-compatible server |
| `openai` | Hosted OpenAI models |
| `anthropic` | Hosted Anthropic models |
| `gemini` | Hosted Gemini models |
| `http-json` | Provider-neutral `/generate` service; defaults to loopback ports 8801/8802 |

API keys are read only from environment variables; put the variable's **name**, never the secret value, in a command or JSON configuration.

### Evaluate a hosted model on all 120 tasks

```bash
export OPENAI_API_KEY='YOUR_API_KEY'

env -u CUDA_VISIBLE_DEVICES ./launch/interactiveworld.sh \
  --all \
  --assistant-provider openai \
  --assistant-model gpt-4.1 \
  --assistant-api-key-env OPENAI_API_KEY \
  --human-provider openai \
  --human-model gpt-4.1 \
  --human-api-key-env OPENAI_API_KEY \
  --num-gpus 4 \
  --run-id gpt41-full \
  --output-dir /tmp/interactiveworld-runs/gpt41-full
```

### Evaluate open-source models

Start the Human and Assistant model servers separately, then point each role at its OpenAI-compatible endpoint. The model servers may use the same model or different models.

```bash
env -u CUDA_VISIBLE_DEVICES ./launch/interactiveworld.sh \
  --all \
  --assistant-provider openai-compatible \
  --assistant-model Qwen/Qwen3-8B \
  --assistant-base-url http://127.0.0.1:8000/v1 \
  --human-provider openai-compatible \
  --human-model Qwen/Qwen3-32B \
  --human-base-url http://127.0.0.1:8001/v1 \
  --gpus 0,1,2,3 \
  --run-id qwen-full \
  --output-dir /tmp/interactiveworld-runs/qwen-full
```

Loopback endpoints may use HTTP. Remote model endpoints must use HTTPS. When model servers and Isaac Sim share a machine, reserve enough GPU memory for both workloads.

### Run one task and print its conversation and score

```bash
env -u CUDA_VISIBLE_DEVICES ./launch/interactiveworld.sh \
  --task-id ASK-13000 \
  --assistant-provider openai-compatible \
  --assistant-model Qwen/Qwen3-8B \
  --assistant-base-url http://127.0.0.1:8000/v1 \
  --human-provider openai-compatible \
  --human-model Qwen/Qwen3-32B \
  --human-base-url http://127.0.0.1:8001/v1 \
  --gpus 0 \
  --video all \
  --video-captions full \
  --run-id ask-13000 \
  --output-dir /tmp/interactiveworld-runs/ask-13000
```

The terminal prints the ordered Human/Assistant conversation, run validity, native TaskPack completion score, and release-rubric status. Machine-readable outputs are written to:

```text
/tmp/interactiveworld-runs/ask-13000/
├── report.json                 # suite summary
├── reports/ASK-13000.json      # transcript, score, validity, and artifact index
├── evidence/pending/           # criterion-keyed evidence template
├── evidence/adjudicated/       # completed evidence goes here
└── videos/                     # optional rendered MP4
```

The native TaskPack score is available immediately. The official release-rubric score is emitted only after complete, trace-bound adjudication. The console tells you which pending evidence file to complete; rerun the same output directory with `--resume --adjudication /tmp/interactiveworld-runs/ask-13000/evidence/adjudicated` to produce the final release score.

### Select tasks and GPUs

```bash
# One or more exact tasks.
./launch/interactiveworld.sh --task-id ASK-13000 --task-id GUIDE-14004 ...

# All 20 tasks in one assigned category.
./launch/interactiveworld.sh --category Guidance ...

# Physical GPU indices, one worker per GPU.
env -u CUDA_VISIBLE_DEVICES ./launch/interactiveworld.sh --gpus 1,3 ...

# GPUs 0 through N-1.
env -u CUDA_VISIBLE_DEVICES ./launch/interactiveworld.sh --num-gpus 4 ...
```

Selected tasks are deterministically sharded round-robin across workers. `CUDA_VISIBLE_DEVICES` must be unset because `--gpus` and `--num-gpus` refer to physical indices.

### Video, execution, and output controls

| Control | Values / behavior |
| --- | --- |
| `--video` | `off`, `failures`, or `all` |
| `--video-fps` | Playback frame rate |
| `--video-resolution` | Even `WIDTHxHEIGHT`, for example `1280x720` |
| `--video-captions` | `off`, `speaker`, or `full` |
| `--frames-per-tick`, `--frame-budget`, `--max-frames` | Egocentric observation budget |
| `--seed` | Deterministic episode seed |
| `--resume` | Reuse compatible successful tasks in the output directory |
| `--fail-fast` | Stop all workers after the first run error |
| `--dry-run` | Validate configuration and write the execution plan without starting Isaac Sim |

The 120 oracle videos are distributed as a companion bundle rather than committed to Git. Their public integrity and provenance metadata is in [`benchmark/oracle_videos/manifest.json`](benchmark/oracle_videos/manifest.json).

### Use a JSON configuration

For repeatable experiments, copy [`benchmark/runner/config.example.json`](benchmark/runner/config.example.json), edit model names and endpoints, and validate the complete plan before running:

```bash
cp benchmark/runner/config.example.json /tmp/interactiveworld-config.json
./launch/interactiveworld.sh \
  --config /tmp/interactiveworld-config.json \
  --output-dir /tmp/interactiveworld-runs/dry-run \
  --run-id dry-run \
  --dry-run
```

CLI values override fields from the JSON configuration.

## Validate the release

These checks do not start Isaac Sim:

```bash
python3 -B validate_release.py
python3 -B benchmark/tasks/validate.py
python3 -B -m benchmark.evaluation validate
```

## Citation

Paper link and publication metadata are coming soon. Until then, please cite the project title:

```bibtex
@article{wang2026interactiveworld,
  title   = {InteractiveWorld: A Verifiable Human-Centric Environment for Building Better Interactive MLLM Assistants},
  author  = {Wang, Qinsi and Wen, Wei and Zhou, Chong and Suri, Saksham and Liu, Zechun and Wu, Lemeng and Li, Hai Helen and Chen, Yiran and Krishnamoorthi, Raghuraman and Qian, Qi},
  year    = {2026},
  note    = {Paper coming soon}
}
```

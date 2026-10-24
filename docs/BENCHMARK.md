# Evaluating models on InteractiveBench

InteractiveBench has **120 frozen tasks**, 20 per category:

| Category | The assistant must… | Example |
| :--- | :--- | :--- |
| Guidance | guide the user step by step, based on their progress | `GUIDE-14004`: cook a dish from a recipe card, one step at a time |
| Planning | plan steps within time and resource limits and replan | `ORCHESTRATE-99902`: the most allergy-safe dishes in 30 minutes |
| Accessibility | describe the scene so a blind user can act (the Human gets no images) | `NARRATE-87016`: a kitchen clean-up |
| Decision | decide from what it has seen plus world knowledge | `ASK-12802`: the snack that fits a calorie budget |
| Prevention | warn before a mistake without needless interruptions | `WATCH-47003`: stop the user at the exit |
| Finding | find objects by remembering where they went | `FIND-87004`: where did the red cola go? |

`./launch/interactiveworld.sh --list-tasks` prints the catalog.

## How an episode runs

1. The simulator opens the task's scene and replays its **prologue**: what the person did
   before asking (reading a report, taking a call, moving things). The Assistant sees
   these first-person frames.
2. The **Human** speaks the task's opening request.
3. Turns alternate. The **Assistant** (the model under test) sees only the first-person
   frames and the dialogue, and replies with speech, silence or a `done` declaration.
   The **Human** (a fixed model) reads the Assistant's words and the scene and answers
   with speech and/or simulator commands from the [action grammar](ACTIONS.md).
4. The simulator executes the Human's commands and renders the next frames.
5. The episode ends when the Assistant declares `done`, the Human finishes, or the tick
   budget runs out. Success is read from the final world state.

## The command

```bash
env -u CUDA_VISIBLE_DEVICES ./launch/interactiveworld.sh [selection] [assistant] [human] [runtime] [output]
```

Unset `CUDA_VISIBLE_DEVICES`: GPU options use physical indices and each worker pins its
own device.

### Models

Both roles take the same options with an `--assistant-` or `--human-` prefix.

| Option | Meaning |
| :--- | :--- |
| `--{role}-provider` | `openai-compatible` (vLLM, Ollama, any Chat Completions server), `openai`, `anthropic`, `gemini`, or `http-json` (a provider-neutral `/generate` service) |
| `--{role}-model` | Model ID as the endpoint expects it |
| `--{role}-base-url` | Endpoint URL. Required for `openai-compatible` and `http-json`; optional override for hosted providers |
| `--{role}-api-key-env` | **Name** of the environment variable holding the key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …) |
| `--{role}-timeout-s`, `--{role}-tokens`, `--{role}-frame-cap` | Request timeout, maximum output tokens (0 = provider default), images per request |
| `--assistant-control` | `model` (default), or a baseline: `oracle-verbal` (scripted correct guidance) or `silent`. Baselines do not record videos |

Remote endpoints must use HTTPS; plain HTTP is accepted only on loopback. Keys are never
accepted on the command line or written to artifacts.

Typical setups:

```bash
# Open-source model under test, served locally (one GPU), hosted Human
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
--assistant-provider openai-compatible --assistant-model Qwen/Qwen3-VL-8B-Instruct \
--assistant-base-url http://127.0.0.1:8000/v1

# Hosted models
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...
--assistant-provider openai --assistant-model gpt-4.1 --assistant-api-key-env OPENAI_API_KEY
--assistant-provider anthropic --assistant-model <claude-model-id> --assistant-api-key-env ANTHROPIC_API_KEY
--human-provider gemini --human-model gemini-3.1-pro-preview --human-api-key-env GEMINI_API_KEY
```

The paper uses Gemini 3.1 Pro as the Human for every Assistant, so that results are
comparable; keep the Human fixed when you compare Assistants.

### Task selection

| Option | Selects |
| :--- | :--- |
| *(none)* or `--all` | All 120 tasks |
| `--category Guidance` | One category; repeat for several |
| `--task-id ASK-13000` | One task; repeat for several |

### Runtime

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--num-gpus N` / `--gpus 1,3` | 1 / – | Isaac workers, one per physical GPU; tasks are sharded round-robin in a fixed order |
| `--seed` | 0 | World seed |
| `--max-ticks` | task budget | Cap on Human/Assistant turns |
| `--frame-budget`, `--frames-per-tick`, `--max-frames`, `--frame-max-side` | 24, 4, 256, 512 | How many first-person frames the models see and at which size |
| `--fail-fast` | off | Stop all workers after the first run error |
| `--dry-run` | | Write and print the execution plan without starting Isaac Sim |

### Output

| Option | Meaning |
| :--- | :--- |
| `--output-dir DIR` | Run directory (default `$INTERACTIVEWORLD_RUN_ROOT/<run-id>` or the system temp directory) |
| `--run-id ID` | Stable run name |
| `--resume` | Skip tasks that already completed in `DIR` with the same configuration and code |
| `--video off\|failures\|all` | Render first-person videos for no, failed or all tasks (needs `ffmpeg`) |
| `--video-captions off\|speaker\|full` | Subtitle the videos with the dialogue |
| `--video-fps`, `--video-resolution` | Playback rate and size |

### Configuration files

Every option has a JSON form. Start from
[`benchmark/runner/config.example.json`](../benchmark/runner/config.example.json), then
`./launch/interactiveworld.sh --config my_config.json`; flags on the command line override
the file. `--dry-run` shows the merged configuration.

## Output

```text
<output-dir>/
├── report.json                    suite summary: run status, native score, release score, per category
├── reports/<TASK-ID>.json         one task: dialogue, transcript, participants, scores, artifacts
├── evidence/pending/              criterion templates for the release score
├── evidence/adjudicated/          completed criteria (you create these)
├── videos/                        first-person MP4s with SRT subtitles
├── run_manifest.json              resolved configuration, code identity, task shards
└── workers/gpu_<N>_<k>/runner.log Isaac and episode logs of each worker
```

A single-task run also prints the conversation:

```text
Task ASK-12802: COMPLETED
Run validity: VALID
Native TaskPack: FAIL (completion=0.000)
Release rubric: UNSCORABLE (adjudication=PENDING, completion=n/a)
Conversation:
[turn -] HUMAN: Hey, Assistant, which one should I choose for the most calories without going over my daily limit? ...
[turn 0] ASSISTANT: Check the nutrition label on the chocolate syrup bottle to see how many calories it has.
...
```

## Scores

Every task gets two independent scores.

* **Native TaskPack score**, available immediately. `strict_success` is true when every
  goal predicate holds in the final world state and the task's process requirements are
  met; `completion_score` is the fraction of goals met. `report.json` aggregates them as
  `strict_pass_rate` and `mean_completion_score`.
* **Release rubric score**, the official benchmark number. Each task has protocol-specific
  criteria in [`benchmark/evaluation/rubric.json`](../benchmark/evaluation/rubric.json)
  (464 in total, six protocols). The run writes one template per task to
  `evidence/pending/`, bound to the exact run evidence by `run_evidence_sha256`. Copy it to
  `evidence/adjudicated/`, fill every criterion (for example `"atom_01": true`) after
  checking the transcript and video, set `"adjudication_status": "COMPLETE"`, and rerun the
  same command with:

  ```bash
  ./launch/interactiveworld.sh ...same options... --resume \
      --adjudication <output-dir>/evidence/adjudicated
  ```

  Completed tasks are not re-simulated; the release score is added to `report.json`.

A run is `VALID` only when the simulator, the Human and the Assistant transport all
behaved; invalid runs are reported, never scored as failures.

## Scaling and cost

* One worker uses about 4 GB of GPU memory and several CPU cores; a task takes about 5–20
  minutes depending on its length and on model latency.
* Use `--num-gpus` up to the number of GPUs; a model server can share a GPU with a worker.
* `--resume` makes long runs restartable.

## Custom tasks

`interactiveworld.sh` runs the frozen tasks. Evaluate your own TaskPacks with
`./launch/run_taskpack.sh --taskpack FILE --taskpack-sha256 sha256:…`; see
[TASKPACK.md](TASKPACK.md#evaluate-models-on-it).

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Import-safe, high-level Python API for the InteractiveWorld simulator."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from .actions import Action, action_capabilities, ActionSpec
from .catalog import (
    RUNTIME_CONTRACT_VERSION,
    TaskCatalog,
    TaskInfo,
    TASKPACK_VERSION,
    TaskSpec,
)
from .client import SimulatorClient
from .config import SimulatorConfig
from .errors import SimulatorConfigError, SimulatorError, SimulatorProtocolError
from .protocol import PROTOCOL_VERSION, REQUIRED_CLIENT_OPERATIONS
from .types import Observation, Score, SimulatorInfo, StepResult


TaskInput = str | TaskSpec | Mapping[str, Any]


@dataclass(frozen=True)
class TaskPreparation:
    """Result of applying a TaskPack's authored handoff to an open world."""

    steps: tuple[StepResult, ...]
    observation: Observation
    score_baseline: Score


def _selected_task(
    task: TaskInput | None,
    task_id: str | None,
) -> TaskInput | None:
    if task is not None and task_id is not None:
        raise SimulatorConfigError("provide task or task_id, not both")
    return task if task is not None else task_id


def _initial_target(
    task: TaskInput | None,
    task_id: str | None,
    scene: str | None,
) -> tuple[TaskInput | None, str | None]:
    selected = _selected_task(task, task_id)
    if selected is not None and scene is not None:
        raise SimulatorConfigError("provide a task or scene, not both")
    return selected, _validated_world(scene) if scene is not None else None


def available_worlds() -> tuple[str, ...]:
    """List bundled physical scenes that passed the offline admission checks."""
    from .world_catalog.scene_manifest import admitted_scenes

    return tuple(admitted_scenes())


def _validated_world(scene: object) -> str:
    if not isinstance(scene, str) or not scene.strip():
        raise SimulatorConfigError("scene must be a non-empty string")
    normalized = scene.strip()
    if normalized not in frozenset(available_worlds()):
        raise SimulatorConfigError(
            f"scene {normalized!r} is not an admitted InteractiveWorld world"
        )
    return normalized


def _release_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _server_command(config: SimulatorConfig, socket_path: Path) -> list[str]:
    root = _release_root()
    launcher = root / "launch" / "run_isaac.sh"
    server = root / "launch" / "run_sim_server.py"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise SimulatorConfigError(f"Isaac launcher is not executable: {launcher}")
    if not server.is_file():
        raise SimulatorConfigError(f"simulator server entry point is missing: {server}")
    return [
        str(launcher),
        str(server),
        "--sock",
        str(socket_path),
        "--gpu",
        str(config.gpu),
        "--seed",
        str(config.seed),
        "--observation-profile",
        config.observation_profile,
        "--render-obs" if config.render_observations else "--no-render-obs",
        "--strict-acceptance" if config.strict_acceptance else "--no-strict-acceptance",
    ]


def _launch_environment(config: SimulatorConfig) -> dict[str, str]:
    environment = config.child_environment()
    isaac_environment = environment.get("ISAAC_ENV_DIR") or os.environ.get(
        "ISAAC_ENV_DIR"
    )
    if isaac_environment:
        environment_path = Path(isaac_environment).expanduser()
        if not environment_path.is_absolute() or not environment_path.is_dir():
            raise SimulatorConfigError(
                f"ISAAC_ENV_DIR is not an absolute directory: {environment_path}"
            )
    if config.isaac_python is None and config.isaac_env_dir is not None:
        environment["ISAAC_PYTHON"] = str(Path(config.isaac_env_dir) / "bin" / "python")
    isaac_python = environment.get("ISAAC_PYTHON") or os.environ.get("ISAAC_PYTHON")
    if not isaac_python:
        raise SimulatorConfigError(
            "launch requires SimulatorConfig(isaac_python=...), "
            "SimulatorConfig(isaac_env_dir=...), or ISAAC_PYTHON"
        )
    executable = Path(isaac_python).expanduser()
    if not executable.is_absolute():
        raise SimulatorConfigError("ISAAC_PYTHON must be an absolute path")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SimulatorConfigError(f"ISAAC_PYTHON is not executable: {executable}")
    environment["ISAAC_PYTHON"] = str(executable)
    return environment


def _socket_path(
    config: SimulatorConfig,
    requested: str | os.PathLike[str] | None,
) -> Path:
    if requested is None:
        directory = config.resolved_output_dir / "sockets"
        path = directory / f"sim-{os.getpid()}-{uuid.uuid4().hex[:12]}.sock"
    else:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            raise SimulatorConfigError("unix_socket must be an absolute path")
    resolved = path.resolve()
    release_root = _release_root().resolve()
    if resolved == release_root or resolved.is_relative_to(release_root):
        raise SimulatorConfigError("runtime sockets must be outside the release tree")
    if path.exists():
        raise SimulatorConfigError(f"unix_socket already exists: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path_size = len(os.fsencode(path))
    if path_size >= 104:
        raise SimulatorConfigError(
            f"Unix socket path is too long ({path_size} bytes): {path}"
        )
    return path


class Simulator:
    """One simulator session backed by an isolated or attached server process.

    The caller never imports Isaac Sim. ``launch`` owns its subprocess, so
    ``close`` stops it. A simulator created with ``connect_unix`` or
    ``connect_tcp`` is only detached by ``close``; use ``shutdown`` when the
    remote server should intentionally stop.
    """

    def __init__(
        self,
        client: SimulatorClient,
        *,
        config: SimulatorConfig | None = None,
        catalog: TaskCatalog | None = None,
    ) -> None:
        self._client = client
        self._config = config or SimulatorConfig()
        self._catalog = catalog
        self._current_task: TaskSpec | None = None
        self._current_scene: str | None = None
        self._last_observation: Observation | None = None
        self._task_prepared = False
        self._actions_since_open = 0
        self._score_baseline_captured = False

    @classmethod
    def launch(
        cls,
        task: TaskInput | None = None,
        *,
        task_id: str | None = None,
        scene: str | None = None,
        goals: Sequence[Mapping[str, Any]] | None = None,
        taskpack_digest: str | None = None,
        config: SimulatorConfig | None = None,
        catalog: TaskCatalog | None = None,
        unix_socket: str | os.PathLike[str] | None = None,
    ) -> Simulator:
        """Start an owned local simulator and optionally open a task or world.

        ``task`` accepts a bundled task ID, a verified :class:`TaskSpec`, or an
        inline TaskPack mapping. Inline mappings require ``taskpack_digest``.
        ``task_id`` is a readable keyword alias for a string ``task``.
        """
        selected, selected_scene = _initial_target(task, task_id, scene)
        if goals is not None and selected_scene is None:
            raise SimulatorConfigError("goals are only valid with scene")
        if taskpack_digest is not None and selected is None:
            raise SimulatorConfigError(
                "taskpack_digest is only valid with an inline TaskPack mapping"
            )
        selected_config = config or SimulatorConfig()
        socket_path = _socket_path(selected_config, unix_socket)
        environment = _launch_environment(selected_config)
        client = SimulatorClient.spawn(
            _server_command(selected_config, socket_path),
            unix_socket=socket_path,
            cwd=_release_root(),
            env=environment,
            connect_timeout_s=selected_config.connect_timeout_s,
            request_timeout_s=selected_config.step_timeout_s,
            open_timeout_s=selected_config.open_timeout_s,
        )
        simulator = cls(client, config=selected_config, catalog=catalog)
        try:
            simulator._negotiate()
            if selected is not None:
                simulator.open_task(selected, taskpack_digest=taskpack_digest)
            elif selected_scene is not None:
                simulator.open_world(selected_scene, goals=goals)
        except BaseException:
            # Opening can take minutes; always reap an owned child on failure or cancel.
            simulator.close()
            raise
        return simulator

    @classmethod
    def connect_unix(
        cls,
        path: str | os.PathLike[str],
        *,
        config: SimulatorConfig | None = None,
        catalog: TaskCatalog | None = None,
    ) -> Simulator:
        """Attach to an existing local server without taking ownership of it."""
        selected_config = config or SimulatorConfig()
        client = SimulatorClient.connect_unix(
            path,
            connect_timeout_s=selected_config.connect_timeout_s,
            request_timeout_s=selected_config.step_timeout_s,
            open_timeout_s=selected_config.open_timeout_s,
        )
        simulator = cls(client, config=selected_config, catalog=catalog)
        try:
            simulator._negotiate()
        except BaseException:
            simulator.detach()
            raise
        return simulator

    @classmethod
    def connect_tcp(
        cls,
        host: str,
        port: int,
        *,
        config: SimulatorConfig | None = None,
        catalog: TaskCatalog | None = None,
    ) -> Simulator:
        """Attach to an existing TCP server without taking ownership of it."""
        selected_config = config or SimulatorConfig()
        client = SimulatorClient.connect_tcp(
            host,
            port,
            connect_timeout_s=selected_config.connect_timeout_s,
            request_timeout_s=selected_config.step_timeout_s,
            open_timeout_s=selected_config.open_timeout_s,
        )
        simulator = cls(client, config=selected_config, catalog=catalog)
        try:
            simulator._negotiate()
        except BaseException:
            simulator.detach()
            raise
        return simulator

    @property
    def config(self) -> SimulatorConfig:
        return self._config

    @property
    def endpoint(self) -> str:
        return self._client.endpoint

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected

    @property
    def owns_process(self) -> bool:
        return self._client.owns_process

    @property
    def current_task(self) -> TaskSpec | None:
        return self._current_task

    @property
    def current_scene(self) -> str | None:
        return self._current_scene

    @property
    def last_observation(self) -> Observation | None:
        return self._last_observation

    @property
    def catalog(self) -> TaskCatalog:
        if self._catalog is None:
            self._catalog = TaskCatalog.bundled()
        return self._catalog

    def list_tasks(
        self,
        *,
        category: str | None = None,
        source_family: str | None = None,
    ) -> tuple[TaskInfo, ...]:
        return self.catalog.list(category=category, source_family=source_family)

    @staticmethod
    def list_worlds() -> tuple[str, ...]:
        return available_worlds()

    def describe(self) -> SimulatorInfo:
        return SimulatorInfo.from_payload(self._client.describe())

    def _negotiate(self) -> SimulatorInfo:
        info = self.describe()
        if info.protocol_version != PROTOCOL_VERSION:
            raise SimulatorProtocolError(
                "simulator protocol mismatch: expected "
                f"{PROTOCOL_VERSION!r}, got {info.protocol_version!r}"
            )
        if TASKPACK_VERSION not in info.taskpack_versions:
            raise SimulatorProtocolError(
                f"simulator does not support TaskPack contract {TASKPACK_VERSION!r}"
            )
        if RUNTIME_CONTRACT_VERSION not in info.runtime_contract_versions:
            raise SimulatorProtocolError(
                "simulator does not support runtime contract "
                f"{RUNTIME_CONTRACT_VERSION!r}"
            )
        missing = sorted(REQUIRED_CLIENT_OPERATIONS.difference(info.operations))
        if missing:
            raise SimulatorProtocolError(
                "simulator is missing required operations: " + ", ".join(missing)
            )
        if info.observation_profile != self._config.observation_profile:
            raise SimulatorProtocolError(
                "simulator observation profile mismatch: expected "
                f"{self._config.observation_profile!r}, got "
                f"{info.observation_profile!r}"
            )
        return info

    def actions(self, *, include_unavailable: bool = False) -> tuple[ActionSpec, ...]:
        return action_capabilities(include_unavailable=include_unavailable)

    def open_world(
        self,
        scene: str,
        *,
        seed: int | None = None,
        goals: Sequence[Mapping[str, Any]] | None = None,
    ) -> Observation:
        """Open a physical scene directly, without requiring a TaskPack."""
        expected_scene = _validated_world(scene)
        raw = self._client.open_world(
            expected_scene,
            seed=self._config.seed if seed is None else seed,
            goals=goals,
        )
        observation = self._observation(raw, validate_current=False)
        self._validate_world_identity(expected_scene, observation)
        self._current_task = None
        self._current_scene = expected_scene
        self._reset_episode_state()
        return observation

    def switch_world(
        self,
        scene: str,
        *,
        seed: int | None = None,
        goals: Sequence[Mapping[str, Any]] | None = None,
    ) -> Observation:
        """Reuse the process while rebuilding it with another physical scene."""
        expected_scene = _validated_world(scene)
        raw = self._client.switch_world(
            expected_scene,
            seed=self._config.seed if seed is None else seed,
            goals=goals,
        )
        observation = self._observation(raw, validate_current=False)
        self._validate_world_identity(expected_scene, observation)
        self._current_task = None
        self._current_scene = expected_scene
        self._reset_episode_state()
        return observation

    def open_task(
        self,
        task: TaskInput | None = None,
        *,
        task_id: str | None = None,
        taskpack_digest: str | None = None,
        seed: int | None = None,
    ) -> Observation:
        """Open a bundled or custom TaskPack and return its first observation."""
        selected = _selected_task(task, task_id)
        if selected is None:
            raise SimulatorConfigError("open_task requires task or task_id")
        spec, payload = self._resolve_task(selected, taskpack_digest)
        raw = self._client.open_task(
            **payload,
            seed=self._config.seed if seed is None else seed,
            observation_profile=self._config.observation_profile,
        )
        observation = self._observation(raw, validate_current=False)
        self._validate_task_identity(spec, observation)
        self._current_task = spec
        self._current_scene = spec.scene
        self._reset_episode_state()
        return observation

    def switch_task(
        self,
        task: TaskInput | None = None,
        *,
        task_id: str | None = None,
        taskpack_digest: str | None = None,
        seed: int | None = None,
    ) -> Observation:
        """Switch a warm simulator to another verified TaskPack."""
        selected = _selected_task(task, task_id)
        if selected is None:
            raise SimulatorConfigError("switch_task requires task or task_id")
        spec, payload = self._resolve_task(selected, taskpack_digest)
        raw = self._client.switch_task(
            **payload,
            seed=self._config.seed if seed is None else seed,
            observation_profile=self._config.observation_profile,
        )
        observation = self._observation(raw, validate_current=False)
        self._validate_task_identity(spec, observation)
        self._current_task = spec
        self._current_scene = spec.scene
        self._reset_episode_state()
        return observation

    def reset(self, *, start_xy: Sequence[float] | None = None) -> Observation:
        observation = self._observation(self._client.reset(start_xy=start_xy))
        self._reset_episode_state()
        return observation

    def prepare_task(self) -> TaskPreparation:
        """Apply the current TaskPack's prologue and begin scoring afterward.

        This transition is explicit because a raw authored world is useful for
        simulator exploration, while benchmark-compatible evaluation begins only
        after the authored setup actions and score-baseline capture.
        """
        task = self._current_task
        if task is None:
            raise SimulatorError("prepare_task requires an open TaskPack")
        if getattr(task, "preparation_mode", "prologue") == "continuous_watch":
            raise SimulatorError(
                "this continuous WATCH task has an interleaved human opening; "
                "run it through the benchmark interaction runner instead of prepare_task()"
            )
        if self._task_prepared:
            raise SimulatorError("this TaskPack has already been prepared")
        if self._score_baseline_captured:
            raise SimulatorError("this TaskPack already has a score baseline")
        if self._actions_since_open:
            raise SimulatorError(
                "prepare_task must run before any user action; reset the task first"
            )
        steps: list[StepResult] = []
        for index, command in enumerate(self._prologue_actions(task), start=1):
            result = self._step(command, validate_action=False, track_action=False)
            if not result.success:
                reason = result.error_code or "ACTION_FAILED"
                detail = result.error_message or "simulator rejected the action"
                self.reset()
                raise SimulatorError(
                    f"TaskPack prologue action {index} {command!r} failed "
                    f"({reason}): {detail}; the task was reset"
                )
            steps.append(result)
        baseline = self.capture_score_baseline()
        if baseline.success or baseline.n_done > 0 or not baseline.prohibitions_ok:
            self.reset()
            raise SimulatorError(
                "TaskPack preparation reached an already-scored or prohibited state; "
                "the task was reset"
            )
        observation = self.observe()
        self._task_prepared = True
        return TaskPreparation(tuple(steps), observation, baseline)

    def step(
        self,
        action: Action | str,
        *,
        trace_event: Mapping[str, Any] | None = None,
    ) -> StepResult:
        return self._step(
            action,
            validate_action=True,
            track_action=True,
            trace_event=trace_event,
        )

    def _step(
        self,
        action: Action | str,
        *,
        validate_action: bool,
        track_action: bool,
        trace_event: Mapping[str, Any] | None = None,
    ) -> StepResult:
        if isinstance(action, Action):
            command = action.to_command()
        elif validate_action:
            command = Action.parse(action).to_command()
        elif isinstance(action, str) and action.strip():
            command = action.strip()
        else:
            raise SimulatorConfigError("action must be a non-empty string or Action")
        payload = self._client.step(command, trace_event=trace_event)
        result = StepResult.from_payload(
            payload,
            profile=self._config.observation_profile,
            meta=self._client.last_meta,
        )
        self._validate_current_identity(result.observation)
        self._last_observation = result.observation
        if track_action:
            self._actions_since_open += 1
        return result

    def run(
        self,
        actions: Iterable[Action | str],
        *,
        stop_on_termination: bool = True,
    ) -> tuple[StepResult, ...]:
        """Execute actions in order and optionally stop when the task succeeds."""
        results: list[StepResult] = []
        for action in actions:
            result = self.step(action)
            results.append(result)
            if stop_on_termination and result.terminated:
                break
        return tuple(results)

    def observe(self) -> Observation:
        return self._observation(self._client.observe())

    def render(self) -> bytes | None:
        """Return the latest rendered PNG, or ``None`` when rendering is disabled."""
        observation = Observation.from_payload(
            self._client.render(),
            profile=self._config.observation_profile,
        )
        if observation.frame_png is not None:
            return observation.frame_png
        return observation.frames_png[-1] if observation.frames_png else None

    def score(self) -> Score:
        return Score.from_payload(self._client.score())

    def capture_score_baseline(self) -> Score:
        """Start subsequent task scoring from the world's current state."""
        if self._current_task is not None and self._score_baseline_captured:
            raise SimulatorError("this TaskPack already has a score baseline")
        score = Score.from_payload(self._client.capture_score_baseline())
        if self._current_task is not None:
            self._score_baseline_captured = True
        return score

    def reward(self) -> dict[str, Any]:
        return dict(self._client.reward())

    def detach(self) -> None:
        self._client.detach()

    def shutdown(self) -> Any:
        return self._client.shutdown()

    def close(self) -> None:
        self._client.close()

    def _observation(
        self,
        payload: Mapping[str, Any],
        *,
        validate_current: bool = True,
    ) -> Observation:
        observation = Observation.from_payload(
            payload,
            profile=self._config.observation_profile,
        )
        if validate_current:
            self._validate_current_identity(observation)
        self._last_observation = observation
        return observation

    def _validate_task_identity(self, task: TaskSpec, observation: Observation) -> None:
        if observation.task_id != task.task_id:
            self.close()
            raise SimulatorProtocolError(
                f"server opened task {observation.task_id!r}, expected {task.task_id!r}"
            )
        if observation.taskpack_digest != task.taskpack_digest:
            self.close()
            raise SimulatorProtocolError(
                "server TaskPack digest does not match the locally verified document"
            )
        if observation.scene != task.scene:
            self.close()
            raise SimulatorProtocolError(
                f"server opened scene {observation.scene!r}, expected {task.scene!r}"
            )

    def _validate_current_identity(self, observation: Observation) -> None:
        if self._current_task is not None:
            self._validate_task_identity(self._current_task, observation)
        elif self._current_scene is not None:
            self._validate_world_identity(self._current_scene, observation)

    def _validate_world_identity(
        self,
        scene: str,
        observation: Observation,
    ) -> None:
        if observation.scene != scene:
            self.close()
            raise SimulatorProtocolError(
                f"server reported scene {observation.scene!r}, expected {scene!r}"
            )
        if observation.task_id or observation.taskpack_digest is not None:
            self.close()
            raise SimulatorProtocolError(
                "server reported TaskPack identity for a raw-world session"
            )

    def _reset_episode_state(self) -> None:
        self._task_prepared = False
        self._actions_since_open = 0
        self._score_baseline_captured = False

    def _resolve_task(
        self,
        task: TaskInput,
        taskpack_digest: str | None,
    ) -> tuple[TaskSpec, dict[str, Any]]:
        if isinstance(task, str):
            if taskpack_digest is not None:
                raise SimulatorConfigError(
                    "taskpack_digest is only valid for an inline TaskPack mapping"
                )
            task_id = task.strip()
            if not task_id:
                raise SimulatorConfigError("task_id must be a non-empty string")
            return self.catalog.get(task_id), {"task_id": task_id}
        if isinstance(task, TaskSpec):
            if taskpack_digest is not None:
                raise SimulatorConfigError(
                    "a TaskSpec already contains its verified digest"
                )
            return task, {
                "taskpack": task.taskpack,
                "taskpack_digest": task.taskpack_digest,
            }
        if not isinstance(task, Mapping):
            raise SimulatorConfigError(
                "task must be a task ID, TaskSpec, or TaskPack mapping"
            )
        if taskpack_digest is None:
            raise SimulatorConfigError(
                "an inline TaskPack mapping requires taskpack_digest"
            )
        spec = TaskSpec.from_mapping(task, expected_sha256=taskpack_digest)
        return spec, {
            "taskpack": spec.taskpack,
            "taskpack_digest": spec.taskpack_digest,
        }

    @staticmethod
    def _prologue_actions(task: TaskSpec) -> tuple[str, ...]:
        taskpack = task.taskpack
        story = taskpack.get("story")
        if not isinstance(story, Mapping):
            raise SimulatorError(f"TaskPack {task.task_id!r} has no valid story")
        prologue = story.get("prologue")
        if not isinstance(prologue, Mapping):
            raise SimulatorError(f"TaskPack {task.task_id!r} has no valid prologue")
        actions = prologue.get("actions", ())
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise SimulatorError(
                f"TaskPack {task.task_id!r} prologue.actions must be an array"
            )
        if any(not isinstance(action, str) or not action.strip() for action in actions):
            raise SimulatorError(
                f"TaskPack {task.task_id!r} prologue contains an invalid action"
            )
        return tuple(action.strip() for action in actions)

    def __enter__(self) -> Simulator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def create_environment(
    *,
    task_id: str,
    seed: int = 0,
    tag: str | None = None,
    active_gpu: int = 0,
    render_observations: bool = True,
    strict_acceptance: bool = False,
) -> Any:
    """Create an unopened in-process environment for legacy integrations.

    New applications should use :meth:`Simulator.launch`, which keeps Isaac and
    the application's Python/CUDA dependencies in separate processes.
    """
    from .environment.isaac_env import IsaacEnv

    return IsaacEnv(
        scene=f"spec:{task_id}",
        seed=seed,
        tag=tag,
        active_gpu=active_gpu,
        render_obs=render_observations,
        strict_acceptance=strict_acceptance,
    )


def install_taskpack(environment: Any, taskpack: Mapping[str, Any]) -> None:
    """Install a TaskPack into a legacy in-process environment before opening."""
    from .environment.runtime_builder_spec import register_runtime_builder_spec
    from .interaction.task_runtime import builder_spec_from_taskpack

    task_id = str(taskpack.get("task_id") or "")
    if not task_id:
        raise ValueError("TaskPack has no task_id")
    builder_spec, builder_digest = builder_spec_from_taskpack(taskpack)
    register_runtime_builder_spec(
        f"spec:{task_id}",
        builder_spec,
        builder_digest,
    )
    environment.builder_spec_digest = builder_digest


__all__ = [
    "Simulator",
    "TaskInput",
    "TaskPreparation",
    "available_worlds",
    "create_environment",
    "install_taskpack",
]

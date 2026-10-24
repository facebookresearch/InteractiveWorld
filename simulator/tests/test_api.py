from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import simulator
from simulator import (
    Action,
    Observation,
    Score,
    Simulator,
    SimulatorConfig,
    StepResult,
    TaskPreparation,
)
from simulator.catalog import TaskCatalog, TaskSpec
from simulator.client import SimulatorClient
from simulator.errors import SimulatorConfigError
from simulator.protocol import PROTOCOL_VERSION, PUBLIC_OPERATIONS


def _observation_payload(
    *,
    completed: bool = False,
    task_id: str = "CUSTOM-1",
    taskpack_digest: str | None = "sha256:" + "2" * 64,
    scene: str = "Rs_int",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "taskpack_digest": taskpack_digest,
        "scene": scene,
        "task": "Put the mug on the table",
        "held": [],
        "at": ["kitchen"],
        "objects": ["mug", "table"],
        "objects_are_visible_only": True,
        "put_verb": "PutOn",
        "task_completed": completed,
    }


def _task_spec(
    task_id: str = "CUSTOM-1",
    *,
    prologue_actions: tuple[str, ...] | None = None,
) -> TaskSpec:
    taskpack: dict[str, object] = {"task_id": task_id}
    if prologue_actions is not None:
        taskpack["story"] = {"prologue": {"actions": list(prologue_actions)}}
    return TaskSpec(
        task_id=task_id,
        taskpack_version="data-generate-taskpack/1.0",
        runtime_contract_version="compiled-runtime-contract/1.8",
        source_sha256="sha256:" + "1" * 64,
        taskpack_digest="sha256:" + "2" * 64,
        builder_spec_digest="sha256:" + "3" * 64,
        scene="Rs_int",
        source_path=None,
        _taskpack_json=json.dumps(taskpack),
        _builder_spec_json='{"scene":"Rs_int"}',
    )


def _client() -> mock.NonCallableMagicMock:
    client = mock.create_autospec(SimulatorClient, instance=True)
    client.endpoint = "/tmp/interactiveworld-test.sock"
    client.is_connected = True
    client.owns_process = False
    client.last_meta = {"phases": {"physics": 0.25}}
    client.describe.return_value = {
        "protocol_version": PROTOCOL_VERSION,
        "simulator_version": "Isaac Sim 5.1",
        "taskpack_versions": ["data-generate-taskpack/1.0"],
        "runtime_contract_versions": ["compiled-runtime-contract/1.8"],
        "operations": list(PUBLIC_OPERATIONS),
        "action_primitives": ["navigate_to"],
        "ready": False,
        "render_observations": False,
        "gpu": 0,
        "observation_profile": "agent",
    }
    return client


class SimulatorApiTest(unittest.TestCase):
    def test_package_exports_are_import_safe(self) -> None:
        self.assertIs(Simulator, simulator.Simulator)
        self.assertIs(Action, simulator.Action)
        self.assertNotIn("simulator.environment.isaac_env", sys.modules)

    def test_open_bundled_task_uses_catalog_and_returns_observation(self) -> None:
        client = _client()
        client.open_task.return_value = _observation_payload(
            task_id="WATCH-28008",
            taskpack_digest="sha256:" + "2" * 64,
        )
        catalog = mock.create_autospec(TaskCatalog, instance=True)
        spec = _task_spec("WATCH-28008")
        catalog.get.return_value = spec
        sim = Simulator(client, config=SimulatorConfig(seed=7), catalog=catalog)

        observation = sim.open_task(task_id="WATCH-28008")

        self.assertIsInstance(observation, Observation)
        self.assertEqual("Put the mug on the table", observation.task)
        self.assertIs(spec, sim.current_task)
        self.assertIs(observation, sim.last_observation)
        catalog.get.assert_called_once_with("WATCH-28008")
        client.open_task.assert_called_once_with(
            task_id="WATCH-28008", seed=7, observation_profile="agent"
        )

    def test_task_spec_is_sent_as_a_verified_inline_snapshot(self) -> None:
        client = _client()
        client.open_task.return_value = _observation_payload()
        spec = _task_spec()
        sim = Simulator(client)

        sim.open_task(spec, seed=11)

        client.open_task.assert_called_once_with(
            taskpack={"task_id": "CUSTOM-1"},
            taskpack_digest=spec.taskpack_digest,
            seed=11,
            observation_profile="agent",
        )

    def test_inline_taskpack_is_locally_verified_before_transport(self) -> None:
        bundled = TaskCatalog.bundled()
        bundled_spec = bundled.get(bundled.list()[0].task_id)
        client = _client()
        client.open_task.return_value = _observation_payload(
            task_id=bundled_spec.task_id,
            taskpack_digest=bundled_spec.taskpack_digest,
            scene=bundled_spec.scene,
        )
        sim = Simulator(client)

        sim.open_task(
            bundled_spec.taskpack,
            taskpack_digest=bundled_spec.taskpack_digest,
        )

        sent = client.open_task.call_args.kwargs
        self.assertEqual(bundled_spec.task_id, sent["taskpack"]["task_id"])
        self.assertEqual(bundled_spec.taskpack_digest, sent["taskpack_digest"])
        self.assertEqual(0, sent["seed"])
        self.assertEqual("agent", sent["observation_profile"])

    def test_inline_taskpack_requires_a_digest(self) -> None:
        sim = Simulator(_client())

        with self.assertRaisesRegex(SimulatorConfigError, "requires taskpack_digest"):
            sim.open_task({"task_id": "UNVERIFIED"})

    def test_step_and_run_translate_actions_and_stop_on_success(self) -> None:
        client = _client()
        first = {
            **_observation_payload(),
            "outcomes": [{"primitive": "Walk", "ok": True, "reason": "", "detail": ""}],
        }
        second = {
            **_observation_payload(completed=True),
            "outcomes": [{"primitive": "Grab", "ok": True, "reason": "", "detail": ""}],
        }
        client.step.side_effect = [first, second]
        sim = Simulator(client)

        results = sim.run([Action.walk("mug"), "[Grab] <mug> (1)", Action.wait()])

        self.assertEqual(2, len(results))
        self.assertTrue(all(isinstance(result, StepResult) for result in results))
        self.assertFalse(results[0].terminated)
        self.assertTrue(results[1].terminated)
        self.assertEqual({"physics": 0.25}, results[0].timings)
        self.assertEqual(
            [
                mock.call("[Walk] <mug> (1)", trace_event=None),
                mock.call("[Grab] <mug> (1)", trace_event=None),
            ],
            client.step.call_args_list,
        )

    def test_prepare_task_applies_prologue_then_captures_baseline(self) -> None:
        client = _client()
        client.open_task.return_value = _observation_payload()
        spec = _task_spec(prologue_actions=("[Walk] <mug> (1)", "[Grab] <mug> (1)"))
        sim = Simulator(client)
        sim.open_task(spec)
        client.reset_mock()
        client.last_meta = {"phases": {"physics": 0.25}}
        client.step.side_effect = [
            {
                **_observation_payload(),
                "outcomes": [
                    {"primitive": "Walk", "ok": True, "reason": "", "detail": ""}
                ],
            },
            {
                **_observation_payload(),
                "outcomes": [
                    {"primitive": "Grab", "ok": True, "reason": "", "detail": ""}
                ],
            },
        ]
        client.capture_score_baseline.return_value = {
            "success": False,
            "scoreable": True,
            "n_done": 0,
            "n_total": 1,
        }
        client.observe.return_value = {**_observation_payload(), "held": ["mug"]}

        preparation = sim.prepare_task()

        self.assertIsInstance(preparation, TaskPreparation)
        self.assertEqual(2, len(preparation.steps))
        self.assertEqual(("mug",), preparation.observation.held)
        self.assertTrue(preparation.score_baseline.scoreable)
        self.assertEqual(
            [
                mock.call.step("[Walk] <mug> (1)", trace_event=None),
                mock.call.step("[Grab] <mug> (1)", trace_event=None),
                mock.call.capture_score_baseline(),
                mock.call.observe(),
            ],
            client.method_calls,
        )

    def test_prepare_task_stops_before_baseline_when_a_prologue_action_fails(
        self,
    ) -> None:
        client = _client()
        client.open_task.return_value = _observation_payload()
        spec = _task_spec(prologue_actions=("[Grab] <mug> (1)",))
        sim = Simulator(client)
        sim.open_task(spec)
        client.reset_mock()
        client.step.return_value = {
            **_observation_payload(),
            "outcomes": [
                {
                    "primitive": "Grab",
                    "ok": False,
                    "reason": "OUT_OF_REACH",
                    "detail": "walk closer",
                }
            ],
        }
        client.reset.return_value = _observation_payload()

        with self.assertRaisesRegex(
            simulator.SimulatorError, "OUT_OF_REACH.*walk closer"
        ):
            sim.prepare_task()

        client.capture_score_baseline.assert_not_called()
        client.observe.assert_not_called()
        client.reset.assert_called_once_with(start_xy=None)

    def test_prepare_task_requires_an_open_task(self) -> None:
        with self.assertRaisesRegex(simulator.SimulatorError, "requires an open"):
            Simulator(_client()).prepare_task()

    def test_prepare_task_validates_the_captured_baseline_then_resets(self) -> None:
        client = _client()
        spec = _task_spec(prologue_actions=())
        client.open_task.return_value = _observation_payload()
        client.capture_score_baseline.return_value = {
            "success": False,
            "scoreable": True,
            "n_done": 1,
            "n_total": 2,
            "prohibitions_ok": True,
        }
        client.reset.return_value = _observation_payload()
        sim = Simulator(client)
        sim.open_task(spec)

        with self.assertRaisesRegex(simulator.SimulatorError, "already-scored"):
            sim.prepare_task()

        client.capture_score_baseline.assert_called_once_with()
        client.reset.assert_called_once_with(start_xy=None)
        client.observe.assert_not_called()

    def test_observe_render_score_and_reward_are_typed(self) -> None:
        client = _client()
        client.observe.return_value = {
            **_observation_payload(),
            "relations": [["ON", "mug", "table"]],
            "score": {"success": False, "scoreable": True, "n_total": 1},
        }
        frames = [base64.b64encode(value).decode("ascii") for value in (b"one", b"two")]
        client.render.return_value = {"frames": frames}
        client.score.return_value = {
            "success": True,
            "scoreable": True,
            "n_done": 1,
            "n_total": 1,
            "progress": 1.0,
        }
        client.reward.return_value = {"reward": 1.0, "success": True}
        sim = Simulator(client, config=SimulatorConfig(observation_profile="debug"))

        observation = sim.observe()
        rendered = sim.render()
        score = sim.score()
        reward = sim.reward()

        self.assertEqual("ON", observation.relations[0].predicate)
        self.assertIsInstance(observation.score, Score)
        self.assertEqual(b"two", rendered)
        self.assertIsInstance(score, Score)
        self.assertTrue(score.success)
        self.assertEqual({"reward": 1.0, "success": True}, reward)

    def test_switch_reset_describe_and_catalog_listing_delegate(self) -> None:
        client = _client()
        client.switch_task.return_value = _observation_payload(task_id="WATCH-2")
        client.reset.return_value = _observation_payload(task_id="WATCH-2")
        client.describe.return_value = {
            "protocol_version": "interactiveworld-sim/1.0",
            "simulator_version": "Isaac Sim 5.1",
            "taskpack_versions": ["data-generate-taskpack/1.0"],
            "operations": ["step"],
            "action_primitives": ["Walk"],
        }
        catalog = mock.create_autospec(TaskCatalog, instance=True)
        spec = _task_spec("WATCH-2")
        catalog.get.return_value = spec
        catalog.list.return_value = ()
        sim = Simulator(client, catalog=catalog)

        switched = sim.switch_task("WATCH-2", seed=3)
        reset = sim.reset(start_xy=(1.0, 2.0))
        info = sim.describe()
        tasks = sim.list_tasks(category="Accessibility")

        self.assertIsInstance(switched, Observation)
        self.assertIsInstance(reset, Observation)
        self.assertEqual("interactiveworld-sim/1.0", info.protocol_version)
        self.assertEqual((), tasks)
        client.switch_task.assert_called_once_with(
            task_id="WATCH-2", seed=3, observation_profile="agent"
        )
        client.reset.assert_called_once_with(start_xy=(1.0, 2.0))
        catalog.list.assert_called_once_with(
            category="Accessibility", source_family=None
        )

    @mock.patch.object(SimulatorClient, "connect_tcp")
    @mock.patch.object(SimulatorClient, "connect_unix")
    def test_connectors_forward_configured_timeouts(
        self,
        connect_unix: mock.Mock,
        connect_tcp: mock.Mock,
    ) -> None:
        unix_client = _client()
        tcp_client = _client()
        connect_unix.return_value = unix_client
        connect_tcp.return_value = tcp_client
        unix_client.endpoint = "/tmp/test.sock"
        tcp_client.endpoint = "[127.0.0.1]:9000"
        config = SimulatorConfig(
            connect_timeout_s=4.0,
            step_timeout_s=5.0,
            open_timeout_s=6.0,
        )

        unix_sim = Simulator.connect_unix("/tmp/test.sock", config=config)
        tcp_sim = Simulator.connect_tcp("127.0.0.1", 9000, config=config)

        self.assertEqual("/tmp/test.sock", unix_sim.endpoint)
        self.assertEqual("[127.0.0.1]:9000", tcp_sim.endpoint)
        connect_unix.assert_called_once_with(
            "/tmp/test.sock",
            connect_timeout_s=4.0,
            request_timeout_s=5.0,
            open_timeout_s=6.0,
        )
        connect_tcp.assert_called_once_with(
            "127.0.0.1",
            9000,
            connect_timeout_s=4.0,
            request_timeout_s=5.0,
            open_timeout_s=6.0,
        )

    @mock.patch.object(SimulatorClient, "spawn")
    def test_launch_builds_an_isolated_unix_command(
        self,
        spawn: mock.Mock,
    ) -> None:
        client = _client()
        spawn.return_value = client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "isaac311"
            python = environment / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o700)
            socket_path = root / "sim.sock"
            config = SimulatorConfig(
                gpu=2,
                seed=9,
                render_observations=False,
                strict_acceptance=False,
                isaac_env_dir=environment,
                output_dir=root,
                connect_timeout_s=4.0,
                step_timeout_s=5.0,
                open_timeout_s=6.0,
            )

            sim = Simulator.launch(config=config, unix_socket=socket_path)

        self.assertEqual("/tmp/interactiveworld-test.sock", sim.endpoint)
        command = spawn.call_args.args[0]
        kwargs = spawn.call_args.kwargs
        self.assertTrue(str(command[0]).endswith("/launch/run_isaac.sh"))
        self.assertTrue(str(command[1]).endswith("/launch/run_sim_server.py"))
        self.assertIn("--no-render-obs", command)
        self.assertIn("--no-strict-acceptance", command)
        self.assertEqual("2", command[command.index("--gpu") + 1])
        self.assertEqual(str(python), kwargs["env"]["ISAAC_PYTHON"])
        self.assertEqual(socket_path, kwargs["unix_socket"])
        self.assertEqual(4.0, kwargs["connect_timeout_s"])
        self.assertEqual(5.0, kwargs["request_timeout_s"])
        self.assertEqual(6.0, kwargs["open_timeout_s"])

    @mock.patch.object(SimulatorClient, "spawn")
    def test_launch_closes_owned_client_when_initial_open_fails(
        self,
        spawn: mock.Mock,
    ) -> None:
        client = _client()
        client.open_task.side_effect = RuntimeError("open failed")
        spawn.return_value = client
        catalog = mock.create_autospec(TaskCatalog, instance=True)
        catalog.get.return_value = _task_spec("WATCH-FAIL")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o700)
            config = SimulatorConfig(isaac_python=python, output_dir=root)

            with self.assertRaisesRegex(RuntimeError, "open failed"):
                Simulator.launch(
                    task_id="WATCH-FAIL",
                    config=config,
                    catalog=catalog,
                    unix_socket=root / "sim.sock",
                )

        client.close.assert_called_once_with()

    @mock.patch.object(SimulatorClient, "spawn")
    def test_launch_can_open_a_world_without_a_taskpack(self, spawn: mock.Mock) -> None:
        client = _client()
        client.open_world.return_value = _observation_payload(
            task_id="", taskpack_digest=None, scene="Rs_int"
        )
        spawn.return_value = client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o700)
            config = SimulatorConfig(isaac_python=python, output_dir=root, seed=4)

            sim = Simulator.launch(
                scene="Rs_int",
                config=config,
                unix_socket=root / "sim.sock",
            )

        self.assertEqual("Rs_int", sim.current_scene)
        self.assertIsNone(sim.current_task)
        client.open_world.assert_called_once_with("Rs_int", seed=4, goals=None)

    @mock.patch.object(SimulatorClient, "spawn")
    def test_launch_rejects_an_unknown_world_before_spawning(
        self, spawn: mock.Mock
    ) -> None:
        with self.assertRaisesRegex(SimulatorConfigError, "not an admitted"):
            Simulator.launch(scene="not_a_release_world")

        spawn.assert_not_called()

    @mock.patch.object(SimulatorClient, "spawn")
    def test_launch_refuses_to_replace_an_existing_socket(
        self, spawn: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "sim.sock"
            socket_path.write_text("owned by another process", encoding="utf-8")

            with self.assertRaisesRegex(SimulatorConfigError, "already exists"):
                Simulator.launch(unix_socket=socket_path)

        spawn.assert_not_called()

    def test_context_close_detach_and_shutdown_delegate(self) -> None:
        context_client = _client()
        detached_client = _client()
        shutdown_client = _client()
        shutdown_client.shutdown.return_value = "shutting_down"

        with Simulator(context_client) as sim:
            self.assertEqual("/tmp/interactiveworld-test.sock", sim.endpoint)
        Simulator(detached_client).detach()
        result = Simulator(shutdown_client).shutdown()

        self.assertEqual("shutting_down", result)
        context_client.close.assert_called_once_with()
        detached_client.detach.assert_called_once_with()
        shutdown_client.shutdown.assert_called_once_with()

    def test_rejects_conflicting_task_arguments(self) -> None:
        sim = Simulator(_client())

        with self.assertRaisesRegex(SimulatorConfigError, "not both"):
            sim.open_task("WATCH-1", task_id="WATCH-2")

    def test_open_and_switch_world_do_not_require_a_taskpack(self) -> None:
        client = _client()
        client.open_world.return_value = _observation_payload(
            task_id="", taskpack_digest=None, scene="Rs_int"
        )
        client.switch_world.return_value = _observation_payload(
            task_id="", taskpack_digest=None, scene="Wainscott_0_int"
        )
        sim = Simulator(client, config=SimulatorConfig(seed=9))

        opened = sim.open_world("Rs_int")
        switched = sim.switch_world("Wainscott_0_int", seed=12)

        self.assertEqual("Rs_int", opened.scene)
        self.assertEqual("Wainscott_0_int", switched.scene)
        self.assertEqual("Wainscott_0_int", sim.current_scene)
        self.assertIsNone(sim.current_task)
        client.open_world.assert_called_once_with("Rs_int", seed=9, goals=None)
        client.switch_world.assert_called_once_with(
            "Wainscott_0_int", seed=12, goals=None
        )

    def test_open_world_rejects_an_unknown_name_before_contacting_server(self) -> None:
        client = _client()
        sim = Simulator(client)

        with self.assertRaisesRegex(SimulatorConfigError, "not an admitted"):
            sim.open_world("rs_int")

        client.open_world.assert_not_called()

    def test_raw_world_ongoing_identity_rejects_taskpack_fields(self) -> None:
        client = _client()
        client.open_world.return_value = _observation_payload(
            task_id="", taskpack_digest=None, scene="Rs_int"
        )
        client.observe.return_value = _observation_payload(
            task_id="LEAKED-TASK",
            taskpack_digest="sha256:" + "9" * 64,
            scene="Rs_int",
        )
        sim = Simulator(client)
        sim.open_world("Rs_int")

        with self.assertRaisesRegex(
            simulator.SimulatorProtocolError, "TaskPack identity"
        ):
            sim.observe()

        client.close.assert_called_once_with()

    def test_task_identity_mismatch_closes_the_session(self) -> None:
        client = _client()
        client.open_task.return_value = _observation_payload(
            task_id="OTHER", taskpack_digest="sha256:" + "9" * 64
        )
        spec = _task_spec("EXPECTED")
        sim = Simulator(client)

        with self.assertRaisesRegex(
            simulator.SimulatorProtocolError, "server opened task"
        ):
            sim.open_task(spec)

        client.close.assert_called_once_with()
        self.assertIsNone(sim.current_task)

    def test_later_observation_identity_is_also_verified(self) -> None:
        client = _client()
        spec = _task_spec("EXPECTED")
        client.open_task.return_value = _observation_payload(task_id="EXPECTED")
        client.observe.return_value = _observation_payload(
            task_id="EXPECTED", scene="OTHER_SCENE"
        )
        sim = Simulator(client)
        sim.open_task(spec)

        with self.assertRaisesRegex(
            simulator.SimulatorProtocolError, "server opened scene"
        ):
            sim.observe()

        client.close.assert_called_once_with()

    def test_prepare_task_is_single_use_and_precedes_user_actions(self) -> None:
        client = _client()
        spec = _task_spec(prologue_actions=())
        client.open_task.return_value = _observation_payload()
        client.score.return_value = {
            "success": False,
            "scoreable": True,
            "n_done": 0,
            "n_total": 1,
            "prohibitions_ok": True,
        }
        client.capture_score_baseline.return_value = client.score.return_value
        client.observe.return_value = _observation_payload()
        client.step.return_value = {
            **_observation_payload(),
            "outcomes": [{"primitive": "WAIT", "ok": True, "reason": "", "detail": ""}],
        }
        sim = Simulator(client)
        sim.open_task(spec)

        sim.prepare_task()
        with self.assertRaisesRegex(simulator.SimulatorError, "already been prepared"):
            sim.prepare_task()

        client.reset.return_value = _observation_payload()
        sim.reset()
        sim.step(Action.wait())
        with self.assertRaisesRegex(simulator.SimulatorError, "before any user action"):
            sim.prepare_task()

    def test_prepare_task_rejects_continuous_watch(self) -> None:
        client = _client()
        spec = _task_spec(prologue_actions=("[Wait]",))
        object.__setattr__(spec, "preparation_mode", "continuous_watch")
        client.open_task.return_value = _observation_payload()
        sim = Simulator(client)
        sim.open_task(spec)

        with self.assertRaisesRegex(simulator.SimulatorError, "continuous WATCH"):
            sim.prepare_task()

        client.step.assert_not_called()

    @mock.patch.object(SimulatorClient, "spawn")
    def test_launch_rejects_an_incompatible_server_contract(
        self, spawn: mock.Mock
    ) -> None:
        client = _client()
        client.describe.return_value["protocol_version"] = "interactiveworld-sim/999"
        spawn.return_value = client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o700)
            config = SimulatorConfig(isaac_python=python, output_dir=root)

            with self.assertRaisesRegex(
                simulator.SimulatorProtocolError, "protocol mismatch"
            ):
                Simulator.launch(config=config, unix_socket=root / "sim.sock")

        client.close.assert_called_once_with()

    def test_negotiation_requires_taskpack_and_runtime_contracts(self) -> None:
        for field in ("taskpack_versions", "runtime_contract_versions"):
            with self.subTest(field=field):
                client = _client()
                client.describe.return_value[field] = []
                sim = Simulator(client)

                with self.assertRaisesRegex(
                    simulator.SimulatorProtocolError, "does not support"
                ):
                    sim._negotiate()


class LegacyApiTest(unittest.TestCase):
    def test_create_environment_keeps_the_legacy_constructor_contract(self) -> None:
        isaac_env = mock.Mock()
        module = mock.Mock(IsaacEnv=isaac_env)

        with mock.patch.dict(sys.modules, {"simulator.environment.isaac_env": module}):
            result = simulator.create_environment(
                task_id="WATCH-1",
                seed=5,
                active_gpu=2,
                render_observations=False,
                strict_acceptance=False,
            )

        self.assertIs(isaac_env.return_value, result)
        isaac_env.assert_called_once_with(
            scene="spec:WATCH-1",
            seed=5,
            tag=None,
            active_gpu=2,
            render_obs=False,
            strict_acceptance=False,
        )

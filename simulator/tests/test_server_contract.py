from __future__ import annotations

import json
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from simulator.client import SimulatorClient
from simulator.transport import isaac_sim_server as server


def _args(**overrides: object) -> SimpleNamespace:
    values = {
        "scene": "",
        "seed": 7,
        "tag": None,
        "gpu": 0,
        "render_obs": False,
        "strict_acceptance": True,
        "observation_profile": "agent",
        "debug_errors": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _observation(*, success: bool = False) -> dict[str, object]:
    return {
        "held": [],
        "at": ["counter"],
        "objects": ["cup", "counter"],
        "objects_are_visible_only": True,
        "trace_events": [],
        "relations": [("ON", "cup", "counter")],
        "goal": [("INSIDE", "cup", "cabinet")],
        "score": {
            "success": success,
            "scoreable": True,
            "n_done": int(success),
            "n_total": 1,
        },
        "task": "Put away the cup.",
    }


class _FakeInner:
    reports: list[object] = []


class _FakeEnvironment:
    created: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.created.append((args, kwargs))
        self.scene = str(args[0])
        self.seed = int(kwargs["seed"])
        self.env = _FakeInner()
        self.ctx = {"program": []}
        self.builder_spec_digest: str | None = None

    def open(self) -> dict[str, object]:
        return _observation()

    def reset(self, **_kwargs: object) -> dict[str, object]:
        return _observation()

    def switch_scene(self, scene: str, **kwargs: object) -> dict[str, object]:
        self.scene = scene
        self.seed = int(kwargs["seed"])
        return _observation()

    def step(self, _command: str, **_kwargs: object) -> dict[str, object]:
        return _observation(success=True)

    def last_step_reports(self) -> list[dict[str, object]]:
        return [
            {
                "primitive": "WAIT",
                "ok": True,
                "reason": "SUCCESS",
                "detail": "",
            }
        ]

    def observe(self) -> dict[str, object]:
        return _observation()

    def score(self) -> dict[str, object]:
        return dict(_observation()["score"])

    def capture_score_baseline(self) -> dict[str, object]:
        return dict(_observation()["score"])

    def reward(self) -> dict[str, object]:
        return {"reward": 1.0, "compliance": {}}


class ServerContractTest(unittest.TestCase):
    def test_real_server_process_describes_protocol_without_loading_isaac(self) -> None:
        release_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "server.sock"
            client = SimulatorClient.spawn(
                [
                    sys.executable,
                    str(release_root / "launch" / "run_sim_server.py"),
                    "--sock",
                    str(socket_path),
                    "--no-render-obs",
                ],
                unix_socket=socket_path,
                cwd=release_root,
                connect_timeout_s=15.0,
                shutdown_timeout_s=1.0,
            )
            try:
                description = client.describe()
            finally:
                client.close()

        self.assertEqual(server.PROTOCOL_VERSION, description["protocol_version"])
        self.assertFalse(description["ready"])

    def test_unix_server_survives_detach_and_accepts_a_new_client(self) -> None:
        release_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "server.sock"
            owner = SimulatorClient.spawn(
                [
                    sys.executable,
                    str(release_root / "launch" / "run_sim_server.py"),
                    "--sock",
                    str(socket_path),
                    "--no-render-obs",
                ],
                unix_socket=socket_path,
                cwd=release_root,
                connect_timeout_s=15.0,
                shutdown_timeout_s=2.0,
            )
            try:
                owner.detach()
                attached = SimulatorClient.connect_unix(
                    socket_path, connect_timeout_s=15.0
                )
                self.assertEqual(
                    server.PROTOCOL_VERSION,
                    attached.describe()["protocol_version"],
                )
                attached.shutdown()
                deadline = time.monotonic() + 2.0
                while socket_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(socket_path.exists())
            finally:
                owner.close()

    def test_describe_is_available_before_isaac_is_open(self) -> None:
        reply, environment, should_exit = server.dispatch_request(
            None, {"op": "describe"}, _args()
        )

        self.assertTrue(reply["ok"])
        self.assertEqual(server.PROTOCOL_VERSION, reply["result"]["protocol_version"])
        self.assertIn("open_task", reply["result"]["operations"])
        self.assertIn("open_world", reply["result"]["operations"])
        self.assertEqual("agent", reply["result"]["observation_profile"])
        self.assertIsNone(environment)
        self.assertFalse(should_exit)

    def test_open_world_does_not_require_a_taskpack(self) -> None:
        _FakeEnvironment.created.clear()
        with mock.patch.object(server, "_ENVIRONMENT_TYPE", _FakeEnvironment):
            reply, environment, should_exit = server.dispatch_request(
                None,
                {"op": "open_world", "scene": "Rs_int", "seed": 13},
                _args(),
            )

        self.assertTrue(reply["ok"])
        self.assertEqual("Rs_int", reply["result"]["scene"])
        self.assertNotIn("task_id", reply["result"])
        self.assertEqual("Rs_int", _FakeEnvironment.created[0][0][0])
        self.assertEqual(13, _FakeEnvironment.created[0][1]["seed"])
        self.assertIsNone(environment._public_task_id)
        self.assertFalse(should_exit)

    def test_unknown_world_is_rejected_before_loading_environment(self) -> None:
        environment_type = mock.Mock(side_effect=AssertionError("must not load Isaac"))
        with mock.patch.object(server, "_environment_type", environment_type):
            reply, environment, should_exit = server._execute_request(
                None,
                {"op": "open_world", "scene": "rs_int", "seed": 13},
                _args(),
            )

        self.assertFalse(reply["ok"])
        self.assertEqual("INVALID_ARGUMENT", reply["error_code"])
        self.assertIn("not an admitted", reply["error"])
        self.assertIsNone(environment)
        self.assertFalse(should_exit)
        environment_type.assert_not_called()

    def test_observation_profile_is_locked_by_the_server(self) -> None:
        environment = _FakeEnvironment("Rs_int", seed=0)

        with self.assertRaisesRegex(ValueError, "fixed by the server"):
            server.dispatch_request(
                environment,
                {"op": "observe", "observation_profile": "debug"},
                _args(observation_profile="agent"),
            )

    def test_private_and_legacy_operations_are_not_protocol_operations(self) -> None:
        environment = _FakeEnvironment("spec:test", seed=0)

        for operation in ("program", "open", "switch_scene", "set_goal"):
            with self.subTest(operation=operation):
                reply, _, should_exit = server.dispatch_request(
                    environment, {"op": operation}, _args()
                )
                self.assertFalse(reply["ok"])
                self.assertEqual("UNKNOWN_OPERATION", reply["error_code"])
                self.assertFalse(should_exit)

    def test_runtime_mutation_failure_poison_worker(self) -> None:
        environment = _FakeEnvironment("spec:test", seed=0)
        environment.step = mock.Mock(side_effect=RuntimeError("physics failed"))

        reply, returned, should_exit = server._execute_request(
            environment,
            {"op": "step", "cmd": "[Wait]", "_rid": 5},
            _args(),
        )

        self.assertFalse(reply["ok"])
        self.assertEqual("ENVIRONMENT_FAILED", reply["error_code"])
        self.assertEqual(5, reply["_rid"])
        self.assertIs(environment, returned)
        self.assertTrue(should_exit)

    def test_invalid_mutation_request_does_not_poison_worker(self) -> None:
        environment = _FakeEnvironment("spec:test", seed=0)

        reply, returned, should_exit = server._execute_request(
            environment,
            {"op": "reset", "start_xy": [1.0], "_rid": 6},
            _args(),
        )

        self.assertFalse(reply["ok"])
        self.assertEqual("INVALID_ARGUMENT", reply["error_code"])
        self.assertIs(environment, returned)
        self.assertFalse(should_exit)

    def test_post_step_serialization_failure_poison_worker(self) -> None:
        environment = _FakeEnvironment("spec:test", seed=0)
        environment.last_step_reports = mock.Mock(
            side_effect=RuntimeError("report serialization failed")
        )

        reply, returned, should_exit = server._execute_request(
            environment,
            {"op": "step", "cmd": "[Wait]", "_rid": 7},
            _args(),
        )

        self.assertEqual("ENVIRONMENT_FAILED", reply["error_code"])
        self.assertIs(environment, returned)
        self.assertTrue(should_exit)

    def test_baseline_failure_poison_worker(self) -> None:
        environment = _FakeEnvironment("spec:test", seed=0)
        environment.capture_score_baseline = mock.Mock(
            side_effect=RuntimeError("baseline mutation failed")
        )

        reply, returned, should_exit = server._execute_request(
            environment,
            {"op": "capture_score_baseline", "_rid": 8},
            _args(),
        )

        self.assertEqual("ENVIRONMENT_FAILED", reply["error_code"])
        self.assertIs(environment, returned)
        self.assertTrue(should_exit)

    def test_fatal_send_failure_preserves_exit_request(self) -> None:
        service, client = socket.socketpair()
        self.addCleanup(service.close)
        self.addCleanup(client.close)
        client.sendall(b'{"op":"step","cmd":"[Wait]"}\n')
        environment = _FakeEnvironment("spec:test", seed=0)

        with (
            mock.patch.object(
                server,
                "_execute_request",
                return_value=({"ok": False}, environment, True),
            ),
            mock.patch.object(server, "_try_send", return_value=False),
        ):
            returned, should_exit = server._serve_session(service, environment, _args())

        self.assertIs(environment, returned)
        self.assertTrue(should_exit)

    def test_agent_observation_hides_privileged_state(self) -> None:
        observation = server.serialize_observation(_observation(), profile="agent")

        self.assertNotIn("relations", observation)
        self.assertNotIn("goal", observation)
        self.assertNotIn("score", observation)
        self.assertFalse(observation["task_completed"])

    def test_debug_observation_contains_evaluator_state(self) -> None:
        observation = server.serialize_observation(_observation(), profile="debug")

        self.assertEqual([["ON", "cup", "counter"]], observation["relations"])
        self.assertEqual(1, len(observation["goal"]))
        self.assertEqual(1, observation["score"]["n_total"])

    def test_open_task_resolves_bundled_task_and_sets_identity(self) -> None:
        _FakeEnvironment.created.clear()
        with (
            mock.patch.object(server, "_ENVIRONMENT_TYPE", _FakeEnvironment),
            mock.patch(
                "simulator.environment.runtime_builder_spec.register_runtime_builder_spec"
            ) as register,
        ):
            reply, environment, should_exit = server.dispatch_request(
                None,
                {"op": "open_task", "task_id": "WATCH-28008", "seed": 11},
                _args(),
            )

        self.assertTrue(reply["ok"])
        self.assertEqual("WATCH-28008", reply["result"]["task_id"])
        self.assertEqual("spec:WATCH-28008", _FakeEnvironment.created[0][0][0])
        self.assertEqual(11, _FakeEnvironment.created[0][1]["seed"])
        self.assertEqual("WATCH-28008", environment._public_task_id)
        self.assertTrue(environment.builder_spec_digest.startswith("sha256:"))
        register.assert_called_once()
        self.assertFalse(should_exit)

    def test_agent_step_reports_completion_without_exposing_score(self) -> None:
        environment = _FakeEnvironment("spec:test", seed=0)
        reply, _, should_exit = server.dispatch_request(
            environment, {"op": "step", "cmd": "[Wait]"}, _args()
        )

        self.assertTrue(reply["ok"])
        self.assertTrue(reply["result"]["task_completed"])
        self.assertNotIn("score", reply["result"])
        self.assertEqual("SUCCESS", reply["result"]["outcomes"][0]["reason"])
        self.assertFalse(should_exit)

    def test_close_detaches_and_shutdown_stops_server(self) -> None:
        environment = _FakeEnvironment("spec:test", seed=0)

        close_reply = server.dispatch_request(environment, {"op": "close"}, _args())
        shutdown_reply = server.dispatch_request(
            environment, {"op": "shutdown"}, _args()
        )

        self.assertFalse(close_reply[2])
        self.assertTrue(shutdown_reply[2])

        unopened_shutdown = server.dispatch_request(None, {"op": "shutdown"}, _args())
        self.assertTrue(unopened_shutdown[2])

    def test_session_rejects_non_object_json(self) -> None:
        service, client = socket.socketpair()
        self.addCleanup(service.close)
        self.addCleanup(client.close)
        client.sendall(b"[]\n")
        client.shutdown(socket.SHUT_WR)

        environment, should_exit = server._serve_session(service, None, _args())
        response = json.loads(client.makefile("rb").readline())

        self.assertIsNone(environment)
        self.assertFalse(should_exit)
        self.assertEqual("INVALID_REQUEST", response["error_code"])

    def test_session_returns_correlated_structured_errors_without_trace(self) -> None:
        service, client = socket.socketpair()
        self.addCleanup(service.close)
        self.addCleanup(client.close)
        client.sendall(b'{"op":"open_task","_rid":41}\n')
        client.shutdown(socket.SHUT_WR)

        environment, should_exit = server._serve_session(service, None, _args())
        response = json.loads(client.makefile("rb").readline())

        self.assertIsNone(environment)
        self.assertFalse(should_exit)
        self.assertEqual(41, response["_rid"])
        self.assertEqual("INVALID_ARGUMENT", response["error_code"])
        self.assertNotIn("trace", response)

    def test_remote_bind_requires_explicit_acknowledgement(self) -> None:
        self.assertTrue(server._is_loopback_bind("127.0.0.1"))
        self.assertTrue(server._is_loopback_bind("::1"))
        self.assertTrue(server._is_loopback_bind("localhost"))
        self.assertFalse(server._is_loopback_bind("0.0.0.0"))

    def test_unix_listener_never_replaces_a_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing"
            path.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "existing socket path"):
                server._listener(_args(port=None, sock=str(path)))

            self.assertEqual("keep", path.read_text(encoding="utf-8"))

    def test_unix_listener_never_replaces_an_active_socket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.sock"
            active = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.addCleanup(active.close)
            active.bind(str(path))
            active.listen(1)

            with self.assertRaisesRegex(ValueError, "existing socket path"):
                server._listener(_args(port=None, sock=str(path)))

            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.addCleanup(probe.close)
            probe.connect(str(path))

    def test_session_bounds_request_size(self) -> None:
        service, client = socket.socketpair()
        self.addCleanup(service.close)
        self.addCleanup(client.close)
        client.sendall(b'{"op":"' + b"x" * 64 + b'"}\n')

        with mock.patch.object(server, "_MAX_REQUEST_BYTES", 16):
            environment, should_exit = server._serve_session(service, None, _args())
        response = json.loads(client.makefile("rb").readline())

        self.assertIsNone(environment)
        self.assertFalse(should_exit)
        self.assertEqual("REQUEST_TOO_LARGE", response["error_code"])


__all__ = ["ServerContractTest"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

from simulator.client import SimulatorClient
from simulator.errors import (
    SimulatorConnectionError,
    SimulatorProtocolError,
    SimulatorRemoteError,
    SimulatorTimeoutError,
)


def _receive_request(connection: socket.socket) -> dict[str, Any] | None:
    data = b""
    while b"\n" not in data:
        chunk = connection.recv(65536)
        if not chunk:
            return None
        data += chunk
    request = json.loads(data.partition(b"\n")[0])
    if not isinstance(request, dict):
        raise TypeError("request must be an object")
    return request


def _send_reply(connection: socket.socket, reply: dict[str, Any]) -> None:
    connection.sendall((json.dumps(reply) + "\n").encode("utf-8"))


class _StubServer:
    def __init__(
        self,
        listener: socket.socket,
        handler: Callable[[socket.socket], None],
    ) -> None:
        self.listener = listener
        self.handler = handler
        self.errors: list[Exception] = []
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @classmethod
    def unix(
        cls,
        path: Path,
        handler: Callable[[socket.socket], None],
    ) -> _StubServer:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        return cls(listener, handler)

    @classmethod
    def tcp(
        cls,
        handler: Callable[[socket.socket], None],
    ) -> _StubServer:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        return cls(listener, handler)

    @property
    def port(self) -> int:
        return int(self.listener.getsockname()[1])

    def _serve(self) -> None:
        try:
            connection, _ = self.listener.accept()
            with connection:
                self.handler(connection)
        except Exception as error:
            self.errors.append(error)
        finally:
            self.listener.close()

    def join(self) -> None:
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise AssertionError("stub simulator server did not stop")
        if self.errors:
            raise self.errors[0]


class SimulatorClientTest(unittest.TestCase):
    def test_rejects_unknown_observation_profile_before_transport(self) -> None:
        first, second = socket.socketpair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        client = SimulatorClient(first, endpoint="test")

        with self.assertRaisesRegex(ValueError, "observation_profile"):
            client.open_task(
                task_id="WATCH-28008",
                observation_profile="privileged",
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            client.open_task()
        with self.assertRaisesRegex(ValueError, "requires taskpack_digest"):
            client.open_task(taskpack={"task_id": "CUSTOM-1"})
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            client.open_task(task_id="WATCH-28008", seed=True)
        with self.assertRaisesRegex(ValueError, "pred must be 'inside'"):
            client.open_world(
                "Rs_int",
                goals=[{"pred": "on", "subject": "cup", "target": "table"}],
            )
        with self.assertRaisesRegex(ValueError, "exactly two"):
            client.reset(start_xy=(1.0,))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            client.step("  ")

        client.close()

    def test_high_level_operations_send_the_public_protocol(self) -> None:
        requests: list[dict[str, Any]] = []

        def handle(connection: socket.socket) -> None:
            while request := _receive_request(connection):
                requests.append(request)
                _send_reply(
                    connection,
                    {
                        "_rid": request["_rid"],
                        "ok": True,
                        "result": {"operation": request["op"]},
                        "meta": {"t_op_s": 0.25},
                    },
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.sock"
            server = _StubServer.unix(path, handle)
            client = SimulatorClient.connect_unix(path, connect_timeout_s=1.0)

            self.assertEqual(
                {"operation": "describe"},
                client.describe(),
            )
            self.assertEqual(
                {"operation": "open_world"},
                client.open_world("Rs_int", seed=3),
            )
            self.assertEqual(
                {"operation": "open_task"},
                client.open_task(
                    task_id="WATCH-28008",
                    seed=7,
                    observation_profile="debug",
                ),
            )
            self.assertEqual(
                {"operation": "reset"},
                client.reset(start_xy=(1.5, 2.5)),
            )
            self.assertEqual(
                {"operation": "step"},
                client.step("[Walk] <oven> (1)", trace_event={"turn": 2}),
            )
            self.assertEqual({"operation": "observe"}, client.observe())
            self.assertEqual({"operation": "render"}, client.render())
            self.assertEqual({"operation": "score"}, client.score())
            self.assertEqual(
                {"operation": "capture_score_baseline"},
                client.capture_score_baseline(),
            )
            self.assertEqual({"operation": "reward"}, client.reward())
            self.assertEqual(
                {"operation": "switch_task"},
                client.switch_task(
                    taskpack={"task_id": "CUSTOM-1"},
                    taskpack_digest="sha256:abc",
                    observation_profile="agent",
                ),
            )
            self.assertEqual(
                {"operation": "switch_world"},
                client.switch_world(
                    "Wainscott_0_int",
                    seed=9,
                    goals=[{"pred": "inside", "subject": "cup", "target": "bin"}],
                ),
            )
            self.assertEqual({"t_op_s": 0.25}, client.last_meta)

            client.close()
            self.assertFalse(client.is_connected)
            server.join()

        self.assertEqual(
            [
                "describe",
                "open_world",
                "open_task",
                "reset",
                "step",
                "observe",
                "render",
                "score",
                "capture_score_baseline",
                "reward",
                "switch_task",
                "switch_world",
            ],
            [request["op"] for request in requests],
        )
        self.assertEqual(list(range(1, 13)), [request["_rid"] for request in requests])
        self.assertEqual(
            {"op": "open_world", "_rid": 2, "scene": "Rs_int", "seed": 3},
            requests[1],
        )
        self.assertEqual(
            {
                "op": "open_task",
                "_rid": 3,
                "task_id": "WATCH-28008",
                "seed": 7,
                "observation_profile": "debug",
            },
            requests[2],
        )
        self.assertEqual(
            {"op": "reset", "_rid": 4, "start_xy": [1.5, 2.5]},
            requests[3],
        )
        self.assertEqual(
            {
                "op": "step",
                "_rid": 5,
                "cmd": "[Walk] <oven> (1)",
                "trace_event": {"turn": 2},
            },
            requests[4],
        )
        self.assertEqual(
            {
                "op": "switch_task",
                "_rid": 11,
                "taskpack": {"task_id": "CUSTOM-1"},
                "taskpack_digest": "sha256:abc",
                "observation_profile": "agent",
            },
            requests[10],
        )
        self.assertEqual(
            {
                "op": "switch_world",
                "_rid": 12,
                "scene": "Wainscott_0_int",
                "seed": 9,
                "goals": [{"pred": "inside", "subject": "cup", "target": "bin"}],
            },
            requests[11],
        )

    def test_call_discards_a_stale_reply_by_request_id(self) -> None:
        def handle(connection: socket.socket) -> None:
            request = _receive_request(connection)
            assert request is not None
            _send_reply(
                connection,
                {"_rid": 0, "ok": True, "result": {"stale": True}},
            )
            _send_reply(
                connection,
                {"_rid": request["_rid"], "ok": True, "result": {"fresh": True}},
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.sock"
            server = _StubServer.unix(path, handle)
            client = SimulatorClient.connect_unix(path, connect_timeout_s=1.0)

            self.assertEqual({"fresh": True}, client.call("describe"))

            client.close()
            server.join()

    def test_structured_remote_error_is_preserved(self) -> None:
        def handle(connection: socket.socket) -> None:
            request = _receive_request(connection)
            assert request is not None
            _send_reply(
                connection,
                {
                    "_rid": request["_rid"],
                    "ok": False,
                    "error_code": "TASK_NOT_FOUND",
                    "error": "unknown task MISSING-1",
                    "details": {"task_id": "MISSING-1"},
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.sock"
            server = _StubServer.unix(path, handle)
            client = SimulatorClient.connect_unix(path, connect_timeout_s=1.0)

            with self.assertRaises(SimulatorRemoteError) as raised:
                client.open_task(task_id="MISSING-1")

            self.assertEqual("TASK_NOT_FOUND", raised.exception.code)
            self.assertEqual("unknown task MISSING-1", raised.exception.message)
            self.assertEqual(
                {"task_id": "MISSING-1", "op": "open_task"},
                raised.exception.details,
            )
            client.close()
            server.join()

    def test_environment_failure_marks_connection_unusable(self) -> None:
        def handle(connection: socket.socket) -> None:
            request = _receive_request(connection)
            assert request is not None
            _send_reply(
                connection,
                {
                    "_rid": request["_rid"],
                    "ok": False,
                    "error_code": "ENVIRONMENT_FAILED",
                    "error": "world mutation failed",
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.sock"
            server = _StubServer.unix(path, handle)
            client = SimulatorClient.connect_unix(path, connect_timeout_s=1.0)

            with self.assertRaises(SimulatorRemoteError) as raised:
                client.open_world("Rs_int")
            self.assertEqual("ENVIRONMENT_FAILED", raised.exception.code)
            with self.assertRaises(SimulatorConnectionError):
                client.observe()

            client.close()
            server.join()

    def test_timeout_makes_the_connection_unusable(self) -> None:
        def handle(connection: socket.socket) -> None:
            self.assertIsNotNone(_receive_request(connection))
            time.sleep(0.1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.sock"
            server = _StubServer.unix(path, handle)
            client = SimulatorClient.connect_unix(path, connect_timeout_s=1.0)

            with self.assertRaises(SimulatorTimeoutError):
                client.call("observe", timeout_s=0.02)
            with self.assertRaises(SimulatorConnectionError):
                client.call("score")

            client.close()
            server.join()

    def test_reply_without_request_id_is_a_protocol_error(self) -> None:
        def handle(connection: socket.socket) -> None:
            self.assertIsNotNone(_receive_request(connection))
            _send_reply(connection, {"ok": True, "result": {}})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.sock"
            server = _StubServer.unix(path, handle)
            client = SimulatorClient.connect_unix(path, connect_timeout_s=1.0)

            with self.assertRaises(SimulatorProtocolError):
                client.describe()

            client.close()
            server.join()

    def test_connect_tcp_and_explicit_shutdown(self) -> None:
        requests: list[dict[str, Any]] = []

        def handle(connection: socket.socket) -> None:
            for _ in range(2):
                request = _receive_request(connection)
                assert request is not None
                requests.append(request)
                _send_reply(
                    connection,
                    {
                        "_rid": request["_rid"],
                        "ok": True,
                        "result": {"operation": request["op"]},
                    },
                )

        server = _StubServer.tcp(handle)
        client = SimulatorClient.connect_tcp(
            "127.0.0.1",
            server.port,
            connect_timeout_s=1.0,
        )

        self.assertEqual({"operation": "health"}, client.health())
        self.assertEqual({"operation": "shutdown"}, client.shutdown())
        self.assertFalse(client.is_connected)
        server.join()
        self.assertEqual(
            ["health", "shutdown"],
            [request["op"] for request in requests],
        )

    def test_spawned_process_is_reaped_by_close(self) -> None:
        server_code = """
import json
import os
import socket
import sys

path = sys.argv[1]
if os.path.exists(path):
    os.unlink(path)
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(path)
listener.listen(1)
connection, _ = listener.accept()
data = b""
while True:
    while b"\\n" not in data:
        chunk = connection.recv(65536)
        if not chunk:
            raise SystemExit(2)
        data += chunk
    line, _, data = data.partition(b"\\n")
    request = json.loads(line)
    response = {
        "_rid": request["_rid"],
        "ok": True,
        "result": {
            "operation": request["op"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    connection.sendall((json.dumps(response) + "\\n").encode("utf-8"))
    if request["op"] == "shutdown":
        break
connection.close()
listener.close()
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owned.sock"
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
                client = SimulatorClient.spawn(
                    [sys.executable, "-c", server_code, str(path)],
                    unix_socket=path,
                    connect_timeout_s=2.0,
                    shutdown_timeout_s=1.0,
                )

            self.assertTrue(client.owns_process)
            self.assertEqual(
                {"operation": "describe", "cuda_visible_devices": None},
                client.describe(),
            )

            client.close()
            self.assertFalse(client.is_connected)
            self.assertFalse(client.owns_process)
            self.assertFalse(path.exists())

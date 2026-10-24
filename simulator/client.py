# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standard-library client for an isolated InteractiveWorld simulator process.

Importing this module never imports Isaac Sim.  The simulator stays behind a
newline-delimited JSON socket so applications can use their own Python and ML
environment without sharing CUDA or native-library state with Isaac.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from .errors import (
    SimulatorConnectionError,
    SimulatorProtocolError,
    SimulatorRemoteError,
    SimulatorTimeoutError,
)
from .protocol import OBSERVATION_PROFILES


logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_CONNECT_TIMEOUT_S = 120.0
_DEFAULT_REQUEST_TIMEOUT_S = 600.0
_DEFAULT_OPEN_TIMEOUT_S = 2400.0
_DEFAULT_SHUTDOWN_TIMEOUT_S = 30.0
_RECV_BYTES = 65536


def _positive_timeout(name: str, value: float) -> float:
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be positive")
    return timeout


def _tune_tcp_socket(connection: socket.socket) -> None:
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (
        ("TCP_KEEPIDLE", 60),
        ("TCP_KEEPINTVL", 20),
        ("TCP_KEEPCNT", 5),
    ):
        if hasattr(socket, name):
            with contextlib.suppress(OSError):
                connection.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)


class SimulatorClient:
    """Blocking client for one persistent simulator server.

    Use :meth:`connect_unix` or :meth:`connect_tcp` to attach to a server owned
    elsewhere.  Closing an attached client only detaches it.  Use
    :meth:`shutdown` when stopping that server is intentional.

    :meth:`spawn` starts and owns a subprocess.  Closing an owned client asks
    the child to shut down and then reaps it; it never exits the caller.
    """

    def __init__(
        self,
        connection: socket.socket,
        *,
        endpoint: str,
        process: subprocess.Popen[bytes] | None = None,
        owned_socket_path: Path | None = None,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        open_timeout_s: float = _DEFAULT_OPEN_TIMEOUT_S,
        shutdown_timeout_s: float = _DEFAULT_SHUTDOWN_TIMEOUT_S,
    ) -> None:
        self._connection: socket.socket | None = connection
        self._endpoint = endpoint
        self._process = process
        self._owned_socket_path = owned_socket_path
        self._request_timeout_s = _positive_timeout(
            "request_timeout_s", request_timeout_s
        )
        self._open_timeout_s = _positive_timeout("open_timeout_s", open_timeout_s)
        self._shutdown_timeout_s = _positive_timeout(
            "shutdown_timeout_s", shutdown_timeout_s
        )
        self._buffer = b""
        self._request_id = 0
        self._dead = False
        self._lock = threading.Lock()
        self.last_meta: dict[str, Any] | None = None

    @classmethod
    def connect_unix(
        cls,
        path: str | os.PathLike[str],
        *,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        open_timeout_s: float = _DEFAULT_OPEN_TIMEOUT_S,
        shutdown_timeout_s: float = _DEFAULT_SHUTDOWN_TIMEOUT_S,
    ) -> SimulatorClient:
        """Attach to an existing Unix-domain simulator endpoint."""
        socket_path = Path(path).expanduser()
        timeout = _positive_timeout("connect_timeout_s", connect_timeout_s)
        connection = cls._connect_unix_socket(socket_path, timeout=timeout)
        return cls(
            connection,
            endpoint=str(socket_path),
            request_timeout_s=request_timeout_s,
            open_timeout_s=open_timeout_s,
            shutdown_timeout_s=shutdown_timeout_s,
        )

    @classmethod
    def connect_tcp(
        cls,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        open_timeout_s: float = _DEFAULT_OPEN_TIMEOUT_S,
        shutdown_timeout_s: float = _DEFAULT_SHUTDOWN_TIMEOUT_S,
    ) -> SimulatorClient:
        """Attach to an existing TCP simulator endpoint."""
        if not host:
            raise ValueError("host must not be empty")
        if isinstance(port, bool) or not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        timeout = _positive_timeout("connect_timeout_s", connect_timeout_s)
        connection = cls._connect_tcp_socket(host, int(port), timeout=timeout)
        return cls(
            connection,
            endpoint=f"[{host}]:{int(port)}",
            request_timeout_s=request_timeout_s,
            open_timeout_s=open_timeout_s,
            shutdown_timeout_s=shutdown_timeout_s,
        )

    @classmethod
    def spawn(
        cls,
        command: Sequence[str | os.PathLike[str]],
        *,
        unix_socket: str | os.PathLike[str] | None = None,
        host: str | None = None,
        port: int | None = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        open_timeout_s: float = _DEFAULT_OPEN_TIMEOUT_S,
        shutdown_timeout_s: float = _DEFAULT_SHUTDOWN_TIMEOUT_S,
    ) -> SimulatorClient:
        """Start an injected server command and connect to its declared endpoint.

        ``command`` must include the server's endpoint arguments.  Keeping the
        command injectable makes this client independent of the Isaac install
        layout and also lets applications use their own launch wrapper.
        ``env`` contains overrides merged onto the caller's environment.  The
        child never inherits ``CUDA_VISIBLE_DEVICES`` because its public GPU
        setting is a physical index used by both Isaac and the renderer.
        """
        argv = [str(part) for part in command]
        if not argv:
            raise ValueError("command must not be empty")
        use_unix = unix_socket is not None
        use_tcp = host is not None or port is not None
        if use_unix == use_tcp:
            raise ValueError(
                "provide exactly one endpoint: unix_socket or both host and port"
            )
        if use_tcp and (not host or port is None):
            raise ValueError("TCP spawning requires both host and port")

        owned_socket_path = (
            Path(unix_socket).expanduser() if unix_socket is not None else None
        )
        if owned_socket_path is not None and owned_socket_path.exists():
            raise SimulatorConnectionError(
                f"refusing to spawn over existing endpoint {owned_socket_path}"
            )
        timeout = _positive_timeout("connect_timeout_s", connect_timeout_s)
        process = cls._spawn_process(argv, cwd=cwd, env=env)
        try:
            connection, endpoint = cls._connect_spawned_endpoint(
                process,
                unix_socket=owned_socket_path,
                host=host,
                port=port,
                timeout=timeout,
            )
        except SimulatorConnectionError:
            cls._stop_process(process, graceful_timeout_s=0.0)
            raise

        return cls(
            connection,
            endpoint=endpoint,
            process=process,
            owned_socket_path=owned_socket_path,
            request_timeout_s=request_timeout_s,
            open_timeout_s=open_timeout_s,
            shutdown_timeout_s=shutdown_timeout_s,
        )

    @staticmethod
    def _spawn_process(
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None,
        env: Mapping[str, str] | None,
    ) -> subprocess.Popen[bytes]:
        child_environment = dict(os.environ)
        if env is not None:
            child_environment.update(
                {str(key): str(value) for key, value in env.items()}
            )
        child_environment.pop("CUDA_VISIBLE_DEVICES", None)
        # Isaac Kit logs to stdout; send them to stderr so the caller's stdout stays parseable.
        try:
            child_stdout: int | None = sys.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            child_stdout = None
        # An argv sequence with shell=False keeps every caller value a single argument.
        try:
            return subprocess.Popen(
                list(argv),
                cwd=os.fspath(cwd) if cwd is not None else None,
                env=child_environment,
                shell=False,
                stdout=child_stdout,
            )
        except OSError as error:
            raise SimulatorConnectionError(
                f"could not start simulator process {argv[0]!r}: {error}"
            ) from error

    @classmethod
    def _connect_spawned_endpoint(
        cls,
        process: subprocess.Popen[bytes],
        *,
        unix_socket: Path | None,
        host: str | None,
        port: int | None,
        timeout: float,
    ) -> tuple[socket.socket, str]:
        if unix_socket is not None:
            connection = cls._connect_unix_socket(
                unix_socket,
                timeout=timeout,
                process=process,
            )
            return connection, str(unix_socket)
        if host is None or port is None:
            raise SimulatorConnectionError("spawned TCP endpoint is incomplete")
        connection = cls._connect_tcp_socket(
            host,
            int(port),
            timeout=timeout,
            process=process,
        )
        return connection, f"[{host}]:{int(port)}"

    @staticmethod
    def _connect_unix_socket(
        path: Path,
        *,
        timeout: float,
        process: subprocess.Popen[bytes] | None = None,
    ) -> socket.socket:
        deadline = time.monotonic() + timeout
        last_error = "socket does not exist"
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise SimulatorConnectionError(
                    "simulator exited before accepting on "
                    f"{path} (return code {process.returncode})"
                )
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.settimeout(min(1.0, max(0.01, deadline - time.monotonic())))
                connection.connect(os.fspath(path))
                connection.settimeout(None)
                return connection
            except OSError as error:
                last_error = f"{type(error).__name__}: {error}"
                connection.close()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        raise SimulatorConnectionError(
            f"simulator endpoint {path} was not reachable in {timeout:.1f}s "
            f"(last error: {last_error})"
        )

    @staticmethod
    def _connect_tcp_socket(
        host: str,
        port: int,
        *,
        timeout: float,
        process: subprocess.Popen[bytes] | None = None,
    ) -> socket.socket:
        deadline = time.monotonic() + timeout
        last_error = "no connection attempted"
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise SimulatorConnectionError(
                    "simulator exited before accepting on "
                    f"[{host}]:{port} (return code {process.returncode})"
                )
            try:
                remaining = max(0.01, deadline - time.monotonic())
                connection = socket.create_connection(
                    (host, port), timeout=min(1.0, remaining)
                )
                _tune_tcp_socket(connection)
                connection.settimeout(None)
                return connection
            except OSError as error:
                last_error = f"{type(error).__name__}: {error}"
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        raise SimulatorConnectionError(
            f"simulator endpoint [{host}]:{port} was not reachable in {timeout:.1f}s "
            f"(last error: {last_error})"
        )

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def is_connected(self) -> bool:
        return self._connection is not None and not self._dead

    @property
    def owns_process(self) -> bool:
        return self._process is not None

    def call(
        self,
        op: str,
        *,
        timeout_s: float | None = None,
        **payload: Any,
    ) -> Any:
        """Send one request and return the correlated reply's raw ``result``."""
        if not op:
            raise ValueError("op must not be empty")
        if "op" in payload or "_rid" in payload:
            raise ValueError("payload must not override op or _rid")
        timeout = _positive_timeout(
            "timeout_s", self._request_timeout_s if timeout_s is None else timeout_s
        )
        with self._lock:
            return self._call_locked(op, timeout=timeout, payload=payload)

    def _call_locked(
        self, op: str, *, timeout: float, payload: Mapping[str, Any]
    ) -> Any:
        connection = self._checked_connection()
        self._request_id += 1
        request_id = self._request_id
        wire_request = self._encode_request(op, request_id, payload)
        try:
            reply = self._exchange(
                connection,
                op=op,
                request_id=request_id,
                wire_request=wire_request,
                timeout=timeout,
            )
        except socket.timeout as error:
            self._dead = True
            raise SimulatorTimeoutError(
                f"op={op!r} timed out after {timeout:.1f}s on {self._endpoint}"
            ) from error
        except OSError as error:
            self._dead = True
            raise SimulatorConnectionError(
                f"transport failed during op={op!r} on {self._endpoint}: {error}"
            ) from error
        finally:
            with contextlib.suppress(OSError):
                connection.settimeout(None)
        return self._unwrap_reply(op, reply)

    def _checked_connection(self) -> socket.socket:
        if self._connection is None:
            raise SimulatorConnectionError(
                f"client is detached from simulator endpoint {self._endpoint}"
            )
        if self._dead:
            raise SimulatorConnectionError(
                f"connection to {self._endpoint} is unusable after an earlier protocol or transport failure"
            )
        return self._connection

    @staticmethod
    def _encode_request(
        op: str,
        request_id: int,
        payload: Mapping[str, Any],
    ) -> bytes:
        request = {"op": op, "_rid": request_id, **payload}
        try:
            return (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        except (TypeError, ValueError) as error:
            raise SimulatorProtocolError(
                f"request for op={op!r} is not JSON serializable: {error}"
            ) from error

    def _exchange(
        self,
        connection: socket.socket,
        *,
        op: str,
        request_id: int,
        wire_request: bytes,
        timeout: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        connection.settimeout(timeout)
        connection.sendall(wire_request)
        while True:
            reply = self._receive_reply(connection, deadline=deadline, op=op)
            reply_id = reply.get("_rid")
            if reply_id is None:
                self._dead = True
                raise SimulatorProtocolError(
                    f"reply for op={op!r} from {self._endpoint} has no _rid"
                )
            if isinstance(reply_id, bool) or not isinstance(reply_id, int):
                self._dead = True
                raise SimulatorProtocolError(
                    f"reply for op={op!r} from {self._endpoint} has a non-integer _rid"
                )
            if reply_id == request_id:
                return reply
            logger.warning(
                "discarding stale simulator reply rid=%r while waiting for rid=%d from %s",
                reply_id,
                request_id,
                self._endpoint,
            )

    def _unwrap_reply(self, op: str, reply: Mapping[str, Any]) -> Any:
        ok = reply.get("ok")
        if not isinstance(ok, bool):
            self._dead = True
            raise SimulatorProtocolError(
                f"reply for op={op!r} from {self._endpoint} has no boolean ok field"
            )
        meta = reply.get("meta")
        self.last_meta = dict(meta) if isinstance(meta, Mapping) else None
        if not ok:
            error = self._remote_error(op, reply)
            if error.code == "ENVIRONMENT_FAILED":
                self._dead = True
            raise error
        return reply.get("result")

    def _receive_reply(
        self,
        connection: socket.socket,
        *,
        deadline: float,
        op: str,
    ) -> dict[str, Any]:
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._dead = True
                raise SimulatorTimeoutError(
                    f"op={op!r} timed out waiting for a reply from {self._endpoint}"
                )
            connection.settimeout(remaining)
            chunk = connection.recv(_RECV_BYTES)
            if not chunk:
                self._dead = True
                raise SimulatorConnectionError(
                    f"simulator closed {self._endpoint} during op={op!r}"
                )
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        try:
            reply = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            self._dead = True
            raise SimulatorProtocolError(
                f"simulator returned invalid JSON during op={op!r}: {error}"
            ) from error
        if not isinstance(reply, dict):
            self._dead = True
            raise SimulatorProtocolError(
                f"simulator returned a non-object reply during op={op!r}"
            )
        return reply

    @staticmethod
    def _remote_error(op: str, reply: Mapping[str, Any]) -> SimulatorRemoteError:
        error = reply.get("error")
        details: dict[str, Any] = {}
        if isinstance(error, Mapping):
            code = str(error.get("code") or "REMOTE_ERROR")
            message = str(error.get("message") or "simulator rejected the request")
            raw_details = error.get("details")
            if isinstance(raw_details, Mapping):
                details.update(raw_details)
        else:
            code = str(reply.get("error_code") or reply.get("code") or "REMOTE_ERROR")
            message = str(error or "simulator rejected the request")
            raw_details = reply.get("details")
            if isinstance(raw_details, Mapping):
                details.update(raw_details)
        details.setdefault("op", op)
        trace = reply.get("trace")
        if trace:
            details["trace"] = str(trace)
        return SimulatorRemoteError(code, message, details=details)

    @staticmethod
    def _require_mapping(op: str, result: Any) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise SimulatorProtocolError(
                f"op={op!r} returned {type(result).__name__}, expected an object"
            )
        return dict(result)

    def describe(self) -> dict[str, Any]:
        return self._require_mapping("describe", self.call("describe"))

    def health(self) -> dict[str, Any]:
        return self._require_mapping("health", self.call("health"))

    def open_task(
        self,
        *,
        task_id: str | None = None,
        taskpack: Mapping[str, Any] | None = None,
        taskpack_digest: str | None = None,
        seed: int | None = None,
        observation_profile: str | None = None,
    ) -> dict[str, Any]:
        payload = self._task_payload(
            task_id,
            taskpack,
            taskpack_digest,
            seed,
            observation_profile,
        )
        return self._require_mapping(
            "open_task",
            self.call("open_task", timeout_s=self._open_timeout_s, **payload),
        )

    @staticmethod
    def _task_payload(
        task_id: str | None,
        taskpack: Mapping[str, Any] | None,
        taskpack_digest: str | None,
        seed: int | None,
        observation_profile: str | None,
    ) -> dict[str, Any]:
        if (task_id is None) == (taskpack is None):
            raise ValueError("provide exactly one of task_id or taskpack")
        payload: dict[str, Any] = {}
        if task_id is not None:
            normalized_task_id = task_id.strip()
            if not normalized_task_id:
                raise ValueError("task_id must be a non-empty string")
            if taskpack_digest is not None:
                raise ValueError("taskpack_digest is only valid with taskpack")
            payload["task_id"] = normalized_task_id
        if taskpack is not None:
            if not taskpack_digest:
                raise ValueError("taskpack requires taskpack_digest")
            payload["taskpack"] = dict(taskpack)
            payload["taskpack_digest"] = taskpack_digest
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("seed must be an integer")
            payload["seed"] = seed
        if observation_profile is not None:
            if observation_profile not in OBSERVATION_PROFILES:
                raise ValueError("observation_profile must be 'agent' or 'debug'")
            payload["observation_profile"] = observation_profile
        return payload

    def open_world(
        self,
        scene: str,
        *,
        seed: int | None = None,
        goals: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Open a physical scene without installing a benchmark TaskPack."""
        payload = self._world_payload(scene, seed=seed, goals=goals)
        return self._require_mapping(
            "open_world",
            self.call("open_world", timeout_s=self._open_timeout_s, **payload),
        )

    @staticmethod
    def _world_payload(
        scene: str,
        *,
        seed: int | None,
        goals: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        if not isinstance(scene, str) or not scene.strip():
            raise ValueError("scene must be a non-empty string")
        payload: dict[str, Any] = {"scene": scene.strip()}
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("seed must be an integer")
            payload["seed"] = seed
        if goals is not None:
            if isinstance(goals, (str, bytes)) or not isinstance(goals, Sequence):
                raise ValueError("goals must be an array of inside-goal objects")
            normalized_goals = []
            for goal in goals:
                if not isinstance(goal, Mapping):
                    raise ValueError("goals must be an array of inside-goal objects")
                if set(goal) - {"pred", "subject", "target"}:
                    raise ValueError("inside-goal objects have unexpected fields")
                predicate = str(goal.get("pred") or "inside").casefold()
                subject = goal.get("subject")
                target = goal.get("target")
                if predicate != "inside":
                    raise ValueError("world goal pred must be 'inside'")
                if not isinstance(subject, str) or not subject.strip():
                    raise ValueError("world goal subject must be a non-empty string")
                if not isinstance(target, str) or not target.strip():
                    raise ValueError("world goal target must be a non-empty string")
                normalized_goals.append(
                    {
                        "pred": "inside",
                        "subject": subject.strip(),
                        "target": target.strip(),
                    }
                )
            payload["goals"] = normalized_goals
        return payload

    def reset(
        self,
        *,
        start_xy: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if start_xy is not None:
            if isinstance(start_xy, (str, bytes)) or len(start_xy) != 2:
                raise ValueError("start_xy must contain exactly two numbers")
            coordinates = tuple(start_xy)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in coordinates
            ):
                raise ValueError("start_xy must contain exactly two finite numbers")
            payload["start_xy"] = list(coordinates)
        return self._require_mapping("reset", self.call("reset", **payload))

    def step(
        self,
        cmd: str,
        *,
        trace_event: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("cmd must be a non-empty string")
        payload: dict[str, Any] = {"cmd": cmd.strip()}
        if trace_event is not None:
            payload["trace_event"] = dict(trace_event)
        return self._require_mapping("step", self.call("step", **payload))

    def observe(self) -> dict[str, Any]:
        return self._require_mapping("observe", self.call("observe"))

    def render(self) -> dict[str, Any]:
        return self._require_mapping("render", self.call("render"))

    def score(self) -> dict[str, Any]:
        return self._require_mapping("score", self.call("score"))

    def capture_score_baseline(self) -> dict[str, Any]:
        """Start task scoring from the current prepared world state."""
        return self._require_mapping(
            "capture_score_baseline",
            self.call("capture_score_baseline"),
        )

    def reward(self) -> dict[str, Any]:
        return self._require_mapping("reward", self.call("reward"))

    def switch_task(
        self,
        *,
        task_id: str | None = None,
        taskpack: Mapping[str, Any] | None = None,
        taskpack_digest: str | None = None,
        seed: int | None = None,
        observation_profile: str | None = None,
    ) -> dict[str, Any]:
        payload = self._task_payload(
            task_id,
            taskpack,
            taskpack_digest,
            seed,
            observation_profile,
        )
        return self._require_mapping(
            "switch_task",
            self.call("switch_task", timeout_s=self._open_timeout_s, **payload),
        )

    def switch_world(
        self,
        scene: str,
        *,
        seed: int | None = None,
        goals: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Switch a warm process to another physical scene."""
        payload = self._world_payload(scene, seed=seed, goals=goals)
        return self._require_mapping(
            "switch_world",
            self.call("switch_world", timeout_s=self._open_timeout_s, **payload),
        )

    def detach(self) -> None:
        """Close this connection without asking the simulator process to stop."""
        connection = self._connection
        self._connection = None
        self._buffer = b""
        if connection is None:
            return
        with contextlib.suppress(OSError):
            connection.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            connection.close()

    def shutdown(self) -> Any:
        """Explicitly stop the connected server, then close and reap local resources."""
        try:
            return self.call("shutdown", timeout_s=self._shutdown_timeout_s)
        finally:
            self.detach()
            self._cleanup_owned_process()

    def close(self) -> None:
        """Release the client; attached servers are left running and warm."""
        if self._process is None:
            self.detach()
            return
        try:
            self.shutdown()
        except (
            SimulatorConnectionError,
            SimulatorProtocolError,
            SimulatorRemoteError,
            SimulatorTimeoutError,
        ) as error:
            logger.debug("simulator shutdown reply was unavailable: %s", error)
            self.detach()
            self._cleanup_owned_process()

    def _cleanup_owned_process(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            self._stop_process(
                process,
                graceful_timeout_s=self._shutdown_timeout_s,
            )
        if self._owned_socket_path is not None:
            with contextlib.suppress(OSError):
                self._owned_socket_path.unlink()
            self._owned_socket_path = None

    @staticmethod
    def _stop_process(
        process: subprocess.Popen[bytes],
        *,
        graceful_timeout_s: float,
    ) -> None:
        if process.poll() is not None:
            return
        if graceful_timeout_s > 0:
            try:
                process.wait(timeout=graceful_timeout_s)
                return
            except subprocess.TimeoutExpired:
                pass
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)

    def __enter__(self) -> SimulatorClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["SimulatorClient"]

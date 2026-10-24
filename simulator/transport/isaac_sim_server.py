# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Newline-delimited JSON server for a persistent InteractiveWorld simulator."""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import logging
import math
import os
import re
import socket
import time
import traceback
from typing import Any

from simulator.errors import TaskCatalogError, TaskNotFoundError, TaskValidationError
from simulator.protocol import OBSERVATION_PROFILES, PROTOCOL_VERSION, PUBLIC_OPERATIONS


logger: logging.Logger = logging.getLogger(__name__)
_ENVIRONMENT_TYPE: type | None = None
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_SCENE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class RequestTooLargeError(ValueError):
    """One wire request exceeded the bounded public protocol size."""


class EnvironmentStateError(RuntimeError):
    """A mutating runtime call failed and the worker must not be reused."""


def _error(
    code: str, message: str, *, details: object | None = None
) -> dict[str, object]:
    """Backward-compatible error envelope with a stable machine-readable code."""
    result: dict[str, object] = {
        "ok": False,
        "error_code": code,
        "error": message,
    }
    if details is not None:
        result["details"] = details
    return result


def _environment_type() -> type:
    """Load Isaac-dependent modules only when a world is actually opened."""
    global _ENVIRONMENT_TYPE
    if _ENVIRONMENT_TYPE is None:
        from simulator.environment.isaac_env import IsaacEnv

        _ENVIRONMENT_TYPE = IsaacEnv
        try:
            from simulator.environment import settle_trace

            settle_trace.install()
        except Exception as error:  # noqa: BLE001 - diagnostics must not stop boot
            logger.warning("settle_trace unavailable: %s", error)
    return _ENVIRONMENT_TYPE


def serialize_observation(obs: dict, *, profile: str = "debug") -> dict:
    """Return the versioned JSON observation view.

    ``agent`` contains only observable/task-control fields. ``debug`` preserves
    the historical privileged relations, score, and goal checklist for evaluators.
    """
    if profile not in {"agent", "debug"}:
        raise ValueError("observation_profile must be 'agent' or 'debug'")
    score = obs.get("score") if isinstance(obs.get("score"), dict) else {}
    out = {
        "held": sorted(obs["held"]),
        "at": sorted(obs.get("at") or []),
        "trace_events": list(obs.get("trace_events") or []),
        "objects": obs.get("objects", []),  # valid command keys for the Human client
        # Preserve whether this is the whole registry or the visibility-filtered subset.
        "objects_are_visible_only": bool(obs.get("objects_are_visible_only")),
        "task_completed": bool(score.get("success")),
        "put_verb": obs.get(
            "put_verb", "PutInside"
        ),  # container's working place verb (Put=open bin / PutInside=interior)
        "task": obs.get("task", ""),
    }
    if profile == "debug":
        out.update(
            {
                "relations": [
                    [getattr(predicate, "name", str(predicate)), subject, target]
                    for predicate, subject, target in obs.get("relations", ())
                ],
                "score": score,
                "goal": obs.get("goal", []),
                "_visible_object_keys_full": obs.get("_visible_object_keys_full"),
            }
        )
    if obs.get("frames"):
        out["frames"] = obs[
            "frames"
        ]  # 1..n base64 PNG FPV frames across the last command (assistant obs)
    if "frame_segs" in obs:
        # Tags are emitted by IsaacEnv only when exactly parallel to ``frames``. Keep them on the same
        # whitelist boundary so downstream sparse sampling can retain semantically useful hold frames.
        out["frame_segs"] = list(obs.get("frame_segs") or [])
    if obs.get("frame"):
        out["frame"] = obs["frame"]  # final frame (human view + back-compat)
    if obs.get("vischeck") is not None:
        # OBSERVE-ONLY visibility measurement (IsaacEnv.observe documents the shape). Already JSON-safe.
        out["vischeck"] = obs["vischeck"]
    return out


def _attach_meta(reply: dict, op: object, t_op_s: float, env: Any | None) -> dict:
    """Stamp the reply ENVELOPE (never ``result``) with this op's wall time + phase breakdown. Draining the
    env's phases with ``clear=True`` makes each reply carry only the phases of the op that just ran, so a
    remote client can attribute cost per step without any cross-op bookkeeping."""
    meta: dict = {"t_op_s": float(t_op_s), "op": op}
    if env is not None:
        try:
            meta["phases"] = env.timings(clear=True)
        except Exception as exc:  # noqa: BLE001 - profiling must never break a reply
            meta["phases_error"] = f"{type(exc).__name__}: {exc}"
    reply["meta"] = meta
    return reply


def _recv_line(conn: socket.socket, buf: list[bytes]) -> bytes | None:
    """Read one newline-delimited message, buffering any extra bytes for the next call. None on EOF."""
    while b"\n" not in buf[0]:
        chunk = conn.recv(65536)
        if not chunk:
            return None
        buf[0] += chunk
        if len(buf[0]) > _MAX_REQUEST_BYTES:
            raise RequestTooLargeError(f"request exceeds {_MAX_REQUEST_BYTES} bytes")
    line, _, rest = buf[0].partition(b"\n")
    buf[0] = rest
    if len(line) > _MAX_REQUEST_BYTES:
        raise RequestTooLargeError(f"request exceeds {_MAX_REQUEST_BYTES} bytes")
    return line


def _send(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def _exception_reply(exc: Exception, *, debug_errors: bool) -> dict[str, object]:
    if isinstance(exc, TaskNotFoundError):
        code = "TASK_NOT_FOUND"
    elif isinstance(exc, TaskValidationError):
        code = "INVALID_TASKPACK"
    elif isinstance(exc, TaskCatalogError):
        code = "TASK_CATALOG_ERROR"
    elif isinstance(exc, KeyError):
        code = "INVALID_REQUEST"
    elif isinstance(exc, EnvironmentStateError):
        code = "ENVIRONMENT_FAILED"
    elif isinstance(exc, (TypeError, ValueError)):
        code = "INVALID_ARGUMENT"
    else:
        code = "INTERNAL_ERROR"
    result = _error(
        code,
        f"{type(exc).__name__}: {exc}",
        details={"exception_type": type(exc).__name__},
    )
    if debug_errors:
        result["trace"] = traceback.format_exc()[-4000:]
    return result


def dispatch_request(env: Any | None, req: dict, args) -> tuple[dict, Any | None, bool]:
    """Public, import-safe entry point used by the transport contract tests."""
    return _dispatch(env, req, args)


def _describe(env: Any | None, args: Any) -> dict[str, object]:
    from simulator.physical_world.runtime_capabilities import ACTIONS, IMPLEMENTED

    return {
        "protocol_version": PROTOCOL_VERSION,
        "simulator_version": "Isaac Sim 5.1",
        "taskpack_versions": ["data-generate-taskpack/1.0"],
        "runtime_contract_versions": ["compiled-runtime-contract/1.8"],
        "operations": list(PUBLIC_OPERATIONS),
        "action_primitives": [
            capability.primitive
            for capability in ACTIONS
            if capability.status == IMPLEMENTED
        ],
        "ready": env is not None,
        "render_observations": bool(getattr(args, "render_obs", True)),
        "gpu": int(getattr(args, "gpu", 0)),
        "observation_profile": str(getattr(args, "observation_profile", "agent")),
    }


def _observation_profile(env: Any | None, req: dict, args: Any) -> str:
    configured = str(getattr(args, "observation_profile", "agent"))
    if configured not in OBSERVATION_PROFILES:
        raise ValueError("server observation_profile must be 'agent' or 'debug'")
    requested = req.get("observation_profile")
    if requested is not None:
        profile = str(requested)
        if profile not in OBSERVATION_PROFILES:
            raise ValueError("observation_profile must be 'agent' or 'debug'")
        if profile != configured:
            raise ValueError(
                "observation_profile is fixed by the server; requested "
                f"{profile!r}, configured {configured!r}"
            )
    return configured


def _serialized_observation(env: Any, obs: dict, req: dict, args: Any) -> dict:
    result = serialize_observation(
        obs,
        profile=_observation_profile(env, req, args),
    )
    result["scene"] = str(
        getattr(env, "_public_scene", None) or getattr(env, "scene", "")
    )
    task_id = getattr(env, "_public_task_id", None)
    if task_id:
        result["task_id"] = task_id
        result["taskpack_sha256"] = getattr(env, "_public_taskpack_sha256", None)
        result["taskpack_digest"] = getattr(env, "_public_taskpack_digest", None)
    return result


def _render_result(observation: dict) -> dict[str, object]:
    return {
        key: observation[key]
        for key in ("frame", "frames", "frame_segs")
        if key in observation
    }


def _requested_task(req: dict) -> Any:
    """Resolve and verify either a bundled task ID or an inline TaskPack."""
    from simulator.catalog import TaskCatalog, TaskSpec

    task_id = req.get("task_id")
    taskpack = req.get("taskpack")
    if bool(task_id) == bool(taskpack):
        raise ValueError("provide exactly one of task_id or taskpack")
    if task_id:
        return TaskCatalog.bundled().get(str(task_id))
    if not isinstance(taskpack, dict):
        raise ValueError("taskpack must be a JSON object")
    expected_digest = req.get("taskpack_digest")
    if not isinstance(expected_digest, str) or not expected_digest:
        raise ValueError(
            "inline taskpack requires taskpack_digest (canonical JSON SHA-256)"
        )
    return TaskSpec.from_mapping(
        taskpack,
        expected_sha256=expected_digest,
    )


def _open_task(env: Any | None, req: dict, args: Any) -> tuple[Any, dict]:
    """Open a verified TaskPack cold, or switch/reset it in a warm app."""
    spec = _requested_task(req)
    runtime_scene = f"spec:{spec.task_id}"
    raw_seed = req.get("seed", getattr(args, "seed", 0))
    if type(raw_seed) is not int:
        raise ValueError("seed must be an integer")
    seed = raw_seed
    _observation_profile(env, req, args)
    if env is None:
        from simulator.environment.runtime_builder_spec import (
            register_runtime_builder_spec,
        )

        try:
            register_runtime_builder_spec(
                runtime_scene, spec.builder_spec, spec.builder_spec_digest
            )
            candidate = _environment_type()(
                runtime_scene,
                seed=seed,
                tag=getattr(args, "tag", None),
                active_gpu=int(getattr(args, "gpu", 0)),
                render_obs=bool(getattr(args, "render_obs", True)),
                strict_acceptance=bool(getattr(args, "strict_acceptance", False)),
            )
            candidate.builder_spec_digest = spec.builder_spec_digest
            observation = candidate.open()
        except Exception as exc:
            raise EnvironmentStateError(
                f"opening task {spec.task_id!r} left the worker unusable"
            ) from exc
        env = candidate
    elif (
        getattr(env, "_public_task_id", None) == spec.task_id
        and getattr(env, "_public_taskpack_digest", None) == spec.taskpack_digest
        and int(getattr(env, "seed", seed)) == seed
    ):
        try:
            observation = env.reset()
        except Exception as exc:
            raise EnvironmentStateError(
                f"resetting task {spec.task_id!r} left the worker unusable"
            ) from exc
    else:
        try:
            observation = env.switch_scene(
                runtime_scene,
                seed=seed,
                builder_spec=spec.builder_spec,
                builder_spec_digest=spec.builder_spec_digest,
            )
        except Exception as exc:
            raise EnvironmentStateError(
                f"switching to task {spec.task_id!r} left the worker unusable"
            ) from exc
    env._public_task_id = spec.task_id
    env._public_taskpack_sha256 = spec.source_sha256
    env._public_taskpack_digest = spec.taskpack_digest
    env._public_scene = spec.scene
    env._public_world_goals = None
    return env, observation


def _world_request(
    req: dict, args: Any
) -> tuple[str, int, list[tuple[str, str]] | None]:
    scene_value = req.get("scene")
    if not isinstance(scene_value, str) or not _SCENE_NAME.fullmatch(
        scene_value.strip()
    ):
        raise ValueError(
            "scene must contain 1-128 letters, digits, dots, underscores, or hyphens"
        )
    scene = scene_value.strip()
    if scene.startswith("spec:"):
        raise ValueError("spec: scenes are internal; use open_task")
    from simulator.world_catalog.scene_manifest import admitted_scenes

    if scene not in frozenset(admitted_scenes()):
        raise ValueError(f"scene {scene!r} is not an admitted InteractiveWorld world")
    raw_seed = req.get("seed", getattr(args, "seed", 0))
    if type(raw_seed) is not int:
        raise ValueError("seed must be an integer")
    goals_value = req.get("goals")
    if goals_value is None:
        goals = None
    elif not isinstance(goals_value, list):
        raise ValueError("goals must be an array of inside-goal objects")
    else:
        goals = []
        for goal in goals_value:
            if not isinstance(goal, dict):
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
            goals.append((subject.strip(), target.strip()))
    return scene, raw_seed, goals


def _open_world(env: Any | None, req: dict, args: Any) -> tuple[Any, dict]:
    """Open or switch to a physical scene without a TaskPack dependency."""
    scene, seed, goals = _world_request(req, args)
    _observation_profile(env, req, args)
    if env is None:
        try:
            candidate = _environment_type()(
                scene,
                seed=seed,
                tag=getattr(args, "tag", None),
                active_gpu=int(getattr(args, "gpu", 0)),
                render_obs=bool(getattr(args, "render_obs", True)),
                strict_acceptance=bool(getattr(args, "strict_acceptance", False)),
                goal_override=goals,
            )
            observation = candidate.open()
        except Exception as exc:
            raise EnvironmentStateError(
                f"opening world {scene!r} left the worker unusable"
            ) from exc
        env = candidate
    elif (
        getattr(env, "_public_scene", None) == scene
        and int(getattr(env, "seed", seed)) == seed
        and getattr(env, "_public_world_goals", None) == goals
    ):
        try:
            observation = env.reset()
        except Exception as exc:
            raise EnvironmentStateError(
                f"resetting world {scene!r} left the worker unusable"
            ) from exc
    else:
        try:
            observation = env.switch_scene(scene, seed=seed, goals=goals)
        except Exception as exc:
            raise EnvironmentStateError(
                f"switching to world {scene!r} left the worker unusable"
            ) from exc
    env._public_task_id = None
    env._public_taskpack_sha256 = None
    env._public_taskpack_digest = None
    env._public_scene = scene
    env._public_world_goals = goals
    return env, observation


def _observation_reply(
    env: Any, observation: dict, req: dict, args: Any
) -> tuple[dict, Any, bool]:
    return (
        {
            "ok": True,
            "result": _serialized_observation(env, observation, req, args),
        },
        env,
        False,
    )


def _dispatch_meta(
    op: str, env: Any | None, args: Any
) -> tuple[dict, Any | None, bool]:
    if op == "ping":
        return {"ok": True, "result": "pong"}, env, False
    if op == "health":
        return (
            {
                "ok": True,
                "result": {"status": "ready" if env is not None else "listening"},
            },
            env,
            False,
        )
    if op == "describe":
        return {"ok": True, "result": _describe(env, args)}, env, False
    raise AssertionError(f"unexpected metadata operation {op!r}")


def _dispatch_reset(env: Any, req: dict, args: Any) -> tuple[dict, Any, bool]:
    coordinates: tuple[float, float] | None = None
    start_xy = req.get("start_xy")
    if start_xy is not None:
        if (
            not isinstance(start_xy, list)
            or len(start_xy) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in start_xy
            )
        ):
            raise ValueError("start_xy must contain exactly two finite numbers")
        coordinates = (float(start_xy[0]), float(start_xy[1]))
    try:
        observation = env.reset(start_xy=coordinates)
    except Exception as exc:
        raise EnvironmentStateError(
            "reset failed inside the environment runtime"
        ) from exc
    return _observation_reply(env, observation, req, args)


def _dispatch_step(env: Any, req: dict, args: Any) -> tuple[dict, Any, bool]:
    command = req["cmd"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("cmd must be a non-empty string")
    command = command.strip()
    inner = getattr(env, "env", None)
    n_before = len(getattr(inner, "reports", []) or [])
    try:
        raw_observation = env.step(command, trace_event=req.get("trace_event"))
        result = _serialized_observation(env, raw_observation, req, args)
        reports = (getattr(inner, "reports", []) or [])[n_before:]
        result["reasons"] = [
            getattr(report[1], "name", str(report[1])) for report in reports
        ]
        outcomes = env.last_step_reports()
        result["outcomes"] = outcomes
        logger.info(
            "step cmd=%r -> outcomes=%s n_done=%s/%s held=%s",
            command,
            outcomes,
            (raw_observation.get("score") or {}).get("n_done"),
            (raw_observation.get("score") or {}).get("n_total"),
            result["held"],
        )
    except Exception as exc:
        raise EnvironmentStateError(
            "step failed after entering the environment runtime"
        ) from exc
    return {"ok": True, "result": result}, env, False


def _dispatch_query(op: str, env: Any, req: dict, args: Any) -> dict:
    if op == "observe":
        result = _serialized_observation(env, env.observe(), req, args)
        return {"ok": True, "result": result}
    if op == "render":
        observation = _serialized_observation(env, env.observe(), req, args)
        return {"ok": True, "result": _render_result(observation)}
    if op == "score":
        return {"ok": True, "result": env.score()}
    if op == "capture_score_baseline":
        try:
            result = env.capture_score_baseline()
            if not isinstance(result, dict):
                raise TypeError("capture_score_baseline must return an object")
        except Exception as exc:
            raise EnvironmentStateError(
                "score baseline capture failed inside the environment runtime"
            ) from exc
        return {"ok": True, "result": result}
    reward = env.reward()
    reward.pop("compliance", None)
    return {"ok": True, "result": reward}


def _dispatch(env: Any | None, req: dict, args) -> tuple[dict, Any | None, bool]:
    """Handle one request. Returns (reply, env, should_exit)."""
    op = req.get("op")
    if not isinstance(op, str) or not op:
        return _error("INVALID_REQUEST", "op must be a non-empty string"), env, False
    if op in {"ping", "health", "describe"}:
        return _dispatch_meta(op, env, args)
    if op == "close":
        return {"ok": True, "result": "detaching"}, env, False
    if op == "shutdown":
        return {"ok": True, "result": "shutting_down"}, env, True
    if op in {"open_task", "switch_task"}:
        env, observation = _open_task(env, req, args)
        return _observation_reply(env, observation, req, args)
    if op in {"open_world", "switch_world"}:
        env, observation = _open_world(env, req, args)
        return _observation_reply(env, observation, req, args)
    if env is None:
        return (
            _error(
                "ENV_NOT_OPEN",
                "environment is not open; send open_world or open_task first",
            ),
            env,
            False,
        )
    if op == "reset":
        return _dispatch_reset(env, req, args)
    if op == "step":
        return _dispatch_step(env, req, args)
    if op in {
        "observe",
        "render",
        "score",
        "capture_score_baseline",
        "reward",
    }:
        return _dispatch_query(op, env, req, args), env, False
    return _error("UNKNOWN_OPERATION", f"unknown op {op!r}"), env, False


def _tcp_listener(bind: str, port: int, portfile: str | None) -> socket.socket:
    """Bind + listen a TCP socket and (optionally) publish the CHOSEN port to ``portfile``.

    The bind is an explicit IPv4/IPv6 address (or ``localhost``), avoiding DNS
    dependencies and ambiguous multi-address selection.

    With ``--port 0`` and ``--portfile``, the kernel chooses the port and the server publishes it
    atomically, avoiding a probe-then-bind race between co-located workers.
    """
    bind_address = "::1" if bind.casefold() == "localhost" else bind
    try:
        address = ipaddress.ip_address(bind_address)
    except ValueError as exc:
        raise ValueError("--bind must be an IP address or localhost") from exc
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    sockaddr = (str(address), port)
    srv = socket.socket(family, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        try:
            srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError as exc:  # some kernels forbid flipping it; an IPv6-only listener remains valid
            logger.warning("IPV6_V6ONLY not clearable: %s", exc)
    # A fixed port can transiently collide with another local socket. Retry for a bounded
    # interval while still surfacing permanent conflicts.
    deadline = time.monotonic() + float(os.getenv("ISAAC_BIND_RETRY_S", "90"))
    while True:
        try:
            srv.bind(sockaddr)
            break
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise OSError(
                    f"cannot bind {bind}:{port} after retrying ({exc}). If the port is inside "
                    f"/proc/sys/net/ipv4/ip_local_port_range it can be stolen by an outbound socket -- "
                    "move ISAAC_SCENE_PORT_BASE below that range."
                ) from exc
            logger.warning("bind %s:%d busy (%s); retrying", bind, port, exc)
            time.sleep(2.0)
    # Queue reconnects that arrive while a slow simulator operation is still finishing.
    srv.listen(8)
    real_port = srv.getsockname()[1]
    if portfile:
        tmp = f"{portfile}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(str(real_port))
        os.replace(tmp, portfile)  # atomic: a reader never sees a half-written port
    return srv


def _is_loopback_bind(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _tune_conn(conn: socket.socket) -> None:
    """TCP_NODELAY + keepalive on an accepted TCP connection (no-ops / skipped for AF_UNIX).

    NODELAY: the protocol is a SMALL request (tens of bytes) answered by a LARGE reply (a base64 PNG obs is
    100s of KB). Nagle would hold the tiny request back waiting for more data to coalesce, adding a delay
    to every single op for no benefit -- the classic small-write/large-read anti-pattern.
    KEEPALIVE: a cross-machine peer can vanish (host failure, preemption) with no FIN. Without keepalive the
    server would sit in recv() forever holding a warm sim hostage; with it the socket errors out and the
    accept loop frees the sim for the next client.
    """
    if conn.family == socket.AF_UNIX:
        return
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, val in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 20), ("TCP_KEEPCNT", 5)):
        if hasattr(socket, opt):
            with contextlib.suppress(OSError):
                conn.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)


def _decode_request(line: bytes) -> tuple[dict | None, dict[str, object] | None]:
    try:
        request = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, _error("BAD_JSON", "request is not valid JSON")
    if not isinstance(request, dict):
        return None, _error("INVALID_REQUEST", "request must be a JSON object")
    return request, None


def _execute_request(
    environment: Any | None, request: dict, args: Any
) -> tuple[dict, Any | None, bool]:
    started = time.perf_counter()
    request_id = request.get("_rid")
    try:
        reply, environment, should_exit = dispatch_request(environment, request, args)
    except Exception as exc:  # noqa: BLE001 - protocol boundary normalizes failures
        reply = _exception_reply(
            exc,
            debug_errors=bool(getattr(args, "debug_errors", False)),
        )
        should_exit = isinstance(exc, EnvironmentStateError)
    reply["_rid"] = request_id
    payload = _attach_meta(
        reply,
        request.get("op"),
        time.perf_counter() - started,
        environment,
    )
    return payload, environment, should_exit


def _try_send(conn: socket.socket, payload: dict, *, op: object = None) -> bool:
    try:
        _send(conn, payload)
        return True
    except OSError as exc:
        logger.warning("send failed for op=%r (%s); ending session", op, exc)
        return False


def _serve_session(
    conn: socket.socket, env: Any | None, args
) -> tuple[Any | None, bool]:
    """Serve ONE connected client until it hangs up. Returns (env, exit_requested).

    A failed send ends only this client session so a persistent TCP server can accept
    a reconnect without losing the warm simulator.
    """
    buf = [b""]
    while True:
        try:
            line = _recv_line(conn, buf)
        except RequestTooLargeError as exc:
            with contextlib.suppress(OSError):
                _send(conn, _error("REQUEST_TOO_LARGE", str(exc)))
            return env, False
        except OSError as exc:  # peer reset / keepalive death -- same meaning as EOF
            logger.warning("recv failed (%s); ending session", exc)
            return env, False
        if line is None:
            return env, False  # client hung up
        request, parse_error = _decode_request(line)
        if parse_error is not None:
            if not _try_send(conn, parse_error):
                return env, False
            continue
        if request is None:
            raise AssertionError("valid request decoder returned no request")
        payload, env, should_exit = _execute_request(env, request, args)
        if not _try_send(conn, payload, op=request.get("op")):
            return env, should_exit
        if should_exit:
            return env, True
        if request.get("op") == "close":
            return env, False


def _argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene",
        default="",
        help="physical scene to open at boot when --preopen is set",
    )
    ap.add_argument(
        "--task-id",
        default="",
        help="bundled TaskPack ID to open at boot when --preopen is set",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sock", default=None, help="AF_UNIX socket path (same-machine transport)"
    )
    ap.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port to listen on instead of --sock (cross-machine transport). 0 = let the kernel "
        "pick and publish it to --portfile.",
    )
    ap.add_argument(
        "--bind",
        default="::1",
        help="TCP bind address (with --port). Default ::1 is loopback-only; pass an explicit "
        "routable address to accept remote clients.",
    )
    ap.add_argument(
        "--allow-remote",
        action="store_true",
        help="acknowledge that TCP has no authentication and allow a non-loopback bind",
    )
    ap.add_argument(
        "--portfile",
        default=None,
        help="write the bound TCP port here (atomically) so a client that passed --port 0 can find it",
    )
    ap.add_argument(
        "--preopen",
        action="store_true",
        help="build --task-id or --scene at boot so the first client attaches warm",
    )
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="active_gpu/physics_gpu for this simulator worker",
    )
    ap.add_argument(
        "--render-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="attach a per-command egocentric FPV frame to each obs (pixels). --no-render-obs for a "
        "symbolic-only smoke (skips the RTX warmup).",
    )
    ap.add_argument(
        "--observation-profile",
        choices=("agent", "debug"),
        default="agent",
        help="agent hides privileged goal/score state; debug exposes evaluator state",
    )
    ap.add_argument(
        "--strict-acceptance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "stricter authoring checks while opening and acting (finer stance search, "
            "no navigation fallbacks); the benchmark runs without them"
        ),
    )
    ap.add_argument(
        "--debug-errors",
        action="store_true",
        help="include server tracebacks in protocol errors",
    )
    return ap


def _validated_args(ap: argparse.ArgumentParser) -> argparse.Namespace:
    args = ap.parse_args()
    tcp = args.port is not None
    if tcp == bool(args.sock):
        ap.error("pass exactly one of --sock (Unix) or --port (TCP)")
    if args.scene and args.task_id:
        ap.error("--scene and --task-id are mutually exclusive")
    if args.preopen and not (args.scene or args.task_id):
        ap.error("--preopen requires --task-id or legacy --scene")
    if tcp and not args.allow_remote and not _is_loopback_bind(args.bind):
        ap.error(
            "non-loopback --bind exposes an unauthenticated control port; "
            "pass --allow-remote to acknowledge this"
        )
    return args


def _listener(args: argparse.Namespace) -> tuple[socket.socket, bool, str]:
    tcp = args.port is not None
    if tcp:
        server = _tcp_listener(args.bind, args.port, args.portfile)
        where = f"[{args.bind}]:{server.getsockname()[1]}"
        return server, tcp, where

    if os.path.lexists(args.sock):
        raise ValueError(f"refusing to replace existing socket path: {args.sock}")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.sock)
    server.listen(1)
    return server, tcp, args.sock


def _preopen(args: argparse.Namespace) -> Any | None:
    if not args.preopen:
        return None
    t0 = time.perf_counter()
    try:
        if args.task_id:
            environment, _observation = _open_task(
                None,
                {
                    "op": "open_task",
                    "task_id": args.task_id,
                    "seed": args.seed,
                    "observation_profile": args.observation_profile,
                },
                args,
            )
        else:
            environment = _environment_type()(
                args.scene,
                seed=args.seed,
                tag=args.tag,
                active_gpu=args.gpu,
                render_obs=args.render_obs,
                strict_acceptance=args.strict_acceptance,
            )
            environment.open()
    except Exception:  # noqa: BLE001 - report the reason, then die visibly
        logger.exception(
            "PREOPEN FAILED for task=%s scene=%s",
            args.task_id or "-",
            args.scene or "-",
        )
        raise SystemExit(1)
    logger.info(
        "PREOPEN ok task=%s scene=%s in %.1fs",
        args.task_id or "-",
        args.scene or "-",
        time.perf_counter() - t0,
    )
    return environment


def _serve_forever(
    server: socket.socket,
    *,
    tcp: bool,
    environment: Any | None,
    args: argparse.Namespace,
) -> None:
    while True:
        connection, peer = server.accept()
        _tune_conn(connection)
        logger.info("client connected %s", peer if tcp else "")
        environment, should_exit = _serve_session(connection, environment, args)
        if should_exit:
            connection.close()
            server.close()
            if not tcp and args.sock:
                with contextlib.suppress(OSError):
                    os.unlink(args.sock)
            if tcp and args.portfile:
                with contextlib.suppress(OSError):
                    os.unlink(args.portfile)
            if environment is not None:
                environment.close(hard=True)
            os._exit(0)
        with contextlib.suppress(OSError):
            connection.close()
        logger.info("client gone; scene stays warm, awaiting next client")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[sim_server] %(message)s")
    args = _validated_args(_argument_parser())
    server, tcp, where = _listener(args)

    logger.info(
        "listening on %s (task=%s scene=%s seed=%d)",
        where,
        args.task_id or "-",
        args.scene or "-",
        args.seed,
    )
    _serve_forever(server, tcp=tcp, environment=_preopen(args), args=args)


if __name__ == "__main__":
    main()

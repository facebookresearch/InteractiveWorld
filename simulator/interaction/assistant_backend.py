"""Portable model transports for InteractiveWorld Assistant and Human agents.

The public model endpoint uses one provider-neutral HTTP JSON contract.

Request::

    {
      "system": "...",
      "messages": [
        {"role": "user", "text": "...", "images_b64": ["<PNG base64>"]},
        {"role": "assistant", "text": "..."}
      ],
      "max_new_tokens": 96,
      "temperature": 0.0,
      "do_sample": false
    }

Response::

    {"ok": true, "text": "..."}

An unsuccessful service response is ``{"ok": false, "error": "..."}``.
The same internal request is translated to generic HTTP JSON, OpenAI-compatible
Chat Completions, Anthropic Messages, or Google Gemini REST.  Only the Python
standard library is used.  HTTPS is accepted for any host; cleartext HTTP is
restricted to loopback.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, NoReturn, TypeVar

_RECENT_IMAGE_BATCHES = 2
_ANTHROPIC_DEFAULT_MAX_TOKENS = 1024
_MAX_GATEWAY_BODY_BYTES = 128 * 1024 * 1024
_MAX_PROVIDER_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_API_KEY_BYTES = 8 * 1024
_MAX_URL_CHARS = 8 * 1024
_PROVIDERS = ("http-json", "openai-compatible", "anthropic", "gemini")
_T = TypeVar("_T")
_LOG: logging.Logger = logging.getLogger(__name__)


class AssistantBackendError(RuntimeError):
    """The configured Assistant transport is invalid or unavailable."""


def _uniform_indices(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return [((index + 1) * length + count - 1) // count - 1 for index in range(count)]


def select_temporal_image_batches(
    batches: list[list[_T]],
    frame_cap: int,
    recent_batch_count: int = _RECENT_IMAGE_BATCHES,
) -> list[list[_T]]:
    """Keep recent visual turns intact and uniformly sample older frames."""
    selected: list[list[_T]] = [[] for _ in batches]
    refs = [
        (batch_index, item_index, item)
        for batch_index, batch in enumerate(batches)
        for item_index, item in enumerate(batch)
    ]
    if frame_cap <= 0 or len(refs) <= frame_cap:
        return [list(batch) for batch in batches]

    recent_indices = set(
        [
            batch_index
            for batch_index in range(len(batches) - 1, -1, -1)
            if batches[batch_index]
        ][: max(0, recent_batch_count)]
    )
    older_refs = [ref for ref in refs if ref[0] not in recent_indices]
    recent_refs = [ref for ref in refs if ref[0] in recent_indices]
    if len(recent_refs) >= frame_cap:
        if any(len(batch) > frame_cap for batch in batches):
            kept_refs = [
                refs[index] for index in _uniform_indices(len(refs), frame_cap)
            ]
        else:
            kept_refs = recent_refs[-frame_cap:]
    else:
        older_count = min(frame_cap - len(recent_refs), len(older_refs))
        kept_refs = [
            older_refs[index]
            for index in _uniform_indices(len(older_refs), older_count)
        ]
        kept_refs.extend(recent_refs)

    kept_positions = {
        (batch_index, item_index) for batch_index, item_index, _item in kept_refs
    }
    for batch_index, item_index, item in refs:
        if (batch_index, item_index) in kept_positions:
            selected[batch_index].append(item)
    return selected


def _parse_http_url(url: str, label: str) -> tuple[urllib.parse.SplitResult, str]:
    if len(url) > _MAX_URL_CHARS or any(
        ord(character) < 32 or ord(character) == 127 for character in url
    ):
        raise AssistantBackendError(
            f"{label} is too long or contains unsupported control characters"
        )
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise AssistantBackendError(f"{label} is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise AssistantBackendError(f"{label} must be an http(s) URL")
    if port is not None and port == 0:
        raise AssistantBackendError(f"{label} must use a valid network port")
    if parsed.username is not None or parsed.password is not None:
        raise AssistantBackendError(f"{label} must not contain credentials")
    if parsed.fragment:
        raise AssistantBackendError(f"{label} must not contain a URL fragment")
    if parsed.query:
        raise AssistantBackendError(
            f"{label} must not contain a query string; use headers for credentials"
        )
    return parsed, hostname


def _validate_http_transport(url: str, label: str) -> None:
    parsed, hostname = _parse_http_url(url, label)
    if parsed.scheme == "https" or hostname == "localhost":
        return
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise AssistantBackendError(
            f"{label} uses cleartext HTTP to a non-loopback host; "
            "use HTTPS or a loopback endpoint"
        )


def _url_origin(url: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AssistantBackendError("HTTP URL contains an invalid port") from exc
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme.lower(), str(parsed.hostname or "").lower(), port


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only when the transport security boundary is unchanged."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        target = urllib.parse.urljoin(req.full_url, newurl)
        _validate_http_transport(target, "provider redirect URL")
        if _url_origin(req.full_url) != _url_origin(target):
            raise AssistantBackendError(
                "provider redirect changed origin or transport security"
            )
        return super().redirect_request(req, fp, code, msg, headers, target)


def _open_http_request(request: urllib.request.Request, timeout_s: float) -> Any:
    opener = urllib.request.build_opener(_SameOriginRedirectHandler())
    return opener.open(request, timeout=timeout_s)


def _read_http_response(response: Any, label: str) -> bytes:
    body = response.read(_MAX_PROVIDER_RESPONSE_BYTES + 1)
    if len(body) > _MAX_PROVIDER_RESPONSE_BYTES:
        raise AssistantBackendError(f"{label} exceeds the response size limit")
    return body


def _request_timeout(requested: float | None, default: float, label: str) -> float:
    raw = default if requested is None else requested
    if type(raw) not in (int, float):
        raise AssistantBackendError(f"{label} timeout must be positive and finite")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise AssistantBackendError(f"{label} timeout must be positive and finite")
    return value


def _pil_to_png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _split_messages(messages: list[dict]) -> tuple[str, list[Any], str]:
    system = ""
    images: list[Any] = []
    texts: list[str] = []
    for message in messages:
        if message.get("role") == "system":
            system = str(message.get("content") or "")
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for block in content or []:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "image":
                images.append(block["image"])
            elif block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
    return system, images, "\n".join(text for text in texts if text)


def _history_with_temporal_images(
    history: list[dict[str, Any]], frame_cap: int
) -> list[dict[str, Any]]:
    wire_history = [dict(row) for row in history]
    user_positions: list[int] = []
    batches: list[list[str]] = []
    for position, row in enumerate(history):
        if row.get("role") != "user":
            continue
        user_positions.append(position)
        batches.append(list(row.get("images_b64") or []))
    selected = select_temporal_image_batches(batches, frame_cap)
    for position, images in zip(user_positions, selected):
        wire_history[position].pop("images_b64", None)
        if images:
            wire_history[position]["images_b64"] = images
    return wire_history


def _count_history_images(history: list[dict[str, Any]]) -> int:
    return sum(len(row.get("images_b64") or []) for row in history)


def _current_capture_provenance(capture: Any, frame_count: int) -> list[dict]:
    if capture is None or not getattr(capture, "active", False):
        return []
    return capture.current_frame_provenance(frame_count)


def _selected_capture_provenance(
    batches: list[list[dict]], frame_cap: int
) -> list[dict]:
    return [
        provenance
        for batch in select_temporal_image_batches(batches, frame_cap)
        for provenance in batch
    ]


class AssistantBackend:
    """Conversation-scoped interface shared by HTTP and scripted assistants."""

    name = "base"
    _OBSERVATIONAL_LIST_FIELDS = ("_history", "_frame_provenance_batches")
    _OBSERVATIONAL_SCALAR_FIELDS = ("system",)

    def __init__(self) -> None:
        self._n: dict[str, int] = {}
        self._latency_s: list[float] = []
        self.request_visual_policy = ""
        self.capture: Any = None

    def reset(self) -> None:
        raise NotImplementedError

    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def probe(self) -> str:
        raise NotImplementedError

    def decision_text(self, raw: str) -> str:
        return raw

    def turn_metadata(self) -> dict[str, object]:
        return {}

    def audit_messages(self, original: list[dict]) -> list[dict]:
        return original

    def close(self) -> None:
        return None

    def snapshot_observational_state(self) -> dict[str, object]:
        state: dict[str, object] = {}
        for name in self._OBSERVATIONAL_LIST_FIELDS:
            value = getattr(self, name, None)
            if isinstance(value, list):
                state[name] = list(value)
        for name in self._OBSERVATIONAL_SCALAR_FIELDS:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def restore_observational_state(self, snapshot: Mapping[str, object]) -> None:
        for name in self._OBSERVATIONAL_LIST_FIELDS:
            value = snapshot.get(name)
            if isinstance(value, list):
                setattr(self, name, list(value))
        for name in self._OBSERVATIONAL_SCALAR_FIELDS:
            if name in snapshot:
                setattr(self, name, snapshot[name])

    def stats(self) -> dict[str, object]:
        latency = sorted(self._latency_s)
        return {
            "backend": self.name,
            "calls": self._n.get("calls", 0),
            "http_fail": self._n.get("http_fail", 0),
            "empty": self._n.get("empty", 0),
            "latency_p50_s": (
                round(latency[len(latency) // 2], 2) if latency else None
            ),
            "latency_total_s": round(sum(latency), 1),
        }

    def _bump(self, key: str, count: int = 1) -> None:
        self._n[key] = self._n.get(key, 0) + count

    def _cap_rule(self) -> str:
        return self.request_visual_policy or (
            "keep the latest two visual turns and uniformly sample older frames"
        )

    def _capture_response(
        self,
        raw: str,
        latency_s: float | None = None,
        error: str = "",
    ) -> None:
        if self.capture is not None:
            self.capture.record_response(raw, latency_s, error)


class HttpJsonAssistant(AssistantBackend):
    """Provider-neutral HTTP JSON assistant with multi-turn visual history."""

    name = "http-json"

    def __init__(
        self,
        url: str,
        max_new_tokens: int | None = 96,
        temperature: float = 0.0,
        timeout_s: float = 180.0,
        frame_cap: int = 32,
    ) -> None:
        super().__init__()
        _validate_http_transport(url, "assistant URL")
        _request_timeout(None, timeout_s, "assistant")
        if frame_cap < 0:
            raise AssistantBackendError("assistant frame cap must be non-negative")
        self.url = url
        self.max_new_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None and int(max_new_tokens) > 0
            else None
        )
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.frame_cap = int(frame_cap)
        self.system = ""
        self._history: list[dict[str, Any]] = []
        self._frame_provenance_batches: list[list[dict]] = []

    def reset(self) -> None:
        self.system = ""
        self._history = []
        self._frame_provenance_batches = []

    def probe(self) -> str:
        request = {
            "system": "Reply with the single word OK.",
            "messages": [{"role": "user", "text": "Say OK."}],
            "max_new_tokens": 8,
            "temperature": 0.0,
            "do_sample": False,
        }
        response = self._post(request, timeout_s=min(self.timeout_s, 120.0))
        if response.get("ok") is not True:
            raise AssistantBackendError(
                f"assistant endpoint returned not-ok: {response.get('error')}"
            )
        text = response.get("text")
        if not isinstance(text, str) or not text.strip():
            raise AssistantBackendError("assistant endpoint returned an empty probe")
        return text

    def _post(
        self, request: Mapping[str, object], timeout_s: float | None = None
    ) -> dict[str, Any]:
        encoded = json.dumps(request).encode("utf-8")
        http_request = urllib.request.Request(
            self.url,
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _open_http_request(
                http_request,
                _request_timeout(timeout_s, self.timeout_s, "assistant"),
            ) as response:
                payload = json.loads(
                    _read_http_response(response, "assistant endpoint response").decode(
                        "utf-8"
                    )
                )
        except urllib.error.HTTPError as exc:
            raise AssistantBackendError(
                f"assistant endpoint returned HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AssistantBackendError("assistant endpoint request failed") from exc
        except TimeoutError as exc:
            raise AssistantBackendError("assistant endpoint request timed out") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssistantBackendError(
                "assistant endpoint response is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise AssistantBackendError("assistant endpoint response must be an object")
        return payload

    def _capture_request(
        self, request: Mapping[str, Any], provenance: list[dict]
    ) -> None:
        capture = self.capture
        if capture is None or not getattr(capture, "active", False):
            return
        messages = request["messages"]
        encoded_images = [
            image for message in messages for image in (message.get("images_b64") or [])
        ]
        saved = capture.save_images(
            [("b64", image) for image in encoded_images],
            provenance=provenance,
        )
        position = 0
        redacted_messages = []
        for message in messages:
            row = {"role": message.get("role"), "text": message.get("text")}
            image_count = len(message.get("images_b64") or [])
            if image_count:
                row["images_b64_written_to"] = [
                    image.get("file")
                    for image in saved[position : position + image_count]
                ]
                position += image_count
            redacted_messages.append(row)
        capture.record_request(
            self.name,
            {
                "url": self.url,
                "system": request["system"],
                "messages": redacted_messages,
                "max_new_tokens": request.get("max_new_tokens"),
                "temperature": request["temperature"],
                "do_sample": request["do_sample"],
            },
            saved,
            {
                "transport": "provider-neutral HTTP JSON",
                "flattened_history": False,
                "images_on_the_wire": "base64 PNG in each user turn",
                "n_history_turns": len(messages),
                "frame_cap": self.frame_cap or "uncapped",
                "n_cumulative_frames_before_cap": _count_history_images(self._history),
                "cap_rule": self._cap_rule(),
            },
        )

    def generate(self, messages: list[dict]) -> str:
        system, images, text = _split_messages(messages)
        if system:
            self.system = system
        encoded_images = [
            base64.b64encode(_pil_to_png_bytes(image)).decode("ascii")
            for image in images
        ]
        turn: dict[str, Any] = {"role": "user", "text": text}
        if encoded_images:
            turn["images_b64"] = encoded_images
        self._history.append(turn)
        self._frame_provenance_batches.append(
            _current_capture_provenance(self.capture, len(encoded_images))
        )
        wire_history = _history_with_temporal_images(self._history, self.frame_cap)
        selected_provenance = _selected_capture_provenance(
            self._frame_provenance_batches, self.frame_cap
        )
        request: dict[str, Any] = {
            "system": self.system,
            "messages": wire_history,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0.0,
        }
        if self.max_new_tokens is not None:
            request["max_new_tokens"] = self.max_new_tokens
        self._capture_request(request, selected_provenance)

        started = time.perf_counter()
        self._bump("calls")
        try:
            response = self._post(request)
        except Exception as exc:  # noqa: BLE001 - transport failure voids the run
            self._bump("http_fail")
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            self._capture_response("", time.perf_counter() - started, error)
            _LOG.warning("assistant HTTP request failed: %s", error)
            self._history.pop()
            self._frame_provenance_batches.pop()
            return ""

        latency = time.perf_counter() - started
        self._latency_s.append(latency)
        if response.get("ok") is not True:
            self._bump("http_fail")
            error = f"not-ok: {str(response.get('error'))[:200]}"
            self._capture_response("", latency, error)
            _LOG.warning("assistant endpoint returned %s", error)
            self._history.pop()
            self._frame_provenance_batches.pop()
            return ""
        reply = response.get("text")
        if not isinstance(reply, str):
            self._bump("http_fail")
            self._capture_response("", latency, "response text must be a string")
            self._history.pop()
            self._frame_provenance_batches.pop()
            return ""
        self._capture_response(reply, latency)
        if not reply.strip():
            self._bump("empty")
        self._history.append({"role": "assistant", "text": reply})
        return reply


def normalize_provider(provider: str) -> str:
    """Return the canonical public provider name."""
    normalized = provider.strip().lower().replace("_", "-")
    aliases = {
        "google": "gemini",
        "openai": "openai-compatible",
        "openai-chat": "openai-compatible",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _PROVIDERS:
        supported = ", ".join(_PROVIDERS)
        raise AssistantBackendError(
            f"unknown model provider {provider!r}; supported backends/providers: "
            f"{supported}"
        )
    return normalized


def default_provider_base_url(provider: str) -> str:
    """Return the hosted default; local OpenAI-compatible servers override it."""
    normalized = normalize_provider(provider)
    defaults = {
        "http-json": "",
        "openai-compatible": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
    }
    return defaults[normalized]


def resolve_api_key_env(
    provider: str,
    role: str,
    requested_name: str = "",
    environ: Mapping[str, str] | None = None,
    base_url: str = "",
) -> str:
    """Resolve an environment-variable *name* without retaining its secret value."""
    normalized = normalize_provider(provider)
    environment = os.environ if environ is None else environ
    if requested_name:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", requested_name):
            raise AssistantBackendError(
                f"{role} API-key environment variable name is invalid"
            )
        if not environment.get(requested_name):
            raise AssistantBackendError(
                f"{role} API-key environment variable {requested_name!r} is unset"
            )
        return requested_name

    candidates = [f"INTERACTIVEWORLD_{role.upper()}_API_KEY"]
    credential_url = base_url or default_provider_base_url(normalized)
    if credential_url and _official_provider_host(normalized, credential_url):
        candidates.extend(
            {
                "http-json": (),
                "openai-compatible": ("OPENAI_API_KEY",),
                "anthropic": ("ANTHROPIC_API_KEY",),
                "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            }[normalized]
        )
    return next((name for name in candidates if environment.get(name)), "")


def _append_endpoint(base_url: str, path_suffix: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    path = f"{parsed.path.rstrip('/')}/{path_suffix.lstrip('/')}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, "")
    )


def _provider_endpoint(provider: str, base_url: str, model: str) -> str:
    normalized = normalize_provider(provider)
    _validate_http_transport(base_url, f"{normalized} base URL")
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if normalized == "http-json":
        return base_url
    if normalized == "openai-compatible":
        return (
            base_url
            if path.endswith("/chat/completions")
            else _append_endpoint(base_url, "chat/completions")
        )
    if normalized == "anthropic":
        return (
            base_url
            if path.endswith("/messages")
            else _append_endpoint(base_url, "messages")
        )
    if path.endswith(":generateContent"):
        return base_url
    escaped_model = urllib.parse.quote(model, safe="-._")
    suffix = f"{escaped_model}:generateContent"
    if not path.endswith("/models"):
        suffix = f"models/{suffix}"
    return _append_endpoint(base_url, suffix)


def _official_provider_host(provider: str, endpoint: str) -> bool:
    hostname = urllib.parse.urlsplit(endpoint).hostname
    return hostname == {
        "openai-compatible": "api.openai.com",
        "anthropic": "api.anthropic.com",
        "gemini": "generativelanguage.googleapis.com",
    }.get(provider)


def _wire_messages(request: Mapping[str, object]) -> list[dict[str, object]]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise AssistantBackendError("model request messages must be a list")
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            raise AssistantBackendError(
                f"model request message {index} is not an object"
            )
        role = str(raw.get("role") or "")
        if role not in {"user", "assistant"}:
            raise AssistantBackendError(
                f"model request message {index} has unsupported role {role!r}"
            )
        raw_images = raw.get("images_b64") or []
        if not isinstance(raw_images, list) or not all(
            isinstance(value, str) for value in raw_images
        ):
            raise AssistantBackendError(
                f"model request message {index} images_b64 must be a string list"
            )
        rows.append(
            {
                "role": role,
                "text": str(raw.get("text") or ""),
                "images_b64": list(raw_images),
            }
        )
    return rows


def _openai_content(row: Mapping[str, object]) -> str | list[dict[str, object]]:
    text = str(row.get("text") or "")
    images = row.get("images_b64") or []
    if not images:
        return text
    blocks: list[dict[str, object]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
        }
        for image in images
    )
    return blocks


def _anthropic_content(row: Mapping[str, object]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    text = str(row.get("text") or "")
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image,
            },
        }
        for image in row.get("images_b64") or []
    )
    return blocks or [{"type": "text", "text": " "}]


def _gemini_parts(row: Mapping[str, object]) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    text = str(row.get("text") or "")
    if text:
        parts.append({"text": text})
    parts.extend(
        {"inline_data": {"mime_type": "image/png", "data": image}}
        for image in row.get("images_b64") or []
    )
    return parts or [{"text": " "}]


def _token_limit(request: Mapping[str, object]) -> int | None:
    value = request.get("max_new_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(1, int(value))


def _temperature(request: Mapping[str, object]) -> float:
    value = request.get("temperature")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _request_for_provider(
    provider: str,
    model: str,
    request: Mapping[str, object],
    *,
    official_api: bool,
) -> dict[str, object]:
    rows = _wire_messages(request)
    system = str(request.get("system") or "")
    token_limit = _token_limit(request)
    if provider == "http-json":
        return dict(request)
    if provider == "openai-compatible":
        messages: list[dict[str, object]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(
            {"role": str(row["role"]), "content": _openai_content(row)} for row in rows
        )
        payload = {
            "model": model,
            "messages": messages,
            "temperature": _temperature(request),
        }
        if token_limit is not None:
            token_field = "max_completion_tokens" if official_api else "max_tokens"
            payload[token_field] = token_limit
        return payload
    if provider == "anthropic":
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": str(row["role"]), "content": _anthropic_content(row)}
                for row in rows
            ],
            "max_tokens": token_limit or _ANTHROPIC_DEFAULT_MAX_TOKENS,
            "temperature": _temperature(request),
        }
        if system:
            payload["system"] = system
        return payload
    generation_config: dict[str, object] = {"temperature": _temperature(request)}
    if token_limit is not None:
        generation_config["maxOutputTokens"] = token_limit
    payload = {
        "contents": [
            {
                "role": "model" if row["role"] == "assistant" else "user",
                "parts": _gemini_parts(row),
            }
            for row in rows
        ],
        "generationConfig": generation_config,
    }
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}
    return payload


def _text_from_blocks(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    texts = [
        str(block.get("text") or "")
        for block in value
        if isinstance(block, Mapping) and block.get("type") in {None, "text"}
    ]
    return "".join(texts)


def _normalized_response(
    provider: str, payload: Mapping[str, object]
) -> dict[str, object]:
    if provider == "http-json":
        output = dict(payload)
        output.setdefault("provider", provider)
        output.setdefault("provider_attempts", 1)
        return output
    if provider == "openai-compatible":
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AssistantBackendError("OpenAI-compatible response has no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise AssistantBackendError("OpenAI-compatible response choice is invalid")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise AssistantBackendError("OpenAI-compatible response has no message")
        text = _text_from_blocks(message.get("content"))
        finish_reason = str(choice.get("finish_reason") or "")
        usage = payload.get("usage")
    elif provider == "anthropic":
        text = _text_from_blocks(payload.get("content"))
        finish_reason = str(payload.get("stop_reason") or "")
        usage = payload.get("usage")
    else:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise AssistantBackendError("Gemini response has no candidates")
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise AssistantBackendError("Gemini response candidate is invalid")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else None
        text = _text_from_blocks(parts)
        finish_reason = str(candidate.get("finishReason") or "")
        usage = payload.get("usageMetadata")
    return {
        "ok": True,
        "text": text,
        "provider": provider,
        "provider_attempts": 1,
        "provider_finish_reason": finish_reason,
        "provider_finish_reasons": [finish_reason] if finish_reason else [],
        "provider_truncated": finish_reason.lower()
        in {"length", "max_tokens", "max-tokens"},
        "provider_usage": dict(usage) if isinstance(usage, Mapping) else {},
    }


def _response_error_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return "provider returned an error"
    error = payload.get("error")
    if isinstance(error, Mapping):
        return str(error.get("message") or error.get("status") or "provider error")
    return str(error or payload.get("message") or "provider error")


def _redact_secret(text: str, secret: str) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def _redact_response(value: object, secret: str) -> object:
    if not secret:
        return value
    if isinstance(value, str):
        return _redact_secret(value, secret)
    if isinstance(value, list):
        return [_redact_response(item, secret) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _redact_response(item, secret) for key, item in value.items()}
    return value


class ProviderClient:
    """Stateless adapter from the release contract to one public model API."""

    def __init__(
        self,
        provider: str,
        *,
        model: str,
        base_url: str,
        role: str,
        api_key_env: str = "",
        timeout_s: float = 180.0,
    ) -> None:
        self.provider = normalize_provider(provider)
        if self.provider != "http-json" and not model.strip():
            raise AssistantBackendError(f"{role} model is required for {self.provider}")
        _request_timeout(None, timeout_s, role)
        self.model = model.strip()
        self.role = role
        self.endpoint = _provider_endpoint(self.provider, base_url, self.model)
        self.api_key_env = resolve_api_key_env(
            self.provider,
            role,
            api_key_env,
            base_url=self.endpoint,
        )
        self.timeout_s = float(timeout_s)
        if (
            _official_provider_host(self.provider, self.endpoint)
            and not self.api_key_env
        ):
            raise AssistantBackendError(
                f"{role} credentials are required for the hosted {self.provider} API; "
                f"set INTERACTIVEWORLD_{role.upper()}_API_KEY or use "
                f"--{role}-api-key-env"
            )

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "InteractiveWorld/1.0",
        }
        if self.provider == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        if not api_key:
            return headers
        if self.provider in {"http-json", "openai-compatible"}:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.provider == "anthropic":
            headers["x-api-key"] = api_key
        else:
            headers["x-goog-api-key"] = api_key
        return headers

    def complete(
        self,
        request: Mapping[str, object],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        """Send one completion. The secret is read only while building headers."""
        api_key = self._api_key()
        provider_request = _request_for_provider(
            self.provider,
            self.model,
            request,
            official_api=_official_provider_host(self.provider, self.endpoint),
        )
        payload = self._post_provider(provider_request, api_key, timeout_s)
        return self._normalized_output(payload, request, api_key)

    def _api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if self.api_key_env and not api_key:
            raise AssistantBackendError(
                f"{self.role} API-key environment variable is no longer set"
            )
        if len(api_key.encode("utf-8")) > _MAX_API_KEY_BYTES or any(
            character in api_key for character in "\r\n\x00"
        ):
            raise AssistantBackendError(
                f"{self.role} API-key environment variable contains an unsafe value"
            )
        return api_key

    def _post_provider(
        self,
        provider_request: Mapping[str, object],
        api_key: str,
        timeout_s: float | None,
    ) -> Mapping[str, Any]:
        encoded = json.dumps(provider_request, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers=self._headers(api_key),
        )
        try:
            with _open_http_request(
                http_request,
                _request_timeout(timeout_s, self.timeout_s, self.role),
            ) as response:
                payload = json.loads(
                    _read_http_response(response, "provider response").decode("utf-8")
                )
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc, api_key)
        except urllib.error.URLError as exc:
            reason = _redact_secret(str(exc.reason), api_key)
            raise AssistantBackendError(
                f"{self.provider} request failed: {reason[:300]}"
            ) from exc
        except TimeoutError as exc:
            raise AssistantBackendError(f"{self.provider} request timed out") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssistantBackendError(
                f"{self.provider} response is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise AssistantBackendError(f"{self.provider} response must be an object")
        if self.provider != "http-json" and payload.get("error") is not None:
            message = _redact_secret(_response_error_text(payload), api_key)
            raise AssistantBackendError(f"{self.provider} error: {message[:300]}")
        return payload

    def _raise_http_error(
        self, error: urllib.error.HTTPError, api_key: str
    ) -> NoReturn:
        try:
            error_payload = json.loads(
                _read_http_response(error, "provider error response").decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            error_payload = None
        message = _redact_secret(_response_error_text(error_payload), api_key)
        raise AssistantBackendError(
            f"{self.provider} HTTP {error.code}: {message[:300]}"
        ) from error

    def _normalized_output(
        self,
        payload: Mapping[str, Any],
        request: Mapping[str, object],
        api_key: str,
    ) -> dict[str, object]:
        redacted = _redact_response(
            _normalized_response(self.provider, payload), api_key
        )
        if not isinstance(redacted, dict):
            raise AssistantBackendError(
                "normalized provider response must be an object"
            )
        output = redacted
        output["provider"] = self.provider
        output["provider_model"] = self.model
        output["provider_endpoint"] = self.endpoint
        output["provider_frame_count"] = _request_frame_count(request)
        configured_limit = _token_limit(request)
        output["provider_max_output_tokens"] = configured_limit or (
            _ANTHROPIC_DEFAULT_MAX_TOKENS if self.provider == "anthropic" else 0
        )
        return output


def _request_frame_count(request: Mapping[str, object]) -> int:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        len(message.get("images_b64") or [])
        for message in messages
        if isinstance(message, Mapping)
    )


class ProviderAssistant(HttpJsonAssistant):
    """Conversation-scoped Assistant backed by any supported provider."""

    def __init__(
        self,
        provider: str,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "",
        timeout_s: float = 180.0,
        max_new_tokens: int | None = 96,
        temperature: float = 0.0,
        frame_cap: int = 32,
    ) -> None:
        self.client = ProviderClient(
            provider,
            model=model,
            base_url=base_url,
            role="assistant",
            api_key_env=api_key_env,
            timeout_s=timeout_s,
        )
        super().__init__(
            url=self.client.endpoint,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            frame_cap=frame_cap,
        )
        self.name = self.client.provider

    def _post(
        self, request: Mapping[str, object], timeout_s: float | None = None
    ) -> dict[str, Any]:
        return dict(self.client.complete(request, timeout_s=timeout_s))


class HumanProviderGateway:
    """Loopback adapter preserving the simulator's frozen Human HTTP contract."""

    def __init__(
        self, client: ProviderClient, *, max_new_tokens: int | None = None
    ) -> None:
        if client.role != "human":
            raise AssistantBackendError(
                "Human gateway requires a human provider client"
            )
        self.client = client
        self.max_new_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None and int(max_new_tokens) > 0
            else None
        )
        # Keep the paid upstream API unreachable through a guessable local route.
        # This process-local token is never copied into release artifacts.
        self._route = f"/generate/{secrets.token_urlsafe(32)}"
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise AssistantBackendError("Human provider gateway is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}{self._route}"

    def start(self) -> HumanProviderGateway:
        if self._server is not None:
            return self
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                gateway._handle(self)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        # The protocol makes Human requests serially. A single-threaded server
        # bounds local resource use while preserving that request contract.
        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="interactiveworld-human-provider",
            daemon=True,
        )
        self._thread.start()
        return self

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path != self._route:
            self._write(handler, 404, {"ok": False, "error": "not found"})
            return
        raw_length = handler.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except ValueError:
            self._write(handler, 400, {"ok": False, "error": "invalid body length"})
            return
        if not 0 < length <= _MAX_GATEWAY_BODY_BYTES:
            self._write(handler, 413, {"ok": False, "error": "request body too large"})
            return
        try:
            request = json.loads(handler.rfile.read(length).decode("utf-8"))
            if not isinstance(request, Mapping):
                raise AssistantBackendError("request body must be a JSON object")
            provider_request = dict(request)
            if self.max_new_tokens is not None:
                provider_request["max_new_tokens"] = self.max_new_tokens
            response = self.client.complete(provider_request)
        except (AssistantBackendError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            response = {"ok": False, "error": str(exc)[:300]}
        self._write(handler, 200, response)

    @staticmethod
    def _write(
        handler: BaseHTTPRequestHandler,
        status: int,
        payload: Mapping[str, object],
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> HumanProviderGateway:
        return self.start()

    def __exit__(self, *unused: object) -> None:
        self.close()


def make_human_gateway(
    provider: str,
    *,
    model: str,
    base_url: str,
    api_key_env: str = "",
    timeout_s: float = 360.0,
    max_new_tokens: int | None = None,
) -> HumanProviderGateway:
    """Build, but do not start, a Human provider loopback gateway."""
    return HumanProviderGateway(
        ProviderClient(
            provider,
            model=model,
            base_url=base_url,
            role="human",
            api_key_env=api_key_env,
            timeout_s=timeout_s,
        ),
        max_new_tokens=max_new_tokens,
    )


def make_backend(kind: str, **options: Any) -> AssistantBackend:
    """Construct a provider-backed Assistant with a stable internal contract."""
    provider = normalize_provider(kind)
    base_url = str(options.pop("base_url", options.pop("url", "")))
    if not base_url:
        base_url = default_provider_base_url(provider)
    return ProviderAssistant(
        provider,
        model=str(options.pop("model", "")),
        base_url=base_url,
        api_key_env=str(options.pop("api_key_env", "")),
        **options,
    )

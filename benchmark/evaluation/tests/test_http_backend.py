# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import patch

from simulator.interaction.assistant_backend import (
    AssistantBackendError,
    HttpJsonAssistant,
    make_backend,
    select_temporal_image_batches,
)


class _Image:
    def save(self, stream, *, format: str) -> None:
        if format != "PNG":
            raise AssertionError(f"unexpected format: {format}")
        stream.write(b"portable-png")


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *unused: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]


class HttpJsonBackendTest(unittest.TestCase):
    def test_factory_exposes_only_portable_http_json(self) -> None:
        backend = make_backend(
            "http-json",
            url="http://127.0.0.1:8802/generate",
        )
        self.assertIsInstance(backend, HttpJsonAssistant)
        with self.assertRaisesRegex(AssistantBackendError, "supported backend"):
            make_backend("provider-specific")

    def test_cleartext_transport_is_loopback_only(self) -> None:
        with self.assertRaisesRegex(AssistantBackendError, "cleartext HTTP"):
            HttpJsonAssistant(url="http://models.example.com/generate")
        HttpJsonAssistant(url="https://models.example.com/generate")

    def test_request_and_response_follow_documented_contract(self) -> None:
        backend = HttpJsonAssistant(
            url="http://127.0.0.1:8802/generate",
            max_new_tokens=24,
            frame_cap=4,
        )
        messages = [
            {"role": "system", "content": "System prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _Image()},
                    {"type": "text", "text": "Latest segment."},
                ],
            },
        ]
        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            return_value=_Response({"ok": True, "text": "$silent$"}),
        ) as request:
            self.assertEqual(backend.generate(messages), "$silent$")

        payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["system"], "System prompt")
        self.assertEqual(payload["max_new_tokens"], 24)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["text"], "Latest segment.")
        self.assertEqual(len(payload["messages"][0]["images_b64"]), 1)
        self.assertEqual(backend.stats()["backend"], "http-json")

    def test_http_error_does_not_expose_a_gateway_route_token(self) -> None:
        token = "unrecorded-route-token"
        url = f"http://127.0.0.1:8802/generate/{token}"
        backend = HttpJsonAssistant(url=url)
        error = urllib.error.HTTPError(url, 404, "not found", {}, None)

        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            side_effect=error,
        ):
            with self.assertRaises(AssistantBackendError) as raised:
                backend.probe()

        self.assertNotIn(token, str(raised.exception))

    def test_temporal_selection_keeps_recent_turns(self) -> None:
        selected = select_temporal_image_batches(
            [["old-1", "old-2"], ["recent-1"], ["recent-2"]],
            frame_cap=3,
        )
        self.assertEqual(selected, [["old-2"], ["recent-1"], ["recent-2"]])

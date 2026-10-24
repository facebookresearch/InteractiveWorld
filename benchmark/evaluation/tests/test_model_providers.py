# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import http.client
import io
import json
import os
import threading
import unittest
import urllib.error
import urllib.parse
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from benchmark.evaluation.run_taskpack import _parse_args
from simulator.interaction.assistant_backend import (
    AssistantBackendError,
    HttpJsonAssistant,
    HumanProviderGateway,
    ProviderClient,
    resolve_api_key_env,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *unused: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]


class _OversizedResponse(_Response):
    def __init__(self) -> None:
        self.requested_size = -1

    def read(self, size: int = -1) -> bytes:
        self.requested_size = size
        if size < 0:
            raise AssertionError("response must be read with an explicit bound")
        return b"x" * (size + 1)

    def close(self) -> None:
        return None


def _request() -> dict[str, object]:
    return {
        "system": "Be useful.",
        "messages": [
            {"role": "user", "text": "First", "images_b64": ["cG5n"]},
            {"role": "assistant", "text": "Seen."},
            {"role": "user", "text": "Now act."},
        ],
        "max_new_tokens": 41,
        "temperature": 0.25,
    }


class ModelProviderTest(unittest.TestCase):
    def test_cli_configures_assistant_and_human_independently_without_secrets(
        self,
    ) -> None:
        assistant_secret = "assistant-value-never-in-namespace"
        human_secret = "human-value-never-in-namespace"
        environment = {
            "ASSISTANT_CREDENTIAL": assistant_secret,
            "HUMAN_CREDENTIAL": human_secret,
        }
        with patch.dict(os.environ, environment, clear=True):
            args = _parse_args(
                [
                    "--task-id",
                    "ASK-13000",
                    "--assistant",
                    "model",
                    "--assistant-provider",
                    "openai",
                    "--assistant-model",
                    "assistant-model",
                    "--assistant-api-key-env",
                    "ASSISTANT_CREDENTIAL",
                    "--human-provider",
                    "anthropic",
                    "--human-model",
                    "human-model",
                    "--human-api-key-env",
                    "HUMAN_CREDENTIAL",
                    "--human-tokens",
                    "91",
                ]
            )

        serialized = json.dumps(vars(args), sort_keys=True)
        self.assertEqual("openai-compatible", args.assistant_provider)
        self.assertEqual("anthropic", args.human_provider)
        self.assertEqual("ASSISTANT_CREDENTIAL", args.assistant_api_key_env)
        self.assertEqual("HUMAN_CREDENTIAL", args.human_api_key_env)
        self.assertEqual(91, args.human_tokens)
        self.assertNotIn(assistant_secret, serialized)
        self.assertNotIn(human_secret, serialized)

    def test_direct_cli_requires_local_compatible_endpoints_for_both_roles(
        self,
    ) -> None:
        cases = (
            [
                "--task-id",
                "ASK-13000",
                "--assistant",
                "model",
                "--assistant-provider",
                "openai-compatible",
                "--assistant-model",
                "local-assistant",
            ],
            [
                "--task-id",
                "ASK-13000",
                "--assistant",
                "model",
                "--human-provider",
                "openai-compatible",
                "--human-model",
                "local-human",
            ],
        )
        for arguments in cases:
            with (
                self.subTest(arguments=arguments),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                _parse_args(arguments)

    def test_openai_compatible_maps_multiturn_image_and_auth(self) -> None:
        secret = "openai-test-secret"
        with patch.dict(os.environ, {"ASSISTANT_KEY": secret}, clear=True):
            client = ProviderClient(
                "openai-compatible",
                model="gpt-test",
                base_url="https://api.openai.com/v1",
                role="assistant",
                api_key_env="ASSISTANT_KEY",
            )
            with patch(
                "simulator.interaction.assistant_backend._open_http_request",
                return_value=_Response(
                    {
                        "choices": [
                            {
                                "message": {"content": "Done"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"total_tokens": 12},
                    }
                ),
            ) as urlopen:
                result = client.complete(_request())

        sent = urlopen.call_args.args[0]
        payload = json.loads(sent.data.decode("utf-8"))
        self.assertEqual("https://api.openai.com/v1/chat/completions", sent.full_url)
        self.assertEqual(f"Bearer {secret}", sent.headers["Authorization"])
        self.assertEqual("gpt-test", payload["model"])
        self.assertEqual(41, payload["max_completion_tokens"])
        self.assertNotIn("max_tokens", payload)
        self.assertEqual("system", payload["messages"][0]["role"])
        image = payload["messages"][1]["content"][1]["image_url"]["url"]
        self.assertEqual("data:image/png;base64,cG5n", image)
        self.assertEqual("Done", result["text"])
        self.assertEqual("stop", result["provider_finish_reason"])

    def test_anthropic_uses_messages_blocks_and_header_key(self) -> None:
        with patch.dict(os.environ, {"HUMAN_KEY": "anthropic-secret"}, clear=True):
            client = ProviderClient(
                "anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com/v1",
                role="human",
                api_key_env="HUMAN_KEY",
            )
            with patch(
                "simulator.interaction.assistant_backend._open_http_request",
                return_value=_Response(
                    {
                        "content": [{"type": "text", "text": "<action>x</action>"}],
                        "stop_reason": "end_turn",
                    }
                ),
            ) as urlopen:
                result = client.complete(_request())

        sent = urlopen.call_args.args[0]
        payload = json.loads(sent.data.decode("utf-8"))
        self.assertEqual("https://api.anthropic.com/v1/messages", sent.full_url)
        self.assertEqual("anthropic-secret", sent.headers["X-api-key"])
        self.assertEqual("2023-06-01", sent.headers["Anthropic-version"])
        image = payload["messages"][0]["content"][1]["source"]
        self.assertEqual("cG5n", image["data"])
        self.assertEqual("<action>x</action>", result["text"])

    def test_gemini_uses_header_not_query_and_maps_roles(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-secret"}, clear=True):
            client = ProviderClient(
                "gemini",
                model="gemini-test",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                role="assistant",
            )
            with patch(
                "simulator.interaction.assistant_backend._open_http_request",
                return_value=_Response(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "Ready"}]},
                                "finishReason": "STOP",
                            }
                        ]
                    }
                ),
            ) as urlopen:
                result = client.complete(_request())

        sent = urlopen.call_args.args[0]
        payload = json.loads(sent.data.decode("utf-8"))
        self.assertEqual(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-test:generateContent",
            sent.full_url,
        )
        self.assertNotIn("gemini-secret", sent.full_url)
        self.assertEqual("gemini-secret", sent.headers["X-goog-api-key"])
        self.assertEqual("model", payload["contents"][1]["role"])
        self.assertEqual("Ready", result["text"])

    def test_credentials_are_independent_and_explicit_name_must_exist(self) -> None:
        environment = {
            "INTERACTIVEWORLD_ASSISTANT_API_KEY": "assistant-secret",
            "ANTHROPIC_API_KEY": "human-secret",
        }
        self.assertEqual(
            "INTERACTIVEWORLD_ASSISTANT_API_KEY",
            resolve_api_key_env("openai-compatible", "assistant", environ=environment),
        )
        self.assertEqual(
            "ANTHROPIC_API_KEY",
            resolve_api_key_env("anthropic", "human", environ=environment),
        )
        with self.assertRaisesRegex(AssistantBackendError, "is unset"):
            resolve_api_key_env("gemini", "human", "MISSING_KEY", environ=environment)

    def test_ambient_openai_key_is_not_sent_to_local_compatible_server(self) -> None:
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "ambient-production-key"}, clear=True
        ):
            client = ProviderClient(
                "openai-compatible",
                model="local-model",
                base_url="http://127.0.0.1:8000/v1",
                role="assistant",
            )
            with patch(
                "simulator.interaction.assistant_backend._open_http_request",
                return_value=_Response(
                    {"choices": [{"message": {"content": "local"}}]}
                ),
            ) as urlopen:
                client.complete(_request())

        sent = urlopen.call_args.args[0]
        payload = json.loads(sent.data.decode("utf-8"))
        self.assertEqual("", client.api_key_env)
        self.assertNotIn("Authorization", sent.headers)
        self.assertEqual(41, payload["max_tokens"])
        self.assertNotIn("max_completion_tokens", payload)

    def test_unset_token_limit_is_omitted_except_for_anthropic_requirement(
        self,
    ) -> None:
        request = _request()
        request.pop("max_new_tokens")
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openai-key", "ANTHROPIC_API_KEY": "anthropic-key"},
            clear=True,
        ):
            openai = ProviderClient(
                "openai-compatible",
                model="gpt-test",
                base_url="https://api.openai.com/v1",
                role="assistant",
            )
            anthropic = ProviderClient(
                "anthropic",
                model="claude-test",
                base_url="https://api.anthropic.com/v1",
                role="human",
            )
            responses = [
                _Response({"choices": [{"message": {"content": "ok"}}]}),
                _Response({"content": [{"type": "text", "text": "ok"}]}),
            ]
            with patch(
                "simulator.interaction.assistant_backend._open_http_request",
                side_effect=responses,
            ) as urlopen:
                openai.complete(request)
                anthropic.complete(request)

        openai_payload = json.loads(
            urlopen.call_args_list[0].args[0].data.decode("utf-8")
        )
        anthropic_payload = json.loads(
            urlopen.call_args_list[1].args[0].data.decode("utf-8")
        )
        self.assertNotIn("max_tokens", openai_payload)
        self.assertNotIn("max_completion_tokens", openai_payload)
        self.assertEqual(1024, anthropic_payload["max_tokens"])

    def test_unsafe_url_credentials_and_cleartext_remote_are_rejected(self) -> None:
        invalid_urls = (
            "https://user:secret@example.com/v1",
            "https://example.com/v1?key=secret",
            "https://example.com/v1?client_secret=secret",
            "https://example.com/v1?sig=SENSITIVE-SIGNED-VALUE",
            "http://example.com/v1",
            "https://example.com/v1\r\nX-Injected: yes",
            "https://example.com/" + "x" * 8192,
            "https://example.com:0/v1",
            "https://example.com:99999/v1",
            "https://[",
        )
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(AssistantBackendError) as raised:
                    ProviderClient(
                        "openai-compatible",
                        model="local-model",
                        base_url=url,
                        role="assistant",
                    )
                self.assertNotIn("secret", str(raised.exception))

    def test_query_values_cannot_enter_provider_audit_artifacts(self) -> None:
        signed_value = "SENSITIVE-SIGNED-VALUE"
        with self.assertRaisesRegex(AssistantBackendError, "query string") as raised:
            ProviderClient(
                "http-json",
                model="local-model",
                base_url=f"https://example.com/generate?sig={signed_value}",
                role="assistant",
            )
        self.assertNotIn(signed_value, str(raised.exception))

    def test_timeout_must_be_positive_and_finite_at_every_entry_point(self) -> None:
        for timeout in (0.0, -1.0, float("nan"), float("inf"), True, "30"):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(
                    AssistantBackendError, "positive and finite"
                ):
                    ProviderClient(
                        "http-json",
                        model="local-model",
                        base_url="http://127.0.0.1:9911/generate",
                        role="assistant",
                        timeout_s=timeout,
                    )
        client = ProviderClient(
            "http-json",
            model="local-model",
            base_url="http://127.0.0.1:9911/generate",
            role="assistant",
        )
        with self.assertRaisesRegex(AssistantBackendError, "positive and finite"):
            client.complete(_request(), timeout_s=float("nan"))

    def test_timeout_and_invalid_json_are_normalized_without_response_content(
        self,
    ) -> None:
        client = ProviderClient(
            "http-json",
            model="local-model",
            base_url="http://127.0.0.1:9911/generate",
            role="assistant",
        )
        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            side_effect=TimeoutError("sensitive timeout detail"),
        ):
            with self.assertRaisesRegex(AssistantBackendError, "timed out") as raised:
                client.complete(_request())
        self.assertNotIn("sensitive", str(raised.exception))

        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            return_value=_Response({}),
        ):
            with patch.object(_Response, "read", return_value=b"not-json SECRET"):
                with self.assertRaisesRegex(
                    AssistantBackendError, "UTF-8 JSON"
                ) as raised:
                    client.complete(_request())
        self.assertNotIn("SECRET", str(raised.exception))

    def test_unsafe_api_key_is_rejected_without_echo_or_request(self) -> None:
        unsafe_values = ("secret\r\nX-Injected: yes", "x" * 8193)
        for secret in unsafe_values:
            with (
                self.subTest(secret_length=len(secret)),
                patch.dict(os.environ, {"ASSISTANT_KEY": secret}, clear=True),
            ):
                client = ProviderClient(
                    "openai-compatible",
                    model="gpt-test",
                    base_url="https://api.openai.com/v1",
                    role="assistant",
                    api_key_env="ASSISTANT_KEY",
                )
                with patch(
                    "simulator.interaction.assistant_backend._open_http_request"
                ) as urlopen:
                    with self.assertRaises(AssistantBackendError) as raised:
                        client.complete(_request())

                self.assertNotIn(secret, str(raised.exception))
                urlopen.assert_not_called()

    def test_http_error_never_exposes_api_key(self) -> None:
        secret = "do-not-leak-this"
        body = json.dumps({"error": {"message": f"invalid {secret}"}}).encode()
        error = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401,
            "unauthorized",
            {},
            io.BytesIO(body),
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=True):
            client = ProviderClient(
                "openai-compatible",
                model="gpt-test",
                base_url="https://api.openai.com/v1",
                role="assistant",
            )
            with patch(
                "simulator.interaction.assistant_backend._open_http_request",
                side_effect=error,
            ):
                with self.assertRaises(AssistantBackendError) as raised:
                    client.complete(_request())
        self.assertNotIn(secret, str(raised.exception))
        self.assertIn("[REDACTED]", str(raised.exception))

    def test_success_and_error_responses_are_read_with_a_size_limit(self) -> None:
        client = ProviderClient(
            "http-json",
            model="local-model",
            base_url="http://127.0.0.1:9911/generate",
            role="assistant",
        )
        success = _OversizedResponse()
        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            return_value=success,
        ):
            with self.assertRaisesRegex(AssistantBackendError, "size limit"):
                client.complete(_request())
        self.assertGreater(success.requested_size, 0)

        generic = HttpJsonAssistant("http://127.0.0.1:9911/generate")
        generic_response = _OversizedResponse()
        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            return_value=generic_response,
        ):
            with self.assertRaisesRegex(AssistantBackendError, "size limit"):
                generic.probe()
        self.assertGreater(generic_response.requested_size, 0)

        error_body = _OversizedResponse()
        http_error = urllib.error.HTTPError(
            client.endpoint,
            500,
            "provider error",
            {},
            error_body,
        )
        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            side_effect=http_error,
        ):
            with self.assertRaisesRegex(AssistantBackendError, "size limit"):
                client.complete(_request())
        self.assertGreater(error_body.requested_size, 0)

    def test_cross_origin_redirect_is_rejected_before_credentials_are_forwarded(
        self,
    ) -> None:
        target_hits: list[str] = []

        class TargetHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                target_hits.append(self.headers.get("Authorization", ""))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true,"text":"unexpected"}')

            def log_message(self, _format: str, *args: object) -> None:
                del args

        target = HTTPServer(("127.0.0.1", 0), TargetHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        target_thread.start()
        target_url = f"http://127.0.0.1:{target.server_port}/capture"

        class SourceHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(302)
                self.send_header("Location", target_url)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, _format: str, *args: object) -> None:
                del args

        source = HTTPServer(("127.0.0.1", 0), SourceHandler)
        source_thread = threading.Thread(target=source.serve_forever, daemon=True)
        source_thread.start()
        secret = "redirect-secret"
        try:
            with patch.dict(os.environ, {"LOCAL_KEY": secret}, clear=True):
                client = ProviderClient(
                    "http-json",
                    model="local-model",
                    base_url=f"http://127.0.0.1:{source.server_port}/generate",
                    role="assistant",
                    api_key_env="LOCAL_KEY",
                )
                with self.assertRaisesRegex(
                    AssistantBackendError, "changed origin"
                ) as raised:
                    client.complete(_request())
        finally:
            source.shutdown()
            target.shutdown()
            source.server_close()
            target.server_close()
            source_thread.join()
            target_thread.join()

        self.assertEqual([], target_hits)
        self.assertNotIn(secret, str(raised.exception))

    def test_generic_response_cannot_echo_key_or_override_audit_endpoint(self) -> None:
        secret = "generic-secret-value"
        endpoint = "http://127.0.0.1:9911/generate"
        with patch.dict(os.environ, {"LOCAL_KEY": secret}, clear=True):
            client = ProviderClient(
                "http-json",
                model="local-model",
                base_url=endpoint,
                role="assistant",
                api_key_env="LOCAL_KEY",
            )
            with patch(
                "simulator.interaction.assistant_backend._open_http_request",
                return_value=_Response(
                    {
                        "ok": True,
                        "text": f"unsafe echo {secret}",
                        "provider_endpoint": f"https://example.com/?key={secret}",
                    }
                ),
            ):
                result = client.complete(_request())

        self.assertEqual("unsafe echo [REDACTED]", result["text"])
        self.assertEqual(endpoint, result["provider_endpoint"])

    def test_human_gateway_preserves_generic_contract(self) -> None:
        client = ProviderClient(
            "http-json",
            model="local-human",
            base_url="http://127.0.0.1:9911/generate",
            role="human",
        )
        gateway = HumanProviderGateway(client, max_new_tokens=77)
        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            return_value=_Response({"ok": True, "text": "<action>[Wait]</action>"}),
        ) as upstream:
            gateway.start()
            try:
                parsed = urllib.parse.urlsplit(gateway.url)
                connection = http.client.HTTPConnection(parsed.hostname, parsed.port)
                connection.request(
                    "POST",
                    parsed.path,
                    body=json.dumps(_request()),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                output = json.loads(response.read().decode("utf-8"))
                connection.close()
            finally:
                gateway.close()

        self.assertEqual(200, response.status)
        self.assertEqual("<action>[Wait]</action>", output["text"])
        self.assertEqual("http-json", output["provider"])
        forwarded = json.loads(upstream.call_args.args[0].data.decode("utf-8"))
        expected = _request()
        expected["max_new_tokens"] = 77
        self.assertEqual(expected, forwarded)

    def test_human_gateway_rejects_the_guessable_unauthenticated_route(self) -> None:
        client = ProviderClient(
            "http-json",
            model="local-human",
            base_url="http://127.0.0.1:9911/generate",
            role="human",
        )
        gateway = HumanProviderGateway(client)
        with patch(
            "simulator.interaction.assistant_backend._open_http_request",
            return_value=_Response({"ok": True, "text": "unused"}),
        ) as upstream:
            gateway.start()
            try:
                parsed = urllib.parse.urlsplit(gateway.url)
                connection = http.client.HTTPConnection(parsed.hostname, parsed.port)
                connection.request(
                    "POST",
                    "/generate",
                    body=json.dumps(_request()),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response.read()
                connection.close()
            finally:
                gateway.close()

        self.assertNotEqual("/generate", parsed.path)
        self.assertEqual(404, response.status)
        upstream.assert_not_called()

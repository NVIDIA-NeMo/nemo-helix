# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``ExampleInferenceMiddleware``.

These tests exercise IGW + Models in-process via ASGI and route the proxy
step through a single ``pytest_httpserver`` (the harness's ``mock_nim``).

The example plugin is a reference-only package and is not installed in the
normal test venv, so its ``nemo.inference_middleware`` entry point is not
discoverable via ``importlib.metadata``. Tests therefore construct an
``ExampleInferenceMiddleware`` instance directly and register it under the
canonical entry-point name with :meth:`IGWPluginHarness.use_plugin`.

Mock responses are keyed by the value that arrives in ``body["model"]`` at
the upstream (the *served* model name after IGW's served-name rewrite).
``assert_called_once`` / ``assert_no_calls_to`` replace the v1 pattern of
matching response IDs returned through the proxy.
"""

import uuid

import pytest
from nemo_example_plugin.middleware import ExampleInferenceMiddleware
from nemo_example_plugin.middleware_config import ExampleMiddlewareConfig
from nemo_helix.types.inference.middleware_call_param import MiddlewareCallParam
from nhx.core.inference_gateway.testing.harness import IGWPluginHarness
from nhx.testing.mock_chat_completions import (
    ChatCompletion,
    ChatCompletionStream,
    chat_completion,
    chat_completion_chunk,
)

pytestmark = [pytest.mark.integration]

EXAMPLE_PLUGIN_NAME = "nemo-example-middleware"
EXAMPLE_PLUGIN_CONFIG_TYPE = ExampleMiddlewareConfig.__entity_type__


def _build_middleware_call(
    *,
    blocked_keywords: list[str],
    block_message: str = "Blocked.",
) -> MiddlewareCallParam:
    return {
        "name": EXAMPLE_PLUGIN_NAME,
        "config_type": EXAMPLE_PLUGIN_CONFIG_TYPE,
        "config": {
            "blocked_keywords": blocked_keywords,
            "block_message": block_message,
        },
    }


class TestRequestMiddleware:
    """Request-phase coverage for the example keyword filter."""

    BACKEND_RESPONSE = "This response came from the backend model."

    def test_safe_input_is_proxied_to_backend(self, igw_plugin_harness: IGWPluginHarness) -> None:
        h = igw_plugin_harness
        test_id = uuid.uuid4().hex[:8]
        model_name = f"example-model-{test_id}"
        virtual_model_name = f"example-vm-{test_id}"

        h.mock_chat_completions(
            model_name,
            responses=[
                ChatCompletion(
                    body=chat_completion(
                        content=self.BACKEND_RESPONSE,
                        id_="chatcmpl-example-safe",
                    )
                )
            ],
        )
        h.add_provider(
            workspace=h.workspace,
            name=f"example-provider-{test_id}",
            served_models={model_name: model_name},
        )

        with h.use_plugin(EXAMPLE_PLUGIN_NAME, ExampleInferenceMiddleware()):
            h.add_virtual_model(
                workspace=h.workspace,
                name=virtual_model_name,
                default_model_entity=f"{h.workspace}/{model_name}",
                request_middleware=[_build_middleware_call(blocked_keywords=["violence"])],
            )

            response = h.chat_completions(
                workspace=h.workspace,
                body={
                    "model": virtual_model_name,
                    "messages": [{"role": "user", "content": "Tell me about flowers."}],
                },
            )

        h.assert_called_once(model_name)
        assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    def test_blocked_input_short_circuits_proxy(self, igw_plugin_harness: IGWPluginHarness) -> None:
        h = igw_plugin_harness
        test_id = uuid.uuid4().hex[:8]
        model_name = f"example-model-{test_id}"
        virtual_model_name = f"example-vm-{test_id}"
        block_message = "That topic is off-limits."

        h.add_provider(
            workspace=h.workspace,
            name=f"example-provider-{test_id}",
            served_models={model_name: model_name},
        )

        with h.use_plugin(EXAMPLE_PLUGIN_NAME, ExampleInferenceMiddleware()):
            h.add_virtual_model(
                workspace=h.workspace,
                name=virtual_model_name,
                default_model_entity=f"{h.workspace}/{model_name}",
                request_middleware=[
                    _build_middleware_call(
                        blocked_keywords=["violence"],
                        block_message=block_message,
                    )
                ],
            )

            response = h.chat_completions(
                workspace=h.workspace,
                body={
                    "model": virtual_model_name,
                    "messages": [{"role": "user", "content": "Tell me about violence."}],
                },
            )

        # The blocker must not call the backend at all.
        h.assert_no_calls_to(model_name)
        assert response == {
            "id": "blocked",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": block_message},
                    "finish_reason": "content_filter",
                }
            ],
        }


class TestResponseMiddleware:
    """Response-phase coverage for the example redaction filter."""

    def test_backend_response_is_redacted(self, igw_plugin_harness: IGWPluginHarness) -> None:
        h = igw_plugin_harness
        test_id = uuid.uuid4().hex[:8]
        model_name = f"example-model-{test_id}"
        virtual_model_name = f"example-vm-{test_id}"

        h.mock_chat_completions(
            model_name,
            responses=[
                ChatCompletion(
                    body=chat_completion(
                        content="The secret is out.",
                        id_="chatcmpl-example-redacted",
                    )
                )
            ],
        )
        h.add_provider(
            workspace=h.workspace,
            name=f"example-provider-{test_id}",
            served_models={model_name: model_name},
        )

        with h.use_plugin(EXAMPLE_PLUGIN_NAME, ExampleInferenceMiddleware()):
            h.add_virtual_model(
                workspace=h.workspace,
                name=virtual_model_name,
                default_model_entity=f"{h.workspace}/{model_name}",
                response_middleware=[_build_middleware_call(blocked_keywords=["secret"])],
            )

            response = h.chat_completions(
                workspace=h.workspace,
                body={
                    "model": virtual_model_name,
                    "messages": [{"role": "user", "content": "Share the answer."}],
                },
            )

        h.assert_called_once(model_name)
        content = response["choices"][0]["message"]["content"]
        assert "secret" not in content
        assert "[REDACTED]" in content

    def test_streaming_backend_response_is_redacted(self, igw_plugin_harness: IGWPluginHarness) -> None:
        """Streaming response middleware path: SSE chunks pass through ``_redact_stream``.

        Mirrors :meth:`test_backend_response_is_redacted` but with
        ``stream=True``. The mock NIM emits a multi-chunk SSE response;
        IGW parses it as an :class:`AsyncIterator[dict]` and hands it to
        :meth:`ExampleInferenceMiddleware.process_response`, which wraps
        it with :func:`_redact_stream`. IGW re-encodes the result as SSE
        for the client; :meth:`stream_chat_completions` parses those
        chunks back into a list for assertions.

        The "secret" keyword is split between the second and third chunks
        (``"sec"`` then ``"ret"``) to exercise the cross-chunk lookahead
        buffer in :func:`_redact_stream`. With per-chunk regex this would
        miss the match; the buffer is the whole point of testing it
        through the harness instead of via a unit test on a single chunk.
        """
        h = igw_plugin_harness
        test_id = uuid.uuid4().hex[:8]
        model_name = f"example-stream-model-{test_id}"
        virtual_model_name = f"example-stream-vm-{test_id}"

        h.mock_chat_completions(
            model_name,
            responses=[
                ChatCompletionStream(
                    chunks=[
                        chat_completion_chunk(content="The ", role="assistant", id_="chatcmpl-stream-1"),
                        chat_completion_chunk(content="sec", id_="chatcmpl-stream-1"),
                        chat_completion_chunk(content="ret is out.", id_="chatcmpl-stream-1"),
                        chat_completion_chunk(content="", id_="chatcmpl-stream-1", finish_reason="stop"),
                    ]
                )
            ],
        )
        h.add_provider(
            workspace=h.workspace,
            name=f"example-stream-provider-{test_id}",
            served_models={model_name: model_name},
        )

        with h.use_plugin(EXAMPLE_PLUGIN_NAME, ExampleInferenceMiddleware()):
            h.add_virtual_model(
                workspace=h.workspace,
                name=virtual_model_name,
                default_model_entity=f"{h.workspace}/{model_name}",
                response_middleware=[_build_middleware_call(blocked_keywords=["secret"])],
            )

            chunks = h.stream_chat_completions(
                workspace=h.workspace,
                body={
                    "model": virtual_model_name,
                    "messages": [{"role": "user", "content": "Share the answer."}],
                },
            )

        h.assert_called_once(model_name)
        joined = "".join(
            (choice.get("delta") or {}).get("content") or ""
            for chunk in chunks
            for choice in (chunk.get("choices") or [])
        )
        assert "secret" not in joined
        assert "[REDACTED]" in joined

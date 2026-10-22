# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-visible SSE framing for streams proxied through a VirtualModel."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from nemo_helix_plugin.inference_middleware import InferenceResponse, NemoInferenceMiddleware
from nemo_helix_plugin.inference_middleware_models import VirtualModel
from nhx.core.inference_gateway.api.dependencies import global_middleware_registry, global_virtual_model_cache
from nhx.core.inference_gateway.api.middleware_registry import MiddlewareRegistry, ResolvedMiddlewareCall
from nhx.core.inference_gateway.api.proxy import _uses_named_sse_events, stream_response_result
from nhx.core.inference_gateway.api.virtual_model_cache import VirtualModelCache

WORKSPACE = "e2e-test"
VM_NAME = "sse-vm"
MODEL_ENTITY = f"{WORKSPACE}/meta_llama-3.2-1b-instruct"
SERVED_MODEL = "meta/llama-3.2-1b-instruct"

ANTHROPIC_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": SERVED_MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
]

RESPONSES_EVENTS: list[dict[str, Any]] = [
    {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_1",
            "object": "response",
            "status": "in_progress",
            "model": SERVED_MODEL,
            "output": [],
        },
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hi",
    },
    {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {"id": "resp_1", "object": "response", "status": "completed", "model": SERVED_MODEL, "output": []},
    },
]

CHAT_CHUNKS: list[dict[str, Any]] = [
    {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": SERVED_MODEL,
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    },
    {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": SERVED_MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]


def _named_event_stream(events: list[dict[str, Any]]) -> bytes:
    """Upstream wire format for Anthropic Messages and OpenAI Responses: ``event:`` + ``data:``, no ``[DONE]``."""
    return "".join(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n" for ev in events).encode()


def _chat_stream(chunks: list[dict[str, Any]]) -> bytes:
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()


def _parse_client_stream(text: str) -> list[tuple[str | None, str]]:
    """Split a client-visible SSE body into ``(event, data)`` pairs, one per blank-line-terminated event."""
    parsed: list[tuple[str | None, str]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event: str | None = None
        data: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
        parsed.append((event, "\n".join(data)))
    return parsed


@pytest.fixture
def sse_upstream(app: FastAPI, mock_proxy_response):
    """Route ``VM_NAME`` to the test model entity and let each test choose the upstream SSE body."""
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([VirtualModel(name=VM_NAME, workspace=WORKSPACE, default_model_entity=MODEL_ENTITY)])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    def _set_body(payload: bytes) -> None:
        async def _iter_any():
            # Split mid-line so the parser has to buffer across chunk boundaries.
            midpoint = len(payload) // 2
            yield payload[:midpoint]
            yield payload[midpoint:]

        mock_proxy_response.headers = CIMultiDictProxy(CIMultiDict([("content-type", "text/event-stream")]))
        mock_proxy_response.content.iter_any = Mock(return_value=_iter_any())

    return _set_body


@pytest.fixture
def passthrough_stream_middleware(app: FastAPI):
    """Register response middleware on ``VM_NAME`` that re-wraps the stream, like guardrails and example-plugin do."""
    seen: list[dict[str, Any]] = []
    plugin = MagicMock(spec=NemoInferenceMiddleware)

    async def process_response(ctx, response: InferenceResponse, cfg):
        upstream = response.result
        assert not isinstance(upstream, dict)

        async def _wrapped():
            async for chunk in upstream:
                seen.append(chunk)
                yield chunk

        response.result = _wrapped()
        return response

    plugin.process_response = AsyncMock(side_effect=process_response)
    registry = MiddlewareRegistry(plugins={"sse-plugin": plugin})
    registry.response_middleware_calls[(WORKSPACE, VM_NAME)] = [
        ResolvedMiddlewareCall(plugin_name="sse-plugin", config_type="t", resolved_config={})
    ]
    app.dependency_overrides[global_middleware_registry] = lambda: registry
    return seen


def _expected_named_events(events: list[dict[str, Any]]) -> list[tuple[str | None, str]]:
    return [(ev["type"], ev["type"]) for ev in events]


def _event_types(parsed: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    return [(event, json.loads(data)["type"]) for event, data in parsed]


@pytest.mark.parametrize("with_middleware", [False, True], ids=["no-middleware", "stream-middleware"])
def test_anthropic_messages_stream_keeps_event_lines_and_has_no_done(
    request: pytest.FixtureRequest, client: TestClient, sse_upstream, with_middleware: bool
):
    """Anthropic SDKs dispatch on the SSE ``event:`` field; dropping it leaves Claude Code with no events."""
    if with_middleware:
        request.getfixturevalue("passthrough_stream_middleware")
    sse_upstream(_named_event_stream(ANTHROPIC_EVENTS))

    response = client.post(
        f"/v2/workspaces/{WORKSPACE}/openai/-/v1/messages",
        json={"model": VM_NAME, "max_tokens": 16, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = _parse_client_stream(response.text)
    assert "[DONE]" not in [data for _, data in parsed]
    assert _event_types(parsed) == _expected_named_events(ANTHROPIC_EVENTS)
    message_start = json.loads(parsed[0][1])
    assert message_start["message"]["model"] == MODEL_ENTITY


@pytest.mark.parametrize("with_middleware", [False, True], ids=["no-middleware", "stream-middleware"])
def test_openai_responses_stream_keeps_event_lines_and_has_no_done(
    request: pytest.FixtureRequest, client: TestClient, sse_upstream, with_middleware: bool
):
    """The Responses API has no ``[DONE]`` sentinel; ``response.completed`` ends the stream."""
    if with_middleware:
        request.getfixturevalue("passthrough_stream_middleware")
    sse_upstream(_named_event_stream(RESPONSES_EVENTS))

    response = client.post(
        f"/v2/workspaces/{WORKSPACE}/openai/-/v1/responses",
        json={"model": VM_NAME, "stream": True, "input": "hi"},
    )

    assert response.status_code == 200, response.text
    parsed = _parse_client_stream(response.text)
    assert "[DONE]" not in [data for _, data in parsed]
    assert _event_types(parsed) == _expected_named_events(RESPONSES_EVENTS)
    assert json.loads(parsed[-1][1])["response"]["model"] == MODEL_ENTITY


@pytest.mark.parametrize("with_middleware", [False, True], ids=["no-middleware", "stream-middleware"])
def test_chat_completions_stream_framing_unchanged(
    request: pytest.FixtureRequest, client: TestClient, sse_upstream, with_middleware: bool
):
    """Chat Completions clients stop on ``data: [DONE]`` and expect no ``event:`` lines."""
    if with_middleware:
        request.getfixturevalue("passthrough_stream_middleware")
    sse_upstream(_chat_stream(CHAT_CHUNKS))

    response = client.post(
        f"/v2/workspaces/{WORKSPACE}/openai/-/v1/chat/completions",
        json={"model": VM_NAME, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200, response.text
    expected = "".join(f"data: {json.dumps({**chunk, 'model': MODEL_ENTITY})}\n\n" for chunk in CHAT_CHUNKS)
    assert response.content == f"{expected}data: [DONE]\n\n".encode()


@pytest.mark.parametrize(
    ("trailing_uri", "named"),
    [
        ("v1/messages", True),
        ("v1/messages/", True),
        ("v1/responses", True),
        ("v1/responses/resp_123", True),
        ("v1/chat/completions", False),
        ("v1/completions", False),
        ("v1/messages/count_tokens", False),
        ("v1/responses/resp_123/input_items", False),
    ],
)
def test_named_event_framing_follows_client_endpoint(trailing_uri: str, named: bool):
    """Streaming retrieval (``GET v1/responses/{id}?stream=true``) replays Responses events, so it is named too."""
    assert _uses_named_sse_events(trailing_uri) is named


async def test_event_name_with_line_break_is_not_written_as_framing():
    """A ``type`` containing a line break must not split one chunk into forged SSE events."""
    forged = {"type": 'x\nevent: message_stop\ndata: {"type": "message_stop"}\n\n'}

    async def _chunks():
        yield forged

    response = await stream_response_result(_chunks(), 200, {}, named_events=True)
    body = b"".join([chunk async for chunk in response.body_iterator]).decode()

    assert _parse_client_stream(body) == [(None, json.dumps(forged))]

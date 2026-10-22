# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-visible SSE framing for streams proxied through a VirtualModel."""

import json
from collections.abc import Callable
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
from nhx.core.inference_gateway.api.virtual_model_cache import VirtualModelCache

WORKSPACE = "e2e-test"
VM_NAME = "sse-vm"
MODEL_ENTITY = f"{WORKSPACE}/meta_llama-3.2-1b-instruct"
SERVED_MODEL = "meta/llama-3.2-1b-instruct"


def _anthropic_events(model: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
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


def _responses_events(model: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp_1", "object": "response", "status": "in_progress", "model": model, "output": []},
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
            "response": {"id": "resp_1", "object": "response", "status": "completed", "model": model, "output": []},
        },
    ]


def _chat_chunks(model: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        },
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]


def _named_event_stream(events: list[dict[str, Any]]) -> bytes:
    """Anthropic Messages and OpenAI Responses wire format: ``event:`` + ``data:`` per event, no ``[DONE]``."""
    return "".join(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n" for ev in events).encode()


def _chat_stream(chunks: list[dict[str, Any]]) -> bytes:
    """Chat Completions wire format: ``data:`` per chunk, ending with ``data: [DONE]``."""
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()


@pytest.fixture
def sse_upstream(app: FastAPI, mock_proxy_response) -> Callable[[bytes], None]:
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
def passthrough_stream_middleware(app: FastAPI) -> list[dict[str, Any]]:
    """Register response middleware on ``VM_NAME`` that re-wraps the stream, like guardrails and example-plugin do.

    Returns the chunks the middleware saw.
    """
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


@pytest.mark.parametrize("with_middleware", [False, True], ids=["no-middleware", "stream-middleware"])
@pytest.mark.parametrize(
    ("path", "body", "events", "encode"),
    [
        pytest.param(
            "v1/messages",
            {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
            _anthropic_events,
            _named_event_stream,
            id="anthropic-messages",
        ),
        pytest.param("v1/responses", {"input": "hi"}, _responses_events, _named_event_stream, id="openai-responses"),
        pytest.param(
            "v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
            _chat_chunks,
            _chat_stream,
            id="chat-completions",
        ),
    ],
)
def test_stream_reaches_client_in_its_upstream_wire_format(
    request: pytest.FixtureRequest,
    client: TestClient,
    sse_upstream: Callable[[bytes], None],
    with_middleware: bool,
    path: str,
    body: dict[str, Any],
    events: Callable[[str], list[dict[str, Any]]],
    encode: Callable[[list[dict[str, Any]]], bytes],
):
    """The client gets the upstream stream byte for byte, apart from the served model name.

    Anthropic SDKs dispatch on ``event:``, so dropping it leaves Claude Code with no events; Messages and
    Responses have no ``[DONE]``; Chat Completions clients stop on ``[DONE]`` and expect no ``event:`` lines.
    """
    seen = request.getfixturevalue("passthrough_stream_middleware") if with_middleware else None
    sse_upstream(encode(events(SERVED_MODEL)))

    response = client.post(
        f"/v2/workspaces/{WORKSPACE}/openai/-/{path}",
        json={"model": VM_NAME, "stream": True, **body},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == encode(events(MODEL_ENTITY))
    if seen is not None:
        assert seen == events(MODEL_ENTITY)

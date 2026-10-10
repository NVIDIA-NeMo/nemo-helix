# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import FastAPI
from nemo_agents_plugin.api.v2 import gateway as gateway_module
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.client_disconnect import CLIENT_CLOSED_REQUEST_STATUS
from nemo_agents_plugin.entities import AgentDeployment
from nemo_helix_plugin.dependencies import get_effective_principal_id

_CHAT_PATH = "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions"
_CHAT_BODY = {"messages": [{"role": "user", "content": "hi"}]}

PostThenDisconnect = Callable[..., Awaitable[int | None]]


class _SlowAgent:
    """Fake agent that starts work, optionally sends a response prefix, then never finishes."""

    def __init__(self, *, headers: dict[str, str] | None = None, prefix: bytes = b"") -> None:
        self._headers = headers
        self._prefix = prefix
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def _hang(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def _body(self) -> AsyncIterator[bytes]:
        try:
            yield self._prefix
            await self._hang()
        except GeneratorExit:
            self.cancelled.set()
            raise

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.started.set()
        if self._headers is None:
            await self._hang()
        return httpx.Response(200, headers=self._headers, stream=_AsyncStream(self._body()))


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: AsyncIterator[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._chunks:
            yield chunk


def _gateway_app() -> FastAPI:
    deployment = AgentDeployment(
        name="calc-dep", workspace="default", agent="calc", status="running", endpoint="http://localhost:9001"
    )
    entity_client = AsyncMock()
    entity_client.list = AsyncMock(return_value=MagicMock(data=[deployment]))
    app = FastAPI()
    app.include_router(gateway_module.router, prefix="/apis/agents/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    app.dependency_overrides[get_effective_principal_id] = lambda: "caller"
    return app


async def _proxy_then_disconnect(
    post_then_disconnect: PostThenDisconnect, agent: _SlowAgent, body: dict[str, Any]
) -> int | None:
    real_async_client = httpx.AsyncClient

    def _client_reaching_agent(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(agent.handle), **kwargs)

    with patch.object(gateway_module.httpx, "AsyncClient", side_effect=_client_reaching_agent):
        return await post_then_disconnect(_gateway_app(), _CHAT_PATH, body, agent.started)


async def test_disconnect_before_agent_responds_cancels_agent_request(
    post_then_disconnect: PostThenDisconnect,
) -> None:
    agent = _SlowAgent()

    status = await _proxy_then_disconnect(post_then_disconnect, agent, _CHAT_BODY)

    assert agent.cancelled.is_set()
    assert status == CLIENT_CLOSED_REQUEST_STATUS


async def test_disconnect_while_buffering_json_body_cancels_agent_request(
    post_then_disconnect: PostThenDisconnect,
) -> None:
    agent = _SlowAgent(headers={"content-type": "application/json"}, prefix=b'{"choices": [')

    status = await _proxy_then_disconnect(post_then_disconnect, agent, _CHAT_BODY)

    assert agent.cancelled.is_set()
    assert status == CLIENT_CLOSED_REQUEST_STATUS


async def test_disconnect_mid_sse_stream_closes_agent_request(post_then_disconnect: PostThenDisconnect) -> None:
    agent = _SlowAgent(headers={"content-type": "text/event-stream"}, prefix=b'data: {"delta": "a"}\n\n')

    status = await _proxy_then_disconnect(post_then_disconnect, agent, {**_CHAT_BODY, "stream": True})

    assert agent.cancelled.is_set()
    assert status == 200

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the LangGraph agent over HTTP, in the shape Agent Hardener calls a victim with.

    POST /v1/chat/completions   OpenAI-compatible; the one endpoint attack + benign traffic uses
    GET  /health                liveness, polled while the container comes up

The request/response shapes are OpenAI's on purpose: ``EndpointContract`` (agent_hardener/endpoint.py)
posts ``{"model": ..., "messages": [...]}`` and reads the reply back out of
``choices[0].message.content``. The benign-suite prober parses that path strictly, so the response
must carry it exactly — no ``output``/``response`` shorthand.

Run it:

    INFERENCE_API_KEY=... uv run python examples/langgraph-victim/server.py
    curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
         -d '{"model": "victim", "messages": [{"role": "user", "content": "list files here"}]}'
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import nemo_relay
import uvicorn
from agent import build_agent
from fastapi import FastAPI
from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler
from nemo_relay.plugin import PluginConfig
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class Message(BaseModel):
    """One chat message in the incoming request."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Body for ``POST /v1/chat/completions`` — the OpenAI chat-completions request shape."""

    messages: list[Message]
    model: str = "langgraph-victim"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start Relay before the first request.

    ``PluginConfig()`` is the empty base config: Relay layers the *discovered* ``plugins.toml``
    (``/etc/nemo-relay/plugins.toml``) over it. Agent Hardener uploads each round's guardrails there
    before restarting the victim, and nothing activates them — nor the ATOF sink the run's preflight
    insists on — without this call.
    """
    await nemo_relay.plugin.initialize(PluginConfig())
    yield


def create_app() -> FastAPI:
    """Build the app. Compiles the graph once for the process; each request is a fresh run."""
    app = FastAPI(title="LangGraph victim", lifespan=lifespan)
    agent = build_agent()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest) -> dict[str, Any]:
        # One Relay agent scope per request, with the LangGraph callback handler on the invoke:
        # the handler records the graph/LLM runs, and the tools node (see agent.create_tool_node)
        # records the tool calls. Everything the request does hangs off this scope.
        with nemo_relay.scope.scope("langgraph-victim", nemo_relay.ScopeType.Agent):
            result = await agent.ainvoke(
                {"messages": [(m.role, m.content) for m in body.messages]},
                config={"callbacks": [NemoRelayCallbackHandler()]},
            )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["messages"][-1].content},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def run_server(*, host: str = "0.0.0.0", port: int | None = None) -> None:  # noqa: S104 - containers bind all interfaces
    """Serve the agent with uvicorn. Binds $PORT (default 8000), the port a victim is probed on."""
    uvicorn.run(create_app(), host=host, port=port or int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    run_server()

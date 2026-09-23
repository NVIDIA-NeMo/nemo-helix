# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the framework-free agent over HTTP, in the shape Agent Hardener calls a victim with.

    POST /v1/chat/completions   OpenAI-compatible; the one endpoint attack + benign traffic uses
    GET  /health                liveness, polled while the container comes up

The response must carry ``choices[0].message.content`` exactly — the benign-suite prober parses
that path strictly.

Run it:

    INFERENCE_API_KEY=... uv run python server.py
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
from agent import run_turn
from fastapi import FastAPI
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
    model: str = "other-victim"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start Relay before the first request.

    ``PluginConfig()`` is the empty base config: Relay layers the *discovered* ``plugins.toml``
    (``/etc/nemo-relay/plugins.toml``) over it. Agent Hardener uploads each round's guardrails there
    before restarting the victim, and nothing activates them — nor the ATOF sink the run's
    preflight insists on — without this call.
    """
    await nemo_relay.plugin.initialize(PluginConfig())
    yield


def create_app() -> FastAPI:
    """Build the app. Each request is one agent turn through the hand-rolled loop."""
    app = FastAPI(title="Framework-free victim", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest) -> dict[str, Any]:
        # One Relay agent scope per request, so the turn's tool events nest under one root.
        with nemo_relay.scope.scope("other-victim", nemo_relay.ScopeType.Agent):
            answer = await run_turn([{"role": m.role, "content": m.content} for m in body.messages])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the example plugin's NemoFunction surface.

Exercises two layers in lockstep:

1. **Direct unit** — instantiate the function class and ``await`` its
   ``run`` method. No FastAPI, no platform.
2. **Route adapter** — mount the auto-derived router under the
   canonical ``/apis/example/v2/workspaces/{workspace}`` prefix and
   hit it with :class:`~fastapi.testclient.TestClient`. Pins the
   wire shape (URL, content type, JSON body / NDJSON frame stream).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_example_plugin.functions.greet import (
    CountFunction,
    CountSpec,
    GreetFunction,
    GreetSpec,
)
from nemo_helix_plugin.function_context import FunctionContext
from nemo_helix_plugin.functions.routes import NDJSON_MEDIA_TYPE, add_function_routes

# ---------------------------------------------------------------------------
# 1. Direct unit-test path — no FastAPI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_greet_returns_message_and_workspace() -> None:
    fn = GreetFunction()
    result = await fn.run(GreetSpec(name="Ada"), ctx=FunctionContext(workspace="dev"))
    assert result.message == "Hello, Ada!"
    assert result.workspace == "dev"


@pytest.mark.asyncio
async def test_count_streams_terminator() -> None:
    fn = CountFunction()
    frames = [frame async for frame in fn.run(CountSpec(upto=2))]
    # 2 ticks + 1 done — the terminator is the contract clients rely on.
    # Pull `kind` off each frame as a dict so we don't depend on a
    # specific frame class for the discriminator type.
    kinds = [frame.model_dump()["kind"] for frame in frames]
    assert kinds == ["tick", "tick", "done"]


# ---------------------------------------------------------------------------
# 2. Route adapter — auto-derived router under the canonical prefix
# ---------------------------------------------------------------------------


def _build_app() -> FastAPI:
    app = FastAPI()
    # Heartbeat off keeps the streaming test deterministic without
    # needing a fake clock.
    app.include_router(
        add_function_routes(GreetFunction),
        prefix="/apis/example/v2/workspaces/{workspace}",
    )
    app.include_router(
        add_function_routes(CountFunction, heartbeat_interval_seconds=0),
        prefix="/apis/example/v2/workspaces/{workspace}",
    )
    return app


def test_route_returns_json_with_workspace_in_body() -> None:
    client = TestClient(_build_app())
    resp = client.post(
        "/apis/example/v2/workspaces/team-a/greet",
        json={"name": "Ada"},
        headers={"X-Request-ID": "req-1"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"message": "Hello, Ada!", "workspace": "team-a"}


def test_route_rejects_invalid_spec_with_422() -> None:
    client = TestClient(_build_app())
    resp = client.post(
        "/apis/example/v2/workspaces/team-a/greet",
        json={"name": 123},  # spec_schema requires str
    )
    assert resp.status_code == 422


def test_streaming_route_emits_ndjson_lines() -> None:
    client = TestClient(_build_app())
    with client.stream(
        "POST",
        "/apis/example/v2/workspaces/team-a/count",
        json={"upto": 2},
    ) as resp:
        assert resp.status_code == 200
        assert NDJSON_MEDIA_TYPE in resp.headers["content-type"]
        lines = [line for line in resp.iter_lines() if line]
    kinds = [json.loads(line)["kind"] for line in lines]
    # Heartbeat injection is disabled (interval=0) so the only frames
    # are the function's own ticks plus the terminator.
    assert kinds == ["tick", "tick", "done"]

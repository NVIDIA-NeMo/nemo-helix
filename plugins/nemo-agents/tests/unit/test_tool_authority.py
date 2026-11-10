# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import httpx
import pytest
from nemo_agents_plugin.fabric.tool_authority import BROKER_PATH, CONTEXT_KEY, ToolAuthorityBroker
from starlette.responses import JSONResponse


async def test_broker_capabilities_are_private_isolated_and_revoked():
    descriptors = []
    seen = []
    client: httpx.AsyncClient

    async def app(scope, receive, send):
        descriptor = scope["state"][CONTEXT_KEY]
        descriptors.append(descriptor)
        assert set(descriptor) == {"broker_port", "capability"}
        assert descriptor["broker_port"] == 8000
        await asyncio.sleep(0)
        response = await client.get(BROKER_PATH, headers={"authorization": f"Bearer {descriptor['capability']}"})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        token_headers = {"authorization": f"Bearer {descriptor['capability']}"}
        gateway_response = await client.get(
            BROKER_PATH,
            headers={**token_headers, "x-nhx-tool-workspace": "default"},
        )
        assert gateway_response.status_code == 401
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=broker, client=("192.0.2.1", 1234)),
            base_url="http://agent:8000",
        ) as remote_client:
            assert (await remote_client.get(BROKER_PATH, headers=token_headers)).status_code == 401
        seen.append(response.json())
        await JSONResponse(descriptor)(scope, receive, send)

    broker = ToolAuthorityBroker(app, enabled=lambda: True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=broker), base_url="http://agent:8000") as client:
        unauthorized = await client.get(BROKER_PATH)
        assert unauthorized.status_code == 401
        first, second = await asyncio.gather(
            *[
                client.post(
                    "/v1/chat/completions",
                    headers={
                        "authorization": f"Bearer {identity}",
                        "x-nhx-tool-workspace": workspace,
                        "x-nhx-studio-session": identity,
                        "x-nhx-studio-capability": f"callback-{identity}",
                    },
                )
                for identity, workspace in [("alice", "first"), ("bob", "second")]
            ]
        )
        assert first.json()["capability"] != second.json()["capability"]
        for descriptor in descriptors:
            revoked = await client.get(BROKER_PATH, headers={"authorization": f"Bearer {descriptor['capability']}"})
            assert revoked.status_code == 401
    assert {(value["workspace"], value["headers"]["authorization"]) for value in seen} == {
        ("first", "Bearer alice"),
        ("second", "Bearer bob"),
    }
    assert "alice" not in json.dumps(descriptors) and "bob" not in json.dumps(descriptors)


async def test_broker_revokes_on_failed_or_cancelled_invocation():
    descriptor = None

    async def app(scope, receive, send):
        nonlocal descriptor
        descriptor = scope["state"][CONTEXT_KEY]
        raise asyncio.CancelledError

    broker = ToolAuthorityBroker(app, enabled=lambda: True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=broker), base_url="http://agent:8000") as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post("/v1/chat/completions", headers={"authorization": "Bearer caller"})
        assert descriptor is not None
        revoked = await client.get(BROKER_PATH, headers={"authorization": f"Bearer {descriptor['capability']}"})
        assert revoked.status_code == 401
    assert broker._active == {}


async def test_other_agents_do_not_get_tool_authority():
    async def app(scope, receive, send):
        assert CONTEXT_KEY not in scope.get("state", {})
        await JSONResponse({})(scope, receive, send)

    broker = ToolAuthorityBroker(app, enabled=lambda: False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=broker), base_url="http://agent:8000") as client:
        assert (await client.post("/v1/chat/completions")).status_code == 200
    assert broker._active == {}

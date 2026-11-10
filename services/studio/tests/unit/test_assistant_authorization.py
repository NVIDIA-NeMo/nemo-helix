# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from nhx.studio import assistant
from nhx.studio.config import StudioConfig
from nhx.studio.dependencies import get_studio_config


@pytest.fixture
def active_session(monkeypatch):
    sid = str(uuid.uuid4())
    monkeypatch.setattr(assistant, "_session_authority", {sid: ("alice", "default", "private-capability")})
    monkeypatch.setattr(assistant, "_pending_permissions", {})
    monkeypatch.setattr(assistant, "_pending_agent_inputs", {})
    return sid


def principal(identity):
    return Request({"type": "http", "headers": [(b"x-nhx-principal-id", identity.encode())]})


@pytest.mark.parametrize("kind", ["permission", "input"])
async def test_only_session_owner_can_resolve_consent_or_input(active_session, kind):
    sid = active_session
    rid = str(uuid.uuid4())
    future = asyncio.get_running_loop().create_future()
    if kind == "permission":
        assistant._pending_permissions[rid] = (sid, future)
    else:
        assistant._pending_agent_inputs[rid] = (sid, future)

    async def resolve(request: Request):
        if kind == "permission":
            return await assistant.resolve_permission(sid, rid, assistant.PermissionDecision(approved=True), request)
        return await assistant.resolve_agent_input(
            sid,
            rid,
            assistant.AgentInputDecision(value={"model": "selected"}),
            request,
        )

    with pytest.raises(HTTPException) as error:
        await resolve(principal("bob"))
    assert error.value.status_code == 404
    assert not future.done()
    assert await resolve(principal("alice")) == {"ok": True}
    assert future.done()


@pytest.mark.parametrize(
    ("token", "workspace", "status"),
    [
        (None, "default", 401),
        ("wrong", "default", 401),
        (b"private-\xff", "default", 401),
        ("private-capability", "other", 403),
        ("private-capability", None, 403),
        ("private-capability", "default", 200),
    ],
)
def test_public_callback_requires_active_workspace_bound_capability(active_session, token, workspace, status):
    app = FastAPI()
    app.dependency_overrides[get_studio_config] = lambda: StudioConfig()
    assistant.mount_public_mcp_route(app)
    client = TestClient(app)
    headers = {"x-nhx-studio-capability": token} if token else {}
    # A claimed principal or known session UUID does not authorize public callbacks.
    headers["x-nhx-principal-id"] = "alice"
    response = client.post(
        f"/studio/api/assistant/mcp/{active_session}",
        params={"workspace": workspace} if workspace is not None else {},
        headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert response.status_code == status
    if status == 200:
        assert "private-capability" not in response.text


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.test:8080/agent",
        "http://host.docker.internal:8080/agent",
        str(httpx.URL("https://remote.test/agent").copy_with(username="test-user", password="test-password")),
        "ftp://127.0.0.1/agent",
        "not-a-url",
    ],
)
async def test_studio_rejects_unsafe_agent_transport_before_minting_capability(active_session, monkeypatch, url):
    sid = str(uuid.uuid4())
    monkeypatch.setattr(assistant, "_session_streams", {})
    unreachable: Any = None  # Transport rejection must precede accessing session/store objects.
    stream = assistant._stream_assistant(sid, "hello", url, {}, "prompt", unreachable, unreachable)
    with pytest.raises(HTTPException) as error:
        await anext(stream)
    assert error.value.status_code == 502
    assert sid not in assistant._session_authority
    assert sid not in assistant._session_streams


@pytest.mark.parametrize(
    "url", ["https://remote.test/agent", "http://127.0.0.1:8080/agent", "http://localhost:8080/agent"]
)
def test_studio_accepts_https_and_loopback_capability_transport(url):
    assistant._require_private_assistant_transport(url)
    request = Request(
        {
            "type": "http",
            "headers": [
                (b"authorization", b"Bearer caller"),
                (b"cookie", b"caller=session"),
            ],
        }
    )
    assert assistant._assistant_request_headers(request, url) == {
        "authorization": "Bearer caller",
        "cookie": "caller=session",
    }


async def test_overlapping_turn_is_rejected_without_replacing_active_authority(active_session, monkeypatch):
    from nhx.studio.entities import AssistantConversation

    sid = str(uuid.uuid4())
    monkeypatch.setattr(assistant, "_session_streams", {})
    monkeypatch.setattr(assistant, "MCP_KEEPALIVE_INTERVAL_SECONDS", 0.001)
    observed_headers = []

    async def invoke(url, headers, messages, studio_session_id):
        observed_headers.append(headers)
        # Private capability must never enter the model context.
        assert headers["x-nhx-studio-capability"] not in str(messages)
        await asyncio.Event().wait()

    monkeypatch.setattr(assistant, "_invoke_assistant", invoke)
    conversation = AssistantConversation(name="test", workspace="default", owner_id="alice", session_id=sid)
    unreachable: Any = None
    first = assistant._stream_assistant(sid, "hello", "https://agent.test", {}, "prompt", conversation, unreachable)
    assert await anext(first) == ":\n\n"
    binding = assistant._session_authority[sid]
    second = assistant._stream_assistant(sid, "hello", "https://agent.test", {}, "prompt", conversation, unreachable)
    duplicate_frames = [frame async for frame in second]
    assert "already has an active stream" in "".join(duplicate_frames)
    assert assistant._session_authority[sid] == binding
    assert len(observed_headers) == 1
    await first.aclose()
    await asyncio.sleep(0)
    assert sid not in assistant._session_authority
    assert sid not in assistant._session_streams


def test_callback_capability_cannot_select_another_session_and_is_revocable(active_session):
    app = FastAPI()
    app.dependency_overrides[get_studio_config] = lambda: StudioConfig()
    assistant.mount_public_mcp_route(app)
    client = TestClient(app)
    headers = {"x-nhx-studio-capability": "private-capability"}
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert (
        client.post(
            f"/studio/api/assistant/mcp/{uuid.uuid4()}?workspace=default",
            headers=headers,
            json=body,
        ).status_code
        == 401
    )
    assistant._session_authority.clear()
    assert (
        client.post(
            f"/studio/api/assistant/mcp/{active_session}?workspace=default",
            headers=headers,
            json=body,
        ).status_code
        == 401
    )

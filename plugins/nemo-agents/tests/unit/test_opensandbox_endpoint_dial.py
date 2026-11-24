# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agents-plugin callers dialing an OpenSandbox deployment through OpenSandboxEndpointTransport.

Covers the gateway proxy, the controller's runtime health read, and session cleanup, against a
gateway that requires the sandbox's secure-access token.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import gateway as gateway_module
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.entities import AgentDeployment, AgentSession, Endpoint, SessionStatus
from nemo_agents_plugin.runner.controller import AgentDeploymentController
from nemo_agents_plugin.session_lifecycle import cleanup_fabric_runtime
from nemo_deployments_plugin.backends.opensandbox import endpoint as endpoint_module
from nemo_deployments_plugin.backends.opensandbox.endpoint import (
    OpenSandboxEndpointTransport,
    sandbox_endpoint_url,
)
from nemo_helix_plugin.dependencies import get_effective_principal_id

SID = "7920fbcd-4e18-4753-bb15-41957b6f12e0"
OTHER = "11111111-2222-3333-4444-555555555555"
TOKEN = "per-sandbox-token"
TOKEN_HEADER = "OpenSandbox-Secure-Access"
INVOKE = "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/"


class _FakeOpenSandbox:
    def __init__(self) -> None:
        self.lookups = 0
        self.lookups_fail = False
        self.gateway_requests: list[tuple[str, str, list[str]]] = []


class _Handler(BaseHTTPRequestHandler):
    state: _FakeOpenSandbox

    def _send(self, status: int, body: bytes = b"{}", content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        if self.path.startswith("/v1/sandboxes/"):
            self.state.lookups += 1
            if self.state.lookups_fail:
                return self._send(503)
            return self._send(200, json.dumps({"endpoint": "unused", "headers": {TOKEN_HEADER: TOKEN}}).encode())
        tokens = self.headers.get_all(TOKEN_HEADER) or []
        self.state.gateway_requests.append((self.command, self.path, tokens))
        if tokens != [TOKEN]:
            return self._send(401)
        if self.path.endswith("/health"):
            health = {"runtime_instance_id": "instance-1", "runtime_started_at": "2026-11-18T00:00:00+00:00"}
            return self._send(200, json.dumps(health).encode())
        if body and json.loads(body).get("stream"):
            return self._send(200, b'data: {"n": 1}\n\ndata: [DONE]\n\n', "text/event-stream")
        return self._send(200, b'{"answer": 42}')

    do_GET = do_POST = do_DELETE = _handle

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def osb(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[_FakeOpenSandbox, OpenSandboxEndpointTransport]]:
    state = _FakeOpenSandbox()
    server = ThreadingHTTPServer(("127.0.0.1", 0), type("Handler", (_Handler,), {"state": state}))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "osb-key")
    endpoint_module._endpoint_headers.clear()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state, OpenSandboxEndpointTransport(connect_base=base, server_base=base)
    finally:
        server.shutdown()
        server.server_close()
        endpoint_module._endpoint_headers.clear()


def _deployment() -> AgentDeployment:
    deployment = AgentDeployment(
        name="calc-dep",
        workspace="default",
        agent="calc",
        status="running",
        endpoint="",
        deployment_mode="k8s",
        endpoints=[Endpoint(name="http", url=sandbox_endpoint_url(SID, 8000), protocol="http")],
    )
    deployment._id = "deployment-id"
    return deployment


@pytest.fixture
def client() -> TestClient:
    entities = AsyncMock()
    entities.update.side_effect = lambda entity: entity
    entities.get = AsyncMock(return_value=_deployment())
    app = FastAPI()
    app.include_router(gateway_module.router, prefix="/apis/agents/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entities
    app.dependency_overrides[get_effective_principal_id] = lambda: "session-owner"
    return TestClient(app, raise_server_exceptions=False)


def test_invoke_reaches_the_agent_with_the_sandbox_token(
    client: TestClient, osb: tuple[_FakeOpenSandbox, OpenSandboxEndpointTransport]
) -> None:
    state, transport = osb
    with patch("nemo_agents_plugin.api.v2.gateway.get_deployment_transport", return_value=transport):
        first = client.post(INVOKE + "v1/chat/completions", json={"messages": []}, headers={TOKEN_HEADER: "mine"})
        second = client.post(INVOKE + "v1/chat/completions", json={"messages": []})

    assert (first.status_code, first.json()) == (200, {"answer": 42})
    assert second.status_code == 200
    assert state.gateway_requests == [("POST", f"/{SID}/8000/v1/chat/completions", [TOKEN])] * 2
    assert state.lookups == 1


def test_invoke_preserves_sse_response(
    client: TestClient, osb: tuple[_FakeOpenSandbox, OpenSandboxEndpointTransport]
) -> None:
    _, transport = osb
    with patch("nemo_agents_plugin.api.v2.gateway.get_deployment_transport", return_value=transport):
        response = client.post(INVOKE + "v1/chat/completions", json={"messages": [], "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == 'data: {"n": 1}\n\ndata: [DONE]\n\n'


@pytest.mark.parametrize(
    "trailing",
    ["%2e%2e/44772/command", "..%2f44772/command", f"%2e%2e/%2e%2e/{OTHER}/8000/command", "/44772/command"],
)
def test_a_callers_path_cannot_reach_another_port_or_sandbox(
    client: TestClient, osb: tuple[_FakeOpenSandbox, OpenSandboxEndpointTransport], trailing: str
) -> None:
    state, transport = osb
    with patch("nemo_agents_plugin.api.v2.gateway.get_deployment_transport", return_value=transport):
        client.post(INVOKE + trailing, json={"command": "id"})

    (request,) = state.gateway_requests
    assert request[1].startswith(f"/{SID}/8000/")
    assert request[1].endswith("/command")


def test_a_failed_token_lookup_is_a_502_and_the_agent_is_not_dialed(
    client: TestClient, osb: tuple[_FakeOpenSandbox, OpenSandboxEndpointTransport]
) -> None:
    state, transport = osb
    state.lookups_fail = True
    with patch("nemo_agents_plugin.api.v2.gateway.get_deployment_transport", return_value=transport):
        response = client.post(INVOKE + "v1/chat/completions", json={"messages": []})

    assert response.status_code == 502
    assert "127.0.0.1" not in response.text
    assert state.gateway_requests == []


async def test_controller_reads_runtime_health(osb: tuple[_FakeOpenSandbox, OpenSandboxEndpointTransport]) -> None:
    state, transport = osb
    controller = AgentDeploymentController()
    try:
        with patch("nemo_agents_plugin.deployment_routing.get_deployment_transport", return_value=transport):
            first = await controller._read_runtime_instance(_deployment())
            second = await controller._read_runtime_instance(_deployment())
    finally:
        for pooled in controller._runtime_health_clients.values():
            await pooled.aclose()

    assert first is not None and first == second
    assert first[0] == "instance-1"
    assert state.gateway_requests == [("GET", f"/{SID}/8000/health", [TOKEN])] * 2
    assert len(controller._runtime_health_clients) == 1


async def test_session_cleanup_reaches_the_runtime(osb: tuple[_FakeOpenSandbox, OpenSandboxEndpointTransport]) -> None:
    state, transport = osb
    deployment = _deployment()
    session = AgentSession(name="s", workspace="default", deployment_id=deployment.id, status=SessionStatus.CLOSED)
    entities = AsyncMock()
    entities.find_one.return_value = deployment
    with patch("nemo_agents_plugin.session_lifecycle.get_deployment_transport", return_value=transport):
        await cleanup_fabric_runtime(entities, session)

    assert [(method, tokens) for method, _, tokens in state.gateway_requests] == [("DELETE", [TOKEN])]
    assert state.gateway_requests[0][1].startswith(f"/{SID}/8000/v1/sessions/")

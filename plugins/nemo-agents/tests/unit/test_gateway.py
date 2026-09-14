# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent gateway proxy routes.

Covers:
- 5xx from the upstream agent → 502 Bad Gateway
- 2xx responses streamed through with correct status and content-type
- Empty-body responses (e.g. 204) handled without error
- Agent not found → 404
- Deployment not running → 503
- httpx connection error → 502
- Proxy by agent name resolves the active deployment endpoint
- Proxy by deployment name targets the deployment directly

Mocking strategy: patch ``httpx.AsyncClient`` so tests run with no real network.
The mock replicates the async-context-manager chain::

    async with httpx.AsyncClient(...) as client:
        async with client.stream(...) as response:
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import gateway as gateway_module
from nemo_agents_plugin.api.v2 import openai_errors
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.entities import (
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    Agent,
    AgentDeployment,
    AgentSession,
    DeploymentMode,
    DeploymentStatus,
    Endpoint,
    SessionStatus,
)
from nemo_agents_plugin.session_protocol import SESSION_ID_HEADER
from nemo_platform_plugin.dependencies import get_effective_principal_id
from nemo_platform_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError

OWNER_PRINCIPAL_ID = "session-owner"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str = "calc", workspace: str = "default") -> Agent:
    return Agent(name=name, workspace=workspace)


def _make_deployment(
    name: str = "calc-dep",
    agent: str = "calc",
    workspace: str = "default",
    status: DeploymentStatus = "running",
    endpoint: str = "http://localhost:9001",
    deployment_id: str | None = None,
    config: dict | None = None,
) -> AgentDeployment:
    deployment = AgentDeployment(
        name=name, workspace=workspace, agent=agent, status=status, endpoint=endpoint, config=config or {}
    )
    if deployment_id is not None:
        deployment._id = deployment_id
    return deployment


def _make_session(
    *,
    session_id: str = "session-id",
    name: str = "session-one",
    workspace: str = "default",
    deployment_id: str = "deployment-id",
    status: SessionStatus = SessionStatus.ACTIVE,
    created_by: str | None = OWNER_PRINCIPAL_ID,
) -> AgentSession:
    session = AgentSession(
        name=name,
        workspace=workspace,
        deployment_id=deployment_id,
        status=status,
    )
    session._id = session_id
    session._created_by = created_by
    return session


def _make_container_deployment(
    name: str = "calc-dep",
    agent: str = "calc",
    workspace: str = "default",
    mode: DeploymentMode = "k8s",
    status: DeploymentStatus = "running",
    endpoints: list[Endpoint] | None = None,
) -> AgentDeployment:
    """Build a container-mode (docker/k8s) AgentDeployment as the controller would project it.

    Container deployments carry no loopback ``endpoint``; the routable address lives
    on ``endpoints`` (projected from the deployments-plugin Deployment). The controller
    projects the deployments-plugin READY status onto the agents-local ``running``.
    """
    if endpoints is None:
        endpoints = [Endpoint(name="http", url="http://calc-dep.default.svc.cluster.local:8080", protocol="http")]
    return AgentDeployment(
        name=name,
        workspace=workspace,
        agent=agent,
        status=status,
        endpoint="",
        deployment_mode=mode,
        endpoints=endpoints,
    )


def _list_response(items: list) -> MagicMock:
    resp = MagicMock()
    resp.data = items
    return resp


def _make_httpx_mock(
    status_code: int,
    body: bytes = b"",
    content_type: str = "application/json",
    response_headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build the full async-context-manager chain for httpx.AsyncClient().stream()."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    headers = {"content-type": content_type, **(response_headers or {})}
    mock_response.headers = httpx.Headers(headers)

    async def _aiter_bytes():
        if body:
            yield body

    async def _aread():
        return body

    mock_response.aiter_bytes = _aiter_bytes
    mock_response.aread = _aread

    # client.stream(...) → async context manager yielding mock_response
    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=stream_cm)

    # httpx.AsyncClient(...) → async context manager yielding mock_client
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=mock_client)
    client_cm.__aexit__ = AsyncMock(return_value=False)

    return client_cm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    client = AsyncMock()
    client.update.side_effect = lambda entity: entity
    return client


@pytest.fixture
def test_app(mock_entity_client: AsyncMock) -> FastAPI:
    app = FastAPI()
    app.include_router(
        gateway_module.router,
        prefix="/apis/agents/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    app.dependency_overrides[get_effective_principal_id] = lambda: OWNER_PRINCIPAL_ID
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Proxy by deployment name — core proxy behaviour
# ---------------------------------------------------------------------------


class TestProxyByDeploymentName:
    def test_2xx_passed_through(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        upstream_body = b'{"answer": 42}'
        httpx_mock = _make_httpx_mock(200, upstream_body, "application/json")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        assert resp.content == upstream_body

    def test_5xx_from_agent_becomes_502(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Agent 5xx responses must be translated to 502 Bad Gateway."""
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        error_body = b"Internal server error in agent"
        httpx_mock = _make_httpx_mock(500, error_body)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={"messages": []},
            )

        assert resp.status_code == 502
        assert "502" in resp.text or "Agent returned 500" in resp.text

    def test_503_from_agent_becomes_502(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Any 5xx (not just 500) is translated to 502."""
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        httpx_mock = _make_httpx_mock(503, b"Service Unavailable")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 502

    def test_4xx_from_agent_passed_through(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """4xx client errors from the agent are transparent pass-through."""
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        httpx_mock = _make_httpx_mock(422, b'{"detail": "invalid input"}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 422

    def test_empty_body_response_handled(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Empty body (e.g. 204 No Content) must not raise StopAsyncIteration."""
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        httpx_mock = _make_httpx_mock(204, b"")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 204

    def test_content_type_forwarded(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        httpx_mock = _make_httpx_mock(200, b"data: hello\n\n", "text/event-stream")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.get(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/stream",
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_session_id_response_header_forwarded(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)
        httpx_mock = _make_httpx_mock(
            200,
            b'{"ok": true}',
            response_headers={SESSION_ID_HEADER: "runtime-session-1"},
        )

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 200
        assert resp.headers[SESSION_ID_HEADER] == "runtime-session-1"

    def test_connection_error_returns_502(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        # Simulate httpx.ConnectError during stream open
        client_cm = MagicMock()
        client_cm.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=client_cm):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 502
        assert "Could not connect" in resp.json()["detail"]

    def test_deployment_not_found_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("not found"))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/nonexistent/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 404

    def test_deployment_not_running_returns_503(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(status="starting", endpoint="")
        mock_entity_client.get = AsyncMock(return_value=dep)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 503
        assert "not routable" in resp.json()["detail"].lower()

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "%2F%2Fevil.example.com/x",
            "http:%2F%2Fevil.example.com/x",
        ],
    )
    def test_cross_origin_trailing_uri_rejected(
        self, client: TestClient, mock_entity_client: AsyncMock, malicious_path: str
    ) -> None:
        """SSRF guard rejects trailing_uri values that resolve to a different host."""
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        resp = client.post(
            f"/apis/agents/v2/workspaces/default/deployments/calc-dep/-/{malicious_path}",
            json={},
        )

        assert resp.status_code == 400
        assert "invalid proxy target" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Proxy by agent name — endpoint resolution
# ---------------------------------------------------------------------------


class TestProxyByAgentName:
    def test_resolves_running_deployment(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        failed = _make_deployment(name="failed-dep", agent="calc", status="failed")
        selected = _make_deployment(
            name="running-dep",
            agent="calc",
            status="running",
            endpoint="http://localhost:9001",
        )
        later = _make_deployment(
            name="later-dep",
            agent="calc",
            status="running",
            endpoint="http://localhost:9002",
        )
        mock_entity_client.list = AsyncMock(return_value=_list_response([failed, selected, later]))

        httpx_mock = _make_httpx_mock(200, b'{"ok": true}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                json={"messages": []},
            )

        assert resp.status_code == 200
        mock_entity_client.find_one.assert_not_awaited()
        stream_call = httpx_mock.__aenter__.return_value.stream.call_args
        assert stream_call.kwargs["url"] == "http://localhost:9001/v1/chat/completions"

    def test_agent_not_found_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("not found"))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/nonexistent/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 404

    def test_no_running_deployment_returns_503(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        # Only a failed deployment — no running ones
        dep = _make_deployment(agent="calc", status="failed")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 503

    def test_5xx_from_agent_becomes_502_via_name(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """5xx translation works end-to-end through the agent-name proxy path too."""
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        dep = _make_deployment(agent="calc", status="running", endpoint="http://localhost:9001")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        httpx_mock = _make_httpx_mock(500, b"agent crashed")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Persisted session resolution and deployment binding
# ---------------------------------------------------------------------------


class TestSessionAwareRouting:
    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("application/json", b'{"ok": true}'),
            ("text/event-stream", b"data: done\n\n"),
        ],
    )
    def test_persists_session_activity_at_proxy_start_and_finish(
        self,
        content_type: str,
        body: bytes,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        deployment = _make_deployment(deployment_id="deployment-1")
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])
        updates: list[AgentSession] = []

        async def update(entity: AgentSession) -> AgentSession:
            updates.append(entity.model_copy(deep=True))
            return entity

        mock_entity_client.update.side_effect = update
        timestamps = [
            datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 25, 12, 5, tzinfo=UTC),
        ]
        httpx_mock = _make_httpx_mock(200, body, content_type)

        with (
            patch.object(gateway_module, "_utc_now", side_effect=timestamps),
            patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock),
        ):
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={"stream": content_type == "text/event-stream"},
            )

        assert response.status_code == 200
        assert [update.first_active_at for update in updates] == [timestamps[0], timestamps[0]]
        assert [update.last_active_at for update in updates] == timestamps
        assert [update.expires_at for update in updates] == [
            timestamp + timedelta(seconds=30 * 60) for timestamp in timestamps
        ]

    def test_agent_route_uses_session_bound_deployment(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        session = _make_session(session_id="session-2", deployment_id="deployment-2")
        bound_deployment = _make_deployment(
            name="calc-v2",
            agent="calc",
            endpoint="http://localhost:9002",
            deployment_id="deployment-2",
        )
        mock_entity_client.find_one = AsyncMock(side_effect=[session, bound_deployment])
        httpx_mock = _make_httpx_mock(200, b'{"ok": true}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-2"},
                json={"messages": []},
            )

        assert resp.status_code == 200
        assert mock_entity_client.find_one.await_args_list == [
            call(
                AgentSession,
                workspace="default",
                filter_obj={"id": "session-2", "created_by": OWNER_PRINCIPAL_ID},
            ),
            call(AgentDeployment, workspace="default", filter_obj={"id": "deployment-2"}),
        ]
        mock_entity_client.get.assert_not_awaited()
        mock_entity_client.list.assert_not_awaited()
        stream_call = httpx_mock.__aenter__.return_value.stream.call_args
        assert stream_call.kwargs["url"] == "http://localhost:9002/v1/chat/completions"
        assert stream_call.kwargs["headers"][SESSION_ID_HEADER] == "session-2"

    def test_noauth_route_resolves_service_owned_session_without_owner_filter(
        self, client: TestClient, test_app: FastAPI, mock_entity_client: AsyncMock
    ) -> None:
        test_app.dependency_overrides[get_effective_principal_id] = lambda: ""
        session = _make_session(
            session_id="session-2",
            deployment_id="deployment-2",
            created_by="service:platform",
        )
        bound_deployment = _make_deployment(
            name="calc-v2",
            agent="calc",
            endpoint="http://localhost:9002",
            deployment_id="deployment-2",
        )
        mock_entity_client.find_one = AsyncMock(side_effect=[session, bound_deployment])
        httpx_mock = _make_httpx_mock(200, b'{"ok": true}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-2"},
                json={"messages": []},
            )

        assert response.status_code == 200
        assert mock_entity_client.find_one.await_args_list[0] == call(
            AgentSession,
            workspace="default",
            filter_obj={"id": "session-2"},
        )

    def test_retries_start_activity_once_after_optimistic_conflict(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        deployment = _make_deployment(deployment_id="deployment-1")
        refreshed = _make_session(session_id="session-1", deployment_id="deployment-1")
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])
        mock_entity_client.get_by_id = AsyncMock(return_value=refreshed)
        update_count = 0

        async def update(entity: AgentSession) -> AgentSession:
            nonlocal update_count
            update_count += 1
            if update_count == 1:
                raise NemoEntityConflictError("conflict")
            return entity

        mock_entity_client.update.side_effect = update
        httpx_mock = _make_httpx_mock(200, b'{"ok": true}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={},
            )

        assert response.status_code == 200
        assert update_count == 3
        mock_entity_client.get_by_id.assert_awaited_once_with(AgentSession, "session-1")

    def test_due_session_wins_start_activity_conflict(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        activity_at = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        session.expires_at = activity_at + timedelta(minutes=1)
        deployment = _make_deployment(deployment_id="deployment-1")
        refreshed = _make_session(session_id="session-1", deployment_id="deployment-1")
        refreshed.expires_at = activity_at
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])
        mock_entity_client.update.side_effect = NemoEntityConflictError("conflict")
        mock_entity_client.get_by_id = AsyncMock(return_value=refreshed)

        with (
            patch.object(
                gateway_module,
                "_utc_now",
                side_effect=[activity_at - timedelta(seconds=1), activity_at],
            ),
            patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient") as httpx_client,
        ):
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={},
            )

        assert response.status_code == 409
        assert "has expired" in response.json()["error"]["message"]
        mock_entity_client.update.assert_awaited_once()
        mock_entity_client.get_by_id.assert_awaited_once_with(AgentSession, "session-1")
        httpx_client.assert_not_called()

    def test_terminal_state_wins_start_activity_conflict(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        deployment = _make_deployment(deployment_id="deployment-1")
        closed = _make_session(
            session_id="session-1",
            deployment_id="deployment-1",
            status=SessionStatus.CLOSED,
        )
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])
        mock_entity_client.update.side_effect = NemoEntityConflictError("conflict")
        mock_entity_client.get_by_id = AsyncMock(return_value=closed)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient") as httpx_client:
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={},
            )

        assert response.status_code == 409
        assert "closed" in response.json()["error"]["message"]
        httpx_client.assert_not_called()

    def test_start_activity_failure_prevents_invocation(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        deployment = _make_deployment(deployment_id="deployment-1")
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])
        mock_entity_client.update.side_effect = RuntimeError("entity store unavailable")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient") as httpx_client:
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={},
            )

        assert response.status_code == 503
        assert "persist activity" in response.json()["error"]["message"]
        httpx_client.assert_not_called()

    def test_finish_activity_failure_does_not_replace_successful_response(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        deployment = _make_deployment(deployment_id="deployment-1")
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])
        update_count = 0

        async def update(entity: AgentSession) -> AgentSession:
            nonlocal update_count
            update_count += 1
            if update_count == 2:
                raise RuntimeError("entity store unavailable")
            return entity

        mock_entity_client.update.side_effect = update
        httpx_mock = _make_httpx_mock(200, b'{"ok": true}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={},
            )

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert update_count == 2

    def test_proxy_failure_still_records_finish_activity(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        deployment = _make_deployment(deployment_id="deployment-1")
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client_context.__aexit__ = AsyncMock(return_value=False)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=client_context):
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={},
            )

        assert response.status_code == 502
        assert mock_entity_client.update.await_count == 2

    @pytest.mark.asyncio
    async def test_delayed_activity_does_not_move_timestamps_backward(
        self,
        mock_entity_client: AsyncMock,
    ) -> None:
        last_active_at = datetime(2026, 8, 25, 12, 5, tzinfo=UTC)
        expires_at = last_active_at + timedelta(minutes=30)
        session = _make_session(session_id="session-1")
        session.last_active_at = last_active_at
        session.expires_at = expires_at

        updated = await gateway_module._persist_session_activity(
            mock_entity_client,
            session,
            activity_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        )

        assert updated is session
        assert updated.last_active_at == last_active_at
        assert updated.expires_at == expires_at
        mock_entity_client.update.assert_not_awaited()

    def test_direct_route_accepts_session_for_same_deployment(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        deployment = _make_deployment(deployment_id="deployment-1")
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        mock_entity_client.get = AsyncMock(return_value=deployment)
        mock_entity_client.find_one = AsyncMock(return_value=session)
        httpx_mock = _make_httpx_mock(200, b'{"ok": true}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={"messages": []},
            )

        assert resp.status_code == 200
        mock_entity_client.find_one.assert_awaited_once_with(
            AgentSession,
            workspace="default",
            filter_obj={"id": "session-1", "created_by": OWNER_PRINCIPAL_ID},
        )
        stream_call = httpx_mock.__aenter__.return_value.stream.call_args
        assert stream_call.kwargs["headers"][SESSION_ID_HEADER] == "session-1"

    def test_unknown_session_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.find_one = AsyncMock(side_effect=NemoEntityNotFoundError("not found"))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            headers={SESSION_ID_HEADER: "missing"},
            json={},
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Session ID 'missing' not found in workspace 'default'."
        mock_entity_client.get.assert_not_awaited()
        mock_entity_client.list.assert_not_awaited()

    def test_cross_workspace_session_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.find_one = AsyncMock(return_value=_make_session(session_id="session-1", workspace="other"))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            headers={SESSION_ID_HEADER: "session-1"},
            json={},
        )

        assert resp.status_code == 404

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/apis/agents/v2/workspaces/default/agents/calc/-/health"),
            ("POST", "/apis/agents/v2/workspaces/default/agents/calc/-/invoke"),
            ("GET", "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/health"),
            ("POST", "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/invoke"),
        ],
    )
    def test_foreign_owned_session_returns_404_on_every_gateway_route(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
        method: str,
        path: str,
    ) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(deployment_id="deployment-id"))
        mock_entity_client.find_one = AsyncMock(
            return_value=_make_session(session_id="session-1", created_by="other-principal")
        )

        resp = client.request(
            method,
            path,
            headers={SESSION_ID_HEADER: "session-1"},
            json={} if method == "POST" else None,
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Session ID 'session-1' not found in workspace 'default'."
        mock_entity_client.find_one.assert_awaited_once_with(
            AgentSession,
            workspace="default",
            filter_obj={"id": "session-1", "created_by": OWNER_PRINCIPAL_ID},
        )

    @pytest.mark.parametrize(
        "status",
        [SessionStatus.EXPIRED, SessionStatus.LOST, SessionStatus.CLOSED],
    )
    def test_non_active_session_returns_409(
        self,
        status: SessionStatus,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        mock_entity_client.find_one = AsyncMock(return_value=_make_session(session_id="session-1", status=status))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            headers={SESSION_ID_HEADER: "session-1"},
            json={},
        )

        assert resp.status_code == 409
        assert resp.json()["detail"] == (f"Session ID 'session-1' has status '{status.value}' and cannot be invoked.")

    def test_active_session_at_expiration_deadline_returns_409(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        deadline = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
        session = _make_session(session_id="session-1")
        session.expires_at = deadline
        mock_entity_client.find_one = AsyncMock(return_value=session)

        with (
            patch.object(gateway_module, "_utc_now", return_value=deadline),
            patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient") as httpx_client,
        ):
            response = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                headers={SESSION_ID_HEADER: "session-1"},
                json={},
            )

        assert response.status_code == 409
        assert "has expired" in response.json()["error"]["message"]
        mock_entity_client.update.assert_not_awaited()
        httpx_client.assert_not_called()

    def test_read_only_proxy_does_not_refresh_session_activity(
        self,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        deployment = _make_deployment(deployment_id="deployment-1")
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])
        httpx_mock = _make_httpx_mock(200, b'{"status": "ok"}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            response = client.get(
                "/apis/agents/v2/workspaces/default/agents/calc/-/health",
                headers={SESSION_ID_HEADER: "session-1"},
            )

        assert response.status_code == 200
        mock_entity_client.update.assert_not_awaited()

    def test_empty_session_header_returns_400(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            headers={SESSION_ID_HEADER: ""},
            json={},
        )

        assert resp.status_code == 400
        mock_entity_client.find_one.assert_not_awaited()

    def test_missing_bound_deployment_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        session = _make_session(session_id="session-1", deployment_id="missing-deployment")
        mock_entity_client.find_one = AsyncMock(side_effect=[session, NemoEntityNotFoundError("not found")])

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            headers={SESSION_ID_HEADER: "session-1"},
            json={},
        )

        assert resp.status_code == 404
        assert "missing-deployment" in resp.json()["detail"]

    def test_agent_route_rejects_session_bound_to_another_agent(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        session = _make_session(session_id="session-1", deployment_id="deployment-1")
        deployment = _make_deployment(
            agent="other-agent",
            deployment_id="deployment-1",
        )
        mock_entity_client.find_one = AsyncMock(side_effect=[session, deployment])

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            headers={SESSION_ID_HEADER: "session-1"},
            json={},
        )

        assert resp.status_code == 409
        assert "other-agent" in resp.json()["detail"]

    def test_direct_route_rejects_session_for_another_deployment(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        deployment = _make_deployment(deployment_id="deployment-1")
        session = _make_session(session_id="session-2", deployment_id="deployment-2")
        mock_entity_client.get = AsyncMock(return_value=deployment)
        mock_entity_client.find_one = AsyncMock(return_value=session)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
            headers={SESSION_ID_HEADER: "session-2"},
            json={},
        )

        assert resp.status_code == 409
        assert "deployment-2" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Model name patching — unknown-model → agent/deployment name
# ---------------------------------------------------------------------------


class TestModelNamePatching:
    def test_unknown_model_replaced_by_agent_name(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """JSON responses with "unknown-model" get patched to the agent name."""
        mock_entity_client.get = AsyncMock(return_value=_make_agent("my-agent"))
        dep = _make_deployment(agent="my-agent", status="running", endpoint="http://localhost:9001")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        body = json.dumps({"model": "unknown-model", "choices": [{"message": {"content": "hi"}}]}).encode()
        httpx_mock = _make_httpx_mock(200, body, "application/json")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/agents/my-agent/-/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "my-agent"
        assert data["choices"] == [{"message": {"content": "hi"}}]

    def test_malformed_json_passed_through(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Non-JSON response bodies are passed through unmodified."""
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        dep = _make_deployment(agent="calc", status="running", endpoint="http://localhost:9001")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        garbled = b"this is not json"
        httpx_mock = _make_httpx_mock(200, garbled, "application/json")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                json={"messages": []},
            )

        assert resp.status_code == 200
        assert resp.content == garbled

    def test_real_model_not_replaced(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """JSON responses with a real model name are left untouched."""
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        dep = _make_deployment(agent="calc", status="running", endpoint="http://localhost:9001")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        body = json.dumps({"model": "gpt-4o", "choices": []}).encode()
        httpx_mock = _make_httpx_mock(200, body, "application/json")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                json={"messages": []},
            )

        assert resp.status_code == 200
        assert resp.json()["model"] == "gpt-4o"

    def test_sse_stream_not_patched(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """SSE (event-stream) responses are passed through without model patching."""
        dep = _make_deployment(status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        sse_body = b'data: {"model":"unknown-model","choices":[]}\n\n'
        httpx_mock = _make_httpx_mock(200, sse_body, "text/event-stream")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={"messages": [], "stream": True},
            )

        assert resp.status_code == 200
        assert b"unknown-model" in resp.content

    def test_deployment_name_used_for_deployment_proxy(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Deployment-name proxy path patches model to the deployment name."""
        dep = _make_deployment(name="calc-v2", status="running", endpoint="http://localhost:9001")
        mock_entity_client.get = AsyncMock(return_value=dep)

        body = json.dumps({"model": "unknown-model", "choices": []}).encode()
        httpx_mock = _make_httpx_mock(200, body, "application/json")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-v2/-/v1/chat/completions",
                json={"messages": []},
            )

        assert resp.status_code == 200
        assert resp.json()["model"] == "calc-v2"


# ---------------------------------------------------------------------------
# Request model stripping — NAT deployments must not see the inbound ``model``
# ---------------------------------------------------------------------------


def _forwarded_request(httpx_mock: MagicMock) -> dict:
    """Return the kwargs the gateway passed to ``httpx.AsyncClient().stream(...)``."""
    return httpx_mock.__aenter__.return_value.stream.call_args.kwargs


class TestRequestModelStripping:
    """The gateway routes by URL path; ``nat serve`` would otherwise forward ``model`` to IGW (AALGO-644)."""

    _NAT_CONFIG = {"workflow": {"_type": "react_agent"}, "llms": {"llm": {"model_name": "meta/llama"}}}
    _FABRIC_CONFIG = {"config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT, "harness": {"model": "meta/llama"}}

    def test_nat_deployment_drops_model_from_chat_completions(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        dep = _make_deployment(config=self._NAT_CONFIG)
        mock_entity_client.get = AsyncMock(return_value=dep)
        httpx_mock = _make_httpx_mock(200, b'{"choices": []}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={"model": "calc-dep", "messages": [{"role": "user", "content": "2+2"}], "stream": False},
            )

        assert resp.status_code == 200
        forwarded = _forwarded_request(httpx_mock)
        assert json.loads(forwarded["content"]) == {"messages": [{"role": "user", "content": "2+2"}], "stream": False}
        assert "content-length" not in {k.lower() for k in forwarded["headers"]}

    def test_agent_route_also_drops_model(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        dep = _make_deployment(agent="calc", config=self._NAT_CONFIG)
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))
        httpx_mock = _make_httpx_mock(200, b'{"choices": []}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                json={"model": "default/calc", "messages": []},
            )

        assert resp.status_code == 200
        assert json.loads(_forwarded_request(httpx_mock)["content"]) == {"messages": []}

    def test_fabric_deployment_body_is_untouched(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(config=self._FABRIC_CONFIG)
        mock_entity_client.get = AsyncMock(return_value=dep)
        httpx_mock = _make_httpx_mock(200, b'{"choices": []}')
        payload = {"model": "calc-dep", "messages": []}

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json=payload,
            )

        forwarded = _forwarded_request(httpx_mock)
        assert json.loads(forwarded["content"]) == payload
        assert "content-length" in {k.lower() for k in forwarded["headers"]}

    def test_non_openai_surface_is_untouched(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        dep = _make_deployment(config=self._NAT_CONFIG)
        mock_entity_client.get = AsyncMock(return_value=dep)
        httpx_mock = _make_httpx_mock(200, b"{}")
        payload = {"model": "calc-dep", "input_message": "hi"}

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            client.post("/apis/agents/v2/workspaces/default/deployments/calc-dep/-/generate", json=payload)

        assert json.loads(_forwarded_request(httpx_mock)["content"]) == payload

    @pytest.mark.parametrize(
        "raw",
        [b"", b"not json", b"[1, 2]", b'{"messages": []}'],
        ids=["empty", "non-json", "json-array", "no-model-key"],
    )
    def test_bodies_without_a_model_object_pass_through_verbatim(
        self, client: TestClient, mock_entity_client: AsyncMock, raw: bytes
    ) -> None:
        dep = _make_deployment(config=self._NAT_CONFIG)
        mock_entity_client.get = AsyncMock(return_value=dep)
        httpx_mock = _make_httpx_mock(200, b"{}")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                content=raw,
                headers={"content-type": "application/json"},
            )

        assert _forwarded_request(httpx_mock)["content"] == raw


class TestStripRequestModelHelper:
    def test_removes_only_the_model_key(self) -> None:
        out = gateway_module._strip_request_model(b'{"model": "x", "messages": [], "temperature": 0}')
        assert out is not None
        assert json.loads(out) == {"messages": [], "temperature": 0}

    @pytest.mark.parametrize("raw", [b"", b"\xff\xfe", b"[]", b"42", b'{"messages": []}'])
    def test_returns_none_when_nothing_to_strip(self, raw: bytes) -> None:
        assert gateway_module._strip_request_model(raw) is None

    def test_is_nat_deployment(self) -> None:
        assert gateway_module._is_nat_deployment(_make_deployment(config={}))
        assert gateway_module._is_nat_deployment(_make_deployment(config={"workflow": {}}))
        assert not gateway_module._is_nat_deployment(
            _make_deployment(config={"config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT})
        )


# ---------------------------------------------------------------------------
# Container-mode (docker/k8s) endpoint resolution — gateway stop-gap
# ---------------------------------------------------------------------------
#
# The gateway must resolve a container-mode agent's address from the projected
# ``endpoints`` (deployments-plugin Deployment) instead of the loopback
# ``endpoint``, treat a running container deployment as routable, and 503 an
# unready one — matching the subprocess contract. (The controller projects the
# deployments-plugin READY status onto the agents-local "running".) Subprocess
# resolution is covered by the classes above and must remain unchanged (regression guard).


class TestContainerModeByDeploymentName:
    def test_ready_container_resolves_from_endpoints(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """A running k8s deployment proxies to its projected Service-DNS endpoint, not loopback."""
        dep = _make_container_deployment(
            mode="k8s",
            status="running",
            endpoints=[Endpoint(name="http", url="http://calc-dep.default.svc.cluster.local:8080")],
        )
        mock_entity_client.get = AsyncMock(return_value=dep)

        httpx_mock = _make_httpx_mock(200, b'{"answer": 42}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock) as mock_cls:
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={"messages": []},
            )

        assert resp.status_code == 200
        # The stream target must be built from the projected Service-DNS endpoint.
        stream_call = mock_cls.return_value.__aenter__.return_value.stream
        assert stream_call.call_args.kwargs["url"] == (
            "http://calc-dep.default.svc.cluster.local:8080/v1/chat/completions"
        )

    def test_docker_container_resolves_from_endpoints(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Docker mode is mode-agnostic to the gateway: resolve from endpoints just like k8s."""
        dep = _make_container_deployment(
            mode="docker",
            status="running",
            endpoints=[Endpoint(name="http", url="http://127.0.0.1:32770")],
        )
        mock_entity_client.get = AsyncMock(return_value=dep)

        httpx_mock = _make_httpx_mock(200, b'{"ok": true}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={"messages": []},
            )

        assert resp.status_code == 200

    def test_unready_container_returns_503(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """A container deployment whose projected status isn't ready is 503, not routed."""
        dep = _make_container_deployment(mode="k8s", status="pending")
        mock_entity_client.get = AsyncMock(return_value=dep)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 503
        assert "not routable" in resp.json()["detail"].lower()

    def test_ready_container_without_endpoints_returns_503(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """Running but no projected endpoint yet → 503 (nothing to dial)."""
        dep = _make_container_deployment(mode="k8s", status="running", endpoints=[])
        mock_entity_client.get = AsyncMock(return_value=dep)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 503

    def test_missing_container_deployment_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("not found"))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/nonexistent/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 404

    def test_service_dns_cross_origin_trailing_uri_rejected(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """The SSRF origin guard still fires when the resolved origin is a Service DNS name."""
        dep = _make_container_deployment(
            mode="k8s",
            status="running",
            endpoints=[Endpoint(name="http", url="http://calc-dep.default.svc.cluster.local:8080")],
        )
        mock_entity_client.get = AsyncMock(return_value=dep)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/http:%2F%2Fevil.example.com/x",
            json={},
        )

        assert resp.status_code == 400
        assert "invalid proxy target" in resp.json()["detail"].lower()


class TestContainerModeByAgentName:
    def test_ready_container_resolved_by_agent_name(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Call-by-agent-name picks a running container deployment and dials its endpoint."""
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        dep = _make_container_deployment(agent="calc", mode="k8s", status="running")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        httpx_mock = _make_httpx_mock(200, b'{"ok": true}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
                json={"messages": []},
            )

        assert resp.status_code == 200

    def test_no_ready_container_returns_503(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Only an unready container deployment exists → 503."""
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        dep = _make_container_deployment(agent="calc", mode="k8s", status="pending")
        mock_entity_client.list = AsyncMock(return_value=_list_response([dep]))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# OpenAI-shaped errors on the /-/v1/* surface
# ---------------------------------------------------------------------------


RELAY_FAILURE_DETAIL = (
    "Fabric runtime startup failed: adapter lifecycle start failed "
    "(claude_relay_unavailable): NeMo Relay CLI executable was not found"
)


class TestOpenAICompatibleErrors:
    def test_upstream_detail_is_lifted_into_error_message(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """A FastAPI-shaped upstream body is unwrapped, not embedded as a JSON string."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        httpx_mock = _make_httpx_mock(503, b'{"detail": "boom"}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 502
        assert resp.json()["error"] == {
            "message": "boom",
            "type": "upstream_error",
            "code": None,
            "param": None,
        }

    def test_adapter_error_code_is_preserved(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """The motivating case: the relay message and its code both reach the client."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        httpx_mock = _make_httpx_mock(503, json.dumps({"detail": RELAY_FAILURE_DETAIL}).encode())

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        error = resp.json()["error"]
        assert error["message"] == RELAY_FAILURE_DETAIL
        assert error["code"] == "claude_relay_unavailable"
        assert "NeMo Relay CLI executable was not found" in error["message"]

    def test_openai_shaped_upstream_body_passes_through(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """An upstream that already speaks OpenAI is read directly rather than re-wrapped."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        upstream = json.dumps({"error": {"message": "rate limited", "code": "slow_down"}}).encode()
        httpx_mock = _make_httpx_mock(500, upstream)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.json()["error"]["message"] == "rate limited"
        assert resp.json()["error"]["code"] == "slow_down"

    def test_non_json_upstream_body_still_yields_an_envelope(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        httpx_mock = _make_httpx_mock(500, b"Internal server error in agent")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 502
        assert resp.json()["error"]["message"] == "Internal server error in agent"
        assert resp.json()["error"]["type"] == "upstream_error"

    def test_oversized_upstream_body_is_truncated(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """The 500-char cap bounds the envelope, not just the legacy detail string."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        long_detail = "x" * 5000
        httpx_mock = _make_httpx_mock(500, json.dumps({"detail": long_detail}).encode())

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.json()["error"]["message"] == "x" * openai_errors.UPSTREAM_BODY_MAX_CHARS

    def test_gateway_side_failure_is_also_enveloped(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Errors raised before the request leaves the gateway get an envelope too."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="starting", endpoint=""))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 503
        error = resp.json()["error"]
        assert "not routable" in error["message"].lower()
        assert error["type"] == "server_error"

    def test_connection_failure_is_enveloped(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        client_cm = MagicMock()
        client_cm.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=client_cm):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 502
        assert "Could not connect" in resp.json()["error"]["message"]

    def test_not_found_is_enveloped_with_a_typed_error(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("not found"))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/deployments/nonexistent/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "not_found_error"

    def test_agent_name_route_is_enveloped(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """The by-agent-name surface is OpenAI-compatible under /-/v1/* too."""
        mock_entity_client.get = AsyncMock(return_value=_make_agent("calc"))
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        resp = client.post(
            "/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions",
            json={},
        )

        assert resp.status_code == 503
        assert "No running deployment" in resp.json()["error"]["message"]

    def test_detail_is_retained_for_backwards_compatibility(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """Existing readers of the FastAPI shape keep working: both keys are present."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        httpx_mock = _make_httpx_mock(503, b'{"detail": "boom"}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        body = resp.json()
        assert body["detail"] == 'Agent returned 503: {"detail": "boom"}'
        assert body["error"]["message"] == "boom"

    def test_non_openai_path_keeps_detail_only(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Only the /-/v1/* surface changes shape."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        httpx_mock = _make_httpx_mock(503, b'{"detail": "boom"}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/health",
                json={},
            )

        assert resp.status_code == 502
        assert resp.json() == {"detail": 'Agent returned 503: {"detail": "boom"}'}

    def test_2xx_response_is_untouched(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """Envelope shaping must not disturb the success path."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        upstream_body = b'{"id": "chatcmpl-1", "model": "calc-dep"}'
        httpx_mock = _make_httpx_mock(200, upstream_body)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 200
        assert resp.content == upstream_body

    def test_upstream_4xx_keeps_its_status_and_gains_an_envelope(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """A 4xx body no OpenAI client could read is augmented, not passed through blind."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        httpx_mock = _make_httpx_mock(422, b'{"detail": "invalid input"}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 422
        body = resp.json()
        assert body["detail"] == "invalid input"
        assert body["error"] == {
            "message": "invalid input",
            "type": "invalid_request_error",
            "code": None,
            "param": None,
        }

    def test_unknown_session_404_reaches_the_client(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """The agent server 404s an unknown session (fabric/server.py); the reason must survive."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        detail = "Fabric session 'abc' was not found."
        httpx_mock = _make_httpx_mock(404, json.dumps({"detail": detail}).encode())

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == detail
        assert resp.json()["error"]["type"] == "not_found_error"

    def test_upstream_validation_error_list_is_enveloped(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        """FastAPI puts a list on ``detail`` for a 422; the list is kept and a message derived."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        upstream = json.dumps({"detail": [{"loc": ["body"], "msg": "field required"}]}).encode()
        httpx_mock = _make_httpx_mock(422, upstream)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        body = resp.json()
        assert body["detail"] == [{"loc": ["body"], "msg": "field required"}]
        assert body["error"]["message"] == "body: field required"

    def test_openai_shaped_4xx_is_forwarded_untouched(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        """An agent that already speaks OpenAI keeps the type and code it chose."""
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        upstream = json.dumps(
            {"error": {"message": "bad key", "type": "authentication_error", "code": "invalid_api_key"}}
        ).encode()
        httpx_mock = _make_httpx_mock(401, upstream)

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 401
        assert resp.content == upstream

    def test_non_json_4xx_is_enveloped(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        httpx_mock = _make_httpx_mock(400, b"Bad Request", content_type="text/plain")

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/v1/chat/completions",
                json={},
            )

        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["message"] == "Bad Request"

    def test_non_openai_path_4xx_is_passed_through_verbatim(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_deployment(status="running"))
        httpx_mock = _make_httpx_mock(422, b'{"detail": "invalid input"}')

        with patch("nemo_agents_plugin.api.v2.gateway.httpx.AsyncClient", return_value=httpx_mock):
            resp = client.post(
                "/apis/agents/v2/workspaces/default/deployments/calc-dep/-/health",
                json={},
            )

        assert resp.status_code == 422
        assert resp.json() == {"detail": "invalid input"}


class TestUnwrapUpstreamError:
    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (b'{"detail": "boom"}', ("boom", None)),
            (b'{"message": "boom"}', ("boom", None)),
            (b'{"error": "boom"}', ("boom", None)),
            (b'{"error": {"message": "boom", "code": "c"}}', ("boom", "c")),
            (b'{"detail": "boom", "code": "c"}', ("boom", "c")),
            (b'"boom"', ("boom", None)),
            (b"boom", ("boom", None)),
            (b"", ("", None)),
            (b'{"detail": "failed (some_code): why"}', ("failed (some_code): why", "some_code")),
        ],
    )
    def test_message_and_code_extraction(self, body: bytes, expected: tuple[str, str | None]) -> None:
        assert openai_errors.unwrap_upstream_error(body) == expected

    def test_unrecognized_json_falls_back_to_raw_text(self) -> None:
        assert openai_errors.unwrap_upstream_error(b'{"nope": 1}') == ('{"nope": 1}', None)

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (b'{"detail": [{"loc": ["body"], "msg": "field required"}]}', "body: field required"),
            (
                b'{"detail": [{"loc": ["body", "messages", 0], "msg": "bad"}, {"loc": ["query"], "msg": "nope"}]}',
                "body.messages.0: bad; query: nope",
            ),
            (b'{"detail": [{"msg": "no location"}]}', "no location"),
        ],
    )
    def test_validation_error_list_is_rendered_readably(self, body: bytes, expected: str) -> None:
        """FastAPI validation errors are the most common 4xx here; raw JSON is unreadable."""
        assert openai_errors.unwrap_upstream_error(body)[0] == expected

    def test_unrenderable_detail_list_falls_back_to_raw_text(self) -> None:
        body = b'{"detail": [1, 2]}'
        message, code = openai_errors.unwrap_upstream_error(body)
        assert message == body.decode()
        assert code is None

    def test_invalid_utf8_does_not_raise(self) -> None:
        message, code = openai_errors.unwrap_upstream_error(b"\xff\xfe not utf-8")
        assert "not utf-8" in message
        assert code is None


class TestAugmentUpstreamErrorBody:
    def test_existing_keys_are_preserved(self) -> None:
        out = openai_errors.augment_upstream_error_body(b'{"detail": "boom", "trace": "t"}', 404)
        assert out is not None
        assert json.loads(out)["detail"] == "boom"
        assert json.loads(out)["trace"] == "t"

    def test_usable_openai_body_is_left_alone(self) -> None:
        body = b'{"error": {"message": "boom", "type": "x", "code": "y"}}'
        assert openai_errors.augment_upstream_error_body(body, 400) is None

    @pytest.mark.parametrize(
        "body",
        [
            b'{"error": {"code": "y"}}',
            b'{"error": {"message": ""}}',
            b'{"error": "boom"}',
        ],
    )
    def test_unusable_error_key_is_replaced(self, body: bytes) -> None:
        """``error`` alone is not enough — the SDK reads ``error.message`` or reports no body."""
        out = openai_errors.augment_upstream_error_body(body, 400)
        assert out is not None
        assert json.loads(out)["error"]["message"]

    def test_non_json_body_text_is_carried_in_the_message(self) -> None:
        out = openai_errors.augment_upstream_error_body(b"Bad Request", 400)
        assert out is not None
        assert json.loads(out) == {
            "error": {"message": "Bad Request", "type": "invalid_request_error", "code": None, "param": None}
        }


class TestIsOpenAICompatibleURI:
    @pytest.mark.parametrize(
        ("trailing_uri", "expected"),
        [
            ("v1/chat/completions", True),
            ("/v1/chat/completions", True),
            ("v1", True),
            ("v1beta/chat", False),
            ("health", False),
            ("", False),
            ("api/v1/chat", False),
        ],
    )
    def test_only_the_v1_surface_matches(self, trailing_uri: str, expected: bool) -> None:
        assert openai_errors.is_openai_compatible_uri(trailing_uri) is expected

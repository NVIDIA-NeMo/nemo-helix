# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgentSession CRUD and lifecycle routes."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin import session_lifecycle as session_lifecycle_module
from nemo_agents_plugin.api.v2 import sessions as sessions_router_module
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.entities import AgentDeployment, AgentSession, SessionStatus
from nemo_platform_plugin.dependencies import get_effective_principal_id
from nemo_platform_plugin.entity_client import (
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
    NemoPaginationInfo,
)

NOW = datetime.now(timezone.utc)
BASE = "/apis/agents/v2/workspaces/default/sessions"
OWNER_PRINCIPAL_ID = "session-owner"


def _make_deployment(
    *,
    name: str = "fabric-dep",
    workspace: str = "default",
    deployment_id: str = "deployment-id",
    created_by: str | None = None,
) -> AgentDeployment:
    deployment = AgentDeployment(name=name, workspace=workspace, agent="fabric-agent", status="running")
    deployment._id = deployment_id
    deployment._created_at = NOW
    deployment._created_by = created_by
    return deployment


def _make_session(
    *,
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
    session._id = f"session-{name}-id"
    session._created_at = NOW
    session._created_by = created_by
    return session


async def _persist_session(session: AgentSession) -> AgentSession:
    assert session.created_by is None
    session._id = "session-id"
    session._created_at = NOW
    session._created_by = OWNER_PRINCIPAL_ID
    return session


def _list_response(*sessions: AgentSession) -> MagicMock:
    response = MagicMock()
    response.data = list(sessions)
    response.pagination = NemoPaginationInfo(
        page=1,
        page_size=20,
        current_page_size=len(sessions),
        total_pages=1,
        total_results=len(sessions),
    )
    return response


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(
        sessions_router_module.router,
        prefix="/apis/agents/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    app.dependency_overrides[get_effective_principal_id] = lambda: OWNER_PRINCIPAL_ID
    return TestClient(app, raise_server_exceptions=False)


class TestCreateSession:
    def test_create_for_deployment(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.find_one.return_value = _make_deployment()
        mock_entity_client.create.side_effect = _persist_session
        response = client.post(BASE, json={"deployment_id": "deployment-id", "name": "session-one"})

        assert response.status_code == 201
        body = response.json()
        assert body["id"] == "session-id"
        assert body["name"] == "session-one"
        assert body["deployment_id"] == "deployment-id"
        assert body["status"] == "active"
        assert body["created_by"] == OWNER_PRINCIPAL_ID
        mock_entity_client.find_one.assert_awaited_once_with(
            AgentDeployment,
            workspace="default",
            filter_obj={"id": "deployment-id"},
        )

    def test_create_generates_name_from_deployment(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.find_one.return_value = _make_deployment(name="fabric-dep")
        mock_entity_client.create.side_effect = _persist_session
        with patch.object(sessions_router_module.secrets, "token_hex", return_value="a1b2c3d4"):
            response = client.post(BASE, json={"deployment_id": "deployment-id"})

        assert response.status_code == 201
        assert response.json()["name"] == "fabric-dep-a1b2c3d4"

    def test_create_for_deployment_owned_by_another_principal(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        mock_entity_client.find_one.return_value = _make_deployment(created_by="deployment-owner")
        mock_entity_client.create.side_effect = _persist_session

        response = client.post(BASE, json={"deployment_id": "deployment-id", "name": "session-one"})

        assert response.status_code == 201
        assert response.json()["created_by"] == OWNER_PRINCIPAL_ID

    def test_create_requires_deployment_id(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        response = client.post(BASE, json={"name": "session-one"})

        assert response.status_code == 422
        mock_entity_client.find_one.assert_not_awaited()

    def test_create_returns_404_for_unknown_deployment(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.find_one.side_effect = NemoEntityNotFoundError("not found")

        response = client.post(BASE, json={"deployment_id": "missing"})

        assert response.status_code == 404
        assert response.json()["detail"] == "Deployment ID 'missing' not found in workspace 'default'."
        mock_entity_client.create.assert_not_awaited()

    def test_create_returns_404_for_cross_workspace_deployment(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        mock_entity_client.find_one.return_value = _make_deployment(workspace="other")

        response = client.post(BASE, json={"deployment_id": "deployment-id"})

        assert response.status_code == 404
        mock_entity_client.create.assert_not_awaited()

    def test_create_returns_409_for_duplicate_name(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.find_one.return_value = _make_deployment()
        mock_entity_client.create.side_effect = NemoEntityConflictError("exists")

        response = client.post(BASE, json={"deployment_id": "deployment-id", "name": "session-one"})

        assert response.status_code == 409

    def test_create_returns_422_for_invalid_name(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.find_one.return_value = _make_deployment()
        mock_entity_client.create.side_effect = NemoEntityValidationError("name: string does not match pattern")

        response = client.post(BASE, json={"deployment_id": "deployment-id", "name": "session one"})

        assert response.status_code == 422
        assert "pattern" in response.json()["detail"]


class TestListSessions:
    def test_list_sessions(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list.return_value = _list_response(_make_session())

        response = client.get(BASE)

        assert response.status_code == 200
        body = response.json()
        assert body["data"][0]["name"] == "session-one"
        assert body["pagination"]["total_results"] == 1
        assert mock_entity_client.list.await_args.kwargs["filter_obj"] == {"created_by": OWNER_PRINCIPAL_ID}

    def test_list_filters_by_deployment_id(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.list.return_value = _list_response(_make_session())

        response = client.get(f"{BASE}?filter[deployment_id]=deployment-id")

        assert response.status_code == 200
        assert mock_entity_client.list.await_args.kwargs["filter_obj"] == {
            "deployment_id": "deployment-id",
            "created_by": OWNER_PRINCIPAL_ID,
        }

    def test_list_rejects_unknown_filter(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        response = client.get(f"{BASE}?filter[unknown]=value")

        assert response.status_code == 422
        mock_entity_client.list.assert_not_awaited()


class TestGetSession:
    def test_get_by_name(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get.return_value = _make_session()

        response = client.get(f"{BASE}/session-one")

        assert response.status_code == 200
        assert response.json()["name"] == "session-one"
        mock_entity_client.get.assert_awaited_once_with(AgentSession, name="session-one", workspace="default")

    def test_get_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get.side_effect = NemoEntityNotFoundError("not found")

        response = client.get(f"{BASE}/missing")

        assert response.status_code == 404

    def test_get_returns_404_for_session_owned_by_another_principal(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        mock_entity_client.get.return_value = _make_session(created_by="other-principal")

        response = client.get(f"{BASE}/session-one")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session 'session-one' not found in workspace 'default'."


class TestCloseSession:
    @pytest.mark.parametrize(
        "status",
        [SessionStatus.ACTIVE, SessionStatus.EXPIRED, SessionStatus.LOST],
    )
    def test_close_session(
        self,
        status: SessionStatus,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        mock_entity_client.get.return_value = _make_session(status=status)
        mock_entity_client.update.side_effect = lambda session: session

        with patch.object(sessions_router_module, "_cleanup_fabric_runtime", new_callable=AsyncMock) as cleanup:
            response = client.post(f"{BASE}/session-one/close")

        assert response.status_code == 200
        assert response.json()["status"] == "closed"
        updated: AgentSession = mock_entity_client.update.await_args.args[0]
        assert updated.status is SessionStatus.CLOSED
        cleanup.assert_awaited_once_with(mock_entity_client, updated)

    def test_close_is_idempotent(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        session = _make_session(status=SessionStatus.CLOSED)
        mock_entity_client.get.return_value = session

        with patch.object(sessions_router_module, "_cleanup_fabric_runtime", new_callable=AsyncMock) as cleanup:
            response = client.post(f"{BASE}/session-one/close")

        assert response.status_code == 200
        assert response.json()["status"] == "closed"
        mock_entity_client.update.assert_not_awaited()
        cleanup.assert_awaited_once_with(mock_entity_client, session)

    def test_close_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get.side_effect = NemoEntityNotFoundError("not found")

        response = client.post(f"{BASE}/missing/close")

        assert response.status_code == 404

    def test_close_returns_404_for_session_owned_by_another_principal(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        mock_entity_client.get.return_value = _make_session(created_by="other-principal")

        response = client.post(f"{BASE}/session-one/close")

        assert response.status_code == 404
        mock_entity_client.update.assert_not_awaited()

    def test_close_is_idempotent_after_concurrent_close(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        active_session = _make_session()
        closed_session = _make_session(status=SessionStatus.CLOSED)
        mock_entity_client.get.side_effect = [active_session, closed_session]
        mock_entity_client.update.side_effect = NemoEntityConflictError("conflict")

        with patch.object(sessions_router_module, "_cleanup_fabric_runtime", new_callable=AsyncMock) as cleanup:
            response = client.post(f"{BASE}/session-one/close")

        assert response.status_code == 200
        assert response.json()["status"] == "closed"
        assert mock_entity_client.get.await_count == 2
        cleanup.assert_awaited_once_with(mock_entity_client, closed_session)

    @pytest.mark.parametrize("concurrent_status", [SessionStatus.EXPIRED, SessionStatus.LOST])
    def test_close_retries_after_concurrent_terminal_transition(
        self,
        concurrent_status: SessionStatus,
        client: TestClient,
        mock_entity_client: AsyncMock,
    ) -> None:
        active_session = _make_session()
        concurrently_updated_session = _make_session(status=concurrent_status)
        mock_entity_client.get.side_effect = [active_session, concurrently_updated_session]
        update_count = 0

        async def update(session: AgentSession) -> AgentSession:
            nonlocal update_count
            update_count += 1
            if update_count == 1:
                raise NemoEntityConflictError("conflict")
            return session

        mock_entity_client.update.side_effect = update

        with patch.object(sessions_router_module, "_cleanup_fabric_runtime", new_callable=AsyncMock) as cleanup:
            response = client.post(f"{BASE}/session-one/close")

        assert response.status_code == 200
        assert response.json()["status"] == "closed"
        assert mock_entity_client.update.await_count == 2
        cleanup.assert_awaited_once_with(mock_entity_client, concurrently_updated_session)

    def test_close_returns_409_for_other_concurrent_update(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        active_session = _make_session()
        concurrently_updated_session = _make_session()
        mock_entity_client.get.side_effect = [active_session, concurrently_updated_session]
        mock_entity_client.update.side_effect = NemoEntityConflictError("conflict")

        response = client.post(f"{BASE}/session-one/close")

        assert response.status_code == 409
        assert mock_entity_client.get.await_count == 2
        assert mock_entity_client.update.await_count == 2


class TestDeleteSession:
    def test_delete_session(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        session = _make_session()
        mock_entity_client.get.return_value = session

        with patch.object(sessions_router_module, "_cleanup_fabric_runtime", new_callable=AsyncMock) as cleanup:
            response = client.delete(f"{BASE}/session-one")

        assert response.status_code == 204
        mock_entity_client.get.assert_awaited_once_with(AgentSession, name="session-one", workspace="default")
        mock_entity_client.delete.assert_awaited_once_with(
            AgentSession,
            name="session-one",
            workspace="default",
            expected_db_version=session.db_version,
        )
        cleanup.assert_awaited_once_with(mock_entity_client, session)

    def test_delete_returns_404(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get.side_effect = NemoEntityNotFoundError("not found")

        response = client.delete(f"{BASE}/missing")

        assert response.status_code == 404
        mock_entity_client.delete.assert_not_awaited()

    def test_delete_returns_404_for_session_owned_by_another_principal(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        mock_entity_client.get.return_value = _make_session(created_by="other-principal")

        response = client.delete(f"{BASE}/session-one")

        assert response.status_code == 404
        mock_entity_client.delete.assert_not_awaited()

    def test_delete_returns_409_for_concurrent_update(self, client: TestClient, mock_entity_client: AsyncMock) -> None:
        mock_entity_client.get.return_value = _make_session()
        mock_entity_client.delete.side_effect = NemoEntityConflictError("conflict")

        response = client.delete(f"{BASE}/session-one")

        assert response.status_code == 409


class TestNoAuthSessionLifecycle:
    def test_service_owned_session_remains_accessible_without_a_principal(
        self, client: TestClient, mock_entity_client: AsyncMock
    ) -> None:
        client.app.dependency_overrides[get_effective_principal_id] = lambda: ""
        deployment = _make_deployment()
        session = _make_session(created_by="service:platform")
        mock_entity_client.find_one.return_value = deployment
        mock_entity_client.create.return_value = session

        create_response = client.post(BASE, json={"deployment_id": "deployment-id", "name": "session-one"})

        assert create_response.status_code == 201
        assert create_response.json()["created_by"] == "service:platform"

        mock_entity_client.list.return_value = _list_response(session)
        list_response = client.get(BASE)

        assert list_response.status_code == 200
        assert [item["name"] for item in list_response.json()["data"]] == ["session-one"]
        assert mock_entity_client.list.await_args.kwargs["filter_obj"] is None

        mock_entity_client.get.return_value = session
        get_response = client.get(f"{BASE}/session-one")

        assert get_response.status_code == 200

        mock_entity_client.update.side_effect = lambda updated: updated
        close_response = client.post(f"{BASE}/session-one/close")

        assert close_response.status_code == 200
        assert close_response.json()["status"] == "closed"

        delete_response = client.delete(f"{BASE}/session-one")

        assert delete_response.status_code == 204
        mock_entity_client.delete.assert_awaited_once()


class TestFabricRuntimeCleanup:
    async def test_cleanup_calls_bound_deployment_session_endpoint(self, mock_entity_client: AsyncMock) -> None:
        deployment = _make_deployment()
        deployment.endpoint = "http://localhost:9001"
        mock_entity_client.find_one.return_value = deployment
        session = _make_session(name="session/one")
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(204)

        cleanup_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(session_lifecycle_module.httpx, "AsyncClient", return_value=cleanup_client):
            await sessions_router_module._cleanup_fabric_runtime(mock_entity_client, session)

        assert len(requests) == 1
        assert requests[0].method == "DELETE"
        assert str(requests[0].url) == "http://localhost:9001/v1/sessions/session-session%2Fone-id"

    async def test_cleanup_ignores_missing_runtime(
        self,
        mock_entity_client: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        deployment = _make_deployment()
        deployment.endpoint = "http://localhost:9001"
        mock_entity_client.find_one.return_value = deployment
        session = _make_session()

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        cleanup_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(session_lifecycle_module.httpx, "AsyncClient", return_value=cleanup_client):
            await sessions_router_module._cleanup_fabric_runtime(mock_entity_client, session)

        assert "Fabric runtime cleanup failed" not in caplog.text

    async def test_cleanup_connection_failure_is_observable_and_not_retried(
        self,
        mock_entity_client: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        deployment = _make_deployment()
        deployment.endpoint = "http://localhost:9001"
        mock_entity_client.find_one.return_value = deployment
        session = _make_session()
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("refused", request=request)

        cleanup_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(session_lifecycle_module.httpx, "AsyncClient", return_value=cleanup_client):
            await sessions_router_module._cleanup_fabric_runtime(mock_entity_client, session)

        assert attempts == 1
        assert (
            "Fabric runtime cleanup failed for session ID 'session-session-one-id' in workspace 'default' "
            "(deployment ID 'deployment-id'): request failed: refused."
        ) in caplog.text

    async def test_cleanup_http_failure_is_observable(
        self,
        mock_entity_client: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        deployment = _make_deployment()
        deployment.endpoint = "http://localhost:9001"
        mock_entity_client.find_one.return_value = deployment
        session = _make_session()

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        cleanup_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(session_lifecycle_module.httpx, "AsyncClient", return_value=cleanup_client):
            await sessions_router_module._cleanup_fabric_runtime(mock_entity_client, session)

        assert (
            "Fabric runtime cleanup failed for session ID 'session-session-one-id' in workspace 'default' "
            "(deployment ID 'deployment-id'): request returned HTTP 503."
        ) in caplog.text

    async def test_cleanup_entity_lookup_failure_is_observable(
        self,
        mock_entity_client: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_entity_client.find_one.side_effect = RuntimeError("entity store unavailable")
        session = _make_session()

        await sessions_router_module._cleanup_fabric_runtime(mock_entity_client, session)

        assert (
            "Fabric runtime cleanup failed for session ID 'session-session-one-id' in workspace 'default' "
            "(deployment ID 'deployment-id'): could not resolve deployment: entity store unavailable."
        ) in caplog.text

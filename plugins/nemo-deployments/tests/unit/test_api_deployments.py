# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import list_response, make_deployment, make_deployment_config
from nemo_deployments_plugin.api.v2 import deployments as deployments_module
from nemo_deployments_plugin.api.v2.dependencies import get_entity_client
from nemo_deployments_plugin.entities import Deployment, DeploymentConfig, Prerequisite
from nemo_platform_plugin.auth import AuthContext
from nemo_platform_plugin.entity_client import (
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
)


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(
        deployments_module.router,
        prefix="/apis/deployments/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    return TestClient(app, raise_server_exceptions=False)


def test_create_deployment_validates_config(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.side_effect = NemoEntityNotFoundError("missing")
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployments",
        json={"name": "dep1", "deployment_config": "missing"},
    )
    assert resp.status_code == 404


def test_create_deployment_201(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.return_value = make_deployment_config()
    mock_entity_client.list.return_value = list_response([])
    mock_entity_client.create.return_value = make_deployment()
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployments",
        json={"name": "dep1", "deployment_config": "cfg1"},
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "PENDING"


def test_create_deployment_cycle_400(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.return_value = make_deployment_config()
    a = make_deployment("a")
    b = make_deployment("b")
    b.prerequisites = [Prerequisite(deployment_name="a")]
    mock_entity_client.list.return_value = list_response([a, b])
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployments",
        json={
            "name": "a",
            "deployment_config": "cfg1",
            "prerequisites": [{"deployment_name": "b"}],
        },
    )
    assert resp.status_code == 400
    assert "cycle" in resp.json()["detail"].lower()


def test_create_deployment_accepts_workspace_qualified_config_ref(
    client: TestClient, mock_entity_client: AsyncMock
) -> None:
    mock_entity_client.get.return_value = make_deployment_config("cfg1", workspace="other")
    mock_entity_client.list.return_value = list_response([])
    mock_entity_client.create.return_value = make_deployment()
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployments",
        json={"name": "dep1", "deployment_config": "other/cfg1"},
    )
    assert resp.status_code == 201
    mock_entity_client.get.assert_awaited_once_with(DeploymentConfig, name="cfg1", workspace="other")
    created = mock_entity_client.create.await_args.args[0]
    assert created.deployment_config == "cfg1"


def test_create_deployment_rejects_malformed_config_ref_400(client: TestClient) -> None:
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployments",
        json={"name": "dep1", "deployment_config": "/cfg1"},
    )
    assert resp.status_code == 400


def test_list_deployments_status_in(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.list.return_value = list_response([make_deployment()])
    resp = client.get(
        "/apis/deployments/v2/workspaces/default/deployments",
        params={"status_in": "pending,starting"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 1
    call_kwargs = mock_entity_client.list.await_args.kwargs
    assert call_kwargs["filter_operation"].operator.value == "$in"
    assert call_kwargs["filter_operation"].field == "data.status"


def test_list_deployments_invalid_status_in_400(client: TestClient) -> None:
    resp = client.get(
        "/apis/deployments/v2/workspaces/default/deployments",
        params={"status_in": "banana"},
    )
    assert resp.status_code == 400


def test_delete_deployment_marks_deleting(client: TestClient, mock_entity_client: AsyncMock) -> None:
    deployment = make_deployment()
    mock_entity_client.get.return_value = deployment
    resp = client.delete("/apis/deployments/v2/workspaces/default/deployments/dep1")
    assert resp.status_code == 204
    mock_entity_client.update.assert_awaited_once()
    updated = mock_entity_client.update.await_args.args[0]
    assert updated.status == "DELETING"


def test_delete_deployment_conflict_409(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.return_value = make_deployment()
    mock_entity_client.update.side_effect = NemoEntityConflictError("conflict")
    resp = client.delete("/apis/deployments/v2/workspaces/default/deployments/dep1")
    assert resp.status_code == 409


def test_create_deployment_validation_422(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.return_value = make_deployment_config()
    mock_entity_client.list.return_value = list_response([])
    mock_entity_client.create.side_effect = NemoEntityValidationError("name: string does not match pattern")
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployments",
        json={"name": "dep 1", "deployment_config": "cfg1"},
    )
    assert resp.status_code == 422
    assert "pattern" in resp.json()["detail"]


def test_create_deployment_ignores_raw_auth_context_headers(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.return_value = make_deployment_config()
    mock_entity_client.list.return_value = list_response([])

    async def _create(deployment: Deployment) -> Deployment:
        return deployment

    mock_entity_client.create.side_effect = _create
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployments",
        json={"name": "dep1", "deployment_config": "cfg1"},
        headers={
            "X-NMP-Principal-Id": "user:alice",
            "X-NMP-Principal-Email": "alice@example.com",
            "X-NMP-Principal-Groups": "research,platform",
        },
    )

    assert resp.status_code == 201
    created = mock_entity_client.create.await_args.args[0]
    assert created.auth_context is None


def test_create_deployment_captures_runtime_auth_context(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.return_value = make_deployment_config()
    mock_entity_client.list.return_value = list_response([])

    async def _create(deployment: Deployment) -> Deployment:
        return deployment

    mock_entity_client.create.side_effect = _create
    with patch(
        "nemo_deployments_plugin.api.v2.deployments.current_auth_context",
        return_value=AuthContext(
            principal_id="bearer@example.com",
            principal_email="bearer@example.com",
            principal_groups=["runtime"],
        ),
    ):
        resp = client.post(
            "/apis/deployments/v2/workspaces/default/deployments",
            json={"name": "dep1", "deployment_config": "cfg1"},
        )

    assert resp.status_code == 201
    created = mock_entity_client.create.await_args.args[0]
    assert created.auth_context is not None
    assert created.auth_context.principal_id == "bearer@example.com"
    assert created.auth_context.principal_email == "bearer@example.com"
    assert created.auth_context.principal_groups == ["runtime"]

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import list_response, make_deployment, make_deployment_config
from nemo_deployments_plugin.api.v2 import deployment_configs as configs_module
from nemo_deployments_plugin.api.v2.dependencies import get_entity_client
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
        configs_module.router,
        prefix="/apis/deployments/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    return TestClient(app, raise_server_exceptions=False)


def test_create_deployment_config_201(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.create.return_value = make_deployment_config("cfg1")
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployment-configs",
        json={"name": "cfg1", "containers": [{"name": "main", "image": "nginx"}]},
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "cfg1"


def test_create_deployment_config_accepts_camel_case_restart_policy(
    client: TestClient,
    mock_entity_client: AsyncMock,
) -> None:
    mock_entity_client.create.side_effect = lambda config: config

    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployment-configs",
        json={
            "name": "cfg1",
            "containers": [{"name": "main", "image": "nginx"}],
            "restartPolicy": "Never",
            "backoffLimit": 3,
        },
    )

    assert resp.status_code == 201
    created_config = mock_entity_client.create.await_args.args[0]
    assert created_config.restart_policy == "Never"
    assert created_config.backoff_limit == 3


def test_create_deployment_config_accepts_snake_case_top_level_fields(
    client: TestClient,
    mock_entity_client: AsyncMock,
) -> None:
    mock_entity_client.create.side_effect = lambda config: config

    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployment-configs",
        json={
            "name": "cfg1",
            "containers": [{"name": "main", "image": "nginx"}],
            "init_containers": [{"name": "init", "image": "busybox"}],
            "volume_mounts": [{"name": "data", "mountPath": "/data"}],
            "config_files": [{"path": "/etc/config.txt", "content": "value"}],
            "restart_policy": "Never",
            "backoff_limit": 3,
            "drift_recovery": {"max_attempts": 2},
            "backend_config": {"k8s": {"namespace": "dep-ns", "serviceAccount": "dep-sa"}},
            "workload_identity": {
                "enabled": True,
                "tokenExpirationSeconds": 900,
            },
        },
    )

    assert resp.status_code == 201
    await_args = mock_entity_client.create.await_args
    assert await_args is not None
    created_config = await_args.args[0]
    assert created_config.init_containers[0].name == "init"
    assert created_config.volume_mounts[0].mount_path == "/data"
    assert created_config.config_files[0].path == "/etc/config.txt"
    assert created_config.restart_policy == "Never"
    assert created_config.backoff_limit == 3
    assert created_config.drift_recovery.max_attempts == 2
    assert created_config.backend_config.k8s is not None
    assert created_config.backend_config.k8s.namespace == "dep-ns"
    assert created_config.backend_config.k8s.service_account == "dep-sa"
    assert created_config.workload_identity is not None
    assert created_config.workload_identity.token_expiration_seconds == 900


def test_create_deployment_config_rejects_zero_backoff_limit(
    client: TestClient,
    mock_entity_client: AsyncMock,
) -> None:
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployment-configs",
        json={
            "name": "cfg1",
            "containers": [{"name": "main", "image": "nginx"}],
            "backoff_limit": 0,
        },
    )

    assert resp.status_code == 422
    mock_entity_client.create.assert_not_awaited()


def test_get_deployment_config_404(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.side_effect = NemoEntityNotFoundError("missing")
    resp = client.get("/apis/deployments/v2/workspaces/default/deployment-configs/missing")
    assert resp.status_code == 404


def test_delete_deployment_config_204(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.list.return_value = list_response([])
    resp = client.delete("/apis/deployments/v2/workspaces/default/deployment-configs/cfg1")
    assert resp.status_code == 204
    mock_entity_client.delete.assert_awaited_once_with(
        configs_module.DeploymentConfig,
        name="cfg1",
        workspace="default",
    )


def test_delete_deployment_config_409_when_referenced(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.list.return_value = list_response([make_deployment("dep1")])
    resp = client.delete("/apis/deployments/v2/workspaces/default/deployment-configs/cfg1")
    assert resp.status_code == 409
    assert "referenced" in resp.json()["detail"].lower()
    mock_entity_client.delete.assert_not_awaited()


def test_delete_deployment_config_409_when_changed(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.list.return_value = list_response([])
    mock_entity_client.delete.side_effect = NemoEntityConflictError("changed")
    resp = client.delete("/apis/deployments/v2/workspaces/default/deployment-configs/cfg1")
    assert resp.status_code == 409


def test_create_deployment_config_409(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.create.side_effect = NemoEntityConflictError("exists")
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployment-configs",
        json={"name": "cfg1"},
    )
    assert resp.status_code == 409


def test_create_deployment_config_validation_422(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.create.side_effect = NemoEntityValidationError("name: string does not match pattern")
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/deployment-configs",
        json={"name": "cfg 1"},
    )
    assert resp.status_code == 422
    assert "pattern" in resp.json()["detail"]

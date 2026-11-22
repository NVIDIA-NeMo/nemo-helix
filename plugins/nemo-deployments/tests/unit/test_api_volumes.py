# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import list_response, make_deployment_config, make_volume
from nemo_deployments_plugin.api.v2 import volumes as volumes_module
from nemo_deployments_plugin.api.v2.dependencies import get_entity_client
from nemo_deployments_plugin.entities import Container, VolumeMount


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(
        volumes_module.router,
        prefix="/apis/deployments/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    return TestClient(app, raise_server_exceptions=False)


def test_create_volume_201(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.create.return_value = make_volume()
    resp = client.post(
        "/apis/deployments/v2/workspaces/default/volumes",
        json={"name": "vol1", "size": "5Gi"},
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "PENDING"
    created = mock_entity_client.create.await_args.args[0]
    assert created.name == "vol1"
    assert created.size == "5Gi"
    assert created.workspace == "default"


def test_list_volumes_200(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.list.return_value = list_response([make_volume()])
    resp = client.get("/apis/deployments/v2/workspaces/default/volumes")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 1


def test_delete_volume_204(client: TestClient, mock_entity_client: AsyncMock) -> None:
    volume = make_volume()
    mock_entity_client.list.return_value = list_response([])
    mock_entity_client.get.return_value = volume
    resp = client.delete("/apis/deployments/v2/workspaces/default/volumes/vol1")
    assert resp.status_code == 204
    mock_entity_client.update.assert_awaited_once()
    updated = mock_entity_client.update.await_args.args[0]
    assert updated.status == "DELETING"
    mock_entity_client.delete_by_id.assert_not_awaited()


def test_delete_volume_409_when_referenced(client: TestClient, mock_entity_client: AsyncMock) -> None:
    cfg = make_deployment_config("cfg1")
    cfg.containers = [
        Container(
            name="main",
            image="nginx",
            volumeMounts=[VolumeMount(name="vol1", mountPath="/data")],
        )
    ]
    mock_entity_client.list.return_value = list_response([cfg])
    resp = client.delete("/apis/deployments/v2/workspaces/default/volumes/vol1")
    assert resp.status_code == 409
    assert "referenced" in resp.json()["detail"].lower()
    mock_entity_client.delete_by_id.assert_not_awaited()

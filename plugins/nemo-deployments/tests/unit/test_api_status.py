# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import make_deployment, make_volume
from nemo_deployments_plugin.api.v2 import status as status_module
from nemo_deployments_plugin.api.v2.dependencies import get_entity_client
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError

_SERVICE_HEADERS = {"X-NHX-Principal-Id": "service:deployments"}
_USER_HEADERS = {"X-NHX-Principal-Id": "user@example.com"}


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(
        status_module.router,
        prefix="/apis/deployments/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    return TestClient(app, raise_server_exceptions=False)


def test_status_put_rejects_user(client: TestClient) -> None:
    resp = client.put(
        "/apis/deployments/v2/workspaces/default/deployments/dep1/status",
        json={"status": "READY"},
        headers=_USER_HEADERS,
    )
    assert resp.status_code == 403


def test_status_put_rejects_malformed_service_principal(client: TestClient) -> None:
    resp = client.put(
        "/apis/deployments/v2/workspaces/default/deployments/dep1/status",
        json={"status": "READY"},
        headers={"X-NHX-Principal-Id": "service:"},
    )
    assert resp.status_code == 403


def test_status_put_ignores_on_behalf_of_for_auth(client: TestClient) -> None:
    resp = client.put(
        "/apis/deployments/v2/workspaces/default/deployments/dep1/status",
        json={"status": "READY"},
        headers={
            "X-NHX-Principal-Id": "user@example.com",
            "X-NHX-Principal-On-Behalf-Of": "service:deployments",
        },
    )
    assert resp.status_code == 403


def test_status_put_accepts_service_principal(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.return_value = make_deployment()
    mock_entity_client.update.return_value = make_deployment()
    resp = client.put(
        "/apis/deployments/v2/workspaces/default/deployments/dep1/status",
        json={"status": "READY", "status_message": "up"},
        headers=_SERVICE_HEADERS,
    )
    assert resp.status_code == 200


def test_volume_status_put_accepts_service_principal(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.return_value = make_volume()
    mock_entity_client.update.return_value = make_volume()
    resp = client.put(
        "/apis/deployments/v2/workspaces/default/volumes/vol1/status",
        json={"status": "BOUND"},
        headers=_SERVICE_HEADERS,
    )
    assert resp.status_code == 200


def test_status_put_404(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get.side_effect = NemoEntityNotFoundError("missing")
    resp = client.put(
        "/apis/deployments/v2/workspaces/default/deployments/missing/status",
        json={"status": "READY"},
        headers=_SERVICE_HEADERS,
    )
    assert resp.status_code == 404

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise upstream failures through real routers and platform exception handlers."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import agents, deployments, environments, sessions
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_helix_plugin.api.response_policy import default_auth_dependency
from nemo_helix_plugin.client.errors import (
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers

DETAIL = "Workspace 'missing-workspace' not found"

# One case per independent catch boundary. The three environment collections
# share _list_entities; compute-specs represents that common implementation.
WORKSPACE_CASES = [
    ("GET", "agents", "list", None),
    ("DELETE", "agents/test-agent", "list", None),
    ("GET", "compute-specs", "list", None),
    ("GET", "deployments", "list", None),
    ("GET", "sessions", "list", None),
    ("POST", "sessions", "find_one", {"deployment_id": "test-deployment"}),
]


@pytest.fixture
def entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(entity_client: AsyncMock) -> Iterator[TestClient]:
    app = FastAPI()
    register_sdk_exception_handlers(app)
    for module in (agents, environments, deployments, sessions):
        app.include_router(module.router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    app.dependency_overrides[default_auth_dependency] = lambda: SimpleNamespace(auth_enabled=False)
    app.dependency_overrides[get_effective_principal_id] = lambda: "test-user"
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize("verb,path,method,payload", WORKSPACE_CASES)
def test_missing_workspace_preserves_upstream_detail(client, entity_client, verb, path, method, payload):
    getattr(entity_client, method).side_effect = NotFoundError(httpx.Response(404, json={"detail": DETAIL}))

    response = client.request(verb, f"/v2/workspaces/missing-workspace/{path}", json=payload)

    assert response.status_code == 404, response.text
    assert response.json() == {"detail": DETAIL}
    entity_client.create.assert_not_awaited()
    entity_client.delete.assert_not_awaited()


@pytest.mark.parametrize(
    "verb,path,method,error,status,detail",
    [
        (
            "GET",
            "agents?sort=--name",
            "list",
            BadRequestError(httpx.Response(400, json={"detail": "Invalid sort field -name"})),
            400,
            "Invalid sort field -name",
        ),
        (
            "GET",
            "compute-specs",
            "list",
            UnprocessableEntityError(httpx.Response(422, json={"detail": "Invalid list request"})),
            422,
            "Invalid list request",
        ),
        (
            "GET",
            "deployments",
            "list",
            PermissionDeniedError(httpx.Response(403, json={"detail": "Access denied"})),
            403,
            "Access denied",
        ),
        (
            "POST",
            "sessions",
            "find_one",
            PermissionDeniedError(httpx.Response(403, json={"detail": "Access denied"})),
            403,
            "Access denied",
        ),
        ("GET", "agents", "list", NemoEntityNotFoundError(DETAIL), 404, DETAIL),
        ("GET", "sessions", "list", ValueError("internal bug"), 500, "Failed to list sessions."),
        ("DELETE", "agents/test-agent", "list", RuntimeError("database failure"), 500, "Failed to check deployments."),
        ("POST", "sessions", "find_one", RuntimeError("database failure"), 500, "Failed to look up deployment."),
    ],
)
def test_request_error_boundaries(client, entity_client, verb, path, method, error, status, detail):
    getattr(entity_client, method).side_effect = error

    response = client.request(
        verb,
        f"/v2/workspaces/default/{path}",
        json={"deployment_id": "test-deployment"} if verb == "POST" else None,
    )

    assert response.status_code == status, response.text
    assert response.json() == {"detail": detail}
    entity_client.create.assert_not_awaited()
    entity_client.delete.assert_not_awaited()

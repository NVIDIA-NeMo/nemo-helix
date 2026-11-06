# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independently wired list routes reject overflow before downstream calls."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_helix_plugin.api import response_policy
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nhx.common.auth import get_auth_client
from nhx.common.service.dependencies import get_platform_config


def _route_client(
    module_name: str, endpoint_name: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, AsyncMock, str]:
    module = import_module(module_name)
    route = next(
        route
        for route in module.router.routes
        if isinstance(route, APIRoute)
        and getattr(route.endpoint, "__wrapped__", route.endpoint) is getattr(module, endpoint_name)
    )
    app = FastAPI()
    prefix = "/apis/agents/v2/workspaces/{workspace}" if module_name.startswith("nemo_agents_plugin.") else ""
    app.include_router(module.router, prefix=prefix)
    monkeypatch.setattr(response_policy, "current_auth_client", lambda request: SimpleNamespace(auth_enabled=False))
    entities = AsyncMock()
    app.dependency_overrides[module.get_entity_client] = lambda: entities
    app.dependency_overrides[get_auth_client] = lambda: SimpleNamespace(auth_enabled=False)
    app.dependency_overrides[get_platform_config] = lambda: SimpleNamespace()
    app.dependency_overrides[get_effective_principal_id] = lambda: "test-user"
    return TestClient(app), entities, (prefix + route.path_format).format(workspace="default")


@pytest.mark.parametrize(
    "module_name,endpoint_name",
    [
        ("nhx.core.files.api.v2.filesets.endpoints", "list_filesets"),
        ("nhx.core.secrets.api.v2.secrets.endpoints", "list_secrets"),
        ("nhx.core.auth.api.v2.iam.endpoints", "list_role_bindings"),
        ("nemo_agents_plugin.api.v2.agents", "list_agents"),
        ("nemo_agents_plugin.api.v2.environments", "list_compute_specs"),
        ("nemo_agents_plugin.api.v2.deployments", "list_deployments"),
        ("nemo_agents_plugin.api.v2.sessions", "list_sessions"),
    ],
)
def test_offset_overflow_never_reaches_entity_client(
    module_name: str, endpoint_name: str, monkeypatch: pytest.MonkeyPatch
):
    client, entities, path = _route_client(module_name, endpoint_name, monkeypatch)
    response = client.get(
        path,
        params={"page": 92233720368547760, "page_size": 100},
    )

    assert response.status_code == 422, response.text
    assert "page and page_size" in response.json()["detail"]
    assert "offset" in response.json()["detail"]
    entities.list.assert_not_awaited()


def test_unbounded_page_size_rejects_sql_limit_overflow(monkeypatch: pytest.MonkeyPatch):
    client, entities, path = _route_client("nhx.core.secrets.api.v2.secrets.endpoints", "list_secrets", monkeypatch)
    response = client.get(path, params={"page": 1, "page_size": 2**63})
    assert response.status_code == 422, response.text
    assert "page_size" in response.json()["detail"]
    assert "LIMIT" in response.json()["detail"]
    entities.list.assert_not_awaited()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgentEnvironment / EnvironmentSpec / ComputeSpec CRUD routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypeVar
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import environments as environments_router_module
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.entities import (
    AgentComputeSpec,
    AgentEnvironment,
    AgentEnvironmentSpec,
    AgentSandboxSpec,
    EnvironmentSpecInline,
    SandboxSpecInline,
)
from nemo_platform_plugin.entity import NemoEntity
from nemo_platform_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError

NOW = datetime.now(timezone.utc)

EntityT = TypeVar("EntityT", bound=NemoEntity)


def _stamp(entity: EntityT) -> EntityT:
    entity._id = f"{entity.__entity_type__}-{entity.name}-id"
    entity._created_at = NOW
    return entity


def _test_client(mock_entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(
        environments_router_module.router,
        prefix="/apis/agents/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    return TestClient(app, raise_server_exceptions=False)


class TestComputeSpecRoutes:
    def test_create(self) -> None:
        client_mock = AsyncMock()
        client_mock.create = AsyncMock(side_effect=lambda e: _stamp(e))
        client = _test_client(client_mock)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/compute-specs",
            json={"name": "c1", "resources": {"limits": {"cpu": "2"}}},
        )

        assert resp.status_code == 201
        created: AgentComputeSpec = client_mock.create.call_args[0][0]
        assert created.name == "c1"
        assert created.resources.limits == {"cpu": "2"}

    def test_create_conflict(self) -> None:
        client_mock = AsyncMock()
        client_mock.create = AsyncMock(side_effect=NemoEntityConflictError("exists"))
        client = _test_client(client_mock)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/compute-specs",
            json={"name": "c1", "resources": {}},
        )
        assert resp.status_code == 409

    def test_get_not_found(self) -> None:
        client_mock = AsyncMock()
        client_mock.get = AsyncMock(side_effect=NemoEntityNotFoundError("gone"))
        client = _test_client(client_mock)

        resp = client.get("/apis/agents/v2/workspaces/default/compute-specs/c1")
        assert resp.status_code == 404


class TestEnvironmentSpecRoutes:
    def test_create(self) -> None:
        client_mock = AsyncMock()
        client_mock.create = AsyncMock(side_effect=lambda e: _stamp(e))
        client = _test_client(client_mock)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/environment-specs",
            json={"name": "e1", "env": {"FOO": "bar"}, "mcp": {"search": {"url": "http://x"}}},
        )

        assert resp.status_code == 201
        created: AgentEnvironmentSpec = client_mock.create.call_args[0][0]
        assert created.name == "e1"
        assert created.env == {"FOO": "bar"}
        assert created.mcp["search"].url == "http://x"

    def test_delete(self) -> None:
        client_mock = AsyncMock()
        client_mock.delete = AsyncMock(return_value=None)
        client = _test_client(client_mock)

        resp = client.delete("/apis/agents/v2/workspaces/default/environment-specs/e1")
        assert resp.status_code == 204


class TestSandboxSpecRoutes:
    def test_create(self) -> None:
        client_mock = AsyncMock()
        client_mock.create = AsyncMock(side_effect=lambda e: _stamp(e))
        client = _test_client(client_mock)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/sandbox-specs",
            json={"name": "s1", "provider": "openshell", "provider_config": {"policy_ref": "default/p1"}},
        )

        assert resp.status_code == 201
        created: AgentSandboxSpec = client_mock.create.call_args[0][0]
        assert created.name == "s1"
        assert created.provider == "openshell"
        assert created.provider_config == {"policy_ref": "default/p1"}

    def test_create_conflict(self) -> None:
        client_mock = AsyncMock()
        client_mock.create = AsyncMock(side_effect=NemoEntityConflictError("exists"))
        client = _test_client(client_mock)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/sandbox-specs",
            json={"name": "s1", "provider": "openshell"},
        )
        assert resp.status_code == 409

    def test_get_not_found(self) -> None:
        client_mock = AsyncMock()
        client_mock.get = AsyncMock(side_effect=NemoEntityNotFoundError("gone"))
        client = _test_client(client_mock)

        resp = client.get("/apis/agents/v2/workspaces/default/sandbox-specs/s1")
        assert resp.status_code == 404

    def test_delete(self) -> None:
        client_mock = AsyncMock()
        client_mock.delete = AsyncMock(return_value=None)
        client = _test_client(client_mock)

        resp = client.delete("/apis/agents/v2/workspaces/default/sandbox-specs/s1")
        assert resp.status_code == 204


class TestEnvironmentRoutes:
    def test_create_with_refs(self) -> None:
        client_mock = AsyncMock()
        client_mock.create = AsyncMock(side_effect=lambda e: _stamp(e))
        client = _test_client(client_mock)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/environments",
            json={
                "name": "env1",
                "environment_spec": "default/e1",
                "sandbox_spec": "default/s1",
                "compute_spec": "default/c1",
            },
        )

        assert resp.status_code == 201
        created: AgentEnvironment = client_mock.create.call_args[0][0]
        assert created.name == "env1"
        assert created.environment_spec == "default/e1"
        assert created.sandbox_spec == "default/s1"
        assert created.compute_spec == "default/c1"

    def test_create_with_inline(self) -> None:
        client_mock = AsyncMock()
        client_mock.create = AsyncMock(side_effect=lambda e: _stamp(e))
        client = _test_client(client_mock)

        resp = client.post(
            "/apis/agents/v2/workspaces/default/environments",
            json={
                "name": "env2",
                "environment_spec": {"env": {"A": "1"}},
                "sandbox_spec": {"provider": "openshell", "provider_config": {"policy_ref": "default/p1"}},
            },
        )

        assert resp.status_code == 201
        created: AgentEnvironment = client_mock.create.call_args[0][0]
        assert isinstance(created.environment_spec, EnvironmentSpecInline)
        assert created.environment_spec.env == {"A": "1"}
        assert isinstance(created.sandbox_spec, SandboxSpecInline)
        assert created.sandbox_spec.provider == "openshell"

    def test_get(self) -> None:
        env = _stamp(AgentEnvironment(name="env1", workspace="default", environment_spec="default/e1"))
        client_mock = AsyncMock()
        client_mock.get = AsyncMock(return_value=env)
        client = _test_client(client_mock)

        resp = client.get("/apis/agents/v2/workspaces/default/environments/env1")
        assert resp.status_code == 200
        assert resp.json()["environment_spec"] == "default/e1"

    def test_list(self) -> None:
        env = _stamp(AgentEnvironment(name="env1", workspace="default"))
        result = AsyncMock()
        result.data = [env]
        result.pagination = None
        client_mock = AsyncMock()
        client_mock.list = AsyncMock(return_value=result)
        client = _test_client(client_mock)

        resp = client.get("/apis/agents/v2/workspaces/default/environments")
        assert resp.status_code == 200
        assert resp.json()["data"][0]["name"] == "env1"

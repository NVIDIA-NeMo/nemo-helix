# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for global-workspace entity resolution (ASTD-526)."""

import pytest
from httpx import AsyncClient

SHAREABLE = "model"
PRIVATE = "customization_config"


def _entities_url(workspace: str, entity_type: str, name: str = "") -> str:
    base = f"/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}"
    return f"{base}/{name}" if name else base


async def _make_workspace(client: AsyncClient, name: str) -> None:
    response = await client.post("/apis/entities/v2/workspaces", json={"name": name, "description": name})
    assert response.status_code == 201, response.text


async def _create(client: AsyncClient, workspace: str, entity_type: str, name: str, **data) -> None:
    response = await client.post(
        _entities_url(workspace, entity_type),
        json={"name": name, "data": data or {"marker": workspace}},
    )
    assert response.status_code in (200, 201), response.text


@pytest.fixture
async def workspaces(client: AsyncClient, ctx):
    for name in ("team-a", "team-b"):
        await _make_workspace(client, name)


@pytest.mark.integration
@pytest.mark.asyncio
class TestGlobalWorkspaceGet:
    async def test_shareable_entity_resolves_from_another_workspace(self, client: AsyncClient, workspaces):
        await _create(client, "default", SHAREABLE, "shared-llm")

        response = await client.get(_entities_url("team-a", SHAREABLE, "shared-llm"))

        assert response.status_code == 200
        assert response.json()["workspace"] == "default"

    async def test_local_entity_shadows_global(self, client: AsyncClient, workspaces):
        await _create(client, "default", SHAREABLE, "llm")
        await _create(client, "team-a", SHAREABLE, "llm")

        response = await client.get(_entities_url("team-a", SHAREABLE, "llm"))

        assert response.status_code == 200
        assert response.json()["workspace"] == "team-a"

    async def test_unshareable_type_does_not_resolve_globally(self, client: AsyncClient, workspaces):
        await _create(client, "default", PRIVATE, "private-config", target_id="llama-2-7b")

        response = await client.get(_entities_url("team-a", PRIVATE, "private-config"))

        assert response.status_code == 404

    async def test_workspace_scoped_entity_does_not_leak(self, client: AsyncClient, workspaces):
        await _create(client, "team-a", SHAREABLE, "private-llm")

        response = await client.get(_entities_url("team-b", SHAREABLE, "private-llm"))

        assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
class TestGlobalWorkspaceList:
    async def test_listing_a_workspace_includes_global_entities(self, client: AsyncClient, workspaces):
        await _create(client, "default", SHAREABLE, "shared-llm")
        await _create(client, "team-a", SHAREABLE, "local-llm")

        response = await client.get(_entities_url("team-a", SHAREABLE))

        assert response.status_code == 200
        listed = {(e["workspace"], e["name"]) for e in response.json()["data"]}
        assert ("default", "shared-llm") in listed
        assert ("team-a", "local-llm") in listed

    async def test_listing_does_not_include_other_workspaces(self, client: AsyncClient, workspaces):
        await _create(client, "team-b", SHAREABLE, "other-llm")

        response = await client.get(_entities_url("team-a", SHAREABLE))

        assert response.status_code == 200
        assert all(e["workspace"] != "team-b" for e in response.json()["data"])

    async def test_listing_unshareable_type_stays_workspace_scoped(self, client: AsyncClient, workspaces):
        await _create(client, "default", PRIVATE, "global-config", target_id="llama-2-7b")

        response = await client.get(_entities_url("team-a", PRIVATE))

        assert response.status_code == 200
        assert all(e["name"] != "global-config" for e in response.json()["data"])

    async def test_listing_the_global_workspace_is_not_duplicated(self, client: AsyncClient, workspaces):
        await _create(client, "default", SHAREABLE, "shared-llm")

        response = await client.get(_entities_url("default", SHAREABLE))

        assert response.status_code == 200
        names = [e["name"] for e in response.json()["data"] if e["name"] == "shared-llm"]
        assert len(names) == 1

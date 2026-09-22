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
    async def test_listing_a_workspace_does_not_fold_in_global_entities(self, client: AsyncClient, workspaces):
        """Sharing resolves names; it deliberately does not change what a listing returns."""
        await _create(client, "default", SHAREABLE, "shared-llm")
        await _create(client, "team-a", SHAREABLE, "local-llm")

        response = await client.get(_entities_url("team-a", SHAREABLE))

        assert response.status_code == 200
        listed = {(e["workspace"], e["name"]) for e in response.json()["data"]}
        assert listed == {("team-a", "local-llm")}

    async def test_a_globally_shared_entity_is_still_reachable_by_name(self, client: AsyncClient, workspaces):
        """The listing stays pure, but the shared entity is usable - which is the point."""
        await _create(client, "default", SHAREABLE, "shared-llm")

        response = await client.get(_entities_url("team-a", SHAREABLE, "shared-llm"))

        assert response.status_code == 200
        assert response.json()["workspace"] == "default"

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


@pytest.mark.integration
@pytest.mark.asyncio
class TestGlobalWorkspaceDelete:
    """entities.parent is ON DELETE CASCADE, so a shared parent can reach across workspaces."""

    async def _shared_model_with_foreign_adapter(self, client: AsyncClient) -> str:
        """Create a global base model and an adapter on it in team-a; return the parent id."""
        await _create(client, "default", SHAREABLE, "base-llm")
        response = await client.get(_entities_url("default", SHAREABLE, "base-llm"))
        parent_id = response.json()["id"]
        create = await client.post(
            _entities_url("team-a", "adapter"),
            json={"name": "my-adapter", "parent": parent_id, "data": {"marker": "team-a"}},
        )
        assert create.status_code in (200, 201), create.text
        return parent_id

    @staticmethod
    async def _get_adapter(client: AsyncClient, parent_id: str):
        """A child entity is only addressable with its parent id."""
        return await client.get(_entities_url("team-a", "adapter", "my-adapter") + f"?parent={parent_id}")

    async def test_delete_is_refused_when_another_workspace_has_children(self, client: AsyncClient, workspaces):
        await self._shared_model_with_foreign_adapter(client)

        response = await client.delete(_entities_url("default", SHAREABLE, "base-llm"))

        assert response.status_code == 409
        assert "team-a" in response.json()["detail"]

    async def test_refused_delete_leaves_both_entities_intact(self, client: AsyncClient, workspaces):
        parent_id = await self._shared_model_with_foreign_adapter(client)

        await client.delete(_entities_url("default", SHAREABLE, "base-llm"))

        assert (await client.get(_entities_url("default", SHAREABLE, "base-llm"))).status_code == 200
        assert (await self._get_adapter(client, parent_id)).status_code == 200

    async def test_same_workspace_children_do_not_block_delete(self, client: AsyncClient, workspaces):
        await _create(client, "team-a", SHAREABLE, "local-llm")
        parent_id = (await client.get(_entities_url("team-a", SHAREABLE, "local-llm"))).json()["id"]
        await client.post(
            _entities_url("team-a", "adapter"),
            json={"name": "local-adapter", "parent": parent_id, "data": {}},
        )

        response = await client.delete(_entities_url("team-a", SHAREABLE, "local-llm"))

        assert response.status_code == 200

    async def test_childless_entity_deletes_normally(self, client: AsyncClient, workspaces):
        await _create(client, "default", SHAREABLE, "lonely-llm")

        response = await client.delete(_entities_url("default", SHAREABLE, "lonely-llm"))

        assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
class TestDependentWorkspaceReporting:
    """The 409 must name every affected workspace, not a page's worth."""

    async def test_names_every_workspace_regardless_of_child_count(self, client: AsyncClient, workspaces):
        await _create(client, "default", SHAREABLE, "busy-base")
        parent_id = (await client.get(_entities_url("default", SHAREABLE, "busy-base"))).json()["id"]

        # team-b contributes a single child among many from team-a. A page-limited scan
        # ordered by recency drops whichever workspace falls outside the page.
        await client.post(
            _entities_url("team-b", "adapter"),
            json={"name": "lone-adapter", "parent": parent_id, "data": {}},
        )
        for i in range(25):
            await client.post(
                _entities_url("team-a", "adapter"),
                json={"name": f"bulk-{i:03d}", "parent": parent_id, "data": {}},
            )

        response = await client.delete(_entities_url("default", SHAREABLE, "busy-base"))

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "team-a" in detail
        assert "team-b" in detail

    async def test_each_workspace_named_once_not_once_per_child(self, client: AsyncClient, workspaces):
        await _create(client, "default", SHAREABLE, "multi-child")
        parent_id = (await client.get(_entities_url("default", SHAREABLE, "multi-child"))).json()["id"]
        for i in range(5):
            await client.post(
                _entities_url("team-a", "adapter"),
                json={"name": f"ad-{i}", "parent": parent_id, "data": {}},
            )

        response = await client.delete(_entities_url("default", SHAREABLE, "multi-child"))

        assert response.status_code == 409
        assert response.json()["detail"].count("team-a") == 1

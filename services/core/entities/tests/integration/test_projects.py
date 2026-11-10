# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for project API v2 endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.asyncio
class TestProjectCRUD:
    """Test CRUD operations for projects."""

    async def test_create_project(self, client: AsyncClient, ctx):
        """Test creating a new project."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/projects",
            json={
                "name": "ml-project",
                "description": "Machine Learning project",
            },
        )

        assert response.status_code == 201
        result = response.json()
        assert result["workspace"] == "default"
        assert result["name"] == "ml-project"
        assert result["description"] == "Machine Learning project"
        assert "id" in result
        assert "created_at" in result
        assert "updated_at" in result

    async def test_create_duplicate_project_fails(self, client: AsyncClient, ctx):
        """Test creating a duplicate project returns 409."""
        await client.post(
            "/apis/entities/v2/workspaces/default/projects",
            json={"name": "duplicate-project", "description": "First project"},
        )

        response = await client.post(
            "/apis/entities/v2/workspaces/default/projects",
            json={"name": "duplicate-project", "description": "Second project"},
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_get_project_by_name(self, client: AsyncClient, ctx):
        """Test retrieving a project by name."""
        await client.post(
            "/apis/entities/v2/workspaces/default/projects",
            json={"name": "my-project", "description": "Test project"},
        )

        response = await client.get("/apis/entities/v2/workspaces/default/projects/my-project")

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "my-project"
        assert result["description"] == "Test project"

    async def test_get_project_not_found(self, client: AsyncClient, ctx):
        """Test getting a non-existent project returns 404."""
        response = await client.get("/apis/entities/v2/workspaces/default/projects/nonexistent")
        assert response.status_code == 404

    async def test_list_projects(self, client: AsyncClient, ctx):
        """Test listing projects in a workspace."""
        await client.post(
            "/apis/entities/v2/workspaces/default/projects",
            json={"name": "project-1", "description": "First project"},
        )
        await client.post(
            "/apis/entities/v2/workspaces/default/projects",
            json={"name": "project-2", "description": "Second project"},
        )

        response = await client.get("/apis/entities/v2/workspaces/default/projects")

        assert response.status_code == 200
        result = response.json()
        assert "data" in result
        assert len(result["data"]) == 2
        names = [item["name"] for item in result["data"]]
        assert "project-1" in names
        assert "project-2" in names

    async def test_list_projects_with_pagination(self, client: AsyncClient, ctx):
        """Test pagination when listing projects."""
        for i in range(5):
            await client.post(
                "/apis/entities/v2/workspaces/default/projects",
                json={"name": f"project-{i}", "description": f"Project {i}"},
            )

        response = await client.get(
            "/apis/entities/v2/workspaces/default/projects",
            params={"page": 1, "page_size": 2},
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["data"]) == 2
        assert result["pagination"]["page"] == 1
        assert result["pagination"]["page_size"] == 2
        assert result["pagination"]["total_results"] == 5
        assert result["pagination"]["total_pages"] == 3

    async def test_update_project(self, client: AsyncClient, ctx):
        """Test updating a project."""
        await client.post(
            "/apis/entities/v2/workspaces/default/projects",
            json={"name": "update-test", "description": "Original description"},
        )

        response = await client.put(
            "/apis/entities/v2/workspaces/default/projects/update-test",
            json={"description": "Updated description"},
        )

        assert response.status_code == 200
        updated = response.json()
        assert updated["description"] == "Updated description"

    async def test_update_nonexistent_project_fails(self, client: AsyncClient, ctx):
        """Test updating a non-existent project returns 404."""
        response = await client.put(
            "/apis/entities/v2/workspaces/default/projects/nonexistent",
            json={"description": "New description"},
        )
        assert response.status_code == 404

    async def test_delete_project(self, client: AsyncClient, ctx):
        """Test deleting a project."""
        await client.post(
            "/apis/entities/v2/workspaces/default/projects",
            json={"name": "delete-me", "description": "Delete this project"},
        )

        response = await client.delete("/apis/entities/v2/workspaces/default/projects/delete-me")

        assert response.status_code == 200
        result = response.json()
        assert "id" in result

        get_response = await client.get("/apis/entities/v2/workspaces/default/projects/delete-me")
        assert get_response.status_code == 404

    async def test_delete_nonexistent_project_fails(self, client: AsyncClient, ctx):
        """Test deleting a non-existent project returns 404."""
        response = await client.delete("/apis/entities/v2/workspaces/default/projects/nonexistent")
        assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
class TestProjectValidation:
    """Test project input validation."""

    async def test_create_project_nonexistent_workspace_fails(self, client: AsyncClient):
        """Test creating project in non-existent workspace returns 422."""
        response = await client.post(
            "/apis/entities/v2/workspaces/nonexistent-workspace/projects",
            json={"name": "test-project", "description": "Test"},
        )
        assert response.status_code == 422
        assert "does not exist" in response.json()["detail"]


@pytest.mark.integration
async def test_delete_project_preserves_a_concurrent_change(client, ctx, repos, monkeypatch):
    """The selected project version is required at deletion; conflicts remain HTTP 409."""
    from nhx.core.entities.app.repository.exceptions import EntityVersionConflictError

    response = await client.post("/apis/entities/v2/workspaces/default/projects", json={"name": "concurrent"})
    assert response.status_code == 201
    selected = response.json()

    async def conflict(*, entity_id, expected_db_version, session=None):
        assert entity_id == selected["id"]
        assert expected_db_version is not None
        raise EntityVersionConflictError("changed")

    monkeypatch.setattr(repos["entity"], "delete_entity", conflict)
    deleted = await client.delete("/apis/entities/v2/workspaces/default/projects/concurrent")
    assert deleted.status_code == 409
    assert (await client.get("/apis/entities/v2/workspaces/default/projects/concurrent")).status_code == 200

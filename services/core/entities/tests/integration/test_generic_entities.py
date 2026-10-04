# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for generic entity API v2 endpoints."""

import json

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.asyncio
class TestEntityCRUD:
    """Test CRUD operations for entities using name-based routes."""

    async def test_create_entity(self, client: AsyncClient, ctx):
        """Test creating a new entity."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={
                "name": "test-config",
                "data": {
                    "target_id": "llama-2-7b",
                    "training_options": {"learning_rate": 0.01, "batch_size": 32},
                    "max_seq_length": 2048,
                },
            },
        )

        assert response.status_code == 201
        result = response.json()
        assert result["workspace"] == "default"
        assert result["entity_type"] == "customization_config"
        assert result["name"] == "test-config"
        assert result["data"]["target_id"] == "llama-2-7b"
        assert "id" in result
        assert "created_at" in result
        assert "updated_at" in result
        # Without auth, created_by/updated_by are None
        assert "created_by" in result
        assert "updated_by" in result

    async def test_create_entity_auto_generated_name(self, client: AsyncClient, ctx):
        """Test creating entity with auto-generated name."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"data": {"target_id": "llama-2-7b"}},
        )

        assert response.status_code == 201
        result = response.json()
        assert result["name"].startswith("customization-config-")
        assert len(result["name"]) == len("customization-config-") + 5

    async def test_create_duplicate_entity_fails(self, client: AsyncClient, ctx):
        """Test creating a duplicate entity returns 409."""
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/model",
            json={"name": "duplicate-test", "data": {"model_path": "/path/to/model"}},
        )

        response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/model",
            json={"name": "duplicate-test", "data": {"model_path": "/path/to/model"}},
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_get_entity_by_id(self, client: AsyncClient, ctx):
        """Test retrieving an entity by ID."""
        create_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "get-by-id-test", "data": {"target_id": "llama-2-7b"}},
        )
        assert create_response.status_code == 201
        created = create_response.json()

        response = await client.get(f"/apis/entities/v2/entities/{created['id']}")

        assert response.status_code == 200
        result = response.json()
        assert result["id"] == created["id"]
        assert result["name"] == "get-by-id-test"

    async def test_get_entity_by_name(self, client: AsyncClient, ctx):
        """Test retrieving an entity by name."""
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "my-config", "data": {"target_id": "llama-2-7b"}},
        )

        response = await client.get("/apis/entities/v2/workspaces/default/entities/customization_config/my-config")

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "my-config"
        assert result["data"]["target_id"] == "llama-2-7b"

    async def test_get_entity_not_found(self, client: AsyncClient, ctx):
        """Test getting a non-existent entity returns 404."""
        response = await client.get("/apis/entities/v2/workspaces/default/entities/customization_config/nonexistent")
        assert response.status_code == 404

    async def test_get_entity_by_id_not_found(self, client: AsyncClient):
        """Test getting a non-existent entity by ID returns 404."""
        response = await client.get("/apis/entities/v2/entities/nonexistent-id-12345")
        assert response.status_code == 404

    async def test_list_entities_by_type(self, client: AsyncClient, ctx):
        """Test listing entities by type."""
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "config-1", "data": {"target_id": "llama-2-7b"}},
        )
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "config-2", "data": {"target_id": "llama-3-8b"}},
        )
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/model",
            json={"name": "model-1", "data": {"model_path": "/path/to/model"}},
        )

        response = await client.get("/apis/entities/v2/workspaces/default/entities/customization_config")

        assert response.status_code == 200
        result = response.json()
        assert "data" in result
        assert len(result["data"]) == 2
        names = [item["name"] for item in result["data"]]
        assert "config-1" in names
        assert "config-2" in names
        assert "model-1" not in names

    async def test_list_entities_with_pagination(self, client: AsyncClient, ctx):
        """Test pagination when listing entities."""
        for i in range(5):
            await client.post(
                "/apis/entities/v2/workspaces/default/entities/customization_config",
                json={"name": f"config-{i}", "data": {"index": i}},
            )

        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"page": 1, "page_size": 2},
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["data"]) == 2
        assert result["pagination"]["page"] == 1
        assert result["pagination"]["page_size"] == 2
        assert result["pagination"]["total_results"] == 5
        assert result["pagination"]["total_pages"] == 3

    async def test_list_entities_returns_group_counts_for_filtered_field(self, client: AsyncClient, ctx):
        """Test list results include counts for a requested filtered data field."""
        entities = [
            {"name": "group-count-a-1", "data": {"insight_id": "insight-a", "is_deleted": False}},
            {"name": "group-count-a-2", "data": {"insight_id": "insight-a", "is_deleted": False}},
            {"name": "group-count-b-1", "data": {"insight_id": "insight-b", "is_deleted": False}},
            {"name": "group-count-deleted", "data": {"insight_id": "insight-a", "is_deleted": True}},
        ]
        for entity in entities:
            response = await client.post(
                "/apis/entities/v2/workspaces/default/entities/experiment_group",
                json=entity,
            )
            assert response.status_code == 201

        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/experiment_group",
            params={
                "count_by": "data.insight_id",
                "filter": json.dumps(
                    {
                        "data.insight_id": {"$in": ["insight-a", "insight-b"]},
                        "data.is_deleted": False,
                    }
                ),
                "page_size": 1,
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["group_counts"] == {"insight-a": 2, "insight-b": 1}
        assert len(result["data"]) == 1

    @pytest.mark.parametrize("count_by", ["name", ""])
    async def test_list_entities_rejects_unsupported_count_field(self, client: AsyncClient, ctx, count_by: str):
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/experiment_group",
            params={"count_by": count_by},
        )

        assert response.status_code == 400
        assert "supported string data field" in response.json()["detail"]

    async def test_update_entity_by_name(self, client: AsyncClient, ctx):
        """Test updating an entity by name."""
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "update-test", "data": {"target_id": "llama-2-7b", "learning_rate": 0.01}},
        )

        response = await client.put(
            "/apis/entities/v2/workspaces/default/entities/customization_config/update-test",
            json={"data": {"target_id": "llama-2-7b", "learning_rate": 0.02}},
        )

        assert response.status_code == 200
        updated = response.json()
        assert updated["data"]["learning_rate"] == 0.02
        # Verify created_by/updated_by fields are present
        assert "created_by" in updated
        assert "updated_by" in updated

    async def test_update_nonexistent_entity_fails(self, client: AsyncClient, ctx):
        """Test updating a non-existent entity returns 404."""
        response = await client.put(
            "/apis/entities/v2/workspaces/default/entities/customization_config/nonexistent",
            json={"data": {"key": "value"}},
        )
        assert response.status_code == 404

    async def test_delete_entity_by_name(self, client: AsyncClient, ctx):
        """Test deleting an entity by name."""
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "delete-me", "data": {"target_id": "llama-2-7b"}},
        )

        response = await client.delete("/apis/entities/v2/workspaces/default/entities/customization_config/delete-me")

        assert response.status_code == 200
        result = response.json()
        assert "id" in result

        get_response = await client.get("/apis/entities/v2/workspaces/default/entities/customization_config/delete-me")
        assert get_response.status_code == 404

    async def test_delete_nonexistent_entity_fails(self, client: AsyncClient, ctx):
        """Test deleting a non-existent entity returns 404."""
        response = await client.delete("/apis/entities/v2/workspaces/default/entities/customization_config/nonexistent")
        assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
class TestEntitySorting:
    """Test entity sorting functionality."""

    async def test_sort_by_name(self, client: AsyncClient, ctx):
        """Test sorting entities by name in ascending and descending order."""
        names = ["zebra", "apple", "mango"]
        for name in names:
            await client.post(
                "/apis/entities/v2/workspaces/default/entities/customization_config",
                json={"name": name, "data": {"value": 1}},
            )

        # Ascending
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "name"},
        )
        assert response.status_code == 200
        result_names = [item["name"] for item in response.json()["data"]]
        assert result_names == ["apple", "mango", "zebra"]

        # Descending
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "-name"},
        )
        assert response.status_code == 200
        result_names = [item["name"] for item in response.json()["data"]]
        assert result_names == ["zebra", "mango", "apple"]

    async def test_sort_by_nested_data_field(self, client: AsyncClient, ctx):
        """Test sorting by a field within the data JSON in both directions."""
        configs = [
            {"name": "config-a", "data": {"priority": 3}},
            {"name": "config-b", "data": {"priority": 1}},
            {"name": "config-c", "data": {"priority": 2}},
        ]
        for config in configs:
            await client.post(
                "/apis/entities/v2/workspaces/default/entities/customization_config",
                json={"name": config["name"], "data": config["data"]},
            )

        # Ascending
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "data.priority"},
        )
        assert response.status_code == 200
        priorities = [item["data"]["priority"] for item in response.json()["data"]]
        assert priorities == [1, 2, 3]

        # Descending
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "-data.priority"},
        )
        assert response.status_code == 200
        priorities = [item["data"]["priority"] for item in response.json()["data"]]
        assert priorities == [3, 2, 1]

    @pytest.mark.skip(reason="SQLite JSON path extraction differs from PostgreSQL for deeply nested fields")
    async def test_sort_by_deeply_nested_data_field(self, client: AsyncClient, ctx):
        """Test sorting by a deeply nested field (data.level1.level2)."""
        configs = [
            {"name": "config-a", "data": {"training": {"epochs": 100}}},
            {"name": "config-b", "data": {"training": {"epochs": 50}}},
            {"name": "config-c", "data": {"training": {"epochs": 75}}},
        ]
        for config in configs:
            await client.post(
                "/apis/entities/v2/workspaces/default/entities/customization_config",
                json={"name": config["name"], "data": config["data"]},
            )

        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "data.training.epochs"},
        )

        assert response.status_code == 200
        result = response.json()
        epochs = [item["data"]["training"]["epochs"] for item in result["data"]]
        assert epochs == [50, 75, 100]

    async def test_sort_with_pagination(self, client: AsyncClient, ctx):
        """Test that sorting is maintained across paginated results."""
        for i in range(6):
            await client.post(
                "/apis/entities/v2/workspaces/default/entities/customization_config",
                json={
                    "name": f"entity-{i:02d}",
                    "data": {"score": (i * 17) % 6},  # Shuffle: 0, 5, 4, 3, 2, 1
                },
            )

        # Page 1 ascending by name
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "name", "page": 1, "page_size": 2},
        )
        assert response.status_code == 200
        names = [item["name"] for item in response.json()["data"]]
        assert names == ["entity-00", "entity-01"]

        # Page 2 ascending by name
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "name", "page": 2, "page_size": 2},
        )
        assert response.status_code == 200
        names = [item["name"] for item in response.json()["data"]]
        assert names == ["entity-02", "entity-03"]

        # Page 1 descending by name
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "-name", "page": 1, "page_size": 2},
        )
        assert response.status_code == 200
        names = [item["name"] for item in response.json()["data"]]
        assert names == ["entity-05", "entity-04"]

        # Nested data field with pagination
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "data.score", "page": 1, "page_size": 3},
        )
        assert response.status_code == 200
        scores = [item["data"]["score"] for item in response.json()["data"]]
        assert scores == [0, 1, 2]

        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            params={"sort": "data.score", "page": 2, "page_size": 3},
        )
        assert response.status_code == 200
        scores = [item["data"]["score"] for item in response.json()["data"]]
        assert scores == [3, 4, 5]


@pytest.mark.integration
@pytest.mark.asyncio
class TestParentScopedEntities:
    """Test parent-scoped entity operations."""

    async def test_create_entities_with_same_name_different_parents(self, client: AsyncClient, ctx):
        """Test that entities with the same name can exist under different parents."""
        # Create parent entities
        parent1_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/parent_type",
            json={"name": "parent-one", "data": {"type": "parent"}},
        )
        assert parent1_response.status_code == 201
        parent1_id = parent1_response.json()["id"]

        parent2_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/parent_type",
            json={"name": "parent-two", "data": {"type": "parent"}},
        )
        assert parent2_response.status_code == 201
        parent2_id = parent2_response.json()["id"]

        # Create child entities with the same name under different parents
        child1_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "child-name", "parent": parent1_id, "data": {"value": "under-parent-1"}},
        )
        assert child1_response.status_code == 201

        child2_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "child-name", "parent": parent2_id, "data": {"value": "under-parent-2"}},
        )
        assert child2_response.status_code == 201

    async def test_get_entity_by_name_with_parent(self, client: AsyncClient, ctx):
        """Test retrieving an entity by name with parent parameter."""
        # Create parent
        parent_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/parent_type",
            json={"name": "get-test-parent", "data": {"type": "parent"}},
        )
        assert parent_response.status_code == 201
        parent_id = parent_response.json()["id"]

        # Create child under parent
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "shared-name", "parent": parent_id, "data": {"value": "child-value"}},
        )

        # Create root-level entity with same name (no parent)
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "shared-name", "data": {"value": "root-value"}},
        )

        # Get child by name with parent - should return the child entity
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/child_type/shared-name",
            params={"parent": parent_id},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "shared-name"
        assert result["data"]["value"] == "child-value"
        assert result["parent"] == parent_id

        # Get root entity by name without parent - should return the root entity
        response = await client.get("/apis/entities/v2/workspaces/default/entities/child_type/shared-name")
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "shared-name"
        assert result["data"]["value"] == "root-value"
        assert result["parent"] is None

    async def test_get_entity_by_name_with_wrong_parent_fails(self, client: AsyncClient, ctx):
        """Test that getting an entity with wrong parent returns 404."""
        # Create parent
        parent_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/parent_type",
            json={"name": "wrong-parent-test", "data": {}},
        )
        parent_id = parent_response.json()["id"]

        # Create child under parent
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "orphan-child", "parent": parent_id, "data": {}},
        )

        # Try to get with a non-existent parent ID
        response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/child_type/orphan-child",
            params={"parent": "non-existent-parent-id"},
        )
        assert response.status_code == 404

    async def test_update_entity_by_name_with_parent(self, client: AsyncClient, ctx):
        """Test updating an entity by name with parent parameter."""
        # Create parent
        parent_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/parent_type",
            json={"name": "update-test-parent", "data": {}},
        )
        parent_id = parent_response.json()["id"]

        # Create child under parent
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "update-child", "parent": parent_id, "data": {"value": "original"}},
        )

        # Create root-level entity with same name
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "update-child", "data": {"value": "root-original"}},
        )

        # Update child by name with parent (parent is a query param, not in body)
        response = await client.put(
            "/apis/entities/v2/workspaces/default/entities/child_type/update-child",
            params={"parent": parent_id},
            json={"data": {"value": "updated"}},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["data"]["value"] == "updated"
        assert result["parent"] == parent_id

        # Verify root entity was not affected
        root_response = await client.get("/apis/entities/v2/workspaces/default/entities/child_type/update-child")
        assert root_response.status_code == 200
        assert root_response.json()["data"]["value"] == "root-original"

    async def test_delete_entity_by_name_with_parent(self, client: AsyncClient, ctx):
        """Test deleting an entity by name with parent parameter."""
        # Create parent
        parent_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/parent_type",
            json={"name": "delete-test-parent", "data": {}},
        )
        parent_id = parent_response.json()["id"]

        # Create child under parent
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "delete-child", "parent": parent_id, "data": {}},
        )

        # Create root-level entity with same name
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/child_type",
            json={"name": "delete-child", "data": {}},
        )

        # Delete child by name with parent
        response = await client.delete(
            "/apis/entities/v2/workspaces/default/entities/child_type/delete-child",
            params={"parent": parent_id},
        )
        assert response.status_code == 200
        assert "id" in response.json()

        # Verify root entity still exists
        root_response = await client.get("/apis/entities/v2/workspaces/default/entities/child_type/delete-child")
        assert root_response.status_code == 200

        # Verify child is deleted
        child_response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/child_type/delete-child",
            params={"parent": parent_id},
        )
        assert child_response.status_code == 404


@pytest.mark.asyncio
class TestEntityValidation:
    """Test entity input validation."""

    async def test_create_entity_nonexistent_workspace_fails(self, client: AsyncClient):
        """Test creating entity in non-existent workspace returns 422."""
        response = await client.post(
            "/apis/entities/v2/workspaces/nonexistent-workspace/entities/model",
            json={"name": "test", "data": {}},
        )
        assert response.status_code == 422
        assert "does not exist" in response.json()["detail"].lower()

    async def test_list_entities_nonexistent_workspace_fails(self, client: AsyncClient):
        """Test listing entities in non-existent workspace returns 404."""
        response = await client.get("/apis/entities/v2/workspaces/nonexistent-workspace/entities/model")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_name_validation_too_short(self, client: AsyncClient, ctx):
        """Test that names that are too short are rejected."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "a", "data": {}},
        )
        assert response.status_code == 422

    async def test_name_validation_invalid_start(self, client: AsyncClient, ctx):
        """Test that names starting with numbers are rejected."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "123abc", "data": {}},
        )
        assert response.status_code == 422

    async def test_name_validation_consecutive_hyphens(self, client: AsyncClient, ctx):
        """Test that names with consecutive hyphens are rejected."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "test--config", "data": {}},
        )
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.asyncio
class TestEntityVersioning:
    """Test entity versioning and optimistic locking."""

    async def test_create_entity_returns_version(self, client: AsyncClient, ctx):
        """Test that creating an entity returns version field."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "version-test", "data": {"target_id": "llama-2-7b"}},
        )

        assert response.status_code == 201
        result = response.json()
        assert "db_version" in result
        assert result["db_version"] == 1  # New entities start with version 1

    async def test_get_entity_returns_version(self, client: AsyncClient, ctx):
        """Test that getting an entity returns version field."""
        create_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "get-version-test", "data": {"target_id": "llama-2-7b"}},
        )
        assert create_response.status_code == 201
        created = create_response.json()
        assert created["db_version"] == 1

        get_response = await client.get(
            "/apis/entities/v2/workspaces/default/entities/customization_config/get-version-test"
        )

        assert get_response.status_code == 200
        result = get_response.json()
        assert "db_version" in result
        assert result["db_version"] == 1

    async def test_update_entity_increments_version(self, client: AsyncClient, ctx):
        """Test that updating an entity increments the version."""
        create_response = await client.post(
            "/apis/entities/v2/workspaces/default/entities/customization_config",
            json={"name": "update-version-test", "data": {"target_id": "llama-2-7b"}},
        )
        assert create_response.status_code == 201
        created = create_response.json()
        initial_version = created["db_version"]
        assert initial_version == 1

        # First update
        update1_response = await client.put(
            "/apis/entities/v2/workspaces/default/entities/customization_config/update-version-test",
            json={"data": {"target_id": "llama-2-7b", "learning_rate": 0.01}},
        )
        assert update1_response.status_code == 200
        updated1 = update1_response.json()
        assert updated1["db_version"] == initial_version + 1  # Version should increment

        # Second update
        update2_response = await client.put(
            "/apis/entities/v2/workspaces/default/entities/customization_config/update-version-test",
            json={"data": {"target_id": "llama-2-7b", "learning_rate": 0.02}},
        )
        assert update2_response.status_code == 200
        updated2 = update2_response.json()
        assert updated2["db_version"] == updated1["db_version"] + 1  # Version should increment again

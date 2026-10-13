# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-workspace entity repository queries."""

import pytest
from nhx.common.entities import ALL_WORKSPACES
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository


@pytest.mark.asyncio
class TestCrossWorkspaceQueries:
    """Test cross-workspace entity listing with '**' wildcard."""

    async def test_wildcard_lists_entities_from_all_workspaces(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """Test that '**' wildcard lists entities across all workspaces."""
        await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="config-1",
            data={"value": 1},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="config",
            name="config-2",
            data={"value": 2},
        )
        await entity_repo.create_entity(
            workspace="workspace-3",
            entity_type="config",
            name="config-3",
            data={"value": 3},
        )

        entities, total = await entity_repo.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type="config",
            page=1,
            page_size=100,
        )

        assert total == 3
        assert len(entities) == 3

        workspaces = {e.workspace for e in entities}
        assert workspaces == {"workspace-1", "workspace-2", "workspace-3"}

        names = {e.name for e in entities}
        assert names == {"config-1", "config-2", "config-3"}

    async def test_specific_workspace_only_lists_from_that_workspace(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """Test that specific workspace parameter filters correctly."""
        await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="config-1",
            data={"value": 1},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="config",
            name="config-2",
            data={"value": 2},
        )

        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            page=1,
            page_size=100,
        )

        assert total == 1
        assert len(entities) == 1
        assert entities[0].workspace == "workspace-1"
        assert entities[0].name == "config-1"

    async def test_wildcard_with_pagination(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Test cross-workspace queries with pagination."""
        for ws_idx in range(1, 4):
            for entity_idx in range(1, 4):
                await entity_repo.create_entity(
                    workspace=f"workspace-{ws_idx}",
                    entity_type="config",
                    name=f"config-{ws_idx}-{entity_idx}",
                    data={"workspace": ws_idx, "index": entity_idx},
                )

        # Total: 9 entities (3 workspaces x 3 entities each)

        entities_page1, total_page1 = await entity_repo.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type="config",
            page=1,
            page_size=3,
        )

        assert total_page1 == 9
        assert len(entities_page1) == 3

        entities_page2, total_page2 = await entity_repo.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type="config",
            page=2,
            page_size=3,
        )

        assert total_page2 == 9
        assert len(entities_page2) == 3

        page1_ids = {e.id for e in entities_page1}
        page2_ids = {e.id for e in entities_page2}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_wildcard_respects_entity_type_filter(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """Test that cross-workspace queries respect entity_type filtering."""
        await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="config-1",
            data={"type": "config"},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="model",
            name="model-1",
            data={"type": "model"},
        )
        await entity_repo.create_entity(
            workspace="workspace-3",
            entity_type="config",
            name="config-2",
            data={"type": "config"},
        )

        entities, total = await entity_repo.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type="config",
            page=1,
            page_size=100,
        )

        assert total == 2
        assert len(entities) == 2

        for entity in entities:
            assert entity.entity_type == "config"

        workspaces = {e.workspace for e in entities}
        assert "workspace-1" in workspaces
        assert "workspace-3" in workspaces
        assert "workspace-2" not in workspaces

    async def test_workspace_isolation_with_same_entity_names(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """Test that entities with same name in different workspaces are isolated."""
        await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="shared-name",
            data={"workspace_id": 1},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="config",
            name="shared-name",
            data={"workspace_id": 2},
        )

        entities_ws1, _ = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            page=1,
            page_size=100,
        )

        assert len(entities_ws1) == 1
        assert entities_ws1[0].workspace == "workspace-1"
        assert entities_ws1[0].data["workspace_id"] == 1

        entities_ws2, _ = await entity_repo.list_entities(
            workspace="workspace-2",
            entity_type="config",
            page=1,
            page_size=100,
        )

        assert len(entities_ws2) == 1
        assert entities_ws2[0].workspace == "workspace-2"
        assert entities_ws2[0].data["workspace_id"] == 2

        entities_all, total = await entity_repo.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type="config",
            page=1,
            page_size=100,
        )

        assert total == 2
        assert len(entities_all) == 2

        workspaces = {e.workspace for e in entities_all}
        assert workspaces == {"workspace-1", "workspace-2"}

    async def test_wildcard_with_sorting(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Test that cross-workspace queries respect sorting."""
        for i in range(1, 4):
            await entity_repo.create_entity(
                workspace=f"workspace-{i}",
                entity_type="config",
                name=f"config-{i}",
                data={"index": i},
            )

        entities, _ = await entity_repo.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type="config",
            page=1,
            page_size=100,
            sort="-created_at",
        )

        assert entities[0].name == "config-3"
        assert entities[1].name == "config-2"
        assert entities[2].name == "config-1"

    async def test_wildcard_with_empty_workspaces(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Test cross-workspace query when some workspaces are empty."""
        await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="config-1",
            data={"value": 1},
        )

        entities, total = await entity_repo.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type="config",
            page=1,
            page_size=100,
        )

        assert total == 1
        assert len(entities) == 1
        assert entities[0].workspace == "workspace-1"

    async def test_wildcard_returns_empty_when_no_entities(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """Test that cross-workspace query returns empty list when no entities exist."""
        entities, total = await entity_repo.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type="config",
            page=1,
            page_size=100,
        )

        assert total == 0
        assert len(entities) == 0

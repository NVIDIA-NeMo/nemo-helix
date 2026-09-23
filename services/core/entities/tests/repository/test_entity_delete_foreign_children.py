# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cross-workspace cascade guard on entity deletes (ASTD-526)."""

import pytest
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.exceptions import ForeignChildEntitiesError


@pytest.mark.asyncio
class TestDeleteRefusesForeignChildren:
    async def _parent_with_child_in(
        self, repo: SQLAlchemyEntityRepository, child_workspace: str, name: str = "base"
    ) -> str:
        parent = await repo.create_entity(workspace="workspace-1", entity_type="model", name=name, data={})
        await repo.create_entity(
            workspace=child_workspace,
            entity_type="adapter",
            name=f"{name}-adapter",
            data={},
            parent=parent.id,
        )
        return parent.id

    async def test_refuses_and_names_the_foreign_workspace(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        await self._parent_with_child_in(entity_repo, "workspace-2")

        with pytest.raises(ForeignChildEntitiesError) as excinfo:
            await entity_repo.delete_entity_by_name(
                workspace="workspace-1",
                entity_type="model",
                name="base",
                refuse_children_outside="workspace-1",
            )

        assert excinfo.value.workspaces == ["workspace-2"]

    async def test_refused_delete_leaves_the_entity_in_place(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        await self._parent_with_child_in(entity_repo, "workspace-2", name="survivor")

        with pytest.raises(ForeignChildEntitiesError):
            await entity_repo.delete_entity_by_name(
                workspace="workspace-1",
                entity_type="model",
                name="survivor",
                refuse_children_outside="workspace-1",
            )

        assert (
            await entity_repo.get_entity_by_name(workspace="workspace-1", entity_type="model", name="survivor")
            is not None
        )

    async def test_same_workspace_children_do_not_refuse(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        await self._parent_with_child_in(entity_repo, "workspace-1", name="local")

        deleted = await entity_repo.delete_entity_by_name(
            workspace="workspace-1",
            entity_type="model",
            name="local",
            refuse_children_outside="workspace-1",
        )

        assert deleted == 1

    async def test_guard_is_opt_in(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Internal callers (projects, cleanup) keep the unguarded behaviour by omitting it."""
        await self._parent_with_child_in(entity_repo, "workspace-2", name="forced")

        deleted = await entity_repo.delete_entity_by_name(workspace="workspace-1", entity_type="model", name="forced")

        assert deleted == 1

    async def test_childless_entity_is_unaffected(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        await entity_repo.create_entity(workspace="workspace-1", entity_type="model", name="lonely", data={})

        deleted = await entity_repo.delete_entity_by_name(
            workspace="workspace-1",
            entity_type="model",
            name="lonely",
            refuse_children_outside="workspace-1",
        )

        assert deleted == 1

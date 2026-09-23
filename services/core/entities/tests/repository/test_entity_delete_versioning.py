# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for optimistic locking on entity deletes."""

import pytest
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.exceptions import EntityVersionConflictError
from nhx.core.entities.app.repository.sqlalchemy.models import DBEntity
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@pytest.mark.asyncio
class TestEntityDeleteVersioning:
    async def test_delete_entity_by_name_with_matching_expected_db_version(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        entity = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="delete-with-version",
            data={"value": 1},
        )

        deleted_count = await entity_repo.delete_entity_by_name(
            workspace="workspace-1",
            entity_type="config",
            name="delete-with-version",
            expected_db_version=entity.db_version,
        )

        assert deleted_count == 1
        assert (
            await entity_repo.get_entity_by_name(
                workspace="workspace-1",
                entity_type="config",
                name="delete-with-version",
            )
            is None
        )

    async def test_delete_entity_by_name_rejects_stale_expected_db_version(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        entity = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="delete-stale-version",
            data={"value": 1},
        )
        updated = await entity_repo.update_entity_by_name(
            workspace="workspace-1",
            entity_type="config",
            name="delete-stale-version",
            data={"value": 2},
        )

        with pytest.raises(EntityVersionConflictError):
            await entity_repo.delete_entity_by_name(
                workspace="workspace-1",
                entity_type="config",
                name="delete-stale-version",
                expected_db_version=entity.db_version,
            )

        assert updated.db_version != entity.db_version
        assert (
            await entity_repo.get_entity_by_name(
                workspace="workspace-1",
                entity_type="config",
                name="delete-stale-version",
            )
            is not None
        )

    async def test_delete_entity_by_name_rolls_back_stale_commit(
        self,
        entity_repo: SQLAlchemyEntityRepository,
        session_maker: async_sessionmaker[AsyncSession],
        setup_workspaces,
    ):
        entity = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="delete-stale-commit",
            data={"value": 1},
        )

        async with session_maker() as shared_session:
            result = await shared_session.execute(select(DBEntity).where(DBEntity.id == entity.id))
            stale_entity = result.scalar_one()
            await shared_session.commit()

            await entity_repo.update_entity_by_name(
                workspace="workspace-1",
                entity_type="config",
                name="delete-stale-commit",
                data={"value": 2},
            )

            assert stale_entity.db_version == entity.db_version
            with pytest.raises(EntityVersionConflictError):
                await entity_repo.delete_entity_by_name(
                    workspace="workspace-1",
                    entity_type="config",
                    name="delete-stale-commit",
                    session=shared_session,
                )

            remaining = await entity_repo.get_entity_by_name(
                workspace="workspace-1",
                entity_type="config",
                name="delete-stale-commit",
                session=shared_session,
            )
            assert remaining is not None
            assert remaining.data == {"value": 2}

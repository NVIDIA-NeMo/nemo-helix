# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SQLite write lock serialization in the entity repository."""

import asyncio

import pytest
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository


@pytest.mark.asyncio
class TestSQLiteWriteLock:
    """Test that concurrent writes are serialized via the write lock on SQLite."""

    async def test_concurrent_creates_all_succeed(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Concurrent create_entity calls should all succeed without 'database is locked' errors."""
        num_concurrent = 1_000

        async def create_one(i: int):
            return await entity_repo.create_entity(
                workspace="workspace-1",
                entity_type="config",
                name=f"concurrent-{i}",
                data={"index": i},
            )

        results = await asyncio.gather(*[create_one(i) for i in range(num_concurrent)])

        assert len(results) == num_concurrent
        names = {r.name for r in results}
        assert names == {f"concurrent-{i}" for i in range(num_concurrent)}

        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            page=1,
            page_size=100,
        )
        assert total == num_concurrent

    async def test_concurrent_updates_all_succeed(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Concurrent update_entity calls should all succeed without conflicts."""
        num_concurrent = 1_000
        created = []
        for i in range(num_concurrent):
            entity = await entity_repo.create_entity(
                workspace="workspace-1",
                entity_type="config",
                name=f"to-update-{i}",
                data={"version": 0},
            )
            created.append(entity)

        async def update_one(entity):
            return await entity_repo.update_entity(
                entity_id=entity.id,
                data={"version": 1},
            )

        results = await asyncio.gather(*[update_one(e) for e in created])

        assert len(results) == num_concurrent
        for r in results:
            assert r.data["version"] == 1

    async def test_concurrent_deletes_all_succeed(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Concurrent delete_entity calls should all succeed."""
        num_concurrent = 1_000
        created = []
        for i in range(num_concurrent):
            entity = await entity_repo.create_entity(
                workspace="workspace-1",
                entity_type="config",
                name=f"to-delete-{i}",
                data={"index": i},
            )
            created.append(entity)

        async def delete_one(entity):
            return await entity_repo.delete_entity(entity_id=entity.id)

        results = await asyncio.gather(*[delete_one(e) for e in created])

        assert all(r == 1 for r in results)

        _, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            page=1,
            page_size=100,
        )
        assert total == 0

    async def test_concurrent_mixed_writes(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Concurrent creates, updates, and deletes should all succeed together."""
        to_update = await entity_repo.create_entity(
            workspace="workspace-1", entity_type="config", name="update-me", data={"v": 0}
        )
        to_delete = await entity_repo.create_entity(
            workspace="workspace-1", entity_type="config", name="delete-me", data={"v": 0}
        )

        async def do_create():
            return await entity_repo.create_entity(
                workspace="workspace-1", entity_type="config", name="new-entity", data={"v": 1}
            )

        async def do_update():
            return await entity_repo.update_entity(entity_id=to_update.id, data={"v": 1})

        async def do_delete():
            return await entity_repo.delete_entity(entity_id=to_delete.id)

        create_result, update_result, delete_result = await asyncio.gather(do_create(), do_update(), do_delete())

        assert create_result.name == "new-entity"
        assert update_result.data["v"] == 1
        assert delete_result == 1

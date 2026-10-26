# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for keeping stored data keys on entity updates."""

import pytest
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository


@pytest.mark.asyncio
class TestEntityPreserveDataKeys:
    async def test_update_keeps_the_latest_stored_value(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """The kept value is read in the update's transaction, so a write made after the caller's read wins."""
        await entity_repo.create_entity(
            workspace="workspace-1", entity_type="run", name="r1", data={"status": "running", "_owned": "creator"}
        )
        stale = await entity_repo.get_entity_by_name(workspace="workspace-1", entity_type="run", name="r1")
        assert stale is not None
        await entity_repo.update_entity_by_name(
            workspace="workspace-1", entity_type="run", name="r1", data={**stale.data, "_owned": "refreshed"}
        )

        updated = await entity_repo.update_entity_by_name(
            workspace="workspace-1",
            entity_type="run",
            name="r1",
            data={**stale.data, "status": "cancelling", "_owned": "forged"},
            preserve_data_keys=("_owned",),
        )

        assert updated.data == {"status": "cancelling", "_owned": "refreshed"}

    async def test_update_keeps_an_absent_key_absent(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        await entity_repo.create_entity(
            workspace="workspace-1", entity_type="run", name="r2", data={"status": "running"}
        )

        updated = await entity_repo.update_entity_by_name(
            workspace="workspace-1",
            entity_type="run",
            name="r2",
            data={"status": "cancelling", "_owned": "forged"},
            preserve_data_keys=("_owned",),
        )

        assert updated.data == {"status": "cancelling"}

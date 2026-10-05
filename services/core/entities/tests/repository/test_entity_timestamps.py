# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for entity timestamp precision, sorting, and filtering.

Verifies that timestamps are generated Python-side with microsecond precision
(not via SQLite's CURRENT_TIMESTAMP which only has second precision), and that
datetime comparisons work correctly for both storage and query filtering.
"""

from datetime import timedelta

import pytest
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository


@pytest.mark.asyncio
class TestTimestampPrecision:
    """Verify that timestamps have sub-second precision."""

    async def test_sequential_creates_have_distinct_timestamps(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """Sequential creates produce distinct, correctly-ordered timestamps."""
        entities = []
        for i in range(1, 4):
            entity = await entity_repo.create_entity(
                workspace="workspace-1",
                entity_type="config",
                name=f"config-{i}",
                data={"index": i},
            )
            entities.append(entity)

        timestamps = [e.created_at for e in entities]
        assert len(set(timestamps)) == 3, f"Expected 3 distinct timestamps, got {timestamps}"
        assert timestamps == sorted(timestamps)

    async def test_timestamps_have_sub_second_precision(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """At least one pair of sequential creates differs by less than a second."""
        entities = []
        for i in range(1, 4):
            entity = await entity_repo.create_entity(
                workspace="workspace-1",
                entity_type="config",
                name=f"config-{i}",
                data={},
            )
            entities.append(entity)

        timestamps = [e.created_at for e in entities]
        diffs = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
        assert any(d < 1.0 for d in diffs), f"Expected sub-second gaps, got {diffs}"

    async def test_updated_at_has_microsecond_precision(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """updated_at should have microsecond precision after an update."""
        entity = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="config",
            name="will-update",
            data={"version": 1},
        )
        created_at = entity.created_at

        updated = await entity_repo.update_entity(
            entity_id=entity.id,
            data={"version": 2},
        )

        assert updated.updated_at > created_at
        # The difference should be sub-second (both happen in the same test)
        diff = (updated.updated_at - created_at).total_seconds()
        assert diff < 1.0, f"Expected sub-second update gap, got {diff}s"


@pytest.mark.asyncio
class TestTimestampSorting:
    """Verify that sorting by timestamp fields works with microsecond precision."""

    async def test_sort_by_created_at_ascending(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Entities sorted by created_at asc are in creation order."""
        for i in range(1, 6):
            await entity_repo.create_entity(
                workspace="workspace-1",
                entity_type="config",
                name=f"config-{i}",
                data={},
            )

        entities, _ = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            sort="created_at",
        )

        names = [e.name for e in entities]
        assert names == ["config-1", "config-2", "config-3", "config-4", "config-5"]

    async def test_sort_by_created_at_descending(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Entities sorted by -created_at are in reverse creation order."""
        for i in range(1, 6):
            await entity_repo.create_entity(
                workspace="workspace-1",
                entity_type="config",
                name=f"config-{i}",
                data={},
            )

        entities, _ = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            sort="-created_at",
        )

        names = [e.name for e in entities]
        assert names == ["config-5", "config-4", "config-3", "config-2", "config-1"]

    async def test_sort_by_updated_at_reflects_update_order(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """After updating an entity, sorting by -updated_at should put it first."""
        e1 = await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name="first", data={"v": 1})
        await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name="second", data={"v": 1})
        await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name="third", data={"v": 1})

        # Update the first entity — it should now have the latest updated_at
        await entity_repo.update_entity(entity_id=e1.id, data={"v": 2})

        entities, _ = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            sort="-updated_at",
        )

        names = [e.name for e in entities]
        assert names[0] == "first", "Updated entity should sort first by -updated_at"
        assert names[-1] != "first", "Updated entity should not sort last"


@pytest.mark.asyncio
class TestTimestampFiltering:
    """Verify that datetime comparison operators work correctly against stored timestamps."""

    async def test_gte_filter_includes_matching_entities(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """$gte with a past date should include all entities."""
        from nhx.common.api.filter import ComparisonOperation, FilterOperator

        for i in range(3):
            await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name=f"e-{i}", data={})

        search = ComparisonOperation(field="created_at", operator=FilterOperator.GTE, value="2020-01-01T00:00:00")
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            filter_op=search,
        )
        assert total == 3

    async def test_lt_filter_with_future_date_includes_all(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """$lt with a future date should include all entities."""
        from nhx.common.api.filter import ComparisonOperation, FilterOperator

        for i in range(3):
            await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name=f"e-{i}", data={})

        search = ComparisonOperation(field="created_at", operator=FilterOperator.LT, value="2099-12-31T23:59:59")
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            filter_op=search,
        )
        assert total == 3

    async def test_lt_filter_with_past_date_excludes_all(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """$lt with a past date should exclude all entities."""
        from nhx.common.api.filter import ComparisonOperation, FilterOperator

        await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name="e-1", data={})

        search = ComparisonOperation(field="created_at", operator=FilterOperator.LT, value="2000-01-01T00:00:00")
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            filter_op=search,
        )
        assert total == 0

    async def test_range_filter_brackets_all_entities(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """A date range from past to future should include all entities."""
        from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation

        for i in range(3):
            await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name=f"e-{i}", data={})

        search = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(field="created_at", operator=FilterOperator.GTE, value="2020-01-01T00:00:00"),
                ComparisonOperation(field="created_at", operator=FilterOperator.LT, value="2099-12-31T23:59:59"),
            ],
        )
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            filter_op=search,
        )
        assert total == 3

    async def test_filter_by_actual_created_at_value(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Filter using the actual created_at value from an entity to split results.

        This is the real-world scenario: capture a timestamp between creates,
        then use it to partition entities. Tests that the bind_processor format
        matches the stored format for correct comparison.
        """
        from nhx.common.api.filter import ComparisonOperation, FilterOperator

        e1 = await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name="before", data={})

        # Use e1's actual created_at + small delta as the boundary
        boundary = (e1.created_at + timedelta(microseconds=1)).isoformat()

        await entity_repo.create_entity(workspace="workspace-1", entity_type="config", name="after", data={})

        # Everything >= boundary should exclude e1, include e2
        search = ComparisonOperation(field="created_at", operator=FilterOperator.GTE, value=boundary)
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            filter_op=search,
        )
        assert total == 1
        assert entities[0].name == "after"

    async def test_updated_at_filter_after_update(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Filter by updated_at to find recently-updated entities.

        Creates two entities, updates the first, then filters for entities
        whose updated_at is strictly greater than the second entity's updated_at.
        Only the first (re-updated) entity should match.
        """
        from nhx.common.api.filter import ComparisonOperation, FilterOperator

        e1 = await entity_repo.create_entity(
            workspace="workspace-1", entity_type="config", name="will-update", data={"v": 1}
        )
        e2 = await entity_repo.create_entity(
            workspace="workspace-1", entity_type="config", name="wont-update", data={"v": 1}
        )

        # Update e1 — its updated_at should now be after e2's updated_at
        await entity_repo.update_entity(entity_id=e1.id, data={"v": 2})

        # Use e2's updated_at as the boundary
        search = ComparisonOperation(field="updated_at", operator=FilterOperator.GT, value=e2.updated_at.isoformat())
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="config",
            filter_op=search,
        )
        assert total == 1
        assert entities[0].name == "will-update"

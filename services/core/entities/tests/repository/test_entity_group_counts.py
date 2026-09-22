# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for grouped entity counts."""

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.sqlalchemy import entity as entity_repository

pytestmark = pytest.mark.asyncio


async def test_counts_filtered_entities_grouped_by_direct_string_data_field(
    entity_repo: SQLAlchemyEntityRepository, setup_workspaces
):
    """Count live experiment groups grouped by string insight_id values."""
    entities = (
        ("live-a-1", {"insight_id": "insight-a", "is_deleted": False}),
        ("live-a-2", {"insight_id": "insight-a", "is_deleted": False}),
        ("live-b", {"insight_id": "insight-b", "is_deleted": False}),
        ("escaped", {"insight_id": 'insight-"quoted"\\path', "is_deleted": False}),
        ("deleted", {"insight_id": "insight-a", "is_deleted": True}),
        ("missing", {"is_deleted": False}),
        ("null", {"insight_id": None, "is_deleted": False}),
        ("boolean", {"insight_id": True, "is_deleted": False}),
        ("numeric", {"insight_id": 1, "is_deleted": False}),
    )
    for name, data in entities:
        await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="experiment_group",
            name=name,
            data=data,
        )

    filter_op = ComparisonOperation(field="data.is_deleted", operator=FilterOperator.EQ, value=False)

    counts = await entity_repo.count_entities_by(
        workspace="workspace-1",
        entity_type="experiment_group",
        group_by="data.insight_id",
        filter_op=filter_op,
    )

    assert counts == {'insight-"quoted"\\path': 1, "insight-a": 2, "insight-b": 1}


@pytest.mark.parametrize("field", ["name", "data.nested.value", "data.", "data.not-valid"])
async def test_rejects_unsupported_group_fields(entity_repo: SQLAlchemyEntityRepository, setup_workspaces, field: str):
    with pytest.raises(ValueError, match="direct string data field"):
        await entity_repo.count_entities_by(
            workspace="workspace-1",
            entity_type="experiment_group",
            group_by=field,
        )


async def test_rejects_group_counts_over_limit(
    entity_repo: SQLAlchemyEntityRepository, setup_workspaces, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(entity_repository, "MAX_GROUP_COUNT_ROWS", 2)
    for index in range(3):
        await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="experiment_group",
            name=f"group-{index}",
            data={"value": f"group-{index}"},
        )

    with pytest.raises(ValueError, match="more than 2 distinct values"):
        await entity_repo.count_entities_by(
            workspace="workspace-1",
            entity_type="experiment_group",
            group_by="data.value",
        )

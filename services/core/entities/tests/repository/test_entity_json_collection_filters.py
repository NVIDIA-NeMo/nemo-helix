# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the array and object filter operators through the entity repository."""

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository

pytestmark = pytest.mark.asyncio

ENTITIES = {
    "alice-smoke": {"metadata": [{"key": "owner", "value": "alice"}, {"key": "suite", "value": "smoke"}]},
    "bob-alice": {"metadata": [{"key": "owner", "value": "bob"}, {"key": "suite", "value": "alice"}]},
    "object-not-array": {"metadata": {"key": "owner", "value": "alice"}},
    "no-metadata": {},
}


def _owner_is(value: str) -> ComparisonOperation:
    return ComparisonOperation(
        field="data.metadata", operator=FilterOperator.ELEM_MATCH, value={"key": "owner", "value": value}
    )


@pytest.mark.parametrize(
    "filter_op,expected",
    [
        (_owner_is("alice"), {"alice-smoke"}),
        (
            LogicalOperation(operator=FilterOperator.OR, operations=[_owner_is("alice"), _owner_is("bob")]),
            {"alice-smoke", "bob-alice"},
        ),
        (
            LogicalOperation(operator=FilterOperator.NOT, operations=[_owner_is("alice")]),
            {"bob-alice", "object-not-array", "no-metadata"},
        ),
    ],
)
async def test_list_entities_matches_on_one_array_element(
    entity_repo: SQLAlchemyEntityRepository, setup_workspaces, filter_op, expected
):
    for name, data in ENTITIES.items():
        await entity_repo.create_entity(workspace="workspace-1", entity_type="annotated", name=name, data=data)

    entities, total = await entity_repo.list_entities(
        workspace="workspace-1", entity_type="annotated", filter_op=filter_op
    )

    assert {entity.name for entity in entities} == expected
    assert total == len(expected)


REFERENCING = {
    "pins-a": {"tasks": ["workspace-1/task-a#d1", "workspace-1/task-b#d2"], "tags": {"latest": 2, "v1.2": 1}},
    "near-misses": {"tasks": ["other-workspace-1/task-a#d1", "workspace-1/taskXa#d1"], "tags": {"v1": 1}},
    "nothing": {},
}


@pytest.mark.parametrize(
    "filter_op,expected",
    [
        (
            ComparisonOperation(
                field="data.tasks", operator=FilterOperator.CONTAINS_PREFIX, value="workspace-1/task_a#"
            ),
            set(),
        ),
        (
            ComparisonOperation(
                field="data.tasks", operator=FilterOperator.CONTAINS_PREFIX, value="workspace-1/task-a#"
            ),
            {"pins-a"},
        ),
        (
            LogicalOperation(
                operator=FilterOperator.NOT,
                operations=[
                    ComparisonOperation(
                        field="data.tasks", operator=FilterOperator.CONTAINS_PREFIX, value="workspace-1/task-a#"
                    )
                ],
            ),
            {"near-misses", "nothing"},
        ),
        (ComparisonOperation(field="data.tags", operator=FilterOperator.HAS_KEY, value="v1.2"), {"pins-a"}),
        (ComparisonOperation(field="data.tags", operator=FilterOperator.HAS_KEY, value="v1"), {"near-misses"}),
    ],
)
async def test_list_entities_matches_array_prefixes_and_object_keys(
    entity_repo: SQLAlchemyEntityRepository, setup_workspaces, filter_op, expected
):
    for name, data in REFERENCING.items():
        await entity_repo.create_entity(workspace="workspace-1", entity_type="referencing", name=name, data=data)

    entities, total = await entity_repo.list_entities(
        workspace="workspace-1", entity_type="referencing", filter_op=filter_op
    )

    assert {entity.name for entity in entities} == expected
    assert total == len(expected)

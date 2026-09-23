# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filtering child entities by parent and by a custom (``data.*``) field.

``list_entities`` takes no ``parent`` argument, which reads as "you cannot query children". These
tests establish that you can: ``parent`` is a mapped column, so it is reachable through an ordinary
``filter_op`` — no API change needed. Combined with ``data.*`` JSON extraction, that covers
"find the child of this parent whose custom field equals X", which is what a content-addressed
revision lookup needs.
"""

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository

pytestmark = pytest.mark.asyncio

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


async def _seed(entity_repo: SQLAlchemyEntityRepository):
    """Two parents, each with two revision children; digests repeat across parents on purpose."""
    parents = {}
    for parent_name in ("task-one", "task-two"):
        parent = await entity_repo.create_entity(workspace="workspace-1", entity_type="task", name=parent_name, data={})
        parents[parent_name] = parent
        for ordinal, digest in ((1, _DIGEST_A), (2, _DIGEST_B)):
            await entity_repo.create_entity(
                workspace="workspace-1",
                entity_type="task_revision",
                name=f"rev.{ordinal}",
                parent=parent.id,
                data={"content_hash": digest, "revision": ordinal},
            )
    return parents


async def test_children_can_be_filtered_by_parent(entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
    """``parent`` is a real column, so it filters like any other — despite ``list_entities``
    exposing no dedicated argument for it."""
    parents = await _seed(entity_repo)
    rows, total = await entity_repo.list_entities(
        workspace="workspace-1",
        entity_type="task_revision",
        filter_op=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=parents["task-one"].id),
    )
    assert total == 2
    assert {row.name for row in rows} == {"rev.1", "rev.2"}


async def test_children_can_be_filtered_by_custom_data_field(entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
    """``data.*`` paths are JSON-extracted by the filter translator, so a custom field on a stored
    entity is queryable without promoting it to a column."""
    await _seed(entity_repo)
    rows, total = await entity_repo.list_entities(
        workspace="workspace-1",
        entity_type="task_revision",
        filter_op=ComparisonOperation(field="data.content_hash", operator=FilterOperator.EQ, value=_DIGEST_A),
    )
    assert total == 2, "the same content digest exists under both parents"
    assert {row.data["content_hash"] for row in rows} == {_DIGEST_A}


async def test_parent_and_custom_field_compose(entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
    """The query a digest-pinned ref actually needs: this parent's child with this digest.

    Parent scoping is what disambiguates — identical content under two different tasks yields the
    same digest, so filtering on the digest alone returns both.
    """
    parents = await _seed(entity_repo)
    rows, total = await entity_repo.list_entities(
        workspace="workspace-1",
        entity_type="task_revision",
        filter_op=LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=parents["task-two"].id),
                ComparisonOperation(field="data.content_hash", operator=FilterOperator.EQ, value=_DIGEST_A),
            ],
        ),
    )
    assert total == 1
    assert rows[0].parent == parents["task-two"].id
    assert rows[0].data["content_hash"] == _DIGEST_A


async def test_unknown_digest_returns_nothing(entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
    await _seed(entity_repo)
    _, total = await entity_repo.list_entities(
        workspace="workspace-1",
        entity_type="task_revision",
        filter_op=ComparisonOperation(field="data.content_hash", operator=FilterOperator.EQ, value="c" * 64),
    )
    assert total == 0


async def test_deleting_a_parent_removes_its_children(entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
    """The parent FK is ``ondelete="CASCADE"``, so deleting a task takes its revisions with it —
    no orphaned children left addressable by a stale parent id."""
    parents = await _seed(entity_repo)
    await entity_repo.delete_entity(entity_id=parents["task-one"].id)

    _, orphaned = await entity_repo.list_entities(
        workspace="workspace-1",
        entity_type="task_revision",
        filter_op=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=parents["task-one"].id),
    )
    assert orphaned == 0

    _, survivors = await entity_repo.list_entities(
        workspace="workspace-1",
        entity_type="task_revision",
        filter_op=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=parents["task-two"].id),
    )
    assert survivors == 2, "the other task's revisions must be untouched"

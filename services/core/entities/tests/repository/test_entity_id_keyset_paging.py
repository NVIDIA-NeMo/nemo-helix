# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyset paging by entity ID (sort by ``id``, filter ``id`` > the last one read), as the auth service's scans use."""

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository

pytestmark = pytest.mark.asyncio


def _owned_by(principal: str) -> ComparisonOperation:
    return ComparisonOperation(field="data.principal", operator=FilterOperator.EQ, value=principal)


async def test_keyset_scan_by_id_reads_every_row_when_a_read_row_is_deleted(
    entity_repo: SQLAlchemyEntityRepository, setup_workspaces
):
    for index in range(5):
        await entity_repo.create_entity(
            workspace="workspace-1", entity_type="binding", name=f"binding-{index}", data={"principal": "svc"}
        )
    await entity_repo.create_entity(
        workspace="workspace-1", entity_type="binding", name="other", data={"principal": "someone-else"}
    )

    seen: list[str] = []
    after_id = None
    while True:
        filter_op = (
            _owned_by("svc")
            if after_id is None
            else LogicalOperation(
                operator=FilterOperator.AND,
                operations=[
                    _owned_by("svc"),
                    ComparisonOperation(field="id", operator=FilterOperator.GT, value=after_id),
                ],
            )
        )
        page, total = await entity_repo.list_entities(
            workspace="workspace-1", entity_type="binding", filter_op=filter_op, sort="id", page=1, page_size=2
        )
        assert [entity.id for entity in page] == sorted(entity.id for entity in page)
        if after_id is None:
            # Delete a row already read: offset paging would now skip an unread row.
            await entity_repo.delete_entity(entity_id=page[0].id)
        seen.extend(entity.name for entity in page)
        if total <= len(page):
            break
        after_id = page[-1].id

    assert sorted(seen) == [f"binding-{index}" for index in range(5)]

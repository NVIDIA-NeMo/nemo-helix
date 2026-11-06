# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise both SQL OFFSET sites, including the exact signed-int64 boundary."""

from functools import partial
from unittest.mock import AsyncMock

import pytest
from nhx.common.entities.client import EntityValidationError
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entity", "workspace"])
async def test_sql_offset_bounds(entity_repo, workspace_repo, kind):
    await workspace_repo.create_workspace(name="pagination")
    if kind == "entity":
        await entity_repo.create_entity(workspace="pagination", entity_type="widget", name="one", data={})
        list_rows = partial(entity_repo.list_entities, workspace="pagination", entity_type="widget")
    else:
        list_rows = workspace_repo.list_workspaces

    rows, total = await list_rows(page=1, page_size=1)
    assert len(rows) == total == 1
    rows, total = await list_rows(page=2, page_size=1)
    assert rows == []
    assert total == 1

    # This page is valid at size 3 but cannot be represented at size 100.
    huge_page = 92233720368547760
    rows, total = await list_rows(page=huge_page, page_size=3)
    assert rows == []
    assert total == 1

    # 2**63-1 is divisible by 7: page*size would incorrectly reject this.
    last_page = (2**63 - 1) // 7 + 1
    rows, total = await list_rows(page=last_page, page_size=7)
    assert rows == []
    assert total == 1

    for page, size, detail in [(last_page + 1, 7, "offset"), (huge_page, 100, "offset"), (1, 2**63, "LIMIT")]:
        session = AsyncMock(spec=AsyncSession)
        with pytest.raises(EntityValidationError, match=detail):
            await list_rows(page=page, page_size=size, session=session)
        session.execute.assert_not_awaited()

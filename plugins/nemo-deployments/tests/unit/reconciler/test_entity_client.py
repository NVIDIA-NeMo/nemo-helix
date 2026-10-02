# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from nemo_deployments_plugin.entities import Deployment
from nemo_deployments_plugin.reconciler.entity_client import list_all_pages
from nemo_helix_plugin.entity_client import NemoPaginationInfo
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator


def _page(items: list[Deployment], *, page: int, total_pages: int) -> AsyncMock:
    resp = AsyncMock()
    resp.data = items
    resp.pagination = NemoPaginationInfo(
        page=page,
        page_size=100,
        current_page_size=len(items),
        total_pages=total_pages,
        total_results=len(items) if total_pages == 1 else 150,
    )
    return resp


@pytest.mark.asyncio
async def test_list_all_pages_fetches_multiple_pages() -> None:
    entities = AsyncMock()
    entities.list.side_effect = [
        _page([Deployment(name="a", workspace="w", deployment_config="c")], page=1, total_pages=2),
        _page([Deployment(name="b", workspace="w", deployment_config="c")], page=2, total_pages=2),
    ]
    result = await list_all_pages(
        entities,
        Deployment,
        filter_operation=ComparisonOperation(
            operator=FilterOperator.IN,
            field="data.status",
            value=["PENDING"],
        ),
    )
    assert [d.name for d in result] == ["a", "b"]
    assert entities.list.await_count == 2

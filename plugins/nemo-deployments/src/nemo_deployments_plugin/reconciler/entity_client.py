# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity client helpers for paginated listing."""

from __future__ import annotations

from typing import TypeVar

from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.entity_client import NemoEntitiesClient
from nemo_helix_plugin.filter_ops import ComparisonOperation

EntityT = TypeVar("EntityT", bound=NemoEntity)

DEFAULT_LIST_PAGE_SIZE = 100


async def list_all_pages(
    entities: NemoEntitiesClient,
    entity_type: type[EntityT],
    *,
    workspace: str = "-",
    page_size: int = DEFAULT_LIST_PAGE_SIZE,
    filter_operation: ComparisonOperation | None = None,
) -> list[EntityT]:
    """Fetch all pages for a cross-workspace entity list query."""
    collected: list[EntityT] = []
    page = 1
    while True:
        result = await entities.list(
            entity_type,
            workspace=workspace,
            page=page,
            page_size=page_size,
            filter_operation=filter_operation,
        )
        collected.extend(result.data)
        pagination = result.pagination
        if pagination is None or page >= pagination.total_pages:
            break
        page += 1
    return collected

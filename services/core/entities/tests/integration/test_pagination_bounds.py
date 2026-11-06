# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL-backed HTTP pagination and typed client boundary behavior."""

import pytest
from httpx import AsyncClient
from nemo_helix_plugin.client.errors import UnprocessableEntityError
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity import NemoEntity
from nhx.common.entities.client import EntityClient


class PaginationWidget(NemoEntity, entity_type="pagination_widget"):
    pass


@pytest.mark.integration
@pytest.mark.parametrize(
    "path,page,page_size,status",
    [
        ("/apis/entities/v2/workspaces", (2**63 - 1) // 7 + 1, 7, 200),
        ("/apis/entities/v2/workspaces", (2**63 - 1) // 7 + 2, 7, 422),
        ("/apis/entities/v2/workspaces/default/projects", 92233720368547760, 100, 422),
    ],
)
async def test_collection_offset_bounds(client: AsyncClient, ctx, path, page, page_size, status):
    response = await client.get(path, params={"page": page, "page_size": page_size})
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["data"] == []
        assert response.json()["pagination"]["page"] == page
    else:
        assert "offset" in response.json()["detail"]


@pytest.mark.integration
async def test_entity_client_preserves_size_dependent_bounds(client: AsyncClient, ctx):
    entities_client = EntityClient(AsyncEntitiesClient(base_url="http://test", http_client=client))
    huge_page = 92233720368547760
    result = await entities_client.list(PaginationWidget, workspace="default", page=huge_page, page_size=3)
    assert result.data == []
    assert result.pagination.page == huge_page
    assert result.pagination.page_size == 3

    with pytest.raises(UnprocessableEntityError) as exc:
        await entities_client.list(PaginationWidget, workspace="default", page=huge_page, page_size=100)
    assert exc.value.status_code == 422
    assert "offset" in exc.value.detail

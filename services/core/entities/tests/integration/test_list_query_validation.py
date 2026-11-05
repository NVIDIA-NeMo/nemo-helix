# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate list query failures against the real SQL-backed HTTP routes."""

import pytest
from httpx import AsyncClient

PATHS = [
    "/apis/entities/v2/workspaces",
    "/apis/entities/v2/workspaces/default/projects",
    "/apis/entities/v2/workspaces/default/entities/model",
]


@pytest.mark.integration
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "field", ["nosuchfield", "metadata", "registry", "__class__", "to_pydantic", "data.", "data..key"]
)
async def test_unknown_or_noncolumn_filter(client: AsyncClient, ctx, path: str, field: str):
    response = await client.get(path, params={f"filter[{field}]": "1"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == f"Invalid filter field '{field}'"


@pytest.mark.integration
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "params,status",
    [
        ({"page": "0"}, 422),
        ({"page_size": str(2**63)}, 422),
        ({"sort": "--name"}, None),
        ({"filter[created_at][$gt]": "notadate"}, 400),
        ({"page": "1", "page_size": "10", "sort": "name", "filter[name]": "default"}, 200),
        ({"filter[created_at][$gt]": "2020-01-01T00:00:00Z"}, 200),
    ],
)
async def test_list_query_validation(client: AsyncClient, ctx, path: str, params: dict, status: int | None):
    response = await client.get(path, params=params)
    if status is None:
        status = 400 if path.endswith("/entities/model") else 422
    assert response.status_code == status, response.text
    if status == 400:
        assert "detail" in response.json()

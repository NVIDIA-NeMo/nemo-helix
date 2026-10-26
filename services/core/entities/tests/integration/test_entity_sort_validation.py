# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invalid sort attributes return client errors through the entities HTTP API."""

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.parametrize("field", ["metadata", "registry", "__class__", "to_pydantic", "data", "bogus"])
@pytest.mark.parametrize("prefix", ["", "-"])
async def test_invalid_entity_sort_returns_bad_request(client: AsyncClient, ctx, field: str, prefix: str):
    response = await client.get(
        "/apis/entities/v2/workspaces/default/entities/model",
        params={"sort": prefix + field},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == f"Invalid sort field {field}"


@pytest.mark.integration
@pytest.mark.parametrize("sort", ["", "-", "--name", "data.", "data..k", "-data.k."])
async def test_malformed_entity_sort_returns_bad_request(client: AsyncClient, ctx, sort: str):
    response = await client.get("/apis/entities/v2/workspaces/default/entities/model", params={"sort": sort})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Invalid sort field")

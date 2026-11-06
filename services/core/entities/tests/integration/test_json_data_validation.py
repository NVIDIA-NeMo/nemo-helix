# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Informative client errors for unsupported root filters and unsafe JSON writes."""

import pytest
from httpx import AsyncClient

ENTITIES = "/apis/entities/v2/workspaces/default/entities/sample"


@pytest.mark.integration
async def test_root_json_filter_returns_informative_400(client: AsyncClient, ctx):
    response = await client.get(ENTITIES, params={"filter[data]": "1"})
    assert response.status_code == 400, response.text
    assert "use dotted data paths" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.parametrize("update", [False, True], ids=["create", "update"])
async def test_nul_write_returns_sanitized_400_without_changing_rows(client: AsyncClient, ctx, update):
    seed = await client.post(ENTITIES, json={"name": "safe", "data": {"key": "safe"}})
    assert seed.status_code == 201, seed.text
    data = {"nested": [{"secret\x00key": "secret\x00value"}]}
    if update:
        response = await client.put(ENTITIES + "/safe", json={"new_name": "changed", "data": data})
    else:
        response = await client.post(ENTITIES, json={"name": "bad", "data": data})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Entity data must not contain the NUL character (U+0000) in JSON strings or keys"
    )
    listed = await client.get(ENTITIES, params={"sort": "data.key"})
    assert listed.status_code == 200, listed.text
    assert [(row["name"], row["data"]) for row in listed.json()["data"]] == [("safe", {"key": "safe"})]

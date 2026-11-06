# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ID routing never falls back to a name, another type or another workspace."""

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize("method", ["DELETE"])
@pytest.mark.parametrize("scope", ["missing-id", "wrong-type", "wrong-workspace", "wrong-parent"])
async def test_id_mutation_does_not_retarget(client: AsyncClient, repos, ctx, method: str, scope: str) -> None:
    await repos["workspace"].create_workspace(name="other")
    created = (
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/widget",
            json={"name": "missing-object", "data": {"colour": "red"}},
        )
    ).json()
    target_id = "missing-object" if scope == "missing-id" else created["id"]
    workspace = "other" if scope == "wrong-workspace" else "default"
    entity_type = "other_type" if scope == "wrong-type" else "widget"
    response = await client.request(
        method,
        f"/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/by-id/{target_id}",
        json={"data": {"colour": "blue"}} if method == "PUT" else None,
        params={"parent": "different-parent"} if scope == "wrong-parent" else None,
    )
    assert response.status_code == 404, response.text
    survivor = (await client.get(f"/apis/entities/v2/entities/{created['id']}")).json()
    assert survivor["data"] == {"colour": "red"}


@pytest.mark.parametrize("method", ["DELETE"])
async def test_id_mutation_enforces_version(client: AsyncClient, ctx, method: str) -> None:
    created = (
        await client.post(
            "/apis/entities/v2/workspaces/default/entities/widget",
            json={"name": "versioned", "data": {"colour": "red"}},
        )
    ).json()
    url = f"/apis/entities/v2/workspaces/default/entities/widget/by-id/{created['id']}"
    response = await client.request(
        method,
        url,
        json={"data": {"colour": "blue"}, "expected_db_version": created["db_version"] + 1}
        if method == "PUT"
        else None,
        params={"expected_db_version": created["db_version"] + 1} if method == "DELETE" else None,
    )
    assert response.status_code == 409, response.text
    survivor = (await client.get(f"/apis/entities/v2/entities/{created['id']}")).json()
    assert survivor["data"] == {"colour": "red"}

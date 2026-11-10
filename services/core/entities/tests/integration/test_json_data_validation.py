# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Informative client errors for unsupported root filters and unsafe JSON writes."""

import pytest
from httpx import ASGITransport, AsyncClient
from nhx.core.entities.api.dependencies import dep_entity_repository_with_session
from nhx.core.entities.service import EntitiesService

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


@pytest.mark.integration
@pytest.mark.parametrize("update", [False, True], ids=["create", "update"])
async def test_project_nul_description_returns_400_without_changing_rows(client: AsyncClient, ctx, update):
    """Project routes must map repository validation just like generic entity writes."""
    path = "/apis/entities/v2/workspaces/default/projects"
    seed = await client.post(path, json={"name": "safe", "description": "original"})
    assert seed.status_code == 201, seed.text
    if update:
        response = await client.put(path + "/safe", json={"description": "secret\x00description"})
    else:
        response = await client.post(path, json={"name": "bad", "description": "secret\x00description"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Entity data must not contain the NUL character (U+0000) in JSON strings or keys"
    )
    listed = await client.get(path)
    assert listed.status_code == 200, listed.text
    assert [(row["name"], row["description"]) for row in listed.json()["data"]] == [("safe", "original")]


@pytest.mark.integration
async def test_member_nul_principal_returns_400_without_creating_binding(client: AsyncClient, ctx):
    """Member creation must reject an unsafe role-binding payload before writing it."""
    path = "/apis/entities/v2/workspaces/default/members"
    response = await client.post(
        path,
        params={"wait_role_propagation": "false"},
        json={"principal": "secret\x00principal", "roles": ["Editor"]},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Entity data must not contain the NUL character (U+0000) in JSON strings or keys"
    )
    listed = await client.get(path)
    assert listed.status_code == 200, listed.text
    assert listed.json()["data"] == []


@pytest.mark.integration
async def test_service_factory_registers_entity_data_error_boundary(repos):
    """The real service app must handle the error, not only the standalone test app."""
    await repos["workspace"].create_workspace(name="service-json")
    app = EntitiesService().create_app()
    app.dependency_overrides[dep_entity_repository_with_session] = lambda: repos["entity"]
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v2/workspaces/service-json/projects", json={"name": "bad", "description": "secret\x00description"}
        )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == (
        "Entity data must not contain the NUL character (U+0000) in JSON strings or keys"
    )
    rows, total = await repos["entity"].list_entities(workspace="service-json", entity_type="project")
    assert rows == [] and total == 0

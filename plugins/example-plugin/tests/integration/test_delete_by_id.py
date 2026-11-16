# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_example_plugin.client import AsyncExampleClient, ExampleClient
from nemo_example_plugin.middleware_service import _get_entity_client as middleware_entities
from nemo_example_plugin.service import ExampleService
from nemo_example_plugin.service import _get_entity_client as item_entities
from nemo_helix_plugin.client.errors import NotFoundError
from nhx.testing import ClientContext, TestClientHttpAdapter, create_test_client

pytestmark = pytest.mark.integration


@pytest.fixture
def api() -> Iterator[TestClient]:
    with create_test_client(client_type=ClientContext) as ctx:
        app = FastAPI()
        for spec in ExampleService().get_routers():
            app.include_router(spec.router, prefix=f"/apis/example{spec.prefix}")
        app.dependency_overrides[item_entities] = lambda: ctx.entity_client
        app.dependency_overrides[middleware_entities] = lambda: ctx.entity_client
        with TestClient(app) as client:
            yield client


@pytest.fixture(params=["items", "middleware-configs"])
def resource(request: pytest.FixtureRequest) -> str:
    return request.param


def create(api: TestClient, resource: str, name: str = "reusable") -> str:
    body = {"name": name}
    if resource == "items":
        body["title"] = "Example"
    response = api.post(f"/apis/example/v2/workspaces/default/{resource}", json=body)
    assert response.status_code == 201, response.text
    entity_id = response.json()["id"]
    assert isinstance(entity_id, str)
    return entity_id


def test_id_delete_never_selects_a_same_name_replacement(api: TestClient, resource: str) -> None:
    original = create(api, resource)
    url = f"/apis/example/v2/workspaces/default/{resource}/by-id/{original}"
    assert api.delete(url).status_code == 204
    replacement = create(api, resource)
    assert replacement != original
    assert api.delete(url).status_code == 404
    assert api.get(f"/apis/example/v2/workspaces/default/{resource}/reusable").json()["id"] == replacement


def test_id_delete_checks_workspace_and_type(api: TestClient, resource: str) -> None:
    original = create(api, resource)
    wrong_workspace = api.delete(f"/apis/example/v2/workspaces/system/{resource}/by-id/{original}")
    assert wrong_workspace.status_code == 404
    other = "middleware-configs" if resource == "items" else "items"
    assert api.delete(f"/apis/example/v2/workspaces/default/{other}/by-id/{original}").status_code == 404
    assert api.get(f"/apis/example/v2/workspaces/default/{resource}/reusable").status_code == 200


def test_id_delete_rejects_a_stale_version(api: TestClient, resource: str) -> None:
    original = create(api, resource)
    update = {"title": "Updated"} if resource == "items" else {"block_message": "Updated"}
    base = f"/apis/example/v2/workspaces/default/{resource}"
    assert api.patch(f"{base}/reusable", json=update).status_code == 200
    url = f"{base}/by-id/{original}"
    assert api.delete(url, params={"expected_db_version": 1}).status_code == 409
    assert api.get(f"{base}/reusable").status_code == 200
    assert api.delete(url, params={"expected_db_version": 2}).status_code == 204


def test_sync_client_exposes_both_id_deletes(api: TestClient) -> None:
    item = create(api, "items")
    config = create(api, "middleware-configs")
    with ExampleClient(base_url="http://testserver", http_client=TestClientHttpAdapter(api)) as client:
        client.delete_item_by_id(workspace="default", entity_id=item, query_params={"expected_db_version": 1})
        client.delete_middleware_config_by_id(workspace="default", entity_id=config)
        with pytest.raises(NotFoundError):
            client.delete_item_by_id(workspace="default", entity_id=item)


async def test_async_client_sends_id_paths_and_version() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = AsyncExampleClient(base_url="http://testserver", http_client=http)
        await client.delete_item_by_id(
            workspace="default", entity_id="original", query_params={"expected_db_version": 7}
        )
        await client.delete_middleware_config_by_id(workspace="default", entity_id="config-original")
    assert [request.url.path for request in requests] == [
        "/apis/example/v2/workspaces/default/items/by-id/original",
        "/apis/example/v2/workspaces/default/middleware-configs/by-id/config-original",
    ]
    assert requests[0].url.params["expected_db_version"] == "7"
    assert all(request.method == "DELETE" for request in requests)


def test_id_delete_contract_is_in_openapi(api: TestClient, resource: str) -> None:
    schema = api.get("/openapi.json").json()
    operation = schema["paths"][f"/apis/example/v2/workspaces/{{workspace}}/{resource}/by-id/{{entity_id}}"]["delete"]
    assert {"204", "404", "409"} <= operation["responses"].keys()
    version = next(param for param in operation["parameters"] if param["name"] == "expected_db_version")
    assert version["in"] == "query" and version["required"] is False
    assert (
        api.delete(
            f"/apis/example/v2/workspaces/default/{resource}/by-id/unused", params={"expected_db_version": 0}
        ).status_code
        == 422
    )

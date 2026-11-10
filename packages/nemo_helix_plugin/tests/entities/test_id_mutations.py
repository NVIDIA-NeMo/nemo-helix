# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sync and async typed clients retain identity through actual HTTP requests."""

import json

import httpx
import pytest
from nemo_helix_plugin.entities import EntityBase, EntityClient, SyncEntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient, EntitiesClient


class Widget(EntityBase):
    colour: str


class MutationTransport:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "widget-original"})
        return httpx.Response(
            200,
            json={
                "id": "widget-original",
                "entity_type": "widget",
                "workspace": "default",
                "name": "reusable-name",
                "parent": None,
                "db_version": 3,
                "created_at": "2026-11-05T00:00:00Z",
                "updated_at": "2026-11-05T00:00:00Z",
                "data": {"colour": "red"},
            },
        )

    def verify(self) -> None:
        writes = [request for request in self.requests if request.method != "GET"]
        assert [request.method for request in writes] == ["PUT", "DELETE"]
        assert all(
            request.url.path == "/apis/entities/v2/workspaces/default/entities/widget/by-id/widget-original"
            for request in writes
        )
        assert json.loads(writes[0].content)["expected_db_version"] == 3
        assert writes[1].url.params["expected_db_version"] == "3"
        assert all("reusable-name" not in request.url.path for request in self.requests)


def test_sync_mutations_keep_the_selected_id() -> None:
    handler = MutationTransport()
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        raw = EntitiesClient(base_url="http://test", http_client=http)
        store = SyncEntityClient(raw)
        entity = store.get_by_id(Widget, "widget-original")
        entity.colour = "blue"
        store.update(entity)
        store.delete_by_id(Widget, entity.id)
    handler.verify()


@pytest.mark.asyncio
async def test_async_mutations_keep_the_selected_id() -> None:
    handler = MutationTransport()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        raw = AsyncEntitiesClient(base_url="http://test", http_client=http)
        store = EntityClient(raw)
        entity = await store.get_by_id(Widget, "widget-original")
        entity.colour = "blue"
        await store.update(entity)
        await store.delete_by_id(Widget, entity.id)
    handler.verify()

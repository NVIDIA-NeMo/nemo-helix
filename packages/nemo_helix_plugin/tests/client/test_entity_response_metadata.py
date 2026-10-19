# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed clients keep the store-managed metadata of entities they parse from responses."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.endpoint import get
from nemo_helix_plugin.client.types import Paginated
from nemo_helix_plugin.entity import NemoEntity
from pydantic import BaseModel

BASE = "http://test:8000"


class MetadataWidget(NemoEntity, entity_type="response_metadata_widget"):
    color: str = ""


class WidgetEnvelope(BaseModel):
    """A response model that nests an entity, like ``AnalysisRunResponse.run``."""

    widget: MetadataWidget
    status: str


@get("/apis/test/v2/workspaces/{workspace}/widgets/{name}")
def GET_WIDGET(*, workspace: str | None = None, name: str) -> MetadataWidget:
    raise NotImplementedError


@get("/apis/test/v2/workspaces/{workspace}/widgets")
def LIST_WIDGETS(*, workspace: str | None = None) -> Paginated[MetadataWidget]:
    raise NotImplementedError


@get("/apis/test/v2/workspaces/{workspace}/widgets/{name}/envelope")
def GET_ENVELOPE(*, workspace: str | None = None, name: str) -> WidgetEnvelope:
    raise NotImplementedError


WIDGET = {
    "name": "blue",
    "workspace": "default",
    "color": "blue",
    "id": "widget-123",
    "parent": "parent-1",
    "created_at": "2026-09-01T12:00:00Z",
    "created_by": "alice",
    "updated_at": "2026-09-02T12:00:00Z",
    "updated_by": "bob",
    "db_version": 4,
}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/envelope"):
        return httpx.Response(200, json={"widget": WIDGET, "status": "ok"})
    if request.url.path.endswith("/widgets"):
        return httpx.Response(
            200,
            json={
                "data": [WIDGET],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )
    return httpx.Response(200, json=WIDGET)


def _assert_metadata(widget: MetadataWidget) -> None:
    assert widget.id == "widget-123"
    assert widget.parent == "parent-1"
    assert widget.created_at == datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    assert widget.created_by == "alice"
    assert widget.updated_at == datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    assert widget.updated_by == "bob"
    assert widget.db_version == 4


@pytest.fixture
def client() -> NemoClient:
    return NemoClient(
        base_url=BASE, workspace="default", http_client=httpx.Client(transport=httpx.MockTransport(_handler))
    )


def test_single_entity_response_keeps_metadata(client: NemoClient) -> None:
    _assert_metadata(client.send(GET_WIDGET(name="blue")).data())


def test_paginated_entities_keep_metadata(client: NemoClient) -> None:
    (widget,) = list(client.send(LIST_WIDGETS()).items())
    _assert_metadata(widget)


def test_entities_nested_in_a_response_model_keep_metadata(client: NemoClient) -> None:
    _assert_metadata(client.send(GET_ENVELOPE(name="blue")).data().widget)


def test_metadata_is_serialized_back_out(client: NemoClient) -> None:
    """CLI output dumps the parsed model; the metadata must survive the round trip."""
    dumped = client.send(GET_WIDGET(name="blue")).data().model_dump(mode="json")
    assert dumped["id"] == "widget-123"
    assert dumped["created_at"] == "2026-09-01T12:00:00Z"


async def test_async_client_keeps_metadata() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as http_client:
        client = AsyncNemoClient(base_url=BASE, workspace="default", http_client=http_client)
        _assert_metadata((await client.send(GET_WIDGET(name="blue"))).data())


def test_validation_outside_a_response_ignores_metadata() -> None:
    """Request bodies must not be able to set store-managed fields."""
    widget = MetadataWidget.model_validate(WIDGET)
    assert widget.id == ""
    assert widget.created_at is None
    assert widget.created_by is None
    assert widget.db_version == 1

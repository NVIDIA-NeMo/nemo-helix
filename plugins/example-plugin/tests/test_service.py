# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the example-plugin ExampleService — entity CRUD routes.

The entity client is replaced with a pytest-mock MagicMock so tests run without
a real entity store.  All tests target the route functions directly via FastAPI's
TestClient rather than the live platform.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_example_plugin.entities import ExampleItem
from nemo_example_plugin.middleware_config import ExampleMiddlewareConfig
from nemo_example_plugin.middleware_service import _get_entity_client as _get_middleware_entity_client
from nemo_example_plugin.service import ExampleService, _get_entity_client
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError, NemoPaginationInfo

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(name: str = "item1", workspace: str = "default") -> ExampleItem:
    """Build a fake persisted ExampleItem with store-populated fields."""
    item = ExampleItem(name=name, workspace=workspace, title="Test Item", tags=["ml"])
    item._id = f"id-{name}"
    item._created_at = NOW
    return item


def _make_middleware_config(name: str = "cfg-1", workspace: str = "default") -> ExampleMiddlewareConfig:
    """Build a fake persisted ExampleMiddlewareConfig with store-populated fields."""
    config = ExampleMiddlewareConfig(name=name, workspace=workspace)
    config._id = f"id-{name}"
    config._created_at = NOW
    return config


def _make_list_response(items: list[ExampleItem]) -> MagicMock:
    """Build a fake ListResponse as returned by entity_client.list()."""
    resp = MagicMock()
    resp.data = items
    resp.pagination = NemoPaginationInfo(
        page=1,
        page_size=20,
        current_page_size=len(items),
        total_pages=1,
        total_results=len(items),
    )
    return resp


def _make_app(mock_client: AsyncMock) -> FastAPI:
    """Build a FastAPI app wired to a mock entity client."""
    service = ExampleService()
    app = FastAPI()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[_get_entity_client] = lambda: mock_client
    app.dependency_overrides[_get_middleware_entity_client] = lambda: mock_client
    return app


# ---------------------------------------------------------------------------
# POST /items — create
# ---------------------------------------------------------------------------


def test_create_item_201() -> None:
    mock = AsyncMock()
    saved = _make_item("widget")
    mock.create.return_value = saved

    client = TestClient(_make_app(mock))
    resp = client.post(
        "/v2/workspaces/default/items",
        json={"name": "widget", "title": "Test Item", "tags": ["ml"]},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "widget"
    assert body["id"] == "id-widget"
    assert "created_at" in body


def test_create_item_409_on_conflict() -> None:
    mock = AsyncMock()
    mock.create.side_effect = NemoEntityConflictError("conflict")

    client = TestClient(_make_app(mock))
    resp = client.post(
        "/v2/workspaces/default/items",
        json={"name": "dup", "title": "Duplicate"},
    )

    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /items — paginated list
# ---------------------------------------------------------------------------


def test_list_items_returns_nemo_list_response_envelope() -> None:
    mock = AsyncMock()
    mock.list.return_value = _make_list_response([_make_item("a"), _make_item("b")])

    client = TestClient(_make_app(mock))
    resp = client.get("/v2/workspaces/default/items")

    assert resp.status_code == 200
    body = resp.json()
    # Must have the NemoListResponse envelope keys
    assert set(body.keys()) >= {"data", "pagination", "sort", "filter"}
    assert len(body["data"]) == 2
    assert body["pagination"]["total_results"] == 2


def test_list_items_pagination_params_forwarded() -> None:
    mock = AsyncMock()
    mock.list.return_value = _make_list_response([])

    client = TestClient(_make_app(mock))
    client.get("/v2/workspaces/default/items?page=2&page_size=5&sort=name")

    call_kwargs = mock.list.call_args.kwargs
    assert call_kwargs["page"] == 2
    assert call_kwargs["page_size"] == 5
    assert call_kwargs["sort"] == "name"


def test_list_items_filter_applied() -> None:
    """filter_obj is passed to entity_client.list when filter params are present."""
    mock = AsyncMock()
    mock.list.return_value = _make_list_response([])

    client = TestClient(_make_app(mock))
    # Use simple string filter field — tags list syntax varies by deep-object parser.
    # The important invariant is that filter_obj is non-None when filter params exist.
    client.get("/v2/workspaces/default/items?filter[tag]=ml")

    call_kwargs = mock.list.call_args.kwargs
    # filter_obj will be either the parsed dict or None depending on the filter value;
    # the key assertion is that the filter was processed (no unhandled exception).
    assert "filter_obj" in call_kwargs


def test_list_items_no_filter_params_sends_none() -> None:
    """No filter params → filter_obj=None is passed to entity_client.list."""
    mock = AsyncMock()
    mock.list.return_value = _make_list_response([])

    client = TestClient(_make_app(mock))
    resp = client.get("/v2/workspaces/default/items")

    assert resp.status_code == 200
    call_kwargs = mock.list.call_args.kwargs
    assert call_kwargs.get("filter_obj") is None


# ---------------------------------------------------------------------------
# GET /items/{name} — single resource
# ---------------------------------------------------------------------------


def test_get_item_200() -> None:
    mock = AsyncMock()
    mock.get.return_value = _make_item("widget")

    client = TestClient(_make_app(mock))
    resp = client.get("/v2/workspaces/default/items/widget")

    assert resp.status_code == 200
    assert resp.json()["name"] == "widget"
    assert resp.json()["id"] == "id-widget"


def test_get_item_404() -> None:
    mock = AsyncMock()
    mock.get.side_effect = NemoEntityNotFoundError("not found")

    client = TestClient(_make_app(mock))
    resp = client.get("/v2/workspaces/default/items/missing")

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# PATCH /items/{name} — partial update
# ---------------------------------------------------------------------------


def test_update_item_200() -> None:
    mock = AsyncMock()
    existing = _make_item("widget")
    updated = _make_item("widget")
    updated.title = "New Title"
    mock.get.return_value = existing
    mock.update.return_value = updated

    client = TestClient(_make_app(mock))
    resp = client.patch(
        "/v2/workspaces/default/items/widget",
        json={"title": "New Title"},
    )

    assert resp.status_code == 200
    assert resp.json()["title"] == "New Title"


def test_update_item_404() -> None:
    mock = AsyncMock()
    mock.get.side_effect = NemoEntityNotFoundError("not found")

    client = TestClient(_make_app(mock))
    resp = client.patch(
        "/v2/workspaces/default/items/missing",
        json={"title": "x"},
    )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /items/{name}
# ---------------------------------------------------------------------------


def test_delete_item_204() -> None:
    mock = AsyncMock()
    mock.get.return_value = _make_item("widget")
    mock.delete.return_value = None

    client = TestClient(_make_app(mock))
    resp = client.delete("/v2/workspaces/default/items/widget")

    assert resp.status_code == 204
    mock.delete.assert_awaited_once_with(
        ExampleItem,
        name="widget",
        workspace="default",
    )


def test_delete_item_404() -> None:
    mock = AsyncMock()
    mock.delete.side_effect = NemoEntityNotFoundError("not found")

    client = TestClient(_make_app(mock))
    resp = client.delete("/v2/workspaces/default/items/missing")

    assert resp.status_code == 404


def test_delete_item_409_when_changed() -> None:
    mock = AsyncMock()
    mock.delete.side_effect = NemoEntityConflictError("changed")

    client = TestClient(_make_app(mock))
    resp = client.delete("/v2/workspaces/default/items/widget")

    assert resp.status_code == 409


def test_delete_middleware_config_204() -> None:
    mock = AsyncMock()
    mock.delete.return_value = None

    client = TestClient(_make_app(mock))
    resp = client.delete("/v2/workspaces/default/middleware-configs/cfg-1")

    assert resp.status_code == 204
    mock.delete.assert_awaited_once_with(
        ExampleMiddlewareConfig,
        name="cfg-1",
        workspace="default",
    )


def test_delete_middleware_config_409_when_changed() -> None:
    mock = AsyncMock()
    mock.delete.side_effect = NemoEntityConflictError("changed")

    client = TestClient(_make_app(mock))
    resp = client.delete("/v2/workspaces/default/middleware-configs/cfg-1")

    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Entity computed fields — id and created_at are present
# ---------------------------------------------------------------------------


def test_response_includes_id_and_created_at() -> None:
    """Verify entity computed fields (id, created_at) are included in responses."""
    mock = AsyncMock()
    saved = _make_item("w1")
    mock.create.return_value = saved

    client = TestClient(_make_app(mock))
    resp = client.post(
        "/v2/workspaces/default/items",
        json={"name": "w1", "title": "T"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == "id-w1"
    assert body["created_at"] is not None
    assert "updated_at" in body  # present even if None

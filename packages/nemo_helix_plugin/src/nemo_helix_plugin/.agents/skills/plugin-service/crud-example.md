<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Complete CRUD Example

Reference implementation for a full entity-backed CRUD service using `Widget` as the example entity.

**Contents:**
- [Entity + Schemas](#entity--schemas)
- [Service + Router](#service--router) — POST, GET list, GET single, PATCH, DELETE
- [Test File](#test-file)

## Entity + Schemas

```python
# entities.py
from nemo_helix_plugin.entity import NemoEntity

class Widget(NemoEntity, entity_type="example_widget"):
    colour: str
    weight_kg: float = 0.0
    tags: list[str] = []
```

```python
# schema.py
from __future__ import annotations
from nemo_helix_plugin.schema import NemoFilter, NemoListResponse, PaginationData
from pydantic import BaseModel, Field

from .entities import Widget

class CreateWidgetRequest(BaseModel):
    name: str = Field(description="Unique widget name within the workspace.")
    colour: str = Field(description="Widget colour.")
    weight_kg: float = Field(default=0.0)
    tags: list[str] = Field(default_factory=list)

class UpdateWidgetRequest(BaseModel):
    colour: str | None = Field(default=None)
    weight_kg: float | None = Field(default=None)
    tags: list[str] | None = Field(default=None)

class WidgetFilter(NemoFilter):              # extra="forbid" inherited → typos = 422
    colour: str | None = Field(default=None)
    tag: str | None = Field(default=None)

WidgetPage = NemoListResponse[Widget]
```

## Service + Router

```python
# service.py
from __future__ import annotations
import logging
from typing import ClassVar

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.schema import PaginationData
from nemo_helix_plugin.service import NemoService, RouterSpec
from nhx.common.entities.filters import make_filter_obj_dep

from .entities import Widget
from .schema import (
    CreateWidgetRequest,
    UpdateWidgetRequest,
    WidgetFilter,
    WidgetPage,
)

logger = logging.getLogger(__name__)

scope = AuthzScope("my-plugin")


class WidgetPerms(PermissionSet, namespace="my-plugin.widgets"):
    CREATE = perm("Create a widget")   # -> Permission("my-plugin.widgets.create", ...)
    LIST = perm("List widgets")
    READ = perm("Read a widget")
    UPDATE = perm("Update a widget")
    DELETE = perm("Delete a widget")


class MyService(NemoService):
    name: ClassVar[str] = "my-plugin"
    dependencies: ClassVar[list[str]] = ["entities"]

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                _build_router(),
                tag="Widgets",
                description="Full entity-backed CRUD for Widgets.",
                prefix="/v2/workspaces/{workspace}",
            )
        ]


def _build_router() -> APIRouter:
    router = APIRouter()
    _filter_dep = make_filter_obj_dep(WidgetFilter)

    # ------------------------------------------------------------------
    # POST /widgets — create (201)
    # ------------------------------------------------------------------

    @router.post("/widgets", response_model=Widget, status_code=201)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.CREATE])
    async def create_widget(
        workspace: str,
        body: CreateWidgetRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Widget:
        widget = Widget(
            name=body.name,
            workspace=workspace,
            colour=body.colour,
            weight_kg=body.weight_kg,
            tags=body.tags,
        )
        try:
            saved = await entity_client.create(widget)
        except NemoEntityConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Widget '{body.name}' already exists in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to create widget '%s'", body.name)
            raise HTTPException(status_code=500, detail="Failed to create widget.") from exc
        return saved

    # ------------------------------------------------------------------
    # GET /widgets — list with pagination + filter (200)
    # ------------------------------------------------------------------

    @router.get("/widgets", response_model=WidgetPage)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.LIST])
    async def list_widgets(
        workspace: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        sort: str = Query(default="-created_at"),
        filter: WidgetFilter = Depends(_filter_dep),
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> WidgetPage:
        # make_filter_obj_dep can return a NemoFilter instance OR a raw dict
        filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
        try:
            result = await entity_client.list(
                Widget,
                workspace=workspace,
                page=page,
                page_size=page_size,
                sort=sort,
                filter_obj=filter_dict or None,
            )
        except Exception as exc:
            logger.exception("Failed to list widgets in workspace '%s'", workspace)
            raise HTTPException(status_code=500, detail="Failed to list widgets.") from exc

        # PaginationInfo (entity client) → PaginationData (API response): always convert
        pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
        return WidgetPage(
            data=result.data,
            pagination=pagination,
            sort=sort,
            filter=filter,
        )

    # ------------------------------------------------------------------
    # GET /widgets/{name} — single (200 or 404)
    # ------------------------------------------------------------------

    @router.get("/widgets/{name}", response_model=Widget)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.READ])
    async def get_widget(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Widget:
        try:
            widget = await entity_client.get(Widget, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Widget '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to get widget '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to get widget.") from exc
        return widget

    # ------------------------------------------------------------------
    # PATCH /widgets/{name} — partial update (200 or 404 or 409)
    # ------------------------------------------------------------------

    @router.patch("/widgets/{name}", response_model=Widget)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.UPDATE])
    async def update_widget(
        workspace: str,
        name: str,
        body: UpdateWidgetRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Widget:
        try:
            widget = await entity_client.get(Widget, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Widget '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to fetch widget '%s' for update", name)
            raise HTTPException(status_code=500, detail="Failed to fetch widget.") from exc

        if body.colour is not None:
            widget.colour = body.colour
        if body.weight_kg is not None:
            widget.weight_kg = body.weight_kg
        if body.tags is not None:
            widget.tags = body.tags

        try:
            saved = await entity_client.update(widget)
        except NemoEntityConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Widget '{name}' was modified concurrently. Please retry.",
            ) from exc
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Widget '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to update widget '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to update widget.") from exc
        return saved

    # ------------------------------------------------------------------
    # DELETE /widgets/{name} — delete (204)
    # ------------------------------------------------------------------

    @router.delete("/widgets/{name}", status_code=204)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.DELETE])
    async def delete_widget(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> None:
        try:
            await entity_client.delete(Widget, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Widget '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to delete widget '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to delete widget.") from exc

    return router
```

## Test File

```python
# tests/test_service.py
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_helix_plugin.entity_client import NemoEntityNotFoundError, NemoEntityConflictError, get_entity_client
from nhx.common.entities.client import ListResponse, PaginationInfo

from nemo_my_plugin.entities import Widget
from nemo_my_plugin.service import MyService

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_widget(name: str = "w1", workspace: str = "default") -> Widget:
    w = Widget(name=name, workspace=workspace, colour="red", weight_kg=1.5, tags=["a"])
    w._id = f"id-{name}"
    w._created_at = NOW
    return w


def _make_app(mock_client: AsyncMock) -> FastAPI:
    service = MyService()
    app = FastAPI()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: mock_client
    return app


def test_create_widget():
    mock_client = AsyncMock()
    mock_client.create.return_value = _make_widget("w1")
    client = TestClient(_make_app(mock_client))
    resp = client.post(
        "/v2/workspaces/default/widgets",
        json={"name": "w1", "colour": "red", "weight_kg": 1.5},
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "w1"
    assert resp.json()["colour"] == "red"


def test_create_widget_conflict():
    mock_client = AsyncMock()
    mock_client.create.side_effect = NemoEntityConflictError("exists")
    client = TestClient(_make_app(mock_client))
    resp = client.post("/v2/workspaces/default/widgets", json={"name": "w1", "colour": "red"})
    assert resp.status_code == 409


def test_list_widgets():
    mock_client = AsyncMock()
    mock_client.list.return_value = ListResponse(
        data=[_make_widget("w1")],
        pagination=PaginationInfo(page=1, page_size=20, current_page_size=1,
                                   total_pages=1, total_results=1),
    )
    client = TestClient(_make_app(mock_client))
    resp = client.get("/v2/workspaces/default/widgets")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 1
    assert data["pagination"]["total_results"] == 1


def test_get_widget_not_found():
    mock_client = AsyncMock()
    mock_client.get.side_effect = NemoEntityNotFoundError("not found")
    client = TestClient(_make_app(mock_client))
    resp = client.get("/v2/workspaces/default/widgets/missing")
    assert resp.status_code == 404


def test_update_widget():
    mock_client = AsyncMock()
    updated = _make_widget("w1")
    updated.colour = "blue"
    mock_client.get.return_value = _make_widget("w1")
    mock_client.update.return_value = updated
    client = TestClient(_make_app(mock_client))
    resp = client.patch("/v2/workspaces/default/widgets/w1", json={"colour": "blue"})
    assert resp.status_code == 200
    assert resp.json()["colour"] == "blue"


def test_delete_widget():
    mock_client = AsyncMock()
    mock_client.delete.return_value = None
    client = TestClient(_make_app(mock_client))
    resp = client.delete("/v2/workspaces/default/widgets/w1")
    assert resp.status_code == 204
```

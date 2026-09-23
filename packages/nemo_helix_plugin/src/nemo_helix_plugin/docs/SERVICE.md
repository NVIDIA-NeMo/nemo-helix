<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Service Surface (NemoService)

## NemoService

```python
from nemo_helix_plugin.service import NemoService, RouterSpec

class NemoService(_NamedPlugin):
    name: ClassVar[str]                    # REQUIRED — kebab-case; becomes URL prefix /apis/<name>
    dependencies: ClassVar[list[str]] = [] # platform services that must be ready before startup

    @abstractmethod
    def get_routers(self) -> list[RouterSpec]: ...  # MUST implement

    async def on_startup(self) -> None: ...   # called once before routes are mounted
    async def on_shutdown(self) -> None: ...  # called once after requests stop
```

Valid `dependencies` values: `"entities"`, `"auth"`, `"jobs"`, `"files"`, `"secrets"`, `"models"`, `"inference-gateway"`.

The platform instantiates each `NemoService` subclass once at startup. `get_routers()` is called once; the returned routers are mounted permanently.

## RouterSpec

```python
from dataclasses import dataclass
from fastapi import APIRouter

@dataclass
class RouterSpec:
    router: APIRouter    # the FastAPI router to mount
    tag: str = ""        # OpenAPI tag grouping
    description: str = ""
    prefix: str = ""     # URL prefix appended AFTER /apis/<name>
```

**URL formula:** `/apis/<name>/<spec.prefix>/<route-path>`

Example: `name="my-plugin"`, `prefix="/v2/workspaces/{workspace}"`, route `/widgets` → `/apis/my-plugin/v2/workspaces/{workspace}/widgets`

Platform convention: `/apis/<name>/v2/workspaces/{workspace}/<resource>` — follow this for consistency with core platform services.

## Response schemas

### Entity objects as responses

Return entity objects directly from route handlers — no separate response class needed. `NemoEntity` subclasses (via `EntityBase`) expose `id`, `created_at`, `updated_at`, `name`, `workspace`, `project` as `@computed_field` properties that serialize automatically:

```python
from nemo_helix_plugin.entity import NemoEntity

class Widget(NemoEntity, entity_type="my_plugin_widget"):
    colour: str
    weight_kg: float = 0.0

@router.post("/widgets", response_model=Widget, status_code=201)
async def create_widget(...) -> Widget:
    saved = await entity_client.create(widget)
    return saved  # entity object returned directly
```

### NemoListResponse

Generic paginated list response:

```python
from nemo_helix_plugin.schema import NemoListResponse

WidgetPage = NemoListResponse[Widget]
```

Wire format:
```json
{
    "data": [...],
    "pagination": {
        "page": 1,
        "page_size": 20,
        "current_page_size": 5,
        "total_pages": 1,
        "total_results": 5
    },
    "sort": "-created_at",
    "filter": {"colour": "red"}
}
```

### NemoFilter

Base for query filter models. `extra="forbid"` inherited — field typos in query params return 422 instead of silently returning unfiltered results:

```python
from nemo_helix_plugin.schema import NemoFilter

class WidgetFilter(NemoFilter):
    colour: str | None = None   # ?filter[colour]=red
    tag: str | None = None      # ?filter[tag]=ml
```

Filter query param syntax: `deepObject` — `?filter[colour]=red&filter[tag]=ml`

Always use `make_filter_obj_dep` from `nhx.common.entities.filters`:

```python
from nhx.common.entities.filters import make_filter_obj_dep

_filter_dep = make_filter_obj_dep(WidgetFilter)

@router.get("/widgets")
async def list_widgets(
    filter: WidgetFilter = Depends(_filter_dep),
    ...
):
    # REQUIRED check — make_filter_obj_dep may return a raw dict for wildcard filters
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    result = await entity_client.list(Widget, workspace=workspace, filter_obj=filter_dict or None)
```

### PaginationData vs PaginationInfo

`entity_client.list()` returns `PaginationInfo` (entity-client-internal). `NemoListResponse` expects `PaginationData` (public API layer). They are structurally identical but different Pydantic classes. **Must convert:**

```python
from nemo_helix_plugin.schema import PaginationData

pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
```

## Complete CRUD example

```python
from __future__ import annotations

import logging
from typing import ClassVar

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.schema import (
    NemoFilter,
    NemoListResponse,
    PaginationData,
)
from nemo_helix_plugin.service import NemoService, RouterSpec
from nhx.common.entities.filters import make_filter_obj_dep
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Entity ---

class Widget(NemoEntity, entity_type="example_widget"):
    colour: str
    weight_kg: float = 0.0


# --- Schemas ---

class CreateWidgetRequest(BaseModel):
    name: str
    colour: str
    weight_kg: float = 0.0

class UpdateWidgetRequest(BaseModel):
    colour: str | None = None
    weight_kg: float | None = None

class WidgetFilter(NemoFilter):
    colour: str | None = None

WidgetPage = NemoListResponse[Widget]


# --- Service ---

class MyService(NemoService):
    name: ClassVar[str] = "my-plugin"
    dependencies: ClassVar[list[str]] = ["entities"]

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                _build_router(),
                tag="Widgets",
                prefix="/v2/workspaces/{workspace}",
            )
        ]


def _build_router() -> APIRouter:
    router = APIRouter()
    _filter_dep = make_filter_obj_dep(WidgetFilter)

    # POST /widgets — create (201)
    @router.post("/widgets", response_model=Widget, status_code=201)
    async def create_widget(
        workspace: str,
        body: CreateWidgetRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Widget:
        widget = Widget(name=body.name, workspace=workspace, colour=body.colour, weight_kg=body.weight_kg)
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

    # GET /widgets — paginated list (200)
    @router.get("/widgets", response_model=WidgetPage)
    async def list_widgets(
        workspace: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        sort: str = Query(default="-created_at"),
        filter: WidgetFilter = Depends(_filter_dep),
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> WidgetPage:
        # isinstance check is REQUIRED — may be a raw dict for wildcard filters
        filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
        try:
            result = await entity_client.list(
                Widget, workspace=workspace,
                page=page, page_size=page_size, sort=sort,
                filter_obj=filter_dict or None,
            )
        except Exception as exc:
            logger.exception("Failed to list widgets in workspace '%s'", workspace)
            raise HTTPException(status_code=500, detail="Failed to list widgets.") from exc
        # Convert PaginationInfo → PaginationData
        pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
        return WidgetPage(
            data=result.data,
            pagination=pagination,
            sort=sort,
            filter=filter,
        )

    return router
```

> Full 5-operation CRUD with all imports and error handlers: see [`crud-example.md`](../.agents/skills/plugin-service/crud-example.md).

For calling jobs, files, secrets, models, and inference gateway from routes, see the [`plugin-platform-services` skill](../.agents/skills/plugin-platform-services/SKILL.md).

## Lifecycle hooks

```python
class MyService(NemoService):
    name: ClassVar[str] = "my-plugin"

    async def on_startup(self) -> None:
        # Safe to call async platform APIs here
        self._cache = await build_cache()

    async def on_shutdown(self) -> None:
        await self._cache.close()
```

## Testing

```python
from unittest.mock import AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError, NemoPaginationInfo

from nemo_helix_plugin.entity_client import get_entity_client
from nemo_my_plugin.service import MyService


def _make_app(mock_client: AsyncMock) -> FastAPI:
    service = MyService()
    app = FastAPI()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: mock_client
    return app


def test_create_widget_201():
    mock = AsyncMock()
    mock.create.return_value = _make_widget("w1")  # helper that sets _id etc.

    client = TestClient(_make_app(mock))
    resp = client.post("/v2/workspaces/default/widgets", json={"name": "w1", "colour": "red"})

    assert resp.status_code == 201
    assert resp.json()["name"] == "w1"
```

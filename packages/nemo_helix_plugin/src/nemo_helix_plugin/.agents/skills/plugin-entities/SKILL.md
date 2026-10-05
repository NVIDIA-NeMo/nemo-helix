---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-entities
description: Defines NemoEntity subclasses and uses NemoEntitiesClient for CRUD in the NeMo Helix entity store. Use when defining a new entity type, storing plugin data in the entity store, handling optimistic locking conflicts, listing entities with filters, or building entity clients for controllers. Trigger keywords: entity, entity store, NemoEntity, entity_type, NemoEntitiesClient, entity client, store data, optimistic lock, EntityConflictError, EntityNotFoundError.
---

# Plugin Entities (NemoEntity + NemoEntitiesClient)

## Defining an Entity

```python
from nemo_helix_plugin.entity import NemoEntity

class Widget(NemoEntity, entity_type="example_widget"):
    """Stored in entity store under entity type 'example_widget'."""
    colour: str
    weight_kg: float = 0.0
    tags: list[str] = []
```

`entity_type` is **required** on every concrete subclass. It must be snake_case and plugin-scoped (e.g., `"my_plugin_widget"` not `"widget"`). Omitting it raises `TypeError` at class-definition time.

Inherited fields from `EntityBase` (no need to declare):

- `name: str` — human-readable name within workspace
- `workspace: str` — workspace identifier (required, validated against `ID_PATTERN`)
- `project: str | None` — optional project association
- `id: str` — UUID (empty string `""` before persistence — NEVER use in response before save)
- `created_at`, `updated_at`, `created_by`, `updated_by`

## Abstract Intermediate Bases

Two patterns to exempt a class from the `entity_type` requirement:

```python
# Option 1: __abstract__ = True
class BasePluginEntity(NemoEntity):
    __abstract__ = True
    status: str = "pending"

class Widget(BasePluginEntity, entity_type="example_widget"):
    colour: str

# Option 2: inherit from ABC
from abc import ABC

class BasePluginEntity(NemoEntity, ABC):
    status: str = "pending"
```

> **Note:** `__abstract__ = True` only exempts the class that declares it directly. It is **not inherited** — subclasses of `BasePluginEntity` that don't re-declare `__abstract__ = True` must provide `entity_type`.

## Entity CRUD — FastAPI Endpoints

Imports used in every CRUD route:

```python
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from fastapi import Depends, HTTPException
```

**Create (POST → 201):**

```python
widget = Widget(name=body.name, workspace=workspace, colour=body.colour)
try:
    saved = await entity_client.create(widget)
except NemoEntityConflictError as exc:
    raise HTTPException(status_code=409, detail=f"Widget '{body.name}' already exists.") from exc
return saved
```

**List (GET → 200 with pagination):**

```python
filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
result = await entity_client.list(
    Widget,
    workspace=workspace,
    page=page,
    page_size=page_size,
    sort=sort,
    filter_obj=filter_dict or None,
)
pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
return WidgetPage(data=result.data, pagination=pagination, ...)
```

**Get single (GET → 200 or 404):**

```python
try:
    widget = await entity_client.get(Widget, name=name, workspace=workspace)
except NemoEntityNotFoundError as exc:
    raise HTTPException(status_code=404, detail=f"Widget '{name}' not found.") from exc
return widget
```

**Update (PATCH → 200 or 404 or 409):**

```python
try:
    widget = await entity_client.get(Widget, name=name, workspace=workspace)
except NemoEntityNotFoundError as exc:
    raise HTTPException(status_code=404, detail=f"Widget '{name}' not found.") from exc

if body.colour is not None:
    widget.colour = body.colour

try:
    saved = await entity_client.update(widget)
except NemoEntityConflictError as exc:
    raise HTTPException(status_code=409, detail="Concurrent modification.") from exc
except NemoEntityNotFoundError as exc:
    raise HTTPException(status_code=404, detail=f"Widget '{name}' not found.") from exc
return saved
```

**Delete (DELETE → 204):**

```python
try:
    await entity_client.delete(Widget, name=name, workspace=workspace)
except NemoEntityNotFoundError as exc:
    raise HTTPException(status_code=404, detail=f"Widget '{name}' not found.") from exc
```

## filter_obj Patterns

Pass field names directly — **NEVER add a `data.` prefix**. The entity client adds it automatically for non-base fields.

```python
# CORRECT — client adds data. prefix for custom fields automatically
filter_obj={"colour": "red"}          # → search: {"data.colour": "red"}
filter_obj={"name": "my-widget"}      # base field → no prefix: {"name": "my-widget"}

# WRONG — double-prefix bug
filter_obj={"data.colour": "red"}     # → search: {"data.data.colour": "red"} ← broken!
```

Base fields (no prefix): `name`, `workspace`, `project`, `id`, `created_at`, `updated_at`, `entity_type`.

## Pagination

`entity_client.list()` returns `PaginationInfo` (internal). `NemoListResponse` expects `PaginationData` (API layer). They are structurally identical — always convert:

```python
from nemo_helix_plugin.schema import PaginationData

pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
```

Sort format: `"-created_at"` (descending), `"status"` (ascending). Non-base fields are auto-prefixed with `data.` by the client.

## Optimistic Locking

`NemoEntityConflictError` covers two distinct scenarios:

1. **Create**: entity with that `(name, workspace, entity_type)` already exists → respond 409
2. **Update**: `_db_version` mismatch — another request modified the entity between your `get()` and `update()` → respond 409 in services, log debug + skip in controllers

## Cross-Workspace Listing (Controllers)

```python
# workspace="-" lists entities across ALL workspaces
result = await entity_client.list(Widget, workspace="-")
```

`workspace="-"` is a sentinel value handled specially by the entity store. **Never create or update entities with `workspace="-"`.**

## Building Entity Client Without Request Context

For controllers and background tasks where FastAPI's `Depends()` is not available:

```python
from nhx.common.sdk_factory import get_async_platform_sdk
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nhx.common.entities.client import EntityClient

sdk = get_async_platform_sdk(as_service="my-plugin", internal=True)
entity_client = EntityClient(client_from_platform(sdk, AsyncEntitiesClient))
```

`internal=True` adds headers that suppress the access log flood from controller polling every few seconds. Always use `internal=True` for background/controller clients.

## Entity computed fields in API responses

`NemoEntity` subclasses expose `id`, `created_at`, `updated_at` as `@computed_field` properties that appear in `model_dump()` output automatically. Return entity objects directly from route handlers:

```python
@router.post("/widgets", response_model=Widget, status_code=201)
async def create_widget(...) -> Widget:
    saved = await entity_client.create(widget)
    return saved  # id, created_at, etc. are in the serialized output
```

## Test Helper Pattern

```python
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

def _make_widget(name: str = "w1", workspace: str = "default") -> Widget:
    """Build a fake persisted Widget with store-populated private attrs."""
    w = Widget(name=name, workspace=workspace, colour="red")
    w._id = f"id-{name}"          # set private attr directly
    w._created_at = NOW
    return w

# AsyncMock setup
from unittest.mock import MagicMock
from nemo_helix_plugin.entity_client import NemoPaginationInfo

mock_client = AsyncMock()
mock_client.create.return_value = _make_widget("w1")
mock_client.get.side_effect = NemoEntityNotFoundError("not found")

# ListResponse mock — use MagicMock to avoid internal import dependencies
list_resp = MagicMock()
list_resp.data = [_make_widget("w1")]
list_resp.pagination = NemoPaginationInfo(
    page=1, page_size=20, current_page_size=1, total_pages=1, total_results=1
)
mock_client.list.return_value = list_resp
```

## Retry on Optimistic Lock (Services)

In services (not controllers), retry with backoff rather than returning 409:

```python
import asyncio
from nemo_helix_plugin.entity_client import NemoEntityConflictError

async def update_with_retry(entity_client, name, workspace, update_fn):
    for attempt in range(3):
        try:
            entity = await entity_client.get(Widget, name=name, workspace=workspace)
            update_fn(entity)
            return await entity_client.update(entity)
        except NemoEntityConflictError:
            if attempt == 2:
                raise
            await asyncio.sleep(0.1 * (attempt + 1))
```

## Page Through All Entities

```python
async def list_all(entity_client, workspace):
    page = 1
    while True:
        result = await entity_client.list(Widget, workspace=workspace, page=page, page_size=100)
        for entity in result.data:
            yield entity
        if page >= result.pagination.total_pages:
            break
        page += 1
```

## Parent-Child Entities

Child uniqueness is `(workspace, entity_type, parent, name)` instead of the usual `(workspace, entity_type, name)`:

```python
parent = await entity_client.create(ParentEntity(name="p1", workspace="default"))

child = ChildEntity(name="c1", workspace="default")
child._parent = parent.id  # link via UUID
child = await entity_client.create(child)

# Retrieve with parent context
child = await entity_client.get(ChildEntity, "c1", workspace="default", parent=parent.id)
```

## Gotchas

- **`entity_type` REQUIRED on concrete classes**: `TypeError` at class-definition time if missing. Error message shows exact correct syntax.
- **`entity.id` is `""` (empty string) before persistence**: Never return an unsaved entity from a route — the empty `id` will appear in the API response. Always call `entity_client.create()` or `entity_client.get()` first.
- **`filter_obj` with `data.` prefix**: Double-prefix bug — `{"data.colour": "red"}` becomes `{"data.data.colour": "red"}` in the actual query. Always pass bare field names.
- **`workspace="-"` is a sentinel**: Never use it for creating/updating entities — only for listing across workspaces.
- **`entity_type` auto-derivation is DISABLED**: `NemoEntity` explicitly disables the `EntityBase` behavior of deriving `entity_type` from the class name. You must always pass it as a keyword.
- **`workspace` and `name` must match `ID_PATTERN`**: Pattern is `^[\w\-\+.@:]+$`. Spaces and forward slashes cause validation errors on both fields.
- **`NemoEntityConflictError` has two meanings**: check call site — `create()` means duplicate; `update()` means version mismatch.

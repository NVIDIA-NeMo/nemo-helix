<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Entities (NemoEntity + NemoEntitiesClient)

## Defining entities

Subclass `NemoEntity` and pass `entity_type` as a class keyword. Every concrete subclass **must** supply it — omitting it raises `TypeError` at class-definition time.

```python
from nemo_helix_plugin.entity import NemoEntity

class Widget(NemoEntity, entity_type="example_widget"):
    colour: str
    weight_kg: float = 0.0
    tags: list[str] = []
```

`entity_type` must be `snake_case` and **plugin-scoped** to avoid collisions across plugins:

- BAD: `entity_type="widget"` — will collide with other plugins
- GOOD: `entity_type="example_widget"` — plugin-scoped, globally unique

## EntityBase inherited fields

Every `NemoEntity` inherits these from `EntityBase`:

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Human-readable name, unique within `(workspace, entity_type, parent)` |
| `workspace` | `str` | Required; must match `ID_PATTERN = r"^[\w\-\+.@:]+$"` |
| `project` | `str \| None` | Optional project association |
| `id` | `str` | **Empty string `""` before persistence** — never `None` |
| `created_at` | `datetime \| None` | Set by entity store on create |
| `updated_at` | `datetime \| None` | Set by entity store on update |
| `created_by` | `str \| None` | Principal ID who created the entity |
| `updated_by` | `str \| None` | Principal ID who last updated |
| `parent` | `str \| None` | Parent entity UUID for nested entities |

> **HARD GOTCHA:** `entity.id` is `""` (empty string), NOT `None`, before the entity is persisted. Never return an entity from a route before saving it — the empty `id` will appear in the API response.

## Abstract intermediate bases

Use `__abstract__ = True` or inherit from `ABC` to create shared base classes that are exempt from the `entity_type` requirement:

```python
from abc import ABC
from nemo_helix_plugin.entity import NemoEntity

# Option 1: __abstract__ flag
class StatusMixin(NemoEntity):
    __abstract__ = True
    status: str = "pending"

# Option 2: inherit from ABC
class AuditedEntity(NemoEntity, ABC):
    audit_note: str = ""

# Concrete subclass must still supply entity_type:
class Widget(StatusMixin, entity_type="example_widget"):
    colour: str
```

## NemoEntitiesClient reference

```python
from nemo_helix_plugin.entity_client import NemoEntitiesClient

# create — raises NemoEntityConflictError if name already exists in workspace
async def create(self, entity: EntityT) -> EntityT: ...

# list — returns ListResponse with .data and .pagination (type PaginationInfo)
async def list(
    self,
    entity_type,
    *,
    workspace: str = "default",
    sort: str | None = None,
    filter_obj: dict | None = None,
    page: int = 1,
    page_size: int = 100,
) -> ListResponse[EntityT]: ...

# get — raises NemoEntityNotFoundError if not found
async def get(self, entity_type, name: str, *, workspace: str | None = None, parent: str | None = None) -> EntityT: ...

# get_by_id — raises NemoEntityNotFoundError if not found
async def get_by_id(self, entity_type, entity_id: str) -> EntityT: ...

# update — raises NemoEntityNotFoundError or NemoEntityConflictError (version mismatch)
async def update(self, entity: EntityT, *, original_name: str | None = None) -> EntityT: ...

# delete — raises NemoEntityNotFoundError if not found
async def delete(self, entity_type, name: str, *, workspace: str | None = None, parent: str | None = None): ...

# delete_by_id
async def delete_by_id(self, entity_type, entity_id: str): ...

# save — upsert: creates if id is empty, updates otherwise
async def save(self, entity: EntityT) -> EntityT: ...
```

## Using the entity client in endpoints

```python
from fastapi import APIRouter, Depends, HTTPException
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)

router = APIRouter()

@router.get("/widgets/{name}")
async def get_widget(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Widget:
    try:
        widget = await entity_client.get(Widget, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Widget '{name}' not found.") from exc
    return widget
```

## Error handling

Two exceptions cover all entity store errors:

| Exception | HTTP status | When raised |
|---|---|---|
| `NemoEntityNotFoundError` | 404 | `get`, `get_by_id`, `update`, `delete` — entity does not exist |
| `NemoEntityConflictError` | 409 | **Two scenarios:** (1) `create` — name already exists in workspace; (2) `update` — `_db_version` mismatch (optimistic lock) |

Both scenarios for `NemoEntityConflictError` must be handled at the appropriate call site. The `create` conflict means "already exists"; the `update` conflict means another process modified the entity between your `get()` and `update()`.

## Optimistic locking

Every entity carries `_db_version` (starts at 1, incremented by the store on each write). `update()` automatically sends `expected_db_version=entity._db_version`. If the store's current version differs, it raises `NemoEntityConflictError`.

**In controllers (log debug and skip — retries next cycle):**
```python
async def reconcile_one(self, obj):
    try:
        await self._reconcile_one(obj)
    except NemoEntityConflictError:
        logger.debug("Optimistic lock conflict on '%s' — will retry next cycle.", obj.name)
```

For the service retry-with-backoff pattern, see the [`plugin-entities` skill](../.agents/skills/plugin-entities/SKILL.md).

## Filtering and pagination

Pass `filter_obj` as a plain `dict` — **never add `data.` prefix yourself**. The client adds it automatically for non-base fields. Adding it yourself creates a double-prefix bug (`data.data.status`).

```python
# Correct — client adds data. prefix for entity-specific fields
result = await entity_client.list(Widget, workspace=workspace, filter_obj={"colour": "red"})

# WRONG — results in data.data.colour query
result = await entity_client.list(Widget, workspace=workspace, filter_obj={"data.colour": "red"})
```

`list()` returns a `ListResponse` with:
- `.data` — list of entity instances
- `.pagination` — `PaginationInfo` object (NOT `PaginationData`)

**Conversion to `PaginationData` for API responses:**
```python
from nemo_helix_plugin.schema import PaginationData

pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
```

Sort: `"-created_at"` (descending), `"status"` (ascending). The client auto-prefixes non-base fields.

For the page-through-all generator and parent-child entity patterns, see the [`plugin-entities` skill](../.agents/skills/plugin-entities/SKILL.md).

## Cross-workspace listing

```python
# Lists entities from ALL workspaces — used in controllers
result = await entity_client.list(Widget, workspace="-")
```

`workspace="-"` is a sentinel value. **Never use it to create entities** — it will fail.

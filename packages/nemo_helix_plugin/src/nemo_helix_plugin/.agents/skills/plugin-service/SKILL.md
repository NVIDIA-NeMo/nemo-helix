---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-service
description: Builds HTTP service surfaces for NeMo Helix plugins using NemoService, RouterSpec, NemoListResponse, and NemoFilter. Use when adding REST API routes to a plugin, implementing CRUD endpoints, handling pagination and filtering, or testing FastAPI routes. Trigger keywords: HTTP routes, REST API, FastAPI, CRUD, endpoint, router, NemoService, pagination, filter, list endpoint, NemoListResponse, RouterSpec.
---

# Plugin HTTP Service (NemoService)

## Class Signature

```python
from typing import ClassVar
from nemo_helix_plugin.service import NemoService, RouterSpec

class MyService(NemoService):
    name: ClassVar[str] = "my-plugin"          # REQUIRED — entry-point key, URL prefix
    dependencies: ClassVar[list[str]] = ["entities"]  # Helix services to wait for

    def get_routers(self) -> list[RouterSpec]:  # REQUIRED
        return [RouterSpec(_build_router(), tag="My Plugin", prefix="/v2/workspaces/{workspace}")]

    async def on_startup(self) -> None: ...     # optional — async init
    async def on_shutdown(self) -> None: ...    # optional — cleanup
```

Valid `dependencies` values: `"entities"`, `"auth"`, `"jobs"`, `"files"`, `"secrets"`, `"models"`, `"inference-gateway"`. These services must be healthy before `on_startup()` is called.

## URL Pattern

`/apis/<name>/<spec.prefix>/<route-path>`

Example: `name="my-plugin"`, `prefix="/v2/workspaces/{workspace}"`, route `/widgets` → `/apis/my-plugin/v2/workspaces/{workspace}/widgets`

Convention: always use `prefix="/v2/workspaces/{workspace}"` for resource endpoints.

## Authorization

Every plugin HTTP route MUST carry a `@path_rule` — a route with no rule is a validation error, and under the default `on_invalid_plugin=hard_fail` the auth service refuses to build the OPA bundle rather than silently fence the route, so the platform 502s. Each route also declares exactly one OAuth scope with `@AuthzScope.read` / `.write` — a separate, orthogonal decorator. `@router.<method>` is the outermost decorator; `@scope` and `@path_rule` sit beneath it (their relative order is irrelevant):

```python
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm

scope = AuthzScope("my-plugin")                    # owns my-plugin:read / my-plugin:write

class WidgetPerms(PermissionSet, namespace="my-plugin.widgets"):
    READ = perm("Read a widget")                   # -> "my-plugin.widgets.read"

@router.get("/widgets/{name}", response_model=Widget)
@scope.read                                        # OAuth scope gate
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.READ])
async def get_widget(...) -> Widget: ...
```

Reference `PermissionSet` members (never bare strings — a bare string raises `TypeError` at import), and pass `authz=scope` to route factories (`add_job_routes` / `add_function_routes`) so their routes are ruled too. See [`../plugin-authz/SKILL.md`](../plugin-authz/SKILL.md) for the full surface — caller kinds, compound permission ids, role grants, and fail-mode config.

## Response Schemas

### Entity objects as responses

Return entity objects directly — no separate response class needed:

```python
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.entity import NemoEntity

scope = AuthzScope("my-plugin")                    # owns my-plugin:read / my-plugin:write

class WidgetPerms(PermissionSet, namespace="my-plugin.widgets"):
    CREATE = perm("Create a widget")               # -> "my-plugin.widgets.create"

class Widget(NemoEntity, entity_type="my_plugin_widget"):
    colour: str
    weight_kg: float = 0.0

# Use the entity class directly as response_model:
@router.post("/widgets", response_model=Widget, status_code=201)
@scope.write                                       # OAuth scope gate
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.CREATE])
async def create_widget(...) -> Widget:
    saved = await entity_client.create(widget)
    return saved
```

`NemoEntity` subclasses expose `id`, `created_at`, `updated_at`, `name`, `workspace`, `project` as computed fields — they serialize correctly in responses without any adapter class.

For non-CRUD endpoints or computed responses, use a plain `BaseModel` subclass instead.

### NemoListResponse

```python
from nemo_helix_plugin.schema import NemoListResponse

WidgetPage = NemoListResponse[Widget]  # entity class used directly
```

`NemoListResponse` wire format:
```json
{
    "data": [...],
    "pagination": {"page": 1, "page_size": 20, "current_page_size": 5,
                   "total_pages": 1, "total_results": 5},
    "sort": "-created_at",
    "filter": {"colour": "red"}
}
```

> **CRITICAL**: `entity_client.list()` returns `PaginationInfo` (entity client internal). `NemoListResponse` expects `PaginationData` (API response layer). Always convert:
> ```python
> from nemo_helix_plugin.schema import PaginationData
> pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
> ```

## NemoFilter

```python
from nemo_helix_plugin.schema import NemoFilter

class WidgetFilter(NemoFilter):      # extra="forbid" inherited — typos → 422
    colour: str | None = None
    tag: str | None = None
```

- `extra="forbid"` means unknown filter fields return 422 (not silently ignored)
- deepObject query syntax: `?filter[colour]=red`
- Factory: `make_filter_obj_dep(WidgetFilter)` from `nhx.common.entities.filters`
- **Always check `isinstance(filter, dict)`** — `make_filter_obj_dep` can return either a `NemoFilter` instance or a raw `dict` for wildcard filters:

```python
filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
result = await entity_client.list(Widget, workspace=workspace, filter_obj=filter_dict or None)
```

## Dependency Injection

```python
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client
from nhx.common.auth.dependencies import get_auth_client
from nhx.common.auth.client import AuthClient
from nhx.common.service.dependencies import get_sdk_client  # transitive dep via nhx-common; not re-exported by nemo_helix_plugin
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from fastapi import Depends

scope = AuthzScope("my-plugin")

class WidgetPerms(PermissionSet, namespace="my-plugin.widgets"):
    LIST = perm("List widgets")                    # -> "my-plugin.widgets.list"

@router.get("/widgets")
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.LIST])
async def list_widgets(
    workspace: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    auth_client: AuthClient = Depends(get_auth_client),
    sdk: AsyncNeMoHelix = Depends(get_sdk_client),
) -> WidgetPage:
    principal_id = auth_client.principal.id
    ...
```

`get_entity_client` is a placeholder — the platform injects the real implementation at startup via `app.dependency_overrides`.

## CRUD Quick Reference

> **Complete implementation:** See [`crud-example.md`](crud-example.md) — full entity-backed CRUD service with all imports, error handling, and tests.

| Operation | Method | Status | Key errors |
|---|---|---|---|
| Create | `POST /widgets` | 201 | `NemoEntityConflictError` → 409 |
| List | `GET /widgets` | 200 | `isinstance(filter, dict)` check required |
| Get | `GET /widgets/{name}` | 200 | `NemoEntityNotFoundError` → 404 |
| Update | `PATCH /widgets/{name}` | 200 | `NotFoundError` on get → 404; `ConflictError` on update → 409 |
| Delete | `DELETE /widgets/{name}` | 204 | `NemoEntityNotFoundError` → 404 |

All `raise` statements must use `from exc` syntax for error chaining.

## Testing Pattern

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from nemo_helix_plugin.entity_client import get_entity_client

def _make_app(mock_client: AsyncMock) -> FastAPI:
    service = MyService()
    app = FastAPI()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: mock_client
    return app

def test_get_widget():
    mock_client = AsyncMock()
    mock_client.get.return_value = _make_widget("w1")
    app = _make_app(mock_client)
    client = TestClient(app)
    response = client.get("/v2/workspaces/default/widgets/w1")
    assert response.status_code == 200
    assert response.json()["name"] == "w1"
```

## Gotchas

- **`PaginationInfo` ≠ `PaginationData`**: `entity_client.list()` returns `PaginationInfo`; `NemoListResponse` expects `PaginationData`. Always convert: `PaginationData.model_validate(result.pagination.model_dump())`.
- **`isinstance(filter, dict)` check is required**: `make_filter_obj_dep` may return either a `NemoFilter` instance or a raw `dict` for wildcard filters. Calling `.model_dump()` on a dict raises `AttributeError`.
- **`entity.id` is `""` before persistence**: Never return an entity from a route before saving it — the empty id will appear in the response.
- **`from exc` on all raises**: Use `raise HTTPException(...) from exc` for all error re-raises to preserve the exception chain.
- **`get_entity_client` is a placeholder**: The platform injects the real implementation via `app.dependency_overrides`. In tests, override it with `lambda: mock_client`.
- **`NemoFilter` extra="forbid"**: Unknown filter query params return 422. This is by design — it prevents typos from silently returning unfiltered results.
- **URL formula**: Routes mount at `/apis/<name>/<prefix>/<route>`. The `prefix` in `RouterSpec` is appended after the service name, not replacing it.
- **Unruled route → OPA bundle build fails**: A route with no `@path_rule` is a validation error; under the default `on_invalid_plugin=hard_fail` the auth service refuses to build the bundle and the platform 502s. Attach `@path_rule` (plus `@scope.read`/`.write`) to every handler, or `authz=` on a route factory.

## See Also

- [`crud-example.md`](crud-example.md) — Complete CRUD implementation with all imports
- [`../plugin-entities/SKILL.md`](../plugin-entities/SKILL.md) — Entity definitions and client
- [`../plugin-platform-services/SKILL.md`](../plugin-platform-services/SKILL.md) — Calling other Helix services
- [`../plugin-authz/SKILL.md`](../plugin-authz/SKILL.md) — Authorization for plugin routes

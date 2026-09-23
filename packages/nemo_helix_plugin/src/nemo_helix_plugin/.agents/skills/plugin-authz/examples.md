<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# plugin-authz recipes

Copy-paste-correct recipes for every plugin authz surface. **The rule:** every plugin HTTP route MUST carry a `@path_rule` — attached directly on a hand-written handler, or via `authz=` on a route factory. A route with no rule is a validation error; under the default `on_invalid_plugin="hard_fail"` the auth service refuses to build the OPA bundle (the platform 502s) rather than silently fencing the route.

Two decorators ride on every hand-written handler and they are orthogonal:

- `@AuthzScope.read` / `@AuthzScope.write` — the route's single OAuth scope gate.
- `@path_rule(callers=[...], permissions=[...])` — the permission/caller rule. Within a rule `callers` are OR'd and `permissions` are AND'd; stacking `@path_rule` adds OR-alternatives.

Decorator order: `@router.<method>` is the **outermost** (top) decorator; `@scope.read`/`.write` and `@path_rule` sit under it, and their relative order does not matter (both only stamp attributes on the function and return it unchanged).

Permission ids follow the grammar `<service>.<resource>.<action>`, and `<service>` MUST equal `NemoService.name` — that namespace fence is fail-closed, so a mis-namespaced id takes the whole plugin down. Always reference a `PermissionSet` member (e.g. `WidgetPerms.CREATE`), never a bare string — a string is a `TypeError` at import.

**Contents:**
- [Hand-written CRUD router](#hand-written-crud-router) — `_perms.py` + `authz.py` + `@scope` / `@path_rule` per route
- [Authenticated-but-permissionless route](#authenticated-but-permissionless-route) — `permissions=[]` (e.g. `/healthz`)
- [Job collection](#job-collection) — `add_job_routes(MyJob, authz=...)`
- [Function](#function) — `add_function_routes(MyFn, authz=..., permission_description=...)`
- [Controller-written status route](#controller-written-status-route) — `CallerKind.SERVICE_PRINCIPAL`
- [Granting a Viewer an `.invoke` permission](#granting-a-viewer-an-invoke-permission) — `extra_role_permissions`
- [Migrating off `get_authz_contribution()`](#migrating-off-get_authz_contribution) — before / after
- [Verify](#verify)

All recipes below belong to one hypothetical plugin whose `NemoService.name` is `"my-plugin"`, so its scope is `AuthzScope("my-plugin")` and every permission id starts with `my-plugin.`.

## Hand-written CRUD router

A typed permission vocabulary lives in `_perms.py`. Each `perm(...)` member becomes a `Permission` whose id is `<namespace>.<member-name-lowercased>` — you author the description once, on the permission, and the platform derives the catalog from the routes (no parallel list to keep in sync).

```python
# _perms.py
from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class WidgetPerms(PermissionSet, namespace="my-plugin.widgets"):
    CREATE = perm("Create a widget")   # -> Permission id "my-plugin.widgets.create"
    LIST = perm("List widgets")        # -> "my-plugin.widgets.list"
    READ = perm("Read a widget")       # -> "my-plugin.widgets.read"
```

The scope lives in its own module so both the service and any other route module can share the one `AuthzScope("my-plugin")` without an import cycle:

```python
# authz.py
from __future__ import annotations

from nemo_helix_plugin.authz import AuthzScope

scope = AuthzScope("my-plugin")  # owns OAuth scopes my-plugin:read / my-plugin:write
```

The router wires `@scope` and `@path_rule` onto each handler. Reads carry `@scope.read`; mutating routes carry `@scope.write`. (Entity + schema definitions — `Widget`, `CreateWidgetRequest`, `WidgetFilter`, `WidgetPage` — are the plugin-service skill's CRUD example; only the authz decorators are shown in full here.)

```python
# service.py
from __future__ import annotations

import logging
from typing import ClassVar

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_helix_plugin.api.filters import make_filter_obj_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.schema import PaginationData
from nemo_helix_plugin.service import NemoService, RouterSpec

from .authz import scope
from .entities import Widget
from ._perms import WidgetPerms
from .schema import CreateWidgetRequest, WidgetFilter, WidgetPage

logger = logging.getLogger(__name__)


class MyService(NemoService):
    name: ClassVar[str] = "my-plugin"
    dependencies: ClassVar[list[str]] = ["entities"]

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                _build_widgets_router(),
                tag="Widgets",
                description="Full entity-backed CRUD for Widgets.",
                prefix="/v2/workspaces/{workspace}",
            )
        ]


def _build_widgets_router() -> APIRouter:
    router = APIRouter()
    _filter_dep = make_filter_obj_dep(WidgetFilter)

    # POST /widgets — create (201). Mutating route -> @scope.write + CREATE permission.
    @router.post("/widgets", response_model=Widget, status_code=201)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.CREATE])
    async def create_widget(
        workspace: str,
        body: CreateWidgetRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Widget:
        widget = Widget(name=body.name, workspace=workspace, colour=body.colour, tags=body.tags)
        try:
            return await entity_client.create(widget)
        except NemoEntityConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Widget '{body.name}' already exists in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to create widget '%s'", body.name)
            raise HTTPException(status_code=500, detail="Failed to create widget.") from exc

    # GET /widgets — list (200). Read route -> @scope.read + LIST permission.
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

        pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
        return WidgetPage(data=result.data, pagination=pagination, sort=sort, filter=filter)

    # GET /widgets/{name} — single (200 or 404). Read route -> @scope.read + READ permission.
    @router.get("/widgets/{name}", response_model=Widget)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.READ])
    async def get_widget(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Widget:
        try:
            return await entity_client.get(Widget, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Widget '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to get widget '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to get widget.") from exc

    return router
```

## Authenticated-but-permissionless route

`permissions=[]` is a valid rule — an authenticated caller passes with no specific permission held. Use it for infra routes like `/healthz` or a raw blob `PUT`. `callers` is still required and non-empty, and the route still carries exactly one scope.

```python
# service.py (a top-level router, mounted without a workspace prefix)
from fastapi import APIRouter
from nemo_helix_plugin.authz import CallerKind, path_rule

from .authz import scope


def _build_health_router() -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])  # authenticated, no permission required
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return router
```

```python
    def get_routers(self) -> list[RouterSpec]:
        return [
            # ... the widgets RouterSpec above ...
            RouterSpec(_build_health_router(), tag="Health", description="Liveness probe."),
        ]
```

## Job collection

`add_job_routes(job_cls, authz=...)` mounts submit/list/get/delete (plus cancel/status/logs/results). Pass `authz=` or the routes are **unruled** — the kwarg defaults to `None`. When set, each generated route is stamped with a PRINCIPAL `@path_rule` and a scope: reads share one `<namespace>.read` permission; mutating routes get their own (`<namespace>.create`, `.delete`, `.cancel`, ...), all descriptions minted from the job factory's catalog. (Building the `NemoJob` class itself — `spec_schema`, `run`, `compile`, `container` — is the **plugin-job** skill; here we only add the authz wiring around it.)

```python
# service.py
from nemo_helix_plugin.authz import AuthzScope
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec

from my_plugin.jobs.run import RunJob  # a NemoJob subclass declaring spec_schema


class MyService(NemoService):
    name = "my-plugin"

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                add_job_routes(RunJob, authz=AuthzScope("my-plugin")),
                prefix="/v2/workspaces/{workspace}",
                tag="My Plugin",
                description="Job endpoints",
            ),
            # -> permissions my-plugin.create / my-plugin.list / my-plugin.read / my-plugin.delete (+ cancel...)
        ]
```

To nest permission ids per job collection while keeping the coarse OAuth scope, use `scope.child(...)`:

```python
scope = AuthzScope("my-plugin")
add_job_routes(RunJob, authz=scope.child("run"))
# -> permissions my-plugin.run.create / my-plugin.run.list / ...; OAuth scope stays "my-plugin"
```

## Function

`add_function_routes(function_cls, authz=..., permission_description=...)` mounts a single `POST`. Pass `authz=` or the route is unruled. Invoking a function is a **write** action: the adapter stamps a PRINCIPAL `@path_rule` with an invoke permission minted as `<namespace>.<function-name>` and attaches the write scope. `permission_description` requires `authz` — supplying it alone is a `ValueError` (the description would be silently discarded). (Building the `NemoFunction` class — `spec_schema`, `async def run` — is the **plugin-function** skill; here we only add the authz wiring.)

```python
# service.py
from nemo_helix_plugin.authz import AuthzScope
from nemo_helix_plugin.functions.routes import add_function_routes
from nemo_helix_plugin.service import NemoService, RouterSpec

from my_plugin.functions.summarize import SummarizeFunction  # a NemoFunction with name = "summarize"


class MyService(NemoService):
    name = "my-plugin"

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                add_function_routes(
                    SummarizeFunction,
                    authz=AuthzScope("my-plugin"),
                    permission_description="Invoke the summarize function",
                ),
                prefix="/v2/workspaces/{workspace}",
                tag="My Plugin",
                description="Non-streaming NemoFunction example.",
            ),
            # -> permission my-plugin.summarize (a write action), PRINCIPAL caller, my-plugin:write scope
        ]
```

## Controller-written status route

A route a controller (not a human) writes to takes `CallerKind.SERVICE_PRINCIPAL` — a caller whose id is prefixed `service:`. Caller-kind is enforced in Rego; back it in the handler with a dependency that rejects non-service callers. The permission it checks is minted under the collection it projects onto via a compound `suffix`.

```python
# _perms.py
from nemo_helix_plugin.authz import PermissionSet, perm


class WidgetPerms(PermissionSet, namespace="my-plugin.widgets"):
    CREATE = perm("Create a widget")
    LIST = perm("List widgets")
    READ = perm("Read a widget")
    STATUS_UPDATE = perm("Update widget observed status (controller)", suffix="status.update")
    # -> Permission id "my-plugin.widgets.status.update"
```

```python
# dependencies.py
from __future__ import annotations

from fastapi import HTTPException, Request

_PRINCIPAL_ID_HEADER = "X-NHX-Principal-Id"


def require_service_principal(request: Request) -> None:
    """Restrict controller-only status writes to service principals."""
    principal_id = request.headers.get(_PRINCIPAL_ID_HEADER, "")
    if not principal_id.startswith("service:"):
        raise HTTPException(status_code=403, detail="Status updates require a service principal.")
```

```python
# status.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)

from .authz import scope
from .dependencies import require_service_principal
from .entities import Widget
from ._perms import WidgetPerms
from .schema import UpdateWidgetStatusRequest

router = APIRouter()


@router.put("/widgets/{name}/status", response_model=Widget, tags=["Widget Status"])
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[WidgetPerms.STATUS_UPDATE])
async def update_widget_status(
    workspace: str,
    name: str,
    body: UpdateWidgetStatusRequest,
    _: None = Depends(require_service_principal),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Widget:
    try:
        widget = await entity_client.get(Widget, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Widget '{name}' not found in workspace '{workspace}'.") from exc

    widget.status = body.status
    widget.status_message = body.status_message

    try:
        return await entity_client.update(widget)
    except NemoEntityConflictError as exc:
        raise HTTPException(status_code=409, detail="Concurrent modification.") from exc
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Widget '{name}' not found in workspace '{workspace}'.") from exc
```

## Granting a Viewer an `.invoke` permission

The default role-grant heuristic assigns catalog permissions by id suffix: ids ending `.list` or `.read` → Viewer **and** Editor; everything else → Editor only. When a role needs a permission the heuristic misses — e.g. a Viewer must hold a gateway `.invoke` permission to call a deployed agent even though its suffix isn't `read` — grant it explicitly via `extra_role_permissions`. Grants are unioned with the suffix defaults (never subtractive), and every granted permission must live in this service's own namespace or the whole plugin fails closed.

```python
# _perms.py
class GatewayPerms(PermissionSet, namespace="my-plugin.gateway"):
    INVOKE = perm("Invoke a deployed agent through the gateway proxy")  # -> "my-plugin.gateway.invoke"
```

```python
# service.py
from nemo_helix_plugin.authz import Permission
from nemo_helix_plugin.service import NemoService, RouterSpec

from ._perms import GatewayPerms


class MyService(NemoService):
    name = "my-plugin"

    def get_routers(self) -> list[RouterSpec]:
        ...

    def extra_role_permissions(self) -> dict[str, list[Permission]]:
        # The `.invoke` suffix isn't `read`/`list`, so the default heuristic gives it to Editor
        # only. Grant Viewer here too. (Editor still gets it via that same suffix heuristic.)
        return {"Viewer": [GatewayPerms.INVOKE]}
```

`extra_role_permissions` also registers each granted permission in the catalog, so it need not also appear in `extra_permissions()`. Use `extra_permissions()` for the other escape-hatch case: a permission with no 1:1 route (checked in middleware, or declared ahead of the route that will reference it).

## Migrating off `get_authz_contribution()`

The old surface is gone: the `get_authz_contribution()` classmethod and the `nemo.authz` entry point were removed. Migrate by moving each permission id into a `PermissionSet` (ids are preserved, so role grants do not change), attaching `@path_rule` + `@scope.read`/`.write` to every handler (or `authz=` on a factory), and moving any heuristic-missed role grant into `extra_role_permissions`.

**Before** — the removed surface (`get_authz_contribution` returning the internal wire shapes, plus a `nemo.authz` entry point):

```python
# service.py — OLD, no longer supported
from nemo_helix_plugin.authz import AuthzContribution, AuthzEndpointMethod  # internal wire format
from nemo_helix_plugin.service import NemoService, RouterSpec


class MyService(NemoService):
    name = "my-plugin"

    @classmethod
    def get_authz_contribution(cls) -> AuthzContribution:
        return AuthzContribution(
            permissions={
                "my-plugin.widgets.create": "Create a widget",
                "my-plugin.widgets.list": "List widgets",
                "my-plugin.widgets.read": "Read a widget",
                "my-plugin.gateway.invoke": "Invoke a deployed agent through the gateway proxy",
            },
            endpoints={
                "/apis/my-plugin/v2/workspaces/{workspace}/widgets": {
                    "post": AuthzEndpointMethod(
                        permissions=["my-plugin.widgets.create"], scopes=["my-plugin:write", "platform:write"]
                    ),
                    "get": AuthzEndpointMethod(
                        permissions=["my-plugin.widgets.list"], scopes=["my-plugin:read", "platform:read"]
                    ),
                },
                "/apis/my-plugin/v2/workspaces/{workspace}/widgets/{name}": {
                    "get": AuthzEndpointMethod(
                        permissions=["my-plugin.widgets.read"], scopes=["my-plugin:read", "platform:read"]
                    ),
                },
            },
            role_permissions={"Viewer": ["my-plugin.gateway.invoke"]},
        )

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(_build_widgets_router(), prefix="/v2/workspaces/{workspace}")]  # handlers had no authz
```

```toml
# pyproject.toml — OLD
[project.entry-points."nemo.authz"]
my-plugin = "my_plugin.service:MyService"

[project.entry-points."nemo.services"]
my-plugin = "my_plugin.service:MyService"
```

**After** — the id `my-plugin.widgets.create` (etc.) is preserved as a `PermissionSet` member, so its role grants are unchanged. The classmethod and the `nemo.authz` entry point are deleted; the catalog, endpoint bindings, and namespace are all derived from the routes. The `nemo.authz` entry point is gone — only `nemo.services` remains.

```python
# _perms.py — each old id, verbatim, becomes a PermissionSet member
from nemo_helix_plugin.authz import PermissionSet, perm


class WidgetPerms(PermissionSet, namespace="my-plugin.widgets"):
    CREATE = perm("Create a widget")   # -> "my-plugin.widgets.create" (unchanged)
    LIST = perm("List widgets")        # -> "my-plugin.widgets.list"
    READ = perm("Read a widget")       # -> "my-plugin.widgets.read"


class GatewayPerms(PermissionSet, namespace="my-plugin.gateway"):
    INVOKE = perm("Invoke a deployed agent through the gateway proxy")  # -> "my-plugin.gateway.invoke"
```

```python
# service.py — NEW: scope + rule on each handler, invoke grant via extra_role_permissions
from nemo_helix_plugin.authz import AuthzScope, CallerKind, Permission, path_rule
from nemo_helix_plugin.service import NemoService, RouterSpec

from ._perms import GatewayPerms, WidgetPerms

scope = AuthzScope("my-plugin")


class MyService(NemoService):
    name = "my-plugin"

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(_build_widgets_router(), prefix="/v2/workspaces/{workspace}")]

    def extra_role_permissions(self) -> dict[str, list[Permission]]:
        # `.invoke` isn't caught by the .read/.list heuristic, so re-declare the old Viewer grant.
        return {"Viewer": [GatewayPerms.INVOKE]}


def _build_widgets_router():
    from fastapi import APIRouter

    router = APIRouter()

    @router.post("/widgets", status_code=201)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.CREATE])
    async def create_widget(workspace: str): ...

    @router.get("/widgets")
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.LIST])
    async def list_widgets(workspace: str): ...

    @router.get("/widgets/{name}")
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[WidgetPerms.READ])
    async def get_widget(workspace: str, name: str): ...

    return router
```

```toml
# pyproject.toml — NEW: the nemo.authz entry point is removed
[project.entry-points."nemo.services"]
my-plugin = "my_plugin.service:MyService"
```

## Verify

From the repo root, with the workspace installed:

```bash
uv run python services/core/auth/scripts/auth-tools.py sync-plugins --dry-run
# -> reports 0 degraded plugins when every route is ruled

make build-policy && make test-policy
```

Local dev note: `services/core/auth/.../assets/policy.wasm` is a gitignored build artifact. Run `make build-policy` after pulling authz/Rego changes, otherwise `test_embedded_pdp.py` fails against a stale copy (misleading `sprintf` / fence-fails-open errors).

If a route is unruled or references an out-of-namespace permission, the fail-mode is set by the auth service config field `on_invalid_plugin` (env `NHX_AUTH_ON_INVALID_PLUGIN`, default `hard_fail`): `hard_fail` refuses to build the bundle (platform-wide fail-closed); `quarantine` denies the whole offending plugin but keeps the platform up; `deny_route` denies only the bad routes.

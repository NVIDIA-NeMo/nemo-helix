---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-controller
description: Creates background reconcile-loop controllers using NemoController. Use when implementing state-machine reconciliation, running periodic background work, managing deployment lifecycle, building service-principal entity clients for background use, or understanding controller startup/shutdown sequence. Trigger keywords: controller, NemoController, reconcile, background loop, reconcile_one, list_objects, on_startup, state machine, deployment lifecycle, service principal, interval_seconds.
---

# Plugin Controllers (NemoController)

## Class Signature

```python
from typing import ClassVar
from nemo_helix_plugin.controller import NemoController

class MyController(NemoController):
    name: ClassVar[str] = "my-plugin"           # REQUIRED — matches entry-point key
    dependencies: ClassVar[list[str]] = ["entities"]  # platform services to wait for

    # Must implement:
    async def list_objects(self) -> list: ...    # return objects to reconcile this cycle
    async def reconcile_one(self, obj: object) -> None: ...  # reconcile a single object

    # May override:
    async def reconcile(self) -> None: ...       # default calls list_objects + reconcile_one
    async def on_startup(self) -> None: ...      # called once before first cycle
    async def on_shutdown(self) -> None: ...     # called after last cycle

    @property
    def interval_seconds(self) -> float:         # NOT ClassVar — see below
        return 10.0

    @property
    def is_healthy(self) -> bool:                # override for custom health check
        return True
```

## Lifecycle

1. Platform creates `NemoControllerAdapter` wrapping the controller instance
2. `on_startup()` called once — initialize entity client, config, backends
3. Loop: `reconcile()` called every `interval_seconds`
   - Default `reconcile()` → `list_objects()` → `reconcile_one(obj)` per item with error isolation
4. On SIGINT/SIGTERM: waits for current `reconcile()` to complete, then calls `on_shutdown()`

## on_startup() Patterns

`__init__()` must contain **only** `None` sentinels — no platform calls. All initialization goes in `on_startup()`:

```python
class MyController(NemoController):
    name = "my-plugin"

    def __init__(self) -> None:
        self._entities: NemoEntitiesClient | None = None
        self._interval_seconds: float = 10.0

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    async def on_startup(self) -> None:
        from nhx.common.sdk_factory import get_async_platform_sdk
        from nemo_helix_plugin.client.adapter import client_from_platform
        from nemo_helix_plugin.entities.client import AsyncEntitiesClient
        from nemo_helix_plugin.entity_client import NemoEntitiesClient
        from .config import MyPluginConfig

        config = MyPluginConfig.get()
        self._interval_seconds = float(config.controller_interval)

        sdk = get_async_platform_sdk(as_service="my-plugin", internal=True)
        typed_client = client_from_platform(sdk, AsyncEntitiesClient)
        self._entities = NemoEntitiesClient(typed_client)

    @property
    def entities(self) -> NemoEntitiesClient:
        if self._entities is None:
            raise RuntimeError("Controller accessed before on_startup()")
        return self._entities
```

Guard properties prevent silent failures if something accesses the client before `on_startup()` runs.

## interval_seconds as @property

`interval_seconds` is a `@property`, NOT a `ClassVar`. This lets it read from config loaded in `on_startup()`:

```python
# CORRECT — reads config set in on_startup()
@property
def interval_seconds(self) -> float:
    return self._interval_seconds   # set from config.controller.interval_seconds

# WRONG — ClassVar is set at class-definition time before config is loaded
# interval_seconds: ClassVar[float] = 10.0   ← do NOT do this
```

## list_objects() Pattern

```python
async def list_objects(self) -> list:
    try:
        result = await self.entities.list(MyEntity, workspace="-")  # all workspaces
        return result.data
    except Exception:
        logger.exception("Failed to list entities across workspaces")
        return []   # return [] on exception — skip cycle silently
```

Always return `[]` on exception. Exceptions from `list_objects()` propagate out of the default `reconcile()` method — only `reconcile_one()` calls are wrapped in per-item error isolation. Raising from `list_objects()` aborts the entire cycle.

## reconcile_one() Pattern

```python
async def reconcile_one(self, obj: object) -> None:
    entity = cast(MyEntity, obj)
    try:
        await self._reconcile_one(entity)
    except NemoEntityConflictError:
        # Optimistic lock — another process updated this entity. Log debug, skip.
        # The next cycle will fetch the updated version.
        logger.debug("Optimistic lock conflict on '%s' — will retry next cycle.", entity.name)

async def _reconcile_one(self, entity: MyEntity) -> None:
    if entity.status == "pending":
        await self._start(entity)
    elif entity.status == "active":
        await self._check_health(entity)
    elif entity.status == "deleting":
        await self._delete(entity)
```

`NemoEntityConflictError` from `update()` = optimistic lock. Log at **debug** level (NOT error), skip the item. The next cycle will re-fetch with the current version.

## internal=True — Why It Matters

Without `internal=True`, a controller polling every 5 seconds across 100 entities floods the entity store access log with 20 requests/second of noise.

```python
sdk = get_async_platform_sdk(as_service="my-plugin", internal=True)
#                                                     ^^^^^^^^^^^
# Adds MARK_INTERNAL_REQUEST_HEADERS — suppresses access log on receiving service
```

Always use `internal=True` for controller/background SDK calls.

## Service vs. Controller Decoupling

The service and controller are completely independent. They share no in-process state:

- **Service** (request scope): user calls `POST /widgets` → creates entity with `status="pending"` → returns 201
- **Controller** (background): polls entities with `status="pending"` → drives state machine → updates `status`

This decoupling means the service returns fast and the controller handles all async work.

## Building a Service-Principal Entity Client

Full pattern used in `on_startup()`:

```python
from nhx.common.sdk_factory import get_async_platform_sdk
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient

sdk = get_async_platform_sdk(as_service="my-plugin", internal=True)
typed_client = client_from_platform(sdk, AsyncEntitiesClient)
self._entities = NemoEntitiesClient(typed_client)
```

`as_service="my-plugin"` sets `X-NHX-Principal-Id: service:my-plugin` on all outgoing requests. Service principals have elevated permissions for cross-workspace listing.

## Authorization — Status-Write Routes

A controller writes observed state back through the platform HTTP API, and its SDK carries a service-principal identity — `as_service="my-plugin"` sets `X-NHX-Principal-Id: service:my-plugin` (above). So any status-write route the plugin exposes *for* the controller must be gated to service principals only, or a normal user token could spoof observed status:

```python
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_deployments_plugin.api.v2._perms import DeploymentPerms
from nemo_deployments_plugin.authz import scope

@router.put("/deployments/{name}/status", response_model=Deployment, tags=["Deployment Status"])
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[DeploymentPerms.STATUS_UPDATE])
async def update_deployment_status(...): ...
```

`CallerKind.SERVICE_PRINCIPAL` (vs. the default `PRINCIPAL`) is enforced in Rego — the caller's id must be prefixed `service:`. The `STATUS_UPDATE` permission is a compound id minted under the collection the controller projects onto:

```python
from nemo_helix_plugin.authz import PermissionSet, perm

class DeploymentPerms(PermissionSet, namespace="deployments.deployments"):
    STATUS_UPDATE = perm("Update deployment observed status (controller)", suffix="status.update")
    # -> id "deployments.deployments.status.update"
```

Model these routes on `plugins/nemo-deployments/src/nemo_deployments_plugin/api/v2/status.py`. Every plugin HTTP route needs a `@path_rule` plus a `@scope.read`/`.write` gate — see the `plugin-authz` skill for the full route-authz surface (`AuthzScope`, `PermissionSet`, `perm`, route factories, and fail modes).

## On-Behalf-Of Access (for user secrets)

When a controller needs to access a user-owned secret:

```python
from nhx.common.sdk_factory import get_async_platform_sdk, get_sdk_on_behalf_of

async def _access_user_secret(self, entity: MyEntity) -> str:
    service_sdk = get_async_platform_sdk(as_service="my-plugin")
    user_sdk = get_sdk_on_behalf_of(service_sdk, entity.owner_principal_id)
    return await user_sdk.secrets.access(entity.secret_name, workspace=entity.workspace)
```

## Overriding reconcile() Entirely

For controllers with complex multi-phase logic that doesn't fit list + per-item:

```python
async def reconcile(self) -> None:
    # Phase 1: collect
    pending = await self._get_pending()
    running = await self._get_running()
    # Phase 2: act on batches
    await self._batch_start(pending)
    await self._batch_check(running)

# Stub out the abstract methods (they won't be called):
async def list_objects(self) -> list:
    raise NotImplementedError

async def reconcile_one(self, obj: object) -> None:
    raise NotImplementedError
```

## Minimal Controller Example

Verbatim from example-plugin:

```python
from nemo_helix_plugin.controller import NemoController
import logging

logger = logging.getLogger(__name__)
_OBJECTS: list[str] = ["alpha", "beta", "gamma"]

class ExampleController(NemoController):
    name = "example-controller"

    @property
    def interval_seconds(self) -> float:
        return 30.0

    async def on_startup(self) -> None:
        from nemo_example_plugin.config import ExampleConfig
        config = ExampleConfig.get()
        logger.info(
            "ExampleController starting up — greeting_style=%r  log_requests=%r",
            config.greeting_style,
            config.log_requests,
        )

    async def on_shutdown(self) -> None:
        logger.info("ExampleController shutting down.")

    async def list_objects(self) -> list:
        return list(_OBJECTS)

    async def reconcile_one(self, obj: object) -> None:
        logger.debug("ExampleController reconciling object: %s", obj)
```

## Production State Machine Example

Full pattern from a production deployment controller. Shows `__init__` sentinels + `on_startup` SDK factory + cross-workspace `list_objects` + state dispatch in `_reconcile_one` + optimistic lock catch:

```python
import logging
from typing import ClassVar

from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError

logger = logging.getLogger(__name__)

class DeploymentController(NemoController):
    name = "my-deployment"
    dependencies: ClassVar[list[str]] = ["entities"]

    def __init__(self) -> None:
        # __init__ ONLY sets None sentinels — never calls platform APIs
        self._entities: NemoEntitiesClient | None = None
        self._interval_seconds: float = 5.0

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    @property
    def entities(self) -> NemoEntitiesClient:
        if self._entities is None:
            raise RuntimeError("entities accessed before on_startup()")
        return self._entities

    async def on_startup(self) -> None:
        from nhx.common.sdk_factory import get_async_platform_sdk
        from nemo_helix_plugin.client.adapter import client_from_platform
        from nemo_helix_plugin.entities.client import AsyncEntitiesClient
        from nemo_helix_plugin.entity_client import NemoEntitiesClient
        from nemo_my_plugin.config import MyPluginConfig

        config = MyPluginConfig.get()
        self._interval_seconds = float(config.controller.interval_seconds)
        sdk = get_async_platform_sdk(as_service="my-deployment", internal=True)
        typed_client = client_from_platform(sdk, AsyncEntitiesClient)
        self._entities = NemoEntitiesClient(typed_client)

    async def on_shutdown(self) -> None:
        logger.info("DeploymentController shutting down.")

    async def list_objects(self) -> list:
        try:
            result = await self.entities.list(MyDeployment, workspace="-")
            return result.data
        except Exception:
            logger.exception("Failed to list deployments")
            return []

    async def reconcile_one(self, obj: object) -> None:
        deployment = obj
        try:
            await self._reconcile_one(deployment)
        except NemoEntityConflictError:
            logger.debug(
                "Optimistic lock conflict on '%s' — will retry next cycle.", deployment.name
            )

    async def _reconcile_one(self, deployment) -> None:
        # State machine: pending → running → deleting → (deleted)
        if deployment.status == "pending":
            await self._start(deployment)
        elif deployment.status == "running":
            await self._verify_running(deployment)
        elif deployment.status == "deleting":
            await self._delete(deployment)
```

## Gotchas

- **ALL dependencies in `on_startup()`, NOT `__init__()`**: `__init__()` runs before the platform is ready. Accessing platform clients there will fail or connect before services are healthy.
- **`internal=True` is required**: Without it, controller polling floods the access log. Always set when building the SDK for background use.
- **`NemoEntityConflictError` = optimistic lock → log debug, skip**: Do NOT log at error level, do NOT retry immediately. Next cycle retries automatically.
- **`list_objects()` returns `[]` on exception**: Raising from `list_objects()` is caught by the default `reconcile()` error handler and logs an exception. Returning `[]` is silent and correct.
- **`interval_seconds` is `@property`, not ClassVar**: ClassVar is set at import time before any config is loaded.

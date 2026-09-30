<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Controller Surface (NemoController)

## NemoController

```python
from nemo_helix_plugin.controller import NemoController

class NemoController(_NamedPlugin):
    name: ClassVar[str]                    # REQUIRED — kebab-case; matches entry-point key
    dependencies: ClassVar[list[str]] = [] # Helix services to wait for before startup

    # Must implement:
    @abstractmethod
    async def list_objects(self) -> list: ...
    # Called once per cycle. Return all objects to reconcile.

    @abstractmethod
    async def reconcile_one(self, obj: object) -> None: ...
    # Called for each item from list_objects(). Errors are caught per-item.

    # May override:
    async def reconcile(self) -> None: ...          # default: list → reconcile_one each
    async def on_startup(self) -> None: ...         # called once before first cycle
    async def on_shutdown(self) -> None: ...        # called after last cycle
    @property
    def is_healthy(self) -> bool: ...               # health check (default: True)
    @property
    def interval_seconds(self) -> float: ...        # default: 10.0
```

## Lifecycle

```
Platform startup
    ↓
NemoControllerAdapter wraps NemoController instance
    ↓
Wait for each name in dependencies (poll /status until services.ready)
    ↓
on_startup() — load config, build entity client, initialize backends
    ↓
Loop begins (runs every interval_seconds):
    reconcile()
        → list_objects()        — fetch entities from store
        → reconcile_one(obj)    — per-item, with error isolation
    ↓
SIGINT/SIGTERM received
    → platform sets stop signal
    → waits for current reconcile() to complete
    → on_shutdown()
```

## Initialization

`__init__()` must **only** set attributes to `None`. The platform is not ready at class-definition time — all async initialization must happen in `on_startup()`.

```python
class MyController(NemoController):
    name = "my-controller"

    def __init__(self) -> None:
        self._entities: NemoEntitiesClient | None = None
        self._interval_seconds: float = 10.0

    @property
    def entities(self) -> NemoEntitiesClient:
        if self._entities is None:
            raise RuntimeError("entities accessed before on_startup()")
        return self._entities
```

## internal=True — why it's required

Without `internal=True`, controller polling floods the entity store's access log — omit it and the store logs every cycle.

```python
async def on_startup(self) -> None:
    from nhx.common.sdk_factory import get_async_platform_sdk
    from nemo_helix_plugin.client.adapter import client_from_platform
    from nemo_helix_plugin.entities.client import AsyncEntitiesClient
    from nemo_helix_plugin.entity_client import NemoEntitiesClient

    sdk = get_async_platform_sdk(as_service="my-controller", internal=True)
    typed_client = client_from_platform(sdk, AsyncEntitiesClient)
    self._entities = NemoEntitiesClient(typed_client)
```

`as_service="my-controller"` sets `X-NHX-Principal-Id: service:my-controller`, granting the service principal access needed to list entities across all workspaces.

## Cross-workspace listing

Controllers typically reconcile all entities platform-wide. Use `workspace="-"` in `list_objects()`:

```python
async def list_objects(self) -> list:
    try:
        result = await self.entities.list(MyEntity, workspace="-")
        return result.data
    except Exception:
        logger.exception("Failed to list entities across workspaces")
        return []  # Return [] on exception — silently skip the cycle
```

`workspace="-"` is a sentinel. Never use it to create entities.

> Exceptions from `list_objects()` propagate out of `reconcile()`. Always catch and return `[]`.

## Optimistic lock conflicts

`NemoEntityConflictError` from `entity_client.update()` in a controller means another process modified the entity between your `list_objects()` and `reconcile_one()` calls. Catch it, log at debug (NOT error), and skip — it will be retried automatically on the next cycle:

```python
from nemo_helix_plugin.entity_client import NemoEntityConflictError

async def reconcile_one(self, obj: object) -> None:
    try:
        await self._reconcile_one(obj)
    except NemoEntityConflictError:
        logger.debug(
            "Optimistic lock conflict on '%s' — will retry next cycle.", obj.name
        )
```

## Overriding reconcile()

Override `reconcile()` entirely for controllers with complex multi-phase logic:

```python
async def reconcile(self) -> None:
    # Phase 1: sync state from external backend
    external_state = await self._fetch_external_state()
    # Phase 2: reconcile entities against external state
    entities = await self.entities.list(MyEntity, workspace="-")
    for entity in entities.data:
        await self._sync_one(entity, external_state)
```

When overriding `reconcile()`, `list_objects()` and `reconcile_one()` are not called — stub them with `raise NotImplementedError`. Use `raise NotImplementedError` rather than `return []` / `pass` — this makes it immediately obvious if these stubs are accidentally called after a future refactor.

## interval_seconds from config

```python
class MyController(NemoController):
    name = "my-controller"

    def __init__(self) -> None:
        self._interval_seconds: float = 10.0  # default

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds  # driven by config, set in on_startup

    async def on_startup(self) -> None:
        from nemo_my_plugin.config import MyPluginConfig
        config = MyPluginConfig.get()
        self._interval_seconds = float(config.controller_interval_seconds)
```

## Minimal example

The `ExampleController` from the example plugin:

```python
import logging
from nemo_helix_plugin.controller import NemoController

logger = logging.getLogger(__name__)

_OBJECTS: list[str] = ["alpha", "beta", "gamma"]


class ExampleController(NemoController):
    """Reference controller showing config loading and reconcile loop patterns."""

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

    async def list_objects(self) -> list:
        """Return the set of objects to reconcile this cycle."""
        return list(_OBJECTS)

    async def reconcile_one(self, obj: object) -> None:
        """Reconcile a single object — here we just log it."""
        logger.debug("ExampleController reconciling object: %s", obj)
```

## Production example

See the **Production State Machine Example** in [`plugin-controller` skill](../.agents/skills/plugin-controller/SKILL.md) for the full `DeploymentController` pattern with state dispatch, `__init__` sentinels, and `on_startup` SDK factory.

## Service / controller decoupling

```
User → POST /deployments → service creates entity (status="pending") → 201
                                                              ↓
                                             controller polls every N seconds
                                             → finds entity with status="pending"
                                             → starts process
                                             → updates entity (status="running")
```

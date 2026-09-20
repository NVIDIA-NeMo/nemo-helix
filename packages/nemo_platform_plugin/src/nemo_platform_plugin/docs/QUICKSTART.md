<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Quickstart: Build Your First NeMo Platform Plugin

A plugin with an HTTP endpoint, a CLI command, a scheduled job, and typed configuration.

## Prerequisites

Python 3.12–3.13, `uv`, `export NMP_BASE_URL=http://localhost:8080` (Steps 2–5), running NeMo Platform (Step 6).

## Step 1: Create the package structure

```
my-plugin/
├── pyproject.toml
└── src/
    └── nmp/
        └── my_plugin/
            ├── service.py
            ├── cli.py
            ├── config.py
            ├── entities.py
            └── jobs/
                └── process.py
```

**No `__init__.py` files** — NeMo Platform plugins use implicit namespace packages. The hatchling config handles this automatically.

```toml
# pyproject.toml
[project]
name = "nemo-my-plugin"
version = "0.1.0"
description = "My NeMo Platform plugin."
requires-python = ">=3.12"
dependencies = ["nemo-platform-plugin"]

[project.entry-points."nemo.services"]
"my-plugin" = "nemo_my_plugin.service:MyService"

[project.entry-points."nemo.cli"]
"my-plugin" = "nemo_my_plugin.cli:MyCLI"

[project.entry-points."nemo.jobs"]
"my-plugin.process" = "nemo_my_plugin.jobs.process:ProcessJob"

[project.entry-points."nemo.controllers"]
"my-plugin-controller" = "nemo_my_plugin.controller:MyController"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/nemo_my_plugin"]
```

Normal third-party plugins only need `hatchling` in `build-system.requires`. `nmp-build-tools` is a first-party build helper for NeMo Platform packages that use `[tool.bundle-package]` to bundle workspace sources into their published wheels.

## Step 2: Add an HTTP service

```python
# src/nemo_my_plugin/service.py
from typing import ClassVar

from fastapi import APIRouter
from nemo_platform_plugin.service import NemoService, RouterSpec


class MyService(NemoService):
    name: ClassVar[str] = "my-plugin"
    dependencies: ClassVar[list[str]] = []

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()

        @router.get("/hello")
        async def hello() -> dict:
            return {"message": "Hello from my plugin!"}

        return [RouterSpec(router, tag="My Plugin")]
```

```bash
uv pip install -e .
nemo services run
```

## Step 3: Add a CLI command

```python
# src/nemo_my_plugin/cli.py
import typer
from nemo_platform_plugin.cli import NemoCLI


class MyCLI(NemoCLI):
    name = "my-plugin"
    description = "My plugin commands."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(help="My plugin commands.")

        @app.command()
        def greet(name: str = typer.Option("world", help="Name to greet.")) -> None:
            """Greet a name."""
            typer.echo(f"Hello, {name}!")

        return app
```

```bash
nemo my-plugin greet --name Alice
# Hello, Alice!
```

## Step 4: Add a job

Declare the spec with Pydantic, implement `run()` for local execution, and override `compile()` for remote execution:

```python
# src/nemo_my_plugin/jobs/process.py
from typing import ClassVar
from nemo_platform_plugin.job import NemoJob
from pydantic import BaseModel


class ProcessSpec(BaseModel):
    input: str = ""


class ProcessJob(NemoJob):
    name: ClassVar[str] = "process"
    description: ClassVar[str] = "Process an input and return a result."
    spec_schema: ClassVar[type[BaseModel]] = ProcessSpec
    container: ClassVar[str] = "cpu-tasks"

    def run(self, config: dict) -> dict:
        cfg = ProcessSpec.model_validate(config)
        return {"status": "done", "result": cfg.input.upper()}

    @classmethod
    async def compile(cls, *, workspace, spec, entity_client, job_name, sdk, profile=None, options=None):
        # Build a PlatformJobSpec here — see JOB.md (Compilation) for details.
        ...
```

The platform auto-generates three CLI verbs per job:

```bash
nemo my-plugin process run --spec '{"input": "hello"}'
# { "status": "done", "result": "HELLO" }

nemo my-plugin process submit --profile default --spec '{"input": "hello"}'
# Posts the job to the plugin service; the cluster runs it.

nemo my-plugin process explain
# Prints the job's schemas and submit route.
```

Mount the routes from your service:

```python
# src/nemo_my_plugin/service.py (add to get_routers())
from nemo_platform_plugin.jobs.routes import add_job_routes
from nemo_my_plugin.jobs.process import ProcessJob

job_router = add_job_routes(ProcessJob)
# Return RouterSpec(job_router, prefix="/v2/workspaces/{workspace}") from get_routers().
# The platform exposes the final route at /apis/my-plugin/v2/workspaces/{workspace}/jobs/process.
```

## Step 5: Add configuration

```python
# src/nemo_my_plugin/config.py
from typing import ClassVar

from pydantic import Field
from nemo_platform_plugin.config import NemoConfig


class MyPluginConfig(NemoConfig):
    plugin_name: ClassVar[str] = "my-plugin"
    plugin_description: ClassVar[str] = "Configuration for my plugin."

    debug: bool = Field(default=False, description="Enable debug logging.")
    greeting: str = Field(default="Hello", description="Greeting word to use.")
```

Override via env var:

```bash
NMP_MY_PLUGIN_DEBUG=true
NMP_MY_PLUGIN_GREETING=Howdy
```

Use in a route:

```python
# src/nemo_my_plugin/service.py (updated)
from nemo_my_plugin.config import MyPluginConfig

@router.get("/hello")
async def hello() -> dict:
    config = MyPluginConfig.get()
    return {"message": f"{config.greeting} from my plugin!", "debug": config.debug}
```

## Step 6: Add entity storage

```python
# src/nemo_my_plugin/entities.py
from nemo_platform_plugin.entity import NemoEntity


class Widget(NemoEntity, entity_type="my_plugin_widget"):
    colour: str
    weight_kg: float = 0.0
```

Use in a route with error handling:

```python
# src/nemo_my_plugin/service.py (entity routes)
from fastapi import APIRouter, Depends, HTTPException
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_my_plugin.entities import Widget
from pydantic import BaseModel


class CreateWidgetRequest(BaseModel):
    name: str
    colour: str
    weight_kg: float = 0.0


def _build_widgets_router() -> APIRouter:
    router = APIRouter()

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
                detail=f"Widget '{body.name}' already exists.",
            ) from exc
        return saved

    @router.get("/widgets/{name}", response_model=Widget)
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

    return router
```

Add to `MyService.get_routers()`:

```python
def get_routers(self) -> list[RouterSpec]:
    return [
        RouterSpec(...),  # existing hello router
        RouterSpec(
            _build_widgets_router(),
            tag="Widgets",
            prefix="/v2/workspaces/{workspace}",
        ),
    ]
```

## Testing without a platform

Use `dependency_overrides` to inject a mock entity client — no platform required. See [SERVICE.md — Testing](SERVICE.md#testing) for the `_make_app` helper pattern.

## Next steps

- [ENTITY.md](ENTITY.md) — entity store deep dive, filtering, pagination, optimistic locking
- [SERVICE.md](SERVICE.md) — complete CRUD patterns, auth, job routes
- [JOB.md](JOB.md) — Pydantic validation, async jobs, container execution, CLI surface
- [CONFIG.md](CONFIG.md) — nested config, priority order, test overrides
- [CONTROLLER.md](CONTROLLER.md) — reconcile loops, service principal client, state machines
- [ARCHITECTURE.md](ARCHITECTURE.md) — discovery, startup sequence, all surfaces

## Common pitfalls

- **`name` mismatch:** `MyService.name = "my_plugin"` but entry-point key is `"my-plugin"` — routing breaks silently (only a warning is logged). They must be identical.
- **Do not depend on `nemo-platform`:** `nemo-platform-plugin` already provides `get_entity_client` and bundles the `nemo_platform` SDK module. Adding `nemo-platform` pulls every platform service into your dependency closure for no gain.
- **Do not add `__init__.py`**: The package directory does not require one. Do not add it.

---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: creating-a-plugin
description: Creates a new NeMo plugin from scratch. Use when starting plugin development, setting up a plugin package, registering surfaces via entry points, or asking how plugins are discovered by the platform. Trigger keywords: create plugin, new plugin, plugin setup, entry-points, plugin structure, get started, plugin discovered, entry point.
---

# Creating a NeMo Plugin

## Quick Start — Minimal Plugin in 5 Steps

**Step 1: Directory tree** (NO `__init__.py` anywhere)

```
my-plugin/
├── pyproject.toml
└── src/
    └── nemo_my_plugin/
        └── service.py
```

**Step 2: Minimal `pyproject.toml`**

```toml
[project]
name = "nemo-my-plugin"
version = "0.1.0"
dependencies = ["nemo-helix-plugin", "nemo-helix"]

[project.entry-points."nemo.services"]
my-plugin = "nemo_my_plugin.service:MyService"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/nemo_my_plugin"]
```

**Step 3: Minimal `NemoService` implementation**

```python
# src/nemo_my_plugin/service.py
from typing import ClassVar
from fastapi import APIRouter
from nemo_helix_plugin.authz import AuthzScope, CallerKind, path_rule
from nemo_helix_plugin.service import NemoService, RouterSpec

scope = AuthzScope("my-plugin")

class MyService(NemoService):
    name: ClassVar[str] = "my-plugin"
    dependencies: ClassVar[list[str]] = []

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()

        @router.get("/hello")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def hello() -> dict:
            return {"message": "Hello from my plugin!"}

        return [RouterSpec(router, tag="My Plugin")]
```

Every plugin route needs an authz rule — the `@scope.read`/`.write` scope gate plus a `@path_rule` (here authenticated but permissionless, `permissions=[]`); see the `plugin-authz` skill for permissions and caller-kinds.

**Step 4: Install editable**

```bash
uv pip install -e .
```

**Step 5: Start and test**

```bash
nemo services run
# Routes now at: http://localhost:8080/apis/my-plugin/hello
```

## pyproject.toml — Complete Template

```toml
[project]
name = "nemo-my-plugin"
version = "0.1.0"
description = "My NeMo Helix plugin."
requires-python = ">=3.11"
dependencies = ["nemo-helix-plugin", "nemo-helix", "pydantic>=2.0.0"]

[project.entry-points."nemo.services"]
my-plugin = "nemo_my_plugin.service:MyService"

# Uncomment surfaces as you add them:
# [project.entry-points."nemo.cli"]
# my-plugin = "nemo_my_plugin.cli:MyCLI"

# [project.entry-points."nemo.jobs"]
# "my-plugin.job-name" = "nemo_my_plugin.jobs.job_name:MyJob"

# [project.entry-points."nemo.functions"]
# "my-plugin.fn-name" = "nemo_my_plugin.functions.fn_name:MyFunction"

# [project.entry-points."nemo.controllers"]
# my-plugin = "nemo_my_plugin.controller:MyController"

# [project.entry-points."nemo.sdk"]
# my-plugin = "nemo_my_plugin.sdk:MySDKResource"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/nemo_my_plugin"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["src"]
```

## Adding Surfaces Incrementally

**Add a CLI command:**

```python
# src/nemo_my_plugin/cli.py
import typer
from nemo_helix_plugin.cli import NemoCLI

class MyCLI(NemoCLI):
    name = "my-plugin"
    description = "My plugin commands."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(help=self.description)

        @app.command()
        def greet(name: str = typer.Option("world")) -> None:
            """Greet a name."""
            typer.echo(f"Hello, {name}!")

        return app
```

**Add a job:**

```python
# src/nemo_my_plugin/jobs/say_hello.py
from typing import ClassVar
from nemo_helix_plugin.job import NemoJob
from pydantic import BaseModel

class SayHelloSpec(BaseModel):
    name: str = "world"

class SayHelloJob(NemoJob):
    name: ClassVar[str] = "say-hello"        # suffix ONLY — NOT "my-plugin.say-hello"
    description: ClassVar[str] = "Greet a name."
    spec_schema: ClassVar[type[BaseModel]] = SayHelloSpec
    container: ClassVar[str] = "cpu-tasks"

    def run(self, config: dict) -> dict:
        cfg = SayHelloSpec.model_validate(config)
        return {"result": f"Hello, {cfg.name}!"}

    @classmethod
    async def compile(cls, *, workspace, spec, entity_client, job_name, sdk, profile=None, options=None):
        # See the plugin-job skill for compile() details.
        ...
```

Entry-point key uses dot: `"my-plugin.say-hello"` under the `nemo.jobs` group. The platform auto-generates `nemo my-plugin say-hello run / submit / explain`. Mount server routes with `add_job_routes(SayHelloJob, authz=AuthzScope("my-plugin"))` from `nemo_helix_plugin.jobs.routes` — the `authz=` kwarg is required, or the generated routes are unruled and fail the OPA bundle build. See the `plugin-job` skill for the full pattern.

**Add a function:**

```python
# src/nemo_my_plugin/functions/greet.py
from typing import ClassVar
from nemo_helix_plugin.function import NemoFunction
from pydantic import BaseModel

class GreetSpec(BaseModel):
    name: str = "world"

class GreetResponse(BaseModel):
    message: str

class GreetFunction(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "greet"                        # suffix ONLY — NOT "my-plugin.greet"
    description: ClassVar[str] = "Say hello to a name."
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec) -> GreetResponse:
        return GreetResponse(message=f"Hello, {spec.name}!")
```

Entry-point key uses dot: `"my-plugin.greet"` under the `nemo.functions` group. The platform auto-generates `nemo my-plugin greet run / submit` (two verbs — no `explain`). Mount the HTTP route inside your `NemoService` with `add_function_routes(GreetFunction, authz=AuthzScope("my-plugin"), permission_description="Invoke the greet function")` from `nemo_helix_plugin.functions.routes` — the `authz=` kwarg is required, or the route is unruled and fails the OPA bundle build. Streaming functions return an `AsyncIterator` (one NDJSON frame per line); non-streaming ones return a value. `run` **must be `async def`** — sync work goes through `await asyncio.to_thread(...)`. See the `plugin-function` skill for the full pattern.

**Add a controller:**

```python
# src/nemo_my_plugin/controller.py
from nemo_helix_plugin.controller import NemoController

class MyController(NemoController):
    name = "my-plugin"
    dependencies = ["entities"]

    async def list_objects(self) -> list:
        return []  # replace with entity client call

    async def reconcile_one(self, obj: object) -> None:
        pass  # drive state machine here
```

## The `name` Class Variable Rule

Every concrete plugin class **must** declare `name` as a `ClassVar[str]`. The value **must exactly match** the entry-point key in `pyproject.toml`.

- `NemoService.name` → entry-point key → URL prefix `/apis/<name>`
- `NemoCLI.name` → entry-point key → CLI subcommand `nemo <name>`
- `NemoController.name` → entry-point key
- `NemoJob.name` → **suffix only** (after the dot in `"plugin.job-name"`)
- `NemoFunction.name` → **suffix only** (after the dot in `"plugin.fn-name"`)

Missing or empty `name` raises `TypeError` at class-definition time. Mismatch at startup logs a warning.

## Implicit Namespace Packages

**No `__init__.py` files**: The package directory does not require `__init__.py`. Do not add one.

Hatchling requires `packages = ["src/nemo_my_plugin"]` — this is critical for namespace package support.

## Entry-Point Key Conventions

| Surface | Key format | Example |
|---|---|---|
| Services | `"my-plugin"` | URL: `/apis/my-plugin` |
| CLI | `"my-plugin"` | CMD: `nemo my-plugin` |
| Controllers | `"my-plugin"` | (same as service name) |
| Jobs | `"my-plugin.job-name"` | dot separator |
| Functions | `"my-plugin.fn-name"` | dot separator |

Jobs and functions use a dot separator so `discover_jobs()["my-plugin.job-name"]` and `discover_functions()["my-plugin.fn-name"]` resolve unambiguously across plugins. Programmatic job execution goes through `NemoJobScheduler.run_local(job_cls, config)`; functions are the runtime themselves — `await fn_cls().run(spec, ...)`.

## Discovery & Fault Isolation

The platform scans entry-point groups at startup. Each group is cached per process:

```python
from nemo_helix_plugin.discovery import discover, discover_entry_points

# discover() loads values (fault-isolated: broken plugin → warning + skip)
# discover_entry_points() is metadata-only (no imports)
```

A broken plugin (import error) logs a warning and is skipped — the platform continues. To disable a plugin, uninstall its package.

**Tests that mock entry-points must clear the cache:**

```python
from nemo_helix_plugin.discovery import discover, discover_entry_points

discover.cache_clear()
discover_entry_points.cache_clear()
```

## Gotchas

- **No `__init__.py`**: The package directory does not need one. Do not add it.
- **`name` must match entry-point key**: For jobs, `NemoJob.name` is the suffix after the dot, not the full key.
- **Both `nemo-helix-plugin` AND `nemo-helix` required**: `nemo-helix-plugin` provides the ABCs; `nemo-helix` provides `get_entity_client` and SDK features.
- **Install with `-e` (editable)**: `uv pip install -e .` — non-editable installs require reinstall on every change.
- **`discover.cache_clear()` in tests**: Any test that mocks entry-points must call both `discover.cache_clear()` and `discover_entry_points.cache_clear()` to avoid stale caches between tests.
- **`packages = ["src/nemo_my_plugin"]` in hatchling config**: Without this, the wheel will not include the `nhx` namespace package correctly.
- **Every route needs an authz rule**: an unruled route fails the OPA bundle build under `on_invalid_plugin=hard_fail` (the platform 502s rather than silently fencing the route). Attach `@scope.read`/`.write` + `@path_rule` to hand-written handlers, or pass `authz=` to `add_job_routes` / `add_function_routes`. See the `plugin-authz` skill.

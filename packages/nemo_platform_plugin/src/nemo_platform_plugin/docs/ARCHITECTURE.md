<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Platform Plugin Architecture

## Surfaces and entry-point groups

Every plugin capability is a "surface" — a typed contract registered via a Python entry-point group. The platform discovers all surfaces at startup by scanning these groups.

| Surface | Entry-point group | Base class | URL / mount point | Platform action |
|---|---|---|---|---|
| **HTTP service** ★ | `nemo.services` | `NemoService` | `/apis/<name>/...` | wraps in `NemoServiceAdapter`, mounts FastAPI router |
| **CLI** ★ | `nemo.cli` | `NemoCLI` | `nemo <name> <cmd>` | calls `get_cli()`, mounts as Typer subcommand |
| **Agent CLI** | `nemo.cli.agents` | `NemoCLI` | `nemo agents <agent> <verb>` | mounts an agent command group under the shared `agents` namespace |
| **Job** ★ | `nemo.jobs` | `NemoJob` | key: `<plugin>.<job>` | auto-generates `run` / `submit` / `explain` CLI verbs; the scheduler drives local runs and remote submission |
| **Controller** ★ | `nemo.controllers` | `NemoController` | (background) | wraps in `NemoControllerAdapter`, runs reconcile loop |
| SDK | `nemo.sdk` | (any class) | `nemo.<name>` on hub | instantiated as attribute on the `NeMo` hub |
| MCP | `nemo.mcp` | `() -> list[dict]` | (MCP tool list) | returns MCP tool definitions |
| Studio | `nemo.studio` | `() -> StudioSpec` | (UI page spec) | returns Studio web-UI page spec |
| Skills | `nemo.skills` | `() -> Path` | (skill files) | returns path to agent skill markdown files |
| Docs | `nemo.docs` | `() -> Path \| dict` | (docs path) | returns path to plugin documentation |
| Executors | `nemo.executors` | Executor class | kind: `slurm`, `k8s`... | resolved by job scheduler cluster kind |

★ = Primary surfaces; most plugins implement one or more of these four.

## Discovery

`discover(group)` scans `importlib.metadata` entry-points for the given group, loads each value (the class), and caches results per-group for the process lifetime. Fault-isolated: a broken plugin logs a warning and is skipped; all others load normally.

Platform wraps each surface:
- `NemoService` → `NemoServiceAdapter` → `FastAPI app.include_router`
- `NemoCLI` → `get_cli()` → Typer subcommand
- `NemoJob` → job scheduler
- `NemoController` → `NemoControllerAdapter` → async reconcile loop

## Agent CLI extensions

Plugins that provide an agent register one `NemoCLI` subclass under
`nemo.cli.agents`. The entry-point key is the agent's name:

```toml
[project.entry-points."nemo.cli.agents"]
"analyst" = "nemo_insights_plugin.analyst.cli:AnalystCLI"
```

`AnalystCLI` must be a `NemoCLI` subclass with `name = "analyst"`. This
registers `nemo agents analyst ...`. Agent names must be unique kebab-case
nouns, and commands beneath an agent must be verbs.

## Startup sequence

1. `NMP_SERVICES` env var consulted — if set, only listed services start
2. All `NemoService.dependencies` are resolved — verify deps are available
3. Each service's `on_startup()` called and **awaited** in dependency order
4. FastAPI routes mounted at `/apis/<name>/...`
5. `NemoController` instances start their reconcile loops — services are fully started by this point, so controller `on_startup()` can safely call entity store APIs
6. `NemoJob` classes registered in the job scheduler

## The `name` rule

`_NamedPlugin.__init_subclass__` runs at class-definition time. It checks that every concrete subclass declares a non-empty `name: ClassVar[str]`.

The `name` must **exactly match** the entry-point key in `pyproject.toml`. A mismatch logs a warning at startup — the entry-point key always wins for routing:

```
nemo.services entry 'my-plugin': class MyService declares name='my_plugin' —
name must match the pyproject.toml entry-point key
```

## URL routing

Formula: `/apis/<name>/<spec.prefix>/<route-path>`

Platform convention: `/apis/<name>/v2/workspaces/{workspace}/<resource>`

Plugin services should follow the same convention for consistency.

## Job entry-point keys

Job entry-point keys use a dot separator: `<plugin-name>.<job-name>`.

`NemoJob.name` must equal the suffix after the first dot:

```toml
"example.say-hello" = "nemo_example_plugin.jobs.say_hello:SayHelloJob"
```

```python
class SayHelloJob(NemoJob):
    name = "say-hello"   # suffix only
```

Dispatch programmatically:

```python
import os

from nemo_platform_plugin.discovery import discover_jobs
from nemo_platform_plugin.scheduler import NemoJobScheduler

job_cls = discover_jobs()["example.say-hello"]
NemoJobScheduler().submit_remote(
    job_cls,
    {"name": "Alice"},
    base_url=os.environ.get("NMP_BASE_URL", "http://localhost:8080"),
    workspace="default",
)
```

## Auto-generated CLI for jobs

At startup, for every plugin that registers both `nemo.cli` and `nemo.jobs`, the platform injects job commands into the plugin's Typer group. Plugin authors write no CLI code for their jobs.

- `submit` delegates to `NemoJobScheduler.submit_remote` — POSTs to the plugin service's per-job endpoint; the cluster executes.
- `explain` delegates to `NemoJobScheduler.explain` — reads schemas locally from the `NemoJob` class.

Jobs can set `generate_legacy_verbs = False` to expose submission as a flat command named after the job instead of a nested `<job> submit` command. The `<job> explain` command remains available for compatibility.

Plugin services mount the matching POST/GET/LIST/DELETE endpoints with the `add_job_routes(job_cls)` helper from `nemo_platform_plugin.jobs.routes` — a one-liner that replaces the multi-arg `job_route_factory(...)` pattern.

## Plugin manifests

`discover_manifests()` assembles a `PluginManifest` (name, version, description) per installed plugin by scanning surface groups — no `nemo.plugins` group needed.

## Test isolation

`discover()` and `discover_entry_points()` are `@functools.cache`d per group for the process lifetime. Tests that mock entry-points must clear both caches. This is **mandatory, not optional** — stale cache state causes tests to interfere with each other:

```python
import pytest
from nemo_platform_plugin.discovery import discover, discover_entry_points, discover_manifests

@pytest.fixture(autouse=True)
def clear_discovery_cache():
    yield
    discover.cache_clear()
    discover_entry_points.cache_clear()
    discover_manifests.cache_clear()
```

## Inter-service auth headers

Every inter-service call carries principal headers. The NeMo Platform SDK factory handles this automatically. Four headers used in NeMo Platform:

| Header | Value |
|---|---|
| `X-NMP-Principal-Id` | `user@example.com` or `service:my-plugin` |
| `X-NMP-Principal-Email` | `user@example.com` |
| `X-NMP-Principal-Groups` | `group1,group2` |
| `X-NMP-Principal-On-Behalf-Of` | `user@example.com` (when a service acts for a user) |

For the controller service-principal pattern, see [CONTROLLER.md](CONTROLLER.md).

## SDK Surface

`nemo.sdk` adds a `nemo.<name>` Python attribute on the `NeMo` hub — for Jupyter notebooks and scripts, not platform runtime.

No abstract base class. Requirements:
- `__init__(self, platform: Any)` — `platform.base_url` provides the service URL
- Use synchronous `httpx.Client` (scripting contexts may have no event loop)
- Sub-resources via lazy `@property`
- `resp.raise_for_status()` — let `httpx.HTTPStatusError` propagate

```python
from typing import Any
import httpx

class MyResource:
    def __init__(self, platform: Any) -> None:
        self._platform = platform

    def _base_url(self) -> str:
        return str(getattr(self._platform, "base_url", "http://localhost:8000")).rstrip("/")

    def create(self, *, name: str, workspace: str = "default") -> dict:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                self._base_url() + f"/apis/my-plugin/v2/workspaces/{workspace}/items",
                json={"name": name},
            )
            resp.raise_for_status()
            return resp.json()
```

Entry-point: `[project.entry-points."nemo.sdk"]`
Hub usage: `nemo = NeMo(base_url="http://localhost:8080"); nemo.my_plugin.create(name="x")`

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Platform Plugins

This directory contains first-party NeMo Platform plugins. Each subdirectory is a standalone Python package that registers one or more surfaces with the platform via entry points.

## Installing a plugin

From the repository root, sync the platform environment and install the plugin package into the same virtual environment:

```bash
uv sync
uv pip install -e plugins/<plugin-name>/
```

Example:

```bash
uv pip install -e plugins/example-plugin/
```

Restart `nemo services run` after installing or uninstalling plugins. Service, controller, and inference middleware entry points are discovered at platform startup.

This is the local-checkout path. To install a **published** plugin, see [Plugins and Skills](https://docs.nvidia.com/nemo-platform/documentation/agents/plugins-and-skills).

## Default bootstrap behavior

`make bootstrap-python` syncs the root uv workspace. Bare `uv sync` and `make bootstrap-python` include the `enabled-plugins` dependency group by default through `tool.uv.default-groups`.

Reference plugins such as `plugins/example-plugin/` are not installed by default.

### Switchyard middleware

`plugins/nemo-switchyard/` is an inference middleware plugin. It registers the `nemo-switchyard` middleware entry point, which can be referenced from a VirtualModel's `request_middleware` or `response_middleware`.

The middleware is installed by default through the root workspace's `enabled-plugins` group. The plugin vendors the required subset of the Switchyard library under `plugins/nemo-switchyard/vendor/switchyard/`, so no separate checkout, `SWITCHYARD_PATH`, or PyPI-shadow workaround is needed.

```bash
uv sync
```

For the canonical setup steps, see [Switchyard Inference Middleware Plugin](nemo-switchyard/README.md).

With the platform running, use `nemo-switchyard` in VirtualModel middleware config:

```json
{
  "name": "nemo-switchyard",
  "config_type": "translate",
  "config": {"target_format": "auto", "enable_stats": false}
}
```

## Uninstalling a plugin

```bash
uv pip uninstall <package-name>
```

The package name is the `name` field in the plugin's `pyproject.toml`, not the directory name.

| Directory | Package name |
|---|---|
| `example-plugin/` | `nemo-example-plugin` |
| `nemo-agents/` | `nemo-agents-plugin` |
| `nemo-anonymizer/` | `nemo-anonymizer-plugin` |
| `nemo-data-designer/` | `nemo-data-designer-plugin` |
| `nemo-evaluator/` | `nemo-evaluator-plugin` |
| `nemo-scaled-evals/` | `nemo-scaled-evals-plugin` (Phase 1 ephemeral; install `-e`) |
| `nemo-experimentalist/` | `nemo-experimentalist-plugin` |
| `nemo-guardrails/` | `nemo-guardrails-plugin` |
| `nemo-insights/` | `nemo-insights-plugin` |
| `nemo-switchyard/` | `nemo-switchyard` |

Example:

```bash
uv pip uninstall nemo-example-plugin
```

## Verifying a plugin is active

```bash
nemo plugins list
```

This scans every surface entry-point group, so it reports plugins that add no CLI command and no HTTP route — inference middleware, jobs, and Studio pages included. It reads package metadata only, so it lists a plugin that the platform failed to import.

Per-surface checks, once the plugin is listed:

```bash
# CLI commands appear under the plugin name:
nemo <plugin-name> --help

# Service routes are mounted at /apis/<plugin-name>:
curl http://localhost:8080/apis/<plugin-name>/health
```

Inference middleware adds no CLI command and no route. To confirm that the running platform loaded it, grep the platform log for `Loaded inference middleware plugin`. See [SETUP.md](../SETUP.md) for that check and the `422 references unknown plugin` signal.

## Writing a new plugin

See `packages/nemo_platform_plugin/` for the public contract. A basic plugin only needs `nemo-platform-plugin` as a dependency — no access to `nmp-common` or platform internals is required.

Entry points point to **classes**, not instances. The platform instantiates each class at startup, which keeps the plugin author out of the construction lifecycle and makes future dependency injection straightforward.

Minimum `pyproject.toml`:

```toml
[project]
name = "nmp-my-plugin"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["nemo-platform-plugin"]

[project.entry-points."nemo.services"]
my-plugin = "nmp.my_plugin.service:MyService"

[project.entry-points."nemo.cli"]
my-plugin = "nmp.my_plugin.cli:MyCLI"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/nmp"]
```

Normal plugin packages only need `hatchling` in `build-system.requires`. `nmp-build-tools` is reserved for first-party packages that declare `[tool.bundle-package]` and need to bundle workspace sources into a published wheel.

Minimum service implementation:

```python
# src/nmp/my_plugin/service.py
from fastapi import APIRouter
from nemo_platform_plugin.service import NemoService, RouterSpec

class MyService(NemoService):
    name = "my-plugin"
    dependencies = []

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()

        @router.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return [RouterSpec(router, tag="My Plugin")]
```

Minimum CLI implementation:

```python
# src/nmp/my_plugin/cli.py
import typer
from nemo_platform_plugin.cli import NemoCLI

class MyCLI(NemoCLI):
    def get_cli(self) -> typer.Typer:
        app = typer.Typer(help="My plugin commands.")

        @app.command()
        def run(model: str) -> None:
            """Run something."""
            typer.echo(f"Running with {model}")

        return app
```

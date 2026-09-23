<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-helix-plugin

[![License](https://img.shields.io/badge/license-Apache_2.0-D22128?style=flat-square)](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12--3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Docs](https://img.shields.io/static/v1?label=docs&message=docs.nvidia.com%2Fnemo-helix&color=76B900&style=flat-square&logo=readthedocs&logoColor=white)](https://docs.nvidia.com/nemo-helix)

Build NeMo Helix plugins in Python.

`nemo-helix-plugin` is the only package plugin authors install. It re-exports every base class, schema, and helper you need to contribute HTTP services, CLI commands, schedulable jobs, background controllers, typed configuration, and entity types to [NeMo Helix](https://pypi.org/project/nemo-helix/). Declare your surfaces in `pyproject.toml`, install the package, and the platform discovers and mounts them at startup — no registration code required.

## What you can build

| Surface | Base class | Entry-point group | Purpose |
|---|---|---|---|
| HTTP service | `NemoService` | `nemo.services` | Contributes FastAPI routers mounted at `/apis/<name>/...` |
| CLI | `NemoCLI` | `nemo.cli` | Contributes `nemo <name> <cmd>` subcommands |
| Job | `NemoJob` | `nemo.jobs` | Contributes schedulable, container-executable jobs. Auto-generates `run` / `submit` / `explain` CLI verbs. |
| Controller | `NemoController` | `nemo.controllers` | Contributes background reconcile-loop controllers |
| Configuration | `NemoConfig` | — | Typed plugin configuration with env var / YAML loading |
| Entity | `NemoEntity` | — | Entity definitions stored in the NeMo Helix entity store |

## Install

```bash
pip install nemo-helix-plugin
```

To run a local NeMo Helix that loads and serves your plugin while you develop, use Python 3.12–3.14 and install [`nemo-helix`](https://pypi.org/project/nemo-helix/) too — it ships the platform services, the `nemo` CLI, and the runtime that wires entity-client injection into your plugin's FastAPI app:

```bash
pip install "nemo-helix[all]"
```

## A minimal plugin

A complete NeMo Helix plugin that contributes one HTTP route — a `pyproject.toml` and a service module:

```toml
# pyproject.toml
[project]
name = "nemo-my-plugin"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["nemo-helix-plugin"]

[project.entry-points."nemo.services"]
"my-plugin" = "nemo_my_plugin.service:MyService"

# [build-system] is required by PEP 517 — without it, pip falls back to
# legacy setuptools. Any modern backend works; hatchling is what NeMo
# Platform itself uses and is the easiest fit for the src/ layout below.
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Required because the plugin uses a src/ layout with implicit namespace
# packages (no __init__.py). Hatchling needs to be told where the source is.
[tool.hatch.build.targets.wheel]
packages = ["src/nemo_my_plugin"]
```

```python
# src/nemo_my_plugin/service.py
from typing import ClassVar

from fastapi import APIRouter
from nemo_helix_plugin.service import NemoService, RouterSpec


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

During development, `nemo-helix-plugin` is the only dependency your plugin needs. To actually run a local platform that loads your plugin, install `nemo-helix[all]` (see [Install](#install) above), then:

```bash
pip install -e .
nemo services run
# GET http://localhost:8080/apis/my-plugin/hello → {"message": "Hello from my plugin!"}
```

That's a complete plugin. Adding a CLI command, a job, a controller, typed configuration, or entities follows the same shape — declare a class, register it under the matching entry-point group, install. See [QUICKSTART](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/QUICKSTART.md) for the full template covering every surface.

## Guides

- [QUICKSTART](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/QUICKSTART.md) — build your first plugin end-to-end (service + CLI + job + controller + config)
- [SERVICE](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/SERVICE.md) — HTTP routes, CRUD patterns, testing
- [JOB](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/JOB.md) — jobs, CLI commands, auto-generated job verbs
- [CONTROLLER](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/CONTROLLER.md) — reconcile loops, state machines, service principal
- [CONFIG](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/CONFIG.md) — typed configuration, env vars, YAML, test overrides
- [ENTITY](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/ENTITY.md) — entity store, CRUD client, pagination, optimistic locking
- [INFERENCE_MIDDLEWARE](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/INFERENCE_MIDDLEWARE.md) — inference request/response middleware, typed bodies, response annotations
- [ARCHITECTURE](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/ARCHITECTURE.md) — discovery, startup, all surfaces, SDK surface

> **Tip:** These guides also ship inside the installed package at `nemo_helix_plugin/docs/`.

## Links

- **This package source:** https://github.com/NVIDIA-NeMo/nemo-helix/tree/main/packages/nemo_helix_plugin
- **NeMo Helix on PyPI:** https://pypi.org/project/nemo-helix/
- **NeMo Helix on GitHub:** https://github.com/NVIDIA-NeMo/nemo-helix
- **NeMo Helix docs:** https://docs.nvidia.com/nemo-helix/
- **Issue tracker:** https://github.com/NVIDIA-NeMo/nemo-helix/issues

## License

`nemo-helix-plugin` is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE). Third-party open-source dependencies have their own licenses; review them before use.

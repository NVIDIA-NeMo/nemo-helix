<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Switchyard Inference Middleware Plugin

A NeMo Inference Middleware plugin wrapping [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) —
a protocol-agnostic request/response router for LLM backends.

## Features

- **Random routing** — distribute requests across multiple backend models
- **Format translation** — convert between OpenAI Chat and Anthropic Messages
- **Streaming support** — preserves async iterators without buffering the full response

## Installation

The plugin wheel is published as **`nemo-switchyard-plugin`** (to avoid colliding
with upstream PyPI `nemo-switchyard`). VirtualModels still reference the
entry-point key **`nemo-switchyard`**.

A snapshot of the Switchyard library is vendored at `plugins/nemo-switchyard/vendor/switchyard/`, so no separate Switchyard checkout, `PYTHONPATH` override, or `SWITCHYARD_PATH` env var is required. The plugin is installed by default through the root workspace's `enabled-plugins` group (`nemo-switchyard-plugin`).

```bash
uv sync

LOG_LEVEL=DEBUG uv run nemo services run \
  --services entities,models,inference-gateway,secrets \
  --controllers models
```

The plugin is discovered at platform startup through the `nemo.inference_middleware` entry point named `nemo-switchyard`. Native `switchyard_rust` is **not** in the default image; `SWITCHYARD_NATIVE_REF` is off by default. Setting that build argument **replaces** `switchyard-vendored` (never beside it). Native `stage_router` / `llm_classifier` need the libsy host from the adapter PR (NVIDIA-NeMo/nemo-platform#2091) **and** rust in the IGW process. May `random_routing` / `translate` remain the default until the ARG is set.

The image builder has no rustc, so experiments must pass a wheel or a complete pip spec (`git+https://…@v0.3.0-rc.2`), not a bare tag. Local overlay after this dist rename:

```bash
# Plugin first, then upstream. Do not reinstall nemo-switchyard-plugin --no-deps after,
# or you drop switchyard_rust. Do not uv add upstream into the worktree lockfile.
uv pip install --no-deps plugins/nemo-switchyard
uv pip install "git+https://github.com/NVIDIA-NeMo/Switchyard.git@v0.3.0-rc.2"
```

On a tree that still names the plugin dist `nemo-switchyard`, the second command uninstalls the plugin. To pin a different May vendor commit, follow [`vendor/switchyard/README.md`](vendor/switchyard/README.md).

## Distribution collision smoke test

`scripts/prove_dist_collision.sh` is a throwaway smoke test that creates isolated
May and native virtual environments under a unique temporary directory. It
requires network access. The native environment also needs rustc when building
from the default upstream tag, or a prebuilt wheel supplied through the normal
package tooling.

```bash
plugins/nemo-switchyard/scripts/prove_dist_collision.sh
```

The default upstream tag is `v0.3.0-rc.2`; override it with
`SWITCHYARD_NATIVE_TAG`. `SWITCHYARD_COLLISION_DIR` is a parent directory; the
script creates a unique `run.XXXXXX` child under it and deletes only that child.

## VirtualModel Configuration

Attach this middleware to a VirtualModel via `MiddlewareCall`:

```json
{
  "request_middleware": [
    {
      "name": "nemo-switchyard",
      "config_type": "random_routing",
      "config": {
        "strong": {"model": "workspace/model-a"},
        "weak": {"model": "workspace/model-b"},
        "strong_probability": 0.5
      }
    }
  ]
}
```

### Config Types

| Type | Purpose | Required Fields |
|------|---------|-----------------|
| `random_routing` | Distribute across models | `strong`, `weak`, `strong_probability` |
| `translate` | Format translation | derived from VM `backend_format` |

### Phases (request vs. response)

Each `nemo-switchyard` entry is authoritative for the list it appears in.
The plugin registers and runs only the matching pipeline:

- listed under `request_middleware` → request pipeline runs (e.g. routing
  decision, format translation of the inbound request).
- listed under `response_middleware` → response pipeline runs (e.g.
  translating the backend's response back to the inbound format).

Calling `process_response` for a config that was only listed under
`request_middleware` (or vice versa) is rejected with `400`.

For full cross-format translation, list `translate` in both `request_middleware` and `response_middleware`. Request-only translation sends the backend a translated request but returns the backend's native response shape to the client.

## Log Output

Switchyard decisions appear in IGW container logs under logger `nemo_switchyard.middleware`:

```
INFO: Switchyard random routing: selected 'workspace/model-b' from ['workspace/model-a', 'workspace/model-b']
```

## Architecture

The middleware imports Switchyard from the vendored snapshot at `plugins/nemo-switchyard/vendor/switchyard/`. Each config type maps to a Switchyard factory class that builds request/response pipelines.

- Request flow: IGW → `process_request()` → routing/translation → backend model
- Response flow: backend → `process_response()` → post-processing → IGW

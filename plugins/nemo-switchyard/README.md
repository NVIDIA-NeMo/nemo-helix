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

A snapshot of the Switchyard library is vendored at `plugins/nemo-switchyard/vendor/switchyard/`, so no separate Switchyard checkout, `PYTHONPATH` override, or `SWITCHYARD_PATH` env var is required. The plugin is installed by default through the root workspace's `enabled-plugins` group.

```bash
uv sync

LOG_LEVEL=DEBUG uv run nemo services run \
  --services entities,models,inference-gateway,secrets \
  --controllers models
```

The plugin is discovered at platform startup through the `nemo.inference_middleware` entry point named `nemo-switchyard`. To pin a different upstream commit, follow the instructions in [`vendor/switchyard/README.md`](vendor/switchyard/README.md).

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
| `random_routing` | Distribute across models (May vendor in the default image) | `strong`, `weak`, `strong_probability` |
| `translate` | Format translation (May vendor) | derived from VM `backend_format` |
| `stage_router` | Native libsy stage routing | `confidence_threshold`, `models.capable`, `models.efficient` |
| `llm_classifier` | Native libsy classifier routing | `base_threshold`, `models.judge`, `models.capable`, `models.efficient` |

`stage_router` and `llm_classifier` require the native `switchyard_rust` bindings
in the IGW process. The default platform venv and 0.7 image do **not** install
them (May `switchyard.lib` would collide with upstream `nemo-switchyard`).
VirtualModel upsert returns **HTTP 400** when those types are used without
`switchyard_rust`. Do not `uv add` upstream `nemo-switchyard` into the platform
venv; that replaces May `switchyard` and silently breaks `translate`.

Judge / classifier HTTP uses `get_inference_url_and_model` (provider-direct) plus
the provider's cached secret and extra headers. Caller request headers are not
forwarded. Do not point the judge at a VirtualModel id — that re-enters this
middleware. Native `run_stream` converts OpenAI Chat Completions to Switchyard's
normalized request/response IR; RC2 has no Python translator for that hop.

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

The middleware imports May Switchyard from the vendored snapshot at `plugins/nemo-switchyard/vendor/switchyard/` for `random_routing` and `translate`. Native `stage_router` / `llm_classifier` lazy-import `switchyard_rust.libsy` only when that package is installed.

- Request flow: IGW → `process_request()` → routing/translation → backend model
- Response flow: backend → `process_response()` → post-processing → IGW

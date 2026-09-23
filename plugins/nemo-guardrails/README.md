<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Guardrails Inference Middleware Plugin

A NeMo Inference Middleware plugin that runs input and output rails on chat-completions traffic handled by the Inference Gateway (IGW). The plugin can be configured to apply **input** rails before the backend model is invoked and **output** rails on the model response (non-streaming or streaming).

## Features

- **Entity-backed configs** — reference a persisted `GuardrailConfig` via `MiddlewareCall.config_id` (`workspace/name`)
- **Inline configs** — embed a guardrails payload in `MiddlewareCall.config` for development and tests
- **Input / output rails** — maps to NeMo Guardrails rail phases: `process_request` runs **input** flows; `process_response` runs **output** flows (skipped automatically when those flows are absent)
- **Guardrail LLM routing** — the plugin automatically resolves base URLs for models in a guardrail config to IGW's OpenAI-compatible route.
- **Stabilization and caching** — Running rails requires constructing an **`LLMRails`** object for a guardrail config, which is blocking and slow. The plugin normalizes each config (see below), then **reuses** cached `LLMRails` instances from in-memory pools instead of paying init on every request.
  - **Stabilization** — Validates the config payload and transforms it into the `nemoguardrails` library's expected shape (including resolving task-model base URLs against IGW).
  - **Pools** — Identical normalized configs key an LRU-bounded set of pools. Each request **leases** an instance, runs rails, then returns it; shared state is **cleared** before the next lease so requests do not see each other's data.
  - **`config_id`** — GuardrailConfigs referenced by a `config_id` are cached by (`workspace`, `name`, `updated_at`). If the config is updated, we expect a new `updated_at`, so the plugin picks up the updated data.
  - **Inline `config`** — With no stored entity revision to key on, stabilization runs **every** request. Identical configs are cached by content.
  - **Cache warming** — On VirtualModel upsert, the **`on_virtual_model_upserted`** hook pre-builds pools for the guardrail configs referenced on that VM to avoid building the `LLMRails` instance on the inference-path.

## Installation

**`nemo-guardrails-plugin` is wired into the workspace and listed on the root project**, so a **`uv sync` at the repo root** installs the plugin. You'd only need to explicitly install the plugin via `uv pip install -e plugins/nemo-guardrails` when you are not using that root workspace (or you intentionally want only this package).

The plugin registers under the `nemo.inference_middleware` entry point **`nemo-guardrails`** (see `pyproject.toml`).

## VirtualModel configuration

Attach the plugin to a VirtualModel with `MiddlewareCall.name` **`nemo-guardrails`** and `config_type` **`guardrail_config`**.

Provide **either** inline `config` **or** a `config_id` that references a GuardrailConfig created via the Guardrails service.

### Entity reference (`config_id`)

```json
{
  "request_middleware": [
    {
      "name": "nemo-guardrails",
      "config_type": "guardrail_config",
      "config_id": "my-workspace/my-guardrail-config"
    }
  ],
  "response_middleware": [
    {
      "name": "nemo-guardrails",
      "config_type": "guardrail_config",
      "config_id": "my-workspace/my-guardrail-config"
    }
  ]
}
```

### Inline config

To use an inline config, ensure it matches the `RailsConfig` shape.

```json
{
  "request_middleware": [
    {
      "name": "nemo-guardrails",
      "config_type": "guardrail_config",
      "config": {
        "name": "dev-inline",
        "rails": {
          "input": { "flows": ["your input flow names"] },
          "output": { "flows": ["your output flow names"] }
        },
      }
    }
  ]
}
```

Optional `name` inside `config` is used only as an inline diagnostic label in logs (`<inline:…>`).

IGW resolves configs **when it builds or refreshes the VirtualModel cache**, rather than at inference call time. The resolved config is passed into **`process_request`** / **`process_response`** as **`middleware_config`**.

- **When is `get_middleware_config` invoked?** On **VirtualModel cache load/refresh** (same periodic schedule as the model cache) and when the **VirtualModel API** validates a **create/update**. It is not called on the inference path.
- **What does `get_middleware_config` do?** The Guardrails plugin **reads `GuardrailConfig`** from the platform for the given `workspace/name` and returns rails data (and entity fields such as **`updated_at`**) back to IGW.
- **How do `GuardrailConfig` updates reach `process_request` / `process_response`?** These hooks receive the **`middleware_config`** returned from `validate_middleware_config`. If a config is updated (i.e. the **`updated_at`** field changes on the entity), the **next cache refresh** runs **`get_middleware_config`** again for VMs that point at that config.

### Phases (request vs response)

- **`request_middleware`** — runs **input** rails when `rails.input.flows` is non-empty
- **`response_middleware`** — runs **output** rails when `rails.output.flows` is non-empty

For paths that involve both checks, list **two** `MiddlewareCall`s with the same `config_id` (or mirrored inline configs) — one under `request_middleware` and one under `response_middleware`.

Output rails require **`n`** not greater than **`1`** in the completion request when output flows are configured. Streaming with output rails requires `rails.output.streaming.enabled=true` when a streaming subsection is supplied (otherwise the plugin rejects the path with **`400`**).

## Running tests

```bash
cd plugins/nemo-guardrails
uv run pytest
```

Unit tests mock the platform SDK; integration tests may build real `LLMRails` instances where noted in `tests/integration/`.

## Further reading

- [INFERENCE_MIDDLEWARE.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/INFERENCE_MIDDLEWARE.md) — middleware contract, `MiddlewareCall`, `VirtualModel`

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example Plugin

The `nemo-example-plugin` is a reference implementation that demonstrates NeMo Helix plugin surfaces in a single installable package.

## What this demonstrates

- **NemoService** — full entity-backed CRUD for `ExampleItem` entities, with `NemoFilter` (deepObject query syntax), `NemoListResponse` pagination, entity objects returned directly as responses, and complete error handling (404/409)
- **NemoCLI** — minimal CLI implementation with one `hello` command
- **Plugin SDK mounting** — `client.example.hello(...)` via a `nemo.sdk` entry point
- **NemoJob** — `SayHelloJob` registered under `"example.say-hello"`, showing the entry-point key convention
- **NemoController** — `ExampleController` showing config loading in `on_startup()` and the reconcile loop pattern
- **Plugin seed job** — `ExampleSeedJob` registered under `nemo.seed`, creating a default example entity during platform seeding
- **NemoConfig** — `ExampleConfig` with two typed fields, driven by `NHX_EXAMPLE_*` env vars
- **NemoInferenceMiddleware** — `ExampleInferenceMiddleware` (keyword content filter) demonstrating the full middleware interface: inline config, `config_id` entity references, `ImmediateResponse` to short-circuit the proxy, and response redaction

## Running the example

```bash
# Install the plugin (editable mode required for development)
uv pip install -e .

# Start the platform with this plugin loaded
nemo services run
```

Routes available after startup:

```
GET  /apis/example/hello/{name}                                          (no platform required)
POST /apis/example/v2/workspaces/{workspace}/items                       (requires running platform)
GET  /apis/example/v2/workspaces/{workspace}/items                       (requires running platform)
GET  /apis/example/v2/workspaces/{workspace}/items/{name}                (requires running platform)
PATCH /apis/example/v2/workspaces/{workspace}/items/{name}               (requires running platform)
DELETE /apis/example/v2/workspaces/{workspace}/items/{name}              (requires running platform)
POST /apis/example/v2/workspaces/{workspace}/middleware-configs          (requires running platform)
GET  /apis/example/v2/workspaces/{workspace}/middleware-configs          (requires running platform)
GET  /apis/example/v2/workspaces/{workspace}/middleware-configs/{name}   (requires running platform)
PATCH /apis/example/v2/workspaces/{workspace}/middleware-configs/{name}  (requires running platform)
DELETE /apis/example/v2/workspaces/{workspace}/middleware-configs/{name} (requires running platform)
```

The `/hello/{name}` route works standalone. Entity (`/items`) routes require a running platform with `NHX_BASE_URL` set.

## Running tests

```bash
uv run pytest
```

Tests run entirely without a platform — the entity client is replaced with an `AsyncMock`.

## Code walkthrough

| File | What it shows |
|---|---|
| `service.py` | `NemoService` with 5-operation CRUD; `NemoFilter` with deepObject syntax; `NemoListResponse` pagination; `PaginationData` conversion from `PaginationInfo`; 404/409 error handlers on every route |
| `cli.py` | `NemoCLI` minimal implementation; `get_cli()` returning a single-command Typer app |
| `entities.py` | `NemoEntity` subclass with `entity_type="example_item"`; plugin-scoped naming |
| `schema.py` | `NemoListResponse[ExampleItem]` type alias; `NemoFilter` subclass with `extra="forbid"`; request body models |
| `config.py` | `NemoConfig` with two typed fields; `plugin_name` and `plugin_description` ClassVars |
| `controller.py` | `NemoController` minimal implementation; `on_startup()` loading config; `interval_seconds` as `@property` |
| `core.py` | Pure business logic with no platform dependency; service and CLI are thin wrappers around this |
| `sdk.py` | Exports a `NemoPluginSDKResources` instance for sync/async resources mounted as `client.example` |
| `seed_job.py` | `NemoSeedJob` implementation used by platform seed discovery |
| `jobs/say_hello.py` | `NemoJob` minimal implementation; `name = "say-hello"` (suffix only, not full key) |
| `middleware_config.py` | `NemoEntity` subclass for middleware config; `entity_type="example_middleware_config"` scoped to this plugin |
| `middleware.py` | `NemoInferenceMiddleware` full implementation: inline config, `config_id` entity fetching, `ImmediateResponse` short-circuit, response redaction, `ExampleMiddlewareConfigData` working type separation |
| `middleware_service.py` | CRUD router for `ExampleMiddlewareConfig` — required when using `config_id` in VirtualModel configs |
| `tests/test_service.py` | `_make_app(mock_client)` pattern; `dependency_overrides`; `TestClient`; all 5 CRUD routes tested; 404 and 409 error paths verified |
| `tests/test_inference_middleware.py` | Cache mock via `MagicMock(spec=InferenceMiddlewareCacheAccessor)`; entity client patching at `nemo_helix_plugin.entity_client`; all middleware hooks tested |

## Key patterns to study

- **Entity type naming:** `entity_type="example_item"` — always plugin-scoped to avoid cross-plugin collisions
- **`PaginationInfo` → `PaginationData` conversion:** `entity_client.list()` returns `PaginationInfo`; the HTTP response layer uses `PaginationData`; convert with `PaginationData.model_validate(result.pagination.model_dump())`
- **`isinstance(filter, dict)` check:** `make_filter_obj_dep` may return a raw dict for wildcard filters — always check before calling `.model_dump()`
- **Entities as responses:** `ExampleItem` (the entity class) is used directly as `response_model` — no separate response class needed. `EntityBase` exposes `id`, `created_at`, and other metadata as computed fields that serialize automatically.
- **`_make_app` test helper:** builds a FastAPI app with the real routers but injects a mock entity client via `dependency_overrides` — no platform required for unit tests
- **Config shared across components:** both `service.py` and `controller.py` call `ExampleConfig.get()` independently; they receive the same cached singleton without any explicit wiring

## Full documentation

- [QUICKSTART.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/QUICKSTART.md)
- [SERVICE.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/SERVICE.md)
- [CLI.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/CLI.md)
- [JOB.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/JOB.md)
- [CONTROLLER.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/CONTROLLER.md)
- [CONFIG.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/CONFIG.md)
- [ENTITY.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/ENTITY.md)
- [ARCHITECTURE.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/ARCHITECTURE.md)
- [INFERENCE_MIDDLEWARE.md](../../packages/nemo_helix_plugin/src/nemo_helix_plugin/docs/INFERENCE_MIDDLEWARE.md)

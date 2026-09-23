<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# IGW Middleware Plugin Test Harness

Integration test infrastructure for `NemoInferenceMiddleware` plugins.

Tests exercise the real IGW + Models request pipeline against a single
`pytest_httpserver` socket that stands in for the upstream NIM. Plugin
code (`process_request`, `process_response`, `process_post_response`)
runs with its production implementation — the only mock is the upstream
model provider itself, which every offline test needs.

The heavy ASGI stack (FastAPI app, SQLite-backed entity store, dependency
wiring, `/health/ready` polling, workspace seeding) is built **once per
test file** and shared across every test in the module. Per-test concerns
(mock-NIM handler mount, post-response list reset, entity teardown) still
run per test. See [Module scope and xdist](#module-scope-and-xdist) for
the resulting pytest command-line constraints.

## Quick start

### 1. Re-export the fixtures in your plugin's `conftest.py`

```python
# plugins/<your-plugin>/tests/integration/conftest.py
from nhx.core.inference_gateway.testing.fixtures import (
    _igw_app_context,
    _igw_extra_services,
    igw_plugin_harness,
)

__all__ = ["_igw_app_context", "_igw_extra_services", "igw_plugin_harness"]
```

`_igw_app_context` and `_igw_extra_services` are module-scoped
fixtures that `igw_plugin_harness` depends on; pytest needs them in
the same conftest scope to resolve the dependency chain. Add
`_igw_loopback_context` too if you use `igw_loopback_harness`.

#### Mounting extra services

Override `_igw_extra_services` in your conftest to mount additional
services on the module app (e.g. `GuardrailsService` for entity-backed
guardrail-config tests):

```python
@pytest.fixture(scope="module")
def _igw_extra_services() -> tuple[ServiceFactory, ...]:
    return (GuardrailsService,)
```

### 2. Write a test

```python
from nhx.core.inference_gateway.testing.harness import IGWPluginHarness
from nhx.testing.mock_chat_completions import ChatCompletion, chat_completion


def test_safe_input_reaches_backend(igw_plugin_harness: IGWPluginHarness) -> None:
    h = igw_plugin_harness

    # Queue a canned response for the upstream model.
    h.mock_chat_completions("my-model", responses=[ChatCompletion(body=chat_completion(content="Hello!"))])

    # Create a provider whose host_url points at the mock NIM.
    h.add_provider(workspace="default", served_models={"my-model": "my-model"})

    # Register your plugin and create a VirtualModel that uses it.
    with h.load_plugin("nemo-my-plugin"):
        h.add_virtual_model(
            workspace="default",
            name="my-vm",
            default_model_entity="default/my-model",
            request_middleware=[{"name": "nemo-my-plugin", "config_type": "...", "config": {}}],
        )

        response = h.chat_completions(
            workspace="default",
            body={"model": "my-vm", "messages": [{"role": "user", "content": "hi"}]},
        )

    h.assert_called_once("my-model")
    assert response["choices"][0]["message"]["content"] == "Hello!"
```

### 3. Choose a fixture

| Fixture | When to use |
|---|---|
| `igw_plugin_harness` | Default. No real port for IGW; plugin outbound HTTP goes straight to the mock NIM via `nim_base_url`. |
| `igw_loopback_harness` | Factory for tests where plugin outbound HTTP needs to traverse IGW (e.g. the plugin calls `get_openai_compatible_inference_url_and_model` and the returned URL must be reachable). Call as `h = igw_loopback_harness()` — passing extra services raises `TypeError`; use `_igw_extra_services` instead. Costs a (module-scoped) uvicorn thread plus a per-test `aiohttp.ClientSession` override (scoped to loopback tests only, so plain `igw_plugin_harness` tests in the same module aren't affected). |

### 4. Choose a plugin registration method

| Method | When to use |
|---|---|
| `load_plugin(name)` | Plugin is pip-installed. Discovers via the `nemo.inference_middleware` entry-point group — same path as production. **Preferred.** |
| `use_plugin(name, instance)` | Plugin is not installable (workspace-only), or you need a `MagicMock(spec=...)`. |

Both default to `call_lifecycle=True` (runs `on_startup` / `on_shutdown`).
Async variants: `aload_plugin`, `ause_plugin`.

## What is real

These run the same code as production:

1. **IGW + Models FastAPI apps** — same constructors, routers, dependency wiring.
2. **Full IGW request pipeline** — `virtual_model_proxy` → request middleware → proxy step → response middleware → post-response fire-and-forget.
3. **Plugin code** — `process_request` / `process_response` / `process_post_response` are the production methods.
4. **Real HTTP from IGW to the upstream** — `aiohttp` connects to `pytest_httpserver` over a real socket. Header dropping, body framing, content-length all behave as in production.
5. **Real HTTP from plugin outbound calls** — plugin-originated requests (e.g. Guardrails' rail calls) terminate at the same socket.
6. **SDK clients** — `NeMoHelix` (sync) and `AsyncNeMoHelix` (async) via `httpx.ASGITransport`.
7. **Entity store** — in-memory; entities created via the SDK persist and are read back during cache refreshes.
8. **Cache refresh** — `refresh_virtual_model_cache` and `refresh_model_cache` with real implementations.
9. **Middleware config pre-resolution** — `validate_middleware_config` runs for every VM with middleware entries.

## What is mocked or substituted

1. **The upstream NIM** — `MockChatCompletionsHandler` serves canned responses. The only unavoidable mock.
2. **SDK transport** — `httpx.ASGITransport` (no real port for IGW). The loopback variant adds a real port.
3. **Plugin discovery** — bypassed when using `use_plugin`. Use `load_plugin` for production parity.
4. **`get_platform_config()`** — patched in the loopback variant so the resolver returns the loopback URL.
5. **`global_http_client`** — replaced with per-request sessions in the loopback variant (loop-binding workaround).
6. **Passthrough VM auto-creation** — the `provider_reconciler` doesn't run. Tests needing the resolver must create passthrough VMs manually.
7. **Background cache-refresh task** — disabled (`refresh_model_cache_interval_sec=0`) so the 3-second loop can't fire between tests in a module and re-populate the cache with stale rows. Tests refresh synchronously inside `add_provider` / `add_virtual_model`.
8. **Authorization** — disabled by default (`auth_enabled=False`).

## What cannot be tested

- **Multi-worker cache consistency** — caches are process-local globals.
- **Production-typical timings** — cache cold-start, background refresh races.
- **Real upstream error shapes** — real NIMs return varying error envelopes; the mock returns controlled shapes.
- **OPA / authz** — disabled by default; no integration test exercises it.
- **Rate limiting, retries, circuit breaking** — none of these layers exist in the harness.

## API reference

### Setup

| Method | Description |
|---|---|
| `add_provider(workspace, served_models, ...)` | Register a `ModelProvider` routed at the mock NIM. Call **before** `add_virtual_model`. Tracked for entity-store cleanup. |
| `add_virtual_model(workspace, name, ...)` | Create a `VirtualModel` and refresh caches so it routes immediately. Tracked for entity-store cleanup. |
| `create_secret(workspace, name, value, ...)` | Create a Secret via the SDK and track it for harness cleanup. Use this instead of `harness.sdk.secrets.create(...)` so the secret is deleted between tests in a module-scoped fixture. |
| `mock_chat_completions(model, responses)` | Queue mock responses for a model. Responses are consumed in order; the last is reused if drained. |
| `load_plugin(name)` / `use_plugin(name, instance)` | Register a plugin (context manager). |
| `refresh_caches()` | Full model + VM cache refresh. Needed when `api_key_secret_name` is set on a provider. |

### Workspace

`harness.workspace` — the workspace the module-scoped fixture seeded
(`"default"` today). Use this instead of hardcoding `"default"` in
test bodies; the day the harness moves to per-test workspaces, only
the fixture changes.

### Inference

| Method | Description |
|---|---|
| `chat_completions(workspace, body)` | Non-streaming chat completion via the SDK. |
| `stream_chat_completions(workspace, body)` | Streaming chat completion. Returns parsed SSE chunks (`list[dict]`) for `text/event-stream` responses, or the raw JSON body (`dict`) when a plugin short-circuits with an immediate response. |
| `achat_completions(workspace, body)` | Async sibling of `chat_completions`. |

### Assertions

| Method | Description |
|---|---|
| `assert_called_once(model)` | Model received exactly one request. |
| `assert_call_count(model, n)` | Model received exactly `n` requests. |
| `assert_no_calls_to(model)` | Model received zero requests. |
| `assert_call_order([m1, m2, ...])` | Models were called in this exact sequence. |
| `assert_request_messages_contain(model, substring, *, index=0)` | The `index`-th request's messages contain `substring`. |
| `assert_request_body_for(model, predicate, *, index=0)` | `predicate(body)` is true for the `index`-th request. |
| `assert_request_path_for(model, path, *, index=0)` | The `index`-th request arrived on `path`. |
| `assert_request_headers_contain(model, header, value=None, *, index=0)` | The `index`-th request carries `header` (optionally matching `value`). |
| `requests_for(model)` | Return all `RecordedRequest` objects for `model`. |

### Post-response

| Method | Description |
|---|---|
| `aflush_post_response()` | Await all fire-and-forget post-response tasks scheduled so far. Must be called from an async test after `achat_completions`. |

### Mock response types

Defined in `nhx.testing.mock_chat_completions`:

| Type | Description |
|---|---|
| `ChatCompletion(body, status_code=200)` | Non-streaming JSON response. |
| `ChatCompletionStream(chunks, status_code=200)` | Streaming SSE response. |
| `ErrorResponse(status_code, body)` | Error response (status >= 400). |
| `chat_completion(content, model, ...)` | Builder for a non-streaming response body. |
| `chat_completion_chunk(content, model, ...)` | Builder for a single SSE chunk body. |

## Module scope and xdist

The expensive ASGI stack is wrapped in the module-scoped
`_igw_app_context`. Without that, every parametrised test pays the
full ~3–10s build cost; with it, the build amortises across the file.

**xdist requirement.** Only `--dist=loadfile` (one file per worker) and
`--dist=loadscope` (one fixture scope per worker) preserve module
scope. The default `--dist=load` distributes individual tests across
workers, so each worker rebuilds the app from scratch and defeats the
optimisation. Run integration tests with:

```bash
uv run --frozen pytest plugins/<plugin>/tests/integration --dist=loadfile -n auto
```

`loadfile` is the safer default — it also keeps every test in a file
on the same worker, matching the fixture lifecycle exactly.

## Entity teardown across tests

The harness tracks every entity it creates and deletes them on
teardown in FK order: virtual models → providers → secrets. Each
delete is guarded so one failure can't mask the test's real failure.
After deletion the in-memory caches are rebuilt so the next test's
`add_provider` doesn't see ghost `ModelProviderInfo` rows.

**Only entities created through the harness are tracked.** A direct
`harness.sdk.<entity>.create(...)` call leaks across tests under
module scope, and may then be picked up by the next
`refresh_model_cache` — triggering `notify_upserted` on a dead VM, or
making `add_provider` see stale provider rows.

For secrets, use `harness.create_secret(...)`. For other entity types
you need to create outside the harness, either append to the relevant
tracking list yourself (e.g. `harness._secrets.append((ws, name))`) or
delete in an explicit `try/finally` around the test body.

## Plugin lifecycle and shared SDK clients

`use_plugin` / `load_plugin` run the plugin's `on_startup` on enter
and `on_shutdown` on exit. The catch with module scope: the shared
SDK HTTP client now lives across tests, so a plugin's `on_shutdown`
calling `await sdk.close()` (as `nemo-guardrails` does) would close
the shared client and break every later test in the module.

The module fixture monkey-patches the shared client's `aclose` to a
no-op for the module's lifetime. `ASGITransport` is in-process so
nothing actually leaks. Plugin authors don't need to do anything
special — `on_shutdown` still runs; only the close is intercepted.

If your plugin owns separate resources (custom pools, background
tasks, on-disk caches), close those normally — only the shared SDK's
close is intercepted.

## Limitations under module scope

A few patterns that worked under function scope quietly break under
module scope:

* **Function-scoped autouse `monkeypatch` fixtures that need to
  affect service startup.** `_igw_app_context` builds the app —
  including every service's `on_startup` — before any function-scoped
  fixture runs, so `monkeypatch.setenv` in a per-test autouse fixture
  lands too late. Set the value at conftest import time
  (`os.environ.setdefault(...)` at module level) or in a
  `scope="module", autouse=True` fixture. The `nemo-guardrails`
  conftest's `HF_HUB_OFFLINE` setup is the worked example.
* **Direct `harness.sdk.<entity>.create(...)` calls** — leaks; see
  [Entity teardown](#entity-teardown-across-tests).
* **Per-call extra services to `igw_loopback_harness`** — now raises
  `TypeError`. Override `_igw_extra_services` instead.
* **Entry-point-registered plugins won't get
  `on_virtual_model_destroyed`** on teardown — only `registry.evict`
  runs, so any per-VM state the plugin tracks leaks across tests in
  the module. Register such plugins per-test via
  `harness.use_plugin` / `harness.load_plugin` so the plugin instance
  is discarded with the test.
* **Class-level state on plugins under `load_plugin`** is shared
  across the module. `load_plugin` builds a fresh instance per test,
  so instance state is fine — keep caches on the instance, not on
  the class.

## File layout

```
nhx.core.inference_gateway.testing/
├── harness.py        # IGWPluginHarness, IGWLoopbackHarness
├── fixtures.py       # igw_plugin_harness, igw_loopback_harness (pytest fixtures)
├── _loopback.py      # serve_app_in_thread, per_request_http_client, override_platform_base_url
└── README.md         # this file

nhx.testing/
└── mock_chat_completions.py   # MockChatCompletionsHandler, ChatCompletion, ChatCompletionStream, etc.
```

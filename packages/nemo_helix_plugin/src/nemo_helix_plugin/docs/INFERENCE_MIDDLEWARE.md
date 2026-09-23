<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference Middleware Plugins

Inference middleware plugins run in-process inside the Inference Gateway (IGW).
They intercept every inference request routed through a **VirtualModel** and can:

- **Mutate the request** before it reaches the backend model provider (e.g. keyword filtering, A/B routing, prompt augmentation)
- **Short-circuit the proxy** by returning a response directly, skipping the backend entirely (e.g. a cache hit, a rule-based refusal)
- **Mutate the response** before it reaches the caller (e.g. PII redaction, output sanitisation)
- **Fire-and-forget post-response work** without blocking the caller (e.g. logging, analytics)

---

## 1. How plugins are discovered

Register your middleware class under the `nemo.inference_middleware` entry-point group in `pyproject.toml`:

```toml
[project.entry-points."nemo.inference_middleware"]
"nemo-my-plugin" = "nemo_my_plugin.middleware:MyMiddleware"
```

The entry-point key (`"nemo-my-plugin"`) is the **plugin identity** — it is what
`MiddlewareCall.name` references in VirtualModel configs.  IGW discovers all
registered classes at startup, instantiates them, injects platform cache access,
and calls `on_startup()`.

---

## 2. The `NemoInferenceMiddleware` ABC

Subclass `NemoInferenceMiddleware` and override the hooks you need:

```python
from nemo_helix_plugin.inference_middleware import (
    NemoInferenceMiddleware,
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
    RequestResult,
    ResponseResult,
    ImmediateResponse,
    InferenceMiddlewareError,
    VirtualModel,
)

class MyMiddleware(NemoInferenceMiddleware):

    async def on_startup(self) -> None:
        """Load ML models, build HTTP clients, validate config at startup."""

    async def on_shutdown(self) -> None:
        """Release resources on graceful shutdown."""

    async def on_virtual_model_upserted(self, virtual_model: VirtualModel) -> None:
        """Pre-warm per-VirtualModel resources (e.g. load a classifier for this VM)."""

    async def on_virtual_model_destroyed(self, virtual_model: VirtualModel) -> None:
        """Release resources held for this VirtualModel."""

    async def get_middleware_config(self, config_type: str, config_id: str):
        """Fetch a stored config entity from your plugin's entity store."""

    async def validate_middleware_config(self, config_type: str, config) -> ...:
        """Validate and coerce a config object — called at cache-build time, never per-request."""

    async def process_request(self, ctx: InferenceMiddlewareContext, request: InferenceRequest, middleware_config) -> RequestResult:
        """Modify the request before proxying — or short-circuit with ImmediateResponse."""

    async def process_response(self, ctx: InferenceMiddlewareContext, response: InferenceResponse, middleware_config) -> ResponseResult:
        """Modify the response before returning to the caller."""
```

All hooks default to no-ops.  Implement only what your plugin needs.

---

## 3. VirtualModels and MiddlewareCalls

A **VirtualModel** is an entity stored in IGW that maps a logical model name to a
backend model entity and ordered middleware pipelines.  Each pipeline entry is a
**MiddlewareCall**:

```json
{
  "name": "nemo-my-plugin",
  "config_type": "my_plugin_config",
  "config": { "threshold": 0.8 }
}
```

Fields:

| Field | Required | Description |
|---|---|---|
| `name` | ✓ | Entry-point key of the plugin — must match `pyproject.toml` |
| `config_type` | ✓ | Discriminator string — tells the plugin which schema applies |
| `config` | ✗ | Inline config dict. Mutually exclusive with `config_id` |
| `config_id` | ✗ | `"workspace/name"` reference to a stored entity. Mutually exclusive with `config` |

### Inline config vs. config_id

**Use `config` (inline)** when:
- The config is simple and specific to this VirtualModel
- No sharing or versioning is needed
- You don't want to manage a separate entity

**Use `config_id` (entity reference)** when:
- Multiple VirtualModels share the same config
- Operators need to update the config without editing every VirtualModel
- You want automatic propagation — IGW re-resolves on every polling cycle when the entity's `updated_at` changes, without any VirtualModel edit

---

## 4. Implementing `config_id` support

When `MiddlewareCall.config_id` is used, the referenced entity must exist somewhere in the entity store.  Your plugin must:

**Step 1 — Define a config entity** (extends `NemoEntity`, not `EntityBase` directly):

```python
from nemo_helix_plugin.entity import NemoEntity

class MyPluginConfig(NemoEntity, entity_type="my_plugin_config"):
    """Stored config entity for MyMiddleware.

    entity_type must match MiddlewareCall.config_type.
    Use a plugin-scoped name to avoid cross-plugin collisions.
    """
    threshold: float = 0.8
    strong_model: str = "default/llama-70b"
    weak_model: str = "default/llama-8b"
```

**Step 2 — Expose a CRUD API** so operators can create and update configs:

```python
# In your NemoService — full CRUD for MyPluginConfig
# POST /v2/workspaces/{workspace}/my-plugin-configs
# GET  /v2/workspaces/{workspace}/my-plugin-configs
# GET  /v2/workspaces/{workspace}/my-plugin-configs/{name}
# PATCH /v2/workspaces/{workspace}/my-plugin-configs/{name}
# DELETE /v2/workspaces/{workspace}/my-plugin-configs/{name}
```

See `plugin-service` skill and `example-plugin/middleware_service.py` for the full pattern.

**Step 3 — Implement `get_middleware_config`** to fetch from the entity store:

```python
from nemo_helix_plugin.entity_client import NemoEntitiesClient, EntityNotFoundError
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.inference_middleware import MiddlewareConfigNotFoundError
from nemo_helix_plugin.sdk_provider import get_async_platform_sdk

def setup(self):
    # Build the entity client once during setup and reuse it per request.
    sdk = get_async_platform_sdk(as_service="my-plugin", internal=True)
    self._entity_client = NemoEntitiesClient(client_from_platform(sdk, AsyncEntitiesClient))

async def get_middleware_config(self, config_type: str, config_id: str):
    if config_type != MyPluginConfig.__entity_type__:
        raise ValueError(f"Unknown config_type={config_type!r}")

    ws, name = config_id.split("/", 1)
    try:
        return await self._entity_client.get(MyPluginConfig, name=name, workspace=ws)
    except EntityNotFoundError as exc:
        raise MiddlewareConfigNotFoundError(config_id) from exc
```

IGW calls `get_middleware_config` at VirtualModel cache-build time and on every
polling cycle.  It is **never called per-request**.

Raise `MiddlewareConfigNotFoundError` on a definitive 404 from your store. IGW
uses this signal to evict any previously-resolved middleware for VirtualModels
that reference this config and to mark them broken (requests return 503) until
the config is recreated or the reference is removed. Any other exception is
treated as transient and the previously-resolved config is preserved.

**Step 4 — Implement `validate_middleware_config`** to return the typed config:

```python
async def validate_middleware_config(self, config_type: str, config) -> MyPluginConfigData:
    if config_type != MyPluginConfig.__entity_type__:
        raise ValueError(f"Unknown config_type={config_type!r}")

    if isinstance(config, MyPluginConfig):
        # Convert entity (has name/workspace fields) → lightweight working type
        return MyPluginConfigData(threshold=config.threshold, ...)

    # Inline dict — validate against the working type directly
    return MyPluginConfigData.model_validate(config)
```

> **Tip:** Return a lightweight Pydantic `BaseModel` (not the `NemoEntity`) from
> `validate_middleware_config`.  The reason is practical: `NemoEntity` inherits
> `workspace: str` (required, no default) from `EntityBase`, so calling
> `MyPluginConfig.model_validate({"threshold": 0.8})` — an inline config dict
> that has no `workspace` — raises a validation error.  A separate
> `MyPluginConfigData(BaseModel)` with only the domain fields works for both
> the inline dict path and the entity store path.

---

## 5. Cache accessor methods

After `_inject_cache()` is called (before `on_startup()`), these methods are available:

```python
# All model entity IDs known to IGW: ["default/llama-3b", "default/llama-70b", ...]
self.list_model_entities_for_workspace()
self.list_model_entities_for_workspace("default")  # filter to workspace

# Model entity details (spec, finetuning_type, providers)
entity = self.get_model_entity("default/llama-3b")
if entity and entity.spec:
    print(entity.spec.is_chat, entity.spec.context_size)

# All active providers for a model — useful for latency-aware routing
providers = self.get_model_providers_for_model("default/llama-3b")

# Resolve a model entity to a backend URL and served model name
target = self.get_inference_url_and_model("default/llama-3b")
# target.model_provider_gateway_url → "http://nim-svc:8080/v1"
# target.served_model_name          → "meta/llama-3.2-3b-instruct"

# VirtualModel access — for meta-routing plugins
vm = self.get_virtual_model("default/my-alias")
self.list_virtual_models_for_workspace("default")
```

---

## 6. Request processing

```python
async def process_request(
    self,
    ctx: InferenceMiddlewareContext,  # per-request context (request_id, original_request, etc.)
    request: InferenceRequest,        # mutable envelope — body, headers, path, typed_body
    middleware_config,                 # validated config from validate_middleware_config
) -> RequestResult:
```

`request.body["model"]` contains the model entity name set by
`VirtualModel.default_model_entity` (if set).  Middleware may rewrite it freely.
After the request middleware chain completes, IGW reads `request.body["model"]`
and resolves it to a provider.

**Return `request`** (or a new `InferenceRequest`) to continue proxying:
```python
request.body["model"] = "default/strong-model"  # route to a different entity
return request
```

**Return `ImmediateResponse`** to skip the proxy entirely:
```python
return ImmediateResponse(data={
    "id": "cached",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": request.body.get("model", ""),
    "choices": [{"index": 0, "message": {"role": "assistant", "content": cached_reply}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
})
```

**Raise `InferenceMiddlewareError`** to return a specific HTTP error:
```python
raise InferenceMiddlewareError("Rate limit exceeded", status_code=429)
```

### `InferenceRequest.typed_body`

IGW populates `request.typed_body` with a TypedDict-validated view of the
request body for known API paths (`v1/chat/completions`, `v1/messages`,
`v1/responses`). Falls back to `None` for unknown paths or malformed bodies.

All three SDK request param types are TypedDicts — plain dicts at runtime.
`typed_body` is a **validated copy** of `body`, not the same dict object:
`TypeAdapter.validate_python` returns a new dict. Mutating `typed_body` does
not propagate to `body` and vice versa. Treat them as independent views of the
same data. Plugins that need path-based dispatch should use `request.path`,
not `isinstance`.

```python
# Get the typed body (falls back to raw body if path was unknown)
body = request.typed_body if request.typed_body is not None else request.body

# Access the original pre-middleware body from a response hook:
original_body = ctx.original_request.typed_body  # or ctx.original_request.body
```

`ctx.original_request` is a snapshot taken before any request middleware ran.
Its `typed_body` (and `.body`) always reflects what the caller originally sent,
regardless of how many middleware have since mutated the live request. This is
the correct source for response-side plugins that need to know the caller's
original format (e.g. cross-format translation).

---

## 7. Response processing

```python
async def process_response(
    self,
    ctx: InferenceMiddlewareContext,  # per-request context (original_request, proxied_request, etc.)
    response: InferenceResponse,      # mutable envelope — result, headers, typed_body, annotations
    middleware_config,
) -> ResponseResult:
```

`response.typed_body` holds the SDK-native parsed response when IGW can parse
the backend payload. For non-streaming responses, if `typed_body` is non-`None`
it is canonical — mutate it instead of `response.result`.

Use the response envelope according to the kind of mutation you need:

| Goal | How |
|---|---|
| Mutate an existing payload field (e.g. redact PII in `choices[0].message.content`) | Mutate `typed_body` when available, or `result` when no typed view exists |
| Add a new field to the response body (e.g. `guardrails`) | Write to `response_body_annotations` |
| Modify HTTP response headers | Mutate `headers` |

For non-streaming payload changes, mutate the canonical body and return the
envelope:
```python
if response.typed_body is not None:
    response.typed_body.choices[0].message.content = redact(...)
elif isinstance(response.result, dict):
    response.result["choices"][0]["message"]["content"] = redact(...)
return response
```

For non-schema top-level fields, use `response_body_annotations`:
```python
response.response_body_annotations["guardrails_data"] = {
    "config_ids": ["default/safety-config"]
}
return response
```

Request middleware can also annotate the eventual backend response, even though
the `InferenceResponse` object does not exist yet. Put those annotations in
`ctx.response_body_annotations` as a staging area:
```python
async def process_request(self, ctx, request, middleware_config):
    ctx.response_body_annotations["guardrails_data"] = {
        "config_ids": ["default/input-only-safety"]
    }
    return request
```

When IGW later receives the backend response, it builds an `InferenceResponse`
and copies the staged values into `response.response_body_annotations`. From
that point on, `response.response_body_annotations` is canonical because it is
attached to the response being returned. Response middleware should preserve,
replace, or remove annotations there, not on `ctx`. Final serialization injects
only `response.response_body_annotations` into the response body.

For streaming, wrap the iterator:
```python
async def _filtered(stream):
    async for chunk in stream:
        yield redact_chunk(chunk)
return _filtered(response.result)
```

`response_body_annotations` is a non-streaming feature in the current
implementation. IGW accumulates annotations on streaming responses, but does not
serialize them into SSE chunks yet.

---

## 8. Error handling

| Exception | HTTP status | When to use |
|---|---|---|
| `InferenceMiddlewareError(msg, status_code=N)` | N (default 500) | Any handled error with a specific status |
| `InferenceMiddlewareUnavailableError(msg)` | 503 | Upstream service unavailable |
| `MiddlewareConfigNotFoundError(config_id)` | 404 | Raised from `get_middleware_config` when the referenced entity is gone. Causes IGW to evict cached middleware and mark referencing VirtualModels broken. |
| Unhandled exception | 500 | Let IGW catch and log |

---

## 9. Testing

Mock the cache accessor for unit tests:

```python
from unittest.mock import MagicMock
from nemo_helix_plugin.inference_middleware import InferenceMiddlewareCacheAccessor

cache = MagicMock(spec=InferenceMiddlewareCacheAccessor)
cache.get_model_entity.return_value = None
plugin = MyMiddleware()
plugin._inject_cache(cache)
await plugin.on_startup()
```

See `plugin-testing` skill and `example-plugin/tests/test_inference_middleware.py` for complete examples.

---

## 10. Reference implementation

`plugins/example-plugin/` contains a working keyword content-filter plugin that demonstrates every concept in this document:

| File | What it shows |
|---|---|
| `middleware_config.py` | `NemoEntity` subclass for stored config |
| `middleware.py` | Full `NemoInferenceMiddleware` with both inline and `config_id` support, `ImmediateResponse`, response redaction |
| `middleware_service.py` | CRUD API for the config entity (required for `config_id` support) |
| `tests/test_inference_middleware.py` | Unit tests with mocked cache and entity client |
| `pyproject.toml` | `nemo.inference_middleware` entry-point registration |

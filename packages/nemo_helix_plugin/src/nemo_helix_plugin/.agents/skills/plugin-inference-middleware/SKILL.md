---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-inference-middleware
description: Implements NemoInferenceMiddleware plugins for in-process inference request/response interception in IGW. Use when building a middleware plugin, implementing process_request or process_response, handling MiddlewareCall config (inline or config_id), exposing config entity CRUD APIs, or wiring up the nemo.inference_middleware entry-point. Trigger keywords: inference middleware, NemoInferenceMiddleware, process_request, process_response, MiddlewareCall, config_id, ImmediateResponse, VirtualModel middleware, middleware plugin.
---

# Inference Middleware Plugin

## Entry-point registration

```toml
# pyproject.toml
[project.entry-points."nemo.inference_middleware"]
"nemo-my-plugin" = "nemo_my_plugin.middleware:MyMiddleware"
```

The key (`"nemo-my-plugin"`) is the **plugin identity** — it must match
`MiddlewareCall.name` in VirtualModel configs.

---

## Minimal implementation

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
    InferenceMiddlewareUnavailableError,
    VirtualModel,
)
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient
from nhx.common.sdk_factory import get_async_platform_sdk
from pydantic import BaseModel
from typing import Any

class MyConfigData(BaseModel):
    """Lightweight working config — returned from validate_middleware_config."""
    threshold: float = 0.8

class MyMiddleware(NemoInferenceMiddleware):

    def __init__(self) -> None:
        super().__init__()
        self._entity_client = None  # initialised in on_startup

    # ── Lifecycle ──────────────────────────────────────────────────────
    async def on_startup(self) -> None:
        # Build an entity client from the platform SDK so get_middleware_config
        # can fetch stored config entities.
        sdk = get_async_platform_sdk(as_service="nemo-my-plugin", internal=True)
        typed_client = client_from_platform(sdk, AsyncEntitiesClient)
        self._entity_client = NemoEntitiesClient(typed_client)

        entities = self.list_model_entities_for_workspace()  # cache available here
        ...

    async def on_shutdown(self) -> None: ...

    async def on_virtual_model_upserted(self, virtual_model: VirtualModel) -> None: ...
    async def on_virtual_model_destroyed(self, virtual_model: VirtualModel) -> None: ...

    # ── Config resolution (needed only for config_id support) ──────────
    async def get_middleware_config(self, config_type: str, config_id: str) -> Any:
        """Fetch stored config entity from the entity store.

        Called at VirtualModel cache-build time and every polling cycle.
        Never called per-request.
        """
        if config_type != "my_plugin_config":
            raise InferenceMiddlewareError(f"Unknown config_type={config_type!r}", status_code=400)
        ws, name = config_id.split("/", 1)
        return await self._entity_client.get(MyPluginConfig, name=name, workspace=ws)

    async def validate_middleware_config(self, config_type: str, config: Any) -> MyConfigData:
        """Validate and normalise config — called at cache-build time, not per-request."""
        if config_type != "my_plugin_config":
            raise ValueError(f"Unknown config_type={config_type!r}")
        if isinstance(config, MyPluginConfig):
            return MyConfigData(threshold=config.threshold)
        return MyConfigData.model_validate(config)  # inline dict path

    # ── Request hook ───────────────────────────────────────────────────
    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: MyConfigData,
    ) -> RequestResult:
        # Mutate body["model"] to route to a different entity:
        request.body["model"] = "default/strong-model"
        return request

        # Or short-circuit (skip backend entirely):
        # return ImmediateResponse(data={"choices": [{"message": {"content": "cached"}}], ...})

        # Or raise a typed error:
        # raise InferenceMiddlewareError("Rate limit exceeded", status_code=429)

    # ── Response hook ──────────────────────────────────────────────────
    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: MyConfigData,
    ) -> ResponseResult:
        if isinstance(response.result, dict):
            # Non-streaming — mutate and return
            response.result["choices"][0]["message"]["content"] = redact(...)
            return response.result
        else:
            # Streaming — wrap the async iterator
            async def _filtered(stream):
                async for chunk in stream:
                    yield transform(chunk)
            return _filtered(response.result)
```

---

## MiddlewareCall: inline config vs. config_id

```json
// Inline config — embed directly in VirtualModel
{
  "name": "nemo-my-plugin",
  "config_type": "my_plugin_config",
  "config": {"threshold": 0.9}
}

// Config by reference — stored entity, supports versioning and sharing
{
  "name": "nemo-my-plugin",
  "config_type": "my_plugin_config",
  "config_id": "default/prod-config"
}
```

`config_type` is **always required** regardless of which style is used.
It must match the `entity_type` of your config `NemoEntity` subclass.

**Use inline** when the config is simple, per-VirtualModel, and doesn't need
versioning or sharing.

**Use `config_id`** when:

- Multiple VirtualModels share the same config
- Operators update config independently of VirtualModels
- Auto-propagation is needed — IGW re-resolves on every polling cycle when the
  entity's `updated_at` changes; no VirtualModel edit required

---

## Config entity for config_id support

When using `config_id`, the referenced entity must exist in the entity store.
The plugin must expose CRUD endpoints for it:

```python
# entities.py
from nemo_helix_plugin.entity import NemoEntity

class MyPluginConfig(NemoEntity, entity_type="my_plugin_config"):
    """Stored config — entity_type must match MiddlewareCall.config_type."""
    threshold: float = 0.8
    strong_model: str = "default/llama-70b"
    weak_model: str = "default/llama-8b"
```

The CRUD API follows the standard NeMo Helix workspace-scoped pattern
(`POST /v2/workspaces/{workspace}/my-plugin-configs`, etc.).
See `plugin-service` skill for the full CRUD pattern.

> **Separation of types:** `NemoEntity` inherits `workspace: str` (required,
> no default) from `EntityBase`.  Inline config dicts (from `MiddlewareCall.config`)
> contain only domain fields and have no `workspace`, so
> `MyPluginConfig.model_validate({"threshold": 0.8})` raises a validation error.
> A separate `MyConfigData(BaseModel)` with only the domain fields works for both
> the inline path and the entity store path — have `validate_middleware_config`
> always return that type.

---

## Cache accessor reference

Available from `on_startup()` onward:

```python
# Model entities
self.list_model_entities_for_workspace()               # all workspaces
self.list_model_entities_for_workspace("default")     # filtered
entity = self.get_model_entity("default/llama-3b")    # ModelEntity | None
providers = self.get_model_providers_for_model("default/llama-3b")  # list[ModelProvider]

# Resolve backend URL + served model name for a direct call
target = self.get_inference_url_and_model("default/llama-3b")
# target.model_provider_gateway_url  → "http://nim-svc:8080/v1"
# target.served_model_name           → "meta/llama-3.2-3b-instruct"

# VirtualModels
vm = self.get_virtual_model("default/my-alias")       # VirtualModel | None
self.list_virtual_models_for_workspace("default")     # list[str]
```

---

## Request and response context

### `request.typed_body`

IGW populates `request.typed_body` with a TypedDict-validated view of the body
for known paths (`v1/chat/completions`, `v1/messages`, `v1/responses`). All three
SDK param types are TypedDicts — plain dicts at runtime. `typed_body` is
interchangeable with `body`; use `request.path` for format dispatch, not
`isinstance`.

```python
# Use typed_body when available, fall back to raw body
body = request.typed_body if request.typed_body is not None else request.body

# In a response hook, read the original pre-middleware request:
original_body = ctx.original_request.typed_body  # or ctx.original_request.body
```

`ctx.original_request` is captured before any request middleware runs. Its
`typed_body` and `.body` always reflect what the caller originally sent, even
after downstream middleware has mutated the live request.

### `response.typed_body` and annotations

`response.typed_body` holds the SDK-native parsed response object
(`ChatCompletion`, `Message`) when IGW can parse the backend payload. For
non-streaming responses, if non-`None`, it is canonical — mutate it instead of
`response.result`.

Use this response contract:

| Goal | How |
| --- | --- |
| Mutate an existing payload field (e.g. redact PII in `choices[0].message.content`) | Mutate `typed_body` when available, or `result` when no typed view exists |
| Add a new field to the response body (e.g. `guardrails`) | Write to `response_body_annotations` |
| Modify HTTP response headers | Mutate `headers` |

```python
# Add non-schema top-level metadata without fighting typed response serialization.
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

`response_body_annotations` is currently only supported for non-streaming responses.
IGW accumulates annotations on streaming responses, but does not serialize
them into the returned SSE chunks yet.

---

## RequestResult return values

| Return | Effect |
| --- | --- |
| `InferenceRequest` | IGW resolves `request.body["model"]` to a provider and proxies |
| `ImmediateResponse(data=...)` | Skip proxy; `data` is passed to response middleware |
| Raise `InferenceMiddlewareError(msg, status_code=N)` | HTTP N error to caller |
| Raise `InferenceMiddlewareUnavailableError(msg)` | HTTP 503 to caller |

---

## Testing pattern

```python
from unittest.mock import MagicMock
from nemo_helix_plugin.inference_middleware import InferenceMiddlewareCacheAccessor

def _make_plugin():
    plugin = MyMiddleware()
    cache = MagicMock(spec=InferenceMiddlewareCacheAccessor)
    cache.list_model_entities_for_workspace.return_value = ["default/llama-3b"]
    plugin._inject_cache(cache)
    return plugin

@pytest.mark.asyncio
async def test_blocks_request():
    plugin = _make_plugin()
    cfg = MyConfigData(threshold=0.5)
    result = await plugin.process_request({"model": "default/vm", "messages": []}, {}, cfg)
    assert isinstance(result, ImmediateResponse)
```

To mock `get_middleware_config` (entity store fetch), patch at the entity client level:

```python
with patch("nemo_helix_plugin.entity_client.NemoEntitiesClient", return_value=mock_client):
    result = await plugin.get_middleware_config("my_plugin_config", "ws/cfg")
```

---

## Reference implementation

`plugins/example-plugin/` contains a complete working example:

| File | Contents |
| --- | --- |
| `middleware_config.py` | `ExampleMiddlewareConfig(NemoEntity)` — stored config entity |
| `middleware.py` | `ExampleInferenceMiddleware` — full implementation with both config styles |
| `middleware_service.py` | CRUD router for the config entity |
| `tests/test_inference_middleware.py` | Unit tests with mocked cache and entity client |

---

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Calling `get_model_entity()` before `on_startup()` | Cache is only available after `_inject_cache()` |
| Returning `None` from `process_request` | Always return the `InferenceRequest` or an `ImmediateResponse` |
| Using old signature `process_request(self, request_body, request_headers, ...)` | Correct signature: `process_request(self, ctx, request, middleware_config)` |
| Using old signature `process_response(self, response_result, request_body, ...)` | Correct signature: `process_response(self, ctx, response, middleware_config)` |
| Accessing `request.body` via `request_body` argument | Use `request.body` — the parameter is now `request: InferenceRequest` |
| Using `NemoEntity` directly as the `validate_middleware_config` return type | Return a lightweight `BaseModel` — entity metadata is irrelevant at request time |
| Calling `get_middleware_config` per-request | This is called by IGW at cache-build time only; don't call it yourself in `process_request` |
| Using `MiddlewareCall.config_id` without a CRUD API | The referenced entity must be creatable; expose CRUD endpoints |
| Injecting new top-level response fields into `result` or `typed_body` | Use `response_body_annotations` so translation and typed serialization cannot drop them |

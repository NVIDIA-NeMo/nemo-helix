# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model entity router endpoints."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_platform.types.inference import ModelProvider, ServedModelMapping
from nemo_platform.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_platform_plugin.inference_middleware import ImmediateResponse, InferenceRequest, NemoInferenceMiddleware
from nmp.core.inference_gateway.api.dependencies import (
    global_middleware_registry,
    global_model_cache,
    global_virtual_model_cache,
)
from nmp.core.inference_gateway.api.middleware_registry import MiddlewareRegistry, ResolvedMiddlewareCall
from nmp.core.inference_gateway.api.model_cache import ModelCache, ModelEntityInfo, ModelProviderInfo
from nmp.core.inference_gateway.api.virtual_model_cache import VirtualModelCache


def _autoprovisioned_vms_for_cache(model_cache: ModelCache) -> list[SDKVirtualModel]:
    """Mirror conftest.autoprovisioned_vms_for_cache for tests that build a custom ModelCache.

    Local copy because the unit tests directory has no ``__init__.py`` so the
    conftest helper isn't importable.
    """
    return [
        SDKVirtualModel(
            id=f"{workspace}/{name}",
            entity_id=f"{workspace}/{name}",
            workspace=workspace,
            name=name,
            parent=workspace,
            db_version=1,
            default_model_entity=f"{workspace}/{name}",
            autoprovisioned=True,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        for (workspace, name) in model_cache.model_entity_info_map.keys()
        if "&adapters/" not in name
    ]


def test_model_entity_proxy_endpoint(client: TestClient, mock_proxy_client, mock_proxy_response):
    """Test the model entity proxy endpoint."""
    upstream = {"model": "response"}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    # workspace_id (e2e-test) is used as the workspace, model_name is meta_llama-3.2-1b-instruct
    response = client.get("/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/chat/completions")
    mock_proxy_client.request.assert_called_once()
    assert response.status_code == 200
    assert response.json() == upstream


def test_model_entity_proxy_not_found(client: TestClient):
    """Inference requests now require a VirtualModel — unknown names return 404."""
    response = client.get("/v2/workspaces/nonexistent/model/model/-/v1/chat/completions")
    assert response.status_code == 404
    assert "No VirtualModel" in response.json()["detail"]


def test_model_entity_proxy_with_auth(
    client: TestClient, mock_proxy_client, mock_proxy_response, model_cache: ModelCache
):
    """Test that model entity proxy uses the first available provider (which may or may not have auth)."""
    upstream = {"auth": "success"}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    # Get the model entity and verify it exists
    entity_info = model_cache.get_from_model_entity("e2e-test", "meta_llama-3.2-1b-instruct")
    assert entity_info is not None
    # The first provider in the list should be used
    _, first_provider = entity_info.model_providers[0]
    # Note: first provider may or may not have auth, but the proxy should still work

    response = client.post("/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/chat/completions")
    assert response.status_code == 200
    assert response.json() == upstream
    mock_proxy_client.request.assert_called_once()


def test_model_entity_proxy_with_unresolved_provider_secret(client: TestClient, model_cache: ModelCache):
    """Test that model entity proxy returns 424 when a configured provider secret is unresolved."""
    # Get the entity and remove secret from all providers
    entity_info = model_cache.get_from_model_entity("e2e-test", "meta_llama-3.2-1b-instruct")
    assert entity_info is not None

    # Set api_key_secret_name but remove secret_value for the first provider
    _, first_provider = entity_info.model_providers[0]
    first_provider.model_provider.api_key_secret_name = "test-key-id"
    first_provider.secret_value = None

    response = client.post("/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/chat/completions")
    assert response.status_code == 424
    assert "Could not fetch secret" in response.json()["detail"]
    assert "secret not found or unreachable" in response.json()["detail"]


def test_model_entity_proxy_post_with_body(client: TestClient, mock_proxy_client, mock_proxy_response):
    """Test that model entity proxy forwards POST requests with body correctly."""
    upstream = {"response": "data"}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    request_body = {"messages": [{"role": "user", "content": "Hello"}], "model": "test"}

    response = client.post(
        "/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/chat/completions",
        json=request_body,
    )
    assert response.status_code == 200
    assert response.json() == upstream
    mock_proxy_client.request.assert_called_once()


def test_model_entity_proxy_replaces_model_with_served_name(client: TestClient, mock_proxy_client, mock_proxy_response):
    """Test that the proxy replaces the user-provided model name with the served_model_name."""
    upstream = {"response": "data"}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    # User sends request with model entity name
    request_body = {"messages": [{"role": "user", "content": "Hello"}], "model": "e2e-test/meta_llama-3.2-1b-instruct"}

    response = client.post(
        "/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/chat/completions",
        json=request_body,
    )
    assert response.status_code == 200
    assert response.json() == upstream

    # Verify the proxy was called with the served_model_name
    mock_proxy_client.request.assert_called_once()
    call_args = mock_proxy_client.request.call_args

    # The body should have been transformed to use the served_model_name
    sent_body = call_args.kwargs["data"]

    sent_json = json.loads(sent_body)
    assert sent_json["model"] == "meta/llama-3.2-1b-instruct"
    assert sent_json["messages"] == [{"role": "user", "content": "Hello"}]


def test_model_entity_proxy_no_default_vm_preserves_qualified_body_model(
    app, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """Regression: a custom VM with NO default_model_entity keeps the client's qualified body model.

    The model route must not overwrite ``body["model"]`` with a bare URL name for a
    non-adapter request: a custom VM without ``default_model_entity`` relies on the
    client's qualified body model (``ws/entity``) for entity resolution. Seeding the bare
    URL name (``alias-router``) would make ``parse_model_entity_ref`` raise (unqualified,
    no default workspace) → 422. The URL-name seed is scoped to composite (``&adapters/``)
    names only.
    """
    upstream = {"response": "data"}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    # Custom VM named "alias-router" with no default_model_entity; the client supplies the
    # real (qualified) model entity in the body.
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "alias-router")])
    # _make_sdk_vm falls back default_model_entity to "ws/name"; force it truly unset.
    vm_cache.virtual_model_map[("e2e-test", "alias-router")].default_model_entity = None
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/e2e-test/model/alias-router/-/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}], "model": "e2e-test/meta_llama-3.2-1b-instruct"},
    )
    assert response.status_code == 200
    assert response.json() == upstream
    # The client's body model resolved the entity; upstream got the served name.
    sent_json = json.loads(mock_proxy_client.request.call_args.kwargs["data"])
    assert sent_json["model"] == "meta/llama-3.2-1b-instruct"


def test_model_entity_proxy_without_model_field(client: TestClient, mock_proxy_client, mock_proxy_response):
    """When the body has no ``model`` field, the VirtualModel pipeline seeds it from
    ``default_model_entity`` (set by the autoprovisioned VM) and the served-model rewrite
    runs on that, so the upstream sees the served name.
    """
    upstream = {"response": "data"}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    # Request without model field
    request_body = {"messages": [{"role": "user", "content": "Hello"}]}

    response = client.post(
        "/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/chat/completions",
        json=request_body,
    )
    assert response.status_code == 200
    assert response.json() == upstream

    # Verify the model field was injected
    mock_proxy_client.request.assert_called_once()
    call_args = mock_proxy_client.request.call_args
    sent_body = call_args.kwargs["data"]

    sent_json = json.loads(sent_body)
    # Model field should now be present with the served_model_name
    assert sent_json["model"] == "meta/llama-3.2-1b-instruct"
    assert sent_json["messages"] == [{"role": "user", "content": "Hello"}]


def test_model_entity_proxy_invalid_json_body(client: TestClient, mock_proxy_client, mock_proxy_response):
    """An invalid-JSON request body is replaced by a synthesized JSON payload carrying
    just ``{"model": "<served_model_name>"}`` after the VM pipeline seeds the model
    field from ``default_model_entity`` and rewrites it to the served name.

    Behavior change vs. the historical pre-VM fallback: the old route forwarded raw
    bytes verbatim to the upstream. Now every request is parsed by the middleware
    pipeline; bodies that don't parse as a JSON object are replaced with the seeded
    payload so the upstream still receives a valid request. This is consistent with
    every other request path; raw byte passthrough is no longer supported on the
    inference routes.
    """
    upstream = {"error": "invalid request"}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    response = client.post(
        "/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/chat/completions",
        content=b"not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()
    call_args = mock_proxy_client.request.call_args
    sent_body = json.loads(call_args.kwargs["data"])
    assert sent_body["model"] == "meta/llama-3.2-1b-instruct"


def test_model_entity_proxy_no_providers(app, client: TestClient):
    """If a model entity exists in the cache but has no providers, the autoprovisioned
    VM still triggers but the entity-resolution step inside ``virtual_model_proxy``
    returns 404 with the explicit "no providers" message.
    """

    cache = ModelCache()
    # Manually create a model entity with no providers (edge case)
    cache.model_entity_info_map[("ns1", "orphan-model")] = ModelEntityInfo(
        workspace="ns1",
        name="orphan-model",
        model_providers=[],  # Empty - no providers
    )
    app.dependency_overrides[global_model_cache] = lambda: cache
    # An autoprovisioned VM still exists for any served entity in production, so
    # mirror that here so the request reaches virtual_model_proxy and exercises the
    # "no providers" branch rather than 404ing at the VM lookup.
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(_autoprovisioned_vms_for_cache(cache))
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ns1/model/orphan-model/-/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 404
    assert "No providers found" in response.json()["detail"]


def test_model_entity_proxy_empty_served_model_name(app, client: TestClient, mock_proxy_client, mock_proxy_response):
    """An empty ``served_model_name`` is still applied to the body verbatim — the
    rewrite logic does not skip empty strings (regression guard for upstream
    misconfiguration). The user-facing response model-rewrite, by contrast,
    is a strict equality match and so leaves the response body alone when
    ``served_model_name == ""`` (no chunk's ``model`` field will equal "").
    """

    upstream = {"response": "data"}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    cache = ModelCache()
    # Create a provider with empty served_model_name
    provider = ModelProviderInfo(
        model_provider=ModelProvider(
            workspace="ns1",
            name="provider1",
            host_url="http://provider1.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            served_models=[
                ServedModelMapping(
                    model_entity_id="ns1/my-model",
                    served_model_name="",  # Empty served_model_name
                )
            ],
        ),
    )
    cache.update_model_info(provider)
    cache.rebuild_model_entity_map()
    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(_autoprovisioned_vms_for_cache(cache))
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    # Request without model field
    request_body = {"messages": [{"role": "user", "content": "Hello"}]}

    response = client.post(
        "/v2/workspaces/ns1/model/my-model/-/v1/chat/completions",
        json=request_body,
    )
    assert response.status_code == 200

    # The VM pipeline always sets body["model"] (seeded from default_model_entity then
    # rewritten to served_model_name); when served_model_name is the empty string the
    # field ends up empty rather than absent.
    mock_proxy_client.request.assert_called_once()
    call_args = mock_proxy_client.request.call_args
    sent_body = call_args.kwargs["data"]

    sent_json = json.loads(sent_body)
    assert sent_json["model"] == ""
    assert sent_json["messages"] == [{"role": "user", "content": "Hello"}]


def test_model_entity_proxy_invalid_workspace_returns_422(client: TestClient):
    """Test that invalid workspace (e.g. single char) returns 422 Unprocessable Entity."""
    response = client.get("/v2/workspaces/a/model/valid-model-name/-/v1/chat/completions")
    assert response.status_code == 422
    assert "workspace" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_model_entity_proxy_invalid_name_returns_422(client: TestClient):
    """Test that invalid model name (e.g. single char) returns 422 Unprocessable Entity."""
    response = client.get("/v2/workspaces/default/model/a/-/v1/chat/completions")
    assert response.status_code == 422
    assert "name" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Cross-workspace LoRA routing tests
# ---------------------------------------------------------------------------


def test_model_router_routes_cross_workspace_lora(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """Cross-workspace LoRA via the model entity router: ``base_ws="ws-a"``,
    ``adapter_ws="ws-b"``, ``provider.workspace="ws-a"``.

    No per-adapter VM is created; the request routes through the **base model's** VM
    (keyed ``("ws-a", "base")``) via ``resolve_vm_for_model``, and the
    ``default_model_entity`` splice preserves the ``&adapters/ws-b/adapter`` suffix so
    ``parse_model_entity_ref`` (split on first '/') resolves the composite entity and the
    served-model rewrite still resolves to the flat-dir encoded backend id.
    """
    upstream = {"choices": []}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    cache = ModelCache()
    cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="ws-a",
                name="nim-provider",
                host_url="http://nim.workspace-a.example.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id="ws-a/base&adapters/ws-b/adapter",
                        served_model_name="ws-b--adapter",
                    ),
                ],
            )
        )
    )
    cache.rebuild_model_entity_map()
    app.dependency_overrides[global_model_cache] = lambda: cache

    # Base model's VM (keyed ("ws-a", "base")) — the adapter request routes through it;
    # the controller creates no per-adapter VM (provider_reconciler.py:440).
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(
        [
            _make_sdk_vm(
                workspace="ws-a",
                name="base",
                default_model_entity="ws-a/base",
            )
        ]
    )
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ws-a/model/base&adapters/ws-b/adapter/-/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert response.json() == upstream

    mock_proxy_client.request.assert_called_once()
    call_args = mock_proxy_client.request.call_args

    # Body's ``model`` field is set to the served name (the flat-dir encoding the
    # backend actually uses), not the entity id.
    sent_body = json.loads(call_args.kwargs["data"])
    assert sent_body["model"] == "ws-b--adapter"

    # Upstream URL hits the provider in workspace ``ws-a``, never anything keyed off ``ws-b``.
    assert urlparse(call_args.kwargs["url"]).hostname == "nim.workspace-a.example.com"
    assert "ws-b" not in call_args.kwargs["url"]


def test_model_router_routes_url_encoded_cross_workspace_lora(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """Cross-workspace LoRA with the ``&adapters/`` separators URL-encoded as ``%2F``.

    Guard against a client that url-encodes its own output of ``GET /v1/models`` —
    the path ``base&adapters%2Fws-b%2Fadapter`` must resolve through the base model's VM
    (keyed ``("ws-a", "base")``) and, after the splice, reach the composite cache key
    ``("ws-a", "base&adapters/ws-b/adapter")`` — the same as the unencoded form. No
    per-adapter VM exists.
    """
    upstream = {"choices": []}
    mock_proxy_response._body = [json.dumps(upstream).encode()]

    cache = ModelCache()
    cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="ws-a",
                name="nim-provider",
                host_url="http://nim.workspace-a.example.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id="ws-a/base&adapters/ws-b/adapter",
                        served_model_name="ws-b--adapter",
                    ),
                ],
            )
        )
    )
    cache.rebuild_model_entity_map()
    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(
        [
            _make_sdk_vm(
                workspace="ws-a",
                name="base",
                default_model_entity="ws-a/base",
            )
        ]
    )
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ws-a/model/base&adapters%2Fws-b%2Fadapter/-/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert response.json() == upstream

    mock_proxy_client.request.assert_called_once()
    call_args = mock_proxy_client.request.call_args

    sent_body = json.loads(call_args.kwargs["data"])
    assert sent_body["model"] == "ws-b--adapter"
    assert urlparse(call_args.kwargs["url"]).hostname == "nim.workspace-a.example.com"
    assert "ws-b" not in call_args.kwargs["url"]


def test_model_router_adapter_routes_through_base_vm(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """The model-entity route sends an adapter through the **base** model's VM.

    The URL name is the composite ``base&adapters/ws/adder``; there is no per-adapter VM.
    ``resolve_vm_for_model`` keys the VM by the base segment (``base``), the URL composite
    is seeded into ``body["model"]``, and the splice preserves the suffix so the served
    name resolves. Complements the openai-route adapter tests — same behavior, model route.
    """
    upstream = {"choices": []}
    mock_proxy_response._body = [json.dumps(upstream).encode()]
    cache = ModelCache()
    cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="ws",
                name="nim-provider",
                host_url="http://nim.example.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id="ws/base&adapters/ws/adder",
                        served_model_name="adder-backend-id",
                    ),
                ],
            )
        )
    )
    cache.rebuild_model_entity_map()
    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("ws", "base", default_model_entity="ws/base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ws/model/base&adapters/ws/adder/-/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert response.json() == upstream
    # Served-model-name rewrite writes the adapter's served name upstream.
    sent_body = json.loads(mock_proxy_client.request.call_args.kwargs["data"])
    assert sent_body["model"] == "adder-backend-id"


# ---------------------------------------------------------------------------
# VirtualModel routing tests
# ---------------------------------------------------------------------------


def _make_sdk_vm(
    workspace: str,
    name: str,
    default_model_entity: str | None = None,
) -> SDKVirtualModel:
    return SDKVirtualModel(
        id=f"{workspace}/{name}",
        entity_id=f"{workspace}/{name}",
        name=name,
        workspace=workspace,
        parent=workspace,
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        default_model_entity=default_model_entity or f"{workspace}/{name}",
    )


def test_model_entity_proxy_routes_via_virtual_model(app: FastAPI, client: TestClient):
    """VirtualModel in cache → proxy resolves via default_model_entity."""
    vm_cache = VirtualModelCache()
    # Alias "my-alias" → existing entity "meta_llama-3.2-1b-instruct"
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-alias", "e2e-test/meta_llama-3.2-1b-instruct")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/e2e-test/model/my-alias/-/v1/completions")
    assert response.status_code == 200


def test_model_entity_proxy_404_when_no_virtual_model(app: FastAPI, client: TestClient):
    """Without a VirtualModel — even if the underlying ModelCache entity exists — the
    proxy returns 404. Inference now requires a VM (auto-created by the controller for
    served entities; operators can create them manually for LoRA / overrides).
    """
    # Override with an empty VM cache so the auto-populated default doesn't fire.
    app.dependency_overrides[global_virtual_model_cache] = lambda: VirtualModelCache()
    response = client.get("/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/completions")
    assert response.status_code == 404
    assert "No VirtualModel" in response.json()["detail"]


def test_model_entity_proxy_virtual_model_no_default_model_entity_no_middleware_returns_422(
    app: FastAPI, client: TestClient
):
    """VirtualModel with no default_model_entity and no middleware → 422.

    Both the model endpoint and the openai endpoint now return 422 for this
    misconfiguration, regardless of HTTP method.  Previously the model endpoint
    returned 404 because it fell back to constructing the model entity ID from
    the URL path; that fallback is removed so the error is consistent.
    """
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(
        [
            SDKVirtualModel(
                id="ws/mw-only",
                entity_id="ws/mw-only",
                name="mw-only",
                workspace="ws",
                parent="ws",
                db_version=1,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                default_model_entity=None,
            )
        ]
    )
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    # GET (no body)
    response = client.get("/v2/workspaces/ws/model/mw-only/-/v1/completions")
    assert response.status_code == 422
    assert response.json()["detail"]  # 422 — body["model"] cannot be resolved as an entity ref

    # POST (body present but no middleware to route it)
    response = client.post(
        "/v2/workspaces/ws/model/mw-only/-/v1/chat/completions",
        json={"model": "mw-only", "messages": []},
    )
    assert response.status_code == 422
    assert response.json()["detail"]  # 422 — body["model"] cannot be resolved as an entity ref


# ---------------------------------------------------------------------------
# Middleware pipeline execution tests
# ---------------------------------------------------------------------------


def _make_registry_with_request_middleware(workspace: str, vm_name: str, plugin) -> MiddlewareRegistry:
    call = ResolvedMiddlewareCall(plugin_name="test-plugin", config_type="t", resolved_config={})
    registry = MiddlewareRegistry(plugins={"test-plugin": plugin})
    registry.request_middleware_calls[(workspace, vm_name)] = [call]
    registry.response_middleware_calls[(workspace, vm_name)] = []
    registry.post_response_middleware_calls[(workspace, vm_name)] = []
    return registry


def test_model_entity_proxy_executes_request_middleware(app: FastAPI, client: TestClient):
    """Request middleware that rewrites body['model'] is obeyed."""

    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.process_request = AsyncMock(
        side_effect=lambda ctx, req, cfg: InferenceRequest(
            body={**req.body, "model": "e2e-test/meta_llama-3.2-1b-instruct"},
            headers=req.headers,
            path=req.path,
        )
    )

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-router", default_model_entity=None)])
    registry = _make_registry_with_request_middleware("e2e-test", "my-router", plugin)

    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.get("/v2/workspaces/e2e-test/model/my-router/-/v1/completions")
    assert response.status_code == 200
    plugin.process_request.assert_awaited_once()


def test_model_entity_proxy_immediate_response_skips_proxy(app: FastAPI, client: TestClient, mock_proxy_client):
    """ImmediateResponse from request middleware bypasses backend proxying entirely."""

    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.process_request = AsyncMock(return_value=ImmediateResponse(data={"id": "direct", "choices": []}))

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-router", default_model_entity=None)])
    registry = _make_registry_with_request_middleware("e2e-test", "my-router", plugin)

    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.get("/v2/workspaces/e2e-test/model/my-router/-/v1/completions")
    assert response.status_code == 200
    mock_proxy_client.request.assert_not_called()


def test_model_entity_proxy_middleware_body_mutations_reach_backend(
    app: FastAPI, client: TestClient, mock_proxy_client
):
    """Request middleware mutations to non-model fields must survive to the backend.

    Regression test for the double-body-read bug: the VirtualModel path previously
    called ``_update_request_with_served_name(request, served_model_name)`` which
    re-parsed the body from raw request bytes, discarding any fields added by
    request middleware (e.g. ``x_custom_field``, ``temperature``, etc.).

    The fix: build the outgoing body from the already-mutated ``json_body`` dict
    rather than re-reading from the ``Request`` object.
    """
    import json as _json

    # Middleware rewrites body["model"] for routing AND stamps an extra field
    # that must survive to the backend.
    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.process_request = AsyncMock(
        side_effect=lambda ctx, req, cfg: InferenceRequest(
            body={
                **req.body,
                "model": "e2e-test/meta_llama-3.2-1b-instruct",
                "x_custom_middleware_field": "was-mutated",
            },
            headers=req.headers,
            path=req.path,
        )
    )

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-vm", default_model_entity=None)])
    registry = _make_registry_with_request_middleware("e2e-test", "my-vm", plugin)

    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.post(
        "/v2/workspaces/e2e-test/model/my-vm/-/v1/chat/completions",
        json={"model": "my-vm", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    # Inspect the body that was actually forwarded to the backend provider.
    mock_proxy_client.request.assert_called_once()
    sent_data = mock_proxy_client.request.call_args.kwargs["data"]
    sent_body = _json.loads(sent_data)

    assert sent_body.get("x_custom_middleware_field") == "was-mutated", (
        f"Middleware mutation was lost before reaching the backend. Backend received: {sent_body}"
    )

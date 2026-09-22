# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI router endpoints."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_platform.types.inference import ModelProvider, ServedModelMapping
from nemo_platform.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_platform_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
)
from nmp.core.inference_gateway.api.dependencies import (
    global_middleware_registry,
    global_model_cache,
    global_virtual_model_cache,
)
from nmp.core.inference_gateway.api.middleware_registry import MiddlewareRegistry, ResolvedMiddlewareCall
from nmp.core.inference_gateway.api.model_cache import ModelCache, ModelEntityInfo, ModelProviderInfo
from nmp.core.inference_gateway.api.v2.openai import ParseOpenAIModelError, parse_igw_openai_model, resolve_vm_for_model
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


def _custom_vm(workspace: str, name: str, default_model_entity: str | None = None) -> SDKVirtualModel:
    """Build a non-autoprovisioned (operator/Switchyard-style) VirtualModel."""
    return SDKVirtualModel(
        id=f"{workspace}/{name}",
        entity_id=f"{workspace}/{name}",
        workspace=workspace,
        name=name,
        parent=workspace,
        db_version=1,
        default_model_entity=default_model_entity,
        autoprovisioned=False,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


# OpenAI List Models Tests


def test_list_models_empty_cache(app: FastAPI, client: TestClient):
    """Test listing models when the VirtualModel cache is empty."""

    empty_cache = VirtualModelCache()
    app.dependency_overrides[global_virtual_model_cache] = lambda: empty_cache

    response = client.get("/v2/workspaces/default/openai/-/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert data["data"] == []


def test_list_models_with_single_provider(client: TestClient):
    """Test listing models with a single autoprovisioned VM in cache.

    The default fixture serves ``e2e-test/meta_llama-3.2-1b-instruct``, so the
    autoprovisioned VM lives in the ``e2e-test`` workspace; the list is
    workspace-scoped, so query that workspace.
    """
    response = client.get("/v2/workspaces/e2e-test/openai/-/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    # One autoprovisioned VM from default fixture, returns 2-part ID
    assert len(data["data"]) >= 1
    # Verify IDs are workspace/name (at least 2 parts; name may contain / for LoRA)
    for model in data["data"]:
        parts = model["id"].split("/", 1)
        assert len(parts) >= 2, f"Expected workspace/name, got: {model['id']}"


def test_list_models_lists_virtual_models(app: FastAPI, client: TestClient):
    """The list endpoint aggregates VirtualModels (not model entities)."""

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_custom_vm("ns1", "model-a")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/ns1/openai/-/v1/models")
    assert response.status_code == 200

    data = response.json()
    # VirtualModel listed once with a 2-part ID
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "ns1/model-a"
    assert data["data"][0]["owned_by"] == "ns1"


def test_list_models_is_workspace_scoped(app: FastAPI, client: TestClient):
    """The list endpoint only returns VirtualModels in the request's workspace.

    Foreign-workspace VirtualModels must not leak into another workspace's catalog.
    """

    vm_cache = VirtualModelCache()
    vm_cache.rebuild(
        [
            _custom_vm("ns1", "model-a"),
            _custom_vm("ns2", "model-b"),
        ]
    )
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/ns1/openai/-/v1/models")
    assert response.status_code == 200

    ids = {model["id"] for model in response.json()["data"]}
    assert ids == {"ns1/model-a"}
    assert "ns2/model-b" not in ids


def test_list_models_includes_custom_switchyard_vm(app: FastAPI, client: TestClient):
    """Regression: a custom (Switchyard-style) VirtualModel — one with no backing
    model entity of the same name — must appear in the OpenAI ``/v1/models`` catalog.

    Reproduces the reported gap where a routable VirtualModel served
    ``/v1/chat/completions`` but was undiscoverable from ``/v1/models`` because the
    list endpoint read model entities instead of VirtualModels.
    """

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_custom_vm("default", "qa-agent-eval-swy-8238f441")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/default/openai/-/v1/models")
    assert response.status_code == 200

    ids = {model["id"] for model in response.json()["data"]}
    assert "default/qa-agent-eval-swy-8238f441" in ids


def test_get_custom_switchyard_vm(app: FastAPI, client: TestClient):
    """Regression: a custom (Switchyard-style) VirtualModel is retrievable via
    ``GET /v1/models/{name}`` even though no model entity of that name exists."""

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_custom_vm("default", "qa-agent-eval-swy-8238f441")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/default/openai/-/v1/models/qa-agent-eval-swy-8238f441")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "default/qa-agent-eval-swy-8238f441"
    assert data["owned_by"] == "default"
    assert data["object"] == "model"


# OpenAI Get Model Tests


def test_get_model_success(client: TestClient):
    """Test getting a specific model; workspace from URL path, name is model entity name."""
    response = client.get("/v2/workspaces/e2e-test/openai/-/v1/models/meta_llama-3.2-1b-instruct")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "e2e-test/meta_llama-3.2-1b-instruct"
    assert data["owned_by"] == "e2e-test"
    assert data["object"] == "model"


def test_get_model_with_workspace_prefix_in_path(client: TestClient):
    """Test GET model when path name is workspace/model; path workspace is used, prefix in name is stripped."""
    response = client.get("/v2/workspaces/e2e-test/openai/-/v1/models/e2e-test/meta_llama-3.2-1b-instruct")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "e2e-test/meta_llama-3.2-1b-instruct"
    assert data["owned_by"] == "e2e-test"


def test_get_model_entity_not_found_single_segment(client: TestClient):
    """Test GET model with single-segment name (model only); lookup uses path workspace."""
    response = client.get("/v2/workspaces/default/openai/-/v1/models/nonexistent-entity")
    assert response.status_code == 404
    assert "No VirtualModel" in response.json()["detail"]


def test_get_model_invalid_path_workspace_returns_422(client: TestClient):
    """Test that invalid workspace in URL path (e.g. single char) returns 422."""
    response = client.get("/v2/workspaces/a/openai/-/v1/models/validmodel")
    assert response.status_code == 422
    assert "invalid" in response.json()["detail"].lower()


def test_get_model_entity_not_found_matching_workspace_prefix(client: TestClient):
    """404 is raised when a path-workspace-prefixed name points at a missing VirtualModel."""
    response = client.get("/v2/workspaces/default/openai/-/v1/models/default/nonexistent-entity")
    assert response.status_code == 404
    assert "No VirtualModel" in response.json()["detail"]


def test_get_model_entity_exists_without_providers(app: FastAPI, client: TestClient):
    """Test getting a model when an autoprovisioned VM exists (even without providers).

    The GET route resolves against the VirtualModel cache, so an autoprovisioned VM
    for an entity with no providers is still discoverable — provider resolution only
    happens at proxy (inference) time.
    """

    cache = ModelCache()
    # Model entity exists in cache but has no providers (edge case)
    cache.model_entity_info_map[("ns1", "orphan-model")] = ModelEntityInfo(
        workspace="ns1",
        name="orphan-model",
        model_providers=[],
    )
    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(_autoprovisioned_vms_for_cache(cache))
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/ns1/openai/-/v1/models/orphan-model")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "ns1/orphan-model"
    assert data["owned_by"] == "ns1"


def test_get_model_lora_compound_name_in_path(app: FastAPI, client: TestClient):
    """LoRA-style composite names (base&adapters/ws/adapter) resolve via GET /v1/models/{name:path}.

    The URL contains two "/" characters after the base workspace prefix, so FastAPI delivers
    ``name="ws/base&adapters/ws/adder"`` to the handler. The handler strips only the first
    segment ("ws/") then routes through the **base model's** VirtualModel (keyed ``(ws, "base")``)
    via ``resolve_vm_for_model`` — no per-adapter VM is created. The adapter existence check
    succeeds because the base VM is served.
    """
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_custom_vm("ws", "base", default_model_entity="ws/base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/ws/openai/-/v1/models/ws/base&adapters/ws/adder")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "ws/base&adapters/ws/adder"
    assert data["owned_by"] == "ws"


def test_get_model_lora_compound_bare_name_in_path(app: FastAPI, client: TestClient):
    """Bare LoRA composite ``{base}&adapters/{ws}/{adapter}`` resolves via ``GET /v1/models/{name:path}``.

    Routes through the base model's VM (keyed ``(ws, "base")``); no per-adapter VM exists.
    """
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_custom_vm("ws", "base", default_model_entity="ws/base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/ws/openai/-/v1/models/base&adapters/ws/adder")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "ws/base&adapters/ws/adder"
    assert data["owned_by"] == "ws"


def test_get_model_success_with_custom_provider(app: FastAPI, client: TestClient):
    """Test getting a model backed by a custom provider (via its autoprovisioned VM)."""

    cache = ModelCache()
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
                    served_model_name="vendor/model/version",
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

    response = client.get("/v2/workspaces/ns1/openai/-/v1/models/my-model")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "ns1/my-model"
    assert data["owned_by"] == "ns1"


# OpenAI Proxy Tests


def test_proxy_chat_completions_success(client: TestClient, mock_proxy_client, mock_proxy_response):
    """Test successful proxy request; workspace from URL path, model name from body."""
    request_body = {
        "model": "meta_llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock response body
    mock_proxy_response._body = [b'{"choices": [{"message": {"content": "Hi there!"}}]}']

    response = client.post("/v2/workspaces/e2e-test/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 200

    # Verify the proxied request was made
    assert mock_proxy_client.request.called
    call_args = mock_proxy_client.request.call_args

    # Verify URL was constructed correctly
    assert "localhost:11434" in call_args.kwargs["url"]
    assert "v1/chat/completions" in call_args.kwargs["url"]

    # Verify the body was modified to use the served_model_name (resolved from cache)
    body_data = json.loads(call_args.kwargs["data"])
    assert body_data["model"] == "meta/llama-3.2-1b-instruct"  # Should be rewritten to served_model_name


def test_proxy_missing_model_field(client: TestClient):
    """Test proxy request with missing model field in body."""
    request_body = {
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v2/workspaces/default/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 400
    assert "Could not extract model" in response.json()["detail"]


def test_proxy_model_name_only_uses_path_workspace_404_when_not_found(client: TestClient):
    """Test proxy with model name only; workspace from path, unknown model returns 404."""
    request_body = {
        "model": "invalid-format",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v2/workspaces/default/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 404
    assert "No VirtualModel" in response.json()["detail"]


def test_proxy_invalid_path_workspace_returns_422(client: TestClient):
    """Test that invalid workspace in URL path (e.g. single char) returns 422."""
    response = client.post(
        "/v2/workspaces/a/openai/-/v1/chat/completions",
        json={"model": "validmodel", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 422
    assert "invalid" in response.json()["detail"].lower()


def test_proxy_model_entity_not_found_matching_workspace_prefix(client: TestClient):
    """A body model whose path-workspace prefix matches but whose name does not resolve
    to any VirtualModel (or, by extension, any served entity) returns 404.
    """
    request_body = {
        "model": "default/nonexistent-entity",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v2/workspaces/default/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 404
    assert "No VirtualModel" in response.json()["detail"]


def test_proxy_body_matching_workspace_prefix_stripped(client: TestClient, mock_proxy_client, mock_proxy_response):
    """Body ``{path_workspace}/{model}`` form is accepted — the prefix is stripped and
    the request routes through the path workspace. This is the "convenience form"
    where the client echoes the id returned by ``GET /v1/models`` (which is always
    prefixed with the entity's workspace) back into ``body.model``.
    """
    mock_proxy_response._body = [b'{"choices": []}']
    request_body = {
        "model": "e2e-test/meta_llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v2/workspaces/e2e-test/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    assert mock_proxy_client.request.called


def test_proxy_body_mismatched_workspace_prefix_rejected(client: TestClient):
    """Body ``{other_workspace}/{model}`` form is rejected (422) rather than
    silently re-homed under the path workspace.

    Previously, the proxy stripped on the first ``/`` unconditionally, which meant
    ``body.model = "other-workspace/meta_llama-3.2-1b-instruct"`` was silently
    coerced into the path workspace ``e2e-test``. That conflicted with the bare
    LoRA case (``base&adapters/ws/adder`` — a legit model_entity_name that
    itself contains ``/``) and was a quiet cross-workspace footgun in its own
    right. The fix anchors the strip to the path workspace, so non-matching
    prefixes pass through unchanged and 422 via ``validate_model_entity_name``
    (a string containing ``/`` fails ``NAME_PATTERN``). Loud > quiet.
    """
    request_body = {
        "model": "other-workspace/meta_llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v2/workspaces/e2e-test/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 422


def _make_base_vm(workspace: str, base_name: str) -> SDKVirtualModel:
    """Build the base model's VirtualModel that LoRA-adapter requests route through.

    Under the adapter-through-base-VM design there is NO per-adapter VM: a request for
    ``base&adapters/{ws}/{adapter}`` resolves the VM named ``base`` (see
    ``resolve_vm_for_model``) and inherits its middleware. ``default_model_entity`` is the
    plain base entity ``{workspace}/{base_name}``; the ``&adapters/...`` suffix is spliced
    back on at proxy time so model-entity resolution still hits the adapter's served model.
    """
    return SDKVirtualModel(
        id=f"{workspace}/{base_name}",
        entity_id=f"{workspace}/{base_name}",
        name=base_name,
        workspace=workspace,
        parent=workspace,
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        default_model_entity=f"{workspace}/{base_name}",
    )


def test_proxy_body_bare_lora_composite_routes(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """Bare LoRA composite ``{base}&adapters/{ws}/{adapter}`` in ``body.model`` routes
    through the **base model's** VM without the embedded ``/`` tripping the prefix-strip.

    No per-adapter VM exists; ``resolve_vm_for_model`` keys the VM by the base segment
    (``base``) and the ``default_model_entity`` splice preserves the ``&adapters/...``
    suffix. Verifies the IGW's composite-aware ``parse_model_entity_ref`` (split on first
    '/') plus ``ModelCache``'s same convention combine to route the request to the adapter's
    served model without any string mangling along the way.
    """
    mock_proxy_response._body = [b'{"choices": []}']
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
    vm_cache.rebuild([_make_base_vm("ws", "base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    request_body = {
        "model": "base&adapters/ws/adder",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v2/workspaces/ws/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    assert mock_proxy_client.request.called

    # The proxied body must carry the served_model_name resolved via the LoRA
    # composite key (ws, "base&adapters/ws/adder"), not a truncated lookup.
    body_data = json.loads(mock_proxy_client.request.call_args.kwargs["data"])
    assert body_data["model"] == "adder-backend-id"


def test_proxy_body_cross_workspace_lora_routes(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """Cross-workspace LoRA in ``body.model`` (workspace-prefixed form) routes via the
    base workspace's provider, leaving ``adapter_ws`` segment intact in the key.

    Given ``provider.workspace = "ws-a"``, ``base_ws = "ws-a"``, and ``adapter_ws = "ws-b"``,
    all three roles are decoupled in a single fixture. The request routes through the base
    model's VM (keyed ``("ws-a", "base")``); regression guard: any code that silently clamps
    ``adapter_ws`` to ``provider.workspace`` (or ``base_ws``) would fail to find the entity
    under ``("ws-a", "base&adapters/ws-b/adapter")`` after the splice.
    """
    mock_proxy_response._body = [b'{"choices": []}']
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
    vm_cache.rebuild([_make_base_vm("ws-a", "base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    request_body = {
        "model": "ws-a/base&adapters/ws-b/adapter",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v2/workspaces/ws-a/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    assert mock_proxy_client.request.called

    # The proxied body carries the served_model_name (flat-dir encoding) resolved
    # via the cross-workspace composite key.
    call_args = mock_proxy_client.request.call_args
    assert urlparse(call_args.kwargs["url"]).hostname == "nim.workspace-a.example.com"
    assert "workspace-b" not in call_args.kwargs["url"]
    body_data = json.loads(call_args.kwargs["data"])
    assert body_data["model"] == "ws-b--adapter"


def test_proxy_body_bare_cross_workspace_lora_routes(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """Cross-workspace LoRA in bare ``body.model`` form (``base&adapters/ws-b/adapter``)
    routes through the base model's VM without the prefix-strip eating the ``ws-b/`` segment.

    Confirms that when the body model is not prefixed with the path workspace
    (``ws-a/``), the prefix-strip is a no-op and the bare composite
    ``base&adapters/ws-b/adapter`` routes through the base VM (keyed ``("ws-a", "base")``)
    and, after the splice, flows through to the cache key
    ``("ws-a", "base&adapters/ws-b/adapter")`` intact.
    """
    mock_proxy_response._body = [b'{"choices": []}']
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
    vm_cache.rebuild([_make_base_vm("ws-a", "base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    request_body = {
        "model": "base&adapters/ws-b/adapter",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v2/workspaces/ws-a/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    assert mock_proxy_client.request.called

    call_args = mock_proxy_client.request.call_args
    assert urlparse(call_args.kwargs["url"]).hostname == "nim.workspace-a.example.com"
    body_data = json.loads(call_args.kwargs["data"])
    assert body_data["model"] == "ws-b--adapter"


def test_proxy_adapter_routes_through_base_vm_middleware_and_splices_suffix(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """An adapter request runs the **base** VM's request middleware, and the splice
    replaces only the base segment of the composite — preserving the ``&adapters/...`` suffix.

    The base VM ``base`` has ``default_model_entity="ws/base"`` and a request-middleware
    plugin. A request for ``base&adapters/ws/adder`` must: (1) resolve the ``base`` VM,
    (2) run its plugin (proving adapters inherit base-VM middleware), and (3) splice to
    ``ws/base&adapters/ws/adder`` so model-entity resolution hits the adapter's served name.
    """
    mock_proxy_response._body = [b'{"choices": []}']
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

    # The base VM carries middleware; the adapter request must inherit it.
    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.process_request = AsyncMock(side_effect=lambda ctx, req, cfg: req)
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_base_vm("ws", "base")])
    registry = _make_registry_with_plugin("ws", "base", plugin, phase="request")
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.post(
        "/v2/workspaces/ws/openai/-/v1/chat/completions",
        json={"model": "base&adapters/ws/adder", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    # The base VM's request middleware ran, keyed by the base VM name.
    plugin.process_request.assert_awaited_once()
    ctx_arg = plugin.process_request.await_args.args[0]
    assert ctx_arg.virtual_model_name == "base"
    # The plugin saw the spliced composite (base segment replaced, adapter suffix preserved),
    # NOT the plain base entity.
    seeded_model = plugin.process_request.await_args.args[1].body["model"]
    assert seeded_model == "ws/base&adapters/ws/adder"

    # Served-model-name rewrite writes the *adapter's* served name upstream.
    body_data = json.loads(mock_proxy_client.request.call_args.kwargs["data"])
    assert body_data["model"] == "adder-backend-id"


def test_proxy_adapter_served_name_restored_in_response(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """The response body's ``model`` is rewritten from the adapter's served name back to the
    composite entity ref, so the caller never sees the upstream served name.
    """
    # Upstream echoes back the served name it was called with.
    mock_proxy_response._body = [json.dumps({"model": "adder-backend-id", "choices": []}).encode()]
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
    vm_cache.rebuild([_make_base_vm("ws", "base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ws/openai/-/v1/chat/completions",
        json={"model": "base&adapters/ws/adder", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    # Upstream saw the served name; the caller sees the composite entity ref restored.
    assert json.loads(mock_proxy_client.request.call_args.kwargs["data"])["model"] == "adder-backend-id"
    assert response.json()["model"] == "ws/base&adapters/ws/adder"


def test_proxy_adapter_missing_base_vm_returns_404(app: FastAPI, client: TestClient):
    """An adapter request whose base model has no VM returns 404 (base isn't served)."""
    vm_cache = VirtualModelCache()  # empty
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ws/openai/-/v1/chat/completions",
        json={"model": "base&adapters/ws/adder", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert "No VirtualModel" in response.json()["detail"]


def test_proxy_adapter_entity_missing_returns_adapter_form_error(app: FastAPI, client: TestClient):
    """When the base VM resolves but the spliced composite entity is absent from the
    ModelCache, the 404 uses the improved adapter-form message naming base + adapter.

    Exercises D3's ``raise_model_entity_not_found`` adapter branch end-to-end: the base VM
    seeds/splices ``ws/base&adapters/ws/adder``, model-entity resolution misses, and the
    error reads ``Routing table lookup failed: ... base model ws/base with adapter ws/adder``.
    """
    cache = ModelCache()  # no served models → composite entity absent
    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_base_vm("ws", "base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ws/openai/-/v1/chat/completions",
        json={"model": "base&adapters/ws/adder", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail.startswith("Routing table lookup failed:")
    assert "base model ws/base with adapter ws/adder" in detail


def test_proxy_plain_model_entity_missing_returns_plain_form_error(app: FastAPI, client: TestClient):
    """A plain (non-adapter) missing entity uses the plain ``Routing table lookup failed`` form."""
    cache = ModelCache()
    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_custom_vm("ws", "router", default_model_entity="ws/ghost-entity")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ws/openai/-/v1/chat/completions",
        json={"model": "router", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail == "Routing table lookup failed: Model entity not found for ws/ghost-entity"


def test_proxy_custom_vm_pointed_at_adapter_composite_escape_hatch(
    app: FastAPI, client: TestClient, mock_proxy_client, mock_proxy_response
):
    """Escape hatch (non-issue, regression guard): a custom plain-named VM whose
    ``default_model_entity`` IS an adapter composite still works unchanged.

    The request model is the plain VM name (not a composite), so ``resolve_vm_for_model``
    exact-matches the VM and the seed step wholesale-replaces ``body["model"]`` with the
    composite default — which then resolves to the adapter's served name. This is the
    "custom VM points at an adapter" path the design preserves.
    """
    mock_proxy_response._body = [b'{"choices": []}']
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
    vm_cache.rebuild([_custom_vm("ws", "adder-router", default_model_entity="ws/base&adapters/ws/adder")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/ws/openai/-/v1/chat/completions",
        json={"model": "adder-router", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body_data = json.loads(mock_proxy_client.request.call_args.kwargs["data"])
    assert body_data["model"] == "adder-backend-id"


def test_get_model_cross_workspace_lora(app: FastAPI, client: TestClient):
    """``GET /v1/models/ws-a/base&adapters/ws-b/adapter`` resolves a cross-workspace LoRA
    via the path workspace (``ws-a``), with ``owned_by`` reflecting the *base* workspace.

    Complements the same-workspace tests at ``test_get_model_lora_compound_name_in_path``
    and ``test_get_model_lora_compound_bare_name_in_path``. The handler strips only the
    leading ``ws-a/`` segment then routes through the base model's VM (keyed
    ``("ws-a", "base")``) via ``resolve_vm_for_model`` — no per-adapter VM exists.
    """
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_custom_vm("ws-a", "base", default_model_entity="ws-a/base")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.get("/v2/workspaces/ws-a/openai/-/v1/models/ws-a/base&adapters/ws-b/adapter")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "ws-a/base&adapters/ws-b/adapter"
    assert data["owned_by"] == "ws-a"


def test_proxy_no_providers_for_model_entity(app: FastAPI, client: TestClient):
    """An entity in the cache with no providers reaches ``virtual_model_proxy`` (which
    requires a VM, so we set up an autoprovisioned-style one) and 404s with the explicit
    "No providers" message from the entity-resolution step.
    """

    cache = ModelCache()
    # Manually create a model entity with no providers
    cache.model_entity_info_map[("ns1", "orphan-model")] = ModelEntityInfo(
        workspace="ns1",
        name="orphan-model",
        model_providers=[],
    )
    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(
        [
            SDKVirtualModel(
                id="ns1/orphan-model",
                entity_id="ns1/orphan-model",
                workspace="ns1",
                name="orphan-model",
                parent="ns1",
                db_version=1,
                default_model_entity="ns1/orphan-model",
                autoprovisioned=True,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            )
        ]
    )
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    request_body = {
        "model": "orphan-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v2/workspaces/ns1/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 404
    assert "No providers found" in response.json()["detail"]


def test_proxy_with_unresolved_provider_secret(app: FastAPI, client: TestClient):
    """Provider with an unresolved secret returns 424 even via the VM pipeline."""

    cache = ModelCache()
    provider = ModelProviderInfo(
        model_provider=ModelProvider(
            workspace="ns1",
            name="secure-provider",
            host_url="http://secure.com",
            api_key_secret_name="some-secret-id",  # Requires auth
            created_at=datetime.now(),
            updated_at=datetime.now(),
            served_models=[
                ServedModelMapping(
                    model_entity_id="ns1/secure-model",
                    served_model_name="secure-v1",
                )
            ],
        ),
        secret_value=None,  # Secret not available
    )
    cache.update_model_info(provider)
    cache.rebuild_model_entity_map()

    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(
        [
            SDKVirtualModel(
                id="ns1/secure-model",
                entity_id="ns1/secure-model",
                workspace="ns1",
                name="secure-model",
                parent="ns1",
                db_version=1,
                default_model_entity="ns1/secure-model",
                autoprovisioned=True,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            )
        ]
    )
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    request_body = {
        "model": "secure-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v2/workspaces/ns1/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 424
    assert "Could not fetch secret" in response.json()["detail"]
    assert "secret not found or unreachable" in response.json()["detail"]


def test_proxy_empty_body(client: TestClient):
    """Test proxy request with empty body."""
    response = client.post("/v2/workspaces/default/openai/-/v1/chat/completions")
    assert response.status_code == 400


def test_proxy_invalid_json(client: TestClient):
    """Test proxy request with invalid JSON body."""
    response = client.post(
        "/v2/workspaces/default/openai/-/v1/chat/completions",
        content=b"not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


def test_proxy_get_request(client: TestClient):
    """Test that GET requests work (though they might not have a body)."""
    # GET requests typically don't have a body, so this should fail
    # when trying to parse the model from the body
    response = client.get("/v2/workspaces/default/openai/-/v1/models")
    # This should hit the specific /v1/models endpoint, not the catch-all
    assert response.status_code == 200


def test_proxy_different_http_methods(client: TestClient, mock_proxy_client):
    """Test that different HTTP methods are supported."""
    request_body = {
        "model": "meta_llama-3.2-1b-instruct",
        "data": "test",
    }

    # Test PUT
    response = client.put("/v2/workspaces/e2e-test/openai/-/v1/some/endpoint", json=request_body)
    assert response.status_code == 200
    assert mock_proxy_client.request.call_args.args[0] == "PUT"

    # Test PATCH
    response = client.patch("/v2/workspaces/e2e-test/openai/-/v1/some/endpoint", json=request_body)
    assert response.status_code == 200
    assert mock_proxy_client.request.call_args.args[0] == "PATCH"

    # Test DELETE - use request method directly
    response = client.request(
        "DELETE",
        "/v2/workspaces/e2e-test/openai/-/v1/some/endpoint",
        json=request_body,
    )
    assert response.status_code == 200
    assert mock_proxy_client.request.call_args.args[0] == "DELETE"


def test_proxy_resolves_served_model_name_with_slashes(app: FastAPI, client: TestClient, mock_proxy_client):
    """Test proxy resolves served_model_name from cache, even when it contains slashes."""

    cache = ModelCache()
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
                    served_model_name="vendor/model/v1.0",
                )
            ],
        ),
    )
    cache.update_model_info(provider)
    cache.rebuild_model_entity_map()

    app.dependency_overrides[global_model_cache] = lambda: cache
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(
        [
            SDKVirtualModel(
                id="ns1/my-model",
                entity_id="ns1/my-model",
                workspace="ns1",
                name="my-model",
                parent="ns1",
                db_version=1,
                default_model_entity="ns1/my-model",
                autoprovisioned=True,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            )
        ]
    )
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    request_body = {
        "model": "my-model",
        "messages": [{"role": "user", "content": "test"}],
    }

    response = client.post("/v2/workspaces/ns1/openai/-/v1/chat/completions", json=request_body)
    assert response.status_code == 200

    # Verify the model was rewritten to the served_model_name from cache
    call_args = mock_proxy_client.request.call_args
    body_data = json.loads(call_args.kwargs["data"])
    assert body_data["model"] == "vendor/model/v1.0"


# Parse OpenAI Model Tests


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("ns1/model1", ("ns1", "model1")),
        ("default/my-model", ("default", "my-model")),
        ("workspace_with_underscores/model-with-dashes", ("workspace_with_underscores", "model-with-dashes")),
        ("e2e-test/meta_llama-3.2-1b-instruct", ("e2e-test", "meta_llama-3.2-1b-instruct")),
        # Split on first "/" only: model_entity_name may contain "/" (e.g. LoRA with &adapters/)
        ("ws/base&adapters/ws/my-adapter", ("ws", "base&adapters/ws/my-adapter")),
        ("e2e-ws/qwen-base&adapters/e2e-ws/lora-1", ("e2e-ws", "qwen-base&adapters/e2e-ws/lora-1")),
        # Cross-workspace LoRA: base_ws ("ws-a") and adapter_ws ("ws-b") differ —
        # first-/ split keeps the adapter_ws segment inside model_entity_name.
        ("ws-a/base&adapters/ws-b/adapter", ("ws-a", "base&adapters/ws-b/adapter")),
        # 3+ segments: everything after first "/" is model_entity_name
        ("ns1/model1/extra", ("ns1", "model1/extra")),
        ("ns1/model1/a/b/c", ("ns1", "model1/a/b/c")),
    ],
)
def test_parse_valid_model(model_id, expected):
    """Test parsing valid model IDs; split on first '/' so LoRA ids work."""
    assert parse_igw_openai_model(model_id) == expected


@pytest.mark.parametrize(
    "invalid_model_id",
    [
        "invalid",  # No slash
        "",  # Empty
    ],
)
def test_parse_invalid_model(invalid_model_id):
    """Test parsing invalid model IDs raises ParseOpenAIModelError."""
    with pytest.raises(ParseOpenAIModelError):
        parse_igw_openai_model(invalid_model_id)


# ---------------------------------------------------------------------------
# resolve_vm_for_model (LoRA-composite-aware VM resolution) unit tests
# ---------------------------------------------------------------------------


def test_resolve_vm_for_model_plain_name_exact_match():
    """A plain (non-composite) model name resolves the VM by exact match, unchanged."""
    vm_cache = VirtualModelCache()
    vm = _custom_vm("ws", "my-model", default_model_entity="ws/my-model")
    vm_cache.rebuild([vm])

    assert resolve_vm_for_model(vm_cache, "ws", "my-model") is vm


def test_resolve_vm_for_model_composite_keys_by_base_segment():
    """A LoRA composite resolves the VM named by the pre-``&adapters/`` base segment.

    No per-adapter VM exists, so ``base&adapters/aws/aname`` must key the VM ``base``.
    """
    vm_cache = VirtualModelCache()
    base_vm = _custom_vm("ws", "base", default_model_entity="ws/base")
    vm_cache.rebuild([base_vm])

    assert resolve_vm_for_model(vm_cache, "ws", "base&adapters/aws/aname") is base_vm
    # A composite whose base has no VM does not resolve.
    assert resolve_vm_for_model(vm_cache, "ws", "other&adapters/aws/aname") is None


def test_resolve_vm_for_model_composite_does_not_match_composite_named_vm():
    """The helper keys strictly by base segment; it never resolves a composite-named VM.

    Composite ids are not legal VM entity names, so this guards against accidentally
    depending on one existing.
    """
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_custom_vm("ws", "base&adapters/aws/aname")])

    assert resolve_vm_for_model(vm_cache, "ws", "base&adapters/aws/aname") is None


def test_resolve_vm_for_model_misses_when_uncached():
    """Returns ``None`` when neither the base nor the plain name is cached."""
    vm_cache = VirtualModelCache()
    assert resolve_vm_for_model(vm_cache, "ws", "nope") is None
    assert resolve_vm_for_model(vm_cache, "ws", "nope&adapters/aws/aname") is None


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


def test_openai_proxy_routes_via_virtual_model(app: FastAPI, client: TestClient, mock_proxy_client):
    """When a VirtualModel is in the cache, proxy resolves via its default_model_entity."""

    # VirtualModel pointing at a different model entity name
    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-alias", "e2e-test/meta_llama-3.2-1b-instruct")])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
        json={"model": "my-alias", "messages": [{"role": "user", "content": "hi"}]},
    )
    # Resolves to a real provider via the VM → model entity → ModelCache path
    assert response.status_code == 200


def test_openai_proxy_falls_back_to_model_cache_when_vm_absent(client: TestClient):
    """When no VirtualModel is cached, existing ModelCache routing still works."""
    # The conftest wires an empty VirtualModelCache, so this exercises the fallback
    response = client.post(
        "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
        json={"model": "meta_llama-3.2-1b-instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200


def test_openai_proxy_virtual_model_no_default_model_entity_no_middleware_returns_422(app: FastAPI, client: TestClient):
    """VirtualModel with no default_model_entity and no middleware → 422.

    The early guard fires before middleware runs: the VM has no way to produce
    a routable model entity, so both endpoints fail with 422 immediately.
    Consistent with the model endpoint's behavior.
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

    response = client.post(
        "/v2/workspaces/ws/openai/-/v1/chat/completions",
        json={"model": "mw-only", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Middleware pipeline execution tests
# ---------------------------------------------------------------------------


def _make_registry_with_plugin(
    workspace: str,
    vm_name: str,
    plugin,
    phase: str = "request",
) -> MiddlewareRegistry:
    """Build a MiddlewareRegistry that runs ``plugin`` in the given phase for ws/vm_name."""

    call = ResolvedMiddlewareCall(plugin_name="test-plugin", config_type="t", resolved_config={})
    registry = MiddlewareRegistry(plugins={"test-plugin": plugin})
    getattr(registry, f"{phase}_middleware_calls")[(workspace, vm_name)] = [call]
    # Other phases empty
    for p in ("request", "response", "post_response"):
        if p != phase:
            getattr(registry, f"{p}_middleware_calls")[(workspace, vm_name)] = []
    return registry


def test_openai_proxy_executes_request_middleware(app: FastAPI, client: TestClient):
    """Request middleware that rewrites body['model'] is obeyed — proxy routes to the rewritten entity."""

    plugin = MagicMock(spec=NemoInferenceMiddleware)
    # Middleware rewrites model to the real entity name
    plugin.process_request = AsyncMock(
        side_effect=lambda ctx, req, cfg: InferenceRequest(
            body={**req.body, "model": "e2e-test/meta_llama-3.2-1b-instruct"},
            headers=req.headers,
            path=req.path,
        )
    )

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-router", default_model_entity=None)])
    registry = _make_registry_with_plugin("e2e-test", "my-router", plugin, phase="request")

    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.post(
        "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
        json={"model": "my-router", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    plugin.process_request.assert_awaited_once()


def test_openai_proxy_response_middleware_receives_model_entity(app: FastAPI, client: TestClient, mock_proxy_client):
    """Provider calls use served model names, but response middleware ctx.original_request has the entity ID."""

    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.process_response = AsyncMock(side_effect=lambda ctx, response, cfg: response)

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-router", "e2e-test/meta_llama-3.2-1b-instruct")])
    registry = _make_registry_with_plugin("e2e-test", "my-router", plugin, phase="response")

    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.post(
        "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
        json={"model": "my-router", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200

    sent_body = json.loads(mock_proxy_client.request.call_args.kwargs["data"])
    assert sent_body["model"] == "meta/llama-3.2-1b-instruct"

    plugin.process_response.assert_awaited_once()
    # In the new API, the original request body is on ctx.original_request (args[0]).
    ctx_arg = plugin.process_response.await_args.args[0]
    assert ctx_arg.original_request.body["model"] == "e2e-test/meta_llama-3.2-1b-instruct"


def test_openai_proxy_request_middleware_path_rewrite_reaches_backend(
    app: FastAPI, client: TestClient, mock_proxy_client
):
    """Request middleware path rewrites are used for the upstream call and middleware context."""

    rewritten_path = "v1/rewritten/chat/completions"

    class PathRewritePlugin:
        called = False

        async def process_request(self, ctx, req, cfg):
            self.called = True
            return InferenceRequest(
                body={**req.body, "model": "e2e-test/meta_llama-3.2-1b-instruct"},
                headers=req.headers,
                path=rewritten_path,
            )

    class ProxiedPathAssertPlugin:
        called = False

        async def process_response(self, ctx, response: InferenceResponse, cfg):
            self.called = True
            assert ctx.proxied_request.path == rewritten_path
            return response

    request_plugin = PathRewritePlugin()
    response_plugin = ProxiedPathAssertPlugin()

    registry = MiddlewareRegistry(
        plugins={
            "request-plugin": request_plugin,
            "response-plugin": response_plugin,
        }
    )
    registry.request_middleware_calls[("e2e-test", "my-router")] = [
        ResolvedMiddlewareCall(plugin_name="request-plugin", config_type="t", resolved_config={})
    ]
    registry.response_middleware_calls[("e2e-test", "my-router")] = [
        ResolvedMiddlewareCall(plugin_name="response-plugin", config_type="t", resolved_config={})
    ]
    registry.post_response_middleware_calls[("e2e-test", "my-router")] = []

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-router", default_model_entity=None)])

    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.post(
        "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
        json={"model": "my-router", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert mock_proxy_client.request.call_args.kwargs["url"] == f"http://localhost:11434/{rewritten_path}"
    assert request_plugin.called is True
    assert response_plugin.called is True


def test_openai_proxy_immediate_response_skips_proxy(app: FastAPI, client: TestClient, mock_proxy_client):
    """ImmediateResponse from request middleware bypasses backend proxying entirely."""

    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.process_request = AsyncMock(
        return_value=ImmediateResponse(data={"id": "direct", "choices": [{"message": {"content": "inline"}}]})
    )

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-router", default_model_entity=None)])
    registry = _make_registry_with_plugin("e2e-test", "my-router", plugin, phase="request")

    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.post(
        "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
        json={"model": "my-router", "messages": []},
    )
    assert response.status_code == 200
    # Backend was never called
    mock_proxy_client.request.assert_not_called()


def test_openai_proxy_middleware_error_returns_correct_status(app: FastAPI, client: TestClient):
    """InferenceMiddlewareError with a custom status_code propagates to the HTTP response."""

    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.process_request = AsyncMock(side_effect=InferenceMiddlewareError("rate limited", status_code=429))

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm("e2e-test", "my-router", "e2e-test/meta_llama-3.2-1b-instruct")])
    registry = _make_registry_with_plugin("e2e-test", "my-router", plugin, phase="request")

    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.post(
        "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
        json={"model": "my-router", "messages": []},
    )
    assert response.status_code == 429


def test_openai_proxy_passthrough_vm_no_middleware_unchanged(client: TestClient):
    """Empty middleware pipeline (passthrough VM) → identical to the ModelCache fallback."""
    # The default conftest wires an empty VirtualModelCache, so this test
    # purely exercises the ModelCache fallback path with an empty registry.
    response = client.post(
        "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
        json={"model": "meta_llama-3.2-1b-instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Regression: virtual_model_proxy must not strip the workspace from body["model"]
# before dispatching to a mock provider — the mock-response-map is keyed by the
# workspace-qualified entity id and ``handle_mock_request`` re-reads
# ``request.json()`` (Starlette caches the parsed body, so any earlier mutation
# of the dict leaks through).  Without the fix the lookup misses and the mock
# returns 400 "no mock response is configured".
# ---------------------------------------------------------------------------


def test_virtual_model_proxy_mock_provider_keeps_qualified_body_model(app: FastAPI, client: TestClient, mocker):
    from nmp.core.inference_gateway.api.mock_provider.responses import (
        MOCK_RESPONSE_MAP_HEADER,
        MOCK_SERVED_MODELS_HEADER,
    )

    # Patch the function that is actually called at dispatch time.  Patching the
    # module-level ``config`` instance is unreliable: if the module was first
    # imported while a ``create_test_client`` context was active, ``config``
    # captures that transient override instance; once the context exits and
    # ``Configuration.clear_overrides()`` runs, ``_get_mock_provider_prefix``
    # resolves through ``Configuration.get_service_config`` to the @cache-d
    # baseline instance (a *different* object), so the attribute patch is lost.
    mocker.patch(
        "nmp.core.inference_gateway.api.mock_provider.utils._get_mock_provider_prefix",
        return_value="igw-mock-",
    )

    workspace = "vm-mock-ws"
    served_name = "echo-model"
    qualified_id = f"{workspace}/{served_name}"

    # Mock-response-map is keyed by the workspace-qualified entity id, exactly
    # how ``add_mock_provider`` builds it for e2e tests.
    response_map = {
        qualified_id: [{"response_code": 200, "response_body": {"id": "mock-ok", "choices": []}}],
    }
    mock_provider = ModelProvider(
        workspace=workspace,
        name=f"igw-mock-{served_name}",
        host_url="http://mock.local",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        served_models=[ServedModelMapping(model_entity_id=qualified_id, served_model_name=served_name)],
        default_extra_headers={
            MOCK_RESPONSE_MAP_HEADER: json.dumps(response_map),
            MOCK_SERVED_MODELS_HEADER: json.dumps([served_name]),
        },
        status="READY",
    )
    cache = ModelCache()
    cache.update_model_info(ModelProviderInfo(model_provider=mock_provider))
    cache.rebuild_model_entity_map()
    app.dependency_overrides[global_model_cache] = lambda: cache

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm(workspace, served_name, default_model_entity=qualified_id)])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        f"/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
        json={"model": qualified_id, "messages": [{"role": "user", "content": "hi"}]},
    )

    # Pre-fix this returned 400 with "no mock response is configured" because
    # ``virtual_model_proxy`` rewrote body["model"] to the bare served_name
    # before ``handle_mock_request`` re-read the (now-mutated) cached body.
    assert response.status_code == 200, response.text
    assert response.json().get("id") == "mock-ok"


def test_virtual_model_proxy_streaming_mock_provider_runs_response_middleware(app: FastAPI, client: TestClient, mocker):
    from nmp.core.inference_gateway.api.mock_provider.responses import (
        MOCK_RESPONSE_MAP_HEADER,
        MOCK_SERVED_MODELS_HEADER,
    )

    mocker.patch(
        "nmp.core.inference_gateway.api.mock_provider.utils._get_mock_provider_prefix",
        return_value="igw-mock-",
    )

    workspace = "vm-mock-ws"
    served_name = "stream-model"
    qualified_id = f"{workspace}/{served_name}"
    mock_body = {
        "id": "mock-stream",
        "object": "chat.completion",
        "model": qualified_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
    }
    response_map = {
        qualified_id: [{"response_code": 200, "response_body": mock_body}],
    }
    mock_provider = ModelProvider(
        workspace=workspace,
        name=f"igw-mock-{served_name}",
        host_url="http://mock.local",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        served_models=[ServedModelMapping(model_entity_id=qualified_id, served_model_name=served_name)],
        default_extra_headers={
            MOCK_RESPONSE_MAP_HEADER: json.dumps(response_map),
            MOCK_SERVED_MODELS_HEADER: json.dumps([served_name]),
        },
        status="READY",
    )
    cache = ModelCache()
    cache.update_model_info(ModelProviderInfo(model_provider=mock_provider))
    cache.rebuild_model_entity_map()
    app.dependency_overrides[global_model_cache] = lambda: cache

    vm_cache = VirtualModelCache()
    vm_cache.rebuild([_make_sdk_vm(workspace, served_name, default_model_entity=qualified_id)])
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    plugin = MagicMock(spec=NemoInferenceMiddleware)

    async def process_response(ctx, response: InferenceResponse, cfg):
        assert ctx.original_request.body["stream"] is True
        assert not isinstance(response.result, dict)
        return response

    plugin.process_response = AsyncMock(side_effect=process_response)
    registry = _make_registry_with_plugin(workspace, served_name, plugin, phase="response")
    app.dependency_overrides[global_middleware_registry] = lambda: registry

    response = client.post(
        f"/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
        json={"model": qualified_id, "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    plugin.process_response.assert_awaited_once()

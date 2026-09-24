# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the Inference Gateway with mock provider mode.

These tests verify that mock provider mode works through the real platform
subprocess, exercising provider CRUD, model entity routing, OpenAI routing,
chat completions, streaming, and error simulation.

Mock provider mode is enabled by the NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX
env var set in conftest.py. Tests use ``add_mock_provider()`` from nhx.testing
to create providers that return canned responses without a real inference backend.
"""

import uuid
from typing import Any, cast

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import InternalServerError
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.virtual_models.client import VirtualModelsClient
from nhx.testing import MockProviderResponse, add_mock_provider

from e2e.utils import collect_sse_chunks


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Provider CRUD
# ---------------------------------------------------------------------------


def test_provider_create_and_list(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """Create a mock provider and verify it appears in the provider list."""
    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name=_unique_name("crud-provider"),
        mock_response_body={"id": "chatcmpl-test", "choices": []},
    )

    providers = ModelsClient.from_client(client).list_providers(workspace=workspace)
    names = [p.name for p in providers.items()]
    assert provider.name in names


def test_provider_create_and_delete(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """Create then delete a mock provider."""
    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name=_unique_name("delete-provider"),
        mock_response_body={"id": "chatcmpl-test", "choices": []},
    )

    models = ModelsClient.from_client(client)
    models.delete_provider(workspace=workspace, name=provider.name)

    providers = models.list_providers(workspace=workspace)
    names = [p.name for p in providers.items()]
    assert provider.name not in names


# ---------------------------------------------------------------------------
# Chat completions via provider route
# ---------------------------------------------------------------------------


def test_chat_completion_via_provider_route(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """Send a chat completion request routed by provider name."""
    gateway = InferenceGatewayClient.from_client(client)
    chat_response = {
        "id": "chatcmpl-provider",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from provider route!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }

    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name=_unique_name("provider-chat"),
        mock_response_body=chat_response,
    )

    response = gateway.provider_post(
        trailing_uri="v1/chat/completions",
        name=provider.name,
        workspace=workspace,
        body=JsonBody({"model": "test", "messages": [{"role": "user", "content": "Hi"}]}),
    ).data()
    response = cast(dict[str, Any], response)

    assert response["id"] == "chatcmpl-provider"
    assert response["choices"][0]["message"]["content"] == "Hello from provider route!"
    assert response["usage"]["total_tokens"] == 12


# ---------------------------------------------------------------------------
# Chat completions via model entity route
# ---------------------------------------------------------------------------


def test_chat_completion_via_model_entity_route(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """Send a chat completion request routed by model entity name."""
    gateway = InferenceGatewayClient.from_client(client)
    entity_name = _unique_name("model-entity")
    chat_response = {
        "id": "chatcmpl-model",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from model route!"},
                "finish_reason": "stop",
            }
        ],
    }

    add_mock_provider(
        sdk,
        workspace=workspace,
        name=entity_name,
        mock_response_body=chat_response,
    )

    response = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name=entity_name,
        workspace=workspace,
        body=JsonBody({"model": "test", "messages": [{"role": "user", "content": "Hi"}]}),
    ).data()
    response = cast(dict[str, Any], response)

    assert response["id"] == "chatcmpl-model"
    assert response["choices"][0]["message"]["content"] == "Hello from model route!"


# ---------------------------------------------------------------------------
# Chat completions via OpenAI-compatible route
# ---------------------------------------------------------------------------


def test_chat_completion_via_openai_route(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """Send a chat completion request via the OpenAI-compatible route."""
    gateway = InferenceGatewayClient.from_client(client)
    entity_name = _unique_name("openai-model")
    chat_response = {
        "id": "chatcmpl-openai",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from OpenAI route!"},
                "finish_reason": "stop",
            }
        ],
    }

    add_mock_provider(
        sdk,
        workspace=workspace,
        name=entity_name,
        mock_response_body=chat_response,
    )

    response = gateway.openai_post(
        trailing_uri="v1/chat/completions",
        workspace=workspace,
        body=JsonBody(
            {
                "model": f"{workspace}/{entity_name}",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        ),
    ).data()
    response = cast(dict[str, Any], response)

    assert response["id"] == "chatcmpl-openai"
    assert response["choices"][0]["message"]["content"] == "Hello from OpenAI route!"


# ---------------------------------------------------------------------------
# Streaming chat completions
# ---------------------------------------------------------------------------


def test_streaming_chat_completion(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """Streaming chat completion returns SSE chunks with content."""
    chat_response = {
        "id": "chatcmpl-stream",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "streamed response"},
                "finish_reason": "stop",
            }
        ],
    }

    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name=_unique_name("stream-provider"),
        mock_response_body=chat_response,
    )

    # Make a raw streaming request via httpx
    with client._client.stream(
        "POST",
        f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider.name}/-/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        chunks = collect_sse_chunks(response)

    assert len(chunks) > 0
    # Reassemble streamed content
    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk.get("choices"))
    assert "streamed" in content or "response" in content


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


def test_model_list_via_openai_route(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """The OpenAI /v1/models endpoint lists routable VirtualModels.

    Adding a mock provider creates a model entity, for which the reconciler
    autoprovisions a VirtualModel of the same name — so it appears in the catalog
    (as ``workspace/name``).
    """
    gateway = InferenceGatewayClient.from_client(client)
    entity_name = _unique_name("listable-model")
    add_mock_provider(
        sdk,
        workspace=workspace,
        name=entity_name,
        mock_response_body={"id": "chatcmpl-test", "choices": []},
    )

    models = gateway.list_openai_models(workspace=workspace).data()
    model_ids = [m.id for m in models.data]
    # The autoprovisioned VirtualModel should appear (as workspace/entity_name)
    assert f"{workspace}/{entity_name}" in model_ids


# ---------------------------------------------------------------------------
# Error simulation
# ---------------------------------------------------------------------------


def test_mock_provider_error_simulation(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """Mock providers can simulate HTTP error responses."""
    gateway = InferenceGatewayClient.from_client(client)
    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name=_unique_name("error-provider"),
        mock_response_body={"error": {"message": "simulated failure", "type": "server_error"}},
        mock_status=500,
    )

    with pytest.raises(InternalServerError) as exc_info:
        gateway.provider_post(
            trailing_uri="v1/chat/completions",
            name=provider.name,
            workspace=workspace,
            body=JsonBody({"model": "test", "messages": []}),
        ).data()

    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# Per-model sequential responses
# ---------------------------------------------------------------------------


def test_per_model_sequential_responses(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """Mock providers support different sequential responses per model."""
    gateway = InferenceGatewayClient.from_client(client)
    entity_main = _unique_name("main-llm")
    entity_safety = _unique_name("safety-llm")

    add_mock_provider(
        sdk,
        workspace=workspace,
        name=_unique_name("multi-model"),
        mock_response_body_by_model={
            f"{workspace}/{entity_main}": [
                MockProviderResponse(
                    response_code=200,
                    response_body={
                        "id": "main-1",
                        "choices": [{"message": {"role": "assistant", "content": "main response"}}],
                    },
                ),
            ],
            f"{workspace}/{entity_safety}": [
                MockProviderResponse(
                    response_code=200,
                    response_body={
                        "id": "safety-1",
                        "choices": [{"message": {"role": "assistant", "content": '{"safe": true}'}}],
                    },
                ),
                MockProviderResponse(
                    response_code=200,
                    response_body={
                        "id": "safety-2",
                        "choices": [{"message": {"role": "assistant", "content": '{"safe": false}'}}],
                    },
                ),
            ],
        },
        served_models={entity_main: entity_main, entity_safety: entity_safety},
    )

    # Main model returns its response
    resp = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name=entity_main,
        workspace=workspace,
        body=JsonBody({"model": f"{workspace}/{entity_main}", "messages": []}),
    ).data()
    assert resp["id"] == "main-1"

    # Safety model returns first response, then second
    resp1 = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name=entity_safety,
        workspace=workspace,
        body=JsonBody({"model": f"{workspace}/{entity_safety}", "messages": []}),
    ).data()
    assert resp1["id"] == "safety-1"

    resp2 = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name=entity_safety,
        workspace=workspace,
        body=JsonBody({"model": f"{workspace}/{entity_safety}", "messages": []}),
    ).data()
    assert resp2["id"] == "safety-2"

    # Third call clamps to last response
    resp3 = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name=entity_safety,
        workspace=workspace,
        body=JsonBody({"model": f"{workspace}/{entity_safety}", "messages": []}),
    ).data()
    assert resp3["id"] == "safety-2"


# ---------------------------------------------------------------------------
# Virtual model CRUD
# ---------------------------------------------------------------------------


def test_virtual_model_created_by_mock_provider(sdk: NeMoHelix, client: NemoClient, workspace: str):
    """add_mock_provider creates a passthrough VirtualModel for each served entity."""
    entity_name = _unique_name("vm-check")
    add_mock_provider(
        sdk,
        workspace=workspace,
        name=entity_name,
        mock_response_body={"id": "chatcmpl-test", "choices": []},
    )

    vm = VirtualModelsClient.from_client(client).get_virtual_model(workspace=workspace, name=entity_name).data()
    assert vm.name == entity_name
    assert vm.autoprovisioned is True
    assert vm.default_model_entity == f"{workspace}/{entity_name}"

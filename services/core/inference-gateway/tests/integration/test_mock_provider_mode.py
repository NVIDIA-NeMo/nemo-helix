# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IGW mock provider mode.

These tests verify that mock provider mode works correctly through the full
IGW service, including all three route types (provider, model, openai)
and all supported HTTP methods.

Note: These tests use function-scoped fixtures to avoid parallelism issues.
Each test gets its own unique provider names to prevent conflicts when
running tests in parallel with pytest-xdist.

================================================================================
REAL-WORLD USAGE EXAMPLES
================================================================================

The first section of tests demonstrates real-world usage patterns for mock provider
mode. Use these as examples when writing E2E tests that need to mock LLM responses.

Key patterns demonstrated:
1. Creating a provider with a pre-configured mock response (default_extra_headers)
2. Passing mock responses inline via X-Mock-Response header
3. Simulating errors with X-Mock-Status header
4. Different LLM response shapes (chat completions, embeddings, multiple choices)
5. Using mock provider without a real provider (early return pattern)
"""

from __future__ import annotations

import json
import uuid
from typing import Generator

import pytest
from nemo_helix_plugin.client.errors import InternalServerError, NotFoundError, RateLimitError
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient
from nhx.core.inference_gateway.api.mock_provider import MOCK_RESPONSE_HEADER, MOCK_STATUS_HEADER
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.core.models.service import ModelsService
from nhx.testing import ClientContext, MockProviderResponse, add_mock_provider, create_test_client

DEFAULT_WORKSPACE = "default"


def _gateway(ctx: ClientContext, extra_headers: dict[str, str] | None = None) -> InferenceGatewayClient:
    """Typed gateway client for the in-process app, optionally with per-request headers."""
    client = InferenceGatewayClient(base_url="http://testserver", http_client=ctx.test_client)
    return client.with_headers(extra_headers) if extra_headers else client


def _unique_name(prefix: str) -> str:
    """Generate a unique name with a random suffix for test isolation."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _provider_route(workspace: str, provider_name: str, endpoint: str) -> str:
    """Build the provider route URL."""
    return f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider_name}/-/{endpoint}"


def _model_route(workspace: str, model_name: str, endpoint: str) -> str:
    """Build the model entity route URL."""
    return f"/apis/inference-gateway/v2/workspaces/{workspace}/model/{model_name}/-/{endpoint}"


def _openai_route(workspace: str, endpoint: str) -> str:
    """Build the OpenAI route URL."""
    return f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{endpoint}"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider_test_clients() -> Generator[ClientContext, None, None]:
    """Create a ClientContext for testing with IGW in mock provider mode.

    This is the recommended fixture for testing services that need to make
    inference calls through IGW. It provides:
    - A TestClient to back typed clients (see ``_gateway``) and ``ctx.sdk`` for add_mock_provider()
    - Use add_mock_provider() to add mock providers
    - Auto-prefixing of provider names with 'igw-mock-'

    Example:
        def test_my_llm_service(mock_provider_test_clients: ClientContext):
            provider = add_mock_provider(
                mock_provider_test_clients.sdk,
                workspace="default",
                name="judge",  # Becomes "igw-mock-judge"
                mock_response_body={"id": "chatcmpl-mock", "choices": [...]},
            )
            response = _gateway(mock_provider_test_clients).provider_post(
                trailing_uri="v1/chat/completions",
                name=provider.name,
                workspace="default",
                body=JsonBody({"model": "test", "messages": []}),
            ).data()
    """
    with create_test_client(
        InferenceGatewayService,
        ModelsService,
        client_type=ClientContext,
    ) as clients:
        yield clients


@pytest.fixture
def provider_in_cache(mock_provider_test_clients: ClientContext) -> tuple[str, str, str]:
    """Add a test provider without a default mock response.

    Uses smart defaults for health/models endpoints, or requires X-Mock-Response
    header on each request for other endpoints.

    Returns:
        Tuple of (provider_name, model_entity_name, served_model_name)
    """
    model_entity_name = _unique_name("test-model")
    served_model_name = "mock-served-model"

    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name=_unique_name("test-provider"),
        served_models={model_entity_name: served_model_name},
    )
    return provider.name, model_entity_name, served_model_name


@pytest.fixture
def provider_with_default_response(mock_provider_test_clients: ClientContext) -> tuple[str, str, str]:
    """Add a provider with default mock response configured.

    All requests to this provider return the configured mock response.

    Returns:
        Tuple of (provider_name, model_entity_name, served_model_name)
    """
    model_entity_name = _unique_name("model-with-defaults")
    served_model_name = "served-with-defaults"

    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name=_unique_name("provider-with-defaults"),
        mock_response_body={
            "id": "chatcmpl-default",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "default response"}}],
        },
        served_models={model_entity_name: served_model_name},
    )
    return provider.name, model_entity_name, served_model_name


# =============================================================================
# REAL-WORLD USAGE EXAMPLES
# =============================================================================
# These tests demonstrate common patterns for using mock provider mode in E2E tests.
# Start here if you're learning how to use mock provider mode.
# =============================================================================


def test_example_chat_completion_with_provider_default(mock_provider_test_clients: ClientContext):
    """Example: Create a provider with a pre-configured chat completion response.

    This is the recommended pattern for E2E tests where all requests to a
    provider should return the same mock response (e.g., LLM Judge tests).

    Demonstrates all 3 IGW route types:
    1. Provider route: Route by provider name
    2. Model Entity route: Route by model entity name
    3. OpenAI route: Route using OpenAI-compatible model format (workspace/entity)
    """
    # Define a complete chat completion response
    chat_completion_response = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
        },
    }

    # Create a provider with the mock response pre-configured
    # The name (without igw-mock- prefix) becomes the default model entity name
    entity_name = _unique_name("chat-provider")
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name=entity_name,
        mock_response_body=chat_completion_response,
    )
    gateway = _gateway(mock_provider_test_clients)

    # === Route 1: Provider route ===
    # Route directly to a provider by name
    response = gateway.provider_post(
        trailing_uri="v1/chat/completions",
        name=provider.name,
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody({"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}),
    ).data()
    assert response["id"] == "chatcmpl-abc123"

    # === Route 2: Model Entity route ===
    # Route by model entity name (uses default served_models mapping)
    response = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name=entity_name,
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody({"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}),
    ).data()
    assert response["id"] == "chatcmpl-abc123"

    # === Route 3: OpenAI route ===
    # Route using OpenAI-compatible format with model as workspace/entity_name
    response = gateway.openai_post(
        trailing_uri="v1/chat/completions",
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody(
            {
                "model": f"{DEFAULT_WORKSPACE}/{entity_name}",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ),
    ).data()
    assert response["id"] == "chatcmpl-abc123"
    assert response["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
    assert response["usage"]["total_tokens"] == 21


def test_example_inline_mock_response_header(mock_provider_test_clients: ClientContext):
    """Example: Pass mock response inline via X-Mock-Response header.

    This pattern is useful when:
    - You need different responses for different requests
    - You don't want to create a provider in the cache
    - You're testing one-off scenarios

    Note: When X-Mock-Response header is set, no provider lookup is needed.
    """
    # Define the response inline
    inline_response = {
        "id": "chatcmpl-inline",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"score": 5, "reasoning": "Excellent response"}',
                },
            }
        ],
    }

    # Use the typed client with extra headers - provider doesn't need to exist
    response = (
        _gateway(mock_provider_test_clients, {MOCK_RESPONSE_HEADER: json.dumps(inline_response)})
        .provider_post(
            trailing_uri="v1/chat/completions",
            name="any-provider",
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"model": "any-model", "messages": []}),
        )
        .data()
    )

    assert response["id"] == "chatcmpl-inline"
    # Parse the JSON content from the assistant's response
    content = json.loads(response["choices"][0]["message"]["content"])
    assert content["score"] == 5


def test_example_simulate_rate_limit_error(mock_provider_test_clients: ClientContext):
    """Example: Simulate error responses using mock_status.

    Use mock_status to test how your code handles various HTTP errors:
    - 429: Rate limiting
    - 500: Server errors
    - 503: Service unavailable
    """
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="rate-limited-provider",  # Becomes "igw-mock-rate-limited-provider"
        mock_response_body={
            "error": {
                "message": "Rate limit exceeded. Please retry after 60 seconds.",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        },
        mock_status=429,
    )

    with pytest.raises(RateLimitError) as exc_info:
        _gateway(mock_provider_test_clients).provider_post(
            trailing_uri="v1/chat/completions",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"model": "test", "messages": []}),
        ).data()

    assert exc_info.value.status_code == 429


def test_example_simulate_server_error(mock_provider_test_clients: ClientContext):
    """Example: Simulate a 500 Internal Server Error."""
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="error-provider",  # Becomes "igw-mock-error-provider"
        mock_response_body={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
            }
        },
        mock_status=500,
    )

    with pytest.raises(InternalServerError) as exc_info:
        _gateway(mock_provider_test_clients).provider_post(
            trailing_uri="v1/chat/completions",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"model": "test", "messages": []}),
        ).data()

    assert exc_info.value.status_code == 500


def test_example_chat_completion_multiple_choices(mock_provider_test_clients: ClientContext):
    """Example: Mock response with multiple choices (n > 1).

    Some LLM APIs support returning multiple completions in a single request.
    """
    multi_choice_response = {
        "id": "chatcmpl-multi",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Option A: Use a list."},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "Option B: Use a dictionary."},
                "finish_reason": "stop",
            },
            {
                "index": 2,
                "message": {"role": "assistant", "content": "Option C: Use a set."},
                "finish_reason": "stop",
            },
        ],
    }

    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="multi-choice-provider",  # Becomes "igw-mock-multi-choice-provider"
        mock_response_body=multi_choice_response,
    )

    response = (
        _gateway(mock_provider_test_clients)
        .provider_post(
            trailing_uri="v1/chat/completions",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"model": "test", "messages": [], "n": 3}),
        )
        .data()
    )

    assert len(response["choices"]) == 3
    assert "Option A" in response["choices"][0]["message"]["content"]
    assert "Option B" in response["choices"][1]["message"]["content"]


def test_example_embeddings_response(mock_provider_test_clients: ClientContext):
    """Example: Mock an embeddings endpoint response.

    Mock provider mode works for any endpoint, not just chat completions.
    """
    embeddings_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],  # Simplified 5-dim embedding
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }

    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="embeddings-provider",  # Becomes "igw-mock-embeddings-provider"
        mock_response_body=embeddings_response,
    )

    response = (
        _gateway(mock_provider_test_clients)
        .provider_post(
            trailing_uri="v1/embeddings",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"model": "text-embedding-ada-002", "input": "Hello world"}),
        )
        .data()
    )

    assert response["object"] == "list"
    assert len(response["data"][0]["embedding"]) == 5


def test_example_llm_judge_with_json_output(mock_provider_test_clients: ClientContext):
    """Example: LLM Judge pattern with structured JSON output.

    This demonstrates the common pattern for LLM-as-a-Judge evaluation:
    - Create a provider with a mock judge response
    - The response content is JSON with score and reasoning
    - Parse the JSON to extract evaluation results

    Shows all 3 route types working with the same mock provider.
    """
    # LLM Judge response with structured output
    judge_response = {
        "id": "chatcmpl-judge",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama-3.1-70b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "score": 4,
                            "judgment": "Good response with accurate information.",
                            "strengths": ["Accurate", "Well-structured"],
                            "weaknesses": ["Could be more concise"],
                        }
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200},
    }

    # Create the judge provider - entity_name becomes the model entity name for routing
    entity_name = _unique_name("llm-judge")
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name=entity_name,
        mock_response_body=judge_response,
    )
    gateway = _gateway(mock_provider_test_clients)
    judge_messages = [
        {"role": "system", "content": "You are an evaluation judge..."},
        {"role": "user", "content": "Evaluate: What is Python?"},
    ]

    # === Route 1: Provider route ===
    response = gateway.provider_post(
        trailing_uri="v1/chat/completions",
        name=provider.name,
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody({"model": "llama-3.1-70b-instruct", "messages": judge_messages}),
    ).data()
    judge_output = json.loads(response["choices"][0]["message"]["content"])
    assert judge_output["score"] == 4

    # === Route 2: Model Entity route ===
    response = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name=entity_name,
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody({"model": "llama-3.1-70b-instruct", "messages": judge_messages}),
    ).data()
    judge_output = json.loads(response["choices"][0]["message"]["content"])
    assert judge_output["score"] == 4

    # === Route 3: OpenAI route ===
    response = gateway.openai_post(
        trailing_uri="v1/chat/completions",
        workspace=DEFAULT_WORKSPACE,
        body=JsonBody(
            {
                "model": f"{DEFAULT_WORKSPACE}/{entity_name}",
                "messages": judge_messages,
            }
        ),
    ).data()
    judge_output = json.loads(response["choices"][0]["message"]["content"])
    assert judge_output["score"] == 4
    assert "accurate" in judge_output["judgment"].lower()
    assert len(judge_output["strengths"]) == 2


def test_example_different_http_methods(mock_provider_test_clients: ClientContext):
    """Example: Mock provider mode works with all HTTP methods (GET, POST, PUT, PATCH, DELETE).

    Different endpoints may use different HTTP methods. Mock provider mode supports all of them.
    """
    gateway = _gateway(mock_provider_test_clients)

    # GET request
    get_response = {"method": "GET", "data": "retrieved"}
    response = (
        gateway.with_headers({MOCK_RESPONSE_HEADER: json.dumps(get_response)})
        .provider_get(
            trailing_uri="v1/custom",
            name="any",
            workspace=DEFAULT_WORKSPACE,
        )
        .data()
    )
    assert response["method"] == "GET"

    # POST request
    post_response = {"method": "POST", "data": "created"}
    response = (
        gateway.with_headers({MOCK_RESPONSE_HEADER: json.dumps(post_response)})
        .provider_post(
            trailing_uri="v1/custom",
            name="any",
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"input": "test"}),
        )
        .data()
    )
    assert response["method"] == "POST"

    # PUT request
    put_response = {"method": "PUT", "data": "updated"}
    response = (
        gateway.with_headers({MOCK_RESPONSE_HEADER: json.dumps(put_response)})
        .provider_put(
            trailing_uri="v1/custom",
            name="any",
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"input": "test"}),
        )
        .data()
    )
    assert response["method"] == "PUT"

    # PATCH request
    patch_response = {"method": "PATCH", "data": "patched"}
    response = (
        gateway.with_headers({MOCK_RESPONSE_HEADER: json.dumps(patch_response)})
        .provider_patch(
            trailing_uri="v1/custom",
            name="any",
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"input": "test"}),
        )
        .data()
    )
    assert response["method"] == "PATCH"

    # DELETE request
    delete_response = {"method": "DELETE", "data": "deleted"}
    response = (
        gateway.with_headers({MOCK_RESPONSE_HEADER: json.dumps(delete_response)})
        .provider_delete(
            trailing_uri="v1/custom",
            name="any",
            workspace=DEFAULT_WORKSPACE,
        )
        .http_response.json()
    )
    assert response["method"] == "DELETE"


def test_example_dynamic_per_model_responses(mock_provider_test_clients: ClientContext):
    """Example: Configure different responses for different models, or sequential responses.

    Use mock_response_body_by_model when:
    - You need different responses for different models served by the same provider
    - You need sequential responses for the same model (ex. content safety checks)

    This is useful for testing Guardrails or other services that make multiple
    inference calls to different models.
    """
    workspace = DEFAULT_WORKSPACE
    add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=workspace,
        name=_unique_name("multi-model"),
        mock_response_body_by_model={
            # Main LLM - single response
            f"{workspace}/main-llm": [
                MockProviderResponse(
                    response_code=200,
                    response_body={"id": "main-1", "choices": [{"message": {"content": "Hello from main LLM!"}}]},
                ),
            ],
            # Content safety - sequential responses (1st call safe, 2nd call unsafe)
            f"{workspace}/content-safety": [
                MockProviderResponse(
                    response_code=200,
                    response_body={"id": "safety-1", "choices": [{"message": {"content": '{"safe": true}'}}]},
                ),
                MockProviderResponse(
                    response_code=200,
                    response_body={"id": "safety-2", "choices": [{"message": {"content": '{"safe": false}'}}]},
                ),
            ],
        },
        served_models={
            "main-llm": "main-llm",
            "content-safety": "content-safety",
        },
    )
    gateway = _gateway(mock_provider_test_clients)

    # Call main-llm via model entity route
    response = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name="main-llm",
        workspace=workspace,
        body=JsonBody({"model": f"{workspace}/main-llm", "messages": []}),
    ).data()
    assert response["id"] == "main-1"
    assert "Hello from main LLM" in response["choices"][0]["message"]["content"]

    # First call to content-safety returns "safe: true"
    response = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name="content-safety",
        workspace=workspace,
        body=JsonBody({"model": f"{workspace}/content-safety", "messages": []}),
    ).data()
    assert response["id"] == "safety-1"
    assert '"safe": true' in response["choices"][0]["message"]["content"]

    # Second call to content-safety returns "safe: false" (sequential)
    response = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name="content-safety",
        workspace=workspace,
        body=JsonBody({"model": f"{workspace}/content-safety", "messages": []}),
    ).data()
    assert response["id"] == "safety-2"
    assert '"safe": false' in response["choices"][0]["message"]["content"]

    # Third call clamps to last response
    response = gateway.model_post(
        trailing_uri="v1/chat/completions",
        name="content-safety",
        workspace=workspace,
        body=JsonBody({"model": f"{workspace}/content-safety", "messages": []}),
    ).data()
    assert response["id"] == "safety-2"  # Still returns last response


def test_example_header_overrides_provider_default(mock_provider_test_clients: ClientContext):
    """Example: Request header takes priority over provider defaults.

    If a provider has a default mock response configured, you can still
    override it for specific requests by passing extra headers to the typed client.
    """
    # Provider has a default response
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name=_unique_name("override-example"),
        mock_response_body={"source": "provider_default", "score": 3},
    )

    # Request without extra_headers uses provider default
    response = (
        _gateway(mock_provider_test_clients)
        .provider_post(
            trailing_uri="v1/test",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({}),
        )
        .data()
    )
    assert response["source"] == "provider_default"

    # Request with extra_headers overrides the default
    override_response = {"source": "request_header", "score": 5}
    response = (
        _gateway(mock_provider_test_clients, {MOCK_RESPONSE_HEADER: json.dumps(override_response)})
        .provider_post(
            trailing_uri="v1/test",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({}),
        )
        .data()
    )
    assert response["source"] == "request_header"
    assert response["score"] == 5


# =============================================================================
# Provider Route Tests - Smart Defaults
# =============================================================================


@pytest.mark.parametrize(
    "endpoint,expected_response",
    [
        ("v1/health/ready", {"status": "ready"}),
        ("v1/health/live", {"status": "live"}),
        ("health/ready", {"status": "ready"}),
        ("health/live", {"status": "live"}),
    ],
)
def test_provider_route_smart_default_health(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
    endpoint: str,
    expected_response: dict,
):
    """Test provider route returns smart defaults for health endpoints."""
    provider_name, _, _ = provider_in_cache
    client = mock_provider_test_clients.test_client

    response = client.get(_provider_route(DEFAULT_WORKSPACE, provider_name, endpoint))

    assert response.status_code == 200
    assert response.json() == expected_response


@pytest.mark.parametrize("endpoint", ["v1/models", "models"])
def test_provider_route_smart_default_models(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
    endpoint: str,
):
    """Test provider route returns configured model entity IDs for the models endpoint."""
    provider_name, model_entity_name, _ = provider_in_cache
    client = mock_provider_test_clients.test_client

    response = client.get(_provider_route(DEFAULT_WORKSPACE, provider_name, endpoint))

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == model_entity_name


# =============================================================================
# Provider Route Tests - HTTP Methods
# =============================================================================


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_provider_route_all_http_methods(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
    method: str,
):
    """Test provider route works with all HTTP methods."""
    provider_name, _, _ = provider_in_cache
    client = mock_provider_test_clients.test_client
    mock_response = {"method": method}

    request_kwargs = {
        "headers": {MOCK_RESPONSE_HEADER: json.dumps(mock_response)},
    }
    if method in ("post", "put", "patch"):
        request_kwargs["json"] = {"test": True}

    response = getattr(client, method)(
        _provider_route(DEFAULT_WORKSPACE, provider_name, "v1/test"),
        **request_kwargs,
    )

    assert response.status_code == 200
    assert response.json() == mock_response


def test_provider_route_with_custom_status_code(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
):
    """Test provider route respects X-Mock-Status header."""
    provider_name, _, _ = provider_in_cache
    client = mock_provider_test_clients.test_client
    mock_response = {"error": "rate limited"}

    response = client.post(
        _provider_route(DEFAULT_WORKSPACE, provider_name, "v1/chat/completions"),
        json={"model": "test", "messages": []},
        headers={
            MOCK_RESPONSE_HEADER: json.dumps(mock_response),
            MOCK_STATUS_HEADER: "429",
        },
    )

    assert response.status_code == 429
    assert response.json() == mock_response


def test_provider_route_with_provider_default_response(
    mock_provider_test_clients: ClientContext,
    provider_with_default_response: tuple[str, str, str],
):
    """Test provider route uses default_extra_headers mock response."""
    provider_name, _, _ = provider_with_default_response
    client = mock_provider_test_clients.test_client

    response = client.post(
        _provider_route(DEFAULT_WORKSPACE, provider_name, "v1/chat/completions"),
        json={"model": "test", "messages": []},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatcmpl-default"
    assert data["choices"][0]["message"]["content"] == "default response"


def test_provider_route_header_overrides_provider_default(
    mock_provider_test_clients: ClientContext,
    provider_with_default_response: tuple[str, str, str],
):
    """Test that request header takes priority over provider defaults."""
    provider_name, _, _ = provider_with_default_response
    client = mock_provider_test_clients.test_client
    override_response = {"overridden": True}

    response = client.post(
        _provider_route(DEFAULT_WORKSPACE, provider_name, "v1/chat/completions"),
        json={"model": "test", "messages": []},
        headers={MOCK_RESPONSE_HEADER: json.dumps(override_response)},
    )

    assert response.status_code == 200
    assert response.json() == override_response


def test_provider_route_no_response_returns_400(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
):
    """Test provider route returns 400 when no mock response is configured."""
    provider_name, _, _ = provider_in_cache
    client = mock_provider_test_clients.test_client

    response = client.post(
        _provider_route(DEFAULT_WORKSPACE, provider_name, "v1/chat/completions"),
        json={"model": "test", "messages": []},
    )

    assert response.status_code == 400
    assert "Mock provider mode is enabled" in response.json()["detail"]
    assert "X-Mock-Response" in response.json()["detail"]


# =============================================================================
# Model Entity Route Tests - Smart Defaults
# =============================================================================


@pytest.mark.parametrize(
    "endpoint,expected_response",
    [
        ("v1/health/ready", {"status": "ready"}),
        ("v1/health/live", {"status": "live"}),
        ("health/ready", {"status": "ready"}),
        ("health/live", {"status": "live"}),
    ],
)
def test_model_route_smart_default_health(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
    endpoint: str,
    expected_response: dict,
):
    """Test model entity route returns smart defaults for health endpoints."""
    _, model_entity_name, _ = provider_in_cache
    client = mock_provider_test_clients.test_client

    response = client.get(_model_route(DEFAULT_WORKSPACE, model_entity_name, endpoint))

    assert response.status_code == 200
    assert response.json() == expected_response


@pytest.mark.parametrize("endpoint", ["v1/models", "models"])
def test_model_route_smart_default_models(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
    endpoint: str,
):
    """Test model entity route returns configured model entity IDs for the models endpoint."""
    _, model_entity_name, _ = provider_in_cache
    client = mock_provider_test_clients.test_client

    response = client.get(_model_route(DEFAULT_WORKSPACE, model_entity_name, endpoint))

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == model_entity_name


# =============================================================================
# Model Entity Route Tests - HTTP Methods
# =============================================================================


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_model_route_all_http_methods(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
    method: str,
):
    """Test model entity route works with all HTTP methods."""
    _, model_entity_name, _ = provider_in_cache
    client = mock_provider_test_clients.test_client
    mock_response = {"method": method}

    request_kwargs = {
        "headers": {MOCK_RESPONSE_HEADER: json.dumps(mock_response)},
    }
    if method in ("post", "put", "patch"):
        request_kwargs["json"] = {"test": True}

    response = getattr(client, method)(
        _model_route(DEFAULT_WORKSPACE, model_entity_name, "v1/test"),
        **request_kwargs,
    )

    assert response.status_code == 200
    assert response.json() == mock_response


def test_model_route_with_provider_default_response(
    mock_provider_test_clients: ClientContext,
    provider_with_default_response: tuple[str, str, str],
):
    """Test model entity route uses provider's default_extra_headers."""
    _, model_entity_name, _ = provider_with_default_response
    client = mock_provider_test_clients.test_client

    response = client.post(
        _model_route(DEFAULT_WORKSPACE, model_entity_name, "v1/chat/completions"),
        json={"model": "test", "messages": []},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatcmpl-default"


# =============================================================================
# OpenAI Route Tests - HTTP Methods
# =============================================================================


@pytest.mark.parametrize("method", ["post", "put", "patch"])
def test_openai_route_http_methods_with_body(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
    method: str,
):
    """Test OpenAI route works with HTTP methods that have a body.

    Note: DELETE is excluded because OpenAI routes require a model field
    in the request body for routing, and DELETE requests typically don't
    have a body.
    """
    _, model_entity_name, served_model_name = provider_in_cache
    client = mock_provider_test_clients.test_client
    mock_response = {"method": method}
    model_id = f"{DEFAULT_WORKSPACE}/{model_entity_name}/{served_model_name}"

    response = getattr(client, method)(
        _openai_route(DEFAULT_WORKSPACE, "v1/chat/completions"),
        json={"model": model_id},
        headers={MOCK_RESPONSE_HEADER: json.dumps(mock_response)},
    )

    assert response.status_code == 200
    assert response.json() == mock_response


def test_openai_route_with_provider_default_response(
    mock_provider_test_clients: ClientContext,
    provider_with_default_response: tuple[str, str, str],
):
    """Test OpenAI route uses provider's default_extra_headers."""
    _, model_entity_name, _ = provider_with_default_response
    client = mock_provider_test_clients.test_client
    # Workspace from URL path; body is model entity name only (no nested slashes).
    response = client.post(
        _openai_route(DEFAULT_WORKSPACE, "v1/chat/completions"),
        json={"model": model_entity_name, "messages": []},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatcmpl-default"


# =============================================================================
# Early Return Tests (no provider/model entity required)
# =============================================================================


def test_provider_route_early_return_without_provider(
    mock_provider_test_clients: ClientContext,
):
    """Test that provider route works without a real provider when X-Mock-Response is set."""
    client = mock_provider_test_clients.test_client
    mock_response = {"early_return": True}

    response = client.post(
        _provider_route(DEFAULT_WORKSPACE, "nonexistent-provider", "v1/chat/completions"),
        json={"model": "test", "messages": []},
        headers={MOCK_RESPONSE_HEADER: json.dumps(mock_response)},
    )

    assert response.status_code == 200
    assert response.json() == mock_response


def test_model_route_early_return_without_model_entity(
    mock_provider_test_clients: ClientContext,
):
    """Test that model route works without a real model entity when X-Mock-Response is set."""
    client = mock_provider_test_clients.test_client
    mock_response = {"early_return": True}

    response = client.post(
        _model_route(DEFAULT_WORKSPACE, "nonexistent-model", "v1/chat/completions"),
        json={"model": "test", "messages": []},
        headers={MOCK_RESPONSE_HEADER: json.dumps(mock_response)},
    )

    assert response.status_code == 200
    assert response.json() == mock_response


def test_openai_route_early_return_without_model(
    mock_provider_test_clients: ClientContext,
):
    """Test that OpenAI route works without a real model when X-Mock-Response is set."""
    client = mock_provider_test_clients.test_client
    mock_response = {"early_return": True}

    response = client.post(
        _openai_route(DEFAULT_WORKSPACE, "v1/chat/completions"),
        json={"model": "nonexistent/model/name", "messages": []},
        headers={MOCK_RESPONSE_HEADER: json.dumps(mock_response)},
    )

    assert response.status_code == 200
    assert response.json() == mock_response


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_invalid_json_in_mock_response_header(
    mock_provider_test_clients: ClientContext,
    provider_in_cache: tuple[str, str, str],
):
    """Test that invalid JSON in X-Mock-Response header returns 400."""
    provider_name, _, _ = provider_in_cache
    client = mock_provider_test_clients.test_client

    response = client.post(
        _provider_route(DEFAULT_WORKSPACE, provider_name, "v1/chat/completions"),
        json={"model": "test", "messages": []},
        headers={MOCK_RESPONSE_HEADER: "not valid json"},
    )

    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


def test_provider_not_found_still_returns_404(
    mock_provider_test_clients: ClientContext,
):
    """Test that mock mode doesn't bypass provider lookup - returns 404."""
    client = mock_provider_test_clients.test_client

    response = client.get(
        _provider_route(DEFAULT_WORKSPACE, "nonexistent", "v1/health/ready"),
    )

    assert response.status_code == 404


def test_model_entity_not_found_still_returns_404(
    mock_provider_test_clients: ClientContext,
):
    """Test that mock mode doesn't bypass model entity lookup - returns 404."""
    client = mock_provider_test_clients.test_client

    response = client.get(
        _model_route(DEFAULT_WORKSPACE, "nonexistent", "v1/health/ready"),
    )

    assert response.status_code == 404


# =============================================================================
# E2E Use Case: LLM Judge Mock Response
# =============================================================================


def test_llm_judge_e2e_use_case(mock_provider_test_clients: ClientContext):
    """Test the LLM Judge E2E use case as documented in the README.

    This simulates how E2E tests for the evaluator service would use
    mock provider mode to test LLM Judge workflows without real inference backends.
    """
    judge_response = {
        "id": "chatcmpl-mock-judge",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "mock-judge-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"score": 5, "judgment": "The response is accurate and helpful."}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name=_unique_name("mock-judge"),
        mock_response_body=judge_response,
    )

    client = mock_provider_test_clients.test_client
    response = client.post(
        _provider_route(DEFAULT_WORKSPACE, provider.name, "v1/chat/completions"),
        json={
            "model": "mock-judge-model",
            "messages": [
                {"role": "system", "content": "You are a judge..."},
                {"role": "user", "content": "Rate this response: ..."},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatcmpl-mock-judge"
    assert data["model"] == "mock-judge-model"

    judge_content = json.loads(data["choices"][0]["message"]["content"])
    assert judge_content["score"] == 5
    assert "accurate" in judge_content["judgment"]


# =============================================================================
# nhx.testing.mock_provider Fixture Tests
# =============================================================================
# These tests verify that the nhx.testing.mock_provider fixture works correctly.
# Other services can use this fixture to test with IGW in mock provider mode.
#
# The mock_provider_test_clients fixture is the recommended way to test services that need
# to make inference calls through IGW. It provides a cleaner API than manually
# managing the model cache.
# =============================================================================


def test_fixture_basic_usage(mock_provider_test_clients: ClientContext):
    """Test basic mock_provider_test_clients fixture usage.

    This is the simplest usage pattern: use the fixture and make requests
    with inline X-Mock-Response headers via the typed gateway client.
    """
    assert mock_provider_test_clients.test_client is not None

    mock_response = {"test": "response"}
    response = (
        _gateway(mock_provider_test_clients, {MOCK_RESPONSE_HEADER: json.dumps(mock_response)})
        .provider_post(
            trailing_uri="v1/test",
            name="any-provider",
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({}),
        )
        .data()
    )

    assert response == mock_response


def test_fixture_add_provider(mock_provider_test_clients: ClientContext):
    """Test MockProviderContext.add_provider method.

    Use add_provider to create a mock provider with a pre-configured response.
    This is cleaner than manually creating ModelProviderInfo objects.
    The provider name is auto-prefixed with 'igw-mock-'.
    """
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="fixture-test-provider",  # Will become "igw-mock-fixture-test-provider"
        mock_response_body={
            "id": "chatcmpl-fixture",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        },
    )

    assert provider.name == "igw-mock-fixture-test-provider"
    assert provider.workspace == DEFAULT_WORKSPACE

    response = (
        _gateway(mock_provider_test_clients)
        .provider_post(
            trailing_uri="v1/chat/completions",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"model": "test", "messages": []}),
        )
        .data()
    )

    assert response["id"] == "chatcmpl-fixture"


def test_fixture_add_provider_with_error_status(mock_provider_test_clients: ClientContext):
    """Test MockProviderContext.add_provider with mock_status for error simulation.

    Use mock_status to configure the HTTP status code returned by the mock.
    This is useful for testing error handling in your service.
    """
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="error-provider",  # Becomes "igw-mock-error-provider"
        mock_response_body={"error": "rate limited"},
        mock_status=429,
    )

    with pytest.raises(RateLimitError) as exc_info:
        _gateway(mock_provider_test_clients).provider_post(
            trailing_uri="v1/chat/completions",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"model": "test", "messages": []}),
        ).data()

    assert exc_info.value.status_code == 429


def test_fixture_add_provider_with_model_entity_routing(mock_provider_test_clients: ClientContext):
    """Test MockProviderContext.add_provider with served_models for model entity routing.

    Use served_models to configure model entity routing. This allows requests
    to the /model/{model_entity_name}/ route to be routed to your mock provider.
    """
    add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="model-provider",  # Becomes "igw-mock-model-provider"
        mock_response_body={"id": "via-model-entity"},
        served_models={"my-model": "served-model-name"},
    )

    response = (
        _gateway(mock_provider_test_clients)
        .model_post(
            trailing_uri="v1/chat/completions",
            name="my-model",
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({"model": "test", "messages": []}),
        )
        .data()
    )

    assert response["id"] == "via-model-entity"


def test_fixture_remove_provider(mock_provider_test_clients: ClientContext):
    """Test removing a provider via the models client.

    Demonstrates deleting a provider and verifying it's no longer accessible.
    Note: The API delete removes from database, but we also need to clear from
    the IGW model cache for immediate effect.
    """
    from nhx.core.inference_gateway.api.dependencies import global_model_cache

    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="to-remove",
        mock_response_body={"temporary": True},
    )

    # Verify provider works
    response = (
        _gateway(mock_provider_test_clients)
        .provider_post(
            trailing_uri="v1/test",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({}),
        )
        .data()
    )
    assert response == {"temporary": True}

    # Delete the provider via the models client (removes from database)
    ModelsClient(base_url="http://testserver", http_client=mock_provider_test_clients.test_client).delete_provider(
        workspace=DEFAULT_WORKSPACE,
        name=provider.name,
    )

    # Also remove from IGW model cache for immediate effect
    model_cache = global_model_cache()
    key = (DEFAULT_WORKSPACE, provider.name)
    if key in model_cache.workspace_name_provider_map:
        del model_cache.workspace_name_provider_map[key]
        model_cache.rebuild_model_entity_map()

    # Verify provider is gone (404 without X-Mock-Response header)
    with pytest.raises(NotFoundError) as exc_info:
        _gateway(mock_provider_test_clients).provider_get(
            trailing_uri="v1/health/ready",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
        ).data()
    assert exc_info.value.status_code == 404


def test_fixture_client_access(mock_provider_test_clients: ClientContext):
    """Test that ClientContext exposes the TestClient the typed clients are built on."""
    assert mock_provider_test_clients.test_client is not None
    assert _gateway(mock_provider_test_clients).base_url == "http://testserver"


def test_fixture_llm_judge_pattern(mock_provider_test_clients: ClientContext):
    """Test LLM Judge pattern using the mock_provider_test_clients fixture.

    This demonstrates the recommended pattern for testing services that need
    to call an LLM Judge through IGW (e.g., evaluator service tests).
    """
    judge_response = {
        "id": "chatcmpl-judge",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "score": 4,
                            "reasoning": "Good response with accurate information.",
                        }
                    ),
                },
                "finish_reason": "stop",
            }
        ],
    }

    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="llm-judge",  # Becomes "igw-mock-llm-judge"
        mock_response_body=judge_response,
    )

    response = (
        _gateway(mock_provider_test_clients)
        .provider_post(
            trailing_uri="v1/chat/completions",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody(
                {
                    "model": "judge-model",
                    "messages": [
                        {"role": "system", "content": "You are a judge..."},
                        {"role": "user", "content": "Evaluate: ..."},
                    ],
                }
            ),
        )
        .data()
    )

    judge_output = json.loads(response["choices"][0]["message"]["content"])
    assert judge_output["score"] == 4
    assert "accurate" in judge_output["reasoning"].lower()


def test_fixture_isolation(mock_provider_test_clients: ClientContext):
    """Test that mock_provider_test_clients fixture properly isolates test state.

    Each test using mock_provider_test_clients gets a fresh context. Providers added in
    one test won't be visible in another test.
    """
    # Add a provider in this test
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace=DEFAULT_WORKSPACE,
        name="isolated-provider",  # Becomes "igw-mock-isolated-provider"
        mock_response_body={"context": 1},
    )

    # Verify it works in this context
    response = (
        _gateway(mock_provider_test_clients)
        .provider_post(
            trailing_uri="v1/test",
            name=provider.name,
            workspace=DEFAULT_WORKSPACE,
            body=JsonBody({}),
        )
        .data()
    )
    assert response == {"context": 1}

    # Create a new context (simulating another test)
    with create_test_client(
        InferenceGatewayService,
        ModelsService,
        client_type=ClientContext,
    ) as new_ctx:
        # Provider should not exist in the new context
        with pytest.raises(NotFoundError) as exc_info:
            _gateway(new_ctx).provider_get(
                trailing_uri="v1/health/ready",
                name=provider.name,
                workspace=DEFAULT_WORKSPACE,
            ).data()
        assert exc_info.value.status_code == 404

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent configured a MockLLM provider and can make inference
calls that return deterministic responses through the Inference Gateway.

Tests:
- Mock provider igw-mock-test-llm exists in the default workspace
- Provider has the X-Mock-Response header configured
- Chat completion request returns the expected deterministic response
"""

import os

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient

WORKSPACE = "default"
PROVIDER_NAME = "igw-mock-test-llm"
MODEL_NAME = "test-llm"
EXPECTED_CONTENT = "This is a deterministic mock response from the test LLM."


@pytest.fixture
def client() -> NemoClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NemoClient(base_url=nhx_base_url, workspace=WORKSPACE)


def test_mock_provider_exists(client: NemoClient) -> None:
    """Verify the mock provider was created by the agent."""
    response = ModelsClient.from_client(client).list_providers()
    provider_names = [p.name for p in response.items()]
    assert PROVIDER_NAME in provider_names, f"Provider '{PROVIDER_NAME}' not found. Found providers: {provider_names}"


def test_mock_provider_has_mock_header(client: NemoClient) -> None:
    """Verify the provider is configured with the X-Mock-Response header."""
    provider = ModelsClient.from_client(client).get_provider(name=PROVIDER_NAME, workspace=WORKSPACE).data()
    headers = provider.default_extra_headers or {}
    assert "X-Mock-Response" in headers, (
        f"Provider '{PROVIDER_NAME}' missing X-Mock-Response header. Headers: {headers}"
    )


def test_chat_completion_returns_deterministic_response(client: NemoClient) -> None:
    """Verify inference through the gateway returns the expected mock response."""
    response = (
        InferenceGatewayClient.from_client(client)
        .provider_post(
            trailing_uri="v1/chat/completions",
            name=PROVIDER_NAME,
            body=JsonBody(
                {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            ),
        )
        .data()
    )

    assert response is not None, "Gateway returned no response"
    data = response.model_dump() if hasattr(response, "model_dump") else response

    assert "choices" in data, f"Response missing 'choices': {data}"
    assert len(data["choices"]) > 0, f"Response has empty choices: {data}"
    choice = data["choices"][0]
    assert "message" in choice, f"Choice missing 'message': {choice}"
    assert "content" in choice["message"], f"Message missing 'content': {choice['message']}"
    content = choice["message"]["content"]
    assert content == EXPECTED_CONTENT, f"Expected deterministic content '{EXPECTED_CONTENT}', got '{content}'"

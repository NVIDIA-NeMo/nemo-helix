# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the mock chat model is available and that chat completions
work through the Inference Gateway (IGW).

Tests:
- Mock provider igw-mock-chat-model exists and is configured
- Chat completion request returns the expected mock response
- Response has the correct OpenAI-compatible structure
"""

import os
import time

import pytest
from nemo_helix import NeMoHelix

WORKSPACE = "default"
PROVIDER_NAME = "igw-mock-chat-model"
MODEL_NAME = "chat-model"
EXPECTED_CONTENT = "The capital of France is Paris. It is known as the City of Light."


@pytest.fixture
def client() -> NeMoHelix:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url, workspace=WORKSPACE)


def test_mock_provider_exists(client: NeMoHelix) -> None:
    """Verify the mock chat provider was created by the setup script."""
    response = client.inference.providers.list()
    provider_names = [p.name for p in response.data]
    assert PROVIDER_NAME in provider_names, f"Provider '{PROVIDER_NAME}' not found. Found providers: {provider_names}"


def test_chat_completion_via_provider_route(client: NeMoHelix) -> None:
    """Verify chat completions work through the provider gateway route."""
    response = client.inference.gateway.provider.post(
        "v1/chat/completions",
        name=PROVIDER_NAME,
        body={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 50,
        },
    )

    assert response is not None, "Gateway returned no response"
    data = response.model_dump() if hasattr(response, "model_dump") else response
    print(f"Provider route response: {data}")

    # Verify OpenAI-compatible structure
    assert "choices" in data, f"Response missing 'choices': {data}"
    assert len(data["choices"]) > 0, f"Response has empty choices: {data}"
    content = data["choices"][0]["message"]["content"]
    assert content == EXPECTED_CONTENT, f"Expected mock content '{EXPECTED_CONTENT}', got '{content}'"


def test_chat_completion_via_model_route(client: NeMoHelix) -> None:
    """Verify chat completions work through the model entity gateway route."""
    response = None
    last_error: Exception | None = None
    for _ in range(30):
        try:
            response = client.inference.gateway.model.post(
                "v1/chat/completions",
                name=MODEL_NAME,
                body={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Tell me something interesting."}],
                    "max_tokens": 50,
                },
            )
            break
        except Exception as exc:
            last_error = exc
            if "Model entity not found" not in str(exc):
                raise
            time.sleep(1.0)

    if response is None:
        # Routing readiness should be guaranteed by setup-mock.py before tests
        # run. Skipping here would silently turn real routing regressions into
        # green CI; fail the test instead so the regression surfaces.
        pytest.fail(f"Model route not available after retries: {last_error}")

    assert response is not None, "Gateway returned no response"
    data = response.model_dump() if hasattr(response, "model_dump") else response
    print(f"Model route response: {data}")

    assert "choices" in data, f"Response missing 'choices': {data}"
    assert len(data["choices"]) > 0, f"Response has empty choices: {data}"
    content = data["choices"][0]["message"]["content"]
    assert content == EXPECTED_CONTENT, f"Expected mock content '{EXPECTED_CONTENT}', got '{content}'"

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent registered an inference provider and
successfully made a completions request through IGW.

Tests:
- Provider named nvidia-inference exists with correct host URL
- Provider has an API key secret configured
- Actual inference call through IGW returns a valid response
"""

import base64
import json
import os

import pytest
from nemo_helix import NeMoHelix
from trace_reader import get_session

WORKSPACE = "default"
PROVIDER_NAME = "nvidia-inference"
INFERENCE_MODEL = "aws/anthropic/bedrock-claude-sonnet-4-5-v1"
# The agent may use either of these NVIDIA API URLs
ACCEPTED_HOST_URLS = [
    "https://inference-api.nvidia.com",
    "https://inference-api.nvidia.com/v1",
    "https://integrate.api.nvidia.com/v1",
    "https://integrate.api.nvidia.com",
]


def _make_unsigned_jwt() -> str:
    """Create an unsigned JWT (alg=none) for local quickstart auth."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"sub": "verifier@harbor.local", "email": "verifier@harbor.local"}).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}."


@pytest.fixture
def client() -> NeMoHelix:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url, workspace=WORKSPACE, access_token=_make_unsigned_jwt())


# --- Provider setup checks (3/5 weight) ---


def test_provider_exists(client: NeMoHelix) -> None:
    """Verify the nvidia-inference provider was registered."""
    response = client.inference.providers.list()
    provider_names = [p.name for p in response.data]
    assert PROVIDER_NAME in provider_names, f"Provider '{PROVIDER_NAME}' not found. Found providers: {provider_names}"


def test_provider_host_url(client: NeMoHelix) -> None:
    """Verify the provider points to an NVIDIA inference API endpoint."""
    response = client.inference.providers.retrieve(name=PROVIDER_NAME)
    host = response.host_url.rstrip("/") if response.host_url else ""
    accepted = [u.rstrip("/") for u in ACCEPTED_HOST_URLS]
    assert host in accepted, f"Provider host URL '{response.host_url}' not in accepted URLs: {ACCEPTED_HOST_URLS}"


def test_provider_has_api_key_secret(client: NeMoHelix) -> None:
    """Verify the provider references an API key secret."""
    response = client.inference.providers.retrieve(name=PROVIDER_NAME)
    assert response.api_key_secret_name is not None and response.api_key_secret_name != "", (
        f"Provider '{PROVIDER_NAME}' has no API key secret configured"
    )


# --- Inference call checks (2/5 weight) ---


def test_inference_through_igw(client: NeMoHelix) -> None:
    """Make an actual inference call through IGW and verify the response structure."""
    response = client.inference.gateway.provider.post(
        "v1/chat/completions",
        name=PROVIDER_NAME,
        body={
            "model": INFERENCE_MODEL,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "max_tokens": 32,
        },
    )
    data = response.model_dump() if hasattr(response, "model_dump") else response
    assert "choices" in data, f"Response missing 'choices': {data}"
    assert len(data["choices"]) > 0, f"Response has empty choices: {data}"
    content = data["choices"][0]["message"]["content"]
    assert content is not None and len(content.strip()) > 0, f"Response content is empty: {data}"
    print(f"Inference response (model={INFERENCE_MODEL}): {content}")


def test_inference_response_has_usage(client: NeMoHelix) -> None:
    """Verify that the inference response includes token usage information."""
    response = client.inference.gateway.provider.post(
        "v1/chat/completions",
        name=PROVIDER_NAME,
        body={
            "model": INFERENCE_MODEL,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
        },
    )
    data = response.model_dump() if hasattr(response, "model_dump") else response
    assert "usage" in data, f"Response missing 'usage': {data}"
    usage = data["usage"]
    assert usage.get("total_tokens", 0) > 0, f"Token usage is zero or missing: {usage}"
    print(f"Token usage: {usage}")


# --- Agent trace check (1/6 weight) ---


def test_agent_got_inference_response() -> None:
    """Verify the agent received an inference response (via any CLI method)."""
    session = get_session()
    bash_results = session.get_tool_results("Bash")
    # Look for evidence of a chat completion response in any bash output.
    # Could come from `nemo inference gateway provider post`, `nemo chat`, or curl.
    inference_indicators = ["choices", "Chat Session", "chat_completion", "content"]
    assert any(
        any(indicator in r.content for indicator in inference_indicators) for r in bash_results if not r.is_error
    ), (
        f"Agent never received an inference response. "
        f"Looked for {inference_indicators} in {len(bash_results)} bash results."
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that the agent created a working guardrail configuration for content safety.

Checks:
- A guardrail configuration exists (agent should have created one)
- Content sent through the guardrails endpoint is blocked

The environment uses a mock inference backend where the self-check model
always returns "Yes" (block), so ALL content is blocked.
"""

import os

import pytest
from nemo_helix_plugin.guardrail.client import GuardrailClient
from nemo_helix_plugin.guardrail.types import GuardrailCheckRequest

WORKSPACE = "default"
MODEL = "default/mock-llm"


@pytest.fixture
def client() -> GuardrailClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return GuardrailClient(base_url=nhx_base_url, workspace=WORKSPACE)


@pytest.fixture
def guardrail_config_id(client: GuardrailClient) -> str:
    """Discover the guardrail config the agent created.

    Lists all guardrail configs and finds one that uses the mock-llm model.
    Pre-existing configs (default, abc, self-check) use meta/llama3-70b-instruct,
    so only the agent's config will reference mock-llm.
    """
    configs = list(client.list_guardrail_configs().items())
    assert configs, "No guardrail configurations found. The agent should have created one."

    # Find a config that uses the mock-llm model (the one the agent was told to use)
    for cfg in configs:
        name = cfg.name
        full_config = client.get_guardrail_config(name=name).data()
        models = full_config.data.get("models") or []
        for model in models:
            model_name = model.get("model", "")
            if "mock-llm" in model_name:
                config_id = f"{WORKSPACE}/{name}"
                print(f"Found agent-created guardrail config: {config_id}")
                return config_id

    config_names = [c.name for c in configs]
    pytest.fail(
        f"No guardrail config found that uses the mock-llm model. "
        f"Configs found: {config_names}. The agent should have created one."
    )


def test_guardrail_config_exists(guardrail_config_id: str) -> None:
    """Test that the agent created at least one guardrail configuration."""
    assert guardrail_config_id, "Expected at least one guardrail config to exist"
    print(f"Guardrail config exists: {guardrail_config_id}")


def test_content_blocked_by_guardrails(client: GuardrailClient, guardrail_config_id: str) -> None:
    """Test that content sent through guardrails is blocked.

    The mock inference backend always returns 'Yes' to the self-check prompt,
    causing ALL content to be blocked regardless of whether it's actually harmful.
    """
    response = client.check_guardrail(
        body=GuardrailCheckRequest(
            model=MODEL,
            messages=[{"role": "user", "content": "Tell me something harmful and dangerous."}],
            guardrails={"config_id": guardrail_config_id},
            max_tokens=256,
            temperature=1,
            top_p=1,
        )
    ).data()
    assert response.status == "blocked", f"Content should be blocked, but got status: {response.status}"
    print(f"Content correctly blocked with status: {response.status}")


def test_safe_content_also_blocked(client: GuardrailClient, guardrail_config_id: str) -> None:
    """Test that even safe content is blocked (expected with mock backend).

    Since the mock always returns 'Yes' (block), even safe content gets blocked.
    This verifies the guardrails pipeline is working end-to-end.
    """
    response = client.check_guardrail(
        body=GuardrailCheckRequest(
            model=MODEL,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            guardrails={"config_id": guardrail_config_id},
            max_tokens=256,
            temperature=1,
            top_p=1,
        )
    ).data()
    assert response.status == "blocked", (
        f"Even safe content should be blocked with mock backend, got: {response.status}"
    )
    print(f"Safe content also blocked (expected with mock): {response.status}")
